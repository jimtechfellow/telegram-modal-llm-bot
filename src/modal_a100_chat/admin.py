"""Explicit deployment and smoke-test controls; never silently start paid compute."""

import argparse
import asyncio
import fcntl
import json
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from threading import Lock
from uuid import uuid4

import modal

from .backend import ModalBackend, validate_result
from .config import Config
from .constants import APP_NAME, MODEL_CACHE_KEY, MODEL_ID, MODEL_REVISION, VOLUME_NAME
from .weights import load_manifest


def local_check():
    from .inference import engine_options
    from .modal_app import SINGLE_USE

    root = Path(__file__).resolve().parents[2]
    manifest = load_manifest(root / "model-manifest.json")
    return {
        "status": "PASS",
        "kind": "local_configuration_only",
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "volume": VOLUME_NAME,
        "files": len(manifest["files"]),
        "engine_options": engine_options(),
        "container_settings": SINGLE_USE,
        "actual_gpu_load": "NOT_RUN",
    }


async def wait_idle(function, timeout=90):
    deadline = time.monotonic() + timeout
    while True:
        stats = await function.get_current_stats.aio()
        if stats.num_total_runners == 0 and stats.num_running_inputs == 0 and stats.backlog == 0:
            return {"num_total_runners": 0, "num_running_inputs": 0, "backlog": 0}
        if time.monotonic() >= deadline:
            raise TimeoutError("Container release has not been confirmed")
        await asyncio.sleep(2)


_ADMIN_LOCK = Lock()


@contextmanager
def _admin_lock(state_dir):
    # Non-blocking process-wide lock plus a file lock for separate CLI processes.
    if not _ADMIN_LOCK.acquire(blocking=False):
        raise RuntimeError("Another administrative operation is active")
    try:
        state_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        with (state_dir / "admin.lock").open("a") as lock:
            os.chmod(lock.name, 0o600)
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError("Another administrative operation is active") from None
            yield
    finally:
        _ADMIN_LOCK.release()


def _sync_state_dir(state_dir):
    descriptor = os.open(state_dir, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_new_record(path, record):
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(record, stream)
        stream.flush()
        os.fsync(stream.fileno())


def _reserve_admin_call(state_dir, function_name, environment):
    if function_name not in {"infer", "download_model"} or not environment:
        raise RuntimeError("Administrative function target is not confirmed")
    record = {
        "call_id": None, "app": APP_NAME, "function": function_name,
        "environment": environment, "operation_id": uuid4().hex,
    }
    try:
        _write_new_record(state_dir / "admin-call.json", record)
    except FileExistsError:
        raise RuntimeError("An unconfirmed administrative call already exists") from None
    _sync_state_dir(state_dir)
    return record


def _assert_admin_owner(state_dir, record):
    try:
        current = json.loads((state_dir / "admin-call.json").read_text())
    except (OSError, ValueError):
        raise RuntimeError("Administrative call record ownership is unconfirmed") from None
    if current != record:
        raise RuntimeError("Administrative call record belongs to another operation")


def _set_admin_call(state_dir, record, call_id):
    _assert_admin_owner(state_dir, record)
    updated = {**record, "call_id": call_id}
    temporary = state_dir / f"admin-call.{record['operation_id']}.tmp"
    try:
        _write_new_record(temporary, updated)
        _assert_admin_owner(state_dir, record)
        os.replace(temporary, state_dir / "admin-call.json")
        _sync_state_dir(state_dir)
    finally:
        temporary.unlink(missing_ok=True)
    return updated


def _clear_admin_call(state_dir, record):
    _assert_admin_owner(state_dir, record)
    (state_dir / "admin-call.json").unlink()
    _sync_state_dir(state_dir)


async def collect(
    call, state_dir, max_seconds=4200, *, function, function_name, environment, record=None
):
    if record is None:
        with _admin_lock(state_dir):
            record = _reserve_admin_call(state_dir, function_name, environment)
            return await collect(
                call, state_dir, max_seconds, function=function, function_name=function_name,
                environment=environment, record=record,
            )
    if record["function"] != function_name or record["environment"] != environment:
        raise RuntimeError("Administrative function target differs from durable record")
    received = False
    deadline = time.monotonic() + max_seconds
    try:
        record = _set_admin_call(state_dir, record, call.object_id)
        print(json.dumps({"call_id": call.object_id, "status": "running"}), flush=True)
        # The get coroutine is not cancellation-safe. Each bounded get completes before the next.
        from modal.exception import TimeoutError as ModalTimeoutError

        while True:
            try:
                result = await call.get.aio(timeout=15)
                received = True
                break
            except (ModalTimeoutError, TimeoutError):
                if time.monotonic() >= deadline:
                    raise TimeoutError("Administrative call exceeded its bounded wait") from None
        release = await wait_idle(function)
        _clear_admin_call(state_dir, record)
        return result, release
    except BaseException:
        if not received:
            try:
                await call.cancel.aio(terminate_containers=True)
            except BaseException:
                pass
            # Cancellation acknowledgement or failure is not release confirmation.
            try:
                await wait_idle(function)
                _clear_admin_call(state_dir, record)
            except BaseException:
                pass
        raise


async def execute(args, config):
    with _admin_lock(config.state_dir):
        if args.command in {"download", "smoke"} and (config.state_dir / "admin-call.json").exists():
            raise RuntimeError("An unconfirmed administrative call already exists")
        return await _execute(args, config)


async def _execute(args, config):
    backend = ModalBackend(config)
    await backend.check_workspace()
    environment = config.modal_environment
    if args.command == "deploy":
        from .deployment_verification import verify_remote_settings
        from .modal_app import HF_SECRET_NAME, app, cpu_image, gpu_image

        token = os.environ.get("HF_TOKEN")
        if not token:
            raise RuntimeError("HF_TOKEN must be injected into the process environment")
        await modal.Volume.objects.create.aio(
            VOLUME_NAME,
            version=1,
            allow_existing=True,
            environment_name=environment,
        )
        volume = modal.Volume.from_name(VOLUME_NAME, environment_name=environment)
        await volume.hydrate.aio()
        if (await volume.info.aio()).name != VOLUME_NAME:
            raise RuntimeError("Volume readback mismatch")
        # Update only this application's named secret, never any existing service credential.
        secret = modal.Secret.from_name(HF_SECRET_NAME, environment_name=environment)
        try:
            await secret.hydrate.aio()
        except modal.exception.NotFoundError:
            await modal.Secret.objects.create.aio(
                HF_SECRET_NAME,
                {"HF_TOKEN": token},
                environment_name=environment,
            )
        else:
            await secret.update.aio({"HF_TOKEN": token})
        if (await secret.info.aio()).name != HF_SECRET_NAME:
            raise RuntimeError("Secret metadata readback mismatch")
        await app.deploy.aio(environment_name=environment)
        deployed = await modal.App.lookup.aio(APP_NAME, environment_name=environment)
        info = await deployed.info.aio(refresh=True)
        if set(info.functions) != {"infer", "download_model"} or info.servers:
            raise RuntimeError("Deployed function readback mismatch")
        controls = await verify_remote_settings(
            info.functions,
            {"infer": gpu_image.object_id, "download_model": cpu_image.object_id},
            volume.object_id,
            secret.object_id,
        )
        return {
            "status": "PASS",
            "kind": "deployed_resource_configuration_only",
            "controls": controls,
            "app_id": info.app_id,
            "functions": info.functions,
            "volume_id": volume.object_id,
            "volume_name": VOLUME_NAME,
            "gpu_load": "NOT_RUN",
        }
    if args.command == "download":
        function = modal.Function.from_name(APP_NAME, "download_model", environment_name=environment)
        # Reserve before spawn: interruption without a returned id must fail closed.
        record = _reserve_admin_call(config.state_dir, "download_model", environment)
        report, release = await collect(
            await function.spawn.aio(), config.state_dir, function=function,
            function_name="download_model", environment=environment, record=record,
        )
        volume = modal.Volume.from_name(VOLUME_NAME, environment_name=environment)
        chunks = []
        async for chunk in volume.read_file.aio(MODEL_CACHE_KEY + "/verified.json"):
            chunks.append(chunk)
            if sum(map(len, chunks)) > 1024 * 1024:
                raise RuntimeError("Unexpectedly large verification report")
        stored = json.loads(b"".join(chunks))
        if stored != report or stored["model_id"] != MODEL_ID or not stored["full_hash_verified"]:
            raise RuntimeError("Volume verification report readback mismatch")
        report["container_release"] = release
        return report
    if args.command == "smoke":
        record = _reserve_admin_call(config.state_dir, "infer", environment)
        result, release = await collect(
            await backend.spawn(
                [
                    {"role": "system", "content": "请用简体中文简短回答。"},
                    {"role": "user", "content": "你好，请用中文介绍你可以如何帮助我。"},
                ]
            ),
            config.state_dir,
            function=backend.function, function_name="infer", environment=environment, record=record,
        )
        validate_result(result)
        result["container_release"] = release
        return result
    if args.command == "status":
        return await backend.stats()
    if args.command == "cancel-admin":
        path = config.state_dir / "admin-call.json"
        record = json.loads(path.read_text())
        if (
            record.get("app") != APP_NAME or record.get("environment") != environment
            or record.get("function") not in {"infer", "download_model"}
            or not record.get("call_id") or not record.get("operation_id")
        ):
            raise RuntimeError(
                "Administrative call target or identifier is unconfirmed; scoped stop required"
            )
        function = modal.Function.from_name(
            APP_NAME, record["function"], environment_name=record["environment"]
        )
        call = modal.FunctionCall.from_id(record["call_id"])
        try:
            await call.cancel.aio(terminate_containers=True)
        finally:
            release = await wait_idle(function)
            _clear_admin_call(config.state_dir, record)
        return {"status": "cancelled", "call_id": call.object_id, "container_release": release}
    if args.command == "stop-app":
        app = await modal.App.lookup.aio(APP_NAME, environment_name=environment)
        info = await app.info.aio(refresh=True)
        result = subprocess.run(
            [sys.executable, "-m", "modal", "app", "stop", "--env", environment, "--yes", info.app_id],
            capture_output=True,
            text=True,
            timeout=90,
        )
        if result.returncode:
            raise RuntimeError("Scoped Modal app stop failed; raw output suppressed")
        after = await app.info.aio(refresh=True)
        # Return actual lifecycle fields; the operator must verify stopped before declaring rollback complete.
        from dataclasses import asdict

        return {"app_id": after.app_id, "lifecycle": asdict(after.lifecycle), "volume_preserved": VOLUME_NAME}
    raise ValueError("Unknown operation")


def main():
    import logging

    logging.disable(logging.CRITICAL)
    os.umask(0o077)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=["local-check", "deploy", "download", "smoke", "status", "cancel-admin", "stop-app"],
    )
    parser.add_argument("--config", type=Path, default=Path.home() / ".config/modal_a100_chat/config.toml")
    parser.add_argument("--execute", action="store_true", help="Explicitly authorize this remote mutation")
    args = parser.parse_args()
    if args.command == "local-check":
        print(json.dumps(local_check(), ensure_ascii=False, indent=2))
        return
    if args.command != "status" and not args.execute:
        parser.error("Remote mutations require --execute after explicit user authorization")
    try:
        result = asyncio.run(execute(args, Config.load(args.config)))
        print(json.dumps(result, ensure_ascii=False, default=str, indent=2))
    except (KeyboardInterrupt, asyncio.CancelledError) as exc:
        print(json.dumps({"status": "INTERRUPTED", "error_type": type(exc).__name__}))
        raise SystemExit(130) from None
    except Exception as exc:
        # Do not render credential-bearing HTTP exceptions or URLs.
        print(json.dumps({"status": "BLOCKED", "error_type": type(exc).__name__}))
        sys.exit(1)


if __name__ == "__main__":
    main()
