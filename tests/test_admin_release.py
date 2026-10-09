"""All administrative mocks are local-only; no credentials or APIs are used."""

import asyncio
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from modal_a100_chat import admin
from modal_a100_chat.constants import APP_NAME, MODEL_DIR, MODEL_ID, MODEL_REVISION


def local_result():
    return {
        "text": "local-only",
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "loaded_model_path": MODEL_DIR,
        "loaded_dtype": "torch.bfloat16",
        "quantization": None,
        "gpu_name": "local-only A100",
        "gpu_memory_bytes": 80 * 1024**3,
    }


@pytest.fixture
def local_admin(monkeypatch):
    state = SimpleNamespace(
        spawns=0,
        cancels=0,
        idle_checks=[],
        fail_get=False,
        fail_cancel=False,
        fail_idle=False,
        interrupt_get=False,
        interrupt_spawn=False,
        interrupt_cancel=False,
        get_entered=asyncio.Event(),
        get_release=asyncio.Event(),
        idle_hook=None,
        lookups=[],
    )
    state.get_release.set()

    async def scaling(**kwargs):
        return SimpleNamespace(**kwargs)

    async def get(**kwargs):
        assert kwargs["timeout"] == 15
        state.get_entered.set()
        await state.get_release.wait()
        if state.interrupt_get:
            raise asyncio.CancelledError("local-only interruption")
        if state.fail_get:
            raise RuntimeError("local-only get failure")
        return local_result()

    async def cancel(**kwargs):
        assert kwargs == {"terminate_containers": True}
        state.cancels += 1
        if state.interrupt_cancel:
            raise asyncio.CancelledError("local-only cancellation interruption")
        if state.fail_cancel:
            raise RuntimeError("local-only cancel failure")

    state.call = SimpleNamespace(
        object_id="local-only-admin-call", get=SimpleNamespace(aio=get), cancel=SimpleNamespace(aio=cancel)
    )

    async def spawn(*args):
        state.spawns += 1
        if state.interrupt_spawn:
            raise asyncio.CancelledError("local-only spawn interruption")
        return state.call

    state.functions = {
        name: SimpleNamespace(
            name=name,
            spawn=SimpleNamespace(aio=spawn),
            update_autoscaler=SimpleNamespace(aio=scaling),
        )
        for name in ("infer", "download_model")
    }

    async def idle(function, timeout=90):
        assert function is state.functions[function.name]
        state.idle_checks.append(function.name)
        if state.idle_hook:
            state.idle_hook()
        if state.fail_idle:
            raise TimeoutError("local-only release failure")
        return {"num_total_runners": 0, "num_running_inputs": 0, "backlog": 0}

    class LocalBackend:
        def __init__(self, config):
            self.function = state.functions["infer"]

        async def check_workspace(self):
            pass

        async def check_inference(self):
            pass

        async def configure(self):
            pass

        async def stats(self):
            return {"num_total_runners": 0, "num_running_inputs": 0, "backlog": 0}

        async def spawn(self, messages):
            return await spawn(messages)

        async def wait_idle(self):
            return await idle(self.function)

    def from_name(app, name, *, environment_name):
        assert app == APP_NAME
        state.lookups.append((name, environment_name))
        return state.functions[name]

    def from_id(identifier):
        assert identifier == state.call.object_id
        return state.call

    def no_volume(*args, **kwargs):
        raise AssertionError("Volume requests forbidden in local-only tests")

    monkeypatch.setattr(admin, "ModalBackend", LocalBackend)
    monkeypatch.setattr(admin, "wait_idle", idle)
    monkeypatch.setattr(admin.modal.Function, "from_name", from_name)
    monkeypatch.setattr(admin.modal.FunctionCall, "from_id", from_id)
    monkeypatch.setattr(admin.modal.Volume, "from_name", no_volume)
    return state


def args(command):
    return SimpleNamespace(command=command)


def seed_record(config, **overrides):
    record = {
        "call_id": "local-only-admin-call",
        "app": APP_NAME,
        "function": "download_model",
        "environment": config.modal_environment,
        "operation_id": "local-only-record-owner",
    }
    record.update(overrides)
    config.state_dir.mkdir(parents=True, exist_ok=True)
    path = config.state_dir / "admin-call.json"
    path.write_text(json.dumps(record))
    return path, record


@pytest.mark.parametrize("fail_get,fail_cancel", [(False, False), (True, False), (True, True)])
async def test_admin_unconfirmed_release_retains_exact_function_target(
    config, local_admin, fail_get, fail_cancel
):
    local_admin.fail_idle, local_admin.fail_get, local_admin.fail_cancel = True, fail_get, fail_cancel
    with pytest.raises((RuntimeError, TimeoutError)):
        await admin.execute(args("smoke"), config)
    path = config.state_dir / "admin-call.json"
    assert path.exists()
    record = json.loads(path.read_text())
    assert record["call_id"] == local_admin.call.object_id
    assert record["function"] == "infer" and record["app"] == APP_NAME
    assert record["environment"] == config.modal_environment
    assert local_admin.idle_checks == ["infer"]
    assert local_admin.cancels == int(fail_get)


@pytest.mark.parametrize("fail_get,fail_cancel", [(False, False), (True, False), (True, True)])
async def test_admin_verified_release_clears_only_owned_record(config, local_admin, fail_get, fail_cancel):
    local_admin.fail_get, local_admin.fail_cancel = fail_get, fail_cancel
    if fail_get:
        with pytest.raises(RuntimeError, match="get failure"):
            await admin.execute(args("smoke"), config)
    else:
        result = await admin.execute(args("smoke"), config)
        assert result["container_release"]["num_total_runners"] == 0
    assert not (config.state_dir / "admin-call.json").exists()
    assert local_admin.idle_checks == ["infer"]


@pytest.mark.parametrize("fail_cancel", [False, True])
async def test_cancel_admin_release_failure_keeps_record_and_confirms_exact_target(
    config, local_admin, fail_cancel
):
    path, record = seed_record(config)
    local_admin.fail_idle, local_admin.fail_cancel = True, fail_cancel
    with pytest.raises((RuntimeError, TimeoutError)):
        await admin.execute(args("cancel-admin"), config)
    assert json.loads(path.read_text()) == record
    assert local_admin.idle_checks == ["download_model"]
    assert local_admin.lookups == [("download_model", config.modal_environment)]


@pytest.mark.parametrize("fail_cancel", [False, True])
async def test_cancel_admin_clears_only_after_exact_idle_confirmation(config, local_admin, fail_cancel):
    path, _ = seed_record(config)
    local_admin.fail_cancel = fail_cancel
    if fail_cancel:
        with pytest.raises(RuntimeError, match="cancel failure"):
            await admin.execute(args("cancel-admin"), config)
    else:
        result = await admin.execute(args("cancel-admin"), config)
        assert result["container_release"]["num_total_runners"] == 0
    assert not path.exists()
    assert local_admin.idle_checks == ["download_model"]


async def test_cancel_admin_unknown_target_fails_closed(config, local_admin):
    path, record = seed_record(config)
    del record["function"]
    path.write_text(json.dumps(record))
    with pytest.raises(RuntimeError):
        await admin.execute(args("cancel-admin"), config)
    assert json.loads(path.read_text()) == record
    assert local_admin.cancels == 0 and local_admin.idle_checks == []


@pytest.mark.parametrize("command", ["smoke", "download"])
async def test_existing_admin_record_blocks_before_any_spawn(config, local_admin, command):
    path, record = seed_record(config)
    with pytest.raises(RuntimeError):
        await admin.execute(args(command), config)
    assert local_admin.spawns == 0
    assert json.loads(path.read_text()) == record


@pytest.mark.parametrize("command", ["smoke", "cancel-admin", "deploy", "download", "stop-app"])
async def test_running_bot_blocks_mutating_admin_commands(config, local_admin, command):
    from modal_a100_chat.bot import exclusive_bot_lock

    with exclusive_bot_lock(config.state_dir), pytest.raises(RuntimeError, match="Bot or"):
        await admin.execute(args(command), config)
    assert local_admin.spawns == local_admin.cancels == 0


async def test_pending_bot_call_blocks_admin_smoke_before_spawn(config, local_admin):
    from modal_a100_chat.store import Store

    store = Store(config.state_dir)
    try:
        store.start(1, config.allowed_chat_id, "fixture")
        store.set_call(1, "fixture-bot-call")
        with pytest.raises(RuntimeError, match="durable bot calls"):
            await admin.execute(args("smoke"), config)
        assert local_admin.spawns == local_admin.cancels == 0
    finally:
        store.db.close()


@pytest.mark.parametrize("command,other_state", [("smoke", False), ("cancel-admin", False), ("smoke", True)])
async def test_process_wide_admin_exclusivity_prevents_concurrent_record_mutation(
    config, local_admin, command, other_state
):
    local_admin.get_release.clear()
    first = asyncio.create_task(admin.execute(args("smoke"), config))
    await local_admin.get_entered.wait()
    path = config.state_dir / "admin-call.json"
    original = json.loads(path.read_text())
    second_config = replace(config, state_dir=config.state_dir / "other") if other_state else config
    try:
        with pytest.raises(RuntimeError):
            await asyncio.wait_for(admin.execute(args(command), second_config), timeout=0.1)
        assert local_admin.spawns == 1 and local_admin.cancels == 0
        assert json.loads(path.read_text()) == original
    finally:
        local_admin.get_release.set()
        await first
    assert not path.exists()


@pytest.mark.parametrize("command", ["smoke", "cancel-admin"])
async def test_admin_deletion_checks_record_ownership(config, local_admin, command):
    if command == "cancel-admin":
        seed_record(config)
    path = config.state_dir / "admin-call.json"
    other = {"call_id": "another-local-only-call", "operation_id": "another-local-only-owner"}
    local_admin.idle_hook = lambda: path.write_text(json.dumps(other))
    with pytest.raises(RuntimeError):
        await admin.execute(args(command), config)
    assert json.loads(path.read_text()) == other


async def test_admin_interruption_retains_handle_until_release_confirmed(config, local_admin):
    local_admin.interrupt_get, local_admin.fail_idle = True, True
    with pytest.raises(asyncio.CancelledError):
        await admin.execute(args("smoke"), config)
    record = json.loads((config.state_dir / "admin-call.json").read_text())
    assert record["call_id"] == local_admin.call.object_id and record["function"] == "infer"
    assert local_admin.idle_checks == ["infer"]


async def test_admin_interrupted_spawn_leaves_durable_reservation_and_blocks_replay(config, local_admin):
    local_admin.interrupt_spawn = True
    with pytest.raises(asyncio.CancelledError):
        await admin.execute(args("smoke"), config)
    path = config.state_dir / "admin-call.json"
    record = json.loads(path.read_text())
    assert record["call_id"] is None and record["function"] == "infer"
    with pytest.raises(RuntimeError):
        await admin.execute(args("smoke"), config)
    assert json.loads(path.read_text()) == record and local_admin.spawns == 1


@pytest.mark.parametrize("idle_fails", [False, True])
async def test_cancel_admin_interruption_still_requires_exact_idle_confirmation(
    config, local_admin, idle_fails
):
    path, record = seed_record(config)
    local_admin.interrupt_cancel, local_admin.fail_idle = True, idle_fails
    with pytest.raises((asyncio.CancelledError, TimeoutError)):
        await admin.execute(args("cancel-admin"), config)
    assert local_admin.idle_checks == ["download_model"]
    assert path.exists() is idle_fails
    if idle_fails:
        assert json.loads(path.read_text()) == record
