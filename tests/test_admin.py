"""Local-only administrative call doubles; never contact Modal or claim GPU acceptance."""

import asyncio
import json
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from modal_a100_chat import admin


@pytest.mark.parametrize("cancel_fails", [False, True])
@pytest.mark.parametrize("idle_fails", [False, True])
async def test_admin_failed_wait_retains_id_until_idle_confirmed(
    tmp_path, monkeypatch, cancel_fails, idle_fails
):
    async def get(**kwargs):
        raise RuntimeError("local-only remote failure")

    async def cancel(**kwargs):
        assert kwargs["terminate_containers"] is True
        if cancel_fails:
            raise RuntimeError("local-only cancellation failure")

    function = object()
    checks = []

    async def idle(target):
        assert target is function
        checks.append(target)
        if idle_fails:
            raise TimeoutError("local-only idle failure")
        return {"num_total_runners": 0, "num_running_inputs": 0, "backlog": 0}

    monkeypatch.setattr(admin, "wait_idle", idle)
    call = SimpleNamespace(
        object_id="fixture-call-id", get=SimpleNamespace(aio=get), cancel=SimpleNamespace(aio=cancel)
    )
    with pytest.raises(RuntimeError):
        await admin.collect(
            call, tmp_path, function=function, function_name="infer", environment="local-only"
        )
    path = tmp_path / "admin-call.json"
    assert path.exists() is idle_fails
    assert checks == [function]
    if idle_fails:
        assert json.loads(path.read_text())["call_id"] == call.object_id


async def test_admin_bounded_wait_cancels_and_confirms_idle(tmp_path, monkeypatch):
    async def get(**kwargs):
        raise TimeoutError("local-only timed get")

    cancelled, checks = [], []
    function = object()

    async def cancel(**kwargs):
        cancelled.append(kwargs)

    async def idle(target):
        assert target is function
        checks.append(target)
        return {"num_total_runners": 0, "num_running_inputs": 0, "backlog": 0}

    monkeypatch.setattr(admin, "wait_idle", idle)
    call = SimpleNamespace(
        object_id="fixture-call-id", get=SimpleNamespace(aio=get), cancel=SimpleNamespace(aio=cancel)
    )
    with pytest.raises(TimeoutError, match="bounded wait"):
        await admin.collect(
            call, tmp_path, max_seconds=0, function=function, function_name="infer", environment="local-only"
        )
    assert cancelled == [{"terminate_containers": True}] and checks == [function]
    assert not (tmp_path / "admin-call.json").exists()


@pytest.mark.parametrize("busy_field", ["num_total_runners", "num_running_inputs", "backlog"])
async def test_admin_idle_requires_every_target_count_zero(monkeypatch, busy_field):
    zero = {"num_total_runners": 0, "num_running_inputs": 0, "backlog": 0}
    responses = [SimpleNamespace(**{**zero, busy_field: 1}), SimpleNamespace(**zero)]
    checks, sleeps = [], []

    async def stats():
        checks.append(True)
        return responses.pop(0)

    async def sleep(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr(admin.asyncio, "sleep", sleep)
    function = SimpleNamespace(get_current_stats=SimpleNamespace(aio=stats))
    assert await admin.wait_idle(function) == zero
    assert len(checks) == 2 and sleeps == [2]


async def test_admin_idle_deadline_remains_bounded(monkeypatch):
    ticks = iter([0, 90])
    monkeypatch.setattr(admin, "time", SimpleNamespace(monotonic=lambda: next(ticks)))

    async def stats():
        return SimpleNamespace(num_total_runners=1, num_running_inputs=0, backlog=0)

    function = SimpleNamespace(get_current_stats=SimpleNamespace(aio=stats))
    with pytest.raises(TimeoutError, match="release"):
        await admin.wait_idle(function, timeout=90)


def test_admin_lock_is_exclusive_across_local_processes_and_reusable(tmp_path, repository):
    script = '''
import sys
from pathlib import Path
from modal_a100_chat.admin import _admin_lock
try:
    with _admin_lock(Path(sys.argv[1])):
        raise AssertionError("local-only lock unexpectedly acquired")
except RuntimeError as exc:
    print(type(exc).__name__)
'''
    env = {"PATH": os.defpath, "HOME": str(tmp_path), "PYTHONPATH": str(repository / "src")}
    with admin._admin_lock(tmp_path):
        result = subprocess.run(
            [sys.executable, "-c", script, str(tmp_path)],
            env=env, capture_output=True, text=True, timeout=15,
        )
    assert result.returncode == 0 and result.stdout.strip() == "RuntimeError" and result.stderr == ""
    with admin._admin_lock(tmp_path):
        pass


@pytest.mark.parametrize("exception_type", [KeyboardInterrupt, asyncio.CancelledError])
def test_admin_process_interruption_is_sanitized(config, monkeypatch, capsys, exception_type):
    async def interrupted(args, config):
        raise exception_type("local-only-credential-message") from RuntimeError("local-only-secret-chain")

    monkeypatch.setattr(admin, "execute", interrupted)
    monkeypatch.setattr(admin.Config, "load", lambda path: config)
    monkeypatch.setattr(sys, "argv", ["local-only-admin", "smoke", "--execute"])
    with pytest.raises(SystemExit) as exc:
        admin.main()
    output = capsys.readouterr()
    assert exc.value.code == 130
    assert json.loads(output.out) == {"status": "INTERRUPTED", "error_type": exception_type.__name__}
    assert output.err == ""
