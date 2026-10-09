"""Deterministic unit double; these tests never contact Modal or prove GPU acceptance."""

import asyncio
from types import SimpleNamespace

import pytest

from modal_a100_chat.backend import validate_result
from modal_a100_chat.constants import MODEL_DIR, MODEL_ID, MODEL_REVISION
from modal_a100_chat.runner import Runner


def fixture_result():
    return {
        "text": "单元测试回复",
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "loaded_model_path": MODEL_DIR,
        "loaded_dtype": "torch.bfloat16",
        "quantization": None,
        "gpu_name": "unit-fixture A100",
        "gpu_memory_bytes": 80 * 1024**3,
    }


class BackendDouble:
    def __init__(self):
        self.spawn_entered = asyncio.Event()
        self.spawn_release = asyncio.Event()
        self.spawn_release.set()
        self.poll_entered = asyncio.Event()
        self.poll_release = asyncio.Event()
        self.poll_release.set()
        self.cancels = []
        self.spawns = 0
        self.fail_spawn = False
        self.fail_cancel = False
        self.busy = False
        self.backlog = 0
        self.warm = False
        self.windows = []
        self.release_entered = asyncio.Event()
        self.release_resume = asyncio.Event()
        self.release_resume.set()
        self.fail_idle = False

    async def configure(self):
        self.windows.append(300)

    async def release(self):
        self.windows.append(2)
        self.release_entered.set()
        await self.release_resume.wait()
        result = await self.wait_idle()
        self.warm = False
        return result

    async def check_workspace(self):
        return None

    async def check_inference(self):
        return None

    async def spawn(self, messages):
        self.spawns += 1
        self.spawn_entered.set()
        await self.spawn_release.wait()
        if self.fail_spawn:
            raise RuntimeError("unit failure")
        return SimpleNamespace(object_id="fixture-call-id")

    def from_id(self, identifier):
        return SimpleNamespace(object_id=identifier)

    async def poll(self, call):
        self.poll_entered.set()
        await self.poll_release.wait()
        return fixture_result()

    async def cancel(self, call):
        if self.fail_cancel:
            raise RuntimeError("unit failure")
        self.cancels.append(call.object_id)
        self.poll_release.set()

    async def wait_idle(self):
        if self.fail_idle:
            raise TimeoutError("fixture release unconfirmed")
        return {"num_total_runners": 0, "num_running_inputs": 0, "backlog": 0}

    async def stats(self):
        return {
            "num_total_runners": int(self.warm or self.busy),
            "num_running_inputs": int(self.busy),
            "backlog": self.backlog,
        }


async def test_second_turn_uses_warm_gpu_without_waiting_for_release(config, store, backend):
    runner = Runner(config, store, backend)
    await runner.recover()
    backend.warm = True

    async def reply(text):
        pass

    for update in (1, 2):
        task = await runner.start(update, config.allowed_chat_id, "question", reply)
        assert task is not None
        await task
    assert backend.spawns == 2 and runner.ready
    assert len(store.history(config.allowed_chat_id)) == 4
    assert not store.pending()
    assert store.last_job()["release_confirmed"] == 0
    assert 2 not in backend.windows


async def test_idle_stop_blocks_new_start_until_actual_release(config, store, backend):
    runner = Runner(config, store, backend)
    await runner.recover()

    async def reply(text):
        pass

    await (await runner.start(1, config.allowed_chat_id, "question", reply))
    backend.warm = True
    backend.release_resume.clear()
    stopping = asyncio.create_task(runner.stop())
    try:
        await asyncio.wait_for(backend.release_entered.wait(), timeout=2)
        starting = asyncio.create_task(runner.start(2, config.allowed_chat_id, "next", reply))
        await asyncio.sleep(0)
        assert not runner.ready and store.last_job()["release_confirmed"] == 0
        assert backend.spawns == 1 and not backend.cancels and not starting.done()
    finally:
        backend.release_resume.set()
        response = await asyncio.wait_for(stopping, timeout=2)
    assert "关闭已确认" in response
    first_job = store.db.execute("SELECT * FROM jobs WHERE update_id=1").fetchone()
    assert first_job["release_confirmed"] == 1 and runner.ready
    await asyncio.wait_for(await starting, timeout=2)
    assert backend.windows[-1] == 300


async def test_idle_stop_failure_is_durable_and_recoverable(config, store, backend):
    runner = Runner(config, store, backend)
    await runner.recover()

    async def reply(text):
        pass

    await (await runner.start(1, config.allowed_chat_id, "question", reply))
    backend.warm, backend.fail_idle = True, True
    response = await runner.stop()
    assert "未确认" in response and not runner.ready
    assert store.pending()[0]["status"] == "completed"
    restored = Runner(config, store, backend)
    with pytest.raises(TimeoutError):
        await restored.recover()
    assert not restored.ready and not backend.cancels
    backend.fail_idle = False
    await restored.recover()
    assert restored.ready and not store.pending()
    assert len(store.history(config.allowed_chat_id)) == 2


async def test_unknown_remote_task_is_never_cancelled_by_idle_stop(config, store, backend):
    runner = Runner(config, store, backend)
    await runner.recover()
    backend.busy = True
    assert "未确认调用" in await runner.stop()
    assert not backend.cancels and 2 not in backend.windows


async def test_unknown_backlog_blocks_both_submission_and_stop(config, store, backend):
    runner = Runner(config, store, backend)
    await runner.recover()
    backend.backlog = 1

    async def reply(text):
        pass

    assert await runner.start(1, config.allowed_chat_id, "question", reply) is None
    assert "未确认调用" in await runner.stop()
    assert backend.spawns == 0 and not backend.cancels and 2 not in backend.windows


async def test_stop_cancellation_cannot_overlap_new_generation(config, store, backend):
    runner = Runner(config, store, backend)
    await runner.recover()
    backend.poll_release.clear()
    cancel_entered, cancel_resume = asyncio.Event(), asyncio.Event()
    original_cancel = backend.cancel

    async def cancel(call):
        cancel_entered.set()
        await cancel_resume.wait()
        await original_cancel(call)

    backend.cancel = cancel

    async def reply(text):
        pass

    first = await runner.start(1, config.allowed_chat_id, "question", reply)
    await asyncio.wait_for(backend.poll_entered.wait(), timeout=2)
    stopping = asyncio.create_task(runner.stop())
    try:
        await asyncio.wait_for(cancel_entered.wait(), timeout=2)
        starting = asyncio.create_task(runner.start(2, config.allowed_chat_id, "next", reply))
        await asyncio.sleep(0)
        assert not starting.done() and backend.spawns == 1 and not runner.ready
    finally:
        cancel_resume.set()
        backend.poll_release.set()
        await asyncio.wait_for(stopping, timeout=2)
        await asyncio.wait_for(first, timeout=2)
    second = await asyncio.wait_for(starting, timeout=2)
    if second is not None:
        await asyncio.wait_for(second, timeout=2)
    assert backend.cancels == ["fixture-call-id"]
    assert store.db.execute("SELECT status FROM jobs WHERE update_id=1").fetchone()[0] == "cancelled"


async def test_repeated_stop_cancels_exact_call_only_once(config, store, backend):
    runner = Runner(config, store, backend)
    await runner.recover()
    backend.poll_release.clear()
    backend.release_resume.clear()

    async def reply(text):
        pass

    task = await runner.start(1, config.allowed_chat_id, "question", reply)
    await asyncio.wait_for(backend.poll_entered.wait(), timeout=2)
    try:
        await runner.stop()
        await asyncio.wait_for(backend.release_entered.wait(), timeout=2)
        await runner.stop()
        assert backend.cancels == ["fixture-call-id"] and not runner.ready
    finally:
        backend.release_resume.set()
        await asyncio.wait_for(task, timeout=2)
    assert runner.ready and not store.pending()


async def test_request_timeout_cancels_and_releases_without_history(config, store, backend, monkeypatch):
    runner = Runner(config, store, backend)
    await runner.recover()
    ticks = iter([0, 1201])
    monkeypatch.setattr("modal_a100_chat.runner.time", SimpleNamespace(monotonic=lambda: next(ticks)))

    async def poll(call):
        return None

    async def reply(text):
        pass

    backend.poll = poll
    await (await runner.start(1, config.allowed_chat_id, "question", reply))
    assert store.last_job()["status"] == "cancelled" and store.last_job()["release_confirmed"] == 1
    assert not store.history(config.allowed_chat_id) and backend.cancels == ["fixture-call-id"]
    assert backend.windows[-1] == 2 and runner.ready


async def test_unresolved_admin_record_blocks_bot_recovery(config, store, backend):
    (config.state_dir / "admin-call.json").write_text("{}")
    runner = Runner(config, store, backend)
    await runner.recover()
    assert not runner.ready and not backend.windows and not backend.cancels


@pytest.mark.parametrize(
    "record_text", ['{"call_id": null}', '{"call_id": "unconfirmed-admin-call"}', "invalid JSON"]
)
async def test_stop_cannot_bypass_unresolved_admin_record(config, store, backend, record_text):
    record_path = config.state_dir / "admin-call.json"
    record_path.write_text(record_text)
    runner = Runner(config, store, backend)
    await runner.recover()
    replies = []

    async def reply(text):
        replies.append(text)

    for update in (1, 2):
        assert "BLOCKED" in await runner.stop()
        assert not runner.ready
        assert await runner.start(update, config.allowed_chat_id, "question", reply) is None
        assert "BLOCKED" in replies[-1]
    assert record_path.read_text() == record_text
    assert backend.spawns == 0 and not backend.cancels and not backend.windows
    assert store.last_job() is None

    # Only the administrative recovery flow may resolve this record.
    record_path.unlink()
    await runner.recover()
    assert runner.ready
    await (await runner.start(3, config.allowed_chat_id, "question", reply))
    assert backend.spawns == 1
    assert "关闭已确认" in await runner.stop()
    assert runner.ready and store.last_job()["release_confirmed"] == 1


async def test_admin_record_blocks_submission_even_after_successful_recovery(config, store, backend):
    runner = Runner(config, store, backend)
    await runner.recover()
    assert runner.ready
    record_path = config.state_dir / "admin-call.json"
    record_path.write_text('{"call_id": null}')
    windows_before = backend.windows.copy()
    replies = []

    async def reply(text):
        replies.append(text)

    task = await runner.start(1, config.allowed_chat_id, "question", reply)
    if task is not None:
        await task
    assert task is None and not runner.ready
    assert "BLOCKED" in replies[-1]
    assert record_path.exists() and store.last_job() is None
    assert backend.spawns == 0 and not backend.cancels and backend.windows == windows_before


async def test_admin_record_does_not_prevent_stopping_known_active_call(config, store, backend):
    runner = Runner(config, store, backend)
    await runner.recover()
    backend.poll_release.clear()
    replies = []

    async def reply(text):
        replies.append(text)

    task = await runner.start(1, config.allowed_chat_id, "question", reply)
    await asyncio.wait_for(backend.poll_entered.wait(), timeout=2)
    record_path = config.state_dir / "admin-call.json"
    record_path.write_text('{"call_id": null}')
    await runner.stop()
    await asyncio.wait_for(task, timeout=2)
    assert backend.cancels == ["fixture-call-id"] and backend.windows[-1] == 2
    assert store.last_job()["status"] == "cancelled" and store.last_job()["release_confirmed"] == 1
    assert record_path.read_text() == '{"call_id": null}'

    next_task = await runner.start(2, config.allowed_chat_id, "next", reply)
    if next_task is not None:
        await next_task
    assert next_task is None and not runner.ready and backend.spawns == 1
    assert "BLOCKED" in replies[-1]


async def test_shutdown_reports_unconfirmed_idle_release(config, store, backend):
    runner = Runner(config, store, backend)
    await runner.recover()

    async def reply(text):
        pass

    await (await runner.start(1, config.allowed_chat_id, "question", reply))
    backend.fail_idle = True
    with pytest.raises(TimeoutError, match="Shutdown release"):
        await runner.close()
    assert store.pending()[0]["status"] == "completed"


async def test_cancel_confirmation_delivery_failure_does_not_reopen_released_job(config, store, backend):
    runner = Runner(config, store, backend)
    await runner.recover()
    backend.poll_release.clear()

    async def reply(text):
        raise RuntimeError("fixture Telegram delivery failure")

    task = await runner.start(1, config.allowed_chat_id, "question", reply)
    await asyncio.wait_for(backend.poll_entered.wait(), timeout=2)
    await runner.stop()
    await asyncio.wait_for(task, timeout=2)
    assert store.last_job()["status"] == "cancelled"
    assert store.last_job()["release_confirmed"] == 1
    assert runner.ready and not store.pending()
    assert backend.cancels == ["fixture-call-id"]


async def test_repeated_stop_during_cancel_notification_preserves_readiness(config, store, backend):
    runner = Runner(config, store, backend)
    await runner.recover()
    backend.poll_release.clear()
    delivery_entered, delivery_resume = asyncio.Event(), asyncio.Event()

    async def reply(text):
        delivery_entered.set()
        await delivery_resume.wait()

    task = await runner.start(1, config.allowed_chat_id, "question", reply)
    await asyncio.wait_for(backend.poll_entered.wait(), timeout=2)
    await runner.stop()
    try:
        await asyncio.wait_for(delivery_entered.wait(), timeout=2)
        assert "关闭已确认" in await runner.stop()
        assert runner.ready and not store.pending()
    finally:
        delivery_resume.set()
        await asyncio.wait_for(task, timeout=2)
    assert runner.ready and backend.cancels == ["fixture-call-id"]


@pytest.fixture
def backend():
    return BackendDouble()


async def test_success_history_and_dedup(config, store, backend):
    replies = []

    async def reply(text):
        replies.append(text)

    runner = Runner(config, store, backend)
    await runner.recover()
    task = await runner.start(1, config.allowed_chat_id, "问题", reply)
    await task
    assert store.last_job()["status"] == "completed"
    assert len(store.history(config.allowed_chat_id)) == 2
    assert await runner.start(1, config.allowed_chat_id, "问题", reply) is None
    assert backend.spawns == 1


async def test_stop_during_spawn_cancels_after_handle(config, store, backend):
    backend.spawn_release.clear()
    runner = Runner(config, store, backend)
    await runner.recover()
    replies = []

    async def reply(text):
        replies.append(text)

    task = await runner.start(1, config.allowed_chat_id, "问题", reply)
    await backend.spawn_entered.wait()
    response = await runner.stop()
    assert "等待" in response and not backend.cancels
    backend.spawn_release.set()
    await task
    assert backend.cancels == ["fixture-call-id"]
    assert store.last_job()["status"] == "cancelled"
    assert not store.history(config.allowed_chat_id)


async def test_stop_inflight_discard_result(config, store, backend):
    backend.poll_release.clear()
    runner = Runner(config, store, backend)
    await runner.recover()

    async def reply(text):
        pass

    task = await runner.start(1, config.allowed_chat_id, "问题", reply)
    await backend.poll_entered.wait()
    await runner.stop()
    await task
    assert backend.cancels and store.last_job()["status"] == "cancelled"
    assert not store.history(config.allowed_chat_id)


async def test_restart_cancels_durable_call_without_resubmit(config, store, backend):
    store.start(1, config.allowed_chat_id, "问题")
    store.set_call(1, "fixture-call-id")
    runner = Runner(config, store, backend)
    await runner.recover()
    assert backend.cancels == ["fixture-call-id"] and backend.spawns == 0
    assert runner.ready and store.last_job()["status"] == "cancelled"


async def test_ambiguous_spawn_fails_closed(config, store, backend):
    backend.fail_spawn = True
    runner = Runner(config, store, backend)
    await runner.recover()

    async def reply(text):
        pass

    await (await runner.start(1, config.allowed_chat_id, "问题", reply))
    assert not runner.ready and store.pending()[0]["call_id"] is None
    restored = Runner(config, store, backend)
    await restored.recover()
    assert not restored.ready and backend.spawns == 1


async def test_remote_still_busy_not_resubmitted(config, store, backend):
    backend.busy = True
    runner = Runner(config, store, backend)
    await runner.recover()

    async def reply(text):
        pass

    assert await runner.start(1, config.allowed_chat_id, "问题", reply) is None
    assert backend.spawns == 0 and not store.last_job()


@pytest.mark.parametrize(
    "key,value",
    [
        ("model_id", "another/model"),
        ("revision", "0" * 40),
        ("loaded_model_path", "/other"),
        ("loaded_dtype", "torch.float16"),
        ("quantization", "fp8"),
        ("gpu_name", "H100"),
        ("gpu_memory_bytes", 40 * 1024**3),
    ],
)
def test_mismatching_engine_evidence_rejected(key, value):
    report = fixture_result()
    report[key] = value
    with pytest.raises(ValueError):
        validate_result(report)


class ReleaseBackendDouble(BackendDouble):
    """Local-only release failures; never calls an external API."""

    def __init__(self):
        super().__init__()
        self.fail_idle = False
        self.idle_checks = 0

    async def wait_idle(self):
        self.idle_checks += 1
        if self.fail_idle:
            raise TimeoutError("local-only idle failure")
        return await super().wait_idle()


@pytest.mark.parametrize("delivery_fails", [False, True])
async def test_completed_history_does_not_wait_for_warm_gpu_release(config, store, delivery_fails):
    backend = ReleaseBackendDouble()
    backend.fail_idle = True
    replies = []

    async def reply(text):
        replies.append(text)
        if delivery_fails:
            raise RuntimeError("local-only delivery failure")

    runner = Runner(config, store, backend)
    await runner.recover()
    await (await runner.start(1, config.allowed_chat_id, "问题", reply))
    assert backend.idle_checks == 0
    assert not store.pending()
    assert store.last_job()["status"] == "completed"
    assert len(store.history(config.allowed_chat_id)) == 2
    assert runner.ready
    assert backend.cancels == []
    assert replies.count(fixture_result()["text"]) == 1

    async def blocked_reply(text):
        replies.append(text)

    await (await runner.start(2, config.allowed_chat_id, "下一轮", blocked_reply))
    assert backend.spawns == 2


async def test_delivery_failure_retains_history_and_chat_readiness(config, store):
    backend = ReleaseBackendDouble()
    runner = Runner(config, store, backend)
    await runner.recover()

    async def reply(text):
        raise RuntimeError("local-only delivery failure")

    await (await runner.start(1, config.allowed_chat_id, "问题", reply))
    assert backend.idle_checks == 0
    assert runner.ready and not store.pending()
    assert store.last_job()["status"] == "completed"
    assert len(store.history(config.allowed_chat_id)) == 2
    assert backend.cancels == []


@pytest.mark.parametrize("operation", ["restart", "stop"])
async def test_completed_pending_release_survives_restart_or_stop(config, store, operation):
    from modal_a100_chat.store import Store

    backend = ReleaseBackendDouble()
    backend.fail_idle = True
    runner = Runner(config, store, backend)
    await runner.recover()
    delivered = []

    async def reply(text):
        delivered.append(text)

    await (await runner.start(1, config.allowed_chat_id, "问题", reply))
    assert "未确认" in await runner.stop()
    reopened = Store(config.state_dir)
    try:
        recovered = Runner(config, reopened, backend) if operation == "restart" else runner
        if operation == "restart":
            with pytest.raises(TimeoutError):
                await recovered.recover()
        else:
            assert "未确认" in await recovered.stop()
        assert not recovered.ready
        assert reopened.pending()[0]["call_id"] == "fixture-call-id"
        assert backend.cancels == []
        backend.fail_idle = False
        if operation == "restart":
            await recovered.recover()
        else:
            await recovered.stop()
        assert recovered.ready and not reopened.pending()
        assert reopened.last_job()["status"] == "completed"
        assert len(reopened.history(config.allowed_chat_id)) == 2
        assert backend.cancels == [] and backend.spawns == 1
        assert delivered.count(fixture_result()["text"]) == 1
    finally:
        reopened.db.close()


async def test_stop_during_delivery_does_not_cancel_successful_history(config, store):
    backend = ReleaseBackendDouble()
    runner = Runner(config, store, backend)
    await runner.recover()
    delivery_entered, delivery_release = asyncio.Event(), asyncio.Event()

    async def reply(text):
        delivery_entered.set()
        await delivery_release.wait()

    task = await runner.start(1, config.allowed_chat_id, "问题", reply)
    await delivery_entered.wait()
    try:
        await runner.stop()
        assert backend.cancels == []
    finally:
        delivery_release.set()
        await task
    assert store.last_job()["status"] == "completed"
    assert not store.pending() and len(store.history(config.allowed_chat_id)) == 2
