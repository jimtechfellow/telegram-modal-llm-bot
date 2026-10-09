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

    async def check_workspace(self):
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
        return {"num_total_runners": 0, "num_running_inputs": 0, "backlog": 0}

    async def stats(self):
        return {"num_total_runners": int(self.busy), "num_running_inputs": 0, "backlog": 0}


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
async def test_completed_history_stays_pending_until_release_confirmed(config, store, delivery_fails):
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
    assert backend.idle_checks == 1
    assert [row["call_id"] for row in store.pending()] == ["fixture-call-id"]
    assert store.last_job()["status"] == "completed"
    assert len(store.history(config.allowed_chat_id)) == 2
    assert not runner.ready
    assert backend.cancels == []
    assert replies.count(fixture_result()["text"]) == 1

    async def blocked_reply(text):
        replies.append(text)

    assert await runner.start(2, config.allowed_chat_id, "不能重试", blocked_reply) is None
    assert backend.spawns == 1


async def test_delivery_failure_still_confirms_idle_and_retains_history(config, store):
    backend = ReleaseBackendDouble()
    runner = Runner(config, store, backend)
    await runner.recover()

    async def reply(text):
        raise RuntimeError("local-only delivery failure")

    await (await runner.start(1, config.allowed_chat_id, "问题", reply))
    assert backend.idle_checks == 1
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
    reopened = Store(config.state_dir)
    try:
        recovered = Runner(config, reopened, backend) if operation == "restart" else runner
        if operation == "restart":
            with pytest.raises(TimeoutError):
                await recovered.recover()
        else:
            with pytest.raises(TimeoutError):
                await recovered.stop()
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
