"""Exercise the real SDK adapter with local metadata doubles, never remote resources."""

from dataclasses import make_dataclass, replace
from types import SimpleNamespace

import pytest

from modal_a100_chat import backend as module


@pytest.fixture
def backend(config, monkeypatch):
    Settings = make_dataclass(
        "Settings",
        [
            ("min_containers", int),
            ("max_containers", int),
            ("buffer_containers", int),
            ("scaledown_window", int),
        ],
    )
    Counts = make_dataclass(
        "Counts",
        [
            ("num_total_runners", int),
            ("num_running_inputs", int),
            ("backlog", int),
        ],
    )
    state = SimpleNamespace(
        updates=[],
        counts=[Counts(0, 0, 0)],
        mismatch=False,
        info=SimpleNamespace(
            single_use_containers=False, concurrency_info=None, gpus=[("A100-80GB", 1)], max_retries=0
        ),
    )

    async def scaling(**kwargs):
        state.updates.append(kwargs)
        settings = Settings(**kwargs)
        return replace(settings, min_containers=1) if state.mismatch else settings

    async def stats():
        return state.counts.pop(0) if len(state.counts) > 1 else state.counts[0]

    async def info(**kwargs):
        assert kwargs == {"refresh": True}
        return state.info

    function = SimpleNamespace(
        update_autoscaler=SimpleNamespace(aio=scaling),
        get_current_stats=SimpleNamespace(aio=stats),
        info=SimpleNamespace(aio=info),
    )
    monkeypatch.setattr(module.modal.Function, "from_name", lambda *args, **kwargs: function)
    return module.ModalBackend(config), state, Counts


async def test_native_scaler_reuses_same_pool_and_restores_configured_window(backend):
    adapter, state, _ = backend
    adapter.config = replace(adapter.config, gpu_idle_timeout_seconds=123)
    await adapter.configure()
    await adapter.release()
    await adapter.configure()
    assert adapter.effective_idle_seconds == 123
    assert [update["scaledown_window"] for update in state.updates] == [123, 2, 123]
    assert all(
        (u["min_containers"], u["max_containers"], u["buffer_containers"]) == (0, 1, 0) for u in state.updates
    )


async def test_effective_settings_mismatch_does_not_claim_configuration(backend):
    adapter, state, _ = backend
    state.mismatch = True
    with pytest.raises(RuntimeError, match="autoscaler"):
        await adapter.configure()
    assert adapter.effective_idle_seconds is None


@pytest.mark.parametrize("busy", [(1, 0, 0), (0, 1, 0), (0, 0, 1)])
async def test_release_requires_all_three_counts_zero(backend, monkeypatch, busy):
    adapter, state, Counts = backend
    state.counts = [Counts(*busy), Counts(0, 0, 0)]
    sleeps = []

    async def sleep(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr(module.asyncio, "sleep", sleep)
    assert await adapter.release() == {"num_total_runners": 0, "num_running_inputs": 0, "backlog": 0}
    assert sleeps == [2]


async def test_release_timeout_does_not_claim_zero(backend, monkeypatch):
    adapter, state, Counts = backend
    state.counts = [Counts(1, 0, 0)]
    ticks = iter([0, 90])
    monkeypatch.setattr(module, "time", SimpleNamespace(monotonic=lambda: next(ticks)))
    with pytest.raises(TimeoutError, match="shutdown not confirmed"):
        await adapter.release()
    assert adapter.effective_idle_seconds == 2


@pytest.mark.parametrize(
    "field,value",
    [
        ("single_use_containers", True),
        ("concurrency_info", SimpleNamespace(max_inputs=2)),
        ("gpus", [("H100", 1)]),
        ("max_retries", 1),
    ],
)
async def test_incompatible_remote_function_blocks_reuse(backend, field, value):
    adapter, state, _ = backend
    await adapter.check_inference()
    setattr(state.info, field, value)
    with pytest.raises(RuntimeError, match="dedicated serial"):
        await adapter.check_inference()
    assert not state.updates


async def test_cancel_targets_only_supplied_call(backend):
    adapter, _, _ = backend
    requests = []

    async def cancel(**kwargs):
        requests.append(kwargs)

    await adapter.cancel(SimpleNamespace(cancel=SimpleNamespace(aio=cancel)))
    assert requests == [{"terminate_containers": True}]
