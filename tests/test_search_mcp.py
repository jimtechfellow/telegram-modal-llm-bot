"""Validate pi-web-access MCP artifacts, strict Exa routing and safe failures."""

import asyncio
import importlib
import json
import shutil
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from search_bot.pi_web_access import PiWebAccess, parse_artifact
from search_bot.service import ResearchError


def artifact(**changes):
    return {"type": "research", "provider": "exa", "summary": "真实检索摘要。", "sources": [
        {"title": "官方文档", "url": "https://docs.example.org/reference"}
    ], **changes}


def test_accepts_source_artifact_and_deduplicates_urls():
    data = artifact()
    data["sources"] *= 2
    result = parse_artifact(json.dumps(data), 5)
    assert len(result.sources) == 1 and result.summary == "真实检索摘要。"


@pytest.mark.parametrize("data", [
    {}, artifact(provider="duckduckgo"), artifact(errors=[{"error": "secret-from-provider"}]),
    artifact(sources=[]), artifact(summary=""), artifact(summary=17),
    artifact(sources=[{"title": "错误", "url": "file:///etc/passwd"}]),
    artifact(sources=[{"title": "错误", "url": "https://user:pass@example.org"}]),
    artifact(sources=[{"title": "错误", "url": "https://127.0.0.1/x"}]),
    artifact(sources=[{"title": "错误", "url": "https://metadata.internal/x"}]),
    artifact(sources=[{"title": "错误", "url": "https://docs.example.org/\nInjected"}]),
])
def test_invalid_artifacts_fail_closed_without_raw_error(data):
    with pytest.raises(ResearchError) as caught:
        parse_artifact(json.dumps(data), 5)
    assert "secret-from-provider" not in str(caught.value)


async def test_missing_key_never_starts_mcp(monkeypatch):
    monkeypatch.delenv("EXA_API_KEY", raising=False)
    with pytest.raises(ResearchError, match="EXA_API_KEY"):
        await PiWebAccess().search("容器")


async def test_one_strict_exa_search_and_one_local_artifact_read(monkeypatch):
    from search_bot import pi_web_access as module

    monkeypatch.setenv("EXA_API_KEY", "unit-only-exa-key")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "must-not-be-in-child-environment")
    calls, parameters = [], []

    class ClientDouble:
        server_info = SimpleNamespace(name="pi-web-access", version="0.37.0")

        async def __aenter__(self):
            await self.transport.__aenter__()
            return self

        async def __aexit__(self, *args):
            await self.transport.__aexit__(*args)

        async def call_tool(self, name, arguments):
            calls.append((name, arguments))
            text = ("Artifact responseId: research-unit (retrievable via get_search_content)."
                    if name == "source_check" else json.dumps(artifact()))
            return SimpleNamespace(is_error=False, content=[SimpleNamespace(type="text", text=text)])

    def client(transport, **kwargs):
        result = ClientDouble()
        result.transport = transport
        return result

    @asynccontextmanager
    async def stdio(params, **kwargs):
        parameters.append(params)
        yield object()

    monkeypatch.setattr(module, "Client", client)
    monkeypatch.setattr(module, "stdio_client", stdio)
    result = await PiWebAccess(command="unit-mcp").search("容器")
    assert result.sources[0].url == "https://docs.example.org/reference"
    assert calls[0] == ("source_check", {"claim": "容器", "queries": ["容器"], "provider": "exa",
                                          "numResults": 5, "fetchContent": False, "proxy": ""})
    assert [name for name, _ in calls] == ["source_check", "get_search_content"]
    assert parameters[0].env["EXA_API_KEY"] == "unit-only-exa-key"
    assert parameters[0].env["EXA_BASE_URL"] == "https://api.exa.ai"
    assert "TELEGRAM_BOT_TOKEN" not in parameters[0].env
    assert not parameters[0].args  # no shell, npx, package download, or key in argv


@pytest.fixture
def real_offline_mcp(monkeypatch, repository):
    from search_bot import pi_web_access as module

    monkeypatch.setenv("EXA_API_KEY", "unit-only-exa-key")
    actual_stdio = module.stdio_client
    stdio_module = importlib.import_module("mcp.client.stdio")
    actual_spawn = stdio_module._create_platform_compatible_process
    processes = []
    started = asyncio.Event()

    async def spawn(**kwargs):
        process = await actual_spawn(**kwargs)
        processes.append(process)
        started.set()
        return process

    def offline_stdio(params, **kwargs):
        params.command = shutil.which("node")
        assert params.command, "Install Node >=22.19 and run npm ci --prefix mcp before the full suite"
        params.args = [str(repository / "tests/fixtures/pi_exa_offline.mjs")]
        return actual_stdio(params, **kwargs)

    monkeypatch.setattr(module, "stdio_client", offline_stdio)
    monkeypatch.setattr(stdio_module, "_create_platform_compatible_process", spawn)
    started.processes = processes
    yield started
    assert processes and all(process.returncode is not None for process in processes)


async def test_real_pinned_mcp_uses_paid_exa_http_path_without_any_network(real_offline_mcp):
    result = await PiWebAccess().search("容器复用")
    assert result.sources[0].url == "https://docs.example.org/lifecycle"
    assert "空闲窗口" in result.summary


async def test_real_pinned_mcp_exa_failure_is_explicit_and_does_not_fallback(real_offline_mcp):
    with pytest.raises(ResearchError) as caught:
        await PiWebAccess().search("模拟失败")
    assert "unit-only-exa-key" not in str(caught.value)


async def test_real_pinned_mcp_timeout_closes_child(real_offline_mcp):
    with pytest.raises(ResearchError):
        await PiWebAccess(timeout_seconds=5).search("模拟超时")


async def test_real_pinned_mcp_cancellation_closes_child(real_offline_mcp):
    task = asyncio.create_task(PiWebAccess().search("模拟超时"))
    await asyncio.wait_for(real_offline_mcp.wait(), 2)
    await asyncio.sleep(0.2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.parametrize("cancel_again", [False, True])
async def test_stop_overlapping_deadline_waits_for_real_child_exit(
    real_offline_mcp, monkeypatch, cancel_again
):
    stdio_module = importlib.import_module("mcp.client.stdio")
    actual_stop = stdio_module._stop_server_process
    cleaning = asyncio.Event()

    async def delayed_stop(process):
        cleaning.set()
        await asyncio.sleep(0.6)  # Cleanup necessarily outlasts the search deadline.
        await actual_stop(process)

    monkeypatch.setattr(stdio_module, "_stop_server_process", delayed_stop)
    adapter = PiWebAccess()
    adapter.timeout_seconds = 0.4
    task = asyncio.create_task(adapter.search("模拟超时"))
    try:
        await asyncio.wait_for(real_offline_mcp.wait(), 2)
        task.cancel()
        await asyncio.wait_for(cleaning.wait(), 2)
        if cancel_again:
            task.cancel()
        done, _ = await asyncio.wait({task}, timeout=4)
        assert task in done, "Search cancellation hung across its deadline"
        with pytest.raises(asyncio.CancelledError):
            task.result()
        assert all(p.returncode is not None for p in real_offline_mcp.processes)
    finally:
        # Reproduction must not leak children even when run against the broken code.
        for process in real_offline_mcp.processes:
            if process.returncode is None:
                process.kill()
                await process.wait()
        await asyncio.wait({task}, timeout=2)
        if task.done() and not task.cancelled():
            task.exception()


async def test_cleanup_wait_is_bounded_without_recancelling_worker(monkeypatch):
    from search_bot import pi_web_access as module
    from search_bot.service import ResearchCleanupError

    monkeypatch.setattr(module, "CLEANUP_TIMEOUT_SECONDS", 0.05)
    cleaning, finish = asyncio.Event(), asyncio.Event()

    async def run():
        try:
            await asyncio.Future()
        finally:
            cleaning.set()
            await finish.wait()

    worker = asyncio.create_task(run())
    waiter = asyncio.create_task(module._wait_for_search(worker, cleaning, 0.01))
    try:
        await asyncio.wait_for(cleaning.wait(), 1)
        waiter.cancel()
        with pytest.raises(ResearchCleanupError, match="退出未确认"):
            await asyncio.wait_for(waiter, 1)
        assert not worker.done() and worker.cancelling() == 1
    finally:
        finish.set()
        with pytest.raises(asyncio.CancelledError):
            await worker


@pytest.mark.parametrize("stop", [False, True])
async def test_real_transport_cleanup_error_is_not_reported_as_success(
    real_offline_mcp, monkeypatch, stop
):
    from search_bot.service import ResearchCleanupError

    stdio_module = importlib.import_module("mcp.client.stdio")
    actual_stop = stdio_module._stop_server_process

    async def failed_stop(process):
        await actual_stop(process)  # Never leave a child behind while injecting the error.
        raise OSError("synthetic transport cleanup failure")

    monkeypatch.setattr(stdio_module, "_stop_server_process", failed_stop)
    task = asyncio.create_task(PiWebAccess().search("模拟超时" if stop else "容器"))
    if stop:
        await asyncio.wait_for(real_offline_mcp.wait(), 2)
        task.cancel()
    with pytest.raises(ResearchCleanupError):
        await asyncio.wait_for(task, 4)


@pytest.mark.parametrize("stop", [False, True])
async def test_deadline_or_stop_does_not_interrupt_normal_transport_exit(
    real_offline_mcp, monkeypatch, stop
):
    stdio_module = importlib.import_module("mcp.client.stdio")
    actual_stop = stdio_module._stop_server_process
    cleaning = asyncio.Event()

    async def delayed_stop(process):
        cleaning.set()
        await asyncio.sleep(0.6)
        await actual_stop(process)

    monkeypatch.setattr(stdio_module, "_stop_server_process", delayed_stop)
    adapter = PiWebAccess()
    adapter.timeout_seconds = 0.4
    task = asyncio.create_task(adapter.search("容器"))
    try:
        await asyncio.wait_for(cleaning.wait(), 2)
        if stop:
            task.cancel()
        with pytest.raises(asyncio.CancelledError if stop else ResearchError):
            await asyncio.wait_for(task, 4)
        assert all(p.returncode is not None for p in real_offline_mcp.processes)
    finally:
        for process in real_offline_mcp.processes:
            if process.returncode is None:
                process.kill()
                await process.wait()
        await asyncio.wait({task}, timeout=2)
        if task.done() and not task.cancelled():
            task.exception()
