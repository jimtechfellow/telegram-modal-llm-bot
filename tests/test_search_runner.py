"""Research shares the existing durable single-task GPU lifecycle."""

import asyncio
import json

import pytest
from test_runner import BackendDouble, fixture_result

from modal_a100_chat.runner import Runner
from search_bot.service import ResearchCleanupError, ResearchError, SearchBot, SearchEvidence, Source


class SearchDouble:
    def __init__(self):
        self.queries = []
        self.entered = asyncio.Event()
        self.resume = asyncio.Event()
        self.resume.set()
        self.fail = False

    async def search(self, topic):
        self.queries.append(topic)
        self.entered.set()
        await self.resume.wait()
        if self.fail:
            raise ResearchError("付费 Exa 搜索失败；未回退。")
        return SearchEvidence((Source("真实来源", "https://docs.example.org/reference"),), "容器支持复用。")


class ReportBackend(BackendDouble):
    async def spawn(self, messages):
        self.messages = messages
        return await super().spawn(messages)

    async def poll(self, call):
        result = await super().poll(call)
        result["text"] = json.dumps({"findings": [{"text": "容器支持复用。", "sources": [1]}],
                                     "limitations": "仅根据检索摘要。"})
        return result


@pytest.fixture
def research(config, store):
    search, backend, replies = SearchDouble(), ReportBackend(), []
    runner = Runner(config, store, backend, search_bot=SearchBot(search))

    async def reply(text):
        replies.append(text)

    return runner, search, backend, replies, reply


async def test_research_report_history_and_dedup_share_normal_lifecycle(research, config, store):
    runner, search, backend, replies, reply = research
    await runner.recover()
    await (await runner.start(1, config.allowed_chat_id, "调研 GPU 复用", reply))
    assert search.queries == ["GPU 复用"] and backend.spawns == 1
    assert "研究报告" in replies[-1] and "https://docs.example.org/reference" in replies[-1]
    assert store.history(config.allowed_chat_id)[-1]["content"] == replies[-1]
    assert store.last_job()["status"] == "completed" and not store.pending()
    assert store.last_job()["release_confirmed"] == 0 and runner.ready
    assert await runner.start(1, config.allowed_chat_id, "调研 GPU 复用", reply) is None
    assert search.queries == ["GPU 复用"] and backend.spawns == 1


async def test_normal_chat_never_uses_search(research, config):
    runner, search, backend, replies, reply = research
    await runner.recover()
    await (await runner.start(1, config.allowed_chat_id, "你好", reply))
    assert not search.queries and backend.spawns == 1


async def test_stop_during_search_never_spawns_model_and_preserves_admission(research, config, store):
    runner, search, backend, replies, reply = research
    await runner.recover()
    search.resume.clear()
    task = await runner.start(1, config.allowed_chat_id, "调研 GPU", reply)
    await asyncio.wait_for(search.entered.wait(), 2)
    assert store.last_job()["status"] == "searching" and runner.phase == "检索中"
    assert await runner.start(2, config.allowed_chat_id, "next", reply) is None
    assert "取消" in await runner.stop()
    await asyncio.wait_for(task, 2)
    assert backend.spawns == 0 and not backend.cancels
    assert runner.ready and not runner.active and not store.pending()
    assert store.last_job()["status"] == "cancelled" and store.last_job()["release_confirmed"] == 1
    assert not store.history(config.allowed_chat_id)


async def test_stop_during_report_generation_cancels_exact_remote_call(research, config, store):
    runner, search, backend, replies, reply = research
    await runner.recover()
    backend.poll_release.clear()
    task = await runner.start(1, config.allowed_chat_id, "调研 GPU", reply)
    await asyncio.wait_for(backend.poll_entered.wait(), 2)
    assert runner.phase == "生成报告"
    await runner.stop()
    await asyncio.wait_for(task, 2)
    assert backend.cancels == ["fixture-call-id"]
    assert store.last_job()["release_confirmed"] == 1 and not store.history(config.allowed_chat_id)


async def test_search_failure_has_no_gpu_or_history_and_next_chat_is_allowed(research, config, store):
    runner, search, backend, replies, reply = research
    await runner.recover()
    search.fail = True
    await (await runner.start(1, config.allowed_chat_id, "调研 GPU", reply))
    assert "搜索失败" in replies[-1] and backend.spawns == 0
    assert store.last_job()["status"] == "failed" and runner.ready and not store.pending()
    assert not store.history(config.allowed_chat_id)
    await (await runner.start(2, config.allowed_chat_id, "你好", reply))
    assert backend.spawns == 1 and len(search.queries) == 1


@pytest.mark.parametrize("stop", [False, True])
async def test_unconfirmed_mcp_exit_stays_blocked_even_after_stop_and_recovery(
    research, config, store, stop
):
    runner, search, backend, replies, reply = research

    async def unconfirmed(topic):
        search.entered.set()
        try:
            await search.resume.wait()
        finally:
            raise ResearchCleanupError("synthetic cleanup failure")

    search.search = unconfirmed
    await runner.recover()
    if stop:
        search.resume.clear()
    task = await runner.start(1, config.allowed_chat_id, "调研 GPU", reply)
    await search.entered.wait()
    if stop:
        await runner.stop()
    await task
    assert "退出未确认" in replies[-1] and not runner.ready
    assert store.last_job()["error"] == "search_cleanup_unconfirmed"
    assert store.pending() and not store.history(config.allowed_chat_id)
    assert not backend.spawns and not backend.cancels
    assert await runner.start(2, config.allowed_chat_id, "你好", reply) is None
    assert "BLOCKED" in await runner.stop()
    await runner.recover()
    assert not runner.ready and store.pending()


async def test_invalid_model_citations_release_gpu_without_committing_report(research, config, store):
    runner, search, backend, replies, reply = research

    async def bad_result(call):
        return fixture_result()

    backend.poll = bad_result
    await runner.recover()
    await (await runner.start(1, config.allowed_chat_id, "调研 GPU", reply))
    assert "校验失败" in replies[-1]
    assert backend.cancels == ["fixture-call-id"] and backend.windows[-1] == 2
    assert store.last_job()["status"] == "failed" and not store.history(config.allowed_chat_id)


async def test_searching_restart_does_not_replay_search_or_allocate_gpu(research, config, store):
    runner, search, backend, replies, reply = research
    store.start(1, config.allowed_chat_id, "调研 GPU")
    store.mark(1, "searching")
    await runner.recover()
    assert not search.queries and backend.spawns == 0 and not backend.cancels
    assert store.last_job()["status"] == "cancelled" and runner.ready and not store.pending()


async def test_ambiguous_model_spawn_after_search_still_fails_closed(research, config, store):
    runner, search, backend, replies, reply = research
    await runner.recover()
    backend.fail_spawn = True
    await (await runner.start(1, config.allowed_chat_id, "调研 GPU", reply))
    assert not runner.ready and store.pending()[0]["status"] == "starting"
    assert "提交状态未确认" in replies[-1]


async def test_admin_record_blocks_research_without_search_charge(research, config):
    runner, search, backend, replies, reply = research
    await runner.recover()
    (config.state_dir / "admin-call.json").write_text("{}")
    assert await runner.start(1, config.allowed_chat_id, "调研 GPU", reply) is None
    assert not search.queries and not runner.ready and backend.spawns == 0


async def test_stop_before_research_task_starts_never_searches(research, config, store):
    runner, search, backend, replies, reply = research
    await runner.recover()
    task = await runner.start(1, config.allowed_chat_id, "调研 GPU", reply)
    await runner.stop()
    await asyncio.wait_for(task, 2)
    assert not search.queries and not backend.spawns and runner.ready
    assert store.last_job()["status"] == "cancelled"


@pytest.mark.parametrize("failure", ["stop", "report"])
async def test_unconfirmed_cleanup_keeps_durable_state_without_unhandled_task_failure(
    research, config, store, failure
):
    runner, search, backend, replies, reply = research
    await runner.recover()
    backend.fail_idle = True
    if failure == "stop":
        search.resume.clear()
    else:
        async def invalid_report(call):
            return fixture_result()
        backend.poll = invalid_report
    task = await runner.start(1, config.allowed_chat_id, "调研 GPU", reply)
    if failure == "stop":
        await asyncio.wait_for(search.entered.wait(), 2)
        await runner.stop()
    await asyncio.wait_for(task, 2)
    assert not runner.ready and store.pending() and not store.history(config.allowed_chat_id)
    assert "未确认" in replies[-1]


async def test_recover_searching_never_releases_unknown_running_call(research, config, store):
    runner, search, backend, replies, reply = research
    store.start(1, config.allowed_chat_id, "调研 GPU")
    store.mark(1, "searching")
    backend.busy = True
    await runner.recover()
    assert not runner.ready and store.pending() and not backend.cancels
    assert not backend.windows and not search.queries


async def test_repeated_stop_does_not_interrupt_search_process_cleanup(research, config, store):
    runner, search, backend, replies, reply = research
    cleaning, finish_cleanup = asyncio.Event(), asyncio.Event()

    async def slow_cleanup(topic):
        search.entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaning.set()
            await finish_cleanup.wait()

    search.search = slow_cleanup
    await runner.recover()
    task = await runner.start(1, config.allowed_chat_id, "调研 GPU", reply)
    try:
        await asyncio.wait_for(search.entered.wait(), 2)
        await runner.stop()
        await asyncio.wait_for(cleaning.wait(), 2)
        await runner.stop()
        assert runner.active.research_task.cancelling() == 1
        assert not task.done() and not backend.spawns
    finally:
        finish_cleanup.set()
        await asyncio.wait_for(task, 2)
    assert not store.pending() and runner.ready
