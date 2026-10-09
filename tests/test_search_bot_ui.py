"""Telegram delegates every message; no search business logic in its handlers."""

from types import SimpleNamespace

from telegram.ext import MessageHandler
from test_runner import BackendDouble
from test_search_runner import ReportBackend, SearchDouble

from modal_a100_chat.bot import build_application
from search_bot.service import SearchBot


async def test_telegram_research_route_and_normal_chat_remain_separate(config):
    app = build_application(config, "0:unit-test-not-a-real-token")
    runner = app.bot_data["runner"]
    search, replies = SearchDouble(), []
    runner.search_bot = SearchBot(search)

    async def reply(text):
        replies.append(text)

    handler = next(h for h in app.handlers[0] if isinstance(h, MessageHandler))
    update = SimpleNamespace(
        update_id=1, effective_chat=SimpleNamespace(id=config.allowed_chat_id, type="private"),
        effective_user=SimpleNamespace(id=config.allowed_user_id),
        effective_message=SimpleNamespace(text="普通私人聊天", reply_text=reply),
    )
    try:
        runner.backend = BackendDouble()
        await runner.recover()
        await handler.callback(update, None)
        await runner.active.task
        assert not search.queries
        runner.backend = ReportBackend()
        update.update_id = 2
        update.effective_message.text = "调研 GPU"
        await handler.callback(update, None)
        await runner.active.task
        assert search.queries == ["GPU"] and "研究报告" in replies[-1]
        assert "普通私人聊天" not in str(runner.backend.messages)
        assert "https://docs.example.org/reference" in replies[-1]
    finally:
        runner.store.db.close()
