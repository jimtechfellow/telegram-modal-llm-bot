"""Authorized status queries never submit inference or refresh idle residency."""

from types import SimpleNamespace

import pytest

from modal_a100_chat.bot import build_application


@pytest.mark.parametrize("containers,label", [(1, "暖容器空闲"), (0, "已释放")])
async def test_status_distinguishes_warm_from_released_without_keepalive(config, containers, label):
    application = build_application(config, "0:unit-test-not-a-real-token")
    runner = application.bot_data["runner"]
    replies = []

    class ReadOnlyBackend:
        effective_idle_seconds = 300

        async def stats(self):
            return {"num_total_runners": containers, "num_running_inputs": 0, "backlog": 0}

    async def reply(text):
        replies.append(text)

    runner.backend = ReadOnlyBackend()
    update = SimpleNamespace(
        effective_chat=SimpleNamespace(id=config.allowed_chat_id, type="private"),
        effective_user=SimpleNamespace(id=config.allowed_user_id),
        effective_message=SimpleNamespace(reply_text=reply),
    )
    handler = next(h for h in application.handlers[0] if getattr(h, "commands", None) == {"status"})
    try:
        await handler.callback(update, None)
        assert label in replies[0] and "300 秒" in replies[0]
        assert runner.store.last_job() is None
    finally:
        runner.store.db.close()
