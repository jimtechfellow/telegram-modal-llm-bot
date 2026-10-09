"""Handler-level denial test with no Telegram or Modal network calls."""

from types import SimpleNamespace

from modal_a100_chat.bot import build_application


async def test_unauthorized_message_never_queries_or_spawns_gpu(config):
    application = build_application(config, "0:unit-test-not-a-real-token")
    runner = application.bot_data["runner"]
    replies = []

    class NeverBackend:
        async def stats(self):
            raise AssertionError("Unauthorized handler contacted Modal")

    runner.backend = NeverBackend()

    async def reply(text):
        replies.append(text)

    update = SimpleNamespace(
        effective_chat=SimpleNamespace(id=1, type="private"),
        effective_user=SimpleNamespace(id=1),
        effective_message=SimpleNamespace(text="未授权消息", reply_text=reply),
        update_id=1,
    )
    try:
        for handler in application.handlers[0]:
            await handler.callback(update, None)
        assert replies and all(text == "未授权。" for text in replies)
        assert runner.store.last_job() is None
    finally:
        runner.store.db.close()
