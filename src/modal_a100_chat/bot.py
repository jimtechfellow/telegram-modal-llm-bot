"""Independent Telegram long polling; every handler checks private chat + sender."""

import argparse
import fcntl
import logging
import os
import sys
from contextlib import contextmanager
from pathlib import Path

from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters

from .config import Config
from .constants import MODEL_ID, MODEL_REVISION, VOLUME_NAME
from .services import create_runner
from .store import Store


@contextmanager
def exclusive_bot_lock(state_dir):
    """Bot and mutating admin commands share this lock, including across processes."""
    state_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (state_dir / "bot.lock").open("a") as lock:
        os.chmod(lock.name, 0o600)
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Bot or a mutating administrative operation is active") from None
        yield


def build_application(config, token):
    store = Store(config.state_dir)
    runner = create_runner(config, store)

    async def authorized(update):
        chat, user, message = update.effective_chat, update.effective_user, update.effective_message
        allowed = bool(chat and user and config.authorized(chat.id, user.id, chat.type))
        if not allowed and message:
            await message.reply_text("未授权。")
        return allowed

    async def reply(message, text):
        # Telegram limits UTF-16 code units; avoid splitting surrogate pairs.
        part = ""
        length = 0
        for char in text:
            units = len(char.encode("utf-16-le")) // 2
            if length + units > 3500:
                await message.reply_text(part)
                part, length = "", 0
            part += char
            length += units
        if part:
            await message.reply_text(part)

    async def text_handler(update, context):
        if not await authorized(update):
            return
        message = update.effective_message
        if len(message.text) > 16000:
            await message.reply_text("消息过长，请缩短；不会启动 GPU。")
            return
        await runner.start(
            update.update_id,
            update.effective_chat.id,
            message.text,
            lambda text: reply(message, text),
        )

    async def status_handler(update, context):
        if not await authorized(update):
            return
        last = store.last_job()
        lines = [f"模型：{MODEL_ID}", f"Revision：{MODEL_REVISION}", f"Volume：{VOLUME_NAME}"]
        lines += [
            f"本地任务：{last['status'] if last else '无'}",
            f"业务阶段：{runner.phase}",
            f"可提交：{runner.ready}",
            f"空闲保留配置：{config.gpu_idle_timeout_seconds} 秒",
            f"最近确认空闲窗口：{runner.backend.effective_idle_seconds} 秒",
        ]
        try:
            stats = await runner.backend.stats()
            if not any(stats.values()):
                store.confirm_completed_releases()
            if runner.releasing:
                state = "释放中或释放未确认"
            elif stats["num_running_inputs"] or stats["backlog"]:
                state = "推理中" if runner.active else "存在未确认调用"
            elif runner.active and not runner.active.completed:
                state = "正在提交或等待推理"
            else:
                state = "暖容器空闲（仍计费）" if stats["num_total_runners"] else "已释放"
            lines += [
                f"GPU 状态：{state}",
                f"GPU 容器：{stats['num_total_runners']}",
                f"运行请求：{stats['num_running_inputs']}；排队：{stats['backlog']}",
            ]
        except Exception:
            lines += ["Modal 状态查询失败，未宣称 GPU 已关闭。"]
        await reply(update.effective_message, "\n".join(lines))

    async def stop_handler(update, context):
        if await authorized(update):
            await update.effective_message.reply_text(await runner.stop())

    async def unknown_handler(update, context):
        if await authorized(update):
            await update.effective_message.reply_text("支持普通文本、调研 XXX、/status、/stop。")

    async def post_init(application):
        # Refuse to hijack an already webhook-driven bot.
        info = await application.bot.get_webhook_info()
        if info.url:
            raise RuntimeError("Provided bot has an existing webhook; use a new independent bot")
        await runner.recover()

    async def post_stop(application):
        await runner.close()

    async def error_handler(update, context):
        if isinstance(update, Update) and update.effective_message:
            if await authorized(update):
                await update.effective_message.reply_text("接入服务请求失败；没有自动重试推理。")

    application = (
        Application.builder()
        .token(token)
        .concurrent_updates(False)
        .post_init(post_init)
        .post_stop(post_stop)
        .build()
    )
    application.add_handler(CommandHandler("status", status_handler))
    application.add_handler(CommandHandler("stop", stop_handler))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_handler))
    application.add_handler(MessageHandler(filters.ALL, unknown_handler))
    application.add_error_handler(error_handler)
    application.bot_data["runner"] = runner
    return application


def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path.home() / ".config/modal_a100_chat/config.toml")
    args = parser.parse_args()
    os.umask(0o077)
    # httpx request logs contain Telegram API token URLs; do not enable logging.
    logging.disable(logging.CRITICAL)
    config = Config.load(args.config)
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        parser.error("Inject TELEGRAM_BOT_TOKEN into the process environment")
    with exclusive_bot_lock(config.state_dir):
        app = build_application(config, token)
        app.run_polling(
            timeout=30, drop_pending_updates=False, allowed_updates=["message"], bootstrap_retries=0
        )


def main():
    # This boundary covers both console-script and python -m startup/runtime failures.
    # PTB InvalidToken may contain the entire token; never render messages or chains.
    try:
        _main()
    except Exception as exc:
        name = type(exc).__name__
        safe_type = "".join(c for c in name if c.isascii() and (c.isalnum() or c == "_"))
        print(safe_type or "Exception", file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
