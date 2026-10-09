"""One durable call at a time; /stop never loses a call that is still spawning."""

import asyncio
import time
from dataclasses import dataclass

from .backend import validate_result


@dataclass
class Active:
    update_id: int
    call: object = None
    stopped: bool = False
    task: object = None
    completed: bool = False


class Runner:
    def __init__(self, config, store, backend):
        self.config, self.store, self.backend = config, store, backend
        self.lock = asyncio.Lock()
        self.active = None
        self.ready = False

    async def recover(self):
        self.ready = False
        await self.backend.check_workspace()
        for row in self.store.pending():
            if not row["call_id"]:
                # A crash during spawn may have left a remote call without a durable id.
                # Never resubmit or pretend that it is safe. Scoped admin action is required.
                self.ready = False
                return
            completed = row["status"] == "completed"
            if not completed:
                await self.backend.cancel(self.backend.from_id(row["call_id"]))
            await self.backend.wait_idle()
            self.store.confirm_release(
                row["update_id"], None if completed else "cancelled",
                None if completed else "restart_recovery",
            )
        self.ready = True

    async def start(self, update_id, chat_id, text, reply):
        async with self.lock:
            if not self.ready or self.store.pending() and self.active is None:
                await reply("BLOCKED：有未确认的历史任务，请先查看 /status 并处理。")
                return None
            if self.active:
                await reply("当前已有任务；请等待，或使用 /stop。")
                return None
            stats = await self.backend.stats()
            if any(stats[key] for key in ("num_total_runners", "num_running_inputs", "backlog")):
                await reply("GPU 上次任务仍在释放或存在未确认调用，暂不提交新任务。")
                return None
            if not self.store.start(update_id, chat_id, text):
                await reply("这条消息已处理或登记，不会重复触发 GPU。")
                return None
            active = Active(update_id)
            self.active = active
            messages = [{"role": "system", "content": self.config.system_prompt}]
            messages += self.store.history(chat_id) + [{"role": "user", "content": text}]
            active.task = asyncio.create_task(self._run(active, messages, reply))
            return active.task

    async def _run(self, active, messages, reply):
        try:
            active.call = await self.backend.spawn(messages)
            self.store.set_call(active.update_id, active.call.object_id)
            if active.stopped:
                await self._cancel(active)
                await reply("已取消任务，GPU 容器关闭已确认。")
                return
            deadline = time.monotonic() + self.config.request_timeout_seconds
            while not active.stopped:
                result = await self.backend.poll(active.call)
                if active.stopped:
                    break
                if result is not None:
                    self.store.complete(active.update_id, validate_result(result))
                    active.completed = True
                    try:
                        await reply(result["text"])
                    finally:
                        # Delivery is not a release acknowledgement, even when it fails.
                        await self._release_completed(active.update_id)
                    return
                if time.monotonic() >= deadline:
                    active.stopped = True
                    break
            await self._cancel(active)
            await reply("任务已停止，GPU 容器关闭已确认。")
        except Exception as exc:
            # Never log raw remote exceptions, prompt contents, API URLs, or credentials.
            if active.completed:
                self.ready = not self.store.pending()
                await self._safe_reply(reply, "回复投递或 GPU 关闭确认异常；历史已保留，请查看 /status。")
            elif active.call:
                try:
                    await self._cancel(active)
                    self.store.mark(active.update_id, "failed", type(exc).__name__)
                except Exception:
                    self.store.mark(active.update_id, "cancel_requested", "remote_cancel_unconfirmed")
                await self._safe_reply(reply, "任务失败；未自动重试。请使用 /status 检查 GPU 状态。")
            else:
                # Keep starting status to fail closed after an ambiguous spawn failure.
                self.ready = False
                await self._safe_reply(
                    reply, "提交状态未确认，已阻止后续任务；需管理员核查此专用 Modal App。"
                )
        finally:
            async with self.lock:
                if self.active is active:
                    self.active = None
                if self.store.pending():
                    self.ready = False

    async def _release_completed(self, update_id):
        self.ready = False
        await self.backend.wait_idle()
        self.store.confirm_release(update_id)
        self.ready = not self.store.pending()

    async def _cancel(self, active):
        if active.completed:
            await self._release_completed(active.update_id)
            return
        self.ready = False
        self.store.mark(active.update_id, "cancel_requested")
        await self.backend.cancel(active.call)
        await self.backend.wait_idle()
        self.store.confirm_release(active.update_id, "cancelled")
        self.ready = not self.store.pending()

    @staticmethod
    async def _safe_reply(reply, text):
        try:
            await reply(text)
        except Exception:
            pass

    async def stop(self):
        async with self.lock:
            active = self.active
            if active:
                active.stopped = True
        if not active:
            pending = self.store.pending()
            if pending:
                self.ready = False
            for row in pending:
                if row["call_id"]:
                    completed = row["status"] == "completed"
                    if not completed:
                        await self.backend.cancel(self.backend.from_id(row["call_id"]))
                    await self.backend.wait_idle()
                    self.store.confirm_release(row["update_id"], None if completed else "cancelled")
                else:
                    return "BLOCKED：有无调用标识的未确认任务，需管理员停止此项目专用 App。"
            if pending:
                self.ready = not self.store.pending()
            return "当前无本地任务。"
        if active.completed:
            return "历史已保存；正在确认 GPU 容器关闭，不会取消或重放已完成的推理。"
        if active.call is None:
            return "取消已登记；正在等待 Modal 返回调用标识，返回后立即取消。"
        # Cancellation is executed by this task, not by cancelling the local asyncio task.
        await self.backend.cancel(active.call)
        return "已向 Modal 请求取消并终止 GPU 容器；正在确认关闭。"

    async def close(self):
        await self.stop()
        active = self.active
        if active and active.task:
            done, _ = await asyncio.wait([active.task], timeout=100)
            if not done:
                raise TimeoutError("Shutdown cancellation not confirmed; durable call retained")
