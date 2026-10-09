"""One durable call at a time; /stop never loses a call that is still spawning."""

import asyncio
import time
from dataclasses import dataclass, field

from .backend import validate_result


@dataclass
class Active:
    update_id: int
    call: object = None
    stopped: bool = False
    task: object = None
    completed: bool = False
    cancel_sent: bool = False
    cancel_lock: object = field(default_factory=asyncio.Lock)
    cleanup_started: bool = False
    released: bool = False


class Runner:
    def __init__(self, config, store, backend):
        self.config, self.store, self.backend = config, store, backend
        self.lock = asyncio.Lock()
        self.active = None
        self.ready = False
        self.releasing = False

    async def recover(self):
        self.ready = False
        await self.backend.check_workspace()
        await self.backend.check_inference()
        if (self.config.state_dir / "admin-call.json").exists():
            return
        pending = self.store.pending()
        for row in pending:
            if not row["call_id"]:
                # A crash during spawn may have left a remote call without a durable id.
                # Never resubmit or pretend that it is safe. Scoped admin action is required.
                self.ready = False
                return
            completed = row["status"] == "completed"
            if not completed:
                await self.backend.cancel(self.backend.from_id(row["call_id"]))
        if pending:
            await self.backend.release()
        for row in pending:
            completed = row["status"] == "completed"
            self.store.confirm_release(
                row["update_id"],
                None if completed else "cancelled",
                None if completed else "restart_recovery",
            )
        stats = await self.backend.stats()
        if stats["num_running_inputs"] or stats["backlog"] or stats["num_total_runners"] > 1:
            return
        if not any(stats.values()):
            self.store.confirm_completed_releases()
        await self.backend.configure()
        self.releasing = False
        self.ready = True

    async def start(self, update_id, chat_id, text, reply):
        async with self.lock:
            if (self.config.state_dir / "admin-call.json").exists():
                self.ready = False
            if not self.ready or self.store.pending() and self.active is None:
                await reply("BLOCKED：有未确认的历史任务，请先查看 /status 并处理。")
                return None
            if self.active:
                await reply("当前已有任务；请等待，或使用 /stop。")
                return None
            stats = await self.backend.stats()
            if stats["num_running_inputs"] or stats["backlog"] or stats["num_total_runners"] > 1:
                await reply("GPU 存在未确认调用，暂不提交新任务。")
                return None
            if not any(stats.values()):
                self.store.confirm_completed_releases()
            await self.backend.configure()
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
                await self._safe_reply(reply, "已取消任务，GPU 容器关闭已确认。")
                return
            deadline = time.monotonic() + self.config.request_timeout_seconds
            while not active.stopped:
                result = await self.backend.poll(active.call)
                if active.stopped:
                    break
                if result is not None:
                    self.store.complete(active.update_id, validate_result(result))
                    active.completed = True
                    # Completion is durable; intentional warm GPU residency is not an active job.
                    await reply(result["text"])
                    return
                if time.monotonic() >= deadline:
                    active.stopped = True
                    break
            await self._cancel(active)
            await self._safe_reply(reply, "任务已停止，GPU 容器关闭已确认。")
        except Exception as exc:
            # Never log raw remote exceptions, prompt contents, API URLs, or credentials.
            if active.completed:
                await self._safe_reply(reply, "回复投递异常；历史已保留，请查看 /status。")
            elif active.call:
                try:
                    if active.cleanup_started:
                        raise RuntimeError("Release remains unconfirmed") from None
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

    async def _release_completed(self):
        self.ready = False
        self.releasing = True
        self.store.request_completed_release()
        await self.backend.release()
        self.store.confirm_completed_releases()
        self.releasing = False
        self.ready = not self.store.pending()

    async def _request_cancel(self, active):
        async with active.cancel_lock:
            if not active.cancel_sent:
                self.store.mark(active.update_id, "cancel_requested")
                await self.backend.cancel(active.call)
                active.cancel_sent = True

    async def _cancel(self, active):
        if active.completed:
            await self._release_completed()
            return
        self.ready = False
        self.releasing = True
        active.cleanup_started = True
        await self._request_cancel(active)
        await self.backend.release()
        self.store.confirm_release(active.update_id, "cancelled")
        self.store.confirm_completed_releases()
        active.released = True
        self.releasing = False
        self.ready = not self.store.pending()

    @staticmethod
    async def _safe_reply(reply, text):
        try:
            await reply(text)
        except Exception:
            pass

    async def stop(self):
        # Hold admission closed until cancellation/idle release is acknowledged.
        # A stale cancel must never race a new input on the same warm container.
        async with self.lock:
            active = self.active
            self.ready = False
            if active:
                active.stopped = True
                if active.released:
                    self.ready = not self.store.pending()
                    return "GPU 容器关闭已确认。"
                if active.call is None:
                    return "取消已登记；正在等待 Modal 返回调用标识，返回后立即取消。"
                try:
                    if active.completed:
                        await self._release_completed()
                        return "历史已保存；GPU 容器关闭已确认。"
                    await self._request_cancel(active)
                    return "已向 Modal 请求取消并终止 GPU 容器；正在确认关闭。"
                except Exception:
                    return "GPU 释放未确认，已阻止新任务；请查看 /status 或重试 /stop。"
            if (self.config.state_dir / "admin-call.json").exists():
                return "BLOCKED：有未确认的管理员任务，需先通过管理员恢复流程处理。"
            pending = self.store.pending()
            if any(not row["call_id"] for row in pending):
                return "BLOCKED：有无调用标识的未确认任务，需管理员停止此项目专用 App。"
            try:
                if not pending:
                    stats = await self.backend.stats()
                    if stats["num_running_inputs"] or stats["backlog"] or stats["num_total_runners"] > 1:
                        return "BLOCKED：存在未确认调用；不会取消未知任务。"
                self.store.request_completed_release()
                self.releasing = True
                for row in pending:
                    if row["status"] != "completed":
                        self.store.mark(row["update_id"], "cancel_requested")
                        await self.backend.cancel(self.backend.from_id(row["call_id"]))
                await self.backend.release()
                for row in pending:
                    self.store.confirm_release(
                        row["update_id"], None if row["status"] == "completed" else "cancelled"
                    )
                self.store.confirm_completed_releases()
                self.releasing = False
                self.ready = not self.store.pending()
                return "GPU 容器关闭已确认。"
            except Exception:
                return "GPU 释放未确认，已阻止新任务；请查看 /status 或重试 /stop。"

    async def close(self):
        await self.stop()
        active = self.active
        if active and active.task:
            done, _ = await asyncio.wait([active.task], timeout=100)
            if not done:
                raise TimeoutError("Shutdown cancellation not confirmed; durable call retained")
        if self.store.pending() or self.releasing or not self.ready:
            raise TimeoutError("Shutdown release not confirmed; durable state retained")
