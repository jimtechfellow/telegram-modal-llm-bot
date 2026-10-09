"""Small adapter around the official asynchronous Modal API."""

import asyncio
import time
from dataclasses import asdict

import modal
from modal.exception import TimeoutError as ModalTimeoutError

from .constants import APP_NAME, GPU, MODEL_DIR, MODEL_ID, MODEL_REVISION


async def set_idle_window(function, seconds):
    """Update the existing pool, never create a Function.with_options variant."""
    scaling = await function.update_autoscaler.aio(
        min_containers=0,
        max_containers=1,
        buffer_containers=0,
        scaledown_window=seconds,
    )
    if (
        scaling.min_containers,
        scaling.max_containers,
        scaling.buffer_containers,
        scaling.scaledown_window,
    ) != (0, 1, 0, seconds):
        raise RuntimeError("Effective GPU autoscaler settings mismatch")
    return scaling


class ModalBackend:
    def __init__(self, config):
        self.config = config
        self.function = modal.Function.from_name(APP_NAME, "infer", environment_name=config.modal_environment)
        self.effective_idle_seconds = None

    async def configure(self):
        await set_idle_window(self.function, self.config.gpu_idle_timeout_seconds)
        self.effective_idle_seconds = self.config.gpu_idle_timeout_seconds

    async def release(self):
        # The shortest native window releases an idle container without a new GPU input.
        await set_idle_window(self.function, 2)
        self.effective_idle_seconds = 2
        return await self.wait_idle()

    async def check_workspace(self):
        workspace = modal.Workspace.from_context()
        await workspace.hydrate.aio()
        if workspace.name != self.config.modal_workspace:
            raise RuntimeError("Modal credential workspace differs from configuration")

    async def check_inference(self):
        info = await self.function.info.aio(refresh=True)
        if (
            info.single_use_containers
            or info.concurrency_info is not None
            or info.gpus != [(GPU, 1)]
            or info.max_retries != 0
        ):
            raise RuntimeError("Deployed inference must use the dedicated serial reusable GPU function")

    async def spawn(self, messages):
        return await self.function.spawn.aio(messages)

    @staticmethod
    def from_id(call_id):
        return modal.FunctionCall.from_id(call_id)

    @staticmethod
    async def poll(call):
        try:
            return await call.get.aio(timeout=2)
        except (ModalTimeoutError, TimeoutError):
            return None

    @staticmethod
    async def cancel(call):
        await call.cancel.aio(terminate_containers=True)

    async def stats(self):
        return asdict(await self.function.get_current_stats.aio())

    async def wait_idle(self, timeout=90):
        deadline = time.monotonic() + timeout
        while True:
            stats = await self.stats()
            if stats["num_total_runners"] == 0 and stats["num_running_inputs"] == 0 and stats["backlog"] == 0:
                return stats
            if time.monotonic() >= deadline:
                raise TimeoutError("GPU container shutdown not confirmed")
            await asyncio.sleep(2)


def validate_result(result):
    if (
        result.get("model_id") != MODEL_ID
        or result.get("revision") != MODEL_REVISION
        or result.get("loaded_model_path") != MODEL_DIR
        or result.get("loaded_dtype") != "torch.bfloat16"
        or result.get("quantization")
        or "A100" not in result.get("gpu_name", "")
        or result.get("gpu_memory_bytes", 0) < 75 * 1024**3
        or not result.get("text")
    ):
        raise ValueError("Actual engine evidence differs from the requested Qwen3.8 BF16 / A100")
    return result
