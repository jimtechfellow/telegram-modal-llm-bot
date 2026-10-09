"""Small adapter around the official asynchronous Modal API."""

import asyncio
import time
from dataclasses import asdict

import modal
from modal.exception import TimeoutError as ModalTimeoutError

from .constants import APP_NAME, MODEL_DIR, MODEL_ID, MODEL_REVISION


class ModalBackend:
    def __init__(self, config):
        self.config = config
        self.function = modal.Function.from_name(APP_NAME, "infer", environment_name=config.modal_environment)

    async def check_workspace(self):
        workspace = modal.Workspace.from_context()
        await workspace.hydrate.aio()
        if workspace.name != self.config.modal_workspace:
            raise RuntimeError("Modal credential workspace differs from configuration")

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
