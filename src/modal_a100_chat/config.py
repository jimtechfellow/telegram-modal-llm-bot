"""Non-secret configuration; authorization includes chat and sender."""

import tomllib
from dataclasses import dataclass
from pathlib import Path

from .constants import DEFAULT_GPU_IDLE_SECONDS, MODEL_ID, MODEL_REVISION


@dataclass(frozen=True)
class Config:
    allowed_chat_id: int
    allowed_user_id: int
    modal_workspace: str
    modal_environment: str
    state_dir: Path
    system_prompt: str
    request_timeout_seconds: int = 1200
    model_id: str = MODEL_ID
    model_revision: str = MODEL_REVISION
    gpu_idle_timeout_seconds: int = DEFAULT_GPU_IDLE_SECONDS
    search_mcp_command: str | None = None
    search_timeout_seconds: int = 60
    search_max_results: int = 5

    def __post_init__(self):
        if type(self.gpu_idle_timeout_seconds) is not int or not 2 <= self.gpu_idle_timeout_seconds <= 1200:
            raise ValueError("GPU idle timeout must be an integer in 2..1200 seconds")
        if type(self.search_timeout_seconds) is not int or not 5 <= self.search_timeout_seconds <= 90:
            raise ValueError("Search timeout must be an integer in 5..90 seconds")
        if type(self.search_max_results) is not int or not 1 <= self.search_max_results <= 5:
            raise ValueError("Search results must be an integer in 1..5")
        if self.search_mcp_command is not None and (
            not isinstance(self.search_mcp_command, str) or not self.search_mcp_command.strip()
        ):
            raise ValueError("Search MCP command must be an executable path, not an empty value")

    @classmethod
    def load(cls, path: Path):
        with path.open("rb") as stream:
            values = tomllib.load(stream)
        for key in ("allowed_chat_id", "allowed_user_id"):
            if type(values.get(key)) is not int or values[key] <= 0:
                raise ValueError(f"{key} must be a confirmed private Telegram id")
        if not values.get("modal_workspace") or "REPLACE" in values["modal_workspace"]:
            raise ValueError("Confirm the Modal workspace before deployment")
        if values.get("model_id", MODEL_ID) != MODEL_ID or (
            values.get("model_revision", MODEL_REVISION) != MODEL_REVISION
        ):
            raise ValueError("Configuration must use the exact pinned Qwen3.8 model")
        values["state_dir"] = Path(values["state_dir"]).expanduser()
        if not 30 <= values.get("request_timeout_seconds", 1200) <= 1200:
            raise ValueError("Request timeout must be 30..1200 seconds")
        return cls(**values)

    def authorized(self, chat_id: int, user_id: int, chat_type: str) -> bool:
        return chat_type == "private" and chat_id == self.allowed_chat_id and user_id == self.allowed_user_id
