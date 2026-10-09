"""Exact target invariants: do not add fallbacks or quantization."""

MODEL_ID = "orcarouter/Qwen3.8-27B-Uncensored"
MODEL_REVISION = "8cb32d72080f6a47bf34dc5adf8067daa6c63a31"
APP_NAME = "modal-a100-chat"
VOLUME_NAME = "modal-a100-chat-qwen38-bf16"
GPU = "A100-80GB"
VLLM_VERSION = "0.31.0"
MAX_MODEL_LEN = 4096
MAX_OUTPUT_TOKENS = 512
MODEL_CACHE_KEY = "orcarouter--Qwen3.8-27B-Uncensored/" + MODEL_REVISION
MODEL_DIR = "/models/" + MODEL_CACHE_KEY
