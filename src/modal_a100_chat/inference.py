"""One verified vLLM engine per container; sequential calls reuse loaded weights."""

import json
import os
import re
import subprocess
import time
from functools import lru_cache
from pathlib import Path
from uuid import uuid4

from .constants import (
    GPU,
    MAX_MODEL_LEN,
    MAX_OUTPUT_TOKENS,
    MODEL_DIR,
    MODEL_ID,
    MODEL_REVISION,
    VLLM_VERSION,
    VOLUME_NAME,
)
from .weights import digest_file, verify


def engine_options():
    return dict(
        model=MODEL_DIR,
        tokenizer=MODEL_DIR,
        dtype="bfloat16",
        quantization=None,
        load_format="safetensors",
        trust_remote_code=False,
        tensor_parallel_size=1,
        max_model_len=MAX_MODEL_LEN,
        max_num_seqs=1,
        gpu_memory_utilization=0.85,
        enforce_eager=True,
        enable_prefix_caching=False,
        language_model_only=True,
    )


def fit_prompt(tokenizer, messages):
    messages = [dict(m) for m in messages]
    if not messages or messages[-1]["role"] != "user":
        raise ValueError("A current user message is required")
    while True:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        if len(tokenizer.encode(prompt, add_special_tokens=False)) <= MAX_MODEL_LEN - MAX_OUTPUT_TOKENS:
            return prompt, messages
        start = 1 if messages[0]["role"] == "system" else 0
        if len(messages) - start <= 1:
            raise ValueError("Current message exceeds the context limit")
        del messages[start : start + 2]  # trim complete historical user/assistant pairs only


@lru_cache(maxsize=1)
def load_engine(manifest_path):
    import torch
    import vllm
    from vllm import LLM

    from .weights import load_manifest

    start = time.monotonic()
    manifest_path = Path(manifest_path)
    manifest = load_manifest(manifest_path)
    root = Path(MODEL_DIR)
    marker = json.loads((root / "verified.json").read_text())
    if (
        marker.get("model_id") != MODEL_ID
        or marker.get("revision") != MODEL_REVISION
        or marker.get("full_hash_verified") is not True
        or marker.get("manifest_sha256") != digest_file(manifest_path)
    ):
        raise RuntimeError("Volume has no matching fully verified model")
    verify(root, manifest, full_hash=False)
    if vllm.__version__ != VLLM_VERSION:
        raise RuntimeError("Unexpected vLLM image version")
    if torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Exactly one BF16-capable GPU is required")
    gpu_name = torch.cuda.get_device_name(0)
    total_bytes = torch.cuda.get_device_properties(0).total_memory
    if "A100" not in gpu_name or total_bytes < 75 * 1024**3:
        raise RuntimeError("The fixed A100-80GB GPU is not present; no fallback allowed")
    smi = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,memory.total,uuid", "--format=csv,noheader,nounits"],
        text=True,
        timeout=10,
    ).strip()
    llm = LLM(**engine_options())
    loaded = llm.llm_engine.vllm_config.model_config
    if str(loaded.model) != MODEL_DIR or str(loaded.dtype) != "torch.bfloat16" or loaded.quantization:
        raise RuntimeError("Loaded model path/dtype/quantization mismatch")
    return llm, {
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "loaded_model_path": str(loaded.model),
        "loaded_dtype": str(loaded.dtype),
        "quantization": loaded.quantization,
        "precision_evidence": "full-hash-verified BF16 Safetensors and actual vLLM model config",
        "gpu_requested": GPU,
        "gpu_name": gpu_name,
        "gpu_memory_bytes": total_bytes,
        "nvidia_smi": smi,
        "volume_name": VOLUME_NAME,
        "manifest_sha256": marker["manifest_sha256"],
        "vllm_version": vllm.__version__,
        "container_id": os.environ["MODAL_TASK_ID"],
        "engine_instance_id": uuid4().hex,
        "engine_load_seconds": time.monotonic() - start,
    }


def run_inference(messages, manifest_path):
    from vllm import SamplingParams

    start = time.monotonic()
    hits = load_engine.cache_info().hits
    llm, evidence = load_engine(manifest_path)
    reused = load_engine.cache_info().hits > hits
    prompt, used_messages = fit_prompt(llm.get_tokenizer(), messages)
    output = llm.generate(
        [prompt],
        SamplingParams(max_tokens=MAX_OUTPUT_TOKENS, temperature=0.7, top_p=0.9),
        use_tqdm=False,
    )[0]
    text = re.sub(r"<think>.*?</think>", "", output.outputs[0].text, flags=re.S).strip()
    if not text:
        raise RuntimeError("The model produced no final text")
    return {
        **evidence,
        "text": text,
        "engine_reused": reused,
        "context_messages": len(used_messages),
        "prompt_tokens": len(output.prompt_token_ids),
        "output_tokens": len(output.outputs[0].token_ids),
        "elapsed_seconds": time.monotonic() - start,
    }
