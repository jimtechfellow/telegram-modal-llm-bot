"""Fake CUDA/vLLM only: exercise the real inference path without a GPU or network."""

import json
import sys
from types import SimpleNamespace

import pytest

from modal_a100_chat import inference
from modal_a100_chat.constants import MODEL_ID, MODEL_REVISION, VLLM_VERSION


@pytest.fixture
def engine(monkeypatch, tmp_path):
    marker = {
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "full_hash_verified": True,
        "manifest_sha256": "fixture-hash",
    }
    (tmp_path / "verified.json").write_text(json.dumps(marker))
    state = SimpleNamespace(loads=0, checks=0, generations=0, fail_load=False)
    tokenizer = SimpleNamespace(
        apply_chat_template=lambda messages, **kwargs: messages[-1]["content"],
        encode=lambda text, **kwargs: list(text),
    )

    class LLM:
        def __init__(self, **kwargs):
            state.loads += 1
            if state.fail_load:
                raise RuntimeError("fixture initialization failure")
            self.llm_engine = SimpleNamespace(
                vllm_config=SimpleNamespace(
                    model_config=SimpleNamespace(
                        model=str(tmp_path),
                        dtype="torch.bfloat16",
                        quantization=None,
                    )
                )
            )

        def get_tokenizer(self):
            return tokenizer

        def generate(self, prompts, params, **kwargs):
            state.generations += 1
            return [
                SimpleNamespace(
                    prompt_token_ids=[1],
                    outputs=[SimpleNamespace(text="fixture reply", token_ids=[2])],
                )
            ]

    def verify(*args, **kwargs):
        state.checks += 1

    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            cuda=SimpleNamespace(
                device_count=lambda: 1,
                is_bf16_supported=lambda: True,
                get_device_name=lambda index: "fixture A100",
                get_device_properties=lambda index: SimpleNamespace(total_memory=80 * 1024**3),
            )
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm",
        SimpleNamespace(
            __version__=VLLM_VERSION,
            LLM=LLM,
            SamplingParams=lambda **kwargs: kwargs,
        ),
    )
    monkeypatch.setattr(inference, "MODEL_DIR", str(tmp_path))
    monkeypatch.setattr(inference, "digest_file", lambda path: "fixture-hash")
    monkeypatch.setattr(inference, "verify", verify)
    monkeypatch.setattr("modal_a100_chat.weights.load_manifest", lambda path: {})
    monkeypatch.setattr(inference.subprocess, "check_output", lambda *args, **kwargs: "fixture GPU")
    monkeypatch.setenv("MODAL_TASK_ID", "fixture-container")
    yield state
    if hasattr(inference, "load_engine"):
        inference.load_engine.cache_clear()


def test_two_generations_reuse_loaded_engine(engine):
    messages = [{"role": "user", "content": "hello"}]
    first = inference.run_inference(messages, "fixture-manifest")
    second = inference.run_inference(messages, "fixture-manifest")
    assert engine.loads == engine.checks == 1
    assert engine.generations == 2
    assert first["engine_instance_id"] == second["engine_instance_id"]
    assert first["container_id"] == second["container_id"] == "fixture-container"
    assert first["engine_load_seconds"] >= 0
    assert first["engine_reused"] is False and second["engine_reused"] is True
    inference.load_engine.cache_clear()  # simulate a fresh container process
    third = inference.run_inference(messages, "fixture-manifest")
    assert engine.loads == 2
    assert third["engine_instance_id"] != first["engine_instance_id"]


def test_failed_initialization_is_not_cached(engine):
    engine.fail_load = True
    messages = [{"role": "user", "content": "hello"}]
    with pytest.raises(RuntimeError, match="initialization"):
        inference.run_inference(messages, "fixture-manifest")
    engine.fail_load = False
    result = inference.run_inference(messages, "fixture-manifest")
    assert result["text"] == "fixture reply" and engine.loads == 2
