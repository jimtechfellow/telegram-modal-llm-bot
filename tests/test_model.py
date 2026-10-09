"""Pinning, actual file/header checks and rejection tests; no mocked GPU claim."""

import copy
import json
import struct

import pytest

from modal_a100_chat.constants import GPU, MODEL_DIR, MODEL_ID, MODEL_REVISION, VOLUME_NAME
from modal_a100_chat.inference import engine_options, fit_prompt
from modal_a100_chat.modal_app import GPU_REUSE, SINGLE_USE
from modal_a100_chat.weights import digest_file, load_manifest, read_safetensors_header, verify


def test_exact_qwen38_target(repository):
    manifest = load_manifest(repository / "model-manifest.json")
    assert MODEL_ID == "orcarouter/Qwen3.8-27B-Uncensored"
    assert MODEL_REVISION == "8cb32d72080f6a47bf34dc5adf8067daa6c63a31"
    assert "Qwen3.8" in MODEL_DIR and MODEL_REVISION in MODEL_DIR
    assert VOLUME_NAME == "modal-a100-chat-qwen38-bf16"
    assert GPU == "A100-80GB"
    assert len(manifest["files"]) == 26
    shards = {f["name"] for f in manifest["files"] if f["name"].endswith(".safetensors")}
    assert shards == {f"model-{number:05d}-of-00018.safetensors" for number in range(1, 19)}
    assert len(shards) == 18
    assert manifest["total_weight_bytes"] == sum(f["size"] for f in manifest["files"] if f["name"] in shards)


@pytest.mark.parametrize("mutation", ["wrong_model", "wrong_revision", "gguf", "missing", "traversal"])
def test_manifest_rejects_mutation(repository, tmp_path, mutation):
    data = json.loads((repository / "model-manifest.json").read_text())
    if mutation == "wrong_model":
        data["model_id"] = "another/model"
    elif mutation == "wrong_revision":
        data["revision"] = "0" * 40
    elif mutation == "gguf":
        data["files"][0]["name"] = "weights.gguf"
    elif mutation == "missing":
        data["files"].pop()
    else:
        data["files"][0]["name"] = "../outside.safetensors"
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        load_manifest(path)


def tensor_file(path, dtype="BF16", broken=False):
    header = {"weight": {"dtype": dtype, "shape": [2], "data_offsets": [0, 4 if not broken else 6]}}
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"\0" * 4)


def test_real_safetensors_bf16_header(tmp_path):
    path = tmp_path / "sample.safetensors"
    tensor_file(path)
    assert read_safetensors_header(path)["weight"]["dtype"] == "BF16"


@pytest.mark.parametrize("kind", ["FP16", "offset", "truncated"])
def test_bad_safetensors_rejected(tmp_path, kind):
    path = tmp_path / "sample.safetensors"
    if kind == "truncated":
        path.write_bytes(b"123")
    else:
        tensor_file(path, dtype="F16" if kind == "FP16" else "BF16", broken=kind == "offset")
    with pytest.raises(ValueError):
        read_safetensors_header(path)


def small_fixture(tmp_path):
    tensor_file(tmp_path / "sample.safetensors")
    (tmp_path / "config.json").write_text(
        json.dumps(
            {"text_config": {"dtype": "bfloat16"}, "architectures": ["Qwen3_5ForConditionalGeneration"]}
        )
    )
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"weight": "sample.safetensors"}})
    )
    files = []
    for path in tmp_path.iterdir():
        files.append({"name": path.name, "size": path.stat().st_size, "sha256": digest_file(path)})
    return {"files": files, "expected_architectures": ["Qwen3_5ForConditionalGeneration"]}


def test_full_integrity_and_nested_dtype(tmp_path):
    manifest = small_fixture(tmp_path)
    result = verify(tmp_path, manifest)
    assert result["full_hash_verified"] and result["dtype"] == "BF16"
    assert result["model_id"] == MODEL_ID
    path = tmp_path / "sample.safetensors"
    raw = path.read_bytes()
    path.write_bytes(raw[:-1] + b"1")
    with pytest.raises(ValueError, match="Checksum"):
        verify(tmp_path, manifest)


def test_wrong_index_rejected(tmp_path):
    manifest = small_fixture(tmp_path)
    (tmp_path / "model.safetensors.index.json").write_text('{"weight_map": {"other": "sample.safetensors"}}')
    with pytest.raises(ValueError):
        verify(tmp_path, manifest, full_hash=False)


def test_engine_remains_bf16_single_gpu():
    options = engine_options()
    assert options["model"] == MODEL_DIR
    assert options["dtype"] == "bfloat16" and options["quantization"] is None
    assert options["max_model_len"] == 4096 and options["tensor_parallel_size"] == 1
    assert options["max_num_seqs"] == 1 and options["enforce_eager"]
    assert SINGLE_USE == dict(
        max_containers=1, min_containers=0, buffer_containers=0, single_use_containers=True, retries=0
    )
    assert GPU_REUSE == dict(
        max_containers=1,
        min_containers=0,
        buffer_containers=0,
        single_use_containers=False,
        scaledown_window=300,
        retries=0,
    )


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["enable_thinking"] is False
        return "|".join(m["content"] for m in messages)

    def encode(self, text, **kwargs):
        return list(text)


def test_context_trims_pairs_not_current_query():
    messages = [
        {"role": "system", "content": "系统"},
        {"role": "user", "content": "x" * 4000},
        {"role": "assistant", "content": "以前回复"},
        {"role": "user", "content": "现在的问题"},
    ]
    original = copy.deepcopy(messages)
    _, used = fit_prompt(Tokenizer(), messages)
    assert used == [original[0], original[-1]] and messages == original


def test_oversize_current_query_does_not_change_model():
    with pytest.raises(ValueError, match="context"):
        fit_prompt(Tokenizer(), [{"role": "user", "content": "x" * 5000}])
