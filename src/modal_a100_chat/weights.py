"""Exact allowlist download; full SHA, SafeTensors BF16, index consistency checks."""

import hashlib
import json
import os
import struct
from pathlib import Path

from .constants import MODEL_ID, MODEL_REVISION


def load_manifest(path: Path) -> dict:
    result = json.loads(path.read_text())
    if result["model_id"] != MODEL_ID or result["revision"] != MODEL_REVISION:
        raise ValueError("Wrong model/revision")
    names = [f["name"] for f in result["files"]]
    if len(set(names)) != len(names):
        raise ValueError("Duplicate file")
    for name in names:
        if Path(name).name != name or name.startswith(".") or "gguf" in name.lower():
            raise ValueError("Unsafe/unwanted file")
    if set(names) != {
        *(f"model-{number:05d}-of-00018.safetensors" for number in range(1, 19)),
        "config.json",
        "generation_config.json",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "preprocessor_config.json",
        "video_preprocessor_config.json",
    }:
        raise ValueError("Exact BF16 model file allowlist required")
    return result


def digest_file(path: Path, git_blob=False) -> str:
    digest = hashlib.sha1() if git_blob else hashlib.sha256()
    if git_blob:
        digest.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_safetensors_header(path: Path) -> dict:
    with path.open("rb") as stream:
        raw = stream.read(8)
        if len(raw) != 8:
            raise ValueError("Incomplete SafeTensors header")
        length = struct.unpack("<Q", raw)[0]
        if not 1 <= length <= 16 * 1024 * 1024:
            raise ValueError("Invalid SafeTensors header size")
        header = json.loads(stream.read(length))
    available = path.stat().st_size - length - 8
    intervals = []
    tensors = {}
    for name, tensor in header.items():
        if name == "__metadata__":
            continue
        if tensor["dtype"] != "BF16":
            raise ValueError(f"Non-BF16 tensor: {name}")
        start, end = tensor["data_offsets"]
        elements = 1
        for dim in tensor["shape"]:
            if type(dim) is not int or dim < 0:
                raise ValueError("Invalid tensor shape")
            elements *= dim
        if not 0 <= start <= end <= available or end - start != 2 * elements:
            raise ValueError("Invalid BF16 data offsets")
        intervals.append((start, end))
        tensors[name] = tensor
    intervals.sort()
    if not tensors or intervals[0][0] != 0 or intervals[-1][1] != available:
        raise ValueError("Incomplete tensor data")
    if any(a[1] != b[0] for a, b in zip(intervals, intervals[1:], strict=False)):
        raise ValueError("Gaps/overlapping tensor data")
    return tensors


def verify(root: Path, manifest: dict, full_hash=True) -> dict:
    tensor_files = {}
    for file in manifest["files"]:
        path = root / file["name"]
        if path.is_symlink() or not path.is_file() or path.stat().st_size != file["size"]:
            raise ValueError(f"Missing/wrong-size file: {file['name']}")
        if full_hash:
            is_git = not file.get("sha256")
            expected = file["git_blob_sha1"] if is_git else file["sha256"]
            if digest_file(path, is_git) != expected:
                raise ValueError(f"Checksum mismatch: {file['name']}")
        if path.suffix == ".safetensors":
            for name in read_safetensors_header(path):
                if name in tensor_files:
                    raise ValueError("Duplicate tensor across shards")
                tensor_files[name] = file["name"]
    index = json.loads((root / "model.safetensors.index.json").read_text())
    if tensor_files != index["weight_map"]:
        raise ValueError("SafeTensors index does not match all actual tensors")
    config = json.loads((root / "config.json").read_text())
    text_config = config.get("text_config", {})
    dtype = config.get("dtype", config.get("torch_dtype")) or text_config.get(
        "dtype", text_config.get("torch_dtype")
    )
    if dtype != "bfloat16" or config.get("quantization_config"):
        raise ValueError("Model config is not unquantized BF16")
    if config.get("architectures") != manifest["expected_architectures"]:
        raise ValueError("Architecture differs from pinned repository metadata")
    return {
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "dtype": "BF16",
        "files_verified": len(manifest["files"]),
        "tensors_verified": len(tensor_files),
        "full_hash_verified": full_hash,
    }


def download(root: Path, manifest: dict) -> dict:
    from huggingface_hub import hf_hub_download

    root.mkdir(parents=True, exist_ok=True)
    for file in manifest["files"]:
        # One explicit filename at a time; never snapshot_download or git clone.
        hf_hub_download(
            repo_id=MODEL_ID,
            revision=MODEL_REVISION,
            filename=file["name"],
            local_dir=root,
            token=os.environ["HF_TOKEN"],
        )
    report = verify(root, manifest, full_hash=True)
    temp = root / "verified.json.tmp"
    temp.write_text(json.dumps(report, indent=2))
    os.replace(temp, root / "verified.json")
    return report
