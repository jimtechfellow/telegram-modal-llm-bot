"""Private Modal Functions: one reusable GPU container that scales to zero."""

import json
from pathlib import Path

import modal

from .constants import APP_NAME, DEFAULT_GPU_IDLE_SECONDS, GPU, MODEL_DIR, VOLUME_NAME

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = "/opt/model-manifest.json"
HF_SECRET_NAME = "modal-a100-chat-hf"
VLLM_IMAGE = "vllm/vllm-openai@sha256:c1c9f6fd5c109ba7f0546a59f5b2f15fb87f64c77782e90a27b648b42a8e67c3"
app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
cpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install("huggingface-hub==2.1.1")
    .add_local_python_source("modal_a100_chat")
    .add_local_file(REPO_ROOT / "model-manifest.json", MANIFEST)
)
gpu_image = (
    modal.Image.from_registry(
        VLLM_IMAGE,
        setup_dockerfile_commands=[
            "RUN ln -sf /usr/bin/python3 /usr/local/bin/python",
            "RUN python --version && python -m pip --version",
        ],
    )
    .entrypoint([])
    .env({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "VLLM_LOGGING_LEVEL": "WARNING"})
    .add_local_python_source("modal_a100_chat")
    .add_local_file(REPO_ROOT / "model-manifest.json", MANIFEST)
)
SINGLE_USE = dict(
    max_containers=1,
    min_containers=0,
    buffer_containers=0,
    single_use_containers=True,
    retries=0,
)
GPU_REUSE = dict(
    max_containers=1,
    min_containers=0,
    buffer_containers=0,
    single_use_containers=False,
    scaledown_window=DEFAULT_GPU_IDLE_SECONDS,
    retries=0,
)


@app.function(
    image=cpu_image,
    volumes={"/models": volume},
    secrets=[modal.Secret.from_name(HF_SECRET_NAME)],
    cpu=2,
    memory=4096,
    timeout=3600,
    **SINGLE_USE,
)
def download_model():
    from .weights import digest_file, download, load_manifest

    report = download(Path(MODEL_DIR), load_manifest(Path(MANIFEST)))
    report["manifest_sha256"] = digest_file(Path(MANIFEST))
    (Path(MODEL_DIR) / "verified.json").write_text(json.dumps(report, indent=2))
    volume.commit()
    return report


@app.function(
    image=gpu_image,
    gpu=GPU,
    volumes={"/models": volume.read_only()},
    cpu=4,
    memory=98304,
    timeout=1200,
    startup_timeout=600,
    block_network=True,
    **GPU_REUSE,
)
def infer(messages):
    from .inference import run_inference

    return run_inference(messages, MANIFEST)
