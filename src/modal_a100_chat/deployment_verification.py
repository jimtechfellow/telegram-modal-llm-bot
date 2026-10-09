"""Verify exact deployed controls using official SDK metadata, not local decorators."""

from dataclasses import asdict

import modal

from .constants import DEFAULT_GPU_IDLE_SECONDS, GPU


def assert_function_info(name, info, image_id, volume_id, hf_secret_id):
    infer = name == "infer"
    if (
        info.gpus != ([(GPU, 1)] if infer else [])
        or info.single_use_containers is not (not infer)
        or info.concurrency_info is not None
        or info.max_retries != 0
        or info.timeout != (1200 if infer else 3600)
        or info.image_info.image_id != image_id
        or info.web_info is not None
        or info.method_details
        or (
            info._inner_server_info is not None
            and (
                info._inner_server_info.port
                or info._inner_server_info.server_url
                or info._inner_server_info.sessioned
            )
        )
        or info.cloud_bucket_mounts
        or info.schedule is not None
        or info.secrets != ([] if infer else [hf_secret_id])
        or (infer and info.block_network is not True)
    ):
        raise RuntimeError("Deployed function controls differ from the fixed private configuration")
    if set(info.volumes) != {"/models"}:
        raise RuntimeError("Deployed Volume mount mismatch")
    mount = info.volumes["/models"]
    if mount.volume_id != volume_id or mount.read_only != infer or mount.sub_path is not None:
        raise RuntimeError("Deployed model Volume identity/access mismatch")


async def verify_remote_settings(
    function_ids,
    image_ids,
    volume_id,
    hf_secret_id,
    gpu_idle_timeout_seconds=DEFAULT_GPU_IDLE_SECONDS,
):
    evidence = {}
    for name, identifier in function_ids.items():
        function = modal.Function.from_id(identifier)
        # Explicitly set only this project's safe limits using the public SDK.
        window = {"scaledown_window": gpu_idle_timeout_seconds} if name == "infer" else {}
        await function.update_autoscaler.aio(
            min_containers=0,
            max_containers=1,
            buffer_containers=0,
            **window,
        )
        # The documented no-argument call preserves all values and returns CURRENT effective settings.
        # Function.info exposes the remaining safety fields but not autoscaler limits in SDK1.6.1.
        scaling = await function.update_autoscaler.aio()
        if (scaling.min_containers, scaling.max_containers, scaling.buffer_containers) != (0, 1, 0):
            raise RuntimeError("Effective deployed autoscaler settings mismatch")
        if name == "infer" and scaling.scaledown_window != gpu_idle_timeout_seconds:
            raise RuntimeError("Effective deployed GPU idle timeout mismatch")
        info = await function.info.aio(refresh=True)
        assert_function_info(name, info, image_ids[name], volume_id, hf_secret_id)
        if await function.get_web_url.aio():
            raise RuntimeError("A public web endpoint is prohibited")
        evidence[name] = {
            "function_id": identifier,
            "info": asdict(info),
            "autoscaler": asdict(scaling),
            "autoscaler_readback": "official no-argument update_autoscaler preserves current settings",
        }
    return evidence
