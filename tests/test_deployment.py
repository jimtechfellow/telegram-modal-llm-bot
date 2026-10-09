"""Local metadata doubles only; not evidence of a real Modal deployment."""

from dataclasses import make_dataclass
from types import SimpleNamespace

import pytest

from modal_a100_chat.deployment_verification import assert_function_info, verify_remote_settings


def info(name="infer"):
    return SimpleNamespace(
        gpus=[("A100-80GB", 1)] if name == "infer" else [],
        single_use_containers=name != "infer",
        concurrency_info=None,
        max_retries=0,
        timeout=1200 if name == "infer" else 3600,
        image_info=SimpleNamespace(image_id="fixture-image-" + name),
        volumes={
            "/models": SimpleNamespace(volume_id="fixture-volume", read_only=name == "infer", sub_path=None)
        },
        web_info=None,
        method_details=None,
        _inner_server_info=None,
        cloud_bucket_mounts={},
        schedule=None,
        secrets=[] if name == "infer" else ["fixture-secret"],
        block_network=name == "infer",
    )


def test_valid_exact_deployed_metadata():
    for name in ("infer", "download_model"):
        assert_function_info(name, info(name), "fixture-image-" + name, "fixture-volume", "fixture-secret")


@pytest.mark.parametrize(
    "field,value",
    [
        ("gpus", [("A100-40GB", 1)]),
        ("gpus", [("A100-80GB", 2)]),
        ("gpus", [("A100-80GB", 1), ("H100", 1)]),
        ("single_use_containers", True),
        ("concurrency_info", SimpleNamespace(max_inputs=2, target_inputs=2)),
        ("max_retries", 1),
        ("max_retries", None),
        ("web_info", SimpleNamespace(web_url="https://unit.invalid")),
        ("schedule", "fixture"),
        ("block_network", False),
        ("secrets", ["fixture-unexpected-secret"]),
    ],
)
def test_remote_mismatch_rejected(field, value):
    fixture = info()
    setattr(fixture, field, value)
    with pytest.raises(RuntimeError):
        assert_function_info("infer", fixture, "fixture-image-infer", "fixture-volume", "fixture-secret")


@pytest.mark.parametrize("kind", ["image", "volume", "writeable"])
def test_wrong_image_mount_rejected(kind):
    fixture = info()
    if kind == "image":
        fixture.image_info.image_id = "wrong-image"
    elif kind == "volume":
        fixture.volumes["/models"].volume_id = "wrong-volume"
    else:
        fixture.volumes["/models"].read_only = False
    with pytest.raises(RuntimeError):
        assert_function_info("infer", fixture, "fixture-image-infer", "fixture-volume", "fixture-secret")


@pytest.mark.parametrize("web_url", [None, ""])
async def test_official_effective_settings_readback(monkeypatch, web_url):
    calls = []
    Settings = make_dataclass(
        "Settings",
        [
            ("min_containers", int),
            ("max_containers", int),
            ("buffer_containers", int),
            ("scaledown_window", int),
        ],
    )
    Record = make_dataclass("Record", [("gpus", list)])

    class FunctionDouble:
        async def refresh(self, **kwargs):
            assert kwargs == {"refresh": True}
            return Record(gpus=[])

        async def scaler(self, **kwargs):
            calls.append(kwargs)
            return Settings(0, 1, 0, 300)

        async def url(self):
            return web_url

        def __init__(self):
            self.info = SimpleNamespace(aio=self.refresh)
            self.update_autoscaler = SimpleNamespace(aio=self.scaler)
            self.get_web_url = SimpleNamespace(aio=self.url)

    monkeypatch.setattr(
        "modal_a100_chat.deployment_verification.modal.Function.from_id", lambda identifier: FunctionDouble()
    )
    monkeypatch.setattr("modal_a100_chat.deployment_verification.assert_function_info", lambda *a: None)
    report = await verify_remote_settings(
        {"infer": "fixture-id"}, {"infer": "fixture-image"}, "fixture-volume", "fixture-secret"
    )
    assert report["infer"]["autoscaler"]["min_containers"] == 0
    assert calls == [
        {"min_containers": 0, "max_containers": 1, "buffer_containers": 0, "scaledown_window": 300},
        {},
    ]
