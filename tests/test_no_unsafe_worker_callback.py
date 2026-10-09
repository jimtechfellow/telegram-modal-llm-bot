"""Avoid optional Python callback serialization; preserve required engine evidence."""

import ast
import inspect

from modal_a100_chat.inference import load_engine, run_inference


def test_inference_does_not_require_insecure_callback_serialization():
    source = inspect.getsource(load_engine) + "\n" + inspect.getsource(run_inference)
    tree = ast.parse(source)
    calls = [
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    ]
    assert "apply_model" not in calls
    assert "VLLM_ALLOW_INSECURE_SERIALIZATION" not in source
    for evidence in (
        "loaded.model",
        "loaded.dtype",
        "loaded.quantization",
        "get_device_name",
        "total_memory",
        "verify(root, manifest",
    ):
        assert evidence in source
