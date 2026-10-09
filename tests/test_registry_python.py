"""Local regression for Modal's documented registry Python executable contract."""

import ast
import inspect

from modal_a100_chat import modal_app


def test_pinned_vllm_registry_exposes_existing_python3_as_python():
    tree = ast.parse(inspect.getsource(modal_app))
    registry = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "from_registry"
    )
    setup = next((kw.value for kw in registry.keywords if kw.arg == "setup_dockerfile_commands"), None)
    assert setup is not None, "Modal requires python on PATH; the fixed vLLM image has python3 only"
    commands = ast.literal_eval(setup)
    assert "RUN ln -sf /usr/bin/python3 /usr/local/bin/python" in commands
    assert "RUN python --version && python -m pip --version" in commands
    assert not any(kw.arg == "add_python" for kw in registry.keywords)
