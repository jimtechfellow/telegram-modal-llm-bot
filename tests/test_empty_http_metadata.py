"""Actual SDK metadata can include an empty HTTP config for a non-web Function."""

from types import SimpleNamespace

import pytest
from test_deployment import info

from modal_a100_chat.deployment_verification import assert_function_info


def test_empty_http_config_is_not_a_server_endpoint():
    fixture = info()
    fixture._inner_server_info = SimpleNamespace(server_url="", port=0, sessioned=False)
    assert_function_info("infer", fixture, "fixture-image-infer", "fixture-volume", "fixture-secret")


@pytest.mark.parametrize(
    "port,url,sessioned", [(8000, "", False), (0, "https://unit.invalid", False), (0, "", True)]
)
def test_configured_inner_server_is_rejected(port, url, sessioned):
    fixture = info()
    fixture._inner_server_info = SimpleNamespace(server_url=url, port=port, sessioned=sessioned)
    with pytest.raises(RuntimeError):
        assert_function_info("infer", fixture, "fixture-image-infer", "fixture-volume", "fixture-secret")
