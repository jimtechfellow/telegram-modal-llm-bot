"""Unit fixtures are not GPU/Telegram acceptance evidence."""

import socket
from pathlib import Path

import pytest

from modal_a100_chat.config import Config
from modal_a100_chat.store import Store


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Unit tests must never contact Telegram, Modal or any other network service."""

    def blocked(*args, **kwargs):
        raise AssertionError("Network access is forbidden in offline unit tests")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked)


@pytest.fixture
def config(tmp_path):
    return Config(100000001, 100000001, "unit-workspace", "main", tmp_path / "state", "中文")


@pytest.fixture
def store(config):
    result = Store(config.state_dir)
    yield result
    result.db.close()


@pytest.fixture
def repository():
    return Path(__file__).resolve().parents[1]
