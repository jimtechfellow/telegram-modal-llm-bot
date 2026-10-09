"""Local-only process probes: fake PTB failures, no Telegram/Modal requests."""

import os
import subprocess
import sys

import pytest


@pytest.mark.parametrize("entry", ["main", "module"])
@pytest.mark.parametrize("phase", ["startup", "runtime"])
def test_bot_process_boundary_redacts_token_and_exception_chain(tmp_path, repository, entry, phase):
    script = r'''
import logging
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from telegram.error import InvalidToken
from telegram.ext import Application
from modal_a100_chat.config import Config

def fail():
    try:
        raise RuntimeError("local-only-chain-credential")
    except RuntimeError as cause:
        raise InvalidToken("local-only-telegram-token-entire-value") from cause

class LocalApplication:
    bot_data = {}
    def add_handler(self, *args): pass
    def add_error_handler(self, *args): pass
    def run_polling(self, **kwargs): fail()

class LocalBuilder:
    def __getattr__(self, name):
        return lambda *args, **kwargs: self
    def build(self):
        if phase == "startup": fail()
        return LocalApplication()

entry, phase, state = sys.argv[1:]
config = SimpleNamespace(state_dir=Path(state), modal_environment="local-only",
                         search_mcp_command=None, search_timeout_seconds=60, search_max_results=5)
logging.disable(logging.CRITICAL)
with patch.object(Config, "load", return_value=config), \
     patch.object(Application, "builder", return_value=LocalBuilder()), \
     patch("socket.socket.connect", side_effect=AssertionError("Network forbidden in local-only probe")):
    sys.argv = ["local-only"]
    if entry == "module":
        runpy.run_module("modal_a100_chat.bot", run_name="__main__")
    else:
        from modal_a100_chat.bot import main
        main()
'''
    # Supply only fabricated credentials; do not pass through any real credential environment.
    env = {
        "PATH": os.defpath,
        "HOME": str(tmp_path),
        "PYTHONPATH": str(repository / "src"),
        "TELEGRAM_BOT_TOKEN": "0:local-only-fake-token",
    }
    result = subprocess.run(
        [sys.executable, "-c", script, entry, phase, str(tmp_path / "state")],
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
    )
    output = result.stdout + result.stderr
    assert result.returncode == 1
    assert "local-only-telegram-token-entire-value" not in output
    assert "local-only-chain-credential" not in output
    assert "Traceback" not in output
    assert result.stderr.strip() == "InvalidToken"
    assert result.stdout == ""
