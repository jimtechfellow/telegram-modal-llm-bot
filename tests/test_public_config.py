"""Public examples must stay unconfigured and exclude real runtime artifacts."""

import subprocess

import pytest

from modal_a100_chat.config import Config


def test_example_requires_explicit_owner_configuration(repository):
    with pytest.raises(ValueError, match="confirmed private Telegram id"):
        Config.load(repository / "config.example.toml")
    variables = {}
    for line in (repository / ".env.example").read_text().splitlines():
        if line and not line.startswith("#"):
            key, value = line.split("=", 1)
            variables[key] = value
    for key in ("TELEGRAM_BOT_TOKEN", "HF_TOKEN", "MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET"):
        assert variables[key] == ""


def test_gitignore_excludes_runtime_data_but_keeps_examples(repository):
    ignored = [
        ".env",
        ".env.production",
        ".modal.toml",
        "config.toml",
        "bws-map.json",
        "history.sqlite3",
        "history.sqlite3-wal",
        "chat.db",
        "admin-call.json",
        "weights/model.safetensors",
        "model.gguf",
        "logs/bot.log",
        "evidence/private.json",
    ]
    for name in ignored:
        result = subprocess.run(
            ["git", "check-ignore", "--no-index", "-q", name], cwd=repository, check=False
        )
        assert result.returncode == 0, name
    for name in (".env.example", "config.example.toml", "uv.lock", "model-manifest.json"):
        result = subprocess.run(
            ["git", "check-ignore", "--no-index", "-q", name], cwd=repository, check=False
        )
        assert result.returncode == 1, name
