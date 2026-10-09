import pytest

from modal_a100_chat.config import Config
from modal_a100_chat.store import Store


@pytest.mark.parametrize(
    "chat,user,kind,allowed",
    [
        (100000001, 100000001, "private", True),
        (1, 100000001, "private", False),
        (100000001, 1, "private", False),
        (100000001, 100000001, "group", False),
        (100000001, 100000001, "channel", False),
    ],
)
def test_owner_only(config, chat, user, kind, allowed):
    assert config.authorized(chat, user, kind) is allowed


def test_history_survives_restart_and_deduplicates(config):
    store = Store(config.state_dir)
    query = "中文问题'); DROP TABLE messages; --"
    assert store.start(1, config.allowed_chat_id, query)
    assert not store.start(1, config.allowed_chat_id, query)
    store.set_call(1, "fixture-call-id")
    store.complete(1, {"text": "中文回复", "model_id": config.model_id})
    store.db.close()
    restored = Store(config.state_dir)
    try:
        assert restored.history(config.allowed_chat_id) == [
            {"role": "user", "content": query},
            {"role": "assistant", "content": "中文回复"},
        ]
        assert restored.pending()[0]["call_id"] == "fixture-call-id"
        restored.confirm_release(1)
        assert not restored.pending()
    finally:
        restored.db.close()


def test_legacy_completed_call_requires_release_confirmation(config):
    """Local-only legacy SQLite fixture: no remote result or API call."""
    import sqlite3

    config.state_dir.mkdir(parents=True)
    with sqlite3.connect(config.state_dir / "history.sqlite3") as database:
        database.execute(
            "CREATE TABLE jobs (update_id INTEGER PRIMARY KEY, chat_id INTEGER NOT NULL, "
            "user_text TEXT NOT NULL, status TEXT NOT NULL, call_id TEXT, created REAL NOT NULL, "
            "evidence TEXT, error TEXT)"
        )
        database.execute(
            "INSERT INTO jobs VALUES (1, ?, 'local-only', 'completed', 'local-only-call', 0, '{}', NULL)",
            (config.allowed_chat_id,),
        )
    restored = Store(config.state_dir)
    try:
        assert restored.pending()[0]["call_id"] == "local-only-call"
        restored.confirm_release(1)
        assert not restored.pending() and restored.last_job()["status"] == "completed"
    finally:
        restored.db.close()


def test_cancelled_result_cannot_enter_history(store, config):
    store.start(1, config.allowed_chat_id, "问题")
    store.mark(1, "cancelled")
    with pytest.raises(RuntimeError):
        store.complete(1, {"text": "不应保存"})
    assert store.history(config.allowed_chat_id) == []


def test_load_config_without_removed_billing_gate(repository, tmp_path):
    text = (repository / "config.example.toml").read_text().replace("REPLACE_WITH_VERIFIED_WORKSPACE", "test")
    text = text.replace("allowed_chat_id = 0", "allowed_chat_id = 100000001")
    text = text.replace("allowed_user_id = 0", "allowed_user_id = 100000001")
    path = tmp_path / "config.toml"
    path.write_text(text)
    assert Config.load(path).model_id == "orcarouter/Qwen3.8-27B-Uncensored"
    path.write_text(text.replace("orcarouter/Qwen3.8-27B-Uncensored", "another/model"))
    with pytest.raises(ValueError, match="pinned"):
        Config.load(path)
