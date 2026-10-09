"""Oracle-local durable history, exact call ids, and update deduplication."""

import json
import sqlite3
import time
from pathlib import Path


class Store:
    def __init__(self, directory: Path):
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        directory.chmod(0o700)
        self.db = sqlite3.connect(directory / "history.sqlite3")
        (directory / "history.sqlite3").chmod(0o600)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS jobs (
              update_id INTEGER PRIMARY KEY, chat_id INTEGER NOT NULL,
              user_text TEXT NOT NULL, status TEXT NOT NULL, call_id TEXT,
              created REAL NOT NULL, evidence TEXT, error TEXT,
              release_confirmed INTEGER NOT NULL DEFAULT 0);
            CREATE TABLE IF NOT EXISTS messages (
              id INTEGER PRIMARY KEY, update_id INTEGER NOT NULL, chat_id INTEGER NOT NULL,
              role TEXT NOT NULL, content TEXT NOT NULL);
        """)
        columns = {row["name"] for row in self.db.execute("PRAGMA table_info(jobs)")}
        if "release_confirmed" not in columns:
            # Legacy completed calls also require an idle check, never inference replay.
            with self.db:
                self.db.execute("ALTER TABLE jobs ADD COLUMN release_confirmed INTEGER NOT NULL DEFAULT 0")

    def start(self, update_id, chat_id, text):
        with self.db:
            cursor = self.db.execute(
                "INSERT OR IGNORE INTO jobs(update_id, chat_id, user_text, status, created) "
                "VALUES (?, ?, ?, 'starting', ?)",
                (update_id, chat_id, text, time.time()),
            )
        return bool(cursor.rowcount)

    def set_call(self, update_id, call_id):
        with self.db:
            self.db.execute(
                "UPDATE jobs SET call_id=?, status='running' WHERE update_id=?", (call_id, update_id)
            )

    def mark(self, update_id, status, error=None):
        with self.db:
            self.db.execute("UPDATE jobs SET status=?, error=? WHERE update_id=?", (status, error, update_id))

    def complete(self, update_id, result):
        with self.db:
            row = self.db.execute("SELECT * FROM jobs WHERE update_id=?", (update_id,)).fetchone()
            if row["status"] != "running":
                raise RuntimeError("Cancelled/non-running result cannot enter history")
            self.db.executemany(
                "INSERT INTO messages(update_id, chat_id, role, content) VALUES (?, ?, ?, ?)",
                [
                    (update_id, row["chat_id"], "user", row["user_text"]),
                    (update_id, row["chat_id"], "assistant", result["text"]),
                ],
            )
            evidence = {k: v for k, v in result.items() if k != "text"}
            self.db.execute(
                "UPDATE jobs SET status='completed', evidence=? WHERE update_id=?",
                (json.dumps(evidence, ensure_ascii=False), update_id),
            )

    def confirm_release(self, update_id, status=None, error=None):
        # Call only after the exact function's idle check succeeds. History is independent.
        with self.db:
            self.db.execute(
                "UPDATE jobs SET release_confirmed=1, status=COALESCE(?, status), error=? WHERE update_id=?",
                (status, error, update_id),
            )

    def request_completed_release(self):
        # Reuse the existing error field to durably distinguish an explicit /stop
        # from ordinary completed calls whose GPU is intentionally still warm.
        with self.db:
            self.db.execute(
                "UPDATE jobs SET error='gpu_release_requested' "
                "WHERE status='completed' AND release_confirmed=0"
            )

    def confirm_completed_releases(self):
        # Only call after the exact inference pool is observed at zero.
        with self.db:
            self.db.execute(
                "UPDATE jobs SET release_confirmed=1, error=NULL "
                "WHERE status='completed' AND release_confirmed=0"
            )

    def history(self, chat_id):
        rows = self.db.execute(
            "SELECT role, content FROM (SELECT id, role, content FROM messages "
            "WHERE chat_id=? ORDER BY id DESC LIMIT 40) ORDER BY id",
            (chat_id,),
        ).fetchall()
        return [dict(row) for row in rows]

    def pending(self):
        return [
            dict(row)
            for row in self.db.execute(
                "SELECT * FROM jobs WHERE status IN ('starting', 'running', 'cancel_requested') "
                "OR (status='completed' AND release_confirmed=0 AND error='gpu_release_requested')"
            )
        ]

    def last_job(self):
        row = self.db.execute("SELECT * FROM jobs ORDER BY created DESC LIMIT 1").fetchone()
        return dict(row) if row else None
