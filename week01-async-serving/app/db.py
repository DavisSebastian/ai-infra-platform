"""SQLite setup. Seeds a tiny `users` table so the endpoints have something to read."""

import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "users.db"

SEED_USERS = [
    (1, "Ada Lovelace", "ada@example.com"),
    (2, "Alan Turing", "alan@example.com"),
    (3, "Grace Hopper", "grace@example.com"),
    (4, "Linus Torvalds", "linus@example.com"),
    (5, "Barbara Liskov", "barbara@example.com"),
]


def init_db() -> None:
    # Runs once at startup, before the server accepts traffic, so a blocking call is fine here.
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, name TEXT NOT NULL, email TEXT NOT NULL)"
        )
        conn.executemany("INSERT OR REPLACE INTO users VALUES (?, ?, ?)", SEED_USERS)
