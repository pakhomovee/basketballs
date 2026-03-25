"""SQLite database for job state and comments."""

from __future__ import annotations

import aiosqlite
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "storage" / "db.sqlite3"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    status TEXT NOT NULL DEFAULT 'queued',
    video_name TEXT NOT NULL,
    created_at TEXT NOT NULL,
    error TEXT
);

CREATE TABLE IF NOT EXISTS comments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL REFERENCES jobs(id),
    timestamp_sec REAL NOT NULL,
    text TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""


async def get_db() -> aiosqlite.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    db = await aiosqlite.connect(str(DB_PATH))
    db.row_factory = aiosqlite.Row
    await db.executescript(_SCHEMA)
    return db


async def create_job(db: aiosqlite.Connection, job_id: str, video_name: str, created_at: str) -> None:
    await db.execute(
        "INSERT INTO jobs (id, status, video_name, created_at) VALUES (?, 'queued', ?, ?)",
        (job_id, video_name, created_at),
    )
    await db.commit()


async def update_job_status(db: aiosqlite.Connection, job_id: str, status: str, error: str | None = None) -> None:
    await db.execute("UPDATE jobs SET status = ?, error = ? WHERE id = ?", (status, error, job_id))
    await db.commit()


async def get_job(db: aiosqlite.Connection, job_id: str) -> dict | None:
    cursor = await db.execute("SELECT * FROM jobs WHERE id = ?", (job_id,))
    row = await cursor.fetchone()
    if row is None:
        return None
    return dict(row)


async def list_jobs(db: aiosqlite.Connection) -> list[dict]:
    cursor = await db.execute("SELECT * FROM jobs ORDER BY created_at DESC")
    return [dict(row) for row in await cursor.fetchall()]


async def add_comment(
    db: aiosqlite.Connection,
    job_id: str,
    timestamp_sec: float,
    text: str,
    created_at: str,
) -> int:
    cursor = await db.execute(
        "INSERT INTO comments (job_id, timestamp_sec, text, created_at) VALUES (?, ?, ?, ?)",
        (job_id, timestamp_sec, text, created_at),
    )
    await db.commit()
    return cursor.lastrowid  # type: ignore[return-value]


async def get_comments(db: aiosqlite.Connection, job_id: str) -> list[dict]:
    cursor = await db.execute(
        "SELECT * FROM comments WHERE job_id = ? ORDER BY timestamp_sec",
        (job_id,),
    )
    return [dict(row) for row in await cursor.fetchall()]
