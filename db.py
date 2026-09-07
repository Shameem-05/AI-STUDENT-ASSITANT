"""
db.py
-----
SQLite persistence for conversation history and student profile.

Why SQLite: zero setup (no server, no daemon), a single file on disk,
built into the Python standard library -- ideal for a low-end device that
needs history to survive between runs without any extra install.

Everything is scoped by session_id so multiple students can share one
installation (e.g. a shared lab computer) without mixing up each other's
history -- pass a roll number or name as the session id.
"""

import sqlite3
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), "assistant.db")


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    conn = get_conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS turns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            timestamp TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS profile (
            session_id TEXT PRIMARY KEY,
            department TEXT,
            semester TEXT,
            roll_number TEXT,
            updated_at TEXT
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_turns_session ON turns(session_id)")
    conn.commit()
    conn.close()


def save_turn(session_id: str, role: str, content: str) -> None:
    conn = get_conn()
    conn.execute(
        "INSERT INTO turns (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
        (session_id, role, content, datetime.now().isoformat(timespec="seconds")),
    )
    conn.commit()
    conn.close()


def load_history(session_id: str, limit: int = 20) -> list[dict]:
    conn = get_conn()
    rows = conn.execute(
        "SELECT role, content, timestamp FROM turns WHERE session_id=? ORDER BY id DESC LIMIT ?",
        (session_id, limit),
    ).fetchall()
    conn.close()
    return [dict(r) for r in reversed(rows)]


def save_profile(session_id: str, profile: dict) -> None:
    conn = get_conn()
    conn.execute(
        """
        INSERT INTO profile (session_id, department, semester, roll_number, updated_at)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(session_id) DO UPDATE SET
            department=excluded.department,
            semester=excluded.semester,
            roll_number=excluded.roll_number,
            updated_at=excluded.updated_at
        """,
        (
            session_id,
            profile.get("department"),
            profile.get("semester"),
            profile.get("roll_number"),
            datetime.now().isoformat(timespec="seconds"),
        ),
    )
    conn.commit()
    conn.close()


def load_profile(session_id: str) -> dict:
    conn = get_conn()
    row = conn.execute(
        "SELECT department, semester, roll_number FROM profile WHERE session_id=?",
        (session_id,),
    ).fetchone()
    conn.close()
    if row:
        return {"department": row["department"], "semester": row["semester"], "roll_number": row["roll_number"]}
    return {"department": None, "semester": None, "roll_number": None}


def list_sessions() -> list[str]:
    conn = get_conn()
    rows = conn.execute("SELECT DISTINCT session_id FROM turns").fetchall()
    conn.close()
    return [r["session_id"] for r in rows]


def clear_session(session_id: str) -> None:
    conn = get_conn()
    conn.execute("DELETE FROM turns WHERE session_id=?", (session_id,))
    conn.execute("DELETE FROM profile WHERE session_id=?", (session_id,))
    conn.commit()
    conn.close()
