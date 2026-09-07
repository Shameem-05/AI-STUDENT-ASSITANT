"""
memory.py
---------
Implements the MEMORY capability of the agent, now backed by SQLite (db.py)
so conversation history and the student profile survive between runs --
important on a shared or low-end device where the app gets closed and
reopened between sessions.

Each ConversationMemory is scoped to a session_id (e.g. a student's roll
number) so multiple students can use the same installation without mixing
up each other's history.

Two kinds of memory, same as before:
1. Short-term (conversation) memory -- rolling chat history for this session.
2. Long-term (profile) memory -- department/semester/roll number, persisted
   in SQLite so the agent doesn't have to re-ask across sessions.
"""

from dataclasses import dataclass
from typing import Optional

import db


@dataclass
class Turn:
    role: str          # "user" or "assistant"
    content: str
    timestamp: str = ""


class ConversationMemory:
    """Holds short-term dialogue history and long-term student profile facts, persisted to SQLite."""

    def __init__(self, session_id: str = "default", max_turns: int = 20):
        self.session_id = session_id
        self.max_turns = max_turns
        db.init_db()
        rows = db.load_history(session_id, limit=max_turns)
        self.history: list[Turn] = [
            Turn(role=r["role"], content=r["content"], timestamp=r["timestamp"]) for r in rows
        ]
        self.profile: dict = db.load_profile(session_id)

    # ---------- short-term memory ----------

    def add_user_turn(self, text: str) -> None:
        self.history.append(Turn(role="user", content=text))
        db.save_turn(self.session_id, "user", text)
        self._trim()

    def add_assistant_turn(self, text: str) -> None:
        self.history.append(Turn(role="assistant", content=text))
        db.save_turn(self.session_id, "assistant", text)
        self._trim()

    def _trim(self) -> None:
        if len(self.history) > self.max_turns:
            self.history = self.history[-self.max_turns:]

    def recent_history_text(self, n: int = 6) -> str:
        recent = self.history[-n:]
        return "\n".join(f"{t.role.upper()}: {t.content}" for t in recent)

    def last_user_message(self) -> Optional[str]:
        for t in reversed(self.history):
            if t.role == "user":
                return t.content
        return None

    # ---------- long-term / profile memory ----------

    def update_profile(self, **kwargs) -> None:
        changed = False
        for k, v in kwargs.items():
            if v is not None and self.profile.get(k) != v:
                self.profile[k] = v
                changed = True
        if changed:
            db.save_profile(self.session_id, self.profile)

    def profile_text(self) -> str:
        known = {k: v for k, v in self.profile.items() if v}
        if not known:
            return "No student profile information known yet."
        return ", ".join(f"{k}={v}" for k, v in known.items())

    # ---------- lightweight auto-extraction ----------

    def auto_extract_profile(self, text: str) -> None:
        """
        Heuristic extraction so memory works without a full NER model, e.g.
        "I am in semester 4" -> semester=4. Swap for an LLM function-call or
        NER model in a larger deployment.
        """
        import re

        sem_match = re.search(r"semester\s*(\d)", text, re.IGNORECASE)
        if sem_match:
            self.update_profile(semester=sem_match.group(1))

        dept_match = re.search(r"\b(CSE|IT|ECE|EEE|MECH|CIVIL|CS|AI|DS)\b", text, re.IGNORECASE)
        if dept_match:
            self.update_profile(department=dept_match.group(1).upper())

        roll_match = re.search(r"roll\s*(?:no\.?|number)?\s*[:\-]?\s*(\w+)", text, re.IGNORECASE)
        if roll_match:
            self.update_profile(roll_number=roll_match.group(1))

    def reset(self) -> None:
        db.clear_session(self.session_id)
        self.history = []
        self.profile = {"department": None, "semester": None, "roll_number": None}
