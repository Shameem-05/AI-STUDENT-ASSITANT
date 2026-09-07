"""
tools.py
--------
Implements the TOOLS capability of the agent.

Tools are structured, deterministic functions the agent calls when a question
needs an exact lookup rather than fuzzy document retrieval -- e.g. "when is my
exam", "any new notices", "what's the fee deadline". This mirrors how a real
deployment would call out to the college's Student Information System (SIS)
APIs, a notices database, a fees microservice, etc.

Each tool:
  - takes plain keyword arguments
  - returns a small dict (never raw prose) so the agent layer decides how to
    phrase the final answer to the student
  - is registered in TOOL_REGISTRY with a name, description and a simple
    trigger-keyword list, which the agent's router uses to decide whether a
    tool applies to a given user query (in a production system this routing
    would instead be done by an LLM's native function/tool-calling).
"""

from datetime import date

# ---------------------------------------------------------------------------
# Mock backend data (stand-ins for real Student Information System APIs)
# ---------------------------------------------------------------------------

EXAM_SCHEDULE = {
    "CS301": {"exam_date": "2026-09-16", "time": "10:00 AM - 1:00 PM", "room": "Block A - 204"},
    "CS402": {"exam_date": "2026-09-18", "time": "2:00 PM - 5:00 PM", "room": "Block B - 101"},
    "AI501": {"exam_date": "2026-09-20", "time": "10:00 AM - 1:00 PM", "room": "Block C - 305"},
}

FEE_STATUS = {
    "default": {
        "tuition_due": "2026-09-20",
        "amount": "Rs. 45,000",
        "late_fine_after_due": "Rs. 500 per week",
        "hostel_maintenance_due": "2026-09-25",
    }
}

NOTICE_KEYWORDS = {
    "exam": "NOTICE #101: Mid-semester exams run 2026-09-15 to 2026-09-22. Timetable published 2026-09-08.",
    "library": "NOTICE #102: Library open 8AM-11PM weekdays, 9AM-6PM weekends until exams end.",
    "placement": "NOTICE #103: TNSDC Besant Campus Connect placement drive - registration closes 2026-09-10.",
    "fee": "NOTICE #104: Tuition fee due 2026-09-20, late fine Rs.500/week after that.",
    "lecture": "NOTICE #105: Guest lecture 'Building Agentic AI Systems on the Cloud' on 2026-09-12, 2PM, Main Auditorium.",
    "hostel": "NOTICE #106: Hostel quarterly maintenance fee due 2026-09-25.",
    "sports": "NOTICE #107: Annual Sports Day postponed from 2026-09-10 to 2026-09-24.",
}


# ---------------------------------------------------------------------------
# Tool functions
# ---------------------------------------------------------------------------

def get_exam_schedule(course_code: str) -> dict:
    """Look up the exam date/time/room for a given course code, e.g. 'CS301'."""
    code = course_code.strip().upper()
    record = EXAM_SCHEDULE.get(code)
    if not record:
        return {"found": False, "course_code": code,
                "message": f"No exam schedule found for course code '{code}'."}
    return {"found": True, "course_code": code, **record}


def get_fee_status(student_id: str = "default") -> dict:
    """Look up tuition/hostel fee due dates and amounts."""
    record = FEE_STATUS.get(student_id, FEE_STATUS["default"])
    return {"found": True, **record}


def check_notices(keyword: str) -> dict:
    """Search the live notice board for a keyword (e.g. 'exam', 'hostel', 'placement')."""
    key = keyword.strip().lower()
    matches = [v for k, v in NOTICE_KEYWORDS.items() if key in k or k in key]
    if not matches:
        # fall back to substring search across all notice text
        matches = [v for v in NOTICE_KEYWORDS.values() if key in v.lower()]
    return {"found": bool(matches), "keyword": key, "notices": matches}


def get_today() -> dict:
    """Return today's date, useful for the agent to reason about deadlines."""
    return {"today": date.today().isoformat()}


# ---------------------------------------------------------------------------
# Registry used by the agent's router (agent.py)
# ---------------------------------------------------------------------------

TOOL_REGISTRY = {
    "get_exam_schedule": {
        "fn": get_exam_schedule,
        "description": "Get exam date, time and room for a course code (e.g. CS301, CS402, AI501).",
        "trigger_keywords": ["exam date", "exam schedule", "when is my exam", "exam time", "exam room"],
    },
    "get_fee_status": {
        "fn": get_fee_status,
        "description": "Get tuition/hostel fee due dates, amounts and late fine policy.",
        "trigger_keywords": ["fee", "fees", "tuition", "due date", "late fine", "payment deadline"],
    },
    "check_notices": {
        "fn": check_notices,
        "description": "Search the live college notice board for a topic keyword.",
        "trigger_keywords": ["notice", "notices", "announcement", "circular"],
    },
    "get_today": {
        "fn": get_today,
        "description": "Get today's date.",
        "trigger_keywords": ["today's date", "what is the date", "current date"],
    },
}
