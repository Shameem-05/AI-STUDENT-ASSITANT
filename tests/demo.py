"""
demo.py
-------
Non-interactive demo that runs a scripted conversation through the agent to
show RAG, Tools, and Memory all working together, and confirm SQLite
persistence. Uses a dedicated 'demo_session' and resets it first so the
output is repeatable each time you run this.

Run with:  python tests/demo.py   (from the project root)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import StudentSupportAgent

SESSION_ID = "demo_session"

SCRIPTED_QUESTIONS = [
    "What is the minimum attendance percentage required for exams?",   # RAG: regulations
    "What topics are covered in the Cloud Computing Fundamentals course?",  # RAG: syllabus
    "When is my CS402 exam and which room?",                            # TOOL: exam schedule
    "Are there any notices about placements?",                          # TOOL: notices
    "When is the tuition fee due and is there a late fine?",            # TOOL: fees
    "I am in semester 4, CSE department. What's my backlog limit?",     # MEMORY: profile extraction
    "What did I just tell you about my department?",                    # MEMORY: recall
]


def run_demo():
    agent = StudentSupportAgent(session_id=SESSION_ID)
    agent.memory.reset()  # start clean each run for repeatable output

    print("Knowledge base stats:", agent.retriever.stats())
    print("=" * 70)

    for i, q in enumerate(SCRIPTED_QUESTIONS, 1):
        print(f"\n[{i}] STUDENT: {q}")
        answer = agent.handle(q)
        print(f"    ASSISTANT: {answer}")

    print("\n" + "=" * 70)
    print("Final remembered profile:", agent.memory.profile)

    # confirm persistence: reload a fresh agent for the same session and
    # check the history/profile survived
    reloaded = StudentSupportAgent(session_id=SESSION_ID)
    print(f"Reloaded from SQLite -> {len(reloaded.memory.history)} turns, profile: {reloaded.memory.profile}")


if __name__ == "__main__":
    run_demo()
