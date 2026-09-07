"""
agent.py
--------
The orchestrator that ties together the three required agent capabilities:

    RAG + TOOLS + MEMORY

Flow for every incoming student message:
  1. MEMORY   - log the user's turn; auto-extract any profile facts
                (semester, department, roll no.) mentioned in passing.
  2. ROUTING  - decide whether the question needs a TOOL (an exact,
                structured lookup like exam date / fee due date / notices)
                or plain document RAG (regulations / syllabus / FAQ), based
                on keyword matching against each tool's trigger list.
                (In a production system with a real LLM, this routing step
                would instead be done via the model's native function/tool
                calling -- see README.)
  3. RETRIEVE - if no tool matched (or in addition to it), run the RAG
                retriever over the knowledge base for supporting context.
  4. GENERATE - hand {question, tool result, retrieved context, memory,
                profile} to llm_client.generate_response() to produce the
                final answer.
  5. MEMORY   - log the assistant's turn.
"""

import os

from memory.memory import ConversationMemory
from tools.tools import TOOL_REGISTRY
from rag.retriever import Retriever
from llm import llm_client

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


class StudentSupportAgent:
    def __init__(self, session_id: str = "default"):
        self.memory = ConversationMemory(session_id=session_id)
        self.retriever = Retriever(data_dir=DATA_DIR)

    # ---------- routing ----------

    def _match_tool(self, question: str):
        """
        Lightweight keyword router. Returns (tool_name, tool_fn) or (None, None).

        Two passes:
          1. Strong structural signal: a course-code pattern (e.g. CS301) plus
             any exam-ish word -> get_exam_schedule, regardless of exact phrasing.
          2. Fallback: word-overlap scoring against each tool's trigger_keywords
             (counts individual keyword words present in the question, not
             whole-phrase substring matches, so "when is my CS402 exam" still
             matches "when is my exam" / "exam schedule" style triggers).
        """
        import re

        # generic connector words carry no signal about which tool applies,
        # and would otherwise cause false matches (e.g. "my", "what", "is")
        STOPWORDS = {
            "is", "the", "my", "what", "what's", "when", "do", "does", "how",
            "can", "i", "a", "an", "of", "for", "to", "in", "on", "at", "are",
            "there", "any", "and", "am",
        }

        q = question.lower()

        # Strong structural signal: a course-code pattern (e.g. CS301) next to
        # an exam-ish word -> get_exam_schedule regardless of exact phrasing.
        if re.search(r"\b[a-z]{2,4}\d{3}\b", q) and any(w in q for w in ("exam", "schedule", "room")):
            return "get_exam_schedule", TOOL_REGISTRY["get_exam_schedule"]["fn"]

        q_words = set(re.findall(r"[a-z']+", q)) - STOPWORDS

        best_tool = None
        best_hits = 0
        for name, spec in TOOL_REGISTRY.items():
            kw_words = set()
            for kw in spec["trigger_keywords"]:
                kw_words.update(kw.split())
            kw_words -= STOPWORDS
            hits = len(q_words & kw_words)
            if hits > best_hits:
                best_hits = hits
                best_tool = name
        if best_tool:
            return best_tool, TOOL_REGISTRY[best_tool]["fn"]
        return None, None

    def _extract_tool_arg(self, question: str, tool_name: str) -> dict:
        """Pull a plausible argument out of the question for the matched tool."""
        import re
        if tool_name == "get_exam_schedule":
            m = re.search(r"\b([A-Za-z]{2,4}\d{3})\b", question)
            return {"course_code": m.group(1) if m else "CS301"}
        if tool_name == "check_notices":
            for kw in ("exam", "library", "placement", "fee", "lecture", "hostel", "sports"):
                if kw in question.lower():
                    return {"keyword": kw}
            return {"keyword": question.split()[-1]}
        return {}

    _MEMORY_RECALL_PATTERNS = (
        r"what did i.*tell you", r"what do you know about me", r"my profile",
        r"what did i.*say", r"do you remember", r"what have i told you",
    )

    def _is_memory_recall(self, question: str) -> bool:
        import re
        q = question.lower()
        return any(re.search(p, q) for p in self._MEMORY_RECALL_PATTERNS)

    # ---------- main entry point ----------

    def handle(self, question: str) -> str:
        # 1. memory: log + extract profile facts
        self.memory.add_user_turn(question)
        self.memory.auto_extract_profile(question)

        # 1b. direct memory-recall questions are answered from profile/history
        # alone -- routing to RAG here would surface irrelevant documents,
        # since "what did I tell you" has no match in the knowledge base.
        if self._is_memory_recall(question):
            answer = f"Here's what I currently have noted about you: {self.memory.profile_text()}."
            self.memory.add_assistant_turn(answer)
            return answer

        # 2. routing: does a tool apply?
        tool_name, tool_fn = self._match_tool(question)
        tool_result = None
        if tool_fn:
            args = self._extract_tool_arg(question, tool_name)
            tool_result = tool_fn(**args)

        # 3. retrieve supporting document context (RAG)
        # Skip RAG only when a tool already fully answered a live-data question.
        context_chunks = []
        if not tool_result or tool_name == "check_notices":
            context_chunks = self.retriever.query(question, top_k=3)
        elif tool_result and not tool_result.get("found", True):
            # tool found nothing useful -> fall back to document search
            context_chunks = self.retriever.query(question, top_k=3)

        # 4. generate final answer
        answer = llm_client.generate_response(
            question=question,
            context_chunks=context_chunks,
            tool_result=tool_result,
            memory_text=self.memory.recent_history_text(n=6),
            profile_text=self.memory.profile_text(),
        )

        # 5. memory: log assistant turn
        self.memory.add_assistant_turn(answer)
        return answer

    def debug_info(self) -> dict:
        return {
            "retriever_stats": self.retriever.stats(),
            "profile": self.memory.profile,
            "num_turns": len(self.memory.history),
        }
