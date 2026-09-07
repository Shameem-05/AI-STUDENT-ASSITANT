"""
llm_client.py
-------------
Pluggable "brain" that turns (retrieved context + tool results + memory) into
a final natural-language answer.

Backends, tried automatically in this order:

  1. Local Ollama (default). Uses only Python's built-in urllib -- no pip
     install needed for this path at all. Requires `ollama serve` running
     locally with a model pulled (see README for low-end model picks).
     Configure with OLLAMA_HOST (default http://localhost:11434) and
     OLLAMA_MODEL (default llama3.2:1b).
  2. IBM watsonx.ai -- optional, only used if WATSONX_API_KEY/PROJECT_ID/URL
     are set (needs `pip install requests`).
  3. OpenAI-compatible API -- optional, only used if OPENAI_API_KEY is set
     (needs `pip install openai`).
  4. Offline template fallback -- zero dependencies, always works. Produces
     a clear, readable answer directly from retrieved context/tool output
     with no model call at all, so the app is always usable even with no
     internet and no local model installed.

agent.py never needs to know which backend actually answered -- swap
providers by editing only this file.
"""

import os
import json
import textwrap
import urllib.request
import urllib.error


SYSTEM_PROMPT = """You are the AI Student Support Assistant for a college.
Answer ONLY using the CONTEXT and TOOL RESULTS provided below -- do not invent
facts. If the answer isn't in the context or tool results, say you don't have
that information and suggest who the student should contact.
Be concise, warm, and structured (use short bullet points for multi-part answers).
Always take the student's profile / conversation history into account so you
don't ask for information you already know.
"""

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:1b")


def _build_prompt(question: str, context_chunks: list[dict], tool_result: dict | None,
                   memory_text: str, profile_text: str) -> str:
    context_block = "\n\n".join(
        f"[Source: {c['source']}] {c['text']}" for c in context_chunks
    ) or "(no relevant document context found)"

    tool_block = f"TOOL RESULT: {tool_result}" if tool_result else "TOOL RESULT: (no tool was called)"

    return textwrap.dedent(f"""
    STUDENT PROFILE: {profile_text}

    RECENT CONVERSATION:
    {memory_text}

    CONTEXT (retrieved from college documents):
    {context_block}

    {tool_block}

    STUDENT QUESTION: {question}

    Answer the student's question now, using only the information above.
    """).strip()


# ---------------------------------------------------------------------------
# Backend 1: local Ollama (stdlib urllib only -- no extra pip installs)
# ---------------------------------------------------------------------------

def _call_ollama(prompt: str) -> str | None:
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": f"{SYSTEM_PROMPT}\n\n{prompt}",
        "stream": False,
        # kept small/deterministic so it stays fast on low-end hardware
        "options": {"temperature": 0.3, "num_predict": 350},
    }
    try:
        req = urllib.request.Request(
            f"{OLLAMA_HOST}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            text = (data.get("response") or "").strip()
            return text or None
    except (urllib.error.URLError, ConnectionRefusedError, TimeoutError, OSError):
        # Ollama not running / not installed -- silently fall through to the
        # next backend so the app still works.
        return None
    except Exception as e:
        print(f"[llm_client] Ollama call failed, falling back: {e}")
        return None


# ---------------------------------------------------------------------------
# Backend 2: IBM watsonx.ai (optional)
# ---------------------------------------------------------------------------

def _call_watsonx(prompt: str) -> str | None:
    api_key = os.getenv("WATSONX_API_KEY")
    project_id = os.getenv("WATSONX_PROJECT_ID")
    url = os.getenv("WATSONX_URL")
    if not (api_key and project_id and url):
        return None
    try:
        import requests  # optional dependency, only imported if configured
        resp = requests.post(
            f"{url}/ml/v1/text/generation?version=2024-05-01",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model_id": "ibm/granite-13b-instruct-v2",
                "input": f"{SYSTEM_PROMPT}\n\n{prompt}",
                "project_id": project_id,
                "parameters": {"max_new_tokens": 400, "temperature": 0.3},
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["results"][0]["generated_text"].strip()
    except Exception as e:
        print(f"[llm_client] watsonx call failed, falling back: {e}")
        return None


# ---------------------------------------------------------------------------
# Backend 3: OpenAI-compatible API (optional)
# ---------------------------------------------------------------------------

def _call_openai(prompt: str) -> str | None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    try:
        from openai import OpenAI  # optional dependency, only imported if configured
        client = OpenAI(api_key=api_key, base_url=os.getenv("OPENAI_BASE_URL") or None)
        resp = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"[llm_client] OpenAI call failed, falling back: {e}")
        return None


# ---------------------------------------------------------------------------
# Backend 4: offline fallback (zero dependencies, always available)
# ---------------------------------------------------------------------------

def _offline_fallback(question: str, context_chunks: list[dict], tool_result: dict | None) -> str:
    lines = []

    if tool_result:
        lines.append("**From live college systems:**")
        for k, v in tool_result.items():
            if k in ("found",):
                continue
            if isinstance(v, list):
                if v:
                    lines.extend(f"- {item}" for item in v)
                else:
                    lines.append("- No matching entries found.")
            else:
                pretty_key = k.replace("_", " ").title()
                lines.append(f"- {pretty_key}: {v}")
        lines.append("")

    if context_chunks:
        lines.append("**From college documents:**")
        for c in context_chunks:
            snippet = c["text"] if len(c["text"]) < 400 else c["text"][:400].rsplit(".", 1)[0] + "."
            lines.append(f"- ({c['source']}) {snippet}")
    elif not tool_result:
        lines.append(
            "I couldn't find that in the college regulations, syllabus, FAQ, or notices. "
            "Please contact the Administrative Office or your Head of Department for help with this."
        )

    return "\n".join(lines).strip()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_response(question: str, context_chunks: list[dict], tool_result: dict | None,
                       memory_text: str, profile_text: str) -> str:
    """Tries Ollama, then watsonx, then OpenAI, then falls back offline."""
    prompt = _build_prompt(question, context_chunks, tool_result, memory_text, profile_text)

    for backend in (_call_ollama, _call_watsonx, _call_openai):
        result = backend(prompt)
        if result:
            return result

    return _offline_fallback(question, context_chunks, tool_result)
