# AI Student Support Assistant
### TNSDC – IBM Agentic AI Campus Connect — Use Case #1
### Local edition: Ollama + SQLite, zero pip dependencies

An agent that answers college-related questions (regulations, syllabus, FAQs,
notices) using **RAG + Tools + Memory** — generated locally by **Ollama**,
with conversation history persisted in **SQLite**. Built to run comfortably
on low-end devices: the core app has no third-party pip dependencies at all.

```
Student question
      │
      ▼
 ┌─────────────┐   1. log turn to SQLite, auto-extract profile facts
 │   MEMORY    │◄──────────────────────────────────────────────────┐
 │  (SQLite)   │                                                    │
 └─────────────┘                                                    │
      │                                                              │
      ▼                                                              │
 ┌─────────────┐   keyword match against          ┌──────────────────┴──┐
 │   ROUTER    │──────────────────────────────────▶│  is it a live/       │
 └─────────────┘                                    │  structured data     │
      │  no tool matched, or tool found nothing     │  question?           │
      ▼                                             └───────────────────────┘
 ┌─────────────┐                                            │ yes
 │     RAG     │◄───────────────────────────────────────────┘ no
 │ (pure-Python│                                              ▼
 │  TF-IDF)    │                                      ┌──────────────┐
 └─────────────┘                                      │    TOOLS     │
      │ retrieved                                      │ (exam sched, │
      │ passages                                       │  fees,       │
      ▼                                                │  notices)    │
 ┌─────────────────────────────────────────────────────┴─────────────┐
 │                    Ollama (local model)                            │
 │        falls back to watsonx / OpenAI / offline template           │
 └──────────────────────────────────────────────────────────────────┘
      │
      ▼
  Final answer  →  logged back into SQLite
```

## What each capability is, concretely

| Capability | Where | What it does |
|---|---|---|
| **RAG** | `rag/retriever.py` | Pure-Python TF-IDF + cosine similarity over `data/*.txt` — no scikit-learn/numpy |
| **Tools** | `tools/tools.py` | Deterministic lookups a real SIS/notice-board API would provide: `get_exam_schedule`, `get_fee_status`, `check_notices`, `get_today` |
| **Memory** | `memory/memory.py` + `db.py` | Chat history + student profile (semester, department, roll no.), persisted in SQLite (`assistant.db`), keyed by session id |
| **Agent** | `agent.py` | Routes each question to a tool and/or RAG, merges context + memory, calls the answer generator |
| **Answer generation** | `llm/llm_client.py` | Local **Ollama** first, then optional watsonx/OpenAI if configured, then an offline template — always produces an answer |

## Quick start

```bash
cd ai-student-assistant
python main.py
```

That's it — `pip install -r requirements.txt` is a no-op by design; the
core app only uses the Python standard library (`sqlite3`, `urllib`, `re`,
`math`, `json`). It will run and answer questions even with no model
installed (using the offline template), so you can try it immediately.

### Enabling local AI answers with Ollama

1. Install Ollama: https://ollama.com (Windows/macOS/Linux)
2. Start the server: `ollama serve` (or it may already be running as a service)
3. Pull a small model suitable for low-end hardware:

   ```bash
   ollama pull llama3.2:1b      # ~1.3GB, good balance for a laptop
   # or, for very low-end / low-RAM devices:
   ollama pull qwen2.5:0.5b     # ~400MB, fastest, still usable
   # or, if the device can spare more RAM for better answers:
   ollama pull phi3:mini        # ~2.3GB, noticeably better quality
   ```
4. Run the app as normal — it auto-detects Ollama on startup and tells you
   which model it's using:

   ```bash
   python main.py
   ```

   If you pulled a different model, either edit the default in
   `llm/llm_client.py` or set an environment variable:

   ```bash
   export OLLAMA_MODEL="qwen2.5:0.5b"
   python main.py
   ```

No Ollama running? The app still works — it just answers from the
retrieved documents/tool results directly (offline template), which is
often perfectly readable for FAQ-style answers.

### Example session

```
Session id (e.g. your roll number, Enter for 'default'): 21CS045

[ok] Connected to Ollama -- using model 'llama3.2:1b'.

You: What is the minimum attendance percentage required for exams?
Assistant: You need at least 75% attendance in each course to be eligible
for the end-semester exam. If you're between 65-74%, you can apply for
condonation through your Head of Department with a valid reason.

You: When is my CS402 exam and which room?
Assistant: Your CS402 exam is on 2026-09-18, 2:00 PM - 5:00 PM, in Block B - 101.

You: I am in semester 4, CSE department. What's my backlog limit?
Assistant: You can carry a maximum of 3 backlog subjects at a time...

You: /exit
Goodbye!
```

Close the terminal and run `python main.py --session 21CS045` again later —
the full conversation and profile are still there, loaded from SQLite.

### Scripted demo (no typing needed)

```bash
python tests/demo.py
```

Runs 7 questions covering RAG, Tools, and Memory, then reloads the session
from SQLite to prove persistence actually works.

## Project structure

```
ai-student-assistant/
├── main.py                 # CLI chat loop, session handling, Ollama status check
├── agent.py                # orchestrator: routing + RAG + tools + memory
├── db.py                   # SQLite persistence (history + profile)
├── requirements.txt        # empty by design — see file for optional extras
├── data/                   # knowledge base for RAG
│   ├── regulations.txt
│   ├── syllabus.txt
│   ├── faq.txt
│   └── notices.txt
├── rag/
│   └── retriever.py         # pure-Python TF-IDF chunking + retrieval
├── tools/
│   └── tools.py               # exam schedule, fees, notices, date lookups
├── memory/
│   └── memory.py               # conversation history + student profile (SQLite-backed)
├── llm/
│   └── llm_client.py             # Ollama / watsonx / OpenAI / offline answer generation
└── tests/
    └── demo.py                    # scripted end-to-end conversation + persistence check
```

`assistant.db` is created automatically on first run, next to `db.py`.
Delete it any time to wipe all sessions' history and start fresh.

## Multiple students on one shared device

Every session is isolated by `session_id`. On a shared lab computer:

```bash
python main.py --session 21CS045
python main.py --session 21IT012
```

Each student's history and profile stay separate in the same `assistant.db`
file. `/reset` inside the chat only clears the *current* session.

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `OLLAMA_HOST` | `http://localhost:11434` | Where Ollama is running |
| `OLLAMA_MODEL` | `llama3.2:1b` | Which pulled model to use |
| `WATSONX_API_KEY` / `WATSONX_PROJECT_ID` / `WATSONX_URL` | — | Optional: use IBM watsonx instead |
| `OPENAI_API_KEY` / `OPENAI_MODEL` / `OPENAI_BASE_URL` | — | Optional: use an OpenAI-compatible API instead |

Backend priority is always: **Ollama → watsonx → OpenAI → offline template**.

## Extending to real vector embeddings

`rag/retriever.py` uses pure-Python TF-IDF — fast, dependency-free, fully
offline, and good enough for a keyword-rich domain like this. To upgrade to
neural embeddings for better semantic matching on a more capable device:

1. `pip install sentence-transformers faiss-cpu`
2. In `Retriever._build_index`, replace the TF-IDF vectors with
   `SentenceTransformer("all-MiniLM-L6-v2").encode(corpus)` and build a
   `faiss.IndexFlatIP` index.
3. In `Retriever.query`, embed the question the same way and use
   `index.search()` instead of the dict-based cosine similarity.

The rest of the agent (`agent.py`, `tools.py`, `memory.py`, `llm_client.py`)
does not need to change — `Retriever.query()` keeps the same return shape.

## Mapping to the assignment brief

- **Use case #1 (AI Student Support Assistant)** — answers college-related
  questions from regulations, syllabus, FAQs, and notices ✔
- **Key agent capabilities: RAG + Tools + Memory** — all three implemented,
  running on a local Ollama model, with memory persisted in SQLite ✔
- **Runs on low-end devices** — zero required pip dependencies, no compiled
  packages, and a lightweight local model option (`qwen2.5:0.5b`, ~400MB) ✔
