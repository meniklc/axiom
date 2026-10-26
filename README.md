# Axiom - SelfRAG Copilot

A production-style **Self-RAG** for operations and incident response. It searches private operational knowledge in Pinecone first, self-grades the retrieved evidence, corrects weak retrieval/generation, and uses **internet search only when the private knowledge base is insufficient**.

The project also demonstrates **LangGraph SQLite persistence memory** so follow-up questions can reuse the same incident context through a stable `thread_id`.

## Architecture

```text
User Incident / Follow-up
        ↓
LangGraph SQLite Memory
        ↓
Contextualize Follow-up
        ↓
Decide Retrieval
        ↓
Private Pinecone Knowledge Base
        ↓
Grade Retrieved Documents
   ┌────┴───────────────┐
Relevant             Weak / Missing
   ↓                      ↓
Generate              Rewrite Query
   ↓                      ↓
IsSUP              Retry Private KB
   ↓                      ↓
Revise if needed   Internet Search Fallback
   ↓                      ↓
IsUSE              Grade Web Evidence
   ↓                      ↓
Final Answer ← Generate → IsSUP → IsUSE
        ↓
SQLite Checkpoint / Memory
```

## Tech stack

- **LangGraph** — Self-RAG workflow, conditional routing, persistent thread state
- **Groq `openai/gpt-oss-120b`** — routing, query rewriting, generation, relevance grading, IsSUP and IsUSE
- **Hugging Face `BAAI/bge-small-en-v1.5`** — embeddings for private operational documents
- **Pinecone** — private runbooks/SOPs/postmortems knowledge base
- **Tavily** — controlled internet-search fallback
- **SQLite** — LangGraph persistence memory + simple audit database for the demo
- **FastAPI** — application/API backend
- **HTML/CSS/JavaScript** — incident-command-center UI with document upload
- **Docker** — deployment packaging

## Project structure

```text
CloudOps-Sentinel-Enterprise-Incident-Response-Self-RAG-Copilot/
├── app.py
├── data_ingestion.py          
├── src/
│   ├── config.py
│   ├── db.py
│   ├── ingestion.py
│   ├── models.py
│   ├── self_rag.py
│   └── vectorstore.py
├── documents/                
│   ├── checkout-api-runbook.md
│   ├── payments-high-cpu-runbook.md
│   └── deployment-rollback-sop.md
├── templates/index.html
├── static/styles.css
├── static/app.js
├── uploads/                  
├── data/                      
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── .env.example
```

# Key Features

Self-RAG workflow with retrieval and generation verification

Private-first retrieval using Pinecone

Evidence grading before generating answers

IsSUP — verifies whether the answer is supported by evidence

IsUSE — evaluates answer usefulness

Query rewriting and retrieval retry for weak results

Tavily web fallback when private knowledge is insufficient

LangGraph SQLite memory for contextual follow-up questions

Document upload for adding new knowledge to the system

FastAPI backend with a simple web interface

LangSmith tracing for workflow observability

Activate venv ->  .\.venv\Scripts\Activate.ps1


## Evaluation tab: Self-RAG vs Naive RAG

Open **Evaluation** in the left nav (or `/evaluation`). It is a separate page from the chat console, with its own routes, files and SQLite database (`data/evaluation.db`).

For **any question you type**, both systems answer it (Naive RAG = same Pinecone index, top-k chunks stuffed into one prompt, no grading or verification; Self-RAG = the existing LangGraph pipeline, run on a throw-away thread). An independent LLM judge then scores both with the same rubric:

| Group | Metric |
|---|---|
| Hallucination | Faithfulness, Hallucination rate (unsupported claims / total claims) |
| Retrieval | Context precision, Context sufficiency |
| Answer | Answer relevancy, Answer completeness |

Charts: per-query grouped bars, claim breakdown, scorecard, then across all runs: average scores, a histogram (pick any metric), head-to-head wins and the latency trade-off. **Run sample benchmark** evaluates 8 built-in questions to fill the charts quickly.

Optional: set `EVAL_JUDGE_MODEL` to judge with a different Groq model than the one generating answers. Tests (no API keys needed): `pytest tests`.

Code: `src/evaluation/` (pipelines, judge, store, api), `templates/evaluation.html`, `static/eval.js`, `static/eval.css`.
