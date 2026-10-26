"""Runs the two systems being compared on the same question.

* Naive RAG  - retrieve top-k from the same Pinecone index -> stuff ALL chunks into one prompt -> answer.
               No grading, no query rewriting, no verification, no fallback.
* Self-RAG   - the existing LangGraph pipeline from src/self_rag.py, run unchanged on a throw-away thread
               (so evaluation never pollutes a user's chat memory). We only read its final state.
"""
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import List, Optional

from langchain_core.prompts import ChatPromptTemplate

from src.evaluation.llm import get_llm
from src.vectorstore import get_retriever

_graph_lock = threading.Lock()


@dataclass
class SystemRun:
    key: str
    label: str
    answer: str = ""
    route: str = ""
    latency_s: float = 0.0
    chunks: List[dict] = field(default_factory=list)  # evidence the generator actually saw
    context_text: str = ""
    abstained: bool = False        # system declined to answer (no reliable evidence)
    no_retrieval: bool = False     # answered from general knowledge, no evidence to ground against
    details: dict = field(default_factory=dict)
    error: Optional[str] = None


def _chunk(doc) -> dict:
    m = doc.metadata or {}
    web = m.get("source_type") == "web"
    title = m.get("title") or m.get("document_name") or m.get("source") or "Evidence"
    page = m.get("page")
    if page is not None and not web:
        try:
            title += f" (p.{int(page) + 1})"
        except (TypeError, ValueError):
            pass
    return {"title": title, "type": "web" if web else "internal", "text": doc.page_content or ""}


def format_context(chunks: List[dict]) -> str:
    return "\n\n---\n\n".join(f"[{i}] {c['title']}\n{c['text']}" for i, c in enumerate(chunks, 1))


# --------------------------------------------------------------------------- Naive RAG
NAIVE_SYSTEM = (
    "You are a helpful assistant. Use the following retrieved context to answer the question. "
    "If the context does not contain the answer, say you don't know."
)


def run_naive_rag(question: str) -> SystemRun:
    run = SystemRun(key="naive", label="Naive RAG")
    t0 = time.perf_counter()
    try:
        docs = get_retriever().invoke(question)
        run.chunks = [_chunk(d) for d in docs]
        run.context_text = format_context(run.chunks)
        prompt = ChatPromptTemplate.from_messages([
            ("system", NAIVE_SYSTEM),
            ("human", "Context:\n{context}\n\nQuestion: {question}"),
        ])
        out = get_llm().invoke(prompt.format_messages(context=run.context_text, question=question))
        run.answer = out.content if isinstance(out.content, str) else str(out.content)
        run.route = "Private Runbooks (top-k, ungraded)"
        run.details = {"retrieved": len(docs), "rewrites": 0, "revisions": 0, "web_search": False}
    except Exception as e:  # noqa: BLE001 - surfaced to the UI, never crashes the comparison
        run.error = f"{type(e).__name__}: {e}"
    run.latency_s = round(time.perf_counter() - t0, 2)
    return run


# --------------------------------------------------------------------------- Self-RAG
def _get_graph():
    """Reuse the app's compiled graph (and its SQLite connection) exactly as run_self_rag() does."""
    from src import self_rag as sr
    with _graph_lock:
        if sr._graph is None:
            sr._graph = sr.build_graph()
        return sr._graph


def run_self_rag_detailed(question: str) -> SystemRun:
    run = SystemRun(key="self_rag", label="Self-RAG")
    t0 = time.perf_counter()
    try:
        graph = _get_graph()
        initial = {
            "user_question": question, "question": question, "memory": [],
            "retrieval_query": question, "web_query": "", "docs": [], "relevant_docs": [],
            "context": "", "answer": "", "support_status": "", "evidence": [],
            "usefulness": "", "use_reason": "", "support_retries": 0,
            "retrieval_rewrites": 0, "web_rewrites": 0, "source_mode": "internal",
            "used_web_search": False, "trace": [],
        }
        cfg = {"configurable": {"thread_id": f"eval-{uuid.uuid4()}"}, "recursion_limit": 60}
        res = graph.invoke(initial, config=cfg)
        mode = res.get("source_mode", "none")
        run.answer = res.get("answer", "")
        run.route = {"internal": "Private Runbooks", "web": "Internet Search",
                     "direct": "General Knowledge", "none": "No Reliable Evidence"}.get(mode, mode)
        run.abstained = mode == "none"
        run.no_retrieval = mode == "direct"
        if mode in ("internal", "web"):
            run.chunks = [_chunk(d) for d in res.get("relevant_docs", [])]
        run.context_text = format_context(run.chunks)
        run.details = {
            "retrieved": len(res.get("docs", []) or []),
            "kept_after_grading": len(res.get("relevant_docs", []) or []),
            "rewrites": int(res.get("retrieval_rewrites", 0)) + int(res.get("web_rewrites", 0)),
            "revisions": int(res.get("support_retries", 0)),
            "web_search": bool(res.get("used_web_search")),
            "is_sup": res.get("support_status", ""),
            "is_use": res.get("usefulness", ""),
            "trace": res.get("trace", []),
        }
    except Exception as e:  # noqa: BLE001
        run.error = f"{type(e).__name__}: {e}"
    run.latency_s = round(time.perf_counter() - t0, 2)
    return run
