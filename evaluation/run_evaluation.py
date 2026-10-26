"""Compare internal-only traditional RAG with the existing full Axiom workflow.

Run from the repository root: python -m evaluation.run_evaluation
"""
from __future__ import annotations
import csv, json, time, uuid
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from src.config import get_settings
from src.vectorstore import get_retriever
from src.self_rag import run_self_rag

ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "evaluation" / "dataset.json"
RESULTS_DIR = ROOT / "evaluation" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_CSV = RESULTS_DIR / "results.csv"

try:
    from langchain_core.callbacks import UsageMetadataCallbackHandler
except ImportError:
    UsageMetadataCallbackHandler = None


def token_total(callback) -> str:
    """Return provider-reported token count when the installed SDK exposes it."""
    if callback is None:
        return ""
    try:
        usage = callback.usage_metadata or {}
        total = sum(int(v.get("total_tokens", 0)) for v in usage.values())
        return str(total) if total else ""
    except Exception:
        return ""


def make_usage_callback():
    return UsageMetadataCallbackHandler() if UsageMetadataCallbackHandler else None


def source_name(doc_or_source: Any) -> str:
    if isinstance(doc_or_source, Document):
        meta = doc_or_source.metadata or {}
        raw = meta.get("document_name") or meta.get("title") or meta.get("source") or ""
    elif isinstance(doc_or_source, dict):
        raw = doc_or_source.get("title") or doc_or_source.get("source") or ""
    else:
        raw = str(doc_or_source or "")
    return Path(str(raw)).name.strip().lower()


def format_docs(docs: list[Document]) -> str:
    parts = []
    for i, doc in enumerate(docs, 1):
        name = source_name(doc) or f"source-{i}"
        parts.append(f"[SOURCE {i}: {name}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts) if parts else "No internal documents were retrieved."


def run_baseline(question: str) -> tuple[dict, str]:
    """Simple RAG: retrieve top-k internal chunks, then generate once."""
    callback = make_usage_callback()
    start = time.perf_counter()
    docs = get_retriever().invoke(question)
    s = get_settings()
    llm = ChatGroq(api_key=s.groq_api_key, model=s.groq_model, temperature=0)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an incident-response assistant. Answer using only the supplied internal runbook evidence. Do not invent infrastructure facts, commands, thresholds, or procedures. If evidence is insufficient, clearly say what is unknown. Give concise actionable steps and identify which source supports them."),
        ("human", "Question:\n{question}\n\nRetrieved evidence:\n{context}"),
    ])
    config = {"callbacks": [callback]} if callback else {}
    response = llm.invoke(prompt.format_messages(question=question, context=format_docs(docs)), config=config)
    latency_ms = (time.perf_counter() - start) * 1000
    answer = response.content if hasattr(response, "content") else str(response)
    found_sources = sorted({source_name(d) for d in docs if source_name(d)})
    result = {
        "answer": answer,
        "sources": found_sources,
        "all_sources": found_sources,
        "latency_ms": round(latency_ms, 2),
        "token_total": token_total(callback),
        "route": "Traditional RAG (internal retrieval only)",
        "support_status": "not_checked",
        "usefulness": "not_checked",
        "trace": "",
    }
    return result, format_docs(docs)


def run_axiom(question: str) -> tuple[dict, str]:
    """Run the current complete Self-RAG graph, with a unique thread per test."""
    callback = make_usage_callback()
    start = time.perf_counter()
    result = run_self_rag(question, thread_id=f"eval-{uuid.uuid4().hex}", callbacks=[callback] if callback else None)
    latency_ms = (time.perf_counter() - start) * 1000
    all_sources = sorted({source_name(s) for s in result.get("sources", []) if source_name(s)})
    # Compare internal retrieval against the runbook answer key. Keep web sources
    # separately so external search does not masquerade as an internal runbook hit.
    internal_sources = sorted({
        source_name(item) for item in result.get("sources", [])
        if item.get("type") != "web" and source_name(item)
    })
    return {
        "answer": result.get("answer", ""),
        "sources": internal_sources,
        "all_sources": all_sources,
        "latency_ms": round(latency_ms, 2),
        "token_total": token_total(callback),
        "route": result.get("route", ""),
        "support_status": result.get("support_status", ""),
        "usefulness": result.get("usefulness", ""),
        "trace": " | ".join(result.get("trace", [])),
    }, ""


def source_scores(expected: list[str], returned: list[str]) -> tuple[float, float]:
    exp = {Path(x).name.lower() for x in expected}
    got = {Path(x).name.lower() for x in returned}
    overlap = len(exp & got)
    recall = overlap / len(exp) if exp else 0.0
    precision = overlap / len(got) if got else 0.0
    return round(precision, 4), round(recall, 4)


def main():
    s = get_settings()
    if not s.groq_api_key:
        raise SystemExit("GROQ_API_KEY is missing. Add it to your .env file first.")
    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    rows = []
    for i, item in enumerate(dataset, 1):
        print(f"[{i}/{len(dataset)}] {item['id']}: {item['question']}", flush=True)
        for version, fn in (("traditional_rag", run_baseline), ("complete_axiom", run_axiom)):
            try:
                result, _context = fn(item["question"])
                precision, recall = source_scores(item["relevant_sources"], result["sources"])
                rows.append({
                    "question_id": item["id"],
                    "question": item["question"],
                    "version": version,
                    "expected_sources": ";".join(item["relevant_sources"]),
                    "returned_sources": ";".join(result["sources"]),
                    "all_sources_including_web": ";".join(result.get("all_sources", result["sources"])),
                    "source_precision": precision,
                    "source_recall": recall,
                    "expected_points": " | ".join(item["expected_points"]),
                    "answer": result["answer"],
                    "latency_ms": result["latency_ms"],
                    "reported_total_tokens": result["token_total"],
                    "route": result["route"],
                    "support_status": result["support_status"],
                    "usefulness": result["usefulness"],
                    "trace": result["trace"],
                    "manual_correctness_0_2": "",
                    "manual_faithfulness_0_2": "",
                    "manual_citation_accuracy_0_2": "",
                    "error": "",
                })
                print(f"    {version}: {result['latency_ms']:.0f} ms; source recall={recall:.2f}", flush=True)
            except Exception as exc:
                rows.append({
                    "question_id": item["id"], "question": item["question"], "version": version,
                    "expected_sources": ";".join(item["relevant_sources"]), "returned_sources": "", "all_sources_including_web": "",
                    "source_precision": 0, "source_recall": 0,
                    "expected_points": " | ".join(item["expected_points"]), "answer": "",
                    "latency_ms": "", "reported_total_tokens": "", "route": "", "support_status": "",
                    "usefulness": "", "trace": "", "manual_correctness_0_2": "",
                    "manual_faithfulness_0_2": "", "manual_citation_accuracy_0_2": "",
                    "error": f"{type(exc).__name__}: {exc}",
                })
                print(f"    ERROR {version}: {type(exc).__name__}: {exc}", flush=True)
    fields = list(rows[0].keys()) if rows else []
    with RESULTS_CSV.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    print(f"\nSaved {len(rows)} result rows to: {RESULTS_CSV}")
    print("Next: manually fill the three manual_* columns with 0, 1, or 2, then run: python -m evaluation.metrics")

if __name__ == "__main__":
    main()
