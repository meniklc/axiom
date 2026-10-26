"""Orchestrates one evaluation: run both systems on the question, judge both, persist, return."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict

from src.evaluation import store
from src.evaluation.judge import evaluate_system
from src.evaluation.pipelines import run_naive_rag, run_self_rag_detailed

SAMPLE_QUESTIONS = [
    # covered by the bundled runbooks
    "Our checkout API is returning 502 errors after deployment. What should the on-call engineer check first?",
    "CPU on the payments service is above 90%. What is our recommended response procedure?",
    "What must be verified before and after rolling back a production deployment?",
    "At what error rate and duration should we roll back checkout-api after a bad release?",
    # only partly covered / not covered - where naive RAG tends to fill gaps with invented detail
    "What is the on-call escalation path if the checkout API stays down for 30 minutes?",
    "What is the maximum replica count the payments autoscaler is allowed to reach?",
    "How do I rotate the database password for the payments service?",
    "What should I verify when a Kubernetes pod keeps restarting with CrashLoopBackOff?",
]


def _system_payload(question: str, run) -> dict:
    ev = evaluate_system(question, run)
    rel = ev["chunk_relevance"]
    chunks = [{"title": c["title"], "type": c["type"], "text": c["text"][:700],
               "relevant": rel[i] if i < len(rel) else None} for i, c in enumerate(run.chunks)]
    return {
        "label": run.label, "answer": run.answer, "route": run.route, "latency_s": run.latency_s,
        "abstained": run.abstained, "no_retrieval": run.no_retrieval,
        "metrics": ev["metrics"], "claims": ev["claims"], "notes": ev["notes"],
        "chunks": chunks, "details": run.details, "errors": ev["errors"],
    }


def evaluate_query(question: str) -> dict:
    question = question.strip()
    with ThreadPoolExecutor(max_workers=2) as ex:      # both systems answer concurrently
        fn, fs = ex.submit(run_naive_rag, question), ex.submit(run_self_rag_detailed, question)
        naive, selfrag = fn.result(), fs.result()
    if naive.error and selfrag.error:
        raise RuntimeError(f"Both pipelines failed. Naive: {naive.error} | Self-RAG: {selfrag.error}")
    with ThreadPoolExecutor(max_workers=2) as ex:      # judge both concurrently
        fn, fs = ex.submit(_system_payload, question, naive), ex.submit(_system_payload, question, selfrag)
        systems = {"naive": fn.result(), "self_rag": fs.result()}
    result = {"question": question, "systems": systems,
              "winners": store.winners(systems["naive"]["metrics"], systems["self_rag"]["metrics"])}
    store.save_run(result)
    return result
