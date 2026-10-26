"""Reference-free, RAGAS-style metrics computed with an LLM judge.

Because users can ask ANY question, there is no gold answer, so every metric is computed from
(question, answer, evidence) alone. The same judge, prompts and rubric are applied to BOTH systems.
The judge is deliberately independent of Self-RAG's internal IsSUP / IsUSE signals.

Metrics (all 0..1 unless noted):
  faithfulness         supported claims / total claims in the answer            (higher = better)
  hallucination_rate   unsupported claims / total claims  (= 1 - faithfulness)  (LOWER  = better)
  context_precision    share of the evidence chunks given to the generator that are relevant
  context_sufficiency  does the evidence contain what is needed to answer? 1 / 0.5 / 0
  answer_relevancy     does the answer address the question asked?  (1-5 scaled to 0-1)
  answer_completeness  does it cover every part of the question?    (1-5 scaled to 0-1)
"""
import time
from typing import List, Literal, Optional

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from src.evaluation.llm import get_judge_llm

CHUNK_CHARS = 2500


# Flat schemas only (lists of ints/strings) - robust with Groq json_schema structured output.
class ClaimList(BaseModel):
    claims: List[str] = Field(default_factory=list)


class ClaimVerification(BaseModel):
    supported_indices: List[int] = Field(default_factory=list)
    unsupported_notes: List[str] = Field(default_factory=list)


class ContextJudgement(BaseModel):
    relevant_indices: List[int] = Field(default_factory=list)
    sufficiency: Literal["sufficient", "partial", "insufficient"]
    note: str = ""


class AnswerJudgement(BaseModel):
    relevancy: int
    completeness: int
    note: str = ""


import re, threading
_judge_lock = threading.Semaphore(1)   # one judge call at a time

def _structured(schema, messages, attempts: int = 6):
    last = None
    for i in range(attempts):
        try:
            with _judge_lock:
                return get_judge_llm().with_structured_output(schema, method="json_schema").invoke(messages)
        except Exception as e:  # noqa: BLE001
            last = e
            m = re.search(r"try again in ([\d.]+)s", str(e))
            time.sleep(float(m.group(1)) + 1 if m else 5 * (i + 1))
    raise last


def _numbered(chunks: List[dict]) -> str:
    return "\n\n".join(f"[{i}] ({c.get('title','')})\n{c['text'][:CHUNK_CHARS]}" for i, c in enumerate(chunks, 1))


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


# ----------------------------------------------------------------------------- hallucination
def extract_claims(answer: str) -> List[str]:
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Break the answer into atomic, self-contained FACTUAL claims (max 15), each understandable "
                   "on its own. Include concrete facts, steps, commands, thresholds and causal statements. "
                   "Skip greetings, formatting, hedges, and statements that merely say information is unavailable."),
        ("human", "Answer:\n{answer}"),
    ])
    out = _structured(ClaimList, prompt.format_messages(answer=answer))
    return [c.strip() for c in out.claims if c and c.strip()][:15]


def verify_claims(claims: List[str], chunks: List[dict]) -> List[dict]:
    """Return [{'text', 'supported'}]. A claim is supported ONLY if the evidence states or directly entails it."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a strict fact-checker. For each numbered claim decide whether the numbered evidence "
                   "explicitly states it or directly entails it. Plausible-sounding general knowledge that is NOT in "
                   "the evidence counts as UNSUPPORTED (that is a hallucination). Return the 1-based numbers of "
                   "supported claims in supported_indices, and a short note per unsupported claim."),
        ("human", "Claims:\n{claims}\n\nEvidence:\n{evidence}"),
    ])
    claim_txt = "\n".join(f"{i}. {c}" for i, c in enumerate(claims, 1))
    out = _structured(ClaimVerification, prompt.format_messages(claims=claim_txt, evidence=_numbered(chunks)))
    ok = {i for i in out.supported_indices if 1 <= i <= len(claims)}
    notes = list(out.unsupported_notes)
    res, ni = [], 0
    for i, c in enumerate(claims, 1):
        item = {"text": c, "supported": i in ok}
        if i not in ok:
            item["note"] = notes[ni] if ni < len(notes) else ""
            ni += 1
        res.append(item)
    return res


def judge_hallucination(answer: str, chunks: List[dict], abstained: bool, no_retrieval: bool) -> dict:
    """faithfulness / hallucination_rate plus the claim-level breakdown."""
    empty = {"faithfulness": None, "hallucination_rate": None, "claims": [], "note": ""}
    if abstained:
        return {**empty, "note": "Abstained - made no claims (nothing to hallucinate)."}
    if no_retrieval or not chunks:
        return {**empty, "note": "Answered without evidence - grounding cannot be measured."}
    claims = extract_claims(answer)
    if not claims:
        return {**empty, "note": "No factual claims found in the answer."}
    verdicts = verify_claims(claims, chunks)
    sup = sum(1 for v in verdicts if v["supported"])
    f = sup / len(verdicts)
    return {"faithfulness": f, "hallucination_rate": 1 - f, "claims": verdicts, "note": ""}


# ----------------------------------------------------------------------------- retrieval quality
def judge_context(question: str, chunks: List[dict]) -> dict:
    if not chunks:
        return {"context_precision": None, "context_sufficiency": 0.0, "chunk_relevance": [], "note": "No evidence was provided to the generator."}
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Judge retrieved evidence for a question. (1) List in relevant_indices the 1-based numbers of "
                   "chunks that contain information useful for answering the question; unrelated chunks are NOT "
                   "relevant. (2) Rate sufficiency: 'sufficient' if the chunks together contain what is needed to "
                   "answer fully, 'partial' if only some of it, 'insufficient' if little or none. Add a one-line note."),
        ("human", "Question:\n{question}\n\nRetrieved chunks:\n{chunks}"),
    ])
    out = _structured(ContextJudgement, prompt.format_messages(question=question, chunks=_numbered(chunks)))
    rel = {i for i in out.relevant_indices if 1 <= i <= len(chunks)}
    return {
        "context_precision": len(rel) / len(chunks),
        "context_sufficiency": {"sufficient": 1.0, "partial": 0.5, "insufficient": 0.0}[out.sufficiency],
        "chunk_relevance": [(i in rel) for i in range(1, len(chunks) + 1)],
        "note": out.note,
    }


# ----------------------------------------------------------------------------- answer quality
def judge_answer(question: str, answer: str) -> dict:
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Score an answer to a question on two 1-5 integer scales. relevancy: 5 = directly and fully on "
                   "topic, 1 = off-topic or evasive. completeness: 5 = addresses every part of the question with "
                   "actionable detail, 1 = misses most of it. A clear 'I could not find this' refusal is relevant "
                   "but NOT complete (completeness 1). Do not judge factual accuracy. Add a one-line note."),
        ("human", "Question:\n{question}\n\nAnswer:\n{answer}"),
    ])
    out = _structured(AnswerJudgement, prompt.format_messages(question=question, answer=answer or "(empty)"))
    scale = lambda v: _clamp01((max(1, min(5, int(v))) - 1) / 4)  # noqa: E731
    return {"answer_relevancy": scale(out.relevancy), "answer_completeness": scale(out.completeness), "note": out.note}


METRIC_KEYS = ["faithfulness", "hallucination_rate", "context_precision",
               "context_sufficiency", "answer_relevancy", "answer_completeness"]


def evaluate_system(question: str, run) -> dict:
    """Compute all metrics for one SystemRun. Each judge call fails independently -> that metric is None."""
    from concurrent.futures import ThreadPoolExecutor
    errors: List[str] = []
    metrics = {k: None for k in METRIC_KEYS}
    out = {"claims": [], "chunk_relevance": [], "notes": {}, "errors": errors}
    if run.error:
        errors.append(run.error)
        return {"metrics": metrics, **out}

    def safe(fn, *a):
        try:
            return fn(*a)
        except Exception as e:  # noqa: BLE001
            errors.append(f"{fn.__name__}: {type(e).__name__}: {e}")
            return None

    with ThreadPoolExecutor(max_workers=3) as ex:
        fh = ex.submit(safe, judge_hallucination, run.answer, run.chunks, run.abstained, run.no_retrieval)
        fc = ex.submit(safe, judge_context, question, run.chunks)
        fa = ex.submit(safe, judge_answer, question, run.answer)
        h, c, a = fh.result(), fc.result(), fa.result()
    if h:
        metrics["faithfulness"], metrics["hallucination_rate"] = h["faithfulness"], h["hallucination_rate"]
        out["claims"], out["notes"]["hallucination"] = h["claims"], h["note"]
    if c:
        metrics["context_precision"], metrics["context_sufficiency"] = c["context_precision"], c["context_sufficiency"]
        out["chunk_relevance"], out["notes"]["context"] = c["chunk_relevance"], c["note"]
    if a:
        metrics["answer_relevancy"], metrics["answer_completeness"] = a["answer_relevancy"], a["answer_completeness"]
        out["notes"]["answer"] = a["note"]
    return {"metrics": metrics, **out}
