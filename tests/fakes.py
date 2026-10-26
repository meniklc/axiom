"""Offline fakes (no API keys needed) used to test the evaluation tab against the REAL Self-RAG graph."""
import re
from langchain_core.documents import Document

DOCS = [
    Document(page_content="Checkout API 502 after deployment: check readiness probe failures before restarting pods.",
             metadata={"title": "checkout-api-runbook.md", "source_type": "internal"}),
    Document(page_content="Rollback to the last known healthy image if the 502 error rate stays above 5% for five minutes.",
             metadata={"title": "checkout-api-runbook.md", "source_type": "internal"}),
    Document(page_content="Payments service high CPU: compare CPU across replicas and check queue depth.",
             metadata={"title": "payments-high-cpu-runbook.md", "source_type": "internal"}),
]


class _Msg:
    def __init__(self, content): self.content = content


def _text(messages):
    return "\n".join(getattr(m, "content", str(m)) for m in messages)


class FakeRetriever:
    def invoke(self, q): return [Document(page_content=d.page_content, metadata=dict(d.metadata)) for d in DOCS]


class _Structured:
    def __init__(self, schema): self.schema = schema

    def invoke(self, messages):
        t, n = _text(messages), self.schema.__name__
        if n == "RetrieveDecision": return self.schema(should_retrieve=True)
        if n == "RelevanceDecision": return self.schema(is_relevant=("Payments service" not in t.split("Document:")[-1]))
        if n == "QueryRewrite": return self.schema(query="checkout api 502 readiness probe rollback")
        if n == "SupportDecision": return self.schema(status="fully_supported", evidence=[])
        if n == "UsefulnessDecision": return self.schema(status="useful", reason="ok")
        if n == "ClaimList":
            ans = t.split("Answer:")[-1]
            return self.schema(claims=[s.strip() for s in re.split(r"(?<=\.)\s+", ans.strip()) if s.strip()])
        if n == "ClaimVerification":
            claims = re.findall(r"^\d+\. (.*)$", t.split("Evidence:")[0], flags=re.M)
            return self.schema(supported_indices=[i for i, c in enumerate(claims, 1) if "database primary" not in c],
                               unsupported_notes=["Not stated in the evidence"])
        if n == "ContextJudgement":
            chunks = re.findall(r"^\[(\d+)\]", t.split("Retrieved chunks:")[-1], flags=re.M)
            rel = [int(i) for i in chunks if int(i) <= 2]
            return self.schema(relevant_indices=rel, sufficiency="sufficient" if len(rel) == len(chunks) else "partial", note="fake")
        if n == "AnswerJudgement": return self.schema(relevancy=5, completeness=4, note="fake")
        raise AssertionError(f"unexpected schema {n}")


class FakeLLM:
    def with_structured_output(self, schema, method=None): return _Structured(schema)

    def invoke(self, messages):
        t = _text(messages)
        if "retrieved context" in t:      # naive RAG prompt: no grounding discipline -> adds an invented step
            return _Msg("Check readiness probes first. Roll back if errors persist. Restart the database primary.")
        return _Msg("Check readiness probes first. Roll back if errors stay above 5%.")


def install(monkeypatch, tmp_path):
    """Patch every external dependency (LLM, Pinecone) and isolate the SQLite files."""
    import src.self_rag as sr
    import src.evaluation.pipelines as pl
    import src.evaluation.judge as jd
    from src.evaluation import store
    monkeypatch.setattr(sr, "_llm", lambda: FakeLLM())
    monkeypatch.setattr(sr, "get_retriever", lambda: FakeRetriever())
    monkeypatch.setattr(pl, "get_retriever", lambda: FakeRetriever())
    monkeypatch.setattr(pl, "get_llm", lambda *a, **k: FakeLLM())
    monkeypatch.setattr(jd, "get_judge_llm", lambda: FakeLLM())
    monkeypatch.setattr(store, "DB_PATH", tmp_path / "evaluation.db")
    monkeypatch.setattr(sr, "_graph", None)
