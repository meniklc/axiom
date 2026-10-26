from fastapi.testclient import TestClient
from tests.fakes import install


def _client(monkeypatch, tmp_path):
    install(monkeypatch, tmp_path)
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "audit.db"))
    import app as app_module
    return TestClient(app_module.app)


def test_existing_routes_still_work(monkeypatch, tmp_path):
    c = _client(monkeypatch, tmp_path)
    assert c.get("/").status_code == 200
    assert c.get("/api/health").json()["status"] == "ok"
    assert "/evaluation" in c.get("/").text          # nav link present


def test_eval_run_compares_both_systems(monkeypatch, tmp_path):
    c = _client(monkeypatch, tmp_path)
    assert c.get("/evaluation").status_code == 200
    r = c.post("/api/eval/run", json={"question": "Checkout API returns 502 after deploy, what do I check?"})
    assert r.status_code == 200, r.text
    d = r.json()
    n, s = d["systems"]["naive"], d["systems"]["self_rag"]
    # naive invented a step -> hallucination; self-rag grounded
    assert n["metrics"]["hallucination_rate"] > 0
    assert s["metrics"]["hallucination_rate"] == 0
    assert s["metrics"]["faithfulness"] == 1.0
    # naive saw all 3 chunks (1 irrelevant); self-rag graded it away
    assert n["metrics"]["context_precision"] < 1.0
    assert s["metrics"]["context_precision"] == 1.0
    assert len(s["chunks"]) == 2 and len(n["chunks"]) == 3
    assert d["winners"]["faithfulness"] == "self_rag" and d["winners"]["hallucination_rate"] == "self_rag"
    assert s["details"]["kept_after_grading"] == 2

    assert len(c.get("/api/eval/history").json()) == 1
    assert c.get(f"/api/eval/runs/{d['id']}").json()["question"] == d["question"]
    c.post("/api/eval/run", json={"question": "Second question about 502 errors"})
    sm = c.get("/api/eval/summary").json()
    assert sm["n_runs"] == 2 and sm["hallucination_reduction"] == 1.0
    assert sum(sm["histograms"]["hallucination_rate"]["self_rag"]) == 2
    assert c.delete("/api/eval/history").status_code == 200
    assert c.get("/api/eval/summary").json()["n_runs"] == 0


def test_validation(monkeypatch, tmp_path):
    c = _client(monkeypatch, tmp_path)
    assert c.post("/api/eval/run", json={"question": "x"}).status_code == 422
    assert c.get("/api/eval/runs/999").status_code == 404
