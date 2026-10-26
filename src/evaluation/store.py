"""Isolated SQLite store for evaluation runs (data/evaluation.db) + aggregation for the charts.
Completely separate from the chat audit database in src/db.py."""
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from src.evaluation.judge import METRIC_KEYS

DB_PATH = Path(__file__).resolve().parents[2] / "data" / "evaluation.db"
LOWER_IS_BETTER = {"hallucination_rate"}
SCORED = [k for k in METRIC_KEYS if k != "hallucination_rate"]  # hallucination_rate mirrors faithfulness
BINS = 5
TIE = 0.01


def _conn():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(DB_PATH)
    c.execute("""CREATE TABLE IF NOT EXISTS eval_runs (
        id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT NOT NULL,
        question TEXT NOT NULL, payload_json TEXT NOT NULL)""")
    return c


def winners(naive: dict, selfrag: dict) -> dict:
    out = {}
    for k in METRIC_KEYS:
        a, b = naive.get(k), selfrag.get(k)
        if a is None or b is None:
            out[k] = None
        elif abs(a - b) < TIE:
            out[k] = "tie"
        else:
            self_better = (b < a) if k in LOWER_IS_BETTER else (b > a)
            out[k] = "self_rag" if self_better else "naive"
    return out


def save_run(result: dict) -> int:
    created = datetime.now(timezone.utc).isoformat()
    result["created_at"] = created
    with _conn() as c:
        cur = c.execute("INSERT INTO eval_runs (created_at, question, payload_json) VALUES (?,?,?)",
                        (created, result["question"], json.dumps(result, ensure_ascii=False)))
        result["id"] = cur.lastrowid
    return result["id"]


def get_run(run_id: int) -> Optional[dict]:
    with _conn() as c:
        row = c.execute("SELECT id, payload_json FROM eval_runs WHERE id=?", (run_id,)).fetchone()
    if not row:
        return None
    d = json.loads(row[1])
    d["id"] = row[0]
    return d


def list_runs(limit: int = 50) -> list:
    """Compact rows (no answers/claims) for the history table."""
    with _conn() as c:
        rows = c.execute("SELECT id, payload_json FROM eval_runs ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    out = []
    for rid, pj in rows:
        d = json.loads(pj)
        out.append({
            "id": rid, "created_at": d["created_at"], "question": d["question"], "winners": d.get("winners", {}),
            "systems": {k: {"metrics": v["metrics"], "latency_s": v["latency_s"]} for k, v in d["systems"].items()},
        })
    return out


def clear_runs() -> None:
    with _conn() as c:
        c.execute("DELETE FROM eval_runs")


def _bin(v: float) -> int:
    return min(int(v * BINS), BINS - 1)


def summary(limit: int = 500) -> dict:
    runs = list_runs(limit)
    sysk = ["naive", "self_rag"]
    vals = {s: {k: [] for k in METRIC_KEYS} for s in sysk}
    lat = {s: [] for s in sysk}
    wins = {"naive": 0, "self_rag": 0, "tie": 0}
    for r in runs:
        for s in sysk:
            sd = r["systems"].get(s)
            if not sd:
                continue
            for k in METRIC_KEYS:
                v = sd["metrics"].get(k)
                if v is not None:
                    vals[s][k].append(v)
            if sd.get("latency_s"):
                lat[s].append(sd["latency_s"])
        for k in SCORED:
            w = r["winners"].get(k)
            if w:
                wins[w] += 1
    avg = lambda xs: (sum(xs) / len(xs)) if xs else None  # noqa: E731
    averages = {s: {k: avg(vals[s][k]) for k in METRIC_KEYS} for s in sysk}
    histograms = {}
    for k in METRIC_KEYS:
        histograms[k] = {s: [sum(1 for v in vals[s][k] if _bin(v) == b) for b in range(BINS)] for s in sysk}
    hn, hs = averages["naive"]["hallucination_rate"], averages["self_rag"]["hallucination_rate"]
    reduction = None
    if hn is not None and hs is not None and hn > 0:
        reduction = (hn - hs) / hn
    return {
        "n_runs": len(runs),
        "averages": averages,
        "avg_latency_s": {s: avg(lat[s]) for s in sysk},
        "wins": wins,
        "histograms": histograms,
        "bins": BINS,
        "hallucination_reduction": reduction,   # relative reduction of avg hallucination rate, Self-RAG vs Naive
    }
