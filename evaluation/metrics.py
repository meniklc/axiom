"""Summarize evaluation results. Run after manually scoring results.csv."""
import csv, json, statistics
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "evaluation" / "results" / "results.csv"
OUT_PATH = ROOT / "evaluation" / "results" / "summary.json"
SCORE_FIELDS = {
    "answer_correctness_pct": "manual_correctness_0_2",
    "faithfulness_pct": "manual_faithfulness_0_2",
    "citation_accuracy_pct": "manual_citation_accuracy_0_2",
}

def avg(vals):
    return round(statistics.mean(vals), 2) if vals else None

def main():
    if not CSV_PATH.exists():
        raise SystemExit("results.csv not found. Run python -m evaluation.run_evaluation first.")
    with CSV_PATH.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    summary = {"note": "Manual scores are on a 0-2 scale; percentage = mean score / 2 * 100. Blank scores are excluded.", "versions": {}}
    for version in ("traditional_rag", "complete_axiom"):
        subset = [r for r in rows if r.get("version") == version and not r.get("error")]
        item = {"successful_questions": len(subset), "failed_rows": sum(1 for r in rows if r.get("version") == version and r.get("error"))}
        for field, label in (("source_precision", "mean_source_precision_pct"), ("source_recall", "mean_source_recall_pct")):
            vals = [float(r[field]) for r in subset if r.get(field) not in (None, "")]
            item[label] = round(avg(vals) * 100, 2) if vals else None
        lat = [float(r["latency_ms"]) for r in subset if r.get("latency_ms") not in (None, "")]
        item["mean_latency_ms"] = avg(lat)
        item["p95_latency_ms"] = round(sorted(lat)[min(len(lat)-1, int(0.95 * len(lat)))], 2) if lat else None
        toks = [float(r["reported_total_tokens"]) for r in subset if r.get("reported_total_tokens") not in (None, "")]
        item["mean_reported_tokens"] = avg(toks)
        for out_key, field in SCORE_FIELDS.items():
            vals = [float(r[field]) for r in subset if r.get(field) not in (None, "")]
            item[out_key] = round(avg(vals) / 2 * 100, 2) if vals else None
            item[out_key.replace("_pct", "_scored_n")] = len(vals)
        statuses = [r.get("support_status", "") for r in subset]
        item["axiom_fully_supported_count"] = statuses.count("fully_supported") if version == "complete_axiom" else None
        item["axiom_partially_or_unsupported_count"] = sum(x in ("partially_supported", "no_support") for x in statuses) if version == "complete_axiom" else None
        summary["versions"][version] = item
    OUT_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"\nSaved: {OUT_PATH}")
if __name__ == "__main__": main()
