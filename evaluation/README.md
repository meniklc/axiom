# Axiom quick evaluation (30 questions)

This evaluation compares:

- `traditional_rag`: one internal Pinecone retrieval followed by one Groq answer-generation call.
- `complete_axiom`: the existing Axiom Self-RAG graph, including its current routing, relevance grading, retries, optional Tavily fallback, support checking and usefulness checking.

This is an end-to-end system comparison. Since complete Axiom includes web fallback and multiple checks, any difference cannot be attributed to Self-RAG alone. Do not claim causal attribution without an ablation experiment.

## 1. Prerequisites

- Run commands from the repository root (`Axiom-main` after extracting the zip).
- `.env` must contain working `GROQ_API_KEY`, `PINECONE_API_KEY`, and the same Pinecone index/namespace already populated with the three runbooks. `TAVILY_API_KEY` is optional but affects full Axiom's web fallback.
- Confirm `GROQ_MODEL` is a model available to your Groq account.
- Keep the same index, namespace, model, and document corpus for both versions.

## 2. Install and ingest

```bash
pip install -r requirements.txt
python data_ingestion.py
```

Only ingest if the target Pinecone index/namespace is the intended development evaluation namespace. Re-ingestion may add duplicate chunks if metadata/IDs differ from previously indexed records.

## 3. Run the 30-question experiment

```bash
python -m evaluation.run_evaluation
```

The runner runs each question through both versions and writes 60 rows to `evaluation/results/results.csv`. Each Axiom question receives a unique LangGraph thread ID so one question's conversation state does not contaminate another. The runner records provider-reported total tokens only when the installed LangChain/Groq callback exposes them; blank token fields mean usage was not reported, not zero tokens.

If API rate limits occur, wait and rerun. A rerun overwrites `results.csv`; copy the file first if you need to preserve partial results. Failed calls are retained as rows with an `error` field and should not be counted as successful measurements.

## 4. Score answer quality manually

Open `evaluation/results/results.csv` in Excel or LibreOffice. For every answer row, fill the following columns with 0, 1, or 2:

- `manual_correctness_0_2`: 0 = incorrect/misses key safe steps; 1 = partly correct; 2 = all essential expected points correct and no unsafe contradiction.
- `manual_faithfulness_0_2`: 0 = important factual claims unsupported by the retrieved evidence; 1 = mixed support; 2 = factual claims are supported by the sources/evidence available to that version.
- `manual_citation_accuracy_0_2`: 0 = cited/identified sources do not support the answer; 1 = partly correct; 2 = the identified sources support the key claims. If the answer gives no source attribution, score 0 or 1 consistently and state that rule in the report.

Use `expected_points` and `expected_sources` as your answer key. Ideally, ask your guide or another reviewer to score a sample without seeing which version generated it. Do not treat Axiom's internal support-check status as ground truth; report it separately.

## 5. Calculate summary metrics

After saving the manually scored CSV:

```bash
python -m evaluation.metrics
```

This writes `evaluation/results/summary.json`. Human-score percentages are calculated as mean 0-2 score divided by 2 times 100. Blank manual scores are excluded, so score every successful answer before reporting aggregate results.

## 6. Metrics and limitations

- `mean_source_precision_pct` and `mean_source_recall_pct` compare returned internal document filenames with expected relevant internal source filenames. `all_sources_including_web` separately records sources from web fallback. They are document-level source metrics, not chunk-level IR precision/recall.
- Mean and P95 latency are measured end-to-end for each version, including LLM calls and any optional web fallback. Run serially to reduce load-related variability.
- `mean_reported_tokens` is populated only when provider token usage is exposed by the installed callback. Do not replace blanks with estimated values.
- Manual correctness, faithfulness, and citation accuracy are the primary quality metrics. The 30 questions are a small, hand-authored set based on three sample runbooks; describe results as an initial evaluation, not a general guarantee.
- Run the experiment once end-to-end; if time permits, repeat the same set and report the run count. Do not tune prompts against the same questions and then present those same questions as an independent test set.
- The dataset is runbook-grounded. It does not comprehensively evaluate unanswerable questions, prompt injection, production incidents, or live web search accuracy.
