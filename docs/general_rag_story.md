# General RAG Experiment Storyline (Search Grounding)

## Goal
Build a stable, general search-grounding system without benchmark-specific tuning, then evaluate whether adding retrieved web context changes model behavior.

## Step 1: Fix the Baseline
- System: `vanilla` (same model, same prompts, no external evidence).
- Why: establish a clean reference before any retrieval component is introduced.
- Output: `vanilla_predictions.jsonl`, `llm_usage_vanilla.jsonl`.

## Step 2: Freeze Retrieval Once
- System: `freeze_search_cache.py` with `pipeline_variant=general`.
- Why: separate retrieval variance from generation variance. All compared runs consume the same frozen retrieval artifacts.
- Output: `search_cache.jsonl` (queries, hits, candidate chunks, selected evidence).

## Step 3: Run General Search Grounding
- System: `search_general`.
- Pipeline: `query -> search -> rank -> context -> LLM`.
- Policy: if evidence exists, concatenate evidence directly into prompt; otherwise fallback to vanilla prompt.
- Output: `search_predictions.jsonl`, `llm_usage_search.jsonl`.

## Step 4: Compare Fairly
- Compare only aligned IDs between `vanilla` and `search_general`.
- Metrics: overall accuracy, per-dataset accuracy, win/tie/loss, SeeGULL stereotype rate.
- Output: `analysis_summary.json`, dataset/overall plots.

## Step 5: Inspect Cost and Behavior
- Token analysis from usage logs (`visualize_token_usage.py`).
- Why: confirm whether performance changes are driven by evidence quality or just higher token budget.
- Output: `analysis/token_usage/*`.

## Step 6: Case Study for Failure Attribution
- Use frozen cache + predictions to inspect where errors arise:
  - query quality,
  - retrieval mismatch,
  - ranking mismatch,
  - context interference.
- Keep this diagnostic layer separate from the core pipeline to avoid hidden benchmark-specific logic in production runs.
