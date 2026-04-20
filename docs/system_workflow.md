# Geo-Cultural Grounding Project Workflow

## 1. Project Goal

This project studies whether retrieval-augmented generation (RAG) can improve factual or culturally grounded reasoning, and whether retrieval may amplify or reduce biased outputs.

The current implementation focuses on a **general search-grounding system** rather than benchmark-specific pipelines.

The core research questions are:

1. Can search-grounding improve factual or culturally grounded QA?
2. Can the same retrieval pipeline reduce harmful bias or stereotype-related errors?
3. Are gains and harms dataset-dependent?

## 2. Top-Level Code Structure

### 2.1 Main execution entry

- `src/main.py`

This is the main runtime entry point.

Supported modes:

- `vanilla`
- `kb`
- `search`
- `all`

For the current experiments, the important modes are:

- `vanilla`: answer directly without retrieval
- `search`: run the general search-grounding pipeline

### 2.2 Main pipeline implementation

- `src/pipeline.py`

This file contains the main pipeline classes:

- `VanillaPipeline`
- `SearchPipeline` (legacy)
- `GeneralSearchPipeline` (current main system)
- `KBPipeline`

The current experiments use:

- `VanillaPipeline`
- `GeneralSearchPipeline`

### 2.3 Search and evidence construction

- `src/search_grounding.py`

This file is responsible for:

- sending web search queries
- collecting search hits
- deduplicating hits
- extracting page text
- chunking text into evidence candidates

### 2.4 Evaluation

- `src/eval.py`

This file implements task-aware evaluation:

- `mcq_accuracy`
- `short_answer_exact_match`
- `short_answer_containment_match`
- `bias_probe_metrics`
- `evaluate_rows`

### 2.5 Experiment runners

- `scripts/run_general_ablation.py`
  Used for the legacy geo-cultural benchmarks.

- `scripts/run_external_ablation.py`
  Used for the external benchmark ablations with task-specific metrics.

- `scripts/freeze_search_cache.py`
  Used to freeze retrieval outputs so that downstream system variants share the same retrieved evidence.

## 3. Current Unified System

All current datasets use the same **general search-grounding architecture**.

There is no benchmark-specific retrieval pipeline.

The only dataset-specific differences are:

1. input task format
2. output verbalization
3. evaluation metric

## 4. End-to-End Pipeline

The current general search-grounding pipeline is:

1. Read one benchmark item
2. Determine whether the item is:
   - MCQ
   - bias-probe MCQ
   - short QA
3. Build search queries
4. Run web search
5. Extract page text and build candidate chunks
6. Rank candidate chunks
7. Organize evidence
8. Gate evidence
9. Answer with or without evidence
10. Evaluate with dataset-appropriate metrics

## 5. Detailed Step-by-Step Workflow

### 5.1 Input loading

Code:

- `src/main.py`
- `src/pipeline.py`

Each row is loaded from a JSONL benchmark file.

Examples of supported row types:

#### MCQ

Used by:

- BLEnD
- NormAd
- SeeGULL
- BBQ
- TruthfulQA-Binary

Important fields:

- `question`
- `choices`
- `answer`

#### Bias probe MCQ

Used by:

- SocialStigmaQA

Important fields:

- `question`
- `choices`
- `biased_answer`
- `allowed_non_biased_answers`

#### Short QA

Used by:

- PopQA

Important fields:

- `question`
- `answers`

### 5.2 Vanilla branch

Code:

- `src/pipeline.py` -> `VanillaPipeline`

Flow:

1. Format the prompt directly from the original question
2. Ask the LLM to answer
3. For MCQ, verbalize to `A/B/C/...`
4. For short QA, normalize to a short answer string

This branch does not use retrieval.

### 5.3 Search-grounding branch

Code:

- `src/pipeline.py` -> `GeneralSearchPipeline`

This is the main research pipeline.

#### Step A: Query rewrite

Current design:

- simple natural-language rewrite
- no planner
- no critic

The system rewrites the benchmark question into more natural search queries.

The goal is:

- keep entities and relations
- remove benchmark artifacts
- produce queries that look like realistic search requests

Examples:

- factual QA:
  transform a benchmark-like question into a natural factual search

- stereotype/bias prompt:
  try to search for evidence relevant to the claim, not raw annotation wording

#### Step B: Web search

Code:

- `src/search_grounding.py`

The system:

1. submits queries to the search backend
2. retrieves web results
3. deduplicates URLs
4. keeps a limited number of pages per domain

#### Step C: Evidence extraction

The system:

1. downloads page text
2. falls back to snippets when full text is weak
3. chunks page content into candidate evidence blocks

Each candidate contains:

- title
- URL
- domain
- chunk text
- current score

#### Step D: Ranking

The ranking pipeline is:

1. lexical rank
2. embedding pre-rank
3. cross-encoder semantic rerank
4. optional domain/noise adjustments
5. URL diversification

This is still one unified pipeline, not benchmark-specific logic.

#### Step E: Noise filter

The system downweights or removes low-quality sources, such as:

- homework sites
- flashcard pages
- answer-key style content
- noisy educational aggregation pages

#### Step F: Evidence organize

Code:

- `src/pipeline.py` -> `_organize_evidence`

The system uses the LLM to compress top evidence into concise evidence notes.

Purpose:

- reduce prompt noise
- preserve only decision-relevant evidence
- improve final answer prompt quality

#### Step G: Evidence gate

Code:

- `src/pipeline.py` -> `_should_use_evidence`

The system decides whether the retrieved evidence is strong enough to use.

If evidence is weak, noisy, or not decision-relevant, it falls back to a non-augmented answer.

This gate is one reason the full system may behave conservatively on some datasets.

#### Step H: Final answer

Two possible outcomes:

1. `answer_augmented`
   The system answers using organized evidence.

2. `answer_fallback`
   The system ignores the retrieved evidence and answers like a vanilla model.

## 6. Current Ablation Systems

All current benchmark comparisons use the same 6-system ladder.

### 6.1 Vanilla

No retrieval baseline.

### 6.2 Simple RAG

Raw Query + Lexical

Meaning:

- no rewrite
- no semantic rerank
- no organize
- no gate

### 6.3 Simple Rewrite + Lexical

Meaning:

- simple natural-language rewrite
- lexical ranking only

### 6.4 Simple Rewrite + Semantic Rerank

Meaning:

- rewrite
- embedding pre-rank
- cross-encoder rerank

### 6.5 Simple Rewrite + Semantic Rerank + Noise Filter

Meaning:

- rewrite
- semantic rerank
- source/noise filtering

### 6.6 Full General RAG

Meaning:

- rewrite
- semantic rerank
- noise filter
- evidence organize
- evidence gate

## 7. Why Retrieval Is Frozen

For the ablation experiments, systems that share the same retrieval setup reuse the same frozen cache.

Purpose:

1. keep retrieval constant
2. isolate downstream module effects
3. make ablation comparisons fair

Example:

- `planning_rag`
- `planning_semantic_rag`
- `planning_semantic_noise_filter`
- `full_general_rag`

all share the same `planning` retrieval cache within the same experiment folder.

## 8. Supported Benchmarks

### 8.1 Legacy geo-cultural benchmarks

Main balanced file:

- `data/eval_balanced_200_strict.jsonl`

Contains:

- BLEnD
- NormAd
- SeeGULL

Typical runner:

- `scripts/run_general_ablation.py`

### 8.2 External benchmarks

Processed benchmark files are under:

- `data/benchmarks/external/processed`

Sampled experimental files are under:

- `data/benchmarks/external/sampled`

Current external `200`-sample sets:

- `bbq_200.jsonl`
- `socialstigmaqa_200.jsonl`
- `truthfulqa_200.jsonl`
- `popqa_200.jsonl`

Typical runner:

- `scripts/run_external_ablation.py`

## 9. Evaluation Metrics

The system does **not** use one single metric for all datasets.

### 9.1 MCQ datasets

Used by:

- BLEnD
- NormAd
- SeeGULL
- BBQ
- TruthfulQA-Binary

Metric:

- `accuracy`

### 9.2 Bias probe datasets

Used by:

- SocialStigmaQA

Metrics:

- `bias_rate`
- `non_biased_rate`
- `valid_rate`

Primary metric for comparison:

- `non_biased_rate`

### 9.3 Short QA datasets

Used by:

- PopQA

Metrics:

- `exact_match`
- `containment_match`

Primary metric for comparison:

- `containment_match`

## 10. Important Result Directories

### 10.1 Legacy balanced benchmark ablation

- `outputs/general_ablation/balanced200_20260405`

### 10.2 External benchmark ablations

- `outputs/external_ablation/bbq_200_20260406`
- `outputs/external_ablation/socialstigmaqa_200_20260406`
- `outputs/external_ablation/truthfulqa_200_20260406`
- `outputs/external_ablation/popqa_200_20260407`

Each directory contains:

- `configs/`
- `caches/`
- `runs/`
- `analysis/ablation_summary.json`
- `analysis/primary_metric_ablation.png`

## 11. Current High-Level Findings

### 11.1 Legacy benchmarks

The same general RAG pipeline does not uniformly improve all legacy benchmarks.

Observed pattern:

- some factual/cultural datasets show limited or mixed gains
- stereotype-style datasets such as SeeGULL remain difficult

### 11.2 External benchmarks

#### BBQ

The best system is not the full system.

The best result is:

- `Simple Rewrite + Lexical`

#### SocialStigmaQA

RAG does not hurt this benchmark in the current setup.

Best systems:

- `Simple RAG`
- `Full General RAG`

#### TruthfulQA-Binary

RAG improves over weaker retrieval variants, but current full RAG still does not beat vanilla.

#### PopQA

RAG clearly helps.

The best result is achieved by:

- `Simple Rewrite + Semantic Rerank + Noise Filter`

## 12. What Is Shared Across All Datasets

Shared components:

- same `GeneralSearchPipeline`
- same retrieval backend
- same ranking stack
- same noise filtering logic
- same evidence organization logic
- same evidence gate logic
- same model family

Therefore, the current study is testing a **general system**, not a benchmark-specific system.

## 13. What Differs Across Datasets

Only these parts differ:

1. benchmark file contents
2. task type
3. output verbalization
4. evaluation metric

The retrieval-and-answering architecture remains the same.

## 14. How to Reproduce the Main Experiments

### 14.1 Legacy balanced benchmark

```powershell
conda run -n RAG python scripts/run_general_ablation.py --config configs/config_openai_general_rag_balanced200.yaml --out-root outputs/general_ablation --tag balanced200_YYYYMMDD
```

### 14.2 External benchmark ablation

Example for BBQ:

```powershell
conda run -n RAG python scripts/run_external_ablation.py --config configs/external/config_bbq_200.yaml --out-root outputs/external_ablation --tag bbq_200_YYYYMMDD
```

Example for PopQA:

```powershell
conda run -n RAG python scripts/run_external_ablation.py --config configs/external/config_popqa_200.yaml --out-root outputs/external_ablation --tag popqa_200_YYYYMMDD
```

## 15. Current Limitations

1. The full system can be overly conservative because evidence gate may reject too many retrieved contexts.
2. Some datasets benefit more from simple retrieval than from the full organize+gate stack.
3. Open web evidence quality is highly variable.
4. Strong base models sometimes already know the answer, limiting the marginal gain from retrieval.
5. Different factual benchmarks measure different capabilities, so “RAG helps factuality” is not universally true.

## 16. Recommended Next Steps

1. Create unified cross-dataset summary plots combining legacy and external benchmarks.
2. Run case studies on:
   - `vanilla-only wins`
   - `RAG-only wins`
   - `gate-rejected but relevant evidence`
3. Analyze whether evidence gate is too conservative for some factual datasets.
4. Compare general RAG against a gate-off ablation without changing the retrieval pipeline.

