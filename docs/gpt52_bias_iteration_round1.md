# GPT-5.2 Bias Iteration Round 1

## Scope

First optimization target:

- Model: `gpt-5.2`
- Dataset: `SeeGULL_100`
- Goal: reduce retrieval-induced bias and task mismatch on stereotype-label questions

This round intentionally uses `100` examples instead of `200` to shorten iteration time while keeping a fixed benchmark slice.

## Baseline

Reference result:

- [ablation_summary.json](/d/geo_cultural_grounding/outputs/fiveway_benchmark_suite_100/seegull_100/analysis/ablation_summary.json)

Current GPT-5.2 scores on `SeeGULL_100`:

| System | Accuracy |
| --- | ---: |
| Vanilla | 0.55 |
| Simple RAG | 0.37 |
| Rewrite | 0.35 |
| Rerank | 0.37 |
| Noise Filter | 0.35 |

Main problem:

- All search-based variants are substantially below `Vanilla`.
- The strongest observed issue is not ordinary retrieval weakness; it is retrieval drift on bias-sensitive label tasks.

## Case Analysis

Case analysis artifact:

- [report.md](/d/geo_cultural_grounding/outputs/bias_analysis/gpt52_seegull100_planning_rag/report.md)

Source predictions:

- [search_predictions.jsonl](/d/geo_cultural_grounding/outputs/fiveway_benchmark_suite_100/seegull_100/runs/planning_rag/search_predictions.jsonl)

Aggregate counts from the first heuristic pass:

| Tag | Count |
| --- | ---: |
| `wrong_with_grounding` | 65 |
| `query_artifact_leakage` | 62 |
| `rewrite_no_effect` | 60 |
| `default_unsure_prediction` | 54 |
| `weak_claim_alignment` | 19 |
| `claim_missing_from_evidence` | 19 |
| `lexical_hijack` | 16 |

Interpretation:

1. The dominant failure is **not** missing retrieval, but **wrong grounding**.
2. The rewrite stage often falls back to the original question, so benchmark annotation wording leaks into the search query.
3. Once the query contains artifact terms such as `dominant annotation label`, retrieval often drifts into:
   - annotation tutorials
   - homework / notes pages
   - unrelated lexical meanings of `dominant`
4. Many wrong cases end in `C / Unsure`, which suggests the system uses noisy evidence and then collapses to a generic abstention pattern.

## Round 1 Optimization

### Optimization A: fix `bias_safe` fallback

Code change:

- [pipeline.py](/d/geo_cultural_grounding/src/pipeline.py)

What changed:

1. For `bias_safe` rewrite policy, if LLM rewrite fails or produces no usable query, the system no longer falls back to the raw benchmark question.
2. Instead, it builds deterministic fallback queries from the extracted claim pair.

Reason:

- The old fallback reintroduced annotation wording into retrieval.
- That behavior is directly responsible for a large fraction of observed `SeeGULL` drift cases.

### Optimization B: test `bias_aware` pipeline on the same 100-item split

New config:

- [config_gpt52_seegull_100_biasaware_round1.yaml](/d/geo_cultural_grounding/configs/bias_iteration/config_gpt52_seegull_100_biasaware_round1.yaml)

What it changes:

- `search_pipeline_type: bias_aware`
- `enable_query_feedback_retry: true`
- `query_feedback_max_retry: 1`
- keeps evidence organization + evidence gate + strict feature checks

Reason:

- `SeeGULL` is a bias-sensitive stereotype-label task.
- The project already contains a dedicated `BiasAwareSearchPipeline`; this round explicitly tests it on the smaller fixed split.

## Server Rerun Commands

### A. Re-run the standard 5-way suite after the fallback fix

```bash
cd /dtu/p1/yanwen/bias-rag
module purge
module load python3/3.11.9
source .venv/bin/activate
export PYTHONPATH="$VIRTUAL_ENV/lib/python3.11/site-packages:${PYTHONPATH}"
source /dtu/p1/yanwen/bias-rag/.env.private

cd /dtu/p1/yanwen/bias-rag/project
python scripts/run_general_ablation.py \
  --config configs/legacy_100_ablation/config_seegull_100_ablation.yaml \
  --out-root outputs/gpt52_bias_iteration \
  --tag seegull_100_round1_fix \
  --provider openai \
  --model gpt-5.2 \
  --temperature 0.0
```

Expected output root:

- `outputs/gpt52_bias_iteration/seegull_100_round1_fix/`

### B. Run the dedicated bias-aware candidate

```bash
cd /dtu/p1/yanwen/bias-rag
module purge
module load python3/3.11.9
source .venv/bin/activate
export PYTHONPATH="$VIRTUAL_ENV/lib/python3.11/site-packages:${PYTHONPATH}"
source /dtu/p1/yanwen/bias-rag/.env.private

cd /dtu/p1/yanwen/bias-rag/project
python -m src.main --mode search --config configs/bias_iteration/config_gpt52_seegull_100_biasaware_round1.yaml
```

Expected output root:

- `outputs/bias_iteration/gpt52_seegull_100_biasaware_round1/`

## What To Record After Rerun

### Standard 5-way rerun

Check:

- `outputs/gpt52_bias_iteration/seegull_100_round1_fix/analysis/ablation_summary.json`

Primary comparison:

- `Vanilla`
- `planning_rag`
- `planning_semantic_rag`
- `planning_semantic_noise_filter`
- `full_general_rag`

### Bias-aware candidate

Check:

- `outputs/bias_iteration/gpt52_seegull_100_biasaware_round1/search_predictions.jsonl`
- `outputs/bias_iteration/gpt52_seegull_100_biasaware_round1/metrics.json`

## Success Criteria For Round 1

Minimum success:

- at least one search-based system reaches or exceeds `0.55` on `SeeGULL_100`

Secondary success:

- lower fraction of cases with:
  - raw-question fallback
  - annotation wording leakage
  - lexical hijack

## Next Step After Round 1

If the fallback fix helps but still remains below `Vanilla`, the next optimization should be:

1. add an explicit claim-alignment filter before answer generation for stereotype-label tasks
2. tighten evidence gate specifically when:
   - claim entity is absent from evidence
   - top evidence is tutorial / annotation / homework content
3. run the same loop on `BBQ`
