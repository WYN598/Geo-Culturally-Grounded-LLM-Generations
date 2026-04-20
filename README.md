# Geo-Cultural Grounding

This project is now focused on a single main line: a general search-based RAG system for studying the trade-off between knowledge gains and bias amplification.

## Main scope

- `vanilla`: direct answering without retrieval
- `search_general`: general search-grounded RAG
- benchmark evaluation on `BLEnD`, `NormAd`, and `SeeGULL`
- layered diagnostics and case studies

The current research goal is not benchmark-specific tuning. The goal is to compare progressively stronger general RAG designs and study whether bias can be reduced without hurting knowledge-focused tasks.

## Main files

- [src/main.py](D:/geo_cultural_grounding/src/main.py): experiment entrypoint
- [src/pipeline.py](D:/geo_cultural_grounding/src/pipeline.py): `VanillaPipeline` and `GeneralSearchPipeline`
- [src/search_grounding.py](D:/geo_cultural_grounding/src/search_grounding.py): web search and page chunking
- [src/llm_client.py](D:/geo_cultural_grounding/src/llm_client.py): OpenAI client
- [scripts/run_general_workflow.py](D:/geo_cultural_grounding/scripts/run_general_workflow.py): end-to-end workflow
- [scripts/freeze_search_cache.py](D:/geo_cultural_grounding/scripts/freeze_search_cache.py): freeze retrieval artifacts
- [scripts/run_matrix.py](D:/geo_cultural_grounding/scripts/run_matrix.py): run `vanilla` vs `search_general`
- [scripts/analyze_matrix.py](D:/geo_cultural_grounding/scripts/analyze_matrix.py): main result analysis
- [scripts/analyze_search_diagnostics.py](D:/geo_cultural_grounding/scripts/analyze_search_diagnostics.py): layered failure analysis
- [scripts/export_case_prompt.py](D:/geo_cultural_grounding/scripts/export_case_prompt.py): reconstruct prompts for case study
- [scripts/prepare_benchmarks.py](D:/geo_cultural_grounding/scripts/prepare_benchmarks.py): benchmark preparation
- [scripts/visualize_token_usage.py](D:/geo_cultural_grounding/scripts/visualize_token_usage.py): token usage plots

## Main configs

- [configs/config_openai_general_rag.yaml](D:/geo_cultural_grounding/configs/config_openai_general_rag.yaml): formal OpenAI experiment config
- [configs/config.yaml](D:/geo_cultural_grounding/configs/config.yaml): local smoke config

## Data

- `data/eval_clean_stable.jsonl`: cleaned evaluation set used by the formal workflow
- `data/benchmark_eval.jsonl`: prepared benchmark set
- `data/sample_eval.jsonl`: small smoke-test sample

## Run

Formal experiment:

```bash
python scripts/run_general_workflow.py \
  --config configs/config_openai_general_rag.yaml \
  --out-root outputs/general_workflow \
  --provider openai \
  --model gpt-5.2 \
  --temperature 0 \
  --tag round_general_v2 \
  --refresh-cache
```

## Current baseline result

Direct search-content concatenation gave:

- `BLEnD`: `0.8553 -> 0.8816`
- `NormAd`: `0.8200 -> 0.7400`
- `SeeGULL`: `0.5100 -> 0.3400`

This is the current baseline phenomenon the project is trying to explain and improve.
