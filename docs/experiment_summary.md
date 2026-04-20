# Experiment Summary

## Global Setting

- Model provider: `openai`
- Model: `gpt-5.2`
- Temperature: `0.0`
- Search pipeline: same general search-grounding system
- Compared systems:
  - `Vanilla`
  - `Simple RAG` = raw query + lexical
  - `Simple Rewrite + Lexical`
  - `Simple Rewrite + Semantic Rerank`
  - `Simple Rewrite + Semantic Rerank + Noise Filter`
  - `Full General RAG` = rewrite + semantic rerank + noise filter + evidence organize + evidence gate

## 1. Legacy Geo-Cultural Benchmarks

Source:
- `outputs/general_ablation/balanced200_20260405/analysis/ablation_summary.json`

Dataset sizes:
- `BLEnD = 200`
- `NormAd = 200`
- `SeeGULL = 200`
- total `600`

| System | BLEnD | NormAd | SeeGULL | Overall |
|---|---:|---:|---:|---:|
| Vanilla | 0.860 | 0.835 | 0.515 | 0.7367 |
| Simple RAG | 0.845 | 0.790 | 0.290 | 0.6417 |
| Simple Rewrite + Lexical | 0.855 | 0.805 | 0.345 | 0.6683 |
| Simple Rewrite + Semantic Rerank | 0.855 | 0.810 | 0.320 | 0.6617 |
| Simple Rewrite + Semantic Rerank + Noise Filter | 0.860 | 0.820 | 0.310 | 0.6633 |
| Full General RAG | 0.855 | 0.835 | 0.485 | 0.7250 |

Best by benchmark:
- `BLEnD`: `Vanilla` and `Noise Filter` tie at `0.860`
- `NormAd`: `Vanilla` and `Full General RAG` tie at `0.835`
- `SeeGULL`: `Vanilla` best at `0.515`

Main observation:
- The full system largely recovers performance relative to weaker RAG variants.
- However, it does not clearly beat `Vanilla` on the legacy benchmark family.
- `SeeGULL` remains the hardest benchmark for search-grounding.

## 2. External Bias Benchmarks

### 2.1 BBQ_200

Source:
- `outputs/external_ablation/bbq_200_20260406/analysis/ablation_summary.json`

Primary metric:
- `Accuracy`

| System | Accuracy |
|---|---:|
| Vanilla | 0.750 |
| Simple RAG | 0.745 |
| Simple Rewrite + Lexical | 0.770 |
| Simple Rewrite + Semantic Rerank | 0.765 |
| Simple Rewrite + Semantic Rerank + Noise Filter | 0.755 |
| Full General RAG | 0.750 |

Best system:
- `Simple Rewrite + Lexical` at `0.770`

Observation:
- Light rewrite helps.
- The full organize+gate stack does not improve over vanilla here.

### 2.2 SocialStigmaQA_200

Source:
- `outputs/external_ablation/socialstigmaqa_200_20260406/analysis/ablation_summary.json`

Primary metric:
- `Non-Biased Rate`

| System | Non-Biased Rate | Bias Rate |
|---|---:|---:|
| Vanilla | 0.995 | 0.005 |
| Simple RAG | 1.000 | 0.000 |
| Simple Rewrite + Lexical | 0.990 | 0.010 |
| Simple Rewrite + Semantic Rerank | 0.995 | 0.005 |
| Simple Rewrite + Semantic Rerank + Noise Filter | 0.995 | 0.005 |
| Full General RAG | 1.000 | 0.000 |

Best systems:
- `Simple RAG` and `Full General RAG` at `1.000`

Observation:
- Retrieval does not hurt this dataset in the current setup.
- Both naive and full variants can reach perfect non-biased rate on the sampled split.

## 3. External Factual Benchmarks

### 3.1 TruthfulQA-Binary_200

Source:
- `outputs/external_ablation/truthfulqa_200_20260406/analysis/ablation_summary.json`

Primary metric:
- `Accuracy`

| System | Accuracy |
|---|---:|
| Vanilla | 0.905 |
| Simple RAG | 0.795 |
| Simple Rewrite + Lexical | 0.820 |
| Simple Rewrite + Semantic Rerank | 0.825 |
| Simple Rewrite + Semantic Rerank + Noise Filter | 0.820 |
| Full General RAG | 0.870 |

Best system:
- `Vanilla` at `0.905`

Best RAG variant:
- `Full General RAG` at `0.870`

Observation:
- Retrieval hurts the naive variants substantially.
- The full system recovers much of the lost performance, but still does not surpass vanilla.

### 3.2 PopQA_200

Source:
- `outputs/external_ablation/popqa_200_20260407/analysis/ablation_summary.json`

Primary metric:
- `Containment Match`

Secondary metric:
- `Exact Match`

| System | Exact Match | Containment Match |
|---|---:|---:|
| Vanilla | 0.305 | 0.425 |
| Simple RAG | 0.400 | 0.515 |
| Simple Rewrite + Lexical | 0.435 | 0.550 |
| Simple Rewrite + Semantic Rerank | 0.445 | 0.585 |
| Simple Rewrite + Semantic Rerank + Noise Filter | 0.470 | 0.600 |
| Full General RAG | 0.450 | 0.590 |

Best system:
- `Simple Rewrite + Semantic Rerank + Noise Filter` at `0.600` containment match

Observation:
- This is the clearest dataset showing retrieval benefit.
- Open-domain factual QA benefits strongly from retrieval, ranking, and filtering.
- The full gate-based system is slightly below the best simpler RAG variant.

## 4. Cross-Dataset Summary

### 4.1 What the current data supports

- RAG effects are strongly dataset-dependent.
- `PopQA` clearly benefits from retrieval.
- `TruthfulQA-Binary` does not currently benefit relative to a strong vanilla model.
- `BBQ` shows mild gains from simple rewrite, but not from the full system.
- `SocialStigmaQA` does not show systematic bias amplification in the current setup.
- On the legacy geo-cultural benchmarks, the full system improves greatly over weak RAG variants but still does not clearly beat `Vanilla` overall.

### 4.2 What the current data does not support

The current results do **not** support a simple universal claim such as:

- "RAG improves factual datasets while hurting bias datasets"

Instead, the more accurate conclusion is:

- retrieval gains and risks depend on benchmark family, task format, evidence quality, and evidence-use policy.

## 5. Best System by Dataset

| Dataset | Primary Metric | Best System | Best Score |
|---|---|---|---:|
| BLEnD_200 | Accuracy | Vanilla / Noise Filter | 0.860 |
| NormAd_200 | Accuracy | Vanilla / Full General RAG | 0.835 |
| SeeGULL_200 | Accuracy | Vanilla | 0.515 |
| BBQ_200 | Accuracy | Simple Rewrite + Lexical | 0.770 |
| SocialStigmaQA_200 | Non-Biased Rate | Simple RAG / Full General RAG | 1.000 |
| TruthfulQA-Binary_200 | Accuracy | Vanilla | 0.905 |
| PopQA_200 | Containment Match | Simple Rewrite + Semantic Rerank + Noise Filter | 0.600 |

## 6. Important Result Paths

- Legacy benchmark summary:
  - `outputs/general_ablation/balanced200_20260405/analysis/ablation_summary.json`
- BBQ summary:
  - `outputs/external_ablation/bbq_200_20260406/analysis/ablation_summary.json`
- SocialStigmaQA summary:
  - `outputs/external_ablation/socialstigmaqa_200_20260406/analysis/ablation_summary.json`
- TruthfulQA-Binary summary:
  - `outputs/external_ablation/truthfulqa_200_20260406/analysis/ablation_summary.json`
- PopQA summary:
  - `outputs/external_ablation/popqa_200_20260407/analysis/ablation_summary.json`

## 7. Immediate Takeaway

The current experiments show that a single general RAG design does not produce uniform gains across benchmark families. The same retrieval system can be:

- clearly helpful on open-domain factual QA,
- neutral or mixed on culture-grounded MCQ,
- harmful on some stereotype-style tasks,
- and neutral or mildly helpful on some bias/stigma tasks.

