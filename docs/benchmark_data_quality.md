**Benchmark Data Quality Audit**

Audit scope:
- [blend_200.jsonl](d:/geo_cultural_grounding/data/benchmarks/legacy_single/blend_200.jsonl)
- [normad_200.jsonl](d:/geo_cultural_grounding/data/benchmarks/legacy_single/normad_200.jsonl)
- [seegull_200.jsonl](d:/geo_cultural_grounding/data/benchmarks/legacy_single/seegull_200.jsonl)
- [bbq_200.jsonl](d:/geo_cultural_grounding/data/benchmarks/external/sampled/bbq_200.jsonl)
- [socialstigmaqa_200.jsonl](d:/geo_cultural_grounding/data/benchmarks/external/sampled/socialstigmaqa_200.jsonl)
- [truthfulqa_200.jsonl](d:/geo_cultural_grounding/data/benchmarks/external/sampled/truthfulqa_200.jsonl)
- [popqa_200.jsonl](d:/geo_cultural_grounding/data/benchmarks/external/sampled/popqa_200.jsonl)

Common checks:
- row count
- dataset/task-type consistency
- missing required fields
- duplicate ids
- duplicate questions
- label validity
- obvious text anomalies

**Summary**

| Benchmark | Rows | Format | Field Completeness | Duplicate IDs | Duplicate Questions | Label Validity | Main Quality Note |
|---|---:|---|---|---:|---:|---|---|
| BLEnD | 200 | MCQ-4 | Good | 0 | 11 | Good | Some repeated question stems with different options |
| NormAd | 200 | MCQ-2 | Good | 0 | 0 | Good | Cleanest dataset structurally |
| SeeGULL | 200 | MCQ-3 | Good | 0 | 0 | Good | All questions contain benchmark-style annotation wording |
| BBQ | 200 | MCQ-3 | Good | 0 | 0 | Good | One minor whitespace anomaly; otherwise clean |
| SocialStigmaQA | 200 | Bias-probe MCQ-3 | Good | 0 | 0 | Good | Contains non-breaking spaces; evaluation must use allowed non-biased answers |
| TruthfulQA-Binary | 200 | MCQ-2 | Good | 0 | 0 | Good | Binary proxy task, not official full TruthfulQA protocol |
| PopQA | 200 | Short QA | Good | 0 | 1 | N/A | One duplicate question; open-answer evaluation needed |

**Per Benchmark**

**BLEnD**
- Rows: `200`
- Task type: all `mcq`
- Choice count: all `4`
- Answer distribution: `A/B/C/D = 50/50/50/50`
- Missing required fields: `0`
- Invalid labels: `0`
- Duplicate ids: `0`
- Duplicate questions: `11`

Interpretation:
- Structurally clean.
- The main issue is duplicated question stems with different option sets. This means BLEnD is not fully iid at the question-stem level.
- Example repeated stems include:
  - `What is the most popular tourist attraction for foreign visitors in Indonesia?`
  - `Which city in China hosts festivals the most?`
  - `What is the cheapest fast food in West Java?`

Recommendation:
- Acceptable for current experiments.
- For strict statistical testing, be aware that some stem-level duplication remains.

**NormAd**
- Rows: `200`
- Task type: all `mcq`
- Choice count: all `2`
- Answer distribution: `A/B = 100/100`
- Missing required fields: `0`
- Invalid labels: `0`
- Duplicate ids: `0`
- Duplicate questions: `0`

Interpretation:
- Structurally very clean.
- Balanced labels.
- No obvious contamination or formatting artifacts beyond natural scenario templating.

Recommendation:
- Good benchmark for stable controlled comparisons.

**SeeGULL**
- Rows: `200`
- Task type: all `mcq`
- Choice count: all `3`
- Answer distribution: `A=96, B=90, C=14`
- Missing required fields: `0`
- Invalid labels: `0`
- Duplicate ids: `0`
- Duplicate questions: `0`
- All `200/200` questions contain the phrase `dominant annotation label`
- All `200/200` use the `For the claim 'X -> Y'` template

Interpretation:
- Structurally clean.
- Conceptually high-risk for search-grounding because the question wording is benchmark-internal, unnatural, and not directly web-searchable.
- This is a task-format issue, not a broken-file issue.

Recommendation:
- Good for studying stereotype-label tasks.
- Do not interpret it as a standard factual search benchmark.

**BBQ**
- Rows: `200`
- Task type: all `mcq`
- Choice count: all `3`
- Answer distribution: `A=66, B=67, C=67`
- Missing required fields: `0`
- Invalid labels: `0`
- Duplicate ids: `0`
- Duplicate questions: `0`
- Text anomalies: `1`

Interpretation:
- Structurally strong.
- One minor formatting anomaly was found:
  - double space in one question: `A colleague and  a transgender man ...`
- This is negligible and does not affect benchmark usability.

Recommendation:
- Good benchmark for QA-style bias testing.

**SocialStigmaQA**
- Rows: `200`
- Task type: all `bias_probe_mcq`
- Choice count: all `3`
- `biased_answer` distribution: `A=100, B=100`
- `allowed_non_biased_answers`:
  - `('B', 'C') = 100`
  - `('A', 'C') = 100`
- Missing required fields: `0`
- Invalid labels: `0`
- Duplicate ids: `0`
- Duplicate questions: `0`

Interpretation:
- Structurally clean.
- Contains non-breaking spaces in some prompts, but this is a text-normalization issue, not a dataset validity issue.
- The important quality point is evaluation semantics:
  - this benchmark cannot be scored with plain MCQ accuracy
  - it must use `biased_answer` together with `allowed_non_biased_answers`

Recommendation:
- Good for stigma-oriented bias evaluation.
- Use the corrected `bias_probe` evaluator only.

**TruthfulQA-Binary**
- Rows: `200`
- Task type: all `mcq`
- Choice count: all `2`
- Answer distribution: `A=100, B=100`
- Missing required fields: `0`
- Invalid labels: `0`
- Duplicate ids: `0`
- Duplicate questions: `0`
- Category coverage includes:
  - `Misconceptions`
  - `Economics`
  - `Law`
  - `Health`
  - `Sociology`
  - `Fiction`
  - `Myths and Fairytales`
  - `Stereotypes`
  - `Conspiracies`
  - `Superstitions`

Interpretation:
- Structurally clean.
- The key limitation is conceptual:
  - this is a binary proxy built from `best_answer` vs `best_incorrect_answer`
  - it is not the official full TruthfulQA evaluation protocol

Recommendation:
- Use it as a controlled binary factuality benchmark.
- Do not report it as full TruthfulQA without qualification.

**PopQA**
- Rows: `200`
- Task type: all `short_qa`
- Missing required fields: `0`
- Duplicate ids: `0`
- Duplicate questions: `1`
- Example duplicate question:
  - `Who was the director of The Take?`

Interpretation:
- Structurally clean overall.
- One duplicate question remains.
- Open-answer format means answer normalization and containment metrics matter more than exact string equality.

Recommendation:
- Good retrieval-sensitive factual QA benchmark.
- Keep using `exact_match` and `containment_match`.

**Overall Assessment**

The datasets are generally usable. There are no broken-file issues in the current `200`-sample benchmark set.

The main remaining data-quality risks are not file corruption but benchmark design properties:
- BLEnD: repeated stems with different options
- SeeGULL: benchmark-style unnatural label wording
- SocialStigmaQA: requires task-specific evaluator semantics
- TruthfulQA-Binary: proxy version, not official full protocol
- PopQA: one duplicate question and open-answer normalization sensitivity
