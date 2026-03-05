# Geo-Cultural Grounding (MVP Reproduction)

最小可行复现代码，覆盖论文中的三种设置：
1. Vanilla LLM
2. KB-grounding（本地文化 KB + 检索增强）
3. Search-grounding（网络搜索 + 检索增强）

## 1) 安装

```bash
python -m venv .venv
. .venv/Scripts/activate   # Windows PowerShell: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 2) 配置

```bash
copy .env.example .env
```

- 无 API Key：自动走 `mock` 模式，也能完整跑通。
- 有 API Key：设置 `.env` 后改 `configs/config.yaml` 的 `llm.provider=openai`。

## 3) 准备数据（MVP 已附样例）

- `data/sample_kb.jsonl`: 文化知识库
- `data/sample_eval.jsonl`: 评测问题（多选）

可替换为你的真实数据，字段见下方“数据格式”。

## 4) 运行

```bash
python -m src.main --mode all --config configs/config.yaml
```

运行后输出：
- `outputs/vanilla_predictions.jsonl`
- `outputs/kb_predictions.jsonl`
- `outputs/search_predictions.jsonl`
- `outputs/metrics.json`

## 5) 数据格式

### KB 文档 (`*.jsonl`)
每行一个 JSON：
```json
{"id":"doc1","source":"CultureAtlas","country":"Japan","text":"In Japan, exchanging business cards with both hands is polite."}
```

### 评测样本 (`*.jsonl`)
每行一个 JSON：
```json
{
  "id":"q1",
  "task_type":"mcq",
  "question":"In Japan business meetings, how should you pass a name card?",
  "choices":["With one hand casually","With both hands respectfully","Throw it on table","No card etiquette"],
  "answer":"B"
}
```

- `answer` 用 `A/B/C/...`。

## 6) 论文对齐说明（MVP）

- Query rewriting: `KBPipeline.rewrite_query`
- Vector retrieval: `TfidfKBIndex.search`
- Relevance check: `KBPipeline.relevance_filter`
- Prompt augmentation: `build_augmented_prompt`
- Search grounding: `WebSearcher.search_and_fetch`

这是最小可行版本，用于先跑通实验闭环。后续可替换为：
- 更强 embedding / 向量库（FAISS/Qdrant）
- 专用搜索 API（如 SerpAPI/Google CSE）
- 自动化评分器（BLEnD/SeeGULL 对应指标）

## 7) 完整 Search-Grounding（当前版本已支持）

- `query expansion`: 多 query 检索（`query_expansion_n`）
- `web retrieval`: 搜索结果去重 + 域名配额（`keep_per_domain`）
- `evidence extraction`: 网页正文清洗 + chunk 切分（`chunk_chars/overlap_chars`）
- `relevance filtering`: 先 lexical 排序，再可选 LLM relevance boost（`llm_relevance`）
- `manual verbalizer`: 兜底将模型输出映射到 `A/B/C/...`
- `trace logging`: `search_trace` 中保留 queries、URL、score、证据文本

## 8) 一键准备 Benchmark（BLEnD / NormAd / SeeGULL）

```bash
python scripts/prepare_benchmarks.py
```

可选参数：
```bash
python scripts/prepare_benchmarks.py --datasets blend normad seegull --out data/benchmark_eval.jsonl
```

脚本会：
- 自动尝试下载 BLEnD（HF）和 NormAd（HF）
- 自动下载 SeeGULL GitHub 压缩包并解压
- 统一转换为本项目评测格式 `jsonl`
- 输出转换统计（raw/kept/skipped）

完成后，把配置改为：
- `experiment.eval_path: data/benchmark_eval.jsonl`

## 9) 严格 Search-Grounding 实验（冻结检索）

1. 冻结检索证据（确保对比实验用同一批网页证据）：
```bash
python scripts/freeze_search_cache.py --config configs/config.yaml --out-cache outputs/search_cache.jsonl --limit 50
```

2. 批量运行矩阵（Vanilla / Search-selective / Search-non-selective）：
```bash
python scripts/run_matrix.py --config configs/config.yaml --out-root outputs/strict_search --limit 50 --provider openai --temperature 0
```

3. 统计检验（McNemar + bootstrap + permutation）：
```bash
python scripts/stats_report.py ^
  --base outputs/strict_search/vanilla/vanilla_predictions.jsonl ^
  --search-selective outputs/strict_search/search_selective/search_predictions.jsonl ^
  --search-non-selective outputs/strict_search/search_non_selective/search_predictions.jsonl ^
  --out outputs/strict_search/stats_report.json
```

说明：
- `run_matrix.py` 默认启用 `use_cache_only=true`，不会重新联网检索。
- `stats_report.py` 会输出：
  - factual 侧：accuracy 差值 + McNemar 显著性
  - stereotype 侧（SeeGULL）：stereotype rate 差值 + paired permutation p-value
