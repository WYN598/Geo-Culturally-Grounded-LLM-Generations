# Geo-Cultural Grounding

一个用于复现和扩展 geo-cultural grounding 实验的 Python 工程。

当前代码支持三种推理设置：
1. `vanilla`：不使用外部证据的直接答题
2. `kb`：本地文化知识库检索增强（TF-IDF 或 Dense embedding）
3. `search`：网络检索增强（DuckDuckGo + 网页抓取 + chunk 选择）

## 项目状态（基于当前代码）

- 已实现完整实验闭环：数据准备 -> 运行矩阵 -> 统计检验 -> 可视化 -> 质量校验。
- 已实现两类“严格对比”流程：
  - Search-grounding 严格实验（冻结搜索证据后复跑）
  - KB-grounding 严格实验（冻结 KB 检索结果后复跑）
- 支持 `mock` 和 `openai` 两种 LLM provider。
- `kb` 支持 `tfidf` 与 `dense` 两种后端；`dense` 使用 OpenAI embedding，可选 FAISS。

## 目录结构

```text
geo_cultural_grounding/
├─ src/                     # 核心推理与评测逻辑
│  ├─ main.py               # 入口：vanilla/kb/search/all
│  ├─ pipeline.py           # VanillaPipeline / KBPipeline / SearchPipeline
│  ├─ retrieval.py          # TF-IDF 与 Dense KB 索引
│  ├─ search_grounding.py   # 搜索、网页清洗、证据切块
│  ├─ llm_client.py         # OpenAI / mock 客户端
│  └─ eval.py               # accuracy 计算
├─ scripts/                 # 数据准备、矩阵实验、统计与可视化脚本
├─ configs/                 # 实验配置模板
├─ data/                    # 数据集与KB（默认被 .gitignore 忽略）
├─ outputs/                 # 实验输出（默认被 .gitignore 忽略）
├─ requirements.txt
└─ README.md
```

## 安装

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

配置环境变量：

```bash
copy .env.example .env
```

`.env` 关键项：
- `OPENAI_API_KEY`
- `OPENAI_BASE_URL`（默认 `https://api.openai.com/v1`）

说明：
- `llm.provider=mock` 时可不填 API Key。
- `llm.provider=openai` 时必须提供 API Key。
- 若启用 `kb_grounding.backend=dense`，还需要 API Key 生成 embedding。

## 一分钟快速跑通（本地样例）

默认配置 `configs/config.yaml` 使用：
- `data/sample_eval.jsonl`（3 条样例）
- `data/sample_kb.jsonl`（5 条样例）
- `llm.provider=mock`

运行：

```bash
python -m src.main --mode all --config configs/config.yaml
```

输出：
- `outputs/vanilla_predictions.jsonl`
- `outputs/kb_predictions.jsonl`
- `outputs/search_predictions.jsonl`
- `outputs/metrics.json`

## 数据准备

### 1) 准备 benchmark 评测集（BLEnD / NormAd / SeeGULL）

```bash
python scripts/prepare_benchmarks.py --out data/benchmark_eval.jsonl
```

常用参数：
- `--datasets blend normad seegull`
- `--max-per-dataset 30`
- `--seed 42`

脚本会自动下载并转换为统一 MCQ `jsonl` 格式。

### 2) 构建 KB 语料（可选）

```bash
python scripts/build_kb_corpus.py \
  --include-seed \
  --seed-kb data/sample_kb.jsonl \
  --include-seegull \
  --out data/kb_full.jsonl
```

可选扩展：
- `--extra-jsonl ...` 合并额外 KB 文档
- `--wikipedia-countries-file ...` 拉取维基国家摘要

### 3) 构建 Dense KB 索引（可选）

```bash
python scripts/build_kb_dense_index.py \
  --kb-path data/kb_full.jsonl \
  --out-dir data/kb_dense_index \
  --embedding-model text-embedding-3-small
```

## 运行实验

### A. 单次运行（按 mode）

```bash
python -m src.main --mode vanilla --config configs/config.yaml
python -m src.main --mode kb      --config configs/config_openai_kb.yaml
python -m src.main --mode search  --config configs/config_openai_scale.yaml
```

### B. Search-grounding 严格矩阵实验（推荐）

1. 冻结检索证据（确保后续对比使用同一批网页证据）：

```bash
python scripts/freeze_search_cache.py \
  --config configs/config_openai_scale.yaml \
  --out-cache outputs/strict_search/search_cache.jsonl
```

2. 运行矩阵（Vanilla / Search-selective / Search-non-selective）：

```bash
python scripts/run_matrix.py \
  --config configs/config_openai_scale.yaml \
  --out-root outputs/strict_search \
  --limit 300 \
  --provider openai \
  --model gpt-4o-mini \
  --temperature 0
```

### C. KB-grounding 严格矩阵实验（推荐）

1. 冻结 KB 检索结果：

```bash
python scripts/freeze_kb_cache.py \
  --config configs/config_openai_kb.yaml \
  --out-cache outputs/strict_kb/kb_cache.jsonl
```

2. 运行矩阵（Vanilla / KB-selective / KB-non-selective）：

```bash
python scripts/run_kb_matrix.py \
  --config configs/config_openai_kb.yaml \
  --out-root outputs/strict_kb \
  --limit 300 \
  --provider openai \
  --model gpt-4o-mini \
  --temperature 0
```

## 统计检验与可视化

### Search 统计

```bash
python scripts/stats_report.py \
  --base outputs/strict_search/vanilla/vanilla_predictions.jsonl \
  --search-selective outputs/strict_search/search_selective/search_predictions.jsonl \
  --search-non-selective outputs/strict_search/search_non_selective/search_predictions.jsonl \
  --out outputs/strict_search/stats_report.json
```

### KB 统计

```bash
python scripts/kb_stats_report.py \
  --base outputs/strict_kb/vanilla/vanilla_predictions.jsonl \
  --kb-selective outputs/strict_kb/kb_selective/kb_predictions.jsonl \
  --kb-non-selective outputs/strict_kb/kb_non_selective/kb_predictions.jsonl \
  --out outputs/strict_kb/kb_stats_report.json
```

### 结果可视化

```bash
python scripts/analyze_matrix.py --matrix-root outputs/strict_search --out-dir outputs/analysis
python scripts/analyze_kb_matrix.py --matrix-root outputs/strict_kb --out-dir outputs/kb_analysis
python scripts/visualize_results.py --input-dir outputs/benchmark_quick --out-dir outputs/figures
```

### 工作流校验

```bash
python scripts/validate_workflow.py \
  --eval-path outputs/strict_search/eval_subset.jsonl \
  --vanilla-path outputs/strict_search/vanilla/vanilla_predictions.jsonl \
  --search-selective-path outputs/strict_search/search_selective/search_predictions.jsonl \
  --search-non-selective-path outputs/strict_search/search_non_selective/search_predictions.jsonl

python scripts/validate_kb_workflow.py \
  --eval-path outputs/strict_kb/eval_subset.jsonl \
  --vanilla-path outputs/strict_kb/vanilla/vanilla_predictions.jsonl \
  --kb-selective-path outputs/strict_kb/kb_selective/kb_predictions.jsonl \
  --kb-non-selective-path outputs/strict_kb/kb_non_selective/kb_predictions.jsonl
```

## 配置文件说明

- `configs/config.yaml`：最小样例，默认 `mock`。
- `configs/config_benchmark_quick.yaml`：快速 benchmark 验证配置。
- `configs/config_openai_scale.yaml`：Search-grounding 规模实验配置。
- `configs/config_openai_scale_balanced.yaml`：平衡/降成本版 Search 配置。
- `configs/config_openai_search_recheck.yaml`：Search 重检配置（更强检索参数）。
- `configs/config_openai_kb.yaml`：KB-grounding（dense）主配置。
- `configs/config_openai_kb_dense_smoke.yaml`：KB dense 小样本冒烟测试。
- `configs/config_openai_large.yaml`：较大评测集配置模板。

## 数据格式

### KB 文档（jsonl）

```json
{"id":"kb1","source":"CultureAtlas","country":"Japan","text":"In Japan, exchanging business cards with both hands is respectful."}
```

### 评测样本（jsonl）

```json
{
  "id": "q1",
  "dataset": "BLEnD",
  "task_type": "mcq",
  "question": "...",
  "choices": ["A) ...", "B) ...", "C) ...", "D) ..."],
  "answer": "B"
}
```

## 可复现性与注意事项

- Search 实验若不先冻结 cache，会受到实时网络内容变化影响。
- `run_matrix.py` / `run_kb_matrix.py` 在矩阵运行时会设置 `use_cache_only=true`，保证对比公平。
- `mock` 模式仅用于流程验证，不代表真实模型能力。
- 选择 `use_faiss=true` 前请先自行安装 FAISS（当前 `requirements.txt` 未默认安装）。
- `.gitignore` 默认忽略 `/data` 与 `/outputs`，这两个目录通常是本地实验资产。

## 常见命令速查

```bash
# 运行全部模式
python -m src.main --mode all --config configs/config.yaml

# 只跑 Search
python -m src.main --mode search --config configs/config_openai_scale.yaml

# 只跑 KB
python -m src.main --mode kb --config configs/config_openai_kb.yaml
```
