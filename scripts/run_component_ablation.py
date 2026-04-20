import argparse
import copy
import json
import os
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline import load_jsonl


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def run_cmd(cmd: List[str]) -> None:
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def write_yaml(path: str, cfg: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)


def cache_file_has_content(path: Path) -> bool:
    if not path.exists() or path.stat().st_size <= 0:
        return False
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            for line in f:
                if line.strip():
                    return True
    except Exception:
        return False
    return False


def prepare_subset(eval_path: str, out_path: str, limit: int) -> str:
    if limit <= 0:
        return eval_path
    rows = load_jsonl(eval_path)
    rows = rows[:limit]
    ensure_dir(str(Path(out_path).parent))
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return out_path


def run_is_complete(output_dir: Path, mode: str) -> bool:
    metrics_path = output_dir / "metrics.json"
    pred_file = output_dir / ("vanilla_predictions.jsonl" if mode == "vanilla" else "search_predictions.jsonl")
    return metrics_path.exists() and pred_file.exists() and metrics_path.stat().st_size > 0 and pred_file.stat().st_size > 0


def acc(rows: List[Dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    return sum(1 for r in rows if str(r.get("pred", "")) == str(r.get("answer", ""))) / len(rows)


def by_dataset_acc(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    g: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        g[str(r.get("dataset", "unknown"))].append(r)
    return {k: acc(v) for k, v in sorted(g.items())}


def align_by_id(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {str(r.get("id", "")): r for r in rows if str(r.get("id", ""))}


def _log_comb(n: int, k: int) -> float:
    import math

    if k < 0 or k > n:
        return float("-inf")
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def _binom_prob(n: int, k: int, p: float = 0.5) -> float:
    import math

    return math.exp(_log_comb(n, k) + k * math.log(p) + (n - k) * math.log(1.0 - p))


def mcnemar_exact(base_rows: List[Dict[str, Any]], test_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    b = align_by_id(base_rows)
    t = align_by_id(test_rows)
    ids = sorted(set(b.keys()) & set(t.keys()))
    b_only = 0
    t_only = 0
    for i in ids:
        bc = 1 if str(b[i].get("pred", "")) == str(b[i].get("answer", "")) else 0
        tc = 1 if str(t[i].get("pred", "")) == str(t[i].get("answer", "")) else 0
        if bc == 1 and tc == 0:
            b_only += 1
        elif bc == 0 and tc == 1:
            t_only += 1
    n = b_only + t_only
    if n == 0:
        return {"b_only": b_only, "t_only": t_only, "n_discordant": 0, "p_value_two_sided": 1.0}
    tail = min(b_only, t_only)
    p = 0.0
    for k in range(0, tail + 1):
        p += _binom_prob(n, k, 0.5)
    p_two_sided = min(1.0, 2.0 * p)
    return {"b_only": b_only, "t_only": t_only, "n_discordant": n, "p_value_two_sided": p_two_sided}


def win_tie_loss(base_rows: List[Dict[str, Any]], test_rows: List[Dict[str, Any]]) -> Dict[str, int]:
    b = align_by_id(base_rows)
    t = align_by_id(test_rows)
    ids = sorted(set(b.keys()) & set(t.keys()))
    win = tie = loss = 0
    for i in ids:
        bc = 1 if str(b[i].get("pred", "")) == str(b[i].get("answer", "")) else 0
        tc = 1 if str(t[i].get("pred", "")) == str(t[i].get("answer", "")) else 0
        if tc > bc:
            win += 1
        elif tc < bc:
            loss += 1
        else:
            tie += 1
    return {"win": win, "tie": tie, "loss": loss, "n": len(ids)}


def plot_overall(overall: Dict[str, float], out_path: str) -> None:
    labels = list(overall.keys())
    vals = [overall[k] for k in labels]
    plt.figure(figsize=(9.8, 4.5))
    bars = plt.bar(labels, vals)
    plt.ylim(0, 1.05)
    plt.ylabel("Accuracy")
    plt.title("Component Ablation: Overall Accuracy")
    plt.xticks(rotation=20, ha="right")
    for b, v in zip(bars, vals):
        plt.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.3f}", ha="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_dataset_grouped(ds_map: Dict[str, Dict[str, float]], out_path: str) -> None:
    systems = list(ds_map.keys())
    datasets = sorted({d for m in ds_map.values() for d in m.keys()})
    if not datasets:
        return
    x = list(range(len(datasets)))
    width = max(0.12, 0.78 / max(1, len(systems)))
    plt.figure(figsize=(10.8, 4.8))
    for i, s in enumerate(systems):
        vals = [ds_map[s].get(d, 0.0) for d in datasets]
        offset = (i - (len(systems) - 1) / 2) * width
        plt.bar([xx + offset for xx in x], vals, width=width, label=s)
    plt.xticks(x, datasets)
    plt.ylim(0, 1.05)
    plt.ylabel("Accuracy")
    plt.title("Component Ablation: Accuracy by Dataset")
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def build_groups(base_scfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    common = {
        "pipeline_variant": "general",
        "llm_relevance": False,
        "domain_priors": {},
        "query_expansion_n": int(base_scfg.get("query_expansion_n", 2)),
        "enable_evidence_organization": False,
        "enable_evidence_gate": False,
        "min_evidence_score": 0.0,
        "summary_max_items": int(base_scfg.get("summary_max_items", 4)),
        "enable_query_feedback_retry": False,
        "query_feedback_max_retry": 0,
    }
    default_low_quality_domains = list(base_scfg.get("low_quality_domains", []) or [])
    default_low_quality_keywords = list(base_scfg.get("low_quality_url_keywords", []) or [])
    emb_model = str(base_scfg.get("embedding_model", "text-embedding-3-small"))
    sem_model = str(base_scfg.get("semantic_reranker_model", "cross-encoder/ms-marco-MiniLM-L-12-v2"))
    sem_device = str(base_scfg.get("semantic_reranker_device", "cuda"))
    sem_batch = int(base_scfg.get("semantic_reranker_batch_size", 32))
    emb_top_m = int(base_scfg.get("embedding_preranker_top_m", 36))
    emb_w = float(base_scfg.get("embedding_preranker_weight", 0.15))
    sem_top_m = int(base_scfg.get("semantic_reranker_top_m", 24))
    sem_w = float(base_scfg.get("semantic_reranker_weight", 0.2))

    return [
        {
            "name": "rewrite_off",
            "axis": "rewrite",
            "retrieval_key": "rw0",
            "description": "No rewrite; use the raw question for retrieval.",
            "params": {
                **common,
                "llm_query_rewrite": False,
                "embedding_preranker": "none",
                "semantic_reranker": "none",
                "low_quality_domains": [],
                "low_quality_url_keywords": [],
            },
        },
        {
            "name": "rewrite_on",
            "axis": "rewrite",
            "retrieval_key": "rw1",
            "description": "Simple natural-language rewrite, no rerank/noise.",
            "params": {
                **common,
                "llm_query_rewrite": True,
                "embedding_preranker": "none",
                "semantic_reranker": "none",
                "low_quality_domains": [],
                "low_quality_url_keywords": [],
            },
        },
        {
            "name": "rerank_off",
            "axis": "rerank",
            "retrieval_key": "rw1",
            "description": "Fixed rewrite_on retrieval cache; lexical ranking only.",
            "params": {
                **common,
                "llm_query_rewrite": True,
                "embedding_preranker": "none",
                "semantic_reranker": "none",
                "low_quality_domains": [],
                "low_quality_url_keywords": [],
            },
        },
        {
            "name": "rerank_on",
            "axis": "rerank",
            "retrieval_key": "rw1",
            "description": "Fixed rewrite_on retrieval cache; add embedding + cross-encoder rerank.",
            "params": {
                **common,
                "llm_query_rewrite": True,
                "embedding_preranker": "openai",
                "embedding_model": emb_model,
                "embedding_preranker_top_m": emb_top_m,
                "embedding_preranker_weight": emb_w,
                "semantic_reranker": "cross_encoder",
                "semantic_reranker_model": sem_model,
                "semantic_reranker_top_m": sem_top_m,
                "semantic_reranker_weight": sem_w,
                "semantic_reranker_device": sem_device,
                "semantic_reranker_batch_size": sem_batch,
                "low_quality_domains": [],
                "low_quality_url_keywords": [],
            },
        },
        {
            "name": "noise_off",
            "axis": "noise_filter",
            "retrieval_key": "rw1",
            "description": "Fixed rewrite_on retrieval cache; rerank on, noise filter off.",
            "params": {
                **common,
                "llm_query_rewrite": True,
                "embedding_preranker": "openai",
                "embedding_model": emb_model,
                "embedding_preranker_top_m": emb_top_m,
                "embedding_preranker_weight": emb_w,
                "semantic_reranker": "cross_encoder",
                "semantic_reranker_model": sem_model,
                "semantic_reranker_top_m": sem_top_m,
                "semantic_reranker_weight": sem_w,
                "semantic_reranker_device": sem_device,
                "semantic_reranker_batch_size": sem_batch,
                "low_quality_domains": [],
                "low_quality_url_keywords": [],
            },
        },
        {
            "name": "noise_on",
            "axis": "noise_filter",
            "retrieval_key": "rw1",
            "description": "Fixed rewrite_on retrieval cache; rerank on, noise filter on.",
            "params": {
                **common,
                "llm_query_rewrite": True,
                "embedding_preranker": "openai",
                "embedding_model": emb_model,
                "embedding_preranker_top_m": emb_top_m,
                "embedding_preranker_weight": emb_w,
                "semantic_reranker": "cross_encoder",
                "semantic_reranker_model": sem_model,
                "semantic_reranker_top_m": sem_top_m,
                "semantic_reranker_weight": sem_w,
                "semantic_reranker_device": sem_device,
                "semantic_reranker_batch_size": sem_batch,
                "low_quality_domains": default_low_quality_domains,
                "low_quality_url_keywords": default_low_quality_keywords,
            },
        },
    ]


def run(args: argparse.Namespace) -> None:
    with open(args.config, "r", encoding="utf-8-sig") as f:
        base_cfg = yaml.safe_load(f)

    run_root = Path(args.out_root) / (args.tag.strip() if args.tag.strip() else now_tag())
    ensure_dir(str(run_root))
    ensure_dir(str(run_root / "configs"))
    ensure_dir(str(run_root / "caches"))
    ensure_dir(str(run_root / "runs"))
    ensure_dir(str(run_root / "analysis"))

    cfg_run = copy.deepcopy(base_cfg)
    if args.provider:
        cfg_run["llm"]["provider"] = args.provider
    if args.model:
        cfg_run["llm"]["model"] = args.model
    cfg_run["llm"]["temperature"] = float(args.temperature)
    eval_subset = run_root / "eval_subset.jsonl"
    cfg_run["experiment"]["eval_path"] = prepare_subset(
        str(cfg_run["experiment"]["eval_path"]),
        str(eval_subset),
        int(args.limit),
    )
    eval_rows = load_jsonl(str(cfg_run["experiment"]["eval_path"]))
    eval_ids = {str(r.get("id", "")).strip() for r in eval_rows if str(r.get("id", "")).strip()}

    base_cfg_path = run_root / "configs" / "config_base.yaml"
    write_yaml(str(base_cfg_path), cfg_run)

    runs: List[Dict[str, Any]] = []

    vanilla_out = run_root / "runs" / "vanilla"
    ensure_dir(str(vanilla_out))
    fixed_vanilla_path = str(args.fixed_vanilla_preds or "").strip()
    if fixed_vanilla_path:
        vanilla_rows = load_jsonl(fixed_vanilla_path)
        fixed_ids = {str(r.get("id", "")).strip() for r in vanilla_rows if str(r.get("id", "")).strip()}
        if eval_ids and fixed_ids != eval_ids:
            missing = len(eval_ids - fixed_ids)
            extra = len(fixed_ids - eval_ids)
            raise ValueError(
                f"fixed vanilla ids mismatch eval set: missing={missing}, extra={extra}, "
                f"eval_n={len(eval_ids)}, fixed_n={len(fixed_ids)}"
            )
        # Keep a local copy for reproducibility.
        local_vanilla_path = vanilla_out / "vanilla_predictions.jsonl"
        with open(local_vanilla_path, "w", encoding="utf-8") as f:
            for r in vanilla_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        vanilla_metrics = {
            "vanilla_acc": acc(vanilla_rows),
            "vanilla_usage": {"fixed_baseline": True, "source_path": fixed_vanilla_path},
        }
        with open(vanilla_out / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(vanilla_metrics, f, ensure_ascii=False, indent=2)
    else:
        vanilla_cfg = copy.deepcopy(cfg_run)
        vanilla_cfg["experiment"]["output_dir"] = str(vanilla_out)
        vanilla_cfg_path = run_root / "configs" / "config_vanilla.yaml"
        write_yaml(str(vanilla_cfg_path), vanilla_cfg)
        if not run_is_complete(vanilla_out, "vanilla"):
            run_cmd([sys.executable, "-m", "src.main", "--mode", "vanilla", "--config", str(vanilla_cfg_path)])
        vanilla_rows = load_jsonl(str(vanilla_out / "vanilla_predictions.jsonl"))
    runs.append(
        {
            "name": "vanilla",
            "axis": "baseline",
            "description": "No retrieval baseline.",
            "output_dir": str(vanilla_out),
            "overall_accuracy": acc(vanilla_rows),
            "dataset_accuracy": by_dataset_acc(vanilla_rows),
            "n": len(vanilla_rows),
            "params": {"mode": "vanilla"},
        }
    )

    groups = build_groups(cfg_run.get("search_grounding", {}) or {})
    frozen_for_key: Dict[str, Path] = {}

    for g in groups:
        name = str(g["name"])
        params = dict(g["params"])
        retrieval_key = str(g["retrieval_key"])
        cache_path = run_root / "caches" / f"{retrieval_key}.jsonl"

        if retrieval_key not in frozen_for_key or args.refresh_cache or not cache_file_has_content(cache_path):
            freeze_cfg = copy.deepcopy(cfg_run)
            freeze_cfg["search_grounding"].update(params)
            freeze_cfg["search_grounding"]["cache_path"] = str(cache_path)
            freeze_cfg["search_grounding"]["use_cache_only"] = False
            # Important: include raw candidates so downstream noise/rerank toggles are fair.
            freeze_cfg["search_grounding"]["include_candidate_details"] = True
            freeze_cfg["search_grounding"]["enable_evidence_organization"] = False
            freeze_cfg["search_grounding"]["enable_evidence_gate"] = False
            freeze_cfg_path = run_root / "configs" / f"config_freeze_{retrieval_key}.yaml"
            write_yaml(str(freeze_cfg_path), freeze_cfg)
            run_cmd(
                [
                    sys.executable,
                    "scripts/freeze_search_cache.py",
                    "--config",
                    str(freeze_cfg_path),
                    "--out-cache",
                    str(cache_path),
                    "--limit",
                    "0",
                ]
            )
            frozen_for_key[retrieval_key] = cache_path
        else:
            frozen_for_key[retrieval_key] = cache_path

        run_cfg = copy.deepcopy(cfg_run)
        run_out = run_root / "runs" / name
        run_cfg["experiment"]["output_dir"] = str(run_out)
        run_cfg["search_grounding"].update(params)
        run_cfg["search_grounding"]["cache_path"] = str(frozen_for_key[retrieval_key])
        run_cfg["search_grounding"]["use_cache_only"] = True
        run_cfg["search_grounding"]["include_candidate_details"] = False
        run_cfg["search_grounding"]["enable_evidence_organization"] = False
        run_cfg["search_grounding"]["enable_evidence_gate"] = False

        run_cfg_path = run_root / "configs" / f"config_{name}.yaml"
        write_yaml(str(run_cfg_path), run_cfg)
        if not run_is_complete(run_out, "search"):
            run_cmd([sys.executable, "-m", "src.main", "--mode", "search", "--config", str(run_cfg_path)])

        rows = load_jsonl(str(run_out / "search_predictions.jsonl"))
        runs.append(
            {
                "name": name,
                "axis": str(g.get("axis", "")),
                "description": str(g.get("description", "")),
                "retrieval_key": retrieval_key,
                "cache_path": str(frozen_for_key[retrieval_key]),
                "output_dir": str(run_out),
                "overall_accuracy": acc(rows),
                "dataset_accuracy": by_dataset_acc(rows),
                "n": len(rows),
                "params": params,
            }
        )

    rows_by_name = {"vanilla": vanilla_rows}
    for r in runs:
        if r["name"] == "vanilla":
            continue
        rows_by_name[r["name"]] = load_jsonl(str(Path(r["output_dir"]) / "search_predictions.jsonl"))

    overall = {r["name"]: float(r["overall_accuracy"]) for r in runs}
    ds_map = {r["name"]: dict(r["dataset_accuracy"]) for r in runs}

    comparisons: Dict[str, Any] = {}
    pairs = [
        ("rewrite_on", "rewrite_off"),
        ("rerank_on", "rerank_off"),
        ("noise_on", "noise_off"),
    ]
    for a, b in pairs:
        if a in rows_by_name and b in rows_by_name:
            ra = rows_by_name[a]
            rb = rows_by_name[b]
            comparisons[f"{a}_vs_{b}"] = {
                "delta_acc": acc(ra) - acc(rb),
                "win_tie_loss": win_tie_loss(rb, ra),
                "mcnemar_exact": mcnemar_exact(rb, ra),
            }

    for name, rows in rows_by_name.items():
        if name == "vanilla":
            continue
        comparisons[f"{name}_vs_vanilla"] = {
            "delta_acc": acc(rows) - acc(vanilla_rows),
            "win_tie_loss": win_tie_loss(vanilla_rows, rows),
            "mcnemar_exact": mcnemar_exact(vanilla_rows, rows),
        }

    plot_overall(overall, str(run_root / "analysis" / "overall_accuracy_components.png"))
    plot_dataset_grouped(ds_map, str(run_root / "analysis" / "dataset_accuracy_components.png"))

    summary = {
        "run_root": str(run_root),
        "config": str(base_cfg_path),
        "eval_path": str(cfg_run["experiment"]["eval_path"]),
        "model": cfg_run["llm"],
        "design": {
            "goal": "Isolate effects of rewrite, rerank, and noise filter under fixed model/eval.",
            "fairness_notes": [
                "Vanilla baseline can be fixed by --fixed-vanilla-preds to avoid run-to-run drift.",
                "rewrite comparison uses separate frozen retrieval caches (rewrite_off vs rewrite_on).",
                "rerank/noise comparisons reuse rewrite_on cache and replay from raw candidate evidence.",
                "evidence organize/gate are disabled for this component-level study.",
            ],
            "pairs": pairs,
        },
        "runs": runs,
        "overall_accuracy": overall,
        "dataset_accuracy": ds_map,
        "comparisons": comparisons,
    }
    summary_path = run_root / "analysis" / "component_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps({"summary_path": str(summary_path), **summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run focused component ablation: rewrite vs rerank vs noise filter.")
    parser.add_argument("--config", default="configs/config_openai_general_rag.yaml")
    parser.add_argument("--out-root", default="outputs/component_ablation")
    parser.add_argument("--tag", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument(
        "--fixed-vanilla-preds",
        default="",
        help="Optional path to a fixed vanilla_predictions.jsonl baseline (no vanilla rerun).",
    )
    parser.add_argument("--provider", default="")
    parser.add_argument("--model", default="")
    parser.add_argument("--temperature", type=float, default=0.0)
    run(parser.parse_args())
