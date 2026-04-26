import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline import jsonl_integrity_summary, load_jsonl


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
    rows = sorted(
        rows,
        key=lambda r: hashlib.sha256(
            json.dumps(
                {
                    "id": str(r.get("id", "")),
                    "dataset": str(r.get("dataset", "")),
                    "question": str(r.get("question", "")),
                },
                sort_keys=True,
                ensure_ascii=False,
            ).encode("utf-8")
        ).hexdigest(),
    )[:limit]
    ensure_dir(str(Path(out_path).parent))
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return out_path


def run_is_complete(output_dir: Path, mode: str, expected_n: int) -> bool:
    metrics_path = output_dir / "metrics.json"
    pred_file = output_dir / ("vanilla_predictions.jsonl" if mode == "vanilla" else "search_predictions.jsonl")
    if not (metrics_path.exists() and pred_file.exists() and metrics_path.stat().st_size > 0 and pred_file.stat().st_size > 0):
        return False
    try:
        integrity = jsonl_integrity_summary(str(pred_file), expected_n=expected_n)
    except Exception:
        return False
    return bool(integrity.get("is_complete", False))


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


def _log_comb(n: int, k: int) -> float:
    import math

    if k < 0 or k > n:
        return float("-inf")
    # log(n choose k)
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def _binom_prob(n: int, k: int, p: float = 0.5) -> float:
    import math

    return math.exp(_log_comb(n, k) + k * math.log(p) + (n - k) * math.log(1.0 - p))


def mcnemar_exact(base_rows: List[Dict[str, Any]], test_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    b = align_by_id(base_rows)
    t = align_by_id(test_rows)
    ids = sorted(set(b.keys()) & set(t.keys()))
    b_only = 0  # base correct, test wrong
    t_only = 0  # test correct, base wrong
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


def plot_overall(overall: Dict[str, float], out_path: str) -> None:
    labels = list(overall.keys())
    vals = [overall[k] for k in labels]
    plt.figure(figsize=(10.5, 4.8))
    bars = plt.bar(labels, vals)
    plt.ylim(0, 1.05)
    plt.ylabel("Accuracy")
    plt.title("Layered Experiment: Overall Accuracy")
    plt.xticks(rotation=25, ha="right")
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
    plt.figure(figsize=(11.2, 5.0))
    for i, s in enumerate(systems):
        vals = [ds_map[s].get(d, 0.0) for d in datasets]
        offset = (i - (len(systems) - 1) / 2) * width
        plt.bar([xx + offset for xx in x], vals, width=width, label=s)
    plt.xticks(x, datasets)
    plt.ylim(0, 1.05)
    plt.ylabel("Accuracy")
    plt.title("Layered Experiment: Accuracy by Dataset")
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def build_groups(base_scfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    q_common = {
        "pipeline_variant": "general",
        "llm_relevance": False,
        "domain_priors": {},
        "strict_feature_checks": True,
        "embedding_preranker": "none",
        "semantic_reranker": "none",
        "enable_evidence_organization": False,
        "enable_evidence_gate": False,
        "min_evidence_score": 0.0,
        "low_quality_domains": [],
        "low_quality_url_keywords": [],
    }
    query_groups = [
        {
            "name": "Q0_simple_query",
            "phase": "query",
            "retrieval_key": "q0",
            "description": "Raw question retrieval with no rewrite or retry.",
            "params": {
                **q_common,
                "query_expansion_n": 1,
                "llm_query_rewrite": False,
                "enable_query_feedback_retry": False,
            },
        },
        {
            "name": "Q1_simple_rewrite",
            "phase": "query",
            "retrieval_key": "q1",
            "description": "Simple natural-language rewrite without retry.",
            "params": {
                **q_common,
                "query_expansion_n": int(base_scfg.get("query_expansion_n", 2)),
                "llm_query_rewrite": True,
                "enable_query_feedback_retry": False,
            },
        },
        {
            "name": "Q2_rewrite_retry",
            "phase": "query",
            "retrieval_key": "q2",
            "description": "Simple rewrite with one-step feedback retry.",
            "params": {
                **q_common,
                "query_expansion_n": int(base_scfg.get("query_expansion_n", 2)),
                "llm_query_rewrite": True,
                "enable_query_feedback_retry": True,
                "query_feedback_max_retry": 1,
                "query_retry_min_top_score": float(base_scfg.get("query_retry_min_top_score", 0.12)),
            },
        },
    ]

    # Downstream groups reuse retrieval cache from Q2 to isolate ranking/context modules.
    d_query_fixed = {
        "query_expansion_n": int(base_scfg.get("query_expansion_n", 2)),
        "llm_query_rewrite": True,
        "enable_query_feedback_retry": True,
        "query_feedback_max_retry": 1,
        "query_retry_min_top_score": float(base_scfg.get("query_retry_min_top_score", 0.12)),
    }
    downstream_groups = [
        {
            "name": "D0_lexical",
            "phase": "downstream",
            "retrieval_key": "q2",
            "description": "Fixed Q2 retrieval; lexical ranking only, no noise filter, no gate.",
            "params": {
                "pipeline_variant": "general",
                "llm_relevance": False,
                "domain_priors": {},
                "embedding_preranker": "none",
                "semantic_reranker": "none",
                "enable_evidence_organization": False,
                "enable_evidence_gate": False,
                "min_evidence_score": 0.0,
                "low_quality_domains": [],
                "low_quality_url_keywords": [],
                **d_query_fixed,
            },
        },
        {
            "name": "D1_semantic_rerank",
            "phase": "downstream",
            "retrieval_key": "q2",
            "description": "Fixed Q2 retrieval; embedding pre-rank + cross-encoder rerank.",
            "params": {
                "pipeline_variant": "general",
                "llm_relevance": False,
                "domain_priors": {},
                "embedding_preranker": "openai",
                "embedding_model": str(base_scfg.get("embedding_model", "text-embedding-3-small")),
                "embedding_preranker_top_m": int(base_scfg.get("embedding_preranker_top_m", 24)),
                "embedding_preranker_weight": float(base_scfg.get("embedding_preranker_weight", 0.15)),
                "semantic_reranker": "cross_encoder",
                "semantic_reranker_model": str(
                    base_scfg.get("semantic_reranker_model", "cross-encoder/ms-marco-MiniLM-L-12-v2")
                ),
                "semantic_reranker_top_m": int(base_scfg.get("semantic_reranker_top_m", 12)),
                "semantic_reranker_weight": float(base_scfg.get("semantic_reranker_weight", 0.2)),
                "semantic_reranker_device": str(base_scfg.get("semantic_reranker_device", "cuda")),
                "semantic_reranker_batch_size": int(base_scfg.get("semantic_reranker_batch_size", 32)),
                "enable_evidence_organization": False,
                "enable_evidence_gate": False,
                "min_evidence_score": 0.0,
                "low_quality_domains": [],
                "low_quality_url_keywords": [],
                **d_query_fixed,
            },
        },
        {
            "name": "D2_semantic_noise",
            "phase": "downstream",
            "retrieval_key": "q2",
            "description": "Fixed Q2 retrieval; semantic rerank + general noise filter.",
            "params": {
                "pipeline_variant": "general",
                "llm_relevance": False,
                "domain_priors": {},
                "embedding_preranker": "openai",
                "embedding_model": str(base_scfg.get("embedding_model", "text-embedding-3-small")),
                "embedding_preranker_top_m": int(base_scfg.get("embedding_preranker_top_m", 24)),
                "embedding_preranker_weight": float(base_scfg.get("embedding_preranker_weight", 0.15)),
                "semantic_reranker": "cross_encoder",
                "semantic_reranker_model": str(
                    base_scfg.get("semantic_reranker_model", "cross-encoder/ms-marco-MiniLM-L-12-v2")
                ),
                "semantic_reranker_top_m": int(base_scfg.get("semantic_reranker_top_m", 12)),
                "semantic_reranker_weight": float(base_scfg.get("semantic_reranker_weight", 0.2)),
                "semantic_reranker_device": str(base_scfg.get("semantic_reranker_device", "cuda")),
                "semantic_reranker_batch_size": int(base_scfg.get("semantic_reranker_batch_size", 32)),
                "enable_evidence_organization": False,
                "enable_evidence_gate": False,
                "min_evidence_score": 0.0,
                "low_quality_domains": list(base_scfg.get("low_quality_domains", []) or []),
                "low_quality_url_keywords": list(base_scfg.get("low_quality_url_keywords", []) or []),
                **d_query_fixed,
            },
        },
        {
            "name": "D3_full",
            "phase": "downstream",
            "retrieval_key": "q2",
            "description": "Fixed Q2 retrieval; full stack with organize+gate.",
            "params": {
                "pipeline_variant": "general",
                "llm_relevance": False,
                "domain_priors": {},
                "embedding_preranker": "openai",
                "embedding_model": str(base_scfg.get("embedding_model", "text-embedding-3-small")),
                "embedding_preranker_top_m": int(base_scfg.get("embedding_preranker_top_m", 24)),
                "embedding_preranker_weight": float(base_scfg.get("embedding_preranker_weight", 0.15)),
                "semantic_reranker": "cross_encoder",
                "semantic_reranker_model": str(
                    base_scfg.get("semantic_reranker_model", "cross-encoder/ms-marco-MiniLM-L-12-v2")
                ),
                "semantic_reranker_top_m": int(base_scfg.get("semantic_reranker_top_m", 12)),
                "semantic_reranker_weight": float(base_scfg.get("semantic_reranker_weight", 0.2)),
                "semantic_reranker_device": str(base_scfg.get("semantic_reranker_device", "cuda")),
                "semantic_reranker_batch_size": int(base_scfg.get("semantic_reranker_batch_size", 32)),
                "enable_evidence_organization": bool(base_scfg.get("enable_evidence_organization", True)),
                "enable_evidence_gate": bool(base_scfg.get("enable_evidence_gate", True)),
                "min_evidence_score": float(base_scfg.get("min_evidence_score", 0.16)),
                "summary_max_items": int(base_scfg.get("summary_max_items", 4)),
                "low_quality_domains": list(base_scfg.get("low_quality_domains", []) or []),
                "low_quality_url_keywords": list(base_scfg.get("low_quality_url_keywords", []) or []),
                **d_query_fixed,
            },
        },
    ]
    return query_groups + downstream_groups


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

    subset_eval = run_root / "eval_subset.jsonl"
    cfg_run["experiment"]["eval_path"] = prepare_subset(
        str(cfg_run["experiment"]["eval_path"]),
        str(subset_eval),
        args.limit,
    )
    expected_n = len(load_jsonl(str(cfg_run["experiment"]["eval_path"])))

    base_cfg_path = run_root / "configs" / "config_base.yaml"
    write_yaml(str(base_cfg_path), cfg_run)

    runs: List[Dict[str, Any]] = []

    # Vanilla baseline.
    vanilla_cfg = copy.deepcopy(cfg_run)
    vanilla_out = run_root / "runs" / "vanilla"
    vanilla_cfg["experiment"]["output_dir"] = str(vanilla_out)
    vanilla_cfg_path = run_root / "configs" / "config_vanilla.yaml"
    write_yaml(str(vanilla_cfg_path), vanilla_cfg)
    if not run_is_complete(vanilla_out, "vanilla", expected_n):
        run_cmd([sys.executable, "-m", "src.main", "--mode", "vanilla", "--config", str(vanilla_cfg_path)])
    vanilla_rows = load_jsonl(str(vanilla_out / "vanilla_predictions.jsonl"))
    runs.append(
        {
            "name": "vanilla",
            "phase": "baseline",
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
        retrieval_key = str(g["retrieval_key"])
        params = dict(g["params"])

        cache_path = run_root / "caches" / f"{retrieval_key}.jsonl"
        if retrieval_key not in frozen_for_key or args.refresh_cache or not cache_file_has_content(cache_path):
            freeze_cfg = copy.deepcopy(cfg_run)
            freeze_cfg["search_grounding"].update(params)
            # Freeze should keep candidate details for later replay/rerank.
            freeze_cfg["search_grounding"]["cache_path"] = str(cache_path)
            freeze_cfg["search_grounding"]["use_cache_only"] = False
            freeze_cfg["search_grounding"]["include_candidate_details"] = True
            freeze_cfg["experiment"]["output_dir"] = str(run_root / "runs" / f"{name}_freeze")
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

        run_cfg_path = run_root / "configs" / f"config_{name}.yaml"
        write_yaml(str(run_cfg_path), run_cfg)
        if not run_is_complete(run_out, "search", expected_n):
            run_cmd([sys.executable, "-m", "src.main", "--mode", "search", "--config", str(run_cfg_path)])

        pred_rows = load_jsonl(str(run_out / "search_predictions.jsonl"))
        runs.append(
            {
                "name": name,
                "phase": str(g.get("phase", "")),
                "description": str(g.get("description", "")),
                "retrieval_key": retrieval_key,
                "cache_path": str(frozen_for_key[retrieval_key]),
                "output_dir": str(run_out),
                "overall_accuracy": acc(pred_rows),
                "dataset_accuracy": by_dataset_acc(pred_rows),
                "n": len(pred_rows),
                "params": params,
            }
        )

    # Analysis
    rows_by_name: Dict[str, List[Dict[str, Any]]] = {"vanilla": vanilla_rows}
    for r in runs:
        name = str(r["name"])
        if name == "vanilla":
            continue
        pred_path = Path(r["output_dir"]) / "search_predictions.jsonl"
        if pred_path.exists():
            rows_by_name[name] = load_jsonl(str(pred_path))

    overall = {r["name"]: float(r["overall_accuracy"]) for r in runs}
    ds_map = {r["name"]: dict(r["dataset_accuracy"]) for r in runs}

    vanilla_ref = rows_by_name.get("vanilla", [])
    comparisons = {}
    for name, rows in rows_by_name.items():
        if name == "vanilla":
            continue
        wtl = win_tie_loss(vanilla_ref, rows)
        mcn = mcnemar_exact(vanilla_ref, rows)
        comparisons[f"{name}_vs_vanilla"] = {
            "delta_acc": acc(rows) - acc(vanilla_ref),
            "win_tie_loss": wtl,
            "mcnemar_exact": mcn,
        }

    # Key internal comparisons
    q_ref = rows_by_name.get("Q0_simple_query", [])
    if q_ref:
        for name in ["Q1_simple_rewrite", "Q2_rewrite_retry"]:
            if name in rows_by_name:
                comparisons[f"{name}_vs_Q0_simple_query"] = {
                    "delta_acc": acc(rows_by_name[name]) - acc(q_ref),
                    "win_tie_loss": win_tie_loss(q_ref, rows_by_name[name]),
                    "mcnemar_exact": mcnemar_exact(q_ref, rows_by_name[name]),
                }

    d_ref = rows_by_name.get("D0_lexical", [])
    if d_ref:
        for name in ["D1_semantic_rerank", "D2_semantic_noise", "D3_full"]:
            if name in rows_by_name:
                comparisons[f"{name}_vs_D0_lexical"] = {
                    "delta_acc": acc(rows_by_name[name]) - acc(d_ref),
                    "win_tie_loss": win_tie_loss(d_ref, rows_by_name[name]),
                    "mcnemar_exact": mcnemar_exact(d_ref, rows_by_name[name]),
                }

    plot_overall(overall, str(run_root / "analysis" / "overall_accuracy_layered.png"))
    plot_dataset_grouped(ds_map, str(run_root / "analysis" / "dataset_accuracy_layered.png"))

    summary = {
        "run_root": str(run_root),
        "config": str(base_cfg_path),
        "eval_path": str(cfg_run["experiment"]["eval_path"]),
        "model": cfg_run["llm"],
        "experiment_design": {
            "query_phase": ["Q0_simple_query", "Q1_simple_rewrite", "Q2_rewrite_retry"],
            "downstream_phase": ["D0_lexical", "D1_semantic_rerank", "D2_semantic_noise", "D3_full"],
            "notes": [
                "Downstream phase uses retrieval cache frozen from Q2 to isolate ranking/context effects.",
                "All runs use identical eval split and model settings.",
            ],
        },
        "runs": runs,
        "overall_accuracy": overall,
        "dataset_accuracy": ds_map,
        "comparisons": comparisons,
    }
    summary_path = run_root / "analysis" / "layered_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps({"summary_path": str(summary_path), **summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run layered controlled experiments for general search-RAG.")
    parser.add_argument("--config", default="configs/config_openai_general_rag.yaml")
    parser.add_argument("--out-root", default="outputs/layered_experiments")
    parser.add_argument("--tag", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--provider", default="")
    parser.add_argument("--model", default="")
    parser.add_argument("--temperature", type=float, default=0.0)
    run(parser.parse_args())
