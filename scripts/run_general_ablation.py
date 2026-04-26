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
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline import (
    build_search_cache_fingerprint,
    cache_meta_matches,
    dump_jsonl,
    jsonl_integrity_summary,
    load_jsonl,
)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def run_cmd(cmd: List[str]) -> None:
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def write_yaml(path: str, cfg: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8-sig") as f:
        return yaml.safe_load(f)


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
    rows = load_jsonl(eval_path, strict=True)
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
    dump_jsonl(out_path, rows)
    return out_path


def now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def acc(rows: List[Dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    return sum(1 for r in rows if str(r.get("pred", "")) == str(r.get("answer", ""))) / len(rows)


def by_dataset_acc(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("dataset", "unknown"))].append(row)
    return {k: acc(v) for k, v in sorted(grouped.items())}


def plot_overall(overall: Dict[str, float], out_path: str) -> None:
    labels = list(overall.keys())
    vals = [overall[k] for k in labels]
    plt.figure(figsize=(8, 4.5))
    bars = plt.bar(labels, vals)
    plt.ylim(0, 1.05)
    plt.ylabel("Accuracy")
    plt.title("General Search-RAG Ablation")
    plt.xticks(rotation=20, ha="right")
    for bar, val in zip(bars, vals):
        plt.text(bar.get_x() + bar.get_width() / 2, val + 0.02, f"{val:.3f}", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_by_dataset(dataset_scores: Dict[str, Dict[str, float]], out_path: str) -> None:
    systems = list(dataset_scores.keys())
    datasets = sorted({d for mapping in dataset_scores.values() for d in mapping.keys()})
    if not datasets:
        return
    x = list(range(len(datasets)))
    width = 0.16
    plt.figure(figsize=(10, 4.8))
    for i, system in enumerate(systems):
        vals = [dataset_scores[system].get(d, 0.0) for d in datasets]
        offset = (i - (len(systems) - 1) / 2) * width
        plt.bar([xx + offset for xx in x], vals, width=width, label=system)
    plt.xticks(x, datasets)
    plt.ylim(0, 1.05)
    plt.ylabel("Accuracy")
    plt.title("Accuracy by Benchmark")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def summarize_run(output_dir: Path, name: str, params: Dict[str, Any], expected_n: int) -> Dict[str, Any]:
    pred_file = output_dir / ("vanilla_predictions.jsonl" if name == "vanilla" else "search_predictions.jsonl")
    integrity = jsonl_integrity_summary(str(pred_file), expected_n=expected_n) if pred_file.exists() else {
        "path": str(pred_file),
        "total_lines": 0,
        "nonempty_lines": 0,
        "parsed_lines": 0,
        "bad_lines": 0,
        "bad_line_numbers": [],
        "expected_n": expected_n,
        "missing_rows": expected_n,
        "is_complete": False,
    }
    rows = load_jsonl(str(pred_file), strict=True, expected_n=expected_n) if pred_file.exists() and integrity["is_complete"] else []
    metrics_path = output_dir / "metrics.json"
    metrics = {}
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
    used_evidence_n = 0
    if name != "vanilla":
        used_evidence_n = sum(1 for r in rows if bool((r.get("search_trace", {}) or {}).get("used_evidence", False)))
    return {
        "name": name,
        "output_dir": str(output_dir),
        "params": params,
        "n": len(rows),
        "expected_n": expected_n,
        "parsed_n": integrity["parsed_lines"],
        "bad_rows": integrity["bad_lines"],
        "prediction_integrity": integrity,
        "overall_accuracy": acc(rows),
        "dataset_accuracy": by_dataset_acc(rows),
        "used_evidence_n": used_evidence_n,
        "used_evidence_rate": ((used_evidence_n / len(rows)) if rows and name != "vanilla" else 0.0),
        "metrics": metrics,
    }


def run_is_complete(output_dir: Path, name: str, expected_n: int) -> bool:
    metrics_path = output_dir / "metrics.json"
    pred_file = output_dir / ("vanilla_predictions.jsonl" if name == "vanilla" else "search_predictions.jsonl")
    if not (metrics_path.exists() and pred_file.exists() and metrics_path.stat().st_size > 0 and pred_file.stat().st_size > 0):
        return False
    try:
        integrity = jsonl_integrity_summary(str(pred_file), expected_n=expected_n)
    except Exception:
        return False
    return bool(integrity.get("is_complete", False))


def build_search_variants(base_scfg: Dict[str, Any], include_full_general: bool = True) -> List[Dict[str, Any]]:
    default_domains = list(base_scfg.get("low_quality_domains", []) or [])
    default_keywords = list(base_scfg.get("low_quality_url_keywords", []) or [])
    common = {
        "pipeline_variant": "general",
        "llm_relevance": False,
        "domain_priors": {},
        "strict_feature_checks": True,
    }
    variants = [
        {
            "name": "simple_rag",
            "retrieval_key": "simple",
            "description": "Raw question search, lexical ranking, direct evidence use.",
            "params": {
                **common,
                "llm_query_rewrite": False,
                "embedding_preranker": "none",
                "semantic_reranker": "none",
                "enable_evidence_organization": False,
                "enable_evidence_gate": False,
                "min_evidence_score": 0.0,
                "low_quality_domains": [],
                "low_quality_url_keywords": [],
            },
        },
        {
            "name": "planning_rag",
            "retrieval_key": "planning",
            "description": "Simple natural-language rewrite before search, lexical ranking, direct evidence use.",
            "params": {
                **common,
                "llm_query_rewrite": True,
                "embedding_preranker": "none",
                "semantic_reranker": "none",
                "enable_evidence_organization": False,
                "enable_evidence_gate": False,
                "min_evidence_score": 0.0,
                "low_quality_domains": [],
                "low_quality_url_keywords": [],
            },
        },
        {
            "name": "planning_semantic_rag",
            "retrieval_key": "planning",
            "description": "Simple rewrite plus embedding pre-rank and cross-encoder rerank.",
            "params": {
                **common,
                "llm_query_rewrite": True,
                "embedding_preranker": "openai",
                "semantic_reranker": "cross_encoder",
                "enable_evidence_organization": False,
                "enable_evidence_gate": False,
                "min_evidence_score": 0.0,
                "low_quality_domains": [],
                "low_quality_url_keywords": [],
            },
        },
        {
            "name": "planning_semantic_noise_filter",
            "retrieval_key": "planning",
            "description": "Simple rewrite plus semantic ranking and source/noise filtering.",
            "params": {
                **common,
                "llm_query_rewrite": True,
                "embedding_preranker": "openai",
                "semantic_reranker": "cross_encoder",
                "enable_evidence_organization": False,
                "enable_evidence_gate": False,
                "min_evidence_score": 0.0,
                "low_quality_domains": default_domains,
                "low_quality_url_keywords": default_keywords,
            },
        },
        {
            "name": "full_general_rag",
            "retrieval_key": "planning",
            "description": "Full system: simple rewrite, semantic ranking, noise filtering, evidence organization, and gate.",
            "params": {
                **common,
                "llm_query_rewrite": True,
                "embedding_preranker": str(base_scfg.get("embedding_preranker", "openai")),
                "embedding_model": str(base_scfg.get("embedding_model", "text-embedding-3-small")),
                "embedding_preranker_top_m": int(base_scfg.get("embedding_preranker_top_m", 36)),
                "embedding_preranker_weight": float(base_scfg.get("embedding_preranker_weight", 0.15)),
                "semantic_reranker": str(base_scfg.get("semantic_reranker", "cross_encoder")),
                "semantic_reranker_model": str(base_scfg.get("semantic_reranker_model", "cross-encoder/ms-marco-MiniLM-L-12-v2")),
                "semantic_reranker_top_m": int(base_scfg.get("semantic_reranker_top_m", 24)),
                "semantic_reranker_weight": float(base_scfg.get("semantic_reranker_weight", 0.2)),
                "semantic_reranker_device": str(base_scfg.get("semantic_reranker_device", "cuda")),
                "semantic_reranker_batch_size": int(base_scfg.get("semantic_reranker_batch_size", 32)),
                "enable_evidence_organization": bool(base_scfg.get("enable_evidence_organization", True)),
                "enable_evidence_gate": bool(base_scfg.get("enable_evidence_gate", True)),
                "min_evidence_score": float(base_scfg.get("min_evidence_score", 0.16)),
                "summary_max_items": int(base_scfg.get("summary_max_items", 4)),
                "low_quality_domains": default_domains,
                "low_quality_url_keywords": default_keywords,
            },
        },
    ]
    if not include_full_general:
        variants = [v for v in variants if str(v.get("name", "")) != "full_general_rag"]
    return variants


def select_search_variants(
    base_scfg: Dict[str, Any],
    include_full_general: bool = True,
) -> List[Dict[str, Any]]:
    return build_search_variants(base_scfg, include_full_general=include_full_general)


def run(args: argparse.Namespace) -> None:
    base_cfg = load_yaml(args.config)
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
    expected_n = len(load_jsonl(str(cfg_run["experiment"]["eval_path"]), strict=True))

    base_cfg_path = run_root / "configs" / "config_base.yaml"
    write_yaml(str(base_cfg_path), cfg_run)

    runs: List[Dict[str, Any]] = []

    vanilla_cfg = copy.deepcopy(cfg_run)
    vanilla_out = run_root / "runs" / "vanilla"
    vanilla_cfg["experiment"]["output_dir"] = str(vanilla_out)
    vanilla_cfg_path = run_root / "configs" / "config_vanilla.yaml"
    write_yaml(str(vanilla_cfg_path), vanilla_cfg)
    if not run_is_complete(vanilla_out, "vanilla", expected_n):
        run_cmd([sys.executable, "-m", "src.main", "--mode", "vanilla", "--config", str(vanilla_cfg_path)])
    runs.append(summarize_run(vanilla_out, "vanilla", {"mode": "vanilla"}, expected_n))

    variants = select_search_variants(
        cfg_run.get("search_grounding", {}) or {},
        include_full_general=not bool(args.no_full_general),
    )
    frozen_for_key: Dict[str, Path] = {}
    for variant in variants:
        name = str(variant["name"])
        params = dict(variant["params"])
        retrieval_key = str(variant["retrieval_key"])
        cache_path = run_root / "caches" / f"{retrieval_key}.jsonl"

        freeze_cfg = copy.deepcopy(cfg_run)
        freeze_cfg["search_grounding"].update(params)
        freeze_cfg["search_grounding"]["cache_path"] = str(cache_path)
        freeze_cfg["search_grounding"]["use_cache_only"] = False
        freeze_cfg["search_grounding"]["include_candidate_details"] = True
        cache_meta = build_search_cache_fingerprint(freeze_cfg)
        cache_matches, _ = cache_meta_matches(str(cache_path), cache_meta)

        if retrieval_key not in frozen_for_key or args.refresh_cache or not cache_file_has_content(cache_path) or not cache_matches:
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

        search_cfg = copy.deepcopy(cfg_run)
        search_out = run_root / "runs" / name
        search_cfg["experiment"]["output_dir"] = str(search_out)
        search_cfg["search_grounding"].update(params)
        search_cfg["search_grounding"]["cache_path"] = str(frozen_for_key[retrieval_key])
        search_cfg["search_grounding"]["use_cache_only"] = True
        search_cfg["search_grounding"]["include_candidate_details"] = False
        search_cfg_path = run_root / "configs" / f"config_{name}.yaml"
        write_yaml(str(search_cfg_path), search_cfg)
        if not run_is_complete(search_out, name, expected_n):
            run_cmd([sys.executable, "-m", "src.main", "--mode", "search", "--config", str(search_cfg_path)])
        run_summary = summarize_run(search_out, name, params, expected_n)
        run_summary["description"] = str(variant.get("description", "") or "").strip()
        run_summary["retrieval_key"] = retrieval_key
        run_summary["cache_path"] = str(frozen_for_key[retrieval_key])
        runs.append(run_summary)

    overall = {r["name"]: r["overall_accuracy"] for r in runs}
    dataset_scores = {r["name"]: r["dataset_accuracy"] for r in runs}

    plot_overall(overall, str(run_root / "analysis" / "overall_accuracy_ablation.png"))
    plot_by_dataset(dataset_scores, str(run_root / "analysis" / "dataset_accuracy_ablation.png"))

    summary = {
        "experiment_design": {
            "goal": "Compare increasingly stronger general RAG designs under fixed model and evaluation data.",
            "systems": [
                {
                    "name": "vanilla",
                    "description": "No retrieval baseline.",
                },
                *[
                    {
                        "name": str(v["name"]),
                        "description": str(v.get("description", "") or "").strip(),
                        "retrieval_key": str(v["retrieval_key"]),
                    }
                    for v in variants
                ],
            ],
            "notes": [
                "Runs sharing the same retrieval_key reuse the same frozen search cache.",
                "This keeps retrieval fixed while comparing downstream ranking and evidence-use modules.",
            ],
        },
        "run_root": str(run_root),
        "config": str(base_cfg_path),
        "eval_path": str(cfg_run["experiment"]["eval_path"]),
        "model": cfg_run["llm"],
        "runs": runs,
    }
    summary_path = run_root / "analysis" / "ablation_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps({"summary_path": str(summary_path), **summary}, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run cumulative ablation for general search-RAG.")
    parser.add_argument("--config", default="configs/config_openai_general_rag.yaml")
    parser.add_argument("--out-root", default="outputs/general_ablation")
    parser.add_argument("--tag", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--provider", default="")
    parser.add_argument("--model", default="")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--no-full-general", action="store_true")
    run(parser.parse_args())
