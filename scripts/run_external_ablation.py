import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

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
from scripts.run_general_ablation import select_search_variants


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


def model_output_dir_name(model_name: str) -> str:
    safe = str(model_name or "model").strip().replace("/", "_").replace("\\", "_").replace(" ", "_")
    if safe.lower().startswith("gpt-"):
        safe = "GPT-" + safe[4:]
    return f"{safe}_output"


def experiment_dir_name(cfg: Dict[str, Any]) -> str:
    eval_path = str(((cfg.get("experiment", {}) or {}).get("eval_path", "") or "")).strip()
    stem = Path(eval_path).stem
    return stem or "experiment"


def resolve_run_root(base_dir: str, cfg: Dict[str, Any], tag: str) -> Path:
    if str(tag or "").strip():
        return Path(base_dir) / str(tag).strip()
    model_dir = model_output_dir_name(str((cfg.get("llm", {}) or {}).get("model", "")))
    exp_dir = experiment_dir_name(cfg)
    return Path(base_dir) / model_dir / exp_dir


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


def infer_primary_metric(rows: List[Dict[str, Any]]) -> Tuple[str, str, str]:
    if not rows:
        return "unknown", "score", "Score"
    datasets = {str(row.get("dataset", "unknown")) for row in rows}
    if len(datasets) != 1:
        raise ValueError(f"Expected a single dataset in external run, got: {sorted(datasets)}")
    task_types = {str(row.get("task_type", "") or "").strip().lower() for row in rows}
    has_answer = all("answer" in row for row in rows)
    has_biased_answer = all("biased_answer" in row for row in rows)
    has_answers = all("answers" in row for row in rows)
    dataset = next(iter(datasets))
    if task_types and task_types.issubset({"honest_completion", "honest_generation"}):
        return dataset, "honest_score", "HONEST Score (Lower Better)"
    if task_types and task_types == {"geopolitical_mcq"}:
        return dataset, "controller_match_rate", "Controller Match Rate"
    if task_types and task_types == {"ethical_pair_mcq"}:
        return dataset, "acceptable_rate", "Acceptable Choice Rate"
    if has_answer:
        return dataset, "accuracy", "Accuracy"
    if has_biased_answer:
        return dataset, "non_biased_rate", "Non-Biased Rate"
    if has_answers:
        return dataset, "containment_match", "Containment Match"
    return dataset, "score", "Score"


def primary_score_from_metrics(metrics: Dict[str, Any], run_name: str, dataset: str, metric_name: str) -> float:
    eval_key = "vanilla_eval" if run_name == "vanilla" else "search_eval"
    dataset_metrics = (((metrics or {}).get(eval_key, {}) or {}).get("dataset_metrics", {}) or {}).get(dataset, {}) or {}
    value = dataset_metrics.get(metric_name, 0.0)
    try:
        return float(value)
    except Exception:
        return 0.0


def secondary_metrics_from_metrics(metrics: Dict[str, Any], run_name: str, dataset: str) -> Dict[str, Any]:
    eval_key = "vanilla_eval" if run_name == "vanilla" else "search_eval"
    return ((((metrics or {}).get(eval_key, {}) or {}).get("dataset_metrics", {}) or {}).get(dataset, {}) or {})


def plot_overall(
    overall: Dict[str, float],
    out_path: str,
    ylabel: str,
    title: str,
    lower_is_better: bool = False,
) -> None:
    labels = list(overall.keys())
    vals = [overall[k] for k in labels]
    plt.figure(figsize=(8, 4.5))
    if lower_is_better:
        ranked = sorted(zip(labels, vals), key=lambda x: x[1])
        labels = [x[0] for x in ranked]
        vals = [x[1] for x in ranked]
        bars = plt.barh(labels, vals)
        plt.xlim(0, 1.05)
        plt.xlabel(ylabel)
        plt.title(f"{title}\nLower is better")
        plt.gca().invert_yaxis()
        for bar, val in zip(bars, vals):
            plt.text(val + 0.02, bar.get_y() + bar.get_height() / 2, f"{val:.3f}", va="center", fontsize=9)
    else:
        bars = plt.bar(labels, vals)
        plt.ylim(0, 1.05)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.xticks(rotation=20, ha="right")
        for bar, val in zip(bars, vals):
            plt.text(bar.get_x() + bar.get_width() / 2, val + 0.02, f"{val:.3f}", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def summarize_run(output_dir: Path, name: str, params: Dict[str, Any], dataset: str, metric_name: str, expected_n: int) -> Dict[str, Any]:
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
        "primary_metric": metric_name,
        "primary_score": primary_score_from_metrics(metrics, name, dataset, metric_name),
        "dataset_metrics": secondary_metrics_from_metrics(metrics, name, dataset),
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


def run(args: argparse.Namespace) -> None:
    stage = str(getattr(args, "stage", "all") or "all").strip().lower()
    if stage not in {"all", "prepare", "run"}:
        raise ValueError(f"Unsupported stage: {stage}")
    base_cfg = load_yaml(args.config)
    cfg_run = copy.deepcopy(base_cfg)
    if args.provider:
        cfg_run["llm"]["provider"] = args.provider
    if args.model:
        cfg_run["llm"]["model"] = args.model
    cfg_run["llm"]["temperature"] = float(args.temperature)
    if args.search_pipeline_type:
        cfg_run.setdefault("search_grounding", {})
        cfg_run["search_grounding"]["search_pipeline_type"] = str(args.search_pipeline_type).strip().lower()
    run_root = resolve_run_root(args.out_root, cfg_run, args.tag)
    ensure_dir(str(run_root))
    ensure_dir(str(run_root / "configs"))
    ensure_dir(str(run_root / "caches"))
    ensure_dir(str(run_root / "runs"))
    ensure_dir(str(run_root / "analysis"))

    subset_eval = run_root / "eval_subset.jsonl"
    cfg_run["experiment"]["eval_path"] = prepare_subset(
        str(cfg_run["experiment"]["eval_path"]),
        str(subset_eval),
        args.limit,
    )

    eval_rows = load_jsonl(str(cfg_run["experiment"]["eval_path"]), strict=True)
    dataset, metric_name, metric_label = infer_primary_metric(eval_rows)
    expected_n = len(eval_rows)

    base_cfg_path = run_root / "configs" / "config_base.yaml"
    write_yaml(str(base_cfg_path), cfg_run)

    vanilla_cfg = copy.deepcopy(cfg_run)
    vanilla_out = run_root / "runs" / "vanilla"
    vanilla_cfg["experiment"]["output_dir"] = str(vanilla_out)
    vanilla_cfg_path = run_root / "configs" / "config_vanilla.yaml"
    write_yaml(str(vanilla_cfg_path), vanilla_cfg)

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

        need_refresh = retrieval_key not in frozen_for_key or args.refresh_cache or not cache_file_has_content(cache_path) or not cache_matches
        if stage in {"all", "prepare"} and need_refresh:
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
            if stage == "run" and not cache_file_has_content(cache_path):
                raise RuntimeError(
                    f"Missing frozen cache for retrieval_key={retrieval_key}: {cache_path}. "
                    "Run this script with --stage prepare first."
                )
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

    if stage == "prepare":
        prep_summary = {
            "stage": "prepare",
            "run_root": str(run_root),
            "config": str(base_cfg_path),
            "eval_path": str(cfg_run["experiment"]["eval_path"]),
            "dataset": dataset,
            "primary_metric": metric_name,
            "model": cfg_run["llm"],
            "cache_paths": {str(v["retrieval_key"]): str(run_root / "caches" / f"{v['retrieval_key']}.jsonl") for v in variants},
            "variants": [str(v["name"]) for v in variants],
        }
        print(json.dumps(prep_summary, ensure_ascii=True, indent=2))
        return

    runs: List[Dict[str, Any]] = []
    if not run_is_complete(vanilla_out, "vanilla", expected_n):
        run_cmd([sys.executable, "-m", "src.main", "--mode", "vanilla", "--config", str(vanilla_cfg_path)])
    runs.append(summarize_run(vanilla_out, "vanilla", {"mode": "vanilla"}, dataset, metric_name, expected_n))

    for variant in variants:
        name = str(variant["name"])
        params = dict(variant["params"])
        retrieval_key = str(variant["retrieval_key"])
        search_out = run_root / "runs" / name
        search_cfg_path = run_root / "configs" / f"config_{name}.yaml"
        if not run_is_complete(search_out, name, expected_n):
            run_cmd([sys.executable, "-m", "src.main", "--mode", "search", "--config", str(search_cfg_path)])
        run_summary = summarize_run(search_out, name, params, dataset, metric_name, expected_n)
        run_summary["description"] = str(variant.get("description", "") or "").strip()
        run_summary["retrieval_key"] = retrieval_key
        run_summary["cache_path"] = str(run_root / "caches" / f"{retrieval_key}.jsonl")
        runs.append(run_summary)

    overall = {r["name"]: r["primary_score"] for r in runs}
    plot_overall(
        overall,
        str(run_root / "analysis" / "primary_metric_ablation.png"),
        ylabel=metric_label,
        title=f"{dataset} Ablation ({metric_label})",
        lower_is_better=(metric_name == "honest_score"),
    )

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
        "dataset": dataset,
        "primary_metric": metric_name,
        "primary_metric_label": metric_label,
        "model": cfg_run["llm"],
        "runs": runs,
    }
    summary_path = run_root / "analysis" / "ablation_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps({"summary_path": str(summary_path), **summary}, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run cumulative ablation for external benchmarks with task-specific metrics.")
    parser.add_argument("--config", default="configs/external/config_bbq_200.yaml")
    parser.add_argument("--out-root", default="outputs")
    parser.add_argument("--tag", default="")
    parser.add_argument("--stage", default="all", choices=["all", "prepare", "run"])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--provider", default="")
    parser.add_argument("--model", default="")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--search-pipeline-type", default="")
    parser.add_argument("--no-full-general", action="store_true")
    run(parser.parse_args())
