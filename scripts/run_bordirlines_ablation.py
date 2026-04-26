import argparse
import copy
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline import jsonl_integrity_summary, load_jsonl, dump_jsonl


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def run_cmd(cmd: List[str]) -> None:
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


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
    return stem or "bordirlines"


def resolve_run_root(base_dir: str, cfg: Dict[str, Any], tag: str) -> Path:
    if str(tag or "").strip():
        return Path(base_dir) / str(tag).strip()
    model_dir = model_output_dir_name(str((cfg.get("llm", {}) or {}).get("model", "")))
    exp_dir = experiment_dir_name(cfg)
    return Path(base_dir) / model_dir / exp_dir


def write_yaml(path: Path, cfg: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)


def now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def prepare_subset(eval_path: str, out_path: Path, limit: int) -> str:
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
    dump_jsonl(str(out_path), rows)
    return str(out_path)


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


def extract_doc_id(url: str) -> str:
    if not url.startswith("bordirlines://"):
        return ""
    return url.rsplit("/", 1)[-1]


def load_cache_by_id(path: str) -> Dict[str, Dict[str, Any]]:
    rows = load_jsonl(path, strict=True)
    return {str(row.get("id", "")): row for row in rows if str(row.get("id", "")).strip()}


def retrieval_metrics(pred_rows: List[Dict[str, Any]], cache_by_id: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    if not pred_rows:
        return {
            "selected_precision": 0.0,
            "selected_recall": 0.0,
            "avg_selected_docs": 0.0,
            "selected_doc_lang_distribution": {},
        }

    precision_scores: List[float] = []
    recall_scores: List[float] = []
    selected_sizes: List[int] = []
    doc_lang_counts: Dict[str, int] = {}

    for row in pred_rows:
        cache_row = cache_by_id.get(str(row.get("id", "")), {})
        candidates = cache_row.get("raw_candidate_evidence", []) or []
        candidate_meta = {str(c.get("doc_id", "")): c for c in candidates if str(c.get("doc_id", "")).strip()}
        relevant_ids = {doc_id for doc_id, meta in candidate_meta.items() if bool(meta.get("relevant", False))}
        selected = ((row.get("search_trace", {}) or {}).get("selected_evidence", []) or [])
        selected_doc_ids = [extract_doc_id(str(ev.get("url", "") or "")) for ev in selected]
        selected_doc_ids = [doc_id for doc_id in selected_doc_ids if doc_id]
        if selected_doc_ids:
            relevant_selected = sum(1 for doc_id in selected_doc_ids if doc_id in relevant_ids)
            precision_scores.append(relevant_selected / len(selected_doc_ids))
            selected_sizes.append(len(selected_doc_ids))
            if relevant_ids:
                recall_scores.append(relevant_selected / len(relevant_ids))
            for doc_id in selected_doc_ids:
                lang = str(candidate_meta.get(doc_id, {}).get("doc_lang", "unknown") or "unknown")
                doc_lang_counts[lang] = doc_lang_counts.get(lang, 0) + 1
        elif relevant_ids:
            precision_scores.append(0.0)
            recall_scores.append(0.0)
            selected_sizes.append(0)

    total_lang = sum(doc_lang_counts.values())
    lang_dist = {lang: count / total_lang for lang, count in sorted(doc_lang_counts.items())} if total_lang else {}
    return {
        "selected_precision": (sum(precision_scores) / len(precision_scores)) if precision_scores else 0.0,
        "selected_recall": (sum(recall_scores) / len(recall_scores)) if recall_scores else 0.0,
        "avg_selected_docs": (sum(selected_sizes) / len(selected_sizes)) if selected_sizes else 0.0,
        "selected_doc_lang_distribution": lang_dist,
    }


def fixed_variants(base_scfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    default_domains = list(base_scfg.get("low_quality_domains", []) or [])
    default_keywords = list(base_scfg.get("low_quality_url_keywords", []) or [])
    common = {
        "pipeline_variant": "general",
        "strict_feature_checks": True,
        "use_cache_only": True,
        "llm_query_rewrite": False,
        "query_expansion_n": 1,
    }
    return [
        {
            "name": "fixed_candidate_rag",
            "description": "Replay official fixed candidates with lexical ranking only.",
            "params": {
                **common,
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
            "name": "fixed_candidate_semantic_rag",
            "description": "Fixed candidates plus embedding pre-rank and cross-encoder rerank.",
            "params": {
                **common,
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
            "name": "fixed_candidate_semantic_noise_filter",
            "description": "Fixed candidates plus semantic ranking and noise filtering.",
            "params": {
                **common,
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
            "name": "full_fixed_candidate_rag",
            "description": "Fixed candidates plus semantic ranking, noise filtering, evidence organization, and gate.",
            "params": {
                **common,
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


def summarize_run(output_dir: Path, name: str, params: Dict[str, Any], expected_n: int, cache_by_id: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
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
        with metrics_path.open("r", encoding="utf-8") as f:
            metrics = json.load(f)
    dataset_metrics = (((metrics.get("vanilla_eval") if name == "vanilla" else metrics.get("search_eval")) or {}).get("dataset_metrics", {}) or {})
    dataset_name = next(iter(dataset_metrics.keys()), "")
    retrieval = retrieval_metrics(rows, cache_by_id) if name != "vanilla" else {}
    score = float((((dataset_metrics.get(dataset_name, {}) or {}).get("controller_match_rate", 0.0)) if dataset_name else 0.0))
    return {
        "name": name,
        "output_dir": str(output_dir),
        "params": params,
        "n": len(rows),
        "expected_n": expected_n,
        "prediction_integrity": integrity,
        "controller_match_rate": score,
        "dataset_metrics": (dataset_metrics.get(dataset_name, {}) if dataset_name else {}),
        "retrieval_metrics": retrieval,
        "metrics": metrics,
    }


def plot_overall(overall: Dict[str, float], out_path: Path) -> None:
    labels = list(overall.keys())
    vals = [overall[k] for k in labels]
    plt.figure(figsize=(8, 4.5))
    bars = plt.bar(labels, vals)
    plt.ylim(0, 1.05)
    plt.ylabel("Controller Match Rate")
    plt.title("BordIRLines Fixed-Candidate Ablation")
    plt.xticks(rotation=20, ha="right")
    for bar, val in zip(bars, vals):
        plt.text(bar.get_x() + bar.get_width() / 2, val + 0.02, f"{val:.3f}", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def run(args: argparse.Namespace) -> None:
    base_cfg = load_yaml(args.config)
    cfg_run = copy.deepcopy(base_cfg)
    if args.provider:
        cfg_run["llm"]["provider"] = args.provider
    if args.model:
        cfg_run["llm"]["model"] = args.model
    cfg_run["llm"]["temperature"] = float(args.temperature)
    run_root = resolve_run_root(args.out_root, cfg_run, args.tag)
    for subdir in ["configs", "runs", "analysis"]:
        ensure_dir(run_root / subdir)
    subset_eval = run_root / "eval_subset.jsonl"
    cfg_run["experiment"]["eval_path"] = prepare_subset(str(cfg_run["experiment"]["eval_path"]), subset_eval, args.limit)
    expected_n = len(load_jsonl(str(cfg_run["experiment"]["eval_path"]), strict=True))
    cache_by_id = load_cache_by_id(str(cfg_run["search_grounding"]["cache_path"]))

    base_cfg_path = run_root / "configs" / "config_base.yaml"
    write_yaml(base_cfg_path, cfg_run)

    runs: List[Dict[str, Any]] = []

    vanilla_cfg = copy.deepcopy(cfg_run)
    vanilla_out = run_root / "runs" / "vanilla"
    vanilla_cfg["experiment"]["output_dir"] = str(vanilla_out)
    vanilla_cfg_path = run_root / "configs" / "config_vanilla.yaml"
    write_yaml(vanilla_cfg_path, vanilla_cfg)
    if not run_is_complete(vanilla_out, "vanilla", expected_n):
        run_cmd([sys.executable, "-m", "src.main", "--mode", "vanilla", "--config", str(vanilla_cfg_path)])
    runs.append(summarize_run(vanilla_out, "vanilla", {"mode": "vanilla"}, expected_n, cache_by_id))

    for variant in fixed_variants(cfg_run.get("search_grounding", {}) or {}):
        name = str(variant["name"])
        params = dict(variant["params"])
        search_cfg = copy.deepcopy(cfg_run)
        search_out = run_root / "runs" / name
        search_cfg["experiment"]["output_dir"] = str(search_out)
        search_cfg["search_grounding"].update(params)
        search_cfg["search_grounding"]["use_cache_only"] = True
        search_cfg["search_grounding"]["include_candidate_details"] = False
        search_cfg_path = run_root / "configs" / f"config_{name}.yaml"
        write_yaml(search_cfg_path, search_cfg)
        if not run_is_complete(search_out, name, expected_n):
            run_cmd([sys.executable, "-m", "src.main", "--mode", "search", "--config", str(search_cfg_path)])
        run_summary = summarize_run(search_out, name, params, expected_n, cache_by_id)
        run_summary["description"] = str(variant.get("description", "") or "")
        runs.append(run_summary)

    overall = {run["name"]: run["controller_match_rate"] for run in runs}
    plot_overall(overall, run_root / "analysis" / "controller_match_rate_ablation.png")

    summary = {
        "run_root": str(run_root),
        "config": str(base_cfg_path),
        "runs": runs,
    }
    summary_path = run_root / "analysis" / "ablation_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps({"summary_path": str(summary_path), **summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run fixed-candidate BordIRLines ablations without live search.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--out-root", default="outputs")
    parser.add_argument("--tag", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--provider", default="")
    parser.add_argument("--model", default="")
    parser.add_argument("--temperature", type=float, default=0.0)
    run(parser.parse_args())
