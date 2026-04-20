import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import yaml


def load_jsonl(path: str) -> List[Dict]:
    rows: List[Dict] = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def dump_jsonl(path: str, rows: List[Dict]) -> None:
    out_dir = os.path.dirname(path) or "."
    os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def run_cmd(cmd):
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def write_yaml(path: str, cfg: Dict) -> None:
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


def now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def prepare_subset(eval_path: str, out_path: str, limit: int) -> str:
    if limit <= 0:
        return eval_path
    rows = load_jsonl(eval_path)
    rows = rows[:limit]
    dump_jsonl(out_path, rows)
    return out_path


def run(args: argparse.Namespace) -> None:
    with open(args.config, "r", encoding="utf-8-sig") as f:
        cfg = yaml.safe_load(f)

    scfg = cfg.get("search_grounding", {}) or {}

    root = Path(args.out_root)
    run_root = root / (args.tag.strip() if args.tag.strip() else now_tag())
    ensure_dir(str(run_root))

    cfg_run = dict(cfg)
    cfg_run["experiment"] = dict(cfg.get("experiment", {}))
    cfg_run["llm"] = dict(cfg.get("llm", {}))
    cfg_run["search_grounding"] = dict(scfg)

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

    cfg_path = run_root / "config_run.yaml"
    write_yaml(str(cfg_path), cfg_run)

    cache_path = run_root / "search_cache.jsonl"
    matrix_root = run_root / "matrix"
    analysis_root = run_root / "analysis"
    ensure_dir(str(matrix_root))
    ensure_dir(str(analysis_root))

    cache_ok = cache_file_has_content(cache_path)
    if args.refresh_cache or not cache_ok:
        run_cmd(
            [
                sys.executable,
                "scripts/freeze_search_cache.py",
                "--config",
                str(cfg_path),
                "--out-cache",
                str(cache_path),
                "--limit",
                "0",
            ]
        )
    matrix_cache_path = matrix_root / "search_cache.jsonl"
    matrix_cache_ok = cache_file_has_content(matrix_cache_path)
    if cache_file_has_content(cache_path) and (args.refresh_cache or not matrix_cache_ok):
        shutil.copyfile(cache_path, matrix_cache_path)

    run_cmd(
        [
            sys.executable,
            "scripts/run_matrix.py",
            "--config",
            str(cfg_path),
            "--out-root",
            str(matrix_root),
            "--limit",
            "0",
            "--provider",
            str(cfg_run["llm"].get("provider", "")),
            "--model",
            str(cfg_run["llm"].get("model", "")),
            "--temperature",
            str(float(cfg_run["llm"].get("temperature", 0.0))),
        ]
    )

    run_cmd(
        [
            sys.executable,
            "scripts/analyze_matrix.py",
            "--matrix-root",
            str(matrix_root),
            "--out-dir",
            str(analysis_root),
        ]
    )

    vanilla_usage = matrix_root / "vanilla" / "llm_usage_vanilla.jsonl"
    search_usage = matrix_root / "search_general" / "llm_usage_search.jsonl"
    usage_args = []
    if vanilla_usage.exists():
        usage_args.append(f"vanilla={vanilla_usage}")
    if search_usage.exists():
        usage_args.append(f"search_general={search_usage}")
    if usage_args:
        run_cmd(
            [
                sys.executable,
                "scripts/visualize_token_usage.py",
                "--usage",
                *usage_args,
                "--out-dir",
                str(analysis_root / "token_usage"),
            ]
        )

    summary = {
        "run_root": str(run_root),
        "config_used": str(cfg_path),
        "eval_path": str(cfg_run["experiment"]["eval_path"]),
        "cache_path": str(cache_path),
        "matrix_root": str(matrix_root),
        "analysis_root": str(analysis_root),
        "pipeline_variant": "general",
        "steps": [
            "freeze_search_cache",
            "run_matrix(vanilla + search_general)",
            "analyze_matrix",
            "visualize_token_usage",
        ],
    }
    summary_path = run_root / "workflow_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run stable general RAG workflow end-to-end.")
    parser.add_argument("--config", default="configs/config_openai_general_rag.yaml")
    parser.add_argument("--out-root", default="outputs/general_workflow")
    parser.add_argument("--tag", default="", help="optional fixed run tag folder name")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--provider", default="")
    parser.add_argument("--model", default="")
    parser.add_argument("--temperature", type=float, default=0.0)
    run(parser.parse_args())
