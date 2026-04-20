import argparse
import copy
import json
import os
import random
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline import dump_jsonl, load_jsonl


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def run_cmd(cmd):
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True)


def prepare_subset(eval_path: str, out_path: str, limit: int, stratify_by: str = "", seed: int = 42) -> str:
    if limit <= 0:
        return eval_path

    rows = load_jsonl(eval_path)
    if not rows:
        dump_jsonl(out_path, [])
        return out_path

    if stratify_by:
        groups = defaultdict(list)
        for r in rows:
            if stratify_by in {"dataset+answer", "dataset_answer"}:
                key = f"{r.get('dataset', 'unknown')}::{r.get('answer', 'unknown')}"
            else:
                key = str(r.get(stratify_by, "unknown"))
            groups[key].append(r)

        rng = random.Random(seed)
        keys = sorted(groups.keys())
        for k in keys:
            rng.shuffle(groups[k])

        selected = []
        per_group = max(1, limit // max(1, len(keys)))
        for k in keys:
            selected.extend(groups[k][:per_group])

        if len(selected) < limit:
            leftovers = []
            for k in keys:
                leftovers.extend(groups[k][per_group:])
            rng.shuffle(leftovers)
            selected.extend(leftovers[: max(0, limit - len(selected))])

        rows = selected[:limit]
    else:
        rows = rows[:limit]

    dump_jsonl(out_path, rows)
    return out_path


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


def run(args):
    with open(args.config, "r", encoding="utf-8-sig") as f:
        base_cfg = yaml.safe_load(f)

    out_root = Path(args.out_root)
    ensure_dir(str(out_root))

    subset_eval = out_root / "eval_subset.jsonl"
    eval_path = prepare_subset(
        base_cfg["experiment"]["eval_path"],
        str(subset_eval),
        args.limit,
        stratify_by=args.stratify_by,
        seed=args.seed,
    )

    cache_path = out_root / "search_cache.jsonl"
    if args.refresh_cache or not cache_file_has_content(cache_path):
        freeze_cfg = copy.deepcopy(base_cfg)
        freeze_cfg["experiment"]["eval_path"] = eval_path
        if args.provider:
            freeze_cfg["llm"]["provider"] = args.provider
        if args.model:
            freeze_cfg["llm"]["model"] = args.model
        freeze_cfg["llm"]["temperature"] = float(args.temperature)

        freeze_cfg_path = out_root / "config_freeze.yaml"
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

    matrix = [
        {"name": "vanilla", "mode": "vanilla", "selection_mode": None},
        {"name": "search_general", "mode": "search", "selection_mode": None},
    ]

    summary = {"runs": {}, "cache_path": str(cache_path), "eval_path": str(eval_path)}

    for run_def in matrix:
        cfg = copy.deepcopy(base_cfg)
        cfg["experiment"]["eval_path"] = eval_path
        run_out = out_root / run_def["name"]
        cfg["experiment"]["output_dir"] = str(run_out)

        if args.provider:
            cfg["llm"]["provider"] = args.provider
        if args.model:
            cfg["llm"]["model"] = args.model
        cfg["llm"]["temperature"] = float(args.temperature)

        if run_def["mode"] == "search":
            scfg = cfg["search_grounding"]
            scfg["cache_path"] = str(cache_path)
            scfg["use_cache_only"] = True
            if run_def["selection_mode"] is not None:
                scfg["selection_mode"] = run_def["selection_mode"]
            scfg["include_candidate_details"] = False

        cfg_path = out_root / f"config_{run_def['name']}.yaml"
        write_yaml(str(cfg_path), cfg)

        run_cmd([sys.executable, "-m", "src.main", "--mode", run_def["mode"], "--config", str(cfg_path)])

        metrics_path = run_out / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path, "r", encoding="utf-8") as f:
                summary["runs"][run_def["name"]] = json.load(f)
        else:
            summary["runs"][run_def["name"]] = {"error": "metrics not found"}

    summary_path = out_root / "matrix_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps({"summary_path": str(summary_path), **summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run strict search-grounding matrix with frozen retrieval cache.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--out-root", default="outputs/strict_search")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--stratify-by", default="dataset+answer")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--provider", default="")
    parser.add_argument("--model", default="")
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()
    run(args)
