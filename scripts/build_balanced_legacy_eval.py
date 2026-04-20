import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from prepare_benchmarks import (
    convert_blend,
    convert_normad,
    convert_seegull,
    dedup,
    ensure_dir,
    rebalance_by_dataset,
    write_jsonl,
)


def run(args: argparse.Namespace) -> None:
    raw_root = Path(args.raw_root)
    out_path = Path(args.out)
    ensure_dir(raw_root)

    blend_rows, blend_report = convert_blend(0, args.seed)
    normad_rows, normad_report = convert_normad(0, args.seed)
    seegull_rows, seegull_report = convert_seegull(raw_root, 0, args.seed)

    all_rows: List[Dict[str, Any]] = []
    all_rows.extend(blend_rows)
    all_rows.extend(normad_rows)
    all_rows.extend(seegull_rows)

    deduped = dedup(all_rows)
    final_rows = rebalance_by_dataset(deduped, args.max_per_dataset, seed=args.seed, label_key="answer")
    write_jsonl(out_path, final_rows)

    by_dataset: Dict[str, int] = {}
    for row in final_rows:
        ds = str(row.get("dataset", "unknown"))
        by_dataset[ds] = by_dataset.get(ds, 0) + 1

    summary: Dict[str, Any] = {
        "output": str(out_path),
        "seed": args.seed,
        "max_per_dataset": args.max_per_dataset,
        "counts": by_dataset,
        "source_reports": {
            "blend": blend_report,
            "normad": normad_report,
            "seegull": seegull_report,
        },
        "note": "Built by full conversion -> global dedup -> per-dataset balanced sampling.",
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build a strict balanced legacy benchmark set.")
    parser.add_argument("--raw-root", default="data/benchmarks/raw")
    parser.add_argument("--out", default="data/eval_balanced_200_strict.jsonl")
    parser.add_argument("--max-per-dataset", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run(args)
