import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.stats_report import analyze_pair, load_jsonl


def run(args):
    base_rows = load_jsonl(args.base)
    selective_rows = load_jsonl(args.kb_selective)

    report = {
        "comparison_kb_selective_vs_base": analyze_pair(
            base_rows, selective_rows, n_boot=args.n_boot, n_perm=args.n_perm
        )
    }

    if args.kb_non_selective:
        non_rows = load_jsonl(args.kb_non_selective)
        report["comparison_kb_non_selective_vs_base"] = analyze_pair(
            base_rows, non_rows, n_boot=args.n_boot, n_perm=args.n_perm
        )
        report["comparison_kb_selective_vs_non_selective"] = analyze_pair(
            non_rows, selective_rows, n_boot=args.n_boot, n_perm=args.n_perm
        )

    out_dir = os.path.dirname(args.out) or "."
    os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps({"out": args.out, **report}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Statistical report for strict KB-grounding experiments.")
    parser.add_argument("--base", required=True, help="Baseline predictions jsonl (vanilla_predictions.jsonl)")
    parser.add_argument("--kb-selective", required=True, help="Selective KB predictions jsonl")
    parser.add_argument("--kb-non-selective", default="", help="Optional non-selective KB predictions jsonl")
    parser.add_argument("--out", default="outputs/strict_kb/kb_stats_report.json")
    parser.add_argument("--n-boot", type=int, default=5000)
    parser.add_argument("--n-perm", type=int, default=10000)
    args = parser.parse_args()
    run(args)
