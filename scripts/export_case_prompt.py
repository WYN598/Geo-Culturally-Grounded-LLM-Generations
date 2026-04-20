import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline import build_augmented_prompt, build_grounded_answer_prompt, format_mcq_prompt


def load_jsonl(path: str):
    rows = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Export the exact model prompt for a case id.")
    parser.add_argument("--preds", required=True, help="Prediction jsonl file")
    parser.add_argument("--id", required=True, help="Case id")
    parser.add_argument("--out", required=True, help="Output txt file")
    args = parser.parse_args()

    rows = load_jsonl(args.preds)
    target = None
    for row in rows:
        if str(row.get("id", "")).strip() == args.id:
            target = row
            break

    if target is None:
        raise SystemExit(f"case id not found: {args.id}")

    evidence = list(target.get("evidence", []) or [])
    trace = dict(target.get("search_trace", {}) or {})
    organized = list(trace.get("organized_evidence", []) or [])
    if organized:
        prompt = build_grounded_answer_prompt(target["question"], target["choices"], organized)
        stage = "search_answer_augmented"
    elif evidence:
        prompt = build_augmented_prompt(target["question"], target["choices"], evidence)
        stage = "search_answer_augmented"
    else:
        prompt = format_mcq_prompt(target["question"], target["choices"])
        stage = "search_answer_fallback"

    out_dir = os.path.dirname(args.out) or "."
    os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(prompt)

    meta = {
        "id": target.get("id", ""),
        "dataset": target.get("dataset", ""),
        "stage": stage,
        "pred": target.get("pred", ""),
        "answer": target.get("answer", ""),
        "out": args.out,
    }
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
