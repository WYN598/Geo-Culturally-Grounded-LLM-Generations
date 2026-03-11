import argparse
import json
from typing import Any, Dict, List, Set


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def dump_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def run(args: argparse.Namespace) -> None:
    eval_rows = load_jsonl(args.eval_path)
    pred_rows = load_jsonl(args.pred_path)
    keep_ids: Set[str] = {str(r.get("id", "")).strip() for r in eval_rows if str(r.get("id", "")).strip()}

    out_rows = [r for r in pred_rows if str(r.get("id", "")).strip() in keep_ids]
    dump_jsonl(args.out_path, out_rows)

    report = {
        "eval_path": args.eval_path,
        "pred_path": args.pred_path,
        "out_path": args.out_path,
        "n_eval_ids": len(keep_ids),
        "n_pred_in": len(pred_rows),
        "n_pred_out": len(out_rows),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter prediction jsonl rows by eval IDs.")
    parser.add_argument("--eval-path", required=True)
    parser.add_argument("--pred-path", required=True)
    parser.add_argument("--out-path", required=True)
    run(parser.parse_args())
