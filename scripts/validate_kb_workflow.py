import argparse
import json
import os
from collections import Counter, defaultdict
from typing import Dict, List, Tuple


def load_jsonl(path: str) -> List[Dict]:
    rows: List[Dict] = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def answer_distribution(rows: List[Dict]) -> Dict[str, Dict[str, int]]:
    out: Dict[str, Dict[str, int]] = {}
    ds_map: Dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        ds_map[str(r.get("dataset", "unknown"))][str(r.get("answer", ""))] += 1
    for ds, c in sorted(ds_map.items()):
        out[ds] = dict(c)
    return out


def invalid_prediction_count(rows: List[Dict]) -> Tuple[int, int]:
    bad = 0
    for r in rows:
        n = len(r.get("choices", []) or [])
        allowed = {chr(ord("A") + i) for i in range(n)}
        pred = str(r.get("pred", "")).strip().upper()
        if pred not in allowed:
            bad += 1
    return bad, len(rows)


def id_set(rows: List[Dict]) -> set:
    return {str(r.get("id", "")).strip() for r in rows if str(r.get("id", "")).strip()}


def run(args: argparse.Namespace) -> None:
    report: Dict[str, Dict] = {}
    eval_rows = load_jsonl(args.eval_path)
    report["eval"] = {
        "path": args.eval_path,
        "n": len(eval_rows),
        "dataset_counts": dict(Counter(str(r.get("dataset", "unknown")) for r in eval_rows)),
        "answer_distribution": answer_distribution(eval_rows),
    }

    preds = {}
    for name, path in [
        ("vanilla", args.vanilla_path),
        ("kb_selective", args.kb_selective_path),
        ("kb_non_selective", args.kb_non_selective_path),
    ]:
        if not path or not os.path.exists(path):
            continue
        rows = load_jsonl(path)
        bad, total = invalid_prediction_count(rows)
        preds[name] = {
            "path": path,
            "n": total,
            "invalid_pred_count": bad,
            "invalid_pred_rate": (bad / total) if total else 0.0,
        }
        if name.startswith("kb_"):
            z = sum(1 for r in rows if len((r.get("evidence", []) or [])) == 0)
            preds[name]["zero_evidence_count"] = z
            preds[name]["zero_evidence_rate"] = (z / total) if total else 0.0

    report["predictions"] = preds

    if preds:
        id_info = {}
        id_map = {k: id_set(load_jsonl(v["path"])) for k, v in preds.items()}
        names = sorted(id_map.keys())
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = names[i], names[j]
                inter = len(id_map[a] & id_map[b])
                id_info[f"{a}_vs_{b}"] = {
                    "shared_ids": inter,
                    "only_in_left": len(id_map[a] - id_map[b]),
                    "only_in_right": len(id_map[b] - id_map[a]),
                }
        report["id_alignment"] = id_info

    out_dir = os.path.dirname(args.out) or "."
    os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate KB workflow outputs for common data/eval pitfalls.")
    parser.add_argument("--eval-path", required=True)
    parser.add_argument("--vanilla-path", default="")
    parser.add_argument("--kb-selective-path", default="")
    parser.add_argument("--kb-non-selective-path", default="")
    parser.add_argument("--out", default="outputs/kb_validation_report.json")
    args = parser.parse_args()
    run(args)
