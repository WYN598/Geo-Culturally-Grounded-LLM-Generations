import argparse
import json
import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Set, Tuple


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


def norm_text(s: str) -> str:
    s = str(s or "").strip().lower()
    s = re.sub(r"^[a-z]\s*[\)\.]\s*", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


def answer_text(row: Dict[str, Any]) -> str:
    ans = str(row.get("answer", "")).strip().upper()
    choices = list(row.get("choices", []) or [])
    idx = ord(ans) - ord("A") if ans else -1
    if 0 <= idx < len(choices):
        return norm_text(choices[idx])
    return ""


def is_temporal(question: str) -> bool:
    temporal_kw = re.compile(
        r"\b(latest|current|today|newest|as of|this year|recent|highest|largest|most famous)\b",
        flags=re.IGNORECASE,
    )
    return bool(temporal_kw.search(str(question or "")))


def group_by_question(rows: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    g: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        g[norm_text(r.get("question", ""))].append(r)
    return g


def pick_representative(group: List[Dict[str, Any]]) -> Dict[str, Any]:
    # deterministic: shortest id then lexicographically smallest id
    return sorted(group, key=lambda x: (len(str(x.get("id", ""))), str(x.get("id", ""))))[0]


def run(args: argparse.Namespace) -> None:
    rows = load_jsonl(args.eval_path)
    q_groups = group_by_question(rows)

    kept: List[Dict[str, Any]] = []
    dropped_ids: Set[str] = set()
    dropped_reasons: Dict[str, str] = {}

    conflicting_questions: List[Tuple[str, List[str], List[str]]] = []

    for q, grp in q_groups.items():
        if not q:
            for r in grp:
                rid = str(r.get("id", ""))
                dropped_ids.add(rid)
                dropped_reasons[rid] = "empty_question"
            continue

        ans_texts = sorted({answer_text(x) for x in grp if answer_text(x)})
        ds_set = sorted({str(x.get("dataset", "")) for x in grp})
        has_answer_text_conflict = len(ans_texts) > 1

        if has_answer_text_conflict and args.drop_answer_conflict_questions:
            conflicting_questions.append((q, ans_texts, [str(x.get("id", "")) for x in grp]))
            for r in grp:
                rid = str(r.get("id", ""))
                dropped_ids.add(rid)
                dropped_reasons[rid] = "duplicate_question_answer_conflict"
            continue

        # dedup same question (keep one representative)
        if args.dedup_by_question and len(grp) > 1:
            rep = pick_representative(grp)
            kept.append(rep)
            rep_id = str(rep.get("id", ""))
            for r in grp:
                rid = str(r.get("id", ""))
                if rid != rep_id:
                    dropped_ids.add(rid)
                    dropped_reasons[rid] = "duplicate_question_dedup"
            continue

        kept.extend(grp)

    final_rows: List[Dict[str, Any]] = []
    for r in kept:
        rid = str(r.get("id", ""))
        if args.drop_temporal_questions and is_temporal(str(r.get("question", ""))):
            dropped_ids.add(rid)
            dropped_reasons[rid] = "temporal_or_volatile_question"
            continue
        final_rows.append(r)

    # stable ordering by original input sequence
    order = {str(r.get("id", "")): i for i, r in enumerate(rows)}
    final_rows = sorted(final_rows, key=lambda x: order.get(str(x.get("id", "")), 10**9))

    dump_jsonl(args.out_eval, final_rows)

    ds_counts = Counter(str(r.get("dataset", "unknown")) for r in final_rows)
    reason_counts = Counter(dropped_reasons.values())
    report = {
        "input_eval": args.eval_path,
        "output_eval": args.out_eval,
        "n_input": len(rows),
        "n_output": len(final_rows),
        "dataset_counts_output": dict(ds_counts),
        "drop_counts_by_reason": dict(reason_counts),
        "num_conflicting_duplicate_questions": len(conflicting_questions),
        "sample_conflicting_questions": [
            {"question": q, "answer_texts": ans_texts, "ids": ids[:10]}
            for q, ans_texts, ids in conflicting_questions[:10]
        ],
        "dropped_ids_path": args.out_dropped_ids,
    }

    with open(args.out_report, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    with open(args.out_dropped_ids, "w", encoding="utf-8") as f:
        for rid in sorted(dropped_ids):
            f.write(rid + "\n")

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build a cleaner benchmark subset by removing noisy question variants.")
    parser.add_argument("--eval-path", required=True)
    parser.add_argument("--out-eval", required=True)
    parser.add_argument("--out-report", required=True)
    parser.add_argument("--out-dropped-ids", required=True)
    parser.add_argument("--drop-answer-conflict-questions", action="store_true")
    parser.add_argument("--dedup-by-question", action="store_true")
    parser.add_argument("--drop-temporal-questions", action="store_true")
    run(parser.parse_args())
