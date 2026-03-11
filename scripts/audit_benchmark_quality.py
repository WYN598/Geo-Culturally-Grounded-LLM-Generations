import argparse
import json
import re
from collections import Counter, defaultdict
from typing import Any, Dict, List


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def norm_text(s: str) -> str:
    s = str(s or "").strip().lower()
    s = re.sub(r"^[a-z]\s*[\)\.]\s*", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


def has_mojibake(s: str) -> bool:
    # Common mojibake / bad decode indicators seen in multilingual corpora.
    patterns = [
        "�",
        "Ã",
        "Â",
        "â€",
        "鈥",
        "铆",
        "谩",
        "路",
    ]
    return any(p in s for p in patterns)


def run(args: argparse.Namespace) -> None:
    rows = load_jsonl(args.eval_path)

    ds_count = Counter(str(r.get("dataset", "unknown")) for r in rows)

    id_count = Counter(str(r.get("id", "")).strip() for r in rows)
    dup_ids = [k for k, v in id_count.items() if k and v > 1]

    q_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        q_groups[norm_text(r.get("question", ""))].append(r)
    q_count = Counter({k: len(v) for k, v in q_groups.items()})
    dup_questions = [k for k, v in q_count.items() if k and v > 1]
    dup_question_conflicts = []
    dup_question_answer_text_conflicts = []
    for q in dup_questions:
        grp = q_groups[q]
        ans_set = {str(x.get("answer", "")).strip().upper() for x in grp}
        choice_set = {tuple(norm_text(c) for c in (x.get("choices", []) or [])) for x in grp}
        ds_set = {str(x.get("dataset", "")) for x in grp}
        ans_text_set = set()
        for x in grp:
            choices = list(x.get("choices", []) or [])
            ans = str(x.get("answer", "")).strip().upper()
            idx = ord(ans) - ord("A") if ans else -1
            if 0 <= idx < len(choices):
                ans_text_set.add(norm_text(choices[idx]))
        detail = {
            "question": q,
            "n": len(grp),
            "datasets": sorted(ds_set),
            "answers": sorted(ans_set),
            "answer_texts": sorted(ans_text_set),
            "ids": [str(x.get("id", "")) for x in grp][:10],
        }
        if len(choice_set) > 1 or len(ds_set) > 1:
            dup_question_conflicts.append(detail)
        if len(ans_text_set) > 1:
            dup_question_answer_text_conflicts.append(detail)

    invalid_answer = []
    choice_collision = []
    encoding_suspect = []
    temporal_or_volatile = []

    temporal_kw = re.compile(
        r"\b(latest|current|today|newest|as of|this year|recent|highest|largest|most famous)\b",
        flags=re.IGNORECASE,
    )

    for r in rows:
        rid = str(r.get("id", "")).strip()
        question = str(r.get("question", ""))
        choices = [str(x) for x in (r.get("choices", []) or [])]
        ans = str(r.get("answer", "")).strip().upper()

        allowed = {chr(ord("A") + i) for i in range(len(choices))}
        if ans not in allowed:
            invalid_answer.append(rid)

        nchoices = [norm_text(c) for c in choices if norm_text(c)]
        if len(nchoices) != len(set(nchoices)):
            choice_collision.append(rid)

        if has_mojibake(question) or any(has_mojibake(c) for c in choices):
            encoding_suspect.append(rid)

        if temporal_kw.search(question):
            temporal_or_volatile.append(rid)

    by_dataset_issues: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    issue_sets = {
        "invalid_answer": set(invalid_answer),
        "choice_collision": set(choice_collision),
        "encoding_suspect": set(encoding_suspect),
        "temporal_or_volatile": set(temporal_or_volatile),
    }
    for r in rows:
        rid = str(r.get("id", "")).strip()
        ds = str(r.get("dataset", "unknown"))
        for k, s in issue_sets.items():
            if rid in s:
                by_dataset_issues[ds][k] += 1

    report = {
        "eval_path": args.eval_path,
        "n": len(rows),
        "dataset_counts": dict(ds_count),
        "duplicate_id_count": len(dup_ids),
        "duplicate_question_count": len(dup_questions),
        "duplicate_question_conflict_count": len(dup_question_conflicts),
        "duplicate_question_answer_text_conflict_count": len(dup_question_answer_text_conflicts),
        "invalid_answer_count": len(invalid_answer),
        "choice_collision_count": len(choice_collision),
        "encoding_suspect_count": len(encoding_suspect),
        "temporal_or_volatile_count": len(temporal_or_volatile),
        "by_dataset_issues": {k: dict(v) for k, v in sorted(by_dataset_issues.items())},
        "samples": {
            "duplicate_ids": dup_ids[:10],
            "duplicate_questions": dup_questions[:5],
            "duplicate_question_conflicts": dup_question_conflicts[:10],
            "duplicate_question_answer_text_conflicts": dup_question_answer_text_conflicts[:10],
            "invalid_answer_ids": invalid_answer[:20],
            "choice_collision_ids": choice_collision[:20],
            "encoding_suspect_ids": encoding_suspect[:20],
            "temporal_or_volatile_ids": temporal_or_volatile[:20],
        },
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audit benchmark quality and potential data issues.")
    parser.add_argument("--eval-path", required=True)
    parser.add_argument("--out", default="outputs/benchmark_audit.json")
    run(parser.parse_args())
