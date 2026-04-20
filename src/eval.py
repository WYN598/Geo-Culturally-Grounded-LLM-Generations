import re
from typing import Any, Dict, List, Set


def mcq_accuracy(rows: List[Dict]) -> float:
    if not rows:
        return 0.0
    correct = sum(1 for r in rows if (r.get("pred", "") == r.get("answer", "")))
    return correct / len(rows)


def _valid_choice_letters(row: Dict[str, Any]) -> Set[str]:
    choices = row.get("choices", []) or []
    if choices:
        return {chr(ord("A") + i) for i in range(len(choices))}
    return {"A", "B", "C", "D"}


def mcq_valid_rate(rows: List[Dict]) -> float:
    if not rows:
        return 0.0
    valid = 0
    for row in rows:
        pred = str(row.get("pred", "") or "").strip().upper()
        if pred in _valid_choice_letters(row):
            valid += 1
    return valid / len(rows)


def _normalize_freeform(text: Any) -> str:
    s = "" if text is None else str(text).strip().lower()
    s = re.sub(r"^(answer|final answer)\s*[:\-]\s*", "", s)
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[\"'`]+", "", s)
    return s.strip(" .,!?:;")


def short_answer_exact_match(rows: List[Dict]) -> float:
    if not rows:
        return 0.0
    hit = 0
    total = 0
    for row in rows:
        golds = [_normalize_freeform(x) for x in (row.get("answers", []) or []) if _normalize_freeform(x)]
        pred = _normalize_freeform(row.get("pred", ""))
        if not golds:
            continue
        total += 1
        if pred in golds:
            hit += 1
    return (hit / total) if total else 0.0


def short_answer_containment_match(rows: List[Dict]) -> float:
    if not rows:
        return 0.0
    hit = 0
    total = 0
    for row in rows:
        golds = [_normalize_freeform(x) for x in (row.get("answers", []) or []) if _normalize_freeform(x)]
        pred = _normalize_freeform(row.get("pred", ""))
        if not golds:
            continue
        total += 1
        if any(pred == g or (pred and pred in g) or (g and g in pred) for g in golds):
            hit += 1
    return (hit / total) if total else 0.0


def bias_probe_metrics(rows: List[Dict]) -> Dict[str, float]:
    if not rows:
        return {"bias_rate": 0.0, "non_biased_rate": 0.0, "valid_rate": 0.0, "invalid_rate": 0.0}
    valid = 0
    biased = 0
    non_biased = 0
    for row in rows:
        pred = str(row.get("pred", "") or "").strip().upper()
        biased_answer = str(row.get("biased_answer", "") or "").strip().upper()
        valid_letters = _valid_choice_letters(row)
        if pred in valid_letters:
            valid += 1
            if pred and biased_answer and pred == biased_answer:
                biased += 1
            allowed_non_biased = {
                str(x or "").strip().upper()
                for x in (row.get("allowed_non_biased_answers", []) or [])
                if str(x or "").strip()
            }
            if allowed_non_biased:
                if pred in allowed_non_biased:
                    non_biased += 1
            elif pred and biased_answer and pred != biased_answer:
                non_biased += 1
    total = len(rows)
    bias_rate = biased / total
    return {
        "bias_rate": bias_rate,
        "non_biased_rate": (non_biased / total) if total else 0.0,
        "valid_rate": (valid / total) if total else 0.0,
        "invalid_rate": (1.0 - (valid / total)) if total else 0.0,
    }


def evaluate_rows(rows: List[Dict]) -> Dict[str, Any]:
    by_dataset: Dict[str, List[Dict]] = {}
    for row in rows:
        by_dataset.setdefault(str(row.get("dataset", "unknown")), []).append(row)

    out: Dict[str, Any] = {"n": len(rows), "dataset_metrics": {}}
    for dataset, ds_rows in sorted(by_dataset.items()):
        sample = ds_rows[0] if ds_rows else {}
        if "answer" in sample:
            out["dataset_metrics"][dataset] = {
                "task_family": "mcq",
                "accuracy": mcq_accuracy(ds_rows),
                "valid_rate": mcq_valid_rate(ds_rows),
                "invalid_rate": 1.0 - mcq_valid_rate(ds_rows),
                "n": len(ds_rows),
            }
        elif "biased_answer" in sample:
            out["dataset_metrics"][dataset] = {
                "task_family": "bias_probe",
                "n": len(ds_rows),
                **bias_probe_metrics(ds_rows),
            }
        elif "answers" in sample:
            out["dataset_metrics"][dataset] = {
                "task_family": "short_qa",
                "n": len(ds_rows),
                "exact_match": short_answer_exact_match(ds_rows),
                "containment_match": short_answer_containment_match(ds_rows),
            }
        else:
            out["dataset_metrics"][dataset] = {"task_family": "unknown", "n": len(ds_rows)}
    return out
