import argparse
import json
import os
import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def normalize_text(text: str) -> str:
    s = str(text or "").lower().strip()
    s = re.sub(r"^[a-z]\s*[\)\.]\s*", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


def answer_text(row: Dict[str, Any]) -> str:
    ans = str(row.get("answer", "")).strip().upper()
    choices = list(row.get("choices", []) or [])
    if not ans or not choices:
        return ""
    idx = ord(ans) - ord("A")
    if idx < 0 or idx >= len(choices):
        return ""
    return normalize_text(choices[idx])


def choice_texts(row: Dict[str, Any]) -> List[str]:
    return [normalize_text(c) for c in (row.get("choices", []) or []) if normalize_text(c)]


def any_contains(texts: List[str], needle: str) -> bool:
    n = normalize_text(needle)
    if len(n) < 2:
        return False
    for t in texts:
        if n in normalize_text(t):
            return True
    return False


def clip(text: str, n: int = 260) -> str:
    s = str(text or "").strip().replace("\n", " ")
    if len(s) <= n:
        return s
    return s[: n - 3] + "..."


def idx_by_id(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        rid = str(r.get("id", "")).strip()
        if rid:
            out[rid] = r
    return out


def split_failure_bucket(
    is_correct: bool,
    candidate_has_gold: Optional[bool],
    selected_has_gold: bool,
) -> str:
    if is_correct:
        return "correct"
    if candidate_has_gold is False:
        return "search_fail"
    if candidate_has_gold is True and not selected_has_gold:
        return "rank_fail"
    if selected_has_gold:
        return "context_or_reasoning_fail"
    return "search_or_rank_fail_unknown"


def analyze(
    search_rows: List[Dict[str, Any]],
    cache_by_id: Dict[str, Dict[str, Any]],
    vanilla_by_id: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    n = len(search_rows)
    dataset_rows: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    rows_with_tags: List[Dict[str, Any]] = []

    for r in search_rows:
        ds = str(r.get("dataset", "unknown"))

        rid = str(r.get("id", "")).strip()
        trace = r.get("search_trace", {}) or {}
        selected_ev = list(trace.get("selected_evidence", []) or [])
        selected_texts = [str(x.get("text", "")) for x in selected_ev if isinstance(x, dict)]
        selected_urls = [str(x.get("url", "")) for x in selected_ev if isinstance(x, dict)]

        cached = cache_by_id.get(rid, {})
        candidate_ev = list(cached.get("candidate_evidence", []) or [])
        candidate_texts = [str(x.get("text", "")) for x in candidate_ev if isinstance(x, dict)]

        gold_txt = answer_text(r)
        selected_has_gold = any_contains(selected_texts, gold_txt)
        candidate_has_gold = any_contains(candidate_texts, gold_txt) if candidate_texts else None

        q_list = [str(x) for x in (trace.get("queries", []) or []) if str(x).strip()]
        q_norm = [normalize_text(x) for x in q_list]
        q_identical = len(set(q_norm)) <= 1
        q_contains_gold = any_contains(q_list, gold_txt)
        q_missing = len(q_list) == 0

        is_correct = str(r.get("pred", "")).upper() == str(r.get("answer", "")).upper()
        bucket = split_failure_bucket(is_correct, candidate_has_gold, selected_has_gold)

        v = vanilla_by_id.get(rid, {})
        vanilla_correct = (
            str(v.get("pred", "")).upper() == str(v.get("answer", "")).upper() if v else None
        )
        degraded_vs_vanilla = bool(v) and vanilla_correct and (not is_correct)

        tagged = {
            "id": rid,
            "dataset": ds,
            "question": r.get("question", ""),
            "choices": r.get("choices", []),
            "answer": r.get("answer", ""),
            "pred_search": r.get("pred", ""),
            "pred_vanilla": v.get("pred", "") if v else "",
            "is_correct": is_correct,
            "vanilla_correct": vanilla_correct,
            "degraded_vs_vanilla": degraded_vs_vanilla,
            "bucket": bucket,
            "queries": q_list,
            "query_issue": {"missing_queries": q_missing, "low_diversity": q_identical, "contains_gold": q_contains_gold},
            "retrieved_hits": int(trace.get("retrieved_hits", 0) or 0),
            "dedup_hits": int(trace.get("dedup_hits", 0) or 0),
            "candidate_chunks": int(trace.get("candidate_chunks", 0) or 0),
            "top_selected_score": float(trace.get("top_selected_score", 0.0) or 0.0),
            "selected_has_gold": selected_has_gold,
            "candidate_has_gold": candidate_has_gold,
            "selected_urls": selected_urls,
            "selected_evidence_preview": [clip(x) for x in selected_texts[:2]],
            "answer_text": gold_txt,
            "choice_overlap_in_selected": any(any_contains(selected_texts, c) for c in choice_texts(r)),
        }
        rows_with_tags.append(tagged)
        dataset_rows[ds].append(tagged)

    def agg(sub_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        m = len(sub_rows)
        if m == 0:
            return {}
        bucket_counts = Counter(x["bucket"] for x in sub_rows)
        degraded = [x for x in sub_rows if x["degraded_vs_vanilla"]]
        return {
            "n": m,
            "accuracy": sum(1 for x in sub_rows if x["is_correct"]) / m,
            "retrieval_stats": {
                "avg_retrieved_hits": sum(x["retrieved_hits"] for x in sub_rows) / m,
                "avg_dedup_hits": sum(x["dedup_hits"] for x in sub_rows) / m,
                "avg_candidate_chunks": sum(x["candidate_chunks"] for x in sub_rows) / m,
                "zero_candidate_chunks_rate": sum(1 for x in sub_rows if x["candidate_chunks"] == 0) / m,
                "zero_selected_evidence_rate": sum(1 for x in sub_rows if len(x["selected_urls"]) == 0) / m,
            },
            "query_stats": {
                "missing_queries_rate": sum(1 for x in sub_rows if x["query_issue"]["missing_queries"]) / m,
                "low_diversity_queries_rate": sum(1 for x in sub_rows if x["query_issue"]["low_diversity"]) / m,
                "query_contains_gold_rate": sum(1 for x in sub_rows if x["query_issue"]["contains_gold"]) / m,
            },
            "failure_buckets": dict(bucket_counts),
            "failure_bucket_rate": {k: v / m for k, v in bucket_counts.items()},
            "selected_has_gold_rate": sum(1 for x in sub_rows if x["selected_has_gold"]) / m,
            "candidate_has_gold_known_rate": (
                sum(1 for x in sub_rows if x["candidate_has_gold"] is not None) / m
            ),
            "candidate_has_gold_rate_when_known": (
                sum(1 for x in sub_rows if x["candidate_has_gold"] is True)
                / max(1, sum(1 for x in sub_rows if x["candidate_has_gold"] is not None))
            ),
            "degraded_vs_vanilla_count": len(degraded),
            "degraded_vs_vanilla_rate": len(degraded) / m,
            "avg_top_selected_score_correct": (
                sum(x["top_selected_score"] for x in sub_rows if x["is_correct"])
                / max(1, sum(1 for x in sub_rows if x["is_correct"]))
            ),
            "avg_top_selected_score_wrong": (
                sum(x["top_selected_score"] for x in sub_rows if not x["is_correct"])
                / max(1, sum(1 for x in sub_rows if not x["is_correct"]))
            ),
        }

    overall = agg(rows_with_tags)
    by_dataset = {ds: agg(rows) for ds, rows in sorted(dataset_rows.items())}

    return {"overall": overall, "by_dataset": by_dataset, "tagged_rows": rows_with_tags}


def build_case_studies(tagged_rows: List[Dict[str, Any]], top_k: int) -> Dict[str, Any]:
    buckets = ["search_fail", "rank_fail", "context_or_reasoning_fail", "search_or_rank_fail_unknown"]
    out: Dict[str, Any] = {}
    for b in buckets:
        rows = [x for x in tagged_rows if x["bucket"] == b]
        rows = sorted(
            rows,
            key=lambda x: (
                1 if x.get("degraded_vs_vanilla") else 0,
                x.get("top_selected_score", 0.0),
            ),
            reverse=True,
        )
        out[b] = rows[:top_k]
    return out


def run(args: argparse.Namespace) -> None:
    search_rows = load_jsonl(args.search_preds)
    cache_rows = load_jsonl(args.search_cache) if args.search_cache and os.path.exists(args.search_cache) else []
    vanilla_rows = load_jsonl(args.vanilla_preds) if args.vanilla_preds and os.path.exists(args.vanilla_preds) else []

    cache_by_id = idx_by_id(cache_rows)
    vanilla_by_id = idx_by_id(vanilla_rows)

    report = analyze(search_rows, cache_by_id, vanilla_by_id)
    cases = build_case_studies(report["tagged_rows"], top_k=args.top_k_cases)

    out_dir = os.path.dirname(args.out_json) or "."
    os.makedirs(out_dir, exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        out_report = {k: v for k, v in report.items() if k != "tagged_rows"}
        json.dump(out_report, f, ensure_ascii=False, indent=2)

    out_cases_dir = os.path.dirname(args.out_cases) or "."
    os.makedirs(out_cases_dir, exist_ok=True)
    with open(args.out_cases, "w", encoding="utf-8") as f:
        json.dump(cases, f, ensure_ascii=False, indent=2)

    print(
        json.dumps(
            {
                "out_json": args.out_json,
                "out_cases": args.out_cases,
                "overall": report["overall"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Layered diagnostics for search-grounding results.")
    parser.add_argument("--search-preds", required=True, help="search_predictions.jsonl")
    parser.add_argument("--search-cache", default="", help="frozen search cache jsonl with candidate_evidence")
    parser.add_argument("--vanilla-preds", default="", help="vanilla_predictions.jsonl for degraded-case detection")
    parser.add_argument("--out-json", default="outputs/analysis/search_diagnostics.json")
    parser.add_argument("--out-cases", default="outputs/analysis/search_case_studies.json")
    parser.add_argument("--top-k-cases", type=int, default=15)
    run(parser.parse_args())
