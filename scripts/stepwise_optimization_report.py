import argparse
import json
import os
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


def mean(vals: List[float]) -> float:
    if not vals:
        return 0.0
    return sum(vals) / len(vals)


def _subset(rows: List[Dict[str, Any]], ds: str) -> List[Dict[str, Any]]:
    if ds == "overall":
        return rows
    return [r for r in rows if str(r.get("dataset", "")) == ds]


def summarize_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    if n == 0:
        return {}

    correct = sum(1 for r in rows if bool(r.get("is_correct", False)))
    bucket = Counter(str(r.get("bucket", "")) for r in rows)

    sf = bucket.get("search_fail", 0)
    rf = bucket.get("rank_fail", 0)
    cf = bucket.get("context_or_reasoning_fail", 0)

    search_fail_rows = [r for r in rows if str(r.get("bucket", "")) == "search_fail"]
    query_missing_rate = (
        sum(1 for r in search_fail_rows if bool((r.get("query_issue", {}) or {}).get("missing_queries", False)))
        / max(1, len(search_fail_rows))
    )
    query_low_div_rate = (
        sum(1 for r in search_fail_rows if bool((r.get("query_issue", {}) or {}).get("low_diversity", False)))
        / max(1, len(search_fail_rows))
    )
    query_contains_gold_rate = (
        sum(1 for r in search_fail_rows if bool((r.get("query_issue", {}) or {}).get("contains_gold", False)))
        / max(1, len(search_fail_rows))
    )

    rank_rows = [r for r in rows if str(r.get("bucket", "")) == "rank_fail"]
    context_rows = [r for r in rows if str(r.get("bucket", "")) == "context_or_reasoning_fail"]

    degraded = [r for r in rows if bool(r.get("degraded_vs_vanilla", False))]

    return {
        "n": n,
        "acc": correct / n,
        "failure_rate": {
            "search_fail": sf / n,
            "rank_fail": rf / n,
            "context_or_reasoning_fail": cf / n,
        },
        "failure_count": {
            "search_fail": sf,
            "rank_fail": rf,
            "context_or_reasoning_fail": cf,
        },
        "query_diagnostics_within_search_fail": {
            "missing_queries_rate": query_missing_rate,
            "low_diversity_rate": query_low_div_rate,
            "contains_gold_rate": query_contains_gold_rate,
        },
        "rank_diagnostics": {
            "mean_top_score_rank_fail": mean([float(r.get("top_selected_score", 0.0) or 0.0) for r in rank_rows]),
            "count": len(rank_rows),
        },
        "context_diagnostics": {
            "mean_top_score_context_fail": mean([float(r.get("top_selected_score", 0.0) or 0.0) for r in context_rows]),
            "count": len(context_rows),
        },
        "degraded_vs_vanilla": {
            "count": len(degraded),
            "rate": len(degraded) / n,
            "by_bucket": dict(Counter(str(r.get("bucket", "")) for r in degraded)),
        },
        "priority": sorted(
            [
                ("search", sf / n),
                ("rank", rf / n),
                ("context", cf / n),
            ],
            key=lambda x: x[1],
            reverse=True,
        ),
    }


def propose_actions(summary: Dict[str, Any]) -> List[str]:
    pri = summary.get("priority", [])
    actions: List[str] = []
    for name, _ in pri:
        if name == "search":
            actions.append("Search: improve query rewrite quality and source filtering; raise candidate_has_gold rate.")
        elif name == "rank":
            actions.append("Rank: tune re-ranking features and domain priors; optimize top-k hit retention.")
        elif name == "context":
            actions.append("Context: strengthen evidence gating and prompt constraints to reduce noisy augmentation.")
    return actions


def run(args: argparse.Namespace) -> None:
    tagged_rows = load_jsonl(args.tagged_jsonl)
    datasets = sorted({str(r.get("dataset", "unknown")) for r in tagged_rows})
    scopes = ["overall"] + datasets

    report: Dict[str, Any] = {"from": args.tagged_jsonl, "scopes": {}}
    for s in scopes:
        sub = _subset(tagged_rows, s)
        summary = summarize_rows(sub)
        summary["suggested_actions"] = propose_actions(summary)
        report["scopes"][s] = summary

    out_dir = os.path.dirname(args.out_json) or "."
    os.makedirs(out_dir, exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    md_lines = [
        "# Stepwise Optimization Report",
        "",
        f"- Input: `{args.tagged_jsonl}`",
        "",
    ]
    for s in scopes:
        item = report["scopes"].get(s, {})
        if not item:
            continue
        md_lines.append(f"## {s}")
        md_lines.append(f"- n: {item['n']}")
        md_lines.append(f"- acc: {item['acc']:.4f}")
        md_lines.append(
            "- fail rates: "
            f"search={item['failure_rate']['search_fail']:.4f}, "
            f"rank={item['failure_rate']['rank_fail']:.4f}, "
            f"context={item['failure_rate']['context_or_reasoning_fail']:.4f}"
        )
        md_lines.append(
            "- query diagnostics (within search_fail): "
            f"missing={item['query_diagnostics_within_search_fail']['missing_queries_rate']:.4f}, "
            f"low_div={item['query_diagnostics_within_search_fail']['low_diversity_rate']:.4f}, "
            f"contains_gold={item['query_diagnostics_within_search_fail']['contains_gold_rate']:.4f}"
        )
        md_lines.append(
            "- degraded vs vanilla: "
            f"{item['degraded_vs_vanilla']['count']} ({item['degraded_vs_vanilla']['rate']:.4f})"
        )
        md_lines.append("- priority: " + " > ".join([f"{x[0]}({x[1]:.4f})" for x in item["priority"]]))
        for a in item["suggested_actions"]:
            md_lines.append(f"- action: {a}")
        md_lines.append("")

    md_out = args.out_md
    out_md_dir = os.path.dirname(md_out) or "."
    os.makedirs(out_md_dir, exist_ok=True)
    with open(md_out, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print(json.dumps({"out_json": args.out_json, "out_md": md_out}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate stepwise optimization report from tagged diagnostics.")
    parser.add_argument("--tagged-jsonl", required=True)
    parser.add_argument("--out-json", default="outputs/analysis/stepwise_optimization_report.json")
    parser.add_argument("--out-md", default="outputs/analysis/stepwise_optimization_report.md")
    run(parser.parse_args())
