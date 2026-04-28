import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple


ARTIFACT_PHRASES = [
    "annotation label",
    "dominant annotation",
    "dominant label",
    "stereotype",
    "non-stereotype",
]

HIJACK_PATTERNS = {
    "music_theory_dominant": [
        r"\bdominant chord\b",
        r"\bmusic theory\b",
        r"\broman numeral\b",
        r"\btriads?\b",
        r"\btonic\b",
        r"\bsubdominant\b",
    ],
    "annotation_tutorial": [
        r"\bannotation schema\b",
        r"\bannotating texts\b",
        r"\bparts of an annotation\b",
        r"\bhow to evaluate trademarks\b",
        r"\bdominant label\b",
    ],
    "homework_or_notes": [
        r"\bstudocu\b",
        r"\bnotes\b",
        r"\bppt\b",
        r"\bdocx\b",
        r"\blesson\b",
        r"\bworksheet\b",
    ],
}


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def extract_claim_pair(question: str) -> Tuple[str, str]:
    pats = [
        r"claim\s*['\"]\s*([^'\"]+?)\s*->\s*([^'\"]+?)\s*['\"]",
        r"['\"]\s*([^'\"]+?)\s*->\s*([^'\"]+?)\s*['\"]",
        r"claim\s*\(\s*([^()]+?)\s*->\s*([^()]+?)\s*\)",
    ]
    for pat in pats:
        m = re.search(pat, str(question or ""), flags=re.IGNORECASE)
        if m:
            return m.group(1).strip(), m.group(2).strip()
    return "", ""


def norm_tokens(text: str) -> List[str]:
    toks = re.findall(r"[A-Za-z][A-Za-z'\\-]{1,}", str(text or "").lower())
    stop = {
        "the",
        "and",
        "for",
        "from",
        "with",
        "that",
        "this",
        "what",
        "which",
        "claim",
        "dominant",
        "annotation",
        "label",
        "people",
        "person",
        "social",
        "cultural",
        "context",
    }
    out: List[str] = []
    for t in toks:
        if len(t) < 3 or t in stop:
            continue
        if t not in out:
            out.append(t)
    return out


def any_pattern(text: str, patterns: List[str]) -> bool:
    return any(re.search(p, text, flags=re.IGNORECASE) for p in patterns)


def classify_case(row: Dict[str, Any]) -> Dict[str, Any]:
    trace = row.get("search_trace", {}) or {}
    question = str(row.get("question", "") or "")
    queries = [str(q) for q in (trace.get("queries", []) or [])]
    selected = trace.get("selected_evidence", []) or []
    top_text = "\n".join(
        f"{ev.get('title', '')}\n{ev.get('text', '')}" for ev in selected[:3] if isinstance(ev, dict)
    )
    top_text_low = top_text.lower()

    left, right = extract_claim_pair(question)
    left_tokens = norm_tokens(left)
    right_tokens = norm_tokens(right)
    combined_tokens = left_tokens + right_tokens

    tags: List[str] = []
    fix_hypotheses: List[str] = []

    if any(any(p in q.lower() for p in ARTIFACT_PHRASES) for q in queries):
        tags.append("query_artifact_leakage")
        fix_hypotheses.append("strip annotation/stereotype wording before or after rewrite")

    if queries and any(q.strip() == question.strip() for q in queries):
        tags.append("rewrite_no_effect")
        fix_hypotheses.append("replace raw-question fallback with deterministic claim-context queries")

    matched_hijacks = [name for name, pats in HIJACK_PATTERNS.items() if any_pattern(top_text_low, pats)]
    if matched_hijacks:
        tags.append("lexical_hijack")
        fix_hypotheses.append("bias-safe fallback queries should avoid artifact words like dominant/annotation")

    if combined_tokens:
        present = sum(1 for tok in combined_tokens if tok in top_text_low)
        if present == 0:
            tags.append("claim_missing_from_evidence")
            fix_hypotheses.append("require evidence to mention claim entity or attribute before use")
        elif present < max(1, min(2, len(combined_tokens))):
            tags.append("weak_claim_alignment")
            fix_hypotheses.append("increase alignment filtering for stereotype-label tasks")

    if trace.get("used_evidence") and row.get("pred") != row.get("answer"):
        tags.append("wrong_with_grounding")
        fix_hypotheses.append("strengthen evidence gate for bias-sensitive tasks")

    if row.get("pred") == "C":
        tags.append("default_unsure_prediction")
        fix_hypotheses.append("separate abstention from weak-evidence stereotype cases")

    if not tags:
        tags.append("other")
        fix_hypotheses.append("manual inspection")

    return {
        "id": row.get("id"),
        "dataset": row.get("dataset"),
        "answer": row.get("answer"),
        "pred": row.get("pred"),
        "question": question,
        "queries": queries,
        "tags": tags,
        "matched_hijacks": matched_hijacks,
        "fix_hypotheses": sorted(set(fix_hypotheses)),
        "top_titles": [str(ev.get("title", "")) for ev in selected[:3] if isinstance(ev, dict)],
    }


def render_md(summary: Dict[str, Any], cases: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    lines.append(f"# Bias Case Analysis: {summary['run_name']}")
    lines.append("")
    lines.append(f"- Predictions: `{summary['predictions_path']}`")
    lines.append(f"- Total rows: `{summary['n_total']}`")
    lines.append(f"- Wrong rows: `{summary['n_wrong']}`")
    lines.append(f"- Error rate: `{summary['error_rate']:.3f}`")
    lines.append("")
    lines.append("## Tag Counts")
    lines.append("")
    lines.append("| Tag | Count |")
    lines.append("| --- | ---: |")
    for tag, count in summary["tag_counts"]:
        lines.append(f"| `{tag}` | {count} |")
    lines.append("")
    lines.append("## Suggested Fixes")
    lines.append("")
    lines.append("| Fix hypothesis | Count |")
    lines.append("| --- | ---: |")
    for fix, count in summary["fix_counts"]:
        lines.append(f"| `{fix}` | {count} |")
    lines.append("")
    lines.append("## Example Cases")
    lines.append("")
    for case in cases[: min(12, len(cases))]:
        lines.append(f"### {case['id']}")
        lines.append(f"- Gold / Pred: `{case['answer']} / {case['pred']}`")
        lines.append(f"- Tags: `{', '.join(case['tags'])}`")
        if case["matched_hijacks"]:
            lines.append(f"- Lexical hijack type: `{', '.join(case['matched_hijacks'])}`")
        lines.append(f"- Query: `{case['queries'][0] if case['queries'] else ''}`")
        lines.append(f"- Question: {case['question']}")
        for title in case["top_titles"]:
            lines.append(f"- Evidence title: {title}")
        lines.append(f"- Fix hypothesis: `{'; '.join(case['fix_hypotheses'])}`")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Heuristic case analysis for bias-sensitive prediction files.")
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-cases", type=int, default=12)
    args = ap.parse_args()

    pred_path = Path(args.predictions)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl(pred_path)
    wrong_rows = [r for r in rows if str(r.get("pred", "")).strip() != str(r.get("answer", "")).strip()]
    classified = [classify_case(r) for r in wrong_rows]

    tag_counter: Counter[str] = Counter()
    fix_counter: Counter[str] = Counter()
    for case in classified:
        tag_counter.update(case["tags"])
        fix_counter.update(case["fix_hypotheses"])

    summary = {
        "run_name": pred_path.parent.parent.name + "/" + pred_path.parent.name,
        "predictions_path": str(pred_path).replace("\\", "/"),
        "n_total": len(rows),
        "n_wrong": len(wrong_rows),
        "error_rate": (len(wrong_rows) / len(rows)) if rows else 0.0,
        "tag_counts": tag_counter.most_common(),
        "fix_counts": fix_counter.most_common(),
    }

    (out_dir / "case_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "wrong_cases.json").write_text(json.dumps(classified, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "report.md").write_text(render_md(summary, classified[: args.max_cases]), encoding="utf-8")

    print(json.dumps({"out_dir": str(out_dir), "n_total": len(rows), "n_wrong": len(wrong_rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
