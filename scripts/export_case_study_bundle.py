import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline import build_augmented_prompt, build_grounded_answer_prompt, format_mcq_prompt


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def normalize_id(x: Any) -> str:
    return str(x or "").strip()


def idx_by_id(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        rid = normalize_id(r.get("id"))
        if rid:
            out[rid] = r
    return out


def usage_by_item(usage_rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for r in usage_rows:
        rid = normalize_id(r.get("item_id"))
        if not rid:
            continue
        if rid not in out:
            out[rid] = {
                "total_tokens": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "calls": 0,
                "by_stage": {},
            }
        slot = out[rid]
        pt = int(r.get("prompt_tokens", 0) or 0)
        ct = int(r.get("completion_tokens", 0) or 0)
        tt = int(r.get("total_tokens", 0) or (pt + ct))
        stg = str(r.get("stage", "unknown"))
        slot["total_tokens"] += tt
        slot["prompt_tokens"] += pt
        slot["completion_tokens"] += ct
        slot["calls"] += 1
        if stg not in slot["by_stage"]:
            slot["by_stage"][stg] = {
                "calls": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            }
        s = slot["by_stage"][stg]
        s["calls"] += 1
        s["prompt_tokens"] += pt
        s["completion_tokens"] += ct
        s["total_tokens"] += tt
    return out


def final_prompt_for_row(row: Dict[str, Any]) -> Tuple[str, str]:
    trace = dict(row.get("search_trace", {}) or {})
    organized = list(trace.get("organized_evidence", []) or [])
    evidence = list(row.get("evidence", []) or [])
    if organized:
        return "search_answer_augmented", build_grounded_answer_prompt(row["question"], row["choices"], organized)
    if evidence:
        return "search_answer_augmented", build_augmented_prompt(row["question"], row["choices"], evidence)
    return "search_answer_fallback", format_mcq_prompt(row["question"], row["choices"])


def group_candidates_by_url(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, Dict[str, Any]] = {}
    for ch in chunks:
        url = str(ch.get("url", "") or "").strip()
        if not url:
            continue
        if url not in grouped:
            grouped[url] = {
                "url": url,
                "domain": str(ch.get("domain", "") or ""),
                "title": str(ch.get("title", "") or ""),
                "query_set": [],
                "chunks": [],
                "chunk_count": 0,
            }
        slot = grouped[url]
        q = str(ch.get("query", "") or "")
        if q and q not in slot["query_set"]:
            slot["query_set"].append(q)
        slot["chunks"].append(
            {
                "score": float(ch.get("score", 0.0) or 0.0),
                "text": str(ch.get("text", "") or ""),
            }
        )
        slot["chunk_count"] += 1
    return sorted(grouped.values(), key=lambda x: x["chunk_count"], reverse=True)


def run(args: argparse.Namespace) -> None:
    run_root = Path(args.run_root).resolve()
    analysis_dir = run_root / "analysis" / args.out_subdir
    cases_dir = analysis_dir / "cases"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    cases_dir.mkdir(parents=True, exist_ok=True)

    summary_path = run_root / "analysis" / "ablation_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"summary not found: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    runs_meta = {str(r["name"]): r for r in summary.get("runs", [])}
    if args.focus_system not in runs_meta:
        raise ValueError(f"focus system not found in summary: {args.focus_system}")

    all_preds: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for name in runs_meta.keys():
        pred_file = run_root / "runs" / name / ("vanilla_predictions.jsonl" if name == "vanilla" else "search_predictions.jsonl")
        rows = load_jsonl(pred_file)
        all_preds[name] = idx_by_id(rows)

    focus_rows = list(all_preds[args.focus_system].values())
    if args.limit > 0:
        focus_rows = focus_rows[: args.limit]

    focus_meta = runs_meta[args.focus_system]
    cache_path = Path(str(focus_meta.get("cache_path", "") or "").strip())
    if not cache_path.is_absolute():
        cache_path = ROOT / cache_path
    cache_by_id = idx_by_id(load_jsonl(cache_path)) if cache_path.exists() else {}

    usage_map_by_system: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for name in runs_meta.keys():
        usage_file = run_root / "runs" / name / ("llm_usage_vanilla.jsonl" if name == "vanilla" else "llm_usage_search.jsonl")
        usage_map_by_system[name] = usage_by_item(load_jsonl(usage_file))

    manifest_rows: List[Dict[str, Any]] = []
    for row in focus_rows:
        rid = normalize_id(row.get("id"))
        if not rid:
            continue
        stage, final_prompt = final_prompt_for_row(row if args.focus_system != "vanilla" else row)
        focus_trace = dict(row.get("search_trace", {}) or {})
        cache_row = dict(cache_by_id.get(rid, {}) or {})
        candidate_chunks = list(cache_row.get("candidate_evidence", []) or [])

        per_system: Dict[str, Any] = {}
        for sys_name, pred_by_id in all_preds.items():
            srow = pred_by_id.get(rid, {})
            if not srow:
                continue
            pred = str(srow.get("pred", "")).strip()
            ans = str(srow.get("answer", "")).strip()
            per_system[sys_name] = {
                "pred": pred,
                "answer": ans,
                "is_correct": pred == ans,
                "usage": usage_map_by_system.get(sys_name, {}).get(rid, {}),
            }

        case_obj = {
            "id": rid,
            "dataset": row.get("dataset", ""),
            "question": row.get("question", ""),
            "choices": row.get("choices", []),
            "gold_answer": row.get("answer", ""),
            "focus_system": args.focus_system,
            "focus_prediction": row.get("pred", ""),
            "focus_is_correct": str(row.get("pred", "")) == str(row.get("answer", "")),
            "final_stage": stage,
            "final_prompt": final_prompt,
            "all_system_predictions": per_system,
            "search_trace": focus_trace,
            "cache_trace": {
                "cache_path": str(cache_path),
                "query_source": cache_row.get("query_source", ""),
                "search_plan": cache_row.get("search_plan", {}),
                "queries": cache_row.get("queries", []),
                "search_events": cache_row.get("search_events", []),
                "retrieved_hits": cache_row.get("retrieved_hits", 0),
                "dedup_hits": cache_row.get("dedup_hits", 0),
                "raw_candidate_chunks": cache_row.get("raw_candidate_chunks", 0),
                "candidate_chunks": cache_row.get("candidate_chunks", 0),
                "filter_stats": cache_row.get("filter_stats", {}),
            },
            "selected_evidence_chunks": focus_trace.get("selected_evidence", []),
            "organized_evidence": focus_trace.get("organized_evidence", []),
            "evidence_organization": focus_trace.get("evidence_organization", {}),
            "evidence_gate_reason": focus_trace.get("evidence_gate_reason", ""),
            "used_evidence": bool(focus_trace.get("used_evidence", False)),
            "candidate_evidence_chunks_full": candidate_chunks,
            "candidate_evidence_grouped_by_url": group_candidates_by_url(candidate_chunks),
        }

        case_path = cases_dir / f"{rid}.json"
        case_path.write_text(json.dumps(case_obj, ensure_ascii=False, indent=2), encoding="utf-8")

        manifest_rows.append(
            {
                "id": rid,
                "dataset": str(row.get("dataset", "")),
                "focus_system": args.focus_system,
                "pred": str(row.get("pred", "")),
                "answer": str(row.get("answer", "")),
                "is_correct": str(row.get("pred", "")) == str(row.get("answer", "")),
                "used_evidence": bool(focus_trace.get("used_evidence", False)),
                "case_file": str(case_path),
            }
        )

    manifest_jsonl = analysis_dir / "case_manifest.jsonl"
    with manifest_jsonl.open("w", encoding="utf-8") as f:
        for r in manifest_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    manifest_csv = analysis_dir / "case_manifest.csv"
    with manifest_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["id", "dataset", "focus_system", "pred", "answer", "is_correct", "used_evidence", "case_file"],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    meta = {
        "run_root": str(run_root),
        "focus_system": args.focus_system,
        "num_cases_exported": len(manifest_rows),
        "analysis_dir": str(analysis_dir),
        "cases_dir": str(cases_dir),
        "manifest_jsonl": str(manifest_jsonl),
        "manifest_csv": str(manifest_csv),
    }
    meta_path = analysis_dir / "case_export_meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export full per-case bundle (search contents, prompts, traces, outcomes).")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--focus-system", default="full_general_rag")
    parser.add_argument("--out-subdir", default="case_study_full")
    parser.add_argument("--limit", type=int, default=0)
    run(parser.parse_args())

