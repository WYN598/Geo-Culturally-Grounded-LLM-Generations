import argparse
import json
import os
from typing import Any, Dict, List

import yaml

from .eval import evaluate_rows, mcq_accuracy
from .llm_client import LLMClient
from .pipeline import (
    BiasAwareSearchPipeline,
    GeneralSearchPipeline,
    KBPipeline,
    VanillaPipeline,
    dump_jsonl,
    jsonl_integrity_summary,
    load_jsonl,
    load_kb_cache,
    load_search_cache,
)
from .retrieval import make_kb_index
from .search_grounding import WebSearcher


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def summarize_search_trace(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {
            "n": 0,
            "used_evidence_n": 0,
            "used_evidence_rate": 0.0,
            "stage_counts": {},
            "gate_reason_top": [],
        }
    used = 0
    stage_counts: Dict[str, int] = {}
    gate_reasons: Dict[str, int] = {}
    for row in rows:
        trace = row.get("search_trace", {}) or {}
        if bool(trace.get("used_evidence", False)):
            used += 1
        stage = str(
            trace.get("final_stage", "")
            or ("search_answer_augmented" if trace.get("used_evidence", False) else "search_answer_fallback")
        )
        if stage:
            stage_counts[stage] = stage_counts.get(stage, 0) + 1
        reason = str(trace.get("evidence_gate_reason", "") or "").strip()
        if reason:
            gate_reasons[reason] = gate_reasons.get(reason, 0) + 1
    top_reasons = sorted(gate_reasons.items(), key=lambda x: x[1], reverse=True)[:10]
    return {
        "n": len(rows),
        "used_evidence_n": used,
        "used_evidence_rate": (used / len(rows)) if rows else 0.0,
        "stage_counts": stage_counts,
        "gate_reason_top": [{"reason": k, "count": v} for k, v in top_reasons],
    }


def run(mode: str, config_path: str) -> None:
    with open(config_path, "r", encoding="utf-8-sig") as f:
        cfg = yaml.safe_load(f)

    exp = cfg["experiment"]
    llm_cfg = cfg["llm"]

    ensure_dir(exp["output_dir"])

    llm = LLMClient(
        provider=llm_cfg.get("provider", "openai"),
        model=llm_cfg.get("model", "gpt-4o-mini"),
        temperature=float(llm_cfg.get("temperature", 0.2)),
        max_tokens=int(llm_cfg.get("max_tokens", 300)),
    )

    eval_rows = load_jsonl(exp["eval_path"], strict=True)
    metrics = {}

    if mode in ["vanilla", "all"]:
        llm.reset_usage_log()
        pipe = VanillaPipeline(llm)
        preds = []
        for item in eval_rows:
            pred, raw = pipe.predict(item)
            preds.append({**item, "pred": pred, "raw_output": raw})
        out = os.path.join(exp["output_dir"], "vanilla_predictions.jsonl")
        dump_jsonl(out, preds)
        metrics["vanilla_prediction_integrity"] = jsonl_integrity_summary(out, expected_n=len(eval_rows))
        if preds and all("answer" in row for row in preds):
            metrics["vanilla_acc"] = mcq_accuracy(preds)
        metrics["vanilla_eval"] = evaluate_rows(preds)
        usage_out = os.path.join(exp["output_dir"], "llm_usage_vanilla.jsonl")
        llm.dump_usage_log(usage_out)
        metrics["vanilla_usage"] = llm.usage_summary()

    if mode in ["kb", "all"]:
        llm.reset_usage_log()
        kcfg = cfg["kb_grounding"]
        kb = make_kb_index(exp["kb_path"], kcfg)
        kb_cache_by_id = {}
        kb_cache_path = str(kcfg.get("cache_path", "") or "").strip()
        if kb_cache_path and os.path.exists(kb_cache_path):
            kb_cache_by_id = load_kb_cache(kb_cache_path)

        pipe = KBPipeline(
            llm=llm,
            kb_index=kb,
            retrieve_top_n=int(kcfg.get("retrieve_top_n", 5)),
            cache_by_id=kb_cache_by_id,
            use_cache_only=bool(kcfg.get("use_cache_only", False)),
        )
        preds = []
        for item in eval_rows:
            pred, evidence, trace, raw = pipe.predict(item)
            preds.append({**item, "pred": pred, "raw_output": raw, "evidence": evidence, "kb_trace": trace})
        out = os.path.join(exp["output_dir"], "kb_predictions.jsonl")
        dump_jsonl(out, preds)
        metrics["kb_prediction_integrity"] = jsonl_integrity_summary(out, expected_n=len(eval_rows))
        if preds and all("answer" in row for row in preds):
            metrics["kb_acc"] = mcq_accuracy(preds)
        metrics["kb_eval"] = evaluate_rows(preds)
        usage_out = os.path.join(exp["output_dir"], "llm_usage_kb.jsonl")
        llm.dump_usage_log(usage_out)
        metrics["kb_usage"] = llm.usage_summary()

    if mode in ["search", "all"]:
        llm.reset_usage_log()
        scfg = cfg["search_grounding"]
        web = WebSearcher(
            search_engine=str(scfg.get("search_engine", "ddgs")),
            timeout_sec=int(scfg.get("timeout_sec", 8)),
            max_page_chars=int(scfg.get("max_page_chars", 5000)),
            chunk_chars=int(scfg.get("chunk_chars", 900)),
            overlap_chars=int(scfg.get("overlap_chars", 120)),
            min_chars=int(scfg.get("min_chars", 80)),
            min_snippet_chars=int(scfg.get("min_snippet_chars", 60)),
            ignored_domains=list(scfg.get("ignored_domains", [])) or None,
            max_retries=int(scfg.get("max_retries", 2)),
            sleep_min_sec=float(scfg.get("sleep_min_sec", 0.05)),
            sleep_max_sec=float(scfg.get("sleep_max_sec", 0.25)),
            google_region=str(scfg.get("google_region", "us")),
            google_lang=str(scfg.get("google_lang", "en")),
            google_safe=str(scfg.get("google_safe", "off")),
            google_pause_min_sec=float(scfg.get("google_pause_min_sec", 1.0)),
            google_pause_max_sec=float(scfg.get("google_pause_max_sec", 3.0)),
            google_process_factor=int(scfg.get("google_process_factor", 3)),
            google_fallback_to_ddgs=bool(scfg.get("google_fallback_to_ddgs", True)),
            google_fail_open_after=int(scfg.get("google_fail_open_after", 3)),
            google_disable_sec=int(scfg.get("google_disable_sec", 600)),
        )

        cache_by_id = {}
        cache_path = str(scfg.get("cache_path", "") or "").strip()
        if cache_path and os.path.exists(cache_path):
            cache_by_id = load_search_cache(cache_path)

        search_pipeline_type = str(scfg.get("search_pipeline_type", "general") or "general").strip().lower()
        pipe_cls = BiasAwareSearchPipeline if search_pipeline_type == "bias_aware" else GeneralSearchPipeline
        pipe = pipe_cls(
            llm=llm,
            web=web,
            search_top_n=int(scfg.get("search_top_n", 5)),
            keep_top_k=int(scfg.get("keep_top_k", 3)),
            query_expansion_n=int(scfg.get("query_expansion_n", 2)),
            max_pages=int(scfg.get("max_pages", 8)),
            keep_per_domain=int(scfg.get("keep_per_domain", 2)),
            llm_query_rewrite=bool(scfg.get("llm_query_rewrite", True)),
            rewrite_policy=str(scfg.get("rewrite_policy", "auto")),
            llm_relevance=bool(scfg.get("llm_relevance", False)),
            llm_relevance_top_m=int(scfg.get("llm_relevance_top_m", 6)),
            embedding_preranker=str(scfg.get("embedding_preranker", "openai")),
            embedding_model=str(scfg.get("embedding_model", "text-embedding-3-small")),
            embedding_preranker_top_m=int(scfg.get("embedding_preranker_top_m", 24)),
            embedding_preranker_weight=float(scfg.get("embedding_preranker_weight", 0.15)),
            semantic_reranker=str(scfg.get("semantic_reranker", "none")),
            semantic_reranker_model=str(scfg.get("semantic_reranker_model", "cross-encoder/ms-marco-MiniLM-L-12-v2")),
            semantic_reranker_top_m=int(scfg.get("semantic_reranker_top_m", 12)),
            semantic_reranker_weight=float(scfg.get("semantic_reranker_weight", 0.2)),
            semantic_reranker_device=str(scfg.get("semantic_reranker_device", "cuda")),
            semantic_reranker_batch_size=int(scfg.get("semantic_reranker_batch_size", 32)),
            diversify_by_url=bool(scfg.get("diversify_by_url", True)),
            domain_priors=dict(scfg.get("domain_priors", {}) or {}),
            enable_hyde=bool(scfg.get("enable_hyde", False)),
            hyde_query_n=int(scfg.get("hyde_query_n", 1)),
            risk_medium_threshold=float(scfg.get("risk_medium_threshold", 1.5)),
            risk_high_threshold=float(scfg.get("risk_high_threshold", 3.5)),
            bias_query_max_n=int(scfg.get("bias_query_max_n", 4)),
            enable_balance_gate=bool(scfg.get("enable_balance_gate", True)),
            route_bonus_primary=float(scfg.get("route_bonus_primary", 0.02)),
            route_bonus_claim_testing=float(scfg.get("route_bonus_claim_testing", 0.03)),
            route_bonus_counter_evidence=float(scfg.get("route_bonus_counter_evidence", 0.05)),
            route_bonus_confounder_context=float(scfg.get("route_bonus_confounder_context", 0.04)),
            enable_query_feedback_retry=bool(scfg.get("enable_query_feedback_retry", True)),
            query_feedback_max_retry=int(scfg.get("query_feedback_max_retry", 1)),
            query_retry_min_top_score=float(scfg.get("query_retry_min_top_score", 0.12)),
            enable_evidence_organization=bool(scfg.get("enable_evidence_organization", True)),
            enable_evidence_gate=bool(scfg.get("enable_evidence_gate", True)),
            min_evidence_score=float(scfg.get("min_evidence_score", 0.16)),
            summary_max_items=int(scfg.get("summary_max_items", 4)),
            low_quality_domains=list(scfg.get("low_quality_domains", []) or []),
            low_quality_url_keywords=list(scfg.get("low_quality_url_keywords", []) or []),
            cache_by_id=cache_by_id,
            use_cache_only=bool(scfg.get("use_cache_only", False)),
            include_candidate_details=bool(scfg.get("include_candidate_details", False)),
            strict_feature_checks=bool(scfg.get("strict_feature_checks", False)),
        )
        preds = []
        for item in eval_rows:
            pred, evidence, trace, raw = pipe.predict(item)
            preds.append({**item, "pred": pred, "raw_output": raw, "evidence": evidence, "search_trace": trace})
        out = os.path.join(exp["output_dir"], "search_predictions.jsonl")
        dump_jsonl(out, preds)
        metrics["search_prediction_integrity"] = jsonl_integrity_summary(out, expected_n=len(eval_rows))
        if preds and all("answer" in row for row in preds):
            metrics["search_acc"] = mcq_accuracy(preds)
        metrics["search_eval"] = evaluate_rows(preds)
        metrics["search_trace_summary"] = summarize_search_trace(preds)
        metrics["search_runtime_status"] = ((preds[0].get("search_trace", {}) or {}).get("runtime_status", {})) if preds else {}
        usage_out = os.path.join(exp["output_dir"], "llm_usage_search.jsonl")
        llm.dump_usage_log(usage_out)
        metrics["search_usage"] = llm.usage_summary()

    metrics_path = os.path.join(exp["output_dir"], "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(json.dumps(metrics, ensure_ascii=True, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["vanilla", "kb", "search", "all"], default="all")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    run(args.mode, args.config)


if __name__ == "__main__":
    main()






