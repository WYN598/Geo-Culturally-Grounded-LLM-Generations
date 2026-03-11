import argparse
import json
import os

import yaml

from .eval import mcq_accuracy
from .llm_client import LLMClient
from .pipeline import KBPipeline, SearchPipeline, VanillaPipeline, dump_jsonl, load_jsonl, load_kb_cache, load_search_cache
from .retrieval import make_kb_index
from .search_grounding import WebSearcher


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


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

    eval_rows = load_jsonl(exp["eval_path"])
    metrics = {}

    if mode in ["vanilla", "all"]:
        llm.reset_usage_log()
        pipe = VanillaPipeline(llm)
        preds = []
        for item in eval_rows:
            pred = pipe.predict(item)
            preds.append({**item, "pred": pred})
        out = os.path.join(exp["output_dir"], "vanilla_predictions.jsonl")
        dump_jsonl(out, preds)
        metrics["vanilla_acc"] = mcq_accuracy(preds)
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
            keep_top_k=int(kcfg.get("keep_top_k", 3)),
            selection_mode=str(kcfg.get("selection_mode", "selective")),
            min_evidence_score=float(kcfg.get("min_evidence_score", 0.0)),
            cache_by_id=kb_cache_by_id,
            use_cache_only=bool(kcfg.get("use_cache_only", False)),
            include_candidate_details=bool(kcfg.get("include_candidate_details", False)),
        )
        preds = []
        for item in eval_rows:
            pred, evidence, trace = pipe.predict(item)
            preds.append({**item, "pred": pred, "evidence": evidence, "kb_trace": trace})
        out = os.path.join(exp["output_dir"], "kb_predictions.jsonl")
        dump_jsonl(out, preds)
        metrics["kb_acc"] = mcq_accuracy(preds)
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

        pipe = SearchPipeline(
            llm=llm,
            web=web,
            search_top_n=int(scfg.get("search_top_n", 5)),
            keep_top_k=int(scfg.get("keep_top_k", 3)),
            query_expansion_n=int(scfg.get("query_expansion_n", 2)),
            max_pages=int(scfg.get("max_pages", 8)),
            keep_per_domain=int(scfg.get("keep_per_domain", 2)),
            llm_relevance=bool(scfg.get("llm_relevance", True)),
            llm_relevance_top_m=int(scfg.get("llm_relevance_top_m", 8)),
            selection_mode=str(scfg.get("selection_mode", "selective")),
            min_evidence_score=float(scfg.get("min_evidence_score", 0.0)),
            require_choice_overlap=bool(scfg.get("require_choice_overlap", False)),
            diversify_by_url=bool(scfg.get("diversify_by_url", False)),
            domain_priors=dict(scfg.get("domain_priors", {}) or {}),
            label_task_force_use_evidence=bool(scfg.get("label_task_force_use_evidence", False)),
            dataset_overrides=dict(scfg.get("dataset_overrides", {}) or {}),
            cache_by_id=cache_by_id,
            use_cache_only=bool(scfg.get("use_cache_only", False)),
            include_candidate_details=bool(scfg.get("include_candidate_details", False)),
            snippet_only_penalty=float(scfg.get("snippet_only_penalty", 0.0)),
            label_semantic_bonus=float(scfg.get("label_semantic_bonus", 0.0)),
            label_noise_penalty=float(scfg.get("label_noise_penalty", 0.0)),
            label_retry_min_semantic_overlap=float(scfg.get("label_retry_min_semantic_overlap", 0.06)),
            label_min_semantic_overlap_for_use=float(scfg.get("label_min_semantic_overlap_for_use", 0.0)),
            label_min_top_score_for_use=float(scfg.get("label_min_top_score_for_use", 0.0)),
        )
        preds = []
        for item in eval_rows:
            pred, evidence, trace = pipe.predict(item)
            preds.append({**item, "pred": pred, "evidence": evidence, "search_trace": trace})
        out = os.path.join(exp["output_dir"], "search_predictions.jsonl")
        dump_jsonl(out, preds)
        metrics["search_acc"] = mcq_accuracy(preds)
        usage_out = os.path.join(exp["output_dir"], "llm_usage_search.jsonl")
        llm.dump_usage_log(usage_out)
        metrics["search_usage"] = llm.usage_summary()

    metrics_path = os.path.join(exp["output_dir"], "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["vanilla", "kb", "search", "all"], default="all")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    run(args.mode, args.config)


if __name__ == "__main__":
    main()
