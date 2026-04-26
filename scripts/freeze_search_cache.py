import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.llm_client import LLMClient
from src.pipeline import (
    GeneralSearchPipeline,
    build_search_cache_fingerprint,
    dump_jsonl,
    load_jsonl,
    write_cache_meta,
)
from src.search_grounding import WebSearcher


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def run(config_path: str, out_cache: str, limit: int = 0) -> None:
    with open(config_path, "r", encoding="utf-8-sig") as f:
        cfg = yaml.safe_load(f)

    exp = cfg["experiment"]
    llm_cfg = cfg["llm"]
    scfg = cfg["search_grounding"]

    rows = load_jsonl(exp["eval_path"], strict=True)
    if limit > 0:
        rows = rows[:limit]

    llm = LLMClient(
        provider=llm_cfg.get("provider", "mock"),
        model=llm_cfg.get("model", "gpt-4o-mini"),
        temperature=float(llm_cfg.get("temperature", 0.0)),
        max_tokens=int(llm_cfg.get("max_tokens", 300)),
    )

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

    pipe = GeneralSearchPipeline(
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
        enable_query_feedback_retry=bool(scfg.get("enable_query_feedback_retry", True)),
        query_feedback_max_retry=int(scfg.get("query_feedback_max_retry", 1)),
        query_retry_min_top_score=float(scfg.get("query_retry_min_top_score", 0.12)),
        enable_evidence_organization=bool(scfg.get("enable_evidence_organization", True)),
        enable_evidence_gate=bool(scfg.get("enable_evidence_gate", True)),
        min_evidence_score=float(scfg.get("min_evidence_score", 0.16)),
        summary_max_items=int(scfg.get("summary_max_items", 4)),
        low_quality_domains=list(scfg.get("low_quality_domains", []) or []),
        low_quality_url_keywords=list(scfg.get("low_quality_url_keywords", []) or []),
        include_candidate_details=True,
        strict_feature_checks=bool(scfg.get("strict_feature_checks", False)),
    )

    out_dir = os.path.dirname(out_cache) or "."
    ensure_dir(out_dir)
    cache_meta = build_search_cache_fingerprint(cfg)

    # Resume support: if cache exists, skip ids that are already frozen.
    existing_ids = set()
    rewrite_cache = False
    if os.path.exists(out_cache):
        try:
            with open(out_cache + ".meta.json", "r", encoding="utf-8-sig") as mf:
                existing_meta = json.load(mf)
            if json.dumps(existing_meta, sort_keys=True, ensure_ascii=False) != json.dumps(cache_meta, sort_keys=True, ensure_ascii=False):
                rewrite_cache = True
            for r in load_jsonl(out_cache, strict=True):
                rid = str(r.get("id", "")).strip()
                if rid:
                    existing_ids.add(rid)
            if existing_ids:
                print(f"[freeze-search] resume from existing cache: {len(existing_ids)} items", flush=True)
        except Exception:
            # If cache is corrupted/unreadable, start a new file.
            existing_ids = set()
            rewrite_cache = True
    if rewrite_cache:
        existing_ids = set()

    total = len(rows)
    done = 0
    if not rewrite_cache:
        for item in rows:
            rid = str(item.get("id", "")).strip()
            if rid and rid in existing_ids:
                done += 1

    mode = "a" if os.path.exists(out_cache) and existing_ids and not rewrite_cache else "w"
    with open(out_cache, mode, encoding="utf-8") as f:
        for item in rows:
            rid = str(item.get("id", "")).strip()
            if rid and rid in existing_ids:
                continue

            _, trace = pipe.prepare_evidence(item)
            row = {
                "id": rid,
                "question": item.get("question", ""),
                "frozen_at_utc": dt.datetime.utcnow().isoformat() + "Z",
                **trace,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            done += 1
            if done % 10 == 0 or done == total:
                print(f"[freeze-search] {done}/{total}", flush=True)

    usage_path = os.path.join(out_dir, "llm_usage_freeze_search.jsonl")
    llm.dump_usage_log(usage_path)
    write_cache_meta(out_cache, cache_meta)

    summary = {
        "cache_path": out_cache,
        "num_items": done,
        "cache_meta_path": out_cache + ".meta.json",
        "config": config_path,
        "provider": llm.provider,
        "usage_path": usage_path,
        "usage_summary": llm.usage_summary(),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Freeze search retrieval artifacts for reproducible search-grounding experiments.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--out-cache", default="outputs/search_cache.jsonl")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    run(args.config, args.out_cache, args.limit)
