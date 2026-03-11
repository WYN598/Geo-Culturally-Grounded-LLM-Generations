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
from src.pipeline import SearchPipeline, dump_jsonl, load_jsonl
from src.search_grounding import WebSearcher


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def run(config_path: str, out_cache: str, limit: int = 0) -> None:
    with open(config_path, "r", encoding="utf-8-sig") as f:
        cfg = yaml.safe_load(f)

    exp = cfg["experiment"]
    llm_cfg = cfg["llm"]
    scfg = cfg["search_grounding"]

    rows = load_jsonl(exp["eval_path"])
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
        include_candidate_details=True,
        snippet_only_penalty=float(scfg.get("snippet_only_penalty", 0.0)),
        label_semantic_bonus=float(scfg.get("label_semantic_bonus", 0.0)),
        label_noise_penalty=float(scfg.get("label_noise_penalty", 0.0)),
        label_retry_min_semantic_overlap=float(scfg.get("label_retry_min_semantic_overlap", 0.06)),
        label_min_semantic_overlap_for_use=float(scfg.get("label_min_semantic_overlap_for_use", 0.0)),
        label_min_top_score_for_use=float(scfg.get("label_min_top_score_for_use", 0.0)),
    )

    out_dir = os.path.dirname(out_cache) or "."
    ensure_dir(out_dir)

    # Resume support: if cache exists, skip ids that are already frozen.
    existing_ids = set()
    if os.path.exists(out_cache):
        try:
            for r in load_jsonl(out_cache):
                rid = str(r.get("id", "")).strip()
                if rid:
                    existing_ids.add(rid)
            if existing_ids:
                print(f"[freeze-search] resume from existing cache: {len(existing_ids)} items", flush=True)
        except Exception:
            # If cache is corrupted/unreadable, start a new file.
            existing_ids = set()

    total = len(rows)
    done = 0
    for item in rows:
        rid = str(item.get("id", "")).strip()
        if rid and rid in existing_ids:
            done += 1

    mode = "a" if os.path.exists(out_cache) and existing_ids else "w"
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

    summary = {
        "cache_path": out_cache,
        "num_items": done,
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
