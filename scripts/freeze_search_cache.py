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
        timeout_sec=int(scfg.get("timeout_sec", 8)),
        max_page_chars=int(scfg.get("max_page_chars", 5000)),
        chunk_chars=int(scfg.get("chunk_chars", 900)),
        overlap_chars=int(scfg.get("overlap_chars", 120)),
        min_chars=int(scfg.get("min_chars", 80)),
        ignored_domains=list(scfg.get("ignored_domains", [])) or None,
        max_retries=int(scfg.get("max_retries", 2)),
        sleep_min_sec=float(scfg.get("sleep_min_sec", 0.05)),
        sleep_max_sec=float(scfg.get("sleep_max_sec", 0.25)),
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
        include_candidate_details=True,
    )

    frozen_rows: List[Dict[str, Any]] = []
    for item in rows:
        _, trace = pipe.prepare_evidence(item)
        frozen_rows.append(
            {
                "id": str(item.get("id", "")),
                "question": item.get("question", ""),
                "frozen_at_utc": dt.datetime.utcnow().isoformat() + "Z",
                **trace,
            }
        )

    out_dir = os.path.dirname(out_cache) or "."
    ensure_dir(out_dir)
    dump_jsonl(out_cache, frozen_rows)

    summary = {
        "cache_path": out_cache,
        "num_items": len(frozen_rows),
        "config": config_path,
        "provider": llm.provider,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Freeze search retrieval artifacts for reproducible search-grounding experiments.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--out-cache", default="outputs/search_cache.jsonl")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    run(args.config, args.out_cache, args.limit)
