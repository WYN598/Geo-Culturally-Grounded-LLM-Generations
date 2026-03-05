import argparse
import json
import os

import yaml

from .eval import mcq_accuracy
from .llm_client import LLMClient
from .pipeline import KBPipeline, SearchPipeline, VanillaPipeline, dump_jsonl, load_jsonl, load_search_cache
from .retrieval import TfidfKBIndex
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
        provider=llm_cfg.get("provider", "mock"),
        model=llm_cfg.get("model", "gpt-4o-mini"),
        temperature=float(llm_cfg.get("temperature", 0.2)),
        max_tokens=int(llm_cfg.get("max_tokens", 300)),
    )

    eval_rows = load_jsonl(exp["eval_path"])
    metrics = {}

    if mode in ["vanilla", "all"]:
        pipe = VanillaPipeline(llm)
        preds = []
        for item in eval_rows:
            pred = pipe.predict(item)
            preds.append({**item, "pred": pred})
        out = os.path.join(exp["output_dir"], "vanilla_predictions.jsonl")
        dump_jsonl(out, preds)
        metrics["vanilla_acc"] = mcq_accuracy(preds)

    if mode in ["kb", "all"]:
        kb = TfidfKBIndex.from_jsonl(exp["kb_path"])
        pipe = KBPipeline(
            llm=llm,
            kb_index=kb,
            retrieve_top_n=int(cfg["kb_grounding"].get("retrieve_top_n", 5)),
            keep_top_k=int(cfg["kb_grounding"].get("keep_top_k", 3)),
        )
        preds = []
        for item in eval_rows:
            pred, evidence = pipe.predict(item)
            preds.append({**item, "pred": pred, "evidence": evidence})
        out = os.path.join(exp["output_dir"], "kb_predictions.jsonl")
        dump_jsonl(out, preds)
        metrics["kb_acc"] = mcq_accuracy(preds)

    if mode in ["search", "all"]:
        scfg = cfg["search_grounding"]
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
            cache_by_id=cache_by_id,
            use_cache_only=bool(scfg.get("use_cache_only", False)),
            include_candidate_details=bool(scfg.get("include_candidate_details", False)),
        )
        preds = []
        for item in eval_rows:
            pred, evidence, trace = pipe.predict(item)
            preds.append({**item, "pred": pred, "evidence": evidence, "search_trace": trace})
        out = os.path.join(exp["output_dir"], "search_predictions.jsonl")
        dump_jsonl(out, preds)
        metrics["search_acc"] = mcq_accuracy(preds)

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
