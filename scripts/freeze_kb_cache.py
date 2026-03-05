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
from src.pipeline import KBPipeline, dump_jsonl, load_jsonl
from src.retrieval import make_kb_index


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def run(config_path: str, out_cache: str, limit: int = 0) -> None:
    with open(config_path, "r", encoding="utf-8-sig") as f:
        cfg = yaml.safe_load(f)

    exp = cfg["experiment"]
    llm_cfg = cfg["llm"]
    kcfg = cfg["kb_grounding"]

    rows = load_jsonl(exp["eval_path"])
    if limit > 0:
        rows = rows[:limit]

    llm = LLMClient(
        provider=llm_cfg.get("provider", "mock"),
        model=llm_cfg.get("model", "gpt-4o-mini"),
        temperature=float(llm_cfg.get("temperature", 0.0)),
        max_tokens=int(llm_cfg.get("max_tokens", 300)),
    )
    kb = make_kb_index(exp["kb_path"], kcfg)

    pipe = KBPipeline(
        llm=llm,
        kb_index=kb,
        retrieve_top_n=int(kcfg.get("retrieve_top_n", 5)),
        keep_top_k=int(kcfg.get("keep_top_k", 3)),
        selection_mode=str(kcfg.get("selection_mode", "selective")),
        min_evidence_score=float(kcfg.get("min_evidence_score", 0.0)),
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
        "kb_path": exp["kb_path"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Freeze KB retrieval artifacts for reproducible KB-grounding experiments.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--out-cache", default="outputs/kb_cache.jsonl")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    run(args.config, args.out_cache, args.limit)
