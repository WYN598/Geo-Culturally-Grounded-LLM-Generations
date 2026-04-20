import copy
import json
from pathlib import Path
from typing import Dict, List

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def dump_jsonl(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    source_eval = ROOT / "data" / "eval_balanced_200_strict.jsonl"
    source_cfg = ROOT / "configs" / "config_openai_general_rag_balanced200.yaml"
    out_eval_dir = ROOT / "data" / "benchmarks" / "legacy_single"
    out_cfg_dir = ROOT / "configs" / "legacy_single"
    out_eval_dir.mkdir(parents=True, exist_ok=True)
    out_cfg_dir.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl(source_eval)
    with open(source_cfg, "r", encoding="utf-8-sig") as f:
        base_cfg = yaml.safe_load(f)

    datasets = ["BLEnD", "NormAd", "SeeGULL"]
    for dataset in datasets:
        ds_rows = [row for row in rows if str(row.get("dataset", "")) == dataset]
        out_eval = out_eval_dir / f"{dataset.lower()}_200.jsonl"
        dump_jsonl(out_eval, ds_rows)

        cfg = copy.deepcopy(base_cfg)
        cfg["experiment"]["eval_path"] = str(out_eval.relative_to(ROOT)).replace("\\", "/")
        cfg["experiment"]["output_dir"] = f"outputs/legacy_single/{dataset.lower()}_200/search_general"
        cfg["search_grounding"]["cache_path"] = f"outputs/legacy_single/{dataset.lower()}_200/search_cache_general.jsonl"

        out_cfg = out_cfg_dir / f"config_{dataset.lower()}_200.yaml"
        with open(out_cfg, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

        print(f"[legacy-single] wrote {out_eval} ({len(ds_rows)} rows)")
        print(f"[legacy-single] wrote {out_cfg}")


if __name__ == "__main__":
    main()
