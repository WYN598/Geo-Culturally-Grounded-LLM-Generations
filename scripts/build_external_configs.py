import copy
from pathlib import Path
from typing import Dict

import yaml

ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = ROOT / "configs" / "external" / "config_honest_100.yaml"
OUT_DIR = ROOT / "configs" / "external"


DATASET_SPECS: Dict[str, Dict[str, str]] = {
    "cbbq": {
        "sample_prefix": "cbbq",
        "output_prefix": "cbbq",
    },
    "borderlines": {
        "sample_prefix": "borderlines",
        "output_prefix": "borderlines",
    },
    "msqad": {
        "sample_prefix": "msqad",
        "output_prefix": "msqad",
    },
}


def model_output_dir_name(model_name: str) -> str:
    safe = str(model_name or "model").strip().replace("/", "_").replace("\\", "_").replace(" ", "_")
    if safe.lower().startswith("gpt-"):
        safe = "GPT-" + safe[4:]
    return f"{safe}_output"


def main() -> None:
    with BASE_CONFIG.open("r", encoding="utf-8-sig") as f:
        base_cfg = yaml.safe_load(f)
    model_dir = model_output_dir_name(str((base_cfg.get("llm", {}) or {}).get("model", "")))

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for dataset_name, spec in DATASET_SPECS.items():
        for size in [100, 200, 300]:
            cfg = copy.deepcopy(base_cfg)
            cfg["experiment"]["eval_path"] = f"data/benchmarks/external/sampled/{spec['sample_prefix']}_{size}.jsonl"
            cfg["experiment"]["output_dir"] = f"outputs/{model_dir}/{spec['output_prefix']}_{size}/search_general"
            cfg["search_grounding"]["cache_path"] = f"outputs/{model_dir}/{spec['output_prefix']}_{size}/search_cache_general.jsonl"
            cfg["llm"]["max_tokens"] = 64
            out_path = OUT_DIR / f"config_{dataset_name}_{size}.yaml"
            with out_path.open("w", encoding="utf-8") as f:
                yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
            print(f"[external-config] wrote {out_path}")


if __name__ == "__main__":
    main()
