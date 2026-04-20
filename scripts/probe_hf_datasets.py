import argparse
import json
from typing import Any, Dict, List

from datasets import get_dataset_config_names, load_dataset


def try_load(dataset_id: str, config: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "dataset": dataset_id,
        "config": config,
        "ok": False,
        "detail": "",
        "splits": [],
    }
    try:
        if config:
            ds = load_dataset(dataset_id, config)
        else:
            ds = load_dataset(dataset_id)
        out["ok"] = True
        if hasattr(ds, "keys"):
            splits = list(ds.keys())
            out["splits"] = splits
            out["detail"] = ",".join(splits)
        else:
            out["detail"] = f"rows={len(ds)}"
    except Exception as e:
        out["detail"] = str(e).split("\n")[0][:240]
    return out


def run(args: argparse.Namespace) -> None:
    probes: List[Dict[str, Any]] = []
    for dataset_id in args.datasets:
        if args.with_configs:
            try:
                cfgs = get_dataset_config_names(dataset_id)
            except Exception:
                cfgs = []
            if cfgs:
                for cfg in cfgs[: args.max_configs]:
                    probes.append(try_load(dataset_id, cfg))
                continue
        probes.append(try_load(dataset_id))
    print(json.dumps({"probes": probes}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Probe Hugging Face datasets for accessibility.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        help="dataset IDs to probe",
    )
    parser.add_argument("--with-configs", action="store_true")
    parser.add_argument("--max-configs", type=int, default=12)
    run(parser.parse_args())

