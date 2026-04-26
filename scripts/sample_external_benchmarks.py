import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def balanced_sample(rows: List[Dict[str, Any]], limit: int, label_key: str, seed: int) -> List[Dict[str, Any]]:
    if limit <= 0 or len(rows) <= limit:
        return rows
    by_label: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        key = str(row.get(label_key, "unknown"))
        by_label.setdefault(key, []).append(row)
    rng = random.Random(seed)
    labels = sorted(by_label.keys())
    for label in labels:
        rng.shuffle(by_label[label])

    # Round-robin allocation keeps the sample as balanced as possible even when
    # the requested limit is smaller than the number of strata.
    selected: List[Dict[str, Any]] = []
    label_order = list(labels)
    rng.shuffle(label_order)
    offsets = {label: 0 for label in labels}
    while len(selected) < limit:
        added_this_round = False
        for label in label_order:
            idx = offsets[label]
            bucket = by_label[label]
            if idx >= len(bucket):
                continue
            selected.append(bucket[idx])
            offsets[label] = idx + 1
            added_this_round = True
            if len(selected) >= limit:
                break
        if not added_this_round:
            break
    rng.shuffle(selected)
    return selected


def random_sample(rows: List[Dict[str, Any]], limit: int, seed: int) -> List[Dict[str, Any]]:
    if limit <= 0 or len(rows) <= limit:
        return rows
    rng = random.Random(seed)
    out = list(rows)
    rng.shuffle(out)
    return out[:limit]


def run(args: argparse.Namespace) -> None:
    processed_root = Path(args.processed_root)
    out_root = Path(args.out_root)
    ensure_dir(out_root)

    specs = {
        "bbq": {"filename": "bbq.jsonl", "label_key": "answer"},
        "socialstigmaqa": {"filename": "socialstigmaqa.jsonl", "label_key": "biased_answer"},
        "truthfulqa": {"filename": "truthfulqa.jsonl", "label_key": "answer"},
        "popqa": {"filename": "popqa.jsonl", "label_key": None},
        "cbbq": {"filename": "cbbq.jsonl", "label_key": "sampling_bucket"},
        "borderlines": {"filename": "borderlines.jsonl", "label_key": "sampling_bucket"},
        "msqad": {"filename": "msqad.jsonl", "label_key": "sampling_bucket"},
        "culturalbench_easy": {"filename": "culturalbench_easy.jsonl", "label_key": "answer"},
        "culturalbench_hard": {"filename": "culturalbench_hard.jsonl", "label_key": "answer"},
        "espanstereo": {"filename": "espanstereo.jsonl", "label_key": "answer"},
        "honest": {"filename": "honest.jsonl", "label_key": "sampling_bucket"},
    }
    selected_specs = {name: specs[name] for name in args.datasets}

    combined: List[Dict[str, Any]] = []
    report: Dict[str, Any] = {"seed": args.seed, "per_dataset": args.per_dataset, "datasets": {}}

    for name, spec in selected_specs.items():
        rows = load_jsonl(processed_root / spec["filename"])
        if spec["label_key"]:
            sampled = balanced_sample(rows, args.per_dataset, spec["label_key"], args.seed)
        else:
            sampled = random_sample(rows, args.per_dataset, args.seed)
        out_path = out_root / f"{name}_{args.per_dataset}.jsonl"
        write_jsonl(out_path, sampled)
        combined.extend(sampled)
        report["datasets"][name] = {
            "input": len(rows),
            "sampled": len(sampled),
            "output": str(out_path),
        }

    combined_path = out_root / f"external_mix_{args.per_dataset}x{len(selected_specs)}.jsonl"
    write_jsonl(combined_path, combined)
    report["combined_output"] = str(combined_path)
    report_path = out_root / f"sampling_report_{args.per_dataset}x{len(selected_specs)}.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample external benchmarks into a balanced experimental subset.")
    parser.add_argument("--processed-root", default="data/benchmarks/external/processed")
    parser.add_argument("--out-root", default="data/benchmarks/external/sampled")
    parser.add_argument("--per-dataset", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["bbq", "socialstigmaqa", "truthfulqa", "popqa"],
        choices=[
            "bbq",
            "socialstigmaqa",
            "truthfulqa",
            "popqa",
            "cbbq",
            "borderlines",
            "msqad",
            "culturalbench_easy",
            "culturalbench_hard",
            "espanstereo",
            "honest",
        ],
    )
    args = parser.parse_args()
    run(args)
