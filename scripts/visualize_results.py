import argparse
import json
import os
from collections import defaultdict
from typing import Dict, List

import matplotlib.pyplot as plt


def load_jsonl(path: str) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def acc(rows: List[Dict]) -> float:
    if not rows:
        return 0.0
    return sum(1 for r in rows if str(r.get("pred", "")) == str(r.get("answer", ""))) / len(rows)


def by_dataset_acc(rows: List[Dict]) -> Dict[str, float]:
    g = defaultdict(list)
    for r in rows:
        ds = str(r.get("dataset", "unknown"))
        g[ds].append(r)
    return {k: acc(v) for k, v in sorted(g.items())}


def by_dataset_n(rows: List[Dict]) -> Dict[str, int]:
    g = defaultdict(int)
    for r in rows:
        ds = str(r.get("dataset", "unknown"))
        g[ds] += 1
    return dict(sorted(g.items()))


def evidence_len_stats(rows: List[Dict]) -> List[int]:
    vals = []
    for r in rows:
        ev = r.get("evidence", [])
        if isinstance(ev, list):
            vals.append(sum(len(str(x)) for x in ev))
        else:
            vals.append(0)
    return vals


def plot_overall_acc(metrics: Dict[str, float], out_path: str) -> None:
    keys = [k for k in ["vanilla_acc", "kb_acc", "search_acc"] if k in metrics]
    vals = [metrics[k] for k in keys]
    labels = [k.replace("_acc", "") for k in keys]

    plt.figure(figsize=(6, 4))
    bars = plt.bar(labels, vals)
    plt.ylim(0, 1.05)
    plt.title("Overall Accuracy")
    plt.ylabel("Accuracy")
    for b, v in zip(bars, vals):
        plt.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_dataset_grouped(acc_map: Dict[str, Dict[str, float]], out_path: str) -> None:
    datasets = sorted({d for sys_name in acc_map for d in acc_map[sys_name].keys()})
    systems = sorted(acc_map.keys())
    if not datasets:
        return

    x = list(range(len(datasets)))
    width = 0.25 if len(systems) >= 3 else 0.35

    plt.figure(figsize=(9, 4.5))
    for i, sys_name in enumerate(systems):
        vals = [acc_map[sys_name].get(d, 0.0) for d in datasets]
        offset = (i - (len(systems) - 1) / 2) * width
        plt.bar([xx + offset for xx in x], vals, width=width, label=sys_name)

    plt.xticks(x, datasets)
    plt.ylim(0, 1.05)
    plt.ylabel("Accuracy")
    plt.title("Accuracy by Dataset")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_search_evidence_hist(evi_lens: List[int], out_path: str) -> None:
    if not evi_lens:
        return
    plt.figure(figsize=(7, 4))
    plt.hist(evi_lens, bins=20)
    plt.title("Search Evidence Length Distribution")
    plt.xlabel("Total evidence chars per sample")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def run(input_dir: str, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)

    metrics_path = os.path.join(input_dir, "metrics.json")
    vanilla_path = os.path.join(input_dir, "vanilla_predictions.jsonl")
    kb_path = os.path.join(input_dir, "kb_predictions.jsonl")
    search_path = os.path.join(input_dir, "search_predictions.jsonl")

    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    vanilla_rows = load_jsonl(vanilla_path) if os.path.exists(vanilla_path) else []
    kb_rows = load_jsonl(kb_path) if os.path.exists(kb_path) else []
    search_rows = load_jsonl(search_path) if os.path.exists(search_path) else []

    plot_overall_acc(metrics, os.path.join(out_dir, "overall_accuracy.png"))

    acc_map: Dict[str, Dict[str, float]] = {}
    if vanilla_rows:
        acc_map["vanilla"] = by_dataset_acc(vanilla_rows)
    if kb_rows:
        acc_map["kb"] = by_dataset_acc(kb_rows)
    if search_rows:
        acc_map["search"] = by_dataset_acc(search_rows)
    if acc_map:
        plot_dataset_grouped(acc_map, os.path.join(out_dir, "dataset_accuracy_grouped.png"))

    evi_lens = evidence_len_stats(search_rows)
    if evi_lens:
        plot_search_evidence_hist(evi_lens, os.path.join(out_dir, "search_evidence_len_hist.png"))

    summary = {
        "input_dir": input_dir,
        "num_samples": {
            "vanilla": len(vanilla_rows),
            "kb": len(kb_rows),
            "search": len(search_rows),
        },
        "overall_metrics": metrics,
        "dataset_counts": by_dataset_n(search_rows if search_rows else vanilla_rows),
        "dataset_acc": acc_map,
        "figures": [
            os.path.join(out_dir, "overall_accuracy.png"),
            os.path.join(out_dir, "dataset_accuracy_grouped.png"),
            os.path.join(out_dir, "search_evidence_len_hist.png"),
        ],
    }

    summary_path = os.path.join(out_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize experiment outputs.")
    parser.add_argument("--input-dir", default="outputs/benchmark_quick")
    parser.add_argument("--out-dir", default="outputs/figures")
    args = parser.parse_args()
    run(args.input_dir, args.out_dir)
