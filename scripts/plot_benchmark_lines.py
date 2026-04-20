import argparse
import json
import os
from typing import Dict, List

import matplotlib.pyplot as plt


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def run(summary_path: str, out_dir: str) -> None:
    with open(summary_path, "r", encoding="utf-8-sig") as f:
        summary = json.load(f)

    runs: List[Dict] = list(summary.get("runs", []) or [])
    if not runs:
        raise ValueError("No runs found in summary.")

    benchmarks = sorted(
        {k for r in runs for k in (r.get("dataset_accuracy", {}) or {}).keys()},
        key=lambda x: {"BLEnD": 0, "NormAd": 1, "SeeGULL": 2}.get(x, 99),
    )
    labels = [str(r.get("name", "")) for r in runs]
    ds_values: Dict[str, List[float]] = {}
    for ds in benchmarks:
        vals = []
        for r in runs:
            vals.append(float((r.get("dataset_accuracy", {}) or {}).get(ds, 0.0) or 0.0))
        ds_values[ds] = vals

    ensure_dir(out_dir)

    # Single combined figure.
    plt.figure(figsize=(10.5, 5.0))
    for ds in benchmarks:
        plt.plot(labels, ds_values[ds], marker="o", linewidth=2, label=ds)
    plt.ylim(0, 1.02)
    plt.ylabel("Accuracy")
    plt.title("Accuracy Trends by Benchmark")
    plt.xticks(rotation=25, ha="right")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    combined_path = os.path.join(out_dir, "benchmark_lines_combined.png")
    plt.savefig(combined_path, dpi=180)
    plt.close()

    # One figure per benchmark.
    for ds in benchmarks:
        plt.figure(figsize=(10.0, 4.2))
        vals = ds_values[ds]
        plt.plot(labels, vals, marker="o", linewidth=2, color="#1f77b4")
        for x, y in zip(labels, vals):
            plt.text(x, y + 0.012, f"{y:.3f}", ha="center", fontsize=8)
        plt.ylim(0, 1.02)
        plt.ylabel("Accuracy")
        plt.title(f"{ds} Accuracy Across Systems")
        plt.xticks(rotation=25, ha="right")
        plt.grid(axis="y", alpha=0.25)
        plt.tight_layout()
        out_path = os.path.join(out_dir, f"benchmark_line_{ds}.png")
        plt.savefig(out_path, dpi=180)
        plt.close()

    print(
        json.dumps(
            {
                "summary_path": summary_path,
                "out_dir": out_dir,
                "benchmarks": benchmarks,
                "files": ["benchmark_lines_combined.png"]
                + [f"benchmark_line_{ds}.png" for ds in benchmarks],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot benchmark-level line charts from summary json.")
    parser.add_argument("--summary", required=True, help="Path to component_summary.json or similar.")
    parser.add_argument("--out-dir", required=True, help="Output directory for plots.")
    args = parser.parse_args()
    run(args.summary, args.out_dir)

