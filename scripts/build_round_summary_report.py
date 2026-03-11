import json
import os
from dataclasses import dataclass
from typing import Dict, List

import matplotlib.pyplot as plt


@dataclass
class RoundEntry:
    round_id: str
    name: str
    stats_path: str
    opt_position: str
    opt_method: str


def _load_stats(path: str) -> Dict:
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)["comparison_search_selective_vs_base"]


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def main() -> None:
    out_dir = "outputs/optimization_round5_optimal_scheme/round_summary"
    _ensure_dir(out_dir)

    rounds: List[RoundEntry] = [
        RoundEntry(
            round_id="R1",
            name="Round1 Clean Stable Baseline",
            stats_path="outputs/strict_openai_recheck/benchmark_cleaning/stats_clean_stable.json",
            opt_position="Benchmark/Data",
            opt_method="Dataset quality audit + clean-stable subset lock (remove unstable/noisy items).",
        ),
        RoundEntry(
            round_id="R2",
            name="Round2 Search Layer (best=v16)",
            stats_path="outputs/optimization_round2_search_layer/search_selective_searchlayer_opt_ddgs_tuned_v16_gpt52/stats_vs_vanilla.json",
            opt_position="Search/Retrieval",
            opt_method="Dataset overrides, label-task force-use control, snippet fallback retrieval.",
        ),
        RoundEntry(
            round_id="R3",
            name="Round3 Stagewise (best=v17d)",
            stats_path="outputs/optimization_round3_stagewise/search_selective_v17d_balanced/stats_vs_vanilla.json",
            opt_position="Ranking/Gating",
            opt_method="snippet_only penalty + domain priors + per-dataset evidence gates.",
        ),
        RoundEntry(
            round_id="R4",
            name="Round4 Query Fix (v18)",
            stats_path="outputs/optimization_round4_query_fix/search_selective_v18_query_fix/stats_vs_vanilla.json",
            opt_position="Query Rewriting",
            opt_method="Claim cleaning, label-semantic query templates, semantic-overlap retry.",
        ),
        RoundEntry(
            round_id="R5",
            name="Round5 Optimal Scheme (best=v19d)",
            stats_path="outputs/optimization_round5_optimal_scheme/search_selective_v19d_no_force_label/stats_vs_vanilla_clean_stable.json",
            opt_position="Search+Rank+Gate",
            opt_method="Mixed label queries + semantic/noise scoring + stricter label evidence gating.",
        ),
    ]

    loaded: List[Dict] = []
    for r in rounds:
        s = _load_stats(r.stats_path)
        loaded.append(
            {
                "round_id": r.round_id,
                "name": r.name,
                "stats_path": r.stats_path,
                "opt_position": r.opt_position,
                "opt_method": r.opt_method,
                "overall_base": s["overall"]["base_acc"],
                "overall_test": s["overall"]["test_acc"],
                "overall_diff": s["overall"]["acc_diff_test_minus_base"],
                "p_value": s["overall"]["mcnemar"]["p_value_exact"],
                "BLEnD_diff": s["by_dataset"]["BLEnD"]["acc_diff_test_minus_base"],
                "NormAd_diff": s["by_dataset"]["NormAd"]["acc_diff_test_minus_base"],
                "SeeGULL_diff": s["by_dataset"]["SeeGULL"]["acc_diff_test_minus_base"],
                "BLEnD_base": s["by_dataset"]["BLEnD"]["base_acc"],
                "NormAd_base": s["by_dataset"]["NormAd"]["base_acc"],
                "SeeGULL_base": s["by_dataset"]["SeeGULL"]["base_acc"],
                "BLEnD_test": s["by_dataset"]["BLEnD"]["test_acc"],
                "NormAd_test": s["by_dataset"]["NormAd"]["test_acc"],
                "SeeGULL_test": s["by_dataset"]["SeeGULL"]["test_acc"],
            }
        )

    baseline = loaded[0]["overall_base"]
    round_labels = ["Baseline"] + [x["round_id"] for x in loaded]
    overall_vals = [baseline] + [x["overall_test"] for x in loaded]

    # Figure 1: overall trend
    fig1 = plt.figure(figsize=(9, 4.8))
    ax1 = fig1.add_subplot(111)
    ax1.plot(round_labels, overall_vals, marker="o", linewidth=2.2, color="#1f77b4")
    ax1.axhline(y=baseline, color="#444444", linestyle="--", linewidth=1.2, label=f"Vanilla Baseline={baseline:.3f}")
    for i, v in enumerate(overall_vals):
        ax1.text(i, v + 0.004, f"{v:.3f}", ha="center", fontsize=9)
    ax1.set_ylim(0.62, 0.75)
    ax1.set_title("Overall Accuracy vs Baseline Across Rounds")
    ax1.set_ylabel("Accuracy")
    ax1.legend(loc="lower right")
    ax1.grid(alpha=0.25)
    fig1.tight_layout()
    fig1_path = os.path.join(out_dir, "overall_trend_vs_baseline.png")
    fig1.savefig(fig1_path, dpi=160)
    plt.close(fig1)

    # Figure 2: dataset deltas by round
    fig2 = plt.figure(figsize=(10, 5.2))
    ax2 = fig2.add_subplot(111)
    x = list(range(len(loaded)))
    w = 0.24
    blend = [r["BLEnD_diff"] for r in loaded]
    normad = [r["NormAd_diff"] for r in loaded]
    seegull = [r["SeeGULL_diff"] for r in loaded]
    ax2.bar([i - w for i in x], blend, width=w, label="BLEnD", color="#2ca02c")
    ax2.bar(x, normad, width=w, label="NormAd", color="#1f77b4")
    ax2.bar([i + w for i in x], seegull, width=w, label="SeeGULL", color="#d62728")
    ax2.axhline(y=0.0, color="#444444", linewidth=1.0)
    ax2.set_xticks(x)
    ax2.set_xticklabels([r["round_id"] for r in loaded])
    ax2.set_title("Delta vs Baseline by Dataset (Search - Vanilla)")
    ax2.set_ylabel("Accuracy Delta")
    ax2.legend(loc="lower right")
    ax2.grid(axis="y", alpha=0.25)
    fig2.tight_layout()
    fig2_path = os.path.join(out_dir, "dataset_delta_vs_baseline.png")
    fig2.savefig(fig2_path, dpi=160)
    plt.close(fig2)

    # Figures 3-5: per-benchmark absolute accuracy by round
    per_dataset_plots: Dict[str, str] = {}
    datasets = [
        ("BLEnD", "BLEnD_base", "BLEnD_test", "#2ca02c"),
        ("NormAd", "NormAd_base", "NormAd_test", "#1f77b4"),
        ("SeeGULL", "SeeGULL_base", "SeeGULL_test", "#d62728"),
    ]
    for ds_name, ds_base_key, ds_test_key, color in datasets:
        ds_base = loaded[0][ds_base_key]
        ds_vals = [ds_base] + [r[ds_test_key] for r in loaded]
        fig = plt.figure(figsize=(9, 4.8))
        ax = fig.add_subplot(111)
        ax.plot(round_labels, ds_vals, marker="o", linewidth=2.2, color=color)
        ax.axhline(y=ds_base, color="#444444", linestyle="--", linewidth=1.2, label=f"{ds_name} Baseline={ds_base:.3f}")
        y_min = min(ds_vals) - 0.06
        y_max = max(ds_vals) + 0.06
        ax.set_ylim(max(0.0, y_min), min(1.0, y_max))
        for i, v in enumerate(ds_vals):
            ax.text(i, v + 0.004, f"{v:.3f}", ha="center", fontsize=9)
        ax.set_title(f"{ds_name} Accuracy vs Baseline Across Rounds")
        ax.set_ylabel("Accuracy")
        ax.legend(loc="lower right")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        ds_path = os.path.join(out_dir, f"{ds_name.lower()}_accuracy_vs_baseline.png")
        fig.savefig(ds_path, dpi=160)
        plt.close(fig)
        per_dataset_plots[ds_name] = ds_path

    summary_json_path = os.path.join(out_dir, "round_summary_data.json")
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump({"baseline_acc": baseline, "rounds": loaded}, f, ensure_ascii=False, indent=2)

    md_path = os.path.join(out_dir, "ROUND_SUMMARY_REPORT.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Round-by-Round Summary vs Baseline\n\n")
        f.write(f"- Vanilla baseline accuracy: `{baseline:.4f}`\n")
        f.write("- Eval set: `eval_clean_stable` (n=276)\n\n")
        f.write("## Visualizations\n\n")
        fig1_link = fig1_path.replace("\\", "/")
        fig2_link = fig2_path.replace("\\", "/")
        blend_link = per_dataset_plots["BLEnD"].replace("\\", "/")
        normad_link = per_dataset_plots["NormAd"].replace("\\", "/")
        seegull_link = per_dataset_plots["SeeGULL"].replace("\\", "/")
        f.write(f"- Overall trend: `[{os.path.basename(fig1_path)}]({fig1_link})`\n")
        f.write(f"- Dataset deltas: `[{os.path.basename(fig2_path)}]({fig2_link})`\n")
        f.write(f"- BLEnD accuracy: `[{os.path.basename(per_dataset_plots['BLEnD'])}]({blend_link})`\n")
        f.write(f"- NormAd accuracy: `[{os.path.basename(per_dataset_plots['NormAd'])}]({normad_link})`\n")
        f.write(f"- SeeGULL accuracy: `[{os.path.basename(per_dataset_plots['SeeGULL'])}]({seegull_link})`\n\n")
        f.write("## Results Table\n\n")
        f.write("| Round | Optimization Position | Optimization Method | Overall | Delta vs Baseline | BLEnD Delta | NormAd Delta | SeeGULL Delta | p-value |\n")
        f.write("|---|---|---|---:|---:|---:|---:|---:|---:|\n")
        for r in loaded:
            f.write(
                f"| {r['round_id']} | {r['opt_position']} | {r['opt_method']} | "
                f"{r['overall_test']:.4f} | {r['overall_diff']:+.4f} | "
                f"{r['BLEnD_diff']:+.4f} | {r['NormAd_diff']:+.4f} | {r['SeeGULL_diff']:+.4f} | "
                f"{r['p_value']:.4f} |\n"
            )
        f.write("\n## Key Takeaways\n\n")
        f.write("- Round5 (v19d) is the best overall (`0.7246`, `+0.0543` vs baseline, p=0.0357).\n")
        f.write("- BLEnD and NormAd are consistently improved in later rounds.\n")
        f.write("- SeeGULL remains the hardest and is usually below baseline; Round5 narrows the gap to `-0.01`.\n")

    print(
        json.dumps(
            {
                "out_dir": out_dir,
                "overall_plot": fig1_path,
                "delta_plot": fig2_path,
                "blend_plot": per_dataset_plots["BLEnD"],
                "normad_plot": per_dataset_plots["NormAd"],
                "seegull_plot": per_dataset_plots["SeeGULL"],
                "report_md": md_path,
                "summary_json": summary_json_path,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
