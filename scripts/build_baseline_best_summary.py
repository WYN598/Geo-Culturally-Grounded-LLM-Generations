import json
import os
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def accuracy(rows: List[Dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    correct = 0
    for r in rows:
        if str(r.get("pred", "")) == str(r.get("answer", "")):
            correct += 1
    return correct / len(rows)


def acc_by_dataset(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        ds = str(r.get("dataset", "unknown"))
        groups.setdefault(ds, []).append(r)
    return {k: accuracy(v) for k, v in sorted(groups.items())}


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def plot_dataset_acc(
    baseline: Dict[str, float],
    v16: Dict[str, float],
    best: Dict[str, float],
    out_path: str,
) -> None:
    datasets = sorted(set(baseline.keys()) | set(v16.keys()) | set(best.keys()))
    x = list(range(len(datasets)))
    width = 0.24
    vals_base = [baseline.get(d, 0.0) for d in datasets]
    vals_v16 = [v16.get(d, 0.0) for d in datasets]
    vals_best = [best.get(d, 0.0) for d in datasets]

    plt.figure(figsize=(9, 5))
    plt.bar([i - width for i in x], vals_base, width=width, label="Vanilla Baseline")
    plt.bar(x, vals_v16, width=width, label="Search v16")
    plt.bar([i + width for i in x], vals_best, width=width, label="Search v17d (Best)")
    plt.xticks(x, datasets)
    plt.ylim(0, 1.0)
    plt.ylabel("Accuracy")
    plt.title("Accuracy by Dataset")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_delta_vs_baseline(baseline: Dict[str, float], best: Dict[str, float], out_path: str) -> None:
    datasets = sorted(set(baseline.keys()) | set(best.keys()))
    deltas = [best.get(d, 0.0) - baseline.get(d, 0.0) for d in datasets]
    colors = ["#2ca02c" if d >= 0 else "#d62728" for d in deltas]

    plt.figure(figsize=(8, 4.8))
    bars = plt.bar(datasets, deltas, color=colors)
    plt.axhline(0.0, color="black", linewidth=1)
    plt.ylabel("Delta Accuracy (v17d - Vanilla)")
    plt.title("Improvement by Dataset")
    for b, v in zip(bars, deltas):
        plt.text(b.get_x() + b.get_width() / 2, v + (0.01 if v >= 0 else -0.03), f"{v:+.3f}", ha="center")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_failure_buckets(v16_scope: Dict[str, Any], best_scope: Dict[str, Any], out_path: str) -> None:
    labels = ["v16", "v17d"]
    buckets = ["search_fail", "rank_fail", "context_or_reasoning_fail"]
    v16_vals = [float(v16_scope["failure_rate"][b]) for b in buckets]
    best_vals = [float(best_scope["failure_rate"][b]) for b in buckets]
    vals = [v16_vals, best_vals]

    plt.figure(figsize=(8.5, 4.8))
    bottoms = [0.0, 0.0]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    names = ["Search Fail", "Rank Fail", "Context Fail"]
    for i, _ in enumerate(buckets):
        cur = [vals[0][i], vals[1][i]]
        plt.bar(labels, cur, bottom=bottoms, color=colors[i], label=names[i])
        bottoms = [bottoms[j] + cur[j] for j in range(2)]
    plt.ylabel("Failure Rate")
    plt.title("Failure Bucket Comparison (Overall)")
    plt.ylim(0, 0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_token_compare(v16_usage: Dict[str, Any], best_usage: Dict[str, Any], out_path: str) -> None:
    labels = ["Prompt", "Completion", "Total"]
    v16_vals = [
        int(v16_usage.get("prompt_tokens", 0)),
        int(v16_usage.get("completion_tokens", 0)),
        int(v16_usage.get("total_tokens", 0)),
    ]
    best_vals = [
        int(best_usage.get("prompt_tokens", 0)),
        int(best_usage.get("completion_tokens", 0)),
        int(best_usage.get("total_tokens", 0)),
    ]
    x = list(range(len(labels)))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))
    ax1.bar([i - width / 2 for i in x], v16_vals, width=width, label="v16")
    ax1.bar([i + width / 2 for i in x], best_vals, width=width, label="v17d")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_title("Token Usage")
    ax1.legend()

    v16_fallback = int(v16_usage.get("by_stage", {}).get("search_answer_fallback", {}).get("num_calls", 0))
    v16_aug = int(v16_usage.get("by_stage", {}).get("search_answer_augmented", {}).get("num_calls", 0))
    best_fallback = int(best_usage.get("by_stage", {}).get("search_answer_fallback", {}).get("num_calls", 0))
    best_aug = int(best_usage.get("by_stage", {}).get("search_answer_augmented", {}).get("num_calls", 0))

    ax2.bar(["v16", "v17d"], [v16_aug, best_aug], label="augmented_calls")
    ax2.bar(["v16", "v17d"], [v16_fallback, best_fallback], bottom=[v16_aug, best_aug], label="fallback_calls")
    ax2.set_title("Call Mix (Augmented vs Fallback)")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_tuning_timeline(acc_map: Dict[str, float], out_path: str) -> None:
    order = ["v16", "v17a", "v17b", "v17c", "v17d"]
    xs = [k for k in order if k in acc_map]
    ys = [acc_map[k] for k in xs]
    plt.figure(figsize=(8, 4.5))
    plt.plot(xs, ys, marker="o")
    for x, y in zip(xs, ys):
        plt.text(x, y + 0.004, f"{y:.3f}", ha="center")
    plt.ylim(min(ys) - 0.02, max(ys) + 0.03)
    plt.ylabel("Overall Accuracy")
    plt.title("Optimization Timeline")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def main() -> None:
    root = Path("outputs")
    out_dir = root / "optimization_round3_stagewise" / "baseline_best_summary"
    fig_dir = out_dir / "figures"
    ensure_dir(str(fig_dir))

    vanilla_path = root / "optimization_round1_clean_stable" / "vanilla_clean_stable.jsonl"
    v16_pred = root / "optimization_round2_search_layer" / "search_selective_searchlayer_opt_ddgs_tuned_v16_gpt52" / "search_predictions.jsonl"
    best_pred = root / "optimization_round3_stagewise" / "search_selective_v17d_balanced" / "search_predictions.jsonl"

    v16_metrics_path = root / "optimization_round2_search_layer" / "search_selective_searchlayer_opt_ddgs_tuned_v16_gpt52" / "metrics.json"
    best_metrics_path = root / "optimization_round3_stagewise" / "search_selective_v17d_balanced" / "metrics.json"

    v16_step_path = root / "optimization_round2_search_layer" / "search_selective_searchlayer_opt_ddgs_tuned_v16_gpt52" / "stepwise_optimization_report.json"
    best_step_path = root / "optimization_round3_stagewise" / "search_selective_v17d_balanced" / "stepwise_optimization_report.json"

    stats_vs_vanilla_path = root / "optimization_round3_stagewise" / "search_selective_v17d_balanced" / "stats_vs_vanilla.json"
    stats_vs_v16_path = root / "optimization_round3_stagewise" / "search_selective_v17d_balanced" / "stats_vs_v16.json"

    vanilla_rows = load_jsonl(str(vanilla_path))
    v16_rows = load_jsonl(str(v16_pred))
    best_rows = load_jsonl(str(best_pred))

    baseline_acc_ds = acc_by_dataset(vanilla_rows)
    v16_acc_ds = acc_by_dataset(v16_rows)
    best_acc_ds = acc_by_dataset(best_rows)

    v16_metrics = load_json(str(v16_metrics_path))
    best_metrics = load_json(str(best_metrics_path))
    v16_step = load_json(str(v16_step_path))
    best_step = load_json(str(best_step_path))
    stats_vs_vanilla = load_json(str(stats_vs_vanilla_path))
    stats_vs_v16 = load_json(str(stats_vs_v16_path))

    # timeline from existing runs
    timeline_paths = {
        "v16": root / "optimization_round2_search_layer" / "search_selective_searchlayer_opt_ddgs_tuned_v16_gpt52" / "metrics.json",
        "v17a": root / "optimization_round3_stagewise" / "search_selective_v17a_quality_prior" / "metrics.json",
        "v17b": root / "optimization_round3_stagewise" / "search_selective_v17b_strict_normad" / "metrics.json",
        "v17c": root / "optimization_round3_stagewise" / "search_selective_v17c_light_normad" / "metrics.json",
        "v17d": root / "optimization_round3_stagewise" / "search_selective_v17d_balanced" / "metrics.json",
    }
    timeline_acc: Dict[str, float] = {}
    for k, p in timeline_paths.items():
        if p.exists():
            timeline_acc[k] = float(load_json(str(p)).get("search_acc", 0.0))

    # plots
    plot_dataset_acc(baseline_acc_ds, v16_acc_ds, best_acc_ds, str(fig_dir / "acc_by_dataset.png"))
    plot_delta_vs_baseline(baseline_acc_ds, best_acc_ds, str(fig_dir / "delta_vs_baseline.png"))
    plot_failure_buckets(
        v16_step["scopes"]["overall"],
        best_step["scopes"]["overall"],
        str(fig_dir / "failure_bucket_v16_vs_v17d.png"),
    )
    plot_token_compare(
        v16_metrics["search_usage"],
        best_metrics["search_usage"],
        str(fig_dir / "token_compare_v16_vs_v17d.png"),
    )
    plot_tuning_timeline(timeline_acc, str(fig_dir / "tuning_timeline.png"))

    overall_baseline = accuracy(vanilla_rows)
    overall_v16 = accuracy(v16_rows)
    overall_best = accuracy(best_rows)

    overall_delta_vs_baseline = overall_best - overall_baseline
    overall_delta_vs_v16 = overall_best - overall_v16
    tokens_v16 = int(v16_metrics["search_usage"]["total_tokens"])
    tokens_best = int(best_metrics["search_usage"]["total_tokens"])
    token_change = (tokens_best - tokens_v16) / max(tokens_v16, 1)

    ov_stats_vanilla = stats_vs_vanilla["comparison_search_selective_vs_base"]["overall"]
    ov_stats_v16 = stats_vs_v16["comparison_search_selective_vs_base"]["overall"]

    report_path = out_dir / "SUMMARY_BASELINE_TO_BEST.md"
    md = []
    md.append("# Baseline to Best Summary (Search-Grounding)")
    md.append("")
    md.append("## 1) Scope")
    md.append("- Baseline: `vanilla` on `eval_clean_stable.jsonl`")
    md.append("- Search reference: `v16`")
    md.append("- Current best: `v17d`")
    md.append("- Model: `gpt-5.2`, temperature `0.0`, same benchmark split.")
    md.append("")
    md.append("## 2) What was optimized")
    md.append("- Search-layer robustness: snippet fallback for failed page fetch (`snippet_only` chunks).")
    md.append("- Query quality on label tasks: claim-aware query expansion + retry.")
    md.append("- Label-noise filtering: remove annotation/tutorial-style noisy pages for label tasks.")
    md.append("- Ranking quality: low-quality domain priors + `snippet_only_penalty` in lexical ranking.")
    md.append("- Dataset-aware evidence gating: stricter NormAd/BLEnD thresholds to reduce noisy augmentation.")
    md.append("")
    md.append("## 3) Metric changes")
    md.append(f"- Overall accuracy: baseline `{overall_baseline:.4f}` -> v16 `{overall_v16:.4f}` -> best(v17d) `{overall_best:.4f}`")
    md.append(f"- Delta vs baseline: `{overall_delta_vs_baseline:+.4f}`")
    md.append(f"- Delta vs v16: `{overall_delta_vs_v16:+.4f}`")
    md.append("")
    md.append("| Dataset | Baseline | v16 | v17d (best) | Delta(v17d-baseline) |")
    md.append("|---|---:|---:|---:|---:|")
    for ds in sorted(best_acc_ds.keys()):
        b = baseline_acc_ds.get(ds, 0.0)
        s16 = v16_acc_ds.get(ds, 0.0)
        bst = best_acc_ds.get(ds, 0.0)
        md.append(f"| {ds} | {b:.4f} | {s16:.4f} | {bst:.4f} | {bst-b:+.4f} |")
    md.append("")
    md.append("## 4) Statistical notes")
    md.append(
        f"- v17d vs baseline McNemar p-value: `{ov_stats_vanilla['mcnemar']['p_value_exact']:.4f}`; "
        f"acc diff CI95: `{ov_stats_vanilla['acc_diff_bootstrap_ci95'][0]:+.4f} .. {ov_stats_vanilla['acc_diff_bootstrap_ci95'][1]:+.4f}`."
    )
    md.append(
        f"- v17d vs v16 McNemar p-value: `{ov_stats_v16['mcnemar']['p_value_exact']:.4f}`; "
        f"acc diff CI95: `{ov_stats_v16['acc_diff_bootstrap_ci95'][0]:+.4f} .. {ov_stats_v16['acc_diff_bootstrap_ci95'][1]:+.4f}`."
    )
    md.append("")
    md.append("## 5) Failure decomposition")
    v16_over = v16_step["scopes"]["overall"]["failure_rate"]
    bst_over = best_step["scopes"]["overall"]["failure_rate"]
    md.append(f"- Search fail: v16 `{v16_over['search_fail']:.4f}` -> v17d `{bst_over['search_fail']:.4f}`")
    md.append(f"- Rank fail: v16 `{v16_over['rank_fail']:.4f}` -> v17d `{bst_over['rank_fail']:.4f}`")
    md.append(f"- Context fail: v16 `{v16_over['context_or_reasoning_fail']:.4f}` -> v17d `{bst_over['context_or_reasoning_fail']:.4f}`")
    md.append("- Main gain comes from reducing context failures; search failures remain the largest bottleneck.")
    md.append("")
    md.append("## 6) Cost / token usage")
    md.append(f"- Total tokens: v16 `{tokens_v16}` -> v17d `{tokens_best}` ({token_change:+.2%}).")
    md.append("- v17d uses more fallback calls and fewer augmented calls than v16.")
    md.append("")
    md.append("## 7) Visualizations")
    md.append("### Accuracy by dataset")
    md.append("![acc_by_dataset](figures/acc_by_dataset.png)")
    md.append("")
    md.append("### Delta vs baseline")
    md.append("![delta_vs_baseline](figures/delta_vs_baseline.png)")
    md.append("")
    md.append("### Failure bucket comparison")
    md.append("![failure_bucket_v16_vs_v17d](figures/failure_bucket_v16_vs_v17d.png)")
    md.append("")
    md.append("### Token comparison")
    md.append("![token_compare_v16_vs_v17d](figures/token_compare_v16_vs_v17d.png)")
    md.append("")
    md.append("### Tuning timeline")
    md.append("![tuning_timeline](figures/tuning_timeline.png)")
    md.append("")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))

    machine_summary = {
        "report_path": str(report_path),
        "figures_dir": str(fig_dir),
        "overall": {
            "baseline": overall_baseline,
            "v16": overall_v16,
            "best_v17d": overall_best,
            "delta_vs_baseline": overall_delta_vs_baseline,
            "delta_vs_v16": overall_delta_vs_v16,
        },
        "tokens": {
            "v16_total": tokens_v16,
            "v17d_total": tokens_best,
            "change_ratio": token_change,
        },
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(machine_summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(machine_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
