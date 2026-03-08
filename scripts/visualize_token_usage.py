import argparse
import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def parse_usage_arg(values: List[str]) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for v in values:
        if "=" in v:
            label, path = v.split("=", 1)
            out.append((label.strip(), path.strip()))
        else:
            path = v.strip()
            label = os.path.splitext(os.path.basename(path))[0]
            out.append((label, path))
    return out


def aggregate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    summary = {
        "num_calls": len(rows),
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "by_stage": defaultdict(lambda: {"num_calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}),
    }
    for r in rows:
        p = int(r.get("prompt_tokens", 0) or 0)
        c = int(r.get("completion_tokens", 0) or 0)
        t = int(r.get("total_tokens", 0) or (p + c))
        stage = str(r.get("stage", "unknown"))
        summary["prompt_tokens"] += p
        summary["completion_tokens"] += c
        summary["total_tokens"] += t
        bs = summary["by_stage"][stage]
        bs["num_calls"] += 1
        bs["prompt_tokens"] += p
        bs["completion_tokens"] += c
        bs["total_tokens"] += t
    summary["by_stage"] = dict(summary["by_stage"])
    return summary


def plot_overall(summaries: Dict[str, Dict[str, Any]], out_path: str) -> None:
    labels = list(summaries.keys())
    prompt_vals = [summaries[k]["prompt_tokens"] for k in labels]
    completion_vals = [summaries[k]["completion_tokens"] for k in labels]

    plt.figure(figsize=(8, 4.5))
    plt.bar(labels, prompt_vals, label="prompt_tokens")
    plt.bar(labels, completion_vals, bottom=prompt_vals, label="completion_tokens")
    plt.ylabel("Tokens")
    plt.title("Token Usage by Run")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_stage_breakdown(label: str, summary: Dict[str, Any], out_path: str, top_n: int = 12) -> None:
    by_stage = summary.get("by_stage", {})
    items = sorted(by_stage.items(), key=lambda x: x[1].get("total_tokens", 0), reverse=True)[:top_n]
    if not items:
        return
    stages = [k for k, _ in items]
    totals = [v.get("total_tokens", 0) for _, v in items]
    calls = [v.get("num_calls", 0) for _, v in items]

    fig, ax1 = plt.subplots(figsize=(10, 4.8))
    ax1.bar(stages, totals, color="#4C72B0")
    ax1.set_ylabel("Total Tokens", color="#4C72B0")
    ax1.tick_params(axis="x", rotation=35)
    ax1.set_title(f"Stage Token Breakdown: {label}")

    ax2 = ax1.twinx()
    ax2.plot(stages, calls, color="#DD8452", marker="o")
    ax2.set_ylabel("Num Calls", color="#DD8452")

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    usage_items = parse_usage_arg(args.usage)
    summaries: Dict[str, Dict[str, Any]] = {}

    for label, path in usage_items:
        if not os.path.exists(path):
            continue
        rows = load_jsonl(path)
        summaries[label] = aggregate(rows)

    os.makedirs(args.out_dir, exist_ok=True)
    summary_path = os.path.join(args.out_dir, "token_usage_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)

    if summaries:
        plot_overall(summaries, os.path.join(args.out_dir, "token_usage_overall.png"))
        for label, summary in summaries.items():
            plot_stage_breakdown(
                label,
                summary,
                os.path.join(args.out_dir, f"token_usage_stage_{label}.png"),
                top_n=args.top_n_stages,
            )

    print(
        json.dumps(
            {
                "out_dir": args.out_dir,
                "summary_path": summary_path,
                "num_runs": len(summaries),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize LLM token usage logs.")
    parser.add_argument(
        "--usage",
        nargs="+",
        required=True,
        help="One or more usage log files. Format: label=path or path",
    )
    parser.add_argument("--out-dir", default="outputs/token_usage")
    parser.add_argument("--top-n-stages", type=int, default=12)
    run(parser.parse_args())
