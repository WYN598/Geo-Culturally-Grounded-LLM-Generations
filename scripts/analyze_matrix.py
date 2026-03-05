import argparse
import json
import os
from collections import defaultdict
from typing import Dict, List

import matplotlib.pyplot as plt


def load_jsonl(path: str) -> List[Dict]:
    rows: List[Dict] = []
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
        g[str(r.get("dataset", "unknown"))].append(r)
    return {k: acc(v) for k, v in sorted(g.items())}


def align_by_id(rows: List[Dict]) -> Dict[str, Dict]:
    return {str(r.get("id", "")): r for r in rows if str(r.get("id", ""))}


def win_tie_loss(base_rows: List[Dict], test_rows: List[Dict]) -> Dict[str, int]:
    b = align_by_id(base_rows)
    t = align_by_id(test_rows)
    ids = sorted(set(b.keys()) & set(t.keys()))

    win = tie = loss = 0
    for i in ids:
        bc = 1 if str(b[i].get("pred", "")) == str(b[i].get("answer", "")) else 0
        tc = 1 if str(t[i].get("pred", "")) == str(t[i].get("answer", "")) else 0
        if tc > bc:
            win += 1
        elif tc < bc:
            loss += 1
        else:
            tie += 1
    return {"win": win, "tie": tie, "loss": loss, "n": len(ids)}


def stereotype_rate(rows: List[Dict]) -> float:
    seegull = [r for r in rows if str(r.get("dataset", "")).lower() == "seegull"]
    if not seegull:
        return 0.0

    valid = []
    for r in seegull:
        choices = [str(c).lower() for c in (r.get("choices", []) or [])]
        if not choices:
            continue
        # Only valid when option A explicitly means stereotype.
        first = choices[0]
        if "stereotype" in first and "non-stereotype" not in first:
            valid.append(1.0 if str(r.get("pred", "")).upper() == "A" else 0.0)
    if not valid:
        return 0.0
    return sum(valid) / len(valid)


def plot_overall(overall: Dict[str, float], out_path: str) -> None:
    labels = list(overall.keys())
    vals = [overall[k] for k in labels]
    plt.figure(figsize=(6.5, 4))
    bars = plt.bar(labels, vals)
    plt.ylim(0, 1.05)
    plt.ylabel("Accuracy")
    plt.title("Overall Accuracy (Matrix)")
    for b, v in zip(bars, vals):
        plt.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_dataset_grouped(ds_map: Dict[str, Dict[str, float]], out_path: str) -> None:
    systems = list(ds_map.keys())
    datasets = sorted({d for m in ds_map.values() for d in m.keys()})
    if not datasets:
        return
    x = list(range(len(datasets)))
    width = 0.25

    plt.figure(figsize=(9.5, 4.5))
    for i, s in enumerate(systems):
        vals = [ds_map[s].get(d, 0.0) for d in datasets]
        offset = (i - (len(systems) - 1) / 2) * width
        plt.bar([xx + offset for xx in x], vals, width=width, label=s)

    plt.xticks(x, datasets)
    plt.ylim(0, 1.05)
    plt.ylabel("Accuracy")
    plt.title("Accuracy by Dataset")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_wtl(wtl: Dict[str, int], out_path: str, title: str) -> None:
    labels = ["win", "tie", "loss"]
    vals = [wtl.get("win", 0), wtl.get("tie", 0), wtl.get("loss", 0)]
    plt.figure(figsize=(6, 4))
    plt.bar(labels, vals)
    plt.title(title)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_stereotype(st_map: Dict[str, float], out_path: str) -> None:
    labels = list(st_map.keys())
    vals = [st_map[k] for k in labels]
    plt.figure(figsize=(6.5, 4))
    bars = plt.bar(labels, vals)
    plt.ylim(0, 1.05)
    plt.ylabel("Stereotype Selection Rate (SeeGULL)")
    plt.title("SeeGULL Stereotype Rate")
    for b, v in zip(bars, vals):
        plt.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def run(matrix_root: str, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)

    v_path = os.path.join(matrix_root, "vanilla", "vanilla_predictions.jsonl")
    ss_path = os.path.join(matrix_root, "search_selective", "search_predictions.jsonl")
    sn_path = os.path.join(matrix_root, "search_non_selective", "search_predictions.jsonl")

    vanilla = load_jsonl(v_path)
    search_sel = load_jsonl(ss_path)
    search_non = load_jsonl(sn_path)

    overall = {
        "vanilla": acc(vanilla),
        "search_selective": acc(search_sel),
        "search_non_selective": acc(search_non),
    }

    ds_map = {
        "vanilla": by_dataset_acc(vanilla),
        "search_selective": by_dataset_acc(search_sel),
        "search_non_selective": by_dataset_acc(search_non),
    }

    wtl_sel = win_tie_loss(vanilla, search_sel)
    wtl_non = win_tie_loss(vanilla, search_non)

    stereo = {
        "vanilla": stereotype_rate(vanilla),
        "search_selective": stereotype_rate(search_sel),
        "search_non_selective": stereotype_rate(search_non),
    }

    plot_overall(overall, os.path.join(out_dir, "overall_accuracy_matrix.png"))
    plot_dataset_grouped(ds_map, os.path.join(out_dir, "dataset_accuracy_matrix.png"))
    plot_wtl(wtl_sel, os.path.join(out_dir, "wtl_search_selective_vs_vanilla.png"), "Search Selective vs Vanilla")
    plot_wtl(wtl_non, os.path.join(out_dir, "wtl_search_non_selective_vs_vanilla.png"), "Search Non-Selective vs Vanilla")
    plot_stereotype(stereo, os.path.join(out_dir, "seegull_stereotype_rate.png"))

    summary = {
        "matrix_root": matrix_root,
        "overall_accuracy": overall,
        "dataset_accuracy": ds_map,
        "win_tie_loss": {
            "search_selective_vs_vanilla": wtl_sel,
            "search_non_selective_vs_vanilla": wtl_non,
        },
        "seegull_stereotype_rate": stereo,
        "note": "SeeGULL stereotype rate is only meaningful when option A explicitly denotes stereotype.",
    }

    summary_path = os.path.join(out_dir, "analysis_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps({"out_dir": out_dir, "summary_path": summary_path, **summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze strict matrix outputs.")
    parser.add_argument("--matrix-root", required=True)
    parser.add_argument("--out-dir", default="outputs/analysis")
    args = parser.parse_args()
    run(args.matrix_root, args.out_dir)
