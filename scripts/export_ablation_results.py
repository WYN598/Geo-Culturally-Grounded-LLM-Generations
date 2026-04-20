import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List


SYSTEM_ORDER = [
    "vanilla",
    "simple_rag",
    "planning_rag",
    "planning_semantic_rag",
    "planning_semantic_noise_filter",
]

SYSTEM_LABELS = {
    "vanilla": "Vanilla LLM",
    "simple_rag": "Simple RAG",
    "planning_rag": "Rewrite",
    "planning_semantic_rag": "Rerank",
    "planning_semantic_noise_filter": "Noise Filter",
}


def load_summary(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def benchmark_label_from_tag(tag: str) -> str:
    mapping = {
        "blend_100": "BLEnD_100",
        "normad_100": "NormAd_100",
        "seegull_100": "SeeGULL_100",
        "culturalbench_easy_100": "CulturalBench-Easy_100",
        "culturalbench_hard_100": "CulturalBench-Hard_100",
    }
    return mapping.get(tag, tag)


def summarize_run_root(root: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for summary_path in sorted(root.glob("*/analysis/ablation_summary.json")):
        tag = summary_path.parent.parent.name
        summary = load_summary(summary_path)
        model = str((summary.get("model", {}) or {}).get("model", ""))
        run_by_name = {str(r.get("name", "")): r for r in summary.get("runs", [])}
        row = {
            "Benchmark": benchmark_label_from_tag(tag),
            "Model": model,
        }
        for system in SYSTEM_ORDER:
            run = run_by_name.get(system, {})
            val = run.get("overall_accuracy", "")
            row[SYSTEM_LABELS[system]] = f"{float(val):.2f}" if val != "" else ""
        rows.append(row)
    return rows


def write_csv(path: Path, rows: List[Dict[str, str]]) -> None:
    if not rows:
        return
    fieldnames = ["Benchmark", "Model"] + [SYSTEM_LABELS[s] for s in SYSTEM_ORDER]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: List[Dict[str, str]]) -> None:
    if not rows:
        return
    headers = ["Benchmark", "Model"] + [SYSTEM_LABELS[s] for s in SYSTEM_ORDER]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row.get(h, "") for h in headers) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export 5-way ablation summaries into CSV/Markdown tables.")
    parser.add_argument("--root", required=True, help="Root directory containing <tag>/analysis/ablation_summary.json")
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--out-md", required=True)
    args = parser.parse_args()

    rows = summarize_run_root(Path(args.root))
    write_csv(Path(args.out_csv), rows)
    write_markdown(Path(args.out_md), rows)
    print(json.dumps({"rows": len(rows), "out_csv": args.out_csv, "out_md": args.out_md}, ensure_ascii=True))


if __name__ == "__main__":
    main()
