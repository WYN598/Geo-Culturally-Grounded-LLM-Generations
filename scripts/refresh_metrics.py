import argparse
import json
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval import evaluate_rows, mcq_accuracy
from src.main import summarize_search_trace
from src.pipeline import jsonl_integrity_summary, load_jsonl


def refresh_run_metrics(run_dir: Path) -> Dict[str, Any]:
    metrics_path = run_dir / "metrics.json"
    metrics: Dict[str, Any] = {}
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8-sig") as f:
            metrics = json.load(f)

    vanilla_pred = run_dir / "vanilla_predictions.jsonl"
    if vanilla_pred.exists():
        integrity = jsonl_integrity_summary(str(vanilla_pred))
        metrics["vanilla_prediction_integrity"] = integrity
        rows = load_jsonl(str(vanilla_pred), strict=bool(integrity.get("bad_lines", 0) == 0))
        if rows and all("answer" in row for row in rows):
            metrics["vanilla_acc"] = mcq_accuracy(rows)
        metrics["vanilla_eval"] = evaluate_rows(rows)

    search_pred = run_dir / "search_predictions.jsonl"
    if search_pred.exists():
        integrity = jsonl_integrity_summary(str(search_pred))
        metrics["search_prediction_integrity"] = integrity
        rows = load_jsonl(str(search_pred), strict=bool(integrity.get("bad_lines", 0) == 0))
        if rows and all("answer" in row for row in rows):
            metrics["search_acc"] = mcq_accuracy(rows)
        metrics["search_eval"] = evaluate_rows(rows)
        metrics["search_trace_summary"] = summarize_search_trace(rows)
        metrics["search_runtime_status"] = ((rows[0].get("search_trace", {}) or {}).get("runtime_status", {})) if rows else {}

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh metrics.json from existing prediction files.")
    parser.add_argument("--run-dir", required=True, help="Run directory containing metrics.json and prediction files.")
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    metrics = refresh_run_metrics(run_dir)
    print(json.dumps({"run_dir": str(run_dir), "metrics_keys": sorted(metrics.keys())}, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
