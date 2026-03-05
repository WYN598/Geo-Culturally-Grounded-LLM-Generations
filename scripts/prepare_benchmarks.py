import argparse
import csv
import json
import random
import re
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests

try:
    from datasets import load_dataset
except Exception:
    load_dataset = None


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def normalize_text(x: Any) -> str:
    return "" if x is None else str(x).strip()


def letter_choices(values: List[str]) -> List[str]:
    return [f"{chr(ord('A') + i)}) {v.strip()}" for i, v in enumerate(values)]


def balanced_sample(rows: List[Dict[str, Any]], limit: int, label_key: str = "answer", seed: int = 42) -> List[Dict[str, Any]]:
    if limit <= 0 or len(rows) <= limit:
        return rows

    groups: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        k = str(r.get(label_key, "unknown"))
        groups.setdefault(k, []).append(r)

    rng = random.Random(seed)
    keys = sorted(groups.keys())
    for k in keys:
        rng.shuffle(groups[k])

    selected: List[Dict[str, Any]] = []
    per = max(1, limit // max(1, len(keys)))

    for k in keys:
        selected.extend(groups[k][:per])

    if len(selected) < limit:
        leftovers: List[Dict[str, Any]] = []
        for k in keys:
            leftovers.extend(groups[k][per:])
        rng.shuffle(leftovers)
        selected.extend(leftovers[: max(0, limit - len(selected))])

    return selected[:limit]


def rebalance_by_dataset(
    rows: List[Dict[str, Any]],
    max_per_dataset: int,
    seed: int = 42,
    label_key: str = "answer",
) -> List[Dict[str, Any]]:
    if max_per_dataset <= 0:
        return rows

    by_dataset: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        ds = str(r.get("dataset", "unknown"))
        by_dataset.setdefault(ds, []).append(r)

    out: List[Dict[str, Any]] = []
    for ds in sorted(by_dataset.keys()):
        sampled = balanced_sample(by_dataset[ds], max_per_dataset, label_key=label_key, seed=seed)
        out.extend(sampled)
    return out


def _load_hf(dataset_id: str, config: Optional[str] = None):
    if load_dataset is None:
        raise RuntimeError("datasets not installed. Run: pip install datasets")
    if config:
        return load_dataset(dataset_id, config)
    return load_dataset(dataset_id)


def _parse_blend_choices(raw: Any) -> List[str]:
    if isinstance(raw, dict):
        items = sorted(raw.items(), key=lambda kv: kv[0])
        return [normalize_text(v) for _, v in items]
    s = normalize_text(raw)
    if not s:
        return []
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            items = sorted(obj.items(), key=lambda kv: kv[0])
            return [normalize_text(v) for _, v in items]
    except Exception:
        pass

    out: List[str] = []
    for line in s.splitlines():
        m = re.match(r"^[A-Z][\.)]\s*(.+)$", line.strip())
        if m:
            out.append(m.group(1).strip())
    return out


def _clean_blend_question(prompt: str) -> str:
    s = normalize_text(prompt)
    if not s:
        return s
    s = s.split("Without any explanation", 1)[0].strip()
    s = s.split("Provide as JSON format", 1)[0].strip()
    s = re.sub(r"\s+", " ", s)
    return s


def convert_blend(limit: int, seed: int) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    ds = _load_hf("nayeon212/BLEnD", "multiple-choice-questions")
    split = "test" if "test" in ds else next(iter(ds.keys()))
    raw_len = len(ds[split])

    if limit > 0:
        n = min(limit * 20, raw_len)
        rng = random.Random(seed)
        idxs = rng.sample(range(raw_len), n)
    else:
        idxs = range(raw_len)

    out: List[Dict[str, Any]] = []
    skipped = 0
    for idx in idxs:
        r = ds[split][idx]
        q = _clean_blend_question(r.get("prompt", ""))
        raw_choices = _parse_blend_choices(r.get("choices"))
        if not q or len(raw_choices) < 2:
            skipped += 1
            continue
        choices = letter_choices(raw_choices)
        ans = normalize_text(r.get("answer_idx", "")).upper()
        if ans not in [chr(ord("A") + i) for i in range(len(choices))]:
            skipped += 1
            continue
        out.append(
            {
                "id": f"BLEnD_{idx}",
                "dataset": "BLEnD",
                "task_type": "mcq",
                "question": q,
                "choices": choices,
                "answer": ans,
            }
        )

    out = balanced_sample(out, limit, label_key="answer", seed=seed)
    report = {"raw": raw_len, "kept": len(out), "skipped": skipped, "split": split}
    return out, report


def convert_normad(limit: int, seed: int) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    ds = _load_hf("akhilayerukola/NormAd")
    split = "train" if "train" in ds else next(iter(ds.keys()))

    out: List[Dict[str, Any]] = []
    skipped = 0
    for idx, r in enumerate(ds[split]):
        story = normalize_text(r.get("Story", ""))
        label = normalize_text(r.get("Gold Label", "")).lower()
        if not story or label not in {"yes", "no"}:
            skipped += 1
            continue
        out.append(
            {
                "id": f"NormAd_{idx}",
                "dataset": "NormAd",
                "task_type": "mcq",
                "question": story,
                "choices": ["A) Yes", "B) No"],
                "answer": "A" if label == "yes" else "B",
            }
        )

    out = balanced_sample(out, limit, label_key="answer", seed=seed)
    report = {"raw": len(ds[split]), "kept": len(out), "skipped": skipped, "split": split}
    return out, report


def download_and_extract_zip(url: str, dest_dir: Path) -> Path:
    ensure_dir(dest_dir)
    local_root = dest_dir / "SeeGULL-Multilingual-main"
    if (local_root / "dataset" / "stereotypes").exists():
        return local_root

    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    zf = zipfile.ZipFile(BytesIO(resp.content))
    zf.extractall(dest_dir)
    dirs = [p for p in dest_dir.iterdir() if p.is_dir()]
    return dirs[0] if dirs else dest_dir


def convert_seegull(raw_root: Path, limit: int, seed: int) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    extracted = download_and_extract_zip(
        "https://github.com/google-research-datasets/SeeGULL-Multilingual/archive/refs/heads/main.zip",
        raw_root / "seegull",
    )

    stereo_files = sorted((extracted / "dataset" / "stereotypes").glob("*.csv"))
    out: List[Dict[str, Any]] = []
    raw = 0
    skipped = 0

    for fp in stereo_files:
        with fp.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for r in reader:
                raw += 1
                identity = normalize_text(r.get("translated_identity") or r.get("identity"))
                attr = normalize_text(r.get("translated_attribute_list") or r.get("attribute"))
                if not identity or not attr:
                    skipped += 1
                    continue

                s_cnt = float(normalize_text(r.get("stereotype", 0)) or 0)
                n_cnt = float(normalize_text(r.get("non_stereotype", 0)) or 0)
                u_cnt = float(normalize_text(r.get("unsure", 0)) or 0)
                counts = [s_cnt, n_cnt, u_cnt]
                ans_idx = int(max(range(3), key=lambda i: counts[i]))

                out.append(
                    {
                        "id": f"SeeGULL_{len(out)}",
                        "dataset": "SeeGULL",
                        "task_type": "mcq",
                        "task_variant": "seegull_dominant_label_proxy",
                        "question": f"For the claim '{identity} -> {attr}', what is the dominant annotation label?",
                        "choices": ["A) Stereotype", "B) Non-stereotype", "C) Unsure"],
                        "answer": chr(ord("A") + ans_idx),
                    }
                )

    out = balanced_sample(out, limit, label_key="answer", seed=seed)
    report = {
        "raw": raw,
        "kept": len(out),
        "skipped": skipped,
        "files": len(stereo_files),
        "source_dir": str(extracted),
        "note": "SeeGULL conversion is a dominant-label proxy task, not the full original paper protocol.",
    }
    return out, report


def dedup(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for r in rows:
        key = (r["dataset"], r["question"].strip().lower(), tuple(r["choices"]), r["answer"])
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def run(args: argparse.Namespace) -> None:
    raw_root = Path(args.raw_root)
    out_path = Path(args.out)
    ensure_dir(raw_root)

    all_rows: List[Dict[str, Any]] = []
    report: Dict[str, Any] = {}

    if "blend" in args.datasets:
        try:
            rows, rep = convert_blend(args.max_per_dataset, args.seed)
            all_rows.extend(rows)
            rep["status"] = "ok"
            report["blend"] = rep
        except Exception as e:
            report["blend"] = {"status": f"failed: {e}"}

    if "normad" in args.datasets:
        try:
            rows, rep = convert_normad(args.max_per_dataset, args.seed)
            all_rows.extend(rows)
            rep["status"] = "ok"
            report["normad"] = rep
        except Exception as e:
            report["normad"] = {"status": f"failed: {e}"}

    if "seegull" in args.datasets:
        try:
            rows, rep = convert_seegull(raw_root, args.max_per_dataset, args.seed)
            all_rows.extend(rows)
            rep["status"] = "ok"
            report["seegull"] = rep
        except Exception as e:
            report["seegull"] = {"status": f"failed: {e}"}

    final_rows = dedup(all_rows)
    final_rows = rebalance_by_dataset(final_rows, args.max_per_dataset, seed=args.seed, label_key="answer")
    write_jsonl(out_path, final_rows)

    summary = {
        "output": str(out_path),
        "total_kept_after_dedup": len(final_rows),
        "datasets": report,
        "note": "Sampling is label-balanced by answer when max_per_dataset > 0.",
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and convert BLEnD/NormAd/SeeGULL into project MCQ jsonl.")
    parser.add_argument("--datasets", nargs="+", default=["blend", "normad", "seegull"], choices=["blend", "normad", "seegull"])
    parser.add_argument("--raw-root", default="data/benchmarks/raw")
    parser.add_argument("--out", default="data/benchmark_eval.jsonl")
    parser.add_argument("--max-per-dataset", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run(args)
