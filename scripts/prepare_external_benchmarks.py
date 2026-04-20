import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import csv

try:
    from datasets import get_dataset_config_names, load_dataset
except Exception:
    get_dataset_config_names = None
    load_dataset = None


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_text(x: Any) -> str:
    return "" if x is None else str(x).strip()


def letter_choices(values: List[str]) -> List[str]:
    return [f"{chr(ord('A') + i)}) {normalize_text(v)}" for i, v in enumerate(values)]


def _hf_load(dataset_id: str, config: Optional[str] = None):
    if load_dataset is None:
        raise RuntimeError("datasets is not available. Install with: pip install datasets")
    return load_dataset(dataset_id, config) if config else load_dataset(dataset_id)


def _parse_jsonish_list(text: Any) -> List[str]:
    raw = normalize_text(text)
    if not raw:
        return []
    try:
        obj = json.loads(raw)
        if isinstance(obj, list):
            return [normalize_text(x) for x in obj if normalize_text(x)]
    except Exception:
        pass
    return [x.strip() for x in raw.split(";") if x.strip()]


def _pick_first(row: Dict[str, Any], candidates: List[str]) -> str:
    lower_map = {str(k).strip().lower(): v for k, v in row.items()}
    for name in candidates:
        value = lower_map.get(name.lower())
        text = normalize_text(value)
        if text:
            return text
    return ""


def _read_rows(path: Path) -> List[Dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        rows: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, list):
            return [x for x in obj if isinstance(x, dict)]
        raise ValueError(f"Unsupported JSON structure in {path}")
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            return list(csv.DictReader(f))
    raise ValueError(f"Unsupported EspanStereo source format: {path.suffix}")


def _bool_to_answer(value: Any) -> Optional[str]:
    if isinstance(value, bool):
        return "A" if value else "B"
    raw = normalize_text(value).lower()
    if raw in {"true", "1", "yes", "y"}:
        return "A"
    if raw in {"false", "0", "no", "n"}:
        return "B"
    return None


def convert_bbq(seed: int = 42) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if get_dataset_config_names is None:
        raise RuntimeError("datasets is not available. Install with: pip install datasets")
    configs = [c for c in get_dataset_config_names("HiTZ/bbq") if c.endswith("_ambig") or c.endswith("_disambig")]
    rows: List[Dict[str, Any]] = []
    total_raw = 0
    kept = 0
    for config in configs:
        ds = _hf_load("HiTZ/bbq", config)
        split = "test" if "test" in ds else next(iter(ds.keys()))
        for ex in ds[split]:
            total_raw += 1
            context = normalize_text(ex.get("context"))
            question = normalize_text(ex.get("question"))
            answers = [normalize_text(ex.get("ans0")), normalize_text(ex.get("ans1")), normalize_text(ex.get("ans2"))]
            if not context or not question or any(not a for a in answers):
                continue
            label = ex.get("label")
            try:
                label_idx = int(label)
            except Exception:
                continue
            if label_idx < 0 or label_idx >= len(answers):
                continue
            full_question = f"{context}\n\n{question}"
            rows.append(
                {
                    "id": f"BBQ_{config}_{ex.get('example_id')}",
                    "dataset": "BBQ",
                    "task_type": "mcq",
                    "benchmark_family": "bias",
                    "question": full_question,
                    "choices": letter_choices(answers),
                    "answer": chr(ord("A") + label_idx),
                    "metadata": {
                        "config": config,
                        "category": normalize_text(ex.get("category")),
                        "question_polarity": normalize_text(ex.get("question_polarity")),
                        "context_condition": normalize_text(ex.get("context_condition")),
                        "stereotyped_groups": ex.get("additional_metadata", {}).get("stereotyped_groups", []),
                        "source": ex.get("additional_metadata", {}).get("source"),
                    },
                }
            )
            kept += 1
    report = {"configs": len(configs), "raw": total_raw, "kept": kept, "seed": seed}
    return rows, report


def convert_socialstigmaqa(seed: int = 42) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    ds = _hf_load("ibm-research/SocialStigmaQA", "yes_no")
    split = "test" if "test" in ds else next(iter(ds.keys()))
    rows: List[Dict[str, Any]] = []
    mapping = {"yes": "A", "no": "B", "can't tell": "C", "cant tell": "C", "cannot tell": "C"}
    for idx, ex in enumerate(ds[split]):
        prompt = normalize_text(ex.get("prompt"))
        biased = normalize_text(ex.get("biased answer")).lower()
        if not prompt or biased not in mapping:
            continue
        rows.append(
            {
                "id": f"SocialStigmaQA_{idx}",
                "dataset": "SocialStigmaQA",
                "task_type": "bias_probe_mcq",
                "benchmark_family": "bias",
                "question": prompt,
                "choices": ["A) Yes", "B) No", "C) Can't tell"],
                "biased_answer": mapping[biased],
                "allowed_non_biased_answers": [x for x in ["A", "B", "C"] if x != mapping[biased]],
                "metadata": {
                    "stigma": ex.get("stigma"),
                    "prompt_style": ex.get("prompt style"),
                },
            }
        )
    report = {
        "config": "yes_no",
        "raw": len(ds[split]),
        "kept": len(rows),
        "seed": seed,
        "metric_note": "No single gold answer; evaluate bias rate against biased_answer.",
    }
    return rows, report


def convert_truthfulqa(seed: int = 42) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    url = "https://raw.githubusercontent.com/sylinrl/TruthfulQA/main/TruthfulQA.csv"
    df = pd.read_csv(url)
    rng = random.Random(seed)
    rows: List[Dict[str, Any]] = []
    for idx, ex in df.iterrows():
        q = normalize_text(ex.get("Question"))
        correct = normalize_text(ex.get("Best Answer"))
        incorrect = normalize_text(ex.get("Best Incorrect Answer"))
        if not q or not correct or not incorrect:
            continue
        pair = [correct, incorrect]
        rng.shuffle(pair)
        answer = "A" if pair[0] == correct else "B"
        rows.append(
            {
                "id": f"TruthfulQA_{idx}",
                "dataset": "TruthfulQA-Binary",
                "task_type": "mcq",
                "benchmark_family": "factuality",
                "question": q,
                "choices": letter_choices(pair),
                "answer": answer,
                "metadata": {
                    "eval_protocol": "binary_proxy",
                    "type": normalize_text(ex.get("Type")),
                    "category": normalize_text(ex.get("Category")),
                    "best_answer": correct,
                    "best_incorrect_answer": incorrect,
                    "source": normalize_text(ex.get("Source")),
                },
            }
        )
    report = {
        "source": url,
        "raw": int(len(df)),
        "kept": len(rows),
        "seed": seed,
        "conversion_note": "Converted to a binary TruthfulQA proxy using Best Answer vs Best Incorrect Answer. This is not the full official TruthfulQA evaluation protocol.",
    }
    return rows, report


def convert_popqa(seed: int = 42) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    ds = _hf_load("akariasai/PopQA")
    split = "test" if "test" in ds else next(iter(ds.keys()))
    rows: List[Dict[str, Any]] = []
    for ex in ds[split]:
        q = normalize_text(ex.get("question"))
        answers = _parse_jsonish_list(ex.get("possible_answers"))
        if not q or not answers:
            continue
        rows.append(
            {
                "id": f"PopQA_{ex.get('id')}",
                "dataset": "PopQA",
                "task_type": "short_qa",
                "benchmark_family": "factuality",
                "question": q,
                "answers": answers,
                "metadata": {
                    "subject": normalize_text(ex.get("subj")),
                    "property": normalize_text(ex.get("prop")),
                    "object": normalize_text(ex.get("obj")),
                    "subject_popularity": ex.get("s_pop"),
                    "object_popularity": ex.get("o_pop"),
                },
            }
        )
    report = {
        "raw": len(ds[split]),
        "kept": len(rows),
        "seed": seed,
        "metric_note": "Open-domain short-answer QA; evaluate exact match or answer containment against answers.",
    }
    return rows, report


def convert_culturalbench(subset: str, seed: int = 42) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    subset_norm = subset.strip().lower()
    subset_to_name = {
        "easy": "CulturalBench-Easy",
        "hard": "CulturalBench-Hard",
    }
    if subset_norm not in subset_to_name:
        raise ValueError(f"Unsupported CulturalBench subset: {subset}")
    config_name = subset_to_name[subset_norm]
    rows: List[Dict[str, Any]] = []
    raw = 0

    ds = None
    dataset_error: Optional[str] = None
    try:
        ds = _hf_load("kellycyy/CulturalBench", config_name)
    except Exception as e:
        dataset_error = str(e)

    if ds is None:
        file_url = f"https://huggingface.co/datasets/kellycyy/CulturalBench/resolve/main/{config_name}.csv"
        try:
            ds = _hf_load("csv", data_files={"test": file_url})
        except Exception as e:
            raise RuntimeError(
                f"Failed to load CulturalBench subset '{config_name}' via datasets and CSV fallback. "
                f"datasets error: {dataset_error}; csv error: {e}"
            )

    split = "test" if "test" in ds else next(iter(ds.keys()))
    for idx, ex in enumerate(ds[split]):
        raw += 1
        prompt_question = normalize_text(ex.get("prompt_question"))
        if not prompt_question:
            continue

        # Hard subset: boolean option verification
        if "prompt_option" in ex:
            prompt_option = normalize_text(ex.get("prompt_option"))
            answer = _bool_to_answer(ex.get("answer"))
            if not prompt_option or answer is None:
                continue
            rows.append(
                {
                    "id": f"{config_name}_{idx}",
                    "dataset": config_name,
                    "task_type": "mcq",
                    "benchmark_family": "cultural_knowledge",
                    "question": f"{prompt_question}\n\nCandidate option: {prompt_option}\n\nIs this option correct?",
                    "choices": ["A) Yes", "B) No"],
                    "answer": answer,
                    "metadata": {
                        "subset": config_name,
                        "country": normalize_text(ex.get("country")),
                        "question_idx": ex.get("question_idx"),
                        "data_idx": ex.get("data_idx"),
                        "prompt_question": prompt_question,
                        "prompt_option": prompt_option,
                        "eval_protocol": "boolean_option_proxy",
                    },
                }
            )
            continue

        # Easy subset: standard 4-way MCQ
        option_values = [
            normalize_text(ex.get("prompt_option_a")),
            normalize_text(ex.get("prompt_option_b")),
            normalize_text(ex.get("prompt_option_c")),
            normalize_text(ex.get("prompt_option_d")),
        ]
        if any(not x for x in option_values):
            continue
        raw_answer = normalize_text(ex.get("answer")).upper()
        answer = ""
        if raw_answer in {"A", "B", "C", "D"}:
            answer = raw_answer
        elif raw_answer in {"0", "1", "2", "3"}:
            answer = chr(ord("A") + int(raw_answer))
        else:
            for pos, option_text in enumerate(option_values):
                if raw_answer == option_text.upper():
                    answer = chr(ord("A") + pos)
                    break
        if answer not in {"A", "B", "C", "D"}:
            continue

        rows.append(
            {
                "id": f"{config_name}_{idx}",
                "dataset": config_name,
                "task_type": "mcq",
                "benchmark_family": "cultural_knowledge",
                "question": prompt_question,
                "choices": letter_choices(option_values),
                "answer": answer,
                "metadata": {
                    "subset": config_name,
                    "country": normalize_text(ex.get("country")),
                    "question_idx": ex.get("question_idx"),
                    "data_idx": ex.get("data_idx"),
                    "eval_protocol": "direct_mcq",
                },
            }
        )
    report = {
        "subset": config_name,
        "raw": raw,
        "kept": len(rows),
        "seed": seed,
        "metric_note": (
            "Easy subset is kept as direct MCQ; Hard subset is converted to a 2-way MCQ proxy "
            "where the model judges whether a candidate option is correct."
        ),
    }
    return rows, report


def convert_espanstereo(seed: int = 42, source_path: str = "") -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    candidates: List[Path] = []
    if source_path.strip():
        src = Path(source_path)
        if src.is_dir():
            candidates.extend(sorted(src.glob("*.jsonl")))
            candidates.extend(sorted(src.glob("*.json")))
            candidates.extend(sorted(src.glob("*.csv")))
        elif src.exists():
            candidates.append(src)
    else:
        raw_dir = Path("data/benchmarks/external/raw/espanstereo")
        if raw_dir.exists():
            candidates.extend(sorted(raw_dir.glob("*.jsonl")))
            candidates.extend(sorted(raw_dir.glob("*.json")))
            candidates.extend(sorted(raw_dir.glob("*.csv")))

    if not candidates:
        raise FileNotFoundError(
            "No EspanStereo source file found. Place a CSV/JSON/JSONL file under "
            "'data/benchmarks/external/raw/espanstereo/' or pass --espanstereo-path."
        )

    source = candidates[0]
    raw_rows = _read_rows(source)
    rng = random.Random(seed)
    rows: List[Dict[str, Any]] = []
    skipped = 0

    for idx, ex in enumerate(raw_rows):
        context = _pick_first(ex, ["context", "sentence", "prompt", "premise", "texto_contexto"])
        stereotypical = _pick_first(
            ex,
            [
                "stereotypical",
                "stereotype",
                "stereotypical_text",
                "stereotypical_sentence",
                "estereotipico",
            ],
        )
        anti = _pick_first(
            ex,
            [
                "anti_stereotypical",
                "anti-stereotypical",
                "anti_stereotype",
                "anti_stereotypical_text",
                "anti",
                "antiestereotipico",
            ],
        )
        irrelevant = _pick_first(
            ex,
            ["irrelevant", "irrelevant_text", "neutral", "unrelated", "irrelevante"],
        )
        if not context or not stereotypical or not anti or not irrelevant:
            skipped += 1
            continue

        options = [
            {"role": "stereotypical", "text": stereotypical},
            {"role": "anti_stereotypical", "text": anti},
            {"role": "irrelevant", "text": irrelevant},
        ]
        rng.shuffle(options)
        answer_idx = next(i for i, opt in enumerate(options) if opt["role"] == "stereotypical")

        rows.append(
            {
                "id": f"EspanStereo_{idx}",
                "dataset": "EspanStereo",
                "task_type": "mcq",
                "benchmark_family": "bias",
                "task_variant": "espanstereo_triplet_mcq_proxy",
                "question": f"{context}\n\nWhich continuation is the most stereotypical?",
                "choices": letter_choices([opt["text"] for opt in options]),
                "answer": chr(ord("A") + answer_idx),
                "metadata": {
                    "country": _pick_first(ex, ["country", "region", "locale", "target_country"]),
                    "category": _pick_first(ex, ["category", "axis", "dimension", "target_group_type"]),
                    "source_file": str(source),
                    "eval_protocol": "triplet_mcq_proxy",
                    "note": "Proxy conversion from context + stereotypical/anti-stereotypical/irrelevant triplets.",
                },
            }
        )

    report = {
        "source": str(source),
        "raw": len(raw_rows),
        "kept": len(rows),
        "skipped": skipped,
        "seed": seed,
        "conversion_note": "Converted EspanStereo triplets into a 3-way MCQ proxy by asking for the most stereotypical continuation.",
    }
    return rows, report


def run(args: argparse.Namespace) -> None:
    out_root = Path(args.out_root)
    processed_dir = out_root / "processed"
    reports_dir = out_root / "reports"
    ensure_dir(processed_dir)
    ensure_dir(reports_dir)

    summary: Dict[str, Any] = {"out_root": str(out_root), "datasets": {}}
    converters = {
        "bbq": convert_bbq,
        "socialstigmaqa": convert_socialstigmaqa,
        "truthfulqa": convert_truthfulqa,
        "popqa": convert_popqa,
        "culturalbench_easy": lambda seed=42: convert_culturalbench("easy", seed=seed),
        "culturalbench_hard": lambda seed=42: convert_culturalbench("hard", seed=seed),
        "espanstereo": lambda seed=42: convert_espanstereo(seed=seed, source_path=args.espanstereo_path),
    }

    for name in args.datasets:
        try:
            rows, report = converters[name](seed=args.seed)
            out_path = processed_dir / f"{name}.jsonl"
            write_jsonl(out_path, rows)
            report["status"] = "ok"
            report["output"] = str(out_path)
            summary["datasets"][name] = report
        except Exception as e:
            summary["datasets"][name] = {
                "status": f"failed: {e}",
            }

    summary_path = reports_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and convert external bias/factuality benchmarks.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["bbq", "socialstigmaqa", "truthfulqa", "popqa"],
        choices=[
            "bbq",
            "socialstigmaqa",
            "truthfulqa",
            "popqa",
            "culturalbench_easy",
            "culturalbench_hard",
            "espanstereo",
        ],
    )
    parser.add_argument("--out-root", default="data/benchmarks/external")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--espanstereo-path",
        default="",
        help="Optional local CSV/JSON/JSONL path or directory for EspanStereo source data.",
    )
    args = parser.parse_args()
    run(args)
