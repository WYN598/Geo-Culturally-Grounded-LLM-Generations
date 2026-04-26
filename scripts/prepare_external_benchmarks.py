import argparse
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import csv

try:
    from datasets import Dataset, get_dataset_config_names, load_dataset
except Exception:
    Dataset = None
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


def _hf_cache_root() -> Path:
    return Path.home() / ".cache" / "huggingface" / "datasets"


def _load_cached_arrow_rows(pattern: str) -> List[Dict[str, Any]]:
    if Dataset is None:
        raise RuntimeError("datasets is not available. Install with: pip install datasets")
    rows: List[Dict[str, Any]] = []
    for path in sorted(_hf_cache_root().glob(pattern)):
        ds = Dataset.from_file(str(path))
        rows.extend(ds.to_list())
    if not rows:
        raise FileNotFoundError(f"No cached arrow files matched: {pattern}")
    return rows


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


HONEST_DEFAULT_SUBSETS = [
    "en_binary",
    "es_binary",
    "fr_binary",
    "it_binary",
    "pt_binary",
    "ro_binary",
]
HONEST_ALL_SUBSETS = HONEST_DEFAULT_SUBSETS + ["en_queer_nonqueer"]
MSQAD_LANGUAGE_FIELDS = {
    "en": "English",
    "ko": "Korean",
    "zh": "Chinese",
    "es": "Spanish",
    "de": "German",
    "hi": "Hindi",
}
HONEST_REMOTE_FILES = {
    "en_binary": "https://raw.githubusercontent.com/MilaNLProc/honest/main/resources/binary/en_template.tsv",
    "es_binary": "https://raw.githubusercontent.com/MilaNLProc/honest/main/resources/binary/es_template.tsv",
    "fr_binary": "https://raw.githubusercontent.com/MilaNLProc/honest/main/resources/binary/fr_template.tsv",
    "it_binary": "https://raw.githubusercontent.com/MilaNLProc/honest/main/resources/binary/it_template.tsv",
    "pt_binary": "https://raw.githubusercontent.com/MilaNLProc/honest/main/resources/binary/pt_template.tsv",
    "ro_binary": "https://raw.githubusercontent.com/MilaNLProc/honest/main/resources/binary/ro_template.tsv",
    "en_queer_nonqueer": "https://raw.githubusercontent.com/MilaNLProc/honest/main/resources/queer_nonqueer/en_template.tsv",
}


def _clean_honest_query(text: str) -> str:
    out = normalize_text(text).replace("[M]", " ")
    out = re.sub(r"\s+", " ", out)
    out = re.sub(r"\s+([,.;!?])", r"\1", out)
    return out.strip()


def _slugify(text: Any) -> str:
    value = normalize_text(text).lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_") or "unknown"


def _load_honest_subset(subset: str, source_root: str = "") -> pd.DataFrame:
    filename = f"{subset}.tsv"
    candidates: List[Path] = []
    if source_root.strip():
        root = Path(source_root)
        if root.is_dir():
            candidates.append(root / filename)
            candidates.append(root / f"{subset}.csv")
        elif root.exists():
            candidates.append(root)
    else:
        raw_root = Path("data/benchmarks/external/raw/honest")
        candidates.append(raw_root / filename)
        candidates.append(raw_root / f"{subset}.csv")

    for path in candidates:
        if path.exists():
            sep = "\t" if path.suffix.lower() == ".tsv" else ","
            return pd.read_csv(path, sep=sep)

    url = HONEST_REMOTE_FILES.get(subset)
    if not url:
        raise ValueError(f"Unsupported HONEST subset: {subset}")
    try:
        return pd.read_csv(url, sep="\t")
    except Exception as e:
        raise RuntimeError(
            f"Failed to load HONEST subset '{subset}' from local raw files or remote URL. "
            f"Place '{filename}' under data/benchmarks/external/raw/honest/ or pass --honest-source-root."
        ) from e


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


def convert_honest(
    seed: int = 42,
    subsets: Optional[List[str]] = None,
    source_root: str = "",
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    subset_list = [str(x).strip() for x in (subsets or HONEST_DEFAULT_SUBSETS) if str(x).strip()]
    if not subset_list:
        subset_list = list(HONEST_DEFAULT_SUBSETS)
    if len(subset_list) == 1 and subset_list[0].lower() == "all":
        subset_list = list(HONEST_ALL_SUBSETS)

    rows: List[Dict[str, Any]] = []
    raw = 0
    for subset in subset_list:
        df = _load_honest_subset(subset, source_root=source_root)
        lang = subset.split("_", 1)[0].lower()
        identity_set = "queer_nonqueer" if "queer" in subset.lower() else "binary"
        for idx, ex in enumerate(df.to_dict("records")):
            raw += 1
            template_masked = normalize_text(ex.get("template_masked"))
            if not template_masked or "[M]" not in template_masked:
                continue
            retrieval_query = _clean_honest_query(template_masked)
            if not retrieval_query:
                continue
            category = normalize_text(ex.get("category")).lower() or "unknown"
            template_type = normalize_text(ex.get("type")).lower() or "unknown"
            raw_template = normalize_text(ex.get("raw"))
            rows.append(
                {
                    "id": f"HONEST_{subset}_{idx}",
                    "dataset": "HONEST",
                    "task_type": "honest_completion",
                    "benchmark_family": "bias",
                    "question": template_masked,
                    "retrieval_query": retrieval_query,
                    "sampling_bucket": f"{lang}:{category}",
                    "metadata": {
                        "subset": subset,
                        "language": lang,
                        "identity_set": identity_set,
                        "category": category,
                        "identity": normalize_text(ex.get("identity")),
                        "number": normalize_text(ex.get("number")).lower() or "unknown",
                        "template_type": template_type,
                        "template_masked": template_masked,
                        "raw_template": raw_template,
                        "eval_protocol": "official_honest_score",
                    },
                }
            )

    report = {
        "subsets": subset_list,
        "raw": raw,
        "kept": len(rows),
        "seed": seed,
        "metric_note": "Generative sentence-completion benchmark scored with the official HONEST evaluator. Lower HONEST score is better.",
        "source_mode": "local_or_remote_tsv",
    }
    return rows, report


def convert_cbbq(seed: int = 42) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    cache_pattern = "walledai___cbbq/default/*/*/*.arrow"
    if list(_hf_cache_root().glob(cache_pattern)):
        split_rows = _load_cached_arrow_rows("walledai___cbbq/default/*/*/*.arrow")
    else:
        ds = _hf_load("walledai/CBBQ")
        split_rows = list(ds["train" if "train" in ds else next(iter(ds.keys()))])
    rows: List[Dict[str, Any]] = []
    raw = 0
    for idx, ex in enumerate(split_rows):
        raw += 1
        context = normalize_text(ex.get("context"))
        question = normalize_text(ex.get("question"))
        choices = [normalize_text(x) for x in (ex.get("choices") or []) if normalize_text(x)]
        if not question or len(choices) < 2:
            continue
        full_question = question if not context else f"{context}\n\n{question}"
        try:
            answer_idx = int(ex.get("answer"))
        except Exception:
            continue
        if answer_idx < 0 or answer_idx >= len(choices):
            continue
        category = normalize_text(ex.get("category")).lower() or "unknown"
        rows.append(
            {
                "id": f"CBBQ_{idx}",
                "dataset": "CBBQ",
                "task_type": "mcq",
                "benchmark_family": "bias",
                "question": full_question,
                "choices": letter_choices(choices),
                "answer": chr(ord("A") + answer_idx),
                "sampling_bucket": f"zh:{category}",
                "metadata": {
                    "language": "zh",
                    "category": category,
                    "eval_protocol": "direct_mcq",
                },
            }
        )
    report = {
        "raw": raw,
        "kept": len(rows),
        "seed": seed,
        "metric_note": "Chinese BBQ-style MCQ benchmark evaluated with accuracy and slice metrics.",
    }
    return rows, report


def convert_borderlines(seed: int = 42) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    query_pattern = "manestay___borderlines/queries/*/*/*.arrow"
    territory_pattern = "manestay___borderlines/territories/*/*/*.arrow"
    if list(_hf_cache_root().glob(query_pattern)) and list(_hf_cache_root().glob(territory_pattern)):
        query_rows = _load_cached_arrow_rows("manestay___borderlines/queries/*/*/*.arrow")
        query_splits = {}
        for row in query_rows:
            query_id = normalize_text(row.get("QueryID"))
            lang = query_id.rsplit("_", 1)[-1].lower() if "_" in query_id else "unknown"
            query_splits.setdefault(lang, []).append(row)
        territory_rows = _load_cached_arrow_rows("manestay___borderlines/territories/*/*/*.arrow")
    else:
        queries_ds = _hf_load("manestay/borderlines", "queries")
        territories_ds = _hf_load("manestay/borderlines", "territories")
        query_splits = {str(name): list(split_rows) for name, split_rows in queries_ds.items()}
        territory_split = "train" if "train" in territories_ds else next(iter(territories_ds.keys()))
        territory_rows = list(territories_ds[territory_split])
    rows: List[Dict[str, Any]] = []
    raw = 0
    for split_name, split_rows in query_splits.items():
        lang = normalize_text(split_name).lower()
        for idx, ex in enumerate(split_rows):
            raw += 1
            try:
                territory_idx = int(ex.get("Index_Territory"))
            except Exception:
                continue
            if territory_idx < 0 or territory_idx >= len(territory_rows):
                continue
            territory_info = territory_rows[territory_idx]
            claimants_native = [normalize_text(x) for x in (ex.get("Claimants_Native") or []) if normalize_text(x)]
            claimants_en = [normalize_text(x) for x in (territory_info.get("Claimants") or []) if normalize_text(x)]
            question = normalize_text(ex.get("Query_Native"))
            if not question or len(claimants_native) < 2 or len(claimants_native) != len(claimants_en):
                continue
            controller = normalize_text(territory_info.get("Controller"))
            answer = ""
            if controller and controller.lower() != "unknown" and controller in claimants_en:
                answer = chr(ord("A") + claimants_en.index(controller))
            region = normalize_text(territory_info.get("Region")).lower() or "unknown"
            territory = normalize_text(territory_info.get("Territory")) or normalize_text(ex.get("QueryID"))
            rows.append(
                {
                    "id": f"BorderLines_{normalize_text(ex.get('QueryID'))}",
                    "dataset": "BorderLines",
                    "task_type": "geopolitical_mcq",
                    "benchmark_family": "bias",
                    "question": question,
                    "choices": letter_choices(claimants_native),
                    "answer": answer,
                    "sampling_bucket": f"{lang}:{region}",
                    "metadata": {
                        "language": lang,
                        "region": region,
                        "territory": territory,
                        "controller": controller,
                        "claimants_en": claimants_en,
                        "claimants_native": claimants_native,
                        "population": territory_info.get("Population"),
                        "query_id": normalize_text(ex.get("QueryID")),
                        "has_gold": bool(answer),
                        "eval_protocol": "controller_match_plus_cross_language_consistency",
                    },
                }
            )
    report = {
        "raw": raw,
        "kept": len(rows),
        "seed": seed,
        "metric_note": "Geopolitical MCQ benchmark evaluated with controller match rate and cross-language consistency.",
    }
    return rows, report


def convert_msqad(
    seed: int = 42,
    source_root: str = "",
    model_source: str = "gpt-3.5-turbo-0125",
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    base_root = Path(source_root) if source_root.strip() else Path("data/benchmarks/external/raw/msqad_repo/MSQAD")
    source_dir = base_root / model_source if base_root.is_dir() and (base_root / model_source).exists() else base_root
    if not source_dir.exists():
        raise FileNotFoundError(
            "MSQAD source directory was not found. Clone the official repo into "
            "'data/benchmarks/external/raw/msqad_repo/' or pass --msqad-source-root."
        )
    topic_files = sorted(source_dir.glob("*.json"))
    if not topic_files:
        raise FileNotFoundError(f"No MSQAD JSON files found under {source_dir}")

    rng = random.Random(seed)
    rows: List[Dict[str, Any]] = []
    raw = 0
    for topic_path in topic_files:
        topic = topic_path.stem
        topic_slug = _slugify(topic)
        with topic_path.open("r", encoding="utf-8-sig") as f:
            examples = json.load(f)
        if not isinstance(examples, list):
            continue
        for idx, ex in enumerate(examples):
            raw += 1
            for lang, prefix in MSQAD_LANGUAGE_FIELDS.items():
                question = normalize_text(ex.get(f"{prefix}_question"))
                acceptable = normalize_text(ex.get(f"{prefix}_acceptable_response"))
                non_acceptable = normalize_text(ex.get(f"{prefix}_non-acceptable_response"))
                if not question or not acceptable or not non_acceptable:
                    continue
                options = [
                    {"kind": "acceptable", "text": acceptable},
                    {"kind": "non_acceptable", "text": non_acceptable},
                ]
                rng.shuffle(options)
                answer_idx = next(i for i, opt in enumerate(options) if opt["kind"] == "acceptable")
                rows.append(
                    {
                        "id": f"MSQAD_{topic_slug}_{idx}_{lang}",
                        "dataset": "MSQAD",
                        "task_type": "ethical_pair_mcq",
                        "benchmark_family": "bias",
                        "question": question,
                        "choices": letter_choices([opt["text"] for opt in options]),
                        "answer": chr(ord("A") + answer_idx),
                        "sampling_bucket": f"{lang}:{topic_slug}",
                        "metadata": {
                            "language": lang,
                            "topic": topic,
                            "title": normalize_text(ex.get("title")),
                            "subtitle": normalize_text(ex.get("subtitle")),
                            "news_url": normalize_text(ex.get("news_url")),
                            "generated_keywords": ex.get("generated_keywords", []),
                            "eval_protocol": "acceptable_vs_non_acceptable_pair_mcq_proxy",
                            "source_model": model_source,
                        },
                    }
                )
    report = {
        "source_dir": str(source_dir),
        "topics": len(topic_files),
        "raw": raw,
        "kept": len(rows),
        "seed": seed,
        "metric_note": (
            "Converted MSQAD acceptable/non-acceptable response pairs into a 2-way MCQ proxy "
            "that asks the model to choose the ethically acceptable response."
        ),
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
        "cbbq": convert_cbbq,
        "borderlines": convert_borderlines,
        "msqad": lambda seed=42: convert_msqad(
            seed=seed,
            source_root=args.msqad_source_root,
            model_source=args.msqad_model_source,
        ),
        "culturalbench_easy": lambda seed=42: convert_culturalbench("easy", seed=seed),
        "culturalbench_hard": lambda seed=42: convert_culturalbench("hard", seed=seed),
        "espanstereo": lambda seed=42: convert_espanstereo(seed=seed, source_path=args.espanstereo_path),
        "honest": lambda seed=42: convert_honest(
            seed=seed,
            subsets=args.honest_subsets,
            source_root=args.honest_source_root,
        ),
    }

    failed_datasets: List[str] = []
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
            failed_datasets.append(name)

    summary["failed_datasets"] = failed_datasets
    summary["ok"] = len(failed_datasets) == 0

    summary_path = reports_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if failed_datasets:
        raise SystemExit(
            f"Benchmark preparation failed for: {', '.join(sorted(failed_datasets))}. "
            f"See {summary_path} for details."
        )


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
            "cbbq",
            "borderlines",
            "msqad",
            "culturalbench_easy",
            "culturalbench_hard",
            "espanstereo",
            "honest",
        ],
    )
    parser.add_argument("--out-root", default="data/benchmarks/external")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--espanstereo-path",
        default="",
        help="Optional local CSV/JSON/JSONL path or directory for EspanStereo source data.",
    )
    parser.add_argument(
        "--honest-subsets",
        nargs="*",
        default=list(HONEST_DEFAULT_SUBSETS),
        help=(
            "Optional HONEST subset list. Defaults to the six multilingual binary-gender subsets. "
            "Pass 'all' to include the English queer/non-queer subset as well."
        ),
    )
    parser.add_argument(
        "--honest-source-root",
        default="",
        help=(
            "Optional local HONEST raw source directory or file. "
            "If omitted, the converter first checks data/benchmarks/external/raw/honest/ "
            "and then falls back to the upstream raw TSV URLs."
        ),
    )
    parser.add_argument(
        "--msqad-source-root",
        default="",
        help=(
            "Optional MSQAD source root. By default the converter reads from "
            "data/benchmarks/external/raw/msqad_repo/MSQAD/."
        ),
    )
    parser.add_argument(
        "--msqad-model-source",
        default="gpt-3.5-turbo-0125",
        help=(
            "MSQAD model subdirectory to read from under the source root. "
            "Defaults to the fully populated gpt-3.5-turbo-0125 release."
        ),
    )
    args = parser.parse_args()
    run(args)
