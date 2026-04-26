import argparse
import csv
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def letter_choices(values: List[str]) -> List[str]:
    return [f"{chr(ord('A') + i)}) {normalize_text(v)}" for i, v in enumerate(values)]


def load_tsv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def choose_label(human_value: Any, llm_value: Any, annotation_type: str) -> Any:
    if annotation_type == "human":
        return human_value
    if annotation_type == "llm":
        return llm_value
    return human_value if human_value not in {None, ""} else llm_value


def extract_query_choices(query_text: str, fallback_claimants: List[str]) -> List[str]:
    markers = list(re.finditer(r"([A-Z])\)\s*", query_text))
    if len(markers) < 2:
        return list(fallback_claimants)
    parts: List[str] = []
    for idx, match in enumerate(markers):
        start = match.end()
        end = markers[idx + 1].start() if idx + 1 < len(markers) else len(query_text)
        option = query_text[start:end].strip()
        option = re.sub(r"\s+(or|ou|oder|o|或|или|atau|veya|বা|और|ou bien)\s*$", "", option, flags=re.IGNORECASE)
        option = option.strip(" ؟?。！？،,.;:")
        parts.append(option)
    cleaned = [x for x in parts if x]
    if len(cleaned) == len(fallback_claimants):
        return cleaned
    return list(fallback_claimants)


def build_output_stem(query_langs: List[str], system: str, mode: str) -> str:
    lang_part = "-".join(sorted(query_langs))
    return f"bordirlines_{lang_part}_{system}_{mode}"


def load_annotation_maps(source_root: Path, llm_mode: str) -> Tuple[Dict[Tuple[str, str], Dict[str, Any]], Dict[Tuple[str, str], Dict[str, Any]]]:
    human_rows = load_tsv(source_root / "data" / "human_annotations.tsv")
    llm_rows = load_tsv(source_root / "data" / "llm_annotations.tsv")
    human_map = {(row["query_id"], row["doc_id"]): row for row in human_rows}
    llm_map = {(row["query_id"], row["doc_id"]): row for row in llm_rows}
    normalized_llm_map: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for key, row in llm_map.items():
        normalized_llm_map[key] = {
            "relevant": normalize_text(row.get(f"relevant_{llm_mode}")),
            "territory": normalize_text(row.get(f"territory_{llm_mode}")).split(") ", 1)[-1] if normalize_text(row.get(f"territory_{llm_mode}")) else "",
        }
    normalized_human_map: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for key, row in human_map.items():
        normalized_human_map[key] = {
            "relevant": normalize_text(row.get("relevant")),
            "territory": normalize_text(row.get("territory")),
        }
    return normalized_human_map, normalized_llm_map


def as_bool(value: Any) -> bool:
    return normalize_text(value).upper() == "TRUE"


def run(args: argparse.Namespace) -> None:
    source_root = Path(args.source_root)
    data_root = source_root / "data"
    if not data_root.exists():
        raise FileNotFoundError(f"BordIRLines data root was not found: {data_root}")

    with (data_root / "all_docs.json").open("r", encoding="utf-8") as f:
        docs = json.load(f)
    query_rows = load_tsv(data_root / "queries.tsv")
    query_map = {row["query_id"]: row for row in query_rows}
    human_map, llm_map = load_annotation_maps(source_root, llm_mode=args.llm_mode)

    out_root = Path(args.out_root)
    ensure_dir(out_root / "processed")
    ensure_dir(out_root / "caches")
    ensure_dir(out_root / "reports")

    summary: Dict[str, Any] = {"outputs": []}
    for system in args.systems:
        for mode in args.modes:
            eval_rows: List[Dict[str, Any]] = []
            cache_rows: List[Dict[str, Any]] = []
            for query_lang in args.query_langs:
                hits_path = data_root / query_lang / system / mode / f"{query_lang}_query_hits.tsv"
                if not hits_path.exists():
                    raise FileNotFoundError(f"Missing BordIRLines hits file: {hits_path}")
                hits_rows = load_tsv(hits_path)
                grouped_hits: Dict[str, List[Dict[str, str]]] = {}
                for row in hits_rows:
                    grouped_hits.setdefault(row["query_id"], []).append(row)

                for query_id, q_hits in sorted(grouped_hits.items()):
                    query_entry = query_map.get(query_id)
                    if not query_entry:
                        continue
                    claimants = [normalize_text(x) for x in normalize_text(query_entry.get("Claimants")).split(";") if normalize_text(x)]
                    if len(claimants) < 2:
                        continue
                    controller = normalize_text(query_entry.get("Controller"))
                    answer = ""
                    if controller and controller.lower() != "unknown" and controller in claimants:
                        answer = chr(ord("A") + claimants.index(controller))
                    query_text = normalize_text(query_entry.get("query_text"))
                    choices = letter_choices(extract_query_choices(query_text, claimants))
                    cache_id = f"BordIRLines_{system}_{mode}_{query_id}"

                    eval_rows.append(
                        {
                            "id": cache_id,
                            "dataset": f"BordIRLines-{system}-{mode}",
                            "task_type": "geopolitical_mcq",
                            "benchmark_family": "bias",
                            "question": query_text,
                            "choices": choices,
                            "answer": answer,
                            "metadata": {
                                "language": query_lang,
                                "territory": normalize_text(query_entry.get("territory")),
                                "controller": controller,
                                "claimants_en": claimants,
                                "query_id": query_id,
                                "system": system,
                                "mode": mode,
                                "has_gold": bool(answer),
                                "eval_protocol": "fixed_candidate_geopolitical_rag",
                            },
                        }
                    )

                    raw_candidate_evidence: List[Dict[str, Any]] = []
                    for hit in q_hits[: args.n_hits]:
                        doc_id = normalize_text(hit.get("doc_id"))
                        doc_lang = normalize_text(hit.get("doc_lang"))
                        human_data = human_map.get((query_id, doc_id), {})
                        llm_data = llm_map.get((query_id, doc_id), {})
                        relevant = choose_label(
                            as_bool(human_data.get("relevant")),
                            as_bool(llm_data.get("relevant")),
                            args.annotation_type,
                        )
                        viewpoint = choose_label(
                            normalize_text(human_data.get("territory")),
                            normalize_text(llm_data.get("territory")),
                            args.annotation_type,
                        )
                        raw_candidate_evidence.append(
                            {
                                "query": query_text,
                                "title": f"{doc_id} ({doc_lang})",
                                "url": f"bordirlines://{query_id}/{doc_lang}/{doc_id}",
                                "domain": f"bordirlines.{doc_lang}",
                                "score": float(hit.get("score") or 0.0),
                                "text": str(docs.get(doc_lang, {}).get(doc_id, "")),
                                "doc_id": doc_id,
                                "doc_lang": doc_lang,
                                "query_id": query_id,
                                "relevant_human": as_bool(human_data.get("relevant")),
                                "relevant_llm": as_bool(llm_data.get("relevant")),
                                "relevant": bool(relevant),
                                "viewpoint_human": normalize_text(human_data.get("territory")),
                                "viewpoint_llm": normalize_text(llm_data.get("territory")),
                                "viewpoint": normalize_text(viewpoint),
                            }
                        )

                    cache_rows.append(
                        {
                            "id": cache_id,
                            "question": query_text,
                            "queries": [query_text],
                            "retrieved_hits": len(q_hits[: args.n_hits]),
                            "dedup_hits": len(q_hits[: args.n_hits]),
                            "raw_candidate_chunks": len(raw_candidate_evidence),
                            "raw_candidate_evidence": raw_candidate_evidence,
                            "search_plan": {
                                "source": "bordirlines",
                                "query_id": query_id,
                                "query_lang": query_lang,
                                "system": system,
                                "mode": mode,
                                "annotation_type": args.annotation_type,
                                "llm_mode": args.llm_mode,
                            },
                        }
                    )

            stem = build_output_stem(args.query_langs, system, mode)
            eval_path = out_root / "processed" / f"{stem}.jsonl"
            cache_path = out_root / "caches" / f"{stem}.jsonl"
            write_jsonl(eval_path, eval_rows)
            write_jsonl(cache_path, cache_rows)
            summary["outputs"].append(
                {
                    "system": system,
                    "mode": mode,
                    "query_langs": list(args.query_langs),
                    "eval_path": str(eval_path),
                    "cache_path": str(cache_path),
                    "rows": len(eval_rows),
                }
            )

    summary_path = out_root / "reports" / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare fixed-candidate BordIRLines evaluation rows and cache files.")
    parser.add_argument("--source-root", default="data/benchmarks/external/raw/bordirlines_repo")
    parser.add_argument("--out-root", default="data/benchmarks/bordirlines")
    parser.add_argument("--query-langs", nargs="+", default=["en"])
    parser.add_argument("--systems", nargs="+", choices=["openai", "m3"], default=["openai"])
    parser.add_argument("--modes", nargs="+", choices=["qlang", "qlang_en", "en", "rel_langs"], default=["rel_langs"])
    parser.add_argument("--n-hits", type=int, default=10)
    parser.add_argument("--annotation-type", choices=["human", "llm", "auto"], default="human")
    parser.add_argument("--llm-mode", choices=["zeroshot", "fewshot"], default="fewshot")
    run(parser.parse_args())
