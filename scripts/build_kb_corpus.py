import argparse
import csv
import json
import re
import urllib.parse
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import requests


def read_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", str(s or "")).strip()


def split_text(text: str, chunk_chars: int) -> List[str]:
    text = normalize_text(text)
    if not text:
        return []
    if len(text) <= chunk_chars:
        return [text]
    out = []
    start = 0
    while start < len(text):
        out.append(text[start : start + chunk_chars].strip())
        start += chunk_chars
    return [x for x in out if x]


def load_seed_kb(path: Path) -> List[Dict]:
    out: List[Dict] = []
    for r in read_jsonl(path):
        out.append(
            {
                "source": str(r.get("source", "SeedKB")),
                "country": str(r.get("country", "")),
                "text": normalize_text(str(r.get("text", ""))),
            }
        )
    return out


def parse_country_from_filename(name: str) -> str:
    stem = name.rsplit(".", 1)[0]
    m = re.search(r"\bin\s+(.+)$", stem)
    return m.group(1).strip() if m else ""


def load_seegull_kb(raw_root: Path, max_rows: int = 0) -> List[Dict]:
    stereo_dir = raw_root / "seegull" / "SeeGULL-Multilingual-main" / "dataset" / "stereotypes"
    if not stereo_dir.exists():
        return []

    rows: List[Dict] = []
    csv_files = sorted(stereo_dir.glob("*.csv"))
    for fp in csv_files:
        country = parse_country_from_filename(fp.name)
        with fp.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for r in reader:
                identity = normalize_text(r.get("translated_identity") or r.get("identity") or "")
                attr = normalize_text(r.get("translated_attribute_list") or r.get("attribute") or "")
                if not identity or not attr:
                    continue
                s = normalize_text(r.get("stereotype", "0"))
                n = normalize_text(r.get("non_stereotype", "0"))
                u = normalize_text(r.get("unsure", "0"))
                text = (
                    f"Group-attribute claim: '{identity}' -> '{attr}'. "
                    f"SeeGULL annotation counts: stereotype={s}, non_stereotype={n}, unsure={u}. "
                    "Use this as cultural-sensitivity context, not as absolute truth."
                )
                rows.append({"source": "SeeGULL", "country": country, "text": text})
                if max_rows > 0 and len(rows) >= max_rows:
                    return rows
    return rows


def load_extra_jsonl(paths: List[Path]) -> List[Dict]:
    rows: List[Dict] = []
    for p in paths:
        for r in read_jsonl(p):
            text = normalize_text(str(r.get("text", "")))
            if not text:
                continue
            rows.append(
                {
                    "source": str(r.get("source", "ExtraSource")),
                    "country": str(r.get("country", "")),
                    "text": text,
                }
            )
    return rows


def load_wikipedia_country_summaries(countries_file: Path, max_countries: int = 0, lang: str = "en") -> List[Dict]:
    if not countries_file.exists():
        return []

    with countries_file.open("r", encoding="utf-8-sig") as f:
        countries = [normalize_text(x) for x in f if normalize_text(x)]

    if max_countries > 0:
        countries = countries[:max_countries]

    rows: List[Dict] = []
    sess = requests.Session()
    for c in countries:
        title = urllib.parse.quote(c.replace(" ", "_"))
        url = f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{title}"
        try:
            resp = sess.get(url, timeout=12)
            if not resp.ok:
                continue
            obj = resp.json()
            extract = normalize_text(obj.get("extract", ""))
            if not extract:
                continue
            rows.append({"source": "Wikipedia", "country": c, "text": extract})
        except Exception:
            continue
    return rows


def dedup_docs(rows: List[Dict]) -> List[Dict]:
    seen = set()
    out = []
    for r in rows:
        key = (normalize_text(r.get("source", "")).lower(), normalize_text(r.get("country", "")).lower(), normalize_text(r.get("text", "")).lower())
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def run(args: argparse.Namespace) -> None:
    raw_root = Path(args.raw_root)
    out_path = Path(args.out)

    rows: List[Dict] = []
    if args.include_seed:
        rows.extend(load_seed_kb(Path(args.seed_kb)))
    if args.include_seegull:
        rows.extend(load_seegull_kb(raw_root, max_rows=args.max_seegull_rows))
    if args.extra_jsonl:
        rows.extend(load_extra_jsonl([Path(x) for x in args.extra_jsonl]))
    if args.wikipedia_countries_file:
        rows.extend(
            load_wikipedia_country_summaries(
                Path(args.wikipedia_countries_file),
                max_countries=args.max_wikipedia_countries,
                lang=args.wikipedia_lang,
            )
        )

    rows = dedup_docs(rows)
    final_rows: List[Dict] = []
    idx = 0
    for r in rows:
        chunks = split_text(str(r.get("text", "")), args.chunk_chars)
        for t in chunks:
            final_rows.append(
                {
                    "id": f"kb_{idx}",
                    "source": str(r.get("source", "")),
                    "country": str(r.get("country", "")),
                    "text": t,
                }
            )
            idx += 1

    write_jsonl(out_path, final_rows)
    summary = {
        "out": str(out_path),
        "num_docs": len(final_rows),
        "chunk_chars": args.chunk_chars,
        "sources": sorted({str(r.get("source", "")) for r in final_rows}),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build KB corpus for KB-grounding.")
    parser.add_argument("--raw-root", default="data/benchmarks/raw")
    parser.add_argument("--out", default="data/kb_full.jsonl")
    parser.add_argument("--chunk-chars", type=int, default=900)

    parser.add_argument("--include-seed", action="store_true")
    parser.add_argument("--seed-kb", default="data/sample_kb.jsonl")
    parser.add_argument("--include-seegull", action="store_true")
    parser.add_argument("--max-seegull-rows", type=int, default=0)

    parser.add_argument("--extra-jsonl", nargs="*", default=[])

    parser.add_argument("--wikipedia-countries-file", default="")
    parser.add_argument("--max-wikipedia-countries", type=int, default=0)
    parser.add_argument("--wikipedia-lang", default="en")

    args = parser.parse_args()
    run(args)
