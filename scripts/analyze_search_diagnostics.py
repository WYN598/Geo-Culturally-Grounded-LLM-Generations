import argparse
import json
import math
import os
import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def normalize_text(text: str) -> str:
    s = str(text or "").lower().strip()
    s = re.sub(r"^[a-z]\s*[\)\.]\s*", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z]{2,}", normalize_text(text))


def answer_text(row: Dict[str, Any]) -> str:
    ans = str(row.get("answer", "")).strip().upper()
    choices = list(row.get("choices", []) or [])
    if not ans or not choices:
        return ""
    idx = ord(ans) - ord("A")
    if idx < 0 or idx >= len(choices):
        return ""
    return normalize_text(choices[idx])


def choice_texts(row: Dict[str, Any]) -> List[str]:
    return [normalize_text(c) for c in (row.get("choices", []) or []) if normalize_text(c)]


def any_contains(texts: List[str], needle: str) -> bool:
    n = normalize_text(needle)
    if len(n) < 2:
        return False
    for t in texts:
        if n in normalize_text(t):
            return True
    return False


def clip(text: str, n: int = 260) -> str:
    s = str(text or "").strip().replace("\n", " ")
    if len(s) <= n:
        return s
    return s[: n - 3] + "..."


def idx_by_id(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        rid = str(r.get("id", "")).strip()
        if rid:
            out[rid] = r
    return out


def split_failure_bucket(
    is_correct: bool,
    candidate_has_gold: Optional[bool],
    selected_has_gold: bool,
) -> str:
    if is_correct:
        return "correct"
    if candidate_has_gold is False:
        return "search_fail"
    if candidate_has_gold is True and not selected_has_gold:
        return "rank_fail"
    if selected_has_gold:
        return "context_or_reasoning_fail"
    return "search_or_rank_fail_unknown"


def cosine_sim(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return float(dot / (math.sqrt(na) * math.sqrt(nb)))


class SemanticMatcher:
    def __init__(
        self,
        mode: str = "hybrid",
        provider: str = "openai",
        model: str = "text-embedding-3-small",
        threshold: float = 0.78,
        max_candidates: int = 12,
        text_max_chars: int = 320,
        batch_size: int = 64,
    ) -> None:
        self.mode = (mode or "hybrid").strip().lower()
        if self.mode not in {"lexical", "semantic", "hybrid"}:
            raise ValueError("match mode must be one of lexical/semantic/hybrid")

        self.provider = (provider or "openai").strip().lower()
        self.model = model
        self.threshold = float(threshold)
        self.max_candidates = max(1, int(max_candidates))
        self.text_max_chars = max(64, int(text_max_chars))
        self.batch_size = max(1, int(batch_size))

        self._emb_cache: Dict[str, List[float]] = {}
        self.stats: Dict[str, int] = {
            "semantic_checks": 0,
            "embed_calls": 0,
            "embedded_texts": 0,
            "cache_hits": 0,
            "cache_misses": 0,
        }

        self.client = None
        if self.mode in {"semantic", "hybrid"}:
            if self.provider != "openai":
                raise ValueError("Only openai provider is supported for semantic matching.")
            from openai import OpenAI

            api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAIAPI")
            if not api_key:
                raise RuntimeError(
                    "Semantic matching needs OPENAI_API_KEY/OPENAIAPI when mode is semantic/hybrid."
                )
            base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
            self.client = OpenAI(api_key=api_key, base_url=base_url)

    @staticmethod
    def _dedupe_texts(texts: List[str]) -> List[str]:
        seen = set()
        out: List[str] = []
        for t in texts:
            t2 = normalize_text(t)
            if not t2 or t2 in seen:
                continue
            seen.add(t2)
            out.append(t2)
        return out

    def _clip_for_embedding(self, text: str) -> str:
        return normalize_text(text)[: self.text_max_chars]

    def _lexical_overlap_score(self, a: str, b: str) -> float:
        ta = set(tokenize(a))
        tb = set(tokenize(b))
        if not ta or not tb:
            return 0.0
        inter = len(ta & tb)
        return float(inter) / float(max(1, len(ta)))

    def _shortlist(self, gold_text: str, texts: List[str]) -> List[str]:
        uniq = self._dedupe_texts(texts)
        if len(uniq) <= self.max_candidates:
            return uniq

        scored: List[Tuple[float, int, str]] = []
        for t in uniq:
            score = self._lexical_overlap_score(gold_text, t)
            scored.append((score, len(t), t))
        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
        out = [t for _, _, t in scored[: self.max_candidates]]
        return out

    def _embed_many(self, texts: List[str]) -> None:
        if not texts:
            return
        missing: List[str] = []
        for t in texts:
            if t in self._emb_cache:
                self.stats["cache_hits"] += 1
            else:
                self.stats["cache_misses"] += 1
                missing.append(t)
        if not missing:
            return
        if self.client is None:
            raise RuntimeError("Semantic client is not initialized.")

        for i in range(0, len(missing), self.batch_size):
            batch = missing[i : i + self.batch_size]
            resp = self.client.embeddings.create(model=self.model, input=batch)
            self.stats["embed_calls"] += 1
            self.stats["embedded_texts"] += len(batch)
            for txt, item in zip(batch, resp.data):
                self._emb_cache[txt] = list(item.embedding or [])

    def _semantic_best(self, gold_text: str, texts: List[str]) -> Tuple[float, str]:
        self.stats["semantic_checks"] += 1
        gold = self._clip_for_embedding(gold_text)
        cands = [self._clip_for_embedding(t) for t in self._shortlist(gold_text, texts)]
        cands = [c for c in cands if c]
        if not gold or not cands:
            return 0.0, ""

        self._embed_many([gold] + cands)
        gold_vec = self._emb_cache.get(gold, [])
        best_sim = 0.0
        best_text = ""
        for c in cands:
            sim = cosine_sim(gold_vec, self._emb_cache.get(c, []))
            if sim > best_sim:
                best_sim = sim
                best_text = c
        return best_sim, best_text

    def match_texts(self, texts: List[str], gold_text: str) -> Tuple[bool, str, Dict[str, Any]]:
        lexical_hit = any_contains(texts, gold_text)
        info: Dict[str, Any] = {
            "lexical_hit": lexical_hit,
            "semantic_hit": False,
            "semantic_max_sim": 0.0,
            "semantic_best_text": "",
        }

        if self.mode == "lexical":
            return lexical_hit, ("lexical" if lexical_hit else "none"), info

        if self.mode == "hybrid" and lexical_hit:
            return True, "lexical", info

        max_sim, best_text = self._semantic_best(gold_text, texts)
        semantic_hit = max_sim >= self.threshold
        info["semantic_hit"] = semantic_hit
        info["semantic_max_sim"] = float(max_sim)
        info["semantic_best_text"] = best_text

        if self.mode == "semantic":
            return semantic_hit, ("semantic" if semantic_hit else "none"), info

        hit = lexical_hit or semantic_hit
        if lexical_hit:
            source = "lexical"
        elif semantic_hit:
            source = "semantic"
        else:
            source = "none"
        return hit, source, info

    def summary(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "provider": self.provider,
            "model": self.model,
            "threshold": self.threshold,
            "max_candidates": self.max_candidates,
            "text_max_chars": self.text_max_chars,
            "batch_size": self.batch_size,
            "stats": dict(self.stats),
        }


def analyze(
    search_rows: List[Dict[str, Any]],
    cache_by_id: Dict[str, Dict[str, Any]],
    vanilla_by_id: Dict[str, Dict[str, Any]],
    matcher: SemanticMatcher,
) -> Dict[str, Any]:
    dataset_rows: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    rows_with_tags: List[Dict[str, Any]] = []

    for r in search_rows:
        ds = str(r.get("dataset", "unknown"))

        rid = str(r.get("id", "")).strip()
        trace = r.get("search_trace", {}) or {}
        selected_ev = list(trace.get("selected_evidence", []) or [])
        selected_texts = [str(x.get("text", "")) for x in selected_ev if isinstance(x, dict)]
        selected_urls = [str(x.get("url", "")) for x in selected_ev if isinstance(x, dict)]

        cached = cache_by_id.get(rid, {})
        candidate_ev = list(cached.get("candidate_evidence", []) or [])
        candidate_texts = [str(x.get("text", "")) for x in candidate_ev if isinstance(x, dict)]

        gold_txt = answer_text(r)
        selected_has_gold, selected_match_source, selected_match_info = matcher.match_texts(selected_texts, gold_txt)
        if candidate_texts:
            candidate_has_gold, candidate_match_source, candidate_match_info = matcher.match_texts(candidate_texts, gold_txt)
        else:
            candidate_has_gold = None
            candidate_match_source = "unknown"
            candidate_match_info = {
                "lexical_hit": False,
                "semantic_hit": False,
                "semantic_max_sim": 0.0,
                "semantic_best_text": "",
            }

        q_list = [str(x) for x in (trace.get("queries", []) or []) if str(x).strip()]
        q_norm = [normalize_text(x) for x in q_list]
        q_identical = len(set(q_norm)) <= 1
        q_contains_gold = any_contains(q_list, gold_txt)
        q_missing = len(q_list) == 0

        is_correct = str(r.get("pred", "")).upper() == str(r.get("answer", "")).upper()
        bucket = split_failure_bucket(is_correct, candidate_has_gold, selected_has_gold)

        v = vanilla_by_id.get(rid, {})
        vanilla_correct = (
            str(v.get("pred", "")).upper() == str(v.get("answer", "")).upper() if v else None
        )
        degraded_vs_vanilla = bool(v) and vanilla_correct and (not is_correct)

        tagged = {
            "id": rid,
            "dataset": ds,
            "question": r.get("question", ""),
            "choices": r.get("choices", []),
            "answer": r.get("answer", ""),
            "pred_search": r.get("pred", ""),
            "pred_vanilla": v.get("pred", "") if v else "",
            "is_correct": is_correct,
            "vanilla_correct": vanilla_correct,
            "degraded_vs_vanilla": degraded_vs_vanilla,
            "bucket": bucket,
            "queries": q_list,
            "query_issue": {"missing_queries": q_missing, "low_diversity": q_identical, "contains_gold": q_contains_gold},
            "retrieved_hits": int(trace.get("retrieved_hits", 0) or 0),
            "dedup_hits": int(trace.get("dedup_hits", 0) or 0),
            "candidate_chunks": int(trace.get("candidate_chunks", 0) or 0),
            "top_selected_score": float(trace.get("top_selected_score", 0.0) or 0.0),
            "selected_has_gold": selected_has_gold,
            "candidate_has_gold": candidate_has_gold,
            "selected_match_source": selected_match_source,
            "candidate_match_source": candidate_match_source,
            "selected_lexical_hit": bool(selected_match_info.get("lexical_hit", False)),
            "candidate_lexical_hit": bool(candidate_match_info.get("lexical_hit", False)),
            "selected_semantic_hit": bool(selected_match_info.get("semantic_hit", False)),
            "candidate_semantic_hit": bool(candidate_match_info.get("semantic_hit", False)),
            "selected_semantic_max_sim": float(selected_match_info.get("semantic_max_sim", 0.0) or 0.0),
            "candidate_semantic_max_sim": float(candidate_match_info.get("semantic_max_sim", 0.0) or 0.0),
            "selected_semantic_best_preview": clip(str(selected_match_info.get("semantic_best_text", "")), n=160),
            "candidate_semantic_best_preview": clip(str(candidate_match_info.get("semantic_best_text", "")), n=160),
            "selected_urls": selected_urls,
            "selected_evidence_preview": [clip(x) for x in selected_texts[:2]],
            "answer_text": gold_txt,
            "choice_overlap_in_selected": any(any_contains(selected_texts, c) for c in choice_texts(r)),
        }
        rows_with_tags.append(tagged)
        dataset_rows[ds].append(tagged)

    def agg(sub_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        m = len(sub_rows)
        if m == 0:
            return {}
        bucket_counts = Counter(x["bucket"] for x in sub_rows)
        degraded = [x for x in sub_rows if x["degraded_vs_vanilla"]]
        known_candidate_rows = [x for x in sub_rows if x["candidate_has_gold"] is not None]
        return {
            "n": m,
            "accuracy": sum(1 for x in sub_rows if x["is_correct"]) / m,
            "retrieval_stats": {
                "avg_retrieved_hits": sum(x["retrieved_hits"] for x in sub_rows) / m,
                "avg_dedup_hits": sum(x["dedup_hits"] for x in sub_rows) / m,
                "avg_candidate_chunks": sum(x["candidate_chunks"] for x in sub_rows) / m,
                "zero_candidate_chunks_rate": sum(1 for x in sub_rows if x["candidate_chunks"] == 0) / m,
                "zero_selected_evidence_rate": sum(1 for x in sub_rows if len(x["selected_urls"]) == 0) / m,
            },
            "query_stats": {
                "missing_queries_rate": sum(1 for x in sub_rows if x["query_issue"]["missing_queries"]) / m,
                "low_diversity_queries_rate": sum(1 for x in sub_rows if x["query_issue"]["low_diversity"]) / m,
                "query_contains_gold_rate": sum(1 for x in sub_rows if x["query_issue"]["contains_gold"]) / m,
            },
            "failure_buckets": dict(bucket_counts),
            "failure_bucket_rate": {k: v / m for k, v in bucket_counts.items()},
            "selected_has_gold_rate": sum(1 for x in sub_rows if x["selected_has_gold"]) / m,
            "candidate_has_gold_known_rate": len(known_candidate_rows) / m,
            "candidate_has_gold_rate_when_known": (
                sum(1 for x in known_candidate_rows if x["candidate_has_gold"] is True) / max(1, len(known_candidate_rows))
            ),
            "selected_semantic_hit_rate": sum(1 for x in sub_rows if x["selected_semantic_hit"]) / m,
            "candidate_semantic_hit_rate_when_known": (
                sum(1 for x in known_candidate_rows if x["candidate_semantic_hit"]) / max(1, len(known_candidate_rows))
            ),
            "degraded_vs_vanilla_count": len(degraded),
            "degraded_vs_vanilla_rate": len(degraded) / m,
            "avg_top_selected_score_correct": (
                sum(x["top_selected_score"] for x in sub_rows if x["is_correct"])
                / max(1, sum(1 for x in sub_rows if x["is_correct"]))
            ),
            "avg_top_selected_score_wrong": (
                sum(x["top_selected_score"] for x in sub_rows if not x["is_correct"])
                / max(1, sum(1 for x in sub_rows if not x["is_correct"]))
            ),
        }

    overall = agg(rows_with_tags)
    by_dataset = {ds: agg(rows) for ds, rows in sorted(dataset_rows.items())}

    return {"overall": overall, "by_dataset": by_dataset, "tagged_rows": rows_with_tags}


def build_case_studies(tagged_rows: List[Dict[str, Any]], top_k: int) -> Dict[str, Any]:
    buckets = ["search_fail", "rank_fail", "context_or_reasoning_fail", "search_or_rank_fail_unknown"]
    out: Dict[str, Any] = {}
    for b in buckets:
        rows = [x for x in tagged_rows if x["bucket"] == b]
        rows = sorted(
            rows,
            key=lambda x: (
                1 if x.get("degraded_vs_vanilla") else 0,
                x.get("top_selected_score", 0.0),
            ),
            reverse=True,
        )
        out[b] = rows[:top_k]
    return out


def run(args: argparse.Namespace) -> None:
    search_rows = load_jsonl(args.search_preds)
    cache_rows = load_jsonl(args.search_cache) if args.search_cache and os.path.exists(args.search_cache) else []
    vanilla_rows = load_jsonl(args.vanilla_preds) if args.vanilla_preds and os.path.exists(args.vanilla_preds) else []

    cache_by_id = idx_by_id(cache_rows)
    vanilla_by_id = idx_by_id(vanilla_rows)

    matcher = SemanticMatcher(
        mode=args.match_mode,
        provider=args.semantic_provider,
        model=args.semantic_model,
        threshold=args.semantic_threshold,
        max_candidates=args.semantic_max_candidates,
        text_max_chars=args.semantic_text_max_chars,
        batch_size=args.semantic_batch_size,
    )

    report = analyze(search_rows, cache_by_id, vanilla_by_id, matcher=matcher)
    cases = build_case_studies(report["tagged_rows"], top_k=args.top_k_cases)

    out_dir = os.path.dirname(args.out_json) or "."
    os.makedirs(out_dir, exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        out_report = {k: v for k, v in report.items() if k != "tagged_rows"}
        out_report["match_config"] = matcher.summary()
        json.dump(out_report, f, ensure_ascii=False, indent=2)

    out_cases_dir = os.path.dirname(args.out_cases) or "."
    os.makedirs(out_cases_dir, exist_ok=True)
    with open(args.out_cases, "w", encoding="utf-8") as f:
        json.dump(cases, f, ensure_ascii=False, indent=2)

    if args.out_tagged:
        out_tagged_dir = os.path.dirname(args.out_tagged) or "."
        os.makedirs(out_tagged_dir, exist_ok=True)
        with open(args.out_tagged, "w", encoding="utf-8") as f:
            for row in report["tagged_rows"]:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(
        json.dumps(
            {
                "out_json": args.out_json,
                "out_cases": args.out_cases,
                "out_tagged": args.out_tagged,
                "match_config": matcher.summary(),
                "overall": report["overall"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Layered diagnostics for search-grounding results.")
    parser.add_argument("--search-preds", required=True, help="search_predictions.jsonl")
    parser.add_argument("--search-cache", default="", help="frozen search cache jsonl with candidate_evidence")
    parser.add_argument("--vanilla-preds", default="", help="vanilla_predictions.jsonl for degraded-case detection")
    parser.add_argument("--out-json", default="outputs/analysis/search_diagnostics.json")
    parser.add_argument("--out-cases", default="outputs/analysis/search_case_studies.json")
    parser.add_argument("--out-tagged", default="", help="optional jsonl with per-sample layered tags")
    parser.add_argument("--top-k-cases", type=int, default=15)
    parser.add_argument(
        "--match-mode",
        default="hybrid",
        choices=["lexical", "semantic", "hybrid"],
        help="How to detect whether evidence supports the gold answer.",
    )
    parser.add_argument(
        "--semantic-provider",
        default="openai",
        choices=["openai"],
        help="Provider for semantic matching embeddings.",
    )
    parser.add_argument("--semantic-model", default="text-embedding-3-small")
    parser.add_argument("--semantic-threshold", type=float, default=0.78)
    parser.add_argument("--semantic-max-candidates", type=int, default=12)
    parser.add_argument("--semantic-text-max-chars", type=int, default=320)
    parser.add_argument("--semantic-batch-size", type=int, default=64)
    run(parser.parse_args())

