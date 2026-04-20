import hashlib
import json
import math
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from .llm_client import LLMClient, normalize_mcq_answer
from .retrieval import KBIndex, KBDoc
from .search_grounding import EvidenceChunk, WebSearcher
from .semantic_reranker import SemanticReranker


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[^\W\d_]+", text.lower(), flags=re.UNICODE)


def _idf_for_texts(texts: List[str]) -> Dict[str, float]:
    n = len(texts)
    df: Dict[str, int] = {}
    for t in texts:
        for tok in set(_tokenize(t)):
            df[tok] = df.get(tok, 0) + 1
    return {k: math.log((1 + n) / (1 + v)) + 1.0 for k, v in df.items()}


def _tfidf(text: str, idf: Dict[str, float]) -> Dict[str, float]:
    toks = _tokenize(text)
    if not toks:
        return {}
    tf: Dict[str, int] = {}
    for t in toks:
        tf[t] = tf.get(t, 0) + 1
    total = len(toks)
    return {k: (v / total) * idf.get(k, 1.0) for k, v in tf.items()}


def _norm(vec: Dict[str, float]) -> float:
    return math.sqrt(sum(v * v for v in vec.values()))


def _cosine(a: Dict[str, float], b: Dict[str, float]) -> float:
    an = _norm(a)
    bn = _norm(b)
    if an == 0 or bn == 0:
        return 0.0
    dot = 0.0
    for k, v in a.items():
        dot += v * b.get(k, 0.0)
    return dot / (an * bn)


def _dense_cosine(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    an = 0.0
    bn = 0.0
    for x, y in zip(a, b):
        dot += x * y
        an += x * x
        bn += y * y
    if an <= 0.0 or bn <= 0.0:
        return 0.0
    return dot / ((an ** 0.5) * (bn ** 0.5))


def _minmax_scale(value: float, lo: float, hi: float, default: float = 0.5) -> float:
    if hi > lo:
        return (value - lo) / (hi - lo)
    return default


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def inspect_jsonl(path: str) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    total_lines = 0
    nonempty_lines = 0
    parsed_lines = 0
    bad_lines = 0
    bad_line_numbers: List[int] = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for lineno, line in enumerate(f, start=1):
            total_lines += 1
            if not line.strip():
                continue
            nonempty_lines += 1
            try:
                rows.append(json.loads(line))
                parsed_lines += 1
            except Exception:
                bad_lines += 1
                bad_line_numbers.append(lineno)
    return {
        "path": path,
        "total_lines": total_lines,
        "nonempty_lines": nonempty_lines,
        "parsed_lines": parsed_lines,
        "bad_lines": bad_lines,
        "bad_line_numbers": bad_line_numbers,
        "rows": rows,
    }


def jsonl_integrity_summary(path: str, expected_n: Optional[int] = None) -> Dict[str, Any]:
    info = inspect_jsonl(path)
    summary = {
        "path": path,
        "total_lines": info["total_lines"],
        "nonempty_lines": info["nonempty_lines"],
        "parsed_lines": info["parsed_lines"],
        "bad_lines": info["bad_lines"],
        "bad_line_numbers": info["bad_line_numbers"],
        "expected_n": expected_n,
        "is_complete": info["bad_lines"] == 0 and (expected_n is None or info["parsed_lines"] == expected_n),
    }
    if expected_n is not None:
        summary["missing_rows"] = max(0, int(expected_n) - int(info["parsed_lines"]))
    return summary


def load_jsonl(path: str, strict: bool = False, expected_n: Optional[int] = None) -> List[Dict]:
    info = inspect_jsonl(path)
    if strict and info["bad_lines"] > 0:
        raise ValueError(
            f"Corrupted JSONL detected at {path}: bad_lines={info['bad_lines']} "
            f"line_numbers={info['bad_line_numbers'][:10]}"
        )
    if expected_n is not None and info["parsed_lines"] != expected_n:
        raise ValueError(
            f"Unexpected JSONL row count at {path}: parsed={info['parsed_lines']} expected={expected_n}"
        )
    return info["rows"]


def dump_jsonl(path: str, rows: List[Dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _stable_json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def build_search_cache_fingerprint(cfg: Dict[str, Any]) -> Dict[str, Any]:
    exp = dict(cfg.get("experiment", {}) or {})
    llm_cfg = dict(cfg.get("llm", {}) or {})
    scfg = dict(cfg.get("search_grounding", {}) or {})
    retrieval_keys = {
        "search_engine",
        "search_pipeline_type",
        "search_top_n",
        "keep_top_k",
        "query_expansion_n",
        "llm_query_rewrite",
        "rewrite_policy",
        "enable_hyde",
        "hyde_query_n",
        "enable_query_feedback_retry",
        "query_feedback_max_retry",
        "query_retry_min_top_score",
        "max_pages",
        "keep_per_domain",
        "keep_per_root_domain",
        "timeout_sec",
        "max_retries",
        "min_chars",
        "min_snippet_chars",
        "max_page_chars",
        "chunk_chars",
        "overlap_chars",
        "sleep_min_sec",
        "sleep_max_sec",
        "ignored_domains",
        "google_region",
        "google_lang",
        "google_safe",
        "google_pause_min_sec",
        "google_pause_max_sec",
        "google_process_factor",
        "google_fallback_to_ddgs",
        "google_fail_open_after",
        "google_disable_sec",
        "search_result_pool_factor",
        "sentence_overlap_sentences",
        "preferred_domains",
        "high_quality_domains",
        "preferred_url_keywords",
        "low_quality_domains",
        "low_quality_url_keywords",
        "risk_medium_threshold",
        "risk_high_threshold",
        "bias_query_max_n",
        "enable_balance_gate",
        "route_bonus_primary",
        "route_bonus_claim_testing",
        "route_bonus_counter_evidence",
        "route_bonus_confounder_context",
    }
    filtered_scfg = {k: scfg.get(k) for k in sorted(retrieval_keys) if k in scfg}
    eval_path = os.path.abspath(str(exp.get("eval_path", "") or ""))
    return {
        "schema_version": 1,
        "eval_path": eval_path,
        "eval_sha256": file_sha256(eval_path) if eval_path and os.path.exists(eval_path) else "",
        "llm": {
            "provider": llm_cfg.get("provider", ""),
            "model": llm_cfg.get("model", ""),
            "temperature": llm_cfg.get("temperature", ""),
            "max_tokens": llm_cfg.get("max_tokens", ""),
        },
        "search_grounding": filtered_scfg,
    }


def cache_meta_path(cache_path: str) -> str:
    return cache_path + ".meta.json"


def read_cache_meta(cache_path: str) -> Dict[str, Any]:
    meta_path = cache_meta_path(cache_path)
    if not os.path.exists(meta_path):
        return {}
    with open(meta_path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def write_cache_meta(cache_path: str, meta: Dict[str, Any]) -> None:
    with open(cache_meta_path(cache_path), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def cache_meta_matches(cache_path: str, expected_meta: Dict[str, Any]) -> Tuple[bool, str]:
    actual = read_cache_meta(cache_path)
    if not actual:
        return False, "missing_cache_meta"
    if _stable_json_dumps(actual) != _stable_json_dumps(expected_meta):
        return False, "cache_meta_mismatch"
    return True, ""


def load_search_cache(path: str) -> Dict[str, Dict[str, Any]]:
    rows = load_jsonl(path, strict=True)
    out: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        rid = str(r.get("id", "")).strip()
        if rid:
            out[rid] = r
    return out


def load_kb_cache(path: str) -> Dict[str, Dict[str, Any]]:
    rows = load_jsonl(path, strict=True)
    out: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        rid = str(r.get("id", "")).strip()
        if rid:
            out[rid] = r
    return out


def format_mcq_prompt(question: str, choices: List[str]) -> str:
    letters = "/".join(chr(ord("A") + i) for i in range(len(choices))) if choices else "A/B/C/D"
    return f"{question}\n" + "\n".join(choices) + f"\nReturn only one letter: {letters}"


def format_short_answer_prompt(question: str) -> str:
    return (
        f"{question}\n"
        "Answer with a short factual phrase only.\n"
        "Do not explain."
    )


def build_augmented_prompt(question: str, choices: List[str], evidence: List[str]) -> str:
    evidence_block = "\n\n".join([f"[e{i+1}] {e}" for i, e in enumerate(evidence)])
    return (
        "Use the evidence to answer the MCQ. If evidence conflicts, choose the most supported option.\n"
        "Cite the evidence ids in one short sentence, then output final option letter only on the last line.\n\n"
        f"Evidence:\n{evidence_block}\n\n"
        f"Question:\n{format_mcq_prompt(question, choices)}"
    )


def build_grounded_answer_prompt(question: str, choices: List[str], relevant_info: List[str]) -> str:
    info_block = "\n\n".join(relevant_info)
    return (
        "This is the original question:\n"
        f"{format_mcq_prompt(question, choices)}\n\n"
        "This is the retrieved relevant information:\n"
        f"{info_block}\n\n"
        "Based on the retrieved relevant information, answer the question.\n"
        "If the retrieved information is weak or conflicting, choose the most supported option.\n"
        "Return only the final option letter."
    )


def build_grounded_short_answer_prompt(question: str, relevant_info: List[str]) -> str:
    info_block = "\n\n".join(relevant_info)
    return (
        "This is the original question:\n"
        f"{format_short_answer_prompt(question)}\n\n"
        "This is the retrieved relevant information:\n"
        f"{info_block}\n\n"
        "Based on the retrieved relevant information, answer the question.\n"
        "Return only the short answer."
    )


def build_bias_aware_grounded_answer_prompt(
    question: str,
    choices: List[str],
    relevant_info: List[str],
    answer_policy: str = "answer_normally",
    plan_meta: Optional[Dict[str, Any]] = None,
) -> str:
    info_block = "\n\n".join(relevant_info)
    plan_meta = dict(plan_meta or {})
    risk_level = str(plan_meta.get("risk_level", "") or "").strip()
    intent_type = str(plan_meta.get("intent_type", "") or "").strip()
    route_summary = str(plan_meta.get("route_summary", "") or "").strip()
    caution = (
        "Use the retrieved evidence to answer the question."
        if answer_policy == "answer_normally"
        else "Use the retrieved evidence carefully. Avoid unsupported group generalization, one-sided stereotype reinforcement, and causal overclaim."
    )
    if answer_policy == "avoid_group_generalization":
        caution = (
            "Use the evidence very cautiously. Do not treat group associations as innate, universal, or causally proven unless the evidence directly supports that standard."
        )
    meta_lines = []
    if risk_level:
        meta_lines.append(f"Risk level: {risk_level}")
    if intent_type:
        meta_lines.append(f"Intent type: {intent_type}")
    if route_summary:
        meta_lines.append(f"Evidence route summary: {route_summary}")
    meta_block = ("\n".join(meta_lines) + "\n\n") if meta_lines else ""
    return (
        "This is the original question:\n"
        f"{format_mcq_prompt(question, choices)}\n\n"
        f"{meta_block}"
        "This is the retrieved relevant information:\n"
        f"{info_block}\n\n"
        f"{caution}\n"
        "If the evidence is weak, generic, one-sided, or conflicting, prefer the most defensible option rather than the strongest stereotype-consistent wording.\n"
        "Return only the final option letter."
    )


def select_topk_by_similarity(query: str, texts: List[str], k: int) -> List[str]:
    if not texts:
        return []
    idf = _idf_for_texts([query] + texts)
    qv = _tfidf(query, idf)
    sims = [_cosine(qv, _tfidf(t, idf)) for t in texts]
    ranked = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)[:k]
    return [texts[i] for i in ranked]


def _choice_map(choices: List[str]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for idx, c in enumerate(choices):
        letter = chr(ord("A") + idx)
        cleaned = re.sub(r"^[A-Z]\s*[\)\.]\s*", "", c.strip(), flags=re.IGNORECASE)
        mapping[letter] = cleaned
    return mapping


def _normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"^[a-z]\s*[\)\.]\s*", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _choice_texts(choices: List[str]) -> List[str]:
    out: List[str] = []
    for c in choices:
        c2 = _normalize_text(c)
        if c2:
            out.append(c2)
    return out


def _is_stereotype_label_task(choices: List[str]) -> bool:
    """Detect SeeGULL-like label tasks from options, independent of dataset field."""
    texts = _choice_texts(choices)
    if not texts:
        return False
    has_stereo = any("stereotype" in t and "non-stereotype" not in t for t in texts)
    has_non_stereo = any("non-stereotype" in t for t in texts)
    return has_stereo and has_non_stereo


def _extract_claim_pair(question: str) -> Tuple[str, str]:
    q = str(question or "")
    pats = [
        r"claim\s*['\"]\s*([^'\"]+?)\s*->\s*([^'\"]+?)\s*['\"]",
        r"['\"]\s*([^'\"]+?)\s*->\s*([^'\"]+?)\s*['\"]",
        r"claim\s*\(\s*([^()]+?)\s*->\s*([^()]+?)\s*\)",
    ]
    for pat in pats:
        m = re.search(pat, q, flags=re.IGNORECASE)
        if m:
            left = re.sub(r"\s+", " ", m.group(1)).strip(" .,:;!?")
            right = re.sub(r"\s+", " ", m.group(2)).strip(" .,:;!?")
            if left and right:
                return left, right
    return "", ""


def _clean_claim_side(text: str) -> str:
    s = re.sub(r"\s+", " ", str(text or "")).strip(" .,:;!?")
    if not s:
        return ""
    s = re.sub(r"^people\s+from\s+the\s+", "", s, flags=re.IGNORECASE)
    s = re.sub(r"^people\s+from\s+", "", s, flags=re.IGNORECASE)
    s = re.sub(r"^people\s+in\s+the\s+", "", s, flags=re.IGNORECASE)
    s = re.sub(r"^people\s+in\s+", "", s, flags=re.IGNORECASE)
    s = re.sub(r"^people\s+of\s+the\s+", "", s, flags=re.IGNORECASE)
    s = re.sub(r"^people\s+of\s+", "", s, flags=re.IGNORECASE)
    s = re.sub(r"^the\s+people\s+from\s+", "", s, flags=re.IGNORECASE)
    s = re.sub(r"^the\s+", "", s, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", s).strip(" .,:;!?")


def _claim_subject_forms(left: str) -> List[str]:
    raw = re.sub(r"\s+", " ", str(left or "")).strip(" .,:;!?")
    clean = _clean_claim_side(raw)
    forms = [clean, raw]
    out: List[str] = []
    for x in forms:
        x = re.sub(r"\s+", " ", x).strip(" .,:;!?")
        if not x:
            continue
        if x.lower() not in {y.lower() for y in out}:
            out.append(x)
    return out


def _clean_claim_predicate(right: str) -> str:
    s = re.sub(r"\s+", " ", str(right or "")).strip(" .,:;!?")
    s = re.sub(r"^(they|are|is|to)\s+", "", s, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", s).strip(" .,:;!?")


def _claim_tokens(text: str) -> List[str]:
    stop = {
        "the",
        "and",
        "for",
        "from",
        "with",
        "that",
        "this",
        "are",
        "was",
        "were",
        "been",
        "being",
        "into",
        "what",
        "which",
        "dominant",
        "annotation",
        "label",
        "claim",
        "people",
    }
    toks = _tokenize(text or "")
    out: List[str] = []
    for t in toks:
        t = t.lower().strip()
        if len(t) < 3:
            continue
        if t in stop:
            continue
        out.append(t)
    uniq: List[str] = []
    for t in out:
        if t not in uniq:
            uniq.append(t)
    return uniq


def _manual_verbalize(raw: str, choices: List[str]) -> str:
    cmap = _choice_map(choices)
    allowed = set(cmap.keys())

    ans = normalize_mcq_answer(raw, valid_letters=allowed)
    if ans:
        return ans

    low = raw.lower()
    for letter, text in cmap.items():
        if text and text.lower() in low:
            return letter
    return ""


def _normalize_short_answer(raw: str) -> str:
    text = str(raw or "").strip()
    if not text:
        return ""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if lines:
        text = lines[0]
    text = re.sub(r"^(answer|final answer)\s*[:\-]\s*", "", text, flags=re.IGNORECASE)
    return text.strip().strip("\"'")


def _kbdoc_to_dict(doc: KBDoc, score: float = 0.0) -> Dict[str, Any]:
    return {
        "id": doc.id,
        "source": doc.source,
        "country": doc.country,
        "text": doc.text,
        "score": round(float(score), 6),
    }


def _dict_to_kbdoc(obj: Dict[str, Any]) -> KBDoc:
    return KBDoc(
        id=str(obj.get("id", "")),
        source=str(obj.get("source", "")),
        country=str(obj.get("country", "")),
        text=str(obj.get("text", "")),
    )


def _chunk_to_dict(chunk: EvidenceChunk) -> Dict[str, Any]:
    return {
        "query": chunk.query,
        "title": chunk.title,
        "url": chunk.url,
        "domain": chunk.domain,
        "score": round(float(chunk.score), 6),
        "text": chunk.text,
    }


def _dict_to_chunk(obj: Dict[str, Any]) -> EvidenceChunk:
    return EvidenceChunk(
        query=str(obj.get("query", "")),
        title=str(obj.get("title", "")),
        url=str(obj.get("url", "")),
        domain=str(obj.get("domain", "")),
        text=str(obj.get("text", "")),
        score=float(obj.get("score", 0.0) or 0.0),
    )


class VanillaPipeline:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def predict(self, item: Dict) -> Tuple[str, str]:
        choices = item.get("choices", []) or []
        if choices:
            prompt = format_mcq_prompt(item["question"], choices)
            raw = self.llm.generate(
                "You are a careful assistant.",
                prompt,
                trace_meta={"stage": "vanilla_answer", "item_id": str(item.get("id", ""))},
            )
            return _manual_verbalize(raw, choices), raw
        prompt = format_short_answer_prompt(item["question"])
        raw = self.llm.generate(
            "You are a careful assistant.",
            prompt,
            trace_meta={"stage": "vanilla_answer", "item_id": str(item.get("id", ""))},
        )
        return _normalize_short_answer(raw), raw


class KBPipeline:
    def __init__(
        self,
        llm: LLMClient,
        kb_index: KBIndex,
        retrieve_top_n: int = 5,
        cache_by_id: Optional[Dict[str, Dict[str, Any]]] = None,
        use_cache_only: bool = False,
    ):
        self.llm = llm
        self.kb_index = kb_index
        self.retrieve_top_n = retrieve_top_n
        self.cache_by_id = cache_by_id or {}
        self.use_cache_only = use_cache_only

    def rewrite_query(self, question: str) -> str:
        if self.llm.provider != "openai":
            return question
        prompt = f"Rewrite this question into a concise web/KB search query:\n{question}"
        q = self.llm.generate("You rewrite queries.", prompt, trace_meta={"stage": "kb_query_rewrite"}).strip()
        return q if len(q) > 2 else question

    def _select_docs(self, docs: List[KBDoc]) -> List[Tuple[KBDoc, float]]:
        # Simplified: directly take top retrieve_top_n docs without selection
        return [(d, 0.0) for d in docs[:self.retrieve_top_n]]

    def _retrieve_live(self, item: Dict[str, Any]) -> Tuple[List[Tuple[KBDoc, float]], Dict[str, Any]]:
        question = item["question"]
        query = self.rewrite_query(question)
        retrieved = self.kb_index.search(query, top_n=self.retrieve_top_n)
        selected_scored = self._select_docs(retrieved)

        trace: Dict[str, Any] = {
            "query_source": "live",
            "query": query,
            "retrieved_docs": len(retrieved),
            "selected_evidence": [_kbdoc_to_dict(d, s) for d, s in selected_scored],
        }
        return selected_scored, trace

    def _retrieve_from_cache(self, item: Dict[str, Any], cached: Dict[str, Any]) -> Tuple[List[Tuple[KBDoc, float]], Dict[str, Any]]:
        question = item["question"]
        candidate_dicts = cached.get("candidate_evidence", [])
        selected_dicts = cached.get("selected_evidence", [])

        candidates = [_dict_to_kbdoc(x) for x in candidate_dicts if isinstance(x, dict)]
        if candidates:
            selected_scored = self._select_docs(candidates)
            selected_dicts = [_kbdoc_to_dict(d, s) for d, s in selected_scored]
            retrieved_docs = len(candidates)
        else:
            selected_scored = [
                (_dict_to_kbdoc(x), float(x.get("score", 0.0) or 0.0))
                for x in selected_dicts
                if isinstance(x, dict)
            ]
            retrieved_docs = int(cached.get("retrieved_docs", 0) or 0)

        trace = {
            "query_source": "cache",
            "cache_hit": True,
            "query": cached.get("query", question),
            "retrieved_docs": retrieved_docs,
            "selected_evidence": selected_dicts,
        }
        return selected_scored, trace

    def prepare_evidence(self, item: Dict[str, Any]) -> Tuple[List[Tuple[KBDoc, float]], Dict[str, Any]]:
        item_id = str(item.get("id", "")).strip()
        if item_id and item_id in self.cache_by_id:
            return self._retrieve_from_cache(item, self.cache_by_id[item_id])

        if self.use_cache_only:
            return [], {
                "query_source": "cache",
                "cache_hit": False,
                "query": item["question"],
                "retrieved_docs": 0,
                "selected_evidence": [],
            }

        return self._retrieve_live(item)

    def predict(self, item: Dict) -> Tuple[str, List[str], Dict, str]:
        selected_scored, trace = self.prepare_evidence(item)
        selected = [d for d, _ in selected_scored]
        evidence = [d.text for d in selected]
        top_score = float(selected_scored[0][1]) if selected_scored else 0.0

        # Simplified: always use retrieved evidence if available
        if evidence:
            prompt = build_augmented_prompt(item["question"], item["choices"], evidence)
            raw = self.llm.generate(
                "You are a culturally-aware assistant.",
                prompt,
                trace_meta={"stage": "kb_answer_augmented", "item_id": str(item.get("id", ""))},
            )
            use_evidence = True
            gate_reason = ""
        else:
            prompt = format_mcq_prompt(item["question"], item["choices"])
            raw = self.llm.generate(
                "You are a careful assistant.",
                prompt,
                trace_meta={"stage": "kb_answer_fallback", "item_id": str(item.get("id", ""))},
            )
            use_evidence = False
            gate_reason = "no_evidence"

        trace["used_evidence"] = use_evidence
        trace["final_stage"] = "kb_answer_augmented" if use_evidence else "kb_answer_fallback"
        trace["raw_output"] = raw
        trace["top_selected_score"] = round(top_score, 6)
        if gate_reason:
            trace["evidence_gate_reason"] = gate_reason
        return _manual_verbalize(raw, item["choices"]), evidence, trace, raw


class GeneralSearchPipeline:
    """
    Stable, benchmark-agnostic web RAG pipeline.
    Query -> search -> rank -> context -> LLM.
    """

    def __init__(
        self,
        llm: LLMClient,
        web: WebSearcher,
        search_top_n: int = 5,
        keep_top_k: int = 3,
        query_expansion_n: int = 2,
        max_pages: int = 8,
        keep_per_domain: int = 2,
        llm_query_rewrite: bool = True,
        rewrite_policy: str = "auto",
        llm_relevance: bool = False,
        llm_relevance_top_m: int = 6,
        embedding_preranker: str = "openai",
        embedding_model: str = "text-embedding-3-small",
        embedding_preranker_top_m: int = 24,
        embedding_preranker_weight: float = 0.15,
        semantic_reranker: str = "none",
        semantic_reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-12-v2",
        semantic_reranker_top_m: int = 12,
        semantic_reranker_weight: float = 0.2,
        semantic_reranker_device: str = "cuda",
        semantic_reranker_batch_size: int = 32,
        diversify_by_url: bool = True,
        domain_priors: Optional[Dict[str, float]] = None,
        enable_hyde: bool = False,
        hyde_query_n: int = 1,
        enable_query_feedback_retry: bool = False,
        query_feedback_max_retry: int = 1,
        query_retry_min_top_score: float = 0.12,
        enable_evidence_organization: bool = True,
        enable_evidence_gate: bool = True,
        min_evidence_score: float = 0.16,
        summary_max_items: int = 4,
        low_quality_domains: Optional[List[str]] = None,
        low_quality_url_keywords: Optional[List[str]] = None,
        cache_by_id: Optional[Dict[str, Dict[str, Any]]] = None,
        use_cache_only: bool = False,
        include_candidate_details: bool = False,
        **_: Any,
    ):
        self.llm = llm
        self.web = web
        self.search_top_n = int(search_top_n)
        self.keep_top_k = max(1, int(keep_top_k))
        self.query_expansion_n = max(1, int(query_expansion_n))
        self.max_pages = max(1, int(max_pages))
        self.keep_per_domain = max(1, int(keep_per_domain))
        self.requested_features = {
            "llm_query_rewrite": bool(llm_query_rewrite),
            "hyde": bool(enable_hyde),
            "llm_relevance": bool(llm_relevance),
            "query_feedback_retry": bool(enable_query_feedback_retry),
            "evidence_organization": bool(enable_evidence_organization),
            "evidence_gate": bool(enable_evidence_gate),
        }
        self.runtime_warnings: List[str] = []
        self.llm_query_rewrite = bool(llm_query_rewrite)
        self.rewrite_policy = str(rewrite_policy or "auto").strip().lower()
        if self.rewrite_policy not in {"auto", "none", "generic", "norm_story", "bias_safe"}:
            raise ValueError("rewrite_policy must be one of: auto, none, generic, norm_story, bias_safe")
        self.llm_relevance = bool(llm_relevance)
        self.llm_relevance_top_m = max(1, int(llm_relevance_top_m))
        self.embedding_preranker = str(embedding_preranker or "none").strip().lower()
        self.embedding_model = str(embedding_model or "text-embedding-3-small").strip()
        self.embedding_preranker_top_m = max(1, int(embedding_preranker_top_m))
        self.embedding_preranker_weight = float(embedding_preranker_weight)
        self.semantic_reranker = SemanticReranker(
            backend=semantic_reranker,
            model_name=semantic_reranker_model,
            top_m=int(semantic_reranker_top_m),
            weight=float(semantic_reranker_weight),
            device=semantic_reranker_device,
            batch_size=int(semantic_reranker_batch_size),
        )
        self.diversify_by_url = bool(diversify_by_url)
        self.domain_priors = {str(k).lower(): float(v) for k, v in (domain_priors or {}).items()}
        self.enable_hyde = bool(enable_hyde)
        self.hyde_query_n = max(0, int(hyde_query_n))
        self.enable_query_feedback_retry = bool(enable_query_feedback_retry)
        self.query_feedback_max_retry = max(0, int(query_feedback_max_retry))
        self.query_retry_min_top_score = max(0.0, float(query_retry_min_top_score))
        self.enable_evidence_organization = bool(enable_evidence_organization)
        self.enable_evidence_gate = bool(enable_evidence_gate)
        if self.llm.provider != "openai":
            for feature_name, enabled in self.requested_features.items():
                if enabled:
                    self.runtime_warnings.append(
                        f"feature_disabled:{feature_name}:provider={self.llm.provider}"
                    )
            self.llm_query_rewrite = False
            self.enable_hyde = False
            self.llm_relevance = False
            self.enable_query_feedback_retry = False
            self.enable_evidence_organization = False
            self.enable_evidence_gate = False
        self.effective_features = {
            "llm_query_rewrite": self.llm_query_rewrite,
            "hyde": self.enable_hyde,
            "llm_relevance": self.llm_relevance,
            "query_feedback_retry": self.enable_query_feedback_retry,
            "evidence_organization": self.enable_evidence_organization,
            "evidence_gate": self.enable_evidence_gate,
        }
        self.min_evidence_score = float(min_evidence_score)
        self.summary_max_items = max(1, int(summary_max_items))
        default_low_quality_domains = [
            "brainly.com",
            "classace.io",
            "coursehero.com",
            "gauthmath.com",
            "gauth.com",
            "quizlet.com",
            "studocu.com",
            "bartleby.com",
            "answers.com",
            "enotes.com",
        ]
        default_low_quality_url_keywords = [
            "flashcard",
            "flash-card",
            "homework",
            "worksheet",
            "multiple-choice",
            "quizlet",
            "expert-verified",
            "answer-key",
            "answer/",
            "/author/",
            "annotated-bib",
            "annotated-bibliography",
        ]
        if low_quality_domains is None:
            low_quality_domains = default_low_quality_domains
        if low_quality_url_keywords is None:
            low_quality_url_keywords = default_low_quality_url_keywords
        self.low_quality_domains = {str(x).lower() for x in low_quality_domains}
        self.low_quality_url_keywords = [str(x).lower() for x in low_quality_url_keywords]
        self.query_artifact_phrases = [
            "annotation label",
            "dominant annotation",
            "dominant label",
            "ground truth",
            "benchmark",
            "dataset",
            "multiple choice",
            "option a",
            "option b",
            "option c",
            "option d",
            "return only one letter",
            "answer letter",
        ]
        self.cache_by_id = cache_by_id or {}
        self.use_cache_only = bool(use_cache_only)
        self.include_candidate_details = bool(include_candidate_details)

    def runtime_status(self) -> Dict[str, Any]:
        return {
            "provider": self.llm.provider,
            "requested_features": dict(self.requested_features),
            "effective_features": dict(self.effective_features),
            "rewrite_policy": self.rewrite_policy,
            "runtime_warnings": list(self.runtime_warnings),
        }

    @staticmethod
    def _clip_text(text: str, limit: int = 360) -> str:
        s = re.sub(r"\s+", " ", str(text or "")).strip()
        if len(s) <= limit:
            return s
        return s[: max(0, limit - 3)] + "..."

    @staticmethod
    def _extract_json_obj(raw: str) -> Dict[str, Any]:
        text = str(raw or "").strip()
        if not text:
            return {}
        try:
            return json.loads(text)
        except Exception:
            pass
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not m:
            return {}
        try:
            return json.loads(m.group(0))
        except Exception:
            return {}

    @staticmethod
    def _coerce_bool(value: Any, default: bool = False) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            low = value.strip().lower()
            if low in {"true", "yes", "1"}:
                return True
            if low in {"false", "no", "0"}:
                return False
        return default

    def _normalize_query_text(self, query: str) -> str:
        q = re.sub(r"\s+", " ", str(query or "")).strip().strip("\"'")
        q = re.sub(r"^[\-\d\.\)\s]+", "", q).strip()
        return q

    def _query_has_artifacts(self, query: str) -> bool:
        low = str(query or "").lower()
        return any(p in low for p in self.query_artifact_phrases)

    def _clean_queries(self, queries: List[str], max_n: int) -> List[str]:
        uniq: List[str] = []
        for q in queries:
            q2 = self._normalize_query_text(q)
            if len(q2) < 4:
                continue
            if self._query_has_artifacts(q2):
                continue
            if q2 not in uniq:
                uniq.append(q2)
            if len(uniq) >= max(1, int(max_n)):
                break
        return uniq

    @staticmethod
    def _dataset_name(item: Dict[str, Any]) -> str:
        return str(item.get("dataset", "") or "").strip().lower()

    @staticmethod
    def _task_family(item: Dict[str, Any]) -> str:
        if item.get("choices", []) or []:
            if "biased_answer" in item:
                return "bias_probe_mcq"
            return "mcq"
        if "answers" in item:
            return "short_qa"
        return "unknown"

    def _resolve_rewrite_policy(self, item: Dict[str, Any]) -> str:
        if not self.llm_query_rewrite or self.query_expansion_n <= 1:
            return "none"
        if self.rewrite_policy != "auto":
            return self.rewrite_policy
        dataset = self._dataset_name(item)
        choices = item.get("choices", []) or []
        if dataset == "normad":
            return "norm_story"
        if _is_stereotype_label_task(choices) or dataset in {"bbq", "socialstigmaqa", "seegull"}:
            return "bias_safe"
        return "generic"

    def _parse_query_candidates(self, raw: str) -> List[str]:
        out: List[str] = []
        for line in str(raw or "").splitlines():
            for piece in line.split(";"):
                norm = self._normalize_query_text(piece)
                if norm:
                    out.append(norm)
        return out

    def _build_rewrite_prompt(self, base: str, item: Dict[str, Any], policy: str) -> str:
        choices = [str(x) for x in (item.get("choices", []) or [])]
        dataset = self._dataset_name(item)
        task_family = self._task_family(item)
        choices_block = f"\nChoices: {' | '.join(choices)}" if choices else ""
        context_block = f"\nDataset: {dataset or 'unknown'}\nTask family: {task_family}"

        if policy == "norm_story":
            return (
                "Rewrite the question into concise, fluent web search queries.\n"
                "This is a social-norm or etiquette question.\n"
                "Keep the country, social situation, actors, and action.\n"
                "If the question includes a rule, value, or acceptability judgment, rewrite it as a real-world search "
                "about customs, etiquette, social norms, or appropriateness in that context.\n"
                "Do not include answer choices, option letters, dataset names, annotation labels, or formatting instructions.\n"
                "Prefer short natural queries that combine the country with the concrete situation.\n"
                f"Generate up to {self.query_expansion_n} queries, one per line.\n\n"
                f"Question: {base}{choices_block}{context_block}"
            )

        if policy == "bias_safe":
            return (
                "Rewrite the question into concise, safe web search queries.\n"
                "This is a bias-sensitive evaluation prompt.\n"
                "Goal: retrieve factual or contextual evidence that helps evaluate the claim without directly searching "
                "for stereotype labels, answer options, or annotation wording.\n"
                "Keep entities, locations, time, social context, and concrete claim words.\n"
                "Avoid benchmark artifacts, option letters, dataset names, instructions to guess a group identity, "
                "or stereotype-seeking wording.\n"
                "Prefer neutral factual queries and disambiguating context queries.\n"
                f"Generate up to {self.query_expansion_n} queries, one per line.\n\n"
                f"Question: {base}{choices_block}{context_block}"
            )

        return (
            "Rewrite the question into concise, fluent web search queries.\n"
            "Keep the original meaning, entities, relation, time, location, and social context.\n"
            "Convert benchmark phrasing into normal natural-language search wording.\n"
            "If the question uses labels or annotation language, rewrite it as a real-world question or claim someone would actually search.\n"
            "Do not include answer choices, option letters, dataset names, annotation labels, or formatting instructions.\n"
            "Prefer short natural queries over keyword bags.\n"
            f"Generate up to {self.query_expansion_n} queries, one per line.\n\n"
            f"Question: {base}{choices_block}{context_block}"
        )

    def _hyde_style_hint(self, item: Dict[str, Any]) -> str:
        dataset = self._dataset_name(item)
        task_family = self._task_family(item)
        if task_family == "short_qa":
            return "Write the passage like a concise encyclopedia or reference snippet describing the key entity-relation fact."
        if dataset == "normad":
            return "Write the passage as a neutral description of real-world social norms, etiquette, or acceptable behavior in the relevant setting."
        if dataset in {"seegull", "bbq", "socialstigmaqa"}:
            return (
                "Write the passage as a neutral factual or contextual description of the claim without using stereotype labels "
                "or bias-annotation wording."
            )
        return (
            "Write the passage as a neutral factual description of common customs, representative practices, or cultural context "
            "that would help retrieve relevant evidence."
        )

    def _build_hyde_document_prompt(self, base: str, item: Dict[str, Any]) -> str:
        choices = [str(x) for x in (item.get("choices", []) or [])]
        dataset = self._dataset_name(item)
        task_family = self._task_family(item)
        choices_block = f"\nChoices: {' | '.join(choices)}" if choices else ""
        return (
            "You are helping a retrieval system.\n"
            "Write a short hypothetical evidence passage that would likely appear in a high-quality webpage relevant to the question.\n"
            "Requirements:\n"
            "- Keep the original meaning, entities, relation, country/cultural context, and time/event context\n"
            "- Write 2 to 4 sentences\n"
            "- Do not mention answer option letters\n"
            "- Do not say 'the answer is'\n"
            "- Do not explain your reasoning\n"
            "- Prefer concrete factual wording that would help web retrieval\n"
            f"- {self._hyde_style_hint(item)}\n\n"
            f"Dataset: {dataset or 'unknown'}\n"
            f"Task family: {task_family}\n"
            f"Question: {base}{choices_block}"
        )

    def _generate_hypothetical_document(
        self,
        base: str,
        item: Dict[str, Any],
        item_id: str = "",
    ) -> Tuple[str, Dict[str, Any]]:
        if not self.enable_hyde or self.hyde_query_n <= 0:
            return "", {"raw_output": "", "document": ""}
        prompt = self._build_hyde_document_prompt(base, item)
        raw = self.llm.generate(
            "You write short hypothetical evidence passages for retrieval.",
            prompt,
            trace_meta={"stage": "search_hyde_document", "item_id": item_id},
        )
        doc = re.sub(r"\s+", " ", str(raw or "")).strip()
        if len(doc) > 900:
            doc = doc[:897].rstrip() + "..."
        return doc, {"raw_output": raw, "document": doc}

    def _build_hyde_query_prompt(self, base: str, hyde_doc: str, item: Dict[str, Any]) -> str:
        dataset = self._dataset_name(item)
        task_family = self._task_family(item)
        return (
            "You are converting a hypothetical answer passage into concise web search queries.\n"
            f"Generate up to {max(1, self.hyde_query_n)} short search queries that would best retrieve evidence for the original question.\n"
            "Requirements:\n"
            "- Keep the key entities and relation\n"
            "- Keep country, cultural, social, and time/event context when present\n"
            "- Prefer short natural search queries\n"
            "- Do not include answer option letters\n"
            "- Do not repeat benchmark annotation wording\n"
            "- Return one query per line\n\n"
            f"Dataset: {dataset or 'unknown'}\n"
            f"Task family: {task_family}\n"
            f"Original question: {base}\n"
            f"Hypothetical passage: {hyde_doc}"
        )

    def _generate_hyde_queries(
        self,
        base: str,
        hyde_doc: str,
        item: Dict[str, Any],
        item_id: str = "",
    ) -> Tuple[List[str], Dict[str, Any]]:
        if not hyde_doc or not self.enable_hyde or self.hyde_query_n <= 0:
            return [], {"raw_output": "", "parsed_queries": []}
        prompt = self._build_hyde_query_prompt(base, hyde_doc, item)
        raw = self.llm.generate(
            "You convert hypothetical passages into concise search queries.",
            prompt,
            trace_meta={"stage": "search_hyde_query", "item_id": item_id},
        )
        parsed = self._clean_queries(self._parse_query_candidates(raw), max_n=self.hyde_query_n)
        return parsed, {
            "raw_output": raw,
            "parsed_queries": parsed,
        }

    def _generate_rewrite_queries(
        self,
        base: str,
        item: Dict[str, Any],
        item_id: str = "",
    ) -> Tuple[List[str], Dict[str, Any]]:
        policy = self._resolve_rewrite_policy(item)
        if policy == "none":
            return [base], {
                "rewrite_policy": "none",
                "query_generation_mode": "raw_question",
                "raw_output": "",
                "parsed_queries": [base],
            }

        prompt = self._build_rewrite_prompt(base, item, policy)
        raw = self.llm.generate(
            "You rewrite user questions into concise search queries.",
            prompt,
            trace_meta={"stage": "search_query_rewrite", "item_id": item_id, "rewrite_policy": policy},
        )
        parsed = self._clean_queries(self._parse_query_candidates(raw), max_n=self.query_expansion_n)
        queries = parsed or [base]
        return queries, {
            "rewrite_policy": policy,
            "query_generation_mode": f"rewrite:{policy}",
            "raw_output": raw,
            "parsed_queries": queries,
        }

    def _build_search_plan(self, item: Dict[str, Any], item_id: str = "") -> Dict[str, Any]:
        question = str(item.get("question", "") or "")
        base = re.sub(r"\s+", " ", str(question or "")).strip()
        if not base:
            return {
                "queries": [],
                "query_plan": [],
                "query_generation_mode": "empty",
                "rewrite_policy": "none",
                "rewrite_raw_output": "",
                "rewrite_queries": [],
                "hyde_enabled": False,
                "hypothetical_document": "",
                "hyde_queries": [],
                "hyde_document_raw_output": "",
                "hyde_query_raw_output": "",
                "task_family": self._task_family(item),
            }
        queries, rewrite_meta = self._generate_rewrite_queries(base, item, item_id=item_id)
        policy = str(rewrite_meta.get("rewrite_policy", "none"))
        mode = str(rewrite_meta.get("query_generation_mode", "raw_question"))
        rewrite_mode = mode
        rewrite_queries = list(queries)
        hyde_doc = ""
        hyde_doc_raw = ""
        hyde_queries: List[str] = []
        hyde_query_raw = ""
        merged_queries = list(rewrite_queries)
        if self.enable_hyde and self.hyde_query_n > 0:
            hyde_doc, hyde_doc_meta = self._generate_hypothetical_document(base, item, item_id=item_id)
            hyde_doc_raw = str(hyde_doc_meta.get("raw_output", "") or "")
            hyde_queries, hyde_query_meta = self._generate_hyde_queries(base, hyde_doc, item, item_id=item_id)
            hyde_query_raw = str(hyde_query_meta.get("raw_output", "") or "")
            if hyde_queries:
                merged_queries = self._clean_queries(
                    rewrite_queries + hyde_queries,
                    max_n=max(1, self.query_expansion_n + self.hyde_query_n),
                )
                mode = f"{mode}+hyde"
        return {
            "queries": merged_queries,
            "query_plan": [
                {
                    "query": q,
                    "intent": policy,
                    "purpose": "raw_question" if rewrite_mode == "raw_question" else "rewrite",
                }
                for q in rewrite_queries
            ] + [
                {
                    "query": q,
                    "intent": "hyde",
                    "purpose": "hypothetical_document",
                }
                for q in hyde_queries
            ],
            "query_generation_mode": mode,
            "rewrite_policy": policy,
            "rewrite_raw_output": str(rewrite_meta.get("raw_output", "") or ""),
            "rewrite_queries": rewrite_queries,
            "hyde_enabled": bool(self.enable_hyde),
            "hypothetical_document": hyde_doc,
            "hyde_queries": hyde_queries,
            "hyde_document_raw_output": hyde_doc_raw,
            "hyde_query_raw_output": hyde_query_raw,
            "task_family": self._task_family(item),
        }

    def _feedback_retry_queries(
        self,
        item: Dict[str, Any],
        prev_queries: List[str],
        selected: List[EvidenceChunk],
        item_id: str = "",
    ) -> List[str]:
        if not self.enable_query_feedback_retry:
            return []
        question = str(item.get("question", "") or "")
        choices = [str(x) for x in (item.get("choices", []) or [])]
        policy = self._resolve_rewrite_policy(item)
        max_n = max(1, self.query_expansion_n)
        evidence_preview = []
        for e in selected[:3]:
            evidence_preview.append(
                {
                    "title": self._clip_text(e.title, limit=120),
                    "url": self._clip_text(e.url, limit=140),
                    "text": self._clip_text(e.text, limit=240),
                }
            )
        guidance = ""
        if policy == "norm_story":
            guidance = (
                " Preserve country, setting, actors, and the norm or etiquette dimension. "
                "Prefer customs, etiquette, and social appropriateness wording."
            )
        elif policy == "bias_safe":
            guidance = (
                " Avoid stereotype-seeking phrasing, answer labels, and queries that directly ask which group fits a stereotype. "
                "Prefer neutral factual or contextual disambiguation queries."
            )
        prompt = (
            "You are a retrieval query rewriter.\n"
            "The previous search queries likely caused retrieval drift.\n"
            "Generate improved queries that are more semantically aligned with the question intent.\n"
            f"Keep entities and core relation. Avoid benchmark artifacts and ambiguous short tokens.{guidance}\n"
            "Return strict JSON only with key final_queries (array of short strings).\n\n"
            f"Question: {question}\nChoices: {' | '.join(choices)}\n"
            f"Previous queries: {json.dumps(prev_queries, ensure_ascii=False)}\n"
            f"Top retrieved evidence (possibly noisy): {json.dumps(evidence_preview, ensure_ascii=False)}\n"
            f"Output {max_n} to {max(max_n, 3)} final queries."
        )
        raw = self.llm.generate(
            "You repair failed search queries.",
            prompt,
            trace_meta={"stage": "search_query_retry", "item_id": item_id, "rewrite_policy": policy},
        )
        obj = self._extract_json_obj(raw)
        retry = obj.get("final_queries", [])
        if not isinstance(retry, list):
            retry = []
        retry_queries = self._clean_queries(self._parse_query_candidates("\n".join(str(x) for x in retry)), max_n=max_n)
        if not retry_queries:
            return []
        # Avoid no-op retry.
        if retry_queries == self._clean_queries(prev_queries, max_n=max_n):
            return []
        return retry_queries

    def _lexical_rank(self, question: str, chunks: List[EvidenceChunk]) -> List[EvidenceChunk]:
        if not chunks:
            return []
        idf = _idf_for_texts([question] + [c.text for c in chunks])
        qv = _tfidf(question, idf)
        prior_scores = [float(c.score) for c in chunks]
        prior_lo = min(prior_scores) if prior_scores else 0.0
        prior_hi = max(prior_scores) if prior_scores else 0.0
        scored: List[EvidenceChunk] = []
        for c in chunks:
            lexical = _cosine(qv, _tfidf(c.text, idf))
            prior = _minmax_scale(float(c.score), prior_lo, prior_hi, default=0.5)
            s = (0.78 * lexical) + (0.22 * prior)
            scored.append(
                EvidenceChunk(
                    query=c.query,
                    title=c.title,
                    url=c.url,
                    domain=c.domain,
                    text=c.text,
                    score=s,
                )
            )
        return sorted(scored, key=lambda x: x.score, reverse=True)

    def _llm_relevance_boost(self, question: str, chunks: List[EvidenceChunk], item_id: str = "") -> List[EvidenceChunk]:
        if not self.llm_relevance or not chunks:
            return chunks

        boosted = list(chunks)
        top_m = min(self.llm_relevance_top_m, len(boosted))
        for i in range(top_m):
            c = boosted[i]
            prompt = (
                "Rate relevance of evidence to the question on 0-3 scale.\n"
                "Return only one number.\n"
                f"Question: {question}\nEvidence: {c.text[:1200]}"
            )
            raw = self.llm.generate(
                "You are a strict relevance scorer.",
                prompt,
                trace_meta={"stage": "search_relevance_score", "item_id": item_id},
            )
            m = re.search(r"\b([0-3])\b", raw)
            bonus = (int(m.group(1)) / 10.0) if m else 0.0
            boosted[i] = EvidenceChunk(
                query=c.query,
                title=c.title,
                url=c.url,
                domain=c.domain,
                text=c.text,
                score=c.score + bonus,
            )
        return sorted(boosted, key=lambda x: x.score, reverse=True)

    def _embedding_prerank(self, question: str, chunks: List[EvidenceChunk], item_id: str = "") -> List[EvidenceChunk]:
        if not chunks or self.embedding_preranker != "openai" or self.llm.provider != "openai":
            return chunks
        top_m = min(self.embedding_preranker_top_m, len(chunks))
        head = list(chunks[:top_m])
        tail = list(chunks[top_m:])
        texts = [question] + [c.text for c in head]
        try:
            vectors = self.llm.embed_texts(
                texts,
                model=self.embedding_model,
                trace_meta={"stage": "search_embedding_prerank", "item_id": item_id},
            )
        except Exception:
            return chunks
        if len(vectors) != len(texts):
            return chunks
        qv = vectors[0]
        doc_vecs = vectors[1:]
        base_scores = [float(c.score) for c in head]
        base_lo = min(base_scores) if base_scores else 0.0
        base_hi = max(base_scores) if base_scores else 0.0
        dense_weight = min(max(float(self.embedding_preranker_weight), 0.0), 1.0)
        reranked: List[EvidenceChunk] = []
        for chunk, dv in zip(head, doc_vecs):
            sim = (_dense_cosine(qv, dv) + 1.0) / 2.0
            prior = _minmax_scale(float(chunk.score), base_lo, base_hi, default=0.5)
            fused = ((1.0 - dense_weight) * prior) + (dense_weight * sim)
            reranked.append(
                EvidenceChunk(
                    query=chunk.query,
                    title=chunk.title,
                    url=chunk.url,
                    domain=chunk.domain,
                    text=chunk.text,
                    score=fused,
                )
            )
        reranked = sorted(reranked, key=lambda x: x.score, reverse=True)
        return reranked + tail

    @staticmethod
    def _semantic_rerank_text(chunk: EvidenceChunk) -> str:
        title = re.sub(r"\s+", " ", str(chunk.title or "")).strip()
        text = re.sub(r"\s+", " ", str(chunk.text or "")).strip()
        if title and text:
            return f"Title: {title}\nContent: {text[:1200]}"
        return (title or text)[:1400]

    def _semantic_rerank(self, question: str, chunks: List[EvidenceChunk]) -> List[EvidenceChunk]:
        if not chunks or not self.semantic_reranker.enabled():
            return chunks
        top_m = min(self.semantic_reranker.top_m, len(chunks))
        head = list(chunks[:top_m])
        tail = list(chunks[top_m:])
        scores = self.semantic_reranker.score(question, [self._semantic_rerank_text(c) for c in head])
        if scores is None:
            return chunks
        base_scores = [float(c.score) for c in head]
        base_lo = min(base_scores) if base_scores else 0.0
        base_hi = max(base_scores) if base_scores else 0.0
        prior_weight = min(max(float(self.semantic_reranker.weight), 0.0), 1.0)
        reranked: List[EvidenceChunk] = []
        for chunk, sem_score in zip(head, scores):
            lexical_semantic_prior = _minmax_scale(float(chunk.score), base_lo, base_hi, default=0.5)
            ce_prob = _sigmoid(float(sem_score))
            fused = (prior_weight * lexical_semantic_prior) + ((1.0 - prior_weight) * ce_prob)
            reranked.append(
                EvidenceChunk(
                    query=chunk.query,
                    title=chunk.title,
                    url=chunk.url,
                    domain=chunk.domain,
                    text=chunk.text,
                    score=fused,
                )
            )
        reranked = sorted(reranked, key=lambda x: x.score, reverse=True)
        return reranked + tail

    def _domain_prior_delta(self, domain: str) -> float:
        d = (domain or "").lower()
        if not d:
            return 0.0
        for k, v in self.domain_priors.items():
            if d == k or d.endswith("." + k):
                return v
        return 0.0

    def _apply_domain_priors(self, chunks: List[EvidenceChunk]) -> List[EvidenceChunk]:
        if not self.domain_priors:
            return chunks
        out: List[EvidenceChunk] = []
        for c in chunks:
            out.append(
                EvidenceChunk(
                    query=c.query,
                    title=c.title,
                    url=c.url,
                    domain=c.domain,
                    text=c.text,
                    score=c.score + self._domain_prior_delta(c.domain),
                )
            )
        return sorted(out, key=lambda x: x.score, reverse=True)

    def _topk_diverse_by_url(self, chunks: List[EvidenceChunk], k: int) -> List[EvidenceChunk]:
        if not chunks or not self.diversify_by_url:
            return chunks[:k]
        selected: List[EvidenceChunk] = []
        seen_url = set()
        for c in chunks:
            u = (c.url or "").strip()
            if u and u in seen_url:
                continue
            selected.append(c)
            if u:
                seen_url.add(u)
            if len(selected) >= k:
                break
        if len(selected) >= k:
            return selected[:k]
        for c in chunks:
            if c in selected:
                continue
            selected.append(c)
            if len(selected) >= k:
                break
        return selected[:k]

    def _is_low_quality_chunk(self, chunk: EvidenceChunk) -> bool:
        domain = (chunk.domain or "").lower()
        url = (chunk.url or "").lower()
        text = (chunk.text or "").lower()
        if any(domain == d or domain.endswith("." + d) for d in self.low_quality_domains):
            return True
        if any(k in url for k in self.low_quality_url_keywords):
            return True
        if text.count("flashcards") >= 1 or text.count("quizlet") >= 1:
            return True
        if "annotated bibliography" in text and "etiquette" not in text and "social norm" not in text:
            return True
        if "expert-verified" in text and "etiquette" not in text and "policy" not in text:
            return True
        return False

    def _filter_candidates(self, candidates: List[EvidenceChunk]) -> Tuple[List[EvidenceChunk], Dict[str, int]]:
        filtered: List[EvidenceChunk] = []
        seen = set()
        dropped_low_quality = 0
        dropped_duplicate = 0
        for c in candidates:
            if self._is_low_quality_chunk(c):
                dropped_low_quality += 1
                continue
            key = re.sub(r"\s+", " ", (c.text or "").lower())[:220]
            if key in seen:
                dropped_duplicate += 1
                continue
            seen.add(key)
            filtered.append(c)
        stats = {
            "dropped_low_quality": int(dropped_low_quality),
            "dropped_duplicate": int(dropped_duplicate),
        }
        return filtered, stats

    def _organize_evidence(
        self,
        question: str,
        choices: List[str],
        selected: List[EvidenceChunk],
        search_plan: Optional[Dict[str, Any]] = None,
        item_id: str = "",
    ) -> Tuple[List[str], Dict[str, Any]]:
        if not selected:
            return [], {"items": [], "selected_ids": [], "use_evidence": False, "overall_reason": "no_evidence"}
        if not self.enable_evidence_organization:
            notes = [f"[e{i+1}] {self._clip_text(c.text, limit=360)}" for i, c in enumerate(selected[: self.summary_max_items])]
            return notes, {
                "items": [],
                "selected_ids": [f"e{i+1}" for i in range(len(notes))],
                "evidence_notes": notes,
                "use_evidence": True,
                "overall_reason": "organization_disabled",
            }

        evidence_block = "\n\n".join(
            [
                f"[e{i+1}] source={c.domain} title={self._clip_text(c.title, 120)} text={self._clip_text(c.text, 900)}"
                for i, c in enumerate(selected[: self.summary_max_items])
            ]
        )
        sp = search_plan or {}
        plan_block_parts: List[str] = []
        if sp.get("information_need"):
            plan_block_parts.append(f"Information need: {sp.get('information_need')}")
        if sp.get("ambiguities"):
            plan_block_parts.append("Ambiguities: " + " | ".join(str(x) for x in (sp.get("ambiguities") or [])[:4]))
        if sp.get("search_axes"):
            plan_block_parts.append("Search axes: " + " | ".join(str(x) for x in (sp.get("search_axes") or [])[:4]))
        if sp.get("evidence_requirements"):
            plan_block_parts.append(
                "Evidence requirements: " + " | ".join(str(x) for x in (sp.get("evidence_requirements") or [])[:4])
            )
        plan_block = "\n".join(plan_block_parts).strip()
        prompt = (
            "You are an evidence analyst for a retrieval-augmented QA system.\n"
            "Your job is to examine retrieved web evidence and decide which pieces are genuinely useful for answering the question.\n"
            "Be strict.\n"
            "Do not keep evidence that is off-topic, generic, repetitive, low-quality, clearly copied from homework/help sites, weakly related but not decision-relevant, or likely to introduce stereotype-heavy or noisy associations without direct support.\n"
            "Prefer evidence that is specific to the question, directly relevant to the decision to be made, concrete rather than vague, from credible sources, and useful for distinguishing between the answer options.\n"
            "Return strict JSON only.\n"
            "Use this exact schema:\n"
            "{\n"
            '  "items": [{"evidence_id": "e1", "relevance_score": 0, "credibility_score": 0, "usefulness_score": 0, "keep": false, "reason": "..."}],\n'
            '  "selected_ids": ["e1"],\n'
            '  "evidence_notes": ["[e1] ..."],\n'
            '  "use_evidence": true,\n'
            '  "overall_reason": "..."\n'
            "}\n\n"
            f"Question: {question}\nChoices: {' | '.join(choices)}\n\n"
            f"{plan_block}\n\n"
            f"Evidence:\n{evidence_block}"
        )
        raw = self.llm.generate(
            "You compress evidence into concise relevant notes.",
            prompt,
            trace_meta={"stage": "search_evidence_organize", "item_id": item_id},
        )
        obj = self._extract_json_obj(raw)
        notes_raw = obj.get("evidence_notes", [])
        if not isinstance(notes_raw, list):
            notes_raw = []
        cleaned: List[str] = []
        for x in notes_raw:
            x = re.sub(r"\s+", " ", str(x or "")).strip()
            if not re.match(r"^\[e\d+\]", x):
                continue
            cleaned.append(x)
            if len(cleaned) >= self.summary_max_items:
                break
        if cleaned:
            obj["evidence_notes"] = cleaned
            return cleaned, obj
        fallback = [f"[e{i+1}] {self._clip_text(c.text, limit=360)}" for i, c in enumerate(selected[: self.summary_max_items])]
        obj["evidence_notes"] = fallback
        if "use_evidence" not in obj:
            obj["use_evidence"] = True
        if "overall_reason" not in obj:
            obj["overall_reason"] = "organization_fallback"
        return fallback, obj

    def _should_use_evidence(
        self,
        question: str,
        choices: List[str],
        selected: List[EvidenceChunk],
        organized_evidence: List[str],
        search_plan: Optional[Dict[str, Any]] = None,
        organization_trace: Optional[Dict[str, Any]] = None,
        item_id: str = "",
    ) -> Tuple[bool, str]:
        if not selected or not organized_evidence:
            return False, "no_evidence"
        top_score = float(selected[0].score)
        if top_score < self.min_evidence_score:
            return False, "low_score"
        if organization_trace:
            org_use = self._coerce_bool(organization_trace.get("use_evidence"), default=True)
            if not org_use:
                return False, str(organization_trace.get("overall_reason", "") or "organization_rejected").strip()
        if not self.enable_evidence_gate:
            return True, ""

        info_block = "\n".join(organized_evidence[: self.summary_max_items])
        sp = search_plan or {}
        plan_block_parts: List[str] = []
        if sp.get("information_need"):
            plan_block_parts.append(f"Information need: {sp.get('information_need')}")
        if sp.get("evidence_requirements"):
            plan_block_parts.append(
                "Evidence requirements: " + " | ".join(str(x) for x in (sp.get("evidence_requirements") or [])[:4])
            )
        if sp.get("ambiguities"):
            plan_block_parts.append("Ambiguities: " + " | ".join(str(x) for x in (sp.get("ambiguities") or [])[:4]))
        plan_block = "\n".join(plan_block_parts).strip()
        prompt = (
            "You are a strict evidence gatekeeper for a retrieval-augmented QA system.\n"
            "Decide whether the retrieved information is strong enough to be used in the final answer.\n"
            "Use the evidence only if it is specific, on-topic, and materially helpful for distinguishing between answer options.\n"
            "Reject the evidence if it is weak, noisy, generic, conflicting, stereotype-heavy without direct support, or only loosely related.\n"
            "Return strict JSON only with keys use_evidence, reason, confidence.\n\n"
            f"Question: {question}\nChoices: {' | '.join(choices)}\n\n"
            f"{plan_block}\n\n"
            f"Retrieved information:\n{info_block}"
        )
        raw = self.llm.generate(
            "You are a strict evidence gatekeeper.",
            prompt,
            trace_meta={"stage": "search_evidence_gate", "item_id": item_id},
        )
        obj = self._extract_json_obj(raw)
        use_evidence = self._coerce_bool(obj.get("use_evidence"), default=False)
        reason = str(obj.get("reason", "") or obj.get("overall_reason", "") or "").strip()
        return use_evidence, reason or ("accepted" if use_evidence else "llm_gate_rejected")

    def _select_chunks(
        self,
        question: str,
        candidates: List[EvidenceChunk],
        item_id: str = "",
    ) -> List[EvidenceChunk]:
        ranked = self._lexical_rank(question, candidates)
        ranked = self._embedding_prerank(question, ranked, item_id=item_id)
        ranked = self._semantic_rerank(question, ranked)
        ranked = self._llm_relevance_boost(question, ranked, item_id=item_id)
        ranked = self._apply_domain_priors(ranked)
        return self._topk_diverse_by_url(ranked, self.keep_top_k)

    def _run_retrieval_round(
        self,
        question: str,
        queries: List[str],
        item_id: str = "",
        retry: bool = False,
    ) -> Dict[str, Any]:
        all_hits = []
        search_events: List[Dict[str, Any]] = []
        for q in queries:
            q_hits = self.web.search(q, top_n=self.search_top_n)
            all_hits.extend(q_hits)
            event = self.web.last_search_event()
            if event:
                e2 = dict(event)
                if retry:
                    e2["retry"] = True
                search_events.append(e2)

        dedup_hits = self.web.dedupe_hits(all_hits, keep_per_domain=self.keep_per_domain)
        raw_candidates = self.web.build_candidate_chunks(dedup_hits, max_pages=self.max_pages)
        candidates, filter_stats = self._filter_candidates(raw_candidates)
        selected = self._select_chunks(question, candidates, item_id=item_id)
        top_score = float(selected[0].score) if selected else 0.0
        return {
            "queries": list(queries),
            "all_hits": all_hits,
            "search_events": search_events,
            "dedup_hits": dedup_hits,
            "raw_candidates": raw_candidates,
            "candidates": candidates,
            "filter_stats": filter_stats,
            "selected": selected,
            "top_score": top_score,
        }

    def _retrieve_live(self, item: Dict[str, Any]) -> Tuple[List[EvidenceChunk], Dict[str, Any]]:
        question = item["question"]
        item_id = str(item.get("id", ""))
        search_plan = self._build_search_plan(item, item_id=item_id)
        queries = list(search_plan.get("queries", []) or [])
        if not queries:
            queries = [question]

        round_1 = self._run_retrieval_round(question, queries, item_id=item_id, retry=False)
        final_round = round_1
        retry_attempted = False
        retry_queries: List[str] = []
        retry_applied = False
        retry_events: List[Dict[str, Any]] = []

        should_retry = (
            self.enable_query_feedback_retry
            and self.query_feedback_max_retry > 0
            and (
                len(round_1["candidates"]) == 0
                or len(round_1["selected"]) == 0
                or float(round_1.get("top_score", 0.0) or 0.0) < self.query_retry_min_top_score
            )
        )
        if should_retry:
            retry_attempted = True
            retry_queries = self._feedback_retry_queries(
                item=item,
                prev_queries=list(queries),
                selected=list(round_1.get("selected", []) or []),
                item_id=item_id,
            )
            if retry_queries:
                round_2 = self._run_retrieval_round(
                    question,
                    retry_queries,
                    item_id=item_id,
                    retry=True,
                )
                retry_events = list(round_2.get("search_events", []) or [])
                improved = False
                if len(round_1["selected"]) == 0 and len(round_2["selected"]) > 0:
                    improved = True
                elif len(round_2["selected"]) > len(round_1["selected"]):
                    improved = True
                elif float(round_2.get("top_score", 0.0) or 0.0) > float(round_1.get("top_score", 0.0) or 0.0) + 1e-6:
                    improved = True
                if improved:
                    final_round = round_2
                    queries = list(retry_queries)
                    retry_applied = True
        if retry_attempted:
            search_plan["retry"] = {
                "attempted": True,
                "applied": retry_applied,
                "queries": list(retry_queries),
            }

        all_hits = list(final_round["all_hits"])
        dedup_hits = list(final_round["dedup_hits"])
        raw_candidates = list(final_round["raw_candidates"])
        candidates = list(final_round["candidates"])
        filter_stats = dict(final_round["filter_stats"])
        selected = list(final_round["selected"])
        search_events = list(final_round["search_events"])
        if retry_events and not retry_applied:
            search_events.extend(retry_events)

        trace: Dict[str, Any] = {
            "pipeline_variant": "general",
            "query_source": "live",
            "search_plan": search_plan,
            "queries": queries,
            "retrieved_hits": len(all_hits),
            "dedup_hits": len(dedup_hits),
            "candidate_chunks": len(candidates),
            "raw_candidate_chunks": len(raw_candidates),
            "filter_stats": filter_stats,
            "selected_evidence": [_chunk_to_dict(s) for s in selected],
            "embedding_preranker": {
                "backend": self.embedding_preranker,
                "model": self.embedding_model,
                "top_m": self.embedding_preranker_top_m,
                "weight": self.embedding_preranker_weight,
            },
            "semantic_reranker": self.semantic_reranker.status(),
            "search_events": search_events,
            "query_retry_attempted": retry_attempted,
            "query_retry_applied": retry_applied,
            "query_retry_queries": retry_queries,
        }
        if self.include_candidate_details:
            trace["raw_candidate_evidence"] = [_chunk_to_dict(c) for c in raw_candidates]
            trace["candidate_evidence"] = [_chunk_to_dict(c) for c in candidates]
        return selected, trace

    def _retrieve_from_cache(self, item: Dict[str, Any], cached: Dict[str, Any]) -> Tuple[List[EvidenceChunk], Dict[str, Any]]:
        question = item["question"]
        item_id = str(item.get("id", ""))

        candidate_dicts = cached.get("raw_candidate_evidence", [])
        if not candidate_dicts:
            candidate_dicts = cached.get("candidate_evidence", [])
        selected_dicts = cached.get("selected_evidence", [])
        cached_search_plan = cached.get("search_plan", {}) or {}

        raw_candidates = [_dict_to_chunk(x) for x in candidate_dicts if isinstance(x, dict)]
        candidates, filter_stats = self._filter_candidates(raw_candidates)
        if candidates:
            selected = self._select_chunks(question, candidates, item_id=item_id)
            selected_dicts = [_chunk_to_dict(s) for s in selected]
        else:
            selected = [_dict_to_chunk(x) for x in selected_dicts if isinstance(x, dict)]

        trace = {
            "pipeline_variant": "general",
            "query_source": "cache",
            "cache_hit": True,
            "search_plan": cached_search_plan,
            "queries": cached.get("queries", [question]),
            "retrieved_hits": int(cached.get("retrieved_hits", 0) or 0),
            "dedup_hits": int(cached.get("dedup_hits", 0) or 0),
            "candidate_chunks": len(candidates) if candidates else int(cached.get("candidate_chunks", 0) or 0),
            "raw_candidate_chunks": len(raw_candidates) if raw_candidates else int(cached.get("raw_candidate_chunks", 0) or 0),
            "filter_stats": filter_stats,
            "selected_evidence": selected_dicts,
            "embedding_preranker": {
                "backend": self.embedding_preranker,
                "model": self.embedding_model,
                "top_m": self.embedding_preranker_top_m,
                "weight": self.embedding_preranker_weight,
            },
            "semantic_reranker": self.semantic_reranker.status(),
            "search_events": cached.get("search_events", []),
        }
        if self.include_candidate_details and candidates:
            trace["candidate_evidence"] = [_chunk_to_dict(c) for c in candidates]
        return selected, trace

    def prepare_evidence(self, item: Dict[str, Any]) -> Tuple[List[EvidenceChunk], Dict[str, Any]]:
        item_id = str(item.get("id", "")).strip()
        if item_id and item_id in self.cache_by_id:
            return self._retrieve_from_cache(item, self.cache_by_id[item_id])

        if self.use_cache_only:
            return [], {
                "pipeline_variant": "general",
                "query_source": "cache",
                "cache_hit": False,
                "search_plan": {},
                "queries": [item["question"]],
                "retrieved_hits": 0,
                "dedup_hits": 0,
                "candidate_chunks": 0,
                "raw_candidate_chunks": 0,
                "filter_stats": {"dropped_low_quality": 0, "dropped_duplicate": 0},
                "selected_evidence": [],
                "embedding_preranker": {
                    "backend": self.embedding_preranker,
                    "model": self.embedding_model,
                    "top_m": self.embedding_preranker_top_m,
                    "weight": self.embedding_preranker_weight,
                },
                "semantic_reranker": self.semantic_reranker.status(),
                "search_events": [],
            }

        return self._retrieve_live(item)

    def predict(self, item: Dict) -> Tuple[str, List[str], Dict, str]:
        selected, trace = self.prepare_evidence(item)
        raw_evidence = [s.text for s in selected]
        search_plan = trace.get("search_plan", {}) or {}
        choices = item.get("choices", []) or []
        is_mcq = bool(choices)
        organized_evidence, organization_trace = self._organize_evidence(
            item["question"],
            choices,
            selected,
            search_plan=search_plan,
            item_id=str(item.get("id", "")),
        )
        top_score = float(selected[0].score) if selected else 0.0

        use_evidence, gate_reason = self._should_use_evidence(
            item["question"],
            choices,
            selected,
            organized_evidence,
            search_plan=search_plan,
            organization_trace=organization_trace,
            item_id=str(item.get("id", "")),
        )

        if use_evidence:
            prompt = (
                build_grounded_answer_prompt(item["question"], choices, organized_evidence)
                if is_mcq
                else build_grounded_short_answer_prompt(item["question"], organized_evidence)
            )
            raw = self.llm.generate(
                "You are a careful assistant that relies on provided evidence.",
                prompt,
                trace_meta={"stage": "search_answer_augmented", "item_id": str(item.get("id", ""))},
            )
        else:
            prompt = format_mcq_prompt(item["question"], choices) if is_mcq else format_short_answer_prompt(item["question"])
            raw = self.llm.generate(
                "You are a careful assistant.",
                prompt,
                trace_meta={"stage": "search_answer_fallback", "item_id": str(item.get("id", ""))},
            )

        trace["used_evidence"] = use_evidence
        trace["final_stage"] = "search_answer_augmented" if use_evidence else "search_answer_fallback"
        trace["runtime_status"] = self.runtime_status()
        trace["organized_evidence"] = organized_evidence
        trace["evidence_organization"] = organization_trace
        trace["top_selected_score"] = round(top_score, 6)
        if gate_reason:
            trace["evidence_gate_reason"] = gate_reason
        pred = _manual_verbalize(raw, choices) if is_mcq else _normalize_short_answer(raw)
        trace["raw_output"] = raw
        return pred, raw_evidence, trace, raw


class BiasAwareSearchPipeline(GeneralSearchPipeline):
    """
    Risk-aware, multi-route search pipeline for bias-sensitive retrieval.
    Reuses the existing web retrieval stack but replaces query planning,
    route weighting, evidence gating, and answer policy.
    """

    def __init__(
        self,
        *args: Any,
        risk_medium_threshold: float = 1.5,
        risk_high_threshold: float = 3.5,
        bias_query_max_n: int = 4,
        enable_balance_gate: bool = True,
        route_bonus_primary: float = 0.02,
        route_bonus_claim_testing: float = 0.03,
        route_bonus_counter_evidence: float = 0.05,
        route_bonus_confounder_context: float = 0.04,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.risk_medium_threshold = float(risk_medium_threshold)
        self.risk_high_threshold = float(risk_high_threshold)
        self.bias_query_max_n = max(1, int(bias_query_max_n))
        self.enable_balance_gate = bool(enable_balance_gate)
        self.route_bonus_primary = float(route_bonus_primary)
        self.route_bonus_claim_testing = float(route_bonus_claim_testing)
        self.route_bonus_counter_evidence = float(route_bonus_counter_evidence)
        self.route_bonus_confounder_context = float(route_bonus_confounder_context)

    def runtime_status(self) -> Dict[str, Any]:
        out = super().runtime_status()
        out["pipeline_type"] = "bias_aware"
        out["bias_settings"] = {
            "risk_medium_threshold": self.risk_medium_threshold,
            "risk_high_threshold": self.risk_high_threshold,
            "bias_query_max_n": self.bias_query_max_n,
            "enable_balance_gate": self.enable_balance_gate,
            "route_bonus_primary": self.route_bonus_primary,
            "route_bonus_claim_testing": self.route_bonus_claim_testing,
            "route_bonus_counter_evidence": self.route_bonus_counter_evidence,
            "route_bonus_confounder_context": self.route_bonus_confounder_context,
        }
        return out

    @staticmethod
    def _safe_json_list(value: Any) -> List[str]:
        if isinstance(value, list):
            return [str(x).strip() for x in value if str(x).strip()]
        if isinstance(value, str) and value.strip():
            return [value.strip()]
        return []

    def _risk_level_from_score(self, score: float) -> str:
        if score >= self.risk_high_threshold:
            return "high"
        if score >= self.risk_medium_threshold:
            return "medium"
        return "low"

    def _infer_intent_type(self, question: str, choices: List[str], dataset: str) -> str:
        q = question.lower()
        if _is_stereotype_label_task(choices) or dataset in {"seegull", "bbq", "socialstigmaqa"}:
            return "stereotype_testing"
        if any(x in q for x in ["should ", "acceptable", "appropriate", "polite", "rude", "okay to", "is it okay"]):
            return "normative"
        if any(x in q for x in ["cause", "causes", "because", "why are", "why do", "leads to", "result in"]):
            return "causal"
        if any(x in q for x in ["more likely", "less likely", "better than", "worse than", "compared with", "compared to"]):
            return "comparative"
        if any(x in q for x in ["what", "which", "who", "when", "where", "how many", "how much"]):
            return "factual"
        return "ambiguous"

    def _rule_based_risk_analysis(self, item: Dict[str, Any]) -> Dict[str, Any]:
        question = str(item.get("question", "") or "")
        choices = [str(x) for x in (item.get("choices", []) or [])]
        dataset = self._dataset_name(item)
        q = question.lower()
        sensitivity_patterns = {
            "gender": [r"\bmen\b", r"\bwomen\b", r"\bfemale\b", r"\bmale\b", r"\bgender\b", r"\bgirls?\b", r"\bboys?\b"],
            "race": [r"\brace\b", r"\bracial\b", r"\bblack\b", r"\bwhite\b", r"\basian\b", r"\blatino\b", r"\bethnic"],
            "nationality": [r"\bimmigrant", r"\bmigrants?\b", r"\bnationality\b", r"\b[a-z]+ian(s)?\b", r"\b[a-z]+ese\b", r"\b[a-z]+i\b"],
            "religion": [r"\bmuslim\b", r"\bchristian\b", r"\bjewish\b", r"\bhindu\b", r"\bbuddhist\b", r"\breligion\b"],
            "sexual_orientation": [r"\bgay\b", r"\blesbian\b", r"\bqueer\b", r"\bsexual orientation\b", r"\blgbt"],
            "age": [r"\bold people\b", r"\byoung people\b", r"\bolder adults?\b", r"\belderly\b", r"\bteenagers?\b"],
            "disability": [r"\bdisabled\b", r"\bdisability\b", r"\bautistic\b", r"\bblind\b", r"\bdeaf\b"],
            "immigration": [r"\bimmigrant\b", r"\brefugee\b", r"\basylum seeker\b", r"\bmigration\b"],
        }
        sensitivity_type: List[str] = []
        for key, patterns in sensitivity_patterns.items():
            if any(re.search(p, q) for p in patterns):
                sensitivity_type.append(key)
        stereotype_patterns = [
            r"\bnaturally\b",
            r"\binnately\b",
            r"\binherently\b",
            r"\bborn to\b",
            r"\bwhy are\b",
            r"\bwhy do\b",
            r"\btend to be\b",
            r"\busually are\b",
            r"\bmore lazy\b",
            r"\bmore violent\b",
            r"\bmore intelligent\b",
            r"\bless intelligent\b",
            r"\bbetter at\b",
            r"\bworse at\b",
        ]
        causal_patterns = [
            r"\bcause\b",
            r"\bcauses\b",
            r"\bcausal\b",
            r"\bgenetic\b",
            r"\bbiological\b",
            r"\bhardwired\b",
            r"\binnate\b",
        ]
        normative_patterns = [
            r"\bshould\b",
            r"\bacceptable\b",
            r"\bappropriate\b",
            r"\bpolite\b",
            r"\brude\b",
            r"\bokay\b",
        ]
        stereotype_framing = 1 if any(re.search(p, q) for p in stereotype_patterns) else 0
        causal_generalization = 1 if any(re.search(p, q) for p in causal_patterns) else 0
        normative_harm = 1 if any(re.search(p, q) for p in normative_patterns) and sensitivity_type else 0
        group_sensitive = 1 if sensitivity_type else 0
        essentialization = 1 if stereotype_framing or causal_generalization else 0
        intent_type = self._infer_intent_type(question, choices, dataset)
        if intent_type == "stereotype_testing":
            stereotype_framing = max(stereotype_framing, 1)
            group_sensitive = max(group_sensitive, 1)
        clearly_factual = 1 if intent_type == "factual" and not stereotype_framing and not causal_generalization else 0
        risk_score = (
            1.5 * group_sensitive
            + 2.0 * stereotype_framing
            + 2.0 * essentialization
            + 1.5 * causal_generalization
            + 1.0 * normative_harm
            - 0.5 * clearly_factual
        )
        risk_factors: List[str] = []
        if group_sensitive:
            risk_factors.append("sensitive_group")
        if stereotype_framing:
            risk_factors.append("stereotype_framing")
        if essentialization:
            risk_factors.append("essentialization")
        if causal_generalization:
            risk_factors.append("causal_generalization")
        if normative_harm:
            risk_factors.append("normative_risk")
        if dataset in {"seegull", "bbq", "socialstigmaqa"}:
            risk_factors.append("bias_sensitive_benchmark")
        return {
            "sensitive": bool(group_sensitive),
            "sensitivity_type": sensitivity_type,
            "risk_score": round(float(risk_score), 3),
            "risk_level": self._risk_level_from_score(risk_score),
            "intent_type": intent_type,
            "risk_factors": risk_factors,
            "group_sensitive": group_sensitive,
            "stereotype_framing": stereotype_framing,
            "essentialization": essentialization,
            "causal_generalization": causal_generalization,
            "normative_harm": normative_harm,
        }

    def _default_source_preferences(self, analysis: Dict[str, Any]) -> List[str]:
        if str(analysis.get("risk_level", "low")) == "high":
            return [
                "peer-reviewed research",
                "official statistics",
                "government or NGO reports",
                "systematic reviews",
            ]
        return [
            "reputable reference sources",
            "official institutions",
        ]

    def _build_bias_rewrite_prompt(self, base: str, item: Dict[str, Any], analysis: Dict[str, Any]) -> str:
        choices = [str(x) for x in (item.get("choices", []) or [])]
        dataset = self._dataset_name(item)
        task_family = self._task_family(item)
        choices_block = f"\nChoices: {' | '.join(choices)}" if choices else ""
        return (
            "You are a bias-aware query rewriter for an inclusive retrieval-augmented system.\n"
            "Convert the user question into retrieval queries that preserve the real information need while minimizing stereotype amplification and one-sided evidence retrieval.\n"
            "Rules:\n"
            "1. Preserve legitimate group constraints when needed.\n"
            "2. Do not transform uncertainty into factual certainty.\n"
            "3. Do not restate defamatory or stereotypical claims as if they were established facts.\n"
            "4. Prefer evidence-seeking formulations over stereotype-seeking formulations.\n"
            "5. For sensitive or medium/high-risk questions, generate a balanced query set including primary, claim-testing, counter-evidence, and confounder/context queries when useful.\n"
            "6. Counter-evidence and confounder queries should prefer variation, structural factors, measurement limits, and alternative explanations.\n"
            "7. Keep queries concise and search-engine friendly.\n"
            "8. Output valid JSON only.\n\n"
            "Return JSON with schema:\n"
            "{\n"
            '  "sensitive": boolean,\n'
            '  "sensitivity_type": [string],\n'
            '  "risk_level": "low|medium|high",\n'
            '  "risk_score": number,\n'
            '  "intent_type": "factual|causal|comparative|normative|stereotype_testing|ambiguous",\n'
            '  "rewrite_strategy": "preserve|neutralize|decompose|balance",\n'
            '  "core_information_need": string,\n'
            '  "queries": [\n'
            '    {"type": "primary|claim_testing|counter_evidence|confounder_context", "query": string}\n'
            "  ],\n"
            '  "source_preferences": [string],\n'
            '  "retrieval_notes": string,\n'
            '  "answer_policy_hint": "answer_normally|answer_with_qualification|avoid_group_generalization"\n'
            "}\n\n"
            f"Dataset: {dataset or 'unknown'}\n"
            f"Task family: {task_family}\n"
            f"Rule-based risk analysis: {json.dumps(analysis, ensure_ascii=False)}\n"
            f"Question: {base}{choices_block}"
        )

    @staticmethod
    def _normalize_query_type(value: str) -> str:
        low = str(value or "").strip().lower()
        if low in {"neutral_factual", "neutral", "factual", "primary"}:
            return "primary"
        if low in {"claim_testing", "claim-test", "claim"}:
            return "claim_testing"
        if low in {"counter_evidence", "counter", "counterevidence"}:
            return "counter_evidence"
        if low in {"confounder_context", "confounder", "context"}:
            return "confounder_context"
        return "primary"

    def _extract_query_objects(self, value: Any) -> List[Dict[str, str]]:
        out: List[Dict[str, str]] = []
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    q = self._normalize_query_text(str(item.get("query", "") or ""))
                    if not q:
                        continue
                    out.append({
                        "type": self._normalize_query_type(str(item.get("type", "primary") or "primary")),
                        "query": q,
                    })
                elif isinstance(item, str):
                    q = self._normalize_query_text(item)
                    if q:
                        out.append({"type": "primary", "query": q})
        return out

    def _build_default_bias_search_plan(
        self,
        item: Dict[str, Any],
        item_id: str,
        analysis: Dict[str, Any],
        raw_output: str = "",
    ) -> Dict[str, Any]:
        fallback = GeneralSearchPipeline._build_search_plan(self, item, item_id=item_id)
        query_objects = [{"type": "primary", "query": q} for q in (fallback.get("queries", []) or [])]
        return {
            "queries": [x["query"] for x in query_objects],
            "query_plan": [
                {
                    "query": x["query"],
                    "intent": str(analysis.get("intent_type", "ambiguous")),
                    "purpose": x["type"],
                }
                for x in query_objects
            ],
            "query_generation_mode": str(fallback.get("query_generation_mode", "fallback_general")),
            "rewrite_policy": str(fallback.get("rewrite_policy", "none")),
            "rewrite_raw_output": str(fallback.get("rewrite_raw_output", "") or raw_output),
            "risk_analysis": analysis,
            "sensitive": bool(analysis.get("sensitive", False)),
            "sensitivity_type": list(analysis.get("sensitivity_type", []) or []),
            "risk_level": str(analysis.get("risk_level", "low")),
            "risk_score": float(analysis.get("risk_score", 0.0) or 0.0),
            "intent_type": str(analysis.get("intent_type", "ambiguous")),
            "rewrite_strategy": "preserve" if not analysis.get("sensitive") else "neutralize",
            "core_information_need": str(item.get("question", "") or ""),
            "source_preferences": self._default_source_preferences(analysis),
            "retrieval_notes": "fallback_general_planner",
            "answer_policy_hint": (
                "answer_with_qualification" if str(analysis.get("risk_level", "low")) in {"medium", "high"} else "answer_normally"
            ),
            "task_family": self._task_family(item),
        }

    def _build_search_plan(self, item: Dict[str, Any], item_id: str = "") -> Dict[str, Any]:
        question = str(item.get("question", "") or "")
        base = re.sub(r"\s+", " ", question).strip()
        if not base:
            return {
                "queries": [],
                "query_plan": [],
                "query_generation_mode": "empty",
                "rewrite_policy": "none",
                "rewrite_raw_output": "",
                "risk_analysis": {},
                "sensitive": False,
                "sensitivity_type": [],
                "risk_level": "low",
                "risk_score": 0.0,
                "intent_type": "ambiguous",
                "rewrite_strategy": "preserve",
                "core_information_need": "",
                "source_preferences": [],
                "retrieval_notes": "",
                "answer_policy_hint": "answer_normally",
                "task_family": self._task_family(item),
            }
        analysis = self._rule_based_risk_analysis(item)
        if not analysis.get("sensitive") and str(analysis.get("risk_level", "low")) == "low":
            return self._build_default_bias_search_plan(item, item_id=item_id, analysis=analysis)
        prompt = self._build_bias_rewrite_prompt(base, item, analysis)
        raw = self.llm.generate(
            "You are a bias-aware query planner.",
            prompt,
            trace_meta={"stage": "search_bias_aware_rewrite", "item_id": item_id},
        )
        obj = self._extract_json_obj(raw)
        query_objects = self._extract_query_objects(obj.get("queries", []))
        cleaned: List[Dict[str, str]] = []
        seen = set()
        for qo in query_objects:
            q = qo["query"]
            q_list = self._clean_queries([q], max_n=1)
            if not q_list:
                continue
            q2 = q_list[0]
            key = (qo["type"], q2)
            if key in seen:
                continue
            seen.add(key)
            cleaned.append({"type": qo["type"], "query": q2})
            if len(cleaned) >= self.bias_query_max_n:
                break
        if not cleaned:
            return self._build_default_bias_search_plan(item, item_id=item_id, analysis=analysis, raw_output=raw)
        result_analysis = dict(analysis)
        result_analysis["sensitive"] = self._coerce_bool(obj.get("sensitive"), default=bool(analysis.get("sensitive", False)))
        result_analysis["sensitivity_type"] = self._safe_json_list(obj.get("sensitivity_type")) or list(analysis.get("sensitivity_type", []) or [])
        score_obj = obj.get("risk_score", analysis.get("risk_score", 0.0))
        try:
            result_analysis["risk_score"] = round(float(score_obj), 3)
        except Exception:
            result_analysis["risk_score"] = float(analysis.get("risk_score", 0.0) or 0.0)
        risk_level = str(obj.get("risk_level", "") or "").strip().lower()
        if risk_level not in {"low", "medium", "high"}:
            risk_level = self._risk_level_from_score(float(result_analysis.get("risk_score", 0.0) or 0.0))
        result_analysis["risk_level"] = risk_level
        intent_type = str(obj.get("intent_type", "") or "").strip().lower()
        if intent_type not in {"factual", "causal", "comparative", "normative", "stereotype_testing", "ambiguous"}:
            intent_type = str(analysis.get("intent_type", "ambiguous"))
        result_analysis["intent_type"] = intent_type
        rewrite_strategy = str(obj.get("rewrite_strategy", "") or "").strip().lower()
        if rewrite_strategy not in {"preserve", "neutralize", "decompose", "balance"}:
            rewrite_strategy = "balance" if risk_level in {"medium", "high"} else "preserve"
        answer_policy_hint = str(obj.get("answer_policy_hint", "") or "").strip().lower()
        if answer_policy_hint not in {"answer_normally", "answer_with_qualification", "avoid_group_generalization"}:
            answer_policy_hint = "answer_with_qualification" if risk_level in {"medium", "high"} else "answer_normally"
        return {
            "queries": [x["query"] for x in cleaned],
            "query_plan": [
                {
                    "query": x["query"],
                    "intent": intent_type,
                    "purpose": x["type"],
                }
                for x in cleaned
            ],
            "query_generation_mode": "bias_aware_structured_rewrite",
            "rewrite_policy": "bias_aware",
            "rewrite_raw_output": raw,
            "risk_analysis": result_analysis,
            "sensitive": bool(result_analysis.get("sensitive", False)),
            "sensitivity_type": list(result_analysis.get("sensitivity_type", []) or []),
            "risk_level": risk_level,
            "risk_score": float(result_analysis.get("risk_score", 0.0) or 0.0),
            "intent_type": intent_type,
            "rewrite_strategy": rewrite_strategy,
            "core_information_need": str(obj.get("core_information_need", "") or base),
            "source_preferences": self._safe_json_list(obj.get("source_preferences")) or self._default_source_preferences(result_analysis),
            "retrieval_notes": str(obj.get("retrieval_notes", "") or "").strip(),
            "answer_policy_hint": answer_policy_hint,
            "task_family": self._task_family(item),
        }

    @staticmethod
    def _query_type_map(search_plan: Dict[str, Any]) -> Dict[str, str]:
        mapping: Dict[str, str] = {}
        for qp in (search_plan.get("query_plan", []) or []):
            if not isinstance(qp, dict):
                continue
            q = str(qp.get("query", "") or "").strip()
            if q and q not in mapping:
                mapping[q] = str(qp.get("purpose", "primary") or "primary").strip().lower()
        return mapping

    def _route_bonus(self, query_type: str, risk_level: str) -> float:
        qtype = self._normalize_query_type(query_type)
        if qtype == "claim_testing":
            bonus = self.route_bonus_claim_testing
        elif qtype == "counter_evidence":
            bonus = self.route_bonus_counter_evidence
        elif qtype == "confounder_context":
            bonus = self.route_bonus_confounder_context
        else:
            bonus = self.route_bonus_primary
        if risk_level == "low":
            bonus *= 0.5
        elif risk_level == "high" and qtype in {"counter_evidence", "confounder_context"}:
            bonus *= 1.25
        return bonus

    def _apply_bias_route_weights(
        self,
        chunks: List[EvidenceChunk],
        query_type_map: Dict[str, str],
        risk_level: str,
    ) -> List[EvidenceChunk]:
        if not chunks:
            return []
        out: List[EvidenceChunk] = []
        for c in chunks:
            qtype = query_type_map.get(c.query, "primary")
            bonus = self._route_bonus(qtype, risk_level)
            low = c.text.lower()
            if risk_level == "high" and qtype == "primary":
                if any(term in low for term in ["naturally", "innately", "genetic", "hardwired", "born to"]):
                    bonus -= 0.03
            out.append(
                EvidenceChunk(
                    query=c.query,
                    title=c.title,
                    url=c.url,
                    domain=c.domain,
                    text=c.text,
                    score=float(c.score) + bonus,
                )
            )
        return out

    def _selected_with_route_dicts(self, selected: List[EvidenceChunk], query_type_map: Dict[str, str]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for s in selected:
            obj = _chunk_to_dict(s)
            obj["query_type"] = query_type_map.get(s.query, "primary")
            out.append(obj)
        return out

    def _route_counts(self, selected: List[EvidenceChunk], query_type_map: Dict[str, str]) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for s in selected:
            qtype = query_type_map.get(s.query, "primary")
            counts[qtype] = counts.get(qtype, 0) + 1
        return counts

    def _route_summary(self, selected: List[EvidenceChunk], query_type_map: Dict[str, str]) -> str:
        counts = self._route_counts(selected, query_type_map)
        if not counts:
            return ""
        return ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))

    def _balanced_route_ok(self, selected: List[EvidenceChunk], query_type_map: Dict[str, str], risk_level: str) -> bool:
        counts = self._route_counts(selected, query_type_map)
        if risk_level == "high":
            return counts.get("counter_evidence", 0) > 0 or counts.get("confounder_context", 0) > 0
        if risk_level == "medium":
            return len(counts) >= 2
        return True

    def _rebalance_selected_chunks(
        self,
        selected: List[EvidenceChunk],
        candidates: List[EvidenceChunk],
        query_type_map: Dict[str, str],
        risk_level: str,
    ) -> List[EvidenceChunk]:
        if not selected or risk_level != "high":
            return selected
        if self._balanced_route_ok(selected, query_type_map, risk_level):
            return selected
        replacement = None
        for cand in candidates:
            qtype = query_type_map.get(cand.query, "primary")
            if qtype not in {"counter_evidence", "confounder_context"}:
                continue
            if any(cand.url == s.url for s in selected):
                continue
            replacement = cand
            break
        if replacement is None:
            return selected
        adjusted = list(selected[:-1]) + [replacement]
        adjusted.sort(key=lambda x: x.score, reverse=True)
        return self._topk_diverse_by_url(adjusted, self.keep_top_k)

    def _run_bias_retrieval_round(
        self,
        question: str,
        search_plan: Dict[str, Any],
        item_id: str = "",
        retry: bool = False,
    ) -> Dict[str, Any]:
        queries = list(search_plan.get("queries", []) or [])
        query_type_map = self._query_type_map(search_plan)
        risk_level = str(search_plan.get("risk_level", "low") or "low")
        all_hits: List[Any] = []
        search_events: List[Dict[str, Any]] = []
        for q in queries:
            q_hits = self.web.search(q, top_n=self.search_top_n)
            all_hits.extend(q_hits)
            event = self.web.last_search_event()
            if event:
                e2 = dict(event)
                if retry:
                    e2["retry"] = True
                e2["query_type"] = query_type_map.get(q, "primary")
                search_events.append(e2)
        dedup_hits = self.web.dedupe_hits(all_hits, keep_per_domain=self.keep_per_domain)
        raw_candidates = self.web.build_candidate_chunks(dedup_hits, max_pages=self.max_pages)
        candidates, filter_stats = self._filter_candidates(raw_candidates)
        weighted_candidates = self._apply_bias_route_weights(candidates, query_type_map, risk_level)
        selected = self._select_chunks(question, weighted_candidates, item_id=item_id)
        selected = self._rebalance_selected_chunks(selected, weighted_candidates, query_type_map, risk_level)
        top_score = float(selected[0].score) if selected else 0.0
        return {
            "queries": queries,
            "query_type_map": query_type_map,
            "all_hits": all_hits,
            "search_events": search_events,
            "dedup_hits": dedup_hits,
            "raw_candidates": raw_candidates,
            "candidates": weighted_candidates,
            "filter_stats": filter_stats,
            "selected": selected,
            "top_score": top_score,
        }

    def _retrieve_live(self, item: Dict[str, Any]) -> Tuple[List[EvidenceChunk], Dict[str, Any]]:
        question = item["question"]
        item_id = str(item.get("id", ""))
        search_plan = self._build_search_plan(item, item_id=item_id)
        queries = list(search_plan.get("queries", []) or []) or [question]
        search_plan["queries"] = queries
        round_1 = self._run_bias_retrieval_round(question, search_plan, item_id=item_id, retry=False)
        final_round = round_1
        retry_attempted = False
        retry_queries: List[str] = []
        retry_applied = False
        retry_events: List[Dict[str, Any]] = []
        should_retry = (
            self.enable_query_feedback_retry
            and self.query_feedback_max_retry > 0
            and (
                len(round_1["candidates"]) == 0
                or len(round_1["selected"]) == 0
                or float(round_1.get("top_score", 0.0) or 0.0) < self.query_retry_min_top_score
            )
        )
        if should_retry:
            retry_attempted = True
            retry_queries = self._feedback_retry_queries(
                item=item,
                prev_queries=list(queries),
                selected=list(round_1.get("selected", []) or []),
                item_id=item_id,
            )
            if retry_queries:
                retry_plan = dict(search_plan)
                retry_plan["queries"] = list(retry_queries)
                retry_plan["query_plan"] = [
                    {
                        "query": q,
                        "intent": str(search_plan.get("intent_type", "ambiguous") or "ambiguous"),
                        "purpose": "primary",
                    }
                    for q in retry_queries
                ]
                round_2 = self._run_bias_retrieval_round(question, retry_plan, item_id=item_id, retry=True)
                retry_events = list(round_2.get("search_events", []) or [])
                improved = False
                if len(round_1["selected"]) == 0 and len(round_2["selected"]) > 0:
                    improved = True
                elif len(round_2["selected"]) > len(round_1["selected"]):
                    improved = True
                elif float(round_2.get("top_score", 0.0) or 0.0) > float(round_1.get("top_score", 0.0) or 0.0) + 1e-6:
                    improved = True
                if improved:
                    final_round = round_2
                    search_plan = retry_plan
                    queries = list(retry_queries)
                    retry_applied = True
        if retry_attempted:
            search_plan["retry"] = {
                "attempted": True,
                "applied": retry_applied,
                "queries": list(retry_queries),
            }
        all_hits = list(final_round["all_hits"])
        dedup_hits = list(final_round["dedup_hits"])
        raw_candidates = list(final_round["raw_candidates"])
        candidates = list(final_round["candidates"])
        filter_stats = dict(final_round["filter_stats"])
        selected = list(final_round["selected"])
        search_events = list(final_round["search_events"])
        if retry_events and not retry_applied:
            search_events.extend(retry_events)
        query_type_map = dict(final_round.get("query_type_map", {}) or {})
        trace: Dict[str, Any] = {
            "pipeline_variant": "bias_aware",
            "query_source": "live",
            "search_plan": search_plan,
            "queries": queries,
            "retrieved_hits": len(all_hits),
            "dedup_hits": len(dedup_hits),
            "candidate_chunks": len(candidates),
            "raw_candidate_chunks": len(raw_candidates),
            "filter_stats": filter_stats,
            "selected_evidence": self._selected_with_route_dicts(selected, query_type_map),
            "route_counts": self._route_counts(selected, query_type_map),
            "route_summary": self._route_summary(selected, query_type_map),
            "embedding_preranker": {
                "backend": self.embedding_preranker,
                "model": self.embedding_model,
                "top_m": self.embedding_preranker_top_m,
                "weight": self.embedding_preranker_weight,
            },
            "semantic_reranker": self.semantic_reranker.status(),
            "search_events": search_events,
            "query_retry_attempted": retry_attempted,
            "query_retry_applied": retry_applied,
            "query_retry_queries": retry_queries,
        }
        if self.include_candidate_details:
            trace["raw_candidate_evidence"] = [_chunk_to_dict(c) for c in raw_candidates]
            trace["candidate_evidence"] = self._selected_with_route_dicts(candidates, query_type_map)
        return selected, trace

    def _retrieve_from_cache(self, item: Dict[str, Any], cached: Dict[str, Any]) -> Tuple[List[EvidenceChunk], Dict[str, Any]]:
        question = item["question"]
        item_id = str(item.get("id", ""))
        candidate_dicts = cached.get("raw_candidate_evidence", [])
        if not candidate_dicts:
            candidate_dicts = cached.get("candidate_evidence", [])
        selected_dicts = cached.get("selected_evidence", [])
        cached_search_plan = cached.get("search_plan", {}) or {}
        query_type_map = self._query_type_map(cached_search_plan)
        raw_candidates = [_dict_to_chunk(x) for x in candidate_dicts if isinstance(x, dict)]
        candidates, filter_stats = self._filter_candidates(raw_candidates)
        candidates = self._apply_bias_route_weights(candidates, query_type_map, str(cached_search_plan.get("risk_level", "low") or "low"))
        if candidates:
            selected = self._select_chunks(question, candidates, item_id=item_id)
            selected = self._rebalance_selected_chunks(
                selected,
                candidates,
                query_type_map,
                str(cached_search_plan.get("risk_level", "low") or "low"),
            )
            selected_dicts = self._selected_with_route_dicts(selected, query_type_map)
        else:
            selected = [_dict_to_chunk(x) for x in selected_dicts if isinstance(x, dict)]
        trace = {
            "pipeline_variant": "bias_aware",
            "query_source": "cache",
            "cache_hit": True,
            "search_plan": cached_search_plan,
            "queries": cached.get("queries", [question]),
            "retrieved_hits": int(cached.get("retrieved_hits", 0) or 0),
            "dedup_hits": int(cached.get("dedup_hits", 0) or 0),
            "candidate_chunks": len(candidates) if candidates else int(cached.get("candidate_chunks", 0) or 0),
            "raw_candidate_chunks": len(raw_candidates) if raw_candidates else int(cached.get("raw_candidate_chunks", 0) or 0),
            "filter_stats": filter_stats,
            "selected_evidence": selected_dicts,
            "route_counts": self._route_counts(selected, query_type_map),
            "route_summary": self._route_summary(selected, query_type_map),
            "embedding_preranker": {
                "backend": self.embedding_preranker,
                "model": self.embedding_model,
                "top_m": self.embedding_preranker_top_m,
                "weight": self.embedding_preranker_weight,
            },
            "semantic_reranker": self.semantic_reranker.status(),
            "search_events": cached.get("search_events", []),
        }
        if self.include_candidate_details and candidates:
            trace["candidate_evidence"] = self._selected_with_route_dicts(candidates, query_type_map)
        return selected, trace

    def _should_use_evidence(
        self,
        question: str,
        choices: List[str],
        selected: List[EvidenceChunk],
        organized_evidence: List[str],
        search_plan: Optional[Dict[str, Any]] = None,
        organization_trace: Optional[Dict[str, Any]] = None,
        item_id: str = "",
    ) -> Tuple[bool, str]:
        use_evidence, reason = super()._should_use_evidence(
            question,
            choices,
            selected,
            organized_evidence,
            search_plan=search_plan,
            organization_trace=organization_trace,
            item_id=item_id,
        )
        if not use_evidence or not self.enable_balance_gate:
            return use_evidence, reason
        sp = search_plan or {}
        risk_level = str(sp.get("risk_level", "low") or "low")
        if risk_level not in {"medium", "high"}:
            return use_evidence, reason
        qmap = self._query_type_map(sp)
        if not self._balanced_route_ok(selected, qmap, risk_level):
            return False, "balance_gate_insufficient_route_balance"
        return True, reason

    def _determine_answer_policy(
        self,
        search_plan: Dict[str, Any],
        selected: List[EvidenceChunk],
        use_evidence: bool,
    ) -> str:
        hint = str(search_plan.get("answer_policy_hint", "") or "").strip().lower()
        if hint not in {"answer_normally", "answer_with_qualification", "avoid_group_generalization"}:
            hint = ""
        risk_level = str(search_plan.get("risk_level", "low") or "low")
        qmap = self._query_type_map(search_plan)
        balanced = self._balanced_route_ok(selected, qmap, risk_level)
        if hint == "avoid_group_generalization":
            return hint
        if risk_level == "high":
            if use_evidence and balanced:
                return "answer_with_qualification"
            return "avoid_group_generalization"
        if risk_level == "medium":
            return "answer_with_qualification"
        return hint or "answer_normally"

    def predict(self, item: Dict) -> Tuple[str, List[str], Dict, str]:
        selected, trace = self.prepare_evidence(item)
        raw_evidence = [s.text for s in selected]
        search_plan = trace.get("search_plan", {}) or {}
        choices = item.get("choices", []) or []
        is_mcq = bool(choices)
        organized_evidence, organization_trace = self._organize_evidence(
            item["question"],
            choices,
            selected,
            search_plan=search_plan,
            item_id=str(item.get("id", "")),
        )
        top_score = float(selected[0].score) if selected else 0.0
        use_evidence, gate_reason = self._should_use_evidence(
            item["question"],
            choices,
            selected,
            organized_evidence,
            search_plan=search_plan,
            organization_trace=organization_trace,
            item_id=str(item.get("id", "")),
        )
        answer_policy = self._determine_answer_policy(search_plan, selected, use_evidence)
        route_summary = trace.get("route_summary", "") or self._route_summary(selected, self._query_type_map(search_plan))
        if use_evidence:
            if is_mcq:
                prompt = build_bias_aware_grounded_answer_prompt(
                    item["question"],
                    choices,
                    organized_evidence,
                    answer_policy=answer_policy,
                    plan_meta={
                        "risk_level": search_plan.get("risk_level", ""),
                        "intent_type": search_plan.get("intent_type", ""),
                        "route_summary": route_summary,
                    },
                )
            else:
                prompt = build_grounded_short_answer_prompt(item["question"], organized_evidence)
            raw = self.llm.generate(
                "You are a careful assistant that relies on provided evidence and avoids unsupported group generalization.",
                prompt,
                trace_meta={"stage": "search_answer_augmented", "item_id": str(item.get("id", ""))},
            )
        else:
            if is_mcq and answer_policy != "answer_normally":
                prompt = (
                    "Answer the multiple-choice question carefully.\n"
                    "Avoid unsupported group generalization, stereotype reinforcement, and causal overclaim.\n\n"
                    f"{format_mcq_prompt(item['question'], choices)}"
                )
            else:
                prompt = format_mcq_prompt(item["question"], choices) if is_mcq else format_short_answer_prompt(item["question"])
            raw = self.llm.generate(
                "You are a careful assistant.",
                prompt,
                trace_meta={"stage": "search_answer_fallback", "item_id": str(item.get("id", ""))},
            )
        trace["used_evidence"] = use_evidence
        trace["final_stage"] = "search_answer_augmented" if use_evidence else "search_answer_fallback"
        trace["runtime_status"] = self.runtime_status()
        trace["organized_evidence"] = organized_evidence
        trace["evidence_organization"] = organization_trace
        trace["top_selected_score"] = round(top_score, 6)
        trace["answer_policy"] = answer_policy
        if gate_reason:
            trace["evidence_gate_reason"] = gate_reason
        pred = _manual_verbalize(raw, choices) if is_mcq else _normalize_short_answer(raw)
        trace["raw_output"] = raw
        return pred, raw_evidence, trace, raw











