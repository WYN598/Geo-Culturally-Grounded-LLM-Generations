import json
import math
import re
from typing import Any, Dict, List, Optional, Tuple

from .llm_client import LLMClient, normalize_mcq_answer
from .retrieval import KBIndex, KBDoc
from .search_grounding import EvidenceChunk, WebSearcher


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


def load_jsonl(path: str) -> List[Dict]:
    rows: List[Dict] = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def dump_jsonl(path: str, rows: List[Dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_search_cache(path: str) -> Dict[str, Dict[str, Any]]:
    rows = load_jsonl(path)
    out: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        rid = str(r.get("id", "")).strip()
        if rid:
            out[rid] = r
    return out


def load_kb_cache(path: str) -> Dict[str, Dict[str, Any]]:
    rows = load_jsonl(path)
    out: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        rid = str(r.get("id", "")).strip()
        if rid:
            out[rid] = r
    return out


def format_mcq_prompt(question: str, choices: List[str]) -> str:
    return f"{question}\n" + "\n".join(choices) + "\nReturn only one letter: A/B/C/D..."


def build_augmented_prompt(question: str, choices: List[str], evidence: List[str]) -> str:
    evidence_block = "\n\n".join([f"[e{i+1}] {e}" for i, e in enumerate(evidence)])
    return (
        "Use the evidence to answer the MCQ. If evidence conflicts, choose the most supported option.\n"
        "Cite the evidence ids in one short sentence, then output final option letter only on the last line.\n\n"
        f"Evidence:\n{evidence_block}\n\n"
        f"Question:\n{format_mcq_prompt(question, choices)}"
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
    return "A"


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

    def predict(self, item: Dict) -> str:
        prompt = format_mcq_prompt(item["question"], item["choices"])
        raw = self.llm.generate(
            "You are a careful assistant.",
            prompt,
            trace_meta={"stage": "vanilla_answer", "item_id": str(item.get("id", ""))},
        )
        return _manual_verbalize(raw, item["choices"])


class KBPipeline:
    def __init__(
        self,
        llm: LLMClient,
        kb_index: KBIndex,
        retrieve_top_n: int = 5,
        keep_top_k: int = 3,
        selection_mode: str = "selective",
        min_evidence_score: float = 0.0,
        cache_by_id: Optional[Dict[str, Dict[str, Any]]] = None,
        use_cache_only: bool = False,
        include_candidate_details: bool = False,
    ):
        self.llm = llm
        self.kb_index = kb_index
        self.retrieve_top_n = retrieve_top_n
        self.keep_top_k = keep_top_k
        self.selection_mode = selection_mode
        self.min_evidence_score = float(min_evidence_score)
        self.cache_by_id = cache_by_id or {}
        self.use_cache_only = use_cache_only
        self.include_candidate_details = include_candidate_details

        if self.selection_mode not in {"selective", "non_selective"}:
            raise ValueError("selection_mode must be 'selective' or 'non_selective'")

    def rewrite_query(self, question: str) -> str:
        if self.llm.provider != "openai":
            return question
        prompt = f"Rewrite this question into a concise web/KB search query:\n{question}"
        q = self.llm.generate("You rewrite queries.", prompt, trace_meta={"stage": "kb_query_rewrite"}).strip()
        return q if len(q) > 2 else question

    def _select_docs(self, question: str, docs: List[KBDoc]) -> List[Tuple[KBDoc, float]]:
        if not docs:
            return []
        if self.selection_mode == "non_selective":
            return [(d, 0.0) for d in docs[: self.keep_top_k]]

        texts = [d.text for d in docs]
        idf = _idf_for_texts([question] + texts)
        qv = _tfidf(question, idf)
        scored: List[Tuple[KBDoc, float]] = []
        for d in docs:
            s = _cosine(qv, _tfidf(d.text, idf))
            scored.append((d, s))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[: self.keep_top_k]

    def _retrieve_live(self, item: Dict[str, Any]) -> Tuple[List[Tuple[KBDoc, float]], Dict[str, Any]]:
        question = item["question"]
        query = self.rewrite_query(question)
        retrieved = self.kb_index.search(query, top_n=self.retrieve_top_n)
        selected_scored = self._select_docs(question, retrieved)

        trace: Dict[str, Any] = {
            "query_source": "live",
            "selection_mode": self.selection_mode,
            "query": query,
            "retrieved_docs": len(retrieved),
            "selected_evidence": [_kbdoc_to_dict(d, s) for d, s in selected_scored],
        }
        if self.include_candidate_details:
            trace["candidate_evidence"] = [_kbdoc_to_dict(d, 0.0) for d in retrieved]
        return selected_scored, trace

    def _retrieve_from_cache(self, item: Dict[str, Any], cached: Dict[str, Any]) -> Tuple[List[Tuple[KBDoc, float]], Dict[str, Any]]:
        question = item["question"]
        candidate_dicts = cached.get("candidate_evidence", [])
        selected_dicts = cached.get("selected_evidence", [])

        candidates = [_dict_to_kbdoc(x) for x in candidate_dicts if isinstance(x, dict)]
        if candidates:
            selected_scored = self._select_docs(question, candidates)
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
            "selection_mode": self.selection_mode,
            "query": cached.get("query", question),
            "retrieved_docs": retrieved_docs,
            "selected_evidence": selected_dicts,
        }
        if self.include_candidate_details and candidates:
            trace["candidate_evidence"] = [_kbdoc_to_dict(c, 0.0) for c in candidates]
        return selected_scored, trace

    def prepare_evidence(self, item: Dict[str, Any]) -> Tuple[List[Tuple[KBDoc, float]], Dict[str, Any]]:
        item_id = str(item.get("id", "")).strip()
        if item_id and item_id in self.cache_by_id:
            return self._retrieve_from_cache(item, self.cache_by_id[item_id])

        if self.use_cache_only:
            return [], {
                "query_source": "cache",
                "cache_hit": False,
                "selection_mode": self.selection_mode,
                "query": item["question"],
                "retrieved_docs": 0,
                "selected_evidence": [],
            }

        return self._retrieve_live(item)

    def predict(self, item: Dict) -> Tuple[str, List[str], Dict]:
        selected_scored, trace = self.prepare_evidence(item)
        selected = [d for d, _ in selected_scored]
        evidence = [d.text for d in selected]
        top_score = float(selected_scored[0][1]) if selected_scored else 0.0

        use_evidence = bool(evidence)
        gate_reason = ""
        if not evidence:
            gate_reason = "no_evidence"
            use_evidence = False
        elif self.selection_mode == "selective" and self.min_evidence_score > 0 and top_score < self.min_evidence_score:
            gate_reason = "low_score"
            use_evidence = False

        if use_evidence:
            prompt = build_augmented_prompt(item["question"], item["choices"], evidence)
            raw = self.llm.generate(
                "You are a culturally-aware assistant.",
                prompt,
                trace_meta={"stage": "kb_answer_augmented", "item_id": str(item.get("id", ""))},
            )
        else:
            prompt = format_mcq_prompt(item["question"], item["choices"])
            raw = self.llm.generate(
                "You are a careful assistant.",
                prompt,
                trace_meta={"stage": "kb_answer_fallback", "item_id": str(item.get("id", ""))},
            )

        trace["used_evidence"] = use_evidence
        trace["top_selected_score"] = round(top_score, 6)
        if gate_reason:
            trace["evidence_gate_reason"] = gate_reason
        return _manual_verbalize(raw, item["choices"]), evidence, trace


class SearchPipeline:
    def __init__(
        self,
        llm: LLMClient,
        web: WebSearcher,
        search_top_n: int = 5,
        keep_top_k: int = 3,
        query_expansion_n: int = 2,
        max_pages: int = 8,
        keep_per_domain: int = 2,
        llm_relevance: bool = True,
        llm_relevance_top_m: int = 8,
        selection_mode: str = "selective",
        min_evidence_score: float = 0.0,
        require_choice_overlap: bool = False,
        diversify_by_url: bool = False,
        domain_priors: Optional[Dict[str, float]] = None,
        label_task_force_use_evidence: bool = False,
        dataset_overrides: Optional[Dict[str, Dict[str, Any]]] = None,
        cache_by_id: Optional[Dict[str, Dict[str, Any]]] = None,
        use_cache_only: bool = False,
        include_candidate_details: bool = False,
        snippet_only_penalty: float = 0.0,
        label_semantic_bonus: float = 0.0,
        label_noise_penalty: float = 0.0,
        label_retry_min_semantic_overlap: float = 0.06,
        label_min_semantic_overlap_for_use: float = 0.0,
        label_min_top_score_for_use: float = 0.0,
    ):
        self.llm = llm
        self.web = web
        self.search_top_n = search_top_n
        self.keep_top_k = keep_top_k
        self.query_expansion_n = query_expansion_n
        self.max_pages = max_pages
        self.keep_per_domain = keep_per_domain
        self.llm_relevance = llm_relevance and (self.llm.provider == "openai")
        self.llm_relevance_top_m = llm_relevance_top_m
        self.selection_mode = selection_mode
        self.min_evidence_score = float(min_evidence_score)
        self.require_choice_overlap = bool(require_choice_overlap)
        self.diversify_by_url = bool(diversify_by_url)
        self.domain_priors = {str(k).lower(): float(v) for k, v in (domain_priors or {}).items()}
        self.label_task_force_use_evidence = bool(label_task_force_use_evidence)
        self.dataset_overrides: Dict[str, Dict[str, Any]] = {}
        for k, v in (dataset_overrides or {}).items():
            if isinstance(v, dict):
                self.dataset_overrides[str(k).strip().lower()] = dict(v)
        self.cache_by_id = cache_by_id or {}
        self.use_cache_only = use_cache_only
        self.include_candidate_details = include_candidate_details
        self.snippet_only_penalty = max(0.0, float(snippet_only_penalty))
        self.label_semantic_bonus = float(label_semantic_bonus)
        self.label_noise_penalty = max(0.0, float(label_noise_penalty))
        self.label_retry_min_semantic_overlap = max(0.0, float(label_retry_min_semantic_overlap))
        self.label_min_semantic_overlap_for_use = max(0.0, float(label_min_semantic_overlap_for_use))
        self.label_min_top_score_for_use = max(0.0, float(label_min_top_score_for_use))
        self.label_noise_keywords = [
            "annotation",
            "annotated bibliography",
            "bibliography",
            "citation",
            "label claim",
            "arcgis",
            "calculation",
            "guides.lib",
            "how to annotate",
            "annotating",
        ]
        self.label_semantic_keywords = [
            "stereotype",
            "generalization",
            "bias",
            "prejudice",
            "trope",
        ]
        self.label_retry_min_overlap = 0.10

        if self.selection_mode not in {"selective", "non_selective"}:
            raise ValueError("selection_mode must be 'selective' or 'non_selective'")

    def _build_label_claim_queries(self, question: str, max_n: int = 3, retry: bool = False) -> List[str]:
        left, right = _extract_claim_pair(question)
        if not left or not right:
            return [question]

        pred = _clean_claim_predicate(right) or right
        subs = _claim_subject_forms(left)
        sub = subs[0] if subs else left
        raw: List[str] = []
        if retry:
            # Retry mode: label-semantics heavier but keep one factual fallback.
            raw.extend(
                [
                    f"stereotype about {sub} {pred}",
                    f"generalization linking {sub} with {pred}",
                    f"bias or prejudice about {sub} {pred}",
                    f"{sub} known for {pred}",
                    f"{left} {right}",
                ]
            )
        else:
            # Primary mode: mixed retrieval (factual + label semantics) to avoid
            # over-shifting toward generic stereotype-discourse pages.
            raw.extend(
                [
                    f"{sub} {pred}",
                    f"{sub} known for {pred}",
                    f"stereotype about {sub} {pred}",
                    f"{left} {right}",
                ]
            )
        # Preserve one factual fallback with raw left/right.
        raw.append(f"{left} {right}")
        uniq: List[str] = []
        for q in raw:
            q = re.sub(r"\s+", " ", q).strip()
            if not q:
                continue
            if q not in uniq:
                uniq.append(q)
            if len(uniq) >= max(1, max_n):
                break
        return uniq or [question]

    def _expand_queries(
        self,
        question: str,
        item_id: str = "",
        choices: Optional[List[str]] = None,
    ) -> List[str]:
        if _is_stereotype_label_task(choices or []):
            return self._build_label_claim_queries(question, max_n=self.query_expansion_n, retry=False)

        if self.query_expansion_n <= 1 or self.llm.provider != "openai":
            return [question]

        prompt = (
            "Generate up to 3 short web search queries for this question.\n"
            "Rules: one query per line, no numbering, no quotes.\n"
            f"Question: {question}"
        )
        raw = self.llm.generate(
            "You are a query rewriting assistant.",
            prompt,
            trace_meta={"stage": "search_query_rewrite", "item_id": item_id},
        )
        lines = [re.sub(r"^[\-\d\.\)\s]+", "", x).strip() for x in raw.splitlines()]
        cleaned = [x for x in lines if len(x) > 3]
        uniq: List[str] = []
        for q in [question] + cleaned:
            if q not in uniq:
                uniq.append(q)
            if len(uniq) >= self.query_expansion_n:
                break
        return uniq or [question]

    def _chunk_has_claim_overlap(self, chunk: EvidenceChunk, claim_tokens: List[str]) -> bool:
        if not claim_tokens:
            return False
        bag = " ".join([chunk.title or "", chunk.url or "", chunk.text or ""]).lower()
        for t in claim_tokens:
            if t and t in bag:
                return True
        return False

    def _candidate_claim_overlap_rate(self, candidates: List[EvidenceChunk], claim_tokens: List[str]) -> float:
        if not candidates or not claim_tokens:
            return 0.0
        n = len(candidates)
        hit = sum(1 for c in candidates if self._chunk_has_claim_overlap(c, claim_tokens))
        return float(hit) / float(n)

    def _chunk_has_label_semantic(self, chunk: EvidenceChunk) -> bool:
        bag = " ".join([chunk.title or "", chunk.url or "", chunk.text or ""]).lower()
        return any(k in bag for k in self.label_semantic_keywords)

    def _chunk_has_label_noise(self, chunk: EvidenceChunk) -> bool:
        meta = f"{chunk.title} {chunk.url}".lower()
        low_text = (chunk.text or "").lower()
        return any(k in meta for k in self.label_noise_keywords) or any(
            k in low_text[:500] for k in self.label_noise_keywords[:5]
        )

    def _candidate_label_semantic_overlap_rate(self, candidates: List[EvidenceChunk]) -> float:
        if not candidates:
            return 0.0
        n = len(candidates)
        hit = sum(1 for c in candidates if self._chunk_has_label_semantic(c))
        return float(hit) / float(n)

    def _filter_label_noise_candidates(
        self,
        candidates: List[EvidenceChunk],
        claim_tokens: List[str],
    ) -> Tuple[List[EvidenceChunk], int]:
        if not candidates:
            return candidates, 0
        kept: List[EvidenceChunk] = []
        removed = 0
        for c in candidates:
            noise = self._chunk_has_label_noise(c)
            has_claim = self._chunk_has_claim_overlap(c, claim_tokens)
            has_label_semantic = self._chunk_has_label_semantic(c)
            if noise and not has_claim and not has_label_semantic:
                removed += 1
                continue
            kept.append(c)
        return kept, removed

    def _lexical_rank(self, question: str, chunks: List[EvidenceChunk], is_label_task: bool = False) -> List[EvidenceChunk]:
        if not chunks:
            return []
        idf = _idf_for_texts([question] + [c.text for c in chunks])
        qv = _tfidf(question, idf)
        scored: List[EvidenceChunk] = []
        for c in chunks:
            s = _cosine(qv, _tfidf(c.text, idf))
            if self.snippet_only_penalty > 0 and str(c.text or "").startswith("[snippet_only]"):
                s -= self.snippet_only_penalty
            if is_label_task:
                if self._chunk_has_label_semantic(c):
                    s += self.label_semantic_bonus
                if self.label_noise_penalty > 0 and self._chunk_has_label_noise(c) and not self._chunk_has_label_semantic(c):
                    s -= self.label_noise_penalty
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
            delta = self._domain_prior_delta(c.domain)
            out.append(
                EvidenceChunk(
                    query=c.query,
                    title=c.title,
                    url=c.url,
                    domain=c.domain,
                    text=c.text,
                    score=c.score + delta,
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

    def _dataset_cfg(self, item: Dict[str, Any]) -> Dict[str, Any]:
        ds = str(item.get("dataset", "")).strip().lower()
        if not ds:
            return {}
        return dict(self.dataset_overrides.get(ds, {}))

    @staticmethod
    def _to_int(val: Any, default: int) -> int:
        try:
            return int(val)
        except Exception:
            return int(default)

    @staticmethod
    def _to_float(val: Any, default: float) -> float:
        try:
            return float(val)
        except Exception:
            return float(default)

    @staticmethod
    def _to_bool(val: Any, default: bool) -> bool:
        if isinstance(val, bool):
            return val
        if val is None:
            return bool(default)
        s = str(val).strip().lower()
        if s in {"1", "true", "yes", "y", "on"}:
            return True
        if s in {"0", "false", "no", "n", "off"}:
            return False
        return bool(default)

    def _select_chunks(
        self,
        question: str,
        candidates: List[EvidenceChunk],
        item_id: str = "",
        keep_top_k_override: Optional[int] = None,
        is_label_task: bool = False,
    ) -> List[EvidenceChunk]:
        if not candidates:
            return []
        k = max(1, int(keep_top_k_override if keep_top_k_override is not None else self.keep_top_k))
        if self.selection_mode == "non_selective":
            return self._topk_diverse_by_url(candidates, k)

        ranked = self._lexical_rank(question, candidates, is_label_task=is_label_task)
        ranked = self._llm_relevance_boost(question, ranked, item_id=item_id)
        ranked = self._apply_domain_priors(ranked)
        return self._topk_diverse_by_url(ranked, k)

    def _retrieve_live(self, item: Dict[str, Any]) -> Tuple[List[EvidenceChunk], Dict[str, Any]]:
        question = item["question"]
        item_id = str(item.get("id", ""))
        ds_cfg = self._dataset_cfg(item)
        keep_top_k_eff = self._to_int(ds_cfg.get("keep_top_k", self.keep_top_k), self.keep_top_k)
        is_label_task = _is_stereotype_label_task(item.get("choices", []) or [])
        claim_left, claim_right = _extract_claim_pair(question)
        claim_tokens = _claim_tokens(f"{claim_left} {claim_right}")
        queries = self._expand_queries(question, item_id=item_id, choices=item.get("choices", []))

        all_hits = []
        search_events: List[Dict[str, Any]] = []
        for q in queries:
            q_hits = self.web.search(q, top_n=self.search_top_n)
            all_hits.extend(q_hits)
            event = self.web.last_search_event()
            if event:
                search_events.append(event)

        dedup_hits = self.web.dedupe_hits(all_hits, keep_per_domain=self.keep_per_domain)
        candidates = self.web.build_candidate_chunks(dedup_hits, max_pages=self.max_pages)
        initial_overlap_rate = self._candidate_claim_overlap_rate(candidates, claim_tokens)
        initial_semantic_rate = self._candidate_label_semantic_overlap_rate(candidates)
        retry_triggered = False
        retry_queries: List[str] = []
        retry_reasons: List[str] = []
        noise_removed = 0

        should_retry_label = False
        if is_label_task and claim_tokens and initial_overlap_rate < self.label_retry_min_overlap:
            should_retry_label = True
            retry_reasons.append("low_claim_overlap")
        if is_label_task and initial_semantic_rate < self.label_retry_min_semantic_overlap:
            should_retry_label = True
            retry_reasons.append("low_label_semantic_overlap")

        if should_retry_label:
            retry_queries = self._build_label_claim_queries(question, max_n=max(self.query_expansion_n, 3), retry=True)
            if retry_queries:
                retry_triggered = True
                for q in retry_queries:
                    q_hits = self.web.search(q, top_n=self.search_top_n)
                    all_hits.extend(q_hits)
                    event = self.web.last_search_event()
                    if event:
                        event = dict(event)
                        event["retry"] = True
                        search_events.append(event)
                dedup_hits = self.web.dedupe_hits(all_hits, keep_per_domain=self.keep_per_domain)
                candidates = self.web.build_candidate_chunks(dedup_hits, max_pages=self.max_pages)

        if is_label_task:
            candidates, noise_removed = self._filter_label_noise_candidates(candidates, claim_tokens)

        post_overlap_rate = self._candidate_claim_overlap_rate(candidates, claim_tokens)
        post_semantic_rate = self._candidate_label_semantic_overlap_rate(candidates)
        selected = self._select_chunks(
            question,
            candidates,
            item_id=item_id,
            keep_top_k_override=keep_top_k_eff,
            is_label_task=is_label_task,
        )

        trace: Dict[str, Any] = {
            "query_source": "live",
            "selection_mode": self.selection_mode,
            "queries": queries,
            "retrieved_hits": len(all_hits),
            "dedup_hits": len(dedup_hits),
            "candidate_chunks": len(candidates),
            "selected_evidence": [_chunk_to_dict(s) for s in selected],
            "search_events": search_events,
            "is_label_task": is_label_task,
            "claim_left": claim_left,
            "claim_right": claim_right,
            "label_initial_overlap_rate": round(initial_overlap_rate, 6),
            "label_post_overlap_rate": round(post_overlap_rate, 6),
            "label_initial_semantic_overlap_rate": round(initial_semantic_rate, 6),
            "label_post_semantic_overlap_rate": round(post_semantic_rate, 6),
            "label_retry_triggered": retry_triggered,
            "label_retry_queries": retry_queries,
            "label_retry_reasons": retry_reasons,
            "label_noise_removed": int(noise_removed),
        }
        if self.include_candidate_details:
            trace["candidate_evidence"] = [_chunk_to_dict(c) for c in candidates]

        return selected, trace

    def _retrieve_from_cache(self, item: Dict[str, Any], cached: Dict[str, Any]) -> Tuple[List[EvidenceChunk], Dict[str, Any]]:
        question = item["question"]
        item_id = str(item.get("id", ""))
        ds_cfg = self._dataset_cfg(item)
        keep_top_k_eff = self._to_int(ds_cfg.get("keep_top_k", self.keep_top_k), self.keep_top_k)
        is_label_task = _is_stereotype_label_task(item.get("choices", []) or [])
        candidate_dicts = cached.get("candidate_evidence", [])
        selected_dicts = cached.get("selected_evidence", [])

        candidates = [_dict_to_chunk(x) for x in candidate_dicts if isinstance(x, dict)]
        if candidates:
            selected = self._select_chunks(
                question,
                candidates,
                item_id=item_id,
                keep_top_k_override=keep_top_k_eff,
                is_label_task=is_label_task,
            )
            selected_dicts = [_chunk_to_dict(s) for s in selected]
        else:
            selected = [_dict_to_chunk(x) for x in selected_dicts if isinstance(x, dict)]

        trace = {
            "query_source": "cache",
            "cache_hit": True,
            "selection_mode": self.selection_mode,
            "queries": cached.get("queries", [question]),
            "retrieved_hits": int(cached.get("retrieved_hits", 0) or 0),
            "dedup_hits": int(cached.get("dedup_hits", 0) or 0),
            "candidate_chunks": len(candidates) if candidates else int(cached.get("candidate_chunks", 0) or 0),
            "selected_evidence": selected_dicts,
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
                "query_source": "cache",
                "cache_hit": False,
                "selection_mode": self.selection_mode,
                "queries": [item["question"]],
                "retrieved_hits": 0,
                "dedup_hits": 0,
                "candidate_chunks": 0,
                "selected_evidence": [],
                "search_events": [],
            }

        return self._retrieve_live(item)

    def predict(self, item: Dict) -> Tuple[str, List[str], Dict]:
        selected, trace = self.prepare_evidence(item)
        evidence = [s.text for s in selected]
        evidence_low = [_normalize_text(x) for x in evidence]
        choice_texts = _choice_texts(item["choices"])
        is_label_task = _is_stereotype_label_task(item["choices"])
        ds_cfg = self._dataset_cfg(item)
        min_score_eff = self._to_float(ds_cfg.get("min_evidence_score", self.min_evidence_score), self.min_evidence_score)
        require_overlap_eff = self._to_bool(ds_cfg.get("require_choice_overlap", self.require_choice_overlap), self.require_choice_overlap)
        force_label_eff = self._to_bool(
            ds_cfg.get("label_task_force_use_evidence", self.label_task_force_use_evidence),
            self.label_task_force_use_evidence,
        )
        top_score = float(selected[0].score) if selected else 0.0
        selected_semantic_overlap = 0.0
        if selected:
            selected_semantic_overlap = float(
                sum(1 for s in selected if self._chunk_has_label_semantic(s)) / len(selected)
            )

        use_evidence = bool(evidence)
        gate_reason = ""
        if not evidence:
            gate_reason = "no_evidence"
            use_evidence = False
        elif self.selection_mode == "selective" and min_score_eff > 0 and top_score < min_score_eff:
            gate_reason = "low_score"
            use_evidence = False
        elif require_overlap_eff:
            overlap = False
            for c in choice_texts:
                if len(c) < 2:
                    continue
                if any(c in ev for ev in evidence_low):
                    overlap = True
                    break
            if not overlap:
                gate_reason = "no_choice_overlap"
                use_evidence = False
        elif is_label_task and self.label_min_top_score_for_use > 0 and top_score < self.label_min_top_score_for_use:
            gate_reason = "low_label_top_score"
            use_evidence = False
        elif (
            is_label_task
            and self.label_min_semantic_overlap_for_use > 0
            and selected_semantic_overlap < self.label_min_semantic_overlap_for_use
        ):
            gate_reason = "low_label_semantic_overlap"
            use_evidence = False

        if is_label_task and force_label_eff and evidence:
            # Force the model to consume retrieved evidence on label tasks to expose
            # potential search-grounding side effects (paper-style phenomenon study).
            if (
                top_score >= self.label_min_top_score_for_use
                and selected_semantic_overlap >= self.label_min_semantic_overlap_for_use
            ):
                use_evidence = True
                gate_reason = "forced_label_task_evidence"

        if use_evidence:
            prompt = build_augmented_prompt(item["question"], item["choices"], evidence)
            raw = self.llm.generate(
                "You are a culturally-aware assistant.",
                prompt,
                trace_meta={"stage": "search_answer_augmented", "item_id": str(item.get("id", ""))},
            )
        else:
            prompt = format_mcq_prompt(item["question"], item["choices"])
            raw = self.llm.generate(
                "You are a careful assistant.",
                prompt,
                trace_meta={"stage": "search_answer_fallback", "item_id": str(item.get("id", ""))},
            )

        trace["used_evidence"] = use_evidence
        trace["is_label_task"] = is_label_task
        trace["effective_min_evidence_score"] = round(float(min_score_eff), 6)
        trace["effective_require_choice_overlap"] = bool(require_overlap_eff)
        trace["effective_label_force_use_evidence"] = bool(force_label_eff)
        trace["effective_label_min_top_score_for_use"] = round(float(self.label_min_top_score_for_use), 6)
        trace["effective_label_min_semantic_overlap_for_use"] = round(float(self.label_min_semantic_overlap_for_use), 6)
        trace["selected_label_semantic_overlap"] = round(float(selected_semantic_overlap), 6)
        trace["top_selected_score"] = round(top_score, 6)
        if gate_reason:
            trace["evidence_gate_reason"] = gate_reason
        pred = _manual_verbalize(raw, item["choices"])
        return pred, evidence, trace
