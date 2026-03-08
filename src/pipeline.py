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
        cache_by_id: Optional[Dict[str, Dict[str, Any]]] = None,
        use_cache_only: bool = False,
        include_candidate_details: bool = False,
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
        self.cache_by_id = cache_by_id or {}
        self.use_cache_only = use_cache_only
        self.include_candidate_details = include_candidate_details

        if self.selection_mode not in {"selective", "non_selective"}:
            raise ValueError("selection_mode must be 'selective' or 'non_selective'")

    def _expand_queries(self, question: str, item_id: str = "") -> List[str]:
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

    def _lexical_rank(self, question: str, chunks: List[EvidenceChunk]) -> List[EvidenceChunk]:
        if not chunks:
            return []
        idf = _idf_for_texts([question] + [c.text for c in chunks])
        qv = _tfidf(question, idf)
        scored: List[EvidenceChunk] = []
        for c in chunks:
            s = _cosine(qv, _tfidf(c.text, idf))
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

    def _select_chunks(self, question: str, candidates: List[EvidenceChunk], item_id: str = "") -> List[EvidenceChunk]:
        if not candidates:
            return []
        if self.selection_mode == "non_selective":
            return self._topk_diverse_by_url(candidates, self.keep_top_k)

        ranked = self._lexical_rank(question, candidates)
        ranked = self._llm_relevance_boost(question, ranked, item_id=item_id)
        ranked = self._apply_domain_priors(ranked)
        return self._topk_diverse_by_url(ranked, self.keep_top_k)

    def _retrieve_live(self, item: Dict[str, Any]) -> Tuple[List[EvidenceChunk], Dict[str, Any]]:
        question = item["question"]
        item_id = str(item.get("id", ""))
        queries = self._expand_queries(question, item_id=item_id)

        all_hits = []
        for q in queries:
            all_hits.extend(self.web.search(q, top_n=self.search_top_n))

        dedup_hits = self.web.dedupe_hits(all_hits, keep_per_domain=self.keep_per_domain)
        candidates = self.web.build_candidate_chunks(dedup_hits, max_pages=self.max_pages)
        selected = self._select_chunks(question, candidates, item_id=item_id)

        trace: Dict[str, Any] = {
            "query_source": "live",
            "selection_mode": self.selection_mode,
            "queries": queries,
            "retrieved_hits": len(all_hits),
            "dedup_hits": len(dedup_hits),
            "candidate_chunks": len(candidates),
            "selected_evidence": [_chunk_to_dict(s) for s in selected],
        }
        if self.include_candidate_details:
            trace["candidate_evidence"] = [_chunk_to_dict(c) for c in candidates]

        return selected, trace

    def _retrieve_from_cache(self, item: Dict[str, Any], cached: Dict[str, Any]) -> Tuple[List[EvidenceChunk], Dict[str, Any]]:
        question = item["question"]
        item_id = str(item.get("id", ""))
        candidate_dicts = cached.get("candidate_evidence", [])
        selected_dicts = cached.get("selected_evidence", [])

        candidates = [_dict_to_chunk(x) for x in candidate_dicts if isinstance(x, dict)]
        if candidates:
            selected = self._select_chunks(question, candidates, item_id=item_id)
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
            }

        return self._retrieve_live(item)

    def predict(self, item: Dict) -> Tuple[str, List[str], Dict]:
        selected, trace = self.prepare_evidence(item)
        evidence = [s.text for s in selected]
        evidence_low = [_normalize_text(x) for x in evidence]
        choice_texts = _choice_texts(item["choices"])
        top_score = float(selected[0].score) if selected else 0.0

        use_evidence = bool(evidence)
        gate_reason = ""
        if not evidence:
            gate_reason = "no_evidence"
            use_evidence = False
        elif self.selection_mode == "selective" and self.min_evidence_score > 0 and top_score < self.min_evidence_score:
            gate_reason = "low_score"
            use_evidence = False
        elif self.require_choice_overlap:
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
        trace["top_selected_score"] = round(top_score, 6)
        if gate_reason:
            trace["evidence_gate_reason"] = gate_reason
        pred = _manual_verbalize(raw, item["choices"])
        return pred, evidence, trace
