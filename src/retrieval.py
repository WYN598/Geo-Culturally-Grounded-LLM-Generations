import json
import math
import re
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class KBDoc:
    id: str
    source: str
    country: str
    text: str


class TfidfKBIndex:
    def __init__(self, docs: List[KBDoc]):
        self.docs = docs
        self.idf = self._build_idf([d.text for d in docs])
        self.doc_vecs = [self._tfidf_vector(d.text) for d in docs]
        self.doc_norms = [self._norm(v) for v in self.doc_vecs]

    @classmethod
    def from_jsonl(cls, path: str) -> "TfidfKBIndex":
        docs: List[KBDoc] = []
        with open(path, "r", encoding="utf-8-sig") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                docs.append(
                    KBDoc(
                        id=obj.get("id", ""),
                        source=obj.get("source", ""),
                        country=obj.get("country", ""),
                        text=obj.get("text", ""),
                    )
                )
        return cls(docs)

    def search(self, query: str, top_n: int = 5) -> List[KBDoc]:
        if not self.docs:
            return []
        q_vec = self._tfidf_vector(query)
        q_norm = self._norm(q_vec)
        sims = [self._cosine(q_vec, q_norm, dv, dn) for dv, dn in zip(self.doc_vecs, self.doc_norms)]
        ranked = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)[:top_n]
        return [self.docs[i] for i in ranked]

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"[a-zA-Z]+", text.lower())

    def _build_idf(self, docs: List[str]) -> Dict[str, float]:
        n_docs = len(docs)
        df: Dict[str, int] = {}
        for text in docs:
            for t in set(self._tokenize(text)):
                df[t] = df.get(t, 0) + 1
        return {t: math.log((1 + n_docs) / (1 + c)) + 1.0 for t, c in df.items()}

    def _tfidf_vector(self, text: str) -> Dict[str, float]:
        toks = self._tokenize(text)
        if not toks:
            return {}
        tf: Dict[str, int] = {}
        for t in toks:
            tf[t] = tf.get(t, 0) + 1
        total = len(toks)
        return {t: (c / total) * self.idf.get(t, 1.0) for t, c in tf.items()}

    @staticmethod
    def _norm(vec: Dict[str, float]) -> float:
        return math.sqrt(sum(v * v for v in vec.values()))

    @staticmethod
    def _cosine(a: Dict[str, float], an: float, b: Dict[str, float], bn: float) -> float:
        if an == 0 or bn == 0:
            return 0.0
        dot = 0.0
        for k, v in a.items():
            dot += v * b.get(k, 0.0)
        return dot / (an * bn)
