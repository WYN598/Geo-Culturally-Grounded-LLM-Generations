import json
import math
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol

import numpy as np

try:
    import faiss  # type: ignore

    _FAISS_AVAILABLE = True
except Exception:
    faiss = None
    _FAISS_AVAILABLE = False


@dataclass
class KBDoc:
    id: str
    source: str
    country: str
    text: str


class KBIndex(Protocol):
    def search(self, query: str, top_n: int = 5) -> List[KBDoc]:
        ...


def load_kb_docs(path: str) -> List[KBDoc]:
    docs: List[KBDoc] = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            docs.append(
                KBDoc(
                    id=str(obj.get("id", "")),
                    source=str(obj.get("source", "")),
                    country=str(obj.get("country", "")),
                    text=str(obj.get("text", "")),
                )
            )
    return docs


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        d = np.linalg.norm(x)
        if d == 0:
            return x
        return x / d
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms


class OpenAIEmbedder:
    def __init__(self, model: str = "text-embedding-3-small", batch_size: int = 64, dimensions: int = 0):
        self.model = model
        self.batch_size = max(1, int(batch_size))
        self.dimensions = int(dimensions)
        self.max_retries = int(os.getenv("OPENAI_EMBED_RETRIES", "6"))
        self.timeout = float(os.getenv("OPENAI_TIMEOUT_SEC", "60"))
        self._client = None

    @staticmethod
    def _default_openai_base_url() -> str:
        return "https://api.openai.com/v1"

    def _get_embedding_api_key(self) -> str:
        return (
            os.getenv("OPENAI_EMBEDDING_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or os.getenv("OPENAIAPI")
            or ""
        )

    def _get_embedding_base_url(self) -> str:
        return os.getenv("OPENAI_EMBEDDING_BASE_URL") or os.getenv(
            "OPENAI_BASE_URL",
            self._default_openai_base_url(),
        )

    def _get_client(self):
        if self._client is not None:
            return self._client
        from openai import OpenAI

        api_key = self._get_embedding_api_key()
        if not api_key:
            raise RuntimeError(
                "Dense KB retrieval requires OPENAI_EMBEDDING_API_KEY/OPENAI_API_KEY/OPENAIAPI for embeddings."
            )
        self._client = OpenAI(
            api_key=api_key,
            base_url=self._get_embedding_base_url(),
            timeout=self.timeout,
            max_retries=2,
        )
        return self._client

    def _embed_batch(self, texts: List[str]) -> np.ndarray:
        client = self._get_client()
        kwargs: Dict[str, Any] = {"model": self.model, "input": texts}
        if self.dimensions > 0:
            kwargs["dimensions"] = self.dimensions
        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                resp = client.embeddings.create(**kwargs)
                arr = np.array([x.embedding for x in resp.data], dtype=np.float32)
                return arr
            except Exception as e:
                last_err = e
                if attempt + 1 < self.max_retries:
                    time.sleep(min(10.0, 1.2 * (2 ** attempt)))
                    continue
                break
        if last_err is not None:
            raise last_err
        return np.zeros((len(texts), 1), dtype=np.float32)

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)
        mats: List[np.ndarray] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            mats.append(self._embed_batch(batch))
        return np.vstack(mats)

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_texts([text])[0]


class DenseKBIndex:
    def __init__(self, docs: List[KBDoc], embeddings: np.ndarray, embedder: OpenAIEmbedder, use_faiss: bool = True):
        self.docs = docs
        self.embedder = embedder
        self.embeddings = _l2_normalize(np.asarray(embeddings, dtype=np.float32))
        self.dim = int(self.embeddings.shape[1]) if self.embeddings.size else 0
        self.use_faiss = bool(use_faiss and _FAISS_AVAILABLE and self.dim > 0)
        self.index = None
        if self.use_faiss:
            self.index = faiss.IndexFlatIP(self.dim)
            self.index.add(self.embeddings)

    @classmethod
    def from_jsonl(cls, kb_path: str, embedder: OpenAIEmbedder, use_faiss: bool = True) -> "DenseKBIndex":
        docs = load_kb_docs(kb_path)
        vecs = embedder.embed_texts([d.text for d in docs])
        return cls(docs=docs, embeddings=vecs, embedder=embedder, use_faiss=use_faiss)

    @staticmethod
    def is_valid_dir(index_dir: str) -> bool:
        p = os.path.abspath(index_dir)
        return os.path.exists(os.path.join(p, "docs.jsonl")) and os.path.exists(os.path.join(p, "vectors.npy"))

    def save(self, index_dir: str) -> None:
        os.makedirs(index_dir, exist_ok=True)
        docs_path = os.path.join(index_dir, "docs.jsonl")
        vec_path = os.path.join(index_dir, "vectors.npy")
        meta_path = os.path.join(index_dir, "meta.json")
        with open(docs_path, "w", encoding="utf-8") as f:
            for d in self.docs:
                f.write(
                    json.dumps(
                        {"id": d.id, "source": d.source, "country": d.country, "text": d.text},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        np.save(vec_path, self.embeddings)
        meta = {
            "backend": "dense",
            "embedding_model": self.embedder.model,
            "dim": self.dim,
            "num_docs": len(self.docs),
            "use_faiss": bool(self.use_faiss),
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        if self.use_faiss and self.index is not None:
            faiss.write_index(self.index, os.path.join(index_dir, "faiss.index"))

    @classmethod
    def from_dir(cls, index_dir: str, embedder: Optional[OpenAIEmbedder] = None, use_faiss: bool = True) -> "DenseKBIndex":
        docs = load_kb_docs(os.path.join(index_dir, "docs.jsonl"))
        vecs = np.load(os.path.join(index_dir, "vectors.npy")).astype(np.float32)
        meta_path = os.path.join(index_dir, "meta.json")
        meta = {}
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        if embedder is None:
            embedder = OpenAIEmbedder(
                model=str(meta.get("embedding_model", "text-embedding-3-small")),
            )
        obj = cls(docs=docs, embeddings=vecs, embedder=embedder, use_faiss=use_faiss)
        faiss_path = os.path.join(index_dir, "faiss.index")
        if obj.use_faiss and os.path.exists(faiss_path):
            obj.index = faiss.read_index(faiss_path)
        return obj

    def search(self, query: str, top_n: int = 5) -> List[KBDoc]:
        if not self.docs:
            return []
        q = _l2_normalize(self.embedder.embed_query(query).astype(np.float32))
        k = max(1, min(top_n, len(self.docs)))
        if self.use_faiss and self.index is not None:
            _, idx = self.index.search(q.reshape(1, -1), k)
            ids = [int(i) for i in idx[0] if int(i) >= 0]
        else:
            sims = self.embeddings @ q
            ids = np.argsort(-sims)[:k].tolist()
        return [self.docs[i] for i in ids]


class TfidfKBIndex:
    def __init__(self, docs: List[KBDoc]):
        self.docs = docs
        self.idf = self._build_idf([d.text for d in docs])
        self.doc_vecs = [self._tfidf_vector(d.text) for d in docs]
        self.doc_norms = [self._norm(v) for v in self.doc_vecs]

    @classmethod
    def from_jsonl(cls, path: str) -> "TfidfKBIndex":
        return cls(load_kb_docs(path))

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
        return re.findall(r"[^\W\d_]+", text.lower(), flags=re.UNICODE)

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


def make_kb_index(kb_path: str, kcfg: Dict[str, Any]) -> KBIndex:
    backend = str(kcfg.get("backend", "tfidf")).strip().lower()
    if backend == "tfidf":
        return TfidfKBIndex.from_jsonl(kb_path)

    if backend != "dense":
        raise ValueError(f"Unsupported kb backend: {backend}")

    embedder = OpenAIEmbedder(
        model=str(kcfg.get("embedding_model", "text-embedding-3-small")),
        batch_size=int(kcfg.get("embedding_batch_size", 64)),
        dimensions=int(kcfg.get("embedding_dimensions", 0)),
    )
    dense_dir = str(kcfg.get("dense_index_dir", "") or "").strip()
    use_faiss = bool(kcfg.get("use_faiss", True))
    auto_save = bool(kcfg.get("dense_auto_save", True))

    if dense_dir and DenseKBIndex.is_valid_dir(dense_dir):
        return DenseKBIndex.from_dir(dense_dir, embedder=embedder, use_faiss=use_faiss)

    index = DenseKBIndex.from_jsonl(kb_path, embedder=embedder, use_faiss=use_faiss)
    if dense_dir and auto_save:
        index.save(dense_dir)
    return index
