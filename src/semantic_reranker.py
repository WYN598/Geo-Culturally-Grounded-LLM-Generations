from typing import List, Optional

import torch


class SemanticReranker:
    def __init__(
        self,
        backend: str = "none",
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-12-v2",
        top_m: int = 12,
        weight: float = 0.2,
        device: str = "cuda",
        batch_size: int = 32,
    ):
        self.backend = str(backend or "none").strip().lower()
        self.model_name = str(model_name or "cross-encoder/ms-marco-MiniLM-L-12-v2").strip()
        self.top_m = max(1, int(top_m))
        self.weight = float(weight)
        self.device = str(device or "cuda").strip().lower()
        self.batch_size = max(1, int(batch_size))
        self._model = None
        self._load_error = ""

    def enabled(self) -> bool:
        return self.backend != "none"

    def status(self) -> dict:
        return {
            "backend": self.backend,
            "model_name": self.model_name,
            "top_m": self.top_m,
            "weight": self.weight,
            "device": self._resolved_device(),
            "batch_size": self.batch_size,
            "available": self._ensure_model(),
            "load_error": self._load_error,
        }

    def _resolved_device(self) -> str:
        if self.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cuda" and not torch.cuda.is_available():
            return "cpu"
        return self.device

    def _ensure_model(self) -> bool:
        if self.backend == "none":
            return False
        if self._model is not None:
            return True
        if self._load_error:
            return False
        if self.backend != "cross_encoder":
            self._load_error = f"unsupported_backend:{self.backend}"
            return False
        try:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name, device=self._resolved_device())
            return True
        except Exception as e:  # pragma: no cover - optional dependency/runtime
            self._load_error = str(e)
            self._model = None
            return False

    def score(self, query: str, texts: List[str]) -> Optional[List[float]]:
        if not texts or not self._ensure_model():
            return None
        try:
            pairs = [[query, text] for text in texts]
            preds = self._model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
            return [float(x) for x in preds]
        except Exception as e:  # pragma: no cover - runtime/model failure
            self._load_error = str(e)
            return None
