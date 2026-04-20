import json
import os
import re
import time
import unicodedata
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

from dotenv import load_dotenv


class LLMClient:
    def __init__(
        self,
        provider: str = "mock",
        model: str = "gpt-4o-mini",
        temperature: float = 0.2,
        max_tokens: int = 300,
    ):
        load_dotenv()
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.usage_log: List[Dict[str, Any]] = []
        self.max_api_retries = int(os.getenv("OPENAI_MAX_RETRIES", "6") or 6)
        self.api_retry_base_sec = float(os.getenv("OPENAI_RETRY_BASE_SEC", "2.0") or 2.0)

    @staticmethod
    def _default_openai_base_url() -> str:
        return "https://api.openai.com/v1"

    def _get_chat_api_key(self) -> str:
        return os.getenv("OPENAI_API_KEY") or os.getenv("OPENAIAPI") or ""

    def _get_embedding_api_key(self) -> str:
        return (
            os.getenv("OPENAI_EMBEDDING_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or os.getenv("OPENAIAPI")
            or ""
        )

    def _get_chat_base_url(self) -> str:
        return os.getenv("OPENAI_BASE_URL", self._default_openai_base_url())

    def _get_embedding_base_url(self) -> str:
        return os.getenv("OPENAI_EMBEDDING_BASE_URL") or self._get_chat_base_url()

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        trace_meta: Optional[Dict[str, Any]] = None,
    ) -> str:
        api_key = self._get_chat_api_key()
        usage: Dict[str, int] = {}
        if self.provider == "openai":
            if not api_key:
                raise RuntimeError("LLM provider is 'openai' but OPENAI_API_KEY/OPENAIAPI is not set.")
            text, usage = self._generate_openai(system_prompt, user_prompt)
            self._record_usage(system_prompt, user_prompt, text, usage, trace_meta=trace_meta)
            return text
        if self.provider == "mock":
            text = self._generate_mock(system_prompt, user_prompt)
            usage = self._estimate_usage(system_prompt, user_prompt, text)
            self._record_usage(system_prompt, user_prompt, text, usage, trace_meta=trace_meta)
            return text
        raise ValueError(f"Unsupported LLM provider: {self.provider}")

    @staticmethod
    def _sanitize_text_for_json(text: str) -> str:
        if not text:
            return ""
        s = str(text)
        out = []
        for ch in s:
            oc = ord(ch)
            # Drop surrogate code points and NUL/control chars that can break JSON payloads.
            if 0xD800 <= oc <= 0xDFFF:
                continue
            if oc == 0:
                continue
            cat = unicodedata.category(ch)
            if oc < 32 and ch not in {"\n", "\r", "\t"}:
                continue
            # Drop other control/format/private-use/unassigned chars that often leak from scraped pages
            # and can make downstream JSON serialization or API parsing unstable.
            if cat in {"Cc", "Cf", "Cs", "Co", "Cn"} and ch not in {"\n", "\r", "\t"}:
                continue
            out.append(ch)
        cleaned = "".join(out)
        cleaned = unicodedata.normalize("NFKC", cleaned)
        return cleaned.encode("utf-8", "ignore").decode("utf-8", "ignore")

    @classmethod
    def _ultra_sanitize_text_for_json(cls, text: str) -> str:
        cleaned = cls._sanitize_text_for_json(text)
        ascii_safe = cleaned.encode("ascii", "ignore").decode("ascii", "ignore")
        ascii_safe = re.sub(r"\s+", " ", ascii_safe).strip()
        return ascii_safe

    @staticmethod
    def _truncate_text(text: str, max_chars: int = 12000) -> str:
        if len(text) <= max_chars:
            return text
        return text[:max_chars]

    def embed_texts(
        self,
        texts: List[str],
        model: str = "text-embedding-3-small",
        trace_meta: Optional[Dict[str, Any]] = None,
    ) -> List[List[float]]:
        api_key = self._get_embedding_api_key()
        if self.provider != "openai":
            raise ValueError(f"Embedding provider unsupported for current provider: {self.provider}")
        if not api_key:
            raise RuntimeError(
                "Embedding provider is 'openai' but OPENAI_EMBEDDING_API_KEY/OPENAI_API_KEY/OPENAIAPI is not set."
            )
        vectors, usage = self._embed_openai(texts, model=model)
        self._record_embedding_usage(texts, vectors, usage, model=model, trace_meta=trace_meta)
        return vectors

    def _generate_openai(self, system_prompt: str, user_prompt: str) -> Tuple[str, Dict[str, int]]:
        from openai import OpenAI
        from openai import BadRequestError

        api_key = self._get_chat_api_key()
        client = OpenAI(api_key=api_key, base_url=self._get_chat_base_url())
        system_prompt = self._sanitize_text_for_json(system_prompt)
        user_prompt = self._sanitize_text_for_json(user_prompt)
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        try:
            resp = self._call_with_retries(lambda: client.chat.completions.create(**payload))
        except BadRequestError as e:
            msg = str(e)
            if "max_tokens" in msg and "max_completion_tokens" in msg:
                payload.pop("max_tokens", None)
                payload["max_completion_tokens"] = self.max_tokens
                resp = self._call_with_retries(lambda: client.chat.completions.create(**payload))
            elif "parse the JSON body" in msg or "not valid JSON" in msg:
                # One more conservative retry with very strict ASCII-safe payload sanitization.
                safe_system = self._truncate_text(self._ultra_sanitize_text_for_json(system_prompt))
                safe_user = self._truncate_text(self._ultra_sanitize_text_for_json(user_prompt))
                payload["messages"] = [
                    {"role": "system", "content": safe_system},
                    {"role": "user", "content": safe_user},
                ]
                try:
                    resp = self._call_with_retries(lambda: client.chat.completions.create(**payload))
                except BadRequestError as e2:
                    msg2 = str(e2)
                    if "parse the JSON body" in msg2 or "not valid JSON" in msg2:
                        return "", {
                            "prompt_tokens": self._estimate_tokens(safe_system) + self._estimate_tokens(safe_user),
                            "completion_tokens": 0,
                            "total_tokens": self._estimate_tokens(safe_system) + self._estimate_tokens(safe_user),
                        }
                    raise
            else:
                raise
        text = resp.choices[0].message.content or ""
        usage = {
            "prompt_tokens": int(getattr(resp.usage, "prompt_tokens", 0) or 0),
            "completion_tokens": int(getattr(resp.usage, "completion_tokens", 0) or 0),
            "total_tokens": int(getattr(resp.usage, "total_tokens", 0) or 0),
        }
        if usage["total_tokens"] <= 0:
            usage = self._estimate_usage(system_prompt, user_prompt, text)
        return text, usage

    def _embed_openai(self, texts: List[str], model: str) -> Tuple[List[List[float]], Dict[str, int]]:
        from openai import OpenAI

        api_key = self._get_embedding_api_key()
        client = OpenAI(api_key=api_key, base_url=self._get_embedding_base_url())
        clean_texts = [self._sanitize_text_for_json(t) for t in texts]
        resp = self._call_with_retries(lambda: client.embeddings.create(model=model, input=clean_texts))
        vectors = [list(row.embedding) for row in resp.data]
        usage = {
            "prompt_tokens": int(getattr(resp.usage, "prompt_tokens", 0) or 0),
            "completion_tokens": 0,
            "total_tokens": int(getattr(resp.usage, "total_tokens", 0) or 0),
        }
        if usage["total_tokens"] <= 0:
            usage = {
                "prompt_tokens": sum(self._estimate_tokens(t) for t in texts),
                "completion_tokens": 0,
                "total_tokens": sum(self._estimate_tokens(t) for t in texts),
            }
        return vectors, usage

    def _generate_mock(self, system_prompt: str, user_prompt: str) -> str:
        text = f"{system_prompt}\n{user_prompt}".lower()

        patterns = [
            (r"both hands", "B"),
            (r"punctual", "C"),
            (r"diverse|overgeneral", "B"),
        ]
        for pat, ans in patterns:
            if re.search(pat, text):
                return ans

        return "A"

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        if not text:
            return 0
        return max(1, len(text) // 4)

    def _estimate_usage(self, system_prompt: str, user_prompt: str, output_text: str) -> Dict[str, int]:
        prompt_tokens = self._estimate_tokens(system_prompt) + self._estimate_tokens(user_prompt)
        completion_tokens = self._estimate_tokens(output_text)
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }

    def _record_usage(
        self,
        system_prompt: str,
        user_prompt: str,
        output_text: str,
        usage: Dict[str, int],
        trace_meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        meta = dict(trace_meta or {})
        rec = {
            "ts_utc": datetime.now(timezone.utc).isoformat(),
            "provider": self.provider,
            "model": self.model,
            "stage": str(meta.pop("stage", "unknown")),
            "item_id": str(meta.pop("item_id", "")),
            "prompt_chars": len(system_prompt) + len(user_prompt),
            "output_chars": len(output_text),
            "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
            "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
            "total_tokens": int(usage.get("total_tokens", 0) or 0),
            "meta": meta,
        }
        self.usage_log.append(rec)

    def _call_with_retries(self, fn):
        last_error = None
        for attempt in range(max(1, self.max_api_retries)):
            try:
                return fn()
            except Exception as e:
                last_error = e
                msg = str(e).lower()
                retryable = any(
                    token in msg
                    for token in [
                        "connection error",
                        "connecterror",
                        "timeout",
                        "timed out",
                        "temporarily unavailable",
                        "rate limit",
                        "winerror 10013",
                        "getaddrinfo failed",
                    ]
                )
                if not retryable or attempt >= self.max_api_retries - 1:
                    raise
                sleep_sec = self.api_retry_base_sec * (2 ** attempt)
                time.sleep(min(sleep_sec, 30.0))
        raise last_error

    def _record_embedding_usage(
        self,
        texts: List[str],
        vectors: List[List[float]],
        usage: Dict[str, int],
        model: str,
        trace_meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        meta = dict(trace_meta or {})
        rec = {
            "ts_utc": datetime.now(timezone.utc).isoformat(),
            "provider": self.provider,
            "model": model,
            "stage": str(meta.pop("stage", "embedding")),
            "item_id": str(meta.pop("item_id", "")),
            "prompt_chars": sum(len(t) for t in texts),
            "output_chars": len(vectors) * (len(vectors[0]) if vectors else 0),
            "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
            "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
            "total_tokens": int(usage.get("total_tokens", 0) or 0),
            "meta": {
                "num_inputs": len(texts),
                "embedding_dim": len(vectors[0]) if vectors else 0,
                **meta,
            },
        }
        self.usage_log.append(rec)

    def reset_usage_log(self) -> None:
        self.usage_log = []

    def usage_summary(self) -> Dict[str, Any]:
        summary: Dict[str, Any] = {
            "num_calls": len(self.usage_log),
            "prompt_tokens": sum(int(r.get("prompt_tokens", 0) or 0) for r in self.usage_log),
            "completion_tokens": sum(int(r.get("completion_tokens", 0) or 0) for r in self.usage_log),
            "total_tokens": sum(int(r.get("total_tokens", 0) or 0) for r in self.usage_log),
            "by_stage": {},
        }
        by_stage: Dict[str, Dict[str, int]] = {}
        for r in self.usage_log:
            stage = str(r.get("stage", "unknown"))
            if stage not in by_stage:
                by_stage[stage] = {
                    "num_calls": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                }
            by_stage[stage]["num_calls"] += 1
            by_stage[stage]["prompt_tokens"] += int(r.get("prompt_tokens", 0) or 0)
            by_stage[stage]["completion_tokens"] += int(r.get("completion_tokens", 0) or 0)
            by_stage[stage]["total_tokens"] += int(r.get("total_tokens", 0) or 0)
        summary["by_stage"] = by_stage
        return summary

    def dump_usage_log(self, path: str) -> None:
        out_dir = os.path.dirname(path) or "."
        os.makedirs(out_dir, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for row in self.usage_log:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_mcq_answer(text: str, valid_letters: Optional[Set[str]] = None) -> str:
    if not text:
        return ""

    up = text.upper()
    allowed = {x.upper() for x in valid_letters} if valid_letters else None

    def _ok(letter: str) -> bool:
        return bool(letter) and (allowed is None or letter in allowed)

    lines = [ln.strip() for ln in up.splitlines() if ln.strip()]
    tail = list(reversed(lines[-8:])) if lines else []

    line_patterns = [
        r"^(?:FINAL\s+ANSWER|ANSWER|OPTION|CHOICE)\s*[:\-]?\s*\(?([A-Z])\)?$",
        r"^\(?([A-Z])\)?[\.\)]?$",
    ]
    for ln in tail:
        for pat in line_patterns:
            m = re.match(pat, ln)
            if m and _ok(m.group(1)):
                return m.group(1)

    global_patterns = [
        r"\b(?:FINAL\s+ANSWER|ANSWER|OPTION|CHOICE)\s*[:\-]?\s*\(?([A-Z])\)?\b",
        r"\(([A-Z])\)",
        r"\b([A-Z])\b",
    ]
    for pat in global_patterns:
        for m in re.finditer(pat, up):
            letter = m.group(1)
            if _ok(letter):
                return letter

    return ""
