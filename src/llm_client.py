import json
import os
import re
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

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        trace_meta: Optional[Dict[str, Any]] = None,
    ) -> str:
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAIAPI")
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

    def _generate_openai(self, system_prompt: str, user_prompt: str) -> Tuple[str, Dict[str, int]]:
        from openai import OpenAI
        from openai import BadRequestError

        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAIAPI")
        client = OpenAI(api_key=api_key, base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
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
            resp = client.chat.completions.create(**payload)
        except BadRequestError as e:
            msg = str(e)
            if "max_tokens" in msg and "max_completion_tokens" in msg:
                payload.pop("max_tokens", None)
                payload["max_completion_tokens"] = self.max_tokens
                resp = client.chat.completions.create(**payload)
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
