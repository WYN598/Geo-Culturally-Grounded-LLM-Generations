import os
import re
from typing import Optional, Set

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

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAIAPI")
        if self.provider == "openai":
            if not api_key:
                raise RuntimeError("LLM provider is 'openai' but OPENAI_API_KEY/OPENAIAPI is not set.")
            return self._generate_openai(system_prompt, user_prompt)
        if self.provider == "mock":
            return self._generate_mock(system_prompt, user_prompt)
        raise ValueError(f"Unsupported LLM provider: {self.provider}")

    def _generate_openai(self, system_prompt: str, user_prompt: str) -> str:
        from openai import OpenAI

        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAIAPI")
        client = OpenAI(api_key=api_key, base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
        resp = client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return resp.choices[0].message.content or ""

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
