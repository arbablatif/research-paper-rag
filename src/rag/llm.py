"""Provider-agnostic LLM client with a disk cache.

Every response is cached under .cache/llm keyed by (provider, model, prompt), so
re-running the app or the eval on the same question never calls the API twice.
"""
import hashlib
import json
import os
import re

from .config import settings
from .prompts import REFUSAL


class LLM:
    def __init__(self, provider: str | None = None):
        self.provider = (provider or settings.provider).lower()
        self.model = {
            "gemini": settings.gemini_model,
            "anthropic": settings.anthropic_model,
            "openai": settings.openai_model,
            "mock": "mock",
        }[self.provider]
        settings.cache_dir.mkdir(parents=True, exist_ok=True)

    # ---- cache ----
    def _cache_path(self, prompt: str):
        key = hashlib.sha256(f"{self.provider}|{self.model}|{prompt}".encode()).hexdigest()
        return settings.cache_dir / f"{key}.json"

    def generate(self, prompt: str) -> str:
        if self.provider == "mock":
            return self._mock(prompt)  # mock is deterministic; no need to cache
        path = self._cache_path(prompt)
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))["response"]
        text = getattr(self, f"_{self.provider}")(prompt)
        path.write_text(json.dumps({"provider": self.provider, "model": self.model,
                                    "response": text}), encoding="utf-8")
        return text

    # ---- providers (SDKs imported lazily so only the chosen one must be installed) ----
    def _gemini(self, prompt: str) -> str:
        from google import genai
        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        resp = client.models.generate_content(
            model=self.model, contents=prompt, config={"temperature": 0})
        return (resp.text or "").strip()

    def _anthropic(self, prompt: str) -> str:
        import anthropic
        client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
        resp = client.messages.create(model=self.model, max_tokens=512,
                                      messages=[{"role": "user", "content": prompt}])
        return "".join(b.text for b in resp.content if b.type == "text").strip()

    def _openai(self, prompt: str) -> str:
        from openai import OpenAI
        client = OpenAI()  # reads OPENAI_API_KEY
        resp = client.chat.completions.create(model=self.model,
                                              messages=[{"role": "user", "content": prompt}])
        return resp.choices[0].message.content.strip()

    @staticmethod
    def _mock(prompt: str) -> str:
        """Offline stand-in: quote the first passage's opening sentence with its citation."""
        m = re.search(r"\[1\] \((.+?), p\. (\d+)\)\n(.+?)(?:\n\n\[2\]|\n\nQuestion:)", prompt, re.S)
        if not m:
            return REFUSAL
        doc, page, text = m.groups()
        sentence = re.split(r"(?<=[.!?])\s", text.strip().replace("\n", " "))[0][:300]
        return f"{sentence} [{doc}, p. {page}]"
