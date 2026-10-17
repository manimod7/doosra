"""Language-model providers behind one tiny interface: complete(system, prompt, max_tokens) -> text.

Picked from environment variables, tried in order, first success wins:
  GEMINI_API_KEY     Google Gemini API (AI Studio key; the free tier, or the
                     $10/month Cloud credit that comes with Google AI Pro)
  GROQ_API_KEY       Groq (free tier; does not train on your prompts)
  ANTHROPIC_API_KEY  Claude API (billed separately from a Claude Pro plan)
DOOSRA_LLM_ORDER can reorder them, e.g. "groq,gemini". Model names can be
overridden with DOOSRA_GEMINI_MODEL, DOOSRA_GROQ_MODEL, DOOSRA_CLAUDE_MODEL.

Nothing here ever produces a probability: explain.py only lets these models
reword numbers the simulator already computed.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import httpx

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"


class LLMError(RuntimeError):
    pass


@dataclass
class GeminiClient:
    api_key: str
    model: str = field(default_factory=lambda: os.environ.get("DOOSRA_GEMINI_MODEL", "gemini-flash-latest"))
    transport: httpx.BaseTransport | None = None
    name: str = "gemini"

    def complete(self, system: str, prompt: str, max_tokens: int) -> str:
        body = {
            "system_instruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"maxOutputTokens": max_tokens, "temperature": 0.3},
        }
        with httpx.Client(timeout=30, transport=self.transport) as c:
            r = c.post(GEMINI_URL.format(model=self.model), json=body, headers={"x-goog-api-key": self.api_key})
        if r.status_code >= 400:
            raise LLMError(f"gemini {r.status_code}")
        try:
            parts = r.json()["candidates"][0]["content"]["parts"]
        except (KeyError, IndexError, ValueError) as e:
            raise LLMError("gemini returned no text") from e
        return "".join(p.get("text", "") for p in parts).strip()


@dataclass
class GroqClient:
    api_key: str
    model: str = field(default_factory=lambda: os.environ.get("DOOSRA_GROQ_MODEL", "openai/gpt-oss-20b"))
    transport: httpx.BaseTransport | None = None
    name: str = "groq"

    def complete(self, system: str, prompt: str, max_tokens: int) -> str:
        body = {"model": self.model, "max_tokens": max_tokens, "temperature": 0.3,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}]}
        with httpx.Client(timeout=30, transport=self.transport) as c:
            r = c.post(GROQ_URL, json=body, headers={"authorization": f"Bearer {self.api_key}"})
        if r.status_code >= 400:
            raise LLMError(f"groq {r.status_code}")
        try:
            return (r.json()["choices"][0]["message"]["content"] or "").strip()
        except (KeyError, IndexError, ValueError) as e:
            raise LLMError("groq returned no text") from e


@dataclass
class ClaudeClient:
    api_key: str
    model: str = field(default_factory=lambda: os.environ.get("DOOSRA_CLAUDE_MODEL", "claude-haiku-5-5"))
    name: str = "claude"

    def complete(self, system: str, prompt: str, max_tokens: int) -> str:
        import anthropic  # optional dependency

        msg = anthropic.Anthropic(api_key=self.api_key).messages.create(
            model=self.model, max_tokens=max_tokens, system=system, messages=[{"role": "user", "content": prompt}])
        return "".join(getattr(b, "text", "") for b in msg.content).strip()


@dataclass
class Chain:
    """Try each provider in turn; remember which one answered."""
    clients: list
    last_used: str | None = None

    @property
    def name(self) -> str:
        return "+".join(c.name for c in self.clients)

    def complete(self, system: str, prompt: str, max_tokens: int) -> str:
        errors = []
        for c in self.clients:
            try:
                text = c.complete(system, prompt, max_tokens)
                if text:
                    self.last_used = c.name
                    return text
                errors.append(f"{c.name}: empty")
            except Exception as e:  # any provider failure falls through to the next
                errors.append(f"{c.name}: {type(e).__name__}")
        raise LLMError("; ".join(errors) or "no providers")


def from_env(env: dict | None = None) -> Chain | None:
    env = os.environ if env is None else env
    makers = {
        "gemini": lambda: GeminiClient(env["GEMINI_API_KEY"]) if env.get("GEMINI_API_KEY") else None,
        "groq": lambda: GroqClient(env["GROQ_API_KEY"]) if env.get("GROQ_API_KEY") else None,
        "claude": lambda: ClaudeClient(env["ANTHROPIC_API_KEY"]) if env.get("ANTHROPIC_API_KEY") else None,
    }
    order = [x.strip() for x in env.get("DOOSRA_LLM_ORDER", "gemini,groq,claude").split(",") if x.strip() in makers]
    clients = [c for c in (makers[k]() for k in order) if c is not None]
    return Chain(clients) if clients else None
