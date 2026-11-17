"""Generators for local runs.

Two generators, one interface (generate(input_prompt) -> str):

1. OpenRouterGenerator — calls a free OpenRouter model with the exact
   challenge prompt. Stand-in ONLY: the challenge fixes the generator
   to google/gemma-3n-E4B-it, which is not on OpenRouter; local runs
   use nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free (the model
   the 12/12 baseline used; gemma-4 free is 429-limited upstream)
   and the report always records which model actually generated. API
   key is read at runtime from the OpenRouter credentials file (key
   "key12"), never printed or stored in this repo.

2. LexicalAnswerer — no model at all: picks the option whose text best
   overlaps the retrieved passages (TF overlap). A floor that isolates
   retrieval quality from generation quality.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import requests

from .questions import ChallengeQuestion
from .retrieval import RetrievedChunk

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
# Stand-in for local runs only. The challenge generator is gemma-3n-E4B-it.
# gemma-4-26b-a4b-it:free (the earlier default) is 429-limited upstream
# as of 2026-11-17, so the default is the model the 12/12 baseline
# actually used: nemotron-3-nano-omni (reasoning — use max_tokens 512).
DEFAULT_FREE_MODEL = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free"
CREDENTIALS_PATH = Path.home() / "workspace/makefun/openrouter-credentials.json"


def load_openrouter_key(name: str = "key12") -> str:
    """Read an OpenRouter API key by name from the credentials file."""
    data = json.loads(CREDENTIALS_PATH.read_text(encoding="utf-8"))
    for entry in data.get("keys", []):
        if entry.get("name") == name:
            return entry.get("full") or entry.get("key") or ""
    raise KeyError(f"OpenRouter key {name!r} not found in credentials file")


class OpenRouterGenerator:
    def __init__(
        self,
        model: str = DEFAULT_FREE_MODEL,
        api_key: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 8,
        timeout: int = 120,
    ):
        self.model = model
        self.api_key = api_key or load_openrouter_key()
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

    def generate(self, input_prompt: str) -> str:
        import time

        system, _, user = input_prompt.partition("\n\n")
        last_error: Exception | None = None
        # Circuit breaker: three consecutive rate-limit responses means
        # the free quota is exhausted for now — fail fast instead of
        # grinding through the remaining questions on backoff timers.
        consecutive_429 = 0
        for attempt, wait in enumerate([0, 15, 45, 120]):
            if wait:
                time.sleep(wait)
            try:
                resp = requests.post(
                    OPENROUTER_URL,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.model,
                        "messages": [
                            {"role": "system", "content": system},
                            {"role": "user", "content": user},
                        ],
                        "temperature": self.temperature,
                        "max_tokens": self.max_tokens,
                    },
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                msg = resp.json()["choices"][0]["message"]
                content = msg.get("content") or ""
                if content:
                    return content
                # Reasoning models (e.g. nemotron-3-nano, north-mini-code)
                # spend the token budget thinking and leave content null.
                reasoning = msg.get("reasoning") or ""
                return reasoning
            except requests.exceptions.HTTPError as exc:
                last_error = exc
                # Retry only on rate limits / transient server errors.
                status = exc.response.status_code if exc.response is not None else 0
                if status == 429:
                    consecutive_429 += 1
                    if consecutive_429 >= 3:
                        raise RuntimeError(
                            "OpenRouter free quota exhausted (3 consecutive 429s); "
                            "stop and retry later or rotate the stand-in model"
                        ) from exc
                else:
                    consecutive_429 = 0
                if status not in (429, 502, 503) or attempt == 3:
                    raise
        raise last_error  # pragma: no cover - loop always raises first


class LexicalAnswerer:
    """Pick the option whose words overlap the retrieved passages most."""

    def __init__(self, chunks: list[RetrievedChunk]):
        self.chunks = chunks

    def generate(self, question: ChallengeQuestion) -> str:
        if question.is_true_false:
            return self._answer_tf(question)
        passage_text = " ".join(c.text.lower() for c in self.chunks)
        best_letter, best_score = "", -1.0
        for letter, text in question.options.items():
            words = set(re.findall(r"[a-z0-9]+", text.lower()))
            if not words:
                continue
            hits = sum(1 for w in words if w in passage_text)
            score = hits / len(words)
            if score > best_score:
                best_letter, best_score = letter, score
        return best_letter

    def _answer_tf(self, question: ChallengeQuestion) -> str:
        # No lexical signal for bare true/false statements: default TRUE.
        # This is a coin-flip floor (documents the retrieval-only value);
        # the LLM generator answers TF from the passages instead.
        return "TRUE"
