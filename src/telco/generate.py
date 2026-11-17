"""Generators for local runs.

Three generators, one interface (generate(input_prompt) -> str):

1. OpenRouterGenerator — calls a free OpenRouter model with the exact
   challenge prompt. Stand-in ONLY: the challenge fixes the generator
   to google/gemma-3n-E4B-it, which is not on OpenRouter; local runs
   use nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free (the model
   the 12/12 baseline used; gemma-4 free is 429-limited upstream)
   and the report always records which model actually generated. API
   key is read at runtime from the OpenRouter credentials file (key
   "key12"), never printed or stored in this repo.

2. GeminiGenerator — calls a model directly via the Gemini API
   (generativelanguage.googleapis.com). Also a stand-in ONLY (the
   fixed generator is gemma-3n-E4B-it). The intended model,
   gemini-2.5-flash, 404s for these new keys ("no longer available
   to new users"), and the Gemini 3.x flash models hang from this VM,
   so the working stand-in is google/gemma-4-26b-a4b-it — same model
   family as the fixed generator, served through the same API. It
   emits thought parts before its answer (like the nemotron reasoning
   stand-in) and is capacity-limited on the free tier (~70% of
   requests succeed first try; 503s retry with backoff). API keys
   are the user's own Google AI Studio keys, read at runtime from
   the Gemini credentials file (gemini_key_1, gemini_key_2), never
   printed or stored in this repo. Two keys let the sweep split
   load across two free-tier quotas.

3. LexicalAnswerer — no model at all: picks the option whose text best
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
GEMINI_URL_TEMPLATE = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
)
# Stand-in for local runs only. The challenge generator is gemma-3n-E4B-it.
# gemma-4-26b-a4b-it:free (the earlier default) is 429-limited upstream
# as of 2026-11-17, so the default is the model the 12/12 baseline
# actually used: nemotron-3-nano-omni (reasoning — use max_tokens 512).
DEFAULT_FREE_MODEL = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free"
# Direct Gemini-API stand-in. gemini-2.5-flash was the first choice but
# returns 404 for these new keys ("no longer available to new users");
# the Gemini 3.x flash models hang from this VM (read timeouts). The
# working stand-in is gemma-4-26b-a4b-it (verified live 2026-11-17:
# ~7/10 requests succeed first try, the rest 503 and retry).
DEFAULT_GEMINI_MODEL = "gemma-4-26b-a4b-it"
CREDENTIALS_PATH = Path.home() / "workspace/makefun/openrouter-credentials.json"
GEMINI_CREDENTIALS_PATH = Path.home() / "workspace/makefun/gemini-credentials.json"


def load_openrouter_key(name: str = "key12") -> str:
    """Read an OpenRouter API key by name from the credentials file."""
    data = json.loads(CREDENTIALS_PATH.read_text(encoding="utf-8"))
    for entry in data.get("keys", []):
        if entry.get("name") == name:
            return entry.get("full") or entry.get("key") or ""
    raise KeyError(f"OpenRouter key {name!r} not found in credentials file")


def load_gemini_key(name: str = "gemini_key_1") -> str:
    """Read a Gemini API key by name from the Gemini credentials file.

    The file holds plain string values (gemini_key_1, gemini_key_2);
    a dict value with a "key"/"full" field is accepted too.
    """
    data = json.loads(GEMINI_CREDENTIALS_PATH.read_text(encoding="utf-8"))
    value = data.get(name)
    if isinstance(value, dict):
        value = value.get("full") or value.get("key") or ""
    if isinstance(value, str) and value:
        return value
    raise KeyError(f"Gemini key {name!r} not found in credentials file")


def parse_gemini_response(data: dict) -> str:
    """Extract the model text from a Gemini generateContent response.

    Returns "" when the response has no text (safety block, empty
    candidates). Thought parts (thinking enabled) are skipped — only
    the answer text is returned.
    """
    candidates = data.get("candidates") or []
    if not candidates:
        return ""
    parts = (candidates[0].get("content") or {}).get("parts") or []
    return "".join(
        p.get("text", "")
        for p in parts
        if isinstance(p, dict) and not p.get("thought")
    )


class GeminiGenerator:
    """Direct Gemini-API generator (stand-in only, see module docstring).

    The model (gemma-4-26b-a4b-it) emits thought parts before its
    answer, like the nemotron reasoning stand-in, so max_tokens is
    512 and parse_answer takes the last valid token. Temperature 0.
    A 2-second pace between requests plus retries on 429/500/503
    with backoff handles the free tier's capacity 503s; the same
    3-consecutive-429s circuit breaker as the OpenRouter path stops
    a run when the quota itself is exhausted.
    """

    def __init__(
        self,
        model: str = DEFAULT_GEMINI_MODEL,
        api_key: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
        timeout: int = 120,
        pace_seconds: float = 2.0,
    ):
        self.model = model
        self.api_key = api_key or load_gemini_key()
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.pace_seconds = pace_seconds

    def generate(self, input_prompt: str) -> str:
        import time

        system, _, user = input_prompt.partition("\n\n")
        last_error: Exception | None = None
        consecutive_429 = 0
        for attempt, wait in enumerate([0, 15, 45, 120]):
            if wait:
                time.sleep(wait)
            elif self.pace_seconds:
                time.sleep(self.pace_seconds)
            try:
                resp = requests.post(
                    GEMINI_URL_TEMPLATE.format(model=self.model),
                    headers={
                        "x-goog-api-key": self.api_key,
                        "Content-Type": "application/json",
                    },
                    json={
                        "systemInstruction": {"parts": [{"text": system}]},
                        "contents": [{"parts": [{"text": user}]}],
                        "generationConfig": {
                            "temperature": self.temperature,
                            "maxOutputTokens": self.max_tokens,
                        },
                    },
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                return parse_gemini_response(resp.json())
            except requests.exceptions.HTTPError as exc:
                last_error = exc
                status = exc.response.status_code if exc.response is not None else 0
                if status == 429:
                    consecutive_429 += 1
                    if consecutive_429 >= 3:
                        raise RuntimeError(
                            "Gemini free quota exhausted (3 consecutive 429s); "
                            "stop and retry later or rotate the key"
                        ) from exc
                else:
                    consecutive_429 = 0
                if status not in (429, 500, 503) or attempt == 3:
                    raise
        raise last_error  # pragma: no cover - loop always raises first


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
