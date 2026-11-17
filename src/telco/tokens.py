"""Token counting: cl100k_base, exactly as the challenge counts prompts.

The official pipeline counts `input_prompt` with tiktoken cl100k_base.
We use tiktoken when installed (it is: pip install tiktoken); the
fallback estimator exists only so `check`/`score` never crash on a
machine without it, and it is clearly marked approximate.
"""
from __future__ import annotations

ENCODING = "cl100k_base"

try:
    import tiktoken

    _ENC = tiktoken.get_encoding(ENCODING)
    HAVE_TIKTOKEN = True
except Exception:  # pragma: no cover - tiktoken is installed in practice
    _ENC = None
    HAVE_TIKTOKEN = False


def count_tokens(text: str) -> int:
    """Number of cl100k_base tokens in `text`."""
    if HAVE_TIKTOKEN:
        return len(_ENC.encode(text))
    # Fallback: ~4 chars per token (OpenAI rule of thumb). Approximate only.
    return max(1, len(text) // 4)
