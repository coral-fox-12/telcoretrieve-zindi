"""Prompt-builder hooks for the TelcoRetrieve phase-2 Kaggle GPU run.

budget850: budgeted context fill to 850 passage tokens. Mirrors the local
scaffold's build_prompt_budgeted (and upstream `evaluate`
--context-token-budget): keep passages in rank order while their cumulative
cl100k_base token count stays within the budget; always keep at least the
top-ranked passage; then build the prompt (which also enforces the official
4096-token cap). The budget counts passage text only — system message,
question and options are fixed overhead counted in the prompt total.

Usage (run export from the directory holding this file):
    open-telco-rag challenge export ... --prompt-builder my_hooks:budget850
"""
from __future__ import annotations

from open_telco_rag.challenge.prompt import build_prompt
from open_telco_rag.challenge.tokens import count_tokens

BUDGET_TOKENS = 850


def budget850(question, passages, max_prompt_tokens):
    kept = []
    used = 0
    for passage in passages:
        n = count_tokens(passage.text)
        if kept and used + n > BUDGET_TOKENS:
            break
        kept.append(passage)
        used += n
    return build_prompt(question, kept, max_prompt_tokens)
