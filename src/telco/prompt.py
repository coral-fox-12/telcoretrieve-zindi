"""Prompt construction — mirrors open_telco_rag/challenge/prompt.py exactly.

- SYSTEM_MESSAGE: "You answer 3GPP questions from the provided context.
  Reply with only the answer."
- MC instruction: "Answer with only the letter of the correct option."
- TF instruction: "Answer with only TRUE or FALSE."
- User message: "Passages:\n[1] Spec: X | Release: Y\n<text>\n\n..."
  then "Question: ...", then "Options:\nA. ...", then the instruction.
  No passages -> context is the literal string "None".
- input_prompt = system + "\\n\\n" + user.
- build_prompt drops the lowest-ranked passages until the prompt fits
  max_prompt_tokens (default 4096); raises PromptTooLong if even the
  bare question does not fit.
"""
from __future__ import annotations

from dataclasses import dataclass

from .questions import ChallengeQuestion
from .tokens import count_tokens

SYSTEM_MESSAGE = "You answer 3GPP questions from the provided context. Reply with only the answer."
MULTIPLE_CHOICE_INSTRUCTION = "Answer with only the letter of the correct option."
TRUE_FALSE_INSTRUCTION = "Answer with only TRUE or FALSE."

MAX_PROMPT_TOKENS = 4096


class PromptTooLong(ValueError):
    pass


@dataclass
class Passage:
    text: str
    spec: str = ""
    release: str = ""


@dataclass
class Prompt:
    input_prompt: str
    n_tokens: int
    n_passages: int


def _passage(number: int, passage: Passage) -> str:
    label = " | ".join(
        part
        for part in (
            f"Spec: {passage.spec}" if passage.spec else "",
            f"Release: {passage.release}" if passage.release else "",
        )
        if part
    )
    return f"[{number}] {label}\n{passage.text}" if label else f"[{number}]\n{passage.text}"


def _user_message(question: ChallengeQuestion, passages: list[Passage]) -> str:
    context = "\n\n".join(_passage(i, p) for i, p in enumerate(passages, start=1)) or "None"
    parts = [f"Passages:\n{context}", f"Question: {question.question}"]
    if question.is_true_false:
        parts.append(TRUE_FALSE_INSTRUCTION)
    else:
        parts.append(
            "Options:\n"
            + "\n".join(f"{letter}. {text}" for letter, text in question.options.items())
        )
        parts.append(MULTIPLE_CHOICE_INSTRUCTION)
    return "\n\n".join(parts)


def _assemble(question: ChallengeQuestion, passages: list[Passage]) -> Prompt:
    user = _user_message(question, passages)
    input_prompt = f"{SYSTEM_MESSAGE}\n\n{user}"
    return Prompt(input_prompt, count_tokens(input_prompt), len(passages))


def build_prompt(
    question: ChallengeQuestion,
    passages: list[Passage],
    max_prompt_tokens: int | None = MAX_PROMPT_TOKENS,
) -> Prompt:
    """Messages and input_prompt from one place; drop lowest-ranked passages to fit."""
    for kept in range(len(passages), -1, -1):
        prompt = _assemble(question, passages[:kept])
        if max_prompt_tokens is None or prompt.n_tokens <= max_prompt_tokens:
            return prompt
    raise PromptTooLong(
        f"question {question.id}: {prompt.n_tokens} tokens without any passage, "
        f"cap is {max_prompt_tokens}"
    )


def build_prompt_budgeted(
    question: ChallengeQuestion,
    passages: list[Passage],
    context_budget_tokens: int,
    max_prompt_tokens: int | None = MAX_PROMPT_TOKENS,
) -> Prompt:
    """Budgeted context fill (upstream --context-token-budget mode).

    Keep passages in rank order while their cumulative token count stays
    within the context budget; always keep at least the top-ranked
    passage. The budget counts passage text only (system message +
    question + options are fixed overhead, counted in the total).
    """
    kept: list[Passage] = []
    used = 0
    for passage in passages:
        n = count_tokens(passage.text)
        if kept and used + n > context_budget_tokens:
            break
        kept.append(passage)
        used += n
    prompt = _assemble(question, kept)
    if max_prompt_tokens is not None and prompt.n_tokens > max_prompt_tokens:
        # Budgeted fill should stay far under the cap; fall back to the
        # drop-lowest behaviour if a pathological question overflows.
        return build_prompt(question, kept, max_prompt_tokens)
    return prompt
