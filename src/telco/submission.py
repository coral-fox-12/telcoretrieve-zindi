"""Submission file: write, validate (mirrors official `challenge check`)
and parse generator answers.

Columns: ID,input_prompt,answer. The file must be UTF-8 without BOM,
LF line endings, one row per question ID, valid answers only
(option letter for MC, TRUE/FALSE for TF), and every input_prompt at
most 4096 tokens. `check` mirrors the official validator and exits
non-zero (raises SubmissionError) on any violation.
"""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

from .prompt import MAX_PROMPT_TOKENS
from .questions import ChallengeQuestion
from .tokens import count_tokens

HEADER = ["ID", "input_prompt", "answer"]


class SubmissionError(ValueError):
    pass


@dataclass
class SubmissionRow:
    id: str
    input_prompt: str
    answer: str


def parse_answer(raw: str, question: ChallengeQuestion) -> str:
    """Extract the scored answer from a generator's raw output.

    The generator is instructed to reply with only the letter / TRUE /
    FALSE, but outputs drift: leading prose, quotes, lowercase, a
    trailing period. We take the first standalone valid token; failing
    that, the first character if it is a valid option letter.
    Returns "" when nothing valid is found (counts as wrong).
    """
    text = (raw or "").strip()
    if not text:
        return ""
    upper = text.upper()
    if question.is_true_false:
        found = re.findall(r"\b(TRUE|FALSE)\b", upper)
        return found[-1] if found else ""
    valid = set(question.valid_answers)
    # Last standalone letter token: for a plain "B" or "The answer is C."
    # first == last; for a reasoning trace the conclusion comes last, so
    # the last valid token is the answer, not an early mention.
    found = [m.group(1) for m in re.finditer(r"\b([A-Z])\b", upper) if m.group(1) in valid]
    if found:
        return found[-1]
    # first character fallback
    if upper and upper[0] in valid:
        return upper[0]
    return ""


def write_submission(rows: list[SubmissionRow], path: str | Path) -> None:
    path = Path(path)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(HEADER)
        for row in rows:
            writer.writerow([row.id, row.input_prompt, row.answer])


def read_submission(path: str | Path) -> list[SubmissionRow]:
    import io

    with open(path, newline="", encoding="utf-8") as fh:
        raw = fh.read()
    if raw.startswith("\ufeff"):
        raise SubmissionError("submission has a UTF-8 BOM")
    if "\r" in raw:
        raise SubmissionError("submission has CR characters (must be LF only)")
    rows = []
    # StringIO (not splitlines): quoted fields may contain newlines.
    for row in csv.DictReader(io.StringIO(raw)):
        rows.append(
            SubmissionRow(
                id=row["ID"], input_prompt=row["input_prompt"], answer=row["answer"]
            )
        )
    return rows


def check_submission(
    path: str | Path,
    questions: list[ChallengeQuestion],
    max_prompt_tokens: int = MAX_PROMPT_TOKENS,
) -> list[SubmissionRow]:
    """Validate a submission file against the questions. Returns the rows.

    Raises SubmissionError on: wrong header, missing/extra/duplicate IDs,
    invalid answer, prompt over the token cap, non-UTF-8 content.
    """
    rows = read_submission(path)
    by_id = {q.id: q for q in questions}
    seen: set[str] = set()
    for row in rows:
        if row.id in seen:
            raise SubmissionError(f"duplicate ID {row.id}")
        seen.add(row.id)
        q = by_id.get(row.id)
        if q is None:
            raise SubmissionError(f"unknown ID {row.id}")
        if row.answer not in q.valid_answers:
            raise SubmissionError(
                f"ID {row.id}: invalid answer {row.answer!r} "
                f"(valid: {q.valid_answers})"
            )
        n = count_tokens(row.input_prompt)
        if n > max_prompt_tokens:
            raise SubmissionError(
                f"ID {row.id}: prompt is {n} tokens, over the "
                f"{max_prompt_tokens} cap"
            )
    missing = set(by_id) - seen
    if missing:
        raise SubmissionError(f"missing IDs: {sorted(missing)[:5]}")
    return rows
