"""Scoring — the exact challenge formulas.

Score = 0.7 * accuracy + 0.3 * token_efficiency
  accuracy         = correct / n_questions (exact match; empty/invalid = wrong)
  token_efficiency = 100 * B / (B + T), B = 850 * n_questions,
                     T = total cl100k_base tokens of input_prompt
Minimum accuracy is 80% on the whole set; below that the submission
is disqualified regardless of token efficiency.
"""
from __future__ import annotations

from dataclasses import dataclass

BASELINE_TOKENS_PER_QUESTION = 850
MIN_ACCURACY = 0.80
ACCURACY_WEIGHT = 0.7
EFFICIENCY_WEIGHT = 0.3


@dataclass
class Score:
    n_questions: int
    n_correct: int
    accuracy: float  # 0..1
    total_prompt_tokens: int
    token_efficiency: float  # 0..100
    overall: float  # 0..100
    passes_accuracy_gate: bool

    def as_dict(self) -> dict:
        return {
            "n_questions": self.n_questions,
            "n_correct": self.n_correct,
            "accuracy": round(self.accuracy, 4),
            "total_prompt_tokens": self.total_prompt_tokens,
            "token_efficiency": round(self.token_efficiency, 2),
            "overall": round(self.overall, 2),
            "passes_accuracy_gate": self.passes_accuracy_gate,
        }


def token_efficiency(total_prompt_tokens: int, n_questions: int) -> float:
    baseline = BASELINE_TOKENS_PER_QUESTION * n_questions
    return 100.0 * baseline / (baseline + total_prompt_tokens)


def score_submission(
    answers: dict[str, str],
    reference: dict[str, str],
    total_prompt_tokens: int,
) -> Score:
    n = len(reference)
    correct = sum(1 for qid, ref in reference.items() if answers.get(qid) == ref)
    accuracy = correct / n if n else 0.0
    eff = token_efficiency(total_prompt_tokens, n)
    overall = 100.0 * (ACCURACY_WEIGHT * accuracy + EFFICIENCY_WEIGHT * eff / 100.0)
    return Score(
        n_questions=n,
        n_correct=correct,
        accuracy=accuracy,
        total_prompt_tokens=total_prompt_tokens,
        token_efficiency=eff,
        overall=overall,
        passes_accuracy_gate=accuracy >= MIN_ACCURACY,
    )
