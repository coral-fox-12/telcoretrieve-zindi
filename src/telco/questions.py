"""Question loading: Zindi Test.csv and the local validation JSON.

Test.csv columns: ID,question,options — options is a JSON object
mapping option letter to option text for multiple choice, {} for
true/false. No answers are shipped with the challenge.
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ChallengeQuestion:
    id: str
    question: str
    options: dict[str, str] = field(default_factory=dict)
    answer: str = ""  # reference answer, only for local validation sets
    evidence_chunk: str = ""

    @property
    def is_true_false(self) -> bool:
        return not self.options

    @property
    def valid_answers(self) -> list[str]:
        if self.is_true_false:
            return ["TRUE", "FALSE"]
        return sorted(self.options.keys())


def load_test_csv(path: str | Path) -> list[ChallengeQuestion]:
    questions = []
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            options = json.loads(row["options"]) if row["options"] else {}
            questions.append(
                ChallengeQuestion(id=row["ID"], question=row["question"], options=options)
            )
    return questions


def load_validation_json(path: str | Path) -> list[ChallengeQuestion]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    questions = []
    for item in data["questions"]:
        questions.append(
            ChallengeQuestion(
                id=item["id"],
                question=item["question"],
                options=item.get("options", {}),
                answer=item.get("answer", ""),
                evidence_chunk=item.get("evidence_chunk", ""),
            )
        )
    return questions
