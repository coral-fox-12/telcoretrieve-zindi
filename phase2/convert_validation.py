"""Convert the local 60-question validation set to the official Test CSV
format for `open-telco-rag challenge export`, plus the reference answers
CSV for `challenge score`.

Input : ../data/validation_questions_v2.json  ({"questions": [...]})
Output: validation.csv          (ID,question,options) — options is a JSON
                                  object for MC, empty for true/false
        validation_answers.csv  (ID,answer)
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def main() -> None:
    data = json.loads((ROOT / "data/validation_questions_v2.json").read_text(encoding="utf-8"))
    questions = data["questions"]
    assert len(questions) == 60, f"expected 60 questions, got {len(questions)}"
    ids = [q["id"] for q in questions]
    assert len(set(ids)) == len(ids), "duplicate question ids"

    with open(HERE / "validation.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, quoting=csv.QUOTE_ALL, lineterminator="\n")
        w.writerow(["ID", "question", "options"])
        for q in questions:
            options = q.get("options") or {}
            w.writerow([q["id"], q["question"], json.dumps(options, ensure_ascii=False) if options else ""])

    with open(HERE / "validation_answers.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, quoting=csv.QUOTE_ALL, lineterminator="\n")
        w.writerow(["ID", "answer"])
        for q in questions:
            assert q.get("answer"), f"{q['id']}: missing answer"
            w.writerow([q["id"], q["answer"]])

    n_mc = sum(1 for q in questions if q.get("options"))
    print(f"wrote validation.csv + validation_answers.csv: {len(questions)} questions ({n_mc} MC, {len(questions) - n_mc} TF)")


if __name__ == "__main__":
    main()
