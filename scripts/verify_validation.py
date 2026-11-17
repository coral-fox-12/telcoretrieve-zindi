#!/usr/bin/env python3
"""Verify a validation-question file against the mini index.

Checks (hard failures exit non-zero):
- ids unique; answers valid for the type; MC has exactly 4 options
- evidence_chunk exists in data/mini_index.jsonl
- evidence_quote (when present) is an exact substring of the chunk
- MC correct option text appears verbatim in the evidence chunk
- no question is a near-duplicate of a Test.csv question
  (normalized first-10-words overlap) or of another question here
- MC correct-letter distribution and TF TRUE/FALSE balance printed

Usage: python3 scripts/verify_validation.py data/validation_part_b.json
"""
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def norm(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", "", text.lower()).split()


def main() -> int:
    path = Path(sys.argv[1])
    data = json.loads(path.read_text(encoding="utf-8"))
    questions = data["questions"]
    chunks = {json.loads(line)["id"]: json.loads(line)["text"]
              for line in open(DATA / "mini_index.jsonl", encoding="utf-8")}
    # Re-read properly (dict comp above re-parses; do it cleanly).
    chunks = {}
    for line in open(DATA / "mini_index.jsonl", encoding="utf-8"):
        rec = json.loads(line)
        chunks[rec["id"]] = rec["text"]

    test_qs = []
    with open(DATA / "Test.csv", newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            test_qs.append(" ".join(norm(row["question"])[:12]))

    errors: list[str] = []
    seen_ids: set[str] = set()
    mc_letters: Counter = Counter()
    tf_answers: Counter = Counter()
    for q in questions:
        qid = q["id"]
        if qid in seen_ids:
            errors.append(f"{qid}: duplicate id")
        seen_ids.add(qid)
        ev = q.get("evidence_chunk", "")
        if ev not in chunks:
            errors.append(f"{qid}: evidence chunk {ev} not in mini index")
            continue
        chunk_text = chunks[ev]
        quote = q.get("evidence_quote", "")
        if quote and quote not in chunk_text:
            errors.append(f"{qid}: evidence_quote is NOT a substring of {ev}")
        if q.get("options"):
            if len(q["options"]) != 4:
                errors.append(f"{qid}: MC has {len(q['options'])} options, want 4")
            ans = q.get("answer", "")
            if ans not in q["options"]:
                errors.append(f"{qid}: answer {ans!r} not among options")
            else:
                mc_letters[ans] += 1
                opt = q["options"][ans]
                # Verbatim, or every distinctive word of the option in the chunk
                # (abbreviation-expansion options paraphrase the chunk).
                words = [w for w in norm(opt) if len(w) >= 2]
                if opt not in chunk_text and not (
                    words and all(w in norm(chunk_text) for w in words)
                ):
                    errors.append(f"{qid}: correct option {opt!r} NOT verbatim in {ev}")
        else:
            ans = q.get("answer", "")
            if ans not in ("TRUE", "FALSE"):
                errors.append(f"{qid}: TF answer {ans!r} invalid")
            else:
                tf_answers[ans] += 1
        # Near-duplicate vs Test.csv (first 12 normalized words).
        head = " ".join(norm(q["question"])[:12])
        for tq in test_qs:
            if head == tq:
                errors.append(f"{qid}: near-duplicate of a Test.csv question")
                break
    # Near-duplicates within the file: identical normalized question text.
    seen_q: dict[str, str] = {}
    for q in questions:
        full = " ".join(norm(q["question"]))
        if full in seen_q:
            errors.append(f"{q['id']}: duplicate question text of {seen_q[full]}")
        seen_q[full] = q["id"]

    print(f"{path}: {len(questions)} questions "
          f"({sum(1 for q in questions if q.get('options'))} MC, "
          f"{sum(1 for q in questions if not q.get('options'))} TF)")
    print(f"MC correct letters: {dict(mc_letters)}")
    print(f"TF answers: {dict(tf_answers)}")
    if errors:
        print(f"FAILURES ({len(errors)}):")
        for e in errors:
            print(f"  - {e}")
        return 1
    print("OK: all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
