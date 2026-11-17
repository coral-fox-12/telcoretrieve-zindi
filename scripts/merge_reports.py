#!/usr/bin/env python3
"""Merge a partial (aborted) run with its continuation run.

When a run aborts (e.g. free-quota exhaustion), resume it by running
the remaining questions into a second directory, then merge:

    python3 -m telco.evaluate --questions data/remaining.json \
        --index <same> --top-k <same> --rerank-top-k <same> \
        --out runs/M1_cont --generator openrouter --max-tokens 512
    python3 scripts/merge_reports.py runs/M1_512_k5_llm runs/M1_cont \
        runs/M1_512_k5_llm_merged

Merges per_question (base error entry dropped), submission.csv rows,
and recomputes the score over all answered questions. The merged
report's config comes from the base run; both runs must use the same
variant settings and generator model.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from telco.questions import load_validation_json
from telco.scoring import score_submission
from telco.submission import check_submission, write_submission, SubmissionRow

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    base_dir, cont_dir, out_dir = (Path(p) for p in sys.argv[1:4])
    base = json.loads((base_dir / "report.json").read_text(encoding="utf-8"))
    cont = json.loads((cont_dir / "report.json").read_text(encoding="utf-8"))
    if base["config"] != cont["config"]:
        print("config mismatch between runs, refusing to merge:", file=sys.stderr)
        print("  base:", base["config"], file=sys.stderr)
        print("  cont:", cont["config"], file=sys.stderr)
        return 1

    per_question = [p for p in base["per_question"] if "error" not in p]
    per_question += cont["per_question"]
    answered = [p for p in per_question if p.get("answer")]
    answers = {p["id"]: p["answer"] for p in answered}
    total_tokens = sum(p["prompt_tokens"] for p in answered)

    questions = load_validation_json(ROOT / "data/validation_questions_v2.json")
    reference = {q.id: q.answer for q in questions if q.id in answers}
    score = score_submission(answers, reference, total_tokens)

    report = {
        # Full validation set size (a sliced base run reports only its
        # slice length; the merged report covers the whole set).
        "n_questions": len(questions),
        "n_answered": len(answered),
        "per_question": per_question,
        "config": base["config"],
        "score": score.as_dict(),
        "merged_from": [str(base_dir), str(cont_dir)],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    # Rebuild submission rows from the two submission.csv files.
    import csv as csv_mod

    prompt_by_id: dict[str, str] = {}
    for d in (base_dir, cont_dir):
        with open(d / "submission.csv", newline="", encoding="utf-8") as fh:
            for row in csv_mod.DictReader(fh):
                prompt_by_id[row["ID"]] = row["input_prompt"]
    sub_rows = [
        SubmissionRow(id=p["id"], input_prompt=prompt_by_id[p["id"]], answer=p["answer"])
        for p in answered
    ]
    write_submission(sub_rows, out_dir / "submission.csv")
    if len(answered) == len(questions):
        check_submission(out_dir / "submission.csv", questions)
    else:
        print(f"WARNING: only {len(answered)}/{len(questions)} answered; "
              f"skipping submission check (merge more continuations)")
    print(f"merged {len(answered)}/{len(questions)} answered "
          f"-> {out_dir}/report.json + submission.csv")
    print("score:", json.dumps(score.as_dict()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
