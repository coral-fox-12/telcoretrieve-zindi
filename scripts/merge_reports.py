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
variant settings and generator model — unless
--allow-generator-mismatch is given, which merges runs whose configs
differ ONLY in "generator" (e.g. completing a nemotron M1 slice with
a Gemini continuation) and records the mismatch in the report. A
mismatched merge is honest but confounded: do not quote its score as
a single-generator result.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from telco.questions import load_validation_json
from telco.scoring import score_submission
from telco.submission import check_submission, write_submission, SubmissionRow

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser(description="Merge a partial run with its continuation")
    ap.add_argument("base_dir")
    ap.add_argument("cont_dir")
    ap.add_argument("out_dir")
    ap.add_argument(
        "--allow-generator-mismatch",
        action="store_true",
        help="merge even if the two runs' configs differ ONLY in "
        "'generator'; records the mismatch in the merged report",
    )
    args = ap.parse_args()
    base_dir, cont_dir, out_dir = (
        Path(args.base_dir),
        Path(args.cont_dir),
        Path(args.out_dir),
    )
    base = json.loads((base_dir / "report.json").read_text(encoding="utf-8"))
    cont = json.loads((cont_dir / "report.json").read_text(encoding="utf-8"))
    generator_mismatch = None
    if base["config"] != cont["config"]:
        if args.allow_generator_mismatch and all(
            base["config"].get(k) == cont["config"].get(k)
            for k in base["config"]
            if k != "generator"
        ):
            generator_mismatch = {
                "base": base["config"].get("generator"),
                "cont": cont["config"].get("generator"),
            }
            print(
                "WARNING: merging runs with different generators "
                f"(base={generator_mismatch['base']}, "
                f"cont={generator_mismatch['cont']}); the merged score is "
                "a MIXED-generator result, not a single-generator number",
                file=sys.stderr,
            )
        else:
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
    if generator_mismatch is not None:
        report["generator_mismatch"] = generator_mismatch
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
