"""Evaluation loop: questions -> retrieval -> prompt -> generate -> submission.

Usage:
    python3 -m telco.evaluate --questions data/validation_questions.json \
        --index data/mini_index.jsonl --out runs/baseline_v1 \
        --generator openrouter   # or: lexical, gemini
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .generate import (
    DEFAULT_FREE_MODEL,
    DEFAULT_GEMINI_MODEL,
    GeminiGenerator,
    LexicalAnswerer,
    OpenRouterGenerator,
    load_gemini_key,
    load_openrouter_key,
)
from .prompt import MAX_PROMPT_TOKENS, build_prompt, build_prompt_budgeted
from .questions import ChallengeQuestion, load_test_csv, load_validation_json
from .retrieval import BM25Index
from .scoring import score_submission
from .submission import (
    SubmissionRow,
    check_submission,
    parse_answer,
    write_submission,
)

ROOT = Path(__file__).resolve().parents[1]


def load_questions(path: Path) -> list[ChallengeQuestion]:
    if path.suffix == ".json":
        return load_validation_json(path)
    return load_test_csv(path)


def run(
    questions: list[ChallengeQuestion],
    index: BM25Index,
    generator,
    top_k: int = 10,
    rerank_top_k: int = 5,
    out_dir: Path | None = None,
    context_budget: int | None = None,
    index_path: str = "",
) -> dict:
    """Run the full loop. Returns the report dict; writes submission.csv
    + report.json into out_dir when given."""
    rows: list[SubmissionRow] = []
    answers: dict[str, str] = {}
    total_tokens = 0
    per_question: list[dict] = []
    abort_error: str | None = None

    for q in questions:
        hits = index.search(q.question, top_k=top_k)
        if context_budget is not None:
            # Budgeted context fill: the budget truncates, not rerank_top_k.
            # kept must be exactly the passages that made it into the prompt
            # (build_prompt_budgeted keeps a rank-ordered prefix of them).
            prompt = build_prompt_budgeted(q, [h.as_passage() for h in hits], context_budget)
            kept = hits[: prompt.n_passages]
        else:
            # Local "rerank": keep the top rerank_top_k BM25 hits.
            kept = hits[:rerank_top_k]
            prompt = build_prompt(q, [h.as_passage() for h in kept])
        total_tokens += prompt.n_tokens
        try:
            if isinstance(generator, LexicalAnswerer):
                generator.chunks = kept
                raw = generator.generate(q)
            else:
                raw = generator.generate(prompt.input_prompt)
        except Exception as exc:  # noqa: BLE001 - record and stop gracefully
            # A generator failure (e.g. free-quota exhaustion) must not
            # lose the questions already answered: record the failure,
            # stop the run, and save a partial report below.
            per_question.append(
                {
                    "id": q.id,
                    "raw_generator_output": "",
                    "answer": "",
                    "prompt_tokens": prompt.n_tokens,
                    "n_passages": prompt.n_passages,
                    "retrieved_ids": [h.id for h in kept],
                    "evidence_in_context": q.evidence_chunk
                    in {h.id for h in kept} | {h.parent for h in kept if h.parent},
                    "error": str(exc),
                }
            )
            abort_error = str(exc)
            # The failed question's prompt was counted above; drop it so
            # the partial score covers answered questions only.
            total_tokens -= prompt.n_tokens
            break
        answer = parse_answer(raw, q)
        answers[q.id] = answer
        rows.append(SubmissionRow(id=q.id, input_prompt=prompt.input_prompt, answer=answer))
        per_question.append(
            {
                "id": q.id,
                "raw_generator_output": raw,
                "answer": answer,
                "prompt_tokens": prompt.n_tokens,
                "n_passages": prompt.n_passages,
                "retrieved_ids": [h.id for h in kept],
                # Evidence counts if the evidence chunk itself — or, on a
                # sub-512 index, one of its child chunks — is in context.
                "evidence_in_context": q.evidence_chunk
                in {h.id for h in kept} | {h.parent for h in kept if h.parent},
            }
        )

    # On abort, score only the questions that were actually answered.
    reference = {
        q.id: q.answer
        for q in questions
        if q.answer and (abort_error is None or q.id in answers)
    }
    report: dict = {
        "n_questions": len(questions),
        "n_answered": len(answers),
        "per_question": per_question,
        "config": {
            "generator": getattr(generator, "model", "lexical"),
            "top_k": top_k,
            "rerank_top_k": rerank_top_k,
            "context_budget": context_budget,
            "index": index_path,
        },
    }
    if abort_error is not None:
        report["aborted"] = True
        report["abort_error"] = abort_error
    if reference:
        score = score_submission(answers, reference, total_tokens)
        report["score"] = score.as_dict()
    else:
        report["score"] = None
        report["total_prompt_tokens"] = total_tokens

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        write_submission(rows, out_dir / "submission.csv")
        # Write the report BEFORE check so raw outputs survive a check
        # failure (diagnosing a bad generator needs them).
        (out_dir / "report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        # A partial (aborted) submission is incomplete by design; the
        # full-set check would reject it for missing ids.
        if abort_error is None:
            check_submission(out_dir / "submission.csv", questions)
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description="TelcoRetrieve local evaluation loop")
    ap.add_argument("--questions", required=True)
    ap.add_argument("--index", default=str(ROOT / "data/mini_index.jsonl"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--generator", choices=["openrouter", "lexical", "gemini"],
                    default="openrouter")
    ap.add_argument("--max-tokens", type=int, default=8,
                    help="generator max_tokens (8 = challenge profile; "
                    "raise for reasoning stand-ins, e.g. 512)")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--rerank-top-k", type=int, default=5)
    ap.add_argument("--context-budget", type=int, default=None,
                    help="budgeted context fill: keep top-ranked passages "
                    "while their cumulative tokens stay within this budget "
                    "(overrides --rerank-top-k)")
    ap.add_argument("--key-name", default="key12",
                    help="key name in the credentials file: OpenRouter key "
                    "(openrouter-credentials.json, e.g. key2) or Gemini key "
                    "(gemini-credentials.json, e.g. gemini_key_1)")
    ap.add_argument("--model", default=None,
                    help="generator model id; defaults to the stand-in for "
                    "the chosen provider (nemotron for openrouter, "
                    "gemma-4-26b-a4b-it for gemini)")
    ap.add_argument("--offset", type=int, default=0,
                    help="question slice start (0-based) for splitting a "
                    "run across several free-quota accounts")
    ap.add_argument("--limit", type=int, default=None,
                    help="question slice length; combine slices with "
                    "scripts/merge_reports.py")
    args = ap.parse_args()

    questions = load_questions(Path(args.questions))
    if args.offset or args.limit is not None:
        end = None if args.limit is None else args.offset + args.limit
        questions = questions[args.offset:end]
        print(f"slice: questions[{args.offset}:{end}] -> {len(questions)} questions")
    index = BM25Index.from_jsonl(args.index)
    if args.generator == "lexical":
        generator = LexicalAnswerer([])
    elif args.generator == "gemini":
        generator = GeminiGenerator(
            model=args.model or DEFAULT_GEMINI_MODEL,
            api_key=load_gemini_key(args.key_name),
            max_tokens=args.max_tokens,
        )
    else:
        generator = OpenRouterGenerator(
            model=args.model or DEFAULT_FREE_MODEL,
            api_key=load_openrouter_key(args.key_name),
            max_tokens=args.max_tokens,
        )
    report = run(
        questions,
        index,
        generator,
        top_k=args.top_k,
        rerank_top_k=args.rerank_top_k,
        out_dir=Path(args.out) if args.out else None,
        context_budget=args.context_budget,
        index_path=args.index,
    )
    print(json.dumps(report.get("score"), indent=2))
    if args.out:
        print(f"wrote {args.out}/submission.csv + report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
