# TelcoRetrieve Challenge — Zindi (GSMA / ITU / AMD)

**€25,000 cash** — 1st €8,000 / 2nd €5,000 / 3rd €3,500 / 4th €2,000 / 5th €1,500.
Top 3 also get an MWC Barcelona Leader Pass + up to €1,500 travel.
Competition: https://zindi.world/competitions/telcoretrieve-challenge-unlocking-telco-knowledge
Pipeline (Apache 2.0): https://github.com/Znbne/telco-retrieve

## Verified facts (live, 2026-11-17)

- **Open to all** — no country exclusion (only Russia cannot receive prize
  payments). Solo or team up to 3. Prizes go to the individual / team leader.
- **Dates (VN time):** public phase 13 Nov 2026 – 8 Jan 2027 (100 questions);
  unseen-question phase 8–13 Jan; **submissions close 05:25 VN 14 Jan 2027**
  (`submissions_closed_at` 2027-01-13T22:25Z). Entries close 06:59 VN 7 Jan
  (`entries_close_at`) — join early. Top-10 code review 15–31 Jan on AMD
  GPU instances; winners present at MWC Barcelona 1–4 Mar 2027.
- **Task:** improve the TelcoRetrieve RAG pipeline (retrieval, chunking,
  hybrid search, reranking, prompt construction) on a **fixed generator
  `google/gemma-3n-E4B-it`** and fixed OTel embedding (`OTel-Embedding-0.6B`,
  fine-tuning allowed) + reranker (`OTel-Reranker-0.6B`).
- **Score = 0.7 × accuracy + 0.3 × token efficiency**, token efficiency =
  100 × B / (B + T), B = 850 tokens/question, T = total `cl100k_base`
  tokens of `input_prompt`. **Accuracy < 80% = disqualified.**
  Every `input_prompt` ≤ 4,096 tokens. One generator call per question.
- **Submission file:** CSV `ID,input_prompt,answer`. `input_prompt` is the
  system message + blank line + user message, exactly as sent to the
  generator. Answer = option letter (A–E) or TRUE/FALSE. Exact-match scoring.
- **Limits:** 1 submission/day, 20 total. Choose 2 submissions for the
  private leaderboard (public ~33.3% of test, private 66.7%).
- **Rules:** open-source tools only; only the challenge datasets; pretrained
  models OK if openly available; no AutoML; no paid services / trials
  needing a credit card. Top 10 must hand over code within 48h; code must
  reproduce the score. Winners 1st–3rd assign copyright to Zindi.
- **Payout:** bank transfer / PayPal / international transfer; winners show
  proof of ID + proof of residence + bank letter. Fees deducted unless
  prize < $500.
- **Field (17 Nov):** 322 participants, 83 on the leaderboard.
- **Account:** Zindi `coral-fox-12` (coral-fox-12@proton.me), email
  confirmed, **joined the competition (participation 245201)**.
  Credentials in `~/workspace/makefun/accounts-created.json` under
  `zindi_coralfox`. API note: send the token as an `auth_token` header
  (not `Authorization: Bearer`); sign-in uses the username, not email.
- **Data (downloaded, `data/`):** `Test.csv` (100 questions: 50 MC + 50 TF,
  all citing a 3GPP Rel-19 spec), `SampleSubmission.csv`,
  `VariableDefinitions.csv`. Index: `GSMA/telco-retrieve-chunks`
  (`telcoretrievechallenge/fixed-grid/text_baseline/otel-0.6b`, ~10 GB).

## Scaffold layout

```
src/telco/
  prompt.py      exact upstream prompt semantics (system msg, passage
                 format, drop-lowest-to-fit-4096)
  tokens.py      cl100k_base counting (tiktoken)
  questions.py   Test.csv + validation JSON loaders
  submission.py  write / check (mirrors official `challenge check`) /
                 answer parsing
  scoring.py     accuracy / token efficiency / overall / 80% gate
  retrieval.py   BM25 over the mini index (stdlib; dense path plugs in here)
  generate.py    OpenRouter :free stand-in + lexical answerer
  evaluate.py    full loop -> submission.csv + report.json
scripts/build_mini_index.py   chunks TS 29.280 + TS 23.434 (<=512 tok)
data/
  Test.csv / SampleSubmission.csv / VariableDefinitions.csv  (from Zindi)
  TS29280.md / TS23434.md      real Rel-19 spec markdown (GSMA/3GPP on HF)
  mini_index.jsonl             688 chunks, <=512 tokens each
  mini_index_256.jsonl         1,436 chunks, <=256 tokens each
                               (each with a "parent" 512-chunk id)
  validation_questions.json    12 questions (6 MC + 6 TF) authored from
                               TS 29.280 chunks, answers verbatim in the
                               cited evidence chunk
  validation_questions_v2.json 60 questions (30 MC + 30 TF): the original
                               12 + 48 authored 2026-11-17 from TS 23.434
                               (c0001–c0638) and TS 29.280 (c0033–c0050),
                               same method, machine-verified
                               (scripts/verify_validation.py)
  validation_part_a/b.json     the two 24-question authoring batches
tests/test_telco.py            25 tests, all passing
runs/baseline_v1_lexical/      lexical floor (see below)
runs/baseline_v1_llm/          free-model run (see below)
runs/L1_512_k5/ … L7_256_k2/   phase-1 lean-prompt variants (see below)
```

## How to run

```bash
python3 -m unittest discover -s tests            # 25 tests
python3 scripts/build_mini_index.py              # rebuild the mini index
PYTHONPATH=src python3 -m telco.evaluate \
  --questions data/validation_questions.json \
  --index data/mini_index.jsonl \
  --out runs/baseline_v1_lexical --generator lexical
PYTHONPATH=src python3 -m telco.evaluate \
  --questions data/validation_questions_v2.json \
  --index data/mini_index.jsonl \
  --out runs/M1_512_k5_llm --generator openrouter \
  --max-tokens 512
# Lean variants (see runs/PHASE1.md for the comparison):
#   --rerank-top-k 3            (fewer passages)
#   --index data/mini_index_256.jsonl --rerank-top-k 3   (256-tok chunks)
#   --context-budget 850        (budgeted context fill, upstream mode)
#   --max-tokens 512            (required for reasoning stand-ins)
# Full sweep driver: bash scripts/run_llm_variants.sh
# Verify a validation file: python3 scripts/verify_validation.py <file>
```

## Baselines (local eval, 12 validation questions)

| run | generator | accuracy | token eff. | overall | gate |
|---|---|---|---|---|---|
| `baseline_v1_lexical` | BM25 + option-overlap, TF defaults TRUE | 9/12 = 75% | 26.46 | 60.44 | FAIL (<80%) |
| `baseline_v1_llm` | BM25 + nemotron-3-nano-omni-30b-a3b-reasoning:free (stand-in, NOT the fixed gemma-3n) | 12/12 = 100% | 26.46 | 77.94 | PASS |

## Phase 1 — lean prompts on the 60-question validation set (2026-11-17)

Lexical floor (BM25 + option-overlap, TF defaults TRUE — so accuracy
is a retrieval floor, not a generator score). Token efficiency is the
real signal here:

| run | index | passages | tok/q | token eff. | lexical acc. | overall | evidence in ctx |
|---|---|---|---|---|---|---|---|
| `L1_512_k5` | 512 tok | 5 | 2,464 | 25.65 | 55.0% | 46.20 | 54/60 |
| `L2_512_k3` | 512 tok | 3 | 1,535 | 35.64 | 53.3% | 48.02 | 48/60 |
| `L3_256_k5` | 256 tok | 5 | 1,316 | 39.24 | 51.7% | 47.94 | 44/60 |
| `L4_256_k3` | 256 tok | 3 | 832 | 50.53 | 50.0% | 50.16 | 43/60 |
| `L5_256_b850` | 256 tok | ~3.2 (budget 850) | 880 | 49.12 | 51.7% | 50.90 | 43/60 |
| `L6_512_b850` | 512 tok | ~1.2 (budget 850) | 664 | 56.14 | 53.3% | 54.18 | 39/60 |
| `L7_256_k2` | 256 tok | 2 | 587 | 59.14 | 50.0% | 52.74 | 40/60 |

Reading: cutting passages barely moves lexical accuracy (50–55% —
the lexical answerer saturates) but evidence-in-context falls as
prompts get leaner (90% → 65–72%). The LLM runs (same variants,
stand-in generator) are what rank the variants for real: an LLM can
only answer from evidence it actually sees, so the accuracy cost of
lean prompts shows up there, not here.

LLM stand-in runs (nemotron-3-nano-omni, temp 0, max_tokens 512):
only the M1 baseline got started — **17/60 answered, 15/17 correct
(88.2%), overall 69.67** — then the sweep hit OpenRouter's free
daily cap (50 requests/day per account, all free models combined)
and stopped at val-018. **Do not quote 88.2% as a 60-question
number**: the 17 are the easiest slice (original 12 + first 5 of
part A). M2–M6 did not run. Full analysis + resumption options
(wait for the UTC-midnight reset, $10 credits → 1,000 req/day, or
skip to the phase-2 GPU run with the real generator) in
`runs/PHASE1.md`.

Validation-set caveats (disclosed, not hidden):
- `val-005` ("What underlying protocol are the Sv interface messages
  based on?") duplicates a real `Test.csv` question word-for-word —
  authored independently from the same TS 29.280 chunk, answer
  derived from the spec, not from the test. Kept (1 of 60) so the
  12-question baselines stay comparable; it is flagged by
  `scripts/verify_validation.py`.
- `val-003`/`val-004` evidence citations were fixed 2026-11-17: the
  message-type table rows live in `TS29280-c0010`, not `c0009`
  (which only holds the table title).
- MC correct letters in v2 skew A (11/6/6/7) because the original 12
  were 5×A; the 48 new questions are balanced 12/12/12/12. TF is
  exactly 15 TRUE / 15 FALSE.

Read the 12/12 honestly: the validation set was authored from the same
chunks the index contains, so answers are verbatim in evidence (4/12 were
even answered right without the evidence chunk in context). It proves the
loop works end-to-end — prompt, retrieval, parse, check, score — not that
the real 100 questions will score 100%. The stand-in is a 30B reasoning
model; the fixed Gemma 3n E4B will behave differently. Token efficiency
26.46 is the real story: at 850 tok/question (eff. 50) the same accuracy
would score 85.0; at 400 tok/question (eff. ~68) it would score ~90.4.
Lean prompts are the biggest lever, not the last accuracy points.

Lexical read: MC 6/6 (option text overlaps the passages); TF 3/6 (all
TRUE by default). Evidence chunk in the top-5 context for 8/12. Prompts
average ~2,360 tokens/question vs the 850 baseline — **token efficiency
is the biggest lever**: 5 passages × ~470 tok is far too fat. The winning
systems retrieve only what matters: fewer, shorter, better-ranked passages.

## Improvement plan (in priority order)

1. **Leaner prompts (token efficiency, 30% of score).** Try top-k 3–5
   after rerank, 256-token chunks, budgeted context fill (the upstream
   `--context-token-budget` mode). Measure accuracy vs tokens per setting;
   the optimum is where marginal accuracy per token flattens.
2. **Better retrieval (accuracy).** BM25 misses table chunks (evidence
   ranked 15th for val-003). Add: dense OTel-0.6B embeddings + hybrid
   (dense + BM25) + OTel reranker — the fixed models, run on Kaggle GPU.
   Also: table-aware chunking (keep tables whole), acronym expansion,
   question-keyword injection (`--include-keywords`).
3. **Answer parsing robustness.** The generator sometimes adds prose;
   the parser already takes the first valid letter/TRUE/FALSE. Add a
   second-chance re-prompt on empty parse (still one generator call per
   question? No — the rules say each question may use ONE generator
   call, and `input_prompt` is that one prompt. So no re-prompting;
   parsing must be right the first time).
4. **Validation set growth.** DONE 2026-11-17: 12 → 60 questions
   (`data/validation_questions_v2.json`, machine-verified). Before
   trusting any ranking between variants, re-check on the 60-set.
5. **Full-scale run (Kaggle GPU, see RUNBOOK.md).** Official CLI against
   the 10 GB fixed index + vLLM Gemma 3n; submit to Zindi (1/day, 20
   total — spend submissions on real improvements only).

## Notes / gotchas

- Zindi sign-in uses the **username**, not the email. API auth is the
  `auth_token` header. Email confirmation is a POST to
  `/auth/confirmations/confirm` with `{token}` (the link in the welcome
  email) — a plain GET of the link does not confirm.
- `git clone` and `curl -o` writes flap on this VM (files land 0-byte);
  retry, verify sizes, or write via the file tools / shell heredocs.
  `git init` and shell redirections persist reliably.
- OpenRouter :free is rate-limited upstream (429s) — the generator
  retries with backoff (15s/45s/120s). Free quota is 50 req/day.
- The real Test.csv has NO answers — local scoring is only possible on
  the validation set. Never tune on Test.csv answers (there are none);
  the public leaderboard is the only signal for the real 100.
