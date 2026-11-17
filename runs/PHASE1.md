# Phase 1 — lean prompts on the 60-question validation set (2026-11-17)

Stand-in generator for all LLM runs:
`nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free`, temperature 0,
max_tokens 512. (gemma-4 free is 429-limited upstream; the fixed
competition generator is gemma-3n-E4B-it, NOT on OpenRouter — the real
numbers come from the phase-2 Kaggle GPU run.)

## 1. Lexical floor (full 60 questions, BM25 + option-overlap, TF defaults TRUE)

Accuracy here is a retrieval floor, not a generator score. The signal
is token efficiency and evidence-in-context.

| run | index | passages | tok/q | token eff. | lexical acc. | overall | evidence |
|---|---|---|---|---|---|---|---|
| L1_512_k5 | 512 tok | 5 | 2,464 | 25.65 | 55.0% | 46.20 | 54/60 |
| L2_512_k3 | 512 tok | 3 | 1,535 | 35.64 | 53.3% | 48.02 | 48/60 |
| L3_256_k5 | 256 tok | 5 | 1,316 | 39.24 | 51.7% | 47.94 | 44/60 |
| L4_256_k3 | 256 tok | 3 | 832 | 50.53 | 50.0% | 50.16 | 43/60 |
| L5_256_b850 | 256 tok | ~3.2 (budget 850) | 880 | 49.12 | 51.7% | 50.90 | 43/60 |
| L6_512_b850 | 512 tok | ~1.2 (budget 850) | 664 | 56.14 | 53.3% | 54.18 | 39/60 |
| L7_256_k2 | 256 tok | 2 | 587 | 59.14 | 50.0% | 52.74 | 40/60 |

Lexical MC accuracy is flat (15–18/30) across variants — the lexical
answerer saturates. Evidence-in-context falls as prompts lean out
(90% → 65–72%). An LLM can only answer from evidence it sees, so the
real accuracy cost of leaning shows up in the LLM runs, not here.

## 2. LLM stand-in — M1 baseline, PARTIAL (17/60 answered, then quota wall)

| run | answered | accuracy | tok/q | token eff. | overall | gate | evidence |
|---|---|---|---|---|---|---|---|
| M1_512_k5_llm | 17/60 | 88.2% (15/17) | 2,374 | 26.36 | 69.67 | PASS | 14/17 |

**The 17 are the first 17 questions (val-001..val-017) — the original
12 plus the first 5 of part A — i.e. the easiest slice of the set
(the 12/12 baseline was measured on the original 12). Do NOT quote
88.2% as the 60-question accuracy.** The two misses are instructive:
- val-012 (TF, evidence IN context): the model reasoned from the
  passage and still answered TRUE (ref FALSE) — a reasoning failure,
  not a retrieval failure.
- val-017 (MC, evidence NOT in context): retrieval miss → the model
  guessed D (ref A). This is the exact failure mode lean prompts
  increase.

M2–M6 (512/b850, 256/k3, 256/b850, 256/k2, 256/k5) did NOT run.

## 3. Why the sweep stopped: OpenRouter free daily quota

`Rate limit exceeded: free-models-per-day` — the free tier is
**50 requests/day per account, across ALL free models** (not per
model, not per key; the other keys on the same account share the
cap). Usage on 2026-11-17 before the sweep: 12 baseline + Codex
pings + probes ≈ 30; M1 burned 17 more and hit the wall at val-018.
Failed 429s do not count (no generation), successful ones do.

Resumption options, in order of preference:
1. **Wait for the daily reset** (UTC midnight = 07:00 VN) and resume:
   run the remaining questions (val-018..val-060) into a second dir,
   then `python3 scripts/merge_reports.py runs/M1_512_k5_llm
   runs/M1_cont runs/M1_512_k5_llm_merged`. Then run M2–M6 the same
   way (60 requests each — one variant per day at the 50/day cap, or
   two variants across two days). Cost: $0, delay: 1–6 days.
2. **$10 OpenRouter credits** → 1,000 free-model requests/day
   (per OpenRouter's own message). Needs the user's explicit approval
   — spending money is never assumed. Unlocks the whole sweep in one
   day plus all future stand-in runs. Cost: $10 once.
3. **Skip the stand-in ranking; go straight to phase 2** (Kaggle GPU,
   vLLM, the REAL gemma-3n-E4B-it generator). The stand-in ranking is
   a proxy anyway — the GPU run measures the actual fixed generator,
   which is the ground truth for submission. The variant shortlist
   from the lexical + M1-partial evidence: 512/b850 first (best
   lexical overall at 664 tok/q), then 256/k3 (832 tok/q), then the
   512/k5 baseline as the control.

## 4. Lean-prompt ceiling (what the score formula allows)

overall = 70 × accuracy + 0.3 × token_efficiency. At 100% accuracy:
512/k5 → 77.7, 512/b850 → 86.8, 256/k3 → 85.2, 256/k2 → 87.7.
At 90% accuracy: 512/k5 → 70.7, 512/b850 → 79.8, 256/k3 → 78.2,
256/k2 → 80.7. At 85%: 512/k5 → 67.2, 512/b850 → 76.3, 256/k3 →
74.7, 256/k2 → 77.2.

So the baseline config caps at ~71 even at 90% accuracy, while the
lean configs reach ~78–81 at the same accuracy. **The entire game is
whether accuracy holds ≥85–90% when the evidence shrinks.** That is
exactly what the unfinished M2–M6 sweep was meant to measure — and
what the phase-2 GPU run will measure with the real generator.

## 5. Phase-2 readiness

Ready: 60-question validation set (verified), 256-token index (+parent
mapping), budgeted context fill, abort-safe evaluation loop with
partial reports + merge script, variant shortlist above, public repo
(github.com/coral-fox-12/telcoretrieve-zindi), RUNBOOK §5.
Still to do (phase 2, NOT run yet): Kaggle notebook (L4 x4 or T4 x2),
`open-telco-rag pull-hf` the 10 GB fixed index, serve gemma-3n-E4B-it
via vLLM on :8000, export with the official CLI per variant,
`challenge check`, submit the best variant to Zindi (1/day, 20 total —
spend submissions only on clear local winners).
