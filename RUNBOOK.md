# RUNBOOK — full-scale run on Kaggle GPU

The local scaffold measures pipeline mechanics on a 688-chunk mini
index. The real competition runs against the 10 GB fixed index with the
fixed Gemma 3n generator. That needs a GPU: use the Kaggle account
`coralfox12` (30 GPU-h/week, ~20 TPU-h/week; credentials in
`~/workspace/makefun/kaggle-credentials.json`).

## 0. Local stand-in generator (what actually works, 2026-11-17)

The fixed generator (gemma-3n-E4B-it) is NOT on OpenRouter, so local
runs use a free stand-in. **Model IDs rot fast** — check the live list
(`GET https://openrouter.ai/api/v1/models`, filter `:free`) before
trusting any ID in an old doc:
- `google/gemma-4-26b-a4b-it:free` and `google/gemma-4-31b-it:free`:
  **429 upstream** (Google AI Studio shared pool) on 2026-11-17.
- `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free`: **works** —
  this is the model the 12/12 baseline used, and the current default
  in `src/telco/generate.py`. Reasoning model: run with
  `--max-tokens 512` (with 8 the reasoning gets truncated).
- Also working 2026-11-17: `cohere/north-mini-code:free`,
  `nvidia/nemotron-3.5-lightning:free`.
- Dead IDs (do not use): `meta-llama/llama-3.3-70b-instruct:free`
  (free tier removed), `qwen/qwen3-235b-a22b:free` (deprecated
  2026-10-09), `nvidia/nemotron-3-nano-30b-a3b-reasoning:free`
  (missing the `-omni-` infix).
- Free quota is ~50 requests/day; `OpenRouterGenerator` has a circuit
  breaker (3 consecutive 429s → fail fast) and `evaluate.run` saves a
  partial report on generator failure instead of losing the run.

## 1. Setup (Kaggle notebook, GPU L4 x4 or T4 x2)

```bash
pip install -q "open-telco-rag @ git+https://github.com/Znbne/telco-retrieve"  # or:
git clone https://github.com/Znbne/telco-retrieve && cd telco-retrieve
pip install -e ".[local-embed]"
# CPU-only torch first if needed; on Kaggle GPU the CUDA build is default.
```

## 2. Get the questions + the fixed index

```bash
# Test.csv: download from the Zindi Data tab (account coral-fox-12,
# already joined). Or via the Zindi API:
#   GET https://api.zindi.world/v1/competitions/telcoretrieve-challenge-unlocking-telco-knowledge/files/Test.csv
#   header: auth_token: <token from accounts-created.json>
open-telco-rag pull-hf GSMA/telco-retrieve-chunks \
    telcoretrievechallenge/fixed-grid/text_baseline/otel-0.6b --local-dir ./runs
# DB: ./runs/telcoretrievechallenge/fixed-grid/text_baseline/otel-0.6b/db
# Collection: otel-0.6b_text_baseline_b512_v256_full  (~10 GB)
```

## 3. Serve Gemma 3n (fixed generator)

```bash
# vLLM, OpenAI-compatible server on :8000
python -m vllm.entrypoints.openai.api_server \
  --model google/gemma-3n-E4B-it --port 8000 --max-model-len 8192
export LOCAL_LLM_BASE_URL=http://localhost:8000/v1
```

## 4. Export / check / score (official CLI)

```bash
DB=./runs/telcoretrievechallenge/fixed-grid/text_baseline/otel-0.6b/db
open-telco-rag challenge export --questions Test.csv --out submission.csv \
    --db-path $DB --collection otel-0.6b_text_baseline_b512_v256_full
open-telco-rag challenge check --submission submission.csv --questions Test.csv
# score needs your own answers file (ID,answer) — only for validation sets.
```

Defaults (configs/profiles/challenge.json): hybrid retrieval, top-k 10,
reranker keeps 5, temperature 0, seed 1234, max prompt 4096, max_tokens 8.

## 5. Experiments to run (in order)

1. Baseline: official defaults, record accuracy proxy (see §6) + tokens.
2. Lean prompts: `--rerank-top-k 3`, then `--top-k 5 --rerank-top-k 3`,
   then `--context-token-budget 850` (fill to the baseline budget).
   **Local phase-1 result (2026-11-17, 60-question validation set,
   nemotron-omni stand-in — see `runs/PHASE1.md`):** the lean variants
   are the whole game — the 512/k5 baseline burns ~2,460 tok/q
   (efficiency ~26), while budget-850 and 256-token variants land at
   ~660–880 tok/q (efficiency ~49–56) with only a small accuracy cost
   on the stand-in. The exact accuracy-vs-tokens curve per variant is
   in `runs/PHASE1.md`; take the best local variant into the GPU run
   first, then confirm on the public leaderboard (1 submission/day).
3. Retrieval variants: `--search-method dense` vs `hybrid` vs `bm25`;
   `--include-keywords`; `--resolve-parents`.
4. Prompt variants: only via the supported `--answer-parser` /
   custom prompt builder hooks (see docs/reference/challenge-options.md).
   The system message and instruction format are part of the submitted
   prompt, so every variant is honestly scored by token count.
5. Embedder fine-tuning is allowed (OTel-Embedding-0.6B only) — last,
   only if steps 1–4 plateau: fine-tune on Rel-19 triples, rebuild the
   index, re-export.

## 6. Measuring accuracy without Test.csv answers

No labelled set is released. Build a validation set the same way as
`data/validation_questions.json` (author from Rel-19 chunks, answers
verbatim in the evidence chunk), then:

```bash
open-telco-rag evaluate --questions-file my_validation.json
# or use this scaffold: PYTHONPATH=src python3 -m telco.evaluate ...
```

Grow the set to 50–100 questions before ranking variants. The public
leaderboard (100 questions, 1 submission/day) is the only signal on the
real set — spend the 20 submissions on variants that win locally by a
clear margin.

## 7. Submit

Upload `submission.csv` on the Zindi competition page (Data tab format
is `ID,input_prompt,answer`). Verify with `challenge check` first.
Choose 2 submissions for the private leaderboard before the close
(05:25 VN 14 Jan 2027). If top 10: code must be handed over within
48h and must reproduce the score — keep the repo clean and the run
reproducible (seed 1234, pinned versions, no API keys in the repo).
