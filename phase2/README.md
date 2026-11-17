# Phase 2 — full-scale Kaggle GPU run (real generator)

The ground-truth measurement for TelcoRetrieve: the FIXED generator
`google/gemma-3n-E4B-it` (via vLLM) over the FIXED 10 GB index, scored
with the official CLI on our 60-question validation set.

## What is here

| File | Purpose |
|---|---|
| `notebook.ipynb` | The Kaggle notebook (13 cells): install → pull index → serve Gemma 3n → 3 exports → check/score → summary |
| `kernel-metadata.json` | Kernel `coralfox12/telcoretrieve-phase-2`: T4 x2, internet on, model `google/gemma-3n/transformers/gemma-3n-e4b-it/2` |
| `build_notebook.py` | Rebuilds `notebook.ipynb` + `kernel-metadata.json` (input files embedded as base64) |
| `convert_validation.py` | `data/validation_questions_v2.json` → `validation.csv` + `validation_answers.csv` |
| `validation.csv` | 60 questions in official Test CSV format (ID,question,options) |
| `validation_answers.csv` | Reference answers (ID,answer) for `challenge score` |
| `my_hooks.py` | `--prompt-builder my_hooks:budget850` (budgeted fill to 850 passage tokens) |
| `output/` | Kernel outputs pulled with `kaggle kernels output` (arrives when the run finishes) |
| `PHASE2.md` | Results report (written when the run finishes) |

## Design decisions (why, not just what)

- **Model from the Kaggle hub, not HF.** `google/gemma-3n-E4B-it` is
  gated on Hugging Face (license + token). The Kaggle model hub hosts the
  same transformers weights (`google/gemma-3n/transformers/gemma-3n-e4b-it/2`,
  ~31.5 GB), attached as a kernel model source and mounted read-only under
  `/kaggle/input/`. No HF account, token, or license step.
- **vLLM at 0.70 GPU-memory utilisation.** The export also loads the 0.6B
  embedder + 0.6B reranker on cuda:0 (~2.5 GB); 0.70 leaves ~4.8 GB free per
  T4 for them. `--enforce-eager` avoids CUDA-graph compile risk on T4;
  generations are 8 tokens, so the speed cost is irrelevant.
- **Official `challenge export` for all variants.** The submission path is
  the measurement: `export` → `check` → `score` gives the exact challenge
  formula (0.7×accuracy + 0.3×token efficiency, 80% accuracy gate).
  `--resume` makes each export restartable from its `.partial` file.
- **256-token chunk variant is NOT reproducible on the fixed index.**
  The official collection is `text_baseline`, chunks ≤512 tokens
  (`bound_tokens 512`, verified from the ingest settings JSON on HF).
  There is no 256-token collection. The third variant is therefore `k3`
  (`--rerank-top-k 3`): fewer passages at the fixed 512-token chunk size —
  the closest faithful analogue. The b850 variant is the budgeted-fill
  analogue of phase-1's best lexical variant (512/b850, 664 tok/q).
- **Validation first, 60 questions, 3 variants in one session.** The index
  (~10.4 GB) is downloaded once and reused; generation is 60×8 tokens per
  variant, so setup dominates the clock. Session wall time ≈ GPU-hours;
  budget ≤6 h for this first pass.

## Re-run

```bash
cd ~/workspace/kaggle/telcoretrieve/phase2
python3 build_notebook.py            # rebuild notebook + metadata
/tmp/kaggle-venv/bin/kaggle kernels push -p .
/tmp/kaggle-venv/bin/kaggle kernels status coralfox12/telcoretrieve-phase-2
/tmp/kaggle-venv/bin/kaggle kernels output coralfox12/telcoretrieve-phase-2 -p output
```

## Status

- 2026-11-17 ~09:35 +07: kernel v1 pushed, status RUNNING
  (https://www.kaggle.com/code/coralfox12/telcoretrieve-phase-2).
  Polling in the background; outputs + PHASE2.md on completion.
