# Phase 2 — Kaggle GPU run with the REAL generator (google/gemma-3n-E4B-it)

## Status: BLOCKED on Gemma license acceptance (no results yet)

- Kernel: https://www.kaggle.com/code/coralfox12/telcoretrieve-phase-2
  (`coralfox12/telcoretrieve-phase-2`, T4 x2, internet on)
- **v1** pushed 2026-11-17 ~09:25 +07 → **ERROR 09:34 +07** (~9 min in).
  No logs survived (v1 only printed `tail -3` of pip output to cell output).
- **v2** pushed 2026-11-17 ~10:45 +07 → **ERROR 11:00 +07** (~13 min in).
  v2 logs everything to `phase2/*.log`, and the logs DID survive — see below.

## What v2 proved (from its surviving logs)

| Step | Result | Time (UTC) |
|---|---|---|
| Install `open-telco-rag[local-embed]` | OK (pip exit 0; only dependency-warning noise) | 03:47–03:51 |
| Install `vllm` | OK — vllm 0.34.0, torch 2.14.0+cu130, transformers 5.20.2, sentence-transformers 5.7.0, chromadb 1.5.9 | 03:51 |
| Pull 10 GB fixed index (`pull-hf`) | OK — 12/12 files in 39 s; 9.3 GB on disk, 11 GB free | 03:51–03:52 |
| vLLM cell | **FAILED instantly** — `config.json candidates: []` | 03:52 |

Full kernel log: `phase2/output/telcoretrieve-phase-2.log` (95 lines).
Run markers: `phase2/output/phase2/run.log`. Install versions:
`phase2/output/phase2/versions.txt`.

## Root cause: the Gemma model was never mounted (`/kaggle/input` empty)

The vLLM cell failed with `IndexError: list index out of range` at
`MODEL_DIR = os.path.dirname([c for c in cands if "gemma-3n-e4b-it" in c][0])`
because the glob found **zero** `config.json` files under `/kaggle/input`.

Verified from this VM (Kaggle API, account `coralfox12`):

- `model_instance_get('google/gemma-3n/transformers/gemma-3n-e4b-it')` → OK
  (slug `gemma-3n-e4b-it`, version 2, transformers framework). The instance
  exists and the `kernel-metadata.json` source string
  `google/gemma-3n/transformers/gemma-3n-e4b-it/2` is the correct form
  (the CLI push accepted it; the version-less form is rejected).
- `model_instance_version_files('.../gemma-3n-e4b-it/2')` → **HTTP 403
  Forbidden**. The account can see the instance metadata but cannot read
  its files. This is the Kaggle Gemma license gate: until the account
  accepts the Gemma license, the model cannot be attached to a kernel and
  the mount silently stays empty.

Cross-check on Hugging Face (same root cause, different platform):

- `google/gemma-3n-E4B-it` file downloads return **HTTP 401** without a
  token, and **HTTP 403** with the `coral-fox-12` fine-grained token
  (`coralfox-inference`, scope includes "read gated repos") — the HF
  account has not accepted the Gemma license either.
- The HF path is **not a workaround for the kernel anyway**: the official
  weights are fp32, ~31.5 GB, and the Kaggle working disk is 20 GB
  (9.3 GB already used by the index). The weights can only reach the
  kernel as a mounted model (input storage, separate quota).

## Unblock (needs a browser step — generic subagents cannot do this)

1. Sign in to Kaggle as **coralfox12**.
2. Open https://www.kaggle.com/models/google/gemma-3n and **accept the
   Gemma license** (the license prompt on the model page).
3. Verify: `kaggle models instances files
   google/gemma-3n/transformers/gemma-3n-e4b-it` returning a file list
   instead of HTTP 403.
4. Re-run the kernel — either re-run v2 from the Kaggle UI, or push the
   v3 notebook (already built locally, `phase2/notebook.ipynb`):
   `cd phase2 && /tmp/kaggle-venv/bin/kaggle kernels push -p .`
   (Rebuild the Kaggle venv first if `/tmp` was wiped:
   `python3 -m venv /tmp/kaggle-venv && /tmp/kaggle-venv/bin/pip install -q kaggle six`.)

Open question for the parent: the sibling Gemma 4 Phase 2 kernel also
mounts a Gemma model — if it mounts successfully, the license may
already be accepted for the family and something else is wrong; if it
also fails to mount, the license step above is confirmed as the fix.

## v3 notebook changes (built, not pushed)

- The vLLM cell now fails fast with an explicit message instead of a bare
  `IndexError`:
  `MODEL NOT MOUNTED: /kaggle/input has no gemma-3n-e4b-it. The Kaggle
  account must accept the Gemma license at
  https://www.kaggle.com/models/google/gemma-3n first, then re-run.`
  It also prints `ls -R /kaggle/input` for diagnosis.
- Everything else unchanged from v2 (per-cell logging to `phase2/*.log`
  + `run.log` markers, split pip installs with `--no-cache-dir` and
  import verification, `HF_HUB_CACHE=/kaggle/working/hf-cache` for the
  index pull).

## GPU-hours used so far

- v1: ~9 min × 2 GPUs ≈ 0.30 GPU-h. v2: ~13 min × 2 GPUs ≈ 0.43 GPU-h.
  Total ≈ **0.7 GPU-h** of the ~6 GPU-h first-pass budget. No results
  yet — the run never reached generation.

## Results (pending the unblock)

Per-variant accuracy / token efficiency / overall on the 60-question
validation set with the real generator — to be filled in once the kernel
completes. Stand-in ranking to test: 512/b850 first, then 256/k3
(k3 = `--rerank-top-k 3`; a true 256-token chunk variant is not
reproducible on the fixed 512-token index), then 512/k5 control.
Note: the Gemini stand-in sweep finished 12:11 +07 (runs/GEMINI_SWEEP.md)
and CONFIRMS the shortlist — raw overall G1 512/k5 73.91 > G2 512/b850
70.51 > G3 256/k3 67.79, but that ranking is distorted by empty answers
(G2 14/60 empties): answered-only accuracy is G2 100% (46/46) > G1 96.6%
> G3 86.5%, and 5 of 6 re-tested empties recovered correct at a 4096-token
output budget. The empties look like a stand-in artifact (Gemma 4's
thought consuming the 512-token output cap); the real Gemma 3n run is the
only way to confirm it doesn't transfer.
