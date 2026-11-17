#!/bin/bash
# LLM stand-in variant sweep on the 60-question validation set.
# Model: nemotron-3-nano-omni (the 12/12 baseline model; gemma-4 is
# 429-limited upstream as of 2026-11-17). Reasoning model -> max_tokens 512.
# Stops the sweep if a run aborts (free-quota exhaustion); partial
# reports are still saved by evaluate.py.
set -u
cd ~/workspace/kaggle/telcoretrieve
export PYTHONPATH=src
MODEL="nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free"
LOG=runs/llm_driver.log
: > "$LOG"

run_variant() {
  name=$1; shift
  echo "=== $name $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a "$LOG"
  python3 -m telco.evaluate --questions data/validation_questions_v2.json \
    --generator openrouter --model "$MODEL" --max-tokens 512 "$@" \
    --out "runs/$name" >> "$LOG" 2>&1
  if python3 -c "import json,sys; sys.exit(0 if json.load(open('runs/$name/report.json')).get('aborted') else 1)" 2>/dev/null; then
    echo "ABORTED: $name — stopping driver (see runs/$name/report.json)" | tee -a "$LOG"
    return 1
  fi
  return 0
}

run_variant M1_512_k5_llm   --index data/mini_index.jsonl      --top-k 10 --rerank-top-k 5   || exit 1
run_variant M2_512_b850_llm --index data/mini_index.jsonl      --top-k 10 --context-budget 850 || exit 1
run_variant M3_256_k3_llm   --index data/mini_index_256.jsonl  --top-k 10 --rerank-top-k 3   || exit 1
run_variant M4_256_b850_llm --index data/mini_index_256.jsonl  --top-k 10 --context-budget 850 || exit 1
run_variant M5_256_k2_llm   --index data/mini_index_256.jsonl  --top-k 10 --rerank-top-k 2   || exit 1
run_variant M6_256_k5_llm   --index data/mini_index_256.jsonl  --top-k 10 --rerank-top-k 5   || exit 1
echo "ALL DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG"
