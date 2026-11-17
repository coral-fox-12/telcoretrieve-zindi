"""TelcoRetrieve Challenge scaffold.

Implements the challenge loop with the exact semantics of the upstream
TelcoRetrieve pipeline (Apache 2.0, github.com/Znbne/telco-retrieve):
fixed generator google/gemma-3n-E4B-it, fixed OTel embedder/reranker,
score = 0.7 * accuracy + 0.3 * token_efficiency, min accuracy 80%,
prompt cap 4096 cl100k_base tokens, submission columns ID,input_prompt,answer.

This scaffold is the *local* development harness: BM25 lexical retrieval
over the small mini index (data/mini_index.jsonl) and a swappable
generator (OpenRouter :free stand-in for local runs; the real Gemma 3n
runs on Kaggle GPU per RUNBOOK.md). The prompt, token counting,
submission validation and scoring mirror the official CLI exactly so
numbers measured here transfer to the official `challenge export/check/score`.
"""
