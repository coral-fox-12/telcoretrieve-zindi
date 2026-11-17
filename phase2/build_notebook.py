"""Build the Phase 2 Kaggle notebook (notebook.ipynb) + kernel metadata.

The notebook is self-contained: validation.csv, validation_answers.csv and
my_hooks.py are embedded as base64 and written to /kaggle/working/phase2
at runtime. The 10 GB fixed index is pulled once per session with the
official CLI; Gemma 3n E4B-it comes from the Kaggle model hub (attached
via kernel-metadata model_sources), so no HF token / license step is needed.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

DB = "/kaggle/working/runs/telcoretrievechallenge/fixed-grid/text_baseline/otel-0.6b/db"
COLL = "otel-0.6b_text_baseline_b512_v256_full"
Q = "/kaggle/working/phase2/validation.csv"
REF = "/kaggle/working/phase2/validation_answers.csv"


def b64(name: str) -> str:
    return base64.b64encode((HERE / name).read_bytes()).decode()


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(True)}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(True),
    }


CELLS = [
    md(
        """# TelcoRetrieve — Phase 2: real-generator GPU run (60-question validation set)

Fixed generator **google/gemma-3n-E4B-it** served by vLLM on :8000, fixed
10 GB index (`otel-0.6b_text_baseline_b512_v256_full`), official CLI
`open-telco-rag challenge export` + `check` + `score`.

Variants (all on the official pipeline: hybrid retrieval, top-k 10,
OTel reranker, temperature 0, max 8 answer tokens, seed 1234):
- **control** — official defaults (rerank keeps 5)
- **b850** — defaults + budgeted context fill to 850 passage tokens
  (`--prompt-builder my_hooks:budget850`)
- **k3** — defaults + `--rerank-top-k 3`

NOTE on the phase-1 "256/k3" variant: the fixed index chunks are
**≤512 tokens** (text_baseline, bound_tokens 512) — there is no 256-token
collection in the official index, so the 256-token chunk size cannot be
reproduced here. `k3` is the closest faithful analogue on the fixed index:
fewer passages at the fixed 512-token chunk size.

Budget: ≤6 GPU-hours for this first pass. The session wall time is the
GPU time (T4 x2); it is printed at the end.
"""
    ),
    code(
        """import os, time
T0 = time.time()
WORK = "/kaggle/working"
P2 = f"{WORK}/phase2"
os.makedirs(P2, exist_ok=True)
open(f"{P2}/t0.txt", "w").write(str(T0))
print("WORK:", WORK)
!nvidia-smi --query-gpu=name,memory.total --format=csv
!df -h /kaggle/working | tail -1
!python3 -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '| gpus', torch.cuda.device_count())"
"""
    ),
    code(
        """# Install the official CLI (with local embedding) + vLLM.
# v1 failed in the first ~7 min (pip install or index pull — cell output
# was lost). v2 logs EVERYTHING to phase2/*.log so a failure is diagnosable
# from the kernel outputs, and verifies each install before moving on.
!echo "CELL install start $(date -u +%H:%M:%S)" >> /kaggle/working/phase2/run.log
!pip install -q --no-cache-dir "open-telco-rag[local-embed] @ git+https://github.com/Znbne/telco-retrieve" > /kaggle/working/phase2/pip-otr.log 2>&1; echo "open-telco-rag pip exit: $?"
!pip install -q --no-cache-dir vllm > /kaggle/working/phase2/pip-vllm.log 2>&1; echo "vllm pip exit: $?"
!python3 -c "import open_telco_rag; print('open-telco-rag OK')" 2>&1 | tail -2
!python3 -c "import vllm, torch; print('vllm', vllm.__version__, '| torch', torch.__version__, '| cuda', torch.cuda.is_available())" 2>&1 | tail -2
!python3 -c "import sentence_transformers, chromadb, tiktoken; print('sentence-transformers', sentence_transformers.__version__)" 2>&1 | tail -2
!pip list 2>/dev/null | grep -i -E "^(torch|transformers|vllm|sentence-transformers|chromadb|accelerate|numpy|pydantic) " | tee /kaggle/working/phase2/versions.txt
!open-telco-rag challenge export --help | head -5
!echo "CELL install done $(date -u +%H:%M:%S)" >> /kaggle/working/phase2/run.log
"""
    ),
    code(
        f"""# Write the input files (embedded as base64 so the notebook is self-contained).
import base64, pathlib
FILES = {{
    "validation.csv": "{b64('validation.csv')}",
    "validation_answers.csv": "{b64('validation_answers.csv')}",
    "my_hooks.py": "{b64('my_hooks.py')}",
}}
for name, blob in FILES.items():
    p = pathlib.Path("/kaggle/working/phase2") / name
    p.write_bytes(base64.b64decode(blob))
    print(name, p.stat().st_size, "bytes")
!ls -la /kaggle/working/phase2/
"""
    ),
    code(
        """# Pull the fixed 10 GB index once (cached for all 3 variants in this session).
# HF_HUB_CACHE is set explicitly: v1's download cache appeared under runs/
# (risking double disk use against the 20 GB working quota).
!echo "CELL pull-hf start $(date -u +%H:%M:%S)" >> /kaggle/working/phase2/run.log
!df -h /kaggle/working | tail -1
!cd /kaggle/working && HF_HUB_CACHE=/kaggle/working/hf-cache open-telco-rag pull-hf GSMA/telco-retrieve-chunks telcoretrievechallenge/fixed-grid/text_baseline/otel-0.6b --local-dir /kaggle/working/runs > /kaggle/working/phase2/pull.log 2>&1; echo "pull-hf exit: $?"; tail -5 /kaggle/working/phase2/pull.log
!ls -la /kaggle/working/runs/telcoretrievechallenge/fixed-grid/text_baseline/otel-0.6b/db/ | head
!du -sh /kaggle/working/runs /kaggle/working/hf-cache 2>/dev/null; df -h /kaggle/working | tail -1
!echo "CELL pull-hf done $(date -u +%H:%M:%S)" >> /kaggle/working/phase2/run.log
"""
    ),
    code(
        """# Serve Gemma 3n E4B-it with vLLM (model attached from the Kaggle model hub).
import glob, os, subprocess, time, urllib.request, json
open("/kaggle/working/phase2/run.log", "a").write(f"CELL vllm start {time.strftime('%H:%M:%S', time.gmtime())} UTC\\n")
cands = glob.glob("/kaggle/input/**/config.json", recursive=True)
print("config.json candidates:", cands)
print("/kaggle/input contents:", subprocess.run(["ls", "-R", "/kaggle/input"], capture_output=True, text=True).stdout[:1000])
matches = [c for c in cands if "gemma-3n-e4b-it" in c]
if not matches:
    raise SystemExit("MODEL NOT MOUNTED: /kaggle/input has no gemma-3n-e4b-it. "
        "The Kaggle account must accept the Gemma license at "
        "https://www.kaggle.com/models/google/gemma-3n first, then re-run. "
        "(Kaggle returns 403 on the model's files until the license is accepted.)")
MODEL_DIR = os.path.dirname(matches[0])
print("MODEL_DIR:", MODEL_DIR)
LOG = "/kaggle/working/phase2/vllm.log"
cmd = ["python", "-m", "vllm.entrypoints.openai.api_server",
       "--model", MODEL_DIR,
       "--served-model-name", "google/gemma-3n-E4B-it",
       "--port", "8000", "--max-model-len", "8192",
       "--tensor-parallel-size", "2", "--dtype", "bfloat16",
       "--gpu-memory-utilization", "0.70", "--enforce-eager",
       "--max-num-seqs", "4", "--disable-log-stats"]
logf = open(LOG, "w")
proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT)
print("vLLM pid:", proc.pid)
deadline = time.time() + 20 * 60
up = False
while time.time() < deadline:
    try:
        with urllib.request.urlopen("http://localhost:8000/v1/models", timeout=5) as r:
            data = json.loads(r.read())
            print("server up:", [m["id"] for m in data.get("data", [])])
            up = True
            break
    except Exception:
        time.sleep(15)
if not up:
    tail = "\\n".join(open(LOG).read().splitlines()[-100:])
    print("VLLM FAILED TO START — last 100 log lines:")
    print(tail)
    open("/kaggle/working/phase2/run.log", "a").write("CELL vllm FAILED\\n" + tail[-2000:] + "\\n")
    raise SystemExit("vLLM did not come up in 20 min")
open("/kaggle/working/phase2/run.log", "a").write(f"CELL vllm done {time.strftime('%H:%M:%S', time.gmtime())} UTC\\n")
"""
    ),
    code(
        """# Smoke-test generation (expect a TRUE/FALSE letter back).
import json, urllib.request
payload = {"model": "google/gemma-3n-E4B-it",
           "messages": [{"role": "system", "content": "You answer 3GPP questions from the provided context. Reply with only the answer."},
                        {"role": "user", "content": "Passages:\\nNone\\n\\nQuestion: Is the sky blue?\\n\\nAnswer with only TRUE or FALSE."}],
           "temperature": 0, "max_tokens": 8}
req = urllib.request.Request("http://localhost:8000/v1/chat/completions",
    data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req, timeout=120) as r:
    out = json.loads(r.read())
print("smoke reply:", repr(out["choices"][0]["message"]["content"]))
"""
    ),
    code(
        f"""# Variant 1/3 — control: official defaults (hybrid, top-k 10, rerank keeps 5).
!echo "CELL export-control start $(date -u +%H:%M:%S)" >> /kaggle/working/phase2/run.log
!cd /kaggle/working/phase2 && time LOCAL_LLM_BASE_URL=http://localhost:8000/v1 open-telco-rag challenge export --questions {Q} --out /kaggle/working/phase2/control/submission.csv --db-path {DB} --collection {COLL} --timeout 300 --resume > /kaggle/working/phase2/export-control.log 2>&1; echo "export-control exit: $?"; tail -5 /kaggle/working/phase2/export-control.log
!echo "CELL export-control done $(date -u +%H:%M:%S)" >> /kaggle/working/phase2/run.log
"""
    ),
    code(
        f"""# Variant 2/3 — b850: defaults + budgeted context fill to 850 passage tokens.
!echo "CELL export-b850 start $(date -u +%H:%M:%S)" >> /kaggle/working/phase2/run.log
!cd /kaggle/working/phase2 && time LOCAL_LLM_BASE_URL=http://localhost:8000/v1 open-telco-rag challenge export --questions {Q} --out /kaggle/working/phase2/b850/submission.csv --db-path {DB} --collection {COLL} --timeout 300 --resume --prompt-builder my_hooks:budget850 > /kaggle/working/phase2/export-b850.log 2>&1; echo "export-b850 exit: $?"; tail -5 /kaggle/working/phase2/export-b850.log
!echo "CELL export-b850 done $(date -u +%H:%M:%S)" >> /kaggle/working/phase2/run.log
"""
    ),
    code(
        f"""# Variant 3/3 — k3: defaults + rerank keeps 3 (fewer passages, fixed 512-token chunks).
!echo "CELL export-k3 start $(date -u +%H:%M:%S)" >> /kaggle/working/phase2/run.log
!cd /kaggle/working/phase2 && time LOCAL_LLM_BASE_URL=http://localhost:8000/v1 open-telco-rag challenge export --questions {Q} --out /kaggle/working/phase2/k3/submission.csv --db-path {DB} --collection {COLL} --timeout 300 --resume --rerank-top-k 3 > /kaggle/working/phase2/export-k3.log 2>&1; echo "export-k3 exit: $?"; tail -5 /kaggle/working/phase2/export-k3.log
!echo "CELL export-k3 done $(date -u +%H:%M:%S)" >> /kaggle/working/phase2/run.log
"""
    ),
    code(
        """# Check + score each variant with the official metrics.
!cd /kaggle/working/phase2 && for v in control b850 k3; do echo "===== $v : check ====="; open-telco-rag challenge check --submission /kaggle/working/phase2/$v/submission.csv --questions /kaggle/working/phase2/validation.csv; echo "===== $v : score (official) ====="; open-telco-rag challenge score --submission /kaggle/working/phase2/$v/submission.csv --reference /kaggle/working/phase2/validation_answers.csv; done
"""
    ),
    code(
        """# Independent summary: accuracy (overall / MC / TF), token efficiency,
# overall score, tok/q — recomputed from the submission CSVs (cross-checks
# the official `challenge score` numbers above).
import csv, json
import tiktoken
enc = tiktoken.get_encoding("cl100k_base")
ref = {r["ID"]: r["answer"] for r in csv.DictReader(open("/kaggle/working/phase2/validation_answers.csv"))}
qtype = {}
for r in csv.DictReader(open("/kaggle/working/phase2/validation.csv")):
    qtype[r["ID"]] = "TF" if not r["options"] else "MC"
summary = {}
for variant in ("control", "b850", "k3"):
    rows = list(csv.DictReader(open(f"/kaggle/working/phase2/{variant}/submission.csv")))
    total_tokens = sum(len(enc.encode(r["input_prompt"], disallowed_special=())) for r in rows)
    n = len(rows)
    correct = sum(1 for r in rows if r["answer"] == ref.get(r["ID"]))
    mc = [r for r in rows if qtype[r["ID"]] == "MC"]
    tf = [r for r in rows if qtype[r["ID"]] == "TF"]
    acc_mc = 100 * sum(1 for r in mc if r["answer"] == ref.get(r["ID"])) / max(1, len(mc))
    acc_tf = 100 * sum(1 for r in tf if r["answer"] == ref.get(r["ID"])) / max(1, len(tf))
    acc = 100 * correct / n
    eff = 100 * 850 * n / (850 * n + total_tokens)
    overall = 0.7 * acc + 0.3 * eff
    summary[variant] = {
        "n": n, "accuracy_pct": round(acc, 2), "accuracy_mc_pct": round(acc_mc, 2),
        "accuracy_tf_pct": round(acc_tf, 2), "token_efficiency_pct": round(eff, 2),
        "overall": round(overall, 2), "total_prompt_tokens": total_tokens,
        "tokens_per_question": round(total_tokens / n, 1),
        "gate_80pct_accuracy": acc >= 80,
    }
    print(variant, json.dumps(summary[variant]))
best = max(summary, key=lambda v: summary[v]["overall"])
print("WINNER (overall):", best, summary[best]["overall"])
json.dump(summary, open("/kaggle/working/phase2/summary.json", "w"), indent=2)
print("wrote /kaggle/working/phase2/summary.json")
"""
    ),
    code(
        """# Wrap-up: session wall time ~= GPU-hours on T4 x2.
import time
t0 = float(open("/kaggle/working/phase2/t0.txt").read())
elapsed_h = (time.time() - t0) / 3600
print(f"session wall time: {elapsed_h:.2f} h (~= GPU-hours on T4 x2, budget was 6 h)")
!df -h /kaggle/working | tail -1
!find /kaggle/working/phase2 -type f | sort
print("DONE — outputs in /kaggle/working/phase2/")
"""
    ),
]

NB = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"},
    },
    "nbformat": 4,
    "nbformat_minor": 4,
}

(HERE / "notebook.ipynb").write_text(json.dumps(NB, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"wrote notebook.ipynb: {len(CELLS)} cells")

META = {
    "id": "coralfox12/telcoretrieve-phase-2",
    "title": "TelcoRetrieve Phase 2",
    "code_file": "notebook.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": True,
    "enable_gpu": True,
    "enable_tpu": False,
    "enable_internet": True,
    "dataset_sources": [],
    "competition_sources": [],
    "kernel_sources": [],
    "model_sources": ["google/gemma-3n/transformers/gemma-3n-e4b-it/2"],
}
(HERE / "kernel-metadata.json").write_text(json.dumps(META, indent=2), encoding="utf-8")
print("wrote kernel-metadata.json")
