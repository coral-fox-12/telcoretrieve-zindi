#!/usr/bin/env python3
"""Push this repo to GitHub (coral-fox-12) via the Contents API.

git push from this sandbox fails (sandbox exec writes flap), so files
go up through the REST API: create the repo if needed, then PUT each
file (base64) with its current sha when it already exists.

Excluded on purpose:
- data/Test.csv — competition data; download from Zindi after joining
  (the repo README says so). Never publish the test questions.
- __pycache__, scratch *.txt files at the repo root, t2/ (empty).

Usage:
    python3 scripts/push_github.py [--repo telcoretrieve-zindi] [--dry-run]

The PAT is read at runtime from ~/workspace/makefun/github-pat.txt
and never printed or committed.
"""
import argparse
import base64
import json
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
OWNER = "coral-fox-12"
PAT_PATH = Path.home() / "workspace/makefun/github-pat.txt"
API = "https://api.github.com"

INCLUDE = [
    "README.md",
    "RUNBOOK.md",
    ".gitignore",
    "src/__init__.py",
    "src/telco/__init__.py",
    "src/telco/evaluate.py",
    "src/telco/generate.py",
    "src/telco/prompt.py",
    "src/telco/questions.py",
    "src/telco/retrieval.py",
    "src/telco/scoring.py",
    "src/telco/submission.py",
    "src/telco/tokens.py",
    "tests/__init__.py",
    "tests/test_telco.py",
    "scripts/build_mini_index.py",
    "scripts/add_parents_256.py",
    "scripts/verify_validation.py",
    "scripts/merge_reports.py",
    "scripts/run_llm_variants.sh",
    "scripts/push_github.py",
    "data/SampleSubmission.csv",
    "data/VariableDefinitions.csv",
    "data/TS29280.md",
    "data/TS23434.md",
    "data/mini_index.jsonl",
    "data/mini_index_256.jsonl",
    "data/validation_questions.json",
    "data/validation_questions_v2.json",
    "data/validation_part_a.json",
    "data/validation_part_b.json",
]


def run_files(root: Path) -> list[str]:
    files = list(INCLUDE)
    if (root / "runs/PHASE1.md").exists():
        files.append("runs/PHASE1.md")
    for run_dir in sorted((root / "runs").iterdir()):
        if not run_dir.is_dir():
            continue
        for name in ("report.json", "submission.csv"):
            p = run_dir / name
            if p.exists():
                files.append(str(p.relative_to(root)))
    return files


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", default="telcoretrieve-zindi")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    pat = PAT_PATH.read_text(encoding="utf-8").strip()
    headers = {
        "Authorization": f"token {pat}",
        "Accept": "application/vnd.github+json",
    }
    session = requests.Session()
    session.headers.update(headers)

    files = run_files(ROOT)
    missing = [f for f in files if not (ROOT / f).exists()]
    if missing:
        print(f"MISSING files, aborting: {missing}", file=sys.stderr)
        return 1
    print(f"{len(files)} files to push to {OWNER}/{args.repo}")
    if args.dry_run:
        for f in files:
            print(f"  would push {f} ({(ROOT / f).stat().st_size} B)")
        return 0

    r = session.get(f"{API}/repos/{OWNER}/{args.repo}")
    if r.status_code == 404:
        r = session.post(
            f"{API}/user/repos",
            json={
                "name": args.repo,
                "description": "TelcoRetrieve (Zindi, €25K) — RAG pipeline improvement for 3GPP Rel-19 QA: lean prompts, 60-question validation set, reproducible local eval loop.",
                "private": False,
                "has_issues": True,
                "has_projects": False,
                "has_wiki": False,
            },
        )
        r.raise_for_status()
        print(f"created repo {OWNER}/{args.repo}")
    elif r.status_code != 200:
        print(f"repo check failed: {r.status_code} {r.text[:300]}", file=sys.stderr)
        return 1
    else:
        print(f"repo {OWNER}/{args.repo} exists, updating files")

    pushed, updated = 0, 0
    for rel in files:
        path = ROOT / rel
        content = base64.b64encode(path.read_bytes()).decode()
        url = f"{API}/repos/{OWNER}/{args.repo}/contents/{rel}"
        existing = session.get(url)
        payload: dict = {
            "message": f"Phase 1: 60-question validation set + lean-prompt variants ({rel})",
            "content": content,
        }
        if existing.status_code == 200:
            payload["sha"] = existing.json()["sha"]
            updated += 1
        elif existing.status_code != 404:
            print(f"  !! GET {rel} -> {existing.status_code}", file=sys.stderr)
            return 1
        resp = session.put(url, json=payload)
        if resp.status_code not in (200, 201):
            print(f"  !! PUT {rel} -> {resp.status_code}: {resp.text[:300]}", file=sys.stderr)
            return 1
        pushed += 1
        print(f"  pushed {rel} ({path.stat().st_size} B)")
    print(f"done: {pushed} files pushed ({updated} updated), 0 failed")
    print(f"https://github.com/{OWNER}/{args.repo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
