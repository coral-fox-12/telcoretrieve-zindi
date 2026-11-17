#!/usr/bin/env python3
"""Add a "parent" field to the 256-token mini index.

Each 256-token chunk is assigned the 512-token chunk with maximum
token-set containment (the chunk it overlaps most). NOTE: the
256-chunk boundaries do NOT strictly nest inside the 512-chunk
boundaries (greedy flush points differ), so a 256-chunk can straddle
two 512-chunks; the parent is the dominant one, and
evidence_in_context on the 256-index is therefore approximate.
Tiny fragments (<5 word-tokens, shards of a word from token-slice
splits, e.g. "auses.") get no parent — they carry no evidence value.

Parent matching is by token-set containment (not substring): the
256-chunking re-joins sentences with single spaces and re-decodes
token slices, so raw text can differ at boundaries while the tokens
are the same.

Usage:
    python3 scripts/add_parents_256.py
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokens(text: str) -> set[str]:
    return set(TOKEN_RE.findall(text.lower()))


def main() -> int:
    chunks512 = [json.loads(line) for line in open(DATA / "mini_index.jsonl", encoding="utf-8")]
    chunks256 = [json.loads(line) for line in open(DATA / "mini_index_256.jsonl", encoding="utf-8")]

    # Inverted index over 512-chunks, per spec.
    by_spec: dict[str, list[dict]] = {}
    for c in chunks512:
        by_spec.setdefault(c["id"].split("-")[0], []).append(c)
    posting: dict[str, dict[str, list[int]]] = {}
    toksets: dict[str, list[set[str]]] = {}
    for spec, cs in by_spec.items():
        posting[spec] = {}
        toksets[spec] = []
        for i, c in enumerate(cs):
            ts = tokens(c["text"])
            toksets[spec].append(ts)
            for t in ts:
                posting[spec].setdefault(t, []).append(i)

    n_parented = 0
    low = 0
    for c in chunks256:
        spec = c["id"].split("-")[0]
        ts = tokens(c["text"])
        if len(ts) < 5:
            c["parent"] = ""
            continue
        # Candidates: chunks sharing the rarest tokens of this chunk.
        df = {t: len(posting[spec].get(t, [])) for t in ts}
        probes = sorted(ts, key=lambda t: df[t])[:5]
        cand: set[int] = set()
        for t in probes:
            cand.update(posting[spec].get(t, []))
        best_id, best_cont = "", 0.0
        for i in cand:
            inter = len(ts & toksets[spec][i])
            cont = inter / len(ts)
            if cont > best_cont:
                best_cont, best_id = cont, by_spec[spec][i]["id"]
        c["parent"] = best_id
        if best_id:
            n_parented += 1
            if best_cont < 0.9:
                low += 1
        else:
            print(f"WARN no parent for {c['id']}", file=sys.stderr)
    print(f"256 chunks: {len(chunks256)}, parented: {n_parented}, containment <0.9: {low}")
    with open(DATA / "mini_index_256.jsonl", "w", encoding="utf-8") as fh:
        for c in chunks256:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"wrote parents -> {DATA / 'mini_index_256.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
