#!/usr/bin/env python3
"""Build the mini index for the TelcoRetrieve scaffold.

Reads the two downloaded Rel-19 spec markdowns (TS 29.280 v19.0.0 and
TS 23.434 v19.x) and splits them into <=512-token chunks (cl100k_base,
mirroring the upstream text_baseline bound of 512 tokens), one chunk per
line of JSONL:

    {"id": "TS29280-c0001", "spec": "TS 29.280", "release": "19",
     "text": "...", "n_tokens": 231}

Usage:
    python3 scripts/build_mini_index.py
"""
import argparse
import json
import re
import sys
from pathlib import Path

import tiktoken

ENC = tiktoken.get_encoding("cl100k_base")
MAX_TOKENS = 512
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

SPECS = [
    ("TS29280.md", "TS 29.280", "19"),
    ("TS23434.md", "TS 23.434", "19"),
]


def count(text: str) -> int:
    return len(ENC.encode(text))


def split_blocks(md: str) -> list[str]:
    """Split markdown into paragraph-ish blocks (blank-line separated)."""
    blocks = [b.strip() for b in re.split(r"\n\s*\n", md)]
    return [b for b in blocks if b and len(b) > 20]


def chunk_blocks(blocks: list[str]) -> list[str]:
    """Greedily accumulate blocks into <=MAX_TOKENS chunks.

    A single over-long block is hard-split on sentence boundaries, then
    on plain character slices as a last resort.
    """
    chunks: list[str] = []
    cur: list[str] = []
    cur_tokens = 0

    def emit(text: str) -> None:
        """Append text, splitting by exact token count if over the bound."""
        ids = ENC.encode(text)
        if len(ids) <= MAX_TOKENS:
            chunks.append(text)
        else:
            for i in range(0, len(ids), MAX_TOKENS):
                chunks.append(ENC.decode(ids[i : i + MAX_TOKENS]))

    def flush() -> None:
        nonlocal cur, cur_tokens
        if cur:
            emit("\n\n".join(cur))
        cur, cur_tokens = [], 0

    for block in blocks:
        n = count(block)
        if n > MAX_TOKENS:
            flush()
            # hard-split the long block on sentences
            sentences = re.split(r"(?<=[.!?])\s+", block)
            piece: list[str] = []
            piece_tokens = 0
            for sent in sentences:
                sn = count(sent)
                if sn > MAX_TOKENS:
                    if piece:
                        emit(" ".join(piece))
                        piece, piece_tokens = [], 0
                    # last resort: split the long sentence by exact token count
                    ids = ENC.encode(sent)
                    for i in range(0, len(ids), MAX_TOKENS):
                        chunks.append(ENC.decode(ids[i : i + MAX_TOKENS]))
                    continue
                if piece_tokens + sn > MAX_TOKENS:
                    emit(" ".join(piece))
                    piece, piece_tokens = [], 0
                piece.append(sent)
                piece_tokens += sn
            if piece:
                emit(" ".join(piece))
            continue
        if cur_tokens + n > MAX_TOKENS:
            flush()
        cur.append(block)
        cur_tokens += n
    flush()
    return chunks


def main() -> int:
    out_path = DATA / "mini_index.jsonl"
    max_tokens = 512
    if len(sys.argv) > 1:
        ap = argparse.ArgumentParser(description=__doc__)
        ap.add_argument("--max-tokens", type=int, default=512)
        ap.add_argument("--out", default=str(DATA / "mini_index.jsonl"))
        args = ap.parse_args()
        max_tokens = args.max_tokens
        out_path = Path(args.out)
    global MAX_TOKENS
    MAX_TOKENS = max_tokens
    n_total = 0
    with open(out_path, "w", encoding="utf-8") as fh:
        for fname, spec, release in SPECS:
            path = DATA / fname
            if not path.exists():
                print(f"SKIP missing {path}", file=sys.stderr)
                continue
            md = path.read_text(encoding="utf-8", errors="replace")
            chunks = chunk_blocks(split_blocks(md))
            prefix = spec.replace(" ", "").replace(".", "")
            for i, text in enumerate(chunks, start=1):
                rec = {
                    "id": f"{prefix}-c{i:04d}",
                    "spec": spec,
                    "release": release,
                    "text": text,
                    "n_tokens": count(text),
                }
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_total += 1
            print(f"{spec}: {len(chunks)} chunks from {fname}")
    print(f"wrote {n_total} chunks -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
