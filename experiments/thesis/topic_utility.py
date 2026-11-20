#!/usr/bin/env python
"""Utility side of the content floor: how much of a conversation survives a topic-only release?

For each codebook size K (and codebook source, matching ``run_experiment.py --unknown-snap-k
--snap-codebook``), on one known window, release every unknown conversation as its topic centre
and measure two embedding-level proxies -- free to compute, but proxies, not answer quality:

* ``cosine``  -- cosine between the conversation and the centre it is released as.
* ``neighbors_kept`` -- of the 10 most similar conversations *by other users* (in the whole
  corpus) found by the real embedding, the share the centre still finds. Other users only, so this
  is "is it still about the same thing" with the identity signal left out.

Usage::

    python experiments/thesis/topic_utility.py --source swe_chat --ks 4 16 64 128 256 512
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_experiment import (known_configurations, load_documents_and_features,  # noqa: E402
                            snap_to_codebook)


def unit(rows: np.ndarray) -> np.ndarray:
    rows = np.array(rows, dtype=np.float32)
    rows /= np.maximum(np.linalg.norm(rows, axis=1, keepdims=True), 1e-12)
    return rows


def neighbors(queries: np.ndarray, pool: np.ndarray, excluded: np.ndarray, k: int) -> np.ndarray:
    """Top-k pool rows per query, skipping (query, pool) pairs marked in ``excluded``."""
    out = np.empty((len(queries), k), dtype=np.int64)
    for start in range(0, len(queries), 256):
        scores = queries[start:start + 256] @ pool.T
        scores[excluded[start:start + 256]] = -np.inf
        out[start:start + 256] = np.argpartition(-scores, k, axis=1)[:, :k]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", default="swe_chat")
    parser.add_argument("--data-dir", default="/Users/vandanasingh/prompt_anonymity/data/hf")
    parser.add_argument("--window", default="0075")
    parser.add_argument("--ks", type=int, nargs="+", default=[4, 16, 64, 128, 256, 512])
    parser.add_argument("--queries", type=int, default=1000)
    parser.add_argument("--pool", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    frame, embeddings = load_documents_and_features(args.data_dir, args.source, "gemini_embedding_2")
    _, known, unknown = known_configurations(len(frame), [args.window], 0.25)[0]
    authors = frame["author_id"].to_numpy()
    rng = np.random.default_rng(args.seed)
    unknown_rows = np.arange(len(frame))[unknown]
    queries = rng.choice(unknown_rows, min(args.queries, len(unknown_rows)), replace=False)
    pool_rows = (np.arange(len(frame)) if len(frame) <= args.pool
                 else np.sort(rng.choice(len(frame), args.pool, replace=False)))
    pool = unit(embeddings[pool_rows])
    clean = unit(embeddings[queries])
    excluded = authors[queries][:, None] == authors[pool_rows][None, :]
    clean_top = neighbors(clean, pool, excluded, 10)

    known_frame = frame.iloc[known]
    released_users = set(frame.iloc[unknown]["author_id"])
    rows = []
    for codebook in ("known", "other_users"):
        source = embeddings[known]
        if codebook == "other_users":
            source = source[~known_frame["author_id"].isin(released_users).to_numpy()]
        for k in args.ks:
            if k > len(source):
                print(f"skip {codebook} K={k}: only {len(source)} codebook documents")
                continue
            centres = unit(snap_to_codebook(source, embeddings[queries], k, args.seed))
            top = neighbors(centres, pool, excluded, 10)
            kept = np.mean([len(set(a) & set(b)) / 10 for a, b in zip(clean_top, top)])
            row = {"codebook": codebook, "k": k, "cosine": float((centres * clean).sum(1).mean()),
                   "neighbors_kept": float(kept), "codebook_docs": len(source)}
            rows.append(row)
            print(f"{codebook:12s} K={k:5d}  cosine={row['cosine']:.3f}  "
                  f"neighbors kept={row['neighbors_kept']:.3f}", flush=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
