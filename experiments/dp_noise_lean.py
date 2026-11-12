#!/usr/bin/env python
"""Memory-lean nearest-neighbour attack for the Laplace-noise sweep on large splits (WildChat).

Same data loading, temporal split (known = first 75%), known-side z-scoring and max-cosine
nearest-neighbour scoring as ``run_experiment.py --attacks nearest_neighbor --known-windows 0075``,
but it scores only the in-set unknown documents, a block at a time, keeping just each document's
best author. Peak memory is ~3 GB on WildChat instead of ~7.5 GB. It reports the two closed-set
numbers (``closed_set_top1`` per conversation and ``id_acc1``, the share of target users with at
least one conversation attributed correctly); run the noise-free case first and check that it
matches ``run_experiment.py`` before trusting the sweep.

Usage::

    python experiments/dp_noise_lean.py --source wildchat --data-dir data/hf --out data/dp_noise
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from run_experiment import (  # noqa: E402
    ZSCORE_ROW_BATCH, known_configurations, laplace_release, load_documents_and_features,
)


def zscore_stats(known: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    total = np.zeros(known.shape[1])
    for start in range(0, len(known), ZSCORE_ROW_BATCH):
        total += known[start:start + ZSCORE_ROW_BATCH].sum(axis=0, dtype=np.float64)
    center = total / len(known)
    squares = np.zeros(known.shape[1])
    for start in range(0, len(known), ZSCORE_ROW_BATCH):
        squares += ((known[start:start + ZSCORE_ROW_BATCH] - center) ** 2).sum(axis=0)
    scale = np.sqrt(squares / len(known))
    return center.astype(np.float32), np.where(scale > 0, scale, 1.0).astype(np.float32)


def zscore_unit_inplace(block: np.ndarray, center: np.ndarray, scale: np.ndarray) -> None:
    """Z-score with known-side stats, then scale rows to unit length (cosine), in place."""
    for start in range(0, len(block), ZSCORE_ROW_BATCH):
        rows = block[start:start + ZSCORE_ROW_BATCH]
        rows -= center
        rows /= scale
        rows /= np.maximum(np.linalg.norm(rows, axis=1, keepdims=True), 1e-12)


def attack(known: np.ndarray, known_codes: np.ndarray, n_authors: int,
           queries: np.ndarray, block: int = 512) -> np.ndarray:
    """Best author per query by max cosine over that author's known documents."""
    order = np.argsort(known_codes, kind="stable")
    known_sorted, codes_sorted = known[order], known_codes[order]
    starts = np.flatnonzero(np.r_[True, codes_sorted[1:] != codes_sorted[:-1]])
    assert len(starts) == n_authors
    best = np.empty(len(queries), dtype=np.int64)
    for start in range(0, len(queries), block):
        similarity = queries[start:start + block] @ known_sorted.T
        per_author = np.maximum.reduceat(similarity, starts, axis=1)
        best[start:start + block] = per_author.argmax(axis=1)
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", default="wildchat")
    parser.add_argument("--data-dir", default="data/hf")
    parser.add_argument("--out", default="data/dp_noise")
    parser.add_argument("--epsilons", type=float, nargs="+",
                        default=[1, 10, 100, 500, 1000, 2000, 5000, 10000, 20000, 100000])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    frame, embeddings = load_documents_and_features(args.data_dir, args.source,
                                                    "gemini_embedding_2")
    _, known_slice, unknown_slice = known_configurations(len(frame), ["0075"], 0.25)[0]
    known_labels = frame["author_id"].to_numpy()[known_slice]
    unknown_labels = frame["author_id"].to_numpy()[unknown_slice]
    authors, known_codes = np.unique(known_labels, return_inverse=True)
    in_set = np.isin(unknown_labels, authors)
    targets = unknown_labels[in_set]
    raw_queries = embeddings[unknown_slice][in_set].copy()   # only the scored rows, ~0.2 GB

    known = embeddings[known_slice]                          # a view; z-scored in place below
    center, scale = zscore_stats(known)
    zscore_unit_inplace(known, center, scale)
    print(f"{len(known):,} known docs / {len(authors):,} authors -> {len(targets):,} in-set "
          f"unknown docs / {len(np.unique(targets)):,} target users", flush=True)

    rows = []
    for eps in [None, *args.epsilons]:
        began = time.time()
        queries = (raw_queries.copy() if eps is None
                   else laplace_release(raw_queries, eps, "bound", args.seed))
        zscore_unit_inplace(queries, center, scale)
        correct = authors[attack(known, known_codes, len(authors), queries)] == targets
        per_user = pd.Series(correct).groupby(targets).any()
        row = {"epsilon": np.inf if eps is None else eps, "seed": args.seed,
               "closed_set_top1": float(correct.mean()), "id_acc1": float(per_user.mean()),
               "random_top1_known_pool": 1 / len(authors), "n_in_set": int(len(targets)),
               "n_known_authors": int(len(authors))}
        rows.append(row)
        print(f"eps={row['epsilon']:g}: top1={row['closed_set_top1']:.4f} "
              f"users={row['id_acc1']:.4f}  ({time.time() - began:.0f}s)", flush=True)
        pd.DataFrame(rows).to_csv(Path(args.out) / f"{args.source}_lean_results.csv", index=False)


if __name__ == "__main__":
    main()
