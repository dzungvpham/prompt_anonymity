#!/usr/bin/env python
"""Does content distinctiveness predict which users the real attack re-identifies? (Route B test)

Route B's user-level bound says an oracle attacker's error on a user shrinks like
``sum_v BC(P_u, P_v)^k``: what matters is how much a user's topic distribution overlaps with their
*closest* other user (Bhattacharyya coefficient, BC), and how many conversations they send (k).
This script measures, per target user in one known window:

* ``max_bc``   -- overlap of the user's known-side topic histogram with the most similar other
  known user's (k-means topics on the known side, smoothed histograms). Low = distinctive content.
* ``n_known`` / ``n_unknown`` -- how many conversations they have on each side.
* ``identified`` -- from an existing run's predictions: did the attack get at least one of their
  unknown conversations right at top-1? (and ``hit_rate``: share of their conversations right.)

and reports rank correlations and how well each quantity separates identified from not.

Usage::

    python experiments/thesis/route_b_per_user.py --source swe_chat --run <results dir> --k 64
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_experiment import known_configurations, load_documents_and_features  # noqa: E402


def unit(rows: np.ndarray) -> np.ndarray:
    rows = np.array(rows, dtype=np.float32)
    rows /= np.maximum(np.linalg.norm(rows, axis=1, keepdims=True), 1e-12)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", default="swe_chat")
    parser.add_argument("--data-dir", default="/Users/vandanasingh/prompt_anonymity/data/hf")
    parser.add_argument("--run", required=True, help="results dir with predictions_*_<window>.csv")
    parser.add_argument("--window", default="0075")
    parser.add_argument("--k", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-known", type=int, default=1,
                        help="Only score users with at least this many known conversations: a\n"
                             "histogram from 1-2 documents is mostly smoothing, which makes a\n"
                             "light user look undistinctive for lack of data.")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    frame, embeddings = load_documents_and_features(args.data_dir, args.source, "gemini_embedding_2")
    _, known, unknown = known_configurations(len(frame), [args.window], 0.25)[0]
    known_authors = frame["author_id"].to_numpy()[known]
    unknown_authors = frame["author_id"].to_numpy()[unknown]

    codebook = MiniBatchKMeans(n_clusters=args.k, random_state=args.seed, n_init=3, batch_size=4096)
    topics = codebook.fit_predict(unit(embeddings[known]))
    authors, codes = np.unique(known_authors, return_inverse=True)
    counts = np.zeros((len(authors), args.k))
    np.add.at(counts, (codes, topics), 1.0)
    prior = (counts.sum(0) + 1) / (counts.sum() + args.k)
    hist = (counts + prior) / (counts.sum(1, keepdims=True) + 1)   # smoothed, rows sum to 1
    root = np.sqrt(hist)
    bc = root @ root.T
    np.fill_diagonal(bc, -np.inf)
    max_bc = pd.Series(bc.max(1), index=authors)

    predictions = pd.read_csv(next(Path(args.run).glob(f"predictions_*_known{args.window}.csv")))
    predictions = predictions[predictions["author_in_known"]]
    per_user = predictions.assign(hit=predictions["true_author_rank"] <= 1).groupby("true_author")
    table = pd.DataFrame({
        "identified": per_user["hit"].any(),
        "hit_rate": per_user["hit"].mean(),
        "n_unknown": per_user.size(),
    })
    table["n_known"] = pd.Series(counts.sum(1), index=authors).reindex(table.index)
    table["max_bc"] = max_bc.reindex(table.index)
    table["distinctiveness"] = 1 - table["max_bc"]
    table = table[table["n_known"] >= args.min_known]

    print(f"{args.source} known{args.window}: {len(table)} target users, "
          f"{int(table['identified'].sum())} identified (top-1, >=1 conversation); K={args.k}, "
          f"min known {args.min_known}")
    for column in ("distinctiveness", "n_known", "n_unknown"):
        rho, p = spearmanr(table[column], table["hit_rate"])
        auc = roc_auc_score(table["identified"], table[column])
        print(f"  {column:15s} Spearman vs hit rate {rho:+.3f} (p={p:.3g})   "
              f"AUC for identified {auc:.3f}")
    halves = pd.qcut(table["distinctiveness"], 2, labels=["less distinctive", "more distinctive"])
    print(table.groupby(halves, observed=True)[["identified", "hit_rate", "n_known"]]
          .mean().round(3).to_string())
    if args.out:
        table.to_csv(args.out)


if __name__ == "__main__":
    main()
