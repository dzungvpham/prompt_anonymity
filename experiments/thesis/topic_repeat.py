#!/usr/bin/env python
"""The mechanism behind the content floor, in one number: do people come back to the same topics?

On one known window, cluster the known side into K topics (k-means, as ``--snap-codebook known``)
and put every unknown conversation of an in-set user into its topic. Then ask:

* ``own``      -- is this conversation's topic one the user has already talked about (at least one
  of their known conversations is in it)?
* ``stranger`` -- the same question against a *different* user with about as much history
  (same decile of known-conversation count, drawn at random). This is the baseline: topics that
  everyone uses are "repeats" for strangers too.

``own`` well above ``stranger`` means topics are personal and a topic id alone points at people;
``own`` close to ``stranger`` means topics are shared and the topic gives little away.

``crowd_median`` -- for the repeats, how many *other* known users have also used that topic. A
repeat points at you only if the crowd is small.

Usage::

    python experiments/thesis/topic_repeat.py --source swe_chat --ks 16 64 128 512
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans

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
    parser.add_argument("--window", default="0075")
    parser.add_argument("--ks", type=int, nargs="+", default=[16, 64, 128, 512])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    frame, embeddings = load_documents_and_features(args.data_dir, args.source, "gemini_embedding_2")
    _, known, unknown = known_configurations(len(frame), [args.window], 0.25)[0]
    authors = frame["author_id"].to_numpy()
    known_authors, unknown_authors = authors[known], authors[unknown]
    users, known_codes = np.unique(known_authors, return_inverse=True)
    index = {u: i for i, u in enumerate(users)}
    in_set = np.array([a in index for a in unknown_authors])
    query_codes = np.array([index[a] for a in unknown_authors[in_set]])
    n_known = np.bincount(known_codes, minlength=len(users))

    # A stranger for each query: another known user in the same decile of history size.
    rng = np.random.default_rng(args.seed)
    decile = pd.qcut(pd.Series(n_known).rank(method="first"), 10, labels=False).to_numpy()
    by_decile = {d: np.flatnonzero(decile == d) for d in range(10)}
    stranger = np.empty_like(query_codes)
    for i, u in enumerate(query_codes):
        pool = by_decile[decile[u]]
        pick = u
        while pick == u:
            pick = rng.choice(pool)
        stranger[i] = pick

    known_rows = unit(embeddings[known])
    query_rows = unit(embeddings[unknown][in_set])
    rows = []
    for k in args.ks:
        codebook = MiniBatchKMeans(n_clusters=k, random_state=args.seed, n_init=3, batch_size=4096)
        known_topics = codebook.fit_predict(known_rows)
        query_topics = codebook.predict(query_rows)
        seen = np.zeros((len(users), k), dtype=bool)
        seen[known_codes, known_topics] = True
        own = seen[query_codes, query_topics]
        other = seen[stranger, query_topics]
        # Crowd: how many *other* known users have also talked about this topic. A repeat only
        # identifies you if few others share the topic.
        crowd = seen.sum(0)[query_topics] - own
        row = {"k": k, "queries": len(query_codes), "users": len(np.unique(query_codes)),
               "own": float(own.mean()), "stranger": float(other.mean()),
               "own_user_mean": float(pd.Series(own).groupby(query_codes).mean().mean()),
               "stranger_user_mean": float(pd.Series(other).groupby(query_codes).mean().mean()),
               "crowd_median": float(np.median(crowd[own])) if own.any() else float("nan"),
               "crowd_share": float(np.median(crowd[own]) / (len(users) - 1)) if own.any() else float("nan"),
               "known_users": len(users)}
        rows.append(row)
        print(f"K={k:5d}  repeat own={row['own']:.3f}  stranger={row['stranger']:.3f}  "
              f"(per-user mean {row['own_user_mean']:.3f} vs {row['stranger_user_mean']:.3f}; "
              f"{row['queries']} conversations, {row['users']} users)  repeats share their topic with a "
              f"median {row['crowd_median']:.0f} of {len(users) - 1} other users", flush=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
