#!/usr/bin/env python
"""Is topic-only attribution a lottery among the crowd? Per-conversation hit rate vs. crowd size.

For every in-set test conversation of one window, recompute its own-data topic (same codebook
as the runner: MiniBatchKMeans on the known side, same seed) and record:

* ``repeat`` -- the author has used this topic before (on the known side);
* ``crowd``  -- how many *other* known users have used it;
* ``hit``    -- the topic-only run ranked the true author first (joined on doc_id).

If the attacker could only tell "has used this topic" from "hasn't", a repeat would be a fair
lottery among the crowd: hit rate 1 / (crowd + 1). The histogram attacker also knows *how much*
each person uses the topic, so beating that line measures how much "how often" adds to "whether".

Usage::

    python experiments/thesis/crowd_lottery.py --source wildchat --k 1024 --run <topic run> --out x.csv
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
    parser.add_argument("--source", default="wildchat")
    parser.add_argument("--data-dir", default="/Users/vandanasingh/prompt_anonymity/data/hf")
    parser.add_argument("--window", default="0075")
    parser.add_argument("--k", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    frame, embeddings = load_documents_and_features(args.data_dir, args.source, "gemini_embedding_2")
    _, known, unknown = known_configurations(len(frame), [args.window], 0.25)[0]
    authors = frame["author_id"].to_numpy()
    users, codes = np.unique(authors[known], return_inverse=True)
    index = {u: i for i, u in enumerate(users)}
    codebook = MiniBatchKMeans(n_clusters=args.k, random_state=args.seed, n_init=3, batch_size=4096)
    known_topics = codebook.fit_predict(unit(embeddings[known]))
    unknown_frame = frame.iloc[unknown]
    unknown_topics = codebook.predict(unit(embeddings[unknown]))
    del embeddings

    seen = np.zeros((len(users), args.k), dtype=bool)
    seen[codes, known_topics] = True
    users_per_topic = seen.sum(0)

    predictions = pd.read_csv(next(Path(args.run).glob(f"predictions_*_known{args.window}.csv")),
                              usecols=["doc_id", "position", "author_in_known", "true_author_rank"])
    n_documents = int(predictions["position"].max()) + 1
    predictions = predictions[(predictions["position"] >= round(0.75 * n_documents))
                              & predictions["author_in_known"]
                              & predictions["true_author_rank"].notna()]
    table = pd.DataFrame({"doc_id": unknown_frame["doc_id"].to_numpy(),
                          "author": unknown_frame["author_id"].to_numpy(),
                          "topic": unknown_topics})
    table = table.merge(predictions[["doc_id", "true_author_rank"]], on="doc_id")
    table["hit"] = table["true_author_rank"] <= 1
    user_code = table["author"].map(index).to_numpy()
    table["repeat"] = seen[user_code, table["topic"].to_numpy()]
    table["crowd"] = users_per_topic[table["topic"].to_numpy()] - table["repeat"].astype(int)
    table.to_csv(args.out, index=False)

    print(f"{args.source} known{args.window} K={args.k}: {len(table)} in-set test conversations; "
          f"{table['repeat'].mean():.1%} repeats")
    print(f"hit rate: repeats {table.loc[table['repeat'], 'hit'].mean():.3f}, "
          f"non-repeats {table.loc[~table['repeat'], 'hit'].mean():.3f}")
    repeats = table[table["repeat"]]
    edges = [0, 1, 2, 4, 8, 16, 32, 64, 128, 10 ** 9]
    labels = ["0", "1", "2-3", "4-7", "8-15", "16-31", "32-63", "64-127", "128+"]
    bins = pd.cut(repeats["crowd"], edges, right=False, labels=labels)
    summary = repeats.groupby(bins, observed=True).agg(
        conversations=("hit", "size"), hit_rate=("hit", "mean"),
        lottery=("crowd", lambda c: float(np.mean(1.0 / (c + 1)))))
    summary["hit / lottery"] = summary["hit_rate"] / summary["lottery"]
    print("\nRepeats, by crowd size (other users who used the topic):")
    print(summary.round(3).to_string())


if __name__ == "__main__":
    main()
