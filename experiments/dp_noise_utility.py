#!/usr/bin/env python
"""Utility side of the Laplace-noise sweep: how much of the embedding's meaning survives?

For a sample of conversations, compare the clean and noisy embeddings on two proxies:

* ``cosine_to_original`` -- cosine between the noisy release and the clean unit vector.
* ``topic_neighbors_kept_at10`` -- overlap between the 10 nearest *other users'* conversations
  found by the clean vs. the noisy embedding. Other users only, so this measures "what is this
  about" with the identity signal excluded.

Then joins with ``<source>_dp_noise_results.csv`` (from ``dp_noise_sweep.py``) and plots
re-identification kept vs. usefulness kept.

Usage::

    python experiments/dp_noise_utility.py --source swe_chat --data-dir data/hf --out data/dp_noise
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parent))
from run_experiment import laplace_release  # noqa: E402

EPSILONS = (1, 10, 100, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000)


def load_unit_embeddings(path: Path, pool_size: int | None, seed: int):
    table = pq.read_table(path)
    authors = table.column("author_id").to_numpy()
    rows = np.arange(len(authors))
    if pool_size is not None and pool_size < len(rows):
        rows = np.sort(np.random.default_rng(seed).choice(rows, pool_size, replace=False))
    columns = [c for c in table.column_names if c not in ("doc_id", "author_id")]
    matrix = np.empty((len(rows), len(columns)), dtype=np.float32)
    for i, column in enumerate(columns):
        matrix[:, i] = table.column(column).to_numpy()[rows]
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix, authors[rows]


def utility_table(unit, authors, n_queries: int, k: int, epsilons, seed: int) -> pd.DataFrame:
    queries = np.random.default_rng(seed).choice(len(unit), n_queries, replace=False)
    same_author = authors[queries][:, None] == authors[None, :]
    clean = unit[queries] @ unit.T
    clean[same_author] = -np.inf
    clean_top = np.argpartition(-clean, k, axis=1)[:, :k]
    rows = []
    for eps in epsilons:
        noisy = laplace_release(unit[queries], eps, "bound", seed)
        cosine = (noisy * unit[queries]).sum(1) / np.linalg.norm(noisy, axis=1)
        scores = noisy @ unit.T
        scores[same_author] = -np.inf
        top = np.argpartition(-scores, k, axis=1)[:, :k]
        kept = np.mean([len(set(a) & set(b)) / k for a, b in zip(clean_top, top)])
        rows.append({"epsilon": float(eps), "cosine_to_original": float(cosine.mean()),
                     "topic_neighbors_kept_at10": float(kept)})
        print(f"eps={eps:g}: cosine={cosine.mean():.3f}  neighbors kept={kept:.3f}")
    return pd.DataFrame(rows)


def plot(source: str, out: Path, utility: pd.DataFrame) -> pd.DataFrame:
    results = pd.read_csv(out / f"{source}_dp_noise_results.csv")
    base = results[~np.isfinite(results["epsilon"])].iloc[0]
    merged = (results[np.isfinite(results["epsilon"])].groupby("epsilon")
              [["closed_set_top1", "margin_dir_at_far10"]].mean().reset_index()
              .merge(utility, on="epsilon"))
    merged["reid_kept"] = merged["closed_set_top1"] / base["closed_set_top1"]
    merged.round(4).to_csv(out / f"{source}_tradeoff.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    axes[0].plot(merged["epsilon"], merged["reid_kept"], "o-",
                 label="Re-identification (top-1, share of no-noise level)")
    axes[0].plot(merged["epsilon"], merged["topic_neighbors_kept_at10"], "s-",
                 label="Usefulness (similar chats from other users still found)")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("DP epsilon (per conversation)")
    axes[0].set_ylabel("fraction kept")
    axes[0].legend(fontsize=8, loc="center left")
    axes[0].set_title("Both come back at the same epsilon", fontsize=10)

    axes[1].plot(merged["topic_neighbors_kept_at10"], merged["reid_kept"], "o-")
    for _, row in merged.iterrows():
        if row["epsilon"] in (500, 1000, 2000, 5000, 10000, 100000):
            axes[1].annotate(f"ε={row['epsilon']:g}",
                             (row["topic_neighbors_kept_at10"], row["reid_kept"]),
                             textcoords="offset points", xytext=(5, -10), fontsize=8)
    axes[1].plot([0, 1], [0, 1], "k:", label="re-ID = usefulness")
    axes[1].set_xlabel("usefulness kept")
    axes[1].set_ylabel("re-identification kept")
    axes[1].legend(fontsize=8)
    axes[1].set_title("Above the diagonal = re-ID returns faster than usefulness", fontsize=10)
    fig.suptitle(f"{source}: Laplace noise on unknown Gemini embeddings")
    fig.tight_layout()
    fig.savefig(out / f"{source}_tradeoff.png", dpi=150)
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", default="swe_chat")
    parser.add_argument("--data-dir", default="data/hf")
    parser.add_argument("--out", default="data/dp_noise")
    parser.add_argument("--pool-size", type=int, default=None,
                        help="Subsample the retrieval pool (WildChat is large).")
    parser.add_argument("--queries", type=int, default=800)
    parser.add_argument("--epsilons", type=float, nargs="+", default=list(EPSILONS))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out = Path(args.out)
    unit, authors = load_unit_embeddings(
        Path(args.data_dir) / f"{args.source}_gemini_embedding_2.parquet", args.pool_size, args.seed)
    utility = utility_table(unit, authors, args.queries, 10, args.epsilons, args.seed)
    utility.to_csv(out / f"{args.source}_utility.csv", index=False)
    print(plot(args.source, out, utility).round(3).to_string())


if __name__ == "__main__":
    main()
