#!/usr/bin/env python
"""Per-user test of the crowd explanation: are users whose topics few others share the ones caught?

On one known window, fit the same own-data codebook as ``run_experiment.py --unknown-snap-k K
--snap-codebook known`` (MiniBatchKMeans on the known side, same seed) and, per in-set target user:

* ``crowd``      -- median, over the user's *known* conversations, of how many *other* known users
  have also talked about that conversation's topic. Small = the user's topics are niche.
* ``repeat``     -- share of the user's *unknown* conversations whose topic is in their own known
  history.
* ``n_known`` / ``n_unknown`` -- history size and how many chances the attacker gets.
* ``caught_<run>`` -- for each predictions directory given: was at least one of the user's
  unknown conversations attributed to them at top-1?

Because heavy users are both easier to catch and tend to have more varied topics, the crowd effect
is reported *within* bins of n_unknown (so users are compared to others with as many chances), plus
a logistic regression with both counts as controls.

Usage::

    python experiments/thesis/crowd_per_user.py --source wildchat --k 1024 \\
        --run full=<nn results dir> --run topic=<topic-only results dir> --out users.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_experiment import known_configurations, load_documents_and_features  # noqa: E402


def unit(rows: np.ndarray) -> np.ndarray:
    rows = np.array(rows, dtype=np.float32)
    rows /= np.maximum(np.linalg.norm(rows, axis=1, keepdims=True), 1e-12)
    return rows


def caught(run_dir: str, window: str) -> pd.Series:
    path = next(Path(run_dir).glob(f"predictions_*_known{window}.csv"))
    table = pd.read_csv(path, usecols=["true_author", "author_in_known", "true_author_rank"])
    table = table[table["author_in_known"]]
    return (table["true_author_rank"] <= 1).groupby(table["true_author"]).any()


def logit_z(frame: pd.DataFrame, outcome: str, columns: list[str]) -> dict[str, tuple[float, float]]:
    """Logistic regression coefficients and z-scores (Newton/IRLS, standardized predictors)."""
    x = frame[columns].to_numpy(dtype=float)
    x = (x - x.mean(0)) / x.std(0)
    x = np.column_stack([np.ones(len(x)), x])
    y = frame[outcome].to_numpy(dtype=float)
    beta = np.zeros(x.shape[1])
    for _ in range(50):
        p = 1 / (1 + np.exp(-x @ beta))
        hessian = x.T @ (x * (p * (1 - p))[:, None]) + 1e-8 * np.eye(len(beta))
        step = np.linalg.solve(hessian, x.T @ (y - p))
        beta += step
        if np.abs(step).max() < 1e-8:
            break
    se = np.sqrt(np.diag(np.linalg.inv(hessian)))
    return {c: (beta[i + 1], beta[i + 1] / se[i + 1]) for i, c in enumerate(columns)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", default="swe_chat")
    parser.add_argument("--data-dir", default="/Users/vandanasingh/prompt_anonymity/data/hf")
    parser.add_argument("--window", default="0075")
    parser.add_argument("--k", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run", action="append", default=[], metavar="NAME=DIR")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    frame, embeddings = load_documents_and_features(args.data_dir, args.source, "gemini_embedding_2")
    _, known, unknown = known_configurations(len(frame), [args.window], 0.25)[0]
    authors = frame["author_id"].to_numpy()
    users, known_codes = np.unique(authors[known], return_inverse=True)
    index = {u: i for i, u in enumerate(users)}
    unknown_authors = authors[unknown]
    in_set = np.array([a in index for a in unknown_authors])
    query_codes = np.array([index[a] for a in unknown_authors[in_set]])

    codebook = MiniBatchKMeans(n_clusters=args.k, random_state=args.seed, n_init=3, batch_size=4096)
    known_topics = codebook.fit_predict(unit(embeddings[known]))
    query_topics = codebook.predict(unit(embeddings[unknown][in_set]))
    del embeddings

    seen = np.zeros((len(users), args.k), dtype=bool)
    seen[known_codes, known_topics] = True
    users_per_topic = seen.sum(0)
    # Other users sharing each known conversation's topic (the user themself always counts once).
    crowd_of_doc = users_per_topic[known_topics] - 1
    targets = np.unique(query_codes)
    table = pd.DataFrame({
        "crowd": pd.Series(crowd_of_doc).groupby(known_codes).median().reindex(targets).to_numpy(),
        "repeat": pd.Series(seen[query_codes, query_topics]).groupby(query_codes).mean()
                    .reindex(targets).to_numpy(),
        "n_known": np.bincount(known_codes)[targets],
        "n_unknown": np.bincount(query_codes)[targets],
    }, index=pd.Index(users[targets], name="user"))
    for spec in args.run:
        name, run_dir = spec.split("=", 1)
        table[f"caught_{name}"] = caught(run_dir, args.window).reindex(table.index).fillna(False)

    print(f"{args.source} known{args.window} K={args.k}: {len(table)} target users "
          f"({len(users) - 1} other known users); median crowd {table['crowd'].median():.0f}")
    bins = pd.qcut(table["n_unknown"].rank(method="first"), 3, labels=["few", "middle", "many"])
    niche = table["crowd"] <= table.groupby(bins, observed=True)["crowd"].transform("median")
    for spec in args.run:
        name = spec.split("=", 1)[0]
        column = f"caught_{name}"
        auc = roc_auc_score(table[column], -table["crowd"]) if table[column].nunique() > 1 else np.nan
        print(f"\n[{name}] caught overall {table[column].mean():.3f}; "
              f"AUC of small crowd for caught {auc:.3f}")
        summary = table.assign(bin=bins, niche=np.where(niche, "niche topics", "crowded topics"))
        pivot = summary.pivot_table(index="bin", columns="niche", values=column, aggfunc="mean",
                                    observed=True)
        pivot["users"] = summary.groupby("bin", observed=True).size()
        pivot["conversations"] = summary.groupby("bin", observed=True)["n_unknown"].agg(
            lambda s: f"{s.min()}-{s.max()}")
        print(pivot.round(3).to_string())
        fit = table.assign(log_crowd=np.log1p(table["crowd"]), log_known=np.log(table["n_known"]),
                           log_unknown=np.log(table["n_unknown"]))
        if fit[column].nunique() > 1:
            for c, (b, z) in logit_z(fit, column, ["log_crowd", "log_known", "log_unknown"]).items():
                print(f"  logit {c:12s} coef {b:+.3f}  z {z:+.2f}")
    table.to_csv(args.out)


if __name__ == "__main__":
    main()
