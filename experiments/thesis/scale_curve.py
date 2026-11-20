#!/usr/bin/env python
"""How does linkage fall as the population grows? Topic only vs. full embedding, on WildChat.

matched_scale.py showed that a SWE-chat-sized slice of WildChat behaves like SWE-chat. This sweeps
the population size instead: draw N WildChat users at random (everyone they wrote, in the same
temporal window), run the full-embedding nearest-neighbour attack and the topic-only attack, and
report users caught at least once. Topic granularity is held fixed in *conversations per topic*
(``--docs-per-topic``, K = known conversations / that), so a larger population gets more topics
of the same size rather than coarser ones.

The attacks are the lean re-implementations of matched_scale.py (validated there against the
runner), with the author max computed by a sort + ``np.maximum.reduceat`` in row chunks so 10^4+
authors fit in memory.

Usage::

    python experiments/thesis/scale_curve.py --sizes 125 500 2000 8000 --out out.csv
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_experiment import known_configurations, load_documents_and_features  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from matched_scale import unit, zscore  # noqa: E402


def caught_from_max(known: np.ndarray, codes: np.ndarray, queries: np.ndarray,
                    truth: np.ndarray, chunk: int = 512) -> float:
    """Users caught at least once under max-linkage cosine scoring (ties = miss, as the runner)."""
    order = np.argsort(codes, kind="stable")
    known, codes = known[order], codes[order]
    starts = np.flatnonzero(np.r_[True, codes[1:] != codes[:-1]])
    hits = np.empty(len(queries), dtype=bool)
    for s in range(0, len(queries), chunk):
        scores = np.maximum.reduceat(queries[s:s + chunk] @ known.T, starts, axis=1)
        true = scores[np.arange(len(scores)), truth[s:s + chunk]]
        hits[s:s + chunk] = (((scores > true[:, None]).sum(1) == 0)
                             & ((scores == true[:, None]).sum(1) == 1))
    return float(pd.Series(hits).groupby(truth).any().mean())


def caught_topic(known_x, codes, n_users, unknown_x, truth, k, seed) -> float:
    codebook = MiniBatchKMeans(n_clusters=k, random_state=seed, n_init=3, batch_size=4096)
    codebook.fit(unit(known_x))
    centres = unit(codebook.cluster_centers_)
    released = centres[codebook.predict(unit(unknown_x))]
    zk, zr = zscore(known_x, released)
    distinct, topic_of_query = np.unique(zr, axis=0, return_inverse=True)
    topic_of_query = np.ravel(topic_of_query)
    zk_unit, distinct_unit = unit(zk), unit(distinct)
    topic_of_known = np.concatenate([(zk_unit[s:s + 8192] @ distinct_unit.T).argmax(1)
                                     for s in range(0, len(zk_unit), 8192)])
    counts = np.zeros((n_users, len(distinct)))
    np.add.at(counts, (codes, topic_of_known), 1.0)
    prior = (counts.sum(0) + 1) / (counts.sum() + len(distinct))
    log_hist = np.log((counts + prior) / (counts.sum(1, keepdims=True) + 1))
    hits = np.empty(len(truth), dtype=bool)
    for s in range(0, len(truth), 4096):
        scores = log_hist[:, topic_of_query[s:s + 4096]].T
        true = scores[np.arange(len(scores)), truth[s:s + 4096]]
        hits[s:s + 4096] = (((scores > true[:, None]).sum(1) == 0)
                            & ((scores == true[:, None]).sum(1) == 1))
    return float(pd.Series(hits).groupby(truth).any().mean())


def evaluate(known_x, known_authors, unknown_x, unknown_authors, docs_per_topic, seed):
    users, codes = np.unique(known_authors, return_inverse=True)
    index = {u: i for i, u in enumerate(users)}
    in_set = np.array([a in index for a in unknown_authors])
    unknown_x, truth = unknown_x[in_set], np.array([index[a] for a in unknown_authors[in_set]])
    zk, zu = zscore(known_x, unknown_x)
    full = caught_from_max(unit(zk), codes, unit(zu), truth)
    k = max(2, round(len(known_x) / docs_per_topic))
    topic = caught_topic(known_x, codes, len(users), unknown_x, truth, k, seed)
    return {"known_users": len(users), "target_users": len(np.unique(truth)),
            "known_docs": len(known_x), "k": k, "full": full, "topic": topic}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", default="/Users/vandanasingh/prompt_anonymity/data/hf")
    parser.add_argument("--source", default="wildchat")
    parser.add_argument("--window", default="0075")
    parser.add_argument("--sizes", type=int, nargs="+", default=[125, 500, 2000, 8000])
    parser.add_argument("--replicates", type=int, nargs="+", default=[10, 5, 3, 2],
                        help="replicates per size (same order as --sizes)")
    parser.add_argument("--docs-per-topic", type=float, default=23.4,
                        help="default = SWE-chat known0075 at K=128 (2,992 / 128)")
    parser.add_argument("--min-docs", type=int, default=1,
                        help="only draw users with at least this many conversations in the whole\n"
                             "corpus, to vary population size while holding user activity fixed")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--validate", action="store_true",
                        help="first run on the whole of --source, to check against the runner")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    frame, x = load_documents_and_features(args.data_dir, args.source, "gemini_embedding_2")
    _, known, unknown = known_configurations(len(frame), [args.window], 0.25)[0]
    authors = frame["author_id"].to_numpy()
    k_rows, u_rows = np.arange(len(frame))[known], np.arange(len(frame))[unknown]
    rows = []
    if args.validate:
        r = evaluate(x[k_rows], authors[k_rows], x[u_rows], authors[u_rows],
                     args.docs_per_topic, args.seed)
        print("whole corpus:", r, flush=True)
        rows.append({"size": "all", "replicate": 0, **r})
    totals = pd.Series(authors).value_counts()
    eligible = set(totals[totals >= args.min_docs].index)
    known_users = np.array([u for u in np.unique(authors[k_rows]) if u in eligible])
    print(f"{len(known_users)} known users with >= {args.min_docs} conversations", flush=True)
    rng = np.random.default_rng(args.seed)
    for size, reps in zip(args.sizes, args.replicates):
        for replicate in range(reps):
            started = time.time()
            chosen = rng.choice(known_users, min(size, len(known_users)), replace=False)
            kk = k_rows[np.isin(authors[k_rows], chosen)]
            uu = u_rows[np.isin(authors[u_rows], chosen)]
            r = evaluate(x[kk], authors[kk], x[uu], authors[uu], args.docs_per_topic,
                         args.seed + replicate)
            rows.append({"size": size, "replicate": replicate, **r})
            print(f"N={size} rep {replicate}: {r}  ({time.time() - started:.0f}s)", flush=True)
            pd.DataFrame(rows).to_csv(args.out, index=False)
    out = pd.DataFrame(rows)
    print(out.groupby("size", sort=False)[["known_users", "target_users", "k", "full", "topic"]]
          .mean().round(3).to_string())


if __name__ == "__main__":
    main()
