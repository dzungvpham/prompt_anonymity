#!/usr/bin/env python
"""Is "coding users are identified by topic, chat users aren't" real, or just a scale effect?

SWE-chat has ~125 known users and ~3k known conversations; WildChat has ~20k users and ~130k
conversations. Topics, crowds and candidate counts are not comparable at that difference in
scale. This script builds **SWE-chat-sized WildChat replicas**: for every SWE-chat user in the
window (with their known/unknown conversation counts), draw a WildChat user with the nearest
counts, without replacement, and keep only those users' conversations, in the same temporal
window. Then run the same two attacks on both, at the same K:

* full embedding, nearest neighbour (cosine, max over an author's documents, known-side z-score:
  the runner's ``nearest_neighbor``),
* topic only (own-data codebook, the runner's ``--unknown-snap-k`` + ``topic_histogram``),

and report users caught at least once (closed set, in-set target users). ``--validate`` runs the
lean attacks on full SWE-chat first, so they can be checked against the runner's numbers.

Usage::

    python experiments/thesis/matched_scale.py --ks 64 128 512 --replicates 20 --out out.csv
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


def zscore(known: np.ndarray, unknown: np.ndarray):
    center = known.astype(np.float64).mean(0)
    scale = known.astype(np.float64).std(0)
    scale = np.where(scale > 0, scale, 1.0)
    return (((known - center) / scale).astype(np.float32),
            ((unknown - center) / scale).astype(np.float32))


def author_max_scores(similarity: np.ndarray, codes: np.ndarray, n_authors: int) -> np.ndarray:
    scores = np.full((similarity.shape[0], n_authors), -np.inf, dtype=np.float32)
    for a in range(n_authors):
        columns = codes == a
        scores[:, a] = similarity[:, columns].max(1)
    return scores


def caught(scores: np.ndarray, truth: np.ndarray) -> float:
    """Share of target users with >= 1 conversation whose true author scores strictly best.

    A tie for best is a miss, as in the runner (tied ranks are averaged, so rank 1.5 > 1).
    """
    true_score = scores[np.arange(len(truth)), truth]
    hit = ((scores > true_score[:, None]).sum(1) == 0) & ((scores == true_score[:, None]).sum(1) == 1)
    return float(pd.Series(hit).groupby(truth).any().mean())


def evaluate(known_x, known_authors, unknown_x, unknown_authors, ks, seed):
    users, codes = np.unique(known_authors, return_inverse=True)
    index = {u: i for i, u in enumerate(users)}
    in_set = np.array([a in index for a in unknown_authors])
    unknown_x, truth = unknown_x[in_set], np.array([index[a] for a in unknown_authors[in_set]])

    # Full embedding: z-score on the known side, cosine, max over each author's documents.
    zk, zu = zscore(known_x, unknown_x)
    full = caught(author_max_scores(unit(zu) @ unit(zk).T, codes, len(users)), truth)

    result = {"target_users": len(np.unique(truth)), "known_users": len(users),
              "known_docs": len(known_x), "full": full}
    for k in ks:
        # Topic only: snap unknown rows to own-data centres, then the histogram attack, which
        # assigns (z-scored) known rows to the (z-scored) released centres.
        codebook = MiniBatchKMeans(n_clusters=k, random_state=seed, n_init=3, batch_size=4096)
        codebook.fit(unit(known_x))
        centres = unit(codebook.cluster_centers_)
        released = centres[codebook.predict(unit(unknown_x))]
        zk, zr = zscore(known_x, released)
        distinct, topic_of_query = np.unique(zr, axis=0, return_inverse=True)
        topic_of_known = (unit(zk) @ unit(distinct).T).argmax(1)
        counts = np.zeros((len(users), len(distinct)))
        np.add.at(counts, (codes, topic_of_known), 1.0)
        prior = (counts.sum(0) + 1) / (counts.sum() + len(distinct))
        log_hist = np.log((counts + prior) / (counts.sum(1, keepdims=True) + 1))
        result[f"topic_k{k}"] = caught(log_hist[:, np.ravel(topic_of_query)].T, truth)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", default="/Users/vandanasingh/prompt_anonymity/data/hf")
    parser.add_argument("--window", default="0075")
    parser.add_argument("--ks", type=int, nargs="+", default=[64, 128, 512])
    parser.add_argument("--replicates", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    swe, swe_x = load_documents_and_features(args.data_dir, "swe_chat", "gemini_embedding_2")
    _, s_known, s_unknown = known_configurations(len(swe), [args.window], 0.25)[0]
    s_authors = swe["author_id"].to_numpy()
    rows = []
    if args.validate:
        r = evaluate(swe_x[s_known], s_authors[s_known], swe_x[s_unknown], s_authors[s_unknown],
                     args.ks, args.seed)
        print("SWE-chat (lean attacks):", {k: round(v, 3) if isinstance(v, float) else v
                                          for k, v in r.items()}, flush=True)
        rows.append({"dataset": "swe_chat", "replicate": -1, **r})
        if args.replicates == 0:
            pd.DataFrame(rows).to_csv(args.out, index=False)
            return

    # SWE-chat users' (known, unknown) counts in this window: the profile to match.
    side = np.full(len(swe), "", dtype=object)
    side[np.arange(len(swe))[s_known]] = "known"
    side[np.arange(len(swe))[s_unknown]] = "unknown"
    profile = pd.crosstab(s_authors[side != ""], side[side != ""])

    wc, wc_x = load_documents_and_features(args.data_dir, "wildchat", "gemini_embedding_2")
    _, w_known, w_unknown = known_configurations(len(wc), [args.window], 0.25)[0]
    w_authors = wc["author_id"].to_numpy()
    w_side = np.full(len(wc), "", dtype=object)
    w_side[np.arange(len(wc))[w_known]] = "known"
    w_side[np.arange(len(wc))[w_unknown]] = "unknown"
    w_profile = pd.crosstab(w_authors[w_side != ""], w_side[w_side != ""])
    w_log = np.log1p(w_profile[["known", "unknown"]].to_numpy(dtype=float))

    rng = np.random.default_rng(args.seed)
    for replicate in range(args.replicates):
        taken = np.zeros(len(w_profile), dtype=bool)
        chosen = []
        for _, (n_known, n_unknown) in profile[["known", "unknown"]].sample(
                frac=1, random_state=int(rng.integers(1 << 31))).iterrows():
            distance = np.abs(w_log - np.log1p([n_known, n_unknown])).sum(1)
            distance[taken] = np.inf
            best = np.flatnonzero(distance == distance.min())
            pick = rng.choice(best)
            taken[pick] = True
            chosen.append(w_profile.index[pick])
        keep = np.isin(w_authors, chosen)
        k_rows = np.flatnonzero(keep & (w_side == "known"))
        u_rows = np.flatnonzero(keep & (w_side == "unknown"))
        r = evaluate(wc_x[k_rows], w_authors[k_rows], wc_x[u_rows], w_authors[u_rows],
                     args.ks, args.seed + replicate)
        rows.append({"dataset": "wildchat_matched", "replicate": replicate, **r})
        print(f"WildChat replica {replicate}:", {k: round(v, 3) if isinstance(v, float) else v
                                                for k, v in r.items()}, flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False)
    print(out.groupby("dataset").mean(numeric_only=True).round(3).to_string())


if __name__ == "__main__":
    main()
