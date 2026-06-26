"""Experiment harnesses built on :class:`LinkageRanking`: the headline table and the
candidate-pool-size sweep.
"""

from __future__ import annotations

import random

import numpy as np
import pandas as pd

from .ranking import LinkageRanking


def headline_accuracy(ranking: LinkageRanking, *, top_ks=(1, 5, 10)) -> pd.DataFrame:
    """Top-k accuracy (conversation- and identity-level) vs. the random baseline, over the
    full candidate pool, one row per k.

    Columns: ``top``, ``n_identities``, ``conv_acc``, ``id_acc``, ``random_id``,
    ``advantage`` (``id_acc - random_id``).
    """
    n_identities = len(np.unique(ranking.unknown_labels))
    rows = []
    for k in top_ks:
        conv_acc = ranking.top_k_accuracy(k, level="conversation")
        id_acc = ranking.top_k_accuracy(k, level="identity")
        random_id = ranking.random_guessing_accuracy(k)
        rows.append(
            {
                "top": k,
                "n_identities": n_identities,
                "conv_acc": conv_acc,
                "id_acc": id_acc,
                "random_id": random_id,
                "advantage": id_acc - random_id,
            }
        )
    return pd.DataFrame(rows)


def pool_size_sweep(
    ranking: LinkageRanking, *, pool_step: int = 25, n_sims: int = 100, top_k: int = 1, seed: int = 47
) -> pd.DataFrame:
    """Measure top-k identity accuracy on random sub-pools of candidate users.

    For each pool size (``pool_step``, ``2*pool_step``, ... up to the full pool) this
    draws ``n_sims`` random subsets of identities and scores identity-level accuracy and
    the random baseline within each subset, tracing how re-identification changes with
    the number of candidate users. The full-pool point is deterministic (every subset is
    the whole pool), so its ``n_sims`` rows coincide.

    Columns: ``n`` (pool size), ``top``, ``id_acc``, ``random_id``.
    """
    identities = np.unique(ranking.unknown_labels)  # sorted -> reproducible sampling
    n_identities = len(identities)
    pool_sizes = list(range(pool_step, n_identities, pool_step)) + [n_identities]

    rng = random.Random(seed)
    rows = []
    for pool_size in pool_sizes:
        for _ in range(n_sims):
            # At the full pool every draw is the whole set; pass None for the fast path.
            subset = None if pool_size >= n_identities else rng.sample(identities.tolist(), pool_size)
            id_acc = ranking.top_k_accuracy(top_k, level="identity", candidate_identities=subset)
            random_id = ranking.random_guessing_accuracy(top_k, candidate_identities=subset)
            rows.append({"n": pool_size, "top": top_k, "id_acc": id_acc, "random_id": random_id})
    return pd.DataFrame(rows)
