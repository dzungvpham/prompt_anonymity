"""The headline top-k table, built on :class:`LinkageRanking`.

This module also held ``pool_size_sweep``, which re-scored the ranking against random sub-pools of
the identities to show how accuracy depends on the number of candidates. It was removed on
2026-08-04 with its only caller, the fixed-split experiment runner. The measurement itself did not
go: ``experiments/plot_results.py`` computes the same curve in closed form (``subpool_weights``)
straight from a run's CMC, which needs no sampling and no re-ranking.
"""

from __future__ import annotations

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
