"""The headline top-k table, built on :class:`LinkageRanking`.

``pool_size_sweep`` (accuracy vs. candidate-pool size) used to live here; it's now computed in
closed form from a run's CMC by ``experiments/plot_results.py`` instead.
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
