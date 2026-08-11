"""Low-level ranking helpers shared by the metrics and the evaluation primitives.

Kept in one private module so the stateless metrics
(:mod:`prompt_anonymity.evaluation.metrics`) and the precompute-once
:class:`~prompt_anonymity.evaluation.LinkageRanking` rank candidates identically and
cannot drift apart.
"""

from __future__ import annotations

import numpy as np


def rank_known_by_distance(distances) -> np.ndarray:
    """Return, per unknown row, the known-column indices ordered nearest-first.

    ``distances`` is the ``(n_unknown, n_known)`` matrix from an attack (a DataFrame or
    array). A stable ascending argsort keeps the ranking deterministic when two known
    conversations are equidistant from an unknown one (ties break by column order).
    """
    return np.argsort(np.asarray(distances), axis=1, kind="stable")


def first_k_distinct(labels_in_rank_order, k) -> set:
    """Return the first ``k`` distinct labels along a nearest-first sequence.

    ``labels_in_rank_order`` is the sequence of known-conversation identities ordered
    from nearest to farthest for a single unknown conversation. Collapsing it to its
    first ``k`` distinct identities is the candidate short-list the adversary would
    produce; the walk stops as soon as ``k`` distinct identities are seen. Returns fewer
    than ``k`` only when the sequence contains fewer than ``k`` distinct identities.
    """
    top_identities: set = set()
    for label in labels_in_rank_order:
        top_identities.add(label)
        if len(top_identities) == k:
            break
    return top_identities
