"""Nearest-neighbor linkage (re-identification) attack.

The adversary represents every conversation as a vector -- a semantic embedding or a
stylometric feature vector -- and scores each *unknown* (anonymous) conversation
against each *known* (labeled) conversation by the distance between their vectors.
Ranking the known conversations from nearest to farthest is the adversary's guess at
who authored each unknown conversation; :mod:`prompt_anonymity.metrics` turns that
ranking into top-k accuracy.

This module only builds the distance matrix. Choosing which conversations are known
vs. unknown and constructing their identity labels is the caller's responsibility
(and will be handled by future data-loading helpers).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist


def nearest_neighbor_attack(
    known_embeddings,
    unknown_embeddings,
    *,
    metric: str = "euclidean",
) -> pd.DataFrame:
    """Score every unknown conversation against every known conversation by distance.

    Parameters
    ----------
    known_embeddings : array-like of shape (n_known, n_features)
        One vector per *known* (labeled) conversation. May be semantic embeddings or
        stylometric feature vectors; values must be numeric and finite (fill any
        StyloMetrix NaNs before calling, e.g. ``np.nan_to_num(...)``).
    unknown_embeddings : array-like of shape (n_unknown, n_features)
        One vector per *unknown* (anonymous) conversation to re-identify. Must have
        the same number of features as ``known_embeddings``.
    metric : str or callable, default ``"euclidean"``
        Distance metric forwarded to :func:`scipy.spatial.distance.cdist`. Use
        ``"euclidean"`` for StyloMetrix features and ``"cosine"`` for semantic
        embeddings (e.g. Gemini); these reproduce, respectively, the
        negative-Euclidean and L2-normalized dot-product rankings used in the
        WildChat and SWE-chat analyses. (Cosine distance is invariant to vector
        norm, so it matches the normalize-then-dot-product path without normalizing
        first.) Any cdist-compatible metric name or a custom callable works, but it
        must be a *distance* -- smaller meaning more similar -- because the metrics
        rank candidates in ascending order.

    Returns
    -------
    pandas.DataFrame of shape (n_unknown, n_known)
        Entry ``[u, k]`` is the distance between unknown conversation ``u`` (row) and
        known conversation ``k`` (column); smaller means more similar. Rows align to
        ``unknown_embeddings`` and columns to ``known_embeddings`` by position. The
        index and columns are a plain ``RangeIndex``: identity labels are kept out of
        the matrix and supplied separately to the metrics in the same row/column
        order, so the matrix stays a purely numeric distance table.
    """
    known = np.asarray(known_embeddings, dtype=float)
    unknown = np.asarray(unknown_embeddings, dtype=float)
    if known.ndim != 2 or unknown.ndim != 2:
        raise ValueError(
            "known_embeddings and unknown_embeddings must be 2-D arrays "
            f"(got shapes {known.shape} and {unknown.shape})."
        )
    if known.shape[1] != unknown.shape[1]:
        raise ValueError(
            "known_embeddings and unknown_embeddings must have the same number of "
            f"features (got {known.shape[1]} and {unknown.shape[1]})."
        )

    # distance_matrix[u, k] = distance(unknown u, known k); smaller = more similar.
    distance_matrix = cdist(unknown, known, metric=metric)
    return pd.DataFrame(distance_matrix)
