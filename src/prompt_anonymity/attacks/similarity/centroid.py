"""Cosine similarity to each author's mean direction: the reference point for everything else."""

from __future__ import annotations

import numpy as np

from ..common import class_means, unit_rows


class CentroidCosine:
    """Baseline: cosine similarity to each author's mean direction.

    The attack the rolling-window experiment shipped with. Kept as the reference point every
    supervised method below is measured against.
    """

    name = "cosine"

    def fit(self, embeddings, labels):
        self.authors, codes = np.unique(labels, return_inverse=True)
        self.centroids = unit_rows(class_means(unit_rows(embeddings), codes, len(self.authors)))
        return self

    def score(self, embeddings):
        return unit_rows(embeddings) @ self.centroids.T

