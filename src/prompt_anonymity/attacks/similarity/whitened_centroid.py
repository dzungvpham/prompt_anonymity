"""WCCN: cosine to the author centroid after whitening by the within-author covariance."""

from __future__ import annotations

import numpy as np

from ..common import class_means, inverse_sqrt, unit_rows, within_class_covariance


class WhitenedCentroid:
    """Cosine to the centroid after whitening by the within-author covariance (WCCN).

    Speaker verification's standard first move: whitening equalises within-author noise across
    directions, so what remains of the distance is between-author structure. Length-normalising
    after whitening keeps the transformed vectors well-behaved enough for a cosine to mean
    something.
    """

    name = "wccn"

    def __init__(self, shrinkage: float = 0.2):
        self.shrinkage = shrinkage

    def fit(self, embeddings, labels):
        self.authors, codes = np.unique(labels, return_inverse=True)
        self.mean = embeddings.mean(axis=0)
        covariance = within_class_covariance(embeddings - self.mean, codes, len(self.authors),
                                             self.shrinkage)
        self.transform = inverse_sqrt(covariance)
        whitened = unit_rows((embeddings - self.mean) @ self.transform)
        self.centroids = unit_rows(class_means(whitened, codes, len(self.authors)))
        return self

    def project(self, embeddings):
        """Map documents into the whitened space (useful for clustering experiments)."""
        return unit_rows((embeddings - self.mean) @ self.transform)

    def score(self, embeddings):
        return self.project(embeddings) @ self.centroids.T

