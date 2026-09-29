"""Cosine to the author centroid inside the LDA discriminant subspace.

Filed with the similarity attacks rather than the multiclass ones on purpose: linear
discriminant analysis supplies the *space*, but the classifier is still a centroid comparison.
Nothing here fits a per-author decision function.
"""

from __future__ import annotations

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from ..common import class_means, unit_rows


class LDACentroid:
    """Project onto the LDA discriminant subspace, then cosine to the centroid.

    LDA maximises between-author over within-author scatter, which both denoises and discards
    the feature directions that carry no author information at all.

    ``n_components`` is a *ceiling*: LDA cannot produce more than ``n_authors - 1`` discriminants,
    so a request for more is silently clamped rather than raised. That matters when the same
    setting is reused across differently sized author pools -- a hyper-parameter search over
    subsampled folds would otherwise fail on the small ones for a reason that has nothing to do
    with the setting's quality.
    """

    name = "lda"

    def __init__(self, n_components: int | None = None):
        self.n_components = n_components

    def fit(self, embeddings, labels):
        self.authors, codes = np.unique(labels, return_inverse=True)
        available = min(len(self.authors) - 1, embeddings.shape[1])
        n_components = min(self.n_components, available) if self.n_components else available
        self.model = LinearDiscriminantAnalysis(
            solver="eigen", shrinkage="auto", n_components=n_components
        ).fit(embeddings, codes)
        self.centroids = unit_rows(class_means(unit_rows(self.model.transform(embeddings)),
                                               codes, len(self.authors)))
        return self

    def project(self, embeddings):
        return unit_rows(self.model.transform(embeddings))

    def score(self, embeddings):
        return self.project(embeddings) @ self.centroids.T

