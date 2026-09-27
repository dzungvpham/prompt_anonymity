"""LDA dimensionality reduction, then WCCN whitening inside that reduced subspace.

:class:`~.lda.LDACentroid` and :class:`~.whitened_centroid.WhitenedCentroid` are tested elsewhere
as *competing* attacks. Here they're chained instead: the speaker-verification literature treats
"LDA followed by WCCN" as the standard strong combination for embeddings that aren't already
end-to-end discriminative. LDA maximises between-author over within-author scatter and discards
directions with no author signal; WCCN's whitening then equalises what noise remains *inside that
already-denoised space*, rather than fighting the full embedding's noise directly.
"""

from __future__ import annotations

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from ..common import unit_rows
from .whitened_centroid import WhitenedCentroid


class LDAWCCN:
    """Project onto the LDA discriminant subspace, then fit WCCN inside it.

    ``n_components`` is a ceiling on the LDA subspace, same contract as
    :class:`~.lda.LDACentroid`: clamped to ``n_authors - 1`` rather than raised, so a sweep over
    differently sized author pools doesn't fail on the small ones. ``shrinkage`` is WCCN's own
    knob, applied to the covariance estimate inside the LDA subspace rather than the original
    embedding space.
    """

    name = "lda_wccn"

    def __init__(self, n_components: int | None = None, shrinkage: float = 0.2):
        self.n_components = n_components
        self.shrinkage = shrinkage

    def fit(self, embeddings, labels):
        self.authors, codes = np.unique(labels, return_inverse=True)
        available = min(len(self.authors) - 1, embeddings.shape[1])
        n_components = min(self.n_components, available) if self.n_components else available
        self.lda_model = LinearDiscriminantAnalysis(
            solver="eigen", shrinkage="auto", n_components=n_components
        ).fit(embeddings, codes)
        projected = self._project_lda(embeddings)
        self.wccn = WhitenedCentroid(shrinkage=self.shrinkage).fit(projected, labels)
        return self

    def _project_lda(self, embeddings):
        return unit_rows(self.lda_model.transform(embeddings))

    def project(self, embeddings):
        """Map documents through LDA then into WCCN's whitened space (for inspection/reuse)."""
        return self.wccn.project(self._project_lda(embeddings))

    def score(self, embeddings):
        return self.wccn.score(self._project_lda(embeddings))
