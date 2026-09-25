"""LDA dimensionality reduction, then WCCN whitening inside that reduced subspace.

:class:`~.lda.LDACentroid` and :class:`~.whitened_centroid.WhitenedCentroid` are both tested here
as *competing* attacks -- alternatives, never chained. The speaker-verification literature this
project's WCCN was drawn from treats them differently: for embeddings that aren't already
end-to-end discriminative, "LDA followed by WCCN" is the standard strong combination, not LDA
*or* WCCN alone (see e.g. arXiv:2204.03965, "Scoring of Large-Margin Embeddings for Speaker
Verification: Cosine or PLDA?", and the WCCN+cosine / LDA+PLDA comparisons in fused-system
speaker-verification papers). LDA maximises between-author over within-author scatter and
discards directions with no author signal at all; WCCN's within-class whitening then equalises
what noise remains *inside that already-denoised space*, rather than fighting the full,
un-reduced embedding's noise directly.
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
    differently sized author pools doesn't fail on the small ones for a reason unrelated to the
    setting itself. ``shrinkage`` is WCCN's own knob, applied to the covariance estimate inside
    the LDA subspace rather than the original embedding space -- likely a different optimum than
    plain WCCN's, since the subspace is lower-dimensional and already between-author-denoised.
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
