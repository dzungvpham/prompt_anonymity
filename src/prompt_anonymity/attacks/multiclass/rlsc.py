"""Weighted one-vs-all regularised least squares: the eager classifier that scales."""

from __future__ import annotations

import numpy as np

from ..common import group_by_author


class RegularizedLeastSquares:
    """Weighted one-vs-all regularised least squares (RLSC): one factorisation for every author.

    The eager classifier that survives a large author pool, and the method Narayanan et al. used
    to attack 100,000 blogs at IEEE S&P 2012. Every other discriminative attack here pays per
    author -- :class:`LogisticAttribution` fits ``T`` coupled weight vectors,
    :class:`SupportVectorAttribution` fits ``T(T-1)/2`` pairwise problems, and
    :class:`GradientBoostedTrees` grows one tree per author *per boosting round*. Squared loss has
    a closed form instead, so the ``d x d`` system is factorised **once** and each author is one
    more right-hand side: ``O(n d^2 + d^3 + d^2 T)`` rather than ``O(n d^2 + T d^3)``.

    The masking correction is **per author, on the left-hand side**, and it has to be. With ``T``
    authors a one-vs-all problem has ``T-1`` times more negatives than positives, so plain least
    squares calls almost everything negative. The fix is to up-weight each author's own documents
    by ``(n - n_a) / n_a`` so the two sides of *that* author's problem carry equal total weight.

    That weight depends on which author is currently positive, so it perturbs the Gram matrix
    once per author and a single shared factorisation is *not* enough. Woodbury is what rescues
    the complexity: the perturbation is ``(a_a - 1) X_a^T X_a``, a rank-``n_a`` update, so each
    author costs one ``n_a x n_a`` solve against the shared inverse rather than a fresh ``d x d``
    factorisation. Authors with more documents than dimensions take the direct solve instead,
    which is cheaper for them.

    .. warning::
       Do **not** simplify this to one shared Gram matrix with a rescaled target, which is the
       obvious-looking way to keep the ``O(n d^2 + d^3)`` cost. All ``T`` right-hand sides then
       differ only by a common scale and a common additive vector, so ``argmax`` over authors is
       *identical* for every choice of target -- the whole method degenerates into a whitened
       nearest-centroid rule (see :class:`WhitenedCentroid`) and the masking correction becomes a
       no-op.

    Beware the shape of the problem: masking bites once the number of classes is on the same
    order as the dimensionality of the data, so feature normalisation can matter more than the
    classifier -- try ``--standardize`` before concluding anything about this attack.
    """

    name = "rlsc"

    def __init__(self, alpha: float = 1.0, dtype=np.float32):
        self.alpha = alpha
        self.dtype = dtype

    def fit(self, embeddings, labels):
        embeddings = np.asarray(embeddings, dtype=float)
        self.authors, codes = np.unique(labels, return_inverse=True)
        n_documents, n_features = embeddings.shape
        n_authors = len(self.authors)

        self.mean = embeddings.mean(axis=0)
        centred = embeddings - self.mean
        total = centred.sum(axis=0)

        # The unweighted system, shared by every author and inverted once.
        base = centred.T @ centred + self.alpha * np.eye(n_features)
        base_inverse = np.linalg.inv(base)

        # Group each author's documents contiguously so the rank-n_a update is a slice.
        groups = group_by_author(codes, n_authors)
        sorted_rows = centred[groups.order]
        starts, stops = groups.starts, groups.stops

        self.coef = np.empty((n_features, n_authors))
        for author in range(n_authors):
            rows = sorted_rows[starts[author]:stops[author]]
            n_own = len(rows)
            # Positives up-weighted to carry the same total weight as the negatives.
            weight = (n_documents - n_own) / n_own
            # Targets are +1 on the author's own documents and -1 on everyone else's, so the
            # right-hand side needs only this author's sum and the corpus sum.
            right_hand_side = (weight + 1.0) * rows.sum(axis=0) - total

            if abs(weight - 1.0) < 1e-12:            # nothing to update; the base system is it
                self.coef[:, author] = base_inverse @ right_hand_side
            elif n_own <= n_features:                 # Woodbury: one n_own x n_own solve
                projected = rows @ base_inverse
                middle = np.eye(n_own) / (weight - 1.0) + projected @ rows.T
                self.coef[:, author] = (base_inverse @ right_hand_side
                                        - projected.T @ np.linalg.solve(
                                            middle, projected @ right_hand_side))
            else:                                     # more documents than dimensions: direct
                self.coef[:, author] = np.linalg.solve(
                    base + (weight - 1.0) * rows.T @ rows, right_hand_side)

        # Solved in float64 for the conditioning, stored narrow: the coefficients are tiny
        # (n_features x n_authors) but the score matrix they produce is not, and a float64 one
        # is 6 GB at 86,000 documents against 14,000 authors.
        self.coef = self.coef.astype(self.dtype)
        self.mean = self.mean.astype(self.dtype)
        return self

    def score(self, embeddings):
        return (np.asarray(embeddings, dtype=self.dtype) - self.mean) @ self.coef

