"""Support vector machine over the known authors, one-vs-one folded to one column each."""

from __future__ import annotations

import numpy as np
from sklearn.svm import SVC


class SupportVectorAttribution:
    """Support vector machine over the known authors; the score is the decision function.

    The other classical discriminative answer alongside :class:`LogisticAttribution`: logistic
    regression fits the whole conditional distribution, while an SVM only cares about documents
    near each boundary. The RBF kernel additionally buys non-linearity, which no other attack here
    has.

    Uses ``sklearn.svm.SVC`` rather than ``LinearSVC`` even for ``kernel="linear"``: ``SVC`` is
    one-vs-one (many tiny pairwise problems), whereas ``LinearSVC`` is one-vs-rest (each author
    trained against the whole corpus), and one-vs-one is cheaper at this author-pool size.

    ``decision_function_shape="ovr"`` folds the pairwise votes back into one column per author, so
    the output has the same shape as every other attack's. Those margins are *not* posteriors --
    :func:`prompt_anonymity.evaluation.metrics.max_softmax_confidence` will report a near-uniform
    confidence for them, so read this attack's calibration numbers as meaningless rather than as
    bad.
    """

    name = "svm"

    def __init__(self, C: float = 1.0, kernel: str = "rbf", gamma: str | float = "scale"):
        self.C = C
        self.kernel = kernel
        self.gamma = gamma

    def fit(self, embeddings, labels):
        self.authors, codes = np.unique(labels, return_inverse=True)
        self.model = SVC(C=self.C, kernel=self.kernel, gamma=self.gamma,
                         decision_function_shape="ovr").fit(embeddings, codes)
        return self

    def score(self, embeddings):
        return self.model.decision_function(embeddings)

