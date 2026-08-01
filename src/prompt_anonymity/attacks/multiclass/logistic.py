"""Multinomial logistic regression over the known authors."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression


class LogisticAttribution:
    """Multinomial logistic regression over the known authors; the score is the class logit.

    The best method measured on swe-chat, and the reason is worth stating: the generative
    methods above model *where each author sits*, while this learns *what separates them*. With
    196 noisy features and ~124 authors, the discriminative objective spends its capacity on the
    directions that actually discriminate, which nearly doubles top-1 over cosine.

    ``class_weight="balanced"`` matters here -- known authors range from 1 to 325 documents, and
    without it the handful of prolific authors dominate the objective.
    """

    name = "logistic"

    def __init__(self, C: float = 1.0, class_weight: str | None = "balanced", max_iter: int = 3000):
        self.C = C
        self.class_weight = class_weight
        self.max_iter = max_iter

    def fit(self, embeddings, labels):
        self.authors, codes = np.unique(labels, return_inverse=True)
        self.model = LogisticRegression(
            C=self.C, max_iter=self.max_iter, class_weight=self.class_weight
        ).fit(embeddings, codes)
        return self

    def score(self, embeddings):
        return self.model.decision_function(embeddings)

