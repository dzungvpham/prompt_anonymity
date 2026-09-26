"""Multinomial logistic regression over the known authors."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression


class LogisticAttribution:
    """Multinomial logistic regression over the known authors; the score is the class logit.

    Unlike the generative methods, which model *where each author sits*, this learns *what
    separates them*, spending its capacity on the discriminating directions.

    ``class_weight="balanced"`` matters: known authors vary widely in document count, and without
    it the most prolific authors dominate the objective.
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

