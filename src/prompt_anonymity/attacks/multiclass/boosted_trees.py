"""Gradient-boosted decision trees (XGBoost) over the known authors."""

from __future__ import annotations

import numpy as np


class GradientBoostedTrees:
    """Gradient-boosted decision trees (XGBoost) over the known authors.

    The only non-linear, non-metric attack here: every other one ultimately compares documents
    along straight lines in feature space. StyloMetrix features are heterogeneous -- ratios in
    [0, 1] next to raw counts, many near-zero for most documents -- and trees handle that mix
    natively, splitting on thresholds instead of weighting directions, and picking up interactions
    between features that a linear model cannot express.

    The score is the **log** class probability, not the raw probability. It is the same ranking
    either way, but log-space is what the downstream machinery expects: cohort normalisation
    z-scores across authors (meaningful for log-odds-like quantities, not for probabilities that
    sum to 1), and it makes ``softmax(score)`` recover the model's own posterior exactly, so the
    calibration metrics in :mod:`prompt_anonymity.metrics.detection` measure something real for
    this attack.

    ``xgboost`` is imported lazily so the rest of the package works without it installed.
    """

    name = "xgboost"

    def __init__(self, n_estimators: int = 300, max_depth: int = 3, learning_rate: float = 0.3,
                 subsample: float = 1.0, n_jobs: int = -1):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.n_jobs = n_jobs

    def fit(self, embeddings, labels):
        from xgboost import XGBClassifier  # lazy: keeps xgboost an optional runtime dependency

        self.authors, codes = np.unique(labels, return_inverse=True)
        self.model = XGBClassifier(
            n_estimators=self.n_estimators, max_depth=self.max_depth,
            learning_rate=self.learning_rate, subsample=self.subsample,
            tree_method="hist", objective="multi:softprob", n_jobs=self.n_jobs, verbosity=0,
        ).fit(embeddings, codes)
        return self

    def score(self, embeddings):
        return np.log(np.clip(self.model.predict_proba(embeddings), 1e-12, None))

