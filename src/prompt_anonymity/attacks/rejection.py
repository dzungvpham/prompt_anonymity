"""Turning an author-score matrix into an accept/reject decision.

The attacks in :mod:`prompt_anonymity.attacks.attribution` answer "which known author?". In any
real corpus most anonymous documents were written by somebody the attacker has never seen, so a
usable attack also has to answer "**is** it one of them?" -- and the score matrix already
contains the evidence, if it is read the right way.

The catch is that raw scores are not comparable across documents. A short, generic document
scores low against *every* candidate and a distinctive one scores high against all of them, so
thresholding a raw maximum mostly measures document length. Normalising within a document
(:func:`cohort_normalize`) fixes that, and is worth about 0.09 of DIR@10% on swe-chat.

Score the resulting decision with :mod:`prompt_anonymity.metrics.detection`.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression


def cohort_normalize(scores: np.ndarray) -> np.ndarray:
    """Z-score each document's scores across the candidate authors (T-norm).

    Borrowed from speaker verification, where the same problem appears as utterance-dependent
    score offsets. Normalising within a document turns the score into "how far above its own
    cohort this author stands", which *is* comparable between documents.
    """
    mean = scores.mean(axis=1, keepdims=True)
    deviation = scores.std(axis=1, keepdims=True)
    return (scores - mean) / np.where(deviation > 0, deviation, 1.0)


def rejection_score(scores: np.ndarray, normalize: bool = True) -> np.ndarray:
    """Out-of-set score per document (higher = more likely *not* one of the known authors).

    The negated best author score, cohort-normalised by default. Pass ``normalize=False`` to
    threshold the raw score instead, which is usually worse for the reason above.
    """
    if normalize:
        scores = cohort_normalize(scores)
    return -scores.max(axis=1)


def rejection_features(scores: np.ndarray) -> np.ndarray:
    """Scale-free summary of one document's score vector, for :class:`LearnedRejector`.

    Every feature asks "how *peaked* is this document's evidence?", never "how large is it", so a
    rejector trained on a simulation with 83 candidate authors still means the same thing at test
    time with 124.
    """
    ordered = np.sort(scores, axis=1)[:, ::-1]
    mean = scores.mean(axis=1, keepdims=True)
    deviation = scores.std(axis=1, keepdims=True)
    z = (ordered - mean) / np.where(deviation > 0, deviation, 1.0)

    shifted = scores - scores.max(axis=1, keepdims=True)
    probability = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)
    ordered_probability = np.sort(probability, axis=1)[:, ::-1]
    entropy = -(probability * np.log(probability + 1e-12)).sum(axis=1) / np.log(scores.shape[1])
    k = min(10, scores.shape[1] - 1)

    return np.column_stack([
        z[:, 0],                              # how far the best author stands above the cohort
        z[:, 0] - z[:, 1],                    # margin over the runner-up, in cohort units
        z[:, 0] - z[:, k],                    # margin over the k-th, i.e. peak sharpness
        z[:, :5].mean(axis=1),
        ordered_probability[:, 0],            # max softmax probability
        ordered_probability[:, 0] - ordered_probability[:, 1],
        entropy,                              # normalised softmax entropy
    ])


class LearnedRejector:
    """Learn the accept/reject decision instead of thresholding a single statistic.

    The attacker has no out-of-set examples, but it can manufacture them: hold out a third of
    the known authors, fit the identification model *without* them, and score their documents.
    Those score vectors are exactly what a genuinely unseen author produces. A binary model over
    the shape features from :func:`rejection_features` then learns to tell them from the shapes
    in-set documents produce -- combining margin, entropy and peakedness rather than betting
    everything on the top score.

    Fitted entirely on known documents; ``score`` returns P(out-of-set).

    .. note::
       Kept as a **documented negative result**, not a recommendation. On swe-chat it overfits
       the simulation and loses to the plain cohort-normalised score (DIR@10% 0.055 vs 0.106):
       the shape of an out-of-set score vector differs between a simulation with 83 candidate
       authors and a real window with 124, and the extra parameters latch onto that difference.
    """

    def __init__(self, C: float = 1.0):
        self.C = C

    def fit(self, build_attack, embeddings, labels, folds):
        features, targets = [], []
        for train_rows, query_rows in folds:
            fitted = build_attack().fit(embeddings[train_rows], labels[train_rows])
            scores = fitted.score(embeddings[query_rows])
            features.append(rejection_features(scores))
            targets.append(~np.isin(labels[query_rows], fitted.authors))
        self.model = LogisticRegression(C=self.C, max_iter=3000, class_weight="balanced").fit(
            np.vstack(features), np.concatenate(targets)
        )
        return self

    def score(self, scores):
        return self.model.predict_proba(rejection_features(scores))[:, 1]
