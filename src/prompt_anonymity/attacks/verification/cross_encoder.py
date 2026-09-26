"""Cross-encoder authorship verification: a learned same-author classifier over document pairs.

Trains a binary classifier on pairs of (known, known) conversation feature vectors, labeled 1 if
same author else 0, then scores each unknown document against every known one and aggregates to
an author score.

Filed under *verification* because that is the task: "were these two documents written by the
same person?" is authorship verification, the pairwise counterpart of attribution.

This is a **cross-encoder**, not a two-tower model: :func:`_build_pair_features` *interacts* the
two vectors (elementwise ``|a - b|`` and ``a * b``) before any scoring, so nothing can be
precomputed per document and every pair costs a full model evaluation -- which is why this attack
is the expensive one in the package.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.experimental import enable_halving_search_cv  # noqa: F401
from sklearn.model_selection import HalvingRandomSearchCV, GroupKFold
from scipy.stats import randint, uniform
from ...core import AttackData
from ..common import group_by_author


def _build_pair_features(vecs_a: np.ndarray, vecs_b: np.ndarray) -> np.ndarray:
    """
    Rich pair representation for two-tower linkage.

    Accepts either:
    - single vectors: shape (features,)
    - batches: shape (n_pairs, features)

    Returns matching shape:
    - single pair: (4 * features,)
    - batch: (n_pairs, 4 * features)
    """

    vecs_a = np.asarray(vecs_a)
    vecs_b = np.asarray(vecs_b)

    # Single pair during training
    if vecs_a.ndim == 1:
        diff = np.abs(vecs_a - vecs_b)
        product = vecs_a * vecs_b
        return np.concatenate([diff, product])

    # Batch during inference
    diff = np.abs(vecs_a - vecs_b)
    product = vecs_a * vecs_b
    return np.concatenate([diff, product], axis=1)


def _make_training_pairs(embeddings: np.ndarray, labels: np.ndarray, seed: int = 47):
    rng = np.random.default_rng(seed)
    n = len(labels)
    pos_pairs, neg_pairs = [], []

    label_to_idx = {}
    for i, lbl in enumerate(labels):
        label_to_idx.setdefault(lbl, []).append(i)

    # Positive pairs: same author, different conversation.
    for idx_list in label_to_idx.values():
        if len(idx_list) < 2:
            continue
        for i in range(len(idx_list)):
            for j in range(i + 1, len(idx_list)):
                pos_pairs.append((idx_list[i], idx_list[j]))

    # Negative pairs: sample same number, different authors, to keep classes balanced.
    n_neg = len(pos_pairs)
    attempts = 0
    while len(neg_pairs) < n_neg and attempts < n_neg * 20:
        i, j = rng.integers(0, n, size=2)
        if labels[i] != labels[j]:
            neg_pairs.append((i, j))
        attempts += 1

    pairs = pos_pairs + neg_pairs

    y = np.array(
        [1] * len(pos_pairs) +
        [0] * len(neg_pairs)
    )

    X = np.array([
        _build_pair_features(embeddings[i], embeddings[j])
        for i, j in pairs
    ],
        dtype=np.float32,
        )

    groups = np.array([
        labels[i]
        for i, j in pairs
    ])

    return X, y, groups


def _tune_xgb(X_train, y_train, groups, seed=47, max_search_samples=20000):
    if len(y_train) > max_search_samples:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(y_train), size=max_search_samples, replace=False)
        X_train, y_train, groups = X_train[idx], y_train[idx], groups[idx]


    param_dist = {
        "max_depth": randint(2, 8),
        "learning_rate": uniform(0.01, 0.29),
        "subsample": uniform(0.5, 0.5),
        "colsample_bytree": uniform(0.5, 0.5),
        "min_child_weight": randint(1, 10),
    }

    cv = GroupKFold(n_splits=5)

    # Lazy: attacks/__init__.py imports this package eagerly to build the registry, and a
    # top-level xgboost import would make it a hard requirement of every entry point.
    from xgboost import XGBClassifier

    search = HalvingRandomSearchCV(
    estimator=XGBClassifier(
        eval_metric="logloss",
        random_state=seed,
        n_jobs=2,
        tree_method="hist",
    ),
    param_distributions=param_dist,
    resource="n_estimators",
    max_resources=400,
    min_resources=25,
    scoring="average_precision",
    cv=cv,
    random_state=seed,
    n_jobs=2,
)

    search.fit(
        X_train,
        y_train,
        groups=groups,
    )

    print("Best params:", search.best_params_)
    print("Best CV score:", search.best_score_)

    return search.best_estimator_

def run_cross_encoder(data: AttackData, seed: int = 47, linkage: str = "mean") -> pd.DataFrame:
    """Train on known-known pairs, score unknown-vs-known pairs, aggregate to author scores.

    Returns an ``[n_unknown x n_authors]`` frame of same-author probabilities, **higher = more
    likely**, with the author labels as columns. ``linkage="mean"`` gives an author their average
    probability across their documents (the previous ``aggregate_by_identity`` behaviour);
    ``"max"`` gives them their best single document, matching
    :class:`~prompt_anonymity.attacks.similarity.NearestNeighbor`'s default.
    """
    X_train, y_train, groups = _make_training_pairs(
    data.known_embeddings,
    data.known_labels,
    seed=seed,
    )

    clf = _tune_xgb(
    X_train,
    y_train,
    groups,
    seed=seed,
    )

    n_unknown, n_known = data.n_unknown, data.n_known
    authors, codes = np.unique(data.known_labels, return_inverse=True)
    groups = group_by_author(codes, len(authors))
    known_sorted = np.asarray(data.known_embeddings)[groups.order]

    # One row per unknown document, one column per AUTHOR. The per-document probabilities are
    # reduced to their author immediately rather than kept, both because that is the unit every
    # metric scores and because an [n_unknown x n_known] matrix does not fit at corpus scale.
    author_scores = np.zeros((n_unknown, len(authors)))
    for u in range(n_unknown):
        pair_feats = _build_pair_features(
            np.tile(data.unknown_embeddings[u], (n_known, 1)), known_sorted
        )
        same_author_prob = clf.predict_proba(pair_feats)[:, 1]
        if linkage == "mean":
            author_scores[u] = (np.add.reduceat(same_author_prob, groups.starts)
                                / groups.counts)
        else:
            author_scores[u] = np.maximum.reduceat(same_author_prob, groups.starts)

    return pd.DataFrame(author_scores, columns=authors)