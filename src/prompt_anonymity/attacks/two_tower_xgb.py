"""Two-tower XGBoost linkage attack.

Trains a binary classifier on pairs of (known, known) conversation feature vectors,
labeled 1 if same author else 0. At inference, scores each unknown conversation
against every known conversation using 1 - P(same author) as a "distance", so it
plugs into the same top-k evaluation as nearest_neighbor_attack.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from sklearn.experimental import enable_halving_search_cv  # noqa: F401
from sklearn.model_selection import HalvingRandomSearchCV, GroupKFold
from scipy.stats import randint, uniform
from ..core import AttackData


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
        concat = np.concatenate([vecs_a, vecs_b])

        return np.concatenate(
            [diff, product, concat]
        )

    # Batch during inference
    diff = np.abs(vecs_a - vecs_b)
    product = vecs_a * vecs_b
    concat = np.concatenate([vecs_a, vecs_b], axis=1)

    return np.concatenate(
        [diff, product, concat],
        axis=1,
    )


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


def _tune_xgb(X_train, y_train, groups, seed=47):

    param_dist = {
        "max_depth": randint(2, 8),
        "learning_rate": uniform(0.01, 0.29),
        "subsample": uniform(0.5, 0.5),
        "colsample_bytree": uniform(0.5, 0.5),
        "min_child_weight": randint(1, 10),
    }

    cv = GroupKFold(n_splits=5)

    search = HalvingRandomSearchCV(
    estimator=XGBClassifier(
        eval_metric="logloss",
        random_state=seed,
        n_jobs=2,
    ),
    param_distributions=param_dist,
    resource="n_estimators",
    max_resources=400,
    min_resources=25,
    scoring="neg_log_loss",
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

def run_two_tower_xgb(data: AttackData, seed: int = 47, aggregate_by_identity: bool = True) -> pd.DataFrame:
    """Train on known-known pairs, score unknown-vs-known pairs, return a distance matrix."""
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
    distance_matrix = np.zeros((n_unknown, n_known))

    for u in range(n_unknown):
        pair_feats = _build_pair_features(
            np.tile(data.unknown_embeddings[u], (n_known, 1)), data.known_embeddings
        )
        same_author_prob = clf.predict_proba(pair_feats)[:, 1]
        distance_matrix[u, :] = 1.0 - same_author_prob  # smaller distance = more likely same author

    # added to aggregate by identity, so that every conversation from the same known identity gets an identical score
    # only this if block and aggregate_by_identity: bool = True in the class parameter
    if aggregate_by_identity:
        # Replace each known conversation's distance with its identity's mean distance,
        # so every conversation from the same known identity gets an identical score.
        known_labels = data.known_labels
        for label in np.unique(known_labels):
            idx = np.where(known_labels == label)[0]
            mean_dist = distance_matrix[:, idx].mean(axis=1, keepdims=True)
            distance_matrix[:, idx] = mean_dist

    return pd.DataFrame(distance_matrix)