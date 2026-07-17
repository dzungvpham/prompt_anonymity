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
    y = np.array([1] * len(pos_pairs) + [0] * len(neg_pairs))
    X = np.array([_build_pair_features(embeddings[i], embeddings[j]) for i, j in pairs])
    return X, y


def run_two_tower_xgb(data: AttackData, seed: int = 47) -> pd.DataFrame:
    """Train on known-known pairs, score unknown-vs-known pairs, return a distance matrix."""
    X_train, y_train = _make_training_pairs(data.known_embeddings, data.known_labels, seed=seed)


    ''' 
    version 1: clf = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        eval_metric="logloss",
        random_state=seed,
    )
    
    this is version 2:
    clf is model object that will learn from training examples
    XGBClassifier means the model is an XGBoost classifier, which is a tree-based boosting model commonly used for classification tasks.
    '''
    clf = XGBClassifier(
        n_estimators=200, # the model will build 200 decision trees in sequence, with each one improving the previous ones.
        max_depth=3, # the maximum depth of each decision tree is limited to 3 (small), which helps prevent overfitting and keeps the model simpler.
        learning_rate=0.05, # controls how much the model adjusts its weights with each new tree. A smaller learning rate means slower learning but can lead to better generalization.
        subsample=0.8, # only 80% of the training data is randomly selected for each tree, which helps prevent overfitting and improves generalization.
        colsample_bytree=0.8, # only 80% of the features are randomly selected for each tree, which also helps prevent overfitting and improves generalization.
        min_child_weight=5, # the minimum sum of instance weights (hessian) needed in a child node to make a split.
        eval_metric="logloss", # the model will use log loss as the evaluation metric during training, which is suitable for binary classification tasks.
        random_state=seed, # makes the training reproducible so the same seed gives the same result
        )
    clf.fit(X_train, y_train)

    n_unknown, n_known = data.n_unknown, data.n_known
    distance_matrix = np.zeros((n_unknown, n_known))

    for u in range(n_unknown):
        pair_feats = _build_pair_features(
            np.tile(data.unknown_embeddings[u], (n_known, 1)), data.known_embeddings
        )
        same_author_prob = clf.predict_proba(pair_feats)[:, 1]
        distance_matrix[u, :] = 1.0 - same_author_prob  # smaller distance = more likely same author

    return pd.DataFrame(distance_matrix)