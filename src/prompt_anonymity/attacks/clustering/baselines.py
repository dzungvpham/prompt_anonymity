"""Reference partitions: what a clustering result has to beat to mean anything.

BCubed is not an accuracy, and it has no obvious zero. Two degenerate partitions score
surprisingly well -- all-singletons takes precision 1.0 for free, one-big-cluster takes recall
1.0 -- and a third, a random partition, inherits whatever credit the *shape* of a clustering earns
independently of who is in which cluster. Reporting a BCubed F without these beside it says
almost nothing.

Each function here returns **one label per document**, so a baseline is scored by exactly the code
path a real attack is scored by (:mod:`prompt_anonymity.evaluation.metrics.clustering`). That is
deliberate: a closed-form baseline computed by a different route is a second implementation of the
measure, and the two drift.

Following PAN 2016's author-clustering task, which defines BASELINE-Singleton and
BASELINE-Random. Two more are added here for this project:

``random_labels``
    PAN draws a random number of clusters and assigns documents to them at random. Permuting an
    existing labelling instead makes the null **matched**: it holds the cluster-size distribution
    exactly fixed and destroys only the association between documents and clusters, which is the
    thing under test. Against PAN's version, a method could score well merely by guessing the
    right number of clusters.

``metadata_labels``
    Partition by observable metadata -- language, model provider -- with no text at all. This one
    is specific to this project and it is load-bearing: a stylometric featurizer can separate
    languages almost perfectly, so an attack on a multilingual corpus can score well on language
    alone. Anything a metadata partition already achieves is not evidence about writing style.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def singleton_labels(n_documents: int) -> np.ndarray:
    """Every document in a cluster of its own -- PAN's BASELINE-Singleton.

    BCubed precision 1.0 by construction, recall ``mean_i 1/|A_i|``. The attacker who declines to
    link anything, and the reason precision is never quoted here without ``singleton_rate``.
    """
    return np.arange(n_documents, dtype=np.int64)


def single_cluster_labels(n_documents: int) -> np.ndarray:
    """Every document in one cluster: recall 1.0, precision ``mean_i |A_i|/N``."""
    return np.zeros(n_documents, dtype=np.int64)


def random_labels(reference_labels: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """A random partition with exactly ``reference_labels``' cluster-size distribution.

    Implemented as a permutation of the reference labelling, which is the same thing: shuffling
    which document carries which label leaves every cluster's size untouched and makes the
    assignment independent of authorship. Pass the *true* author labels to ask what a partition of
    the right shape scores by luck, or a *predicted* labelling to ask what that particular
    clustering's shape was worth before any of its content.

    Average over repetitions -- PAN uses 50 -- and seed the generator, so a redrawn baseline
    cannot move a published number.
    """
    return np.asarray(reference_labels).ravel()[rng.permutation(len(reference_labels))]


def metadata_labels(frame: pd.DataFrame, columns: str | list[str]) -> np.ndarray:
    """Partition by the given metadata columns, reading no text at all.

    Groups documents by the tuple of values in ``columns`` (``language_primary``, ``model_owner``,
    or both). Missing values form their own group rather than being dropped: an absent language
    label is itself observable to an attacker.
    """
    columns = [columns] if isinstance(columns, str) else list(columns)
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(f"metadata column(s) {missing} not in the document frame; "
                       f"available: {sorted(frame.columns)}")
    keys = frame[columns].astype("string").fillna("<missing>").agg("|".join, axis=1)
    return pd.factorize(keys)[0].astype(np.int64)
