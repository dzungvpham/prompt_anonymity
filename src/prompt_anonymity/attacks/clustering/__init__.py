"""Clustering attacks: group anonymous conversations by author, without naming anyone.

Every other package under :mod:`prompt_anonymity.attacks` answers *"which enrolled author wrote
this document?"* and returns an ``[n_documents x n_authors]`` score matrix. This one answers a
different question -- *"which of these documents were written by the same person?"* -- and returns
**one cluster label per document**. No author is named and no known side is required, which makes
it a distinct threat model: an attacker with no labelled documents can still re-assemble a person's
sessions out of an anonymised dump, and that is what makes identification cheap afterwards -- one
cluster de-anonymised by any means hands over every document in it
(:mod:`prompt_anonymity.evaluation.metrics.clustering` reports that as ``amplification``).

The task, the measure and the baselines follow PAN 2016's author-clustering shared task
(Stamatatos et al., CLEF 2016), which is also where BCubed comes from -- see the metrics module.

Contract
--------
``cluster(embeddings) -> labels``, one integer per document, with
:data:`~prompt_anonymity.evaluation.metrics.clustering.NOISE_LABEL` (``-1``) for documents the
attack declined to place. The number of clusters is **not** an input: the attacker does not know
how many people wrote the log, and a method that has to be told is answering an easier question.
Where a labelled known side exists it is used to choose hyper-parameters, never to enrol authors.

Modules
-------
=========================  ====================================================================
:mod:`.graph`              the exact k-nearest-neighbour graph every algorithm is built on
:mod:`.algorithms`         HDBSCAN, Leiden, average linkage, connected components + the registry
:mod:`.baselines`          the partitions a real result has to beat (singleton, random, metadata)
=========================  ====================================================================

``experiments/run_clustering.py`` drives the algorithms, and its ``--diagnostics`` flag measures
the substrate first -- whether the neighbour graph carries authorship at all, before any algorithm
is asked to exploit it.
"""

from __future__ import annotations

from .algorithms import (
    CLUSTERING_ATTACKS,
    CLUSTERING_SPACES,
    CLUSTERING_SPACES_QUANTILE,
    MAX_DENSE_DOCUMENTS,
    edge_quantile,
    NOISE_LABEL,
    AverageLinkageClustering,
    BaselineClustering,
    ComponentwiseAgglomerative,
    ClusteringAttack,
    HDBSCANClustering,
    LeidenClustering,
    ThresholdComponents,
    get_clustering_attack,
    parameter_grid,
)
from .baselines import (
    metadata_labels,
    random_labels,
    single_cluster_labels,
    singleton_labels,
)
from .graph import NeighborGraph, build_neighbor_graph
from .projection import (
    LinearProjection,
    PROJECTION_FITTERS,
    fit_projection,
    identity_projection,
)
from .rescoring import GRAPH_RESCORINGS, rescore_graph

__all__ = [
    "CLUSTERING_SPACES_QUANTILE",
    "ComponentwiseAgglomerative",
    "edge_quantile",
    "GRAPH_RESCORINGS",
    "LinearProjection",
    "PROJECTION_FITTERS",
    "fit_projection",
    "identity_projection",
    "rescore_graph",
    "NeighborGraph",
    "build_neighbor_graph",
    "CLUSTERING_ATTACKS",
    "CLUSTERING_SPACES",
    "MAX_DENSE_DOCUMENTS",
    "NOISE_LABEL",
    "ClusteringAttack",
    "HDBSCANClustering",
    "LeidenClustering",
    "AverageLinkageClustering",
    "ThresholdComponents",
    "BaselineClustering",
    "get_clustering_attack",
    "parameter_grid",
    "singleton_labels",
    "single_cluster_labels",
    "random_labels",
    "metadata_labels",
]
