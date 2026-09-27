"""Metrics for scoring linkage and authorship-attribution attacks.

Five families, each answering a different question about the same attack. The first four consume a
**score matrix** (documents x candidate authors, higher = more likely) rather than a distance
matrix; negate a distance matrix to get one. The fifth consumes a partition instead, since it scores
a different threat model.

:mod:`~prompt_anonymity.evaluation.metrics.accuracy` -- *how often is the attack right?*
    ``top_k_accuracy`` at the conversation and identity levels, plus the chance baseline.

:mod:`~prompt_anonymity.evaluation.metrics.ranking` -- *how close was it when it was wrong, and to whom?*
    Rank-based summaries over the whole ranking (``mrr``, mean percentile rank, ``cmc_curve``), plus
    author-averaged views that keep a few prolific users from deciding the headline.

:mod:`~prompt_anonymity.evaluation.metrics.retrieval` -- *can the attacker find everything one user wrote?*
    The same scores read column-wise, with each known author as a query: average precision and
    R-precision per author. A different threat model from identification.

:mod:`~prompt_anonymity.evaluation.metrics.detection` -- *is this document's author known at all?*
    Open-set / verification metrics for the reject option: AUROC, equal error rate, DIR@FAR,
    PAN's c@1, and calibration of the reported confidence.

:mod:`~prompt_anonymity.evaluation.metrics.clustering` -- *which of these share an author?*
    Scores a **partition** of anonymous documents rather than a ranking of named candidates, since
    it measures *linkability* rather than identifiability: BCubed precision/recall/F, the
    link-ranking view, degenerate baselines, and exposure measures.
"""

from .accuracy import random_guessing_accuracy, top_k_accuracy
from .clustering import (
    BCubedScores,
    ClusterContingency,
    NOISE_LABEL,
    bcubed_from_contingency,
    bcubed_scores,
    clustering_summary,
    expand_noise,
    link_ranking_metrics,
    pairwise_scores,
    per_author_clustering,
    single_cluster_baseline,
    singleton_baseline,
)
from .detection import (
    c_at_1,
    calibration_metrics,
    selective_classification,
    detection_auroc,
    detection_identification_rate,
    equal_error_rate,
    max_softmax_confidence,
)
from .ranking import (
    cmc_curve,
    macro_f1_score,
    macro_top_k_accuracy,
    per_author_ranking,
    ranking_summary,
    true_author_ranks,
)
from .retrieval import author_query_metrics, retrieval_summary

__all__ = [
    # accuracy
    "random_guessing_accuracy",
    "top_k_accuracy",
    # ranking
    "cmc_curve",
    "macro_f1_score",
    "macro_top_k_accuracy",
    "per_author_ranking",
    "ranking_summary",
    "true_author_ranks",
    # retrieval
    "author_query_metrics",
    "retrieval_summary",
    # detection
    "c_at_1",
    "calibration_metrics",
    "selective_classification",
    "detection_auroc",
    "detection_identification_rate",
    "equal_error_rate",
    "max_softmax_confidence",
    # clustering (linkability)
    "NOISE_LABEL",
    "BCubedScores",
    "ClusterContingency",
    "bcubed_scores",
    "bcubed_from_contingency",
    "clustering_summary",
    "expand_noise",
    "link_ranking_metrics",
    "pairwise_scores",
    "per_author_clustering",
    "singleton_baseline",
    "single_cluster_baseline",
]
