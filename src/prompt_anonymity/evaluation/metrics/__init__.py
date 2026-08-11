"""Metrics for scoring linkage and authorship-attribution attacks.

Four families, each answering a different question about the same attack. All but the first
consume a **score matrix** (documents x candidate authors, higher = more likely) rather than a
distance matrix, which is the orientation an attribution model produces; negate a distance
matrix to move between them.

:mod:`~prompt_anonymity.evaluation.metrics.accuracy` -- *how often is the attack right?*
    ``top_k_accuracy`` at the conversation and identity levels, and ``random_guessing_accuracy``
    for the matching chance baseline, so the difference is the adversary's advantage.

:mod:`~prompt_anonymity.evaluation.metrics.ranking` -- *how close was it when it was wrong, and to whom?*
    Rank-based summaries that use the whole ranking instead of a few cutoffs (``mrr``, mean
    percentile rank, the full ``cmc_curve``), plus the author-averaged views that keep a handful
    of prolific users from deciding the headline (``macro_top_k_accuracy``, ``macro_f1_score``,
    ``per_author_ranking``).

:mod:`~prompt_anonymity.evaluation.metrics.retrieval` -- *can the attacker find everything one user wrote?*
    The same scores read column-wise, with each known author as a query: average precision and
    R-precision per author. A different threat model from identification, and the direction in
    which mean average precision is not degenerate.

:mod:`~prompt_anonymity.evaluation.metrics.detection` -- *is this document's author known at all?*
    Open-set / verification metrics for the reject option: AUROC, equal error rate, DIR@FAR,
    PAN's c@1, and calibration of the reported confidence.
"""

from .accuracy import random_guessing_accuracy, top_k_accuracy
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
]
