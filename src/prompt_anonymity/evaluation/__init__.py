"""Everything that turns an attack's or a defense's output into a number.

The attacks and defenses produce artifacts; everything here measures them. Three layers:

``evaluation.metrics``
    Stateless scoring of one score matrix -- top-k accuracy and the chance baseline
    (:mod:`~prompt_anonymity.evaluation.metrics.accuracy`), rank distributions
    (:mod:`~prompt_anonymity.evaluation.metrics.ranking`), open-set accept/reject scoring
    (:mod:`~prompt_anonymity.evaluation.metrics.detection`), and the flipped query direction
    (:mod:`~prompt_anonymity.evaluation.metrics.retrieval`).

:class:`LinkageRanking` / :func:`headline_accuracy`
    Rank once, reuse the ranking, instead of each metric re-deriving it from the score matrix.

``evaluation.utility``
    The other axis: not whether an attack can re-identify an author, but what a defense cost --
    whether the rewritten text still says what the original said.

**``evaluation.utility`` is deliberately not imported here.** Its metrics reach for heavy optional
dependencies (an API client, ``torch``/``transformers``) that ``run_experiment.py`` should not have
to load on every run. Import it explicitly (``from prompt_anonymity.evaluation import utility``).
"""

from __future__ import annotations

from .metrics import (
    author_query_metrics,
    c_at_1,
    calibration_metrics,
    cmc_curve,
    detection_auroc,
    detection_identification_rate,
    equal_error_rate,
    macro_f1_score,
    macro_top_k_accuracy,
    max_softmax_confidence,
    per_author_ranking,
    random_guessing_accuracy,
    ranking_summary,
    retrieval_summary,
    selective_classification,
    top_k_accuracy,
    true_author_ranks,
)
from .ranking import LinkageRanking
from .sweep import headline_accuracy

__all__ = [
    "LinkageRanking",
    "headline_accuracy",
    "author_query_metrics",
    "c_at_1",
    "calibration_metrics",
    "cmc_curve",
    "detection_auroc",
    "detection_identification_rate",
    "equal_error_rate",
    "macro_f1_score",
    "macro_top_k_accuracy",
    "max_softmax_confidence",
    "per_author_ranking",
    "random_guessing_accuracy",
    "ranking_summary",
    "retrieval_summary",
    "selective_classification",
    "top_k_accuracy",
    "true_author_ranks",
]
