"""Everything that turns an attack's or a defense's output into a number.

This is the scoring half of the package: the attacks and defenses produce artifacts, and
everything here measures them. Three layers, from stateless functions up to whole evaluation axes:

``evaluation.metrics``
    Stateless scoring of one score matrix -- top-k accuracy and the chance baseline
    (:mod:`~prompt_anonymity.evaluation.metrics.accuracy`), rank distributions
    (:mod:`~prompt_anonymity.evaluation.metrics.ranking`), open-set accept/reject scoring
    (:mod:`~prompt_anonymity.evaluation.metrics.detection`), and the flipped query direction
    (:mod:`~prompt_anonymity.evaluation.metrics.retrieval`).

:class:`LinkageRanking` / :func:`headline_accuracy`
    Rank once, reuse the ranking. The metrics above each re-derive what they need from a score
    matrix; this pair does the expensive part once and serves several tables from it.

``evaluation.utility``
    The **other axis**. Everything above asks whether an attack can re-identify an author --
    how well a defense hid someone. That subpackage asks what the defense cost: whether the
    rewritten text still says what the original said. See its own docstring for the metrics.

**``evaluation.utility`` is deliberately not imported here.** Its metrics reach for heavy
optional dependencies -- an API client, and for the local scorers ``torch`` and
``transformers`` -- and ``experiments/run_experiment.py`` imports this package on every run
without ever touching them. Import it explicitly (``from prompt_anonymity.evaluation import
utility``) and it will load what that particular metric needs, when it needs it.

This package absorbed two former top-level packages on 2026-08-11: ``prompt_anonymity.metrics``
(now ``evaluation.metrics``) and ``prompt_anonymity.utility`` (now ``evaluation.utility``).
Anything on disk or in a note referring to those paths predates the move.
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
