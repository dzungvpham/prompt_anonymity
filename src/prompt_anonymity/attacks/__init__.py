"""Linkage attacks, organised by what an attack actually does.

Every attack answers the same question -- *which known author wrote this unknown document?* --
and every attack answers it in the same shape: an ``[n_documents x n_authors]`` **score** matrix
where **higher means more likely this author**, with the author order in ``self.authors``.
Identification is ``argmax``; :mod:`prompt_anonymity.attacks.ood` turns the same matrix into an
accept/reject decision, and :mod:`prompt_anonymity.metrics` scores it.

The author is the unit throughout. There is deliberately no conversation-level attack any more:
a matrix over known *conversations* answers a different question from the one the metrics report,
and mixing the two invites reading a distance as a score with the sign inverted.

Packages
--------
=============================================  ===========================================
:mod:`~prompt_anonymity.attacks.similarity`    summarise each author, score the match
:mod:`~prompt_anonymity.attacks.multiclass`    fit a decision function per author
:mod:`~prompt_anonymity.attacks.llm`           shortlist cheaply, then let a judge reorder
:mod:`~prompt_anonymity.attacks.verification`  learned same-author scoring over pairs
:mod:`~prompt_anonymity.attacks.ood`           score matrix -> accept or reject
=============================================  ===========================================

:mod:`~prompt_anonymity.attacks.common` holds the helpers the families share -- the linear
algebra, and the author-grouping layout that lets an aggregation run as one ``reduceat`` pass.

Interfaces
----------
The two families of *estimator* share one contract: ``fit(embeddings, labels) -> self``, then
``score(embeddings) -> (n_documents, n_authors)``. Register a new one in
:data:`ATTRIBUTION_ATTACKS` and it becomes selectable by name with no change to the runner.

The LLM rerankers and the cross-encoder take a whole :class:`~prompt_anonymity.core.AttackData`
instead, because they need more than embeddings -- raw conversation text for the judges, known
labels for the shortlist. They return the same author-score frame.
"""

from __future__ import annotations

from .common import (
    AuthorGroups,
    class_means,
    group_by_author,
    inverse_sqrt,
    unit_rows,
    within_class_covariance,
)
from .llm import (
    LLM_ATTACKS,
    AuthorCandidates,
    BradleyTerryTournamentAttack,
    EuclideanLLMJudgeAttack,
    author_candidates,
    bt_tournament_attack,
    euclidean_llm_judge_attack,
)
from .multiclass import (
    MULTICLASS_ATTACKS,
    GradientBoostedTrees,
    LogisticAttribution,
    RegularizedLeastSquares,
    SupportVectorAttribution,
)
from .ood import LearnedRejector, cohort_normalize, rejection_features, rejection_score
from .similarity import (
    SIMILARITY_ATTACKS,
    CentroidCosine,
    LDACentroid,
    NearestNeighbor,
    PLDA,
    WhitenedCentroid,
    block_row_count,
    blocked_distances,
)
from .verification import VERIFICATION_ATTACKS, run_cross_encoder

# One registry over both estimator families, so a caller selects an attack by name without
# caring which package it lives in. Values are classes, not instances: each is constructed per
# fit, because threshold calibration refits the same configuration on many simulation folds.
ATTRIBUTION_ATTACKS = {**SIMILARITY_ATTACKS, **MULTICLASS_ATTACKS}


def get_attribution_attack(name: str):
    """Look up a registered attribution attack by name."""
    try:
        return ATTRIBUTION_ATTACKS[name]
    except KeyError:
        raise ValueError(
            f"unknown attribution attack {name!r}; available: {sorted(ATTRIBUTION_ATTACKS)}"
        ) from None


__all__ = [
    "ATTRIBUTION_ATTACKS",
    "get_attribution_attack",
    # similarity-based
    "SIMILARITY_ATTACKS",
    "NearestNeighbor",
    "CentroidCosine",
    "WhitenedCentroid",
    "LDACentroid",
    "PLDA",
    "blocked_distances",
    "block_row_count",
    # multiclass discriminant
    "MULTICLASS_ATTACKS",
    "LogisticAttribution",
    "SupportVectorAttribution",
    "GradientBoostedTrees",
    "RegularizedLeastSquares",
    # llm-assisted rerankers
    "LLM_ATTACKS",
    "EuclideanLLMJudgeAttack",
    "euclidean_llm_judge_attack",
    "BradleyTerryTournamentAttack",
    "bt_tournament_attack",
    "author_candidates",
    "AuthorCandidates",
    # pairwise verification
    "VERIFICATION_ATTACKS",
    "run_cross_encoder",
    # open-set
    "cohort_normalize",
    "rejection_score",
    "rejection_features",
    "LearnedRejector",
    # shared helpers
    "AuthorGroups",
    "group_by_author",
    "unit_rows",
    "class_means",
    "within_class_covariance",
    "inverse_sqrt",
]
