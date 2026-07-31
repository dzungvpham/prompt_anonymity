"""Linkage attacks: two families, two registries.

**Unsupervised, conversation-level** (``ATTACKS``) -- an attack ranks the known (labeled)
conversations for each unknown (anonymous) conversation by producing an
``[n_unknown x n_known]`` **distance** matrix, which the metrics in
:mod:`prompt_anonymity.metrics` turn into top-k accuracy. Two layers are exposed:

* the low-level functions (:func:`nearest_neighbor_attack`) that operate directly on
  embedding arrays -- use these when you already have the arrays in hand; and
* the ``AttackData``-level interface (``Attack``, ``ATTACKS``, :func:`run_attack`) that
  runs an attack on a loaded dataset by name -- use this to keep experiment code
  agnostic to which attack is selected.

Add one by writing an ``Attack`` (a callable from :class:`~prompt_anonymity.core.AttackData`
to a distance DataFrame) and registering it in ``ATTACKS``.

**Supervised, author-level** (:data:`ATTRIBUTION_ATTACKS`, in
:mod:`~prompt_anonymity.attacks.attribution`) -- a model *trained on the known side*, which is
legitimate because the known conversations and their labels belong to the attacker. It exposes
``fit(embeddings, labels)`` / ``score(embeddings)`` and produces an ``[n_docs x n_authors]``
**score** matrix (higher = more likely), not a distance matrix over conversations. Training on
the attacker's own data is worth roughly a doubling of top-1 over the same features scored by
cosine distance, so the two families are not interchangeable baselines --
:mod:`~prompt_anonymity.attacks.rejection` then turns a score matrix into an accept/reject
decision for the open-set case.

The registries are separate because the interfaces are: anything consuming ``ATTACKS`` expects
a distance DataFrame and would misread a score matrix as one (the sign is inverted). Convert
with ``-scores`` when a distance-shaped consumer needs one.
"""

from __future__ import annotations
from typing import Callable
import pandas as pd
from .two_tower_xgb import run_two_tower_xgb
from ..core import AttackData
from .nearest_neighbor import nearest_neighbor_attack
from .euclidean_llm_judge import EuclideanLLMJudgeAttack, euclidean_llm_judge_attack
from .bt_tournament import BradleyTerryTournamentAttack, bt_tournament_attack
from .attribution import (
    ATTRIBUTION_ATTACKS,
    CentroidCosine,
    LDACentroid,
    LogisticAttribution,
    NearestNeighbor,
    PLDA,
    WhitenedCentroid,
    get_attribution_attack,
)
from .rejection import LearnedRejector, cohort_normalize, rejection_features, rejection_score

# An attack maps a loaded dataset to an [n_unknown x n_known] distance matrix.
Attack = Callable[[AttackData], pd.DataFrame]


def run_nearest_neighbor(data: AttackData) -> pd.DataFrame:
    """Run :func:`nearest_neighbor_attack` on an :class:`AttackData`, using its metric."""
    return nearest_neighbor_attack(data.known_embeddings, data.unknown_embeddings, metric=data.metric)


def run_euclidean_llm_judge(data: AttackData) -> pd.DataFrame:
    """Rerank the nearest-neighbor top-K with the default LLM judge (uncached; see
    :func:`~prompt_anonymity.attacks.euclidean_llm_judge_attack` to pass a ``cache_dir`` or
    tune the judge)."""
    return euclidean_llm_judge_attack(data)


def run_bt_tournament(data: AttackData) -> pd.DataFrame:
    """Rerank the nearest-neighbor top-K with the default Bradley-Terry pairwise-judge
    tournament (uncached; see :func:`~prompt_anonymity.attacks.bt_tournament_attack` to pass a
    ``cache_dir`` or tune the tournament)."""
    return bt_tournament_attack(data)


# Registry so callers can select an attack by name (e.g. from a CLI argument).
ATTACKS: dict[str, Attack] = {
    "nearest_neighbor": run_nearest_neighbor,
    "two_tower_xgb": run_two_tower_xgb,
    "euclidean_llm_judge": run_euclidean_llm_judge,
    "bt_tournament": run_bt_tournament,
}


def get_attack(name: str) -> Attack:
    """Look up a registered attack by name."""
    try:
        return ATTACKS[name]
    except KeyError:
        raise ValueError(f"unknown attack {name!r}; available: {sorted(ATTACKS)}") from None


def run_attack(name: str, data: AttackData) -> pd.DataFrame:
    """Run the named attack on ``data`` and return its distance matrix."""
    return get_attack(name)(data)


__all__ = [
    # unsupervised, conversation-level: AttackData -> distance matrix
    "nearest_neighbor_attack",
    "EuclideanLLMJudgeAttack",
    "euclidean_llm_judge_attack",
    "BradleyTerryTournamentAttack",
    "bt_tournament_attack",
    "Attack",
    "ATTACKS",
    "get_attack",
    "run_attack",
    "run_nearest_neighbor",
    "run_two_tower_xgb",
    "run_euclidean_llm_judge",
    "run_bt_tournament",
    # supervised, author-level: fit(known) -> score matrix over authors
    "ATTRIBUTION_ATTACKS",
    "get_attribution_attack",
    "NearestNeighbor",
    "CentroidCosine",
    "WhitenedCentroid",
    "LDACentroid",
    "LogisticAttribution",
    "PLDA",
    # score matrix -> accept/reject
    "cohort_normalize",
    "rejection_score",
    "rejection_features",
    "LearnedRejector",
]
