"""Linkage attacks.

An attack ranks the known (labeled) conversations for each unknown (anonymous)
conversation by producing an ``[n_unknown x n_known]`` distance matrix, which the
metrics in :mod:`prompt_anonymity.metrics` turn into top-k accuracy.

Two layers are exposed:

* the low-level functions (:func:`nearest_neighbor_attack`) that operate directly on
  embedding arrays -- use these when you already have the arrays in hand; and
* the ``AttackData``-level interface (``Attack``, ``ATTACKS``, :func:`run_attack`) that
  runs an attack on a loaded dataset by name -- use this to keep experiment code
  agnostic to which attack is selected.

Add an attack by writing an ``Attack`` (a callable from
:class:`~prompt_anonymity.core.AttackData` to a distance DataFrame) and registering it
in ``ATTACKS``.
"""

from __future__ import annotations
from typing import Callable
import pandas as pd
from .two_tower_xgb import run_two_tower_xgb
from ..core import AttackData
from .nearest_neighbor import nearest_neighbor_attack
from .euclidean_llm_judge import EuclideanLLMJudgeAttack, euclidean_llm_judge_attack
from .bt_tournament import BradleyTerryTournamentAttack, bt_tournament_attack

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
]
