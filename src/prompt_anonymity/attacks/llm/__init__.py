"""LLM-assisted attacks: a cheap vector attack shortlists, a language model reorders.

Both attacks here are **rerankers**. A vector attack proposes the most likely authors, and a
language-model judge reads the actual conversation text and reorders the shortlist. The judge
only ever sees a handful of candidates per document, which is what makes the cost bearable and
also what bounds the gain: a reranker can only fix what its shortlist contains.

The shortlist is over **authors**, built by :mod:`prompt_anonymity.attacks.llm.candidates` --
see that module for why shortlisting conversations instead quietly answers a different question.

=========================  ==================================================================
attack                     how the judge is asked
=========================  ==================================================================
``euclidean_llm_judge``    one prompt per document: "which of these k candidates wrote it?"
``bt_tournament``          Swiss-paired 1-vs-1 comparisons, Elo-updated over several rounds
=========================  ==================================================================

Both gate on the base attack's best/second-best author margin, so the judge is spent only on the
rows where the vectors were close to a coin flip, and both leave shortlist membership untouched
-- only the order within it, and hence top-1, can move.
"""

from __future__ import annotations

from .bt_tournament import BradleyTerryTournamentAttack, bt_tournament_attack
from .candidates import AuthorCandidates, author_candidates
from .euclidean_llm_judge import EuclideanLLMJudgeAttack, euclidean_llm_judge_attack

LLM_ATTACKS = {
    "euclidean_llm_judge": euclidean_llm_judge_attack,
    "bt_tournament": bt_tournament_attack,
}

__all__ = ["LLM_ATTACKS", "EuclideanLLMJudgeAttack", "euclidean_llm_judge_attack",
           "BradleyTerryTournamentAttack", "bt_tournament_attack",
           "author_candidates", "AuthorCandidates"]
