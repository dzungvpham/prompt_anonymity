"""LLM-assisted attacks: a cheap vector attack shortlists, a language model reorders.

Both attacks here are **rerankers**. A vector attack proposes the most likely authors, and a
language-model judge reads the actual conversation text and reorders the shortlist. The judge
only ever sees a handful of candidates per document, which is what makes the cost bearable and
also what bounds the gain: a reranker can only fix what its shortlist contains.

The shortlist is over **authors**, built by :mod:`prompt_anonymity.attacks.llm.candidates` --
see that module for why shortlisting conversations instead quietly answers a different question.

=========================  ==================================================================
attack                     how the reranker is asked
=========================  ==================================================================
``euclidean_llm_judge``    one prompt per document: "which of these k candidates wrote it?"
``bt_tournament``          Swiss-paired 1-vs-1 comparisons, Elo-updated over several rounds
``listwise_llm_rerank``    one prompt per document: "rank ALL k, and say why for each position"
``listwise_jina_rerank``   the same shortlist scored locally by jina-reranker-v3.5 (no API)
=========================  ==================================================================

All four gate on the base attack's best/second-best author margin, so the reranker is spent only on
the rows where the vectors were close to a coin flip, and all four leave shortlist membership
untouched. What differs is how much of the order they may rewrite. The first two promote a single
winner, so only top-1 can move; the two ``listwise_*`` attacks rewrite the whole shortlist, so
top-1 through top-(k-1) move and only top-k stays pinned to the base attack -- see
:mod:`prompt_anonymity.attacks.llm.listwise`, which holds the shortlist presentation and fold-back
they share.

``listwise_jina_rerank`` is the control for ``listwise_llm_rerank``: identical shortlist, identical
presentation, no API bill, so the gap between them prices the frontier model rather than the idea.
"""

from __future__ import annotations

from .bt_tournament import BradleyTerryTournamentAttack, bt_tournament_attack
from .candidates import AuthorCandidates, author_candidates
from .euclidean_llm_judge import EuclideanLLMJudgeAttack, euclidean_llm_judge_attack
from .listwise import Presentation, detail_table, fold_listwise, present
from .listwise_jina_rerank import ListwiseJinaRerankAttack, listwise_jina_rerank_attack
from .listwise_llm_rerank import ListwiseLLMRerankAttack, listwise_llm_rerank_attack

LLM_ATTACKS = {
    "euclidean_llm_judge": euclidean_llm_judge_attack,
    "bt_tournament": bt_tournament_attack,
    "listwise_llm_rerank": listwise_llm_rerank_attack,
    "listwise_jina_rerank": listwise_jina_rerank_attack,
}

__all__ = ["LLM_ATTACKS", "EuclideanLLMJudgeAttack", "euclidean_llm_judge_attack",
           "BradleyTerryTournamentAttack", "bt_tournament_attack",
           "ListwiseLLMRerankAttack", "listwise_llm_rerank_attack",
           "ListwiseJinaRerankAttack", "listwise_jina_rerank_attack",
           "author_candidates", "AuthorCandidates",
           "Presentation", "present", "fold_listwise", "detail_table"]
