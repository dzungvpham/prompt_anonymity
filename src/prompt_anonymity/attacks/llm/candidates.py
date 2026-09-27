"""Shortlisting the candidates an LLM judge is asked about.

Both rerankers in this package work the same way: a cheap vector attack proposes a shortlist, an
expensive language-model judge reorders it. This module builds the shortlist, and the unit it
shortlists is the **author**, not the conversation.

Why authors and not conversations
---------------------------------
The question being asked is "who wrote this?", so the candidate set has to be a set of people.
Taking the k nearest *conversations* instead answers a different question and answers it badly:

* **The shortlist silently shrinks.** One prolific author can own many of the k nearest
  documents, so a k-document shortlist covers fewer than k candidates. The judge is handed
  duplicates of the same person instead of alternatives.
* **The judge's ballot does not match the metric being reported.** Top-k accuracy is scored over
  authors, so a document shortlist makes the reranker optimise one thing and the metric measure
  another.
* **The recall ceiling is lower.** A reranker can only fix what its shortlist contains, and at
  equal k the author shortlist contains strictly more people.

So the shortlist is the top-k authors under
:class:`~prompt_anonymity.attacks.similarity.NearestNeighbor`, and each shortlisted author is
*represented* to the judge by their own document that is nearest to the unknown one -- their best
case, which is the fair thing to put in front of a judge that reads one text per candidate.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from ..similarity import NearestNeighbor, blocked_distances


class AuthorCandidates(NamedTuple):
    """A per-document shortlist of authors, each with a representative known conversation.

    Attributes
    ----------
    authors : numpy.ndarray of shape (n_authors,)
        Author labels, indexing the columns of ``scores``.
    scores : numpy.ndarray of shape (n_unknown, n_authors)
        Base attack score, higher = more likely. The reranker edits a copy of this.
    author_index : numpy.ndarray of shape (n_unknown, k)
        Column indices into ``scores`` of each row's shortlisted authors, best first.
    document_index : numpy.ndarray of shape (n_unknown, k)
        For each shortlisted author, the row index (into the *known* arrays as originally passed)
        of that author's document nearest to this unknown one -- the text to show the judge.
    margin : numpy.ndarray of shape (n_unknown,)
        Score gap between the best and second-best **author**. Small means the top two candidates
        are near-tied, which is where a judge can plausibly help; the rerankers gate on it.
    """

    authors: np.ndarray
    scores: np.ndarray
    author_index: np.ndarray
    document_index: np.ndarray
    margin: np.ndarray


def author_candidates(known_embeddings, known_labels, unknown_embeddings, *,
                      top_k: int, metric: str = "cosine",
                      linkage: str = "max") -> AuthorCandidates:
    """Shortlist the ``top_k`` most likely authors per unknown document.

    Parameters
    ----------
    known_embeddings, unknown_embeddings : array-like of shape (n, n_features)
    known_labels : array-like of shape (n_known,)
        Author label of each known document.
    top_k : int
        Shortlist size, clipped to the number of authors available.
    metric, linkage
        Passed to :class:`~prompt_anonymity.attacks.similarity.NearestNeighbor`.

    Returns
    -------
    AuthorCandidates
    """
    ranker = NearestNeighbor(metric=metric, linkage=linkage)
    ranker.fit(known_embeddings, known_labels)
    scores = np.asarray(ranker.score(unknown_embeddings), dtype=float)

    n_unknown, n_authors = scores.shape
    k = int(min(top_k, n_authors))
    if k < 1:
        raise ValueError(f"top_k must leave at least one candidate (got {top_k} with {n_authors} authors).")

    # Best-first shortlist. argpartition finds the k best without sorting all n_authors columns,
    # which matters when the pool is tens of thousands wide; the k survivors are then sorted.
    if k < n_authors:
        shortlist = np.argpartition(-scores, k - 1, axis=1)[:, :k]
    else:
        shortlist = np.tile(np.arange(n_authors), (n_unknown, 1))
    shortlist_scores = np.take_along_axis(scores, shortlist, axis=1)
    within = np.argsort(-shortlist_scores, axis=1, kind="stable")
    author_index = np.take_along_axis(shortlist, within, axis=1)

    ordered = np.take_along_axis(scores, author_index, axis=1)
    margin = (ordered[:, 0] - ordered[:, 1] if k >= 2
              else np.zeros(n_unknown, dtype=float))

    document_index = _nearest_document_per_author(
        known_embeddings, unknown_embeddings, ranker, author_index, metric
    )
    return AuthorCandidates(authors=ranker.authors, scores=scores, author_index=author_index,
                            document_index=document_index, margin=margin)


def _nearest_document_per_author(known_embeddings, unknown_embeddings, ranker,
                                 author_index, metric) -> np.ndarray:
    """For each (document, shortlisted author) pair, that author's nearest known document.

    Grouped by author rather than looped per pair: every unknown document that shortlisted the
    same author is scored against that author's documents in one vectorised call, so the cost is
    one small distance computation per *distinct shortlisted author* instead of one per pair.
    """
    known = np.asarray(known_embeddings)
    unknown = np.asarray(unknown_embeddings)
    document_index = np.empty(author_index.shape, dtype=np.intp)

    rows, slots = np.nonzero(np.ones_like(author_index, dtype=bool))
    flat_authors = author_index.ravel()
    order = np.argsort(flat_authors, kind="stable")
    boundaries = np.flatnonzero(np.diff(flat_authors[order])) + 1

    for group in np.split(order, boundaries):
        if not len(group):
            continue
        author = int(flat_authors[group[0]])
        documents = ranker.documents_of(author)
        group_rows = rows[group]
        if len(documents) == 1:
            document_index[group_rows, slots[group]] = documents[0]
            continue
        # One block per author group; these are small (rows that picked this author x that
        # author's documents), so a single block is the normal case.
        best = np.empty(len(group_rows), dtype=np.intp)
        for start, block in blocked_distances(unknown[group_rows], known[documents],
                                              metric=metric, dtype=np.float64):
            best[start:start + len(block)] = documents[block.argmin(axis=1)]
        document_index[group_rows, slots[group]] = best
    return document_index
