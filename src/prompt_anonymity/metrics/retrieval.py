"""Author-as-query retrieval metrics: *find everything this user wrote*.

Every other metric in this package runs the attack in the identification direction -- given an
anonymous document, which user wrote it? This module runs the same score matrix the other way:
given a **known user as the query**, rank all the anonymous documents and see how many of theirs
come to the top. That is a genuinely different attacker capability (targeted surveillance of one
person, rather than triage of one document) and it is usually the one a privacy threat model
cares about, so it deserves its own numbers rather than being inferred from top-k accuracy.

It also fixes a degeneracy. Identification has exactly one relevant item per query, which
collapses mean average precision onto the mean reciprocal rank already reported by
:mod:`prompt_anonymity.metrics.ranking` -- reporting both would be reporting one number twice.
In this direction a query has many relevant documents, so average precision is non-trivial and
measures what it is meant to: whether the user's documents are concentrated at the top of the
ranking or scattered through it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


def author_query_metrics(scores, candidate_authors, true_authors) -> pd.DataFrame:
    """Rank every document against each author in turn; report per-author retrieval quality.

    Parameters
    ----------
    scores : array-like of shape (n_documents, n_candidates)
        Attack score for every (document, candidate author) pair, higher = more likely. The
        same matrix :func:`prompt_anonymity.metrics.ranking.true_author_ranks` consumes, read
        column-wise instead of row-wise.
    candidate_authors : array-like of shape (n_candidates,)
        Author identifier owning each column.
    true_authors : array-like of shape (n_documents,)
        True author of each document.

    Returns
    -------
    pandas.DataFrame
        One row per author who owns at least one document (an author with none has no relevant
        item, so average precision is undefined for them), sorted best-retrieved first:

        ``author``, ``n_relevant``
            The query and how many documents it should retrieve.
        ``average_precision``
            Precision averaged over the positions of that author's documents, from
            scikit-learn's ``average_precision_score``. Rewards concentration at the top.
        ``r_precision``
            Precision at cutoff R, where R is the author's own document count -- a
            self-normalising operating point, so it is comparable between a user with 5
            documents and one with 200 in a way that precision@10 is not.
        ``random_average_precision``
            Chance baseline, ``n_relevant / n_documents``: the precision of a random ranking,
            which both metrics above reduce to when the scores carry no signal.

    Notes
    -----
    Average the ``average_precision`` column for **MAP** and ``r_precision`` for mean
    R-precision; :func:`retrieval_summary` does that with the matching baselines.
    """
    scores = np.asarray(scores, dtype=float)
    candidate_authors = np.asarray(candidate_authors)
    true_authors = np.asarray(true_authors)
    if scores.shape != (len(true_authors), len(candidate_authors)):
        raise ValueError(
            f"scores shape {scores.shape} does not match ({len(true_authors)} documents, "
            f"{len(candidate_authors)} candidate authors)."
        )

    column_of = {author: index for index, author in enumerate(candidate_authors)}
    rows = []
    for author in np.unique(true_authors):
        column = column_of.get(author)
        if column is None:  # no score column: this author was never a candidate
            continue
        relevant = true_authors == author
        n_relevant = int(relevant.sum())
        column_scores = scores[:, column]
        if not np.isfinite(column_scores).all():
            # A candidate filter (``run_experiment.py --language-aware``) scores the documents
            # this author was never a candidate for as -inf, meaning "would never be retrieved":
            # they belong at the bottom of the ranking. Both metrics below read only the *order*
            # of the scores, so substituting a value below every real one is exact -- and it is
            # necessary, because average_precision_score rejects non-finite input outright.
            eligible = np.isfinite(column_scores)
            floor = column_scores[eligible].min() - 1.0 if eligible.any() else 0.0
            column_scores = np.where(eligible, column_scores, floor)
        # Documents ranked by how strongly this author's model claims them.
        order = np.argsort(-column_scores, kind="mergesort")
        rows.append({
            "author": author,
            "n_relevant": n_relevant,
            "average_precision": float(average_precision_score(relevant, column_scores)),
            "r_precision": float(relevant[order][:n_relevant].mean()),
            "random_average_precision": n_relevant / len(true_authors),
        })
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    return table.sort_values("average_precision", ascending=False, ignore_index=True)


def retrieval_summary(table: pd.DataFrame) -> dict:
    """Collapse :func:`author_query_metrics` to ``map``, ``mean_r_precision`` and their baseline.

    Authors are weighted equally (macro averaging), which is the convention for MAP and is also
    the right choice here: weighting by document count would let a few prolific users decide the
    number, exactly the skew the macro metrics in :mod:`prompt_anonymity.metrics.ranking` exist
    to avoid.
    """
    if table.empty:
        return {"map": float("nan"), "mean_r_precision": float("nan"), "random_map": float("nan")}
    return {
        "map": float(table["average_precision"].mean()),
        "mean_r_precision": float(table["r_precision"].mean()),
        "random_map": float(table["random_average_precision"].mean()),
    }
