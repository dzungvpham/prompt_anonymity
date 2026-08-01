"""Metrics derived from a per-document ranking of the candidate authors.

An attribution attack scores *every* candidate author for every unknown document, which is a
much richer object than its top-1 guess: the **rank** of the true author says how close the
attack came even when it was wrong. Everything in this module is computed from those ranks, so
a single :func:`true_author_ranks` call feeds the whole family cheaply.

These answer two questions the top-k tables in :mod:`prompt_anonymity.metrics.accuracy` cannot:

* **How close was the attack overall?** ``top_k_accuracy`` samples the ranking at a handful of
  cutoffs. :func:`ranking_summary` reports the mean reciprocal rank and the mean percentile
  rank, both of which use the whole ranking, and :func:`cmc_curve` traces every cutoff at once.
  The percentile rank matters when comparing experiments whose candidate pools differ in size:
  "top-10 of 81 candidates" and "top-10 of 124" are not the same achievement, but their
  percentile ranks are directly comparable.
* **Is the attack re-identifying *users*, or just the users who write a lot?** Document-weighted
  (micro) accuracy is dominated by prolific authors -- in one SWE-chat window a single author
  owns 184 of 819 documents. The ``macro_`` functions average over authors instead, and
  :func:`per_author_ranking` returns the whole distribution, which is the honest way to state a
  privacy result: not "26% of documents" but "how many users are at serious risk, and how badly".

Score orientation
-----------------
Everything here takes **scores, where higher means more likely** -- the orientation an
attribution model produces. That is the opposite of the distance matrices consumed by
:func:`prompt_anonymity.metrics.top_k_accuracy`; negate a distance matrix to use it here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


def true_author_ranks(scores, candidate_authors, true_authors) -> np.ndarray:
    """Rank of each document's true author among the candidates (1 = top of the ranking).

    Parameters
    ----------
    scores : array-like of shape (n_documents, n_candidates)
        Attack score for every (document, candidate author) pair, higher = more likely.
    candidate_authors : array-like of shape (n_candidates,)
        Author identifier owning each column of ``scores``.
    true_authors : array-like of shape (n_documents,)
        True author of each row. Every one must appear in ``candidate_authors``: a document
        whose author is not a candidate has no rank at all, so filter those out first. They
        are the out-of-set documents, scored by :mod:`prompt_anonymity.metrics.detection`.

    Returns
    -------
    numpy.ndarray of shape (n_documents,)
        Ranks as **floats**, because ties are averaged: if the true author ties with one other
        candidate for the best score, the rank is 1.5 rather than an arbitrary 1 or 2. Averaging
        keeps every metric below unbiased under ties instead of rewarding or punishing whichever
        order the sort happened to produce -- which matters for attacks whose scores are coarse
        or saturate.

    Notes
    -----
    Only the true author's rank is computed, never the full ranking. Sorting each row would
    build an ``(n_documents, n_candidates)`` rank matrix -- 6 GB at 50,000 documents over 15,000
    candidates -- and then discard all but one entry per row. Counting how many candidates beat
    the true author gives the identical number (ties included, see below) in a single pass, which
    is both faster and bounded by one boolean array rather than a float64 one.
    """
    scores = np.asarray(scores)
    if not np.issubdtype(scores.dtype, np.floating):
        scores = scores.astype(float)
    candidate_authors = np.asarray(candidate_authors)
    true_authors = np.asarray(true_authors)
    if scores.ndim != 2:
        raise ValueError(f"scores must be 2-D (n_documents, n_candidates); got shape {scores.shape}.")
    if scores.shape[1] != len(candidate_authors):
        raise ValueError(
            f"candidate_authors has {len(candidate_authors)} entries but scores has "
            f"{scores.shape[1]} columns (one per candidate author)."
        )
    if scores.shape[0] != len(true_authors):
        raise ValueError(
            f"true_authors has {len(true_authors)} entries but scores has {scores.shape[0]} "
            f"rows (one per document)."
        )

    column_of = {author: index for index, author in enumerate(candidate_authors)}
    missing = set(np.unique(true_authors).tolist()) - set(column_of)
    if missing:
        raise ValueError(
            f"{len(missing)} true author(s) are not candidates, so they have no rank "
            f"(e.g. {sorted(missing)[:3]}). Restrict to in-set documents first."
        )

    true_columns = np.array([column_of[author] for author in true_authors])
    true_scores = scores[np.arange(len(true_authors)), true_columns][:, None]

    # A value beaten by ``better`` candidates and tied with ``tied`` others (itself excluded)
    # occupies ranks better+1 ... better+tied+1, whose average is better + 1 + tied/2. This is
    # exactly scipy's method="average", without sorting anything.
    better = (scores > true_scores).sum(axis=1)
    tied = (scores == true_scores).sum(axis=1) - 1
    return better + 1.0 + tied / 2.0


def ranking_summary(ranks, n_candidates: int) -> dict:
    """Whole-ranking summary statistics, each paired with its uniform-random baseline.

    Parameters
    ----------
    ranks : array-like of shape (n_documents,)
        Output of :func:`true_author_ranks`.
    n_candidates : int
        Size of the candidate pool the attack ranked (the number of columns it scored, not the
        number of authors that happen to appear in the documents). This sets every baseline.

    Returns
    -------
    dict
        ``mrr``
            Mean reciprocal rank, ``mean(1 / rank)``. The standard summary for retrieval with
            exactly one relevant item per query, which is what author attribution is. Dominated
            by the top of the ranking: moving the true author from rank 2 to rank 1 is worth as
            much as moving it from rank 10 to rank 2. Equals scikit-learn's
            ``label_ranking_average_precision_score`` for the single-relevant-label case, and
            equals mean average precision, so reporting MAP as well would be redundant. (nDCG
            likewise collapses to a monotone function of the rank here and adds nothing.)
        ``random_mrr``
            MRR of a uniformly random ranking, ``H_n / n`` for the harmonic number ``H_n``.
        ``mean_percentile_rank``
            Mean of ``1 - (rank - 1) / (n_candidates - 1)``: the fraction of candidates the
            attack places *below* the true author. 1.0 = always ranked first, 0.5 = chance,
            regardless of pool size -- the one metric here that is comparable across
            experiments with different numbers of candidates.
        ``median_rank`` / ``mean_rank``
            Where the true author lands, in candidates. The median is the interpretable
            privacy statement ("the attacker can shortlist the median document to N
            candidates"); the mean is dragged around by the tail.
        ``random_median_rank``
            ``(n_candidates + 1) / 2``, the median rank under uniform-random ordering.
    """
    ranks = np.asarray(ranks, dtype=float)
    if n_candidates < 1:
        raise ValueError(f"n_candidates must be a positive integer (got {n_candidates}).")
    if ranks.size == 0:
        return dict.fromkeys(
            ("mrr", "random_mrr", "mean_percentile_rank", "median_rank", "mean_rank",
             "random_median_rank"), float("nan"))

    harmonic = float(np.sum(1.0 / np.arange(1, n_candidates + 1)))
    # A single-candidate pool makes the percentile rank degenerate (everything is first).
    percentile = (np.ones_like(ranks) if n_candidates == 1
                  else 1.0 - (ranks - 1.0) / (n_candidates - 1))
    return {
        "mrr": float(np.mean(1.0 / ranks)),
        "random_mrr": harmonic / n_candidates,
        "mean_percentile_rank": float(np.mean(percentile)),
        "median_rank": float(np.median(ranks)),
        "mean_rank": float(np.mean(ranks)),
        "random_median_rank": (n_candidates + 1) / 2,
    }


def cmc_curve(ranks, n_candidates: int, ks=None) -> pd.DataFrame:
    """Cumulative match characteristic: top-k accuracy at *every* cutoff k.

    The CMC curve is the re-identification literature's name for the top-k accuracy curve, and
    it is what the three-row headline table is sampling. Plotting it whole shows the shape of
    the attack -- whether it is confidently right, or merely narrowing a large pool -- which
    three cutoffs cannot: two attacks with identical top-1 can have very different curves.

    Parameters
    ----------
    ranks : array-like of shape (n_documents,)
        Output of :func:`true_author_ranks`.
    n_candidates : int
        Candidate-pool size, i.e. where the curve necessarily reaches 1.0.
    ks : sequence of int, optional
        Cutoffs to evaluate. Defaults to every ``k`` from 1 to ``n_candidates``.

    Returns
    -------
    pandas.DataFrame
        Columns ``k``, ``accuracy`` (fraction of documents whose true author ranks within k),
        ``random`` (``min(k, n) / n``) and ``advantage`` (the difference). Averaged tie ranks
        are compared with ``<=``, so a document tied for first counts as a hit only from the
        cutoff its averaged rank reaches -- the conservative reading.
    """
    ranks = np.asarray(ranks, dtype=float)
    if ks is None:
        ks = range(1, int(n_candidates) + 1)
    rows = []
    for k in ks:
        accuracy = float(np.mean(ranks <= k)) if ranks.size else float("nan")
        chance = min(int(k), n_candidates) / n_candidates
        rows.append({"k": int(k), "accuracy": accuracy, "random": chance,
                     "advantage": accuracy - chance})
    return pd.DataFrame(rows)


def macro_top_k_accuracy(ranks, true_authors, k: int = 1) -> float:
    """Top-k accuracy averaged over **authors** rather than over documents.

    Each author's own document-level accuracy is computed first, then those are averaged with
    equal weight. This is the number to quote when the document counts are skewed: the
    document-weighted (micro) version answers "what fraction of traffic can be attributed",
    which a handful of heavy users can carry on their own, while this one answers "how well
    does the attack do against a typical user".

    Not the same as identity-level accuracy in :func:`prompt_anonymity.metrics.top_k_accuracy`,
    which counts an author as re-identified if *any single one* of their documents lands in the
    top k. That is the attacker's best case; this is their average case.
    """
    ranks = np.asarray(ranks, dtype=float)
    true_authors = np.asarray(true_authors)
    if k < 1:
        raise ValueError(f"k must be a positive integer (got {k}).")
    if ranks.size == 0:
        return float("nan")
    hit = ranks <= k
    return float(np.mean([hit[true_authors == author].mean() for author in np.unique(true_authors)]))


def macro_f1_score(true_authors, predicted_authors) -> float:
    """Macro-averaged F1 over the authors, via scikit-learn.

    The standard closed-set attribution summary, and a useful complement to accuracy because it
    penalises an attack that wins by funnelling everything into a few prolific authors: doing so
    wrecks precision on those classes and recall on the rest. Labels are the authors present in
    ``true_authors``, so candidates that never occur do not pad the average with zeros;
    predictions naming an absent author still cost precision on the class they were taken from.
    """
    true_authors = np.asarray(true_authors)
    predicted_authors = np.asarray(predicted_authors)
    if true_authors.size == 0:
        return float("nan")
    return float(f1_score(true_authors, predicted_authors, labels=np.unique(true_authors),
                          average="macro", zero_division=0))


def per_author_ranking(ranks, true_authors, top_ks=(1, 5, 10)) -> pd.DataFrame:
    """Per-author breakdown of the ranking: the risk distribution behind the averages.

    An average accuracy hides which users are actually exposed. In practice the risk is highly
    unequal -- on SWE-chat most target users are never attributed correctly even once while a
    few are attributed nearly always -- and that distribution, not its mean, is what a privacy
    claim should rest on.

    Returns one row per distinct author in ``true_authors``, sorted most-exposed first, with
    columns ``author``, ``n_documents``, ``best_rank`` (their single most-identifiable
    document), ``median_rank``, and for each k in ``top_ks`` both ``top<k>_accuracy`` (the share
    of their documents ranked within k) and ``reidentified_top<k>`` (whether *any* of them was --
    the identity-level criterion). Join it with
    :func:`prompt_anonymity.metrics.retrieval.author_query_metrics` on ``author`` for the
    matching retrieval view.
    """
    ranks = np.asarray(ranks, dtype=float)
    true_authors = np.asarray(true_authors)
    rows = []
    for author in np.unique(true_authors):
        author_ranks = ranks[true_authors == author]
        row = {"author": author, "n_documents": len(author_ranks),
               "best_rank": float(author_ranks.min()), "median_rank": float(np.median(author_ranks))}
        for k in top_ks:
            row[f"top{k}_accuracy"] = float(np.mean(author_ranks <= k))
            row[f"reidentified_top{k}"] = bool(np.any(author_ranks <= k))
        rows.append(row)
    table = pd.DataFrame(rows)
    return table.sort_values([f"top{top_ks[0]}_accuracy", "n_documents"], ascending=False,
                             ignore_index=True)
