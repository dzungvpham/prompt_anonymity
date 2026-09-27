"""Extrinsic scoring for author *clustering*: was one person's traffic grouped together?

Every other module in :mod:`prompt_anonymity.evaluation.metrics` scores a ranking over **named**
candidate authors -- *identifiability*, "who wrote this?". This one scores a **partition** of
anonymous documents against the true authorship -- *linkability*, "which of these were written by
the same person?". No author is ever named, and the attack that produced the partition may have
had no labelled data at all, which is what makes it the stronger privacy claim: a log dump with
every identifier stripped is still partitionable, and a partition is what turns one successful
de-anonymisation into an entire user's history (see ``amplification`` below).

The measure family is **BCubed**, following PAN 2016's author-clustering task (Stamatatos et al.,
*Clustering by Authorship Within and Across Documents*, CLEF 2016 Working Notes). BCubed is
element-based rather than set-based, which is what makes it usable here: it needs no matching
between predicted clusters and true authors, so it is defined when the two have wildly different
cardinalities -- and it satisfies the four formal constraints (homogeneity, completeness, rag bag,
cluster-size vs quantity) that Amigo et al. show most extrinsic clustering measures violate.

Definitions, verbatim from the task overview. For a document ``d_i``, let ``C_i`` be the set of
documents in its predicted cluster (including itself) and ``A_i`` the set of documents in the
collection by its true author (including itself)::

    precision(d_i) = |C_i n A_i| / |C_i|        recall(d_i) = |C_i n A_i| / |A_i|

    BCubed precision = mean_i precision(d_i)    BCubed recall = mean_i recall(d_i)

    BCubed F = harmonic mean of the two *averages*

That last line is the one to get right: the F-score is the harmonic mean of the two overall
averages, **not** the average of per-document F-scores. The two differ, and the published PAN
figures are the former.

One contingency table, every number
-----------------------------------
All of it -- BCubed, the pairwise/link view, the per-author table, the exposure numbers -- is a
function of the ``(cluster x author)`` count table ``n_ca`` and the two margins, because

    sum_i |C_i n A_i| / |C_i|  =  sum_c sum_a n_ca^2 / |C_c|

and likewise for recall. So :class:`ClusterContingency` is built once in ``O(n)`` and everything
reads off it. Nothing here materialises a pairwise matrix -- infeasible at corpus scale -- and the
closed forms below are exact.

Noise is expanded, never dropped
--------------------------------
Density-based clusterers label some documents as noise (:data:`NOISE_LABEL`), and a graph-based
one leaves isolated documents unclustered. :func:`expand_noise` gives each such document **its own
singleton cluster**, which is the honest reading -- the attack declined to link it to anything --
and it is applied by default everywhere in this module. Dropping those documents instead would
delete exactly the cases the attack found hardest and inflate every score; a run that leaves 80%
of the corpus as noise would report near-perfect precision on the 20% it was sure about.

The counterpart guard is that **all-singletons is a legal partition with BCubed precision 1.0**,
so precision alone can always be maximised by declining to do anything. Every summary here
therefore carries ``singleton_rate`` and the effective-on-linked scores next to the headline, and
:func:`singleton_baseline` is the reference line any real result has to clear.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

#: The label density-based clusterers use for "not in any cluster". :func:`expand_noise` turns
#: every such document into its own singleton rather than dropping it.
NOISE_LABEL = -1


def expand_noise(labels: np.ndarray, noise_label: int = NOISE_LABEL) -> np.ndarray:
    """Replace each ``noise_label`` entry with a fresh cluster id of its own.

    Returns integer codes, so the result is safe to compare and count regardless of what the
    input labels were. Documents that were already clustered keep their grouping; the ones the
    attack refused to place become singletons, which is what "unlinked" means when the score is a
    measure of how much the attack managed to link together.
    """
    labels = np.asarray(labels).ravel()
    _, codes = np.unique(labels, return_inverse=True)
    codes = codes.ravel().astype(np.int64, copy=True)
    isolated = labels == noise_label
    if isolated.any():
        codes[isolated] = codes.max() + 1 + np.arange(int(isolated.sum()), dtype=np.int64)
    return codes


@dataclass(frozen=True)
class ClusterContingency:
    """The ``(cluster x author)`` count table in sparse triplet form, plus both margins.

    Held as three aligned arrays over the **occupied** cells only (``pair_cluster[j]``,
    ``pair_author[j]``, ``pair_counts[j]``), because the dense ``n_clusters x n_authors`` table is
    infeasible at corpus scale while the number of occupied cells is at most the number of
    documents.

    Attributes
    ----------
    n_documents : int
        Documents scored. Every average in this module is over these.
    cluster_sizes, author_sizes : np.ndarray
        ``|C_c|`` and ``|A_a|``, indexed by the same codes ``pair_cluster`` / ``pair_author`` use.
    pair_cluster, pair_author, pair_counts : np.ndarray
        One entry per occupied cell: which cluster, which author, and ``n_ca``.
    """

    n_documents: int
    cluster_sizes: np.ndarray
    author_sizes: np.ndarray
    pair_cluster: np.ndarray
    pair_author: np.ndarray
    pair_counts: np.ndarray

    @classmethod
    def from_labels(cls, cluster_labels: np.ndarray, author_labels: np.ndarray,
                    expand_noise_labels: bool = True) -> "ClusterContingency":
        """Build the table from one predicted label per document and one true author per document.

        ``expand_noise_labels`` applies :func:`expand_noise` first, which is the default policy for
        this module -- see the module docstring for why unclustered documents are scored as
        singletons rather than dropped.
        """
        cluster_labels = np.asarray(cluster_labels).ravel()
        author_labels = np.asarray(author_labels).ravel()
        if len(cluster_labels) != len(author_labels):
            raise ValueError(f"cluster_labels ({len(cluster_labels)}) and author_labels "
                             f"({len(author_labels)}) must be one per document.")
        if len(cluster_labels) == 0:
            raise ValueError("cannot score an empty collection.")

        cluster_codes = (expand_noise(cluster_labels) if expand_noise_labels
                         else np.unique(cluster_labels, return_inverse=True)[1].ravel())
        _, author_codes = np.unique(author_labels, return_inverse=True)
        author_codes = author_codes.ravel()

        cluster_sizes = np.bincount(cluster_codes)
        author_sizes = np.bincount(author_codes)

        # Occupied cells, found by folding the pair into a single integer key. The fold is exact
        # as long as the product fits an int64, which it does by a wide margin: the worst case is
        # one cluster and one author per document, i.e. n^2, and n is in the tens of thousands.
        folded = cluster_codes.astype(np.int64) * len(author_sizes) + author_codes
        keys, counts = np.unique(folded, return_counts=True)
        return cls(
            n_documents=len(cluster_labels),
            cluster_sizes=cluster_sizes,
            author_sizes=author_sizes,
            pair_cluster=keys // len(author_sizes),
            pair_author=keys % len(author_sizes),
            pair_counts=counts,
        )

    @property
    def n_clusters(self) -> int:
        return len(self.cluster_sizes)

    @property
    def n_authors(self) -> int:
        return len(self.author_sizes)

    @property
    def correct_pairs_including_self(self) -> float:
        """``sum_ca n_ca^2`` -- ordered same-author, same-cluster document pairs, self included.

        The sufficient statistic almost everything here is built from: BCubed's two numerators are
        this quantity weighted by the two margins, and the exposure numbers are it unweighted.
        """
        return float(np.sum(self.pair_counts.astype(np.float64) ** 2))


@dataclass(frozen=True)
class BCubedScores:
    """BCubed precision, recall and F-score for one partition."""

    precision: float
    recall: float
    f_score: float


def _harmonic(precision: float, recall: float) -> float:
    """Harmonic mean, ``0.0`` when either side is zero (rather than a divide-by-zero warning)."""
    total = precision + recall
    return float(2.0 * precision * recall / total) if total > 0 else 0.0


def bcubed_scores(cluster_labels: np.ndarray, author_labels: np.ndarray,
                  expand_noise_labels: bool = True) -> BCubedScores:
    """BCubed precision / recall / F-score, as defined by PAN 2016 (see the module docstring)."""
    return bcubed_from_contingency(
        ClusterContingency.from_labels(cluster_labels, author_labels, expand_noise_labels))


def bcubed_from_contingency(table: ClusterContingency) -> BCubedScores:
    """BCubed from an already-built :class:`ClusterContingency`.

    Uses the closed form ``P = (1/N) sum_ca n_ca^2 / |C_c|``, which is the per-document average
    written as a sum over occupied cells: the ``n_ca`` documents in cell ``(c, a)`` each see the
    same ``n_ca`` correct neighbours out of ``|C_c|``.
    """
    squared = table.pair_counts.astype(np.float64) ** 2
    precision = float(np.sum(squared / table.cluster_sizes[table.pair_cluster]) / table.n_documents)
    recall = float(np.sum(squared / table.author_sizes[table.pair_author]) / table.n_documents)
    return BCubedScores(precision, recall, _harmonic(precision, recall))


def pairwise_scores(table: ClusterContingency) -> dict:
    """Link-level precision / recall / F1: the same partition read as a set of same-author claims.

    A partition asserts a link between every pair of documents it puts in one cluster. This scores
    those assertions directly, which is the view that matches PAN's second subtask and the one to
    quote when the claim is about *links* rather than about documents. It is far harsher than
    BCubed on large clusters -- a cluster of size *m* asserts ``m(m-1)/2`` links, so one over-merge
    costs quadratically -- which is exactly why both are reported.

    All three counts are closed forms over the contingency table; no pair is ever enumerated.
    """
    def n_pairs(sizes: np.ndarray) -> float:
        sizes = sizes.astype(np.float64)
        return float(np.sum(sizes * (sizes - 1.0) / 2.0))

    true_positive = n_pairs(table.pair_counts)
    predicted = n_pairs(table.cluster_sizes)
    actual = n_pairs(table.author_sizes)
    precision = true_positive / predicted if predicted > 0 else 0.0
    recall = true_positive / actual if actual > 0 else 0.0
    return {
        "link_precision": precision,
        "link_recall": recall,
        "link_f1": _harmonic(precision, recall),
        "n_predicted_links": predicted,
        "n_true_links": actual,
        "n_correct_links": true_positive,
    }


def per_author_clustering(table: ClusterContingency,
                          author_values: np.ndarray | None = None) -> pd.DataFrame:
    """One row per author: how much of their traffic the attack managed to reassemble.

    The macro counterpart to the document-weighted headline, and the same reading as the project's
    existing per-user risk curve -- on both corpora a small number of prolific users own a large
    share of the documents, so a document-weighted BCubed says little about a typical person.

    Columns
    -------
    n_documents
        Documents this author wrote in the scored collection.
    largest_cluster_documents, max_cluster_share
        The most of this author's documents that ended up in any single cluster, and that as a
        share of their total. ``max_cluster_share`` is the author's best achievable BCubed recall
        and is the natural "how much of me was reassembled" number.
    any_link
        Whether at least two of their documents were co-clustered. The linkability analogue of
        the identity-level "identified at least once" reading: for a user whose harm is being
        linked at all, this is the event that matters.
    fully_reassembled
        All of their documents in one cluster **and** that cluster containing nobody else --
        complete recovery of the person's traffic with no contamination.
    """
    largest = np.zeros(table.n_authors, dtype=np.int64)
    np.maximum.at(largest, table.pair_author, table.pair_counts)

    # A cell is a complete, uncontaminated recovery when it holds all of the author's documents
    # and all of the cluster's. Both margins are indexed by the cell's own codes.
    complete = ((table.pair_counts == table.author_sizes[table.pair_author])
                & (table.pair_counts == table.cluster_sizes[table.pair_cluster]))
    fully = np.zeros(table.n_authors, dtype=bool)
    fully[table.pair_author[complete]] = True

    frame = pd.DataFrame({
        "n_documents": table.author_sizes,
        "largest_cluster_documents": largest,
        "max_cluster_share": largest / table.author_sizes,
        "any_link": largest >= 2,
        "fully_reassembled": fully,
    })
    if author_values is not None:
        frame.insert(0, "author_id", np.asarray(author_values))
    return frame.sort_values("max_cluster_share", ascending=False, kind="mergesort")


def clustering_summary(cluster_labels: np.ndarray, author_labels: np.ndarray,
                       expand_noise_labels: bool = True) -> dict:
    """Every scalar this module defines, from one pass over the labels.

    The privacy-specific entries, which the standard clustering measures do not cover:

    ``amplification``
        Mean number of **extra** documents by the same author sitting in a document's own cluster.
        This is what converts a clustering result into the identification results the rest of the
        project reports: an adversary who de-anonymises one document by any means walks away with
        this many more of that person's conversations for free. ``0.0`` means the partition adds
        nothing to a single-document attack.
    ``any_link_rate``, ``fully_reassembled_rate``
        Share of *authors* with at least two documents co-clustered, and share whose traffic was
        recovered whole and uncontaminated. Author-weighted, so a handful of prolific users cannot
        carry them.

        **``any_link_rate`` and ``amplification`` are recall-side and must never be quoted bare.**
        Neither looks at what *else* is in the cluster, so both are maximised by merging
        everything: the one-cluster partition attains ``any_link_rate`` exactly -- its ceiling is
        ``linkable_author_rate``, the share of authors with two or more documents -- and drives
        ``amplification`` to its maximum too. They are the right *privacy* quantities, but they
        describe an attack only alongside
        ``bcubed_precision``. The ceilings are therefore returned next to them
        (``any_link_ceiling``, ``amplification_ceiling``) so a reader of one row cannot miss them,
        exactly as ``singleton_rate`` guards the precision side against the mirror-image trick.
    ``singleton_rate``, ``n_singleton_clusters``
        Share of documents left alone. Read every precision figure against this: a partition of
        all singletons scores ``bcubed_precision`` 1.0 while linking nothing, so precision without
        this number beside it is not interpretable.
    ``largest_cluster_share``
        The opposite failure, and the one a headline F-score hides most easily: a good-looking F
        can be a mixture of a giant meaningless blob, a mass of untouched singletons, and
        genuinely good mid-size clusters, describing none of the three. Chaining is single
        linkage's classic failure and it does not announce itself in precision, recall or F.
    ``linked_bcubed_*``
        BCubed recomputed over the documents the attack actually placed with somebody else. This
        is what the attack achieved *when it committed*, and it cannot be gamed by declining to
        cluster -- the complement of the trap above.
    """
    cluster_labels = np.asarray(cluster_labels).ravel()
    author_labels = np.asarray(author_labels).ravel()
    n_noise = int(np.sum(cluster_labels == NOISE_LABEL))

    table = ClusterContingency.from_labels(cluster_labels, author_labels, expand_noise_labels)
    bcubed = bcubed_from_contingency(table)
    authors = per_author_clustering(table)

    in_singleton = table.cluster_sizes[
        expand_noise(cluster_labels) if expand_noise_labels
        else np.unique(cluster_labels, return_inverse=True)[1].ravel()] == 1

    summary = {
        "n_documents": table.n_documents,
        "n_authors": table.n_authors,
        "n_clusters": table.n_clusters,
        "bcubed_precision": bcubed.precision,
        "bcubed_recall": bcubed.recall,
        "bcubed_f": bcubed.f_score,
        **pairwise_scores(table),
        # Exposure: extra same-author documents a single de-anonymisation hands over.
        "amplification": (table.correct_pairs_including_self - table.n_documents) / table.n_documents,
        "any_link_rate": float(authors["any_link"].mean()),
        # What a partition that merges everything would score on the two recall-side measures --
        # carried on every row so neither can be read as an achievement without its ceiling. Both
        # are properties of the collection: an author with one document cannot be linked to
        # themselves, and the most same-author company a document can have is its author's total.
        "any_link_ceiling": float((table.author_sizes >= 2).mean()),
        "amplification_ceiling": float(
            np.sum(table.author_sizes.astype(np.float64) * (table.author_sizes - 1))
            / table.n_documents),
        "fully_reassembled_rate": float(authors["fully_reassembled"].mean()),
        "mean_max_cluster_share": float(authors["max_cluster_share"].mean()),
        "noise_rate": n_noise / table.n_documents,
        "n_singleton_clusters": int(np.sum(table.cluster_sizes == 1)),
        "singleton_rate": float(in_singleton.mean()),
        # The chaining guard, mirroring `singleton_rate` on the other side: what share of the
        # collection ended up in one cluster. 1.0 is the single-cluster partition.
        "largest_cluster_share": float(table.cluster_sizes.max() / table.n_documents),
    }

    # What the attack achieved on the documents it was willing to link. Undefined -- not zero --
    # when it linked nothing, because there is no subset to average over.
    if (~in_singleton).any():
        linked = bcubed_scores(cluster_labels[~in_singleton], author_labels[~in_singleton],
                               expand_noise_labels)
        summary.update({"linked_bcubed_precision": linked.precision,
                        "linked_bcubed_recall": linked.recall,
                        "linked_bcubed_f": linked.f_score})
    else:
        summary.update({"linked_bcubed_precision": float("nan"),
                        "linked_bcubed_recall": float("nan"),
                        "linked_bcubed_f": float("nan")})
    return summary


def link_ranking_metrics(relevant: np.ndarray, n_true_links: float) -> dict:
    """Score a **ranked list of candidate authorship links** -- PAN 2016's second subtask.

    The other half of the task: rather than committing to a partition, rank candidate pairs of
    documents by how confident the attacker is that they share an author. It isolates the quality
    of the *similarity* from the quality of the *clustering algorithm*, which is the distinction
    that decides where effort is worth spending -- a partition can only be as good as the pairwise
    signal underneath it.

    Parameters
    ----------
    relevant : np.ndarray of bool
        Whether each candidate link is a true same-author pair, **in ranked order**, most
        confident first.
    n_true_links : float
        ``sum_a |A_a| choose 2`` over the whole collection -- every true link there is, including
        the ones that never became candidates.

    Notes
    -----
    ``link_ap`` divides by ``n_true_links`` rather than by the number of true links present among
    the candidates, following the task definition. That is the strict reading and the honest one
    here: a candidate set built from a k-nearest-neighbour graph cannot contain every true pair,
    and its misses are real failures of the attack rather than an artifact of the measure.
    ``candidate_link_recall`` reports how much of the ceiling the candidate set kept, so the two
    effects stay separable, and ``link_ap_in_candidates`` scores the ranking alone.

    Average precision does not punish verbosity -- a true link counts wherever it lands -- so
    ``link_r_precision`` and ``link_p_at_10`` are reported next to it for the top of the ranking,
    exactly as the overview paper does.
    """
    relevant = np.asarray(relevant).ravel().astype(bool)
    n_candidates = len(relevant)
    found = int(relevant.sum())
    if n_candidates == 0 or n_true_links <= 0:
        return {"link_ap": 0.0, "link_ap_in_candidates": 0.0, "link_r_precision": 0.0,
                "link_p_at_10": 0.0, "candidate_link_recall": 0.0, "n_candidate_links": 0}

    hits = np.cumsum(relevant)
    precision_at = hits / np.arange(1, n_candidates + 1)
    summed = float(np.sum(precision_at[relevant]))

    r = int(min(n_true_links, n_candidates))
    return {
        "link_ap": summed / n_true_links,
        "link_ap_in_candidates": summed / found if found else 0.0,
        "link_r_precision": float(hits[r - 1]) / n_true_links if r else 0.0,
        "link_p_at_10": float(hits[min(10, n_candidates) - 1]) / min(10, n_candidates),
        "candidate_link_recall": found / n_true_links,
        "n_candidate_links": n_candidates,
    }


def auc_from_histogram(positive_counts: np.ndarray, negative_counts: np.ndarray) -> float:
    """Same-author **verification AUC** from binned scores: P(a true pair outranks a false one).

    The threshold-free, algorithm-free view of the same corpus. BCubed scores a *partition*, so it
    confounds two things -- whether the similarity knows who wrote what, and whether the clustering
    algorithm assembled that knowledge correctly. This measures only the first: given one
    same-author pair and one different-author pair drawn at random, how often is the same-author
    pair scored more similar? It is the standard authorship-*verification* number and it is what
    upper-bounds any method built on the same scores.

    Two properties make it the right companion to BCubed rather than a replacement:

    * **It is prevalence-free**, so it is comparable across corpora with very different
      same-author-pair rates, where every precision-like number is not.
    * **It is prevalence-blind, which is the same fact seen as a hazard.** At low prevalence a
      high AUC is entirely compatible with a useless attack, since stranger pairs still vastly
      outnumber true pairs even after most are ranked below it. So a high AUC is *necessary* for a
      clustering attack to work and nowhere near sufficient, and it must be read next to
      :func:`average_precision_from_histogram` and the prevalence itself.

    Parameters
    ----------
    positive_counts, negative_counts : np.ndarray
        Same-author and different-author pair counts per score bin, in **ascending distance**
        order -- bin 0 is the most similar. Binning is what makes this affordable: the full pair
        count at corpus scale cannot be held or sorted, but streaming pairs into a fixed histogram
        costs one pass and a few kilobytes.

    Notes
    -----
    Ties inside a bin are split evenly, which is the standard convention and makes the result
    exact for a score that is genuinely discrete. For continuous scores the bin width sets the
    resolution; verified against :func:`sklearn.metrics.roc_auc_score` to well below any
    difference being compared here.
    """
    positive_counts = np.asarray(positive_counts, dtype=np.float64)
    negative_counts = np.asarray(negative_counts, dtype=np.float64)
    n_positive, n_negative = positive_counts.sum(), negative_counts.sum()
    if n_positive == 0 or n_negative == 0:
        return float("nan")
    # Negatives strictly more distant than each bin, plus half the bin's own negatives for ties.
    negatives_after = n_negative - np.cumsum(negative_counts)
    return float(np.sum(positive_counts * (negatives_after + 0.5 * negative_counts))
                 / (n_positive * n_negative))


def average_precision_from_histogram(positive_counts: np.ndarray,
                                     negative_counts: np.ndarray) -> float:
    """Average precision over all pairs, from the same histogram -- AUC's imbalance-aware twin.

    Where AUC asks how the two pair populations are *ordered*, this asks what an attacker actually
    gets by walking the ranking from the top: precision averaged over the positions where a true
    link appears. Under the extreme imbalance of a large corpus the two diverge sharply, and this
    is the one that tracks whether a threshold exists that yields usable links. Its chance value is
    the prevalence itself, so report the ratio.

    Within-bin ordering is unknown, so a bin's positives are credited at the precision achieved
    over the whole bin -- the standard treatment of a tied block, and exact in the limit of narrow
    bins.
    """
    positive_counts = np.asarray(positive_counts, dtype=np.float64)
    negative_counts = np.asarray(negative_counts, dtype=np.float64)
    n_positive = positive_counts.sum()
    if n_positive == 0:
        return float("nan")
    retrieved = np.cumsum(positive_counts + negative_counts)
    found = np.cumsum(positive_counts)
    with np.errstate(invalid="ignore", divide="ignore"):
        precision = np.where(retrieved > 0, found / retrieved, 0.0)
    return float(np.sum(precision * positive_counts) / n_positive)


def singleton_baseline(author_labels: np.ndarray) -> BCubedScores:
    """The all-singletons partition, in closed form -- the line every result has to clear.

    Every document alone, so precision is 1.0 by construction and recall is ``mean_i 1/|A_i|``.
    PAN 2016 reports this as BASELINE-Singleton and calls it "very hard to beat", which is true
    only in their regime, where each collection has few documents per author. It is much weaker on
    chat logs, where a user contributes many conversations -- which is precisely what makes these
    corpora a usable testbed for the task.
    """
    author_labels = np.asarray(author_labels).ravel()
    _, sizes = np.unique(author_labels, return_counts=True)
    _, codes = np.unique(author_labels, return_inverse=True)
    recall = float(np.mean(1.0 / sizes[codes.ravel()]))
    return BCubedScores(1.0, recall, _harmonic(1.0, recall))


def single_cluster_baseline(author_labels: np.ndarray) -> BCubedScores:
    """The everything-in-one-cluster partition: recall 1.0, precision ``mean_i |A_i|/N``.

    The opposite degenerate extreme from :func:`singleton_baseline`, and the reason recall is
    never quoted on its own here.
    """
    author_labels = np.asarray(author_labels).ravel()
    _, sizes = np.unique(author_labels, return_counts=True)
    _, codes = np.unique(author_labels, return_inverse=True)
    precision = float(np.mean(sizes[codes.ravel()] / len(author_labels)))
    return BCubedScores(precision, 1.0, _harmonic(precision, 1.0))
