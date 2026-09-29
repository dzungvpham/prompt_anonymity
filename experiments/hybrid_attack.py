#!/usr/bin/env python
"""Do the two attacks help each other? Re-identification and clustering, run against one another.

This project runs two attacks on the same corpus and the same split, and they have never met.
``run_experiment.py`` asks *which enrolled author wrote this document?* and needs labelled history;
``run_clustering.py`` asks *which of these documents share an author?* and needs nothing but the
log. Both read the same test quarter. So each one holds information the other never sees:

* the clustering knows which unknown documents **go together**, which the attribution attack scores
  one document at a time and therefore cannot use;
* the attribution attack knows which unknown documents **look like the same enrolled author**,
  which is a labelled same-author signal the clustering is not allowed to ask for.

This script closes both loops and measures the direction and size of each effect.

Prior work: this combination is well trodden, just not in privacy
-----------------------------------------------------------------
Nothing here is a new idea about *machines*; what is new is measuring it as an attack. The four
closest literatures, and what each predicts:

``clustering -> identification``
    **Set-based / transductive recognition.** In speaker recognition this is diarization feeding
    speaker ID: cluster the segments, pool the embeddings, identify the *cluster* rather than the
    segment. In image retrieval it is **average query expansion** (Chum et al., "Total Recall",
    ICCV 2007): re-issue the query as the mean of its own top results.
    **Their warning is the load-bearing one for us.** Chum et al. found that query expansion
    *without* spatial verification performs **worse than no expansion at all** -- pooling over an
    impure set injects a wrong document's evidence into a right one's decision. Hence the size
    gate below, which is our stand-in for their spatial verification.

``identification -> clustering``
    **Semi-supervised / constrained clustering** (must-link constraints) and **collective entity
    resolution** (Bhattacharya & Getoor, TKDD 2007), where the entities of co-occurring references
    are resolved *jointly* rather than one pair at a time. Two documents the attack independently
    assigns to the same enrolled author, confidently, are a must-link constraint that no
    text-only clustering could have derived. Measured here and removed -- see "Direction B" below;
    the literature is kept because it is what predicted the arm, and what a future attempt would
    have to beat.

``both at once``
    **Generalized Category Discovery** (Vaze et al., CVPR 2022) is our exact problem statement in
    another vocabulary: given a labelled set of known categories and an unlabelled set holding
    *both* known and novel categories, classify the known and discover the novel. Our known side is
    the labelled set, our test quarter is the unlabelled one, and most of its authors are novel.

``the re-ID analogue``
    Person re-identification does both. **k-reciprocal re-ranking** (Zhong et al., CVPR 2017)
    rewrites the query-gallery distance using the gallery's own neighbourhood structure -- the
    clustering-helps-identification direction, done at the level of the distance rather than the
    decision -- and **clustering-based pseudo-labelling** for unsupervised domain adaptation is the
    other direction, where DBSCAN over the unlabelled target domain manufactures the labels a
    classifier is then trained on.

What is measured
----------------
One split, shared by both attacks and identical to the two existing families: the known side is
``[0, 0.75)``, the collection under attack is the final quarter.

**Direction A, ``cluster -> identify``.** Every document's score row is standardised across
authors (the cohort normalisation ``accept_score`` already uses), the rows are averaged within
each cluster, and every member of the cluster takes the pooled argmax. Swept over a **maximum
cluster size**: a cluster larger than the gate is left alone and its documents keep their own
answers. ``gate = 1`` is the attack alone and is asserted to reproduce it exactly. The **oracle**
row pools by the true author partition, which bounds what a perfect clustering could buy and
separates "the idea is wrong" from "the clustering is not good enough yet".

**The reported gain is itself oracle-gated, and has to be quoted that way.** The winning gate is
chosen by reading top-1 off the collection under attack, so it is what a *perfect* chooser of the
gate would have got, not what an attacker would. That is deliberate, and it is the conservative
direction for the conclusion: the honest reading of a small positive number here is "at most this
much, and only if you already knew where to stop". A real attacker would select the gate on the
``[0.50, 0.75)`` tuning slice, the way ``run_clustering.py`` selects everything else -- not
implemented, because a ceiling this low does not justify the machinery.

**Direction B, ``identify -> cluster``, was measured and REMOVED on 2026-08-17.** Do not
re-implement it without reading this. Two arms were built and swept: ``must_link``, which joined
documents the attack confidently assigned to one enrolled author by an extra edge before the
components were taken, over a margin threshold; and ``gallery``, which clustered in the *score*
space instead of the feature space. Both were read against the text-only clustering on the same
edge-budget frontier.

What killed it was not that the gain was small but that it was **redundant**: measured on
WildChat/Gemini/nearest-neighbour, ``must_link`` is worth **+0.0093 BCubed F over a raw-feature
graph, +0.0039 over a contrastive one, and +0.0007 over contrastive+timing** -- it collapses as
the graph improves, because the constraints only ever supplied what a better same-author metric
supplies properly, and supplies once rather than per-threshold. Over the original seven cells it
ranged +0.000 to +0.021 with an inverted U (trusting the top 10-35% of documents helped, trusting
all of them cost -0.081 on swe-chat) and was exactly 0.000 on one WildChat cell. ``gallery``
never had a consistent sign at all: -0.083 to +0.041, positive for the discriminative attacks and
negative for Gemini/nearest-neighbour, which is about as much as one should expect from treating a
distance vector as a learned representation.

The lever both arms were reaching for is the clustering metric itself -- see
``run_clustering.py --projection contrastive``, which delivers an order of magnitude more.

Run (from the repo root)::

    python experiments/hybrid_attack.py --source swe_chat --feature gemini_embedding_2
    sbatch scripts/hybrid_attack_slurm.sh --source wildchat --feature gemini_embedding_2

Outputs, under ``experiments/hybrid/<dataset>_<defense>_<feature>_<attack>[_<projection>][_time<w>]/``:

* ``hybrid_results.csv`` -- one row per arm, with its reference.
* ``pool_sweep.csv`` -- direction A over the size gate, at every cutoff in ``SUMMARY_TOP_KS``.
* ``cmc_results.csv`` -- the **full CMC curve** for the three arms of the results table: top-k at
  every k, at both the document and the identity level, each against its own random baseline.
* ``frontier.csv`` -- the clustering arm's edge-budget frontier, one row per budget.

**Read the curve, not only top-1.** Pooling replaces a document's score row with its cluster's, so
it can move the true author a long way up the ranking without changing the argmax -- a top-1 number
reports that as no effect at all. The identity-level curve is where the sign warning above bites:
pooling makes a cluster's members share one answer, so hits concentrate onto fewer people and that
level can fall while the document level rises. The two levels have different baselines (a prolific
author gives a random guesser one attempt per document) and must be read against their own.

Result files written before 2026-08-17 also carry ``B_identify_helps_cluster`` rows and a
``constraint_sweep.csv``; those numbers are real, they just have no code behind them any more.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from prompt_anonymity.attacks.clustering import build_neighbor_graph  # noqa: E402
from prompt_anonymity.attacks.clustering.projection import (  # noqa: E402
    PROJECTION_FITTERS,
    fit_projection,
)
from prompt_anonymity.attacks.clustering.rescoring import temporal_fusion  # noqa: E402
from prompt_anonymity.evaluation.metrics.clustering import bcubed_scores  # noqa: E402
from prompt_anonymity.evaluation.metrics.ranking import (  # noqa: E402
    cmc_curve,
    true_author_ranks,
)

from run_clustering import PROJECTION_FIT_FRACTION, elapsed_seconds  # noqa: E402
from run_experiment import (  # noqa: E402
    build_attack,
    load_documents_and_features,
    standardize,
)

OUTPUT_ROOT = REPO_ROOT / "experiments" / "hybrid"
DATA_DIR = REPO_ROOT / "data" / "hf"

#: Share of the timeline the attacker holds labelled. Matches ``run_experiment.py``'s ``known0075``
#: and ``run_clustering.py``'s ``KNOWN_FRACTION``, which is the whole point: the two attacks being
#: combined here have to be looking at the same documents for the combination to mean anything.
KNOWN_FRACTION = 0.75

#: Cluster-size gates direction A is swept over. 1 is the attack alone (a cluster of one document
#: pools nothing), and the top of the range is past the largest cluster either corpus produces, so
#: the sweep spans "no pooling" to "pool everything".
POOL_GATES = (1, 2, 3, 5, 8, 12, 20, 35, 60, 100, 200, 500, 2_000, 10_000, 10**9)

#: Neighbours the graphs are built at. Same default as ``run_clustering.MAX_NEIGHBORS``, so the
#: text-only reference arm here is the same graph that family attacks.
MAX_NEIGHBORS = 50

#: Budgets the edge-budget frontier is evaluated at, as a share of the available edges. Geometric,
#: because the interesting region is at the sparse end -- the partition collapses into one giant
#: component long before the budget is exhausted.
BUDGET_STEPS = 40

#: Cutoffs the summary tables carry beyond top-1, so ``pool_sweep.csv`` can be read at k > 1 without
#: going back to the full curve. **1 is deliberately absent**: ``top1_*`` stays the ``argmax`` number
#: that reproduces ``run_experiment.py``'s ``closed_set_top1``, while everything derived from a rank
#: uses the conservative tie convention (see :func:`identification_scores`), and giving the two
#: readings of k=1 adjacent columns in one table would invite reading a tie artefact as an effect.
SUMMARY_TOP_KS = (5, 10, 20)

#: Documents ranked per block in :func:`ranks_of_true_author`. Ranking materialises a
#: ``[block x n_authors]`` slab plus the two boolean masks :func:`true_author_ranks` compares with,
#: so this bounds a large run's memory against the whole score matrix's.
RANK_BLOCK = 1_024


def split_bounds(n_documents: int) -> tuple[slice, slice]:
    """``(known, test)`` -- the labelled history and the collection under attack."""
    cut = int(round(KNOWN_FRACTION * n_documents))
    return slice(0, cut), slice(cut, n_documents)


def row_standardise(scores: np.ndarray) -> np.ndarray:
    """Z-score each document's scores **across authors**, in place where possible.

    The cohort normalisation, and it is what makes pooling meaningful at all. Raw scores are not
    comparable between documents: ``nearest_neighbor`` returns negated distances, so a document
    that happens to sit in a dense region scores higher against *every* author than one in a sparse
    region, and an unnormalised mean over a cluster would be dominated by whichever member is most
    generally similar rather than by whichever is most distinctively so. After this, every row has
    mean 0 and unit spread, so a high value means "unusually like this author *for this document*"
    -- exactly the reading ``accept_score`` already takes.

    Constant rows (an attack that gave one document the same score everywhere) would divide by
    zero; their spread is left at 1, which maps them to all-zeros and lets them contribute nothing
    to a pool rather than contributing a NaN that would poison it.
    """
    scores = np.asarray(scores, dtype=np.float32)
    centre = scores.mean(axis=1, keepdims=True)
    spread = scores.std(axis=1, keepdims=True)
    np.copyto(spread, 1.0, where=(spread == 0))
    # In place, on purpose: the caller owns this array, and `(scores - centre) / spread` would hold
    # extra copies of a matrix that can be large enough to matter under this project's memory cap.
    np.subtract(scores, centre, out=scores)
    np.divide(scores, spread, out=scores)
    return scores


def ranks_of_true_author(matrix: np.ndarray, rows: np.ndarray, true_authors: np.ndarray,
                         authors: np.ndarray) -> np.ndarray:
    """Rank of each document's true author under the score row it is judged by, blocked.

    ``matrix[rows[i]]`` is the row document *i* is ranked in: its **own** row of the score matrix
    for the unpooled arm (``rows`` is then just the document's index), or its **cluster's** pooled
    row for a pooled one. One function therefore serves both arms, and neither ever builds a second
    full ``[n_documents x n_authors]`` matrix, which the gather ``matrix[rows]`` would be.

    Ties are averaged, because :func:`true_author_ranks` averages them: a document tied with one
    other candidate for the best score gets rank 1.5, so it is *not* a top-1 hit under ``rank <= 1``
    while ``argmax`` may still name its author. That is the same conservative reading
    ``run_experiment.py``'s ``cmc_results.csv`` takes against its own ``closed_set_top1``, which is
    why the two families' k=1 numbers are comparable to each other and each differs from its own
    ``argmax`` top-1 by a hair. Pooling makes ties *more* likely, not less -- a pooled row is shared
    by every member of a cluster -- so the convention is worth stating rather than inheriting
    silently.
    """
    ranks = np.empty(len(rows), dtype=np.float64)
    for start in range(0, len(rows), RANK_BLOCK):
        stop = min(start + RANK_BLOCK, len(rows))
        ranks[start:stop] = true_author_ranks(matrix[rows[start:stop]], authors,
                                              true_authors[start:stop])
    return ranks


def pooled_outcomes(scores: np.ndarray, clusters: np.ndarray, gates, evaluated: np.ndarray,
                    true_authors: np.ndarray, authors: np.ndarray,
                    plain: np.ndarray, plain_ranks: np.ndarray) -> dict[int, dict]:
    """Predicted author **and** true-author rank per document, for every gate, in one pass.

    ``gate`` caps the cluster size that is allowed to pool: members of a larger cluster keep their
    own row. That is this script's stand-in for the spatial verification Chum et al. found to be
    the difference between query expansion helping and hurting -- an impure pool does not merely
    fail to help, it overwrites correct answers with the majority's wrong one.

    The per-cluster mean is computed **once** and every gate then only chooses which clusters may
    write their answer back, because the pooled answer for a cluster does not depend on the gate --
    only whether it is used does. The same holds for the *ranking*, which is what makes the full CMC
    curve affordable at all: a document's rank under pooling is fixed by its cluster, so the pooled
    rows are ranked once rather than once per gate.

    One ``np.add.reduceat`` over the cluster-sorted matrix rather than a loop of masks, the same
    contiguous-block trick ``attacks/common.group_by_author`` uses, since the mask form would
    rebuild a large boolean array per cluster.

    ``evaluated`` is the boolean mask of documents that have a correct answer at all -- the in-set
    ones. Predictions are returned for **every** document (the caller masks them, as
    ``run_experiment.py`` does) but ranks only for the evaluated ones, since an out-of-set author
    holds no column and therefore has no rank; ``true_authors`` is likewise the evaluated documents'
    labels alone.

    ``plain`` and ``plain_ranks`` are the unpooled arm -- what every gate falls back to for a
    cluster it will not pool. They are passed in rather than derived here because they are a
    property of the score matrix and not of the partition, so deriving them would repeat a full
    ``argmax`` and a full ranking pass over that matrix for the second partition and again for the
    caller's reference row.
    """
    order = np.argsort(clusters, kind="stable")
    sizes = np.bincount(clusters[order]) if clusters.size else np.array([], dtype=np.int64)
    sizes = sizes[sizes > 0]
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    # Which contiguous block each document landed in, inverted back into document order, so a
    # document can look up its own pooled row without another sort.
    block_of = np.empty(len(clusters), dtype=np.int64)
    block_of[order] = np.repeat(np.arange(len(sizes)), sizes)
    size_of = sizes[block_of]

    pooled = np.add.reduceat(scores[order], starts, axis=0)
    pooled_choice = pooled.argmax(axis=1)
    # `argmax` of the sum is the argmax of the mean -- the positive 1/size is constant within a
    # block -- so the division the mean would need is not performed at all. The same cancellation
    # is what lets the ranks below be taken against the summed row: dividing a row by a positive
    # constant cannot reorder it.
    pooled_ranks = ranks_of_true_author(pooled, block_of[np.flatnonzero(evaluated)],
                                        true_authors, authors)
    del pooled

    results = {}
    for gate in gates:
        pools = (size_of > 1) & (size_of <= gate)
        results[gate] = {
            "predictions": np.where(pools, pooled_choice[block_of], plain),
            "ranks": np.where(pools[evaluated], pooled_ranks, plain_ranks),
        }
    return results


def identification_scores(predicted: np.ndarray, truth_index: np.ndarray, in_set: np.ndarray,
                          authors_of: np.ndarray, ranks: np.ndarray) -> dict:
    """Top-1 and a few deeper cutoffs, all restricted to the in-set documents.

    Out-of-set documents are counted but not scored, exactly as ``run_experiment.py --ood none``
    does: their author is not on the known side, so there is no correct answer for the attack to
    give and including them would measure the split rather than the attack.

    ``ranks`` covers the in-set documents only, in their order, and carries the whole ranking rather
    than the decision -- which is what a top-1 number cannot say. Pooling replaces a document's own
    score row with its cluster's, so it can move the true author from rank 40 to rank 3 without ever
    changing the answer, and the top-1 columns alone would call that nothing. Each cutoff is
    reported at both levels:

    ``top<k>_doc``
        share of in-set **documents** whose true author is within k -- what share of traffic a
        k-long shortlist catches.
    ``top<k>_identity``
        share of in-set **authors** with at least one such document. The attacker only has to
        succeed once, so this is the higher number and the one a privacy claim answers. Note the
        sign warning in the module docstring: pooling makes every member of a cluster share one
        answer, so it *concentrates* hits onto fewer people and this level can fall while
        ``top<k>_doc`` rises.

    ``mrr`` and ``median_rank`` summarise the whole ranking in one number each -- MRR dominated by
    the top of it, the median robust to the tail.

    The ``top1_*`` columns stay ``argmax``-derived while every ``top<k>_*`` above comes from a rank,
    and the two conventions differ under ties (see :func:`ranks_of_true_author`). That is deliberate
    and matches the attribution family: ``top1_doc`` is the number that reproduces
    ``run_experiment.py``'s ``closed_set_top1``, and the conservative k=1 reading is in
    ``cmc_results.csv``.
    """
    hit = (predicted == truth_index) & in_set
    frame = pd.DataFrame({"author": authors_of[in_set], "hit": hit[in_set]})
    per_author = frame.groupby("author")["hit"]
    scores = {
        "n_in_set": int(in_set.sum()),
        "top1_doc": float(hit[in_set].mean()) if in_set.any() else float("nan"),
        "top1_identity": float(per_author.any().mean()) if in_set.any() else float("nan"),
        "top1_macro": float(per_author.mean().mean()) if in_set.any() else float("nan"),
    }
    ranks = np.asarray(ranks, dtype=float)
    # One row per author holding their best document's rank: "was this person linked at least once
    # within k" is exactly "is their minimum rank <= k", so the identity level costs a groupby and
    # not a pass per cutoff.
    best = pd.Series(ranks, index=authors_of[in_set]).groupby(level=0).min().to_numpy()
    for k in SUMMARY_TOP_KS:
        scores[f"top{k}_doc"] = float(np.mean(ranks <= k)) if ranks.size else float("nan")
        scores[f"top{k}_identity"] = float(np.mean(best <= k)) if best.size else float("nan")
    scores["mrr"] = float(np.mean(1.0 / ranks)) if ranks.size else float("nan")
    scores["median_rank"] = float(np.median(ranks)) if ranks.size else float("nan")
    return scores


def cmc_table(ranks: np.ndarray, authors_of: np.ndarray, n_candidates: int) -> pd.DataFrame:
    """The full cumulative match characteristic curve, at **every** k, both levels, with baselines.

    Top-1 says whether the attacker won outright; the curve says how close they were when they did
    not, which is the question a pooled arm exists to move. Two attacks with the same top-1 can put
    the true author at rank 2 or at rank 2,000, and only one of those is a privacy result.

    Columns are ``level``, ``k``, ``accuracy``, ``random``, ``advantage`` and ``n_units``, spelled
    to match ``run_experiment.py``'s ``cmc_results.csv`` so the two families' curves can be read
    side by side (that file has no ``level`` column and is entirely ``document``).

    ``document``
        :func:`prompt_anonymity.evaluation.metrics.ranking.cmc_curve` verbatim, against the uniform
        ``min(k, n) / n`` baseline. The candidate pool is every known author, i.e. every column the
        attack scored, not the authors who happen to appear in the collection under attack -- the
        attacker does not know which of them will show up.
    ``identity``
        an author counts at k if **any one** of their documents ranks within k, the definition
        behind ``run_experiment.py``'s ``id_acc<k>``. Its baseline is correspondingly the chance
        that at least one of an author's ``c`` documents lands, ``1 - (1 - min(k, n)/n)^c``,
        averaged over authors -- far above the document-level one, because a prolific author gives
        a random guesser that many attempts. **Never read the identity curve against the document
        baseline**; that gap is mostly the author's document count.

    Both curves are built from one histogram of the ranks rather than a pass per k, and the identity
    baseline is grouped by distinct document count rather than per author.
    """
    ranks = np.asarray(ranks, dtype=float)
    document = cmc_curve(ranks, n_candidates)
    document.insert(0, "level", "document")
    document["n_units"] = len(ranks)

    per_author = pd.Series(ranks, index=authors_of).groupby(level=0)
    best = per_author.min().to_numpy()
    counts = per_author.size().to_numpy()

    k = np.arange(1, n_candidates + 1)
    # `ceil` before the histogram makes the cumulative count identical to `best <= k` under averaged
    # tie ranks: a rank of 1.5 is a hit from k=2, never from k=1. Truncating instead is the bug
    # `plot_results.py` documents -- it counted a tied-for-first document as a top-1 hit.
    reached = np.bincount(np.ceil(best).astype(np.int64), minlength=n_candidates + 2)
    identity_accuracy = np.cumsum(reached)[1:n_candidates + 1] / len(best) if len(best) else np.full(
        n_candidates, np.nan)

    # `1 - E_a[(1 - p_k)^c_a]`, evaluated once per distinct document count rather than per author.
    miss = 1.0 - k / n_candidates
    sizes, weights = np.unique(counts, return_counts=True)
    identity_random = 1.0 - (miss[:, None] ** sizes[None, :] @ (weights / weights.sum()))

    identity = pd.DataFrame({"level": "identity", "k": k, "accuracy": identity_accuracy,
                             "random": identity_random,
                             "advantage": identity_accuracy - identity_random,
                             "n_units": len(best)})
    return pd.concat([document, identity], ignore_index=True)


def budget_frontier(source: np.ndarray, target: np.ndarray, distance: np.ndarray,
                    n_documents: int, authors: np.ndarray, label: str) -> pd.DataFrame:
    """BCubed at every edge budget for one edge list -- the arm's whole frontier in one pass.

    Threshold-and-connect is single linkage cut at a height, so sweeping *how many of the candidate
    edges are kept* is the same sweep expressed in the quantity that actually decides the
    partition. It also makes representations with incomparable distance scales directly
    comparable, which no shared threshold grid could do. Same argument, and the same construction,
    as ``improve_clustering.py``'s frontier.

    Selection-optimistic on purpose: direction A is handed the budget that maximises BCubed F on
    the collection under attack, which is the *best case* for the partition it pools over, so a
    small gain measured against it is the strong form of a small gain.
    """
    finite = np.isfinite(distance)
    source, target, distance = source[finite], target[finite], distance[finite]
    order = np.argsort(distance, kind="stable")
    budgets = np.unique(np.geomspace(1, max(len(order), 1), BUDGET_STEPS).astype(np.int64))
    rows = []
    for budget in budgets:
        kept = order[:budget]
        edge_source, edge_target = source[kept], target[kept]
        adjacency = csr_matrix((np.ones(len(edge_source), dtype=np.int8),
                                (edge_source, edge_target)),
                               shape=(n_documents, n_documents))
        labels = connected_components(adjacency, directed=False)[1]
        scores = bcubed_scores(labels, authors)
        rows.append({"arm": label, "edge_budget": int(budget),
                     "bcubed_f": scores.f_score, "bcubed_precision": scores.precision,
                     "bcubed_recall": scores.recall,
                     "n_clusters": int(len(np.unique(labels))),
                     "largest_cluster_share": float(np.bincount(labels).max() / n_documents)})
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the re-identification and clustering attacks against each other.")
    parser.add_argument("--source", default="wildchat")
    parser.add_argument("--feature", default="gemini_embedding_2")
    parser.add_argument("--defense", default="base")
    parser.add_argument("--attack", default="nearest_neighbor",
                        help="Attribution attack supplying the score matrix.")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--metric", default="cosine")
    parser.add_argument("--max-neighbors", type=int, default=MAX_NEIGHBORS)
    # The clustering arm's own knobs, spelled and defaulted exactly as `run_clustering.py` spells
    # them, so a hybrid cell can be pointed at the partition that family actually publishes rather
    # than only at its weakest one. They touch the GRAPH ONLY -- the attribution arm keeps the raw
    # standardised features, because a projection fitted on labelled history is a different attack
    # and folding it in would stop `attack_alone` reproducing that family's headline.
    parser.add_argument("--projection", default="none",
                        choices=sorted(PROJECTION_FITTERS) + ["none"],
                        help="Metric-learning projection fitted on the first half of the timeline "
                             "and used for the clustering arm's space. Appended to the output "
                             "directory name.")
    parser.add_argument("--projection-components", type=int, default=1024)
    parser.add_argument("--projection-shrinkage", type=float, default=0.1)
    parser.add_argument("--projection-steps", type=int, default=30_000)
    parser.add_argument("--projection-temperature", type=float, default=0.05)
    parser.add_argument("--projection-batch", type=int, default=1024)
    parser.add_argument("--projection-hard-negatives", type=int, default=0)
    parser.add_argument("--time-weight", type=float, default=0.0,
                        help="Weight on the elapsed-time term fused into every edge score. 0 is "
                             "the pure-text graph. Appended to the output directory name.")
    # Defaults are `run_experiment.py`'s, deliberately and to the letter, so that this script's
    # `attack_alone` row *is* that family's number for the same cell rather than something near
    # it -- which is what lets every delta below be read against the published headline. The one
    # documented departure is tuning: that family tunes by default and this does not (see the
    # module docstring), so a tunable attack is run at its configured hyper-parameters.
    parser.add_argument("--standardize", action=argparse.BooleanOptionalAction, default=True,
                        help="Z-score the features on known-side statistics before the attack "
                             "(default: on, as in run_experiment.py).")
    parser.add_argument("--regularization", type=float, default=1.0)
    parser.add_argument("--balanced", action="store_true")
    parser.add_argument("--shrinkage", type=float, default=0.1)
    parser.add_argument("--linkage", default="mean")
    parser.add_argument("--jobs", type=int, default=-1)
    return parser.parse_args()


def projection_kwargs(args) -> dict:
    """Only the hyper-parameters the chosen projection takes; see ``run_clustering.py``'s twin.

    The fitters share one flag namespace but not one signature, so passing all of them would be a
    ``TypeError`` on every fitter but ``contrastive``.
    """
    if args.projection == "wccn":
        return {"shrinkage": args.projection_shrinkage}
    if args.projection == "lda":
        return {"n_components": args.projection_components,
                "shrinkage": args.projection_shrinkage}
    if args.projection == "contrastive":
        return {"n_components": args.projection_components, "steps": args.projection_steps,
                "temperature": args.projection_temperature,
                "authors_per_batch": args.projection_batch,
                "hard_negatives": args.projection_hard_negatives}
    return {}


def main() -> None:
    args = parse_args()
    tag = f"{args.source}_{args.defense}_{args.feature}_{args.attack}"
    # The graph's space is part of what the run claims, so it goes in the directory name -- the
    # same contract `run_clustering.py` applies, and the reason a projected run cannot silently
    # overwrite the raw-graph one it is meant to be compared against.
    if args.projection != "none":
        tag = f"{tag}_{args.projection}"
    if args.time_weight > 0:
        tag = f"{tag}_time{args.time_weight:g}"
    output_dir = args.output_dir or (OUTPUT_ROOT / tag)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(tag)

    frame, embeddings = load_documents_and_features(
        args.data_dir, args.source, args.feature,
        defense="none" if args.defense == "base" else args.defense)
    embeddings = np.nan_to_num(embeddings)
    known, test = split_bounds(len(frame))

    # **Each attack gets the feature space its own runner gives it, and they differ.**
    # `run_experiment.py` z-scores on known-side statistics by default; `run_clustering.py` does
    # not, it only fills NaNs. Forcing one space on both would make every delta below partly a
    # preprocessing effect, so the attribution side is standardised, the graph is built on the raw
    # vectors, and both arms are then exactly the arms their families publish (asserted against
    # them: the `attack_alone` row reproduces that family's `closed_set_top1` to the digit).
    attack_known, attack_test = embeddings[known], embeddings[test]
    if args.standardize:
        attack_known, attack_test = standardize(attack_known, attack_test)
    known_labels = frame["author_id"].to_numpy()[known]
    test_labels = frame["author_id"].to_numpy()[test]
    known_authors = np.unique(known_labels)
    in_set = np.isin(test_labels, known_authors)
    print(f"  known: {len(known_labels):,} documents, {len(known_authors):,} authors")
    print(f"  test:  {len(test_labels):,} documents, {len(np.unique(test_labels)):,} authors, "
          f"{in_set.sum():,} in-set ({in_set.mean():.1%})")

    # --- the two attacks, each exactly as its own runner would run it -------
    started = time.perf_counter()
    fitted = build_attack(args.attack, args)().fit(attack_known, known_labels)
    scores = row_standardise(fitted.score(attack_test))
    print(f"  {args.attack}: scored {scores.shape[0]:,} x {scores.shape[1]:,} in "
          f"{time.perf_counter() - started:.0f}s")
    # Where each document's true author sits in the score matrix's columns; -1 when out of set, a
    # value `argmax` can never return, so an out-of-set document is never accidentally correct.
    truth_index = np.searchsorted(fitted.authors, test_labels)
    truth_index = np.where(in_set, truth_index, -1)

    # The clustering arm's space. Raw features by default, as `run_clustering.py` uses them; with
    # `--projection`, the fit window is that file's PROJECTION_FIT_FRACTION and is imported rather
    # than restated, because the two have to agree. It is the FIRST HALF and not the whole known
    # side for a reason that still binds here even though this script tunes nothing: the partition
    # is then the one that family publishes, not a stronger cousin fitted on more labels.
    graph_space = embeddings
    if args.projection != "none":
        started = time.perf_counter()
        fit_window = slice(0, int(round(PROJECTION_FIT_FRACTION * len(frame))))
        projection = fit_projection(args.projection, embeddings[fit_window],
                                    frame["author_id"].to_numpy()[fit_window],
                                    **projection_kwargs(args))
        graph_space = projection.transform(embeddings)
        print(f"  projection {projection.name}: fitted on {fit_window.stop:,} documents, "
              f"{graph_space.shape[1]} dimensions, {time.perf_counter() - started:.0f}s")

    started = time.perf_counter()
    k = min(args.max_neighbors, len(test_labels) - 1)
    graph = build_neighbor_graph(graph_space[test], k, metric=args.metric)
    if args.time_weight > 0:
        # Fused after the graph is built, on the same collection the graph covers -- the elapsed
        # time is a property of the edge, not of the space, so it composes with any --projection.
        graph = temporal_fusion(graph, elapsed_seconds(frame)[test], args.time_weight)
    text_source, text_target, text_distance = graph.edges(k)
    print(f"  text graph: k={k}, {len(text_distance):,} edges, "
          f"{time.perf_counter() - started:.0f}s")
    del embeddings, graph_space, attack_known, attack_test

    rows = []

    # --- direction A: does knowing what goes together improve who wrote it? ---
    # The clustering arm is the one `run_clustering.py` ships: single-linkage components over the
    # text graph. Its partition is taken at the budget that maximises BCubed F on this collection,
    # which is an oracle choice and is labelled as one -- it is the *best case* for direction A, so
    # a negative result under it is the strong form of the negative result.
    text_frontier = budget_frontier(text_source, text_target, text_distance,
                                    len(test_labels), test_labels, "text")
    best = text_frontier.loc[text_frontier["bcubed_f"].idxmax()]
    finite = np.isfinite(text_distance)
    order = np.argsort(text_distance[finite], kind="stable")[:int(best["edge_budget"])]
    adjacency = csr_matrix(
        (np.ones(len(order), dtype=np.int8),
         (text_source[finite][order], text_target[finite][order])),
        shape=(len(test_labels),) * 2)
    clusters = connected_components(adjacency, directed=False)[1]
    print(f"  clustering: BCubed F={best['bcubed_f']:.4f} at budget {int(best['edge_budget']):,}, "
          f"{int(best['n_clusters']):,} clusters, largest "
          f"{best['largest_cluster_share']:.1%} of the collection")

    _, truth_clusters = np.unique(test_labels, return_inverse=True)
    in_set_labels = test_labels[in_set]
    # The unpooled arm, computed once here rather than inside each partition's sweep: it is what
    # every gate falls back to, it is the `attack_alone` reference row, and neither the argmax nor
    # the ranking depends on the partition -- so deriving it per partition would cost extra passes
    # over a large matrix for no reason.
    plain_choice = scores.argmax(axis=1)
    plain_ranks = ranks_of_true_author(scores, np.flatnonzero(in_set), in_set_labels,
                                       fitted.authors)
    pool_rows, outcomes = [], {}
    for name, partition in (("attack_clusters", clusters), ("oracle_partition", truth_clusters)):
        outcomes[name] = pooled_outcomes(scores, partition, POOL_GATES, in_set, in_set_labels,
                                         fitted.authors, plain_choice, plain_ranks)
        for gate, outcome in outcomes[name].items():
            measured = identification_scores(outcome["predictions"], truth_index, in_set,
                                             test_labels, outcome["ranks"])
            pool_rows.append({"partition": name, "gate": gate, **measured})
    pool_sweep = pd.DataFrame(pool_rows)

    baseline = pool_sweep[(pool_sweep["partition"] == "attack_clusters")
                          & (pool_sweep["gate"] == 1)].iloc[0]
    # gate=1 pools nothing, so it has to *be* the attack alone. Asserted rather than trusted: this
    # is the reference every number in this direction is read against, and a silent difference
    # would move every one of them.
    plain = identification_scores(plain_choice, truth_index, in_set, test_labels, plain_ranks)
    assert abs(plain["top1_doc"] - baseline["top1_doc"]) < 1e-12, "gate=1 is not the attack alone"
    rows.append({"direction": "A_cluster_helps_identify", "arm": "attack_alone",
                 "setting": "", **plain})
    # The full curve for the three arms the results table reports, so an effect that moves the
    # ranking without moving the answer is visible. The two pooled arms are drawn at the gate that
    # maximises **top-1**, which is the gate `hybrid_results.csv` reports and is oracle-chosen (see
    # the module docstring) -- so this is the curve of the arm quoted there, not the best curve
    # available at some other k. `pool_sweep.csv` carries top-5/10/20 at every gate for that.
    summary_columns = ["n_in_set", "top1_doc", "top1_identity", "top1_macro", "mrr", "median_rank"]
    summary_columns += [f"top{k}_{level}" for k in SUMMARY_TOP_KS
                        for level in ("doc", "identity")]
    curves = [cmc_table(plain_ranks, in_set_labels, len(fitted.authors)).assign(
        arm="attack_alone", setting="")]
    for name in ("attack_clusters", "oracle_partition"):
        subset = pool_sweep[pool_sweep["partition"] == name]
        winner = subset.loc[subset["top1_doc"].idxmax()]
        gate = int(winner["gate"])
        rows.append({"direction": "A_cluster_helps_identify",
                     "arm": f"pool_{name}", "setting": f"gate={gate}",
                     **{key: winner[key] for key in summary_columns}})
        curves.append(cmc_table(outcomes[name][gate]["ranks"], in_set_labels,
                                len(fitted.authors)).assign(arm=f"pool_{name}",
                                                            setting=f"gate={gate}"))
        print(f"  pool[{name}]: best top-1 {winner['top1_doc']:.4f} at gate {gate} "
              f"(attack alone {plain['top1_doc']:.4f}, "
              f"{winner['top1_doc'] - plain['top1_doc']:+.4f}); "
              + ", ".join(f"top-{k} {winner[f'top{k}_doc']:.4f} "
                          f"({winner[f'top{k}_doc'] - plain[f'top{k}_doc']:+.4f})"
                          for k in SUMMARY_TOP_KS))

    # The clustering arm's own score, carried so a cell records the partition direction A was
    # given rather than only what pooling it bought. It is the frontier's best, i.e. an oracle
    # budget, for the reason the frontier's docstring gives.
    rows.append({"direction": "clustering_alone", "arm": "text_only", "setting": "",
                 **{key: float(best[key]) for key in
                    ("bcubed_f", "bcubed_precision", "bcubed_recall")}})

    results = pd.DataFrame(rows)
    cmc = pd.concat(curves, ignore_index=True)[
        ["arm", "setting", "level", "k", "accuracy", "random", "advantage", "n_units"]]
    for column, value in (("attack", args.attack), ("feature", args.feature),
                          ("defense", args.defense), ("dataset", args.source)):
        results.insert(0, column, value)
        cmc.insert(0, column, value)
    results.to_csv(output_dir / "hybrid_results.csv", index=False)
    pool_sweep.to_csv(output_dir / "pool_sweep.csv", index=False)
    text_frontier.to_csv(output_dir / "frontier.csv", index=False)
    cmc.to_csv(output_dir / "cmc_results.csv", index=False)
    print(f"\nWrote {output_dir}/hybrid_results.csv, pool_sweep.csv, frontier.csv, "
          "cmc_results.csv")


if __name__ == "__main__":
    main()
