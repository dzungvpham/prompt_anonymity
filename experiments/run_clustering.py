#!/usr/bin/env python
"""Author-clustering attacks: group an anonymised log by author, naming nobody.

The linkability counterpart to ``run_experiment.py``. That script asks *which enrolled author
wrote this document?* and needs the attacker to hold labelled documents by the target; this one
asks *which of these documents share an author?* and needs nothing but the log. The measures are
BCubed and its companions (:mod:`prompt_anonymity.evaluation.metrics.clustering`), following PAN
2016's author-clustering task.

The split, and what the known side is for
-----------------------------------------
Documents are ordered by ``ended_at``. The **last 25% is the collection under attack**; the first
75% is the attacker's own labelled history. Same cut ``run_experiment.py`` makes under
``known0075``, reached through the same loader, so the two families cover the same documents.

The known side is used for **hyper-parameter selection only** -- never to enrol an author, and
never read at attack time. Specifically, tuning runs on the *last quarter of the known side*
(positions 50-75%), not on all of it, because every parameter here is an **absolute** quantity
whose optimum is set by the *shape* of the problem instance (``min_cluster_size`` is a document
count; ``distance_threshold`` is a radius whose meaning depends on how crowded the space is). The
whole known side is a different shape from the test quarter, so a threshold tuned there transfers
as an under-linking one; the final quarter of the known side is the same size, adjacent in time,
and the closest available match.

Two author scopes (``--scopes``), and why their scores must not be differenced
------------------------------------------------------------------------------
The two slices hold disjoint *documents*, but not disjoint *authors*: some test-quarter authors
also wrote on the known side. Nothing enrols them -- no author is ever a label here -- but the
hyper-parameters were chosen with their documents visible, and ``--projection`` is *fitted* on
them, so a run that scores only the whole collection cannot say whether its result depends on
people the attacker held labels for. So the collection is attacked twice, and both scopes land in
``clustering_results.csv`` under a ``scope`` column:

``all``      the test quarter whole. The threat model, and the number every figure draws.
``unseen``   only the documents whose author never appears in the known side ``[0, 0.75)``. The
             tuning slice is restricted the same way against the history that precedes *it*
             (``[0, 0.50)``), so it stays a simulation of the instance being attacked.

Each scope is a complete run -- its own neighbour graphs, its own search, its own baselines -- and
that is the point: **one scope's BCubed F may not be compared with the other's**. Restricting to
unseen authors changes the *shape* of the problem. Sharper still, the restriction is not
independent of the answer: the corpus keeps only authors with at least two documents in total, so
an author absent from the known side must have **at least two in the test quarter**. The
``unseen`` collection therefore contains no single-document authors at all *by construction*, and
singletons are the documents no method can link -- that alone lifts the singleton baseline. Read
each scope against **its own** baseline rows, which is why they are recomputed per scope rather
than shared.

``unseen`` is small on swe-chat and should be read as an indication rather than a measurement
there.

Did tuning help? Two references, neither of them the tuned number itself
-----------------------------------------------------------------------
``untuned_bcubed_f`` is the class defaults -- what somebody gets by not thinking about it -- and is
weak evidence alone, since the defaults are a choice made in this repo and bad ones would flatter
tuning for free. ``--oracle-sweep`` re-runs the whole grid on the **test** collection, giving the
median (what an arbitrary reasonable configuration scores) and the best (what a perfect chooser
would have reached). Both are **oracles**: nothing may select a configuration on them.

``--diagnostics``: measuring the problem rather than the attack
---------------------------------------------------------------
Everything above scores a *partition*, so it confounds two things: whether the features know who
wrote what, and whether the algorithm assembled that knowledge correctly. ``--diagnostics`` adds
the algorithm-free half, which needs no clustering run at all and is the cheap first look at a new
corpus or feature:

* **Same-author verification AUC** over every pair, streamed into a fixed histogram. Prevalence-free,
  hence the only figure here comparable *across* corpora -- and prevalence-blind, hence optimistic,
  so it is reported next to average precision and the prevalence itself.
* **Neighbour-graph quality per k** -- edge precision, hit rate and neighbour recall against their
  own chance references, plus the connected-component structure. This is what explains a tuned
  low neighbour count: at a high k the graph becomes a single component, so a loose method chains
  the whole corpus into one cluster.
* **A within-language control** on edge precision, since a strong within-corpus language separator
  can make a result that merely matches a language partition look like evidence about writing style.
* **Authorship-link ranking** (PAN's second subtask) -- AP, R-precision and P@10 over the graph's
  edges, which separates the quality of the similarity from the quality of the algorithm.

The reference partitions (baselines) have one implementation (:func:`run_baselines`) and this file
has one definition of the split, so the diagnostics and the attack always agree on both.

Run (from the repo root)::

    python experiments/run_clustering.py --source swe_chat --feature gemini_embedding_2
    python experiments/run_clustering.py --source wildchat --defense styleremix \
        --algorithms leiden hdbscan
    python experiments/run_clustering.py --source wildchat --diagnostics --algorithms  # no attack
    python experiments/run_clustering.py --source wildchat --scopes unseen   # only the strangers
    sbatch scripts/run_clustering_slurm.sh --source wildchat --defense base --oracle-sweep

Outputs, under ``experiments/clustering/<dataset>_<defense>_<feature>/``. The ``all`` scope keeps
the file names it has always had and every other scope appends its own, so a directory written
before ``--scopes`` existed is still read the same way:

* ``clustering_results.csv`` -- one row per (scope, algorithm), plus one per (scope, baseline):
  every BCubed, link-level, exposure and shape measure, the chosen hyper-parameters, and how the
  tuning transferred (``tuning_optimism``, ``untuned_bcubed_f``, ``tuning_gain_over_default``).
* ``clusters_<algorithm>[_<scope>].csv`` -- ``doc_id`` -> cluster label for the collection under
  attack, with ``author_in_known`` beside it, so any downstream slice can be re-scored without
  re-running the attack.
* ``author_report_<algorithm>[_<scope>].csv`` -- per author: how much of their traffic was
  reassembled.
* ``tuning_trials.csv`` -- every configuration tried on the simulation slice, with its score.
* ``oracle_sweep.csv`` (``--oracle-sweep``) -- the same grid scored on the test collection.
* ``diagnostics.csv`` / ``graph_diagnostics.csv`` (``--diagnostics``) -- one row per slice, and one
  row per (slice, k), for the algorithm-free measures above. A scope's slices are named
  ``test_<scope>`` / ``tuning_<scope>``.

The directory is **not** under ``experiments/results/``, whose four-part names are a contract
``plot_results.py`` parses and whose ``rolling_results.csv`` schema is built for ranking attacks.
That file does read *this* directory though, drawing the clustering figures into
``experiments/plots/<dataset>/clustering/``, so there is still exactly one script that draws every
figure in the project. This one writes CSVs and stops, like every other runner.
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components
from scipy.special import gammaln

# Expected, not a problem: the neighbour graph is deliberately sparse, so it has several connected
# components and scikit-learn joins them to finish the tree. Left unfiltered it prints once per
# configuration and buries the actual results.
warnings.filterwarnings("ignore", message=".*number of connected components.*")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from prompt_anonymity.attacks.clustering import (  # noqa: E402
    CLUSTERING_ATTACKS,
    CLUSTERING_SPACES,
    CLUSTERING_SPACES_QUANTILE,
    BaselineClustering,
    MAX_DENSE_DOCUMENTS,
    build_neighbor_graph,
    get_clustering_attack,
    parameter_grid,
)
from prompt_anonymity.attacks.clustering.projection import (  # noqa: E402
    PROJECTION_FITTERS,
    fit_projection,
)
from prompt_anonymity.attacks.clustering.rescoring import (  # noqa: E402
    GRAPH_RESCORINGS,
    rescore_graph,
    balanced_time_weight,
    temporal_fusion,
    winsor_bounds,
)
from prompt_anonymity.attacks.similarity.kernel import blocked_distances  # noqa: E402
from prompt_anonymity.evaluation.metrics.clustering import (  # noqa: E402
    auc_from_histogram,
    average_precision_from_histogram,
    bcubed_scores,
    clustering_summary,
    link_ranking_metrics,
    per_author_clustering,
    ClusterContingency,
    single_cluster_baseline,
    singleton_baseline,
)

from run_experiment import load_documents_and_features, standardize  # noqa: E402

OUTPUT_ROOT = REPO_ROOT / "experiments" / "clustering"
DATA_DIR = REPO_ROOT / "data" / "hf"

#: Share of the timeline the attacker holds as labelled history; the rest -- the final quarter --
#: is the anonymous collection to be clustered. Matches ``run_experiment.py``'s ``known0075``, so
#: the two attack families cover exactly the same documents.
KNOWN_FRACTION = 0.75

#: Share of the timeline ``--projection`` is fitted on. **Deliberately 0.50 and not
#: ``KNOWN_FRACTION``**, even though the attacker holds labels out to 0.75. The slice from 0.50 to
#: 0.75 is what :func:`tune` selects hyper-parameters on, and a projection fitted through it would
#: have seen those documents' authors -- so the tuning slice would stop being a simulation of the
#: attack and start being a fit on the training set, silently inflating every
#: ``tuning_bcubed_f`` and every ``tuning_optimism``. Holding the fit to the first half keeps the
#: two stages disjoint and matches exactly what ``experiments/improve_clustering.py`` measured. It
#: costs the projection a quarter of the available labels, which is the right trade: a projection
#: is a low-dimensional object fitted on tens of thousands of documents, where the hyper-parameter
#: selection has one labelled slice and nothing to spare.
PROJECTION_FIT_FRACTION = 0.50

#: Neighbour counts the graph diagnostics report at. The graph is built once at the largest and
#: truncated for the rest, since neighbours are stored in distance order.
NEIGHBOR_COUNTS = (1, 2, 3, 5, 10, 20, 50, 100)

#: Bins the pairwise-score histogram is accumulated into. Cosine distance is bounded on [0, 2], so
#: this gives a fine resolution while keeping an all-pairs AUC/AP cheap in memory (a fixed-size
#: histogram rather than one entry per pair).
SCORE_BINS = 200_000

#: Cosine distance's exact upper bound. Any other metric has its range estimated from a sample.
COSINE_MAX_DISTANCE = 2.0

#: Seed for the sampled range estimate on non-cosine metrics, and for anything else drawn here.
DIAGNOSTIC_SEED = 20260812

#: Largest ``k`` any configuration may ask for. The graph is built once at this width and every
#: configuration truncates it, so a sweep over ``neighbors`` costs one build rather than one per
#: value -- and the build is by far the expensive part.
MAX_NEIGHBORS = 50

#: Metadata partitions scored alongside the real attacks. Whatever these reach is available to an
#: attacker who reads no text at all.
METADATA_BASELINES = ("language_primary", "model_owner")

#: Author populations the collection is attacked over, in the order they are run and written. See
#: the module docstring: ``all`` is the threat model and ``unseen`` is the validity check, they are
#: separate runs of everything, and their scores are not comparable with each other.
TEST_AUTHOR_SCOPES = ("all", "unseen")

#: Below this many documents a scope is skipped with a note rather than attacked. A collection of a
#: handful of documents is not a clustering problem, and on a small corpus an author restriction is
#: exactly the thing that produces one.
MIN_SCOPE_DOCUMENTS = 10

#: Multipliers laid around the *balanced* weight to make ``--tune-time-weight``'s grid.
#:
#: The grid is derived per run rather than fixed, because the weight where the two terms of
#: ``temporal_fusion`` contribute equally is a property of the corpus, not a constant:
#: :func:`~prompt_anonymity.attacks.clustering.rescoring.balanced_time_weight` reads it off the
#: tuning graph, and this brackets it geometrically.
#:
#: **0.0 is always in the grid** -- the pure-text control, so an algorithm that gains nothing from
#: session structure can decline the fusion rather than being forced into it. The top is clipped
#: below 1.0 because a weight of exactly 1 is timing *only*: it reads no text at all, and a search
#: allowed to select it could report a text attack that is not one.
TIME_WEIGHT_MULTIPLIERS = (0.25, 0.5, 1.0, 2.0, 4.0)

#: Largest weight the derived grid may contain; see above for why it is not 1.0.
MAX_TIME_WEIGHT = 0.8


def time_weight_grid(balance: float) -> tuple[float, ...]:
    """The candidate weights for one run: 0, then :data:`TIME_WEIGHT_MULTIPLIERS` around ``balance``."""
    weights = {0.0} | {min(round(balance * m, 4), MAX_TIME_WEIGHT)
                       for m in TIME_WEIGHT_MULTIPLIERS}
    return tuple(sorted(weights))


def describe_bounds(bounds: tuple[float, float]) -> str:
    """One term's winsorising bracket, for the run log and the results row."""
    return f"[{bounds[0]:.4g}, {bounds[1]:.4g}]"


def slice_bounds(n_documents: int) -> tuple[slice, slice]:
    """``(tuning, test)`` -- the simulation slice and the collection under attack.

    Both are a quarter of the corpus. The tuning slice is the quarter immediately before the test
    set, i.e. the tail of the attacker's own labelled history; see the module docstring for why it
    is that rather than the whole known side.
    """
    cut = int(round(KNOWN_FRACTION * n_documents))
    return slice(int(round(0.50 * n_documents)), cut), slice(cut, n_documents)


def scope_suffix(scope: str) -> str:
    """Filename and slice-name suffix for one scope; empty for ``all``.

    So the default scope's files keep the names they have always had, and every result directory
    written before ``--scopes`` existed stays readable by exactly the code that read it before.
    """
    return "" if scope == "all" else f"_{scope}"


def variant_name(args) -> str:
    """The fourth part of the output directory name: which *representation* was clustered.

    Four positional parts, ``<dataset>_<defense>_<feature>_<variant>``, deliberately the shape
    ``experiments/results/`` has used all along -- the variant occupies the slot an attribution
    run spells with its attack, because it is the same kind of thing: the method, as opposed to
    the data it was pointed at.

    The variant is the composition of the three axes that change what an edge score *means* --
    ``--projection``, ``--rescoring`` and the elapsed-time fusion -- in pipeline order, or
    ``plain`` when none of them is set. ``--standardize`` and ``--known-defense`` are **not** part
    of it and stay trailing suffixes: they qualify a run rather than name a method, and a name
    whose fourth part could be any of a dozen combinations is not a positional slot.

    A tuned weight spells itself ``time``, with no number: under ``--tune-time-weight`` the value
    is an *outcome* recorded per row of ``clustering_results.csv``, and each algorithm may have
    chosen a different one, so there is no single number the directory could honestly carry.
    """
    parts = []
    if args.projection != "none":
        parts.append(args.projection)
    if args.rescoring != "none":
        parts.append(args.rescoring)
    if args.tune_time_weight:
        parts.append("time")
    elif args.time_weight > 0:
        parts.append(f"time{args.time_weight:g}")
    return "_".join(parts) or "plain"


def scope_positions(authors: np.ndarray, scope: str) -> tuple[np.ndarray, np.ndarray]:
    """``(tuning, test)`` document positions for one author scope.

    ``all`` is the two slices whole. ``unseen`` keeps only the documents whose author the attacker
    holds no labelled history for **at that point on the timeline**: the test collection drops
    every author who appears anywhere in the known side ``[0, 0.75)``, and the tuning slice drops
    every author who appears in the history that precedes it, ``[0, 0.50)``.

    Mirroring the restriction rather than applying the test collection's author set to both is what
    keeps the tuning slice a *simulation* -- and here the alternative is not merely worse, it is
    empty: ``[0.50, 0.75)`` is itself inside ``[0, 0.75)``, so every author in the tuning slice
    appears in the known side by definition and restricting it against that interval would leave no
    documents at all. The history that precedes the slice is the only reference that means for it
    what the known side means for the test collection.

    The mirror is not exact, and the asymmetry is worth knowing: the test collection is at the end
    of the corpus, so "absent from everything before it" also means "at least two documents inside
    it" (the corpus keeps no author with fewer than two documents overall). The tuning slice has a
    future its authors can write in, so it keeps a few single-document authors that the test
    collection cannot -- the two instances still match closely in practice.
    """
    tuning_window, test_window = slice_bounds(len(authors))
    tuning = np.arange(tuning_window.start, tuning_window.stop)
    test = np.arange(test_window.start, test_window.stop)
    if scope == "all":
        return tuning, test
    if scope != "unseen":
        raise SystemExit(f"unknown scope {scope!r}; available: {list(TEST_AUTHOR_SCOPES)}")
    known = np.unique(authors[:test_window.start])          # everything the attacker holds labels for
    history = np.unique(authors[:tuning_window.start])       # ... as of the tuning slice
    return (tuning[~np.isin(authors[tuning], history)],
            test[~np.isin(authors[test], known)])


def take_rows(matrix: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """``matrix[positions]``, as a **view** when the positions are one contiguous run.

    Fancy indexing always copies, and the ``all`` scope's positions are a whole slice -- so the
    plain spelling would copy a large matrix to hand back something already in memory, on a run
    whose peak is what decides whether it survives the job's memory cap.
    """
    if len(positions) and positions[-1] - positions[0] == len(positions) - 1:
        return matrix[positions[0]:positions[-1] + 1]
    return matrix[positions]


def score_partition(labels: np.ndarray, authors: np.ndarray) -> dict:
    """Every measure for one partition."""
    return clustering_summary(labels, authors)


# --- diagnostics (--diagnostics): measure the problem, not the attack --------

def collection_shape(authors: np.ndarray) -> dict:
    """Properties of the clustering problem itself, before any attack.

    ``authors_per_document`` is the one that governs the rest: it decides how strong the singleton
    baseline is, and this project's corpora sit in a much sparser regime than PAN's own collections,
    which is why the singleton baseline is far lower here.
    """
    values, sizes = np.unique(authors, return_counts=True)
    per_document = sizes[np.unique(authors, return_inverse=True)[1].ravel()]
    n = len(authors)
    n_true_links = float(np.sum(sizes.astype(np.float64) * (sizes - 1) / 2))
    return {
        "n_documents": n,
        "n_authors": len(values),
        "authors_per_document": len(values) / n,
        "mean_docs_per_author": float(sizes.mean()),
        "median_docs_per_author": float(np.median(sizes)),
        "max_docs_per_author": int(sizes.max()),
        "share_authors_linkable": float((sizes >= 2).mean()),
        "share_docs_linkable": float((per_document >= 2).mean()),
        "n_true_links": n_true_links,
        # Probability that two documents drawn at random share an author. Every edge-precision and
        # link-ranking figure is read against this -- it varies a great deal across corpora, so an
        # unreferenced "edge precision 0.15" says nothing at all on its own.
        "random_link_precision": n_true_links / (n * (n - 1) / 2),
    }


def random_neighbor_references(authors: np.ndarray, k: int) -> dict:
    """What ``k`` neighbours drawn at random would score. Closed form, no sampling.

    Without these, ``hit_at_k`` cannot be compared across corpora: the same k is a much larger
    fraction of a small collection than a large one, so the same raw value means different things.
    """
    n = len(authors)
    _, codes = np.unique(np.asarray(authors), return_inverse=True)
    others = np.bincount(codes.ravel())[codes.ravel()] - 1
    pool = n - 1
    # P(no same-author document among k draws) = C(pool - others, k) / C(pool, k), through
    # log-gammas because the binomials overflow long before they cancel at these sizes.
    log_miss = np.where(
        pool - others >= k,
        gammaln(pool - others + 1) - gammaln(np.maximum(pool - others - k, 0) + 1)
        - gammaln(pool + 1) + gammaln(pool - k + 1),
        -np.inf)
    return {"random_hit_at_k": float(np.mean(1.0 - np.exp(log_miss))),
            "random_neighbor_recall": k / pool}


def within_language_chance(authors: np.ndarray, languages: np.ndarray) -> float:
    """P(same author | same language) for a random pair -- the language-controlled reference.

    The plain chance reference asks how often two documents from the whole collection share an
    author, which flatters any similarity that is partly a language detector (an English-only
    pipeline run over every document is exactly that). So the honest question is not "is an edge better than a random pair?" but "is a
    within-language edge better than a random *within-language* pair?".
    """
    frame = pd.DataFrame({"author": np.asarray(authors), "language": np.asarray(languages)})
    same_author = total = 0.0
    for _, group in frame.groupby("language", dropna=False, observed=True):
        sizes = group["author"].value_counts().to_numpy(dtype=np.float64)
        same_author += float(np.sum(sizes * (sizes - 1) / 2))
        total += len(group) * (len(group) - 1) / 2
    return same_author / total if total > 0 else float("nan")


def graph_diagnostics(graph, authors: np.ndarray,
                      languages: np.ndarray | None = None) -> pd.DataFrame:
    """Per-``k`` measures of how much authorship the neighbour graph carries.

    Three easily-confused quantities. ``nn_precision`` is the share of edges joining two documents
    by one person -- what a community-detection method sees, and if it is low the dense regions of
    the graph are topics rather than people. ``hit_at_k`` is the share of documents with at least
    one same-author neighbour, an upper bound on what any edge-joining method can link at all.
    ``neighbor_recall`` is the share of an author's *other* documents reachable in one hop.

    Also the connected-component structure, which decides whether a threshold-and-connect method
    is viable: once a giant component forms, transitive closure merges most of the corpus.
    """
    rows = []
    labels = np.asarray(authors)
    _, author_codes = np.unique(labels, return_inverse=True)
    author_codes = author_codes.ravel()
    author_sizes = np.bincount(author_codes)
    others = author_sizes[author_codes] - 1
    has_partner = others > 0
    n = len(labels)
    chance_edge = float(np.sum(author_sizes.astype(np.float64) * (author_sizes - 1) / 2)
                        / (n * (n - 1) / 2))

    language_codes = language_chance = None
    if languages is not None:
        _, language_codes = np.unique(np.asarray(languages).astype(str), return_inverse=True)
        language_codes = language_codes.ravel()
        language_chance = within_language_chance(labels, np.asarray(languages).astype(str))

    for k in NEIGHBOR_COUNTS:
        if k > graph.k:
            continue
        view = graph.truncate(k)
        valid = np.isfinite(view.distances)
        same = (author_codes[view.indices] == author_codes[:, None]) & valid

        language_columns = {}
        if language_codes is not None:
            same_language = (language_codes[view.indices] == language_codes[:, None]) & valid
            within = same_language.sum()
            language_columns = {
                "nn_same_language": float(within / valid.sum()),
                "nn_precision_within_language": float((same & same_language).sum() / within)
                                                if within else float("nan"),
                "random_nn_precision_within_language": language_chance,
            }

        matched = same.sum(axis=1)
        component_count, assignment = connected_components(view.to_sparse(), directed=False,
                                                           return_labels=True)
        component_sizes = np.bincount(assignment)
        rows.append({
            "k": k,
            "nn_precision": float(same.sum() / valid.sum()),
            "random_nn_precision": chance_edge,
            **language_columns,
            "hit_at_k": float((matched > 0).mean()),
            "neighbor_recall": float(np.mean(matched[has_partner] / others[has_partner])),
            **random_neighbor_references(labels, k),
            "mean_distance": float(view.distances[valid].mean()),
            "n_components": int(component_count),
            "largest_component_share": float(component_sizes.max() / view.n_documents),
            "n_singleton_components": int(np.sum(component_sizes == 1)),
        })
    return pd.DataFrame(rows)


def link_ranking(graph, authors: np.ndarray, n_true_links: float, k: int) -> dict:
    """PAN's authorship-link ranking over the graph's edges, ranked by similarity.

    ``link_ap`` divides by *every* true link in the collection, including ones the candidate set
    never contained, so a narrow candidate set is penalised for what it missed;
    ``candidate_link_recall`` reports that loss separately.
    """
    source, target, distance = graph.edges(k)
    _, author_codes = np.unique(np.asarray(authors), return_inverse=True)
    author_codes = author_codes.ravel()
    order = np.argsort(distance, kind="stable")            # most similar first
    relevant = author_codes[source[order]] == author_codes[target[order]]
    return {**link_ranking_metrics(relevant, n_true_links), "link_ranking_k": k}


def verification_diagnostics(embeddings: np.ndarray, authors: np.ndarray, metric: str,
                             bins: int = SCORE_BINS, working_memory_mb: int = 256) -> dict:
    """Same-author verification AUC over **every** pair, plus its per-document and imbalance twins.

    The algorithm-free reading: no partition built on these scores can recover a link the scores
    themselves rank below the strangers.

    ``verification_auc_macro`` is a correction rather than a refinement. Same-author pairs grow
    quadratically in an author's document count, so a single prolific author's documents can
    contribute a large share of every true link in the collection, making the pair-weighted AUC
    substantially a statement about a handful of people rather than about the population.

    One streamed pass over the full pairwise matrix, never materialised. The macro half sorts each
    row, which is the more expensive part. **This is a second pass, after the neighbour graph's**;
    fusing them would halve the distance cost and is the obvious optimisation if this ever runs on
    something larger. Only ``metric="cosine"`` has an exact bound to bin against; any other metric
    has its range sampled, and over-range distances fall in the last bin.
    """
    embeddings = np.asarray(embeddings, dtype=np.float32)
    n = len(embeddings)
    _, codes = np.unique(np.asarray(authors), return_inverse=True)
    codes = codes.ravel()

    if metric == "cosine":
        upper = COSINE_MAX_DISTANCE
    else:
        sample = np.random.default_rng(DIAGNOSTIC_SEED).choice(n, size=min(n, 512), replace=False)
        upper = float(max(block.max() for _, block in blocked_distances(
            embeddings[sample], embeddings, metric=metric))) * 1.05
    scale = (bins - 1) / upper

    positive_counts = np.zeros(bins, dtype=np.int64)
    negative_counts = np.zeros(bins, dtype=np.int64)
    macro_auc_total, macro_documents = 0.0, 0

    for start, block in blocked_distances(embeddings, embeddings, metric=metric,
                                          working_memory_mb=working_memory_mb):
        rows = np.arange(len(block))
        block[rows, start + rows] = np.inf                 # a document is not its own pair
        same = codes[start:start + len(block), None] == codes[None, :]
        same[rows, start + rows] = False

        # Each row of a block spans the whole collection, so a document's ranking is complete
        # within one block and needs no cross-block accumulation.
        ordered = np.sort(block, axis=1)
        for row in rows:
            positives = block[row][same[row]]
            n_positive = len(positives)
            n_negative = n - 1 - n_positive
            if n_positive == 0 or n_negative == 0:
                continue
            positives = np.sort(positives)
            closer = (np.searchsorted(ordered[row], positives, "left")
                      - np.searchsorted(positives, positives, "left"))
            up_to = (np.searchsorted(ordered[row], positives, "right")
                     - np.searchsorted(positives, positives, "right"))
            macro_auc_total += float(np.sum((n_negative - up_to) + 0.5 * (up_to - closer))
                                     / (n_positive * n_negative))
            macro_documents += 1
        del ordered

        # Clip before the cast: the diagonal is +inf and casting that to an integer is undefined --
        # it lands in the *nearest* bin on this platform, counting every document as its own
        # most-similar pair.
        np.multiply(block, scale, out=block)
        np.clip(block, 0, bins - 1, out=block)
        binned = block.astype(np.int32)
        # Negatives as "everything minus the positives" rather than a ``~same`` mask, which would
        # copy out ~67 million non-pairs per block where ``ravel`` is a view.
        totals = np.bincount(binned.ravel(), minlength=bins)
        positives = np.bincount(binned[same], minlength=bins)
        positive_counts += positives
        negative_counts += totals - positives
        negative_counts[bins - 1] -= len(block)            # take the diagonal back out exactly
        del binned, totals, positives, same

    # The stream visits both (i, j) and (j, i), so every bin holds twice its unordered count. AUC,
    # AP and prevalence are ratios in which the factor cancels exactly -- verified against sklearn
    # on the whole pairwise matrix -- so only the reported pair *count* is halved, to mean the same
    # thing as ``n_true_links``.
    n_positive, n_negative = positive_counts.sum(), negative_counts.sum()
    return {
        "verification_auc": auc_from_histogram(positive_counts, negative_counts),
        "verification_auc_macro": (macro_auc_total / macro_documents
                                   if macro_documents else float("nan")),
        "verification_ap": average_precision_from_histogram(positive_counts, negative_counts),
        "verification_prevalence": float(n_positive / (n_positive + n_negative)),
        "verification_pairs": int((n_positive + n_negative) // 2),
        "verification_documents_scored": macro_documents,
        "verification_bins": bins,
    }


def slice_diagnostics(name: str, frame: pd.DataFrame, embeddings: np.ndarray, graph,
                      metric: str) -> tuple[dict, pd.DataFrame]:
    """Every algorithm-free measure for one slice: its shape, its graph, its pairwise separability."""
    authors = frame["author_id"].to_numpy()
    shape = collection_shape(authors)
    per_k = graph_diagnostics(graph, authors, frame.get("language_primary"))
    row = {"slice": name, "metric": metric, **shape,
           **verification_diagnostics(embeddings, authors, metric),
           **link_ranking(graph, authors, shape["n_true_links"], graph.k)}
    per_k.insert(0, "slice", name)
    return row, per_k


def tune(algorithm: str, graph, authors: np.ndarray, limit: int | None = None,
         spaces: dict | None = None) -> tuple[dict, pd.DataFrame]:
    """Choose hyper-parameters by maximising BCubed F on a labelled slice.

    Exhaustive over :data:`CLUSTERING_SPACES`, because these grids are small (24-48 points) and the
    graph they share is already built -- so the search costs one clustering per point and nothing
    else. Returns the best settings and the full trials table, which is written out: a search that
    only reports its winner cannot be checked for a flat optimum, and a flat optimum is exactly
    what would make the transfer to the test set safe.
    """
    spaces = spaces or CLUSTERING_SPACES
    grid = parameter_grid(spaces[algorithm])
    if limit is not None:
        grid = grid[:limit]
    factory = get_clustering_attack(algorithm)
    trials = []
    for settings in grid:
        started = time.perf_counter()
        try:
            labels = factory(**settings).cluster(graph)
            scores = bcubed_scores(labels, authors)
            row = {**settings, "bcubed_f": scores.f_score, "bcubed_precision": scores.precision,
                   "bcubed_recall": scores.recall,
                   "n_clusters": int(len(np.unique(labels)))}
        except Exception as error:                      # a configuration that cannot run is data
            row = {**settings, "bcubed_f": float("nan"), "error": str(error)[:200]}
        trials.append({**row, "seconds": time.perf_counter() - started})
    table = pd.DataFrame(trials).sort_values("bcubed_f", ascending=False, kind="mergesort")
    if table["bcubed_f"].isna().all():
        raise SystemExit(f"every {algorithm} configuration failed; see the trials table.")
    # Restore each value's original Python type from the *space*, not from the retrieved dtype.
    # A single failed trial puts a NaN in the frame, which promotes that whole column to float64 --
    # so an integer parameter like `neighbors` comes back as 25.0 and fails as a slice index. The
    # grid is the authority on what type each parameter is; the frame is only how it travelled.
    winner = table.iloc[0]
    best = {}
    for key, values in spaces[algorithm].items():
        template = values[0]
        if isinstance(template, bool):
            best[key] = bool(winner[key])
        elif isinstance(template, (int, np.integer)):
            best[key] = int(winner[key])
        elif isinstance(template, (float, np.floating)):
            best[key] = float(winner[key])
        else:
            best[key] = str(winner[key])
    return best, table


@dataclass
class Collection:
    """One slice of the timeline under one author scope, with the graph built over it.

    The unit everything below works on. Splitting the run by scope means every stage -- the graph,
    the search, the baselines, the diagnostics -- has to be told *which* documents it is looking
    at rather than reading a module-level slice, and this is that argument.
    """

    scope: str                      #: one of :data:`TEST_AUTHOR_SCOPES`
    role: str                       #: ``"test"`` (under attack) or ``"tuning"`` (the simulation)
    positions: np.ndarray           #: row indices into the whole ordered corpus
    frame: pd.DataFrame             #: the documents themselves, re-indexed from 0
    graph: object = None            #: the neighbour graph over them, once built
    seconds: np.ndarray | None = None   #: ``ended_at`` for these documents, when the weight is
    #: being tuned -- the graph is then left UNFUSED and :func:`run_algorithm` fuses a copy of it
    #: per candidate weight, which is the only way a weight can be searched over.
    fusion_scales: tuple[float, float] | None = None   #: ``(d0, tau)``, read off the TUNING graph
    #: and shared by both collections: the weight is selected against one scoring function and has
    #: to be deployed with the same one, so these are fixed on the known side rather than
    #: recomputed per graph (see ``rescoring.fusion_scales``).
    weight_grid: tuple[float, ...] = ()   #: candidate weights, derived from the tuning graph's
    #: balance point by :func:`time_weight_grid`.

    @property
    def authors(self) -> np.ndarray:
        return self.frame["author_id"].to_numpy()

    @property
    def name(self) -> str:
        """Slice name as it appears in ``diagnostics.csv``: ``test``, ``tuning_unseen``, ..."""
        return f"{self.role}{scope_suffix(self.scope)}"


def prepare_scope(scope: str, frame: pd.DataFrame, embeddings: np.ndarray,
                  seconds: np.ndarray | None, args,
                  known_embeddings: np.ndarray | None = None
                  ) -> tuple[Collection, Collection] | None:
    """Both collections for one scope, graphs built, time-fused and rescored -- or ``None``.

    ``None`` means the scope is too small to attack (:data:`MIN_SCOPE_DOCUMENTS`), which is a note
    and not an error: the other scope's results are still worth having, and an author restriction
    emptying a slice is itself a fact about the corpus.

    The two graphs are built at one ``k``, the smaller of what either collection can supply, so a
    configuration selected on the simulation slice is expressible on the collection under attack.
    That makes ``k`` a property of the scope, and a scope whose collections are much smaller is
    searched over a correspondingly narrower ``neighbors`` grid -- printed, because it is a
    difference between the two runs that nothing else records.
    """
    tuning_positions, test_positions = scope_positions(frame["author_id"].to_numpy(), scope)
    label = f"scope={scope}"
    if min(len(tuning_positions), len(test_positions)) < MIN_SCOPE_DOCUMENTS:
        print(f"  {label}: SKIPPED -- {len(test_positions):,} documents under attack and "
              f"{len(tuning_positions):,} to tune on, under the {MIN_SCOPE_DOCUMENTS}-document "
              f"floor.")
        return None

    collections = tuple(
        Collection(scope, role, positions,
                   frame.iloc[positions].reset_index(drop=True))
        for role, positions in (("test", test_positions), ("tuning", tuning_positions)))

    shape = collection_shape(collections[0].authors)
    print(f"  {label}: test {shape['n_documents']:,} documents, {shape['n_authors']:,} authors "
          f"(r={shape['authors_per_document']:.3f}); tuning {len(tuning_positions):,} documents, "
          f"{len(np.unique(collections[1].authors)):,} authors")
    print(f"  {'':<{len(label)}}  baselines: "
          f"singleton F={singleton_baseline(collections[0].authors).f_score:.3f}, "
          f"one-cluster F={single_cluster_baseline(collections[0].authors).f_score:.3f}")

    k = min(args.max_neighbors, len(test_positions) - 1, len(tuning_positions) - 1)
    started = time.perf_counter()
    # Each collection is built from ITS OWN matrix: the tuning slice is the attacker's history
    # (undefended by default) and the test slice is the collection under attack. The two are the
    # same array unless --known-defense differs, so this is a no-op in the matched condition.
    sources = {"test": embeddings,
               "tuning": embeddings if known_embeddings is None else known_embeddings}
    for collection in collections:
        collection.graph = build_neighbor_graph(
            take_rows(sources[collection.role], collection.positions), k, metric=args.metric)
    print(f"  {'':<{len(label)}}  built two k={k} graphs in {time.perf_counter() - started:.0f}s")

    # The edge score is `temporal_fusion` at EVERY weight, including zero -- a run with no timing
    # is the same formula with the time term multiplied by nothing, not a different scoring
    # function. So the scales are derived and the transform applied unconditionally, and a `plain`
    # run differs from a `time` one only in `w`.
    if seconds is not None:
        for collection in collections:
            collection.seconds = seconds[collection.positions]
    # Both scales come from the TUNING collection and are then applied to both graphs. That
    # asymmetry is the point: the weight is chosen on the tuning slice, so the transform it
    # was chosen under has to be the transform the test collection is scored with.
    tuning = next(c for c in collections if c.role == "tuning")
    # Per graph, deliberately: each collection is bracketed by its OWN quantiles rather than
    # inheriting the tuning slice's, for the reasons `winsor_bounds` documents. The selected weight
    # is unaffected either way, because the search only ever sees the tuning graph.
    for collection in collections:
        collection.fusion_scales = winsor_bounds(collection.graph, collection.seconds)
    if seconds is None:
        # tau is NaN here and is never read: the time term is skipped outright at w=0.
        for collection in collections:
            collection.graph = temporal_fusion(collection.graph, None, 0.0,
                                               collection.fusion_scales)
        shown = next(c for c in collections if c.role == "test").fusion_scales
        print(f"  {'':<{len(label)}}  text-only edge scores, winsorised to "
              f"{describe_bounds(shown[0])} (fusion at weight 0)")
    else:
        balance = balanced_time_weight(tuning.graph, tuning.seconds, tuning.fusion_scales)
        for collection in collections:
            collection.weight_grid = time_weight_grid(balance)
        for collection in collections:
            print(f"  {'':<{len(label)}}  {collection.role} winsorised to text "
                  f"{describe_bounds(collection.fusion_scales[0])}, time "
                  f"{describe_bounds(collection.fusion_scales[1])} h"
                  + (f" (balanced weight {balance:.3f})" if collection.role == "tuning" else ""))
        if args.tune_time_weight:
            # Left unfused on purpose: the weight is a hyper-parameter here, so fusing once now
            # would fix the very thing the search is about. `run_algorithm` fuses a copy of each
            # graph per candidate weight instead -- w=0 included, so the pure-text candidate is
            # scored in the same space as every other one.
            print(f"  {'':<{len(label)}}  elapsed time will be fused per candidate weight "
                  f"({', '.join(f'{w:g}' for w in collections[0].weight_grid)})")
        else:
            for collection in collections:
                collection.graph = temporal_fusion(collection.graph, collection.seconds,
                                                   args.time_weight, collection.fusion_scales)
            print(f"  {'':<{len(label)}}  fused elapsed time at weight {args.time_weight:g} into "
                  f"both graphs")

    if args.rescoring != "none":
        # Applied to BOTH graphs, so the tuning slice simulates exactly what the test collection
        # will be attacked with. Rescoring after the build is free: it re-ranks the candidates the
        # build already found (see the rescoring module on why that is an approximation and a
        # one-sided one).
        for collection in collections:
            collection.graph = rescore_graph(collection.graph, args.rescoring,
                                             args.rescoring_locality)
        print(f"  {'':<{len(label)}}  rescored both graphs with {args.rescoring}"
              f"(locality={args.rescoring_locality}) -> metric {collections[0].graph.metric!r}")
    return collections


def elapsed_seconds(frame: pd.DataFrame) -> np.ndarray:
    """``ended_at`` as POSIX seconds, NaN where the document carries no timestamp."""
    stamps = pd.to_datetime(frame["ended_at"], errors="coerce", utc=True)
    # Subtracting the epoch and asking the timedelta for seconds, rather than `.astype("int64")`
    # over a hardcoded 1e9. That divisor assumes the parsed dtype is nanoseconds, and pandas 3
    # parses an ISO string to `datetime64[us]` -- which silently made every gap 1000x too small
    # and, because `fusion_scales` derives tau from the same array, invisible in `t / tau`.
    # `total_seconds()` already yields NaN for NaT, so there is no second pass to write the
    # missing entries -- and pandas hands back a read-only view, so `.copy()` keeps the array
    # writable for callers that fill or mask it in place.
    return (stamps - pd.Timestamp(0, tz="UTC")).dt.total_seconds().to_numpy().copy()


def tune_over_time_weights(algorithm: str, tuning: Collection, tuning_authors: np.ndarray,
                           args) -> tuple[dict, float, pd.DataFrame]:
    """Search the elapsed-time weight jointly with ``algorithm``'s own hyper-parameters.

    ``(settings, weight, trials)``. One :func:`tune` pass per candidate in
    the run's derived weight grid, over a copy of the tuning graph fused at that weight (with the
    scales the tuning graph fixed), and the winner
    is the best cell of the whole product -- so the weight is chosen exactly as every other
    hyper-parameter is, **on the labelled tuning slice [0.50, 0.75) and never on the collection
    under attack**.

    Per algorithm rather than once per run, for the same reason ``neighbors`` is: what the fusion
    is worth depends on how the method reads the graph, and connected components at a tight radius
    is not asking the same question of an edge that Leiden's modularity is. The cost is one
    existing search per candidate weight, and the trials table keeps every cell so a flat or
    bimodal optimum in the weight is visible rather than inferred from the winner.
    """
    best_settings, best_weight, best_score, tables = None, 0.0, -np.inf, []
    for weight in tuning.weight_grid:
        graph = temporal_fusion(tuning.graph, tuning.seconds, weight, tuning.fusion_scales)
        settings, trials = tune(algorithm, graph, tuning_authors, args.tune_limit,
                                search_spaces(args, algorithm))
        trials.insert(0, "time_weight", weight)
        tables.append(trials)
        score = float(trials.iloc[0]["bcubed_f"])
        if score > best_score:
            best_settings, best_weight, best_score = settings, weight, score
    combined = pd.concat(tables, ignore_index=True).sort_values(
        "bcubed_f", ascending=False, kind="mergesort")
    return best_settings, best_weight, combined


def run_algorithm(algorithm: str, test: Collection, test_authors: np.ndarray,
                  tuning: Collection, tuning_authors: np.ndarray, args) -> tuple[dict, np.ndarray, pd.DataFrame]:
    """Tune on the simulation slice, then attack the test collection once with the winner.

    Takes the two :class:`Collection` objects rather than their graphs because
    ``--tune-time-weight`` makes the graph itself part of what is searched: the weight decides the
    edge scores, so the tuning graph has to be re-fused per candidate and the test graph fused
    once, at the end, with whatever won.
    """
    started = time.perf_counter()
    test_graph, time_weight = test.graph, args.time_weight
    if args.tune_time_weight:
        settings, time_weight, trials = tune_over_time_weights(
            algorithm, tuning, tuning_authors, args)
        # The collection under attack is fused ONCE, with the weight the tuning slice chose. It is
        # never scored at any other weight, which is what keeps the choice honest.
        test_graph = temporal_fusion(test.graph, test.seconds, time_weight, test.fusion_scales)
    else:
        settings, trials = tune(algorithm, tuning.graph, tuning_authors, args.tune_limit,
                                search_spaces(args, algorithm))
    tuning_seconds = time.perf_counter() - started
    tuned_score = float(trials.iloc[0]["bcubed_f"])
    # Named before the attack, not after: the failure path below returns this same table, and a
    # trials frame missing the column its rows are keyed by concatenates into NaNs.
    trials.insert(0, "algorithm", algorithm)

    # The winner is re-fitted on a *different* graph, so a configuration that ran on the
    # simulation slice can still fail on the collection under attack. That is worth one algorithm,
    # not the whole cell: the others' results stand, and the trials table records what happened.
    started = time.perf_counter()
    try:
        labels = get_clustering_attack(algorithm)(**settings).cluster(test_graph)
    except Exception as error:
        print(f"  {algorithm:<28s} FAILED on the test collection with "
              f"[{settings}]: {type(error).__name__}: {error}")
        return None, None, trials
    attack_seconds = time.perf_counter() - started

    row = {
        "algorithm": algorithm,
        "hyperparameters": ", ".join(
            f"{key}={value:.4g}" if isinstance(value, float) else f"{key}={value}"
            for key, value in sorted(settings.items())),
        **score_partition(labels, test_authors),
        # Per row, because --tune-time-weight lets each algorithm choose its own; it repeats the
        # run-level `time_weight` column when the weight is fixed instead.
        "time_weight": time_weight,
        # Without these the run cannot be reconstructed from its own outputs: a weight only means
        # something alongside the brackets the two terms were winsorised to. These are the TEST
        # collection's own (see `winsor_bounds` on why they are per-graph); the tuning slice's are
        # in the run log.
        "fusion_text_bounds": ("" if test.fusion_scales is None
                               else describe_bounds(test.fusion_scales[0])),
        "fusion_time_bounds_hours": ("" if test.fusion_scales is None
                                     else describe_bounds(test.fusion_scales[1])),
        "tuning_bcubed_f": tuned_score,
        # Named for its sign: POSITIVE means the tuning slice scored HIGHER than the test
        # collection, i.e. tuning where the labels are visible was optimistic.
        "tuning_optimism": float("nan"),                     # filled below, needs the test score
        "seconds_tuning": tuning_seconds,
        "seconds_attack": attack_seconds,
    }
    row["tuning_optimism"] = row["tuning_bcubed_f"] - row["bcubed_f"]

    # Did tuning actually buy anything? Two references, because "untuned" has no canonical meaning
    # for these methods and the obvious one is gameable:
    #
    #   `untuned_bcubed_f`  - the class defaults, i.e. what someone gets by not thinking about it.
    #                         Weak as evidence on its own: the defaults are a choice made in this
    #                         repo, and picking bad ones would flatter tuning for free.
    #   `--oracle-sweep`    - the whole grid re-run on the test collection, which gives the median
    #                         (what a configuration drawn at random from a sane range scores) and
    #                         the best (what a perfect chooser would have got). Neither depends on
    #                         a default anyone picked, and the median is the honest reference.
    try:
        default_labels = get_clustering_attack(algorithm)().cluster(test_graph)
        row["untuned_bcubed_f"] = bcubed_scores(default_labels, test_authors).f_score
    except Exception:
        row["untuned_bcubed_f"] = float("nan")       # defaults may not run on this graph
    row["tuning_gain_over_default"] = row["bcubed_f"] - row["untuned_bcubed_f"]
    return row, labels, trials


def oracle_sweep(algorithm: str, graph, authors: np.ndarray, limit: int | None,
                 spaces: dict | None = None) -> pd.DataFrame:
    """Every configuration in the grid, scored on the **test** collection.

    An oracle and labelled as one: nothing may select a configuration on these numbers. It exists
    to bound the tuning question from both sides -- ``median`` is what an arbitrary reasonable
    configuration scores, ``max`` is what a perfect chooser would have reached, and the tuned
    configuration's own score sits somewhere between. Without it, "tuning helped" is a claim about
    one point with nothing to compare it to.
    """
    grid = parameter_grid((spaces or CLUSTERING_SPACES)[algorithm])
    if limit is not None:
        grid = grid[:limit]
    factory = get_clustering_attack(algorithm)
    rows = []
    for settings in grid:
        try:
            scores = bcubed_scores(factory(**settings).cluster(graph), authors)
            rows.append({**settings, "bcubed_f": scores.f_score})
        except Exception as error:
            rows.append({**settings, "bcubed_f": float("nan"), "error": str(error)[:200]})
    table = pd.DataFrame(rows)
    table.insert(0, "algorithm", algorithm)
    return table


def run_baselines(collection: Collection) -> list[dict]:
    """The reference partitions for one collection, scored through the same path as the attacks.

    **Recomputed per scope, never shared.** A baseline is a property of the collection it is
    computed on, and the two scopes are different collections: restricting to authors the attacker
    has no history for also removes every author with a single document (see the module
    docstring), which moves the singleton partition's score on its own. Sharing one set of
    baseline rows would put the ``unseen`` attack's F against a line drawn for a different problem.
    """
    frame, authors, graph = collection.frame, collection.authors, collection.graph
    rows = []
    for kind in ("singleton", "single_cluster"):
        labels = BaselineClustering(kind=kind).cluster(graph)
        rows.append({"algorithm": f"baseline_{kind}", "hyperparameters": "",
                     **score_partition(labels, authors)})
    # Matched-random: the true cluster-size distribution with the association destroyed. Averaged
    # over repetitions the way PAN does, seeded so a redraw cannot move the number.
    replicates = [score_partition(
        BaselineClustering(kind="random", metadata=authors, seed=seed).cluster(graph), authors)
        for seed in range(50)]
    rows.append({"algorithm": "baseline_random", "hyperparameters": "50 seeded replicates",
                 **{key: float(np.mean([r[key] for r in replicates]))
                    for key in replicates[0] if isinstance(replicates[0][key], (int, float))}})
    for column in METADATA_BASELINES:
        if column in frame.columns:
            labels = BaselineClustering(kind="metadata",
                                        metadata=frame[column].to_numpy()).cluster(graph)
            rows.append({"algorithm": f"baseline_{column}", "hyperparameters": "",
                         **score_partition(labels, authors)})
    return [{"scope": collection.scope, **row} for row in rows]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cluster an anonymised log by author.")
    parser.add_argument("--source", default="wildchat")
    parser.add_argument("--feature", default="gemini_embedding_2")
    parser.add_argument("--defense", default="base")
    parser.add_argument("--known-defense", default="base",
                        help="Defense applied to the KNOWN side -- the labelled history the "
                             "attacker holds, which here is never enrolled but IS what "
                             "hyper-parameters are tuned on and what --projection and "
                             "--standardize are fitted on (default: base, i.e. undefended). The "
                             "default is the deployment threat model: a user adopts a defense "
                             "today, so whatever leaked earlier is original text. Set it equal to "
                             "--defense for the matched condition, where the defense has always "
                             "been on. A non-default value is appended to the output directory.")
    parser.add_argument("--algorithms", nargs="*", default=sorted(CLUSTERING_ATTACKS),
                        help="Methods to run. Pass none (`--algorithms`) with --diagnostics to "
                             "measure the corpus without attacking it.")
    parser.add_argument("--scopes", nargs="+", default=list(TEST_AUTHOR_SCOPES),
                        choices=list(TEST_AUTHOR_SCOPES),
                        help="Author populations to attack the collection over. 'all' is the test "
                             "quarter whole; 'unseen' keeps only the authors absent from the "
                             "known side, i.e. the ones no hyper-parameter and no --projection "
                             "was chosen with. Each is a complete run and lands in the same "
                             "directory under a 'scope' column -- their BCubed scores are NOT "
                             "comparable with each other (see the module docstring).")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--metric", default="cosine")
    parser.add_argument("--standardize", action=argparse.BooleanOptionalAction, default=False,
                        help="Z-score the features per column on known-side statistics before "
                             "anything else, exactly as run_experiment.py does by default. "
                             "**Off here, and that asymmetry between the two families is real, "
                             "not an oversight of spelling**: cosine already normalises each "
                             "document by its own norm (per row), which is a different axis from "
                             "this (per column), and every clustering result on disk was produced "
                             "without it. A non-default value is appended to the output directory "
                             "name so the two cannot overwrite each other.")
    parser.add_argument("--projection", default="none", choices=sorted(PROJECTION_FITTERS) + ["none"],
                        help="Fit a metric-learning projection on the FIRST HALF of the timeline "
                             "and cluster in that space instead of the raw feature space. See "
                             "PROJECTION_FIT_FRACTION for why it is the first half and not the "
                             "whole known side. A non-default value is appended to the output "
                             "directory name, which takes the run out of the three-part set "
                             "plot_results.py draws.")
    parser.add_argument("--projection-components", type=int, default=1024,
                        help="Output dimension for 'lda' and 'contrastive'.")
    parser.add_argument("--projection-shrinkage", type=float, default=0.1,
                        help="Covariance shrinkage for 'wccn' and 'lda'.")
    parser.add_argument("--projection-steps", type=int, default=30_000,
                        help="Gradient steps for 'contrastive'.")
    parser.add_argument("--projection-temperature", type=float, default=0.05,
                        help="Softmax temperature for 'contrastive'.")
    parser.add_argument("--projection-batch", type=int, default=1024,
                        help="Authors per batch for 'contrastive'; each contributes two documents.")
    parser.add_argument("--rescoring", default="none", choices=sorted(GRAPH_RESCORINGS),
                        help="Hubness correction applied to both neighbour graphs before "
                             "clustering. Like --projection, a non-default value is appended to "
                             "the output directory name.")
    parser.add_argument("--time-weight", type=float, default=0.0,
                        help="Weight on the elapsed-time term fused into every edge score (see "
                             "rescoring.temporal_fusion). 0 is the pure-text attack. A non-zero "
                             "value is appended to the output directory name, because it changes "
                             "what the result claims: style AND session structure, not style.")
    parser.add_argument("--tune-time-weight", action="store_true",
                        help="Choose --time-weight by search instead of fixing it: each algorithm "
                             "is tuned over a grid laid around the balanced weight (see "
                             "time_weight_grid, which derives it from the tuning graph rather "
                             "than fixing it, because where the two terms balance is a property "
                             "of the corpus) jointly with its own hyper-parameters, on the "
                             "tuning slice [0.50, 0.75) and never on the collection "
                             "under attack. The winner "
                             "goes in clustering_results.csv's per-row `time_weight` and the "
                             "whole product is in tuning_trials.csv. Costs one existing search "
                             "per candidate weight. Tags the directory `time` rather than "
                             "`time<w>`, since the value is a result of the run and not a setting "
                             "of it, and cannot be combined with a non-zero --time-weight.")
    parser.add_argument("--rescoring-locality", type=int, default=10,
                        help="Neighbours each rescoring summarises a point's neighbourhood over.")
    parser.add_argument("--projection-hard-negatives", type=int, default=0,
                        help="Refresh interval in steps for hard-negative batches in "
                             "'contrastive'; 0 keeps random batches.")
    parser.add_argument("--max-neighbors", type=int, default=MAX_NEIGHBORS)
    parser.add_argument("--diagnostics", action="store_true",
                        help="Also measure the problem itself -- verification AUC, neighbour-graph "
                             "quality per k, the within-language control and the link ranking -- "
                             "for both slices. Needs no clustering algorithm.")
    parser.add_argument("--oracle-sweep", action="store_true",
                        help="Also score the whole grid on the TEST collection, to bound what "
                             "tuning could have achieved. An oracle: never select on it.")
    parser.add_argument("--tune-limit", type=int, default=None,
                        help="Cap the grid, for smoke tests. Omit for the full search.")
    return parser.parse_args()


def search_spaces(args, algorithm: str) -> dict:
    """The hyper-parameter grid for one algorithm.

    **Every distance threshold is searched as a quantile of the graph's own edge weights**, and
    there is no absolute-radius mode any more. An absolute radius is only meaningful in the space
    it was tuned in, and this run has at least two spaces in it whatever the flags say:

    * ``--projection``, ``--rescoring`` and the elapsed-time fusion each rewrite the edge scores.
      The fusion is the easiest to miss because it is not a projection --
      ``rescoring.temporal_fusion`` returns ``(1-w)*z(distance) + w*z(log-hours)``, standardised
      over the graph's own edges, so its scores centre near zero and go negative.
    * **The tuning slice and the collection under attack are not one space either.**
      ``--known-defense`` defaults to ``base``, so on a defended cell the threshold is chosen on a
      graph built from *undefended* vectors and applied to one built from defended vectors, whose
      distances sit at a different scale. Nothing in the flags marks that, which is why the mode
      cannot be inferred from them and had to stop being a choice.

    A quantile asks the same question of every one of those ("keep the closest 28% of candidate
    edges") and lands in the right place in each. It is also not an oracle: :func:`edge_quantile`
    reads the *unlabelled* edge weights of the collection being clustered, which is data the
    attacker holds.

    What this costs is that a defense's effect on the distance *scale* becomes invisible -- a
    quantile keeps the same share of edges however far apart the rewrite pushed the documents, so
    a defense can only register through changes in edge ranking. That is a real limitation of
    every number produced here and belongs in any caption comparing defenses.

    Only the three threshold-based methods have quantile grids; ``leiden``'s ``resolution`` and
    ``hdbscan``'s ``min_cluster_size`` are not distances, so they fall back to
    :data:`CLUSTERING_SPACES` and carry across spaces unchanged.
    """
    if algorithm in CLUSTERING_SPACES_QUANTILE:
        return CLUSTERING_SPACES_QUANTILE
    return CLUSTERING_SPACES


def projection_kwargs(args) -> dict:
    """Only the hyper-parameters the chosen projection actually takes.

    The three fitters share one flag namespace but not one signature, so passing all of them would
    be a ``TypeError`` on every fitter but ``contrastive``.
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


def attack_scope(test: Collection, tuning: Collection, known_authors: np.ndarray,
                 output_dir: Path, args) -> tuple[list[dict], list, list]:
    """Baselines and every algorithm for one scope: ``(result rows, trials, oracle tables)``.

    The whole per-scope run in one place, so that a second scope is a loop iteration rather than a
    second copy of the pipeline. Per-algorithm files are written as each finishes, so an
    interrupted run still leaves the partitions it did produce; the summary tables are the
    caller's, since they hold every scope.
    """
    suffix = scope_suffix(test.scope)
    test_authors, tuning_authors = test.authors, tuning.authors
    rows = run_baselines(test)
    for row in rows:
        print(f"  [{test.scope}] {row['algorithm']:<28s} BCubed F={row['bcubed_f']:.4f}")
    print()

    trials_tables, oracle_tables = [], []
    for algorithm in args.algorithms:
        if algorithm not in CLUSTERING_ATTACKS:
            raise SystemExit(f"unknown algorithm {algorithm!r}; "
                             f"available: {sorted(CLUSTERING_ATTACKS)}")
        # A method that cannot run at this scale is skipped with a note rather than killing the
        # cell: the other algorithms' results are still worth having, and the reason is a property
        # of the method (see MAX_DENSE_DOCUMENTS) that a reader of the table needs to know. The
        # limit is on the collection, so a scope may clear it where another does not -- which is
        # itself worth printing, since it means the two scopes ran different sets of algorithms.
        if algorithm == "average_linkage" and len(test_authors) > MAX_DENSE_DOCUMENTS:
            print(f"  [{test.scope}] {algorithm:<28s} SKIPPED: needs a dense "
                  f"{len(test_authors):,}^2 matrix "
                  f"({len(test_authors) ** 2 * 8 / 1e9:.1f} GB), over the "
                  f"{MAX_DENSE_DOCUMENTS:,}-document limit.")
            continue
        row, labels, trials = run_algorithm(algorithm, test, test_authors,
                                            tuning, tuning_authors, args)
        trials.insert(0, "scope", test.scope)
        trials_tables.append(trials)
        if row is None:
            continue
        row = {"scope": test.scope, **row}
        rows.append(row)
        # `author_in_known` rides along so a downstream re-score can split this partition by
        # whether the attacker held history for the document's author, without reloading the
        # corpus and re-deriving the boundary. It is the column the `unseen` scope selects on, so
        # it is constant *false* there -- carried anyway, so the two scopes' files have one schema
        # and a reader never has to know which produced it.
        pd.DataFrame({"doc_id": test.frame["doc_id"].to_numpy(), "cluster": labels,
                      "author_in_known": np.isin(test_authors, known_authors)}).to_csv(
            output_dir / f"clusters_{algorithm}{suffix}.csv", index=False)
        table = ClusterContingency.from_labels(labels, test_authors)
        per_author_clustering(table, np.unique(test_authors)).to_csv(
            output_dir / f"author_report_{algorithm}{suffix}.csv", index=False)
        print(f"  [{test.scope}] {algorithm:<28s} BCubed F={row['bcubed_f']:.4f} "
              f"(P={row['bcubed_precision']:.3f} R={row['bcubed_recall']:.3f}), "
              f"clusters={row['n_clusters']:,}, any-link={row['any_link_rate']:.3f}, "
              f"amplification={row['amplification']:.2f}")
        if args.oracle_sweep:
            sweep = oracle_sweep(algorithm, test.graph, test_authors, args.tune_limit,
                                 search_spaces(args, algorithm))
            sweep.insert(0, "scope", test.scope)
            oracle_tables.append(sweep)
            usable = sweep["bcubed_f"].dropna()
            row["oracle_best_bcubed_f"] = float(usable.max())
            row["oracle_median_bcubed_f"] = float(usable.median())
            print(f"  {'':<28s} untuned default F={row['untuned_bcubed_f']:.4f} "
                  f"(tuning gain {row['tuning_gain_over_default']:+.4f}) | "
                  f"grid on test: median={row['oracle_median_bcubed_f']:.4f} "
                  f"best={row['oracle_best_bcubed_f']:.4f}")
        else:
            print(f"  {'':<28s} untuned default F={row['untuned_bcubed_f']:.4f} "
                  f"(tuning gain {row['tuning_gain_over_default']:+.4f})")
        print(f"  {'':<28s} tuned on slice F={row['tuning_bcubed_f']:.4f} "
              f"(tuning optimism {row['tuning_optimism']:+.4f}), "
              f"{row['seconds_tuning']:.0f}s tune + {row['seconds_attack']:.0f}s attack "
              f"[{row['hyperparameters']}]")
    return rows, trials_tables, oracle_tables


def main() -> None:
    args = parse_args()
    if args.tune_time_weight and args.time_weight > 0:
        raise SystemExit("--tune-time-weight searches the weight; --time-weight fixes it. Pass "
                         "one or the other, not both (the directory name can only say which "
                         "happened, not both).")
    tag = f"{args.source}_{args.defense}_{args.feature}_{variant_name(args)}"
    # A defended known side is a different experiment: the tuning slice and any fitted projection
    # then carry the same appended text as the collection under attack. Suffixed so it cannot
    # overwrite the deployment-model run, and so `plot_results.py` reads it as its own directory.
    if args.known_defense != "base":
        tag = f"{tag}_knowndef-{args.known_defense}"
    # After the variant rather than before --projection, which is where it used to sit. The
    # positional slot wins over pipeline order: `--standardize` rewrites the space the projection
    # is fitted on, so reading it first was truer to what happens, but the fourth part of the name
    # has to be the variant and nothing else for `plot_results.parse_clustering_run_name` to find
    # it there.
    if args.standardize:
        tag = f"{tag}_zscore"
    output_dir = args.output_dir or (OUTPUT_ROOT / tag)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame, embeddings = load_documents_and_features(
        args.data_dir, args.source, args.feature,
        defense="none" if args.defense == "base" else args.defense)
    embeddings = np.nan_to_num(embeddings)

    # The attacker's own history, which by default is UNDEFENDED text. Nothing here enrols an
    # author, but the known side still decides hyper-parameters and fits --projection, so it has
    # to come from the matrix the attacker would really hold. Row-aligned to `frame` through the
    # same loader and the same `doc_id` join, so one position means the same document in both.
    known_embeddings = embeddings
    if args.known_defense != args.defense:
        known_frame, known_embeddings = load_documents_and_features(
            args.data_dir, args.source, args.feature,
            defense="none" if args.known_defense == "base" else args.known_defense)
        if not known_frame["doc_id"].equals(frame["doc_id"]):
            raise SystemExit(
                f"the known-side and test-side feature files do not cover the same documents in "
                f"the same order ({len(known_frame):,} vs {len(frame):,} rows) -- one of them is "
                f"stale; rebuild it with `compute_features --source {args.source} --defense ...`.")
        known_embeddings = np.nan_to_num(known_embeddings)
        print(f"  known side: {args.known_defense} ({known_embeddings.shape[0]:,} documents); "
              f"collection under attack: {args.defense}")

    if args.standardize:
        # Known-side statistics, `[0, KNOWN_FRACTION)`, and NOT PROJECTION_FIT_FRACTION's first
        # half: that bound exists because a *supervised* fit reaching into the tuning slice would
        # have seen those documents' authors. A z-score reads no labels, so it cannot leak one,
        # and matching run_experiment.py's known side is what makes this the same operation that
        # family performs. Fixed rather than per-scope on purpose -- statistics that moved between
        # `all` and `unseen` would put the two scopes in different spaces.
        # `standardize` returns a block per argument; the first is the known side re-scored, which
        # is a view's worth of wasted work and is dropped. The public function is called rather
        # than the arithmetic inlined precisely because "what this project means by standardising"
        # is the thing under test.
        known_side = slice(0, int(round(KNOWN_FRACTION * len(frame))))
        # Fitted on the KNOWN matrix (undefended by default) and applied to both, so the tuning
        # slice and the collection under attack end up in one space -- statistics differing
        # between them would make the two graphs incomparable before any algorithm ran.
        statistics_source = known_embeddings[known_side]
        if known_embeddings is not embeddings:
            known_embeddings = standardize(statistics_source, known_embeddings)[1]
        embeddings = standardize(statistics_source, embeddings)[1]
        print(f"  standardized on {known_side.stop:,} known-side documents "
              f"({embeddings.shape[1]} columns)")

    if args.projection != "none":
        fit_window = slice(0, int(round(PROJECTION_FIT_FRACTION * len(frame))))
        started = time.perf_counter()
        # Fitted on the known matrix: a projection learned from defended history is not what an
        # attacker holding original text would have.
        projection = fit_projection(args.projection, known_embeddings[fit_window],
                                    frame["author_id"].to_numpy()[fit_window],
                                    **projection_kwargs(args))
        # Whole-corpus transform, not per-window: the projection is fitted blind to which slice a
        # document falls in, and applying it once keeps the two graphs in the same space.
        if known_embeddings is not embeddings:
            known_embeddings = projection.transform(known_embeddings)
        embeddings = projection.transform(embeddings)
        print(f"  projection {projection.name}: fitted on {fit_window.stop:,} documents "
              f"({frame['author_id'].iloc[fit_window].nunique():,} authors), "
              f"{embeddings.shape[1]} dimensions, {time.perf_counter() - started:.0f}s")

    print(tag)
    seconds = (elapsed_seconds(frame)
               if args.time_weight > 0 or args.tune_time_weight else None)
    # Deduplicated, order preserved: a repeated --scopes value would otherwise attack the same
    # collection twice and write two identical rows for it under one scope name.
    scopes = list(dict.fromkeys(args.scopes))
    prepared = {scope: prepare_scope(scope, frame, embeddings, seconds, args, known_embeddings)
                for scope in scopes}
    prepared = {scope: pair for scope, pair in prepared.items() if pair is not None}
    if not prepared:
        raise SystemExit("no scope has enough documents to attack; nothing to do.")

    # Before `embeddings` is released: the verification pass needs the vectors, not the graph.
    if args.diagnostics:
        started = time.perf_counter()
        diagnostic_rows, graph_tables = [], []
        for collections in prepared.values():
            for collection in collections:
                row, per_k = slice_diagnostics(
                    collection.name, collection.frame,
                    take_rows(embeddings, collection.positions), collection.graph, args.metric)
                diagnostic_rows.append(row)
                graph_tables.append(per_k)
                print(f"  diagnostics [{collection.name}]: "
                      f"verification AUC={row['verification_auc']:.4f} "
                      f"(macro {row['verification_auc_macro']:.4f}), AP={row['verification_ap']:.4f} "
                      f"({row['verification_ap'] / row['verification_prevalence']:.0f}x prevalence), "
                      f"1-NN edge precision={per_k.iloc[0]['nn_precision']:.3f} "
                      f"({per_k.iloc[0]['nn_precision'] / per_k.iloc[0]['random_nn_precision']:.0f}x chance)")
        pd.DataFrame(diagnostic_rows).to_csv(output_dir / "diagnostics.csv", index=False)
        pd.concat(graph_tables, ignore_index=True).to_csv(
            output_dir / "graph_diagnostics.csv", index=False)
        print(f"  diagnostics took {time.perf_counter() - started:.0f}s")
    del embeddings
    print()

    # Every author the attacker holds labelled documents for. Read once here, from the whole
    # corpus, so it means the same thing in both scopes' output files.
    known_authors = np.unique(frame["author_id"].to_numpy()[:slice_bounds(len(frame))[1].start])

    rows, all_trials, oracle_tables = [], [], []
    for test, tuning in prepared.values():
        scope_rows, scope_trials, scope_oracle = attack_scope(test, tuning, known_authors,
                                                              output_dir, args)
        rows.extend(scope_rows)
        all_trials.extend(scope_trials)
        oracle_tables.extend(scope_oracle)
        print()

    results = pd.DataFrame(rows)
    # `time_weight` is written per row by `run_algorithm` -- under --tune-time-weight each
    # algorithm chose its own, so there is no single run-level value to state. The baseline rows
    # have no such column, which is correct: a reference partition reads no graph.
    for column, value in (("rescoring", args.rescoring),
                          ("projection", args.projection),
                          ("feature", args.feature),
                          ("defense", args.defense), ("dataset", args.source)):
        results.insert(0, column, value)
    if "time_weight" not in results:
        results.insert(0, "time_weight", args.time_weight)
    results.to_csv(output_dir / "clustering_results.csv", index=False)
    if all_trials:
        pd.concat(all_trials, ignore_index=True).to_csv(output_dir / "tuning_trials.csv",
                                                        index=False)
    if oracle_tables:
        pd.concat(oracle_tables, ignore_index=True).to_csv(output_dir / "oracle_sweep.csv",
                                                           index=False)
    print(f"Wrote {output_dir}/clustering_results.csv ({', '.join(prepared)} scope"
          f"{'s' if len(prepared) > 1 else ''}), clusters_*.csv, author_report_*.csv")


if __name__ == "__main__":
    main()
