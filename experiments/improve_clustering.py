#!/usr/bin/env python
"""Search for a better author-clustering attack, scored **only on the tuning slice**.

``run_clustering.py`` reports one configuration per algorithm on the collection under attack. This
script is the development loop that sits behind it: it sweeps representations, graph rescorings
and sparsification levels against the *labelled* tuning slice, so a method can be built and
selected without ever reading the test quarter.

The three slices, and the rule
------------------------------
Documents are ordered by ``ended_at``, exactly as everywhere else in this project.

======================  ==========  ==================================================
``[0.00, 0.50)``        history     Labelled. **The only slice anything may be fitted
                                    on** -- projections, whitening, calibration.
``[0.50, 0.75)``        eval        The simulated collection under attack. Labels are
                                    read to *score* a partition and for nothing else.
                                    This is what every number below is measured on.
``[0.75, 1.00)``        test        Untouched. ``--slice test`` exists and refuses to
                                    run without ``--i-am-done-developing``.
======================  ==========  ==================================================

That history/eval boundary is what makes a supervised representation legitimate here rather than
leakage: the projection is fitted on documents that ended before the collection under attack
began, and is then applied to every document in it blind. No document in the eval slice is
labelled to the attack, no author in it is enrolled, and the number of clusters is still not an
input.

**Every number this script prints is selection-optimistic** and has to be read that way. Sweeping
hundreds of configurations against one labelled slice and reporting the best is a search over the
eval set; the honest estimate of what the winner is worth comes from running it once on the test
quarter afterwards, which is what ``run_clustering.py`` is for. The point of the sweep is to find
*which* method to spend that single measurement on.

The frontier, and why the sweep is over an edge budget
-------------------------------------------------------
Threshold-and-connect is single-linkage cut at a height, so a sweep over ``distance_threshold`` is
a sweep over *how many of the candidate edges are kept*. Sweeping the budget directly instead of
the threshold has three advantages and no cost: rescored distances are on incomparable scales (CSLS
distances are routinely negative), so a shared threshold grid would mean different things to
different variants; the budget is what actually determines the partition; and the whole frontier
for one graph comes from a single sort plus one connected-components pass per budget, which is
what makes a sweep this wide affordable.

The stages
----------
Each is one ``--stage``; they write separate tables and are compared by ``--summarize``.

===============  =============================================================================
``graph``        the raw feature space, swept over rescorings, ``k`` and edge budget. The
                 baseline every other stage is read against
``projection``   closed-form metric learning on history (WCCN, LDA), then the same sweep
``contrastive``  a projection trained with an in-batch contrastive loss. **The only stage that
                 wants a GPU**
``edge_scorer``  a learned pairwise cross-encoder re-ranking the graph's edges
``algorithms``   Leiden / HDBSCAN / componentwise average linkage on one graph, as a
                 counterpart to the single-linkage frontier the other stages sweep
``seen_unseen``  splits one partition's BCubed by whether the document's author also wrote in
                 history -- the validity check on every supervised result here
``all``          ``graph`` + ``projection`` + ``contrastive`` in one process
===============  =============================================================================

Run (from the repo root)::

    python experiments/improve_clustering.py --stage graph            # cheap, CPU, minutes
    sbatch scripts/improve_clustering_cpu_slurm.sh --stage projection
    sbatch scripts/improve_clustering_slurm.sh --stage contrastive    # GPU
    python experiments/improve_clustering.py --summarize              # leaderboard over them all

Outputs, under ``experiments/clustering_dev/<source>_<defense>_<feature>/``:

* ``frontier_<run-tag>.csv`` -- every (projection, rescoring, k, budget) with its BCubed scores.
* ``best_<run-tag>.csv`` -- the top configuration per (projection, rescoring), for reading.
* ``leaderboard.csv`` (``--summarize``) -- the best row of every method across every stage.
* ``projections/<name>.npy`` -- fitted projection matrices, so a later stage or
  ``run_clustering.py --projection`` can reuse one without refitting.

``--run-tag`` names the output files; without it they are named after the stage, so two runs of one
stage (a fused-feature sweep and a plain one) would overwrite each other.
"""

from __future__ import annotations

import argparse
import hashlib
import json
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
    LinearProjection,
    fit_projection,
    identity_projection,
)
from prompt_anonymity.attacks.clustering.edge_scoring import fit_edge_scorer  # noqa: E402
from prompt_anonymity.attacks.clustering.linkage import capped_linkage  # noqa: E402
from prompt_anonymity.attacks.clustering.rescoring import (  # noqa: E402
    global_distance_moments,
    temporal_fusion,
    zscore_distances,
)
from prompt_anonymity.attacks.clustering.rescoring import rescore_graph  # noqa: E402
from prompt_anonymity.attacks.common import unit_rows  # noqa: E402
from prompt_anonymity.evaluation.metrics.clustering import bcubed_scores  # noqa: E402

from run_experiment import load_documents_and_features  # noqa: E402

OUTPUT_ROOT = REPO_ROOT / "experiments" / "clustering_dev"
CACHE_ROOT = REPO_ROOT / "data" / ".cache" / "clustering_dev"
DATA_DIR = REPO_ROOT / "data" / "hf"

#: Slice boundaries as fractions of the ``ended_at``-ordered corpus. ``eval`` is
#: ``run_clustering.py``'s tuning slice and ``test`` is its collection under attack, so a winner
#: found here transfers to that script without re-deriving anything.
SLICES = {"history": (0.00, 0.50), "eval": (0.50, 0.75), "test": (0.75, 1.00)}

#: Width every neighbour graph is built at. The rescorings re-rank within these candidates
#: (see :mod:`~prompt_anonymity.attacks.clustering.rescoring`), so this bounds how far a pair can
#: be promoted, and every ``--k-caps`` value is a column slice of one build.
GRAPH_WIDTH = 100

#: Per-document neighbour caps swept. ``connected``'s tuned value on this corpus is 2, and the
#: interesting range is entirely below 10 -- but the wide end is kept because a rescored graph has
#: no reason to share the cosine graph's optimum, which is the hypothesis under test.
DEFAULT_K_CAPS = (1, 2, 3, 5, 10, 25, 50)

#: Edge budgets, as a geometric grid over the number of candidate edges kept. Fine enough that the
#: BCubed F peak is located to well under its bootstrap width, coarse enough that a full sweep is
#: minutes. The frontier is smooth in this parameter, so nothing is hiding between grid points.
#: Raised from 48 after the first sweep: at 48 the four grid points bracketing the peak spanned
#: 37k-110k edges and 0.07 of BCubed F, which locates a maximum far too loosely to rank two
#: methods against each other. The whole sweep is ~3 s per (rescoring, locality), so resolution
#: here is close to free.
BUDGET_STEPS = 160


def slice_indices(n_documents: int, name: str) -> slice:
    """Document range for a named slice of the ordered corpus."""
    start, end = SLICES[name]
    return slice(int(round(start * n_documents)), int(round(end * n_documents)))


def fast_bcubed(labels: np.ndarray, author_codes: np.ndarray, n_authors: int) -> tuple:
    """BCubed precision, recall and F, in one pass over the cluster/author contingency.

    Algebraically identical to
    :func:`~prompt_anonymity.evaluation.metrics.clustering.bcubed_scores` -- each document's
    contribution is ``|cluster ∩ author| / |cluster|`` and ``.../|author|``, so summing
    ``count^2 / size`` over the contingency's non-zero cells and dividing by ``n`` gives the same
    two means. ``--verify-bcubed`` checks that against the library implementation, which stays the
    reference; this exists only because the sweep evaluates it a few thousand times and the
    library builds a sparse contingency object each call.
    """
    n = len(labels)
    _, cluster_codes = np.unique(labels, return_inverse=True)
    cluster_codes = cluster_codes.ravel()
    n_clusters = int(cluster_codes.max()) + 1

    key = cluster_codes.astype(np.int64) * n_authors + author_codes
    cells, counts = np.unique(key, return_counts=True)
    counts = counts.astype(np.float64)
    cluster_sizes = np.bincount(cluster_codes, minlength=n_clusters).astype(np.float64)
    author_sizes = np.bincount(author_codes, minlength=n_authors).astype(np.float64)

    precision = float((counts ** 2 / cluster_sizes[cells // n_authors]).sum() / n)
    recall = float((counts ** 2 / author_sizes[cells % n_authors]).sum() / n)
    f_score = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return precision, recall, f_score, n_clusters


def budget_frontier(source: np.ndarray, target: np.ndarray, distance: np.ndarray,
                    author_codes: np.ndarray, n_authors: int, n_documents: int,
                    steps: int = BUDGET_STEPS) -> list[dict]:
    """BCubed along the whole precision/recall frontier of one graph, by edge budget.

    Edges are sorted once by rescored distance; each budget takes a prefix and runs connected
    components, which *is* single-linkage cut at the threshold that prefix corresponds to. The
    returned rows therefore trace out exactly the curve a ``distance_threshold`` sweep would, on a
    grid that means the same thing for every rescoring.
    """
    order = np.argsort(distance, kind="stable")
    source, target, distance = source[order], target[order], distance[order]
    n_edges = len(distance)
    if n_edges == 0:
        return []

    budgets = np.unique(np.geomspace(1, n_edges, steps).astype(np.int64))
    rows = []
    for budget in budgets:
        adjacency = csr_matrix((np.ones(budget, dtype=np.int8),
                                (source[:budget], target[:budget])),
                               shape=(n_documents, n_documents))
        _, labels = connected_components(adjacency, directed=False)
        precision, recall, f_score, n_clusters = fast_bcubed(labels, author_codes, n_authors)
        sizes = np.bincount(labels)
        rows.append({
            "edge_budget": int(budget),
            "edge_share": float(budget / n_edges),
            # The threshold this prefix corresponds to, so a winning row can be reproduced by
            # `ThresholdComponents` without re-deriving the budget.
            "distance_threshold": float(distance[budget - 1]),
            "bcubed_precision": precision,
            "bcubed_recall": recall,
            "bcubed_f": f_score,
            "n_clusters": n_clusters,
            "largest_cluster": int(sizes.max()),
            "largest_cluster_share": float(sizes.max() / n_documents),
            "singleton_share": float((sizes == 1).sum() / n_documents),
        })
    return rows


def graph_edges(graph, k_cap: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Deduplicated undirected edge list of the graph truncated to ``k_cap`` neighbours."""
    return graph.truncate(min(k_cap, graph.k)).edges()


def cached_graph(embeddings: np.ndarray, width: int, tag: str, cache_dir: Path):
    """Build a neighbour graph, or read it back if this exact input was built before.

    Keyed by the *content* of the projected embeddings, so a changed projection can never serve a
    stale graph and an unchanged one never rebuilds. The build is by far the most expensive step
    in the sweep -- minutes at 3,072 dimensions against seconds for everything downstream -- and
    the sweep re-enters it for every stage.
    """
    digest = hashlib.blake2b(np.ascontiguousarray(embeddings).view(np.uint8),
                             digest_size=8).hexdigest()
    path = cache_dir / f"graph_{tag}_{digest}_k{width}.npz"
    if path.exists():
        stored = np.load(path)
        from prompt_anonymity.attacks.clustering import NeighborGraph
        return NeighborGraph(stored["indices"], stored["distances"], str(stored["metric"]))

    started = time.perf_counter()
    graph = build_neighbor_graph(embeddings, width, metric="cosine")
    print(f"    built k={width} graph over {len(embeddings):,} x {embeddings.shape[1]} "
          f"in {time.perf_counter() - started:.0f}s", flush=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez(path, indices=graph.indices, distances=graph.distances, metric=graph.metric)
    return graph


def sweep_graph(graph, author_codes: np.ndarray, n_authors: int, projection_name: str,
                rescorings: tuple[str, ...], k_caps: tuple[int, ...],
                localities: tuple[int, ...]) -> list[dict]:
    """Every (rescoring, locality, k, budget) over one built graph."""
    rows = []
    n_documents = graph.n_documents
    for rescoring in rescorings:
        for locality in (localities if rescoring != "none" else (0,)):
            started = time.perf_counter()
            view = rescore_graph(graph, rescoring, locality) if rescoring != "none" else graph
            for k_cap in k_caps:
                source, target, distance = graph_edges(view, k_cap)
                finite = np.isfinite(distance)
                for row in budget_frontier(source[finite], target[finite], distance[finite],
                                           author_codes, n_authors, n_documents):
                    rows.append({"projection": projection_name, "rescoring": rescoring,
                                 "locality": locality, "k": k_cap, **row})
            best = max((r["bcubed_f"] for r in rows
                        if r["rescoring"] == rescoring and r["locality"] == locality), default=0)
            print(f"    {rescoring:<14s} locality={locality:<3d} best F={best:.4f}  "
                  f"[{time.perf_counter() - started:.0f}s]", flush=True)
    return rows


class EnsembleProjection:
    """Several linear projections applied together, so cosine averages over them.

    Diversity from *hyper-parameters* rather than seeds: the saved projections differ in width,
    temperature and training length, which spreads the errors further apart than re-seeding one
    configuration would. Duck-types :class:`LinearProjection` -- same ``name``, ``transform`` and
    ``n_components`` -- so nothing downstream branches on which it was handed.
    """

    def __init__(self, matrices: list[np.ndarray], name: str):
        self.matrices, self.name = matrices, name

    @property
    def n_components(self) -> int:
        return sum(matrix.shape[1] for matrix in self.matrices)

    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        scale = np.float32(1.0 / np.sqrt(len(self.matrices)))
        blocks = [unit_rows(np.asarray(embeddings, dtype=np.float32)
                            @ matrix.astype(np.float32)) * scale
                  for matrix in self.matrices]
        return np.hstack(blocks).astype(np.float32)


def fuse_features(blocks: list[np.ndarray], weights: list[float]) -> np.ndarray:
    """Concatenate unit-normalised feature blocks so cosine becomes their weighted mean cosine.

    With each block L2-normalised and scaled by ``sqrt(w)``, the concatenation has unit norm and
    ``x . y = sum_b w_b cos_b(x, y)``. So fusing two featurizers is a column stack and nothing
    else -- no second graph, no rank aggregation, no distance matrix to blend -- and the weight
    means exactly what it looks like it means. That equivalence is the only reason this is done at
    the feature level rather than by combining two neighbour graphs, where "average the distances"
    would have to invent a rule for pairs present in one graph and absent from the other.
    """
    total = float(sum(weights))
    return np.hstack([unit_rows(np.nan_to_num(block)).astype(np.float32)
                      * np.float32(np.sqrt(weight / total))
                      for block, weight in zip(blocks, weights)])


def load_slices(args) -> tuple[dict, dict]:
    """Documents and features for the history and evaluation slices, as ``{name: (frame, X)}``.

    With ``--extra-feature`` the two parquets are joined through :func:`fuse_features`. They are
    read by the same loader and therefore land in the same ``doc_id`` order, which is what makes a
    column stack safe; the loader's own one-to-one join is what would fail loudly otherwise.
    """
    frame, embeddings = load_documents_and_features(
        args.data_dir, args.source, args.feature,
        defense="none" if args.defense == "base" else args.defense)
    if args.extra_feature:
        extra_frame, extra = load_documents_and_features(
            args.data_dir, args.source, args.extra_feature,
            defense="none" if args.defense == "base" else args.defense)
        if not extra_frame["doc_id"].equals(frame["doc_id"]):
            raise SystemExit(f"{args.feature} and {args.extra_feature} cover different documents; "
                             f"fusing them would misalign every row.")
        embeddings = fuse_features([embeddings, extra], [1.0, args.extra_feature_weight])
        del extra
        print(f"  fused {args.feature} + {args.extra_feature} "
              f"(weight {args.extra_feature_weight}) -> {embeddings.shape[1]} dimensions")

    n = len(frame)
    out = {}
    for name in ("history", args.slice):
        window = slice_indices(n, name)
        out[name] = (frame.iloc[window].reset_index(drop=True),
                     np.nan_to_num(embeddings[window]).astype(np.float32))
    del embeddings
    return out, {name: slice_indices(n, name) for name in out}


def sweep_algorithms(graph, author_codes: np.ndarray, n_authors: int, projection_name: str,
                     k_caps: tuple[int, ...]) -> list[dict]:
    """The algorithm zoo on one graph, as a counterpart to the single-linkage frontier.

    ``budget_frontier`` sweeps connected components exhaustively because that method is a prefix of
    a sorted edge list and costs nothing to re-cut. The other three have real parameters and real
    fits, so they get an explicit grid. Everything is scored through the same
    :func:`fast_bcubed`, on the same graph, so a difference between rows here is a difference
    between methods and not between harnesses.
    """
    from prompt_anonymity.attacks.clustering import (
        ComponentwiseAgglomerative,
        HDBSCANClustering,
        LeidenClustering,
    )
    from prompt_anonymity.evaluation.metrics.clustering import expand_noise

    # Thresholds as QUANTILES of this graph's own edge distances, never absolute numbers. A
    # learned projection changes the scale entirely -- WildChat's single-linkage optimum is at
    # distance 0.13 under cosine and 0.39 after the contrastive fit -- so an absolute grid tuned in
    # one space tests a different, and usually useless, part of the other. The first pass here did
    # exactly that and scored average linkage at 0.336 with a grid whose whole range sat below the
    # useful threshold.
    _, _, all_distances = graph.edges()
    all_distances = np.sort(all_distances[np.isfinite(all_distances)])
    def at_quantile(q: float) -> float:
        return float(all_distances[min(int(q * len(all_distances)), len(all_distances) - 1)])

    configurations = []
    for k_cap in k_caps:
        for resolution in (0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0):
            for objective in ("cpm", "modularity"):
                configurations.append(("leiden", LeidenClustering(
                    neighbors=k_cap, resolution=resolution, objective=objective)))
        for min_cluster_size in (2, 3, 5):
            configurations.append(("hdbscan", HDBSCANClustering(
                neighbors=k_cap, min_cluster_size=min_cluster_size)))
        # The single-linkage optimum sits near the 4-8% quantile of the candidate edges on both
        # spaces measured, so the link stage brackets that and the merge stage runs below it.
        for link_quantile in (0.04, 0.06, 0.10, 0.20):
            for cut_quantile in (0.005, 0.01, 0.02, 0.04, 0.06):
                if cut_quantile >= link_quantile:
                    continue
                configurations.append(("componentwise_agglomerative", ComponentwiseAgglomerative(
                    neighbors=k_cap, link_threshold=at_quantile(link_quantile),
                    distance_threshold=at_quantile(cut_quantile),
                    # The whole point of this method is splitting the giant component, so it must
                    # be big enough to hold WildChat's. 30,000^2 float64 is 7.2 GB, which fits the
                    # 32 GB job; leaving the default would silently pass the blob through whole.
                    max_component=30_000)))

    rows = []
    for name, attack in configurations:
        started = time.perf_counter()
        try:
            labels = attack.cluster(graph)
        except Exception as error:                      # a configuration that cannot run is data
            rows.append({"projection": projection_name, "algorithm": name,
                         "settings": json.dumps(attack.settings(), default=str),
                         "bcubed_f": float("nan"), "error": f"{type(error).__name__}: {error}"[:200]})
            continue
        # HDBSCAN's -1 means "not placed", which BCubed must read as a singleton rather than as one
        # giant cluster of every unplaced document -- the difference is most of its precision.
        labels = expand_noise(np.asarray(labels))
        precision, recall, f_score, n_clusters = fast_bcubed(labels, author_codes, n_authors)
        sizes = np.bincount(labels - labels.min())
        rows.append({
            "projection": projection_name, "algorithm": name,
            "settings": json.dumps(attack.settings(), default=str),
            "bcubed_precision": precision, "bcubed_recall": recall, "bcubed_f": f_score,
            "n_clusters": n_clusters, "largest_cluster": int(sizes.max()),
            "largest_cluster_share": float(sizes.max() / len(labels)),
            "seconds": time.perf_counter() - started,
        })
    for name in sorted({row["algorithm"] for row in rows}):
        best = max((row["bcubed_f"] for row in rows
                    if row["algorithm"] == name and row["bcubed_f"] == row["bcubed_f"]), default=0)
        print(f"    {name:<30s} best F={best:.4f}", flush=True)
    return rows


def per_document_bcubed(labels: np.ndarray, author_codes: np.ndarray,
                        n_authors: int) -> tuple[np.ndarray, np.ndarray]:
    """Each document's own BCubed precision and recall contribution.

    :func:`fast_bcubed` returns their means. Keeping the per-document vector is what makes the
    measure decomposable over any partition of the documents -- which is the only reason the
    seen/unseen split below is a valid decomposition rather than a re-scoring of a subset.
    """
    _, clusters = np.unique(labels, return_inverse=True)
    clusters = clusters.ravel()
    key = clusters.astype(np.int64) * n_authors + author_codes
    _, inverse, counts = np.unique(key, return_inverse=True, return_counts=True)
    shared = counts[inverse].astype(np.float64)
    cluster_sizes = np.bincount(clusters).astype(np.float64)
    author_sizes = np.bincount(author_codes, minlength=n_authors).astype(np.float64)
    return shared / cluster_sizes[clusters], shared / author_sizes[author_codes]


def components_at_budget(graph, k_cap: int, budget: int) -> np.ndarray:
    """The partition one row of a frontier table describes, rebuilt from its ``(k, budget)``."""
    source, target, distance = graph_edges(graph, k_cap)
    finite = np.isfinite(distance)
    source, target, distance = source[finite], target[finite], distance[finite]
    order = np.argsort(distance, kind="stable")[:budget]
    adjacency = csr_matrix((np.ones(len(order), dtype=np.int8), (source[order], target[order])),
                           shape=(graph.n_documents,) * 2)
    return connected_components(adjacency, directed=False)[1]


def seen_unseen_report(labels: np.ndarray, author_codes: np.ndarray, n_authors: int,
                       eval_authors: np.ndarray, history_authors: np.ndarray, name: str) -> list:
    """Split a partition's BCubed by whether the document's author also wrote in history.

    **The validity check on every supervised result here.** The projection and the edge scorer are
    fitted on labelled history, and roughly half the eval slice's documents are by authors who
    appear in it. If a method's gain lived entirely on those authors it would be memorisation --
    real under this threat model, but not a same-author *metric*, and it would not transfer to a
    corpus the attacker has no history for. The unseen column is the one that says which it is.
    """
    precision, recall = per_document_bcubed(labels, author_codes, n_authors)
    seen = np.isin(eval_authors, history_authors)
    rows = []
    for group, mask in (("all", np.ones(len(seen), bool)), ("seen", seen), ("unseen", ~seen)):
        p, r = float(precision[mask].mean()), float(recall[mask].mean())
        rows.append({"method": name, "group": group, "n_documents": int(mask.sum()),
                     "n_authors": int(len(np.unique(eval_authors[mask]))),
                     "bcubed_precision": p, "bcubed_recall": r,
                     "bcubed_f": 2 * p * r / (p + r) if p + r else 0.0})
    return rows



def sorted_edges(graph, k_cap: int):
    """``(source, target, order)`` for one k-truncation, edges ordered closest-first."""
    source, target, distance = graph_edges(graph, k_cap)
    finite = np.isfinite(distance)
    source, target, distance = source[finite], target[finite], distance[finite]
    return source, target, np.argsort(distance, kind="stable"), distance


def linkage_rows(graph, author_codes, n_authors, projection_name, k_caps, caps, rules):
    """Size-constrained single linkage, swept over (k, cap, rule, budget).

    One pass per (k, cap, rule) covers every budget, because the linkage is incremental -- the
    same trick that makes the unconstrained frontier affordable.
    """
    rows = []
    for k_cap in k_caps:
        source, target, order, distance = sorted_edges(graph, k_cap)
        budgets = np.unique(np.geomspace(1, len(order), BUDGET_STEPS).astype(np.int64))
        for cap in caps:
            for rule in (rules if cap > 0 else rules[:1]):   # cap=0 ignores the rule
                started = time.perf_counter()
                for used, labels in capped_linkage(source, target, order, graph.n_documents,
                                                   cap, rule, budgets):
                    precision, recall, f_score, n_clusters = fast_bcubed(
                        labels, author_codes, n_authors)
                    sizes = np.bincount(labels)
                    rows.append({"projection": projection_name, "rescoring": "none",
                                 "algorithm": f"capped_linkage[{rule}]", "cap": cap,
                                 "k": k_cap, "edge_budget": int(used),
                                 "distance_threshold": float(
                                     distance[order[min(used, len(order)) - 1]]) if used else 0.0,
                                 "bcubed_precision": precision, "bcubed_recall": recall,
                                 "bcubed_f": f_score, "n_clusters": n_clusters,
                                 "largest_cluster": int(sizes.max()),
                                 "largest_cluster_share": float(sizes.max() / len(labels)),
                                 "singleton_share": float((sizes == 1).sum() / len(labels))})
                best = max(row["bcubed_f"] for row in rows
                           if row["cap"] == cap and row["k"] == k_cap
                           and row["algorithm"].endswith(f"[{rule}]"))
                print(f"    k={k_cap:<3d} cap={cap:<5d} rule={rule:<5s} best F={best:.4f}  "
                      f"[{time.perf_counter() - started:.0f}s]", flush=True)
    return rows


def temporal_rows(graph, frame, author_codes, n_authors, projection_name, k_caps, weights,
                  per_language: bool = False):
    """Fuse the time gap between two documents into the edge score, and sweep the weight.

    **Both ends of the sweep are controls.** ``weight=0`` is the pure-cosine attack this is
    measured against; ``weight=1`` is timing alone, reading no text at all -- the same kind of
    metadata reference as ``baseline_language_primary`` in ``run_clustering.py``, and the number
    that says how much of any gain is style rather than session structure.

    Both terms are standardised over the candidate-edge population before mixing, because a cosine
    distance and a log-hour gap have no common scale and a raw sum would be whichever happens to
    have the larger variance.
    """
    times = pd.to_datetime(frame["ended_at"], errors="coerce", utc=True)
    seconds = times.astype("int64").to_numpy() / 1e9
    seconds[times.isna().to_numpy()] = np.nan

    def standardise(values):
        usable = np.isfinite(values)
        centre, spread = values[usable].mean(), values[usable].std()
        out = (values - centre) / (spread if spread > 0 else 1.0)
        out[~usable] = np.nanmax(out[usable])          # unknown time = maximally far apart
        return out

    rows = []
    for k_cap in k_caps:
        source, target, distance = graph_edges(graph, k_cap)
        finite = np.isfinite(distance)
        source, target, distance = source[finite], target[finite], distance[finite]
        gap = np.abs(seconds[source] - seconds[target]) / 3600.0
        cosine_z = standardise(distance.astype(np.float64))
        time_z = standardise(np.log1p(gap))
        languages = frame["language_primary"].astype(str).to_numpy()
        # Group an edge by the unordered pair of its endpoints' languages.
        pair_key = pd.factorize(pd.Series(
            [f"{a}|{b}" for a, b in zip(np.minimum(languages[source], languages[target]),
                                        np.maximum(languages[source], languages[target]))]))[0]
        for weight in weights:
            fused = (1 - weight) * cosine_z + weight * time_z
            if per_language:
                # Standardise within each language pair, so the same budget cuts each group at its
                # own quantile rather than at a shared absolute score.
                frame_scores = pd.DataFrame({"g": pair_key, "s": fused})
                stats = frame_scores.groupby("g")["s"].agg(["mean", "std", "size"])
                # A group too small to estimate a spread keeps the global scale rather than being
                # rescaled by noise.
                usable = (stats["size"] >= 200) & (stats["std"] > 0)
                centre = np.where(usable, stats["mean"], 0.0)[pair_key]
                spread = np.where(usable, stats["std"], 1.0)[pair_key]
                fused = (fused - centre) / spread
            for row in budget_frontier(source, target, fused, author_codes, n_authors,
                                       graph.n_documents):
                rows.append({"projection": projection_name,
                             "rescoring": f"time{weight:g}" + ("+lang" if per_language else ""),
                             "algorithm": "connected", "locality": 0, "k": k_cap, **row})
            tag = f"time{weight:g}" + ("+lang" if per_language else "")
            best = max(row["bcubed_f"] for row in rows
                       if row["rescoring"] == tag and row["k"] == k_cap)
            print(f"    k={k_cap:<3d} time weight={weight:<4g} best F={best:.4f}", flush=True)
    return rows



def fused_edges(graph, frame, k_cap: int, weight: float):
    """Candidate edges with the cosine/time fusion applied, ordered closest-first."""
    source, target, distance = graph_edges(graph, k_cap)
    finite = np.isfinite(distance)
    source, target, distance = source[finite], target[finite], distance[finite]
    if weight <= 0:
        return source, target, distance.astype(np.float64)
    times = pd.to_datetime(frame["ended_at"], errors="coerce", utc=True)
    seconds = times.astype("int64").to_numpy() / 1e9
    seconds[times.isna().to_numpy()] = np.nan

    def standardise(values):
        usable = np.isfinite(values)
        centre, spread = values[usable].mean(), values[usable].std()
        out = (values - centre) / (spread if spread > 0 else 1.0)
        out[~usable] = np.nanmax(out[usable])
        return out

    gap = np.abs(seconds[source] - seconds[target]) / 3600.0
    return source, target, ((1 - weight) * standardise(distance.astype(np.float64))
                            + weight * standardise(np.log1p(gap)))


def constrained_rows(graph, frame, author_codes, n_authors, projection_name, k_caps,
                     cohesion_quantiles, caps, rules, time_weight, tag):
    """Cohesion- and size-constrained linkage over (optionally time-fused) edges.

    The two constraint families and the fusion are orthogonal -- one changes which edges are
    offered, the others change which offered edges are accepted -- so this sweeps them together
    and lets the tuning slice say whether their gains add or overlap.
    """
    rows = []
    for k_cap in k_caps:
        source, target, distance = fused_edges(graph, frame, k_cap, time_weight)
        order = np.argsort(distance, kind="stable")
        budgets = np.unique(np.geomspace(1, len(order), BUDGET_STEPS).astype(np.int64))
        ordered = distance[order]
        for quantile in cohesion_quantiles:
            ceiling = (float("inf") if quantile >= 1.0
                       else float(ordered[min(int(quantile * len(ordered)), len(ordered) - 1)]))
            for cap in caps:
                for rule in (rules if cap > 0 else rules[:1]):
                    started = time.perf_counter()
                    for used, labels in capped_linkage(
                            source, target, order, graph.n_documents, cap, rule, budgets,
                            distance=distance, max_mean_distance=ceiling):
                        precision, recall, f_score, n_clusters = fast_bcubed(
                            labels, author_codes, n_authors)
                        sizes = np.bincount(labels)
                        rows.append({
                            "projection": projection_name, "rescoring": f"time{time_weight:g}",
                            "algorithm": f"constrained[{rule}]", "cap": cap,
                            "cohesion_quantile": quantile, "k": k_cap, "edge_budget": int(used),
                            "distance_threshold": float(ordered[max(used - 1, 0)]),
                            "bcubed_precision": precision, "bcubed_recall": recall,
                            "bcubed_f": f_score, "n_clusters": n_clusters,
                            "largest_cluster": int(sizes.max()),
                            "largest_cluster_share": float(sizes.max() / len(labels)),
                            "singleton_share": float((sizes == 1).sum() / len(labels))})
                    best = max(row["bcubed_f"] for row in rows
                               if row["k"] == k_cap and row["cohesion_quantile"] == quantile
                               and row["cap"] == cap and row["algorithm"].endswith(f"[{rule}]"))
                    print(f"    k={k_cap:<3d} cohesion_q={quantile:<6g} cap={cap:<5d} "
                          f"rule={rule:<5s} best F={best:.4f} "
                          f"[{time.perf_counter() - started:.0f}s]", flush=True)
    return rows



def time_candidate_rows(graph, frame, features, author_codes, n_authors, projection_name,
                        k_caps, time_neighbors, weight):
    """Add temporally adjacent documents to the candidate set, not just the cosine neighbours.

    **This is the one idea here that can raise recall rather than trade it.** Every other method
    re-ranks or filters a candidate set fixed by cosine k-NN, so a pair the embedding never
    proposed can never be linked however good the scoring gets -- and the loss decomposition says
    the reachable precision headroom runs out at F = 0.677 while recall sits at 0.511. Two
    conversations by one person four minutes apart but about different subjects are exactly the
    pair cosine will not propose and timing will.

    The cost is that the candidate set grows with strangers: this corpus averages a document every
    three minutes, so a document's temporal neighbours are mostly other people. That is what the
    fused score and the budget sweep are for -- a bad candidate is only a bad *offer*, and the
    frontier decides how many offers to accept.
    """
    times = pd.to_datetime(frame["ended_at"], errors="coerce", utc=True)
    seconds = times.astype("int64").to_numpy() / 1e9
    seconds[times.isna().to_numpy()] = np.nan
    dated = np.flatnonzero(np.isfinite(seconds))
    chronological = dated[np.argsort(seconds[dated], kind="stable")]

    rows = []
    for k_cap in k_caps:
        base_source, base_target, base_distance = graph_edges(graph, k_cap)
        finite = np.isfinite(base_distance)
        base_source, base_target = base_source[finite], base_target[finite]
        base_distance = base_distance[finite].astype(np.float64)
        for width in time_neighbors:
            if width == 0:
                source, target, distance = base_source, base_target, base_distance
            else:
                # Each document paired with the `width` documents that follow it in time; the
                # symmetric partner comes from the earlier document's own window, so the union is
                # every pair within `width` positions of each other.
                offsets = np.arange(1, width + 1)
                left = np.repeat(chronological[:-1], len(offsets))
                positions = (np.repeat(np.arange(len(chronological) - 1), len(offsets))
                             + np.tile(offsets, len(chronological) - 1))
                keep = positions < len(chronological)
                left, right = left[keep], chronological[positions[keep]]
                # Cosine for exactly these pairs -- one row-wise dot product, not a matrix.
                extra = 1.0 - np.einsum("ij,ij->i", features[left], features[right]).astype(np.float64)
                source = np.concatenate([base_source, np.minimum(left, right)])
                target = np.concatenate([base_target, np.maximum(left, right)])
                distance = np.concatenate([base_distance, extra])
                # One pair can be proposed by both routes; keep it once, at its true distance.
                _, unique = np.unique(source.astype(np.int64) * graph.n_documents + target,
                                      return_index=True)
                source, target, distance = source[unique], target[unique], distance[unique]

            if weight > 0:
                def standardise(values):
                    centre, spread = values.mean(), values.std()
                    return (values - centre) / (spread if spread > 0 else 1.0)
                gap = np.abs(seconds[source] - seconds[target]) / 3600.0
                gap[~np.isfinite(gap)] = np.nanmax(gap[np.isfinite(gap)])
                scored = ((1 - weight) * standardise(distance)
                          + weight * standardise(np.log1p(gap)))
            else:
                scored = distance

            for row in budget_frontier(source, target, scored, author_codes, n_authors,
                                       graph.n_documents):
                rows.append({"projection": projection_name,
                             "rescoring": f"time{weight:g}+cand{width}", "algorithm": "connected",
                             "locality": 0, "k": k_cap, "time_neighbors": width,
                             "n_candidates": len(source), **row})
            best = max(row["bcubed_f"] for row in rows
                       if row["k"] == k_cap and row["time_neighbors"] == width)
            print(f"    k={k_cap:<3d} time_neighbors={width:<3d} candidates={len(source):>9,} "
                  f"best F={best:.4f}", flush=True)
    return rows



#: Pair features the learned edge model reads. Every one is symmetric in the two endpoints and
#: *relative* rather than absolute -- a rank, an overlap count, a gap, a distance against the
#: endpoints' own neighbourhood radii -- which is what lets a model fitted on the history slice
#: transfer to a different collection of a different size. An absolute feature (a raw timestamp,
#: a document index) would fit the history window and mean nothing outside it.
PAIR_FEATURE_NAMES = ("cosine", "log_time_gap_hours", "shared_neighbors", "rank_min", "rank_max",
                      "same_language", "radius_min", "radius_max", "margin_min", "margin_max")


def pair_features(graph, frame, k_cap: int):
    """``(source, target, X)`` -- candidate edges of one collection with their pair features.

    Built once per collection and shared by the fit and the application, so the history side and
    the collection under attack are described in exactly the same terms.
    """
    from scipy.sparse import csr_matrix

    view = graph.truncate(k_cap)
    source, target, distance = view.edges()
    finite = np.isfinite(distance)
    source, target = source[finite].astype(np.int64), target[finite].astype(np.int64)
    distance = distance[finite].astype(np.float64)
    n = view.n_documents
    valid = np.isfinite(view.distances)

    rows = np.repeat(np.arange(n, dtype=np.int64), view.k)[valid.ravel()]
    columns = view.indices.ravel().astype(np.int64)[valid.ravel()]
    positions = np.tile(np.arange(1, view.k + 1), n)[valid.ravel()]

    # Shared neighbours for every candidate edge at once: one sparse product, where a per-edge set
    # intersection would be a Python loop over ~900,000 pairs.
    adjacency = csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, columns)), shape=(n, n))
    overlap = (adjacency @ adjacency.T).tocsr()
    shared = np.asarray(overlap[source, target]).ravel()

    rank = csr_matrix((positions.astype(np.float32), (rows, columns)), shape=(n, n)).tocsr()
    forward = np.asarray(rank[source, target]).ravel()
    backward = np.asarray(rank[target, source]).ravel()
    forward[forward == 0] = view.k + 1
    backward[backward == 0] = view.k + 1

    radius = np.where(valid, view.distances, np.nan)
    radius = np.nanmean(radius, axis=1)
    radius = np.nan_to_num(radius, nan=float(np.nanmax(radius)))

    times = pd.to_datetime(frame["ended_at"], errors="coerce", utc=True)
    seconds = times.astype("int64").to_numpy() / 1e9
    seconds[times.isna().to_numpy()] = np.nan
    gap = np.abs(seconds[source] - seconds[target]) / 3600.0
    gap[~np.isfinite(gap)] = np.nanmax(gap[np.isfinite(gap)]) if np.isfinite(gap).any() else 0.0

    languages = frame["language_primary"].astype(str).to_numpy()
    margin_a, margin_b = distance - radius[source], distance - radius[target]

    features = np.column_stack([
        distance,
        np.log1p(gap),
        shared / max(view.k, 1),
        np.minimum(forward, backward),
        np.maximum(forward, backward),
        (languages[source] == languages[target]).astype(np.float64),
        np.minimum(radius[source], radius[target]),
        np.maximum(radius[source], radius[target]),
        np.minimum(margin_a, margin_b),
        np.maximum(margin_a, margin_b),
    ])
    return source, target, features


def to_quantiles(values: np.ndarray) -> np.ndarray:
    """Each column replaced by its within-collection empirical quantile, in [0, 1]."""
    out = np.empty_like(values, dtype=np.float64)
    for column in range(values.shape[1]):
        out[:, column] = pd.Series(values[:, column]).rank(method="average", pct=True).to_numpy()
    return out


def pair_model_rows(graph, eval_frame, history_graph, history_frame, author_codes, n_authors,
                    projection_name, k_caps, normalize: bool = False):
    """Fit a same-author model on history candidate edges, apply it to the eval graph's.

    The diagnostic measured five signals on the candidate edges and every one of them carries
    something: cosine 0.77 AUROC, time gap 0.76, shared neighbours 0.72, reciprocal rank 0.65,
    same-language 0.53. Only two have ever been mixed here, by a hand-tuned scalar weight. This
    asks whether a model that sees all of them, with their interactions, does better -- and it is
    the honest way to combine them, because the mixing is fitted on labelled history rather than
    chosen against the slice the result is reported on.

    Distinct from the failed :mod:`~prompt_anonymity.attacks.clustering.edge_scoring`
    cross-encoder, which read the two 1,024-dimensional vectors and had 2M parameters to overfit
    with. This reads ten scalars.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier

    _, history_codes = np.unique(history_frame["author_id"].to_numpy(), return_inverse=True)
    history_codes = history_codes.ravel()

    rows = []
    for k_cap in k_caps:
        source, target, train_x = pair_features(history_graph, history_frame, k_cap)
        train_y = (history_codes[source] == history_codes[target]).astype(np.int64)
        if normalize:
            train_x = to_quantiles(train_x)
        started = time.perf_counter()
        model = HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.1, max_leaf_nodes=31, l2_regularization=1.0,
            early_stopping=True, validation_fraction=0.15, random_state=20260814).fit(
                train_x, train_y)
        print(f"    k={k_cap:<3d} fitted on {len(train_y):,} history edges "
              f"({train_y.mean():.3f} positive) in {time.perf_counter() - started:.0f}s; "
              f"importances via permutation skipped", flush=True)

        eval_source, eval_target, eval_x = pair_features(graph, eval_frame, k_cap)
        if normalize:
            eval_x = to_quantiles(eval_x)
        # Score in log-odds, not probability: the ordering is the same but float32 sigmoid
        # saturates and ties the confident edges, which is the trap the cross-encoder fell into.
        scored = -model.decision_function(eval_x)
        for row in budget_frontier(eval_source, eval_target, scored, author_codes, n_authors,
                                   graph.n_documents):
            rows.append({"projection": projection_name,
                         "rescoring": "pair_model_q" if normalize else "pair_model",
                         "algorithm": "connected", "locality": 0, "k": k_cap, **row})
        print(f"    k={k_cap:<3d} best F="
              f"{max(r['bcubed_f'] for r in rows if r['k'] == k_cap):.4f}", flush=True)
    return rows


def summarize(output_dir: Path) -> pd.DataFrame:
    """Leaderboard over every ``frontier_*.csv`` this directory holds.

    The stages are run separately -- different machines, different queues, hours apart -- so the
    only place they can be compared is here. Rows are keyed by everything that distinguishes a
    method, and the reported score is the best that method reached anywhere in its own sweep. That
    is the selection-optimistic number the module docstring warns about, and it is the right one
    for *ranking* candidates; it is not an estimate of what any of them scores out of sample.
    """
    frames = []
    for path in sorted(output_dir.glob("frontier_*.csv")):
        table = pd.read_csv(path)
        if "edge_budget" not in table.columns and "algorithm" not in table.columns:
            print(f"  skipping {path.name}: not a frontier sweep (no edge_budget or algorithm)")
            continue
        table.insert(0, "run", path.stem.replace("frontier_", ""))
        frames.append(table)
    if not frames:
        raise SystemExit(f"no frontier_*.csv under {output_dir}; run a stage first.")

    table = pd.concat(frames, ignore_index=True)
    table = table[table["bcubed_f"].notna()]
    if "algorithm" not in table.columns:
        table["algorithm"] = "connected"
    table["algorithm"] = table["algorithm"].fillna("connected")
    keys = ["run", "projection", "algorithm", "rescoring"]
    for key in keys:
        if key not in table.columns:
            table[key] = ""
        table[key] = table[key].fillna("")
    return (table.sort_values("bcubed_f", ascending=False)
                 .groupby(keys, as_index=False).head(1)
                 .sort_values("bcubed_f", ascending=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search for a better clustering attack on the tuning slice only.")
    parser.add_argument("--source", default="wildchat")
    parser.add_argument("--feature", default="gemini_embedding_2")
    parser.add_argument("--defense", default="base")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--slice", default="eval", choices=sorted(SLICES),
                        help="Which slice to SCORE on. Anything but 'eval' needs the safety flag.")
    parser.add_argument("--i-am-done-developing", action="store_true",
                        help="Required to score on the test quarter. Every configuration choice "
                             "made on the eval slice is spent the moment this is passed.")
    parser.add_argument("--stage", default="graph",
                        choices=["graph", "projection", "contrastive", "algorithms",
                                 "edge_scorer", "seen_unseen", "linkage", "temporal",
                                 "zscore", "cohesion", "combined", "time_candidates",
                                 "pair_model", "temporal_graph", "all"])
    parser.add_argument("--time-neighbors", nargs="*", type=int, default=[0, 2, 5, 10, 20],
                        help="Temporally-adjacent documents added as candidate edges per document "
                             "in --stage time_candidates; 0 is the cosine-only control.")
    parser.add_argument("--cohesion-quantiles", nargs="*", type=float,
                        default=[0.005, 0.01, 0.02, 0.04, 0.08, 0.15, 0.3, 1.0],
                        help="Cohesion ceilings for --stage cohesion, as quantiles of the edge "
                             "distance distribution; 1.0 is unconstrained (the control).")
    parser.add_argument("--per-language-threshold", action="store_true",
                        help="Standardise edge scores within each language pair before the budget "
                             "cut, so one global budget becomes a language-adaptive threshold. "
                             "Motivated by the error analysis: at one shared operating point "
                             "Korean is over-merged (P 0.37, R 0.90) and Russian under-merged "
                             "(P 0.76, R 0.37), so no single cut is right for both.")
    parser.add_argument("--pair-model-normalize", action="store_true",
                        help="Replace each pair feature by its within-collection quantile before "
                             "fitting and applying. The history side is twice the size of the "
                             "collection under attack and its k-NN graph is correspondingly "
                             "denser (edge precision 0.299 against 0.35), so absolute features "
                             "like the raw cosine and the neighbourhood radius do not mean the "
                             "same thing on both; a quantile does.")
    parser.add_argument("--ensemble-projections", nargs="*", type=Path, default=[],
                        help="Extra projection matrices to concatenate with --projection-file. "
                             "Each block is unit-normalised and scaled by 1/sqrt(n), so cosine in "
                             "the stacked space is the mean of the per-projection cosines.")
    parser.add_argument("--best-time-weight", type=float, default=0.4,
                        help="Time weight --stage combined fuses in, chosen on the tuning slice "
                             "by --stage temporal.")
    parser.add_argument("--caps", nargs="*", type=int,
                        default=[0, 2, 3, 5, 8, 16, 32, 64, 128, 256],
                        help="Maximum cluster sizes for --stage linkage; 0 is unconstrained "
                             "single linkage, i.e. the control.")
    parser.add_argument("--linkage-rules", nargs="*", default=["both", "cap"])
    parser.add_argument("--time-weights", nargs="*", type=float,
                        default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0],
                        help="Weight on the time-gap term for --stage temporal; 0.0 is pure "
                             "cosine (the control) and 1.0 is timing alone (the other control).")
    parser.add_argument("--edge-scorer-hidden", nargs="*", type=int, default=[512])
    parser.add_argument("--edge-scorer-steps", nargs="*", type=int, default=[4000])
    parser.add_argument("--edge-scorer-dropout", nargs="*", type=float, default=[0.1])
    parser.add_argument("--extra-feature", default=None,
                        help="A second feature parquet to fuse with --feature (see fuse_features).")
    parser.add_argument("--extra-feature-weight", type=float, default=1.0,
                        help="Weight of --extra-feature's cosine in the fused similarity; "
                             "--feature always carries 1.0.")
    parser.add_argument("--projection-file", type=Path, default=None,
                        help="Load a saved projection matrix instead of fitting one. Required by "
                             "--stage algorithms unless --stage all refits it in the same run.")
    parser.add_argument("--rescorings", nargs="*",
                        default=["none", "csls", "local_scaling", "mutual_knn"])
    parser.add_argument("--k-caps", nargs="*", type=int, default=list(DEFAULT_K_CAPS))
    parser.add_argument("--localities", nargs="*", type=int, default=[5, 10, 25])
    parser.add_argument("--graph-width", type=int, default=GRAPH_WIDTH)
    parser.add_argument("--wccn-shrinkage", nargs="*", type=float, default=[0.01, 0.1, 0.5])
    parser.add_argument("--lda-components", nargs="*", type=int, default=[64, 256, 1024])
    parser.add_argument("--contrastive-components", nargs="*", type=int, default=[256, 512])
    parser.add_argument("--contrastive-steps", nargs="*", type=int, default=[4000])
    parser.add_argument("--contrastive-temperature", nargs="*", type=float, default=[0.05, 0.1])
    parser.add_argument("--contrastive-batch", nargs="*", type=int, default=[512])
    parser.add_argument("--contrastive-hidden", nargs="*", type=int, default=[0])
    parser.add_argument("--contrastive-lr", nargs="*", type=float, default=[1e-3])
    parser.add_argument("--contrastive-hard-negatives", nargs="*", type=int, default=[0],
                        help="Refresh interval in steps for hard-negative batch construction; "
                             "0 keeps random batches.")
    parser.add_argument("--from-run", default=None,
                        help="For --stage seen_unseen: take (k, edge_budget) from this run tag's "
                             "best row only. Needed because a --extra-feature run also reports "
                             "its projection as 'identity'.")
    parser.add_argument("--run-tag", default=None,
                        help="Suffix for this run's output files, so two runs of one stage "
                             "(e.g. a fused-feature sweep and a plain one) do not overwrite "
                             "each other. Defaults to the stage name alone.")
    parser.add_argument("--summarize", action="store_true",
                        help="Print the leaderboard over every frontier_*.csv already written "
                             "and exit. Runs nothing.")
    parser.add_argument("--verify-bcubed", action="store_true",
                        help="Check fast_bcubed against the library implementation and exit.")
    args = parser.parse_args()
    args.run_tag = args.run_tag or args.stage
    if args.slice != "eval" and not args.i_am_done_developing:
        raise SystemExit(
            f"--slice {args.slice} scores on a held-out slice. This script is the development "
            f"loop; every configuration here was chosen by looking at 'eval'. Pass "
            f"--i-am-done-developing if that is really what you want.")
    return args


def main() -> None:
    args = parse_args()
    tag = f"{args.source}_{args.defense}_{args.feature}"
    output_dir = args.output_dir or (OUTPUT_ROOT / tag)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = CACHE_ROOT / tag
    cache_dir.mkdir(parents=True, exist_ok=True)

    if args.summarize:
        best = summarize(output_dir)
        columns = [c for c in ("run", "projection", "algorithm", "rescoring", "k", "edge_budget",
                               "distance_threshold", "bcubed_f", "bcubed_precision",
                               "bcubed_recall", "n_clusters", "largest_cluster_share")
                   if c in best.columns]
        pd.set_option("display.width", 200, "display.max_colwidth", 62)
        print(best[columns].head(30).to_string(index=False))
        best.to_csv(output_dir / "leaderboard.csv", index=False)
        print(f"\nWrote {output_dir}/leaderboard.csv ({len(best)} methods)")
        return

    slices, windows = load_slices(args)
    eval_frame, eval_features = slices[args.slice]
    history_frame, history_features = slices["history"]
    eval_authors = eval_frame["author_id"].to_numpy()
    _, author_codes = np.unique(eval_authors, return_inverse=True)
    author_codes = author_codes.ravel()
    n_authors = int(author_codes.max()) + 1

    print(f"{tag}\n  scoring on '{args.slice}' {SLICES[args.slice]}: "
          f"{len(eval_frame):,} documents, {n_authors:,} authors")
    print(f"  fitting on 'history' {SLICES['history']}: {len(history_frame):,} documents, "
          f"{history_frame['author_id'].nunique():,} authors")

    if args.verify_bcubed:
        rng = np.random.default_rng(0)
        labels = rng.integers(0, 500, size=len(eval_authors))
        mine = fast_bcubed(labels, author_codes, n_authors)
        theirs = bcubed_scores(labels, eval_authors)
        print(f"  fast_bcubed  P={mine[0]:.10f} R={mine[1]:.10f} F={mine[2]:.10f}")
        print(f"  library      P={theirs.precision:.10f} R={theirs.recall:.10f} "
              f"F={theirs.f_score:.10f}")
        print(f"  max |diff| = {max(abs(mine[0] - theirs.precision), abs(mine[1] - theirs.recall), abs(mine[2] - theirs.f_score)):.3e}")
        return

    projections: list[LinearProjection] = []
    if args.stage in ("graph", "all"):
        projections.append(identity_projection(eval_features.shape[1]))
    if args.stage in ("projection", "all"):
        for shrinkage in args.wccn_shrinkage:
            started = time.perf_counter()
            projections.append(fit_projection("wccn", history_features,
                                              history_frame["author_id"].to_numpy(),
                                              shrinkage=shrinkage))
            print(f"  fitted {projections[-1].name} in {time.perf_counter() - started:.0f}s",
                  flush=True)
        for components in args.lda_components:
            started = time.perf_counter()
            projections.append(fit_projection("lda", history_features,
                                              history_frame["author_id"].to_numpy(),
                                              n_components=components))
            print(f"  fitted {projections[-1].name} in {time.perf_counter() - started:.0f}s",
                  flush=True)
    if args.stage in ("contrastive", "all"):
        from itertools import product
        for components, temperature, steps, batch, hidden, rate, hard in product(
                args.contrastive_components, args.contrastive_temperature,
                args.contrastive_steps, args.contrastive_batch, args.contrastive_hidden,
                args.contrastive_lr, args.contrastive_hard_negatives):
            started = time.perf_counter()
            fitted = fit_projection(
                "contrastive", history_features, history_frame["author_id"].to_numpy(),
                n_components=components, steps=steps, temperature=temperature,
                authors_per_batch=batch, hidden=hidden, learning_rate=rate,
                hard_negatives=hard, verbose=False)
            # The default name records only some of the grid; make every swept axis visible, since
            # two rows of the results table that differ only in an unnamed parameter are unusable.
            fitted.name = (f"contrastive(d={components}, t={temperature}, steps={steps}, "
                           f"batch={batch}, hidden={hidden}, lr={rate}, hard={hard})")
            projections.append(fitted)
            print(f"  fitted {fitted.name} in {time.perf_counter() - started:.0f}s", flush=True)
    if args.projection_file is not None:
        matrices = [np.load(args.projection_file)] + [np.load(path)
                                                      for path in args.ensemble_projections]
        if len(matrices) == 1:
            projections.append(LinearProjection(matrices[0], args.projection_file.stem))
        else:
            # Column-stacking L2-normalised blocks scaled by 1/sqrt(n) makes cosine in the stacked
            # space exactly the mean of the per-projection cosines -- the same identity
            # `fuse_features` uses for two featurizers, applied to two views of one.
            projections.append(EnsembleProjection(matrices,
                                                  f"ensemble{len(matrices)}({args.projection_file.stem[:28]})"))
    if not projections:
        # `--stage algorithms` fits nothing of its own, so with no `--projection-file` it would
        # otherwise sweep an empty list and fail at the summary. The base space is the right
        # default: it is the comparison every algorithm result has to be read against.
        projections.append(identity_projection(eval_features.shape[1]))

    matrix_dir = output_dir / "projections"
    matrix_dir.mkdir(exist_ok=True)
    rows = []
    for projection in projections:
        print(f"\n  == {projection.name} ({projection.n_components} components)", flush=True)
        # An EnsembleProjection holds several matrices and has no single one to save; it is
        # reproduced from its component files instead, which is why the flag takes paths.
        if projection.name != "identity" and hasattr(projection, "matrix"):
            np.save(matrix_dir / f"{projection.name}.npy", projection.matrix)
        projected = (eval_features if projection.name == "identity"
                     else projection.transform(eval_features))
        graph = cached_graph(projected, args.graph_width, args.slice, cache_dir)
        if args.stage == "seen_unseen":
            # (k, budget) comes from this projection's own best row in the frontier tables
            # already written, so the partition analysed is exactly the one that was reported.
            board = summarize(output_dir)
            match = board[board["projection"] == projection.name]
            if args.from_run:
                match = match[match["run"] == args.from_run]
            if match.empty:
                raise SystemExit(f"no frontier row for {projection.name!r}"
                                 + (f" in run {args.from_run!r}" if args.from_run else "")
                                 + "; run its stage first.")
            top = match.iloc[0]
            # `projection` alone does NOT identify a feature space -- a --extra-feature run is
            # also called "identity" -- so the run this (k, budget) came from is printed, and
            # --from-run pins it. Taking a fused run's budget onto the raw graph is a silent
            # mismatch that cost 0.001 of BCubed F before this was added.
            print(f"    using k={int(top['k'])}, budget={int(top['edge_budget']):,} "
                  f"from run {top['run']!r} (F={top['bcubed_f']:.4f} there)", flush=True)
            labels = components_at_budget(graph, int(top["k"]), int(top["edge_budget"]))
            rows.extend(seen_unseen_report(labels, author_codes, n_authors, eval_authors,
                                           history_frame["author_id"].to_numpy(),
                                           projection.name))
            continue
        if args.stage == "temporal_graph":
            # The SAME code path run_clustering.py uses, so this checks that the graph-level
            # transform reproduces the edge-level sweep that selected the weight. The two
            # standardise over slightly different populations -- the (n, k) array counts a mutual
            # pair twice where the deduplicated edge list counts it once -- so they are expected
            # to agree closely rather than exactly.
            seconds = (pd.to_datetime(eval_frame["ended_at"], errors="coerce", utc=True)
                       .astype("int64").to_numpy() / 1e9)
            seconds[pd.to_datetime(eval_frame["ended_at"], errors="coerce",
                                   utc=True).isna().to_numpy()] = np.nan
            for weight in args.time_weights:
                view = temporal_fusion(graph, seconds, weight) if weight > 0 else graph
                for k_cap in args.k_caps:
                    source, target, distance = graph_edges(view, k_cap)
                    finite = np.isfinite(distance)
                    for row in budget_frontier(source[finite], target[finite], distance[finite],
                                               author_codes, n_authors, graph.n_documents):
                        rows.append({"projection": projection.name,
                                     "rescoring": f"graphtime{weight:g}", "algorithm": "connected",
                                     "locality": 0, "k": k_cap, **row})
                    print(f"    w={weight:<5g} k={k_cap:<3d} best F="
                          f"{max(r['bcubed_f'] for r in rows if r['k'] == k_cap and r['rescoring'] == f'graphtime{weight:g}'):.4f}",
                          flush=True)
            continue
        if args.stage == "pair_model":
            history_projected = (history_features if projection.name == "identity"
                                 else projection.transform(history_features))
            history_graph = cached_graph(history_projected, args.graph_width, "history", cache_dir)
            rows.extend(pair_model_rows(graph, eval_frame, history_graph, history_frame,
                                        author_codes, n_authors, projection.name, args.k_caps,
                                        args.pair_model_normalize))
            continue
        if args.stage == "time_candidates":
            rows.extend(time_candidate_rows(
                graph, eval_frame, projected, author_codes, n_authors, projection.name,
                args.k_caps, args.time_neighbors, args.best_time_weight))
            continue
        if args.stage in ("cohesion", "combined"):
            rows.extend(constrained_rows(
                graph, eval_frame, author_codes, n_authors, projection.name, args.k_caps,
                args.cohesion_quantiles, args.caps, args.linkage_rules,
                args.best_time_weight if args.stage == "combined" else 0.0, args.stage))
            continue
        if args.stage == "linkage":
            rows.extend(linkage_rows(graph, author_codes, n_authors, projection.name,
                                     args.k_caps, args.caps, args.linkage_rules))
            continue
        if args.stage == "temporal":
            rows.extend(temporal_rows(graph, eval_frame, author_codes, n_authors,
                                      projection.name, args.k_caps, args.time_weights,
                                      args.per_language_threshold))
            continue
        if args.stage == "zscore":
            mean, spread = global_distance_moments(projected)
            print(f"    global distance moments: mean {mean.mean():.4f}, "
                  f"std {spread.mean():.4f}", flush=True)
            view = zscore_distances(graph, mean, spread)
            for k_cap in args.k_caps:
                source, target, distance = graph_edges(view, k_cap)
                finite = np.isfinite(distance)
                for row in budget_frontier(source[finite], target[finite], distance[finite],
                                           author_codes, n_authors, graph.n_documents):
                    rows.append({"projection": projection.name, "rescoring": "zscore",
                                 "algorithm": "connected", "locality": 0, "k": k_cap, **row})
                print(f"    k={k_cap:<3d} best F="
                      f"{max(r['bcubed_f'] for r in rows if r['k'] == k_cap):.4f}", flush=True)
            continue
        if args.stage == "edge_scorer":
            # The scorer reads the projected space, so it is fitted on the projected HISTORY --
            # the same slice and the same rule as the projection itself.
            history_projected = (history_features if projection.name == "identity"
                                 else projection.transform(history_features))
            from itertools import product as _product
            for hidden, steps, dropout in _product(args.edge_scorer_hidden,
                                                   args.edge_scorer_steps,
                                                   args.edge_scorer_dropout):
                started = time.perf_counter()
                scorer = fit_edge_scorer(history_projected,
                                         history_frame["author_id"].to_numpy(),
                                         hidden=hidden, steps=steps, dropout=dropout)
                rescored = scorer.rescore(graph, projected)
                print(f"  {scorer.name} fitted+scored in "
                      f"{time.perf_counter() - started:.0f}s", flush=True)
                for k_cap in args.k_caps:
                    source, target, distance = graph_edges(rescored, k_cap)
                    finite = np.isfinite(distance)
                    for row in budget_frontier(source[finite], target[finite], distance[finite],
                                               author_codes, n_authors, graph.n_documents):
                        rows.append({"projection": projection.name,
                                     "rescoring": scorer.name, "locality": 0,
                                     "k": k_cap, **row})
                best = max(r["bcubed_f"] for r in rows if r["rescoring"] == scorer.name)
                print(f"    best F={best:.4f}", flush=True)
            continue
        if args.stage == "algorithms":
            rows.extend(sweep_algorithms(graph, author_codes, n_authors, projection.name,
                                         tuple(args.k_caps)))
            continue
        rows.extend(sweep_graph(graph, author_codes, n_authors, projection.name,
                                tuple(args.rescorings), tuple(args.k_caps),
                                tuple(args.localities)))

    table = pd.DataFrame(rows)
    table.insert(0, "slice", args.slice)

    if args.stage == "seen_unseen":
        # Deliberately NOT named frontier_*: these rows are a decomposition of one partition, not
        # a sweep of methods, and summarize() would otherwise rank the "unseen" subgroup against
        # real results. It did, once, and put a subgroup at the top of the leaderboard.
        table.to_csv(output_dir / f"{args.run_tag}.csv", index=False)
        print(table.to_string(index=False))
        print(f"\nWrote {output_dir}/{args.run_tag}.csv")
        return

    table.to_csv(output_dir / f"frontier_{args.run_tag}.csv", index=False)

    group = ["projection", "algorithm"] if args.stage == "algorithms" else ["projection", "rescoring"]
    columns = (["projection", "algorithm", "settings"] if args.stage == "algorithms"
               else ["projection", "rescoring", "locality", "k", "edge_budget",
                     "distance_threshold"])
    best = (table.sort_values("bcubed_f", ascending=False)
                 .groupby(group, as_index=False).head(1)
                 .sort_values("bcubed_f", ascending=False))
    best.to_csv(output_dir / f"best_{args.run_tag}.csv", index=False)
    print(f"\nBest per {tuple(group)}, scored on '{args.slice}':")
    print(best[columns + ["bcubed_f", "bcubed_precision", "bcubed_recall", "n_clusters",
                          "largest_cluster_share"]].to_string(index=False))
    print(f"\nWrote {output_dir}/frontier_{args.run_tag}.csv and best_{args.run_tag}.csv")


if __name__ == "__main__":
    main()
