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
                                 "edge_scorer", "seen_unseen", "all"])
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
        projections.append(LinearProjection(np.load(args.projection_file),
                                            args.projection_file.stem))
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
        if projection.name != "identity":
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
