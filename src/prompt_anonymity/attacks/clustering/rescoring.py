"""Hubness corrections on the neighbour graph: the same vectors, a better-behaved distance.

Every method in :mod:`.algorithms` reads a
:class:`~prompt_anonymity.attacks.clustering.graph.NeighborGraph` and nothing else, so the cheapest
place to improve a clustering attack is not the algorithm but the *edge weights it is handed*.
This module rewrites those weights in place -- same candidate edges, same vectors, a different
notion of "close".

Why this family and not another
-------------------------------
The failure ``connected`` exhibits is chaining, and chaining in a high-dimensional embedding is
usually **hubness**: as dimension grows, a small number of points become the nearest neighbour of
disproportionately many others, for reasons that have nothing to do with the similarity being
measured (Radovanovic et al., JMLR 2010). A hub is a bridge, and single linkage propagates through
bridges. Raising the distance threshold rations chaining without addressing why the bridges are
there; these transforms make a hub's edges *expensive*, which is a different axis from the
threshold and therefore the one worth trying first.

All three are the standard corrections:

===================  ===================================================================
:func:`csls`         Cross-domain Similarity Local Scaling (Conneau et al., ICLR 2018) --
                     subtract each endpoint's mean distance to its own neighbourhood, so
                     a point that is close to everything gains nothing from being close
:func:`local_scaling`  Zelnik-Manor & Perona (NIPS 2004) -- divide by each endpoint's own
                     neighbourhood radius, making the distance scale local rather than
                     global
:func:`mutual_knn`   Keep an edge only when both endpoints list the other among their own
                     ``k`` nearest. The oldest fix and the bluntest: a hub is listed by
                     many and lists few, so most of its edges vanish
===================  ===================================================================

What is exact here and what is an approximation
-----------------------------------------------
:func:`mutual_knn` is exact -- it only deletes edges, and the information it needs (each
document's own neighbour list) is entirely in the graph.

:func:`csls` and :func:`local_scaling` are **re-rankings of a fixed candidate set**. The true
k-nearest neighbours under a rescaled distance need not be the k nearest under the original one,
and recovering them exactly would mean another full pass over the pairwise matrix. Instead the
graph is built wide (``k=100``) under cosine, rescaled, and re-sorted within those candidates.
The approximation is one-sided and its direction is known: a pair that the rescaling would have
promoted from outside the top 100 is missed, so a measured gain is real and a measured null may
be understating the transform. Building wide is what keeps that gap small.

**The output is no longer a metric**, and nothing here pretends otherwise. CSLS distances are
routinely negative and satisfy no triangle inequality. That is harmless for every consumer in this
package -- they compare against a threshold or feed a graph algorithm -- but it does mean a
threshold tuned under cosine is meaningless after rescaling, so :attr:`NeighborGraph.metric` is
stamped with the transform's name and thresholds must be re-swept. Sweeping an *edge budget*
rather than an absolute threshold (``experiments/improve_clustering.py``) sidesteps the issue
entirely and is how these variants are compared.
"""

from __future__ import annotations

import numpy as np

from .graph import NeighborGraph

#: Neighbours each transform summarises an endpoint's own neighbourhood over. Ten is the value
#: CSLS was introduced with and is well inside the range where the estimate is stable; local
#: scaling's original paper uses the 7th neighbour, which is the same order.
DEFAULT_LOCALITY = 10


def _resort(indices: np.ndarray, distances: np.ndarray, metric: str) -> NeighborGraph:
    """Re-order each row by the rewritten distance, keeping ``inf`` padding at the end.

    Every consumer relies on :class:`NeighborGraph` storing neighbours in increasing distance
    order -- ``truncate`` is a column slice, and the graph diagnostics read column 0 as the
    nearest neighbour. A transform that reorders distances without reordering rows would break
    both silently, which is why this is not left to the caller.
    """
    order = np.argsort(distances, axis=1, kind="stable")
    return NeighborGraph(np.take_along_axis(indices, order, axis=1).astype(np.int32),
                         np.take_along_axis(distances, order, axis=1).astype(np.float32),
                         metric)


def neighborhood_radius(graph: NeighborGraph, locality: int = DEFAULT_LOCALITY) -> np.ndarray:
    """Mean distance from each document to its ``locality`` nearest neighbours.

    The per-point quantity both scaling transforms are built on: small for a document sitting in
    a crowded region (a hub, or a member of a tight duplicate cluster), large for an isolated one.
    Padding entries are excluded, so a document with fewer than ``locality`` finite neighbours is
    averaged over what it has; a document with none gets ``nan``, which
    :func:`csls` and :func:`local_scaling` turn back into "no edge".
    """
    width = min(locality, graph.k)
    head = graph.distances[:, :width].astype(np.float64)
    finite = np.isfinite(head)
    counts = finite.sum(axis=1)
    totals = np.where(finite, head, 0.0).sum(axis=1)
    return np.where(counts > 0, totals / np.maximum(counts, 1), np.nan)


def csls(graph: NeighborGraph, locality: int = DEFAULT_LOCALITY) -> NeighborGraph:
    """Cross-domain Similarity Local Scaling: subtract both endpoints' neighbourhood radius.

    ``d'(i, j) = d(i, j) - (r_i + r_j) / 2``, with ``r`` from :func:`neighborhood_radius`. In
    similarity terms this is Conneau et al.'s ``2 cos(i, j) - r_i - r_j`` up to the factor of two,
    which cannot change any ranking or any partition and keeps the output on the scale of a
    distance.

    The correction is exactly the one hubness calls for: a document that is close to *everything*
    has a small ``r`` and therefore has to be closer still before an edge counts, while a genuine
    pair in a sparse region is no longer punished for living somewhere empty. Distances become
    negative for pairs closer than either endpoint's typical neighbour, which is intended -- the
    quantity is a contrast, not a length.
    """
    radius = neighborhood_radius(graph, locality)
    rescaled = (graph.distances.astype(np.float64)
                - 0.5 * (radius[:, None] + radius[graph.indices]))
    # A padding entry has no partner and must not become the *most* attractive edge by way of a
    # NaN sorting first; put it back where it belongs.
    rescaled[~np.isfinite(graph.distances)] = np.inf
    rescaled[np.isnan(rescaled)] = np.inf
    return _resort(graph.indices, rescaled, f"csls{locality}({graph.metric})")


def local_scaling(graph: NeighborGraph, locality: int = DEFAULT_LOCALITY) -> NeighborGraph:
    """Zelnik-Manor & Perona local scaling: divide by both endpoints' neighbourhood radius.

    ``d'(i, j) = d(i, j)^2 / (r_i r_j)``. The multiplicative counterpart to :func:`csls`, and the
    difference is not cosmetic: dividing makes the transform scale-free, so it is insensitive to
    the overall spread of the embedding but reacts more violently to a document whose
    neighbourhood radius is near zero -- which on this corpus means near-duplicate text. Both are
    swept because which failure mode dominates is an empirical question.

    Radii are floored at a small positive value: an exact-duplicate cluster genuinely has ``r = 0``
    and would otherwise divide by zero, turning the one case where the graph is *most* confident
    into a NaN.
    """
    radius = np.maximum(neighborhood_radius(graph, locality), 1e-6)
    squared = graph.distances.astype(np.float64) ** 2
    rescaled = squared / (radius[:, None] * radius[graph.indices])
    rescaled[~np.isfinite(graph.distances)] = np.inf
    rescaled[np.isnan(rescaled)] = np.inf
    return _resort(graph.indices, rescaled, f"localscale{locality}({graph.metric})")


def mutual_knn(graph: NeighborGraph, locality: int = DEFAULT_LOCALITY) -> NeighborGraph:
    """Drop every edge the two endpoints do not both agree on.

    An edge ``(i, j)`` survives only if ``j`` is among ``i``'s ``locality`` nearest **and** ``i``
    is among ``j``'s. :meth:`NeighborGraph.edges` unions the two neighbour relations, which is the
    right default for coverage and the wrong one for chaining: a hub that 500 documents list as
    their nearest neighbour contributes 500 edges while listing only ``k`` itself, and single
    linkage then merges all 500 into one cluster. The mutual graph keeps at most ``k`` of them.

    Dropped edges become ``inf`` rather than being compacted away, so the result is still a
    rectangular :class:`NeighborGraph` of the same width and every downstream ``truncate`` keeps
    working. Note the surviving distances are **unchanged** -- this is a filter, not a rescaling,
    and it composes with either of the other two.
    """
    width = min(locality, graph.k)
    n = graph.n_documents
    head = graph.indices[:, :width].astype(np.int64)
    finite = np.isfinite(graph.distances[:, :width])

    rows = np.repeat(np.arange(n, dtype=np.int64), width)
    directed = (rows * n + head.ravel())[finite.ravel()]
    directed.sort()

    # An edge is mutual iff the reversed ordered pair is also present in the directed list.
    reversed_key = graph.indices.astype(np.int64) * n + np.arange(n, dtype=np.int64)[:, None]
    position = np.searchsorted(directed, reversed_key)
    np.clip(position, 0, len(directed) - 1, out=position)
    keep = directed[position] == reversed_key
    keep &= np.isfinite(graph.distances)
    # Only the head of each row was offered as a candidate, so nothing beyond it can be mutual.
    keep[:, width:] = False

    distances = np.where(keep, graph.distances, np.inf)
    return _resort(graph.indices, distances, f"mutual{locality}({graph.metric})")


#: Name -> transform, so a driver selects one by string exactly as ``CLUSTERING_ATTACKS`` works
#: for the algorithms. ``none`` is present so "no rescoring" is a value of the same axis rather
#: than a branch in the caller.
GRAPH_RESCORINGS = {
    "none": lambda graph, locality=DEFAULT_LOCALITY: graph,
    "csls": csls,
    "local_scaling": local_scaling,
    "mutual_knn": mutual_knn,
}


def rescore_graph(graph: NeighborGraph, name: str,
                  locality: int = DEFAULT_LOCALITY) -> NeighborGraph:
    """Apply a registered rescoring by name."""
    try:
        transform = GRAPH_RESCORINGS[name]
    except KeyError:
        raise ValueError(f"unknown graph rescoring {name!r}; "
                         f"available: {sorted(GRAPH_RESCORINGS)}") from None
    return transform(graph, locality)
