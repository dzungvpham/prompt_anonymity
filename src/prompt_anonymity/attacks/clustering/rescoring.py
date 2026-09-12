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


def _resort(indices: np.ndarray, distances: np.ndarray, metric: str,
            max_distance: float | None = None) -> NeighborGraph:
    """Re-order each row by the rewritten distance, keeping ``inf`` padding at the end.

    Every consumer relies on :class:`NeighborGraph` storing neighbours in increasing distance
    order -- ``truncate`` is a column slice, and the graph diagnostics read column 0 as the
    nearest neighbour. A transform that reorders distances without reordering rows would break
    both silently, which is why this is not left to the caller.

    ``max_distance`` defaults to ``None`` because it is the right answer for the hubness family:
    the input graph's ceiling does not survive a transform that subtracts a neighbourhood mean,
    and CSLS distances are routinely negative. Only :func:`temporal_fusion`, whose output is
    bounded by construction, passes one.
    """
    order = np.argsort(distances, axis=1, kind="stable")
    return NeighborGraph(np.take_along_axis(indices, order, axis=1).astype(np.int32),
                         np.take_along_axis(distances, order, axis=1).astype(np.float32),
                         metric, max_distance)


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


def shared_neighbors(graph: NeighborGraph, locality: int = DEFAULT_LOCALITY) -> NeighborGraph:
    """Re-rank each edge by how much the two endpoints' neighbourhoods overlap.

    Second-order (Jarvis-Patrick) similarity: two documents are close if they *keep the same
    company*, whatever their direct distance says. It is the cheap relative of Koppel & Winter's
    impostors method -- there, a pair is scored by how often one is the other's nearest neighbour
    across random impostor sets and feature subsets; here the k-nearest-neighbour lists play the
    part of the impostor draw, at no extra distance computation.

    Measured on WildChat's tuning slice, the raw overlap count separates same-author from
    different-author candidate edges at AUROC **0.72** against cosine's 0.77 -- weaker on its own,
    but computed from the graph's *structure* rather than its weights, so what it adds is not what
    cosine already knows.

    Ties are broken by the original distance, which matters more here than for any other transform
    in this module: the overlap is a small integer (0..k), so a pure overlap ordering would leave
    tens of thousands of edges tied and let an arbitrary sort decide the partition.
    """
    width = min(locality, graph.k)
    view = graph.truncate(width)
    valid = np.isfinite(view.distances)
    sets = [frozenset(row[mask].tolist()) for row, mask in zip(view.indices, valid)]

    overlap = np.zeros(graph.indices.shape, dtype=np.float64)
    for row in range(graph.n_documents):
        own = sets[row]
        for column, neighbour in enumerate(graph.indices[row]):
            if np.isfinite(graph.distances[row, column]):
                overlap[row, column] = len(own & sets[neighbour])
    # Negated so smaller is closer, and the distance is folded in at a weight small enough that it
    # only ever breaks ties within one overlap level.
    rescaled = -overlap + graph.distances.astype(np.float64) / (2 * (width + 1))
    rescaled[~np.isfinite(graph.distances)] = np.inf
    return _resort(graph.indices, rescaled, f"snn{locality}({graph.metric})")


def zscore_distances(graph: NeighborGraph, reference_mean: np.ndarray,
                     reference_std: np.ndarray) -> NeighborGraph:
    """Standardise each edge against **both** endpoints' own global distance distribution.

    ``d'(A, B) = max( (d - mu_A)/sigma_A , (d - mu_B)/sigma_B )``.

    This is Kocher's SPATIUM rule (PAN 2017 runner-up, and second at PAN 2016), which asks whether
    a distance is small *relative to the distances that document has to everything else* -- and it
    is a materially different question from the one :func:`csls` asks. CSLS subtracts a **local**
    mean, over the ten nearest neighbours, so it measures local density; this subtracts the
    **global** mean and divides by the global standard deviation, so it measures how unusual the
    pair is for those two documents. The local version was measured here and lost (0.467 against
    0.516); the global one had not been tried.

    ``max`` of the two directions rather than the mean, because SPATIUM requires the evidence to
    hold from both endpoints -- its "at least two of four hints" rule is a conjunction, and taking
    the worse of the two standardised scores is that conjunction expressed as a single number.
    """
    scale = np.where(reference_std > 0, reference_std, 1.0)
    left = (graph.distances.astype(np.float64) - reference_mean[:, None]) / scale[:, None]
    right = ((graph.distances.astype(np.float64) - reference_mean[graph.indices])
             / scale[graph.indices])
    rescaled = np.maximum(left, right)
    rescaled[~np.isfinite(graph.distances)] = np.inf
    return _resort(graph.indices, rescaled, f"zscore({graph.metric})")


def global_distance_moments(embeddings: np.ndarray, metric: str = "cosine",
                            sample: int = 4096, seed: int = 20260814) -> tuple:
    """``(mean, std)`` of each document's distance to the whole collection, from a random sample.

    SPATIUM computes these over every other document, which is free on a 50-document PAN problem
    and 1.9 billion pairs here. A random sample of the collection estimates the same two moments
    to a standard error of ``sigma/sqrt(sample)`` -- at 4,096 that is under 2% of one standard
    deviation, far below the resolution any threshold sweep can use. The sample is shared across
    all rows and seeded, so the transform is deterministic.
    """
    from ..similarity.kernel import blocked_distances

    embeddings = np.asarray(embeddings, dtype=np.float32)
    rows = np.random.default_rng(seed).choice(
        len(embeddings), size=min(sample, len(embeddings)), replace=False)
    total = np.zeros(len(embeddings))
    total_square = np.zeros(len(embeddings))
    count = 0
    for start, block in blocked_distances(embeddings[rows], embeddings, metric=metric):
        total += block.sum(axis=0)
        total_square += (block.astype(np.float64) ** 2).sum(axis=0)
        count += len(block)
    mean = total / count
    return mean, np.sqrt(np.maximum(total_square / count - mean ** 2, 0.0))


#: Fraction of each tail :func:`winsor_bounds` clips, in percent. 0.1% of a ~50,000-edge graph is
#: ~50 edges per side -- enough to be immune to a single outlier (the failure that ruled out plain
#: min-max) while clipping far less than a 3-sigma rule does on a skewed variable. Measured on
#: swe-chat, a 3-sigma clip takes 0.9% off cosine's NEAR tail, where the strongest same-author
#: evidence lives; a quantile clips the same fraction whatever the skew.
WINSOR_PERCENTILE = 0.1

def winsor_bounds(graph: NeighborGraph, seconds: np.ndarray | None
                  ) -> tuple[tuple[float, float], tuple[float, float]]:
    """``((d_lo, d_hi), (t_lo, t_hi))`` -- the winsorising bracket for each term.

    The :data:`WINSOR_PERCENTILE` and its complement, over one graph's own finite edges.

    **Called per graph -- the collection under attack is bracketed by its own quantiles, not the
    tuning slice's.** Fixing the bracket on the tuning slice is the more obviously consistent
    choice (the weight is selected under one scoring function and deployed with it) and it was the
    first implementation, but it loses on both counts that were measured. A bracket clips at a
    fixed *value*, so when the two slices' distributions move it puts real mass on the clip:
    swe-chat's tuning slice ends at 372.9 h against the test collection's 643.5 h, which pinned
    **14.4% of the test collection's time gaps** to exactly 1.0 -- not a tail but a seventh of the
    edges collapsed into one tied value. Per-graph brackets removed that for +0.016 mean BCubed F
    on ``all`` and +0.004 on ``unseen``, and **the selected weight did not change in any of the ten
    (scope x algorithm) cells** -- the search runs on the tuning graph either way, so this changes
    only how the winner is deployed.

    It also matches what the rest of the package already does: :func:`~.algorithms.edge_quantile`
    sets every threshold from the quantiles of *the graph being clustered*. Reading no labels, only
    the anonymous collection's own edge weights, is inside the threat model in both places.

    A degenerate term (every value identical, so ``hi == lo``) is widened to a unit interval rather
    than allowed to divide by zero, which makes that term a constant 0 -- the right "this half
    contributes nothing" behaviour.
    """
    def bracket(values: np.ndarray) -> tuple[float, float]:
        usable = values[np.isfinite(values)]
        if len(usable) == 0:
            return 0.0, 1.0
        low, high = np.percentile(usable, [WINSOR_PERCENTILE, 100.0 - WINSOR_PERCENTILE])
        return (float(low), float(high)) if high > low else (float(low), float(low) + 1.0)

    finite = np.isfinite(graph.distances)
    if seconds is None:
        return bracket(graph.distances[finite]), (0.0, 1.0)
    hours = np.abs(seconds[:, None] - seconds[graph.indices]) / 3600.0
    return bracket(graph.distances[finite]), bracket(hours[finite & np.isfinite(hours)])


def balanced_time_weight(graph: NeighborGraph, seconds: np.ndarray, bounds) -> float:
    """The weight at which both terms of :func:`temporal_fusion` contribute equal spread.

    ``w* = IQR(text) / (IQR(text) + IQR(time))`` over the graph's finite edges, which is the
    centre a weight grid should be laid around. It is not a tuned value and not a default -- it is
    where "half and half" actually falls once both terms are on their own saturating curves, and
    it moves with the corpus. IQR rather than standard deviation because the time gaps have a long
    tail that a variance would chase.
    """
    text, time_term = _fusion_terms(graph, seconds, bounds)
    finite = np.isfinite(graph.distances)
    def iqr(values):
        low, high = np.percentile(values[finite], [25, 75])
        return float(high - low)
    text_spread, time_spread = iqr(text), iqr(time_term)
    total = text_spread + time_spread
    return 0.5 if total <= 0 else text_spread / total


def _squash(values: np.ndarray, bounds: tuple[float, float]) -> np.ndarray:
    """One raw quantity mapped into ``[0, 1]``: linear between ``bounds``, clipped outside them."""
    low, high = bounds
    return np.clip((values - low) / (high - low), 0.0, 1.0)


def _text_term(graph: NeighborGraph, bounds: tuple[float, float]) -> np.ndarray:
    """The text half of the fusion, in ``[0, 1]``."""
    finite = np.isfinite(graph.distances)
    return _squash(np.where(finite, graph.distances, 0.0).astype(np.float64), bounds)


def _time_term(graph: NeighborGraph, seconds: np.ndarray,
               bounds: tuple[float, float]) -> np.ndarray:
    """The elapsed-time half, in ``[0, 1]``."""
    finite = np.isfinite(graph.distances)
    hours = np.abs(seconds[:, None] - seconds[graph.indices]) / 3600.0
    known = finite & np.isfinite(hours)
    # A pair with no usable timestamp is placed at the far end -- treated as maximally far apart --
    # so a corpus with missing times degrades toward the pure-text attack rather than failing.
    gap = _squash(np.where(known, hours, 0.0), bounds)
    gap[finite & ~known] = 1.0
    return gap


def _fusion_terms(graph: NeighborGraph, seconds: np.ndarray,
                  bounds) -> tuple[np.ndarray, np.ndarray]:
    """Both terms, each in ``[0, 1]``, before they are mixed."""
    return _text_term(graph, bounds[0]), _time_term(graph, seconds, bounds[1])


def temporal_fusion(graph: NeighborGraph, seconds: np.ndarray | None, weight: float,
                    bounds) -> NeighborGraph:
    """Mix the elapsed time between two documents into the edge score.

    ``d'(A, B) = (1 - w) * t(d) + w * t(hours apart)``, where ``t`` is each term's own
    **winsorised linear map**: linear between that term's :data:`WINSOR_PERCENTILE` and its
    complement (:func:`winsor_bounds`), clipped flat outside them. Both terms land in ``[0, 1]``,
    so the fused score does too -- **non-negative and bounded by construction**, which is what lets
    the agglomerative methods accept it (``sklearn``'s ``distance_threshold`` refuses a negative
    radius) without an offset or a rank transform.

    Two properties the earlier formulas lacked, in the order they were arrived at:

    * **Bounded.** The original standardised each term (``z(d)``, ``z(log1p(hours))``), which put
      roughly half the fused edges below zero and killed ``average_linkage`` and
      ``componentwise_agglomerative`` outright -- and since the search raises when every
      configuration fails, it took the whole cell with it, finished results included.
    * **Equal range by construction.** Both terms span exactly ``[0, 1]``, so ``w`` means what it
      says and the balance point (:func:`balanced_time_weight`) lands near 0.3 rather than having
      to be discovered. An exponential CDF (``1 - exp(-x/s)``, the intermediate attempt) is bounded
      too, but its spread depends on the shape of each variable, which is why it needed the balance
      measured and why its grid sat in the wrong place.

    The cost is ties: everything past a bound collapses to one value. That is deliberate and it
    lands where it can be afforded -- clipping a fixed *fraction* rather than a fixed number of
    standard deviations is what keeps it off the near tail. A 3-sigma rule would take **0.9% off
    cosine's near side**, where the strongest same-author evidence is, because candidate-edge
    cosine distances are left-skewed (skew -0.63); the quantile clips 0.1% per side whatever the
    skew.

    **At ``weight = 0`` this is the pure-text attack, and it is still applied.** There is no
    short-circuit back to the raw cosine graph: ``w = 0`` is a value of the same formula, not a
    different scoring function, so a run with no timing is the same pipeline with one term zeroed.
    That is what makes ``plain`` and a tuned run that selects ``w = 0`` the same experiment. Only
    the time term is skipped, which is why ``seconds`` may be ``None`` -- ``0 * gap`` is zero for
    any gap, so a collection with no timestamps at all still has a ``w = 0`` result.

    For the three threshold methods that changes nothing: the map is monotone in ``d`` and a
    quantile threshold reads only the order. ``leiden`` and ``hdbscan`` do read magnitudes -- the
    first weights edges ``1 - d``, the second computes stabilities from ``1/d`` -- so for those two
    it is a real change. Measured on swe-chat's ``plain`` runs, leiden **gains** (BCubed F 0.5331
    raw cosine -> 0.5883 on ``all``, 0.7154 -> 0.7629 on ``unseen``) and hdbscan loses a little
    (0.3307 -> 0.3272, 0.6926 -> 0.6793).

    **Timing is attacker-visible metadata, not a leak.** An anonymised log carries timestamps; this
    project already scores ``baseline_language_primary`` and ``baseline_model_owner`` as metadata
    partitions for the same reason. What it changes is the *claim*: a result with ``weight > 0``
    says writing style **and session structure** link a user, not style alone.

    **What it is worth, and where it is not.** On swe-chat's ``all`` scope the fusion never hurts:
    +0.000, +0.012, +0.114, +0.103, +0.014 BCubed F over the same run's ``w = 0`` arm across the
    five algorithms. On ``unseen`` it is negative for every algorithm (mean -0.098), and that is a
    **weight-selection** failure rather than a fusion one -- the tuning slice ranks the candidate
    weights almost independently of the test collection (Spearman +0.057 over the grid, measured
    across ten cells), and on ``unseen`` it picks 0.657 where the test optimum is near 0.05. Read
    any ``unseen`` timing result with that in mind, and see :func:`balanced_time_weight`.
    """
    finite = np.isfinite(graph.distances)
    if not finite.any():
        return graph                     # nothing to transform; a padding-only graph is degenerate

    fused = (1 - weight) * _text_term(graph, bounds[0])
    if weight > 0:
        if seconds is None:
            raise ValueError(f"temporal_fusion at weight {weight:g} needs timestamps; "
                             f"`seconds` is None.")
        fused += weight * _time_term(graph, seconds, bounds[1])
    fused[~finite] = np.inf
    # Bounded by construction: both terms are in [0, 1] and the mixture is convex, so the fused
    # score cannot leave [0, 1). The dense linkage methods need that ceiling for their absent-edge
    # sentinel, and the graph is the only thing that knows it.
    return _resort(graph.indices, fused, f"fuse{weight:g}({graph.metric})", max_distance=1.0)


#: Name -> transform, so a driver selects one by string exactly as ``CLUSTERING_ATTACKS`` works
#: for the algorithms. ``none`` is present so "no rescoring" is a value of the same axis rather
#: than a branch in the caller. ``zscore`` is absent because it needs the embeddings, not just the
#: graph -- :func:`global_distance_moments` has to be called first, so its driver wires it by hand.
GRAPH_RESCORINGS = {
    "none": lambda graph, locality=DEFAULT_LOCALITY: graph,
    "csls": csls,
    "local_scaling": local_scaling,
    "mutual_knn": mutual_knn,
    "shared_neighbors": shared_neighbors,
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
