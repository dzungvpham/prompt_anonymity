"""Constrained single linkage: take the cheap links, refuse the ones that build a blob.

``connected`` adds every edge under a threshold and takes the transitive closure, which is single
linkage cut at a height. Its failure is that one bad edge welds two people together permanently,
and the damage grows with what it welds.

**This is the PAN 2016 winner's fix, and it is a constraint on the merge rather than a repair
afterwards.** Bagnall's system is described in the task overview as "constraining a single-linkage
approach to avoid merging large clusters". Cutting a finished blob at a stricter threshold just
moves along the same precision/recall frontier the threshold already optimised, discarding the
good links along with the bad; refusing the merge *at the moment it would happen* keeps every link
made before it.

Two rules, because "large" has two readings
--------------------------------------------
``cap``       refuse a merge whose result would exceed ``max_cluster_size``. Simple, and it bounds
              the damage any single bad edge can do -- but it also refuses a *correct* merge for a
              genuinely prolific author.
``both``      refuse only when **both** sides are already larger than ``max_cluster_size``. A big
              cluster may still absorb a singleton, so a real author keeps growing; what is
              forbidden is welding two established groups together, which is the merge that costs
              the most precision. This is the closer reading of the PAN description.

Neither is a repair and neither can be applied afterwards: both depend on the state of the
partition at the moment an edge is considered, which is information the finished partition no
longer contains.
"""

from __future__ import annotations

import numpy as np


def capped_linkage(source: np.ndarray, target: np.ndarray, order: np.ndarray, n_documents: int,
                   max_cluster_size: int, rule: str = "both",
                   checkpoints: np.ndarray | None = None,
                   distance: np.ndarray | None = None,
                   max_mean_distance: float = float("inf")):
    """Single linkage over edges in ``order``, refusing merges that violate the size rule.

    Parameters
    ----------
    source, target : numpy.ndarray
        Edge endpoints.
    order : numpy.ndarray
        Indices into the edge arrays, in the order edges should be considered -- normally
        ``argsort`` of the distance, so the closest pair is merged first.
    n_documents : int
        Documents in the collection; labels are returned for all of them.
    max_cluster_size : int
        The size the rule is stated against. ``0`` disables the constraint, which makes this
        exactly :class:`~prompt_anonymity.attacks.clustering.algorithms.ThresholdComponents` and
        is what the sweep uses as its own control.
    rule : {"both", "cap"}
        See the module docstring.
    distance : numpy.ndarray, optional
        Edge distances, required only when ``max_mean_distance`` is finite.
    max_mean_distance : float
        Cohesion constraint: refuse a merge whose resulting cluster would have a **mean accepted
        edge distance** above this. ``inf`` disables it.

        This is Bagnall's "modified agglomerative approach where each link's score is adjusted to
        the mean of all the links in the cluster it forms" from his PAN 2016 submission -- a
        *cohesion* constraint where the two size rules above are *capacity* constraints. It does
        not care how big a cluster is, only whether it is still tight: a prolific author whose
        documents are genuinely close keeps merging, while a chain of individually-acceptable but
        collectively loose links gets stopped.
    checkpoints : numpy.ndarray, optional
        Edge counts at which to snapshot the partition. Because the algorithm is incremental, a
        whole budget sweep costs **one** pass rather than one pass per budget -- the same saving
        ``budget_frontier`` gets for unconstrained components, which is what makes the constrained
        version affordable to sweep at all.

    Returns
    -------
    list of (int, numpy.ndarray)
        ``(edges_considered, labels)`` at each checkpoint. With no checkpoints, one entry for the
        full edge list.
    """
    if rule not in ("both", "cap"):
        raise ValueError(f"unknown linkage rule {rule!r}; expected 'both' or 'cap'.")

    if np.isfinite(max_mean_distance) and distance is None:
        raise ValueError("max_mean_distance needs the edge distances.")

    parent = np.arange(n_documents, dtype=np.int64)
    size = np.ones(n_documents, dtype=np.int64)
    # Sum and count of the edges actually accepted into each cluster, so the would-be mean of a
    # merge is O(1) rather than a pass over the cluster's pairs.
    edge_sum = np.zeros(n_documents, dtype=np.float64)
    edge_count = np.zeros(n_documents, dtype=np.int64)

    def find(node: int) -> int:
        # Iterative with path halving: recursion depth on a 43,000-document chain would overflow,
        # and this is the inner loop of the whole method.
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    marks = ([] if checkpoints is None else sorted(int(value) for value in checkpoints))
    if not marks:
        marks = [len(order)]
    snapshots, next_mark = [], 0

    for position, edge in enumerate(order):
        while next_mark < len(marks) and marks[next_mark] == position:
            snapshots.append((position, _labels(parent, n_documents, find)))
            next_mark += 1
        a, b = find(int(source[edge])), find(int(target[edge]))
        if a == b:
            continue
        if max_cluster_size > 0:
            if rule == "cap" and size[a] + size[b] > max_cluster_size:
                continue
            if rule == "both" and size[a] > max_cluster_size and size[b] > max_cluster_size:
                continue
        merged_sum = merged_count = 0.0
        if np.isfinite(max_mean_distance):
            merged_sum = edge_sum[a] + edge_sum[b] + float(distance[edge])
            merged_count = edge_count[a] + edge_count[b] + 1
            if merged_sum / merged_count > max_mean_distance:
                continue
        if size[a] < size[b]:                        # union by size, so find() stays shallow
            a, b = b, a
        parent[b] = a
        size[a] += size[b]
        if np.isfinite(max_mean_distance):
            edge_sum[a], edge_count[a] = merged_sum, merged_count

    while next_mark < len(marks):
        snapshots.append((min(marks[next_mark], len(order)), _labels(parent, n_documents, find)))
        next_mark += 1
    return snapshots


def _labels(parent: np.ndarray, n_documents: int, find) -> np.ndarray:
    """Current component label per document, compacted to ``0..n_clusters-1``.

    Resolved by **vectorised pointer doubling** rather than by calling ``find`` per node: this runs
    once per budget checkpoint across a whole sweep, so a Python loop over every node here would
    cost more than the entire union-find it is reporting on. ``parent[parent]`` halves the
    remaining depth each pass, converging in a handful of passes.
    """
    root = parent.copy()
    while True:
        stepped = root[root]
        if np.array_equal(stepped, root):
            break
        root = stepped
    return np.unique(root, return_inverse=True)[1].ravel().astype(np.int64)
