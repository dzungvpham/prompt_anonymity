"""The k-nearest-neighbour graph every clustering attack is built on.

A clustering attack has to compare a document against *every other document*, not against a few
thousand author summaries, so the pairwise matrix it would like to hold is ``n x n``: 1.9 billion
entries on WildChat's 43,127-document test quarter, 7.4 GB at float32 and 14.8 at float64, against
a 16 GB job cap. Nothing in this package ever materialises it.

Instead each document keeps only its ``k`` nearest neighbours, which is all any of the clustering
algorithms actually reads -- HDBSCAN needs core distances and a minimum spanning tree, Leiden
needs a graph, agglomerative linkage needs a connectivity structure. The full matrix is streamed
one row block at a time through
:func:`~prompt_anonymity.attacks.similarity.kernel.blocked_distances` (the project's one distance
implementation, BLAS-backed and ~150x faster than ``cdist`` on cosine) and each block is reduced
to its top ``k`` and dropped.

**The neighbours are exact, not approximate.** An ANN index would trade recall for time, and a
missed neighbour is a false negative in precisely the hard cases that separate one clustering
method from another. Measured cost of doing it exactly, on this machine's login node with no GPU:
19 s for k=50 over 43,127 x 196 StyloMetrix vectors.

**Choosing k is a real experimental parameter, not an implementation detail.** It caps what any
graph-based method can link: two documents that are not neighbours can only ever be co-clustered
through a chain of other documents. ``experiments/run_clustering.py --diagnostics`` measures that
cap directly -- ``hit_at_k`` and ``neighbor_recall`` -- before any algorithm is run.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix

from ..similarity.kernel import blocked_distances


@dataclass(frozen=True)
class NeighborGraph:
    """Each document's ``k`` nearest neighbours, in increasing distance order.

    Attributes
    ----------
    indices : np.ndarray, shape (n, k), int32
        Row index of each neighbour. Column 0 is the nearest. A document is never its own
        neighbour.
    distances : np.ndarray, shape (n, k), float32
        Matching distances under :attr:`metric`; smaller means more similar, the same orientation
        :func:`~prompt_anonymity.attacks.similarity.kernel.blocked_distances` uses. ``inf`` pads
        the tail when a collection holds fewer than ``k + 1`` documents.
    metric : str
        The metric the distances were computed under, carried so a consumer cannot silently mix
        a cosine graph with a euclidean threshold.
    max_distance : float or None
        Largest distance :attr:`metric` admits, when it admits one -- 2.0 for cosine, 1.0 for the
        saturating fused score. ``None`` means unbounded or unknown (CSLS distances are routinely
        negative and have no ceiling), and consumers fall back to the observed maximum. Read
        through :attr:`far_distance` rather than directly.
    """

    indices: np.ndarray
    distances: np.ndarray
    metric: str = "cosine"
    max_distance: float | None = None

    @property
    def n_documents(self) -> int:
        return len(self.indices)

    @property
    def k(self) -> int:
        return self.indices.shape[1]

    @property
    def far_distance(self) -> float:
        """A finite stand-in for "these two are not neighbours at all".

        The dense linkage methods have to give an absent edge *some* number, because the linkage
        arithmetic averages it, and the honest one is the largest distance the space admits.
        Where the metric is bounded that is :attr:`max_distance`; where it is not, the observed
        maximum is the best available stand-in.

        **It is not a free choice.** The sentinel only means "far" relative to the real distances,
        so a transform that compresses the real ones without moving the sentinel makes absent
        edges look more repulsive than they are, and one that compresses the sentinel too makes
        them look less. That is why the bound travels with the graph instead of being re-derived
        from the metric's name at each call site.
        """
        if self.max_distance is not None:
            return float(self.max_distance)
        finite = self.distances[np.isfinite(self.distances)]
        return float(finite.max()) if len(finite) else 1.0

    def truncate(self, k: int) -> "NeighborGraph":
        """The same graph restricted to the ``k`` nearest neighbours of each document.

        Neighbours are stored in distance order, so a smaller ``k`` is a column slice. This is
        what makes a sweep over ``k`` cost one graph build rather than one per value.
        """
        if k > self.k:
            raise ValueError(f"cannot widen a graph built with k={self.k} to k={k}; rebuild it.")
        return NeighborGraph(self.indices[:, :k], self.distances[:, :k], self.metric,
                             self.max_distance)

    def edges(self, k: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The graph as a deduplicated **undirected** edge list ``(source, target, distance)``.

        "*i* is a neighbour of *j*" and "*j* is a neighbour of *i*" are one edge, so each pair
        appears once, oriented ``source < target``. This is the candidate set for the
        authorship-link ranking view: PAN 2016's second subtask ranks candidate links by
        confidence and scores the ranking with average precision, and these edges are the
        candidates any graph-based attacker would actually consider.

        Padding entries (``inf`` distance, used when the collection is smaller than ``k + 1``)
        are dropped.
        """
        graph = self if k is None else self.truncate(k)
        source = np.repeat(np.arange(graph.n_documents, dtype=np.int64), graph.k)
        target = graph.indices.ravel().astype(np.int64)
        distance = graph.distances.ravel()

        keep = np.isfinite(distance)
        source, target, distance = source[keep], target[keep], distance[keep]

        low = np.minimum(source, target)
        high = np.maximum(source, target)
        # Fold the ordered pair into one key so the deduplication is a single np.unique. The
        # product fits an int64 comfortably: n is in the tens of thousands, so n^2 is ~1e9.
        _, first = np.unique(low * graph.n_documents + high, return_index=True)
        return low[first], high[first], distance[first]

    def to_sparse(self, k: int | None = None, symmetrize: bool = True) -> csr_matrix:
        """The graph as a sparse distance matrix, for ``HDBSCAN(metric="precomputed")``.

        Absent entries read as infinite distance, which is what makes a sparse graph a usable
        stand-in for the dense matrix. ``symmetrize`` takes the union of the two neighbour
        relations (``max`` of the two stored values, which are equal where both exist) because
        k-nearest-neighbour is not a symmetric relation and scikit-learn rejects an asymmetric
        precomputed matrix.

        Zero distances are a genuine hazard here and are handled: an exact duplicate pair has
        distance 0.0, which a sparse matrix cannot tell from "no edge". They are bumped to the
        smallest positive float instead, so a duplicate stays the *closest* possible neighbour
        rather than silently becoming a non-neighbour.
        """
        source, target, distance = self.edges(k)
        distance = np.maximum(distance.astype(np.float64), np.finfo(np.float64).tiny)
        n = self.n_documents
        matrix = csr_matrix((distance, (source, target)), shape=(n, n))
        return matrix.maximum(matrix.T) if symmetrize else matrix


def build_neighbor_graph(embeddings: np.ndarray, k: int, *, metric: str = "cosine",
                         working_memory_mb: int = 1024) -> NeighborGraph:
    """Exact ``k``-nearest-neighbour graph over ``embeddings``, one row block at a time.

    Parameters
    ----------
    embeddings : array-like, shape (n_documents, n_features)
        One vector per document. The whole collection is both query and reference -- this is a
        self-join, which is what distinguishes clustering from the attribution attacks, where the
        two sides are different sets.
    k : int
        Neighbours to keep per document. Build once at the largest value a sweep needs and use
        :meth:`NeighborGraph.truncate` for the rest.
    metric : str, default ``"cosine"``
        Passed through to :func:`blocked_distances`; ``"cosine"`` takes the BLAS fast path.
    working_memory_mb : int, default 1024
        Target size of one streamed block. The peak allocation is one block
        (``block_rows x n_documents``) plus the output, never the full matrix.

    Notes
    -----
    A document's self-match is excluded by writing ``inf`` onto the block's diagonal before the
    selection, rather than by taking ``k + 1`` neighbours and filtering afterwards. That
    distinction matters on real data: with exact duplicate documents -- which both corpora
    contain despite deduplication, since dedup is per identity -- several entries tie at distance
    0.0 and a filter that removes "the one at distance zero" can drop the wrong row.
    """
    embeddings = np.asarray(embeddings, dtype=np.float32)
    n = len(embeddings)
    if k < 1:
        raise ValueError(f"k must be at least 1 (got {k}).")
    if n < 2:
        raise ValueError(f"need at least two documents to build a neighbour graph (got {n}).")
    width = min(k, n - 1)

    indices = np.zeros((n, k), dtype=np.int32)
    distances = np.full((n, k), np.inf, dtype=np.float32)
    for start, block in blocked_distances(embeddings, embeddings, metric=metric,
                                          working_memory_mb=working_memory_mb):
        rows = np.arange(len(block))
        block[rows, start + rows] = np.inf          # no document is its own neighbour
        # argpartition puts the `width` smallest first in arbitrary order; sort only those.
        nearest = np.argpartition(block, width - 1, axis=1)[:, :width]
        nearest_distances = np.take_along_axis(block, nearest, axis=1)
        order = np.argsort(nearest_distances, axis=1, kind="stable")
        indices[start:start + len(block), :width] = np.take_along_axis(nearest, order, axis=1)
        distances[start:start + len(block), :width] = np.take_along_axis(
            nearest_distances, order, axis=1)
    # Cosine is the one metric here with a ceiling known before any distance is computed; every
    # other name reaches `pairwise_distances_chunked`, where the bound is the metric's business
    # and not this function's to assert.
    return NeighborGraph(indices, distances, metric, 2.0 if metric == "cosine" else None)
