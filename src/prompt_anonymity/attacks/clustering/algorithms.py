"""The clustering algorithms themselves: a neighbour graph in, one label per document out.

Five methods, chosen to span the two families PAN 2016 found among its submissions plus the two
reference points that make their numbers readable:

==============================  ============================================================
``hdbscan``                     density-based; finds clusters of varying density and refuses
                                to place documents in sparse regions (they come back as
                                :data:`~prompt_anonymity.evaluation.metrics.clustering.NOISE_LABEL`)
``leiden``                      community detection on the similarity graph; guarantees
                                well-connected communities, which is the defect Leiden was
                                introduced to fix in Louvain
``average_linkage``             bottom-up agglomerative merging under a distance cut -- the
                                classic authorship-clustering method and the one to beat.
                                Needs the dense matrix, so it stops at
                                :data:`MAX_DENSE_DOCUMENTS`
``componentwise_agglomerative``  the same merging run inside one connected component at a
                                time, which is *exactly equivalent* under a connectivity
                                constraint and is what makes average linkage runnable on
                                WildChat
``connected``                   threshold plus transitive closure; the naive attacker,
                                present to exhibit the chaining failure the others are built
                                to avoid -- and, measured, the one to beat on both corpora
==============================  ============================================================

Everything here consumes a :class:`~prompt_anonymity.attacks.clustering.graph.NeighborGraph`
rather than the raw vectors. That is not a convenience: the pairwise matrix is 1.9 billion entries
on WildChat's test quarter and none of these methods needs more than each document's nearest
neighbours. Building it once and sharing it across a hyper-parameter search is also what makes
tuning affordable -- the graph is the expensive part, and a sweep over ``resolution`` or
``min_cluster_size`` re-reads it for free.

The contract
------------
``ClusteringAttack.cluster(graph) -> labels``: one integer per document, ``-1`` for "not placed".
**The number of clusters is never an input.** An attacker does not know how many people wrote an
anonymised log, and a method that has to be told is answering an easier question than the threat
model poses. Hyper-parameters are chosen on a labelled *simulation* slice instead
(``experiments/run_clustering.py``), never on the collection under attack.

Why ``k`` matters more than any hyper-parameter here
----------------------------------------------------
Measured in phase 1: at ``k=100`` the symmetrised neighbour graph of both corpora is a **single
connected component**, so ``connected`` degenerates to one cluster and Leiden's resolution has to
do all the work. At ``k=1`` the largest component holds 0.2-0.4% of WildChat. Every method here is
therefore swept over ``k`` as well as its own parameters, and ``k`` is recorded next to them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.sparse.csgraph import connected_components

from .graph import NeighborGraph

#: Documents the attack declined to place. Mirrors
#: :data:`prompt_anonymity.evaluation.metrics.clustering.NOISE_LABEL`, duplicated here so the
#: attack side does not import the evaluation side.
NOISE_LABEL = -1

#: Above this, a method needing the dense pairwise matrix is refused rather than left to be
#: OOM-killed. Only :class:`AverageLinkageClustering` is affected -- scikit-learn's precomputed
#: agglomerative path has no sparse form. 20,000 documents is 3.2 GB at float64, which fits the
#: 24 GB jobs this project runs; WildChat's 43,127-document test quarter would be 14.9 GB on top
#: of a 2 GB feature matrix, and dies. The other three methods read only the neighbour graph.
MAX_DENSE_DOCUMENTS = 20_000


def edge_quantile(distances: np.ndarray, quantile: float) -> float:
    """The distance at a given quantile of a graph's own finite edge weights.

    **A threshold expressed as a quantile is the only kind that transfers between feature spaces.**
    ``distance_threshold`` is an absolute radius, so a grid tuned under raw cosine is meaningless
    after a learned projection rescales the space -- measured on WildChat, single linkage peaks at
    0.131 under cosine and 0.428 after the contrastive fit, and the shipped grid
    (``[0.05 ... 0.3]``) does not contain the second at all. A quantile asks the same question of
    both ("keep the closest 28% of candidate edges") and lands in the right place in each.

    Taken over the **k-truncated** edge list the attack will actually use, so the quantile equals
    the ``edge_share`` that ``experiments/improve_clustering.py`` sweeps and the two are directly
    comparable.
    """
    finite = distances[np.isfinite(distances)]
    if len(finite) == 0:
        return float("inf")
    index = min(int(quantile * len(finite)), len(finite) - 1)
    return float(np.partition(finite, index)[index])


@dataclass
class ClusteringAttack:
    """Base class: a configured clustering method, callable on a neighbour graph.

    Subclasses implement :meth:`cluster` and declare their hyper-parameters as dataclass fields.
    :attr:`neighbors` is shared by all of them because every method is defined over a ``k``-nearest
    -neighbour graph and ``k`` is as much a hyper-parameter as anything else -- see the module
    docstring for why it dominates.
    """

    #: Neighbours per document the method is allowed to see. ``None`` uses the whole graph.
    neighbors: int | None = None

    def prepared(self, graph: NeighborGraph) -> NeighborGraph:
        """The graph truncated to this configuration's ``k``."""
        return graph if self.neighbors is None else graph.truncate(self.neighbors)

    def cluster(self, graph: NeighborGraph) -> np.ndarray:
        raise NotImplementedError

    def settings(self) -> dict:
        """The hyper-parameters, for the results table."""
        return {field_name: getattr(self, field_name) for field_name in self.__dataclass_fields__}


@dataclass
class HDBSCANClustering(ClusteringAttack):
    """Density-based clustering over the neighbour graph (scikit-learn's HDBSCAN).

    Runs with ``metric="precomputed"`` on the **sparse** symmetrised graph, where an absent edge
    reads as infinite distance. The dense alternative is a 7.4 GB float32 matrix at WildChat's
    scale, and float64 -- which scikit-learn would want -- is 14.8 GB against a 16 GB cap.

    Two consequences of the sparse substrate, both real and both documented rather than hidden:
    a document with no edge to the rest of its author's traffic can only ever come back as noise,
    and the result depends on ``k``. That is the honest form of the method here, not a compromise:
    an attacker at this scale cannot compute the dense matrix either.

    ``min_cluster_size`` is the smallest group the method will call a cluster, and it is the
    parameter that most needs a *matched* tuning slice -- it is an absolute document count, so its
    optimum moves with how many documents an author writes in the window.

    ``cluster_selection_epsilon`` was, until this was added, never searched: :data:`CLUSTERING_SPACES`
    only swept ``neighbors``/``min_cluster_size``/``cluster_selection_method``, leaving epsilon
    pinned at its default 0.0 -- HDBSCAN's own density hierarchy with no additional flat merging,
    which is the maximum-precision/minimum-recall end of what the method can do. Measured on
    swe-chat/Gemini/``all`` scope: the tuned point this produced scored BCubed F 0.327 (base
    defense) to 0.152 (embad_gemini) against 0.469-0.556 for the two threshold methods, which *do*
    have their merge threshold in the search space -- and sweeping epsilon by hand finds F up to
    0.558, matching or beating them. So ``cluster_selection_epsilon_quantile`` exists to let the
    tuner reach that point on its own. It is a **quantile** of the view's own k-truncated edge
    weights (:func:`edge_quantile`), not an absolute radius, for the reason every other threshold in
    this package is: an absolute value means something different after a projection or after
    ``rescoring.temporal_fusion``'s winsorised rescale (applied even at weight 0), while a quantile
    means the same thing in any of those spaces. ``None`` (the default) keeps
    ``cluster_selection_epsilon`` as given, so existing configurations are unaffected.
    """

    min_cluster_size: int = 2
    min_samples: int | None = None
    cluster_selection_epsilon: float = 0.0
    cluster_selection_epsilon_quantile: float | None = None
    cluster_selection_method: str = "eom"

    def cluster(self, graph: NeighborGraph) -> np.ndarray:
        view = self.prepared(graph)
        epsilon = (self.cluster_selection_epsilon if self.cluster_selection_epsilon_quantile is None
                  else edge_quantile(view.edges()[2], self.cluster_selection_epsilon_quantile))
        adjacency = view.to_sparse()
        n_components, component = connected_components(adjacency, directed=False,
                                                       return_labels=True)
        if n_components == 1:
            return self._fit(adjacency, epsilon)

        # scikit-learn refuses a disconnected sparse graph outright ("HDBSCAN cannot be performed
        # on a disconnected graph"), which would rule the method out at every k that does not
        # produce one giant component -- i.e. most of the useful range, since phase 1 measured the
        # graph fragmenting fast below k=100. Running it per component is not a workaround but the
        # definition: mutual reachability between two components is infinite, so their cluster
        # hierarchies are independent and a joint fit could not have merged them anyway.
        labels = np.full(view.n_documents, NOISE_LABEL, dtype=np.int64)
        next_label = 0
        for index in range(n_components):
            members = np.flatnonzero(component == index)
            if len(members) < max(2, int(self.min_cluster_size)):
                continue                       # too small to hold a cluster: stays noise
            block = self._fit(adjacency[members][:, members], epsilon)
            found = block >= 0
            labels[members[found]] = block[found] + next_label
            next_label += int(block.max()) + 1 if found.any() else 0
        return labels

    def _fit(self, adjacency, epsilon: float) -> np.ndarray:
        from sklearn.cluster import HDBSCAN

        return HDBSCAN(
            min_cluster_size=max(2, int(self.min_cluster_size)),
            min_samples=self.min_samples,
            cluster_selection_epsilon=float(epsilon),
            cluster_selection_method=self.cluster_selection_method,
            metric="precomputed",
            copy=True,
        ).fit_predict(adjacency)


@dataclass
class LeidenClustering(ClusteringAttack):
    """Leiden community detection on the neighbour graph, with similarity as the edge weight.

    Leiden over Louvain because Louvain can return internally **disconnected** communities -- it
    will merge a node's neighbourhood into a community the node itself does not connect to -- and
    a disconnected "author" is meaningless here. Leiden's refinement phase guarantees every
    community is internally connected, which is exactly the property this task needs.

    ``resolution`` is the density a group must reach to stay separate: higher splits more. Under
    ``objective="cpm"`` it is directly comparable to an edge weight (a community must be denser
    than ``resolution``), which makes it interpretable but *scale-dependent* on the weights;
    under ``"modularity"`` it is relative to the graph's own null model, so it moves with graph
    size. Both are swept, because which one transfers better from the tuning slice is an empirical
    question rather than a settled one.

    Edge weights are ``1 - distance`` clipped at zero -- cosine similarity for a cosine graph.
    Leiden maximises *weight* inside communities, so a distance would invert the objective.
    """

    resolution: float = 1.0
    objective: str = "cpm"
    iterations: int = 2
    seed: int = 20260812

    def cluster(self, graph: NeighborGraph) -> np.ndarray:
        import igraph
        import leidenalg

        view = self.prepared(graph)
        source, target, distance = view.edges()
        weight = np.clip(1.0 - distance.astype(np.float64), 0.0, None)

        network = igraph.Graph(n=view.n_documents,
                               edges=list(zip(source.tolist(), target.tolist())))
        network.es["weight"] = weight.tolist()

        partition_type = (leidenalg.CPMVertexPartition if self.objective == "cpm"
                          else leidenalg.RBConfigurationVertexPartition)
        partition = leidenalg.find_partition(
            network, partition_type, weights="weight",
            resolution_parameter=float(self.resolution),
            n_iterations=int(self.iterations), seed=int(self.seed))
        return np.asarray(partition.membership, dtype=np.int64)


@dataclass
class AverageLinkageClustering(ClusteringAttack):
    """Agglomerative average-linkage merging under a distance cut, constrained to the graph.

    The classic bottom-up authorship-clustering method, and the family PAN 2016's stronger
    submissions came from. ``distance_threshold`` replaces ``n_clusters``, so the method decides
    how many authors it found -- the contract this package requires.

    The connectivity constraint is what makes it runnable: unconstrained average linkage needs the
    full pairwise matrix, and with it scikit-learn merges only along graph edges. It also changes
    the result, and in the right direction, since two documents with no path between them in the
    neighbour graph are not evidence of a shared author.

    **Average rather than single linkage**, deliberately: single linkage merges on the one closest
    pair and is exactly what produces the chaining that :class:`ThresholdComponents` exists to
    demonstrate. Phase 1 found the highest-similarity pairs in WildChat are byte-identical
    throwaway prompts (``h``, ``yes``, ``test``) written by *different* users, so a method that
    merges on the closest pair starts by welding unrelated people together.
    """

    distance_threshold: float = 0.5
    linkage: str = "average"

    #: When set, overrides :attr:`distance_threshold` with the corresponding quantile of this
    #: graph's own edge weights (:func:`edge_quantile`), as on :class:`ThresholdComponents` and
    #: :class:`ComponentwiseAgglomerative` and for the same reason. Resolved over the k-truncated
    #: **edge list**, not over the dense matrix the fit runs on: the edge list is the population
    #: every other method's quantile is taken over, so one quantile means one thing across the
    #: search, and the dense matrix is mostly the "far apart" filler value below rather than
    #: candidate pairs.
    distance_quantile: float | None = None

    def cluster(self, graph: NeighborGraph) -> np.ndarray:
        from sklearn.cluster import AgglomerativeClustering

        view = self.prepared(graph)
        if view.n_documents > MAX_DENSE_DOCUMENTS:
            raise MemoryError(
                f"average linkage needs a dense {view.n_documents:,}^2 distance matrix "
                f"({view.n_documents ** 2 * 8 / 1e9:.1f} GB at float64), over the "
                f"{MAX_DENSE_DOCUMENTS:,}-document limit. scikit-learn's precomputed path has no "
                f"sparse form, so this method does not scale to WildChat; the other three run on "
                f"the neighbour graph and do."
            )
        cut = (self.distance_threshold if self.distance_quantile is None
               else edge_quantile(view.edges()[2], self.distance_quantile))
        model = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=float(cut),
            metric="precomputed",
            linkage=self.linkage,
            connectivity=view.to_sparse(),
        )
        # Absent edges have to be a finite "far apart" rather than infinity: the linkage arithmetic
        # averages them. Two documents the graph does not join are given the largest distance the
        # space admits -- see `NeighborGraph.far_distance`, which knows the bound because the graph
        # carries it, rather than this line re-deriving it from the metric's name.
        dense = view.to_sparse().toarray()
        far = view.far_distance
        dense[dense == 0] = far
        np.fill_diagonal(dense, 0.0)
        return model.fit_predict(dense)


@dataclass
class ComponentwiseAgglomerative(ClusteringAttack):
    """Average linkage at WildChat's scale, by merging inside one connected component at a time.

    :class:`AverageLinkageClustering` refuses anything over
    :data:`MAX_DENSE_DOCUMENTS` because scikit-learn's precomputed path has no sparse form, which
    put the classic authorship-clustering method out of reach on the corpus that needs it most.
    This gets it back, and **not by approximating**: a connectivity-constrained agglomeration can
    only ever merge along graph edges, so two documents in different connected components of the
    graph are never candidates for the same cluster. Their linkage trees are therefore independent
    and running the method per component gives *exactly* the partition one dense fit would, at a
    peak cost of the largest component squared rather than the collection squared. On WildChat's
    tuning slice that is 9,400^2 (0.7 GB) instead of 43,128^2 (14.9 GB).

    **That equivalence holds only because each component's own connectivity submatrix is passed to
    scikit-learn along with its distances** -- see the comment in :meth:`cluster`. It was verified
    rather than assumed: on a synthetic 300-document graph the per-component partition is identical
    to one global constrained fit at four different threshold pairs, and the first version of this
    class, which omitted the submatrix, disagreed completely (94 clusters against 6) and scored
    0.107 lower on WildChat's tuning slice.

    The graph is first cut at :attr:`link_threshold` to form those components -- which is
    :class:`ThresholdComponents` -- and average linkage then splits each one under
    :attr:`distance_threshold`. So this is strictly a *refinement* of the connected-components
    partition and can only raise its precision, never its recall. That is the intended shape: the
    chaining single linkage produces is exactly what average linkage is supposed to undo, and
    doing it this way makes the two directly comparable at a matched first stage.

    ``linkage`` is exposed because ``complete`` is a free variation on the same machinery, but
    ``average`` is the default for the reason the sibling class documents: the closest pair in this
    corpus is frequently two different people writing ``"yes"``.
    """

    #: Cut used to form the components average linkage then works inside. Looser than
    #: :attr:`distance_threshold` by construction -- it decides what is *considered*, where the
    #: other decides what is merged.
    link_threshold: float = 0.2
    distance_threshold: float = 0.15
    linkage: str = "average"

    #: Quantile overrides, as on :class:`ThresholdComponents` and for the same reason.
    link_quantile: float | None = None
    distance_quantile: float | None = None

    #: Components above this are left as they are rather than split, with a warning. A component
    #: this large is a chaining failure that average linkage cannot repair anyway, and the dense
    #: matrix it would need is what this class exists to avoid.
    max_component: int = 20_000

    def cluster(self, graph: NeighborGraph) -> np.ndarray:
        from scipy.sparse import csr_matrix
        from sklearn.cluster import AgglomerativeClustering

        view = self.prepared(graph)
        source, target, distance = view.edges()
        link = (self.link_threshold if self.link_quantile is None
                else edge_quantile(distance, self.link_quantile))
        cut = (self.distance_threshold if self.distance_quantile is None
               else edge_quantile(distance, self.distance_quantile))
        keep = distance <= link
        source, target, distance = source[keep], target[keep], distance[keep]
        n = view.n_documents
        adjacency = csr_matrix((np.ones(len(distance), dtype=np.int8), (source, target)),
                               shape=(n, n))
        n_components, component = connected_components(adjacency, directed=False,
                                                       return_labels=True)

        # Edges bucketed by component once, rather than re-scanned per component: the giant holds
        # most of them and a per-component pass over the whole list is quadratic in components.
        edge_component = component[source]
        edge_order = np.argsort(edge_component, kind="stable")
        edge_starts = np.searchsorted(edge_component[edge_order], np.arange(n_components + 1))

        members_order = np.argsort(component, kind="stable")
        member_starts = np.searchsorted(component[members_order], np.arange(n_components + 1))

        far = view.far_distance
        labels = np.empty(n, dtype=np.int64)
        next_label = 0
        for index in range(n_components):
            members = members_order[member_starts[index]:member_starts[index + 1]]
            size = len(members)
            if size == 1:
                labels[members] = next_label
                next_label += 1
                continue
            if size > self.max_component:
                # Left whole, and said out loud. Silently passing it through produces a row that
                # looks like an average-linkage result and is really `connected` with its giant
                # cluster untouched -- measured on WildChat's tuning slice at
                # ``link_threshold=0.16``, that is a 60.9% cluster and a BCubed F that says
                # nothing about average linkage at all.
                import warnings

                warnings.warn(
                    f"component of {size:,} documents exceeds max_component="
                    f"{self.max_component:,} and was NOT split; this partition is "
                    f"connected-components on that component, not average linkage. Raise "
                    f"max_component (it costs size^2 x 8 bytes = "
                    f"{size ** 2 * 8 / 1e9:.1f} GB) or lower link_threshold.")
                labels[members] = next_label
                next_label += 1
                continue

            position = np.full(n, -1, dtype=np.int64)
            position[members] = np.arange(size)
            block = edge_order[edge_starts[index]:edge_starts[index + 1]]
            rows, columns = position[source[block]], position[target[block]]
            dense = np.full((size, size), far, dtype=np.float64)
            dense[rows, columns] = distance[block]
            dense[columns, rows] = distance[block]
            np.fill_diagonal(dense, 0.0)

            # The connectivity submatrix is what makes the equivalence in the class docstring
            # true, and leaving it out is not a small difference: without it scikit-learn averages
            # the `far` fill for every non-adjacent pair inside the component, which is a distance
            # this graph never measured. Verified on a synthetic 300-document graph -- with it, the
            # per-component partition is *identical* to one global constrained fit at three
            # different (link_threshold, distance_threshold) pairs; without it the two disagree
            # completely (94 clusters against 6).
            connectivity = csr_matrix(
                (np.ones(len(block), dtype=np.int8), (rows, columns)), shape=(size, size))
            connectivity = connectivity.maximum(connectivity.T)

            found = AgglomerativeClustering(
                n_clusters=None, distance_threshold=float(cut),
                metric="precomputed", linkage=self.linkage,
                connectivity=connectivity).fit_predict(dense)
            labels[members] = found + next_label
            next_label += int(found.max()) + 1
        return labels


@dataclass
class ThresholdComponents(ClusteringAttack):
    """Keep every edge closer than a threshold, then take connected components.

    Equivalently: **single-linkage hierarchical clustering cut at height** ``distance_threshold``.
    It was written here as a naive foil for the other three and then **beat all of them on both
    corpora** (BCubed F 0.566 on swe-chat, 0.487 on WildChat), so the framing has been corrected --
    it is a contender, and on this task the one to beat.

    Its known weakness is single linkage's: transitive closure means one wrong edge merges two
    people permanently. That weakness is real and *large* here, and it is why the method must never
    be reported without ``largest_cluster_share``: at its tuned settings it still puts **30.9% of
    WildChat and 52.1% of swe-chat into one cluster**. The BCubed F above is a mixture of that blob
    (precision ~0), a fifth of the corpus left as singletons (precision 1), and the useful mid-size
    clusters between them.

    What the tuner does with the two parameters is the interesting part: it does not tune a model,
    it **rations the opportunity to chain**. Both corpora selected ``distance_threshold=0.15``
    (cosine, i.e. similarity >= 0.85) and WildChat selected ``neighbors=2`` -- starving the graph of
    edges so transitive closure has few paths to propagate along. Phase 1 explains why: at ``k=100``
    the symmetrised graph is a *single* connected component on both corpora, where a loose
    threshold would return one cluster holding everybody.
    """

    distance_threshold: float = 0.3

    #: When set, overrides :attr:`distance_threshold` with the corresponding quantile of this
    #: graph's own edge weights (:func:`edge_quantile`). Required for any run in a learned space,
    #: where an absolute radius means nothing.
    distance_quantile: float | None = None

    def cluster(self, graph: NeighborGraph) -> np.ndarray:
        from scipy.sparse import csr_matrix

        view = self.prepared(graph)
        source, target, distance = view.edges()
        threshold = (self.distance_threshold if self.distance_quantile is None
                     else edge_quantile(distance, self.distance_quantile))
        keep = distance <= threshold
        adjacency = csr_matrix(
            (np.ones(int(keep.sum())), (source[keep], target[keep])),
            shape=(view.n_documents, view.n_documents))
        _, labels = connected_components(adjacency, directed=False, return_labels=True)
        return labels.astype(np.int64)


#: Name -> class, so a driver selects a method by string and a new one becomes selectable by
#: registering it here, exactly as :data:`~prompt_anonymity.attacks.ATTRIBUTION_ATTACKS` works for
#: the identification family.
CLUSTERING_ATTACKS = {
    "hdbscan": HDBSCANClustering,
    "leiden": LeidenClustering,
    "average_linkage": AverageLinkageClustering,
    "connected": ThresholdComponents,
    "componentwise_agglomerative": ComponentwiseAgglomerative,
}


def get_clustering_attack(name: str):
    """Look up a registered clustering attack by name."""
    try:
        return CLUSTERING_ATTACKS[name]
    except KeyError:
        raise ValueError(f"unknown clustering attack {name!r}; "
                         f"available: {sorted(CLUSTERING_ATTACKS)}") from None


#: What each method's hyper-parameter search draws from, mirroring
#: ``run_experiment.HYPERPARAMETER_SPACES``. Grids rather than distributions: these spaces are
#: small, and a grid makes the search reproducible without a seed and legible in the trials table.
#: ``neighbors`` is in every one of them because phase 1 measured it as the dominant parameter.
CLUSTERING_SPACES: dict[str, dict[str, list]] = {
    "hdbscan": {
        "neighbors": [5, 10, 25, 50],
        "min_cluster_size": [2, 3, 5],
        "cluster_selection_method": ["eom", "leaf"],
        # 0.0 resolves to the single closest edge (edge_quantile's index 0), not literally the old
        # fixed epsilon=0.0 default -- but it is the closest this quantile parameterisation gets to
        # "almost no merging", so it keeps that regime reachable rather than forcing the tuner away
        # from it. The rest brackets where a hand sweep found the F optimum on swe-chat/Gemini
        # (quantile ~0.40-0.46, F up to 0.558 against 0.327 at literal epsilon=0 -- class docstring).
        "cluster_selection_epsilon_quantile": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
    },
    "leiden": {
        "neighbors": [5, 10, 25, 50],
        "resolution": [0.05, 0.1, 0.2, 0.4, 0.6, 0.8],
        "objective": ["cpm", "modularity"],
    },
    "average_linkage": {
        "neighbors": [5, 10, 25],
        "distance_threshold": [0.15, 0.25, 0.35, 0.45, 0.55],
    },
    "connected": {
        "neighbors": [1, 2, 5, 10],
        # Refined 2026-08-13 after a fine sweep on WildChat's tuning slice: the F optimum sits at
        # 0.131 with k=10, which the old five-point grid ([0.05, 0.1, 0.15, 0.2, 0.3]) could not
        # express -- it selected 0.15/k=2 for F=0.4991 where 0.131/k=10 scores 0.5164. A grid
        # coarse enough to miss the optimum by 0.017 is not measuring the method.
        "distance_threshold": [0.05, 0.1, 0.12, 0.13, 0.14, 0.15, 0.2, 0.3],
    },
    "componentwise_agglomerative": {
        "neighbors": [5, 10, 25],
        "link_threshold": [0.15, 0.2],
        "distance_threshold": [0.08, 0.10, 0.12, 0.14],
    },
}


#: The same search, expressed in quantiles instead of absolute distances. Used for any run in a
#: learned space (``run_clustering.py --projection``), where :data:`CLUSTERING_SPACES`' radii are
#: meaningless -- see :func:`edge_quantile`. The range brackets every optimum measured on
#: WildChat's tuning slice across four feature spaces (edge shares 0.17 to 0.36).
#: Only the two threshold-based methods appear: Leiden's ``resolution`` and HDBSCAN's
#: ``min_cluster_size`` are not distances, so their grids carry over unchanged.
CLUSTERING_SPACES_QUANTILE: dict[str, dict[str, list]] = {
    # Bracketed wider than `connected`'s and NOT yet validated against a measured optimum, unlike
    # the entries below it. Average linkage merges on the *mean* distance between two groups, so
    # it tolerates a larger radius than single linkage does -- its absolute grid ran to 0.55 where
    # `connected`'s stopped at 0.3. Over-bracketing is the safe error here: the documented failure
    # on this file was a grid too coarse to contain the optimum, which selected 0.15/k=2 for
    # F=0.4991 where the true peak scored 0.5164. Narrow this once a sweep says where the peak is.
    "average_linkage": {
        "neighbors": [5, 10, 25],
        "distance_quantile": [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70],
    },
    "connected": {
        "neighbors": [2, 3, 5, 10, 25],
        "distance_quantile": [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50],
    },
    "componentwise_agglomerative": {
        "neighbors": [5, 10, 25],
        "link_quantile": [0.20, 0.30, 0.40],
        "distance_quantile": [0.05, 0.10, 0.15, 0.20],
    },
}


def parameter_grid(space: dict[str, list]) -> list[dict]:
    """Every combination in ``space``, as a list of keyword dicts."""
    from itertools import product

    keys = sorted(space)
    return [dict(zip(keys, values)) for values in product(*(space[key] for key in keys))]


@dataclass
class BaselineClustering(ClusteringAttack):
    """A reference partition dressed as an attack, so it runs through the same driver.

    ``kind`` is ``singleton``, ``single_cluster``, ``random`` or a metadata column name. Keeping
    these on the same code path as the real methods is what guarantees the baseline in a results
    table was scored identically to the method it is being compared against.
    """

    kind: str = "singleton"
    metadata: np.ndarray | None = field(default=None, repr=False)
    seed: int = 20260812

    def cluster(self, graph: NeighborGraph) -> np.ndarray:
        n = graph.n_documents
        if self.kind == "singleton":
            return np.arange(n, dtype=np.int64)
        if self.kind == "single_cluster":
            return np.zeros(n, dtype=np.int64)
        if self.kind == "random":
            if self.metadata is None:
                raise ValueError("the random baseline needs a reference labelling to permute.")
            reference = np.asarray(self.metadata).ravel()
            return reference[np.random.default_rng(self.seed).permutation(len(reference))]
        if self.kind == "metadata":
            if self.metadata is None:
                raise ValueError("the metadata baseline needs a column of values.")
            import pandas as pd
            return pd.factorize(pd.Series(self.metadata).astype("string").fillna("<missing>"))[0]
        raise ValueError(f"unknown baseline kind {self.kind!r}")

    def settings(self) -> dict:
        return {"kind": self.kind, "neighbors": self.neighbors}
