"""The clustering algorithms themselves: a neighbour graph in, one label per document out.

Four methods, chosen to span the two families PAN 2016 found among its submissions plus the two
reference points that make their numbers readable:

======================  ====================================================================
``hdbscan``             density-based; finds clusters of varying density and refuses to place
                        documents in sparse regions (they come back as
                        :data:`~prompt_anonymity.evaluation.metrics.clustering.NOISE_LABEL`)
``leiden``              community detection on the similarity graph; guarantees well-connected
                        communities, which is the defect Leiden was introduced to fix in Louvain
``average_linkage``     bottom-up agglomerative merging under a distance cut -- the classic
                        authorship-clustering method and the one to beat
``connected``           threshold plus transitive closure; the naive attacker, present to
                        exhibit the chaining failure the other three are built to avoid
======================  ====================================================================

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
    """

    min_cluster_size: int = 2
    min_samples: int | None = None
    cluster_selection_epsilon: float = 0.0
    cluster_selection_method: str = "eom"

    def cluster(self, graph: NeighborGraph) -> np.ndarray:
        view = self.prepared(graph)
        adjacency = view.to_sparse()
        n_components, component = connected_components(adjacency, directed=False,
                                                       return_labels=True)
        if n_components == 1:
            return self._fit(adjacency)

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
            block = self._fit(adjacency[members][:, members])
            found = block >= 0
            labels[members[found]] = block[found] + next_label
            next_label += int(block.max()) + 1 if found.any() else 0
        return labels

    def _fit(self, adjacency) -> np.ndarray:
        from sklearn.cluster import HDBSCAN

        return HDBSCAN(
            min_cluster_size=max(2, int(self.min_cluster_size)),
            min_samples=self.min_samples,
            cluster_selection_epsilon=float(self.cluster_selection_epsilon),
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
        model = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=float(self.distance_threshold),
            metric="precomputed",
            linkage=self.linkage,
            connectivity=view.to_sparse(),
        )
        # Absent edges have to be a finite "far apart" rather than infinity: the linkage arithmetic
        # averages them. Two documents the graph does not join are given the maximum distance the
        # metric admits, which is 2.0 for cosine and the observed maximum otherwise.
        dense = view.to_sparse().toarray()
        far = 2.0 if view.metric == "cosine" else float(view.distances[np.isfinite(view.distances)].max())
        dense[dense == 0] = far
        np.fill_diagonal(dense, 0.0)
        return model.fit_predict(dense)


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

    def cluster(self, graph: NeighborGraph) -> np.ndarray:
        from scipy.sparse import csr_matrix

        view = self.prepared(graph)
        source, target, distance = view.edges()
        keep = distance <= self.distance_threshold
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
        "distance_threshold": [0.05, 0.1, 0.15, 0.2, 0.3],
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
