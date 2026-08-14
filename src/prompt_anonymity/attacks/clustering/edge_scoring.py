"""A learned same-author score on the graph's edges, replacing the distance that put them there.

:mod:`.projection` improves the clustering attack by changing the *space* the neighbour graph is
built in, but whatever it learns, the final comparison is still a cosine -- a bilinear form in the
two vectors. This module drops that restriction: a small network reads both vectors together and
returns ``P(same author)``, which is what the edge weight was always standing in for.

Why a cross-encoder and not a better embedding
-----------------------------------------------
Cosine after a linear map ``W`` is ``x' M y`` with ``M = W W^T`` -- symmetric, positive
semi-definite, and *rank-limited by the output dimension*. It cannot express "these two agree on
the features that matter for this kind of document but not that kind", which is exactly the
judgement a same-author decision needs when a corpus mixes languages, lengths and registers.
Interacting the pair first (``|a - b|`` and ``a * b``, the standard sentence-pair encoding) and
putting a network on top removes the constraint at a cost of one forward pass per candidate edge.

That cost is affordable only because the candidate set is already narrow. Scoring all pairs is
9.3e8 forward passes on WildChat's quarter; scoring the ``k``-nearest-neighbour graph's edges is
~2e6. So this is deliberately a **re-ranker over an existing graph**, never a way to build one --
the embedding still decides what is considered, and the network only decides what survives.

The same shape as :mod:`.projection`, and the same rule about labels
--------------------------------------------------------------------
Fitted on the labelled history slice, applied blind. Negatives are mined from the history graph's
own edges rather than drawn at random, because a random pair of documents is trivially separable
at this prevalence (0.13% of WildChat pairs share an author) and a model trained on those learns
nothing about the decisions it will actually be asked to make. Both halves of that mining use
history labels only.

The output is a probability, so ``1 - p`` is used as the graph distance. That keeps every consumer
unchanged -- the threshold sweeps, the algorithms, the frontier -- and it makes the edge weight
interpretable for once, which the CSLS and local-scaling outputs never were.
"""

from __future__ import annotations

import time

import numpy as np

from .graph import NeighborGraph

#: Candidate neighbours mined per history document when building the negative pool. Wide enough
#: that the hard negatives are genuinely hard, narrow enough that the mining graph is cheap.
NEGATIVE_MINING_NEIGHBORS = 20

#: Edges scored per forward pass at inference. Sized so the interaction tensor stays well under a
#: gigabyte at 1,024 dimensions (``batch x 2d`` float32), which is what a 2-million-edge graph
#: needs to stream rather than materialise.
SCORING_BATCH = 65_536


def _pair_features(left, right):
    """The standard sentence-pair interaction, ``[|a - b|, a * b]``.

    Both halves are symmetric in the two arguments, so the score is too -- which it must be, since
    "same author" is a symmetric relation and an asymmetric scorer would give an edge two different
    weights depending on which endpoint the graph happened to store first. The raw vectors are
    deliberately **not** concatenated: they would let the model condition on *who* rather than on
    *whether the two match*, which is identification, not the verification this attack is allowed.
    """
    import torch

    return torch.cat([(left - right).abs(), left * right], dim=1)


def mine_training_pairs(embeddings: np.ndarray, author_ids: np.ndarray, *,
                        neighbors: int = NEGATIVE_MINING_NEIGHBORS, metric: str = "cosine",
                        seed: int = 20260813, verbose: bool = True
                        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(left, right, label)`` for training, with negatives mined from the neighbour graph.

    Positives are every within-author pair, capped per author so that one prolific writer does not
    supply most of the training set -- WildChat's heaviest history author has enough documents to
    contribute more pairs on their own than a thousand median authors combined.

    Negatives come from each document's own nearest neighbours that turn out to be somebody else.
    They are the pairs this scorer exists to reject: a random negative is separable by topic alone
    and teaches the model nothing about the edges it will see.
    """
    generator = np.random.default_rng(seed)
    _, codes = np.unique(np.asarray(author_ids), return_inverse=True)
    codes = codes.ravel()

    started = time.perf_counter()
    from .graph import build_neighbor_graph

    graph = build_neighbor_graph(embeddings, neighbors, metric=metric)
    source, target, _ = graph.edges()
    negative = codes[source] != codes[target]
    if verbose:
        print(f"    mined {int(negative.sum()):,} hard negatives from a k={neighbors} history "
              f"graph in {time.perf_counter() - started:.0f}s", flush=True)

    # Positives: sample within each author's block rather than enumerating pairs, which is
    # quadratic and unnecessary -- the count below is already larger than the negative pool.
    order = np.argsort(codes, kind="stable")
    sorted_codes = codes[order]
    starts = np.searchsorted(sorted_codes, np.arange(codes.max() + 1))
    counts = np.bincount(sorted_codes)
    usable = np.flatnonzero(counts >= 2)
    #: Pairs drawn per author: flat, so every author contributes equally regardless of how much
    #: they wrote. The alternative (all pairs) is dominated by a handful of people.
    per_author = 16
    repeated = np.repeat(usable, per_author)
    first = starts[repeated] + generator.integers(0, counts[repeated])
    second = starts[repeated] + generator.integers(0, counts[repeated])
    same = first == second
    second[same] = (starts[repeated][same]
                    + (first[same] - starts[repeated][same] + 1) % counts[repeated][same])
    positive_left, positive_right = order[first], order[second]

    left = np.concatenate([positive_left, source[negative]])
    right = np.concatenate([positive_right, target[negative]])
    label = np.concatenate([np.ones(len(positive_left), dtype=np.float32),
                            np.zeros(int(negative.sum()), dtype=np.float32)])
    if verbose:
        print(f"    training pairs: {len(positive_left):,} positive, "
              f"{int(negative.sum()):,} negative", flush=True)
    return left, right, label


class EdgeScorer:
    """A fitted pairwise same-author model, callable on a neighbour graph.

    Holds a torch module and the device it lives on. Kept as a class rather than a function
    because :meth:`rescore` is called once per graph and the model has to survive between the fit
    and every use of it.
    """

    def __init__(self, model, device: str, name: str):
        self.model = model
        self.device = device
        self.name = name

    def score_pairs(self, embeddings: np.ndarray, left: np.ndarray, right: np.ndarray, *,
                    as_logit: bool = True) -> np.ndarray:
        """Same-author score for each ``(left[i], right[i])`` pair, streamed in batches.

        **Returns the logit by default, not the probability, and that is load-bearing.** A sigmoid
        is monotone, so in exact arithmetic the two rank edges identically -- but in float32
        ``sigmoid(x)`` rounds to exactly 1.0 above about x = 17, and a confident model puts a large
        share of its true edges there. Measured on WildChat's tuning slice, scoring in probability
        space left 138 of 556 frontier rows at a distance of exactly 0.0: a tied block that the
        edge-budget prefix then admits in arbitrary order, which welded 52% of the collection into
        one cluster and cost ~0.11 of BCubed F. The logit keeps every one of those edges distinct.

        ``as_logit=False`` returns the calibrated probability, which is what to use when the number
        itself is wanted (a report, a threshold with a meaning) rather than an ordering.
        """
        import torch

        features = torch.from_numpy(np.asarray(embeddings, dtype=np.float32)).to(self.device)
        out = np.empty(len(left), dtype=np.float32)
        self.model.eval()
        with torch.no_grad():
            for start in range(0, len(left), SCORING_BATCH):
                stop = min(start + SCORING_BATCH, len(left))
                rows = torch.from_numpy(left[start:stop].astype(np.int64)).to(self.device)
                columns = torch.from_numpy(right[start:stop].astype(np.int64)).to(self.device)
                logits = self.model(_pair_features(features[rows], features[columns])).squeeze(1)
                out[start:stop] = (logits if as_logit else torch.sigmoid(logits)).cpu().numpy()
        return out

    def rescore(self, graph: NeighborGraph, embeddings: np.ndarray) -> NeighborGraph:
        """The same graph with the negated same-author logit in place of the distance.

        Edge set unchanged -- this re-ranks, it does not retrieve (see the module docstring). Rows
        are re-sorted afterwards because every consumer relies on neighbours being in increasing
        distance order. The values are negated logits, so they are unbounded and frequently
        negative; like every other rescoring here the output is an ordering, not a metric, and
        thresholds have to be re-swept rather than carried over.
        """
        n, width = graph.indices.shape
        rows = np.repeat(np.arange(n, dtype=np.int64), width)
        columns = graph.indices.ravel().astype(np.int64)
        finite = np.isfinite(graph.distances).ravel()

        distances = np.full(n * width, np.inf, dtype=np.float32)
        distances[finite] = -self.score_pairs(embeddings, rows[finite], columns[finite])
        distances = distances.reshape(n, width)

        order = np.argsort(distances, axis=1, kind="stable")
        return NeighborGraph(np.take_along_axis(graph.indices, order, axis=1),
                             np.take_along_axis(distances, order, axis=1),
                             f"edgescore({graph.metric})")


def fit_edge_scorer(embeddings: np.ndarray, author_ids: np.ndarray, *, hidden: int = 512,
                    steps: int = 4000, batch_size: int = 4096, learning_rate: float = 1e-3,
                    weight_decay: float = 1e-4, dropout: float = 0.1,
                    neighbors: int = NEGATIVE_MINING_NEIGHBORS, seed: int = 20260813,
                    device: str | None = None, verbose: bool = True) -> EdgeScorer:
    """Train the pairwise scorer on labelled history.

    A two-layer MLP over :func:`_pair_features`, with the positives up-weighted to balance the
    mined pool. Balancing rather than resampling: the negative pool is what it is (every hard
    negative the history graph offers) and throwing most of it away to match the positive count
    would discard the informative half of the training set.

    ``dropout`` matters more here than the layer sizes. The input is a ``2d``-dimensional
    interaction of two embeddings that a previous stage already fitted on these same documents, so
    the model sits downstream of one fit and can memorise its residue without it.
    """
    import torch

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    generator = np.random.default_rng(seed)

    embeddings = np.asarray(embeddings, dtype=np.float32)
    left, right, label = mine_training_pairs(embeddings, author_ids, neighbors=neighbors,
                                             seed=seed, verbose=verbose)

    features = torch.from_numpy(embeddings).to(device)
    left_t = torch.from_numpy(left.astype(np.int64)).to(device)
    right_t = torch.from_numpy(right.astype(np.int64)).to(device)
    label_t = torch.from_numpy(label).to(device)

    n_features = embeddings.shape[1] * 2
    model = torch.nn.Sequential(
        torch.nn.Linear(n_features, hidden), torch.nn.ReLU(), torch.nn.Dropout(dropout),
        torch.nn.Linear(hidden, hidden // 2), torch.nn.ReLU(), torch.nn.Dropout(dropout),
        torch.nn.Linear(hidden // 2, 1)).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    positive_weight = torch.tensor(float((label == 0).sum() / max((label == 1).sum(), 1)),
                                   device=device)

    started = time.perf_counter()
    model.train()
    for step in range(steps):
        rows = torch.from_numpy(generator.integers(0, len(left), size=batch_size)).to(device)
        logits = model(_pair_features(features[left_t[rows]], features[right_t[rows]])).squeeze(1)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, label_t[rows], pos_weight=positive_weight)
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()
        if verbose and (step % 500 == 0 or step == steps - 1):
            print(f"    step {step:>5d}/{steps}  loss={loss.item():.4f}  "
                  f"[{time.perf_counter() - started:.0f}s on {device}]", flush=True)

    return EdgeScorer(model, device,
                      f"edge_scorer(hidden={hidden}, steps={steps}, dropout={dropout}, "
                      f"neighbors={neighbors})")
