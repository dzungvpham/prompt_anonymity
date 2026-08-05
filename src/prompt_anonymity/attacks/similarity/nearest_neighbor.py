"""Nearest-neighbour attribution: an author scores as well as their single closest document."""

from __future__ import annotations

import numpy as np

from ..common import group_by_author, unit_rows
from .kernel import blocked_distances


class NearestNeighbor:
    """The original attack: an author scores as well as their single closest known document.

    Note what the author-level aggregation buys: ranking *authors* by their nearest document is
    exactly the ranking :class:`~prompt_anonymity.evaluation.LinkageRanking` already derives for
    ``id_acc``, but it also makes ``conv_acc`` mean "the true author is among the top k
    **authors**" rather than "among the authors of the top k **documents**". The latter is a
    harder question at the same k -- the 5 nearest documents cover only ~4 distinct authors on
    swe-chat, and the 10 nearest only ~7 -- so this aggregation is what makes top-k comparable
    across methods, and it is why the LLM rerankers in :mod:`prompt_anonymity.attacks.llm`
    shortlist authors through this class rather than shortlisting documents directly.

    ``linkage="max"`` (the default, meaning maximum similarity / minimum distance) is the
    nearest-neighbour attack proper; ``"mean"`` averages over all of an author's documents.
    Mean linkage is *almost* :class:`~prompt_anonymity.attacks.similarity.CentroidCosine` -- it is
    exactly that attack without the final re-normalisation of the centroid (see :meth:`fit`) --
    but the difference is real rather than cosmetic, so both are worth running.

    Scaling
    -------
    Sized for author pools in the tens of thousands. The pairwise distances come from
    :func:`~prompt_anonymity.attacks.similarity.blocked_distances`, which supplies the BLAS fast
    path and the memory bound; on top of that this class adds two things:

    * **The per-author reduction is one pass, not a Python loop.** :meth:`fit` groups the known
      side by author so ``ufunc.reduceat`` aggregates every author in a single sweep of each
      block. The loop it replaces rebuilt an ``n_known`` boolean mask once per author -- 15,000
      times over at the scales this now runs at.
    * **Mean linkage never touches the blocks at all** (see :meth:`fit`).

    End to end at 50,000 unknown x 100,000 known x 3,072 dimensions over 15,000 authors: about
    two minutes on 36 CPU cores for ``linkage="max"`` and 13 seconds for ``linkage="mean"``,
    against roughly four hours for the ``cdist`` formulation this replaced.
    """

    name = "nearest_neighbor"

    def __init__(self, metric: str = "cosine", linkage: str = "max",
                 working_memory_mb: int = 2048, dtype=np.float32):
        self.metric = metric
        self.linkage = linkage
        self.working_memory_mb = working_memory_mb
        self.dtype = dtype

    def fit(self, embeddings, labels):
        self.authors, codes = np.unique(labels, return_inverse=True)
        known = np.asarray(embeddings, dtype=self.dtype)

        groups = group_by_author(codes, len(self.authors))
        self._known = np.ascontiguousarray(known[groups.order])
        self._starts = groups.starts
        self._stops = groups.stops
        # Kept so callers that shortlist authors can map a (row, author) pair back to the actual
        # known documents behind it -- see prompt_anonymity.attacks.llm.author_candidates.
        self._document_index = groups.order
        # Held in the working dtype rather than as integers: these are only ever a divisor for
        # mean linkage, and NumPy would promote a float32 score block divided by an int64 count
        # back to float64, quietly doubling the size of the score matrix.
        self._counts = groups.counts.astype(self.dtype)

        if self._uses_cosine and self.linkage == "mean":
            # Cosine distance is affine in the second vector, so averaging it over an author's
            # documents commutes with the dot product:
            #     mean_j (1 - x.y_j) = 1 - x.(mean_j y_j)
            # One centroid per author therefore reproduces mean linkage *exactly* while shrinking
            # the known side from n_known vectors to n_authors.
            #
            # The centroid is deliberately left un-normalised, which is the whole difference from
            # CentroidCosine: its length records how tightly the author's documents cluster, so a
            # diffuse author is penalised. Re-normalising here would change the top-1 pick on a
            # non-trivial fraction of documents.
            known_unit = unit_rows(self._known)
            self._centroids = (np.add.reduceat(known_unit, self._starts, axis=0)
                               / self._counts[:, None])
        return self

    @property
    def _uses_cosine(self) -> bool:
        """Whether the closed-form mean-linkage shortcut applies (it is cosine-only)."""
        return self.metric == "cosine"

    def documents_of(self, author_index: int) -> np.ndarray:
        """Row indices, in the *original* fit order, of one author's known documents."""
        return self._document_index[self._starts[author_index]:self._stops[author_index]]

    def score(self, embeddings):
        if self._uses_cosine and self.linkage == "mean":
            query = np.asarray(embeddings, dtype=self.dtype)
            return unit_rows(query) @ self._centroids.T - 1.0

        scores = np.empty((len(embeddings), len(self.authors)), dtype=self.dtype)
        for start, block in blocked_distances(
            embeddings, self._known, metric=self.metric,
            working_memory_mb=self.working_memory_mb, dtype=self.dtype,
        ):
            rows = slice(start, start + len(block))
            if self.linkage == "mean":
                scores[rows] = np.add.reduceat(block, self._starts, axis=1) / self._counts
            else:
                scores[rows] = np.minimum.reduceat(block, self._starts, axis=1)
        return -scores  # negate: higher must mean "more likely this author"
