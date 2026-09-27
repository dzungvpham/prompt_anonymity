"""Blocked pairwise-distance kernel: the one place distances are actually computed.

One implementation, two consumers with different needs:
:class:`~prompt_anonymity.attacks.similarity.NearestNeighbor` reduces each row block to
per-author scores and never materialises the matrix; the LLM rerankers in
:mod:`prompt_anonymity.attacks.llm` use it to find each unknown document's nearest known
document within a shortlisted author. Both get the same arithmetic from
:func:`blocked_distances`.

Why this exists rather than a direct :func:`scipy.spatial.distance.cdist` call
------------------------------------------------------------------------------
``cdist``'s cosine is a naive C loop over pairs that also forces float64. Normalising both sides
once and handing the product to BLAS computes the same arithmetic far faster at scale.

The blocking matters just as much: a full pairwise distance matrix at this project's scale would
be tens of gigabytes, too large for a single allocation on a memory-limited job. Yielding row
blocks lets the caller reduce each one and drop it. Metrics other than cosine fall back to
:func:`sklearn.metrics.pairwise_distances_chunked`, which is BLAS-backed for ``"euclidean"`` and
blocks a per-block ``cdist`` for everything else -- so even an exotic metric gets the memory
bound, just not the speed.

Everything here is **exact**. Approximate nearest-neighbour indexing would trade recall for time,
and a missed neighbour is a false negative in precisely the hard cases that separate one attack
from another, which biases the headline re-identification number downward and non-uniformly.
"""

from __future__ import annotations

from typing import Iterator

import numpy as np
import sklearn
from sklearn.metrics import pairwise_distances_chunked

from ..common import unit_rows


def block_row_count(n_reference: int, working_memory_mb: int, dtype) -> int:
    """How many query rows produce a score block within the memory budget."""
    row_bytes = max(n_reference * np.dtype(dtype).itemsize, 1)
    return max(1, int(working_memory_mb * 1024 * 1024 // row_bytes))


def blocked_distances(query, reference, *, metric: str = "cosine",
                      working_memory_mb: int = 2048,
                      dtype=np.float32) -> Iterator[tuple[int, np.ndarray]]:
    """Yield ``(start_row, block)`` pairs covering the pairwise distance matrix.

    Parameters
    ----------
    query, reference : array-like of shape (n, n_features)
        Rows of ``query`` index the blocks; every block spans all of ``reference``.
    metric : str, default ``"cosine"``
        ``"cosine"`` takes the BLAS fast path. Any other name is forwarded to
        :func:`~sklearn.metrics.pairwise_distances_chunked`, which accepts every scipy metric.
    working_memory_mb : int, default 2048
        Target size of one block. Only an approximation for the non-cosine path, where sklearn
        picks the block size from the same budget by its own rule.
    dtype : default ``numpy.float32``
        Working precision for the cosine path. Halves both the product cost and the block
        against float64, and cosine rankings are unaffected at that precision. The fallback path
        yields whatever sklearn produces, which follows the input dtype.

    Yields
    ------
    (int, numpy.ndarray)
        Start row of the block and the block itself, shape ``(block_rows, n_reference)``.
        **Distances** -- smaller means more similar. Blocks arrive in row order and each is
        freshly allocated, so a caller may reduce, keep or discard one without affecting the next.
    """
    query = np.asarray(query, dtype=dtype)
    reference = np.asarray(reference, dtype=dtype)
    if query.ndim != 2 or reference.ndim != 2:
        raise ValueError(
            f"query and reference must be 2-D (got {query.shape} and {reference.shape})."
        )
    if query.shape[1] != reference.shape[1]:
        raise ValueError(
            "query and reference must have the same number of features "
            f"(got {query.shape[1]} and {reference.shape[1]})."
        )

    if metric == "cosine":
        query_unit, reference_unit = unit_rows(query), unit_rows(reference)
        rows = block_row_count(len(reference), working_memory_mb, dtype)
        for start in range(0, len(query_unit), rows):
            # Cosine distance is 1 - similarity; both steps in place so the product is the only
            # block-sized allocation.
            block = query_unit[start:start + rows] @ reference_unit.T
            np.negative(block, out=block)
            block += 1.0
            yield start, block
        return

    start = 0
    with sklearn.config_context(working_memory=working_memory_mb):
        for block in pairwise_distances_chunked(query, reference, metric=metric):
            yield start, block
            start += len(block)
