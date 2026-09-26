"""Similarity attacks: build a summary of each author, then score how well a document matches it.

Every attack here fits a *summary* of an author from their known documents -- the document set
itself, a mean direction, a whitened mean, an LDA-projected mean, a Gaussian -- and scores an
unknown document by comparing it against that summary. None of them fits a decision function
separating one author from another; that is the other family, in
:mod:`prompt_anonymity.attacks.multiclass`.

Not called "distance", deliberately: only :class:`NearestNeighbor` computes one, and even it
returns the negation. :class:`CentroidCosine`, :class:`WhitenedCentroid` and
:class:`LDACentroid` return cosine *similarities*, and :class:`PLDA` returns a same-author
log-likelihood ratio, which is not a distance in any sense -- it can be negative and depends on
how many documents the author was enrolled with. The package contract is "higher = more likely",
the opposite of a distance. Distances do appear here, but one level down, in
:mod:`~prompt_anonymity.attacks.similarity.kernel`.

These methods model *where each author sits*, while the multiclass attacks learn *what separates
them*. The former tends to win when the author pool is large and each author has few documents to
fit a separator from.

======================  =======================================================================
attack                  what it compares
======================  =======================================================================
``nearest_neighbor``    distance to the author's single closest document
``cosine``              cosine to the author's mean direction
``wccn``                the same, after whitening out within-author scatter
``lda``                 the same, inside the LDA discriminant subspace
``lda_wccn``            whitened, inside the LDA discriminant subspace (LDA then WCCN, chained)
``plda``                same-author vs different-author log-likelihood ratio
======================  =======================================================================

:mod:`~prompt_anonymity.attacks.similarity.kernel` holds the blocked, BLAS-backed pairwise-distance
computation these share -- and that :mod:`prompt_anonymity.attacks.llm` borrows to pick a
representative document for a shortlisted author.
"""

from __future__ import annotations

from .centroid import CentroidCosine
from .kernel import block_row_count, blocked_distances
from .lda import LDACentroid
from .lda_wccn import LDAWCCN
from .nearest_neighbor import NearestNeighbor
from .plda import PLDA
from .whitened_centroid import WhitenedCentroid

SIMILARITY_ATTACKS = {
    "nearest_neighbor": NearestNeighbor,
    "cosine": CentroidCosine,
    "wccn": WhitenedCentroid,
    "lda": LDACentroid,
    "lda_wccn": LDAWCCN,
    "plda": PLDA,
}

__all__ = ["SIMILARITY_ATTACKS", "NearestNeighbor", "CentroidCosine", "WhitenedCentroid",
           "LDACentroid", "LDAWCCN", "PLDA", "blocked_distances", "block_row_count"]
