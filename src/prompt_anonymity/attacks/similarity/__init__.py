"""Similarity attacks: build a summary of each author, then score how well a document matches it.

Every attack here fits a *summary* of an author from their known documents -- the document set
itself, a mean direction, a whitened mean, an LDA-projected mean, a Gaussian -- and scores an
unknown document by comparing it against that summary. None of them fits a decision function
separating one author from another; that is the other family, in
:mod:`prompt_anonymity.attacks.multiclass`.

Not called "distance", deliberately: only :class:`NearestNeighbor` computes one, and even it
returns the negation. :class:`CentroidCosine`, :class:`WhitenedCentroid` and
:class:`LDACentroid` return cosine *similarities*, and :class:`PLDA` returns a same-author
log-likelihood ratio, which is not a distance in any sense -- it can be negative and it depends
on how many documents the author was enrolled with. The package contract is "higher = more
likely", the opposite of a distance. Distances do appear here, but one level down, in
:mod:`~prompt_anonymity.attacks.similarity.kernel`.

The distinction is the one the swe-chat results turn on: these methods model *where each author
sits*, while the multiclass ones learn *what separates them*. When authors are few and documents
per author are many, learning the separation wins. When the author pool is huge and most authors
have a handful of documents -- wildchat's median is 3 -- there is not enough per-author data to
fit a separator, and plain nearest neighbour is hard to beat.

======================  =======================================================================
attack                  what it compares
======================  =======================================================================
``nearest_neighbor``    distance to the author's single closest document
``cosine``              cosine to the author's mean direction
``wccn``                the same, after whitening out within-author scatter
``lda``                 the same, inside the LDA discriminant subspace
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
from .nearest_neighbor import NearestNeighbor
from .plda import PLDA
from .whitened_centroid import WhitenedCentroid

SIMILARITY_ATTACKS = {
    "nearest_neighbor": NearestNeighbor,
    "cosine": CentroidCosine,
    "wccn": WhitenedCentroid,
    "lda": LDACentroid,
    "plda": PLDA,
}

__all__ = ["SIMILARITY_ATTACKS", "NearestNeighbor", "CentroidCosine", "WhitenedCentroid",
           "LDACentroid", "PLDA", "blocked_distances", "block_row_count"]
