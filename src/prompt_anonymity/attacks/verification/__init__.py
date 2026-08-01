"""Authorship verification: learned same-author scoring over document *pairs*.

Attribution asks "which of these authors wrote it?"; verification asks "did these two documents
share an author?". The pairwise form needs no fixed author pool at training time, which is why it
generalises to authors never seen during training -- and why it is expensive, since nothing can
be precomputed per document.
"""

from __future__ import annotations

from .cross_encoder import run_cross_encoder

VERIFICATION_ATTACKS = {"cross_encoder": run_cross_encoder}

__all__ = ["VERIFICATION_ATTACKS", "run_cross_encoder"]
