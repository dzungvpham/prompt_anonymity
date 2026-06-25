"""Evaluation harnesses: rank once, then score the full pool and random sub-pools.

``LinkageRanking`` caches each unknown conversation's candidate ranking;
``headline_accuracy`` and ``pool_size_sweep`` build the standard result tables on top of
it.
"""

from __future__ import annotations

from .ranking import LinkageRanking
from .sweep import headline_accuracy, pool_size_sweep

__all__ = ["LinkageRanking", "headline_accuracy", "pool_size_sweep"]
