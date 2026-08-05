"""Evaluation harnesses: rank once, then score the full pool and random sub-pools.

``LinkageRanking`` caches each unknown conversation's candidate ranking;
``headline_accuracy`` builds the standard top-k table on top of it.
"""

from __future__ import annotations

from .ranking import LinkageRanking
from .sweep import headline_accuracy

__all__ = ["LinkageRanking", "headline_accuracy"]
