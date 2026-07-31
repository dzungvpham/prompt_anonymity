"""Plot helpers for linkage results (requires the optional ``viz`` extra)."""

from __future__ import annotations

from .plots import plot_cmc_curve, plot_headline_topk, plot_pool_size_sweep, plot_window_sweep

__all__ = ["plot_cmc_curve", "plot_headline_topk", "plot_pool_size_sweep", "plot_window_sweep"]
