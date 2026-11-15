"""Compact DIR-FAR figures for the simulated-DP sweep: only the 75/25 split, one panel per corpus.

``plot_results.py`` already draws the full six-window facet grid for every run
(``plots_{conv,user}/<dataset>/openset/dirfar/author/by_defense/*.csv``). Six panels with six or
seven lines each is too busy for the page the paper has, and Dzung asked for the 75% known / 25%
target split only. This script re-reads those companion CSVs -- nothing is recomputed -- keeps the
``known0075`` window and draws it through the same helpers (``style_axes``, ``log_far_axis``,
palette, line width, chance dash), so the figure is styled like the rest of the paper.

Outputs (PDF, plus PNG with ``--png``), under ``--out``:
    dp_noise_conv.pdf   conversation-level DP, SWE-Chat | WildChat      (main paper: short)
    dp_noise_user.pdf   user-level DP (eps spent as eps / n per conversation), SWE-Chat | WildChat
    dp_noise_both.pdf   the two stacked, conversation level on top      (appendix)

All three share one legend under the panels: epsilon = infinity (no noise) first, then the
epsilons from large to small. In the user-level rows epsilon is the *user's* budget.

Usage:
    .venv/bin/python experiments/plot_dp_noise.py --plots-root data/dp_noise/paper --out <dir> --png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

import plot_results as pr  # noqa: E402  (the paper's style helpers and constants)

KNOWN = "known0075"
DATASETS = (("swe_chat", "SWE-Chat"), ("wildchat", "WildChat"))
SHORT = {"conv": "conv.", "user": "user"}
CSV = "openset/dirfar/author/by_defense/gemini_embedding_2_nearest_neighbor.csv"


SUPERSCRIPT = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")
INFINITY = "ε = ∞"


def classify(series: str) -> tuple[int, str]:
    """(colour slot, legend label) for one series name from the plot CSVs.

    "No defense" is the no-noise attacker, i.e. epsilon = infinity; it takes slot 0. An epsilon
    takes the slot of its exponent, so 10^k has the same colour in every row and in every figure
    (matching the six-panel grids, where the slot is the series' position in the sweep). A
    ``user ε = 10^k`` series is labelled ``ε = 10^k`` -- the row title says which accounting it is.
    """
    if series == "No defense":
        return 0, INFINITY
    label = series.removeprefix("user ")
    return int(label.split("10")[-1].translate(SUPERSCRIPT)), label


def load(root: Path, level: str, dataset: str, keep) -> list[tuple[int, str, pd.DataFrame]]:
    """The 75/25 curves of one (level, dataset), in file order, as (slot, label, frame)."""
    frame = pd.read_csv(root / f"plots_{level}" / dataset / CSV)
    frame = frame[frame["known_config"] == KNOWN]
    out = []
    for series, group in frame.groupby("series", sort=False):
        if keep(series):
            slot, label = classify(series)
            out.append((slot, label, group.sort_values("false_accept_rate")))
    return out


def draw(axes, curves, dataset: str, title: str, scale: float) -> dict:
    """One panel, in :func:`plot_results.draw_identification_panel`'s style.

    Returns ``{slot: (label, legend handle)}`` for the curves drawn."""
    handles = {}
    for slot, label, curve in curves:
        color = pr.series_style(slot)
        visible = curve[curve["false_accept_rate"] >= pr.FAR_LOG_FLOOR]
        axes.fill_between(visible["false_accept_rate"], visible["ci_low"], visible["ci_high"],
                          color=color, alpha=pr.BAND_ALPHA, linewidth=0, zorder=2)
        axes.plot(visible["false_accept_rate"], visible["identification_rate"], color=color,
                  linewidth=pr.LINE_WIDTH * scale, solid_capstyle="round", zorder=3 + slot * 0.01)
        handles[slot] = (label, Line2D([], [], color=color, linewidth=pr.LINE_WIDTH * scale))
    base = curves[0][2]
    drawn = "random_proportional" if base["random_proportional"].notna().all() else "random"
    base = base[base["false_accept_rate"] >= pr.FAR_LOG_FLOOR]
    axes.plot(base["false_accept_rate"], base[drawn], color=pr.TEXT_MUTED,
              linewidth=1.2 * scale, linestyle=pr.BASELINE_DASH, zorder=2)
    pr.style_axes(axes, "FAR", "DIR", title, title_size=pr.FONT_AXIS_LABEL, title_weight="normal",
                  scale=scale)
    pr.log_far_axis(axes)
    pr.unit_y_axis(axes)
    if dataset == "wildchat":
        pr.zoom_y_axis(axes, *pr.PANEL_Y_LIMITS[("identification_authors", "wildchat")])
    return handles


def figure_for(root: Path, levels: list[str], keep, scale: float = 1.0):
    """A grid with one row per level and one column per corpus, and ONE legend underneath.

    The legend lists every epsilon drawn anywhere, largest first (infinity = no noise, then 10^5,
    10^4, ...), i.e. top-to-bottom in the order the curves sit in the panels.
    """
    rows = len(levels)
    height = 2.9 * rows + 0.3
    figure, grid = plt.subplots(rows, 2, figsize=(7.0, height), squeeze=False,
                                facecolor=pr.SURFACE)
    handles = {}
    for r, level in enumerate(levels):
        for c, (dataset, name) in enumerate(DATASETS):
            curves = load(root, level, dataset, keep)
            title = name if rows == 1 else f"{name} ({SHORT[level]} DP)"
            handles.update(draw(grid[r][c], curves, dataset, title, scale))
    # Slot 0 (infinity) first, then the epsilons from large to small.
    order = sorted(handles, key=lambda slot: (slot != 0, -slot))
    labels = [handles[slot][0] for slot in order]
    figure.tight_layout(rect=(0, 0.3 / height, 1, 1))
    figure.legend([handles[slot][1] for slot in order], labels, frameon=False,
                  fontsize=pr.FONT_LEGEND, labelcolor=pr.TEXT_PRIMARY, ncol=len(labels),
                  loc="lower center", bbox_to_anchor=(0.5, 0.0), borderaxespad=0.2,
                  columnspacing=1.6, handlelength=1.8)
    return figure


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--plots-root", type=Path, required=True,
                        help="Folder holding plots_conv/ and plots_user/ from plot_results.py.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--png", action="store_true")
    parser.add_argument("--drop", nargs="*", default=[],
                        help="Series labels to leave out, e.g. 'ε = 10¹' (a hidden duplicate).")
    args = parser.parse_args()
    pr.WRITE_PNG = args.png
    keep = lambda label: label not in set(args.drop)  # noqa: E731
    for stem, levels in (("dp_noise_conv", ["conv"]), ("dp_noise_user", ["user"]),
                         ("dp_noise_both", ["conv", "user"])):
        print(pr.save_figure(figure_for(args.plots_root, levels, keep), args.out / stem))


if __name__ == "__main__":
    main()
