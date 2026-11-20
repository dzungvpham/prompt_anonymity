#!/usr/bin/env python
"""One summary figure for the content-floor experiment.

Per dataset: share of target users caught at least once (closed-set, user level, mean of the 6
known windows) against the number of topics K, for the own-data and public codebooks, with the
full-embedding attack and the "guess the most active user" baseline as reference lines.
Reads the ``*_author_dir10.csv`` tables written from plot_results.py's author-level DIR-FAR CSVs.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#2b2b29", "#6f6e69", "#e4e3df"


def series_k(series: str) -> tuple[str, int] | None:
    match = re.match(r"Topic only(, public codebook)? \(K=(\d+)\)", series)
    if not match:
        return None
    return ("public" if match.group(1) else "own"), int(match.group(2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", default="/Users/vandanasingh/prompt_anonymity/data/thesis/content_floor")
    parser.add_argument("--chance", nargs=2, type=float, metavar=("SWE", "WILDCHAT"), required=True,
                        help="closed-set user-level proportional chance per dataset")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    panels = [("swe_chat", "SWE-chat (coding, 146 users)", args.chance[0]),
              ("wildchat", "WildChat (general chat, ~25k users)", args.chance[1])]
    figure, axes = plt.subplots(1, 2, figsize=(9.6, 4.3), sharey=False)
    for ax, (name, title, chance) in zip(axes, panels):
        table = pd.read_csv(Path(args.root) / f"{name}_author_dir10.csv")
        full = table[(table.attack == "nearest_neighbor") & (table.series == "No defense")].top1.mean()
        topic = table[table.attack == "topic_histogram"].copy()
        parsed = topic.series.map(series_k)
        topic = topic[parsed.notna()]
        topic[["codebook", "k"]] = pd.DataFrame(parsed.dropna().tolist(), index=topic.index)
        means = topic.groupby(["codebook", "k"]).top1.mean().reset_index()
        for codebook, color, marker, label in (("own", BLUE, "o", "Topic only, own-data topics"),
                                               ("public", ORANGE, "s", "Topic only, public topics")):
            rows = means[means.codebook == codebook].sort_values("k")
            ax.plot(rows.k, 100 * rows.top1, color=color, lw=2, marker=marker, ms=7,
                    mec="white", mew=1.5, label=label, zorder=3)
            last = rows.iloc[-1]
            ax.annotate(f"{100 * last.top1:.0f}%" if last.top1 >= 0.1 else f"{100 * last.top1:.1f}%",
                        (last.k, 100 * last.top1), xytext=(6, 0), textcoords="offset points",
                        va="center", fontsize=9, color=INK)
        ax.axhline(100 * full, color=INK, lw=1.4, ls="--", zorder=2, label="Full prompt embedding (no defense)")
        ax.text(0.02, 100 * full, f"  full prompt embedding: {100 * full:.0f}%", transform=ax.get_yaxis_transform(),
                va="bottom", fontsize=9, color=INK)
        ax.axhline(100 * chance, color=MUTED, lw=1.2, ls=":", zorder=2,
                   label="Baseline: always guess the most active user")
        ax.set_xscale("log", base=2)
        ax.set_xticks(sorted(means.k.unique()))
        ax.set_xticklabels([str(k) for k in sorted(means.k.unique())])
        ax.set_ylim(0, 100 * full * 1.15)
        ax.set_title(title, fontsize=11, color=INK, loc="left")
        ax.set_xlabel("number of topics K (more = more detailed topic)", fontsize=9, color=MUTED)
        ax.grid(axis="y", color=GRID, lw=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=9)
    axes[0].set_ylabel("users caught at least once (%)", fontsize=9, color=MUTED)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=2, frameon=False, handlelength=2.6, fontsize=9,
                  bbox_to_anchor=(0.5, -0.01))
    figure.tight_layout(rect=(0, 0.12, 1, 1))
    figure.savefig(args.out, dpi=200, facecolor="white")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
