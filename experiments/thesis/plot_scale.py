#!/usr/bin/env python
"""Figure for Q4/Q5: users caught vs. population size, heavy vs. random WildChat users."""
import argparse

import matplotlib.pyplot as plt
import pandas as pd

BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#2b2b29", "#6f6e69", "#e4e3df"

parser = argparse.ArgumentParser()
parser.add_argument("--random", required=True)
parser.add_argument("--heavy", required=True)
parser.add_argument("--real", nargs=4, type=float, required=True,
                    metavar=("RANDOM_FULL", "RANDOM_TOPIC", "HEAVY_FULL", "HEAVY_TOPIC"),
                    help="real full-corpus run (about 19,700 known users)")
parser.add_argument("--real-users", type=int, default=19711)
parser.add_argument("--out", required=True)
args = parser.parse_args()

figure, axes = plt.subplots(1, 2, figsize=(9.6, 4.3), sharey=True)
panels = [(args.heavy, "WildChat users with 10+ conversations", args.real[2], args.real[3]),
          (args.random, "Random WildChat users (most have only a few)", args.real[0], args.real[1])]
for ax, (path, title, real_full, real_topic) in zip(axes, panels):
    table = pd.read_csv(path)
    means = table.groupby("known_users")[["full", "topic"]].mean().reset_index()
    for column, color, marker, label in (("full", BLUE, "o", "Full prompt embedding"),
                                         ("topic", ORANGE, "s", "Topic only")):
        ax.plot(means.known_users, 100 * means[column], color=color, lw=2, marker=marker, ms=7,
                mec="white", mew=1.5, label=label + " (sampled population)", zorder=3)
    ax.scatter([args.real_users], [100 * real_full], s=60, facecolor="white", edgecolor=BLUE,
               lw=2, zorder=4, label="Full prompt, real WildChat (all users)")
    ax.scatter([args.real_users], [100 * real_topic], s=60, marker="s", facecolor="white",
               edgecolor=ORANGE, lw=2, zorder=4, label="Topic only, real WildChat (all users)")
    for value in (real_full, real_topic):
        ax.annotate(f"{100 * value:.0f}%", (args.real_users, 100 * value), xytext=(7, 0),
                    textcoords="offset points", va="center", fontsize=9, color=INK)
    ax.set_xscale("log")
    ax.set_xlim(90, 40000)
    ax.set_ylim(0, 100)
    ax.set_title(title, fontsize=11, color=INK, loc="left")
    ax.set_xlabel("number of users the attacker chooses between", fontsize=9, color=MUTED)
    ax.grid(axis="y", color=GRID, lw=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
axes[0].set_ylabel("users caught at least once (%)", fontsize=9, color=MUTED)
handles, labels = axes[0].get_legend_handles_labels()
figure.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=9)
figure.tight_layout(rect=(0, 0.13, 1, 1))
figure.savefig(args.out, dpi=200, facecolor="white")
print("wrote", args.out)
