"""Plot helpers for linkage results.

These render the standard figures from the result tables produced by
:mod:`prompt_anonymity.evaluation`. They require the optional ``viz`` dependencies
(matplotlib, seaborn): install with ``pip install -e ".[viz]"``. ``method_label`` is the
legend name for the attack/feature series (e.g. ``"StyloMetrix"``), paired against the
random-guessing baseline.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless: render to files without a display
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402


def plot_headline_topk(headline, method_label: str, out_path) -> None:
    """Grouped bars of identity accuracy vs. the random baseline at each top-k.

    ``headline`` is the table from :func:`prompt_anonymity.evaluation.headline_accuracy`
    (columns ``top``, ``id_acc``, ``random_id``).
    """
    x = np.arange(len(headline))
    width = 0.38
    plt.figure(figsize=(7, 4))
    plt.bar(x - width / 2, headline["id_acc"], width, label=method_label)
    plt.bar(x + width / 2, headline["random_id"], width, label="Random guessing")
    plt.xticks(x, [f"top-{k}" for k in headline["top"]])
    plt.ylabel("Identification accuracy")
    plt.legend()
    plt.grid(axis="y", linestyle="--", color="lightgray")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_cmc_curve(cmc, method_label: str, out_path) -> None:
    """Cumulative match characteristic: top-k accuracy at every cutoff, one line per window.

    The full curve the three-row headline table samples at k = 1, 5, 10. Its *shape* is the
    interesting part: a curve that shoots up and flattens means the attack is confidently right
    on a subset, while one that climbs steadily means it is merely narrowing a large pool, and
    the two can share a top-1. ``cmc`` is the table from
    :func:`prompt_anonymity.metrics.cmc_curve` (columns ``k``, ``accuracy``, ``random``);
    ``known_fraction`` and ``window`` columns, if present, split it into one line per window.

    k is on a log axis because the informative part of a CMC curve is its first decade, and the
    dashed line is the random baseline, which is linear in k and therefore curved here.
    """
    group_columns = [column for column in ("known_fraction", "window") if column in cmc.columns]
    groups = cmc.groupby(group_columns) if group_columns else [((), cmc)]
    plt.figure(figsize=(7, 4.5))
    colors = sns.color_palette(n_colors=max(1, len(groups) if group_columns else 1))
    for color, (key, group) in zip(colors, groups):
        group = group.sort_values("k")
        key = key if isinstance(key, tuple) else (key,)
        name = " ".join(f"{column.replace('known_fraction', 'known')} {value:.0%}"
                        for column, value in zip(group_columns, key))
        plt.plot(group["k"], group["accuracy"], color=color,
                 label=f"{method_label} {name}".strip())
        plt.plot(group["k"], group["random"], color=color, linestyle="--", alpha=0.45)
    plt.xscale("log")
    plt.xlabel("k (candidate authors returned)")
    plt.ylabel("Top-k accuracy")
    plt.ylim(0, 1.02)
    plt.legend(fontsize="small", title="dashed = random guessing", title_fontsize="small")
    plt.grid(linestyle="--", color="lightgray", which="both")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_window_sweep(sweep, method_label: str, top_k: int, out_path) -> None:
    """Identity accuracy vs. the number of target users, one line per known fraction.

    The rolling-window analogue of :func:`plot_pool_size_sweep`: instead of drawing random
    sub-pools of users, each point is a real experiment on a longer slice of the timeline,
    which brings in more target users because more people show up over a longer period.
    ``sweep`` needs the columns ``known_fraction``, ``window``, ``n_identities``,
    ``id_acc`` and ``random_id``, filtered to a single ``top``.

    Solid lines are the attack, dashed lines in the matching colour are the random baseline
    for the same pool -- both move with the window, so only the gap between them is
    meaningful. Note that widening the window also pushes the unknown side further into the
    future, so a falling line mixes pool growth with temporal drift.
    """
    plt.figure(figsize=(7, 4))
    colors = sns.color_palette(n_colors=max(1, sweep["known_fraction"].nunique()))
    for color, (fraction, group) in zip(colors, sweep.groupby("known_fraction")):
        group = group.sort_values("n_identities")
        plt.plot(group["n_identities"], group["id_acc"], marker="o", color=color,
                 label=f"{method_label}, known {fraction:.0%}")
        plt.plot(group["n_identities"], group["random_id"], marker="o", markersize=4,
                 linestyle="--", color=color, alpha=0.6,
                 label=f"Random guessing, known {fraction:.0%}")
    plt.xlabel("Number of target users in the window")
    plt.ylabel(f"Top-{top_k} identification accuracy")
    plt.legend(fontsize="small")
    plt.grid(linestyle="--", color="lightgray")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_pool_size_sweep(sweep, method_label: str, top_k: int, out_path) -> None:
    """Paired boxplots of identity accuracy and the random baseline vs. candidate-pool size.

    ``sweep`` is the table from :func:`prompt_anonymity.evaluation.pool_size_sweep`
    (columns ``n``, ``id_acc``, ``random_id``).
    """
    melted = sweep.melt(id_vars="n", value_vars=["id_acc", "random_id"], var_name="metric", value_name="value")
    melted["metric"] = melted["metric"].map({"id_acc": method_label, "random_id": "Random guessing"})
    plt.figure(figsize=(7, 4))
    sns.boxplot(
        data=melted,
        x="n",
        y="value",
        hue="metric",
        order=sorted(sweep["n"].unique()),
        showfliers=False,
        width=0.5,
        palette=sns.color_palette(n_colors=2),
    )
    plt.xlabel("Number of candidate users")
    plt.ylabel(f"Top-{top_k} identification accuracy")
    plt.legend(title="")
    plt.grid(linestyle="--", color="lightgray")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
