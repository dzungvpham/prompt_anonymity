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
