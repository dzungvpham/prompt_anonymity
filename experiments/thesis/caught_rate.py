#!/usr/bin/env python
"""Users caught at least once (closed-set, user level) per known window, straight from predictions.

Equal to plot_results.py's author-level ``top1`` (DIR at FAR = 1): the share of in-set target users
with at least one conversation *in the shared test set* (the last 25% of the timeline, as in
plot_results.config_predictions) whose true author is ranked first.

Usage:  python experiments/thesis/caught_rate.py <results dir> [<results dir> ...]
"""
import sys
from pathlib import Path

import pandas as pd

WINDOWS = ["known0025", "known2550", "known5075", "known0050", "known2575", "known0075"]

for run in sys.argv[1:]:
    rates = {}
    for path in Path(run).glob("predictions_*_known*.csv"):
        table = pd.read_csv(path, usecols=["known_config", "position", "true_author",
                                           "author_in_known", "true_author_rank"])
        n_documents = int(table["position"].max()) + 1
        table = table[(table["position"] >= round(0.75 * n_documents)) & table["author_in_known"]
                      & table["true_author_rank"].notna()]
        rates[table["known_config"].iloc[0]] = (table["true_author_rank"] <= 1).groupby(
            table["true_author"]).any().mean()
    row = pd.Series(rates).reindex(WINDOWS)
    print(f"{Path(run).name:70s} mean6 {row.mean():.3f}  " + " ".join(f"{v:.3f}" for v in row))
