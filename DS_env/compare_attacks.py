"""Compare the results CSVs produced by stylometric_attacks.py.

stylometric_attacks.py writes one CSV per (attack, defense, side) run into
defense_data/output/, named wildchat_analysis_<attack>_<defense>[_<side>].csv.
Each file holds N_SIM Monte-Carlo trials per (sample size, top-k) cell — so the
raw rows are noisy and not directly comparable across runs. This script reads
several of those CSVs, averages the trials per (sample, top) cell, and lines the
runs up side by side so you can see which defense actually lowers the attacker's
identity accuracy (and by how much) at each top-k.

Usage:
    # Compare specific runs:
    python compare_attacks.py \
        defense_data/output/wildchat_analysis_euclidean_style_none.csv \
        defense_data/output/wildchat_analysis_euclidean_style_rtt_argos_both_known_unknown.csv

    # Or compare every results CSV in defense_data/output/:
    python compare_attacks.py

Options:
    --metric {id_acc,conv_acc,advantage,id_acc_all_conv,advantage_all_conv}
                     Which column to compare (default: id_acc).
    --tops 1 5 10    Which top-k cutoffs to report (default: all present).
    --plot           Also write a PNG line chart (metric vs sample size) per top-k.
    --outdir DIR     Where to read CSVs from / write the plot (default: the
                     stylometric_attacks OUTPUT_DIR, defense_data/output).
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd

# Reuse the exact output location and filename-slug helper from the attack
# script so the two never drift apart. Importing is cheap: the heavy backends
# (torch/vLLM/llama.cpp) are all lazy-loaded inside their classes.
from stylometric_attacks import OUTPUT_DIR, _slug

# Columns run_trial writes; everything except these two id-ish keys is a metric.
_ID_COLS = ("sample", "top")
# Metrics where LOWER is a better defense (attacker does worse). Used only to
# annotate the printed table with an arrow, never to change the numbers.
_LOWER_IS_BETTER = {"id_acc", "conv_acc", "advantage", "id_acc_all_conv", "advantage_all_conv"}


def _label(path):
    """Short, stable name for a results file: strip the shared
    'wildchat_analysis_' prefix and the '.csv' so the legend/table is readable."""
    base = os.path.basename(path)
    base = base[: -len(".csv")] if base.endswith(".csv") else base
    prefix = "wildchat_analysis_"
    return base[len(prefix):] if base.startswith(prefix) else base


def load_aggregated(path, metric):
    """Read one results CSV and collapse its N_SIM trials to the mean metric per
    (sample, top) cell. Returns a DataFrame indexed by sample with one column per
    top-k, e.g. columns ('top1', 'top5', 'top10')."""
    df = pd.read_csv(path)
    if metric not in df.columns:
        raise SystemExit(
            f"{path!r} has no column {metric!r}. Available: "
            f"{[c for c in df.columns if c not in _ID_COLS]}"
        )
    # Average across the Monte-Carlo trials, then pivot top-k out to columns.
    agg = df.groupby(["sample", "top"])[metric].mean().unstack("top")
    agg.columns = [f"top{int(t)}" for t in agg.columns]
    return agg


def compare(paths, metric, tops):
    """Build a per-top-k comparison table across runs. Returns a dict
    {top-k -> DataFrame(index=sample, columns=run label)} of mean metric values."""
    per_run = {p: load_aggregated(p, metric) for p in paths}

    # Which top-k columns to report: those the user asked for that actually exist.
    present = sorted(
        {c for a in per_run.values() for c in a.columns},
        key=lambda c: int(c[3:]),
    )
    if tops is not None:
        wanted = {f"top{t}" for t in tops}
        present = [c for c in present if c in wanted]
        if not present:
            raise SystemExit(f"None of --tops {tops} are present in the CSVs.")

    tables = {}
    for col in present:
        # One column per run, aligned on the shared sample-size index.
        table = pd.DataFrame(
            {_label(p): agg[col] for p, agg in per_run.items() if col in agg.columns}
        )
        tables[col] = table.sort_index()
    return tables


def _pct_diff(table):
    """Express every run as a percentage difference from the BASELINE (the first
    column, i.e. the first CSV passed) computed PER ROW (per sample size), so each
    sample size is compared against its own baseline rather than a pooled average.
    A negative % means the attacker did worse than baseline at that sample size —
    for lower-is-better metrics, that is the defense working.

    Returns a display table: the baseline column keeps its raw metric values;
    every other column becomes a signed percentage string like '-24.3%'."""
    baseline = table.columns[0]
    disp = pd.DataFrame(index=table.index)
    disp[f"{baseline} (baseline)"] = table[baseline].map(lambda v: f"{v:.4f}")
    for name in table.columns[1:]:
        pct = (table[name] - table[baseline]) / table[baseline].replace(0, np.nan) * 100.0
        disp[f"{name} %Δ"] = pct.map(lambda v: "n/a" if pd.isna(v) else f"{v:+.1f}%")
    return disp


def print_tables(tables, metric):
    arrow = " (lower is a stronger defense)" if metric in _LOWER_IS_BETTER else ""
    for col, table in tables.items():
        k = col[3:]
        print(f"\n=== {metric} @ top-{k}{arrow} ===")
        # Per-sample-size grid: rows = sample size. Baseline shown as raw values,
        # every other run as its per-row % difference from the baseline.
        with pd.option_context("display.max_columns", None, "display.width", 200):
            print(_pct_diff(table))
        # A single headline number per run: the metric averaged over all sample
        # sizes, plus its overall % difference from the baseline, so runs can be
        # ranked at a glance.
        base_col = table.columns[0]
        means = table.mean(axis=0)
        base_mean = means[base_col]
        summary = means.sort_values(ascending=metric in _LOWER_IS_BETTER)
        print(f"\n  mean over sample sizes (best-defense first, %Δ vs {base_col}):")
        for name, val in summary.items():
            if name == base_col:
                print(f"    {val:.4f}  (baseline)          {name}")
            else:
                pct = (val - base_mean) / base_mean * 100.0 if base_mean else float("nan")
                print(f"    {val:.4f}  {pct:+6.1f}% vs baseline  {name}")


def plot_tables(tables, metric, outdir):
    import matplotlib
    matplotlib.use("Agg")  # headless: write a file, never try to open a window
    import matplotlib.pyplot as plt

    n = len(tables)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), squeeze=False)
    for ax, (col, table) in zip(axes[0], tables.items()):
        for name in table.columns:
            ax.plot(table.index, table[name], marker="o", markersize=3, label=name)
        ax.set_title(f"{metric} @ top-{col[3:]}")
        ax.set_xlabel("sample size (num identities)")
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()

    out_png = os.path.join(outdir, f"compare_{_slug(metric)}.png")
    fig.savefig(out_png, dpi=120)
    print(f"\nPlot saved to {out_png}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare stylometric_attacks.py results CSVs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "csvs", nargs="*",
        help="Results CSVs to compare. Default: every wildchat_analysis_*.csv "
             "in the output directory.",
    )
    parser.add_argument("--metric", default="id_acc",
                        help="Column to compare (default: id_acc).")
    parser.add_argument("--tops", nargs="+", type=int, default=None,
                        help="Top-k cutoffs to report (default: all present).")
    parser.add_argument("--plot", action="store_true",
                        help="Also write a PNG line chart of the metric vs sample size.")
    parser.add_argument("--outdir", default=OUTPUT_DIR,
                        help=f"Directory to glob CSVs from / write the plot (default: {OUTPUT_DIR}).")
    args = parser.parse_args()

    paths = args.csvs
    if not paths:
        paths = sorted(glob.glob(os.path.join(args.outdir, "wildchat_analysis_*.csv")))
    if not paths:
        raise SystemExit(f"No results CSVs found in {args.outdir!r}. Pass paths explicitly.")

    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        raise SystemExit(f"File(s) not found: {missing}")

    print(f"Comparing {len(paths)} run(s) on metric '{args.metric}':")
    for p in paths:
        print(f"  - {_label(p)}")

    tables = compare(paths, args.metric, args.tops)
    print_tables(tables, args.metric)
    if args.plot:
        plot_tables(tables, args.metric, args.outdir)


if __name__ == "__main__":
    main()
