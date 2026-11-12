#!/usr/bin/env python
"""Re-identification under a simulated Laplace-DP defense on the unknown embeddings.

Runs ``run_experiment.py`` once per (epsilon, seed) with ``--unknown-noise-epsilon``, plus one
noise-free baseline, then collects the ``rolling_results.csv`` rows and plots re-identification
against epsilon. The known side is always clean; only the attacker's queries are noised.

Usage::

    python experiments/dp_noise_sweep.py --source swe_chat --data-dir data/hf --out data/dp_noise
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RUNNER = Path(__file__).with_name("run_experiment.py")
EPSILONS = (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000)

#: (column, label, chance column or None) for each plotted metric.
METRICS = (
    ("closed_set_top1", "Top-1 accuracy (per conversation)", "random_top1_known_pool"),
    ("id_acc1", "Users with ≥1 conversation identified", "random_id1"),
    ("margin_dir_at_far10", "DIR @ 10% FAR (per conversation)", None),
)


def run(args, out: Path, epsilon: float | None, seed: int) -> Path:
    name = "baseline" if epsilon is None else f"eps{epsilon:g}_s{seed}"
    target = out / args.source / name
    if not (target / "rolling_results.csv").exists():
        command = [sys.executable, str(RUNNER), "--source", args.source,
                   "--feature", args.feature, "--attacks", args.attack,
                   "--data-dir", args.data_dir, "--known-windows", *args.known_windows,
                   "--output-dir", str(target)]
        if epsilon is not None:
            command += ["--unknown-noise-epsilon", str(epsilon), "--noise-seed", str(seed),
                        "--noise-sensitivity", args.sensitivity]
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
    return target


def collect(out: Path, source: str) -> pd.DataFrame:
    rows = []
    for csv in sorted((out / source).glob("*/rolling_results.csv")):
        frame = pd.read_csv(csv)
        name = csv.parent.name
        if name == "baseline":
            frame["epsilon"], frame["seed"] = np.inf, -1
        else:
            eps, seed = name[3:].split("_s")
            frame["epsilon"], frame["seed"] = float(eps), int(seed)
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def plot(results: pd.DataFrame, source: str, path: Path) -> None:
    noisy = results[np.isfinite(results["epsilon"])]
    base = results[~np.isfinite(results["epsilon"])].iloc[0]
    fig, axes = plt.subplots(1, len(METRICS), figsize=(5 * len(METRICS), 4), sharex=True)
    for ax, (column, label, chance) in zip(axes, METRICS):
        summary = noisy.groupby("epsilon")[column].agg(["mean", "std"]).reset_index()
        ax.errorbar(summary["epsilon"], summary["mean"], yerr=summary["std"].fillna(0),
                    marker="o", capsize=3, label="Laplace noise on unknown embeddings")
        ax.axhline(base[column], color="black", linestyle="--", label="no noise")
        if chance is not None:
            ax.axhline(noisy[chance].mean(), color="gray", linestyle=":", label="random guessing")
        ax.set_xscale("log")
        ax.set_xlabel("DP epsilon (per conversation)")
        ax.set_title(label, fontsize=10)
        ax.set_ylim(bottom=0)
    axes[0].legend(fontsize=8)
    fig.suptitle(f"{source}: re-identification vs. epsilon (nearest neighbor, Gemini embeddings)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", default="swe_chat")
    parser.add_argument("--feature", default="gemini_embedding_2")
    parser.add_argument("--attack", default="nearest_neighbor")
    parser.add_argument("--data-dir", default="data/hf")
    parser.add_argument("--known-windows", nargs="+", default=["0075"])
    parser.add_argument("--sensitivity", default="bound", choices=["bound", "empirical"])
    parser.add_argument("--epsilons", type=float, nargs="+", default=list(EPSILONS))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--out", default="data/dp_noise")
    args = parser.parse_args()

    out = Path(args.out)
    jobs = [(None, 0)] + [(eps, seed) for eps in args.epsilons for seed in args.seeds]
    with ThreadPoolExecutor(args.jobs) as pool:
        for done, _ in enumerate(pool.map(lambda job: run(args, out, *job), jobs), 1):
            print(f"\r{done}/{len(jobs)} runs", end="", flush=True)
    print()

    results = collect(out, args.source)
    results.to_csv(out / f"{args.source}_dp_noise_results.csv", index=False)
    plot(results, args.source, out / f"{args.source}_dp_noise.png")
    table = (results.groupby("epsilon")[[m for m, _, _ in METRICS]].mean()
             .sort_index(ascending=False).round(3))
    print(table.to_string())


if __name__ == "__main__":
    main()
