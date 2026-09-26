#!/usr/bin/env python
"""Population estimates, with intervals, from a sampled ``eval_utility.py --metric conversation`` run.

Judging a whole split is hours on the local judge, so the utility axis is estimated from a seeded
random sample instead (``eval_utility.py --limit N --seed S``). This turns the per-conversation
scores into estimates of the split-wide quantities, with 95% intervals:

* ``mean`` -- mean 1-5 conversation score;
* ``usable`` -- share scoring at least 4 (``USABLE_SCORE_THRESHOLD``);
* ``turn_mean`` -- mean of the judge's per-turn scores, pooled over every original turn of the
  sampled conversations (weights a conversation by its turn count, unlike ``mean``).

Four decisions, each load-bearing:

1. **The sample is re-derived, not read off the score file.** ``eval_utility.py`` merges every run
   into one file per (source, defense), so it also holds rows from other samples (e.g. a 50-row
   pilot drawn with a different ``--limit``). Those are not part of this random sample and would
   bias it, so the ``doc_id``s are regenerated from ``--limit``/``--seed`` with the driver's own
   ``sample_positions`` and only those rows are used.
2. **Intervals come from a cluster bootstrap over users**, not conversations: one user's
   conversations share tasks and style, so they are correlated, and resampling conversations would
   understate the noise (the attribution figures resample users for the same reason). Percentile
   intervals, ``--replicates`` draws, seeded. No finite-population correction is applied, so with a
   sample that is a large share of the split the intervals are conservative (at 1,000 of 4,334 the
   correction would narrow them by ~12%).
3. **Unchanged and turns-intact conversations stay in** as the 5s they are: they are part of the
   population, and dropping them would bias every mean down. Rows with no score (skipped as too
   long, or unparsed) are excluded and counted.
4. **Defenses are compared paired**, on the conversations both have a score for, with the same user
   bootstrap applied to the per-conversation difference. Since every defense is scored on the same
   seeded sample, the pairing is free and removes the between-conversation variance from the
   comparison. A defense whose file covers part of the split (``--defended-file``) is compared only
   on the rows it covers -- and its own mean describes those rows, not the split.

    python experiments/summarize_utility.py --source swe_chat --limit 1000 --seed 47 \\
        --defenses openanonymity styleremix dpmlm_eps_100_var_a10
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_utility import OUTPUT_DIR, sample_positions  # noqa: E402

from prompt_anonymity.data.config import hf_dir  # noqa: E402
from prompt_anonymity.evaluation.utility.prompt_judge import USABLE_SCORE_THRESHOLD  # noqa: E402

DEFAULT_REPLICATES = 2000
BOOTSTRAP_SEED = 0


def sampled_rows(source: str, data_dir: Path, limit: int, seed: int) -> pd.DataFrame:
    """``doc_id`` and ``author_id`` of exactly the conversations ``--limit``/``--seed`` sampled."""
    table = pq.read_table(data_dir / f"{source}.parquet", columns=["doc_id", "author_id"])
    frame = table.to_pandas()
    frame["doc_id"] = frame.doc_id.astype(str)
    return frame.iloc[sample_positions(len(frame), limit, seed)].reset_index(drop=True)


def conversation_table(source: str, defense: str, sample: pd.DataFrame) -> pd.DataFrame:
    """The sampled rows of one defense's score file, with per-conversation turn-score sums."""
    path = OUTPUT_DIR / f"{source}_{defense}.csv"
    scores = pd.read_csv(path, dtype={"conv_id": str})
    frame = sample.merge(scores, left_on="doc_id", right_on="conv_id", how="inner")
    turns = frame.get("judge_turn_scores", pd.Series("", index=frame.index)).fillna("")
    parsed = [json.loads(t) if t else [] for t in turns]
    frame["turn_sum"] = [sum(v for v in t if v is not None) for t in parsed]
    frame["turn_count"] = [sum(1 for v in t if v is not None) for t in parsed]
    return frame


def user_bootstrap(frame: pd.DataFrame, statistics, replicates: int, rng) -> np.ndarray:
    """``replicates × len(statistics)`` draws, resampling users with replacement.

    Each statistic is a ratio of per-user sums (``numerator_column``, ``denominator_column``), so a
    replicate is a weighted sum over users -- one matrix product for all replicates.
    """
    users = frame.groupby("author_id")
    sums = {column: users[column].sum().to_numpy(float)
            for pair in statistics for column in pair}
    n_users = len(next(iter(sums.values())))
    weights = rng.multinomial(n_users, np.full(n_users, 1.0 / n_users), size=replicates)
    return np.column_stack([(weights @ sums[num]) / (weights @ sums[den])
                            for num, den in statistics])


def estimate(frame: pd.DataFrame, replicates: int, rng) -> dict:
    """Point estimates and 95% user-bootstrap intervals for mean, usable and turn_mean."""
    scored = frame[frame.judge_score.notna()].copy()
    scored["one"] = 1.0
    scored["usable"] = (scored.judge_score >= USABLE_SCORE_THRESHOLD).astype(float)
    statistics = [("judge_score", "one"), ("usable", "one"), ("turn_sum", "turn_count")]
    draws = user_bootstrap(scored, statistics, replicates, rng)
    result = {"n": len(frame), "n_scored": len(scored), "n_users": scored.author_id.nunique()}
    for (num, den), name, column in zip(statistics, ["mean", "usable", "turn_mean"], draws.T):
        point = scored[num].sum() / scored[den].sum()
        low, high = np.nanpercentile(column, [2.5, 97.5])
        result.update({name: point, f"{name}_low": low, f"{name}_high": high})
    return result


def paired_difference(a: pd.DataFrame, b: pd.DataFrame, replicates: int, rng) -> dict:
    """Mean of (A - B) per conversation on the rows both scored, with a user-bootstrap interval."""
    both = a[["doc_id", "author_id", "judge_score"]].merge(
        b[["doc_id", "judge_score"]], on="doc_id", suffixes=("_a", "_b")).dropna()
    both["difference"] = both.judge_score_a - both.judge_score_b
    both["one"] = 1.0
    draws = user_bootstrap(both, [("difference", "one")], replicates, rng)[:, 0]
    low, high = np.percentile(draws, [2.5, 97.5])
    return {"n_shared": len(both), "mean_a": both.judge_score_a.mean(),
            "mean_b": both.judge_score_b.mean(), "difference": both.difference.mean(),
            "difference_low": low, "difference_high": high}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", default="swe_chat")
    parser.add_argument("--defenses", nargs="+", required=True,
                        help="names as in experiments/utility/<source>_<name>.csv")
    parser.add_argument("--limit", type=int, required=True, help="the --limit the runs used")
    parser.add_argument("--seed", type=int, required=True, help="the --seed the runs used")
    parser.add_argument("--replicates", type=int, default=DEFAULT_REPLICATES)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None,
                        help="CSV of the estimates (default: experiments/utility/"
                             "<source>_summary_n<limit>_seed<seed>.csv)")
    args = parser.parse_args()

    sample = sampled_rows(args.source, args.data_dir or hf_dir(), args.limit, args.seed)
    tables = {d: conversation_table(args.source, d, sample) for d in args.defenses}

    rows = []
    for defense, frame in tables.items():
        rows.append({"defense": defense, "subset": "own rows",
                     **estimate(frame, args.replicates, np.random.default_rng(BOOTSTRAP_SEED))})
    # Every defense again on the rows ALL of them scored, so a partial file's mean has a
    # like-for-like partner (its own rows are not a random sample of the split).
    common = set.intersection(*(set(f.doc_id[f.judge_score.notna()]) for f in tables.values()))
    if any(len(common) < f.judge_score.notna().sum() for f in tables.values()):
        for defense, frame in tables.items():
            rows.append({"defense": defense, "subset": f"shared rows ({len(common)})",
                         **estimate(frame[frame.doc_id.isin(common)], args.replicates,
                                    np.random.default_rng(BOOTSTRAP_SEED))})
    summary = pd.DataFrame(rows)

    pairs = []
    for a, b in itertools.combinations(args.defenses, 2):
        pairs.append({"a": a, "b": b, **paired_difference(
            tables[a], tables[b], args.replicates, np.random.default_rng(BOOTSTRAP_SEED))})
    pairs = pd.DataFrame(pairs)

    with pd.option_context("display.width", 200, "display.max_columns", 30,
                           "display.float_format", "{:.3f}".format):
        print(f"{args.source}: seeded sample of {len(sample):,} (seed {args.seed}); "
              f"95% intervals from {args.replicates:,} user-bootstrap draws\n")
        print(summary.to_string(index=False))
        print("\npaired differences in mean score (A - B), on rows both scored:\n")
        print(pairs.to_string(index=False))

    out = args.out or OUTPUT_DIR / f"{args.source}_summary_n{args.limit}_seed{args.seed}.csv"
    summary.to_csv(out, index=False, float_format="%.6g")
    pairs.to_csv(out.with_name(out.stem + "_paired.csv"), index=False, float_format="%.6g")
    print(f"\nwrote {out} and {out.with_name(out.stem + '_paired.csv').name}")


if __name__ == "__main__":
    main()
