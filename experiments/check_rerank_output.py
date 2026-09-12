#!/usr/bin/env python
"""Is a rerank run's output actually usable? Structural checks, then a sample to read.

``run_rerank.py`` can finish cleanly and still have produced something worthless. A judge that
refused every row, or whose replies never parsed, degrades silently to the plain nearest-neighbor
ordering by design -- that is the right behaviour, and it is indistinguishable from a successful run
if you only look at the accuracy columns. A judge that emits the same sentence for every position,
or that is plainly reasoning about *topic* rather than style, also produces a complete, well-formed
table. So this script asks two different questions:

**Is the output well formed?** Every configuration's three tables exist, parse, agree with each
other, and satisfy the invariants the attack is supposed to guarantee -- chiefly that each document's
``rerank_position`` is a genuine permutation of ``1..k`` and that top-K accuracy at the shortlist
size still equals the baseline's. ``run_rerank.py`` asserts that last one in memory; re-checking it
from disk is what catches a bad *write* rather than a bad computation.

**Is it worth reading?** The share of positions carrying a real reason, how many distinct reasons
there are, and how long they run. These are the numbers that separate a judge that worked from one
that merely ran, and none of them appear in ``rerank_summary.csv``.

Then it prints a few complete reranked documents -- the query's true author, every candidate in the
order the model put them, and each one's justification. No aggregate can tell you whether the judge
is reading style or subject matter; three printed rows can, which is why the SLURM scripts end here
rather than on a number.

Style follows :mod:`prompt_anonymity.data.validate_dataset`: **collect, don't abort**, so one failure
cannot hide five others; every check carries the measured quantity rather than just a verdict; exit
1 if anything failed, so a job script can branch on it.

    python experiments/check_rerank_output.py experiments/results/rerank/<tag>
    python experiments/check_rerank_output.py <dir> --sample 5
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

#: Columns every predictions table must carry (see ``run_rerank.py``).
PREDICTION_COLUMNS = ("position", "doc_id", "true_author", "author_in_known", "best_author",
                      "base_best_author", "true_author_rank", "base_true_author_rank",
                      "n_candidate_authors")
#: Columns every detail table must carry.
DETAIL_COLUMNS = ("row", "unknown_doc", "true_author", "rerank_position", "presented_slot",
                  "distance_rank", "author", "known_doc_row", "is_true_author", "reason",
                  "relevance_score", "applied")
#: Cutoffs ``rerank_summary.csv`` reports, in order; accuracy must be non-decreasing across them.
SUMMARY_KS = (1, 2, 3, 5, 10)

#: A reply that parses is expected on nearly every row. Below this, the run is mostly the distance
#: baseline wearing a reranker's name, and the accuracies should not be reported as a rerank's.
REASON_FILLED_MIN = 0.95
#: Distinct reasons as a share of positions. A judge writing per-candidate justifications produces
#: almost entirely unique text; a low ratio means boilerplate, which is a judge that stopped reading.
REASON_DISTINCT_MIN = 0.50
#: "One or two sentences", as a character band on the median. Under the floor is a truncated or
#: stub reply; over the ceiling the model is writing essays and the token budget is being spent on
#: prose rather than on thinking.
REASON_MEDIAN_MIN_CHARS = 20
REASON_MEDIAN_MAX_CHARS = 600

#: Control characters that have no business in a CSV field (tab, newline and carriage return are
#: excluded -- they are legal inside a quoted field, and are counted separately as a tidiness note).
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def _config_files(results_dir: Path, variant: str, top_k: int) -> tuple[Path, Path]:
    return (results_dir / f"rerank_predictions_{variant}_k{top_k}.csv",
            results_dir / f"rerank_detail_{variant}_k{top_k}.csv")


def _check_predictions(check, label: str, predictions: pd.DataFrame, n_unknown: int) -> None:
    """Shape and internal consistency of one predictions table."""
    missing = [column for column in PREDICTION_COLUMNS if column not in predictions.columns]
    check(f"{label} predictions has every column", not missing, f"missing={missing}")
    if missing:
        return

    check(f"{label} predictions row count matches the summary",
          len(predictions) == n_unknown, f"{len(predictions)} rows, summary says {n_unknown}")
    check(f"{label} doc_id unique", predictions["doc_id"].is_unique,
          f"{len(predictions) - predictions['doc_id'].nunique()} duplicates")

    # A document whose author is not on the known side has no correct answer and so no rank. The
    # two facts must agree exactly: a rank where there should be none is a scoring bug, and a
    # missing rank where the author IS enrolled silently drops that document from every accuracy.
    in_known = predictions["author_in_known"].astype(bool).to_numpy()
    for column in ("true_author_rank", "base_true_author_rank"):
        ranked = predictions[column].notna().to_numpy()
        check(f"{label} {column} is present exactly where the author is enrolled",
              bool((ranked == in_known).all()),
              f"{int((ranked != in_known).sum())} rows disagree")
        values = predictions.loc[predictions[column].notna(), column].to_numpy()
        pool = predictions.loc[predictions[column].notna(), "n_candidate_authors"].to_numpy()
        check(f"{label} {column} lies in [1, n_candidate_authors]",
              bool(len(values) == 0 or ((values >= 1) & (values <= pool)).all()),
              f"min={values.min() if len(values) else 'n/a'}, "
              f"max={values.max() if len(values) else 'n/a'}, pool={pool.max() if len(pool) else 'n/a'}")


def _check_detail(check, label: str, detail: pd.DataFrame, n_unknown: int, top_k: int) -> bool:
    """Shape and per-document permutation structure of one detail table.

    Returns whether the table was intact enough for the cross-table checks to run.
    """
    missing = [column for column in DETAIL_COLUMNS if column not in detail.columns]
    check(f"{label} detail has every column", not missing, f"missing={missing}")
    if missing:
        return False

    check(f"{label} detail row count is n_unknown x k",
          len(detail) == n_unknown * top_k,
          f"{len(detail)} rows, expected {n_unknown} x {top_k} = {n_unknown * top_k}")

    # The three permutation invariants. A malformed one is exactly how a fold-back bug would
    # surface: the accuracies stay plausible while a document is ranked against the wrong candidates.
    wanted = {
        "rerank_position": list(range(1, top_k + 1)),
        "presented_slot": list(range(1, top_k + 1)),
        "distance_rank": list(range(top_k)),
    }
    groups = list(detail.groupby("row", sort=False))
    for column, expected in wanted.items():
        bad = sum(1 for _, group in groups if sorted(group[column].tolist()) != expected)
        check(f"{label} {column} is a permutation of {expected[0]}..{expected[-1]} per document",
              bad == 0, f"{bad} of {len(groups)} documents malformed")

    duplicated = sum(1 for _, group in groups if int(group["is_true_author"].astype(bool).sum()) > 1)
    check(f"{label} at most one true-author candidate per document",
          duplicated == 0, f"{duplicated} documents list the true author twice")
    return True


def _check_cross_table(check, label: str, predictions: pd.DataFrame, detail: pd.DataFrame) -> None:
    """The two tables describe the same documents, and tell the same story about the winner."""
    check(f"{label} both tables cover the same documents",
          set(detail["unknown_doc"].astype(str)) == set(predictions["doc_id"].astype(str)),
          f"{len(set(detail['unknown_doc'].astype(str)) ^ set(predictions['doc_id'].astype(str)))} "
          f"documents in one table only")

    # Where the gate let the rerank through, the model's rank-1 candidate IS the prediction. Where it
    # did not, the base attack's own #1 stands and the detail table's ordering was not applied, so
    # those rows are deliberately excluded rather than expected to agree.
    winners = detail[detail["rerank_position"] == 1].set_index("row").sort_index()
    # `row` indexes the predictions table positionally, so a detail table that mentions a row the
    # predictions table does not have (a truncated write, a half-finished job) would index past the
    # end. Report that as the finding it is rather than letting it raise -- this script exists to
    # survive malformed input, so a traceback here is a bug in the checker, not a result.
    in_bounds = winners.index.to_numpy() < len(predictions)
    check(f"{label} every detail row refers to a predictions row",
          bool(in_bounds.all()),
          f"{int((~in_bounds).sum())} detail rows point past the {len(predictions)}-row table")
    winners = winners[in_bounds]

    applied = winners["applied"].astype(bool).to_numpy()
    if applied.any():
        predicted = predictions["best_author"].astype(str).to_numpy()[winners.index.to_numpy()]
        agree = winners["author"].astype(str).to_numpy()[applied] == predicted[applied]
        check(f"{label} the reranker's #1 is the predicted author on every applied row",
              bool(agree.all()), f"{int((~agree).sum())} of {int(applied.sum())} disagree")
    else:
        check(f"{label} the reranker's #1 is the predicted author on every applied row",
              True, "no rows passed the ambiguity gate")


def _check_summary_row(check, label: str, row: pd.Series, top_k: int) -> None:
    """The accuracies are a well-formed CMC curve, and the shortlist ceiling did not move."""
    accuracies = [(k, row.get(f"top_{k}"), row.get(f"base_top_{k}")) for k in SUMMARY_KS]
    in_range = all(0.0 <= value <= 1.0 for _, value, _ in accuracies if pd.notna(value))
    check(f"{label} accuracies are in [0, 1]", in_range,
          ", ".join(f"top-{k}={value:.3f}" for k, value, _ in accuracies if pd.notna(value)))

    values = [value for _, value, _ in accuracies if pd.notna(value)]
    check(f"{label} accuracy is non-decreasing in k", all(np.diff(values) >= -1e-12),
          " <= ".join(f"{value:.3f}" for value in values))

    # The load-bearing one. The reranker reorders the shortlist and never changes its membership, so
    # top-K at the shortlist size is the base attack's own recall ceiling, to the last bit.
    if top_k in SUMMARY_KS:
        reranked, base = row.get(f"top_{top_k}"), row.get(f"base_top_{top_k}")
        check(f"{label} top-{top_k} still equals the baseline (shortlist ceiling unmoved)",
              pd.notna(reranked) and pd.notna(base) and float(reranked) == float(base),
              f"rerank={reranked}, base={base}")


def _check_reasons(check, label: str, detail: pd.DataFrame) -> None:
    """Is the ``reason`` column worth reading? The checks no accuracy column can make."""
    reasons = detail["reason"].fillna("").astype(str)
    filled = reasons.str.strip().str.len() > 0
    share = float(filled.mean()) if len(reasons) else 0.0
    check(f"{label} reasons present on >= {REASON_FILLED_MIN:.0%} of positions",
          share >= REASON_FILLED_MIN,
          f"{share:.1%} filled ({int((~filled).sum())} of {len(reasons)} empty -- empty means the "
          f"reply did not parse and that document fell back to the distance order)")

    present = reasons[filled]
    if not len(present):
        return

    distinct = present.nunique() / len(present)
    check(f"{label} reasons are mostly distinct (>= {REASON_DISTINCT_MIN:.0%})",
          distinct >= REASON_DISTINCT_MIN,
          f"{distinct:.1%} distinct ({present.nunique()} unique of {len(present)})")

    median = float(present.str.len().median())
    check(f"{label} median reason length is one or two sentences",
          REASON_MEDIAN_MIN_CHARS <= median <= REASON_MEDIAN_MAX_CHARS,
          f"median={median:.0f} chars, min={int(present.str.len().min())}, "
          f"max={int(present.str.len().max())}")

    control = int(present.str.contains(_CONTROL_RE, regex=True).sum())
    check(f"{label} reasons carry no control characters", control == 0,
          f"{control} reasons contain one")
    # Not a failure: a newline inside a quoted CSV field is legal and pandas round-trips it. It is
    # still worth knowing, because `awk`/`cut` on this file would not.
    multiline = int(present.str.contains(r"[\r\n]", regex=True).sum())
    check(f"{label} reasons are single-line (tool-friendly)", multiline == 0,
          f"{multiline} reasons span lines")


def _check_relevance(check, label: str, detail: pd.DataFrame) -> None:
    """The local reranker's own scores: finite, and actually separating the candidates."""
    scores = pd.to_numeric(detail["relevance_score"], errors="coerce")
    check(f"{label} relevance_score is present and finite everywhere",
          bool(np.isfinite(scores.to_numpy()).all()),
          f"{int((~np.isfinite(scores.to_numpy())).sum())} missing or non-finite")

    flat = sum(1 for _, group in detail.groupby("row", sort=False)
               if group["relevance_score"].nunique(dropna=False) <= 1)
    total = detail["row"].nunique()
    # A document whose candidates all score identically was not ranked by the model at all -- the
    # order came entirely from the distance tie-break, so the "rerank" is the baseline for that row.
    check(f"{label} the model separates the candidates it is shown",
          flat == 0, f"{flat} of {total} documents have a single distinct score across all candidates")


def _print_sample(results_dir: Path, summary: pd.DataFrame, n_sample: int, seed: int) -> None:
    """Print whole reranked documents, so a person can judge what the model was reasoning about."""
    if n_sample <= 0:
        return
    rng = np.random.default_rng(seed)
    for _, row in summary.iterrows():
        variant, top_k = str(row["variant"]), int(row["top_k"])
        _, detail_path = _config_files(results_dir, variant, top_k)
        if not detail_path.exists():
            continue
        detail = pd.read_csv(detail_path)
        rows = detail["row"].unique()
        if not len(rows):
            continue
        picked = rng.choice(rows, size=min(n_sample, len(rows)), replace=False)

        print(f"\n{'=' * 78}\nsample: {variant} k={top_k}\n{'=' * 78}")
        for document in sorted(picked):
            block = detail[detail["row"] == document].sort_values("rerank_position")
            first = block.iloc[0]
            correct = block[block["is_true_author"].astype(bool)]
            position = (int(correct["rerank_position"].iloc[0]) if len(correct)
                        else f"not in the top {top_k}")
            print(f"\ndocument {first['unknown_doc']}  true author {first['true_author']}  "
                  f"-> ranked at {position}")
            for _, candidate in block.iterrows():
                mark = " <- TRUE AUTHOR" if bool(candidate["is_true_author"]) else ""
                score = ("" if pd.isna(candidate["relevance_score"])
                         else f"  score {float(candidate['relevance_score']):.4f}")
                print(f"  {int(candidate['rerank_position'])}. {candidate['author']}"
                      f"   (distance rank {int(candidate['distance_rank'])}, "
                      f"shown as candidate {int(candidate['presented_slot'])}){score}{mark}")
                # Not `or ""`: a missing reason reads back from CSV as NaN, which is *truthy*, so
                # the naive guard prints the literal string "nan" under every jina candidate.
                reason = "" if pd.isna(candidate["reason"]) else str(candidate["reason"]).strip()
                if reason:
                    print(f"     {reason}")


def main(results_dir, n_sample: int = 3, seed: int = 47) -> int:
    results_dir = Path(results_dir)
    checks: list[tuple[str, bool, str]] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append((name, bool(ok), detail))

    print(f"checking {results_dir}")
    summary_path = results_dir / "rerank_summary.csv"
    if not summary_path.exists():
        print(f"  FAIL  no rerank_summary.csv in {results_dir}")
        print("\nNothing to check. Did run_rerank.py finish, and is this the right --output-dir?")
        return 1

    summary = pd.read_csv(summary_path)
    check("summary lists at least one configuration", len(summary) > 0, f"{len(summary)} rows")

    for _, row in summary.iterrows():
        variant, top_k = str(row["variant"]), int(row["top_k"])
        label = f"[{variant} k{top_k}]"
        n_unknown = int(row["n_unknown"])
        predictions_path, detail_path = _config_files(results_dir, variant, top_k)

        have_predictions = predictions_path.exists()
        have_detail = detail_path.exists()
        check(f"{label} predictions file exists", have_predictions, predictions_path.name)
        check(f"{label} detail file exists", have_detail, detail_path.name)
        if not (have_predictions and have_detail):
            continue

        try:
            predictions = pd.read_csv(predictions_path)
            detail = pd.read_csv(detail_path)
        except Exception as err:  # noqa: BLE001 - any parse failure is the finding
            check(f"{label} both tables parse as CSV", False, f"{type(err).__name__}: {err}")
            continue
        check(f"{label} both tables parse as CSV", True,
              f"{len(predictions)} + {len(detail)} rows")

        # Malformed input is the whole point of this script, so nothing below may raise: an
        # unanticipated shape becomes a FAIL line like any other, and the remaining configurations
        # still get checked. A traceback here would hide every finding after it.
        try:
            _check_summary_row(check, label, row, top_k)
            _check_predictions(check, label, predictions, n_unknown)
            if _check_detail(check, label, detail, n_unknown, top_k):
                _check_cross_table(check, label, predictions, detail)
                # The two arms answer different questions and are held to different standards: the
                # LLM owes a readable justification per position, the local reranker owes a score.
                if variant == "llm":
                    _check_reasons(check, label, detail)
                else:
                    _check_relevance(check, label, detail)
        except Exception as err:  # noqa: BLE001 - report, never abort
            check(f"{label} checks ran to completion", False, f"{type(err).__name__}: {err}")

    width = max(len(name) for name, _, _ in checks)
    n_fail = sum(not ok for _, ok, _ in checks)
    print()
    for name, ok, detail in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {name:<{width}}  {detail}")
    print(f"\n{len(checks) - n_fail}/{len(checks)} checks passed.")

    _print_sample(results_dir, summary, n_sample, seed)
    return 1 if n_fail else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a rerank run's tables and print a sample to read.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("results_dir",
                        help="directory run_rerank.py wrote (holds rerank_summary.csv)")
    parser.add_argument("--sample", type=int, default=3,
                        help="reranked documents to print per configuration; 0 prints none")
    parser.add_argument("--seed", type=int, default=47, help="seed for which documents are sampled")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    sys.exit(main(args.results_dir, n_sample=args.sample, seed=args.seed))
