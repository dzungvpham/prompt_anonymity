#!/usr/bin/env python
"""Score how much of a prompt a defense preserved: the utility axis, run over a parquet pair.

The attacks measure whether a defense hides *who wrote* a conversation. This measures the other
half of the trade -- whether the rewritten conversation still asks for the same thing, and whether
it is still well-formed text -- with the three metrics in
:mod:`prompt_anonymity.evaluation.utility`. A defense is only interesting where both axes are good;
one alone says nothing.

This script reads the same two files ``prompt_anonymity.data.apply_defenses`` writes and reads --
``<split>.parquet`` for the originals and ``<split>_<defense>.parquet`` for the rewrites -- so no
defending happens here::

    python -m prompt_anonymity.data.apply_defenses --source swe_chat --defense styleremix
    python experiments/eval_utility.py --source swe_chat --defense styleremix   # free, local
    python experiments/eval_utility.py --source swe_chat --defense styleremix --metric conversation --limit 50   # PAID
    python experiments/eval_utility.py --source swe_chat --defense styleremix --metric conversation \
        --judge-backend local   # the same judge against a self-hosted vLLM server (scripts/serve_qwen.sh), free

**One file per (source, defense), holding every metric's scores.**
``experiments/utility/<source>_<defense>.csv`` is keyed by ``conv_id`` and carries one column per
score plus what the judge cost. Each run **merges into** that file rather than replacing it: a
metric fills in its own columns and leaves the others alone, which is what lets the free local
scorers run on a GPU node and the paid judge run on a login node and still end up in one table.
Rows the run did not score keep whatever the file already had.

**Money is always opt-in.** The default ``--metric`` is the two local scorers, which cost GPU time
and nothing else; ``conversation`` calls a hosted judge, one request per conversation, and has to
be asked for by name. Start it with a small ``--limit``: the rubric is meant to be iterated on, and
a full corpus judged under one that turns out to be miscalibrated is money spent on a number that
gets thrown away. ``--judge-backend local`` sends the same rubric to a self-hosted vLLM server
instead (``$LOCAL_LLM_BASE_URL`` or ``localhost:$LOCAL_LLM_PORT``), which records ``0.0`` cost; its
verdicts cache separately, since the served model name is in the key. Every verdict is cached by conversation content, so raising the limit later
re-judges only what is new -- and a re-judged row's ``judge_cost_usd`` stays at what was actually
paid, since a cached row contributes nothing and does not overwrite the recorded figure.

**The sample is seeded and drawn across the whole split**, never a head slice: the parquets are
ordered by time and grouped by author, so the first N rows are a handful of people on one day and
say nothing about how the defense behaves elsewhere.

Rows are matched between the two files **by ``doc_id``, one-to-one**; a missing or duplicated id is
a hard error rather than a silent misalignment, which is the same contract
``experiments/run_experiment.py`` applies to its feature parquets. Only the sampled rows' ``turns``
are read (streamed a batch at a time through
:func:`~prompt_anonymity.data.apply_defenses.read_turns`), because ``turns`` is essentially the
whole dataset -- several GiB on WildChat once it is Python strings -- and a calibration run needs
three of them.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from prompt_anonymity.data.apply_defenses import as_attack_data, defended_stem, read_turns
from prompt_anonymity.data.config import cache_dir as default_cache_dir, hf_dir
from prompt_anonymity.defenses import DEFENSES
from prompt_anonymity.defenses._backends import join_turns
from prompt_anonymity.evaluation.utility import DEFAULT_SEED, UTILITY_METRICS, eval_utility
from prompt_anonymity.evaluation.utility.prompt_judge import (
    DEFAULT_JUDGE_BACKEND, JUDGE_BACKENDS, load_judge_prompt)

REPO_ROOT = Path(__file__).resolve().parents[1]
#: Where the per-(source, defense) score tables land. Deliberately not ``experiments/results/``:
#: that directory's names are a contract parsed by ``plot_results.py``, and these are not attack
#: runs.
OUTPUT_DIR = REPO_ROOT / "experiments" / "utility"

#: Metrics run when ``--metric`` is not given: the two that load a local checkpoint and bill
#: nothing. ``conversation`` is excluded on purpose -- see the module docstring.
DEFAULT_METRICS = ["semantic", "fluency"]

#: Column order in the output file. Literal rather than derived from the metric classes, so the
#: layout is stable no matter which metrics a run happened to include -- reading
#: :attr:`~prompt_anonymity.evaluation.utility.base.UtilityResult.score_columns` off all of them
#: would also mean importing every metric's deep-learning stack to write a header. A column a
#: newly registered metric introduces is not an error: unknown columns are kept, appended at the
#: end, and adding the name here is only about where it sits.
SCORE_COLUMN_ORDER = [
    "conv_id",
    "judge_score",        # conversation: the 1-5 verdict
    "judge_cost_usd",     # conversation: what that verdict actually cost
    "judge_turn_scores",  # conversation: per-turn 1-5 ratings, a JSON list per ORIGINAL turn
    "judge_reason",       # conversation: the judge's one-line explanation
    "judge_reasoning",    # conversation: its per-turn analyses and connections check
    "entailment",         # semantic: P(rewrite entails original)
    "bertscore_recall",   # semantic: token-level recall against the original
    "ppl_ratio",          # fluency: perplexity(rewrite) / perplexity(original)
]

#: Columns holding money spent, which **accumulate** instead of being overwritten. A metric reports
#: what *this* run paid for a row, so a cached verdict comes back as ``0.0`` and adds nothing --
#: but a row genuinely judged twice (a changed rubric, a different ``reasoning_effort``, a deleted
#: cache) was charged twice, and both charges are real. Summing this column across these files is
#: therefore the project's true cumulative API bill, which replacement would quietly understate.
PAID_COLUMNS = {"judge_cost_usd"}


def read_doc_ids(path: Path) -> list[str]:
    """Every ``doc_id`` in a parquet, in row order -- the join key, read without touching ``turns``."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist. Build it first with "
            f"'python -m prompt_anonymity.data.apply_defenses' (defended split) or "
            f"'python -m prompt_anonymity.data.build_dataset' (original split)."
        )
    table = pq.read_table(path, columns=["doc_id"])
    return [str(value) for value in table.column("doc_id").to_pylist()]


def align_positions(original_ids: list[str], defended_ids: list[str], *,
                    allow_missing: bool = False) -> list[int | None]:
    """Row position in the defended split for each original row, or raise.

    With ``allow_missing`` an original row absent from the defended split maps to ``None`` instead
    of raising -- only for ``--defended-file``, whose file may legitimately cover part of the split
    (see ``scripts/salvage_turn_lists.py``).

    The defended parquet is written in its source split's order, so this is usually the identity
    map -- but a sharded defense run reassembled out of order, or a defended file built from a
    different (or partly rebuilt) split, would not be. Checking is cheap and the failure it
    prevents is silent: judging conversation A's original against conversation B's rewrite would
    still produce a plausible-looking score.
    """
    if len(set(defended_ids)) != len(defended_ids):
        raise ValueError("the defended split has duplicate doc_ids; it cannot be joined one-to-one.")
    position_of = {doc_id: position for position, doc_id in enumerate(defended_ids)}
    missing = [doc_id for doc_id in original_ids if doc_id not in position_of]
    if allow_missing:
        return [position_of.get(doc_id) for doc_id in original_ids]
    if missing:
        raise ValueError(
            f"{len(missing):,} of {len(original_ids):,} documents are absent from the defended "
            f"split (e.g. {missing[:3]}). Re-run apply_defenses for the whole split, or point "
            "--data-dir at the directory holding the matching pair."
        )
    return [position_of[doc_id] for doc_id in original_ids]


def sample_positions(total: int, limit: int | None, seed: int) -> list[int]:
    """A seeded random sample of ``limit`` row positions, sorted; every position when unlimited.

    Sorted so the sampled rows are read in split order, and drawn without replacement across the
    whole split for the reason in the module docstring. Matches
    :meth:`~prompt_anonymity.evaluation.utility.base.UtilityMetric._load_sides`, which samples the
    same way when the sampling is left to the metric.
    """
    if limit is None or limit >= total:
        return list(range(total))
    rng = np.random.default_rng(seed)
    return sorted(int(position) for position in rng.choice(total, size=limit, replace=False))


def load_pair(source: str, defense: str, data_dir: Path, limit: int | None, seed: int,
              defended_path: Path | None = None):
    """The aligned ``(original, defended)`` bundles for a sample of the split, plus its full size.

    ``defended_path`` reads the rewrites from that parquet instead of the registered defense's
    ``<split>_<defense>.parquet``. Such a file may cover only part of the split: the sample is still
    drawn over the **whole** split, with the same seed, so it is the same sample every other
    defense is scored on, and sampled conversations the file lacks are dropped and counted. That
    keeps the result paired with the other defenses -- but the dropped rows are not random if the
    file is not (for the salvaged DP-MLM file they are mostly long conversations), so compare it to
    other defenses on the shared rows, not against their full-sample means.

    Each conversation's turn list is joined with the package's turn delimiter, which is what the
    metrics' renderer splits on -- so a defense that dropped or merged a turn is visible as such
    rather than as a wall of text.

    Loaded **once per run** even when several metrics are scored: the parquet read is the same for
    all of them, and on a full split it is the expensive part of the run that is not a forward pass.
    """
    original_path = data_dir / f"{source}.parquet"
    external = defended_path is not None
    if not external:
        defended_path = data_dir / f"{defended_stem(source, defense)}.parquet"

    original_ids = read_doc_ids(original_path)
    defended_ids = read_doc_ids(defended_path)
    defended_of = align_positions(original_ids, defended_ids, allow_missing=external)

    total = len(original_ids)
    positions = sample_positions(total, limit, seed)
    if len(positions) < total:
        print(f"[{source}/{defense}] scoring a seeded sample of {len(positions):,} of "
              f"{total:,} documents (seed={seed})")
    else:
        print(f"[{source}/{defense}] scoring all {total:,} documents")
    if external:
        covered = [position for position in positions if defended_of[position] is not None]
        print(f"[{source}/{defense}] {defended_path.name} covers {len(covered):,} of the "
              f"{len(positions):,} sampled documents; the other {len(positions) - len(covered):,} "
              "are dropped")
        positions = covered

    original_turns = read_turns(source, data_dir, positions)
    # read_turns takes a stem and a directory; a path's own stem and parent say the same thing.
    defended_turns = read_turns(defended_path.name.removesuffix(".parquet"), defended_path.parent,
                                [defended_of[position] for position in positions])

    doc_ids = [original_ids[position] for position in positions]
    # Authors are irrelevant to a utility score (nobody is being attributed here); the bundle wants
    # a label per row, so the doc_id stands in for one.
    reference = as_attack_data([join_turns(turns) for turns in original_turns], doc_ids, doc_ids)
    defended = as_attack_data([join_turns(turns) for turns in defended_turns], doc_ids, doc_ids)
    return reference, defended, total


def read_existing(path: Path) -> pd.DataFrame | None:
    """The score file already at ``path``, or ``None``. ``conv_id`` is forced to string so ids that
    happen to look numeric join against the parquet's rather than becoming floats."""
    if not path.exists():
        return None
    return pd.read_csv(path, dtype={"conv_id": str})


def merge_scores(existing: pd.DataFrame | None, frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Fold this run's score frames into the file's existing contents, keyed by ``conv_id``.

    Column by column, a fresh value wins wherever the run produced one and the previous value
    stands everywhere else -- so scoring one metric never disturbs another's columns, and scoring a
    sample never blanks the rows outside it. New conversations are added.

    :data:`PAID_COLUMNS` are the exception: they are **summed** rather than replaced, because they
    record money rather than a measurement. A cached row contributes ``0.0`` and changes nothing;
    a row judged again under different settings adds the second charge to the first, which is what
    was actually billed.
    """
    merged = (existing if existing is not None
              else pd.DataFrame({"conv_id": pd.Series(dtype="object")}))
    merged = merged.set_index("conv_id")

    for frame in frames:
        frame = frame.drop_duplicates("conv_id").set_index("conv_id")
        merged = merged.reindex(merged.index.union(frame.index))
        for column in frame.columns:
            incoming = frame[column].reindex(merged.index)
            if column not in merged.columns:
                merged[column] = incoming
            elif column in PAID_COLUMNS:
                # Missing on either side means "spent nothing here", not "unknown": a row the file
                # has never seen has paid nothing yet, and a row this run did not touch was not
                # charged by it. A row absent from both stays absent (NaN + NaN -> NaN), so an
                # unscored conversation is not reported as costing $0.
                total = merged[column].fillna(0.0) + incoming.fillna(0.0)
                merged[column] = total.where(merged[column].notna() | incoming.notna())
            else:
                merged[column] = incoming.combine_first(merged[column])

    merged = merged.sort_index().reset_index()
    ordered = [column for column in SCORE_COLUMN_ORDER if column in merged.columns]
    return merged[ordered + [c for c in merged.columns if c not in ordered]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", default="swe_chat",
                        help="split to score, e.g. swe_chat or wildchat (default: swe_chat)")
    parser.add_argument("--defense", default="styleremix", choices=sorted(DEFENSES),
                        help="which defended split to score against the original "
                             "(default: styleremix)")
    parser.add_argument("--metric", nargs="+", default=DEFAULT_METRICS,
                        choices=sorted(UTILITY_METRICS), metavar="METRIC",
                        help="one or more registered utility metrics; each writes its own columns "
                             f"into the one output file. Available: {', '.join(sorted(UTILITY_METRICS))}. "
                             f"'conversation' calls a PAID API, one request per conversation, so it "
                             f"is not in the default ({' '.join(DEFAULT_METRICS)})")
    parser.add_argument("--defended-file", default=None, type=Path,
                        help="score the rewrites in this parquet (doc_id, turns) instead of a "
                             "registered defense's; it may cover part of the split (e.g. the "
                             "output of scripts/salvage_turn_lists.py). Named in the output by "
                             "--label")
    parser.add_argument("--label", default=None,
                        help="--defended-file only: the name for the output file, "
                             "<source>_<label>.csv (default: the file's stem without the "
                             "'<source>_' prefix and a '.salvaged' suffix)")
    parser.add_argument("--data-dir", default=None,
                        help="directory holding <split>.parquet and <split>_<defense>.parquet "
                             "(default: the published-dataset mirror, data/hf)")
    parser.add_argument("--cache-dir", default=None,
                        help="cache root for judge replies, regenerable but PAID -- deleting it "
                             "means re-buying every verdict (default: data/.cache)")
    parser.add_argument("--limit", type=int, default=None,
                        help="score a seeded random sample of this many conversations; for the "
                             "judge, one conversation is one API request (default: the whole split)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help=f"seed for that sample (default: {DEFAULT_SEED})")
    parser.add_argument("--judge-prompt", default=None,
                        help="YAML file with a 'system_prompt' key, to try a rubric without "
                             "editing the packaged one; each distinct rubric caches separately, "
                             "so switching back and forth re-judges nothing "
                             "(default: the packaged conversation_judge.yaml)")
    parser.add_argument("--judge-backend", default=None, choices=sorted(JUDGE_BACKENDS),
                        help=f"where the judge runs (default: {DEFAULT_JUDGE_BACKEND}): 'deepseek' "
                             "is the hosted, billed deployment; 'local' is a self-hosted vLLM "
                             "server (scripts/serve_qwen.sh), free")
    parser.add_argument("--judge-base-url", default=None,
                        help="--judge-backend local only: the server's /v1 URL (default: "
                             "$LOCAL_LLM_BASE_URL, else http://localhost:$LOCAL_LLM_PORT/v1)")
    parser.add_argument("--judge-model", default=None,
                        help="override the judge model id (local default: whatever the server "
                             "serves, read from its /v1/models)")
    parser.add_argument("--judge-temperature", type=float, default=None,
                        help="override the judge's sampling temperature (default per backend: "
                             + ", ".join(f"{name} {backend.temperature}"
                                         for name, backend in sorted(JUDGE_BACKENDS.items()))
                             + "; DeepSeek ignores it while reasoning is on)")
    parser.add_argument("--judge-reasoning-effort", default=None,
                        help="how much the judge thinks before answering (default per backend: "
                             + ", ".join(f"{name} {backend.reasoning_effort}"
                                         for name, backend in sorted(JUDGE_BACKENDS.items()))
                             + "). deepseek: none, low, high, max -- 'none' is ~6x fewer output "
                             "tokens, 'max' the only level measurably above 'low'. local vLLM: "
                             "none, low, medium, xhigh, validated by the server")
    parser.add_argument("--out", default=None,
                        help="where to write the merged score table "
                             "(default: experiments/utility/<source>_<defense>.csv)")
    args = parser.parse_args()

    # The judge flags configure the API-backed metric only -- the local scorers take a checkpoint,
    # not a rubric -- so passing them to a run that never judges anything is a mistake worth naming
    # rather than silently ignoring.
    judge_flags = {"--judge-prompt": args.judge_prompt, "--judge-model": args.judge_model,
                   "--judge-backend": args.judge_backend, "--judge-base-url": args.judge_base_url,
                   "--judge-temperature": args.judge_temperature,
                   "--judge-reasoning-effort": args.judge_reasoning_effort}
    misapplied = [flag for flag, value in judge_flags.items() if value is not None]
    if "conversation" not in args.metric and misapplied:
        parser.error(f"{', '.join(misapplied)} only applies to --metric conversation, which this "
                     f"run does not include (--metric {' '.join(args.metric)}).")
    if args.judge_base_url and args.judge_backend != "local":
        parser.error("--judge-base-url only applies with --judge-backend local.")
    overrides = {}
    if args.judge_backend:
        overrides["judge_backend"] = args.judge_backend
    if args.judge_base_url:
        overrides["judge_base_url"] = args.judge_base_url
    if args.judge_prompt:
        overrides["judge_system_prompt"] = load_judge_prompt(args.judge_prompt)
        print(f"judge rubric: {args.judge_prompt}")
    if args.judge_model:
        overrides["judge_model"] = args.judge_model
    if args.judge_temperature is not None:
        overrides["judge_temperature"] = args.judge_temperature
    if args.judge_reasoning_effort is not None:
        overrides["judge_reasoning_effort"] = args.judge_reasoning_effort

    if args.label and not args.defended_file:
        parser.error("--label only applies with --defended-file.")
    defense_name = args.defense
    if args.defended_file:
        stem = args.defended_file.name.removesuffix(".parquet").removesuffix(".salvaged")
        defense_name = args.label or stem.removeprefix(f"{args.source}_")
    data_dir = Path(args.data_dir) if args.data_dir else hf_dir()
    cache_root = Path(args.cache_dir) if args.cache_dir else default_cache_dir()
    reference, defended, total = load_pair(args.source, defense_name, data_dir, args.limit,
                                           args.seed, defended_path=args.defended_file)

    frames: list[pd.DataFrame] = []
    spent = 0.0
    for metric in args.metric:
        result = eval_utility(metric, defended, cache_dir=cache_root, reference=reference,
                              side="unknown", **(overrides if metric == "conversation" else {}))
        # The sampling happened here rather than inside the metric (so only the sampled rows' text
        # was ever read), so report it here too -- otherwise a sampled run's summary reads like a
        # full one.
        if len(result.table) < total:
            result.sampled_from = total
        print(result.summary())
        if metric == "conversation":
            print(result.usage.summary(result.judge_rates))
            spent += float(np.nansum(result.table["cost_usd"]))
        frames.append(result.scores())

    out_path = Path(args.out) if args.out else OUTPUT_DIR / f"{args.source}_{defense_name}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged = merge_scores(read_existing(out_path), frames)
    # Six significant figures: these are model scores read to two or three decimals, and full
    # float repr makes a column of 0.140238363908106 that is harder to skim and no more true. It
    # is enough for the cost column too, where a per-conversation charge is ~$0.0002.
    merged.to_csv(out_path, index=False, float_format="%.6g")

    print(f"wrote {len(merged):,} rows x {len(merged.columns)} columns -> {out_path}")
    if "judge_cost_usd" in merged.columns:
        # The file is the spend record: nothing else logs what the judge cost, and summing this
        # column over every such file is the project's cumulative API bill.
        print(f"judge cost: ${spent:.4f} this run, ${np.nansum(merged['judge_cost_usd']):.4f} "
              f"cumulative for {args.source}/{defense_name}")


if __name__ == "__main__":
    main()
