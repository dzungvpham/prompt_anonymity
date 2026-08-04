#!/usr/bin/env python
"""Drive ``run_experiment.py`` over the whole experiment grid, once per cell.

The grid is ``{dataset} x {defense} x {feature} x {attack}`` -- :data:`SOURCES`,
:data:`DEFENSES`, :data:`FEATURES`, :data:`ATTACKS`, minus the cells :data:`SOURCE_ATTACKS`
rules out. Each cell becomes one ``run_experiment.py`` invocation and one results directory,
``experiments/results/<dataset>_<defense>_<feature>_<attack>/``, which is the four-part name
``experiments/plot_results.py`` parses. Nothing here plots; run that afterwards.

Two things are skipped, and the distinction matters when reading the plan:

* **Already run.** A cell is done when its directory holds a ``rolling_results.csv`` row for
  every known configuration in :data:`KNOWN_CONFIGS` *and* the matching ``predictions_*.csv``
  beside it (see :func:`completed_configs`). A directory that covers only some of them -- a
  sharded run that was killed part way, which is a real outcome under the 16 GB job cap -- is
  reported as partial and re-run rather than counted as done. ``--force`` re-runs everything
  that has data regardless.
* **No data.** A cell needs ``<split>.parquet`` and ``<split>[_<defense>]_<feature>.parquet``
  in ``--data-dir`` (``data/hf`` by default, *not* ``data/dist`` -- see the note on
  :data:`DATA_DIR`). Missing vectors are not something this script can fix, so those cells are
  reported with the command that would produce them and are skipped even under ``--force``.

Run it::

    python experiments/run_all_experiments.py --dry-run   # what would run, and what would not
    python experiments/run_all_experiments.py             # run the missing cells, cheapest first
    python experiments/run_all_experiments.py --force     # re-run every cell that has data
    # one slice, e.g. to shard the expensive attack across jobs:
    python experiments/run_all_experiments.py --attacks xgboost --defenses openanonymity
    # anything after `--` is appended to every run_experiment.py command line:
    python experiments/run_all_experiments.py -- --no-tune

Cells run **cheapest attack first** (:data:`ATTACKS` is in increasing cost order), so a batch
that is interrupted has completed the runs that were quick to redo. A failing cell does not stop
the rest unless ``--stop-on-error`` is given; the exit status is non-zero if any cell failed.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNNER = REPO_ROOT / "experiments" / "run_experiment.py"
RESULTS_DIR = REPO_ROOT / "experiments" / "results"

#: Where the feature parquets are read from. This is the *mirror of the published dataset* that
#: ``prompt_anonymity.data.download`` writes -- the same default ``run_experiment.py`` uses --
#: and deliberately not ``data/dist``, where ``build_dataset`` / ``compute_features`` /
#: ``apply_defenses`` write. The two hold overlapping copies of the same filenames, so a parquet
#: that was just built is invisible here until it is copied across or ``--data-dir data/dist`` is
#: passed. That is also why "no data" below is reported against this directory specifically.
DATA_DIR = REPO_ROOT / "data" / "hf"


# --- the grid ----------------------------------------------------------------
#
# Kept as literals rather than imported from the package registries, for the same reason
# plot_results.py keeps its vocabulary literal: this is a launcher, and it should be able to
# print a plan without paying for sklearn and the attack package. The names must match
# `prompt_anonymity.defenses.DEFENSES`, `prompt_anonymity.features.FEATURIZERS` and
# `prompt_anonymity.attacks.ATTRIBUTION_ATTACKS` -- an unknown one is rejected up front by
# `resolve_choices` rather than discovered when the subprocess fails.

SOURCES = ("wildchat", "swe_chat")

#: How an undefended run spells its defense: ``base`` in a directory name, ``none`` on the
#: runner's ``--defense`` flag, and no infix at all in the feature parquet's filename.
NO_DEFENSE = "base"

DEFENSES = (NO_DEFENSE, "styleremix", "openanonymity")

FEATURES = ("stylometrix", "gemini_embedding_2")

#: In increasing cost order, which is the order cells are executed in. ``nearest_neighbor`` is a
#: matmul; ``logistic`` and ``xgboost`` fit one decision function per author, so their cost grows
#: with the author count (xgboost measured ~0.097 s per author per 20 boosting rounds).
ATTACKS = ("nearest_neighbor", "logistic", "xgboost")

#: Per-source attack restrictions -- a source absent here gets all of :data:`ATTACKS`.
#:
#: WildChat is nearest-neighbour only. It has 19,711 known authors at the largest configuration
#: and both discriminative attacks are linear in that: xgboost would be ~8 h for a *single* fit
#: at the default 300 estimators, and one-vs-all logistic over ~20,000 classes is in the same
#: territory. These are excluded by design, not pending -- do not add them back without a
#: measurement showing the fit is affordable.
SOURCE_ATTACKS = {"wildchat": ("nearest_neighbor",)}

#: The known configurations every cell is expected to produce, i.e. ``run_experiment.py``'s
#: ``DEFAULT_KNOWN_WINDOWS``. Used only to decide whether a directory is complete; this script
#: never passes ``--known-windows``, so a run that gets a different set of these (because the
#: corpus is too small for one of them) would look permanently incomplete -- which has not
#: happened on either corpus, and would be visible as a cell that re-runs every time.
KNOWN_CONFIGS = ("known0025", "known2550", "known5075", "known0050", "known2575", "known0075")


@dataclass(frozen=True)
class Cell:
    """One point of the grid: one runner invocation, one results directory."""

    source: str
    defense: str
    feature: str
    attack: str

    @property
    def tag(self) -> str:
        """The results directory name -- ``run_experiment.output_tag`` for a default run.

        Four parts, positional, ``base`` for no defense. The dataset part is the source name
        verbatim, which is also its parquet's base name -- one spelling per corpus. Every flag
        this script passes is a default as far as ``output_tag`` is concerned, so the name stays
        in the comparable set that ``plot_results.py`` draws.
        """
        return f"{self.source}_{self.defense}_{self.feature}_{self.attack}"

    def feature_parquet(self, data_dir: Path) -> Path:
        """The vectors this cell attacks: ``<split>[_<defense>]_<feature>.parquet``."""
        infix = "" if self.defense == NO_DEFENSE else f"_{self.defense}"
        return data_dir / f"{self.source}{infix}_{self.feature}.parquet"

    def defended_parquet(self, data_dir: Path) -> Path | None:
        """The *defended text* the vectors would be computed from, or ``None`` if undefended.

        Not an input to the run -- ``run_experiment.py`` reads vectors only -- but its
        presence is what separates "the defense has not been applied to this corpus" from "it
        has, and only the featurization is missing", which are different jobs to go and run.
        """
        if self.defense == NO_DEFENSE:
            return None
        return data_dir / f"{self.source}_{self.defense}.parquet"


def build_grid() -> list[Cell]:
    """Every cell of the grid, in execution order: cheapest attack first.

    Sorting by attack rather than by source means an interrupted batch has finished all the
    nearest-neighbour runs -- the ones that are cheap to redo -- rather than a random prefix.
    Within an attack the order is source, defense, feature, so the plan reads in blocks.
    """
    return [
        Cell(source=source, defense=defense, feature=feature, attack=attack)
        for attack in ATTACKS
        for source in SOURCES
        for defense in DEFENSES
        for feature in FEATURES
        if attack in SOURCE_ATTACKS.get(source, ATTACKS)
    ]


# --- what is already on disk -------------------------------------------------

def completed_configs(run_dir: Path, attack: str) -> set[str]:
    """The known configurations ``run_dir`` holds a finished result for.

    A configuration counts only when **both** halves of its output are there: a summary row in
    ``rolling_results.csv`` and its own ``predictions_<attack>_<config>.csv``. The summary is
    written per run and the per-document file per configuration, so checking one alone would
    call a half-written directory done -- and ``predictions_*.csv`` is the file every figure is
    rebuilt from, so a missing one is a cell that cannot be plotted however complete its
    summary looks.

    The attack is checked because ``rolling_results.csv`` carries an ``attack`` column and a
    directory could in principle have been written by a multi-attack run.
    """
    rolling = run_dir / "rolling_results.csv"
    if not rolling.exists():
        return set()
    try:
        with rolling.open(newline="", encoding="utf-8") as handle:
            scored = {row["known_config"] for row in csv.DictReader(handle)
                      if row.get("attack") == attack}
    except (OSError, KeyError):        # unreadable or written by an older, different schema
        return set()
    return {config for config in scored
            if (run_dir / f"predictions_{attack}_{config}.csv").exists()}


@dataclass
class Plan:
    """What this script decided to do about one cell, and why."""

    cell: Cell
    action: str                        # "run", "done", or "no-data"
    reason: str = ""

    @property
    def will_run(self) -> bool:
        return self.action == "run"


def plan_cell(cell: Cell, data_dir: Path, results_dir: Path, force: bool) -> Plan:
    """Decide whether ``cell`` runs, and record the reason it does not.

    Data availability is checked first and is not overridable: ``--force`` re-runs work, it
    cannot conjure vectors that were never computed.
    """
    documents = data_dir / f"{cell.source}.parquet"
    if not documents.exists():
        return Plan(cell, "no-data", f"{documents.name} missing -- build the dataset first")

    features = cell.feature_parquet(data_dir)
    if not features.exists():
        defended = cell.defended_parquet(data_dir)
        if defended is not None and not defended.exists():
            reason = (f"{features.name} missing; so is {defended.name} -- run "
                      f"`apply_defenses --source {cell.source} --defense {cell.defense}` first")
        else:
            reason = (f"{features.name} missing -- run `compute_features --source {cell.source} "
                      f"--feature {cell.feature}"
                      + (f" --defense {cell.defense}" if defended is not None else "") + "`")
        return Plan(cell, "no-data", reason)

    run_dir = results_dir / cell.tag
    done = completed_configs(run_dir, cell.attack)
    if force:
        return Plan(cell, "run", "forced" if done else "")
    if set(KNOWN_CONFIGS) <= done:
        return Plan(cell, "done", f"{len(KNOWN_CONFIGS)} known configs in {run_dir.name}")
    if done:
        missing = [config for config in KNOWN_CONFIGS if config not in done]
        return Plan(cell, "run", f"partial: {len(done)}/{len(KNOWN_CONFIGS)} present, "
                                 f"missing {', '.join(missing)}")
    return Plan(cell, "run", "")


# --- running -----------------------------------------------------------------

def runner_command(cell: Cell, args: argparse.Namespace) -> list[str]:
    """The ``run_experiment.py`` command line for one cell.

    Only the four grid axes and the settings that decide *where* the work happens are passed;
    everything else is left at the runner's default, which is what keeps the output directory
    name to its four parts and the run inside the comparable set. ``--xgboost-device`` is the one
    value this script sets away from that default (to ``auto``, see :func:`parse_args`); it is
    not part of ``output_tag``, so it changes where the trees are fitted and nothing else about
    how the run is filed. ``--extra`` is appended last so it can override anything here.
    """
    command = [sys.executable, str(RUNNER),
               "--source", cell.source,
               "--feature", cell.feature,
               "--attacks", cell.attack,
               "--data-dir", str(args.data_dir)]
    if cell.defense != NO_DEFENSE:
        command += ["--defense", cell.defense]
    if cell.attack == "xgboost":
        command += ["--xgboost-device", args.xgboost_device]
    return command + args.extra


def run_cell(cell: Cell, args: argparse.Namespace) -> tuple[bool, float]:
    """Run one cell to completion; return ``(succeeded, seconds)``.

    Output goes to the terminal unless ``--log-dir`` is set, in which case each cell gets its own
    ``<tag>.log`` and only the tail of a failing one is echoed -- a 12-cell batch otherwise
    interleaves nothing useful into a scrollback.
    """
    started = time.monotonic()
    if args.log_dir is None:
        result = subprocess.run(runner_command(cell, args), cwd=REPO_ROOT)
    else:
        args.log_dir.mkdir(parents=True, exist_ok=True)
        log_path = args.log_dir / f"{cell.tag}.log"
        print(f"  logging to {log_path}")
        with log_path.open("w", encoding="utf-8") as log:
            result = subprocess.run(runner_command(cell, args), cwd=REPO_ROOT,
                                    stdout=log, stderr=subprocess.STDOUT)
        if result.returncode != 0:
            print(f"  --- last 20 lines of {log_path.name} ---")
            for line in log_path.read_text(encoding="utf-8").splitlines()[-20:]:
                print(f"  {line}")
    return result.returncode == 0, time.monotonic() - started


# --- reporting ---------------------------------------------------------------

#: Column width for the cell tag in the printed plan. Long enough for the longest name the grid
#: can produce (``swe_chat_openanonymity_gemini_embedding_2_nearest_neighbor``).
TAG_WIDTH = 58


def print_plan(plans: list[Plan], header: str) -> None:
    """The plan as one line per cell, grouped by what will happen to it."""
    print(f"\n{header}")
    print("-" * (TAG_WIDTH + 22))
    for action, label in (("run", "RUN"), ("done", "SKIP (done)"), ("no-data", "SKIP (no data)")):
        for plan in (plan for plan in plans if plan.action == action):
            note = f"  {plan.reason}" if plan.reason else ""
            print(f"{label:<15} {plan.cell.tag:<{TAG_WIDTH}}{note}")
    counts = {action: sum(plan.action == action for plan in plans)
              for action in ("run", "done", "no-data")}
    print("-" * (TAG_WIDTH + 22))
    print(f"{len(plans)} cells: {counts['run']} to run, {counts['done']} already done, "
          f"{counts['no-data']} without data")


def format_duration(seconds: float) -> str:
    return f"{int(seconds) // 60}m{int(seconds) % 60:02d}s"


# --- CLI ---------------------------------------------------------------------

def resolve_choices(selected: list[str] | None, available: tuple[str, ...], flag: str) -> tuple[str, ...]:
    """Validate a subset filter against the grid axis it narrows.

    Rejecting an unknown value here rather than passing it through means a typo costs a message
    instead of a cell that silently never matches and a batch that quietly does nothing.
    """
    if not selected:
        return available
    unknown = [value for value in selected if value not in available]
    if unknown:
        raise SystemExit(f"{flag}: unknown value(s) {', '.join(unknown)}; "
                         f"this grid's are {', '.join(available)}.")
    return tuple(value for value in available if value in selected)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the plan -- which cells would run, which are already done, "
                             "and which have no data -- and exit without running anything.")
    parser.add_argument("--force", action="store_true",
                        help="Re-run every cell that has data, including finished ones. Cells "
                             "whose feature parquet is missing are still skipped; this forces "
                             "work to be redone, it cannot create missing vectors.")
    parser.add_argument("--sources", nargs="+", metavar="NAME",
                        help=f"Restrict to these datasets (default: all of {', '.join(SOURCES)}).")
    parser.add_argument("--defenses", nargs="+", metavar="NAME",
                        help=f"Restrict to these defenses, '{NO_DEFENSE}' for undefended "
                             f"(default: all of {', '.join(DEFENSES)}).")
    parser.add_argument("--features", nargs="+", metavar="NAME",
                        help=f"Restrict to these features (default: all of {', '.join(FEATURES)}).")
    parser.add_argument("--attacks", nargs="+", metavar="NAME",
                        help=f"Restrict to these attacks (default: all of {', '.join(ATTACKS)}, "
                             f"subject to the per-source restrictions).")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR,
                        help="Directory holding the parquets, passed straight to the runner "
                             "(default: data/hf, NOT data/dist -- see the module docstring).")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR,
                        help="Where results directories live; what is scanned to decide whether "
                             "a cell is already done (default: experiments/results).")
    parser.add_argument("--log-dir", type=Path, default=None,
                        help="Write each cell's output to <log-dir>/<tag>.log instead of the "
                             "terminal, echoing only the tail of a failing run.")
    parser.add_argument("--xgboost-device", default="auto", choices=["cpu", "cuda", "auto"],
                        help="Passed to the runner for xgboost cells only. Default 'auto': use "
                             "the GPU histogram builder when the wheel has CUDA and a device is "
                             "visible (xgboost probes with a two-row fit), else the CPU one. "
                             "Measured 17x on the shape these cells have (1,000 documents x "
                             "3,072 features, 81 authors: 22.1 s CPU against 1.29 s on one "
                             "A100). Note this overrides the runner's own 'cpu' default, and the "
                             "reason for that default applies here too: the GPU builder sums "
                             "gradients in a different order and can pick different splits, so a "
                             "batch's xgboost numbers depend on whether a GPU was free. Pass "
                             "'cpu' when a cell has to reproduce one run on a different machine.")
    parser.add_argument("--stop-on-error", action="store_true",
                        help="Abort the batch at the first failing cell (default: carry on and "
                             "report the failures at the end).")
    parser.add_argument("extra", nargs="*", metavar="-- RUNNER ARGS",
                        help="Everything after a bare `--` is appended to every runner command "
                             "line, e.g. `-- --no-tune`.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sources = resolve_choices(args.sources, SOURCES, "--sources")
    defenses = resolve_choices(args.defenses, DEFENSES, "--defenses")
    features = resolve_choices(args.features, FEATURES, "--features")
    attacks = resolve_choices(args.attacks, ATTACKS, "--attacks")

    grid = [cell for cell in build_grid()
            if cell.source in sources and cell.defense in defenses
            and cell.feature in features and cell.attack in attacks]
    if not grid:
        raise SystemExit("no cells selected: the --sources/--defenses/--features/--attacks "
                         "filters do not intersect (remember WildChat is nearest_neighbor only).")

    plans = [plan_cell(cell, args.data_dir, args.results_dir, args.force) for cell in grid]
    print_plan(plans, f"grid: {len(grid)} cells, data from {args.data_dir}")

    if args.dry_run:
        print("\n--dry-run: nothing was executed.")
        return 0

    queue = [plan.cell for plan in plans if plan.will_run]
    if not queue:
        print("\nnothing to run.")
        return 0

    failed, batch_started = [], time.monotonic()
    for index, cell in enumerate(queue, start=1):
        print(f"\n=== [{index}/{len(queue)}] {cell.tag}")
        print("  " + " ".join(runner_command(cell, args)))
        succeeded, elapsed = run_cell(cell, args)
        print(f"  {'ok' if succeeded else 'FAILED'} in {format_duration(elapsed)}")
        if not succeeded:
            failed.append(cell)
            if args.stop_on_error:
                print("  --stop-on-error: aborting the batch.")
                break

    print(f"\n{len(queue) - len(failed)}/{len(queue)} cells succeeded in "
          f"{format_duration(time.monotonic() - batch_started)}")
    for cell in failed:
        print(f"  FAILED  {cell.tag}")
    if not failed:
        print("draw the figures with: python experiments/plot_results.py")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
