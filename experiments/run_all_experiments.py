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

Submitting to SLURM
-------------------

``--slurm`` submits **one job per cell** instead of running it here, and is otherwise the same
launcher: the same grid, the same skip rules, the same ``--`` passthrough. It only submits, so it
belongs on a login node and returns in seconds::

    python experiments/run_all_experiments.py --slurm --dry-run   # print the sbatch lines, submit nothing
    python experiments/run_all_experiments.py --slurm             # submit the missing cells

**The split of knowledge is the point.** This file decides *what* runs and which resource
*class* each cell needs -- :func:`resource_profile`, which knows that xgboost wants an
accelerator and WildChat wants memory, and nothing about any cluster. Two files outside it hold
everything site-specific, and they are the only ones another user edits:

* ``scripts/slurm.toml`` -- profile name to ``sbatch`` flags (partitions, limits, accounting).
  Overridable with ``--slurm-config`` / ``$PROMPT_ANONYMITY_SLURM_CONFIG``; one-off flags go on
  the command line with ``--slurm-arg``.
* ``scripts/slurm_job.sh`` -- how a batch shell here builds a working conda env. It carries no
  ``#SBATCH`` directives; every resource flag comes from the TOML.

One job per cell rather than a job array, because cells differ in argv *and* in resource class --
an array shares one allocation, so the nearest-neighbour tasks would each hold a GPU. Per-cell
jobs also fail independently, which is what ``--stop-on-error`` buys in the local path and what
is lost the moment the work is asynchronous (so the two flags are rejected together).

**A queued cell is skipped.** "Already done" is read off ``rolling_results.csv``, which does not
exist while a job is still pending, so re-running the launcher would otherwise submit the whole
grid a second time. Each job is named after its cell, and :func:`queued_job_names` asks ``squeue``
what is already in flight. That guard is *not* lifted by ``--force``: two jobs writing one results
directory corrupt it, so resubmitting means cancelling the job first.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
import tomllib
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNNER = REPO_ROOT / "experiments" / "run_experiment.py"
RESULTS_DIR = REPO_ROOT / "experiments" / "results"

#: Batch wrapper every SLURM job runs: it builds the environment and execs the runner command.
SLURM_WRAPPER = REPO_ROOT / "scripts" / "slurm_job.sh"
#: Default resource config, read only under ``--slurm``. See the module docstring for the split.
SLURM_CONFIG = REPO_ROOT / "scripts" / "slurm.toml"
#: Points at a config file directly, for a machine whose settings are not committed here.
SLURM_CONFIG_ENV = "PROMPT_ANONYMITY_SLURM_CONFIG"
#: Where per-job logs go when ``--log-dir`` is not given.
DEFAULT_LOG_DIR = REPO_ROOT / "scripts" / "logs"

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

#: The collision-seeding arm. ``collision_seeding`` is the defense; the other three are the
#: comparisons that make its result readable, and all four are cheap to produce (the defense is
#: pure string work -- only the featurize and attack stages here cost anything):
#:
#: * ``_k4`` sweeps the privacy knob to larger collision groups (K=4 rather than 12).
#: * ``_full`` applies each marker to 100% of an author's documents instead of 40-70%. Expected to
#:   do WORSE than the default despite the bigger edit, because perfect consistency is a perfectly
#:   reliable feature.
#: * ``_indep`` drops the profile codebook for independent per-author marker draws. Expected to do
#:   worse than ``base``, i.e. worse than no defense, because a near-unique marker combination is a
#:   fingerprint. It is the control showing the codebook is what does the work.
#:
#: ``_k24`` is registered but left out of the default grid: it sits between ``_k4`` and the default
#: and adds a cell to every source without changing the story. Add it with ``--defenses``.
COLLISION_SEEDING = ("collision_seeding", "collision_seeding_k4",
                     "collision_seeding_full", "collision_seeding_indep")

#: The frame-shift arm. ``frame_shift`` rewrites each document into one of 50 topic-heavy scenes
#: drawn by a keyed hash of its doc_id; ``frame_shift_single`` forces the whole corpus into ONE scene
#: and is the convergence-vs-dilution control (and the control for plain length inflation, since both
#: arms lengthen documents the same way). Unlike collision seeding these cost money to produce -- a
#: hosted rewrite per (frame, turn) -- so only the main arm is in the default grid; add the ablation
#: with ``--defenses frame_shift_single`` once the first numbers are in.
FRAME_SHIFT = ("frame_shift",)

DEFENSES = (NO_DEFENSE, "styleremix", "openanonymity") + COLLISION_SEEDING + FRAME_SHIFT

#: ``char_ngram_tfidf`` is here for collision seeding specifically: character n-grams are the
#: channel its markers live in (spelling, punctuation, casing), so it is where the effect should be
#: largest, while ``gemini_embedding_2`` is semantic and should barely move. ``stylometrix`` is what
#: every earlier defense was measured on and is what keeps the numbers comparable to them.
FEATURES = ("stylometrix", "char_ngram_tfidf", "gemini_embedding_2")

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


# --- resource classes --------------------------------------------------------
#
# What a cell needs from a machine, named rather than spelled out: these are the profile names
# `scripts/slurm.toml` maps to sbatch flags. Keeping the *rule* here and the *flags* there is what
# lets another cluster be adopted by editing one TOML -- the rule is a property of the experiment
# and travels with it, the flags are not.

#: Attacks worth allocating a GPU for. Only xgboost has a device to use (measured 17x: 22.1 s CPU
#: against 1.29 s on one A100 at 1,000 documents x 3,072 features over 81 authors). Every other
#: attack here is BLAS on the CPU and would leave a card idle for the whole job.
GPU_ATTACKS = ("xgboost",)

#: Sources whose score matrix does not fit a default allocation. WildChat's is 86,255 x 13,694
#: float32 = 4.72 GB at ``known0050`` alone; runs have been OOM-killed at 16 GB.
LARGE_MEMORY_SOURCES = ("wildchat",)


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


def resource_profile(cell: Cell) -> str:
    """The ``scripts/slurm.toml`` profile this cell should be submitted under.

    Deliberately coarse -- three classes, not a per-cell resource table. A cell that genuinely
    needs something its class does not give it belongs in the TOML's ``[overrides.<tag>]``, which
    keeps the exception next to the numbers it is an exception to.
    """
    if cell.attack in GPU_ATTACKS:
        return "gpu"
    if cell.source in LARGE_MEMORY_SOURCES:
        return "cpu_large"
    return "cpu"


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


def queued_job_names() -> frozenset[str]:
    """Job names this user already has pending or running, as reported by ``squeue``.

    Jobs are named after their cell, so this is what stops a second launch from submitting the
    whole grid again while the first one is still in the queue -- ``rolling_results.csv`` only
    appears when a run *finishes*, so the "already done" check cannot see in-flight work.

    A cluster without ``squeue``, or a ``squeue`` that fails, gets a warning and an empty set:
    losing the guard is a duplicate submission, while refusing to launch over it would break a
    site whose scheduler is queried some other way.
    """
    if shutil.which("squeue") is None:
        print("note: squeue not found -- cannot check for jobs already in the queue.")
        return frozenset()
    try:
        result = subprocess.run(["squeue", "--noheader", "--user", os.environ.get("USER", ""),
                                 "--format=%j"],
                                capture_output=True, text=True, timeout=60, check=True)
    except (subprocess.SubprocessError, OSError) as error:
        print(f"note: squeue failed ({error}) -- not checking for jobs already in the queue.")
        return frozenset()
    return frozenset(line.strip() for line in result.stdout.splitlines() if line.strip())


@dataclass
class Plan:
    """What this script decided to do about one cell, and why."""

    cell: Cell
    action: str                        # "run", "done", "queued", or "no-data"
    reason: str = ""

    @property
    def will_run(self) -> bool:
        return self.action == "run"


def plan_cell(cell: Cell, data_dir: Path, results_dir: Path, force: bool,
              queued: frozenset[str] = frozenset()) -> Plan:
    """Decide whether ``cell`` runs, and record the reason it does not.

    Data availability is checked first and is not overridable: ``--force`` re-runs work, it
    cannot conjure vectors that were never computed. The queue check comes next and is *also* not
    overridable -- two jobs writing one results directory interleave their ``predictions_*.csv``
    and leave it neither run's output, so a resubmission means cancelling the job first.
    """
    documents = data_dir / f"{cell.source}.parquet"
    if not documents.exists():
        return Plan(cell, "no-data", f"{documents.name} missing -- build the dataset first")

    if cell.tag in queued:
        return Plan(cell, "queued", "a job of this name is already pending or running "
                                    "(scancel it to resubmit)")

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


# --- submitting to SLURM ------------------------------------------------------

@dataclass(frozen=True)
class SlurmConfig:
    """``scripts/slurm.toml``, parsed: the site's spelling of each resource class.

    Three tables, all optional and all holding nothing but ``sbatch`` flags -- ``defaults``
    applied to every job, ``profiles`` keyed by :func:`resource_profile`'s return value, and
    ``overrides`` keyed by a cell tag. There is no schema of our own beyond that, on purpose: a
    flag this launcher has never heard of still works, because it is passed straight through.
    """

    path: Path
    defaults: tuple[str, ...]
    profiles: dict[str, tuple[str, ...]]
    overrides: dict[str, tuple[str, ...]]

    def flags_for(self, cell: Cell) -> list[str]:
        """The ``sbatch`` flags for one cell, in precedence order (later wins)."""
        return [*self.defaults,
                *self.profiles[resource_profile(cell)],
                *self.overrides.get(cell.tag, ())]

    def check(self, cells: list[Cell]) -> None:
        """Fail before anything is submitted if a cell's profile is not in the file.

        Checked for the whole batch up front rather than at each submission, so a missing profile
        is a message instead of half a grid queued and the rest abandoned.
        """
        wanted = {resource_profile(cell) for cell in cells}
        missing = sorted(wanted - set(self.profiles))
        if missing:
            raise SystemExit(
                f"{self.path}: no [profiles.{missing[0]}] table"
                + (f" (also missing: {', '.join(missing[1:])})" if len(missing) > 1 else "")
                + f". This grid needs {', '.join(sorted(wanted))}; the file defines "
                + f"{', '.join(sorted(self.profiles)) or 'none'}.")


#: ``$VAR``, ``${VAR}`` or ``${VAR:-fallback}`` inside a config flag. Deliberately not full shell
#: syntax -- these are sbatch flags, not commands, and the two forms below are the whole feature.
VARIABLE_PATTERN = re.compile(r"\$(?:(\w+)|\{(\w+)(?::-([^}]*))?\})")


def load_environment_file() -> None:
    """Merge a ``.env`` into the environment, the way the rest of the project does.

    Same mechanism as ``prompt_anonymity.data.config``: ``python-dotenv`` walks up from the
    working directory and never overwrites a variable already set for real. Optional at runtime,
    because a flag with no ``$`` in it needs none of this.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()


def expand_flag(flag: str, where: str, path: Path) -> str | None:
    """Substitute environment variables into one config flag; ``None`` means "leave it out".

    This is what keeps a *person* out of the committed config: ``--mail-user=${EMAIL:-}`` works
    for whoever set ``EMAIL``, and quietly disappears for whoever did not. The two forms differ in
    what an unset variable means, because the right answer is not the same for every flag:

    * ``$VAR`` / ``${VAR}`` -- **an error**. A missing ``--account=$SLURM_ACCOUNT`` is a job
      rejected at submit time or charged to the wrong place; failing here names the variable
      instead.
    * ``${VAR:-fallback}`` -- the fallback, and an empty one drops the flag entirely. That is the
      spelling for anything optional: no address, no ``--mail-user``, no mail.

    An empty *value* is treated as unset (``EMAIL=`` in a ``.env`` is someone clearing it, not
    asking for an empty address), and a flag left with a trailing ``=`` is dropped rather than
    passed to sbatch as ``--mail-user=``.
    """
    def substitute(match: re.Match[str]) -> str:
        name = match.group(1) or match.group(2)
        value = (os.environ.get(name) or "").strip()
        if value:
            return value
        if match.group(3) is not None:
            return match.group(3)
        raise SystemExit(f"{path}: {where} refers to ${name}, which is not set. Set it (a `.env` "
                         f"beside the project is read too), or write ${{{name}:-...}} to give it "
                         f"a fallback -- an empty one leaves the flag out.")

    expanded = VARIABLE_PATTERN.sub(substitute, flag)
    if not expanded or expanded.endswith("="):
        print(f"note: {path.name}: {where} leaves out '{flag}' (nothing to substitute).")
        return None
    return expanded


def load_slurm_config(explicit: Path | None) -> SlurmConfig:
    """Read the resource config: ``--slurm-config``, then the environment, then the default.

    Resolution mirrors ``prompt_anonymity.data.config`` -- an explicit path wins, then
    ``$PROMPT_ANONYMITY_SLURM_CONFIG`` for a machine whose settings are not committed, then
    ``scripts/slurm.toml``. There is no fallback set of flags baked into this file: a guessed
    partition either fails at submit time or, worse, quietly runs the batch somewhere unintended.
    """
    path = explicit or Path(os.environ.get(SLURM_CONFIG_ENV) or SLURM_CONFIG)
    if not path.exists():
        raise SystemExit(f"{path} not found. --slurm needs a resource config; copy the one at "
                         f"{SLURM_CONFIG.relative_to(REPO_ROOT)} and edit it for your cluster, or "
                         f"point at yours with --slurm-config / ${SLURM_CONFIG_ENV}.")
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise SystemExit(f"{path}: {error}") from error
    load_environment_file()

    def flags(table: dict, where: str) -> tuple[str, ...]:
        value = table.get("sbatch", [])
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            raise SystemExit(f"{path}: {where}.sbatch must be a list of strings, e.g. "
                             f'sbatch = ["--partition=cpu", "--mem=16g"]')
        expanded = (expand_flag(flag, where, path) for flag in value)
        return tuple(flag for flag in expanded if flag is not None)

    return SlurmConfig(
        path=path,
        defaults=flags(raw.get("defaults", {}), "[defaults]"),
        profiles={name: flags(table, f"[profiles.{name}]")
                  for name, table in raw.get("profiles", {}).items()},
        overrides={tag: flags(table, f"[overrides.{tag}]")
                   for tag, table in raw.get("overrides", {}).items()})


def sbatch_command(cell: Cell, args: argparse.Namespace, config: SlurmConfig) -> list[str]:
    """The full ``sbatch`` command line for one cell.

    ``sbatch [flags] script [args]`` forwards everything after the script to it, so the runner
    command line arrives at :data:`SLURM_WRAPPER` untouched, ``--`` passthrough included.

    Job name and log path are set first so a flag from the config file or ``--slurm-arg`` can
    still override them -- but the name is what :func:`queued_job_names` matches on, so renaming
    jobs costs the duplicate-submission guard.
    """
    log_dir = (args.log_dir or DEFAULT_LOG_DIR).resolve()
    return ["sbatch",
            f"--job-name={cell.tag}",
            f"--chdir={REPO_ROOT}",
            f"--output={log_dir}/{cell.tag}-%j.out",
            *config.flags_for(cell),
            *args.slurm_arg,
            str(SLURM_WRAPPER),
            *runner_command(cell, args)]


def submit_cell(cell: Cell, args: argparse.Namespace, config: SlurmConfig) -> str | None:
    """Submit one cell; return its job id, or ``None`` if ``sbatch`` refused it."""
    result = subprocess.run(sbatch_command(cell, args, config), cwd=REPO_ROOT,
                            capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  FAILED to submit: {result.stderr.strip() or result.stdout.strip()}")
        return None
    # "Submitted batch job 12345" -- the id is what a later scancel or sacct needs.
    return result.stdout.split()[-1] if result.stdout.split() else "?"


def submit_all(queue: list[Cell], args: argparse.Namespace, config: SlurmConfig) -> int:
    """Submit every planned cell, one job each. Returns the process exit status."""
    log_dir = (args.log_dir or DEFAULT_LOG_DIR).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)   # SLURM fails a job whose --output dir is absent

    submitted, failed = [], []
    for index, cell in enumerate(queue, start=1):
        print(f"\n=== [{index}/{len(queue)}] {cell.tag}  [{resource_profile(cell)}]")
        print("  " + shlex.join(sbatch_command(cell, args, config)))
        job_id = submit_cell(cell, args, config)
        if job_id is None:
            failed.append(cell)
        else:
            submitted.append((job_id, cell))
            print(f"  submitted as job {job_id}")

    print(f"\n{len(submitted)}/{len(queue)} cells submitted; logs in {log_dir}")
    for cell in failed:
        print(f"  FAILED  {cell.tag}")
    if submitted:
        print(f"  watch:   squeue -u {os.environ.get('USER', '$USER')}")
        print(f"  cancel:  scancel {' '.join(job_id for job_id, _ in submitted)}")
        print("  then draw the figures with: python experiments/plot_results.py")
    return 1 if failed else 0


# --- reporting ---------------------------------------------------------------

#: Column width for the cell tag in the printed plan. Long enough for the longest name the grid
#: can produce (``swe_chat_openanonymity_gemini_embedding_2_nearest_neighbor``).
TAG_WIDTH = 58


def print_plan(plans: list[Plan], header: str) -> None:
    """The plan as one line per cell, grouped by what will happen to it."""
    print(f"\n{header}")
    print("-" * (TAG_WIDTH + 22))
    for action, label in (("run", "RUN"), ("done", "SKIP (done)"),
                          ("queued", "SKIP (queued)"), ("no-data", "SKIP (no data)")):
        for plan in (plan for plan in plans if plan.action == action):
            note = f"  {plan.reason}" if plan.reason else ""
            print(f"{label:<15} {plan.cell.tag:<{TAG_WIDTH}}{note}")
    counts = {action: sum(plan.action == action for plan in plans)
              for action in ("run", "done", "queued", "no-data")}
    print("-" * (TAG_WIDTH + 22))
    print(f"{len(plans)} cells: {counts['run']} to run, {counts['done']} already done, "
          + (f"{counts['queued']} already queued, " if counts["queued"] else "")
          + f"{counts['no-data']} without data")


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
                        help="Where each cell's output goes. Locally: <log-dir>/<tag>.log instead "
                             "of the terminal, echoing only the tail of a failing run. Under "
                             f"--slurm: the job's --output, <log-dir>/<tag>-<jobid>.out, "
                             f"defaulting to {DEFAULT_LOG_DIR.relative_to(REPO_ROOT)}.")
    parser.add_argument("--slurm", action="store_true",
                        help="Submit one SLURM job per cell instead of running them here, and "
                             "exit. Resources come from --slurm-config, the environment a job "
                             "runs in from scripts/slurm_job.sh; this script contributes only the "
                             "resource class (see resource_profile). Combine with --dry-run to "
                             "print the sbatch command lines without submitting anything.")
    parser.add_argument("--slurm-config", type=Path, default=None,
                        help=f"Resource config for --slurm (default: ${SLURM_CONFIG_ENV}, else "
                             f"{SLURM_CONFIG.relative_to(REPO_ROOT)}). Maps each profile name to "
                             "sbatch flags; this is the file to edit for another cluster.")
    parser.add_argument("--slurm-arg", action="append", default=[], metavar="FLAG",
                        help="Extra flag appended to every sbatch command line, overriding the "
                             "config file. Repeatable, and needs the `=` spelling so argparse "
                             "does not read the value as an option of its own: "
                             "--slurm-arg=--qos=short --slurm-arg=--account=my-account.")
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

    if args.stop_on_error and args.slurm:
        raise SystemExit("--stop-on-error is meaningless with --slurm: submission returns before "
                         "any cell has run, and the jobs are independent by design.")

    # Read and validate the resource config before touching the queue, so a typo in the TOML costs
    # a message rather than a partly-submitted grid.
    config = load_slurm_config(args.slurm_config) if args.slurm else None
    if config is not None:
        config.check(grid)
        if not SLURM_WRAPPER.exists():
            raise SystemExit(f"{SLURM_WRAPPER} not found: --slurm submits every job through it.")

    queued = queued_job_names() if args.slurm else frozenset()
    plans = [plan_cell(cell, args.data_dir, args.results_dir, args.force, queued)
             for cell in grid]
    print_plan(plans, f"grid: {len(grid)} cells, data from {args.data_dir}"
                      + (f", resources from {config.path}" if config is not None else ""))

    queue = [plan.cell for plan in plans if plan.will_run]

    if args.dry_run:
        if config is not None:
            for cell in queue:
                print(f"\n{cell.tag}  [{resource_profile(cell)}]")
                print("  " + shlex.join(sbatch_command(cell, args, config)))
        print("\n--dry-run: nothing was submitted." if config is not None
              else "\n--dry-run: nothing was executed.")
        return 0

    if not queue:
        print("\nnothing to run.")
        return 0

    if config is not None:
        return submit_all(queue, args, config)

    failed, batch_started = [], time.monotonic()
    for index, cell in enumerate(queue, start=1):
        print(f"\n=== [{index}/{len(queue)}] {cell.tag}")
        print("  " + shlex.join(runner_command(cell, args)))
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
