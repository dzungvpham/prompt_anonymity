#!/usr/bin/env python
"""Drive this project's two attack runners over the whole experiment grid, once per cell.

**One launcher, two families**, selected with ``--families`` (default: both):

* ``attribution`` -- ``run_experiment.py`` over ``{dataset} x {defense} x {feature} x {attack}``,
  minus the cells :data:`SOURCE_ATTACKS` rules out. One cell is one results directory,
  ``experiments/results/<dataset>_<defense>_<feature>_<attack>/``, the four-part name
  ``plot_results.parse_run_name`` parses.
* ``clustering`` -- ``run_clustering.py`` over ``{dataset} x {defense} x {feature}``. There is no
  attack axis: a clustering run attacks the collection with every algorithm in one process and
  files them all in one directory, ``experiments/clustering/<dataset>_<defense>_<feature>/`` --
  the **three**-part name ``plot_results.parse_clustering_run_name`` parses, deliberately a
  separate function so a run of one kind can never parse as the other.

Nothing here plots; run ``plot_results.py`` afterwards.

Each clustering cell is run under every configuration in :data:`CLUSTERING_VARIANTS` -- by
default both ``plain``, the pure-text run the ``by_defense`` figures draw, and
``contrastive_time``, the strongest configuration measured (WildChat test BCubed F 0.510
baseline, 0.537 with the contrastive projection, **0.562** with elapsed time fused in as well).
The variant is the fourth part of the directory name, so the second lands in
``plot_results.py``'s *variants* family rather than the comparable set, which is deliberate: a
learned projection is not a point on a defense's curve.

**The clustering family has no known-side grid**, unlike the attribution one. ``run_clustering.py``
cuts the timeline once at ``KNOWN_FRACTION`` = 0.75 -- the final quarter is the collection under
attack, [0.50, 0.75) is the labelled slice hyper-parameters are selected on, and [0, 0.50) is what
a ``--projection`` is fitted on. There is no ``--known-windows`` and no six-cell grid, so a
clustering cell is one experiment where an attribution cell is six.

**Every cell of a family gets the same command line, and for clustering that is why this script
exists.** The clustering runs that predate it were submitted by hand and did not agree: WildChat
got ``--oracle-sweep`` and an explicit three-algorithm list, swe-chat got the runner's defaults
and all five, so the two corpora answered different questions and neither could be read against
the other. Here :data:`CLUSTERING_ALGORITHMS`, :data:`CLUSTERING_SCOPES` and the oracle sweep are
properties of the *batch*, spelled once and passed to every cell whatever its source.

The one thing that still differs between corpora is not a setting: ``average_linkage`` needs a
dense ``n^2`` distance matrix and the runner skips it, with a note, above ``MAX_DENSE_DOCUMENTS``
= 20,000 documents. That is a property of the method and the corpus, so
:data:`CLUSTERING_SCALE_LIMITED` records it here too -- otherwise a WildChat cell would be
missing a result it was never going to produce and would look permanently unfinished (see
:func:`expected_results`).

Two things are skipped, and the distinction matters when reading the plan:

* **Already run.** What counts as finished is per family. An attribution cell is done when its
  directory holds a ``rolling_results.csv`` row for every known configuration in
  :data:`KNOWN_CONFIGS` *and* the matching ``predictions_*.csv`` beside it
  (:func:`completed_configs`); a clustering cell when ``clustering_results.csv`` holds a row for
  every (scope, algorithm) the batch asked for *and* the matching
  ``clusters_<algorithm>[_unseen].csv`` (:func:`completed_results`). A directory covering only
  some of them -- a run killed part way, which is a real outcome under the 16 GB job cap and on
  ``cpu-preempt`` -- is reported as partial and re-run rather than counted as done. ``--force``
  re-runs everything that has data regardless.
* **No data.** A cell needs ``<split>.parquet`` and ``<split>[_<defense>]_<feature>.parquet``
  in ``--data-dir`` (``data/hf`` by default, *not* ``data/dist`` -- see the note on
  :data:`DATA_DIR`). Missing vectors are not something this script can fix, so those cells are
  reported with the command that would produce them and are skipped even under ``--force``.

Run it::

    python experiments/run_all_experiments.py --dry-run   # what would run, and what would not
    python experiments/run_all_experiments.py             # run the missing cells, cheapest first
    python experiments/run_all_experiments.py --force     # re-run every cell that has data
    # one family, or one slice of one:
    python experiments/run_all_experiments.py --families clustering
    python experiments/run_all_experiments.py --attacks xgboost --defenses openanonymity
    # anything after `--` is appended to every runner command line:
    python experiments/run_all_experiments.py --families attribution -- --no-tune

Cells run **cheapest first**: the families in :data:`FAMILIES` order whatever order
``--families`` names them in, attribution sorted by attack (:data:`ATTACKS` is in increasing cost
order) and clustering by corpus (:data:`CLUSTERING_SOURCE_ORDER` is), so a batch that is
interrupted has completed the runs that were quick to redo. A failing cell does not stop the rest unless ``--stop-on-error`` is given; the exit status
is non-zero if any cell failed.

**Flags that change the output directory name must not go through ``--``.** ``output_tag`` and
``run_clustering.main`` both append any non-default scope choice to the directory name --
``--language-aware``, ``--ood reject``, ``--known-windows`` on one side, ``--known-defense``,
``--standardize``, ``--projection``, ``--rescoring``, ``--time-weight`` on
the other -- which deliberately takes the run out of the comparable set. Passing one here would
leave this script checking, naming and job-guarding a directory the runner never writes. Those
are variant runs; submit them by hand (``scripts/run_clustering_slurm.sh``, or the GPU script for
``--projection contrastive``). In particular this launcher always leaves ``--known-defense`` at
its default: an undefended known side is the deployment threat model and the only condition whose
name stays in the comparable set.

Submitting to SLURM
-------------------

``--slurm`` submits **one job per cell** instead of running it here, and is otherwise the same
launcher: the same grid, the same skip rules, the same ``--`` passthrough. It only submits, so it
belongs on a login node and returns in seconds::

    python experiments/run_all_experiments.py --slurm --dry-run   # print the sbatch lines, submit nothing
    python experiments/run_all_experiments.py --slurm             # submit the missing cells

**The split of knowledge is the point.** This file decides *what* runs and which resource *class*
each cell needs -- :func:`resource_profile`, which knows that xgboost wants an accelerator, that
WildChat wants memory, and that a clustering cell is large in a different place from an
attribution one; and nothing about any cluster. Two files outside it hold everything
site-specific, and they are the only ones another user edits:

* ``scripts/slurm.toml`` -- profile name to ``sbatch`` flags (partitions, limits, accounting).
  Overridable with ``--slurm-config`` / ``$PROMPT_ANONYMITY_SLURM_CONFIG``; one-off flags go on
  the command line with ``--slurm-arg``.
* ``scripts/slurm_job.sh`` -- how a batch shell here builds a working conda env. It carries no
  ``#SBATCH`` directives; every resource flag comes from the TOML.

One job per cell rather than a job array, because cells differ in argv *and* in resource class --
an array shares one allocation, so the nearest-neighbour tasks would each hold a GPU. Per-cell
jobs also fail independently, which is what ``--stop-on-error`` buys in the local path and what
is lost the moment the work is asynchronous (so the two flags are rejected together).

**A queued cell is skipped.** "Already done" is read off a results file that does not exist while
a job is still pending, so re-running the launcher would otherwise submit the whole grid a second
time. Each job is named after its cell, and :func:`queued_job_names` asks ``squeue`` what is
already in flight. That guard is *not* lifted by ``--force``: two jobs writing one directory
corrupt it, so resubmitting means cancelling the job first. Note the by-hand
``scripts/run_clustering_slurm.sh`` names its jobs ``clustering`` rather than after the cell, so
this guard cannot see a clustering run submitted that way.
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
#: Where each family's runner and results live. ``clustering`` is a separate root rather than a
#: subdirectory of ``results/`` because both directory *names* are contracts other scripts parse,
#: and a three-part name sitting among four-part ones would be read by the wrong parser.
RUNNER = REPO_ROOT / "experiments" / "run_experiment.py"
RESULTS_DIR = REPO_ROOT / "experiments" / "results"
CLUSTERING_RUNNER = REPO_ROOT / "experiments" / "run_clustering.py"
CLUSTERING_DIR = REPO_ROOT / "experiments" / "clustering"

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

#: The frame-pad arm: the same 50-scene codebook, but the document's own turns are left
#: byte-identical and one dense, off-topic turn is APPENDED instead. It is frame_shift's other half
#: on its own -- does added shared content dilute the author, with nothing rewritten? -- and it is
#: effectively free to produce (the padding text is generated once into a bank of 400 passages, ~2
#: cents, then reused across the whole corpus), so unlike frame_shift both it and its single-scene
#: ablation could be run; only the main arm is in the default grid to keep the featurize bill down.
FRAME_PAD = ("frame_pad",)

EMBAD = ("embad", "embad_summary", "embad_gemini")

DEFENSES = ((NO_DEFENSE, "styleremix", "openanonymity")
            + COLLISION_SEEDING + FRAME_SHIFT + FRAME_PAD + EMBAD)

#: ``char_ngram_tfidf`` is here for collision seeding specifically: character n-grams are the
#: channel its markers live in (spelling, punctuation, casing), so it is where the effect should be
#: largest, while ``gemini_embedding_2`` is semantic and should barely move. ``stylometrix`` is what
#: every earlier defense was measured on; it stays selectable by name but is out of the default grid.
FEATURES = ("stylometrix", "char_ngram_tfidf", "gemini_embedding_2")

#: Features the attribution grid uses when ``--features`` is not given. StyloMetrix was dropped on
#: 2026-09-25: ``char_ngram_tfidf`` replaced it as the style feature (it beats it on every swe-chat
#: cell and on WildChat's), and the figures are drawn without it.
ATTRIBUTION_DEFAULT_FEATURES = ("char_ngram_tfidf", "gemini_embedding_2")

#: Features the attribution runner fits per known configuration from the document **text**
#: (``prompt_anonymity.features.KNOWN_SIDE_FEATURES``, spelled out here rather than imported to
#: keep this launcher free of the package's imports). They have no feature parquet, so a cell is
#: ready once its text is -- and a clustering cell, whose runner reads parquets only, never is.
KNOWN_SIDE_FEATURES = ("char_ngram_tfidf",)

#: In increasing cost order, which is the order cells are executed in. ``nearest_neighbor`` is a
#: matmul; the rest fit one decision function per author, so their cost grows with the author
#: count (xgboost measured ~0.097 s per author per 20 boosting rounds). ``logistic_sgd`` fits the
#: same model as ``logistic`` but minibatched on a GPU, which is why it sits below it here despite
#: being the only one of the three that runs at WildChat's 19,711 authors: the whole
#: six-configuration StyloMetrix grid is 20 minutes on one A16.
ATTACKS = ("nearest_neighbor", "wccn", "plda", "lda", "rlsc", "logistic_sgd", "logistic", "xgboost")

#: Per-source attack restrictions -- a source absent here gets all of :data:`ATTACKS`.
#:
#: WildChat gets ``nearest_neighbor`` and ``logistic_sgd``, and nothing else. It has 19,711 known
#: authors at the largest configuration and the two excluded attacks are linear in that: xgboost
#: would be ~8 h for a *single* fit at the default 300 estimators, and sklearn's multinomial
#: ``logistic`` is worse than slow -- its per-iteration logit matrix is 129,382 x 19,711 in
#: float64, 20.4 GB against a 16 GB cap, so it cannot run at all. Measured scaling, wildchat
#: StyloMetrix with the pool subsampled: 26 s at 250 authors, 34 s at 500, 64 s at 1,000, 158 s at
#: 2,000. Do not add either back without a measurement showing the fit is affordable.
#:
#: ``logistic_sgd`` is that same ``logistic`` model fitted so that it does run there, and it is
#: not an optional extra: it roughly **doubles** WildChat's StyloMetrix top-1 over
#: ``nearest_neighbor`` (0.0655 -> 0.1387 at ``known0075``, 1.7-2.1x on every configuration), so a
#: grid without it reports a corpus limit where there was only a solver limit.
SOURCE_ATTACKS = {"wildchat": ("nearest_neighbor", "wccn", "rlsc", "logistic_sgd")}

#: The known configurations every cell is expected to produce, i.e. ``run_experiment.py``'s
#: ``DEFAULT_KNOWN_WINDOWS``. Used only to decide whether a directory is complete; this script
#: never passes ``--known-windows``, so a run that gets a different set of these (because the
#: corpus is too small for one of them) would look permanently incomplete -- which has not
#: happened on either corpus, and would be visible as a cell that re-runs every time.
KNOWN_CONFIGS = ("known0025", "known2550", "known5075", "known0050", "known2575", "known0075")

#: The two families, in the order a batch runs them. Names are what ``--families`` takes.
FAMILIES = ("attribution", "clustering")

#: :data:`SOURCES` in clustering's cost order -- cheapest corpus first, so an interrupted batch
#: has finished the quick cells. The attribution grid orders by attack instead and takes the
#: corpora in :data:`SOURCES` order within each; there the corpus is not the dominant cost.
CLUSTERING_SOURCE_ORDER = ("swe_chat", "wildchat")

#: Features the clustering grid uses when ``--features`` is not given. Every clustering result on
#: disk is on this one: the graph is built from cosine distances, where the 3,072-d semantic
#: vectors carry the same-author signal that makes the attack work at all. ``--features`` still
#: accepts the whole of :data:`FEATURES` and applies to both families at once.
CLUSTERING_DEFAULT_FEATURES = ("gemini_embedding_2",)

#: Algorithms every clustering cell runs, in the order ``run_clustering.py`` sorts them into.
#: Passed explicitly rather than left to the runner's default, so the plan shows what was asked
#: for and a later change to that default cannot silently re-shape a batch.
CLUSTERING_ALGORITHMS = ("average_linkage", "componentwise_agglomerative", "connected",
                         "hdbscan", "leiden")

#: Author scopes every clustering cell attacks the collection under, likewise passed explicitly.
#: Each is a complete run with its own search and baselines, filed in the same directory under a
#: ``scope`` column; their BCubed scores are **not** comparable with each other.
CLUSTERING_SCOPES = ("all", "unseen")

#: Where a clustering algorithm cannot run, so that its absence is not read as an unfinished
#: cell. Mirrors the runner's own guard rather than importing it, for the same reason the grid
#: above is literal: ``average_linkage`` needs a dense ``n_documents^2`` float64 distance matrix,
#: which is 14.9 GB at WildChat's 43,127-document test quarter and over the 20,000-document
#: ``MAX_DENSE_DOCUMENTS`` limit, so the runner skips it there with a printed note.
#: ``componentwise_agglomerative`` is the method that gets average linkage back at that scale by
#: agglomerating inside one connected component at a time, and has no such limit.
CLUSTERING_SCALE_LIMITED = {"average_linkage": ("wildchat",)}

#: The clustering configurations every cell is run under: name -> (``--projection``, whether the
#: elapsed-time weight is searched). The name is the fourth part of the directory,
#: ``<dataset>_<defense>_<feature>_<variant>``, and must be one ``run_clustering.variant_name``
#: can produce and ``plot_results.CLUSTERING_VARIANT_LABELS`` registers.
#:
#: Both are in the default grid, and they are different experiments rather than one superseding
#: the other:
#:
#: * ``plain`` is the pure-text run, the comparable set every ``by_defense`` and
#:   ``precision_recall`` figure draws. Dropping it would empty those figures.
#: * ``time`` fuses elapsed time into the edge score and reads no learned projection. It is the
#:   control that makes the row below readable -- timestamps alone re-link users nearly as well
#:   as a tuned style attack (WildChat test BCubed F 0.463 for timing only against 0.510 for the
#:   baseline), so a fused number has to be shown against it to claim the gain is joint. CPU only,
#:   since there is no projection to fit.
#: * ``contrastive_time`` fits a contrastive projection on the first half of the timeline and
#:   fuses elapsed time into the edge score, which is the strongest configuration measured: on
#:   WildChat's test slice, BCubed F 0.510 baseline -> 0.537 contrastive -> **0.562** with timing.
#:   It goes to the variants family instead, where the strategy is the axis.
#:
#: **Every variant searches its distance thresholds as quantiles of the graph's own edge weights**
#: -- ``run_clustering.py`` has no absolute-radius mode any more (see ``search_spaces``). So the
#: three differ only in what is fused into the edge score, and ``time`` at a searched weight of 0
#: really is ``plain``. What that buys is that one threshold means one thing across a projection,
#: a fusion, and the undefended-tuning-slice/defended-test-collection gap; what it costs is that a
#: defense's effect on the distance *scale* no longer registers, only its effect on ranking.
#:
#: ``lda``, ``wccn`` and the rescorings are **registered variants, just not gridded here**:
#: ``run_clustering.py`` produces them (``--projection lda``, ``--projection wccn --rescoring
#: csls``), ``plot_results.CLUSTERING_VARIANT_LABELS`` names them, and their directories parse.
#: They are out of the default grid because the contrastive projection beat every algorithm and
#: hubness fix in the development sweep, so the default runs the pure-text baseline and the
#: winner. Adding one back is a single line here, e.g. ``"lda": ("lda", False)``.
CLUSTERING_VARIANTS = {
    "plain": ("none", False),
    "time": ("none", True),
    "contrastive_time": ("contrastive", True),
}

#: Prefix of the reference partitions ``run_clustering.py`` scores beside the algorithms
#: (``baseline_singleton`` and friends). They are rows in ``clustering_results.csv`` but not
#: attacks, have no ``clusters_*.csv``, and are not what makes a cell finished.
BASELINE_PREFIX = "baseline_"


# --- resource classes --------------------------------------------------------
#
# What a cell needs from a machine, named rather than spelled out: these are the profile names
# `scripts/slurm.toml` maps to sbatch flags. Keeping the *rule* here and the *flags* there is what
# lets another cluster be adopted by editing one TOML -- the rule is a property of the experiment
# and travels with it, the flags are not.

#: Attacks worth allocating a GPU for. ``xgboost`` measured 17x (22.1 s CPU against 1.29 s on one
#: A100 at 1,000 documents x 3,072 features over 81 authors); ``logistic_sgd`` is ~2 TFLOP per
#: pass of pure matmul and is the one attack here that is *only* practical on a device -- its
#: WildChat StyloMetrix grid is 20 minutes on one A16 against a projected ~9 h on eight CPU cores.
#: Every other attack is BLAS on the CPU and would leave a card idle for the whole job.
GPU_ATTACKS = ("xgboost", "logistic_sgd")

#: Sources whose score matrix does not fit a default allocation. WildChat's is 86,255 x 13,694
#: float32 = 4.72 GB at ``known0050`` alone; runs have been OOM-killed at 16 GB.
LARGE_MEMORY_SOURCES = ("wildchat",)

#: Sources whose *collection* does not fit the small clustering profile -- a different resource
#: story from the one above, which is why the clustering classes are their own. WildChat clusters
#: 43,127 documents: two neighbour graphs, HDBSCAN's minimum spanning tree over ~1.5M edges, and
#: componentwise agglomeration's dense per-component matrix, on top of the 172,509 x 3,072 feature
#: matrix held while loading (2.1 GB). See the notes in ``scripts/slurm.toml``.
LARGE_COLLECTION_SOURCES = ("wildchat",)

#: Projections worth allocating a GPU for. The contrastive fit is 30,000 steps of a 3,072 x 1,024
#: matmul over a 2,048-document batch -- ~6 minutes on one A100 against hours on eight cores. The
#: closed-form projections (``wccn``, ``lda``) and every clustering algorithm are CPU work, so a
#: variant that does not fit a contrastive map stays on the CPU classes.
GPU_PROJECTIONS = ("contrastive",)


@dataclass(frozen=True)
class Cell:
    """One point of the grid: one runner invocation, one results directory.

    Both families share this shape; ``attack`` and ``variant`` are what tell them apart, each
    ``None`` for the family that has no such axis. The alternative -- two dataclasses -- would
    have forked ``Plan``, ``print_plan``, the whole SLURM path and ``main`` along with it, to
    express two absent fields.
    """

    family: str
    source: str
    defense: str
    feature: str
    attack: str | None = None
    variant: str | None = None

    @property
    def tag(self) -> str:
        """The results directory name, for a default run of this cell's family.

        ``run_experiment.output_tag`` (four parts) or ``run_clustering.main``'s tag (three), both
        positional and both spelling no defense ``base``. The dataset part is the source name
        verbatim, which is also its parquet's base name -- one spelling per corpus. Every flag
        this script passes is a default as far as those two are concerned, so the name stays in
        the comparable set that ``plot_results.py`` draws.
        """
        name = f"{self.source}_{self.defense}_{self.feature}"
        return f"{name}_{self.attack if self.attack is not None else self.variant}"

    @property
    def variant_flags(self) -> list[str]:
        """The ``run_clustering.py`` flags this cell's variant adds; empty for ``plain``.

        The *name* is the contract, not these flags: ``run_clustering.variant_name`` derives the
        fourth part of the directory from exactly this combination, and this script names the job,
        checks whether the cell is finished and reports it under that string.
        """
        if self.variant is None:
            return []
        projection, tune_weight = CLUSTERING_VARIANTS[self.variant]
        return (([] if projection == "none" else ["--projection", projection])
                + (["--tune-time-weight"] if tune_weight else []))

    @property
    def results_root(self) -> Path:
        """The root this cell's directory lives under -- one per family."""
        return RESULTS_DIR if self.family == "attribution" else CLUSTERING_DIR

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


def build_grid(families: tuple[str, ...], sources: tuple[str, ...], defenses: tuple[str, ...],
               features: tuple[str, ...], clustering_features: tuple[str, ...],
               attacks: tuple[str, ...], variants: tuple[str, ...]) -> list[Cell]:
    """Every selected cell, in execution order: families in :data:`FAMILIES` order, cheapest first.

    Attribution sorts by attack rather than by source, so an interrupted batch has finished all
    the nearest-neighbour runs -- the ones that are cheap to redo -- rather than a random prefix;
    within an attack the order is source, defense, feature, so the plan reads in blocks.
    Clustering has no attack axis but two others: its cost order is the variant (the plain run
    before the one that has to fit a projection first) and then the corpus, since swe-chat's test
    quarter is ~1,000 documents against WildChat's 43,127 and every stage of it is superlinear in
    that.
    """
    cells: list[Cell] = []
    for family in families:
        if family == "attribution":
            cells += [
                Cell("attribution", source, defense, feature, attack)
                for attack in attacks
                for source in SOURCES
                for defense in defenses
                for feature in features
                if source in sources and attack in SOURCE_ATTACKS.get(source, ATTACKS)
            ]
        else:
            cells += [
                Cell("clustering", source, defense, feature, variant=variant)
                for variant in variants
                for source in CLUSTERING_SOURCE_ORDER
                for defense in defenses
                for feature in clustering_features
                if source in sources
            ]
    return cells


def resource_profile(cell: Cell) -> str:
    """The ``scripts/slurm.toml`` profile this cell should be submitted under.

    Deliberately coarse -- four classes over two independent axes, not a per-cell resource table.
    A cell that genuinely needs something its class does not give it belongs in the TOML's
    ``[overrides.<tag>]``, which keeps the exception next to the numbers it is an exception to.

    The two axes are orthogonal and both matter, which is why there are four and not three.
    Needing a *device* is a property of the attack; needing *host memory and wall time* is a
    property of the corpus, because what is large is the ``[n_unknown x n_authors]`` score matrix
    and that lives on the host whatever fitted it. ``wildchat`` + ``logistic_sgd`` needs both at
    once, and folding it into plain ``gpu`` -- sized for xgboost on swe-chat at 24 GB and a
    four-hour ``short`` QOS -- would hand a 3.4 GB score matrix and a multi-hour Gemini fit an
    allocation that fits neither.

    Clustering has its own two classes over one axis, the corpus, and does not borrow ``cpu`` /
    ``cpu_large``. There is no attack to want an accelerator -- the neighbour graph is a BLAS
    matmul through the project's blocked kernel and Leiden, HDBSCAN and connected components are
    CPU graph algorithms, so a card would sit idle for the whole job -- and what is large about a
    clustering cell is its graphs and its dense per-component matrices, not an
    ``[n_unknown x n_authors]`` score matrix. The two families want different amounts of the same
    resource, and each should be tunable in the TOML without moving the other.
    """
    if cell.family == "clustering":
        projection = CLUSTERING_VARIANTS[cell.variant][0] if cell.variant else "none"
        if projection in GPU_PROJECTIONS:
            return "clustering_gpu"
        return ("clustering_cpu_large" if cell.source in LARGE_COLLECTION_SOURCES
                else "clustering_cpu")
    if cell.attack in GPU_ATTACKS:
        return "gpu_large" if cell.source in LARGE_MEMORY_SOURCES else "gpu"
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


def scope_suffix(scope: str) -> str:
    """Filename suffix for one clustering author scope; empty for ``all``.

    Mirrors ``run_clustering.scope_suffix`` (and ``plot_results.clustering_scope_suffix``, which
    mirrors it for the same reason): the ``all`` scope is spelled by *absence*, so its files keep
    the names they had before ``--scopes`` existed. Restated rather than imported because
    importing the runner would pull in numpy, pandas, scipy and the attack package just to print
    a plan -- the same reason the grid vocabulary above is literal.
    """
    return "" if scope == "all" else f"_{scope}"


def expected_results(cell: Cell, scopes: tuple[str, ...],
                     algorithms: tuple[str, ...]) -> set[tuple[str, str]]:
    """The ``(scope, algorithm)`` pairs a finished clustering ``cell`` should hold.

    The batch's whole request, minus the pairs :data:`CLUSTERING_SCALE_LIMITED` says this corpus
    cannot produce. Without that subtraction every WildChat cell would be one result short
    forever and would be re-run by every launch -- the failure mode :data:`KNOWN_CONFIGS`
    warns about above, except that here it would actually happen.
    """
    return {(scope, algorithm)
            for scope in scopes
            for algorithm in algorithms
            if cell.source not in CLUSTERING_SCALE_LIMITED.get(algorithm, ())}


def completed_results(run_dir: Path) -> set[tuple[str, str]]:
    """The ``(scope, algorithm)`` pairs a clustering ``run_dir`` holds a finished result for.

    The clustering half of :func:`completed_configs`, and it counts a pair only on the same
    two-halves rule: a row in ``clustering_results.csv`` and its own
    ``clusters_<algorithm>[_unseen].csv``. Checking one alone would call a half-written directory
    done -- and the per-document file is what a downstream re-score and every clustering figure
    read, so a missing one is a cell that cannot be used however complete its summary looks.

    A ``clustering_results.csv`` written before the ``scope`` column existed (2026-08-16) is
    entirely the ``all`` scope, which is what the ``or "all"`` below records -- correctly leaving
    such a directory partial, since it really does hold nothing for ``unseen``.
    """
    results = run_dir / "clustering_results.csv"
    if not results.exists():
        return set()
    try:
        with results.open(newline="", encoding="utf-8") as handle:
            scored = {((row.get("scope") or "all"), row["algorithm"])
                      for row in csv.DictReader(handle)}
    except (OSError, KeyError):        # unreadable or written by an older, different schema
        return set()
    return {(scope, algorithm) for scope, algorithm in scored
            if not algorithm.startswith(BASELINE_PREFIX)
            and (run_dir / f"clusters_{algorithm}{scope_suffix(scope)}.csv").exists()}


def describe_missing(missing: set[tuple[str, str]]) -> str:
    """The missing ``(scope, algorithm)`` pairs of a clustering cell, as one line for the plan."""
    return ", ".join(f"{algorithm}[{scope}]" for scope, algorithm in sorted(missing))


def describe_missing_configs(missing: set[str]) -> str:
    """The missing known configurations of an attribution cell, in :data:`KNOWN_CONFIGS` order."""
    return ", ".join(config for config in KNOWN_CONFIGS if config in missing)


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
    """What this script decided to do about one cell, and why.

    Family-agnostic: nothing here or in :func:`print_plan` reads anything off the cell but its
    ``tag``, which is what lets one plan hold both families' cells at once.
    """

    cell: Cell
    action: str                        # "run", "done", "queued", or "no-data"
    reason: str = ""

    @property
    def will_run(self) -> bool:
        return self.action == "run"


def plan_cell(cell: Cell, args: argparse.Namespace,
              queued: frozenset[str] = frozenset()) -> Plan:
    """Decide whether ``cell`` runs, and record the reason it does not.

    Data availability is checked first and is not overridable: ``--force`` re-runs work, it
    cannot conjure vectors that were never computed. The queue check comes next and is *also* not
    overridable -- two jobs writing one results directory interleave their ``predictions_*.csv``
    (or their ``clusters_*.csv``) and leave it neither run's output, so a resubmission means
    cancelling the job first. Only the last step, "is it already finished", differs between the
    families, because only there do they write different files.
    """
    data_dir = args.data_dir
    documents = data_dir / f"{cell.source}.parquet"
    if not documents.exists():
        return Plan(cell, "no-data", f"{documents.name} missing -- build the dataset first")

    if cell.tag in queued:
        return Plan(cell, "queued", "a job of this name is already pending or running "
                                    "(scancel it to resubmit)")

    if cell.feature in KNOWN_SIDE_FEATURES:
        defended = cell.defended_parquet(data_dir)
        if cell.family != "attribution":
            return Plan(cell, "no-data", f"{cell.feature} is fitted inside run_experiment.py and "
                                         f"has no feature parquet for the clustering runner")
        if defended is not None and not defended.exists():
            return Plan(cell, "no-data", f"{defended.name} missing -- run `apply_defenses "
                                         f"--source {cell.source} --defense {cell.defense}` first")
    elif not (features := cell.feature_parquet(data_dir)).exists():
        defended = cell.defended_parquet(data_dir)
        if defended is not None and not defended.exists():
            reason = (f"{features.name} missing; so is {defended.name} -- run "
                      f"`apply_defenses --source {cell.source} --defense {cell.defense}` first")
        else:
            reason = (f"{features.name} missing -- run `compute_features --source {cell.source} "
                      f"--feature {cell.feature}"
                      + (f" --defense {cell.defense}" if defended is not None else "") + "`")
        return Plan(cell, "no-data", reason)

    if cell.family == "attribution":
        run_dir = args.results_dir / cell.tag
        expected: set = set(KNOWN_CONFIGS)
        done: set = completed_configs(run_dir, cell.attack)
        describe = describe_missing_configs
        unit = "known configs"
    else:
        run_dir = args.clustering_dir / cell.tag
        expected = expected_results(cell, tuple(args.scopes), tuple(args.algorithms))
        done = completed_results(run_dir) & expected
        describe = describe_missing
        unit = "(scope, algorithm) results"

    if args.force:
        return Plan(cell, "run", "forced" if done else "")
    if expected <= done:
        return Plan(cell, "done", f"{len(expected)} {unit} in {run_dir.name}")
    if done:
        return Plan(cell, "run", f"partial: {len(done)}/{len(expected)} present, "
                                 f"missing {describe(expected - done)}")
    return Plan(cell, "run", "")


# --- running -----------------------------------------------------------------

def runner_command(cell: Cell, args: argparse.Namespace) -> list[str]:
    """The runner command line for one cell -- which runner depends on the family.

    Only the grid axes, the settings that decide *where* the work happens, and (for clustering)
    the batch-wide experiment configuration are passed; everything else is left at the runner's
    default, which is what keeps the output directory name to its three or four parts and the run
    inside the comparable set. ``--extra`` is appended last so it can override anything here,
    with the caveat in the module docstring about flags that change the directory name.

    ``--xgboost-device`` is the one value this script sets away from a runner default (to
    ``auto``, see :func:`parse_args`); it is not part of ``output_tag``, so it changes where the
    trees are fitted and nothing else about how the run is filed.

    The two runners spell an undefended cell differently and each is given what it expects:
    ``run_experiment.py`` takes ``--defense none`` and is simply not passed the flag, while
    ``run_clustering.py`` takes the directory spelling ``base`` as the value of ``--defense``.
    """
    if cell.family == "clustering":
        command = [sys.executable, str(CLUSTERING_RUNNER),
                   "--source", cell.source,
                   "--defense", cell.defense,
                   "--feature", cell.feature,
                   "--data-dir", str(args.data_dir),
                   "--algorithms", *args.algorithms,
                   "--scopes", *args.scopes,
                   *cell.variant_flags]
        if args.oracle_sweep:
            command.append("--oracle-sweep")
        return command + args.extra

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
    applied to every job, ``profiles`` keyed by a resource-class name (this launcher's are
    :func:`resource_profile`'s return values), and ``overrides`` keyed by a cell tag. There is no
    schema of our own beyond that, on purpose: a flag this launcher has never heard of still
    works, because it is passed straight through.

    The two methods take a profile name and a tag rather than a :class:`Cell`, because a cell's
    resource class is a property of its family and this table is not: keeping the lookup here and
    the rule in :func:`resource_profile` is what lets the clustering classes be added without
    touching either.
    """

    path: Path
    defaults: tuple[str, ...]
    profiles: dict[str, tuple[str, ...]]
    overrides: dict[str, tuple[str, ...]]

    def flags_for(self, profile: str, tag: str) -> list[str]:
        """The ``sbatch`` flags for one cell, in precedence order (later wins)."""
        return [*self.defaults,
                *self.profiles[profile],
                *self.overrides.get(tag, ())]

    def check(self, wanted: set[str]) -> None:
        """Fail before anything is submitted if a resource class is not in the file.

        Checked for the whole batch up front rather than at each submission, so a missing profile
        is a message instead of half a grid queued and the rest abandoned.
        """
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
            *config.flags_for(resource_profile(cell), cell.tag),
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
    parser.add_argument("--families", nargs="+", metavar="NAME",
                        help=f"Attack families to run (default: all of {', '.join(FAMILIES)}). "
                             "'attribution' drives run_experiment.py over the four-part grid, "
                             "'clustering' drives run_clustering.py over the three-part one; the "
                             "--sources/--defenses/--features filters apply to both, --attacks "
                             "only to the first.")
    parser.add_argument("--sources", nargs="+", metavar="NAME",
                        help=f"Restrict to these datasets (default: all of {', '.join(SOURCES)}).")
    parser.add_argument("--defenses", nargs="+", metavar="NAME",
                        help=f"Restrict to these defenses, '{NO_DEFENSE}' for undefended "
                             f"(default: all of {', '.join(DEFENSES)}).")
    parser.add_argument("--features", nargs="+", metavar="NAME",
                        help=f"Restrict to these features, any of {', '.join(FEATURES)} (default: "
                             f"{', '.join(ATTRIBUTION_DEFAULT_FEATURES)} for attribution, but only "
                             f"{', '.join(CLUSTERING_DEFAULT_FEATURES)} for clustering, which is "
                             "the one every clustering result on disk uses). Given explicitly, "
                             "it applies to both families.")
    parser.add_argument("--attacks", nargs="+", metavar="NAME",
                        help=f"Restrict to these attacks (default: all of {', '.join(ATTACKS)}, "
                             f"subject to the per-source restrictions).")
    parser.add_argument("--clustering-variants", nargs="+", metavar="NAME",
                        help="Clustering only: which configurations each cell is run under "
                             f"(default: all of {', '.join(CLUSTERING_VARIANTS)}). 'plain' is the "
                             "pure-text run that keeps the three-part directory name every "
                             "by_defense figure draws; 'contrastive_time' fits a contrastive "
                             "projection on the first half of the timeline and fuses elapsed "
                             "time into the edge score, which is the strongest configuration "
                             "measured (WildChat test BCubed F 0.510 -> 0.562) and lands in the "
                             "variants family rather than the comparable set. Its time weight is "
                             "SEARCHED, not set: run_clustering.py tunes it per algorithm on the "
                             "tuning slice, which is why the directory says `time` and carries no "
                             "number.")
    parser.add_argument("--algorithms", nargs="+", metavar="NAME",
                        default=list(CLUSTERING_ALGORITHMS),
                        help="Clustering only: algorithms every cell runs (default: "
                             f"{', '.join(CLUSTERING_ALGORITHMS)}). One list for the whole batch, "
                             "so the corpora stay comparable; average_linkage is skipped by the "
                             "runner above 20,000 documents, which is expected on WildChat and is "
                             "accounted for when deciding whether a cell is finished.")
    parser.add_argument("--scopes", nargs="+", metavar="NAME",
                        default=list(CLUSTERING_SCOPES), choices=list(CLUSTERING_SCOPES),
                        help="Clustering only: author scopes every cell attacks (default: "
                             f"{', '.join(CLUSTERING_SCOPES)}). 'all' is the test quarter whole, "
                             "'unseen' only the authors absent from the known side; each is a "
                             "complete run into the same directory, and their scores are not "
                             "comparable with each other.")
    parser.add_argument("--oracle-sweep", action=argparse.BooleanOptionalAction, default=False,
                        help="Clustering only: also score the whole hyper-parameter grid on the "
                             "test collection, to bound what tuning could have achieved (writes "
                             "oracle_sweep.csv). It is an ORACLE -- never select on it. Off by "
                             "default, matching the runner: it roughly doubles the search cost, "
                             "and the swe-chat runs on disk were produced without it while the "
                             "WildChat ones had it, which is one of the inconsistencies this "
                             "launcher exists to end.")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR,
                        help="Directory holding the parquets, passed straight to the runner "
                             "(default: data/hf, NOT data/dist -- see the module docstring).")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR,
                        help="Where attribution results live; what is scanned to decide whether "
                             "a cell is already done (default: experiments/results).")
    parser.add_argument("--clustering-dir", type=Path, default=CLUSTERING_DIR,
                        help="The same for clustering results (default: experiments/clustering). "
                             "A separate root because both directory names are contracts "
                             "plot_results.py parses with different functions.")
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
                             "line, e.g. `-- --no-tune`. It reaches BOTH families, so narrow with "
                             "--families when a flag only one of them takes. Do NOT pass a flag "
                             "that changes the output directory name; see the module docstring.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    families = resolve_choices(args.families, FAMILIES, "--families")
    sources = resolve_choices(args.sources, SOURCES, "--sources")
    defenses = resolve_choices(args.defenses, DEFENSES, "--defenses")
    features = resolve_choices(args.features or list(ATTRIBUTION_DEFAULT_FEATURES), FEATURES,
                               "--features")
    attacks = resolve_choices(args.attacks, ATTACKS, "--attacks")
    # An explicit --features applies to both families; left out, clustering takes only the one
    # feature its results are all on rather than the whole attribution default.
    clustering_features = (features if args.features
                           else resolve_choices(list(CLUSTERING_DEFAULT_FEATURES), FEATURES,
                                                "--features"))
    # Canonical order rather than the order they were typed, so the plan and the command lines
    # read the same whatever the caller wrote.
    args.algorithms = list(resolve_choices(args.algorithms, CLUSTERING_ALGORITHMS, "--algorithms"))
    args.scopes = [scope for scope in CLUSTERING_SCOPES if scope in args.scopes]
    variants = resolve_choices(args.clustering_variants, tuple(CLUSTERING_VARIANTS),
                               "--clustering-variants")

    grid = build_grid(families, sources, defenses, features, clustering_features, attacks,
                      variants)
    if not grid:
        raise SystemExit("no cells selected: the --families/--sources/--defenses/--features/"
                         "--attacks/--clustering-variants filters do not intersect (remember "
                         "WildChat is nearest_neighbor and logistic_sgd only).")

    if args.stop_on_error and args.slurm:
        raise SystemExit("--stop-on-error is meaningless with --slurm: submission returns before "
                         "any cell has run, and the jobs are independent by design.")

    # Read and validate the resource config before touching the queue, so a typo in the TOML costs
    # a message rather than a partly-submitted grid.
    config = load_slurm_config(args.slurm_config) if args.slurm else None
    if config is not None:
        config.check({resource_profile(cell) for cell in grid})
        if not SLURM_WRAPPER.exists():
            raise SystemExit(f"{SLURM_WRAPPER} not found: --slurm submits every job through it.")

    queued = queued_job_names() if args.slurm else frozenset()
    plans = [plan_cell(cell, args, queued) for cell in grid]
    header = (f"grid: {len(grid)} cells over {', '.join(families)}, data from {args.data_dir}"
              + (f", resources from {config.path}" if config is not None else ""))
    if "clustering" in families:
        header += (f"\nclustering configuration: {', '.join(variants)}"
                   f" | --algorithms {' '.join(args.algorithms)}"
                   f" --scopes {' '.join(args.scopes)}"
                   + (" --oracle-sweep" if args.oracle_sweep else ""))
    print_plan(plans, header)

    queue = [plan.cell for plan in plans if plan.will_run]

    if args.dry_run:
        for cell in queue:
            if config is not None:
                print(f"\n{cell.tag}  [{resource_profile(cell)}]")
                print("  " + shlex.join(sbatch_command(cell, args, config)))
            else:
                print(f"\n{cell.tag}")
                print("  " + shlex.join(runner_command(cell, args)))
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
