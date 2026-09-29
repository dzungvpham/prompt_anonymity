#!/usr/bin/env python
"""Render every figure this project publishes, from the result CSVs already on disk.

Run it with no arguments::

    python experiments/plot_results.py

Nothing here recomputes an attack: the runners (``run_experiment.py`` and
``run_experiment.py``) write CSVs and stop, and this script turns the accumulated CSVs into
figures. Re-running it is cheap and idempotent, so it is the right thing to run after any new
experiment finishes.

Input is ``experiments/results/<dataset>_<defense>_<feature>_<attack>/``. That four-part name is
the contract -- it is what lets one run be compared against another without opening it -- and each
part must be a name this file knows (:data:`DATASETS`, :data:`DEFENSES`, :data:`FEATURES`,
:data:`ATTACKS`). An undefended run is spelled ``base``, not omitted. Any other directory is
skipped with a note, which is how exploratory output (``tuned_comparison/`` and friends) stays out
of the figures.

A **second** input is ``experiments/clustering/<dataset>_<defense>_<feature>/`` -- the author
*clustering* experiment, whose three-part names have no attack because a clustering attack has no
such axis. It is a different question about the same corpora (does an anonymised log fall apart
into its authors on its own, with nobody named?) rather than a different view of the one above,
so it shares this file's palette, chrome and cache but none of its machinery. See the
``clustering/`` family below.

Output is ``experiments/plots/<dataset>/``, in two families:

``openset/``
    The attribution figures, and the only ones drawn from :data:`RESULTS_DIR`. Each is filed
    ``openset/<figure>/<doc|author>/by_{defense,attack}/``, and **the middle level is what is
    being counted**:

    ``doc/``
        The unit is a **document**: what share of the traffic can be attributed.
    ``author/``
        The unit is a **person**, and they count once **any one** of their documents does. This
        is the right reading when being linked at all is the harm, and it runs well above the
        per-document number. It is not a re-run: the author level is the same predictions
        collapsed to one row per person by :func:`author_table`, taking the best each of their
        documents achieved.

    The innermost level is the two comparison views: ``by_defense/<feature>_<attack>.pdf`` (attack
    fixed, a line per defense -- *does the defense work?*) and ``by_attack/<defense>.pdf``
    (defense fixed, a line per feature+attack -- *which attack is strongest?*). The three figures:

    ``openset/dirfar/``
        **DIR against FAR** -- the open-set identification ("watchlist ROC") curve: the share of
        in-set documents both accepted and ranked top-1, against the share of out-of-set
        documents the same threshold wrongly accepts. k = 1, threshold swept.
    ``openset/dirfar_by_words/``
        DIR at a fixed FAR budget, split by the target conversation's word count.
    ``openset/top_k_cmc/``
        The CMC curve with the rejection filter applied: top-k accuracy against k at a threshold
        pinned to a fixed FAR budget. The orthogonal slice of the same DIR(threshold, k) surface.

    The whole family reads the documents a closed-set figure would drop -- the unknown documents
    whose author is absent from the known side. It rests on two columns those rows do carry:
    ``author_in_known`` (the ground truth) and ``accept_score`` (the cohort-normalised margin,
    which ``run_experiment.py`` writes for every unknown document whether or not ``--ood reject``
    was on). **Every operating point it shows is an oracle one**: these runs were ``--ood none``,
    so the threshold the runner would have calibrated is not recoverable without re-running. The
    curves are threshold-free and unaffected; a point read off one is what a perfect calibrator
    could reach, not what the runner's achieved.
``clustering/``
    The **other experiment** (merged in from ``plot_clustering.py``), and the one
    family that reads :data:`CLUSTERING_DIR` instead of :data:`RESULTS_DIR`. It reports a
    *partition* rather than a ranking, so it has no known-side grid and none of the levels
    above: ``clustering/bcubed/by_defense/<feature>.pdf`` is BCubed F per algorithm with every
    reference partition drawn as a grey bar beside them -- the comparison the figure exists for,
    since an all-singleton partition scores F = 0.285 on WildChat *while linking nothing*; and
    ``clustering/precision_recall/by_algorithm/<feature>.pdf`` puts each algorithm at one point in
    BCubed precision x recall, where a method that buys precision by refusing to cluster is
    visibly doing so.

Every figure is written as a PDF, and as a PNG beside it under ``--png``. Both halves of the work
-- building the curves and drawing the figures -- run across ``--jobs`` processes; see
:func:`run_jobs` for why the job table is a module global and the start method is pinned to fork.

``--families`` draws one or more of these families and nothing else, named after the folders above
(:data:`FIGURE_FAMILIES`). It narrows only what is *drawn*: every other figure keeps its cache
entry, so ``--families clustering`` is a slice of a full sweep rather than a sweep of its own, and
the families it skipped are still current afterwards. That is what it is for -- editing this file
invalidates the whole tree at once (the cache keys on the file's digest), and redrawing every
figure to look at one family is most of a sweep spent on figures nobody is reading.

Every comparison figure is a **3x3 triangular facet grid, one panel per known configuration**,
rows = known-side size, columns = staleness. **Nothing is averaged across the grid**: the
configurations are experimental conditions, not repeated measurements, so a mean over them would
describe no experiment that was run and would move with which cells happened to be included.
Within a panel the series are truncated to the shortest one's k range so every line spans the
same axis; across panels the pools differ by design -- they are what each known side enrolls --
which is why every panel prints its own in-set counts and the grid is not collapsed onto one
axes. The shaded band is a 95 % percentile interval from a clustered bootstrap over *users*, one
draw shared by every configuration and run of a dataset. Every figure is written alongside a
``.csv`` of the exact numbers plotted.

**No figure carries a title, and none has since the subtitle was also removed.**
Both were caption text, and caption text belongs to the document that publishes the figure rather
than to the image. What each figure *is*, is its path -- which is why the path scheme above is a
contract and not a filing convenience -- while its panel headings name the configuration and its
panel notes carry the counts. The prose is still written down: :data:`CURVE_TYPES`' fifth field
per family, and the note above :func:`finish_facets` for the two facts a title used to be the
only carrier of.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: render to files without a display
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402  (proxy handles for the two-legend figures)
from matplotlib.ticker import (FixedLocator,  # noqa: E402  (the FAR axis' 5 half-decade ticks)
                               FuncFormatter,  # noqa: E402  (plain log ticks)
                               LogLocator,  # noqa: E402  (every second decade)
                               MultipleLocator)  # noqa: E402  (quarter ticks on a 0-1 axis)

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "experiments" / "results"

#: The clustering experiment's results, deliberately *not* under :data:`RESULTS_DIR`: its
#: directories are three-part names and its ``clustering_results.csv`` schema is a partition rather
#: than a ranking, so a run of one kind must never parse as a run of the other.
CLUSTERING_DIR = REPO_ROOT / "experiments" / "clustering"
PLOTS_DIR = REPO_ROOT / "experiments" / "plots"


# --- running independent work across cores -----------------------------------
#
# Both halves of this script are embarrassingly parallel -- a run's curves do not depend on any
# other run's, and a figure does not depend on any other figure -- so both are dispatched through
# the one helper below.
#
# THE JOB TABLE IS A MODULE GLOBAL AND THAT IS THE POINT. These jobs carry expensive things (a
# comparison figure's arguments hold every curve it draws; a curve job holds a whole predictions
# table), so pickling them down a pipe would cost more than the work. Instead the jobs are parked
# in `_JOBS` *before* the pool forks, the children inherit them copy-on-write, and only an integer
# each way plus the small result crosses the pipe. That is also why the start method is pinned to
# "fork" rather than left to the platform default: under "spawn" the child re-imports this module
# with an empty `_JOBS`. Linux-only by construction, which this cluster is.

#: Jobs the current pool is executing, as ``(callable, args, kwargs)``. Set by :func:`run_jobs`
#: immediately before the pool is created, read by the forked children, meaningless otherwise.
_JOBS: list[tuple] = []


def _execute_job(index: int):
    """Run one parked job in a worker. Takes an index so nothing large is pickled inbound."""
    function, args, kwargs = _JOBS[index]
    return function(*args, **kwargs)


def run_jobs(jobs: list[tuple], workers: int) -> list:
    """Run ``(callable, args, kwargs)`` jobs across processes, results in submission order.

    Falls back to running them in this process when there is one worker or one job, which is what
    ``--jobs 1`` is for: a traceback from a forked child loses the frames above the fork, so
    debugging a drawing routine is far easier serially.
    """
    if workers <= 1 or len(jobs) <= 1:
        return [function(*args, **kwargs) for function, args, kwargs in jobs]

    global _JOBS
    _JOBS = jobs
    try:
        # Created after `_JOBS` is populated: the workers are forked on first submit, so this
        # ordering is what puts the jobs in their memory.
        with ProcessPoolExecutor(max_workers=min(workers, len(jobs)),
                                 mp_context=multiprocessing.get_context("fork")) as pool:
            return list(pool.map(_execute_job, range(len(jobs))))
    finally:
        _JOBS = []


def default_workers() -> int:
    """Cores to use unless ``--jobs`` says otherwise.

    Capped rather than "every core": each worker holds its own ``(replicates x documents)`` weight
    matrix on top of the tables it inherited, and this cluster's jobs run under a memory cap that
    has killed runs before.
    """
    return max(1, min(8, len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity")
                      else (os.cpu_count() or 1)))

#: Where the built dataset lives -- the same default ``run_experiment.py`` attacks. Read for
#: the proportional baseline's prior and the word-count split: corpus facts that never depended on
#: the attack, so they are joined back on ``doc_id`` rather than copied into every run's
#: predictions. A dataset's parquet is ``<dataset>.parquet``, because the dataset part of a
#: results directory name *is* the split name (see :data:`DATASETS`).
DATA_DIR = REPO_ROOT / "data" / "hf"


# --- the vocabulary a results directory name is built from -------------------
#
# These are the only accepted values for each part of `<dataset>_<defense>_<feature>_<attack>`.
# Keeping them as literals (rather than importing the package registries) keeps this script free
# of the heavy imports those registries pull in, at the cost of having to be kept in sync:
# DEFENSES mirrors `prompt_anonymity.defenses.DEFENSES` (with "none" spelled "base"), FEATURES
# mirrors `prompt_anonymity.features.FEATURIZERS`, and ATTACKS mirrors
# `prompt_anonymity.attacks.ATTRIBUTION_ATTACKS`. A run whose name uses a value missing here is
# skipped, not guessed at -- so adding a new defense or attack means adding it here too.

#: Datasets, spelled the one way the whole project spells them: the same string as the runners'
#: ``--source``, the split, the corpus parquet's basename in :data:`DATA_DIR`, and the dataset
#: part of a results directory. The plots directory follows: ``plots/swe_chat/``. (Only the
#: ``source`` *column* inside the parquets differs, still ``swe-chat`` -- it is hashed into every
#: ``author_id``, so it is data rather than a name. Nothing here reads it.)
DATASETS = ("wildchat", "swe_chat", "wildchat_small", "wildchat_tiny")

#: How an undefended run spells its defense. Written out rather than omitted so that every
#: directory name has the same four parts and can be parsed positionally.
NO_DEFENSE = "base"

#: Defenses, in the order their colours are assigned (see :func:`series_style`) -- the ones that
#: carry the argument first, the DP-MLM epsilon sweep last so it cannot push them off the palette.
DEFENSES = (
    NO_DEFENSE,
    "styleremix",
    "openanonymity",
    "styleremix_openanon",
    "collision_seeding",
    "collision_seeding_k4",
    "collision_seeding_k24",
    "collision_seeding_full",
    "collision_seeding_indep",
    "frame_shift",
    "frame_shift_single",
    "frame_pad",
    "frame_pad_single",
    "dp_mlm",
    "dp_mlm_pii",
    "dp_mlm_var_a10",
    "dp_mlm_var_a25",
    "qwen_rewrite",
    # "rtt_argos" is deliberately absent: the round-trip translations are too poor to report, and a
    # name missing here makes plot_results SKIP that results directory rather than draw it. Add it
    # back to put the arm on the figures again.
    "example_normalization",
    # The measurement-loop defenses. Ahead of the epsilon sweeps for the reason given above -- they
    # carry an argument, so they should hold stable colours -- and `afr_stage1` sits beside `afr`
    # because the two are only ever read against each other.
    "afr",
    "afr_stage1",
    "afr_a00",          # the same run as `afr` (alpha 0.0); listed so its directory is not skipped
    "afr_a25",
    "afr_a50",
    "loo_unlink_b10",
    "loo_unlink_b20",
    "loo_unlink_b30",
    "loo_unlink_b50",
    "loo_unlink_b70",
    *(f"dp_mlm_eps{epsilon}" for epsilon in (10, 25, 50, 100, 250, 500, 1000)),
    "embad",
    "embad_summary",
    "embad_gemini",
)

FEATURES = (
    "gemini_embedding_2",
    "gemini_embedding_001",
    "function_words",
    "character_statistics",
    "char_ngram_tfidf",
    "style_distance",
    # The local embedder `afr` and `loo_unlink` optimize against. Plotting it next to
    # gemini_embedding_2 is the surrogate-overfit check: a large drop here and none there means the
    # defense learned the surrogate rather than the signal.
    "harrier",
    "harrier_imperative",
    "harrier_plain",
)

ATTACKS = (
    "nearest_neighbor",
    "cosine",
    "wccn",
    "lda",
    "plda",
    "logistic",
    "rlsc",
    "svm",
    "xgboost",
    # Appended rather than filed next to `logistic`, and it has to be: an attack's index in this
    # tuple is its colour slot (see METHOD_STRIDE), so inserting one mid-list recolours every
    # attack below it. New entries go on the end regardless of where they belong by meaning.
    "logistic_sgd",
)

#: Keyed by the directory spelling, valued by how the corpus is written in prose and on a figure
#: -- which is the hyphenated "SWE-chat", and stays that way; only the filename changed.
DATASET_LABELS = {"wildchat": "WildChat", "swe_chat": "SWE-chat",
                  "wildchat_small": "WildChat (small)",
                  # The 40-author cut. The author count is IN the label because it is the pool the
                  # attack chooses between, and every absolute number on a figure drawn from it is
                  # conditional on that pool size -- see build_subset --n-authors.
                  "wildchat_tiny": "WildChat (40 authors)"}
DEFENSE_LABELS = {
    NO_DEFENSE: "No defense",
    "styleremix": "StyleRemix",
    "openanonymity": "OpenAnonymity",
    "styleremix_openanon": "StyleRemix + OpenAnonymity",
    # K is in the label because it is the privacy knob the sweep varies: K profiles put N authors
    # into groups of N/K, so a figure comparing them is a figure about collision-group size.
    "collision_seeding": "Collision seeding (K=12)",
    "collision_seeding_k4": "Collision seeding (K=4)",
    "collision_seeding_k24": "Collision seeding (K=24)",
    "collision_seeding_full": "Collision seeding (always on)",
    "collision_seeding_indep": "Collision seeding (independent)",
    # The codebook size is in the label for the same reason K is above: it is the knob. 50 scenes
    # rotate per document; the single-frame arm collapses that to one, so a figure holding both is a
    # figure about dilution versus convergence.
    "frame_shift": "Frame shift (50 frames)",
    "frame_shift_single": "Frame shift (single frame)",
    # Frame pad adds a scene-flavoured turn and rewrites NOTHING, so a figure holding it beside
    # frame_shift splits that defense in two: dilution by added content, versus the rewrite.
    "frame_pad": "Frame pad (50 frames)",
    "frame_pad_single": "Frame pad (single frame)",
    "dp_mlm": "DP-MLM",
    "dp_mlm_pii": "DP-MLM (PII only)",
    "dp_mlm_var_a10": "DP-MLM ± (A=0.1)",
    "dp_mlm_var_a25": "DP-MLM ± (A=0.25)",
    "qwen_rewrite": "Qwen rewrite",
    "example_normalization": "Text normalization",
    # The residual-linkage target alpha is in the label because it is the knob the sweep varies:
    # alpha 0 aims all the way down to a median unrelated document, higher keeps some linkage.
    # "Stage 1 only" names the ABLATION rather than the defense -- same model, same abstraction
    # pass, zero probes -- because on a figure it is the line "AFR" has to beat to mean anything.
    "afr": "AFR (α=0)",
    "afr_a00": "AFR (α=0)",
    "afr_a25": "AFR (α=0.25)",
    "afr_a50": "AFR (α=0.5)",
    "afr_stage1": "AFR stage 1 only (no loop)",
    # The utility budget is loo_unlink's knob, as a percentage.
    **{f"loo_unlink_b{budget}": f"LOO-unlink (b={budget / 100:.1f})"
       for budget in (10, 20, 30, 50, 70)},
    **{f"dp_mlm_eps{epsilon}": f"DP-MLM ε={epsilon}"
       for epsilon in (10, 25, 50, 100, 250, 500, 1000)},
    "embad": "EmBad (local ensemble)",
    "embad_summary": "EmBad (summary)",
    "embad_gemini": "EmBad (Gemini)",
}
FEATURE_LABELS = {
    "gemini_embedding_2": "Gemini Embedding 2",
    "gemini_embedding_001": "Gemini Embedding 001",
    "function_words": "Function words",
    "character_statistics": "Character stats",
    "char_ngram_tfidf": "Character n-gram",
    "style_distance": "StyleDistance",
    "harrier": "Harrier 0.6B",
    "harrier_imperative": "Harrier 0.6B (imperative)",
    "harrier_plain": "Harrier 0.6B (plain)",
}
ATTACK_LABELS = {
    "nearest_neighbor": "Nearest neighbor",
    "cosine": "Cosine centroid",
    # "Nearest centroid" rather than "WCCN centroid": what it *is* to a
    # reader is the centroid matcher beside "Nearest neighbor", and the whitening is the
    # implementation. The registry name stays `wccn`, so every path and CSV is unchanged.
    "wccn": "Nearest centroid",
    "lda": "LDA centroid",
    "plda": "PLDA",
    # **The two logistic labels are named for their FIT, and `logistic_sgd` is the plain one.**
    # It is the attack that runs at WildChat's author counts and therefore the one on every
    # figure, so it takes the unqualified name; the exact lbfgs fit -- which cannot be run on
    # WildChat at all, see CLAUDE.md -- carries the qualifier instead. They must not share a
    # label: `handles.setdefault(item.label, ...)`
    # keys the legend by it, so two attacks spelled the same would collapse into one entry and
    # write one `series` name onto two different curves in the companion CSV. That only bites
    # under `--attacks all`, where both are drawn.
    "logistic": "Logistic (LBFGS)",
    "logistic_sgd": "Logistic",
    "rlsc": "RLSC",
    "svm": "SVM",
    "xgboost": "XGBoost",
}

#: Stride between one feature's block of attack slots and the next. **Frozen, and deliberately
#: not ``len(ATTACKS)``.** Since ``hue = slot % 8``, the stride's residue mod 8 is what decides
#: the whole assignment, so changing the stride silently recolours most feature-attack pairs on
#: every figure already drawn, breaking the rule that a colour follows the entity rather than its
#: position.
#:
#: 17 keeps residue 1, so every existing assignment is preserved exactly, and leaves room for 17
#: attacks before a block overflows into the next feature's. **Add new attacks to the END of**
#: :data:`ATTACKS`: inserting one mid-list shifts the index of everything after it, which moves
#: those hues just as surely. Any replacement must stay ``= 1 (mod 8)`` and ``>= len(ATTACKS)``.
METHOD_STRIDE = 17

#: Reading order for methods, and the colour slot each one owns: feature-major, so a figure's
#: legend runs feature by feature and two runs of the same feature sit next to each other.
#: ``start=1`` because a first feature was removed from the project: its
#: index 0 is left vacant so that every remaining feature keeps the colour slot it always had.
METHOD_SLOTS = {(feature, attack): feature_index * METHOD_STRIDE + attack_index
                for feature_index, feature in enumerate(FEATURES, start=1)
                for attack_index, attack in enumerate(ATTACKS)}

#: Colour slot per **attack alone**, for the ``by_attack`` figures: there colour carries the
#: attack and the dash carries the feature (:data:`FEATURE_DASHES`), which is what keeps a panel
#: readable now that the registry holds ten attacks -- one line per (feature, attack) pair put a
#: dozen hues in a cell and made the legend the tallest thing on the figure.
#:
#: It is :data:`ATTACKS`' own order, so an attack keeps one hue across every figure and corpus,
#: and the append-only rule that protects :data:`METHOD_SLOTS` protects this too: inserting an
#: attack mid-tuple recolours every attack below it.
ATTACK_SLOTS = {attack: index for index, attack in enumerate(ATTACKS)}

#: The attacks a sweep draws unless ``--attacks`` says otherwise. **A drawing default, not a
#: judgement about which attack is right**: every run on disk keeps its results and
#: ``--attacks all`` puts them all back. These three are the set that spans the interesting
#: axes -- the unsupervised baseline every corpus can run (``nearest_neighbor``), the supervised
#: fit that roughly doubles WildChat's headline (``logistic_sgd``), and the covariance-normalised
#: centroid (``wccn``) -- without putting ten lines in a cell.
DEFAULT_ATTACKS = ("nearest_neighbor", "logistic_sgd", "wccn")


# --- palette and chart style -------------------------------------------------
#
# Categorical hues in a fixed order, validated for colour-vision deficiency as a set (adjacent
# pairs, light surface). Never extend this by generating a ninth hue: past eight series a figure
# needs fewer lines, not a made-up colour (see `resolve_slots`).

#: The chart surface, and **pure white on request** rather than the ``dataviz`` skill's default
#: off-white ``#fcfcfb``. A deliberate override of that parameter, not a value to "correct" back:
#: these figures are printed into a white page, and an off-white panel on it reads as a grey box
#: rather than as the paper. It stays safe on the skill's contrast check -- the only one involving
#: the surface, since the other five (lightness band, chroma floor, CVD separation, normal-vision
#: floor, adjacent pairs) are properties of the palette alone.
#:
#: **It is also the marker ring and the bar-gap colour** (the skill's surface ring / spacer, which
#: exist to separate overlapping marks by showing the surface through them), so it must stay one
#: constant -- a ring in the old off-white on a white panel would draw a halo.
SURFACE = "#ffffff"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
TEXT_MUTED = "#8a8983"
GRID = "#e6e5e1"
AXIS = "#c9c8c2"

CATEGORICAL = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100",
               "#e87ba4", "#008300", "#4a3aa7", "#e34948")

#: The dash pattern reserved for the random-guessing baseline, on every figure. Measured series
#: are solid in their own colour, so a dash reads as "not a measurement".
BASELINE_DASH = (0, (4, 3))

#: The one exception: **on a ``by_attack`` figure the dash carries the feature** and colour
#: carries the attack (:data:`ATTACK_SLOTS`), so one attack's two representations sit in a panel
#: as one hue in two patterns rather than as two unrelated colours. Solid Gemini Embedding 2,
#: dash-dot Character n-gram, on request.
#:
#: **Every name in :data:`FEATURES` needs an entry**, because an unregistered feature would fall
#: back to solid and collide with Gemini's -- the same "add it to the vocabulary or it is drawn
#: wrong" rule the labels above follow. The two channels never both carry something: on a
#: ``by_defense`` figure no series dashes at all (colour is the defense, the feature and attack are
#: in the filename).
FEATURE_DASHES = {
    # Dash-dot with **tight** gaps: a plain long dash was tried first, and a wide hole next to a
    # thin line reads as a broken line rather than as a pattern. Two marks and two equal small gaps
    # keep the stroke continuous at a glance while staying obviously not-solid next to Gemini's.
    "gemini_embedding_2": (),
    "gemini_embedding_001": (1.5, 1.5),
    "function_words": (4, 2, 1.5, 2),
    "character_statistics": (3, 2),
    "char_ngram_tfidf": (7, 2, 1.5, 2),
    "style_distance": (5, 2, 1.5, 2, 1.5, 2),
    "harrier": (2, 1.5),
    "harrier_imperative": (6, 3, 2, 3),
    "harrier_plain": (1, 2),
}

#: Every text size on every figure, in points -- named constants rather than scattered literals,
#: so a global size nudge is four numbers instead of many call sites.
#:
#: **An annotation inside a panel is set at the tick size.** A direct label, a panel note, a value
#: on a bar cap and an iso-F level are all things a reader reads *off the plot*, so they should not
#: be smaller than the numbers on the axis they are read against. The ranking is otherwise
#: unchanged -- annotations and ticks at the bottom, then the legend, then axis labels, then a
#: panel heading.
FONT_TICK = 10.5         # axis tick labels
FONT_ANNOTATION = FONT_TICK   # anything written inside the axes; deliberately the same number
FONT_LEGEND = 10.0       # legend entries and legend titles
FONT_AXIS_LABEL = 11.5   # the x and y axis titles
FONT_PANEL_TITLE = 13.0  # a facet's column header, a clustering panel's defense

LINE_WIDTH = 2.0
MARKER_SIZE = 6.0  # >= 8px on the page once the 2px surface ring is added

#: **Every figure on the 3x3 configuration grid** is drawn at twice the file's base point sizes
#: and twice its stroke widths. These are the figures headed for a paper, two side by side at a
#: fraction of the page width; the clustering family keeps the base sizes, is read on
#: screen at full size and does not go through the grid.
#:
#: It is a scale spent along **one dispatch path** -- :func:`plot_config_comparison`, which every
#: grid figure goes through -- rather than a change to the shared `FONT_*` constants, which the
#: unenlarged clustering figures are still measured against. Every helper that sets a point size therefore takes a ``scale`` defaulting to
#: 1.0, and the grid is the only caller that passes anything else.
GRID_SCALE = 2.0
#: The measured series and the chance line, thickened to match the text. Panel chrome --
#: gridlines, spines, tick marks -- deliberately is NOT scaled: it is meant to stay recessive
#: against the ink, which is what doubling only the data lines preserves.
GRID_LINE_WIDTH = LINE_WIDTH * GRID_SCALE
#: Extra air between the grid's rows, as `tight_layout`'s ``h_pad``. The triangular layout is what
#: makes it necessary: a column's x title goes on its own lowest *visible* panel, so a shorter
#: column's title can land at the same height as the headings of the row below it, and at the
#: default padding the two read as one row of text. This is the only padding that is not left to
#: `tight_layout` -- the horizontal one still is, since that is what keeps one panel's y tick
#: labels off its neighbour's frame.
GRID_ROW_PAD = 3.0
#: The grid's canvas, **square**. The width is what a page gives a figure, so the height is the
#: free axis, and squaring it makes each cell taller rather than changing the 3 x 3 shape. One
#: shape for every family is what lets two families be flipped between, which is the whole
#: argument for the shared path scheme.
GRID_FIGSIZE = (12.0, 12.0)
BAND_ALPHA = 0.15  # confidence band: readable under the line, never competing with it

#: **Matplotlib's ``markersize`` is the marker's box, not its area**, so shapes drawn at one size
#: do not *look* one size: at size ``m`` a square fills the whole ``m x m`` box, a circle fills
#: pi/4 of it, a triangle or a diamond fill half, and a five-pointed star barely a third -- which
#: is why a square reads as much larger than the triangle beside it. This is the correction that
#: puts them all at one apparent size, so the size channel carries nothing and shape is free to
#: mean one thing (the defense, on the clustering precision/recall figure).
#:
#: **It is not equal *ink*, and equal ink was tried first.** Scaling by
#: ``sqrt(circle_area / own_area)`` makes a pointed shape measurably bigger than the circle it is
#: meant to match, because the eye reads a marker by its *extent* as well as by its area, and a
#: pointed shape spends its extent on the points. Each factor here is the **geometric mean of the
#: two corrections**, equal area and equal extent (``sqrt(area_scale)``): half-way between the two
#: things a reader is doing at once.
MARKER_SIZE_SCALE = {"o": 1.00, "s": 0.94, "^": 1.12, "v": 1.12, "D": 1.12,
                     "*": 1.29, "P": 1.09, "X": 1.09, ">": 1.12, "<": 1.12}

#: Colour slot per defense and per (feature, attack) pair. A series' colour follows the thing it
#: represents, not its position in a particular figure, so a defense keeps its colour whether it
#: is one of two lines or one of eight -- and the same defense is the same colour in every figure.
#: Defenses and methods alike simply take slots in their declared order; the slot is reduced to a
#: hue by :func:`series_style`, and :func:`resolve_slots` handles the case where two entities on
#: one figure would land on the same hue.
DEFENSE_SLOTS = {defense: index for index, defense in enumerate(DEFENSES)}

#: The marker shapes a figure may use where **shape carries the defense**, in the order they are
#: handed out. All filled, all distinguishable at 7 pt, and none of them the ``x`` the reference
#: partitions are marked with. :data:`MARKER_SIZE_SCALE` has an entry for every one of them.
MARKER_SHAPES = ("o", "s", "^", "*", "P", "X", "D", "v")

#: Shape per defense -- what :data:`DEFENSE_SLOTS` is for colour, and for the same reason: a
#: reader flipping between two figures should see one defense wearing one mark. Before this existed
#: shapes were handed out by *position among the defenses present*, so a defense's shape depended
#: on who else was in the figure; each legend decoded itself, but nothing carried across.
#:
#: **It cannot be `DEFENSE_SLOTS` reduced modulo the shapes**, the way a hue is. There are far
#: fewer shapes than defenses, and the collisions land exactly where it would hurt -- two defenses
#: that appear on one figure together sharing a mark. So the assignment is written down instead.
#:
#: **Register a defense here when it first appears on such a figure.** One that is not registered
#: still draws -- :func:`defense_markers` gives it a shape no registered defense on that figure
#: claimed -- but the shape it gets depends on who else is in the figure, which is the property
#: this table exists to provide.
DEFENSE_MARKERS = {
    NO_DEFENSE: "o",
    "styleremix": "s",
    "openanonymity": "^",
    "embad_summary": "*",
    "embad_gemini": "P",
    "embad": "X",
}


def defense_markers(defenses: list[str]) -> dict[str, str]:
    """Marker shape for each of ``defenses``: fixed per defense where registered, distinct always.

    Registered defenses take their :data:`DEFENSE_MARKERS` shape, which is what makes a mark mean
    the same thing on every figure. Anything unregistered takes the first shape no registered
    defense in *this* figure is using, in the order the defenses were given -- distinct within the
    figure, but not stable outside it.

    Past :data:`MARKER_SHAPES` the pool is exhausted and shapes repeat, at which point shape has
    stopped being a key. Nothing draws close to eight defenses on one figure today; if something
    does, the channel needs more shapes rather than a cleverer fallback.
    """
    taken = {DEFENSE_MARKERS[defense] for defense in defenses if defense in DEFENSE_MARKERS}
    spare = [shape for shape in MARKER_SHAPES if shape not in taken]
    markers, index = {}, 0
    for defense in defenses:
        if defense in DEFENSE_MARKERS:
            markers[defense] = DEFENSE_MARKERS[defense]
            continue
        markers[defense] = (spare[index] if index < len(spare)
                            else MARKER_SHAPES[index % len(MARKER_SHAPES)])
        index += 1
    return markers


def series_style(slot: int) -> str:
    """Colour for the series occupying colour slot ``slot``, cycling over the validated hues.

    Colour is the *primary* channel carrying series identity here, and on most figures the only
    one, so two series on one figure must not share a slot modulo the palette. That is what
    :func:`resolve_slots` checks -- and where a second channel is in play (the dash: the feature on a
    ``by_attack`` figure) it is told so, because a hue
    shared by two differently-dashed lines is a channel rather than a collision.
    """
    return CATEGORICAL[slot % len(CATEGORICAL)]


def line_style(dash: tuple):
    """Matplotlib line style for one dash pattern; ``"-"`` for the empty pattern (solid).

    The one place ``()`` is turned into a solid line, so :data:`DATASET_DASHES` and
    :data:`FEATURE_DASHES` can both spell "no dash" as the empty tuple -- which is what lets a
    series carry a dash it may not use and every drawer treat the channel the same way.
    """
    return (0, dash) if dash else "-"


def resolve_slots(slots: list[int], dashes: list[tuple] | None = None) -> list[int]:
    """Colour slots for the series in one figure, guaranteed to be visually distinct.

    Normally this returns ``slots`` unchanged: each series keeps the slot its *entity* owns, so
    colours mean the same thing across figures. Only if two of the entities in this particular
    figure land on the same style -- possible once slots run past the eight-colour palette, as
    the 63 (feature, attack) methods do -- does it fall back to numbering the series 0, 1, 2, ...
    in the order they were given. Being able to tell two lines apart beats cross-figure
    consistency, and since callers order their series deterministically, so is the fallback.

    ``dashes`` is for the figures with a second channel -- the ``by_attack`` ones, where the
    feature rides on the dash -- and two series there *deliberately* share a hue. Passing it makes the check consider the whole style,
    so a shared hue with different dashes is left alone rather than treated as a collision.
    """
    styles = list(zip((series_style(slot) for slot in slots),
                      dashes if dashes is not None else [()] * len(slots)))
    return slots if len(set(styles)) == len(styles) else list(range(len(slots)))


def style_axes(axes, xlabel: str, ylabel: str, title: str, *,
               title_size: float = FONT_PANEL_TITLE, title_weight: str = "bold",
               title_pad: float = 10.0, scale: float = 1.0) -> None:
    """Apply the shared chart chrome: recessive solid grid, no box, text in text colours.

    Every figure in this file goes through here, which is what makes them look like one set.

    ``title`` is a **panel** heading and is the only text of its kind left: a facet panel's known
    interval, or the defense a clustering panel covers. A figure-level title and subtitle are not
    drawn any more -- both were caption text, which belongs in the document that publishes the
    figure rather than burned into the image. **A single-axes figure therefore passes ``""`` here**,
    since its axes title would be that figure's title. The sentences themselves are kept -- see
    :data:`CURVE_TYPES`' fifth field -- and the note above :func:`finish_facets` has the full record.

    ``title_size``/``title_weight``/``title_pad`` exist for the facet grid, whose headings sit at
    the axis-label size and unbolded -- a cell's heading names the same kind of thing its axis
    titles do, so one of them set larger and heavier claimed a hierarchy the design does not have
    -- and close to the panel they belong to, since the heading is that panel's identity. The
    defaults are what a standalone panel heading (a clustering panel's defense) still uses.

    ``scale`` multiplies every point size this function sets, for the facet grid
    (:data:`GRID_SCALE`). It is a parameter rather than a change to the constants because
    the constants are shared with families that are not enlarged. The *chrome* is deliberately
    outside it -- gridline, spine and tick-mark widths are absolute, so scaled text sits on the
    same recessive frame rather than dragging it up with it.
    """
    axes.set_facecolor(SURFACE)
    axes.grid(True, which="both", color=GRID, linewidth=0.7, linestyle="-")
    axes.set_axisbelow(True)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color(AXIS)
        axes.spines[side].set_linewidth(0.8)
    # **Axis text is primary ink, the tick marks stay recessive.** The numbers on an axis and the
    # name of that axis are read, not chrome, so they take the same ink as a panel heading.
    # `labelcolor` is separate from `color` in `tick_params` precisely so the tick *marks* can stay
    # recessive -- blackening those would thicken the frame the skill wants recessive, and nobody
    # reads a tick mark. Every axis in the file goes through here, including the ones whose tick
    # labels are set by hand elsewhere (they inherit this call's `labelcolor`).
    axes.tick_params(color=TEXT_SECONDARY, labelcolor=TEXT_PRIMARY, labelsize=FONT_TICK * scale,
                     length=3, width=0.8)
    axes.set_xlabel(xlabel, color=TEXT_PRIMARY, fontsize=FONT_AXIS_LABEL * scale)
    axes.set_ylabel(ylabel, color=TEXT_PRIMARY, fontsize=FONT_AXIS_LABEL * scale)
    # Left-aligned rather than centred, matching the axis labels above.
    axes.set_title(title, color=TEXT_PRIMARY, fontsize=title_size * scale,
                   fontweight=title_weight, loc="left", pad=title_pad * scale)


def add_legend(axes, **options):
    """A frameless legend in text colours -- identity never rides on colour alone.

    A surface wash and a border were tried and rejected: a legend inside the plot puts marker keys
    among the marks, and on the precision/recall figure the keys are the same shapes at the same
    size as the data. What actually solves that is geometry -- squaring the axes and sizing the
    figure so the block clears the data entirely -- which is the better fix, because a legend that
    overlaps nothing needs nothing drawn under it.
    """
    legend = axes.legend(frameon=False, fontsize=FONT_LEGEND, labelcolor=TEXT_PRIMARY, **options)
    if legend.get_title().get_text():
        legend.get_title().set_color(TEXT_SECONDARY)
        legend.get_title().set_fontsize(FONT_LEGEND)
    return legend


#: Vertical gap between two stacked legends, in **points** -- roughly half a line of the 8.5 pt
#: legend text, enough that the second title does not read as another row of the first list.
#: Points rather than an axes fraction so the gap is the same size on the page whatever the
#: figure's height; :func:`stack_below` converts it.
LEGEND_STACK_PAD = 5.0


def stack_below(figure, axes, legend):
    """Axes-coordinate anchor placing a second legend's top edge under ``legend``'s bottom one.

    Matplotlib has no notion of one legend continuing another, and the first one's height is not
    known in advance -- it grows a row per series -- so the anchor is measured off the drawn
    artist rather than guessed. Pair it with ``loc="upper center"`` and ``borderaxespad=0``, or
    the legend is padded away from the anchor this returns and the two stop lining up. The
    horizontal anchor is the first legend's *centre*, so the block stays centred however much
    wider one of the two is.

    ``draw_without_rendering`` is what gives the first legend its offset: until the figure has been
    laid out once, its box sits at the origin and the extent read back is meaningless.
    """
    figure.draw_without_rendering()
    renderer = figure.canvas.get_renderer()
    box = legend.get_window_extent(renderer)
    return axes.transAxes.inverted().transform(
        (0.5 * (box.x0 + box.x1), box.y0 - renderer.points_to_pixels(LEGEND_STACK_PAD)))


#: Whether :func:`save_figure` writes a PNG beside each PDF. **Off by default**, and set from
#: ``--png``. The PDF is the artefact -- it is what a paper includes -- and the PNG was only ever
#: for a quick look, but it is the more expensive of the two to render (a raster has to be
#: rendered *and* deflated where the PDF only serialises vectors). Writing both by default meant a
#: real share of every run went on the copy nobody publishes.
WRITE_PNG = False


#: Blank border left around a saved figure, in inches. **As near zero as prints correctly**:
#: ``bbox_inches="tight"`` crops the canvas to the artists' own bounding box and then pads it, and
#: matplotlib's default pad is on every side of every figure in the tree. A figure here is placed
#: by the document that includes it, which adds its own space; burning that into the file means
#: it cannot be taken away.
#:
#: **It is not 0, and the reason is a measurement rather than taste.** The box matplotlib crops to
#: is built from each text artist's *font metrics*, and a rendered antialiased glyph spills a
#: pixel or two past that, so at pad 0 the outermost label is shaved. This value is set above the
#: floor where ink stops touching an outer edge, because a glyph with more overhang -- a
#: parenthesis, an italic, a comma below a baseline -- could need the extra pixel. It is still a
#: fraction of matplotlib's default.
#:
#: **This is the OUTER margin only.** The padding *between* facet panels comes from
#: ``tight_layout``'s own defaults and is deliberately left alone -- it is what keeps one panel's
#: tick labels off the next panel's axis, so tightening it does not gain margin, it causes
#: collisions.
#:
#: Artists drawn outside the axes on purpose (the iso-F labels past the right spine, drawn with
#: ``annotation_clip=False``) are part of that bounding box, so the crop lands outside *them*
#: rather than through them.
FIGURE_PAD_INCHES = 0.02


def save_figure(figure, stem: Path) -> Path:
    """Write ``figure`` as a PDF (and a PNG when :data:`WRITE_PNG`), return the PDF."""
    stem.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = stem.parent / f"{stem.name}.pdf"
    paths = [pdf_path] + ([stem.parent / f"{stem.name}.png"] if WRITE_PNG else [])
    for path in paths:
        figure.savefig(path, dpi=200, facecolor=SURFACE, bbox_inches="tight",
                       pad_inches=FIGURE_PAD_INCHES)
    plt.close(figure)
    return pdf_path


# --- discovering runs --------------------------------------------------------

@dataclass(frozen=True)
class Run:
    """One results directory, with its name parsed into the four axes it encodes."""

    dataset: str
    defense: str
    feature: str
    attack: str
    directory: Path

    @property
    def method(self) -> tuple[str, str]:
        """The (feature, attack) pair -- what varies within one defense's figure."""
        return self.feature, self.attack

    @property
    def method_label(self) -> str:
        return f"{FEATURE_LABELS[self.feature]} / {ATTACK_LABELS[self.attack]}"

    @property
    def defense_label(self) -> str:
        return DEFENSE_LABELS[self.defense]


def parse_run_name(name: str) -> tuple[str, str, str, str] | None:
    """Split ``<dataset>_<defense>_<feature>_<attack>`` into its four parts, or return ``None``.

    Every part may itself contain underscores (``swe_chat``, ``dp_mlm_pii``,
    ``gemini_embedding_2``, ``nearest_neighbor``), so the name cannot be split on ``_``. Instead
    each part is matched against the vocabulary above, from both ends inwards; a candidate that
    leaves an unrecognised remainder is rejected and the next one tried, which is what
    distinguishes ``dp_mlm`` from ``dp_mlm_pii``. Anything that does not parse cleanly -- an
    ad-hoc directory, or a run whose name carries extra qualifiers such as ``_langaware`` -- is
    not a comparable run and returns ``None``.
    """
    for dataset in DATASETS:
        if not name.startswith(f"{dataset}_"):
            continue
        remainder = name[len(dataset) + 1:]
        for attack in ATTACKS:
            if not remainder.endswith(f"_{attack}"):
                continue
            middle = remainder[: -(len(attack) + 1)]
            for defense in DEFENSES:
                if middle.startswith(f"{defense}_") and middle[len(defense) + 1:] in FEATURES:
                    return dataset, defense, middle[len(defense) + 1:], attack
    return None


def discover_runs(results_dir: Path) -> list[Run]:
    """Every parseable results directory, sorted so figures are built in a stable order."""
    runs, skipped = [], []
    for directory in sorted(path for path in results_dir.iterdir() if path.is_dir()):
        parsed = parse_run_name(directory.name)
        if parsed is None:
            skipped.append(directory.name)
            continue
        runs.append(Run(*parsed, directory=directory))
    if skipped:
        print(f"skipped {len(skipped)} director{'y' if len(skipped) == 1 else 'ies'} whose name is "
              f"not <dataset>_<defense>_<feature>_<attack>: {', '.join(skipped)}")
    return runs


# --- the known configurations, and what a figure is allowed to average -------

#: Share of the timeline ``run_experiment.py`` holds out of every known side. Every
#: configuration is scored on it, which is what makes them comparable, so it is also the slice
#: every figure here restricts to. A run whose ``--test-fraction`` differed carries a suffix in
#: its directory name and is skipped by :func:`parse_run_name` before it reaches this file.
TEST_FRACTION = 0.25
TEST_START = 1.0 - TEST_FRACTION

#: Sizes (rows) and gaps (columns) of the facet grid, in reading order. The grid is triangular:
#: only ``size + gap <= 0.75`` can exist, because a known side that is both large and far from the
#: test set would have to start before the corpus does. Empty cells are left empty rather than
#: rearranged away -- the hole *is* the design.
GRID_SIZES = (0.25, 0.50, 0.75)
GRID_GAPS = (0.00, 0.25, 0.50)


@dataclass(frozen=True)
class KnownConfig:
    """One known side: the interval ``[start, end)`` of the timeline the attacker was given.

    Mirrors the runner's class of the same name. The two numbers carry the whole design: ``size``
    is how much labelled data the attacker holds, ``gap`` is how stale it is when the shared test
    set begins. A prefix sweep could only move them together.
    """

    start: float
    end: float

    @property
    def tag(self) -> str:
        return f"known{round(self.start * 100):02d}{round(self.end * 100):02d}"

    @property
    def size(self) -> float:
        return self.end - self.start

    @property
    def gap(self) -> float:
        return TEST_START - self.end

    @property
    def label(self) -> str:
        return f"known {self.start:.0%}-{self.end:.0%}"

    @property
    def size_label(self) -> str:
        return f"{self.size:.0%} of the corpus"

    @property
    def gap_label(self) -> str:
        quarters = round(self.gap * 4)
        return "fresh" if quarters == 0 else f"{quarters} quarter{'s' if quarters > 1 else ''} stale"


def parse_config_tag(tag: str) -> KnownConfig | None:
    """``known0025`` -> ``KnownConfig(0.0, 0.25)``; ``None`` for anything else.

    Four digits exactly. The two-digit ``known25`` of the prefix-sweep layout is deliberately
    *not* accepted: it names a known side that started at the beginning of the corpus and was
    scored on a different set of documents, so it is not a cell of this grid and averaging it in
    would be the confound the design exists to remove.
    """
    body = tag[len("known"):] if tag.startswith("known") else ""
    if len(body) != 4 or not body.isdigit():
        return None
    config = KnownConfig(start=int(body[:2]) / 100, end=int(body[2:]) / 100)
    return config if 0 <= config.start < config.end <= TEST_START + 1e-9 else None


def config_predictions(run: Run, in_set_only: bool = True) -> dict[str, pd.DataFrame]:
    """``run``'s per-document predictions on the **shared test set**, one table per configuration.

    The single source every comparison figure derives from, and it makes two restrictions that
    every caller would otherwise have to remember:

    * **The shared test set only.** Each configuration scores its whole remaining future, but a comparison across configurations is only a
      comparison if they are scored on the same documents -- otherwise a fresher known side is
      also being asked an easier question. The cut is ``position >= round(TEST_START * n)``, an
      integer index, so there is no rounding to reproduce.
    * **In-set documents only**, under the default ``in_set_only=True``. A document whose author
      is absent from that configuration's known side has no correct answer available, so it has
      no rank; counting it would cap the curve below 1 for a reason the attack cannot control.
      How many there are is itself a result -- it is reported per panel, because it is exactly
      the *reach* that grows with the known side.

    ``in_set_only=False`` keeps those documents, which is what the ``openset/`` family is built
    from: there the question is not "who wrote this?" but "is this anyone the attacker knows?",
    and a document with no correct answer is the *positive* class rather than an inconvenience.
    Only that family may pass it -- every figure documented as in-set stays in-set, and the two
    populations are never mixed inside one panel. The extra rows carry ``true_author_rank = NaN``
    by construction, so any caller of the unfiltered tables has to say what a miss means rather
    than inheriting it; ``accept_score`` is required there because it is the only column those
    rows have anything to say through.

    Empty for a run written before this design (a prefix sweep, or the older per-window files);
    those are skipped with a note rather than reinterpreted.
    """
    required = {"true_author_rank", "position"} | (set() if in_set_only else {"accept_score"})
    tables: dict[str, pd.DataFrame] = {}
    for path in sorted(run.directory.glob(f"predictions_{run.attack}_known*.csv")):
        config = parse_config_tag(path.stem[len(f"predictions_{run.attack}_"):])
        if config is None:
            continue
        table = pd.read_csv(path)
        if "attack" in table.columns:
            table = table[table["attack"] == run.attack]
        if table.empty or not required.issubset(table.columns):
            continue
        n_documents = int(table["position"].max()) + 1     # the future always runs to the end
        table = table[table["position"] >= round(TEST_START * n_documents)]
        if in_set_only:
            table = table[table["author_in_known"].astype(bool) & table["true_author_rank"].notna()]
        else:
            # A non-finite score cannot be ranked against the others, so it would silently take
            # whichever end of the sort numpy puts it at. There are none in practice.
            table = table[np.isfinite(table["accept_score"].to_numpy(dtype=float))]
        if not table.empty:
            tables[config.tag] = table
    return tables


# --- uncertainty: a clustered bootstrap over users, never a t interval -------
#
# What replaced the Student-t band this file used to draw across rolling windows. That band was
# not defensible: the windows were nested prefixes of one corpus sharing their known sides, so
# they were not independent draws -- positive correlation inflates the variance of their mean
# while shrinking the sample variance, making `t * s / sqrt(n)` anti-conservative, and on a
# measured run it came out visibly narrower than the spread it claimed to summarise. It also
# mixed two different quantities: how much the number moves as the *design* moves (a sensitivity,
# now shown by the facet grid itself) and how much it would move on another sample of users (a
# confidence interval, which is what this is).

#: Bootstrap replicates behind every band -- enough for the 2.5/97.5 percentile to be stable, and
#: the whole cost is a weighted ``bincount`` per replicate per panel, cheap next to the attack run.
BOOTSTRAP_REPLICATES = 1000

#: Fixed so a figure redrawn tomorrow has the same band as the one in the paper.
BOOTSTRAP_SEED = 20260803


class AuthorBootstrap:
    """Cluster resample of *users*, drawn once and shared by every panel of one dataset.

    Two decisions, both load-bearing:

    * **The unit is the user, not the document.** Documents by one person are strongly
      correlated -- a distinctive, prolific user's documents are all hits -- so resampling
      documents understates the noise badly, the same anti-conservative error as dividing by the
      square root of a nested window count, in a different disguise.
    * **One draw, applied everywhere.** The same replicate's user multiplicities are used for
      every known configuration and every run of the dataset. That is what makes the
      configurations *paired*: a replicate that drops a user drops them from all six panels at
      once, exactly as the overlap behaves in reality, so a difference between panels can be
      bootstrapped without pretending they are independent samples.

    Counts come from a multinomial over the whole test-side population, and a panel weights the
    users that fall in it. The estimator is a ratio -- weighted hits over weighted documents -- so
    the fluctuating total is divided out, and what remains is the variability of "which users the
    attacker happened to be scored against".
    """

    def __init__(self, authors, n_replicates: int = BOOTSTRAP_REPLICATES,
                 seed: int = BOOTSTRAP_SEED) -> None:
        self.authors = pd.Index(sorted(set(authors)))
        self.n_replicates = int(n_replicates)
        rng = np.random.default_rng(seed)
        size = len(self.authors)
        self.counts = (rng.multinomial(size, np.full(size, 1.0 / size), size=self.n_replicates)
                       .astype(np.float32) if size and self.n_replicates else
                       np.empty((0, size), dtype=np.float32))

    def multiplicities(self, author_labels) -> np.ndarray:
        """``(n_replicates, len(author_labels))`` counts for any sequence of author labels.

        Users outside the drawn universe (impossible for panels built from the same dataset)
        would weigh zero, which is the correct behaviour rather than an error.
        """
        index = self.authors.get_indexer(pd.Index(author_labels))
        weights = np.where(index >= 0, index, 0)
        block = self.counts[:, weights]
        return np.where(index >= 0, block, 0.0)

    def document_weights(self, author_labels) -> np.ndarray:
        """``(n_replicates, n_documents)`` multiplicities, one column per document."""
        return self.multiplicities(author_labels)


class PanelWeights:
    """One table's bootstrap multiplicities, built once and shared by every curve type.

    A dozen curve types are computed from the same predictions table, and each used to ask the
    bootstrap for its own copy of the same ``(replicates x documents)`` matrix, rebuilding it
    several times over. Building it once and passing this object down instead is both faster and
    -- the reason that matters more here -- stops several copies of it being live at once under
    this cluster's memory cap.
    """

    def __init__(self, bootstrap: AuthorBootstrap, author_labels) -> None:
        self.bootstrap = bootstrap
        #: ``(replicates x documents)``: each document weighted by its own author's multiplicity.
        self.documents = bootstrap.document_weights(author_labels)

    def for_authors(self, author_labels) -> np.ndarray:
        """``(replicates x authors)`` multiplicities, for a curve whose unit is the user.

        Not cached: it asks with that panel's distinct users rather than with the table's
        one-row-per-document labels.
        """
        return self.bootstrap.multiplicities(author_labels)


def band_from_replicates(replicates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pointwise 95% percentile interval from an already-built ``(replicates x grid)`` matrix.

    Split out from :func:`bootstrap_band` so the curve types that can compute every replicate in
    one vectorised pass share the same summary step instead of reimplementing it.

    **``np.percentile`` wherever it is safe.** ``nanpercentile`` is meaningfully slower whether or
    not the array actually holds a NaN, and a full sweep found every band NaN-free: the curves that
    *can* emit one (a cohort some replicate emptied) never do on real data. The check costs one
    pass against the partition it guards, so the NaN path stays rather than being asserted away.
    """
    percentile = np.nanpercentile if np.isnan(replicates).any() else np.percentile
    low, high = percentile(replicates, [2.5, 97.5], axis=0)
    return np.clip(low, 0.0, 1.0), np.clip(high, 0.0, 1.0)


def bootstrap_band(values, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pointwise 95% percentile interval for a curve, from one panel's replicate weights.

    ``values`` is called once per replicate with that replicate's document weights and returns
    the curve on a fixed grid. Intervals are **pointwise**, not a simultaneous band over the whole
    curve: two curves whose bands overlap at every k are not resolved by this data, but the
    converse is a per-point statement.

    Curve types whose inner loop is a prefix sum over a *sorted* order (the DIR-FAR sweep) stay
    on this per-replicate path: the cut-off is a ``searchsorted`` into each replicate's own
    running total.
    """
    if not len(weights):
        point = values(np.ones(weights.shape[1] if weights.ndim == 2 else 0))
        return point, point
    return band_from_replicates(np.stack([values(row) for row in weights]))


def weighted_cmc(ranks: np.ndarray, pools: np.ndarray, ks: np.ndarray,
                 weights: np.ndarray | None = None) -> np.ndarray:
    """Top-k accuracy at every ``ks``, with each document weighted by its author's multiplicity.

    Mirrors :func:`prompt_anonymity.evaluation.metrics.ranking.cmc_curve` -- the share of documents whose
    true author ranks within k -- but weighted, and written out here rather than imported to keep
    this script free of the package's heavy imports (the same reason its name vocabulary is
    literal). Computed as a weighted histogram over ranks plus a prefix sum, so one replicate
    costs one pass over the documents rather than a ``k x documents`` comparison, which at
    scale would be a prohibitively large array.

    **Ranks are rounded up before bucketing**, which is what makes the histogram equivalent to
    ``ranks <= k``. ``true_author_rank`` averages ties (a true author tied with one other for
    first is rank 1.5), so truncating instead counted that document as a top-1 hit and made every
    CMC curve slightly optimistic, disagreeing with :func:`counting_modes` and with
    ``author_report_*.csv`` about the same number. The package's ``cmc_curve`` documents ``<=`` as
    the conservative reading; this now matches it.
    """
    weights = np.ones(len(ranks), dtype=np.float64) if weights is None else weights
    total = weights.sum()
    if total <= 0:
        return np.zeros(len(ks))
    histogram = np.bincount(np.ceil(ranks).astype(np.int64), weights=weights,
                            minlength=int(ks[-1]) + 2)
    return np.cumsum(histogram)[ks] / total


#: Replicates folded per pass in :func:`grouped_sums`. Bounds the float64 working copy of the
#: weight matrix to a few tens of MB, which matters with several workers live at once.
REPLICATE_BLOCK = 256


def grouped_sums(labels: np.ndarray, weights: np.ndarray
                 ) -> tuple[np.ndarray, np.ndarray]:
    """Every replicate's weighted histogram at once: the labels present, and their sums.

    The batched form of ``np.bincount(labels, weights=row)``, and the reason two of the curve
    types no longer cost a Python call per replicate. Sorting the documents by label *once* turns
    each replicate's histogram into a single ``np.add.reduceat`` over the whole weight matrix, so
    a thousand replicates are one pass rather than a thousand.

    Returns ``(present, sums)`` -- the distinct labels in ascending order and an
    ``(n_replicates, len(present))`` array against them -- rather than a dense histogram over
    every possible label, because the callers' label spaces are sparse relative to the pool size.

    Replicates are folded in blocks because the accumulation wants float64 (``reduceat`` would
    otherwise carry the weights' float32 through a cumulative sum tens of thousands of terms long)
    and a float64 copy of a whole weight matrix would be sizeable per call.
    """
    order = np.argsort(labels, kind="stable")
    sorted_labels = labels[order]
    starts = np.flatnonzero(np.r_[True, np.diff(sorted_labels) != 0])
    present = sorted_labels[starts]
    sums = np.empty((len(weights), len(present)))
    for begin in range(0, len(weights), REPLICATE_BLOCK):
        block = weights[begin:begin + REPLICATE_BLOCK][:, order].astype(np.float64)
        sums[begin:begin + REPLICATE_BLOCK] = np.add.reduceat(block, starts, axis=1)
    return present, sums


def cmc_replicates(ranks: np.ndarray, ks: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """:func:`weighted_cmc` for every replicate at once: ``(n_replicates, len(ks))``.

    The same quantity as calling it in a loop, computed in one pass over :func:`grouped_sums`.
    The cumulative sum runs over the ranks actually present rather than over every integer up to
    the pool size, and ``ks`` is then read off it by ``searchsorted``.
    """
    if not len(ranks) or not len(weights):
        return np.zeros((len(weights), len(ks)))
    present, sums = grouped_sums(np.ceil(ranks).astype(np.int64), weights)
    cumulative = np.cumsum(sums, axis=1)
    totals = cumulative[:, -1:]                 # every document counted, whatever its rank
    # Where each k falls among the ranks present; -1 means "before the smallest", i.e. no hits.
    lookup = np.searchsorted(present, ks, side="right") - 1
    counted = np.where(lookup >= 0, cumulative[:, np.clip(lookup, 0, None)], 0.0)
    return np.divide(counted, totals, out=np.zeros_like(counted), where=totals > 0)


def chance_cmc(pools: np.ndarray, ks: np.ndarray) -> np.ndarray:
    """Uniform random-guessing top-k over the same candidate pools: ``mean(min(k, pool) / pool)``.

    A property of the pool sizes rather than of the attack, so it is a point estimate with no
    band: nothing about it is being measured.

    Kept as the *weakest* reference and carried in every companion CSV, but it is no longer what
    the figures draw -- see :func:`prior_inclusion` for why, and what replaced it.
    """
    pools = np.sort(pools)
    inverse = np.concatenate([[0.0], np.cumsum(1.0 / pools)])
    saturated = np.searchsorted(pools, ks, side="right")
    return (saturated + ks * (inverse[-1] - inverse[saturated])) / len(pools)


# --- the proportional-guessing baseline --------------------------------------
#
# Uniform 1/N is the weakest baseline there is, and on a heavy-tailed corpus it is *far* too
# weak to be the line a reader measures the attack against: an attacker who reads no text at all
# and simply always names the known side's most prolific author does far better while still
# knowing nothing about writing style. So the baseline drawn here is a guesser that knows how many
# documents each known author wrote and nothing else.

#: Replicates behind :func:`prior_inclusion`. The estimate is a mean over prior-weighted random
#: rankings, so its error falls as ``1/sqrt(R)``; well inside the line width and far below the
#: spread between the curves it sits under.
BASELINE_REPLICATES = 2000

#: Fixed, so a figure is reproducible and two runs of this script cannot disagree by a hair. The
#: same reasoning as the seeded :class:`AuthorBootstrap`.
BASELINE_SEED = 20260806

#: Geometric ratio for the k grid the baseline is evaluated on, above the exhaustive head.
BASELINE_K_RATIO = 1.12

#: Every k up to here is on the grid exactly; beyond it the grid is geometric. The head is where
#: a CMC curve is actually read, and it is also where interpolation would be least forgiving.
BASELINE_K_HEAD = 32


def baseline_k_grid(n_candidates: int) -> np.ndarray:
    """The k values :func:`prior_inclusion` evaluates, exhaustive at the head then geometric.

    The inclusion probabilities are an ``(authors x k)`` array, so evaluating them at every k
    would be ``n_candidates`` squared. A log grid costs far fewer columns instead and loses
    nothing: the curve is drawn on a log x axis and is monotone in k, so
    :func:`interpolate_baseline` recovers the intermediate points to well under a pixel.
    """
    head = np.arange(1, min(BASELINE_K_HEAD, n_candidates) + 1)
    if n_candidates <= BASELINE_K_HEAD:
        return head
    steps = int(np.ceil(np.log(n_candidates / BASELINE_K_HEAD) / np.log(BASELINE_K_RATIO)))
    tail = np.unique(np.round(BASELINE_K_HEAD * BASELINE_K_RATIO ** np.arange(1, steps + 1)))
    return np.unique(np.r_[head, tail[tail <= n_candidates], n_candidates]).astype(int)


def prior_inclusion(shares: np.ndarray, ks: np.ndarray, replicates: int = BASELINE_REPLICATES,
                    seed: int = BASELINE_SEED) -> np.ndarray:
    """``(n_authors x len(ks))``: P(author lands in the top k of a prior-weighted random ranking).

    The baseline attacker knows each known author's **share of the known side's documents**,
    ``shares``, and nothing else -- no text, no features. For one document it names k distinct
    authors, drawn without replacement with probability proportional to that share. Equivalently
    it draws a whole random ranking under successive proportional sampling and returns the first
    k, which is the Plackett-Luce model, and that is what makes every k one object instead of one
    experiment per k.

    There is no closed form for the inclusion probability of successive proportional sampling, so
    it is estimated by Monte Carlo through the **exponential-race** construction: draw
    ``E_a ~ Exp(1)`` independently and rank by ``E_a / p_a``. That sequence is distributed exactly
    as sampling without replacement proportional to ``p``, so each replicate is one ``argsort``
    rather than k sequential draws with renormalisation.

    Ranks are folded straight into the k grid rather than kept, because the ranks themselves are
    an ``(R x authors)`` array that is only ever read through the grid.
    """
    shares = np.asarray(shares, dtype=np.float64)
    n_authors = len(shares)
    generator = np.random.default_rng(seed)
    # `searchsorted` maps a 0-based rank to the first grid point that covers it; a rank beyond the
    # last grid point lands in the overflow column, which the cumulative sum below never reaches.
    counts = np.zeros((n_authors, len(ks) + 1))
    author_offsets = np.arange(n_authors) * (len(ks) + 1)
    for begin in range(0, replicates, REPLICATE_BLOCK):
        block = min(REPLICATE_BLOCK, replicates - begin)
        race = generator.exponential(size=(block, n_authors)) / shares
        order = np.argsort(race, axis=1)
        # `order` lists authors best-first; inverting it gives each author its own 0-based rank.
        rank = np.empty_like(order)
        np.put_along_axis(rank, order, np.broadcast_to(np.arange(n_authors), order.shape), axis=1)
        bucket = np.searchsorted(ks, rank + 1, side="left")
        counts += np.bincount((author_offsets + bucket).ravel(),
                              minlength=n_authors * (len(ks) + 1)
                              ).reshape(n_authors, len(ks) + 1)
    return np.cumsum(counts[:, :len(ks)], axis=1) / replicates


def interpolate_baseline(grid_ks: np.ndarray, values: np.ndarray, ks: np.ndarray) -> np.ndarray:
    """A baseline computed on :func:`baseline_k_grid` read off at every k a curve is drawn at.

    Interpolated in ``log k`` because that is the axis the figure uses, so the straight segments
    the eye sees between grid points are the straight segments this draws.
    """
    return np.interp(np.log(ks), np.log(grid_ks), values)


@dataclass
class ProportionalBaseline:
    """One (dataset, known configuration)'s proportional-guessing baseline, at both counting levels.

    ``documents`` is the document-level curve -- ``sum_a q_a * pi_a(k)``, with ``q_a`` the
    author's share of the *scored* documents -- and ``identities`` the identity-level one:
    guesses are drawn independently per document, so an author with ``m_a`` scored documents is
    named at least once with probability ``1 - (1 - pi_a(k)) ** m_a``, averaged over authors.

    Both are point estimates with no band, like :func:`chance_cmc`: a baseline is a property of
    the corpus, and nothing about it is being measured.
    """

    ks: np.ndarray
    documents: np.ndarray
    identities: np.ndarray
    #: Per-author ``pi_a(k)`` on :attr:`ks`, and ``m_a``, kept so a caller can compose the
    #: guesser with something else -- which the two ``openset/`` identification families need,
    #: because the identity level is not linear in the accept rate (see
    #: :meth:`identities_at_rate`). It is the *scored* authors' rows of an array that already
    #: exists, on the sparse k grid rather than every k, which is what keeps it affordable.
    inclusion: np.ndarray
    document_counts: np.ndarray

    @property
    def top1_inclusion(self) -> np.ndarray:
        """Each scored author's chance of being named first. The grid starts at k = 1 exactly."""
        return self.inclusion[:, 0]

    def for_documents(self, ks: np.ndarray) -> np.ndarray:
        return interpolate_baseline(self.ks, self.documents, ks)

    def for_identities(self, ks: np.ndarray) -> np.ndarray:
        return interpolate_baseline(self.ks, self.identities, ks)

    def identities_at_rate(self, rate: np.ndarray) -> np.ndarray:
        """Identity-level top-1 when only a ``rate`` share of documents is answered at all.

        **Not** ``rate * for_identities(1)``, and the difference is not a rounding error. A
        document is named correctly with probability ``rate * pi_a``, and the author escapes only
        if all ``m_a`` of theirs miss, so the composition goes *inside* the power:
        ``mean_a (1 - (1 - rate * pi_a) ** m_a)``. Scaling the composed number instead would be
        right only for authors with one document -- for a heavy user, half the chances is still a
        lot of chances.

        The document level needs no such method: ``sum_a q_a * rate * pi_a`` really is
        ``rate * for_documents(1)``, since expectation is linear where "at least once" is not.
        """
        inclusion = rate[None, :] * self.top1_inclusion[:, None]
        return np.mean(1.0 - (1.0 - inclusion) ** self.document_counts[:, None], axis=0)

    def identities_for_k_at_rate(self, rate: float, ks: np.ndarray) -> np.ndarray:
        """:meth:`identities_at_rate` transposed: one accept rate, swept over **k**.

        The baseline for ``openset/top_k_cmc/author``, where the threshold is pinned and k is the
        axis. Same composition inside the power, with ``pi_a(k)`` in place of ``pi_a(1)``:
        ``mean_a (1 - (1 - rate * pi_a(k)) ** m_a)``.

        Evaluated on :attr:`ks` and interpolated in log k afterwards, exactly as
        :meth:`for_identities` is and for the same reason -- doing it at every k would build a
        far larger array to answer a question the sparse grid already answers.
        """
        composed = np.mean(1.0 - (1.0 - rate * self.inclusion) ** self.document_counts[:, None],
                           axis=0)
        return interpolate_baseline(self.ks, composed, ks)


def known_inclusion(dataset: str, known_config: str) -> tuple[np.ndarray, np.ndarray, pd.Index]:
    """``(ks, inclusion, authors)`` for one known side, memoised across every run that shares it.

    The Monte Carlo is a property of the *known side* -- which authors the attack ranks over and
    how much each of them wrote -- so it is the same for every defense, feature and attack run
    against that configuration. Computing it once per (dataset, configuration) is what keeps it
    off the per-run path, since many runs share one known side and would otherwise each pay for it.

    Warmed in the parent process by :func:`warm_baselines` before the curve workers fork, so the
    children inherit the memo rather than each filling their own copy.
    """
    key = (dataset, known_config)
    if key not in _INCLUSION:
        shares = known_author_shares(dataset, known_config)
        if shares is None:
            _INCLUSION[key] = None
        else:
            ks = baseline_k_grid(len(shares))
            _INCLUSION[key] = (ks, prior_inclusion(shares.to_numpy(), ks), shares.index)
    return _INCLUSION[key]


#: Memo for :func:`known_inclusion`, one entry per (dataset, known configuration).
_INCLUSION: dict[tuple[str, str], tuple | None] = {}


def config_baseline(dataset: str, known_config: str,
                    scored_authors: pd.Series) -> ProportionalBaseline | None:
    """Both baseline curves for one panel: the known-side prior met with the scored documents.

    ``scored_authors`` is one entry per scored document, so its value counts give both ``q_a``
    (the document-level weight) and ``m_a`` (how many chances the guesser gets at that author).
    Authors scored but absent from the known side cannot happen -- an in-set document is by
    definition one whose author is enrolled -- and are dropped defensively rather than left to
    silently reweight the rest.

    ``None`` when the dataset's split parquet is not on disk to supply the prior, which is the
    same condition that costs the word-count split its counts.
    """
    resolved = known_inclusion(dataset, known_config)
    if resolved is None:
        return None
    ks, inclusion, authors = resolved
    counts = scored_authors.value_counts()
    counts = counts[counts.index.isin(authors)]
    if counts.empty:
        return None
    scored = inclusion[authors.get_indexer(counts.index)]
    weights = counts.to_numpy()
    return ProportionalBaseline(
        ks=ks,
        documents=weights @ scored / weights.sum(),
        identities=np.mean(1.0 - (1.0 - scored) ** weights[:, None], axis=0),
        inclusion=scored,
        document_counts=weights,
    )


@dataclass
class ConfigCmc:
    """One (run, known configuration) CMC curve with the rejection filter applied.

    ``curve`` has one row per k with ``accuracy``, the two baselines (``random`` uniform and
    ``random_proportional``, the one drawn) and the pointwise ``ci_low``/``ci_high`` bootstrap
    bounds. ``n_documents``/``n_users`` are the in-set counts the panel is drawn from -- printed
    on the figure because they are not a constant across the grid: a larger or fresher known side
    enrolls more of the test set's users, and that *reach* is part of what it buys.

    Serves both counting levels: ``level`` is ``"document"`` (share of documents whose author
    ranks within k) or ``"identity"`` (share of users with **at least one** document within k),
    and the two differ by a factor of several on the same run.
    """

    curve: pd.DataFrame
    n_documents: int
    n_users: int
    n_candidates: int
    max_k: int
    level: str = "document"
    #: The out-of-set cohort in this level's unit. The panel note it feeds -- ``X out, Y in`` over
    #: ``Z cand.`` -- is shared with ``openset/dirfar``, so the two slices of one
    #: ``DIR(threshold, k)`` surface are annotated alike.
    #:
    #: The achieved FAR is in the companion CSV's ``false_accept_rate`` column rather than a note
    #: line, since it is panel-wide (every attack is held to one budget over one out-of-set
    #: cohort) and :func:`draw_cmc_panel` prints only ``series[0]``'s note for the whole panel --
    #: a per-series quantity (how many in-set documents the threshold keeps) does not belong here
    #: and is instead the CSV's ``n_accepted`` column.
    n_ood: int | None = None
    #: The known side's pool for the ``cand.`` line, in the level's unit (known documents at
    #: ``doc/``, known users at ``author/``), read from ``rolling_results.csv`` by
    #: :func:`run_curves` exactly as :class:`ConfigIdentification`'s is. ``None`` leaves the line
    #: off rather than guessing.
    n_known: int | None = None
    #: Where that note sits. The file's bottom-right default would be free for an unfiltered CMC
    #: curve, which climbs to the top right; ``openset/top_k_cmc/`` cannot use it: the filter
    #: flattens the curve into a low band, so it ends in the bottom right rather than above it, and
    #: the corner is chosen by measurement rather than assumed. A few of swe-chat's author panels
    #: still collide, since their curves step up immediately and stay flat across the full width,
    #: leaving no corner free at all.
    #: Re-check this if a `PANEL_Y_LIMITS` entry or the note's line count changes -- both can move
    #: which corner is free.
    note_corner: str = "lower right"

    @property
    def top1(self) -> float:
        return float(self.curve["accuracy"].iloc[0])


# --- counting per user instead of per document -------------------------------
#
# Every curve in this file exists at two levels, and they answer different questions:
#
#   doc/     what share of the *traffic* can be attributed
#   author/  what share of the *people* were linked -- a person counts once **any one** of their
#            documents is attributed to them, which is the right reading when being linked at all
#            is the harm
#
# The author level is not a re-run: it is the same predictions collapsed to one row per person,
# taking the best each of their documents achieved. That is what `author_table` builds, with the
# document-level column names kept, so builders whose only inputs are `accept_score` and
# `author_in_known` work on it unchanged. The DIR-FAR builders need their own, because "any
# accepted document" is not a property one collapsed row can carry; see
# `weighted_identification_any`.


def author_table(table: pd.DataFrame) -> pd.DataFrame:
    """One row per user, carrying the best each of their documents achieved.

    Column names match the document table so a builder cannot tell the difference, which is what
    lets one implementation serve both levels:

    * ``true_author_rank`` -- their **best** rank, so "rank within k" becomes "at least one
      document within k". This is exactly what :func:`config_identity` counts.
    * ``accept_score`` -- their **lowest**, i.e. their most enrolled-looking document (the score
      runs the other way: higher is more stranger-like). A person is hard to reject if any one of
      their documents looks enrolled.
    * ``hit_score`` -- the confidence of their most confident **correct** document, or ``-inf``
      when they have none. Not a document-level column: it is what
      :func:`weighted_selective_any` needs to know *when* a person becomes linked as the
      attacker's threshold drops, which their best rank alone cannot say.
    * ``n_documents`` -- how many they wrote, so a panel can report both counts.

    A stranger's rank is NaN in every row, so their ``true_author_rank`` stays NaN and their
    ``hit_score`` stays ``-inf``: they can never be correctly linked, at any threshold.
    """
    confidence = -table["accept_score"].to_numpy(dtype=float)
    correct = (table["true_author_rank"].to_numpy(dtype=float) <= 1)
    frame = table.assign(_confidence=confidence,
                         _hit=np.where(correct, confidence, -np.inf))
    grouped = frame.groupby("true_author", sort=True)
    collapsed = pd.DataFrame({
        "true_author_rank": grouped["true_author_rank"].min(),
        "accept_score": grouped["accept_score"].min(),
        "hit_score": grouped["_hit"].max(),
        "author_in_known": grouped["author_in_known"].first(),
        "n_candidate_authors": grouped["n_candidate_authors"].max(),
        "n_documents": grouped.size(),
    }).reset_index()
    return collapsed


# --- binning documents-per-author counts, shared with the word-count split ----------------


def count_bin_labels(edges: tuple[int, ...]) -> tuple[str, ...]:
    """Tick labels for a tuple of bin edges: ``1``, ``2``, ``3-4``, ... , ``33+``.

    Derived from the edges rather than written out beside them, so re-binning cannot leave the
    axis claiming the old ranges.
    """
    labels = []
    for index, low in enumerate(edges):
        high = edges[index + 1] if index + 1 < len(edges) else None
        labels.append(f"{low}+" if high is None else
                      str(low) if high - low == 1 else f"{low}-{high - 1}")
    return tuple(labels)


#: Users a bin needs before its accuracy is drawn. The bootstrap's resampling unit is the user,
#: so a bin standing on three of them is noise however many documents they wrote between them --
#: which is why the gate counts users at *both* levels. The bar stays either way: the population
#: is a result, and a bin dropped for thinness should still be visible as the handful it was.
MIN_AUTHORS_PER_BIN = 5


def bin_means(bins: np.ndarray, values: np.ndarray, n_bins: int) -> np.ndarray:
    """Unweighted mean of ``values`` within each bin; NaN where the bin holds nothing.

    NaN rather than zero, because an empty bin has no mean -- drawing one at zero would put a
    point on the floor of the axis where there is no measurement.
    """
    totals = np.bincount(bins, minlength=n_bins).astype(float)
    sums = np.bincount(bins, weights=np.asarray(values, dtype=float), minlength=n_bins)
    return np.divide(sums, totals, out=np.full(n_bins, np.nan), where=totals > 0)


def binned_rate(bins: np.ndarray, hit: np.ndarray, weights: np.ndarray, n_bins: int
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-bin hit rate and its bootstrap band, at either counting level.

    ``bins`` and ``hit`` carry one entry per counted unit -- a document at the document level, a
    user at the author level -- and ``weights`` is that unit's ``(replicates x units)``
    multiplicity matrix, so one implementation serves both.

    The band goes through :func:`grouped_sums` twice over the **same** label array: once for the
    denominators and once for the numerators, with the hits folded into the weights rather than
    selected out of them. Subsetting the columns instead would hand the two calls different label
    sets the moment a bin holds no hits, and their results have to be divided by each other.
    """
    accuracy = bin_means(bins, hit, n_bins)
    if not len(weights):
        return accuracy, accuracy, accuracy
    present, denominators = grouped_sums(bins, weights)
    _, numerators = grouped_sums(bins, weights * hit.astype(weights.dtype))
    rate = np.divide(numerators, denominators, out=np.full_like(numerators, np.nan),
                     where=denominators > 0)
    low, high = np.full(n_bins, np.nan), np.full(n_bins, np.nan)
    low[present], high[present] = band_from_replicates(rate)
    return accuracy, low, high


# --- the open world: the documents nobody the attacker knows wrote -----------
#
# A closed-set figure answers "which known author wrote this?" only on the documents where
# that question has an answer. In this corpus that is the minority: much of both test quarters was
# written by somebody absent from the known side. The figures in this
# section put them back, using the two columns their rows do carry -- `author_in_known`, which is
# the ground-truth label, and `accept_score`, the attack's cohort-normalised margin, which
# `run_experiment.py` writes for *every* unknown document whether or not `--ood reject` was on.
#
# THE OPERATING POINTS HERE ARE ORACLE ONES. All of these runs were `--ood none`, so the
# threshold `calibrate_threshold` would have chosen is not recoverable -- it needs the known-side
# embeddings and a refit. The curves are threshold-free and so unaffected, but any point read off
# them (a FAR) is picked knowing the true labels. Read them as what a
# perfect calibrator could reach, never as what the runner's calibrator achieved.


#: False-accept rates the DIR-FAR curve is evaluated at -- its x axis, labelled ``FAR``: the share
#: of **out-of-set** rows the threshold wrongly accepts. The curve's column is named
#: ``false_accept_rate``. ``accept_score`` runs the other way round, higher meaning more
#: out-of-set. Dense and linear because unlike a CMC this curve has no privileged decade -- the
#: whole trade-off is the result.
DETECTION_GRID = np.linspace(0.0, 1.0, 101)


# --- open-set identification: rejection and attribution scored as one event --------------------
#
# A closed-set curve asks who wrote a document and is told the answer is somebody the attacker
# knows; a detection curve asks whether it is somebody the attacker knows and never asks who.
# Neither is what an attacker actually does, which is both at once: threshold the rejection
# score, then attribute whatever survives. A document counts here
# only if it clears *both* bars.
#
# Why this shape and not an "overall accuracy" or a balanced one. Scoring correct rejections as
# successes makes the metric a readout of the out-of-set rate, which is high enough that plain
# (N+1)-class accuracy is maximised by rejecting *everything* -- and the whole threshold sweep
# barely moves it. Balanced accuracy fails the same way for a subtler reason: its two arms have
# incomparable ceilings, since the true-reject rate spans [0, 1] while the identification arm
# cannot exceed closed-set top-1, so it is essentially half the true-reject rate and its argmax is
# again the attacker that gives up. DIR-FAR avoids it by never crediting a rejection: the only
# thing on the y axis is a re-identification that survived the filter.

#: The false-accept rate the panel note quotes. 5% is the conventional operating point for a
#: watchlist, low enough that most of an attack's identification power has already been spent and
#: not so low that a handful of out-of-set documents decide it.
#:
#: **The last clause is a real constraint on a corpus with few out-of-set documents in its largest
#: configuration**, where the curve between two operating points is interpolation rather than
#: measurement -- this family interpolates between attained operating points, as a ROC does, while
#: :func:`prompt_anonymity.evaluation.metrics.detection.detection_identification_rate` interpolates
#: between adjacent out-of-set *scores*. Neither is wrong; do not quote a low-FAR number on a small
#: cohort without saying which.
IDENTIFICATION_NOTE_FAR = 0.05


@dataclass
class ConfigIdentification:
    """One (run, configuration) DIR-FAR curve: identification that survives a reject threshold.

    The axes, in the vocabulary of the open-set identification literature this borrows from (it
    is the "watchlist ROC"; NIST FRVT spells the same two quantities FPIR and FNIR):

    * **FAR**, the false-accept rate -- the share of out-of-set documents the threshold wrongly
      lets through. Exactly ``1 -`` :class:`ConfigDetection`'s y axis, so the two families are
      the same threshold sweep read from opposite ends and a point on one locates a point on the
      other.
    * **DIR**, the detection-and-identification rate -- the share of **in-set** documents that
      are both accepted and ranked top-1. A scalar version of this is already in the package as
      :func:`prompt_anonymity.evaluation.metrics.detection.detection_identification_rate`, which
      thresholds at the same percentile of the out-of-set scores.

    Two edges make it a join rather than a third opinion. At ``FAR = 1`` nothing is rejected, so
    ``top1`` is exactly the closed-set top-1 on the same run and configuration (verified to the
    digit on every cell). At ``FAR = 0`` it is what the attacker retains when it
    is not allowed a single false accept. Everything between is the cost of the open world,
    measured in the currency the rest of the project reports.

    ``k = 1`` only. DIR is defined at any rank and the surface ``DIR(tau, k)`` has the CMC curve
    as its ``FAR = 1`` slice, but a second k would need a second visual channel in a panel where
    colour is already the entity and a dash already means "not a measurement". A fixed-threshold
    sweep over k is a separate figure, not a line on this one.
    """

    curve: pd.DataFrame
    #: DIR at ``FAR = 1``: the attack's plain closed-set top-1, which is the identity that ties
    #: this family to the closed-set numbers.
    top1: float
    #: The drawn baseline's own right edge, i.e. what the no-information reference reaches when
    #: nothing is rejected. The slope of the line at the document level; not a slope at the
    #: author level, where the line is curved.
    chance: float
    n_documents: int
    n_users: int
    n_ood: int
    level: str = "document"
    #: The known side's pool in this level's unit -- known **documents** at the document level,
    #: known **users** at the author level -- printed as the panel's ``cand.`` line. Read from
    #: the run's ``rolling_results.csv`` (``n_known_docs`` / ``n_known_authors``) by
    #: :func:`run_curves`; ``None`` when that file lacks the configuration, and the line is then
    #: left off rather than guessed.
    n_candidates: int | None = None

    @property
    def unit(self) -> str:
        """What one row of this curve is -- named on the panel, since the counts differ 5x."""
        return "users" if self.level == "author" else "documents"

    def dir_at(self, far: float) -> float:
        """DIR at one false-accept rate, read off the drawn curve rather than recomputed."""
        return float(np.interp(far, self.curve["false_accept_rate"],
                               self.curve["identification_rate"]))


def weighted_identification(is_ood: np.ndarray, identified: np.ndarray, order: np.ndarray,
                            weights: np.ndarray | None = None) -> np.ndarray:
    """DIR on :data:`DETECTION_GRID`, given a **pre-sorted** most-enrolled-looking-first order.

    :func:`weighted_roc`'s twin, and deliberately built the same way: both cohorts' cumulative
    weights along one order, so a bootstrap replicate costs two prefix sums instead of another
    sort of 43,000 documents. The only difference is what the y axis counts -- accepted **and
    correctly attributed** in-set documents.

    ``identified`` is the in-set-and-top-1 indicator, which is zero on every out-of-set row by
    construction: naming a known author for a stranger is an error at every threshold, and there
    is no rank in the file to argue otherwise.
    """
    weights = np.ones(len(is_ood)) if weights is None else weights
    ordered = weights[order]
    ood = is_ood[order]
    accepted_ood = np.cumsum(ordered * ood)
    hits = np.cumsum(ordered * identified[order])
    total_ood, total_in_set = accepted_ood[-1], (ordered * ~ood).sum()
    if total_ood <= 0 or total_in_set <= 0:
        return np.full(len(DETECTION_GRID), np.nan)
    return np.interp(DETECTION_GRID,
                     np.concatenate([[0.0], accepted_ood / total_ood]),
                     np.concatenate([[0.0], hits / total_in_set]))


def weighted_identification_any(cover: np.ndarray, hit: np.ndarray, is_ood: np.ndarray,
                                weights: np.ndarray | None = None) -> np.ndarray:
    """:func:`weighted_identification` counted per user, at one shared threshold.

    The author level cannot reuse the document helper, for the reason
    :func:`weighted_selective_any` sets out at length: a person is answered and linked at two
    different thresholds. Here it comes out simpler than it does there, because neither axis
    needs both.

    * **FAR** asks only ``cover``, the person's *most* enrolled-looking document. A stranger is
      falsely accepted the moment any one of their documents clears the threshold.
    * **DIR** asks only ``hit``, the confidence of their most confident **correct** document
      (``-inf`` when they have none). ``hit <= cover`` always, so a person whose ``hit`` clears
      the threshold has necessarily been accepted -- which is why this level needs no second
      condition, where the precision denominator of ``the open-set precision-coverage curve`` does.

    Users with no correct document are kept in the **denominator** and out of the running sum,
    via the finite-``hit`` mask: they are in-set people the attack never links, and dropping them
    would divide by the wrong population. That is also why the threshold list is closed with an
    explicit accept-everything point rather than trusting the lowest observed score -- without it
    the right edge would stop at the least confident *stranger* and under-report the top-1.
    """
    weights = np.ones(len(cover)) if weights is None else weights
    total_ood, total_in_set = weights[is_ood].sum(), weights[~is_ood].sum()
    if total_ood <= 0 or total_in_set <= 0:
        return np.full(len(DETECTION_GRID), np.nan)

    # Negated so both lists ascend, which is what `searchsorted` needs: a threshold's count is
    # then "how many of this cohort sit at or above it", read off one prefix sum.
    def running(values: np.ndarray, keep: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        rows = np.flatnonzero(keep)
        order = rows[np.argsort(-values[rows], kind="stable")]
        return -values[order], np.concatenate([[0.0], np.cumsum(weights[order])])

    cover_sorted, cover_running = running(cover, is_ood)
    hit_sorted, hit_running = running(hit, ~is_ood & np.isfinite(hit))
    thresholds = np.unique(np.concatenate([cover_sorted, hit_sorted, [np.inf]]))
    far = cover_running[np.searchsorted(cover_sorted, thresholds, side="right")] / total_ood
    identified = hit_running[np.searchsorted(hit_sorted, thresholds, side="right")] / total_in_set
    return np.interp(DETECTION_GRID, np.concatenate([[0.0], far]),
                     np.concatenate([[0.0], identified]))


def identification_baseline(chance: float) -> np.ndarray:
    """The dashed line at the **document** level: a detector that knows nothing in front of a
    guesser that reads no text.

    A no-information detector accepts each cohort at the same rate, so tolerating ``f`` false
    accepts means keeping ``f`` of the in-set documents too, and the guesser behind it is right
    on ``chance`` of those however many it is shown. The baseline is therefore the straight line
    ``f * chance`` -- through the origin, with the guesser's top-1 as its slope.

    **The author level is not this line scaled**, and must not be drawn with this function: "at
    least once" is not linear in the accept rate, so the composition happens inside the power.
    See :meth:`ProportionalBaseline.identities_at_rate`.

    It is tiny, which is the point: the same near-flat grey line under every curve is what says
    this y axis has no floor to speak of, unlike the detection figure's diagonal.
    """
    return DETECTION_GRID * chance


def config_identification(table: pd.DataFrame, weights: PanelWeights,
                          baseline: ProportionalBaseline | None) -> ConfigIdentification:
    """Document-level DIR-FAR curve plus bootstrap band, over the whole unfiltered test set.

    Both cohorts are in one table on purpose -- the out-of-set documents set the x axis and the
    in-set ones the y -- which is where the two populations share a panel. They are still never mixed *within* an axis.
    """
    is_ood = ~table["author_in_known"].to_numpy(dtype=bool)
    score = table["accept_score"].to_numpy(dtype=float)
    identified = np.where(is_ood, False,
                          np.ceil(table["true_author_rank"].to_numpy(dtype=float)) <= 1
                          ).astype(float)
    order = np.argsort(score, kind="stable")     # most enrolled-looking first: accept a prefix
    curve = weighted_identification(is_ood, identified, order)
    low, high = bootstrap_band(
        lambda row: weighted_identification(is_ood, identified, order, row), weights.documents)
    pools = table.loc[~is_ood, "n_candidate_authors"].to_numpy(dtype=float)
    uniform = float(chance_cmc(pools, np.array([1]))[0]) if len(pools) else float("nan")
    chance = (float(baseline.for_documents(np.array([1.0]))[0]) if baseline else uniform)
    frame = pd.DataFrame({"false_accept_rate": DETECTION_GRID, "identification_rate": curve,
                          "random": identification_baseline(uniform),
                          "random_proportional": (identification_baseline(chance)
                                                  if baseline else np.nan),
                          "ci_low": low, "ci_high": high})
    return ConfigIdentification(curve=frame, top1=float(curve[-1]), chance=chance,
                                n_documents=len(table),
                                n_users=int(table["true_author"].nunique()),
                                n_ood=int(is_ood.sum()))


def config_identification_authors(table: pd.DataFrame, people: pd.DataFrame,
                                  weights: PanelWeights,
                                  baseline: ProportionalBaseline | None) -> ConfigIdentification:
    """:func:`config_identification` counted per user, via :func:`weighted_identification_any`.

    "Was this person linked at all, and did they survive the filter" -- the reading this project
    generally takes as the harm. ``people`` is :func:`author_table`'s collapse of ``table``,
    passed in rather than rebuilt because the open-set loop already has it.
    """
    is_ood = ~people["author_in_known"].to_numpy(dtype=bool)
    cover = -people["accept_score"].to_numpy(dtype=float)
    hit = people["hit_score"].to_numpy(dtype=float)
    curve = weighted_identification_any(cover, hit, is_ood)
    low, high = bootstrap_band(
        lambda row: weighted_identification_any(cover, hit, is_ood, row), weights.documents)
    # Both baselines compose the accept rate *inside* the "at least once", which is what
    # `identities_at_rate` exists for and what `identification_baseline` deliberately cannot do.
    counts = people.loc[~is_ood, "n_documents"].to_numpy(dtype=float)
    n_candidates = int(people["n_candidate_authors"].max())
    uniform = (np.mean(1.0 - (1.0 - DETECTION_GRID[None, :] / n_candidates)
                       ** counts[:, None], axis=0) if len(counts)
               else np.full(len(DETECTION_GRID), np.nan))
    proportional = baseline.identities_at_rate(DETECTION_GRID) if baseline else None
    chance = float((proportional if proportional is not None else uniform)[-1])
    frame = pd.DataFrame({"false_accept_rate": DETECTION_GRID, "identification_rate": curve,
                          "random": uniform,
                          "random_proportional": (proportional if proportional is not None
                                                  else np.nan),
                          "ci_low": low, "ci_high": high})
    return ConfigIdentification(curve=frame, top1=float(curve[-1]), chance=chance,
                                n_documents=len(table), n_users=len(people),
                                n_ood=int(is_ood.sum()), level="author")


# --- the same surface at a pinned threshold: DIR against k ------------------------------------
#
# `openset/dirfar/` fixes k = 1 and sweeps the threshold; this fixes the threshold and
# sweeps k. Together they are two orthogonal slices of DIR(tau, k), and this one is the slice the
# rest of the project already knows how to read: it is the closed-set CMC curve with the
# rejection filter applied, drawn by `draw_cmc_panel` with no new geometry at all.
#
# What it answers that neither neighbour can: a CMC curve says how much an attacker gains from
# being allowed more guesses, and an attacker in the open world does not get to make those
# guesses on documents it has already refused. The gain shrinks under rejection -- rejection keeps
# the documents the attack was already confident *and* right about, so extra candidate slots buy
# less than they do unfiltered.


#: The false-accept budget ``openset/top_k_cmc/`` pins, deliberately different from
#: :data:`IDENTIFICATION_NOTE_FAR`; the k = 1 point equals the DIR-FAR curve read at this FAR, not
#: at the FAR its note prints.
OPENSET_ACCURACY_FAR = 0.10


def threshold_at_far(scores: np.ndarray, is_ood: np.ndarray, far: float) -> tuple[float, float]:
    """``(threshold, achieved FAR)``: the most permissive accept threshold within a FAR budget.

    Accept a document when ``accept_score <= threshold``. Walking the scores from most
    enrolled-looking upward, the out-of-set count is non-decreasing, so "the longest prefix whose
    out-of-set count stays within ``floor(far * n_ood)``" is a threshold and is the largest one
    that honours the budget.

    **Exact rather than interpolated, and that is the difference from the curve family.**
    :func:`weighted_identification` interpolates between attained operating points to land on
    :data:`DETECTION_GRID`; here the operating point is a real one the attacker could choose, so
    the achieved FAR is returned alongside and is what the baseline is composed with. The two
    agree closely on a corpus with plenty of out-of-set documents and can differ where the budget
    is only a handful -- see :data:`IDENTIFICATION_NOTE_FAR`.

    ``-inf`` when the budget does not stretch to a single document, i.e. the most enrolled-looking
    document in the whole test set is a stranger's and nothing can be accepted.
    """
    order = np.argsort(scores, kind="stable")
    seen = np.cumsum(is_ood[order])
    budget = np.floor(far * seen[-1]) if seen[-1] else 0.0
    keep = seen <= budget
    if not keep.any():
        return float("-inf"), 0.0
    last = int(np.flatnonzero(keep)[-1])
    return float(scores[order[last]]), float(seen[last] / seen[-1]) if seen[-1] else 0.0


def rejected_rank(ranks: np.ndarray, accepted: np.ndarray, ks: np.ndarray) -> np.ndarray:
    """``ranks`` with every rejected row pushed past the end of the k axis.

    :func:`weighted_cmc` buckets ``ceil(rank)`` into a histogram of length ``ks[-1] + 2``, so a
    sentinel of ``ks[-1] + 1`` lands in a bin the cumulative sum never reads. That makes "refused"
    and "ranked worse than any k" the same thing to the curve, which is the intended reading: a
    document the attacker threw away is not identified at any k.

    A finite sentinel rather than ``inf`` because the bucketing is an integer cast.
    """
    return np.where(accepted, ranks, float(ks[-1] + 1))


def config_identification_cmc(table: pd.DataFrame, weights: PanelWeights,
                              baseline: ProportionalBaseline | None,
                              far: float = OPENSET_ACCURACY_FAR) -> ConfigCmc:
    """Document-level CMC over the in-set documents that survive a ``far``-budget threshold.

    The denominator is **every** in-set document, not the accepted ones: a document the attacker
    refused is a re-identification it did not make, so it counts as a miss rather than leaving the
    population. That is what makes this curve's k = 1 point the DIR the neighbouring family
    reports, and what stops a stricter threshold from flattering the attack by quietly shrinking
    the question.
    """
    is_ood = ~table["author_in_known"].to_numpy(dtype=bool)
    threshold, achieved = threshold_at_far(table["accept_score"].to_numpy(dtype=float),
                                           is_ood, far)
    in_set = table.loc[~is_ood]
    accepted = in_set["accept_score"].to_numpy(dtype=float) <= threshold
    pools = in_set["n_candidate_authors"].to_numpy(dtype=float)
    ks = np.arange(1, int(pools.max()) + 1)
    ranks = rejected_rank(in_set["true_author_rank"].to_numpy(dtype=float), accepted, ks)
    row_weights = weights.documents[:, (~is_ood).nonzero()[0]] if len(weights.documents) else \
        weights.documents
    accuracy = weighted_cmc(ranks, pools, ks)
    low, high = (band_from_replicates(cmc_replicates(ranks, ks, row_weights))
                 if len(row_weights) else (accuracy, accuracy))
    curve = pd.DataFrame({"k": ks, "accuracy": accuracy,
                          "n_accepted": int(accepted.sum()),
                          "false_accept_rate": achieved,
                          "random": achieved * chance_cmc(pools, ks),
                          "random_proportional": (achieved * baseline.for_documents(ks)
                                                  if baseline else np.nan),
                          "ci_low": low, "ci_high": high})
    return ConfigCmc(curve=curve, n_documents=len(in_set),
                     n_users=int(in_set["true_author"].nunique()),
                     n_candidates=int(pools.max()), max_k=int(ks[-1]), level="document",
                     n_ood=int(is_ood.sum()), note_corner="upper left")


def config_identification_cmc_authors(table: pd.DataFrame, people: pd.DataFrame,
                                      weights: PanelWeights,
                                      baseline: ProportionalBaseline | None,
                                      far: float = OPENSET_ACCURACY_FAR) -> ConfigCmc:
    """:func:`config_identification_cmc` counted per user, at the **author** level's own threshold.

    Two things differ from the document level, and both follow ``openset/dirfar/author``
    rather than being choices made here.

    * **The budget is spent on out-of-set *users*.** A stranger is falsely accepted the moment any
      one of their documents is, so a threshold leaking a fixed share of stranger documents leaks
      a far larger share of stranger people, the more documents each writes. Pinning the *author*
      FAR at ``far`` therefore picks a stricter threshold than the document panel's, which is why
      the two levels are not the same operating point and must not be read against each other at
      a fixed k.
    * **A user's rank is the best one among their *accepted* documents**, so the curve asks "was
      this person linked within k by something that survived the filter". Their best rank overall
      is not enough: it may belong to a document the attacker refused.
    """
    is_ood_user = ~people["author_in_known"].to_numpy(dtype=bool)
    threshold, achieved = threshold_at_far(people["accept_score"].to_numpy(dtype=float),
                                           is_ood_user, far)
    in_set = table.loc[table["author_in_known"].astype(bool)]
    accepted = in_set["accept_score"].to_numpy(dtype=float) <= threshold
    n_candidates = int(in_set["n_candidate_authors"].max())
    ks = np.arange(1, n_candidates + 1)
    ranks = rejected_rank(in_set["true_author_rank"].to_numpy(dtype=float), accepted, ks)
    best = pd.Series(ranks, index=in_set["true_author"].to_numpy()).groupby(level=0).min()
    pools = np.full(len(best), float(n_candidates))
    accuracy = weighted_cmc(best.to_numpy(dtype=float), pools, ks)
    author_weights = weights.bootstrap.multiplicities(best.index)
    low, high = (band_from_replicates(cmc_replicates(best.to_numpy(dtype=float), ks,
                                                     author_weights))
                 if len(author_weights) else (accuracy, accuracy))
    counts = in_set.groupby("true_author").size().reindex(best.index).to_numpy(dtype=float)
    curve = pd.DataFrame({"k": ks, "accuracy": accuracy,
                          "n_accepted": int(accepted.sum()),
                          "false_accept_rate": achieved,
                          "random": np.mean(
                              1.0 - (1.0 - achieved * np.minimum(ks, n_candidates)[None, :]
                                     / n_candidates) ** counts[:, None], axis=0),
                          "random_proportional": (
                              baseline.identities_for_k_at_rate(achieved, ks)
                              if baseline else np.nan),
                          "ci_low": low, "ci_high": high})
    return ConfigCmc(curve=curve, n_documents=len(in_set), n_users=len(best),
                     n_candidates=n_candidates, max_k=int(ks[-1]), level="identity",
                     n_ood=int(is_ood_user.sum()), note_corner="upper left")


# --- the same operating point, split by how long the target conversation is -------------------
#
# `openset/dirfar/` sweeps the threshold over the whole test set; this pins it at one
# operating point and asks which conversations survive it, binned by the target conversation's
# length in words. The threshold is chosen ONCE over every test document (or user) and then read
# within each bin -- it is the attacker's single operating point, not one re-fitted per length,
# which would hand short conversations a looser filter than the attacker could actually use.
# It is observational: a long conversation is a different conversation,
# usually by a different kind of user, not a short one given more words.

#: The false-accept budget this family pins -- the same as ``openset/top_k_cmc/``
#: (:data:`OPENSET_ACCURACY_FAR`) rather than the identification note's tighter
#: (:data:`IDENTIFICATION_NOTE_FAR`), because split across bins a tighter budget pushes a bin's
#: DIR close enough to zero that the length trend is hard to see.
WORDS_IDENTIFICATION_FAR = 0.10

#: Left edge of each word-count bin, at round numbers chosen off both corpora's distributions so
#: no bin is starved, with the open `1000+` bin catching the long pasted-document tail.
#: Gemini reads only a prefix of each document, so the top bins are
#: longer than the feature sees -- a flat tail there is truncation, not a ceiling on what
#: length can buy.
WORD_BIN_EDGES = (1, 20, 50, 200, 1000)

WORD_BIN_LABELS = count_bin_labels(WORD_BIN_EDGES)


@dataclass
class ConfigWordsIdentification:
    """One (run, configuration) DIR at a pinned FAR, per word-count bin of the target.

    ``curve`` has the columns :func:`draw_binned_panel` reads:
    ``bin``/``bin_label``, ``accuracy`` (here the DIR) with ``ci_low``/``ci_high``, both
    baselines, ``n_authors``/``n_documents`` per bin, and ``share`` for the bars. Only in-set
    rows are binned -- DIR's denominator is in-set by definition -- while the out-of-set cohort
    has already done its one job, fixing the threshold.

    ``far`` is the **achieved** false-accept rate (the most permissive attainable threshold
    within the budget, :func:`threshold_at_far`), counted in the level's unit: out-of-set
    documents at ``doc/``, out-of-set users at ``author/`` -- the same pair of operating points
    ``openset/top_k_cmc/`` uses, and for the same reason not comparable across levels.
    """

    curve: pd.DataFrame
    n_documents: int
    n_users: int
    n_ood: int
    far: float
    level: str = "document"
    #: The known pool in the level's unit, printed as the note's ``cand.`` line exactly as
    #: ``openset/dirfar`` prints it; set by :func:`run_curves`, ``None`` leaves it off.
    n_candidates: int | None = None


def config_identification_by_words(table: pd.DataFrame, weights: PanelWeights, dataset: str,
                                   known_config: str, level: str = "document",
                                   far: float = WORDS_IDENTIFICATION_FAR
                                   ) -> ConfigWordsIdentification | None:
    """DIR at a ``far`` budget, binned by the target conversation's word count.

    ``table`` is the **unfiltered** open-set table. A hit is an in-set document that is accepted
    (``accept_score <= threshold``) and ranks its true author first; a refused document is a miss,
    exactly as in :func:`config_identification_cmc`, so each bin's denominator is every in-set
    document in it and the population-weighted mean over bins is ``openset/top_k_cmc/``'s k=1
    point at this budget.

    * **Document level** -- the threshold spends the budget on out-of-set documents; a bin's DIR
      is the share of its in-set documents identified.
    * **Author level** -- the threshold spends it on out-of-set *users* (a stranger is falsely
      accepted when any one of their documents is), as ``openset/top_k_cmc/author`` does. **Each
      user sits in exactly one bin**, chosen by the **mean word count of their in-set (test-side)
      conversations** -- the ones under attack -- on the same :data:`WORD_BIN_EDGES` as the
      document level, and counts as linked if **any** of those conversations was accepted and
      ranked first. That is ``openset/top_k_cmc/author``'s k=1 event, so the bins partition the
      users and their user-weighted mean reproduces that point at this budget. The axis is then a
      property of the *person* -- how much a typical conversation of theirs says -- not of one
      conversation.

    The baselines are a chance detector at the same achieved FAR in front of the proportional
    guesser: ``far * p_a`` per document, and ``1 - (1 - far * p_a) ** m`` per user with ``m``
    their in-set conversations -- the composition inside the power that
    :meth:`ProportionalBaseline.identities_at_rate` explains.

    ``None`` without the corpus parquet, which is the only source of the word counts.
    """
    words = document_word_counts(dataset)
    if words is None:
        return None
    is_ood = ~table["author_in_known"].to_numpy(dtype=bool)
    scores = table["accept_score"].to_numpy(dtype=float)
    if level == "author":
        people = author_table(table)
        is_ood_unit = ~people["author_in_known"].to_numpy(dtype=bool)
        threshold, achieved = threshold_at_far(people["accept_score"].to_numpy(dtype=float),
                                               is_ood_unit, far)
    else:
        is_ood_unit = is_ood
        threshold, achieved = threshold_at_far(scores, is_ood, far)

    rows = np.flatnonzero(~is_ood)
    in_set = table.iloc[rows]
    lengths = in_set["doc_id"].map(words)
    # Every in-set document is in the corpus the run was built from; a missing one means the
    # parquet under DATA_DIR is not that corpus, and it is dropped rather than binned as NaN.
    keep = lengths.notna().to_numpy()
    if not keep.any():
        return None
    rows, in_set, lengths = rows[keep], in_set[keep], lengths[keep]
    n_bins = len(WORD_BIN_EDGES)
    # A zero-word conversation (whitespace only) is clipped into the first bin rather than
    # indexed at -1. There are none in either corpus today.
    document_bins = np.clip(np.searchsorted(WORD_BIN_EDGES, lengths.to_numpy(dtype=float),
                                            side="right") - 1, 0, None)
    hit = ((in_set["accept_score"].to_numpy(dtype=float) <= threshold)
           & (np.ceil(in_set["true_author_rank"].to_numpy(dtype=float)) <= 1))
    shares = known_author_shares(dataset, known_config)
    pool = in_set["n_candidate_authors"].to_numpy(dtype=float)
    prior = (in_set["true_author"].map(shares).to_numpy(dtype=float) if shares is not None
             else 1.0 / pool)

    documents = pd.DataFrame({"author": in_set["true_author"].to_numpy(), "bin": document_bins,
                              "hit": hit, "prior": prior, "pool": pool,
                              "words": lengths.to_numpy(dtype=float)})
    n_users = int(documents["author"].nunique())

    if level == "author":
        # One row per user, binned by the mean length of their in-set conversations.
        people = documents.groupby("author", sort=True).agg(
            hit=("hit", "any"), prior=("prior", "first"), pool=("pool", "max"),
            m=("hit", "size"), words=("words", "mean")).reset_index()
        bins = np.clip(np.searchsorted(WORD_BIN_EDGES, people["words"].to_numpy(dtype=float),
                                       side="right") - 1, 0, None)
        chances = people["m"].to_numpy(dtype=float)
        n_authors = np.bincount(bins, minlength=n_bins).astype(float)
        # A bin's documents are its users' documents, wherever each one's own length falls.
        n_documents = np.bincount(bins, weights=chances, minlength=n_bins)
        unit_hit = people["hit"].to_numpy()
        unit_weights = weights.for_authors(people["author"])
        proportional = 1.0 - (1.0 - achieved * people["prior"].to_numpy(dtype=float)) ** chances
        uniform = 1.0 - (1.0 - achieved / people["pool"].to_numpy(dtype=float)) ** chances
        # The bins partition the users, so the bar is each bin's share of them.
        share = n_authors / n_authors.sum()
    else:
        # A bin's user count gates it (MIN_AUTHORS_PER_BIN); here a user is counted in every
        # bin they have a document in, since the unit is the document.
        n_authors = (documents.groupby("bin")["author"].nunique()
                     .reindex(range(n_bins), fill_value=0).to_numpy(dtype=float))
        n_documents = np.bincount(document_bins, minlength=n_bins).astype(float)
        bins, unit_hit = document_bins, hit
        unit_weights = (weights.documents[:, rows] if len(weights.documents)
                        else weights.documents)
        proportional, uniform = achieved * prior, achieved / pool
        share = n_documents / n_documents.sum()

    accuracy, low, high = binned_rate(bins, unit_hit, unit_weights, n_bins)
    thin = n_authors < MIN_AUTHORS_PER_BIN
    accuracy, low, high = (np.where(thin, np.nan, value) for value in (accuracy, low, high))
    curve = pd.DataFrame({
        "bin": np.arange(n_bins), "bin_label": list(WORD_BIN_LABELS),
        "accuracy": accuracy,
        "random": bin_means(bins, uniform, n_bins),
        "random_proportional": bin_means(bins, proportional, n_bins),
        "ci_low": low, "ci_high": high,
        "n_authors": n_authors.astype(int), "n_documents": n_documents.astype(int),
        "share": share, "false_accept_rate": achieved,
    })
    return ConfigWordsIdentification(curve=curve, n_documents=int(len(in_set)), n_users=n_users,
                                     n_ood=int(is_ood_unit.sum()), far=achieved, level=level)


# --- the comparison figures: one panel per known configuration ----------------

@dataclass
class Series:
    """One line inside one panel: a label, the colour slot its entity owns, and its curve.

    ``curve`` is whichever per-configuration curve object the panel drawer expects -- a
    :class:`ConfigCmc` for :func:`draw_cmc_panel`, a :class:`ConfigIdentification` for
    :func:`draw_identification_panel` -- which is what lets one grouping feed every curve type.
    """

    label: str
    slot: int
    curve: object
    #: An explicit dash pattern. Set by the ``by_attack`` figures, where the dash carries the
    #: **feature** and several series share one hue on purpose. ``None`` is solid.
    dashes: tuple | None = None
    #: What that dash says, spelled out for the companion CSV. Without it two lines that share a
    #: colour legend entry -- one attack's two features -- would write the same ``series`` name on
    #: two different curves.
    dash_label: str | None = None

    @property
    def dash(self) -> tuple:
        return self.dashes if self.dashes is not None else ()

    @property
    def column(self) -> str:
        """Name for this series in a companion CSV, unique even when labels are shared."""
        if self.dash_label:
            return f"{self.dash_label} · {self.label}"
        return self.label


def grid_config(size: float, gap: float) -> KnownConfig | None:
    """The configuration at one cell of the facet grid, or ``None`` where none can exist."""
    start = TEST_START - gap - size
    return None if start < -1e-9 else KnownConfig(start=round(start, 4),
                                                  end=round(TEST_START - gap, 4))


def config_axes_grid(figsize=GRID_FIGSIZE):
    """The 3x3 triangular facet grid: rows are known-side *size*, columns are *staleness*.

    Reading across a row varies only how stale the attacker's data is; reading down a column
    varies only how much of it there is. That separation is the whole point of the experiment
    design, so it is the figure's geometry rather than something a caption has to explain.

    **The canvas is square** (:data:`GRID_FIGSIZE`): the width is what a page gives a figure, so
    the height is the free axis, and squaring it makes each cell taller rather than changing the
    grid's 3 x 3 shape -- the panel headings and the x titles are a fixed physical size, so the
    panels take all of the extra 3 in. A cell taller than it is wide is the shape a curve read
    against quarter gridlines wants, and every family here is drawn at :data:`GRID_SCALE`, where
    the text needs the room.

    The three impossible cells (a known side that is both large and far from the test set would
    have to begin before the corpus does) are hidden, leaving the upper-left triangle. Returns
    the figure, the axes array, a ``tag -> axes`` map for the cells that exist, and the cell the
    legend should be drawn in -- the bottom-right one, which the triangle can never fill, so the
    legend costs no figure height and the panels get the space instead. Its *axis* is turned off
    rather than the axes hidden, because an invisible axes is skipped by ``tight_layout`` and a
    legend anchored to a collapsed cell lands in the wrong place. ``None`` if the grid has no
    empty cell (it would not be triangular), and :func:`finish_facets` falls back to a strip
    under the figure.
    """
    figure, grid = plt.subplots(len(GRID_SIZES), len(GRID_GAPS), figsize=figsize,
                                squeeze=False, sharex=True, sharey=True)
    figure.patch.set_facecolor(SURFACE)
    axes_for: dict[str, object] = {}
    empty = []
    for row, size in enumerate(GRID_SIZES):
        for column, gap in enumerate(GRID_GAPS):
            config = grid_config(size, gap)
            if config is None:
                grid[row][column].set_visible(False)
                empty.append(grid[row][column])
                continue
            axes_for[config.tag] = grid[row][column]
    legend_cell = empty[-1] if empty else None
    if legend_cell is not None:
        legend_cell.set_visible(True)
        legend_cell.set_axis_off()
    return figure, grid, axes_for, legend_cell


def label_facets(grid, axes_for: dict, xlabel: str, ylabel: str, scale: float = 1.0) -> None:
    """Head every panel with its known interval, and put the axis titles on the grid's outer edge.

    **Each cell is headed ``[50%, 75%] known``** -- the slice of the corpus's timeline the
    attacker was given, left-aligned above the panel. It replaces the
    pair of coordinates the grid used to carry: a ``2 quarters stale`` column header over the top
    row, and a ``known 50%`` row label in the slot the y title now occupies. Between them those
    two did name a cell, but only in combination and only for a reader who had found both edges
    of a *triangular* grid, where the outer edge is not the last row. The interval names the cell
    outright, in the units the experiment is specified in -- a run's ``--known-windows XXYY`` tag
    *is* these two numbers -- and it survives a panel being cropped out of the figure on its own,
    which a header two rows up does not. Nothing is lost: staleness is the distance from the interval's end to the test quarter at
    75%, size is its width, and both are still read across a row and down a column.

    The x *title* stays on the outer edge only -- the lowest **visible** axes of each column,
    which is row one for the 75%-size column and row two for the 50% one. It is a sentence, and
    six copies of it inside the grid would be six lines of repeated caption text. The y title is
    the first column's alone for the same reason, in the slot the row label has left free.

    **Every panel carries its own x tick labels**, which is not what ``sharex`` does on its own --
    it labels the bottom row and leaves the rest bare. The axis is shared, so the numbers are the
    same in every cell, but a reader comparing the top-right panel against the bottom-left one
    should not have to trace a column down two cells to find out what the x position under a
    curve is.

    **Y tick labels are the first column's**, which is what ``sharey`` does by default and the
    opposite of the x rule above. The asymmetry is a space argument, not a reading one: a y tick
    label sits *outside* its panel and pushes the next column along, so repeating it in every
    column would cost real width for numbers the shared axis already makes identical across the
    row. An x tick label costs height the row below has already reserved.
    """
    for column, gap in enumerate(GRID_GAPS):
        rows = [row for row, size in enumerate(GRID_SIZES) if grid_config(size, gap) is not None]
        if not rows:
            continue
        for row in rows:
            config = grid_config(GRID_SIZES[row], gap)
            axes = grid[row][column]
            style_axes(axes,
                       xlabel if row == rows[-1] else "",
                       ylabel if column == 0 else "",
                       f"[{config.start:.0%}, {config.end:.0%}] known",
                       # A cell coordinate, not a heading over one: the same size and weight as
                       # the axis titles it sits among, which is what the column headers took
                       # when they were what said which cell this is. Tight against the panel
                       # (4 pt rather than the standalone heading's 10) because it belongs to
                       # that one cell -- in a row of three, a heading floating midway between
                       # two panels reads as belonging to neither.
                       title_size=FONT_AXIS_LABEL, title_weight="normal", title_pad=4.0,
                       scale=scale)
            axes.tick_params(labelbottom=True, labelleft=(column == 0))
            # **A rotated x label's alignment has to be set here, after that call**, and it is
            # `center` on every panel of every family. `tick_params` re-applies the axis's stored
            # tick-label kwargs, which resets `ha` to its default -- so an alignment set where the
            # rotation is (a drawer's own `set_xticklabels`, which runs before this)
            # survived on exactly one panel: the grid's bottom row, the one axes `sharex` leaves
            # labelled and which this call therefore does not touch. Its numbers sat visibly out
            # of line with the identical numbers in the rows above it.
            #
            # **Centred rather than anchored at the tick**, which is the usual pairing for an
            # angled label (`ha="left"` for a clockwise rotation, `"right"` for the other way).
            # An anchored label hangs its whole width off its tick, so the `1.00` at the right-
            # hand end of the axis runs past the panel -- and `tight_layout` pays for that
            # overhang out of every panel's width, on a figure where the panel is the scarce
            # thing. Centring spends half a label at each end instead and buys the area back.
            for label in axes.get_xticklabels():
                label.set_ha("center")
            if column == 0:
                # Set at 45 degrees. Upright (90) was tried first and packed the numbers too
                # tightly: rotating a label puts its *width* on the y axis instead of its height,
                # which is what buys the gap between quarter ticks back.
                #
                # `va="center"` centres each label's box on the gridline it names. It was `top`,
                # which hangs the box *below* the tick -- a rotated label is as tall as the string
                # is long, so that dropped every number visibly under the line it belongs to.
                # Vertical centring is what matplotlib does for an unrotated y label, so this
                # keeps the rotation a change of angle only.
                axes.tick_params(axis="y", labelrotation=45)
                for label in axes.get_yticklabels():
                    label.set_va("center")


def quarter_ticks(axis, full: float = 1.0) -> None:
    """Tick one axis of a 0-to-``full`` share at **every quarter**: 0, 25%, 50%, 75%, 100%.

    **Every axis in this file that carries a share goes through here** -- both axes of a coverage
    or ROC panel, the percentile axes, the clustering figures' precision and recall, the bar
    charts' accuracy. Without it a panel whose x is also a share can tick differently on each axis,
    which is two different griddings of the same unit on one panel.

    The positions are pinned rather than left to matplotlib. Its automatic locator picks a tick
    count from the axes' *size*, so the same family ticks differently on a tall figure and a short
    one, where a quarter is the unit these numbers are actually discussed in ("half the
    documents", "a quarter of the users"). Pinning it makes every panel of every family tick
    identically whatever the figure height, which is what lets two figures be flipped between.

    ``full`` is 1.0 for a fraction and 100 for a share written as a percentage (the two percentile
    axes), so both spellings land on the same five gridlines. An axis that does not span the whole
    unit -- BCubed F reaches ~0.6, and its frame stops above the tallest bar -- simply gets the
    quarters below its top, which is the point: the gridlines mean the same thing on every figure
    here rather than being fitted to each one's range.
    """
    axis.set_major_locator(MultipleLocator(full / 4))


def unit_y_axis(axes) -> None:
    """A share-of-something y axis: 0 to 1, ticked at every quarter by :func:`quarter_ticks`.

    The 1.02 top keeps a curve that reaches exactly 1.0 off the frame, and the locator puts no
    tick there.
    """
    axes.set_ylim(0, 1.02)
    quarter_ticks(axes.yaxis)


#: Panels that do **not** take the unit square, as ``(curve type, dataset) -> (top, tick step)``.
#:
#: **The second deliberate exception to :func:`quarter_ticks`** (the first is the clustering
#: variants dumbbell, whose x is a narrow zoom). The rule everywhere else is that an axis which
#: does not span the unit still gets the quarters below its top, so one gridline means one thing
#: across the project. It is given up here because ``openset/dirfar`` is the one family
#: whose y ceiling is *the attack's own top-1*, which on WildChat leaves most of the panel over
#: empty space. The zoom keeps the spirit -- **five evenly spaced gridlines, still starting at
#: 0** -- and changes only what one of them is worth, which the axis labels say. swe-chat is
#: deliberately absent: its curves reach high enough that the unit square fits them.
#:
#: **Read the two corpora's panels as different scales.** This is the only place in the file
#: where the same family's y axis differs between datasets, so a WildChat panel and a swe-chat
#: one cannot be compared by eye the way two panels of one grid can. The numbers on the axis are
#: what distinguishes them.
PANEL_Y_LIMITS = {
    ("identification", "wildchat"): (0.4, 0.1),
    ("identification_authors", "wildchat"): (0.4, 0.1),
    # The same zoom as `openset/dirfar`, whose layout this family adopts. Check `ci_high`
    # under `--bands` before trusting the ceiling not to clip a band.
    ("identification_words", "wildchat"): (0.4, 0.1),
    ("identification_words_authors", "wildchat"): (0.4, 0.1),
    # Same reasoning one family over, and the ceiling is set by the *baseline* at the author
    # level rather than by the curve -- tight enough that it is worth checking the widest
    # `ci_high` under `--bands` before trusting it.
    ("identification_cmc", "wildchat"): (0.4, 0.1),
    ("identification_cmc_authors", "wildchat"): (0.4, 0.1),
}


def zoom_y_axis(axes, top: float, step: float) -> None:
    """Replace a panel's unit y axis with a ``[0, top]`` one ticked every ``step``.

    Applied *after* the drawer, at :func:`plot_config_comparison`'s single dispatch point, for
    the same reason :data:`GRID_SCALE` is spent there: every grid figure goes through it, so no
    drawer needs to know which corpus it is on or grow a parameter it would ignore. The drawer's
    own :func:`unit_y_axis` call runs first and is simply overridden.

    The top is exact rather than :func:`unit_y_axis`'s 1.02, because the top tick is meant to be
    a labelled gridline at the frame rather than headroom above the highest one. **A band that
    exceeds it is therefore clipped** -- check the family's widest ``ci_high`` before adding an
    entry to :data:`PANEL_Y_LIMITS`.
    """
    axes.set_ylim(0, top)
    axes.yaxis.set_major_locator(MultipleLocator(step))


def log_x_axis(axes, decades_per_tick: int = 1) -> None:
    """A log x axis whose ticks read ``1, 10, 100`` rather than ``10^0, 10^1, 10^2``.

    Every log axis in the file goes through here. Matplotlib's default ``LogFormatterSciNotation``
    is right for an axis spanning many orders of magnitude; these span few and count **candidate
    authors**, which a reader compares against the pool size in the panel note, and a count does
    not compare against a power of ten without doing arithmetic first.

    Thousands are separated for the same reason: a grouped number is read at a glance where a bare
    one is counted. Minor ticks stay unlabelled, as matplotlib leaves them.

    ``decades_per_tick`` thins the labels to every second (or n-th) power of ten, which every
    log axis on the grid needs at :data:`GRID_SCALE`: labelling every decade there runs the
    adjacent numbers into one another. It is the *labels* that are thinned and not the axis --
    the minor ticks and the gridlines behind them are untouched, so a curve is still read against
    every decade.
    """
    axes.set_xscale("log")
    if decades_per_tick != 1:
        axes.xaxis.set_major_locator(LogLocator(base=10.0 ** decades_per_tick))
    axes.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.10g}"))


#: Smallest FAR :func:`log_far_axis` shows. It is exactly ``DETECTION_GRID``'s own step, not an
#: arbitrary floor: a smaller one would put empty decades under the axis, which drew as a straight
#: line dressed up as a measured curve. FAR=0 -- the strictest threshold, reject every stranger --
#: has no image on a log axis and is dropped rather than clipped onto this floor: clipping put its
#: value on top of the real floor sample's x position and the two almost never agree, which drew a
#: near-vertical jump at the left edge that read as a rendering bug.
FAR_LOG_FLOOR = 0.01
#: Five ticks spanning [FAR_LOG_FLOOR, 1.0], chosen less compressed at the low end than plain
#: decades and not quite even half-decade spacing either. The log-scale analogue of five linear
#: quarter-ticks.
FAR_LOG_TICKS = (0.01, 0.03, 0.1, 0.33, 1.0)


def log_far_axis(axes) -> None:
    """FAR on a log x axis, for ``openset/dirfar/`` alone: five ticks at
    :data:`FAR_LOG_TICKS`.

    Most of a DIR-FAR curve's shape lives below FAR=0.1 -- see :func:`draw_identification_panel`
    -- which a linear axis spends three-quarters of the panel's width not showing. Unlike
    :func:`log_x_axis`'s counts, this axis is a share and stays unformatted (``0.03`` rather than
    ``0.030`` or a percent sign), matching the bare fractions of every other share axis.
    """
    axes.set_xscale("log")
    axes.set_xlim(FAR_LOG_FLOOR, 1.0)
    axes.xaxis.set_major_locator(FixedLocator(FAR_LOG_TICKS))
    axes.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    axes.tick_params(axis="x", labelrotation=-45)


#: Where a panel note can sit, as corner -> (anchor, inset direction, horizontal, vertical). The
#: inset is small but not zero -- at 0 the glyphs touch the frame and the tight crop shaves their
#: antialiasing (see :data:`FIGURE_PAD_INCHES`).
NOTE_CORNERS = {
    "lower right": ((1, 0), (-1.5, 3), "right", "bottom"),
    "lower left": ((0, 0), (1.5, 3), "left", "bottom"),
    "upper left": ((0, 1), (1.5, -3), "left", "top"),
    "upper right": ((1, 1), (-1.5, -3), "right", "top"),
}


def panel_note(axes, text: str, scale: float = 1.0, corner: str = "lower right") -> None:
    """The in-set counts, printed inside the panel because they are not constant across the grid.

    A larger or fresher known side enrolls more of the test set's users, so it can attempt more
    of the same documents. That *reach* is part of what the known side buys and part of why two
    panels are not directly comparable at face value -- it belongs on the figure, not in a note
    someone has to look up.

    **Bottom right by default**, since most curve families here rise to the right or start high on
    the left, so the upper-left corner is where a line actually goes and the note would sit on the
    data there. Aligned per line as well as as a block, so a multi-line note reads as one object
    against the panel's corner.

    ``corner`` is for the families whose ink is somewhere else: the word-count panels stand a
    population histogram along the bottom, and the DIR-FAR and filtered top-k panels put theirs in
    the upper left. Each of those corners is free by the family's construction or by measurement
    rather than by how one run came out -- see the drawers.
    """
    (anchor, (dx, dy), ha, va) = NOTE_CORNERS[corner]
    axes.annotate(text, xy=anchor, xytext=(dx * scale, dy * scale), xycoords="axes fraction",
                  textcoords="offset points", color=TEXT_MUTED, fontsize=FONT_ANNOTATION * scale,
                  va=va, ha=ha, ma=ha, zorder=5)


# --- NO FIGURE CARRIES A TITLE -----------------------------------------------
#
# `figure_heading` drew a left-aligned bold line across the top of every figure and reserved a
# band in inches for it. It was removed, and with it the last piece of prose burned into an image:
# first the subtitle, now the title.
#
# The reasoning is the subtitle's, and it applies harder to a title: what a figure *is* belongs
# to the document that publishes it, where it can be edited, translated, footnoted and set in the
# publication's own type. Everything those titles said is still recoverable without opening the
# image -- **the path is the identity** (`plots/<dataset>/<family>/<doc|author>/by_defense/
# <feature>_<attack>.pdf` names the corpus, the family, the counting level and the method), the
# panel headings name the configuration, the panel notes carry the counts, and `CURVE_TYPES`'
# fifth field carries the caveats in prose meant for a caption.
#
# One thing was carried ONLY by a title and is now carried only by the path: the clustering
# figures' author-scope clause (`CLUSTERING_SCOPES`' second field, kept as documentation and no
# longer drawn -- `clustering/unseen/` in the path is what distinguishes the two now). It would
# want a `panel_note` rather than a title if it has to be visible again.
#
# The band is reclaimed by the panels, so every figure's axes grow slightly taller than they
# were. Nothing else moves: the legends, the companion CSVs and the paths are untouched.
#
# **Some drawing functions still take a `dataset` they no longer read** -- the title was its only
# use in `plot_defense_comparison`, `plot_attack_comparison`, `plot_clustering_bcubed` and
# `plot_clustering_precision_recall`. It is kept on purpose rather than cleaned up: the plan builds
# every one of their argument tuples the same way, and anything put back on a figure -- a panel
# note naming the corpus, say -- wants it in hand.


def style_legend(legend, scale: float = 1.0) -> None:
    """The shared legend chrome: a title in secondary ink at the body size."""
    legend.get_title().set_color(TEXT_SECONDARY)
    legend.get_title().set_fontsize(FONT_LEGEND * scale)


def finish_facets(figure, handles: dict, legend_title: str,
                  stem: Path, table: pd.DataFrame, legend_cell=None,
                  extra_legend: tuple[str, dict] | None = None,
                  scale: float = 1.0, row_pad: float | None = None) -> Path:
    """Shared chrome for every facet figure: its legend(s) and its companion CSV.

    **No title** -- see the note above on why no figure here carries one. What the figure is, is
    in its path; what its panels are, is in their headings.

    The legend goes inside ``legend_cell`` -- the corner of the triangular grid that holds no
    panel -- so it takes space the figure was giving away rather than a reserved strip that
    shortens every panel. It is drawn on that axes rather than as a figure legend so it moves
    with the cell under ``tight_layout``; a long one is free to overflow, because the cells above
    and to its left are the grid's other two holes.

    Without a ``legend_cell`` the legend needs a reserved strip under the axes instead, and its
    size is worked out **in inches and then divided by the figure's height**: a legend is a fixed
    physical size, so a fraction tuned on the 9-inch facet grid leaves a short figure's legend
    sitting on its x label. Three columns at most for the same reason -- ``bbox_inches="tight"`` grows the canvas around an over-wide
    legend, and four of these labels are wider than the axes they belong to.

    ``extra_legend`` is a second ``(title, handles)`` block for the figures where colour and dash
    carry two different things and a reader needs both to decode one line -- every ``by_attack``
    one (dash = feature).

    **Where the second block goes depends on which layout the figure has**, and both put it where
    there is room rather than where it would be tidy. With a legend cell it continues *below* the
    first, anchored off the drawn artist (:func:`stack_below`) because that one's height grows a
    row per series, and the pair starts at the cell's top so it grows down into space the triangle
    was giving away anyway. Without one -- a legend in a reserved strip already flush with the figure's bottom
    edge -- there is nothing below to stack into, so the two sit **side by side** in that strip, colour on the left and dash on the right. That is also
    when the strip is sized from ``max(rows, len(extra_handles))``: the taller of the two blocks
    decides how much has to be reserved.
    """
    extra_title, extra_handles = extra_legend or ("", {})
    if legend_cell is None:
        # Two columns rather than three once a second block shares the strip, so the pair cannot
        # run wider than the canvas `bbox_inches="tight"` is about to crop to.
        columns = min(max(len(handles), 1), 2 if extra_handles else 3)
        rows = max(-(-len(handles) // columns), len(extra_handles))
        bottom = (0.26 + 0.24 * rows) / figure.get_size_inches()[1]
    else:
        columns, bottom = 1, 0.0
    # The top is the figure's own edge: there is no heading band to leave room for. ``row_pad``
    # is :data:`GRID_ROW_PAD` for the facet grid and unset for a figure with one row, which has
    # nothing to separate.
    figure.tight_layout(rect=(0, bottom, 1, 1),
                        **({} if row_pad is None else {"h_pad": row_pad}))
    shared = dict(ncol=columns, frameon=False, fontsize=FONT_LEGEND * scale,
                  labelcolor=TEXT_PRIMARY)
    if legend_cell is None:
        # Left-anchored only when it shares the strip; centred otherwise, so no existing figure
        # moves. `Figure.legend` appends rather than replacing, unlike the axes call below.
        legend = figure.legend(
            list(handles.values()), list(handles), title=legend_title,
            loc="lower left" if extra_handles else "lower center",
            bbox_to_anchor=(0.04, 0.005) if extra_handles else (0.5, 0.005), **shared)
        if extra_handles:
            style_legend(scale=scale, legend=figure.legend(
                list(extra_handles.values()), list(extra_handles), title=extra_title,
                loc="lower right", bbox_to_anchor=(0.98, 0.005),
                **{**shared, "ncol": 1}))
    else:
        # Centred when it is the only legend, so nothing on the existing figures moves; anchored
        # to the cell's top when a second one has to fit under it.
        legend = legend_cell.legend(list(handles.values()), list(handles), title=legend_title,
                                    loc="upper center" if extra_handles else "center", **shared)
    style_legend(legend, scale)
    if extra_handles and legend_cell is not None:
        # A second `.legend()` call on an axes *replaces* the first; the artist has to be adopted
        # explicitly to keep both. (`Figure.legend` appends instead, which is why the strip layout
        # above needs none of this.)
        legend_cell.add_artist(legend)
        style_legend(scale=scale, legend=legend_cell.legend(
            list(extra_handles.values()), list(extra_handles), title=extra_title,
            loc="upper center", bbox_to_anchor=tuple(stack_below(figure, legend_cell, legend)),
            borderaxespad=0.0,  # honour the measured anchor instead of re-padding off it
            # The default handle length, matching the colour legend above it: a longer sample
            # was needed only for a dash pattern that has since been tightened to fit a default
            # sample, so keeping the two legend blocks' keys the same length now reads as intended.
            **shared))
    stem.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(stem.parent / f"{stem.name}.csv", index=False)
    return save_figure(figure, stem)


def abbreviate_count(value: int) -> str:
    """``427`` -> ``427``; ``14322`` -> ``14.3k``; ``10000`` -> ``10k``.

    Used by every panel note on the configuration grid, where the counts are drawn at
    :data:`GRID_SCALE` and a grouped five-digit number costs real panel width. Thousands are the
    only magnitude these counts reach, so there is one suffix and no ladder.

    Below 1,000 the number is left exactly as it is, separators and all: that is where the small
    user counts live (427, 744), they are short already, and rounding them would throw away a
    digit that costs nothing to print. A trailing ``.0`` is dropped, so a round 10,000 reads
    ``10k`` rather than ``10.0k``.

    **Only the note is abbreviated.** The companion CSV beside every figure carries the exact
    counts, so nothing here is the only record of a number.
    """
    if value < 1000:
        return f"{value:,}"
    return f"{value / 1000:.1f}".removesuffix(".0") + "k"


def draw_cmc_panel(axes, series: list[Series], handles: dict,
                   scale: float = 1.0) -> pd.DataFrame:
    """One cell's CMC curves: top-k accuracy against k, log x, with bootstrap bands.

    k is on a log axis because the informative part of a CMC curve is its first decade: an
    attacker who has narrowed 20,000 users to 10 has already won, and what happens at k = 5,000
    is noise about the tail. Series are truncated to the shortest one's k range so every line in
    the panel spans the same axis.

    The baseline is drawn once per panel, in neutral grey and the figure's only dash: it depends
    on the known side's authors and how much each wrote, which is a property of the configuration
    rather than of the defense or feature, so every series in a cell shares it. It is the
    **proportional** guesser (:func:`prior_inclusion`) where the corpus parquet is on disk to
    supply the prior, and falls back to uniform ``1/N`` where it is not.

    ``scale`` is :data:`GRID_SCALE` from :func:`plot_config_comparison` and 1.0 anywhere else:
    it thickens the strokes and the note with the text, and leaves the panel's chrome alone.

    Serves the identity-level panels too: the two levels differ in what was counted, not in how it
    is drawn, and which one a panel shows is carried by the y label its curve type registers
    rather than by anything here.
    """
    shared_k = min(item.curve.max_k for item in series)
    rows = []
    slots = resolve_slots([item.slot for item in series], [item.dash for item in series])
    for item, slot in zip(series, slots):
        curve = item.curve.curve
        curve = curve[curve["k"] <= shared_k]
        color = series_style(slot)
        axes.fill_between(curve["k"], curve["ci_low"], curve["ci_high"], color=color,
                          alpha=BAND_ALPHA, linewidth=0, zorder=2)
        axes.plot(curve["k"], curve["accuracy"], color=color, linewidth=LINE_WIDTH * scale,
                  linestyle=line_style(item.dash), solid_capstyle="round", zorder=3)
        handles.setdefault(item.label, Line2D([], [], color=color, linewidth=LINE_WIDTH * scale))
        rows.append(curve.assign(series=item.column))
    baseline = series[0].curve.curve
    baseline = baseline[baseline["k"] <= shared_k]
    drawn = ("random_proportional" if baseline["random_proportional"].notna().all()
             else "random")
    axes.plot(baseline["k"], baseline[drawn], color=TEXT_MUTED, linewidth=1.2 * scale,
              linestyle=BASELINE_DASH, zorder=2)
    # Every second decade on a wide-pool corpus, every decade on a narrow one. The two corpora
    # differ by two orders of magnitude in pool size, so a fixed stride either crowds one or
    # leaves the other with two labels on the whole axis.
    log_x_axis(axes, decades_per_tick=1 if shared_k < 1000 else 2)
    unit_y_axis(axes)
    # One count per line rather than the usual "N docs · M users" pair: at `GRID_SCALE` that pair
    # runs wider than a panel and hangs out over the y tick labels. Every drawer here notes its
    # counts this way for the same reason.
    counts = series[0].curve
    # `openset/dirfar`'s note, verbatim: out/in cohorts in the level's unit, then the known pool.
    # The in-set count is users at the identity level, documents otherwise.
    n_in_set = counts.n_users if counts.level == "identity" else counts.n_documents
    note = f"{abbreviate_count(counts.n_ood)} out, {abbreviate_count(n_in_set)} in"
    if counts.n_known is not None:
        note += f"\n{abbreviate_count(counts.n_known)} cand."
    panel_note(axes, note, scale=scale, corner=counts.note_corner)
    return pd.concat(rows, ignore_index=True)


def draw_words_identification_panel(axes, series: list[Series], handles: dict,
                                    scale: float = 1.0) -> pd.DataFrame:
    """One cell's DIR at a pinned FAR against the target conversation's word count.

    :func:`draw_binned_panel` again -- the categorical bins, the population bars and the gated
    baseline -- with the open-set note: the out/in cohort pair
    ``openset/dirfar`` prints, in the level's unit, and the FAR actually achieved, which
    is a property of the configuration (every attack is held to the same budget over the same
    out-of-set cohort, though each lands on its own attainable point; the first series' is
    printed, and every series' own is in the companion CSV's ``false_accept_rate``).

    The note is ``openset/dirfar``'s -- out/in cohorts and the known pool as ``cand.`` --
    without the FAR, which the path and caption already state. Lines carry no markers; five bins
    are few enough to read without them.
    """
    first = series[0].curve
    n_in_set = first.n_users if first.level == "author" else first.n_documents
    note = f"{abbreviate_count(first.n_ood)} out, {abbreviate_count(n_in_set)} in"
    if first.n_candidates is not None:
        note += f"\n{abbreviate_count(first.n_candidates)} cand."
    return draw_binned_panel(axes, series, handles, scale, note, markers=False)


def draw_binned_panel(axes, series: list[Series], handles: dict, scale: float,
                      note: str, markers: bool = True) -> pd.DataFrame:
    """A binned rate over its own population histogram: the body of every categorical-x family.

    The bars are the population each point stands on -- the histogram the request asked the
    accuracy to be overlaid on -- and they are **not a second y axis**. They are a *share* of the
    panel, which puts them on the same 0-1 scale as the accuracy above them honestly rather than
    by an arbitrary alignment of two ranges, and the share is of whatever the level counts
    (documents at ``doc/``, users at ``author/``), so a bar is the weight its own point carries in
    the panel's overall number. Absolute counts ride in the panel note and in the companion CSV,
    which is where a number that needs to be exact belongs.

    Drawn once per panel in recessive grey: the
    distribution of documents per author (or of words per undefended conversation) is a property
    of the corpus and the configuration, so every line in a cell stands on the same one -- a
    defense rewrites text, not how much of it somebody wrote.

    The x axis is **categorical**, labelled from the curve's own ``bin_label`` column. The bins
    are unequal in width by construction (``1``, ``2``, ``3-4``, ... ``33+``), so they are drawn
    at equal spacing and named on the ticks rather than placed on a count axis where the last bin
    would be five sixths of the width.

    The baseline follows the accuracy's gaps: where a bin was dropped for thinness there is no
    measurement to compare against, and a lone dashed segment over an empty stretch reads as one.
    """
    population = series[0].curve.curve
    axes.bar(population["bin"], population["share"], width=0.72, color=TEXT_MUTED, alpha=0.22,
             linewidth=0.8, edgecolor=SURFACE, zorder=1)
    rows = []
    slots = resolve_slots([item.slot for item in series], [item.dash for item in series])
    for item, slot in zip(series, slots):
        curve = item.curve.curve
        color = series_style(slot)
        axes.fill_between(curve["bin"], curve["ci_low"], curve["ci_high"], color=color,
                          alpha=BAND_ALPHA, linewidth=0, zorder=2)
        # Markers throughout: there are seven of them, and each one is a bin rather than a sample
        # of a continuum -- the line between two of them interpolates nothing.
        marker = (dict(marker="o", markersize=MARKER_SIZE * scale, markeredgecolor=SURFACE,
                       markeredgewidth=2 * scale) if markers else {})
        axes.plot(curve["bin"], curve["accuracy"], color=color, linewidth=LINE_WIDTH * scale,
                  linestyle=line_style(item.dash), solid_capstyle="round", zorder=3, **marker)
        handles.setdefault(item.label, Line2D([], [], color=color, linewidth=LINE_WIDTH * scale))
        rows.append(curve.assign(series=item.column))
    baseline = series[0].curve.curve
    drawn = ("random_proportional" if baseline["random_proportional"].notna().all()
             else "random")
    axes.plot(baseline["bin"], baseline[drawn].where(baseline["accuracy"].notna()),
              color=TEXT_MUTED, linewidth=1.2 * scale, linestyle=BASELINE_DASH, zorder=2)
    labels = list(population["bin_label"])
    axes.set_xticks(np.arange(len(labels)), labels=labels)
    axes.set_xlim(-0.6, len(labels) - 0.4)
    # The bins are the only categorical x axes in the file, and at `GRID_SCALE` a label like
    # "17-32" is wider than its bin pitch, so unrotated labels run into one another. Rotated
    # rather than thinned: every bin is a drawn point and a reader needs to know which one they
    # are looking at. The alignment that pairs with the angle is `label_facets`' -- see the note
    # there.
    if scale != 1.0:
        axes.tick_params(axis="x", labelrotation=45)
    unit_y_axis(axes)
    panel_note(axes, note, scale=scale, corner="upper left")
    return pd.concat(rows, ignore_index=True)


def draw_identification_panel(axes, series: list[Series], handles: dict,
                              scale: float = 1.0) -> pd.DataFrame:
    """One cell's DIR-FAR curves: what identification survives the rejection threshold.

    Read left to right as the attacker loosening its filter. At the left edge it accepts no
    strangers and keeps only what it was surest of; at the right edge it rejects nothing and the
    curve arrives at exactly the closed-set top-1 for the same cell. The *shape*
    between them is the finding this figure exists for -- how much of a headline accuracy is
    still there once the attacker has to decide who is even enrolled.

    The dashed line is :func:`identification_baseline`, and it lies almost on the
    axis: chance identification is a property of a large candidate pool. A curve indistinguishable
    from that dash is an attack that identifies nobody at any threshold.

    **The x axis is log, not the file's usual linear unit square**: most of a curve's rise happens
    below FAR=0.1, which a linear axis spends most of its width not showing. See
    :func:`log_far_axis`.
    """
    rows = []
    slots = resolve_slots([item.slot for item in series], [item.dash for item in series])
    for item, slot in zip(series, slots):
        curve = item.curve.curve
        color = series_style(slot)
        # FAR=0 has no image on the log axis below -- see `log_far_axis` -- so it is excluded
        # from what is drawn. The full curve, FAR=0 included, still goes into `rows` and so into
        # the companion CSV.
        visible = curve[curve["false_accept_rate"] >= FAR_LOG_FLOOR]
        axes.fill_between(visible["false_accept_rate"], visible["ci_low"], visible["ci_high"],
                          color=color, alpha=BAND_ALPHA, linewidth=0, zorder=2)
        axes.plot(visible["false_accept_rate"], visible["identification_rate"], color=color,
                  linewidth=LINE_WIDTH * scale, linestyle=line_style(item.dash),
                  solid_capstyle="round", zorder=3)
        handles.setdefault(item.label, Line2D([], [], color=color, linewidth=LINE_WIDTH * scale))
        rows.append(curve.assign(series=item.column, top1=item.curve.top1))
    baseline = series[0].curve.curve
    visible_baseline = baseline[baseline["false_accept_rate"] >= FAR_LOG_FLOOR]
    drawn = ("random_proportional" if baseline["random_proportional"].notna().all() else "random")
    axes.plot(visible_baseline["false_accept_rate"], visible_baseline[drawn], color=TEXT_MUTED,
              linewidth=1.2 * scale, linestyle=BASELINE_DASH, zorder=2)
    log_far_axis(axes)
    unit_y_axis(axes)
    # **Upper left, and the corner was chosen by measurement rather than by eye.** A monotone
    # curve from (0, 0) to (1, top-1) splits the panel, so which corner it leaves free depends on
    # top-1 *as a fraction of the axis* -- and with `PANEL_Y_LIMITS` that fraction now differs by
    # corpus, which makes a single representative figure worthless for deciding it. Swept across
    # the whole family, this corner collides least often; the default bottom right is where the
    # low-scoring figures (weak features, on both corpora) run along the bottom exactly where it
    # sits. Re-run the sweep before changing this or a `PANEL_Y_LIMITS` entry if it starts to bite.
    #
    # The note is "X out, Y in" on one line and the known pool as "Z cand." beneath it -- the unit
    # is left to the path (`doc/` or `author/`), so the out/in pair and the pool are all counted
    # in whatever that level counts.
    first = series[0].curve
    rate = f"\nDIR@{IDENTIFICATION_NOTE_FAR:.0%} {first.dir_at(IDENTIFICATION_NOTE_FAR):.3f}"
    n_in_set = (first.n_users if first.level == "author" else first.n_documents) - first.n_ood
    counts = f"{abbreviate_count(first.n_ood)} out, {abbreviate_count(n_in_set)} in"
    if first.n_candidates is not None:
        counts += f"\n{abbreviate_count(first.n_candidates)} cand."
    panel_note(axes, f"{counts}{rate if len(series) == 1 else ''}", scale=scale,
               corner="upper left")
    return pd.concat(rows, ignore_index=True)


#: The curve types, each as (panel drawer, subdirectory, x label, y label, description).
#:
#: **Both axis titles are the quantity's short name** -- `Accuracy`, `Precision`, `Coverage`,
#: `Candidate users`. They are drawn at :data:`GRID_SCALE` on a panel a third of the figure wide,
#: where a sentence does not fit; the sentence each one used to be is in the description below it,
#: which is the field captions are copied from.
#:
#: **The counting level is not in the y title**, so both levels of a family read `Accuracy`. The
#: level is in the path (`/doc` against `/author`) and in the panel note's own unit, which is where
#: the rest of the scheme keeps it, and the description says what a unit of each one is.
#:
#: **The description is no longer drawn.** It was the subtitle under each figure's title, and the
#: title itself went the same way; that is caption text and belongs in whatever publishes the
#: figure. It is kept here because it is the one place each family's caveats are written down in a
#: sentence -- copy it into the caption rather than re-deriving it, and keep it current when a
#: family changes.
#:
#: **The subdirectory carries the counting level**, so every family lands at
#: ``<family>/<doc|author>/by_{defense,attack}/`` and a reader flips between two panels that
#: differ in one thing only: whether the unit is a document or a person. That is the whole path
#: scheme -- :func:`plot_defense_comparison` appends ``by_defense/<feature>_<attack>`` to
#: whatever is here and needs to know nothing about levels.
#:
#: Every entry here is drawn by both comparison families (``by_defense`` and ``by_attack``).
CURVE_TYPES = {
    "identification": (draw_identification_panel, "openset/dirfar/doc",
                       "FAR", "DIR",
                       "DIR is the share of in-set documents "
                       "both accepted and ranked top-1, against FAR, the share of out-of-set "
                       "documents the same threshold wrongly accepts. k=1. The right edge (reject "
                       "nothing) is exactly the closed-set top-1 for this cell, so the curve is "
                       "the cost of the open world in the units of a closed-set result. Grey "
                       "dashes: a detector that knows nothing in front of a "
                       "guesser that reads no text, f x the proportional top-1"),
    "identification_authors": (draw_identification_panel, "openset/dirfar/author",
                               "FAR", "DIR",
                               "One person at a time: DIR is the share of in-set users linked at "
                               "least once by a document that survived the threshold, FAR the "
                               "share of out-of-set users with any document accepted. The right "
                               "edge is the closed-set author-level top-1. A person is linked on their "
                               "most confident CORRECT document and accepted on their most "
                               "in-set-looking one, which are two different documents"),
    "identification_words": (draw_words_identification_panel,
                             "openset/dirfar_by_words/doc", "Words", "DIR",
                             "DIR at a 10% FAR budget -- the share of in-set documents both "
                             "accepted and ranked top-1, at the most permissive threshold "
                             "leaking at most 10% of out-of-set documents -- split by the "
                             "target conversation's word count (whitespace tokens over its user "
                             "turns, counted on the UNDEFENDED text so every defense shares the "
                             "bins). "
                             "One threshold for the whole test set, read within each bin. "
                             "Bars: each bin's share of in-set documents. Observational -- a long "
                             "conversation is a different conversation, not a short one given "
                             "more words. Gemini reads only the first 8,192 tokens. Grey "
                             "dashes: a chance detector at the achieved FAR in front of the "
                             "proportional guesser"),
    "identification_words_authors": (draw_words_identification_panel,
                                     "openset/dirfar_by_words/author", "Words", "DIR",
                                     "Share of in-set users linked at least once by an accepted, "
                                     "top-1 conversation, at a threshold leaking at most 10% of "
                                     "out-of-set USERS (stricter than the document panel's, so "
                                     "the two levels are not one operating point). Each user is "
                                     "in ONE bin, by the mean word count of their in-set "
                                     "conversations (same bins as the document level), so the "
                                     "bins partition the users and the user-weighted mean over "
                                     "bins is openset/top_k_cmc/author's k=1 point. Bars: each "
                                     "bin's share of in-set users"),
    "identification_cmc": (draw_cmc_panel, "openset/top_k_cmc/doc",
                           "Top k candidates", "DIR",
                           "The closed-set CMC curve with the rejection filter applied: the share "
                           "of ALL in-set documents ranked within k by a document the attacker "
                           "accepted, at the most permissive threshold within a 10% FAR budget. "
                           "A refused document is a miss at every k. Its k=1 point is the "
                           "openset/dirfar panel read at FAR 10%; the two are orthogonal "
                           "slices of DIR(threshold, k). Grey dashes: a chance detector at the "
                           "same achieved FAR in front of the proportional guesser, which "
                           "saturates at the achieved FAR itself -- a guesser allowed every "
                           "candidate is right about whatever it was handed"),
    "identification_cmc_authors": (draw_cmc_panel, "openset/top_k_cmc/author",
                                   "Top k candidates", "DIR",
                                   "Share of in-set users linked within k by a document that "
                                   "survived the filter. The budget is spent on out-of-set "
                                   "USERS, which is a stricter threshold than the document "
                                   "panel's -- a stranger is falsely accepted the moment any one "
                                   "of their documents is -- so the two levels are NOT the same "
                                   "operating point and must not be read against each other at a "
                                   "fixed k. A user's rank is the best among their ACCEPTED "
                                   "documents, not their best overall. **The baseline overtakes "
                                   "the attack at large k on WildChat and that is a real "
                                   "result**: at k = every candidate both reduce to the share of "
                                   "in-set users with any accepted document, and a chance "
                                   "detector spreads the same 10% of accepts over more people "
                                   "than a real one, which concentrates them on the confident "
                                   "and prolific. It touches more users while identifying none "
                                   "of them, so read the LOW-k end, where the gap is 100x"),
}


#: Which curve family :func:`run_curves` builds feeds which :data:`CURVE_TYPES` entry. Both levels
#: of a family appear here as separate entries -- that is what makes a level cost one line rather
#: than a branch in the driver.
CURVE_FAMILIES = {
    "identification": "identification",
    "identification_authors": "identification_authors",
    "identification_cmc": "identification_cmc",
    "identification_cmc_authors": "identification_cmc_authors",
    "identification_words": "identification_words",
    "identification_words_authors": "identification_words_authors",
}

def plot_config_comparison(kind: str, panels: dict[str, list[Series]],
                           legend_title: str, stem: Path,
                           extra_legend: tuple[str, dict] | None = None,
                           dataset: str | None = None) -> Path:
    """One curve type, one figure, one panel per known configuration.

    Nothing is averaged across the grid. The configurations differ in what the attacker was
    given, which is an experimental condition rather than a repeated measurement -- averaging
    them would report a number describing no experiment that was run, and the mean would move
    with the arbitrary choice of which cells were included.

    ``extra_legend`` is a second legend block for figures whose lines carry two channels -- the
    ``by_attack`` figures, where colour is the attack and dash is the feature.

    **This is where :data:`GRID_SCALE` is spent, for every family at once.** Everything that
    reaches the page through this function -- the panels' text, their strokes, their notes, the
    legend -- is drawn
    at twice the file's base sizes on the square :data:`GRID_FIGSIZE` canvas, and the clustering
    figures, which do not come through here, are untouched. Keeping
    it at the dispatch point rather than in the constants is what lets the drawing helpers serve
    both: each takes a ``scale`` defaulting to 1.0.
    """
    draw, _, xlabel, ylabel, _description = CURVE_TYPES[kind]
    scale = GRID_SCALE
    figure, grid, axes_for, legend_cell = config_axes_grid(figsize=GRID_FIGSIZE)
    handles: dict[str, object] = {}
    rows = []
    for tag, axes in axes_for.items():
        series = panels.get(tag)
        if not series:
            axes.set_visible(False)
            continue
        rows.append(draw(axes, series, handles, scale=scale).assign(known_config=tag))
        # After the drawer, so a family that does not take the unit square overrides the axis its
        # drawer set rather than every drawer having to ask.
        if (kind, dataset) in PANEL_Y_LIMITS:
            zoom_y_axis(axes, *PANEL_Y_LIMITS[(kind, dataset)])
    label_facets(grid, axes_for, xlabel, ylabel, scale=scale)
    # The dash legend's swatches stand for the lines in the panels, so they are thickened with
    # them. They are built by `feature_legend`/`dataset_legend`, which serve every family and so
    # cannot know the scale; restyling the handles here keeps that generality.
    for handle in (extra_legend[1] if extra_legend else {}).values():
        handle.set_linewidth(GRID_LINE_WIDTH)
    return finish_facets(figure, handles, legend_title, stem,
                         pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(),
                         legend_cell=legend_cell, extra_legend=extra_legend, scale=scale,
                         row_pad=GRID_ROW_PAD)


# --- grouping runs into the two comparison families --------------------------
#
# Both families are "hold one axis fixed, draw a line per value of the other". They are written
# once and parameterised by the curve type, so every curve type gets both views.

def method_groups(runs: list[Run]) -> list[tuple[tuple[str, str], list[Run]]]:
    """Runs grouped by (feature, attack) -- one group per ``by_defense`` figure.

    Split out of the drawing so a figure can be *named*, and therefore checked against the cache,
    before any curve is built. Groups come back in colour-slot order and each group's runs in
    defense order, so the series order on a figure is a property of the vocabulary rather than of
    which runs happened to be on disk.
    """
    groups: dict[tuple[str, str], list[Run]] = defaultdict(list)
    for run in runs:
        groups[run.method].append(run)
    return [(method, sorted(members, key=lambda run: DEFENSE_SLOTS[run.defense]))
            for method, members in sorted(groups.items(), key=lambda item: METHOD_SLOTS[item[0]])]


def defense_groups(runs: list[Run]) -> list[tuple[str, list[Run]]]:
    """Runs grouped by defense -- one group per ``by_attack`` figure. :func:`method_groups`' twin."""
    groups: dict[str, list[Run]] = defaultdict(list)
    for run in runs:
        groups[run.defense].append(run)
    return [(defense, sorted(members, key=lambda run: METHOD_SLOTS[run.method]))
            for defense, members in sorted(groups.items(),
                                           key=lambda item: DEFENSE_SLOTS[item[0]])]


def plot_defense_comparison(dataset: str, method: tuple[str, str], runs: list[Run], curves: dict,
                            kind: str, output_dir: Path) -> list[Path]:
    """One figure: every defense measured against one (feature, attack).

    The figure that answers "does the defense work?" -- the attack is held fixed, so the only
    thing that moves between lines is what the defense did to the text.

    Returns an empty list when no run in the group produced this curve family (a run with no
    open-set rows, or no corpus parquet for the word-count split). That is a real outcome rather
    than a failure, and the cache records it as one so the figure is not
    re-planned every sweep.
    """
    panels: dict[str, list[Series]] = defaultdict(list)
    for run in runs:
        for tag, curve in curves.get(run, {}).items():
            panels[tag].append(Series(run.defense_label, DEFENSE_SLOTS[run.defense], curve))
    if not panels:
        return []
    feature, attack = method
    return [plot_config_comparison(
        kind, panels,
        legend_title="Defense",
        stem=output_dir / CURVE_TYPES[kind][1] / "by_defense" / f"{feature}_{attack}",
        dataset=dataset)]


def feature_legend(series_by_panel: dict[str, list[Series]]) -> tuple[str, dict]:
    """The second legend of a ``by_attack`` figure: what the line *pattern* says.

    The dash twin of the colour legend, built the same way :func:`dataset_legend` is and for the
    same reason: a reader needs both blocks to decode one line, so they are stacked rather than
    merged. It holds the features and nothing else -- the baseline stays out of it, because a
    pattern here applies to every measured line uniformly and the baseline is identified by being
    the only grey one.

    Drawn in ink rather than in any series colour, so a sample cannot be misread as belonging to
    one particular attack. Ordered by :data:`FEATURES` so the block is in vocabulary order rather
    than in whatever order the runs came off disk.
    """
    present = [feature for feature in FEATURES
               if any(item.dash_label == FEATURE_LABELS[feature]
                      for panel in series_by_panel.values() for item in panel)]
    return "Feature", {
        FEATURE_LABELS[feature]: Line2D([], [], color=TEXT_SECONDARY, linewidth=LINE_WIDTH,
                                        linestyle=line_style(FEATURE_DASHES[feature]))
        for feature in present}


def plot_attack_comparison(dataset: str, defense: str, runs: list[Run], curves: dict, kind: str,
                           output_dir: Path) -> list[Path]:
    """One figure: every (feature, attack) measured against one defense.

    The transpose of :func:`plot_defense_comparison` -- with the defense held fixed, it says which
    representation and estimator the attacker should reach for.

    **Two channels: colour is the attack, dash is the feature**, with
    a legend for each. It used to be one line per (feature, attack) pair in its own hue, which was
    readable at two attacks and is not at ten -- the registry has grown and a panel was carrying a
    dozen colours. Splitting the two axes onto two channels makes the comparison the figure exists
    for legible directly: one attack's two representations are one hue in two patterns, so
    "Gemini beats the character n-gram under every attack" is read off the patterns rather than by
    matching legend entries in pairs. The colour legend's samples stay **solid** whatever the
    series' dashes -- they stand for the hue, and the dash block speaks for the pattern.

    The complementary narrowing is :data:`DEFAULT_ATTACKS`, which is what keeps the colour channel
    inside the eight-hue palette; ``--attacks all`` widens it again and :func:`resolve_slots` then
    renumbers if two attacks would collide.
    """
    panels: dict[str, list[Series]] = defaultdict(list)
    for run in runs:
        for tag, curve in curves.get(run, {}).items():
            panels[tag].append(Series(ATTACK_LABELS[run.attack], ATTACK_SLOTS[run.attack], curve,
                                      dashes=FEATURE_DASHES[run.feature],
                                      dash_label=FEATURE_LABELS[run.feature]))
    if not panels:
        return []
    return [plot_config_comparison(
        kind, panels,
        legend_title="Attack",
        stem=output_dir / CURVE_TYPES[kind][1] / "by_attack" / defense,
        extra_legend=feature_legend(panels), dataset=dataset)]


# --- the corpus parquet: columns the baselines and the word-count split read ---------------


def corpus_documents(dataset: str) -> pd.DataFrame | None:
    """``doc_id``/``author_id``/``ended_at`` for one dataset, in ``run_experiment.py``'s own order.

    Two figures need something the runs themselves do not carry -- when a document ended, and how
    much each author wrote on the known side -- and both are properties of the *corpus* rather
    than of any attack. Joining them back here keeps one copy of the fact and, the reason that
    matters in practice, makes both available for **every run already on disk**, including ones
    far too expensive to re-run for a column.

    The ordering reproduces ``load_documents_and_features`` exactly, because it is what every
    window boundary is defined against: undated documents dropped (a real share of SWE-chat's
    corpus, so keeping them would shift every boundary), then sorted by ``ended_at`` with ties
    broken by ``doc_id``. Verified against both corpora: the reconstructed ``known0025`` boundary
    lands on the same author count the runner recorded and on the first ``position`` its
    predictions file reports.

    Three columns are read, so the 411 MB WildChat parquet costs a projection rather than a load.
    Memoised because a dataset's runs all need the same frame.
    """
    if dataset in _CORPUS:
        return _CORPUS[dataset]
    # The dataset part of a run's name is the split name, so it is also the parquet's basename.
    path = DATA_DIR / f"{dataset}.parquet"
    frame = None
    if path.exists():
        frame = pd.read_parquet(path, columns=["doc_id", "author_id", "ended_at",
                                               "language_primary"])
        frame["ended_at"] = pd.to_datetime(frame["ended_at"], errors="coerce", utc=True,
                                           format="mixed")
    _CORPUS[dataset] = frame
    return frame


#: Memo for :func:`corpus_documents`, one entry per dataset (``None`` when its parquet is absent).
_CORPUS: dict[str, pd.DataFrame | None] = {}


def document_word_counts(dataset: str) -> pd.Series | None:
    """``doc_id -> word count`` of the **undefended** conversation, for
    ``openset/dirfar_by_words/``.

    A word is a whitespace-separated token, summed over the conversation's user turns (the only
    turns the corpus keeps). That undercounts scripts written without spaces -- a Chinese turn is
    one "word" per run of characters -- which is acceptable for a length axis on two corpora that
    are overwhelmingly space-delimited, and is said in the family's description.

    Counted on the original text rather than a defense's rewrite, deliberately: a ``by_attack``
    figure puts every defense's line on one x axis, and binning each defense by its own rewrite
    would move documents between bins from line to line, so no two points above one tick would
    describe the same conversations.

    Streamed a batch at a time because ``turns`` is the large majority of the WildChat parquet;
    only the counts are kept. Memoised per dataset and warmed by :func:`warm_baselines` in the
    parent, so the forked curve workers inherit it.
    """
    if dataset not in _WORD_COUNTS:
        path = DATA_DIR / f"{dataset}.parquet"
        counts = None
        if path.exists():
            import pyarrow.compute as pc
            import pyarrow.parquet as pq
            ids, words = [], []
            for batch in pq.ParquetFile(path).iter_batches(columns=["doc_id", "turns"],
                                                           batch_size=20_000):
                turns = batch.column("turns")
                per_turn = pc.list_value_length(pc.utf8_split_whitespace(pc.list_flatten(turns)))
                parents = pc.list_parent_indices(turns).to_numpy()
                words.append(np.bincount(parents, weights=per_turn.to_numpy(zero_copy_only=False),
                                         minlength=len(turns)).astype(np.int64))
                ids.append(batch.column("doc_id").to_numpy(zero_copy_only=False))
            counts = pd.Series(np.concatenate(words), index=np.concatenate(ids))
        _WORD_COUNTS[dataset] = counts
    return _WORD_COUNTS[dataset]


#: Memo for :func:`document_word_counts`, one entry per dataset (``None`` when its parquet is
#: absent).
_WORD_COUNTS: dict[str, pd.Series | None] = {}


def known_documents(dataset: str, known_config: str) -> pd.DataFrame | None:
    """The corpus rows on ``known_config``'s known side -- the gallery the attack searched.

    The known side is the interval ``known_config`` names, taken over the dated corpus in
    chronological order: ``round(fraction * n_documents)`` at each end, the same arithmetic
    ``known_configurations`` uses, so this is the real gallery rather than an approximation of it.

    The proportional baseline wants a *distribution* over authors (:func:`known_author_shares`),
    so the boundary arithmetic lives here once. Memoised, because every run of a dataset asks for the same six
    known sides and the slice costs a sort of the whole corpus.

    What is memoised is the *ordering* (:func:`dated_corpus`), one frame per dataset, rather than
    the six slices: a slice keeps its parent frame alive, so caching them would hold six copies of
    the corpus for what the callers reduce to a few thousand counts. The slice itself is an
    ``iloc`` on an already-sorted frame and costs nothing to repeat.
    """
    dated = dated_corpus(dataset)
    if dated is None:
        return None
    config = parse_config_tag(known_config)
    known = dated.iloc[int(round(config.start * len(dated))):
                       int(round(config.end * len(dated)))]
    return known if len(known) else None


def dated_corpus(dataset: str) -> pd.DataFrame | None:
    """:func:`corpus_documents` in ``load_documents_and_features``'s order, memoised per dataset.

    Undated documents dropped -- a real share of swe-chat's corpus, so keeping them would shift
    every window boundary -- then sorted by ``ended_at`` with ``doc_id`` breaking ties. Every
    known-side reconstruction slices this, so the sort is paid once per dataset instead of once
    per (dataset, configuration).
    """
    if dataset not in _DATED_CORPUS:
        frame = corpus_documents(dataset)
        _DATED_CORPUS[dataset] = None if frame is None else frame[
            frame["ended_at"].notna()].sort_values(["ended_at", "doc_id"], kind="mergesort")
    return _DATED_CORPUS[dataset]


#: Memo for :func:`dated_corpus`, one entry per dataset.
_DATED_CORPUS: dict[str, pd.DataFrame | None] = {}


def known_author_counts(dataset: str, known_config: str) -> pd.Series | None:
    """How many documents each author wrote on ``known_config``'s known side."""
    known = known_documents(dataset, known_config)
    return None if known is None else known["author_id"].value_counts()


def known_author_shares(dataset: str, known_config: str) -> pd.Series | None:
    """Each known author's share of the known side's documents: the prior the baseline guesses on.

    :func:`known_author_counts` normalised, and kept as its own function because it is the form
    every baseline wants and normalising at each call site would invite one of them to forget.
    """
    counts = known_author_counts(dataset, known_config)
    return None if counts is None else counts / counts.sum()


# --- author clustering: the other experiment ---------------------------------
#
# These three families were `experiments/plot_clustering.py` until they were merged in here. That
# file already imported this one's palette, chrome and baseline convention, so keeping it separate
# bought nothing and cost two things: its figures sat outside the plot cache (every sweep redrew
# them), and they were filed under `plots/clustering/<dataset>/` rather than beside the corpus
# they describe. They now write `plots/<dataset>/clustering/`, which is the same rule every other
# family follows -- a dataset's figures live under that dataset.
#
# What they draw is a different *experiment*, not a different view of this one. Everything above
# reads `experiments/results/<4-part-name>/` and reports a **ranking** over named authors: the
# attacker holds a labelled known side and asks which enrolled person wrote an unknown document.
# These read `experiments/clustering/<3-part-name>/` and report a **partition**: nobody is named,
# and the question is whether an anonymised log falls apart into its authors on its own. There is
# no known side, so there is no configuration grid and none of the machinery above -- no
# bootstrap, no `PanelWeights`, no `CURVE_TYPES` entry. The drawing routines read their own CSVs
# at draw time, the way `plot_run_detail` does.

#: Clustering algorithms, in the order that fixes each one's colour everywhere. **Append only** --
#: inserting a name shifts the hue of every algorithm below it, exactly the hazard
#: :data:`DEFENSE_SLOTS` and :data:`METHOD_STRIDE` document at length.
#:
#: **This is the colour registry, not the guest list.** It holds every algorithm that has ever
#: been drawn, including ones nothing draws today, precisely so that retiring one cannot move
#: anybody else's hue. What the figures draw is :data:`CLUSTERING_DRAWN_ALGORITHMS`.
CLUSTERING_ALGORITHMS = ("hdbscan", "leiden", "average_linkage", "connected",
                         "componentwise_agglomerative")

CLUSTERING_ALGORITHM_SLOTS = {name: index for index, name in enumerate(CLUSTERING_ALGORITHMS)}

#: The algorithms every clustering figure draws, **in the order it draws them** -- bar order on
#: the BCubed chart, and legend order on the precision/recall figure.
#:
#: Separate from the registry above because the two questions are different: colour is a permanent
#: property of an algorithm (its index there), while this is the subset on show today. Dropping a
#: name here changes nothing about the others; dropping it *there* would recolour every algorithm
#: below it and put a paper's existing figures out of step with a re-run.
#:
#: **``average_linkage`` is out and ``componentwise_agglomerative`` is in.** The two are the same
#: method -- the second runs it one connected component at a time so it fits in memory at
#: WildChat's scale -- and only the second exists on both corpora, which is what decided it: with
#: no WildChat run of the first, keeping it would have left that corpus with three algorithms.
#: Where both exist they agree closely, the per-component variant trading a little recall for
#: precision, which is visible on the precision/recall figure and invisible on the bar chart. It is
#: labelled simply "Agglomerative" now: with nothing to distinguish it from, the qualifier said
#: only that an implementation detail existed.
CLUSTERING_DRAWN_ALGORITHMS = ("hdbscan", "leiden", "connected", "componentwise_agglomerative")

#: Reference partitions, drawn beside the algorithms as **grey bars**. They are properties of the
#: collection rather than measurements of an attack, and grey is the channel that says so -- the
#: same reservation the dashed baseline relies on elsewhere, and grey is a hue no series occupies.
#:
#: They were dashed horizontal lines until they were changed to bars. The dash was the file's own
#: "not a measurement" convention, but it made the one comparison the figure exists for -- did the
#: attack beat doing nothing? -- a matter of reading a bar against a line, with several lines close
#: to each other and their names pushed apart by a de-cluttering pass to stop them stacking. As
#: bars they are on the axis the algorithms are on, named on the same ticks, and the grey still
#: carries what the dash did.
#:
#: ``baseline_model_owner`` is deliberately absent. It partitions by which provider served the
#: conversation, and on a corpus whose documents nearly all come from one provider that is the
#: one-cluster partition under another name. `run_clustering.py` still computes it; nothing draws
#: it.
CLUSTERING_BASELINES = ("baseline_singleton", "baseline_single_cluster", "baseline_random",
                        "baseline_language_primary")

#: The three baselines the precision/recall figure marks, which is a subset: the metadata
#: partitions land in the same corner as ``single_cluster`` and would only crowd it.
CLUSTERING_PR_BASELINES = ("baseline_singleton", "baseline_single_cluster", "baseline_random")

#: Iso-F contours drawn on the precision/recall figure, each labelled where it leaves the right
#: edge. They are the figure's frame of reference: two points on one contour are the same F bought
#: with different trades, which is the comparison F alone cannot show.
#:
#: **The range the results occupy, and no more**: every measured point on both corpora sits well
#: within this range, which is where a reader interpolates, and the low levels place the baselines
#: -- ``Random`` and ``One cluster`` sit near the bottom, ``Singletons`` a little above them on one
#: corpus and just above the floor on the other, which is what the middle level is for. The higher
#: levels were dropped: nothing comes near them, and they ran through the legend block in the
#: top-right corner.
CLUSTERING_ISO_F_LEVELS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)

#: How far above 1.0 a computed precision may sit and still count as "on the top edge". It exists
#: for exactly one sample per contour -- the analytic end point at ``r = F / (2 - F)``, which
#: binary arithmetic puts a couple of ULPs over 1.0 at some levels -- and it is orders of magnitude
#: tighter than the gap to the next sample down, which is ~0.06.
ISO_F_EDGE_TOLERANCE = 1e-9

#: Marker size for the precision/recall figure, in points, before the per-shape correction below.
CLUSTERING_PR_MARKER_SIZE = 7.0


CLUSTERING_LABELS = {
    "hdbscan": "HDBSCAN",
    "leiden": "Leiden",
    # Retired from `CLUSTERING_DRAWN_ALGORITHMS` and kept here for its colour slot and its
    # meaning: it is average linkage over the whole graph, which only SWE-chat has a run of.
    "average_linkage": "Average linkage",
    # The same method, run one connected component at a time so it fits in memory at WildChat's
    # scale. It is the agglomerative row every figure draws now, and the qualifier came off with
    # the row it used to be told apart from.
    "componentwise_agglomerative": "Agglomerative",
    "connected": "Connected comp.",
    # Short names: these sit beside the measured methods, and a parenthetical qualifier on a
    # reference partition reads as a caveat about the *attack*.
    # What "matched" and "all" said is in each figure's own documentation instead.
    "baseline_singleton": "Singletons",
    "baseline_single_cluster": "One cluster",
    "baseline_random": "Random",
    "baseline_language_primary": "By language",
}

#: Opacity of a reference-partition bar. Solid grey would compete with the measured bars for
#: attention; this keeps them legible and clearly recessive, the same call
#: :func:`draw_binned_panel` makes for its population histogram.
CLUSTERING_BASELINE_ALPHA = 0.55

#: Blank slots between the algorithm bars and the reference bars, in bar widths. Enough that the
#: two groups read as two groups without a rule between them.
CLUSTERING_GROUP_GAP = 0.9


#: The fourth part of a clustering directory name -> the label a figure prints for it.
#:
#: The variant is ``run_clustering.variant_name``'s composition of the three axes that change what
#: an edge score *means*: ``--projection``, ``--rescoring`` and the elapsed-time fusion, in that
#: order, or ``plain`` when none is set. Order here is the variants figure's row order, and it is
#: the order the ideas were built in: the pure-text run first, then each ingredient alone, then
#: the combinations.
#:
#: **Append only, and only names ``variant_name`` can actually produce.** A directory whose
#: variant is not in this dict does not parse at all, which is what keeps a trailing ``_zscore``
#: or ``_quantile`` qualifier out of the comparable set. Not every projection x rescoring pair is
#: here -- only the ones that have been run -- so registering a new combination is one line.
#:
#: The by-hand directories that carried a *fixed* weight in the name (``time0.45``,
#: ``contrastive_time0.45``) are not registered: the weight is now searched, so ``time`` means
#: "tuned" and a number in that slot would be a different experiment.
CLUSTERING_VARIANT_LABELS = {
    # **Named for the channel, not for the material**: these two rows
    # differ by whether elapsed time is fused into the edge score, so saying so outright is what
    # a reader needs, and the pair reads as one contrast wherever they appear together -- the
    # variants dumbbells, and the `precision_recall/timing/` figure's "Edge score" legend, which
    # takes its two entries from here.
    "plain": "w/o timing",
    "time": "w/ timing",
    "contrastive": "Contrastive projection",
    "contrastive_time": "Contrastive + timing",
    "lda": "LDA projection",
    "lda_time": "LDA + timing",
    "wccn": "WCCN projection",
    "wccn_csls": "WCCN + CSLS",
    "wccn_local_scaling": "WCCN + local scaling",
    "csls": "CSLS rescoring",
    "local_scaling": "Local scaling",
    "mutual_knn": "Mutual k-NN",
    "shared_neighbors": "Shared neighbours",
}

#: The variant every other clustering family draws: the pure-text run, whose directory is the
#: four-part name with no method applied to the graph. The ``bcubed`` and ``precision_recall``
#: families key their panels on defense, so a run with a learned projection among
#: them would collide with the plain run of the same defense.
PLAIN_VARIANT = "plain"


@dataclass(frozen=True)
class ClusteringRun:
    """One clustering directory, with its name parsed into the four axes it encodes.

    :class:`Run`'s counterpart, and four parts for the same reason: a clustering attack has no
    ``--attacks`` axis, but it does have a *variant* -- which representation was clustered -- and
    that occupies the same positional slot. So what varies is the corpus, what was done to the
    text, how the text was represented, and what was done to the graph.

    The plain three-part names written before the variant part existed no longer parse. That is
    deliberate rather than a migration gap: a directory with no variant part cannot say whether it
    holds a pure-text run or something else, and guessing ``plain`` for it would file a
    timing-fused run among the pure-text ones.
    """

    dataset: str
    defense: str
    feature: str
    variant: str
    directory: Path

    @property
    def variant_label(self) -> str:
        return CLUSTERING_VARIANT_LABELS[self.variant]

    @property
    def defense_label(self) -> str:
        return DEFENSE_LABELS[self.defense]

    @property
    def feature_label(self) -> str:
        return FEATURE_LABELS[self.feature]


def parse_clustering_run_name(name: str) -> tuple[str, str, str, str] | None:
    """Split ``<dataset>_<defense>_<feature>_<variant>`` into its four parts, or ``None``.

    :func:`parse_run_name` with a variant where the attack goes, and it cannot be split on ``_``
    for the same reason: every part may contain one. Each candidate defense is checked against the
    requirement that what follows it is a whole feature name, which is what tells ``dp_mlm`` from
    ``dp_mlm_pii``, and the remainder has to be a registered variant.

    An unregistered variant returns ``None`` rather than being guessed at, so a run carrying a
    trailing qualifier (``_zscore``, ``_quantile``, ``_knowndef-<defense>``) stays out of the
    comparable set exactly as it did before -- those qualify a run rather than name a method, and
    ``run_clustering.py`` deliberately appends them after the variant.
    """
    for dataset in DATASETS:
        if not name.startswith(f"{dataset}_"):
            continue
        remainder = name[len(dataset) + 1:]
        for defense in DEFENSES:
            if not remainder.startswith(f"{defense}_"):
                continue
            tail = remainder[len(defense) + 1:]
            for feature in FEATURES:
                if (tail.startswith(f"{feature}_")
                        and tail[len(feature) + 1:] in CLUSTERING_VARIANT_LABELS):
                    return dataset, defense, feature, tail[len(feature) + 1:]
    return None


def discover_clustering_runs(clustering_dir: Path) -> list[ClusteringRun]:
    """Every parseable clustering directory, sorted so figures are built in a stable order.

    Absent directory is not an error: clustering is one experiment among several, and a checkout
    that has only run the attribution side should still draw its figures.
    """
    if not clustering_dir.exists():
        return []
    runs, skipped = [], []
    for directory in sorted(path for path in clustering_dir.iterdir() if path.is_dir()):
        parsed = parse_clustering_run_name(directory.name)
        if parsed is None:
            skipped.append(directory.name)
            continue
        runs.append(ClusteringRun(*parsed, directory=directory))
    if skipped:
        print(f"skipped {len(skipped)} clustering director{'y' if len(skipped) == 1 else 'ies'} "
              f"whose name is not <dataset>_<defense>_<feature>_<variant>: "
              f"{', '.join(skipped)}")
    # Sorted so the variants figure's rows come out in vocabulary order whatever the filesystem
    # hands back, exactly as the old `discover_clustering_variants` did.
    order = list(CLUSTERING_VARIANT_LABELS)
    return sorted(runs, key=lambda run: (run.dataset, run.defense, run.feature,
                                         order.index(run.variant)))


#: The author scopes ``run_clustering.py --scopes`` writes into one ``clustering_results.csv``, and
#: how each is drawn: ``(subdirectory, heading clause)``. Every clustering family is drawn once per
#: scope, into its own subtree.
#:
#: **``all`` is spelled by absence in both**, mirroring ``run_clustering.scope_suffix``: it is the
#: threat model and the headline, so it keeps the paths it has always had and no existing figure
#: moves when a scope is added.
#:
#: **The second field is documentation now, not drawing.** It was the clause a figure's title
#: ended with, and titles were removed; the subdirectory is what separates the scopes today, so a
#: reader tells them apart by the path rather than by a line on the image. It is kept
#: because it is the wording to reach for if a scope ever has to be visible on the figure again --
#: as a :func:`panel_note`, which is where per-panel facts belong.
#:
#: **The two are never put on one axis.** Each scope is a differently-shaped problem with its own
#: reference partitions, and the difference is not subtle: **BCubed's floor is a closed form of the
#: collection's authors-per-document ratio**. The all-singleton partition has precision 1 and
#: recall ``A / N`` -- the document-weighted mean of ``1 / m_author`` -- so its
#: ``F = 2(A/N) / (1 + A/N)`` is higher on ``unseen`` than on ``all`` on both corpora. Raw F is
#: therefore higher on ``unseen`` while the attack is actually slightly *weaker* there, since the
#: floor rises faster than the attack's own score does. The comparable quantity is each bar's
#: distance from the grey reference bar beside it, which is why those bars are on every panel.
#:
#: **What raises A/N is the loss of the heavy authors, not the absence of singletons.** It is true
#: that ``unseen`` has no single-document authors -- the corpus keeps no author with fewer than two
#: documents, so one absent from the known side has at least two inside the test quarter -- but
#: that mechanism has the wrong sign, and this note used to claim it: a one-document author has a
#: ratio far above the average, so removing them from ``all`` would take the floor *down*. The rise
#: instead comes from excluding everyone with known-side history, who are disproportionately the
#: heavy users.
CLUSTERING_SCOPES = {
    "all": ("", ""),
    "unseen": ("unseen", ", authors with no known-side history"),
}


def clustering_results(run: ClusteringRun, scope: str = "all") -> pd.DataFrame:
    """One run's ``clustering_results.csv``, restricted to one author scope.

    A directory that exists without the file is a run that was interrupted or is still going;
    every drawing routine below treats that as "no series", not as a failure.

    A clustering run attacks its collection under two author scopes (see
    :data:`CLUSTERING_SCOPES`) and writes both into this one file. A file written *before* that
    became true has no ``scope`` column and is entirely the ``all`` scope -- so it answers for ``all``
    and, correctly, holds nothing for any other scope. Returning the whole legacy table for a
    scope it predates would relabel one population as another, which is the one mistake this
    split exists to prevent.
    """
    path = run.directory / "clustering_results.csv"
    if not path.exists():
        return pd.DataFrame()
    table = pd.read_csv(path)
    if "scope" not in table.columns:
        return table if scope == "all" else pd.DataFrame()
    return table[table["scope"] == scope]


def clustering_style(name: str) -> str:
    """Colour for one clustering algorithm, fixed by its position in :data:`CLUSTERING_ALGORITHMS`.

    The **registry**, not :data:`CLUSTERING_DRAWN_ALGORITHMS`: an algorithm keeps the hue it has
    always had whether or not today's figures draw it, and a retirement moves nobody.
    """
    return series_style(CLUSTERING_ALGORITHM_SLOTS[name])


def clustering_scores(table: pd.DataFrame, names, column: str) -> list[tuple[str, float]]:
    """``(name, value)`` for each of ``names`` present in ``table``, in the order given.

    The order is the vocabulary's, never the CSV's, so a run that happened to write its rows in a
    different order does not reorder the bars -- the same reason every group above is sorted by
    colour slot rather than by what was found on disk.
    """
    return [(name, float(table.loc[table["algorithm"] == name, column].iloc[0]))
            for name in names if not table[table["algorithm"] == name].empty]


def clustering_feature_groups(runs: list[ClusteringRun]) -> list[tuple[str, list[ClusteringRun]]]:
    """Clustering runs grouped by feature -- one group per ``by_defense`` figure.

    :func:`method_groups`' counterpart, and the feature alone is the whole method here because a
    clustering attack has no attack axis. Grouping is what keeps the two comparison figures
    correct rather than merely tidy: both key their panels by *defense*, so two features of one
    corpus in one figure would collide on that key and draw one of them twice.
    """
    groups: dict[str, list[ClusteringRun]] = defaultdict(list)
    for run in runs:
        groups[run.feature].append(run)
    return [(feature, sorted(members, key=lambda run: DEFENSE_SLOTS[run.defense]))
            for feature, members in sorted(groups.items(),
                                           key=lambda item: FEATURES.index(item[0]))]


def plot_clustering_bcubed(dataset: str, feature: str, runs: list[ClusteringRun],
                           output_dir: Path, scope: str = "all") -> list[Path]:
    """BCubed F per algorithm, one panel per defense, with the reference partitions beside them.

    **The reference bars are the point of the figure.** An all-singleton partition -- one cluster
    per document, linking nothing at all -- scores F = 0.285 on WildChat and 0.112 on swe-chat,
    because BCubed precision is 1.0 when no two documents are ever put together. A bar chart of
    algorithms alone would therefore read as "0.49, quite good" where the honest statement is
    "0.49 against 0.285 for doing nothing". Drawing the references as bars on the same axis makes
    that a comparison of two bars rather than of a bar against a rule.

    One panel per defense, sharing a y axis, so the columns are directly comparable: the question
    a defended run answers is how far its bars fall from the undefended panel's. ``runs`` is one
    feature's, which is what makes "one panel per defense" a well-defined statement.
    """
    subdirectory = CLUSTERING_SCOPES[scope][0]
    defenses = [run.defense for run in sorted(runs, key=lambda run: DEFENSE_SLOTS[run.defense])]
    tables = {run.defense: clustering_results(run, scope) for run in runs}
    panels = [defense for defense in defenses if not tables[defense].empty]
    if not panels:
        return []

    figure, axes_list = plt.subplots(1, len(panels), figsize=(3.9 * len(panels), 4.4),
                                     sharey=True, squeeze=False)
    figure.patch.set_facecolor(SURFACE)
    rows = []
    # Over the bars that are actually drawn, not over the CSV: `clustering_results.csv` carries
    # rows nothing here draws (`baseline_model_owner`), and sizing the axis to a bar that is not
    # on it would leave dead space no reader could account for.
    tallest = max(value for defense in panels
                  for _, value in clustering_scores(
                      tables[defense],
                      CLUSTERING_DRAWN_ALGORITHMS + CLUSTERING_BASELINES, "bcubed_f"))
    for axes, defense in zip(axes_list[0], panels):
        table = tables[defense]
        measured = clustering_scores(table, CLUSTERING_DRAWN_ALGORITHMS, "bcubed_f")
        reference = clustering_scores(table, CLUSTERING_BASELINES, "bcubed_f")
        # The two groups share one categorical axis with a gap between them, rather than two axes
        # or two figures: they are the same measure on the same collection, and comparing them is
        # the whole job.
        positions = np.concatenate([
            np.arange(len(measured), dtype=float),
            np.arange(len(reference), dtype=float) + len(measured) + CLUSTERING_GROUP_GAP])
        colors = ([clustering_style(name) for name, _ in measured]
                  + [TEXT_MUTED] * len(reference))
        alphas = [1.0] * len(measured) + [CLUSTERING_BASELINE_ALPHA] * len(reference)
        values = [value for _, value in measured] + [value for _, value in reference]
        for position, value, color, alpha in zip(positions, values, colors, alphas):
            axes.bar(position, value, width=0.68, color=color, alpha=alpha, zorder=3)
            # On the cap, in text ink rather than the bar's colour: the bar carries identity, the
            # number is text. Eight bars is few enough to label every one.
            axes.text(position, value + 0.012, f"{value:.3f}", ha="center", va="bottom",
                      fontsize=FONT_ANNOTATION, color=TEXT_SECONDARY)

        style_axes(axes, "", "BCubed F" if defense == panels[0] else "",
                   DEFENSE_LABELS[defense])
        axes.set_xticks(positions)
        axes.set_xticklabels([CLUSTERING_LABELS[name] for name, _ in measured + reference],
                             rotation=30, ha="right", fontsize=FONT_TICK)
        axes.set_xlim(-0.7, positions[-1] + 0.7)
        # Headroom for the value labels on the caps, the panel note and the legend, which all
        # live in the band above the tallest bar. Taken from the tallest bar in the *figure* and
        # not the panel, because `sharey` means the last `set_ylim` wins for all of them anyway --
        # computing it once says so rather than leaving it to call order.
        axes.set_ylim(0, tallest * 1.30)
        quarter_ticks(axes.yaxis)
        axes.grid(False, axis="x")
        # The collection under attack, printed for the same reason the facet grids print their
        # in-set counts: BCubed's reference levels are functions of the collection's shape -- the
        # singleton baseline *is* the mean of 1/(documents by that author) -- so two panels are
        # only comparable at face value when they cover the same one.
        panel_note(axes, f"{int(table['n_documents'].iloc[0]):,} docs · "
                         f"{int(table['n_authors'].iloc[0]):,} authors")
        rows.append(pd.DataFrame({
            "defense": defense,
            "algorithm": [name for name, _ in measured + reference],
            "bcubed_f": values,
            "is_reference": [False] * len(measured) + [True] * len(reference)}))

    # One legend entry -- the algorithms are named on the ticks, so all the legend has to say is
    # what the grey means: that those bars are not an attack.
    #
    # **Below the whole figure, not inside the last panel.** It used to sit at that panel's upper
    # right, facing the panel note at the upper left, and the two collided once the text grew large
    # enough. A panel here is one fifth of the figure, so there is no in-axes corner wide enough for
    # a legend beside a note -- the strip below is, and it also stops the legend belonging to one
    # defense's panel when it describes all of them. The strip is measured in inches over the
    # figure height, exactly as `finish_facets` does it, because a legend is a fixed physical size.
    figure.legend([plt.Rectangle((0, 0), 1, 1, color=TEXT_MUTED,
                                 alpha=CLUSTERING_BASELINE_ALPHA)],
                  ["Reference partition"], loc="lower center", bbox_to_anchor=(0.5, 0.005),
                  frameon=False, fontsize=FONT_LEGEND, labelcolor=TEXT_PRIMARY)
    figure.tight_layout(rect=(0, 0.5 / figure.get_size_inches()[1], 1, 1))

    stem = output_dir / subdirectory / "bcubed" / "by_defense" / feature
    stem.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(rows, ignore_index=True).to_csv(stem.parent / f"{stem.name}.csv", index=False)
    return [save_figure(figure, stem)]


#: The variant whose points are drawn *beside* the pure-text ones on the timing figure, and the
#: fill that tells them apart. The pair is ``plain`` -> ``time``: the same graph with elapsed time
#: fused into the edge score, which is the one variant that changes what is measured rather than
#: how the space is learned, so the two are the same method on two edge definitions.
TIMING_VARIANT = "time"


def plot_clustering_precision_recall(dataset: str, feature: str, runs: list[ClusteringRun],
                                     output_dir: Path, scope: str = "all",
                                     timing: list[ClusteringRun] | None = None) -> list[Path]:
    """Each algorithm as one point in BCubed precision x recall, marker shape per defense.

    The figure that makes the trade legible, and that F alone hides: a method can buy precision by
    declining to cluster, which is exactly where the all-singleton corner sits (precision 1.0, and
    a recall equal to the mean of 1/|documents by that author|). HDBSCAN sits near it -- very high
    precision, low recall, because it leaves documents as noise -- and connected components at the
    opposite corner. Colour carries the algorithm, so it still follows the entity; shape carries
    the defense; **size carries nothing** -- see :data:`MARKER_SIZE_SCALE`, which is what makes
    that true across shapes.

    ``timing`` turns this into the figure's second form, at ``precision_recall/timing/``: the same
    plot with each method's :data:`TIMING_VARIANT` run drawn beside its pure-text one, **hollow**
    (the surface showing through a coloured outline), and a segment joining the pair. Fill becomes
    a fourth channel and it is the right one for this: the two points are the *same* algorithm
    under the *same* defense, so they must share colour and shape, and what the reader wants is
    the vector between them -- which way timing moved that method, and how far. The segment is
    drawn in the algorithm's colour under both marks, so a pair reads as one object.

    The dashed iso-F contours (:data:`CLUSTERING_ISO_F_LEVELS`) are the frame of reference, each
    labelled with its level where it leaves the right-hand edge: two points on one contour scored
    the same F with different trades, and a point between two contours can be read off them.
    """
    subdirectory = CLUSTERING_SCOPES[scope][0]
    tables = {run.defense: clustering_results(run, scope) for run in runs}
    defenses = [run.defense for run in sorted(runs, key=lambda run: DEFENSE_SLOTS[run.defense])
                if not tables[run.defense].empty]
    if not defenses:
        return []
    # A defense with no timing run simply draws its pure-text point alone, which is the honest
    # reading -- there is no pair to show a direction for.
    fused = {run.defense: clustering_results(run, scope) for run in (timing or [])}
    fused = {defense: table for defense, table in fused.items() if not table.empty}
    if timing is not None and not fused:
        return []
    # Shape per defense, from the cross-figure registry rather than from this figure's running
    # order -- see `defense_markers`. It is what lets a reader carry a mark from one corpus's
    # figure to the other's.
    markers = defense_markers(defenses)

    # Sized so the legend block clears the data. The legend is a **fixed physical size** (set by
    # its longest label) while the axes scale with the figure, so the question is what fraction of
    # the plot it covers -- sized to clear the highest-recall points on both corpora with margin.
    #
    # **The timing form needs more**, because its third legend block is taller *and* its points
    # reach further right, so it gets a larger canvas.
    #
    # **What would break either is a longer defense label, not another one**: rows add height, and
    # the width is the longest label. A future `styleremix_openanon` arm would want re-measuring.
    figure, axes = plt.subplots(figsize=(7.6, 6.9) if fused else (6.6, 5.9))
    figure.patch.set_facecolor(SURFACE)

    # Markers are **solid and all one size**. Both were the other way round before, and the reason
    # is worth keeping because it is what this now trades away: some defenses land almost on top of
    # each other, and open outlines of stepped sizes let a coincident pair read as nested rings,
    # where solid marks of one size occlude each other completely. **A defense that barely moves
    # the result can now hide under the arm it barely moved.** The mitigations if that bites: a
    # jitter, a small alpha, or the old ladder back.
    #
    # Size carries nothing now, which is the gain: shape means defense, colour means algorithm,
    # and `MARKER_SIZE_SCALE` evens the shapes out so a square does not read as a bigger result
    # than the triangle beside it.
    rows = []
    for defense in defenses:
        table = tables[defense]
        marker = markers[defense]
        size = CLUSTERING_PR_MARKER_SIZE * MARKER_SIZE_SCALE[marker]
        for name, _ in clustering_scores(table, CLUSTERING_DRAWN_ALGORITHMS, "bcubed_f"):
            row = table[table["algorithm"] == name]
            recall, precision = float(row["bcubed_recall"].iloc[0]), \
                float(row["bcubed_precision"].iloc[0])
            colour = clustering_style(name)
            pair = fused.get(defense, pd.DataFrame())
            pair = pair[pair["algorithm"] == name] if not pair.empty else pair
            if not pair.empty:
                # Segment first, so both marks sit on top of it and the pair reads as one object.
                fused_recall = float(pair["bcubed_recall"].iloc[0])
                fused_precision = float(pair["bcubed_precision"].iloc[0])
                axes.plot([recall, fused_recall], [precision, fused_precision], color=colour,
                          linewidth=1.2, alpha=0.55, solid_capstyle="round", zorder=3)
                # Hollow: the surface shows through, so the outline is the whole mark and the two
                # ends of a pair are told apart by fill alone -- same colour, same shape.
                axes.plot(fused_recall, fused_precision, marker=marker, markersize=size,
                          markerfacecolor=SURFACE, markeredgecolor=colour, markeredgewidth=1.6,
                          linestyle="none", zorder=4)
                rows.append({"defense": defense, "algorithm": name,
                             "variant": TIMING_VARIANT, "bcubed_recall": fused_recall,
                             "bcubed_precision": fused_precision,
                             "bcubed_f": float(pair["bcubed_f"].iloc[0]), "is_reference": False})
            axes.plot(recall, precision, marker=marker, markersize=size,
                      color=colour, markeredgewidth=0, linestyle="none", zorder=4)
            rows.append({"defense": defense, "algorithm": name, "variant": PLAIN_VARIANT,
                         "bcubed_recall": recall, "bcubed_precision": precision,
                         "bcubed_f": float(row["bcubed_f"].iloc[0]), "is_reference": False})

    # Offsets chosen per baseline rather than shared: all three sit against an edge of the unit
    # square, and a single offset direction pushes at least one of them into the data. Singletons
    # are at precision 1.0 (top edge) and random sits alone near the origin corner, so both take
    # their label **centred directly above the mark** -- nothing sits above either, and centred is
    # what reads as "this label belongs to this point". One cluster
    # is the exception: it is pinned to the right edge at recall 1.0, where a centred label would
    # run off the figure, so it keeps its corner offset.
    offsets = {"baseline_singleton": (0, 8), "baseline_single_cluster": (-8, 8),
               "baseline_random": (0, 8)}
    alignment = {"baseline_singleton": "center", "baseline_single_cluster": "right",
                 "baseline_random": "center"}
    base = tables[defenses[0]]
    for name, _ in clustering_scores(base, CLUSTERING_PR_BASELINES, "bcubed_f"):
        row = base[base["algorithm"] == name]
        recall, precision = float(row["bcubed_recall"].iloc[0]), \
            float(row["bcubed_precision"].iloc[0])
        axes.plot(recall, precision, marker="x", markersize=MARKER_SIZE, color=TEXT_MUTED,
                  linestyle="none", zorder=3)
        axes.annotate(CLUSTERING_LABELS[name], (recall, precision), textcoords="offset points",
                      xytext=offsets[name], ha=alignment[name], fontsize=FONT_ANNOTATION,
                      color=TEXT_MUTED)
        rows.append({"defense": defenses[0], "algorithm": name, "variant": PLAIN_VARIANT,
                     "bcubed_recall": recall, "bcubed_precision": precision,
                     "bcubed_f": float(row["bcubed_f"].iloc[0]), "is_reference": True})

    # Iso-F contours, so a reader can see which points are equivalent trades rather than guessing.
    # **Dashed and labelled**: the dash is this file's mark for a line that is not a measurement,
    # which a contour of the metric's own geometry certainly is not, and `AXIS` rather than `GRID`
    # because a dashed hairline at the grid's weight disappears -- these carry a number, so they
    # have to be readable. The label goes where the contour leaves the axes on the right, at
    # ``p = F / (2 - F)`` (set ``r = 1`` in the contour below), which is inside the unit square for
    # every level, so every contour gets one.
    #
    # **Each contour's own top end is forced into its grid**, and without it the low levels
    # visibly failed to reach precision 1.0. A contour reaches it at ``r = F / (2 - F)`` and has
    # its asymptote at ``r = F / 2``, so the whole run from p = 1 down to the first sampled point
    # is only ``F^2 / (2(2 - F))`` wide -- **quadratic in F**, so at low F no sample lands in it
    # from a uniform grid at all. Parametrising by precision instead would fix the top and lose
    # the tail; one exact point costs nothing and puts every contour on the top edge where it
    # belongs.
    base_grid = np.linspace(0.01, 1.0, 200)
    for level in CLUSTERING_ISO_F_LEVELS:
        grid = np.union1d(base_grid, [level / (2 - level)])
        precision = level * grid / (2 * grid - level)
        # The tolerance is what makes the point above actually land: at some levels the end point
        # evaluates to just over 1.0 in floating point, so a bare `<= 1.0` drops the one sample
        # this exists to add. Clamping the kept values is safe because nothing else comes close to
        # the top from below.
        usable = (precision > 0) & (precision <= 1.0 + ISO_F_EDGE_TOLERANCE)
        axes.plot(grid[usable], np.minimum(precision[usable], 1.0), color=AXIS, linewidth=0.9,
                  linestyle=BASELINE_DASH, zorder=1)
        axes.annotate(f"F={level:g}", (1.0, level / (2 - level)), textcoords="offset points",
                      xytext=(5, 0), ha="left", va="center", fontsize=FONT_ANNOTATION,
                      color=TEXT_MUTED,
                      annotation_clip=False)   # the label sits outside the axes, by design

    style_axes(axes, "BCubed recall", "BCubed precision", "")
    # A hair past the unit square on both axes: the one-cluster reference sits at recall exactly
    # 1.0 and the singleton one at precision exactly 1.0, so a hard limit halves both markers.
    axes.set_xlim(0, 1.02)
    axes.set_ylim(0, 1.05)
    quarter_ticks(axes.xaxis)
    quarter_ticks(axes.yaxis)
    # **One unit of recall is one unit of precision.** Both axes span [0, 1], so without this the
    # box is whatever shape `figsize` leaves after the labels, stretching one axis and rendering
    # the unit square as a rectangle. That is not cosmetic here: the iso-F contours are curves in
    # the P x R plane and a reader judges a point by how far it sits from the top-right corner,
    # both of which an unequal scale distorts.
    axes.set_aspect("equal", adjustable="box")

    # **Two legends, one per channel**, the same construction and for the same reason as a
    # ``by_attack`` figure's: a reader needs both to decode one marker, and a single list
    # mixing them reads as one vocabulary -- with the defenses' neutral-ink marks filed under the
    # algorithms' colours, "No defense" looks like another algorithm.
    #
    # The keys must match the marks exactly -- solid, and through the same per-shape size
    # correction -- or a reader matching a mark back to the legend gets the wrong defense.
    drawn = {name for name in CLUSTERING_DRAWN_ALGORITHMS
             if any(not tables[defense][tables[defense]["algorithm"] == name].empty
                    for defense in defenses)}
    by_algorithm = [plt.Line2D([], [], marker="o", linestyle="none",
                               markersize=CLUSTERING_PR_MARKER_SIZE * MARKER_SIZE_SCALE["o"],
                               color=clustering_style(name), markeredgewidth=0,
                               label=CLUSTERING_LABELS[name])
                    for name in CLUSTERING_DRAWN_ALGORITHMS if name in drawn]
    by_defense = [plt.Line2D([], [], marker=markers[defense], linestyle="none",
                             markersize=(CLUSTERING_PR_MARKER_SIZE
                                         * MARKER_SIZE_SCALE[markers[defense]]),
                             color=TEXT_SECONDARY, markeredgewidth=0,
                             label=DEFENSE_LABELS[defense])
                  for defense in defenses]
    # **Inside the axes, top right**, one column each and Algorithm above Defense. That corner is
    # the one place a precision/recall figure is reliably empty: it is high precision *and* high
    # recall at once, which is the ideal no partition here comes near. The three baselines occupy
    # the other three corners, which is why the block was under the axes before.
    first = add_legend(axes, handles=by_algorithm, ncol=1, title="Algorithm",
                       loc="upper right", bbox_to_anchor=(0.995, 0.995))
    # A second `.legend()` call on an axes *replaces* the first, so each block has to be adopted
    # explicitly; every anchor is measured off the block above it, because their heights grow a
    # row per entry and none of them is known before the figure has been laid out once.
    axes.add_artist(first)
    second = add_legend(axes, handles=by_defense, ncol=1, title="Defense", loc="upper center",
                        bbox_to_anchor=tuple(stack_below(figure, axes, first)),
                        borderaxespad=0.0)  # honour the measured anchor, do not re-pad off it
    if fused:
        # The fill channel needs naming or the hollow marks are unexplained. Neutral ink, like the
        # shape keys: this block is about how a mark is drawn, not about any one algorithm.
        axes.add_artist(second)
        by_variant = [
            plt.Line2D([], [], marker="o", linestyle="none", color=TEXT_SECONDARY,
                       markersize=CLUSTERING_PR_MARKER_SIZE * MARKER_SIZE_SCALE["o"],
                       markeredgewidth=0, label=CLUSTERING_VARIANT_LABELS[PLAIN_VARIANT]),
            plt.Line2D([], [], marker="o", linestyle="none", markerfacecolor=SURFACE,
                       markersize=CLUSTERING_PR_MARKER_SIZE * MARKER_SIZE_SCALE["o"],
                       markeredgecolor=TEXT_SECONDARY, markeredgewidth=1.6,
                       label=CLUSTERING_VARIANT_LABELS[TIMING_VARIANT])]
        add_legend(axes, handles=by_variant, ncol=1, title="Edge score", loc="upper center",
                   bbox_to_anchor=tuple(stack_below(figure, axes, second)), borderaxespad=0.0)

    stem = (output_dir / subdirectory / "precision_recall"
            / ("timing" if fused else "by_algorithm") / feature)
    stem.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(stem.parent / f"{stem.name}.csv", index=False)
    return [save_figure(figure, stem)]


#: Height of one row of a variants dumbbell chart, in inches. These charts are read as a ranking,
#: and the rows are single marks rather than anything with internal structure, so the pitch only
#: has to keep two adjacent rows apart.
VARIANT_ROW_HEIGHT = 0.26

#: Row label for the pure-text run drawn beside the variants -- the method with nothing applied
#: to its graph, which is what every other row has to be read against.
BASELINE_VARIANT_LABEL = CLUSTERING_VARIANT_LABELS[PLAIN_VARIANT]


def clustering_prefix(dataset: str, scope: str) -> str:
    """Where a clustering figure of this corpus and author scope is filed, as a plan name.

    The ``all`` scope is spelled by *absence* -- `CLUSTERING_SCOPES[scope][0]` is empty for it --
    so every path that family had before the scopes were added is unchanged.
    """
    return "/".join(part for part in (f"{dataset}/clustering", CLUSTERING_SCOPES[scope][0]) if part)


def plot_clustering_variants(dataset: str, defense: str, runs: list[ClusteringRun],
                             base: ClusteringRun | None, output_dir: Path,
                             scope: str = "all") -> list[Path]:
    """Tuning-slice against test-slice BCubed F, one row per representation strategy.

    **One figure per (dataset, defense).** The variant directories used to be parsed without their
    defense, so every defended run of one corpus collapsed onto the same key and would have drawn
    several identically-labelled rows on one figure. The four-part name carries the defense, so
    the figure holds it fixed -- in its file name, since no figure here carries a title -- and
    each defense's variants are read against the pure-text run of *that* defense rather than
    against the undefended one.

    **The gap between the two dots is the finding, not the level of either.** Each strategy was
    selected by searching a labelled tuning slice, so its tuning score is the number that decided
    it was worth running -- and the honest measure of what it is worth is the test score beside it.
    That gap is *larger for the strategies that looked best*, which is exactly what a search over
    one labelled slice produces.

    Both numbers come from one row of one ``clustering_results.csv`` (``tuning_bcubed_f`` and
    ``bcubed_f``), so they are the same configuration on two slices rather than a best-of-many
    compared against a single run -- the comparison a development sweep cannot make about itself.

    A dumbbell rather than paired bars: the quantity a reader needs is the *change*, and two bars
    per row make that a subtraction done by eye. Rows are ordered by the strategy vocabulary, not
    by score, so a strategy sits in the same place on both corpora's figures.
    """
    # The pure-text run belongs on this figure as a row, not as an absent reference: it writes the
    # same two columns from the same file, and on swe-chat it BEATS two of the three strategies --
    # a fact that is invisible if the figure only draws what was added to it.
    subdirectory = CLUSTERING_SCOPES[scope][0]
    candidates = ([(BASELINE_VARIANT_LABEL, base)] if base is not None else []) + \
                 [(run.variant_label, run) for run in runs]
    rows, drawn = [], []
    for label, run in candidates:
        record = connected_variant_record(run, scope)
        if record is None:
            continue
        drawn.append((label, float(record["tuning_bcubed_f"]), float(record["bcubed_f"])))
        rows.append(variant_row(dataset, run, label, record))
    if len(drawn) < 2:
        return []                                    # one row is a table, not a figure

    # The defense is in the file name, which is the only place it is now: one figure per
    # (dataset, defense), so two defenses' variants cannot land on one path and overwrite it.
    stem = output_dir / subdirectory / "variants" / f"{defense}_tuning_vs_test"
    return draw_variant_dumbbells(drawn, rows, stem)


#: Variants left off :func:`plot_clustering_variants_by_defense`, the one chart that puts every
#: defense's strategies on one axis. ``contrastive_time`` is the fused
#: arm -- a learned projection AND a timing channel -- so it is the row that changes two things
#: at once, and dropping it leaves that chart comparing one intervention per row. Every one of
#: them keeps its own per-defense figure, where the full set is drawn.
COMBINED_VARIANTS_OMITTED = ("contrastive_time",)


def connected_variant_record(run: ClusteringRun, scope: str):
    """One clustering run's ``connected`` row, or ``None`` if it has no tuning score.

    Both variant figures read the same row of the same file: ``tuning_bcubed_f`` and ``bcubed_f``
    are the *same configuration* scored on the tuning slice and on the test slice, which is the
    comparison a development sweep cannot make about itself. A run with no finite tuning score was
    not selected by that search and has nothing to say here.
    """
    table = clustering_results(run, scope)
    if table.empty or "connected" not in set(table["algorithm"]):
        return None
    record = table.loc[table["algorithm"] == "connected"].iloc[0]
    if not np.isfinite(record.get("tuning_bcubed_f", float("nan"))):
        return None
    return record


def variant_row(dataset: str, run: ClusteringRun, label: str, record) -> dict:
    """One row of a variants figure's companion CSV.

    Carries the defense and the variant as their own columns as well as the drawn label, so the
    per-defense CSVs and the combined one have the same schema and can be concatenated.
    """
    return {"dataset": dataset, "defense": run.defense,
            "variant": run.variant, "label": label,
            "algorithm": "connected",
            "tuning_bcubed_f": float(record["tuning_bcubed_f"]),
            "test_bcubed_f": float(record["bcubed_f"]),
            "bcubed_precision": float(record["bcubed_precision"]),
            "bcubed_recall": float(record["bcubed_recall"]),
            "shrinkage": float(record["tuning_bcubed_f"]) - float(record["bcubed_f"]),
            "hyperparameters": record.get("hyperparameters", "")}


def draw_variant_dumbbells(drawn: list[tuple[str, float, float]], rows: list[dict],
                           stem: Path) -> list[Path]:
    """The dumbbell chart itself: one row per ``(label, tuning score, test score)``.

    Shared by the per-defense figure and the combined one so the two are the same chart over
    different rows -- same colours, same arrowhead, same direct labels, same geometry -- and a
    reader flipping between them is comparing rows rather than decoding two designs.

    Rows are drawn top-down in the order given: the caller decides whether that is the strategy
    vocabulary (per defense, so a strategy sits in the same place on both corpora) or a sort.

    **The row pitch is :data:`VARIANT_ROW_HEIGHT`.** At this pitch a value label set below a dot
    would land on the row beneath it, which is why the labels sit on the *outer* side of the test
    dot instead -- the side the arrow points -- vertically centred on the row, so the label is
    beside the mark it names, never over the connector, and never in another row.
    """
    figure, axes = plt.subplots(figsize=(8.2, VARIANT_ROW_HEIGHT * len(drawn) + 1.9))
    figure.patch.set_facecolor(SURFACE)
    positions = np.arange(len(drawn))[::-1]          # first strategy at the top
    tuning_colour, test_colour = series_style(0), series_style(1)

    for position, (_, tuning, test) in zip(positions, drawn):
        axes.plot([test, tuning], [position, position], color=TEXT_MUTED, linewidth=2.0,
                  zorder=1, solid_capstyle="round")
        # An arrowhead at the midpoint, pointing train -> test. A dumbbell
        # says how far the two scores are apart but not which way round they are, and here the
        # direction IS the finding: a head pointing left is a strategy that lost what the search
        # credited it with. It rides on the connector in the connector's own ink -- the endpoint
        # colours are the two slices, and a third colour here would read as a third quantity.
        axes.plot([(tuning + test) / 2], [position], marker=">" if test > tuning else "<",
                  markersize=MARKER_SIZE * MARKER_SIZE_SCALE[">"], color=TEXT_MUTED,
                  markeredgewidth=0, zorder=2)
        # Surface ring on both marks: they overlap the connector and, on a small gap, each other.
        for value, colour in ((tuning, tuning_colour), (test, test_colour)):
            axes.plot([value], [position], marker="o", markersize=MARKER_SIZE + 2,
                      color=colour, markeredgecolor=SURFACE, markeredgewidth=2.0, zorder=3)
        # Direct-label the TEST value only. Labelling both would put a number on every mark, and
        # the test number is the one a reader takes away. Set outside the pair, on the side the
        # dumbbell points: the inner side is the connector, and above or below is the next row.
        outward = 1 if test >= tuning else -1
        axes.annotate(f"{test:.3f}", (test, position), textcoords="offset points",
                      xytext=(9 * outward, 0), ha="left" if outward > 0 else "right",
                      va="center", color=TEXT_PRIMARY, fontsize=FONT_ANNOTATION)

    axes.set_yticks(positions, [label for label, _, _ in drawn])
    axes.tick_params(axis="y", length=0)
    span = [value for _, tuning, test in drawn for value in (tuning, test)]
    # Wider than the 0.06 it was: the value labels now sit outside the marks rather than under
    # them, so the extreme rows need room for a number as well as for a dot.
    margin = 0.14 * (max(span) - min(span) + 1e-9)
    axes.set_xlim(min(span) - margin - 0.01, max(span) + margin)
    axes.set_ylim(-0.55, len(drawn) - 0.45)
    style_axes(axes, "BCubed F (connected components)", "", "")
    axes.grid(axis="y", visible=False)

    # Above the axes, horizontally: the data occupy a different corner on each corpus, so any
    # in-axes corner collides on one of them -- it did, with the bottom row's value label.
    add_legend(axes, handles=[
        plt.Line2D([], [], marker="o", linestyle="none", markersize=MARKER_SIZE + 2,
                   color=tuning_colour, markeredgecolor=SURFACE, markeredgewidth=2.0),
        plt.Line2D([], [], marker="o", linestyle="none", markersize=MARKER_SIZE + 2,
                   color=test_colour, markeredgecolor=SURFACE, markeredgewidth=2.0)],
        labels=["Train", "Test"],
        loc="lower left", bbox_to_anchor=(0.0, 1.005), ncol=2, borderaxespad=0.0,
        handletextpad=0.4, columnspacing=1.6)
    figure.tight_layout()

    stem.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(stem.parent / f"{stem.name}.csv", index=False)
    return [save_figure(figure, stem)]


def plot_clustering_variants_by_defense(dataset: str, runs: list[ClusteringRun],
                                        plain: list[ClusteringRun], output_dir: Path,
                                        scope: str = "all") -> list[Path]:
    """Every defense's strategies on one chart, ordered by what they scored on the test slice.

    :func:`plot_clustering_variants` holds the defense fixed and asks what each strategy added to
    it; this asks the question the other way round -- across the whole defense vocabulary, which
    (defense, strategy) pairs actually cluster, and does the ordering the tuning slice implies
    survive? Sorting by the **test** score is what makes that legible: the rows are a ranking, and
    a train dot sitting to the right of its test dot is a pair the search over-credited.

    **Ascending, so the weakest pair is at the top and the strongest at the bottom** -- the reading
    order of the file, and it puts the rows a defense is judged by (the ones that cluster best)
    nearest the x axis they are read against.

    **The fused contrastive+timing arm is left off** (:data:`COMBINED_VARIANTS_OMITTED`): it moves
    two things at once, and on a chart whose rows are meant to be one intervention each it cannot
    be read against either single-channel row. It is still drawn, with everything else, on that
    defense's own figure.

    **Every row's train dot is one of two numbers, and that is the experiment rather than a
    drawing fault**: ``run_clustering.py``'s ``--known-defense`` defaults to ``base``, so on a
    defended cell the tuning slice is built from *undefended* vectors. The train dot therefore
    carries the strategy and nothing else, identical across every defense sharing it -- so read
    the chart as a ranking of test scores against two shared reference marks rather than as ten
    independent pairs. It is also why the arrow direction is a property of the *pair*: every
    timing-fused row that beats its reference points right, every pure-text row points left.

    One figure per (dataset, scope), so the defense rides in the row label rather than in the
    file name -- which is what the file name says by being ``all_defenses``.
    """
    subdirectory = CLUSTERING_SCOPES[scope][0]
    members = [run for run in list(plain) + list(runs)
               if run.dataset == dataset and run.variant not in COMBINED_VARIANTS_OMITTED]
    rows, drawn = [], []
    # Vocabulary order first, so ties break the way every other figure here orders defenses.
    for run in sorted(members, key=lambda run: (DEFENSE_SLOTS[run.defense], run.variant)):
        record = connected_variant_record(run, scope)
        if record is None:
            continue
        # No separator between the two: with the variants named for the timing channel the label
        # reads as one phrase -- "StyleRemix w/ timing" -- where a middle dot would punctuate a
        # sentence that does not need it.
        label = f"{run.defense_label} {run.variant_label}"
        drawn.append((label, float(record["tuning_bcubed_f"]), float(record["bcubed_f"])))
        rows.append(variant_row(dataset, run, label, record))
    if len(drawn) < 2:
        return []
    order = sorted(range(len(drawn)), key=lambda index: drawn[index][2])   # ascending test F
    drawn = [drawn[index] for index in order]
    rows = [rows[index] for index in order]
    stem = output_dir / subdirectory / "variants" / "all_defenses_tuning_vs_test"
    return draw_variant_dumbbells(drawn, rows, stem)


# --- driver ------------------------------------------------------------------

#: The top-level folders a figure can land in, and the whole vocabulary of ``--families``.
#:
#: ``openset`` is drawn from :data:`RESULTS_DIR` through :data:`CURVE_TYPES` and is one folder in
#: the tree holding its three subfamilies (``dirfar``, ``dirfar_by_words``, ``top_k_cmc``);
#: ``clustering`` reads :data:`CLUSTERING_DIR` instead.
FIGURE_FAMILIES = ("clustering", "openset")


def figure_family(name: str) -> str:
    """Which of :data:`FIGURE_FAMILIES` a planned figure belongs to, from its name alone.

    A :class:`PlannedFigure`'s name *is* its output path without an extension, so the family is
    already a segment of it -- the first one below the dataset -- and nothing has to be recorded
    alongside the plan to recover it.
    """
    return name.partition("/")[2].split("/")[0]


def figure_datasets(figure: "PlannedFigure") -> frozenset:
    """Which corpus a planned figure describes, which is what ``--datasets`` selects on.

    Every figure is filed under its dataset, so the first segment of the name answers it -- the
    same property :func:`figure_family` reads one segment further in, and for the same reason: the
    path is the figure's identity, so nothing has to be recorded beside the plan.
    """
    return frozenset((figure.name.partition("/")[0],))


def parse_args() -> argparse.Namespace:
    """How finely the uncertainty is estimated, which figures are drawn, and how fast.

    There is no window flag any more, and nothing here chooses what is measured. The experiment
    fixes that: ``run_experiment.py`` holds out the final :data:`TEST_FRACTION` of the corpus
    from every known side, so which documents a comparison is made on is a property of the run
    rather than a decision taken at plot time. What is left to choose is how finely the
    uncertainty is estimated (``--bootstrap``), how much of the tree is drawn (``--families``,
    ``--datasets``, ``--png``) and how fast (``--jobs``).
    """
    parser = argparse.ArgumentParser(
        description="Draw every figure in the project from experiments/results/.")
    parser.add_argument("--bands", action="store_true",
                        help=f"Draw the 95%% confidence bands, at {BOOTSTRAP_REPLICATES} "
                             f"replicates. **Off by default**: a band per series is a lot of ink "
                             f"on a panel carrying several, and the bootstrap is most of what a "
                             f"curve build costs. Same as --bootstrap {BOOTSTRAP_REPLICATES}.")
    parser.add_argument("--bootstrap", type=int, default=None, metavar="N",
                        help=f"Replicates behind every band, resampling *users* with replacement. "
                             f"0 (the default) draws no band; --bands is shorthand for "
                             f"{BOOTSTRAP_REPLICATES}, which is the number to publish. One draw "
                             f"is shared by every configuration and run of a dataset, so a "
                             f"replicate that drops a user drops them from every panel at once. "
                             f"Lower it for a fast redraw, not for a figure anyone will read.")
    parser.add_argument("--jobs", type=int, default=default_workers(), metavar="N",
                        help=f"Worker processes for the curve and figure passes (default: "
                             f"{default_workers()}, capped rather than every core because each "
                             f"worker holds its own replicate weight matrix). 1 runs everything "
                             f"in this process, which is what to use when a drawing routine "
                             f"raises -- a traceback from a forked child loses its outer frames.")
    parser.add_argument("--png", action="store_true",
                        help="Also write a 200 dpi PNG beside every PDF. Off by default: the "
                             "PNGs cost more to render than the PDFs they accompany and only the "
                             "PDF goes into a paper.")
    parser.add_argument("--families", nargs="+", choices=FIGURE_FAMILIES, metavar="FAMILY",
                        help=f"Draw only these families and leave the rest of the tree alone. The "
                             f"names are the folders the figures land in: "
                             f"{', '.join(FIGURE_FAMILIES)}. 'openset' covers all three of its "
                             f"subfamilies. Unselected figures keep their manifest entries, so a "
                             f"filtered sweep is a subset of a full one rather than a different "
                             f"one -- which makes this the flag for redrawing one family after "
                             f"editing this file, an edit that marks every figure in the tree "
                             f"stale at once.")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=None,
                        metavar="DATASET",
                        help=f"Draw only these corpora and leave the rest of the tree alone "
                             f"({', '.join(DATASETS)}). Like --families it narrows what is "
                             f"*drawn*, never what is planned or measured, and it cannot move a "
                             f"number: a corpus's figures carry only that corpus's runs, the "
                             f"bootstrap draw is per corpus, and build_curves still reads every "
                             f"run of a corpus it builds any of. So it is not in the cache key -- "
                             f"a narrowed sweep writes what a full one would and leaves every "
                             f"unselected figure's manifest entry standing.")
    parser.add_argument("--attacks", nargs="+", choices=(*ATTACKS, "all"), default=None,
                        metavar="ATTACK",
                        help=f"Which attacks to draw (default: "
                             f"{', '.join(DEFAULT_ATTACKS)}; 'all' draws every one on disk). "
                             f"A run whose attack is not selected is left out of every figure "
                             f"and every legend, which is what keeps a by_attack panel readable "
                             f"now that {len(ATTACKS)} attacks are registered. It narrows the "
                             f"figures only -- no result is touched, so widening it again costs "
                             f"a redraw and nothing else.")
    parser.add_argument("--features", nargs="+", choices=FEATURES, default=None,
                        metavar="FEATURE",
                        help="Which features to draw (default: every feature on disk). The "
                             "feature counterpart of --attacks, on the same terms: a run whose "
                             "feature is not selected is left out of every figure and legend, and "
                             "no result is touched. Colours do not move, since a feature's hue is "
                             "fixed by its FEATURES index rather than by what else is drawn.")
    parser.add_argument("--force", action="store_true",
                        help="Redraw every figure, ignoring the cache. Not needed after editing "
                             "this file (the cache keys on its contents) or after a re-run (they "
                             "key on the predictions files' size and mtime) -- reach for it when "
                             "something outside both changed, such as the corpus parquet the "
                             "baselines and the word-count split read.")
    return parser.parse_args()


# --- what has already been drawn ---------------------------------------------
#
# A sweep is dominated by work that did not need doing: adding one attack leaves most of the tree
# untouched, and re-running the script after editing a caption redraws the whole tree to change one
# figure. The cache is a manifest of what the last sweep drew and of everything that decided it, so a
# figure is skipped exactly when nothing it depends on has moved.
#
# WHAT A FIGURE DEPENDS ON, and every one of these is in its key:
#
#   * the runs it draws -- by name, so a *new* run in a group changes the key of the figures that
#     group feeds and of no others. This is the asymmetry that makes the cache worth having:
#     a new attack rewrites every `by_attack/` figure, because each gains a series, but only adds
#     one `by_defense/` figure and leaves the rest of that view alone.
#   * each of those runs' predictions files, by size and mtime. Contents are not hashed: the files
#     are hundreds of MB and `make` has been right about this for fifty years. A touched file
#     redraws, which is conservative in the harmless direction.
#   * this file, by content digest. Any edit to any drawing routine invalidates everything, which
#     is the property that makes a cached figure trustworthy -- the alternative is a tree of
#     figures drawn by code that no longer exists, and no way to tell which.
#   * the settings that change what is drawn: `--bootstrap`, `--png`.
#
# WHAT IT DOES NOT COVER, deliberately: the corpus parquet in `data/hf/`. The baselines and the
# word-count split read it, but it is a build artefact that changes
# far less often than the results do, and stat-ing it on every sweep would tie the plots' cache to
# a directory that is legitimately swapped between `data/dist/` and `data/hf/`. `--force` is the
# answer when it moves.

#: The manifest, kept **inside** the plots tree rather than in `experiments/.cache/`, so that
#: deleting the figures deletes the memory of them. The two must not be able to disagree.
CACHE_FILE = ".plot_cache.json"


def source_digest() -> str:
    """Content digest of this file: what makes an edit to any drawing routine invalidate the tree.

    Coarse on purpose. Attributing figures to the functions that draw them would be finer and
    would be wrong the first time a shared helper changed -- and the failure mode it would buy is
    a figure that silently disagrees with the code that claims to have drawn it.
    """
    return hashlib.sha1(Path(__file__).read_bytes()).hexdigest()[:16]


def run_digest(run: Run) -> str:
    """One run's inputs as a digest of its CSVs' names, sizes and mtimes.

    Every file in the directory counts, not only ``predictions_*.csv``: ``rolling_results.csv``
    feeds the known-pool counts, and a run whose files disagree with each
    other is one nobody should be reading a cached figure of.
    """
    parts = sorted(f"{path.name}:{path.stat().st_size}:{path.stat().st_mtime_ns}"
                   for path in run.directory.glob("*.csv"))
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:16]


@dataclass(frozen=True)
class PlannedFigure:
    """One figure the sweep intends to draw, named and priced before any curve is built.

    ``name`` is its stable identity and doubles as the manifest key -- it is the output path
    without an extension, so a line of the manifest can be read against the tree by eye.
    ``runs`` is everything whose predictions decide the figure's content, which is what the cache
    compares. ``build`` turns the built curves into a ``run_jobs`` triple; it is deferred because
    planning happens *before* building, which is the whole point -- a figure that is up to date
    contributes nothing to what has to be built.

    ``builds`` is the runs whose **curves** this figure needs, which is not always the runs whose
    data decides it. The clustering figures are the reason: they are stale when their own runs
    change, but read no curve family. Left at ``None`` it is ``runs``, which is the safe reading
    and what every comparison figure wants.

    ``reads_context`` is the coarser question ``builds`` cannot answer: does this figure read
    *anything* :func:`build_curves` produces? Every comparison figure answers yes, while
    every clustering figure answers no, because it opens its own CSVs when it is drawn. Only a
    ``True`` puts a run into the sweep's ``touched`` set, which is what lets ``--families
    clustering`` skip a dataset build whose every output it would throw away. Setting it
    wrongly is not a slow figure but a silently incomplete one, so it is ``True`` by default and
    turned off only where the drawing closure visibly ignores its ``ctx`` argument.
    """

    name: str
    runs: tuple
    build: object
    builds: tuple | None = None
    reads_context: bool = True

    @property
    def needs_curves(self) -> tuple:
        return self.runs if self.builds is None else self.builds

    def key(self, settings: str) -> str:
        payload = "|".join([settings, *(f"{run.directory.name}:{run_digest(run)}"
                                        for run in self.runs)])
        return hashlib.sha1(payload.encode()).hexdigest()[:16]


class PlotCache:
    """The manifest of what the last sweep drew, and the check against it.

    A figure is current when its key matches *and* every file the last sweep recorded for it is
    still on disk -- so deleting a PDF is enough to get it back, without `--force` and without
    knowing anything about this file.

    **A figure that produced no output is recorded, not forgotten.** Plenty of planned figures
    legitimately draw nothing: a word-count figure with no corpus parquet, a clustering scope a
    run has no rows for. Recording the empty result
    is what stops the sweep re-deciding that every time.
    """

    def __init__(self, path: Path, settings: str, force: bool = False) -> None:
        self.path, self.settings, self.force = path, settings, force
        self.entries: dict[str, dict] = {}
        # Loaded even under ``--force``, which suppresses the *check* rather than the memory: the
        # manifest is rewritten from what this object holds, so discarding it here would make
        # ``--force --families <one>`` evict every family it did not draw and turn the next full
        # sweep into a cold one. A full ``--force`` sweep overwrites every entry anyway.
        if path.exists():
            try:
                self.entries = json.loads(path.read_text()).get("figures", {})
            except (OSError, ValueError):
                # A truncated manifest means a redraw, never a crash: it is a cache.
                print(f"  {path.name} is unreadable -- redrawing everything")

    def is_current(self, figure: PlannedFigure) -> bool:
        if self.force:
            return False
        entry = self.entries.get(figure.name)
        if entry is None or entry.get("key") != figure.key(self.settings):
            return False
        return all((self.path.parent / output).exists() for output in entry.get("outputs", ()))

    def record(self, figure: PlannedFigure, outputs: list[Path]) -> None:
        self.entries[figure.name] = {
            "key": figure.key(self.settings),
            "outputs": sorted(str(path.relative_to(self.path.parent)) for path in outputs),
        }

    def save(self, planned: list[PlannedFigure]) -> None:
        """Write the manifest, pruned to what this sweep planned.

        Pruning is what keeps a figure that no longer exists -- a run deleted, a family retired --
        from sitting in the file forever. The figures it drew are left on disk: this script has
        never deleted a figure and guessing which orphans are wanted is not its job.

        ``planned`` is the **whole** plan even when ``--families`` narrowed what was drawn, which
        is the invariant that makes that flag a subset of a full sweep: pruning to the selection
        would evict every family it skipped and make the next full sweep redraw them.
        """
        names = {figure.name for figure in planned}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(
            {"settings": self.settings,
             "figures": {name: entry for name, entry in sorted(self.entries.items())
                         if name in names}},
            indent=1))


def run_curves(run: Run, open_tables: dict[str, pd.DataFrame],
               open_bootstrap: AuthorBootstrap) -> dict:
    """Every curve one run contributes, keyed by curve family then by configuration.

    The unit of work :func:`build_curves` hands to a worker. It is one *run* rather than one
    (run, configuration) cell because a cell is small enough that per-job overhead would start to
    show. ``open_tables`` holds every unknown document, in-set or not (see
    :func:`config_predictions`).
    """
    curves: dict[str, dict] = {family: {} for family in CURVE_FAMILIES}

    # The known pool per configuration, for the identification note's "cand." line: documents
    # at the document level, users at the author level.
    known_pool: dict[str, tuple[int, int]] = {}
    rolling_path = run.directory / "rolling_results.csv"
    if rolling_path.exists():
        rolling = pd.read_csv(rolling_path, usecols=["known_config", "n_known_docs",
                                                     "n_known_authors"])
        known_pool = {row.known_config: (int(row.n_known_docs), int(row.n_known_authors))
                      for row in rolling.drop_duplicates("known_config").itertuples()}

    for tag, table in open_tables.items():
        panel = PanelWeights(open_bootstrap, table["true_author"])
        # The one open-set family with a baseline, so the one that needs the known side's prior.
        # Built from *this* table's in-set rows rather than reusing the in-set loop's: the two
        # populations are filtered on different columns, and a baseline has to describe the
        # documents its own panel counts.
        open_baseline = config_baseline(run.dataset, tag,
                                        table.loc[table["author_in_known"].astype(bool),
                                                  "true_author"])
        curves["identification"][tag] = config_identification(table, panel, open_baseline)
        pool_docs, pool_users = known_pool.get(tag, (None, None))
        curves["identification"][tag].n_candidates = pool_docs
        # The author level is the same curves over one row per person. `author_table` collapses
        # the table and the weights follow it.
        people = author_table(table)
        people_panel = PanelWeights(open_bootstrap, people["true_author"])
        curves["identification_authors"][tag] = config_identification_authors(
            table, people, people_panel, open_baseline)
        curves["identification_authors"][tag].n_candidates = pool_users
        # The other slice of the same surface: threshold pinned, k on the axis.
        curves["identification_cmc"][tag] = config_identification_cmc(table, panel, open_baseline)
        curves["identification_cmc"][tag].n_known = pool_docs
        curves["identification_cmc_authors"][tag] = config_identification_cmc_authors(
            table, people, people_panel, open_baseline)
        curves["identification_cmc_authors"][tag].n_known = pool_users
        # One operating point, split by the target's length. Absent without the corpus parquet,
        # the only source of the word counts.
        for level, suffix in (("document", ""), ("author", "_authors")):
            by_words = config_identification_by_words(table, panel, run.dataset, tag, level=level)
            if by_words is not None:
                by_words.n_candidates = pool_users if level == "author" else pool_docs
                curves[f"identification_words{suffix}"][tag] = by_words
    return curves


def warm_baselines(runs: list[Run], tables: dict[Run, dict[str, pd.DataFrame]]) -> None:
    """Fill :func:`known_inclusion`'s memo in this process, before the curve workers fork.

    The Monte Carlo behind the proportional baseline depends only on (dataset, known
    configuration), so a dataset's many runs want only a handful of results between them. Under
    ``fork`` a child
    inherits whatever the parent has already computed, so filling the memo here means it is paid
    for six times rather than once per worker -- and the workers, which would each have filled
    their own copy, get it for free.
    """
    wanted = sorted({(run.dataset, tag) for run in runs for tag in tables[run]})
    for dataset in sorted({dataset for dataset, _ in wanted}):
        document_word_counts(dataset)
    missing = [key for key in wanted if key not in _INCLUSION]
    if not missing:
        return
    for key in missing:
        known_inclusion(*key)
    absent = sorted({dataset for dataset, tag in missing if _INCLUSION[(dataset, tag)] is None})
    if absent:
        print(f"  no {', '.join(absent)}.parquet under {DATA_DIR} -- the proportional baseline "
              f"needs the known side's document counts, falling back to uniform 1/N")


def build_curves(runs: list[Run], bootstrap_replicates: int, workers: int = 1,
                 needed: list[Run] | None = None):
    """Every curve every figure needs, built once per (run, configuration).

    One pass over the predictions, in workers. Runs whose files predate the known-configuration
    design are reported and skipped -- their known sides were prefixes scored on different
    documents, so they are not cells of this grid.

    **The tables are read here and the curves are built in workers.** Reading stays in this
    process because the bootstrap has to be drawn over the union of every run's users before any
    curve can be computed, and because the tables are the one thing too big to want to send
    anywhere -- the workers inherit them through :func:`run_jobs`'s fork instead. What comes back
    is only the curves, which are a few MB per run.

    **One population, one bootstrap.** The ``openset/`` figures count every unknown document
    (:func:`config_predictions` with ``in_set_only=False``), so the bootstrap resamples the users
    who *appear* in the test set. One draw is shared by every configuration and run of a dataset,
    which is what keeps the panels paired: a replicate that drops a user drops them from all of
    them at once.

    ``needed`` restricts which runs are *built*, never which are *read*: the bootstrap resamples
    the union of every run's users, and narrowing that universe would move the band on every
    figure of the dataset -- including the ones the cache is about to skip, which would then be
    inconsistent with the ones it redraws. So a cached sweep still pays for reading and skips the
    expensive half. The caller guarantees that every run feeding a figure it intends to draw is in
    ``needed``; a figure whose runs are not all built would silently lose the missing series.
    """
    open_tables = {run: config_predictions(run, in_set_only=False) for run in runs}
    for run in runs:
        if not open_tables[run]:
            print(f"  {run.directory.name}: no predictions_<attack>_known<XXYY>.csv -- this run "
                  f"predates the known-configuration design, re-run it to appear in the figures")
    open_authors = {author for run in runs for table in open_tables[run].values()
                    for author in table["true_author"].unique()}
    open_bootstrap = AuthorBootstrap(open_authors, n_replicates=bootstrap_replicates)

    live = [run for run in runs if open_tables[run]]
    warm_baselines(live, open_tables)
    heavy = [run for run in live if needed is None or run in needed]
    results = run_jobs([(run_curves, (run, open_tables[run], open_bootstrap), {})
                        for run in heavy], workers)

    # Keyed by family rather than unpacked into a tuple: a positional bug waiting to happen.
    built: dict[str, dict] = {family: {} for family in CURVE_FAMILIES}
    for run, curves in zip(heavy, results):
        for family in CURVE_FAMILIES:
            # A family is absent for a run when it had nothing to build -- a run with no open-set
            # rows, or no corpus parquet for the word-count split.
            if curves[family]:
                built[family][run] = curves[family]
    return built


class DrawContext:
    """What the build pass produces and the draw pass consumes, keyed by dataset.

    It exists so a :class:`PlannedFigure` can be described before its curves exist: the figure
    holds a closure over this object, and the closure is not called until the build has filled it
    in. A dataset the sweep never had to build is simply absent, which is correct rather than an
    error -- no figure that reads it can be stale, or it would have been built.
    """

    def __init__(self) -> None:
        self.built: dict[str, dict] = {}

    def family(self, dataset: str, family: str) -> dict:
        return self.built.get(dataset, {}).get(family, {})


def dataset_figure_plan(dataset: str, runs: list[Run],
                        output_dir: Path) -> list[PlannedFigure]:
    """One dataset's figures, named and attributed to runs, but not yet drawn or even built.

    **One planned figure is one output figure**, which is what makes the cache worth having: the
    comparison families used to be one job per (curve type, view) drawing every group in it, so a
    single new attack marked all of `by_defense/` dirty. Sharding them also spreads the drawing
    over the pool more evenly.

    Each figure's run tuple is *conservative*: it is every run that could feed the figure, taken
    from the run list alone, without knowing which of them will actually produce that curve
    family. It has to be, because that is only known after building -- and erring this way costs
    an occasional needless redraw, where erring the other way would serve a figure that is missing
    a series.
    """
    plan: list[PlannedFigure] = []

    def add(name: str, figure_runs, build, builds=None, reads_context: bool = True) -> None:
        plan.append(PlannedFigure(name=name, runs=tuple(figure_runs), build=build,
                                  builds=None if builds is None else tuple(builds),
                                  reads_context=reads_context))

    for family, kind in CURVE_FAMILIES.items():
        folder = CURVE_TYPES[kind][1]
        for method, members in method_groups(runs):
            add(f"{dataset}/{folder}/by_defense/{method[0]}_{method[1]}", members,
                lambda ctx, method=method, members=members, family=family, kind=kind:
                (plot_defense_comparison,
                 (dataset, method, members, ctx.family(dataset, family), kind, output_dir), {}))
        for defense, members in defense_groups(runs):
            add(f"{dataset}/{folder}/by_attack/{defense}", members,
                lambda ctx, defense=defense, members=members, family=family, kind=kind:
                (plot_attack_comparison,
                 (dataset, defense, members, ctx.family(dataset, family), kind, output_dir), {}))
    return plan


def clustering_figure_plan(dataset: str, runs: list[ClusteringRun], output_dir: Path,
                           timing: list[ClusteringRun] = ()) -> list[PlannedFigure]:
    """One dataset's clustering figures, planned exactly like every other family's.

    They join the cache on the same terms as the attribution figures -- a figure's key is the runs
    it draws plus their CSVs' size and mtime plus this file's digest -- which is most of what
    merging ``plot_clustering.py`` in here bought: those figures used to be redrawn on every
    invocation because nothing recorded that they were current.

    Every one of them passes ``builds=()`` and ``reads_context=False``: they read their own CSVs
    when they are drawn, so no curve family and no bootstrap is involved, and a stale clustering
    figure must not drag its corpus's attribution runs into a full curve rebuild.

    **Every family is planned once per author scope** (:data:`CLUSTERING_SCOPES`), into its own
    subtree, so the two populations are never crossed inside one figure. A scope a run has no rows
    for draws nothing and is *recorded* as having drawn nothing, which is what
    stops the sweep re-deciding it every time.

    ``timing`` is this corpus's :data:`TIMING_VARIANT` runs, and they buy a *second*
    precision/recall figure rather than changing the first: the pure-text figure is the comparable
    set every other clustering family draws, and putting a fused-edge point on it would be a
    second method under one defense's shape. The pair figure names both sets of runs in its cache
    key, so it redraws when either side moves.
    """
    plan = []
    for scope in CLUSTERING_SCOPES:
        prefix = "/".join(part for part in (f"{dataset}/clustering",
                                            CLUSTERING_SCOPES[scope][0]) if part)
        # One figure per feature, because both of these key their panels or their marker shapes by
        # *defense* -- so a feature is the thing they hold fixed, exactly as `by_defense/` does in
        # the attribution tree, where the file is named for the held-fixed method.
        for feature, members in clustering_feature_groups(runs):
            plan.append(PlannedFigure(
                f"{prefix}/bcubed/by_defense/{feature}", tuple(members),
                lambda ctx, feature=feature, members=members, scope=scope:
                (plot_clustering_bcubed, (dataset, feature, members, output_dir, scope), {}),
                builds=(), reads_context=False))
            plan.append(PlannedFigure(
                f"{prefix}/precision_recall/by_algorithm/{feature}", tuple(members),
                lambda ctx, feature=feature, members=members, scope=scope:
                (plot_clustering_precision_recall,
                 (dataset, feature, members, output_dir, scope), {}),
                builds=(), reads_context=False))
            fused = [run for run in timing if run.feature == feature]
            if fused:
                plan.append(PlannedFigure(
                    f"{prefix}/precision_recall/timing/{feature}", tuple(members) + tuple(fused),
                    lambda ctx, feature=feature, members=members, fused=fused, scope=scope:
                    (plot_clustering_precision_recall,
                     (dataset, feature, members, output_dir, scope), {"timing": fused}),
                    builds=(), reads_context=False))
    return plan


def main() -> None:
    args = parse_args()
    global WRITE_PNG
    WRITE_PNG = args.png
    # `--bootstrap` wins where both are given, so an explicit replicate count is never silently
    # replaced by the shorthand's.
    replicates = (args.bootstrap if args.bootstrap is not None
                  else (BOOTSTRAP_REPLICATES if args.bands else 0))
    families = None if args.families is None else set(args.families)
    datasets = None if args.datasets is None else set(args.datasets)

    # Two experiments, two result roots, and either one alone is enough to draw figures from --
    # a checkout that has only clustered is not an error.
    runs = discover_runs(RESULTS_DIR) if RESULTS_DIR.exists() else []
    # `--attacks` narrows the *figures*, before anything is planned, so an unselected run is not in
    # any figure's run tuple and therefore not in any cache key either -- selecting a different set
    # redraws the families that change and leaves the rest of the tree alone, exactly as adding a
    # run does. Filtering here rather than inside the two comparison families keeps every figure of
    # a sweep describing the same set of attacks.
    attacks = set(ATTACKS) if args.attacks and "all" in args.attacks else set(
        args.attacks or DEFAULT_ATTACKS)
    held_back = [run for run in runs if run.attack not in attacks]
    runs = [run for run in runs if run.attack in attacks]
    if held_back:
        print(f"drawing {len(runs)} run(s) for {len(attacks)} attack(s) "
              f"({', '.join(ATTACK_LABELS[name] for name in ATTACKS if name in attacks)}); "
              f"{len(held_back)} run(s) of other attacks left out -- pass --attacks all for every "
              f"attack on disk")
    # `--features` narrows on exactly the same terms, and just as early.
    features = set(args.features or FEATURES)
    feature_held_back = [run for run in runs if run.feature not in features]
    runs = [run for run in runs if run.feature in features]
    if feature_held_back:
        print(f"{len(feature_held_back)} run(s) of other features left out "
              f"(drawing {', '.join(FEATURE_LABELS[name] for name in FEATURES if name in features)})")
    # One parser now, and the variant splits the result: the pure-text runs are the comparable set
    # every other clustering family draws, and the rest -- a learned projection, a rescoring, a
    # timing fusion -- get the variants family, where the strategy is the axis rather than a
    # contaminant on a panel keyed by defense.
    all_clustering = [run for run in discover_clustering_runs(CLUSTERING_DIR)
                      if run.feature in features]
    clustering = [run for run in all_clustering if run.variant == PLAIN_VARIANT]
    variants = [run for run in all_clustering if run.variant != PLAIN_VARIANT]
    if not runs and not all_clustering:
        raise SystemExit(
            f"no runs named <dataset>_<defense>_<feature>_<attack> under {RESULTS_DIR}"
            f"{' for the selected attacks' if held_back else ''}, and none named "
            f"<dataset>_<defense>_<feature>_<variant> under {CLUSTERING_DIR} -- "
            f"{'pass --attacks all' if held_back else 'run an experiment first'}.")

    by_dataset: dict[str, list[Run]] = defaultdict(list)
    for run in runs:
        by_dataset[run.dataset].append(run)
    clustering_by_dataset: dict[str, list[ClusteringRun]] = defaultdict(list)
    for run in clustering:
        clustering_by_dataset[run.dataset].append(run)
    # Keyed by (dataset, defense): the variants figure holds the defense fixed, so two defenses'
    # runs of one corpus are two figures rather than two sets of rows on one.
    variants_by_defense: dict[tuple[str, str], list[ClusteringRun]] = defaultdict(list)
    for run in variants:
        variants_by_defense[(run.dataset, run.defense)].append(run)

    # --- plan every figure, before reading or building anything ---------------
    #
    context = DrawContext()
    plan: list[PlannedFigure] = []
    for dataset in DATASETS:
        if by_dataset.get(dataset):
            plan += dataset_figure_plan(dataset, by_dataset[dataset], PLOTS_DIR / dataset)
        # Under the corpus it describes, not a tree of its own: the clustering experiment is a
        # different question about the *same* corpus, so `plots/<dataset>/clustering/` files it
        # the way every other family of that dataset's figures is filed.
        if clustering_by_dataset.get(dataset):
            plan += clustering_figure_plan(
                dataset, clustering_by_dataset[dataset], PLOTS_DIR / dataset / "clustering",
                timing=[run for run in variants
                        if run.dataset == dataset and run.variant == TIMING_VARIANT])
        for defense in DEFENSES:
            members = variants_by_defense.get((dataset, defense))
            if not members:
                continue
            # The reference is the pure-text run of THIS defense, not the undefended one: the
            # figure asks what a strategy added to the method, and a defended run's variants have
            # to be read against the defended pure-text run to answer that.
            base = next((run for run in clustering_by_dataset.get(dataset, [])
                         if run.defense == defense), None)
            for scope in CLUSTERING_SCOPES:
                prefix = clustering_prefix(dataset, scope)
                plan.append(PlannedFigure(
                    f"{prefix}/variants/{defense}_tuning_vs_test",
                    tuple(members) + ((base,) if base is not None else ()),
                    lambda ctx, dataset=dataset, defense=defense, members=members, base=base,
                    scope=scope:
                    (plot_clustering_variants,
                     (dataset, defense, members, base, PLOTS_DIR / dataset / "clustering",
                      scope), {}),
                    builds=(), reads_context=False))
        # One chart per (dataset, scope) with every defense's rows on it -- the transpose of the
        # per-defense figures above, and the only place the defense vocabulary and the strategy
        # vocabulary are crossed. Its runs are every plain run of the corpus plus every variant
        # except the omitted ones, so a change to any of them restages it.
        combined = [run for run in variants if run.dataset == dataset
                    and run.variant not in COMBINED_VARIANTS_OMITTED]
        if combined:
            plain_runs = clustering_by_dataset.get(dataset, [])
            for scope in CLUSTERING_SCOPES:
                plan.append(PlannedFigure(
                    f"{clustering_prefix(dataset, scope)}/variants/all_defenses_tuning_vs_test",
                    tuple(combined) + tuple(plain_runs),
                    lambda ctx, dataset=dataset, combined=combined, plain_runs=plain_runs,
                    scope=scope:
                    (plot_clustering_variants_by_defense,
                     (dataset, combined, plain_runs, PLOTS_DIR / dataset / "clustering", scope),
                     {}),
                    builds=(), reads_context=False))

    # The selected attacks are in the key as well: they decide which series a figure carries,
    # and an unselected run is absent from the run tuple, so without them a narrowed sweep and a
    # full one would disagree about a figure they both call current.
    # The feature selection joins the key only when one is made, so a sweep without --features
    # keeps the key every figure already has.
    settings = (f"{source_digest()}|bootstrap={replicates}|png={int(args.png)}"
                f"|attacks={','.join(sorted(attacks))}"
                + (f"|features={','.join(sorted(features))}" if args.features else ""))
    cache = PlotCache(PLOTS_DIR / CACHE_FILE, settings, force=args.force)
    # `--families` narrows what is *drawn*, never what is planned: `plan` stays whole, so the
    # manifest written at the end still describes the tree rather than this run's slice of it and
    # an unselected figure keeps the entry that says it is current. Pruning it to the selection
    # would make every filtered sweep invalidate every family it did not draw, which is exactly
    # the full redraw the flag exists to avoid.
    selected = [figure for figure in plan
                if (families is None or figure_family(figure.name) in families)
                and (datasets is None or figure_datasets(figure) <= datasets)]
    stale = [figure for figure in selected if not cache.is_current(figure)]
    print(f"{len(selected) - len(stale)} of {len(selected)} figure(s) already current"
          f"{'  (--force ignored the cache)' if args.force else ''}")
    scope = []
    if families is not None:
        scope.append(", ".join(sorted(families)))
    if datasets is not None:
        scope.append(", ".join(DATASET_LABELS.get(name, name) for name in sorted(datasets)))
    if scope:
        print(f"{len(plan) - len(selected)} figure(s) outside {' / '.join(scope)} "
              f"left as they are")
    if not stale:
        cache.save(plan)
        print("nothing to draw." if selected else
              "no planned figure is in the selection -- nothing to draw.")
        return

    # --- build only what those figures need -----------------------------------
    #
    # A figure is drawn only if it is stale, and a stale figure puts every run it draws into
    # `needed`, so anything drawn below has all of its series. The converse is the saving: a run
    # no stale figure touches is never built.
    # `touched` decides which datasets are visited at all; `curved` which of their runs pay for
    # a full curve build. They differ for the two figures that are stale on any change but read no
    # curve family, and that difference is what keeps one changed run from rebuilding a corpus.
    # A figure that reads nothing the build produces (`reads_context=False`: the clustering
    # families) is in neither, so a sweep narrowed to those draws without building.
    touched = {run for figure in stale if figure.reads_context for run in figure.runs}
    curved = {run for figure in stale for run in figure.needs_curves}
    for dataset in DATASETS:
        dataset_runs = by_dataset.get(dataset, [])
        if not any(run in touched for run in dataset_runs):
            continue
        wanted = [run for run in dataset_runs if run in curved]
        output_dir = PLOTS_DIR / dataset
        print(f"\n[{DATASET_LABELS[dataset]}] building {len(wanted)} of {len(dataset_runs)} "
              f"run(s) -> {output_dir}/")
        context.built[dataset] = build_curves(dataset_runs, replicates, args.jobs, needed=wanted)

    # --- draw ------------------------------------------------------------------
    jobs = [figure.build(context) for figure in stale]
    print(f"\nDrawing {len(jobs)} figure(s) across {min(args.jobs, len(jobs))} process(es)")
    written = []
    for figure, paths in zip(stale, run_jobs(jobs, args.jobs)):
        cache.record(figure, paths)
        written += paths
    cache.save(plan)

    for path in sorted(written):
        print(f"  {path.relative_to(PLOTS_DIR)}")
    formats = "PDF + PNG" if WRITE_PNG else "PDF"
    print(f"\nWrote {len(written)} figure(s) ({formats}) to {PLOTS_DIR}/")


if __name__ == "__main__":
    main()
