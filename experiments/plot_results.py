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

Output is ``experiments/plots/<dataset>/``, and the path is three nested choices:

    ``<family>/<doc|author>/by_{defense,attack}/``

**The middle level is what is being counted**, and it is the distinction the whole file is built
around:

``doc/``
    The unit is a **document**: what share of the traffic can be attributed.
``author/``
    The unit is a **person**, and they count once **any one** of their documents does. This is the
    right reading when being linked at all is the harm, and it runs well above the per-document
    number -- 0.089 against 0.065 on WildChat/StyloMetrix's largest known side.

It is not a re-run: the author level is the same predictions collapsed to one row per person by
:func:`author_table`, taking the best each of their documents achieved.

The innermost level is the two comparison views: ``by_defense/<feature>_<attack>.pdf`` (attack
fixed, a line per defense -- *does the defense work?*) and ``by_attack/<defense>.pdf`` (defense
fixed, a line per feature+attack -- *which attack is strongest?*).

``accuracy/``
    **CMC** -- top-k accuracy against k. The headline privacy question.
``risk_coverage/``
    **Risk-coverage** -- precision when the attack answers only its most confident documents.
    What a headline accuracy hides: an attack that is usually wrong but knows when it is right.
    At the author level the x axis stays a share of *documents* while the y axis becomes a share
    of *users*, which is what keeps the two panels readable against each other.
``scaling/``
    **Scale** -- top-1 accuracy against the size of the candidate pool. Whether the threat is an
    artefact of a small pool. Drawn only for the attacks in :data:`POOL_INTERPOLABLE_ATTACKS`,
    the ones whose scores do not depend on which other authors are enrolled. Its ``author/``
    level is the only curve in the file that is **bracketed rather than estimated**: the ranks do
    not determine it, so the line is an attained lower bound and the band carries the analytic
    gap as well as the bootstrap (see :func:`author_subpool_bounds`).
``accuracy_by_known_ndocs/`` and ``accuracy_by_test_ndocs/``
    **Accuracy against documents per author**, binned, drawn over the histogram of that
    distribution. The axis the configuration grid cannot isolate: a bigger known side hands the
    attacker more documents *per user* and more users to confuse them with at once, and
    ``scaling/`` answers only the second half of that. Here the candidate pool is fixed inside a
    panel and what varies is how much of a given user's writing exists -- on the known side (what
    the attacker holds) or on the test side (what is under attack). Both are **observational**: a
    user with 30 documents is not a user with 3 who was given more history, they are a heavier
    user. The grey bars are each bin's share of the panel, on the same 0-1 axis rather than a
    second y scale, so a bar is the weight its own point carries.
``temporal/``
    **Temporal decay** -- accuracy against whole weeks elapsed since the attacker's data ends. The
    only figure whose x axis is time, the only one that reads the whole unknown side rather than
    the shared test quarter, and the only one drawn for a single known side
    (:data:`TEMPORAL_KNOWN_CONFIG`) -- so it is two stacked axes rather than a configuration grid.
``openset/risk_coverage/``
    **Open-set precision-coverage** -- the figure above, with the strangers put back. Same
    confidence, same axes, every test document (or person) in the denominator, and a stranger
    counted wrong at every coverage. Flipping between this and ``risk_coverage/`` is the cost of
    the open world.
``openset/detection/``
    **Stranger detection** -- an ROC over ``accept_score``: can the attack tell somebody it has
    never seen from somebody it enrolled? A curve on the diagonal means no reject threshold could
    beat answering everything. The author level scores a person by their most enrolled-looking
    document, and separates far better than the document level does.
``author_risk/by_{defense,attack}/``
    **Per-user risk** -- each user's own accuracy, sorted from most to least exposed. Who carries
    the risk, rather than what it averages to. **Keeps the flat path**: it is a per-user curve by
    construction, so there is no document-level twin to file it against.

The whole ``openset/`` family reads the documents every other figure drops -- on WildChat that is
63-87% of the test quarter, some 6,200 unenrolled authors against ~970 enrolled. It rests on two
columns those rows do carry: ``author_in_known`` (the ground truth) and ``accept_score`` (the
cohort-normalised margin, which ``run_experiment.py`` writes for every unknown document whether
or not ``--ood reject`` was on). **Every operating point it shows is an oracle one**: these runs
were ``--ood none``, so the threshold the runner would have calibrated is not recoverable
without re-running. The curves are threshold-free and unaffected; a point read off one is what a
perfect calibrator could reach, not what the runner's achieved.

Four figures have no ``by_defense``/``by_attack`` split. ``accuracy/macro_micro.pdf`` puts every
run's top-1 next to itself counted three ways -- per document, per user, per identity -- because
which one a paper leads with is a claim about what "anonymity failed" means, not a detail; it
lives under ``accuracy/`` because it is the same number those curves start from.
``plots/cross_dataset/scaling/<doc|author>/by_{defense,attack}/`` is the only family outside the
per-dataset folders: pool size is the single axis along which the corpora are the same experiment
at different scales, so filing it under either one would imply it belonged to that one. It is the
per-dataset tree repeated exactly -- both counting levels, both comparison views, the same file
names, the same 3x3 configuration grid -- so a figure and its single-corpus twin sit at matching
paths and can be read against each other. The one difference is that a panel carries every corpus
at once: **colour is the compared entity, dash is the corpus**, which is what lets one method be
followed from a pool of 124 to a pool of 19,711. Each panel notes its counts per corpus and draws
a grey chance line per corpus -- they differ by up to 0.136 at the author level, where chance moves
with documents per user, and coincide at the document level where both are ``1/n``. Those chance
lines take the *corpus* pattern like everything else on the figure, so the two channels each mean
exactly one thing: the colour legend names the entities and "Random guessing" among them, the
``Dataset`` legend names the patterns. A figure is drawn only where at least two corpora have the
run; with one it would be the per-dataset figure redrawn under a folder claiming otherwise.
``openset/reach.pdf`` is the share of
the test set each known side can attempt at all, in documents and in users -- a property of the
corpus rather than of any attack, and the denominator every other figure is conditioned on.
``openset/separation/<doc|author>/<run>.pdf`` is per run because its two series are cohorts
(enrolled vs stranger) rather than runs, so it cannot carry a line per defense -- but it does
split by counting level like everything else.

``per_run/<run>/`` keeps the per-run figures the runners used to write themselves -- the
per-window CMC curves, the window sweep, and the top-k bars -- so one experiment's own detail is
still available, now regenerated rather than baked in at run time. It is **drawn only under
``--per-run``**: those are 160 of the 262 figures a full sweep writes and none of them compares
runs, so they are diagnostics to reach for rather than output to pay for every time.

Every figure is written as a PDF, and as a PNG beside it under ``--png``. Both halves of the work
-- building the curves and drawing the figures -- run across ``--jobs`` processes; see
:func:`run_jobs` for why the job table is a module global and the start method is pinned to fork.

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
"""

from __future__ import annotations

import argparse
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
from matplotlib.ticker import MaxNLocator  # noqa: E402  (whole-week ticks on the decay figure)

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "experiments" / "results"
PLOTS_DIR = REPO_ROOT / "experiments" / "plots"


# --- running independent work across cores -----------------------------------
#
# Both halves of this script are embarrassingly parallel -- a run's curves do not depend on any
# other run's, and a figure does not depend on any other figure -- so both are dispatched through
# the one helper below.
#
# THE JOB TABLE IS A MODULE GLOBAL AND THAT IS THE POINT. `ProcessPoolExecutor` pickles whatever
# it is handed, and these jobs carry the expensive things: a comparison figure's arguments hold
# every curve it draws (14 MB for one WildChat CMC panel set), and a curve job holds a whole
# predictions table. Sending those down a pipe would cost more than the work. Instead the jobs are
# parked in `_JOBS` *before* the pool forks, the children inherit them copy-on-write, and the only
# thing crossing the pipe is an integer each way plus the small result. That is also why the start
# method is pinned to "fork" rather than left to the platform default: under "spawn" the child
# re-imports this module with an empty `_JOBS` and every argument would have to be pickled after
# all. Linux-only by construction, which this cluster is.

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
    matrix -- 172 MB on WildChat's open-set tables -- on top of the tables it inherited, and this
    cluster's jobs run under a 16 GB cap that has killed runs before.
    """
    return max(1, min(8, len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity")
                      else (os.cpu_count() or 1)))

#: Where the built dataset lives -- the same default ``run_experiment.py`` attacks. Only the
#: temporal figure reads it, and only for two columns: a document's ``ended_at`` is metadata that
#: never depended on the attack, so it is joined back on ``doc_id`` rather than copied into every
#: run's predictions. A dataset's parquet is ``<dataset>.parquet``, because the dataset part of a
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
DATASETS = ("wildchat", "swe_chat")

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
    "rtt_argos",
    "example_normalization",
    *(f"dp_mlm_eps{epsilon}" for epsilon in (10, 25, 50, 100, 250, 500, 1000)),
)

FEATURES = (
    "stylometrix",
    "gemini_embedding_2",
    "gemini_embedding_001",
    "function_words",
    "character_statistics",
    "char_ngram_tfidf",
    "style_distance",
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
DATASET_LABELS = {"wildchat": "WildChat", "swe_chat": "SWE-chat"}
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
    "rtt_argos": "Round-trip translation",
    "example_normalization": "Text normalization",
    **{f"dp_mlm_eps{epsilon}": f"DP-MLM ε={epsilon}"
       for epsilon in (10, 25, 50, 100, 250, 500, 1000)},
}
FEATURE_LABELS = {
    "stylometrix": "StyloMetrix",
    "gemini_embedding_2": "Gemini Embedding 2",
    "gemini_embedding_001": "Gemini Embedding 001",
    "function_words": "Function words",
    "character_statistics": "Character stats",
    "char_ngram_tfidf": "Char n-gram TF-IDF",
    "style_distance": "StyleDistance",
}
ATTACK_LABELS = {
    "nearest_neighbor": "Nearest neighbor",
    "cosine": "Cosine centroid",
    "wccn": "WCCN centroid",
    "lda": "LDA centroid",
    "plda": "PLDA",
    "logistic": "Logistic",
    # Same model as "Logistic", fitted by minibatch Adam so it runs at WildChat's author counts.
    # Named for the fit rather than the model because the two are not interchangeable numbers.
    "logistic_sgd": "Logistic (SGD)",
    "rlsc": "RLSC",
    "svm": "SVM",
    "xgboost": "XGBoost",
}

#: Stride between one feature's block of attack slots and the next. **Frozen, and deliberately
#: not ``len(ATTACKS)``.** Since ``hue = slot % 8``, the stride's residue mod 8 is what decides
#: the whole assignment: at 9 (the attack count when this was written) it is 1, so
#: ``hue == (feature_index + attack_index) % 8``. Registering one more attack made it 10, residue
#: 2 -- which silently recoloured **57 of the 63** feature-attack pairs and every figure already
#: drawn, breaking the rule that a colour follows the entity rather than its position.
#:
#: 17 keeps residue 1, so every existing assignment is preserved exactly, and leaves room for 17
#: attacks before a block overflows into the next feature's. **Add new attacks to the END of**
#: :data:`ATTACKS`: inserting one mid-list shifts the index of everything after it, which moves
#: those hues just as surely. Any replacement must stay ``= 1 (mod 8)`` and ``>= len(ATTACKS)``.
METHOD_STRIDE = 17

#: Reading order for methods, and the colour slot each one owns: feature-major, so a figure's
#: legend runs feature by feature and two runs of the same feature sit next to each other.
METHOD_SLOTS = {(feature, attack): feature_index * METHOD_STRIDE + attack_index
                for feature_index, feature in enumerate(FEATURES)
                for attack_index, attack in enumerate(ATTACKS)}


# --- palette and chart style -------------------------------------------------
#
# Categorical hues in a fixed order, validated for colour-vision deficiency as a set (adjacent
# pairs, light surface). Never extend this by generating a ninth hue: past eight series a figure
# needs fewer lines, not a made-up colour (see `resolve_slots`).

SURFACE = "#fcfcfb"
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

#: The exception, and only on the scaling figures: there a single axes can carry both corpora,
#: and the *feature/attack* has to keep the colour so the same method is one colour across
#: datasets. Dash then carries the dataset. Nowhere else does a measured line dash -- every other
#: figure is one dataset throughout, so there would be nothing for the channel to say.
DATASET_DASHES = {"wildchat": (), "swe_chat": (7, 2, 1.5, 2)}

LINE_WIDTH = 2.0
MARKER_SIZE = 6.0  # >= 8px on the page once the 2px surface ring is added
BAND_ALPHA = 0.15  # confidence band: readable under the line, never competing with it

#: Colour slot per defense and per (feature, attack) pair. A series' colour follows the thing it
#: represents, not its position in a particular figure, so a defense keeps its colour whether it
#: is one of two lines or one of eight -- and the same defense is the same colour in every figure.
#: Defenses and methods alike simply take slots in their declared order; the slot is reduced to a
#: hue by :func:`series_style`, and :func:`resolve_slots` handles the case where two entities on
#: one figure would land on the same hue.
DEFENSE_SLOTS = {defense: index for index, defense in enumerate(DEFENSES)}


def series_style(slot: int) -> str:
    """Colour for the series occupying colour slot ``slot``, cycling over the validated hues.

    Colour is the *only* channel that carries series identity here -- every measured line is
    solid -- so two series on one figure must not share a slot modulo the palette. That is what
    :func:`resolve_slots` checks.
    """
    return CATEGORICAL[slot % len(CATEGORICAL)]


def resolve_slots(slots: list[int], dashes: list[tuple] | None = None) -> list[int]:
    """Colour slots for the series in one figure, guaranteed to be visually distinct.

    Normally this returns ``slots`` unchanged: each series keeps the slot its *entity* owns, so
    colours mean the same thing across figures. Only if two of the entities in this particular
    figure land on the same style -- possible once slots run past the eight-colour palette, as
    the 63 (feature, attack) methods do -- does it fall back to numbering the series 0, 1, 2, ...
    in the order they were given. Being able to tell two lines apart beats cross-figure
    consistency, and since callers order their series deterministically, so is the fallback.

    ``dashes`` is for the scaling figures, where two series *deliberately* share a hue because
    the dataset is carried by the dash. Passing it makes the check consider the whole style, so a
    shared hue with different dashes is left alone rather than treated as a collision.
    """
    styles = list(zip((series_style(slot) for slot in slots),
                      dashes if dashes is not None else [()] * len(slots)))
    return slots if len(set(styles)) == len(styles) else list(range(len(slots)))


def style_axes(axes, xlabel: str, ylabel: str, title: str, subtitle: str = "") -> None:
    """Apply the shared chart chrome: recessive solid grid, no box, text in text colours.

    Every figure in this file goes through here, which is what makes them look like one set.
    """
    axes.set_facecolor(SURFACE)
    axes.grid(True, which="both", color=GRID, linewidth=0.7, linestyle="-")
    axes.set_axisbelow(True)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color(AXIS)
        axes.spines[side].set_linewidth(0.8)
    axes.tick_params(colors=TEXT_SECONDARY, labelsize=9, length=3, width=0.8)
    axes.set_xlabel(xlabel, color=TEXT_SECONDARY, fontsize=10)
    axes.set_ylabel(ylabel, color=TEXT_SECONDARY, fontsize=10)
    # Title and subtitle are set as two left-aligned lines rather than one centred block: the
    # subtitle carries the caveats (how many windows, how far the k axis is trusted) and should
    # read as a sentence under the title, not compete with it.
    axes.set_title(title, color=TEXT_PRIMARY, fontsize=11.5, fontweight="bold",
                   loc="left", pad=22 if subtitle else 10)
    if subtitle:
        axes.annotate(subtitle, xy=(0, 1), xytext=(0, 8), xycoords="axes fraction",
                      textcoords="offset points", color=TEXT_SECONDARY, fontsize=9,
                      va="bottom", ha="left")


def add_legend(axes, **options):
    """A frameless legend in text colours -- identity never rides on colour alone."""
    legend = axes.legend(frameon=False, fontsize=8.5, labelcolor=TEXT_PRIMARY, **options)
    if legend.get_title().get_text():
        legend.get_title().set_color(TEXT_SECONDARY)
        legend.get_title().set_fontsize(8.5)
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
#: for a quick look, but it is the more expensive of the two: measured over a full sweep, the PNGs
#: cost 79 s against the PDFs' 47 s, because a 200 dpi raster has to be rendered *and* deflated
#: where the PDF only serialises vectors. Writing both by default meant a quarter of every run went
#: on the copy nobody publishes.
WRITE_PNG = False


def save_figure(figure, stem: Path) -> Path:
    """Write ``figure`` as a PDF (and a PNG when :data:`WRITE_PNG`), return the PDF."""
    stem.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = stem.parent / f"{stem.name}.pdf"
    paths = [pdf_path] + ([stem.parent / f"{stem.name}.png"] if WRITE_PNG else [])
    for path in paths:
        figure.savefig(path, dpi=200, facecolor=SURFACE, bbox_inches="tight")
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

    * **The shared test set only.** Each configuration scores its whole remaining future (which
      is what the temporal figure needs), but a comparison across configurations is only a
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

#: Bootstrap replicates behind every band. 1,000 is enough for a 2.5/97.5 percentile to be stable
#: to about a third of a percentage point, and the whole cost is a weighted ``bincount`` per
#: replicate per panel -- seconds, against the minutes an attack takes.
BOOTSTRAP_REPLICATES = 1000

#: Fixed so a figure redrawn tomorrow has the same band as the one in the paper.
BOOTSTRAP_SEED = 20260803


class AuthorBootstrap:
    """Cluster resample of *users*, drawn once and shared by every panel of one dataset.

    Two decisions, both load-bearing:

    * **The unit is the user, not the document.** Documents by one person are strongly
      correlated -- a distinctive, prolific user's documents are all hits -- so resampling
      documents understates the noise badly. Measured on swe-chat: a document-level interval came
      out 5.6x narrower than the user-level one, which is the same anti-conservative error as
      dividing by the square root of a nested window count, in a different disguise.
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
    bootstrap for its own copy of the same ``(replicates x documents)`` matrix -- 63 MB on
    WildChat's in-set tables, 172 MB on its open-set ones, rebuilt four and three times over.
    Building it once and passing this object down instead took a measured 24 s to about 7 s, and
    -- the reason that matters more here -- stopped several copies of it being live at once under
    the 16 GB cap.
    """

    def __init__(self, bootstrap: AuthorBootstrap, author_labels) -> None:
        self.bootstrap = bootstrap
        #: ``(replicates x documents)``: each document weighted by its own author's multiplicity.
        self.documents = bootstrap.document_weights(author_labels)

    def for_authors(self, author_labels) -> np.ndarray:
        """``(replicates x authors)`` multiplicities, for a curve whose unit is the user.

        Not cached: only :func:`config_author_risk` needs it, and it asks with that panel's
        distinct users rather than with the table's one-row-per-document labels.
        """
        return self.bootstrap.multiplicities(author_labels)


def band_from_replicates(replicates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pointwise 95% percentile interval from an already-built ``(replicates x grid)`` matrix.

    Split out from :func:`bootstrap_band` so the curve types that can compute every replicate in
    one vectorised pass share the same summary step instead of reimplementing it.

    **``np.percentile`` wherever it is safe.** ``nanpercentile`` is a uniform ~3x slower on this
    numpy whether or not the array actually holds a NaN, and instrumenting a full sweep found all
    831 bands NaN-free: the curves that *can* emit one (a cohort some replicate emptied) never did
    on real data. The check costs one pass against the partition it guards, so the NaN path stays
    rather than being asserted away.
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

    Curve types whose inner loop is a prefix sum over a *sorted* order (risk-coverage, detection)
    stay on this per-replicate path deliberately -- see :func:`weighted_selective`.
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
    WildChat's scale would be a 19,711 x 43,127 array.

    **Ranks are rounded up before bucketing**, which is what makes the histogram equivalent to
    ``ranks <= k``. ``true_author_rank`` averages ties (a true author tied with one other for
    first is rank 1.5), so truncating instead -- as this did until 2026-08-06 -- counted that
    document as a top-1 hit and made every CMC curve slightly optimistic, and made this figure
    disagree with :func:`counting_modes` and with ``author_report_*.csv`` about the same number:
    0.0656 against 0.0651 on WildChat/StyloMetrix `known0075`, 452 of 15,819 documents. The
    package's ``cmc_curve`` documents ``<=`` as the conservative reading; this now matches it.
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
    every possible label, because the callers' label spaces are sparse: a WildChat CMC ranges over
    19,711 possible ranks of which only a few thousand are ever occupied.

    Replicates are folded in blocks because the accumulation wants float64 (``reduceat`` would
    otherwise carry the weights' float32 through a cumulative sum tens of thousands of terms long)
    and a float64 copy of a whole weight matrix would be 126 MB per call.
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
    the pool size, and ``ks`` is then read off it by ``searchsorted``. Measured 2.2x faster than
    the loop on WildChat's largest cell, agreeing with it to 3e-8.
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
# weak to be the line a reader measures the attack against: on WildChat's `known0025` it is
# 0.013%, while an attacker who reads no text at all and simply always names the known side's
# most prolific author gets 2.7% -- 200x more, and still knowing nothing about writing style.
# So the baseline drawn here is a guesser that knows how many documents each known author wrote
# and nothing else.

#: Replicates behind :func:`prior_inclusion`. The estimate is a mean over prior-weighted random
#: rankings, so its error falls as ``1/sqrt(R)``; at 2,000 the k=1 point is within ~3% of its
#: exact value ``sum_a p_a q_a``, which is well inside the line width and far below the spread
#: between the curves it sits under.
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
    would be ``n_candidates`` squared -- 388 million entries on WildChat's largest known side.
    A log grid costs 120 columns instead and loses nothing: the curve is drawn on a log x axis
    and is monotone in k, so :func:`interpolate_baseline` recovers the intermediate points to
    well under a pixel.
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


def chance_identity(counts: np.ndarray, n_candidates: int, ks: np.ndarray) -> np.ndarray:
    """Uniform guessing at the identity level: ``mean_a (1 - (1 - k/N) ** m_a)``.

    The identity-level twin of :func:`chance_cmc`, and the same statement
    ``run_experiment.random_identity_accuracy`` makes: guesses are independent per document, so an
    author with ``m_a`` scored documents escapes only if all ``m_a`` of them are missed. Kept as
    the companion CSV's reference column -- what the figure draws is the proportional baseline.

    Evaluated on :func:`baseline_k_grid` and interpolated, for the same reason the proportional
    one is: the exact form is an ``(authors x k)`` array, 30 million entries on WildChat.
    """
    grid = baseline_k_grid(n_candidates)
    hit = np.minimum(grid, n_candidates) / n_candidates
    values = np.mean(1.0 - (1.0 - hit[None, :]) ** counts[:, None], axis=0)
    return interpolate_baseline(grid, values, ks)


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

    def for_documents(self, ks: np.ndarray) -> np.ndarray:
        return interpolate_baseline(self.ks, self.documents, ks)

    def for_identities(self, ks: np.ndarray) -> np.ndarray:
        return interpolate_baseline(self.ks, self.identities, ks)


def known_inclusion(dataset: str, known_config: str) -> tuple[np.ndarray, np.ndarray, pd.Index]:
    """``(ks, inclusion, authors)`` for one known side, memoised across every run that shares it.

    The Monte Carlo is a property of the *known side* -- which authors the attack ranks over and
    how much each of them wrote -- so it is the same for every defense, feature and attack run
    against that configuration. Computing it once per (dataset, configuration) is what keeps it
    off the per-run path: on WildChat's largest known side it is 19,711 authors by 2,000
    replicates, and there are 21 runs that would otherwise each pay for it.

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
    same condition that costs the temporal figure its timestamps.
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
    )


@dataclass
class ConfigCmc:
    """One (run, known configuration) CMC curve on the shared test set.

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

    @property
    def top1(self) -> float:
        return float(self.curve["accuracy"].iloc[0])


def config_cmc(table: pd.DataFrame, weights: PanelWeights,
               baseline: ProportionalBaseline | None) -> ConfigCmc:
    """Document-level CMC plus bootstrap band for one configuration's in-set test documents."""
    ranks = table["true_author_rank"].to_numpy(dtype=float)
    pools = table["n_candidate_authors"].to_numpy(dtype=float)
    ks = np.arange(1, int(pools.max()) + 1)
    accuracy = weighted_cmc(ranks, pools, ks)
    low, high = (band_from_replicates(cmc_replicates(ranks, ks, weights.documents))
                 if len(weights.documents) else (accuracy, accuracy))
    curve = pd.DataFrame({"k": ks, "accuracy": accuracy, "random": chance_cmc(pools, ks),
                          "random_proportional": (baseline.for_documents(ks)
                                                  if baseline else np.nan),
                          "ci_low": low, "ci_high": high})
    return ConfigCmc(curve=curve, n_documents=len(table),
                     n_users=int(table["true_author"].nunique()),
                     n_candidates=int(pools.max()), max_k=int(ks[-1]), level="document")


def config_identity(table: pd.DataFrame, weights: PanelWeights,
                    baseline: ProportionalBaseline | None) -> ConfigCmc:
    """Identity-level CMC: the share of users with **at least one** document inside the top k.

    The counting mode that asks "was this person linked at all", rather than what share of their
    traffic was. It is the attacker's best case and the right number when being linked once is
    the harm, and it runs several times higher than the document-level curve on the same run.

    A user's whole cluster is drawn or dropped together by :class:`AuthorBootstrap`, so their best
    rank does not move between replicates -- only how many times they are counted. That makes this
    exactly :func:`weighted_cmc` over one row per user (their best rank) with the *user*
    multiplicities as weights, which is the same statement
    :func:`counting_modes` makes at k = 1 and lets both reuse the same two helpers.
    """
    grouped = table.groupby("true_author")["true_author_rank"]
    best, counts = grouped.min(), grouped.size()
    ranks = best.to_numpy(dtype=float)
    n_candidates = int(table["n_candidate_authors"].max())
    pools = np.full(len(best), float(n_candidates))
    ks = np.arange(1, n_candidates + 1)
    accuracy = weighted_cmc(ranks, pools, ks)
    author_weights = weights.for_authors(best.index)
    low, high = (band_from_replicates(cmc_replicates(ranks, ks, author_weights))
                 if len(author_weights) else (accuracy, accuracy))
    curve = pd.DataFrame({"k": ks, "accuracy": accuracy,
                          "random": chance_identity(counts.to_numpy(), n_candidates, ks),
                          "random_proportional": (baseline.for_identities(ks)
                                                  if baseline else np.nan),
                          "ci_low": low, "ci_high": high})
    return ConfigCmc(curve=curve, n_documents=len(table), n_users=len(best),
                     n_candidates=n_candidates, max_k=int(ks[-1]), level="identity")


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
# document-level column names kept, so the existing builders work on it unchanged -- the ones that
# need no special handling are `config_detection` and `config_separation`, whose only inputs are
# `accept_score` and `author_in_known`. The two coverage families do need special handling; see
# `weighted_selective_any`.


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


# --- risk-coverage: what the attack gets right when it only answers what it is sure of ---------

#: Coverages the risk-coverage curve is evaluated at. Dense enough to show the shape, and it
#: deliberately starts above zero: the precision of the single most confident document is a
#: one-sample estimate that swings between 0 and 1 and would dominate the y axis.
COVERAGE_GRID = np.linspace(0.02, 1.0, 50)


@dataclass
class ConfigRiskCoverage:
    """One (run, configuration) risk-coverage curve: precision against how much it answers.

    ``curve`` has one row per coverage with ``precision``, ``recall`` and the pointwise
    ``ci_low``/``ci_high`` bootstrap bounds on the precision.
    """

    curve: pd.DataFrame
    n_documents: int
    n_users: int

    @property
    def full_coverage_precision(self) -> float:
        """Precision when every document is answered -- i.e. plain top-1 accuracy."""
        return float(self.curve["precision"].iloc[-1])


def weighted_selective(confidence: np.ndarray, correct: np.ndarray, order: np.ndarray,
                       weights: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Precision and recall when only the most confident ``COVERAGE_GRID`` share is answered.

    A local restatement of :func:`prompt_anonymity.evaluation.metrics.detection.selective_classification`,
    weighted by author multiplicity and evaluated on a **pre-sorted** order so a bootstrap
    replicate costs two prefix sums rather than another sort. Coverage is a share of the
    resampled documents, so the cut-off is found on the running weight rather than on a row
    count. ``recall`` is correct answers retained as a fraction of those made at full coverage,
    the sense in which Narayanan et al. reported ">80% precision at 50% recall".

    **This one stays a per-replicate loop on purpose.** Folding the replicates into one matrix of
    cumulative sums, as :func:`cmc_replicates` does, was measured 2.5x *slower* here: the cut-off
    is a ``searchsorted`` into each replicate's own running total, which has no batched form, and
    the row-at-a-time search plus the float64 working copy cost more than the prefix sums saved.
    """
    weights = np.ones(len(correct)) if weights is None else weights
    ordered_weights = weights[order]
    answered = np.cumsum(ordered_weights)
    hits = np.cumsum(ordered_weights * correct[order])
    if answered[-1] <= 0:
        return np.full(len(COVERAGE_GRID), np.nan), np.full(len(COVERAGE_GRID), np.nan)
    cut = np.searchsorted(answered, COVERAGE_GRID * answered[-1], side="left")
    cut = np.clip(cut, 0, len(answered) - 1)
    precision = hits[cut] / answered[cut]
    return precision, (hits[cut] / hits[-1] if hits[-1] > 0 else np.full(len(cut), np.nan))


def weighted_selective_any(document_confidence: np.ndarray, document_order: np.ndarray,
                           cover: np.ndarray, hit: np.ndarray,
                           document_weights: np.ndarray | None = None,
                           author_weights: np.ndarray | None = None
                           ) -> tuple[np.ndarray, np.ndarray]:
    """:func:`weighted_selective` counted per user: linked if **any answered document** is correct.

    The author level cannot reuse the document helper, and the reason is worth stating because the
    shortcut is tempting and wrong. Collapsing each user to (their best confidence, whether any
    document of theirs is correct) would credit a hit from a document the attacker never
    answered -- a user whose most confident document is wrong but whose fifth is right would count
    as linked the moment they were answered at all. So a user needs **two** thresholds:

    * ``cover`` -- their most confident document. Below this they are not answered at all.
    * ``hit`` -- their most confident *correct* document (``-inf`` if none). Below this they are
      answered but not yet linked.

    ``hit <= cover`` always, and the gap between them is exactly the region the shortcut gets
    wrong.

    **The x axis stays a share of documents**, set from ``document_confidence`` exactly as the
    document-level curve sets it. That is what keeps the two levels comparable panel to panel, and
    it makes the right edge of this curve the plain identity top-1 accuracy: at coverage 1 every
    document is answered, so every user is answered and every linkable user is linked.

    Ties are possible when two users share a ``cover`` value, which can put a user among the
    linked without being among the answered; the count is clipped so a precision cannot exceed 1.
    """
    document_weights = (np.ones(len(document_confidence)) if document_weights is None
                        else document_weights)
    author_weights = np.ones(len(cover)) if author_weights is None else author_weights
    answered_documents = np.cumsum(document_weights[document_order])
    if answered_documents[-1] <= 0 or author_weights.sum() <= 0:
        return np.full(len(COVERAGE_GRID), np.nan), np.full(len(COVERAGE_GRID), np.nan)
    cut = np.searchsorted(answered_documents, COVERAGE_GRID * answered_documents[-1], side="left")
    threshold = document_confidence[document_order][np.clip(cut, 0, len(answered_documents) - 1)]

    # Both user statistics are read the same way: sort descending, prefix-sum the weights, and a
    # threshold's count is a `searchsorted` into it. Negated because `searchsorted` needs ascending.
    def running(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        order = np.argsort(-values, kind="stable")
        return -values[order], np.concatenate([[0.0], np.cumsum(author_weights[order])])

    cover_sorted, cover_running = running(cover)
    hit_sorted, hit_running = running(hit)
    answered = cover_running[np.searchsorted(cover_sorted, -threshold, side="right")]
    linked = hit_running[np.searchsorted(hit_sorted, -threshold, side="right")]
    linked = np.minimum(linked, answered)
    precision = np.divide(linked, answered, out=np.full(len(cut), np.nan), where=answered > 0)
    return precision, (linked / linked[-1] if linked[-1] > 0
                       else np.full(len(cut), np.nan))


def config_risk_coverage(table: pd.DataFrame, weights: PanelWeights) -> ConfigRiskCoverage:
    """Risk-coverage curve plus bootstrap band for one configuration.

    This is the figure Narayanan et al. led with, and it catches what top-1 cannot: an attack
    that is usually wrong but *knows when it is right* is a far sharper privacy threat than its
    headline accuracy suggests. Read the left edge -- precision when the attacker answers only its
    most confident tenth -- as the number that matters to someone deciding whether to publish.

    **Confidence is the negated ``accept_score``** written to ``predictions_*.csv``, the attack's
    cohort-normalised margin (higher = more out-of-set, hence the negation). That is a variant of
    the "gap statistic" the original used, and a *different* quantity from the max-softmax
    confidence behind ``precision_at_*pct`` in ``rolling_results.csv``, so the two do not agree
    and are not meant to -- the margin ranks confidence considerably better.
    """
    confidence = -table["accept_score"].to_numpy(dtype=float)
    correct = (table["true_author_rank"].to_numpy(dtype=float) <= 1).astype(float)
    order = np.argsort(-confidence, kind="stable")
    precision, recall = weighted_selective(confidence, correct, order)
    low, high = bootstrap_band(
        lambda row: weighted_selective(confidence, correct, order, row)[0], weights.documents)
    curve = pd.DataFrame({"coverage": COVERAGE_GRID, "precision": precision, "recall": recall,
                          "ci_low": low, "ci_high": high})
    return ConfigRiskCoverage(curve=curve, n_documents=len(table),
                              n_users=int(table["true_author"].nunique()))


def config_risk_coverage_authors(table: pd.DataFrame, bootstrap: AuthorBootstrap
                                 ) -> ConfigRiskCoverage:
    """:func:`config_risk_coverage` counted per user, via :func:`weighted_selective_any`.

    Reads "when the attacker answers its most confident x% of documents, what share of the people
    it named anything about did it link correctly at least once?" -- so the y axis is a share of
    *users* while the x axis stays a share of documents, which is what makes this panel readable
    against the ``doc/`` one at the same coverage.
    """
    people = author_table(table)
    confidence = -table["accept_score"].to_numpy(dtype=float)
    order = np.argsort(-confidence, kind="stable")
    cover = -people["accept_score"].to_numpy(dtype=float)
    hit = people["hit_score"].to_numpy(dtype=float)
    author_weights = bootstrap.multiplicities(people["true_author"])
    document_weights = bootstrap.document_weights(table["true_author"])

    precision, recall = weighted_selective_any(confidence, order, cover, hit)
    if len(author_weights):
        low, high = band_from_replicates(np.stack([
            weighted_selective_any(confidence, order, cover, hit, documents, authors)[0]
            for documents, authors in zip(document_weights, author_weights)]))
    else:
        low, high = precision, precision
    curve = pd.DataFrame({"coverage": COVERAGE_GRID, "precision": precision, "recall": recall,
                          "ci_low": low, "ci_high": high})
    return ConfigRiskCoverage(curve=curve, n_documents=len(table), n_users=len(people))


# --- per-author risk: anonymity fails unevenly -------------------------------

#: Shares of the user population the risk curve is evaluated at, most-exposed first. A percentile
#: grid rather than a rank one because configurations contain different numbers of users.
EXPOSURE_GRID = np.linspace(0, 100, 101)


@dataclass
class ConfigAuthorRisk:
    """One (run, configuration) per-user identification rate, sorted most to least exposed.

    ``curve`` has one row per percentile of the user population with that percentile's
    ``accuracy`` and its band; ``never_identified`` is the share of users not identified once.
    """

    curve: pd.DataFrame
    n_documents: int
    n_users: int
    never_identified: float


def weighted_exposure(per_author: np.ndarray, author_weights: np.ndarray | None = None,
                      order: np.ndarray | None = None) -> np.ndarray:
    """The exposure curve: each user's own top-1 accuracy, sorted, read off a percentile grid.

    Users are weighted by their bootstrap multiplicity, so a replicate that drew one user twice
    gives them twice the width on the population axis -- the same curve a real duplicate of that
    user would produce. Positions are the midpoints of each user's slice, so the curve is anchored
    at the centre of a user's width rather than its edge and does not depend on the user count.

    ``order`` is the most-exposed-first ordering of ``per_author``. It is a property of the users'
    accuracies alone, which no replicate changes -- only the widths move -- so the caller sorts
    once and passes it in rather than paying for the same sort on every replicate.
    """
    weights = np.ones(len(per_author)) if author_weights is None else author_weights
    order = np.argsort(per_author)[::-1] if order is None else order
    values, widths = per_author[order], weights[order]
    total = widths.sum()
    if total <= 0:
        return np.full(len(EXPOSURE_GRID), np.nan)
    position = (np.cumsum(widths) - widths / 2) / total * 100
    return np.interp(EXPOSURE_GRID, position, values)


def config_author_risk(table: pd.DataFrame, weights: PanelWeights) -> ConfigAuthorRisk:
    """How re-identification risk is distributed across users, not averaged over them.

    A mean accuracy says nothing about who carries it. Sorting each configuration's users from
    most to least identified and reading off the percentiles turns that into a shape: a curve
    that falls off a cliff means a small group is fully exposed while nearly everyone else is
    untouched, and a gently sloping one means the risk is shared. On WildChat the cliff is the
    real story, and it is exactly the structure a headline accuracy hides.

    The mean of this curve is the macro accuracy in :func:`plot_macro_micro`, read another way.
    """
    per_author = (table.assign(hit=table["true_author_rank"] <= 1)
                  .groupby("true_author")["hit"].mean())
    values = per_author.to_numpy(dtype=float)
    # The bootstrap acts on users directly here -- the curve *is* a distribution over users, so
    # the multiplicity is the width each one occupies rather than a document weight.
    author_weights = weights.for_authors(per_author.index)
    order = np.argsort(values)[::-1]
    low, high = bootstrap_band(lambda row: weighted_exposure(values, row, order), author_weights)
    curve = pd.DataFrame({"percentile": EXPOSURE_GRID,
                          "accuracy": weighted_exposure(values, order=order),
                          "ci_low": low, "ci_high": high})
    return ConfigAuthorRisk(curve=curve, n_documents=len(table), n_users=len(values),
                            never_identified=float((values == 0).mean()))


# --- how much writing does it take? accuracy against documents per author -----
#
# The axis the configuration grid cannot isolate. Moving from a 25% known side to a 75% one hands
# the attacker more documents *per user* and more users to confuse them with at the same time, so
# a difference between two panels cannot be attributed to either; `scaling/` answers the second
# half of that by shrinking the candidate pool while holding the text fixed, and this family
# answers the first. The pool is whatever the configuration enrolled and stays fixed inside a
# panel -- what varies along the x axis is how much of a given user's writing exists.
#
# IT IS OBSERVATIONAL, and every reading has to carry that: a user with 30 known documents is not
# a user with 3 who was given more history, they are a heavier user, and heavier users differ in
# *what* they write as well as how much. The bins compare people, not interventions. The
# proportional baseline is drawn for exactly this reason -- it rises across the bins too, because
# a prolific author is a likelier guess, so the gap between the curve and the dashes is the part
# of the trend that is not simply the prior.
#
# Two sides, two families. `known` bins a user by how much the *attacker* holds for them (the
# evidence behind the gallery entry); `test` bins them by how much of their own traffic is under
# attack. They answer different questions and the second one partly measures the metric: "linked
# at least once" gets more chances the more a user writes. The baseline makes the same statement
# at the same level, which is what keeps that visible rather than hidden.

#: Left edge of each per-author document-count bin. Singletons at 1 and 2 because that is where
#: most of both corpora sits -- WildChat's median author writes 3 documents -- then doubling,
#: which is what keeps the later bins populated on a distribution whose tail reaches 509. The
#: edges are fixed rather than per-panel quantiles so every panel, level, side and corpus shares
#: one x axis and the bins mean the same thing everywhere.
NDOCS_BIN_EDGES = (1, 2, 3, 5, 9, 17, 33)


def ndocs_bin_labels(edges: tuple[int, ...]) -> tuple[str, ...]:
    """Tick labels for :data:`NDOCS_BIN_EDGES`: ``1``, ``2``, ``3-4``, ... , ``33+``.

    Derived from the edges rather than written out beside them, so re-binning cannot leave the
    axis claiming the old ranges.
    """
    labels = []
    for index, low in enumerate(edges):
        high = edges[index + 1] if index + 1 < len(edges) else None
        labels.append(f"{low}+" if high is None else
                      str(low) if high - low == 1 else f"{low}-{high - 1}")
    return tuple(labels)


NDOCS_BIN_LABELS = ndocs_bin_labels(NDOCS_BIN_EDGES)

#: Users a bin needs before its accuracy is drawn. The bootstrap's resampling unit is the user,
#: so a bin standing on three of them is noise however many documents they wrote between them --
#: which is why the gate counts users at *both* levels. The bar stays either way: the population
#: is a result, and a bin dropped for thinness should still be visible as the handful it was.
MIN_AUTHORS_PER_BIN = 5


@dataclass
class ConfigNdocs:
    """One (run, configuration) accuracy curve against how many documents an author has.

    ``curve`` is one row per bin: ``bin`` (its position on the axis) and ``bin_label``, the
    ``accuracy`` with its ``ci_low``/``ci_high`` band, both baselines (``random`` uniform and
    ``random_proportional``, the one drawn), and the bin's population as ``n_authors`` /
    ``n_documents`` with ``share`` the one the bars draw. ``share`` follows the level --
    documents at the document level, users at the author level -- so a bar is always the weight
    its own point carries in the panel's overall number.

    ``side`` is which corpus side the count comes from: ``"known"`` bins users by how much the
    attacker enrolled for them, ``"test"`` by how much of theirs is under attack. ``level`` is
    ``"document"`` (share of that bin's documents attributed) or ``"author"`` (share of that
    bin's users linked at least once).
    """

    curve: pd.DataFrame
    n_documents: int
    n_users: int
    side: str = "known"
    level: str = "document"


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


def config_ndocs(table: pd.DataFrame, weights: PanelWeights, dataset: str, known_config: str,
                 side: str = "known", level: str = "document") -> ConfigNdocs | None:
    """Accuracy against the number of documents an author has, binned, for one configuration.

    The counts come from one of two places and the choice is what ``side`` names. ``"known"``
    reads :func:`known_author_counts` -- the corpus, sliced to the interval this configuration
    enrolled -- so the x axis is the evidence behind the attacker's gallery entry for that user.
    ``"test"`` counts the user's rows in this panel's own scored table, so the x axis is how much
    of their traffic is under attack. Neither is derivable from the other: the two sides are
    disjoint slices of the timeline.

    Both baselines are binned the same way round as the curve above them, which is the split
    :func:`config_baseline` already makes: the document level averages ``p_a`` over the bin's
    documents, and the author level averages ``1 - (1 - p_a) ** m_a`` over its users, with
    ``m_a`` that user's documents in the scored table. That matters most on the ``"test"``
    family, where the metric itself improves with ``m_a`` -- the dashed line rises for the same
    mechanical reason the curve does, so the gap between them is what the attack contributed.

    ``None`` when ``side="known"`` and the corpus parquet is not on disk to reconstruct the known
    side; the ``"test"`` family needs nothing but the predictions and is always available.
    """
    if side == "known":
        per_author = known_author_counts(dataset, known_config)
        if per_author is None:
            return None
    else:
        per_author = table["true_author"].value_counts()

    sizes = table["true_author"].map(per_author)
    # An in-set document's author is enrolled by definition, so a missing count would mean the
    # reconstructed known side disagrees with the run that wrote the file. Dropped rather than
    # left to bin as NaN, which is the defensive stance `config_baseline` takes for the same case.
    keep = sizes.notna().to_numpy()
    if not keep.any():
        return None
    table, sizes = table[keep], sizes[keep]
    document_weights = weights.documents[:, keep] if len(weights.documents) else weights.documents

    shares = known_author_shares(dataset, known_config)
    pool = table["n_candidate_authors"].to_numpy(dtype=float)
    prior = (table["true_author"].map(shares).to_numpy(dtype=float) if shares is not None
             else 1.0 / pool)
    n_bins = len(NDOCS_BIN_EDGES)
    document_bins = np.searchsorted(NDOCS_BIN_EDGES, sizes.to_numpy(dtype=float),
                                    side="right") - 1

    # One row per user, carrying the bin they fall in (a property of the user, so any of their
    # rows will do) and how many chances a per-document guesser gets at them.
    people = pd.DataFrame({"author": table["true_author"].to_numpy(),
                           "rank": table["true_author_rank"].to_numpy(dtype=float),
                           "bin": document_bins, "prior": prior, "pool": pool})
    per_user = people.groupby("author", sort=True).agg(
        best=("rank", "min"), bin=("bin", "first"), prior=("prior", "first"),
        pool=("pool", "max"), n_documents=("rank", "size"))

    n_authors = np.bincount(per_user["bin"].to_numpy(), minlength=n_bins).astype(float)
    n_documents = np.bincount(document_bins, minlength=n_bins).astype(float)

    if level == "author":
        chances = per_user["n_documents"].to_numpy(dtype=float)
        bins = per_user["bin"].to_numpy()
        hit = per_user["best"].to_numpy(dtype=float) <= 1
        unit_weights = weights.for_authors(per_user.index)
        proportional = 1.0 - (1.0 - per_user["prior"].to_numpy(dtype=float)) ** chances
        uniform = 1.0 - (1.0 - 1.0 / per_user["pool"].to_numpy(dtype=float)) ** chances
        population = n_authors
    else:
        bins = document_bins
        hit = table["true_author_rank"].to_numpy(dtype=float) <= 1
        unit_weights = document_weights
        proportional, uniform = prior, 1.0 / pool
        population = n_documents

    accuracy, low, high = binned_rate(bins, hit, unit_weights, n_bins)
    thin = n_authors < MIN_AUTHORS_PER_BIN
    accuracy, low, high = (np.where(thin, np.nan, value) for value in (accuracy, low, high))
    total = population.sum()
    curve = pd.DataFrame({
        "bin": np.arange(n_bins),
        "bin_label": list(NDOCS_BIN_LABELS),
        "accuracy": accuracy,
        # Both baselines stay ungated: they are properties of the bin's population rather than
        # measurements of the attack, so a thin bin still has them and the CSV keeps them. It is
        # the *drawing* that follows the accuracy's gaps -- see `draw_ndocs_panel`.
        "random": bin_means(bins, uniform, n_bins),
        "random_proportional": bin_means(bins, proportional, n_bins),
        "ci_low": low, "ci_high": high,
        "n_authors": n_authors.astype(int), "n_documents": n_documents.astype(int),
        "share": population / total if total else population,
    })
    return ConfigNdocs(curve=curve, n_documents=int(len(table)), n_users=int(len(per_user)),
                       side=side, level=level)


# --- the open world: the documents nobody the attacker knows wrote -----------
#
# Every figure above answers "which known author wrote this?" on the documents where that
# question has an answer. In this corpus that is the minority: 41% of swe-chat's test quarter and
# 63-87% of WildChat's was written by somebody absent from the known side, and WildChat's test
# quarter holds roughly 6,200 such strangers against ~970 enrolled users. The figures in this
# section put them back, using the two columns their rows do carry -- `author_in_known`, which is
# the ground-truth label, and `accept_score`, the attack's cohort-normalised margin, which
# `run_experiment.py` writes for *every* unknown document whether or not `--ood reject` was on.
#
# THE OPERATING POINTS HERE ARE ORACLE ONES. All of these runs were `--ood none`, so the
# threshold `calibrate_threshold` would have chosen is not recoverable -- it needs the known-side
# embeddings and a refit. AUROC and the two curves are threshold-free and so unaffected, but any
# point read off them (an EER, a FAR) is picked knowing the true labels. Read them as what a
# perfect calibrator could reach, never as what the runner's calibrator achieved.


#: False-positive rates the detection curve is evaluated at -- the x axis, labelled ``FPR``. The
#: positive class is **out-of-set**, so this is the share of *in-set* rows wrongly flagged as
#: out-of-set. Note it is a false *reject* of a genuine user, not a false accept: the curve's
#: column is still named ``false_accept_rate`` for compatibility with the CSVs already written,
#: and that name is a misnomer -- ``accept_score`` runs the same way round, higher meaning more
#: out-of-set. Dense and linear because unlike a CMC this curve has no privileged decade -- the
#: whole trade-off is the result.
DETECTION_GRID = np.linspace(0.0, 1.0, 101)


@dataclass
class ConfigOpenSetCoverage:
    """One (run, configuration) precision-coverage curve over the **whole** test set.

    The in-set twin of this is :class:`ConfigRiskCoverage`; the difference between the two is the
    entire cost of the open world, and it is why they are drawn as separate figures at matching
    paths rather than as two lines in one panel.

    ``n_ood`` counts whatever the panel's unit is -- stranger documents at the document level,
    stranger *people* at the author level -- so ``ood_rate`` divides by the matching total. The
    two are far apart: strangers are 63-87% of WildChat's test documents but a larger share of
    its people, because the enrolled users are the prolific ones.
    """

    curve: pd.DataFrame
    n_documents: int
    n_users: int
    n_ood: int
    level: str = "document"

    @property
    def ood_rate(self) -> float:
        total = self.n_users if self.level == "author" else self.n_documents
        return self.n_ood / total if total else float("nan")

    @property
    def full_coverage_precision(self) -> float:
        return float(self.curve["precision"].iloc[-1])


def config_openset_coverage(table: pd.DataFrame, weights: PanelWeights
                            ) -> ConfigOpenSetCoverage:
    """Precision against coverage when the attacker must answer for strangers too.

    Identical machinery to :func:`config_risk_coverage` -- same confidence, same
    :func:`weighted_selective` -- on the unfiltered table, and that is the point: the only thing
    that changes is which documents are in the denominator. A stranger's document is scored as an
    error at every coverage, because naming any known author for it *is* an error; there is no
    reject option in these runs to abstain with.

    So the right edge is not top-1 accuracy but top-1 accuracy times the in-set share, and the
    left edge answers the question that actually matters to someone deciding whether to publish:
    if the attacker only acts on its most confident tenth, are those documents real
    re-identifications or strangers it happened to feel sure about?
    """
    confidence = -table["accept_score"].to_numpy(dtype=float)
    in_set = table["author_in_known"].to_numpy(dtype=bool)
    # A stranger is wrong by construction, whatever rank the file records (it records NaN).
    correct = np.where(in_set, table["true_author_rank"].to_numpy(dtype=float) <= 1, False
                       ).astype(float)
    order = np.argsort(-confidence, kind="stable")
    precision, recall = weighted_selective(confidence, correct, order)
    low, high = bootstrap_band(
        lambda row: weighted_selective(confidence, correct, order, row)[0], weights.documents)
    curve = pd.DataFrame({"coverage": COVERAGE_GRID, "precision": precision, "recall": recall,
                          "ci_low": low, "ci_high": high})
    return ConfigOpenSetCoverage(curve=curve, n_documents=len(table),
                                 n_users=int(table["true_author"].nunique()),
                                 n_ood=int((~in_set).sum()))


def config_openset_coverage_authors(table: pd.DataFrame, bootstrap: AuthorBootstrap
                                    ) -> ConfigOpenSetCoverage:
    """:func:`config_openset_coverage` counted per user, strangers included in the denominator.

    The open-world twin of :func:`config_risk_coverage_authors`, and the strangers need no special
    case: :func:`author_table` leaves a stranger's ``hit_score`` at ``-inf``, so they are answered
    like anybody else and linked at no threshold. ``n_ood`` counts stranger *people* here rather
    than stranger documents, which is the denominator this level is conditioned on.
    """
    people = author_table(table)
    confidence = -table["accept_score"].to_numpy(dtype=float)
    order = np.argsort(-confidence, kind="stable")
    cover = -people["accept_score"].to_numpy(dtype=float)
    hit = people["hit_score"].to_numpy(dtype=float)
    author_weights = bootstrap.multiplicities(people["true_author"])
    document_weights = bootstrap.document_weights(table["true_author"])

    precision, recall = weighted_selective_any(confidence, order, cover, hit)
    if len(author_weights):
        low, high = band_from_replicates(np.stack([
            weighted_selective_any(confidence, order, cover, hit, documents, authors)[0]
            for documents, authors in zip(document_weights, author_weights)]))
    else:
        low, high = precision, precision
    curve = pd.DataFrame({"coverage": COVERAGE_GRID, "precision": precision, "recall": recall,
                          "ci_low": low, "ci_high": high})
    return ConfigOpenSetCoverage(
        curve=curve, n_documents=len(table), n_users=len(people), level="author",
        n_ood=int((~people["author_in_known"].to_numpy(dtype=bool)).sum()))


@dataclass
class ConfigDetection:
    """One (run, configuration) stranger-detection ROC, plus its area.

    ``curve`` has one row per :data:`DETECTION_GRID` point with the true-accept rate and its band.
    ``auroc`` is the exact weighted area, not a trapezoid over that grid.
    """

    curve: pd.DataFrame
    auroc: float
    n_documents: int
    n_users: int
    n_ood: int
    level: str = "document"

    @property
    def unit(self) -> str:
        """What one row of this curve is -- named on the panel, since the counts differ 5x."""
        return "users" if self.level == "author" else "documents"


def weighted_roc(is_ood: np.ndarray, order: np.ndarray, weights: np.ndarray | None = None
                 ) -> tuple[np.ndarray, float]:
    """Detection rate on :data:`DETECTION_GRID`, and the exact weighted AUROC.

    Evaluated on a **pre-sorted** order (most stranger-looking first) for the same reason as
    :func:`weighted_selective`: a bootstrap replicate then costs two prefix sums rather than
    another sort of 43,000 documents.

    The area is computed from the same two cumulative sums rather than by integrating the
    interpolated curve -- for each enrolled document, the stranger weight ranked above it, summed
    and divided by the weight of all cross pairs. That is the Mann-Whitney identity, so it is
    exact where a trapezoid over 101 grid points would not be. Its one approximation is that a
    tied pair is credited to whichever side the sort put first rather than half each: verified
    against ``sklearn.metrics.roc_auc_score``, that is an exact match on WildChat and a 2e-6
    disagreement on swe-chat, where a couple of documents share a margin.
    """
    weights = np.ones(len(is_ood)) if weights is None else weights
    ordered_weights = weights[order]
    ood = is_ood[order]
    ood_weights = ordered_weights * ood
    in_set_weights = ordered_weights * ~ood
    detected = np.cumsum(ood_weights)          # strangers flagged at this threshold
    false_alarm = np.cumsum(in_set_weights)    # enrolled users wrongly flagged
    total_ood, total_in_set = detected[-1], false_alarm[-1]
    if total_ood <= 0 or total_in_set <= 0:
        return np.full(len(DETECTION_GRID), np.nan), float("nan")
    curve = np.interp(DETECTION_GRID,
                      np.concatenate([[0.0], false_alarm / total_in_set]),
                      np.concatenate([[0.0], detected / total_ood]))
    # `detected` includes the current row, which contributes 0 whenever that row is enrolled, so
    # for the enrolled rows it is exactly the stranger weight ranked strictly above them.
    auroc = float((in_set_weights * detected).sum() / (total_ood * total_in_set))
    return curve, auroc


def config_detection(table: pd.DataFrame, weights: PanelWeights,
                     level: str = "document") -> ConfigDetection:
    """Can the attack tell a stranger from an enrolled user at all?

    A different question from every other figure here, and the one an open-set claim rests on: an
    attacker who cannot reject has to name a known author for all of them, which is what caps the
    open-set precision curve. The diagonal is the whole baseline -- a detector at chance means no
    threshold on ``accept_score`` can beat "accept everything" or "reject everything", so a reject
    option would buy nothing.

    Measured, this is where the feature axis separates hardest: Gemini embeddings reach 0.65-0.84
    on swe-chat while StyloMetrix sits at 0.49-0.60 on both corpora, i.e. at chance. The same
    margin that carries no information about *correctness* on the risk-coverage figure carries
    none about *membership* either -- one statistic failing two different ways.
    """
    is_ood = ~table["author_in_known"].to_numpy(dtype=bool)
    score = table["accept_score"].to_numpy(dtype=float)
    order = np.argsort(-score, kind="stable")   # most stranger-looking first
    detection, auroc = weighted_roc(is_ood, order)
    low, high = bootstrap_band(lambda row: weighted_roc(is_ood, order, row)[0], weights.documents)
    curve = pd.DataFrame({"false_accept_rate": DETECTION_GRID, "detection_rate": detection,
                          "ci_low": low, "ci_high": high})
    return ConfigDetection(curve=curve, auroc=auroc, n_documents=len(table),
                           n_users=int(table["true_author"].nunique()),
                           n_ood=int(is_ood.sum()), level=level)


#: Bin edges behind the separation histogram. Forty bins is enough to show a shape without
#: resolving individual documents in a sparse panel.
SEPARATION_EDGES = 41

#: Percentiles the histogram's range is clipped to. A handful of extreme margins would otherwise
#: stretch the axis until both distributions collapsed into the leftmost bins.
SEPARATION_RANGE = (0.5, 99.5)


@dataclass
class ConfigSeparation:
    """One (run, configuration) pair of ``accept_score`` distributions, enrolled vs stranger.

    ``curve`` holds one row per bin with both densities and their bands. This is the diagnostic
    *behind* :class:`ConfigDetection`: the AUROC is a summary of how far these two curves have
    come apart, and only the histogram says whether the overlap is a shifted-but-wide pair (a
    threshold would trade off) or two curves sitting on top of each other (no threshold helps).
    """

    curve: pd.DataFrame
    auroc: float
    n_in_set: int
    n_ood: int
    level: str = "document"

    @property
    def unit(self) -> str:
        return "users" if self.level == "author" else "documents"


def config_separation(table: pd.DataFrame, weights: PanelWeights,
                      level: str = "document") -> ConfigSeparation:
    """The two score distributions the detection curve summarises.

    Each cohort is normalised to its own density rather than plotted as counts: strangers
    outnumber enrolled documents three to one on WildChat, so raw counts would draw one visible
    curve and one flat line along the axis, and the question here is about *shape*, not size.
    """
    score = table["accept_score"].to_numpy(dtype=float)
    is_ood = ~table["author_in_known"].to_numpy(dtype=bool)
    low_edge, high_edge = np.percentile(score, SEPARATION_RANGE)
    if not np.isfinite(low_edge) or high_edge <= low_edge:
        high_edge = low_edge + 1.0
    edges = np.linspace(low_edge, high_edge, SEPARATION_EDGES)
    centres = (edges[:-1] + edges[1:]) / 2
    binned = np.clip(np.digitize(score, edges) - 1, 0, len(centres) - 1)
    # One label per (cohort, bin) cell, so both histograms come out of a single pass: a document
    # falls in exactly one cohort, so the two are disjoint halves of one label space.
    cells = np.where(is_ood, len(centres), 0) + binned

    def densities(document_weights: np.ndarray) -> np.ndarray:
        """Both cohorts' densities end to end, so one bootstrap pass covers the pair."""
        stacked = []
        for cohort in (~is_ood, is_ood):
            histogram = np.bincount(binned, weights=document_weights * cohort,
                                    minlength=len(centres))
            total = histogram.sum()
            stacked.append(histogram / total if total > 0 else np.full(len(centres), np.nan))
        return np.concatenate(stacked)

    def density_replicates(replicate_weights: np.ndarray) -> np.ndarray:
        """:func:`densities` for every replicate at once, via one :func:`grouped_sums` pass."""
        present, sums = grouped_sums(cells, replicate_weights)
        histograms = np.zeros((len(replicate_weights), 2 * len(centres)))
        histograms[:, present] = sums
        halves = histograms.reshape(len(replicate_weights), 2, len(centres))
        totals = halves.sum(axis=2, keepdims=True)
        # A replicate that drew none of a cohort's users has no density to report, not a zero one.
        normalised = np.divide(halves, totals, out=np.full_like(halves, np.nan), where=totals > 0)
        return normalised.reshape(len(replicate_weights), 2 * len(centres))

    point = densities(np.ones(len(score)))
    band_low, band_high = (band_from_replicates(density_replicates(weights.documents))
                           if len(weights.documents) else (point, point))
    half = len(centres)
    curve = pd.DataFrame({
        "accept_score": centres,
        "in_set_density": point[:half], "in_set_ci_low": band_low[:half],
        "in_set_ci_high": band_high[:half],
        "ood_density": point[half:], "ood_ci_low": band_low[half:],
        "ood_ci_high": band_high[half:],
    })
    order = np.argsort(-score, kind="stable")
    return ConfigSeparation(curve=curve, auroc=weighted_roc(is_ood, order)[1],
                            n_in_set=int((~is_ood).sum()), n_ood=int(is_ood.sum()),
                            level=level)


def openset_reach(tables: dict[Run, dict[str, pd.DataFrame]]) -> pd.DataFrame:
    """How much of the test set each known side can even attempt, in documents and in users.

    A property of the *corpus and the configuration*, not of the attack: who wrote what and when
    is fixed, and a defense rewrites text rather than authorship, so every run of a dataset sees
    the same split. It is read from whichever run carries each configuration and cross-checked
    against the rest; a disagreement would mean two runs were scored on different documents under
    one name, which is worth a printed warning rather than a silently averaged number.

    This is the number the per-panel notes have been carrying all along, promoted to a figure
    because it is the answer to "what does a bigger known side buy?" -- reach, not strength.
    """
    rows: dict[str, dict] = {}
    for run, run_tables in tables.items():
        for tag, table in run_tables.items():
            in_set = table["author_in_known"].to_numpy(dtype=bool)
            authors = table["true_author"]
            measured = {
                "n_documents": len(table),
                "n_in_set_documents": int(in_set.sum()),
                "n_users": int(authors.nunique()),
                "n_enrolled_users": int(authors[in_set].nunique()),
            }
            if tag in rows and any(rows[tag][key] != value for key, value in measured.items()):
                print(f"  {run.directory.name}: {tag} covers a different test set from an earlier "
                      f"run of this dataset -- reach figure keeps the first")
                continue
            rows.setdefault(tag, {"known_config": tag, **measured})
    frame = pd.DataFrame(rows.values())
    if frame.empty:
        return frame
    frame["document_reach"] = frame["n_in_set_documents"] / frame["n_documents"]
    frame["user_reach"] = frame["n_enrolled_users"] / frame["n_users"]
    configs = [parse_config_tag(tag) for tag in frame["known_config"]]
    frame["known_size"] = [config.size for config in configs]
    frame["gap"] = [config.gap for config in configs]
    return frame.sort_values(["known_size", "gap"]).reset_index(drop=True)


# --- the comparison figures: one panel per known configuration ----------------

@dataclass
class Series:
    """One line inside one panel: a label, the colour slot its entity owns, and its curve.

    ``curve`` is whichever per-configuration curve object the panel drawer expects -- a
    :class:`ConfigCmc` for :func:`draw_cmc_panel`, a :class:`ConfigRiskCoverage` for
    :func:`draw_risk_coverage_panel` -- which is what lets one grouping feed every curve type.
    """

    label: str
    slot: int
    curve: object
    #: Which corpus the run came from. Only :func:`plot_scaling` uses it -- as the dash pattern,
    #: and to disambiguate CSV columns when two datasets share a label. ``None`` elsewhere.
    dataset: str | None = None

    @property
    def dash(self) -> tuple:
        return DATASET_DASHES.get(self.dataset, ())

    @property
    def column(self) -> str:
        """Name for this series in a companion CSV, unique even when labels are shared."""
        return f"{DATASET_LABELS[self.dataset]} · {self.label}" if self.dataset else self.label


def grid_config(size: float, gap: float) -> KnownConfig | None:
    """The configuration at one cell of the facet grid, or ``None`` where none can exist."""
    start = TEST_START - gap - size
    return None if start < -1e-9 else KnownConfig(start=round(start, 4),
                                                  end=round(TEST_START - gap, 4))


def config_axes_grid(figsize=(12.0, 9.0)):
    """The 3x3 triangular facet grid: rows are known-side *size*, columns are *staleness*.

    Reading across a row varies only how stale the attacker's data is; reading down a column
    varies only how much of it there is. That separation is the whole point of the experiment
    design, so it is the figure's geometry rather than something a caption has to explain.

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


def label_facets(grid, axes_for: dict, xlabel: str, ylabel: str) -> None:
    """Row and column headers for the facet grid, with axis labels only on its outer edge.

    The outer edge of a triangular grid is not its bottom row: the ``50%``-size column ends one
    row early and the ``75%`` one ends after a single cell, so the x label goes on the lowest
    *visible* axes of each column rather than on row three.
    """
    for column, gap in enumerate(GRID_GAPS):
        rows = [row for row, size in enumerate(GRID_SIZES) if grid_config(size, gap) is not None]
        if not rows:
            continue
        header = ("fresh (gap 0)" if not gap else
                  f"{round(gap * 4)} quarter{'s' if round(gap * 4) != 1 else ''} stale")
        for row in rows:
            size = GRID_SIZES[row]
            axes = grid[row][column]
            style_axes(axes,
                       xlabel if row == rows[-1] else "",
                       f"known side = {size:.0%}\n{ylabel}" if column == 0 else "",
                       header if row == rows[0] and row == 0 else "", "")
            if row == rows[-1]:
                # `sharex` hides the tick labels on every row but the grid's last one, and the
                # triangle's outer edge is not its last row. Without this the lowest panel of the
                # two right-hand columns carries an x label over unlabelled ticks -- survivable
                # on a log-k axis a reader can infer, not on a categorical one whose ticks are
                # the only thing naming the bins.
                axes.tick_params(labelbottom=True)


def panel_note(axes, text: str) -> None:
    """The in-set counts, printed inside the panel because they are not constant across the grid.

    A larger or fresher known side enrolls more of the test set's users, so it can attempt more
    of the same documents. That *reach* is part of what the known side buys and part of why two
    panels are not directly comparable at face value -- it belongs on the figure, not in a note
    someone has to look up.
    """
    axes.annotate(text, xy=(0, 1), xytext=(4, -6), xycoords="axes fraction",
                  textcoords="offset points", color=TEXT_MUTED, fontsize=7.5,
                  va="top", ha="left", zorder=5)


def figure_heading(figure, title: str, subtitle: str) -> float:
    """Place a figure-level title and its subtitle, and return the top of the drawing area.

    Two left-aligned lines rather than one centred block, matching :func:`style_axes`: the
    subtitle carries the caveats and should read as a sentence under the title. The reserved band
    scales with the figure's height in inches, so a short figure does not have its title written
    across the axes and a tall one does not leave a stripe of empty surface.
    """
    height = figure.get_size_inches()[1]
    figure.text(0.01, 1 - 0.30 / height, title, color=TEXT_PRIMARY, fontsize=12.5,
                fontweight="bold", ha="left", va="top")
    figure.text(0.01, 1 - 0.58 / height, subtitle, color=TEXT_SECONDARY, fontsize=9,
                ha="left", va="top")
    return 1 - 0.78 / height


def style_legend(legend) -> None:
    """The shared legend chrome: a title in secondary ink at the body size."""
    legend.get_title().set_color(TEXT_SECONDARY)
    legend.get_title().set_fontsize(8.5)


def finish_facets(figure, handles: dict, legend_title: str, title: str, subtitle: str,
                  stem: Path, table: pd.DataFrame, legend_cell=None,
                  extra_legend: tuple[str, dict] | None = None) -> Path:
    """Shared chrome for every facet figure: its legend(s), one title, one companion CSV.

    The legend goes inside ``legend_cell`` -- the corner of the triangular grid that holds no
    panel -- so it takes space the figure was giving away rather than a reserved strip that
    shortens every panel. It is drawn on that axes rather than as a figure legend so it moves
    with the cell under ``tight_layout``; a long one is free to overflow, because the cells above
    and to its left are the grid's other two holes.

    Without a ``legend_cell`` the legend needs a reserved strip under the axes instead, and its
    size is worked out **in inches and then divided by the figure's height**, like
    :func:`figure_heading`: a legend is a fixed physical size, so a fraction tuned on the 9-inch
    facet grid leaves the short temporal figure's legend sitting on its x label. Three columns at
    most for the same reason -- ``bbox_inches="tight"`` grows the canvas around an over-wide
    legend, and four of these labels are wider than the axes they belong to.

    ``extra_legend`` is a second ``(title, handles)`` block continuing *below* the first, for the
    cross-dataset figures where colour and dash carry two different things and a reader needs both
    to decode one line. It is stacked rather than parked in another corner, and the anchor is
    measured off the drawn first legend (:func:`stack_below`) because that one's height grows a
    row per series. The pair starts at the cell's top rather than its centre, so it grows downward
    into space the triangle was giving away either way. **It needs a legend cell**: the reserved
    strip is already flush with the figure's bottom edge and has nothing below it to stack into.
    """
    extra_title, extra_handles = extra_legend or ("", {})
    if legend_cell is None:
        if extra_handles:
            raise ValueError("extra_legend needs a legend cell to stack into -- the reserved "
                             "strip sits on the figure's bottom edge")
        columns = min(max(len(handles), 1), 3)
        rows = -(-len(handles) // columns)
        bottom = (0.26 + 0.24 * rows) / figure.get_size_inches()[1]
    else:
        columns, bottom = 1, 0.0
    figure.tight_layout(rect=(0, bottom, 1, figure_heading(figure, title, subtitle)))
    shared = dict(ncol=columns, frameon=False, fontsize=8.5, labelcolor=TEXT_PRIMARY)
    if legend_cell is None:
        legend = figure.legend(list(handles.values()), list(handles), title=legend_title,
                               loc="lower center", bbox_to_anchor=(0.5, 0.005), **shared)
    else:
        # Centred when it is the only legend, so nothing on the existing figures moves; anchored
        # to the cell's top when a second one has to fit under it.
        legend = legend_cell.legend(list(handles.values()), list(handles), title=legend_title,
                                    loc="upper center" if extra_handles else "center", **shared)
    style_legend(legend)
    if extra_handles:
        # A second `.legend()` call on an axes *replaces* the first; the artist has to be adopted
        # explicitly to keep both.
        legend_cell.add_artist(legend)
        style_legend(legend_cell.legend(
            list(extra_handles.values()), list(extra_handles), title=extra_title,
            loc="upper center", bbox_to_anchor=tuple(stack_below(figure, legend_cell, legend)),
            borderaxespad=0.0,  # honour the measured anchor instead of re-padding off it
            # Long enough that a dash-dot period fits in the sample; on the default handle the
            # pattern is clipped and every entry reads as a solid line.
            handlelength=4.0, **shared))
    stem.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(stem.parent / f"{stem.name}.csv", index=False)
    return save_figure(figure, stem)


def draw_cmc_panel(axes, series: list[Series], handles: dict) -> pd.DataFrame:
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

    Serves the identity-level panels too: the two levels differ in what was counted, not in how it
    is drawn, and which one a panel shows is carried by the y label and subtitle its curve type
    registers rather than by anything here.
    """
    shared_k = min(item.curve.max_k for item in series)
    rows = []
    for item, slot in zip(series, resolve_slots([item.slot for item in series])):
        curve = item.curve.curve
        curve = curve[curve["k"] <= shared_k]
        color = series_style(slot)
        axes.fill_between(curve["k"], curve["ci_low"], curve["ci_high"], color=color,
                          alpha=BAND_ALPHA, linewidth=0, zorder=2)
        axes.plot(curve["k"], curve["accuracy"], color=color, linewidth=LINE_WIDTH,
                  solid_capstyle="round", zorder=3)
        handles.setdefault(item.label, Line2D([], [], color=color, linewidth=LINE_WIDTH))
        rows.append(curve.assign(series=item.label))
    baseline = series[0].curve.curve
    baseline = baseline[baseline["k"] <= shared_k]
    drawn = ("random_proportional" if baseline["random_proportional"].notna().all()
             else "random")
    axes.plot(baseline["k"], baseline[drawn], color=TEXT_MUTED, linewidth=1.2,
              linestyle=BASELINE_DASH, zorder=2)
    axes.set_xscale("log")
    axes.set_ylim(0, 1.02)
    panel_note(axes, f"{series[0].curve.n_documents:,} docs · {series[0].curve.n_users:,} users\n"
                     f"{series[0].curve.n_candidates:,} candidates")
    return pd.concat(rows, ignore_index=True)


def draw_risk_coverage_panel(axes, series: list[Series], handles: dict) -> pd.DataFrame:
    """One cell's risk-coverage curves: precision against the share of documents answered.

    A flat line is an attack whose confidence carries no information; a line that *slopes down to
    the left* is one whose confidence is anti-correlated with being right.
    """
    rows = []
    for item, slot in zip(series, resolve_slots([item.slot for item in series])):
        curve = item.curve.curve
        color = series_style(slot)
        axes.fill_between(curve["coverage"], curve["ci_low"], curve["ci_high"], color=color,
                          alpha=BAND_ALPHA, linewidth=0, zorder=2)
        axes.plot(curve["coverage"], curve["precision"], color=color, linewidth=LINE_WIDTH,
                  solid_capstyle="round", zorder=3)
        handles.setdefault(item.label, Line2D([], [], color=color, linewidth=LINE_WIDTH))
        rows.append(curve.assign(series=item.label))
    axes.set_xlim(0, 1.0)
    axes.set_ylim(0, 1.02)
    panel_note(axes, f"{series[0].curve.n_documents:,} docs · {series[0].curve.n_users:,} users")
    return pd.concat(rows, ignore_index=True)


def draw_author_risk_panel(axes, series: list[Series], handles: dict) -> pd.DataFrame:
    """One cell's per-user risk curves: each user's own accuracy, most exposed first.

    Read it as "the most exposed x% of users are identified at least this often". The area under
    a curve is that run's macro accuracy, so two runs with the same macro number can still have
    very different shapes -- and the shape is what a person deciding whether they are at risk
    actually wants.

    The share of users never identified once is the figure's headline, but it is a property of the
    *panel*, not of the series: it differs in every cell, so it goes in the panel note and in the
    companion CSV rather than in the legend, which is shared across the whole grid. Putting it in
    the legend key also made ``setdefault`` treat one method as a new series in every cell, so the
    legend repeated the same line six times over.
    """
    rows = []
    for item, slot in zip(series, resolve_slots([item.slot for item in series])):
        curve = item.curve.curve
        color = series_style(slot)
        axes.fill_between(curve["percentile"], curve["ci_low"], curve["ci_high"], color=color,
                          alpha=BAND_ALPHA, linewidth=0, zorder=2)
        axes.plot(curve["percentile"], curve["accuracy"], color=color, linewidth=LINE_WIDTH,
                  solid_capstyle="round", zorder=3)
        handles.setdefault(item.label, Line2D([], [], color=color, linewidth=LINE_WIDTH))
        rows.append(curve.assign(series=item.label,
                                 never_identified=item.curve.never_identified))
    axes.set_xlim(0, 100)
    axes.set_ylim(0, 1.02)
    # Only annotated when one series owns the panel: with several, an uncoloured list of rates
    # cannot say which line each belongs to, and the CSV carries them all either way.
    never = (f" · {series[0].curve.never_identified:.0%} never identified"
             if len(series) == 1 else "")
    panel_note(axes, f"{series[0].curve.n_users:,} users{never}")
    return pd.concat(rows, ignore_index=True)


def draw_ndocs_panel(axes, series: list[Series], handles: dict) -> pd.DataFrame:
    """One cell's accuracy against binned per-author document count, over its own histogram.

    The bars are the population each point stands on -- the histogram the request asked the
    accuracy to be overlaid on -- and they are **not a second y axis**. They are a *share* of the
    panel, which puts them on the same 0-1 scale as the accuracy above them honestly rather than
    by an arbitrary alignment of two ranges, and the share is of whatever the level counts
    (documents at ``doc/``, users at ``author/``), so a bar is the weight its own point carries in
    the panel's overall number. Absolute counts ride in the panel note and in the companion CSV,
    which is where a number that needs to be exact belongs.

    Drawn once per panel in recessive grey, like the temporal figure's weekly population: the
    distribution of documents per author is a property of the corpus and the configuration, so
    every line in a cell stands on the same one -- a defense rewrites text, not how much of it
    somebody wrote.

    The x axis is **categorical**. The bins are unequal in width by construction (``1``, ``2``,
    ``3-4``, ... ``33+``), so they are drawn at equal spacing and named on the ticks rather than
    placed on a count axis where the last bin would be five sixths of the width.

    The baseline follows the accuracy's gaps: where a bin was dropped for thinness there is no
    measurement to compare against, and a lone dashed segment over an empty stretch reads as one.
    """
    population = series[0].curve.curve
    axes.bar(population["bin"], population["share"], width=0.72, color=TEXT_MUTED, alpha=0.22,
             linewidth=0.8, edgecolor=SURFACE, zorder=1)
    rows = []
    for item, slot in zip(series, resolve_slots([item.slot for item in series])):
        curve = item.curve.curve
        color = series_style(slot)
        axes.fill_between(curve["bin"], curve["ci_low"], curve["ci_high"], color=color,
                          alpha=BAND_ALPHA, linewidth=0, zorder=2)
        # Markers throughout: there are seven of them, and each one is a bin rather than a sample
        # of a continuum -- the line between two of them interpolates nothing.
        axes.plot(curve["bin"], curve["accuracy"], color=color, linewidth=LINE_WIDTH,
                  marker="o", markersize=MARKER_SIZE, markeredgecolor=SURFACE, markeredgewidth=2,
                  solid_capstyle="round", zorder=3)
        handles.setdefault(item.label, Line2D([], [], color=color, linewidth=LINE_WIDTH))
        rows.append(curve.assign(series=item.label))
    baseline = series[0].curve.curve
    drawn = ("random_proportional" if baseline["random_proportional"].notna().all()
             else "random")
    axes.plot(baseline["bin"], baseline[drawn].where(baseline["accuracy"].notna()),
              color=TEXT_MUTED, linewidth=1.2, linestyle=BASELINE_DASH, zorder=2)
    axes.set_xticks(np.arange(len(NDOCS_BIN_LABELS)), labels=list(NDOCS_BIN_LABELS))
    axes.set_xlim(-0.6, len(NDOCS_BIN_LABELS) - 0.4)
    axes.set_ylim(0, 1.02)
    panel_note(axes, f"{series[0].curve.n_documents:,} docs · {series[0].curve.n_users:,} users")
    return pd.concat(rows, ignore_index=True)


def spans_datasets(series: list[Series]) -> bool:
    """Whether this panel carries more than one corpus -- true only on the cross-dataset figures.

    The three places a scaling panel has to behave differently are all downstream of this: the
    dash channel, the chance line, and the panel note. Everywhere else ``Series.dataset`` is
    ``None`` and the panel is a single corpus's, exactly as before.
    """
    return len({item.dataset for item in series}) > 1


def dataset_linestyle(dataset: str):
    """The line pattern that names one corpus -- solid WildChat, long-dash-dot SWE-chat."""
    return (0, DATASET_DASHES[dataset]) if DATASET_DASHES[dataset] else "-"


def baseline_linestyle(dataset: str | None):
    """How a chance line is drawn: by corpus across datasets, by :data:`BASELINE_DASH` within one.

    Which channel is free decides this, and ``dataset`` is exactly the test -- it is set on the
    cross-dataset figures and ``None`` everywhere else.

    On a per-dataset figure the dash carries nothing, so the baseline takes it and a dash reads as
    "not a measurement", as it does across the whole project. On a cross-dataset figure the dash
    is already spoken for by the corpus, and a baseline exempt from that would be the one line on
    the page whose pattern meant something else. So it joins the system -- WildChat's chance is
    solid grey, SWE-chat's is dash-dot grey -- and **the colour is what carries "not a
    measurement"**: grey is a slot no series can occupy, and it is named in the entity legend
    beside them. Two channels, one meaning each.
    """
    return dataset_linestyle(dataset) if dataset is not None else BASELINE_DASH


def scaling_baselines(series: list[Series]) -> list[tuple[str | None, pd.Series]]:
    """The chance curves a scaling panel draws -- **one per corpus** -- keyed by which corpus.

    Chance depends on the candidate pool and on how much each user wrote: properties of the corpus
    and the configuration, not of the defense or the feature. So a per-dataset panel has exactly
    one, shared by every series in the cell, and this returns that single curve untouched.

    A cross-dataset panel needs one *each*, and averaging them would be wrong rather than merely
    imprecise. At the document level chance is uniform ``1/n`` and the two corpora agree to the
    bit, so the lines coincide and look like one. At the author level it is
    ``1 - (1 - 1/n)^m_a`` averaged over users, which moves with a corpus's documents per user:
    measured on ``known0075``, WildChat and SWE-chat differ by up to **0.136**, and a mean of the
    two would be neither corpus's chance.
    """
    by_dataset: dict[str | None, pd.Series] = {}
    for item in series:
        if item.dataset not in by_dataset:
            curve = item.curve.curve
            by_dataset[item.dataset] = pd.Series(curve["random"].to_numpy(),
                                                 index=curve["n_candidates"].to_numpy())
    return list(by_dataset.items())


def scaling_panel_note(series: list[Series]) -> str:
    """The in-set counts for a scaling cell -- one line per corpus where there is more than one.

    The counts are what a panel is conditioned on, and on a cross-dataset figure they are not one
    number: the two corpora differ by two orders of magnitude, which is the entire reason the
    figure exists. Naming each is the only honest form.
    """
    def counts(item: Series) -> str:
        return f"{item.curve.n_documents:,} docs · {item.curve.n_users:,} users"

    if not spans_datasets(series):
        return counts(series[0])
    labelled: dict[str, str] = {}
    for item in series:
        labelled.setdefault(item.dataset, f"{DATASET_LABELS[item.dataset]} {counts(item)}")
    return "\n".join(labelled[name] for name in DATASETS if name in labelled)


def draw_scaling_panel(axes, series: list[Series], handles: dict) -> pd.DataFrame:
    """One cell's scaling curves: accuracy against how many users the attack ranks over.

    The claim is not the height of any line but the *widening gap*: chance falls away as ``1/n``
    while the attack decays far more slowly, so anonymity does not recover by adding users. A
    hollow ring sits at the pool actually run; everything to its left is the sub-pool
    interpolation, so an interpolated stretch is never mistaken for a measurement.

    Serves both counting levels and both scopes. At the author level ``accuracy`` is the attained
    *lower* bound and the band spans the analytic bracket as well as the bootstrap, which the
    curve type's subtitle says; on the cross-dataset figures the dash carries the corpus, so a
    hue deliberately shared by two lines is a channel rather than a collision -- which is why the
    dashes go to :func:`resolve_slots` too.

    There the chance line joins the colour legend as an entity of its own, because that is the
    channel identifying it once the dash is spoken for -- see :func:`baseline_linestyle`.
    """
    rows = []
    slots = resolve_slots([item.slot for item in series], [item.dash for item in series])
    for item, slot in zip(series, slots):
        curve = item.curve.curve
        color = series_style(slot)
        axes.fill_between(curve["n_candidates"], curve["ci_low"], curve["ci_high"], color=color,
                          alpha=BAND_ALPHA, linewidth=0, zorder=2)
        axes.plot(curve["n_candidates"], curve["accuracy"], color=color, linewidth=LINE_WIDTH,
                  linestyle=(0, item.dash) if item.dash else "-", solid_capstyle="round", zorder=3)
        axes.plot(item.curve.measured["n_candidates"], item.curve.measured["accuracy"],
                  linestyle="none", marker="o", markersize=MARKER_SIZE, markerfacecolor=SURFACE,
                  markeredgecolor=color, markeredgewidth=1.6, zorder=4)
        handles.setdefault(item.label, Line2D([], [], color=color, linewidth=LINE_WIDTH))
        # `column` rather than `label`, so the two corpora's lines stay distinguishable in the
        # companion CSV where they share a label. It *is* the label on a single-corpus figure.
        rows.append(curve.assign(series=item.column))
    for dataset, baseline in scaling_baselines(series):
        axes.plot(baseline.index, baseline.to_numpy(), color=TEXT_MUTED, linewidth=1.2,
                  linestyle=baseline_linestyle(dataset), zorder=2)
        if dataset is not None:
            # Last, so it reads as the odd one out under the measured entities rather than as one
            # of them. Solid: the sample stands for the hue, and the dash legend says what a
            # pattern means for every line on the figure, this one included.
            handles.setdefault("Random guessing", Line2D([], [], color=TEXT_MUTED, linewidth=1.2))
    axes.set_xscale("log")
    axes.set_ylim(0, 1.02)
    panel_note(axes, scaling_panel_note(series))
    return pd.concat(rows, ignore_index=True)


def draw_openset_coverage_panel(axes, series: list[Series], handles: dict) -> pd.DataFrame:
    """One cell's open-set precision-coverage curves: every test document is in the denominator.

    The same geometry as :func:`draw_risk_coverage_panel` on purpose -- the two figures sit at
    matching paths and the difference between them, read by flipping from one to the other, is
    what the open world costs. They are not overlaid: colour is the only channel carrying series
    identity here, so a second line per series would have to dash, and a dash on these figures
    means "not a measurement".

    The panel note carries the out-of-set rate, because unlike the in-set figures the denominator
    is no longer the thing every panel has in common.
    """
    rows = []
    for item, slot in zip(series, resolve_slots([item.slot for item in series])):
        curve = item.curve.curve
        color = series_style(slot)
        axes.fill_between(curve["coverage"], curve["ci_low"], curve["ci_high"], color=color,
                          alpha=BAND_ALPHA, linewidth=0, zorder=2)
        axes.plot(curve["coverage"], curve["precision"], color=color, linewidth=LINE_WIDTH,
                  solid_capstyle="round", zorder=3)
        handles.setdefault(item.label, Line2D([], [], color=color, linewidth=LINE_WIDTH))
        rows.append(curve.assign(series=item.label))
    axes.set_xlim(0, 1.0)
    axes.set_ylim(0, 1.02)
    unit = "of users" if series[0].curve.level == "author" else "of documents"
    panel_note(axes, f"{series[0].curve.n_documents:,} docs · {series[0].curve.n_users:,} users\n"
                     f"{series[0].curve.ood_rate:.0%} {unit} out-of-set")
    return pd.concat(rows, ignore_index=True)


def draw_detection_panel(axes, series: list[Series], handles: dict) -> pd.DataFrame:
    """One cell's stranger-detection ROC curves, with the chance diagonal.

    Reading the curve, with **out-of-set as the positive class**: x (``FPR``) is the share of
    in-set rows the attacker would have to discard, y (``TPR``) is the share of out-of-set rows it
    correctly refuses. A row is a document under ``doc/`` and a person under ``author/``, which is
    why the panel note names its own unit -- the two counts differ about fivefold. A curve on the
    diagonal is a detector that knows nothing, and since the diagonal is the figure's only dashed
    line, "hugging the dash" reads directly as "not measuring anything".

    The AUROC goes in the panel note rather than the legend when one series owns the panel: it
    differs in every cell, and a legend is shared across the whole grid.
    """
    rows = []
    for item, slot in zip(series, resolve_slots([item.slot for item in series])):
        curve = item.curve.curve
        color = series_style(slot)
        axes.fill_between(curve["false_accept_rate"], curve["ci_low"], curve["ci_high"],
                          color=color, alpha=BAND_ALPHA, linewidth=0, zorder=2)
        axes.plot(curve["false_accept_rate"], curve["detection_rate"], color=color,
                  linewidth=LINE_WIDTH, solid_capstyle="round", zorder=3)
        handles.setdefault(item.label, Line2D([], [], color=color, linewidth=LINE_WIDTH))
        rows.append(curve.assign(series=item.label, auroc=item.curve.auroc))
    axes.plot([0, 1], [0, 1], color=TEXT_MUTED, linewidth=1.2, linestyle=BASELINE_DASH, zorder=2)
    axes.set_xlim(0, 1.0)
    axes.set_ylim(0, 1.02)
    area = f" · AUROC {series[0].curve.auroc:.3f}" if len(series) == 1 else ""
    panel_note(axes, f"{series[0].curve.n_ood:,} out-of-set · "
                     f"{series[0].curve.n_documents - series[0].curve.n_ood:,} in-set "
                     f"{series[0].curve.unit}{area}")
    return pd.concat(rows, ignore_index=True)


#: The two cohorts of the separation figure, as (column prefix, label, colour slot). Fixed slots
#: rather than per-figure ones: these two are the same two things in every panel of every run.
SEPARATION_COHORTS = (("in_set", "In-set", 0), ("ood", "Out-of-set", 1))


def draw_separation_panel(axes, series: list[Series], handles: dict) -> pd.DataFrame:
    """One cell's two ``accept_score`` densities: enrolled users against strangers.

    The only panel drawer whose series are **cohorts rather than runs**, which is why the figure
    it belongs to is drawn per run: putting eight defenses in a panel would put sixteen lines in
    it. One :class:`ConfigSeparation` carries both cohorts and this draws both, so the panel gets
    exactly two lines however many runs the directory holds.
    """
    separation = series[0].curve
    curve = separation.curve
    for prefix, label, slot in SEPARATION_COHORTS:
        color = series_style(slot)
        axes.fill_between(curve["accept_score"], curve[f"{prefix}_ci_low"],
                          curve[f"{prefix}_ci_high"], color=color, alpha=BAND_ALPHA,
                          linewidth=0, zorder=2)
        axes.plot(curve["accept_score"], curve[f"{prefix}_density"], color=color,
                  linewidth=LINE_WIDTH, solid_capstyle="round", zorder=3)
        handles.setdefault(label, Line2D([], [], color=color, linewidth=LINE_WIDTH))
    axes.set_ylim(bottom=0)
    panel_note(axes, f"AUROC {separation.auroc:.3f}\n"
                     f"{separation.n_ood:,} out-of-set · {separation.n_in_set:,} in-set "
                     f"{separation.unit}")
    return curve.assign(auroc=separation.auroc)


#: The curve types, each as (panel drawer, subdirectory, x label, y label, subtitle).
#:
#: **The subdirectory carries the counting level**, so every family lands at
#: ``<family>/<doc|author>/by_{defense,attack}/`` and a reader flips between two panels that
#: differ in one thing only: whether the unit is a document or a person. That is the whole path
#: scheme -- :func:`plot_defense_comparisons` appends ``by_defense/<feature>_<attack>`` to
#: whatever is here and needs to know nothing about levels.
#:
#: All of these are drawn by both comparison families except ``separation``, which is per run (see
#: :func:`draw_separation_panel`) and is registered here so it inherits the same grid, legend,
#: companion CSV and path scheme.
#:
#: ``scaling``'s two levels are not symmetric: ``scaling/doc`` is a point estimate and
#: ``scaling/author`` is a bracket, because the ranks under-determine it -- see
#: :func:`author_subpool_bounds`. ``author_risk`` has no document level -- it is a per-user curve by construction, the
#: distribution whose mean is the macro accuracy -- so it keeps the flat path it always had.
CURVE_TYPES = {
    "cmc": (draw_cmc_panel, "accuracy/doc", "k (candidate authors returned)",
            "Top-k accuracy (per document)",
            "Share of documents whose author is ranked within k. Shaded: 95% bootstrap CI over "
            "users. Grey dashes: guessing in proportion to each known author's document count"),
    "identity": (draw_cmc_panel, "accuracy/author", "k (candidate authors returned)",
                 "Share of users linked at least once",
                 "A user counts once their most identifiable document reaches the top k. "
                 "Grey dashes: guessing in proportion to each known author's document count"),
    "risk_coverage": (draw_risk_coverage_panel, "risk_coverage/doc",
                      "Coverage (share of documents answered)",
                      "Precision (per document)",
                      "Confidence is the attack's cohort-normalised margin; in-set documents only"),
    "risk_coverage_authors": (draw_risk_coverage_panel, "risk_coverage/author",
                              "Coverage (share of documents answered)",
                              "Precision (per user)",
                              "Of the users the attacker named anything about, the share it "
                              "linked correctly at least once; in-set users only"),
    "author_risk": (draw_author_risk_panel, "author_risk",
                    "Share of users, most exposed first (%)", "That user's own top-1 accuracy",
                    "A cliff means the risk sits with a few users, not with the average one"),
    # The bars are explained in the subtitle rather than in the y label: the label is prefixed
    # with the row's known-side size and stacked in a 3-inch column, so a parenthetical there
    # runs over the panel above it.
    "ndocs_known": (draw_ndocs_panel, "accuracy_by_known_ndocs/doc",
                    "Documents the attacker holds for that author",
                    "Top-1 accuracy (per document)",
                    "How much of a user's writing the attacker needs, at a fixed candidate pool. "
                    "Bars: that bin's share of the panel's documents. Observational -- a heavier "
                    "user is a different person, not the same one given more history"),
    "ndocs_known_authors": (draw_ndocs_panel, "accuracy_by_known_ndocs/author",
                            "Documents the attacker holds for that author",
                            "Share of users linked at least once",
                            "How much of a user's writing the attacker needs, at a fixed "
                            "candidate pool. Bars: that bin's share of the panel's users. "
                            "Observational -- a heavier user is a different person, not the "
                            "same one given more history"),
    "ndocs_test": (draw_ndocs_panel, "accuracy_by_test_ndocs/doc",
                   "That author's documents in the test set",
                   "Top-1 accuracy (per document)",
                   "The gallery is unchanged across the bins -- this splits the target, not the "
                   "attacker. Bars: that bin's share of the panel's documents"),
    "ndocs_test_authors": (draw_ndocs_panel, "accuracy_by_test_ndocs/author",
                           "That author's documents in the test set",
                           "Share of users linked at least once",
                           "Being linked at least once gets more chances the more a user writes, "
                           "and the dashed baseline rises for that reason too -- the gap between "
                           "them is what the attack contributed. Bars: share of the panel's users"),
    "scaling": (draw_scaling_panel, "scaling/doc", "Candidate users the attack ranks over",
                "Top-1 accuracy (per document)",
                "Line interpolates down to smaller galleries; rings are the pools actually run"),
    "scaling_authors": (draw_scaling_panel, "scaling/author",
                        "Candidate users the attack ranks over",
                        "Share of users linked at least once",
                        "Line is the attained lower bound; the band spans the analytic bracket "
                        "AND the bootstrap, so it is wider than a confidence interval"),
    "openset_coverage": (draw_openset_coverage_panel, "openset/risk_coverage/doc",
                         "Coverage (share of documents answered)",
                         "Precision (per document)",
                         "All test documents: naming any known author for an out-of-set document is an error"),
    "openset_coverage_authors": (draw_openset_coverage_panel, "openset/risk_coverage/author",
                                 "Coverage (share of documents answered)",
                                 "Precision (per user)",
                                 "All test users: an out-of-set user the attacker names is an "
                                 "error at every coverage, because no answer for them is right"),
    "detection": (draw_detection_panel, "openset/detection/doc",
                  "FPR", "TPR",
                  "One document at a time; threshold-free, and grey dashes are a detector that "
                  "knows nothing"),
    "detection_authors": (draw_detection_panel, "openset/detection/author",
                          "FPR", "TPR",
                          "One person at a time, scored by their most in-set-looking "
                          "document; grey dashes are a detector that knows nothing"),
    # The x label is kept short deliberately: a facet column is a third of the figure wide, and
    # the long form ran off the right-hand panel and past the left edge of the figure. What it
    # used to say lives in the subtitle, which has the full width.
    "separation": (draw_separation_panel, "openset/separation/doc",
                   "Rejection score (higher = more out-of-set)",
                   "Share of that cohort's documents",
                   "Score is the attack's cohort-normalised margin; each cohort normalised to "
                   "its own density, since out-of-set documents outnumber in-set ones"),
    "separation_authors": (draw_separation_panel, "openset/separation/author",
                           "Rejection score (higher = more out-of-set)",
                           "Share of that cohort's users",
                           "Each user scored by their most in-set-looking document; each cohort "
                           "normalised to its own density, since out-of-set users outnumber "
                           "in-set ones"),
}


#: Which curve family :func:`run_curves` builds feeds which :data:`CURVE_TYPES` entry. The two
#: vocabularies are separate because ``exposure`` and ``author_risk`` (and ``selective`` and
#: ``risk_coverage``) were named independently, and because ``modes`` is built but drawn by a
#: figure outside the family scheme. Both levels of a family appear here as separate entries --
#: that is what makes a level cost one line rather than a branch in the driver.
CURVE_FAMILIES = {
    "cmc": "cmc",
    "identity": "identity",
    "selective": "risk_coverage",
    "selective_authors": "risk_coverage_authors",
    "exposure": "author_risk",
    "ndocs_known": "ndocs_known",
    "ndocs_known_authors": "ndocs_known_authors",
    "ndocs_test": "ndocs_test",
    "ndocs_test_authors": "ndocs_test_authors",
    "scaling": "scaling",
    "scaling_authors": "scaling_authors",
    "coverage": "openset_coverage",
    "coverage_authors": "openset_coverage_authors",
    "detection": "detection",
    "detection_authors": "detection_authors",
    "separation": "separation",
    "separation_authors": "separation_authors",
}

#: The families drawn per run rather than by the two comparison families -- their series are the
#: two cohorts, not the runs. See :func:`plot_separation_figures`.
PER_RUN_FAMILIES = ("separation", "separation_authors")


def plot_config_comparison(kind: str, panels: dict[str, list[Series]], title: str,
                           legend_title: str, stem: Path,
                           extra_legend: tuple[str, dict] | None = None) -> Path:
    """One curve type, one figure, one panel per known configuration.

    Nothing is averaged across the grid. The configurations differ in what the attacker was
    given, which is an experimental condition rather than a repeated measurement -- averaging
    them would report a number describing no experiment that was run, and the mean would move
    with the arbitrary choice of which cells were included.

    ``extra_legend`` is a second legend block for figures whose lines carry two channels -- the
    cross-dataset scaling figures, where colour is the compared entity and dash is the corpus.
    """
    draw, _, xlabel, ylabel, subtitle = CURVE_TYPES[kind]
    figure, grid, axes_for, legend_cell = config_axes_grid()
    handles: dict[str, object] = {}
    rows = []
    for tag, axes in axes_for.items():
        series = panels.get(tag)
        if not series:
            axes.set_visible(False)
            continue
        rows.append(draw(axes, series, handles).assign(known_config=tag))
    label_facets(grid, axes_for, xlabel, ylabel)
    return finish_facets(figure, handles, legend_title, title, subtitle, stem,
                         pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(),
                         legend_cell=legend_cell, extra_legend=extra_legend)


# --- grouping runs into the two comparison families --------------------------
#
# Both families are "hold one axis fixed, draw a line per value of the other". They are written
# once and parameterised by the curve type, so every curve type gets both views.

def plot_defense_comparisons(dataset: str, runs: list[Run], curves: dict, kind: str,
                             output_dir: Path) -> list[Path]:
    """One figure per (feature, attack): every defense measured against that same attack.

    This is the figure that answers "does the defense work?" -- the attack is held fixed so the
    only thing that moves between lines is what the defense did to the text.
    """
    by_method: dict[tuple[str, str], list[Run]] = defaultdict(list)
    for run in runs:
        if run in curves:
            by_method[run.method].append(run)

    written = []
    for method, method_runs in sorted(by_method.items(), key=lambda item: METHOD_SLOTS[item[0]]):
        method_runs.sort(key=lambda run: DEFENSE_SLOTS[run.defense])
        panels: dict[str, list[Series]] = defaultdict(list)
        for run in method_runs:
            for tag, curve in curves[run].items():
                panels[tag].append(Series(run.defense_label, DEFENSE_SLOTS[run.defense], curve))
        feature, attack = method
        written.append(plot_config_comparison(
            kind, panels,
            title=f"{DATASET_LABELS[dataset]}: defenses under "
                  f"{FEATURE_LABELS[feature]} / {ATTACK_LABELS[attack]}",
            legend_title="Defense",
            stem=output_dir / CURVE_TYPES[kind][1] / "by_defense" / f"{feature}_{attack}"))
    return written


def plot_attack_comparisons(dataset: str, runs: list[Run], curves: dict, kind: str,
                            output_dir: Path) -> list[Path]:
    """One figure per defense: every (feature, attack) measured against that same defense.

    The transpose of :func:`plot_defense_comparisons` -- with the defense held fixed, it says
    which representation and estimator the attacker should reach for.
    """
    by_defense: dict[str, list[Run]] = defaultdict(list)
    for run in runs:
        if run in curves:
            by_defense[run.defense].append(run)

    written = []
    for defense, defense_runs in sorted(by_defense.items(), key=lambda item: DEFENSE_SLOTS[item[0]]):
        defense_runs.sort(key=lambda run: METHOD_SLOTS[run.method])
        panels: dict[str, list[Series]] = defaultdict(list)
        for run in defense_runs:
            for tag, curve in curves[run].items():
                panels[tag].append(Series(run.method_label, METHOD_SLOTS[run.method], curve))
        written.append(plot_config_comparison(
            kind, panels,
            title=f"{DATASET_LABELS[dataset]}: attacks against {DEFENSE_LABELS[defense]}",
            legend_title="Feature / attack",
            stem=output_dir / CURVE_TYPES[kind][1] / "by_attack" / defense))
    return written


def plot_separation_figures(dataset: str, runs: list[Run], separation: dict, kind: str,
                            output_dir: Path) -> list[Path]:
    """One separation figure per run: the two score distributions, one panel per configuration.

    Per run rather than per family because its two series are the *cohorts*, not the runs -- see
    :func:`draw_separation_panel`. It is the only open-set figure that does not compare runs, and
    it is what a reader turns to when a detection curve sits on the diagonal and they want to see
    whether the two distributions are shifted-and-wide or simply the same distribution twice.

    ``kind`` selects which counting level is being drawn, and with it the path and the labels; the
    two levels are otherwise the same figure over a different unit.
    """
    written = []
    # The series are the two cohorts, so the legend names what one line is a population of:
    # at the author level that is the user themselves, not the author *of* something.
    legend_title = "User" if kind.endswith("_authors") else "Document's author"
    for run in sorted(runs, key=lambda run: (DEFENSE_SLOTS[run.defense], METHOD_SLOTS[run.method])):
        curves = separation.get(run)
        if not curves:
            continue
        panels = {tag: [Series(run.method_label, 0, curve)] for tag, curve in curves.items()}
        written.append(plot_config_comparison(
            kind, panels,
            title=f"{DATASET_LABELS[dataset]}: who looks out-of-set? "
                  f"{run.defense_label}, {run.method_label}",
            legend_title=legend_title,
            stem=output_dir / CURVE_TYPES[kind][1] / run.directory.name))
    return written


def plot_openset_reach(dataset: str, reach: pd.DataFrame, output_dir: Path) -> list[Path]:
    """What share of the test set each known side can attempt at all, in documents and in users.

    Both bars are shares of the same test quarter, so they share one axis -- two measures on two
    y scales would be a different figure pretending to be one. The absolute counts ride as direct
    labels instead, in text ink rather than the bar's colour, because the denominator is what
    makes a share mean anything here: WildChat's test quarter is 43,127 documents and the largest
    known side reaches barely a third of them.

    This is the figure that keeps the rest of the project honest. Every accuracy elsewhere is
    conditioned on the documents in these bars, and the bars are also the confound behind any
    cross-configuration comparison: a bigger known side scores better partly because it enrols
    more of the test set's users, and the extra ones are the people with longer histories.
    """
    if reach.empty:
        return []
    figure, axes = plt.subplots(figsize=(7.8, 1.6 + 0.78 * len(reach)))
    figure.patch.set_facecolor(SURFACE)

    positions = np.arange(len(reach))
    height, offsets = 0.32, (-0.20, 0.20)   # 0.08 of surface between the two bars of a group
    columns = (("document_reach", "Documents", "n_in_set_documents", "n_documents"),
               ("user_reach", "Users", "n_enrolled_users", "n_users"))
    for index, ((share, label, part, whole), offset) in enumerate(zip(columns, offsets)):
        axes.barh(positions + offset, reach[share], height=height, color=series_style(index),
                  zorder=3, label=label)
        for row, value in zip(reach.itertuples(), reach[share]):
            axes.annotate(f"{getattr(row, part):,} / {getattr(row, whole):,}",
                          xy=(value, row.Index + offset), xytext=(5, 0),
                          textcoords="offset points", va="center", ha="left",
                          color=TEXT_SECONDARY, fontsize=8)

    # Rows are ordered by known-side size then staleness, the facet grid's own reading order, so
    # the tick leads with those two rather than with the interval -- otherwise three rows all
    # reading "25% of the corpus" are only told apart by an interval the reader has to decode.
    axes.set_yticks(positions)
    axes.set_yticklabels([f"{config.size:.0%} known, {config.gap_label}\n({config.label})"
                          for config in (parse_config_tag(tag) for tag in reach["known_config"])])
    axes.invert_yaxis()   # smallest, freshest known side at the top
    style_axes(axes, "Share of the shared test set the known side reaches", "",
               f"{DATASET_LABELS[dataset]}: what a bigger known side buys is reach",
               "Everything outside these bars was written by someone the attacker never saw, "
               "and is excluded from every in-set figure")
    axes.set_xlim(0, 1.15)   # headroom for the direct labels
    axes.set_xticks(np.linspace(0, 1, 6))
    axes.grid(False, axis="y")
    # Below the axes rather than inside it: every interior corner is reachable by some bar's
    # direct label (swe-chat's largest known side runs to 0.82 and its label past that), and a
    # legend that collides on one dataset but not the other is a layout waiting to break.
    add_legend(axes, loc="upper right", bbox_to_anchor=(1.0, -0.10), ncol=2, title="Counted as")
    figure.tight_layout()

    stem = output_dir / "openset" / "reach"
    stem.parent.mkdir(parents=True, exist_ok=True)
    reach.to_csv(stem.parent / f"{stem.name}.csv", index=False)
    return [save_figure(figure, stem)]


# --- temporal decay: does the attack go stale? -------------------------------

#: The directory each counting level is filed under, so ``temporal/`` matches the path scheme
#: :data:`CURVE_TYPES` spells out literally for every other family.
LEVEL_DIRS = {"document": "doc", "author": "author"}

#: Documents a (known side, week) bin needs before its accuracy is plotted. A week holding three
#: documents produces an accuracy of 0, 1/3, 2/3 or 1, which is noise drawn at full contrast.
MIN_DOCUMENTS_PER_WEEK = 5


#: Which known side the temporal figure draws. One attacker, not an average over three: the
#: three known sides cut the timeline at different dates, so averaging them truncates to the
#: weeks the *shortest* reaches -- on swe-chat 4 weeks against the 25% side's own 8. Taking the
#: earliest side alone keeps the longest run of future, which is the axis this figure exists for.
TEMPORAL_KNOWN_CONFIG = "known0025"


@dataclass
class TemporalDecay:
    """One run's top-1 accuracy per week of the unknown stream.

    ``curve`` is long-form -- ``week``, ``accuracy``, ``random``, ``n_documents``, ``n_users``.
    ``counts`` is the same population *unfiltered*, so the context panel can show a thin week that
    the accuracy line drops.
    """

    curve: pd.DataFrame
    counts: pd.DataFrame
    known_config: str
    level: str = "document"

    @property
    def last_week(self) -> int:
        return int(self.curve["week"].max())


def corpus_documents(dataset: str) -> pd.DataFrame | None:
    """``doc_id``/``author_id``/``ended_at`` for one dataset, in ``run_experiment.py``'s own order.

    Two figures need something the runs themselves do not carry -- when a document ended, and how
    much each author wrote on the known side -- and both are properties of the *corpus* rather
    than of any attack. Joining them back here keeps one copy of the fact and, the reason that
    matters in practice, makes both available for **every run already on disk**, including ones
    far too expensive to re-run for a column.

    The ordering reproduces ``load_documents_and_features`` exactly, because it is what every
    window boundary is defined against: undated documents dropped (SWE-chat's are ~8% of the
    corpus, so keeping them would shift every boundary), then sorted by ``ended_at`` with ties
    broken by ``doc_id``. Verified against both corpora: the reconstructed ``known0025`` boundary
    lands on the same author count the runner recorded (81 and 7,456) and on the first
    ``position`` its predictions file reports.

    Three columns are read, so the 411 MB WildChat parquet costs a projection rather than a load.
    Memoised because a dataset's runs all need the same frame.
    """
    if dataset in _CORPUS:
        return _CORPUS[dataset]
    # The dataset part of a run's name is the split name, so it is also the parquet's basename.
    path = DATA_DIR / f"{dataset}.parquet"
    frame = None
    if path.exists():
        frame = pd.read_parquet(path, columns=["doc_id", "author_id", "ended_at"])
        frame["ended_at"] = pd.to_datetime(frame["ended_at"], errors="coerce", utc=True,
                                           format="mixed")
    _CORPUS[dataset] = frame
    return frame


#: Memo for :func:`corpus_documents`, one entry per dataset (``None`` when its parquet is absent).
_CORPUS: dict[str, pd.DataFrame | None] = {}


def document_end_times(dataset: str) -> pd.Series | None:
    """``doc_id -> ended_at``, the projection of :func:`corpus_documents` the temporal figure uses."""
    frame = corpus_documents(dataset)
    if frame is None:
        return None
    return pd.Series(frame["ended_at"].to_numpy(), index=frame["doc_id"].to_numpy())


def known_author_counts(dataset: str, known_config: str) -> pd.Series | None:
    """How many documents each author wrote on ``known_config``'s known side.

    The known side is the interval ``known_config`` names, taken over the dated corpus in
    chronological order -- ``round(fraction * n_documents)`` at each end, the same arithmetic
    ``known_configurations`` uses, so this is the gallery the attack really searched rather than
    an approximation of it.

    Two figures read it and they want different things from the same slice: the proportional
    baseline needs it as a *distribution* (:func:`known_author_shares`), and
    ``accuracy_by_known_ndocs/`` needs the raw per-author count as its x axis. Memoised, because
    every run of a dataset asks for the same six known sides and the slice costs a sort of the
    whole corpus -- 172,509 rows on WildChat. :func:`warm_baselines` fills this memo in the
    parent process as a side effect of warming the baselines, so the curve workers inherit it.
    """
    key = (dataset, known_config)
    if key not in _KNOWN_COUNTS:
        frame = corpus_documents(dataset)
        if frame is None:
            _KNOWN_COUNTS[key] = None
        else:
            dated = frame[frame["ended_at"].notna()].sort_values(["ended_at", "doc_id"],
                                                                 kind="mergesort")
            config = parse_config_tag(known_config)
            start = int(round(config.start * len(dated)))
            end = int(round(config.end * len(dated)))
            known = dated["author_id"].iloc[start:end]
            _KNOWN_COUNTS[key] = known.value_counts() if len(known) else None
    return _KNOWN_COUNTS[key]


#: Memo for :func:`known_author_counts`, one entry per (dataset, known configuration).
_KNOWN_COUNTS: dict[tuple[str, str], pd.Series | None] = {}


def known_author_shares(dataset: str, known_config: str) -> pd.Series | None:
    """Each known author's share of the known side's documents: the prior the baseline guesses on.

    :func:`known_author_counts` normalised, and kept as its own function because it is the form
    every baseline wants and normalising at each call site would invite one of them to forget.
    """
    counts = known_author_counts(dataset, known_config)
    return None if counts is None else counts / counts.sum()


def temporal_accuracy(run: Run, known_config: str = TEMPORAL_KNOWN_CONFIG,
                      min_documents: int = MIN_DOCUMENTS_PER_WEEK,
                      level: str = "document") -> TemporalDecay | None:
    """Top-1 accuracy per week of the unknown stream, one point per whole week.

    ``level`` is ``"document"`` (share of that week's documents attributed) or ``"author"`` (share
    of that week's *users* with at least one document attributed that week). The thinness gate
    stays on the document count either way, because that is what makes a week's estimate noisy.

    Every other figure here holds time fixed and varies the attack. This one does the opposite,
    and it separates the two things the old window sweep confounded: a longer window contains
    *more users* **and** reaches *further into the future*, so a falling accuracy could be either.
    Binning by elapsed weeks holds the attacker fixed and lets only staleness vary.

    **The weeks are disjoint buckets, not running totals.** Week 3 is the documents that ended in
    the third week after the cut and no others. A cumulative reading would drag every later point
    toward the average and hide exactly the decay this figure is asked to show.

    **A week is not a random sample of the weeks before it**, and this curve does not control for
    that: users who write steadily are still there late, one-off users are not, so a flattening
    line can be the surviving population changing rather than the attack holding up. Read it as
    the aggregate it is. (An earlier version split each week into users making their first
    appearance in the unknown stream and users seen earlier, which separated those two readings;
    it was removed for simplicity on 2026-08-06 and is recoverable from git history.)

    One known side (``known_config``), so there is nothing to average and no interval to draw:
    every point is the whole population of its week. ``counts`` carries the populations instead,
    unfiltered, so a week whose accuracy was dropped for thinness still shows up as the handful of
    people it was.

    ``random`` is that week's proportional-guessing rate: a guesser that knows each known author's
    document share ``p_a`` and nothing else names a document's author with probability
    ``p_{a(doc)}``. This is the k = 1 case of :func:`prior_inclusion`, computed in closed form
    because at k = 1 successive proportional sampling is just one draw from ``p``. Per week rather
    than per figure because the weeks hold different authors, and a property of the pool rather
    than of the attack, so every line in a panel shares it. Falls back to uniform
    ``1 / n_candidate_authors`` when the corpus parquet cannot supply the prior.

    It is counted the same way round as the curve it sits under, which is the same split
    :func:`config_baseline` makes: the document level is the mean of ``p_{a(doc)}`` over the week's
    documents, and the author level is ``mean_a [1 - (1 - p_a) ** m_a]`` over the week's users,
    with ``m_a`` that author's documents *in that week* -- the guesser gets one try per document
    and links the person if any one of them lands. Averaging ``p_a`` over users instead would have
    the baseline answering "is one document guessed" while the curve answers "is the person linked
    at all", and would understate it for anyone who wrote several times in a week.

    A hit is ``best_author == true_author`` rather than ``true_author_rank <= 1`` -- the same
    statement, but those columns are in *every* predictions file this project has written, so this
    needs no re-run and covers every output layout.

    Returns ``None`` when the run has no predictions for that known side, or when the dataset's
    parquet is not on disk to supply the timestamps.
    """
    end_times = document_end_times(run.dataset)
    if end_times is None:
        return None
    path = run.directory / f"predictions_{run.attack}_{known_config}.csv"
    if not path.exists():
        return None
    # This figure is the one place the *whole* future is read rather than the shared test set:
    # staleness is the x axis, so cutting to the last quarter would throw away every week that
    # has anything to say. `known0025` is chosen for the same reason -- it is the configuration
    # with the longest future, three quarters of the corpus.
    table = pd.read_csv(path)
    if "attack" in table.columns:
        table = table[table["attack"] == run.attack]
    table = table[table["author_in_known"].astype(bool)]
    # The pre-change layout wrote one file per window, and its windows are nested prefixes of one
    table = table.drop_duplicates(subset="doc_id")
    stamps = table["doc_id"].map(end_times)
    table, stamps = table[stamps.notna()], stamps[stamps.notna()]
    if table.empty:
        return None

    shares = known_author_shares(run.dataset, known_config)
    chance = (table["true_author"].map(shares).to_numpy(dtype=float) if shares is not None
              else 1.0 / table["n_candidate_authors"].to_numpy(dtype=float))
    elapsed = (stamps - stamps.min()).dt.total_seconds() / (7 * 24 * 3600)
    weekly = pd.DataFrame({
        "week": elapsed.to_numpy().astype(int),
        "hit": (table["best_author"] == table["true_author"]).to_numpy(),
        "author": table["true_author"].to_numpy(),
        "chance": chance,
    })

    grouped = weekly.groupby("week")
    counts = pd.DataFrame({"n_documents": grouped["hit"].size(),
                           "n_users": grouped["author"].nunique()}).reset_index()
    if level == "author":
        # A person counts for the week they wrote in if any one of that week's documents was
        # attributed to them. The week stays the bucket -- this is not "linked at any point", it
        # is "linked while they were writing that week", which is what keeps the axis a decay
        # curve rather than a cumulative one.
        #
        # The baseline has to make the same "at least once" statement or it is answering a
        # different question from the curve above it: `chance` is the constant `p_a` on every one
        # of an author's rows, so the guesser gets one independent try per document that week and
        # misses the person only if it misses all of them.
        per_user = weekly.groupby(["week", "author"]).agg(hit=("hit", "max"),
                                                          share=("chance", "first"),
                                                          n_documents=("hit", "size"))
        per_user["chance"] = 1.0 - (1.0 - per_user["share"]) ** per_user["n_documents"]
        grouped = per_user.reset_index().groupby("week")
    curve = pd.DataFrame({"accuracy": grouped["hit"].mean(),
                          "random": grouped["chance"].mean(),
                          "n_documents": counts.set_index("week")["n_documents"],
                          "n_users": counts.set_index("week")["n_users"]}).reset_index()
    thin = counts.set_index("week")["n_documents"].reindex(curve["week"]).to_numpy()
    curve = curve[thin >= min_documents]
    if curve.empty:
        return None
    return TemporalDecay(curve=curve.sort_values("week"),
                         counts=counts.sort_values("week"),
                         known_config=known_config, level=level)


def plot_temporal_figure(dataset: str, series: list[Series], title: str, legend_title: str,
                         stem: Path) -> Path:
    """One temporal figure: top-1 accuracy against elapsed weeks, one line per :class:`Series`.

    Two rows on one column, unlike every other comparison figure here -- this one is drawn for a
    single known side (:data:`TEMPORAL_KNOWN_CONFIG`), so there is no configuration grid to face
    it over, and the second row is not another curve but the *population* each week was measured
    over. That population is identical for every line in the figure -- a defense rewrites text but
    changes neither who wrote it nor when -- so it is drawn once in recessive grey, never as a
    second y axis. It is load-bearing rather than decorative: the late weeks are thin, and a point
    standing on fifty users should not be read like a point standing on a thousand.

    Series are truncated to the last week *all* of them reach, so one line holding an extra sparse
    week cannot stretch the axis for the rest.
    """
    limit = min(item.curve.last_week for item in series)
    level = series[0].curve.level
    figure, (axes, bars) = plt.subplots(2, 1, figsize=(7.4, 5.4), sharex=True,
                                        gridspec_kw={"height_ratios": [3, 1.2]})
    figure.patch.set_facecolor(SURFACE)

    handles: dict[str, object] = {}
    rows = []
    for item, slot in zip(series, resolve_slots([item.slot for item in series])):
        curve = item.curve.curve
        curve = curve[curve["week"] <= limit]
        color = series_style(slot)
        # The 2 px surface ring that keeps overlapping markers separable turns into a
        # dashed-looking line once the points are dense -- and here a dash means "not a
        # measurement". Past a dozen weeks the markers come off and the line speaks.
        marks = dict(marker="o", markersize=MARKER_SIZE, markeredgecolor=SURFACE,
                     markeredgewidth=2) if len(curve) <= 12 else {}
        axes.plot(curve["week"], curve["accuracy"], color=color, linewidth=LINE_WIDTH,
                  solid_capstyle="round", zorder=3, **marks)
        handles.setdefault(item.label, Line2D([], [], color=color, linewidth=LINE_WIDTH))
        rows.append(curve.assign(series=item.label))
    # Chance depends on the candidate pool, which is the configuration's, not the defense's or the
    # attack's -- so one baseline serves every line, in grey and the figure's only dash.
    baseline = series[0].curve.curve
    baseline = baseline[baseline["week"] <= limit]
    axes.plot(baseline["week"], baseline["random"], color=TEXT_MUTED, linewidth=1.2,
              linestyle=BASELINE_DASH, zorder=2)
    style_axes(axes, "", "Share of users linked" if level == "author" else "Top-1 accuracy",
               "", "")
    axes.set_ylim(bottom=0)

    per_week = (series[0].curve.counts.set_index("week")["n_users"]
                .reindex(range(limit + 1), fill_value=0).to_numpy())
    bars.bar(range(limit + 1), per_week, width=0.7, color=TEXT_MUTED, alpha=0.4,
             linewidth=0.8, edgecolor=SURFACE, zorder=2)
    style_axes(bars, "Weeks after the attacker's data ends", "Users", "")
    bars.set_ylim(bottom=0)
    # The bins are whole weeks; a tick at 1.5 weeks labels a point that cannot exist.
    bars.xaxis.set_major_locator(MaxNLocator(integer=True))

    known = parse_config_tag(series[0].curve.known_config)
    subtitle = (f"Attacker holds {known.start:.0%}-{known.end:.0%} of the timeline; whole unknown "
                f"side, disjoint weekly buckets")
    return finish_facets(figure, handles, legend_title, title, subtitle, stem,
                         pd.concat(rows, ignore_index=True))


def plot_temporal_defense_comparisons(dataset: str, runs: list[Run],
                                      decay: dict[Run, TemporalDecay],
                                      output_dir: Path) -> list[Path]:
    """One figure per (feature, attack): every defense's decay under that same attack.

    The temporal twin of :func:`plot_defense_comparisons`, same question -- does the defense hold
    up? -- asked against elapsed time instead of against k.
    """
    by_method: dict[tuple[str, str], list[Run]] = defaultdict(list)
    for run in runs:
        if run in decay:
            by_method[run.method].append(run)

    written = []
    for method, method_runs in sorted(by_method.items(), key=lambda item: METHOD_SLOTS[item[0]]):
        method_runs.sort(key=lambda run: DEFENSE_SLOTS[run.defense])
        series = [Series(run.defense_label, DEFENSE_SLOTS[run.defense], decay[run])
                  for run in method_runs]
        level = LEVEL_DIRS[series[0].curve.level]
        feature, attack = method
        written.append(plot_temporal_figure(
            dataset, series,
            title=f"{DATASET_LABELS[dataset]}: does re-identification go stale? "
                  f"{FEATURE_LABELS[feature]} / {ATTACK_LABELS[attack]}",
            legend_title="Defense",
            stem=output_dir / "temporal" / level / "by_defense" / f"{feature}_{attack}"))
    return written


def plot_temporal_attack_comparisons(dataset: str, runs: list[Run],
                                     decay: dict[Run, TemporalDecay],
                                     output_dir: Path) -> list[Path]:
    """One figure per defense: every (feature, attack)'s decay against that same defense."""
    by_defense: dict[str, list[Run]] = defaultdict(list)
    for run in runs:
        if run in decay:
            by_defense[run.defense].append(run)

    written = []
    for defense, defense_runs in sorted(by_defense.items(),
                                        key=lambda item: DEFENSE_SLOTS[item[0]]):
        defense_runs.sort(key=lambda run: METHOD_SLOTS[run.method])
        series = [Series(run.method_label, METHOD_SLOTS[run.method], decay[run])
                  for run in defense_runs]
        level = LEVEL_DIRS[series[0].curve.level]
        written.append(plot_temporal_figure(
            dataset, series,
            title=f"{DATASET_LABELS[dataset]}: does re-identification go stale? "
                  f"{DEFENSE_LABELS[defense]}",
            legend_title="Feature / attack",
            stem=output_dir / "temporal" / level / "by_attack" / defense))
    return written

# --- macro vs micro vs identity: three ways to count the same result ---------

#: The three ways this project counts a re-identification, in the order they are drawn, each with
#: where it comes from and what it answers.
COUNTING_MODES = (
    ("micro", "Micro (per document)", "what share of traffic can be attributed"),
    ("macro", "Macro (per user)", "how the attack does against a typical user"),
    ("identity", "Identity (any document)", "was this user re-identified at all"),
)


#: The known configuration the counting-modes figure is drawn for. One cell rather than the grid:
#: the figure's claim is that the three *counting modes* disagree, and repeating it six times
#: would spend a page making the same point. The largest, freshest known side is the attacker's
#: best case, so the spread shown is the one that matters most.
HEADLINE_CONFIG = "known0075"


def counting_modes(tables: dict[str, pd.DataFrame], weights: dict[str, PanelWeights]
                   ) -> pd.DataFrame | None:
    """The same result counted three ways, all from one table so they cannot disagree.

    * **micro** -- share of *documents* attributed. Carried by whoever writes most.
    * **macro** -- mean over *users* of that user's own accuracy. What a typical user faces.
    * **identity** -- share of users with *at least one* document attributed. The attacker's best
      case, and the right number if being linked once is the harm.

    They can differ by a factor of several on the same run, and which one a paper leads with is a
    claim about what "anonymity failed" means rather than a detail. All three are recomputed here
    from ``predictions_*.csv`` instead of being read from three different summary files, so they
    describe exactly the same documents and share one bootstrap resample.
    """
    table = tables.get(HEADLINE_CONFIG)
    panel = weights.get(HEADLINE_CONFIG)
    if table is None or table.empty or panel is None:
        return None
    hits = (table["true_author_rank"].to_numpy(dtype=float) <= 1).astype(float)
    authors = pd.Index(table["true_author"])
    author_index = pd.factorize(authors)[0]
    n_authors = author_index.max() + 1

    def modes(document_weights: np.ndarray) -> np.ndarray:
        """(micro, macro, identity) under one set of document weights."""
        per_author_hits = np.bincount(author_index, weights=document_weights * hits,
                                      minlength=n_authors)
        per_author_docs = np.bincount(author_index, weights=document_weights, minlength=n_authors)
        present = per_author_docs > 0
        if not present.any() or document_weights.sum() <= 0:
            return np.full(3, np.nan)
        rates = per_author_hits[present] / per_author_docs[present]
        # Users weigh their multiplicity in the two per-user modes: a user drawn twice counts
        # twice, exactly as a duplicate of that person in the corpus would.
        author_weight = per_author_docs[present] / np.maximum(
            np.bincount(author_index, minlength=n_authors)[present], 1)
        return np.array([float(document_weights @ hits / document_weights.sum()),
                         float(np.average(rates, weights=author_weight)),
                         float(np.average(rates > 0, weights=author_weight))])

    point = modes(np.ones(len(hits)))
    low, high = bootstrap_band(modes, panel.documents)
    return pd.DataFrame({
        "mode": [name for name, _, _ in COUNTING_MODES],
        "label": [label for _, label, _ in COUNTING_MODES],
        "accuracy": point, "ci_low": low, "ci_high": high,
        "known_config": HEADLINE_CONFIG, "n_documents": len(table),
        "n_users": int(table["true_author"].nunique()),
    })


def plot_macro_micro(dataset: str, runs: list[Run], modes: dict[Run, pd.DataFrame],
                     output_dir: Path) -> list[Path]:
    """One horizontal bar group per run: the same result counted three ways.

    Bars run horizontally because the category labels are run names -- a feature, an attack and
    sometimes a defense -- which do not fit under a vertical bar without rotating the text.
    Whiskers are the 95% clustered-bootstrap interval over users, the same one the curve figures
    shade, and asymmetric because a percentile interval on a proportion has no reason to be
    centred.
    """
    ordered = [run for run in sorted(runs, key=lambda run: (DEFENSE_SLOTS[run.defense],
                                                            METHOD_SLOTS[run.method]))
               if run in modes]
    if not ordered:
        return []

    figure, axes = plt.subplots(figsize=(7.6, 1.4 + 0.85 * len(ordered)))
    figure.patch.set_facecolor(SURFACE)

    # 0.03 of surface between bars, 3 per group. Offsets ascend because the y axis is inverted
    # below, so the first counting mode has to be placed lowest to be drawn at the top.
    height, offsets = 0.24, (-0.27, 0.0, 0.27)
    positions = np.arange(len(ordered))
    for index, ((mode, label, _), offset) in enumerate(zip(COUNTING_MODES, offsets)):
        values = [modes[run].set_index("mode").loc[mode] for run in ordered]
        accuracy = np.array([row["accuracy"] for row in values])
        lower = np.clip(accuracy - np.array([row["ci_low"] for row in values]), 0, None)
        upper = np.clip(np.array([row["ci_high"] for row in values]) - accuracy, 0, None)
        axes.barh(positions + offset, accuracy, height=height,
                  color=series_style(index), zorder=3, label=label,
                  xerr=np.nan_to_num(np.vstack([lower, upper])),
                  error_kw={"ecolor": TEXT_MUTED, "elinewidth": 1.0, "capsize": 2})

    axes.set_yticks(positions)
    axes.set_yticklabels([run.method_label if run.defense == NO_DEFENSE
                          else f"{run.method_label}\n{run.defense_label}" for run in ordered])
    axes.invert_yaxis()  # first run at the top, reading order
    style_axes(axes, "Top-1 accuracy", "", f"{DATASET_LABELS[dataset]}: one result, three ways "
               f"of counting it",
               f"Micro weights documents, macro weights users, identity asks whether any one "
               f"document hit  ·  {parse_config_tag(HEADLINE_CONFIG).label}, shared test set")
    axes.set_xlim(0, 1.02)
    axes.grid(False, axis="y")  # the bars already separate the groups; a y grid would fight them
    add_legend(axes, loc="best", title="Counting mode")
    figure.tight_layout()

    table = pd.concat([modes[run].assign(run=run.directory.name) for run in ordered],
                      ignore_index=True)
    # Sits beside the CMC figures rather than in a family of its own: it is the same top-1
    # accuracy those curves anchor at, only counted three ways instead of swept over k.
    stem = output_dir / "accuracy" / "macro_micro"
    stem.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(stem.parent / f"{stem.name}.csv", index=False)
    return [save_figure(figure, stem)]


# --- scale: does re-identification survive a bigger candidate pool? ----------

#: Attacks the sub-pool interpolation is *exact* for, and so the only ones a scaling figure is
#: drawn for. The criterion is that an author's score does not depend on which other authors are
#: enrolled: nearest-neighbour scores a document against that author's own documents and the
#: cosine centroid against that author's own mean, so shrinking the gallery only removes columns
#: from the score matrix and cannot reorder the survivors.
#:
#: Everything else is excluded even where it lives in ``attacks/similarity/``: ``wccn``, ``lda``
#: and ``plda`` estimate a projection from the whole known set, and the multiclass attacks fit one
#: decision function per author *against all the others*. A genuinely smaller gallery would refit
#: them into a different (and generally easier) problem, so interpolating their measured ranks
#: down would understate them -- a wrong number rather than a missing one.
POOL_INTERPOLABLE_ATTACKS = ("nearest_neighbor", "cosine")

#: Geometric step between the candidate-pool sizes the scaling curve is evaluated at. A *fixed*
#: ratio rather than "sixty points between 2 and this run's pool" so that every run lands on the
#: same lattice and two runs can be compared at a matched pool size -- both on the figure and,
#: without interpolating twice, in the companion CSV. 1.15 gives ~66 points from 2 to 20,000.
POOL_SIZE_RATIO = 1.15


def pool_lattice(largest: int) -> np.ndarray:
    """Candidate-pool sizes from 2 up to ``largest``, geometric, always including ``largest``."""
    steps = int(np.ceil(np.log(largest / 2) / np.log(POOL_SIZE_RATIO))) + 1
    sizes = np.round(2 * POOL_SIZE_RATIO ** np.arange(steps)).astype(int)
    return np.unique(np.append(sizes[sizes <= largest], largest))


@dataclass
class ScalingCurve:
    """One (run, configuration) top-1 accuracy as a function of how many users it chooses between.

    ``curve`` has one row per pool size with ``accuracy``, the ``random`` baseline and the
    bootstrap band. ``measured`` is the single point actually run -- this configuration's real
    pool size and the accuracy observed there -- kept separate so a marker can distinguish it
    from the interpolated line.
    """

    curve: pd.DataFrame
    measured: pd.DataFrame
    n_documents: int
    n_users: int


def subpool_weights(n_candidates: int, pool_sizes: np.ndarray) -> np.ndarray:
    """``(n_candidates, n_pool_sizes)`` matrix turning a rank distribution into a scaling curve.

    A document whose true author the attack ranked *r*-th out of *N* is answered correctly in a
    sub-pool of *n* candidates exactly when none of the r-1 authors it preferred were drawn into
    that sub-pool. For a uniformly random sub-pool containing the true author that probability is
    ``C(N-r, n-1) / C(N-1, n-1)``, so this matrix holds that weight for every (rank, pool size)
    pair and the accuracy against a smaller gallery is the rank distribution times it -- the
    standard gallery-size extrapolation, and the same reasoning
    the package's ``evaluation.pool_size_sweep`` used to implement by sampling (removed
    2026-08-04 -- this closed form replaced it).

    Being a *matrix product* is what makes bootstrapping it affordable: a replicate re-weights
    the rank histogram and multiplies, rather than re-running the binomial sweep. Ranks worse
    than ``N - n + 1`` get weight zero -- there is no room for that many preferred authors to all
    be excluded from a pool of n.

    Binomial coefficients are taken in log space from a table of ``log(i!)``, which keeps the
    whole thing exact and vectorised at N in the tens of thousands, where the coefficients
    themselves overflow long before they cancel.
    """
    log_factorial = np.concatenate(([0.0], np.cumsum(np.log(np.arange(1, n_candidates + 1)))))
    ranks = np.arange(1, n_candidates + 1)
    weights = np.zeros((n_candidates, len(pool_sizes)))
    for index, pool_size in enumerate(pool_sizes):
        spare = n_candidates - ranks - pool_size + 1
        reachable = spare >= 0
        weights[reachable, index] = np.exp(
            log_factorial[n_candidates - ranks[reachable]] - log_factorial[spare[reachable]]
            - log_factorial[n_candidates - 1] + log_factorial[n_candidates - pool_size])
    return weights


def config_scaling(run: Run, table: pd.DataFrame, weights: PanelWeights
                   ) -> ScalingCurve | None:
    """Top-1 accuracy against the number of candidate users, interpolated down from one cell.

    The measured runs give only a handful of pool sizes per dataset -- 81/106/124 on SWE-chat,
    7,456/13,694/19,711 on WildChat -- which on a shared log axis leaves the two corpora as two
    isolated clumps with two orders of magnitude of nothing between them. A configuration's *rank
    distribution* is enough to say what the same attack would have scored against any smaller
    gallery (:func:`subpool_weights`), so each cell contributes a curve from 2 candidates up to
    its own pool and the two datasets overlap instead of merely coexisting.

    It shrinks *who the attacker must choose between* while holding the text, the features and
    the timeline fixed; a genuinely smaller corpus would differ in all of those too.

    **The author-level twin of this is bracketed rather than estimated**
    (:func:`config_scaling_authors`), and the reason is a missing column rather than anything
    fundamental. The interpolation works because a document's rank *r* is a
    sufficient statistic: writing ``B`` for the set of authors preferred over its true author
    (``|B| = r-1``), the document is correct in a random sub-pool *S* of size *n* exactly when
    ``B & S`` is empty, which is ``C(N-r, n-1) / C(N-1, n-1)``.

    A *person* is linked when **at least one** of their documents survives the same draw --
    ``P(exists i: B_i & S = {})`` -- and *S* is shared across their documents, so those events are
    strongly positively correlated. Inclusion-exclusion turns that union into terms in
    ``|B_i U B_j|``, ``|B_i U B_j U B_k|``, and so on: **union cardinalities**, which the ranks do
    not determine, since a rank gives only ``|B_i|``. That is the whole obstacle.
    ``predictions_*.csv`` records the rank and never who outranked whom, so recovering it means
    re-running with the top-*r* candidate identities stored per document -- not the score matrix,
    just the identities.

    **Ranks alone still bracket it, two-sidedly**, which is why the right description is "not
    recorded" rather than "not computable", and which is what ``scaling/author`` draws --
    see :func:`author_subpool_bounds` for the two bounds and :func:`config_scaling_authors` for
    how they become one line and one band.

    Returns ``None`` for any attack outside :data:`POOL_INTERPOLABLE_ATTACKS`, where the
    estimator would not be exact.
    """
    if run.attack not in POOL_INTERPOLABLE_ATTACKS:
        return None
    ranks = table["true_author_rank"].to_numpy(dtype=float)
    n_candidates = int(table["n_candidate_authors"].max())
    pool_sizes = pool_lattice(n_candidates)
    interpolation = subpool_weights(n_candidates, pool_sizes)
    ks = np.arange(1, n_candidates + 1)

    def curve_for(document_weights: np.ndarray | None = None) -> np.ndarray:
        # The CMC is the cumulative rank distribution, so differencing it recovers the mass at
        # each rank; the shortfall from 1 is the documents whose author was never a candidate,
        # which are wrong at every pool size.
        cmc = weighted_cmc(ranks, None, ks, document_weights)
        return np.diff(cmc, prepend=0.0) @ interpolation

    # The bootstrap skips the CMC entirely: `grouped_sums` already *is* the rank distribution the
    # differencing above recovers, so a replicate is its share of each occupied rank times the
    # interpolation rows for those ranks. That never materialises the 19,711-wide cumulative curve
    # for a thousand replicates, which was the largest single array this file built.
    replicate_weights = weights.documents
    if len(replicate_weights):
        present, sums = grouped_sums(np.ceil(ranks).astype(np.int64), replicate_weights)
        totals = sums.sum(axis=1, keepdims=True)
        mass = np.divide(sums, totals, out=np.zeros_like(sums), where=totals > 0)
        low, high = band_from_replicates(mass @ interpolation[present - 1])
    else:
        low = high = curve_for()
    curve = pd.DataFrame({"n_candidates": pool_sizes, "accuracy": curve_for(),
                          "random": 1.0 / pool_sizes.astype(float),
                          "ci_low": low, "ci_high": high})
    measured = pd.DataFrame([{"n_candidates": n_candidates,
                              "accuracy": float((ranks <= 1).mean())}])
    return ScalingCurve(curve=curve, measured=measured, n_documents=len(table),
                        n_users=int(table["true_author"].nunique()))


#: Documents a user may have before :func:`author_subpool_bounds` stops computing the exact
#: inclusion-exclusion upper bound and falls back to Boole. The cost is ``2 ** m`` subsets, and
#: 12 keeps the worst user at 4,096 while covering 84.5% of WildChat's -- the median user has two.
IE_MAX_DOCUMENTS = 12


def author_subpool_bounds(ranks: np.ndarray, weights: np.ndarray, n_candidates: int
                          ) -> tuple[np.ndarray, np.ndarray]:
    """One user's ``(lower, upper)`` bound on being linked at least once, per pool size.

    The document-level estimator is exact because a rank is a sufficient statistic: writing ``B``
    for the set of authors ranked above the true author, the document is right in a sub-pool ``S``
    exactly when ``B & S`` is empty, which :func:`subpool_weights` gives in closed form. A *user*
    is linked when ``exists i: B_i & S = {}``, and ``S`` is shared across their documents, so the
    answer depends on ``|B_i U B_j|``, ``|B_i U B_j U B_k|``, ... -- union cardinalities the ranks
    do not pin down. See :func:`config_scaling` for why that is a missing column rather than a
    fundamental obstacle.

    Both bounds below are closed form and assumption-free about the overlap:

    * **lower** -- a union is at least its largest single event, so ``max_i P_i``, the user's best
      rank. This one is *attained*: nesting (``B_i* <= B_j`` for every j) is realisable for any
      multiset of ranks, so it is the exact minimum, not merely a bound. It is also the plausible
      case, since the same rivals tend to outrank a person across their own documents.
    * **upper** -- inclusion-exclusion with the ``B_i`` taken as disjoint as their sizes permit,
      ``|B_T| = min(sum_i (r_i - 1), N-1)``, which is the largest the union of the *events* can be.
      Boole (``min(1, sum_i P_i)``) is also valid but strictly looser: it needs the events pairwise
      disjoint, which cannot happen whenever a sub-pool can miss two ``B`` sets at once. Measured,
      the difference is 0.427 against 0.437 at 50 candidates on WildChat.

    Both coincide at ``n = n_candidates``, where each reduces to "does any document rank first".

    The subsets are enumerated as a bit matrix rather than with ``itertools``, and folded to their
    distinct capped sums before touching ``weights``, so a 12-document user costs one small matrix
    product rather than 4,096 row lookups.
    """
    rows = weights[np.minimum(ranks, n_candidates) - 1]
    lower = rows.max(axis=0)
    boole = np.minimum(rows.sum(axis=0), 1.0)
    if len(ranks) > IE_MAX_DOCUMENTS:
        return lower, boole
    beaten = ranks.astype(np.int64) - 1                 # |B_i|
    masks = ((np.arange(1, 2 ** len(ranks))[:, None] >> np.arange(len(ranks))) & 1).astype(bool)
    # A subset's union is capped at every other author, and its sign alternates with its size.
    sums = np.minimum(masks @ beaten + 1, n_candidates)
    signs = np.where(masks.sum(axis=1) % 2, 1.0, -1.0)
    distinct, inverse = np.unique(sums, return_inverse=True)
    folded = np.bincount(inverse, weights=signs, minlength=len(distinct))
    upper = np.clip(folded @ weights[distinct - 1], 0.0, 1.0)
    return lower, np.minimum(upper, boole)


def config_scaling_authors(run: Run, table: pd.DataFrame, weights: PanelWeights
                           ) -> ScalingCurve | None:
    """Author-level scaling: the share of users linked at least once, bracketed.

    The only curve in this file whose band is **not** a bootstrap interval alone. The ranks do not
    determine the author-level answer (see :func:`author_subpool_bounds`), so the shaded region
    carries two things at once: the analytic bracket between the two bounds, *and* the clustered
    bootstrap on each edge. It is therefore wider than a confidence interval and is labelled as
    such in the curve type's subtitle -- read it as "the curve is in here", not "the estimate is
    this plus or minus".

    **The line is the lower bound**, because that is the one that is attained rather than merely
    valid, and because "at least this many users are linked" is the statement a privacy claim
    wants. The two bounds meet at the run's own pool size, so the ring is exact.
    """
    if run.attack not in POOL_INTERPOLABLE_ATTACKS:
        return None
    n_candidates = int(table["n_candidate_authors"].max())
    pool_sizes = pool_lattice(n_candidates)
    interpolation = subpool_weights(n_candidates, pool_sizes)

    grouped = table.groupby("true_author", sort=True)["true_author_rank"]
    authors = []
    lower_rows, upper_rows = [], []
    for author, ranks in grouped:
        low, high = author_subpool_bounds(np.ceil(ranks.to_numpy(dtype=float)).astype(np.int64),
                                          interpolation, n_candidates)
        authors.append(author)
        lower_rows.append(low)
        upper_rows.append(high)
    lower_by_author = np.array(lower_rows)
    upper_by_author = np.array(upper_rows)

    # Uniform chance at this level: a user escapes only if every one of their documents misses,
    # so it is `1 - (1 - 1/n) ** m_a` averaged over users -- the sub-pool twin of `chance_identity`.
    counts = grouped.size().to_numpy(dtype=float)[:, None]
    chance = np.mean(1.0 - (1.0 - 1.0 / pool_sizes[None, :]) ** counts, axis=0)

    author_weights = weights.for_authors(pd.Index(authors))
    if len(author_weights):
        totals = author_weights.sum(axis=1, keepdims=True)
        replicate_low = np.divide(author_weights @ lower_by_author, totals,
                                  out=np.zeros((len(author_weights), len(pool_sizes))),
                                  where=totals > 0)
        replicate_high = np.divide(author_weights @ upper_by_author, totals,
                                   out=np.zeros((len(author_weights), len(pool_sizes))),
                                   where=totals > 0)
        # The band spans the bootstrap's low edge of the lower bound to its high edge of the
        # upper one, so it covers sampling noise and the analytic bracket together.
        band_low = band_from_replicates(replicate_low)[0]
        band_high = band_from_replicates(replicate_high)[1]
    else:
        band_low, band_high = lower_by_author.mean(axis=0), upper_by_author.mean(axis=0)

    curve = pd.DataFrame({"n_candidates": pool_sizes,
                          "accuracy": lower_by_author.mean(axis=0),
                          "upper": upper_by_author.mean(axis=0),
                          "random": chance,
                          "ci_low": band_low, "ci_high": band_high})
    best = grouped.min().to_numpy(dtype=float)
    measured = pd.DataFrame([{"n_candidates": n_candidates,
                              "accuracy": float((best <= 1).mean())}])
    return ScalingCurve(curve=curve, measured=measured, n_documents=len(table),
                        n_users=len(authors))


#: The curve families drawn across corpora as well as within one, in the order their folders read.
#: Scaling is the only family that can be: its x axis is the candidate pool, which is the single
#: axis along which the two corpora are the same experiment at different scales. Everything else
#: would be two unrelated numbers sharing an axes -- WildChat's top-1 against SWE-chat's says more
#: about the pool sizes than about either attack, which is exactly what these figures exist to
#: correct for. Both counting levels, because the doc/author split is orthogonal to the scope.
CROSS_DATASET_SCALING_KINDS = ("scaling", "scaling_authors")


def cross_dataset_stem(kind: str, view: str, name: str, output_dir: Path) -> Path:
    """Where one cross-dataset figure goes: ``cross_dataset/<family>/<level>/by_<view>/<name>``.

    The path is assembled from :data:`CURVE_TYPES`' own subdirectory rather than spelled out, so
    the family and level folders are the same names as the per-dataset tree by construction and a
    figure can be flipped against its single-corpus twin at the matching path.
    """
    return output_dir / "cross_dataset" / CURVE_TYPES[kind][1] / f"by_{view}" / name


def cross_dataset_scaling_panels(runs: list[Run], scaling: dict[Run, dict[str, ScalingCurve]],
                                 slot_of, label_of) -> dict[str, list[Series]]:
    """The runs of one comparison group, as ``tag -> series`` panels spanning every corpus.

    ``slot_of``/``label_of`` are what makes the two views one function: they say which entity owns
    the colour (the defense in ``by_defense``, the feature+attack in ``by_attack``). The dataset
    never owns a colour -- it rides on the dash -- so the same entity keeps one hue across both
    corpora, which is what lets a line be followed from one to the other.

    Returns nothing unless at least two corpora survive the filter. A single corpus here would be
    the per-dataset figure redrawn under a folder claiming otherwise, with a dataset legend of one
    entry. The guard is on the *figure*, not on each panel, so a defense only one corpus was run
    against still appears beside the ones both were.
    """
    ordered = sorted((run for run in runs if scaling.get(run)),
                     key=lambda run: (slot_of(run), DATASETS.index(run.dataset)))
    if len({run.dataset for run in ordered}) < 2:
        return {}
    panels: dict[str, list[Series]] = defaultdict(list)
    for run in ordered:
        for tag, curve in scaling[run].items():
            panels[tag].append(Series(label_of(run), slot_of(run), curve, run.dataset))
    return panels


def dataset_legend(panels: dict[str, list[Series]]) -> tuple[str, dict]:
    """The second legend of a cross-dataset figure: what the line *pattern* says.

    It holds the corpora and nothing else. The random-guessing baseline used to sit here too,
    which made the legend answer two questions at once -- "which corpus?" and "which line is not a
    measurement?" -- and left a reader deciding which of the two a dash meant. It is now an entry
    in the colour legend beside the methods, where its hue identifies it, and the pattern here
    applies to every line on the figure uniformly, chance lines included.

    Drawn in ink rather than in any series colour, so it cannot be misread as belonging to one of
    the lines.
    """
    present = [name for name in DATASETS
               if any(item.dataset == name for series in panels.values() for item in series)]
    return "Dataset", {
        DATASET_LABELS[name]: Line2D([], [], color=TEXT_SECONDARY, linewidth=LINE_WIDTH,
                                     linestyle=dataset_linestyle(name))
        for name in present}


def plot_scaling_across_datasets_by_defense(
        runs: list[Run], scaling: dict[Run, dict[str, ScalingCurve]], kind: str,
        output_dir: Path) -> list[Path]:
    """One figure per (feature, attack): every defense's scaling curve, on both corpora at once.

    The cross-dataset twin of :func:`plot_defense_comparisons`, and it asks the same question --
    *does the defense work?* -- of an axis only the two corpora together can span. Within a
    dataset the candidate pool covers a factor of a few; across SWE-chat and WildChat it covers
    more than two orders of magnitude, so "the defense helps" can be separated here from "the
    pool was small".

    Colour is the defense and dash is the corpus. Lines never join two datasets: the axis is
    shared, the experiments are not.

    ``kind`` is the counting level (``scaling`` per document, ``scaling_authors`` per person), and
    it is worth flipping between them here for the same reason as everywhere else: the two are
    weighted differently -- micro against macro -- so their curves cross.
    """
    written = []
    by_method: dict[tuple[str, str], list[Run]] = defaultdict(list)
    for run in runs:
        by_method[run.method].append(run)
    for method, method_runs in sorted(by_method.items(), key=lambda item: METHOD_SLOTS[item[0]]):
        panels = cross_dataset_scaling_panels(method_runs, scaling,
                                              lambda run: DEFENSE_SLOTS[run.defense],
                                              lambda run: run.defense_label)
        if not panels:
            continue
        feature, attack = method
        written.append(plot_config_comparison(
            kind, panels,
            title=f"Defenses under {FEATURE_LABELS[feature]} / {ATTACK_LABELS[attack]}, "
                  f"both corpora",
            legend_title="Defense",
            stem=cross_dataset_stem(kind, "defense", f"{feature}_{attack}", output_dir),
            extra_legend=dataset_legend(panels)))
    return written


def plot_scaling_across_datasets_by_attack(
        runs: list[Run], scaling: dict[Run, dict[str, ScalingCurve]], kind: str,
        output_dir: Path) -> list[Path]:
    """One figure per defense: every feature+attack's scaling curve, on both corpora at once.

    The transpose of :func:`plot_scaling_across_datasets_by_defense`, and the cross-dataset twin
    of :func:`plot_attack_comparisons`: with the defense held fixed it says which representation
    and estimator the attacker should reach for, at a *matched* pool size rather than at whatever
    pool each corpus happens to have. ``by_attack/base.pdf`` is the undefended figure -- how far
    the threat reaches before any countermeasure.

    Colour is the feature+attack and dash is the corpus, so one method can be followed across
    both -- which is the whole point, since the interesting comparison is the same attack at a
    matched pool size rather than either corpus on its own.
    """
    written = []
    by_defense: dict[str, list[Run]] = defaultdict(list)
    for run in runs:
        by_defense[run.defense].append(run)
    for defense, defense_runs in sorted(by_defense.items(),
                                        key=lambda item: DEFENSE_SLOTS[item[0]]):
        panels = cross_dataset_scaling_panels(defense_runs, scaling,
                                              lambda run: METHOD_SLOTS[run.method],
                                              lambda run: run.method_label)
        if not panels:
            continue
        written.append(plot_config_comparison(
            kind, panels,
            title=f"Attacks against {DEFENSE_LABELS[defense]}, both corpora",
            legend_title="Feature / attack",
            stem=cross_dataset_stem(kind, "attack", defense, output_dir),
            extra_legend=dataset_legend(panels)))
    return written


# --- per-run figures ---------------------------------------------------------
#
# These are the figures `run_experiment.py` and `run_experiment.py` used to draw at the end of
# a run. They now live here, rebuilt from the same CSVs, so that the runners only produce numbers
# and every figure in the project comes from one place.

def plot_config_cmc(cmc: pd.DataFrame, title: str, stem: Path) -> Path:
    """One run's CMC curves, one line per known configuration -- the whole-future detail view.

    Unlike the comparison figures this reads ``cmc_results.csv``, which each configuration wrote
    over its **entire** unknown side rather than the shared test set. That is the point of having
    it: the shared test set is what makes configurations comparable, and this is what each one
    actually achieved against everything it was asked to attribute.

    The *shape* is the interesting part: a curve that shoots up and flattens means the attack is
    confidently right about a subset, while one that climbs steadily means it is merely narrowing
    a large pool, and the two can share a top-1.
    """
    groups = sorted(cmc.groupby("known_config"), key=lambda item: str(item[0]))
    figure, axes = plt.subplots(figsize=(7.4, 4.7))
    figure.patch.set_facecolor(SURFACE)
    for slot, (tag, group) in enumerate(groups):
        config = parse_config_tag(str(tag))
        group = group.sort_values("k")
        color = series_style(slot)
        axes.plot(group["k"], group["accuracy"], color=color, linewidth=LINE_WIDTH,
                  solid_capstyle="round", zorder=3,
                  label=config.label if config else str(tag))
        axes.plot(group["k"], group["random"], color=TEXT_MUTED, linewidth=1.0,
                  linestyle=BASELINE_DASH, alpha=0.6, zorder=2)
    style_axes(axes, "k (candidate authors returned)", "Top-k accuracy", title,
               "One line per known configuration, over its whole future; grey dashes are that "
               "configuration's random baseline")
    axes.set_xscale("log")
    axes.set_ylim(0, 1.02)
    add_legend(axes, loc="upper left", ncols=2)
    figure.tight_layout()
    return save_figure(figure, stem)


def plot_pool_growth(sweep: pd.DataFrame, top_k: int, title: str, stem: Path) -> Path:
    """Identity accuracy against the number of target users, one point per known configuration.

    Each point is a real experiment, and the pool grows because the known side does. Solid is the
    attack, dashed grey the random baseline for that same pool -- both move together, so only the
    gap between them means anything. The points are *not* a controlled series: they differ in how
    much data the attacker holds and in how stale it is at once, which is what the facet grid
    exists to separate. Read it as a sanity check on scale, not as a trend.
    """
    figure, axes = plt.subplots(figsize=(7.0, 4.4))
    figure.patch.set_facecolor(SURFACE)
    sweep = sweep.sort_values("n_identities")
    axes.plot(sweep["n_identities"], sweep["id_acc"], color=CATEGORICAL[0], linewidth=LINE_WIDTH,
              marker="o", markersize=MARKER_SIZE, markeredgecolor=SURFACE, markeredgewidth=2,
              zorder=3, label="measured")
    axes.plot(sweep["n_identities"], sweep["random_id"], color=TEXT_MUTED, linewidth=1.2,
              linestyle=BASELINE_DASH, zorder=2)
    for _, row in sweep.iterrows():
        config = parse_config_tag(str(row.get("known_config", "")))
        if config is not None:
            axes.annotate(config.tag[len("known"):], (row["n_identities"], row["id_acc"]),
                          textcoords="offset points", xytext=(0, 9), ha="center",
                          fontsize=7.5, color=TEXT_MUTED)
    style_axes(axes, "Target users on the unknown side", f"Top-{top_k} identification accuracy",
               title, "Grey dashes are random guessing over the same pool")
    axes.set_ylim(bottom=0)
    add_legend(axes, loc="best")  # the lines can sit anywhere in the frame; let it find the gap
    figure.tight_layout()
    return save_figure(figure, stem)


def plot_topk_bars(headline: pd.DataFrame, title: str, stem: Path) -> Path:
    """Identity accuracy against the random baseline at each measured k, as paired bars.

    ``headline`` needs the columns ``top``, ``id_acc`` and ``random_id``.
    """
    # Thin bars with air around them, and a gap between the pair rather than a stroke drawn
    # round each -- the leftover width in the slot is what separates one k from the next.
    positions = np.arange(len(headline), dtype=float)
    width, gap = 0.24, 0.03
    figure, axes = plt.subplots(figsize=(6.4, 3.9))
    figure.patch.set_facecolor(SURFACE)
    axes.bar(positions - (width + gap) / 2, headline["id_acc"], width, color=CATEGORICAL[0],
             label="Attack", zorder=3)
    axes.bar(positions + (width + gap) / 2, headline["random_id"], width, color=AXIS,
             label="Random guessing", zorder=3)
    for position, value in zip(positions, headline["id_acc"]):
        axes.annotate(f"{value:.3f}", (position - (width + gap) / 2, value),
                      textcoords="offset points", xytext=(0, 4), ha="center",
                      color=TEXT_SECONDARY, fontsize=8.5)
    axes.set_xticks(positions, [f"top-{int(k)}" for k in headline["top"]])
    style_axes(axes, "", "Identification accuracy", title)
    axes.grid(axis="x", visible=False)
    axes.set_ylim(0, max(headline["id_acc"].max(), headline["random_id"].max()) * 1.12)
    add_legend(axes, loc="upper left")
    figure.tight_layout()
    return save_figure(figure, stem)


def rolling_headline(rolling: pd.DataFrame, row, top_ks=(1, 5, 10)) -> pd.DataFrame:
    """The per-k headline table for one window, rebuilt from ``rolling_results.csv``.

    That file is one row per window with the k values widened into ``id_acc1``/``id_acc5``/...
    columns; the bar chart wants them back as one row per k.
    """
    available = [k for k in top_ks if f"id_acc{k}" in rolling.columns]
    return pd.DataFrame({"top": available,
                         "id_acc": [row[f"id_acc{k}"] for k in available],
                         "random_id": [row[f"random_id{k}"] for k in available]})


def plot_run_detail(run: Run, output_dir: Path, sweep_top_k: int = 1) -> list[Path]:
    """Every per-run figure the run's own CSVs support, into ``per_run/<run>/``.

    Every figure here comes from ``cmc_results.csv`` and ``rolling_results.csv``, so a directory
    missing one simply produces fewer figures rather than failing.
    """
    stem_dir = output_dir / "per_run" / run.directory.name
    written, scope = [], f"{run.defense_label}, {run.method_label}"

    cmc_path = run.directory / "cmc_results.csv"
    if cmc_path.exists():
        cmc = pd.read_csv(cmc_path)
        if "attack" in cmc.columns:
            cmc = cmc[cmc["attack"] == run.attack]
        if "known_config" in cmc.columns and not cmc.empty:
            written.append(plot_config_cmc(
                cmc, f"{DATASET_LABELS[run.dataset]}: {scope}", stem_dir / "cmc_curve"))

    rolling_path = run.directory / "rolling_results.csv"
    if rolling_path.exists():
        rolling = pd.read_csv(rolling_path)
        rolling = rolling[rolling["attack"] == run.attack] if "attack" in rolling else rolling
        if f"id_acc{sweep_top_k}" in rolling.columns and "known_config" in rolling.columns:
            sweep = rolling.rename(columns={f"id_acc{sweep_top_k}": "id_acc",
                                            f"random_id{sweep_top_k}": "random_id"})
            written.append(plot_pool_growth(
                sweep, sweep_top_k, f"{DATASET_LABELS[run.dataset]}: {scope}",
                stem_dir / f"pool_growth_top{sweep_top_k}"))
        for _, row in rolling.iterrows():
            headline = rolling_headline(rolling, row)
            if headline.empty or "known_config" not in rolling.columns:
                continue
            config = parse_config_tag(str(row["known_config"]))
            written.append(plot_topk_bars(
                headline,
                f"{DATASET_LABELS[run.dataset]}: {scope}\n"
                f"{config.label if config else row['known_config']} → rest",
                stem_dir / f"topk_accuracy_{row['known_config']}"))

    return written


# --- driver ------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """The one knob this script has: how many bootstrap replicates the bands are drawn from.

    There is no window flag any more, and nothing here chooses what is measured. The experiment
    fixes that: ``run_experiment.py`` holds out the final :data:`TEST_FRACTION` of the corpus
    from every known side, so which documents a comparison is made on is a property of the run
    rather than a decision taken at plot time. What is left to choose is only how finely the
    uncertainty is estimated.
    """
    parser = argparse.ArgumentParser(
        description="Draw every figure in the project from experiments/results/.")
    parser.add_argument("--bootstrap", type=int, default=BOOTSTRAP_REPLICATES, metavar="N",
                        help=f"Replicates behind every band, resampling *users* with replacement "
                             f"(default: {BOOTSTRAP_REPLICATES}; 0 draws the curves with no "
                             f"band). One draw is shared by every configuration and run of a "
                             f"dataset, so a replicate that drops a user drops them from every "
                             f"panel at once. Lower it for a fast redraw, not for a figure "
                             f"anyone will read.")
    parser.add_argument("--jobs", type=int, default=default_workers(), metavar="N",
                        help=f"Worker processes for the curve and figure passes (default: "
                             f"{default_workers()}, capped rather than every core because each "
                             f"worker holds its own replicate weight matrix). 1 runs everything "
                             f"in this process, which is what to use when a drawing routine "
                             f"raises -- a traceback from a forked child loses its outer frames.")
    parser.add_argument("--png", action="store_true",
                        help="Also write a 200 dpi PNG beside every PDF. Off by default: the "
                             "PNGs cost more than the PDFs they accompany (79 s against 47 s "
                             "over a full sweep) and only the PDF goes into a paper.")
    parser.add_argument("--per-run", action="store_true",
                        help="Also draw per_run/<run>/ -- each run's own CMC, pool-growth and "
                             "top-k bar figures. Off by default: they are 160 of the 262 figures "
                             "a full sweep writes and none of them compares runs, so they are "
                             "diagnostics rather than results.")
    return parser.parse_args()


def run_curves(run: Run, tables: dict[str, pd.DataFrame], bootstrap: AuthorBootstrap,
               open_tables: dict[str, pd.DataFrame], open_bootstrap: AuthorBootstrap) -> dict:
    """Every curve one run contributes, keyed by curve family then by configuration.

    The unit of work :func:`build_curves` hands to a worker. It is one *run* rather than one
    (run, configuration) cell because the counting-mode figure needs the run's whole table set to
    pick :data:`HEADLINE_CONFIG` out of it, and because a cell is small enough that per-job
    overhead would start to show.

    Each table's replicate weights are built **once** here and shared by every curve family that
    reads it -- see :class:`PanelWeights` for why that matters.
    """
    curves: dict[str, dict] = {family: {} for family in CURVE_FAMILIES}
    weights = {tag: PanelWeights(bootstrap, table["true_author"])
               for tag, table in tables.items()}
    for tag, table in tables.items():
        panel = weights[tag]
        baseline = config_baseline(run.dataset, tag, table["true_author"])
        curves["cmc"][tag] = config_cmc(table, panel, baseline)
        curves["identity"][tag] = config_identity(table, panel, baseline)
        curves["selective"][tag] = config_risk_coverage(table, panel)
        curves["selective_authors"][tag] = config_risk_coverage_authors(table, bootstrap)
        curves["exposure"][tag] = config_author_risk(table, panel)
        # Two sides x two counting levels, all four from the same table and the same weights.
        # The `known` pair needs the corpus parquet and is absent without it (`warm_baselines`
        # is what reports that); the `test` pair reads nothing but the predictions.
        for side in ("known", "test"):
            for level, suffix in (("document", ""), ("author", "_authors")):
                ndocs = config_ndocs(table, panel, run.dataset, tag, side=side, level=level)
                if ndocs is not None:
                    curves[f"ndocs_{side}{suffix}"][tag] = ndocs
        pool_curve = config_scaling(run, table, panel)
        if pool_curve is not None:
            curves["scaling"][tag] = pool_curve
            curves["scaling_authors"][tag] = config_scaling_authors(run, table, panel)
    curves["modes"] = counting_modes(tables, weights)

    for tag, table in open_tables.items():
        panel = PanelWeights(open_bootstrap, table["true_author"])
        curves["coverage"][tag] = config_openset_coverage(table, panel)
        curves["detection"][tag] = config_detection(table, panel)
        curves["separation"][tag] = config_separation(table, panel)
        # The author level is the same three curves over one row per person. `author_table`
        # collapses the table and the weights follow it, so the builders are reused verbatim --
        # only the two coverage families need their own, because "any answered document" is not
        # a property one collapsed row can carry (see `weighted_selective_any`).
        people = author_table(table)
        people_panel = PanelWeights(open_bootstrap, people["true_author"])
        curves["coverage_authors"][tag] = config_openset_coverage_authors(table, open_bootstrap)
        curves["detection_authors"][tag] = config_detection(people, people_panel,
                                                             level="author")
        curves["separation_authors"][tag] = config_separation(people, people_panel,
                                                              level="author")
    return curves


def warm_baselines(runs: list[Run], tables: dict[Run, dict[str, pd.DataFrame]]) -> None:
    """Fill :func:`known_inclusion`'s memo in this process, before the curve workers fork.

    The Monte Carlo behind the proportional baseline depends only on (dataset, known
    configuration), so a dataset's 21 runs want six results between them. Under ``fork`` a child
    inherits whatever the parent has already computed, so filling the memo here means it is paid
    for six times rather than once per worker -- and the workers, which would each have filled
    their own copy, get it for free.
    """
    wanted = {(run.dataset, tag) for run in runs for tag in tables[run]}
    missing = sorted(key for key in wanted if key not in _INCLUSION)
    if not missing:
        return
    for key in missing:
        known_inclusion(*key)
    absent = sorted({dataset for dataset, tag in missing if _INCLUSION[(dataset, tag)] is None})
    if absent:
        print(f"  no {', '.join(absent)}.parquet under {DATA_DIR} -- the proportional baseline "
              f"needs the known side's document counts, falling back to uniform 1/N")


def build_curves(runs: list[Run], bootstrap_replicates: int, workers: int = 1):
    """Every curve every figure needs, built once per (run, configuration).

    One pass over the predictions: the same table feeds the CMC, risk-coverage, per-user risk,
    scaling and counting-mode curves, and each run's curve object is reused by both comparison
    families rather than rebuilt for each. Runs whose files predate the configuration design are
    reported and skipped -- their known sides were prefixes scored on different documents, so
    they are not cells of this grid.

    **Two populations, two bootstraps.** The in-set figures resample the users the attack could
    have attributed; the ``openset/`` figures resample the users who *appear* in the test set,
    which on WildChat is eight times as many. They are deliberately separate draws rather than
    one draw over the union: the in-set bootstrap's universe is what makes its six panels paired
    (a replicate that drops a user drops them from all of them at once), and widening that
    universe with users no in-set panel contains would change every band on every existing figure
    to no purpose. Within each family the one-draw property is preserved.

    **The tables are read here and the curves are built in workers.** Reading stays in this
    process because both bootstraps have to be drawn over the union of every run's users before
    any curve can be computed, and because the tables are the one thing too big to want to send
    anywhere -- the workers inherit them through :func:`run_jobs`'s fork instead. What comes back
    is only the curves, which are a few MB per run.
    """
    tables = {run: config_predictions(run) for run in runs}
    stale = [run for run in runs if not tables[run]]
    for run in stale:
        print(f"  {run.directory.name}: no predictions_<attack>_known<XXYY>.csv -- this run "
              f"predates the known-configuration design, re-run it to appear in the figures")

    # One resample of the dataset's test-side users, shared by every panel: see AuthorBootstrap.
    authors = {author for run in runs for table in tables[run].values()
               for author in table["true_author"].unique()}
    bootstrap = AuthorBootstrap(authors, n_replicates=bootstrap_replicates)

    open_tables = {run: config_predictions(run, in_set_only=False) for run in runs}
    open_authors = {author for run in runs for table in open_tables[run].values()
                    for author in table["true_author"].unique()}
    open_bootstrap = AuthorBootstrap(open_authors, n_replicates=bootstrap_replicates)

    live = [run for run in runs if tables[run]]
    warm_baselines(live, tables)
    jobs = [(run_curves, (run, tables[run], bootstrap, open_tables[run], open_bootstrap), {})
            for run in live]
    results = run_jobs(jobs, workers)

    # Keyed by family rather than unpacked into a tuple: there are twelve of them once both
    # counting levels exist, and a twelve-element tuple is a positional bug waiting to happen.
    built: dict[str, dict] = {family: {} for family in CURVE_FAMILIES}
    modes = {}
    for run, curves in zip(live, results):
        for family in CURVE_FAMILIES:
            # A family is absent for a run when it had nothing to build -- an attack outside
            # POOL_INTERPOLABLE_ATTACKS for scaling, a run with no open-set rows for the rest.
            if curves[family]:
                built[family][run] = curves[family]
        if not curves["scaling"] and run.attack not in POOL_INTERPOLABLE_ATTACKS:
            print(f"  {run.directory.name}: {ATTACK_LABELS[run.attack]} refits against the "
                  f"gallery, so sub-pool interpolation is not exact -- scaling skipped")
        if curves["modes"] is not None:
            modes[run] = curves["modes"]
    reach = openset_reach({run: open_tables[run] for run in runs if open_tables[run]})
    return tables, bootstrap, built, modes, reach


def dataset_figure_jobs(dataset: str, runs: list[Run], built: dict, modes: dict,
                        reach: pd.DataFrame, decay: dict, output_dir: Path,
                        per_run: bool) -> list[tuple]:
    """One dataset's figures as ``(callable, args, kwargs)`` jobs, not yet drawn.

    Splitting the *deciding* from the *drawing* is what lets every figure in the project be drawn
    by one pool: the jobs from both corpora are pooled together before any of them runs, so a
    worker that finishes swe-chat's cheap panels picks up a WildChat one instead of idling.

    The two comparison families are already one job per (curve type, view). The separation
    families are sharded here to one job per run -- they draw one figure per run anyway, and as a
    single job that was the longest pole in the pass at 11 s.
    """
    jobs = []
    for family, kind in CURVE_FAMILIES.items():
        if family in PER_RUN_FAMILIES:
            jobs += [(plot_separation_figures, (dataset, [run], built[family], kind, output_dir), {})
                     for run in runs if run in built[family]]
            continue
        jobs.append((plot_defense_comparisons, (dataset, runs, built[family], kind, output_dir), {}))
        jobs.append((plot_attack_comparisons, (dataset, runs, built[family], kind, output_dir), {}))
    jobs.append((plot_macro_micro, (dataset, runs, modes, output_dir), {}))
    jobs.append((plot_openset_reach, (dataset, reach, output_dir), {}))
    for level_decay in decay.values():
        if not level_decay:
            continue
        temporal_runs = [run for run in runs if run in level_decay]
        jobs.append((plot_temporal_defense_comparisons,
                     (dataset, temporal_runs, level_decay, output_dir), {}))
        jobs.append((plot_temporal_attack_comparisons,
                     (dataset, temporal_runs, level_decay, output_dir), {}))
    if per_run:
        jobs += [(plot_run_detail, (run, output_dir), {}) for run in runs]
    return jobs


def main() -> None:
    args = parse_args()
    global WRITE_PNG
    WRITE_PNG = args.png

    if not RESULTS_DIR.exists():
        raise SystemExit(f"{RESULTS_DIR} does not exist -- run an experiment first.")

    runs = discover_runs(RESULTS_DIR)
    if not runs:
        raise SystemExit(f"no runs named <dataset>_<defense>_<feature>_<attack> under {RESULTS_DIR}")

    by_dataset: dict[str, list[Run]] = defaultdict(list)
    for run in runs:
        by_dataset[run.dataset].append(run)

    jobs: list[tuple] = []
    # Both counting levels, accumulated across datasets -- the cross-dataset figures are the one
    # place a curve from one corpus is drawn beside a curve from another.
    all_scaling: dict[str, dict[Run, dict[str, ScalingCurve]]] = {
        kind: {} for kind in CROSS_DATASET_SCALING_KINDS}
    for dataset in DATASETS:
        dataset_runs = by_dataset.get(dataset, [])
        if not dataset_runs:
            continue
        output_dir = PLOTS_DIR / dataset
        print(f"\n[{DATASET_LABELS[dataset]}] {len(dataset_runs)} run(s) -> {output_dir}/")

        tables, bootstrap, built, modes, reach = build_curves(dataset_runs, args.bootstrap,
                                                              args.jobs)
        for kind in CROSS_DATASET_SCALING_KINDS:
            all_scaling[kind].update(built[kind])

        decay = {level: {} for level in ("doc", "author")}
        for run in dataset_runs:
            for level, name in (("doc", "document"), ("author", "author")):
                staleness = temporal_accuracy(run, level=name)
                if staleness is None:
                    if level == "doc":
                        print(f"  {run.directory.name}: no predictions for "
                              f"{TEMPORAL_KNOWN_CONFIG} (or no split parquet for its timestamps), "
                              f"temporal decay skipped")
                else:
                    decay[level][run] = staleness

        jobs += dataset_figure_jobs(dataset, dataset_runs, built, modes, reach, decay,
                                    output_dir, args.per_run)

    # Figures that span datasets go to plots/cross_dataset/ rather than under either corpus:
    # pool size is the one axis where the two are the same experiment at different scales, and
    # filing that under one of them would imply it belongs to that one. Below that they follow the
    # per-dataset tree exactly -- both counting levels, both comparison views, the same 3x3
    # configuration grid -- so a figure and its single-corpus twin sit at matching paths.
    for kind in CROSS_DATASET_SCALING_KINDS:
        jobs.append((plot_scaling_across_datasets_by_defense,
                     (runs, all_scaling[kind], kind, PLOTS_DIR), {}))
        jobs.append((plot_scaling_across_datasets_by_attack,
                     (runs, all_scaling[kind], kind, PLOTS_DIR), {}))

    print(f"\nDrawing {len(jobs)} figure group(s) across {min(args.jobs, len(jobs))} process(es)")
    written = [path for group in run_jobs(jobs, args.jobs) for path in group]
    for path in sorted(written):
        print(f"  {path.relative_to(PLOTS_DIR)}")

    formats = "PDF + PNG" if WRITE_PNG else "PDF"
    print(f"\nWrote {len(written)} figure(s) ({formats}) to {PLOTS_DIR}/")
    if not args.per_run:
        print("per_run/ diagnostics skipped -- pass --per-run to draw them")


if __name__ == "__main__":
    main()
