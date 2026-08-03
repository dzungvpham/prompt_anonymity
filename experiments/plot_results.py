#!/usr/bin/env python
"""Render every figure this project publishes, from the result CSVs already on disk.

Run it with no arguments::

    python experiments/plot_results.py

Nothing here recomputes an attack: the runners (``run_experiment_v2.py`` and
``run_experiment.py``) write CSVs and stop, and this script turns the accumulated CSVs into
figures. Re-running it is cheap and idempotent, so it is the right thing to run after any new
experiment finishes.

Input is ``experiments/results/<dataset>_<defense>_<feature>_<attack>/``. That four-part name is
the contract -- it is what lets one run be compared against another without opening it -- and each
part must be a name this file knows (:data:`DATASETS`, :data:`DEFENSES`, :data:`FEATURES`,
:data:`ATTACKS`). An undefended run is spelled ``base``, not omitted. Any other directory is
skipped with a note, which is how exploratory output (``tuned_comparison/`` and friends) stays out
of the figures.

Output is ``experiments/plots/<dataset>/``. Four curve types are drawn, one directory each, and
every one of them gets the same two views: ``by_defense/<feature>_<attack>.pdf`` (attack fixed, a
line per defense -- *does the defense work?*) and ``by_method/<defense>.pdf`` (defense fixed, a
line per feature+attack -- *which attack is strongest?*). The layout is uniform on purpose: the
same run appears at the same relative path under every curve type.

``accuracy/by_{defense,method}/``
    **CMC** -- top-k accuracy against k. The headline privacy question.
``risk_coverage/by_{defense,method}/``
    **Risk-coverage** -- precision when the attack answers only its most confident documents.
    What a headline accuracy hides: an attack that is usually wrong but knows when it is right.
``author_risk/by_{defense,method}/``
    **Per-user risk** -- each user's own accuracy, sorted from most to least exposed. Who carries
    the risk, rather than what it averages to.
``scaling/by_{defense,method}/``
    **Scale** -- top-1 accuracy against the size of the candidate pool. Whether the threat is an
    artefact of a small pool. Drawn only for the attacks in :data:`POOL_INTERPOLABLE_ATTACKS`,
    the ones whose scores do not depend on which other authors are enrolled.

Two figures have no ``by_defense``/``by_method`` split. ``accuracy/macro_micro.pdf`` puts every
run's top-1 next to itself counted three ways -- per document, per user, per identity -- because
which one a paper leads with is a claim about what "anonymity failed" means, not a detail; it
lives under ``accuracy/`` because it is the same number those curves start from.
``plots/cross_dataset/scaling.pdf`` is the only figure outside the per-dataset folders: pool size
is the single axis along which the corpora are the same experiment at different scales, so filing
it under either one would imply it belonged to that one.

``per_run/<run>/`` keeps the per-run figures the runners used to write themselves -- the
per-window CMC curves, the window sweep, and the top-k bars -- so one experiment's own detail is
still available, now regenerated rather than baked in at run time.

The CMC groups plot the **curve averaged over the rolling windows**: each run sweeps a
grid of known/unknown splits, and a single curve per run is what makes several runs comparable on
one axes. The mean is unweighted (every window counts once, so a big window cannot drown the
others) and is taken only over the k values every window shares -- the windows have different
candidate-pool sizes, so their curves have different lengths, and averaging past the shortest one
would silently change how many windows each point is an average of. That truncation is why an
averaged curve stops short of 1.0 at its right edge even though every individual window's curve
reaches exactly 1.0 at its own pool size: at the shared cutoff, the windows with larger pools are
still climbing. Each line carries a shaded 95 % interval for its mean across those windows. Every
figure is written alongside a ``.csv`` of the exact numbers plotted.
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
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

#: Where the built dataset lives, and the split parquet each dataset's documents come from --
#: the same defaults ``run_experiment_v2.py`` attacks. Only the temporal figure reads them, and
#: only for two columns: a document's ``ended_at`` is metadata that never depended on the attack,
#: so it is joined back on ``doc_id`` rather than copied into every run's predictions.
DATA_DIR = REPO_ROOT / "data" / "hf"
SPLIT_NAMES = {"wildchat": "wildchat", "swe-chat": "swe_chat"}


# --- the vocabulary a results directory name is built from -------------------
#
# These are the only accepted values for each part of `<dataset>_<defense>_<feature>_<attack>`.
# Keeping them as literals (rather than importing the package registries) keeps this script free
# of the heavy imports those registries pull in, at the cost of having to be kept in sync:
# DEFENSES mirrors `prompt_anonymity.defenses.DEFENSES` (with "none" spelled "base"), FEATURES
# mirrors `prompt_anonymity.features.FEATURIZERS`, and ATTACKS mirrors
# `prompt_anonymity.attacks.ATTRIBUTION_ATTACKS`. A run whose name uses a value missing here is
# skipped, not guessed at -- so adding a new defense or attack means adding it here too.

DATASETS = ("wildchat", "swe-chat")

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
    "dp_mlm",
    "dp_mlm_pii",
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
)

DATASET_LABELS = {"wildchat": "WildChat", "swe-chat": "SWE-chat"}
DEFENSE_LABELS = {
    NO_DEFENSE: "No defense",
    "styleremix": "StyleRemix",
    "openanonymity": "OpenAnonymity",
    "styleremix_openanon": "StyleRemix + OpenAnonymity",
    "dp_mlm": "DP-MLM",
    "dp_mlm_pii": "DP-MLM (PII only)",
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
    "rlsc": "RLSC",
    "svm": "SVM",
    "xgboost": "XGBoost",
}

#: Reading order for methods, and the colour slot each one owns: feature-major, so a figure's
#: legend runs feature by feature and two runs of the same feature sit next to each other.
METHOD_SLOTS = {(feature, attack): feature_index * len(ATTACKS) + attack_index
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
DATASET_DASHES = {"wildchat": (), "swe-chat": (7, 2, 1.5, 2)}

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


def save_figure(figure, stem: Path) -> Path:
    """Write ``figure`` as both PDF (for papers) and PNG (for a quick look), return the PDF."""
    stem.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = stem.parent / f"{stem.name}.pdf"
    for path in (pdf_path, stem.parent / f"{stem.name}.png"):
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

    Every part may itself contain underscores (``swe-chat``, ``dp_mlm_pii``,
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


# --- averaging the rolling windows -------------------------------------------

#: Windows (as fractions of the whole timeline) that :func:`split_windows` cuts each known side
#: into, unless ``--window`` overrides them. These are the three the runner itself used to sweep,
#: so the default figures are the ones this project has always drawn.
DEFAULT_WINDOWS = (0.10, 0.25, 0.50)

#: The windows every figure is actually cut into, replaced by ``--window`` in :func:`main`. Module
#: state rather than a threaded argument because it is one global statement about what the figures
#: mean, read by four otherwise unrelated curve builders; there is exactly one writer.
WINDOWS: tuple[float, ...] = DEFAULT_WINDOWS


def split_windows(table: pd.DataFrame, windows: tuple[float, ...] | None = None) -> list[pd.DataFrame]:
    """Cut one known side's per-document table into the rolling windows to average over.

    ``run_experiment_v2.py`` no longer sweeps a window axis: it fits on the known side and scores
    **everything after it**, once, because the fit never depended on how much of the future was
    being scored. The windows are therefore cut here instead -- a window of *w* is the documents
    up to ``w`` of the whole timeline past the known cut, exactly the slice that used to be its
    own run. ``position`` (the document's index in the ordered corpus) makes that exact, with no
    rounding to reproduce.

    Why cut at all, rather than plot the one full-timeline curve: the window is the unit every
    figure here averages over, and the spread across windows is what the shaded band reports.
    One curve per known side would leave three samples and a band from ``t(df=2) = 4.303``.

    Windows that run past the end of the corpus are dropped rather than truncated, so a 50%
    window is never silently a 40% one. Tables with a ``window`` column (written by the sweeping
    runner) are split on it instead, and anything with neither column is one window -- which is
    what keeps results directories from before either change readable.
    """
    windows = WINDOWS if windows is None else windows
    if "window" in table.columns:
        return [group for _, group in table.groupby("window", sort=True)]
    if "position" not in table.columns or table.empty:
        return [table]
    start = int(table["position"].min())
    n_documents = int(table["position"].max()) + 1   # the unknown side always runs to the end
    # The requested known fraction, not `start / n_documents`: the runner cut at
    # `round(fraction * n_documents)`, so re-deriving the fraction from the cut and rounding
    # again can land a document either side of where the sweeping runner put the boundary.
    fraction = (float(table["known_fraction"].iloc[0]) if "known_fraction" in table.columns
                else start / n_documents)
    cuts = []
    for window in windows:
        end = int(round((fraction + window) * n_documents))
        if end > n_documents:                        # off the end of the timeline: not comparable
            continue
        cut = table[table["position"] < end]
        if not cut.empty:
            cuts.append(cut)
    return cuts or [table]


def prediction_windows(run: Run, windows: tuple[float, ...] | None = None) -> list[pd.DataFrame]:
    """``run``'s per-document predictions, cut into windows and restricted to the in-set rows.

    The single source every windowed figure derives from. Out-of-set documents (whose author is
    absent from the known side) are dropped here rather than by each caller: they have no correct
    answer available, so they have no rank, and counting them would cap every curve below 1 for a
    reason the attack cannot control.

    Empty when the run predates ``true_author_rank`` -- callers fall back to the aggregate CSVs.
    """
    tables = []
    for path in sorted(run.directory.glob(f"predictions_{run.attack}_*.csv")):
        table = pd.read_csv(path)
        if "attack" in table.columns:
            table = table[table["attack"] == run.attack]
        if "true_author_rank" not in table.columns:
            return []
        for window in split_windows(table, windows):
            in_set = window[window["author_in_known"].astype(bool)]
            in_set = in_set[in_set["true_author_rank"].notna()]
            if not in_set.empty:
                tables.append(in_set)
    return tables


def cmc_from_ranks(table: pd.DataFrame) -> pd.DataFrame:
    """One window's CMC curve, rebuilt from per-document true-author ranks.

    Mirrors :func:`prompt_anonymity.metrics.ranking.cmc_curve` -- ``accuracy`` is the share of
    documents whose true author ranks within k and ``random`` is ``mean(min(k, pool) / pool)`` --
    but is written out here rather than imported, to keep this script free of the package's heavy
    imports (the same reason its name vocabulary is literal).

    Both are computed by prefix sum over sorted values rather than a ``k x documents`` comparison,
    which at WildChat's scale would be a 19,711 x 130,000 boolean array per window.
    """
    ranks = np.sort(table["true_author_rank"].to_numpy(dtype=float))
    pool = np.sort(table["n_candidate_authors"].to_numpy(dtype=float))
    n_documents = len(ranks)
    ks = np.arange(1, int(pool[-1]) + 1)
    accuracy = np.searchsorted(ranks, ks, side="right") / n_documents
    # A document whose pool is <= k is a certain hit at k and contributes 1; the rest contribute
    # k / pool, summed via the running total of 1 / pool over the pools still larger than k.
    inverse = np.concatenate([[0.0], np.cumsum(1.0 / pool)])
    saturated = np.searchsorted(pool, ks, side="right")
    chance = (saturated + ks * (inverse[-1] - inverse[saturated])) / n_documents
    return pd.DataFrame({"k": ks, "accuracy": accuracy, "random": chance})


#: Student-t multipliers for a two-sided 95 % interval, indexed by degrees of freedom (n - 1).
#: Tabulated rather than pulled from ``scipy.stats`` to keep this script's imports light; the
#: window grid is small and fixed (eight windows -> df 7 -> 2.365), and anything past the table
#: is close enough to the normal limit that the last entry is used.
T_95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
        8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145,
        15: 2.131, 20: 2.086, 30: 2.042, 60: 2.000}


def t_multiplier(n_samples: int) -> float:
    """Two-sided 95 % Student-t multiplier for a mean of ``n_samples`` observations."""
    if n_samples < 2:
        return 0.0
    degrees = n_samples - 1
    return T_95.get(degrees, T_95[min(T_95, key=lambda df: abs(df - degrees))])


def add_interval(curve: pd.DataFrame, spread: pd.Series, n_samples: int,
                 column: str = "accuracy") -> pd.DataFrame:
    """Add ``ci_low``/``ci_high`` around ``curve[column]`` from a per-row standard deviation.

    ``spread`` is the standard deviation across the ``n_samples`` windows at each row, aligned to
    ``curve``'s index. The bounds are clipped to [0, 1] because every quantity plotted here is a
    proportion; a single window (or a row where every window agreed exactly) gets a zero-width
    band rather than a NaN.
    """
    half_width = t_multiplier(n_samples) * spread.fillna(0.0) / np.sqrt(n_samples)
    curve["ci_low"] = (curve[column] - half_width).clip(lower=0.0)
    curve["ci_high"] = (curve[column] + half_width).clip(upper=1.0)
    return curve


@dataclass
class AveragedCmc:
    """One run's CMC curve, averaged over its rolling windows.

    ``curve`` has one row per k with the mean ``accuracy``, the mean ``random`` baseline, and the
    ``ci_low``/``ci_high`` bounds of the band around the mean; ``n_windows`` is how many
    known/unknown configurations went into that mean and ``max_k`` the largest k they all reached.
    """

    curve: pd.DataFrame
    n_windows: int
    max_k: int

    @property
    def top1(self) -> float:
        return float(self.curve.loc[self.curve["k"] == 1, "accuracy"].iloc[0])

    @property
    def random_top1(self) -> float:
        """Document-level chance accuracy: one over the candidate pool, averaged over windows."""
        return float(self.curve.loc[self.curve["k"] == 1, "random"].iloc[0])


def average_cmc(run: Run) -> AveragedCmc | None:
    """Mean CMC curve over every known/unknown window in ``run``, or ``None`` if it has no CMC.

    Each window is a separate experiment over a different slice of the timeline, and the number
    of candidate authors -- hence the length of the curve -- grows with the known fraction. Only
    the k values *every* window reached are kept, so each averaged point is a mean over the same
    set of windows and the curve cannot quietly change meaning half way along its x axis. The
    mean is unweighted: one window, one vote, regardless of how many documents it held.

    The band is the 95 % Student-t interval for that mean, taken across windows. Read it as a
    description of how much the result moves as the known/unknown split moves -- a wide band means
    the number in the legend depends on which slice of the timeline you look at -- and *not* as a
    sampling error bar: the windows overlap (they are nested prefixes of one corpus), so they are
    not independent draws and the interval is narrower than a true one would be.
    """
    windows = prediction_windows(run)
    if windows:
        cmc = pd.concat([cmc_from_ranks(window) for window in windows], ignore_index=True)
        n_windows = len(windows)
    else:                                    # a run written before per-document ranks were kept
        path = run.directory / "cmc_results.csv"
        if not path.exists():
            return None
        cmc = pd.read_csv(path)
        if "attack" in cmc.columns:
            cmc = cmc[cmc["attack"] == run.attack]
        if cmc.empty:
            return None
        n_windows = cmc.groupby([column for column in ("known_fraction", "window")
                                 if column in cmc.columns]).ngroups
    per_k = cmc.groupby("k")
    complete = per_k.size() == n_windows  # k values present in every window
    curve = add_interval(per_k[["accuracy", "random"]].mean()[complete],
                         per_k["accuracy"].std(ddof=1)[complete], n_windows)

    curve = curve.reset_index().sort_values("k")
    return AveragedCmc(curve=curve, n_windows=n_windows, max_k=int(curve["k"].max()))


# --- risk-coverage: what the attack gets right when it only answers what it is sure of ---------

#: Coverages the risk-coverage curve is evaluated at. Dense enough to show the shape, and it
#: deliberately starts above zero: the precision of the single most confident document is a
#: one-sample estimate that swings between 0 and 1 and would dominate the y axis.
COVERAGE_GRID = np.linspace(0.02, 1.0, 50)


@dataclass
class AveragedRiskCoverage:
    """One run's risk-coverage curve, averaged over its rolling windows.

    ``curve`` has one row per coverage with the mean ``precision``, the mean ``recall``, and the
    ``ci_low``/``ci_high`` band; ``n_windows`` is how many windows went into the mean.
    """

    curve: pd.DataFrame
    n_windows: int

    @property
    def full_coverage_precision(self) -> float:
        """Precision when every document is answered -- i.e. plain top-1 accuracy."""
        return float(self.curve["precision"].iloc[-1])


def selective_precision(confidence: np.ndarray, correct: np.ndarray) -> pd.DataFrame:
    """Precision and recall when only the most confident ``coverage`` of documents is answered.

    A local restatement of :func:`prompt_anonymity.metrics.detection.selective_classification` on
    :data:`COVERAGE_GRID` -- duplicated rather than imported to keep this script out of the
    package's heavy dependencies, and small enough that the duplication is cheaper than the
    coupling. ``recall`` is correct answers retained as a fraction of those made at full coverage,
    the sense in which Narayanan et al. reported ">80% precision at 50% recall".
    """
    order = np.argsort(-confidence, kind="stable")
    cumulative_correct = np.cumsum(correct[order])
    total_correct = int(correct.sum())

    answered = np.clip((COVERAGE_GRID * len(correct)).round().astype(int), 1, len(correct))
    hits = cumulative_correct[answered - 1]
    return pd.DataFrame({
        "coverage": COVERAGE_GRID,
        "precision": hits / answered,
        "recall": hits / total_correct if total_correct else np.nan,
    })


def risk_coverage(run: Run) -> AveragedRiskCoverage | None:
    """Mean risk-coverage curve over every window in ``run``, or ``None`` without predictions.

    This is the figure Narayanan et al. led with, and it catches what top-1 cannot: an attack
    that is usually wrong but *knows when it is right* is a far sharper privacy threat than its
    headline accuracy suggests. Read the left edge -- precision when the attacker answers only its
    most confident tenth -- as the number that matters to someone deciding whether to publish.

    Two definitional choices, both visible in the numbers:

    * **Confidence is the negated ``accept_score``** written to ``predictions_*.csv``, which is the
      attack's cohort-normalised margin (higher = more out-of-set, hence the negation). That is a
      variant of the "gap statistic" the original used, and it is a *different* quantity from the
      max-softmax confidence behind ``precision_at_*pct`` in ``rolling_results.csv``, so the two
      do not agree and are not meant to -- the margin ranks confidence considerably better.
    * **Only in-set documents count** (``author_in_known``), matching the closed-set scoring the
      accuracy metrics use: a document whose author is absent from the known side has no correct
      answer available, so including it would cap precision below 1 for reasons the attack cannot
      control.
    """
    windows = []
    for path in sorted(run.directory.glob(f"predictions_{run.attack}_*.csv")):
        for predictions in split_windows(pd.read_csv(path)):
            in_set = predictions[predictions["author_in_known"].astype(bool)]
            if in_set.empty:
                continue
            windows.append(selective_precision(
                confidence=-in_set["accept_score"].to_numpy(dtype=float),
                correct=(in_set["best_author"] == in_set["true_author"]).to_numpy(),
            ))
    if not windows:
        return None

    per_coverage = pd.concat(windows).groupby("coverage")
    curve = add_interval(per_coverage[["precision", "recall"]].mean(),
                         per_coverage["precision"].std(ddof=1), len(windows), column="precision")
    return AveragedRiskCoverage(curve=curve.reset_index(), n_windows=len(windows))


# --- the two comparison figures ----------------------------------------------

@dataclass
class Series:
    """One line on a comparison figure: a label, the colour slot its entity owns, and the curve.

    ``averaged`` is whichever averaged-curve object the plotter being used expects -- an
    :class:`AveragedCmc` for :func:`plot_cmc_comparison`, an :class:`AveragedRiskCoverage` for
    :func:`plot_risk_coverage_comparison` -- which is what lets one grouping feed both.
    """

    label: str
    slot: int
    averaged: AveragedCmc | AveragedRiskCoverage
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


def plot_cmc_comparison(series: list[Series], title: str, legend_title: str,
                        stem: Path) -> Path:
    """Averaged CMC curves for several runs on one axes, against the shared random baseline.

    All series are truncated to the shortest one's k range so that every line spans the same x
    axis -- runs on the same dataset sweep the same windows, so this normally cuts nothing, and
    when it does the subtitle says how far the axis is trusted. k is on a log axis because the
    informative part of a CMC curve is its first decade: an attacker who has narrowed 20,000
    users to 10 has already won, and what happens at k = 5,000 is noise about the tail.

    Each line carries a shaded 95 % interval for its mean across the windows (see
    :func:`average_cmc` for what that band does and does not claim). Where two bands overlap for
    the whole of their length, the gap between those two lines is not something this experiment
    grid can resolve.

    The random baseline is drawn once, in a neutral grey: it depends only on the candidate-pool
    size, which is a property of the dataset's windows rather than of the defense or feature, so
    every series on a figure shares it. If that ever stops being true (runs with different
    windows landing on the same figure) the mismatch is reported rather than averaged away. It is
    also the only dashed line on the figure -- measured series are solid, identified by colour.
    """
    shared_k = min(item.averaged.max_k for item in series)
    if len({item.averaged.max_k for item in series}) > 1:
        print(f"  note: {stem.name} mixes runs whose windows reach different pool sizes "
              f"({sorted({item.averaged.max_k for item in series})}); truncated to k <= {shared_k}")

    figure, axes = plt.subplots(figsize=(7.4, 4.7))
    figure.patch.set_facecolor(SURFACE)

    baselines = []
    for item, slot in zip(series, resolve_slots([item.slot for item in series])):
        curve = item.averaged.curve
        curve = curve[curve["k"] <= shared_k]
        baselines.append(curve[["k", "random"]].set_index("k")["random"])
        color = series_style(slot)
        axes.fill_between(curve["k"], curve["ci_low"], curve["ci_high"], color=color,
                          alpha=BAND_ALPHA, linewidth=0, zorder=2)
        axes.plot(curve["k"], curve["accuracy"], color=color, linewidth=LINE_WIDTH,
                  solid_capstyle="round", solid_joinstyle="round", zorder=3, label=item.label)
        # An end-dot marks top-1, the headline number, which on a log axis starting at k = 1 sits
        # right on the frame and is easy to lose. The exact value is in the companion .csv: the
        # figure is for comparing curves, and a legend carrying four decimals is read as a table.
        axes.plot([1], [item.averaged.top1], marker="o", markersize=MARKER_SIZE, color=color,
                  markeredgecolor=SURFACE, markeredgewidth=2, zorder=4)

    baseline = pd.concat(baselines, axis=1).mean(axis=1)
    axes.plot(baseline.index, baseline.to_numpy(), color=TEXT_MUTED, linewidth=1.4,
              linestyle=BASELINE_DASH, zorder=2, label="Random guessing")

    n_windows = sorted({item.averaged.n_windows for item in series})
    windows_note = (f"{n_windows[0]} known/unknown windows" if len(n_windows) == 1
                    else f"{min(n_windows)}–{max(n_windows)} known/unknown windows")
    style_axes(axes, "k (candidate authors returned)", "Top-k accuracy (mean over windows)", title,
               f"Mean CMC over {windows_note}, shaded 95% CI; k ≤ {shared_k:,}, "
               f"the smallest candidate pool")
    axes.set_xscale("log")
    axes.set_xlim(1, shared_k)
    axes.set_ylim(0, 1.02)
    add_legend(axes, loc="upper left", title=legend_title)
    figure.tight_layout()

    # The table view: the exact numbers behind every line and band, so nothing is gated behind
    # colour -- and so the overlap between two bands can be checked rather than eyeballed.
    table = pd.DataFrame({"k": baseline.index, "random": baseline.to_numpy()})
    for item in series:
        curve = item.averaged.curve
        curve = curve[curve["k"] <= shared_k]
        for column, suffix in (("accuracy", ""), ("ci_low", " ci_low"), ("ci_high", " ci_high")):
            table[f"{item.label}{suffix}"] = curve[column].to_numpy()
    stem.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(stem.parent / f"{stem.name}.csv", index=False)
    return save_figure(figure, stem)


def plot_risk_coverage_comparison(series: list[Series], title: str, legend_title: str,
                                  stem: Path) -> Path:
    """Precision against coverage for several runs on one axes -- the selective attacker.

    Every point on a line is the same attack under a different willingness to abstain: at
    coverage c it answers only the c most confident documents and this is how often it is right.
    The right edge is plain top-1 accuracy; how far the line climbs to the left of it is the
    quantity a headline accuracy hides. A flat line is an attack whose confidence carries no
    information, and a line that *slopes down to the left* is one whose confidence is
    anti-correlated with being right.

    Coverage is linear here where k is logarithmic on the CMC figures: coverage is already a
    fraction, and the interesting contrast (a tenth of the documents versus all of them) is
    visible without compressing the axis.
    """
    figure, axes = plt.subplots(figsize=(7.4, 4.7))
    figure.patch.set_facecolor(SURFACE)

    for item, slot in zip(series, resolve_slots([item.slot for item in series])):
        curve = item.averaged.curve
        color = series_style(slot)
        axes.fill_between(curve["coverage"], curve["ci_low"], curve["ci_high"], color=color,
                          alpha=BAND_ALPHA, linewidth=0, zorder=2)
        axes.plot(curve["coverage"], curve["precision"], color=color, linewidth=LINE_WIDTH,
                  solid_capstyle="round", solid_joinstyle="round", zorder=3, label=item.label)
        # A dot at full coverage marks where the curve reduces to ordinary top-1 accuracy, which
        # is the number every other figure reports -- the anchor for reading the rest of the line.
        axes.plot([1.0], [item.averaged.full_coverage_precision], marker="o",
                  markersize=MARKER_SIZE, color=color, markeredgecolor=SURFACE,
                  markeredgewidth=2, zorder=4)

    n_windows = sorted({item.averaged.n_windows for item in series})
    windows_note = (f"{n_windows[0]} windows" if len(n_windows) == 1
                    else f"{min(n_windows)}–{max(n_windows)} windows")
    style_axes(axes, "Coverage (most-confident fraction of documents answered)",
               "Precision among answered documents", title,
               f"Mean over {windows_note}, shaded 95% CI; in-set documents, "
               f"ranked by the attack's own margin")
    axes.set_xlim(0, 1.02)
    axes.set_ylim(0, 1.02)
    # Unlike the CMC figures, where every curve rises from the lower left, these can occupy any
    # corner -- a flat line sits wherever its accuracy is -- so the legend has to hunt for a gap.
    add_legend(axes, loc="best", title=legend_title)
    figure.tight_layout()

    table = pd.DataFrame({"coverage": COVERAGE_GRID})
    for item in series:
        curve = item.averaged.curve
        for column, suffix in (("precision", ""), ("recall", " recall"),
                               ("ci_low", " ci_low"), ("ci_high", " ci_high")):
            table[f"{item.label}{suffix}"] = curve[column].to_numpy()
    stem.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(stem.parent / f"{stem.name}.csv", index=False)
    return save_figure(figure, stem)


# --- grouping runs into the two comparison families --------------------------
#
# Both families are "hold one axis fixed, draw a line per value of the other". They are written
# once and parameterised by the plotter, so every curve type (CMC, risk-coverage, and anything
# added later) gets both views without repeating the grouping.

def plot_defense_comparisons(dataset: str, runs: list[Run], averaged: dict[Run, object],
                             plotter, subdirectory: str, output_dir: Path) -> list[Path]:
    """One figure per (feature, attack): every defense measured against that same attack.

    This is the figure that answers "does the defense work?" -- the attack is held fixed so the
    only thing that moves between lines is what the defense did to the text. ``averaged`` supplies
    the curve for each run and ``plotter`` draws it; runs missing from ``averaged`` are skipped,
    which is how a curve type that needs files an older run did not write degrades quietly.
    """
    by_method: dict[tuple[str, str], list[Run]] = defaultdict(list)
    for run in runs:
        if run in averaged:
            by_method[run.method].append(run)

    written = []
    for method, method_runs in sorted(by_method.items(), key=lambda item: METHOD_SLOTS[item[0]]):
        method_runs.sort(key=lambda run: DEFENSE_SLOTS[run.defense])
        series = [Series(run.defense_label, DEFENSE_SLOTS[run.defense], averaged[run], dataset)
                  for run in method_runs]
        feature, attack = method
        written.append(plotter(
            series,
            title=f"{DATASET_LABELS[dataset]}: defenses under "
                  f"{FEATURE_LABELS[feature]} / {ATTACK_LABELS[attack]}",
            legend_title="Defense",
            stem=output_dir / subdirectory / f"{feature}_{attack}",
        ))
    return written


def plot_method_comparisons(dataset: str, runs: list[Run], averaged: dict[Run, object],
                            plotter, subdirectory: str, output_dir: Path) -> list[Path]:
    """One figure per defense: every (feature, attack) measured against that same defense.

    The transpose of :func:`plot_defense_comparisons` -- with the defense held fixed, it says
    which representation and estimator the attacker should reach for.
    """
    by_defense: dict[str, list[Run]] = defaultdict(list)
    for run in runs:
        if run in averaged:
            by_defense[run.defense].append(run)

    written = []
    for defense, defense_runs in sorted(by_defense.items(), key=lambda item: DEFENSE_SLOTS[item[0]]):
        defense_runs.sort(key=lambda run: METHOD_SLOTS[run.method])
        series = [Series(run.method_label, METHOD_SLOTS[run.method], averaged[run], dataset)
                  for run in defense_runs]
        written.append(plotter(
            series,
            title=f"{DATASET_LABELS[dataset]}: attacks against {DEFENSE_LABELS[defense]}",
            legend_title="Feature / attack",
            stem=output_dir / subdirectory / defense,
        ))
    return written


# --- per-author risk: anonymity fails unevenly -------------------------------

#: Shares of the user population the risk curve is evaluated at, most-exposed first. A percentile
#: grid rather than a rank one because windows contain different numbers of users.
EXPOSURE_GRID = np.linspace(0, 100, 101)


@dataclass
class AveragedAuthorRisk:
    """One run's per-user identification rate, sorted from most to least exposed.

    ``curve`` has one row per percentile of the user population with the mean ``accuracy`` at
    that percentile and its band; ``never_identified`` is the share of users not identified once.
    """

    curve: pd.DataFrame
    n_windows: int
    never_identified: float


def per_author_risk(run: Run) -> AveragedAuthorRisk | None:
    """How re-identification risk is distributed across users, not averaged over them.

    A mean accuracy says nothing about who carries it. Sorting each window's users from most to
    least identified and reading off the percentiles turns that into a shape: a curve that falls
    off a cliff means a small group is fully exposed while nearly everyone else is untouched,
    and a gently sloping one means the risk is shared. On WildChat the cliff is the real story --
    roughly 70% of users are never identified once, while a few percent are identified every
    time -- which is exactly the structure a headline accuracy hides and a reviewer asks about.

    Each user's own document-level top-1 accuracy is recomputed per window from
    ``predictions_*.csv`` -- the same quantity ``author_report_*.csv`` reports as
    ``top1_accuracy``, but following the window being plotted, which a report aggregated over the
    whole unknown side cannot. The mean of this curve is the macro accuracy in
    :func:`plot_macro_micro`, read a different way. Runs without per-document ranks fall back to
    the aggregate report.
    """
    exposures = []
    for table in prediction_windows(run):
        # One user's own document-level top-1 accuracy, the same quantity `per_author_ranking`
        # writes as `top1_accuracy` -- recomputed here so it follows the window being plotted.
        exposures.append(table.groupby("true_author")["true_author_rank"]
                         .apply(lambda rank: float((rank <= 1).mean())).to_numpy())
    if not exposures:                        # a run written before per-document ranks were kept
        for path in sorted(run.directory.glob(f"author_report_{run.attack}_*.csv")):
            report = pd.read_csv(path)
            if "attack" in report.columns:
                report = report[report["attack"] == run.attack]
            if report.empty or "top1_accuracy" not in report.columns:
                continue
            for window in split_windows(report):
                exposures.append(window["top1_accuracy"].to_numpy(dtype=float))

    windows = []
    for values in exposures:
        exposure = np.sort(values)[::-1]
        if not len(exposure):
            continue
        # Midpoints of each user's share of the population, so the curve is anchored at the
        # centre of a user's slice rather than its edge and does not depend on the user count.
        position = (np.arange(len(exposure)) + 0.5) / len(exposure) * 100
        windows.append(pd.DataFrame({
            "percentile": EXPOSURE_GRID,
            "accuracy": np.interp(EXPOSURE_GRID, position, exposure),
            "never": float((exposure == 0).mean()),
        }))
    if not windows:
        return None

    per_percentile = pd.concat(windows).groupby("percentile")
    curve = add_interval(per_percentile[["accuracy"]].mean(),
                         per_percentile["accuracy"].std(ddof=1), len(windows))
    never = float(np.mean([window["never"].iloc[0] for window in windows]))
    return AveragedAuthorRisk(curve=curve.reset_index(), n_windows=len(windows),
                              never_identified=never)


def plot_author_risk_comparison(series: list[Series], title: str, legend_title: str,
                                stem: Path) -> Path:
    """Per-user identification rate against the share of users, most exposed first.

    Read it as "the most exposed x% of users are identified at least this often". The area under
    each curve is that run's macro accuracy, so two runs with the same macro number can still
    have very different shapes -- and the shape is what a person deciding whether they are at
    risk actually wants.
    """
    figure, axes = plt.subplots(figsize=(7.4, 4.7))
    figure.patch.set_facecolor(SURFACE)

    for item, slot in zip(series, resolve_slots([item.slot for item in series])):
        curve = item.averaged.curve
        color = series_style(slot)
        axes.fill_between(curve["percentile"], curve["ci_low"], curve["ci_high"], color=color,
                          alpha=BAND_ALPHA, linewidth=0, zorder=2)
        axes.plot(curve["percentile"], curve["accuracy"], color=color, linewidth=LINE_WIDTH,
                  solid_capstyle="round", zorder=3,
                  label=f"{item.label}  ·  {item.averaged.never_identified:.0%} never identified")

    n_windows = sorted({item.averaged.n_windows for item in series})
    windows_note = (f"{n_windows[0]} windows" if len(n_windows) == 1
                    else f"{min(n_windows)}–{max(n_windows)} windows")
    style_axes(axes, "Share of users, most exposed first (%)",
               "That user's own top-1 accuracy", title,
               f"Mean over {windows_note}, shaded 95% CI; a cliff means the risk sits with a few")
    axes.set_xlim(0, 100)
    axes.set_ylim(0, 1.02)
    add_legend(axes, loc="best", title=legend_title)
    figure.tight_layout()

    table = pd.DataFrame({"percentile": EXPOSURE_GRID})
    for item in series:
        table[item.label] = item.averaged.curve["accuracy"].to_numpy()
    stem.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(stem.parent / f"{stem.name}.csv", index=False)
    return save_figure(figure, stem)


# --- temporal decay: does the attack go stale? -------------------------------

#: Documents a (known side, week) bin needs before its accuracy is plotted. A week holding three
#: documents produces an accuracy of 0, 1/3, 2/3 or 1, which is noise drawn at full contrast.
MIN_DOCUMENTS_PER_WEEK = 5


#: Which known side the temporal figure draws. One attacker, not an average over three: the
#: three known sides cut the timeline at different dates, so averaging them truncates to the
#: weeks the *shortest* reaches -- on swe-chat 4 weeks against the 25% side's own 8. Taking the
#: earliest side alone keeps the longest run of future, which is the axis this figure exists for.
TEMPORAL_KNOWN_FRACTION = 0.25


@dataclass
class TemporalDecay:
    """One run's top-1 accuracy per week, split by whether the week's users are new to the attack.

    ``curve`` is long-form -- ``week``, ``cohort``, ``accuracy``, ``n_documents``, ``n_users`` --
    with ``cohort`` in :data:`COHORTS`. ``counts`` is the same population *unfiltered*, so the
    context panel can show a thin week that the accuracy line drops.
    """

    curve: pd.DataFrame
    counts: pd.DataFrame
    known_fraction: float


#: The two kinds of user a week can contain, in the order they are drawn and stacked. "New" is
#: relative to the *unknown stream only*: every user here is enrolled on the known side (that is
#: what makes them attackable at all), so this asks whether the attacker has already had to
#: attribute this person once before, not whether it has ever seen them.
COHORTS = ("new", "returning")

COHORT_LABELS = {"new": "First seen this week", "returning": "Seen in an earlier week"}


def document_end_times(dataset: str) -> pd.Series | None:
    """``doc_id -> ended_at`` for one dataset, read from its split parquet.

    When a document ended is a property of the corpus, not of any attack, so it is joined back
    here rather than written into every run's ``predictions_*.csv``. That keeps one copy of the
    fact, and -- the reason that matters in practice -- it makes the temporal figure available for
    **every run already on disk**, including ones far too expensive to re-run for a column.

    Only two columns are read, so the 411 MB WildChat parquet costs a projection rather than a
    load. Memoised because a dataset's runs all need the same mapping.
    """
    if dataset in _END_TIMES:
        return _END_TIMES[dataset]
    path = DATA_DIR / f"{SPLIT_NAMES[dataset]}.parquet"
    times = None
    if path.exists():
        frame = pd.read_parquet(path, columns=["doc_id", "ended_at"])
        times = pd.Series(pd.to_datetime(frame["ended_at"], errors="coerce", utc=True,
                                         format="mixed").to_numpy(),
                          index=frame["doc_id"].to_numpy())
    _END_TIMES[dataset] = times
    return times


#: Memo for :func:`document_end_times`, one entry per dataset (``None`` when its parquet is absent).
_END_TIMES: dict[str, pd.Series | None] = {}


def temporal_accuracy(run: Run, known_fraction: float = TEMPORAL_KNOWN_FRACTION,
                      min_documents: int = MIN_DOCUMENTS_PER_WEEK) -> TemporalDecay | None:
    """Top-1 accuracy per week, split by whether the week's users are new to the unknown stream.

    Every other figure here holds time fixed and varies the attack. This one does the opposite,
    and it separates the two things the old window sweep confounded: a longer window contains
    *more users* **and** reaches *further into the future*, so a falling accuracy could be either.
    Binning by elapsed weeks holds the attacker fixed and lets only staleness vary.

    **The cohort split removes the remaining confound, which is composition.** Week 12 is not a
    random sample of week 0's people: users who write steadily are still there, one-off users are
    not, and the attacker has by then already attributed the steady ones several times. A single
    line mixes "the attack is going stale" with "the surviving population is different", and those
    have opposite privacy readings. So each week is cut in two:

    * ``new`` -- the user's **first** appearance in the unknown stream. Note "new" is relative to
      the *testing* side only: every user counted here is enrolled on the known side, or the
      document would have no correct answer and be dropped. This is the honest measure of decay,
      because a new user's documents are the same kind of first-contact problem in week 12 as in
      week 0.
    * ``returning`` -- the user appeared in some earlier week. The attacker has met them before,
      and their staying power is itself a signal.

    The label is per *user per week*, not per document: several documents by one person in one
    week share whichever cohort that person is in that week, so a prolific week cannot put the
    same user on both lines.

    **The weeks are disjoint buckets, not running totals.** Week 3 is the documents that ended in
    the third week after the cut and no others. A cumulative reading would drag every later point
    toward the average and hide exactly the decay this figure is asked to show.

    One known side (``known_fraction``), so there is nothing to average and no interval to draw:
    every point is the whole population of its week-and-cohort. ``counts`` carries the populations
    instead, unfiltered, so a week whose accuracy was dropped for thinness still shows up as the
    handful of people it was.

    A hit is ``best_author == true_author`` rather than ``true_author_rank <= 1`` -- the same
    statement, but those columns are in *every* predictions file this project has written, so this
    needs no re-run and covers every output layout.

    Returns ``None`` when the run has no predictions for that known side, or when the dataset's
    parquet is not on disk to supply the timestamps.
    """
    end_times = document_end_times(run.dataset)
    if end_times is None:
        return None
    files = sorted(run.directory.glob(f"predictions_{run.attack}_*.csv"))
    if not files:
        return None
    # The earliest layout stamped nothing on the rows and put every axis in the filename, so the
    # known fraction is recovered from `..._known<pct>[_window<pct>].csv` when the column is
    # missing. It is the one axis this function has to select on.
    frames = []
    for path in files:
        frame = pd.read_csv(path)
        if "known_fraction" not in frame.columns:
            match = re.search(r"_known(\d+)", path.stem)
            if match is None:
                return None
            frame["known_fraction"] = int(match.group(1)) / 100
        frames.append(frame)
    table = pd.concat(frames, ignore_index=True)
    if "attack" in table.columns:
        table = table[table["attack"] == run.attack]
    table = table[np.isclose(table["known_fraction"], known_fraction)]
    table = table[table["author_in_known"].astype(bool)]
    # The pre-change layout wrote one file per window, and its windows are nested prefixes of one
    # another -- so a document appears in up to three of them. Deduplicating collapses those back
    # to the union, which is what a single full-unknown-side file already is. Without it the early
    # weeks would be counted once per window that covers them, and a user's "first week" could be
    # read off a duplicate.
    table = table.drop_duplicates(subset="doc_id")
    stamps = table["doc_id"].map(end_times)
    table, stamps = table[stamps.notna()], stamps[stamps.notna()]
    if table.empty:
        return None

    elapsed = (stamps - stamps.min()).dt.total_seconds() / (7 * 24 * 3600)
    weekly = pd.DataFrame({
        "week": elapsed.to_numpy().astype(int),
        "hit": (table["best_author"] == table["true_author"]).to_numpy(),
        "author": table["true_author"].to_numpy(),
    })
    # A user is "new" in the week they first appear on the unknown side and "returning" in every
    # week after it -- computed per user, then broadcast to that user's documents.
    first_seen = weekly.groupby("author")["week"].transform("min")
    weekly["cohort"] = np.where(weekly["week"] == first_seen, "new", "returning")

    grouped = weekly.groupby(["cohort", "week"])
    counts = pd.DataFrame({"n_documents": grouped["hit"].size(),
                           "n_users": grouped["author"].nunique()}).reset_index()
    curve = pd.DataFrame({"accuracy": grouped["hit"].mean(),
                          "n_documents": grouped["hit"].size(),
                          "n_users": grouped["author"].nunique()}).reset_index()
    curve = curve[curve["n_documents"] >= min_documents]
    if curve.empty:
        return None
    return TemporalDecay(curve=curve.sort_values(["cohort", "week"]),
                         counts=counts.sort_values(["cohort", "week"]),
                         known_fraction=known_fraction)


def plot_temporal_decay(dataset: str, runs: list[Run], decay: dict[Run, TemporalDecay],
                        output_dir: Path) -> list[Path]:
    """Top-1 accuracy against elapsed weeks, one line per attack+defense combination.

    Three rows against one column per defense. The **rows are the cohorts** -- users meeting the
    attack for the first time, then users it has already attributed in an earlier week -- because
    putting both on one axes would double the lines and blow past the eight-hue palette, and
    separating them by dash would say "not a measurement" in a file where that is what a dash
    means. Split into rows, colour keeps meaning the method and the two readings sit one above the
    other on a shared scale, which is the comparison worth making.

    The bottom row is **the population each week was measured over, stacked by cohort**. It is the
    same population for every line in the figure -- a defense rewrites text but changes neither who
    wrote it nor when -- so it is drawn once per column in recessive grey, never as a second y
    axis. It is what makes the accuracy rows legible: it shows the returning share climbing as the
    weeks pass, which is precisely the composition shift the rows above are controlling for.
    """
    present = [defense for defense in DEFENSES if any(run.defense == defense for run in runs)]
    if not present:
        return []

    # Truncate every line to the last week they all reach. Runs still in the pre-change layout
    # stop early for a reason that has nothing to do with their attack: that layout's widest
    # window is half the corpus *by position*, and documents are denser early, so it covers about
    # four weeks of an eight-week future. Drawn untruncated, a defense whose runs happen to be old
    # would look as though its curve simply ended.
    limit = min(int(decay[run].curve["week"].max()) for run in runs)
    short = sorted(run.directory.name for run in runs
                   if int(decay[run].curve["week"].max()) == limit)
    if any(int(decay[run].curve["week"].max()) > limit for run in runs):
        print(f"  temporal: truncated to week {limit}; {', '.join(short)} do not reach further "
              f"(pre-change output layout) -- re-run those to extend the axis")

    figure, axes_grid = plt.subplots(
        3, len(present), figsize=(5.8 * len(present), 8.6), squeeze=False,
        sharex="col", sharey="row", gridspec_kw={"height_ratios": [3, 3, 1.6]})
    figure.patch.set_facecolor(SURFACE)

    handles: dict[str, object] = {}
    for column, defense in enumerate(present):
        panel = sorted((run for run in runs if run.defense == defense),
                       key=lambda run: METHOD_SLOTS[run.method])
        slots = resolve_slots([METHOD_SLOTS[run.method] for run in panel])

        for row, cohort in enumerate(COHORTS):
            axes = axes_grid[row][column]
            for run, slot in zip(panel, slots):
                curve = decay[run].curve
                curve = curve[(curve["cohort"] == cohort) & (curve["week"] <= limit)]
                if curve.empty:
                    continue
                color = series_style(slot)
                # The 2 px surface ring that keeps overlapping markers separable turns into a
                # dashed-looking line once the points are dense -- and here a dash means "not a
                # measurement". Past a dozen weeks the markers come off and the line speaks.
                marks = dict(marker="o", markersize=MARKER_SIZE, markeredgecolor=SURFACE,
                             markeredgewidth=2) if len(curve) <= 12 else {}
                axes.plot(curve["week"], curve["accuracy"], color=color, linewidth=LINE_WIDTH,
                          solid_capstyle="round", zorder=3, **marks)
                handles.setdefault(run.method_label,
                                   Line2D([], [], color=color, linewidth=LINE_WIDTH))
            label = f"Top-1 accuracy\n({COHORT_LABELS[cohort].lower()})" if column == 0 else ""
            style_axes(axes, "", label, DEFENSE_LABELS[defense] if row == 0 else "", "")
            axes.set_ylim(bottom=0)

        # Identical for every line above, so it is read off whichever run reaches furthest.
        counts = max((decay[run].counts for run in panel), key=len)
        counts = counts[counts["week"] <= limit]
        bars = axes_grid[2][column]
        bottom = np.zeros(limit + 1)
        for shade, cohort in zip((0.22, 0.5), COHORTS):
            per_week = (counts[counts["cohort"] == cohort].set_index("week")["n_users"]
                        .reindex(range(limit + 1), fill_value=0).to_numpy())
            bars.bar(range(limit + 1), per_week, bottom=bottom, width=0.7, color=TEXT_MUTED,
                     alpha=shade, linewidth=0.8, edgecolor=SURFACE, zorder=2,
                     label=COHORT_LABELS[cohort] if column == 0 else None)
            bottom = bottom + per_week
        style_axes(bars, "Weeks after the attacker's data ends",
                   "Users" if column == 0 else "", "", "")
        bars.set_ylim(bottom=0)
        # The bins are whole weeks; a tick at 1.5 weeks labels a point that cannot exist.
        bars.xaxis.set_major_locator(MaxNLocator(integer=True))
        if column == 0:
            add_legend(bars, loc="upper right", ncol=2)

    known = next(iter(decay.values())).known_fraction
    figure.suptitle(f"{DATASET_LABELS[dataset]}: does re-identification go stale?  "
                    f"(attacker knows the first {known:.0%})",
                    color=TEXT_PRIMARY, fontsize=12, x=0.01, ha="left")
    # tight_layout does not know about figure-level legends, so the band is reserved first and the
    # legend anchored inside it; anchoring below the figure instead lands it on the x label.
    columns = min(len(handles), 3)
    rows = -(-len(handles) // columns)
    figure.tight_layout()
    figure.subplots_adjust(bottom=0.09 + 0.033 * rows)
    legend = figure.legend(list(handles.values()), list(handles), title="Feature / attack",
                           loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=columns,
                           frameon=False, fontsize=8.5, labelcolor=TEXT_PRIMARY)
    legend.get_title().set_color(TEXT_SECONDARY)
    legend.get_title().set_fontsize(8.5)

    stem = output_dir / "temporal" / "top1_by_week"
    stem.parent.mkdir(parents=True, exist_ok=True)
    pd.concat([decay[run].curve.assign(defense=run.defense, method=run.method_label,
                                       known_fraction=decay[run].known_fraction)
               for run in runs], ignore_index=True).to_csv(
        stem.parent / f"{stem.name}.csv", index=False)
    return [save_figure(figure, stem)]

# --- macro vs micro vs identity: three ways to count the same result ---------

#: The three ways this project counts a re-identification, in the order they are drawn, each with
#: where it comes from and what it answers.
COUNTING_MODES = (
    ("micro", "Micro (per document)", "what share of traffic can be attributed"),
    ("macro", "Macro (per user)", "how the attack does against a typical user"),
    ("identity", "Identity (any document)", "was this user re-identified at all"),
)


def counting_modes(run: Run, averaged: AveragedCmc | None) -> pd.DataFrame | None:
    """The same run's accuracy counted three ways, with the spread across windows.

    Micro weights every document equally, so prolific users carry it; macro weights every user
    equally; identity asks only whether *any one* of a user's documents landed on them, which is
    the attacker's best case. They can differ by a factor of several on the same run, and which
    one a paper leads with is a claim about what "anonymity failed" means.
    """
    path = run.directory / "rolling_results.csv"
    if not path.exists() or averaged is None:
        return None
    rolling = pd.read_csv(path)
    if "attack" in rolling.columns:
        rolling = rolling[rolling["attack"] == run.attack]
    if rolling.empty or not {"macro_conv_acc1", "id_acc1"}.issubset(rolling.columns):
        return None

    # Micro is read from the CMC rather than rolling_results: `closed_set_top1` is scoped to the
    # in-set documents, while the CMC at k=1 is the same document-level number every other figure
    # on this axis reports.
    cmc = pd.read_csv(run.directory / "cmc_results.csv")
    if "attack" in cmc.columns:
        cmc = cmc[cmc["attack"] == run.attack]
    micro = cmc.loc[cmc["k"] == 1, "accuracy"]

    rows = []
    for mode, label, _ in COUNTING_MODES:
        values = {"micro": micro,
                  "macro": rolling["macro_conv_acc1"],
                  "identity": rolling["id_acc1"]}[mode].to_numpy(dtype=float)
        half_width = t_multiplier(len(values)) * np.std(values, ddof=1) / np.sqrt(len(values))
        rows.append({"mode": mode, "label": label, "accuracy": float(values.mean()),
                     "half_width": float(np.nan_to_num(half_width)), "n_windows": len(values)})
    return pd.DataFrame(rows)


def plot_macro_micro(dataset: str, runs: list[Run], modes: dict[Run, pd.DataFrame],
                     output_dir: Path) -> list[Path]:
    """One horizontal bar group per run: the same result counted three ways.

    Bars run horizontally because the category labels are run names -- a feature, an attack and
    sometimes a defense -- which do not fit under a vertical bar without rotating the text.
    Whiskers are the 95% interval across windows, the same one the curve figures shade.
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
        axes.barh(positions + offset, [row["accuracy"] for row in values], height=height,
                  color=series_style(index), zorder=3, label=label,
                  xerr=[row["half_width"] for row in values],
                  error_kw={"ecolor": TEXT_MUTED, "elinewidth": 1.0, "capsize": 2})

    axes.set_yticks(positions)
    axes.set_yticklabels([run.method_label if run.defense == NO_DEFENSE
                          else f"{run.method_label}\n{run.defense_label}" for run in ordered])
    axes.invert_yaxis()  # first run at the top, reading order
    style_axes(axes, "Top-1 accuracy", "", f"{DATASET_LABELS[dataset]}: one result, three ways "
               f"of counting it",
               "Micro weights documents, macro weights users, identity asks whether any one "
               "document hit")
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
    """One run's top-1 accuracy as a function of how many users the attack chooses between.

    ``curve`` has one row per pool size with the mean ``accuracy`` over windows, the ``random``
    baseline, and the ``ci_low``/``ci_high`` band. ``windows`` holds the one measured point per
    window -- its real pool size and the accuracy actually observed there.
    """

    curve: pd.DataFrame
    windows: pd.DataFrame
    n_windows: int


def subpool_accuracy(rank_probability: np.ndarray, pool_sizes: np.ndarray) -> np.ndarray:
    """Expected top-1 accuracy if the attack had to choose between fewer candidates.

    A document whose true author the attack ranked *r*-th out of *N* is answered correctly in a
    sub-pool of *n* candidates exactly when none of the r-1 authors it preferred were drawn into
    that sub-pool. For a uniformly random sub-pool containing the true author that probability is
    ``C(N-r, n-1) / C(N-1, n-1)``, so averaging it over the observed rank distribution gives the
    accuracy the same attack would have reached against a smaller gallery -- the standard
    gallery-size extrapolation, and the same reasoning
    :func:`prompt_anonymity.evaluation.pool_size_sweep` implements by sampling.

    It answers a narrower question than a real experiment on a smaller corpus: it shrinks *who
    the attacker must choose between* while holding the text, the features and the timeline
    fixed. A genuinely smaller dataset would differ in all of those too.

    Parameters
    ----------
    rank_probability : array of shape (n_candidates,)
        ``rank_probability[r - 1]`` is the fraction of documents whose true author was ranked
        *r*-th. It need not sum to 1: the shortfall is the documents whose author was not a
        candidate at all, which are wrong at every pool size.
    pool_sizes : array of int
        Sub-pool sizes to evaluate, each in ``[1, n_candidates]``.

    Notes
    -----
    Binomial coefficients are taken in log space from a table of ``log(i!)``, which keeps the
    whole thing exact and vectorised at N in the tens of thousands where the coefficients
    themselves overflow long before they cancel.
    """
    n_candidates = len(rank_probability)
    log_factorial = np.concatenate(([0.0], np.cumsum(np.log(np.arange(1, n_candidates + 1)))))
    ranks = np.arange(1, n_candidates + 1)

    accuracy = np.empty(len(pool_sizes), dtype=float)
    for index, pool_size in enumerate(pool_sizes):
        # Ranks worse than this cannot survive: there is no room for r-1 preferred authors to
        # all be excluded from a pool of n.
        spare = n_candidates - ranks - pool_size + 1
        reachable = spare >= 0
        log_probability = (log_factorial[n_candidates - ranks[reachable]]
                           - log_factorial[spare[reachable]]
                           - log_factorial[n_candidates - 1]
                           + log_factorial[n_candidates - pool_size])
        accuracy[index] = float(rank_probability[reachable] @ np.exp(log_probability))
    return accuracy


def load_scaling(run: Run) -> ScalingCurve | None:
    """Top-1 accuracy against the number of candidate users, interpolated down from each window.

    The measured runs give only three pool sizes per dataset -- 81/106/124 on SWE-chat,
    7,456/13,694/19,711 on WildChat -- which on a shared log axis leaves the two corpora as two
    isolated clumps with two orders of magnitude of nothing between them. Every window's *rank
    distribution* is already in ``cmc_results.csv``, though, and that is enough to say what the
    same attack would have scored against any smaller gallery (:func:`subpool_accuracy`). So each
    window contributes a curve from 2 candidates up to its own pool, and the two datasets overlap
    instead of merely coexisting.

    Curves are averaged over windows and truncated to the pool sizes every window reached, the
    same convention as :func:`average_cmc`. The measured point of each window is kept separately
    and drawn as a marker, so what was run and what was interpolated stay distinguishable.

    Returns ``None`` for any attack outside :data:`POOL_INTERPOLABLE_ATTACKS`, where the estimator
    would not be exact.
    """
    if run.attack not in POOL_INTERPOLABLE_ATTACKS:
        return None
    derived = prediction_windows(run)
    if derived:
        groups = [cmc_from_ranks(window) for window in derived]
    else:                                    # a run written before per-document ranks were kept
        path = run.directory / "cmc_results.csv"
        if not path.exists():
            return None
        cmc = pd.read_csv(path)
        if "attack" in cmc.columns:
            cmc = cmc[cmc["attack"] == run.attack]
        if cmc.empty:
            return None
        keys = [column for column in ("known_fraction", "window") if column in cmc.columns]
        groups = [group for _, group in cmc.groupby(keys)]
    pool_sizes = pool_lattice(min(int(group["k"].max()) for group in groups))

    curves, measured = [], []
    for group in groups:
        group = group.sort_values("k")
        accuracy = group["accuracy"].to_numpy(dtype=float)
        # The CMC is the cumulative rank distribution, so differencing it recovers the mass at
        # each rank -- accuracy at k=1 is P(rank 1), and each later step is P(rank = k).
        rank_probability = np.diff(accuracy, prepend=0.0)
        curves.append(pd.DataFrame({
            "n_candidates": pool_sizes,
            "accuracy": subpool_accuracy(rank_probability, pool_sizes),
        }))
        measured.append({"n_candidates": int(group["k"].max()), "accuracy": float(accuracy[0])})

    per_pool = pd.concat(curves).groupby("n_candidates")
    curve = add_interval(per_pool[["accuracy"]].mean(), per_pool["accuracy"].std(ddof=1),
                         len(curves))
    curve["random"] = 1.0 / curve.index.to_numpy(dtype=float)
    return ScalingCurve(curve=curve.reset_index(), windows=pd.DataFrame(measured),
                        n_windows=len(curves))


def plot_scaling(series: list[Series], title: str, legend_title: str, stem: Path) -> Path:
    """Top-1 accuracy against candidate-pool size, one line per run, log x.

    The claim is not the height of any line but the *widening gap*: chance falls away as ``1/n``
    while the attack decays far more slowly, so anonymity does not recover by adding users. Open
    markers sit at each window's real pool size -- the pools actually run -- and the line through
    the smaller sizes is the sub-pool interpolation, which is what lets a corpus of dozens and a
    corpus of tens of thousands be read on one axis.
    """
    figure, axes = plt.subplots(figsize=(7.4, 4.7))
    figure.patch.set_facecolor(SURFACE)

    baselines, colors = [], {}
    slots = resolve_slots([item.slot for item in series], [item.dash for item in series])
    for item, slot in zip(series, slots):
        curve, windows = item.averaged.curve, item.averaged.windows
        color = series_style(slot)
        colors.setdefault(item.label, color)
        baselines.append(curve[["n_candidates", "random"]].set_index("n_candidates")["random"])
        axes.fill_between(curve["n_candidates"], curve["ci_low"], curve["ci_high"], color=color,
                          alpha=BAND_ALPHA, linewidth=0, zorder=2)
        axes.plot(curve["n_candidates"], curve["accuracy"], color=color, linewidth=LINE_WIDTH,
                  linestyle=(0, item.dash) if item.dash else "-", solid_capstyle="round", zorder=3)
        # Hollow markers for the windows that were actually run, so an interpolated stretch of
        # line is never mistaken for a measurement.
        axes.plot(windows["n_candidates"], windows["accuracy"], linestyle="none", marker="o",
                  markersize=MARKER_SIZE, markerfacecolor=SURFACE, markeredgecolor=color,
                  markeredgewidth=1.6, zorder=4)

    baseline = pd.concat(baselines, axis=1).mean(axis=1).sort_index()
    axes.plot(baseline.index, baseline.to_numpy(), color=TEXT_MUTED, linewidth=1.4,
              linestyle=BASELINE_DASH, zorder=2)

    style_axes(axes, "Candidate users the attack ranks over", "Top-1 accuracy", title,
               "Line interpolates each window down to smaller galleries; rings are the pools "
               "actually run")
    axes.set_xscale("log")
    axes.set_ylim(0, 1.02)

    # Two legends, because two channels carry two different things. Colour names the method and
    # is shared by both corpora; dash names the corpus and is drawn in ink rather than in any
    # series colour, so neither legend can be misread as belonging to one line.
    method_legend = add_legend(
        axes, loc="upper right", title=legend_title,
        handles=[Line2D([], [], color=color, linewidth=LINE_WIDTH, label=label)
                 for label, color in colors.items()])
    axes.add_artist(method_legend)
    datasets = [name for name in DATASETS if any(item.dataset == name for item in series)]
    add_legend(axes, loc="lower left", title="Line",
               # Long enough that a dash-dot period fits in the sample; on the default handle the
               # pattern is clipped and every entry reads as a solid line.
               handlelength=4.0,
               handles=[Line2D([], [], color=TEXT_SECONDARY, linewidth=LINE_WIDTH,
                               linestyle=(0, DATASET_DASHES[name]) if DATASET_DASHES[name] else "-",
                               label=DATASET_LABELS[name]) for name in datasets]
               + [Line2D([], [], color=TEXT_MUTED, linewidth=1.4, linestyle=BASELINE_DASH,
                         label="Random guessing")])
    figure.tight_layout()

    table = pd.DataFrame({"n_candidates": baseline.index, "random": baseline.to_numpy()})
    for item in series:
        curve = item.averaged.curve.set_index("n_candidates")
        for column, suffix in (("accuracy", ""), ("ci_low", " ci_low"), ("ci_high", " ci_high")):
            table[f"{item.column}{suffix}"] = curve[column].reindex(baseline.index).to_numpy()
    stem.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(stem.parent / f"{stem.name}.csv", index=False)
    return save_figure(figure, stem)


def plot_scaling_across_datasets(runs: list[Run], scaling: dict[Run, ScalingCurve],
                                 output_dir: Path) -> list[Path]:
    """The one figure that is not per-dataset: every corpus's pools on a single log axis.

    Within a dataset the pool spans only a factor of a few; across SWE-chat and WildChat it spans
    more than two orders of magnitude, which is the range needed to say anything about scale at
    all. Lines are per run and never join two datasets -- the axis is shared, the experiments are
    not.

    Only undefended runs are drawn. This figure answers "how far does the threat reach?", and a
    defended run belongs to the separate question of whether a countermeasure helps.

    Colour is the feature+attack and dash is the corpus, so one method can be followed across
    both -- which is the whole point, since the interesting comparison is the same attack at a
    matched pool size rather than either corpus on its own.
    """
    ordered = sorted(
        (run for run in runs if run in scaling and run.defense == NO_DEFENSE),
        key=lambda run: (METHOD_SLOTS[run.method], DATASETS.index(run.dataset)),
    )
    # Because every run's curve now reaches down to two candidates, the corpora overlap on the
    # shared range instead of sitting as two clumps with two empty decades between them.
    if len(ordered) < 2:
        return []
    # Colour is the method alone -- so the same feature+attack is one hue on both corpora --
    # and the dataset rides on the dash instead.
    series = [Series(run.method_label, METHOD_SLOTS[run.method], scaling[run], run.dataset)
              for run in ordered]
    return [plot_scaling(series, title="Re-identification against candidate-pool size",
                         legend_title="Feature / attack",
                         stem=output_dir / "cross_dataset" / "scaling")]


# --- per-run figures ---------------------------------------------------------
#
# These are the figures `run_experiment_v2.py` and `run_experiment.py` used to draw at the end of
# a run. They now live here, rebuilt from the same CSVs, so that the runners only produce numbers
# and every figure in the project comes from one place.

def plot_windowed_cmc(cmc: pd.DataFrame, title: str, stem: Path) -> Path:
    """One run's CMC curves, one line per known/unknown window -- the un-averaged detail view.

    The *shape* is the interesting part: a curve that shoots up and flattens means the attack is
    confidently right about a subset, while one that climbs steadily means it is merely narrowing
    a large pool, and the two can share a top-1.
    """
    keys = [column for column in ("known_fraction", "window") if column in cmc.columns]
    windows = sorted(cmc.groupby(keys), key=lambda item: item[0])
    figure, axes = plt.subplots(figsize=(7.4, 4.7))
    figure.patch.set_facecolor(SURFACE)
    for slot, (key, group) in enumerate(windows):
        # groupby returns a tuple per key, of length 1 when only `known_fraction` is present --
        # which is every run since the window axis moved to `--window`.
        key = key if isinstance(key, tuple) else (key,)
        fraction, window = key[0], (key[1] if len(key) > 1 else None)
        group = group.sort_values("k")
        color = series_style(slot)
        axes.plot(group["k"], group["accuracy"], color=color, linewidth=LINE_WIDTH,
                  solid_capstyle="round", zorder=3,
                  label=(f"known {fraction:.0%} → next {window:.0%}" if window is not None
                         else f"known {fraction:.0%} → rest"))
        axes.plot(group["k"], group["random"], color=TEXT_MUTED, linewidth=1.0,
                  linestyle=BASELINE_DASH, alpha=0.6, zorder=2)
    style_axes(axes, "k (candidate authors returned)", "Top-k accuracy", title,
               "One line per window; grey dashes are that window's random baseline")
    axes.set_xscale("log")
    axes.set_ylim(0, 1.02)
    add_legend(axes, loc="upper left", ncols=2)
    figure.tight_layout()
    return save_figure(figure, stem)


def plot_window_sweep(sweep: pd.DataFrame, top_k: int, title: str, stem: Path) -> Path:
    """Identity accuracy vs. the number of target users, one line per known fraction.

    Each point is a real experiment on a longer slice of the timeline, which brings in more
    target users because more people show up over a longer period. Solid is the attack, dashed
    grey the random baseline for that same pool -- both move with the window, so only the gap
    between them means anything. Widening the window also pushes the unknown side further into
    the future, so a falling line mixes pool growth with temporal drift.
    """
    figure, axes = plt.subplots(figsize=(7.0, 4.4))
    figure.patch.set_facecolor(SURFACE)
    # Without a window axis each known fraction contributes a single row, so grouping on it would
    # draw three one-point series. The rows then form one series in their own right -- accuracy
    # against a pool that grows because the known side does.
    groups = (list(sweep.groupby("known_fraction")) if "window" in sweep.columns
              else [(None, sweep)])
    for slot, (fraction, group) in enumerate(groups):
        group = group.sort_values("n_identities")
        color = series_style(slot)
        axes.plot(group["n_identities"], group["id_acc"], color=color, linewidth=LINE_WIDTH,
                  marker="o", markersize=MARKER_SIZE,
                  markeredgecolor=SURFACE, markeredgewidth=2, zorder=3,
                  label=f"known {fraction:.0%}" if fraction is not None else "measured")
        axes.plot(group["n_identities"], group["random_id"], color=TEXT_MUTED, linewidth=1.2,
                  linestyle=BASELINE_DASH, zorder=2)
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


def plot_pool_size_sweep(sweep: pd.DataFrame, top_k: int, title: str, stem: Path) -> Path:
    """Identity accuracy vs. candidate-pool size, over random sub-pools (``run_experiment.py``).

    ``sweep`` is the table from :func:`prompt_anonymity.evaluation.pool_size_sweep`: many random
    sub-pools per size, so each size gets a spread rather than a point. The attack's spread and
    the random baseline's are drawn as a median line with an inter-quartile band.
    """
    figure, axes = plt.subplots(figsize=(7.0, 4.4))
    figure.patch.set_facecolor(SURFACE)
    grouped = sweep.groupby("n")
    for column, color, label in (("id_acc", CATEGORICAL[0], "Attack"),
                                 ("random_id", TEXT_MUTED, "Random guessing")):
        quartiles = grouped[column].quantile([0.25, 0.5, 0.75]).unstack()
        axes.fill_between(quartiles.index, quartiles[0.25], quartiles[0.75], color=color,
                          alpha=BAND_ALPHA, linewidth=0, zorder=2)
        axes.plot(quartiles.index, quartiles[0.5], color=color, linewidth=LINE_WIDTH,
                  linestyle="-" if column == "id_acc" else BASELINE_DASH, zorder=3, label=label)
    style_axes(axes, "Candidate users in the pool", f"Top-{top_k} identification accuracy", title,
               "Median over the random sub-pools, with the inter-quartile band")
    axes.set_ylim(bottom=0)
    add_legend(axes, loc="upper right")
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

    Which figures appear depends on which runner produced the directory: a rolling-window run
    (``run_experiment_v2.py``) has ``cmc_results.csv`` and ``rolling_results.csv``, while the
    older fixed-split runner leaves ``headline_results.csv`` and ``sweep_results.csv``.
    """
    stem_dir = output_dir / "per_run" / run.directory.name
    written, scope = [], f"{run.defense_label}, {run.method_label}"

    cmc_path = run.directory / "cmc_results.csv"
    if cmc_path.exists():
        cmc = pd.read_csv(cmc_path)
        if "attack" in cmc.columns:
            cmc = cmc[cmc["attack"] == run.attack]
        written.append(plot_windowed_cmc(
            cmc, f"{DATASET_LABELS[run.dataset]}: {scope}", stem_dir / "cmc_curve"))

    rolling_path = run.directory / "rolling_results.csv"
    if rolling_path.exists():
        rolling = pd.read_csv(rolling_path)
        rolling = rolling[rolling["attack"] == run.attack] if "attack" in rolling else rolling
        if f"id_acc{sweep_top_k}" in rolling.columns:
            sweep = rolling.rename(columns={f"id_acc{sweep_top_k}": "id_acc",
                                            f"random_id{sweep_top_k}": "random_id"})
            written.append(plot_window_sweep(
                sweep, sweep_top_k, f"{DATASET_LABELS[run.dataset]}: {scope}",
                stem_dir / f"window_sweep_top{sweep_top_k}"))
        for _, row in rolling.iterrows():
            headline = rolling_headline(rolling, row)
            if headline.empty:
                continue
            window = row["window"] if "window" in rolling.columns else None
            name = (f"topk_accuracy_known{round(row['known_fraction'] * 100)}"
                    + (f"_window{round(window * 100)}" if window is not None else ""))
            span = f"next {window:.0%}" if window is not None else "rest"
            written.append(plot_topk_bars(
                headline,
                f"{DATASET_LABELS[run.dataset]}: {scope}\n"
                f"known {row['known_fraction']:.0%} → {span}",
                stem_dir / name))

    # The older fixed-split runner's two tables.
    headline_path = run.directory / "headline_results.csv"
    if headline_path.exists() and not rolling_path.exists():
        written.append(plot_topk_bars(pd.read_csv(headline_path),
                                      f"{DATASET_LABELS[run.dataset]}: {scope}",
                                      stem_dir / "topk_accuracy"))
    sweep_path = run.directory / "sweep_results.csv"
    if sweep_path.exists():
        written.append(plot_pool_size_sweep(pd.read_csv(sweep_path), sweep_top_k,
                                            f"{DATASET_LABELS[run.dataset]}: {scope}",
                                            stem_dir / f"poolsize_sweep_top{sweep_top_k}"))
    return written


# --- driver ------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """The one knob this script has: which windows to cut each known side into.

    It lives here rather than in the runner because the window is no longer something an
    experiment has to be re-run to change. ``run_experiment_v2.py`` fits on the known side and
    scores the whole remaining timeline once; a window is a chronological prefix of that, so
    asking for a different set of them is a question about the *figures*, answered from the CSVs
    already on disk in seconds.
    """
    parser = argparse.ArgumentParser(
        description="Draw every figure in the project from experiments/results/.")
    parser.add_argument("--window", type=float, nargs="+", default=list(DEFAULT_WINDOWS),
                        metavar="W",
                        help="Fractions of the timeline to cut each known side's unknown "
                             "documents into, one averaged sample per window (default: "
                             "0.1 0.25 0.5). A window running past the end of the corpus is "
                             "dropped rather than truncated. Runs whose CSVs predate the "
                             "per-document `true_author_rank` column ignore this and use the "
                             "windows they were run with.")
    return parser.parse_args()


def main() -> None:
    global WINDOWS
    WINDOWS = tuple(parse_args().window)

    if not RESULTS_DIR.exists():
        raise SystemExit(f"{RESULTS_DIR} does not exist -- run an experiment first.")

    runs = discover_runs(RESULTS_DIR)
    if not runs:
        raise SystemExit(f"no runs named <dataset>_<defense>_<feature>_<attack> under {RESULTS_DIR}")

    by_dataset: dict[str, list[Run]] = defaultdict(list)
    for run in runs:
        by_dataset[run.dataset].append(run)

    total = 0
    all_scaling: dict[Run, ScalingCurve] = {}
    for dataset in DATASETS:
        dataset_runs = by_dataset.get(dataset, [])
        if not dataset_runs:
            continue
        output_dir = PLOTS_DIR / dataset
        print(f"\n[{DATASET_LABELS[dataset]}] {len(dataset_runs)} run(s) -> {output_dir}/")

        # Each curve is built once per run and reused by both families: the same curve appears on
        # one defense-comparison figure and one method-comparison figure, and the wildchat CMC
        # tables are ~9 MB each. A run missing the inputs for one curve type still gets the others.
        averaged: dict[Run, AveragedCmc] = {}
        selective: dict[Run, AveragedRiskCoverage] = {}
        scaling: dict[Run, ScalingCurve] = {}
        exposure: dict[Run, AveragedAuthorRisk] = {}
        modes: dict[Run, pd.DataFrame] = {}
        decay: dict[Run, TemporalDecay] = {}
        for run in dataset_runs:
            curve = average_cmc(run)
            if curve is None:
                print(f"  {run.directory.name}: no cmc_results.csv, CMC figures skipped")
            else:
                averaged[run] = curve
            selective_curve = risk_coverage(run)
            if selective_curve is None:
                print(f"  {run.directory.name}: no predictions_*.csv, risk-coverage skipped")
            else:
                selective[run] = selective_curve
            risk = per_author_risk(run)
            if risk is None:
                print(f"  {run.directory.name}: no author_report_*.csv, per-author risk skipped")
            else:
                exposure[run] = risk
            scaling_curve = load_scaling(run)
            if scaling_curve is not None:
                scaling[run] = scaling_curve
            elif run.attack not in POOL_INTERPOLABLE_ATTACKS:
                print(f"  {run.directory.name}: {ATTACK_LABELS[run.attack]} refits against the "
                      f"gallery, so sub-pool interpolation is not exact -- scaling skipped")
            counted = counting_modes(run, averaged.get(run))
            if counted is not None:
                modes[run] = counted
            staleness = temporal_accuracy(run)
            if staleness is None:
                print(f"  {run.directory.name}: predictions_*.csv has no ended_at column "
                      f"(run predates it), temporal decay skipped -- re-run to include it")
            else:
                decay[run] = staleness
        all_scaling.update(scaling)

        written = []
        for curves, plotter, subdirectory in (
            (averaged, plot_cmc_comparison, "accuracy/by_defense"),
            (selective, plot_risk_coverage_comparison, "risk_coverage/by_defense"),
            (exposure, plot_author_risk_comparison, "author_risk/by_defense"),
            (scaling, plot_scaling, "scaling/by_defense"),
        ):
            written += plot_defense_comparisons(dataset, dataset_runs, curves, plotter,
                                                subdirectory, output_dir)
        for curves, plotter, subdirectory in (
            (averaged, plot_cmc_comparison, "accuracy/by_method"),
            (selective, plot_risk_coverage_comparison, "risk_coverage/by_method"),
            (exposure, plot_author_risk_comparison, "author_risk/by_method"),
            (scaling, plot_scaling, "scaling/by_method"),
        ):
            written += plot_method_comparisons(dataset, dataset_runs, curves, plotter,
                                               subdirectory, output_dir)
        written += plot_macro_micro(dataset, dataset_runs, modes, output_dir)
        if decay:
            written += plot_temporal_decay(dataset, [run for run in dataset_runs if run in decay],
                                           decay, output_dir)
        for run in dataset_runs:
            written += plot_run_detail(run, output_dir)

        for path in written:
            print(f"  {path.relative_to(PLOTS_DIR)}")
        total += len(written)

    # Figures that span datasets go to plots/cross_dataset/ rather than under either corpus:
    # pool size is the one axis where the two are the same experiment at different scales, and
    # filing that under one of them would imply it belongs to that one.
    across = plot_scaling_across_datasets(runs, all_scaling, PLOTS_DIR)
    for path in across:
        print(f"\n{path.relative_to(PLOTS_DIR)}")
    total += len(across)

    print(f"\nWrote {total} figure(s) (PDF + PNG) to {PLOTS_DIR}/")


if __name__ == "__main__":
    main()
