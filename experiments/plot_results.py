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

Output is ``experiments/plots/<dataset>/``. Four curve types are drawn, one directory each, and
every one of them gets the same two views: ``by_defense/<feature>_<attack>.pdf`` (attack fixed, a
line per defense -- *does the defense work?*) and ``by_attack/<defense>.pdf`` (defense fixed, a
line per feature+attack -- *which attack is strongest?*). The layout is uniform on purpose: the
same run appears at the same relative path under every curve type.

``accuracy/by_{defense,attack}/``
    **CMC** -- top-k accuracy against k. The headline privacy question.
``risk_coverage/by_{defense,attack}/``
    **Risk-coverage** -- precision when the attack answers only its most confident documents.
    What a headline accuracy hides: an attack that is usually wrong but knows when it is right.
``author_risk/by_{defense,attack}/``
    **Per-user risk** -- each user's own accuracy, sorted from most to least exposed. Who carries
    the risk, rather than what it averages to.
``scaling/by_{defense,attack}/``
    **Scale** -- top-1 accuracy against the size of the candidate pool. Whether the threat is an
    artefact of a small pool. Drawn only for the attacks in :data:`POOL_INTERPOLABLE_ATTACKS`,
    the ones whose scores do not depend on which other authors are enrolled.

Three figures have no ``by_defense``/``by_attack`` split. ``accuracy/macro_micro.pdf`` puts every
run's top-1 next to itself counted three ways -- per document, per user, per identity -- because
which one a paper leads with is a claim about what "anonymity failed" means, not a detail; it
lives under ``accuracy/`` because it is the same number those curves start from.
``temporal/top1_by_week.pdf`` is the one figure whose x axis is time, and the only one that reads
the whole unknown side rather than the shared test quarter.
``plots/cross_dataset/scaling.pdf`` is the only figure outside the per-dataset folders: pool size
is the single axis along which the corpora are the same experiment at different scales, so filing
it under either one would imply it belonged to that one.

``per_run/<run>/`` keeps the per-run figures the runners used to write themselves -- the
per-window CMC curves, the window sweep, and the top-k bars -- so one experiment's own detail is
still available, now regenerated rather than baked in at run time.

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

#: Keyed by the directory spelling, valued by how the corpus is written in prose and on a figure
#: -- which is the hyphenated "SWE-chat", and stays that way; only the filename changed.
DATASET_LABELS = {"wildchat": "WildChat", "swe_chat": "SWE-chat"}
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


def config_predictions(run: Run) -> dict[str, pd.DataFrame]:
    """``run``'s per-document predictions on the **shared test set**, one table per configuration.

    The single source every comparison figure derives from, and it makes two restrictions that
    every caller would otherwise have to remember:

    * **The shared test set only.** Each configuration scores its whole remaining future (which
      is what the temporal figure needs), but a comparison across configurations is only a
      comparison if they are scored on the same documents -- otherwise a fresher known side is
      also being asked an easier question. The cut is ``position >= round(TEST_START * n)``, an
      integer index, so there is no rounding to reproduce.
    * **In-set documents only.** A document whose author is absent from that configuration's
      known side has no correct answer available, so it has no rank; counting it would cap the
      curve below 1 for a reason the attack cannot control. How many there are is itself a
      result -- it is reported per panel, because it is exactly the *reach* that grows with the
      known side.

    Empty for a run written before this design (a prefix sweep, or the older per-window files);
    those are skipped with a note rather than reinterpreted.
    """
    tables: dict[str, pd.DataFrame] = {}
    for path in sorted(run.directory.glob(f"predictions_{run.attack}_known*.csv")):
        config = parse_config_tag(path.stem[len(f"predictions_{run.attack}_"):])
        if config is None:
            continue
        table = pd.read_csv(path)
        if "attack" in table.columns:
            table = table[table["attack"] == run.attack]
        if table.empty or not {"true_author_rank", "position"}.issubset(table.columns):
            continue
        n_documents = int(table["position"].max()) + 1     # the future always runs to the end
        table = table[table["position"] >= round(TEST_START * n_documents)]
        table = table[table["author_in_known"].astype(bool) & table["true_author_rank"].notna()]
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

    def document_weights(self, author_labels) -> np.ndarray:
        """``(n_replicates, n_documents)`` multiplicities for one panel's documents.

        Users outside the drawn universe (impossible for panels built from the same dataset)
        would weigh zero, which is the correct behaviour rather than an error.
        """
        index = self.authors.get_indexer(pd.Index(author_labels))
        weights = np.where(index >= 0, index, 0)
        block = self.counts[:, weights]
        return np.where(index >= 0, block, 0.0)


def bootstrap_band(values, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pointwise 95% percentile interval for a curve, from one panel's replicate weights.

    ``values`` is called once per replicate with that replicate's document weights and returns
    the curve on a fixed grid. Intervals are **pointwise**, not a simultaneous band over the whole
    curve: two curves whose bands overlap at every k are not resolved by this data, but the
    converse is a per-point statement.
    """
    replicates = np.stack([values(row) for row in weights]) if len(weights) else None
    if replicates is None or not len(replicates):
        point = values(np.ones(weights.shape[1] if weights.ndim == 2 else 0))
        return point, point
    low, high = np.nanpercentile(replicates, [2.5, 97.5], axis=0)
    return np.clip(low, 0.0, 1.0), np.clip(high, 0.0, 1.0)


def weighted_cmc(ranks: np.ndarray, pools: np.ndarray, ks: np.ndarray,
                 weights: np.ndarray | None = None) -> np.ndarray:
    """Top-k accuracy at every ``ks``, with each document weighted by its author's multiplicity.

    Mirrors :func:`prompt_anonymity.metrics.ranking.cmc_curve` -- the share of documents whose
    true author ranks within k -- but weighted, and written out here rather than imported to keep
    this script free of the package's heavy imports (the same reason its name vocabulary is
    literal). Computed as a weighted histogram over ranks plus a prefix sum, so one replicate
    costs one pass over the documents rather than a ``k x documents`` comparison, which at
    WildChat's scale would be a 19,711 x 43,127 array.
    """
    weights = np.ones(len(ranks), dtype=np.float64) if weights is None else weights
    total = weights.sum()
    if total <= 0:
        return np.zeros(len(ks))
    histogram = np.bincount(ranks.astype(np.int64), weights=weights,
                            minlength=int(ks[-1]) + 2)
    return np.cumsum(histogram)[ks] / total


def chance_cmc(pools: np.ndarray, ks: np.ndarray) -> np.ndarray:
    """Random-guessing top-k over the same candidate pools: ``mean(min(k, pool) / pool)``.

    A property of the pool sizes rather than of the attack, so it is a point estimate with no
    band: nothing about it is being measured.
    """
    pools = np.sort(pools)
    inverse = np.concatenate([[0.0], np.cumsum(1.0 / pools)])
    saturated = np.searchsorted(pools, ks, side="right")
    return (saturated + ks * (inverse[-1] - inverse[saturated])) / len(pools)


@dataclass
class ConfigCmc:
    """One (run, known configuration) CMC curve on the shared test set.

    ``curve`` has one row per k with ``accuracy``, the ``random`` baseline and the pointwise
    ``ci_low``/``ci_high`` bootstrap bounds. ``n_documents``/``n_users`` are the in-set counts the
    panel is drawn from -- printed on the figure because they are not a constant across the grid:
    a larger or fresher known side enrolls more of the test set's users, and that *reach* is part
    of what it buys.
    """

    curve: pd.DataFrame
    n_documents: int
    n_users: int
    n_candidates: int
    max_k: int

    @property
    def top1(self) -> float:
        return float(self.curve["accuracy"].iloc[0])


def config_cmc(table: pd.DataFrame, bootstrap: AuthorBootstrap) -> ConfigCmc:
    """CMC curve plus bootstrap band for one configuration's in-set test documents."""
    ranks = table["true_author_rank"].to_numpy(dtype=float)
    pools = table["n_candidate_authors"].to_numpy(dtype=float)
    ks = np.arange(1, int(pools.max()) + 1)
    accuracy = weighted_cmc(ranks, pools, ks)
    low, high = bootstrap_band(lambda weights: weighted_cmc(ranks, pools, ks, weights),
                               bootstrap.document_weights(table["true_author"]))
    curve = pd.DataFrame({"k": ks, "accuracy": accuracy, "random": chance_cmc(pools, ks),
                          "ci_low": low, "ci_high": high})
    return ConfigCmc(curve=curve, n_documents=len(table),
                     n_users=int(table["true_author"].nunique()),
                     n_candidates=int(pools.max()), max_k=int(ks[-1]))


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

    A local restatement of :func:`prompt_anonymity.metrics.detection.selective_classification`,
    weighted by author multiplicity and evaluated on a **pre-sorted** order so a bootstrap
    replicate costs two prefix sums rather than another sort. Coverage is a share of the
    resampled documents, so the cut-off is found on the running weight rather than on a row
    count. ``recall`` is correct answers retained as a fraction of those made at full coverage,
    the sense in which Narayanan et al. reported ">80% precision at 50% recall".
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


def config_risk_coverage(table: pd.DataFrame, bootstrap: AuthorBootstrap) -> ConfigRiskCoverage:
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
        lambda weights: weighted_selective(confidence, correct, order, weights)[0],
        bootstrap.document_weights(table["true_author"]))
    curve = pd.DataFrame({"coverage": COVERAGE_GRID, "precision": precision, "recall": recall,
                          "ci_low": low, "ci_high": high})
    return ConfigRiskCoverage(curve=curve, n_documents=len(table),
                              n_users=int(table["true_author"].nunique()))


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


def weighted_exposure(per_author: np.ndarray, author_weights: np.ndarray | None = None
                      ) -> np.ndarray:
    """The exposure curve: each user's own top-1 accuracy, sorted, read off a percentile grid.

    Users are weighted by their bootstrap multiplicity, so a replicate that drew one user twice
    gives them twice the width on the population axis -- the same curve a real duplicate of that
    user would produce. Positions are the midpoints of each user's slice, so the curve is anchored
    at the centre of a user's width rather than its edge and does not depend on the user count.
    """
    weights = np.ones(len(per_author)) if author_weights is None else author_weights
    order = np.argsort(per_author)[::-1]
    values, widths = per_author[order], weights[order]
    total = widths.sum()
    if total <= 0:
        return np.full(len(EXPOSURE_GRID), np.nan)
    position = (np.cumsum(widths) - widths / 2) / total * 100
    return np.interp(EXPOSURE_GRID, position, values)


def config_author_risk(table: pd.DataFrame, bootstrap: AuthorBootstrap) -> ConfigAuthorRisk:
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
    index = bootstrap.authors.get_indexer(pd.Index(per_author.index))
    weights = bootstrap.counts[:, np.where(index >= 0, index, 0)] if len(bootstrap.counts) else \
        np.empty((0, len(values)), dtype=np.float32)
    weights = np.where(index >= 0, weights, 0.0) if len(weights) else weights
    low, high = bootstrap_band(lambda row: weighted_exposure(values, row), weights)
    curve = pd.DataFrame({"percentile": EXPOSURE_GRID, "accuracy": weighted_exposure(values),
                          "ci_low": low, "ci_high": high})
    return ConfigAuthorRisk(curve=curve, n_documents=len(table), n_users=len(values),
                            never_identified=float((values == 0).mean()))


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


def finish_facets(figure, handles: dict, legend_title: str, title: str, subtitle: str,
                  stem: Path, table: pd.DataFrame, legend_cell=None) -> Path:
    """Shared chrome for every facet figure: one legend, one title, one companion CSV.

    The legend goes inside ``legend_cell`` -- the corner of the triangular grid that holds no
    panel -- so it takes space the figure was giving away rather than a reserved strip that
    shortens every panel. It is drawn on that axes rather than as a figure legend so it moves
    with the cell under ``tight_layout``; a long one is free to overflow, because the cells above
    and to its left are the grid's other two holes.
    """
    if legend_cell is None:
        columns = min(max(len(handles), 1), 4)
        rows = -(-len(handles) // columns)
        bottom = 0.02 + 0.028 * rows
    else:
        columns, bottom = 1, 0.0
    figure.tight_layout(rect=(0, bottom, 1, figure_heading(figure, title, subtitle)))
    if legend_cell is None:
        legend = figure.legend(list(handles.values()), list(handles), title=legend_title,
                               loc="lower center", bbox_to_anchor=(0.5, 0.005), ncol=columns,
                               frameon=False, fontsize=8.5, labelcolor=TEXT_PRIMARY)
    else:
        legend = legend_cell.legend(list(handles.values()), list(handles), title=legend_title,
                                    loc="center", ncol=columns, frameon=False, fontsize=8.5,
                                    labelcolor=TEXT_PRIMARY)
    legend.get_title().set_color(TEXT_SECONDARY)
    legend.get_title().set_fontsize(8.5)
    stem.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(stem.parent / f"{stem.name}.csv", index=False)
    return save_figure(figure, stem)


def draw_cmc_panel(axes, series: list[Series], handles: dict) -> pd.DataFrame:
    """One cell's CMC curves: top-k accuracy against k, log x, with bootstrap bands.

    k is on a log axis because the informative part of a CMC curve is its first decade: an
    attacker who has narrowed 20,000 users to 10 has already won, and what happens at k = 5,000
    is noise about the tail. Series are truncated to the shortest one's k range so every line in
    the panel spans the same axis.

    The random baseline is drawn once per panel, in neutral grey and the figure's only dash: it
    depends on the candidate-pool size, which is a property of the configuration rather than of
    the defense or feature, so every series in a cell shares it.
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
    axes.plot(baseline["k"], baseline["random"], color=TEXT_MUTED, linewidth=1.2,
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


def draw_scaling_panel(axes, series: list[Series], handles: dict) -> pd.DataFrame:
    """One cell's scaling curves: top-1 accuracy against how many users the attack ranks over.

    The claim is not the height of any line but the *widening gap*: chance falls away as ``1/n``
    while the attack decays far more slowly, so anonymity does not recover by adding users. A
    hollow ring sits at the pool actually run; everything to its left is the sub-pool
    interpolation, so an interpolated stretch is never mistaken for a measurement.
    """
    rows = []
    for item, slot in zip(series, resolve_slots([item.slot for item in series])):
        curve = item.curve.curve
        color = series_style(slot)
        axes.fill_between(curve["n_candidates"], curve["ci_low"], curve["ci_high"], color=color,
                          alpha=BAND_ALPHA, linewidth=0, zorder=2)
        axes.plot(curve["n_candidates"], curve["accuracy"], color=color, linewidth=LINE_WIDTH,
                  solid_capstyle="round", zorder=3)
        axes.plot(item.curve.measured["n_candidates"], item.curve.measured["accuracy"],
                  linestyle="none", marker="o", markersize=MARKER_SIZE, markerfacecolor=SURFACE,
                  markeredgecolor=color, markeredgewidth=1.6, zorder=4)
        handles.setdefault(item.label, Line2D([], [], color=color, linewidth=LINE_WIDTH))
        rows.append(curve.assign(series=item.label))
    baseline = series[0].curve.curve
    axes.plot(baseline["n_candidates"], baseline["random"], color=TEXT_MUTED, linewidth=1.2,
              linestyle=BASELINE_DASH, zorder=2)
    axes.set_xscale("log")
    axes.set_ylim(0, 1.02)
    panel_note(axes, f"{series[0].curve.n_documents:,} docs · {series[0].curve.n_users:,} users")
    return pd.concat(rows, ignore_index=True)


#: The four curve types, each as (panel drawer, subdirectory, x label, y label, subtitle).
CURVE_TYPES = {
    "cmc": (draw_cmc_panel, "accuracy", "k (candidate authors returned)", "Top-k accuracy",
            "Shaded: 95% bootstrap CI over users (gallery fixed). Grey dashes: random guessing"),
    "risk_coverage": (draw_risk_coverage_panel, "risk_coverage",
                      "Coverage (share of documents answered)", "Precision",
                      "Confidence is the attack's cohort-normalised margin; in-set documents only"),
    "author_risk": (draw_author_risk_panel, "author_risk",
                    "Share of users, most exposed first (%)", "That user's own top-1 accuracy",
                    "A cliff means the risk sits with a few users, not with the average one"),
    "scaling": (draw_scaling_panel, "scaling", "Candidate users the attack ranks over",
                "Top-1 accuracy",
                "Line interpolates down to smaller galleries; rings are the pools actually run"),
}


def plot_config_comparison(kind: str, panels: dict[str, list[Series]], title: str,
                           legend_title: str, stem: Path) -> Path:
    """One curve type, one figure, one panel per known configuration.

    Nothing is averaged across the grid. The configurations differ in what the attacker was
    given, which is an experimental condition rather than a repeated measurement -- averaging
    them would report a number describing no experiment that was run, and the mean would move
    with the arbitrary choice of which cells were included.
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
                         legend_cell=legend_cell)


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


# --- temporal decay: does the attack go stale? -------------------------------

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
    """One run's top-1 accuracy per week, split by whether the week's users are new to the attack.

    ``curve`` is long-form -- ``week``, ``cohort``, ``accuracy``, ``n_documents``, ``n_users`` --
    with ``cohort`` in :data:`COHORTS`. ``counts`` is the same population *unfiltered*, so the
    context panel can show a thin week that the accuracy line drops.
    """

    curve: pd.DataFrame
    counts: pd.DataFrame
    known_config: str


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
    # The dataset part of a run's name is the split name, so it is also the parquet's basename.
    path = DATA_DIR / f"{dataset}.parquet"
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


def temporal_accuracy(run: Run, known_config: str = TEMPORAL_KNOWN_CONFIG,
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

    One known side (``known_config``), so there is nothing to average and no interval to draw:
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
                         known_config=known_config)


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

    # Truncate every line to the last week they all reach, so a run that happens to hold one
    # extra sparse week does not stretch the axis for everyone.
    limit = min(int(decay[run].curve["week"].max()) for run in runs)
    short = sorted(run.directory.name for run in runs
                   if int(decay[run].curve["week"].max()) == limit)
    if any(int(decay[run].curve["week"].max()) > limit for run in runs):
        print(f"  temporal: truncated to week {limit}; {', '.join(short)} do not reach further")

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

    known = parse_config_tag(next(iter(decay.values())).known_config)
    figure.suptitle(f"{DATASET_LABELS[dataset]}: does re-identification go stale?  "
                    f"(attacker holds {known.start:.0%}-{known.end:.0%} of the timeline)",
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
                                       known_config=decay[run].known_config)
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


#: The known configuration the counting-modes figure is drawn for. One cell rather than the grid:
#: the figure's claim is that the three *counting modes* disagree, and repeating it six times
#: would spend a page making the same point. The largest, freshest known side is the attacker's
#: best case, so the spread shown is the one that matters most.
HEADLINE_CONFIG = "known0075"


def counting_modes(tables: dict[str, pd.DataFrame], bootstrap: AuthorBootstrap
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
    if table is None or table.empty:
        return None
    hits = (table["true_author_rank"].to_numpy(dtype=float) <= 1).astype(float)
    authors = pd.Index(table["true_author"])
    author_index = pd.factorize(authors)[0]
    n_authors = author_index.max() + 1

    def modes(weights: np.ndarray) -> np.ndarray:
        """(micro, macro, identity) under one set of document weights."""
        per_author_hits = np.bincount(author_index, weights=weights * hits, minlength=n_authors)
        per_author_docs = np.bincount(author_index, weights=weights, minlength=n_authors)
        present = per_author_docs > 0
        if not present.any() or weights.sum() <= 0:
            return np.full(3, np.nan)
        rates = per_author_hits[present] / per_author_docs[present]
        # Users weigh their multiplicity in the two per-user modes: a user drawn twice counts
        # twice, exactly as a duplicate of that person in the corpus would.
        author_weight = per_author_docs[present] / np.maximum(
            np.bincount(author_index, minlength=n_authors)[present], 1)
        return np.array([float(weights @ hits / weights.sum()),
                         float(np.average(rates, weights=author_weight)),
                         float(np.average(rates > 0, weights=author_weight))])

    weights = bootstrap.document_weights(table["true_author"])
    point = modes(np.ones(len(hits)))
    low, high = bootstrap_band(modes, weights)
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


def config_scaling(run: Run, table: pd.DataFrame, bootstrap: AuthorBootstrap
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

    Returns ``None`` for any attack outside :data:`POOL_INTERPOLABLE_ATTACKS`, where the
    estimator would not be exact.
    """
    if run.attack not in POOL_INTERPOLABLE_ATTACKS:
        return None
    ranks = table["true_author_rank"].to_numpy(dtype=float)
    n_candidates = int(table["n_candidate_authors"].max())
    pool_sizes = pool_lattice(n_candidates)
    weights = subpool_weights(n_candidates, pool_sizes)
    ks = np.arange(1, n_candidates + 1)

    def curve_for(document_weights: np.ndarray | None = None) -> np.ndarray:
        # The CMC is the cumulative rank distribution, so differencing it recovers the mass at
        # each rank; the shortfall from 1 is the documents whose author was never a candidate,
        # which are wrong at every pool size.
        cmc = weighted_cmc(ranks, None, ks, document_weights)
        return np.diff(cmc, prepend=0.0) @ weights

    low, high = bootstrap_band(curve_for, bootstrap.document_weights(table["true_author"]))
    curve = pd.DataFrame({"n_candidates": pool_sizes, "accuracy": curve_for(),
                          "random": 1.0 / pool_sizes.astype(float),
                          "ci_low": low, "ci_high": high})
    measured = pd.DataFrame([{"n_candidates": n_candidates,
                              "accuracy": float((ranks <= 1).mean())}])
    return ScalingCurve(curve=curve, measured=measured, n_documents=len(table),
                        n_users=int(table["true_author"].nunique()))


def plot_scaling(series: list[Series], title: str, legend_title: str, stem: Path) -> Path:
    """Top-1 accuracy against candidate-pool size, one line per run, log x.

    The claim is not the height of any line but the *widening gap*: chance falls away as ``1/n``
    while the attack decays far more slowly, so anonymity does not recover by adding users. Open
    markers sit at each run's real pool size -- the pools actually run -- and the line through
    the smaller sizes is the sub-pool interpolation, which is what lets a corpus of dozens and a
    corpus of tens of thousands be read on one axis.
    """
    figure, axes = plt.subplots(figsize=(7.4, 4.7))
    figure.patch.set_facecolor(SURFACE)

    baselines, colors = [], {}
    slots = resolve_slots([item.slot for item in series], [item.dash for item in series])
    for item, slot in zip(series, slots):
        curve, measured = item.curve.curve, item.curve.measured
        color = series_style(slot)
        colors.setdefault(item.label, color)
        baselines.append(curve[["n_candidates", "random"]].set_index("n_candidates")["random"])
        axes.fill_between(curve["n_candidates"], curve["ci_low"], curve["ci_high"], color=color,
                          alpha=BAND_ALPHA, linewidth=0, zorder=2)
        axes.plot(curve["n_candidates"], curve["accuracy"], color=color, linewidth=LINE_WIDTH,
                  linestyle=(0, item.dash) if item.dash else "-", solid_capstyle="round", zorder=3)
        # A hollow marker for the pool actually run, so an interpolated stretch of line is
        # never mistaken for a measurement.
        axes.plot(measured["n_candidates"], measured["accuracy"], linestyle="none", marker="o",
                  markersize=MARKER_SIZE, markerfacecolor=SURFACE, markeredgecolor=color,
                  markeredgewidth=1.6, zorder=4)

    baseline = pd.concat(baselines, axis=1).mean(axis=1).sort_index()
    axes.plot(baseline.index, baseline.to_numpy(), color=TEXT_MUTED, linewidth=1.4,
              linestyle=BASELINE_DASH, zorder=2)

    style_axes(axes, "Candidate users the attack ranks over", "Top-1 accuracy", title,
               f"Line interpolates down to smaller galleries; rings are the pools actually "
               f"run  ·  {parse_config_tag(HEADLINE_CONFIG).label}")
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
        curve = item.curve.curve.set_index("n_candidates")
        for column, suffix in (("accuracy", ""), ("ci_low", " ci_low"), ("ci_high", " ci_high")):
            table[f"{item.column}{suffix}"] = curve[column].reindex(baseline.index).to_numpy()
    stem.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(stem.parent / f"{stem.name}.csv", index=False)
    return save_figure(figure, stem)


def plot_scaling_across_datasets(runs: list[Run], scaling: dict[Run, dict[str, ScalingCurve]],
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

    One known configuration (:data:`HEADLINE_CONFIG`), not the grid: this figure asks how far the
    threat reaches, and its x axis is already the candidate pool, so drawing six cells would put
    six nearly parallel lines per method on an axis whose whole point is the comparison *between*
    methods. The largest, freshest known side is the attacker's best case and has the widest
    measured pool, so its interpolation reaches furthest.
    """
    ordered = sorted(
        (run for run in runs
         if run.defense == NO_DEFENSE and HEADLINE_CONFIG in scaling.get(run, {})),
        key=lambda run: (METHOD_SLOTS[run.method], DATASETS.index(run.dataset)),
    )
    # Because every run's curve now reaches down to two candidates, the corpora overlap on the
    # shared range instead of sitting as two clumps with two empty decades between them.
    if len(ordered) < 2:
        return []
    # Colour is the method alone -- so the same feature+attack is one hue on both corpora --
    # and the dataset rides on the dash instead.
    series = [Series(run.method_label, METHOD_SLOTS[run.method],
                     scaling[run][HEADLINE_CONFIG], run.dataset) for run in ordered]
    return [plot_scaling(series, title="Re-identification against candidate-pool size",
                         legend_title="Feature / attack",
                         stem=output_dir / "cross_dataset" / "scaling")]


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
    return parser.parse_args()


def build_curves(runs: list[Run], bootstrap_replicates: int):
    """Every curve every figure needs, built once per (run, configuration).

    One pass over the predictions: the same table feeds the CMC, risk-coverage, per-user risk,
    scaling and counting-mode curves, and each run's curve object is reused by both comparison
    families rather than rebuilt for each. Runs whose files predate the configuration design are
    reported and skipped -- their known sides were prefixes scored on different documents, so
    they are not cells of this grid.
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

    cmc, selective, exposure, scaling, modes = {}, {}, {}, {}, {}
    for run in runs:
        if not tables[run]:
            continue
        cmc[run] = {tag: config_cmc(table, bootstrap) for tag, table in tables[run].items()}
        selective[run] = {tag: config_risk_coverage(table, bootstrap)
                          for tag, table in tables[run].items()}
        exposure[run] = {tag: config_author_risk(table, bootstrap)
                         for tag, table in tables[run].items()}
        pools = {tag: config_scaling(run, table, bootstrap)
                 for tag, table in tables[run].items()}
        pools = {tag: curve for tag, curve in pools.items() if curve is not None}
        if pools:
            scaling[run] = pools
        elif run.attack not in POOL_INTERPOLABLE_ATTACKS:
            print(f"  {run.directory.name}: {ATTACK_LABELS[run.attack]} refits against the "
                  f"gallery, so sub-pool interpolation is not exact -- scaling skipped")
        counted = counting_modes(tables[run], bootstrap)
        if counted is not None:
            modes[run] = counted
    return tables, bootstrap, cmc, selective, exposure, scaling, modes


def main() -> None:
    replicates = parse_args().bootstrap

    if not RESULTS_DIR.exists():
        raise SystemExit(f"{RESULTS_DIR} does not exist -- run an experiment first.")

    runs = discover_runs(RESULTS_DIR)
    if not runs:
        raise SystemExit(f"no runs named <dataset>_<defense>_<feature>_<attack> under {RESULTS_DIR}")

    by_dataset: dict[str, list[Run]] = defaultdict(list)
    for run in runs:
        by_dataset[run.dataset].append(run)

    total = 0
    all_scaling: dict[Run, dict[str, ScalingCurve]] = {}
    for dataset in DATASETS:
        dataset_runs = by_dataset.get(dataset, [])
        if not dataset_runs:
            continue
        output_dir = PLOTS_DIR / dataset
        print(f"\n[{DATASET_LABELS[dataset]}] {len(dataset_runs)} run(s) -> {output_dir}/")

        tables, bootstrap, cmc, selective, exposure, scaling, modes = build_curves(
            dataset_runs, replicates)
        all_scaling.update(scaling)

        decay = {}
        for run in dataset_runs:
            staleness = temporal_accuracy(run)
            if staleness is None:
                print(f"  {run.directory.name}: no predictions for {TEMPORAL_KNOWN_CONFIG} "
                      f"(or no split parquet for its timestamps), temporal decay skipped")
            else:
                decay[run] = staleness

        written = []
        for kind, curves in (("cmc", cmc), ("risk_coverage", selective),
                             ("author_risk", exposure), ("scaling", scaling)):
            written += plot_defense_comparisons(dataset, dataset_runs, curves, kind, output_dir)
            written += plot_attack_comparisons(dataset, dataset_runs, curves, kind, output_dir)
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
