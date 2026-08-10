#!/usr/bin/env python
"""Draw a UMAP projection of a corpus's document embeddings, one figure per dataset.

Run it with no arguments::

    python experiments/plot_datasets.py

This is a *dataset* figure, not a results figure: it reads the feature parquets in ``data/hf/``
directly and never opens ``experiments/results/``, so it needs no experiment to have been run. Its
question is the one every attack in this project rests on -- **do a user's documents sit together
in embedding space at all?** -- and it answers it by eye rather than by a metric.

**Every document in the corpus is projected**, all 172,509 of WildChat's from all 25,357 users;
nothing is sampled away. What makes that legible is that the figure has two layers rather than one:

*context*
    every document that is not in focus, as a **grey density field** -- a fine 2D histogram,
    histogram-equalised so the dense core and the sparse filaments both resolve. Drawn with
    ``imshow``, so it is one raster the size of its own bin grid rather than 170,000 artists.
*focus*
    the :data:`DEFAULT_FOCUS_USERS` most prolific users, each in its own colour from a vendored
    Glasbey palette (see :func:`glasbey_palette`), drawn as rasterized points over the field.

**The figure carries no text at all** -- no title, no axis labels, no legend, no per-point codes,
no note. It is a bare projection, so it drops straight into a paper where the caption does the
talking, and it composes into a multi-panel figure without two competing sets of titles. Nothing
is lost from the record: every parameter is printed to the console on each run and non-default
choices are spelled into the filename by :func:`output_tag`.

Input is ``data/hf/<dataset>[_<defense>]_<feature>.parquet`` (the same feature parquets
``run_experiment.py`` reads) and output is::

    experiments/plots/<dataset>/umap/<defense>_<feature>.pdf

which follows the per-dataset ``<family>/`` scheme ``plot_results.py`` writes into, so these
figures sit beside the results figures for the same corpus rather than in a directory of their own.

The chart surface and the neutral greys are imported from ``plot_results.py`` rather than restated,
so the two scripts stay one visual set.

**Read the geometry loosely.** UMAP axes are arbitrary and only local structure is faithful: which
points are near each other means something, how far apart two clusters are does not, and neither
does the shape or orientation of the whole. It is an illustration of separability, not a
measurement of it -- the measurement is ``plot_results.py``'s accuracy figures.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: render to files without a display
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_hex, to_rgb  # noqa: E402
from scipy.ndimage import gaussian_filter  # noqa: E402

# The sibling script is the project's one place for chart style. Imported as a module as well as
# by name so ``--png`` can set its ``WRITE_PNG`` global, which ``save_figure`` reads.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import plot_results  # noqa: E402
from plot_results import (  # noqa: E402
    DATA_DIR, DATASET_LABELS, DATASETS, DEFENSES, FEATURES, NO_DEFENSE, PLOTS_DIR, REPO_ROOT,
    SURFACE, save_figure,
)

#: Where a computed projection is parked. Keyed by every input that could change it (see
#: :func:`layout_cache_path`), so restyling a figure never recomputes the embedding -- which is the
#: expensive half, minutes on WildChat against seconds to draw.
CACHE_DIR = REPO_ROOT / "experiments" / ".cache" / "umap"

#: The feature to project when ``--feature`` is not given. The semantic vectors are the ones worth
#: looking at: they carry more than double StyloMetrix's attribution accuracy, so they are where a
#: by-eye cluster is expected to show up at all.
DEFAULT_FEATURE = "gemini_embedding_2"

#: Sentinel for ``--focus-users all``: colour *every* user, leaving no grey field at all.
ALL_USERS = -1

#: How many users are drawn in colour, most prolific first, when ``--focus-users`` is not given --
#: **a property of the corpus, because the two corpora differ by two orders of magnitude in users.**
#:
#: SWE-chat's 157 users all fit: measured, a 157-colour palette still holds a minimum pairwise
#: CIELAB separation of 14.3 (median nearest 16.4), roughly six times the just-noticeable
#: difference, so no two users collide even though no reader could hold 157 colours in their head.
#: That is the right reading of this figure anyway -- with no legend and no labels, colour marks a
#: cluster's *extent and coherence*, not which person it belongs to.
#:
#: WildChat's 25,357 cannot, so it takes a focus set over the grey field. **250 is where the
#: palette stops being comfortable, not an arbitrary round number**: the sRGB gamut is 820,338
#: dE^3 in CIELAB and 503,417 after the bounds below, which sphere-packs to ~211 colours at a
#: comfortable dE 15 and ~712 at dE 10, the floor for marks this small. Measured, the construction
#: below holds dE 12.1 at 250 and 9.0 at 500 -- so 250 sits just under the comfortable bound and
#: 500 would be at the limit of what reads apart at all.
DEFAULT_FOCUS_USERS = {"wildchat": 250, "swe_chat": ALL_USERS}

#: Used for a dataset with no entry above.
FALLBACK_FOCUS_USERS = 250

#: UMAP's own defaults, restated so they appear in the cache key. ``n_neighbors`` is the
#: local/global trade-off (small = fine structure), ``min_dist`` how tightly a cluster is allowed
#: to pack, and the metric is **cosine** because that is what every attack in the project compares
#: these vectors with (``--metric``'s default in ``run_experiment.py``).
DEFAULT_N_NEIGHBORS = 15
DEFAULT_MIN_DIST = 0.1
DEFAULT_METRIC = "cosine"
DEFAULT_SEED = 0


# --- reading the vectors -----------------------------------------------------
#
# Read **by column slab, not by row batch**. These parquets hold one row group, so `iter_batches`
# has nothing to skip and materialises the whole group whatever row filter is applied: measured on
# WildChat's 3.0 GB file, keeping 20,000 of 172,509 rows still cost 7.9 GB resident, half the
# cluster's 16 GB cap. Projecting a column slab reads only those column chunks, so the *entire*
# 172,509 x 3,072 matrix comes back in 8.0 s at 2.58 GB -- less memory for all of it than the row
# path spent on a tenth of it.

#: Columns decoded at once. At 172,509 rows a slab of this width is ~177 MB, so peak memory is the
#: result plus one slab.
COLUMN_SLAB = 256


def document_authors(parquet_path: Path) -> np.ndarray:
    """The ``author_id`` of every row of a feature parquet, in file order.

    Only that one column is decoded, which is what makes it cheap on WildChat: the file is 3.0 GB
    and this reads a few megabytes of it. File order is the join key everything else here uses --
    the vectors, the layout and this array are all in it.
    """
    return pq.read_table(parquet_path, columns=["author_id"]).column("author_id").to_pandas(
        ).to_numpy()


def read_all_vectors(parquet_path: Path) -> np.ndarray:
    """Every feature vector in the parquet, as a ``float32`` array in file order.

    Fills one preallocated array a column slab at a time (see the note above), so nothing larger
    than the result is ever live.
    """
    parquet_file = pq.ParquetFile(parquet_path)
    columns = [name for name in parquet_file.schema_arrow.names
               if name not in ("doc_id", "author_id")]
    vectors = np.empty((parquet_file.metadata.num_rows, len(columns)), dtype=np.float32)
    for start in range(0, len(columns), COLUMN_SLAB):
        slab = parquet_file.read(columns=columns[start:start + COLUMN_SLAB])
        for offset, column in enumerate(slab.columns):
            vectors[:, start + offset] = column.combine_chunks().to_numpy(zero_copy_only=False)
        del slab
    return vectors


# --- the projection ----------------------------------------------------------

def layout_cache_path(key: dict) -> Path:
    """Cache file for one projection, named after a hash of everything that could change it."""
    digest = hashlib.sha1(json.dumps(key, sort_keys=True).encode()).hexdigest()[:16]
    return CACHE_DIR / f"{key['dataset']}_{key['defense']}_{key['feature']}_{digest}.npz"


def umap_layout(vectors: np.ndarray, n_neighbors: int, min_dist: float, metric: str,
                seed: int) -> np.ndarray:
    """Project ``vectors`` to two dimensions with UMAP, deterministically.

    ``random_state`` is set, which pins the result at the cost of UMAP's parallel optimiser. That
    is the right trade and it is cheaper than it sounds: measured over all 172,509 WildChat
    documents, seeded is 123 s against 72 s unseeded, and a figure that moves between runs cannot
    be compared with the one in a draft.

    The vectors are fed **raw**, at their full width, so ``metric`` means exactly what
    ``run_experiment.py`` means by it. Reducing them with PCA first would roughly halve the time
    and the memory (259 s / 9.5 GB becomes 134 s / ~5 GB on WildChat) but only by swapping the
    stated metric for one that merely ranks the same way, which is not worth caveating a figure
    over.
    """
    import umap  # imported lazily: it pulls in numba, seconds of import time for nothing if cached

    reducer = umap.UMAP(n_components=2, n_neighbors=min(n_neighbors, len(vectors) - 1),
                        min_dist=min_dist, metric=metric, random_state=seed, verbose=False)
    return np.asarray(reducer.fit_transform(vectors), dtype=np.float32)


def document_layout(dataset: str, defense: str, feature: str, data_dir: Path, options
                    ) -> pd.DataFrame:
    """The frame every figure is drawn from: one row per document, with ``x``/``y`` and its author.

    Reads the cache when one matches, otherwise reads every vector in the parquet and runs UMAP
    over all of them, then writes the cache. What is cached is the *layout*, not the vectors -- a
    couple of megabytes against the gigabytes they came from. Author ids are not cached either;
    they are one cheap column read and keeping them out means the cache cannot fall out of step
    with the parquet it describes.
    """
    parquet_path = feature_parquet(dataset, defense, feature, data_dir)
    key = {"dataset": dataset, "defense": defense, "feature": feature,
           "n_neighbors": options.n_neighbors, "min_dist": options.min_dist,
           "metric": options.metric, "seed": options.seed,
           "max_documents": options.max_documents}
    cache_path = layout_cache_path(key)
    authors = document_authors(parquet_path)

    rows = np.arange(len(authors))
    if 0 < options.max_documents < len(authors):
        # A uniform subsample of *documents*, for a fast look. It thins every user in proportion
        # rather than dropping users, so the picture degrades smoothly instead of changing subject.
        rows = np.sort(np.random.default_rng(options.seed).choice(
            len(authors), options.max_documents, replace=False))

    if cache_path.exists() and not options.refresh:
        cached = np.load(cache_path)
        print(f"  layout from cache: {cache_path.name}")
        return pd.DataFrame({"author_id": authors[rows], "x": cached["x"], "y": cached["y"]})

    print(f"  {len(rows):,} documents from {pd.unique(authors[rows]).size:,} users"
          f" -- reading vectors from {parquet_path.name}")
    started = time.perf_counter()
    vectors = read_all_vectors(parquet_path)
    if len(rows) != len(authors):
        vectors = vectors[rows]
    print(f"  read {vectors.shape[0]:,} x {vectors.shape[1]:,} vectors"
          f" in {time.perf_counter() - started:.1f} s -- projecting")
    started = time.perf_counter()
    coordinates = umap_layout(vectors, options.n_neighbors, options.min_dist, options.metric,
                              options.seed)
    del vectors
    print(f"  UMAP finished in {time.perf_counter() - started:.1f} s")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, x=coordinates[:, 0], y=coordinates[:, 1])
    return pd.DataFrame({"author_id": authors[rows],
                         "x": coordinates[:, 0], "y": coordinates[:, 1]})


def feature_parquet(dataset: str, defense: str, feature: str, data_dir: Path) -> Path:
    """Path of the feature parquet for one (dataset, defense, feature), undefended spelled bare."""
    stem = dataset if defense == NO_DEFENSE else f"{dataset}_{defense}"
    return data_dir / f"{stem}_{feature}.parquet"


# --- the focus palette -------------------------------------------------------
#
# A Glasbey palette, constructed here rather than installed: the algorithm is a greedy max-min in a
# perceptual space over a quantised sRGB cube, which is forty lines, and vendoring it keeps the
# construction and its bounds in the repo instead of behind a dependency. Glasbey et al. (2007)
# used CIELAB and so does this.

#: sRGB cube resolution the palette is chosen from. 24 steps is 13,824 candidates, of which the
#: bounds below keep ~5,000 -- fine enough that the greedy pick is not grid-limited, small enough
#: that the pairwise step is instant.
PALETTE_GRID = 24

#: The bounds that **exclude white, black and grey** by construction, which is the whole reason a
#: palette is generated rather than cycled. Anything too light or too dark to hold a hue is out on
#: ``L*``; anything too close to the neutral axis to read as a colour at all is out on chroma --
#: and that second bound is what keeps a focus user from being mistaken for the grey density field
#: underneath.
PALETTE_LIGHTNESS = (30.0, 85.0)
PALETTE_CHROMA_FLOOR = 30.0

#: Colours the palette is pushed *away* from, as though already chosen. The chart surface and the
#: density ramp go in here, so no focus colour can land on the background or on the field it is
#: drawn over; pure black and white are named as well, since the bounds alone leave near-neutrals
#: at the extremes of the kept region.
PALETTE_SEEDS = ("#fcfcfb", "#e6e5e1", "#a5a49f", "#52514e", "#000000", "#ffffff")


def srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """CIELAB (D65) for an ``(n, 3)`` array of sRGB values in ``[0, 1]``."""
    linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    to_xyz = np.array([[0.4124564, 0.3575761, 0.1804375],
                       [0.2126729, 0.7151522, 0.0721750],
                       [0.0193339, 0.1191920, 0.9503041]])
    xyz = (linear @ to_xyz.T) / np.array([0.95047, 1.0, 1.08883])  # normalised to the D65 white
    f = np.where(xyz > 216 / 24389, np.cbrt(xyz), (841 / 108) * xyz + 4 / 29)
    return np.stack([116 * f[:, 1] - 16,
                     500 * (f[:, 0] - f[:, 1]),
                     200 * (f[:, 1] - f[:, 2])], axis=-1)


def glasbey_palette(n_colours: int) -> list[str]:
    """``n_colours`` maximally distinct hex colours, none of them white, black or grey.

    Greedy max-min: from a quantised sRGB cube filtered to :data:`PALETTE_LIGHTNESS` and
    :data:`PALETTE_CHROMA_FLOOR`, repeatedly take the candidate whose nearest already-chosen colour
    (:data:`PALETTE_SEEDS` counting as chosen from the start) is furthest away in CIELAB. The
    result is a *prefix-stable* sequence -- the first k of a palette of n are the palette of k --
    so adding a user never repaints the others, which is the project's colour rule.

    It is deterministic: ties in the greedy pick fall to the lower grid index, and nothing here is
    random.
    """
    levels = np.linspace(0.0, 1.0, PALETTE_GRID)
    rgb = np.stack(np.meshgrid(levels, levels, levels, indexing="ij"), axis=-1).reshape(-1, 3)
    lab = srgb_to_lab(rgb)
    low, high = PALETTE_LIGHTNESS
    within = ((lab[:, 0] >= low) & (lab[:, 0] <= high)
              & (np.hypot(lab[:, 1], lab[:, 2]) >= PALETTE_CHROMA_FLOOR))
    rgb, lab = rgb[within], lab[within]
    if len(rgb) < n_colours:
        raise ValueError(f"palette bounds leave only {len(rgb)} candidates for {n_colours} colours")

    seeds = srgb_to_lab(np.array([to_rgb(colour) for colour in PALETTE_SEEDS]))
    nearest = np.linalg.norm(lab[:, None, :] - seeds[None, :, :], axis=-1).min(axis=1)
    chosen: list[str] = []
    for _ in range(n_colours):
        pick = int(np.argmax(nearest))
        chosen.append(to_hex(rgb[pick]))
        nearest = np.minimum(nearest, np.linalg.norm(lab - lab[pick], axis=-1))
    return chosen


# --- drawing -----------------------------------------------------------------

#: The grey the context field is shaded with, light to dark. It stops at the project's secondary
#: text grey rather than at black, so even a saturated bin stays lighter than every focus colour
#: drawn over it and the field cannot compete with the marks it is background for.
CONTEXT_RAMP = LinearSegmentedColormap.from_list("context", [SURFACE, "#e6e5e1", "#a5a49f",
                                                             "#52514e"])

#: Bins per side of the density histogram. Fine enough that the field keeps the grain of individual
#: documents rather than reading as blocks -- at this resolution WildChat fills 69,934 of the 1.44M
#: cells, so the field is nearly a scatter, which is the texture it is wanted for.
DEFAULT_DENSITY_BINS = 1200

#: Gaussian smoothing applied to the field, in bins. **Zero by default, deliberately.** A blur
#: knits sparse filaments together, but it is applied after the equalisation and so dilutes exactly
#: the isolated documents the equalisation just lifted -- a lone document ends up lighter than one
#: in a cluster, which reads as "less certain" rather than "one document". Raise it only if the
#: grain is too fine at some other bin count.
DENSITY_BLUR = 0.0

#: Focus marks carry a thin surface ring so overlapping points stay separable (the project's mark
#: spec) and are rasterized, because at full corpus scale there are 172,509 of them. Size is
#: interpolated in log point count -- 9 pt reads well for SWE-chat's 4,334 marks and would be a
#: solid slab at WildChat's 172,509 -- and the ring is dropped once a mark is too small to hold one.
POINT_SIZE_RANGE = (2.5, 9.0)
POINT_COUNT_RANGE = (3.5, 5.3)  # log10 documents, i.e. ~3,000 to ~200,000
POINT_EDGE_WIDTH = 0.35
MIN_RINGED_POINT_SIZE = 4.0


def point_size(n_points: int) -> float:
    """Marker area for a scatter of ``n_points`` marks."""
    low, high = POINT_SIZE_RANGE
    return float(np.clip(np.interp(np.log10(max(n_points, 1)), POINT_COUNT_RANGE, (high, low)),
                         low, high))

#: How much room is left around the projected points, as a share of their spread.
AXES_PADDING = 0.02


def equalise(counts: np.ndarray) -> np.ndarray:
    """Histogram-equalise ``counts`` onto ``[0, 1]``, leaving empty bins at 0.

    Document density is heavily skewed, so a linear ramp paints the core one flat tone and leaves
    every sparse arm on the bottom step. Ranking the occupied bins spends the whole ramp on the
    values that actually occur. Ties share a level, so two bins holding the same count can never
    take different tones.
    """
    levels = np.zeros(counts.shape, dtype=np.float32)
    filled = counts > 0
    values = counts[filled]
    unique, inverse = np.unique(values, return_inverse=True)
    levels[filled] = (np.searchsorted(np.sort(values), unique, side="right") / len(values)
                      )[inverse]
    return levels


def padded_limits(values: np.ndarray) -> tuple[float, float]:
    """``(low, high)`` covering ``values``, with :data:`AXES_PADDING` of their spread either side."""
    low, high = float(values.min()), float(values.max())
    margin = AXES_PADDING * max(high - low, 1e-6)
    return low - margin, high + margin


def draw_umap(frame: pd.DataFrame, focus: list[str], options):
    """Render one dataset's projection: a grey density field with the focus users drawn over it.

    The figure holds **no text of any kind** -- the axes fill the canvas edge to edge, carry no
    ticks, spines, grid, labels or legend, and nothing is annotated. What it contains is the two
    layers and nothing else, so the caption around it is free to say whatever the figure is being
    used to say.
    """
    figure = plt.figure(figsize=(options.figsize, options.figsize), facecolor=SURFACE)
    axes = figure.add_axes((0, 0, 1, 1))  # fill the canvas: no margins to hold labels
    axes.set_facecolor(SURFACE)

    in_focus = frame["author_id"].isin(focus).to_numpy()
    context = frame.loc[~in_focus]
    x_limits, y_limits = padded_limits(frame["x"].to_numpy()), padded_limits(frame["y"].to_numpy())

    # No field when every user is in colour -- on a corpus small enough to colour outright there is
    # nothing left to be context, and binning it would only add an empty raster.
    if len(context):
        # One raster the size of its own bin grid, not one artist per document: `imshow` embeds the
        # field at `bins` resolution whatever the figure's dpi, which is both sharper and ~20x
        # smaller in the PDF than the equivalent rasterized scatter.
        counts, x_edges, y_edges = np.histogram2d(
            context["x"].to_numpy(), context["y"].to_numpy(), bins=options.density_bins,
            range=(x_limits, y_limits))
        field = equalise(counts)
        if DENSITY_BLUR:
            field = gaussian_filter(field, DENSITY_BLUR)
        axes.imshow(np.ma.masked_where(field <= 0, field).T, origin="lower", zorder=1,
                    extent=(x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]), cmap=CONTEXT_RAMP,
                    norm=Normalize(0, 1), interpolation="nearest", aspect="auto")

    coloured = frame.loc[in_focus]
    if len(coloured):
        palette = glasbey_palette(len(focus))
        size = point_size(len(coloured))
        edge = POINT_EDGE_WIDTH if size >= MIN_RINGED_POINT_SIZE else 0.0
        # Drawn in `focus` order -- descending document count -- so the most prolific users go down
        # *first* and everyone else lands on top of them. The alternative buries a 40-document user
        # under a 4,531-document one wherever they overlap, which would make the figure a picture
        # of who writes most rather than of who sits where.
        groups = dict(list(coloured.groupby("author_id", sort=False)))
        for index, user in enumerate(focus):
            rows = groups.get(user)
            if rows is not None:
                axes.scatter(rows["x"], rows["y"], s=size, color=palette[index], alpha=0.9,
                             linewidths=edge, edgecolors=SURFACE, zorder=2, rasterized=True)

    axes.set_xlim(*x_limits)
    axes.set_ylim(*y_limits)
    # One scale on both axes. The two UMAP dimensions are the same kind of thing, so stretching one
    # of them to fill a square figure would make a round cluster look elongated -- a claim about the
    # data that comes entirely from the figure's shape.
    axes.set_aspect("equal")
    axes.set_xticks([])
    axes.set_yticks([])
    axes.grid(False)
    for spine in axes.spines.values():
        spine.set_visible(False)
    return figure


# --- entry point -------------------------------------------------------------

def focus_user_count(options, dataset: str) -> int:
    """How many users to colour for ``dataset``, resolving the default and ``all``.

    :data:`ALL_USERS` survives as-is; the caller clamps it against the users the corpus actually
    has, which is the only place that number is known.
    """
    if options.focus_users is None:
        return DEFAULT_FOCUS_USERS.get(dataset, FALLBACK_FOCUS_USERS)
    return options.focus_users


def output_tag(options, dataset: str) -> str:
    """Suffix marking a figure drawn under non-default choices, empty when everything is default.

    Same reasoning as ``plot_results.py``'s directory-name contract: a figure of a different focus
    set, or of a differently tuned projection, is not the same figure and must not overwrite it.
    It matters more here than it does there, because the figure itself carries no text -- the
    filename and the console log are the whole record of how it was made.
    """
    pieces = []
    focus = focus_user_count(options, dataset)
    if focus != DEFAULT_FOCUS_USERS.get(dataset, FALLBACK_FOCUS_USERS):
        pieces.append("focusall" if focus == ALL_USERS else f"focus{focus}")
    if options.max_documents:
        pieces.append(f"docs{options.max_documents}")
    # The field is only drawn when some user is left out of the focus set, so its bin count cannot
    # change a figure that colours everyone and does not belong in that figure's name.
    if options.density_bins != DEFAULT_DENSITY_BINS and focus != ALL_USERS:
        pieces.append(f"bins{options.density_bins}")
    if options.n_neighbors != DEFAULT_N_NEIGHBORS:
        pieces.append(f"nn{options.n_neighbors}")
    if options.min_dist != DEFAULT_MIN_DIST:
        pieces.append(f"dist{options.min_dist:g}")
    if options.metric != DEFAULT_METRIC:
        pieces.append(options.metric)
    if options.seed != DEFAULT_SEED:
        pieces.append(f"seed{options.seed}")
    return "".join(f"_{piece}" for piece in pieces)


def focus_users_argument(value: str) -> int:
    """``--focus-users`` accepts a count or the word ``all``."""
    if value.strip().lower() == "all":
        return ALL_USERS
    count = int(value)
    if count < 0:
        raise argparse.ArgumentTypeError("--focus-users takes a non-negative count or 'all'")
    return count


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--source", choices=DATASETS, nargs="+", default=list(DATASETS),
                        help="dataset(s) to draw; one figure each")
    parser.add_argument("--feature", choices=FEATURES, default=DEFAULT_FEATURE,
                        help="which feature parquet to project")
    parser.add_argument("--defense", choices=DEFENSES, default=NO_DEFENSE,
                        help="project the defended vectors instead of the originals")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR,
                        help="directory holding the feature parquets")
    parser.add_argument("--focus-users", type=focus_users_argument, default=None,
                        help="how many users to draw in colour over the grey field, most prolific "
                             "first; 'all' colours every user and leaves no field, 0 draws the "
                             "field alone. Default is per dataset: "
                             + ", ".join(f"{name}={'all' if n == ALL_USERS else n}"
                                         for name, n in DEFAULT_FOCUS_USERS.items()))
    parser.add_argument("--max-documents", type=int, default=0,
                        help="project a uniform subsample of this many documents instead of the "
                             "whole corpus, for a fast look; 0 projects everything")
    parser.add_argument("--density-bins", type=int, default=DEFAULT_DENSITY_BINS,
                        help="bins per side of the grey density field")
    parser.add_argument("--n-neighbors", type=int, default=DEFAULT_N_NEIGHBORS,
                        help="UMAP neighbourhood size (small = more local structure)")
    parser.add_argument("--min-dist", type=float, default=DEFAULT_MIN_DIST,
                        help="UMAP minimum separation between projected points")
    parser.add_argument("--metric", default=DEFAULT_METRIC,
                        help="distance UMAP compares vectors with; cosine matches the attacks")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="seeds both the document subsample and UMAP itself")
    parser.add_argument("--figsize", type=float, default=10.0,
                        help="figure edge in inches (square)")
    parser.add_argument("--refresh", action="store_true",
                        help="recompute the projection even if a cached one matches")
    parser.add_argument("--png", action="store_true",
                        help="write a PNG beside each PDF")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    options = parse_arguments(argv)
    plot_results.WRITE_PNG = options.png

    for dataset in options.source:
        parquet_path = feature_parquet(dataset, options.defense, options.feature, options.data_dir)
        if not parquet_path.exists():
            print(f"{dataset}: no {parquet_path.name} in {options.data_dir} -- skipped")
            continue
        print(f"{DATASET_LABELS[dataset]} ({options.defense}, {options.feature})")
        frame = document_layout(dataset, options.defense, options.feature, options.data_dir,
                                options)

        # Ties broken by author id, so the focus set is reproducible rather than left to whatever
        # order `value_counts` happened to return.
        counts = frame["author_id"].value_counts()
        ranked = counts.reset_index()
        ranked.columns = ["author_id", "n_documents"]
        ranked = ranked.sort_values(["n_documents", "author_id"], ascending=[False, True])
        requested = focus_user_count(options, dataset)
        n_focus = counts.size if requested == ALL_USERS else min(requested, counts.size)
        focus = list(ranked["author_id"].head(n_focus))

        figure = draw_umap(frame, focus, options)
        path = save_figure(figure, PLOTS_DIR / dataset / "umap"
                           / f"{options.defense}_{options.feature}{output_tag(options, dataset)}")
        # The figure carries no text, so the run's parameters are recorded here and in the
        # filename instead.
        field = ("no field (every user coloured)" if n_focus == counts.size
                 else f"field {options.density_bins}² bins")
        print(f"  {len(frame):,} documents, {counts.size:,} users;"
              f" {len(focus)} in colour ({ranked['n_documents'].head(n_focus).sum():,}"
              f" documents), {field}")
        print(f"  UMAP n_neighbors={options.n_neighbors}, min_dist={options.min_dist},"
              f" metric={options.metric}, seed={options.seed}")
        print(f"  wrote {path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
