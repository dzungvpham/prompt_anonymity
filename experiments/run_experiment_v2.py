#!/usr/bin/env python
"""Rolling-window authorship attribution on the unified prompt dataset.

A sibling of ``run_experiment.py``. The machinery is the same -- vectors in, nearest neighbour,
accuracy out -- but two things about the experiment differ:

1. **Input is the built parquet pair.** ``<split>.parquet`` (documents) joined on ``doc_id`` to
   ``<split>_<feature>.parquet`` (precomputed vectors from
   ``prompt_anonymity/data/compute_features.py``). No
   featurization, no defense, no GPU: this script only reads vectors that already exist.

2. **A grid of rolling chronological windows.** Documents are ordered by ``ended_at`` and cut at
   fractions of that timeline. For a ``--known-fractions`` value *f* and a ``--window`` value *w*,
   the first *f* of the corpus is **known** and the following *w* is **unknown**, so the attacker
   never sees the future. Both flags take lists and are swept against each other; pairs with
   ``f + w > 1`` are skipped rather than truncated. The defaults (``0.25 0.50 0.75`` against
   ``0.10 0.25 0.50``) give eight independent attacks.

   The two axes ask different questions. *More history* (larger *f*) grows the attacker's
   training set and the candidate pool. *A longer window* (larger *w*) grows the number of
   **target** users, because more people show up over a longer stretch of time -- which is why
   there is no random candidate-pool sweep here any more: sweeping the window varies the pool
   with real data instead of resampling one window's users. The trade-off is that the window
   axis is not a clean control -- a longer window also reaches further into the future -- so
   read a falling line as "larger pool *and* more drift", and compare against the per-window
   random baseline rather than across windows.

3. **The candidate set is open**, and by default that is handled by *scoring only what can be
   scored*: an unknown document whose author is absent from the known side cannot be attributed
   to anyone, so the reported tables cover the in-set documents and the rest are counted
   (``n_ood`` / ``ood_rate``) but not scored. Passing ``--ood reject`` instead turns on an
   explicit reject option, adding the class :data:`OOD_LABEL`, "not one of the known authors",
   and scoring every unknown document. See "The reject option" below for why it is opt-in.

The attack
----------
The known side is fully labelled -- it is the attacker's own data -- so the scorer is **trained**
on it rather than being a fixed distance. ``--attacks`` selects one or more from
:data:`prompt_anonymity.attacks.ATTRIBUTION_ATTACKS`, each run against every window; the default
is multinomial logistic regression over the known authors, which measured best by a wide margin.
Unknown documents contribute nothing but their StyloMetrix vectors, and no unknown label is read
anywhere in this pipeline.

Supervision is what makes the attack work. Cosine to an author centroid treats every StyloMetrix
direction as equally informative, but some vary wildly *within* an author (noise) while others
separate authors (signal). With ~124 candidates that mistake is fatal: the best of 124 impostor
distances lands closer than a genuine match, so the nearest centroid is usually the wrong one.
On the 75% window, learning the weighting instead doubles top-1 (0.129 -> 0.258) and lifts
out-of-set AUROC from 0.523 to 0.631.

The reject option (``--ood reject``, off by default)
----------------------------------------------------
Raw scores are not comparable across documents -- a short, generic document scores low against
*every* author -- so the accept/reject decision uses a **cohort-normalised** score: each
document's scores are z-scored across the candidate authors, and the rejection score is the
negated maximum (:func:`prompt_anonymity.attacks.rejection_score`). Worth ~0.09 of DIR@10%.

The threshold is calibrated **from the known side alone** (:func:`calibrate_threshold`), by
simulating the task inside it: enrol 70% of the known authors, treat the rest as never-seen
impostors, and score held-out documents from both. Two things make that simulation usable:

* The two conditional rates -- P(reject | out-of-set) and P(correct and accepted | in-set) --
  are measured separately and only then combined under an assumed out-of-set rate. Mixing them
  at the simulation's own proportion (~60% out-of-set, nothing like reality) is what made an
  earlier version reject 95% of everything.
* That rate is itself forecast from known data by :func:`estimate_ood_prior`, which replays the
  same chronological split *inside* the known window. The unknown window is never inspected.

What gets measured
------------------
Top-k accuracy is three samples of a ranking over 80-125 candidate authors, so the closed-set
tables are backed by the fuller families in :mod:`prompt_anonymity.metrics` (all computed from
the one score matrix, see :func:`closed_set_detail`):

* **Whole ranking** -- ``mrr``, ``median_rank``, ``mean_percentile_rank``, and the complete CMC
  curve. Percentile rank is the only accuracy-like number here that is comparable across
  windows whose candidate pools differ in size, which the sweep guarantees they do.
* **Per user rather than per document** -- ``macro_conv_acc<k>``, ``macro_f1``, and a per-author
  risk table. The document-weighted headline is not wrong, but on this corpus one author owns a
  fifth of a window's documents, and the macro view answers the question a privacy claim needs:
  how exposed is a *typical* user, and how wide is the spread.
* **Retrieval** -- ``map`` / ``mean_r_precision``, running each known author as a query against
  the anonymous documents ("find everything this person wrote"), a different threat model from
  identification and the only direction where MAP is not a restatement of MRR.
* **Open set** (``--ood reject``) -- ``ood_auroc``, ``ood_eer``, ``dir_at_far<pct>``, and PAN's
  ``c_at_1``, plus ``brier`` / ``ece`` for whether the confidence driving every threshold means
  anything.

Run (from the repo root)::

    python experiments/run_experiment_v2.py
    python experiments/run_experiment_v2.py --attacks nearest_neighbor  # the baseline attack
    python experiments/run_experiment_v2.py --attacks logistic cosine nearest_neighbor
    python experiments/run_experiment_v2.py --ood reject                # add the reject option
    python experiments/run_experiment_v2.py --ood reject --ood-calibration far
    # one known fraction, target pool traced finely:
    python experiments/run_experiment_v2.py --known-fractions 0.5 --window 0.05 0.1 0.2 0.3 0.4 0.5
    # compare the three trainable attacks, each tuned per window on that window's known side:
    python experiments/run_experiment_v2.py --attacks logistic svm xgboost --tune

Hyper-parameters (``--tune``)
-----------------------------
Every attack's settings are chosen by :func:`tune_on_known`, which searches **only the known
side** of the window it is about to attack -- the attacker's own labelled data. Nothing in the
search, down to the standardization statistics, is derived from a document it will later be
scored on.

The search is :class:`sklearn.model_selection.HalvingRandomSearchCV`: sample
``--tune-candidates`` configurations from :data:`HYPERPARAMETER_SPACES`, score them on a small
subsample of each fold's training block, discard all but the best ``1/--tune-factor``, triple the
data, repeat. Sampling rather than gridding also allows continuous ranges for ``C`` and the
learning rate, and lets xgboost past 300 trees -- the old grid's maximum, which it selected in
every single window, meaning the grid rather than the data was setting that answer.

**How much this actually saves, measured.** Less than the rung sizes suggest -- 63 minutes
against the grid's ~75 for the same three-attack sweep, at unchanged accuracy -- and the reason is
worth knowing before tuning anything bigger. Halving assumes cost is proportional to the sample,
but these attacks are dominated by the *author count*: on the swe-chat 75% window a logistic fit
on a ninth of the documents costs 0.23 of the full one, not 0.11, because the 124-class softmax
and the 196x124 coefficient matrix are the same size either way (xgboost 0.18, and only the SVM,
whose cost really is superlinear in the sample, gets the full 0.05). So the first rung is the
expensive one, and ``--tune-candidates`` -- not ``--tune-factor`` -- is the dial that matters.
The other resources sklearn can halve on are no better here: ``max_iter`` is not a real budget
because lbfgs converges in 104-215 iterations, well inside the 3,000 it is allowed.

The larger saving is not in the search at all but in how often it runs. The eight windows share
only *three* distinct known sides (one per ``--known-fractions`` value), and for a given known
fraction every window is attacked from the identical documents, labels and standardisation --
so the search is run once per known side and reused (:func:`tuned_settings`), 3 searches instead
of 8. That is what ``--tune-window`` exists for: the inner validation block is a fixed share of
the known side rather than a copy of the outer window, which is what used to make otherwise
identical searches differ.

Outputs (under ``--output-dir``), all carrying ``known_fraction``, ``window`` and ``attack``
columns, so a multi-attack run stays one tidy table per file:

* ``rolling_results.csv`` -- one row per (window, attack), every summary metric.
* ``headline_results.csv`` / ``cmc_results.csv`` -- one row per (window, attack, k).
* ``predictions_<stem>.csv`` and ``author_report_<stem>.csv`` -- per document and per author
  (risk and retrieval, sorted most-exposed first), where ``<stem>`` is
  ``<attack>_known<pct>_window<pct>``.
* Figures: one ``topk_accuracy_<stem>.pdf`` per combination, and per attack
  ``window_sweep_top<k>_<attack>.pdf`` (accuracy against the number of target users, one line
  per known fraction) and ``cmc_curve_<attack>.pdf`` (one curve per window).
* ``--ood reject`` adds ``ood_sweep.csv``, the open-set columns of ``rolling_results.csv``, and
  the per-document decision in ``predictions_*.csv``.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.spatial.distance import cdist
from scipy.stats import loguniform
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.experimental import enable_halving_search_cv  # noqa: F401  (unlocks the import below)
from sklearn.model_selection import HalvingRandomSearchCV

from prompt_anonymity.attacks import ATTRIBUTION_ATTACKS, rejection_score
from prompt_anonymity.evaluation import LinkageRanking, headline_accuracy
from prompt_anonymity.metrics import (
    author_query_metrics,
    c_at_1,
    calibration_metrics,
    cmc_curve,
    detection_auroc,
    detection_identification_rate,
    equal_error_rate,
    macro_f1_score,
    macro_top_k_accuracy,
    max_softmax_confidence,
    selective_classification,
    per_author_ranking,
    random_guessing_accuracy,
    ranking_summary,
    retrieval_summary,
    true_author_ranks,
)
from prompt_anonymity.viz import plot_cmc_curve, plot_headline_topk, plot_window_sweep

REPO_ROOT = Path(__file__).resolve().parent.parent

FEATURE_LABELS = {  # legend name per feature, matching run_experiment.py
    "stylometrix": "StyloMetrix",
    "function_words": "Function Words",
    "character_statistics": "Character Stats",
    "gemini_embedding_001": "Gemini Embedding 001",
    "gemini_embedding_2": "Gemini Embedding 2",
}
DATA_DIR = REPO_ROOT / "data" / "hf"

# source -> split / parquet base name (must match build_dataset.py SPLIT_NAMES).
SPLIT_NAMES = {"wildchat": "wildchat", "swe-chat": "swe_chat"}

# The extra class: "this document's author is not among the known authors". Not a valid
# author_id (those are ``<source>-<16 hex>``), so it can never collide with a real one.
OOD_LABEL = "<OOD>"

# False-alarm rates at which the detection-and-identification rate is reported. DIR@FAR is the
# standard open-set identification summary: of the in-set documents, the fraction both accepted
# and attributed to the right author, at a threshold that wrongly accepts FAR of the OOD ones.
DIR_FAR_POINTS = (0.05, 0.10, 0.20)


# --- input ------------------------------------------------------------------

def _period(frame: pd.DataFrame) -> str:
    """Compact ``first..last`` date range of a set of documents, noting any undated ones."""
    stamps = [stamp for stamp in frame["ended_at"] if stamp]
    if not stamps:
        return "undated"
    label = f"{min(stamps)[:10]}..{max(stamps)[:10]}"
    n_undated = len(frame) - len(stamps)
    return f"{label} +{n_undated} undated" if n_undated else label


def filter_documents(frame: pd.DataFrame, model_owner: str = "all", language: str = "all") -> pd.DataFrame:
    """Restrict the corpus to one agent provider and/or one primary language.

    Applied **before** the timeline is cut, so the window fractions are quartiles of the filtered
    corpus rather than of the whole split -- otherwise a filter would silently shrink the windows
    and leave gaps between them. Both default to ``"all"`` (no filter).
    """
    for column, value in (("model_owner", model_owner), ("language_primary", language)):
        if value and value.lower() != "all":
            before = len(frame)
            frame = frame[frame[column] == value]
            if frame.empty:
                raise SystemExit(f"no documents with {column} == {value!r}.")
            print(f"{column} == {value!r}: kept {len(frame):,} of {before:,} documents "
                  f"({frame['author_id'].nunique():,} authors)")
    return frame


# Document metadata this script reads: the join key, the chronological order, the label, and the
# two filter axes. Everything else in the parquet -- notably ``turns``, the raw conversation text
# -- is deliberately left on disk (see load_documents_and_features).
DOCUMENT_COLUMNS = ("doc_id", "author_id", "ended_at", "language_primary", "model_owner")


def load_documents_and_features(data_dir, source: str, feature: str, undated: str = "drop",
                                model_owner: str = "all",
                                language: str = "all") -> tuple[pd.DataFrame, np.ndarray]:
    """Load one split's documents and its feature matrix, aligned and ordered by ``ended_at``.

    The two parquets are joined on ``doc_id`` (one-to-one), so a feature file that is stale or
    covers only part of the split is a hard error rather than a silent misalignment. Ties in
    ``ended_at`` are broken by ``doc_id`` so the ordering -- and therefore every window boundary
    -- is deterministic.

    Some documents carry no timestamp at all (SWE-chat: ~8%, all from one agent), and a
    chronological experiment has to decide where they go. ``undated="drop"`` (the default)
    removes them, because placing an undated document anywhere on the timeline invents an
    ordering and risks handing the attacker a document that is really from the future;
    ``undated="known"`` instead treats them as the oldest documents, so they are always on the
    known side -- the reading that an attacker holding undated history would get.
    """
    split = SPLIT_NAMES[source]
    documents_path = Path(data_dir) / f"{split}.parquet"
    features_path = Path(data_dir) / f"{split}_{feature}.parquet"
    for path in (documents_path, features_path):
        if not path.exists():
            raise SystemExit(f"{path} not found -- build it first with\n"
                             f"  python -m prompt_anonymity.data.build_dataset\n"
                             f"  python -m prompt_anonymity.data.compute_features --source {source} "
                             f"--feature {feature}")

    # Read only the metadata this script actually uses. ``turns`` holds the raw conversation
    # text and is 98% of the wildchat parquet (1.07 GB of 1.09 GB uncompressed, several times
    # that once pandas materialises it as Python objects) -- and nothing downstream reads it,
    # because featurisation already happened. Skipping it is most of this function's footprint.
    document_schema = pq.ParquetFile(documents_path).schema_arrow.names
    wanted = [column for column in DOCUMENT_COLUMNS if column in document_schema]
    documents = pd.read_parquet(documents_path, columns=wanted)
    features = pd.read_parquet(features_path)
    missing = set(documents["doc_id"]) - set(features["doc_id"])
    if missing:
        raise SystemExit(
            f"{len(missing):,} of {len(documents):,} documents have no {feature} vector. "
            f"Recompute with `python -m prompt_anonymity.data.compute_features --source {source} "
            f"--feature {feature}`."
        )

    # The feature parquet mirrors some document metadata (``author_id``); keep only the columns
    # that are genuinely features, so the merge cannot collide and the matrix stays numeric.
    # Checked against the *full* document schema, not the narrowed frame above, so a feature
    # named after a metadata column we skipped still cannot sneak into the matrix.
    feature_columns = [column for column in features.columns
                       if column != "doc_id" and column not in document_schema]
    merged = documents.merge(features[["doc_id", *feature_columns]], on="doc_id", how="left",
                             validate="one_to_one")
    merged = filter_documents(merged, model_owner, language)

    n_undated = int(merged["ended_at"].isna().sum())
    if n_undated:
        authors = merged.loc[merged["ended_at"].isna(), "author_id"].nunique()
        if undated == "drop":
            merged = merged[merged["ended_at"].notna()]
            print(f"dropped {n_undated:,} undated documents ({authors} authors) -- no ended_at to order by")
        else:
            # "" sorts before every ISO-8601 timestamp, so these become the oldest documents.
            merged["ended_at"] = merged["ended_at"].fillna("")
            print(f"placing {n_undated:,} undated documents ({authors} authors) at the start of the timeline")

    merged = merged.sort_values(["ended_at", "doc_id"], kind="mergesort").reset_index(drop=True)
    return merged, merged[feature_columns].to_numpy(dtype=float)


def rolling_windows(n_documents: int, known_fractions, window_sizes) -> list[tuple[float, float, slice, slice]]:
    """``(fraction, window, known_slice, unknown_slice)`` for every runnable combination.

    The known side is everything up to ``fraction`` of the timeline and the unknown side is the
    next ``window`` of it, so the attacker only ever sees documents that ended before the ones it
    must attribute. Both axes are swept: each ``(fraction, window)`` pair is its own independent
    experiment, not a fold of a partition, and windows overlap by design.

    Widening the window is how this experiment varies the number of *target* users -- a longer
    stretch of the timeline simply contains more people -- which is why there is no separate
    candidate-pool sweep. Read that axis with the caveat that it is not a clean pool-size
    control: a wider window also reaches further into the future, so accuracy falling as the
    window grows mixes a larger target pool with more temporal drift.

    Combinations running off the end of the timeline (``fraction + window > 1``) are skipped
    rather than truncated, so a 50%-known/50%-window point is never silently compared against a
    50%/40% one. Degenerate combinations (an empty side) are skipped the same way, and it is an
    error only if nothing at all survives.
    """
    windows, skipped = [], []
    for fraction in known_fractions:
        for window in window_sizes:
            if fraction + window > 1.0 + 1e-9:
                skipped.append(f"known {fraction:.0%} + window {window:.0%} > 100%")
                continue
            start = int(round(fraction * n_documents))
            end = min(n_documents, int(round((fraction + window) * n_documents)))
            if start < 2 or end <= start:
                skipped.append(f"known {fraction:.0%} + window {window:.0%} leaves an empty side "
                               f"({start} known, {end - start} unknown of {n_documents} documents)")
                continue
            windows.append((fraction, window, slice(0, start), slice(start, end)))
    if skipped:
        print(f"skipping {len(skipped)} window combination(s): " + "; ".join(skipped))
    if not windows:
        raise SystemExit(
            f"no runnable combination of --known-fractions {list(known_fractions)} and "
            f"--window {list(window_sizes)} over {n_documents} documents."
        )
    return windows


def standardize(known: np.ndarray, unknown: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Z-score both sides using **known-side** statistics only.

    StyloMetrix mixes ratios in [0, 1] with occasional raw counts, so without scaling a handful
    of wide-range columns dominate any distance. Fitting the mean/scale on the known side alone
    keeps the unknown documents out of the attacker's view of the data. Zero-variance columns are
    left alone rather than divided by zero.
    """
    center = known.mean(axis=0)
    scale = known.std(axis=0)
    scale = np.where(scale > 0, scale, 1.0)
    return (known - center) / scale, (unknown - center) / scale


# --- open-set attribution ----------------------------------------------------

def build_attack(name: str, args: argparse.Namespace, overrides: dict | None = None):
    """Construct the named attack from :data:`~prompt_anonymity.attacks.ATTRIBUTION_ATTACKS`.

    Returned as a *factory* of fresh, unfitted attacks rather than one instance, because
    threshold calibration refits the same configuration on each simulation fold. Each attack
    picks up whichever of the CLI tuning flags applies to it; the rest are ignored.
    ``overrides`` (from :func:`tune_on_known`) wins over the flags.
    """
    attack = ATTRIBUTION_ATTACKS[name]
    settings = {}
    if name == "logistic":
        settings = {"C": args.regularization,
                    "class_weight": "balanced" if args.balanced else None}
    elif name in ("wccn", "plda"):
        settings = {"shrinkage": args.shrinkage}
    elif name == "nearest_neighbor":
        settings = {"metric": args.metric}
    settings.update(overrides or {})
    return (lambda: attack(**settings)) if settings else attack


# Hyper-parameter search spaces sampled by --tune, in the form
# :class:`sklearn.model_selection.ParameterSampler` accepts: a mapping of name -> (list of choices
# | scipy distribution with ``.rvs``), or a *list* of such mappings when the space is conditional
# (an SVM's ``gamma`` only exists for the RBF kernel, so each kernel is its own sub-space and one
# is chosen uniformly per draw).
#
# These are continuous where the old grids were a handful of points, which is the part of random
# search that is free: a draw from a range costs exactly what a draw from a list costs. What is
# *not* free is how expensive an individual draw can be -- see the xgboost note below -- so these
# ranges are bounded by fit cost rather than by what is plausible. The xgboost range does reach
# past 300 trees, which every window of the earlier grid search picked: a boundary hit that meant
# the grid, not the data, was setting that answer.
# Attacks absent from this mapping have nothing worth tuning and are used as configured.
HYPERPARAMETER_SPACES: dict[str, dict | list[dict]] = {
    # C stops at 5 for a cost reason, not a modelling one: above it lbfgs needs several times more
    # iterations to converge and a single fit dominates the whole search, while every window that
    # has been searched picked a C below 1. Widen it only alongside a higher max_iter.
    "logistic": {"C": loguniform(0.02, 5.0), "class_weight": [None, "balanced"]},
    "svm": [
        {"kernel": ["linear"], "C": loguniform(0.1, 30.0)},
        {"kernel": ["rbf"], "C": loguniform(1.0, 300.0), "gamma": ["scale"]},
        {"kernel": ["rbf"], "C": loguniform(1.0, 300.0), "gamma": loguniform(1e-3, 1e-1)},
    ],
    # An xgboost fit costs roughly depth x trees x documents, so this is the one space where
    # widening is not free -- a first draft reaching depth 8 and 800 trees made the halving search
    # *slower* than the grid it replaced, having spent the saving on individually huge candidates.
    # The floor on learning_rate is what keeps the tree count honest: a slow learner needs many
    # more rounds to pay off, so the cheap way to allow "more trees than the old grid's 300" is to
    # rule out the configurations that would need thousands of them.
    "xgboost": {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [2, 3, 4, 6],
        "learning_rate": loguniform(0.08, 0.5),
        "subsample": [0.7, 0.85, 1.0],
    },
    # RLSC has exactly one knob, and its cost does not grow with the author pool,
    # so a wide range here is free in a way the xgboost space above is not.
    "rlsc": {"alpha": loguniform(1e-3, 1e3)},
    "wccn": {"shrinkage": loguniform(0.01, 0.9)},
    "plda": {"shrinkage": loguniform(0.01, 0.9)},
    "lda": {"n_components": [None, 16, 32, 64, 128]},
}


def inner_known_folds(n_known: int, window: float, n_folds: int = 3) -> list[tuple[slice, slice]]:
    """Chronological train/validate splits **inside** the known side, for hyper-parameter search.

    These are handed to the search as an explicit ``cv`` rather than letting it default to a
    random k-fold, which would be the wrong simulation twice over. A random fold would let the
    model train on a user's later documents to predict their earlier ones, which the real task
    never permits, and it would keep every author on both sides, whereas the real unknown window
    contains authors who are simply new. Replaying the experiment's own chronological cut inside
    the known window reproduces both effects, at the cost of validating on less data.

    Successive halving subsamples the *training* block of each fold to build its early rungs.
    That preserves both properties -- a random subset of documents that all precede the validation
    block still precedes it, and dropping documents can only make the "unseen author" effect
    stronger, not weaker.

    Each fold trains on ``[0, cut)`` and validates on the following ``window`` fraction **of the
    known side**, with the cuts spaced so the last one ends exactly at the end of it. Every row
    used is a known row: nothing here can see the unknown window, which is the property that
    makes any selection made on top of it honest.

    ``window`` is ``--tune-window`` and is deliberately *not* the experiment's outer window. The
    two are in different units anyway -- one is a share of the known side, the other a share of
    the whole timeline -- so tying them together never made the inner simulation match the outer
    task, it only forced a separate search per window. Decoupling them lets the eight window
    combinations share three searches, one per known side.
    """
    folds = []
    for index in range(n_folds):
        # Space the validation blocks so the final one ends at the end of the known side.
        end_fraction = 1.0 - index * window / max(n_folds, 1)
        end = int(round(end_fraction * n_known))
        start = int(round((end_fraction - window) * n_known))
        if start < 2 or end <= start:
            continue
        folds.append((slice(0, start), slice(start, end)))
    return folds


class TunableAttack(BaseEstimator, ClassifierMixin):
    """scikit-learn estimator wrapping one :data:`ATTRIBUTION_ATTACKS` entry, for the tuner.

    The attribution attacks expose ``fit(embeddings, labels)`` / ``score(embeddings)``, where
    ``score`` returns an ``[n_docs x n_authors]`` matrix. That is deliberately *not* the sklearn
    convention -- ``BaseEstimator.score(X, y)`` means "accuracy" -- so a search meta-estimator
    cannot drive them directly. This adapter supplies the missing half of the interface:
    ``predict`` (argmax over authors), ``classes_``, and ``get_params``/``set_params`` over an
    open-ended settings dict, which is what lets :func:`clone` rebuild a candidate.

    Standardization is folded in here rather than applied to the whole known side up front,
    because each cross-validation fold must derive its mean and scale from its own training block
    only. Fitting them on all of the known data would leak the fold's validation documents into
    the fold's own preprocessing -- a mild leak, but the point of this machinery is that there
    are none.
    """

    def __init__(self, attack_name: str, standardize: bool = True, **settings):
        self.attack_name = attack_name
        self.standardize = standardize
        self.settings = settings

    def get_params(self, deep: bool = True) -> dict:
        """Flatten the settings dict into the search space's own parameter names."""
        return {"attack_name": self.attack_name, "standardize": self.standardize, **self.settings}

    def set_params(self, **params):
        """Accept any hyper-parameter name; unrecognised ones are passed to the attack."""
        settings = dict(self.settings)
        for key, value in params.items():
            if key in ("attack_name", "standardize"):
                setattr(self, key, value)
            else:
                settings[key] = value
        self.settings = settings
        return self

    def fit(self, embeddings, labels):
        embeddings = np.asarray(embeddings, dtype=float)
        if self.standardize:
            scale = embeddings.std(axis=0)
            self.center_ = embeddings.mean(axis=0)
            self.scale_ = np.where(scale > 0, scale, 1.0)
            embeddings = (embeddings - self.center_) / self.scale_
        self.model_ = ATTRIBUTION_ATTACKS[self.attack_name](**self.settings).fit(embeddings, labels)
        self.classes_ = np.asarray(self.model_.authors)
        return self

    def predict(self, embeddings):
        """The highest-scoring author per document."""
        embeddings = np.asarray(embeddings, dtype=float)
        if self.standardize:
            embeddings = (embeddings - self.center_) / self.scale_
        return self.classes_[self.model_.score(embeddings).argmax(axis=1)]


def in_set_top_1(estimator: TunableAttack, embeddings: np.ndarray, labels: np.ndarray) -> float:
    """Top-1 accuracy over the validation documents whose author was in the training block.

    The scorer used to select hyper-parameters. Documents by an author the model never saw are
    excluded rather than counted wrong: they are unattributable by construction, so including
    them would add a term that no hyper-parameter can influence and compress the differences the
    search is trying to resolve. Folds with no attributable document score 0 -- the same for
    every candidate, so it cannot change their ordering.
    """
    in_set = np.isin(labels, estimator.classes_)
    if not in_set.any():
        return 0.0
    return float((estimator.predict(embeddings[in_set]) == labels[in_set]).mean())


def halving_budget(n_known: int, n_authors: int, n_candidates: int, factor: int) -> int:
    """Training-set size for the first rung, sized so the **last** rung uses all the known side.

    The rung count is set by the candidate count, not by this budget: halving divides the field by
    ``factor`` each round, so ``n_candidates`` candidates survive ``1 + log_factor(n_candidates)``
    rounds and then stop, whatever budget they started from. Starting too low therefore does not
    buy an extra round -- it just means the search runs out of candidates early and its winner is
    chosen, and reported, on a fraction of the data. Anchoring to the last rung instead makes the
    winner's ``known_cv_top1`` a full-size number, comparable across searches and against a
    non-halving baseline.

    (Getting this wrong is not loud: at six candidates and factor three an earlier version ran two
    rounds ending at a *third* of the known side, which cost only ~0.006 of measured top-1 but
    moved every reported known-side CV score down by a uniform ~0.055, purely because they were
    measured on less data.)

    The one floor: a rung with fewer than ~2 documents per author is not the task being tuned for,
    and every candidate scores near zero there.
    """
    rungs = 1
    while factor ** rungs <= max(n_candidates, 1):
        rungs += 1
    if rungs == 1:                       # fewer candidates than `factor`: one full-size round
        return n_known
    return int(min(max(n_known // factor ** (rungs - 1), 2 * n_authors),
                   max(n_known // factor, 1)))


def tune_on_known(name: str, embeddings: np.ndarray, labels: np.ndarray,
                  args: argparse.Namespace) -> tuple[dict, pd.DataFrame]:
    """Pick ``name``'s hyper-parameters using known documents only, and say what it picked.

    Runs :class:`sklearn.model_selection.HalvingRandomSearchCV` over
    :data:`HYPERPARAMETER_SPACES` with the chronological folds from :func:`inner_known_folds` and
    :func:`in_set_top_1` as the criterion: start ``--tune-candidates`` candidates on a small
    subsample of each fold's training block, keep the best ``1/factor``, triple the training data,
    repeat, so a configuration only reaches a full-size fit if it has already outscored two thirds
    of the field.

    The budget being subsampled is the training block, never the validation block's position: a
    fold still trains only on documents that precede the ones it is scored on. The subsample is
    drawn once per rung from a fixed seed, so all candidates at a rung see identical data.

    This runs **once per known side** -- see :func:`tuned_settings`, which memoises it across the
    windows that share one. It must not be run once for the whole experiment: a value tuned on the
    75% known side and reused at 25% would have been selected on documents that are the *unknown*
    side of the shorter window, which is exactly the leak the per-known-side search exists to
    avoid. Sharing *within* one known fraction is safe because every such window is attacked from
    the identical labelled set.

    **What it costs, measured.** Against the exhaustive grid this replaced -- same eight windows,
    same folds, same criterion -- it is roughly a wash for a fifth less wall clock (63 vs ~75
    minutes for ``--attacks logistic svm xgboost --tune``). Mean unknown-side top-1 moved by
    -0.001 over the 24 runs, with halving ahead in 9 of them: logistic 0.265 -> 0.254, svm
    0.279 -> 0.291, xgboost 0.311 -> 0.308. On its own known-side criterion the selected
    configuration scored -0.010 (logistic), -0.006 (svm) and +0.003 (xgboost) against the grid's
    best, so the cheaper search is choosing about as well, and the wider continuous ranges make
    up most of what the coarser search loses.

    The failure mode to watch for is a first rung that ranks candidates differently from a
    full-size fit -- an RBF SVM that beats every linear one on all the data can sit below them on
    a third of it, and gets eliminated before it is ever fitted properly. Raise
    ``--tune-candidates``, or drop ``--tune-factor`` to 2, for a larger first rung when a search
    returns something implausible.

    Returns ``(best_settings, trials)``, where ``trials`` is every (candidate, rung) pair that was
    evaluated, so a run's tuning is auditable rather than a hidden choice. A candidate whose fit
    raises scores 0 and is eliminated (sklearn emits a ``FitFailedWarning`` naming it) rather than
    aborting the run or, worse, ranking first as a NaN.
    """
    space = HYPERPARAMETER_SPACES.get(name)
    folds = inner_known_folds(len(labels), args.tune_window, args.tune_folds)
    if not space or not folds:
        return {}, pd.DataFrame()

    factor = max(args.tune_factor, 2)
    with warnings.catch_warnings():
        # Early rungs train on a subsample, so some authors arrive with a single document and the
        # shrinkage estimators say so once per author per fit -- thousands of lines that mean
        # "this rung is small", which is the design. Nothing else is silenced.
        warnings.filterwarnings("ignore", message="Only one sample available")
        search = _halving_search(name, space, folds, factor, embeddings, labels, args)

    # One row per (candidate, rung), so the CSV shows both what was tried and where each
    # candidate was cut. `train_budget` is the rung's resource level as a share of the whole
    # known side; each fold trains on that same share of *its* (shorter) training block.
    trials = pd.DataFrame(search.cv_results_["params"]).drop(
        columns=["attack_name", "standardize"], errors="ignore")
    trials["known_cv_top1"] = search.cv_results_["mean_test_score"]
    trials["known_cv_std"] = search.cv_results_["std_test_score"]
    trials["rung"] = search.cv_results_["iter"]
    trials["train_budget"] = search.cv_results_["n_resources"]
    trials["selected"] = np.arange(len(trials)) == search.best_index_
    best = {key: value for key, value in search.best_params_.items()
            if key not in ("attack_name", "standardize")}
    return best, trials


def tuned_settings(name: str, fraction: float, embeddings: np.ndarray, labels: np.ndarray,
                   args: argparse.Namespace, cache: dict) -> tuple[dict, pd.DataFrame]:
    """:func:`tune_on_known`, run at most once per ``(attack, known fraction)``.

    The experiment sweeps ``--known-fractions`` against ``--window``, but only the first of those
    changes what the attacker holds: for a given known fraction, every window is attacked from the
    same documents, the same labels and the same standardisation. Searching once per known side
    rather than once per pair therefore returns identical settings for a third of the work -- on
    the default 3x3 sweep, 3 searches instead of 8.

    ``cache`` is owned by the caller (:func:`main`) so that its lifetime is one experiment and the
    sharing is visible in the driver rather than hidden in module state. The trials table is
    returned only on a miss, so ``tuning_trials.csv`` holds one block per search instead of the
    same rows repeated once per window.
    """
    key = (name, fraction)
    if key in cache:
        return cache[key], pd.DataFrame()
    settings, trials = tune_on_known(name, embeddings, labels, args)
    cache[key] = settings
    return settings, trials


def _halving_search(name: str, space, folds, factor: int, embeddings: np.ndarray,
                    labels: np.ndarray, args: argparse.Namespace) -> HalvingRandomSearchCV:
    """The fitted search behind :func:`tune_on_known`; split out only to keep that one readable."""
    return HalvingRandomSearchCV(
        TunableAttack(attack_name=name, standardize=args.standardize),
        space,
        n_candidates=args.tune_candidates,
        factor=factor,
        resource="n_samples",
        min_resources=halving_budget(len(labels), len(np.unique(labels)),
                                     args.tune_candidates, factor),
        cv=[(np.arange(train.start, train.stop), np.arange(validate.start, validate.stop))
            for train, validate in folds],
        scoring=in_set_top_1,
        refit=False,           # only the winning settings are wanted; run_window does the real fit
        return_train_score=False,
        error_score=0.0,       # an infeasible candidate loses; it does not abort or win as NaN
        random_state=args.seed,
        n_jobs=args.tune_jobs,
    ).fit(embeddings, labels)


def open_set_folds(labels: np.ndarray, held_out_fraction: float = 0.3, n_folds: int = 3,
                   query_fraction: float = 0.3, seed: int = 47) -> list[tuple[np.ndarray, np.ndarray]]:
    """``(train_rows, query_rows)`` per fold, simulating the open-set task inside the known side.

    The attacker cannot look at the unknown window, but it *can* hold out part of its own
    labelled data and rehearse the decision it is about to make. Each fold enrols
    ``1 - held_out_fraction`` of the known authors and treats the rest as never-seen impostors;
    queries are a sample of each enrolled author's documents (excluded from training, or the
    genuine scores would be optimistic) plus *every* document of the impostor authors. This is
    the only signal available for choosing a threshold, and it uses known labels only.
    """
    authors = np.unique(labels)
    folds = []
    for fold in range(n_folds):
        rng = np.random.default_rng(seed + fold)
        shuffled = rng.permutation(authors)
        n_held = max(1, int(round(held_out_fraction * len(authors))))
        impostors, enrolled = shuffled[:n_held], shuffled[n_held:]

        query_rows = []
        for author in enrolled:
            rows = np.flatnonzero(labels == author)
            if len(rows) < 2:
                continue
            n_query = max(1, int(round(query_fraction * len(rows))))
            query_rows.extend(rng.permutation(rows)[:n_query].tolist())
        impostor_rows = np.flatnonzero(np.isin(labels, impostors))
        if not query_rows or len(impostor_rows) == 0:
            continue
        queries = np.concatenate([np.array(query_rows, dtype=int), impostor_rows])
        train = np.ones(len(labels), dtype=bool)
        train[queries] = False
        if len(np.unique(labels[train])) < 2:
            continue
        folds.append((np.flatnonzero(train), queries))
    if not folds:
        raise SystemExit("open-set simulation failed: too few known authors with two documents.")
    return folds


def estimate_ood_prior(known_labels: np.ndarray, known_fraction: float, window: float) -> float:
    """Forecast the unknown window's out-of-set rate using known documents only.

    The threshold depends on how *common* new authors are, and the simulation in
    :func:`open_set_folds` cannot supply that: it holds out a third of the authors and keeps all
    of their documents, which implies an out-of-set rate near 60% -- nothing like reality. Left
    uncorrected, that pushes the threshold toward rejecting almost everything.

    The rate is forecastable, though, without touching the unknown side: replay the same
    chronological split *inside* the known window and measure how many documents in its final
    slice came from authors absent earlier. On swe-chat this lands at 0.13-0.36 against a true
    0.18-0.30 -- imperfect, but far closer than the simulation's implicit prior, and it is exactly
    the kind of estimate a real attacker could make from their own history.
    """
    known_labels = np.asarray(known_labels)  # already in chronological order
    cut = int(round(len(known_labels) * known_fraction / (known_fraction + window)))
    if cut < 1 or cut >= len(known_labels):
        return float("nan")
    return float((~np.isin(known_labels[cut:], np.unique(known_labels[:cut]))).mean())


def calibrate_threshold(factory, embeddings: np.ndarray, labels: np.ndarray, folds,
                        calibration: str = "accuracy", target_far: float = 0.10,
                        cohort_normalize: bool = True, ood_prior: float = 0.2) -> tuple[float, dict]:
    """Choose the accept/reject threshold from the known-side simulation in :func:`open_set_folds`.

    Each fold yields a labelled genuine/impostor sample of rejection scores drawn entirely from
    known data, and the threshold is read off it two ways:

    * ``calibration="accuracy"`` -- the threshold maximising expected open-set accuracy under an
      assumed out-of-set rate ``ood_prior``. The two conditional rates are measured separately
      on the simulation and only *then* combined with the prior::

          objective(tau) = prior * P(reject | out-of-set)
                         + (1 - prior) * P(correct and accepted | in-set)

      Mixing them at the simulation's own out-of-set proportion instead -- which is near 60% --
      is what made earlier versions of this reject 95% of everything.
    * ``calibration="far"`` -- the threshold that wrongly accepts ``target_far`` of the simulated
      impostors. Fixes the operating point rather than the loss, which is the right choice when
      the cost of a false accept is what matters rather than raw accuracy.

    Returns ``(threshold, diagnostics)``; the diagnostics carry the simulated AUROC and top-1,
    which say how much to trust the threshold *before* any unknown document is touched.
    """
    thresholds, simulated_auroc, simulated_top1 = [], [], []
    for train_rows, query_rows in folds:
        fitted = factory().fit(embeddings[train_rows], labels[train_rows])
        scores = fitted.score(embeddings[query_rows])
        accept_score = rejection_score(scores, cohort_normalize)
        query_labels = labels[query_rows]
        is_ood = ~np.isin(query_labels, fitted.authors)
        correct = (fitted.authors[scores.argmax(axis=1)] == query_labels) & ~is_ood

        simulated_auroc.append(detection_auroc(accept_score, is_ood))
        simulated_top1.append(float(correct[~is_ood].mean()) if (~is_ood).any() else float("nan"))
        if calibration == "far":
            thresholds.append(float(np.percentile(accept_score[is_ood], target_far * 100)))
        else:
            grid = np.unique(np.percentile(accept_score, np.arange(1, 100)))
            rejected_ood = (accept_score[is_ood][:, None] > grid[None, :]).mean(axis=0)
            hit_in_set = (correct[~is_ood][:, None] & (accept_score[~is_ood][:, None] <= grid[None, :])
                          ).mean(axis=0)
            objective = ood_prior * rejected_ood + (1 - ood_prior) * hit_in_set
            thresholds.append(float(grid[int(np.argmax(objective))]))

    return float(np.mean(thresholds)), {
        "calibration_auroc": float(np.mean(simulated_auroc)),
        "calibration_top1": float(np.nanmean(simulated_top1)),
        "calibration_ood_prior": ood_prior,
    }


def open_set_metrics(predicted: np.ndarray, unknown_labels: np.ndarray, known_authors: np.ndarray,
                     nearest_author: np.ndarray, accept_score: np.ndarray) -> dict:
    """Score one window's open-set decision.

    A prediction is correct when it names the true author (for a document whose author is known)
    or rejects (for one whose author is not). ``closed_set_top1`` is the ceiling the same
    centroid ranking would reach with the reject option switched off, and the two baselines
    bracket every trivial strategy: reject everything, or never reject.

    ``c_at_1`` scores the same decision the way the PAN verification tasks do, restricted to the
    in-set documents: rejecting one of those is an abstention rather than an answer, and c@1
    credits it with the attack's own average accuracy instead of counting it as an error. It is
    the metric that says whether the reject option is abstaining on the documents it would have
    got *wrong* -- if it rises above ``in_set_accuracy`` the threshold is doing useful work.
    """
    is_ood = ~np.isin(unknown_labels, known_authors)
    accepted = predicted != OOD_LABEL
    correct = np.where(is_ood, ~accepted, predicted == unknown_labels)
    correctly_ranked = nearest_author == unknown_labels
    n_in_set, n_ood = int((~is_ood).sum()), int(is_ood.sum())

    metrics = {
        "n_in_set": n_in_set,
        "n_ood": n_ood,
        "ood_rate": n_ood / len(unknown_labels),
        "open_set_accuracy": float(correct.mean()),
        "in_set_accuracy": float((predicted == unknown_labels)[~is_ood].mean()) if n_in_set else float("nan"),
        "c_at_1": c_at_1(correctly_ranked[~is_ood], accepted[~is_ood]) if n_in_set else float("nan"),
        "ood_recall": float((~accepted)[is_ood].mean()) if n_ood else float("nan"),
        "ood_precision": float(is_ood[~accepted].mean()) if (~accepted).any() else float("nan"),
        "rejection_rate": float((~accepted).mean()),
        "closed_set_top1": float(correctly_ranked[~is_ood].mean()) if n_in_set else float("nan"),
        "ood_auroc": detection_auroc(accept_score, is_ood),
        # Trivial strategies this has to beat to mean anything.
        "baseline_reject_all": n_ood / len(unknown_labels),
        "baseline_accept_all": float(correctly_ranked.mean()),
    }
    for far in DIR_FAR_POINTS:
        metrics[f"dir_at_far{int(far * 100)}"] = detection_identification_rate(
            accept_score, correctly_ranked, is_ood, far
        )
    return metrics


def ood_threshold_sweep(accept_score: np.ndarray, nearest_author: np.ndarray,
                        unknown_labels: np.ndarray, known_authors: np.ndarray,
                        percentiles=tuple(range(1, 100))) -> pd.DataFrame:
    """Re-score the window across the full range of thresholds.

    The calibrated threshold is a single point on this curve; the sweep shows the whole
    accept/reject trade-off, which is the honest way to read an open-set result -- especially
    when the calibrated point sits on top of a trivial baseline. Candidate thresholds are
    percentiles of the observed accept scores, so the grid spans exactly the range where
    decisions change.
    """
    rows = []
    for percentile in percentiles:
        threshold = float(np.percentile(accept_score, percentile))
        predicted = np.where(accept_score <= threshold, nearest_author, OOD_LABEL)
        metrics = open_set_metrics(predicted, unknown_labels, known_authors, nearest_author, accept_score)
        rows.append({"score_percentile": percentile, "threshold": threshold,
                     **{key: metrics[key] for key in
                        ("open_set_accuracy", "in_set_accuracy", "c_at_1", "ood_recall",
                         "ood_precision", "rejection_rate")}})
    return pd.DataFrame(rows)


# --- separability diagnostic (known side only) ------------------------------

def within_author_mean(embeddings: np.ndarray, labels: np.ndarray, metric: str) -> float:
    """Mean distance between known documents that *share* an author.

    Authors with a single known document contribute nothing (they have no sibling to compare
    against). Reported next to :func:`between_author_mean` purely as a diagnostic: if the two are
    equal the features carry no author signal and every accuracy below is noise.
    """
    pairs = []
    for author in np.unique(labels):
        rows = np.flatnonzero(labels == author)
        if len(rows) < 2:
            continue
        distances = cdist(embeddings[rows], embeddings[rows], metric=metric)
        pairs.append(distances[np.triu_indices(len(rows), k=1)])
    return float(np.concatenate(pairs).mean()) if pairs else float("nan")


def between_author_mean(embeddings: np.ndarray, labels: np.ndarray, metric: str,
                        sample: int = 2000, seed: int = 47) -> float:
    """Mean distance between known documents by *different* authors, for context.

    The other half of the separability diagnostic. Sampled (the full matrix is quadratic and this
    number needs no precision).
    """
    rng = np.random.default_rng(seed)
    rows = rng.choice(len(labels), size=min(sample, len(labels)), replace=False)
    distances = cdist(embeddings[rows], embeddings[rows], metric=metric)
    different = labels[rows][:, None] != labels[rows][None, :]
    return float(distances[different].mean()) if different.any() else float("nan")


# --- attack ------------------------------------------------------------------

def closed_set_table(distances: np.ndarray, known_labels, unknown_labels, args: argparse.Namespace):
    """The standard top-k table from ``run_experiment.py``, on this window's scored documents.

    ``distances`` must already be restricted to unknown documents whose author appears among the
    known authors: top-k ranking is only defined when the true author is in the candidate pool
    (and :class:`LinkageRanking` requires it).

    The random baselines are computed over the **known** authors, which is the pool this attack
    actually searches: every known author is a column in the ranking, and only the denominator
    of ``id_acc`` is restricted to the identities that happen to appear in the window. The
    attacker does not know which of them will show up (in a typical window only a third do), so
    a guesser handed that list would be strictly better informed than the attack it is
    benchmarking.

    * ``random_id`` / ``advantage`` -- identity level, a guesser naming ``k`` of the known
      authors per document. This **overrides** the column :func:`headline_accuracy` computes,
      which guesses among the target identities only; ``run_experiment.py`` keeps that original
      definition, so the two scripts' ``random_id`` columns are not interchangeable.
    * ``random_conv`` / ``advantage_conv`` -- the same pool at the conversation level, i.e. the
      chance of getting one specific document right. Pairs with ``conv_acc``.

    With ~20 unknown documents per author, the identity-level baseline is far higher than the
    conversation-level one (a random guesser gets ~20 attempts per author), so the two levels
    are only comparable against their own baselines. Both baselines move as the window widens,
    which is the whole point of sweeping it: only the gap to the baseline is comparable across
    windows.
    """
    ranking = LinkageRanking(distances, known_labels, unknown_labels)
    headline = headline_accuracy(ranking, top_ks=tuple(args.top_ks))
    n_known_authors = len(np.unique(known_labels))
    headline["n_known_authors"] = n_known_authors
    headline["random_id"] = [
        random_guessing_accuracy(unknown_labels, int(k), n_candidates=n_known_authors)
        for k in headline["top"]
    ]
    headline["advantage"] = headline["id_acc"] - headline["random_id"]
    headline["random_conv"] = [min(int(k), n_known_authors) / n_known_authors for k in headline["top"]]
    headline["advantage_conv"] = headline["conv_acc"] - headline["random_conv"]
    return headline


# --- driver -----------------------------------------------------------------

def run_window(frame: pd.DataFrame, embeddings: np.ndarray, fraction: float, window: float,
               attack: str, known: slice, unknown: slice, args: argparse.Namespace,
               tuning_cache: dict):
    """Run one (window, attack) combination end to end: calibrate, attribute, score.

    Returns ``(scores, predictions, ood_sweep, headline, cmc, author_report, trials)`` -- a
    one-row summary of the window, the per-document decisions, the accept/reject trade-off curve,
    the closed-set top-k table, the full CMC curve, the per-author risk/retrieval breakdown, and
    the hyper-parameter search (empty unless this call was the one that ran it). With ``--ood
    none`` (the default) the reject option is skipped, ``ood_sweep`` is ``None`` and the summary
    covers only the in-set documents.

    ``tuning_cache`` is passed through to :func:`tuned_settings`; see there for why sharing a
    search between windows of the same known fraction is sound.
    """
    known_frame, unknown_frame = frame.iloc[known], frame.iloc[unknown]
    known_embeddings, unknown_embeddings = embeddings[known], embeddings[unknown]
    if args.standardize:
        known_embeddings, unknown_embeddings = standardize(known_embeddings, unknown_embeddings)
    known_labels = known_frame["author_id"].to_numpy()
    unknown_labels = unknown_frame["author_id"].to_numpy()
    known_authors = np.unique(known_labels)

    in_set = np.isin(unknown_labels, known_authors)
    if not in_set.any():
        raise SystemExit(
            f"known fraction {fraction}: none of the {len(unknown_labels):,} unknown documents "
            f"has an author on the known side, so nothing can be scored."
        )

    # Optionally pick this attack's hyper-parameters, using this window's known side only, and
    # reusing the search across windows that share it. Deliberately before anything touches
    # unknown_embeddings.
    settings, trials = ({}, pd.DataFrame())
    if args.tune:
        settings, trials = tuned_settings(attack, fraction, known_embeddings, known_labels,
                                          args, tuning_cache)

    # Fit the attack on the known side (documents + labels, all of which the attacker holds) and
    # score every unknown document against every known author.
    factory = build_attack(attack, args, settings)
    fitted = factory().fit(known_embeddings, known_labels)
    author_scores = fitted.score(unknown_embeddings)          # (n_unknown, n_known_authors)
    best = author_scores.argmax(axis=1)
    predicted_author = fitted.authors[best]
    normalize = not args.ood_raw_distance
    accept_score = rejection_score(author_scores, normalize)  # higher = more out-of-set

    scores = {
        "known_fraction": fraction,
        "window": window,
        "attack": attack,
        # Rounded because the search now samples continuous ranges: an unrounded C prints 17
        # digits of a number whose third one is noise. tuning_trials.csv keeps the exact value.
        "hyperparameters": ", ".join(
            f"{key}={value:.4g}" if isinstance(value, float) else f"{key}={value}"
            for key, value in settings.items()) or "default",
        "known_period": _period(known_frame),
        "unknown_period": _period(unknown_frame),
        "n_known_docs": len(known_labels),
        "n_unknown_docs": len(unknown_labels),
        "n_known_authors": len(known_authors),
        "n_unknown_authors": len(np.unique(unknown_labels)),
        "within_author_mean": within_author_mean(known_embeddings, known_labels, args.metric),
        "between_author_mean": between_author_mean(known_embeddings, known_labels, args.metric, seed=args.seed),
        "random_top1_known_pool": 1.0 / len(known_authors),
    }
    predictions = pd.DataFrame({
        "doc_id": unknown_frame["doc_id"].to_numpy(),
        "true_author": unknown_labels,
        "author_in_known": in_set,
        "best_author": predicted_author,
        "accept_score": accept_score,
    })
    ood_sweep = None

    if args.ood != "none":
        folds = open_set_folds(known_labels, held_out_fraction=args.ood_holdout,
                               n_folds=args.ood_folds, seed=args.seed)
        prior = (estimate_ood_prior(known_labels, fraction, window)
                 if args.ood_prior is None else args.ood_prior)
        threshold, diagnostics = calibrate_threshold(
            factory, known_embeddings, known_labels, folds,
            calibration=args.ood_calibration, target_far=args.ood_target_far,
            cohort_normalize=normalize, ood_prior=prior,
        )
        predicted = np.where(accept_score <= threshold, predicted_author, OOD_LABEL)
        scores.update(open_set_metrics(predicted, unknown_labels, known_authors,
                                       predicted_author, accept_score))
        # Threshold-free companion to ood_auroc: the operating point where letting a stranger
        # through and discarding a known user are equally likely, so it needs no cost assumption.
        ood_eer, eer_threshold = equal_error_rate(accept_score, ~in_set)
        scores.update({"threshold": threshold, "ood_calibration": args.ood_calibration,
                       "ood_eer": ood_eer, "ood_eer_threshold": eer_threshold, **diagnostics})
        predictions = predictions.assign(
            predicted=predicted,
            correct=np.where(in_set, predicted == unknown_labels, predicted == OOD_LABEL),
        )
        ood_sweep = ood_threshold_sweep(accept_score, predicted_author, unknown_labels, known_authors)
        ood_sweep.insert(0, "attack", attack)
        ood_sweep.insert(0, "window", window)
        ood_sweep.insert(0, "known_fraction", fraction)
    else:
        scores.update({
            "n_in_set": int(in_set.sum()),
            "n_ood": int((~in_set).sum()),
            "ood_rate": float((~in_set).mean()),
            "closed_set_top1": float((predicted_author == unknown_labels)[in_set].mean()),
        })

    # Closed-set top-k over the in-set documents. One column per known author (rather than per
    # known document), so `conv_acc` is "true author within the top k authors" -- exactly the
    # document-level top-k of this attack -- and `id_acc` is the identity-level version.
    headline = closed_set_table(-author_scores[in_set], fitted.authors, unknown_labels[in_set], args)
    headline["n_in_set_docs"] = int(in_set.sum())

    in_set_scores, in_set_labels = author_scores[in_set], unknown_labels[in_set]
    cmc, author_report = closed_set_detail(in_set_scores, fitted.authors, in_set_labels, args, scores)

    # Every per-window table is stamped with the three axes it belongs to, in the same order, so
    # the concatenated CSVs can be grouped or filtered on any of them.
    for table in (headline, cmc, author_report):
        table.insert(0, "attack", attack)
        table.insert(0, "window", window)
        table.insert(0, "known_fraction", fraction)
    # The search is not a per-window table -- it belongs to the known side, which is why it has no
    # `window` column and why it is empty on every window after the first that shares one.
    if not trials.empty:
        trials.insert(0, "attack", attack)
        trials.insert(0, "known_fraction", fraction)
    return scores, predictions, ood_sweep, headline, cmc, author_report, trials


def closed_set_detail(scores_matrix: np.ndarray, authors: np.ndarray, true_authors: np.ndarray,
                      args: argparse.Namespace, scores: dict):
    """Everything the three-row headline table leaves on the table, from the same score matrix.

    Fills ``scores`` in place with three families of summary numbers and returns the two tables
    that back them:

    * **Whole-ranking** (:func:`~prompt_anonymity.metrics.ranking_summary`) -- ``mrr``,
      ``mean_percentile_rank``, ``median_rank``. The headline samples the ranking at three
      cutoffs; these use all of it, and ``mean_percentile_rank`` is the only accuracy-like
      number here that is comparable across windows whose candidate pools differ in size.
      Returned in full as the CMC curve, top-k accuracy at every k.
    * **Author-averaged** -- ``macro_conv_acc<k>`` and ``macro_f1``. The headline's ``conv_acc``
      is document-weighted, so a user who writes a fifth of the corpus can carry it on their
      own; these weight every user equally. The per-author table behind them is the risk
      distribution, which is what a privacy claim should actually rest on.
    * **Retrieval** (:mod:`~prompt_anonymity.metrics.retrieval`) -- ``map`` and
      ``mean_r_precision``, running each known author as a *query* against the anonymous
      documents. A different attacker: "find everything this person wrote" rather than "who
      wrote this". Also the only direction in which MAP is not just MRR under another name.

    Calibration of the top-1 confidence (``brier``, ``ece``) goes in too, since every threshold
    in the open-set path assumes that confidence means something.
    """
    n_candidates = len(authors)
    ranks = true_author_ranks(scores_matrix, authors, true_authors)
    predicted = authors[scores_matrix.argmax(axis=1)]

    scores.update(ranking_summary(ranks, n_candidates))
    scores["macro_f1"] = macro_f1_score(true_authors, predicted)
    for k in args.top_ks:
        scores[f"macro_conv_acc{k}"] = macro_top_k_accuracy(ranks, true_authors, k)

    author_report = per_author_ranking(ranks, true_authors, top_ks=tuple(args.top_ks))
    retrieval = author_query_metrics(scores_matrix, authors, true_authors)
    scores.update(retrieval_summary(retrieval))
    author_report = author_report.merge(
        retrieval[["author", "average_precision", "r_precision", "random_average_precision"]],
        on="author", how="left",
    )

    confidence = max_softmax_confidence(scores_matrix)
    calibration = calibration_metrics(confidence, predicted == true_authors)
    scores.update({"brier": calibration["brier"],
                   "ece": calibration["expected_calibration_error"],
                   "mean_confidence": calibration["mean_confidence"]})

    # Selective classification: the attack answers only its most confident documents. Narayanan
    # et al. reported this, not top-1, as the real measure of their 100,000-author attack -- 20%
    # accuracy became >80% precision once it was allowed to pick its battles. Reported at 10/25/50%
    # coverage so a low headline accuracy that hides a confidently-correct subset is visible.
    selective = selective_classification(confidence, predicted == true_authors)
    for _, row in selective.iterrows():
        if row["coverage"] < 1.0:
            scores[f"precision_at_{int(row['coverage'] * 100)}pct"] = row["precision"]
    scores["recall_at_50pct"] = float(
        selective.loc[selective["coverage"] == 0.50, "recall"].iloc[0]
    )
    return cmc_curve(ranks, n_candidates), author_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", default="swe-chat", choices=sorted(SPLIT_NAMES),
                        help="Which built split to attack (default: swe-chat).")
    parser.add_argument("--feature", default="stylometrix",
                        help="Feature parquet to use, i.e. <split>_<feature>.parquet (default: stylometrix).")
    parser.add_argument("--data-dir", default=str(DATA_DIR), help="Directory holding the built parquets.")
    parser.add_argument("--metric", default="cosine",
                        help="Distance metric (any scipy cdist metric; default: cosine).")
    parser.add_argument("--known-fractions", type=float, nargs="+", default=[0.25, 0.50, 0.75],
                        help="Fraction of the timeline that is known (default: 0.25 0.5 0.75). "
                             "Swept against --window; every runnable pair is one experiment.")
    parser.add_argument("--window", type=float, nargs="+", default=[0.10, 0.25, 0.50],
                        help="Fraction of the timeline used as the unknown side, one run per "
                             "value (default: 0.1 0.25 0.5). This is the experiment's target-pool "
                             "axis -- a longer window contains more users -- so it replaces the "
                             "old random candidate-pool sweep. Pairs with --known-fractions "
                             "summing past 1.0 are skipped.")
    parser.add_argument("--model-owner", default="all",
                        help="Restrict to one agent provider, e.g. 'Anthropic' (default: all).")
    parser.add_argument("--language", default="all",
                        help="Restrict to one language_primary, e.g. 'English' (default: all).")
    parser.add_argument("--undated", default="drop", choices=["drop", "known"],
                        help="What to do with documents that have no ended_at: 'drop' them "
                             "(default) or treat them as the oldest, i.e. always known.")
    parser.add_argument("--standardize", action=argparse.BooleanOptionalAction, default=True,
                        help="Z-score features using known-side statistics before scoring "
                             "(default: on; disable with --no-standardize). StyloMetrix mixes "
                             "ratios in [0, 1] with raw counts, so without scaling a handful of "
                             "wide-range columns dominate every method. Measured on swe-chat: "
                             "centroid top-1 0.055 unscaled vs 0.136 standardized, and "
                             "nearest-neighbour top-1 0.092 vs 0.154.")
    parser.add_argument("--attacks", nargs="+", default=["logistic"],
                        choices=sorted(ATTRIBUTION_ATTACKS),
                        help="Attack(s) fitted on the known side, from "
                             "prompt_anonymity.attacks.ATTRIBUTION_ATTACKS; every window runs "
                             "each of them (default: logistic, the best measured on swe-chat -- "
                             "see that module for the comparison). 'cosine' is the unsupervised "
                             "centroid baseline and 'nearest_neighbor' the original attack.")
    parser.add_argument("--regularization", type=float, default=1.0,
                        help="Inverse regularisation strength C for the logistic attack (default: 1).")
    parser.add_argument("--balanced", action="store_true",
                        help="Weight known authors equally in the logistic attack. Off by "
                             "default: the document counts are genuinely informative priors, and "
                             "leaving them in measured better (top-1 0.258 vs 0.252).")
    parser.add_argument("--shrinkage", type=float, default=0.2,
                        help="Covariance shrinkage for the wccn / plda attacks (default: 0.2).")
    parser.add_argument("--tune", action="store_true",
                        help="Search each attack's hyper-parameter space (HYPERPARAMETER_SPACES) "
                             "before scoring, by successive halving over randomly sampled "
                             "configurations. The search runs on chronological folds *inside the "
                             "known side*, so it never sees the documents it will be evaluated "
                             "on. It runs once per (attack, known fraction) and is shared by the "
                             "windows that start from that known side; it is never shared across "
                             "known fractions, because the 75%% known side contains the 25%% "
                             "one's unknown documents. Overrides --regularization / --balanced / "
                             "--shrinkage. Writes every (candidate, rung) pair to "
                             "tuning_trials.csv, one block per search.")
    parser.add_argument("--tune-folds", type=int, default=3,
                        help="Chronological folds inside the known side per --tune search "
                             "(default: 3).")
    parser.add_argument("--tune-window", type=float, default=0.25,
                        help="Validation block of each tuning fold, as a fraction of the known "
                             "side (default: 0.25). Independent of --window on purpose: the two "
                             "are in different units, so tying them together never made the "
                             "inner simulation match the outer task and only forced a separate "
                             "search per window. Keeping it fixed lets every window of one known "
                             "fraction share a single search.")
    parser.add_argument("--tune-candidates", type=int, default=6,
                        help="Configurations sampled per --tune search, i.e. the width of the "
                             "first halving rung (default: 6). This is the main speed dial. A "
                             "first-rung fit is not free -- measured on swe-chat it costs ~0.2 of "
                             "a full-size one, not the 1/9 its data share suggests, because the "
                             "124-class softmax and the 196x124 coefficient matrix do not shrink "
                             "with the sample -- so 12 candidates spend ~2.5 full fits before the "
                             "search has narrowed anything. Raise it when the search is picking "
                             "implausible configurations, not by default.")
    parser.add_argument("--tune-factor", type=int, default=3,
                        help="Halving aggressiveness (default: 3): each rung keeps the best "
                             "1/factor of the candidates and multiplies their training data by "
                             "factor. Together with --tune-candidates it fixes how many rungs "
                             "there are and therefore how small the first one is (see "
                             "halving_budget), so lowering it to 2 buys a larger, more "
                             "trustworthy first rung at roughly double the cost.")
    parser.add_argument("--tune-jobs", type=int, default=1,
                        help="Candidate fits run in parallel per --tune rung (default: 1). The "
                             "xgboost attack already uses every core inside a single fit, so "
                             "raising this oversubscribes for that attack while helping the "
                             "single-threaded ones.")
    parser.add_argument("--ood", default="none", choices=["none", "reject"],
                        help="Open-set handling: 'none' (default) scores only the documents whose "
                             "author is on the known side and counts the rest; 'reject' adds an "
                             "explicit reject option, thresholding the rejection score at a value "
                             "calibrated on the known side. Off by default because rejection is "
                             "much weaker than identification here -- see the module docstring.")
    parser.add_argument("--ood-calibration", default="accuracy", choices=["accuracy", "far"],
                        help="How the threshold is read off the known-side simulation: "
                             "'accuracy' maximises simulated open-set accuracy (default), 'far' "
                             "fixes the simulated false-accept rate at --ood-target-far.")
    parser.add_argument("--ood-target-far", type=float, default=0.10,
                        help="Target false-accept rate for --ood-calibration far (default: 0.10).")
    parser.add_argument("--ood-prior", type=float, default=None,
                        help="Assumed out-of-set rate in the unknown window, used to weight the "
                             "accuracy calibration. Default: forecast from the known side by "
                             "estimate_ood_prior (never from the unknown window).")
    parser.add_argument("--ood-holdout", type=float, default=0.3,
                        help="Fraction of known authors held out as impostors per calibration "
                             "fold (default: 0.3).")
    parser.add_argument("--ood-folds", type=int, default=3,
                        help="Calibration folds to average the threshold over (default: 3).")
    parser.add_argument("--ood-raw-distance", action="store_true",
                        help="Threshold the raw score instead of the cohort-normalised one "
                             "(usually worse: it lets document length drive the decision).")
    parser.add_argument("--top-ks", type=int, nargs="+", default=[1, 5, 10],
                        help="Measure metrics for top k most likely authors (default: 1, 5, 10).")
    parser.add_argument("--sweep-top-k", type=int, default=1,
                        help="k plotted in the window sweep, i.e. accuracy vs. number of target "
                             "users (default: 1). Must be one of --top-ks.")
    parser.add_argument("--seed", type=int, default=47,
                        help="Seed for the open-set calibration folds and the separability "
                             "diagnostic's sampling.")
    parser.add_argument("--output-dir", default=None,
                        help="Where to write outputs (default: experiments/results/<tag>).")
    args = parser.parse_args()
    if args.sweep_top_k not in args.top_ks:
        raise SystemExit(f"--sweep-top-k {args.sweep_top_k} is not among --top-ks {args.top_ks}; "
                         "the window sweep plots a k that was measured.")
    return args


def _percent(fraction: float) -> int:
    """Window fractions as whole percents, for filenames and printed labels."""
    return int(round(fraction * 100))


def output_tag(args: argparse.Namespace) -> str:
    """Short, self-describing directory name for this run's outputs.

    Every non-default choice that changes the numbers appears in the name, so two runs that
    differ in any of them cannot overwrite each other's results.
    """
    metric = "" if args.metric == "cosine" else f"_{args.metric}"
    scaled = "" if args.standardize else "_unstandardized"
    owner = "" if args.model_owner.lower() == "all" else f"_{args.model_owner.lower()}"
    language = "" if args.language.lower() == "all" else f"_{args.language.lower()}"
    openset = "" if args.ood == "none" else f"_openset_{args.ood_calibration}"
    windows = ("" if args.window == [0.10, 0.25, 0.50]
               else "_w" + "-".join(str(_percent(w)) for w in args.window))
    attacks = "-".join(args.attacks)
    return (f"{args.source}_{args.feature}{owner}{language}_{attacks}"
            f"_rolling{windows}{openset}{metric}{scaled}")


def report_window(scores: dict, headline: pd.DataFrame, author_report: pd.DataFrame,
                  args: argparse.Namespace) -> None:
    """Print one (window, attack) combination's results, grouped the way they should be read.

    Order matters here: the open-set block first if it is on, because ``ood_auroc`` decides
    whether any of its operating points mean anything; then the closed-set top-k table; then the
    whole-ranking summary that the table samples; then the per-user views, which routinely tell a
    different story from the document-weighted ones above them.
    """
    tuned = "" if scores["hyperparameters"] == "default" else f"  [tuned: {scores['hyperparameters']}]"
    print(f"\n=== [{scores['attack']}] known {scores['known_fraction']:.0%} "
          f"({scores['known_period']}) -> unknown next {scores['window']:.0%} "
          f"({scores['unknown_period']}){tuned}")
    print(f"  {scores['n_known_docs']:,} known docs / {scores['n_known_authors']:,} authors  ->  "
          f"{scores['n_unknown_docs']:,} unknown docs / {scores['n_unknown_authors']:,} authors")
    print(f"  of the unknown docs, {scores['n_in_set']:,} have a known author and "
          f"{scores['n_ood']:,} do not ({scores['ood_rate']:.1%} out-of-set)")
    print(f"  within-author distance mean {scores['within_author_mean']:.4f} | "
          f"between-author mean {scores['between_author_mean']:.4f}")

    if args.ood != "none":
        print(f"  threshold ({args.ood_calibration}) = {scores['threshold']:+.4f}  ->  "
              f"rejects {scores['rejection_rate']:.1%} of unknown docs "
              f"[known-side simulation: AUROC {scores['calibration_auroc']:.3f}, "
              f"top-1 {scores['calibration_top1']:.3f}]")
        print(f"  open-set accuracy {scores['open_set_accuracy']:.3f}  "
              f"(reject-all {scores['baseline_reject_all']:.3f}, "
              f"accept-all {scores['baseline_accept_all']:.3f})")
        print(f"  in-set accuracy   {scores['in_set_accuracy']:.3f}  "
              f"(ceiling without rejection {scores['closed_set_top1']:.3f}, "
              f"random {scores['random_top1_known_pool']:.4f})")
        print(f"  c@1 {scores['c_at_1']:.3f} on the in-set documents "
              f"(abstention pays iff this beats in-set accuracy)")
        print(f"  OOD recall {scores['ood_recall']:.3f}  precision {scores['ood_precision']:.3f}  |  "
              f"OOD AUROC {scores['ood_auroc']:.3f}  EER {scores['ood_eer']:.3f}  "
              + "  ".join(f"DIR@{int(far * 100)}%={scores[f'dir_at_far{int(far * 100)}']:.3f}"
                          for far in DIR_FAR_POINTS))
        if not np.isnan(scores["ood_auroc"]) and scores["ood_auroc"] < 0.55:
            direction = ("below chance -- out-of-set documents look *more* familiar than "
                         "in-set ones" if scores["ood_auroc"] < 0.5 else "at chance")
            print(f"  ! OOD AUROC is {direction}. The accept score does not separate in-set "
                  "from out-of-set, so no threshold on it beats the reject-all baseline "
                  "(see the module docstring for why).")

    print(f"  closed-set top-k over the {scores['n_in_set']:,} in-set documents "
          f"({int(headline['n_identities'].iloc[0])} target authors, "
          f"{int(headline['n_known_authors'].iloc[0])} in the pool):")
    for _, row in headline.iterrows():
        k = int(row["top"])
        print(f"    top {k:>2}: conv_acc={row['conv_acc']:.3f}  id_acc={row['id_acc']:.3f}  "
              f"|  random_id={row['random_id']:.3f}  advantage={row['advantage']:.3f}  "
              f"|  random_conv={row['random_conv']:.4f}  "
              f"advantage_conv={row['advantage_conv']:.3f}  "
              f"|  macro_conv_acc={scores[f'macro_conv_acc{k}']:.3f}")
    print(f"  whole ranking: MRR {scores['mrr']:.3f} (random {scores['random_mrr']:.3f})  |  "
          f"median rank {scores['median_rank']:.0f} of {scores['n_known_authors']} "
          f"(random {scores['random_median_rank']:.0f})  |  "
          f"mean percentile rank {scores['mean_percentile_rank']:.3f} (random 0.500)")
    print(f"  per user:      macro-F1 {scores['macro_f1']:.3f}  |  "
          f"MAP {scores['map']:.3f} (random {scores['random_map']:.3f})  "
          f"R-precision {scores['mean_r_precision']:.3f}  "
          f"[author as query: find everything one user wrote]")
    exposed = author_report[f"top{args.sweep_top_k}_accuracy"]
    print(f"  risk spread:   {int((exposed > 0.5).sum())} of {len(exposed)} target users have "
          f">50% of their documents attributed, {int((exposed == 0).sum())} never once  |  "
          f"confidence {scores['mean_confidence']:.2f} vs accuracy "
          f"{scores['closed_set_top1']:.2f} (ECE {scores['ece']:.3f})")
    print(f"  when selective: precision {scores['precision_at_10pct']:.3f} @10% coverage  |  "
          f"{scores['precision_at_25pct']:.3f} @25%  |  {scores['precision_at_50pct']:.3f} @50% "
          f"(keeping {scores['recall_at_50pct']:.0%} of its correct answers)")


def main() -> None:
    args = parse_args()
    frame, embeddings = load_documents_and_features(
        args.data_dir, args.source, args.feature, args.undated, args.model_owner, args.language
    )
    print(f"[{args.source}] {len(frame):,} documents x {embeddings.shape[1]} {args.feature} features | "
          f"{frame['author_id'].nunique():,} authors | {_period(frame)} | "
          f"attacks={' '.join(args.attacks)}{' | standardized' if args.standardize else ''}")
    if not args.standardize:
        print("warning: --no-standardize is set. Every attack measured considerably worse without "
              "it (see --standardize --help); this is a diagnostic mode, not a normal run.")

    results, predictions, ood_sweeps, headlines, cmcs, author_reports, all_trials = \
        [], {}, [], [], [], {}, []
    # One hyper-parameter search per (attack, known fraction), shared by every window that starts
    # from that known side. Lives here rather than in run_window so its scope is one experiment.
    tuning_cache: dict[tuple[str, float], dict] = {}
    windows = rolling_windows(len(frame), args.known_fractions, args.window)
    for fraction, window, known, unknown in windows:
        for attack in args.attacks:
            outcome = run_window(frame, embeddings, fraction, window, attack, known, unknown,
                                 args, tuning_cache)
            scores, window_predictions, ood_sweep, headline, cmc, author_report, trials = outcome
            results.append(scores)
            predictions[(fraction, window, attack)] = window_predictions
            headlines.append(headline)
            cmcs.append(cmc)
            author_reports[(fraction, window, attack)] = author_report
            if ood_sweep is not None:
                ood_sweeps.append(ood_sweep)
            if not trials.empty:
                all_trials.append(trials)
            report_window(scores, headline, author_report, args)

    output_dir = Path(args.output_dir) if args.output_dir else REPO_ROOT / "experiments" / "results" / output_tag(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_headlines = pd.concat(headlines, ignore_index=True)
    all_cmcs = pd.concat(cmcs, ignore_index=True)
    pd.DataFrame(results).to_csv(output_dir / "rolling_results.csv", index=False)
    all_headlines.to_csv(output_dir / "headline_results.csv", index=False)
    all_cmcs.to_csv(output_dir / "cmc_results.csv", index=False)
    if ood_sweeps:
        pd.concat(ood_sweeps, ignore_index=True).to_csv(output_dir / "ood_sweep.csv", index=False)
    if all_trials:
        pd.concat(all_trials, ignore_index=True).to_csv(output_dir / "tuning_trials.csv", index=False)

    def stem(fraction: float, window: float, attack: str) -> str:
        """``<attack>_known<pct>_window<pct>`` -- all three axes always, so two runs that differ
        in any of them cannot overwrite each other's per-window files."""
        return f"{attack}_known{_percent(fraction)}_window{_percent(window)}"

    for key, window_predictions in predictions.items():
        window_predictions.to_csv(output_dir / f"predictions_{stem(*key)}.csv", index=False)
    for key, author_report in author_reports.items():
        author_report.to_csv(output_dir / f"author_report_{stem(*key)}.csv", index=False)

    # One top-k figure per (window, attack), plus two per-attack figures across windows, rendered
    # by the same helpers run_experiment.py uses. The across-window figures are split by attack
    # rather than overlaid: 8 windows x several attacks in one axes is unreadable.
    label = FEATURE_LABELS.get(args.feature, args.feature)
    for headline in headlines:
        first = headline.iloc[0]
        plot_headline_topk(headline, label, output_dir /
                           f"topk_accuracy_{stem(first['known_fraction'], first['window'], first['attack'])}.pdf")
    figures = []
    for attack in args.attacks:
        series_label = f"{label} ({attack})" if len(args.attacks) > 1 else label
        sweep_path = output_dir / f"window_sweep_top{args.sweep_top_k}_{attack}.pdf"
        plot_window_sweep(all_headlines[(all_headlines["top"] == args.sweep_top_k)
                                        & (all_headlines["attack"] == attack)],
                          series_label, args.sweep_top_k, sweep_path)
        cmc_path = output_dir / f"cmc_curve_{attack}.pdf"
        plot_cmc_curve(all_cmcs[all_cmcs["attack"] == attack], series_label, cmc_path)
        figures += [sweep_path.name, cmc_path.name]

    stems = [stem(*key) for key in predictions]
    print(f"\nWrote results to {output_dir}/")
    print("  rolling_results.csv, headline_results.csv, cmc_results.csv"
          + (", ood_sweep.csv" if ood_sweeps else ""))
    print("  " + ", ".join(f"predictions_{s}.csv" for s in stems))
    print("  " + ", ".join(f"author_report_{s}.csv" for s in stems))
    print("  " + ", ".join(f"topk_accuracy_{s}.pdf" for s in stems) + ", " + ", ".join(figures))


if __name__ == "__main__":
    main()
