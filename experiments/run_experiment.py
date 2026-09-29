#!/usr/bin/env python
"""Rolling-window authorship attribution on the unified prompt dataset.

**The** experiment runner. Defending happens at build time
(``prompt_anonymity.data.apply_defenses`` then ``compute_features --defense``), so ``--defense``
here selects which already-defended vectors to attack and rewrites nothing. Utility scoring
(how much of a prompt a defense preserved) lives separately in
:mod:`prompt_anonymity.evaluation.utility`, run via ``experiments/eval_utility.py`` over the same
defended parquet.

1. **Input is the built parquet pair.** ``<split>.parquet`` (documents) joined on ``doc_id`` to
   ``<split>_<feature>.parquet`` (precomputed vectors from
   ``prompt_anonymity/data/compute_features.py``). No featurization, no defense, no GPU: this
   script only reads vectors that already exist.

2. **Interval known sides against one held-out test set.** Documents are ordered by ``ended_at``.
   The **final ``--test-fraction`` of the corpus (default 25%) is held out from every known side**
   and is the set every configuration is compared on; each ``--known-windows`` entry is an
   *interval* ``[start, end)`` of the timeline that the attacker is given, written ``XXYY`` in
   whole percents (``0025`` = the first quarter, ``2550`` = the second). Every known side ends at
   or before the test set, so the attacker never sees the future.

   The default six are the complete set of quartile-aligned intervals that do not touch the test
   quarter, and they exist to separate two things a single "known fraction" welds together:

   ==========  =============  ======  =====================
   config      interval       size    gap to the test set
   ==========  =============  ======  =====================
   ``0025``    [0, 0.25)      25%     2 quarters
   ``2550``    [0.25, 0.50)   25%     1 quarter
   ``5075``    [0.50, 0.75)   25%     0
   ``0050``    [0, 0.50)      50%     1 quarter
   ``2575``    [0.25, 0.75)   50%     0
   ``0075``    [0, 0.75)      75%     0
   ==========  =============  ======  =====================

   Read the gap-0 rows against each other for the **volume** effect at fixed recency, and the
   three 25%-size rows for the **staleness** effect at fixed volume. The grid is triangular by
   necessity -- a 75%-size known side cannot also be two quarters stale -- so there is no seventh
   cell.

   Each configuration still **scores its whole remaining future**, not just the test set: it costs
   only a matmul and it is what feeds the weekly temporal-decay figure. Restricting to the shared
   test set is a *plot-time* choice, made on the ``position`` column.

   Comparing configurations on their own in-set documents mixes attack strength with
   **enrollment reach**: a larger or fresher known side enrolls more users, so it scores more
   test documents, and the extra ones are systematically easier. ``plot_results.py`` therefore
   reports per-config accuracy as the headline and every *claim about an axis* on the paired
   intersection.

3. **The candidate pool can be narrowed by language** (``--language-aware``, off by default).
   Language is metadata an attacker reads straight off an anonymous document, and prompt
   anonymisation does not remove it, so restricting each document to the known authors who write
   at least one of its languages (``language_primary`` plus ``language_secondary``) is free
   information. See "Language-aware attribution" below.

4. **The candidate set is open**, and by default that is handled by *scoring only what can be
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
is multinomial logistic regression over the known authors. Unknown documents contribute nothing
but their feature vectors, and no unknown label is read anywhere in this pipeline.

Supervision is what makes the attack work. Cosine to an author centroid treats every feature
direction as equally informative, but some vary wildly *within* an author (noise) while others
separate authors (signal); with a large candidate pool that mistake is fatal, since the best of
many impostor distances lands closer than a genuine match. Learning the weighting instead of
using a fixed distance substantially improves both closed-set and open-set accuracy.

Language-aware attribution (``--language-aware``, off by default)
-----------------------------------------------------------------
Every attack here scores an unknown document against *every* known author, including the ones who
have only ever written in a language the document is not in. ``--language-aware`` removes those
from the document's pool: an unknown document's languages are matched against the languages each
known author is on record as using, and an author who shares none of them is marked ineligible.

It is applied as a **filter on the score matrix, not a change of model**. The attack is fitted
once per window on the whole known side exactly as it would be otherwise, and the filter then
writes ``-inf`` into the ineligible (document, author) cells before anything reads the matrix.
Two things follow. The comparison is clean -- the fitted model is identical to the unfiltered
run's, so the difference between the two runs is the pruning and nothing else. And the cost is
flat: one fit per window rather than one per language group. The work is done group by group
(:func:`language_candidate_groups`): the eligible-author mask depends only on a document's *set*
of languages, so it is derived once per distinct set and applied to that group's rows at once.

**Read the baselines, not the accuracy.** Narrowing the candidate pool raises top-1 whether or not
the attack learned anything, so every chance baseline is recomputed against each document's own
pool -- ``random_id<k>``, ``random_conv``, ``random_mrr``, ``mean_percentile_rank``, the CMC
``random`` column, and ``random_top1_candidate_pool`` -- and only the ``advantage`` columns say
whether the attack itself improved. ``rolling_results.csv`` also carries what the filter costs:
``n_true_author_pruned`` counts in-set documents whose own author writes none of their languages
on the known side and who are therefore now unattributable, and ``n_language_fallback`` counts
documents kept on the full pool because *no* known author writes their language (the filter has
no evidence there, and an empty pool would leave every metric undefined for them).

``--tune`` searches the unfiltered task, which is the consistent choice rather than an
oversight: the filter does not change what is fitted, so the model a language-aware run scores
with is exactly the model the search was selecting.

A trained out-of-set class (``--background``, off by default)
-------------------------------------------------------------
The reject option below reads "is this a stranger?" *off* a score matrix that was never asked the
question: a closed-set softmax has to assign its probability mass to some enrolled author, so a
stranger's document lands on whichever one it is least unlike, and how confident that looks is
only loosely related to whether the author was enrolled. ``--background <split>`` instead trains
the question in, adding one class fitted on documents sampled from another split
(:func:`load_background`) -- ShareChat, which publishes no author at all and is therefore useless
as a labelled side and exactly right as a pool nobody in the known side wrote.

The extra class is split back off before anything reads the score matrix
(:func:`split_background`), so every closed-set table is computed on a matrix of the same shape
and meaning a base run produces. What is new is one column in ``predictions_*.csv``,
``out_of_set_logit`` -- the model's own log-odds that a document belongs to none of the enrolled
authors -- next to the ``accept_score`` every run already writes. ``rolling_results.csv`` carries
``background_auroc`` and ``margin_auroc`` (plus ``*_dir_at_far*``) so the two are read against
each other on the same documents; ``margin_*`` is recorded on **every** run, with or without a
background pool, because a base run is what a background run has to be compared to.

Only the multiclass attacks fit a class per label, so ``--background`` is rejected for the
similarity family rather than silently ignored. See
:class:`~prompt_anonymity.attacks.multiclass.MinibatchLogisticAttribution` for why the plain
``logistic`` cannot be fitted at WildChat's author count at all.

The reject option (``--ood reject``, off by default)
----------------------------------------------------
Raw scores are not comparable across documents -- a short, generic document scores low against
*every* author -- so the accept/reject decision uses a **cohort-normalised** score: each
document's scores are z-scored across the candidate authors, and the rejection score is the
negated maximum (:func:`prompt_anonymity.attacks.rejection_score`).

The threshold is calibrated **from the known side alone** (:func:`calibrate_threshold`), by
simulating the task inside it: enrol 70% of the known authors, treat the rest as never-seen
impostors, and score held-out documents from both. Two things make that simulation usable:

* The two conditional rates -- P(reject | out-of-set) and P(correct and accepted | in-set) --
  are measured separately and only then combined under an assumed out-of-set rate, rather than
  mixed at the simulation's own (unrealistic) proportion.
* That rate is itself forecast from known data by :func:`estimate_ood_prior`, which replays the
  same chronological split *inside* the known window. The unknown window is never inspected.

What gets measured
------------------
Top-k accuracy is a sample of a ranking over the candidate authors, so the closed-set tables are
backed by the fuller families in :mod:`prompt_anonymity.evaluation.metrics` (all computed from the
one score matrix, see :func:`closed_set_detail`):

* **Whole ranking** -- ``mrr``, ``median_rank``, ``mean_percentile_rank``, and the complete CMC
  curve. Percentile rank is the only accuracy-like number here that is comparable across
  windows whose candidate pools differ in size, which the sweep guarantees they do.
* **Per user rather than per document** -- ``macro_conv_acc<k>``, ``macro_f1``, and a per-author
  risk table. The document-weighted headline can be dominated by a handful of prolific authors,
  and the macro view answers the question a privacy claim needs: how exposed is a *typical*
  user, and how wide is the spread.
* **Retrieval** -- ``map`` / ``mean_r_precision``, running each known author as a query against
  the anonymous documents ("find everything this person wrote"), a different threat model from
  identification and the only direction where MAP is not a restatement of MRR.
* **Open set** (``--ood reject``) -- ``ood_auroc``, ``ood_eer``, ``dir_at_far<pct>``, and PAN's
  ``c_at_1``, plus ``brier`` / ``ece`` for whether the confidence driving every threshold means
  anything.

Run (from the repo root)::

    python experiments/run_experiment.py
    python experiments/run_experiment.py --attacks nearest_neighbor  # the baseline attack
    python experiments/run_experiment.py --attacks logistic cosine nearest_neighbor
    python experiments/run_experiment.py --ood reject                # add the reject option
    python experiments/run_experiment.py --ood reject --ood-calibration far
    # only score a document against authors who write its language:
    python experiments/run_experiment.py --source wildchat --language-aware
    # one known side only (takes the run out of the comparable set):
    python experiments/run_experiment.py --known-windows 2575
    # compare the three trainable attacks, each tuned per window on that window's known side:
    python experiments/run_experiment.py --attacks logistic svm xgboost

Hyper-parameters (tuning is **on by default**; ``--no-tune`` opts out)
---------------------------------------------------------------------
Every attack's settings are chosen by :func:`tune_on_known`, which searches **only the known
side** of the window it is about to attack -- the attacker's own labelled data. Nothing in the
search, down to the standardization statistics, is derived from a document it will later be
scored on.

Tuning is the default because an untuned trainable attack measures the default settings rather
than the attack, and a privacy result should not understate the attacker. It costs nothing for
the attacks with no space to search (``nearest_neighbor``, ``cosine``), where it returns
immediately. ``--no-tune`` restores the old behaviour; it does *not* change the output directory
name, so re-running the same configuration either way overwrites the previous results. What was
actually used is recorded per window in ``rolling_results.csv``'s ``hyperparameters`` column, and
in full in ``tuning_trials.csv``.

The search is :class:`sklearn.model_selection.HalvingRandomSearchCV`: sample
``--tune-candidates`` configurations from :data:`HYPERPARAMETER_SPACES`, score them on a small
subsample of each fold's training block, discard all but the best ``1/--tune-factor``, triple the
data, repeat. Sampling rather than gridding also allows continuous ranges for ``C`` and the
learning rate, and lets xgboost search past a fixed tree-count ceiling.

**Halving saves less than the rung sizes suggest**, because these attacks are dominated by the
*author count* rather than the sample size: a fit on a fraction of the documents costs much more
than that fraction of the full one, since the softmax and coefficient matrix are the same size
either way. So the first rung is the expensive one, and ``--tune-candidates`` -- not
``--tune-factor`` -- is the dial that matters. ``max_iter`` is not a real halving lever either,
since lbfgs converges well inside the iterations it is allowed.

The larger saving was never in the search but in how often it runs: it belongs to the known side,
so it runs once per ``--known-windows`` value and is reused across the attacks that share it
(:func:`tuned_settings`). That is also what ``--tune-window`` exists for: the inner validation
block is a fixed share of the known side rather than a copy of the outer task, which is what used
to make otherwise identical searches differ.

Outputs (under ``--output-dir``), all carrying ``known_config`` and ``attack``
columns, so a multi-attack run stays one tidy table per file:

* ``rolling_results.csv`` -- one row per (known config, attack), every summary metric, plus the
  configuration itself (``known_start``, ``known_end``, ``known_size``, ``gap_fraction``,
  ``gap_weeks``) and the identity-level top-k (``id_acc<k>``, ``random_id<k>``,
  ``n_identities``). ``gap_weeks`` is the real elapsed time between the end of the known side and
  the start of the test set, which is what a staleness axis should be labelled in: documents are
  not uniformly dense, so a quarter of the corpus spans a different number of weeks per source.
* ``cmc_results.csv`` -- one row per (known config, attack, k): document-level top-k at every
  k, over the whole unknown side.
* ``predictions_<stem>.csv`` and ``author_report_<stem>.csv`` -- per document and per author
  (risk and retrieval, sorted most-exposed first), where ``<stem>`` is ``<attack>_known<XXYY>``:
  **one file per known configuration**, covering its whole unknown side. ``predictions_*.csv`` is
  the one downstream code should read: ``position`` and ``true_author_rank`` are what let
  ``plot_results.py`` restrict to the shared test set, or to any slice, without re-running the
  attack -- top-k over a subset is the share of it with ``rank <= k``. The author report is
  aggregated over the full unknown side and cannot be re-sliced.
  ``--language-aware`` adds each document's ``languages`` and whether its own author survived the
  filter.
* ``--ood reject`` adds ``ood_sweep.csv``, the open-set columns of ``rolling_results.csv``, and
  the per-document decision in ``predictions_*.csv``.

**No figures.** This script produces numbers only; every figure in the project is drawn by
``experiments/plot_results.py``, which reads these CSVs back and can therefore compare runs
against each other -- something a runner that plots its own output can never do. Run it with no
arguments after any experiment. It finds runs by their directory name, which is why
:func:`output_tag` spells the four axes out in full (``<dataset>_<defense>_<feature>_<attack>``,
with ``base`` for no defense).
"""

from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.spatial.distance import cdist
from scipy.special import logsumexp
from scipy.stats import loguniform
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.experimental import enable_halving_search_cv  # noqa: F401  (unlocks the import below)
from sklearn.model_selection import HalvingRandomSearchCV

from prompt_anonymity.attacks import ATTRIBUTION_ATTACKS, MULTICLASS_ATTACKS, rejection_score
from prompt_anonymity.data.compute_features import read_texts, split_path
from prompt_anonymity.defenses import DEFENSES
from prompt_anonymity.evaluation import LinkageRanking, headline_accuracy
from prompt_anonymity.features import KNOWN_SIDE_FEATURES
from prompt_anonymity.evaluation.metrics import (
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
    ranking_summary,
    retrieval_summary,
    true_author_ranks,
)

REPO_ROOT = Path(__file__).resolve().parent.parent

#: Where ``--data-dir`` looks by default: the *mirror* of the published dataset (written by
#: ``prompt_anonymity.data.download``), not ``data/dist`` where the build modules write. The two
#: hold overlapping copies of the same filenames, so a freshly built or freshly defended parquet in
#: ``data/dist`` is invisible to a run until it is copied here or named with ``--data-dir``.
DATA_DIR = REPO_ROOT / "data" / "hf"

#: How :func:`output_tag` spells "no defense". The results directory names all four axes
#: positionally, so the undefended case needs a name of its own rather than an empty slot;
#: ``experiments/plot_results.py`` parses the same word.
NO_DEFENSE_TAG = "base"

#: The corpora, named the way everything downstream names them: a ``--source`` value is also
#: the split, the parquet base name (``swe_chat.parquet``) and the dataset part of the results
#: directory :func:`output_tag` builds. (The ``source`` *column* inside the parquets still reads
#: ``swe-chat``; it is hashed into every ``author_id``, so it is data rather than a name and did
#: not follow this spelling.)
#:
#: **Deliberately a subset of ``build_dataset.SOURCES``**, which also has ``sharechat`` -- that
#: split has no author (a shared conversation link identifies the conversation, not the person),
#: and every stage here is keyed on the author, so it belongs as an out-of-set/background pool
#: (see :data:`BACKGROUND_SOURCES`) rather than a candidate here.
#:
#: ``wildchat_small`` is the seeded subset :mod:`prompt_anonymity.data.build_subset` cuts out of
#: ``wildchat``, with the same schema and author-keyed structure, sized for quicker iteration.
SOURCES = ("wildchat", "wildchat_small", "wildchat_tiny", "swe_chat")

# The extra class: "this document's author is not among the known authors". Not a valid
# author_id (those are ``<source>-<16 hex>``), so it can never collide with a real one.
OOD_LABEL = "<OOD>"

#: The label ``--background`` documents are trained under. Like :data:`OOD_LABEL` this is not a
#: valid ``author_id``, and for the same reason -- it becomes a class the attack fits, so it has
#: to be impossible for a real author to collide with. The two are distinct because they live at
#: different stages: this one is an input to the fit, ``OOD_LABEL`` is an output of the decision.
BACKGROUND_LABEL = "<BACKGROUND>"

#: Splits usable as a ``--background`` pool. ``sharechat`` is the one built for it: publicly
#: shared conversation links with no author at all, which is what makes it useless as a labelled
#: side and exactly right as a pool of documents nobody in the known side wrote. Note it is a
#: *different corpus* from either attack target -- see :func:`load_background` for what that costs.
BACKGROUND_SOURCES = ("sharechat",)

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

    Applied **before** the timeline is cut, so a configuration's interval is a share of the
    filtered corpus rather than of the whole split -- otherwise a filter would silently shrink the
    known sides and leave gaps between them. Both default to ``"all"`` (no filter).
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


# Document metadata this script reads: the join key, the chronological order, the label, the two
# filter axes, and the second language behind --language-aware. Everything else in the parquet --
# notably ``turns``, the raw conversation text -- is deliberately left on disk (see
# load_documents_and_features). Columns absent from a split's schema are simply not read.
DOCUMENT_COLUMNS = ("doc_id", "author_id", "ended_at", "language_primary", "language_secondary",
                    "model_owner")


#: Scratch column holding each document's row in the feature parquet. Dropped before the frame is
#: returned, so nothing downstream can come to depend on it.
FEATURE_ROW = "_feature_row"

#: Feature columns read per pass in :func:`read_feature_matrix`. Only sets the size of the
#: transient Arrow slab; the output is preallocated whole, so this trades a handful of extra reads
#: of a columnar file against holding the table.
FEATURE_COLUMN_BATCH = 256


def read_feature_matrix(feature_file: pq.ParquetFile, columns: list[str],
                        rows: np.ndarray) -> np.ndarray:
    """The feature parquet as one ``float32`` ``[len(rows) x len(columns)]`` array, in ``rows`` order.

    Built column-slab by column-slab into a preallocated output rather than through pandas. The
    obvious ``read_parquet(...)[columns].to_numpy(dtype=float)`` holds three copies of the matrix
    at once -- the Arrow table, its pandas frame, and a **float64** result at double the width --
    for numbers every attack immediately casts back to float32 anyway (see
    :mod:`prompt_anonymity.attacks.similarity.kernel`).

    ``float32`` is therefore the width the vectors already have on disk and the width they are
    used at; nothing downstream sees a different number. Rows are gathered as each column lands,
    so the un-permuted matrix is never materialised either.
    """
    matrix = np.empty((len(rows), len(columns)), dtype=np.float32)
    for start in range(0, len(columns), FEATURE_COLUMN_BATCH):
        slab = feature_file.read(columns=columns[start:start + FEATURE_COLUMN_BATCH])
        for offset, column in enumerate(slab.columns):
            matrix[:, start + offset] = column.to_numpy()[rows]
        del slab
    return matrix


def load_documents_and_features(data_dir, source: str, feature: str, undated: str = "drop",
                                model_owner: str = "all", language: str = "all",
                                defense: str = "none") -> tuple[pd.DataFrame, np.ndarray]:
    """Load one split's documents and its feature matrix, aligned and ordered by ``ended_at``.

    The two parquets are joined on ``doc_id`` (one-to-one), so a feature file that is stale or
    covers only part of the split is a hard error rather than a silent misalignment. Ties in
    ``ended_at`` are broken by ``doc_id`` so the ordering -- and therefore every window boundary
    -- is deterministic.

    ``defense`` selects *which* feature file, not a transformation done here: a defense rewrote
    the text long before this script ran (``prompt_anonymity.data.apply_defenses``) and the
    vectors of the rewritten text live in ``<split>_<defense>_<feature>.parquet``. The document
    metadata is unchanged by a defense, so it always comes from the undefended ``<split>.parquet``
    -- which is exactly why the two files join on ``doc_id`` at all.

    Some documents carry no timestamp at all (SWE-chat: a minority, all from one agent), and a
    chronological experiment has to decide where they go. ``undated="drop"`` (the default)
    removes them, because placing an undated document anywhere on the timeline invents an
    ordering and risks handing the attacker a document that is really from the future;
    ``undated="known"`` instead treats them as the oldest documents, so they are always on the
    known side -- the reading that an attacker holding undated history would get.
    """
    defended = "" if defense == "none" else f"_{defense}"
    documents_path = Path(data_dir) / f"{source}.parquet"
    features_path = Path(data_dir) / f"{source}{defended}_{feature}.parquet"
    for path in (documents_path, features_path):
        if not path.exists():
            raise SystemExit(f"{path} not found -- build it first with\n"
                             f"  python -m prompt_anonymity.data.build_dataset\n"
                             + (f"  python -m prompt_anonymity.data.apply_defenses --source {source} "
                                f"--defense {defense}\n" if defended else "")
                             + f"  python -m prompt_anonymity.data.compute_features --source {source} "
                             f"--feature {feature}"
                             + (f" --defense {defense}" if defended else ""))

    # Read only the metadata this script actually uses. ``turns`` holds the raw conversation
    # text and is most of the parquet's size -- and nothing downstream reads it, because
    # featurisation already happened. Skipping it is most of this function's footprint.
    document_schema = pq.ParquetFile(documents_path).schema_arrow.names
    wanted = [column for column in DOCUMENT_COLUMNS if column in document_schema]
    documents = pd.read_parquet(documents_path, columns=wanted)

    # The feature parquet mirrors some document metadata (``author_id``); keep only the columns
    # that are genuinely features, so nothing but numbers reaches the matrix. Checked against the
    # *full* document schema, not the narrowed frame above, so a feature named after a metadata
    # column we skipped still cannot sneak into the matrix.
    feature_file = pq.ParquetFile(features_path)
    feature_columns = [column for column in feature_file.schema_arrow.names
                       if column != "doc_id" and column not in document_schema]

    # The join is an index lookup, and the vectors never enter the frame. A merged frame would
    # carry every feature column through the filter, the undated cut, the sort and the
    # reset_index, and pandas copies the whole thing at every one of those steps -- large enough
    # to OOM-kill the job on the biggest feature matrices. Here only the metadata moves around and
    # the matrix is permuted exactly once, on the way out.
    keys = pd.Index(pd.read_parquet(features_path, columns=["doc_id"])["doc_id"])
    if not keys.is_unique:
        raise SystemExit(f"{features_path} repeats a doc_id; the join to documents is one-to-one.")
    documents[FEATURE_ROW] = keys.get_indexer(documents["doc_id"])
    n_missing = int((documents[FEATURE_ROW] < 0).sum())
    if n_missing:
        raise SystemExit(
            f"{n_missing:,} of {len(documents):,} documents have no {feature} vector. "
            f"Recompute with `python -m prompt_anonymity.data.compute_features --source {source} "
            f"--feature {feature}" + (f" --defense {defense}" if defended else "") + "`."
        )

    merged = order_documents(documents, undated, model_owner, language)
    return merged, read_feature_matrix(feature_file, feature_columns,
                                       merged.pop(FEATURE_ROW).to_numpy())


def order_documents(documents: pd.DataFrame, undated: str = "drop", model_owner: str = "all",
                    language: str = "all") -> pd.DataFrame:
    """Filter ``documents``, place or drop the undated ones, and sort them onto the timeline.

    Shared by both loaders so a parquet-backed feature and a known-side-fitted one see exactly the
    same documents in exactly the same order -- every window boundary depends on it. See
    :func:`load_documents_and_features` for what ``undated`` means.
    """
    merged = filter_documents(documents, model_owner, language)

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

    return merged.sort_values(["ended_at", "doc_id"], kind="mergesort").reset_index(drop=True)


def load_documents_and_counts(data_dir, source: str, feature, undated: str = "drop",
                              model_owner: str = "all", language: str = "all",
                              defenses: tuple[str, ...] = ("none",)
                              ) -> tuple[pd.DataFrame, list]:
    """Documents plus the raw counts a :data:`KNOWN_SIDE_FEATURES` entry is fitted on.

    The counterpart of :func:`load_documents_and_features` for a feature with no parquet: it reads
    the document **text** instead, one block per entry of ``defenses`` (``"none"`` is the original
    text, anything else that defense's rewrite), and has ``feature`` count n-grams over all the
    blocks in one shared column space. Returns ``(frame, counts)`` with ``counts[i]`` the sparse
    matrix for ``defenses[i]``, row-aligned to ``frame``.

    Nothing here is fitted: counts are per document. The fitted part (vocabulary, IDF) is done per
    known configuration in :func:`run_window`.
    """
    documents_path = Path(data_dir) / f"{source}.parquet"
    if not documents_path.exists():
        raise SystemExit(f"{documents_path} not found -- build it first with\n"
                         f"  python -m prompt_anonymity.data.build_dataset")
    document_schema = pq.ParquetFile(documents_path).schema_arrow.names
    wanted = [column for column in DOCUMENT_COLUMNS if column in document_schema]
    frame = order_documents(pd.read_parquet(documents_path, columns=wanted),
                            undated, model_owner, language)
    blocks = [load_document_texts(data_dir, source, defense, frame["doc_id"])
              for defense in defenses]
    return frame, feature.count(*blocks)


def load_document_texts(data_dir, source: str, defense: str, doc_ids: pd.Series) -> list[str]:
    """The text of ``doc_ids``, in that order, from the split or one defense's rewrite of it.

    Joined on ``doc_id`` -- a defended file keeps its own row order -- with a missing id a hard
    error, as for a feature parquet. Turns are joined exactly as ``compute_features`` joins them,
    so a known-side feature reads the same text every precomputed feature did.
    """
    defended = None if defense == "none" else defense
    path = split_path(source, data_dir, defended)
    if not path.exists():
        raise SystemExit(f"{path} not found" + (
            f" -- run `python -m prompt_anonymity.data.apply_defenses --source {source} "
            f"--defense {defense}` first" if defended else ""))
    keys = pd.Index(pd.read_parquet(path, columns=["doc_id"])["doc_id"])
    positions = keys.get_indexer(doc_ids)
    if (positions < 0).any():
        raise SystemExit(f"{int((positions < 0).sum()):,} documents are missing from {path}.")
    return read_texts(source, data_dir, positions, defense=defended)


def load_background(data_dir, source: str, feature: str, size: int,
                    seed: int) -> np.ndarray:
    """Feature vectors for a ``--background`` pool: documents the attack learns to *reject*.

    A closed-set softmax has to put its probability mass somewhere, so a document by a stranger is
    assigned to whichever enrolled author happens to be least unlike it, and how confident that
    assignment looks carries little information about whether the author was enrolled at all.
    Training an extra class on documents nobody in the known side wrote gives the model somewhere
    to put that mass, and turns "is this one of mine?" into a question the model answers directly
    rather than one read off the shape of its scores afterwards.

    **This is an attacker-side capability, not a leak.** The pool is a different corpus, drawn
    without reference to the split under attack, and no unknown document or label is touched: an
    attacker holding a public dump of chat logs has exactly this.

    **The cost is that the pool is a different corpus, and that is measurable**: a classifier can
    partly separate the two corpora on surface features alone, so some of this class's capacity
    goes on telling the corpora apart rather than on telling strangers from enrolled users.

    That confound can only *shrink* the measured effect, which is why the experiment is still
    readable: the out-of-set documents at test time are WildChat users who happen not to be
    enrolled, i.e. the same corpus as the in-set ones, so a class that had learned nothing but
    "is this ShareChat" would score them all alike and change no detection number. A gain here is
    therefore a real transfer; a null result is ambiguous between "no transfer" and "the corpus
    gap ate it", and the way to tell those apart is an in-corpus background pool built by holding
    known authors out of the enrolled set.

    ``size`` documents are drawn without replacement under ``seed``, spread over the whole split
    rather than taken from its head, and the row order is sorted so the parquet is read forwards.
    """
    path = Path(data_dir) / f"{source}_{feature}.parquet"
    if not path.exists():
        raise SystemExit(f"{path} not found -- build the background pool's vectors first with\n"
                         f"  python -m prompt_anonymity.data.compute_features --source {source} "
                         f"--feature {feature}")
    handle = pq.ParquetFile(path)
    available = handle.metadata.num_rows
    if size > available:
        raise SystemExit(f"--background-size {size:,} exceeds the {available:,} documents in "
                         f"{path.name}.")
    columns = [column for column in handle.schema_arrow.names
               if column not in ("doc_id", "author_id")]
    rows = np.sort(np.random.default_rng(seed).choice(available, size=size, replace=False))
    return read_feature_matrix(handle, columns, rows)


#: The default known configurations: every quartile-aligned interval that does not touch the
#: held-out test quarter. Ordered by size then start, so a run's output reads smallest-attacker
#: first. See the module docstring for why these six and not others -- they are the complete set,
#: and the grid they form is triangular because a large known side cannot also be far from the
#: test set.
DEFAULT_KNOWN_WINDOWS = ("0025", "2550", "5075", "0050", "2575", "0075")

#: Share of the timeline held out of *every* known side and shared by every configuration as the
#: set they are compared on. A run with a different value is not comparable to the default ones,
#: which is why :func:`output_tag` puts it in the directory name.
DEFAULT_TEST_FRACTION = 0.25


@dataclass(frozen=True)
class KnownConfig:
    """One known side: the interval ``[start, end)`` of the timeline the attacker is given.

    The whole design lives in this pair of numbers. ``size`` is how much labelled data the
    attacker holds and ``gap`` is how stale it is when the test set begins -- the two factors a
    single "known fraction" used to confound, because a prefix that is bigger is also, always,
    fresher.
    """

    start: float
    end: float
    #: Where the shared test set begins; the same for every configuration in a run, and carried
    #: here so a config can describe its own staleness without the caller passing it around.
    test_start: float

    @property
    def tag(self) -> str:
        """``known<XX><YY>`` -- the filename and column form, whole percents, no separator."""
        return f"known{_percent(self.start):02d}{_percent(self.end):02d}"

    @property
    def size(self) -> float:
        """Share of the corpus the attacker holds."""
        return self.end - self.start

    @property
    def gap(self) -> float:
        """Share of the corpus between the end of the known side and the start of the test set."""
        return self.test_start - self.end

    @property
    def label(self) -> str:
        return f"{self.start:.0%}-{self.end:.0%}"


def parse_known_window(spec: str, test_start: float) -> KnownConfig:
    """Turn a ``XXYY`` spec into a :class:`KnownConfig`, or fail with a usable message.

    Four digits, whole percents, start then end: ``0025`` is the first quarter and ``2575`` the
    middle half. The same spelling is used for the CLI, the output filenames and the
    ``known_config`` column, so there is one string to grep for when tracing a number back to the
    configuration that produced it.
    """
    if not (len(spec) == 4 and spec.isdigit()):
        raise SystemExit(f"--known-windows {spec!r}: expected four digits, start then end in "
                         f"whole percents (e.g. 0025 for the first quarter).")
    start, end = int(spec[:2]) / 100, int(spec[2:]) / 100
    if start >= end:
        raise SystemExit(f"--known-windows {spec!r}: start {start:.0%} is not before end {end:.0%}.")
    if end > test_start + 1e-9:
        raise SystemExit(
            f"--known-windows {spec!r}: ends at {end:.0%}, inside the held-out test set that "
            f"starts at {test_start:.0%}. The test set is held out of every known side so all "
            f"configurations are compared on the same documents; lower --test-fraction to "
            f"enlarge the space of usable known sides."
        )
    return KnownConfig(start=start, end=end, test_start=test_start)


def known_configurations(n_documents: int, specs, test_fraction: float
                         ) -> list[tuple[KnownConfig, slice, slice]]:
    """``(config, known_slice, unknown_slice)`` for every runnable known configuration.

    The known side is the interval the configuration names; the unknown side is **everything
    after it**, all the way to the end of the corpus, not just the shared test set. Scoring the
    whole future costs one matmul over rows that were going to be loaded anyway, and it is what
    keeps the weekly temporal-decay figure drawable -- a configuration restricted to the test
    quarter could only ever show the last quarter's worth of staleness. Which documents a given
    figure reads is decided at plot time from the ``position`` column.

    Configurations that leave a degenerate side (fewer than two known documents, or nothing to
    attribute) are skipped; it is an error only if nothing at all survives.
    """
    test_start = 1.0 - test_fraction
    configurations, skipped = [], []
    for spec in specs:
        config = parse_known_window(spec, test_start)
        start, end = int(round(config.start * n_documents)), int(round(config.end * n_documents))
        if end - start < 2 or end >= n_documents:
            skipped.append(f"{config.tag} leaves an empty side ({end - start} known, "
                           f"{n_documents - end} unknown of {n_documents} documents)")
            continue
        configurations.append((config, slice(start, end), slice(end, n_documents)))
    if skipped:
        print(f"skipping {len(skipped)} known configuration(s): " + "; ".join(skipped))
    if not configurations:
        raise SystemExit(
            f"no runnable value in --known-windows {list(specs)} over {n_documents} documents."
        )
    return configurations


def gap_weeks(frame: pd.DataFrame, config: KnownConfig, known: slice) -> float:
    """Real elapsed time between the end of ``config``'s known side and the test set's start.

    The design's staleness axis is defined in *positions* -- equal document counts, which is what
    holds the volume axis exactly fixed -- but positions are not time: documents are far denser
    early in both corpora, so the same fraction of the timeline spans very different amounts of
    real time across corpora. Every figure and table should be labelled with this rather than with
    "quarters", and a cross-dataset staleness axis has to use it. ``nan`` when either end has no
    usable timestamp.
    """
    test_index = int(round(config.test_start * len(frame)))
    if known.stop < 1 or test_index >= len(frame):
        return float("nan")
    stamps = pd.to_datetime(pd.Series([frame["ended_at"].iloc[known.stop - 1],
                                       frame["ended_at"].iloc[test_index]]),
                            format="mixed", utc=True, errors="coerce")
    if stamps.isna().any():
        return float("nan")
    return float((stamps.iloc[1] - stamps.iloc[0]).total_seconds() / (7 * 24 * 3600))


def zscores(args: argparse.Namespace) -> bool:
    """Whether this run z-scores its features: ``--standardize``, unless the feature scales itself.

    A :data:`KNOWN_SIDE_FEATURES` entry (``char_ngram_tfidf``) is TF-IDF with unit-L2 rows, and a
    per-column z-score would divide each column by its own spread -- cancelling the IDF weights
    outright, since IDF *is* a per-column scale. Its scaling is part of the feature, so the run is
    still the default configuration and keeps the default directory name.
    """
    feature = KNOWN_SIDE_FEATURES.get(args.feature)
    return args.standardize and not getattr(feature, "already_scaled", False)


def standardize(known: np.ndarray, *others: np.ndarray) -> tuple[np.ndarray, ...]:
    """Z-score the known side and every other block using **known-side** statistics only.

    Hand-crafted style features mix ratios in [0, 1] with occasional raw counts, so without scaling
    a handful of wide-range columns dominate any distance. Fitting the mean/scale on the known side alone
    keeps the unknown documents out of the attacker's view of the data. Zero-variance columns are
    left alone rather than divided by zero.

    The statistics are accumulated in float64 and the result is written as float32 -- which is
    exactly the arithmetic a float64 z-score followed by the attacks' float32 cast performed, so
    the numbers are unchanged, at half the memory. The z-score itself runs in row chunks so the
    float64 temporary is bounded rather than a second copy of the whole side.

    ``others`` is normally just the unknown side. ``--background`` adds a second block -- the
    out-of-set pool the attack trains against -- and it is deliberately scaled by the *known
    side's* statistics rather than by its own or by the union's: the geometry the authors live in
    then does not move when a background pool is switched on, so a background run and a base run
    differ in what the attack was told and in nothing else.
    """
    center = known.mean(axis=0, dtype=np.float64)
    scale = known.std(axis=0, dtype=np.float64)
    scale = np.where(scale > 0, scale, 1.0)
    return tuple(_zscore(block, center, scale) for block in (known, *others))


#: Rows z-scored per pass. Sizes only the float64 temporary (8,192 x n_features), not the output.
ZSCORE_ROW_BATCH = 8192


def _zscore(block: np.ndarray, center: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """``(block - center) / scale`` in float64, returned as float32, a row chunk at a time."""
    out = np.empty(block.shape, dtype=np.float32)
    for start in range(0, len(block), ZSCORE_ROW_BATCH):
        rows = slice(start, start + ZSCORE_ROW_BATCH)
        out[rows] = (block[rows].astype(np.float64) - center) / scale
    return out


# --- language-aware candidate filtering (--language-aware) -------------------
#
# Language is metadata the attacker gets for free: it is readable straight off an anonymous
# document, and it is not something prompt anonymisation removes. Restricting each document to
# the authors already on record as writing one of its languages is therefore a legitimate
# narrowing of the candidate pool, and on a multilingual corpus like WildChat it is a large one.
#
# Everything below works on the *set* of languages a document is in, ``language_primary`` plus
# ``language_secondary`` where one was detected, so a document that mixes English and Russian
# keeps both authors' pools open rather than being forced onto its majority language.

LANGUAGE_COLUMNS = ("language_primary", "language_secondary")

# Ineligible candidates are marked in the score matrix rather than tracked in a parallel mask: at
# WildChat's scale a boolean [n_unknown x n_authors] mask is another gigabyte, and every consumer
# of the matrix (argmax, true_author_ranks, max_softmax_confidence, rejection_score,
# author_query_metrics) already reads -inf as "ranked below every real candidate".
INELIGIBLE = -np.inf


def document_language_sets(frame: pd.DataFrame) -> np.ndarray:
    """One ``frozenset`` of languages per document, from the primary and secondary columns.

    A document with no detected language at all gets the empty set, which intersects nothing --
    :func:`language_candidate_groups` gives those the unfiltered pool rather than no candidates.
    """
    present = [column for column in LANGUAGE_COLUMNS if column in frame.columns]
    if "language_primary" not in present:
        raise SystemExit(
            "--language-aware needs a language_primary column, which this split's parquet does "
            "not have. Rebuild it with `python -m prompt_anonymity.data.build_dataset`."
        )
    columns = [frame[column].to_numpy(dtype=object) for column in present]
    return np.array(
        [frozenset(value for value in values if isinstance(value, str) and value)
         for values in zip(*columns)],
        dtype=object,
    )


def author_language_sets(languages: np.ndarray, labels: np.ndarray) -> dict:
    """Every language each known author is on record as having written in.

    This is the attacker's whole model of "who writes what": it is built from the known side
    alone, so an author who switches language inside the unknown window can be filtered away from
    their own document. That is a real cost of the heuristic, not a bug -- ``run_window`` counts
    it as ``n_true_author_pruned``.
    """
    seen: dict = {}
    for author, document_languages in zip(labels, languages):
        seen[author] = seen.get(author, frozenset()) | document_languages
    return seen


def language_candidate_groups(unknown_languages: np.ndarray, known_languages: np.ndarray,
                              known_labels: np.ndarray, authors: np.ndarray) -> list:
    """``(languages, document_rows, eligible_authors, is_fallback)`` per distinct language set.

    Grouping is what makes this cheap. The eligible-author mask depends only on the document's
    *set* of languages, so it is derived once per distinct set rather than once per document, and
    applied to that group's rows in a single pass.

    ``eligible_authors`` is a boolean over ``authors`` (the columns of the score matrix, in their
    order): true where the author wrote at least one known document in at least one of the
    group's languages. A group whose set matches no known author at all -- a language nobody in
    the known window writes, or a document with no detected language -- keeps the **full** pool
    and is flagged ``is_fallback``: with no author writing that language the filter has no
    evidence to act on, and blanking the row would leave the document with no candidates and
    every downstream metric undefined for it. The flag is what separates that case from the
    unremarkable one where the filter simply does not bind because every known author happens to
    write the language, which is the norm on an English-dominated corpus.
    """
    by_author = author_language_sets(known_languages, known_labels)
    author_languages = [by_author.get(author, frozenset()) for author in authors]

    rows_by_language: dict = {}
    for row, document_languages in enumerate(unknown_languages):
        rows_by_language.setdefault(document_languages, []).append(row)

    groups = []
    for document_languages, rows in rows_by_language.items():
        eligible = np.array([bool(document_languages & written) for written in author_languages],
                            dtype=bool)
        is_fallback = not eligible.any()
        if is_fallback:
            eligible = np.ones(len(authors), dtype=bool)
        groups.append((document_languages, np.array(rows, dtype=int), eligible, is_fallback))
    return groups


def apply_language_filter(scores: np.ndarray, authors: np.ndarray, unknown_languages: np.ndarray,
                          known_languages: np.ndarray, known_labels: np.ndarray,
                          true_labels: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, dict]:
    """Mark every author who does not write a document's language as ineligible, **in place**.

    In place because the alternative is a second copy of a matrix that can reach several GB on
    WildChat; the attacks all return a freshly computed score matrix, so nothing else holds a
    reference to it. Assignment through :func:`numpy.ix_` with a scalar writes element-wise and
    allocates nothing beyond the index arrays, which is what keeps this inside the job's memory cap.

    Returns ``(candidate_counts, true_author_is_candidate, stats)``:

    * ``candidate_counts`` -- eligible authors per document, i.e. the pool size every chance
      baseline for that document has to be computed against. This is the number that makes a
      language-aware result readable: pruning 20,000 candidates to 300 raises top-1 whether or
      not the attack learned anything, and only the gap to ``random_top1_candidate_pool`` says
      which happened.
    * ``true_author_is_candidate`` -- per document, whether its true author survived the filter.
      False for a document whose author is out-of-set (they were never a candidate), and false
      for the in-set documents the filter has made unattributable, which is the price of the
      heuristic and is summarised as ``n_true_author_pruned``.
    * ``stats`` -- the group count and the fallback count, for ``rolling_results.csv``.
    """
    groups = language_candidate_groups(unknown_languages, known_languages, known_labels, authors)
    column_of = {author: index for index, author in enumerate(authors)}
    candidate_counts = np.empty(len(unknown_languages), dtype=int)
    true_author_is_candidate = np.zeros(len(unknown_languages), dtype=bool)
    n_fallback = 0

    for _, rows, eligible, is_fallback in groups:
        ineligible = np.flatnonzero(~eligible)
        if len(ineligible):
            scores[np.ix_(rows, ineligible)] = INELIGIBLE
        candidate_counts[rows] = int(eligible.sum())
        n_fallback += len(rows) if is_fallback else 0
        if true_labels is not None:
            true_author_is_candidate[rows] = [
                column is not None and bool(eligible[column])
                for column in (column_of.get(author) for author in true_labels[rows])
            ]

    return candidate_counts, true_author_is_candidate, {
        "n_language_groups": len(groups),
        "n_language_fallback": n_fallback,
    }


# --- open-set attribution ----------------------------------------------------

def execution_settings(name: str, args: argparse.Namespace) -> dict:
    """The settings that decide *where* an attack runs rather than what it learns.

    Split out from the rest of :func:`build_attack` because these are the only ones that also
    have to reach :func:`tune_on_known`. The search fits the same attack a few dozen times and
    the final scoring fit once, so a device applied only in :func:`build_attack` would leave the
    expensive part of a ``--tune`` run on the CPU while the run reported itself as using a GPU.

    They are not hyper-parameters: nothing samples them, they never appear in
    ``tuning_trials.csv``, and no tuning result can override one.
    """
    return {"device": args.xgboost_device} if name == "xgboost" else {}


def build_attack(name: str, args: argparse.Namespace, overrides: dict | None = None):
    """Construct the named attack from :data:`~prompt_anonymity.attacks.ATTRIBUTION_ATTACKS`.

    Returned as a *factory* of fresh, unfitted attacks rather than one instance, because
    threshold calibration refits the same configuration on each simulation fold. Each attack
    picks up whichever of the CLI tuning flags applies to it; the rest are ignored.
    ``overrides`` (from :func:`tune_on_known`) wins over the flags.
    """
    attack = ATTRIBUTION_ATTACKS[name]
    settings = execution_settings(name, args)
    if name in ("logistic", "logistic_sgd"):
        # Both spellings of the same model take the same two flags, so that switching between the
        # exact and the minibatch fit changes how it is fitted and nothing about what is fitted.
        settings |= {"C": args.regularization,
                     "class_weight": "balanced" if args.balanced else None}
    elif name in ("wccn", "plda"):
        settings |= {"shrinkage": args.shrinkage}
    elif name == "nearest_neighbor":
        settings |= {"metric": args.metric}
    settings.update(overrides or {})
    return (lambda: attack(**settings)) if settings else attack


# Hyper-parameter search spaces sampled by --tune, in the form
# :class:`sklearn.model_selection.ParameterSampler` accepts: a mapping of name -> (list of choices
# | scipy distribution with ``.rvs``), or a *list* of such mappings when the space is conditional
# (an SVM's ``gamma`` only exists for the RBF kernel, so each kernel is its own sub-space and one
# is chosen uniformly per draw).
#
# These are continuous rather than a handful of grid points, which is the part of random search
# that is free: a draw from a range costs exactly what a draw from a list costs. What is *not*
# free is how expensive an individual draw can be -- see the xgboost note below -- so these ranges
# are bounded by fit cost rather than by what is plausible.
# Attacks absent from this mapping have nothing worth tuning and are used as configured.
HYPERPARAMETER_SPACES: dict[str, dict | list[dict]] = {
    "logistic": {"C": loguniform(0.02, 100.0), "class_weight": [None, "balanced"]},
    # Same two knobs as `logistic`, over a range that reaches lower. The floor is not copied from
    # there because the shape of the problem is not the same at a much larger author count. `steps`
    # and `learning_rate` are deliberately absent: the first is a compute budget rather than a
    # hyper-parameter, and the second measured insensitive over its useful range, so sampling it
    # would spend the search budget learning nothing.
    "logistic_sgd": {"C": loguniform(0.002, 100.0), "class_weight": [None, "balanced"]},
    "svm": [
        {"kernel": ["linear"], "C": loguniform(0.1, 30.0)},
        {"kernel": ["rbf"], "C": loguniform(1.0, 300.0), "gamma": ["scale"]},
        {"kernel": ["rbf"], "C": loguniform(1.0, 300.0), "gamma": loguniform(1e-3, 1e-1)},
    ],
    # An xgboost fit costs roughly depth x trees x documents, so this is the one space where
    # widening is not free -- an overly aggressive range makes the halving search *slower* than a
    # plain grid, having spent the saving on individually huge candidates. The floor on
    # learning_rate keeps the tree count honest: a slow learner needs many more rounds to pay off,
    # so the cheap way to allow more trees is to rule out configurations that would need thousands.
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


#: :class:`TunableAttack`'s own constructor arguments, as opposed to the attack's settings. Kept out
#: of ``tuning_trials.csv`` and of the settings handed on to the real fit.
WRAPPER_PARAMS = ("attack_name", "standardize", "feature_transform")


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

    ``feature_transform`` does the same for a :data:`KNOWN_SIDE_FEATURES` entry: the search is
    then handed raw n-gram counts, and every fit clones the transform and fits its vocabulary and
    IDF on that fit's own training rows before the attack sees anything.
    """

    def __init__(self, attack_name: str, standardize: bool = True, feature_transform=None,
                 **settings):
        self.attack_name = attack_name
        self.standardize = standardize
        self.feature_transform = feature_transform
        self.settings = settings

    def get_params(self, deep: bool = True) -> dict:
        """Flatten the settings dict into the search space's own parameter names."""
        return {"attack_name": self.attack_name, "standardize": self.standardize,
                "feature_transform": self.feature_transform, **self.settings}

    def set_params(self, **params):
        """Accept any hyper-parameter name; unrecognised ones are passed to the attack."""
        settings = dict(self.settings)
        for key, value in params.items():
            if key in WRAPPER_PARAMS:
                setattr(self, key, value)
            else:
                settings[key] = value
        self.settings = settings
        return self

    def fit(self, embeddings, labels):
        if self.feature_transform is not None:
            self.transform_ = clone(self.feature_transform).fit(embeddings)
            embeddings = self.transform_.transform(embeddings)
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
        if self.feature_transform is not None:
            embeddings = self.transform_.transform(embeddings)
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

    (Getting this wrong is not loud: an under-anchored search still finds a reasonable winner, but
    every reported known-side CV score is quietly lower across the board, purely because they were
    measured on less data than the search's own last rung.)

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
                  args: argparse.Namespace, feature_transform=None) -> tuple[dict, pd.DataFrame]:
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

    This runs **once per known configuration** -- see :func:`tuned_settings`, which memoises it
    across the attacks that share one. It must not be run once for the whole experiment: the
    configurations overlap, so a value tuned on ``known0075`` and reused for ``known0025`` would
    have been selected partly on documents that ``known0025`` must attribute, which is exactly the
    leak the per-configuration search exists to avoid.

    Against an exhaustive grid over the same folds and criterion, this is roughly a wash in final
    accuracy for meaningfully less wall clock; the wider continuous ranges make up most of what the
    coarser halving search loses relative to a full grid.

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
        search = _halving_search(name, space, folds, factor, embeddings, labels, args,
                                 feature_transform)

    # One row per (candidate, rung), so the CSV shows both what was tried and where each
    # candidate was cut. `train_budget` is the rung's resource level as a share of the whole
    # known side; each fold trains on that same share of *its* (shorter) training block.
    trials = pd.DataFrame(search.cv_results_["params"]).drop(
        columns=list(WRAPPER_PARAMS), errors="ignore")
    trials["known_cv_top1"] = search.cv_results_["mean_test_score"]
    trials["known_cv_std"] = search.cv_results_["std_test_score"]
    trials["rung"] = search.cv_results_["iter"]
    trials["train_budget"] = search.cv_results_["n_resources"]
    trials["selected"] = np.arange(len(trials)) == search.best_index_
    best = {key: value for key, value in search.best_params_.items()
            if key not in WRAPPER_PARAMS}
    return best, trials


def tuned_settings(name: str, tag: str, embeddings: np.ndarray, labels: np.ndarray,
                   args: argparse.Namespace, cache: dict,
                   feature_transform=None) -> tuple[dict, pd.DataFrame]:
    """:func:`tune_on_known`, run at most once per ``(attack, known configuration)``.

    ``tag`` is the configuration's :attr:`KnownConfig.tag`, and it is the cache key: two
    configurations that hold overlapping documents are still different known sides and must not
    share a search. Kept memoised even though the runner visits each configuration once, because
    ``--attacks`` can name several attacks and the calibration path refits.

    ``cache`` is owned by the caller (:func:`main`) so that its lifetime is one experiment and the
    sharing is visible in the driver rather than hidden in module state. The trials table is
    returned only on a miss, so ``tuning_trials.csv`` holds one block per search instead of the
    same rows repeated once per attack.
    """
    key = (name, tag)
    if key in cache:
        return cache[key], pd.DataFrame()
    settings, trials = tune_on_known(name, embeddings, labels, args, feature_transform)
    cache[key] = settings
    return settings, trials


def base_run_directory(args: argparse.Namespace) -> Path:
    """The undefended run of this same experiment: :func:`output_tag` with ``--defense none``."""
    base = argparse.Namespace(**{**vars(args), "defense": "none"})
    return REPO_ROOT / "experiments" / "results" / output_tag(base)


def load_tuned_settings(source: Path, attacks, tags) -> dict[tuple[str, str], dict]:
    """Every ``(attack, known configuration)``'s selected settings from another run's search.

    Backs ``--tuned-from``. A defended run with the default ``--known-defense none`` tunes on the
    **undefended** known side -- the same vectors, labels, folds and seeded candidates as the base
    run -- so its search is a repeat of the base run's and must pick the same settings; this reads
    them instead of spending the search again. The values come from ``tuning_trials.csv``'s
    selected rows at full precision, not from ``rolling_results.csv``, whose column is rounded to
    four significant figures.

    The returned dict has the shape of :func:`tuned_settings`' cache, which is what it is loaded
    into. An attack with no registered space needs nothing and gets ``{}``. An attack **with** one
    must have a selected row for every configuration, or this refuses -- copying from a run that
    was never tuned would quietly publish the defaults as if a search had chosen them.
    """
    trials_path = source / "tuning_trials.csv"
    trials = (pd.read_csv(trials_path, keep_default_na=False, na_values=[""])
              if trials_path.exists() else pd.DataFrame())
    settings: dict[tuple[str, str], dict] = {}
    for attack in attacks:
        space = HYPERPARAMETER_SPACES.get(attack)
        for tag in tags:
            if not space:
                settings[(attack, tag)] = {}
                continue
            chosen = (trials[(trials["known_config"] == tag) & (trials["attack"] == attack)
                             & (trials["selected"].astype(str) == "True")]
                      if not trials.empty else trials)
            if len(chosen) != 1:
                raise SystemExit(
                    f"--tuned-from {source}: no tuned {attack} settings for {tag} "
                    f"({len(chosen)} selected rows in {trials_path.name}). That run was not tuned "
                    f"(or not for this configuration); tune it first, or drop --tuned-from.")
            keys = space.keys() if isinstance(space, dict) else {k for s in space for k in s}
            row = chosen.iloc[0]
            # class_weight=None is written as an empty cell and read back as NaN.
            settings[(attack, tag)] = {
                key: (None if pd.isna(row[key]) else
                      row[key].item() if hasattr(row[key], "item") else row[key])
                for key in keys if key in row.index}
    return settings


def _halving_search(name: str, space, folds, factor: int, embeddings: np.ndarray,
                    labels: np.ndarray, args: argparse.Namespace,
                    feature_transform=None) -> HalvingRandomSearchCV:
    """The fitted search behind :func:`tune_on_known`; split out only to keep that one readable."""
    return HalvingRandomSearchCV(
        TunableAttack(attack_name=name, standardize=zscores(args),
                      feature_transform=feature_transform, **execution_settings(name, args)),
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
    slice came from authors absent earlier. Imperfect, but far closer than the simulation's
    implicit prior, and it is exactly the kind of estimate a real attacker could make from their
    own history.
    """
    known_labels = np.asarray(known_labels)  # already in chronological order
    cut = int(round(len(known_labels) * known_fraction / (known_fraction + window)))
    if cut < 1 or cut >= len(known_labels):
        return float("nan")
    return float((~np.isin(known_labels[cut:], np.unique(known_labels[:cut]))).mean())


def calibrate_threshold(factory, embeddings: np.ndarray, labels: np.ndarray, folds,
                        calibration: str = "accuracy", target_far: float = 0.10,
                        cohort_normalize: bool = True, ood_prior: float = 0.2,
                        languages: np.ndarray | None = None) -> tuple[float, dict]:
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

    ``languages`` (the known side's language sets, aligned to ``embeddings``) switches on the same
    candidate filter the real window gets. It has to be applied here too, not just at scoring
    time: the rejection score is normalised *across the cohort*, so pruning the cohort changes the
    distribution the threshold is read off, and a threshold calibrated on unpruned scores would be
    applied to a different quantity from the one it was chosen for.

    Returns ``(threshold, diagnostics)``; the diagnostics carry the simulated AUROC and top-1,
    which say how much to trust the threshold *before* any unknown document is touched.
    """
    thresholds, simulated_auroc, simulated_top1 = [], [], []
    for train_rows, query_rows in folds:
        fitted = factory().fit(embeddings[train_rows], labels[train_rows])
        scores = fitted.score(embeddings[query_rows])
        if languages is not None:
            apply_language_filter(scores, fitted.authors, languages[query_rows],
                                  languages[train_rows], labels[train_rows])
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
    # nanmean: cosine is undefined against an all-zero row, which a document containing none of a
    # fitted vocabulary's n-grams is (char_ngram_tfidf); it has no distance to contribute.
    return float(np.nanmean(np.concatenate(pairs))) if pairs else float("nan")


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
    return float(np.nanmean(distances[different])) if different.any() else float("nan")


# --- attack ------------------------------------------------------------------

def random_identity_accuracy(unknown_labels: np.ndarray, k: int, candidate_counts: np.ndarray) -> float:
    """Identity-level random baseline when each document has its own candidate pool.

    Generalises :func:`prompt_anonymity.evaluation.metrics.random_guessing_accuracy`, which assumes a single
    pool shared by every document. A guesser shortlisting ``k`` of document *i*'s ``n_i``
    candidates names the true author with probability ``min(k, n_i) / n_i``, and an identity is
    re-identified if *any* of its documents' guesses lands, so its chance is
    ``1 - prod_i (1 - p_i)``. With a constant pool this is exactly the ``1 - (1 - p) ** c``
    formula the package function uses, so the default (unfiltered) runs are unaffected.

    The product is taken in log space: a document whose whole pool fits inside k contributes a
    zero factor, and multiplying tens of thousands of near-one factors directly loses precision
    on the very identities -- the prolific ones -- whose baseline is highest.
    """
    unknown_labels = np.asarray(unknown_labels)
    counts = np.asarray(candidate_counts, dtype=float)
    miss = 1.0 - np.minimum(k, counts) / counts
    _, inverse = np.unique(unknown_labels, return_inverse=True)
    log_miss = np.zeros(inverse.max() + 1 if inverse.size else 0)
    np.add.at(log_miss, inverse, np.log(np.maximum(miss, np.finfo(float).tiny)))
    return float(np.mean(1.0 - np.exp(log_miss))) if log_miss.size else float("nan")


def closed_set_table(distances: np.ndarray, known_labels, unknown_labels,
                     candidate_counts: np.ndarray, args: argparse.Namespace):
    """The standard top-k table, on this window's scored documents.

    Unlike in the removed fixed-split runner it comes from, this table is not written out on its
    own: its identity-level columns are
    widened into ``rolling_results.csv`` by :func:`identity_level_scores`, and the table itself
    stays in memory to draw ``topk_accuracy_*.pdf`` and ``window_sweep_top<k>_*.pdf``. Its
    document-level columns were already duplicated elsewhere -- see :func:`identity_level_scores`.

    ``distances`` must already be restricted to unknown documents whose author appears among the
    known authors: top-k ranking is only defined when the true author is in the candidate pool
    (and :class:`LinkageRanking` requires it).

    The random baselines are computed over the **known** authors, which is the pool this attack
    actually searches: every known author is a column in the ranking, and only the denominator
    of ``id_acc`` is restricted to the identities that happen to appear in the window. The
    attacker does not know which of them will show up (in a typical window only a third do), so
    a guesser handed that list would be strictly better informed than the attack it is
    benchmarking.

    ``candidate_counts`` is that pool size per document, and it is the whole reason
    ``--language-aware`` is readable. Narrowing a document to the authors who write its language
    raises top-1 whether or not the attack learned anything -- guessing among 300 candidates
    beats guessing among 20,000 -- so the baselines have to narrow with it. Every column below is
    therefore averaged over the documents' own pools rather than computed from one pool size;
    without a candidate filter every entry equals the number of known authors and the formulas
    collapse to the ones they had before.

    * ``random_id`` / ``advantage`` -- identity level, a guesser naming ``k`` of the known
      authors per document. This **overrides** the column :func:`headline_accuracy` computes,
      which guesses among the target identities only. :func:`headline_accuracy` still computes
      the original definition for any other caller, so the two are not interchangeable -- read
      the one in ``rolling_results.csv`` as the known-author-pool baseline.
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
    counts = np.asarray(candidate_counts, dtype=float)
    headline["n_known_authors"] = n_known_authors
    headline["n_candidate_authors"] = float(counts.mean())
    headline["random_id"] = [
        random_identity_accuracy(unknown_labels, int(k), counts) for k in headline["top"]
    ]
    headline["advantage"] = headline["id_acc"] - headline["random_id"]
    headline["random_conv"] = [float(np.mean(np.minimum(int(k), counts) / counts))
                               for k in headline["top"]]
    headline["advantage_conv"] = headline["conv_acc"] - headline["random_conv"]
    return headline


def identity_level_scores(headline: pd.DataFrame) -> dict:
    """Flatten the identity-level rows of :func:`closed_set_table` into summary columns.

    The headline table is one row per k while ``rolling_results.csv`` is one row per
    (window, attack), so the identity-level numbers are widened into ``id_acc<k>`` /
    ``random_id<k>`` and land next to the document-level accuracies they should be read against.
    Three views of the same ranking, easy to confuse:

    * ``closed_set_top1`` -- document-weighted: the share of anonymous *documents* attributed
      correctly, which whoever writes the most can carry on their own.
    * ``macro_conv_acc<k>`` -- the same quantity averaged per author, so every user counts once.
    * ``id_acc<k>`` -- an author counts as re-identified if **any one** of their documents puts
      the true author within the top k. The attacker only has to succeed once, so this is the
      highest of the three, and it is the number a privacy claim has to answer.

    ``random_id<k>`` is its matching baseline (a guesser naming k of the ``n_known_authors``
    candidates gets one attempt per document, so it rises with how much an author wrote), and
    ``n_identities`` is the denominator: target authors present in this window *and* on the known
    side. That is not ``n_unknown_authors``, which counts the out-of-set ones too -- on WildChat
    the two differ by a factor of three.

    The document-level columns of ``headline`` are deliberately not copied: ``conv_acc`` at k=1 is
    ``closed_set_top1`` already, and every other k is a row of ``cmc_results.csv`` (to within the
    CMC's more conservative tie convention, which splits them by <0.002 in practice).
    """
    columns: dict = {"n_identities": int(headline["n_identities"].iloc[0])}
    for _, row in headline.iterrows():
        k = int(row["top"])
        columns[f"id_acc{k}"] = float(row["id_acc"])
        columns[f"random_id{k}"] = float(row["random_id"])
    return columns


# --- driver -----------------------------------------------------------------

def split_background(scores: np.ndarray, authors: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Separate the ``--background`` class's column from the real authors' columns.

    Returns ``(author_scores, authors, out_of_set_logit)``. Everything downstream then sees an
    ``[n_documents x n_real_authors]`` matrix with the same meaning it has on a base run, so the
    closed-set tables, the CMC curve and the language filter need no knowledge that a background
    class existed. Only the third return value is new.

    ``out_of_set_logit`` is ``background - logsumexp(authors)``: the model's own log-odds that
    this document belongs to none of the enrolled authors, which is the point of training the
    class at all. It is a **calibrated posterior quantity**, not a margin -- unlike
    :func:`~prompt_anonymity.attacks.rejection_score`, which has to infer "out of set" from how
    peaked the author scores happen to be, and which is kept alongside it so the two can be read
    against each other on the same run.

    Both reductions run in row blocks against the same memory budget as everything else here:
    ``logsumexp`` over the full score matrix is another float64 copy of it if taken in one go.

    **The author block is a slice, not a fancy index**, which is why the background column's
    position is asserted rather than assumed. Dropping one column of a large matrix with a boolean
    mask copies all of it, and the copy would be live alongside the original -- doubling peak
    memory on a job that is already close to its cap. :data:`BACKGROUND_LABEL` begins with ``<``,
    and every real ``author_id`` with a source name, so ``np.unique`` always sorts it to the front
    and the real authors are a contiguous tail.
    """
    column = int(np.flatnonzero(authors == BACKGROUND_LABEL)[0])
    if column != 0:
        raise AssertionError(
            f"{BACKGROUND_LABEL} sorted to column {column} of {len(authors)}, not the front; "
            "the author block is taken as a slice on the assumption that it is contiguous.")
    background = scores[:, column].astype(np.float64)
    author_scores = scores[:, 1:]
    evidence = np.empty(len(scores))
    block_rows = max(1, 2 ** 28 // max(author_scores.shape[1] * 8, 1))
    for start in range(0, len(author_scores), block_rows):
        rows = slice(start, start + block_rows)
        evidence[rows] = logsumexp(author_scores[rows].astype(np.float64), axis=1)
    return author_scores, authors[1:], background - evidence


def run_window(frame: pd.DataFrame, embeddings: np.ndarray, config: KnownConfig,
               attack: str, known: slice, unknown: slice, args: argparse.Namespace,
               tuning_cache: dict, languages: np.ndarray | None = None,
               background: np.ndarray | None = None,
               known_embeddings_source: np.ndarray | None = None,
               known_side_feature=None):
    """Run one (known configuration, attack) combination end to end: calibrate, attribute, score.

    Returns ``(scores, predictions, ood_sweep, headline, cmc, author_report, trials)`` -- a
    one-row summary of the window, the per-document decisions, the accept/reject trade-off curve,
    the closed-set top-k table, the full CMC curve, the per-author risk/retrieval breakdown, and
    the hyper-parameter search (empty unless this call was the one that ran it). With ``--ood
    none`` (the default) the reject option is skipped, ``ood_sweep`` is ``None`` and the summary
    covers only the in-set documents.

    ``tuning_cache`` is passed through to :func:`tuned_settings`; see there for why sharing a
    search between attacks of the same known configuration is sound.

    ``background`` (``--background``, off by default) is a block of feature vectors for documents
    nobody wrote -- see :func:`load_background`. They are appended to the known side under
    :data:`BACKGROUND_LABEL` so the attack fits one extra class, and that class's column is split
    back off (:func:`split_background`) before anything reads the score matrix. Everything the
    run reports about *identification* is therefore computed on a matrix of exactly the same shape
    and meaning as a base run's; what is new is one extra per-document column,
    ``out_of_set_logit``. Only attacks that fit a class per label can use this, which is why
    :func:`parse_args` rejects it for the similarity family rather than silently ignoring it.

    ``languages`` (``--language-aware``, off by default) turns on the candidate filter: the
    attack is still fitted **once** on the whole known side, and the filter then marks the
    authors who do not write a document's language ineligible in that document's row of the score
    matrix. Keeping the fit global is what makes the comparison clean -- the model is byte for
    byte the one an unfiltered run uses, so the difference between the two runs is the pruning
    and nothing else -- and it is also what keeps the cost flat: one fit per window rather than
    one per language group, which is the difference between running and not running on a corpus
    with many distinct language sets.
    """
    known_frame, unknown_frame = frame.iloc[known], frame.iloc[unknown]
    # The two sides may come from DIFFERENT feature matrices. By default the known side is the
    # undefended text (`--known-defense none`) while `embeddings` holds the defended vectors the
    # attacker is querying with, so the slice has to be taken per configuration rather than once:
    # a document that is unknown under `known0025` is known under `known0075`, and each side must
    # be drawn from its own matrix at that boundary. Both matrices are row-aligned to `frame`
    # (joined on `doc_id` at load), so one index means the same document in either.
    known_source = embeddings if known_embeddings_source is None else known_embeddings_source
    known_embeddings, unknown_embeddings = known_source[known], embeddings[unknown]
    # A known-side feature arrives as raw n-gram counts. Its vocabulary and IDF are fitted here,
    # on this configuration's known documents and nothing else, and every other row is only
    # transformed. The counts are kept for the tuner, which refits the transform per fold.
    known_counts, feature_transform = None, None
    if known_side_feature is not None:
        known_counts, feature_transform = known_embeddings, known_side_feature.transformer()
        fitted_transform = clone(feature_transform).fit(known_counts)
        known_embeddings = fitted_transform.transform(known_counts)
        unknown_embeddings = fitted_transform.transform(unknown_embeddings)
    if zscores(args):
        blocks = standardize(known_embeddings, unknown_embeddings,
                             *([] if background is None else [background]))
        known_embeddings, unknown_embeddings = blocks[0], blocks[1]
        background = blocks[2] if background is not None else None
    known_labels = known_frame["author_id"].to_numpy()
    unknown_labels = unknown_frame["author_id"].to_numpy()
    known_authors = np.unique(known_labels)

    in_set = np.isin(unknown_labels, known_authors)
    if not in_set.any():
        raise SystemExit(
            f"{config.tag}: none of the {len(unknown_labels):,} unknown documents "
            f"has an author on the known side, so nothing can be scored."
        )

    # Optionally pick this attack's hyper-parameters, using this window's known side only, and
    # reusing the search across windows that share it. Deliberately before anything touches
    # unknown_embeddings.
    settings, trials = ({}, pd.DataFrame())
    if args.tune:
        settings, trials = tuned_settings(
            attack, config.tag, known_embeddings if known_counts is None else known_counts,
            known_labels, args, tuning_cache, feature_transform)

    # Fit the attack on the known side (documents + labels, all of which the attacker holds) and
    # score every unknown document against every known author.
    factory = build_attack(attack, args, settings)
    if background is None:
        fitted = factory().fit(known_embeddings, known_labels)
        author_scores = fitted.score(unknown_embeddings)      # (n_unknown, n_known_authors)
        authors, out_of_set_logit = fitted.authors, None
    else:
        # One extra class, fitted on documents no known author wrote. The concatenation is the
        # only place the two blocks are held together; it is released as soon as the fit returns,
        # because at WildChat's scale the known side is already 100 MB and the score matrix that
        # follows is measured in gigabytes.
        fitted = factory().fit(
            np.vstack([known_embeddings, background]),
            np.concatenate([known_labels, np.full(len(background), BACKGROUND_LABEL)]),
        )
        # From here on `authors` is the real authors alone, so every table below is the shape a
        # base run produces. `fitted.authors` still carries the background class and is not read.
        author_scores, authors, out_of_set_logit = split_background(
            fitted.score(unknown_embeddings), fitted.authors)

    # Narrow each document's candidate pool to the authors who write its language(s), before
    # anything reads the matrix: the argmax, the ranking and the cohort-normalised rejection
    # score must all see the same pool.
    known_languages = unknown_languages = None
    if languages is not None:
        known_languages, unknown_languages = languages[known], languages[unknown]
        candidate_counts, true_author_is_candidate, language_stats = apply_language_filter(
            author_scores, authors, unknown_languages, known_languages, known_labels,
            true_labels=unknown_labels,
        )
    else:
        candidate_counts = np.full(len(unknown_labels), len(authors), dtype=int)
        true_author_is_candidate = in_set.copy()
        language_stats = {"n_language_groups": 1, "n_language_fallback": 0}

    best = author_scores.argmax(axis=1)
    predicted_author = authors[best]
    normalize = not args.ood_raw_distance
    accept_score = rejection_score(author_scores, normalize)  # higher = more out-of-set

    # Each document's rank for its own author, computed once here and shared by the per-document
    # CSV and every aggregate below, so the two can never disagree about the same document. This
    # is the column that makes the top-k curves *re-derivable from a slice*: top-k accuracy over
    # any subset of documents is just the share of that subset with rank <= k, which is how
    # ``plot_results.py`` rebuilds the shared test set's CMC without re-running the attack. Documents
    # whose author is not on the known side have no rank (no correct answer exists) and stay NaN.
    document_ranks = np.full(len(unknown_labels), np.nan)
    # The in-set rows are taken **once** and the full matrix released. `author_scores[in_set]` is
    # a fresh copy of most of the matrix, so slicing it in several places (here, for the
    # closed-set table, and for the CMC) can put multiple multi-gigabyte arrays live at once and
    # blow the job's memory cap. Nothing below this point reads the full matrix.
    in_set_scores = author_scores[in_set]
    in_set_labels = unknown_labels[in_set]
    del author_scores
    if in_set.any():
        pruned = ~true_author_is_candidate[in_set]      # unattributable: below every candidate
        document_ranks[in_set] = np.where(
            pruned, np.asarray(candidate_counts)[in_set] + 1,
            true_author_ranks(in_set_scores, authors, in_set_labels))

    scores = {
        "known_config": config.tag,
        "attack": attack,
        # The configuration itself, so every downstream reader has the design in the table
        # rather than having to parse it back out of the tag. `gap_weeks` is the axis a
        # staleness figure should be labelled with: a quarter of the corpus is a different
        # number of weeks in each dataset, and a different number at each end of one timeline.
        "known_start": config.start,
        "known_end": config.end,
        "known_size": config.size,
        "gap_fraction": config.gap,
        "gap_weeks": gap_weeks(frame, config, known),
        # Rounded because the search now samples continuous ranges: an unrounded C prints 17
        # digits of a number whose third one is noise. tuning_trials.csv keeps the exact value.
        # Where the settings came from when they were copied rather than searched (--tuned-from);
        # empty for a run that tuned itself or ran untuned.
        "tuned_from": getattr(args, "tuned_from_resolved", ""),
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
        # The pool the attack actually ranked, which --language-aware shrinks per document, and
        # the chance baseline that goes with it. Read every accuracy below against this, not
        # against random_top1_known_pool: the two are the same number without the filter and can
        # differ by two orders of magnitude with it.
        "language_aware": languages is not None,
        "mean_candidate_authors": float(candidate_counts.mean()),
        "candidate_pool_reduction": 1.0 - float(candidate_counts.mean()) / len(known_authors),
        "random_top1_candidate_pool": float(np.mean(1.0 / candidate_counts)),
        # What the filter costs: in-set documents whose true author writes none of their
        # languages on the known side, and so was pruned out of their own document's pool. These
        # are unattributable by construction and count as errors everywhere below.
        "n_true_author_pruned": int((in_set & ~true_author_is_candidate).sum()),
        "true_author_prune_rate": (float((~true_author_is_candidate)[in_set].mean())
                                   if in_set.any() else float("nan")),
        **language_stats,
    }
    predictions = pd.DataFrame({
        # Index of the document in the whole chronologically ordered corpus. This is what lets
        # every configuration be restricted downstream to the *same* documents: the shared test
        # set is `position >= round(test_start * n_documents)`, cut on an integer index with no
        # rounding to reproduce. Since the unknown side always runs to the end of the corpus,
        # `position.max() + 1` recovers n_documents too, so the cut needs nothing but this file.
        "position": np.arange(unknown.start, unknown.stop),
        "doc_id": unknown_frame["doc_id"].to_numpy(),
        "true_author": unknown_labels,
        "author_in_known": in_set,
        "best_author": predicted_author,
        "accept_score": accept_score,
        # Rows stay in timeline order, so any chronological slice of this table -- the shared
        # test set, or one week of it -- is a subset whose top-k is the share with rank <= k.
        "true_author_rank": document_ranks,
        "n_candidate_authors": candidate_counts,
    })
    if languages is not None:
        predictions = predictions.assign(
            languages=["|".join(sorted(document_languages)) for document_languages in unknown_languages],
            true_author_is_candidate=true_author_is_candidate,
        )
    # How well each available statistic tells an out-of-set document from an in-set one, recorded
    # on **every** run rather than only under `--ood reject`. The reject option chooses an
    # operating point; these are threshold-free and are what a base run has to be compared against
    # for `--background` to be readable at all. `margin_*` is the cohort-normalised margin every
    # run already computes -- the same statistic `ood_auroc` reports, under a name that says which
    # of the two it is.
    correctly_ranked = predicted_author == unknown_labels
    scores["margin_auroc"] = detection_auroc(accept_score, ~in_set)
    for far in DIR_FAR_POINTS:
        scores[f"margin_dir_at_far{int(far * 100)}"] = detection_identification_rate(
            accept_score, correctly_ranked, ~in_set, far)
    if out_of_set_logit is not None:
        # The background class's own verdict, kept beside `accept_score` rather than replacing it:
        # the two are different statistics over the same documents (a trained posterior against a
        # cohort-normalised margin) and the point of the run is which one detects a stranger
        # better. Both run the same way round -- higher means more out-of-set.
        predictions = predictions.assign(out_of_set_logit=out_of_set_logit)
        scores["background_size"] = len(background)
        scores["background_auroc"] = detection_auroc(out_of_set_logit, ~in_set)
        for far in DIR_FAR_POINTS:
            scores[f"background_dir_at_far{int(far * 100)}"] = detection_identification_rate(
                out_of_set_logit, correctly_ranked, ~in_set, far)
    ood_sweep = None

    if args.ood != "none":
        folds = open_set_folds(known_labels, held_out_fraction=args.ood_holdout,
                               n_folds=args.ood_folds, seed=args.seed)
        # The forecast replays this configuration's own known:future ratio inside the known
        # side, so a small early known side is not handed a large known side's prior.
        prior = (estimate_ood_prior(known_labels, config.size, 1.0 - config.end)
                 if args.ood_prior is None else args.ood_prior)
        threshold, diagnostics = calibrate_threshold(
            factory, known_embeddings, known_labels, folds,
            calibration=args.ood_calibration, target_far=args.ood_target_far,
            cohort_normalize=normalize, ood_prior=prior, languages=known_languages,
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
        ood_sweep.insert(0, "known_config", config.tag)
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
    in_set_counts = candidate_counts[in_set]
    # `closed_set_table` wants distances, so the scores are negated -- in a temporary that is
    # dropped as soon as it returns, rather than by keeping a second signed copy alive.
    headline = closed_set_table(-in_set_scores, authors, in_set_labels,
                                in_set_counts, args)
    scores.update(identity_level_scores(headline))

    cmc, author_report = closed_set_detail(in_set_scores, authors, in_set_labels,
                                           in_set_counts, document_ranks[in_set],
                                           args, scores)

    # Every per-window table is stamped with the three axes it belongs to, in the same order, so
    # the concatenated CSVs can be grouped or filtered on any of them. `predictions` needs this as
    # much as the rest: its windows share one file, and without a `window` column the rows of three
    # different unknown periods would be indistinguishable once concatenated.
    for table in (headline, cmc, author_report, predictions):
        table.insert(0, "attack", attack)
        table.insert(0, "known_config", config.tag)
    # The search belongs to the known side, not to any one scored document, which is why it is
    # empty on every attack after the first that shares a known fraction.
    if not trials.empty:
        trials.insert(0, "attack", attack)
        trials.insert(0, "known_config", config.tag)
    return scores, predictions, ood_sweep, headline, cmc, author_report, trials


def closed_set_detail(scores_matrix: np.ndarray, authors: np.ndarray, true_authors: np.ndarray,
                      candidate_counts: np.ndarray, ranks: np.ndarray,
                      args: argparse.Namespace, scores: dict):
    """Everything the three-row headline table leaves on the table, from the same score matrix.

    Fills ``scores`` in place with three families of summary numbers and returns the two tables
    that back them:

    * **Whole-ranking** (:func:`~prompt_anonymity.evaluation.metrics.ranking_summary`) -- ``mrr``,
      ``mean_percentile_rank``, ``median_rank``. The headline samples the ranking at three
      cutoffs; these use all of it, and ``mean_percentile_rank`` is the only accuracy-like
      number here that is comparable across windows whose candidate pools differ in size.
      Returned in full as the CMC curve, top-k accuracy at every k.
    * **Author-averaged** -- ``macro_conv_acc<k>`` and ``macro_f1``. The headline's ``conv_acc``
      is document-weighted, so a user who writes a fifth of the corpus can carry it on their
      own; these weight every user equally. The per-author table behind them is the risk
      distribution, which is what a privacy claim should actually rest on.
    * **Retrieval** (:mod:`~prompt_anonymity.evaluation.metrics.retrieval`) -- ``map`` and
      ``mean_r_precision``, running each known author as a *query* against the anonymous
      documents. A different attacker: "find everything this person wrote" rather than "who
      wrote this". Also the only direction in which MAP is not just MRR under another name.

    Calibration of the top-1 confidence (``brier``, ``ece``) goes in too, since every threshold
    in the open-set path assumes that confidence means something.

    ``candidate_counts`` is each document's pool size, which ``--language-aware`` makes
    document-specific; it sets the chance baselines here for the same reason as in
    :func:`closed_set_table`, and matters most for ``mean_percentile_rank``, whose whole purpose
    is to be comparable across differently-sized pools. Without the filter every entry is the
    number of candidate authors and the numbers are the ones the scalar version produced.

    ``ranks`` is each document's rank for its own author, computed by the caller so that the
    per-document CSV and these aggregates are guaranteed to be the same numbers. Documents that
    ``--language-aware`` has made unattributable arrive already placed one past their own pool --
    "below every candidate the attack actually ranked" -- rather than at their position in the
    full author list, which would read as a middling rank and drag ``mean_percentile_rank`` far
    outside [0, 1]. That is a miss at every k, an ~0 contribution to MRR, and the percentile
    floor.
    """
    predicted = authors[scores_matrix.argmax(axis=1)]

    scores.update(ranking_summary(ranks, candidate_counts))
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
    return cmc_curve(ranks, candidate_counts), author_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", default="swe_chat", choices=sorted(SOURCES),
                        help="Which built split to attack (default: swe_chat).")
    parser.add_argument("--feature", default="char_ngram_tfidf",
                        help="Feature parquet to use, i.e. <split>_<feature>.parquet (default: "
                             "char_ngram_tfidf) -- or a feature fitted per known configuration, which "
                             "has no parquet and reads the document text instead: "
                             + ", ".join(sorted(KNOWN_SIDE_FEATURES)) + ".")
    parser.add_argument("--max-ngrams", type=int, default=None, metavar="N",
                        help="For a known-side feature (char_ngram_tfidf): how many n-grams each "
                             "known side keeps, by frequency on that known side (default: the "
                             "feature's own, 3072). A non-default value is appended to the output "
                             "directory as _top<N>, so it never overwrites the default run.")
    parser.add_argument("--known-defense", default="none", choices=sorted(DEFENSES),
                        help="Defense applied to the KNOWN side, i.e. the labelled history the "
                             "attacker already holds (default: none). The default is deliberate "
                             "and it is the threat model this project assumes: a user adopts a "
                             "defense today, so whatever leaked before it is undefended. Set this "
                             "equal to --defense for the matched condition, where the defense has "
                             "always been on -- a strictly easier problem for the attacker, since "
                             "the appended text is then a component both sides share.")
    parser.add_argument("--defense", default="none", choices=sorted(DEFENSES),
                        help="Attack the vectors of text this defense rewrote, i.e. "
                             "<split>_<defense>_<feature>.parquet (default: none, the original "
                             "text). The defense itself runs beforehand -- "
                             "`python -m prompt_anonymity.data.apply_defenses` then "
                             "`compute_features --defense <name>`; this only selects which "
                             "vectors to attack.")
    parser.add_argument("--data-dir", default=str(DATA_DIR),
                        help="Directory holding the parquets to attack (default: data/hf, the "
                             "mirror of the published dataset that `download` writes). Note this "
                             "is NOT data/dist, where build_dataset / compute_features / "
                             "apply_defenses write -- point it there to attack something you just "
                             "built rather than the published copy.")
    parser.add_argument("--metric", default="cosine",
                        help="Distance metric (any scipy cdist metric; default: cosine).")
    parser.add_argument("--known-windows", nargs="+", default=list(DEFAULT_KNOWN_WINDOWS),
                        metavar="XXYY",
                        help="Known sides, each an interval of the timeline written as four "
                             "digits -- start then end, whole percents (0025 = the first "
                             "quarter, 2575 = the middle half). Default: the six quartile-aligned "
                             "intervals that do not touch the held-out test set, which is the "
                             "complete set of them. One experiment per value; every one is "
                             "scored against the same test set, and each scores its whole "
                             "remaining future so the temporal figures stay drawable. Reading "
                             "them by (size, gap to the test set) is the point of the design: "
                             "0025/2550/5075 differ only in staleness, 5075/2575/0075 only in "
                             "volume.")
    parser.add_argument("--test-fraction", type=float, default=DEFAULT_TEST_FRACTION,
                        help="Final share of the timeline held out of every known side and "
                             "shared by every configuration as the set they are compared on "
                             "(default: 0.25). No --known-windows entry may reach into it. "
                             "Changing it makes a run incomparable to the defaults, so it is "
                             "stamped on the output directory name.")
    parser.add_argument("--model-owner", default="all",
                        help="Restrict to one agent provider, e.g. 'Anthropic' (default: all).")
    parser.add_argument("--language", default="all",
                        help="Restrict to one language_primary, e.g. 'English' (default: all).")
    parser.add_argument("--language-aware", action="store_true",
                        help="Narrow each unknown document's candidate pool to the known authors "
                             "who write at least one of its languages (language_primary plus "
                             "language_secondary). Off by default. The attack is still fitted "
                             "once per window on the whole known side; only the candidate columns "
                             "are filtered, so the model is identical to an unfiltered run and "
                             "the difference between the two is the pruning alone. Every chance "
                             "baseline is recomputed against each document's own pool -- read "
                             "'advantage' and random_top1_candidate_pool, because narrowing "
                             "20,000 candidates to 300 raises top-1 on its own. Redundant with "
                             "--language <one language>, which leaves a single language group.")
    parser.add_argument("--undated", default="drop", choices=["drop", "known"],
                        help="What to do with documents that have no ended_at: 'drop' them "
                             "(default) or treat them as the oldest, i.e. always known.")
    parser.add_argument("--standardize", action=argparse.BooleanOptionalAction, default=True,
                        help="Z-score features using known-side statistics before scoring "
                             "(default: on; disable with --no-standardize). Features that mix "
                             "ratios in [0, 1] with raw counts need it, or a handful of "
                             "wide-range columns dominate every method.")
    parser.add_argument("--attacks", nargs="+", default=["logistic"],
                        choices=sorted(ATTRIBUTION_ATTACKS),
                        help="Attack(s) fitted on the known side, from "
                             "prompt_anonymity.attacks.ATTRIBUTION_ATTACKS; every window runs "
                             "each of them (default: logistic). 'cosine' is the unsupervised "
                             "centroid baseline and 'nearest_neighbor' the original attack.")
    parser.add_argument("--background", default="none",
                        choices=["none", *sorted(BACKGROUND_SOURCES)],
                        help="Train an extra 'none of these authors' class on documents sampled "
                             "from another split (default: none). The pool is drawn without "
                             "reference to the split under attack and no unknown document or "
                             "label is read, so this is an attacker-side capability rather than a "
                             "leak -- see load_background. It gives the softmax somewhere to put "
                             "the probability mass a stranger's document would otherwise be "
                             "forced onto the least-unlike enrolled author, and adds an "
                             "out_of_set_logit column to predictions_*.csv beside the existing "
                             "cohort-normalised accept_score. Only the multiclass attacks fit a "
                             "class per label, so this is rejected for the similarity family.")
    parser.add_argument("--background-size", type=int, default=20000,
                        help="Documents drawn from the --background pool (default: 20,000). This "
                             "is the dial for how heavily the extra class counts, but only while "
                             "--balanced is off (the "
                             "default): the raw document count is then its prior. Under "
                             "--balanced every class carries equal total weight, so the "
                             "background pool counts for as much as one single-document author "
                             "however large it is, and the size stops mattering.")
    parser.add_argument("--regularization", type=float, default=1.0,
                        help="Inverse regularisation strength C for the logistic attack (default: 1).")
    parser.add_argument("--balanced", action="store_true",
                        help="Weight known authors equally in the logistic attack. Off by "
                             "default: the document counts are genuinely informative priors.")
    parser.add_argument("--shrinkage", type=float, default=0.2,
                        help="Covariance shrinkage for the wccn / plda attacks (default: 0.2).")
    parser.add_argument("--xgboost-device", default="cpu", choices=["cpu", "cuda", "auto"],
                        help="Where the xgboost attack fits its trees (default: cpu). 'cuda' uses "
                             "the GPU histogram builder, 'auto' does so only if the wheel has "
                             "CUDA and a device is visible. Not the default because the GPU "
                             "builder sums gradients in a different order and can pick different "
                             "splits, which would make a run reproducible only on the same kind "
                             "of machine.")
    parser.add_argument("--tuned-from", default=None, metavar="base|DIR",
                        help="Skip the hyper-parameter search and reuse the settings another run "
                             "selected, per (attack, known configuration), from its "
                             "tuning_trials.csv. 'base' means this experiment's undefended run "
                             "(same source, feature, attack and scope flags, --defense none). "
                             "Sound only with --known-defense none (the default): the search runs "
                             "on the known side, which is then the same undefended text in both "
                             "runs, so it would pick the same settings. Refused otherwise, and "
                             "refused if the source run was never tuned.")
    parser.add_argument("--tune", action=argparse.BooleanOptionalAction, default=True,
                        help="Search each attack's hyper-parameter space (HYPERPARAMETER_SPACES) "
                             "before scoring, by successive halving over randomly sampled "
                             "configurations (default: on; --no-tune uses the defaults instead). "
                             "The search runs on chronological folds *inside the "
                             "known side*, so it never sees the documents it will be evaluated "
                             "on. It runs once per (attack, known configuration) and is never "
                             "shared across configurations, because they hold different (and "
                             "overlapping) documents. Overrides --regularization / --balanced / "
                             "--shrinkage. Writes every (candidate, rung) pair to "
                             "tuning_trials.csv, one block per search, and the chosen settings to "
                             "the hyperparameters column of rolling_results.csv. No-op for the "
                             "attacks with no space to search (nearest_neighbor, cosine), which "
                             "is why leaving it on by default costs those runs nothing.")
    parser.add_argument("--tune-folds", type=int, default=3,
                        help="Chronological folds inside the known side per --tune search "
                             "(default: 3).")
    parser.add_argument("--tune-window", type=float, default=0.25,
                        help="Validation block of each tuning fold, as a fraction of the known "
                             "side (default: 0.25). A share of the known side rather than a copy "
                             "of the outer task, so the search does not have to be repeated for "
                             "every configuration that happens to hold the same documents.")
    parser.add_argument("--tune-candidates", type=int, default=6,
                        help="Configurations sampled per --tune search, i.e. the width of the "
                             "first halving rung (default: 6). This is the main speed dial. A "
                             "first-rung fit is not free -- its cost is dominated by the author "
                             "count rather than the sample size, so it does not shrink in "
                             "proportion to its data share. Raise it when the search is picking "
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
                        help="k the per-window risk summary is reported at, i.e. the "
                             "top<k>_accuracy column of author_report_*.csv (default: 1). Must be "
                             "one of --top-ks.")
    parser.add_argument("--seed", type=int, default=47,
                        help="Seed for the open-set calibration folds and the separability "
                             "diagnostic's sampling.")
    parser.add_argument("--output-dir", default=None,
                        help="Where to write outputs (default: experiments/results/<tag>).")
    args = parser.parse_args()
    if args.sweep_top_k not in args.top_ks:
        raise SystemExit(f"--sweep-top-k {args.sweep_top_k} is not among --top-ks {args.top_ks}; "
                         "it can only summarise a k that was measured.")
    # A background class is a *class*, so only an attack that fits one per label can be handed it.
    # The similarity family summarises each author instead -- a centroid over the background pool
    # would be a meaningless average of unrelated documents, and nearest-neighbour would just
    # return whichever background document happened to be closest. Rejected rather than ignored,
    # so a run cannot report itself as using a background pool that did nothing.
    unusable = sorted(set(args.attacks) - set(MULTICLASS_ATTACKS))
    if args.background != "none" and unusable:
        raise SystemExit(f"--background {args.background} needs an attack that fits a class per "
                         f"label; {', '.join(unusable)} do not. Available: "
                         f"{', '.join(sorted(MULTICLASS_ATTACKS))}.")
    if args.tuned_from is not None:
        if not args.tune:
            raise SystemExit("--tuned-from replaces the search with another run's result; it "
                             "cannot be combined with --no-tune.")
        if args.known_defense != "none":
            raise SystemExit(f"--tuned-from needs --known-defense none: with --known-defense "
                             f"{args.known_defense} the search runs on defended text, which no "
                             f"undefended run has seen, so its choice cannot be copied.")
        if args.tuned_from == "base" and args.defense == "none":
            raise SystemExit("--tuned-from base on an undefended run would copy the run from "
                             "itself; tune it instead.")
    if args.max_ngrams is not None and args.feature not in KNOWN_SIDE_FEATURES:
        raise SystemExit(f"--max-ngrams applies only to {', '.join(sorted(KNOWN_SIDE_FEATURES))}.")
    if args.feature in KNOWN_SIDE_FEATURES:
        # The background pool is read from a feature parquet, which this feature does not have;
        # and --no-standardize would change the directory name for a run that computes exactly
        # what the default one does (see `zscores`).
        if args.background != "none":
            raise SystemExit(f"--background is not supported with {args.feature}.")
        if not args.standardize:
            raise SystemExit(f"--no-standardize has no effect on {args.feature}, which is never "
                             f"z-scored (see zscores); drop the flag.")
    return args


def _percent(fraction: float) -> int:
    """Window fractions as whole percents, for filenames and printed labels."""
    return int(round(fraction * 100))


def output_tag(args: argparse.Namespace) -> str:
    """Short, self-describing directory name for this run's outputs.

    A default run is exactly ``<dataset>_<defense>_<feature>_<attack>`` -- the four axes,
    positionally, with :data:`NO_DEFENSE_TAG` standing in when there is no defense so the shape
    never changes. That is the name ``experiments/plot_results.py`` parses, and a run named this
    way is one it will put on a comparison figure.

    The dataset part is ``args.source`` verbatim, which is why :data:`SOURCES` spells the corpus
    ``swe_chat``: one name for it across the flag, the parquets and the results tree. It also
    leaves the hyphen free -- it is this function's separator for a multi-attack run, so a source
    containing one was a character doing two jobs.

    Every *non-default* choice that changes the numbers is then appended, so two runs that differ
    in any of them cannot overwrite each other's results. Those extra qualifiers deliberately
    take the name out of the comparable set: a language-aware run and a plain one are not two
    points on the same curve, and silently drawing them as if they were would be worse than
    leaving the qualified run out of the figures.
    """
    metric = "" if args.metric == "cosine" else f"_{args.metric}"
    scaled = "" if args.standardize else "_unstandardized"
    owner = "" if args.model_owner.lower() == "all" else f"_{args.model_owner.lower()}"
    language = "" if args.language.lower() == "all" else f"_{args.language.lower()}"
    language_aware = "_langaware" if args.language_aware else ""
    # A background class changes what the attack was trained on, so a background run is not a
    # point on a base run's curve and must not overwrite one. The size rides in the name too: it
    # is the one setting that changes how heavily the extra class is weighted.
    background = ("" if args.background == "none"
                  else f"_bg-{args.background}{args.background_size // 1000}k")
    openset = "" if args.ood == "none" else f"_openset_{args.ood_calibration}"
    fractions = ("" if tuple(args.known_windows) == DEFAULT_KNOWN_WINDOWS
                 else "_k" + "-".join(args.known_windows))
    # The test set is what every configuration is compared on, so a run that held out a different
    # one is not a point on anyone else's curve, however identical the rest of its flags.
    held_out = ("" if abs(args.test_fraction - DEFAULT_TEST_FRACTION) < 1e-9
                else f"_test{_percent(args.test_fraction)}")
    defense = NO_DEFENSE_TAG if args.defense == "none" else args.defense
    # A defended known side is a different experiment, not a point on this one's curve: the
    # attacker's history carries the same appended text as its queries, so what the defense adds
    # is partly a component both sides share. Defended-known runs therefore take a suffix and
    # leave the comparable set, exactly as `--language-aware` and `--background` do.
    known_defense = ("" if args.known_defense == "none"
                     else f"_knowndef-{args.known_defense}")
    attacks = "-".join(args.attacks)
    # A different vocabulary size is a different feature space, so it must not overwrite the
    # default one -- and, like every suffix here, it leaves the comparable set.
    known_side_feature = KNOWN_SIDE_FEATURES.get(args.feature)
    vocabulary = ("" if args.max_ngrams is None
                  or args.max_ngrams == known_side_feature().max_features
                  else f"_top{args.max_ngrams}")
    return (f"{args.source}_{defense}_{args.feature}_{attacks}{vocabulary}{known_defense}"
            f"{owner}{language}{language_aware}{background}{fractions}{held_out}{openset}"
            f"{metric}{scaled}")


def report_window(scores: dict, headline: pd.DataFrame, author_report: pd.DataFrame,
                  args: argparse.Namespace) -> None:
    """Print one (window, attack) combination's results, grouped the way they should be read.

    Order matters here: the open-set block first if it is on, because ``ood_auroc`` decides
    whether any of its operating points mean anything; then the closed-set top-k table; then the
    whole-ranking summary that the table samples; then the per-user views, which routinely tell a
    different story from the document-weighted ones above them.
    """
    tuned = "" if scores["hyperparameters"] == "default" else f"  [tuned: {scores['hyperparameters']}]"
    print(f"\n=== [{scores['attack']}] {scores['known_config']} "
          f"= known {scores['known_start']:.0%}-{scores['known_end']:.0%} "
          f"({scores['known_period']}), {scores['gap_weeks']:.1f} weeks before the test set "
          f"-> unknown rest {1 - scores['known_end']:.0%} "
          f"({scores['unknown_period']}){tuned}")
    print(f"  {scores['n_known_docs']:,} known docs / {scores['n_known_authors']:,} authors  ->  "
          f"{scores['n_unknown_docs']:,} unknown docs / {scores['n_unknown_authors']:,} authors")
    print(f"  of the unknown docs, {scores['n_in_set']:,} have a known author and "
          f"{scores['n_ood']:,} do not ({scores['ood_rate']:.1%} out-of-set)")
    print(f"  within-author distance mean {scores['within_author_mean']:.4f} | "
          f"between-author mean {scores['between_author_mean']:.4f}")

    if scores["language_aware"]:
        print(f"  language-aware: {scores['n_language_groups']} language group(s) narrow the pool "
              f"to {scores['mean_candidate_authors']:,.1f} candidate authors on average "
              f"({scores['candidate_pool_reduction']:.1%} smaller, random top-1 "
              f"{scores['random_top1_candidate_pool']:.4f} vs "
              f"{scores['random_top1_known_pool']:.4f} unfiltered)")
        print(f"  filter cost:   {scores['n_true_author_pruned']:,} of the {scores['n_in_set']:,} "
              f"in-set docs ({scores['true_author_prune_rate']:.1%}) lost their own author to the "
              f"filter and can no longer be attributed"
              + (f"; {scores['n_language_fallback']:,} doc(s) kept the full pool because no known "
                 f"author writes their language" if scores["n_language_fallback"] else ""))

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

    pool = (f"{scores['mean_candidate_authors']:,.1f} in the pool on average"
            if scores["language_aware"] else f"{scores['n_known_authors']} in the pool")
    print(f"  closed-set top-k over the {scores['n_in_set']:,} in-set documents "
          f"({scores['n_identities']} target authors, {pool}):")
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
    known_side_feature = KNOWN_SIDE_FEATURES.get(args.feature)
    if known_side_feature is not None:
        known_side_feature = known_side_feature(**({} if args.max_ngrams is None
                                                    else {"max_features": args.max_ngrams}))
        run_experiment_grid(args, *load_counts(args, known_side_feature), known_side_feature)
        return
    frame, embeddings = load_documents_and_features(
        args.data_dir, args.source, args.feature, args.undated, args.model_owner, args.language,
        args.defense,
    )
    run_experiment_grid(args, frame, embeddings, load_known_embeddings(args, frame))


def load_counts(args: argparse.Namespace, feature) -> tuple[pd.DataFrame, object, object]:
    """``(frame, counts, known_counts)`` for a known-side feature; ``known_counts`` is ``None``
    unless ``--known-defense`` differs from ``--defense``, as for :func:`load_known_embeddings`."""
    defenses = (args.defense,) if args.known_defense == args.defense else (
        args.defense, args.known_defense)
    frame, counts = load_documents_and_counts(args.data_dir, args.source, feature, args.undated,
                                              args.model_owner, args.language, defenses)
    print(f"[{feature.name}] {feature.params()} | {counts[0].shape[1]:,} distinct n-grams in the "
          f"split; the {feature.max_features:,} kept are chosen, and weighted, per known "
          f"configuration from its known documents alone")
    return frame, counts[0], (counts[1] if len(counts) > 1 else None)


def load_known_embeddings(args: argparse.Namespace, frame: pd.DataFrame):
    """The known side's vectors when ``--known-defense`` differs from ``--defense``, else ``None``."""
    # The known side's vectors, when it is defended differently from the unknown side. Loaded as a
    # second matrix rather than spliced here, because which rows are "known" depends on the known
    # configuration and there are six of them. `load_documents_and_features` applies the same
    # ordering and the same filters to both, so row i is the same document in each -- asserted
    # below rather than assumed, since a silent misalignment would attribute documents to the
    # wrong history.
    known_embeddings_source = None
    if args.known_defense != args.defense:
        known_frame, known_embeddings_source = load_documents_and_features(
            args.data_dir, args.source, args.feature, args.undated, args.model_owner,
            args.language, args.known_defense,
        )
        if not known_frame["doc_id"].equals(frame["doc_id"]):
            raise SystemExit(
                f"the known-side and unknown-side feature files do not cover the same documents "
                f"in the same order ({len(known_frame):,} vs {len(frame):,} rows). Both are "
                f"joined to {args.source}.parquet on doc_id, so this means one of them is stale "
                f"-- rebuild it with `compute_features --source {args.source} --defense ...`.")
    return known_embeddings_source


def run_experiment_grid(args: argparse.Namespace, frame: pd.DataFrame, embeddings, known_embeddings_source,
             known_side_feature=None) -> None:
    """Everything after loading: every (known configuration, attack), each written as it finishes."""
    n_features = (embeddings.shape[1] if known_side_feature is None
                  else known_side_feature.max_features)
    print(f"[{args.source}] {len(frame):,} documents x {n_features} {args.feature} features | "
          f"{frame['author_id'].nunique():,} authors | {_period(frame)} | "
          f"defense={args.defense} (known side: {args.known_defense}) | "
          f"attacks={' '.join(args.attacks)}{' | standardized' if zscores(args) else ''}")
    if not args.standardize:
        print("warning: --no-standardize is set. Every attack measured considerably worse without "
              "it (see --standardize --help); this is a diagnostic mode, not a normal run.")

    # The language sets are a property of the corpus, not of a window, so they are built once and
    # sliced per window -- rebuilding a frozenset per document inside every window would repeat
    # the same work up to eight times over.
    # Loaded once for the whole run, not per configuration: the pool is a fixed draw from another
    # split, so re-sampling it per known side would make the configurations differ in their
    # background as well as in their known interval.
    background = None
    if args.background != "none":
        background = load_background(args.data_dir, args.background, args.feature,
                                     args.background_size, args.seed)
        print(f"background: {len(background):,} documents from {args.background} as a trained "
              f"'none of these authors' class (seed {args.seed})")

    languages = None
    if args.language_aware:
        languages = document_language_sets(frame)
        distinct = len({document_languages for document_languages in languages})
        print(f"language-aware: {distinct} distinct language set(s) across the corpus; each "
              f"unknown document will be scored only against known authors who write one of its "
              f"languages")

    results, ood_sweeps, cmcs, all_trials, stems = [], [], [], [], []
    output_dir = (Path(args.output_dir) if args.output_dir
                  else REPO_ROOT / "experiments" / "results" / output_tag(args))
    output_dir.mkdir(parents=True, exist_ok=True)
    # Clear this run's previous per-configuration files **before** the loop, because each one is
    # now written as soon as its configuration finishes rather than at the end. Rewriting alone is
    # not enough: these are named after what the run produced, so a directory that used to emit
    # `<attack>_known25.csv` would leave it beside a fresh `<attack>_known0025.csv`, and
    # `plot_results.py` globs the pattern. Scoped to the two globs this owns, so
    # `rolling_results.csv` and friends are untouched.
    for previous in (*output_dir.glob("predictions_*.csv"),
                     *output_dir.glob("author_report_*.csv")):
        previous.unlink()
    # One hyper-parameter search per (attack, known configuration). Lives here rather than in
    # run_window so its scope is one experiment.
    tuning_cache: dict[tuple[str, str], dict] = {}
    configurations = known_configurations(len(frame), args.known_windows, args.test_fraction)
    # --tuned-from: the cache is filled up front, so every `tuned_settings` call is a hit and no
    # search runs. Nothing else about the run changes.
    if args.tuned_from is not None:
        source = (base_run_directory(args) if args.tuned_from == "base"
                  else Path(args.tuned_from))
        tuning_cache.update(load_tuned_settings(
            source, args.attacks, [config.tag for config, _, _ in configurations]))
        args.tuned_from_resolved = str(source.resolve())
        print(f"--tuned-from: reusing the settings {source.name} selected; no search will run")
    print(f"held-out test set: final {args.test_fraction:.0%} of the timeline "
          f"({len(frame) - int(round((1 - args.test_fraction) * len(frame))):,} documents), "
          f"shared by all {len(configurations)} known configuration(s): "
          + ", ".join(f"{config.tag} (size {config.size:.0%}, gap {config.gap:.0%})"
                      for config, _, _ in configurations))
    for config, known, unknown in configurations:
        for attack in args.attacks:
            outcome = run_window(frame, embeddings, config, attack, known, unknown,
                                 args, tuning_cache, languages, background,
                                 known_embeddings_source, known_side_feature)
            scores, window_predictions, ood_sweep, headline, cmc, author_report, trials = outcome
            results.append(scores)
            cmcs.append(cmc)
            if ood_sweep is not None:
                ood_sweeps.append(ood_sweep)
            if not trials.empty:
                all_trials.append(trials)
            report_window(scores, headline, author_report, args)
            # Written now, not accumulated: holding six configurations' worth of per-document
            # tables while the largest score matrices are live is how a WildChat run reaches the
            # 16 GB cap. It also means a job killed part way leaves the configurations it did
            # finish, rather than nothing.
            stem = f"{attack}_{config.tag}"
            window_predictions.to_csv(output_dir / f"predictions_{stem}.csv", index=False)
            author_report.to_csv(output_dir / f"author_report_{stem}.csv", index=False)
            stems.append(stem)
            del window_predictions, author_report

    all_cmcs = pd.concat(cmcs, ignore_index=True)
    pd.DataFrame(results).to_csv(output_dir / "rolling_results.csv", index=False)
    all_cmcs.to_csv(output_dir / "cmc_results.csv", index=False)
    if ood_sweeps:
        pd.concat(ood_sweeps, ignore_index=True).to_csv(output_dir / "ood_sweep.csv", index=False)
    if all_trials:
        pd.concat(all_trials, ignore_index=True).to_csv(output_dir / "tuning_trials.csv", index=False)

    print(f"\nWrote results to {output_dir}/")
    print("  rolling_results.csv, cmc_results.csv"
          + (", ood_sweep.csv" if ood_sweeps else ""))
    print("  " + ", ".join(f"predictions_{s}.csv" for s in stems))
    print("  " + ", ".join(f"author_report_{s}.csv" for s in stems))
    print("\nNo figures were drawn. To (re)draw every figure in the project from the CSVs:")
    print("  python experiments/plot_results.py")


if __name__ == "__main__":
    main()
