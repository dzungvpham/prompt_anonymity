#!/usr/bin/env python
"""How does open-set re-identification scale with the number of enrolled users?

``run_experiment.py`` measures DIR@FAR on the gallery the corpus happens to have -- 19,711
enrolled authors on WildChat's ``known0075``. A real attacker's gallery could be 100k or 1M people,
and no more text can be embedded to measure that directly. This script answers the question in two
parts and keeps them visibly apart:

1. **Measured, for every gallery size up to the real one.** Authors are shuffled and the gallery is
   cut to its first ``N`` columns, over several random shuffles. Every in-set probe keeps its true
   author enrolled, so all of them are scored at every size and the curve is *paired* across sizes:
   only the number of distractors changes. At ``N = n_authors`` this is exactly the runner's number.

2. **Modelled, for galleries larger than the real one**, under one assumption: new users are
   *exchangeable* with the current ones, so a probe's score against a new author is a fresh draw from
   that probe's own impostor-score distribution. Each probe's genuine score and cohort statistics do
   not move, which is why only **gallery-independent** attacks are allowed (``nearest_neighbor``,
   ``cosine``: an author's score does not depend on who else is enrolled; ``wccn``/``lda``/``plda``
   learn a space from every author and a multiclass attack refits every class).

Growing the gallery hurts the attacker through two channels, and the model treats them differently
because they behave differently under extrapolation:

``ranking``
    An in-set probe must still beat every impostor. With ``Δ`` new authors it survives with
    probability ``(1 − T(g))^Δ``, where ``g`` is its genuine z-score and ``T`` the chance one impostor
    beats it. ``T`` lies past the last observed impostor, so each probe's top-``K`` impostor z-scores
    are fitted with an **exponential (Gumbel-domain) tail** whose scale is **empirical-Bayes shrunk**
    across probes (:func:`fit_impostor_tails`). Backtested from a 400-author gallery -- 49×, the same
    ratio as 19.7k → 1M -- it predicts the measured closed-set top-1 at 19,711 to within ~0.015 on
    average over shuffles (0.274 against 0.261).

``threshold``
    The 10%-FAR cut-off must rise as strangers' best matches climb. Per-probe tails **over-predict**
    that climb: fitted on the full gallery they are inconsistent with the gallery's own stranger
    maxima (implied cut-off 8.5–8.7 z against an observed 7.97), the excess concentrated in the
    out-of-set probes with the widest tails -- the signature of near-duplicate prompts bunching under
    a ceiling. So the cut-off is extrapolated from **its own measured curve** instead
    (:func:`threshold_curve`), linear in ``ln N`` (central) or in ``sqrt(ln N)`` (the Gaussian-tail
    form, which predicts the cut-off itself best).

The **central estimate** (``hybrid_log``) combines the two. **Read its backtest as leaning high at
the largest step.** Mean relative error against the measured 19,711, over five shuffles
(WildChat/Gemini/``nearest_neighbor``, 2026-11-07): **+42% at 49×** (per shuffle +13% to +71%),
+16 to +18% at 20×, −1 to +4% at 10×, +8 to +12% at 4×, with ±20–40% between shuffles at every
ratio -- which 400 authors land in a sub-gallery matters a lot. An earlier, independent set of five
shuffles came out within 7% at every ratio, so neither set alone is the error. Two biases
partly cancel: the ranking model overstates DIR by 20–40% even at the true cut-off (it
concentrates survival on the highest-scoring probes), and the ``ln N`` cut-off usually, not
always, overshoots. ``hybrid_sqrt_log`` (cut-off closest, DIR highest) and ``tail_model`` (both
channels from the tails; fitted on the full gallery its cut-off is inconsistent with that
gallery's own maxima, so it sits lowest) bracket it. The range reported is over those three
estimators × ``K`` ∈ ``EXCEEDANCE_SENSITIVITY``.

Things that were tried and are not here (do not re-add without reading why): one pooled GPD shape
(the joint MLE is biased when each probe contributes few exceedances -- an incidental-parameter
problem); a shape per decile of tail scale (worse); per-class shapes calibrated to reproduce the
realized maxima at the fitting size (worse extrapolation on both channels).

Baselines
---------
Both are a **chance detector in front of a guesser that reads no text**, as in
``plot_results.py``'s ``openset/dirfar`` family: a detector with no information keeps ``FAR`` of the
in-set documents too, so the baseline DIR is ``FAR × (guesser's top-1)``.

* ``baseline_proportional`` -- the project's dashed line: the guesser names author ``a`` with
  probability equal to ``a``'s share of the gallery's known documents. Beyond the real gallery the
  new authors carry the current mean document count, so a share is ``n_a / (n̄·N)``.
* ``baseline_uniform`` -- ``FAR / N``, carried for reference as every results CSV does.

Outputs, under ``experiments/gallery_scaling/<dataset>_<defense>_<feature>_<attack>/``:

* ``scaling.csv`` -- one row per gallery size: measured below the real gallery, modelled above,
  with the 95% user-clustered bootstrap interval, the estimator range, closed-set top-1, the z
  cut-off and both baselines.
* ``estimators.csv`` -- every modelled estimator × ``K`` × size, the source of the range.
* ``backtest.csv`` -- each estimator fitted on a sub-gallery and scored against the measured real
  gallery, per shuffle.

**Scope**: document-level DIR only (the per-user "identified at least once" version needs a user's
documents to share their new impostors, which this model draws independently); one known window;
every operating point is an oracle one, as everywhere in the project (all runs are ``--ood none``).
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from prompt_anonymity.attacks import ATTRIBUTION_ATTACKS  # noqa: E402

from run_experiment import (  # noqa: E402
    DATA_DIR,
    NO_DEFENSE_TAG,
    known_configurations,
    load_documents_and_features,
    standardize,
)

OUTPUT_ROOT = REPO_ROOT / "experiments" / "gallery_scaling"

#: Attacks whose score for an author depends on that author's documents alone, so adding authors
#: leaves every existing score unchanged -- the property both the sub-gallery measurement and the
#: exchangeable-addition model rest on.
GALLERY_INDEPENDENT_ATTACKS = ("nearest_neighbor", "cosine")

#: Sub-gallery sizes measured; the real gallery size is always appended.
MEASURED_SIZES = (50, 100, 200, 400, 1_000, 2_000, 5_000, 10_000)

#: Gallery sizes modelled beyond the real one.
TARGET_SIZES = (50_000, 100_000, 200_000, 500_000, 1_000_000)

#: Tail size ``K`` for the central estimate, and the values the reported range spans.
EXCEEDANCES = 100
EXCEEDANCE_SENSITIVITY = (50, 100, 200, 400)

#: The cut-off curve is fitted on measured sizes in ``[N / span, N]`` -- the decade nearest the
#: extrapolation, where the curve's slope is the one that continues.
THRESHOLD_FIT_SPAN = 10

#: Sub-gallery fits the backtest runs, each with the tail sizes it can afford (``K`` must stay well
#: under the gallery size), predicting the measured value at the real gallery.
BACKTEST_FITS = {400: (20,), 1_000: (20, 50), 2_000: (20, 40, 100), 5_000: (50, 100)}

#: Probe rows processed per block, which bounds the float64 temporaries to ~300 MB.
ROW_BLOCK = 2048

#: Clip for per-probe tail probabilities inside ``log1p(-p)``.
_PROBABILITY_CEILING = 1 - 1e-15


# --- the score matrix ------------------------------------------------------------------------


def score_matrix(args: argparse.Namespace):
    """The attack's ``[n_unknown × n_known_authors]`` scores on one known window, with labels.

    Reproduces ``run_experiment.py`` exactly -- same loader, ordering, known/unknown slice and
    known-side z-score -- so that at the full gallery every number here equals the runner's (checked:
    ``accept_score`` to 2e-15 and identical ranks on WildChat/Gemini/``nearest_neighbor``).
    ``--score-cache`` keeps the matrix between runs; it is ~3.4 GB on WildChat, so point it at node-
    local scratch, not the project filesystem.
    """
    cache = Path(args.score_cache) if args.score_cache else None
    if cache is not None and (cache / "scores.npy").exists():
        meta = pd.read_parquet(cache / "probes.parquet")
        gallery = pd.read_parquet(cache / "gallery.parquet")
        return np.load(cache / "scores.npy"), gallery, meta

    frame, embeddings = load_documents_and_features(args.data_dir, args.source, args.feature,
                                                    defense=args.defense)
    (config, known, unknown), = known_configurations(len(frame), [args.known_window],
                                                     args.test_fraction)
    known_x, unknown_x = standardize(embeddings[known], embeddings[unknown])
    del embeddings
    known_labels = frame["author_id"].iloc[known].to_numpy()
    settings = {"metric": "cosine"} if args.attack == "nearest_neighbor" else {}
    attack = ATTRIBUTION_ATTACKS[args.attack](**settings).fit(known_x, known_labels)
    del known_x
    scores = attack.score(unknown_x)
    gallery = pd.DataFrame({"author_id": attack.authors})
    gallery["n_known_docs"] = (pd.Series(known_labels).value_counts()
                               .reindex(attack.authors).to_numpy())
    meta = pd.DataFrame({"doc_id": frame["doc_id"].iloc[unknown].to_numpy(),
                         "author_id": frame["author_id"].iloc[unknown].to_numpy()})
    if cache is not None:
        cache.mkdir(parents=True, exist_ok=True)
        np.save(cache / "scores.npy", scores)
        gallery.to_parquet(cache / "gallery.parquet")
        meta.to_parquet(cache / "probes.parquet")
    return scores, gallery, meta


@dataclass
class ProbeTable:
    """The score matrix with its columns in one gallery order and each genuine cell set aside.

    ``impostors`` holds every probe's scores against the gallery in shuffle order, with the in-set
    probes' genuine author replaced by ``-inf`` so a row maximum is always an impostor's. A
    sub-gallery of size ``n`` is the first ``n`` columns **plus, for an in-set probe, its own author**
    wherever that author falls -- which is what keeps every in-set probe scored at every size.
    """

    impostors: np.ndarray           # float32 [n_probes, n_authors]
    genuine: np.ndarray             # float64 [n_probes], nan for out-of-set probes
    genuine_column: np.ndarray      # genuine author's column in shuffle order, -1 out-of-set
    in_set: np.ndarray
    probe_author: np.ndarray        # the bootstrap's resampling unit
    gallery_docs: np.ndarray        # known-side documents per column, shuffle order

    @classmethod
    def build(cls, scores, gallery: pd.DataFrame, probe_authors, order) -> "ProbeTable":
        column = {author: j for j, author in enumerate(gallery["author_id"])}
        original = np.array([column.get(author, -1) for author in probe_authors])
        in_set = original >= 0
        position = np.empty_like(order)
        position[order] = np.arange(len(order))
        genuine_column = np.where(in_set, position[np.maximum(original, 0)], -1)
        impostors = np.empty_like(scores)
        for start in range(0, len(scores), ROW_BLOCK):
            impostors[start:start + ROW_BLOCK] = scores[start:start + ROW_BLOCK][:, order]
        rows = np.flatnonzero(in_set)
        genuine = np.full(len(scores), np.nan)
        genuine[rows] = impostors[rows, genuine_column[rows]]
        impostors[rows, genuine_column[rows]] = -np.inf
        return cls(impostors, genuine, genuine_column, in_set, np.asarray(probe_authors),
                   gallery["n_known_docs"].to_numpy()[order])


@dataclass
class GalleryView:
    """Per-probe statistics of the sub-gallery made of the first ``n`` columns."""

    n: int
    best_impostor: np.ndarray       # raw score
    cohort_mean: np.ndarray
    cohort_std: np.ndarray
    n_impostors: np.ndarray


def gallery_view(table: ProbeTable, n: int) -> GalleryView:
    """Best impostor and cohort statistics over the first ``n`` columns (plus the genuine author).

    The cohort is what ``rejection_score`` normalises over -- every candidate the probe was ranked
    against, genuine author included -- so at ``n = n_authors`` the z-scores are the runner's.
    """
    size = len(table.genuine)
    best, mean, std, count = (np.empty(size) for _ in range(4))
    for start in range(0, size, ROW_BLOCK):
        rows = slice(start, start + ROW_BLOCK)
        block = table.impostors[rows, :n].astype(np.float64)
        finite = np.isfinite(block)
        best[rows] = block.max(axis=1)
        block[~finite] = 0.0
        k = finite.sum(axis=1).astype(float)
        genuine = table.genuine[rows]
        has_genuine = np.isfinite(genuine)
        g = np.where(has_genuine, genuine, 0.0)
        members = k + has_genuine
        mu = (block.sum(axis=1) + g) / members
        mean[rows] = mu
        std[rows] = np.sqrt(np.maximum(((block * block).sum(axis=1) + g * g) / members - mu * mu, 0))
        count[rows] = k
    return GalleryView(n, best, mean, np.where(std > 0, std, 1.0), count)


# --- measurement -----------------------------------------------------------------------------


def threshold_at_far(enrolledness: np.ndarray, is_ood: np.ndarray, far: float,
                     weights: np.ndarray | None = None) -> float:
    """The most permissive z cut-off (accept when ``z >= t``) whose false accepts stay within budget.

    Unweighted it is ``plot_results.threshold_at_far`` on ``accept_score = -z``: walk documents
    from most to least enrolled-looking and keep the longest prefix holding at most
    ``floor(far · n_ood)`` out-of-set documents. Weighted (a bootstrap replicate) the budget is the
    weighted share itself, since a floor of a weighted count means nothing.
    """
    order = np.argsort(-enrolledness, kind="stable")
    w = np.ones(len(enrolledness)) if weights is None else weights
    seen = np.cumsum((w * is_ood)[order])
    budget = np.floor(far * seen[-1]) if weights is None else far * seen[-1]
    keep = seen <= budget + 1e-12
    if not keep.any():
        return np.inf
    return float(enrolledness[order[np.flatnonzero(keep)[-1]]])


def measure(table: ProbeTable, view: GalleryView, far: float, mean=None, std=None,
            weights: np.ndarray | None = None) -> dict:
    """DIR@FAR, closed-set top-1 and the z cut-off on one sub-gallery.

    ``mean``/``std`` override the sub-gallery's own cohort statistics -- the backtest and the
    cut-off curve use the full gallery's, which is what a large gallery's would converge to, so a
    small sub-gallery's noisy row deviation does not leak into a curve meant for extrapolation.
    """
    mean = view.cohort_mean if mean is None else mean
    std = view.cohort_std if std is None else std
    best = np.where(table.in_set, np.fmax(table.genuine, view.best_impostor), view.best_impostor)
    z = (best - mean) / std
    correct = table.in_set & (table.genuine > view.best_impostor)   # a tie is not a top-1 hit
    t = threshold_at_far(z, ~table.in_set, far, weights)
    w = np.ones(len(z)) if weights is None else weights
    in_set_weight = w[table.in_set].sum()
    return {"n_gallery": view.n, "threshold_z": t,
            "top1": (w * correct)[table.in_set].sum() / in_set_weight,
            "dir": (w * (correct & (z >= t)))[table.in_set].sum() / in_set_weight}


def proportional_top1(table: ProbeTable, n: int) -> float:
    """Top-1 of a guesser naming each enrolled author in proportion to their known documents.

    On a sub-gallery an in-set probe's own author is enrolled wherever it falls in the shuffle, so
    its share is ``n_a / (docs in the first n columns + n_a if a lies beyond them)``.
    """
    rows = np.flatnonzero(table.in_set)
    own = table.gallery_docs[table.genuine_column[rows]].astype(float)
    total = table.gallery_docs[:n].sum() + np.where(table.genuine_column[rows] >= n, own, 0.0)
    return float(np.mean(own / total))


def author_weights(probe_author: np.ndarray, replicates: int, rng) -> np.ndarray:
    """``[B, n_probes]`` clustered-bootstrap weights: resample users, carry all their documents."""
    codes, inverse = np.unique(probe_author, return_inverse=True)
    draws = rng.multinomial(len(codes), np.full(len(codes), 1.0 / len(codes)), size=replicates)
    return draws[:, inverse].astype(np.float64)


# --- the impostor-tail model -----------------------------------------------------------------


@dataclass
class TailFit:
    """Each probe's impostor z-score tail: exponential above its ``(K+1)``-th largest impostor.

    ``P(one impostor > x) = tail_fraction · exp(-(x − u) / scale)`` for ``x > u``. ``best`` and
    ``genuine`` are the probe's current best impostor and genuine score in the same z units.
    """

    u: np.ndarray
    tail_fraction: np.ndarray
    scale: np.ndarray
    best: np.ndarray
    genuine: np.ndarray
    in_set: np.ndarray

    def beats(self, x: np.ndarray, rows) -> np.ndarray:
        """P(one new impostor scores above ``x``) for probes ``rows``; ``x`` must be above ``u``."""
        excess = np.maximum(x - self.u[rows], 0.0)
        return self.tail_fraction[rows] * np.exp(-excess / self.scale[rows])


def shrink_log_scale(scale: np.ndarray, exceedances: int) -> np.ndarray:
    """Empirical-Bayes shrinkage of every probe's log tail scale toward the mean over probes.

    An exponential scale fitted to ``K`` exceedances has a log with sampling variance ``≈ 1/K``;
    whatever spread the probes show beyond that is taken as real. Without this, sampling noise
    inflates the widest tails, and those dominate the stranger maxima that set the FAR cut-off.
    """
    positive = scale > 0
    log_scale = np.log(np.where(positive, scale, 1.0))
    center, total = log_scale[positive].mean(), log_scale[positive].var()
    noise = 1.0 / exceedances
    signal = max(total - noise, 0.0)
    shrunk = center + signal / (signal + noise) * (log_scale - center)
    return np.where(positive, np.exp(shrunk), scale)


def fit_impostor_tails(table: ProbeTable, view: GalleryView, exceedances: int,
                       mean: np.ndarray, std: np.ndarray) -> TailFit:
    """Fit every probe's tail to its top ``K`` impostor z-scores over the first ``view.n`` columns.

    The scale is the mean excess over the ``(K+1)``-th largest, which is the exponential MLE, then
    shrunk (:func:`shrink_log_scale`). ``mean``/``std`` are the cohort statistics the z-scores use;
    see :func:`measure` for why the backtest passes the full gallery's.
    """
    k = exceedances
    top = np.empty((len(mean), k + 1))
    for start in range(0, len(mean), ROW_BLOCK):
        rows = slice(start, start + ROW_BLOCK)
        block = table.impostors[rows, :view.n].astype(np.float64)
        largest = -np.partition(-block, k, axis=1)[:, :k + 1]
        top[rows] = (np.sort(largest, axis=1) - mean[rows, None]) / std[rows, None]
    u = top[:, 0]
    scale = shrink_log_scale((top[:, 1:] - u[:, None]).mean(axis=1), k)
    return TailFit(u=u, tail_fraction=k / view.n_impostors, scale=np.maximum(scale, 1e-9),
                   best=(view.best_impostor - mean) / std, genuine=(table.genuine - mean) / std,
                   in_set=table.in_set.copy())


def survival(fit: TailFit, new_authors: float) -> np.ndarray:
    """Per in-set probe: P(still top-1 after ``new_authors`` exchangeable authors are enrolled)."""
    rows = np.flatnonzero(fit.in_set)
    genuine = fit.genuine[rows]
    correct_now = genuine > fit.best[rows]
    beaten = fit.beats(np.where(correct_now, genuine, fit.u[rows] + 1.0), rows)
    return np.where(correct_now,
                    np.exp(new_authors * np.log1p(-np.minimum(beaten, _PROBABILITY_CEILING))), 0.0)


def hybrid_dir(fit: TailFit, new_authors: float, threshold: float,
               weights: np.ndarray | None = None) -> np.ndarray:
    """DIR from the ranking model at a given cut-off: mean of survival × 1[genuine z ≥ cut-off]."""
    rows = np.flatnonzero(fit.in_set)
    hit = survival(fit, new_authors) * (fit.genuine[rows] >= threshold)
    w = (np.ones((1, len(fit.in_set))) if weights is None else np.atleast_2d(weights))[:, rows]
    return (w * hit).sum(axis=1) / w.sum(axis=1)


def tail_model(fit: TailFit, new_authors: float, far: float, iterations: int = 80):
    """DIR, top-1 and cut-off with **both** channels from the tails -- the low end of the range.

    The cut-off solves expected FAR = ``far``: a stranger already above ``t`` is a false accept; one
    below is one with probability ``1 − (1 − T(t))^Δ``. Known to over-predict the cut-off (see the
    module docstring), so it is reported as a bound, never as the estimate.
    """
    out = np.flatnonzero(~fit.in_set)
    best = fit.best[out]

    def expected_far(t):
        beaten = fit.beats(np.full(len(out), t) if np.isscalar(t) else t, out)
        stays_below = np.exp(new_authors * np.log1p(-np.minimum(beaten, _PROBABILITY_CEILING)))
        return np.mean(np.where(best >= t, 1.0, 1.0 - stays_below))

    lo, hi = float(np.min(best)) - 1.0, float(np.max(best)) + 1.0
    while expected_far(hi) > far:
        hi *= 2.0
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if expected_far(mid) > far else (lo, mid)
    return float(hybrid_dir(fit, new_authors, hi)[0]), float(survival(fit, new_authors).mean()), hi


#: How the cut-off is extrapolated: a straight line in the transformed gallery size.
THRESHOLD_FORMS = {
    "log": np.log,                                      # exponential tail: max grows like ln N
    "sqrt_log": lambda n: np.sqrt(np.log(n)),           # Gaussian tail: like sqrt(2 ln N)
}


def threshold_curve(sizes, thresholds, form: str):
    """Least-squares line of the measured cut-off against ``THRESHOLD_FORMS[form](N)``."""
    transform = THRESHOLD_FORMS[form]
    slope, intercept = np.polyfit(transform(np.asarray(sizes, float)), np.asarray(thresholds), 1)
    return lambda n: float(intercept + slope * transform(float(n)))


#: Modelled estimators, central first.
ESTIMATORS = ("hybrid_log", "hybrid_sqrt_log", "tail_model")


def modelled(fit: TailFit, new_authors: float, curves: dict, n: int, far: float) -> dict:
    """Every estimator's (DIR, top-1, cut-off) at gallery size ``n``."""
    top1 = float(survival(fit, new_authors).mean())
    out = {}
    for form, curve in curves.items():
        t = curve(n)
        out[f"hybrid_{form}"] = (float(hybrid_dir(fit, new_authors, t)[0]), top1, t)
    out["tail_model"] = tail_model(fit, new_authors, far)
    return out


# --- driver ----------------------------------------------------------------------------------


def run(args: argparse.Namespace) -> None:
    start = time.time()
    scores, gallery, probes = score_matrix(args)
    n_authors = len(gallery)
    sizes = [n for n in MEASURED_SIZES if n < n_authors] + [n_authors]
    curve_sizes = [n for n in sizes if n >= n_authors / THRESHOLD_FIT_SPAN]
    fits = {n: ks for n, ks in BACKTEST_FITS.items() if n < n_authors}
    rng = np.random.default_rng(args.seed)
    weights = author_weights(probes["author_id"].to_numpy(), args.bootstrap, rng)
    print(f"{len(probes):,} probes x {n_authors:,} authors  ({time.time() - start:.0f}s)", flush=True)

    measured, fixed_thresholds, baselines, backtest = [], [], [], []
    full_table = None
    for draw in range(args.gallery_draws):
        order = rng.permutation(n_authors)
        table = ProbeTable.build(scores, gallery, probes["author_id"].to_numpy(), order)
        full = gallery_view(table, n_authors)
        fixed_by_size = {}
        for n in sizes:
            view = full if n == n_authors else gallery_view(table, n)
            point = measure(table, view, args.far)
            # Each replicate resamples users AND takes its gallery from one shuffle, so the band
            # covers which distractors were drawn as well as which users were.
            mine = weights[draw::args.gallery_draws]
            reps = [measure(table, view, args.far, weights=w) for w in mine]
            measured.append(point | {"draw": draw,
                                     "dir_reps": [r["dir"] for r in reps],
                                     "top1_reps": [r["top1"] for r in reps]})
            fixed = measure(table, view, args.far, full.cohort_mean, full.cohort_std)
            fixed_by_size[n] = fixed["threshold_z"]
            fixed_thresholds.append({"draw": draw, "n_gallery": n, "threshold_z": fixed["threshold_z"]})
            baselines.append({"draw": draw, "n_gallery": n, "proportional_top1": proportional_top1(table, n),
                              "uniform_top1": float(np.mean(1.0 / (n + (table.genuine_column[table.in_set] >= n))))})
        if not args.no_backtest:
            backtest += backtest_draw(table, full, fits, fixed_by_size, args.far, draw)
        if draw == 0:
            full_table = (table, full)
        else:
            del table
        print(f"gallery draw {draw} done ({time.time() - start:.0f}s)", flush=True)
    del scores

    table, full = full_table
    thresholds = pd.DataFrame(fixed_thresholds).groupby("n_gallery").threshold_z.mean()
    curves = {form: threshold_curve(curve_sizes, thresholds.loc[curve_sizes].to_numpy(), form)
              for form in THRESHOLD_FORMS}

    # The modelled sizes: every estimator at every K (the range), the central one with its band.
    rows = []
    for k in EXCEEDANCE_SENSITIVITY:
        fit = fit_impostor_tails(table, full, k, full.cohort_mean, full.cohort_std)
        for n in TARGET_SIZES:
            for estimator, (d, top1, t) in modelled(fit, n - n_authors, curves, n, args.far).items():
                row = {"estimator": estimator, "exceedances": k, "n_gallery": n,
                       "dir": d, "top1": top1, "threshold_z": t}
                if k == args.exceedances and estimator == ESTIMATORS[0]:
                    band = hybrid_dir(fit, n - n_authors, t, weights)
                    surv = survival(fit, n - n_authors)
                    w_in = weights[:, fit.in_set]
                    top1_band = (w_in * surv).sum(axis=1) / w_in.sum(axis=1)
                    row |= {"dir_ci_low": np.percentile(band, 2.5), "dir_ci_high": np.percentile(band, 97.5),
                            "top1_ci_low": np.percentile(top1_band, 2.5),
                            "top1_ci_high": np.percentile(top1_band, 97.5)}
                rows.append(row)
    estimators = pd.DataFrame(rows)

    summary = scaling_table(measured, baselines, estimators, gallery, n_authors, args)
    out = OUTPUT_ROOT / output_tag(args)
    out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out / "scaling.csv", index=False)
    estimators.to_csv(out / "estimators.csv", index=False)
    if backtest:
        pd.DataFrame(backtest).to_csv(out / "backtest.csv", index=False)
    with pd.option_context("display.width", 200):
        print(summary.round(4).to_string(index=False))
    print(f"wrote {out}  ({time.time() - start:.0f}s)")


def backtest_draw(table: ProbeTable, full: GalleryView, fits: dict, fixed_thresholds: dict,
                  far: float, draw: int):
    """Fit every estimator on a sub-gallery of this shuffle and score it against the real gallery.

    Everything an estimator sees is restricted to the sub-gallery: its tails, its current best
    impostors and the cut-off curve (fitted on measured sizes in ``[n_fit / span, n_fit]``). The
    full gallery's cohort statistics are used throughout, as the real extrapolation's are already
    converged. ``in_set_at_true_threshold`` isolates the ranking channel. ``fixed_thresholds`` maps
    each measured size to its cut-off under those statistics.
    """
    n_authors = full.n
    truth = measure(table, full, far)
    fixed = {n: t for n, t in fixed_thresholds.items() if n < n_authors}
    rows = []
    for n_fit, ks in fits.items():
        view = gallery_view(table, n_fit)
        span = [n for n in fixed if n_fit / THRESHOLD_FIT_SPAN <= n <= n_fit]
        curves = {form: threshold_curve(span, [fixed[n] for n in span], form) for form in THRESHOLD_FORMS}
        for k in ks:
            fit = fit_impostor_tails(table, view, k, full.cohort_mean, full.cohort_std)
            predicted = modelled(fit, n_authors - n_fit, curves, n_authors, far)
            predicted["in_set_at_true_threshold"] = (
                float(hybrid_dir(fit, n_authors - n_fit, truth["threshold_z"])[0]),
                float(survival(fit, n_authors - n_fit).mean()), truth["threshold_z"])
            for estimator, (d, top1, t) in predicted.items():
                rows.append({"draw": draw, "n_fit": n_fit, "ratio": n_authors / n_fit, "exceedances": k,
                             "estimator": estimator, "dir": d, "top1": top1, "threshold_z": t,
                             "dir_measured": truth["dir"], "top1_measured": truth["top1"],
                             "threshold_measured": truth["threshold_z"],
                             "dir_relative_error": d / truth["dir"] - 1.0})
    return rows


def scaling_table(measured, baselines, estimators, gallery, n_authors, args) -> pd.DataFrame:
    """One row per gallery size: measured up to the real gallery, the central estimate beyond it."""
    rows = []
    frame = pd.DataFrame(measured)
    base = pd.DataFrame(baselines).groupby("n_gallery")[["proportional_top1", "uniform_top1"]].mean()
    for n, group in frame.groupby("n_gallery"):
        dir_reps = np.concatenate(group["dir_reps"].to_list())
        top1_reps = np.concatenate(group["top1_reps"].to_list())
        rows.append({"n_gallery": n, "source": "measured", "gallery_draws": len(group),
                     "dir": group["dir"].mean(),
                     "dir_ci_low": np.percentile(dir_reps, 2.5), "dir_ci_high": np.percentile(dir_reps, 97.5),
                     "dir_range_low": np.nan, "dir_range_high": np.nan,
                     "top1": group["top1"].mean(),
                     "top1_ci_low": np.percentile(top1_reps, 2.5), "top1_ci_high": np.percentile(top1_reps, 97.5),
                     "threshold_z": group["threshold_z"].mean(),
                     "baseline_proportional": args.far * base.loc[n, "proportional_top1"],
                     "baseline_uniform": args.far * base.loc[n, "uniform_top1"]})
    # Beyond the real gallery, a new author carries the current mean document count, so an
    # enrolled author's share of the gallery's documents is n_a / (n̄ · N).
    share_now = base.loc[n_authors, "proportional_top1"]
    central = estimators[(estimators.estimator == ESTIMATORS[0])
                         & (estimators.exceedances == args.exceedances)].set_index("n_gallery")
    for n in TARGET_SIZES:
        spread = estimators[estimators.n_gallery == n]["dir"]
        row = central.loc[n]
        rows.append({"n_gallery": n, "source": "modelled", "gallery_draws": np.nan,
                     "dir": row["dir"], "dir_ci_low": row["dir_ci_low"], "dir_ci_high": row["dir_ci_high"],
                     "dir_range_low": spread.min(), "dir_range_high": spread.max(),
                     "top1": row["top1"], "top1_ci_low": row["top1_ci_low"], "top1_ci_high": row["top1_ci_high"],
                     "threshold_z": row["threshold_z"],
                     "baseline_proportional": args.far * share_now * n_authors / n,
                     "baseline_uniform": args.far / n})
    return pd.DataFrame(rows)


def output_tag(args: argparse.Namespace) -> str:
    """``<dataset>_<defense>_<feature>_<attack>``, plus any non-default window, test share or FAR."""
    defense = NO_DEFENSE_TAG if args.defense == "none" else args.defense
    tag = f"{args.source}_{defense}_{args.feature}_{args.attack}"
    if args.known_window != "0075":
        tag += f"_known{args.known_window}"
    if args.test_fraction != 0.25:
        tag += f"_test{round(args.test_fraction * 100):02d}"
    if args.far != 0.10:
        tag += f"_far{round(args.far * 100):02d}"
    return tag


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", default="wildchat")
    parser.add_argument("--feature", default="gemini_embedding_2")
    parser.add_argument("--defense", default="none")
    parser.add_argument("--attack", default="nearest_neighbor", choices=GALLERY_INDEPENDENT_ATTACKS)
    parser.add_argument("--data-dir", default=DATA_DIR, type=Path)
    parser.add_argument("--known-window", default="0075",
                        help="one interval in run_experiment.py's XXYY spelling (default: 0075)")
    parser.add_argument("--test-fraction", default=0.25, type=float)
    parser.add_argument("--far", default=0.10, type=float, help="false-accept budget (default: 0.10)")
    parser.add_argument("--gallery-draws", default=5, type=int,
                        help="random author shuffles the sub-galleries are cut from (default: 5)")
    parser.add_argument("--bootstrap", default=200, type=int,
                        help="user-clustered bootstrap replicates (default: 200)")
    parser.add_argument("--exceedances", default=EXCEEDANCES, type=int, choices=EXCEEDANCE_SENSITIVITY,
                        help=f"tail size K of the central estimate (default: {EXCEEDANCES})")
    parser.add_argument("--no-backtest", action="store_true", help="skip the sub-gallery backtest")
    parser.add_argument("--score-cache", default=None,
                        help="directory to keep the score matrix in between runs (use node-local scratch)")
    parser.add_argument("--seed", default=0, type=int)
    args = parser.parse_args()
    if args.bootstrap % args.gallery_draws:
        parser.error("--bootstrap must be a multiple of --gallery-draws (replicates are split across draws)")
    return args


if __name__ == "__main__":
    run(parse_args())
