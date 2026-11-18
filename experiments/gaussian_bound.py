#!/usr/bin/env python
"""Gaussian noise on the target embeddings: the Theorem 1 / Corollary 3 bound against the measured attack.

What re-identification survives if every target embedding is released as ``y = e + sigma z``,
``z ~ N(0, I_d)`` -- the Gaussian mechanism applied to the vector itself, where every registered
defense rewrites the text. Two answers are computed on the same split and compared:

1. **Margins.** For every in-set target ``t`` (``e`` its unit embedding, ``a`` the true author's
   known embedding closest to it) and every known embedding ``b`` of another author,
   ``gamma_{t,b} = (<e,a> - <e,b>) / ||a - b||``, with ``||a - b|| = sqrt(2 - 2<a,b>)`` on unit
   vectors.
2. **Noise grid.** Cosine retentions ``c`` and ``sigma(c) = sqrt((1 - c^2) / (c^2 d))``, the scale
   at which ``cos(y, e) ~= c``; plus ``sigma = 0``.
3. **Bound.** ``l_t = max(0, 1 - sum_b Phi(-gamma_{t,b} / sigma))`` lower-bounds the chance that
   ``a`` still beats every impostor (Theorem 1, a union bound). Conversation level is the mean
   ``l_t``; author level the mean over authors of ``1 - prod_t (1 - l_t)`` (Corollary 3: targets
   draw independent noise).
3b. **Joint bound** (``--ks``). An impostor causes a failure only if it outscores *every* document
   of the true author, so its term may be ``P(b outscores each of A_k)`` for ``A_k`` the author's
   ``k`` documents closest to ``e``: a ``k``-variate normal CDF at ``-gamma_{a,b} / sigma`` with
   correlations ``<a - b, a' - b> / (||a - b|| ||a' - b||)``. Never larger than Theorem 1's term,
   and ``k = 1`` is Theorem 1. Computed exactly for ``k = 2`` (Owen's T) and ``k = 3`` (a
   quadrature over the bivariate CDF); see :func:`joint_sums` for the pruning that keeps it cheap.
4. **Measurement.** Real noise ``y = e + sigma z`` on every target, in-set and out-of-set, then
   nearest-neighbour attribution by cosine -- max over each author's known documents, as
   ``NearestNeighbor`` scores it -- over ``--reps`` independent draws.
5. **Privacy axis.** ``epsilon`` at ``delta = 1e-5`` for L2 sensitivity 2 (two unit vectors are
   at most 2 apart), in closed form and exactly (Balle & Wang 2018, by root-finding).
6. **Open set.** Out-of-set targets are noised too, the T-norm rejection score
   (:func:`~prompt_anonymity.attacks.ood.rejection.rejection_score`) is recomputed at each
   ``sigma``, and author-level DIR @ 10% FAR is reported beside the closed-set rates.

**Geometry is cosine on the raw unit-norm Gemini vectors, not the runner's z-scored space**: the
theorem assumes isotropic noise, and noise is isotropic only in the space it was added in. The
split is ``run_experiment.py``'s ``known0075`` -- known ``[0, 0.75)`` by ``ended_at``, targets the
held-out last quarter -- loaded through its own loader.

**Speed.** The noised score is ``<e + sigma z, k> = <e, k> + sigma <z, k>``, so one draw ``z`` per
rep serves every ``sigma``: each ``sigma`` still sees exactly ``N(0, sigma^2 I)`` noise, and the
curves are paired across ``sigma``. Reps run in forked worker processes, each limited to its share
of the CPUs this process may use (:func:`stage_sim`).

Outputs, under ``--out`` (default ``experiments/gaussian_bound/``, a root of its own: the
directory names under ``results/`` and ``clustering/`` are contracts ``plot_results.py`` parses):

* ``results/meta_<source>.json`` -- split sizes and the ``c`` grid.
* ``results/bound_<source>.npz`` -- per-target ``l_t`` at every ``sigma``, its grouping-slack twin
  ``ell_upper``, the joint bounds ``ell_k<k>`` and the smallest margin;
  ``results/bound_curves_<source>.csv`` the two levels of each (``bound_*`` is Theorem 1,
  ``bound_k<k>_*`` the joint bound).
* ``results/margins_<source>.csv`` -- smallest margin per in-set target.
* ``results/sim_<source>.csv`` -- one row per (rep, ``sigma``): metrics ``A``-``D`` (see
  :func:`metrics`) and the mean ``cos(y, e)``. Reps are appended to a ``.partial.csv`` as they
  finish, so an interrupted run resumes; ``--force`` starts both stages over.
* ``--report`` (:mod:`gaussian_bound_report`): ``figures/*.pdf`` and ``results/summary.md``.

Usage::

    python experiments/gaussian_bound.py --source swe_chat --reps 20
    python experiments/gaussian_bound.py --source wildchat --reps 5 --workers 4   # a SLURM job
    python experiments/gaussian_bound.py --report        # figures + summary.md from saved results

swe-chat takes seconds. A WildChat rep scores 43k noised targets against 129k known documents at
every ``sigma``; run it as a batch job, not on a login node.
"""

from __future__ import annotations

import argparse
import functools
import json
import multiprocessing as mp
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import log_ndtr, ndtr, ndtri, owens_t
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plot_results as P  # noqa: E402  (the author-level DIR the figures use)
import run_experiment as R  # noqa: E402  (loader and known/unknown split)

from prompt_anonymity.attacks.common import group_by_author  # noqa: E402
from prompt_anonymity.attacks.ood.rejection import rejection_score  # noqa: E402
from prompt_anonymity.evaluation.metrics.detection import (  # noqa: E402
    detection_identification_rate,
)

FEATURE = "gemini_embedding_2"
#: ``run_experiment.py``'s 75% known / 25% target configuration.
KNOWN_WINDOW = "0075"
TEST_FRACTION = 0.25
#: The false-accept budget of metrics C and D.
FAR = 0.10
DELTA = 1e-5
#: The retentions Table X and the Section 6.5 numbers are quoted at.
TABLE_C = (0.9, 0.7, 0.5, 0.3, 0.1)
#: ~50 retentions from 0.02 to 0.99 (the spec), the Table X / Section 6.5 points, and c = 1 (sigma = 0).
C_GRID = np.unique(np.round(np.concatenate([np.linspace(0.02, 0.99, 50), TABLE_C, [1.0]]), 6))
#: Per target, the smallest EXACT_MARGINS margins enter the bound exactly; the rest are sorted into
#: TAIL_GROUPS equal-size groups that each contribute size x Phi(-(group minimum) / sigma). That can
#: only lower the bound, so it stays valid. (swe-chat has fewer known embeddings: all exact.)
EXACT_MARGINS = 4000
TAIL_GROUPS = 512
#: 2 - 2<a,b> below this is treated as b == a (duplicate prompt across users): margin 0, per the spec.
DUPLICATE_TOL = 1e-6
DEFAULT_OUT = Path(__file__).resolve().parent / "gaussian_bound"

#: The PI's hand-calculated Table X: c -> (D, sigma, eps), checked against the code in summary.md.
PI_TABLE = {0.9: (0.23, 0.0087, 27_300), 0.7: (1.04, 0.018, 6_400), 0.5: (3.0, 0.031, 2_400),
            0.3: (10.1, 0.057, 770), 0.1: (99, 0.18, 120)}


# --- data ------------------------------------------------------------------------------------------

def unit(rows: np.ndarray) -> np.ndarray:
    return rows / np.linalg.norm(rows, axis=1, keepdims=True)


@dataclass
class Setup:
    """One corpus's ``known0075`` split, in the raw cosine space the bound is stated in."""

    source: str
    d: int
    known_unit: np.ndarray       # (n_known, d) unit rows, sorted so each author is one block
    authors: np.ndarray          # sorted unique known authors (NearestNeighbor's column order)
    starts: np.ndarray           # each author's block in known_unit
    stops: np.ndarray
    target_unit: np.ndarray      # (n_target, d) unit rows, every target (in-set and out-of-set)
    target_labels: np.ndarray
    true_index: np.ndarray       # target -> author index, -1 if out of set

    @property
    def in_set(self) -> np.ndarray:
        return self.true_index >= 0


def load_setup(source: str) -> Setup:
    frame, embeddings = R.load_documents_and_features(R.DATA_DIR, source, FEATURE)
    (_, known, unknown), = R.known_configurations(len(frame), [KNOWN_WINDOW], TEST_FRACTION)
    labels = frame["author_id"].to_numpy()
    target_unit = unit(embeddings[unknown]).astype(np.float32)
    target_labels = labels[unknown]

    authors, codes = np.unique(labels[known], return_inverse=True)
    groups = group_by_author(codes, len(authors))
    lookup = {author: i for i, author in enumerate(authors)}
    true_index = np.array([lookup.get(author, -1) for author in target_labels])
    known_unit = unit(embeddings[known][groups.order].astype(np.float32)).astype(np.float32)
    print(f"[{source}] d={embeddings.shape[1]} known={len(known_unit):,} "
          f"({len(authors):,} authors) targets={len(target_unit):,} "
          f"in-set={int((true_index >= 0).sum()):,}", flush=True)
    return Setup(source, embeddings.shape[1], known_unit, authors, groups.starts, groups.stops,
                 target_unit, target_labels, true_index)


def metrics(correct: np.ndarray, accept: np.ndarray, setup: Setup) -> dict:
    """The four rates, from per-target correctness and rejection score (higher = more out-of-set).

    * ``A`` -- closed-set top-1 over in-set targets (conversation level, DIR @ 100% FAR).
    * ``B`` -- share of in-set authors with at least one target attributed correctly.
    * ``C`` -- document-level DIR @ :data:`FAR`.
    * ``D`` -- author-level DIR @ :data:`FAR`, through the plotter's own author collapse, so it is
      the number an ``openset/dirfar/author`` panel would read at that FAR.
    """
    in_set, labels = setup.in_set, setup.target_labels
    top1 = float(correct[in_set].mean())
    authors_linked = float(pd.Series(correct[in_set]).groupby(labels[in_set]).any().mean())
    documents_dir = detection_identification_rate(accept, correct, ~in_set, FAR)
    table = pd.DataFrame({
        "true_author": labels,
        "accept_score": accept,
        "true_author_rank": np.where(in_set, np.where(correct, 1.0, 2.0), np.nan),
        "author_in_known": in_set,
        "n_candidate_authors": len(setup.authors),
    })
    people = P.author_table(table)
    curve = P.weighted_identification_any(
        -people["accept_score"].to_numpy(float), people["hit_score"].to_numpy(float),
        ~people["author_in_known"].to_numpy(bool))
    authors_dir = float(np.interp(FAR, P.DETECTION_GRID, curve))
    return {"A": top1, "B": authors_linked, "C": float(documents_dir), "D": authors_dir}


# --- noise scale and privacy -----------------------------------------------------------------------

def sigma_of_c(c, d: int):
    c = np.asarray(c, float)
    with np.errstate(divide="ignore"):
        return np.sqrt((1 - c ** 2) / (c ** 2 * d))


def eps_closed(sigma: float, delta: float = DELTA) -> float:
    """The classical tail bound for L2 sensitivity 2: ``2 / sigma^2 + (2 / sigma) sqrt(2 ln(1/delta))``."""
    return 2 / sigma ** 2 + 2 / sigma * np.sqrt(2 * np.log(1 / delta))


def eps_exact(sigma: float, delta: float = DELTA) -> float:
    """Smallest eps with Phi(1/s - eps s/2) - e^eps Phi(-1/s - eps s/2) <= delta (sensitivity 2)."""
    def excess(eps: float) -> float:
        return (ndtr(1 / sigma - eps * sigma / 2)
                - np.exp(eps + log_ndtr(-1 / sigma - eps * sigma / 2)) - delta)

    if excess(0.0) <= 0:
        return 0.0
    hi = eps_closed(sigma, delta)
    while excess(hi) > 0:
        hi *= 2
    return float(brentq(excess, 0.0, hi, xtol=1e-9, rtol=1e-12, maxiter=500))


# --- joint terms: multivariate normal CDFs -----------------------------------------------------------

#: |h| beyond this is certain or impossible to double precision (Phi(38) == 1.0), so clipping there
#: keeps every CDF argument finite without moving a value.
CDF_CLIP = 38.0
#: A correlation this close to +-1 takes its limiting form; Owen's T divides by sqrt(1 - rho^2).
RHO_EDGE = 1e-12
#: In the trivariate CDF, a pair this correlated makes the correlation matrix all but singular and
#: the integrand a spike, so the smallest pairwise CDF -- an upper bound -- is used instead.
TVN_DEGENERATE = 1 - 1e-6
#: Gauss-Legendre nodes for the trivariate CDF's path integral. On 400 real swe-chat (target,
#: impostor) triples 48 nodes agree with 96 to 5e-15; 24 left 8e-9. Measured, with a scipy check,
#: by tmp/gaussian_bound_diagnostics/mvn_accuracy.py.
TVN_NODES = 48


@functools.lru_cache(maxsize=None)
def _gauss_legendre_unit(nodes: int) -> tuple[np.ndarray, np.ndarray]:
    """Gauss-Legendre nodes and weights on [0, 1]."""
    t, w = np.polynomial.legendre.leggauss(nodes)
    return (t + 1) / 2, w / 2


def bivariate_normal_cdf(h: np.ndarray, k: np.ndarray, rho: np.ndarray) -> np.ndarray:
    """``P(X <= h, Y <= k)`` for standard normals ``X, Y`` with correlation ``rho``, elementwise.

    Owen's (1956) reduction to his T function, which scipy evaluates to double precision:
    ``(Phi(h) + Phi(k)) / 2 - T(h, a_h) - T(k, a_k) - beta``, with
    ``a_h = (k - rho h) / (h sqrt(1 - rho^2))``, ``a_k`` symmetrically, and ``beta = 1/2`` when
    ``h`` and ``k`` have opposite signs. Correlations at +-1 take their limits. The result is
    clipped to the Frechet bounds, so rounding can never lift a term above ``min(Phi(h), Phi(k))``,
    which it is known not to exceed -- the property the bound's validity rests on.
    """
    h = np.clip(h, -CDF_CLIP, CDF_CLIP)
    k = np.clip(k, -CDF_CLIP, CDF_CLIP)
    # The form divides by h and k; the CDF is continuous, so an exact zero may move by 1e-300.
    h = np.where(h == 0, 1e-300, h)
    k = np.where(k == 0, 1e-300, k)
    rho = np.clip(rho, -1.0, 1.0)
    ph, pk = ndtr(h), ndtr(k)
    inner = np.abs(rho) < 1 - RHO_EDGE
    s = np.sqrt(np.where(inner, 1 - rho ** 2, 1.0))
    a_h = np.where(inner, (k - rho * h) / (h * s), 0.0)
    a_k = np.where(inner, (h - rho * k) / (k * s), 0.0)
    value = 0.5 * (ph + pk) - owens_t(h, a_h) - owens_t(k, a_k) - np.where(h * k < 0, 0.5, 0.0)
    limit = np.where(rho > 0, np.minimum(ph, pk), np.maximum(0.0, ph + pk - 1))
    value = np.where(inner, value, limit)
    return np.clip(value, np.maximum(0.0, ph + pk - 1), np.minimum(ph, pk))


def trivariate_normal_cdf(h: np.ndarray, rho: np.ndarray, nodes: int = TVN_NODES) -> np.ndarray:
    """``P(X_0 <= h_0, X_1 <= h_1, X_2 <= h_2)`` for standard normals, one row per case.

    ``h`` is ``(n, 3)`` and ``rho`` is ``(n, 3)`` holding ``(r01, r02, r12)``. Genz's (2004) method:
    Plackett's identity, ``d Phi_3 / d r_ij = phi_2(h_i, h_j; r_ij) P(X_l <= h_l | X_i = h_i,
    X_j = h_j)``, integrated along the path that scales the correlations of one coordinate ``x``
    with the other two from 0 to their values while ``r_yz`` stays fixed::

        Phi_3 = Phi(h_x) Phi_2(h_y, h_z; r_yz)
                + int_0^1 [r_xy phi_2(h_x, h_y; t r_xy) Phi(c_z(t)) + r_xz phi_2(h_x, h_z; t r_xz) Phi(c_y(t))] dt

    where ``c_z(t)`` standardises ``h_z`` by the conditional mean and sd of ``z`` given
    ``x = h_x, y = h_y`` under the path's correlations. ``(y, z)`` is the most correlated pair, so
    the correlation that would make the integrand steep is the one handled exactly (by
    :func:`bivariate_normal_cdf`) and the integrand left is smooth in ``t``. A row with two
    near-identical coordinates (``|r| > TVN_DEGENERATE``) takes the smallest of its three pairwise
    CDFs instead, an upper bound, so a bound built on it stays valid.
    """
    h = np.clip(h, -CDF_CLIP, CDF_CLIP)
    r = np.clip(rho, -1.0, 1.0)
    # For each strongest pair (01, 02, 12): the coordinate left over is x.
    xyz = np.array([[2, 0, 1], [1, 0, 2], [0, 1, 2]])
    corr = np.stack([r[:, [1, 2, 0]], r[:, [0, 2, 1]], r[:, [0, 1, 2]]])   # (r_xy, r_xz, r_yz)
    pick = np.abs(r).argmax(axis=1)
    rows = np.arange(len(h))
    h_x, h_y, h_z = (h[rows, xyz[pick, m]] for m in range(3))
    r_xy, r_xz, r_yz = (corr[pick, rows, m] for m in range(3))

    t, w = _gauss_legendre_unit(nodes)
    a, b = t * r_xy[:, None], t * r_xz[:, None]
    hx, hy, hz, ryz = h_x[:, None], h_y[:, None], h_z[:, None], r_yz[:, None]
    det = np.maximum(1 - a ** 2 - b ** 2 - ryz ** 2 + 2 * a * b * ryz, 1e-300)
    one_a, one_b = 1 - a ** 2, 1 - b ** 2
    density_xy = np.exp(-(hx ** 2 - 2 * a * hx * hy + hy ** 2) / (2 * one_a)) / (2 * np.pi * np.sqrt(one_a))
    density_xz = np.exp(-(hx ** 2 - 2 * b * hx * hz + hz ** 2) / (2 * one_b)) / (2 * np.pi * np.sqrt(one_b))
    z_given = (hz - ((b - a * ryz) * hx + (ryz - a * b) * hy) / one_a) / np.sqrt(det / one_a)
    y_given = (hy - ((a - b * ryz) * hx + (ryz - a * b) * hz) / one_b) / np.sqrt(det / one_b)
    integrand = (r_xy[:, None] * density_xy * ndtr(z_given)
                 + r_xz[:, None] * density_xz * ndtr(y_given))
    value = ndtr(h_x) * bivariate_normal_cdf(h_y, h_z, r_yz) + (integrand * w).sum(axis=1)

    degenerate = np.abs(r).max(axis=1) > TVN_DEGENERATE
    if degenerate.any():
        d = degenerate
        value[d] = np.minimum.reduce([bivariate_normal_cdf(h[d, 0], h[d, 1], r[d, 0]),
                                      bivariate_normal_cdf(h[d, 0], h[d, 2], r[d, 1]),
                                      bivariate_normal_cdf(h[d, 1], h[d, 2], r[d, 2])])
    return np.clip(value, 0.0, ndtr(h).min(axis=1))


# --- step 3b: the joint bound over the true author's k closest documents ---------------------------

#: The joint bound sizes this script evaluates exactly (``--ks``): 1 is Theorem 1, 2 a bivariate
#: normal CDF per (target, impostor), 3 a quadrature over one -- about TVN_NODES times the cost.
JOINT_KS = (1, 2, 3)
#: A joint term is evaluated only where the coarser term it refines (Theorem 1's for k = 2, the
#: k = 2 term for k = 3) exceeds this; elsewhere the coarser term stands in. It is never smaller,
#: so the bound stays valid, and over EXACT_MARGINS impostors it costs at most 4e-7 of bound.
JOINT_PRUNE = 1e-10
#: The event-correlation pairs (r01, r02, r12) among a target's three closest true-author documents.
JOINT_PAIRS = ((0, 1), (0, 2), (1, 2))


def closest_true_documents(S: np.ndarray, a_idx: np.ndarray, lo_all: np.ndarray,
                           hi_all: np.ndarray, k: int) -> np.ndarray:
    """``(n, k)`` indices of each target's ``k`` known documents by its true author closest to it.

    ``a*`` (``a_idx``, Theorem 1's document) always comes first, so the ``k = 1`` column is exactly
    Theorem 1. An author with fewer than ``k`` known documents repeats ``a*``: a repeated document
    adds nothing to a joint event (its two coordinates have correlation 1).
    """
    top = np.repeat(a_idx[:, None], k, axis=1)
    for r, (lo, hi) in enumerate(zip(lo_all, hi_all)):
        others = lo + np.argsort(-S[r, lo:hi])
        others = others[others != a_idx[r]][:k - 1]
        top[r, 1:1 + len(others)] = others
    return top


def joint_geometry(known_unit: np.ndarray, S: np.ndarray, top: np.ndarray, exact_idx: np.ndarray,
                   exact_gamma: np.ndarray, sub_block: int = 32) -> tuple[np.ndarray, np.ndarray]:
    """Margins and event correlations of each target's closest true-author documents.

    ``top`` is :func:`closest_true_documents`; ``exact_idx`` and ``exact_gamma`` are the impostors
    the bound treats exactly and their Theorem 1 margins against ``a*`` (``+inf`` marks one of the
    author's own documents, which is not an impostor). Returns

    * ``gamma`` ``(n, k, m)`` -- ``gamma_{a,b}`` for each of the ``k`` documents against each of the
      ``m`` impostors. Row 0 *is* ``exact_gamma``, so the ``k = 1`` term is Theorem 1's bit for
      bit. An impostor identical to ``a`` gets ``-inf`` (certain loss), as in Theorem 1.
    * ``rho`` ``(n, 3, m)`` -- the correlations :data:`JOINT_PAIRS` of the events "``b`` outscores
      ``a``": ``<a - b, a' - b> / (||a - b|| ||a' - b||)``. Unused pairs (``k < 3``) stay 1.
    """
    n, k = top.shape
    m = exact_idx.shape[1]
    gamma = np.empty((n, k, m))
    rho = np.ones((n, len(JOINT_PAIRS), m))
    for start in range(0, n, sub_block):
        sl = slice(start, start + sub_block)
        A = known_unit[top[sl]]                                              # (s, k, d)
        # One product per document rank, so a rank's values do not depend on how many ranks
        # were requested (a differently shaped product rounds differently).
        dot_ab = np.stack([np.take_along_axis(A[:, i] @ known_unit.T, exact_idx[sl], axis=1)
                           for i in range(k)], axis=1).astype(np.float64)   # <a, b>
        A64 = A.astype(np.float64)
        dot_aa = np.einsum("sid,sjd->sij", A64, A64)                         # <a, a'>
        s_a = np.take_along_axis(S[sl], top[sl], axis=1)
        s_b = np.take_along_axis(S[sl], exact_idx[sl], axis=1)
        dist2 = 2.0 - 2.0 * dot_ab
        dist = np.sqrt(np.maximum(dist2, DUPLICATE_TOL))
        g = np.where(dist2 < DUPLICATE_TOL, -np.inf, (s_a[:, :, None] - s_b[:, None, :]) / dist)
        g[np.broadcast_to(np.isposinf(exact_gamma[sl])[:, None, :], g.shape)] = np.inf
        g[:, 0, :] = exact_gamma[sl]
        gamma[sl] = g
        for p, (i, j) in enumerate(JOINT_PAIRS):
            if j < k:
                rho[sl, p] = ((dot_aa[:, i, j, None] - dot_ab[:, i] - dot_ab[:, j] + 1.0)
                              / (dist[:, i] * dist[:, j]))
    return gamma, np.clip(rho, -1.0, 1.0)


def joint_sums(single: np.ndarray, gamma: np.ndarray, rho: np.ndarray, sigma: float,
               ks: list[int]) -> dict[int, np.ndarray]:
    """Per target, ``sum_b P(b outscores each of its k closest true-author documents)``, for each k.

    ``single`` ``(n, m)`` is Theorem 1's terms at ``sigma``; each joint term is evaluated where the
    coarser term it refines exceeds :data:`JOINT_PRUNE` and capped by it, so the sums can only
    shrink as ``k`` grows. A target whose bound is already 0 is not refined at all: Slepian's
    inequality (the joint CDF is at least the product of the marginals when every correlation is
    non-negative; the Frechet bound otherwise) gives a lower bound on its sum, and once that
    reaches 1 the bound is 0 whatever the exact terms are. Such a target's sum is ``inf``.
    """
    h = np.clip(-gamma / sigma, -CDF_CLIP, CDF_CLIP)
    sums = {}
    coarse = single
    for k in ks:
        marginals = ndtr(h[:, :k])
        pairs = rho[:, :1] if k == 2 else rho
        lower = np.where((pairs >= 0).all(axis=1), marginals.prod(axis=1),
                         np.maximum(0.0, marginals.sum(axis=1) - (k - 1)))
        settled = lower.sum(axis=1) >= 1
        term = coarse.copy()
        r, c = np.nonzero((coarse > JOINT_PRUNE) & ~settled[:, None])
        if len(r):
            if k == 2:
                term[r, c] = bivariate_normal_cdf(h[r, 0, c], h[r, 1, c], rho[r, 0, c])
            else:
                term[r, c] = trivariate_normal_cdf(h[r, :, c], rho[r, :, c])
        term = np.minimum(term, coarse)
        sums[k] = np.where(settled, np.inf, term.sum(axis=1))
        coarse = term
    return sums


# --- steps 1 + 3: margins and bounds ---------------------------------------------------------------

def stage_bound(setup: Setup, sigmas: np.ndarray, path: Path, ks=(1,), block: int = 256) -> None:
    known_unit = setup.known_unit
    n_known = len(known_unit)
    in_rows = np.flatnonzero(setup.in_set)
    n_exact = min(EXACT_MARGINS, n_known)
    joint = sorted(k for k in set(ks) if k > 1)
    ell = np.zeros((len(setup.true_index), len(sigmas)))
    ell_upper = np.zeros_like(ell)            # tail groups at their maximum: grouping slack only
    ell_joint = {k: np.zeros_like(ell) for k in joint}
    gamma_min = np.full(len(setup.true_index), np.nan)
    n_dup = 0
    t0 = time.time()
    for b0 in range(0, len(in_rows), block):
        rows = in_rows[b0:b0 + block]
        S = (setup.target_unit[rows] @ known_unit.T).astype(np.float64)
        truth = setup.true_index[rows]
        lo_all, hi_all = setup.starts[truth], setup.stops[truth]
        a_idx = np.array([lo + int(S[r, lo:hi].argmax()) for r, (lo, hi) in enumerate(zip(lo_all, hi_all))])
        dist2 = 2.0 - 2.0 * (known_unit[a_idx] @ known_unit.T).astype(np.float64)
        s_a = S[np.arange(len(rows)), a_idx]
        dup = dist2 < DUPLICATE_TOL
        # A competitor identical to a ties with it under every noise draw, so whether the attack
        # wins depends on tie-breaking: count it as a certain loss (margin -inf, term 1) so the
        # bound holds for any tie-breaking rule. Its reported margin is still 0, as in the spec.
        gam = np.where(dup, -np.inf, (s_a[:, None] - S) / np.sqrt(np.maximum(dist2, DUPLICATE_TOL)))
        for r, (lo, hi) in enumerate(zip(lo_all, hi_all)):
            n_dup += int(dup[r].sum() - dup[r, lo:hi].sum())
            gam[r, lo:hi] = np.inf                                   # the true author's own docs
        gamma_min[rows] = np.where(np.isneginf(gam), 0.0, gam).min(axis=1)
        order = np.argsort(gam, axis=1)                              # +inf (own docs) sort last
        g = np.take_along_axis(gam, order, axis=1)
        exact, rest = g[:, :n_exact], g[:, n_exact:]
        if rest.shape[1]:
            edges = np.linspace(0, rest.shape[1], TAIL_GROUPS + 1).astype(int)
            edges = np.unique(edges)
            size = np.diff(edges).astype(float)
            g_lo, g_hi = rest[:, edges[:-1]], rest[:, edges[1:] - 1]
        # The joint bound refines the exact impostors only; the tail groups keep Theorem 1's
        # terms, which upper-bound the joint ones, so every k shares them and stays valid.
        if joint:
            top = closest_true_documents(S, a_idx, lo_all, hi_all, max(joint))
            joint_gamma, joint_rho = joint_geometry(known_unit, S, top, order[:, :n_exact], exact)
        for j, s in enumerate(sigmas):
            if s == 0:
                ell[rows, j] = ell_upper[rows, j] = (gamma_min[rows] > 0).astype(float)
                for k in joint:
                    ell_joint[k][rows, j] = ell[rows, j]
                continue
            single = ndtr(-exact / s)
            tot = single.sum(axis=1)
            tot_hi = tot.copy()
            tail = 0.0
            if rest.shape[1]:
                tail = (ndtr(-g_lo / s) * size).sum(axis=1)
                tot = tot + tail
                tot_hi = tot_hi + (ndtr(-g_hi / s) * size).sum(axis=1)
            ell[rows, j] = np.maximum(0.0, 1.0 - tot)
            ell_upper[rows, j] = np.maximum(0.0, 1.0 - tot_hi)
            if joint:
                for k, total in joint_sums(single, joint_gamma, joint_rho, s, joint).items():
                    ell_joint[k][rows, j] = np.maximum(0.0, 1.0 - (total + tail))
        print(f"  bound {min(b0 + block, len(in_rows)):,}/{len(in_rows):,} ({time.time() - t0:.0f}s)",
              flush=True)
    np.savez_compressed(path, ell=ell, ell_upper=ell_upper, gamma_min=gamma_min, sigmas=sigmas,
                        n_duplicate_pairs=n_dup, ks=np.array([1, *joint]),
                        **{f"ell_k{k}": ell_joint[k] for k in joint})


def bound_curves(setup: Setup, z) -> pd.DataFrame:
    """Conversation level (mean ``l_t``) and author level (Corollary 3) for every bound in ``z``.

    Columns ``bound_*`` are Theorem 1 (``k = 1``), ``bound_upper_*`` its tail-grouping check, and
    ``bound_k<k>_*`` the joint bound over each target's ``k`` closest true-author documents.
    """
    in_set = setup.in_set
    labels = setup.target_labels[in_set]
    out = {}
    joint = [(f"bound_k{k}", f"ell_k{k}") for k in z["ks"] if k > 1] if "ks" in z.files else []
    for name, key in (("bound", "ell"), ("bound_upper", "ell_upper"), *joint):
        p = z[key][in_set]
        conv = p.mean(axis=0)
        log_miss = pd.DataFrame(np.log1p(-np.minimum(p, 1 - 1e-15))).groupby(labels).sum()
        author = (1 - np.exp(log_miss.to_numpy())).mean(axis=0)
        out[f"{name}_conv"], out[f"{name}_author"] = conv, author
    return pd.DataFrame({"sigma": z["sigmas"], **out})


# --- steps 4 + 6: measurement ----------------------------------------------------------------------

_SETUP: Setup | None = None


def _limit_threads(threads: int) -> None:
    """Pool initializer: one worker's share of the CPUs, so ``--workers`` never oversubscribes."""
    threadpool_limits(threads)


def sim_rep(args) -> list[dict]:
    rep, sigmas, block = args
    setup = _SETUP
    rng = np.random.default_rng([rep, 13])
    known_unit, n = setup.known_unit, len(setup.true_index)
    correct = np.zeros((len(sigmas), n), bool)
    accept = np.empty((len(sigmas), n))
    cos_sum = np.zeros(len(sigmas))
    t0 = time.time()
    for start in range(0, n, block):
        rows = slice(start, start + block)
        e = setup.target_unit[rows]
        S0 = e @ known_unit.T
        Z = rng.standard_normal(e.shape, dtype=np.float32)
        N = Z @ known_unit.T
        ez = np.einsum("ij,ij->i", e, Z, dtype=np.float64)
        zz = np.einsum("ij,ij->i", Z, Z, dtype=np.float64)
        ee = np.einsum("ij,ij->i", e, e, dtype=np.float64)
        for j, s in enumerate(sigmas):
            norm_y = np.sqrt(ee + 2 * s * ez + s * s * zz)
            cos_sum[j] += ((ee + s * ez) / (norm_y * np.sqrt(ee))).sum()
            noisy = (S0 + np.float32(s) * N) / norm_y[:, None].astype(np.float32) - 1.0  # cosine - 1
            author = np.maximum.reduceat(noisy, setup.starts, axis=1)
            correct[j, rows] = author.argmax(axis=1) == setup.true_index[rows]
            accept[j, rows] = rejection_score(author)
    out = [{"rep": rep, "sigma": s, "mean_cos": cos_sum[j] / n, **metrics(correct[j], accept[j], setup)}
           for j, s in enumerate(sigmas)]
    print(f"  rep {rep} done ({time.time() - t0:.0f}s)", flush=True)
    return out


def stage_sim(setup: Setup, sigmas: np.ndarray, path: Path, reps: int, workers: int,
              block: int = 512) -> None:
    global _SETUP
    _SETUP = setup
    partial = path.with_name(path.stem + ".partial.csv")
    done = set(pd.read_csv(partial)["rep"].astype(int)) if partial.exists() else set()
    todo = [(r, sigmas, block) for r in range(reps) if r not in done]
    if todo:
        workers = min(workers, len(todo))
        threads = max(1, len(os.sched_getaffinity(0)) // workers)
        # Forked, so every worker reads the parent's Setup without a copy.
        ctx = mp.get_context("fork")
        with ctx.Pool(workers, initializer=_limit_threads, initargs=(threads,)) as pool:
            for rows in pool.imap_unordered(sim_rep, todo):
                pd.DataFrame(rows).to_csv(partial, mode="a", header=not partial.exists(), index=False)
    frame = pd.read_csv(partial).sort_values(["sigma", "rep"])
    frame.to_csv(path, index=False)


# --- driver ----------------------------------------------------------------------------------------

def run(source: str, out: Path, reps: int, workers: int, force: bool, ks=JOINT_KS) -> None:
    res = out / "results"
    res.mkdir(parents=True, exist_ok=True)
    setup = load_setup(source)
    sigmas = sigma_of_c(C_GRID, setup.d)
    meta = {"source": source, "d": setup.d, "n_known_docs": len(setup.known_unit),
            "n_known_authors": len(setup.authors), "n_targets": len(setup.true_index),
            "n_in_set_targets": int(setup.in_set.sum()), "n_ood_targets": int((~setup.in_set).sum()),
            "n_in_set_authors": int(pd.Series(setup.target_labels[setup.in_set]).nunique()),
            "n_ood_authors": int(pd.Series(setup.target_labels[~setup.in_set]).nunique()),
            "c_grid": C_GRID.tolist(), "ks": sorted(set(ks) | {1})}
    (res / f"meta_{source}.json").write_text(json.dumps(meta, indent=2))

    bound_path = res / f"bound_{source}.npz"
    # A saved bound without every requested k is recomputed: the joint terms come from the same
    # pass over the margins, so there is no cheaper way to add one.
    stale = (not bound_path.exists()
             or not {f"ell_k{k}" for k in ks if k > 1} <= set(np.load(bound_path).files))
    if force or stale:
        print(f"[{source}] margins + bounds, k = {', '.join(map(str, sorted(set(ks) | {1})))}",
              flush=True)
        stage_bound(setup, sigmas, bound_path, ks)
    z = np.load(bound_path)
    bound_curves(setup, z).assign(c=C_GRID).to_csv(res / f"bound_curves_{source}.csv", index=False)
    pd.DataFrame({"target": np.flatnonzero(setup.in_set),
                  "author": setup.target_labels[setup.in_set],
                  "gamma_min": z["gamma_min"][setup.in_set]}).to_csv(res / f"margins_{source}.csv", index=False)

    sim_path = res / f"sim_{source}.csv"
    if force:
        # The partial file is what a resumed run trusts, so it has to go too.
        sim_path.unlink(missing_ok=True)
        sim_path.with_name(sim_path.stem + ".partial.csv").unlink(missing_ok=True)
    if not sim_path.exists() or pd.read_csv(sim_path)["rep"].nunique() < reps:
        print(f"[{source}] measured attack, {reps} reps on {workers} workers", flush=True)
        stage_sim(setup, sigmas, sim_path, reps, workers)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", choices=["swe_chat", "wildchat"])
    parser.add_argument("--reps", type=int, default=20,
                        help="Independent noise draws per sigma (default: 20).")
    parser.add_argument("--workers", type=int, default=4,
                        help="Worker processes for the reps; the CPUs are split between them.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help=f"Output root (default: {DEFAULT_OUT.relative_to(R.REPO_ROOT)}).")
    parser.add_argument("--ks", type=int, nargs="+", default=list(JOINT_KS), choices=JOINT_KS,
                        help="Joint bounds to compute: each impostor's term becomes the chance it "
                             "outscores the true author's k closest documents at once. 1 is "
                             "Theorem 1 (always computed); 3 costs ~30x 2. Default: 1 2 3.")
    parser.add_argument("--force", action="store_true",
                        help="Recompute the bound and every rep instead of resuming.")
    parser.add_argument("--report", action="store_true",
                        help="Only build the figures and summary.md from saved results.")
    args = parser.parse_args()
    if args.report:
        import gaussian_bound_report
        gaussian_bound_report.build(args.out)
        return
    if not args.source:
        parser.error("--source is required unless --report")
    run(args.source, args.out, args.reps, args.workers, args.force, args.ks)


if __name__ == "__main__":
    main()
