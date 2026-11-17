#!/usr/bin/env python
"""Laplace noise on the target embeddings: reproduce the empirical sweep, then overlay theory.

Task spec: ``dp_reproduction_memo.md``. Reproduces Naina's Laplace-noise experiment (NN attack on
Gemini Embedding 2, known = first 75% by time, target = last 25%) and overlays three model curves:

* **Theorem 1** -- a rigorous lower bound on the probability that the clean nearest true-author
  document still beats every impostor document after noise (Bernstein bound for Laplace sums,
  union bound over impostors).
* **Normal approximation** -- the same union bound with each projected noise treated as Gaussian.
* **Model simulation** -- the attack re-run with Gaussian noise of the same covariance on the
  similarities, passed through the repo's own ranking, T-norm rejection and DIR@FAR code.

Geometry
--------
The runner z-scores features by default (``run_experiment.py --standardize``, known-side
statistics), and Naina's numbers are reproduced only with that on. Everything here therefore runs
in one of two **attack spaces**, chosen by ``--geometry``:

* ``raw``    -- cosine on the unit-norm Gemini vectors (the memo's derivation as written).
* ``zscore`` -- cosine after the known-side z-score (the pipeline default, what Naina ran).

Noise is always added in the raw space, to the unit-normalised target, with scale
``b = 2 sqrt(d) / eps``. In the z-scored space the target becomes ``(x + eta - mu) / s``, so the
noise on the similarity to a (unit-normalised, z-scored) known vector ``c`` is ``<eta, c / s>``.
The theory carries over exactly by replacing the difference direction ``a - n_j`` with
``w_j = (a - n_j) / s`` and the margin with ``<q, a - n_j> / ||w_j||``; for ``raw`` ``s = 1`` and
the formulas are literally the memo's.

Usage::

    python experiments/dp_reproduction.py --source swe_chat --geometry zscore
    python experiments/dp_reproduction.py --source wildchat --geometry raw --stages empirical
    python experiments/dp_reproduction.py --summary          # figures + summary.md from saved results

Each stage writes its own file under ``--out`` and is skipped when that file exists (``--force``
re-runs it), so a long WildChat sweep can be resumed stage by stage.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plot_results as P  # noqa: E402  (style constants, author-level DIR collapse)
import run_experiment as R  # noqa: E402  (loader, split, z-score)

from prompt_anonymity.attacks.common import group_by_author  # noqa: E402
from prompt_anonymity.attacks.ood.rejection import rejection_score  # noqa: E402
from prompt_anonymity.attacks.similarity.nearest_neighbor import NearestNeighbor  # noqa: E402
from prompt_anonymity.evaluation.metrics.detection import (  # noqa: E402
    detection_identification_rate,
)

FEATURE = "gemini_embedding_2"
KNOWN_WINDOW = "0075"
TEST_FRACTION = 0.25
FAR = 0.10
EPS_GRID = (1.0, 10.0, 100.0, 500.0, 1e3, 2e3, 5e3, 1e4, 2e4, 5e4, 1e5)
#: Dense grid the closed-form curves are evaluated on (they cost nothing per point).
THEORY_GRID = np.unique(np.concatenate([np.logspace(0, 5, 101), EPS_GRID]))
EMPIRICAL_SEEDS = 3
SIM_SEEDS = 10
SIM_NEIGHBOURS = (50, 200, 1000)
#: Impostors per target whose ||u_j||_inf is computed exactly; the rest use the triangle bound.
EXACT_IMPOSTORS = 1000
#: Log-spaced margin bins the remaining impostors are summarised into (conservatively: each bin
#: contributes count x term(min g in bin, max u in bin), which can only lower p_thm).
TAIL_BINS = 256
BETA = 0.05
QQ_EPS = 2000.0
QQ_DRAWS = 10_000
QQ_DIRECTIONS = 20
METRICS = ("A", "B", "C", "D")
METRIC_TITLES = {
    "A": "(A) Top-1 accuracy (per conversation)",
    "B": "(B) Users with ≥1 conversation identified",
    "C": "(C) Document-level DIR @ 10% FAR",
    "D": "(D) Author-level DIR @ 10% FAR",
}
DATASET_TITLES = {"swe_chat": "SWE-Chat", "wildchat": "WildChat"}
DEFAULT_OUT = Path(__file__).resolve().parent / "results" / "dp_overlay"


# --- data and attack space -----------------------------------------------------------------------

def unit(rows: np.ndarray) -> np.ndarray:
    return rows / np.linalg.norm(rows, axis=1, keepdims=True)


@dataclass
class Setup:
    source: str
    geometry: str
    d: int
    known_raw: np.ndarray        # (n_known, d) as loaded, float32 (unit-norm Gemini vectors)
    known_labels: np.ndarray
    target_unit: np.ndarray      # (n_target, d) unit-normalised, float32
    target_labels: np.ndarray
    authors: np.ndarray          # sorted unique known authors (NearestNeighbor's order)
    starts: np.ndarray           # author group starts in the author-sorted known order
    stops: np.ndarray
    true_index: np.ndarray       # target -> author index, -1 if out of set
    center: np.ndarray           # z-score centre (zeros for raw)
    dinv: np.ndarray             # 1 / z-score scale (ones for raw), float32
    Kh: np.ndarray               # known in attack space, unit rows, author-sorted, float32
    Kw: np.ndarray               # Kh * dinv: the noise-projection vectors, float32

    @property
    def in_set(self) -> np.ndarray:
        return self.true_index >= 0

    def clean_queries(self) -> np.ndarray:
        """Targets in attack space, before the final row normalisation (ranking ignores it)."""
        return ((self.target_unit - self.center) * self.dinv).astype(np.float32)


def load_setup(source: str, geometry: str) -> Setup:
    frame, X = R.load_documents_and_features(R.DATA_DIR, source, FEATURE)
    (_, known, unknown), = R.known_configurations(len(frame), [KNOWN_WINDOW], TEST_FRACTION)
    labels = frame["author_id"].to_numpy()
    known_raw, known_labels = X[known], labels[known]
    target_unit = unit(X[unknown]).astype(np.float32)
    target_labels = labels[unknown]

    authors, codes = np.unique(known_labels, return_inverse=True)
    groups = group_by_author(codes, len(authors))
    lookup = {author: i for i, author in enumerate(authors)}
    true_index = np.array([lookup.get(a, -1) for a in target_labels])

    d = X.shape[1]
    if geometry == "zscore":
        center = known_raw.mean(axis=0, dtype=np.float64)
        scale = known_raw.std(axis=0, dtype=np.float64)
        scale = np.where(scale > 0, scale, 1.0)
    else:
        center, scale = np.zeros(d), np.ones(d)
    dinv = (1.0 / scale).astype(np.float32)
    center = center.astype(np.float32)
    sorted_known = known_raw[groups.order]
    Kh = unit(((sorted_known - center) * dinv).astype(np.float32)).astype(np.float32)
    Kw = (Kh * dinv).astype(np.float32)
    print(f"[{source}/{geometry}] d={d} known={len(known_raw):,} ({len(authors):,} authors) "
          f"targets={len(target_unit):,} in-set={int((true_index >= 0).sum()):,}", flush=True)
    return Setup(source, geometry, d, known_raw, known_labels, target_unit, target_labels,
                 authors, groups.starts, groups.stops, true_index, center, dinv, Kh, Kw)


# --- metrics (A)-(D) -----------------------------------------------------------------------------

def metrics(correct: np.ndarray, accept: np.ndarray, setup: Setup) -> dict:
    """The four panels from per-target correctness and rejection score (higher = more OOD)."""
    in_set, labels = setup.in_set, setup.target_labels
    A = float(correct[in_set].mean())
    B = float(pd.Series(correct[in_set]).groupby(labels[in_set]).any().mean())
    C = detection_identification_rate(accept, correct, ~in_set, FAR)
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
    D = float(np.interp(FAR, P.DETECTION_GRID, curve))
    return {"A": A, "B": B, "C": float(C), "D": D}


def chance_levels(setup: Setup, reps: int = 200, seed: int = 0) -> dict:
    """Monte-Carlo random guesser: uniform author, uniform (uninformative) rejection score."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(reps):
        guess = rng.integers(0, len(setup.authors), len(setup.true_index))
        out.append(metrics(guess == setup.true_index, rng.random(len(guess)), setup))
    return {k: float(np.mean([o[k] for o in out])) for k in METRICS}


# --- empirical Laplace sweep (the repo's own attack code path) --------------------------------------

def empirical_run(setup: Setup, eps: float | None, seed: int, block: int = 4096) -> dict:
    """One NN attack on (optionally) Laplace-noised targets, exactly as run_experiment scores it."""
    U = setup.target_unit
    if eps is not None:
        rng = np.random.default_rng([seed, int(round(eps))])
        U = (U + rng.laplace(0.0, 2.0 * np.sqrt(setup.d) / eps, U.shape)).astype(np.float32)
    K = setup.known_raw
    if setup.geometry == "zscore":
        K, U = R.standardize(K, U)
    nn = NearestNeighbor().fit(K, setup.known_labels)
    assert np.array_equal(nn.authors, setup.authors)
    correct = np.empty(len(U), bool)
    accept = np.empty(len(U))
    for start in range(0, len(U), block):
        rows = slice(start, start + block)
        S = nn.score(U[rows])
        correct[rows] = S.argmax(axis=1) == setup.true_index[rows]
        accept[rows] = rejection_score(S)
    return metrics(correct, accept, setup)


def _partial(path: Path) -> tuple[Path, pd.DataFrame | None]:
    """Row-level cache beside a stage's final file: every finished run is appended at once, so an
    interrupted stage resumes where it stopped instead of starting over."""
    partial = path.with_name(path.stem + ".partial.csv")
    return partial, (pd.read_csv(partial) if partial.exists() else None)


def _append(partial: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(partial, mode="a", header=not partial.exists(), index=False)


def stage_empirical(setup: Setup, path: Path) -> pd.DataFrame:
    partial, done = _partial(path)
    finished = set() if done is None else {(float(e), int(s)) for e, s in zip(done["eps"], done["seed"])}
    jobs = [(np.inf, 0)] + [(eps, seed) for eps in EPS_GRID for seed in range(EMPIRICAL_SEEDS)]
    for eps, seed in jobs:
        if (float(eps), seed) in finished:
            continue
        t = time.time()
        row = {"eps": eps, "seed": seed,
               **empirical_run(setup, None if np.isinf(eps) else eps, seed)}
        _append(partial, [row])
        print(f"  eps={eps:g} seed={seed}: " + " ".join(f"{k}={row[k]:.4f}" for k in METRICS)
              + f" ({time.time() - t:.0f}s)", flush=True)
    frame = pd.read_csv(partial).sort_values(["eps", "seed"]).reset_index(drop=True)
    frame.to_csv(path, index=False)
    partial.unlink()
    return frame


# --- margins and closed-form theory -----------------------------------------------------------------

def theorem_terms(g: np.ndarray, u: np.ndarray, eps: np.ndarray, d: int) -> np.ndarray:
    """exp(-eps^2 g^2 / (16 d + 4 sqrt(d) eps g u)), broadcast over a leading eps axis."""
    e = eps[:, None]
    return np.exp(-(e ** 2) * g ** 2 / (16 * d + 4 * np.sqrt(d) * e * g * u))


def normal_terms(g: np.ndarray, eps: np.ndarray, d: int) -> np.ndarray:
    return norm.cdf(-eps[:, None] * g / (2 * np.sqrt(2 * d)))


def stage_margins(setup: Setup, path: Path, block: int = 256) -> dict:
    """Per correctly-attributed in-set target: margins to every impostor, and the theory curves."""
    Q = setup.clean_queries()
    Kh, Kw = setup.Kh, setup.Kw
    kw_sq = np.einsum("ij,ij->i", Kw, Kw, dtype=np.float64)
    kw_inf = np.abs(Kw).max(axis=1)
    n_known, d = Kh.shape

    in_set_rows = np.flatnonzero(setup.in_set)
    p_thm = np.zeros((len(setup.true_index), len(THEORY_GRID)))
    p_norm = np.zeros_like(p_thm)
    records = []
    t0 = time.time()
    for b0 in range(0, len(in_set_rows), block):
        rows = in_set_rows[b0:b0 + block]
        S = Q[rows] @ Kh.T                                         # (nb, n_known)
        author_best = np.maximum.reduceat(S, setup.starts, axis=1)
        truth = setup.true_index[rows]
        clean_correct = author_best.argmax(axis=1) == truth
        lo_all, hi_all = setup.starts[truth], setup.stops[truth]
        a_all = np.array([lo + int(S[r, lo:hi].argmax()) for r, (lo, hi) in enumerate(zip(lo_all, hi_all))])
        cross_all = Kw[a_all] @ Kw.T                               # <w_a, w_n> for every n
        for r, row in enumerate(rows):
            if not clean_correct[r]:
                continue
            lo, hi, a = lo_all[r], hi_all[r], a_all[r]
            s = S[r].astype(np.float64)
            w_norm = np.sqrt(np.maximum(kw_sq[a] + kw_sq - 2 * cross_all[r], 1e-20))
            m = s[a] - s
            g = np.where(m > 0, m / w_norm, 0.0)
            g[lo:hi] = np.inf                                      # the true author's own docs
            u_bound = (kw_inf[a] + kw_inf) / w_norm
            K = n_known - (hi - lo)
            n_exact = min(EXACT_IMPOSTORS, K)
            imp_idx = np.argpartition(g, n_exact - 1)[:n_exact]
            diff = (Kw[a][None, :] - Kw[imp_idx]).astype(np.float64)
            u_exact = np.abs(diff).max(axis=1) / w_norm[imp_idx]
            g_exact = g[imp_idx]
            rest = np.ones(n_known, bool)
            rest[imp_idx] = False
            rest[lo:hi] = False
            g_rest, u_rest = g[rest], u_bound[rest]

            thm = theorem_terms(g_exact, u_exact, THEORY_GRID, d).sum(axis=1)
            nrm = normal_terms(g_exact, THEORY_GRID, d).sum(axis=1)
            if g_rest.size:
                edges = np.geomspace(max(g_rest.min(), 1e-12), g_rest.max() * (1 + 1e-9), TAIL_BINS + 1)
                bins = np.clip(np.searchsorted(edges, g_rest, side="right") - 1, 0, TAIL_BINS - 1)
                count = np.bincount(bins, minlength=TAIL_BINS)
                gmin = np.full(TAIL_BINS, np.inf)
                umax = np.zeros(TAIL_BINS)
                np.minimum.at(gmin, bins, g_rest)
                np.maximum.at(umax, bins, u_rest)
                used = count > 0
                thm += (theorem_terms(gmin[used], umax[used], THEORY_GRID, d) * count[used]).sum(axis=1)
                nrm += (normal_terms(gmin[used], THEORY_GRID, d) * count[used]).sum(axis=1)
            p_thm[row] = np.maximum(0.0, 1.0 - thm)
            p_norm[row] = np.maximum(0.0, 1.0 - nrm)
            j_min = int(np.argmin(g_exact))
            records.append({
                "target": int(row), "author": setup.target_labels[row],
                "g_min": float(g_exact[j_min]),
                "u_inf": float(max(u_exact.max(), u_rest.max() if u_rest.size else 0.0)),
                "u_inf_exact": float(u_exact.max()),
                "K": K, "a_index": a, "closest_impostor_index": int(imp_idx[j_min]),
            })
        print(f"  margins {min(b0 + block, len(in_set_rows)):,}/{len(in_set_rows):,} "
              f"({time.time() - t0:.0f}s)", flush=True)
    per_target = pd.DataFrame(records)
    np.savez_compressed(path, p_thm=p_thm, p_norm=p_norm, grid=THEORY_GRID,
                        per_target=per_target.to_records(index=False))
    per_target.to_csv(path.with_suffix(".csv"), index=False)
    return load_margins(path)


def load_margins(path: Path) -> dict:
    z = np.load(path, allow_pickle=True)
    return {"p_thm": z["p_thm"], "p_norm": z["p_norm"], "grid": z["grid"],
            "per_target": pd.read_csv(path.with_suffix(".csv"))}


def theory_curves(setup: Setup, margins: dict) -> pd.DataFrame:
    """(A) mean p over in-set targets, (B) mean over in-set users of 1 - prod(1 - p_t)."""
    in_set = setup.in_set
    labels = setup.target_labels[in_set]
    rows = []
    for source, p in (("Theorem 1 (lower bound)", margins["p_thm"]),
                      ("Normal approximation", margins["p_norm"])):
        p_in = p[in_set]
        A = p_in.mean(axis=0)
        log_miss = pd.DataFrame(np.log1p(-np.minimum(p_in, 1 - 1e-15))).groupby(labels).sum()
        B = (1 - np.exp(log_miss.to_numpy())).mean(axis=0)
        for i, eps in enumerate(margins["grid"]):
            rows.append({"eps": eps, "metric": "A", "source": source, "mean": A[i], "sd": 0.0})
            rows.append({"eps": eps, "metric": "B", "source": source, "mean": B[i], "sd": 0.0})
    return pd.DataFrame(rows)


def eps_star(setup: Setup, margins: dict) -> pd.DataFrame:
    pt = margins["per_target"]
    L = np.log(pt["K"] / BETA)
    d = setup.d
    return pd.DataFrame({
        "eps_star": 4 * np.sqrt(d) / pt["g_min"] * (np.sqrt(L) + pt["u_inf"] * L),
        "eps_star_exact_u": 4 * np.sqrt(d) / pt["g_min"] * (np.sqrt(L) + pt["u_inf_exact"] * L),
    })


# --- model simulation (Gaussian noise on the similarities, full covariance) ------------------------

def sim_seed(setup: Setup, Q: np.ndarray, seed: int, variants: list, eps_grid=EPS_GRID,
             block: int = 512) -> list[dict]:
    """One seed of the model: noisy similarities <q,c> + Z through ranking / T-norm / DIR@FAR.

    Z ~ N(0, 2b^2 <Kw_c, Kw_c'>) is drawn for **all** known documents at once as
    sigma * (G @ Kw.T) with G ~ N(0, I_d) -- exactly the covariance the memo specifies -- so the
    T-norm cohort (every known author) moves under noise as it does in the real attack. One G per
    seed is shared by every eps (sigma only rescales it) and by every truncation variant: each
    (eps, variant) still sees the right marginal distribution, and the comparisons are paired.

    ``variants`` holds ``None`` (all neighbours) and/or ints M: only each target's M nearest known
    documents (by clean similarity) receive noise -- the memo's truncated model, as a check.
    """
    rng = np.random.default_rng([seed, 7])
    n = len(Q)
    keys = [(v, eps) for v in variants for eps in eps_grid]
    correct = {k: np.empty(n, bool) for k in keys}
    accept = {k: np.empty(n) for k in keys}
    for start in range(0, n, block):
        rows = slice(start, start + block)
        q = Q[rows]
        S = q @ setup.Kh.T
        G = rng.standard_normal(q.shape, dtype=np.float32)
        N = G @ setup.Kw.T
        q_norm = np.linalg.norm(q, axis=1)
        gd = G * setup.dinv
        for v in variants:
            if v is None:
                Nv = N
            else:
                keep = np.argpartition(-S, v - 1, axis=1)[:, :v]
                Nv = np.zeros_like(N)
                np.put_along_axis(Nv, keep, np.take_along_axis(N, keep, axis=1), axis=1)
            for eps in eps_grid:
                sigma = np.float32(np.sqrt(2.0) * 2.0 * np.sqrt(setup.d) / eps)
                # The full model's query norm moves with the noise; the truncated one keeps |q|.
                norms = np.linalg.norm(q + sigma * gd, axis=1) if v is None else q_norm
                noisy = (S + sigma * Nv) / norms[:, None] - 1.0   # cosine - 1, as NearestNeighbor
                author = np.maximum.reduceat(noisy, setup.starts, axis=1)
                correct[(v, eps)][rows] = author.argmax(axis=1) == setup.true_index[rows]
                accept[(v, eps)][rows] = rejection_score(author)
    return [{"neighbours": v or "all", "eps": eps, "seed": seed,
             **metrics(correct[(v, eps)], accept[(v, eps)], setup)} for v, eps in keys]


def sim_run(setup: Setup, Q: np.ndarray, eps: float, seed: int, neighbours: int | None) -> dict:
    """A single (eps, variant) of :func:`sim_seed`; used for the zero-noise identity check."""
    row = sim_seed(setup, Q, seed, [neighbours], (eps,))[0]
    return {k: row[k] for k in METRICS}


def stage_sim(setup: Setup, path: Path, neighbour_seeds: int) -> pd.DataFrame:
    Q = setup.clean_queries()
    partial, done = _partial(path)
    finished = set() if done is None else set(done["seed"].astype(int))
    for seed in range(SIM_SEEDS):
        if seed in finished:
            continue
        t = time.time()
        variants = [None] + (list(SIM_NEIGHBOURS) if seed < neighbour_seeds else [])
        _append(partial, sim_seed(setup, Q, seed, variants))
        print(f"  sim seed {seed} ({len(variants)} variants, {time.time() - t:.0f}s)", flush=True)
    frame = pd.read_csv(partial)
    frame.to_csv(path, index=False)
    partial.unlink()
    for (v, eps), group in frame.groupby(["neighbours", "eps"], sort=False):
        print(f"  sim M={v} eps={eps:g}: " + " ".join(f"{k}={group[k].mean():.4f}" for k in METRICS))
    return frame


# --- Q-Q check of the normal approximation ------------------------------------------------------------

def stage_qq(setup: Setup, margins: dict, path: Path) -> dict:
    pt = margins["per_target"]
    rng = np.random.default_rng(11)
    pick = pt.sample(n=min(QQ_DIRECTIONS, len(pt)), random_state=11)
    W = setup.Kw[pick["a_index"].to_numpy()] - setup.Kw[pick["closest_impostor_index"].to_numpy()]
    W = W / np.linalg.norm(W, axis=1, keepdims=True)
    b = 2 * np.sqrt(setup.d) / QQ_EPS
    proj = np.empty((QQ_DRAWS, len(W)))
    for s in range(0, QQ_DRAWS, 1000):
        eta = rng.laplace(0.0, b, (1000, setup.d))
        proj[s:s + 1000] = eta @ W.T.astype(np.float64)
    theory_sd = 2 * np.sqrt(2 * setup.d) / QQ_EPS
    ratios = proj.std(axis=0) / theory_sd
    np.save(path, proj / theory_sd)
    return {"eps": QQ_EPS, "sd_ratio_mean": float(ratios.mean()), "sd_ratio_min": float(ratios.min()),
            "sd_ratio_max": float(ratios.max())}


# --- driver ----------------------------------------------------------------------------------------

def run_dataset(source: str, geometry: str, out: Path, stages: set, force: bool,
                neighbour_seeds: int) -> None:
    res = out / "results"
    res.mkdir(parents=True, exist_ok=True)
    tag = f"{source}_{geometry}"
    setup = load_setup(source, geometry)

    meta_path = res / f"meta_{tag}.json"
    if force or not meta_path.exists():
        meta = {"n_known_docs": len(setup.known_raw), "n_known_authors": len(setup.authors),
                "n_targets": len(setup.true_index), "n_in_set": int(setup.in_set.sum()),
                "n_ood": int((~setup.in_set).sum()), "d": setup.d,
                "chance": chance_levels(setup)}
        meta_path.write_text(json.dumps(meta, indent=2))

    if "empirical" in stages:
        path = res / f"empirical_{tag}.csv"
        if force or not path.exists():
            print(f"[{tag}] empirical Laplace sweep", flush=True)
            stage_empirical(setup, path)

    margins_path = res / f"margins_{tag}.npz"
    if stages & {"margins", "qq"}:
        if force or not margins_path.exists():
            print(f"[{tag}] margins + theory", flush=True)
            stage_margins(setup, margins_path)
    if "qq" in stages:
        qq_path = res / f"qq_{tag}.npy"
        if force or not qq_path.exists():
            info = stage_qq(setup, load_margins(margins_path), qq_path)
            (res / f"qq_{tag}.json").write_text(json.dumps(info, indent=2))
            print(f"[{tag}] Q-Q: {info}", flush=True)

    if "sim" in stages:
        path = res / f"sim_{tag}.csv"
        if force or not path.exists():
            print(f"[{tag}] model simulation", flush=True)
            stage_sim(setup, path, neighbour_seeds)

    if margins_path.exists():
        theory = theory_curves(setup, load_margins(margins_path))
        theory.to_csv(res / f"theory_{tag}.csv", index=False)
        eps_star(setup, load_margins(margins_path)).to_csv(res / f"eps_star_{tag}.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", choices=["swe_chat", "wildchat"])
    parser.add_argument("--geometry", choices=["raw", "zscore"], default="zscore")
    parser.add_argument("--stages", nargs="+", default=["empirical", "margins", "qq", "sim"],
                        choices=["empirical", "margins", "qq", "sim"])
    parser.add_argument("--neighbour-seeds", type=int, default=SIM_SEEDS,
                        help="seeds for the truncated-neighbour checks (default: all sim seeds)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--summary", action="store_true",
                        help="only build figures, the long CSV and summary.md from saved results")
    args = parser.parse_args()
    if not args.summary:
        if not args.source:
            parser.error("--source is required unless --summary")
        run_dataset(args.source, args.geometry, args.out, set(args.stages), args.force,
                    args.neighbour_seeds)
    else:
        import dp_reproduction_report
        dp_reproduction_report.build(args.out)


if __name__ == "__main__":
    main()
