#!/usr/bin/env python
"""Gaussian noise on the target embeddings: Theorem 1 / Corollary 3 bounds against the measured attack.

Implements the PI's Section 6 "Measurement spec" (Figure X, Table X, the Section 6.5 numbers):

1. Margins  gamma_{t,b} = (<e,a> - <e,b>) / sqrt(2 - 2<a,b>) for every in-set target t (a = the true
   author's known embedding closest to e) against every known embedding b of another author.
2. Grid of cosine retentions c, sigma(c) = sqrt((1 - c^2) / (c^2 d)), plus sigma = 0.
3. Bounds: l_t = max(0, 1 - sum_b Phi(-gamma_{t,b} / sigma)); conversation level = mean l_t,
   author level = mean over authors of 1 - prod_t (1 - l_t)  (Theorem 1, Corollary 3).
4. Measurement: y = e + sigma z with real Gaussian noise, NN attribution by inner product, R reps.
5. Privacy axis: eps at delta = 1e-5, closed form and exact (Balle & Wang 2018, root-finding).
6. Open set: out-of-set targets are noised too, the paper's own rejection score (T-norm) is
   recalibrated at each sigma, and author-level DIR @ 10% FAR is reported.

Geometry is cosine on the L2-normalised Gemini vectors (no z-score), as the spec and the theorem
assume: isotropic noise is isotropic only in that space. The split is the paper's 75% known /
25% target one, loaded through :func:`dp_reproduction.load_setup`.

Speed: scores of the noised target are <e + sigma z, k> = <e, k> + sigma <z, k>, so one draw z per
rep is shared by every sigma (each sigma still sees exactly N(0, sigma^2 I) noise; the curves are
paired across sigma). Reps run in parallel worker processes.

Usage::

    python experiments/gaussian_bound.py --source swe_chat --reps 20
    python experiments/gaussian_bound.py --source wildchat --reps 5 --workers 4
    python experiments/gaussian_bound.py --report        # figure + summary.md from saved results
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import log_ndtr, ndtr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dp_reproduction as DR  # noqa: E402  (loader, split, metrics A-D incl. DIR@10% FAR)

from prompt_anonymity.attacks.ood.rejection import rejection_score  # noqa: E402

DELTA = 1e-5
TABLE_C = (0.9, 0.7, 0.5, 0.3, 0.1)
#: ~50 retentions from 0.02 to 0.99 (the spec), the Table X / Section 6.5 points, and c = 1 (sigma = 0).
C_GRID = np.unique(np.round(np.concatenate([np.linspace(0.02, 0.99, 50), TABLE_C, [1.0]]), 6))
#: Per target, the smallest EXACT_MARGINS margins enter the bound exactly; the rest are sorted into
#: TAIL_GROUPS equal-size groups that each contribute size x Phi(-(group minimum) / sigma). That can
#: only lower the bound, so it stays valid. (SWE-Chat has fewer known embeddings: all exact.)
EXACT_MARGINS = 4000
TAIL_GROUPS = 512
#: 2 - 2<a,b> below this is treated as b == a (duplicate prompt across users): margin 0, per the spec.
DUPLICATE_TOL = 1e-6
DEFAULT_OUT = Path(__file__).resolve().parent / "results" / "gaussian_bound"

#: The PI's hand-calculated Table X: c -> (D, sigma, eps).
PI_TABLE = {0.9: (0.23, 0.0087, 27_300), 0.7: (1.04, 0.018, 6_400), 0.5: (3.0, 0.031, 2_400),
            0.3: (10.1, 0.057, 770), 0.1: (99, 0.18, 120)}


# --- noise scale and privacy -------------------------------------------------------------------------

def sigma_of_c(c, d: int):
    c = np.asarray(c, float)
    with np.errstate(divide="ignore"):
        return np.sqrt((1 - c ** 2) / (c ** 2 * d))


def eps_closed(sigma: float, delta: float = DELTA) -> float:
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


# --- step 1 + 3: margins and bounds --------------------------------------------------------------

def stage_bound(setup: DR.Setup, sigmas: np.ndarray, path: Path, block: int = 256) -> None:
    Kh = setup.Kh
    n_known = len(Kh)
    in_rows = np.flatnonzero(setup.in_set)
    n_exact = min(EXACT_MARGINS, n_known)
    ell = np.zeros((len(setup.true_index), len(sigmas)))
    ell_upper = np.zeros_like(ell)            # tail groups at their maximum: grouping slack only
    gamma_min = np.full(len(setup.true_index), np.nan)
    n_dup = 0
    t0 = time.time()
    for b0 in range(0, len(in_rows), block):
        rows = in_rows[b0:b0 + block]
        S = (setup.target_unit[rows] @ Kh.T).astype(np.float64)
        truth = setup.true_index[rows]
        lo_all, hi_all = setup.starts[truth], setup.stops[truth]
        a_idx = np.array([lo + int(S[r, lo:hi].argmax()) for r, (lo, hi) in enumerate(zip(lo_all, hi_all))])
        dist2 = 2.0 - 2.0 * (Kh[a_idx] @ Kh.T).astype(np.float64)
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
        g = np.sort(gam, axis=1)                                     # +inf (own docs) sort last
        exact, rest = g[:, :n_exact], g[:, n_exact:]
        if rest.shape[1]:
            edges = np.linspace(0, rest.shape[1], TAIL_GROUPS + 1).astype(int)
            edges = np.unique(edges)
            size = np.diff(edges).astype(float)
            g_lo, g_hi = rest[:, edges[:-1]], rest[:, edges[1:] - 1]
        for j, s in enumerate(sigmas):
            if s == 0:
                ell[rows, j] = ell_upper[rows, j] = (gamma_min[rows] > 0).astype(float)
                continue
            tot = ndtr(-exact / s).sum(axis=1)
            tot_hi = tot.copy()
            if rest.shape[1]:
                tot = tot + (ndtr(-g_lo / s) * size).sum(axis=1)
                tot_hi = tot_hi + (ndtr(-g_hi / s) * size).sum(axis=1)
            ell[rows, j] = np.maximum(0.0, 1.0 - tot)
            ell_upper[rows, j] = np.maximum(0.0, 1.0 - tot_hi)
        print(f"  bound {min(b0 + block, len(in_rows)):,}/{len(in_rows):,} ({time.time() - t0:.0f}s)",
              flush=True)
    np.savez_compressed(path, ell=ell, ell_upper=ell_upper, gamma_min=gamma_min, sigmas=sigmas,
                        n_duplicate_pairs=n_dup)


def bound_curves(setup: DR.Setup, z) -> pd.DataFrame:
    in_set = setup.in_set
    labels = setup.target_labels[in_set]
    out = {}
    for name, key in (("bound", "ell"), ("bound_upper", "ell_upper")):
        p = z[key][in_set]
        conv = p.mean(axis=0)
        log_miss = pd.DataFrame(np.log1p(-np.minimum(p, 1 - 1e-15))).groupby(labels).sum()
        author = (1 - np.exp(log_miss.to_numpy())).mean(axis=0)
        out[f"{name}_conv"], out[f"{name}_author"] = conv, author
    return pd.DataFrame({"sigma": z["sigmas"], **out})


# --- step 4 + 6: measurement ---------------------------------------------------------------------

_SETUP: DR.Setup | None = None


def sim_rep(args) -> list[dict]:
    rep, sigmas, block = args
    setup = _SETUP
    rng = np.random.default_rng([rep, 13])
    Kh, n = setup.Kh, len(setup.true_index)
    correct = np.zeros((len(sigmas), n), bool)
    accept = np.empty((len(sigmas), n))
    cos_sum = np.zeros(len(sigmas))
    t0 = time.time()
    for start in range(0, n, block):
        rows = slice(start, start + block)
        e = setup.target_unit[rows]
        S0 = e @ Kh.T
        Z = rng.standard_normal(e.shape, dtype=np.float32)
        N = Z @ Kh.T
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
    out = [{"rep": rep, "sigma": s, "mean_cos": cos_sum[j] / n, **DR.metrics(correct[j], accept[j], setup)}
           for j, s in enumerate(sigmas)]
    print(f"  rep {rep} done ({time.time() - t0:.0f}s)", flush=True)
    return out


def stage_sim(setup: DR.Setup, sigmas: np.ndarray, path: Path, reps: int, workers: int,
              block: int = 512) -> None:
    global _SETUP
    _SETUP = setup
    partial = path.with_name(path.stem + ".partial.csv")
    done = set(pd.read_csv(partial)["rep"].astype(int)) if partial.exists() else set()
    todo = [(r, sigmas, block) for r in range(reps) if r not in done]
    if todo:
        ctx = mp.get_context("fork")
        with ctx.Pool(min(workers, len(todo))) as pool:
            for rows in pool.imap_unordered(sim_rep, todo):
                pd.DataFrame(rows).to_csv(partial, mode="a", header=not partial.exists(), index=False)
    frame = pd.read_csv(partial).sort_values(["sigma", "rep"])
    frame.to_csv(path, index=False)


# --- driver --------------------------------------------------------------------------------------

def run(source: str, out: Path, reps: int, workers: int, force: bool) -> None:
    res = out / "results"
    res.mkdir(parents=True, exist_ok=True)
    setup = DR.load_setup(source, "raw")
    sigmas = sigma_of_c(C_GRID, setup.d)
    meta = {"source": source, "d": setup.d, "n_known_docs": len(setup.Kh),
            "n_known_authors": len(setup.authors), "n_targets": len(setup.true_index),
            "n_in_set_targets": int(setup.in_set.sum()), "n_ood_targets": int((~setup.in_set).sum()),
            "n_in_set_authors": int(pd.Series(setup.target_labels[setup.in_set]).nunique()),
            "n_ood_authors": int(pd.Series(setup.target_labels[~setup.in_set]).nunique()),
            "c_grid": C_GRID.tolist()}
    (res / f"meta_{source}.json").write_text(json.dumps(meta, indent=2))

    bound_path = res / f"bound_{source}.npz"
    if force or not bound_path.exists():
        print(f"[{source}] margins + bounds", flush=True)
        stage_bound(setup, sigmas, bound_path)
    z = np.load(bound_path)
    bound_curves(setup, z).assign(c=C_GRID).to_csv(res / f"bound_curves_{source}.csv", index=False)
    pd.DataFrame({"target": np.flatnonzero(setup.in_set),
                  "author": setup.target_labels[setup.in_set],
                  "gamma_min": z["gamma_min"][setup.in_set]}).to_csv(res / f"margins_{source}.csv", index=False)

    sim_path = res / f"sim_{source}.csv"
    if force and sim_path.exists():
        sim_path.unlink()
    if not sim_path.exists() or pd.read_csv(sim_path)["rep"].nunique() < reps:
        print(f"[{source}] measured attack, {reps} reps on {workers} workers", flush=True)
        stage_sim(setup, sigmas, sim_path, reps, workers)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", choices=["swe_chat", "wildchat"])
    parser.add_argument("--reps", type=int, default=20)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--report", action="store_true", help="only build figure + summary.md")
    args = parser.parse_args()
    if args.report:
        import gaussian_bound_report
        gaussian_bound_report.build(args.out)
        return
    if not args.source:
        parser.error("--source is required unless --report")
    run(args.source, args.out, args.reps, args.workers, args.force)


if __name__ == "__main__":
    main()
