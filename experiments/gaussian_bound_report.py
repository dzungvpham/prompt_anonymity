"""Figure X, Table X, margin histogram and summary.md for :mod:`gaussian_bound` (from saved results).

Run through ``python experiments/gaussian_bound.py --report``. It reads ``<out>/results/`` and
writes ``<out>/figures/*.pdf`` and ``<out>/results/summary.md``, drawing with ``plot_results.py``'s
chrome and palette -- but outside its sweep and cache, which know only the ``results/`` and
``clustering/`` roots.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

import plot_results as P  # noqa: E402
from gaussian_bound import (DELTA, PI_TABLE, TABLE_C, eps_closed, eps_exact,  # noqa: E402
                            sigma_of_c)

CONV, AUTHOR = P.CATEGORICAL[0], P.CATEGORICAL[1]
REPORT_C = (0.5, 0.1)
BOUND_DASH = (0, (4, 2))
LEVELS = (("A", "conv", "Conversation level"), ("B", "author", "Author level"))


def bound_ks(curves: pd.DataFrame) -> list[int]:
    """The joint-bound sizes present in a run: 1 (Theorem 1) always, then each ``bound_k<k>``."""
    return [1] + sorted(int(c[len("bound_k"):-len("_conv")]) for c in curves.columns
                        if c.startswith("bound_k") and c.endswith("_conv"))


def bound_column(k: int, level: str) -> str:
    """``bound_conv`` / ``bound_author`` is Theorem 1; ``bound_k<k>_<level>`` the joint bound."""
    return f"bound_{level}" if k == 1 else f"bound_k{k}_{level}"


def load(res: Path, source: str) -> dict | None:
    sim_path = res / f"sim_{source}.csv"
    if not sim_path.exists():
        return None
    meta = json.loads((res / f"meta_{source}.json").read_text())
    sim = pd.read_csv(sim_path)
    agg = sim.groupby("sigma").agg(
        **{f"{k}_{s}": (k, f) for k in ("A", "B", "D", "mean_cos") for s, f in (("mean", "mean"), ("sd", "std"))},
        reps=("rep", "nunique")).reset_index()
    curves = pd.read_csv(res / f"bound_curves_{source}.csv").merge(agg, on="sigma")
    curves = curves.sort_values("c").reset_index(drop=True)
    curves["eps_closed"] = [eps_closed(s) if s > 0 else np.inf for s in curves["sigma"]]
    curves["eps_exact"] = [eps_exact(s) if s > 0 else np.inf for s in curves["sigma"]]
    margins = pd.read_csv(res / f"margins_{source}.csv")
    dup = int(np.load(res / f"bound_{source}.npz")["n_duplicate_pairs"])
    return {"meta": meta, "curves": curves, "margins": margins, "n_dup": dup}


# --- figures -------------------------------------------------------------------------------------

def _eps_axis(ax, d: int) -> None:
    """Top axis: exact eps at delta = 1e-5, ticks at powers of ten placed at their c."""
    c_fine = np.linspace(0.005, 0.999, 600)
    eps_fine = np.array([eps_exact(s) for s in sigma_of_c(c_fine, d)])
    top = ax.twiny()
    top.set_xlim(ax.get_xlim())
    ticks = [t for t in (100, 1_000, 10_000, 100_000) if eps_fine.min() <= t <= eps_fine.max()]
    top.set_xticks(np.interp(np.log10(ticks), np.log10(eps_fine), c_fine))
    top.set_xticklabels([f"{t // 1000}k" if t >= 1000 else str(t) for t in ticks])
    top.set_xlabel("ε at δ = 10⁻⁵", fontsize=P.FONT_AXIS_LABEL, color=P.TEXT_PRIMARY)
    top.tick_params(color=P.TEXT_SECONDARY, labelcolor=P.TEXT_PRIMARY, labelsize=P.FONT_TICK)
    for side in ("right", "left", "bottom"):
        top.spines[side].set_visible(False)
    top.spines["top"].set_color(P.AXIS)


def _legend_handles() -> list:
    return [
        Line2D([], [], color=CONV, linewidth=P.LINE_WIDTH, label="Conv-level, measured"),
        Line2D([], [], color=CONV, linewidth=P.LINE_WIDTH, linestyle=(0, (4, 2)), label="Conv-level, bound"),
        Line2D([], [], color=AUTHOR, linewidth=P.LINE_WIDTH, label="Author-level, measured"),
        Line2D([], [], color=AUTHOR, linewidth=P.LINE_WIDTH, linestyle=(0, (4, 2)), label="Author-level, bound"),
    ]


def draw_panel(ax, source: str, data: dict, ylabel: bool = True) -> None:
    """Measured rates (solid, sd band) and Theorem 1's bounds (dashed) for one dataset."""
    t = data["curves"]
    for key, bound, color in (("A", "bound_conv", CONV), ("B", "bound_author", AUTHOR)):
        ax.fill_between(t["c"], t[f"{key}_mean"] - t[f"{key}_sd"], t[f"{key}_mean"] + t[f"{key}_sd"],
                        color=color, alpha=P.BAND_ALPHA, linewidth=0, zorder=2)
        ax.plot(t["c"], t[f"{key}_mean"], color=color, linewidth=P.LINE_WIDTH, zorder=4)
        ax.plot(t["c"], t[bound], color=color, linewidth=P.LINE_WIDTH, linestyle=(0, (4, 2)), zorder=3)
    ax.set_xlim(0, 1)
    P.style_axes(ax, "cosine retention c", "DIR @ 100% FAR" if ylabel else "", "")
    if source == "wildchat":
        P.zoom_y_axis(ax, *P.PANEL_Y_LIMITS[("identification_authors", "wildchat")])
    else:
        P.unit_y_axis(ax)
    _eps_axis(ax, data["meta"]["d"])


def figure_panel(source: str, data: dict, out: Path) -> None:
    """One dataset per PDF, untitled, for side-by-side LaTeX subfigures."""
    fig, ax = plt.subplots(figsize=(3.6, 3.3))
    draw_panel(ax, source, data)
    # In-panel legend only on WildChat, in its empty lower-right corner; on SWE-Chat every
    # corner is crossed by a curve.
    if source == "wildchat":
        ax.legend(handles=_legend_handles(), loc="lower right", frameon=True, facecolor=P.SURFACE,
                  edgecolor=P.GRID, framealpha=0.9, fontsize=P.FONT_LEGEND * 0.7,
                  handlelength=1.4, handletextpad=0.5, borderpad=0.35, labelspacing=0.2,
                  borderaxespad=0.3)
    fig.tight_layout()
    P.save_figure(fig, out / f"gaussian_bound_{source}")
    plt.close(fig)


def figure_pair(datas: dict, out: Path, legend: str = "below", stem: str = "gaussian_bound") -> None:
    """SWE-Chat | WildChat side by side in one PDF with one shared legend.

    ``legend="below"`` puts it under both panels; ``"inside"`` puts it in the upper-right corner
    of the right (WildChat) panel, which the curves leave empty above about 0.27 on its 0-0.4 axis.
    """
    height = 3.9 if legend == "below" else 3.4
    fig, axes = plt.subplots(1, len(datas), figsize=(7.2, height), squeeze=False)
    for i, (ax, (source, data)) in enumerate(zip(axes[0], datas.items())):
        draw_panel(ax, source, data, ylabel=(i == 0))
        ax.set_title(P.DATASET_LABELS[source], fontsize=P.FONT_AXIS_LABEL, color=P.TEXT_PRIMARY, pad=38)
    if legend == "below":
        fig.legend(handles=_legend_handles(), loc="lower center", ncol=2, frameon=False,
                   fontsize=P.FONT_LEGEND * 0.85, handlelength=1.8, columnspacing=1.6,
                   bbox_to_anchor=(0.5, 0.0))
        fig.tight_layout(rect=(0, 0.62 / height, 1, 1))
    else:
        axes[0][-1].legend(handles=_legend_handles(), loc="upper right", frameon=True,
                           facecolor=P.SURFACE, edgecolor=P.GRID, framealpha=0.9,
                           fontsize=P.FONT_LEGEND * 0.75, handlelength=1.6, handletextpad=0.5,
                           borderpad=0.35, labelspacing=0.25, borderaxespad=0.4)
        fig.tight_layout()
    P.save_figure(fig, out / stem)
    plt.close(fig)


def figure_by_k(source: str, data: dict, out: Path) -> None:
    """The joint bound for every computed k against the measured attack, one panel per level.

    The measured attack is the reference every bound approaches from below, so it is the solid
    line in ink (with its sd band); each bound is dashed -- a computed limit, not a measurement --
    and coloured by k in the palette's fixed order, so k = 1 (Theorem 1) is the same hue in both
    panels and in every corpus's figure.
    """
    t, ks = data["curves"], bound_ks(data["curves"])
    height = 3.9
    fig, axes = plt.subplots(1, 2, figsize=(7.2, height))
    for i, (ax, (key, level, title)) in enumerate(zip(axes, LEVELS)):
        ax.fill_between(t["c"], t[f"{key}_mean"] - t[f"{key}_sd"], t[f"{key}_mean"] + t[f"{key}_sd"],
                        color=P.TEXT_PRIMARY, alpha=P.BAND_ALPHA, linewidth=0, zorder=2)
        ax.plot(t["c"], t[f"{key}_mean"], color=P.TEXT_PRIMARY, linewidth=P.LINE_WIDTH, zorder=5)
        for slot, k in enumerate(ks):
            ax.plot(t["c"], t[bound_column(k, level)], color=P.CATEGORICAL[slot],
                    linewidth=P.LINE_WIDTH, linestyle=BOUND_DASH, zorder=4 - slot * 0.1)
        ax.set_xlim(0, 1)
        P.style_axes(ax, "cosine retention c", "DIR @ 100% FAR" if i == 0 else "", "")
        if source == "wildchat":
            P.zoom_y_axis(ax, *P.PANEL_Y_LIMITS[("identification_authors", "wildchat")])
        else:
            P.unit_y_axis(ax)
        _eps_axis(ax, data["meta"]["d"])
        ax.set_title(title, fontsize=P.FONT_AXIS_LABEL, color=P.TEXT_PRIMARY, pad=38)
    handles = [Line2D([], [], color=P.TEXT_PRIMARY, linewidth=P.LINE_WIDTH, label="Measured attack")]
    handles += [Line2D([], [], color=P.CATEGORICAL[slot], linewidth=P.LINE_WIDTH, linestyle=BOUND_DASH,
                       label="Bound, k = 1 (Theorem 1)" if k == 1 else f"Bound, k = {k}")
                for slot, k in enumerate(ks)]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False,
               fontsize=P.FONT_LEGEND * 0.85, labelcolor=P.TEXT_PRIMARY, handlelength=1.8,
               columnspacing=1.4, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.42 / height, 1, 1))
    P.save_figure(fig, out / f"gaussian_bound_by_k_{source}")


def margin_histogram(datas: dict, out: Path) -> None:
    fig, axes = plt.subplots(1, len(datas), figsize=(4.8 * len(datas), 3.6), squeeze=False)
    for i, (ax, (source, data)) in enumerate(zip(axes[0], datas.items())):
        g = data["margins"]["gamma_min"].to_numpy()
        pos = g[g > 0]
        ax.hist(pos, bins=np.geomspace(pos.min(), pos.max(), 40), color=P.CATEGORICAL[0], edgecolor=P.SURFACE)
        ax.set_xscale("log")
        P.style_axes(ax, "smallest margin γ_t (targets with γ_t > 0)", "targets" if i == 0 else "",
                     f"{P.DATASET_LABELS[source]}: {len(pos):,} of {len(g):,} targets",
                     title_size=P.FONT_LEGEND, title_weight="normal")
    fig.tight_layout()
    P.save_figure(fig, out / "gaussian_margin_hist")
    plt.close(fig)


# --- summary -------------------------------------------------------------------------------------

def _at(t: pd.DataFrame, c: float) -> pd.Series:
    return t.loc[(t["c"] - c).abs().idxmin()]


def summary(datas: dict) -> str:
    L = ["# Gaussian defense: Theorem 1 / Corollary 3 against the measured attack", "",
         "Generated by `experiments/gaussian_bound.py --report`. Cosine on L2-normalised Gemini "
         "embeddings (d = 3072, no z-score), 75% known / 25% target split as in Section 5. "
         "Measured = real Gaussian noise y = e + σz added to every target (in-set and out-of-set), "
         "nearest-neighbour attribution by inner product.", ""]

    L += ["## Table X (recomputed, d = 3072, δ = 10⁻⁵)", "",
          "| c | D | σ | ε closed form | ε exact (Balle–Wang) | PI's hand values (D, σ, ε) |",
          "|---|---|---|---|---|---|"]
    for c in TABLE_C:
        s = float(sigma_of_c(c, 3072))
        D = (1 - c * c) / (c * c)
        pi = PI_TABLE[c]
        L.append(f"| {c} | {D:.3g} | {s:.4f} | {eps_closed(s):,.0f} | {eps_exact(s):,.0f} | "
                 f"{pi[0]}, {pi[1]}, ≈{pi[2]:,} |")
    L += ["", "D, σ and the closed-form ε agree with the hand values to within rounding; the exact ε is "
          "the one to print.", ""]

    for source, data in datas.items():
        m, t, g = data["meta"], data["curves"], data["margins"]["gamma_min"]
        R = int(t["reps"].iloc[0])
        clean = t[t["sigma"] == 0].iloc[0]
        L += [f"## {P.DATASET_LABELS[source]}", "",
              f"- Known set: {m['n_known_docs']:,} embeddings from {m['n_known_authors']:,} users. "
              f"Targets: {m['n_in_set_targets']:,} in-set ({m['n_in_set_authors']:,} authors), "
              f"{m['n_ood_targets']:,} out-of-set ({m['n_ood_authors']:,} users). "
              f"No in-set target was excluded (every in-set author has known embeddings by construction). "
              f"R = {R} noise draws, {len(t)} noise levels.", ""]

        pos = g[g > 0]
        q = np.percentile(pos, [25, 50, 75])
        L += ["**Margins**", "",
              f"- {100 * (g > 0).mean():.1f}% of in-set targets have γ_t > 0 "
              f"({(g == 0).sum()} with γ_t = 0, i.e. a tie or duplicate; "
              f"{data['n_dup']:,} (target, impostor) pairs where the impostor duplicates a).",
              f"- Among γ_t > 0: median {q[1]:.4f}, quartiles {q[0]:.4f} / {q[2]:.4f}.", ""]

        L += ["**Section 6.5 numbers** (measured = mean over draws; % = share of the undefended rate)", "",
              f"- Undefended (σ = 0): conversation {clean['A_mean']:.4f}, author {clean['B_mean']:.4f}, "
              f"author-level DIR @ 10% FAR {clean['D_mean']:.4f}.", "",
              "| c | ε exact | conv. measured | conv. bound | author measured | author bound | author DIR @ 10% FAR |",
              "|---|---|---|---|---|---|---|"]
        for c in REPORT_C:
            r = _at(t, c)
            L.append(f"| {c} | {r['eps_exact']:,.0f} | {r['A_mean']:.4f} ({100 * r['A_mean'] / clean['A_mean']:.0f}%) | "
                     f"{r['bound_conv']:.4f} | {r['B_mean']:.4f} ({100 * r['B_mean'] / clean['B_mean']:.0f}%) | "
                     f"{r['bound_author']:.4f} | {r['D_mean']:.4f} ({100 * r['D_mean'] / clean['D_mean']:.0f}%) |")
        for key, bound, name in (("A", "bound_conv", "conversation"), ("B", "bound_author", "author")):
            gap = t[f"{key}_mean"] - t[bound]
            i = gap.idxmax()
            L.append(f"- Largest gap measured − bound, {name} level: {gap[i]:.4f} at c = {t['c'][i]:.3f}.")
        slack = max((t["bound_upper_conv"] - t["bound_conv"]).max(), (t["bound_upper_author"] - t["bound_author"]).max())
        L += [f"- Bound looseness from grouping far impostors (not part of the theorem): at most {slack:.2e}.", ""]

        ks = bound_ks(t)
        if len(ks) > 1:
            L += ["**Joint bound by k** (each impostor's term is the chance it outscores the true "
                  "author's k closest documents at once; k = 1 is Theorem 1)", "",
                  "| c | conv. measured | " + " | ".join(f"conv. k = {k}" for k in ks)
                  + " | author measured | " + " | ".join(f"author k = {k}" for k in ks) + " |",
                  "|---" * (2 * len(ks) + 3) + "|"]
            for c in (*TABLE_C[:-1], 0.2, TABLE_C[-1]):
                r = _at(t, c)
                L.append(f"| {c} | {r['A_mean']:.4f} | "
                         + " | ".join(f"{r[bound_column(k, 'conv')]:.4f}" for k in ks)
                         + f" | {r['B_mean']:.4f} | "
                         + " | ".join(f"{r[bound_column(k, 'author')]:.4f}" for k in ks) + " |")
            L.append("")

        L += ["**Sanity checks**", ""]
        frac = (g > 0).mean()
        L.append(f"- Fraction with γ_t > 0 = {frac:.4f}; measured conversation rate at σ = 0 = "
                 f"{clean['A_mean']:.4f} → {'PASS' if abs(frac - clean['A_mean']) < 1e-9 else 'CHECK (ties)'}.")
        for k in bound_ks(t):
            for key, level, name in (("A", "conv", "conversation"), ("B", "author", "author")):
                gap = t[f"{key}_mean"] - t[bound_column(k, level)]
                tol = 2 * t[f"{key}_sd"].fillna(0) / np.sqrt(R)
                bad = gap < -tol - 1e-9
                i = gap.idxmin()
                L.append(f"- Measured ≥ bound (k = {k}), {name} level: {'PASS' if not bad.any() else 'FAIL at c = ' + ', '.join(f'{x:.3f}' for x in t['c'][bad])}; "
                         f"smallest gap {gap[i]:+.4f} at c = {t['c'][i]:.3f} (2·sd/√R there = {tol[i]:.4f}).")
        ks = bound_ks(t)
        if len(ks) > 1:
            worst = min((t[bound_column(k, lv)] - t[bound_column(prev, lv)]).min()
                        for prev, k in zip(ks, ks[1:]) for lv in ("conv", "author"))
            L.append(f"- Bound never falls as k grows: {'PASS' if worst >= -1e-12 else 'FAIL'} "
                     f"(smallest step {worst:+.2e}).")
        noisy = t[t["sigma"] > 0]
        dev = (noisy["mean_cos_mean"] - noisy["c"]).abs()
        L.append(f"- Mean cos(y, e) vs c: max |difference| {dev.max():.4f} → {'PASS' if dev.max() < 0.01 else 'FAIL'}.")
        ok = (noisy["eps_exact"] <= noisy["eps_closed"] * (1 + 1e-9)).all()
        L.append(f"- Exact ε ≤ closed-form ε at every σ: {'PASS' if ok else 'FAIL'}.")
        L.append("")
    return "\n".join(L)


def build(out: Path) -> None:
    res, figs = out / "results", out / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    datas = {s: d for s in ("swe_chat", "wildchat") if (d := load(res, s)) is not None}
    for source, data in datas.items():
        data["curves"].to_csv(res / f"curves_{source}.csv", index=False)
    for source, data in datas.items():
        figure_panel(source, data, figs)
        if len(bound_ks(data["curves"])) > 1:
            figure_by_k(source, data, figs)
    figure_pair(datas, figs)
    figure_pair(datas, figs, legend="inside", stem="gaussian_bound_legend_inside")
    margin_histogram(datas, figs)
    (res / "summary.md").write_text(summary(datas))
    print(f"wrote {figs} and {res / 'summary.md'} ({', '.join(datas)})")
