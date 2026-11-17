"""Figures, long-form CSV and summary.md for :mod:`dp_reproduction` (built from its saved results).

Run through ``python experiments/dp_reproduction.py --summary``.
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
from scipy.stats import norm  # noqa: E402

import plot_results as P  # noqa: E402
from dp_reproduction import (DATASET_TITLES, EMPIRICAL_SEEDS, METRIC_TITLES, METRICS,  # noqa: E402
                             QQ_EPS, SIM_NEIGHBOURS)

EMPIRICAL = "Laplace noise (empirical)"
THEOREM = "Theorem 1 (lower bound)"
NORMAL = "Normal approximation"
SIM = "Model simulation"
COLORS = {EMPIRICAL: P.CATEGORICAL[0], THEOREM: P.CATEGORICAL[1], NORMAL: P.CATEGORICAL[2],
          SIM: P.CATEGORICAL[3]}
GEOMETRY_TITLES = {"zscore": "z-scored cosine, pipeline default", "raw": "raw cosine on unit vectors"}

#: Naina's values, read by eye off her three plots (about +/- 0.01). Only the points she drew.
NAINA = {
    "swe_chat": {
        "A": {np.inf: 0.56, 1: 0.024, 100: 0.024, 1000: 0.10, 2000: 0.29, 5000: 0.50, 1e4: 0.53,
              1e5: 0.56},
        "B": {np.inf: 0.78, 1: 0.13, 1000: 0.44, 2000: 0.64, 5000: 0.76, 1e4: 0.78},
        "C": {np.inf: 0.36, 1000: 0.02, 2000: 0.14, 5000: 0.32, 1e4: 0.36},
    },
    "wildchat": {
        "A": {np.inf: 0.26, 1: 0.001, 100: 0.001, 500: 0.006, 1000: 0.037, 2000: 0.17,
              5000: 0.247, 1e4: 0.257},
        "B": {np.inf: 0.257, 1: 0.006, 100: 0.009, 500: 0.022, 1000: 0.073, 2000: 0.193,
              5000: 0.25, 1e4: 0.256},
    },
}


def _style(ax, scale: float = 1.0) -> None:
    ax.set_facecolor(P.SURFACE)
    ax.grid(True, color=P.GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(P.AXIS)
    ax.tick_params(colors=P.TEXT_SECONDARY, labelsize=P.FONT_TICK * scale)


# --- loading ---------------------------------------------------------------------------------------

def tags(res: Path) -> list[str]:
    return sorted(p.stem.removeprefix("empirical_") for p in res.glob("empirical_*.csv")
                  if not p.name.endswith(".partial.csv"))


def load(res: Path, tag: str) -> dict:
    out = {"meta": json.loads((res / f"meta_{tag}.json").read_text()),
           "empirical": pd.read_csv(res / f"empirical_{tag}.csv")}
    for name in ("sim", "theory", "eps_star"):
        path = res / f"{name}_{tag}.csv"
        out[name] = pd.read_csv(path) if path.exists() else None
    margins = res / f"margins_{tag}.csv"
    out["per_target"] = pd.read_csv(margins) if margins.exists() else None
    qq = res / f"qq_{tag}.json"
    out["qq"] = json.loads(qq.read_text()) if qq.exists() else None
    out["qq_proj"] = np.load(res / f"qq_{tag}.npy") if (res / f"qq_{tag}.npy").exists() else None
    return out


def long_table(tag: str, data: dict) -> pd.DataFrame:
    source, geometry = tag.rsplit("_", 1)
    rows = []
    emp = data["empirical"]
    clean = emp[np.isinf(emp["eps"])].iloc[0]
    noisy = emp[np.isfinite(emp["eps"])]
    for metric in METRICS:
        rows.append({"eps": np.inf, "metric": metric, "source": "No noise", "mean": clean[metric], "sd": 0.0})
        rows.append({"eps": np.nan, "metric": metric, "source": "Chance",
                     "mean": data["meta"]["chance"][metric], "sd": 0.0})
        for eps, group in noisy.groupby("eps"):
            rows.append({"eps": eps, "metric": metric, "source": EMPIRICAL,
                         "mean": group[metric].mean(), "sd": group[metric].std(ddof=1)})
        if data["sim"] is not None:
            for (m, eps), group in data["sim"].groupby(["neighbours", "eps"]):
                name = SIM if str(m) == "all" else f"{SIM} (M={m})"
                rows.append({"eps": eps, "metric": metric, "source": name,
                             "mean": group[metric].mean(), "sd": group[metric].std(ddof=1)})
    frame = pd.DataFrame(rows)
    if data["theory"] is not None:
        frame = pd.concat([frame, data["theory"]], ignore_index=True)
    frame.insert(0, "geometry", geometry)
    frame.insert(0, "dataset", source)
    return frame


# --- figures ---------------------------------------------------------------------------------------

def overlay_figure(tag: str, table: pd.DataFrame, out: Path, metrics=METRICS, suffix: str = "") -> None:
    """All four panels by default; ``metrics=("A", "B")`` is the two panels Theorem 1 covers."""
    source, geometry = tag.rsplit("_", 1)
    wide = len(metrics) > 2
    fig, axes = plt.subplots(1, len(metrics), figsize=(17, 4.6) if wide else (9, 4.9))
    for ax, metric in zip(axes, metrics):
        t = table[table["metric"] == metric]
        clean = t[t["source"] == "No noise"]["mean"].iloc[0]
        chance = t[t["source"] == "Chance"]["mean"].iloc[0]
        ax.axhline(clean, color=P.TEXT_SECONDARY, linewidth=1.2, linestyle=(0, (1, 2)))
        ax.axhline(chance, color=P.TEXT_MUTED, linewidth=1.2, linestyle=P.BASELINE_DASH)
        for name, kwargs in ((THEOREM, {}), (NORMAL, {"linestyle": (0, (5, 2))}), (SIM, {})):
            s = t[t["source"] == name].sort_values("eps")
            if len(s):
                ax.plot(s["eps"], s["mean"], color=COLORS[name], linewidth=P.LINE_WIDTH, **kwargs)
        e = t[t["source"] == EMPIRICAL].sort_values("eps")
        ax.errorbar(e["eps"], e["mean"], yerr=e["sd"], color=COLORS[EMPIRICAL], marker="o",
                    markersize=P.MARKER_SIZE, markeredgecolor=P.SURFACE, markeredgewidth=1.2,
                    linestyle="none", capsize=3, elinewidth=1.4, zorder=5)
        ax.set_xscale("log")
        ax.set_xlim(0.7, 1.4e5)
        ax.set_ylim(0, max(clean * 1.12, 0.02))
        ax.set_title(METRIC_TITLES[metric], fontsize=P.FONT_AXIS_LABEL, color=P.TEXT_PRIMARY)
        ax.set_xlabel("ε (per conversation)", fontsize=P.FONT_AXIS_LABEL, color=P.TEXT_PRIMARY)
        _style(ax)
    axes[0].set_ylabel("rate", fontsize=P.FONT_AXIS_LABEL, color=P.TEXT_PRIMARY)
    handles = [
        Line2D([], [], color=COLORS[EMPIRICAL], marker="o", linestyle="none", markersize=P.MARKER_SIZE,
               label=f"{EMPIRICAL}, mean ± sd over {EMPIRICAL_SEEDS} seeds"),
        Line2D([], [], color=COLORS[THEOREM], linewidth=P.LINE_WIDTH, label=THEOREM),
        Line2D([], [], color=COLORS[NORMAL], linewidth=P.LINE_WIDTH, linestyle=(0, (5, 2)), label=NORMAL),
        Line2D([], [], color=COLORS[SIM], linewidth=P.LINE_WIDTH, label=SIM),
        Line2D([], [], color=P.TEXT_SECONDARY, linewidth=1.2, linestyle=(0, (1, 2)), label="No noise"),
        Line2D([], [], color=P.TEXT_MUTED, linewidth=1.2, linestyle=P.BASELINE_DASH, label="Chance"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=6 if wide else 3, frameon=False, fontsize=P.FONT_LEGEND,
               bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(f"{DATASET_TITLES[source]}: Laplace noise on target Gemini embeddings, NN attack "
                 f"({GEOMETRY_TITLES[geometry]})", fontsize=P.FONT_PANEL_TITLE, color=P.TEXT_PRIMARY)
    fig.tight_layout(rect=(0, 0.07 if wide else 0.14, 1, 1))
    P.save_figure(fig, out / f"dp_overlay_{tag}{suffix}")
    plt.close(fig)


DIR_LEVELS = {"C": "conversation-level", "D": "author-level"}
Y_LABELS = {"B": "DIR", "C": "DIR @ 10% FAR", "D": "DIR @ 10% FAR"}
#: Legend names in dirfar_figure (the PI's wording); the long CSV and summary keep the source names.
LEGEND_NAMES = {THEOREM: "Laplace lower bound"}


def dirfar_figure(tables: dict, out: Path, metrics: tuple, stem: str, geometry: str = "zscore",
                  curves: tuple = ()) -> None:
    """One metric against ε, SWE-Chat | WildChat side by side, one row per metric in ``metrics``.

    ``curves`` are the model lines to draw (any of THEOREM, NORMAL, SIM). Theorem 1 and the normal
    approximation bound a rank-1 event with no rejection threshold, so they are valid only for (A)
    and (B), never for the DIR@FAR panels (C) and (D). With no model line, the empirical means are
    joined by a thin line.
    """
    if any(c in (THEOREM, NORMAL) for c in curves) and any(m in ("C", "D") for m in metrics):
        raise ValueError("Theorem 1 / normal approximation do not bound DIR@FAR (panels C, D)")
    sources = [s for s in DATASET_TITLES if f"{s}_{geometry}" in tables]
    rows = len(metrics)
    fig, grid = plt.subplots(rows, len(sources), figsize=(9, 3.6 * rows + 0.6), squeeze=False)
    for r, metric in enumerate(metrics):
        for c, source in enumerate(sources):
            ax = grid[r][c]
            t = tables[f"{source}_{geometry}"]
            t = t[t["metric"] == metric]
            clean = t[t["source"] == "No noise"]["mean"].iloc[0]
            chance = t[t["source"] == "Chance"]["mean"].iloc[0]
            ax.axhline(clean, color=P.TEXT_SECONDARY, linewidth=1.2, linestyle=(0, (1, 2)))
            ax.axhline(chance, color=P.TEXT_MUTED, linewidth=1.2, linestyle=P.BASELINE_DASH)
            for name in curves:
                s = t[t["source"] == name].sort_values("eps")
                ax.plot(s["eps"], s["mean"], color=COLORS[name], linewidth=P.LINE_WIDTH,
                        linestyle=(0, (5, 2)) if name == NORMAL else "-")
            e = t[t["source"] == EMPIRICAL].sort_values("eps")
            ax.errorbar(e["eps"], e["mean"], yerr=e["sd"], color=COLORS[EMPIRICAL], marker="o",
                        markersize=P.MARKER_SIZE, markeredgecolor=P.SURFACE, markeredgewidth=1.2,
                        linestyle="-" if not curves else "none", linewidth=1.2, capsize=3,
                        elinewidth=1.4, zorder=5)
            ax.set_xscale("log")
            ax.set_xlim(0.7, 1.4e5)
            ax.set_ylim(0, max(clean * 1.12, 0.02))
            title = DATASET_TITLES[source] if rows == 1 else f"{DATASET_TITLES[source]} ({DIR_LEVELS[metric]})"
            ax.set_title(title, fontsize=P.FONT_AXIS_LABEL, color=P.TEXT_PRIMARY)
            ax.set_xlabel("ε (per conversation)", fontsize=P.FONT_AXIS_LABEL, color=P.TEXT_PRIMARY)
            if c == 0:
                ax.set_ylabel(f"{DIR_LEVELS[metric].capitalize()} DIR @ 10% FAR" if rows > 1
                              else Y_LABELS[metric], fontsize=P.FONT_AXIS_LABEL, color=P.TEXT_PRIMARY)
            _style(ax)
    handles = [
        Line2D([], [], color=COLORS[EMPIRICAL], marker="o", linestyle="-" if not curves else "none",
               linewidth=1.2, markersize=P.MARKER_SIZE,
               label=f"{EMPIRICAL}, mean ± sd over {EMPIRICAL_SEEDS} seeds"),
        *[Line2D([], [], color=COLORS[name], linewidth=P.LINE_WIDTH,
                 linestyle=(0, (5, 2)) if name == NORMAL else "-", label=LEGEND_NAMES.get(name, name))
          for name in curves],
        Line2D([], [], color=P.TEXT_SECONDARY, linewidth=1.2, linestyle=(0, (1, 2)), label="No noise"),
        Line2D([], [], color=P.TEXT_MUTED, linewidth=1.2, linestyle=P.BASELINE_DASH, label="Chance"),
    ]
    height = 3.6 * rows + 0.6
    ncol = len(handles) if len(handles) <= 4 else 3
    legend_rows = -(-len(handles) // ncol)
    fig.legend(handles=handles, loc="lower center", ncol=ncol, frameon=False, fontsize=P.FONT_LEGEND,
               bbox_to_anchor=(0.5, 0.0), columnspacing=1.0, handlelength=1.6)
    fig.tight_layout(rect=(0, (0.2 + 0.3 * legend_rows) / height, 1, 1))
    P.save_figure(fig, out / stem)
    plt.close(fig)


def margin_histogram(tag: str, per_target: pd.DataFrame, out: Path) -> None:
    source, geometry = tag.rsplit("_", 1)
    g = per_target["g_min"].to_numpy()
    g = g[g > 0]
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.hist(g, bins=np.geomspace(g.min(), g.max(), 40), color=P.CATEGORICAL[0], edgecolor=P.SURFACE)
    ax.set_xscale("log")
    ax.set_xlabel("g_min (smallest normalised margin to an impostor)", fontsize=P.FONT_AXIS_LABEL)
    ax.set_ylabel("targets", fontsize=P.FONT_AXIS_LABEL)
    n_zero = int((per_target["g_min"] <= 0).sum())
    ax.set_title(f"{DATASET_TITLES[source]} ({geometry}): {len(per_target):,} correctly attributed targets"
                 + (f", {n_zero} with g_min = 0 not shown" if n_zero else ""),
                 fontsize=P.FONT_LEGEND)
    _style(ax)
    fig.tight_layout()
    P.save_figure(fig, out / f"margin_hist_{tag}")
    plt.close(fig)


def qq_figure(datas: dict, out: Path) -> None:
    avail = [(tag, d) for tag, d in datas.items() if d["qq_proj"] is not None]
    if not avail:
        return
    fig, axes = plt.subplots(1, len(avail), figsize=(4.2 * len(avail), 4.2), squeeze=False)
    for ax, (tag, d) in zip(axes[0], avail):
        proj = d["qq_proj"]
        n = len(proj)
        theo = norm.ppf((np.arange(1, n + 1) - 0.5) / n)
        for j in range(proj.shape[1]):
            ax.plot(theo, np.sort(proj[:, j]), color=P.CATEGORICAL[0], alpha=0.35, linewidth=1.0)
        ax.plot([-4, 4], [-4, 4], color=P.TEXT_SECONDARY, linewidth=1.2, linestyle=(0, (1, 2)))
        ax.set_xlim(-3.2, 3.2)
        ax.set_ylim(-3.2, 3.2)
        ax.set_xlabel("normal quantile", fontsize=P.FONT_AXIS_LABEL)
        ax.set_ylabel("projection / (2√(2d)/ε)", fontsize=P.FONT_AXIS_LABEL)
        source, geometry = tag.rsplit("_", 1)
        ax.set_title(f"{DATASET_TITLES[source]} ({geometry}), ε={QQ_EPS:g}\n"
                     f"sd ratio {d['qq']['sd_ratio_mean']:.3f} "
                     f"[{d['qq']['sd_ratio_min']:.3f}, {d['qq']['sd_ratio_max']:.3f}]",
                     fontsize=P.FONT_LEGEND)
        _style(ax)
    fig.tight_layout()
    P.save_figure(fig, out / f"projection_qq")
    plt.close(fig)


# --- summary ---------------------------------------------------------------------------------------

def crossing(eps: np.ndarray, values: np.ndarray, target: float) -> float:
    """Smallest eps (log-interpolated) at which a curve first reaches ``target``; inf if never."""
    order = np.argsort(eps)
    eps, values = np.asarray(eps)[order], np.asarray(values)[order]
    above = np.flatnonzero(values >= target)
    if not above.size:
        return np.inf
    i = above[0]
    if i == 0:
        return float(eps[0])
    x0, x1 = np.log10(eps[i - 1]), np.log10(eps[i])
    y0, y1 = values[i - 1], values[i]
    return float(10 ** (x0 + (target - y0) / (y1 - y0) * (x1 - x0)))


def fmt_eps(x: float) -> str:
    if not np.isfinite(x):
        return "> 1e5 (never)"
    return f"{x:,.0f}" if x >= 10 else f"{x:.2g}"


def pct(series: pd.Series) -> str:
    q = np.percentile(series, [10, 50, 90])
    return f"median {q[1]:.4g} (p10 {q[0]:.4g}, p90 {q[2]:.4g})"


def summary(datas: dict, tables: dict) -> str:
    lines = ["# Laplace-noise reproduction and theory overlay", "",
             "Generated by `experiments/dp_reproduction.py --summary`. Numbers are as computed; "
             "nothing was tuned.", ""]
    lines += ["## Setup", ""]
    for tag, d in datas.items():
        m = d["meta"]
        lines.append(f"- **{tag}**: {m['n_known_docs']:,} known conversations from "
                     f"{m['n_known_authors']:,} users, {m['n_targets']:,} targets "
                     f"({m['n_in_set']:,} in-set, {m['n_ood']:,} out-of-set), d = {m['d']}.")
    lines += ["", "Geometry `zscore` is the runner's default (`--standardize`, known-side z-score "
              "before cosine) and is what Naina's numbers match. `raw` is cosine on the unit-norm "
              "vectors, which is the attack the memo's formulas describe literally. Theory for "
              "`zscore` uses the same formulas in the z-scored space (direction (a - n_j)/s).", ""]

    # Q1
    lines += ["## 1. Does the reproduction match Naina's plots?", "",
              "Her values were read by eye off her figures (about ±0.01). Mine are means over 3 seeds.", ""]
    for tag, table in tables.items():
        source = tag.rsplit("_", 1)[0]
        if source not in NAINA:
            continue
        lines += [f"**{tag}**", "", "| metric | ε | Naina | reproduced | diff |", "|---|---|---|---|---|"]
        gaps = []
        for metric, points in NAINA[source].items():
            t = table[table["metric"] == metric]
            for eps, value in points.items():
                if np.isinf(eps):
                    mine = t[t["source"] == "No noise"]["mean"].iloc[0]
                else:
                    row = t[(t["source"] == EMPIRICAL) & np.isclose(t["eps"], eps)]
                    mine = row["mean"].iloc[0]
                diff = mine - value
                if abs(diff) > max(0.02, 0.1 * value):
                    gaps.append(f"{metric} at ε={eps:g}: {mine:.3f} vs {value:.3f}")
                lines.append(f"| {metric} | {'none' if np.isinf(eps) else f'{eps:g}'} | {value:.3f} | "
                             f"{mine:.3f} | {diff:+.3f} |")
        lines += ["", "Gaps larger than max(0.02, 10%): " + ("; ".join(gaps) if gaps else "none."), ""]
    lines += ["Not reproduced: her usefulness curve (no code for it exists in the repo or anywhere "
              "on this machine, and the PI said it is not needed). Panels (C) for WildChat and (D) "
              "for both datasets are new.", ""]

    # Q2
    lines += ["## 2. ε at which each curve reaches 50% / 95% of its no-noise value", ""]
    for tag, table in tables.items():
        lines += [f"**{tag}**", "", "| metric | curve | ε @ 50% | ε @ 95% |", "|---|---|---|---|"]
        for metric in METRICS:
            t = table[table["metric"] == metric]
            clean = t[t["source"] == "No noise"]["mean"].iloc[0]
            for name in (EMPIRICAL, THEOREM, NORMAL, SIM):
                s = t[t["source"] == name]
                if not len(s):
                    continue
                lines.append(f"| {metric} | {name} | {fmt_eps(crossing(s['eps'], s['mean'], 0.5 * clean))} | "
                             f"{fmt_eps(crossing(s['eps'], s['mean'], 0.95 * clean))} |")
        lines.append("")

    # Q3
    lines += ["## 3. Is the empirical curve always at or above Theorem 1?", ""]
    for tag, table in tables.items():
        for metric in ("A", "B"):
            t = table[table["metric"] == metric]
            emp = t[t["source"] == EMPIRICAL].set_index("eps")
            thm = t[t["source"] == THEOREM].set_index("eps")["mean"]
            if not len(thm):
                continue
            bound = np.interp(np.log10(emp.index), np.log10(thm.index.to_numpy()), thm.to_numpy())
            slack = emp["mean"].to_numpy() - bound
            tol = 2 * emp["sd"].to_numpy() / np.sqrt(EMPIRICAL_SEEDS)
            bad = slack < -tol - 1e-9
            worst = int(np.argmin(slack))
            verdict = ("**VIOLATED** at ε = " + ", ".join(f"{e:g}" for e in emp.index[bad])) if bad.any() \
                else "holds at every ε"
            lines.append(f"- {tag} ({metric}): {verdict}; smallest empirical - bound = {slack[worst]:+.4f} "
                         f"at ε = {emp.index[worst]:g}.")
    lines.append("")

    # Q4
    lines += ["## 4. Largest gap between empirical and model simulation", ""]
    for tag, table in tables.items():
        for metric in METRICS:
            t = table[table["metric"] == metric]
            emp = t[t["source"] == EMPIRICAL].set_index("eps")["mean"]
            parts = []
            for name in [SIM] + [f"{SIM} (M={m})" for m in SIM_NEIGHBOURS]:
                sim = t[t["source"] == name].set_index("eps")["mean"]
                if not len(sim):
                    continue
                gap = (emp - sim.reindex(emp.index)).dropna()
                worst = gap.abs().idxmax()
                label = "all neighbours" if name == SIM else name.split("(")[1].rstrip(")")
                parts.append(f"{label}: {gap[worst]:+.4f} at ε={worst:g}")
            if parts:
                lines.append(f"- {tag} ({metric}) empirical - simulation: " + "; ".join(parts))
    lines.append("")

    # Q5
    lines += ["## 5. Margins, u_inf, ε*, and the Q-Q check", ""]
    for tag, d in datas.items():
        pt = d["per_target"]
        if pt is None:
            continue
        dim = d["meta"]["d"]
        es = d["eps_star"]
        lines += [f"**{tag}** ({len(pt):,} correctly attributed in-set targets; "
                  f"K = {int(pt['K'].median()):,} impostor conversations each, median)", "",
                  f"- g_min: {pct(pt['g_min'])}; {(pt['g_min'] <= 0).sum()} targets with g_min = 0 (tie).",
                  f"- sqrt(d)·u_inf (exact for the {1000} closest impostors, triangle bound for the "
                  f"rest, as the memo allows): {pct(np.sqrt(dim) * pt['u_inf'])}",
                  f"- sqrt(d)·u_inf over the 1,000 exact impostors only: {pct(np.sqrt(dim) * pt['u_inf_exact'])}",
                  f"- ε* (Theorem 1, β = 0.05): {pct(es['eps_star'][np.isfinite(es['eps_star'])])}",
                  f"- ε* using the exact-only u_inf: "
                  f"{pct(es['eps_star_exact_u'][np.isfinite(es['eps_star_exact_u'])])}"]
        if d["qq"]:
            q = d["qq"]
            lines.append(f"- Q-Q at ε = {q['eps']:g}, 10,000 Laplace draws on 20 real directions: "
                         f"measured sd / (2√(2d)/ε) = {q['sd_ratio_mean']:.4f} "
                         f"(range {q['sd_ratio_min']:.4f} to {q['sd_ratio_max']:.4f}); "
                         f"see `figures/projection_qq.pdf`.")
        lines.append("")
    return "\n".join(lines)


def build(out: Path) -> None:
    res, figs = out / "results", out / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    datas = {tag: load(res, tag) for tag in tags(res)}
    tables = {}
    for tag, data in datas.items():
        table = long_table(tag, data)
        table.to_csv(res / f"dp_overlay_{tag}.csv", index=False)
        tables[tag] = table
        overlay_figure(tag, table, figs)
        overlay_figure(tag, table, figs, metrics=("A", "B"), suffix="_AB")
        if data["per_target"] is not None:
            margin_histogram(tag, data["per_target"], figs)
    qq_figure(datas, figs)
    dirfar_figure(tables, figs, ("C",), "dp_overlay_dirfar_conv")
    dirfar_figure(tables, figs, ("D",), "dp_overlay_dirfar_author")
    dirfar_figure(tables, figs, ("C", "D"), "dp_overlay_dirfar_both")
    # Model-simulation overlay, taken off the figures 2026-11-15 (kept in case it is needed later):
    # dirfar_figure(tables, figs, ("C",), "dp_overlay_dirfar_conv", curves=(SIM,))
    # dirfar_figure(tables, figs, ("D",), "dp_overlay_dirfar_author", curves=(SIM,))
    # dirfar_figure(tables, figs, ("C", "D"), "dp_overlay_dirfar_both", curves=(SIM,))
    dirfar_figure(tables, figs, ("B",), "dp_overlay_users_theorem", curves=(THEOREM, NORMAL))
    (res / "summary.md").write_text(summary(datas, tables))
    print(f"wrote {figs} and {res / 'summary.md'}")
