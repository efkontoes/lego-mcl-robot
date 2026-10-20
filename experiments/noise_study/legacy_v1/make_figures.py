"""Turn ``results/steps.csv`` and ``results/runs.csv`` into the comparison
figures and the summary table.

    python experiments/noise_study/make_figures.py

Design rules used throughout:
- Each variant has one fixed colour in every figure.
- Comparisons show a technique against the V1 baseline drawn in gray, rather than
  seven overlapping lines.
- Lines in low-contrast colours also get a direct end label, not just a legend entry.
- Colour pairs are checked for colour-vision deficiency. Orange (V2) and magenta (V5)
  are too close for normal vision, so they never appear in the same chart.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from legobot.localisation import VARIANTS  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results"
SUCCESS_CM = 10.0

# Reference categorical slots, fixed per variant.
COLORS = {"V1": "#2a78d6", "V2": "#eb6834", "V3": "#1baf7a", "V4": "#eda100",
          "V5": "#e87ba4", "V0": "#4a3aa7", "V0-bug": "#e34948"}
BASELINE = "#898781"  # V1 when it is the reference in an emphasis chart
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SHORT = {"V0-bug": "Coursework + bug", "V0": "Coursework", "V1": "Odometry model",
         "V2": "+ ESS gating", "V3": "+ injection", "V4": "+ KLD", "V5": "All combined"}
LW = 1.6

plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 120, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"], "font.size": 9,
    "text.color": INK, "axes.labelcolor": INK2, "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
    "axes.titlesize": 10, "axes.titleweight": "semibold", "axes.titlelocation": "left",
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.7, "grid.linestyle": "-",
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "legend.frameon": False, "legend.fontsize": 8, "lines.solid_capstyle": "round",
})


# ---------------------------------------------------------------------- data
def _read(path: Path) -> List[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def _runs(runs, scenario, variant):
    return [r for r in runs if r["scenario"] == scenario and r["variant"] == variant]


def _col(rows, key) -> np.ndarray:
    return np.array([float(r[key]) if r[key] not in ("", None) else np.nan for r in rows])


def _series(steps: List[dict], scenario: str, field: str, align: Optional[Dict] = None):
    """variant -> array [seed, step] padded with NaN. ``align`` maps (variant, seed)
    to a step index that becomes x = 0 (used to align runs on the kidnap)."""
    runs = defaultdict(lambda: defaultdict(list))
    for r in steps:
        if r["scenario"] == scenario:
            runs[r["variant"]][int(r["seed"])].append(float(r[field]))
    out = {}
    for variant, seeds in runs.items():
        rows = []
        for seed, values in seeds.items():
            start = align.get((variant, seed), 0) if align else 0
            rows.append(values[start:])
        arr = np.full((len(rows), max(len(v) for v in rows)), np.nan)
        for i, values in enumerate(rows):
            arr[i, : len(values)] = values
        out[variant] = arr
    return out


# ------------------------------------------------------------------- drawing
def _band(ax, data, color, label=None, stat="median", min_runs=5, lw=LW, end_label=None):
    counts = np.sum(~np.isnan(data), axis=0)
    keep = counts >= min_runs
    x = np.arange(data.shape[1])[keep]
    with np.errstate(all="ignore"):
        if stat == "median":
            mid = np.nanmedian(data, axis=0)[keep]
            lo, hi = (np.nanpercentile(data, q, axis=0)[keep] for q in (25, 75))
        else:
            mid = np.nanmean(data, axis=0)[keep]
            sem = (np.nanstd(data, axis=0) / np.sqrt(np.maximum(counts, 1)))[keep]
            lo, hi = mid - 1.96 * sem, mid + 1.96 * sem
    ax.fill_between(x, lo, hi, color=color, alpha=0.12, lw=0)
    ax.plot(x, mid, color=color, lw=lw, label=label)
    if end_label and len(x):
        ax.plot(x[-1], mid[-1], "o", ms=5, color=color, mec=SURFACE, mew=1.5)
        ax.annotate(end_label, (x[-1], mid[-1]), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=8, color=INK2)
    return x, mid


def _title(ax, title, subtitle=None):
    ax.set_title(title, pad=18 if subtitle else 6)
    if subtitle:
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=8, color=INK2, va="bottom")


def _save(fig, out: Path, name: str):
    fig.savefig(out / name, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print("wrote", out / name)


# ------------------------------------------------------------------- figures
def fig_spread(steps, out):
    """1. The coursework bug: noise compounds until the cloud covers 10 km."""
    spread = _series(steps, "tracking", "spread_cm")
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    for v, lab in (("V0-bug", "Coursework + compounding bug"), ("V0", "Coursework, bug fixed"),
                   ("V1", "Odometry model")):
        # The two fixed models converge, so only the bug gets an end label.
        _band(ax, spread[v][:, :60], COLORS[v], lab, end_label=lab if v == "V0-bug" else None)
    ax.set_yscale("log")
    ax.set_xlim(0, 75)
    ax.set_xlabel("sonar update")
    ax.set_ylabel("particle spread (cm, log scale)")
    _title(ax, "The compounding bug makes the particle cloud grow ×2 per move",
           "Tracking scenario, median and IQR over 30 runs · spread is logged up to 10⁶ cm, "
           "so the flat top is that cap")
    ax.legend(loc="upper left")
    _save(fig, out, "spread_blowup.png")


def fig_scorecard(runs, out):
    """2. Every variant × every metric, shaded by how good it is in that column."""
    variants = [v for v in VARIANTS if v != "V0-bug"]
    metrics = [  # (label, scenario, key, aggregate, higher_is_better, fmt)
        ("Tracking\nerror (cm)", "tracking", "mean_error_cm", np.mean, False, "{:.2f}"),
        ("Tracking\ndistinct\nparticles", "tracking", "mean_distinct_ratio", np.mean, True, "{:.0%}"),
        ("Tracking\nms / update", "tracking", "mean_update_ms", np.nanmean, False, "{:.2f}"),
        ("Tracking\nparticles", "tracking", "mean_particles", np.mean, False, "{:.0f}"),
        ("Kidnap\nrecovered", "kidnap", "success", None, True, "{:.0%}"),
        ("Kidnap\nwall\ncontacts", "kidnap", "wall_contacts", np.mean, False, "{:.1f}"),
        ("Global\nconverged", "global", "success", None, True, "{:.0%}"),
        ("Global\nparticles", "global", "mean_particles", np.mean, False, "{:.0f}"),
    ]
    values = np.zeros((len(variants), len(metrics)))
    for j, (_, sc, key, agg, _, _) in enumerate(metrics):
        for i, v in enumerate(variants):
            rows = _runs(runs, sc, v)
            values[i, j] = (np.mean([r[key] == "True" for r in rows]) if agg is None
                            else agg(_col(rows, key)))
    # Goodness per column in [0, 1] (1 = best); identical columns stay neutral.
    good = np.zeros_like(values)
    for j, m in enumerate(metrics):
        col = values[:, j]
        span = col.max() - col.min()
        g = (col - col.min()) / span if span > 1e-12 else np.full(len(col), 0.5)
        good[:, j] = g if m[4] else 1 - g
    cmap = LinearSegmentedColormap.from_list("seq", ["#f0f6fe", "#b7d3f6", "#5598e7", "#1c5cab"])
    fig, ax = plt.subplots(figsize=(10, 3.9))
    ax.imshow(good, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    for i in range(len(variants)):
        for j, m in enumerate(metrics):
            colour = SURFACE if good[i, j] > 0.62 else INK
            ax.text(j, i, m[5].format(values[i, j]), ha="center", va="center", fontsize=8.5,
                    color=colour, fontweight="semibold" if good[i, j] > 0.999 else "normal")
    # 2px-style surface gaps between cells
    for k in np.arange(-0.5, len(metrics), 1):
        ax.axvline(k, color=SURFACE, lw=2.5)
    for k in np.arange(-0.5, len(variants), 1):
        ax.axhline(k, color=SURFACE, lw=2.5)
    ax.set_xticks(range(len(metrics)), [m[0] for m in metrics], fontsize=8)
    ax.set_yticks(range(len(variants)), [f"{v}  {SHORT[v]}" for v in variants], fontsize=8.5)
    ax.tick_params(length=0)
    ax.xaxis.tick_top()
    ax.set_title("Scorecard: darker = better within each column (values are the measurements)",
                 pad=44)
    _save(fig, out, "scorecard.png")


def fig_tradeoff(runs, out):
    """3. What each technique buys and what it costs."""
    variants = [v for v in VARIANTS if v != "V0-bug"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    panels = (
        (axes[0], "tracking", "mean_error_cm", "mean tracking error (cm) ↓",
         "Tracking: accuracy vs cost"),
        (axes[1], "kidnap", "success", "kidnap recovery rate ↑", "Kidnap: recovery vs cost"),
    )
    offsets = {  # (dx, dy, ha) in points, chosen so the labels never collide
        "tracking": {"V0": (8, 4, "left"), "V1": (-8, 7, "right"), "V2": (8, -9, "left"),
                     "V3": (8, -10, "left"), "V4": (-8, 9, "right"), "V5": (8, 5, "left")},
        "kidnap": {"V0": (8, 5, "left"), "V1": (6, -12, "left"), "V2": (-8, 9, "right"),
                   "V3": (8, 5, "left"), "V4": (8, 7, "left"), "V5": (-8, 7, "right")},
    }
    notes = {"tracking": "lower-left is better · narrow y-range: all variants within 0.25 cm",
             "kidnap": "upper-left is better · 30 runs per variant"}
    for ax, sc, key, ylabel, title in panels:
        for v in variants:
            rows = _runs(runs, sc, v)
            x = np.nanmean(_col(rows, "mean_update_ms"))
            y = np.mean([r[key] == "True" for r in rows]) if key == "success" else np.mean(_col(rows, key))
            ax.plot(x, y, "o", ms=8, color=COLORS[v], mec=SURFACE, mew=2, zorder=3)
            dx, dy, ha = offsets[sc][v]
            ax.annotate(f"{v} {SHORT[v]}", (x, y), xytext=(dx, dy), textcoords="offset points",
                        fontsize=8, color=INK2, ha=ha, va="center")
        ax.set_xlabel("ms per sonar update ↓")
        ax.set_ylabel(ylabel)
        ax.margins(x=0.3, y=0.25)
        _title(ax, title, notes[sc])
    axes[1].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    fig.tight_layout()
    _save(fig, out, "tradeoff.png")


def fig_resampling(steps, runs, out):
    """4. ESS gating vs resampling every update."""
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), gridspec_kw={"width_ratios": [1.3, 1.3, 0.8]})
    for ax, field, label in ((axes[0], "ess_ratio", "N_eff / N before resampling"),
                             (axes[1], "distinct_ratio", "distinct particles / N")):
        data = _series(steps, "tracking", field)
        _band(ax, data["V1"], BASELINE, "V1 resample every update", stat="mean")
        _band(ax, data["V2"], COLORS["V2"], "V2 resample only if N_eff < N/2", stat="mean")
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("sonar update")
        ax.set_ylabel(label)
    _title(axes[0], "Weight degeneracy")
    _title(axes[1], "Particle diversity (depletion)")
    axes[1].legend(loc="lower right")
    ax = axes[2]
    vals = [np.mean(_col(_runs(runs, "tracking", v), "resample_fraction")) for v in ("V1", "V2")]
    bars = ax.bar(["V1", "V2"], vals, color=[BASELINE, COLORS["V2"]], width=0.5)
    for b, val in zip(bars, vals):
        ax.annotate(f"{val:.0%}", (b.get_x() + b.get_width() / 2, val), xytext=(0, 3),
                    textcoords="offset points", ha="center", fontsize=8.5, color=INK)
    ax.set_ylim(0, 1.12)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.grid(axis="x", visible=False)
    _title(ax, "Updates that resampled", "out-of-range readings (~17%) skip both")
    fig.tight_layout()
    _save(fig, out, "resampling.png")


def fig_kld(steps, out):
    """5. KLD-sampling: fewer particles, same accuracy (and same failures)."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 5.6), sharex="col")
    for col, sc in enumerate(("tracking", "global")):
        count = _series(steps, sc, "particles")
        err = _series(steps, sc, "pos_error_cm")
        for v, colour, lab in (("V1", BASELINE, "V1 fixed size"), ("V4", COLORS["V4"], "V4 + KLD"),
                               ("V5", COLORS["V5"], "V5 all combined")):
            _band(axes[0, col], count[v], colour, lab)  # lines converge: legend, no end labels
            _band(axes[1, col], err[v], colour, lab)
        axes[0, col].set_ylabel("particles in the filter")
        axes[1, col].set_ylabel("position error (cm)")
        axes[1, col].set_xlabel("sonar update")
        _title(axes[0, col], f"{sc.capitalize()}: particle count")
        _title(axes[1, col], f"{sc.capitalize()}: position error")
    axes[0, 0].legend(loc="upper left")
    axes[0, 1].legend(loc="center right")
    fig.tight_layout()
    _save(fig, out, "kld_particles.png")


def _kidnap_alignment(runs):
    return {(r["variant"], int(r["seed"])): int(r["kidnap_step"])
            for r in runs if r["scenario"] == "kidnap" and r.get("kidnap_step")}


def fig_kidnap(steps, runs, out):
    """6. Recovery after the robot is picked up: each technique vs the baseline."""
    align = _kidnap_alignment(runs)
    err = _series(steps, "kidnap", "pos_error_cm", align)
    order = ["V0", "V2", "V3", "V4", "V5"]
    fig, axes = plt.subplots(1, len(order), figsize=(12.5, 3.2), sharey=True)
    for ax, v in zip(axes, order):
        _band(ax, err["V1"][:, :70], BASELINE, "V1 baseline")
        _band(ax, err[v][:, :70], COLORS[v], f"{v}")
        ax.axhline(SUCCESS_CM, color=AXIS, lw=0.9)
        rate = np.mean([r["success"] == "True" for r in _runs(runs, "kidnap", v)])
        _title(ax, f"{v} {SHORT[v]}", f"recovered in {rate:.0%} of runs")
        ax.set_xlabel("updates since kidnap")
        ax.set_ylim(0, 160)
    axes[0].set_ylabel("position error (cm)")
    axes[0].annotate("10 cm success line", (0, SUCCESS_CM), xytext=(2, 3),
                     textcoords="offset points", fontsize=7.5, color=MUTED)
    axes[0].legend(loc="upper right", handles=[
        plt.Line2D([], [], color=BASELINE, lw=LW, label="V1 baseline"),
        plt.Line2D([], [], color=INK2, lw=LW, label="technique (panel title)")])
    fig.suptitle("Kidnap: median error after the robot is moved 30–50 cm (IQR shaded)",
                 x=0.01, ha="left", fontsize=10, fontweight="semibold", y=1.06)
    fig.tight_layout()
    _save(fig, out, "kidnap_recovery.png")


def fig_injection(steps, runs, out):
    """7. Why injection needs KLD: particles injected vs particles available."""
    align = _kidnap_alignment(runs)
    inj = _series(steps, "kidnap", "injected", align)
    count = _series(steps, "kidnap", "particles", align)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    for v, lab in (("V3", "V3 injection, fixed 100 particles"), ("V5", "V5 injection + KLD")):
        _band(axes[0], inj[v][:, :70], COLORS[v], lab, stat="mean", end_label=v)
        _band(axes[1], count[v][:, :70], COLORS[v], lab, end_label=v)
    axes[0].set_ylabel("random particles injected")
    axes[1].set_ylabel("particles in the filter")
    for ax in axes:
        ax.set_xlabel("updates since kidnap")
    _title(axes[0], "Injected per update (mean, 95% CI)")
    _title(axes[1], "Filter size (median, IQR)")
    axes[1].legend(loc="center right")
    axes[1].set_ylim(0, 1100)
    fig.tight_layout()
    _save(fig, out, "injection.png")


def fig_final_errors(runs, out):
    """8. Distribution of final errors per scenario (one dot per run)."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    rng = np.random.default_rng(0)
    for ax, sc in zip(axes, ("tracking", "kidnap", "global")):
        variants = [v for v in VARIANTS if _runs(runs, sc, v) and v != "V0-bug"]
        for i, v in enumerate(variants):
            e = np.clip(_col(_runs(runs, sc, v), "final_error_cm"), 0.3, None)
            ax.scatter(i + rng.uniform(-0.18, 0.18, len(e)), e, s=14, color=COLORS[v], alpha=0.75,
                       edgecolors=SURFACE, linewidths=0.6, zorder=3)
            ax.plot([i - 0.28, i + 0.28], [np.median(e)] * 2, color=INK, lw=1.4, zorder=4)
        ax.axhline(SUCCESS_CM, color=AXIS, lw=0.9)
        ax.set_yscale("log")
        ax.set_xticks(range(len(variants)), variants)
        ax.grid(axis="x", visible=False)
        _title(ax, sc.capitalize())
    axes[0].set_ylabel("final position error (cm, log)")
    axes[0].annotate("10 cm success line", (-0.4, SUCCESS_CM), xytext=(0, 3),
                     textcoords="offset points", fontsize=7.5, color=MUTED)
    fig.suptitle("Final error of every run (dots) and median (bar); V0-bug omitted: it diverges",
                 x=0.01, ha="left", fontsize=10, fontweight="semibold", y=1.04)
    fig.tight_layout()
    _save(fig, out, "final_errors.png")


def figures(steps, runs, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.png"):
        old.unlink()
    fig_spread(steps, out)
    fig_scorecard(runs, out)
    fig_tradeoff(runs, out)
    fig_resampling(steps, runs, out)
    fig_kld(steps, out)
    fig_kidnap(steps, runs, out)
    fig_injection(steps, runs, out)
    fig_final_errors(runs, out)


# --------------------------------------------------------------------- table
def table(runs: List[dict]) -> str:
    lines = ["| Scenario | Variant | Success | Median final error (cm) | Mean error (cm) | "
             "Median max spread (cm) | N_eff/N | Distinct | Resampled | Mean particles | "
             "Injected / run | ms / update |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for scenario in ("tracking", "kidnap", "global"):
        for v in VARIANTS:
            rows = _runs(runs, scenario, v)
            if not rows:
                continue
            final, mean_err = _col(rows, "final_error_cm"), _col(rows, "mean_error_cm")
            final_txt = "diverged" if np.median(final) >= 1e5 else f"{np.median(final):.1f}"
            mean_txt = "diverged" if np.median(mean_err) >= 1e5 else f"{np.mean(mean_err):.1f}"
            spread = np.median(_col(rows, "max_spread_cm"))
            spread_txt = f"{spread:.2e}" if spread >= 1e4 else f"{spread:.1f}"
            lines.append(
                f"| {scenario} | {v} | {np.mean([r['success'] == 'True' for r in rows]):.0%} | "
                f"{final_txt} | {mean_txt} | {spread_txt} | "
                f"{np.nanmean(_col(rows, 'mean_ess_ratio')):.2f} | "
                f"{np.mean(_col(rows, 'mean_distinct_ratio')):.2f} | "
                f"{np.mean(_col(rows, 'resample_fraction')):.0%} | "
                f"{np.mean(_col(rows, 'mean_particles')):.0f} | "
                f"{np.mean(_col(rows, 'injected_total')):.0f} | "
                f"{np.nanmean(_col(rows, 'mean_update_ms')):.2f} |"
            )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--results", type=Path, default=RESULTS)
    args = parser.parse_args()
    steps = _read(args.results / "steps.csv")
    runs = _read(args.results / "runs.csv")
    figures(steps, runs, args.results / "figures")
    md = table(runs)
    (args.results / "summary.md").write_text(md + "\n")
    print(md)


if __name__ == "__main__":
    main()
