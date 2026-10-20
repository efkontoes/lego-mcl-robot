"""Figures for the v2 noise study, generated only from saved results
(results/analysis/*.csv, results/raw snapshots, traces and likelihood maps).

    python experiments/noise_study/make_figures.py
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import study_lib as S  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from legobot.localisation import VARIANTS, arena  # noqa: E402

A = S.RESULTS / "analysis"
FIG = S.RESULTS / "figures"

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
COLORS = {"V1": "#898781", "V2": "#eb6834", "V3": "#1baf7a", "V4": "#eda100", "V5": "#e87ba4",
          "V6": "#008300", "V7": "#4a3aa7", "V8": "#e34948", "V3-N500": "#86b6ef",
          "V3-N1000": "#2a78d6", "V1-N1000": "#52514e", "V0": "#b7b6ae"}
SEQ = LinearSegmentedColormap.from_list("seq", ["#f0f6fe", "#b7d3f6", "#5598e7", "#1c5cab"])
STAGE_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"]
ORDER = ["V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V3-N500", "V3-N1000", "V1-N1000"]
LW = 1.6

plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 120, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"], "font.size": 9, "text.color": INK,
    "axes.labelcolor": INK2, "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.titlesize": 10,
    "axes.titleweight": "semibold", "axes.titlelocation": "left", "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "legend.frameon": False, "legend.fontsize": 8,
})


def _title(ax, title, sub=None):
    ax.set_title(title, pad=18 if sub else 6)
    if sub:
        ax.text(0, 1.02, sub, transform=ax.transAxes, fontsize=8, color=INK2, va="bottom")


def _suptitle(fig, text, sub=None, y=1.02):
    fig.suptitle(text, x=0.01, ha="left", fontsize=11, fontweight="semibold", y=y)
    if sub:
        fig.text(0.01, y - 0.045, sub, ha="left", fontsize=8.5, color=INK2)


def _save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / name, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print("wrote", FIG / name)


def _pct(ax):
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))


def _short(v):
    return f"{v} {VARIANTS[v].label}"


# =========================================================== 1. scorecard
def fig_scorecard(summ, cl):
    k = summ[summ.scenario == "kidnap"].set_index("variant")
    t = summ[summ.scenario == "tracking"].set_index("variant")
    g = summ[summ.scenario == "global"].set_index("variant")
    clk = cl[cl.scenario == "kidnap"].set_index("variant")
    cols = [  # label, getter, higher is better, format
        ("Tracking\nmean error (cm)\n[A]", lambda v: t.loc[v, "mean_err"], False, "{:.2f}"),
        ("Kidnap sustained\nrecovery\n[A]", lambda v: k.loc[v, "recovered"], True, "{:.0%}"),
        ("Kidnap post-\nkidnap p95 err\n(cm) [A]", lambda v: k.loc[v, "post_p95_err"], False, "{:.0f}"),
        ("Global\nconverged\n[A]", lambda v: g.loc[v, "recovered"] if v in g.index else np.nan, True, "{:.0%}"),
        ("Kidnap mean\nparticles\n[A]", lambda v: k.loc[v, "mean_particles"], False, "{:.0f}"),
        ("Kidnap CPU per\ntrajectory (ms)\n[A]", lambda v: k.loc[v, "cpu_ms"], False, "{:.0f}"),
        ("Closed loop:\nall waypoints\nverified [B]",
         lambda v: clk.loc[v, "all_verified"] if v in clk.index else np.nan, True, "{:.0%}"),
        ("Closed loop:\nwall contacts\nper run [B]",
         lambda v: clk.loc[v, "wall_contacts"] if v in clk.index else np.nan, False, "{:.1f}"),
    ]
    vals = np.array([[c[1](v) for c in cols] for v in ORDER], dtype=float)
    good = np.full_like(vals, np.nan)
    for j, c in enumerate(cols):
        col = vals[:, j]
        ok = ~np.isnan(col)
        lo, hi = np.nanmin(col), np.nanmax(col)
        gg = (col - lo) / (hi - lo) if hi > lo else np.full(len(col), 0.5)
        good[ok, j] = (gg if c[2] else 1 - gg)[ok]
    fig, ax = plt.subplots(figsize=(11.5, 5.6))
    ax.imshow(np.where(np.isnan(good), 0, good), cmap=SEQ, vmin=0, vmax=1, aspect="auto")
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    for i in range(len(ORDER)):
        for j, c in enumerate(cols):
            if np.isnan(vals[i, j]):
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, color="#f0efec"))
                ax.text(j, i, "not run", ha="center", va="center", fontsize=7.5, color=MUTED)
                continue
            ax.text(j, i, c[3].format(vals[i, j]), ha="center", va="center", fontsize=8.5,
                    color=SURFACE if good[i, j] > 0.62 else INK)
    for x in np.arange(-0.5, len(cols)):
        ax.axvline(x, color=SURFACE, lw=2.5)
    for y in np.arange(-0.5, len(ORDER)):
        ax.axhline(y, color=SURFACE, lw=2.5)
    ax.axhline(7.5, color=INK2, lw=1.2)
    ax.set_xticks(range(len(cols)), [c[0] for c in cols], fontsize=7.8)
    ax.set_yticks(range(len(ORDER)), [_short(v) for v in ORDER], fontsize=8.3)
    ax.tick_params(length=0)
    ax.xaxis.tick_top()
    ax.set_title("Scorecard: darker = better within each column. [A] fixed-trajectory replay, "
                 "[B] closed-loop navigation; 100 held-out seeds", pad=58)
    ax.text(0, -0.02, "Rows below the dark line are fixed particle-budget controls. Particle count and CPU: "
            "fewer / lower shaded as better (cheaper). 'not run' = variant not part of that protocol.",
            transform=ax.transAxes, ha="left", va="top", fontsize=7.8, color=INK2)
    _save(fig, "01_scorecard.png")


# =========================================================== 2. factorial interaction
def fig_factorial(summ, effects):
    k = summ[summ.scenario == "kidnap"].set_index("variant")
    cell = {(e, i, kk): v for v in S.MAIN_VARIANTS["kidnap"] if VARIANTS[v].role == "factorial"
            for e, i, kk in [tuple(vars(VARIANTS[v].options).values())]}
    fig = plt.figure(figsize=(12, 3.9))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.25], wspace=0.35)
    for col, ess in enumerate((False, True)):
        ax = fig.add_subplot(gs[0, col])
        for kld, color, lab in ((False, MUTED, "KLD off"), (True, COLORS["V8"], "KLD on")):
            ys, los, his = [], [], []
            for inj in (False, True):
                v = cell[(ess, inj, kld)]
                ys.append(k.loc[v, "recovered"])
                los.append(k.loc[v, "recovered"] - k.loc[v, "recovered_lo"])
                his.append(k.loc[v, "recovered_hi"] - k.loc[v, "recovered"])
            x = np.array([0, 1]) + (0.04 if kld else -0.04)
            ax.errorbar(x, ys, yerr=[los, his], color=color, lw=LW, marker="o", ms=6, capsize=3,
                        label=lab, mec=SURFACE, mew=1.5)
            for xx, yy, inj in zip(x, ys, (False, True)):
                # KLD-off labels to the left of injection-off points, everything else to the right
                left = not inj and not kld
                ax.annotate(cell[(ess, inj, kld)], (xx, yy), xytext=(-9 if left else 9, 0),
                            textcoords="offset points", ha="right" if left else "left", va="center",
                            fontsize=8, color=INK2)
        ax.set_xticks([0, 1], ["injection off", "injection on"])
        ax.set_xlim(-0.45, 1.45)
        ax.set_ylim(0, max(0.6, float(k["recovered_hi"].max()) + 0.05))
        _pct(ax)
        ax.set_ylabel("sustained kidnap recovery")
        _title(ax, f"ESS gating {'on' if ess else 'off'}", "Wilson 95% CI, n = 100 per point")
        ax.legend(loc="upper left")
    ax = fig.add_subplot(gs[0, 2])
    e = effects[(effects.scenario == "kidnap") & (effects.metric == "recovered")].reset_index(drop=True)
    y = np.arange(len(e))[::-1]
    for yy, (_, r) in zip(y, e.iterrows()):
        c = INK if r["supported"] else MUTED
        ax.plot([r["ci_lo"], r["ci_hi"]], [yy, yy], color=c, lw=LW)
        ax.plot(r["estimate"], yy, "o", color=c, ms=6, mec=SURFACE, mew=1.5)
        ax.annotate(f"{r['estimate']:+.1%}", (r["ci_hi"], yy), xytext=(5, 0), textcoords="offset points",
                    va="center", fontsize=8, color=INK2)
    ax.axvline(0, color=AXIS, lw=1)
    names = {"E": "ESS (E)", "I": "Injection (I)", "K": "KLD (K)"}
    ax.set_yticks(y, [names.get(n, n) for n in e["effect"]])
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlabel("effect on sustained recovery (percentage points)")
    _title(ax, "Factorial effects", "±1 contrasts over 8 cells, paired bootstrap 95% CI; black = CI excludes 0")
    _suptitle(fig, "Kidnap recovery across the 2×2×2 design (Protocol A)", y=1.08)
    _save(fig, "02_factorial_interaction.png")


# =========================================================== 3. recovery vs computation
LABEL_OFFSETS = {  # (dx, dy, ha) in points: keeps labels of clustered points apart
    "cpu_ms": {"V1": (-8, -9, "right"), "V2": (-8, 4, "right"), "V7": (8, -7, "left"),
               "V3": (8, -6, "left"), "V6": (-8, 6, "right")},
    "post_mean_particles": {"V1": (-8, -6, "right"), "V2": (-8, 2, "right"), "V3": (8, -5, "left"),
                            "V6": (8, 7, "left"), "V7": (8, -3, "left")},
}


def fig_recovery_cost(summ):
    k = summ[summ.scenario == "kidnap"].set_index("variant")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, xcol, xlabel in ((axes[0], "cpu_ms", "filter CPU per kidnap trajectory (ms, laptop)"),
                             (axes[1], "post_mean_particles", "mean particles after the kidnap")):
        offsets = LABEL_OFFSETS[xcol]
        for v in ORDER:
            r = k.loc[v]
            hollow = VARIANTS[v].role == "budget-control"
            ax.errorbar(r[xcol], r["recovered"],
                        xerr=[[r[xcol] - r[f"{xcol}_lo"]], [r[f"{xcol}_hi"] - r[xcol]]],
                        yerr=[[r["recovered"] - r["recovered_lo"]], [r["recovered_hi"] - r["recovered"]]],
                        fmt="o", ms=8, color=COLORS[v], mfc=SURFACE if hollow else COLORS[v],
                        mec=COLORS[v], mew=2, elinewidth=1, capsize=0, alpha=0.95, zorder=3)
            dx, dy, ha = offsets.get(v, (7, 5, "left"))
            ax.annotate(v, (r[xcol], r["recovered"]), xytext=(dx, dy), textcoords="offset points",
                        fontsize=8, color=INK2, ha=ha, va="center")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("sustained kidnap recovery")
        _pct(ax)
    _title(axes[0], "Recovery vs computation", "filled = factorial variants, hollow = fixed-budget controls")
    _title(axes[1], "Recovery vs particle budget", "95% CIs: Wilson (y), bootstrap (x)")
    fig.tight_layout()
    _save(fig, "03_recovery_vs_computation.png")


# =========================================================== 4. recovery over time
MIN_AT_RISK_HORIZON = 20  # draw KM curves only while >= this many traces are still observed


def fig_recovery_time(km, runs):
    show = ["V1", "V3", "V6", "V8", "V5", "V3-N1000"]
    horizons = runs[(runs.scenario == "kidnap") & (runs.variant == "V1")]["horizon"].to_numpy()
    # Largest t at which at least MIN_AT_RISK_HORIZON traces are still observed (horizons are
    # trace properties, identical for every variant).
    t_cut = float(np.sort(horizons)[::-1][MIN_AT_RISK_HORIZON - 1])
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    ends = {}
    for v in show:
        d = km[km.variant == v].sort_values("t")
        t = np.append(d["t"].to_numpy(), t_cut)
        s = np.append(d["not_recovered"].to_numpy(), np.nan)
        keep = t <= t_cut
        t, s = t[keep], s[keep]
        s[-1] = s[-2] if np.isnan(s[-1]) else s[-1]
        ax.step(t, s, where="post", color=COLORS[v], lw=LW, label=_short(v))
        ends[v] = s[-1]
    order = sorted(show, key=lambda v: ends[v])
    last_y = -1.0
    for v in order:  # end labels at the curve's level, pushed up only to avoid overlap
        y = max(ends[v], last_y + 0.065)
        last_y = y
        ax.plot([t_cut, t_cut + 3], [ends[v], y], color=COLORS[v], lw=0.8)
        ax.annotate(f"{v}: {ends[v]:.0%}", (t_cut + 3, y), xytext=(3, 0),
                    textcoords="offset points", va="center", fontsize=7.5, color=INK2)
    ax.set_xlim(0, t_cut + 18)
    ax.set_ylim(0, 1.03)
    _pct(ax)
    ax.set_xlabel("sonar updates since the kidnap")
    ax.set_ylabel("runs not yet recovered (sustained)")
    ax.legend(loc="lower left", fontsize=7.5)
    _title(ax, "Time to sustained recovery (Kaplan–Meier, Protocol A)",
           f"recovery = error < 10 cm for 5 consecutive updates; traces end after 92–165 updates "
           f"(censored); curves stop at update {t_cut:.0f}, the last point with ≥{MIN_AT_RISK_HORIZON} traces observed")
    _save(fig, "04_recovery_over_time.png")


# =========================================================== 5. case studies
def _load_snaps(scenario, seed, variant):
    path = S.RESULTS / "raw" / "main" / "snapshots" / f"{scenario}_{seed}_{variant}.npz"
    if not path.exists():
        return None
    z = np.load(path)
    return [z[f"arr_{i}"].astype(float) for i in range(len(z.files))]


def _draw_world(ax, world):
    for x0, y0, x1, y1 in world.walls:
        ax.plot([x0, x1], [y0, y1], color="#222", lw=2)
    ax.set_xlim(-8, 218)
    ax.set_ylim(-8, 218)
    ax.set_aspect("equal")
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])


def fig_cases(runs, steps_main):
    world = arena()
    kid = runs[(runs.scenario == "kidnap") & runs.seed.isin(S.SNAPSHOT_SEEDS)]
    picks = []
    for variant in ("V8", "V5"):
        ok = kid[(kid.variant == variant) & (kid.recovered == 1)]
        bad = kid[(kid.variant == variant) & (kid.recovered == 0)]
        if len(ok) and len(bad):
            picks = [(variant, int(ok.seed.iloc[0]), "sustained recovery"),
                     (variant, int(bad.seed.iloc[0]), "no recovery")]
            break
    if not picks:
        return
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.8))
    for row, (variant, seed, outcome) in enumerate(picks):
        g = steps_main[(steps_main.scenario == "kidnap") & (steps_main.variant == variant)
                       & (steps_main.seed == seed)].sort_values("update")
        snaps = _load_snaps("kidnap", seed, variant)
        k0 = int(g.loc[g.since_kidnap >= 0, "update"].iloc[0])
        last = int(g["update"].iloc[-1])
        moments = [("just after the kidnap", k0), ("after the first sweep", min(k0 + 12, last)),
                   ("end of trace", last)]
        ax = axes[row, 0]
        _draw_world(ax, world)
        ax.plot(g.true_x, g.true_y, color="#1565c0", lw=1.4, label="true path")
        post = g[g.since_kidnap >= 0]
        ax.plot(post.est_x, post.est_y, color=COLORS[variant], lw=1.1, ls="-", alpha=0.9, label="estimate after kidnap")
        ax.annotate("", xy=(g.true_x.iloc[k0], g.true_y.iloc[k0]),
                    xytext=(g.true_x.iloc[k0 - 1], g.true_y.iloc[k0 - 1]),
                    arrowprops=dict(arrowstyle="->", color=INK, lw=1.4))
        ax.text(g.true_x.iloc[k0 - 1], g.true_y.iloc[k0 - 1] + 6, "kidnap", fontsize=7.5, color=INK,
                ha="center")
        ax.legend(loc="upper left", fontsize=7)
        _title(ax, f"{variant}, seed {seed}: {outcome}")
        for col, (name, u) in enumerate(moments, start=1):
            ax = axes[row, col]
            _draw_world(ax, world)
            p = snaps[u]
            ax.scatter(p[:, 0], p[:, 1], s=14, color=COLORS[variant], alpha=0.45, lw=0, zorder=2)
            r = g[g["update"] == u].iloc[0]
            ax.plot(r.true_x, r.true_y, "o", ms=8, color="#1565c0", mec=SURFACE, mew=1.5, zorder=4)
            ax.plot(r.est_x, r.est_y, "x", ms=8, color=INK, mew=2, zorder=5)
            _title(ax, f"{name} (update {u})",
                   f"error {r.pos_err:.0f} cm, r95 {r.r95:.0f} cm, {int(r.n_after)} particles")
    _suptitle(fig, "What the particle cloud does after a kidnap: a recovered and a failed run",
              "blue dot = true pose, black × = estimate, dots = particles (≤300 shown); Protocol A replay",
              y=1.04)
    fig.tight_layout(h_pad=3.5)
    _save(fig, "05_case_trajectories.png")


# =========================================================== 6. particle & injection dynamics
def fig_dynamics(dyn):
    show = ["V3", "V8", "V5", "V3-N1000", "V4"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.7))
    for v in show:
        d = dyn[(dyn.variant == v) & (dyn.runs >= 20)].sort_values("since_kidnap")
        axes[0].plot(d.since_kidnap, d.n_after, color=COLORS[v], lw=LW, label=_short(v))
        axes[0].fill_between(d.since_kidnap, d.n_after_q25, d.n_after_q75, color=COLORS[v], alpha=0.07, lw=0)
        axes[1].plot(d.since_kidnap, d.p_inject, color=COLORS[v], lw=LW)
        axes[2].plot(d.since_kidnap, d.injected, color=COLORS[v], lw=LW)
    for ax in axes:
        ax.axvline(0, color=AXIS, lw=1)
        ax.set_xlabel("sonar updates since the kidnap")
    axes[0].set_ylabel("particles (mean, IQR)")
    axes[1].set_ylabel("mean injection probability")
    axes[2].set_ylabel("mean particles injected per update")
    _title(axes[0], "Filter size")
    _title(axes[1], "Injection probability")
    _title(axes[2], "Random particles injected")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=5, fontsize=8,
               bbox_to_anchor=(0.5, -0.08))
    _suptitle(fig, "Particle count and injection around the kidnap (Protocol A, 100 runs per variant)",
              y=1.07)
    fig.tight_layout()
    _save(fig, "06_particle_dynamics.png")


# =========================================================== 7. CPU breakdown
def fig_cpu(cpu, cvp):
    k = cpu[cpu.scenario == "kidnap"].set_index("variant").loc[ORDER]
    stages = [("predict_mean_ms", "motion prediction"), ("t_likelihood_mean_ms", "ray casting + likelihood"),
              ("t_weighting_mean_ms", "weighting / ESS"), ("t_resample_draw_mean_ms", "resampling draw"),
              ("t_kld_bins_mean_ms", "KLD bin accounting"), ("t_injection_mean_ms", "injection"),
              ("t_bookkeeping_mean_ms", "bookkeeping")]
    fig = plt.figure(figsize=(13, 4.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.3, 1], wspace=0.3)
    ax = fig.add_subplot(gs[0, 0])
    left = np.zeros(len(k))
    y = np.arange(len(k))[::-1]
    for (col, name), color in zip(stages, STAGE_COLORS):
        vals = k[col].fillna(0).to_numpy()
        ax.barh(y, vals, left=left, color=color, height=0.62, label=name, edgecolor=SURFACE, lw=1)
        left += vals
    for yy, (v, r) in zip(y, k.iterrows()):
        ax.annotate(f"median {r.cpu_median_ms:.2f} · p95 {r.cpu_p95_ms:.2f} ms · {r.particles_mean:.0f} particles",
                    (left[list(k.index).index(v)], yy), xytext=(4, 0), textcoords="offset points",
                    va="center", fontsize=7.3, color=INK2)
    ax.set_yticks(y, [_short(v) for v in k.index], fontsize=8)
    ax.set_xlabel("mean CPU per valid sonar update (ms), split by stage")
    ax.set_xlim(0, left.max() * 1.75)
    ax.legend(fontsize=7.2, loc="lower right", ncol=1)
    ax.grid(axis="y", visible=False)
    _title(ax, "Where the time goes (kidnap scenario)")
    ax = fig.add_subplot(gs[0, 1])
    d = cvp[cvp.scenario == "kidnap"]
    for v in ("V1", "V4", "V8", "V3-N1000", "V1-N1000"):
        dv = d[d.variant == v]
        bins = np.unique(np.quantile(dv.n_before, np.linspace(0, 1, 9)))
        if len(bins) < 2:
            ax.plot(dv.n_before.median(), dv.cpu_ms.median(), "o", color=COLORS[v], ms=7, label=_short(v))
            continue
        idx = np.clip(np.digitize(dv.n_before, bins) - 1, 0, len(bins) - 2)
        med = dv.groupby(idx).agg(n=("n_before", "median"), c=("cpu_ms", "median"))
        ax.plot(med.n, med.c, "-o", color=COLORS[v], ms=4, lw=LW, label=_short(v))
    ax.set_xlabel("particles entering the update")
    ax.set_ylabel("median CPU per update (ms)")
    ax.legend(fontsize=7.2)
    _title(ax, "Cost per update vs particle count", "binned medians, valid kidnap updates")
    _suptitle(fig, "CPU breakdown (laptop timings, time.perf_counter; not Raspberry Pi)", y=1.05)
    _save(fig, "07_cpu_breakdown.png")


# =========================================================== 8. noise robustness
def fig_noise(grid):
    variants = S.NOISE_VARIANTS
    fig, axes = plt.subplots(2, len(variants), figsize=(13, 6.2))
    for row, (scenario, metric, label, fmt, higher) in enumerate((
            ("kidnap", "recovered", "sustained kidnap recovery", "{:.0%}", True),
            ("tracking", "mean_err", "tracking mean error (cm)", "{:.1f}", False))):
        sub = grid[grid.scenario == scenario]
        vmin, vmax = sub[metric].min(), sub[metric].max()
        for col, v in enumerate(variants):
            ax = axes[row, col]
            m = (sub[sub.variant == v].pivot(index="motion_scale", columns="outlier_prob", values=metric)
                 .sort_index(ascending=False))
            norm = (m.to_numpy() - vmin) / (vmax - vmin) if vmax > vmin else np.zeros(m.shape)
            ax.imshow(norm if higher else 1 - norm, cmap=SEQ, vmin=0, vmax=1, aspect="auto")
            ax.grid(False)
            for i in range(m.shape[0]):
                for j in range(m.shape[1]):
                    good = norm[i, j] if higher else 1 - norm[i, j]
                    ax.text(j, i, fmt.format(m.iloc[i, j]), ha="center", va="center", fontsize=8.5,
                            color=SURFACE if good > 0.62 else INK)
            ax.set_xticks(range(m.shape[1]), [f"{p:.0%}" for p in m.columns])
            ax.set_yticks(range(m.shape[0]), [f"{s:g}×" for s in m.index])
            if col == 0:
                ax.set_ylabel(f"{label}\nmotion noise (× nominal)")
            if row == 1:
                ax.set_xlabel("sonar outlier probability")
            ax.add_patch(Rectangle((list(m.columns).index(0.03) - 0.5, list(m.index).index(1.0) - 0.5), 1, 1,
                                   fill=False, ec=INK, lw=1.4))
            _title(ax, _short(v) if row == 0 else "")
    _suptitle(fig, "Noise robustness: simulator truth varied, filter parameters fixed (Protocol A)",
              "30 seeds (2000–2029) per cell; outlined cell = nominal condition; darker = better; "
              "shading shared within each row", y=1.06)
    fig.tight_layout()
    _save(fig, "08_noise_robustness.png")


# =========================================================== 9. navigation trade-off
NAV_OFFSETS = {"V1": (8, -6, "left"), "V2": (8, 9, "left"), "V4": (8, 1, "left"), "V3": (-8, 6, "right"),
               "V6": (8, -6, "left"), "V5": (8, 0, "left"), "V8": (-8, 7, "right"), "V3-N1000": (8, 4, "left")}


def fig_navigation(cls):
    fig = plt.figure(figsize=(12.5, 4.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1], wspace=0.35)
    ax = fig.add_subplot(gs[0, 0])
    d = cls[cls.scenario == "kidnap"].set_index("variant")
    for v, r in d.iterrows():
        ax.errorbar(r.wall_contacts, r.cl_recovered,
                    xerr=[[r.wall_contacts - r.wall_contacts_lo], [r.wall_contacts_hi - r.wall_contacts]],
                    yerr=[[r.cl_recovered - r.cl_recovered_lo], [r.cl_recovered_hi - r.cl_recovered]], fmt="o",
                    ms=8, color=COLORS[v], mec=COLORS[v],
                    mfc=SURFACE if VARIANTS[v].role != "factorial" else COLORS[v], mew=2, elinewidth=1,
                    capsize=0, zorder=3)
        dx, dy, ha = NAV_OFFSETS.get(v, (8, 5, "left"))
        ax.annotate(f"{v} · {r.final_success:.0%} end < 10 cm", (r.wall_contacts, r.cl_recovered),
                    xytext=(dx, dy), textcoords="offset points", fontsize=7.8, color=INK2, ha=ha, va="center")
    ax.set_xlabel("simulator wall-contact events per run (mean, bootstrap 95% CI)")
    ax.set_ylabel("sustained recovery after the kidnap")
    _pct(ax)
    ax.margins(x=0.12)
    _title(ax, "Kidnap: recovery vs collisions",
           f"budget {S.CLOSED_LOOP_BUDGET['kidnap']} sonar updates, ≤{S.WAYPOINT_ITERATIONS} attempts per waypoint; "
           "Wilson 95% CI on y")
    ax = fig.add_subplot(gs[0, 1])
    t = cls[cls.scenario == "tracking"].set_index("variant")
    k = cls[cls.scenario == "kidnap"].set_index("variant")
    order = [v for v in ORDER if v in t.index]
    y = np.arange(len(order))[::-1]
    for yy, v in zip(y, order):
        for off, frame, color in ((0.17, t, MUTED), (-0.17, k, COLORS["V8"])):
            r = frame.loc[v]
            ax.barh(yy + off, r.all_verified, height=0.32, color=color)
            ax.plot([r.all_verified_lo, r.all_verified_hi], [yy + off] * 2, color=INK, lw=1)
            ax.annotate(f"{r.all_verified:.0%}", (r.all_verified_hi, yy + off), xytext=(3, 0),
                        textcoords="offset points", va="center", fontsize=7.2, color=INK2)
    ax.set_yticks(y, order)
    ax.set_xlim(0, 1.15)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.grid(axis="y", visible=False)
    ax.legend(handles=[Rectangle((0, 0), 1, 1, color=MUTED, label="tracking (0 wall contacts for every variant)"),
                       Rectangle((0, 0), 1, 1, color=COLORS["V8"], label="kidnap")],
              fontsize=7.3, loc="lower center", bbox_to_anchor=(0.5, -0.32), ncol=1)
    _title(ax, "Runs where all 8 waypoints were truly reached",
           "true pose within 10 cm when the waypoint was declared")
    _suptitle(fig, "Closed-loop navigation reliability (Protocol B, 100 seeds; trajectories differ between "
                   "variants, so comparisons are unpaired)", y=1.08)
    _save(fig, "09_navigation_tradeoff.png")


def fig_uncertainty(points, unc):
    from matplotlib.colors import LogNorm

    fig = plt.figure(figsize=(13.5, 4.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.25], wspace=0.32)
    for col, v in enumerate(("V1", "V8")):
        ax = fig.add_subplot(gs[0, col])
        d = points[(points.scenario == "kidnap") & (points.variant == v)]
        _, _, _, im = ax.hist2d(np.clip(d.r95, 0, 119.9), np.clip(d.pos_err, 0, 119.9), bins=40,
                                range=[[0, 120], [0, 120]], cmap=SEQ, norm=LogNorm(vmin=1), cmin=1)
        ax.plot([0, 120], [0, 120], color=INK2, lw=1)
        ax.add_patch(Rectangle((0, S.CONFIDENT_R95_CM), S.CONFIDENT_R95_CM, 120 - S.CONFIDENT_R95_CM,
                               fill=False, ec=COLORS["V8"], lw=1.4))
        ax.text(23, 112, "false confidence\n(r95 ≤ 20 cm, error > 20 cm)", fontsize=7.5, color=INK2, va="top")
        ax.text(112, 100, "error = r95", fontsize=7.5, color=INK2, ha="right", rotation=45)
        ax.set_xlabel("95% credible radius r95 (cm); values > 120 clipped")
        ax.set_ylabel("actual position error (cm)")
        ax.grid(False)
        _title(ax, f"{_short(v)}", "kidnap updates; count per cell, log colour scale")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="updates (log scale)")
    ax = fig.add_subplot(gs[0, 2])
    order = [v for v in ORDER if v in set(unc.variant)]
    y = np.arange(len(order))[::-1]
    for scenario, off, color in (("tracking", 0.2, MUTED), ("kidnap", -0.2, "#2a78d6")):
        u = unc[unc.scenario == scenario].set_index("variant")
        for yy, v in zip(y, order):
            if v not in u.index:
                continue
            r = u.loc[v]
            ax.barh(yy + off, r.coverage95, height=0.36, color=color)
            ax.annotate(f"{r.coverage95:.0%} cover · {r.false_conf_update_frac:.0%} false-conf.",
                        (r.coverage95, yy + off), xytext=(3, 0), textcoords="offset points", va="center",
                        fontsize=6.8, color=INK2)
    ax.axvline(0.95, color=INK2, lw=1)
    ax.text(0.95, len(order) - 0.35, "nominal 95%", fontsize=7.2, color=INK2, ha="center")
    ax.set_yticks(y, order)
    ax.set_xlim(0, 1.6)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.grid(axis="y", visible=False)
    ax.legend(handles=[Rectangle((0, 0), 1, 1, color=MUTED, label="tracking"),
                       Rectangle((0, 0), 1, 1, color="#2a78d6", label="kidnap")],
              fontsize=7.5, loc="lower center", bbox_to_anchor=(0.35, -0.3), ncol=2)
    _title(ax, "Coverage of the 95% credible circle", "share of updates with error ≤ r95; text adds the "
                                                     "false-confidence share")
    _suptitle(fig, "Is the particle cloud honest about its uncertainty? (Protocol A)", y=1.06)
    _save(fig, "10_uncertainty_diagnostics.png")


def _grid_image(px, py, values, step=2.5):
    xs = np.arange(step / 2, 210, step)
    img = np.full((len(xs), len(xs)), np.nan)
    ix = np.round((px - step / 2) / step).astype(int)
    iy = np.round((py - step / 2) / step).astype(int)
    img[iy, ix] = values
    return img


def fig_global(glob, steps_main):
    world = arena()
    maps = np.load(S.RESULTS / "raw" / "global_diag" / "likelihood_maps.npz")
    px, py = maps["px"], maps["py"]
    candidates = glob[(glob.variant == "V1") & glob.seed.isin(S.SNAPSHOT_SEEDS)]
    picks = []
    for cat in ("converged", "ambiguous", "hypothesis lost", "truth implausible"):
        c = candidates[candidates.category == cat]
        if len(c):
            picks.append((int(c.seed.iloc[0]), cat))
    picks = picks[:3]
    fig = plt.figure(figsize=(13, 10))
    gs = fig.add_gridspec(2, len(picks), height_ratios=[1.15, 0.85], hspace=0.45, wspace=0.18)
    for col, (seed, cat) in enumerate(picks):
        ax = fig.add_subplot(gs[0, col])
        ll = maps[f"seed_{seed}"]
        rel = np.clip(ll - ll.max(), -10, 0)  # log-likelihood relative to the best start position
        ax.imshow(_grid_image(px, py, rel), origin="lower", extent=(0, 210, 0, 210), cmap=SEQ, vmin=-10, vmax=0,
                  interpolation="nearest")
        _draw_world(ax, world)
        tr = glob[(glob.seed == seed) & (glob.variant == "V1")].iloc[0]
        g = steps_main[(steps_main.scenario == "global") & (steps_main.variant == "V1") & (steps_main.seed == seed)]
        snaps = _load_snaps("global", seed, "V1")
        if snaps:
            p = snaps[min(11, len(snaps) - 1)]
            ax.scatter(p[:, 0], p[:, 1], s=6, color=INK, alpha=0.5, lw=0)
        best = int(np.argmax(ll))
        ax.plot(px[best], py[best], "P", ms=10, color="#1c5cab", mec=SURFACE, mew=1.5, zorder=5)
        ax.plot(g.true_x.iloc[0], g.true_y.iloc[0], "o", ms=9, color=COLORS["V8"], mec=SURFACE, mew=1.8, zorder=6)
        gap = (f"truth {tr.truth_gap:.1f} nats below the best start" if tr.truth_gap > 0
               else "truth fits as well as the best start")
        ax.set_title(f"seed {seed}: {cat}", pad=44)
        ax.text(0, 1.02, f"{gap}\ncompetitor area: {tr.competitor_area_frac:.0%} of the arena\n"
                f"particle mass near the truth after the sweep: {tr.mass_near_truth_sweep1:.2f}",
                transform=ax.transAxes, fontsize=7.8, color=INK2, va="bottom")
    ax = fig.add_subplot(gs[1, :])
    cats = ["converged", "ambiguous", "hypothesis lost", "truth implausible"]
    colors = ["#1c5cab", "#eda100", COLORS["V8"], MUTED]
    vs = list(S.MAIN_VARIANTS["global"])
    left = np.zeros(len(vs))
    y = np.arange(len(vs))[::-1]
    for cat, color in zip(cats, colors):
        vals = np.array([((glob.variant == v) & (glob.category == cat)).sum() for v in vs], float)
        ax.barh(y, vals, left=left, color=color, height=0.6, label=cat, edgecolor=SURFACE, lw=1)
        left += vals
    ax.set_yticks(y, [_short(v) for v in vs], fontsize=8)
    ax.set_xlabel("runs (of 100)")
    ax.legend(fontsize=8, loc="lower center", bbox_to_anchor=(0.5, -0.32), ncol=4)
    ax.grid(axis="y", visible=False)
    _title(ax, "Outcome of every global run (100 per variant)",
           "converged = final error < 10 cm; failures classified by the prespecified rules in PROTOCOL.md")
    _suptitle(fig, "Global localisation: how well the first 12-reading sonar sweep identifies the start",
              "maps: log-likelihood of each start position relative to the best (max over heading; palest = 10+ "
              "nats worse); blue + = best-fitting start; red dot = true start; black dots = V1 particles after "
              "the sweep", y=0.98)
    _save(fig, "11_global_diagnostic.png")


def main() -> None:
    summ = pd.read_csv(A / "main_summary.csv")
    effects = pd.read_csv(A / "factorial_effects.csv")
    cl = pd.read_csv(A / "closed_loop_summary.csv")
    runs = pd.read_csv(A / "main_runs.csv")
    km = pd.read_csv(A / "kidnap_km.csv")
    dyn = pd.read_csv(A / "kidnap_dynamics.csv")
    cpu = pd.read_csv(A / "cpu_stages.csv")
    cvp = pd.read_csv(A / "cpu_vs_particles.csv.gz")
    grid = pd.read_csv(A / "noise_grid.csv")
    unc = pd.read_csv(A / "uncertainty.csv")
    points = pd.read_csv(A / "uncertainty_points.csv.gz")
    glob = pd.read_csv(A / "global_runs.csv")
    steps_main = S.read_csv_gz((S.RESULTS / "raw" / "main").glob("nominal/*/seeds_*.csv.gz"))
    steps_main["since_kidnap"] = pd.to_numeric(steps_main["since_kidnap"], errors="coerce")
    for old in FIG.glob("*.png") if FIG.exists() else []:
        old.unlink()
    fig_scorecard(summ, cl)
    fig_factorial(summ, effects)
    fig_recovery_cost(summ)
    fig_recovery_time(km, runs)
    fig_cases(runs, steps_main)
    fig_dynamics(dyn)
    fig_cpu(cpu, cvp)
    fig_noise(grid)
    fig_navigation(cl)
    fig_uncertainty(points, unc)
    fig_global(glob, steps_main)


if __name__ == "__main__":
    main()
