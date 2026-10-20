"""Analyse the saved raw results of the v2 study. No simulation is run here.

    python experiments/noise_study/analyze.py

Reads results/raw/** and writes results/analysis/*.csv plus
results/analysis/summary.md. Statistical methods follow PROTOCOL.md.
"""

from __future__ import annotations

import itertools
import json
from typing import Dict, List

import numpy as np
import pandas as pd
import study_lib as S

from legobot.localisation import FACTORIAL, VARIANTS

RAW = S.RESULTS / "raw"
OUT = S.RESULTS / "analysis"
N_BOOT = 10000
FACTORS = {"E": "ess_gating", "I": "injection", "K": "kld"}
RUNAWAY_CM = 300.0


# ------------------------------------------------------------------ loading
def load_replay(batch: str) -> pd.DataFrame:
    df = S.read_csv_gz(RAW.glob(f"{batch}/*/*/seeds_*.csv.gz"))
    df["since_kidnap"] = pd.to_numeric(df["since_kidnap"], errors="coerce")
    df["ess_ratio"] = pd.to_numeric(df["ess_ratio"], errors="coerce")
    df["cpu_ms"] = df["predict_ms"] + df["update_ms"]
    return df


# ------------------------------------------------------------------ per run
def per_run(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (cond, scenario, variant, seed), g in df.groupby(["condition", "scenario", "variant", "seed"],
                                                       sort=False):
        g = g.sort_values("update")
        err = g["pos_err"].to_numpy()
        valid = g["valid"].to_numpy().astype(bool)
        r = {"condition": cond, "scenario": scenario, "variant": variant, "seed": seed,
             "updates": len(g), "valid_updates": int(valid.sum()),
             "mean_err": err.mean(), "final_err": err[-1], "final_success": int(err[-1] < S.SUCCESS_CM),
             "cpu_ms": g["cpu_ms"].sum(), "mean_particles": g["n_after"].mean(),
             "max_particles": g["n_after"].max(),
             "resample_frac": g.loc[valid, "resampled"].mean() if valid.any() else np.nan,
             "ess_ratio": g.loc[valid, "ess_ratio"].mean() if valid.any() else np.nan,
             "unique_pose_frac": g["unique_pose_frac"].mean(), "lineage_frac": g["lineage_frac"].mean(),
             "bins": g["bins"].mean(), "injected": g["injected"].sum(), "p_inject": g["p_inject"].mean(),
             "false_conf_frac": np.mean((g["r95"] <= S.CONFIDENT_R95_CM) & (g["pos_err"] > S.CONFIDENT_R95_CM)),
             "coverage": np.mean(g["pos_err"] <= g["r95"]),
             "mass_near_truth_final": g["mass_near_truth"].iloc[-1]}
        if scenario == "kidnap":
            post = g[g["since_kidnap"] >= 0]
            perr = post["pos_err"].to_numpy()
            idx, observed = S.recovery_stats(perr)
            r.update({"horizon": len(perr), "recovery_update": idx, "recovered": int(observed),
                      "dipped_below": int((perr < S.SUCCESS_CM).any()),
                      "post_mean_err": perr.mean(), "post_median_err": np.median(perr),
                      "post_p95_err": np.percentile(perr, 95),
                      # estimate left the arena (diagonal 297 cm): a runaway filter
                      "runaway": int(perr.max() > RUNAWAY_CM),
                      "post_cpu_ms": post["cpu_ms"].sum(),
                      "post_mean_particles": post["n_after"].mean()})
        if scenario == "global":
            idx, observed = S.recovery_stats(err)
            r.update({"horizon": len(err), "recovery_update": idx, "recovered": int(observed),
                      "mass_near_truth_sweep1": float(g["mass_near_truth"].iloc[min(11, len(g) - 1)]),
                      "mass_near_truth_max": float(g["mass_near_truth"].max())})
        rows.append(r)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ statistics
def boot_mean_ci(x, n_boot=N_BOOT, seed=1):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), (n_boot, len(x)))].mean(axis=1)
    return x.mean(), np.percentile(means, 2.5), np.percentile(means, 97.5)


def variant_summary(runs: pd.DataFrame, scenario: str, metrics: List[str], binary: List[str]):
    out = []
    for variant, g in runs[runs["scenario"] == scenario].groupby("variant", sort=False):
        r = {"scenario": scenario, "variant": variant, "label": VARIANTS[variant].label,
             "role": VARIANTS[variant].role, "n": len(g)}
        for m in binary:
            p, lo, hi = S.wilson(int(g[m].sum()), len(g))
            r.update({m: p, f"{m}_lo": lo, f"{m}_hi": hi, f"{m}_k": int(g[m].sum())})
        for m in metrics:
            mean, lo, hi = boot_mean_ci(g[m])
            r.update({m: mean, f"{m}_lo": lo, f"{m}_hi": hi})
        out.append(r)
    return pd.DataFrame(out)


def _cell_matrix(runs, scenario, metric, variants):
    pivot = runs[runs["scenario"] == scenario].pivot(index="seed", columns="variant", values=metric)
    return pivot[list(variants)].to_numpy().T  # (cells, seeds)


def factorial_effects(runs: pd.DataFrame, scenario: str, metric: str) -> pd.DataFrame:
    """Main effects and interactions with +/-1 coding over the 8 cell means,
    95% CIs from a paired bootstrap over seeds."""
    M = _cell_matrix(runs, scenario, metric, FACTORIAL)
    signs = {f: np.array([1 if getattr(VARIANTS[v].options, attr) else -1 for v in FACTORIAL])
             for f, attr in FACTORS.items()}
    contrasts = {}
    for size in (1, 2, 3):
        for combo in itertools.combinations("EIK", size):
            contrasts["×".join(combo)] = np.prod([signs[c] for c in combo], axis=0) / 4.0
    rng = np.random.default_rng(2024)
    idx = rng.integers(0, M.shape[1], (N_BOOT, M.shape[1]))
    boot_means = M[:, idx].mean(axis=2)  # (cells, boots)
    rows = []
    cell_means = M.mean(axis=1)
    for name, c in contrasts.items():
        est = float(c @ cell_means)
        b = c @ boot_means
        rows.append({"scenario": scenario, "metric": metric, "effect": name, "estimate": est,
                     "ci_lo": float(np.percentile(b, 2.5)), "ci_hi": float(np.percentile(b, 97.5)),
                     "supported": int(np.percentile(b, 2.5) > 0 or np.percentile(b, 97.5) < 0)})
    return pd.DataFrame(rows)


def paired_diff(runs, scenario, metric, a, b) -> Dict:
    pivot = runs[runs["scenario"] == scenario].pivot(index="seed", columns="variant", values=metric)
    est, lo, hi = S.paired_bootstrap(pivot[a].to_numpy(), pivot[b].to_numpy(), n_boot=N_BOOT)
    return {"scenario": scenario, "metric": metric, "comparison": f"{a} - {b}", "a": a, "b": b,
            "mean_a": pivot[a].mean(), "mean_b": pivot[b].mean(), "diff": est, "ci_lo": lo,
            "ci_hi": hi, "supported": int(lo > 0 or hi < 0)}


# ------------------------------------------------------------------ main
def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    md: List[str] = ["# Noise study v2: analysis summary (generated by analyze.py)\n"]

    # ---------------- Protocol A, nominal
    main_df = load_replay("main")
    runs = per_run(main_df)
    runs.to_csv(OUT / "main_runs.csv", index=False)

    common = ["mean_err", "final_err", "cpu_ms", "mean_particles", "resample_frac", "ess_ratio",
              "unique_pose_frac", "lineage_frac", "bins", "injected", "false_conf_frac", "coverage"]
    summ = pd.concat([
        variant_summary(runs, "tracking", common, ["final_success"]),
        variant_summary(runs, "kidnap", common + ["post_mean_err", "post_median_err", "post_p95_err",
                                                   "post_cpu_ms", "post_mean_particles"],
                        ["recovered", "final_success", "dipped_below"]),
        variant_summary(runs, "global", common + ["mass_near_truth_sweep1", "mass_near_truth_max"],
                        ["recovered", "final_success"]),
    ], ignore_index=True)
    summ.to_csv(OUT / "main_summary.csv", index=False)

    effects = pd.concat([
        factorial_effects(runs, "kidnap", "recovered"),
        factorial_effects(runs, "kidnap", "post_mean_err"),
        factorial_effects(runs, "kidnap", "post_median_err"),
        factorial_effects(runs, "kidnap", "cpu_ms"),
        factorial_effects(runs, "tracking", "mean_err"),
        factorial_effects(runs, "tracking", "cpu_ms"),
        factorial_effects(runs, "global", "recovered"),
    ], ignore_index=True)
    effects.to_csv(OUT / "factorial_effects.csv", index=False)

    comparisons = [("kidnap", m, a, b) for m in ("recovered", "cpu_ms", "mean_particles", "post_mean_err", "post_median_err")
                   for a, b in (("V3", "V1"), ("V8", "V4"), ("V8", "V3"), ("V8", "V3-N1000"),
                                ("V8", "V3-N500"), ("V5", "V3-N1000"), ("V5", "V8"),
                                ("V3-N1000", "V3"), ("V3-N500", "V3"), ("V1-N1000", "V1"),
                                ("V6", "V3"), ("V2", "V1"), ("V4", "V1"), ("V7", "V2"))]
    comparisons += [("tracking", m, a, b) for m in ("mean_err", "cpu_ms", "lineage_frac",
                                                     "unique_pose_frac")
                    for a, b in (("V2", "V1"), ("V4", "V1"), ("V3", "V1"), ("V5", "V1"),
                                 ("V8", "V1"), ("V1-N1000", "V1"), ("V0", "V1"))]
    comparisons += [("global", m, a, b) for m in ("recovered", "cpu_ms")
                    for a, b in (("V4", "V1"), ("V8", "V1"), ("V5", "V1"), ("V2", "V1"))]
    pairs = pd.DataFrame([paired_diff(runs, *c) for c in comparisons])
    pairs.to_csv(OUT / "paired_comparisons.csv", index=False)

    # interaction of injection with KLD as simple effects (paired bootstrap)
    kid = runs[runs["scenario"] == "kidnap"].pivot(index="seed", columns="variant", values="recovered")
    a = (kid["V8"] - kid["V4"]).to_numpy()
    b = (kid["V3"] - kid["V1"]).to_numpy()
    est, lo, hi = S.paired_bootstrap(a, b, n_boot=N_BOOT)
    simple = pd.DataFrame([{"contrast": "(V8 - V4) - (V3 - V1): injection effect with KLD minus without",
                            "estimate": est, "ci_lo": lo, "ci_hi": hi}])
    simple.to_csv(OUT / "simple_effects.csv", index=False)

    # Robust post-kidnap error: median across runs of each run's median error,
    # bootstrap CI of that median; runaway rate with Wilson CI.
    robust = []
    kr = runs[runs["scenario"] == "kidnap"]
    for variant, g in kr.groupby("variant", sort=False):
        x = g["post_median_err"].to_numpy()
        boot = np.median(x[np.random.default_rng(3).integers(0, len(x), (N_BOOT, len(x)))], axis=1)
        p, lo, hi = S.wilson(int(g["runaway"].sum()), len(g))
        robust.append({"variant": variant, "median_post_err": float(np.median(x)),
                       "median_post_err_lo": float(np.percentile(boot, 2.5)),
                       "median_post_err_hi": float(np.percentile(boot, 97.5)),
                       "runaway": p, "runaway_lo": lo, "runaway_hi": hi, "runaway_k": int(g["runaway"].sum())})
    robust = pd.DataFrame(robust)
    robust.to_csv(OUT / "kidnap_robust_error.csv", index=False)
    kmed = kr.pivot(index="seed", columns="variant", values="post_median_err")
    med_pairs = []
    for a_, b_ in (("V3", "V1"), ("V8", "V4"), ("V8", "V3"), ("V8", "V3-N1000"), ("V5", "V3-N1000"),
                   ("V3-N1000", "V3"), ("V2", "V1"), ("V4", "V1")):
        est, lo, hi = S.paired_bootstrap(kmed[a_].to_numpy(), kmed[b_].to_numpy(), stat=np.median,
                                         n_boot=2000)
        med_pairs.append({"comparison": f"{a_} - {b_}", "median_a": kmed[a_].median(),
                          "median_b": kmed[b_].median(), "diff_of_medians": est, "ci_lo": lo, "ci_hi": hi})
    med_pairs = pd.DataFrame(med_pairs)
    med_pairs.to_csv(OUT / "kidnap_median_comparisons.csv", index=False)
    # How long runaway episodes last (exploratory, descriptive)
    post = main_df[(main_df["scenario"] == "kidnap") & (main_df["since_kidnap"] >= 0)]
    per = post.groupby(["variant", "seed"])["pos_err"].agg(
        mx="max", frac=lambda x: (x > RUNAWAY_CM).mean(), last="last")
    runaway_dur = (per[per["mx"] > RUNAWAY_CM].groupby("variant")
                   .agg(runaway_runs=("mx", "size"), frac_updates_over_300=("frac", "mean"),
                        final_err_median=("last", "median")).reset_index())
    runaway_dur.to_csv(OUT / "kidnap_runaway_duration.csv", index=False)

    # KM curves (kidnap, nominal)
    km_rows = []
    for variant, g in runs[runs["scenario"] == "kidnap"].groupby("variant"):
        t, s = S.kaplan_meier(g["recovery_update"], g["recovered"].astype(bool))
        km_rows += [{"variant": variant, "t": float(tt), "not_recovered": float(ss)} for tt, ss in zip(t, s)]
        # Extend the last step to the longest horizon (traces differ in length).
        km_rows.append({"variant": variant, "t": float(max(g["horizon"].max(), t[-1])),
                        "not_recovered": float(s[-1]), "end": 1})
    pd.DataFrame(km_rows).to_csv(OUT / "kidnap_km.csv", index=False)

    # CPU: per-update latency and stage breakdown on valid updates
    stage_cols = [c for c in main_df.columns if c.startswith("t_") and c.endswith("_ms")]
    cpu_rows = []
    for (scenario, variant), g in main_df[main_df["valid"] == 1].groupby(["scenario", "variant"]):
        r = {"scenario": scenario, "variant": variant, "updates": len(g),
             "cpu_median_ms": g["cpu_ms"].median(), "cpu_p95_ms": g["cpu_ms"].quantile(0.95),
             "cpu_p99_ms": g["cpu_ms"].quantile(0.99), "cpu_mean_ms": g["cpu_ms"].mean(),
             "predict_mean_ms": g["predict_ms"].mean(), "particles_mean": g["n_after"].mean(),
             "particles_before_mean": g["n_before"].mean(),
             "resample_frac": g["resampled"].mean()}
        for c in stage_cols:
            r[c.replace("_ms", "_mean_ms")] = g[c].mean()
        r["stage_sum_mean_ms"] = g[stage_cols].sum(axis=1).mean() + g["predict_ms"].mean()
        cpu_rows.append(r)
    pd.DataFrame(cpu_rows).to_csv(OUT / "cpu_stages.csv", index=False)

    # per-update cost vs particle count (for the CPU figure), valid updates, kidnap
    sample = main_df[(main_df["valid"] == 1) & main_df["variant"].isin(["V1", "V4", "V8", "V3-N1000",
                                                                         "V1-N1000"])]
    sample[["scenario", "variant", "n_before", "n_after", "resampled", "cpu_ms", "update_ms"]].to_csv(
        OUT / "cpu_vs_particles.csv.gz", index=False)

    # false confidence / coverage pooled per update
    fc = []
    for (scenario, variant), g in main_df.groupby(["scenario", "variant"]):
        confident = g["r95"] <= S.CONFIDENT_R95_CM
        p_fc, lo, hi = S.wilson(int((confident & (g["pos_err"] > S.CONFIDENT_R95_CM)).sum()), len(g))
        runs_fc = (g.assign(fc=confident & (g["pos_err"] > S.CONFIDENT_R95_CM))
                   .groupby("seed")["fc"].any().mean())
        fc.append({"scenario": scenario, "variant": variant, "updates": len(g),
                   "false_conf_update_frac": p_fc, "fc_lo": lo, "fc_hi": hi,
                   "runs_with_false_conf": runs_fc, "coverage95": float(np.mean(g["pos_err"] <= g["r95"])),
                   "confident_frac": float(confident.mean())})
    pd.DataFrame(fc).to_csv(OUT / "uncertainty.csv", index=False)
    main_df[main_df["variant"].isin(["V1", "V3", "V5", "V8"])][
        ["scenario", "variant", "seed", "update", "pos_err", "r95", "spread"]].to_csv(
        OUT / "uncertainty_points.csv.gz", index=False)

    # particle dynamics around the kidnap (aligned)
    dyn = (main_df[(main_df["scenario"] == "kidnap") & (main_df["since_kidnap"] >= -10)]
           .groupby(["variant", "since_kidnap"])
           .agg(n_after=("n_after", "mean"), n_after_q25=("n_after", lambda x: x.quantile(0.25)),
                n_after_q75=("n_after", lambda x: x.quantile(0.75)), p_inject=("p_inject", "mean"),
                injected=("injected", "mean"), pos_err_median=("pos_err", "median"),
                runs=("seed", "nunique")).reset_index())
    dyn.to_csv(OUT / "kidnap_dynamics.csv", index=False)

    # ---------------- Protocol B
    cl_sum = S.read_csv_gz(RAW.glob("closed_loop/*/seeds_*.csv.gz"))
    cl_steps = S.read_csv_gz(RAW.glob("closed_loop/*/steps_*.csv.gz"))
    cl_steps["since_kidnap"] = pd.to_numeric(cl_steps["since_kidnap"], errors="coerce")
    rec = []
    for (variant, seed), g in cl_steps[cl_steps["scenario"] == "kidnap"].groupby(["variant", "seed"]):
        post = g[g["since_kidnap"] >= 0].sort_values("update")["pos_err"].to_numpy()
        idx, observed = S.recovery_stats(post)
        rec.append({"variant": variant, "seed": seed, "scenario": "kidnap", "cl_recovered": int(observed),
                    "cl_recovery_update": idx, "cl_post_mean_err": post.mean() if len(post) else np.nan})
    cl = cl_sum.merge(pd.DataFrame(rec), on=["variant", "seed", "scenario"], how="left")
    cl["all_verified"] = (cl["waypoints_verified"] == cl["waypoints_total"]).astype(int)
    cl["final_success"] = (cl["final_err"] < S.SUCCESS_CM).astype(int)
    cl.to_csv(OUT / "closed_loop_runs.csv", index=False)
    cl_rows = []
    for (scenario, variant), g in cl.groupby(["scenario", "variant"], sort=False):
        r = {"scenario": scenario, "variant": variant, "n": len(g)}
        for m in ("all_verified", "final_success", "budget_exhausted") + (("cl_recovered",) if scenario == "kidnap" else ()):
            p, lo, hi = S.wilson(int(g[m].sum()), len(g))
            r.update({m: p, f"{m}_lo": lo, f"{m}_hi": hi})
        for m in ("waypoints_verified", "waypoints_declared", "wall_contacts", "stalled_waypoints",
                  "sim_time_s", "updates", "resamples", "final_err", "cpu_ms_total"):
            mean, lo, hi = boot_mean_ci(g[m])
            r.update({m: mean, f"{m}_lo": lo, f"{m}_hi": hi})
        r["final_err_median"] = g["final_err"].median()
        cl_rows.append(r)
    cl_summary = pd.DataFrame(cl_rows)
    cl_summary.to_csv(OUT / "closed_loop_summary.csv", index=False)

    # ---------------- noise grid
    noise_df = load_replay("noise")
    nruns = per_run(noise_df)
    nruns.to_csv(OUT / "noise_runs.csv", index=False)
    grid = []
    for (cond, scenario, variant), g in nruns.groupby(["condition", "scenario", "variant"]):
        motion = float(cond.split("_")[0].replace("motion", ""))
        outlier = float(cond.split("_")[1].replace("outlier", ""))
        r = {"condition": cond, "motion_scale": motion, "outlier_prob": outlier, "scenario": scenario,
             "variant": variant, "n": len(g)}
        if scenario == "tracking":
            mean, lo, hi = boot_mean_ci(g["mean_err"])
            fail = int((g["final_err"] >= S.SUCCESS_CM).sum())
            p, plo, phi = S.wilson(fail, len(g))
            r.update({"mean_err": mean, "mean_err_lo": lo, "mean_err_hi": hi, "failure": p,
                      "failure_lo": plo, "failure_hi": phi})
        else:
            p, plo, phi = S.wilson(int(g["recovered"].sum()), len(g))
            r.update({"recovered": p, "recovered_lo": plo, "recovered_hi": phi})
        r["cpu_ms"] = g["cpu_ms"].mean()
        grid.append(r)
    pd.DataFrame(grid).to_csv(OUT / "noise_grid.csv", index=False)

    # ---------------- global diagnostic
    gd = S.read_csv_gz([RAW / "global_diag" / "global_diag.csv.gz"])
    glob = runs[runs["scenario"] == "global"].merge(gd, on="seed")

    def category(r):
        if r["final_success"]:
            return "converged"
        if not r["plausible"]:
            return "truth implausible"
        return "ambiguous" if r["ambiguous"] else "hypothesis lost"
    glob["category"] = glob.apply(category, axis=1)
    glob.to_csv(OUT / "global_runs.csv", index=False)
    cats = glob.groupby(["variant", "category"]).size().unstack(fill_value=0).reset_index()
    cats.to_csv(OUT / "global_categories.csv", index=False)

    # ---------------- markdown summary
    def fmt_p(r, m):
        return f"{r[m]:.0%} [{r[m + '_lo']:.0%}, {r[m + '_hi']:.0%}]"

    def fmt_m(r, m, d=2):
        return f"{r[m]:.{d}f} [{r[m + '_lo']:.{d}f}, {r[m + '_hi']:.{d}f}]"

    md.append("## Protocol A - kidnap (n = 100 seeds per variant)\n")
    md.append("| Variant | Sustained recovery | Final < 10 cm | Dipped below 10 cm | Post-kidnap mean err (cm) | Post median err (cm) "
              "| Post p95 err (cm) | Mean particles | CPU per trajectory (ms) |")
    md.append("|---|---|---|---|---|---|---|---|---|")
    for _, r in summ[summ["scenario"] == "kidnap"].iterrows():
        md.append(f"| {r['variant']} | {fmt_p(r, 'recovered')} | {fmt_p(r, 'final_success')} | "
                  f"{fmt_p(r, 'dipped_below')} | {fmt_m(r, 'post_mean_err', 1)} | {fmt_m(r, 'post_median_err', 1)} | {fmt_m(r, 'post_p95_err', 1)} | "
                  f"{fmt_m(r, 'mean_particles', 0)} | {fmt_m(r, 'cpu_ms', 1)} |")
    md.append("\n### Kidnap: robust error summary (means above are dominated by runaway runs)\n")
    md.append("| Variant | Median over runs of post-kidnap median error (cm) | Runaway (error ever > 300 cm) |")
    md.append("|---|---|---|")
    for _, r in robust.iterrows():
        md.append(f"| {r['variant']} | {fmt_m(r, 'median_post_err', 1)} | {fmt_p(r, 'runaway')} |")
    md.append("\n| Paired comparison | Median A (cm) | Median B (cm) | Difference of medians | 95% CI |")
    md.append("|---|---|---|---|---|")
    for _, r in med_pairs.iterrows():
        md.append(f"| {r['comparison']} | {r['median_a']:.1f} | {r['median_b']:.1f} | "
                  f"{r['diff_of_medians']:.1f} | [{r['ci_lo']:.1f}, {r['ci_hi']:.1f}] |")
    md.append("\n| Variant | Runaway runs | Share of post-kidnap updates > 300 cm | Final error, median (cm) |")
    md.append("|---|---|---|---|")
    for _, r in runaway_dur.iterrows():
        md.append(f"| {r['variant']} | {r['runaway_runs']} | {r['frac_updates_over_300']:.0%} | "
                  f"{r['final_err_median']:.0f} |")
    md.append("\n## Protocol A - tracking\n")
    md.append("| Variant | Mean err (cm) | Final < 10 cm | Resampled | Unique-pose frac | Lineage frac "
              "| Mean particles | CPU per trajectory (ms) |")
    md.append("|---|---|---|---|---|---|---|---|")
    for _, r in summ[summ["scenario"] == "tracking"].iterrows():
        md.append(f"| {r['variant']} | {fmt_m(r, 'mean_err', 2)} | {fmt_p(r, 'final_success')} | "
                  f"{r['resample_frac']:.0%} | {r['unique_pose_frac']:.2f} | {r['lineage_frac']:.2f} | "
                  f"{fmt_m(r, 'mean_particles', 0)} | {fmt_m(r, 'cpu_ms', 1)} |")
    md.append("\n## Protocol A - global\n")
    md.append("| Variant | Sustained convergence | Final < 10 cm | Mean particles | CPU per trajectory (ms) |")
    md.append("|---|---|---|---|---|")
    for _, r in summ[summ["scenario"] == "global"].iterrows():
        md.append(f"| {r['variant']} | {fmt_p(r, 'recovered')} | {fmt_p(r, 'final_success')} | "
                  f"{fmt_m(r, 'mean_particles', 0)} | {fmt_m(r, 'cpu_ms', 1)} |")
    md.append("\n## Factorial effects (±1 contrasts, paired bootstrap 95% CI)\n")
    md.append("| Scenario | Metric | Effect | Estimate | 95% CI | CI excludes 0 |")
    md.append("|---|---|---|---|---|---|")
    for _, r in effects.iterrows():
        md.append(f"| {r['scenario']} | {r['metric']} | {r['effect']} | {r['estimate']:.3f} | "
                  f"[{r['ci_lo']:.3f}, {r['ci_hi']:.3f}] | {'yes' if r['supported'] else 'no'} |")
    md.append("\n## Paired comparisons (Protocol A, paired bootstrap over seeds)\n")
    md.append("| Scenario | Metric | Comparison | Mean A | Mean B | Difference | 95% CI |")
    md.append("|---|---|---|---|---|---|---|")
    for _, r in pairs.iterrows():
        md.append(f"| {r['scenario']} | {r['metric']} | {r['comparison']} | {r['mean_a']:.3f} | "
                  f"{r['mean_b']:.3f} | {r['diff']:.3f} | [{r['ci_lo']:.3f}, {r['ci_hi']:.3f}] |")
    md.append(f"\nSimple-effect interaction: {simple.iloc[0]['contrast']} = {simple.iloc[0]['estimate']:.3f} "
              f"[{simple.iloc[0]['ci_lo']:.3f}, {simple.iloc[0]['ci_hi']:.3f}]\n")
    md.append("\n## Protocol B - closed loop (unpaired; trajectories differ between variants)\n")
    md.append("| Scenario | Variant | All 8 waypoints verified | Final < 10 cm | Post-kidnap sustained recovery "
              "| Wall contacts | Stalled waypoints | Budget exhausted | Sim time (s) |")
    md.append("|---|---|---|---|---|---|---|---|---|")
    for _, r in cl_summary.iterrows():
        rec_txt = fmt_p(r, "cl_recovered") if r["scenario"] == "kidnap" else "n/a"
        md.append(f"| {r['scenario']} | {r['variant']} | {fmt_p(r, 'all_verified')} | {fmt_p(r, 'final_success')} | "
                  f"{rec_txt} | {fmt_m(r, 'wall_contacts', 1)} | {fmt_m(r, 'stalled_waypoints', 2)} | "
                  f"{fmt_p(r, 'budget_exhausted')} | {fmt_m(r, 'sim_time_s', 0)} |")
    md.append("\n## Global diagnostic categories (counts of 100 runs)\n")
    md.append("| " + " | ".join(cats.columns) + " |")
    md.append("|" + "---|" * len(cats.columns))
    for _, r in cats.iterrows():
        md.append("| " + " | ".join(str(v) for v in r.values) + " |")
    md.append(f"\nGlobal traces: plausible truth {gd['plausible'].mean():.0%}, ambiguous first sweep "
              f"{gd['ambiguous'].mean():.0%} (of 100).\n")
    (OUT / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (OUT / "provenance.json").write_text(json.dumps(
        {"generated_by": "analyze.py", "inputs": "results/raw", "n_boot": N_BOOT,
         "git": S.git_revision()}, indent=2), encoding="utf-8")
    print((OUT / "summary.md").read_text(encoding="utf-8")[:6000])


if __name__ == "__main__":
    main()
