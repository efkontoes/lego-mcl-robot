"""Run the revised noise study (v2). Every batch is deterministic and resumable:
results are written as atomic part files and completed parts are skipped.

    python experiments/noise_study/run_study.py traces       # reference traces (Protocol A)
    python experiments/noise_study/run_study.py main         # factorial + budget controls, seeds 1000-1099
    python experiments/noise_study/run_study.py closed_loop  # Protocol B navigation, seeds 1000-1099
    python experiments/noise_study/run_study.py noise        # 3x3 noise grid, seeds 2000-2029
    python experiments/noise_study/run_study.py global_diag  # likelihood-grid diagnostic of global runs
    python experiments/noise_study/run_study.py all

Parameters are frozen in config/robot.yaml and in the constants of
study_lib.py; nothing here is tuned on the evaluation seeds.
"""

from __future__ import annotations

import argparse
import math
import time
from typing import List

import numpy as np
import study_lib as S

from legobot.localisation import arena
from legobot.localisation.sensor_model import SonarModel

BLOCK = 10


def _blocks(seeds) -> List[List[int]]:
    seeds = list(seeds)
    return [seeds[i:i + BLOCK] for i in range(0, len(seeds), BLOCK)]


def _scenarios_for(batch):
    return ("tracking", "kidnap") if batch == "noise" else ("tracking", "kidnap", "global")


def _conditions(batch):
    if batch == "noise":
        return [(m, o) for m in S.MOTION_SCALES for o in S.OUTLIER_PROBS]
    return [(1.0, None)]


# ---------------------------------------------------------------- traces
def batch_traces(include_noise: bool = True) -> None:
    base = S.nominal_config()
    jobs = [("nominal", 1.0, None, S.MAIN_SEEDS, ("tracking", "kidnap", "global"))]
    if include_noise:
        jobs += [(S.condition_name(m, o), m, o, S.NOISE_SEEDS, ("tracking", "kidnap"))
                 for m in S.MOTION_SCALES for o in S.OUTLIER_PROBS]
    for name, m, o, seeds, scenarios in jobs:
        cfg = S.condition_config(base, m, o) if name != "nominal" else base
        for scenario in scenarios:
            path = S.trace_path(name, scenario)
            if path.exists():
                continue
            t0 = time.time()
            S.save_traces([S.generate_trace(cfg, scenario, s, name) for s in seeds], path)
            print(f"traces {name:24s} {scenario:9s} {len(seeds)} seeds ({time.time() - t0:.1f}s)",
                  flush=True)
    S.write_metadata("traces", {"main_seeds": [S.MAIN_SEEDS.start, S.MAIN_SEEDS.stop - 1],
                                "noise_seeds": [S.NOISE_SEEDS.start, S.NOISE_SEEDS.stop - 1],
                                "motion_scales": S.MOTION_SCALES, "outlier_probs": S.OUTLIER_PROBS})


# ---------------------------------------------------------------- replay
def _replay_batch(batch: str, condition: str, scenarios, variants_for, snapshot: bool) -> None:
    cfg = S.nominal_config()  # filters always use the nominal parameters
    for scenario in scenarios:
        traces = S.load_traces(S.trace_path(condition, scenario))
        for block in _blocks(t["seed"] for t in traces):
            part = S.RESULTS / "raw" / batch / condition / scenario / f"seeds_{block[0]}_{block[-1]}.csv.gz"
            if part.exists():
                continue
            t0 = time.time()
            rows = []
            for trace in (t for t in traces if t["seed"] in block):
                for variant in variants_for(scenario):  # variants interleaved per seed
                    keep = (snapshot and scenario != "tracking" and trace["seed"] in S.SNAPSHOT_SEEDS
                            and variant in S.SNAPSHOT_VARIANTS)
                    r, snaps = S.replay(trace, cfg, variant, keep_snapshots=keep)
                    rows.extend(r)
                    if keep:
                        name = f"{scenario}_{trace['seed']}_{variant}.npz"
                        snap = S.RESULTS / "raw" / batch / "snapshots" / name
                        snap.parent.mkdir(parents=True, exist_ok=True)
                        np.savez_compressed(snap, *snaps)
            S.write_csv_gz(rows, part)
            print(f"{batch} {condition:24s} {scenario:9s} seeds {block[0]}-{block[-1]} "
                  f"({time.time() - t0:.1f}s)", flush=True)


def batch_main() -> None:
    S.write_metadata("main", {"seeds": [S.MAIN_SEEDS.start, S.MAIN_SEEDS.stop - 1],
                              "variants": S.MAIN_VARIANTS, "global_particles": S.GLOBAL_PARTICLES,
                              "sustain_updates": S.SUSTAIN_UPDATES, "success_cm": S.SUCCESS_CM})
    _replay_batch("main", "nominal", ("tracking", "kidnap", "global"),
                  lambda sc: S.MAIN_VARIANTS[sc], snapshot=True)


def batch_noise() -> None:
    S.write_metadata("noise", {"seeds": [S.NOISE_SEEDS.start, S.NOISE_SEEDS.stop - 1],
                               "variants": S.NOISE_VARIANTS, "motion_scales": S.MOTION_SCALES,
                               "outlier_probs": S.OUTLIER_PROBS})
    for m in S.MOTION_SCALES:
        for o in S.OUTLIER_PROBS:
            _replay_batch("noise", S.condition_name(m, o), ("tracking", "kidnap"),
                          lambda sc: S.NOISE_VARIANTS, snapshot=False)


# ---------------------------------------------------------------- closed loop
def batch_closed_loop() -> None:
    cfg = S.nominal_config()
    S.write_metadata("closed_loop", {"seeds": [S.MAIN_SEEDS.start, S.MAIN_SEEDS.stop - 1],
                                     "variants": S.CLOSED_LOOP_VARIANTS,
                                     "update_budget": S.CLOSED_LOOP_BUDGET,
                                     "waypoint_iterations": S.WAYPOINT_ITERATIONS})
    for scenario in ("tracking", "kidnap"):
        for block in _blocks(S.MAIN_SEEDS):
            part = S.RESULTS / "raw" / "closed_loop" / scenario / f"seeds_{block[0]}_{block[-1]}.csv.gz"
            if part.exists():
                continue
            t0 = time.time()
            summaries, steps = [], []
            for seed in block:
                for variant in S.CLOSED_LOOP_VARIANTS:
                    summary, rows = S.run_closed_loop(cfg, scenario, variant, seed)
                    summaries.append(summary)
                    steps.extend(rows)
            S.write_csv_gz(steps, part.with_name(part.name.replace("seeds_", "steps_")))
            S.write_csv_gz(summaries, part)
            print(f"closed_loop {scenario:9s} seeds {block[0]}-{block[-1]} ({time.time() - t0:.1f}s)",
                  flush=True)


# ---------------------------------------------------------------- global diagnostic
def _sweep_offsets(trace):
    """Commanded heading offset (from the start) at each sonar reading of the
    first sweep, and the readings themselves."""
    offset, out = 0.0, []
    sweep = S.nominal_config().navigation.sweep_steps
    for kind, value in trace["events"]:
        if kind == "turn":
            offset += value
        elif kind == "move":
            break
        elif kind == "z":
            out.append((offset, value))
            if len(out) == sweep:
                break
    return out


def _loglik(model: SonarModel, x, y, theta0, readings, chunk=120_000):
    total = np.zeros(len(x))
    for offset, z in readings:
        if not model.is_valid(z):
            continue
        for a in range(0, len(x), chunk):
            sl = slice(a, a + chunk)
            pts = np.column_stack([x[sl], y[sl], theta0[sl] + offset])
            total[sl] += np.log(model.likelihood(pts, z))
    return total


def batch_global_diag(grid_cm: float = 2.5, heading_step_deg: float = 5.0) -> None:
    """For every global trace: likelihood of the first sonar sweep over a dense
    grid of start poses (motion noise during the on-the-spot sweep ignored)."""
    out_csv = S.RESULTS / "raw" / "global_diag" / "global_diag.csv.gz"
    if out_csv.exists():
        return
    cfg = S.nominal_config()
    world = arena()
    model = SonarModel(world, cfg.sonar)
    gx, gy = np.meshgrid(np.arange(grid_cm / 2, 210, grid_cm), np.arange(grid_cm / 2, 210, grid_cm))
    keep = world.contains(gx.ravel(), gy.ravel())
    px, py = gx.ravel()[keep], gy.ravel()[keep]
    headings = np.radians(np.arange(0, 360, heading_step_deg))
    X = np.repeat(px, len(headings))
    Y = np.repeat(py, len(headings))
    TH = np.tile(headings, len(px))
    rows, maps = [], {}
    t0 = time.time()
    for trace in S.load_traces(S.trace_path("nominal", "global")):
        readings = _sweep_offsets(trace)
        valid = sum(model.is_valid(z) for _, z in readings)
        ll = _loglik(model, X, Y, TH, readings).reshape(len(px), len(headings))
        pos_ll = ll.max(axis=1)
        sx, sy, sth = trace["start"]
        ll_truth = float(_loglik(model, np.array([sx]), np.array([sy]), np.array([sth]), readings)[0])
        dist = np.hypot(px - sx, py - sy)
        far = dist > S.TRUTH_NEAR_CM
        competitors = far & (pos_ll >= ll_truth - math.log(10))
        rows.append({
            "seed": trace["seed"], "valid_readings": valid, "ll_truth": round(ll_truth, 4),
            "ll_grid_max": round(float(pos_ll.max()), 4),
            "truth_gap": round(float(pos_ll.max()) - ll_truth, 4),
            "plausible": int(ll_truth >= float(pos_ll.max()) - math.log(100)),
            "competitor_cells": int(competitors.sum()),
            "competitor_area_frac": round(float(competitors.sum() / len(px)), 5),
            "ambiguous": int(competitors.any()),
            "frac_cells_better_than_truth": round(float(np.mean(pos_ll > ll_truth)), 5),
        })
        maps[f"seed_{trace['seed']}"] = pos_ll.astype(np.float32)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_csv.with_name("likelihood_maps.npz"), px=px, py=py, **maps)
    S.write_csv_gz(rows, out_csv)
    S.write_metadata("global_diag", {"grid_cm": grid_cm, "heading_step_deg": heading_step_deg,
                                     "plausible_rule": "ll_truth >= max_grid_ll - ln(100)",
                                     "ambiguous_rule": "some cell >20 cm from truth has max-over-heading "
                                                       "ll >= ll_truth - ln(10)"})
    print(f"global_diag {len(rows)} traces ({time.time() - t0:.1f}s)", flush=True)


BATCHES = {"traces": batch_traces, "main": batch_main, "closed_loop": batch_closed_loop,
           "noise": batch_noise, "global_diag": batch_global_diag}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("batch", choices=[*BATCHES, "all"])
    args = p.parse_args()
    order = list(BATCHES) if args.batch == "all" else [args.batch]
    for name in order:
        t0 = time.time()
        BATCHES[name]()
        print(f"== {name} done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
