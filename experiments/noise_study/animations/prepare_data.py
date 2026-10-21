"""Extract compact animation data from the saved study results.

    python experiments/noise_study/animations/prepare_data.py

Runs in the project environment (it needs ``legobot`` and the saved results)
and writes JSON/NPZ files to ``animations/data/``. No simulation is run: the
data come from the saved traces, per-update logs, particle snapshots and
analysis tables. The one computation is the KLD sampler applied to two saved
particle clouds, to illustrate the algorithm.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import study_lib as S  # noqa: E402

from legobot.config import load_config  # noqa: E402
from legobot.localisation import arena  # noqa: E402
from legobot.localisation.kld import bin_codes, kld_bound, kld_resample  # noqa: E402
from legobot.localisation.sensor_model import SonarModel  # noqa: E402

DATA = HERE / "data"
KIDNAP_SEED = 1000  # V8 recovers, V1 does not (see results/analysis/main_runs.csv)
GLOBAL_SEED = 1000  # "hypothesis lost": the sweep singles out the truth, V1 never holds it
MAX_PARTICLES = 150  # per frame, enough to show the cloud's shape
REL_FLOOR = 30.0  # nats: the likelihood maps show log-likelihood relative to the best cell, down to -30


def _r(v, nd=2):
    return None if v is None or (isinstance(v, float) and not math.isfinite(v)) else round(float(v), nd)


def _snapshots(scenario, seed, variant):
    z = np.load(S.RESULTS / "raw" / "main" / "snapshots" / f"{scenario}_{seed}_{variant}.npz")
    out = []
    for i in range(len(z.files)):
        p = z[f"arr_{i}"].astype(float)
        if len(p) > MAX_PARTICLES:
            p = p[np.random.default_rng(i).choice(len(p), MAX_PARTICLES, replace=False)]
        out.append([[_r(x, 1), _r(y, 1)] for x, y, _ in p])
    return out


def _rows(scenario, seed, variants):
    d = S.read_csv_gz((S.RESULTS / "raw" / "main" / "nominal" / scenario).glob("seeds_*.csv.gz"))
    return d[(d.seed == seed) & d.variant.isin(variants)].sort_values(["variant", "update"])


def kidnap_replay():
    trace = next(t for t in S.load_traces(S.trace_path("nominal", "kidnap")) if t["seed"] == KIDNAP_SEED)
    rows = _rows("kidnap", KIDNAP_SEED, ["V1", "V8"])
    steps = []
    for _, r in rows[rows.variant == "V1"].iterrows():
        kind, z = trace["events"][int(r.event)]
        x, y, th = trace["true"][int(r.event)]
        steps.append({"true": [_r(x), _r(y), _r(th, 3)], "z": _r(z, 1), "valid": int(r.valid),
                      "since_kidnap": None if pd.isna(r.since_kidnap) else int(r.since_kidnap)})
    variants = {}
    for v in ("V1", "V8"):
        g = rows[rows.variant == v]
        variants[v] = {"est": [[_r(a), _r(b)] for a, b in zip(g.est_x, g.est_y)],
                       "err": [_r(e, 1) for e in g.pos_err], "r95": [_r(e, 1) for e in g.r95],
                       "n": [int(n) for n in g.n_after], "particles": _snapshots("kidnap", KIDNAP_SEED, v)}
    k = trace["kidnap"]
    out = {"seed": KIDNAP_SEED, "walls": [list(map(float, w)) for w in arena().walls],
           "kidnap": {"from": [_r(v) for v in k["from"]], "to": [_r(v) for v in k["to"]],
                      "displacement_cm": _r(k["displacement_cm"], 1)},
           "steps": steps, "variants": variants}
    (DATA / "kidnap_replay.json").write_text(json.dumps(out, separators=(",", ":")))


def budget():
    s = pd.read_csv(S.RESULTS / "analysis" / "main_summary.csv")
    k = s[s.scenario == "kidnap"]
    pts = [{"variant": r.variant, "label": r.label, "role": r.role, "recovered": _r(r.recovered, 3),
            "lo": _r(r.recovered_lo, 3), "hi": _r(r.recovered_hi, 3), "cpu": _r(r.cpu_ms, 1),
            "cpu_lo": _r(r.cpu_ms_lo, 1), "cpu_hi": _r(r.cpu_ms_hi, 1),
            "particles": _r(r.post_mean_particles, 0)} for r in k.itertuples()]
    (DATA / "budget.json").write_text(json.dumps({"points": pts}, separators=(",", ":")))


def kld_runs():
    cfg = load_config().mcl.kld
    world = arena()
    snaps = {"concentrated": _snapshots_raw("kidnap", KIDNAP_SEED, "V8", 5),
             "spread": _snapshots_raw("kidnap", KIDNAP_SEED, "V8", 40)}
    out = {"epsilon": cfg.epsilon, "z": cfg.z_quantile, "bin_cm": cfg.bin_xy_cm, "n_min": cfg.n_min,
           "n_max": cfg.n_max, "walls": [list(map(float, w)) for w in world.walls], "cases": {}}
    for name, particles in snaps.items():
        w = np.full(len(particles), 1.0 / len(particles))
        draw = kld_resample(particles, w, cfg, np.random.default_rng(7))
        codes = bin_codes(draw.poses, cfg)
        first = np.zeros(len(codes), dtype=bool)
        first[np.unique(codes, return_index=True)[1]] = True
        k_prefix = np.cumsum(first)
        bound = kld_bound(k_prefix, cfg.epsilon, cfg.z_quantile)
        out["cases"][name] = {
            "source": [[_r(x, 1), _r(y, 1)] for x, y, _ in particles],
            "draws": [[_r(x, 1), _r(y, 1)] for x, y, _ in draw.poses],
            "new_bin": first.astype(int).tolist(), "k": k_prefix.tolist(),
            "bound": [_r(b, 1) for b in bound], "n": int(len(draw.poses))}
    (DATA / "kld.json").write_text(json.dumps(out, separators=(",", ":")))


def _snapshots_raw(scenario, seed, variant, index):
    z = np.load(S.RESULTS / "raw" / "main" / "snapshots" / f"{scenario}_{seed}_{variant}.npz")
    return z[f"arr_{index}"].astype(float)


def global_sweep(grid_cm=2.5, heading_step_deg=5.0):
    trace = next(t for t in S.load_traces(S.trace_path("nominal", "global")) if t["seed"] == GLOBAL_SEED)
    cfg = load_config()
    world = arena()
    model = SonarModel(world, cfg.sonar)
    xs = np.arange(grid_cm / 2, 210, grid_cm)
    gx, gy = np.meshgrid(xs, xs)
    inside = world.contains(gx.ravel(), gy.ravel()).reshape(gx.shape)
    px, py = gx[inside], gy[inside]
    headings = np.radians(np.arange(0, 360, heading_step_deg))
    X, Y, TH = np.repeat(px, len(headings)), np.repeat(py, len(headings)), np.tile(headings, len(px))
    offset, readings = 0.0, []
    for kind, value in trace["events"]:
        if kind == "turn":
            offset += value
        elif kind == "z":
            readings.append((offset, value))
            if len(readings) == cfg.navigation.sweep_steps:
                break
    total = np.zeros(len(X))
    maps, ranges = [], []
    sx, sy, sth = trace["start"]
    for off, z in readings:
        if model.is_valid(z):
            pts = np.column_stack([X, Y, TH + off])
            total += np.log(model.likelihood(pts, z))
        pos = total.reshape(len(px), len(headings)).max(axis=1)
        img = np.full(gx.shape, np.nan)
        img[inside] = np.clip(pos - pos.max(), -REL_FLOOR, 0)
        maps.append(img.astype(np.float32))
        hit, _ = world.raycast(sx, sy, sth + off)
        ranges.append(float(hit[0]))
    rows = _rows("global", GLOBAL_SEED, ["V1"])
    snaps = _snapshots("global", GLOBAL_SEED, "V1")[: len(readings)]
    np.savez_compressed(
        DATA / "global_sweep.npz", maps=np.stack(maps), extent=np.array([0, 210, 0, 210]),
        start=np.array(trace["start"]), offsets=np.array([o for o, _ in readings]),
        readings=np.array([np.nan if z is None else z for _, z in readings], dtype=float),
        valid=np.array([int(model.is_valid(z)) for _, z in readings]), wall_range=np.array(ranges),
        walls=world.walls, particles=np.array([np.array(p + [[np.nan, np.nan]] * (MAX_PARTICLES - len(p)))
                                               for p in snaps], dtype=float),
        mass_near_truth=rows["mass_near_truth"].to_numpy()[: len(readings)])


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    kidnap_replay()
    budget()
    kld_runs()
    global_sweep()
    for p in sorted(DATA.iterdir()):
        print(f"{p.name:22s} {p.stat().st_size / 1024:7.1f} KB")


if __name__ == "__main__":
    main()
