"""Shared infrastructure for the revised noise study (v2).

Protocol A - fixed-trajectory replay
    A *reference trace* is generated once per (noise condition, scenario, seed)
    by driving the simulator with a controller that uses the **true** pose
    (no particle filter involved). The trace stores every commanded motion,
    every sonar reading (already corrected to the robot centre, ``None`` when
    the sensor returned nothing), the true pose and the simulated time after
    every event, and the kidnapping event. Every filter variant is then fed
    exactly the same event sequence, so differences between variants come
    from the estimators alone. Each filter's own sampling stays stochastic,
    seeded from (trace seed, variant).

Protocol B - closed-loop navigation
    The filter estimate drives the robot (``legobot.navigation``), so
    trajectories, motion noise and sensor readings diverge between variants.
    Every variant gets the same update budget. Used only for navigation
    reliability, never for estimator-accuracy claims.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from legobot.config import Config, load_config
from legobot.geometry import heading_and_distance
from legobot.hardware import SimBackend
from legobot.localisation import VARIANTS, arena, build_filter
from legobot.localisation.kld import occupied_bins
from legobot.localisation.motion_models import Motion
from legobot.localisation.particle_filter import STAGES
from legobot.navigation import navigate_to_waypoint
from legobot.robot import Robot

STUDY_VERSION = "2.0.0"
ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"

# --- frozen protocol constants (fixed before the evaluation runs) ------------
MAIN_SEEDS = range(1000, 1100)  # held-out: never used for development or tuning
NOISE_SEEDS = range(2000, 2030)  # noise-robustness grid, distinct from MAIN_SEEDS
SNAPSHOT_SEEDS = range(1000, 1010)  # particle snapshots saved for case studies
SUCCESS_CM = 10.0
SUSTAIN_UPDATES = 5  # sustained recovery: below SUCCESS_CM for this many consecutive updates
KIDNAP_AFTER_WAYPOINT = 2  # index -> robot moved after reaching the 3rd waypoint
GLOBAL_PARTICLES = 1000  # initial particle count for every variant in the global scenario
GLOBAL_CYCLES = 4
TRUTH_NEAR_CM, TRUTH_NEAR_RAD = 20.0, math.radians(30)  # "particle mass near the truth"
CONFIDENT_R95_CM = 20.0  # 95% credible radius at or below this = "confident"
WAYPOINT_ITERATIONS = 40  # closed loop: attempts per waypoint
CLOSED_LOOP_BUDGET = {"tracking": 150, "kidnap": 300}  # closed loop: total sonar updates
SNAPSHOT_MAX_PARTICLES = 300

MAIN_VARIANTS = {
    "tracking": ["V0", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V3-N500", "V3-N1000",
                 "V1-N1000"],
    "kidnap": ["V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V3-N500", "V3-N1000",
               "V1-N1000"],
    "global": ["V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8"],
}
SNAPSHOT_VARIANTS = ("V1", "V3", "V5", "V8", "V3-N1000")
CLOSED_LOOP_VARIANTS = ["V1", "V2", "V3", "V4", "V5", "V6", "V8", "V3-N1000"]
NOISE_VARIANTS = ["V1", "V2", "V5", "V8"]
MOTION_SCALES = (0.5, 1.0, 2.0)
OUTLIER_PROBS = (0.0, 0.03, 0.10)


# ============================================================ noise conditions
def condition_name(motion_scale: float = 1.0, outlier: Optional[float] = None) -> str:
    if motion_scale == 1.0 and outlier is None:
        return "nominal"
    return f"motion{motion_scale:g}_outlier{outlier:g}"


def condition_config(cfg: Config, motion_scale: float = 1.0,
                     outlier: Optional[float] = None) -> Config:
    """Simulator truth for a noise condition. Only the simulator section
    changes; the filters keep the nominal parameters."""
    s = cfg.simulation
    sim = replace(
        s,
        trans_std_per_cm=s.trans_std_per_cm * motion_scale,
        drift_std_per_cm=s.drift_std_per_cm * motion_scale,
        rot_std_per_rad=s.rot_std_per_rad * motion_scale,
        rot_std_const=s.rot_std_const * motion_scale,
        sonar_outlier_prob=s.sonar_outlier_prob if outlier is None else outlier,
    )
    return replace(cfg, simulation=sim)


# ======================================================== reference traces
class _Recorder:
    def __init__(self, sim: SimBackend, cfg: Config):
        self.sim, self.cfg = sim, cfg
        self.events: List[list] = []
        self.true: List[List[float]] = []
        self.times: List[float] = []
        self.kidnap: Optional[dict] = None

    def _log(self, event):
        self.events.append(event)
        self.true.append([round(float(v), 4) for v in self.sim.pose])
        self.times.append(round(self.sim.time_s, 4))

    def turn(self, angle: float):
        self.sim.rotate(angle)
        self._log(["turn", float(angle)])

    def move(self, dist: float):
        self.sim.move_straight(dist)
        self._log(["move", float(dist)])

    def measure(self):
        self.sim.wait(self.cfg.navigation.settle_s)
        raw = self.sim.read_sonar()
        z = None if raw is None else float(raw) + self.cfg.sonar.offset_cm
        self._log(["z", z])
        return raw

    def teleport(self, pose):
        before = self.sim.pose.copy()
        self.sim.teleport(pose)
        self.kidnap = {"event_index": len(self.events), "from": [float(v) for v in before],
                       "to": [float(v) for v in self.sim.pose],
                       "displacement_cm": float(math.hypot(*(self.sim.pose[:2] - before[:2])))}
        self._log(["kidnap", None])

    def sweep(self):
        angle = 2 * math.pi / self.cfg.navigation.sweep_steps
        for _ in range(self.cfg.navigation.sweep_steps):
            self.turn(angle)
            self.measure()


def _drive_to(rec: _Recorder, target, nav) -> None:
    """Turn-then-drive controller on the TRUE pose (independent of any filter)."""
    for _ in range(WAYPOINT_ITERATIONS):
        x, y, th = rec.sim.pose
        turn, dist = heading_and_distance(x, y, th, *target)
        if dist < nav.dist_tol_cm:
            return
        if abs(turn) > nav.ang_tol_rad:
            rec.turn(turn)
        else:
            rec.move(min(nav.max_step_cm, dist))
        rec.measure()


def kidnap_pose(world, pose, rng: np.random.Generator) -> np.ndarray:
    """30-50 cm displacement in a random direction, +/-45 deg rotation, at
    least 10 cm from any wall."""
    while True:
        d, a = rng.uniform(30, 50), rng.uniform(-math.pi, math.pi)
        new = np.asarray(pose) + [d * math.cos(a), d * math.sin(a), rng.uniform(-math.pi / 4, math.pi / 4)]
        clear = world.raycast(new[0], new[1], np.linspace(0, 2 * math.pi, 12))[0].min()
        if world.contains(new[0], new[1])[0] and clear > 10:
            return new


def generate_trace(cfg: Config, scenario: str, seed: int, condition: str = "nominal") -> dict:
    world = arena()
    rng = np.random.default_rng(seed)
    nav = cfg.navigation
    start = nav.start if scenario != "global" else tuple(world.sample_free_poses(1, rng)[0])
    sim = SimBackend(cfg, world, pose=start, rng=rng)
    rec = _Recorder(sim, cfg)
    if scenario in ("tracking", "kidnap"):
        for i, wp in enumerate(nav.waypoints):
            _drive_to(rec, wp, nav)
            if scenario == "kidnap" and i == KIDNAP_AFTER_WAYPOINT:
                rec.teleport(kidnap_pose(world, sim.pose, rng))
            if scenario == "kidnap" and i >= KIDNAP_AFTER_WAYPOINT:
                rec.sweep()
    elif scenario == "global":
        for _ in range(GLOBAL_CYCLES):
            rec.sweep()
            raw = sim.read_sonar()  # look ahead before deciding to move
            if raw is not None and 30 < raw < cfg.sonar.max_raw_cm:
                rec.move(min(nav.max_step_cm, raw - 20))
                rec.measure()
    else:
        raise ValueError(scenario)
    return {"scenario": scenario, "seed": seed, "condition": condition,
            "start": [float(v) for v in start], "events": rec.events, "true": rec.true,
            "time": rec.times, "kidnap": rec.kidnap, "wall_contacts": sim.collisions}


def trace_path(condition: str, scenario: str) -> Path:
    return RESULTS / "traces" / condition / f"{scenario}.json.gz"


def save_traces(traces: List[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        json.dump(traces, f, separators=(",", ":"))
    os.replace(tmp, path)


def load_traces(path: Path) -> List[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


# ================================================================ replay
def _variant_rng(seed: int, variant: str) -> np.random.Generator:
    return np.random.default_rng([seed, sorted(VARIANTS).index(variant), 7919])


def _snapshot(pf) -> np.ndarray:
    p = pf.particles
    if len(p) > SNAPSHOT_MAX_PARTICLES:
        idx = np.random.default_rng(0).choice(len(p), SNAPSHOT_MAX_PARTICLES, replace=False)
        p = p[idx]
    return p.astype(np.float16)


def replay(trace: dict, cfg: Config, variant: str, keep_snapshots: bool = False):
    """Feed one reference trace to one filter variant. Returns per-update rows
    (one per sonar event) and, optionally, particle snapshots."""
    world = arena()
    pf = build_filter(cfg, variant, world, rng=_variant_rng(trace["seed"], variant))
    if trace["scenario"] == "global":
        pf.init_uniform(GLOBAL_PARTICLES)
    else:
        pf.init_at(tuple(trace["start"]))
    kidnap_at = trace["kidnap"]["event_index"] if trace["kidnap"] else None
    rows, snaps = [], []
    predict_s = 0.0
    update_no = 0
    pf.reset_timings()
    for i, ((kind, value), truth) in enumerate(zip(trace["events"], trace["true"])):
        if kind == "turn":
            t0 = time.perf_counter()
            pf.predict(Motion.turn(value))
            predict_s += time.perf_counter() - t0
            continue
        if kind == "move":
            t0 = time.perf_counter()
            pf.predict(Motion.straight(value))
            predict_s += time.perf_counter() - t0
            continue
        if kind == "kidnap":
            continue
        n_before = pf.num_particles
        t0 = time.perf_counter()
        used = pf.update(value)
        update_s = time.perf_counter() - t0
        stages = pf.timings
        pf.reset_timings()
        x, y, th = pf.estimate()
        err = math.hypot(x - truth[0], y - truth[1])
        row = {
            "seed": trace["seed"], "variant": variant, "scenario": trace["scenario"],
            "condition": trace["condition"], "update": update_no, "event": i,
            "since_kidnap": (update_no - _updates_before(trace, kidnap_at)) if kidnap_at else "",
            "valid": int(used), "resampled": int(pf.resampled_last_update),
            "n_before": n_before, "n_after": pf.num_particles,
            "ess_ratio": round(pf.last_ess / n_before, 4) if used else "",
            "p_inject": round(pf.last_p_inject, 4), "injected": pf.injected_last_update,
            "true_x": truth[0], "true_y": truth[1], "est_x": round(x, 3), "est_y": round(y, 3),
            "pos_err": round(err, 4),
            "head_err_deg": round(abs(math.degrees((th - truth[2] + math.pi) % (2 * math.pi) - math.pi)), 3),
            "spread": round(pf.spread(), 4), "r95": round(pf.credible_radius(0.95), 4),
            "mass_near_truth": round(pf.mass_near(truth, TRUTH_NEAR_CM, TRUTH_NEAR_RAD), 4),
            "bins": occupied_bins(pf.particles, cfg.mcl.kld),
            "unique_pose_frac": round(pf.distinct_ratio(), 4),
            "lineage_frac": round(pf.lineage_ratio(), 4),
            "predict_ms": round(predict_s * 1000, 5), "update_ms": round(update_s * 1000, 5),
        }
        for stage in STAGES:
            if stage != "predict":
                row[f"t_{stage}_ms"] = round(stages.get(stage, 0.0) * 1000, 5)
        rows.append(row)
        if keep_snapshots:
            snaps.append(_snapshot(pf))
        predict_s = 0.0
        update_no += 1
    return rows, snaps


def _updates_before(trace: dict, kidnap_at: int) -> int:
    return sum(1 for kind, _ in trace["events"][:kidnap_at] if kind == "z")


# ================================================================ closed loop
class _UpdateLog:
    """Wraps ``pf.update`` to log every closed-loop update and enforce the budget."""

    def __init__(self, pf, sim, seed, variant, scenario, budget):
        self.pf, self.sim = pf, sim
        self.rows: List[dict] = []
        self.meta = (seed, variant, scenario)
        self.budget = budget
        self.kidnap_update: Optional[int] = None
        self._update = pf.update
        pf.update = self._logged

    @property
    def remaining(self) -> int:
        return self.budget - len(self.rows)

    def _logged(self, z):
        t0 = time.perf_counter()
        used = self._update(z)
        ms = (time.perf_counter() - t0) * 1000
        x, y, _ = self.pf.estimate()
        truth = self.sim.pose
        seed, variant, scenario = self.meta
        self.rows.append({
            "seed": seed, "variant": variant, "scenario": scenario, "update": len(self.rows),
            "valid": int(used), "resampled": int(self.pf.resampled_last_update),
            "n_after": self.pf.num_particles, "injected": self.pf.injected_last_update,
            "pos_err": round(math.hypot(x - truth[0], y - truth[1]), 4),
            "update_ms": round(ms, 5), "sim_time_s": round(self.sim.time_s, 3),
            "wall_contacts": self.sim.collisions,
        })
        return used


def run_closed_loop(cfg: Config, scenario: str, variant: str, seed: int):
    """Waypoint navigation driven by the filter estimate, with a fixed update budget."""
    world = arena()
    nav = cfg.navigation
    rng = np.random.default_rng(seed)
    sim = SimBackend(cfg, world, pose=nav.start, rng=rng)
    robot = Robot(sim, cfg)
    pf = build_filter(cfg, variant, world, rng=_variant_rng(seed, variant))
    pf.init_at(nav.start)
    log = _UpdateLog(pf, sim, seed, variant, scenario, CLOSED_LOOP_BUDGET[scenario])
    declared = verified = stalled = 0
    budget_exhausted = False
    for i, wp in enumerate(nav.waypoints):
        if log.remaining <= 0:
            budget_exhausted = True
            break
        limit = min(WAYPOINT_ITERATIONS, log.remaining)
        ok = navigate_to_waypoint(robot, pf, wp, nav, max_iterations=limit)
        robot.stop()
        declared += int(ok)
        verified += int(ok and math.hypot(sim.pose[0] - wp[0], sim.pose[1] - wp[1]) < SUCCESS_CM)
        stalled += int(not ok and limit == WAYPOINT_ITERATIONS)
        if scenario == "kidnap" and i == KIDNAP_AFTER_WAYPOINT:
            sim.teleport(kidnap_pose(world, sim.pose, rng))
            log.kidnap_update = len(log.rows)
        if scenario == "kidnap" and i >= KIDNAP_AFTER_WAYPOINT:
            angle = 2 * math.pi / nav.sweep_steps
            for _ in range(min(nav.sweep_steps, max(log.remaining, 0))):
                robot.turn(angle)
                pf.predict(Motion.turn(angle))
                robot.wait(nav.settle_s)
                pf.update(robot.sonar_from_centre())
    else:
        budget_exhausted = log.remaining <= 0
    x, y, _ = pf.estimate()
    summary = {
        "seed": seed, "variant": variant, "scenario": scenario,
        "waypoints_declared": declared, "waypoints_verified": verified,
        "waypoints_total": len(nav.waypoints), "stalled_waypoints": stalled,
        "budget_exhausted": int(budget_exhausted), "updates": len(log.rows),
        "resamples": sum(r["resampled"] for r in log.rows),
        "wall_contacts": sim.collisions, "sim_time_s": round(sim.time_s, 3),
        "final_err": round(math.hypot(x - sim.pose[0], y - sim.pose[1]), 4),
        "kidnap_update": "" if log.kidnap_update is None else log.kidnap_update,
        "cpu_ms_total": round(sum(r["update_ms"] for r in log.rows), 4),
    }
    for r in log.rows:
        r["since_kidnap"] = "" if log.kidnap_update is None else r["update"] - log.kidnap_update
    return summary, log.rows


# ================================================================ metrics
def recovery_stats(errors: Sequence[float], sustain: int = SUSTAIN_UPDATES,
                   threshold: float = SUCCESS_CM) -> Tuple[Optional[int], bool]:
    """First post-kidnap update index from which the error stays below
    ``threshold`` for ``sustain`` consecutive updates. Returns
    ``(index, observed)``; when never satisfied within the horizon the run is
    right-censored and ``index`` is the horizon length."""
    below = np.asarray(errors) < threshold
    run = 0
    for i, b in enumerate(below):
        run = run + 1 if b else 0
        if run >= sustain:
            return i - sustain + 1, True
    return len(below), False


def kaplan_meier(times: Sequence[float], observed: Sequence[bool]):
    """Kaplan-Meier estimate of P(not yet recovered by t). Returns step arrays
    (t, survival)."""
    t = np.asarray(times, dtype=float)
    e = np.asarray(observed, dtype=bool)
    grid = np.unique(t[e]) if e.any() else np.array([])
    surv, s = [1.0], 1.0
    for g in grid:
        at_risk = np.sum(t >= g)
        events = np.sum((t == g) & e)
        s *= 1 - events / at_risk if at_risk else 1
        surv.append(s)
    return np.concatenate([[0.0], grid]), np.array(surv)


def wilson(k: int, n: int, z: float = 1.959964) -> Tuple[float, float, float]:
    """Proportion with Wilson score interval."""
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return p, max(0.0, centre - half), min(1.0, centre + half)


def paired_bootstrap(a: np.ndarray, b: np.ndarray, stat=np.mean, n_boot: int = 10000,
                     seed: int = 12345) -> Tuple[float, float, float]:
    """Difference stat(a) - stat(b) for seed-paired arrays, with a 95%
    percentile bootstrap interval obtained by resampling seeds."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n_boot, len(a)))
    if stat is np.mean:
        diffs = a[idx].mean(axis=1) - b[idx].mean(axis=1)
    else:
        diffs = np.array([stat(a[i]) - stat(b[i]) for i in idx])
    return float(stat(a) - stat(b)), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


# ================================================================ metadata
def git_revision() -> Dict[str, str]:
    def run(*args):
        try:
            return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True,
                                  timeout=10).stdout.strip()
        except Exception:  # noqa: BLE001 - metadata must never break a run
            return "unknown"
    return {"commit": run("rev-parse", "HEAD"),
            "dirty": "yes" if run("status", "--porcelain") else "no"}


def environment() -> Dict[str, str]:
    return {"python": sys.version.split()[0], "numpy": np.__version__,
            "platform": platform.platform(), "processor": platform.processor(),
            "machine": platform.machine(), "cpu_count": str(os.cpu_count())}


def config_fingerprint(path: Optional[Path] = None) -> Dict[str, str]:
    from legobot.config import DEFAULT_CONFIG_PATH

    text = Path(path or DEFAULT_CONFIG_PATH).read_text(encoding="utf-8")
    return {"config_path": str(Path(path or DEFAULT_CONFIG_PATH).name),
            "config_sha256": hashlib.sha256(text.encode()).hexdigest(), "config_yaml": text}


def write_metadata(batch: str, extra: dict) -> Path:
    meta = {"study_version": STUDY_VERSION, "batch": batch,
            "command": " ".join([Path(sys.executable).name] + sys.argv),
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "git": git_revision(), "environment": environment(), **config_fingerprint(), **extra}
    path = RESULTS / "metadata" / f"{batch}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return path


# ================================================================ CSV helpers
def write_csv_gz(rows: Iterable[dict], path: Path) -> None:
    import csv

    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with gzip.open(tmp, "wt", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, restval="")
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


def read_csv_gz(paths: Iterable[Path]):
    import pandas as pd

    frames = [pd.read_csv(p) for p in sorted(paths)]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def nominal_config() -> Config:
    return load_config()
