"""Particle-filter noise study: compare the filter variants in simulation.

Scenarios (all in the practical 4 arena, single forward-facing sonar):

``tracking``  Known start, drive the practical 4 waypoint course (the
              coursework task). Measures how well each variant keeps track
              and how much its particle cloud spreads.
``kidnap``    Same course, but after waypoint 3 the robot is picked up and
              put down 30-50 cm away, rotated by up to 45 degrees. The robot
              does a sonar sweep at every waypoint afterwards. Measures
              recovery.
``global``    Unknown start (uniform particles, 1000 for fixed-size filters).
              The robot alternates sonar sweeps and short moves. Measures
              convergence.

Usage::

    python experiments/noise_study/run_benchmark.py            # 30 seeds
    python experiments/noise_study/run_benchmark.py --seeds 5  # quick run
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np

from legobot.config import Config, load_config
from legobot.hardware import SimBackend
from legobot.localisation import VARIANTS, arena, build_filter
from legobot.localisation.motion_models import Motion
from legobot.localisation.particle_filter import ParticleFilter
from legobot.navigation import navigate_to_waypoint, sonar_sweep
from legobot.robot import Robot

RESULTS = Path(__file__).resolve().parent / "results"
GLOBAL_PARTICLES = 1000
GLOBAL_CYCLES = 4
KIDNAP_AFTER_WAYPOINT = 2  # index -> after the 3rd waypoint
SUCCESS_CM = 10.0


@dataclass
class StepRecord:
    scenario: str
    variant: str
    seed: int
    step: int
    pos_error_cm: float
    heading_error_deg: float
    spread_cm: float
    ess_ratio: float
    distinct_ratio: float
    particles: int
    resampled: int
    injected: int
    update_ms: float


class Probe:
    """Wraps a filter's ``update`` to time it and log one record per update."""

    def __init__(self, scenario: str, variant: str, seed: int, pf: ParticleFilter, sim: SimBackend):
        self.records: List[StepRecord] = []
        self.meta = (scenario, variant, seed)
        self.pf, self.sim = pf, sim
        self._update = pf.update
        pf.update = self._timed_update

    def _timed_update(self, z):
        t0 = time.perf_counter()
        used = self._update(z)
        elapsed = (time.perf_counter() - t0) * 1000
        pf, sim = self.pf, self.sim
        # N_eff of the weights before resampling, relative to the particle count.
        ess = pf.last_ess / pf.num_particles if used else float("nan")
        x, y, th = pf.estimate()
        self.records.append(StepRecord(
            *self.meta, step=len(self.records),
            pos_error_cm=_finite(math.hypot(x - sim.pose[0], y - sim.pose[1])),
            heading_error_deg=_finite(abs(math.degrees(_wrap(th - sim.pose[2])))),
            spread_cm=_finite(pf.spread()), ess_ratio=ess,
            distinct_ratio=pf.distinct_ratio(), particles=pf.num_particles,
            resampled=int(pf.resampled_last_update),
            injected=pf.injected_last_update, update_ms=elapsed if used else 0.0,
        ))
        return used


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def _finite(v: float, cap: float = 1e6) -> float:
    return float(min(v, cap)) if math.isfinite(v) else cap


def _kidnap_pose(sim: SimBackend, rng: np.random.Generator) -> np.ndarray:
    world = sim.world
    while True:
        d, a = rng.uniform(30, 50), rng.uniform(-math.pi, math.pi)
        pose = sim.pose + [d * math.cos(a), d * math.sin(a), rng.uniform(-math.pi / 4, math.pi / 4)]
        clear = world.raycast(pose[0], pose[1], np.linspace(0, 2 * math.pi, 12))[0].min()
        if world.contains(pose[0], pose[1])[0] and clear > 10:
            return pose


def _setup(cfg: Config, variant: str, seed: int, start):
    world = arena()
    rng = np.random.default_rng(seed)
    sim = SimBackend(cfg, world, pose=start, rng=rng)
    pf = build_filter(cfg, variant, world, rng=np.random.default_rng(10_000 + seed))
    return world, rng, sim, pf, Robot(sim, cfg)


def run_tracking(cfg: Config, variant: str, seed: int):
    _, _, sim, pf, robot = _setup(cfg, variant, seed, cfg.navigation.start)
    pf.init_at(cfg.navigation.start)
    probe = Probe("tracking", variant, seed, pf, sim)
    reached = sum(navigate_to_waypoint(robot, pf, wp, cfg.navigation)
                  for wp in cfg.navigation.waypoints)
    return probe, {"waypoints_reached": reached, "wall_contacts": sim.collisions}


def run_kidnap(cfg: Config, variant: str, seed: int):
    _, rng, sim, pf, robot = _setup(cfg, variant, seed, cfg.navigation.start)
    pf.init_at(cfg.navigation.start)
    probe = Probe("kidnap", variant, seed, pf, sim)
    kidnap_step = None
    for i, wp in enumerate(cfg.navigation.waypoints):
        navigate_to_waypoint(robot, pf, wp, cfg.navigation)
        if i == KIDNAP_AFTER_WAYPOINT:
            sim.teleport(_kidnap_pose(sim, rng))
            kidnap_step = len(probe.records)
        if i >= KIDNAP_AFTER_WAYPOINT:
            sonar_sweep(robot, pf, cfg.navigation.sweep_steps)
    return probe, {"kidnap_step": kidnap_step, "wall_contacts": sim.collisions}


def run_global(cfg: Config, variant: str, seed: int):
    world, rng, sim, pf, robot = _setup(cfg, variant, seed, (0, 0, 0))
    sim.teleport(world.sample_free_poses(1, rng)[0])
    pf.init_uniform(None if VARIANTS[variant].options.kld else GLOBAL_PARTICLES)
    probe = Probe("global", variant, seed, pf, sim)
    for _ in range(GLOBAL_CYCLES):
        sonar_sweep(robot, pf, cfg.navigation.sweep_steps)
        raw = sim.read_sonar()
        if raw is not None and 30 < raw < cfg.sonar.max_raw_cm:
            d = min(cfg.navigation.max_step_cm, raw - 20)
            robot.forward(d)
            pf.predict(Motion.straight(d))
            pf.update(robot.sonar_from_centre())
    return probe, {"wall_contacts": sim.collisions}


SCENARIOS: Dict[str, Callable] = {"tracking": run_tracking, "kidnap": run_kidnap,
                                  "global": run_global}
SCENARIO_VARIANTS = {
    "tracking": list(VARIANTS),
    "kidnap": [k for k in VARIANTS if k != "V0-bug"],
    "global": [k for k in VARIANTS if k != "V0-bug"],
}


def summarise(scenario: str, variant: str, seed: int, records: List[StepRecord], extra: dict):
    errors = np.array([r.pos_error_cm for r in records])
    after = errors
    if scenario == "kidnap" and extra.get("kidnap_step") is not None:
        after = errors[extra["kidnap_step"]:]
    recovered_at = next((i for i in range(len(after)) if np.all(after[i:] < SUCCESS_CM)), None)
    used = [r.update_ms for r in records if r.update_ms > 0]
    return {
        "scenario": scenario, "variant": variant, "seed": seed, "updates": len(records),
        "final_error_cm": errors[-1] if len(errors) else np.nan,
        "mean_error_cm": float(np.mean(errors)) if len(errors) else np.nan,
        "final_spread_cm": records[-1].spread_cm if records else np.nan,
        "max_spread_cm": max((r.spread_cm for r in records), default=np.nan),
        "mean_ess_ratio": float(np.nanmean([r.ess_ratio for r in records])) if records else np.nan,
        "mean_distinct_ratio": float(np.mean([r.distinct_ratio for r in records]))
        if records else np.nan,
        "mean_particles": float(np.mean([r.particles for r in records])) if records else np.nan,
        "injected_total": sum(r.injected for r in records),
        "mean_update_ms": float(np.mean(used)) if used else np.nan,
        "success": bool(len(errors) and errors[-1] < SUCCESS_CM),
        "converged_after_updates": recovered_at,
        "resample_fraction": float(np.mean([r.resampled for r in records])) if records else np.nan,
        **extra,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--scenarios", nargs="+", default=list(SCENARIOS), choices=list(SCENARIOS))
    parser.add_argument("--out", type=Path, default=RESULTS)
    args = parser.parse_args()

    cfg = load_config()
    args.out.mkdir(parents=True, exist_ok=True)
    steps: List[StepRecord] = []
    summary: List[dict] = []
    for scenario in args.scenarios:
        for variant in SCENARIO_VARIANTS[scenario]:
            t0 = time.time()
            for seed in range(args.seeds):
                probe, extra = SCENARIOS[scenario](cfg, variant, seed)
                steps.extend(probe.records)
                summary.append(summarise(scenario, variant, seed, probe.records, extra))
            rows = [s for s in summary if s["scenario"] == scenario and s["variant"] == variant]
            print(f"{scenario:9s} {variant:7s} success {np.mean([r['success'] for r in rows]):5.0%}"
                  f"  median final error {np.median([r['final_error_cm'] for r in rows]):9.1f} cm"
                  f"  ({time.time() - t0:.1f}s)", flush=True)

    _write_csv(args.out / "steps.csv", [asdict(s) for s in steps])
    _write_csv(args.out / "runs.csv", summary)
    print(f"wrote {args.out / 'steps.csv'} and {args.out / 'runs.csv'}")


def _write_csv(path: Path, rows: List[dict]) -> None:
    with path.open("w", newline="") as f:
        fields = list(dict.fromkeys(k for row in rows for k in row))  # union, first-seen order
        writer = csv.DictWriter(f, fieldnames=fields, restval="")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
