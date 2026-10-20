"""Waypoint navigation driven by the particle-filter estimate (practicals 3-4)."""

from __future__ import annotations

import math
from typing import Callable, List, Optional, Sequence, Tuple

from legobot.config import NavigationConfig
from legobot.geometry import heading_and_distance
from legobot.localisation.motion_models import Motion
from legobot.localisation.particle_filter import ParticleFilter
from legobot.robot import Robot

UpdateCallback = Callable[[ParticleFilter], None]


def sonar_sweep(robot: Robot, pf: ParticleFilter, steps: int,
                on_update: Optional[UpdateCallback] = None) -> None:
    """Relocalisation manoeuvre: turn a full circle in ``steps`` increments,
    updating the filter after each one. Turns a single forward-facing sonar
    into a coarse range scan, which is far more informative than one beam."""
    angle = 2 * math.pi / steps
    for _ in range(steps):
        robot.turn(angle)
        pf.predict(Motion.turn(angle))
        pf.update(robot.sonar_from_centre())
        if on_update:
            on_update(pf)


def navigate_to_waypoint(
    robot: Robot,
    pf: ParticleFilter,
    target: Tuple[float, float],
    cfg: NavigationConfig,
    on_update: Optional[UpdateCallback] = None,
    max_iterations: int = 60,
) -> bool:
    """Turn-then-drive towards ``target`` using the filter's pose estimate,
    with a sonar measurement update after every motion. Returns True once
    the estimate is within ``dist_tol_cm`` of the target."""
    for _ in range(max_iterations):
        x, y, theta = pf.estimate()
        turn, dist = heading_and_distance(x, y, theta, *target)
        if dist < cfg.dist_tol_cm:
            return True

        if abs(turn) > cfg.ang_tol_rad:
            robot.turn(turn)
            pf.predict(Motion.turn(turn))
        else:
            step = min(cfg.max_step_cm, dist)
            robot.forward(step)
            pf.predict(Motion.straight(step))

        robot.wait(cfg.settle_s)
        pf.update(robot.sonar_from_centre())
        if on_update:
            on_update(pf)
    return False


def run_course(
    robot: Robot,
    pf: ParticleFilter,
    cfg: NavigationConfig,
    waypoints: Optional[Sequence[Tuple[float, float]]] = None,
    on_update: Optional[UpdateCallback] = None,
    on_waypoint: Optional[Callable[[int, bool], None]] = None,
) -> List[bool]:
    """Visit every waypoint in order; returns which ones were reached."""
    reached = []
    for i, wp in enumerate(waypoints or cfg.waypoints):
        ok = navigate_to_waypoint(robot, pf, wp, cfg, on_update)
        robot.stop()
        reached.append(ok)
        if on_waypoint:
            on_waypoint(i, ok)
    return reached
