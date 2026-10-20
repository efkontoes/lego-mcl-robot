"""Dynamic Window Approach (Fox, Burgard & Thrun 1997) for the practical 6
challenge: drive 3 m past red cans seen by the camera.

The planner is purely reactive: it only uses the obstacles visible in the
current frame (no memory). Each control cycle it

1. builds the *dynamic window* of wheel speeds reachable within ``dt`` given
   the acceleration limit,
2. forward-simulates every (v_left, v_right) pair for ``tau_s`` seconds,
3. scores each trajectory: progress to the goal minus penalties for
   entering the safety margin, reversing and stalling (a collision scores
   -1000),
4. sends the best pair to the motors.

All quantities are in cm, cm/s and radians.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np

from legobot.config import Config
from legobot.geometry import robot_to_world
from legobot.vision.detection import blobs_to_floor, detect_red_blobs
from legobot.vision.homography import GroundHomography

COLLISION_SCORE = -1000.0


def predict_pose(vl, vr, x, y, theta, dt, wheel_base):
    """Exact differential-drive motion for constant wheel speeds (vectorised)."""
    vl, vr = np.asarray(vl, float), np.asarray(vr, float)
    straight = np.abs(vr - vl) < 1e-9
    w = (vr - vl) / wheel_base
    safe_dv = np.where(straight, 1.0, vr - vl)
    radius = wheel_base / 2.0 * (vr + vl) / safe_dv
    dtheta = w * dt
    x_arc = x + radius * (np.sin(theta + dtheta) - np.sin(theta))
    y_arc = y - radius * (np.cos(theta + dtheta) - np.cos(theta))
    x_new = np.where(straight, x + vl * dt * np.cos(theta), x_arc)
    y_new = np.where(straight, y + vl * dt * np.sin(theta), y_arc)
    theta_new = np.where(straight, theta, theta + dtheta)
    return x_new, y_new, theta_new


def clearance(x, y, obstacles: np.ndarray, robot_radius: float, obstacle_radius: float):
    """Gap between the robot footprint and the nearest obstacle (vectorised over x, y)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(obstacles) == 0:
        return np.full(np.shape(x), np.inf)
    d = np.hypot(x[..., None] - obstacles[:, 0], y[..., None] - obstacles[:, 1])
    return d.min(axis=-1) - robot_radius - obstacle_radius


@dataclass
class Plan:
    v_left: float
    v_right: float
    score: float
    collisions: int
    candidates: int


class DwaPlanner:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.d = cfg.dwa
        self.wheel_base = cfg.robot.wheel_base_cm
        self.max_speed = cfg.robot.cm_s_per_dps * cfg.dwa.max_dps

    def score(self, vl, vr, pose, obstacles: np.ndarray):
        """Score arrays of candidate wheel speeds. Returns (score, collided)."""
        d = self.d
        x, y, theta = (np.full(np.shape(vl), v, dtype=float) for v in pose)
        min_clear = np.full(np.shape(vl), np.inf)
        dt_sim = d.tau_s / d.sim_steps
        for _ in range(d.sim_steps):
            x, y, theta = predict_pose(vl, vr, x, y, theta, dt_sim, self.wheel_base)
            min_clear = np.minimum(
                min_clear, clearance(x, y, obstacles, self.cfg.robot.radius_cm, d.obstacle_radius_cm)
            )
        collided = min_clear < 0

        gx, gy = d.goal_cm
        start_dist = math.hypot(gx - pose[0], gy - pose[1])
        progress = start_dist - np.hypot(gx - x, gy - y)
        forward = np.clip(progress / max(start_dist, 10.0), -0.5, 1.0)

        violation = np.clip((d.safe_dist_cm - min_clear) / d.safe_dist_cm, 0.0, None)
        obstacle = np.where(min_clear < d.safe_dist_cm, violation**2, 0.0)
        reversing = ((vl < 0) | (vr < 0)).astype(float)
        speed = np.hypot(vl, vr) / math.sqrt(2)
        stagnation = np.clip((d.stagnation_speed_cm_s - speed) / d.stagnation_speed_cm_s, 0.0, 1.0)

        w = d.weights
        total = (w.forward * forward - w.obstacle * obstacle - w.reverse * reversing
                 - w.stagnation * stagnation)
        return np.where(collided, COLLISION_SCORE, total), collided

    def plan(self, pose, current: Tuple[float, float], obstacles: np.ndarray) -> Plan:
        d = self.d
        step = d.max_accel_cm_s2 * d.dt_s
        windows = [
            np.linspace(max(-self.max_speed, v - step), min(self.max_speed, v + step), d.v_samples)
            for v in current
        ]
        vl, vr = np.meshgrid(*windows, indexing="ij")
        scores, collided = self.score(vl.ravel(), vr.ravel(), pose, obstacles)
        best = int(np.argmax(scores))
        return Plan(float(vl.ravel()[best]), float(vr.ravel()[best]), float(scores[best]),
                    int(collided.sum()), scores.size)


@dataclass
class ChallengeLog:
    poses: List[Tuple[float, float, float]] = field(default_factory=list)
    plans: List[Plan] = field(default_factory=list)
    obstacles: List[np.ndarray] = field(default_factory=list)


def run_challenge(robot, cfg: Config, max_cycles: int = 600, realtime: bool = True,
                  on_cycle: Optional[Callable[[int, np.ndarray, list, tuple, Plan], None]] = None
                  ) -> ChallengeLog:
    """Sense (camera) -> plan (DWA) -> act, until odometry says x >= finish line."""
    planner = DwaPlanner(cfg)
    homography = GroundHomography.from_config(cfg.camera)
    d = cfg.dwa
    robot.set_speed_limit(d.motor_power_limit, d.max_dps)
    pose = (0.0, 0.0, 0.0)
    v = (0.0, 0.0)
    log = ChallengeLog()
    for cycle in range(max_cycles):
        if pose[0] >= d.finish_x_cm:
            break
        t0 = time.time()
        frame = robot.camera_frame()
        blobs = detect_red_blobs(frame, cfg.camera)
        floor, kept = blobs_to_floor(blobs, homography, d.min_obstacle_cm, d.max_obstacle_cm)
        obstacles = robot_to_world(floor, pose) if len(floor) else np.empty((0, 2))

        plan = planner.plan(pose, v, obstacles)
        v = (plan.v_left, plan.v_right)
        robot.set_wheel_speeds(v[0] / cfg.robot.cm_s_per_dps, v[1] / cfg.robot.cm_s_per_dps)
        if on_cycle:
            on_cycle(cycle, frame, kept, pose, plan)

        log.poses.append(pose)
        log.plans.append(plan)
        log.obstacles.append(obstacles)
        robot.wait(max(0.0, d.dt_s - (time.time() - t0)) if realtime else d.dt_s)
        # Dead-reckoned pose from the commanded speeds (as in the coursework).
        pose = tuple(float(a) for a in predict_pose(v[0], v[1], *pose, d.dt_s, planner.wheel_base))
    robot.stop()
    log.poses.append(pose)
    return log
