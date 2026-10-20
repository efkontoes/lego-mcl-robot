"""Simulated robot.

Keeps a *true* pose that moves with the noise levels in the ``simulation``
section of the config, which are independent of whatever a particle filter
assumes. Sensors are simulated against that true pose:

* sonar: ray cast to the nearest wall, Gaussian noise, lost echoes at steep
  incidence angles (reported as 255 like the NXT sensor) and random outliers;
* bumpers: wall closer than the robot footprint in front-left / front-right;
* camera: red cans rendered through the calibrated homography
  (:mod:`legobot.vision.synthetic`).
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import numpy as np

from legobot.config import Config
from legobot.geometry import world_to_robot, wrap_angle
from legobot.localisation.world_map import WorldMap

_SUBSTEP_S = 0.02
_BUMPER_ANGLE = 0.35  # rad either side of the heading


class SimBackend:
    def __init__(
        self,
        cfg: Config,
        world: Optional[WorldMap] = None,
        pose: Sequence[float] = (0.0, 0.0, 0.0),
        cans: Sequence[Tuple[float, float]] = (),
        rng: Optional[np.random.Generator] = None,
        noise: bool = True,
    ):
        self.cfg = cfg
        self.world = world
        self.rng = rng if rng is not None else np.random.default_rng()
        self.noise = noise
        self.pose = np.asarray(pose, dtype=float)
        self.cans = np.asarray(cans, dtype=float).reshape(-1, 2)
        self.time_s = 0.0
        self.trajectory: List[np.ndarray] = [self.pose.copy()]
        self.collisions = 0
        self._wheels = (0.0, 0.0)
        self._max_dps = cfg.robot.drive_dps
        self._homography = None

    # ---------------------------------------------------------------- helpers
    def _std(self, value: float) -> float:
        return float(self.rng.normal(0.0, value)) if self.noise and value > 0 else 0.0

    def _record(self) -> None:
        self.pose[2] = wrap_angle(float(self.pose[2]))
        self.trajectory.append(self.pose.copy())

    def _free_distance(self, heading: float) -> float:
        """How far the robot centre can travel along ``heading`` before its
        footprint touches a wall."""
        if self.world is None:
            return math.inf
        dist, _ = self.world.raycast(self.pose[0], self.pose[1], heading)
        return float(dist[0]) - self.cfg.robot.radius_cm

    def _hits_can(self) -> bool:
        if not len(self.cans):
            return False
        reach = self.cfg.robot.radius_cm + self.cfg.dwa.obstacle_radius_cm
        return bool(np.any(np.hypot(*(self.cans - self.pose[:2]).T) < reach))

    def teleport(self, pose: Sequence[float]) -> None:
        """Kidnap the robot (used by the noise study)."""
        self.pose = np.asarray(pose, dtype=float)
        self._record()

    # ---------------------------------------------------------------- motion
    def move_straight(self, distance_cm: float) -> None:
        s = self.cfg.simulation
        actual = distance_cm + self._std(s.trans_std_per_cm * abs(distance_cm))
        heading = self.pose[2] if actual >= 0 else self.pose[2] + math.pi
        free = self._free_distance(heading)
        if abs(actual) > free:  # stopped by a wall
            actual = math.copysign(max(free, 0.0), actual)
            self.collisions += 1
        self.pose[0] += actual * math.cos(self.pose[2])
        self.pose[1] += actual * math.sin(self.pose[2])
        self.pose[2] += self._std(s.drift_std_per_cm * abs(distance_cm))
        self.time_s += abs(distance_cm) / self.cfg.robot.velocity_cm_s
        self._record()

    def rotate(self, angle_rad: float) -> None:
        s = self.cfg.simulation
        r = self.cfg.robot
        self.pose[2] += angle_rad + self._std(s.rot_std_per_rad * abs(angle_rad) + s.rot_std_const)
        self.time_s += abs(angle_rad) * r.wheel_base_cm / 2 / r.velocity_cm_s
        self._record()

    def set_speed_limit(self, power: int, dps: float) -> None:
        self._max_dps = dps

    def set_wheel_speeds(self, left_dps: float, right_dps: float) -> None:
        clip = self._max_dps
        self._wheels = (float(np.clip(left_dps, -clip, clip)), float(np.clip(right_dps, -clip, clip)))

    def stop(self) -> None:
        self._wheels = (0.0, 0.0)

    def wait(self, seconds: float) -> None:
        """Integrate differential-drive kinematics for ``seconds``."""
        r = self.cfg.robot
        s = self.cfg.simulation
        steps = max(1, int(round(seconds / _SUBSTEP_S)))
        dt = seconds / steps
        # Each wheel slips by a random fraction for the duration of this command.
        slip_l = 1.0 + self._std(s.trans_std_per_cm)
        slip_r = 1.0 + self._std(s.trans_std_per_cm)
        vl = self._wheels[0] * r.cm_s_per_dps * slip_l
        vr = self._wheels[1] * r.cm_s_per_dps * slip_r
        for _ in range(steps):
            if vl == 0 and vr == 0:
                break
            v, w = (vl + vr) / 2, (vr - vl) / r.wheel_base_cm
            step = v * dt
            heading = self.pose[2] if step >= 0 else self.pose[2] + math.pi
            if abs(step) > self._free_distance(heading):
                self.collisions += 1
                step = 0.0
            self.pose[0] += step * math.cos(self.pose[2] + w * dt / 2)
            self.pose[1] += step * math.sin(self.pose[2] + w * dt / 2)
            self.pose[2] += w * dt
            if self._hits_can():
                self.collisions += 1
        self.time_s += seconds
        self._record()

    # --------------------------------------------------------------- sensors
    def read_sonar(self) -> Optional[float]:
        s = self.cfg.simulation
        son = self.cfg.sonar
        if self.world is None:
            return son.max_raw_cm
        dist, wall = self.world.raycast(self.pose[0], self.pose[1], self.pose[2])
        if not np.isfinite(dist[0]):
            return son.max_raw_cm
        incidence = self.world.incidence_angle(self.pose[2], wall)[0]
        if self.noise:
            if incidence > s.sonar_max_incidence_rad and self.rng.random() < s.sonar_lost_prob:
                return son.max_raw_cm
            if self.rng.random() < s.sonar_outlier_prob:
                return float(self.rng.integers(0, int(son.max_raw_cm) + 1))
        raw = dist[0] - son.offset_cm + self._std(son.sigma_cm)
        return float(np.clip(round(raw), 0, son.max_raw_cm))

    def read_touch(self) -> Tuple[bool, bool]:
        if self.world is None:
            return False, False
        reach = self.cfg.robot.radius_cm + 1.5
        hit = []
        for side in (+_BUMPER_ANGLE, -_BUMPER_ANGLE):
            dist, _ = self.world.raycast(self.pose[0], self.pose[1], self.pose[2] + side)
            hit.append(bool(dist[0] <= reach))
        return hit[0], hit[1]

    def capture(self) -> np.ndarray:
        from legobot.vision.homography import GroundHomography
        from legobot.vision.synthetic import render_cans

        if self._homography is None:
            self._homography = GroundHomography.from_config(self.cfg.camera)
        cans = world_to_robot(self.cans, self.pose) if len(self.cans) else np.empty((0, 2))
        half_fov = math.radians(self.cfg.simulation.camera_fov_deg) / 2
        visible = cans[np.abs(np.arctan2(cans[:, 1], cans[:, 0])) <= half_fov] if len(cans) else cans
        return render_cans(visible, self._homography, self.cfg.camera,
                           self.cfg.dwa.obstacle_radius_cm, self.cfg.simulation.can_height_cm)

    def close(self) -> None:
        self.stop()
