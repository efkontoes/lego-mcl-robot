"""Load ``config/robot.yaml`` into typed, immutable dataclasses.

Every module receives its parameters from here, so a value (a port, a wheel
calibration, a noise level) is defined exactly once.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "robot.yaml"

Pose = Tuple[float, float, float]


@dataclass(frozen=True)
class RobotConfig:
    degrees_per_cm: float
    velocity_cm_s: float
    wheel_base_cm: float
    radius_cm: float
    drive_dps: float
    motor_power_limit: int
    turn_offset_s: float
    forward_factor: float

    @property
    def cm_s_per_dps(self) -> float:
        """Linear wheel speed produced by one degree-per-second of motor speed."""
        return self.velocity_cm_s / self.drive_dps


@dataclass(frozen=True)
class PortConfig:
    left_motor: str
    right_motor: str
    sonar: int
    left_touch: int
    right_touch: int


@dataclass(frozen=True)
class SonarConfig:
    offset_cm: float
    sigma_cm: float
    min_valid_cm: float
    max_valid_cm: float
    robust_constant: float
    max_raw_cm: float


@dataclass(frozen=True)
class CourseworkNoise:
    sigma_e: float
    sigma_f: float
    sigma_g: float


@dataclass(frozen=True)
class OdometryAlphas:
    a1: float
    a2: float
    a3: float
    a4: float


@dataclass(frozen=True)
class AugmentedConfig:
    alpha_slow: float
    alpha_fast: float
    ratio_threshold: float


@dataclass(frozen=True)
class KldConfig:
    epsilon: float
    z_quantile: float
    bin_xy_cm: float
    bin_theta_deg: float
    n_min: int
    n_max: int


@dataclass(frozen=True)
class MclConfig:
    num_particles: int
    coursework_model: CourseworkNoise
    odometry_alphas: OdometryAlphas
    ess_threshold: float
    augmented: AugmentedConfig
    cluster_radius_cm: float
    kld: KldConfig


@dataclass(frozen=True)
class BehaviourConfig:
    square_side_cm: float
    bumper_reverse_cm: float
    bumper_step_cm: float
    wall_follow_target_cm: float
    wall_follow_kp: float
    wall_follow_window: int
    wall_follow_period_s: float


@dataclass(frozen=True)
class NavigationConfig:
    start: Pose
    waypoints: Tuple[Tuple[float, float], ...]
    dist_tol_cm: float
    ang_tol_rad: float
    max_step_cm: float
    settle_s: float
    sweep_steps: int


@dataclass(frozen=True)
class CameraConfig:
    width: int
    height: int
    homography_points: Tuple[Tuple[float, float, float, float], ...]
    red_hsv_ranges: Tuple[Tuple[Tuple[int, int, int], Tuple[int, int, int]], ...]
    min_blob_area_px: int


@dataclass(frozen=True)
class DwaWeights:
    forward: float
    obstacle: float
    reverse: float
    stagnation: float


@dataclass(frozen=True)
class DwaConfig:
    max_dps: float
    motor_power_limit: int
    max_accel_cm_s2: float
    obstacle_radius_cm: float
    safe_dist_cm: float
    tau_s: float
    dt_s: float
    v_samples: int
    sim_steps: int
    weights: DwaWeights
    stagnation_speed_cm_s: float
    goal_cm: Tuple[float, float]
    finish_x_cm: float
    min_obstacle_cm: float
    max_obstacle_cm: float


@dataclass(frozen=True)
class SimulationConfig:
    trans_std_per_cm: float
    drift_std_per_cm: float
    rot_std_per_rad: float
    rot_std_const: float
    sonar_outlier_prob: float
    sonar_max_incidence_deg: float
    sonar_lost_prob: float
    camera_fov_deg: float
    can_height_cm: float

    @property
    def sonar_max_incidence_rad(self) -> float:
        return math.radians(self.sonar_max_incidence_deg)


@dataclass(frozen=True)
class Config:
    robot: RobotConfig
    ports: PortConfig
    sonar: SonarConfig
    mcl: MclConfig
    behaviours: BehaviourConfig
    navigation: NavigationConfig
    camera: CameraConfig
    dwa: DwaConfig
    simulation: SimulationConfig


def _tuples(rows):
    return tuple(tuple(r) for r in rows)


def load_config(path: Optional[Path] = None) -> Config:
    """Read the YAML config (defaults to ``config/robot.yaml`` in the repo)."""
    raw = yaml.safe_load(Path(path or DEFAULT_CONFIG_PATH).read_text())

    r = raw["robot"]
    profile = r["wheel_profiles"][r["wheels"]]
    robot = RobotConfig(
        degrees_per_cm=profile["degrees_per_cm"],
        velocity_cm_s=profile["velocity_cm_s"],
        wheel_base_cm=r["wheel_base_cm"],
        radius_cm=r["radius_cm"],
        drive_dps=r["drive_dps"],
        motor_power_limit=r["motor_power_limit"],
        turn_offset_s=r["turn_offset_s"],
        forward_factor=r["forward_factor"],
    )

    m = raw["mcl"]
    mcl = MclConfig(
        num_particles=m["num_particles"],
        coursework_model=CourseworkNoise(**m["coursework_model"]),
        odometry_alphas=OdometryAlphas(**m["odometry_alphas"]),
        ess_threshold=m["ess_threshold"],
        augmented=AugmentedConfig(**m["augmented"]),
        cluster_radius_cm=m["cluster_radius_cm"],
        kld=KldConfig(**m["kld"]),
    )

    n = raw["navigation"]
    navigation = NavigationConfig(
        start=tuple(n["start"]),
        waypoints=_tuples(n["waypoints"]),
        dist_tol_cm=n["dist_tol_cm"],
        ang_tol_rad=n["ang_tol_rad"],
        max_step_cm=n["max_step_cm"],
        settle_s=n["settle_s"],
        sweep_steps=n["sweep_steps"],
    )

    c = raw["camera"]
    camera = CameraConfig(
        width=c["width"],
        height=c["height"],
        homography_points=_tuples(c["homography_points"]),
        red_hsv_ranges=tuple((tuple(lo), tuple(hi)) for lo, hi in c["red_hsv_ranges"]),
        min_blob_area_px=c["min_blob_area_px"],
    )

    d = dict(raw["dwa"])
    d["weights"] = DwaWeights(**d["weights"])
    d["goal_cm"] = tuple(d["goal_cm"])
    dwa = DwaConfig(**d)

    return Config(
        robot=robot,
        ports=PortConfig(**raw["ports"]),
        sonar=SonarConfig(**raw["sonar"]),
        mcl=mcl,
        behaviours=BehaviourConfig(**raw["behaviours"]),
        navigation=navigation,
        camera=camera,
        dwa=dwa,
        simulation=SimulationConfig(**raw["simulation"]),
    )
