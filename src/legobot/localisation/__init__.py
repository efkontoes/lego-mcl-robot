"""Monte Carlo Localisation and the filter variants compared in the noise study."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from legobot.config import Config
from legobot.localisation.motion_models import CourseworkModel, Motion, OdometryModel
from legobot.localisation.particle_filter import FilterOptions, ParticleFilter
from legobot.localisation.sensor_model import SonarModel
from legobot.localisation.world_map import WorldMap, arena


@dataclass(frozen=True)
class Variant:
    key: str
    label: str
    motion: str  # "coursework", "coursework-bug" or "odometry"
    options: FilterOptions


VARIANTS: Dict[str, Variant] = {
    v.key: v
    for v in (
        Variant("V0-bug", "Coursework + compounding-sigma bug", "coursework-bug", FilterOptions()),
        Variant("V0", "Coursework model", "coursework", FilterOptions()),
        Variant("V1", "Odometry model", "odometry", FilterOptions()),
        Variant("V2", "V1 + ESS-gated resampling", "odometry", FilterOptions(ess_gating=True)),
        Variant("V3", "V1 + augmented MCL injection", "odometry", FilterOptions(injection=True)),
        Variant("V4", "V1 + KLD-sampling", "odometry", FilterOptions(kld=True)),
        Variant("V5", "All combined", "odometry",
                FilterOptions(ess_gating=True, injection=True, kld=True)),
    )
}

DEFAULT_VARIANT = "V5"


def build_filter(cfg: Config, variant: str = DEFAULT_VARIANT, world: Optional[WorldMap] = None,
                 rng: Optional[np.random.Generator] = None) -> ParticleFilter:
    v = VARIANTS[variant]
    world = world or arena()
    if v.motion == "odometry":
        motion = OdometryModel(cfg.mcl.odometry_alphas)
    else:
        motion = CourseworkModel(cfg.mcl.coursework_model, compounding=v.motion == "coursework-bug")
    return ParticleFilter(world, motion, SonarModel(world, cfg.sonar), cfg.mcl, v.options, rng)


__all__ = ["VARIANTS", "DEFAULT_VARIANT", "Variant", "build_filter", "Motion", "ParticleFilter",
           "FilterOptions", "arena", "WorldMap"]
