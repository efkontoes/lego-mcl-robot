"""Monte Carlo Localisation and the filter variants compared in the noise study.

The study is a 2 x 2 x 2 factorial design over three techniques on top of the
odometry motion model (V1): ESS-gated resampling, augmented-MCL injection and
KLD-sampling (V1-V8). V0 and V0-bug are historical motion-model controls and
the ``-N500`` / ``-N1000`` variants are fixed particle-budget controls.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
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
    num_particles: Optional[int] = None  # fixed-budget override of mcl.num_particles
    role: str = "factorial"  # "factorial", "budget-control" or "historical"


def _odo(key, label, ess=False, inj=False, kld=False, n=None, role="factorial"):
    return Variant(key, label, "odometry", FilterOptions(ess, inj, kld), n, role)


VARIANTS: Dict[str, Variant] = {
    v.key: v
    for v in (
        Variant("V0-bug", "Coursework + compounding-sigma bug", "coursework-bug", FilterOptions(),
                role="historical"),
        Variant("V0", "Coursework model", "coursework", FilterOptions(), role="historical"),
        _odo("V1", "Odometry model"),
        _odo("V2", "V1 + ESS gating", ess=True),
        _odo("V3", "V1 + injection", inj=True),
        _odo("V4", "V1 + KLD", kld=True),
        _odo("V5", "ESS + injection + KLD", ess=True, inj=True, kld=True),
        _odo("V6", "ESS + injection", ess=True, inj=True),
        _odo("V7", "ESS + KLD", ess=True, kld=True),
        _odo("V8", "Injection + KLD", inj=True, kld=True),
        _odo("V3-N500", "Injection, fixed 500 particles", inj=True, n=500, role="budget-control"),
        _odo("V3-N1000", "Injection, fixed 1000 particles", inj=True, n=1000,
             role="budget-control"),
        _odo("V1-N1000", "Odometry model, fixed 1000 particles", n=1000, role="budget-control"),
    )
}

FACTORIAL = ("V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8")
DEFAULT_VARIANT = "V2"  # see docs/noise_study.md section 12


def build_filter(cfg: Config, variant: str = DEFAULT_VARIANT, world: Optional[WorldMap] = None,
                 rng: Optional[np.random.Generator] = None) -> ParticleFilter:
    v = VARIANTS[variant]
    world = world or arena()
    if v.motion == "odometry":
        motion = OdometryModel(cfg.mcl.odometry_alphas)
    else:
        motion = CourseworkModel(cfg.mcl.coursework_model, compounding=v.motion == "coursework-bug")
    mcl = replace(cfg.mcl, num_particles=v.num_particles) if v.num_particles else cfg.mcl
    return ParticleFilter(world, motion, SonarModel(world, cfg.sonar), mcl, v.options, rng)


__all__ = ["VARIANTS", "FACTORIAL", "DEFAULT_VARIANT", "Variant", "build_filter", "Motion",
           "ParticleFilter", "FilterOptions", "arena", "WorldMap"]
