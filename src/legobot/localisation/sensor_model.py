"""Sonar measurement model: p(z | particle pose, map)."""

from __future__ import annotations

from typing import Optional

import numpy as np

from legobot.config import SonarConfig
from legobot.localisation.world_map import WorldMap


class SonarModel:
    """Gaussian around the ray-cast wall distance, plus a constant floor.

    ``z`` is the reading already corrected to the robot centre (raw + offset).
    The constant floor (``robust_constant``) means a single garbage reading
    cannot drive a correct particle's weight to zero. Readings outside the
    trusted range are ignored (the caller gets ``None`` and skips the update).
    """

    def __init__(self, world: WorldMap, cfg: SonarConfig):
        self.world = world
        self.cfg = cfg

    def is_valid(self, z: Optional[float]) -> bool:
        return z is not None and self.cfg.min_valid_cm <= z <= self.cfg.max_valid_cm

    def likelihood(self, particles: np.ndarray, z: float) -> np.ndarray:
        expected, _ = self.world.raycast(particles[:, 0], particles[:, 1], particles[:, 2])
        with np.errstate(invalid="ignore"):
            gauss = np.exp(-((z - expected) ** 2) / (2 * self.cfg.sigma_cm**2))
        gauss = np.where(np.isfinite(expected), gauss, 0.0)
        return gauss + self.cfg.robust_constant
