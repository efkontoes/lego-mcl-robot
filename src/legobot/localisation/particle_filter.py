"""Monte Carlo Localisation (Dellaert, Fox, Burgard & Thrun, ICRA 1999).

:class:`ParticleFilter` is the coursework ``ParticleManager`` rewritten with
NumPy, plus three optional extensions taken from the literature (and from the
way ROS AMCL implements them) that the noise study compares:

``ess_gating``
    Only resample when the effective sample size drops below
    ``ess_threshold * N`` (Kong, Liu & Wong 1994; Liu & Chen 1998). Resampling
    every step, as the coursework did, throws away diversity even when the
    weights are still informative (particle depletion).
``injection``
    Augmented MCL (Thrun, Burgard & Fox 2005, Table 8.3; Gutmann & Fox 2002).
    Short- and long-term averages of the measurement likelihood, ``w_fast`` and
    ``w_slow``, are tracked; when recent measurements fit much worse than usual
    a fraction ``max(0, 1 - w_fast / w_slow)`` of the particles is replaced by
    random poses, so the filter can recover from divergence or kidnapping.
``kld``
    KLD-sampling (Fox 2001, 2003): the number of particles drawn at each
    resampling step adapts to how spread out the belief is.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from legobot.config import MclConfig
from legobot.localisation.kld import kld_sample_count
from legobot.localisation.motion_models import Motion
from legobot.localisation.resampling import (
    effective_sample_size,
    normalise,
    systematic_resample,
)
from legobot.localisation.sensor_model import SonarModel
from legobot.localisation.world_map import WorldMap


@dataclass(frozen=True)
class FilterOptions:
    ess_gating: bool = False
    injection: bool = False
    kld: bool = False


class ParticleFilter:
    def __init__(
        self,
        world: WorldMap,
        motion_model,
        sensor_model: SonarModel,
        cfg: MclConfig,
        options: Optional[FilterOptions] = None,
        rng: Optional[np.random.Generator] = None,
    ):
        self.world = world
        self.motion_model = motion_model
        self.sensor_model = sensor_model
        self.cfg = cfg
        self.options = options or FilterOptions()
        self.rng = rng if rng is not None else np.random.default_rng()
        self.w_slow = 0.0
        self.w_fast = 0.0
        self.resampled_last_update = False
        self.injected_last_update = 0
        self.last_ess = float("nan")  # N_eff right after weighting, before resampling
        self.init_at((0.0, 0.0, 0.0))

    # ------------------------------------------------------------------ init
    @property
    def num_particles(self) -> int:
        return len(self.particles)

    def init_at(self, pose: Tuple[float, float, float], n: Optional[int] = None) -> None:
        """All particles at a known pose (tracking)."""
        n = n or self.cfg.num_particles
        self.particles = np.tile(np.asarray(pose, dtype=float), (n, 1))
        self.weights = np.full(n, 1.0 / n)
        self._fresh = np.zeros(n, dtype=bool)

    def init_uniform(self, n: Optional[int] = None) -> None:
        """Particles spread uniformly over the free space (global localisation)."""
        if n is None:
            n = self.cfg.kld.n_max if self.options.kld else self.cfg.num_particles
        self.particles = self.world.sample_free_poses(n, self.rng)
        self.weights = np.full(n, 1.0 / n)
        self._fresh = np.zeros(n, dtype=bool)

    # ---------------------------------------------------------------- update
    def predict(self, motion: Motion) -> None:
        self.particles = self.motion_model.sample(self.particles, motion, self.rng)

    def update(self, z: Optional[float]) -> bool:
        """Weight by a (centre-corrected) sonar reading, then maybe resample.

        Returns ``False`` if the reading was outside the trusted range and the
        update was skipped.
        """
        self.resampled_last_update = False
        self.injected_last_update = 0
        if not self.sensor_model.is_valid(z):
            return False

        likelihood = self.sensor_model.likelihood(self.particles, z)
        if self.options.injection:
            # Particles injected at the previous step do not vote: they mostly
            # score near the likelihood floor, and counting them would lower
            # w_avg and trigger yet more injection (a runaway feedback loop).
            native = ~self._fresh if (~self._fresh).any() else np.ones_like(self._fresh)
            w_avg = float(np.dot(normalise(self.weights[native]), likelihood[native]))
            # Averages start at the first measurement instead of 0, otherwise
            # w_slow needs hundreds of updates before injection can trigger.
            if self.w_slow == 0.0:
                self.w_slow = self.w_fast = w_avg
            aug = self.cfg.augmented
            self.w_slow += aug.alpha_slow * (w_avg - self.w_slow)
            self.w_fast += aug.alpha_fast * (w_avg - self.w_fast)

        self.weights = normalise(self.weights * likelihood)
        self.last_ess = effective_sample_size(self.weights)
        self._resample()
        return True

    def _injection_probability(self) -> float:
        if not self.options.injection or self.w_slow <= 0:
            return 0.0
        ratio = self.w_fast / (self.cfg.augmented.ratio_threshold * self.w_slow)
        return max(0.0, 1.0 - ratio)

    def _resample(self) -> None:
        n = self.num_particles
        p_inject = self._injection_probability()
        self._fresh = np.zeros(n, dtype=bool)
        if (
            self.options.ess_gating
            and p_inject == 0.0
            and effective_sample_size(self.weights) >= self.cfg.ess_threshold * n
        ):
            return  # weights still diverse enough: keep them

        draw = self.cfg.kld.n_max if self.options.kld else n
        indices = systematic_resample(self.weights, draw, self.rng)
        if self.options.kld:
            # KLD-sampling consumes samples in random order (see kld.py).
            indices = self.rng.permutation(indices)
        candidates = self.particles[indices]

        replace = np.zeros(len(candidates), dtype=bool)
        if p_inject > 0.0:
            replace = self.rng.random(len(candidates)) < p_inject
            count = int(replace.sum())
            if count:
                candidates[replace] = self.world.sample_free_poses(count, self.rng)

        if self.options.kld:
            keep = kld_sample_count(candidates, self.cfg.kld)
            candidates, replace = candidates[:keep], replace[:keep]

        self._fresh = replace
        self.injected_last_update = int(replace.sum())

        self.particles = candidates
        self.weights = np.full(len(candidates), 1.0 / len(candidates))
        self.resampled_last_update = True

    # ------------------------------------------------------------ statistics
    def mean(self, mask: Optional[np.ndarray] = None) -> Tuple[float, float, float]:
        """Weighted mean pose (circular mean for the heading)."""
        p, w = self.particles, self.weights
        if mask is not None:
            p, w = p[mask], normalise(w[mask])
        theta = float(np.arctan2(np.dot(w, np.sin(p[:, 2])), np.dot(w, np.cos(p[:, 2]))))
        return float(np.dot(w, p[:, 0])), float(np.dot(w, p[:, 1])), theta

    def estimate(self) -> Tuple[float, float, float]:
        """Pose estimate of the main particle cluster.

        The weighted mean of *all* particles is dragged around by a minority
        of stray particles (e.g. freshly injected ones, or a second hypothesis).
        Like AMCL, which reports its highest-weight cluster, we take the mean
        of the particles within ``cluster_radius_cm`` of the weighted median.
        """
        cx = _weighted_median(self.particles[:, 0], self.weights)
        cy = _weighted_median(self.particles[:, 1], self.weights)
        near = np.hypot(self.particles[:, 0] - cx, self.particles[:, 1] - cy)
        mask = near <= self.cfg.cluster_radius_cm
        if self.weights[mask].sum() < 0.25:  # no dominant cluster: plain mean
            return self.mean()
        return self.mean(mask)

    def spread(self) -> float:
        """Root-mean-square distance of all particles from their mean (cm)."""
        x, y, _ = self.mean()
        sq = (self.particles[:, 0] - x) ** 2 + (self.particles[:, 1] - y) ** 2
        return float(np.sqrt(np.dot(self.weights, sq)))

    def effective_sample_size(self) -> float:
        return effective_sample_size(self.weights)

    def distinct_ratio(self) -> float:
        """Fraction of particles that are not duplicates (1 = no depletion)."""
        return len(np.unique(self.particles, axis=0)) / self.num_particles


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])
    return float(values[order][np.searchsorted(cumulative, 0.5 * cumulative[-1])])
