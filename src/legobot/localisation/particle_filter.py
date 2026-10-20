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
    resampling step adapts to how spread out the belief is (incremental,
    see :func:`legobot.localisation.kld.kld_resample`).

Every update records per-stage CPU time in :attr:`ParticleFilter.timings`
(seconds, accumulated until :meth:`ParticleFilter.reset_timings`), and the
ancestry of the last ``LINEAGE_WINDOW`` resampling steps so that genealogical
diversity can be measured.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from legobot.config import MclConfig
from legobot.localisation.kld import kld_resample
from legobot.localisation.motion_models import Motion
from legobot.localisation.resampling import (
    effective_sample_size,
    normalise,
    systematic_resample,
)
from legobot.localisation.sensor_model import SonarModel
from legobot.localisation.world_map import WorldMap

LINEAGE_WINDOW = 5  # resampling steps over which ancestry is tracked
STAGES = ("predict", "likelihood", "weighting", "resample_draw", "injection", "kld_bins",
          "bookkeeping")


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
        self.last_p_inject = 0.0
        self.timings: Dict[str, float] = {}
        self.init_at((0.0, 0.0, 0.0))

    # ---------------------------------------------------------------- timing
    def reset_timings(self) -> None:
        self.timings = {}

    def _tick(self, stage: str, t0: float) -> float:
        now = time.perf_counter()
        self.timings[stage] = self.timings.get(stage, 0.0) + (now - t0)
        return now

    # ------------------------------------------------------------------ init
    @property
    def num_particles(self) -> int:
        return len(self.particles)

    def _reset_set(self, particles: np.ndarray) -> None:
        n = len(particles)
        self.particles = particles
        self.weights = np.full(n, 1.0 / n)
        self._fresh = np.zeros(n, dtype=bool)
        self._parents = deque(maxlen=LINEAGE_WINDOW)

    def init_at(self, pose: Tuple[float, float, float], n: Optional[int] = None) -> None:
        """All particles at a known pose (tracking)."""
        n = n or self.cfg.num_particles
        self._reset_set(np.tile(np.asarray(pose, dtype=float), (n, 1)))

    def init_uniform(self, n: Optional[int] = None) -> None:
        """Particles spread uniformly over the free space (global localisation)."""
        if n is None:
            n = self.cfg.kld.n_max if self.options.kld else self.cfg.num_particles
        self._reset_set(self.world.sample_free_poses(n, self.rng))

    # ---------------------------------------------------------------- update
    def predict(self, motion: Motion) -> None:
        t0 = time.perf_counter()
        self.particles = self.motion_model.sample(self.particles, motion, self.rng)
        self._tick("predict", t0)

    def update(self, z: Optional[float]) -> bool:
        """Weight by a (centre-corrected) sonar reading, then maybe resample.

        Returns ``False`` if the reading was outside the trusted range and the
        update was skipped.
        """
        self.resampled_last_update = False
        self.injected_last_update = 0
        self.last_p_inject = 0.0
        if not self.sensor_model.is_valid(z):
            return False

        t0 = time.perf_counter()
        likelihood = self.sensor_model.likelihood(self.particles, z)
        t0 = self._tick("likelihood", t0)
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
        self._tick("weighting", t0)
        self._resample()
        return True

    def _injection_probability(self) -> float:
        if not self.options.injection or self.w_slow <= 0:
            return 0.0
        ratio = self.w_fast / (self.cfg.augmented.ratio_threshold * self.w_slow)
        return max(0.0, 1.0 - ratio)

    def _resample(self) -> None:
        t0 = time.perf_counter()
        n = self.num_particles
        p_inject = self._injection_probability()
        self.last_p_inject = p_inject
        self._fresh = np.zeros(n, dtype=bool)
        if (
            self.options.ess_gating
            and p_inject == 0.0
            and self.last_ess >= self.cfg.ess_threshold * n
        ):
            self._tick("bookkeeping", t0)
            return  # weights still diverse enough: keep them

        if self.options.kld:
            self._tick("bookkeeping", t0)
            draw = kld_resample(self.particles, self.weights, self.cfg.kld, self.rng, p_inject,
                                lambda c: self.world.sample_free_poses(c, self.rng), self.timings)
            t0 = time.perf_counter()
            particles, parents, injected = draw.poses, draw.parents, draw.injected
        else:
            parents = systematic_resample(self.weights, n, self.rng)
            particles = self.particles[parents]
            t0 = self._tick("resample_draw", t0)
            injected = np.zeros(n, dtype=bool)
            if p_inject > 0.0:
                injected = self.rng.random(n) < p_inject
                count = int(injected.sum())
                if count:
                    particles[injected] = self.world.sample_free_poses(count, self.rng)
                    parents = np.where(injected, -1, parents)
            t0 = self._tick("injection", t0)

        self._fresh = injected
        self.injected_last_update = int(injected.sum())
        self._parents.append(parents)
        self.particles = particles
        self.weights = np.full(len(particles), 1.0 / len(particles))
        self.resampled_last_update = True
        self._tick("bookkeeping", t0)

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

    def credible_radius(self, level: float = 0.95) -> float:
        """Radius around :meth:`estimate` containing ``level`` of the particle
        weight (cm): the particle-derived credible circle."""
        x, y, _ = self.estimate()
        d = np.hypot(self.particles[:, 0] - x, self.particles[:, 1] - y)
        order = np.argsort(d)
        cumulative = np.cumsum(self.weights[order])
        return float(d[order][min(np.searchsorted(cumulative, level), len(d) - 1)])

    def mass_near(self, pose, radius_cm: float, angle_rad: Optional[float] = None) -> float:
        """Particle weight within ``radius_cm`` (and ``angle_rad``) of a pose."""
        near = np.hypot(self.particles[:, 0] - pose[0], self.particles[:, 1] - pose[1]) <= radius_cm
        if angle_rad is not None:
            dth = np.abs((self.particles[:, 2] - pose[2] + np.pi) % (2 * np.pi) - np.pi)
            near &= dth <= angle_rad
        return float(self.weights[near].sum())

    def effective_sample_size(self) -> float:
        return effective_sample_size(self.weights)

    def distinct_ratio(self) -> float:
        """Fraction of *current particle poses* that are unique.

        Duplicates created by resampling become numerically distinct again
        after the next noisy prediction, so this is a weak diversity measure;
        see :meth:`lineage_ratio` for ancestry-based diversity.
        """
        return len(np.unique(self.particles, axis=0)) / self.num_particles

    def lineage_ratio(self) -> float:
        """Distinct ancestors from up to ``LINEAGE_WINDOW`` resampling steps ago,
        divided by the particle count. Injected particles start new lineages.
        1.0 = no coalescence over the window; 1/N = a single common ancestor."""
        if not self._parents:
            return 1.0
        ancestors = np.arange(self.num_particles)
        new_lineage = np.zeros(self.num_particles, dtype=bool)
        for parents in reversed(self._parents):
            step = parents[np.maximum(ancestors, 0)]
            new_lineage |= step < 0
            ancestors = np.where(new_lineage, -1, step)
        distinct = len(np.unique(ancestors[~new_lineage])) + int(new_lineage.sum())
        return distinct / self.num_particles


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])
    return float(values[order][np.searchsorted(cumulative, 0.5 * cumulative[-1])])
