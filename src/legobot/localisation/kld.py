"""KLD-sampling (Fox, NIPS 2001 / IJRR 2003).

The number of particles needed so that, with probability ``1 - delta``, the
Kullback-Leibler distance between the sample-based posterior and the true one
stays below ``epsilon`` depends on how many histogram bins ``k`` the particles
occupy::

    n = (k - 1) / (2 eps) * (1 - 2 / (9 (k - 1)) + sqrt(2 / (9 (k - 1))) * z)^3

When the robot is well localised the particles occupy few bins and the filter
can run with very few particles; when it is lost, more bins -> more particles.
This mirrors ``pf_resample_limit`` in ROS AMCL (``amcl/src/amcl/pf/pf.c``).
"""

from __future__ import annotations

import math

import numpy as np

from legobot.config import KldConfig


def kld_bound(k: int, epsilon: float, z_quantile: float) -> float:
    """Required sample size for ``k`` occupied bins (Wilson-Hilferty approx.)."""
    if k <= 1:
        return 0.0
    a = 2.0 / (9.0 * (k - 1))
    return (k - 1) / (2.0 * epsilon) * (1.0 - a + math.sqrt(a) * z_quantile) ** 3


def bin_keys(poses: np.ndarray, cfg: KldConfig) -> np.ndarray:
    """Integer histogram bin of each pose in (x, y, theta)."""
    theta_bin = math.radians(cfg.bin_theta_deg)
    return np.column_stack(
        [
            np.floor(poses[:, 0] / cfg.bin_xy_cm),
            np.floor(poses[:, 1] / cfg.bin_xy_cm),
            np.floor(np.mod(poses[:, 2], 2 * math.pi) / theta_bin),
        ]
    ).astype(np.int64)


def kld_sample_count(poses: np.ndarray, cfg: KldConfig) -> int:
    """Smallest prefix length of ``poses`` (drawn in random order) that
    satisfies the KLD bound, clipped to [n_min, n_max].

    Equivalent to the original algorithm, which draws samples one at a time
    and stops once the number drawn exceeds the bound for the bins seen so far.
    """
    keys = bin_keys(poses[: cfg.n_max], cfg)
    seen = set()
    for i, key in enumerate(map(tuple, keys), start=1):
        seen.add(key)
        if i >= cfg.n_min and i >= kld_bound(len(seen), cfg.epsilon, cfg.z_quantile):
            return i
    return min(len(keys), cfg.n_max)
