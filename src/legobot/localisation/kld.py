"""KLD-sampling (Fox, NIPS 2001 / IJRR 2003).

The number of particles needed so that, with probability ``1 - delta``, the
Kullback-Leibler distance between the sample-based posterior and the true one
stays below ``epsilon`` depends on how many histogram bins ``k`` the particles
occupy::

    n = (k - 1) / (2 eps) * (1 - 2 / (9 (k - 1)) + sqrt(2 / (9 (k - 1))) * z)^3

When the robot is well localised the particles occupy few bins and the filter
can run with very few particles; when it is lost, more bins -> more particles.
This mirrors ``pf_resample_limit`` in ROS AMCL (``amcl/src/amcl/pf/pf.c``).

:func:`kld_resample` follows Fox's procedure: samples are drawn one at a time,
i.i.d. from the weighted particle set (or, with probability ``p_inject``, from
the uniform free-space distribution), the occupied-bin count ``k`` is updated
for every new sample and drawing stops at the first sample count ``n`` with
``n >= n_min`` and ``n >= bound(k)``, or at ``n_max``.

For speed the draws are generated in chunks of increasing size. Because the
samples are i.i.d., truncating a chunk at the first index that satisfies the
stopping rule yields exactly the distribution of the one-at-a-time procedure.
The result is deterministic for a given RNG state.
"""

from __future__ import annotations

import math
import time
from typing import Callable, Dict, NamedTuple, Optional

import numpy as np

from legobot.config import KldConfig

_BIN_OFFSET = 5_000  # bins are clipped to +/- this before encoding to one int64


def kld_bound(k, epsilon: float, z_quantile: float):
    """Required sample size for ``k`` occupied bins (Wilson-Hilferty approx.).

    Works on scalars and arrays. One occupied bin needs no samples beyond
    ``n_min``: the bound is 0 for ``k <= 1``.
    """
    k = np.asarray(k, dtype=float)
    km1 = np.maximum(k - 1.0, 1e-12)
    a = 2.0 / (9.0 * km1)
    bound = km1 / (2.0 * epsilon) * (1.0 - a + np.sqrt(a) * z_quantile) ** 3
    bound = np.where(k <= 1, 0.0, bound)
    return float(bound) if bound.ndim == 0 else bound


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


def bin_codes(poses: np.ndarray, cfg: KldConfig) -> np.ndarray:
    """One int64 per pose identifying its (x, y, theta) bin."""
    keys = bin_keys(poses, cfg)
    bx = np.clip(keys[:, 0], -_BIN_OFFSET + 1, _BIN_OFFSET - 1) + _BIN_OFFSET
    by = np.clip(keys[:, 1], -_BIN_OFFSET + 1, _BIN_OFFSET - 1) + _BIN_OFFSET
    return (bx * (2 * _BIN_OFFSET) + by) * 1000 + keys[:, 2]


def occupied_bins(poses: np.ndarray, cfg: KldConfig) -> int:
    return int(len(np.unique(bin_codes(poses, cfg))))


class KldDraw(NamedTuple):
    poses: np.ndarray  # (n, 3) new particle set
    parents: np.ndarray  # index into the old set, -1 for injected particles
    injected: np.ndarray  # bool mask
    bins: int  # occupied bins when sampling stopped


def kld_resample(
    particles: np.ndarray,
    weights: np.ndarray,
    cfg: KldConfig,
    rng: np.random.Generator,
    p_inject: float = 0.0,
    random_poses: Optional[Callable[[int], np.ndarray]] = None,
    timings: Optional[Dict[str, float]] = None,
) -> KldDraw:
    """Incremental KLD-sampling with optional random-particle injection."""
    cumulative = np.cumsum(weights / weights.sum())
    cumulative[-1] = 1.0
    last = len(weights) - 1
    seen = np.empty(0, dtype=np.int64)
    poses_out, parents_out, injected_out = [], [], []
    n, k = 0, 0
    chunk = max(cfg.n_min, 64)
    t_draw = t_inject = t_bins = 0.0
    while n < cfg.n_max:
        m = min(chunk, cfg.n_max - n)
        t0 = time.perf_counter()
        parents = np.searchsorted(cumulative, rng.random(m), side="right").clip(max=last)
        poses = particles[parents]
        t1 = time.perf_counter()
        injected = np.zeros(m, dtype=bool)
        if p_inject > 0.0:
            injected = rng.random(m) < p_inject
            count = int(injected.sum())
            if count:
                poses[injected] = random_poses(count)
                parents[injected] = -1
        t2 = time.perf_counter()
        codes = bin_codes(poses, cfg)
        uniq, first = np.unique(codes, return_index=True)
        unseen = ~np.isin(uniq, seen)
        is_new = np.zeros(m, dtype=bool)
        is_new[first[unseen]] = True
        k_running = k + np.cumsum(is_new)
        counts = n + np.arange(1, m + 1)
        done = (counts >= cfg.n_min) & (counts >= kld_bound(k_running, cfg.epsilon, cfg.z_quantile))
        finished = bool(done.any())
        stop = int(np.argmax(done)) + 1 if finished else m
        t3 = time.perf_counter()
        t_draw += t1 - t0
        t_inject += t2 - t1
        t_bins += t3 - t2
        poses_out.append(poses[:stop])
        parents_out.append(parents[:stop])
        injected_out.append(injected[:stop])
        n += stop
        k = int(k_running[stop - 1])
        if finished:
            break
        seen = np.concatenate([seen, uniq[unseen]])
        chunk *= 2
    if timings is not None:
        timings["resample_draw"] = timings.get("resample_draw", 0.0) + t_draw
        timings["injection"] = timings.get("injection", 0.0) + t_inject
        timings["kld_bins"] = timings.get("kld_bins", 0.0) + t_bins
    return KldDraw(np.concatenate(poses_out), np.concatenate(parents_out),
                   np.concatenate(injected_out), k)
