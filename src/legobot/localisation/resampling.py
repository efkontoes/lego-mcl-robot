"""Resampling utilities.

* :func:`systematic_resample` - low-variance / systematic resampling (Kitagawa
  1996; *Probabilistic Robotics* Table 4.4). One random offset, N evenly spaced
  pointers, O(N).
* :func:`effective_sample_size` - N_eff = 1 / sum(w^2) (Kong, Liu & Wong 1994),
  used to decide whether resampling is needed at all.
"""

from __future__ import annotations

import numpy as np


def normalise(weights: np.ndarray) -> np.ndarray:
    total = weights.sum()
    if not np.isfinite(total) or total <= 0:
        return np.full(len(weights), 1.0 / len(weights))
    return weights / total


def effective_sample_size(weights: np.ndarray) -> float:
    w = normalise(weights)
    return float(1.0 / np.sum(w**2))


def systematic_resample(weights: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Return ``n`` particle indices drawn in proportion to ``weights``."""
    cumulative = np.cumsum(normalise(weights))
    cumulative[-1] = 1.0  # guards against round-off leaving the last bin short
    pointers = (rng.uniform(0.0, 1.0 / n) + np.arange(n) / n)
    return np.searchsorted(cumulative, pointers, side="right").clip(max=len(weights) - 1)
