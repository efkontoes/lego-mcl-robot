"""Statistics used by the calibration experiments (practicals 1 and 2)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


def error_covariance(xs: Sequence[float], ys: Sequence[float]) -> np.ndarray:
    """2x2 covariance of end-point errors (population covariance, as in the
    practical 1 hand calculation)."""
    return np.cov(np.vstack([xs, ys]), bias=True)


@dataclass(frozen=True)
class SonarStats:
    true_cm: Optional[float]
    mean_cm: Optional[float]
    std_cm: Optional[float]
    bias_cm: Optional[float]
    garbage_fraction: float


def sonar_stats(readings: Sequence[Optional[float]], true_cm: Optional[float] = None,
                min_valid: float = 5, max_valid: float = 250) -> SonarStats:
    """Mean, spread, bias and fraction of garbage readings at one distance."""
    valid = [r for r in readings if r is not None and min_valid <= r <= max_valid]
    garbage = 1 - len(valid) / len(readings) if readings else 1.0
    if not valid:
        return SonarStats(true_cm, None, None, None, garbage)
    mean = float(np.mean(valid))
    return SonarStats(true_cm, mean, float(np.std(valid)),
                      None if true_cm is None else mean - true_cm, garbage)
