import math

import numpy as np
import pytest

from legobot.localisation.kld import kld_bound, kld_sample_count
from legobot.localisation.resampling import effective_sample_size, systematic_resample
from legobot.localisation.sensor_model import SonarModel


def test_likelihood_peaks_at_expected_range(cfg, world):
    model = SonarModel(world, cfg.sonar)
    particles = np.array([[100, 50, 0.0], [90, 50, 0.0], [100, 50, math.pi / 2]])
    lik = model.likelihood(particles, 110)  # wall g is 110 cm ahead of the first particle
    assert lik[0] == pytest.approx(1 + cfg.sonar.robust_constant)
    assert lik[1] < 0.01 and lik[2] < 0.01
    assert lik.min() >= cfg.sonar.robust_constant  # never zero


def test_untrusted_readings_are_skipped(cfg, world):
    model = SonarModel(world, cfg.sonar)
    assert not model.is_valid(None)
    assert not model.is_valid(cfg.sonar.max_valid_cm + 1)
    assert not model.is_valid(cfg.sonar.min_valid_cm - 1)
    assert model.is_valid(60)


def test_systematic_resample_counts_are_proportional(rng):
    weights = np.array([0.5, 0.25, 0.125, 0.125])
    idx = systematic_resample(weights, 800, rng)
    assert len(idx) == 800
    assert np.bincount(idx, minlength=4) == pytest.approx([400, 200, 100, 100], abs=1)


def test_systematic_resample_never_overflows(rng):
    # Round-off can leave the cumulative sum just below 1; the coursework
    # version could then index past the end of the particle list.
    weights = np.full(7, 1 / 7) * (1 - 1e-12)
    for _ in range(200):
        idx = systematic_resample(weights, 7, rng)
        assert idx.max() < 7


def test_effective_sample_size():
    assert effective_sample_size(np.full(10, 0.1)) == pytest.approx(10)
    assert effective_sample_size(np.array([1.0, 0, 0, 0])) == pytest.approx(1)


def test_kld_bound_grows_with_bins(cfg):
    k = cfg.mcl.kld
    bounds = [kld_bound(b, k.epsilon, k.z_quantile) for b in (1, 2, 5, 20, 100)]
    assert bounds[0] == 0
    assert all(a < b for a, b in zip(bounds, bounds[1:]))


def test_kld_sample_count_adapts(cfg, world, rng):
    k = cfg.mcl.kld
    tight = np.tile([100.0, 50.0, 0.0], (k.n_max, 1))
    assert kld_sample_count(tight, k) == k.n_min
    spread = world.sample_free_poses(k.n_max, rng)
    assert kld_sample_count(spread, k) == k.n_max
