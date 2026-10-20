"""Incremental KLD-sampling (legobot.localisation.kld.kld_resample)."""

import numpy as np
import pytest

from legobot.localisation.kld import bin_codes, kld_bound, kld_resample


def _uniform(n, world, seed=0):
    return world.sample_free_poses(n, np.random.default_rng(seed))


def test_bound_is_zero_for_one_bin_and_grows_with_bins(cfg):
    k = cfg.mcl.kld
    assert kld_bound(0, k.epsilon, k.z_quantile) == 0
    assert kld_bound(1, k.epsilon, k.z_quantile) == 0
    bounds = kld_bound(np.array([2, 5, 20, 100]), k.epsilon, k.z_quantile)
    assert np.all(np.diff(bounds) > 0)
    assert bounds[1] == pytest.approx(kld_bound(5, k.epsilon, k.z_quantile))


def test_single_occupied_bin_stops_at_n_min(cfg):
    particles = np.tile([102.0, 52.0, 0.1], (100, 1))
    draw = kld_resample(particles, np.full(100, 0.01), cfg.mcl.kld, np.random.default_rng(1))
    assert len(draw.poses) == cfg.mcl.kld.n_min
    assert draw.bins == 1


def test_stops_at_first_count_satisfying_the_bound(cfg, world):
    k = cfg.mcl.kld
    particles = _uniform(300, world)
    # A ~10 cm cloud with headings in one or two 15-degree bins: a handful of
    # bins, so the bound lands between n_min and n_max (~150 samples).
    particles[:, :2] = 100 + (particles[:, :2] - 100) * 0.05
    particles[:, 2] = 0.15 + 0.1 * np.sin(particles[:, 2])
    draw = kld_resample(particles, np.full(300, 1 / 300), k, np.random.default_rng(3))
    n = len(draw.poses)
    assert k.n_min <= n < k.n_max
    codes = bin_codes(draw.poses, k)
    first_seen = np.zeros(n, dtype=bool)
    first_seen[np.unique(codes, return_index=True)[1]] = True
    k_prefix = np.cumsum(first_seen)
    counts = np.arange(1, n + 1)
    ok = (counts >= k.n_min) & (counts >= kld_bound(k_prefix, k.epsilon, k.z_quantile))
    assert ok[-1] and not ok[:-1].any()
    assert draw.bins == k_prefix[-1]


def test_dispersed_cloud_hits_n_max(cfg, world):
    k = cfg.mcl.kld
    particles = _uniform(2000, world)
    draw = kld_resample(particles, np.full(2000, 1 / 2000), k, np.random.default_rng(4))
    assert len(draw.poses) == k.n_max


def test_reproducible_for_a_seed(cfg, world):
    particles = _uniform(200, world)
    w = np.random.default_rng(9).random(200)
    a = kld_resample(particles, w, cfg.mcl.kld, np.random.default_rng(5), 0.2,
                     lambda c: _uniform(c, world, 11))
    b = kld_resample(particles, w, cfg.mcl.kld, np.random.default_rng(5), 0.2,
                     lambda c: _uniform(c, world, 11))
    assert np.array_equal(a.poses, b.poses) and np.array_equal(a.parents, b.parents)


def test_injected_particles_are_marked_and_drawn_from_free_space(cfg, world):
    particles = np.tile([102.0, 52.0, 0.1], (100, 1))
    rng = np.random.default_rng(6)
    draw = kld_resample(particles, np.full(100, 0.01), cfg.mcl.kld, rng, 1.0,
                        lambda c: world.sample_free_poses(c, rng))
    assert draw.injected.all() and (draw.parents == -1).all()
    assert world.contains(draw.poses[:, 0], draw.poses[:, 1]).all()
    none = kld_resample(particles, np.full(100, 0.01), cfg.mcl.kld, rng)
    assert not none.injected.any() and (none.parents >= 0).all()


def test_samples_follow_the_weights(cfg):
    particles = np.array([[50.0, 50.0, 0.0], [150.0, 50.0, 0.0]])
    draw = kld_resample(particles, np.array([0.8, 0.2]), cfg.mcl.kld, np.random.default_rng(7))
    share = np.mean(draw.parents == 0)
    assert share == pytest.approx(0.8, abs=0.12)
