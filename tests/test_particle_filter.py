import math

import numpy as np
import pytest

from legobot.hardware import SimBackend
from legobot.localisation import VARIANTS, build_filter
from legobot.navigation import run_course
from legobot.robot import Robot


def _course(cfg, world, variant, seed):
    sim = SimBackend(cfg, world, pose=cfg.navigation.start, rng=np.random.default_rng(seed))
    pf = build_filter(cfg, variant, world, rng=np.random.default_rng(seed + 1))
    pf.init_at(cfg.navigation.start)
    reached = run_course(Robot(sim, cfg), pf, cfg.navigation)
    x, y, _ = pf.estimate()
    return reached, math.hypot(x - sim.pose[0], y - sim.pose[1])


@pytest.mark.parametrize("variant", [v for v in VARIANTS if v != "V0-bug"])
def test_every_variant_tracks_the_course(cfg, world, variant):
    reached, error = _course(cfg, world, variant, seed=7)
    assert all(reached)
    assert error < 10


def test_coursework_bug_diverges(cfg, world):
    reached, error = _course(cfg, world, "V0-bug", seed=7)
    assert not all(reached) or error > 100


def test_ess_gating_skips_resampling_when_weights_are_diverse(cfg, world):
    pf = build_filter(cfg, "V2", world, rng=np.random.default_rng(0))
    pf.init_at((100, 50, 0.0))
    pf.update(110)  # every particle agrees: weights stay uniform
    assert not pf.resampled_last_update
    pf.particles[:70, 0] = 60  # most particles now expect 150 cm: N_eff ~ 30 < N/2
    pf.update(110)
    assert pf.resampled_last_update


def test_injection_triggers_when_measurements_stop_fitting(cfg, world):
    pf = build_filter(cfg, "V3", world, rng=np.random.default_rng(0))
    pf.init_at((100, 50, 0.0))
    for _ in range(5):
        pf.update(110)  # consistent readings
    assert pf.injected_last_update == 0
    injected = 0
    for _ in range(10):
        pf.update(40)  # robot was moved: readings no longer fit
        injected += pf.injected_last_update
    assert injected > 0


def test_kld_shrinks_a_concentrated_cloud(cfg, world):
    pf = build_filter(cfg, "V4", world, rng=np.random.default_rng(0))
    pf.init_at((100, 50, 0.0), n=cfg.mcl.kld.n_max)
    pf.particles[:, 0] += np.linspace(0, 1, cfg.mcl.kld.n_max)  # break the uniform weights tie
    pf.update(109.5)
    assert pf.num_particles == cfg.mcl.kld.n_min


def test_estimate_ignores_stray_minority(cfg, world):
    pf = build_filter(cfg, "V1", world, rng=np.random.default_rng(0))
    pf.init_at((100, 50, 0.0))
    pf.particles[:15, :2] = [20, 150]  # 15% of particles far away
    x, y, _ = pf.estimate()
    assert (x, y) == pytest.approx((100, 50))
    mx, _, _ = pf.mean()
    assert mx < 90  # the plain mean is dragged away
