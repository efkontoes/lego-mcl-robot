import math
from dataclasses import replace

import numpy as np
import pytest

from legobot.localisation.motion_models import CourseworkModel, Motion, OdometryModel

N = 20_000


def _cloud(n=N):
    return np.zeros((n, 3))


def test_noise_free_models_are_exact(cfg, rng):
    quiet_odo = OdometryModel(replace(cfg.mcl.odometry_alphas, a1=0, a2=0, a3=0, a4=0))
    quiet_cw = CourseworkModel(replace(cfg.mcl.coursework_model, sigma_e=0, sigma_f=0, sigma_g=0))
    for model in (quiet_odo, quiet_cw):
        p = model.sample(_cloud(3), Motion.turn(math.pi / 2), rng)
        p = model.sample(p, Motion.straight(10), rng)
        assert p[:, :2] == pytest.approx(np.tile([0.0, 10.0], (3, 1)), abs=1e-9)
        assert p[:, 2] == pytest.approx(math.pi / 2)


def test_odometry_translation_noise_scales_with_distance(cfg, rng):
    model = OdometryModel(cfg.mcl.odometry_alphas)
    std10 = model.sample(_cloud(), Motion.straight(10), rng)[:, 0].std()
    std40 = model.sample(_cloud(), Motion.straight(40), rng)[:, 0].std()
    assert std40 / std10 == pytest.approx(4, rel=0.05)
    assert std10 == pytest.approx(math.sqrt(cfg.mcl.odometry_alphas.a3) * 10, rel=0.05)


def test_odometry_turn_noise_scales_with_angle(cfg, rng):
    model = OdometryModel(cfg.mcl.odometry_alphas)
    small = model.sample(_cloud(), Motion.turn(math.radians(10)), rng)[:, 2].std()
    large = model.sample(_cloud(), Motion.turn(math.radians(90)), rng)[:, 2].std()
    assert large / small == pytest.approx(9, rel=0.05)


def test_coursework_turn_noise_ignores_angle(cfg, rng):
    model = CourseworkModel(cfg.mcl.coursework_model)
    small = model.sample(_cloud(), Motion.turn(math.radians(10)), rng)[:, 2].std()
    large = model.sample(_cloud(), Motion.turn(math.radians(90)), rng)[:, 2].std()
    assert small == pytest.approx(large, rel=0.05)


def test_no_compounding_between_calls(cfg, rng):
    for model in (OdometryModel(cfg.mcl.odometry_alphas), CourseworkModel(cfg.mcl.coursework_model)):
        first = model.sample(_cloud(), Motion.straight(20), rng)[:, 0].std()
        for _ in range(10):
            model.sample(_cloud(10), Motion.straight(20), rng)
        later = model.sample(_cloud(), Motion.straight(20), rng)[:, 0].std()
        assert later == pytest.approx(first, rel=0.05)


def test_coursework_bug_compounds(cfg, rng):
    model = CourseworkModel(cfg.mcl.coursework_model, compounding=True)
    for _ in range(10):
        model.sample(_cloud(10), Motion.straight(20), rng)
    assert model.sigma_e == pytest.approx(cfg.mcl.coursework_model.sigma_e * 2**10)
