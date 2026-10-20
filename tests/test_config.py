import pytest


def test_single_parameter_set(cfg):
    # The values the tuned coursework code used, now defined once.
    assert (cfg.ports.left_motor, cfg.ports.right_motor, cfg.ports.sonar) == ("A", "C", 4)
    assert cfg.robot.velocity_cm_s == 9.17
    assert cfg.robot.wheel_base_cm == 15.25
    assert cfg.sonar.offset_cm == 7.0


def test_derived_speed_conversion_is_shared(cfg):
    # The DWA challenge used 9.17 / 200 cm/s per dps; it is now derived, not copied.
    assert cfg.robot.cm_s_per_dps == pytest.approx(9.17 / 200)


def test_dwa_samples_match_window(cfg):
    assert cfg.dwa.v_samples**2 == 169
