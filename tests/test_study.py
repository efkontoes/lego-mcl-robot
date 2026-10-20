"""Protocol infrastructure of the noise study (experiments/noise_study/study_lib.py)."""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "noise_study"))
import study_lib as S  # noqa: E402

from legobot.localisation import FACTORIAL, VARIANTS, build_filter  # noqa: E402


def test_factorial_design_is_complete():
    combos = {tuple(vars(VARIANTS[v].options).values()) for v in FACTORIAL}
    assert len(combos) == 8
    assert VARIANTS["V3-N1000"].num_particles == 1000


def test_budget_control_uses_its_particle_count(cfg, world):
    pf = build_filter(cfg, "V3-N500", world, rng=np.random.default_rng(0))
    pf.init_at((84, 30, 0.0))
    assert pf.num_particles == 500


@pytest.fixture(scope="module")
def trace():
    return S.generate_trace(S.nominal_config(), "kidnap", 4242)


def test_trace_is_deterministic_and_filter_independent(trace):
    again = S.generate_trace(S.nominal_config(), "kidnap", 4242)
    assert again["events"] == trace["events"] and again["true"] == trace["true"]
    assert trace["kidnap"] and 30 <= trace["kidnap"]["displacement_cm"] <= 50


def test_every_variant_receives_identical_inputs(trace):
    cfg = S.nominal_config()
    a, _ = S.replay(trace, cfg, "V1")
    b, _ = S.replay(trace, cfg, "V8")
    assert [r["event"] for r in a] == [r["event"] for r in b]
    assert [(r["true_x"], r["true_y"]) for r in a] == [(r["true_x"], r["true_y"]) for r in b]
    assert [r["valid"] for r in a] == [r["valid"] for r in b]


def test_replay_is_reproducible(trace):
    cfg = S.nominal_config()
    a, _ = S.replay(trace, cfg, "V5")
    b, _ = S.replay(trace, cfg, "V5")
    assert [r["est_x"] for r in a] == [r["est_x"] for r in b]


def test_sustained_recovery_and_censoring():
    errs = [40, 8, 9, 30, 5, 4, 3, 2, 1, 50]
    assert S.recovery_stats(errs, sustain=5) == (4, True)
    assert S.recovery_stats([40, 8, 9, 30], sustain=5) == (4, False)  # censored at the horizon


def test_kaplan_meier_counts_censored_runs_at_risk():
    t, s = S.kaplan_meier([2, 5, 10, 10], [True, True, False, False])
    assert list(t) == [0, 2, 5]
    assert s == pytest.approx([1.0, 0.75, 0.5])


def test_wilson_interval():
    p, lo, hi = S.wilson(11, 30)
    assert p == pytest.approx(11 / 30)
    assert lo == pytest.approx(0.2187, abs=1e-3) and hi == pytest.approx(0.5449, abs=1e-3)
    assert S.wilson(0, 30)[1] == 0.0


def test_lineage_ratio_tracks_coalescence(cfg, world):
    pf = build_filter(cfg, "V1", world, rng=np.random.default_rng(0))
    pf.init_at((100, 50, 0.0))
    assert pf.lineage_ratio() == 1.0
    pf.weights = np.zeros(100)
    pf.weights[7] = 1.0
    pf.last_ess = 1.0
    pf._resample()
    assert pf.lineage_ratio() == pytest.approx(1 / 100)
    assert math.isclose(pf.credible_radius(), 0.0, abs_tol=1e-9)
