import math

import numpy as np
import pytest

from legobot.hardware import SimBackend
from legobot.planning.dwa import COLLISION_SCORE, DwaPlanner, predict_pose
from legobot.vision.detection import blobs_to_floor, detect_red_blobs
from legobot.vision.homography import GroundHomography

cv2 = pytest.importorskip("cv2")


def test_homography_round_trip(cfg):
    h = GroundHomography.from_config(cfg.camera)
    for x, y, u, v in cfg.camera.homography_points:
        pu, pv = h.to_pixel(x, y)
        assert (pu[0], pv[0]) == pytest.approx((u, v), abs=1e-6)
        gx, gy = h.to_ground(u, v)
        assert (gx[0], gy[0]) == pytest.approx((x, y), abs=1e-6)


def test_rendered_can_is_detected_where_it_stands(cfg):
    sim = SimBackend(cfg, cans=[(50, 8)])
    frame = sim.capture()
    blobs = detect_red_blobs(frame, cfg.camera)
    floor, _ = blobs_to_floor(blobs, GroundHomography.from_config(cfg.camera), 0, 200)
    assert len(floor) == 1
    # The blob's bottom is the front edge of the can: x - radius.
    assert floor[0] == pytest.approx((50 - cfg.dwa.obstacle_radius_cm, 8), abs=1.5)


def test_predict_pose_straight_arc_and_spin(cfg):
    wb = cfg.robot.wheel_base_cm
    x, y, th = predict_pose(10, 10, 0, 0, 0, 2, wb)
    assert (float(x), float(y), float(th)) == pytest.approx((20, 0, 0))
    _, _, th = predict_pose(-5, 5, 0, 0, 0, 1, wb)  # spin on the spot
    assert float(th) == pytest.approx(10 / wb)
    # Quarter circle of radius R: v = R*w; end point (R, R).
    radius, w = 30.0, 0.5
    vl, vr = w * (radius - wb / 2), w * (radius + wb / 2)
    x, y, th = predict_pose(vl, vr, 0, 0, 0, (math.pi / 2) / w, wb)
    assert (float(x), float(y), float(th)) == pytest.approx((radius, radius, math.pi / 2))


def test_trajectory_into_obstacle_scores_collision(cfg):
    planner = DwaPlanner(cfg)
    obstacle = np.array([[25.0, 0.0]])
    score, collided = planner.score(np.array([20.0]), np.array([20.0]), (0, 0, 0), obstacle)
    assert collided[0] and score[0] == COLLISION_SCORE


def test_planner_steers_around_obstacle_ahead(cfg):
    planner = DwaPlanner(cfg)
    clear = planner.plan((0, 0, 0), (15, 15), np.empty((0, 2)))
    assert clear.v_left == pytest.approx(clear.v_right, abs=1e-6)
    blocked = planner.plan((0, 0, 0), (15, 15), np.array([[30.0, 2.0]]))
    assert abs(blocked.v_left - blocked.v_right) > 1


def test_dynamic_window_respects_acceleration(cfg):
    planner = DwaPlanner(cfg)
    step = cfg.dwa.max_accel_cm_s2 * cfg.dwa.dt_s
    plan = planner.plan((0, 0, 0), (0, 0), np.empty((0, 2)))
    assert abs(plan.v_left) <= step + 1e-9 and abs(plan.v_right) <= step + 1e-9
