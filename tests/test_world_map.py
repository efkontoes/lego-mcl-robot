import math

import numpy as np
import pytest

from legobot.localisation.world_map import WorldMap


def test_raycast_known_distances(world):
    d, idx = world.raycast([100, 100, 100, 20], [50, 50, 50, 150],
                           [0, math.pi / 2, math.pi, math.pi / 2])
    assert d == pytest.approx([110, 160, 100, 18])
    assert (idx >= 0).all()


def test_raycast_ignores_walls_behind_and_parallel():
    box = WorldMap([(10, -5, 10, 5)], [(0, -10), (0, 10), (20, 10), (20, -10)])
    d, idx = box.raycast([0, 0, 0], [0, 0, 0], [0, math.pi, math.pi / 2])
    assert d[0] == pytest.approx(10)
    assert np.isinf(d[1]) and idx[1] == -1  # wall is behind
    assert np.isinf(d[2])  # parallel to the wall


def test_raycast_misses_beyond_segment_end():
    box = WorldMap([(10, -5, 10, 5)], [(0, -10), (0, 10), (20, 10), (20, -10)])
    d, _ = box.raycast(0, 0, math.atan2(8, 10))  # line hit at y = 8 > 5
    assert np.isinf(d[0])


def test_incidence_angle(world):
    _, idx = world.raycast(100, 50, 0.0)
    assert world.incidence_angle(0.0, idx)[0] == pytest.approx(0.0)
    _, idx = world.raycast(100, 50, math.radians(30))
    assert world.incidence_angle(math.radians(30), idx)[0] == pytest.approx(math.radians(30))


def test_free_space_sampling(world, rng):
    poses = world.sample_free_poses(500, rng)
    assert poses.shape == (500, 3)
    assert world.contains(poses[:, 0], poses[:, 1]).all()
    assert not world.contains([40], [190])[0]  # behind wall b, outside the arena
