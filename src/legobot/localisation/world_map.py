"""Wall map of the arena, vectorised sonar ray casting and free-space sampling."""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
from matplotlib.path import Path as _Polygon

Wall = Tuple[float, float, float, float]

# The practical 4 arena (cm). Letters follow the course handout: O, A, B, ... H.
ARENA_WALLS: Tuple[Wall, ...] = (
    (0, 0, 0, 168),  # a: O -> A
    (0, 168, 84, 168),  # b: A -> B
    (84, 126, 84, 210),  # c: C -> D (the stub between C and B sticks into the room)
    (84, 210, 168, 210),  # d: D -> E
    (168, 210, 168, 84),  # e: E -> F
    (168, 84, 210, 84),  # f: F -> G
    (210, 84, 210, 0),  # g: G -> H
    (210, 0, 0, 0),  # h: H -> O
)

# Outline of the free space enclosed by the walls above.
ARENA_BOUNDARY: Tuple[Tuple[float, float], ...] = (
    (0, 0), (0, 168), (84, 168), (84, 210), (168, 210), (168, 84), (210, 84), (210, 0),
)

_EPS = 1e-9
_TOL = 1e-6


class WorldMap:
    """A set of wall segments plus the polygon of reachable free space."""

    def __init__(self, walls: Sequence[Wall], boundary: Sequence[Tuple[float, float]]):
        self.walls = np.asarray(walls, dtype=float).reshape(-1, 4)
        self.boundary = np.asarray(boundary, dtype=float)
        self._polygon = _Polygon(self.boundary)
        self.x_range = (self.boundary[:, 0].min(), self.boundary[:, 0].max())
        self.y_range = (self.boundary[:, 1].min(), self.boundary[:, 1].max())

    def raycast(self, x, y, theta) -> Tuple[np.ndarray, np.ndarray]:
        """Distance along heading ``theta`` from (x, y) to the first wall hit.

        Accepts scalars or arrays (one ray per element). Returns ``(distance,
        wall_index)``; rays that hit nothing get ``inf`` and index ``-1``.
        """
        x = np.atleast_1d(np.asarray(x, dtype=float))[:, None]
        y = np.atleast_1d(np.asarray(y, dtype=float))[:, None]
        theta = np.atleast_1d(np.asarray(theta, dtype=float))[:, None]
        ax, ay, bx, by = (self.walls[:, i][None, :] for i in range(4))
        cos_t, sin_t = np.cos(theta), np.sin(theta)

        # Distance m along the ray to the infinite line through the wall.
        numerator = (by - ay) * (ax - x) - (bx - ax) * (ay - y)
        denominator = (by - ay) * cos_t - (bx - ax) * sin_t
        parallel = np.abs(denominator) < _EPS
        with np.errstate(divide="ignore", invalid="ignore"):
            m = numerator / np.where(parallel, 1.0, denominator)

        # Keep hits that are in front of the sensor and inside the wall segment.
        hx, hy = x + m * cos_t, y + m * sin_t
        on_segment = (
            (hx >= np.minimum(ax, bx) - _TOL) & (hx <= np.maximum(ax, bx) + _TOL)
            & (hy >= np.minimum(ay, by) - _TOL) & (hy <= np.maximum(ay, by) + _TOL)
        )
        m = np.where(~parallel & (m > 0) & on_segment, m, np.inf)

        index = np.argmin(m, axis=1)
        distance = m[np.arange(m.shape[0]), index]
        index = np.where(np.isfinite(distance), index, -1)
        return distance, index

    def incidence_angle(self, theta, wall_index) -> np.ndarray:
        """Angle between the ray and the normal of the wall it hit (radians)."""
        theta = np.atleast_1d(np.asarray(theta, dtype=float))
        wall_index = np.atleast_1d(wall_index)
        ax, ay, bx, by = self.walls[np.maximum(wall_index, 0)].T
        length = np.hypot(bx - ax, by - ay)
        cos_beta = ((ay - by) * np.cos(theta) + (bx - ax) * np.sin(theta)) / length
        angle = np.arccos(np.clip(np.abs(cos_beta), 0.0, 1.0))
        return np.where(wall_index >= 0, angle, np.pi / 2)

    def contains(self, x, y) -> np.ndarray:
        """True for points strictly inside the free-space polygon."""
        points = np.column_stack([np.atleast_1d(x), np.atleast_1d(y)])
        return self._polygon.contains_points(points)

    def sample_free_poses(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Uniformly sample ``n`` poses (x, y, theta) inside the free space."""
        poses = np.empty((0, 3))
        while len(poses) < n:
            batch = max(2 * (n - len(poses)), 16)
            xs = rng.uniform(*self.x_range, batch)
            ys = rng.uniform(*self.y_range, batch)
            keep = self.contains(xs, ys)
            thetas = rng.uniform(-np.pi, np.pi, batch)
            poses = np.vstack([poses, np.column_stack([xs, ys, thetas])[keep]])
        return poses[:n]


def arena() -> WorldMap:
    """The practical 4 arena."""
    return WorldMap(ARENA_WALLS, ARENA_BOUNDARY)
