"""Ground-plane homography between robot-frame floor points and image pixels.

A flat floor seen by a fixed camera is related to the image by a 3x3
homography ``H`` (8 unknowns). Four ground points with known positions and
pixel locations give 8 linear equations, solved here by least squares.
"""

from __future__ import annotations

from typing import Iterable, Sequence, Tuple

import numpy as np

from legobot.config import CameraConfig


def solve_homography(points: Iterable[Sequence[float]]) -> np.ndarray:
    """Fit H from rows ``(x, y, u, v)`` such that ``[u, v, 1] ~ H [x, y, 1]``."""
    rows, rhs = [], []
    for x, y, u, v in points:
        rows.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        rows.append([0, 0, 0, x, y, 1, -v * x, -v * y])
        rhs.extend([u, v])
    h, *_ = np.linalg.lstsq(np.asarray(rows, float), np.asarray(rhs, float), rcond=None)
    return np.append(h, 1.0).reshape(3, 3)


def _apply(matrix: np.ndarray, a, b) -> Tuple[np.ndarray, np.ndarray]:
    vec = matrix @ np.vstack([np.atleast_1d(a), np.atleast_1d(b), np.ones(np.size(a))])
    return vec[0] / vec[2], vec[1] / vec[2]


class GroundHomography:
    """Converts between floor coordinates (cm, robot frame) and pixels."""

    def __init__(self, H: np.ndarray):
        self.H = H
        self.H_inv = np.linalg.inv(H)

    @classmethod
    def from_config(cls, cfg: CameraConfig) -> GroundHomography:
        return cls(solve_homography(cfg.homography_points))

    def to_pixel(self, x, y):
        return _apply(self.H, x, y)

    def to_ground(self, u, v):
        return _apply(self.H_inv, u, v)


def draw_ground_grid(image, homography: GroundHomography, x_max=80, y_half=30, step=10):
    """Overlay a floor grid (cm) on a camera image to check the calibration."""
    import cv2

    white = (255, 255, 255)
    for x in range(0, x_max + 1, step):
        (ua, ub), (va, vb) = homography.to_pixel([x, x], [-y_half, y_half])
        cv2.line(image, (int(ua), int(va)), (int(ub), int(vb)), white, 1)
        uc, vc = homography.to_pixel(x, 0)
        cv2.putText(image, str(x), (int(uc[0]), int(vc[0])), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    white, 1, cv2.LINE_AA)
    for y in range(-y_half, y_half + 1, step):
        (ua, ub), (va, vb) = homography.to_pixel([0, x_max], [y, y])
        cv2.line(image, (int(ua), int(va)), (int(ub), int(vb)), white, 1)
    return image
