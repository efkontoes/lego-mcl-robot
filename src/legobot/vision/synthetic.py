"""Synthetic camera images for the simulator.

Red cans standing on the floor are rendered by projecting their base circle
through the calibrated ground homography and extruding it upwards in the
image. The result goes through exactly the same detection code as real
frames, so the whole vision pipeline is exercised in simulation.
"""

from __future__ import annotations

import numpy as np

from legobot.config import CameraConfig
from legobot.vision.homography import GroundHomography

FLOOR_BGR = (95, 95, 95)
CAN_BGR = (0, 0, 200)


def render_cans(cans_robot_frame, homography: GroundHomography, cfg: CameraConfig,
                can_radius_cm: float, can_height_cm: float, max_x_cm: float = 160.0):
    import cv2

    image = np.full((cfg.height, cfg.width, 3), FLOOR_BGR, dtype=np.uint8)
    # Draw far cans first so nearer ones occlude them.
    for x, y in sorted(map(tuple, np.asarray(cans_robot_frame).reshape(-1, 2)), key=lambda p: -p[0]):
        if not (can_radius_cm < x <= max_x_cm):
            continue
        phi = np.linspace(0, 2 * np.pi, 24, endpoint=False)
        us, vs = homography.to_pixel(x + can_radius_cm * np.cos(phi), y + can_radius_cm * np.sin(phi))
        base = np.column_stack([us, vs])
        width = us.max() - us.min()
        height = width * can_height_cm / (2 * can_radius_cm)
        body = np.array([[us.min(), vs.mean() - height], [us.max(), vs.mean() - height],
                         [us.max(), vs.mean()], [us.min(), vs.mean()]])
        for poly in (body, base):
            cv2.fillPoly(image, [np.round(poly).astype(np.int32)], CAN_BGR)
    return image
