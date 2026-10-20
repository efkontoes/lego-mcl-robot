"""Red obstacle detection: HSV threshold -> connected components -> floor points."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from legobot.config import CameraConfig
from legobot.vision.homography import GroundHomography


@dataclass(frozen=True)
class Blob:
    u: int  # bottom-centre pixel (where the object touches the floor)
    v: int
    area: int
    centroid: tuple


def red_mask(frame_bgr: np.ndarray, cfg: CameraConfig) -> np.ndarray:
    """Binary mask of red pixels. Red wraps around hue 0, hence several ranges."""
    import cv2

    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lo, hi in cfg.red_hsv_ranges:
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, np.array(lo), np.array(hi)))
    return mask


def detect_red_blobs(frame_bgr: np.ndarray, cfg: CameraConfig) -> List[Blob]:
    """Red regions larger than ``min_blob_area_px``."""
    import cv2

    num, _, stats, centroids = cv2.connectedComponentsWithStats(red_mask(frame_bgr, cfg), 4,
                                                                cv2.CV_32S)
    blobs = []
    for i in range(1, num):  # label 0 is the background
        left, top, width, height, area = stats[i]
        if area < cfg.min_blob_area_px:
            continue
        blobs.append(Blob(u=int(left + width // 2), v=int(top + height - 1), area=int(area),
                          centroid=tuple(centroids[i])))
    return blobs


def blobs_to_floor(blobs: List[Blob], homography: GroundHomography, min_x: float, max_x: float):
    """Project blob bottom-centres onto the floor (robot frame, cm).

    The bottom of a blob is where the object meets the floor, the only part of
    it that lies on the plane the homography was calibrated for.
    """
    if not blobs:
        return np.empty((0, 2)), []
    xs, ys = homography.to_ground([b.u for b in blobs], [b.v for b in blobs])
    keep = (xs >= min_x) & (xs <= max_x)
    kept = [b for b, k in zip(blobs, keep) if k]
    return np.column_stack([xs[keep], ys[keep]]), kept


def annotate(frame_bgr: np.ndarray, blobs: List[Blob], color=(255, 255, 255)) -> np.ndarray:
    import cv2

    out = frame_bgr.copy()
    for b in blobs:
        cv2.circle(out, (b.u, b.v), 6, color, 2)
        cv2.putText(out, f"({b.u},{b.v})", (b.u + 8, b.v), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color,
                    1, cv2.LINE_AA)
    return out
