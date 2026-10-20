"""Small geometry helpers shared across the package."""

from __future__ import annotations

import math

import numpy as np


def wrap_angle(angle):
    """Wrap an angle (scalar or array) to [-pi, pi)."""
    if isinstance(angle, (int, float)):
        return (angle + math.pi) % (2 * math.pi) - math.pi
    return (np.asarray(angle) + np.pi) % (2 * np.pi) - np.pi


def heading_and_distance(x: float, y: float, theta: float, target_x: float, target_y: float):
    """Turn needed to face a target (wrapped to [-pi, pi)) and the distance to it."""
    dx, dy = target_x - x, target_y - y
    turn = wrap_angle(float(math.atan2(dy, dx) - theta))
    return turn, math.hypot(dx, dy)


def robot_to_world(points, pose):
    """Transform robot-frame points (x forward, y left) into the world frame."""
    x, y, theta = pose
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    c, s = math.cos(theta), math.sin(theta)
    wx = x + pts[:, 0] * c - pts[:, 1] * s
    wy = y + pts[:, 0] * s + pts[:, 1] * c
    return np.column_stack([wx, wy])


def world_to_robot(points, pose):
    """Inverse of :func:`robot_to_world`."""
    x, y, theta = pose
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    c, s = math.cos(theta), math.sin(theta)
    dx, dy = pts[:, 0] - x, pts[:, 1] - y
    return np.column_stack([dx * c + dy * s, -dx * s + dy * c])
