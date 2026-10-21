"""Shared look and helpers for the ManimGX scenes (same palette as the figures)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import manimgx as m
import numpy as np

DATA = Path(__file__).resolve().parents[1] / "data"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
WALL = "#222222"
TRUTH = "#1565c0"
COLORS = {"V1": "#898781", "V2": "#eb6834", "V3": "#1baf7a", "V4": "#eda100", "V5": "#e87ba4",
          "V6": "#008300", "V7": "#4a3aa7", "V8": "#e34948", "V3-N500": "#86b6ef",
          "V3-N1000": "#2a78d6", "V1-N1000": "#52514e"}
FONT = "sans-serif"

m.config.background_color = SURFACE


def load_json(name: str) -> dict:
    return json.loads((DATA / name).read_text())


def text(s: str, size: float = 24, color: str = INK, weight: str = "NORMAL") -> m.Text:
    return m.Text(s, font=FONT, font_size=size, color=color, weight=weight)


class World:
    """Maps arena coordinates (cm) to scene coordinates."""

    def __init__(self, center=(0.0, 0.0), size: float = 6.0, extent: float = 210.0):
        self.cx, self.cy = center
        self.k = size / extent
        self.half = extent / 2

    def p(self, x: float, y: float) -> np.ndarray:
        return np.array([self.cx + (x - self.half) * self.k, self.cy + (y - self.half) * self.k, 0.0])

    def walls(self, walls, width: float = 4) -> m.VGroup:
        return m.VGroup(*[m.Line(self.p(a, b), self.p(c, d), color=WALL, stroke_width=width)
                          for a, b, c, d in walls])


def robot(size: float = 0.22, color: str = TRUTH) -> m.Polygon:
    """A triangle pointing along +x (the robot's heading)."""
    return m.Polygon([size, 0, 0], [-0.6 * size, 0.55 * size, 0], [-0.6 * size, -0.55 * size, 0],
                     color=color, fill_color=color, fill_opacity=1, stroke_width=1.5, stroke_color=SURFACE)


def place_robot(tri: m.Polygon, world: World, x: float, y: float, th: float, size: float = 0.22) -> None:
    c, s = math.cos(th), math.sin(th)
    local = [(size, 0), (-0.6 * size, 0.55 * size), (-0.6 * size, -0.55 * size)]
    origin = world.p(x, y)
    pts = [origin + np.array([lx * c - ly * s, lx * s + ly * c, 0.0]) for lx, ly in local]
    tri.set_points_as_corners([*pts, pts[0]])


def cross(size: float = 0.09, color: str = INK, width: float = 3.5) -> m.VGroup:
    return m.VGroup(m.Line([-size, -size, 0], [size, size, 0], color=color, stroke_width=width),
                    m.Line([-size, size, 0], [size, -size, 0], color=color, stroke_width=width))
