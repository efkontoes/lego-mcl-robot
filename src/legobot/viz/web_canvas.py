"""Text protocol understood by the course's web viewer on the Raspberry Pi.

The viewer reads stdout lines such as ``drawLine:(x1, y1, x2, y2)`` and
``drawParticles:[(x, y, theta), ...]`` in pixels. One transform is used for
both walls and particles (the coursework mixed two different scales).
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np

from legobot.localisation.world_map import WorldMap


class WebCanvas:
    def __init__(self, map_size_cm: float = 210, canvas_px: int = 768, margin_frac: float = 0.05,
                 out=print):
        self.map_size = map_size_cm
        self.margin = margin_frac * map_size_cm
        self.scale = canvas_px / (map_size_cm + 2 * self.margin)
        self.out = out

    def to_screen(self, x, y):
        return (x + self.margin) * self.scale, (self.map_size + self.margin - y) * self.scale

    def draw_line(self, x0, y0, x1, y1) -> None:
        sx0, sy0 = self.to_screen(x0, y0)
        sx1, sy1 = self.to_screen(x1, y1)
        self.out(f"drawLine:{(round(sx0, 1), round(sy0, 1), round(sx1, 1), round(sy1, 1))}")

    def draw_walls(self, world: WorldMap) -> None:
        for wall in world.walls:
            self.draw_line(*wall)

    def draw_path(self, points: Iterable) -> None:
        pts = list(points)
        for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
            self.draw_line(x0, y0, x1, y1)

    def draw_particles(self, particles: np.ndarray, weights: Optional[np.ndarray] = None) -> None:
        sx, sy = self.to_screen(particles[:, 0], particles[:, 1])
        data = [(round(a, 1), round(b, 1), round(float(t), 3))
                for a, b, t in zip(sx, sy, particles[:, 2])]
        self.out(f"drawParticles:{data}")

    def draw_image(self, path: str) -> None:
        self.out(f"drawImg:{path}")
