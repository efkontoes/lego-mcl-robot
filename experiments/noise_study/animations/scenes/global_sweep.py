"""Global localisation, seed 1000: the 12-reading sonar sweep pins down the true
start, yet none of V1's 1,000 particles is near it ("hypothesis lost").

Data: the saved global trace and V1 snapshots; the likelihood maps are the
first-sweep log-likelihood over all start positions (max over heading).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import manimgx as m
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import COLORS, DATA, INK, INK2, World, place_robot, robot, text  # noqa: E402

# Sequential blue ramp (light -> dark) matching the figures.
RAMP = np.array([[0xF0, 0xF6, 0xFE], [0xB7, 0xD3, 0xF6], [0x55, 0x98, 0xE7], [0x1C, 0x5C, 0xAB]], float)


ROBOT = COLORS["V2"]  # orange: the blue robot would vanish on the blue heatmap
FLOOR = 30.0  # nats; must match REL_FLOOR in prepare_data.py


def to_rgba(rel: np.ndarray) -> np.ndarray:
    """Relative log-likelihood in [-FLOOR, 0] -> RGBA image (NaN = outside the arena)."""
    v = np.clip((np.nan_to_num(rel, nan=-FLOOR) + FLOOR) / FLOOR, 0, 1)
    pos = v * (len(RAMP) - 1)
    i = np.clip(np.floor(pos).astype(int), 0, len(RAMP) - 2)
    f = (pos - i)[..., None]
    rgb = RAMP[i] * (1 - f) + RAMP[i + 1] * f
    alpha = np.where(np.isnan(rel), 0, 255)[..., None]
    rgb = np.where(np.isnan(rel)[..., None], 255, rgb)
    return np.concatenate([rgb, alpha], axis=-1).astype(np.uint8)[::-1]  # row 0 = top of the image


class GlobalSweep(m.Scene):
    def construct(self) -> None:
        z = np.load(DATA / "global_sweep.npz")
        maps, start, offsets = z["maps"], z["start"], z["offsets"]
        readings, valid, particles = z["readings"], z["valid"], z["particles"]
        world = World(center=(-2.7, -0.3), size=6.2)
        title = text("Global localisation: does the sweep find the robot? (seed 1000)", 26, INK,
                     "BOLD").to_edge(m.UP, buff=0.25)
        walls = world.walls(z["walls"])
        heat = m.ImageMobject(to_rgba(np.where(np.isnan(maps[0]), np.nan, -FLOOR)))
        heat.stretch_to_fit_width(6.2).stretch_to_fit_height(6.2).move_to(world.p(105, 105))
        tri = robot(0.26, ROBOT)
        place_robot(tri, world, start[0], start[1], start[2], 0.26)
        beam = m.Line(world.p(start[0], start[1]), world.p(start[0] + 1, start[1]), color=INK, stroke_width=3)
        beam.set_stroke(opacity=0)
        dots = m.VGroup(*[m.Dot(world.p(0, 0), radius=0.035, color=INK) for _ in range(particles.shape[1])])
        for d in dots:
            d.set_fill(opacity=0)

        panel_x = 3.9
        info = text("", 20, INK)
        expl = m.VGroup(
            text("Shading: how well each start position", 18, INK2),
            text("explains the readings so far", 18, INK2),
            text("(dark = fits best; scale 30 nats)", 18, INK2),
        ).arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.08).move_to([panel_x, 2.2, 0])
        key = m.VGroup(
            m.VGroup(robot(0.16, ROBOT), text("true start (unknown to the filter)", 17, INK2)).arrange(m.RIGHT, buff=0.15),
            m.VGroup(m.Dot(radius=0.05, color=INK), text("V1 particles", 17, INK2)).arrange(m.RIGHT, buff=0.15),
            m.VGroup(m.Square(side_length=0.22, color=COLORS["V8"], stroke_width=3),
                     text("best-fitting start so far", 17, INK2)).arrange(m.RIGHT, buff=0.15),
        ).arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.15).move_to([panel_x, -2.6, 0])
        best = m.Square(side_length=0.3, color=COLORS["V8"], stroke_width=3).set_fill(opacity=0)
        best.set_stroke(opacity=0)
        grid_xy = np.arange(1.25, 210, 2.5)
        self.add(title, heat, walls, dots, best, beam, tri, expl, key)
        self.wait(0.6)

        for i in range(len(offsets)):
            th = start[2] + offsets[i]
            turn = offsets[i] - (offsets[i - 1] if i else 0.0)
            self.play(m.Rotate(tri, angle=turn, about_point=world.p(start[0], start[1])), run_time=0.35)
            ok = bool(valid[i])
            if ok:
                r = float(readings[i])
                beam.put_start_and_end_on(world.p(start[0], start[1]),
                                          world.p(start[0] + r * math.cos(th), start[1] + r * math.sin(th)))
                beam.set_stroke(INK, opacity=0.9)
                status = f"reading {i + 1}/12:  {r:.0f} cm"
            else:
                beam.set_stroke(opacity=0)
                status = f"reading {i + 1}/12:  no usable echo"
            new_heat = m.ImageMobject(to_rgba(maps[i]))
            new_heat.stretch_to_fit_width(6.2).stretch_to_fit_height(6.2).move_to(world.p(105, 105))
            pts = particles[i]
            for k, d in enumerate(dots):
                if k < len(pts) and not np.isnan(pts[k][0]):
                    d.move_to(world.p(*pts[k]))
                    d.set_fill(opacity=0.5)
                else:
                    d.set_fill(opacity=0)
            info.become(text(status, 22, INK, "BOLD").move_to([panel_x, 1.1, 0]))
            row, col = np.unravel_index(np.nanargmax(maps[i]), maps[i].shape)
            best.move_to(world.p(grid_xy[col], grid_xy[row]))
            best.set_stroke(opacity=1)
            self.add(new_heat)
            self.bring_to_front(walls, dots, best, beam, tri, info)
            self.play(m.FadeIn(new_heat), run_time=0.45)
            self.remove(heat)
            heat = new_heat
            self.wait(0.25)

        mass = float(z["mass_near_truth"][-1])
        verdict = m.VGroup(
            text("The readings single out the true start,", 20, INK, "BOLD"),
            text("but no V1 particle is near it:", 20, INK, "BOLD"),
            text(f"particle mass within 20 cm = {mass:.2f}", 20, COLORS["V8"], "BOLD"),
            text("a sampling failure, not sensor ambiguity", 18, INK2),
        ).arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.1).move_to([panel_x, -0.6, 0])
        ring = m.Circle(radius=0.32, color=ROBOT, stroke_width=4).move_to(world.p(start[0], start[1]))
        self.play(m.Create(ring), m.FadeIn(verdict), run_time=0.8)
        self.wait(3.0)
