"""How KLD-sampling decides the particle count: draw samples one at a time,
count the occupied histogram bins, stop when the bound is met.

Data: the real kld_resample run on two saved particle clouds (seed 1000, V8):
a concentrated cloud before the kidnap and a spread cloud after it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import manimgx as m
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import COLORS, INK, INK2, MUTED, World, load_json, text  # noqa: E402

DRAW_COLOR = COLORS["V8"]
BIN_COLOR = "#2a78d6"


class KldSampling(m.Scene):
    def construct(self) -> None:
        d = load_json("kld.json")
        title = text("KLD-sampling: how many particles are enough?", 28, INK, "BOLD").to_edge(m.UP, buff=0.25)
        rule = text("draw samples one by one · count occupied 5 cm × 5 cm × 15° bins k · stop once "
                    "n ≥ bound(k)", 19, INK2).next_to(title, m.DOWN, buff=0.12)
        self.add(title, rule)
        self.run_case(d, "concentrated", "A tight cloud (before the kidnap)", first=True)
        self.run_case(d, "spread", "A spread cloud (after the kidnap)", first=False)
        self.wait(2.0)

    def run_case(self, d, name, heading, first):
        case = d["cases"][name]
        draws = np.array(case["draws"], dtype=float)
        n_total = case["n"]
        kk = np.array(case["k"])
        bound = np.array(case["bound"], dtype=float)
        n_max = d["n_max"]

        # Left: the particle cloud, zoomed on the drawn samples, with its histogram bins.
        lo = draws.min(axis=0) - 6
        hi = draws.max(axis=0) + 6
        extent = float(max(hi - lo)) if name == "concentrated" else 210.0
        center = (lo + hi) / 2 if name == "concentrated" else np.array([105.0, 105.0])
        world = World(center=(-3.6, -0.55), size=5.2, extent=extent)
        world.half = 0.0  # center the view on `center`

        def p(x, y):
            return np.array([world.cx + (x - center[0]) * world.k, world.cy + (y - center[1]) * world.k, 0.0])

        frame = m.Square(side_length=5.2, color=MUTED, stroke_width=1.5).move_to([-3.6, -0.55, 0])
        sub = text(heading, 21, INK, "BOLD").next_to(frame, m.UP, buff=0.12)
        scale_note = text(f"view {extent:.0f} cm wide", 15, INK2).next_to(frame, m.DOWN, buff=0.1)
        walls = m.VGroup()
        if name == "spread":
            walls = m.VGroup(*[m.Line(p(a, b), p(c, e), color="#222222", stroke_width=3)
                               for a, b, c, e in d["walls"]])
        src = m.VGroup(*[m.Dot(p(x, y), radius=0.03, color=MUTED) for x, y in case["source"]])

        # Right: samples drawn vs samples required.
        # Round axis ranges: 0-200 for the tight cloud, 0-1000 x 0-2000 for the spread one.
        x_top = float(n_max) if name == "spread" else 200.0
        y_top = 2000.0 if name == "spread" else 200.0
        axes = m.Axes(x_range=[0, x_top, x_top / 4], y_range=[0, y_top, y_top / 4], x_length=5.6,
                      y_length=4.3, tips=False, axis_config={"color": MUTED, "stroke_width": 2})
        axes.move_to([3.55, -0.45, 0])
        xl = text("samples drawn n", 17, INK2).next_to(axes, m.DOWN, buff=0.45)
        yl = text("samples needed", 17, INK2).rotate(m.PI / 2).next_to(axes, m.LEFT, buff=0.6)
        diag = m.DashedLine(axes.c2p(0, 0), axes.c2p(min(x_top, y_top), min(x_top, y_top)), color=MUTED,
                            stroke_width=2)
        diag_lbl = text("needed = drawn", 15, INK2).next_to(diag.point_from_proportion(0.6), m.DR, buff=0.08)
        ticks = m.VGroup(
            *[text(f"{v:.0f}", 16, INK2).next_to(axes.c2p(v, 0), m.DOWN, buff=0.22)
              for v in np.linspace(0, x_top, 5)],
            *[text(f"{v:.0f}", 16, INK2).next_to(axes.c2p(0, v), m.LEFT, buff=0.22)
              for v in np.linspace(0, y_top, 5)[1:]])
        cap = m.DashedLine(axes.c2p(0, n_max), axes.c2p(x_top, n_max), color=MUTED, stroke_width=1.5) \
            if name == "spread" else m.VGroup()
        cap_lbl = text("n_max = 1000", 15, INK2).next_to(cap, m.UP, buff=0.05).align_to(cap, m.RIGHT) \
            if name == "spread" else m.VGroup()
        nmin = m.DashedLine(axes.c2p(d["n_min"], 0), axes.c2p(d["n_min"], y_top), color=MUTED, stroke_width=1.5)
        nmin_lbl = text("n_min", 15, INK2).next_to(nmin.get_end(), m.DOWN, buff=0.08).shift(0.35 * m.RIGHT)

        t = m.ValueTracker(0)

        def bound_curve():
            n = max(int(t.get_value()), 1)
            xs = np.arange(1, n + 1)
            pts = [axes.c2p(x, min(b, y_top)) for x, b in zip(xs, bound[:n])]
            if len(pts) < 2:
                pts = [axes.c2p(0, 0), axes.c2p(0.01, 0)]
            return m.VMobject(color=BIN_COLOR, stroke_width=3).set_points_as_corners(pts)

        drawn = m.VGroup(*[m.Dot(p(x, y), radius=0.04, color=DRAW_COLOR) for x, y in draws])
        for dot in drawn:
            dot.set_fill(opacity=0)
        bins = {}

        def bin_square(code_xy):
            bx, by = code_xy
            s = d["bin_cm"]
            sq = m.Square(side_length=s * world.k, color=BIN_COLOR, stroke_width=1.2)
            sq.set_fill(BIN_COLOR, opacity=0.12).move_to(p(bx * s + s / 2, by * s + s / 2))
            return sq

        bin_layer = m.VGroup()
        counter = text("", 20, INK)
        state = {"shown": 0}

        def refresh(_mob):
            n = min(int(t.get_value()), n_total)
            if n == state["shown"]:
                return
            for i in range(state["shown"], n):
                drawn[i].set_fill(opacity=0.75)
                x, y = draws[i]
                key = (int(np.floor(x / d["bin_cm"])), int(np.floor(y / d["bin_cm"])))
                if key not in bins:
                    bins[key] = bin_square(key)
                    bin_layer.add(bins[key])
            state["shown"] = n
            k = int(kk[n - 1]) if n else 0
            b = float(bound[n - 1]) if n else 0.0
            counter.become(text(f"n = {n}   ·   bins k = {k}   ·   bound(k) = {b:.0f}", 20, INK)
                           .next_to(axes, m.UP, buff=0.4))

        anchor = m.Dot(radius=0.001).set_fill(opacity=0)
        anchor.add_updater(refresh)
        curve = m.always_redraw(bound_curve)
        group_in = [frame, sub, scale_note, walls, src, axes, ticks, xl, yl, diag, diag_lbl, cap, cap_lbl, nmin,
                    nmin_lbl]
        self.play(*[m.FadeIn(g) for g in group_in], run_time=0.8)
        self.add(bin_layer, drawn, curve, anchor, counter)
        refresh(anchor)
        self.play(t.animate.set_value(n_total), run_time=5.5 if name == "spread" else 4.0, rate_func=m.linear)
        if n_total < n_max:
            verdict = f"Stop at n = {n_total}: only {int(kk[-1])} bins occupied, so a small set suffices"
        else:
            verdict = (f"{int(kk[-1])} bins would need {bound[-1]:.0f} samples: capped at n_max = {n_max}")
        v = text(verdict, 21, INK, "BOLD").to_edge(m.DOWN, buff=0.15)
        self.play(m.FadeIn(v, shift=0.15 * m.UP), run_time=0.5)
        self.wait(2.2)
        everything = [*group_in, bin_layer, drawn, curve, anchor, counter, v]
        if first:
            self.play(*[m.FadeOut(g) for g in everything], run_time=0.6)
            anchor.clear_updaters()
            self.remove(curve)
