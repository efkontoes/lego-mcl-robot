"""Is it the technique or the particle budget? Kidnap recovery vs computation,
revealed in three steps. Data: results/analysis/main_summary.csv (100 seeds)."""

from __future__ import annotations

import sys
from pathlib import Path

import manimgx as m
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import COLORS, INK, INK2, MUTED, SURFACE, load_json, text  # noqa: E402

X_MAX, Y_MAX = 175, 0.8
# Label direction per point, chosen so the labels of clustered points never collide.
LABEL_DIR = {"V1": (-1, -1), "V2": (-1, 0.3), "V7": (1, -0.6), "V3": (1, 0.2), "V6": (-1, 1), "V4": (1, 0),
             "V1-N1000": (-1, 0.4), "V3-N500": (-1, 0), "V3-N1000": (1, 0.6), "V5": (1, 0.6), "V8": (1, 0)}


class BudgetReveal(m.Scene):
    def construct(self) -> None:
        pts = {p["variant"]: p for p in load_json("budget.json")["points"]}
        axes = m.Axes(x_range=[0, X_MAX, 25], y_range=[0, Y_MAX, 0.2], x_length=10.2, y_length=5.2,
                      tips=False, axis_config={"color": MUTED, "stroke_width": 2, "include_ticks": True})
        axes.shift(0.45 * m.DOWN + 0.35 * m.RIGHT)
        xl = text("filter CPU per kidnap trajectory (ms, laptop)", 20, INK2).next_to(axes, m.DOWN, buff=0.45)
        yl = text("sustained kidnap recovery", 20, INK2).rotate(m.PI / 2).next_to(axes, m.LEFT, buff=0.75)
        xt = m.VGroup(*[text(str(v), 16, INK2).next_to(axes.c2p(v, 0), m.DOWN, buff=0.15)
                        for v in range(0, X_MAX + 1, 25)])
        yt = m.VGroup(*[text(f"{int(v * 100)}%", 16, INK2).next_to(axes.c2p(0, v), m.LEFT, buff=0.15)
                        for v in (0, 0.2, 0.4, 0.6, 0.8)])
        title = text("Kidnap recovery vs computation (100 held-out seeds, identical replayed data)", 26, INK,
                     "BOLD").to_edge(m.UP, buff=0.25)
        caption = text("", 22, INK)
        self.add(title)
        self.play(m.Create(axes), m.FadeIn(xl), m.FadeIn(yl), m.FadeIn(xt), m.FadeIn(yt), run_time=1.0)

        def point(v: str, hollow: bool = False) -> m.VGroup:
            p = pts[v]
            c = axes.c2p(p["cpu"], p["recovered"])
            bar = m.Line(axes.c2p(p["cpu"], p["lo"]), axes.c2p(p["cpu"], p["hi"]), color=COLORS[v], stroke_width=3)
            dot = m.Dot(c, radius=0.11, color=COLORS[v])
            if hollow:
                dot.set_fill(SURFACE, opacity=1).set_stroke(COLORS[v], width=4)
            dx, dy = LABEL_DIR.get(v, (1, 1))
            label = text(v, 18, INK2).next_to(dot, np.array([dx, dy, 0.0]), buff=0.1)
            return m.VGroup(bar, dot, label)

        low = ("V1", "V2", "V4", "V7", "V3", "V6")
        best_low = max(pts[v]["recovered"] for v in low)
        caption = text(f"Each technique alone, and ESS + injection: at most {best_low:.0%} recovery", 22, INK,
                       "BOLD").next_to(title, m.DOWN, buff=0.18)
        groups = {v: point(v) for v in low}
        self.play(m.FadeIn(caption), m.LaggedStart(*[m.FadeIn(g, scale=0.6) for g in groups.values()],
                                                   lag_ratio=0.18), run_time=2.0)
        self.wait(1.0)

        hi = {v: point(v) for v in ("V8", "V5")}
        v1, v5, v8 = pts["V1"], pts["V5"], pts["V8"]
        new = text(f"Injection + KLD (V8, V5): {v8['recovered']:.0%}–{v5['recovered']:.0%} recovery, at "
                   f"{v5['cpu'] / v1['cpu']:.0f}–{v8['cpu'] / v1['cpu']:.0f}× the CPU of V1", 22, INK,
                   "BOLD").next_to(title, m.DOWN, buff=0.18)
        self.play(m.FadeOut(caption), run_time=0.3)  # out first, so captions never overlap
        self.play(m.FadeIn(new), m.LaggedStart(*[m.FadeIn(g, scale=0.6) for g in hi.values()], lag_ratio=0.3),
                  run_time=1.4)
        caption = new
        self.wait(1.2)

        q = text("Is it the KLD + injection combination, or just the larger particle budget?", 22, INK,
                 "BOLD").next_to(title, m.DOWN, buff=0.18)
        self.play(m.FadeOut(caption), run_time=0.3)
        self.play(m.FadeIn(q), run_time=0.4)
        caption = q
        self.wait(0.8)

        ctrl = {v: point(v, hollow=True) for v in ("V3-N500", "V3-N1000", "V1-N1000")}
        self.play(m.LaggedStart(*[m.FadeIn(g, scale=0.6) for g in ctrl.values()], lag_ratio=0.35), run_time=1.6)
        a, b = pts["V8"], pts["V3-N500"]
        arrow = m.Arrow(axes.c2p(a["cpu"], a["recovered"]), axes.c2p(b["cpu"], b["recovered"]), buff=0.18,
                        color=INK, stroke_width=4)
        note = text(f"same recovery ({b['recovered']:.0%}), {a['cpu'] / b['cpu']:.1f}× less CPU", 20, INK).next_to(
            arrow, m.DOWN, buff=0.18)
        ans = text("Injection with a fixed 500–1000 particles matches it: the budget, not a synergy", 22,
                   COLORS["V3-N1000"], "BOLD").next_to(title, m.DOWN, buff=0.18)
        self.play(m.GrowArrow(arrow), m.FadeIn(note), m.FadeOut(caption), run_time=0.8)
        self.play(m.FadeIn(ans), run_time=0.5)
        self.wait(1.0)
        legend = text("filled = factorial variants   ·   hollow = fixed-budget controls   ·   bars = Wilson 95% CI",
                      16, INK2).next_to(axes.c2p(X_MAX, 0.32), m.LEFT, buff=0.1)
        self.play(m.FadeIn(legend), run_time=0.5)
        self.wait(2.5)
