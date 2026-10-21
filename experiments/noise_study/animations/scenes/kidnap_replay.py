"""Two filters, identical sensor data: V1 stays confidently lost, V8 recovers.

Data: seed 1000 of the main Protocol A replay (kidnap scenario), particle
snapshots of every update. Rendered with ManimGX.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import manimgx as m

sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import COLORS, INK, INK2, TRUTH, World, cross, load_json, place_robot, robot, text  # noqa: E402

UPDATES_PER_SECOND = 7
MAX_DOTS = 150
LABELS = {"V1": "V1 · odometry model only", "V8": "V8 · injection + KLD-sampling"}


class KidnapReplay(m.Scene):
    def construct(self) -> None:
        d = load_json("kidnap_replay.json")
        steps = d["steps"]
        n_steps = len(steps)
        k0 = next(i for i, s in enumerate(steps) if s["since_kidnap"] == 0)
        t = m.ValueTracker(0)

        title = text("Same recorded sonar data, two particle filters", 30, INK, "BOLD").to_edge(m.UP, buff=0.25)
        counter = text("", 20, INK2)
        self.add(title)

        panels = []
        for variant, cx in (("V1", -3.55), ("V8", 3.55)):
            world = World(center=(cx, -0.05), size=4.95)
            walls = world.walls(d["walls"])
            head = text(LABELS[variant], 22, INK, "BOLD").next_to(walls, m.UP, buff=0.14)
            vd = d["variants"][variant]
            dots = m.VGroup(*[m.Dot(world.p(0, 0), radius=0.042, color=COLORS[variant]) for _ in range(MAX_DOTS)])
            # The filter's own 95% credible circle: small = confident (right or wrong).
            halo = m.Circle(radius=0.1, color=COLORS[variant], stroke_width=2.5).set_fill(COLORS[variant], opacity=0.1)
            est = cross(size=0.07)
            beam = m.Line(world.p(0, 0), world.p(1, 0), color=INK2, stroke_width=2)
            tri = robot()
            hud = text("", 18, INK2)
            panels.append({"variant": variant, "world": world, "dots": dots, "est": est, "beam": beam,
                           "tri": tri, "hud": hud, "data": vd, "walls": walls, "halo": halo, "last": -1})
            self.add(walls, head, halo, dots, beam, est, tri, hud)
        legend = text("blue = true robot   ·   dots = particles   ·   ×  = estimate   ·   circle = filter's 95% "
                      "confidence region   ·   line = sonar beam", 16, INK2).to_edge(m.DOWN, buff=0.12)
        self.add(legend)

        def step_index() -> int:
            return min(int(t.get_value()), n_steps - 1)

        def true_pose(value: float):
            i = min(int(value), n_steps - 1)
            j = min(i + 1, n_steps - 1)
            a, b = steps[i]["true"], steps[j]["true"]
            if j == k0:  # never slide the robot across the kidnap
                return a
            f = value - i
            dth = (b[2] - a[2] + math.pi) % (2 * math.pi) - math.pi
            return [a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1]), a[2] + f * dth]

        def refresh(_mob) -> None:
            i = step_index()
            x, y, th = true_pose(t.get_value())
            for pnl in panels:
                w = pnl["world"]
                place_robot(pnl["tri"], w, x, y, th)
                s = steps[i]
                if s["valid"] and s["z"] is not None:
                    tip = (x + s["z"] * math.cos(th), y + s["z"] * math.sin(th))
                    pnl["beam"].put_start_and_end_on(w.p(x, y), w.p(*tip))
                    pnl["beam"].set_stroke(opacity=0.8)
                else:
                    pnl["beam"].set_stroke(opacity=0)
                if pnl["last"] == i:
                    continue
                pnl["last"] = i
                vd = pnl["data"]
                pts = vd["particles"][i]
                for k, dot in enumerate(pnl["dots"]):
                    if k < len(pts):
                        dot.move_to(w.p(*pts[k]))
                        dot.set_fill(opacity=0.55)
                    else:
                        dot.set_fill(opacity=0)
                pnl["est"].move_to(w.p(*vd["est"][i]))
                err, r95, n = vd["err"][i], vd["r95"][i], vd["n"][i]
                # Drawn radius capped at 60 cm so a spread cloud stays inside the panel; the HUD
                # always reports the exact value.
                ring = m.Circle(radius=max(min(r95, 60.0) * w.k, 0.06), color=COLORS[pnl["variant"]], stroke_width=2.5)
                ring.set_fill(COLORS[pnl["variant"]], opacity=0.1).move_to(w.p(*vd["est"][i]))
                pnl["halo"].become(ring)
                new = text(f"error {err:5.1f} cm   ·   95% radius {r95:5.1f} cm   ·   {n} particles", 18,
                           INK if err < 10 else INK2)
                new.next_to(pnl["walls"], m.DOWN, buff=0.3)
                pnl["hud"].become(new)
            label = "before the kidnap" if i < k0 else f"{i - k0} updates after the kidnap"
            counter.become(text(f"sonar update {i + 1}/{n_steps}  ·  {label}", 20, INK2)
                           .next_to(title, m.DOWN, buff=0.12))

        anchor = m.Dot(radius=0.001).set_fill(opacity=0)
        anchor.add_updater(refresh)
        self.add(anchor, counter)
        refresh(anchor)

        self.wait(0.6)
        self.play(t.animate.set_value(k0 - 1), run_time=(k0 - 1) / UPDATES_PER_SECOND, rate_func=m.linear)

        # The kidnap: show the displacement, then continue.
        kd = d["kidnap"]
        notes = []
        for pnl in panels:
            w = pnl["world"]
            arrow = m.Arrow(w.p(*kd["from"][:2]), w.p(*kd["to"][:2]), buff=0, color=TRUTH, stroke_width=4)
            notes.append(arrow)
        banner = text(f"Kidnap: the robot is picked up and moved {kd['displacement_cm']:.0f} cm. "
                      "Neither filter is told.", 22, TRUTH, "BOLD").to_edge(m.DOWN, buff=0.1)
        self.play(*[m.GrowArrow(a) for a in notes], m.FadeOut(legend), m.FadeIn(banner), run_time=0.8)
        self.play(t.animate.set_value(k0), run_time=0.3, rate_func=m.linear)
        self.wait(1.4)
        self.play(*[m.FadeOut(a) for a in notes], m.FadeOut(banner), m.FadeIn(legend), run_time=0.5)

        self.play(t.animate.set_value(n_steps - 1), run_time=(n_steps - 1 - k0) / UPDATES_PER_SECOND,
                  rate_func=m.linear)

        v1, v8 = d["variants"]["V1"], d["variants"]["V8"]
        verdict = text(f"V1 ends {v1['err'][-1]:.0f} cm off, still sure of itself (95% radius "
                       f"{v1['r95'][-1]:.0f} cm).   V8 recovers to {v8['err'][-1]:.0f} cm.", 20, INK,
                       "BOLD").to_edge(m.DOWN, buff=0.1)
        self.play(m.FadeOut(legend), m.FadeIn(verdict, shift=0.15 * m.UP), run_time=0.6)
        self.wait(2.5)

