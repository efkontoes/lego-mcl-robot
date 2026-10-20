"""Practicals 3-4: waypoint navigation with Monte Carlo Localisation.

Drives the practical 4 course (or waypoints typed in with --interactive)
using the particle-filter estimate, updating it from the sonar after every
move. Particles and walls are printed in the course web-viewer format; in
simulation a GIF and a trajectory figure can also be written.
"""

import math

import numpy as np

from legobot import cli
from legobot.localisation import DEFAULT_VARIANT, VARIANTS, arena, build_filter
from legobot.navigation import navigate_to_waypoint
from legobot.viz.web_canvas import WebCanvas


def main() -> None:
    p = cli.parser(__doc__.splitlines()[0])
    p.add_argument("--variant", choices=list(VARIANTS), default=DEFAULT_VARIANT,
                   help="filter variant (see the noise study)")
    p.add_argument("--interactive", action="store_true", help="type waypoints in metres")
    p.add_argument("--web", action="store_true",
                   help="print drawLine/drawParticles for the web viewer (always on hardware)")
    p.add_argument("--gif", default=None, help="sim only: save an animation, e.g. mcl.gif")
    args = p.parse_args()

    world = arena()
    cfg, robot = cli.setup(args, sim_kwargs=lambda c: {"world": world, "pose": c.navigation.start})
    nav = cfg.navigation
    pf = build_filter(cfg, args.variant, world, rng=np.random.default_rng(args.seed + 1))
    pf.init_at(nav.start)

    web = WebCanvas() if (args.web or args.backend == "brickpi") else None
    recorder = None
    if args.gif and args.backend == "sim":
        from legobot.viz.plots import MclRecorder

        recorder = MclRecorder(robot.backend)

    def on_update(pf):
        if web:
            web.draw_particles(pf.particles)
        if recorder:
            recorder.record(pf)

    if web:
        web.draw_walls(world)
        web.draw_particles(pf.particles)

    try:
        for i, wp in enumerate(_waypoints(args, nav.waypoints)):
            ok = navigate_to_waypoint(robot, pf, wp, nav, on_update)
            x, y, th = pf.estimate()
            status = "reached" if ok else "gave up on"
            print(f"{status} waypoint {i + 1} {wp}: estimate ({x:.1f}, {y:.1f}, "
                  f"{math.degrees(th):.0f} deg)" + _truth(robot))
    except KeyboardInterrupt:
        pass
    finally:
        robot.close()

    if recorder and recorder.snapshots:
        from legobot.viz.plots import mcl_animation

        path = mcl_animation(recorder.snapshots, world, cli.output_path(args.gif), nav.waypoints,
                             title=f"MCL ({VARIANTS[args.variant].label})")
        print(f"saved {path}")


def _waypoints(args, default):
    if not args.interactive:
        yield from default
        return
    while True:
        text = input("waypoint x y in metres (blank to stop): ").strip()
        if not text:
            return
        wx, wy = map(float, text.split())
        yield wx * 100, wy * 100


def _truth(robot) -> str:
    pose = getattr(robot.backend, "pose", None)
    if pose is None:
        return ""
    return f"   true ({pose[0]:.1f}, {pose[1]:.1f}, {math.degrees(pose[2]):.0f} deg)"


if __name__ == "__main__":
    main()
