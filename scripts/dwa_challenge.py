"""Practical 6 challenge: drive 3 m past red cans using the camera and DWA.

In simulation the cans are placed at random along the course, camera frames
are rendered and pass through the real detection pipeline, and a trajectory
figure (plus an optional annotated video) is written to outputs/.
"""

import math

import numpy as np

from legobot import cli
from legobot.planning.dwa import run_challenge


def random_cans(rng: np.random.Generator, n: int = 6):
    """Cans scattered across the 3 m course, at least 25 cm apart."""
    cans = []
    while len(cans) < n:
        c = (rng.uniform(50, 280), rng.uniform(-35, 35))
        if all(math.dist(c, o) > 25 for o in cans):
            cans.append(c)
    return cans


def main() -> None:
    p = cli.parser(__doc__.splitlines()[0])
    p.add_argument("--cans", type=int, default=6, help="sim only: number of cans")
    p.add_argument("--video", default=None, help="save annotated frames, e.g. challenge.mp4")
    args = p.parse_args()
    cans = random_cans(np.random.default_rng(args.seed), args.cans)
    cfg, robot = cli.setup(args, sim_kwargs={"cans": cans}, hw_kwargs={"sonar": False})

    writer = None
    if args.video:
        import cv2

        writer = cv2.VideoWriter(str(cli.output_path(args.video)), cv2.VideoWriter_fourcc(*"mp4v"),
                                 int(round(1 / cfg.dwa.dt_s)), (cfg.camera.width, cfg.camera.height))

    def on_cycle(cycle, frame, blobs, pose, plan):
        if cycle % 10 == 0:
            print(f"cycle {cycle:3d}: odometry x={pose[0]:6.1f} y={pose[1]:6.1f} cm, "
                  f"{len(blobs)} cans in view, v=({plan.v_left:5.1f}, {plan.v_right:5.1f}) cm/s, "
                  f"{plan.collisions}/{plan.candidates} candidate paths collide")
        if writer is not None:
            from legobot.vision.detection import annotate

            writer.write(annotate(frame, blobs, (0, 255, 0)))

    try:
        log = run_challenge(robot, cfg, realtime=args.backend == "brickpi", on_cycle=on_cycle)
    finally:
        robot.close()
        if writer is not None:
            writer.release()

    x, y, _ = log.poses[-1]
    print(f"finished after {len(log.plans)} cycles at odometry x={x:.1f} cm (finish line "
          f"{cfg.dwa.finish_x_cm:.0f} cm)")
    if args.backend == "sim":
        sim = robot.backend
        print(f"true position ({sim.pose[0]:.1f}, {sim.pose[1]:.1f}) cm, can contacts: "
              f"{sim.collisions}")
        _plot(sim, log, cfg, cans)


def _plot(sim, log, cfg, cans):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    true = np.array(sim.trajectory)
    odo = np.array(log.poses)
    fig, ax = plt.subplots(figsize=(8, 3), dpi=110)
    for cx, cy in cans:
        ax.add_patch(plt.Circle((cx, cy), cfg.dwa.obstacle_radius_cm, color="#d32f2f"))
        ax.add_patch(plt.Circle((cx, cy), cfg.dwa.obstacle_radius_cm + cfg.robot.radius_cm,
                                color="#d32f2f", fill=False, ls=":", lw=0.8))
    ax.plot(true[:, 0], true[:, 1], color="#1565c0", label="true path")
    ax.plot(odo[:, 0], odo[:, 1], "--", color="#ef6c00", label="odometry")
    ax.axvline(cfg.dwa.finish_x_cm, color="#2e7d32", lw=1, label="finish line")
    ax.set_aspect("equal")
    ax.set_xlabel("x (cm)")
    ax.set_ylabel("y (cm)")
    ax.legend(fontsize=7, loc="upper left")
    path = cli.output_path("dwa_challenge.png")
    fig.savefig(path, bbox_inches="tight")
    print(f"saved {path}")


if __name__ == "__main__":
    main()
