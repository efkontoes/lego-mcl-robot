"""Practical 1: drive a 40 cm square and measure the end-point error.

On hardware the robot drives the square once (measure where it stops). In
simulation it repeats the square and prints the end-point error covariance,
reproducing the practical 1 experiment.
"""

import numpy as np

from legobot import cli
from legobot.behaviours import drive_square
from legobot.calibration import error_covariance
from legobot.hardware import SimBackend
from legobot.robot import Robot


def main() -> None:
    p = cli.parser(__doc__.splitlines()[0])
    p.add_argument("--side", type=float, default=None, help="side length in cm")
    p.add_argument("--repeats", type=int, default=10, help="simulated repetitions")
    args = p.parse_args()
    cfg, robot = cli.setup(args)
    side = args.side or cfg.behaviours.square_side_cm

    if args.backend == "brickpi":
        try:
            drive_square(robot, side)
        finally:
            robot.close()
        return

    ends = []
    rng = np.random.default_rng(args.seed)
    for _ in range(args.repeats):
        sim = SimBackend(cfg, rng=rng)
        drive_square(Robot(sim, cfg), side)
        ends.append(sim.pose[:2].copy())
    ends = np.array(ends)
    print(f"{args.repeats} simulated squares of {side:.0f} cm, end-point errors (cm):")
    for x, y in ends:
        print(f"  ({x:+6.2f}, {y:+6.2f})")
    print("covariance matrix:\n", np.round(error_covariance(ends[:, 0], ends[:, 1]), 3))


if __name__ == "__main__":
    main()
