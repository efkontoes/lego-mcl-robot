"""Practical 2: hold a fixed sonar distance to the wall ahead (P-control)."""

from legobot import cli
from legobot.behaviours import wall_follow
from legobot.localisation import arena


def main() -> None:
    p = cli.parser(__doc__.splitlines()[0])
    p.add_argument("--target", type=float, default=None, help="distance to hold (cm)")
    p.add_argument("--steps", type=int, default=None,
                   help="control steps (default: forever on hardware, 40 in sim)")
    args = p.parse_args()
    steps = args.steps or (40 if args.backend == "sim" else None)
    # Simulated start: 100 cm from wall g (x = 210), facing it.
    cfg, robot = cli.setup(args, sim_kwargs={"world": arena(), "pose": (110, 40, 0.0)})

    def log(reading, avg_error):
        print(f"sonar {reading} cm   smoothed error {avg_error:+6.1f} cm")

    try:
        wall_follow(robot, args.target, max_steps=steps, on_step=log)
    except KeyboardInterrupt:
        pass
    finally:
        robot.close()


if __name__ == "__main__":
    main()
