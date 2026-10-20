"""Practical 2: drive around, backing off and turning away on bumper hits."""

from legobot import cli
from legobot.behaviours import bumper_avoid
from legobot.localisation import arena


def main() -> None:
    p = cli.parser(__doc__.splitlines()[0])
    p.add_argument("--steps", type=int, default=None,
                   help="number of control steps (default: forever on hardware, 80 in sim)")
    args = p.parse_args()
    steps = args.steps or (80 if args.backend == "sim" else None)
    cfg, robot = cli.setup(args, sim_kwargs={"world": arena(), "pose": (105, 40, 1.2)},
                           hw_kwargs={"sonar": False, "touch": True})
    try:
        hits = bumper_avoid(robot, max_steps=steps)
        print(f"obstacles hit: {hits}")
    except KeyboardInterrupt:
        pass
    finally:
        robot.close()


if __name__ == "__main__":
    main()
