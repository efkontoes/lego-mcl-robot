"""Practical 1: measure encoder degrees per cm.

Lets both wheels spin freely and prints the encoder counts; push the robot
along a ruler and divide the count by the distance. Hardware only.
"""

from legobot import cli


def main() -> None:
    args = cli.parser(__doc__.splitlines()[0]).parse_args()
    cfg, robot = cli.setup(args)
    if args.backend == "sim":
        r = cfg.robot
        print("Encoder calibration needs the real robot (--backend brickpi).")
        print(f"Configured: {r.degrees_per_cm} deg/cm -> {100 * r.degrees_per_cm:.0f} deg per metre")
        return
    backend = robot.backend
    backend.float_motors()
    try:
        while True:
            left, right = backend.encoders()
            print(f"left {left:7d} deg   right {right:7d} deg")
            backend.wait(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        robot.close()


if __name__ == "__main__":
    main()
