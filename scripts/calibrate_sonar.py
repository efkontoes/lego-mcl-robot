"""Practical 2: sonar accuracy at known distances (bias, spread, garbage rate).

On hardware, place the robot facing a wall at each test distance and press
ENTER. In simulation the robot is placed automatically.
"""

import csv

from legobot import cli
from legobot.calibration import sonar_stats
from legobot.localisation import arena

DISTANCES_CM = (20, 40, 60, 80, 100)


def main() -> None:
    p = cli.parser(__doc__.splitlines()[0])
    p.add_argument("--samples", type=int, default=10)
    p.add_argument("--out", default="sonar_calibration_results.csv")
    args = p.parse_args()
    world = arena()
    cfg, robot = cli.setup(args, sim_kwargs={"world": world})

    results = []
    try:
        for dist in DISTANCES_CM:
            if args.backend == "sim":
                # Sensor face `dist` cm from wall g (x = 210), facing it.
                robot.backend.teleport((210 - dist - cfg.sonar.offset_cm, 40, 0.0))
            else:
                input(f"Place the sonar {dist} cm from a wall and press ENTER...")
            readings = []
            for _ in range(args.samples):
                readings.append(robot.sonar_raw())
                robot.wait(0.25)
            s = sonar_stats(readings, dist)
            results.append((s, readings))
            if s.mean_cm is None:
                print(f"{dist:4d} cm: all readings invalid")
            else:
                print(f"{dist:4d} cm: mean {s.mean_cm:6.2f}  std {s.std_cm:5.2f}  "
                      f"bias {s.bias_cm:+5.2f}  garbage {s.garbage_fraction:.0%}")
    finally:
        robot.close()

    path = cli.output_path(args.out)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["true_distance_cm", "mean_cm", "std_cm", "bias_cm", "garbage_fraction",
                    "raw_readings"])
        for s, readings in results:
            w.writerow([s.true_cm, s.mean_cm, s.std_cm, s.bias_cm, s.garbage_fraction, readings])
    print(f"saved {path}")


if __name__ == "__main__":
    main()
