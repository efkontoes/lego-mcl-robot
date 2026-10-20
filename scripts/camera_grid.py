"""Practical 5: overlay the calibrated 10 cm floor grid on camera frames.

If the homography in config/robot.yaml is right, the grid lines sit on the
floor where a tape measure says they should. In simulation a rendered frame
with a few cans is used instead of the camera.
"""

from legobot import cli
from legobot.vision.homography import GroundHomography, draw_ground_grid

SIM_CANS = [(40, 10), (70, -15), (95, 20)]


def main() -> None:
    p = cli.parser(__doc__.splitlines()[0])
    p.add_argument("--frames", type=int, default=None,
                   help="frames to process (default: forever on hardware, 1 in sim)")
    p.add_argument("--out", default="camera_grid.jpg")
    args = p.parse_args()
    frames = args.frames or (1 if args.backend == "sim" else None)
    cfg, robot = cli.setup(args, sim_kwargs={"cans": SIM_CANS}, hw_kwargs={"sonar": False})
    homography = GroundHomography.from_config(cfg.camera)
    print("Homography H =\n", homography.H.round(4))

    import cv2

    path = cli.output_path(args.out)
    try:
        i = 0
        while frames is None or i < frames:
            image = draw_ground_grid(robot.camera_frame(), homography)
            cv2.imwrite(str(path), image)
            print(f"drawImg:{path.resolve()}")
            i += 1
    except KeyboardInterrupt:
        pass
    finally:
        robot.close()


if __name__ == "__main__":
    main()
