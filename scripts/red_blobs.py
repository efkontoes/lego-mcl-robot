"""Practical 5: detect red objects and locate them on the floor.

Thresholds the frame in HSV, finds connected components and projects each
blob's bottom-centre through the inverse homography to floor coordinates.
"""

from legobot import cli
from legobot.vision.detection import annotate, blobs_to_floor, detect_red_blobs
from legobot.vision.homography import GroundHomography

SIM_CANS = [(40, 10), (70, -15), (95, 20)]


def main() -> None:
    p = cli.parser(__doc__.splitlines()[0])
    p.add_argument("--frames", type=int, default=None,
                   help="frames to process (default: forever on hardware, 1 in sim)")
    p.add_argument("--out", default="red_blobs.jpg")
    args = p.parse_args()
    frames = args.frames or (1 if args.backend == "sim" else None)
    cfg, robot = cli.setup(args, sim_kwargs={"cans": SIM_CANS}, hw_kwargs={"sonar": False})
    homography = GroundHomography.from_config(cfg.camera)

    import cv2

    path = cli.output_path(args.out)
    try:
        frame_no = 0
        while frames is None or frame_no < frames:
            frame = robot.camera_frame()
            blobs = detect_red_blobs(frame, cfg.camera)
            floor, kept = blobs_to_floor(blobs, homography, 0, float("inf"))
            for blob, (x, y) in zip(kept, floor):
                print(f"frame {frame_no}: blob area {blob.area:5d} px at pixel ({blob.u}, {blob.v})"
                      f" -> floor ({x:.1f}, {y:.1f}) cm")
            cv2.imwrite(str(path), annotate(frame, blobs))
            print(f"drawImg:{path.resolve()}")
            frame_no += 1
    except KeyboardInterrupt:
        pass
    finally:
        robot.close()


if __name__ == "__main__":
    main()
