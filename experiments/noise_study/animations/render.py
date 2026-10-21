"""Render the ManimGX scenes to MP4 and convert each to a GIF.

Needs a Python 3.13+ environment with ``manimgx``, ``numpy``, ``pillow`` and
``opencv-python-headless`` (see animations/README.md; on a machine without a
working GPU driver, run it under Linux/WSL, where manimgx renders on the CPU).

    python render.py              # all scenes
    python render.py kld_sampling # one scene
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import cv2
from PIL import Image

HERE = Path(__file__).resolve().parent
SCENES = HERE / "scenes"
OUT = HERE.parent / "results" / "animations"
RESOLUTION, FPS = "1280x720", 24
GIF_WIDTH, GIF_FPS, GIF_COLORS = 800, 12, 96

ORDER = ["kidnap_replay", "budget_reveal", "kld_sampling", "global_sweep"]


def manimgx_bin() -> str:
    exe = Path(sys.executable).with_name("manimgx")
    return str(exe) if exe.exists() else "manimgx"


def render(name: str) -> Path:
    mp4 = OUT / f"{name}.mp4"
    t0 = time.time()
    subprocess.run([manimgx_bin(), "render", str(SCENES / f"{name}.py"), "-o", str(mp4), "-r", RESOLUTION,
                    "--fps", str(FPS), "--crf", "20"], check=True, cwd=SCENES)
    print(f"{name}: {mp4.stat().st_size / 1e6:.1f} MB mp4 in {time.time() - t0:.0f}s", flush=True)
    for board in OUT.glob(f"{name}.storyboard*.png"):  # render also writes a storyboard; not needed here
        board.unlink()
    return mp4


def to_gif(mp4: Path) -> Path:
    cap = cv2.VideoCapture(str(mp4))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or FPS
    step = max(1, round(src_fps / GIF_FPS))
    frames, i = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % step == 0:
            h, w = frame.shape[:2]
            frame = cv2.resize(frame, (GIF_WIDTH, round(h * GIF_WIDTH / w)), interpolation=cv2.INTER_AREA)
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        i += 1
    cap.release()
    # One shared palette keeps colours stable from frame to frame and the file small.
    palette = frames[len(frames) // 2].quantize(colors=GIF_COLORS, method=Image.Quantize.MEDIANCUT)
    gif_frames = [f.quantize(palette=palette, dither=Image.Dither.NONE) for f in frames]
    gif = mp4.with_suffix(".gif")
    gif_frames[0].save(gif, save_all=True, append_images=gif_frames[1:], duration=round(1000 * step / src_fps),
                       loop=0, optimize=True, disposal=1)
    print(f"{mp4.stem}: {gif.stat().st_size / 1e6:.1f} MB gif ({len(frames)} frames)", flush=True)
    return gif


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name in (sys.argv[1:] or ORDER):
        to_gif(render(name))


if __name__ == "__main__":
    main()
