# Noise study animations

These four animations are made with [ManimGX](https://github.com/academa-labs/manimgx), a
GPU animation engine with Manim Community's Python API. They are built **only from the
saved study results**: no simulation is re-run.

| Animation | Shows | Data |
|---|---|---|
| [`kidnap_replay`](../results/animations/kidnap_replay.gif) | V1 and V8 fed the *same* recorded sonar data through a kidnap. V1 stays confidently lost; V8 recovers. | seed 1000, Protocol A; per-update logs and particle snapshots |
| [`budget_reveal`](../results/animations/budget_reveal.gif) | Recovery vs CPU, built up step by step. The fixed-budget controls show the injection × KLD gain is a particle-budget effect. | `results/analysis/main_summary.csv` |
| [`kld_sampling`](../results/animations/kld_sampling.gif) | The real `kld_resample` run on a tight and a spread particle cloud: samples drawn, bins filled, stopping rule. | two saved V8 particle snapshots (seed 1000) |
| [`global_sweep`](../results/animations/global_sweep.gif) | The 12-reading sonar sweep sharpening the start-position likelihood, while V1's particles miss the true start. | seed 1000 global trace and V1 snapshots |

MP4 and GIF versions are in [`results/animations/`](../results/animations/).

## Reproduce

**Step 1. Extract the animation data** (project environment; reads saved results only):

```bash
python experiments/noise_study/animations/prepare_data.py   # -> animations/data/
```

**Step 2. Render** (needs Python 3.13+ with ManimGX):

```bash
pip install -e ".[animation]"        # manimgx, pillow, opencv
cd experiments/noise_study/animations
python render.py                     # all scenes -> results/animations/*.mp4 and *.gif
python render.py kidnap_replay       # one scene
```

Useful while editing a scene, in `scenes/`:

```bash
manimgx inspect kidnap_replay.py -t 4,end
```

This writes a storyboard PNG and reports layout problems such as overlapping text.

**No working GPU?** ManimGX renders on the GPU through wgpu. On the machine used here
(Intel UHD, 2023 driver, Windows), it crashed while bringing the GPU up. Its Linux wheels
bundle a CPU Vulkan renderer (lavapipe), so the animations were rendered under WSL with
Ubuntu 24.04 and Python 3.13 (installed with `uv`), each in under 30 s. Updating the GPU
driver may also fix rendering on Windows directly.

## Notes

- Every number shown in an animation is read from the data files, not typed by hand.
- **Drawing simplifications:**
  - At most 150 particles are drawn per frame (snapshots store up to 300).
  - The 95% circle in `kidnap_replay` is drawn with its radius capped at 60 cm, so a
    spread cloud stays inside its panel; the panel's status line shows the exact value.
  - The likelihood maps in `global_sweep` show log-likelihood relative to the best
    start, floored at −30 nats.
- **Licensing:** ManimGX is MIT-licensed, but its native wheels bundle x264
  (GPL-3.0-or-later). It is used only as an optional rendering tool. The rendered videos
  are this project's own output.
