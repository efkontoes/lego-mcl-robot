# From coursework to this repository

This project grew out of the six practicals of the Imperial College robotics course,
done as a group on a BrickPi3 LEGO robot. The table maps each practical to where its
code lives now.

| Practical | Topic | Original files | Now |
|---|---|---|---|
| 1 | Open-loop motion, calibration | `calibration.py`, `position.py`, `velocity.py`, `robot_square.py`, `cov_matrix.py` | `scripts/calibrate_motors.py`, `scripts/drive_square.py`, `legobot/calibration.py` (`error_covariance`) |
| 2 | Touch and sonar feedback | `robot_touch.py`, `robot_sonar.py`, `sonar_calibration.py` | `legobot/behaviours.py` (`bumper_avoid`, `wall_follow`), `scripts/calibrate_sonar.py` |
| 3 | Probabilistic motion, particles | `particle_manager.py`, `square_with_particles.py`, `waypoint_nav.py`, `robotp3.py`, `robot_particles.py` | `legobot/localisation/` (`motion_models.py`, `particle_filter.py`), `scripts/mcl_navigate.py --interactive` |
| 4 | Monte Carlo Localisation | `main_navigation.py`, `sonar_update.py` | `legobot/localisation/sensor_model.py`, `legobot/navigation.py`, `scripts/mcl_navigate.py` |
| 5 | Camera homography, colour blobs | `camera.py`, `homography_grid.py`, `red_blobs.py` | `legobot/vision/` (`homography.py`, `detection.py`, `camera.py`), `scripts/camera_grid.py`, `scripts/red_blobs.py` |
| 6 | Vision-based obstacle challenge | `challenge.py`, `utils.py`, `constants.py`, `simple_challenge.py` | `legobot/planning/dwa.py`, `scripts/dwa_challenge.py` |

## What changed

**One implementation per concept.**
- `robot.py`, `particle_manager.py` and `waypoint_nav.py` each existed twice.
- The sonar likelihood existed twice.
- The homography solve existed three times.

Each now exists once.

**One parameter file.** [`config/robot.yaml`](../config/robot.yaml) replaces the
constants that were scattered across files and disagreed with each other:

| Parameter | Before | Now |
|---|---|---|
| Velocity | 9.5 vs 9.17 cm/s | 9.17 |
| Wheel base | 14 vs 15.25 vs 15 cm | 15.25 |
| Ports | A/C/4 vs B/C/1 | A/C/4 |
| Trusted sonar range | 0–140 vs 20–140 cm | 20–140 |
| DWA speed conversion | separately hard-coded | derived from the wheel calibration |

**A hardware abstraction and a simulator.**
- Every behaviour talks to a `Backend`: the real BrickPi3 or `SimBackend`.
- Timing calibration (turn offset, 0.97 forward factor) lives only in the BrickPi
  backend, because it describes the physical motors.
- The simulator renders camera frames, so the vision pipeline is tested end to end.

**Units.** Centimetres and radians everywhere. The challenge code previously used metres.

**Bugs fixed:**
- The compounding motion noise in the waypoint script. See [noise_study.md](noise_study.md).
- A possible index overflow in resampling.
- Particle outlines and walls drawn at two different scales in the web viewer.
- A loop variable shadowed in `red_blobs.py`, which printed the wrong frame number.
- A hard-coded "/121 trajectories" message (there are 169).
- The DWA goal (3.5 m) not matching the stop condition (3.0 m). Both are now config values.
- `Map.draw()` relying on a global canvas.
- `robotp3.py` calling `forward()` with an argument that doesn't exist.

**Removed:**
- The vendor example scripts (`Examples/`).
- Scratch and deprecated files (`robotp3.py`, `robot_particles.py`, `simple_challenge.py`).
- Stub functions.
- The GitLab template README.
- Committed `__pycache__` folders.

The BrickPi3 driver is vendored unchanged under `src/legobot/hardware/vendor/` (MIT).

**Behaviour change to check on the robot.** `Robot.turn(+angle)` is counter-clockwise
(right wheel forward), the standard convention that the DWA kinematics already used.
The practical 1–4 code turned "left" by driving the left motor forward instead. If the
robot spins the wrong way, swap `left_motor`/`right_motor` in `config/robot.yaml`.
