"""Matplotlib figures and GIFs for the simulator and the noise study."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402

from legobot.localisation.world_map import WorldMap  # noqa: E402


def draw_world(ax, world: WorldMap, waypoints: Optional[Sequence] = None) -> None:
    for x0, y0, x1, y1 in world.walls:
        ax.plot([x0, x1], [y0, y1], color="#222", lw=2.5, solid_capstyle="round")
    if waypoints is not None:
        wp = np.asarray(waypoints)
        ax.plot(wp[:, 0], wp[:, 1], "o", ms=6, mfc="none", mec="#2a7", mew=1.5, label="waypoints")
    pad = 10
    ax.set_xlim(world.x_range[0] - pad, world.x_range[1] + pad)
    ax.set_ylim(world.y_range[0] - pad, world.y_range[1] + pad)
    ax.set_aspect("equal")
    ax.set_xlabel("x (cm)")
    ax.set_ylabel("y (cm)")


def draw_particles(ax, particles: np.ndarray, color="#d33", arrow_len=4.0):
    ax.quiver(particles[:, 0], particles[:, 1], np.cos(particles[:, 2]) * arrow_len,
              np.sin(particles[:, 2]) * arrow_len, color=color, angles="xy", scale_units="xy",
              scale=1, width=0.003, alpha=0.6)


@dataclass
class Snapshot:
    particles: np.ndarray
    true_pose: np.ndarray
    estimate: tuple


@dataclass
class MclRecorder:
    """Collects filter snapshots during a run (pass ``recorder.record`` as a callback)."""

    backend: object
    snapshots: List[Snapshot] = field(default_factory=list)

    def record(self, pf) -> None:
        self.snapshots.append(Snapshot(pf.particles.copy(), self.backend.pose.copy(), pf.estimate()))


def mcl_animation(snapshots: List[Snapshot], world: WorldMap, path: Path, waypoints=None,
                  title: str = "Monte Carlo Localisation", fps: int = 6) -> Path:
    fig, ax = plt.subplots(figsize=(5.5, 5.5), dpi=90)

    def frame(i):
        ax.clear()
        draw_world(ax, world, waypoints)
        s = snapshots[i]
        draw_particles(ax, s.particles)
        truth = np.array([sn.true_pose[:2] for sn in snapshots[: i + 1]])
        est = np.array([sn.estimate[:2] for sn in snapshots[: i + 1]])
        ax.plot(truth[:, 0], truth[:, 1], color="#1565c0", lw=1.5, label="true path")
        ax.plot(est[:, 0], est[:, 1], "--", color="#ef6c00", lw=1.2, label="MCL estimate")
        ax.legend(loc="upper right", fontsize=7)
        ax.set_title(f"{title}  (step {i + 1}/{len(snapshots)})", fontsize=9)

    anim = FuncAnimation(fig, frame, frames=len(snapshots))
    path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return path


def trajectory_figure(world: WorldMap, true_path: np.ndarray, est_path: Optional[np.ndarray],
                      path: Path, waypoints=None, extra=None, title: str = "") -> Path:
    fig, ax = plt.subplots(figsize=(5.5, 5.5), dpi=110)
    draw_world(ax, world, waypoints)
    ax.plot(true_path[:, 0], true_path[:, 1], color="#1565c0", lw=1.5, label="true path")
    if est_path is not None:
        ax.plot(est_path[:, 0], est_path[:, 1], "--", color="#ef6c00", lw=1.2, label="estimate")
    if extra:
        extra(ax)
    ax.legend(loc="upper right", fontsize=7)
    ax.set_title(title, fontsize=9)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path
