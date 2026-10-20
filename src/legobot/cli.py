"""Shared command-line plumbing for the scripts in ``scripts/``."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, Optional, Union

import numpy as np

from legobot.config import Config, load_config
from legobot.hardware import BACKENDS, make_backend
from legobot.robot import Robot

OUTPUT_DIR = Path("outputs")


def parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--backend", choices=BACKENDS, default="sim",
                   help="run on the simulator (default) or the real BrickPi robot")
    p.add_argument("--config", type=Path, default=None, help="alternative robot.yaml")
    p.add_argument("--seed", type=int, default=0, help="simulator random seed")
    return p


def setup(args, sim_kwargs: Union[dict, Callable[[Config], dict], None] = None,
          hw_kwargs: Optional[dict] = None):
    """Load the config and build a :class:`Robot` on the chosen backend.

    ``sim_kwargs`` may be a function of the loaded config (e.g. to start the
    simulated robot at the configured start pose).
    """
    cfg: Config = load_config(args.config)
    if args.backend == "sim":
        extra = sim_kwargs(cfg) if callable(sim_kwargs) else (sim_kwargs or {})
        kwargs = {"rng": np.random.default_rng(args.seed), **extra}
    else:
        kwargs = hw_kwargs or {}
    return cfg, Robot(make_backend(args.backend, cfg, **kwargs), cfg)


def output_path(name: str) -> Path:
    OUTPUT_DIR.mkdir(exist_ok=True)
    return OUTPUT_DIR / name
