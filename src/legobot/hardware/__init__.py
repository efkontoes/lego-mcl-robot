"""Robot backends: the real BrickPi3 robot or the simulator."""

from __future__ import annotations

from legobot.config import Config
from legobot.hardware.base import Backend
from legobot.hardware.sim_backend import SimBackend

BACKENDS = ("sim", "brickpi")


def make_backend(name: str, cfg: Config, **kwargs) -> Backend:
    """``sim`` kwargs go to :class:`SimBackend`, ``brickpi`` kwargs to
    :class:`BrickPiBackend` (``sonar=``, ``touch=``)."""
    if name == "sim":
        return SimBackend(cfg, **kwargs)
    if name == "brickpi":
        from legobot.hardware.brickpi_backend import BrickPiBackend

        return BrickPiBackend(cfg, **kwargs)
    raise ValueError(f"unknown backend {name!r}, expected one of {BACKENDS}")


__all__ = ["Backend", "SimBackend", "make_backend", "BACKENDS"]
