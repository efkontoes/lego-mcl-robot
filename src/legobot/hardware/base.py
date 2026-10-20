"""The interface every robot backend (real BrickPi or simulator) implements."""

from __future__ import annotations

from typing import Optional, Protocol, Tuple

import numpy as np


class Backend(Protocol):
    def move_straight(self, distance_cm: float) -> None:
        """Drive straight (negative = reverse) and stop. Blocks until done."""

    def rotate(self, angle_rad: float) -> None:
        """Turn on the spot, positive = counter-clockwise (left). Blocks."""

    def set_wheel_speeds(self, left_dps: float, right_dps: float) -> None:
        """Continuous wheel speed command in motor degrees per second."""

    def set_speed_limit(self, power: int, dps: float) -> None:
        """Motor power / speed limits (the DWA challenge drives faster)."""

    def stop(self) -> None: ...

    def wait(self, seconds: float) -> None:
        """Let time pass (real sleep, or advance the simulation clock)."""

    def read_sonar(self) -> Optional[float]:
        """Raw sonar reading in cm from the sensor face, ``None`` on error."""

    def read_touch(self) -> Tuple[bool, bool]:
        """(left, right) bumper states."""

    def capture(self) -> np.ndarray:
        """A BGR camera frame."""

    def close(self) -> None: ...
