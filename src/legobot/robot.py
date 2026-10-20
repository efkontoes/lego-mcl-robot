"""High-level robot API used by every behaviour and script."""

from __future__ import annotations

import math
from typing import Optional, Tuple

from legobot.config import Config
from legobot.hardware.base import Backend


class Robot:
    def __init__(self, backend: Backend, cfg: Config):
        self.backend = backend
        self.cfg = cfg

    # ---------------------------------------------------------------- motion
    def forward(self, distance_cm: float) -> None:
        self.backend.move_straight(distance_cm)

    def reverse(self, distance_cm: float) -> None:
        self.backend.move_straight(-distance_cm)

    def turn(self, angle_rad: float) -> None:
        """Turn on the spot; positive = left (counter-clockwise)."""
        self.backend.rotate(angle_rad)

    def turn_left(self, angle_rad: float = math.pi / 2) -> None:
        self.turn(abs(angle_rad))

    def turn_right(self, angle_rad: float = math.pi / 2) -> None:
        self.turn(-abs(angle_rad))

    def set_wheel_speeds(self, left_dps: float, right_dps: float) -> None:
        self.backend.set_wheel_speeds(left_dps, right_dps)

    def set_speed_limit(self, power: int, dps: float) -> None:
        self.backend.set_speed_limit(power, dps)

    def stop(self) -> None:
        self.backend.stop()

    def wait(self, seconds: float) -> None:
        self.backend.wait(seconds)

    # --------------------------------------------------------------- sensors
    def sonar_raw(self) -> Optional[float]:
        return self.backend.read_sonar()

    def sonar_from_centre(self) -> Optional[float]:
        """Sonar distance measured from the robot centre (raw + sensor offset)."""
        raw = self.backend.read_sonar()
        return None if raw is None else raw + self.cfg.sonar.offset_cm

    def bumpers(self) -> Tuple[bool, bool]:
        return self.backend.read_touch()

    def camera_frame(self):
        return self.backend.capture()

    def close(self) -> None:
        self.backend.stop()
        self.backend.close()
