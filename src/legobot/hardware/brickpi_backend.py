"""Real robot: BrickPi3 motors/sensors on a Raspberry Pi.

Motions are open-loop and timed, exactly as calibrated during the practicals:
a straight run lasts ``distance * forward_factor / velocity`` seconds and a
turn ``arc_length / velocity + turn_offset`` seconds.
"""

from __future__ import annotations

import math
import time
from typing import Optional, Tuple

from legobot.config import Config


class BrickPiBackend:
    def __init__(self, cfg: Config, sonar: bool = True, touch: bool = False):
        from legobot.hardware.vendor import brickpi3  # opens the SPI bus on import

        self._brickpi3 = brickpi3
        self.cfg = cfg
        self.bp = brickpi3.BrickPi3()
        self.left = getattr(self.bp, f"PORT_{cfg.ports.left_motor}")
        self.right = getattr(self.bp, f"PORT_{cfg.ports.right_motor}")
        self.sonar_port = getattr(self.bp, f"PORT_{cfg.ports.sonar}")
        self.touch_ports = (
            getattr(self.bp, f"PORT_{cfg.ports.left_touch}"),
            getattr(self.bp, f"PORT_{cfg.ports.right_touch}"),
        )
        if sonar:
            self.bp.set_sensor_type(self.sonar_port, self.bp.SENSOR_TYPE.NXT_ULTRASONIC)
        if touch:
            for port in self.touch_ports:
                self.bp.set_sensor_type(port, self.bp.SENSOR_TYPE.TOUCH)
        self._camera = None
        self.set_speed_limit(cfg.robot.motor_power_limit, cfg.robot.drive_dps)
        self.reset_encoders()

    # ---------------------------------------------------------------- motors
    def reset_encoders(self) -> None:
        for port in (self.left, self.right):
            self.bp.offset_motor_encoder(port, self.bp.get_motor_encoder(port))

    def encoders(self) -> Tuple[int, int]:
        return self.bp.get_motor_encoder(self.left), self.bp.get_motor_encoder(self.right)

    def set_speed_limit(self, power: int, dps: float) -> None:
        for port in (self.left, self.right):
            self.bp.set_motor_limits(port, power, dps)

    def set_wheel_speeds(self, left_dps: float, right_dps: float) -> None:
        self.bp.set_motor_dps(self.left, left_dps)
        self.bp.set_motor_dps(self.right, right_dps)

    def float_motors(self) -> None:
        """Let the wheels spin freely (used to measure degrees per cm by hand)."""
        for port in (self.left, self.right):
            self.bp.set_motor_power(port, self.bp.MOTOR_FLOAT)

    def stop(self) -> None:
        self.set_wheel_speeds(0, 0)

    def _timed(self, left_dps: float, right_dps: float, duration_s: float) -> None:
        if duration_s <= 0:
            return
        self.set_wheel_speeds(left_dps, right_dps)
        time.sleep(duration_s)
        self.stop()

    def move_straight(self, distance_cm: float) -> None:
        r = self.cfg.robot
        dps = math.copysign(r.drive_dps, distance_cm)
        self._timed(dps, dps, abs(distance_cm) * r.forward_factor / r.velocity_cm_s)

    def rotate(self, angle_rad: float) -> None:
        r = self.cfg.robot
        if angle_rad == 0:
            return
        arc_cm = abs(angle_rad) * r.wheel_base_cm / 2
        duration = arc_cm / r.velocity_cm_s + r.turn_offset_s
        dps = math.copysign(r.drive_dps, angle_rad)
        self._timed(-dps, dps, duration)  # right wheel forward -> counter-clockwise

    def wait(self, seconds: float) -> None:
        time.sleep(seconds)

    # --------------------------------------------------------------- sensors
    def read_sonar(self) -> Optional[float]:
        try:
            return float(self.bp.get_sensor(self.sonar_port))
        except self._brickpi3.SensorError:
            return None

    def read_touch(self) -> Tuple[bool, bool]:
        try:
            return tuple(bool(self.bp.get_sensor(p)) for p in self.touch_ports)
        except self._brickpi3.SensorError:
            return False, False

    def capture(self):
        if self._camera is None:
            from legobot.vision.camera import PiCamera

            self._camera = PiCamera(self.cfg.camera)
        return self._camera.capture_bgr()

    def close(self) -> None:
        if self._camera is not None:
            self._camera.close()
        self.bp.reset_all()
