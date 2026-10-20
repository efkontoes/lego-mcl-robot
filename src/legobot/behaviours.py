"""Reactive behaviours from practicals 1 and 2."""

from __future__ import annotations

from collections import deque
from typing import Callable, Optional

from legobot.robot import Robot


def drive_square(robot: Robot, side_cm: float, on_side: Optional[Callable[[int], None]] = None):
    """Practical 1: drive a square of ``side_cm`` and come back to the start."""
    for side in range(4):
        robot.forward(side_cm)
        robot.turn_left()
        if on_side:
            on_side(side)


def bumper_avoid(robot: Robot, max_steps: Optional[int] = None) -> int:
    """Practical 2: drive forward in short steps; when a bumper is pressed,
    back off and turn away from the side that was hit. Returns the number of
    obstacles hit."""
    b = robot.cfg.behaviours
    hits = 0
    step = 0
    while max_steps is None or step < max_steps:
        left, right = robot.bumpers()
        if left or right:
            hits += 1
            robot.stop()
            robot.reverse(b.bumper_reverse_cm)
            if left:  # hit on the left (or both): turn right
                robot.turn_right()
            else:
                robot.turn_left()
        else:
            robot.forward(b.bumper_step_cm)
        step += 1
    return hits


def wall_follow(robot: Robot, target_cm: Optional[float] = None, max_steps: Optional[int] = None,
                on_step: Optional[Callable[[float, float], None]] = None) -> None:
    """Practical 2: keep a fixed sonar distance to the wall ahead with a
    proportional controller on a moving average of the distance error."""
    b = robot.cfg.behaviours
    target = b.wall_follow_target_cm if target_cm is None else target_cm
    limit = robot.cfg.robot.drive_dps
    errors = deque(maxlen=b.wall_follow_window)
    step = 0
    while max_steps is None or step < max_steps:
        reading = robot.sonar_raw()
        if reading is not None:
            errors.append(reading - target)
        if errors:
            avg_error = sum(errors) / len(errors)
            speed = max(-limit, min(limit, b.wall_follow_kp * avg_error))
            robot.set_wheel_speeds(speed, speed)
            if on_step:
                on_step(reading, avg_error)
        robot.wait(b.wall_follow_period_s)
        step += 1
    robot.stop()
