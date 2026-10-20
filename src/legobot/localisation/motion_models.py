"""Particle motion models.

Every robot motion is expressed as ``Motion(rot1, trans, rot2)``: turn by
``rot1``, drive ``trans`` cm along the new heading, then turn by ``rot2``. The
robot in this project only ever turns on the spot or drives straight, so in
practice one of ``rot1``/``trans`` is zero.

Two models are provided:

* :class:`CourseworkModel` - the model from the original practicals (kept as
  the baseline of the noise study, including an optional reproduction of the
  "compounding sigma" bug from the coursework waypoint script).
* :class:`OdometryModel` - ``sample_motion_model_odometry`` from Thrun, Burgard
  & Fox, *Probabilistic Robotics* (2005), Table 5.6, the same model ROS AMCL
  uses for differential-drive robots (``amcl_odom.cpp``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from legobot.config import CourseworkNoise, OdometryAlphas


@dataclass(frozen=True)
class Motion:
    rot1: float = 0.0
    trans: float = 0.0
    rot2: float = 0.0

    @classmethod
    def turn(cls, angle: float) -> Motion:
        return cls(rot1=angle)

    @classmethod
    def straight(cls, distance: float) -> Motion:
        return cls(trans=distance)


class CourseworkModel:
    """Motion model of the original coursework ``ParticleManager``.

    * Straight move of D cm: ``e ~ N(0, sigma_e)``, ``f ~ N(0, sigma_f)`` are both
      multiplied by ``D / 10`` (the sigmas were measured for 10 cm moves), then
      ``x += (D + e) cos(theta)``, ``y += (D + e) sin(theta)``, ``theta += f``.
    * Turn of any angle: ``theta += angle + g`` with ``g ~ N(0, sigma_g)`` - the
      noise does not depend on how far the robot turns.

    ``compounding=True`` reproduces the bug in the coursework ``waypoint_nav.py``
    which also did ``sigma_e *= D / 10`` (and the same for ``sigma_f``) on the
    stored value before every move, so the noise grew geometrically.
    """

    name = "coursework"

    def __init__(self, noise: CourseworkNoise, compounding: bool = False):
        self.sigma_e = noise.sigma_e
        self.sigma_f = noise.sigma_f
        self.sigma_g = noise.sigma_g
        self.compounding = compounding

    def sample(self, particles: np.ndarray, motion: Motion, rng: np.random.Generator) -> np.ndarray:
        p = particles.copy()
        n = len(p)
        if motion.rot1:
            p[:, 2] += motion.rot1 + rng.normal(0.0, self.sigma_g, n)
        if motion.trans:
            d = motion.trans
            if self.compounding:
                self.sigma_e *= abs(d) / 10.0
                self.sigma_f *= abs(d) / 10.0
            e = rng.normal(0.0, self.sigma_e, n) * d / 10.0
            f = rng.normal(0.0, self.sigma_f, n) * d / 10.0
            p[:, 0] += (d + e) * np.cos(p[:, 2])
            p[:, 1] += (d + e) * np.sin(p[:, 2])
            p[:, 2] += f
        if motion.rot2:
            p[:, 2] += motion.rot2 + rng.normal(0.0, self.sigma_g, n)
        return p


class OdometryModel:
    """``sample_motion_model_odometry`` (Probabilistic Robotics, Table 5.6).

    Each of the three motion components gets zero-mean Gaussian noise whose
    *variance* is a weighted sum of the squared rotation and translation::

        rot1_hat  = rot1  + N(0, a1*rot1^2 + a2*trans^2)
        trans_hat = trans + N(0, a3*trans^2 + a4*(rot1^2 + rot2^2))
        rot2_hat  = rot2  + N(0, a1*rot2^2 + a2*trans^2)

    so the noise is proportional to the size of *this* motion only: a turn of
    10 degrees adds far less uncertainty than a turn of 90 degrees, and nothing
    accumulates inside the model between calls.
    """

    name = "odometry"

    def __init__(self, alphas: OdometryAlphas):
        self.a = alphas

    def sample(self, particles: np.ndarray, motion: Motion, rng: np.random.Generator) -> np.ndarray:
        a = self.a
        n = len(particles)
        r1, t, r2 = motion.rot1, motion.trans, motion.rot2
        rot1_hat = r1 + rng.normal(0.0, np.sqrt(a.a1 * r1**2 + a.a2 * t**2), n)
        trans_hat = t + rng.normal(0.0, np.sqrt(a.a3 * t**2 + a.a4 * (r1**2 + r2**2)), n)
        rot2_hat = r2 + rng.normal(0.0, np.sqrt(a.a1 * r2**2 + a.a2 * t**2), n)

        p = particles.copy()
        heading = p[:, 2] + rot1_hat
        p[:, 0] += trans_hat * np.cos(heading)
        p[:, 1] += trans_hat * np.sin(heading)
        p[:, 2] = heading + rot2_hat
        return p
