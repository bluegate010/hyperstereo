"""Frames and formation geometry.

Conventions used throughout the project:
  * World frame is local NEU in metres: x = north, y = east, z = up (altitude above launch).
  * Headings are compass bearings in degrees, clockwise from north.
  * The formation's "heading" is the direction both cameras look. The baseline is
    perpendicular to it, so the left drone sits to the left of the view direction.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np


def wrap_deg(a: float) -> float:
    """Wrap an angle to [-180, 180)."""
    return (a + 180.0) % 360.0 - 180.0


def forward_vec(heading_deg: float) -> np.ndarray:
    r = np.radians(heading_deg)
    return np.array([np.cos(r), np.sin(r), 0.0])


def right_vec(heading_deg: float) -> np.ndarray:
    r = np.radians(heading_deg)
    return np.array([-np.sin(r), np.cos(r), 0.0])


@dataclass(frozen=True)
class Formation:
    """The virtual vehicle the pilot flies, plus the stereo baseline."""

    center: np.ndarray  # NEU, metres
    heading_deg: float
    baseline_m: float

    def slots(self) -> tuple[np.ndarray, np.ndarray]:
        """(left, right) drone positions."""
        half = right_vec(self.heading_deg) * (self.baseline_m / 2.0)
        c = np.asarray(self.center, dtype=float)
        return c - half, c + half

    def moved(self, **kw) -> "Formation":
        return replace(self, **kw)


def step_toward(cur: float, target: float, max_step: float) -> float:
    d = target - cur
    if abs(d) <= max_step:
        return target
    return cur + np.sign(d) * max_step


def step_heading(cur: float, remaining: float, max_step: float) -> tuple[float, float]:
    """Advance a heading by up to max_step along a signed remaining rotation.

    Rotation is tracked as a signed remaining amount rather than a target heading so
    that an explicit "+180" turns clockwise and "-180" turns anticlockwise.
    Returns (new_heading, new_remaining).
    """
    step = float(np.clip(remaining, -max_step, max_step))
    return (cur + step) % 360.0, remaining - step
