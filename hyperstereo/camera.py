"""Pinhole camera model. Camera axes: x right, y down, z forward."""
from __future__ import annotations

import numpy as np

from .geometry import right_vec


def cam_rotation(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    """Rotation whose columns are the camera x/y/z axes expressed in world NEU."""
    y, p, r = np.radians([yaw_deg, pitch_deg, roll_deg])
    f = np.array([np.cos(p) * np.cos(y), np.cos(p) * np.sin(y), np.sin(p)])
    rt = right_vec(yaw_deg)
    dn = np.cross(rt, f)
    rt2 = np.cos(r) * rt + np.sin(r) * dn     # positive roll = right side down
    dn2 = np.cos(r) * dn - np.sin(r) * rt
    return np.column_stack([rt2, dn2, f])


def intrinsics(width: int, height: int, hfov_deg: float) -> np.ndarray:
    fx = (width / 2.0) / np.tan(np.radians(hfov_deg) / 2.0)
    return np.array([[fx, 0.0, (width - 1) / 2.0],
                     [0.0, fx, (height - 1) / 2.0],
                     [0.0, 0.0, 1.0]])


def pixel_grid(width: int, height: int, step: int = 1) -> tuple[np.ndarray, np.ndarray]:
    u, v = np.meshgrid(np.arange(0, width, step, dtype=float), np.arange(0, height, step, dtype=float))
    return u, v


def pixel_rays(K: np.ndarray, R: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Unit world-frame ray directions for pixel coordinates (any shape) -> (..., 3)."""
    x = (u - K[0, 2]) / K[0, 0]
    y = (v - K[1, 2]) / K[1, 1]
    d = np.stack([x, y, np.ones_like(x)], axis=-1) @ R.T
    return d / np.linalg.norm(d, axis=-1, keepdims=True)


def project(points: np.ndarray, cam_pos: np.ndarray, R: np.ndarray, K: np.ndarray,
            at_infinity: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project world points (..., 3) -> (u, v, z_cam). With at_infinity, points are directions."""
    pc = (points if at_infinity else points - cam_pos) @ R
    z = pc[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = K[0, 0] * pc[..., 0] / z + K[0, 2]
        v = K[1, 1] * pc[..., 1] / z + K[1, 2]
    return u, v, z
