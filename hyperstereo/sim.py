"""Lightweight multirotor + gimbal simulator.

It's no substitute for ArduPilot SITL, but it models the effects that matter for
stereo: wind and gusts that tilt each airframe differently, GPS drift, per-vehicle
compass bias, and gimbal bias, lag and jitter. Each vehicle exposes:
  * an *estimated* state (what the autopilot believes) for the controller, and
  * a *true* camera pose for rendering and stereo-error analysis.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import SimConfig
from .controller import Setpoint, VehicleState
from .geometry import forward_vec, right_vec, wrap_deg

G = 9.81


@dataclass
class CameraPose:
    pos: np.ndarray       # true NEU position
    yaw_deg: float        # true camera attitude in the world frame
    pitch_deg: float
    roll_deg: float


def _ou_step(x: np.ndarray, tau: float, sigma: float, dt: float, rng) -> np.ndarray:
    a = np.exp(-dt / tau)
    return x * a + rng.normal(0.0, sigma * np.sqrt(1 - a * a), size=x.shape)


class GimbalModel:
    """3-axis gimbal holding an absolute camera attitude, with lag, bias, jitter and
    residual coupling to airframe motion. Yaw is referenced to the vehicle compass."""

    def __init__(self, cfg: SimConfig, rng: np.random.Generator, compass_bias_deg: float):
        self.cfg, self.rng = cfg, rng
        self.compass_bias = compass_bias_deg
        self.bias = rng.normal(0.0, cfg.gimbal_bias_sigma_deg, 3)  # yaw, pitch, roll
        self.est = np.zeros(3)
        self.resid = np.zeros(3)

    def step(self, dt: float, yaw_cmd: float, pitch_cmd: float, body_rate: np.ndarray) -> None:
        a = 1.0 - np.exp(-dt / self.cfg.gimbal_tau_s)
        diff = np.array([yaw_cmd, pitch_cmd, 0.0]) - self.est
        diff[0] = wrap_deg(diff[0])
        self.est = self.est + a * diff
        self.est[0] %= 360.0
        self.resid = self.cfg.gimbal_rate_coupling * body_rate + self.rng.normal(0.0, self.cfg.gimbal_jitter_deg, 3)

    def true_attitude(self) -> tuple[float, float, float]:
        yaw, pitch, roll = self.est + self.bias + self.resid
        return float((yaw - self.compass_bias) % 360.0), float(pitch), float(roll)


class SimVehicle:
    KP_POS = 0.8
    KV = 2.0
    KI = 0.8
    YAW_RATE = 60.0

    def __init__(self, cfg: SimConfig, launch, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng
        self.p = np.array(launch, float)
        self.v = np.zeros(3)
        self.vi = np.zeros(2)
        self.gust = np.zeros(3)
        s = cfg.gps_sigma_m
        self.gps_err = rng.normal(0.0, 1.0, 3) * np.array([s, s, min(s, 0.3)])
        self.compass_bias = rng.normal(0.0, cfg.compass_bias_sigma_deg)
        self.yaw_est = 0.0
        self.tilt = np.zeros(2)  # body pitch, roll (deg)
        self.body_rate = np.zeros(3)  # yaw, pitch, roll rates (deg/s)
        self.gimbal = GimbalModel(cfg, rng, self.compass_bias)
        self.healthy = True

    # ---------------------------------------------------------------- dynamics
    def step(self, dt: float, sp: Setpoint, wind_mean: np.ndarray) -> None:
        cfg, rng = self.cfg, self.rng
        s = cfg.gps_sigma_m
        scale = np.array([s, s, min(s, 0.3)])  # baro keeps vertical error small
        self.gps_err = _ou_step(self.gps_err / scale, cfg.gps_tau_s, 1.0, dt, rng) * scale
        self.gust = _ou_step(self.gust, cfg.gust_tau_s, cfg.gust_sigma_mps, dt, rng)
        wind = wind_mean + self.gust * np.array([1.0, 1.0, 0.3])

        p_est = self.p + self.gps_err
        on_ground = self.p[2] <= 0.05 and sp.pos[2] <= 0.5
        if on_ground:
            self.v[:] = 0.0
            self.vi[:] = 0.0
            thrust_h = np.zeros(2)
        else:
            err = sp.pos - p_est
            v_des = self.KP_POS * err
            hs = np.linalg.norm(v_des[:2])
            if hs > cfg.max_h_speed_mps:
                v_des[:2] *= cfg.max_h_speed_mps / hs
            v_des[2] = np.clip(v_des[2], -cfg.max_v_speed_mps, cfg.max_v_speed_mps)
            ev = v_des - self.v
            self.vi = np.clip(self.vi + self.KI * ev[:2] * dt, -3.0, 3.0)
            thrust_h = self.KV * ev[:2] + self.vi
            n = np.linalg.norm(thrust_h)
            if n > cfg.max_accel_mps2:
                thrust_h *= cfg.max_accel_mps2 / n
            az = np.clip(self.KV * ev[2], -cfg.max_accel_mps2, cfg.max_accel_mps2)
            drag = -cfg.drag_coeff * (self.v - wind)
            acc = np.array([thrust_h[0], thrust_h[1], az]) + drag
            self.v = self.v + acc * dt
            self.p = self.p + self.v * dt
            if self.p[2] < 0.0:
                self.p[2] = 0.0
                self.v[2] = max(self.v[2], 0.0)

        # Yaw (autopilot tracks its *estimated* heading; truth differs by compass bias)
        prev_yaw = self.yaw_est
        dyaw = np.clip(wrap_deg(sp.yaw_deg - self.yaw_est), -self.YAW_RATE * dt, self.YAW_RATE * dt)
        self.yaw_est = (self.yaw_est + dyaw) % 360.0
        yaw_true = self.yaw_est - self.compass_bias

        # Body tilt from horizontal thrust
        th3 = np.array([thrust_h[0], thrust_h[1], 0.0])
        tilt = np.array([-np.degrees(np.arctan2(th3 @ forward_vec(yaw_true), G)),
                         np.degrees(np.arctan2(th3 @ right_vec(yaw_true), G))])
        rates = np.array([wrap_deg(self.yaw_est - prev_yaw), *(tilt - self.tilt)]) / dt
        self.body_rate = 0.8 * self.body_rate + 0.2 * rates
        self.tilt = tilt

        self.gimbal.step(dt, sp.gimbal_yaw_deg, sp.gimbal_pitch_deg, self.body_rate)

    # ---------------------------------------------------------------- outputs
    def estimated_state(self) -> VehicleState:
        g = self.gimbal.est
        return VehicleState(pos=self.p + self.gps_err, vel=self.v.copy(), cam_yaw_deg=float(g[0]),
                            cam_pitch_deg=float(g[1]), cam_roll_deg=float(g[2]), healthy=self.healthy)

    def camera_pose(self) -> CameraPose:
        return CameraPose(self.p.copy(), *self.gimbal.true_attitude())


class SimBackend:
    """Two simulated vehicles sharing a wind field."""

    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.vehicles = [SimVehicle(cfg, cfg.launch_positions[i], self.rng) for i in range(2)]
        self.t = 0.0

    @property
    def launch_positions(self):
        return [np.array(p, float) for p in self.cfg.launch_positions]

    def states(self) -> tuple[VehicleState, VehicleState]:
        return tuple(v.estimated_state() for v in self.vehicles)  # type: ignore[return-value]

    def camera_poses(self) -> tuple[CameraPose, CameraPose]:
        return tuple(v.camera_pose() for v in self.vehicles)  # type: ignore[return-value]

    def true_positions(self) -> tuple[np.ndarray, np.ndarray]:
        return self.vehicles[0].p.copy(), self.vehicles[1].p.copy()

    def step(self, dt: float, sp_left: Setpoint, sp_right: Setpoint) -> None:
        wind = np.array(self.cfg.wind_mean, float)
        self.vehicles[0].step(dt, sp_left, wind)
        self.vehicles[1].step(dt, sp_right, wind)
        self.t += dt

    def launch(self) -> None:
        pass  # simulated vehicles take off by simply following the first assembly waypoint

    def land(self) -> None:
        pass  # the recovery sequence already brings the vehicles down to a few metres

    def close(self) -> None:
        pass
