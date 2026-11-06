"""ArduPilot SITL backend: drives two real ArduCopter SITL instances over MAVLink.

It has the same interface as sim.SimBackend, so `runner --sitl` swaps it in. Start the
two instances with scripts/start_sitl.sh first.

What SITL adds over the built-in sim: real ArduCopter position control, EKF, arming,
takeoff/land and failsafe behaviour, plus SITL's own wind/turbulence model. Gimbals are
not simulated by SITL here; the camera attitude comes from the same GimbalModel the
built-in sim uses, driven by the vehicle's reported body rates.

Tested against ArduCopter 4.5 SITL (two instances, --speedup 1 and 3): full tour and
fault-injection scenarios fly end to end, including takeoff, arc rotations, HOLD/resume
and LAND. SITL's GPS noise is white noise only (no slow drift), so baseline errors
come out smaller than the built-in sim's.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

from .config import SimConfig
from .controller import Setpoint, VehicleState
from .sim import CameraPose, GimbalModel

R_EARTH = 6378137.0
# Ignore velocity, acceleration and yaw-rate fields; use position + yaw.
TYPE_MASK_POS_YAW = 0b0000_1001_1111_1000


@dataclass
class _Veh:
    conn: object
    gimbal: GimbalModel
    launch: np.ndarray
    healthy: bool = True
    pos: np.ndarray | None = None
    vel: np.ndarray | None = None
    truth: np.ndarray | None = None
    att: np.ndarray | None = None          # roll, pitch, yaw (deg)
    rates: np.ndarray | None = None        # yaw, pitch, roll rates (deg/s)
    boot_ms: int = 0
    lat_ok: bool = False


class SitlBackend:
    def __init__(self, cfg: SimConfig, urls=("tcp:127.0.0.1:5760", "tcp:127.0.0.1:5770"),
                 origin=(-35.363261, 149.165230), timeout_s: float = 60.0):
        from pymavlink import mavutil
        self.mavutil = mavutil
        self.cfg = cfg
        self.origin = origin
        rng = np.random.default_rng(cfg.seed)
        self.vehicles: list[_Veh] = []
        for url, launch in zip(urls, cfg.launch_positions):
            c = mavutil.mavlink_connection(url, source_system=255)
            if c.wait_heartbeat(timeout=timeout_s) is None:
                raise RuntimeError(f"no heartbeat from {url}; is SITL running? (scripts/start_sitl.sh)")
            self.vehicles.append(_Veh(c, GimbalModel(cfg, rng, rng.normal(0, cfg.compass_bias_sigma_deg)),
                                      np.array(launch, float)))
        for v in self.vehicles:
            # GLOBAL_POSITION_INT, ATTITUDE, SIMSTATE at 50 Hz; ATTITUDE also paces the loop.
            for msg_id in (33, 30, 164):
                self._cmd(v, mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, msg_id, 20_000)
            wx, wy = cfg.wind_mean[0], cfg.wind_mean[1]
            spd = math.hypot(wx, wy)
            # SIM_WIND_DIR is the direction the wind blows *from*.
            self._param(v, "SIM_WIND_SPD", spd)
            self._param(v, "SIM_WIND_DIR", (math.degrees(math.atan2(-wy, -wx))) % 360.0)
            self._param(v, "SIM_WIND_TURB", cfg.gust_sigma_mps)
            # SITL's GPS is perfect by default. This adds white noise (the EKF filters it);
            # SITL has no slow-drift model like the built-in sim's.
            self._param(v, "SIM_GPS_NOISE", cfg.gps_sigma_m)
        self.t = 0.0
        self._clock_ms = None
        # Wait for GPS fix + EKF origin: GLOBAL_POSITION_INT reports lat=0 until then.
        deadline = time.time() + timeout_s * 2
        while any(v.pos is None or v.lat_ok is False for v in self.vehicles):
            if time.time() > deadline:
                raise RuntimeError("timed out waiting for SITL position fix")
            self._pump(0.2)

    # ---------------------------------------------------------------- MAVLink helpers
    def _cmd(self, v, cmd, *params):
        p = list(params) + [0] * (7 - len(params))
        v.conn.mav.command_long_send(v.conn.target_system, v.conn.target_component, cmd, 0, *p)

    def _param(self, v, name, value):
        v.conn.mav.param_set_send(v.conn.target_system, v.conn.target_component, name.encode(),
                                  float(value), self.mavutil.mavlink.MAV_PARAM_TYPE_REAL32)

    def _neu(self, lat_e7, lon_e7, rel_alt_mm, launch_alt=0.0):
        lat0, lon0 = self.origin
        n = math.radians(lat_e7 / 1e7 - lat0) * R_EARTH
        e = math.radians(lon_e7 / 1e7 - lon0) * R_EARTH * math.cos(math.radians(lat0))
        return np.array([n, e, rel_alt_mm / 1000.0 + launch_alt])

    def _latlon(self, p):
        lat0, lon0 = self.origin
        lat = lat0 + math.degrees(p[0] / R_EARTH)
        lon = lon0 + math.degrees(p[1] / (R_EARTH * math.cos(math.radians(lat0))))
        return int(lat * 1e7), int(lon * 1e7)

    def _pump(self, wait_s: float = 0.0):
        end = time.time() + wait_s
        while True:
            for v in self.vehicles:
                while (m := v.conn.recv_match(blocking=False)) is not None:
                    self._handle(v, m)
            if time.time() >= end:
                return
            time.sleep(0.002)

    def _handle(self, v, m):
        t = m.get_type()
        if t == "GLOBAL_POSITION_INT":
            v.lat_ok = m.lat != 0
            if not v.lat_ok:
                return
            v.pos = self._neu(m.lat, m.lon, m.relative_alt, v.launch[2])
            v.vel = np.array([m.vx, m.vy, -m.vz]) / 100.0
        elif t == "ATTITUDE":
            v.att = np.degrees([m.roll, m.pitch, m.yaw])
            v.rates = np.degrees([m.yawspeed, m.pitchspeed, m.rollspeed])
            v.boot_ms = max(v.boot_ms, m.time_boot_ms)
        elif t == "SIMSTATE":
            # SIMSTATE altitude is AMSL; keep the EKF's relative altitude for z.
            z = v.pos[2] if v.pos is not None else 0.0
            v.truth = self._neu(m.lat, m.lng, 0.0)
            v.truth[2] = z

    # ---------------------------------------------------------------- backend interface
    @property
    def launch_positions(self):
        return [v.launch.copy() for v in self.vehicles]

    def launch(self, alt: float = 5.0, timeout_s: float = 90.0):
        mav = self.mavutil.mavlink
        for v in self.vehicles:
            v.conn.set_mode("GUIDED")
        deadline = time.time() + timeout_s
        for v in self.vehicles:
            while not v.conn.motors_armed():
                if time.time() > deadline:
                    raise RuntimeError("arming timed out (EKF not ready?)")
                self._cmd(v, mav.MAV_CMD_COMPONENT_ARM_DISARM, 1)
                self._pump(1.0)
                v.conn.recv_match(type="HEARTBEAT", blocking=True, timeout=1)
            self._cmd(v, mav.MAV_CMD_NAV_TAKEOFF, 0, 0, 0, 0, 0, 0, alt)
        while any(v.pos is None or v.pos[2] < alt * 0.9 for v in self.vehicles):
            if time.time() > deadline:
                raise RuntimeError("takeoff timed out")
            self._pump(0.2)

    def states(self):
        out = []
        for v in self.vehicles:
            g = v.gimbal.est
            out.append(VehicleState(pos=v.pos.copy(), vel=v.vel.copy(), cam_yaw_deg=float(g[0]),
                                    cam_pitch_deg=float(g[1]), cam_roll_deg=float(g[2]), healthy=v.healthy))
        return tuple(out)

    def camera_poses(self):
        return tuple(CameraPose(self._truth(v), *v.gimbal.true_attitude()) for v in self.vehicles)

    def _truth(self, v):
        return (v.truth if v.truth is not None else v.pos).copy()

    def true_positions(self):
        return tuple(self._truth(v) for v in self.vehicles)

    def step(self, dt: float, sp_left: Setpoint, sp_right: Setpoint):
        mav = self.mavutil.mavlink
        for v, sp in zip(self.vehicles, (sp_left, sp_right)):
            lat, lon = self._latlon(sp.pos)
            v.conn.mav.set_position_target_global_int_send(
                0, v.conn.target_system, v.conn.target_component, mav.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
                TYPE_MASK_POS_YAW, lat, lon, float(sp.pos[2] - v.launch[2]), 0, 0, 0, 0, 0, 0,
                math.radians(sp.yaw_deg), 0)
            v.gimbal.step(dt, sp.gimbal_yaw_deg, sp.gimbal_pitch_deg,
                          v.rates if v.rates is not None else np.zeros(3))
        # Pace the controller to SITL's clock (works with --speedup). Track a target
        # time rather than "dt since the last message", so message jitter doesn't accumulate.
        v0 = self.vehicles[0]
        if self._clock_ms is None:
            self._clock_ms = v0.boot_ms
        self._clock_ms += dt * 1000.0
        deadline = time.time() + 2.0
        while v0.boot_ms < self._clock_ms - 10 and time.time() < deadline:
            self._pump(0.001)
        self.t += dt

    def land(self):
        for v in self.vehicles:
            v.conn.set_mode("LAND")

    def close(self):
        for v in self.vehicles:
            v.conn.close()
