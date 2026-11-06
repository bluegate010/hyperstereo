"""Formation controller and stereo-availability state machine.

The pilot flies a virtual vehicle (the formation center). Each tick the controller
turns pilot input plus both drones' estimated states into position/yaw/gimbal
setpoints and decides whether the headset should show stereo or a 2D feed.

Mode flow:

    IDLE --start--> ASSEMBLING --> SETTLING --> FADING_IN --> STEREO
    STEREO --rotate / large baseline change--> FADING_OUT --> TRANSIT --> SETTLING ...
    STEREO --recover--> FADING_OUT --> RECOVERING --> LANDED
    any airborne mode --safety--> HOLD --resume--> ASSEMBLING (from current positions)

Rotation is flown as an arc around the formation center so the two drones stay on
opposite sides of it and their separation never drops below the baseline.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from .config import ControllerConfig
from .geometry import Formation, forward_vec, right_vec, step_heading, step_toward, wrap_deg


class Mode(str, Enum):
    IDLE = "IDLE"
    ASSEMBLING = "ASSEMBLING"
    SETTLING = "SETTLING"
    FADING_IN = "FADING_IN"
    STEREO = "STEREO"
    FADING_OUT = "FADING_OUT"
    TRANSIT = "TRANSIT"
    RECOVERING = "RECOVERING"
    LANDED = "LANDED"
    HOLD = "HOLD"


FORMATION_MODES = {Mode.SETTLING, Mode.FADING_IN, Mode.STEREO, Mode.FADING_OUT, Mode.TRANSIT}
AIRBORNE_MODES = FORMATION_MODES | {Mode.ASSEMBLING, Mode.RECOVERING, Mode.HOLD}


@dataclass
class VehicleState:
    pos: np.ndarray                 # estimated position, NEU
    vel: np.ndarray                 # estimated velocity, NEU
    cam_yaw_deg: float = 0.0        # estimated camera attitude in the world frame
    cam_pitch_deg: float = 0.0
    cam_roll_deg: float = 0.0
    healthy: bool = True            # link, nav fix, battery all OK


@dataclass
class PilotInput:
    v_forward: float = 0.0          # m/s, in the formation's view direction
    v_right: float = 0.0
    v_up: float = 0.0
    rotate_deg: float | None = None     # one-shot: signed rotation, + = clockwise
    baseline_m: float | None = None     # one-shot: new target baseline
    view_pitch_deg: float | None = None
    start: bool = False
    recover: bool = False
    resume: bool = False


@dataclass
class Setpoint:
    pos: np.ndarray
    yaw_deg: float
    gimbal_yaw_deg: float           # absolute camera heading
    gimbal_pitch_deg: float


@dataclass
class Output:
    left: Setpoint
    right: Setpoint
    mode: Mode
    stereo_alpha: float             # 0 = 2D (mono_source to both eyes), 1 = full stereo
    mono_source: str                # "L" or "R"
    formation: Formation | None
    land_now: bool = False
    events: list[str] = field(default_factory=list)


class FormationController:
    MAX_CENTER_SPEED = 6.0
    ALT_LIMITS = (30.0, 120.0)

    def __init__(self, cfg: ControllerConfig, launch_left: np.ndarray, launch_right: np.ndarray,
                 heading_deg: float = 0.0, baseline_m: float | None = None):
        self.cfg = cfg
        self.launch = (np.asarray(launch_left, float), np.asarray(launch_right, float))
        mid = (self.launch[0] + self.launch[1]) / 2.0
        b = baseline_m if baseline_m is not None else cfg.baseline_min_m
        self.formation = Formation(np.array([mid[0], mid[1], cfg.cruise_alt_m]), heading_deg % 360.0,
                                   float(np.clip(b, cfg.baseline_min_m, cfg.baseline_max_m)))
        self.mode = Mode.IDLE
        self.alpha = 0.0
        self.view_pitch = cfg.view_pitch_deg
        self.center_vel = np.zeros(3)
        self.rotate_remaining = 0.0
        self.baseline_target = self.formation.baseline_m
        self.after_fade_out: Mode = Mode.TRANSIT
        self.waypoints: list[tuple[np.ndarray, np.ndarray]] = []
        self.hold_pos: tuple[np.ndarray, np.ndarray] | None = None
        self.mono_source = "L"
        self.settle_timer = 0.0
        self.t = 0.0
        self._events: list[str] = []

    # ------------------------------------------------------------------ helpers
    def _set_mode(self, m: Mode, why: str = "") -> None:
        if m != self.mode:
            self._events.append(f"{self.t:7.2f}s {self.mode.value} -> {m.value}" + (f" ({why})" if why else ""))
            self.mode = m
            self.settle_timer = 0.0

    def _plan_spread(self, from_l: np.ndarray, from_r: np.ndarray) -> None:
        """Waypoints that move both drones into formation slots without crossing at equal altitude."""
        sl, sr = self.formation.slots()
        a, s = self.formation.center[2], self.cfg.assembly_alt_split_m / 2.0
        hi, lo = a + s, a - s
        self.waypoints = [
            (np.array([from_l[0], from_l[1], hi]), np.array([from_r[0], from_r[1], lo])),
            (np.array([sl[0], sl[1], hi]), np.array([sr[0], sr[1], lo])),
            (sl.copy(), sr.copy()),
        ]

    def _plan_recovery(self) -> None:
        sl, sr = self.formation.slots()
        a, s = self.formation.center[2], self.cfg.assembly_alt_split_m / 2.0
        ll, lr = self.launch
        f = self.cfg.recovery_final_alt_m
        self.waypoints = [
            (np.array([sl[0], sl[1], a + s]), np.array([sr[0], sr[1], a - s])),
            (np.array([ll[0], ll[1], a + s]), np.array([lr[0], lr[1], a - s])),
            (np.array([ll[0], ll[1], f]), np.array([lr[0], lr[1], f])),
        ]

    def _request_maneuver(self, rotate: float | None, baseline: float | None) -> bool:
        """Queue a rotation / baseline change. Returns True if it needs a fade-out."""
        cfg = self.cfg
        needs_transit = False
        if rotate:
            self.rotate_remaining += rotate
            needs_transit = True
        if baseline is not None:
            b = float(np.clip(baseline, cfg.baseline_min_m, cfg.baseline_max_m))
            self.baseline_target = b
            if abs(b - self.formation.baseline_m) > cfg.live_baseline_max_delta_m:
                needs_transit = True
        return needs_transit

    def _settled(self, states: tuple[VehicleState, VehicleState], stereo_ok: bool) -> bool:
        cfg = self.cfg
        for st, slot in zip(states, self.formation.slots()):
            if np.linalg.norm(st.pos - slot) > cfg.settle_pos_tol_m:
                return False
            if np.linalg.norm(st.vel - self.center_vel) > cfg.settle_vel_tol_mps:
                return False
        a, b = states
        # Cameras must look perpendicular to the baseline, or the epipolar lines tilt and
        # rotation-only rectification can't remove the vertical disparity.
        for st in states:
            if abs(wrap_deg(st.cam_yaw_deg - self.formation.heading_deg)) > cfg.settle_att_tol_deg:
                return False
        att = max(abs(wrap_deg(a.cam_yaw_deg - b.cam_yaw_deg)),
                  abs(a.cam_pitch_deg - b.cam_pitch_deg),
                  abs(a.cam_roll_deg - b.cam_roll_deg))
        return att <= cfg.settle_att_tol_deg and stereo_ok

    # ------------------------------------------------------------------ main step
    def update(self, t: float, dt: float, pilot: PilotInput,
               states: tuple[VehicleState, VehicleState], stereo_ok: bool = True) -> Output:
        self.t = t
        cfg = self.cfg
        f = self.formation
        land_now = False

        if pilot.view_pitch_deg is not None:
            self.view_pitch = float(np.clip(pilot.view_pitch_deg, *cfg.view_pitch_limits_deg))

        # ---- safety checks (any airborne mode except HOLD itself)
        if self.mode in AIRBORNE_MODES and self.mode != Mode.HOLD:
            sep = float(np.linalg.norm(states[0].pos - states[1].pos))
            unhealthy = [i for i, s in enumerate(states) if not s.healthy]
            if unhealthy or sep < cfg.min_separation_m:
                why = f"vehicle {'LR'[unhealthy[0]]} unhealthy" if unhealthy else f"separation {sep:.1f} m"
                hl, hr = states[0].pos.copy(), states[1].pos.copy()
                if sep < cfg.min_separation_m:
                    if hl[2] >= hr[2]:
                        hl[2] += cfg.hold_vertical_split_m
                    else:
                        hr[2] += cfg.hold_vertical_split_m
                self.hold_pos = (hl, hr)
                self.mono_source = "R" if (unhealthy and unhealthy[0] == 0) else "L"
                self.alpha = 0.0
                self.waypoints = []
                self._set_mode(Mode.HOLD, why)

        # ---- pilot commands that change mode
        if self.mode == Mode.IDLE and pilot.start:
            self._plan_spread(states[0].pos, states[1].pos)
            self._set_mode(Mode.ASSEMBLING, "start")
        elif self.mode == Mode.HOLD and pilot.resume and all(s.healthy for s in states):
            # Re-form around the current midpoint at the heading of any interrupted rotation,
            # so the slots and the gimbal heading agree.
            self.formation = f = f.moved(
                center=np.array([*((states[0].pos + states[1].pos) / 2)[:2], f.center[2]]),
                heading_deg=(f.heading_deg + self.rotate_remaining) % 360.0,
                baseline_m=self.baseline_target)
            self.rotate_remaining = 0.0
            self._plan_spread(states[0].pos, states[1].pos)
            self.mono_source = "L"
            self._set_mode(Mode.ASSEMBLING, "resume")
        elif self.mode in FORMATION_MODES and pilot.recover:
            self.rotate_remaining = 0.0
            self.after_fade_out = Mode.RECOVERING
            self._set_mode(Mode.FADING_OUT, "recover")
        elif self.mode in FORMATION_MODES and (pilot.rotate_deg or pilot.baseline_m is not None):
            if self._request_maneuver(pilot.rotate_deg, pilot.baseline_m):
                if self.mode in (Mode.STEREO, Mode.FADING_IN):
                    self.after_fade_out = Mode.TRANSIT
                    self._set_mode(Mode.FADING_OUT, "maneuver requested")
                elif self.mode == Mode.SETTLING:
                    self._set_mode(Mode.TRANSIT, "maneuver requested")

        # ---- virtual-vehicle translation
        if self.mode in FORMATION_MODES:
            v = (forward_vec(f.heading_deg) * pilot.v_forward + right_vec(f.heading_deg) * pilot.v_right)
            hs = np.linalg.norm(v[:2])
            if hs > self.MAX_CENTER_SPEED:
                v *= self.MAX_CENTER_SPEED / hs
            v[2] = float(np.clip(pilot.v_up, -2.0, 2.0))
            c = f.center + v * dt
            c[2] = float(np.clip(c[2], *self.ALT_LIMITS))
            self.center_vel = v
            f = f.moved(center=c)
        else:
            self.center_vel = np.zeros(3)

        # ---- mode-specific behaviour
        if self.mode == Mode.STEREO:
            # Small baseline tweaks are flown live.
            f = f.moved(baseline_m=step_toward(f.baseline_m, self.baseline_target,
                                               cfg.live_baseline_rate_mps * dt))
        elif self.mode == Mode.FADING_OUT:
            self.alpha = max(0.0, self.alpha - dt / cfg.fade_s)
            if self.alpha == 0.0:
                if self.after_fade_out == Mode.RECOVERING:
                    self._plan_recovery()
                self._set_mode(self.after_fade_out)
        elif self.mode == Mode.TRANSIT:
            half = max(f.baseline_m / 2.0, 1.0)
            omega = min(np.degrees(cfg.transit_speed_mps / half), cfg.max_yaw_rate_dps)
            h, self.rotate_remaining = step_heading(f.heading_deg, self.rotate_remaining, omega * dt)
            b = step_toward(f.baseline_m, self.baseline_target, cfg.baseline_rate_mps * dt)
            f = f.moved(heading_deg=h, baseline_m=b)
            if self.rotate_remaining == 0.0 and b == self.baseline_target:
                self._set_mode(Mode.SETTLING, "maneuver complete")
        elif self.mode == Mode.SETTLING:
            if self._settled(states, stereo_ok):
                self.settle_timer += dt
                if self.settle_timer >= cfg.settle_dwell_s:
                    self._set_mode(Mode.FADING_IN, "settled")
            else:
                self.settle_timer = 0.0
        elif self.mode == Mode.FADING_IN:
            if not stereo_ok:
                self._set_mode(Mode.SETTLING, "stereo lock lost")
            else:
                self.alpha = min(1.0, self.alpha + dt / cfg.fade_s)
                if self.alpha == 1.0:
                    self._set_mode(Mode.STEREO)
        if self.mode == Mode.STEREO and not stereo_ok:
            self.after_fade_out = Mode.TRANSIT  # zero-length transit -> straight back to settling
            self._set_mode(Mode.FADING_OUT, "stereo lock lost")

        self.formation = f

        # ---- waypoint sequences
        if self.mode in (Mode.ASSEMBLING, Mode.RECOVERING) and self.waypoints:
            wl, wr = self.waypoints[0]
            if (np.linalg.norm(states[0].pos - wl) < cfg.waypoint_tol_m
                    and np.linalg.norm(states[1].pos - wr) < cfg.waypoint_tol_m):
                self.waypoints.pop(0)
                if not self.waypoints:
                    if self.mode == Mode.ASSEMBLING:
                        self._set_mode(Mode.SETTLING, "assembled")
                    else:
                        land_now = True
                        self._set_mode(Mode.LANDED, "over launch point, handing to LAND")

        return Output(*self._setpoints(states), mode=self.mode, stereo_alpha=self.alpha,
                      mono_source=self.mono_source, formation=self.formation, land_now=land_now,
                      events=self._drain_events())

    def _setpoints(self, states) -> tuple[Setpoint, Setpoint]:
        f = self.formation
        # Gimbals lead to the final heading only once the view is fully 2D. Slewing them during
        # FADING_OUT would show the pilot a partly faded, misaligned stereo pair.
        lead = self.mode not in (Mode.FADING_OUT, Mode.FADING_IN, Mode.STEREO)
        final_heading = (f.heading_deg + (self.rotate_remaining if lead else 0.0)) % 360.0
        pitch = self.view_pitch
        if self.mode == Mode.HOLD and self.hold_pos is not None:
            pl, pr = self.hold_pos
        elif self.mode in (Mode.ASSEMBLING, Mode.RECOVERING) and self.waypoints:
            pl, pr = self.waypoints[0]
        elif self.mode in (Mode.IDLE, Mode.LANDED):
            pl, pr = states[0].pos, states[1].pos
        else:
            pl, pr = f.slots()
        # Vehicles yaw with the formation; gimbals lead to the final heading during transit.
        return (Setpoint(np.asarray(pl, float), f.heading_deg, final_heading, pitch),
                Setpoint(np.asarray(pr, float), f.heading_deg, final_heading, pitch))

    def _drain_events(self) -> list[str]:
        e, self._events = self._events, []
        return e
