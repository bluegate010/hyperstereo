"""Tunable parameters. Defaults are starting guesses meant to be refined in Phase 0."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ControllerConfig:
    # Formation
    cruise_alt_m: float = 100.0
    baseline_min_m: float = 50.0
    baseline_max_m: float = 200.0         # beyond ~200 m the stereo pipeline degrades; see docs/eval-baseline-500m.md
    view_pitch_deg: float = -3.0          # gimbal pitch while in stereo
    view_pitch_limits_deg: tuple[float, float] = (-10.0, 5.0)

    # Motion limits for the virtual vehicle / maneuvers
    transit_speed_mps: float = 8.0        # max drone speed along a rotation arc
    max_yaw_rate_dps: float = 30.0
    baseline_rate_mps: float = 6.0        # rate of change of B during transit
    live_baseline_rate_mps: float = 0.5   # small B tweaks allowed while in stereo
    live_baseline_max_delta_m: float = 5.0

    # Stereo fades
    fade_s: float = 0.6

    # Settle criteria (all must hold for settle_dwell_s)
    settle_pos_tol_m: float = 2.0
    settle_vel_tol_mps: float = 0.6
    settle_att_tol_deg: float = 1.0       # estimated camera attitude difference
    settle_dwell_s: float = 1.0

    # Assembly / recovery
    assembly_alt_split_m: float = 10.0    # altitude separation while drones cross paths
    waypoint_tol_m: float = 1.5
    recovery_final_alt_m: float = 5.0     # hand over to autopilot LAND below this

    # Safety
    min_separation_m: float = 6.0         # 3D; violation -> HOLD with vertical split
    hold_vertical_split_m: float = 8.0


@dataclass
class ComfortConfig:
    """Headset viewing model. Initial values; Phase 0 headset trials should tune these."""

    ipd_m: float = 0.063
    display_hfov_deg: float = 46.0        # horizontal FOV the image occupies in the headset
    focal_distance_m: float = 2.0         # virtual image distance of the headset optics
    comfort_diopters: float = 0.5         # allowed vergence-accommodation mismatch
    max_vertical_disparity_deg: float = 0.25

    @property
    def max_crossed_disparity_deg(self) -> float:
        """Largest crossed disparity whose vergence distance stays in the comfort zone."""
        import math
        near_vergence_m = 1.0 / (1.0 / self.focal_distance_m + self.comfort_diopters)
        return math.degrees(self.ipd_m / near_vergence_m)


@dataclass
class CameraConfig:
    width: int = 480
    height: int = 270
    hfov_deg: float = 60.0


@dataclass
class SimConfig:
    dt: float = 0.02
    seed: int = 1
    # Wind (NEU, m/s) and gusts (Ornstein-Uhlenbeck, independent per drone)
    wind_mean: tuple[float, float, float] = (0.0, 6.0, 0.0)  # blowing toward east
    gust_sigma_mps: float = 1.5
    gust_tau_s: float = 4.0
    # Navigation errors
    gps_sigma_m: float = 1.0              # ~1 m non-RTK; 0.02 for RTK
    gps_tau_s: float = 60.0
    compass_bias_sigma_deg: float = 1.5
    # Gimbal errors
    gimbal_bias_sigma_deg: float = 0.3    # per-axis constant offset, per unit
    gimbal_jitter_deg: float = 0.05
    gimbal_tau_s: float = 0.15
    gimbal_rate_coupling: float = 0.01    # residual deg per deg/s of vehicle body rate
    # Vehicle dynamics
    max_h_speed_mps: float = 12.0
    max_v_speed_mps: float = 3.0
    max_accel_mps2: float = 4.0
    drag_coeff: float = 0.25              # 1/s, linear airframe drag
    launch_positions: tuple[tuple[float, float, float], tuple[float, float, float]] = (
        (0.0, -5.0, 0.0), (0.0, 5.0, 0.0))


@dataclass
class Config:
    controller: ControllerConfig = field(default_factory=ControllerConfig)
    comfort: ComfortConfig = field(default_factory=ComfortConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    sim: SimConfig = field(default_factory=SimConfig)
