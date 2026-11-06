import numpy as np
import pytest

from hyperstereo.config import Config
from hyperstereo.controller import FormationController, Mode, PilotInput, VehicleState
from hyperstereo.geometry import Formation, step_heading, wrap_deg
from hyperstereo.runner import Scenario, fly, summarize_flight


def perfect_states(out):
    """Vehicles that sit exactly on their setpoints with perfect gimbals."""
    return tuple(VehicleState(pos=sp.pos.copy(), vel=np.zeros(3), cam_yaw_deg=sp.gimbal_yaw_deg,
                              cam_pitch_deg=sp.gimbal_pitch_deg) for sp in (out.left, out.right))


def run_ideal(ctrl, pilot_fn, seconds, dt=0.02, t0=0.0):
    ll, lr = ctrl.launch
    states = (VehicleState(ll.copy(), np.zeros(3)), VehicleState(lr.copy(), np.zeros(3)))
    t = t0
    log = []
    while t < t0 + seconds:
        out = ctrl.update(t, dt, pilot_fn(t), states)
        states = perfect_states(out)
        log.append((t, out))
        t += dt
    return log


def test_slots_perpendicular_to_heading():
    f = Formation(np.array([0.0, 0.0, 100.0]), 90.0, 60.0)
    l, r = f.slots()
    assert np.allclose(l, [30.0, 0.0, 100.0])   # looking east, left is north
    assert np.allclose(r, [-30.0, 0.0, 100.0])
    assert np.linalg.norm(l - r) == pytest.approx(60.0)


def test_signed_rotation_direction():
    h, rem = 350.0, -180.0
    for _ in range(200):
        h, rem = step_heading(h, rem, 1.0)
    assert rem == 0.0 and h == pytest.approx(170.0)


def make_ctrl(**kw):
    cfg = Config().controller
    for k, v in kw.items():
        setattr(cfg, k, v)
    return FormationController(cfg, np.array([0.0, -5.0, 0.0]), np.array([0.0, 5.0, 0.0]),
                               heading_deg=0.0, baseline_m=50.0)


def test_assembly_reaches_stereo_and_rotation_returns_to_stereo():
    ctrl = make_ctrl()
    log = run_ideal(ctrl, lambda t: PilotInput(start=t < 0.05), 10.0)
    assert log[-1][1].mode == Mode.STEREO and log[-1][1].stereo_alpha == 1.0

    log = run_ideal(ctrl, lambda t: PilotInput(rotate_deg=180.0 if t < 10.01 else None), 40.0, t0=10.0)
    modes = [o.mode for _, o in log]
    assert Mode.TRANSIT in modes and modes[-1] == Mode.STEREO
    # Stereo is never shown during the maneuver
    assert all(o.stereo_alpha == 0.0 for _, o in log if o.mode == Mode.TRANSIT)
    # Arc keeps separation equal to the baseline throughout
    seps = [np.linalg.norm(o.left.pos - o.right.pos) for _, o in log]
    assert min(seps) == pytest.approx(50.0, abs=1e-6)
    assert wrap_deg(ctrl.formation.heading_deg - 180.0) == pytest.approx(0.0, abs=1e-6)


def test_small_baseline_change_is_live_large_one_is_a_maneuver():
    ctrl = make_ctrl()
    run_ideal(ctrl, lambda t: PilotInput(start=t < 0.05), 10.0)
    log = run_ideal(ctrl, lambda t: PilotInput(baseline_m=54.0 if t < 10.01 else None), 10.0, t0=10.0)
    assert all(o.mode == Mode.STEREO for _, o in log)
    assert ctrl.formation.baseline_m == pytest.approx(54.0)
    log = run_ideal(ctrl, lambda t: PilotInput(baseline_m=100.0 if t < 20.01 else None), 30.0, t0=20.0)
    assert Mode.TRANSIT in [o.mode for _, o in log]
    assert ctrl.formation.baseline_m == pytest.approx(100.0)
    assert log[-1][1].mode == Mode.STEREO


def test_baseline_clamped_to_configured_range():
    ctrl = make_ctrl()
    run_ideal(ctrl, lambda t: PilotInput(start=t < 0.05), 10.0)
    run_ideal(ctrl, lambda t: PilotInput(baseline_m=500.0 if t < 10.01 else None), 30.0, t0=10.0)
    assert ctrl.formation.baseline_m == pytest.approx(200.0)


def test_unhealthy_vehicle_triggers_hold_and_mono_from_healthy_drone():
    ctrl = make_ctrl()
    run_ideal(ctrl, lambda t: PilotInput(start=t < 0.05), 10.0)
    out = ctrl.update(10.0, 0.02, PilotInput(), (
        VehicleState(ctrl.formation.slots()[0], np.zeros(3), healthy=False),
        VehicleState(ctrl.formation.slots()[1], np.zeros(3))))
    assert out.mode == Mode.HOLD and out.stereo_alpha == 0.0 and out.mono_source == "R"


def test_separation_violation_holds_with_vertical_split():
    ctrl = make_ctrl()
    run_ideal(ctrl, lambda t: PilotInput(start=t < 0.05), 10.0)
    p = np.array([0.0, 0.0, 100.0])
    out = ctrl.update(10.0, 0.02, PilotInput(), (VehicleState(p, np.zeros(3)),
                                                 VehicleState(p + [0, 3, 0], np.zeros(3))))
    assert out.mode == Mode.HOLD
    assert abs(out.left.pos[2] - out.right.pos[2]) >= ctrl.cfg.hold_vertical_split_m


def test_assembly_paths_are_altitude_separated_when_crossing():
    """Heading 180 puts the left slot on the east, so the drones must cross over."""
    cfg = Config().controller
    ctrl = FormationController(cfg, np.array([0.0, -5.0, 0.0]), np.array([0.0, 5.0, 0.0]), heading_deg=180.0)
    ctrl.update(0.0, 0.02, PilotInput(start=True), (VehicleState(ctrl.launch[0], np.zeros(3)),
                                                     VehicleState(ctrl.launch[1], np.zeros(3))))
    (l1, r1), (l2, r2), _ = ctrl.waypoints
    assert l1[2] - r1[2] == pytest.approx(cfg.assembly_alt_split_m)
    assert l2[2] - r2[2] == pytest.approx(cfg.assembly_alt_split_m)


def test_resume_after_interrupted_rotation_uses_final_heading():
    ctrl = make_ctrl()
    run_ideal(ctrl, lambda t: PilotInput(start=t < 0.05), 10.0)
    run_ideal(ctrl, lambda t: PilotInput(rotate_deg=90.0 if t < 10.01 else None), 2.0, t0=10.0)
    assert ctrl.mode == Mode.TRANSIT and ctrl.rotate_remaining != 0
    sl, sr = ctrl.formation.slots()
    ctrl.update(12.0, 0.02, PilotInput(), (VehicleState(sl, np.zeros(3), healthy=False),
                                           VehicleState(sr, np.zeros(3))))
    out = ctrl.update(12.02, 0.02, PilotInput(resume=True), (VehicleState(sl, np.zeros(3)),
                                                             VehicleState(sr, np.zeros(3))))
    assert out.mode == Mode.ASSEMBLING
    assert ctrl.formation.heading_deg == pytest.approx(90.0) and ctrl.rotate_remaining == 0.0
    assert out.left.gimbal_yaw_deg == pytest.approx(90.0)


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_simulated_tour_with_wind_and_gps_noise(seed):
    cfg = Config()
    cfg.sim.seed = seed
    scn = Scenario.load("scenarios/tour.json")
    rows, _, events = fly(scn, cfg, 0.0)
    s = summarize_flight(rows, events, cfg)
    assert s["final_mode"] == "LANDED"
    assert s["holds"] == 0
    assert len(s["stereo_gaps_s"]) == 4
    assert s["min_separation_airborne_m"] > cfg.controller.min_separation_m
    assert s["stereo_baseline_err_max_m"] < 5.0


def test_gimbals_do_not_slew_until_fade_out_completes():
    ctrl = make_ctrl()
    run_ideal(ctrl, lambda t: PilotInput(start=t < 0.05), 10.0)
    log = run_ideal(ctrl, lambda t: PilotInput(rotate_deg=90.0 if t < 10.01 else None), 3.0, t0=10.0)
    fading = [o for _, o in log if o.mode == Mode.FADING_OUT]
    assert fading and all(o.left.gimbal_yaw_deg == pytest.approx(0.0) for o in fading)
    transit = [o for _, o in log if o.mode == Mode.TRANSIT]
    assert transit and transit[0].left.gimbal_yaw_deg == pytest.approx(90.0)
