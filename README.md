# Hyperstereo FPV: Phase 0 (simulation)

Two drones fly in tandem 50–100 m apart, and each one feeds one eye of an FPV headset.
The result is extreme parallax: the world looks 800–1600× smaller, like a model. This
repository is the Phase 0 simulation. It covers the formation controller and stereo
state machine, a flight and gimbal simulator, a synthetic landscape renderer, the
stereo pipeline, and comfort metrics scored against ground truth.

## Layout

| Module | What it does |
|---|---|
| `hyperstereo/controller.py` | Virtual-vehicle formation controller and the STEREO / FADING_OUT / TRANSIT / SETTLING / FADING_IN state machine, plus assembly, recovery and HOLD failsafe |
| `hyperstereo/geometry.py` | Frames (NEU, compass headings) and formation slot math |
| `hyperstereo/sim.py` | Multirotor + gimbal sim: wind and gusts, GPS drift, compass bias, gimbal bias, lag and jitter |
| `hyperstereo/sitl.py` | Same interface, driving two ArduPilot SITL instances over MAVLink (**untested**, see below) |
| `hyperstereo/terrain.py` | Procedural 16 km landscape and CPU raymarcher (images + true depth) |
| `hyperstereo/stereo.py` | Feature rectifier, DEM-based convergence, ground mask, SBS composition, ground-truth comfort metrics |
| `hyperstereo/runner.py` | Runs a scenario: flight, telemetry, optional SBS video, scoring |
| `hyperstereo/sweep.py` | Comfort sweep across baseline / lens / altitude: SBS stills, ladder video, metrics table |
| `scenarios/*.json` | Timed pilot commands and fault injection |

## Running

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install numpy scipy opencv-python-headless pymavlink pytest
```

```bash
python -m pytest -q
```

Flight only (a few seconds):

```bash
python -m hyperstereo.runner scenarios/tour.json
```

Flight plus rendered SBS video and stereo scoring (CPU-heavy):

```bash
python -m hyperstereo.runner scenarios/tour.json --render --fps 2
```

Comfort sweep material for the headset:

```bash
python -m hyperstereo.sweep --baselines 25,50,75,100,150 --hfovs 45,60
```

Options: `--rtk` (2 cm GPS), `--wind 10`, `--seed N`, `--width/--height/--hfov`.

Outputs go to `out/<scenario>/`: `telemetry.csv` (50 Hz), `events.log` (mode changes),
`frames.csv` (per-frame stereo metrics), `summary.json` and `sbs.mp4`.

### Against ArduPilot SITL

```bash
cd ~/ardupilot && ./waf configure --board sitl && ./waf copter
```

```bash
ARDUPILOT=~/ardupilot scripts/start_sitl.sh
```

```bash
python -m hyperstereo.runner scenarios/tour.json --sitl
```

`sitl.py` hasn't been run against a live SITL yet, so expect to debug the first run.

## Scenario format

```json
{"name": "tour", "heading_deg": 0, "baseline_m": 50, "duration_s": 400,
 "events": [
   {"t": 1,   "cmd": {"start": true}},
   {"t": 70,  "cmd": {"v_forward": 4.0}},
   {"t": 95,  "cmd": {"rotate_deg": 90}},
   {"t": 150, "cmd": {"baseline_m": 100}},
   {"t": 86,  "fault": {"vehicle": "R", "healthy": false}},
   {"t": 285, "cmd": {"recover": true}}]}
```

Velocities (`v_forward`, `v_right`, `v_up`) stay in effect until changed. `rotate_deg`
(signed, + = clockwise), `baseline_m`, `view_pitch_deg`, `start`, `recover` and `resume`
each fire once. A baseline change of up to 5 m is flown live in stereo; anything
larger becomes a fade-out / transit / settle maneuver.

## Phase 0 findings so far

1. **Rotation as an arc around the center works.** Separation stays equal to B the
   whole time, and stereo is fully off during transit. The gap from request to stereo
   is about 10 s for 90° at B = 50 m and about 24 s for 180° at B = 100 m. The arc speed
   is 8 m/s; settling takes 2–5 s.
2. **Yaw mismatch is the dangerous error, not vertical misalignment.** Compass and
   gimbal yaw bias (~1–2°) add a constant horizontal disparity, which can push the sky
   *beyond infinity* (the eyes diverge). Image features alone can't separate it from
   parallax, because the far mountains aren't at infinity. **Fix:** predict feature
   distances from a public terrain elevation model (DEM) and the GPS pose, estimate the
   yaw error from the difference, and apply it as a rotation. In random trials divergence
   dropped from as much as 0.6° to at most 0.03°.
3. **Vertical alignment is easy.** Feature-based rotation rectification takes raw
   vertical disparity from 0.4–2° down to 0.02–0.06° (95th percentile).
4. **The ground is the real limit on the usable frame.** At 100 m altitude with the
   camera pitched 3° down, everything nearer than about 16·B (for the default headset
   comfort model) has to be masked. That's **30–50% of the frame at B = 50–100 m**.
   Possible responses: fly as high as allowed (120 m), pitch the camera level or slightly
   up, or accept a letterboxed stereo window. Headset trials should decide.
5. **The cameras must look perpendicular to the baseline before stereo is allowed.**
   The sim caught a bug where resuming after a fault mid-rotation left the gimbals at
   the new heading and the drones on the old baseline, giving 1.9° of raw vertical
   disparity. That's now a settle criterion and a regression test.
6. **Non-RTK GPS is fine for image quality** (baseline error about 1.5 m RMS, about 3%
   at B = 50 m), but launch and recovery with the drones 10 m apart left only 8 m of
   separation. RTK still matters for the close-quarters phases.

## Phase 0 exit checklist

- [x] Formation controller with arc rotation, live and maneuvered baseline changes
- [x] Rotate / settle / lock state machine with fades, HOLD failsafe and resume
- [x] Assembly and recovery with altitude-separated crossing
- [x] Stereo pipeline: rectification, divergence-safe convergence, ground mask
- [x] Ground-truth comfort scoring, SBS video output, comfort sweep
- [ ] **Headset trials**: watch `out/sweep/ladder.mp4` and the scenario videos in an
      SBS-capable headset and fill in `comfort_rating` in `out/sweep/results.csv`, then
      tune `ComfortConfig` to match
- [ ] Validate the controller against two ArduPilot SITL instances (`--sitl`)
- [ ] Real-time interactive viewer (fly with a joystick in the headset). The CPU
      raymarcher is offline-only; a WebGL or game-engine front end would reuse
      `controller.py` unchanged.
