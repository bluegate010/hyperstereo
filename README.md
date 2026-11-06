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
| `hyperstereo/sitl.py` | Same interface, driving two ArduPilot SITL instances over MAVLink |
| `hyperstereo/terrain.py` | Procedural 16 km landscape and CPU raymarcher (images + true depth) |
| `hyperstereo/stereo.py` | Feature rectifier, DEM-based convergence, ground mask, SBS composition, ground-truth comfort metrics |
| `hyperstereo/runner.py` | Runs a scenario: flight, telemetry, optional SBS video, scoring |
| `hyperstereo/sweep.py` | Comfort sweep across baseline / lens / altitude: SBS stills, ladder video, metrics table |
| `hyperstereo/live.py` + `viewer/` | Real-time viewer: sim/SITL + controller at 50 Hz, streamed to a three.js page that renders both eyes on the GPU and reads the joystick |
| `scenarios/*.json` | Timed pilot commands and fault injection |

## Running

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install numpy scipy opencv-python-headless pymavlink aiohttp pytest
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
cd ~/ardupilot && pip install dronecan empy==3.3.4 future pexpect && ./waf configure --board sitl && ./waf copter
```

The optional argument is the SITL speedup; the runner paces itself to SITL's clock.

```bash
ARDUPILOT=~/ardupilot scripts/start_sitl.sh 3
```

```bash
python -m hyperstereo.runner scenarios/tour.json --sitl
```

Add `--render --fps 1` to render and score stereo from the SITL flight. SITL's
wind (`SIM_WIND_*`) and GPS noise (`SIM_GPS_NOISE`) are set from the config.
Gimbals are modelled by `GimbalModel`, driven by ArduCopter's reported body rates.

## Flying live (real-time viewer)

```bash
python -m hyperstereo.live
```

Open http://localhost:8765 in Chrome, Edge or Firefox, move the window onto the
headset display (for SBS glasses, switch them to their 3D/SBS mode), click
**Full screen**, and press a gamepad button so the browser picks up the controller.
Press `?` for the control map. Add `--sitl` to fly two ArduPilot SITL instances
(start them with `scripts/start_sitl.sh 1`); `--host 0.0.0.0` serves other devices.

| Gamepad (standard layout) | Keyboard | Action |
|---|---|---|
| Right stick | W S / A D | fly forward/back, left/right |
| Left stick Y | R / F | climb / descend |
| LT / RT | I / K | camera pitch down / up |
| LB / RB | Q / E | rotate −45° / +45° |
| D-pad ← / → | Z / C | rotate −90° / +90° |
| D-pad ↑ / ↓ | ↑ / ↓ | baseline +5 / −5 m (live) |
| Y / A | PgUp / PgDn | baseline +25 / −25 m (maneuver) |
| Start / Back | Enter / Backspace | launch & assemble / recover & land |
| X | H | resume after HOLD |
| B | T | toggle rectification (A/B the raw misaligned pair) |
| | M / V / U | ground mask / cycle view (SBS · anaglyph · 2D · cross-eyed) / HUD |
| | X | cross-eyed view on/off |
| | G / N | inject right-drone fault / new flight |

How it works: the browser renders each eye from the drone's *true* camera pose,
gimbal errors included. Rectification renders the right eye with the left
camera's orientation plus about 0.03° of residual error, which matches what the
feature + DEM pipeline achieved offline. That's mathematically the same as the
rotation-only homography. Snapshots are interpolated (with a slerp for orientation)
50 ms in the past, so motion stays smooth on 60–90 Hz displays. To test without a headset, use the anaglyph
view with red/cyan glasses, or the **cross-eyed** view (`X`, or `?view=cross`). The cross-eyed
view is side by side with the halves swapped, so you can free-view it by crossing your eyes
until the two images merge into a third in the middle. If they won't fuse, make the
window narrower or sit further back. Append `?lod=2` to the URL
to halve the terrain mesh resolution on weak GPUs.

Limitations: the browser path doesn't run the real feature rectifier, which needs
camera images the server never sees. Lock comes from the controller's settle
criteria alone, and the offline runner remains the way to validate lock.

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
6. **Gimbals must not start turning until the view is fully 2D.** The ArduPilot run
   caught the gimbals slewing to the new heading during the fade-out, which put a
   misaligned stereo pair in front of the pilot for about half a second. Fixed, with a
   regression test.
7. **Real ArduCopter is slower than the built-in sim.** Assembly takes 65–95 s instead of
   about 47 s (the climb is limited to about 2.5 m/s), and settling after a 180°
   rotation at B = 100 m takes longer: a 34 s gap instead of 24 s.
8. **Non-RTK GPS is fine for image quality** (baseline error about 1.5 m RMS, about 3%
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
- [x] Validate against two ArduPilot SITL instances (`--sitl`). Tour: 4 maneuvers,
      no holds, landed; stereo: 0 frames without lock, worst vertical error 0.10°, worst
      divergence 0.03°. Fault: HOLD, resume at the interrupted heading, stereo, landed.
- [x] Real-time interactive viewer (`python -m hyperstereo.live`): gamepad or keyboard,
      SBS / anaglyph / 2D output, rectification and mask toggles, sim or SITL backend.
      Tested in headless Chromium; not yet tested with a physical gamepad or headset.
