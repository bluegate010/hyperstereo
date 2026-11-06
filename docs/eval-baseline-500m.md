# Evaluation: raising the maximum baseline to 500 m

**Verdict:** the formation controller handles 500 m fine. The **stereo pipeline doesn't**:
past about 150 m, the current pipeline produces eye divergence and unmasked uncomfortable
parallax. 500 m can work as a gated "landscape mode", but only after three pipeline
changes, and even then only about a quarter of the frame shows comfortable stereo.

## Method

- Comfort sweeps with `python -m hyperstereo.sweep --cell 32`: a 32 km map, so terrain
  exists beyond the roughly 6 km that 500 m needs. Two view headings, B = 50–500 m, AGL 100 / 120 m,
  60° lens, scored against ground truth.
- Diagnostic runs at 120 m AGL comparing the flat-ground mask with a DEM-based mask, and the
  feature rectifier with perfect (oracle) rectification.
- Flight: `scenarios/b500.json` with `--baseline-max 500`, three seeds (wind 6 m/s +
  gusts, 1 m GPS).

## Geometry

With the default headset comfort model (3.6° crossed-disparity limit, 46° display, 60° lens),
the nearest comfortable distance is about **12 × B**:

| B | Nearest comfortable distance | Apparent scale |
|---|---|---|
| 100 m | 1.2 km | ×1,600 |
| 200 m | 2.4 km | ×3,200 |
| 500 m | **6.1 km** | **×7,900** |

At 500 m almost all terrain nearer than 6 km has to be masked or it's uncomfortable. The
usable stereo is the distant ranges and the sky. Atmospheric haze lowers the contrast of
exactly those distant features.

## Stereo results (ground truth)

Worst case over both headings, current pipeline (feature rectifier, DEM convergence,
flat-ground mask):

| B | Unmasked pixels over the limit | Max divergence | Vertical p95 | Masked |
|---|---|---|---|---|
| 50 m | 0.7% | 0.01° | 0.04° | 26–31% |
| 100 m | 0.7% | 0.04° | 0.03° | 42–44% |
| 200 m | 8.5% | 0.28° | 0.15° | 50–51% |
| 300 m | 32% | 0.41° | 0.24° | 52–53% |
| 500 m | **57%** | **0.38°** (1.3° in the diagnostic run) | **0.30°** | 54–55% |

Comfort targets: 0 divergence (anything > 0.1° flagged), vertical < 0.25°.

What fails, and why (diagnostic runs):

1. **Uncomfortable parallax: the flat-ground mask.** Mountain slopes 3–6 km away sit
   *above* the horizon, where the flat-ground mask never reaches. A **DEM-based mask**
   (mask wherever terrain-model disparity exceeds the limit) cuts violations at 500 m from
   12–54% to 0–4%. Masked area then rises to **65–79%**.
2. **Divergence: the yaw calibration.** It's 0° with oracle rectification but up to 1.3° with
   the real pipeline at 500 m. The DEM-based yaw estimate's error grows in proportion to B
   (error ≈ f·B·ΔZ/Z²), and there are fewer feature matches (about 490 at 100 m versus 160–190 at
   500 m) because the two viewpoints differ more.
3. **Vertical misalignment: the rectifier's estimate.** It reaches 0.17–0.27° at 300 m and above,
   against 0.02° for oracle rectification. Wider parallax and fewer matches weaken the fit.
   This is right at the comfort limit.

## Flight results (3 seeds, all landed, 0 holds)

| Maneuver | Gap with no stereo |
|---|---|
| Spread 50 → 500 m | 79 s |
| Rotate 90° at B = 500 m | 54–56 s |
| Rotate 180° at B = 500 m | 103–104 s |
| Shrink 500 → 100 m | 71 s |

For comparison, at 50–100 m the gaps are 10–34 s. Baseline error was 1.0–1.3 m RMS,
0.25% of B, which is negligible. Each drone flew about 1.7 km in the 9.5-minute scenario, and
each 180° rotation at 500 m is a 785 m arc. That's fine for one battery, but repositioning
takes a large share of flight time.

## Operational impact (not simulated)

- **Line of sight:** at 500 m the drones are 250 m on *opposite sides* of the pilot. One
  person can't watch both, so plan on two visual observers (Part 107.31), on top of the
  multi-aircraft waiver (107.35).
- **Terrain clearance:** the drones fly at the same altitude, possibly 500 m apart over
  hilly ground. Check clearance per drone against the DEM, not at the formation center.
- **Links:** 250 m from the ground station is well inside typical video/telemetry range.
- **Collision risk** at cruise is negligible; the takeoff/landing phases are unchanged.

## Recommendation

Keep **100 m** as the default maximum (the pipeline is validated there), and treat
**150 m** as the current pipeline's safe limit. Allow up to 500 m as an opt-in landscape
mode once these exist:

1. **DEM-based ground/terrain mask** (replace or add to the flat-ground mask). It's
   required beyond about 150 m and cheap, because the DEM is already loaded for convergence.
2. **A better yaw reference:** dual-antenna GNSS heading on each drone (about 0.1–0.3°) instead
   of the compass, or calibrate yaw at a short baseline and carry the correction while
   spreading out. This removes the divergence failure.
3. **A scene-adaptive baseline cap:** limit B to (median DEM distance of terrain in view) / 12,
   so the pilot can't select 500 m while looking at a ridge 2 km away.
4. **Rectifier improvements for wide parallax:** weight distant features and reject matches
   whose disparity is implausible for their DEM depth.

Then re-run this evaluation (`sweep --cell 32`, `runner --baseline-max 500`).

Caveats: synthetic terrain and haze, 640×360 images, and comfort limits that are still
model guesses pending headset trials.
