"""Scenario runner.

Pass 1 flies the scenario: simulator + formation controller at 50 Hz, logging telemetry
and the true camera poses at each video frame time.
Pass 2 (optional, parallel) renders both eyes for each frame, runs the stereo
pipeline (feature rectification, DEM convergence, ground mask), composes SBS video and
scores every frame against ground truth.

Pass 2 runs offline, so the controller's `stereo_ok` input in pass 1 comes from the
settle criteria alone. Pass 2 then checks that the rectifier actually had lock on
every frame the pilot saw in stereo, and reports any frames where it didn't.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np

from .config import Config
from .controller import FORMATION_MODES, FormationController, Mode, PilotInput

ONE_SHOT = {"rotate_deg", "baseline_m", "view_pitch_deg", "start", "recover", "resume"}


# ---------------------------------------------------------------------- scenario
@dataclass
class Scenario:
    name: str
    heading_deg: float
    baseline_m: float
    duration_s: float
    events: list[dict]

    @classmethod
    def load(cls, path: str | Path) -> "Scenario":
        d = json.loads(Path(path).read_text())
        return cls(d.get("name", Path(path).stem), d.get("heading_deg", 0.0), d.get("baseline_m", 50.0),
                   d.get("duration_s", 300.0), sorted(d["events"], key=lambda e: e["t"]))


class Timeline:
    """Turns timed events into per-tick PilotInput. Velocities are sticky; others fire once."""

    def __init__(self, events: list[dict]):
        self.events = list(events)
        self.i = 0
        self.sticky = {"v_forward": 0.0, "v_right": 0.0, "v_up": 0.0}

    def step(self, t: float) -> tuple[PilotInput, list[dict]]:
        once: dict = {}
        faults: list[dict] = []
        while self.i < len(self.events) and self.events[self.i]["t"] <= t:
            e = self.events[self.i]
            self.i += 1
            for k, v in e.get("cmd", {}).items():
                if k in ONE_SHOT:
                    once[k] = v
                else:
                    self.sticky[k] = v
            if "fault" in e:
                faults.append(e["fault"])
        return PilotInput(**self.sticky, **once), faults


# ---------------------------------------------------------------------- pass 1: flight
def fly(scn: Scenario, cfg: Config, frame_fps: float, backend=None):
    from .sim import SimBackend
    backend = backend or SimBackend(cfg.sim)
    ctrl = FormationController(cfg.controller, *backend.launch_positions,
                               heading_deg=scn.heading_deg, baseline_m=scn.baseline_m)
    tl = Timeline(scn.events)
    dt = cfg.sim.dt
    rows, frames, events = [], [], []
    next_frame = 0.0
    landed_at = None
    t = 0.0
    while t < scn.duration_s:
        pilot, faults = tl.step(t)
        if pilot.start and ctrl.mode == Mode.IDLE:
            backend.launch()
        for fl in faults:
            backend.vehicles["LR".index(fl["vehicle"])].healthy = fl.get("healthy", False)
        states = backend.states()
        out = ctrl.update(t, dt, pilot, states)
        events += out.events
        backend.step(dt, out.left, out.right)
        pl, pr = backend.true_positions()
        cams = backend.camera_poses()
        f = out.formation
        rows.append({
            "t": round(t, 3), "mode": out.mode.value, "alpha": round(out.stereo_alpha, 3),
            "heading": round(f.heading_deg, 2), "baseline_cmd": round(f.baseline_m, 2),
            "sep_true": round(float(np.linalg.norm(pl - pr)), 3),
            "sep_h_true": round(float(np.linalg.norm((pl - pr)[:2])), 3),
            "dz_true": round(float(pl[2] - pr[2]), 3),
            "err_l": round(float(np.linalg.norm(states[0].pos - out.left.pos)), 3),
            "err_r": round(float(np.linalg.norm(states[1].pos - out.right.pos)), 3),
            **{f"{n}_{a}": round(float(v), 3) for n, p in (("l", pl), ("r", pr)) for a, v in zip("xyz", p)},
            "cam_dyaw": round(float(((cams[1].yaw_deg - cams[0].yaw_deg + 180) % 360) - 180), 3),
            "cam_dpitch": round(cams[1].pitch_deg - cams[0].pitch_deg, 3),
            "cam_droll": round(cams[1].roll_deg - cams[0].roll_deg, 3),
        })
        if frame_fps > 0 and t >= next_frame and out.mode in (FORMATION_MODES | {Mode.HOLD}):
            next_frame = t + 1.0 / frame_fps
            frames.append({
                "t": t, "mode": out.mode.value, "alpha": out.stereo_alpha, "mono": out.mono_source,
                "baseline_est": float(np.linalg.norm(states[0].pos - states[1].pos)),
                "agl_est": float((states[0].pos[2] + states[1].pos[2]) / 2),
                "heading": f.heading_deg,
                "poses": [asdict(c) for c in cams],
                "est": [{"pos": s.pos.tolist(), "yaw": s.cam_yaw_deg, "pitch": s.cam_pitch_deg,
                         "roll": s.cam_roll_deg} for s in states],
            })
        if out.land_now:
            backend.land()
        if out.mode == Mode.LANDED:
            landed_at = landed_at if landed_at is not None else t
            if t - landed_at > 1.0:
                break
        t += dt
    backend.close()
    return rows, frames, events


def summarize_flight(rows: list[dict], events: list[str], cfg: Config) -> dict:
    t = np.array([r["t"] for r in rows])
    mode = np.array([r["mode"] for r in rows])
    stereo = mode == Mode.STEREO.value
    dt = cfg.sim.dt
    s = {
        "duration_s": float(t[-1]) if len(t) else 0.0,
        "final_mode": mode[-1] if len(mode) else None,
        "time_in_stereo_s": float(stereo.sum() * dt),
        "min_separation_m": float(min(r["sep_true"] for r in rows)),
        "min_separation_airborne_m": float(min((r["sep_true"] for r in rows
                                                 if min(r["l_z"], r["r_z"]) > 2.0), default=np.nan)),
        "holds": sum("-> HOLD" in e for e in events),
    }
    if stereo.any():
        bcmd = np.array([r["baseline_cmd"] for r in rows])[stereo]
        sep = np.array([r["sep_h_true"] for r in rows])[stereo]
        dz = np.array([r["dz_true"] for r in rows])[stereo]
        s["stereo_baseline_err_rms_m"] = float(np.sqrt(np.mean((sep - bcmd) ** 2)))
        s["stereo_baseline_err_max_m"] = float(np.max(np.abs(sep - bcmd)))
        s["stereo_dz_max_m"] = float(np.max(np.abs(dz)))
        for k in ("cam_dyaw", "cam_dpitch", "cam_droll"):
            s[f"stereo_{k}_p95_deg"] = float(np.percentile(np.abs([r[k] for r in np.array(rows)[stereo]]), 95))
    # Time from each maneuver start (STEREO -> FADING_OUT) back to STEREO.
    outs, gaps = None, []
    for r in rows:
        if r["mode"] == Mode.FADING_OUT.value and outs is None:
            outs = r["t"]
        if r["mode"] == Mode.STEREO.value and outs is not None:
            gaps.append(round(r["t"] - outs, 2))
            outs = None
    s["stereo_gaps_s"] = gaps
    return s


# ---------------------------------------------------------------------- pass 2: render
_W: dict = {}


def _worker_init(cfg_dict):
    from .camera import intrinsics
    from .stereo import FeatureRectifier, make_dem_depth_fn
    from .terrain import Terrain
    cfg = Config(**{k: v for k, v in cfg_dict.items()})
    _W["cfg"] = cfg
    _W["T"] = Terrain.generate()
    _W["K"] = intrinsics(cfg.camera.width, cfg.camera.height, cfg.camera.hfov_deg)
    _W["dem"] = make_dem_depth_fn(_W["T"])
    _W["rect"] = FeatureRectifier(_W["K"])


def _render_chunk(args):
    import cv2

    from .camera import cam_rotation
    from .stereo import auto_convergence, comfort_metrics, compose_sbs, ground_mask
    chunk, outdir = args
    cfg, T, K, dem, rect = _W["cfg"], _W["T"], _W["K"], _W["dem"], _W["rect"]
    W, H = cfg.camera.width, cfg.camera.height
    rect.reset()
    results = []
    for idx, fr in chunk:
        imgs, rngs, Rs, ps = [], [], [], []
        for p in fr["poses"]:
            R = cam_rotation(p["yaw_deg"], p["pitch_deg"], p["roll_deg"])
            img, rg = T.render(np.array(p["pos"]), R, K, W, H)
            imgs.append(img); rngs.append(rg); Rs.append(R); ps.append(np.array(p["pos"]))
        res = rect.estimate(imgs[0], imgs[1])
        el, er = fr["est"]
        Rl_est = cam_rotation(el["yaw"], el["pitch"], el["roll"])
        conv = auto_convergence(res, K, Rl_est, np.array(el["pos"]), fr["baseline_est"], dem)
        ml = ground_mask(K, W, H, el["yaw"], el["pitch"], el["roll"], fr["agl_est"], fr["baseline_est"], cfg.comfort)
        mr = ground_mask(K, W, H, er["yaw"], er["pitch"], er["roll"], fr["agl_est"], fr["baseline_est"], cfg.comfort)
        cm = comfort_metrics(K, W, H, ((ps[0], Rs[0]), (ps[1], Rs[1])), rngs, conv.H, conv.shift_px, ml, cfg.comfort)
        sbs = compose_sbs(imgs[0], imgs[1], conv.H, conv.shift_px, ml, mr, fr["alpha"], fr["mono"])
        frame = (np.clip(sbs, 0, 1)[..., ::-1] * 255).astype(np.uint8).copy()
        _hud(frame, W, fr, res, conv)
        cv2.imwrite(str(Path(outdir) / f"f{idx:05d}.png"), frame)
        results.append({"idx": idx, "t": round(fr["t"], 2), "mode": fr["mode"], "alpha": round(fr["alpha"], 3),
                        "lock": bool(res.ok), "inliers": res.inliers, "resid_px": round(res.resid_vert_px, 3),
                        "conv_method": conv.method, "yaw_corr_deg": round(conv.yaw_corr_deg, 3),
                        **cm.as_dict()})
    return results


def _hud(frame, W, fr, res, conv):
    """Draw the HUD in both eyes with a small crossed disparity so it floats in front."""
    import cv2
    lines = [f"{fr['mode']}  B={fr['baseline_est']:.0f}m  hdg={fr['heading']:.0f}",
             f"lock={'Y' if res.ok else 'N'} ({res.inliers})  stereo={fr['alpha']:.2f}"]
    for eye, dx in ((0, 6), (1, -6)):
        for i, s in enumerate(lines):
            org = (eye * W + 8 + dx, 16 + 15 * i)
            cv2.putText(frame, s, org, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, s, org, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)


def render_frames(frames, cfg: Config, outdir: Path, workers: int) -> list[dict]:
    outdir.mkdir(parents=True, exist_ok=True)
    items = list(enumerate(frames))
    n = max(1, workers)
    size = max(1, int(np.ceil(len(items) / n)))
    chunks = [(items[i:i + size], str(outdir)) for i in range(0, len(items), size)]
    cfg_dict = {f.name: getattr(cfg, f.name) for f in fields(cfg)}
    with ProcessPoolExecutor(n, initializer=_worker_init, initargs=(cfg_dict,)) as ex:
        out = [r for chunk in ex.map(_render_chunk, chunks) for r in chunk]
    return sorted(out, key=lambda r: r["idx"])


def summarize_frames(fm: list[dict]) -> dict:
    shown = [r for r in fm if r["alpha"] > 0]
    full = [r for r in fm if r["alpha"] >= 1.0]
    s = {"frames": len(fm), "stereo_frames": len(full),
         "stereo_frames_without_lock": sum(not r["lock"] for r in shown)}
    if full:
        for k in ("vert_p95_deg", "max_divergence_deg", "crossed_p99_deg", "violation_frac", "masked_frac"):
            v = np.array([r[k] for r in full])
            s[f"{k}_median"] = round(float(np.median(v)), 4)
            s[f"{k}_worst"] = round(float(v.max()), 4)
        s["divergent_frames"] = sum(r["max_divergence_deg"] > 0.1 for r in full)
    return s


def encode(outdir: Path, fps: float, mp4: Path) -> bool:
    if not shutil.which("ffmpeg"):
        return False
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", str(outdir / "f%05d.png"),
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(mp4)]
    return subprocess.run(cmd).returncode == 0


# ---------------------------------------------------------------------- CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description="Run a hyperstereo formation scenario in simulation.")
    ap.add_argument("scenario")
    ap.add_argument("--out", default="out")
    ap.add_argument("--render", action="store_true", help="render SBS video and score stereo comfort")
    ap.add_argument("--fps", type=float, default=4.0)
    ap.add_argument("--width", type=int, default=480)
    ap.add_argument("--height", type=int, default=270)
    ap.add_argument("--hfov", type=float, default=60.0)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--rtk", action="store_true", help="RTK-grade GPS (2 cm) instead of ~1 m")
    ap.add_argument("--wind", type=float, default=None, help="mean wind speed m/s (from the west)")
    ap.add_argument("--sitl", action="store_true", help="fly two ArduPilot SITL instances instead of the built-in sim")
    a = ap.parse_args(argv)

    cfg = Config()
    cfg.sim.seed = a.seed
    cfg.camera.width, cfg.camera.height, cfg.camera.hfov_deg = a.width, a.height, a.hfov
    if a.rtk:
        cfg.sim.gps_sigma_m = 0.02
    if a.wind is not None:
        cfg.sim.wind_mean = (0.0, a.wind, 0.0)
    scn = Scenario.load(a.scenario)
    out = Path(a.out) / scn.name
    out.mkdir(parents=True, exist_ok=True)

    backend = None
    if a.sitl:
        from .sitl import SitlBackend
        backend = SitlBackend(cfg.sim)
    rows, frames, events = fly(scn, cfg, a.fps if a.render else 0.0, backend)
    with open(out / "telemetry.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (out / "events.log").write_text("\n".join(events) + "\n")
    summary = {"scenario": scn.name, "flight": summarize_flight(rows, events, cfg)}
    print("\n".join(events))

    if a.render and frames:
        fdir = out / "frames"
        if fdir.exists():
            shutil.rmtree(fdir)
        fm = render_frames(frames, cfg, fdir, a.workers)
        with open(out / "frames.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(fm[0]))
            w.writeheader()
            w.writerows(fm)
        summary["stereo"] = summarize_frames(fm)
        if encode(fdir, a.fps, out / "sbs.mp4"):
            summary["video"] = str(out / "sbs.mp4")
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()
