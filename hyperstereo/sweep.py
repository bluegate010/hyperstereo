"""Comfort sweep: render the same scene across baselines / lenses / altitudes.

Produces SBS stills, a "baseline ladder" video (each setting held a few seconds, for
A/B viewing in the headset) and a metrics table. The ground-truth numbers say what the
pipeline delivers. Only eyes in a headset can say what's comfortable, so record those
judgements in the `comfort_rating` column of results.csv.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np

from .camera import cam_rotation, intrinsics
from .config import ComfortConfig
from .geometry import right_vec
from .stereo import FeatureRectifier, auto_convergence, comfort_metrics, compose_sbs, ground_mask, make_dem_depth_fn
from .terrain import Terrain


def render_case(T, dem, B, hfov, agl, yaw, W, H, comfort, rng, center_xy=(0.0, 0.0), mask=True):
    K = intrinsics(W, H, hfov)
    ground = float(T.height(*center_xy))
    c = np.array([center_xy[0], center_xy[1], ground + agl])
    pl, pr = c - right_vec(yaw) * B / 2, c + right_vec(yaw) * B / 2
    # Realistic residual errors after gimbal stabilisation
    Rl = cam_rotation(yaw + rng.normal(0, 1.0), -3 + rng.normal(0, 0.3), rng.normal(0, 0.3))
    Rr = cam_rotation(yaw + rng.normal(0, 1.0), -3 + rng.normal(0, 0.3), rng.normal(0, 0.3))
    il, rl = T.render(pl, Rl, K, W, H)
    ir, rr = T.render(pr, Rr, K, W, H)
    res = FeatureRectifier(K).estimate(il, ir)
    conv = auto_convergence(res, K, cam_rotation(yaw, -3, 0), pl + rng.normal(0, 1, 3),
                            B + rng.normal(0, 1.4), dem)
    ml = ground_mask(K, W, H, yaw, -3, 0, agl, B, comfort) if mask else np.zeros((H, W))
    cm = comfort_metrics(K, W, H, ((pl, Rl), (pr, Rr)), (rl, rr), conv.H, conv.shift_px, ml, comfort)
    sbs = compose_sbs(il, ir, conv.H, conv.shift_px, ml, ml, 1.0)
    img = (np.clip(sbs, 0, 1)[..., ::-1] * 255).astype(np.uint8).copy()
    label = f"B={B:.0f}m  FOV={hfov:.0f}  AGL={agl:.0f}m  scale x{B / comfort.ipd_m:.0f}"
    for eye, dx in ((0, 6), (1, -6)):
        org = (eye * W + 8 + dx, H - 12)
        cv2.putText(img, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return img, res, conv, cm


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="out/sweep")
    ap.add_argument("--baselines", default="25,50,75,100,150")
    ap.add_argument("--hfovs", default="45,60")
    ap.add_argument("--agls", default="100")
    ap.add_argument("--yaw", type=float, default=0.0, help="view heading (0 = north toward lake/mountains)")
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=540)
    ap.add_argument("--hold", type=float, default=6.0, help="seconds per setting in the ladder video")
    ap.add_argument("--no-mask", action="store_true")
    a = ap.parse_args(argv)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    comfort = ComfortConfig()
    T = Terrain.generate()
    dem = make_dem_depth_fn(T)
    rng = np.random.default_rng(5)
    rows, stills = [], []
    for B, hfov, agl in itertools.product(*(map(float, s.split(",")) for s in (a.baselines, a.hfovs, a.agls))):
        img, res, conv, cm = render_case(T, dem, B, hfov, agl, a.yaw, a.width, a.height, comfort, rng,
                                         mask=not a.no_mask)
        name = f"sbs_B{B:.0f}_fov{hfov:.0f}_agl{agl:.0f}.png"
        cv2.imwrite(str(out / name), img)
        stills.append(out / name)
        rows.append({"file": name, "baseline_m": B, "hfov_deg": hfov, "agl_m": agl,
                     "scale_factor": round(B / comfort.ipd_m), "lock": res.ok, "inliers": res.inliers,
                     "conv": conv.method, **cm.as_dict(), "comfort_rating": ""})
        print(f"B={B:5.0f} fov={hfov:3.0f} agl={agl:4.0f}  lock={res.ok} vert95={cm.vert_p95_deg:.3f} "
              f"div={cm.max_divergence_deg:.3f} cross99={cm.crossed_p99_deg:.2f} "
              f"viol={cm.violation_frac:.3f} masked={cm.masked_frac:.2f}")
    with open(out / "results.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    if shutil.which("ffmpeg"):
        lst = out / "ladder.txt"
        lst.write_text("".join(f"file '{p.name}'\nduration {a.hold}\n" for p in stills)
                       + f"file '{stills[-1].name}'\n")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst.name,
                        "-vf", "fps=30", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "ladder.mp4"],
                       cwd=out, check=False)
        print(f"ladder video: {out / 'ladder.mp4'}")


if __name__ == "__main__":
    main()
