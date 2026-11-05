"""Stereo pipeline prototype: rectification, convergence, ground mask, SBS composition,
and ground-truth comfort metrics.

Design notes (things the Phase 0 sim is meant to validate):

* The rectifier estimates the relative camera rotation from image features alone,
  fitting only the *vertical* residuals. Horizontal offsets are a mix of true parallax
  and yaw mismatch, which can't be told apart, so yaw error is absorbed by
  auto-convergence instead.
* In an HMD, zero disparity means optical infinity. Auto-convergence therefore places
  the farthest matched features just in front of infinity, so a compass/gimbal yaw
  mismatch can never make the eyes diverge.
* The ground mask is computed from camera attitude + altitude assuming flat ground
  (it needs no depth sensing, so it works on real hardware). Ground-truth metrics then
  report how much uncomfortable parallax still gets through (hills, etc.).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .camera import cam_rotation, pixel_grid, pixel_rays, project
from .config import ComfortConfig


# ---------------------------------------------------------------------- rotation helpers
def small_rotation(a: float, b: float, c: float) -> np.ndarray:
    """Rotation by angles (rad) about camera x (pitch), y (yaw), z (roll)."""
    return cv2.Rodrigues(np.array([a, b, c], float))[0]


@dataclass
class RectifyResult:
    ok: bool
    H: np.ndarray                     # 3x3 homography applied to the right image
    angles_deg: np.ndarray            # estimated (pitch, yaw, roll) of right relative to left
    inliers: int
    resid_vert_px: float              # median |vertical residual| of inliers after the fit
    pts_l: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    pts_r_rect: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))


class FeatureRectifier:
    """Estimates right-vs-left camera rotation from ORB matches (vertical residuals only)."""

    def __init__(self, K: np.ndarray, min_inliers: int = 40, max_resid_px: float = 1.0,
                 smoothing: float = 0.5):
        self.K = K
        self.Kinv = np.linalg.inv(K)
        self.orb = cv2.ORB_create(nfeatures=1500, fastThreshold=10)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.min_inliers = min_inliers
        self.max_resid_px = max_resid_px
        self.smoothing = smoothing
        self.prev: np.ndarray | None = None
        self.rng = np.random.default_rng(0)

    def _norm(self, pts):
        return (pts - self.K[[0, 1], [2, 2]]) / self.K[0, 0]

    def _solve_lin(self, xl, yl, xr, yr):
        # dy = y_l - y_r ~= -a(1+y^2) + b*x*y + c*x  (small-angle, right-camera coordinates)
        A = np.column_stack([-(1 + yr ** 2), xr * yr, xr])
        sol, *_ = np.linalg.lstsq(A, yl - yr, rcond=None)
        return sol

    def _vert_resid(self, ang, nl, nr):
        R = small_rotation(*ang)
        r = np.column_stack([nr, np.ones(len(nr))]) @ R.T
        return nl[:, 1] - r[:, 1] / r[:, 2], r[:, 0] / r[:, 2]

    def estimate(self, img_l: np.ndarray, img_r: np.ndarray) -> RectifyResult:
        gl = cv2.cvtColor((img_l * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        gr = cv2.cvtColor((img_r * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        kl, dl = self.orb.detectAndCompute(gl, None)
        kr, dr = self.orb.detectAndCompute(gr, None)
        fail = RectifyResult(False, np.eye(3), np.zeros(3), 0, np.inf)
        if dl is None or dr is None or len(kl) < 10 or len(kr) < 10:
            return fail
        m = self.matcher.match(dl, dr)
        if len(m) < 10:
            return fail
        pl = np.array([kl[x.queryIdx].pt for x in m])
        pr = np.array([kr[x.trainIdx].pt for x in m])
        nl, nr = self._norm(pl), self._norm(pr)
        tol = self.max_resid_px * 1.5 / self.K[0, 0]
        best, best_in = None, None
        for _ in range(200):
            s = self.rng.choice(len(m), 3, replace=False)
            ang = self._solve_lin(nl[s, 0], nl[s, 1], nr[s, 0], nr[s, 1])
            res, xr = self._vert_resid(ang, nl, nr)
            inl = (np.abs(res) < tol) & (nl[:, 0] - xr > -0.02)  # disallow strongly divergent matches
            if best_in is None or inl.sum() > best_in.sum():
                best, best_in = ang, inl
        ang = best
        for _ in range(3):  # refine on inliers, then re-select
            ang = ang + self._solve_lin(*nl[best_in].T, *self._apply(ang, nr[best_in]).T)
            res, xr = self._vert_resid(ang, nl, nr)
            best_in = (np.abs(res) < tol) & (nl[:, 0] - xr > -0.02)
        n_in = int(best_in.sum())
        resid_px = float(np.median(np.abs(res[best_in])) * self.K[0, 0]) if n_in else np.inf
        ok = n_in >= self.min_inliers and resid_px <= self.max_resid_px
        if ok and self.prev is not None:
            ang = self.smoothing * self.prev + (1 - self.smoothing) * ang
        if ok:
            self.prev = ang
        elif self.prev is not None:
            ang = self.prev
        H = self.K @ small_rotation(*ang) @ self.Kinv
        pr_rect = cv2.perspectiveTransform(pr[best_in][None], H)[0] if n_in else np.zeros((0, 2))
        return RectifyResult(ok, H, np.degrees(ang), n_in, resid_px, pl[best_in], pr_rect)

    def _apply(self, ang, nr):
        r = np.column_stack([nr, np.ones(len(nr))]) @ small_rotation(*ang).T
        return r[:, :2] / r[:, 2:3]

    def reset(self):
        self.prev = None


def oracle_homography(K, R_l, R_r) -> np.ndarray:
    """Rotation-only rectification from true poses: maps right pixels into a camera
    with the left camera's orientation."""
    return K @ R_l.T @ R_r @ np.linalg.inv(K)


# ---------------------------------------------------------------------- convergence & mask
@dataclass
class Convergence:
    H: np.ndarray          # final homography for the right image (rectification + yaw correction)
    shift_px: float        # residual uniform shift; displayed disparity = d - shift
    yaw_corr_deg: float
    method: str            # "dem", "percentile" or "none"
    n_used: int = 0


def auto_convergence(rect: RectifyResult, K: np.ndarray, R_left_est: np.ndarray, pos_left_est: np.ndarray,
                     baseline_m: float, depth_fn=None, inf_margin_px: float = 1.0,
                     feat_margin_px: float = 0.5, pct: float = 3.0, baseline_rel_err: float = 0.04,
                     min_used: int = 15) -> Convergence:
    """Fix the horizontal alignment so optical infinity lands just in front of zero
    disparity, so the eyes never diverge.

    Yaw mismatch between the cameras adds an offset to every disparity, and matching
    alone can't separate it from parallax. A terrain elevation model (DEM) plus the
    GPS pose predicts each feature's distance, so (observed - predicted) disparity
    estimates the yaw error. Distant features are weighted up because their prediction
    barely depends on baseline (GPS) error. The correction is applied as a true yaw
    rotation, not a pixel shift, so it's exact across the whole field of view. Finally the
    shift is clamped so that no matched feature is pushed into divergence.
    """
    if rect.inliers < 10:
        return Convergence(rect.H, 0.0, 0.0, "none")
    method, n_used, yaw = "percentile", 0, 0.0
    if depth_fn is not None:
        d = rect.pts_l[:, 0] - rect.pts_r_rect[:, 0]
        rays = pixel_rays(K, R_left_est, rect.pts_l[:, 0], rect.pts_l[:, 1])
        t = depth_fn(np.asarray(pos_left_est, float), rays)
        zc = t * (rays @ R_left_est[:, 2])
        sel = np.isfinite(zc) & (zc > 50.0)
        if sel.sum() >= min_used:
            pred = K[0, 0] * baseline_m / zc[sel]
            resid = d[sel] - pred
            w = 1.0 / (0.5 + baseline_rel_err * pred) ** 2
            order = np.argsort(resid)
            cw = np.cumsum(w[order])
            offset = float(resid[order][np.searchsorted(cw, cw[-1] / 2)])  # weighted median
            yaw = float(np.arctan2(offset, K[0, 0]))
            method, n_used = "dem", int(sel.sum())
    H = K @ small_rotation(0.0, yaw, 0.0) @ np.linalg.inv(K) @ rect.H
    pr = rect.pts_r_rect
    if len(pr):
        pr = cv2.perspectiveTransform(pr[None], K @ small_rotation(0.0, yaw, 0.0) @ np.linalg.inv(K))[0]
    d = rect.pts_l[:, 0] - pr[:, 0]
    cap = float(np.percentile(d, pct) - feat_margin_px)
    shift = min(-inf_margin_px, cap) if method == "dem" else cap
    return Convergence(H, shift, float(np.degrees(yaw)), method, n_used)


def make_dem_depth_fn(terrain, blur_cells: float = 2.0, noise_m: float = 5.0, seed: int = 11):
    """Depth predictor from a degraded copy of the terrain, emulating a 30 m public DEM."""
    from scipy.ndimage import gaussian_filter
    from dataclasses import replace as _replace
    rng = np.random.default_rng(seed)
    h = gaussian_filter(terrain.heights, blur_cells) + gaussian_filter(
        rng.normal(0, noise_m * 3, terrain.heights.shape), 1.5)
    dem = _replace(terrain, heights=h)

    def depth_fn(origin, rays):
        return dem.raymarch(origin, rays)
    return depth_fn


def ground_mask(K, width, height, cam_yaw, cam_pitch, cam_roll, agl_m, baseline_m,
                comfort: ComfortConfig, soft_deg: float = 1.0, step: int = 1) -> np.ndarray:
    """0..1 mask of pixels whose *flat-ground* parallax would exceed the comfort limit."""
    R = cam_rotation(cam_yaw, cam_pitch, cam_roll)
    u, v = pixel_grid(width, height, step)
    d = pixel_rays(K, R, u, v)
    elev = np.degrees(np.arcsin(np.clip(d[..., 2], -1, 1)))     # negative = below horizon
    mag = comfort.display_hfov_deg / np.degrees(2 * np.arctan(width / (2 * K[0, 0])))
    max_rad = np.radians(comfort.max_crossed_disparity_deg) / mag
    z_min = baseline_m / max_rad
    e_lim = -np.degrees(np.arctan2(max(agl_m, 1.0), z_min))
    return np.clip((e_lim - elev) / soft_deg + 0.5, 0.0, 1.0)


def apply_mask(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(img, (0, 0), 6)
    gray = blur.mean(axis=2, keepdims=True)
    dim = 0.25 * gray + 0.15 * blur
    m = mask[..., None]
    return img * (1 - m) + dim * m


def shift_h(img: np.ndarray, dx: float) -> np.ndarray:
    M = np.float32([[1, 0, dx], [0, 1, 0]])
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), borderMode=cv2.BORDER_REPLICATE)


def compose_sbs(img_l, img_r, H, shift_px, mask_l, mask_r, alpha, mono_source="L"):
    """Build the side-by-side frame shown in the headset."""
    w, h = img_l.shape[1], img_l.shape[0]
    right = cv2.warpPerspective(img_r, H, (w, h), borderMode=cv2.BORDER_REPLICATE)
    left = shift_h(apply_mask(img_l, mask_l), -shift_px / 2)
    right = shift_h(apply_mask(right, mask_r), +shift_px / 2)
    if alpha < 1.0:
        mono = img_l if mono_source == "L" else img_r  # 2D view: unmasked, zero disparity
        left = alpha * left + (1 - alpha) * mono
        right = alpha * right + (1 - alpha) * mono
    return np.hstack([left, right])


# ---------------------------------------------------------------------- ground-truth metrics
@dataclass
class ComfortMetrics:
    vert_p95_deg: float              # |vertical disparity| 95th percentile (after rectification)
    divergent_frac: float            # fraction of pixels displayed beyond infinity
    max_divergence_deg: float
    crossed_p99_deg: float           # displayed crossed disparity, unmasked pixels
    violation_frac: float            # unmasked pixels exceeding the crossed-disparity limit
    masked_frac: float
    raw_vert_p95_deg: float          # before rectification, for reference

    def as_dict(self):
        return {k: round(float(v), 4) for k, v in self.__dict__.items()}


def true_disparity(K, width, height, pos_l, R_l, pos_r, R_r, range_l, H, shift_px, step=4):
    """Per-pixel displayed (dx, dy) in pixels for the left image, using ground truth.

    Pixels at infinity (sky) are projected as directions.
    """
    u, v = pixel_grid(width, height, step)
    rng = range_l[::step, ::step]
    d = pixel_rays(K, R_l, u, v)
    finite = np.isfinite(rng)
    P = pos_l + d * np.where(finite, rng, 0.0)[..., None]
    ur, vr, _ = project(P, pos_r, R_r, K)
    ui, vi, _ = project(d, pos_r, R_r, K, at_infinity=True)
    ur = np.where(finite, ur, ui)
    vr = np.where(finite, vr, vi)
    raw_dy = v - vr
    q = np.stack([ur, vr, np.ones_like(ur)], -1) @ H.T
    ur2, vr2 = q[..., 0] / q[..., 2], q[..., 1] / q[..., 2]
    dx = (u - shift_px / 2) - (ur2 + shift_px / 2)
    dy = v - vr2
    return dx, dy, raw_dy


def comfort_metrics(K, width, height, poses, ranges, H, shift_px, mask_l, comfort: ComfortConfig,
                    step=4) -> ComfortMetrics:
    (pos_l, R_l), (pos_r, R_r) = poses
    dx, dy, raw_dy = true_disparity(K, width, height, pos_l, R_l, pos_r, R_r, ranges[0], H, shift_px, step)
    deg_per_px = comfort.display_hfov_deg / width
    dxd, dyd = dx * deg_per_px, np.abs(dy) * deg_per_px
    m = mask_l[::step, ::step] if mask_l.shape != dx.shape else mask_l
    visible = m < 0.5
    lim = comfort.max_crossed_disparity_deg
    vis_dx = dxd[visible] if visible.any() else np.array([0.0])
    return ComfortMetrics(
        vert_p95_deg=float(np.percentile(dyd[visible], 95)) if visible.any() else 0.0,
        divergent_frac=float((vis_dx < -0.05).mean()),
        max_divergence_deg=float(max(0.0, -vis_dx.min())),
        crossed_p99_deg=float(np.percentile(vis_dx, 99)),
        violation_frac=float((vis_dx > lim).mean()),
        masked_frac=float(1 - visible.mean()),
        raw_vert_p95_deg=float(np.percentile(np.abs(raw_dy) * deg_per_px, 95)),
    )
