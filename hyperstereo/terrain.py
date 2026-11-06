"""Procedural landscape plus a vectorised CPU heightfield raymarcher.

The terrain is a ~16 km square centred on the launch point. The area within about
1.5 km of launch is gently rolling, and mountains rise beyond 2-3 km (the distances a
50-100 m baseline needs). There's a lake to the north-east to test stereo lock over
featureless water.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .camera import pixel_grid, pixel_rays

SKY_TOP = np.array([0.38, 0.55, 0.85])
SKY_HORIZON = np.array([0.78, 0.84, 0.92])
HAZE = np.array([0.74, 0.80, 0.88])
SUN_DIR = np.array([-0.4, 0.5, 0.75]) / np.linalg.norm([-0.4, 0.5, 0.75])


def _spectral_noise(n: int, beta: float, rng: np.random.Generator) -> np.ndarray:
    kx = np.fft.fftfreq(n)[:, None]
    ky = np.fft.fftfreq(n)[None, :]
    k = np.sqrt(kx * kx + ky * ky)
    k[0, 0] = 1.0
    amp = k ** (-beta)
    amp[0, 0] = 0.0
    phase = rng.uniform(0, 2 * np.pi, (n, n))
    f = np.real(np.fft.ifft2(amp * np.exp(1j * phase)))
    return (f - f.mean()) / f.std()


@dataclass
class Terrain:
    heights: np.ndarray      # (n, n) metres, relative to launch altitude
    albedo: np.ndarray       # (n, n, 3)
    normals: np.ndarray      # (n, n, 3)
    detail: np.ndarray       # (m, m) fine texture tiled over the ground
    cell: float              # metres per cell
    origin: float            # world coordinate of index 0 (same for x and y)
    water_level: float

    @classmethod
    def generate(cls, seed: int = 7, n: int = 1024, cell: float = 16.0) -> "Terrain":
        rng = np.random.default_rng(seed)
        origin = -n * cell / 2.0
        coords = origin + np.arange(n) * cell
        X, Y = np.meshgrid(coords, coords, indexing="ij")  # X = north, Y = east
        r = np.hypot(X, Y)

        base = _spectral_noise(n, 1.6, rng)
        ridge = np.clip(1.0 - np.abs(_spectral_noise(n, 1.45, rng)) / 2.5, 0.0, 1.0) ** 2
        ramp = np.clip((r - 1500.0) / 4000.0, 0.0, 1.0) ** 1.3
        h = 12.0 * base + ramp * (450.0 + 220.0 * base + 700.0 * ridge)
        # A lake basin ~2.5 km north-east of launch.
        lake = np.exp(-(((X - 1800.0) ** 2 + (Y - 1700.0) ** 2) / (2 * 600.0 ** 2)))
        h -= 60.0 * lake
        # Launch site at altitude 0.
        i0 = n // 2
        h -= h[i0, i0]
        water = -18.0

        gx, gy = np.gradient(h, cell)
        nrm = np.stack([-gx, -gy, np.ones_like(h)], axis=-1)
        nrm /= np.linalg.norm(nrm, axis=-1, keepdims=True)
        slope = 1.0 - nrm[..., 2]

        colour_noise = _spectral_noise(n, 0.6, rng)[..., None]
        grass = np.array([0.30, 0.42, 0.18]) + 0.06 * colour_noise * np.array([1.0, 1.2, 0.6])
        forest = np.array([0.16, 0.27, 0.13])
        rock = np.array([0.45, 0.42, 0.38])
        snow = np.array([0.92, 0.93, 0.96])
        forest_mask = np.clip(_spectral_noise(n, 1.3, rng) * 1.5, 0.0, 1.0)[..., None]
        alb = grass * (1 - forest_mask) + forest * forest_mask
        rock_mask = np.clip((slope - 0.12) * 8.0, 0.0, 1.0)[..., None]
        alb = alb * (1 - rock_mask) + rock * rock_mask
        snow_mask = np.clip((h - 850.0) / 120.0 + 0.3 * colour_noise[..., 0], 0.0, 1.0)[..., None]
        alb = alb * (1 - snow_mask) + snow * snow_mask
        under = (h < water)[..., None]
        alb = np.where(under, np.array([0.12, 0.22, 0.32]), alb)
        h = np.maximum(h, water)
        nrm = np.where(under, np.array([0.0, 0.0, 1.0]), nrm)

        detail = 0.75 + 0.25 * np.tanh(_spectral_noise(256, 0.9, rng))
        return cls(h, alb, nrm, detail, cell, origin, water)

    # ------------------------------------------------------------------ sampling
    def _idx(self, x, y):
        fx = np.clip((x - self.origin) / self.cell, 0.0, self.heights.shape[0] - 1.001)
        fy = np.clip((y - self.origin) / self.cell, 0.0, self.heights.shape[1] - 1.001)
        ix, iy = fx.astype(np.int32), fy.astype(np.int32)
        return ix, iy, fx - ix, fy - iy

    def _bilerp(self, grid, ix, iy, tx, ty):
        if grid.ndim == 3:
            tx, ty = tx[..., None], ty[..., None]
        return ((grid[ix, iy] * (1 - tx) + grid[ix + 1, iy] * tx) * (1 - ty)
                + (grid[ix, iy + 1] * (1 - tx) + grid[ix + 1, iy + 1] * tx) * ty)

    def height(self, x, y):
        return self._bilerp(self.heights, *self._idx(np.asarray(x, float), np.asarray(y, float)))

    def in_bounds(self, x, y):
        hi = self.origin + (self.heights.shape[0] - 1) * self.cell
        return (x > self.origin) & (x < hi) & (y > self.origin) & (y < hi)

    # ------------------------------------------------------------------ rendering
    def raymarch(self, o: np.ndarray, d: np.ndarray, t_max: float | None = None,
                 max_iter: int = 600) -> np.ndarray:
        """Distance to the first terrain hit along each ray (inf for sky). d: (N, 3) unit."""
        if t_max is None:  # far enough to cross the whole map diagonally
            t_max = 1.5 * self.heights.shape[0] * self.cell
        n = d.shape[0]
        t = np.full(n, 2.0)
        result = np.full(n, np.inf)
        idx = np.arange(n)
        hmax = float(self.heights.max())
        t_prev = t.copy()
        for _ in range(max_iter):
            if idx.size == 0:
                break
            p = o + d[idx] * t[idx, None]
            gap = p[:, 2] - self.height(p[:, 0], p[:, 1])
            hit = gap < 0.0
            if hit.any():
                hi_idx = idx[hit]
                lo, hi = t_prev[hi_idx].copy(), t[hi_idx].copy()
                for _ in range(8):  # bisection refine
                    mid = 0.5 * (lo + hi)
                    pm = o + d[hi_idx] * mid[:, None]
                    below = pm[:, 2] < self.height(pm[:, 0], pm[:, 1])
                    hi = np.where(below, mid, hi)
                    lo = np.where(below, lo, mid)
                result[hi_idx] = hi
            escaped = (t[idx] > t_max) | ~self.in_bounds(p[:, 0], p[:, 1]) | \
                      ((p[:, 2] > hmax) & (d[idx, 2] >= 0))
            keep = ~hit & ~escaped
            idx, gap = idx[keep], gap[keep]
            t_prev[idx] = t[idx]
            t[idx] += np.maximum(gap * 0.5, 0.5 + t[idx] * 0.004)
        return result

    def shade(self, o: np.ndarray, d: np.ndarray, rng_t: np.ndarray) -> np.ndarray:
        n = d.shape[0]
        out = np.empty((n, 3))
        sky = ~np.isfinite(rng_t)
        e = np.clip(d[:, 2], 0.0, 1.0)[:, None]
        sky_col = SKY_HORIZON * (1 - e) ** 3 + SKY_TOP * (1 - (1 - e) ** 3)
        sun = np.clip(d @ SUN_DIR, 0.0, 1.0)[:, None] ** 200
        out[sky] = (sky_col + sun * np.array([1.0, 0.95, 0.8]))[sky]
        g = ~sky
        if g.any():
            p = o + d[g] * rng_t[g, None]
            ix, iy, tx, ty = self._idx(p[:, 0], p[:, 1])
            alb = self._bilerp(self.albedo, ix, iy, tx, ty)
            nrm = self._bilerp(self.normals, ix, iy, tx, ty)
            m = self.detail.shape[0]
            dx = ((p[:, 0] / 3.0) % m).astype(np.int32)
            dy = ((p[:, 1] / 3.0) % m).astype(np.int32)
            water = p[:, 2] <= self.water_level + 0.05
            det = np.where(water, 1.0, self.detail[dx, dy])[:, None]
            light = 0.35 + 0.75 * np.clip(nrm @ SUN_DIR, 0.0, 1.0)[:, None]
            col = alb * det * light
            fog = 1.0 - np.exp(-rng_t[g] / 9000.0)[:, None]
            out[g] = col * (1 - fog) + HAZE * fog
        return np.clip(out, 0.0, 1.0)

    def render(self, cam_pos: np.ndarray, R: np.ndarray, K: np.ndarray, width: int, height: int):
        """Returns (image float32 HxWx3 in 0..1, range HxW in metres, inf for sky)."""
        u, v = pixel_grid(width, height)
        d = pixel_rays(K, R, u, v).reshape(-1, 3)
        o = np.asarray(cam_pos, float)
        t = self.raymarch(o, d)
        img = self.shade(o, d, t)
        return img.reshape(height, width, 3).astype(np.float32), t.reshape(height, width)
