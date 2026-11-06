"""Real-time viewer server: fly the formation live with a joystick in an SBS headset.

    python -m hyperstereo.live            # built-in sim
    python -m hyperstereo.live --sitl     # two ArduPilot SITL instances (scripts/start_sitl.sh 1)

Then open http://localhost:8765 in a browser (Chrome/Edge/Firefox), go full screen
on the headset display, and plug in a gamepad.

Architecture: a thread runs the backend (sim or SITL) and the unchanged
FormationController at 50 Hz in real time. An aiohttp server serves the
three.js viewer and the terrain assets, and streams state snapshots over a
WebSocket. The browser renders both eyes on the GPU from the drones' *true*
camera poses and sends pilot input back.

Stereo correction runs in the browser as rotation-only rectification: the right eye
is rendered with the left camera's orientation, plus a small residual error
matching what the feature/DEM pipeline achieved offline (about 0.03 deg). Toggle it off
(T) to see the raw, misaligned pair. The ground mask uses the same comfort
model as the offline pipeline.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from .camera import cam_rotation
from .config import Config
from .controller import FormationController, Mode, PilotInput
from .stereo import small_rotation
from .terrain import Terrain

VIEWER_DIR = Path(__file__).parent / "viewer"
MAX_SPEED = 6.0      # m/s, full stick
MAX_CLIMB = 2.0      # m/s
PITCH_RATE = 10.0    # deg/s, full trigger
ONE_SHOT = {"rotate_deg", "baseline_delta", "start", "recover", "resume"}


class LiveSession:
    def __init__(self, cfg: Config, sitl: bool, heading: float, baseline: float, terrain: Terrain):
        self.cfg, self.sitl, self.heading, self.baseline, self.terrain = cfg, sitl, heading, baseline, terrain
        self.lock = threading.Lock()
        self.axes = {"fwd": 0.0, "right": 0.0, "up": 0.0, "pitch": 0.0}
        self.pending: list[dict] = []
        self.flags = {"rectify": True, "mask": True}
        self.events: deque[str] = deque(maxlen=8)
        self.snapshot: dict | None = None
        self.seq = 0
        self.stop = threading.Event()
        self.reset_req = False
        self.rng = np.random.default_rng(cfg.sim.seed + 100)
        self._new_flight()

    # ------------------------------------------------------------------ setup
    def _new_flight(self):
        if self.sitl:
            from .sitl import SitlBackend
            if getattr(self, "backend", None) is not None:
                self.backend.close()
            self.backend = SitlBackend(self.cfg.sim)
        else:
            from .sim import SimBackend
            self.cfg.sim.seed += 1
            self.backend = SimBackend(self.cfg.sim)
        self.ctrl = FormationController(self.cfg.controller, *self.backend.launch_positions,
                                        heading_deg=self.heading, baseline_m=self.baseline)
        self.rect_err = np.zeros(3)  # residual rectification error, deg (pitch, yaw, roll)
        self.flags.update(rectify=True, mask=True)  # don't let a comparison toggle leak into a new flight
        self.t = 0.0
        self.events.clear()
        self.events.append("Ready: press Start / Enter to launch")

    # ------------------------------------------------------------------ input
    def handle(self, msg: dict):
        with self.lock:
            kind = msg.get("type")
            if kind == "axes":
                for k in self.axes:
                    self.axes[k] = float(np.clip(msg.get(k, 0.0), -1.0, 1.0))
            elif kind == "cmd":
                self.pending.append({k: v for k, v in msg.get("cmd", {}).items() if k in ONE_SHOT})
            elif kind == "toggle" and msg.get("flag") in self.flags:
                self.flags[msg["flag"]] = not self.flags[msg["flag"]]
            elif kind == "fault":
                v = self.backend.vehicles["LR".index(msg.get("vehicle", "R"))]
                v.healthy = not v.healthy
                self.events.append(f"fault injection: {msg.get('vehicle', 'R')} healthy={v.healthy}")
            elif kind == "reset":
                self.reset_req = True

    def _pilot(self) -> PilotInput:
        with self.lock:
            a = dict(self.axes)
            cmds, self.pending = self.pending, []
        p = PilotInput(v_forward=a["fwd"] * MAX_SPEED, v_right=a["right"] * MAX_SPEED, v_up=a["up"] * MAX_CLIMB)
        if abs(a["pitch"]) > 0.02:
            p.view_pitch_deg = self.ctrl.view_pitch + a["pitch"] * PITCH_RATE * self.cfg.sim.dt
        for c in cmds:
            if "rotate_deg" in c:
                p.rotate_deg = (p.rotate_deg or 0.0) + float(c["rotate_deg"])
            if "baseline_delta" in c:
                base = p.baseline_m if p.baseline_m is not None else self.ctrl.baseline_target
                p.baseline_m = base + float(c["baseline_delta"])
            for k in ("start", "recover", "resume"):
                if c.get(k):
                    setattr(p, k, True)
        return p

    # ------------------------------------------------------------------ loop
    def run(self):
        dt = self.cfg.sim.dt
        wall0 = time.monotonic()
        while not self.stop.is_set():
            if self.reset_req:
                self.reset_req = False
                self._new_flight()
                wall0 = time.monotonic()
            pilot = self._pilot()
            if pilot.start and self.ctrl.mode == Mode.IDLE:
                self.events.append("launching...")
                self.backend.launch()
            states = self.backend.states()
            out = self.ctrl.update(self.t, dt, pilot, states)
            for e in out.events:
                self.events.append(e.strip())
            self.backend.step(dt, out.left, out.right)
            if out.land_now:
                self.backend.land()
            self._publish(out, states)
            self.t += dt
            if not self.sitl:  # SITL paces itself to its own clock
                lag = wall0 + self.t - time.monotonic()
                if lag > 0:
                    time.sleep(lag)
                elif lag < -0.5:
                    wall0 = time.monotonic() - self.t  # fell behind; don't try to catch up

    def _publish(self, out, states):
        cfg = self.cfg
        cams = self.backend.camera_poses()
        Rs = [cam_rotation(c.yaw_deg, c.pitch_deg, c.roll_deg) for c in cams]
        # Residual rectification error: slow random walk of a few hundredths of a degree.
        self.rect_err = 0.995 * self.rect_err + self.rng.normal(0, 0.003, 3)
        R_right_rect = Rs[0] @ small_rotation(*np.radians(self.rect_err))
        f = out.formation
        mid = (states[0].pos + states[1].pos) / 2
        agl = float(mid[2] - self.terrain.height(mid[0], mid[1]))
        b_est = float(np.linalg.norm(states[0].pos - states[1].pos))
        mag = cfg.comfort.display_hfov_deg / cfg.camera.hfov_deg
        max_rad = math.radians(cfg.comfort.max_crossed_disparity_deg) / mag
        e_lim = -math.degrees(math.atan2(max(agl, 1.0), max(b_est, 1.0) / max_rad))
        pl, pr = self.backend.true_positions()
        with self.lock:
            flags = dict(self.flags)
            events = list(self.events)
        snap = {
            "t": round(self.t, 3),
            "mode": out.mode.value, "alpha": out.stereo_alpha, "mono": out.mono_source,
            "heading": f.heading_deg, "baseline": f.baseline_m, "baseline_target": self.ctrl.baseline_target,
            "sep": float(np.linalg.norm(pl - pr)), "agl": agl, "view_pitch": self.ctrl.view_pitch,
            "cams": [{"pos": c.pos.tolist(), "R": R.T.flatten().tolist()} for c, R in zip(cams, Rs)],
            "right_rect_R": R_right_rect.T.flatten().tolist(),
            "e_lim_deg": e_lim, "conv_margin_deg": 0.05,
            "hfov": cfg.camera.hfov_deg, "flags": flags, "events": events,
            "healthy": [s.healthy for s in states], "sitl": self.sitl,
        }
        with self.lock:
            self.seq += 1
            self.snapshot = snap


# ---------------------------------------------------------------------- terrain assets
def terrain_assets(T: Terrain) -> dict[str, bytes]:
    n = T.heights.shape[0]
    alb = (np.clip(T.albedo, 0, 1) * 255).astype(np.uint8)[..., ::-1]
    det = (np.clip(T.detail, 0, 1) * 255).astype(np.uint8)
    meta = {"n": n, "cell": T.cell, "origin": T.origin, "water_level": T.water_level,
            "hmax": float(T.heights.max()), "detail_n": T.detail.shape[0]}
    return {
        "meta.json": json.dumps(meta).encode(),
        "height.bin": T.heights.astype("<f4").tobytes(),
        "albedo.png": cv2.imencode(".png", alb)[1].tobytes(),
        "detail.png": cv2.imencode(".png", det)[1].tobytes(),
    }


# ---------------------------------------------------------------------- web server
def make_app(session: LiveSession, assets: dict[str, bytes]):
    from aiohttp import WSMsgType, web

    async def index(_):
        return web.FileResponse(VIEWER_DIR / "index.html")

    async def terrain(req):
        name = req.match_info["name"]
        if name not in assets:
            raise web.HTTPNotFound()
        ctype = {"json": "application/json", "bin": "application/octet-stream", "png": "image/png"}[name.rsplit(".", 1)[1]]
        return web.Response(body=assets[name], content_type=ctype)

    async def ws_handler(req):
        ws = web.WebSocketResponse(heartbeat=10)
        await ws.prepare(req)

        async def sender():
            last = -1
            while not ws.closed:
                with session.lock:
                    seq, snap = session.seq, session.snapshot
                if snap is not None and seq != last:
                    last = seq
                    await ws.send_str(json.dumps(snap))
                await asyncio.sleep(1 / 120)

        task = asyncio.create_task(sender())
        try:
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    try:
                        session.handle(json.loads(msg.data))
                    except (ValueError, KeyError, TypeError):
                        pass
        finally:
            task.cancel()
        return ws

    app = web.Application()
    app.router.add_get("/", index)
    app.router.add_get("/terrain/{name}", terrain)
    app.router.add_get("/ws", ws_handler)
    app.router.add_static("/static/", VIEWER_DIR)
    return app


def main(argv=None):
    ap = argparse.ArgumentParser(description="Real-time hyperstereo viewer server.")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1", help="use 0.0.0.0 to view from another device")
    ap.add_argument("--sitl", action="store_true")
    ap.add_argument("--heading", type=float, default=0.0)
    ap.add_argument("--baseline", type=float, default=50.0)
    ap.add_argument("--hfov", type=float, default=60.0, help="camera horizontal FOV, deg")
    ap.add_argument("--display-hfov", type=float, default=None, help="headset horizontal FOV, deg")
    ap.add_argument("--wind", type=float, default=None)
    ap.add_argument("--rtk", action="store_true")
    a = ap.parse_args(argv)

    from aiohttp import web
    cfg = Config()
    cfg.camera.hfov_deg = a.hfov
    if a.display_hfov:
        cfg.comfort.display_hfov_deg = a.display_hfov
    if a.wind is not None:
        cfg.sim.wind_mean = (0.0, a.wind, 0.0)
    if a.rtk:
        cfg.sim.gps_sigma_m = 0.02
    print("generating terrain...")
    T = Terrain.generate()
    session = LiveSession(cfg, a.sitl, a.heading, a.baseline, T)
    th = threading.Thread(target=session.run, daemon=True)
    th.start()
    print(f"viewer: http://{'localhost' if a.host in ('127.0.0.1', '0.0.0.0') else a.host}:{a.port}")
    try:
        web.run_app(make_app(session, terrain_assets(T)), host=a.host, port=a.port, print=None)
    finally:
        session.stop.set()


if __name__ == "__main__":
    main()
