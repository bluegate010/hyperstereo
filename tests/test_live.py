import threading
import time

import numpy as np
import pytest

from hyperstereo.config import Config
from hyperstereo.live import LiveSession, terrain_assets
from hyperstereo.terrain import Terrain


@pytest.fixture(scope="module")
def terrain():
    return Terrain.generate()


def test_terrain_assets(terrain):
    a = terrain_assets(terrain)
    n = terrain.heights.shape[0]
    assert len(a["height.bin"]) == n * n * 4
    assert a["albedo.png"][:4] == b"\x89PNG" and a["detail.png"][:4] == b"\x89PNG"


def test_live_session_launches_and_streams_snapshots(terrain):
    s = LiveSession(Config(), sitl=False, heading=0.0, baseline=50.0, terrain=terrain)
    th = threading.Thread(target=s.run, daemon=True)
    th.start()
    try:
        time.sleep(0.3)
        s.handle({"type": "cmd", "cmd": {"start": True}})
        s.handle({"type": "axes", "fwd": 5.0, "right": 0.0, "up": 0.0, "pitch": 0.0})  # clamped to 1
        s.handle({"type": "toggle", "flag": "rectify"})
        s.handle({"type": "cmd", "cmd": {"bogus": True}})  # unknown keys are dropped
        time.sleep(1.0)
        with s.lock:
            snap = s.snapshot
        assert snap["mode"] == "ASSEMBLING"
        assert s.axes["fwd"] == 1.0
        assert snap["flags"]["rectify"] is False
        assert len(snap["cams"]) == 2 and len(snap["cams"][0]["R"]) == 9
        # Column-major camera axes must be orthonormal.
        R = np.array(snap["cams"][0]["R"]).reshape(3, 3).T
        assert np.allclose(R.T @ R, np.eye(3), atol=1e-9)
        assert snap["e_lim_deg"] < 0
        s.handle({"type": "reset"})
        time.sleep(0.3)
        with s.lock:
            snap = s.snapshot
        assert snap["mode"] == "IDLE" and snap["flags"]["rectify"] is True
    finally:
        s.stop.set()
        th.join(timeout=2)
