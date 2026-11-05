import numpy as np
import pytest

from hyperstereo.camera import cam_rotation, intrinsics, pixel_rays, project
from hyperstereo.config import ComfortConfig
from hyperstereo.geometry import right_vec
from hyperstereo.stereo import (FeatureRectifier, auto_convergence, comfort_metrics, ground_mask,
                                make_dem_depth_fn)
from hyperstereo.terrain import Terrain

W, H = 480, 270
K = intrinsics(W, H, 60.0)


@pytest.fixture(scope="module")
def terrain():
    return Terrain.generate()


def test_camera_axes_and_projection_roundtrip():
    R = cam_rotation(90.0, -10.0, 0.0)
    assert np.allclose(R[:, 0], [-1, 0, 0], atol=1e-9)        # looking east: right is south
    assert R[2, 2] < 0                                          # pitched down
    d = pixel_rays(K, R, np.array([100.0]), np.array([50.0]))
    u, v, _ = project(np.array([5.0, 3.0, 2.0]) + d * 1000, np.array([5.0, 3.0, 2.0]), R, K)
    assert u[0] == pytest.approx(100.0) and v[0] == pytest.approx(50.0)


def test_comfort_limit_from_headset_model():
    c = ComfortConfig(focal_distance_m=2.0, comfort_diopters=0.5, ipd_m=0.063)
    assert c.max_crossed_disparity_deg == pytest.approx(np.degrees(0.063), rel=1e-6)  # near limit = 1 m


def test_ground_mask_covers_near_ground_only():
    m = ground_mask(K, W, H, 0.0, -3.0, 0.0, 100.0, 50.0, ComfortConfig())
    assert m[0].max() == 0.0 and m[-1].min() == 1.0


@pytest.mark.parametrize("B,dyaw,dpitch,droll", [(50, 1.5, 0.4, -0.5), (100, -2.5, -0.3, 0.4)])
def test_pipeline_removes_vertical_disparity_and_prevents_divergence(terrain, B, dyaw, dpitch, droll):
    c = ComfortConfig()
    yaw = 30.0
    ctr = np.array([0.0, 0.0, 100.0])
    pl, pr = ctr - right_vec(yaw) * B / 2, ctr + right_vec(yaw) * B / 2
    Rl, Rr = cam_rotation(yaw, -3, 0.1), cam_rotation(yaw + dyaw, -3 + dpitch, droll)
    il, rl = terrain.render(pl, Rl, K, W, H)
    ir, rr = terrain.render(pr, Rr, K, W, H)
    res = FeatureRectifier(K).estimate(il, ir)
    assert res.ok
    conv = auto_convergence(res, K, cam_rotation(yaw, -3, 0), pl + [0.8, -0.6, 0.2], B + 1.0,
                            make_dem_depth_fn(terrain))
    assert conv.method == "dem"
    m = ground_mask(K, W, H, yaw, -3, 0, 100, B, c)
    cm = comfort_metrics(K, W, H, ((pl, Rl), (pr, Rr)), (rl, rr), conv.H, conv.shift_px, m, c)
    assert cm.raw_vert_p95_deg > 0.3
    assert cm.vert_p95_deg < 0.1
    assert cm.max_divergence_deg < 0.05
    assert cm.violation_frac < 0.05


def test_rectifier_reports_no_lock_on_featureless_view(terrain):
    flat = np.full((H, W, 3), 0.5, np.float32)
    assert not FeatureRectifier(K).estimate(flat, flat).ok
