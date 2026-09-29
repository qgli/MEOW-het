"""Convention contract tests for make_blk_bench (guard against mirrored views and poses).

Class of bug these guard against: every representation is internally
self-consistent and a mirror world is geometrically legal, so nothing crashes
and within-type metrics stay plausible -- only cross-representation seams
expose the corruption. Each test pins one seam directly against the training
code (gpu_aug / camera_models), which defines the conventions.
Runs on a synthetic pack (no BLK data needed): pytest scripts/blk_bench/test_conventions.py
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

THIS = Path(__file__).resolve()
sys.path.insert(0, str(THIS.parent))
sys.path.insert(0, str(THIS.parent.parent.parent))

import make_blk_bench as M                                    # noqa: E402
from mapanything.datasets import camera_models as CM          # noqa: E402
from mapanything.datasets import gpu_aug as GA                # noqa: E402

torch.set_default_dtype(torch.float64)
H, W = 256, 512


def synth_pack():
    """pack in the stored convention: asymmetric smooth rgb (mirror-detectable),
    box-room depth, improper station c2w -- as emitted by blk2pack and the
    first-generation scene packs."""
    r, c = np.meshgrid(np.arange(H) + 0.5, np.arange(W) + 0.5, indexing="ij")
    lat = np.pi / 2 - r / H * np.pi
    lon = c / W * 2 * np.pi
    dirs = M.erp_dir(lat, lon)
    # left/right-asymmetric pattern: 3 gaussian blobs at chirality-breaking
    # longitudes + vertical gradient (a "text"-like canary)
    rgb = np.zeros((H, W, 3), np.float32)
    for k, (lo, la) in enumerate([(0.9, 0.2), (2.0, -0.4), (4.9, 0.5)]):
        rgb[..., k] = np.exp(-((lon - lo) ** 2 / 0.08 + (lat - la) ** 2 / 0.05))
    rgb[..., 1] += 0.3 * (lat + np.pi / 2) / np.pi
    # box room 4x6x3 m centred on the station
    eps = 1e-9
    tx = np.minimum(np.where(dirs[..., 0] > 0, 2.0 / (dirs[..., 0] + eps),
                             -2.0 / (dirs[..., 0] - eps)), 1e9)
    tz = np.minimum(np.where(dirs[..., 2] > 0, 3.0 / (dirs[..., 2] + eps),
                             -3.0 / (dirs[..., 2] - eps)), 1e9)
    ty = np.minimum(np.where(dirs[..., 1] > 0, 1.5 / (dirs[..., 1] + eps),
                             -1.5 / (dirs[..., 1] - eps)), 1e9)
    depth = np.minimum(np.minimum(tx, tz), ty).astype(np.float32)
    yaw30 = M.yaw_R(30.0)
    c2w = np.eye(4)
    c2w[:3, :3] = yaw30 @ np.diag([1.0, -1.0, 1.0])   # improper, det=-1 (pack convention)
    c2w[:3, 3] = [1.0, 2.0, 0.5]
    return {"rgb": rgb, "depth": depth, "valid": np.ones((H, W), bool),
            "rays": dirs.astype(np.float32), "c2w": c2w,
            "tag": "synth", "station": "synth"}


@pytest.fixture(scope="module")
def pk():
    M.BAND_LAT_MIN = -90.0
    return synth_pack()


def corr(a, b):
    return float(np.corrcoef(np.asarray(a, np.float64).ravel(),
                             np.asarray(b, np.float64).ravel())[0, 1])


def train_render(pk, tgt_model, tgt_params, R_a, h, w):
    """the training renderer itself = convention ground truth."""
    r2, d2, ray2, m2, c2w2 = GA.resample_from_camera(
        "spherical", {"hfov": 2 * np.pi, "vfov": np.pi},
        torch.tensor(pk["rgb"], dtype=torch.float64),
        torch.tensor(pk["depth"], dtype=torch.float64),
        torch.tensor(pk["valid"]), tgt_model, tgt_params, h, w,
        R=torch.tensor(R_a), c2w_src=torch.eye(4))
    return r2.numpy(), d2.numpy(), c2w2.numpy()


def cv_basis(yaw_deg, pitch_deg):
    """training-side proper cv basis in the analytic rep (independent
    reimplementation used as the arbiter for view_basis)."""
    z = M.FLIP3 @ M.erp_dir(np.radians(pitch_deg), np.radians(yaw_deg))
    dwn = np.array([0.0, 1.0, 0.0])
    y = dwn - (dwn @ z) * z
    y /= np.linalg.norm(y)
    return np.stack([np.cross(y, z), y, z], 1)


# pack contract: pack ray field == FLIP3 @ training analytic field.
# Known benign residual: exactly 0.5 px (pack rays use pixel-center +0.5,
# CM._grid uses pixel-corner arange); training itself carries this same
# half-pixel when projecting center-convention packs through
# corner-convention CM, so the bound is 0.6 px, not zero.
def test_pack_field_is_flipped_analytic(pk):
    A, _ = CM.unproject("spherical", H, W, {"hfov": 2 * np.pi, "vfov": np.pi}, "cpu")
    A = A.reshape(H, W, 3).numpy()
    A /= np.linalg.norm(A, axis=-1, keepdims=True)
    G = M.grid_erp_view(W, H)
    ang = np.degrees(np.arccos(np.clip(((A * [1, -1, 1]) * G).sum(-1), -1, 1)))
    px = ang.max() / (360.0 / W)
    assert px < 0.6, f"pack/analytic FLIP contract broken: {px:.2f} px off"


# erp view: native layout preserved and not mirrored (chirality canary)
def test_erp_native_and_unmirrored(pk):
    v = M.make_view([pk], 0, "erp", "t", M.yaw_R(0.0) @ M.FLIP3, M.grid_erp_cv(W, H))
    assert corr(v.rgb, pk["rgb"]) > 0.99, "erp view no longer a native pack copy"
    assert corr(v.rgb, pk["rgb"][:, ::-1]) < 0.9, "mirror canary failed (pattern?)"


# pinhole view == training renderer output (direct corr high, mirrored low)
def test_pin_matches_training_renderer(pk):
    yaw, pitch, hfov, S = 70.0, -10.0, 90.0, 256
    v = M.make_view([pk], 0, "pinhole", "t", M.view_basis(yaw, pitch),
                    M.grid_pinhole(S, S, hfov))
    f = (S / 2) / np.tan(np.radians(hfov) / 2)
    rt, dt, _ = train_render(pk, "opencv",
                             {"fx": f, "fy": f, "cx": S / 2, "cy": S / 2},
                             cv_basis(yaw, pitch), S, S)
    assert corr(v.rgb, rt) > 0.98, "pin view drifted from training renderer"
    assert corr(v.rgb, rt[:, ::-1]) < 0.5, "pin view mirrored vs the training renderer"
    ok = (v.depth > 0) & (dt > 0)
    assert np.median(np.abs(v.depth[ok] - dt[ok]) / dt[ok]) < 0.02


# fisheye view == training renderer (fisheye624 with zero coeffs
# reduces to the ideal equidistant model make_blk_bench uses)
def test_fisheye_matches_training_renderer(pk):
    yaw, fov, S = 120.0, 180.0, 256
    grid, inside = M.grid_fisheye(S, fov)
    v = M.make_view([pk], 0, "fisheye", "t", M.view_basis(yaw, 10.0), grid,
                    extra_valid=inside)
    f = (S / 2) / np.radians(fov / 2)
    rt, dt, _ = train_render(pk, "fisheye624",
                             {"fx": f, "fy": f, "cx": S / 2 - 0.5, "cy": S / 2 - 0.5,
                              "k1": 0.0, "k2": 0.0, "k3": 0.0, "k4": 0.0},
                             cv_basis(yaw, 10.0), S, S)
    m = inside & (rt.sum(-1) > 0)
    c_dir = corr(v.rgb[m], rt[m])
    c_flip = corr(v.rgb[m], rt[:, ::-1][m])
    assert c_dir > 0.98, f"fisheye drifted from training renderer ({c_dir:.3f})"
    # margin-based anti-mirror: the synthetic pattern has a flip-invariant
    # component, so the flip corr floor is ~0.6 -- demand a decisive gap
    assert c_dir - c_flip > 0.25, f"fisheye mirrored? dir {c_dir:.3f} flip {c_flip:.3f}"


# every emitted GT c2w is a proper cv pose (det +1)
def test_c2w_proper(pk):
    views = [
        M.make_view([pk], 0, "erp", "e", M.yaw_R(50.0) @ M.FLIP3, M.grid_erp_cv(W, H)),
        M.make_view([pk], 0, "pinhole", "p", M.view_basis(20.0, 5.0),
                    M.grid_pinhole(128, 128, 90.0)),
        M.make_view([pk], 0, "fisheye", "f", M.view_basis(200.0, 10.0),
                    M.grid_fisheye(128, 180.0)[0]),
    ]
    for v in views:
        d = np.linalg.det(v.c2w[:3, :3])
        assert abs(d - 1.0) < 1e-9, f"{v.kind}: improper GT c2w (det={d:.3f})"


# cross-type pose closure: the erp<->pinhole relative pose from make_view
# must equal the training-side composition (a mismatch corrupts blk_mixed)
def test_cross_type_relative_pose(pk):
    v_erp = M.make_view([pk], 0, "erp", "e", M.yaw_R(80.0) @ M.FLIP3,
                        M.grid_erp_cv(W, H))
    v_pin = M.make_view([pk], 0, "pinhole", "p", M.view_basis(35.0, -20.0),
                        M.grid_pinhole(128, 128, 90.0))
    rel_mine = np.linalg.inv(v_erp.c2w) @ v_pin.c2w
    # training side: c2w_cv = (R_station @ FLIP) @ R_a for both views
    c2w_src_cv = pk["c2w"][:3, :3] @ np.diag([1.0, -1.0, 1.0])
    # erp content-rotation by yaw80 in pack rep == FLIP@yaw80@FLIP in analytic rep
    R_erp_a = np.diag([1.0, -1.0, 1.0]) @ M.yaw_R(80.0) @ np.diag([1.0, -1.0, 1.0])
    R_train_erp = c2w_src_cv @ R_erp_a
    R_train_pin = c2w_src_cv @ cv_basis(35.0, -20.0)
    rel_train = R_train_erp.T @ R_train_pin
    dR = rel_mine[:3, :3].T @ rel_train
    ang = np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1)))
    assert ang < 1e-6, f"cross-type rel pose off by {ang:.4f} deg (mixed poison)"


# GT xyz stays in the true world frame under the cv re-representation
def test_xyz_world_invariant(pk):
    v = M.make_view([pk], 0, "erp", "e", M.yaw_R(0.0) @ M.FLIP3, M.grid_erp_cv(W, H))
    xyz = M.xyz_world(v)
    ref = (pk["c2w"][:3, :3] @ (M.grid_erp_view(W, H) * pk["depth"][..., None]
                                ).reshape(-1, 3).T).T + pk["c2w"][:3, 3]
    assert np.nanmax(np.abs(xyz - ref.reshape(H, W, 3))) < 1e-4
