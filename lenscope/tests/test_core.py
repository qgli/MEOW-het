"""Core engine tests: ray casting and the pose-graph sampler (no Blender needed)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lenscope.core import cast, mesh, sampler  # noqa: E402
from lenscope.core.spec import (FLAG_GLASS, SEM, Furniture, Room,  # noqa: E402
                                 SceneSpec, synthetic_apartment)


# ray-cast ground truth
def box_room_spec():
    return SceneSpec(name="box", seed=0, rooms=[Room(0, 0, 4, 3, h=2.5)],
                     portals=[], furniture=[])


def test_raycast_analytic_box():
    """Closed-form check with an asymmetric camera position so that vertical or
    horizontal chirality flips cannot pass (a symmetric position hides them).
    Room [0,4]x[0,3]x[0,2.5]; camera at (2.0, 1.0, 1.0), yaw=0 (+x forward)."""
    spec = box_room_spec()
    soup = mesh.build_from_spec(spec)
    pos = np.array([2.0, 1.0, 1.0])
    g = cast.cast_pose(soup, pos, yaw=0.0, pitch=0.0, W=256, H=128)
    H, W = 128, 256
    # centre pixel: +x wall at 2.0
    assert abs(g["depth"][H // 2, W // 2] - 2.0) < 0.02
    # vertical: top row ≈ straight up -> ceiling at 2.5-1.0=1.5/sin(el);
    # bottom row -> floor at 1.0/sin(el). These differ, so a v-flip fails here.
    el_top = (0.5 - (0.5) / H) * np.pi
    assert abs(g["depth"][0, W // 2] - 1.5 / np.sin(el_top)) < 0.1
    assert abs(g["depth"][H - 1, W // 2] - 1.0 / np.sin(el_top)) < 0.1
    # horizontal chirality: image-right of centre (az=+90) = cam +x = world -y
    # (right = fwd x world_up) -> wall y=0 at 1.0; image-left -> wall y=3 at 2.0.
    col_r, col_l = W // 2 + W // 4, W // 2 - W // 4
    assert abs(g["depth"][H // 2, col_r] - 1.0) < 0.05
    assert abs(g["depth"][H // 2, col_l] - 2.0) < 0.05
    # full coverage in a closed box
    assert (g["depth"] > 0).all()
    assert (g["sem"] > 0).all()


def test_glass_penetration_and_flags():
    """A glass pane in front of a wall: depth must reach the wall, flags carry glass."""
    spec = box_room_spec()
    spec.furniture.append(Furniture(room=0, sem_id=SEM["window"], cx=3.0, cy=1.5,
                                    z0=0.5, sx=0.05, sy=1.0, sz=1.5, flags=FLAG_GLASS))
    soup = mesh.build_from_spec(spec)
    pos = np.array([2.0, 1.5, 1.25])
    g = cast.cast_pose(soup, pos, yaw=0.0, pitch=0.0, W=128, H=64)
    c = g["depth"][32, 64]      # centre: through the pane to wall x=4
    assert abs(c - 2.0) < 0.03, c
    assert g["flags"][32, 64] & FLAG_GLASS


def test_normals_face_camera():
    spec = box_room_spec()
    soup = mesh.build_from_spec(spec)
    g = cast.cast_pose(soup, [2.0, 1.5, 1.25], 0.0, 0.0, W=64, H=32)
    n = g["normal"][16, 32]
    assert np.dot(n, [1, 0, 0]) < -0.9   # wall x=4 normal points back at camera (-x)


# pose-graph sampler
@pytest.fixture(scope="module")
def apt():
    spec = synthetic_apartment(n_rooms=3, seed=7)
    soup = mesh.build_from_spec(spec)
    res = sampler.sample_scene(spec, soup, covis_rays=48, walk_ks=(4, 8), traj=True)
    return spec, soup, res


def test_sampler_guarantees(apt):
    spec, soup, res = apt
    rep = res["report"]
    assert rep["components"] == 1
    assert rep["min_degree"] >= 2
    assert all(rep["k_walk_ok"].values())
    assert rep["n_poses"] >= 24
    assert rep["bridge_poses"] >= 2 * 2          # 2 doors x 2 sides
    rooms = {p["room"] for p in res["poses"]}
    assert rooms == {0, 1, 2}                     # all rooms populated
    assert 0.05 <= rep["supervision_share"] <= 0.30
    assert len(res["trajectory"]) >= 60


def test_sampler_heights(apt):
    _, _, res = apt
    hist = np.array(res["report"]["height_hist"], float)
    frac = hist / hist.sum()
    assert frac[0] > 0.40                         # eye-level dominant


def test_covis_sanity(apt):
    spec, soup, _ = apt
    r0 = spec.rooms[0]
    a = np.array([r0.x0 + 1.0, r0.y0 + 1.0, 1.5])
    b = np.array([r0.x1 - 1.0, r0.y1 - 1.0, 1.5])
    far_room = spec.rooms[2]
    c = np.array([(far_room.x0 + far_room.x1) / 2, 1.5, 1.5])
    same = sampler.covis_pair(soup, a, b, n_rays=96)
    cross = sampler.covis_pair(soup, a, c, n_rays=96)
    assert same > cross                            # same room sees more than 2-doors-away
    assert same > 0.25

