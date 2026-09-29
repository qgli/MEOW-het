"""make_blk_bench.py — zero-shot benchmark generator for the BLK360 laser-scanned scenes.

Source: Leica BLK360 laser-scan station packs (lenscope/blk/blk2pack.py output:
per-station 1536x3072 equirectangular panorama with rgb / along-ray metric
depth / valid mask / rays, registered c2w, station covisibility matrix, hole
analysis). Real captures with laser GT let every method (MEOW and all
baselines) be scored on pose and pointmap against survey-grade ground truth,
zero-shot for all.

Four tracks, all emitted in the realset harness contract
(tuples.json + frames/*.png + gt/<tid>.npz), so scripts/realset/predict_*.py
and eval_realset.py run unchanged:
  blk_pinhole  distortion-free perspective crops, frustum fully inside the
               valid elevation band (the scanner nadir hole never enters);
  blk_fisheye  equidistant 170deg, pitch +10; the nadir hole may appear at the
               disk edge as black (real scanner shadow); GT masks it out;
  blk_erp      full panoramas (hole visible at bottom, GT-masked);
  blk_mixed    {1 erp + 1 fisheye + 2 pinhole} heterogeneous tuples.
--mixed-views adds mixed tracks with other tuple sizes.

Covisibility: every tuple is a chain whose consecutive views pass a per-view
covisibility test computed from the laser GT by mutual projection with depth
agreement max(10 cm, 3 %), estimated from 1,500 sampled points per direction
(covis_pair; blk2pack's station matrix uses the same mutual projection with
10 cm). The station-level matrix pre-filters candidates; the per-view test decides.

Convention: pack rays are y-up, lon(col): 0 -> -z, pi/2 -> -x (checked by a
numeric assert against the stored rays at startup). Virtual view bases are
built OpenCV-style (x right, y down, z forward) inside the pack frame; each
view's c2w = station_c2w @ R_view. GT pointmaps and poses share the same basis
per view, which is all the relative-pose + Sim3 pointmap metrics require.

Usage:
  python scripts/blk_bench/make_blk_bench.py \
    --scene-dir <...>/scene_corridor_meeting --out <...>/blk_bench --seed 0
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

THIS = Path(__file__).resolve()
sys.path.insert(0, str(THIS.parent.parent / "realset"))
from common import save_gt_npz, save_tuples, subsample  # noqa: E402

# labels written into tuples.json ("dataset") and into every tuple ("seq"), as in the released benchmark
DATASET_LABEL = "blk_office"
SEQUENCE_LABEL = "blk_A_office_01"

# ---------------------------------------------------------------- convention
def erp_dir(lat, lon):
    """pack-frame direction for latitude/longitude (y-up; lon 0 -> -z)."""
    cl = np.cos(lat)
    return np.stack([-np.sin(lon) * cl, np.sin(lat), -np.cos(lon) * cl], -1)


def dir_to_rowcol(d, H, W):
    d = d / np.clip(np.linalg.norm(d, axis=-1, keepdims=True), 1e-9, None)
    lat = np.arcsin(np.clip(d[..., 1], -1, 1))
    lon = np.arctan2(-d[..., 0], -d[..., 2]) % (2 * np.pi)
    row = (np.pi / 2 - lat) / np.pi * H - 0.5
    col = lon / (2 * np.pi) * W - 0.5
    return row, col


def assert_convention(pack):
    rays, H, W = pack["rays"], pack["rays"].shape[0], pack["rays"].shape[1]
    rr = np.array([H // 4, H // 2, 3 * H // 4])
    cc = np.array([0, W // 4, W // 2, 3 * W // 4])
    R, C = np.meshgrid(rr, cc, indexing="ij")
    lat = np.pi / 2 - (R + 0.5) / H * np.pi
    lon = (C + 0.5) / W * 2 * np.pi
    pred = erp_dir(lat, lon)
    got = rays[R, C].astype(np.float64)
    got /= np.linalg.norm(got, axis=-1, keepdims=True)
    err = np.degrees(np.arccos(np.clip((pred * got).sum(-1), -1, 1)))
    assert err.max() < 0.2, f"ERP convention mismatch: max {err.max():.3f} deg"
    return float(err.max())


# ---------------------------------------------------------------- pack io
def load_pack(scene_dir: Path, frame):
    d = np.load(scene_dir / frame["pack_file"])
    rgb = d["rgb"].astype(np.float32)
    depth = d["depth"].astype(np.float32)
    valid = (d["mask"] > 0) & (depth > 0.1)
    return {"rgb": rgb, "depth": depth, "valid": valid,
            "rays": d["rays"].astype(np.float32),
            "c2w": np.asarray(frame["camera_to_world_unicol_4x4"], np.float64),
            "tag": frame["tag"], "station": frame.get("station", frame["tag"])}


def sample_pack(pack, dirs):
    """dirs: (h,w,3) unit dirs in pack frame -> rgb (bilinear), depth/valid
    (nearest). Invalid pixels: rgb black, depth 0."""
    H, W = pack["depth"].shape
    row, col = dir_to_rowcol(dirs, H, W)
    r0 = np.clip(np.floor(row).astype(int), 0, H - 1)
    r1 = np.clip(r0 + 1, 0, H - 1)
    c0 = np.floor(col).astype(int) % W
    c1 = (c0 + 1) % W
    fr = np.clip(row - r0, 0, 1)[..., None]
    fc = np.clip(col - c0, 0, 1)[..., None]
    rgb = (pack["rgb"][r0, c0] * (1 - fr) * (1 - fc) +
           pack["rgb"][r1, c0] * fr * (1 - fc) +
           pack["rgb"][r0, c1] * (1 - fr) * fc +
           pack["rgb"][r1, c1] * fr * fc)
    rn = np.clip(np.round(row).astype(int), 0, H - 1)
    cn = np.round(col).astype(int) % W
    depth = pack["depth"][rn, cn]
    valid = pack["valid"][rn, cn]
    # rgb is not blacked out at laser-invalid pixels: the pano camera often
    # sees what the laser can't (glass, screens); inputs keep the real
    # appearance, and GT carries no supervision there (valid=False)
    return rgb, np.where(valid, depth, 0.0), valid


# ---------------------------------------------------------------- ray grids
def grid_pinhole(W, H, hfov_deg):
    fx = (W / 2) / np.tan(np.radians(hfov_deg) / 2)
    u, v = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    d = np.stack([(u - W / 2) / fx, (v - H / 2) / fx, np.ones_like(u)], -1)
    return d / np.linalg.norm(d, axis=-1, keepdims=True)


def grid_fisheye(S, fov_deg):
    f = (S / 2) / np.radians(fov_deg / 2)
    u, v = np.meshgrid(np.arange(S) + 0.5, np.arange(S) + 0.5)
    du, dv = u - S / 2, v - S / 2
    r = np.hypot(du, dv)
    theta = r / f
    inside = theta <= np.radians(fov_deg / 2)
    theta = np.minimum(theta, np.radians(fov_deg / 2))
    az = np.arctan2(dv, du)
    st = np.sin(theta)
    d = np.stack([st * np.cos(az), st * np.sin(az), np.cos(theta)], -1)
    return d, inside


FLIP3 = np.diag([1.0, -1.0, 1.0])
"""Pack-representation -> OpenCV flip (the `_FLIP` of the camera sampler,
mapanything/datasets/camera_sampler.py). The pack stores a y-up, left-handed
representation: pack rays == FLIP3 @ analytic cv field, and the station c2w
rotations have det=-1 (pack convention), which is why training uses
`c2w_cv = R_station @ FLIP`. Every view below therefore (a) expresses dirs_cam
in the proper cv convention the model was trained with, and (b) passes the
rotation to make_view composed with FLIP3, so pack-frame sampling stays
correct and the emitted c2w = station_c2w @ (FLIP3-composed R) is a det=+1 cv
pose. Grids built directly in the pack representation instead mirror the
images relative to the training renderer (gpu_aug.resample_from_camera) and
corrupt every cross-type relative pose in blk_mixed; test_conventions.py
checks these seams."""


def grid_erp_view(W, H):
    """pack-representation ERP grid (rows of the pack's own erp_dir field)."""
    r, c = np.meshgrid(np.arange(H) + 0.5, np.arange(W) + 0.5, indexing="ij")
    lat = np.pi / 2 - r / H * np.pi
    lon = c / W * 2 * np.pi
    return erp_dir(lat, lon)


def grid_erp_cv(W, H):
    """training/cv-convention analytic ERP field (z = image-centre forward,
    x = right, y = down) == FLIP3 @ pack grid; it matches
    mapanything.datasets.camera_models.unproject('spherical') up to the
    half-pixel centre/corner offset checked in test_conventions.py. With
    R = yaw_R(yaw) @ FLIP3 the sampled image is the native pack layout
    (bit-identical to sampling grid_erp_view with yaw_R alone)."""
    return grid_erp_view(W, H) * np.array([1.0, -1.0, 1.0])


def yaw_R(yaw_deg):
    """proper rotation about the pack up-axis (+y); shifts pano content by
    +yaw in longitude (consistent with erp_dir)."""
    a = np.radians(yaw_deg)
    ca, sa = np.cos(a), np.sin(a)
    return np.array([[ca, 0.0, sa], [0.0, 1.0, 0.0], [-sa, 0.0, ca]])


def view_basis(yaw_deg, pitch_deg):
    """training-cv camera basis for a pin/fish view, returned in the pack
    representation for make_view's sampling. Constructed proper in the
    analytic (flipped) representation (z = gaze, y = world-down projected
    out, x = y x z; upright, roll 0), then returned as FLIP3 @ R_a (det -1 by
    design: the improper station c2w composes with it to a proper cv c2w, the
    same `R_station @ FLIP` composition training uses). The rendered images
    match the training renderer (gpu_aug.resample_from_camera); see
    test_conventions.py."""
    z_a = FLIP3 @ erp_dir(np.radians(pitch_deg), np.radians(yaw_deg))
    dwn = np.array([0.0, 1.0, 0.0])          # analytic-rep world-down
    y = dwn - (dwn @ z_a) * z_a
    y /= np.linalg.norm(y)
    x = np.cross(y, z_a)
    return FLIP3 @ np.stack([x, y, z_a], axis=1)   # (3,3) columns, det=-1


# ---------------------------------------------------------------- view synth
BAND_LAT_MIN = None    # set in main from hole_analysis band_rows


class View:
    __slots__ = ("name", "kind", "station_i", "R", "rgb", "depth", "valid",
                 "dirs_cam", "c2w", "w", "h")


def make_view(packs, si, kind, name, R, dirs_cam, extra_valid=None):
    p = packs[si]
    dirs_pack = dirs_cam @ R.T
    rgb, depth, valid = sample_pack(p, dirs_pack)
    if extra_valid is not None:      # lens disk (fisheye): a lens property,
        valid = valid & extra_valid  # black rgb outside the image circle
        rgb[~extra_valid] = 0.0
        depth = np.where(valid, depth, 0.0)
    v = View()
    v.name, v.kind, v.station_i, v.R = name, kind, si, R
    v.rgb, v.depth, v.valid, v.dirs_cam = rgb, depth, valid, dirs_cam
    c2w = p["c2w"].copy()
    c2w[:3, :3] = c2w[:3, :3] @ R
    v.c2w = c2w
    v.h, v.w = depth.shape
    return v


def xyz_world(v: View):
    pts = v.dirs_cam * v.depth[..., None]
    out = (v.c2w[:3, :3] @ pts.reshape(-1, 3).T).T + v.c2w[:3, 3]
    out = out.reshape(pts.shape)
    out[~v.valid] = np.nan
    return out.astype(np.float32)


def frustum_in_band(dirs_cam, R, margin_deg=2.0):
    """True if every ray stays above the nadir-hole band."""
    dp = dirs_cam @ R.T
    lat = np.degrees(np.arcsin(np.clip(dp[..., 1], -1, 1)))
    return lat.min() > BAND_LAT_MIN + margin_deg


# ---------------------------------------------------------------- per-view covis
def covis_pair(vi: View, vj: View, n=1500, rng=None):
    """fraction of vi's valid points seen by vj with 10cm/3% depth agreement,
    symmetrized by min()."""
    def one_way(a, b):
        pa = xyz_world(a)
        flat = pa.reshape(-1, 3)
        ok = np.isfinite(flat).all(1)
        idx = np.flatnonzero(ok)
        if len(idx) == 0:
            return 0.0
        sel = rng.choice(idx, size=min(n, len(idx)), replace=False)
        pw = flat[sel].astype(np.float64)
        w2c = np.linalg.inv(b.c2w)
        pc = (w2c[:3, :3] @ pw.T).T + w2c[:3, 3]
        r = np.linalg.norm(pc, axis=1)
        d = pc / np.clip(r[:, None], 1e-9, None)
        if b.kind == "pinhole":
            infov = d[:, 2] > 0.05
            u = d[:, 0] / np.clip(d[:, 2], 1e-9, None)
            vv = d[:, 1] / np.clip(d[:, 2], 1e-9, None)
            fx = (b.w / 2) / np.tan(np.radians(PIN_HFOV) / 2)
            px = u * fx + b.w / 2
            py = vv * fx + b.h / 2
            infov &= (px >= 0) & (px < b.w) & (py >= 0) & (py < b.h)
        elif b.kind == "fisheye":
            theta = np.arccos(np.clip(d[:, 2], -1, 1))
            infov = theta < np.radians(FISH_FOV / 2)
            f = (b.w / 2) / np.radians(FISH_FOV / 2)
            az = np.arctan2(d[:, 1], d[:, 0])
            px = np.cos(az) * theta * f + b.w / 2
            py = np.sin(az) * theta * f + b.h / 2
            px = np.clip(px, 0, b.w - 1)
            py = np.clip(py, 0, b.h - 1)
        else:                                   # erp: cam dirs are cv ->
            d_pack = d * np.array([1.0, -1.0, 1.0])   # pack rep for rowcol
            py, px = dir_to_rowcol(d_pack, b.h, b.w)
            infov = np.ones(len(d), bool)
            px = np.clip(px, 0, b.w - 1)
            py = np.clip(py, 0, b.h - 1)
        if b.kind == "pinhole":
            px = np.clip(px, 0, b.w - 1)
            py = np.clip(py, 0, b.h - 1)
        ii, jj = py.astype(int), px.astype(int)
        db = b.depth[ii, jj]
        vb = b.valid[ii, jj]
        agree = infov & vb & (np.abs(r - db) < np.maximum(0.10, 0.03 * db))
        return float(agree.mean())
    return min(one_way(vi, vj), one_way(vj, vi))


# ---------------------------------------------------------------- tracks
PIN_HFOV, PIN_W, PIN_H = 75.0, 640, 480
FISH_FOV, FISH_S = 170.0, 720
ERP_W, ERP_H = 1024, 512
XYZ_DS = 2
CLOUD_CAP = 200_000


def save_view_png(track_dir: Path, v: View):
    fp = track_dir / "frames" / f"{v.name}.png"
    if not fp.exists():
        fp.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray((np.clip(v.rgb, 0, 1) * 255).astype(np.uint8)).save(fp)
    return f"frames/{v.name}.png"


def emit_tuple(track_dir, tid, views):
    rels = [save_view_png(track_dir, v) for v in views]
    xyz_l, cloud = [], []
    for v in views:
        xw = xyz_world(v)
        xyz_l.append(xw[::XYZ_DS, ::XYZ_DS])
        fl = xw.reshape(-1, 3)
        cloud.append(fl[np.isfinite(fl).all(1)])
    gt_pts = subsample(np.concatenate(cloud).astype(np.float32),
                       CLOUD_CAP, seed=0)
    # heterogeneous views (mixed track) -> NaN-pad xyz_ds to a common shape;
    # corr_from_uv only ever indexes inside each view's own (h/ds, w/ds)
    # region, and any stray padded hit is NaN -> dropped by the isfinite gate
    Hd = max(x.shape[0] for x in xyz_l)
    Wd = max(x.shape[1] for x in xyz_l)
    xyz_pad = np.full((len(xyz_l), Hd, Wd, 3), np.nan, np.float32)
    for i, x in enumerate(xyz_l):
        xyz_pad[i, :x.shape[0], :x.shape[1]] = x
    save_gt_npz(track_dir / "gt" / f"{tid}.npz",
                np.stack([v.c2w for v in views]),
                xyz_pad, XYZ_DS, gt_pts)
    return {"id": tid, "seq": SEQUENCE_LABEL,
            "views": [{"img": r, "w": int(v.w), "h": int(v.h),
                       "kind": v.kind, "station": int(v.station_i)}
                      for r, v in zip(rels, views)],
            "gt": f"gt/{tid}.npz"}


def station_paths(covis, rng, k, n_paths, thres):
    """random walks of length k on the station graph (edge >= thres)."""
    n = covis.shape[0]
    out, seen, tries = [], set(), 0
    while len(out) < n_paths and tries < n_paths * 400:
        tries += 1
        path = [int(rng.integers(n))]
        while len(path) < k:
            cand = [j for j in range(n)
                    if j not in path and covis[path[-1], j] >= thres]
            if not cand:
                break
            w = np.array([covis[path[-1], j] for j in cand])
            path.append(int(rng.choice(cand, p=w / w.sum())))
        if len(path) == k and tuple(path) not in seen:
            seen.add(tuple(path))   # ordered dedupe: same set, new order = new
            out.append(path)        # chain (different views/covis structure)
    return out


def main():
    global BAND_LAT_MIN
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tuples-per-track", type=int, default=32)
    ap.add_argument("--gate", type=float, default=0.25,
                    help="per-view consecutive-pair covis gate of the pinhole and fisheye "
                         "tracks; the panorama track uses 0.10 and the mixed track "
                         "0.15 unless --allpairs-gate > 0")
    ap.add_argument("--allpairs-gate", type=float, default=0.0,
                    help="per-view covis floor over all view pairs (not just "
                         "consecutive); ~0.10 gives a benchmark closer to the "
                         "training regime; 0 = off (default)")
    ap.add_argument("--walk-thres", type=float, default=0.12,
                    help="station-graph edge threshold for path proposals; "
                         "raise (~0.30) to keep walks inside dense clusters")
    ap.add_argument("--tracks", nargs="+",
                    default=["blk_erp", "blk_pinhole", "blk_fisheye", "blk_mixed"])
    ap.add_argument("--mixed-views", nargs="*", type=int, default=[],
                    help="extra mixed tracks blk_mixed{k} with k views cycling erp,fish,pin,pin (e.g. 6 8 12); "
                         "the four standard tracks are untouched")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    scene = Path(args.scene_dir)
    out = Path(args.out)

    meta = json.load(open(scene / "metadata.json"))
    covis = np.load(scene / "covisibility" / "v0" / "covisibility.npy")
    frames_all = meta["frames"]
    # synthetic scenes carry many base types; the source of truth for
    # re-synthesis is the ERP packs only — filter and subset covis to match
    erp_idx = [i for i, f in enumerate(frames_all)
               if f.get("base_type") == "erp"]
    frames = [frames_all[i] for i in erp_idx]
    if covis.shape[0] == len(frames_all) and len(frames) < len(frames_all):
        covis = covis[np.ix_(erp_idx, erp_idx)]
    H_pack = frames[0]["resolution"][1]
    hole_p = scene / "hole_analysis.json"
    if hole_p.exists():
        holes = json.load(open(hole_p))
        BAND_LAT_MIN = 90.0 - holes["band_rows"][1] / H_pack * 180.0
    else:
        BAND_LAT_MIN = -90.0   # synthetic: no scanner shadow, full sphere valid
    print(f"[blk] {len(frames)} stations, band_lat_min={BAND_LAT_MIN:.1f} deg")

    packs = [load_pack(scene, f) for f in frames]
    conv_err = assert_convention(packs[0])
    print(f"[blk] ERP convention verified (max {conv_err:.4f} deg)")

    report = {"seed": args.seed, "convention_err_deg": conv_err,
              "band_lat_min": BAND_LAT_MIN, "tracks": {}}

    pin_grid = grid_pinhole(PIN_W, PIN_H, PIN_HFOV)
    fish_grid, fish_inside = grid_fisheye(FISH_S, FISH_FOV)
    erp_grid = grid_erp_cv(ERP_W, ERP_H)

    def pin_view(si, yaw, pitch):
        R = view_basis(yaw, pitch)
        if not frustum_in_band(pin_grid, R):
            return None
        nm = f"st{si:02d}_pin_y{int(round(yaw))%360:03d}_p{int(round(pitch)):+03d}"
        return make_view(packs, si, "pinhole", nm, R, pin_grid)

    # ---- coarse candidate bank: stations have height offsets (stairs), so
    # view directions are chosen from data: per-view covis on cheap 160x120
    # proxies, full-res synthesis only for the selected views.
    COARSE = grid_pinhole(160, 120, PIN_HFOV)
    PITCHES = (0.0, 10.0)
    YAWS = tuple(range(0, 360, 30))
    coarse_bank = {}
    for si in range(len(packs)):
        for yaw in YAWS:
            for pitch in PITCHES:
                R = view_basis(yaw, pitch)
                if not frustum_in_band(COARSE, R):
                    continue
                coarse_bank[(si, yaw, pitch)] = make_view(
                    packs, si, "pinhole", f"c{si}_{yaw}_{pitch}", R, COARSE)
    print(f"[blk] coarse bank: {len(coarse_bank)} candidate views")

    ccache = {}
    def coarse_covis(ka, kb):
        key = (ka, kb) if ka < kb else (kb, ka)
        if key not in ccache:
            ccache[key] = covis_pair(coarse_bank[key[0]], coarse_bank[key[1]],
                                     n=600, rng=rng)
        return ccache[key]

    def best_pin_chain(path, gate):
        """greedy: pick (yaw,pitch) per station so consecutive per-view covis
        (coarse proxy) >= gate; returns [(si,yaw,pitch)] or None."""
        cands = {si: [k for k in coarse_bank if k[0] == si] for si in path}
        if any(not cands[si] for si in path):
            return None
        a, b = path[0], path[1]
        best, bv = None, -1.0
        for ka in cands[a]:
            for kb in cands[b]:
                c = coarse_covis(ka, kb)
                if c > bv:
                    bv, best = c, (ka, kb)
        if bv < gate:
            return None
        chain = [best[0], best[1]]
        for nxt in path[2:]:
            kb, bv = None, -1.0
            for k in cands[nxt]:
                c = coarse_covis(chain[-1], k)
                if c > bv:
                    bv, kb = c, k
            if bv < gate:
                return None
            chain.append(kb)
        return [(k[0], k[1], k[2]) for k in chain]

    def fish_view(si, yaw):
        R = view_basis(yaw, 10.0)
        nm = f"st{si:02d}_fish_y{int(round(yaw))%360:03d}"
        return make_view(packs, si, "fisheye", nm, R, fish_grid,
                         extra_valid=fish_inside)

    def erp_view(si, yaw):
        R = yaw_R(yaw) @ FLIP3        # cv dirs composed with FLIP3: the image
        nm = f"st{si:02d}_erp_y{int(round(yaw))%360:03d}"    # keeps the native layout
        return make_view(packs, si, "erp", nm, R, erp_grid)

    def yaw_towards(si, sj):
        # target direction must be expressed in the station camera frame
        # first (the world here is z-up; pack cam is y-up with lon0 = -z)
        d = packs[sj]["c2w"][:3, 3] - packs[si]["c2w"][:3, 3]
        dc = packs[si]["c2w"][:3, :3].T @ d
        n = np.linalg.norm(dc)
        if n < 1e-6:
            return float(rng.uniform(0, 360))
        dc = dc / n
        lon = np.arctan2(-dc[0], -dc[2]) % (2 * np.pi)
        return float(np.degrees(lon))

    def build_track(track, maker, gate, k=4):
        tdir = out / track
        tuples, stats = [], []
        cache = {}
        paths = station_paths(covis, rng, k=k,
                              n_paths=args.tuples_per_track * 16,
                              thres=args.walk_thres)
        for path in paths:
            if len(tuples) >= args.tuples_per_track:
                break
            views = maker(path)
            if views is None:
                continue
            cv = []
            for a, b in zip(views[:-1], views[1:]):
                key = (a.name, b.name)
                if key not in cache:
                    cache[key] = covis_pair(a, b, rng=rng)
                cv.append(cache[key])
            if min(cv) < gate:
                continue
            if args.allpairs_gate > 0:
                ap_ok = True
                for i in range(len(views)):
                    for j in range(i + 1, len(views)):
                        key = (views[i].name, views[j].name)
                        if key not in cache:
                            cache[key] = covis_pair(views[i], views[j], rng=rng)
                        if cache[key] < args.allpairs_gate:
                            ap_ok = False
                            break
                    if not ap_ok:
                        break
                if not ap_ok:
                    continue
            tid = f"{track}_{len(tuples):03d}"
            tuples.append(emit_tuple(tdir, tid, views))
            stats.append({"id": tid, "min_covis": float(min(cv)),
                          "covis": [round(c, 3) for c in cv],
                          "stations": [int(v.station_i) for v in views]})
        save_tuples(tdir, DATASET_LABEL, track, str(tdir), tuples)
        report["tracks"][track] = {
            "n_tuples": len(tuples),
            "min_covis_med": float(np.median([s["min_covis"] for s in stats])) if stats else 0,
            "min_covis_min": float(np.min([s["min_covis"] for s in stats])) if stats else 0,
            "stats": stats}
        print(f"[blk] {track}: {len(tuples)} tuples "
              f"(gate {gate}, med min-covis "
              f"{report['tracks'][track]['min_covis_med']:.3f})")

    # ---- erp: 4 stations, native panos. Consecutive gate 0.10; with
    #      --allpairs-gate > 0 the chain uses --gate instead
    erp_gate = args.gate if args.allpairs_gate > 0 else 0.10
    if "blk_erp" in args.tracks:
        build_track("blk_erp",
                    lambda path: [erp_view(si, float(rng.uniform(0, 360)))
                                  for si in path],
                    gate=erp_gate)

    # ---- pinhole: data-driven chain (coarse search -> full-res synth)
    def mk_pin(path):
        chain = best_pin_chain(path, args.gate)
        if chain is None:
            return None
        return [pin_view(si, yaw, pitch) for si, yaw, pitch in chain]
    if "blk_pinhole" in args.tracks:
        build_track("blk_pinhole", mk_pin, gate=args.gate)

    # ---- fisheye: wide fov, aim at the neighbour (the 170deg coverage
    # tolerates height offsets); the per-view gate still decides
    def mk_fish(path):
        return [fish_view(a, yaw_towards(a, b) + float(rng.uniform(-30, 30)))
                for a, b in zip(path, path[1:] + [path[-2]])]
    if "blk_fisheye" in args.tracks:
        build_track("blk_fisheye", mk_fish, gate=args.gate)

    # ---- mixed: 1 erp + 1 fisheye + 2 pinhole (pinhole legs via coarse search);
    #      consecutive gate 0.15, or 0.20 with --allpairs-gate > 0
    def mk_mixed(path):
        v0 = erp_view(path[0], float(rng.uniform(0, 360)))
        v1 = fish_view(path[1], yaw_towards(path[1], path[0])
                       + float(rng.uniform(-30, 30)))
        chain = best_pin_chain(path[1:], 0.20)   # anchor pin legs on v1's station
        if chain is None:
            return None
        rest = [pin_view(si, yaw, pitch) for si, yaw, pitch in chain[1:]]
        if any(v is None for v in rest):
            return None
        return [v0, v1] + rest
    if "blk_mixed" in args.tracks:
        build_track("blk_mixed", mk_mixed,
                    gate=(0.20 if args.allpairs_gate > 0 else 0.15))

    # ---- mixed at other tuple sizes: kinds cycle erp, fish, pin, pin along a
    #      k-station chain; pinhole runs are anchored on the preceding station as in mk_mixed
    def mk_mixed_k(k):
        pattern = ["erp", "fish", "pin", "pin"]
        def maker(path):
            kinds = [pattern[i % 4] for i in range(k)]
            views, i = [], 0
            while i < k:
                si, kind = path[i], kinds[i]
                if kind == "erp":
                    views.append(erp_view(si, float(rng.uniform(0, 360)))); i += 1
                elif kind == "fish":
                    ref = path[i - 1] if i > 0 else path[i + 1]
                    views.append(fish_view(si, yaw_towards(si, ref) + float(rng.uniform(-30, 30)))); i += 1
                else:
                    j = i
                    while j < k and kinds[j] == "pin":
                        j += 1
                    seg = [path[i - 1]] + path[i:j]          # kinds start with erp, so i > 0 here
                    chain = best_pin_chain(seg, 0.20)
                    if chain is None:
                        return None
                    vs = [pin_view(s, yaw, pitch) for s, yaw, pitch in chain[1:]]
                    if any(v is None for v in vs):
                        return None
                    views.extend(vs); i = j
            return views
        return maker
    for kk in args.mixed_views:
        build_track(f"blk_mixed{kk}", mk_mixed_k(kk),
                    gate=(0.20 if args.allpairs_gate > 0 else 0.15), k=kk)

    with open(out / "BENCH_REPORT.json", "w") as f:
        json.dump(report, f, indent=1)
    print(f"[blk] report -> {out / 'BENCH_REPORT.json'}")


if __name__ == "__main__":
    main()
