#!/usr/bin/env python3
"""ADT (Aria Digital Twin) GT preparation: vrs -> frames + camera-model-free gt npz.

Runs in the `mapanything` conda env plus `pip install projectaria-tools`
(pure wheel). Per sequence dir (Apartment_release_*_seq*_M1292): video.vrs
(RGB fisheye, stream 214-1), depth_images.vrs (digital-twin GT depth on the
RGB grid, uint16 mm), aria_trajectory.csv / ADT GT; all are read through the
official AriaDigitalTwinDataProvider, no hand-parsed formats.

Two tracks are emitted in one pass:
  fisheye : native RGB (rotated upright), fed without calibration: tests
            heterogeneous-camera handling on a real fisheye sensor.
  pinhole : the same frames undistorted to a linear camera (default 640x640,
            f=280, ~98 deg hfov): a pinhole control track for the baselines.

Reprojection checks (numbers printed):
  1. depth semantics (z-depth vs ray range) are chosen on the first tuple by
     cross-view reprojection consistency; the script aborts unless the chosen
     convention reaches median <= 8% and inliers >= 50%.
  2. the same check on the first tuple of every sequence; failing sequences
     are skipped.
Poses and calibration all come from the provider; the upright rotation uses
the official rotate_camera_calib_cw90deg so T_device_camera stays consistent.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import save_gt_npz, save_tuples, subsample  # noqa: E402


def _interp_enum():
    """InterpolationMethod lives in projectaria_tools.core.image on current
    wheels (members BILINEAR/NEAREST_NEIGHBOR; calibration has no such
    attribute). Older wheels are tried via calibration; otherwise raise an
    ImportError listing both locations."""
    try:
        from projectaria_tools.core.image import InterpolationMethod
        return InterpolationMethod
    except ImportError:
        pass
    from projectaria_tools.core import calibration as _c
    im = getattr(_c, "InterpolationMethod", None)
    if im is None:
        raise ImportError(
            "InterpolationMethod found in neither projectaria_tools.core.image "
            "nor .core.calibration — API change; check the installed "
            "projectaria-tools version")
    return im


def get_provider(seq_dir: Path):
    from projectaria_tools.projects.adt import (
        AriaDigitalTwinDataPathsProvider, AriaDigitalTwinDataProvider)
    pp = AriaDigitalTwinDataPathsProvider(str(seq_dir))
    if hasattr(pp, "get_datapaths"):
        paths = pp.get_datapaths()
    else:  # older wheels
        paths = pp.get_datapaths_by_device_num(0)
    return AriaDigitalTwinDataProvider(paths)


def se3_mat(se3) -> np.ndarray:
    m = se3.to_matrix()
    return np.asarray(m, np.float64)


class SeqExtractor:
    def __init__(self, dp, upright: bool, pin_size: int, pin_f: float):
        from projectaria_tools.core import calibration as calib_mod
        from projectaria_tools.core.stream_id import StreamId
        self.calib_mod = calib_mod
        self.interp = _interp_enum()
        self.dp = dp
        self.sid = StreamId("214-1")
        cal = dp.get_aria_camera_calibration(self.sid)
        assert cal is not None, "no RGB calibration in vrs"
        self.upright = upright
        if upright:
            cal = calib_mod.rotate_camera_calib_cw90deg(cal)
        self.cal = cal
        w, h = int(cal.get_image_size()[0]), int(cal.get_image_size()[1])
        self.wh = (w, h)
        self.T_dev_cam = se3_mat(cal.get_transform_device_camera())
        self.lin = calib_mod.get_linear_camera_calibration(
            pin_size, pin_size, pin_f, "camera-rgb",
            cal.get_transform_device_camera())
        self.pin_size, self.pin_f = pin_size, pin_f
        self._ray_lut = {}

    def timestamps(self):
        """RGB timestamps clamped to the ADT ground-truth coverage window.

        The GT (closed-loop poses + digital-twin depth) starts ~3-4 s after and
        ends ~2-3 s before the vrs stream, so a fixed head/tail frame margin
        would query poses outside the GT window. The window comes from the
        provider's get_start_time_ns / get_end_time_ns, shrunk by a 0.5 s
        safety margin."""
        tss = list(self.dp.get_aria_device_capture_timestamps_ns(self.sid))
        t0 = int(self.dp.get_start_time_ns())
        t1 = int(self.dp.get_end_time_ns())
        m = int(0.5e9)  # 0.5 s safety margin inside the GT window
        clamped = [t for t in tss if t0 + m <= t <= t1 - m]
        assert len(clamped) >= 100, (
            f"only {len(clamped)} frames inside GT bounds [{t0},{t1}] "
            f"(stream had {len(tss)}) — sequence unusable")
        return clamped

    def frame(self, ts: int):
        im = self.dp.get_aria_image_by_timestamp_ns(ts, self.sid)
        assert im.is_valid(), f"invalid RGB at {ts}"
        arr = im.data().to_numpy_array()
        if self.upright:
            arr = np.rot90(arr, k=3).copy()
        return arr

    def depth_mm(self, ts: int):
        dm = self.dp.get_depth_image_by_timestamp_ns(ts, self.sid)
        assert dm.is_valid(), f"invalid depth at {ts}"
        arr = dm.data().to_numpy_array()
        if self.upright:
            arr = np.rot90(arr, k=3).copy()
        return arr.astype(np.float64)

    def c2w(self, ts: int) -> np.ndarray:
        p = self.dp.get_aria_3d_pose_by_timestamp_ns(ts)
        assert p.is_valid(), f"no GT pose at {ts}"
        T_scene_dev = se3_mat(p.data().transform_scene_device)
        return T_scene_dev @ self.T_dev_cam

    def undistort(self, img, nearest=False):
        method = (self.interp.NEAREST_NEIGHBOR if nearest
                  else self.interp.BILINEAR)
        return self.calib_mod.distort_by_calibration(img, self.lin, self.cal,
                                                     method)

    def rays(self, stride: int) -> np.ndarray:
        """Unit rays for the (possibly rotated) fisheye grid at a stride."""
        key = stride
        if key not in self._ray_lut:
            w, h = self.wh
            rr = np.full((len(range(0, h, stride)), len(range(0, w, stride)), 3),
                         np.nan)
            for i, v in enumerate(range(0, h, stride)):
                for j, u in enumerate(range(0, w, stride)):
                    r = self.cal.unproject_no_checks(np.array([u, v], np.float64))
                    if r is not None:
                        r = np.asarray(r, np.float64).ravel()
                        rr[i, j] = r / np.linalg.norm(r)
            self._ray_lut[key] = rr
        return self._ray_lut[key]

    def pin_rays(self, stride: int) -> np.ndarray:
        s, f, c = self.pin_size, self.pin_f, (self.pin_size - 1) / 2.0
        vs, us = np.meshgrid(np.arange(0, s, stride), np.arange(0, s, stride),
                             indexing="ij")
        r = np.stack([(us - c) / f, (vs - c) / f, np.ones_like(us, float)], -1)
        return r / np.linalg.norm(r, axis=-1, keepdims=True)


def xyz_from_depth(depth_mm, rays_unit, c2w, conv: str, stride: int):
    """World xyz on the stride grid. conv: 'z' (depth=z) or 'range' (=|ray|)."""
    d = depth_mm[::stride, ::stride] * 1e-3
    d[d <= 1e-4] = np.nan
    r = rays_unit[: d.shape[0], : d.shape[1]]
    if conv == "z":
        scale = d / r[..., 2]
    else:
        scale = d
    cam = r * scale[..., None]
    xyz = cam @ c2w[:3, :3].T + c2w[:3, 3]
    xyz[~np.isfinite(d)] = np.nan
    return xyz


def reproj_score(ex: SeqExtractor, depths, c2ws, conv: str) -> tuple:
    """Project view0 world cloud into view1's fisheye grid, compare depth."""
    rays = ex.rays(8)
    xyz0 = xyz_from_depth(depths[0], rays, c2ws[0], conv, 8).reshape(-1, 3)
    xyz0 = xyz0[np.isfinite(xyz0).all(1)]
    if len(xyz0) < 300:
        return 1.0, 0.0, 0
    w2c1 = np.linalg.inv(c2ws[1])
    cam1 = xyz0 @ w2c1[:3, :3].T + w2c1[:3, 3]
    rng_ = np.linalg.norm(cam1, axis=1)
    ok = cam1[:, 2] > 0.05
    uv = []
    for p in cam1[ok]:
        q = ex.cal.project(p)
        uv.append([np.nan, np.nan] if q is None else np.asarray(q).ravel())
    uv = np.asarray(uv, np.float64)
    w, h = ex.wh
    inb = np.isfinite(uv).all(1)
    inb &= (uv[:, 0] >= 0) & (uv[:, 0] < w - 1) & (uv[:, 1] >= 0) & (uv[:, 1] < h - 1)
    if inb.sum() < 300:
        return 1.0, 0.0, int(inb.sum())
    ui, vi = uv[inb, 0].round().astype(int), uv[inb, 1].round().astype(int)
    dmm = depths[1][vi, ui] * 1e-3
    val = dmm > 1e-4
    if conv == "z":
        ref = cam1[ok][inb][:, 2]
    else:
        ref = rng_[ok][inb]
    rel = np.abs(dmm[val] - ref[val]) / np.maximum(ref[val], 1e-6)
    return float(np.median(rel)), float((rel < 0.08).mean()), int(val.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adt-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-seqs", type=int, default=20)
    ap.add_argument("--tuples-per-seq", type=int, default=4)
    ap.add_argument("--views", type=int, default=8)
    ap.add_argument("--stride-s", type=float, default=0.8,
                    help="seconds between tuple views")
    ap.add_argument("--seed", type=int, default=20260804)
    ap.add_argument("--pin-size", type=int, default=640)
    ap.add_argument("--pin-f", type=float, default=280.0)
    ap.add_argument("--xyz-ds", type=int, default=4)
    ap.add_argument("--cloud-stride", type=int, default=8)
    ap.add_argument("--cloud-cap", type=int, default=200_000)
    ap.add_argument("--no-upright", action="store_true",
                    help="fallback: keep native (rotated) orientation")
    ap.add_argument("--jpeg-q", type=int, default=95)
    args = ap.parse_args()
    from PIL import Image

    root, out = Path(args.adt_root), Path(args.out)
    seqs = sorted([d.name for d in root.iterdir()
                   if d.is_dir() and not d.name.startswith("._")])
    assert seqs, f"no sequences under {root}"
    order = np.random.default_rng(args.seed).permutation(len(seqs))
    upright = not args.no_upright
    frames_root = out / "frames"
    tuples_fe, tuples_pin = [], []
    depth_conv, n_done = None, 0

    for si in order:
        if n_done >= args.n_seqs:
            break
        seq = seqs[si]
        sd = root / seq
        if not (sd / "video.vrs").is_file() or not (sd / "depth_images.vrs").is_file():
            print(f"[skip] {seq}: missing video/depth vrs")
            continue
        try:
            dp = get_provider(sd)
            ex = SeqExtractor(dp, upright, args.pin_size, args.pin_f)
            tss = ex.timestamps()
        except Exception as e:
            print(f"[skip] {seq}: provider failed: {e}")
            continue
        if len(tss) < 300:
            print(f"[skip] {seq}: only {len(tss)} RGB frames")
            continue
        dt = np.median(np.diff(tss[:200])) * 1e-9
        step = max(1, int(round(args.stride_s / dt)))
        span = (args.views - 1) * step
        # timestamps() already clamps to GT bounds (+0.5 s margin); keep only a
        # small extra trim so short sequences still fit 4 tuples
        lo, hi = 15, len(tss) - 15 - span
        if hi <= lo:
            print(f"[skip] {seq}: too short for span {span}")
            continue
        starts = np.linspace(lo, hi, args.tuples_per_seq).astype(int)

        seq_ok = True
        for ti, s0 in enumerate(starts):
            fids = [int(s0 + k * step) for k in range(args.views)]
            try:
                frames = [ex.frame(tss[f]) for f in fids]
                depths = [ex.depth_mm(tss[f]) for f in fids]
                c2ws = [ex.c2w(tss[f]) for f in fids]
            except Exception as e:
                print(f"[skip] {seq} tuple{ti}: {e}")
                seq_ok = False
                break
            if depth_conv is None:  # one-time depth semantics adjudication
                scores = {c: reproj_score(ex, depths, c2ws, c) for c in ("z", "range")}
                print(f"[depth-conv] z: med={scores['z'][0]:.4f} inl={scores['z'][1]:.2%} "
                      f"| range: med={scores['range'][0]:.4f} inl={scores['range'][1]:.2%}")
                depth_conv = min(scores, key=lambda c: scores[c][0])
                med, inl, _n = scores[depth_conv]
                assert inl >= 0.50 and med <= 0.08, (
                    "neither depth convention passes the reprojection gate — "
                    "pose/calib chain broken, do not proceed")
                print(f"[depth-conv] adjudicated: depth = {depth_conv!r}")
            if ti == 0:
                med, inl, n = reproj_score(ex, depths, c2ws, depth_conv)
                print(f"  [gate] {seq}: med={med:.4f} inl={inl:.2%} n={n}")
                if not (med <= 0.08 and inl >= 0.50):
                    print(f"[skip] {seq}: failed reprojection gate")
                    seq_ok = False
                    break

            rays_fe = ex.rays(args.xyz_ds)
            rays_cl = ex.rays(args.cloud_stride)
            pin_rays = ex.pin_rays(args.xyz_ds)
            tid = f"adt_{n_done:02d}_{ti}"
            vs_fe, vs_pin, xyz_fe, xyz_pin, cloud = [], [], [], [], []
            for k, f in enumerate(fids):
                im, dmm, c2w = frames[k], depths[k], c2ws[k]
                fe_rel = f"fisheye/{tid}_{k}.jpg"
                pin_rel = f"pinhole/{tid}_{k}.jpg"
                (frames_root / "fisheye").mkdir(parents=True, exist_ok=True)
                (frames_root / "pinhole").mkdir(parents=True, exist_ok=True)
                Image.fromarray(im).save(frames_root / fe_rel, quality=args.jpeg_q)
                und = ex.undistort(im)
                Image.fromarray(und).save(frames_root / pin_rel, quality=args.jpeg_q)
                w, h = ex.wh
                vs_fe.append({"img": fe_rel, "w": w, "h": h})
                vs_pin.append({"img": pin_rel, "w": args.pin_size, "h": args.pin_size})
                xyz_fe.append(xyz_from_depth(dmm, rays_fe, c2w, depth_conv, args.xyz_ds))
                dpin = ex.undistort(dmm.astype(np.float32), nearest=True).astype(np.float64)
                xyz_pin.append(xyz_from_depth(dpin, pin_rays, c2w, depth_conv, args.xyz_ds))
                cl = xyz_from_depth(dmm, rays_cl, c2w, depth_conv, args.cloud_stride)
                cl = cl.reshape(-1, 3)
                cloud.append(cl[np.isfinite(cl).all(1)])
            gt_pts = subsample(np.concatenate(cloud).astype(np.float32),
                               args.cloud_cap, seed=args.seed)
            save_gt_npz(out / "gt" / f"{tid}_fe.npz", np.stack(c2ws),
                        np.stack(xyz_fe), args.xyz_ds, gt_pts)
            save_gt_npz(out / "gt" / f"{tid}_pin.npz", np.stack(c2ws),
                        np.stack(xyz_pin), args.xyz_ds, gt_pts)
            tuples_fe.append({"id": f"{tid}_fe", "seq": seq, "views": vs_fe,
                              "gt": f"gt/{tid}_fe.npz"})
            tuples_pin.append({"id": f"{tid}_pin", "seq": seq, "views": vs_pin,
                               "gt": f"gt/{tid}_pin.npz"})
        if seq_ok:
            n_done += 1
            print(f"[adt] {n_done}/{args.n_seqs}: {seq} "
                  f"({args.tuples_per_seq} tuples, step={step} frames)")

    assert n_done == args.n_seqs, f"only {n_done} usable sequences"
    p1 = save_tuples(out / "fisheye", "adt", "fisheye", str(frames_root), tuples_fe)
    p2 = save_tuples(out / "pinhole", "adt", "pinhole", str(frames_root), tuples_pin)
    meta = {"depth_conv": depth_conv, "upright": upright, "n_seqs": n_done,
            "pin": [args.pin_size, args.pin_f], "seed": args.seed}
    with open(out / "prepare_meta.json", "w") as f:
        json.dump(meta, f, indent=1)
    print(f"[adt] DONE fisheye={len(tuples_fe)} pinhole={len(tuples_pin)} "
          f"tuples -> {p1} / {p2}\n[adt] meta: {meta}")


if __name__ == "__main__":
    main()
