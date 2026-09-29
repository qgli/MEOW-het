#!/usr/bin/env python3
"""Convert Leica BLK360 G2 stations into packs.

Frame law (zero residual on two-station checks within the same modality):
    az_pano = -az_cloud + pi/2        # mirror (handedness) + 90deg constant
    el_pano = el_cloud
Per-station E57 clouds are in the station-local scanner frame; the station
pose (full R+t) lives in the data3D section of the E57 XML (Cyclone REGISTER
output). The panorama/cloud azimuth law above is a global constant (verified
on stations with very different orientations).

The pack output convention matches the second-generation scenes: rays use the
same equirectangular grid (x=ce*sin az, y=sin el, z=ce*cos az; y-up camera
labels) and c2w is R_scan @ UNICOL_SWAP (det=-1, as in the second-generation
metadata), so BLK packs are drop-in consumable by every loader and evaluation
script of that convention. The pixel layout is unchanged (in both
conventions, u to the right means turning right).

Inputs per station:
    Setup XXX.e57   structured cloud (xyz/rgb/intensity/rowIndex/colIndex)
    Setup XXX.jpg   8K display panorama (Leica ISP tone) -> pack rgb
    Setup XXX.exr   8K linear HDR panorama -> HDR sidecar (archival)
    Setup XXX.txt   panorama pose (position; the orientation quaternion is not used)
Project level: FinalizeReport.pdf (Cyclone link overlap/strength = covisibility prior).

Output: <scene>/blk_poseNN_pack.npz with the same schema as the
second-generation packs (rgb f16 sRGB / rays f16 / depth f16 / mask u8 /
normal f16 (optional) / sem, inst, flags zero placeholders) + covisibility/v0.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np

AZ_SIGN = -1.0            # mirror
AZ_OFFSET = np.pi / 2     # +90 deg
MAX_RANGE = 45.0
# scanner z-up frame -> unicol camera labels (y-up, z-forward):
# blk_ray(u,v) == UNICOL_SWAP @ genesis_erp_rays(u,v) pointwise (0.00 deg).
# Self-inverse, det=-1; second-generation c2w matrices are det=-1 by convention too.
UNICOL_SWAP = np.array([[1.0, 0, 0], [0, 0, 1.0], [0, 1.0, 0]])


def _e57_logical_xml(e57_path: Path) -> str:
    """E57 files are paged: every 1024-byte physical page ends with a 4-byte
    CRC. Reading the XML section as raw bytes therefore interleaves 4 junk
    bytes every 1020, and a float that straddles a page boundary is silently
    corrupted (a translation component that fails to parse defaults to 0.0
    and moves the station by metres; rotations lose mantissa digits).
    De-page first, parse XML second; never regex the raw byte stream."""
    # The XML section sits at the file tail; read the last 4 MB (aligned to
    # the 1024-byte page grid) instead of the whole ~800 MB file. Fall back
    # to a full read if the tail window somehow misses the XML start.
    size = e57_path.stat().st_size
    start_phys = max(0, (size - 4 * 1024 * 1024) // 1024 * 1024)
    with open(e57_path, "rb") as f:
        f.seek(start_phys)
        raw = f.read()
    logical = b"".join(raw[i:i + 1020] for i in range(0, len(raw), 1024))
    start = logical.rfind(b"<?xml")
    if start < 0:
        raw = e57_path.read_bytes()
        logical = b"".join(raw[i:i + 1020] for i in range(0, len(raw), 1024))
        start = logical.rfind(b"<?xml")
    xml = logical[start:].decode("utf-8", errors="ignore")
    return xml[: xml.rfind("</e57Root>") + len("</e57Root>")]


def read_e57_pose(e57_path: Path):
    """Authoritative station pose = the pose node of the data3D section
    (structural, not positional: the XML carries 7 pose nodes -- 1 data3D scan
    pose + 6 images2D camera-representation poses; cube-face poses differ).
    Returns (R 3x3 det=+1 scanner->project, t 3,)."""
    import xml.etree.ElementTree as ET
    xml = re.sub(r'\sxmlns="[^"]*"', "", _e57_logical_xml(e57_path), count=1)
    root = ET.fromstring(xml)
    node = root.find("data3D/vectorChild/pose")
    assert node is not None, f"{e57_path.name}: no data3D pose in E57 XML"

    def vec(tag, keys):
        el = node.find(tag)
        return [float(el.find(k).text) if el is not None and el.find(k) is not None
                and el.find(k).text else 0.0 for k in keys]

    w, x, y, z = vec("rotation", "wxyz") or [1, 0, 0, 0]
    if w == x == y == z == 0.0:
        w = 1.0
    t = np.array(vec("translation", "xyz"))
    R = np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w),   2*(x*z+y*w)],
        [2*(x*y+z*w),   1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w),   2*(y*z+x*w),   1-2*(x*x+y*y)]])
    # invariant: the panorama (spherical) representation pose must equal the
    # scan pose; a firmware/export change that breaks this must fail loudly.
    last = root.findall("images2D/vectorChild/pose")
    if last:
        # E57 writes zero-valued floats as empty elements (<z type="Float"/>,
        # text=None) -- read with the same default-0.0 tolerance as above.
        tr = last[-1].find("translation")
        lt = [float(tr.find(k).text) if tr is not None and tr.find(k) is not None
              and tr.find(k).text else 0.0 for k in "xyz"]
        assert np.abs(np.array(lt) - t).max() < 1e-6, \
            f"{e57_path.name}: pano pose translation != scan pose (lever arm?)"
    return R, t


def cloud_to_erp(e57_path: Path, W: int, H: int):
    """Scanner-centric cloud -> (depth[H,W] m, cloudcolor[H,W,3] u8, valid[H,W])."""
    import pye57

    f = pye57.E57(str(e57_path))
    raw = f.read_scan_raw(0)                      # local scanner frame (no pose)
    d = np.stack([raw["cartesianX"], raw["cartesianY"], raw["cartesianZ"]], 1)
    col = np.stack([raw["colorRed"], raw["colorGreen"], raw["colorBlue"]], 1)
    rng = np.linalg.norm(d, axis=1).astype(np.float32)
    keep = (rng > 0.3) & (rng < MAX_RANGE)
    d, col, rng = d[keep], col[keep], rng[keep]
    dn = d / rng[:, None]
    az = np.arctan2(dn[:, 1], dn[:, 0])
    el = np.arcsin(np.clip(dn[:, 2], -1, 1))
    az_p = AZ_SIGN * az + AZ_OFFSET
    u = ((az_p + np.pi) / (2 * np.pi) * W).astype(np.int64) % W
    v = np.clip(((np.pi / 2 - el) / np.pi * H).astype(np.int64), 0, H - 1)
    o = np.argsort(-rng)                           # nearest wins
    depth = np.zeros((H, W), np.float32)
    cc = np.zeros((H, W, 3), np.uint8)
    depth[v[o], u[o]] = rng[o]
    cc[v[o], u[o]] = col[o]
    return depth, cc, depth > 0


def genesis_erp_rays(W: int, H: int) -> np.ndarray:
    """Unit ray grid in the second-generation (unicol) camera convention that
    the packs carry: x = cos(el)sin(az_p), y = sin(el), z = cos(el)cos(az_p)."""
    us, vs = np.meshgrid(np.arange(W), np.arange(H))
    az_p = (us + 0.5) / W * 2 * np.pi - np.pi
    el = np.pi / 2 - (vs + 0.5) / H * np.pi
    ce = np.cos(el)
    return np.stack([ce * np.sin(az_p), np.sin(el), ce * np.cos(az_p)], -1).astype(np.float32)


def erp_rays(W: int, H: int) -> np.ndarray:
    """Unit ray directions of the equirectangular panorama grid in the project
    frame (cloud frame == project frame for REGISTER exports). Inverts the
    frame law."""
    us, vs = np.meshgrid(np.arange(W), np.arange(H))
    az_p = (us + 0.5) / W * 2 * np.pi - np.pi
    el = np.pi / 2 - (vs + 0.5) / H * np.pi
    az = (az_p - AZ_OFFSET) / AZ_SIGN
    ce = np.cos(el)
    return np.stack([ce * np.cos(az), ce * np.sin(az), np.sin(el)], -1).astype(np.float32)


def _convert_one(task):
    """One station -> pack on disk. Runs in a worker process (stations are
    independent; per-station math identical to the serial path)."""
    i, e57p, out_s, W, H = task
    from PIL import Image
    e57p, out = Path(e57p), Path(out_s)
    stem = e57p.stem
    sdir = e57p.parent
    Rs, t = read_e57_pose(e57p)
    depth_f, _cc, _m = cloud_to_erp(e57p, W, H)
    depth = depth_f.astype(np.float16)
    mask = (depth > 0).astype(np.uint8)
    img_path = next((sdir / f"{stem}{ext}" for ext in (".jpg", ".png")
                     if (sdir / f"{stem}{ext}").exists()), None)
    assert img_path is not None, f"no pano image for {stem}"
    rgb = np.asarray(Image.open(img_path).convert("RGB").resize(
        (W, H), Image.LANCZOS), np.float32) / 255.0
    rays = genesis_erp_rays(W, H)          # second-generation pack convention
    tag = f"blk_pose{i:02d}"
    np.savez_compressed(
        out / f"{tag}_pack.npz",
        rgb=rgb.astype(np.float16), rays=rays.astype(np.float16),
        depth=depth, mask=mask,
        sem=np.zeros((H, W), np.uint8), inst=np.zeros((H, W), np.uint16),
        flags=np.zeros((H, W), np.uint8),
        normal=np.zeros((H, W, 3), np.float16),
    )
    exr = sdir / f"{stem}.exr"
    if exr.exists():
        import shutil
        shutil.copy(exr, out / f"{tag}_erp_hdr.exr")
    c2w = np.eye(4)
    c2w[:3, :3] = Rs @ UNICOL_SWAP          # det=-1, second-generation convention
    c2w[:3, 3] = t
    meta = {"tag": tag, "base_name": "erp", "base_type": "erp",
            "pose_index": i, "fov_deg": 360.0, "resolution": [W, H],
            "camera_to_world_unicol_4x4": [[round(v, 6) for v in row]
                                           for row in c2w.tolist()],
            "pack_file": f"{tag}_pack.npz", "station": stem}
    print(f"[blk2pack] {stem} -> {tag} (valid {mask.mean()*100:.1f}%)", flush=True)
    return i, meta, depth, (Rs, t)


def compute_covis(depths, poses, W, H, device="auto"):
    """Mutual-projection covisibility (10cm agreement). GPU (torch) when
    available, CPU numpy otherwise -- identical math, float-order differences
    only (the two paths agree to within 0.005)."""
    n = len(depths)
    cov = np.eye(n, dtype=np.float32)
    use_gpu = False
    if device in ("auto", "cuda"):
        try:
            import torch
            use_gpu = torch.cuda.is_available()
        except ImportError:
            use_gpu = False
        if device == "cuda" and not use_gpu:
            raise RuntimeError("--device cuda requested but torch/cuda unavailable")
    if use_gpu:
        import torch
        dev = torch.device("cuda")
        with torch.no_grad():
            rays_t = torch.from_numpy(erp_rays(W, H)).to(dev)
            D = torch.from_numpy(np.stack(depths)).to(dev)
            Rs = torch.from_numpy(np.stack([p[0] for p in poses]).astype(np.float32)).to(dev)
            ts = torch.from_numpy(np.stack([p[1] for p in poses]).astype(np.float32)).to(dev)
            for a in range(n):
                da = D[a]
                va = da > 0
                Pa = (rays_t * da.unsqueeze(-1))[va] @ Rs[a].T + ts[a]
                for b in range(n):
                    if a == b:
                        continue
                    d = (Pa - ts[b]) @ Rs[b]
                    r = torch.linalg.norm(d, dim=1)
                    dn = d / r.clamp_min(1e-9).unsqueeze(1)
                    az_p = AZ_SIGN * torch.atan2(dn[:, 1], dn[:, 0]) + AZ_OFFSET
                    el = torch.asin(dn[:, 2].clamp(-1.0, 1.0))
                    uu = ((az_p + np.pi) / (2 * np.pi) * W).long() % W
                    vv = (((np.pi / 2 - el) / np.pi * H).long()).clamp(0, H - 1)
                    db = D[b][vv, uu]
                    vis = (db > 0) & ((db - r).abs() < 0.10)
                    cov[a, b] = float(vis.float().mean().item())
        return cov
    rays_full = erp_rays(W, H)
    for a in range(n):
        Ra, ta = poses[a]
        va = depths[a] > 0
        Pa = (rays_full * depths[a][..., None])[va] @ Ra.T + ta
        for b in range(n):
            if a == b:
                continue
            Rb, tb = poses[b]
            d = (Pa - tb) @ Rb
            r = np.linalg.norm(d, axis=1)
            dn = d / np.maximum(r[:, None], 1e-9)
            az_p = AZ_SIGN * np.arctan2(dn[:, 1], dn[:, 0]) + AZ_OFFSET
            el = np.arcsin(np.clip(dn[:, 2], -1, 1))
            uu = ((az_p + np.pi) / (2 * np.pi) * W).astype(np.int64) % W
            vv = np.clip(((np.pi / 2 - el) / np.pi * H).astype(np.int64), 0, H - 1)
            db = depths[b][vv, uu]
            vis = (db > 0) & (np.abs(db - r) < 0.10)
            cov[a, b] = float(vis.mean())
    return cov


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--station-dir", required=True, help="dir with Setup*.e57/.jpg/.exr/.txt")
    ap.add_argument("--out", required=True, help="scene output dir")
    ap.add_argument("--erp-w", type=int, default=3072)
    ap.add_argument("--jobs", type=int, default=0,
                    help="station-parallel workers (0 = min(8, cpus//4))")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"],
                    help="covis on GPU (torch) when available")
    args = ap.parse_args()
    W = args.erp_w
    H = W // 2
    sdir, out = Path(args.station_dir), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    stations = sorted(sdir.glob("*.e57"))
    n = len(stations)
    jobs = args.jobs or min(8, max(1, (os.cpu_count() or 4) // 4))
    jobs = min(jobs, n)
    tasks = [(i, str(p), str(out), W, H) for i, p in enumerate(stations)]
    if jobs > 1:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(jobs) as pool:
            results = list(pool.imap_unordered(_convert_one, tasks))
    else:
        results = [_convert_one(t) for t in tasks]
    results.sort(key=lambda r: r[0])
    metas = [r[1] for r in results]
    depths = [r[2].astype(np.float32) for r in results]
    poses = [r[3] for r in results]

    cov = compute_covis(depths, poses, W, H, device=args.device)
    cdir = out / "covisibility" / "v0"
    cdir.mkdir(parents=True, exist_ok=True)
    np.save(cdir / "covisibility.npy", np.minimum(cov, cov.T))
    (cdir / "frame_meta.json").write_text(json.dumps({
        "n_frames": n, "source": "blk2pack mutual projection (10cm agree)",
        "frames": [{"tag": m["tag"], "base_name": "erp", "base_type": "erp",
                    "index": i, "pose_index": i} for i, m in enumerate(metas)]}, indent=1))
    (out / "metadata.json").write_text(json.dumps({"frames": metas}, indent=1))
    print(f"[blk2pack] scene done: {n} stations, covis min-offdiag "
          f"{np.min(np.minimum(cov, cov.T)[~np.eye(n, dtype=bool)]) if n > 1 else 1.0:.2f}")


if __name__ == "__main__":
    main()
