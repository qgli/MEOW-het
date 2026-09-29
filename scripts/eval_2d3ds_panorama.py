"""Single-panorama reconstruction on Stanford 2D-3D-S (hand-resized input).

Evaluates checkpoints (e.g. MEOW and the public MapAnything weights) on the
equirectangular (ERP) panoramas of Stanford 2D-3D-S, a public panorama benchmark also
used by CAM3R, Wid3R and Fisheye3R. eval_2d3ds_panorama_v2.py runs the same evaluation
through the official inference path.

Data ($MEOW_2D3DS_ROOT/area_*/pano/{rgb,depth,pose,semantic}):
  - rgb:    4096x2048 RGBA, 2:1 ERP
  - depth:  4096x2048 uint16, metres = raw/512, 65535 = invalid
  - pose:   *.json with camera_rt_matrix (3x4 [R|t] = world->cam, w2c),
            camera_location (world optical centre)
  - ERP rays from generate_rays_erp(H,W) below (Z-forward, Y-up, centre=+Z). The GT
    stays in this Y-up frame; eval_2d3ds_panorama_v2.py converts it to the OpenCV
    (Y-down) camera frame of the predictions.

Metrics (image-only, single-view ERP reconstruction):
  - Sim3-aligned Chamfer L1/L2 (scale-invariant; depth unit ambiguity absorbed)
  - depth rel / delta1 (after Sim3 scale)
Multi-view pose is evaluated by eval_2d3ds_pose.py and eval_2d3ds_pose_v2.py.

Usage:
  CUDA_VISIBLE_DEVICES=0 MEOW_2D3DS_ROOT=/path/to/2d3ds python scripts/eval_2d3ds_panorama.py \
      --ckpts NAME=/path/to/checkpoint.pth MapAnything=checkpoints/facebook_map-anything-apache.pth \
      --area area_1 --num-frames 20 --res 518 \
      --out experiments/pub_bench/2d3ds_recon
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

THIS = Path(__file__).resolve()
ROOT = THIS.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(THIS.parent))

from mapanything.models import _init_hydra_config, init_model  # noqa: E402

# ERP ray generator inlined (avoids importing the bpy-dependent renderer).
# Identical to generate_rays_erp of the first-generation renderer
# (lenscope/gen1/scene_generator/render_unicol_dataset.py): Z-forward frame,
# centre pixel -> +Z, longitude increases rightward.
def generate_rays_erp(H, W):
    j = np.arange(W, dtype=np.float64)
    i = np.arange(H, dtype=np.float64)
    jj, ii = np.meshgrid(j, i)
    lon = (jj + 0.5) / W * 2.0 * np.pi - np.pi          # [-pi, pi)
    lat = np.pi / 2.0 - (ii + 0.5) / H * np.pi          # (-pi/2, pi/2)
    x = np.cos(lat) * np.sin(lon)                        # X-right
    y = np.sin(lat)                                      # Y-up
    z = np.cos(lat) * np.cos(lon)                        # Z-forward
    rays = np.stack([x, y, z], axis=-1)                  # (H,W,3)
    valid = np.ones((H, W), dtype=bool)
    return rays, valid

from pointcloud_metrics import umeyama_sim3, chamfer  # noqa: E402

STANFORD_ROOT = os.environ.get("MEOW_2D3DS_ROOT", ".")
DEPTH_SCALE = 512.0       # 2D3DS: meters = raw / 512
DEPTH_INVALID = 65535


# --------------------------------------------------------------------------
# Model build / load (ar_prob=1.0 so that the aspect-ratio encoder exists;
# checkpoints without trained encoder weights keep its zero initialisation,
# which adds nothing).
# --------------------------------------------------------------------------
def build_model(device):
    cfg = _init_hydra_config(
        "configs/train.yaml",
        overrides=[
            "model=mapanything",
            "machine=default",
            "model/task=aug_training",
            "++model.task.ar_prob=1.0",
        ],
    )
    model = init_model(
        model_str=cfg.model.model_str,
        model_config=cfg.model.model_config,
        torch_hub_force_reload=False,
    )
    return model.to(device).eval()


def load_ckpt(model, sd):
    incompat = model.load_state_dict(sd, strict=False)
    bad = [k for k in incompat.missing_keys if not k.startswith("ar_encoder.")]
    if bad:
        raise RuntimeError(f"Unexpected missing keys: {bad[:8]}")
    if incompat.unexpected_keys:
        raise RuntimeError(f"Unexpected keys: {incompat.unexpected_keys[:8]}")


# --------------------------------------------------------------------------
# 2D3DS pano frame loader
# --------------------------------------------------------------------------
def list_frames(area: str):
    rgb_dir = Path(STANFORD_ROOT) / area / "pano" / "rgb"
    frames = []
    for p in sorted(rgb_dir.glob("*_rgb.png")):
        base = p.name.replace("_rgb.png", "")
        frames.append(base)
    return frames


def load_frame(area: str, base: str, res: int):
    """Load one ERP frame -> dict with RGB array, rays, GT pts3d (cam frame), mask.

    Returns None if fewer than 1000 depth pixels are valid. ERP resized to
    (res, 2*res) (keep 2:1).
    """
    pano = Path(STANFORD_ROOT) / area / "pano"
    H, W = res, res * 2  # keep ERP 2:1
    # RGB
    rgb = Image.open(pano / "rgb" / f"{base}_rgb.png").convert("RGB")
    rgb = rgb.resize((W, H), Image.BILINEAR)
    rgb_np = np.asarray(rgb).astype(np.float32) / 255.0  # (H,W,3)
    # depth (meters); resize with NEAREST to avoid mixing invalid
    dep = Image.open(pano / "depth" / f"{base}_depth.png")
    dep = dep.resize((W, H), Image.NEAREST)
    dep_raw = np.asarray(dep).astype(np.float32)
    valid = (dep_raw > 0) & (dep_raw < DEPTH_INVALID)
    depth_m = np.where(valid, dep_raw / DEPTH_SCALE, 0.0)
    if valid.sum() < 1000:
        return None
    # ERP rays (cam frame, unit; the renderer's Y-up ray frame, not converted to OpenCV)
    rays, ray_valid = generate_rays_erp(H, W)  # (H,W,3), (H,W)
    rays = rays.astype(np.float32)
    # GT cam-frame points = depth(ray-distance) * ray_dir
    gt_cam = depth_m[..., None] * rays  # (H,W,3)
    mask = valid & ray_valid
    return {"rgb": rgb_np, "rays": rays, "gt_cam": gt_cam, "mask": mask,
            "H": H, "W": W}


def to_view(frame, device, data_norm="dinov2"):
    """Build a single MapAnything view dict (image-only) from a loaded frame."""
    import torchvision.transforms as T
    H, W = frame["H"], frame["W"]
    rgb = torch.from_numpy(frame["rgb"]).permute(2, 0, 1)  # (3,H,W) in [0,1]
    # dinov2 normalization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = ((rgb - mean) / std).unsqueeze(0)  # (1,3,H,W)
    view = {
        "img": img.to(device),
        "data_norm_type": [data_norm],
        "true_shape": torch.tensor([[H, W]]).to(device),
        "instance": [f"2d3ds"],
        "idx": [0],
    }
    return view


@torch.no_grad()
def predict_pts3d(model, view):
    """Run model image-only -> per-pixel pts3d (cam or world frame, (H,W,3))."""
    preds = model([view])
    p = preds[0]
    return p["pts3d"][0].detach().cpu().numpy().astype(np.float32)


def eval_recon(pred_xyz, frame):
    """Sim3-aligned Chamfer + depth rel/d1 on one frame (single view)."""
    mask = frame["mask"]
    gt = frame["gt_cam"][mask]          # (N,3)
    pr = pred_xyz[mask]                 # (N,3)
    if len(gt) < 500:
        return None
    s, R, t = umeyama_sim3(pr, gt)
    pr_al = (s * (R @ pr.T).T) + t
    ch = chamfer(pr_al, gt)
    # depth = distance from origin along ray (cam frame both)
    gt_d = np.linalg.norm(gt, axis=1)
    pr_d = np.linalg.norm(pr_al, axis=1)
    rel = np.abs(pr_d - gt_d) / np.maximum(gt_d, 1e-6)
    d1 = (np.maximum(pr_d / np.maximum(gt_d, 1e-6),
                     gt_d / np.maximum(pr_d, 1e-6)) < 1.25).mean()
    return {"chamfer_L1": float(ch["L1"]), "chamfer_L2": float(ch["L2"]),
            "depth_rel": float(np.median(rel)), "delta1": float(d1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--area", default="area_1")
    ap.add_argument("--num-frames", type=int, default=20)
    ap.add_argument("--res", type=int, default=518)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    ckpts = []
    for spec in args.ckpts:
        name, path = spec.split("=", 1)
        assert os.path.isfile(path), f"ckpt not found: {path}"
        ckpts.append((name, path))

    frames_ids = list_frames(args.area)
    rng = np.random.default_rng(args.seed)
    if len(frames_ids) > args.num_frames:
        frames_ids = list(rng.choice(frames_ids, size=args.num_frames, replace=False))
    print(f"[2d3ds] area={args.area} frames={len(frames_ids)} res={args.res} "
          f"ckpts={[n for n,_ in ckpts]}")

    # preload frames once
    frames = []
    for b in frames_ids:
        fr = load_frame(args.area, b, args.res)
        if fr is not None:
            frames.append((b, fr))
    print(f"[2d3ds] loaded {len(frames)} valid frames")

    print("[model] building once")
    model = build_model(args.device)

    records = {}
    for name, path in ckpts:
        c = torch.load(path, map_location="cpu", weights_only=False)
        load_ckpt(model, c.get("model", c))
        res_list = []
        for b, fr in frames:
            view = to_view(fr, args.device)
            pred = predict_pts3d(model, view)
            m = eval_recon(pred, fr)
            if m is not None:
                res_list.append(m)
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()
        agg = {k: float(np.mean([r[k] for r in res_list])) for k in res_list[0]} if res_list else {}
        records[name] = {"n_frames": len(res_list), **agg}
        print(f"  {name:8s} n={len(res_list):3d}  "
              f"chamferL1={agg.get('chamfer_L1',0):.4f}  "
              f"depth_rel={agg.get('depth_rel',0):.4f}  d1={agg.get('delta1',0):.3f}")

    out_json = args.out / "2d3ds_recon_results.json"
    with open(out_json, "w") as f:
        json.dump({"area": args.area, "res": args.res, "records": records}, f, indent=2)
    print(f"\n[out] {out_json}")


if __name__ == "__main__":
    main()
