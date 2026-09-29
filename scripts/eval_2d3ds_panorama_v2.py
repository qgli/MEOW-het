"""Single-view ERP reconstruction on Stanford 2D-3D-S through the official inference path.

Unlike eval_2d3ds_panorama.py, which resizes each panorama by hand to H=res, W=2*res
(518x1036) and calls model([view]) directly, this script uses the official pipeline
(load_images + model.infer; load_images maps a 2:1 panorama to 518x252, as in
eval_2d3ds_pose_v2.py) for single-image inference, then builds the GT exactly as the loader
crops the image (cover-scale to the output size, centre crop), so that prediction and GT are
pixel-aligned. --uncropped-gt resizes the GT from the whole panorama instead (the original
scoring, which is off by the cropped rows).

Single image: one equirectangular panorama (ERP) -> per-pixel pts3d_cam. GT pts3d =
(depth/512) * ERP ray. The ray generator (Y-up, from eval_2d3ds_panorama.py) differs
from the OpenCV Y-down camera frame of MapAnything point maps, so build_gt_at negates
the GT ray Y coordinate. --compare-yflip additionally scores the prediction with its
camera-frame y negated (the reflected convention), as a diagnostic.

Sim3-aligned Chamfer L1/L2 + depth rel/delta1 (scale-invariant). The black pole pixels
of 2D3DS panoramas (pure-black RGB) are masked out so that they do not affect the metric.

Usage:
  CUDA_VISIBLE_DEVICES=0 MEOW_2D3DS_ROOT=/path/to/2d3ds python scripts/eval_2d3ds_panorama_v2.py \
    --ckpts MEOW=<path>:wrap MapAnything=checkpoints/facebook_map-anything-apache.pth \
    --areas area_5a area_5b --num-frames 20 \
    --out experiments/pub_bench/2d3ds_recon_v2
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

THIS = Path(__file__).resolve()
ROOT = THIS.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(THIS.parent))

from mapanything.utils.image import load_images  # noqa: E402
from eval_2d3ds_pose_v2 import (  # noqa: E402
    build_model_imagesonly, load_pth, load_variant_ckpt,
)
from meow_model import VARIANT_OVERRIDES, build_model_with_ar  # noqa: E402
from eval_2d3ds_panorama import (  # noqa: E402
    generate_rays_erp, list_frames, STANFORD_ROOT, DEPTH_SCALE, DEPTH_INVALID,
)
from pointcloud_metrics import umeyama_sim3, chamfer  # noqa: E402


def rgb_path(area, base):
    return str(Path(STANFORD_ROOT) / area / "pano" / "rgb" / f"{base}_rgb.png")


def build_gt_at(area, base, H, W, crop=True):
    """GT cam-frame pts3d at the model's output (H,W): depth/512 * ERP-ray.
    Also returns valid mask (depth valid AND not a black pole pixel).

    crop=True builds the GT the way the loader builds the input: the panorama is scaled to
    cover (H,W) and centre-cropped (a 4096x2048 panorama becomes 518x259, rows 3..254 are
    kept). crop=False resizes the whole panorama to (H,W)."""
    pano = Path(STANFORD_ROOT) / area / "pano"
    rgb_full = Image.open(rgb_path(area, base)).convert("RGB")
    if crop:
        w0, h0 = rgb_full.size
        s = max(W / w0, H / h0) + 1e-8  # as the cover-scale of mapanything.utils.cropping
        Wc, Hc = int(np.floor(w0 * s)), int(np.floor(h0 * s))
        top, left = (Hc - H) // 2, (Wc - W) // 2
    else:
        Wc, Hc, top, left = W, H, 0, 0
    dep = Image.open(pano / "depth" / f"{base}_depth.png").resize((Wc, Hc), Image.NEAREST)
    dep_raw = np.asarray(dep).astype(np.float32)
    valid = (dep_raw > 0) & (dep_raw < DEPTH_INVALID)
    depth_m = np.where(valid, dep_raw / DEPTH_SCALE, 0.0)
    rays, ray_valid = generate_rays_erp(Hc, Wc)
    rays = rays.astype(np.float32)
    rays[..., 1] *= -1.0  # Y-up ray frame -> OpenCV camera frame (Y down).
    gt_cam = depth_m[..., None] * rays
    # black-pole mask (2D3DS fills the poles with black); detected on the RGB at the same size
    rgb = np.asarray(rgb_full.resize((Wc, Hc), Image.BILINEAR)).astype(np.float32) / 255.0
    not_black = rgb.sum(axis=2) >= 0.04
    mask = valid & ray_valid & not_black
    sl = (slice(top, top + H), slice(left, left + W))
    return gt_cam[sl], mask[sl]


@torch.no_grad()
def predict_pts3d_cam(model, area, base, device, memory_efficient=False,
                      variant_inputs=False):
    """Official pipeline single-image -> pts3d_cam (H,W,3) + output (H,W).

    variant_inputs=True (checkpoints loaded with a variant suffix such as :wrap, as
    in het_2d3ds_pose.py): sets the per-view aspect_ratio (such checkpoints are
    trained with ar_prob=1; a 2D3DS panorama is 4096x2048, so exactly 2.0) and
    pano_wrap=True (every input here is a full panorama). With variant_inputs=False
    no extra inputs are set."""
    views = load_images([rgb_path(area, base)], verbose=False)
    if variant_inputs:
        with Image.open(rgb_path(area, base)) as im:
            w0, h0 = im.size
        for v in views:
            v["aspect_ratio"] = torch.tensor([w0 / max(h0, 1)],
                                             dtype=torch.float32)
            v["pano_wrap"] = True
    for v in views:
        for k, val in list(v.items()):
            if torch.is_tensor(val):
                v[k] = val.to(device, non_blocking=True)
    preds = model.infer(views, memory_efficient_inference=memory_efficient,
                        minibatch_size=1 if memory_efficient else None,
                        use_amp=True, amp_dtype="bf16",
                        apply_mask=True, mask_edges=True)
    p = preds[0]
    pts = p["pts3d_cam"][0].detach().cpu().numpy().astype(np.float32)  # (H,W,3)
    H, W = pts.shape[:2]
    return pts, H, W


def eval_recon(pred_xyz, gt_cam, mask):
    gt = gt_cam[mask]
    pr = pred_xyz[mask]
    if len(gt) < 500:
        return None
    s, R, t = umeyama_sim3(pr, gt)
    pr_al = (s * (R @ pr.T).T) + t
    ch = chamfer(pr_al, gt)
    gt_d = np.linalg.norm(gt, axis=1)
    pr_d = np.linalg.norm(pr_al, axis=1)
    rel = np.abs(pr_d - gt_d) / np.maximum(gt_d, 1e-6)
    d1 = (np.maximum(pr_d / np.maximum(gt_d, 1e-6),
                     gt_d / np.maximum(pr_d, 1e-6)) < 1.25).mean()
    return {"chamfer_L1": float(ch["L1"]), "chamfer_L2": float(ch["L2"]),
            "depth_rel": float(np.median(rel)), "delta1": float(d1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True,
                    help="name=path[:variant] specs; :wrap loads the aspect-ratio "
                         "build (AR token + pano_wrap), like het_2d3ds_pose")
    ap.add_argument("--area", default="area_5a")
    ap.add_argument("--areas", nargs="+", default=None,
                    help="evaluate several areas in one run (overrides --area); "
                         "per-area rows + pooled 'combined' are reported")
    ap.add_argument("--num-frames", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--memory-efficient", action="store_true")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--compare-yflip", action="store_true",
                    help="diagnostic: also score the reflected convention, i.e. "
                         "the same prediction after negating camera-frame y")
    ap.add_argument("--uncropped-gt", action="store_true",
                    help="resize the ground truth from the whole panorama (the original scoring); by "
                         "default it is cropped exactly as the loader crops the image")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    # NAME=path[:variant] specs, same grammar as het_2d3ds_pose.py; entries
    # without a suffix use the images-only build and loader.
    ckpts = []
    for spec in args.ckpts:
        name, path = spec.split("=", 1)
        variant = ""
        if ":" in path and path.rsplit(":", 1)[1] in VARIANT_OVERRIDES:
            path, variant = path.rsplit(":", 1)
        assert os.path.isfile(path), f"ckpt not found: {path}"
        ckpts.append((name, path, variant))
    ckpts.sort(key=lambda t: t[2])  # vanilla first, fewest model rebuilds

    areas = args.areas or [args.area]
    rng = np.random.default_rng(args.seed)
    area_frames = {}
    for area in areas:
        ids = list_frames(area)
        if len(ids) > args.num_frames:
            ids = list(rng.choice(ids, size=args.num_frames, replace=False))
        area_frames[area] = ids
    print(f"[2d3ds-recon-v2] areas={areas} frames="
          f"{ {a: len(v) for a, v in area_frames.items()} } "
          f"ckpts={[n for n, _, _ in ckpts]} (official load_images+infer, single-image)")

    model, cur_var = None, None
    records = {}
    records_yflip = {}
    for name, path, variant in ckpts:
        if model is None or variant != cur_var:
            if model is not None:
                del model
                torch.cuda.empty_cache()
            if variant:
                print(f"[model] build ar+{variant} (aspect-ratio encoder)")
                model = build_model_with_ar(args.device, variant)
            else:
                model = build_model_imagesonly(args.device)
            cur_var = variant
        if variant:
            load_variant_ckpt(model, path)
        else:
            load_pth(model, path)
        all_res, per_area = [], {}
        all_res_yflip, per_area_yflip = [], {}
        for area in areas:
            res_list, res_list_yflip = [], []
            for b in area_frames[area]:
                try:
                    pred, H, W = predict_pts3d_cam(model, area, b, args.device,
                                                   args.memory_efficient,
                                                   variant_inputs=bool(variant))
                    gt_cam, mask = build_gt_at(area, b, H, W, crop=not args.uncropped_gt)
                    m = eval_recon(pred, gt_cam, mask)
                    if m is not None:
                        res_list.append(m)
                    if args.compare_yflip:
                        pred_yflip = pred.copy()
                        pred_yflip[..., 1] *= -1
                        m_yflip = eval_recon(pred_yflip, gt_cam, mask)
                        if m_yflip is not None:
                            res_list_yflip.append(m_yflip)
                except Exception as e:
                    print(f"  [warn] {area} frame {b} failed: {e}")
                if args.device.startswith("cuda"):
                    torch.cuda.empty_cache()
            agg = ({k: float(np.mean([r[k] for r in res_list])) for k in res_list[0]}
                   if res_list else {})
            per_area[area] = {"n_frames": len(res_list), **agg}
            all_res.extend(res_list)
            print(f"  {name:8s} [{area}] n={len(res_list):3d}  "
                  f"chamferL1={agg.get('chamfer_L1', 0):.4f}  "
                  f"depth_rel={agg.get('depth_rel', 0):.4f}  d1={agg.get('delta1', 0):.3f}")
            if args.compare_yflip:
                agg_yflip = ({k: float(np.mean([r[k] for r in res_list_yflip]))
                              for k in res_list_yflip[0]}
                             if res_list_yflip else {})
                per_area_yflip[area] = {"n_frames": len(res_list_yflip),
                                        **agg_yflip}
                all_res_yflip.extend(res_list_yflip)
                print(f"  {name:8s} [{area}, yflip] n={len(res_list_yflip):3d}  "
                      f"chamferL1={agg_yflip.get('chamfer_L1', 0):.4f}  "
                      f"depth_rel={agg_yflip.get('depth_rel', 0):.4f}  "
                      f"d1={agg_yflip.get('delta1', 0):.3f}")
        comb = ({k: float(np.mean([r[k] for r in all_res])) for k in all_res[0]}
                if all_res else {})
        records[name] = {"variant": variant, "n_frames": len(all_res), **comb,
                         "per_area": per_area}
        if args.compare_yflip:
            comb_yflip = ({k: float(np.mean([r[k] for r in all_res_yflip]))
                           for k in all_res_yflip[0]}
                          if all_res_yflip else {})
            records_yflip[name] = {"variant": variant,
                                   "n_frames": len(all_res_yflip),
                                   **comb_yflip,
                                   "per_area": per_area_yflip}
        if len(areas) > 1:
            print(f"  {name:8s} [combined] n={len(all_res):3d}  "
                  f"chamferL1={comb.get('chamfer_L1', 0):.4f}  "
                  f"depth_rel={comb.get('depth_rel', 0):.4f}  d1={comb.get('delta1', 0):.3f}")

    out_json = args.out / "2d3ds_recon_v2_results.json"
    with open(out_json, "w") as f:
        json.dump({"area": areas[0] if len(areas) == 1 else None, "areas": areas,
                   "pipeline": "official load_images+model.infer; 2D3DS ERP "
                               "GT converted from the pack y-up frame to OpenCV y-down",
                   "gt": "whole panorama resized" if args.uncropped_gt else "cropped as the loader crops the input",
                   "records": records,
                   "records_yflip": records_yflip if args.compare_yflip else None},
                  f, indent=2)
    print(f"\n[out] {out_json}")


if __name__ == "__main__":
    main()
