"""
In-the-wild image inference with local .pth checkpoint(s).

Preprocessing: by default MapAnything's crop loader (every image is scaled to cover the aspect bucket
closest to the mean aspect ratio and centre-cropped); --squeeze resizes every image without cropping to
the fixed --squeeze-wh size. The evaluators resize every view without cropping to the bucket closest to
the mean aspect ratio of the tuple and route panoramas with the detector, e.g.
scripts/realset/predict_ours.py --input-resize squeeze --pano auto.

Loads N images from a folder, runs each ckpt through model.infer() in
image-only mode (no GT, no intrinsics, no poses), and writes:
  - pred_<name>_world.ply per ckpt: all views merged in the world frame, colored
  - input_rgb_grid.png: the model inputs side by side
  - summary.json

Usage (from the repository root):
    python scripts/infer_wild.py \
        --image-folder <image_dir> \
        --out <output_dir> \
        --ckpts MEOW=<checkpoint.pth>:wrap
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

THIS = Path(__file__).resolve()
ROOT = THIS.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(THIS.parent))  # sibling scripts (meow_model, eval_2d3ds_pose_v2, erp_detect)

from mapanything.models import init_model_from_config  # noqa: E402
from mapanything.utils.image import load_images  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers (PLY writer + image grid)
# ---------------------------------------------------------------------------
DINOV2_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
DINOV2_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def write_ply(path: Path, xyz: np.ndarray, rgb_u8: np.ndarray):
    assert xyz.shape == rgb_u8.shape and xyz.shape[1] == 3
    n = xyz.shape[0]
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    ).encode("ascii")
    dtype = np.dtype(
        [("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
         ("r", "u1"), ("g", "u1"), ("b", "u1")]
    )
    arr = np.empty(n, dtype=dtype)
    arr["x"], arr["y"], arr["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    arr["r"], arr["g"], arr["b"] = rgb_u8[:, 0], rgb_u8[:, 1], rgb_u8[:, 2]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(header)
        f.write(arr.tobytes())


def unnormalize_dinov2_view(img_b3hw: torch.Tensor) -> np.ndarray:
    x = img_b3hw.detach().cpu() * DINOV2_STD + DINOV2_MEAN
    x = x.clamp(0, 1).permute(0, 2, 3, 1).numpy()
    return (x * 255.0).round().astype(np.uint8)[0]


# ---------------------------------------------------------------------------
# Build model from train.yaml hydra config, image-only task, then load .pth
# ---------------------------------------------------------------------------
def build_model(device: str, ar: bool = False) -> torch.nn.Module:
    from mapanything.utils.hf_utils.hf_helpers import init_hydra_config
    from mapanything.models import init_model

    task_overrides = (
        ["model/task=aug_training", "++model.task.ar_prob=1.0"]
        if ar else ["model/task=images_only"]
    )
    cfg = init_hydra_config(
        "configs/train.yaml",
        overrides=[
            "machine=default",
            "model=mapanything",
            *task_overrides,
            "model.encoder.uses_torch_hub=true",
        ],
    )
    model = init_model(
        model_str=cfg.model.model_str,
        model_config=cfg.model.model_config,
        torch_hub_force_reload=False,
    )
    model = model.to(device).eval()
    return model


def load_pth(model: torch.nn.Module, ckpt_path: Path, strict: bool = True):
    print(f"  loading: {ckpt_path}")
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ck.get("model", ck)
    msg = model.load_state_dict(sd, strict=strict)
    print(f"  {msg}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-folder", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True,
                    help="Output dir for PLY + grid")
    ap.add_argument("--ckpts", nargs="+", required=True,
                    help="name=path[:variant] specs; a :wrap suffix loads the "
                         "aspect-ratio build (AR token + pano_wrap head), like "
                         "het_2d3ds_pose. Plain name=path stays images-only.")
    ap.add_argument("--pano-wrap", choices=["auto", "on", "off"], default="auto",
                    help="For :wrap ckpts only: per-view pano_wrap flag. "
                         "auto = blind full-ERP detector (erp_detect); "
                         "on/off = force all views.")
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--heterogeneous", action="store_true",
        help="Load each image independently so they keep their own (H,W) and aspect ratio. "
             "Default load_images forces all views to the same resolution.")
    ap.add_argument(
        "--center-square", action="store_true",
        help="Center-crop every input image to a square before loading. "
             "Useful when mixing fisheye / ERP / landscape / portrait — they all share ar=1.")
    ap.add_argument(
        "--crop-aspect-wh", type=str, default="",
        help="Center-crop every input to W:H aspect ratio (e.g. '2:1' for ERP-like). "
             "Mutually exclusive with --center-square; takes precedence if both set.")
    ap.add_argument("--memory-efficient", action="store_true",
                    help="Use memory_efficient_inference=True (slower, lower VRAM).")
    ap.add_argument("--ar", action="store_true",
                    help="Build model with ar_prob=1.0 (ar_encoder) and feed each "
                         "view its original aspect ratio so the aspect-ratio embedding is "
                         "active. Use for Stage-2 checkpoints; tolerant state-dict load.")
    ap.add_argument("--squeeze", action="store_true",
                    help="Full-field-of-view resizing: non-uniformly resize every image to the "
                         "fixed --squeeze-wh size (no crop, keeps the full FoV incl. 360-degree "
                         "panoramas). Opposite of load_images' equal-ratio "
                         "crop. Use with --ar for Stage-2 checkpoints.")
    ap.add_argument("--squeeze-wh", type=str, default="518x518",
                    help="Common target WxH for --squeeze (must be /14). Default 518x518.")
    ap.add_argument("--apply-confidence-mask", action="store_true",
                    help="Drop low-confidence pixels before PLY export.")
    ap.add_argument("--confidence-percentile", type=float, default=10.0)
    ap.add_argument(
        "--fisheye-mask-substr", type=str, default="",
        help="If non-empty, for every input image whose basename contains this "
             "substring, build a centered circular disk mask matching the model "
             "input resolution and AND it with pred mask before PLY export. "
             "Matches training-time fisheye disk mask (no predictions outside disk).")
    ap.add_argument(
        "--fisheye-disk-fraction", type=float, default=0.95,
        help="Disk radius as fraction of min(H,W)/2 for the fisheye mask. "
             "0.95 = small margin to drop the dark sensor edge.")
    ap.add_argument(
        "--erp-black-mask", action="store_true",
        help="Drop pure-black ERP pole pixels (e.g. 2D3DS panoramas fill the "
             "top/bottom poles with black) from the exported PLY. Detects black "
             "on the model input RGB and removes those points (handles the "
             "irregular wavy pole edge automatically).")
    ap.add_argument(
        "--erp-black-thresh", type=float, default=0.04,
        help="A pixel is 'black' if its summed RGB (0..1) is below this. "
             "0.04 matches the 2D3DS pole fill.")
    return ap.parse_args()


def parse_ckpts(items):
    """Parse NAME=path[:variant] specs. A suffix that is a VARIANT_OVERRIDES key
    (e.g. :wrap) selects build_model_with_ar and the strict load_variant_ckpt, same
    spec grammar as het_2d3ds_pose.py; entries without a suffix use the
    images-only build and load_pth."""
    from meow_model import VARIANT_OVERRIDES
    out = []
    for it in items:
        if "=" not in it:
            raise ValueError(f"Bad --ckpts entry: {it}")
        name, path = it.split("=", 1)
        variant = ""
        if ":" in path and path.rsplit(":", 1)[1] in VARIANT_OVERRIDES:
            path, variant = path.rsplit(":", 1)
        out.append((name, Path(path), variant))
    out.sort(key=lambda t: t[2])  # vanilla first, fewest model rebuilds
    return out


def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"[out] {args.out}")

    ckpts = parse_ckpts(args.ckpts)
    print("[ckpts]")
    for n, p, var in ckpts:
        print(f"  - {n}: {p}  exists={p.exists()}"
              + (f"  [variant:{var}]" if var else ""))

    print(f"[data] loading images from {args.image_folder}")
    image_folder = args.image_folder

    if args.crop_aspect_wh:
        w_r, h_r = [int(x) for x in args.crop_aspect_wh.split(":")]
        target = w_r / h_r
        sq_dir = args.out / f"img_crop_{w_r}x{h_r}"
        sq_dir.mkdir(parents=True, exist_ok=True)
        exts = (".jpg", ".jpeg", ".png", ".heic", ".heif")
        n = 0
        for p in sorted(args.image_folder.iterdir()):
            if p.suffix.lower() not in exts:
                continue
            im = Image.open(p).convert("RGB")
            w, h = im.size
            cur = w / h
            if cur > target:
                new_w = int(round(h * target)); left = (w - new_w) // 2
                cr = im.crop((left, 0, left + new_w, h)); ow, oh = new_w, h
            else:
                new_h = int(round(w / target)); top = (h - new_h) // 2
                cr = im.crop((0, top, w, top + new_h)); ow, oh = w, new_h
            cr.save(sq_dir / (p.stem + ".jpg"), quality=95)
            print(f"  crop {w_r}:{h_r}: {p.name}  {w}x{h} -> {ow}x{oh}")
            n += 1
        print(f"[data] cropped {n} images to {w_r}:{h_r} -> {sq_dir}")
        image_folder = sq_dir
    elif args.center_square:
        # Pre-process: center-crop every image to a square, saved to a folder under
        # --out, so the downstream loader sees uniform ar=1 inputs.
        sq_dir = args.out / "img_centersquare"
        sq_dir.mkdir(parents=True, exist_ok=True)
        exts = (".jpg", ".jpeg", ".png", ".heic", ".heif")
        n = 0
        for p in sorted(args.image_folder.iterdir()):
            if p.suffix.lower() not in exts:
                continue
            im = Image.open(p).convert("RGB")
            w, h = im.size
            side = min(w, h)
            left = (w - side) // 2
            top = (h - side) // 2
            sq = im.crop((left, top, left + side, top + side))
            sq.save(sq_dir / (p.stem + ".jpg"), quality=95)
            print(f"  center-square: {p.name}  {w}x{h} -> {side}x{side}")
            n += 1
        print(f"[data] center-squared {n} images -> {sq_dir}")
        image_folder = sq_dir

    if args.squeeze:
        sw, sh = [int(x) for x in args.squeeze_wh.lower().split("x")]
        exts = (".jpg", ".jpeg", ".png", ".heic", ".heif")
        img_paths = sorted(
            str(p) for p in image_folder.iterdir() if p.suffix.lower() in exts
        )
        print(f"[data] squeeze (preserve-info) mode: {len(img_paths)} images, "
              f"each non-uniformly resized to {sw}x{sh} (NO crop, full FoV kept)")
        views = []
        view_basenames = []
        for ip in img_paths:
            im = Image.open(ip).convert("RGB")
            ow, oh = im.size
            im_r = im.resize((sw, sh), Image.LANCZOS)        # squeeze (no crop)
            arr = np.asarray(im_r).astype(np.float32) / 255.0   # H,W,3
            t = torch.from_numpy(arr).permute(2, 0, 1)[None]    # 1,3,H,W
            t = (t - DINOV2_MEAN) / DINOV2_STD
            views.append(dict(
                img=t, true_shape=np.int32([[sh, sw]]),
                idx=len(views), instance=str(len(views)),
                data_norm_type=["dinov2"],
            ))
            view_basenames.append(os.path.basename(ip))
            print(f" - squeeze {os.path.basename(ip)}  {ow}x{oh} --> {sw}x{sh} "
                  f"(orig AR={ow / oh:.3f})")
    elif args.heterogeneous:
        exts = (".jpg", ".jpeg", ".png", ".heic", ".heif")
        img_paths = sorted(
            str(p) for p in image_folder.iterdir()
            if p.suffix.lower() in exts
        )
        print(f"[data] heterogeneous mode: {len(img_paths)} images, "
              f"each resized independently (fixed_mapping)")
        views = []
        view_basenames = []
        for ip in img_paths:
            v = load_images([ip], verbose=True)
            assert len(v) == 1
            v[0]["instance"] = str(len(views))
            v[0]["idx"] = len(views)
            views.append(v[0])
            view_basenames.append(os.path.basename(ip))
    else:
        views = load_images(str(image_folder), verbose=True)
        exts = (".jpg", ".jpeg", ".png", ".heic", ".heif")
        view_basenames = sorted(
            p.name for p in image_folder.iterdir() if p.suffix.lower() in exts
        )
        if len(view_basenames) != len(views):
            print(f"[warn] basename count {len(view_basenames)} != views {len(views)}, fisheye mask will be skipped")
            view_basenames = [""] * len(views)
    print(f"[data] loaded {len(views)} views")
    if len(views) == 0:
        raise RuntimeError(f"No images found in {args.image_folder}")

    # Move view tensors to device
    for v in views:
        for k, val in list(v.items()):
            if torch.is_tensor(val):
                v[k] = val.to(args.device, non_blocking=True)

    # Aspect-ratio input: attach each view's original (pre-resize) aspect ratio so
    # the aspect-ratio encoder can undo the squeeze. Uses the image_folder actually
    # loaded (post-crop if --center-square/--crop-aspect-wh was applied).
    any_variant = any(var for _, _, var in ckpts)
    ars = []
    if args.ar or any_variant:
        for bn in view_basenames:
            try:
                with Image.open(os.path.join(str(image_folder), bn)) as _im:
                    w, h = _im.size
                ars.append(float(w) / float(max(h, 1)))
            except Exception:
                ars.append(1.0)
    if args.ar:
        for v, a in zip(views, ars):
            v["aspect_ratio"] = torch.tensor([a], device=args.device, dtype=torch.float32)
        print(f"[ar] AR-conditioning ON; per-view aspect_ratio = "
              f"{[round(a, 3) for a in ars]}")

    # Inputs of variant (:wrap) checkpoints: pano_wrap per view. The wrap handles the
    # longitude seam of a full panorama, so it must fire only on full panoramas,
    # decided from the pixels by default (erp_detect: seam as an adjacent column pair
    # and pole convergence). Detected panoramas also get the definitional aspect
    # ratio 2.0 regardless of the delivered aspect.
    wrap_flags = [False] * len(views)
    if any_variant:
        if args.pano_wrap == "on":
            wrap_flags = [True] * len(views)
        elif args.pano_wrap == "auto":
            from erp_detect import detect_full_erp
            for i, bn in enumerate(view_basenames):
                try:
                    with Image.open(os.path.join(str(image_folder), bn)) as _im:
                        is_pano, _info = detect_full_erp(
                            np.asarray(_im.convert("RGB")))
                    wrap_flags[i] = bool(is_pano)
                except Exception:
                    wrap_flags[i] = False
        print(f"[wrap] pano_wrap mode={args.pano_wrap}; per-view flags = "
              f"{wrap_flags}")

    # Save RGB grid (using denormalized img). Heights may differ when
    # heterogeneous: pad each to the max height with zeros.
    rgbs = [unnormalize_dinov2_view(v["img"]) for v in views]
    max_h = max(r.shape[0] for r in rgbs)
    padded = []
    for r in rgbs:
        if r.shape[0] < max_h:
            pad = np.zeros((max_h - r.shape[0], r.shape[1], 3), dtype=r.dtype)
            r = np.concatenate([r, pad], axis=0)
        padded.append(r)
    grid = np.concatenate(padded, axis=1)
    grid_path = args.out / "input_rgb_grid.png"
    Image.fromarray(grid).save(grid_path)
    print(f"[viz] {grid_path}  shape={grid.shape}")

    summary = {"image_folder": str(args.image_folder),
               "n_views": len(views), "results": {}}

    model, cur_var = None, None
    for name, ckpt_path, variant in ckpts:
        print(f"\n=== {name} ===")
        if model is None or variant != cur_var:
            if model is not None:
                del model
                torch.cuda.empty_cache()
            if variant:
                print(f"[model] build ar+{variant} (aspect-ratio encoder) on {args.device}")
                from meow_model import build_model_with_ar
                model = build_model_with_ar(args.device, variant)
            else:
                print("[model] building MapAnything (images_only task) on",
                      args.device)
                model = build_model(args.device, ar=args.ar)
            cur_var = variant
        if variant:
            from eval_2d3ds_pose_v2 import load_variant_ckpt
            load_variant_ckpt(model, ckpt_path)
        else:
            load_pth(model, ckpt_path, strict=not args.ar)

        # Important: model.infer() mutates view dicts internally for some
        # ops; pass a shallow copy of each view dict to avoid cumulative
        # state across multiple ckpts.
        views_copy = [{k: v for k, v in vd.items()} for vd in views]
        if variant:
            # variant checkpoints are trained at ar_prob=1: the aspect-ratio input is
            # the content aspect ratio (2.0 on detected full ERPs); wrap only on panoramas
            for v, a, wf in zip(views_copy, ars, wrap_flags):
                v["aspect_ratio"] = torch.tensor(
                    [2.0 if wf else a], device=args.device,
                    dtype=torch.float32)
                v["pano_wrap"] = bool(wf)

        t0 = time.time()
        with torch.no_grad():
            preds = model.infer(
                views_copy,
                memory_efficient_inference=args.memory_efficient,
                minibatch_size=1 if args.memory_efficient else None,
                use_amp=True,
                amp_dtype="bf16",
                apply_mask=True,
                mask_edges=True,
                apply_confidence_mask=args.apply_confidence_mask,
                confidence_percentile=args.confidence_percentile,
            )
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        dt = time.time() - t0
        print(f"  infer: {dt:.2f} s")

        # Merge per-view world points + image colors into single PLY
        world_xyz_all = []
        world_rgb_all = []
        per_view_npts = []
        for vi, pred in enumerate(preds):
            pts3d = pred["pts3d"][0].detach().cpu().numpy()    # (H,W,3)
            mask = pred["mask"][0].squeeze(-1).detach().cpu().numpy().astype(bool)
            img = pred["img_no_norm"][0].detach().cpu().numpy()  # (H,W,3) in 0..1
            rgb_u8 = (np.clip(img, 0, 1) * 255).round().astype(np.uint8)

            # Apply fisheye disk mask if this view's basename matches the substr.
            bn = view_basenames[vi] if vi < len(view_basenames) else ""
            if args.fisheye_mask_substr and args.fisheye_mask_substr in bn:
                H, W = mask.shape
                cy, cx = H / 2.0, W / 2.0
                R = min(H, W) / 2.0 * args.fisheye_disk_fraction
                yy, xx = np.mgrid[:H, :W]
                disk = ((yy - cy) ** 2 + (xx - cx) ** 2) < R * R
                before = int(mask.sum())
                mask = mask & disk
                after = int(mask.sum())
                print(f"  [disk-mask] view{vi} {bn}  {before:,} -> {after:,} pts "
                      f"(R={R:.0f}/{min(H,W)//2}px)")

            # Apply ERP black-pole mask: drop pure-black pixels (2D3DS fills the
            # top/bottom poles with black; the edge is irregular/wavy so a
            # content-based test handles it without a fixed row cutoff).
            if args.erp_black_mask:
                black = (img.sum(axis=2) < args.erp_black_thresh)  # (H,W) on 0..1 input
                before = int(mask.sum())
                mask = mask & ~black
                after = int(mask.sum())
                print(f"  [erp-black] view{vi} {bn}  {before:,} -> {after:,} pts "
                      f"(dropped {before - after:,} black-pole px)")

            world_xyz_all.append(pts3d[mask])
            world_rgb_all.append(rgb_u8[mask])
            per_view_npts.append(int(mask.sum()))

        if len(world_xyz_all) == 0 or sum(per_view_npts) == 0:
            print("  WARNING: no valid points")
            continue

        xyz = np.concatenate(world_xyz_all, axis=0)
        rgb = np.concatenate(world_rgb_all, axis=0)
        ply_path = args.out / f"pred_{name}_world.ply"
        write_ply(ply_path, xyz, rgb)
        print(f"  [ply] {ply_path.name}  total_pts={xyz.shape[0]:,}  "
              f"per_view={per_view_npts}")

        summary["results"][name] = {
            "ckpt": str(ckpt_path),
            "variant": variant,
            "pano_wrap_flags": [bool(w) for w in wrap_flags] if variant else None,
            "infer_time_s": round(dt, 3),
            "total_pts": int(xyz.shape[0]),
            "per_view_pts": per_view_npts,
            "ply": str(ply_path.name),
        }

        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    summary_path = args.out / "summary.json"
    json.dump(summary, open(summary_path, "w"), indent=2)
    print(f"\n[summary] -> {summary_path}")


if __name__ == "__main__":
    main()
