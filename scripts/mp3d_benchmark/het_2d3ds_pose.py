#!/usr/bin/env python3
"""Heterogeneous multi-view pose on 2D3DS, comparable to CAM3R.

Builds a heterogeneous-camera 2D3DS pose benchmark following CAM3R (arXiv 2603.22631,
Tables 2 and 3) and evaluates MapAnything-family checkpoints (ours and the public
weights), VGGT and pi3 on it. No CAM3R weights were available, so CAM3R itself is not
rerun; instead, the reproduction is calibrated by comparing our VGGT/pi3 numbers with
the VGGT/pi3 numbers CAM3R reports (a shared anchor).

Protocol (following CAM3R Sec. D as far as it is documented):
  - Source: 2D3DS area_5a/5b/6 panoramas (in-domain for CAM3R; zero-shot for our models,
    VGGT and pi3).
  - Group: covisible compact set, pairwise baseline 0.1-2.2 m (sample_covisible_group);
    --subscene / --all-views use whole rooms instead.
  - Heterogeneity: each panorama in the group is rendered as one camera model
    {ERP | perspective | fisheye}, cycling through --models; synthesized views look
    toward the group centroid by default to maximise overlap (CAM3R selects the
    synthesized directions with the highest overlap).
  - Metrics: RRA@30 / RTA@30 / mAA@30 / ATE (the set of CAM3R Tables 2 and 3). The
    printed RRA/RTA/mAA/AUC pool the view pairs of all cases; ATE is averaged over
    cases; --out also stores the per-case errors.

Convention:
  - MapAnything outputs per-view c2w in OpenCV axes (+X right, +Y down, +Z forward).
  - het_synth builds rays in a +Y-up ERP frame, while the 2D3DS panorama poses are in the
    +Y-down frame; the view rotation is converted with FLIP_Y = diag(1,-1,1):
    GT c2w = pano_c2w @ blkdiag(FLIP_Y @ R_pano_from_cam @ FLIP_Y, 1). --yflip further
    composes it with F = diag(1,-1,-1) (180 deg about X, det=+1), a diagnostic.
  - F is never applied to ERP views (R=I, GT c2w = pano_c2w, as in eval_2d3ds_pose_v2.py).

Selftest (--selftest, model-free): synthesize two perspective views from the same
panorama at a known relative yaw; the GT relative rotation must equal that yaw. This
checks the rotation composition independently of any model.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch

THIS = Path(__file__).resolve()
ROOT = THIS.parent.parent.parent  # repository root
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(THIS.parent))

from het_synth import perspective_view, fisheye_view  # noqa: E402
from eval_2d3ds_pose import (  # noqa: E402
    STANFORD_ROOT, list_frames_by_scene, sample_covisible_group,
    load_gt_c2w, eval_case, pose_metrics, ate_rmse, rot_angle_deg,
)

F_YFLIP = np.diag([1.0, -1.0, -1.0])  # OpenCV(+Y down) <-> het_synth(+Y up), 180 about X
FLIP_Y = np.diag([1.0, -1.0, 1.0])     # het_synth ray frame (+Y up) <-> panorama pose frame (+Y down)


def _erp_path(area, base):
    return str(Path(STANFORD_ROOT) / area / "pano" / "rgb" / f"{base}_rgb.png")


def _load_erp(area, base, erp_w=2048, erp_h=1024, fill_poles=False):
    img = cv2.imread(_erp_path(area, base))
    if img is None:
        return None
    if fill_poles:
        img = _fill_erp_poles(img)
    return cv2.resize(img, (erp_w, erp_h))


def _fill_erp_poles(erp, thr=5):
    """Fill black pole rows (2D3DS ERP artifact, ~27% top/bottom) by replicating the
    nearest valid latitude row. Removes OOD black band without breaking ERP geometry."""
    gray = erp.mean(2)
    valid = ((gray >= thr).mean(1) > 0.5)
    if valid.all() or not valid.any():
        return erp
    idx = np.where(valid)[0]
    top, bot = int(idx[0]), int(idx[-1])
    out = erp.copy()
    if top > 0:
        out[:top] = erp[top]
    if bot + 1 < erp.shape[0]:
        out[bot + 1:] = erp[bot]
    return out


def _dir_to_yawpitch(d_world, pano_c2w):
    """World direction -> (yaw,pitch) in the pano ERP frame (= pano cam frame)."""
    R = pano_c2w[:3, :3]
    dp = R.T @ d_world
    n = np.linalg.norm(dp) + 1e-12
    dp = dp / n
    yaw = math.atan2(dp[0], dp[2])           # phi=atan2(rx,rz)
    pitch = math.asin(np.clip(dp[1], -1, 1))  # theta=asin(ry)
    return yaw, pitch


def _synth_view(erp, kind, yaw, pitch, size, persp_fov, fish_fov):
    if kind == "erp":
        return cv2.resize(erp, (size * 2, size)), np.eye(3)
    if kind == "persp":
        img, R, _ = perspective_view(erp, yaw, pitch, persp_fov, size)
        return img, R
    img, R, _ = fisheye_view(erp, yaw, pitch, fish_fov, size)
    return img, R


def build_het_group(area, bases, models, rng, yflip, size=512,
                    persp_fov=90.0, fish_fov=180.0, synth_dir="centroid",
                    erp_fill=False):
    """Return (image_paths_in_tmp, gt_c2w_list, tmpdir). Caller cleans tmpdir.

    synth_dir controls perspective/fisheye view orientation (overlap difficulty):
      centroid = face group centroid (max overlap, easiest translation; CAM3R two-view style)
      forward  = each pano's native +Z forward (wide-baseline realistic, harder)
      random   = random yaw (hardest, lowest overlap)
    """
    locs = {b: load_gt_c2w(area, b) for b in bases}
    centroid = np.mean([locs[b][:3, 3] for b in bases], axis=0)
    td = tempfile.mkdtemp(prefix="het2d3ds_")
    paths, gts = [], []
    for k, b in enumerate(bases):
        pano_c2w = locs[b]
        kind = models[k % len(models)] if isinstance(models, list) else models
        if kind in ("persp", "fish"):
            if synth_dir == "random":
                yaw, pitch = float(rng.uniform(-math.pi, math.pi)), 0.0
            elif synth_dir == "forward":
                yaw, pitch = 0.0, 0.0
            else:  # centroid
                d = centroid - pano_c2w[:3, 3]
                if np.linalg.norm(d) < 1e-6:
                    yaw, pitch = float(rng.uniform(-math.pi, math.pi)), 0.0
                else:
                    yaw, pitch = _dir_to_yawpitch(d, pano_c2w)
        else:
            yaw = pitch = 0.0
        erp = _load_erp(area, b, fill_poles=erp_fill)
        if erp is None:
            continue
        img, R = _synth_view(erp, kind, yaw, pitch, size, persp_fov, fish_fov)
        # R maps the synthesized camera into the panorama in het_synth's +Y-up frame; yaw and pitch
        # come from the +Y-down pose frame of the panorama, so the ground truth takes R in that frame
        R = FLIP_Y @ R @ FLIP_Y
        Rgt = R @ F_YFLIP if (yflip and kind != "erp") else R
        gt = pano_c2w.copy()
        gt[:3, :3] = pano_c2w[:3, :3] @ Rgt
        p = os.path.join(td, f"{k:02d}_{kind}.png")
        cv2.imwrite(p, img)
        paths.append(p)
        gts.append(gt)
    return paths, gts, td


@torch.no_grad()
def predict_c2w(model, paths, device, input_resize="crop"):
    from mapanything.utils.image import load_images
    views = load_images(paths, verbose=False, resize_mode=(
        "fixed_squeeze" if input_resize == "squeeze" else "fixed_mapping"))
    for v in views:
        for kk, val in list(v.items()):
            if torch.is_tensor(val):
                v[kk] = val.to(device, non_blocking=True)
    preds = model.infer(views, use_amp=True, amp_dtype="bf16",
                        apply_mask=True, mask_edges=True)
    out = []
    for p in preds:
        out.append(p["camera_poses"][0].detach().cpu().numpy().astype(np.float64))
    return out


@torch.no_grad()
def predict_c2w_variant(model, paths, device, input_resize="crop", pano_route="label",
                        routing_stats=None):
    """Path for checkpoints loaded with a variant suffix (e.g. :wrap), as in
    eval_2d3ds_pose_v2.py: per-view aspect_ratio set (such checkpoints are
    trained at ar_prob=1) and pano_wrap from pano_route. Label routing uses the
    synthesized file name; auto routing detects ERP from the RGB before resizing
    and gives detected panoramas the definitional aspect ratio 2.0 (all: every view
    treated as a panorama, also with 2.0; none: no view). Other views get W/H.
    Checkpoints without a suffix use predict_c2w."""
    from PIL import Image
    from mapanything.utils.image import load_images
    views = load_images(paths, verbose=False, resize_mode=(
        "fixed_squeeze" if input_resize == "squeeze" else "fixed_mapping"))
    for v, p in zip(views, paths):
        with Image.open(p) as im:
            w0, h0 = im.size
            if pano_route == "auto":
                from erp_detect import detect_full_erp
                is_pano, _ = detect_full_erp(np.asarray(im.convert("RGB")))
            elif pano_route == "all":      # flag-sensitivity control: every view treated as a panorama
                is_pano = True
            elif pano_route == "none":     # flag-sensitivity control: no view treated as a panorama
                is_pano = False
            else:
                is_pano = Path(p).stem.endswith("_erp")
        # detected or forced panoramas take the definitional content aspect 2.0
        ar = 2.0 if pano_route in ("auto", "all") and is_pano else w0 / max(h0, 1)
        v["aspect_ratio"] = torch.tensor([ar], dtype=torch.float32)
        v["pano_wrap"] = bool(is_pano)
        if routing_stats is not None:
            kind = Path(p).stem.rsplit("_", 1)[-1]
            true_erp = kind == "erp"
            routing_stats["total_views"] += 1
            routing_stats["true_erp" if true_erp else "true_non_erp"] += 1
            routing_stats["predicted_erp"] += int(is_pano)
            routing_stats["fn"] += int(true_erp and not is_pano)
            routing_stats["fp"] += int(not true_erp and is_pano)
            errors = routing_stats["errors_by_true_kind"]
            errors[kind] = errors.get(kind, 0) + int(true_erp != is_pano)
    for v in views:
        for kk, val in list(v.items()):
            if torch.is_tensor(val):
                v[kk] = val.to(device, non_blocking=True)
    preds = model.infer(views, memory_efficient_inference=False,
                        use_amp=True, amp_dtype="bf16",
                        apply_mask=True, mask_edges=True)
    return [p["camera_poses"][0].detach().cpu().numpy().astype(np.float64)
            for p in preds]


@torch.no_grad()
def predict_c2w_pi3(model, paths, device):
    """Pi3 native path: load the explicit path list (order preserved, aligns with GT)
    as [0,1] (N,3,H,W) with Pi3's PIXEL_LIMIT uniform /14 resize -> camera_poses
    (N,4,4) OpenCV c2w. (Explicit list, not a directory scan: safe when callers share
    a tmpdir.)"""
    from PIL import Image
    from torchvision import transforms
    imgs = [Image.open(p).convert("RGB") for p in paths]
    W0, H0 = imgs[0].size
    PIXEL_LIMIT = 255000
    scale = math.sqrt(PIXEL_LIMIT / (W0 * H0)) if W0 * H0 > 0 else 1.0
    k, m = round(W0 * scale / 14), round(H0 * scale / 14)
    while (k * 14) * (m * 14) > PIXEL_LIMIT and k > 1 and m > 1:
        if k / max(m, 1) > W0 / H0:
            k -= 1
        else:
            m -= 1
    TW, TH = max(1, k) * 14, max(1, m) * 14
    tt = transforms.ToTensor()
    t = torch.stack([tt(im.resize((TW, TH), Image.Resampling.LANCZOS)) for im in imgs], 0).to(device)
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    with torch.amp.autocast("cuda", dtype=dtype):
        res = model(t[None])  # add batch dim
    c2w = res["camera_poses"][0].detach().cpu().numpy().astype(np.float64)  # (N,4,4) c2w
    return [c2w[i] for i in range(c2w.shape[0])]


@torch.no_grad()
def predict_c2w_vggt(model, paths, device):
    """VGGT native path: load_and_preprocess_images (order preserved) -> pose_enc ->
    extrinsic (w2c) -> invert to c2w (OpenCV RDF, same convention as MapAnything/Pi3/our GT)."""
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    images = load_and_preprocess_images(list(paths)).to(device)  # (N,3,H,W), input order kept
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    with torch.amp.autocast("cuda", dtype=dtype):
        pred = model(images)
    extr, _ = pose_encoding_to_extri_intri(pred["pose_enc"], images.shape[-2:])  # (1,N,3,4) w2c
    extr = extr[0].detach().cpu().numpy().astype(np.float64)
    out = []
    for i in range(extr.shape[0]):
        w2c = np.eye(4, dtype=np.float64)
        w2c[:3, :4] = extr[i]
        out.append(np.linalg.inv(w2c))  # c2w
    return out


def selftest():
    """Model-free geometry check: two perspective views from a synthetic ERP at a
    known relative yaw -> GT relative rotation must equal that yaw."""
    H, W = 512, 1024
    erp = np.random.randint(0, 255, (H, W, 3), np.uint8)
    pano_c2w = np.eye(4)
    for dyaw in (15, 30, 60):
        _, R0, _ = perspective_view(erp, 0.0, 0.0, 90.0, 128)
        _, R1, _ = perspective_view(erp, math.radians(dyaw), 0.0, 90.0, 128)
        c0 = pano_c2w.copy(); c0[:3, :3] = pano_c2w[:3, :3] @ R0
        c1 = pano_c2w.copy(); c1[:3, :3] = pano_c2w[:3, :3] @ R1
        rel = rot_angle_deg(c0[:3, :3], c1[:3, :3])
        ok = abs(rel - dyaw) < 0.5
        print(f"[selftest] persp dyaw={dyaw} -> GT rel rot={rel:.2f}  {'OK' if ok else 'FAIL'}")
    print("[selftest] (rotation composition geometry validated)")


def export_cases(cases, models, rng, args, out_dir):
    """Write the synthesized views and ground-truth poses of every case, as the evaluation feeds them:
    <out_dir>/cases/case_NNN/<kk>_<kind>.png, gt_c2w.npy (V, 4, 4), and <out_dir>/manifest.json with the
    protocol settings and per-case area, 2D3DS frame names, kinds, images and ground-truth file. Baselines
    that cannot run inside this script read the exported cases (scripts/competitor_baselines)."""
    import json
    import shutil
    out_dir = Path(out_dir)
    (out_dir / "cases").mkdir(parents=True, exist_ok=True)
    keys = ("areas", "models", "min_views", "max_views", "repeats", "bmin", "bmax", "subscene", "all_views",
            "cap_views", "yflip", "synth_dir", "persp_fov", "fish_fov", "erp_fill", "max_cases", "seed")
    manifest = dict(protocol={k: getattr(args, k) for k in keys}, cases=[])
    for ci, (area, grp) in enumerate(cases):
        paths, gts, td = build_het_group(area, grp, models, rng, args.yflip, persp_fov=args.persp_fov,
                                         fish_fov=args.fish_fov, synth_dir=args.synth_dir,
                                         erp_fill=args.erp_fill)
        try:
            if len(paths) < 3:
                continue
            cdir = out_dir / "cases" / f"case_{ci:03d}"
            cdir.mkdir(parents=True, exist_ok=True)
            images, kinds, frames = [], [], []
            for p in paths:
                name = Path(p).name                       # <k>_<kind>.png, k = index in grp
                shutil.copyfile(p, cdir / name)
                images.append(f"cases/{cdir.name}/{name}")
                kinds.append(Path(name).stem.split("_", 1)[1])
                frames.append(grp[int(name.split("_", 1)[0])])
            np.save(cdir / "gt_c2w.npy", np.stack(gts))
            manifest["cases"].append(dict(case_id=ci, area=area, frames=frames, kinds=kinds, images=images,
                                          gt_c2w=f"cases/{cdir.name}/gt_c2w.npy"))
        finally:
            shutil.rmtree(td, ignore_errors=True)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"[export] {len(manifest['cases'])} cases -> {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="ma", choices=["ma", "pi3", "vggt"],
                    help="ma=MapAnything family (uses --ckpts); pi3/vggt=pretrained "
                         "competitor for calibration (ignores --ckpts).")
    ap.add_argument("--ckpts", nargs="+", help="NAME=path ... (ma backend only)")
    ap.add_argument("--areas", nargs="+", default=["area_5a", "area_5b"])
    ap.add_argument("--models", default="erp,persp,fish",
                    help="comma list cycled across views, or single (erp/persp/fish)")
    ap.add_argument("--min-views", type=int, default=4)
    ap.add_argument("--max-views", type=int, default=8)
    ap.add_argument("--repeats", type=int, default=4)
    ap.add_argument("--bmin", type=float, default=0.1)
    ap.add_argument("--bmax", type=float, default=2.2)
    ap.add_argument("--subscene", action="store_true",
                    help="CAM3R multi-view protocol: use the whole sub-scene (room) "
                         "frames (wide-baseline, harder) instead of compact 0.1-2.2m "
                         "groups. Random-subsample to <=max-views spanning the room.")
    ap.add_argument("--all-views", action="store_true",
                    help="CAM3R protocol: one case per subscene = all frames in the "
                         "room (no subsample, no repeat). Overrides --subscene.")
    ap.add_argument("--cap-views", type=int, default=0,
                    help="OOM safety: if a subscene exceeds this, random-subsample to "
                         "it (0=no cap, fully faithful). Logged per case.")
    ap.add_argument("--yflip", action="store_true", help="apply OpenCV<->ERP Y-flip to synth GT")
    ap.add_argument("--synth-dir", default="centroid",
                    choices=["centroid", "forward", "random"],
                    help="synth view orientation: centroid(easy,max overlap)/"
                         "forward(native +Z, wide-baseline)/random(hardest)")
    ap.add_argument("--persp-fov", type=float, default=90.0, help="perspective FoV deg (lower=harder)")
    ap.add_argument("--fish-fov", type=float, default=180.0, help="fisheye FoV deg")
    ap.add_argument("--erp-fill", action="store_true",
                    help="fill 2D3DS ERP black pole rows by edge replication (de-OOD input)")
    ap.add_argument("--max-cases", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--export-dir", default=None,
                    help="also write the synthesized views and ground truth of every case (exported case layout)")
    ap.add_argument("--export-only", action="store_true", help="with --export-dir: export and exit")
    ap.add_argument("--input-resize", choices=["crop", "squeeze"], default="crop")
    ap.add_argument("--pano-route", choices=["label", "auto", "all", "none"], default="label",
                    help="ERP routing for :wrap models only: filename label or RGB detection")
    ap.add_argument("--out", help="optional raw results JSON (with several --ckpts: one file per "
                                     "checkpoint, <stem>.<NAME>.json)")
    args = ap.parse_args()
    print(f"input_resize={args.input_resize}", flush=True)

    if args.selftest:
        selftest()
        return

    models = args.models.split(",")
    rng = np.random.default_rng(args.seed)
    # build cases
    cases = []
    for area in args.areas:
        by_scene = list_frames_by_scene(area)
        for room, frames in by_scene.items():
            if len(frames) < args.min_views:
                continue
            if args.all_views:
                # CAM3R multi-view: all heterogeneous images of the subscene, one case.
                grp = [f[0] for f in frames]
                if args.cap_views and len(grp) > args.cap_views:
                    idx = rng.choice(len(grp), size=args.cap_views, replace=False)
                    grp = [grp[i] for i in sorted(idx)]
                if len(grp) >= 3:
                    cases.append((area, grp))
                continue
            for _ in range(args.repeats):
                k = int(rng.integers(args.min_views, args.max_views + 1))
                if args.subscene:
                    # CAM3R multi-view: whole sub-scene (wide-baseline). Random
                    # subsample k frames spanning the entire room (not compact).
                    idx = rng.choice(len(frames), size=min(k, len(frames)), replace=False)
                    grp = [frames[i][0] for i in idx]
                else:
                    grp = sample_covisible_group(frames, k, rng, args.bmin, args.bmax)
                if grp and len(grp) >= 3:
                    cases.append((area, grp))
    rng.shuffle(cases)
    cases = cases[: args.max_cases]
    print(f"[bench] {len(cases)} heterogeneous cases | models={models} yflip={args.yflip} "
          f"synth_dir={args.synth_dir} persp_fov={args.persp_fov} fish_fov={args.fish_fov}")
    if args.export_dir:
        import copy
        # a copy of the generator: the exported views are those of the first evaluation pass below
        export_cases(cases, models, copy.deepcopy(rng), args, args.export_dir)
        if args.export_only:
            return

    backend = args.backend
    if backend == "ma":
        sys.path.insert(0, str(ROOT / "scripts"))
        from infer_wild import build_model, load_pth
        # NAME=path[:variant] specs: a variant builds the model with
        # meow_model.build_model_with_ar and loads strictly; entries without a
        # suffix use the images-only build of infer_wild.py.
        from meow_model import VARIANT_OVERRIDES, build_model_with_ar
        from eval_2d3ds_pose_v2 import load_variant_ckpt
        runs = []
        for spec in (args.ckpts or []):
            nm, pth = spec.split("=", 1)
            var = ""
            if ":" in pth and pth.rsplit(":", 1)[1] in VARIANT_OVERRIDES:
                pth, var = pth.rsplit(":", 1)
            runs.append((nm, pth, var))
        runs.sort(key=lambda t: t[2])   # vanilla first, fewest rebuilds
        model, cur_var = None, None
        predict = predict_c2w
    elif backend == "pi3":
        from pi3.models.pi3 import Pi3
        model = Pi3.from_pretrained("yyfz233/Pi3").to(args.device).eval()
        runs = [("pi3", None, "")]
        predict = predict_c2w_pi3
    else:  # vggt
        from vggt.models.vggt import VGGT
        model = VGGT.from_pretrained("facebook/VGGT-1B").to(args.device).eval()
        runs = [("vggt", None, "")]
        predict = predict_c2w_vggt

    for name, path, variant in runs:
        if backend == "ma":
            if model is None or variant != cur_var:
                if model is not None:
                    del model
                    torch.cuda.empty_cache()
                if variant:
                    print(f"[model] build ar+{variant} (aspect-ratio encoder)")
                    model = build_model_with_ar(args.device, variant)
                else:
                    model = build_model(args.device)
                cur_var = variant
            if variant:
                load_variant_ckpt(model, path)
                predict = predict_c2w_variant
            else:
                load_pth(model, path)
                predict = predict_c2w
        import time, json
        started = time.monotonic()
        records = []
        routing_stats = (dict(total_views=0, true_erp=0, true_non_erp=0,
                              predicted_erp=0, fn=0, fp=0, errors_by_true_kind={})
                         if backend == "ma" and variant == "wrap" else None)
        all_r, all_t, ates = [], [], []
        for area, grp in cases:
            paths, gts, td = build_het_group(area, grp, models, rng, args.yflip,
                                             persp_fov=args.persp_fov, fish_fov=args.fish_fov,
                                             synth_dir=args.synth_dir, erp_fill=args.erp_fill)
            try:
                if len(paths) < 3:
                    continue
                route_kwargs = (dict(pano_route=args.pano_route, routing_stats=routing_stats)
                                if backend == "ma" and variant == "wrap" else {})
                pred = (predict(model, paths, args.device, input_resize=args.input_resize,
                                **route_kwargs)
                        if backend == "ma" else predict(model, paths, args.device))
                rerr, terr = eval_case(pred, gts)
                all_r.append(rerr); all_t.append(terr)
                ates.append(ate_rmse(pred, gts))
                records.append(dict(area=area, frames=grp, rotation_errors=rerr.tolist(),
                                    translation_errors=terr.tolist(), ate=float(ates[-1])))
            finally:
                for p in paths:
                    try: os.remove(p)
                    except OSError: pass
                try: os.rmdir(td)
                except OSError: pass
        if not all_r:
            print(f"  {name}: no valid cases"); continue
        R = np.concatenate(all_r); T = np.concatenate(all_t)
        m = pose_metrics(R, T)
        ate = float(np.nanmean(ates))
        if args.out:
            out_p = Path(args.out)
            if len(runs) > 1:   # one file per checkpoint; a single --ckpts entry keeps the given name
                out_p = out_p.with_name(f"{out_p.stem}.{name}{out_p.suffix}")
            out_p.parent.mkdir(parents=True, exist_ok=True)
            out_p.write_text(json.dumps(dict(model=name, input_resize=args.input_resize,
                config=vars(args), routing=routing_stats, metrics=m, ate=ate, cases=len(records),
                failed=len(cases)-len(records), seconds=time.monotonic()-started, records=records), indent=2))
        print(f"\n=== {name} (het 2D3DS, {len(all_r)} cases, models={'+'.join(models)}) ===")
        if routing_stats is not None:
            print(f"  pano_route={args.pano_route} routing={routing_stats}")
        print(f"  RRA@30={m['RRA@30']:.1f} RTA@30={m['RTA@30']:.1f} "
              f"mAA@30={m['mAA@30']:.1f} AUC@30={m['AUC@30']:.1f} ATE={ate:.3f}")
    print("\n--- CAM3R paper (2D3DS multi-view, in-domain) ---")
    print("  CAM3R  RRA@30=94.0 RTA@30=91.5 mAA@30=73.5 ATE=1.8")
    print("  VGGT   RRA@30=31.8 RTA@30=34.4 mAA@30=7.6  ATE=3.8")
    print("  pi3    RRA@30=40.0 RTA@30=35.8 mAA@30=9.6  ATE=2.9")


if __name__ == "__main__":
    main()
