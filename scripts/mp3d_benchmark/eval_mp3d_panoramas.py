#!/usr/bin/env python3
"""Matterport3D (MP3D) point-map runner on the 18 official test scans, following the
point-map evaluation of Wid3R (arXiv 2602.05321, Table 5).

Per scan: covisibility-sample K panoramas -> stitch each skybox to a 2:1 ERP RGB ->
model.infer -> merged predicted world point cloud + predicted camera centres. GT = the
union of the per-panorama ERP point maps of gt_erp_pointmap.py (perspective depth
back-projected with the OpenGL convention of the .conf poses, 10 m clip); GT camera
centres = panorama centres. Align pred->GT by Umeyama and a least-squares scale/shift
on the K camera-centre correspondences, then ICP on the dense clouds, then
Acc/Comp/N.C. (pointmap_eval). Aggregate over scans and print next to the Wid3R /
pi3 / VGGT numbers published by Wid3R.

Usage: python eval_mp3d_panoramas.py --ckpts MEOW=<path>:wrap MapAnything=<path> \
         [--k 8 --max-scans N --scans-root DIR --gt-cache DIR]

ckpt spec: NAME=path[:variant] with variant in {wrap, ar}. Checkpoints trained with
ar_prob=1 and pano_wrap_dpt=true (the aspect-ratio embedding and the panorama wrap,
e.g. the final MEOW model) need :wrap so the model is built with the matching config
and the checkpoint loads with no unexpected or missing keys. Inputs get aspect_ratio
and pano_wrap=True (every input is a full 360-degree ERP), matching training.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))  # for infer_wild
from cube2equirect import combine_views, SKYBOX_VX, SKYBOX_VY, SKYBOX_FOV
from parse_conf import parse_conf_from_zip, scan_dir
from gt_pointcloud import scene_pointcloud
from gt_erp_pointmap import erp_point_maps_for_scan
from covis_sample import sample_covis, pano_centers
from pointmap_eval import evaluate

from mapanything.utils.image import load_images  # noqa: E402

SCANS_ROOT = os.environ.get("MEOW_MP3D_ROOT", ".")
TEST_18 = ["2t7WUuJeko7", "5ZKStnWn8Zo", "ARNzJeq3xxb", "fzynW3qQPVF", "jtcxE69GiFV",
           "pa4otMbVnkk", "q9vSo1VnCiC", "rqfALeAoiTq", "UwV83HsGsw3", "wc2JMjhGNzB",
           "WYY7iVyf5p8", "YFuZgdQ5vWj", "yqstnuAEVhm", "YVUC4YcDtcY", "gxdoqLR6rwA",
           "gYvKGZ5eRqb", "RPmz2sHmrrY", "Vt2qJdWjCF2"]
# Published MP3D point-map results (Acc/Comp/N.C., mean) from Wid3R (arXiv 2602.05321),
# Table 5. Wid3R is trained on MP3D (in-domain).
WID3R = {"Wid3R": (0.094, 0.087, 0.790), "pi3": (0.315, 1.308, 0.539), "VGGT": (0.327, 1.756, 0.530)}
ERP_W, ERP_H = 2048, 1024


def stitch_erp(scan, uuid):
    import zipfile
    with zipfile.ZipFile(os.path.join(scan, "matterport_skybox_images.zip")) as z:
        imgs = []
        for i in range(6):
            cand = [n for n in z.namelist() if n.endswith(f"{uuid}_skybox{i}_sami.jpg")]
            if not cand:
                return None
            arr = cv2.imdecode(np.frombuffer(z.read(cand[0]), np.uint8), cv2.IMREAD_COLOR)
            imgs.append(arr.astype(np.float32) / 255.0)
    erp, _ = combine_views(imgs, SKYBOX_VX, SKYBOX_VY, [SKYBOX_FOV] * 6, ERP_W, ERP_H)
    return np.clip(erp * 255, 0, 255).astype(np.uint8)


def infer_cloud(model, paths, device):
    """Return (merged_world_pts Nx3, predicted camera centres Kx3).

    Injects per-view aspect_ratio (original W/H, the input of the aspect-ratio
    embedding, trained at ar_prob=1) and pano_wrap=True (full-360 ERP flag for
    the panorama-wrap dense head; ignored by models without it)."""
    from PIL import Image
    views = load_images(paths, verbose=False)
    for v, p in zip(views, paths):
        with Image.open(p) as im:
            w0, h0 = im.size
        v["aspect_ratio"] = torch.tensor([w0 / max(h0, 1)], dtype=torch.float32)
        v["pano_wrap"] = True
    for v in views:
        for k, val in list(v.items()):
            if torch.is_tensor(val):
                v[k] = val.to(device, non_blocking=True)
    with torch.no_grad():
        # infer() defaults to memory_efficient_inference=True; disable it
        # explicitly (the panorama-wrap dense head runs on the full batch).
        preds = model.infer(views, use_amp=True, amp_dtype="bf16",
                            apply_mask=True, mask_edges=True,
                            memory_efficient_inference=False)
    pts, cams = [], []
    for p in preds:
        xyz = p["pts3d"][0].detach().cpu().numpy()
        m = p["mask"][0].squeeze(-1).detach().cpu().numpy().astype(bool)
        pts.append(xyz[m])
        cams.append(p["camera_poses"][0].detach().cpu().numpy()[:3, 3])
    return np.concatenate(pts, 0).astype(np.float64), np.array(cams, np.float64)


def subsample(P, n=120000, rng=None):
    rng = rng or np.random.default_rng(0)
    return P if len(P) <= n else P[rng.choice(len(P), n, replace=False)]


def run_scan(model, scan_id, k, device, rng, dump_dir=None,
             scans_root=SCANS_ROOT, gt_cache=None, gt_stride=4):
    scan = os.path.join(scans_root, scan_id)
    panos = parse_conf_from_zip(scan_dir(scan))
    uuids, C = pano_centers(panos)
    sel = sample_covis(C, k, rng=rng)
    sel_uuids = [uuids[i] for i in sel]
    gt_centers = C[sel]
    with tempfile.TemporaryDirectory() as td:
        paths = []
        kept = []
        for j, u in enumerate(sel_uuids):
            erp = stitch_erp(scan, u)
            if erp is None:
                continue
            p = os.path.join(td, f"{j:02d}.png")
            cv2.imwrite(p, erp)
            if dump_dir:
                os.makedirs(dump_dir, exist_ok=True)
                cv2.imwrite(os.path.join(dump_dir, f"{scan_id}_in{j:02d}.png"),
                            cv2.resize(erp, (768, 384)))
            paths.append(p)
            kept.append(j)
        if len(paths) < 2:
            return None
        pred_pts, pred_cams = infer_cloud(model, paths, device)
    gt_centers = gt_centers[kept]
    # GT = union of per-pano ERP point maps (ERP coverage, 10m clip — Wid3R-style).
    # One conf-parse + one zip-open per scan; npz cache short-circuits everything.
    gt_list = []
    for _u, P, valid in erp_point_maps_for_scan(
            scan, [sel_uuids[j] for j in kept], erp_w=1024, erp_h=512,
            stride=gt_stride, cache_dir=gt_cache):
        gt_list.append(P[valid])
    gt_pts = np.concatenate(gt_list, 0)
    # align (Umeyama + least-squares scale/shift on the camera-centre correspondences
    # -> dense ICP) + Acc/Comp/N.C.
    pred_s = subsample(pred_pts, rng=rng)
    gt_s = subsample(gt_pts, rng=rng)
    corr_ok = len(pred_cams) == len(gt_centers) and len(pred_cams) >= 3
    m = evaluate(pred_s, gt_s,
                 pred_corr=pred_cams if corr_ok else None,
                 gt_corr=gt_centers if corr_ok else None)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True,
                    help="NAME=path[:wrap|:ar] ...")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--max-scans", type=int, default=18)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="Optional per-scan JSON result")
    ap.add_argument("--scans-root", default=SCANS_ROOT,
                    help="MP3D scans dir, one sub-directory per scan "
                         "(default: env MEOW_MP3D_ROOT)")
    ap.add_argument("--gt-cache", default=None,
                    help="npz cache dir from precompute_erp_gt.py (strongly recommended)")
    ap.add_argument("--gt-stride", type=int, default=4,
                    help="source depth subsample for GT (GT lands on 1024x512 ERP; 4 is lossless in effect)")
    args = ap.parse_args()

    from meow_model import VARIANT_OVERRIDES, build_model_with_ar, load_ckpt_inplace
    from infer_wild import build_model, load_pth

    # parse NAME=path[:variant]; group execution by variant so the model is
    # rebuilt only when the architecture flag actually changes
    specs = []
    for spec in args.ckpts:
        name, path = spec.split("=", 1)
        variant = ""
        if ":" in path and path.rsplit(":", 1)[1] in VARIANT_OVERRIDES:
            path, variant = path.rsplit(":", 1)
        specs.append((name, path, variant))
    specs.sort(key=lambda t: t[2])

    model, current_variant = None, None
    scans = TEST_18[: args.max_scans]
    for name, path, variant in specs:
        if model is None or variant != current_variant:
            if model is not None:
                del model
                torch.cuda.empty_cache()
            if variant:
                print(f"[model] build ar+{variant} (config: ar_prob=1, {VARIANT_OVERRIDES[variant]})")
                model = build_model_with_ar(args.device, variant)
            else:
                print("[model] build vanilla (images_only)")
                model = build_model(args.device)
            current_variant = variant
        if variant:
            ck = torch.load(path, map_location="cpu", weights_only=False)
            sd = ck.get("model", ck)
            n_fresh = load_ckpt_inplace(model, sd)   # zero unexpected; only ar_encoder may be fresh
            print(f"  [{name}] loaded ({'strict-equivalent' if n_fresh == 0 else f'{n_fresh} ar_encoder keys fresh'})")
            del ck, sd
        else:
            load_pth(model, path)
        rows = []
        records, failures = [], []
        for sid in scans:
            try:
                m = run_scan(model, sid, args.k, args.device, np.random.default_rng(args.seed),
                             scans_root=args.scans_root, gt_cache=args.gt_cache,
                             gt_stride=args.gt_stride)
            except Exception as e:
                print(f"  [{sid}] FAIL {e}")
                failures.append(dict(scan=sid, error=str(e)))
                continue
            if m:
                rows.append(m)
                records.append(dict(scan=sid, metrics=m))
                print(f"  [{sid}] Acc={m['Acc_mean']:.3f}/{m['Acc_med']:.3f} "
                      f"Comp={m['Comp_mean']:.3f}/{m['Comp_med']:.3f} NC={m['NC_mean']:.3f}/{m['NC_med']:.3f}")
        if rows:
            agg = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
            print(f"\n=== {name} (mean over {len(rows)} scans, K={args.k}) ===")
            print(f"  Acc  mean={agg['Acc_mean']:.3f} med={agg['Acc_med']:.3f}")
            print(f"  Comp mean={agg['Comp_mean']:.3f} med={agg['Comp_med']:.3f}")
            print(f"  N.C. mean={agg['NC_mean']:.3f} med={agg['NC_med']:.3f}")
        if args.out:
            from pathlib import Path
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(json.dumps(dict(model=name, config=vars(args),
                records=records, failures=failures,
                aggregate=agg if rows else None), indent=2))
    print("\n--- Wid3R paper (MP3D Table5, Acc/Comp/N.C. mean) ---")
    for n, (a, c, nc) in WID3R.items():
        print(f"  {n:6s} Acc={a:.3f} Comp={c:.3f} N.C.={nc:.3f}")


if __name__ == "__main__":
    main()
