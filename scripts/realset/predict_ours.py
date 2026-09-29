#!/usr/bin/env python3
"""MEOW predictor for the realset harness (mapanything env).

Same pipeline as eval_2d3ds_pose_v2 / eval_mp3d_panoramas: load_images
(fixed_mapping) -> model.infer -> camera_poses + pts3d(+conf). The model
builders and strict checkpoint loaders are imported from eval_2d3ds_pose_v2
and meow_model, so variant handling (:ar = ar_prob=1; :wrap = ar_prob=1 +
pano_wrap_dpt) is identical to the other evaluation scripts.

--pano controls the pano_wrap flag fed to infer: on for equirectangular
inputs (2D3DS panoramas), off for perspective/fisheye (ADT, Replica),
matching how the model was trained (the panorama wrap handles the
equirectangular seam; it is not a generic switch).

Output: preds/<model>/<tuple_id>.npz per the common.py contract; with the
default --input-resize crop, uv maps the processed grid back to original
pixels via the scale+center-crop composition of fixed_mapping
(common.resize_crop_uv_map).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

THIS = Path(__file__).resolve()
sys.path.insert(0, str(THIS.parent))
sys.path.insert(0, str(THIS.parent.parent))
sys.path.insert(0, str(THIS.parent.parent.parent))    # repo root
from common import (TupleTimer, grid_uv, img_paths, load_tuples,  # noqa: E402
                    resize_crop_uv_map, save_pred_npz, timing_summary)


def squeeze_uv_map(uv, original_wh, processed_wh):
    return (uv + 0.5) * np.asarray(original_wh) / np.asarray(processed_wh) - 0.5


def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--tuples", required=True)
    ap.add_argument("--frames-root", default=None,
                    help="override the frames_root stored in tuples.json")
    ap.add_argument("--ckpt", required=True,
                    help="NAME=path[:wrap] — same spec as eval_2d3ds_pose_v2")
    ap.add_argument("--out", required=True, help="preds root (model name appended)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--pano", nargs="?", const="on", default="off",
                    choices=["off", "on", "auto", "label", "force"],
                    help="pano_wrap flag: bare --pano == on (whole set is ERP, "
                         "e.g. 2D3DS); auto == per-view image-only full-ERP "
                         "detector (erp_detect) — detected "
                         "panos also get the definitional AR token 2.0; use "
                         "auto for heterogeneous tracks (blk_mixed); label uses view.kind==erp")
    ap.add_argument("--pt-stride", type=int, default=4)
    ap.add_argument("--pt-cap", type=int, default=150_000)
    ap.add_argument("--pt-keep", choices=["median", "all", "dense"],
                    default="median",
                    help="point-export filter: median = conf>=per-view median "
                         "(published realset protocol); all = no conf filter "
                         "(model mask still applied); dense = additionally "
                         "disable apply_mask (raw geometry, density probe)")
    ap.add_argument("--limit", type=int, default=0, help="debug: first N tuples")
    ap.add_argument("--ids", nargs="+", default=None,
                    help="only these tuple ids (viz/debug; overrides --limit)")
    ap.add_argument("--input-resize", choices=["crop", "squeeze"], default="crop")
    args = ap.parse_args()
    print(f"input_resize={args.input_resize}", flush=True)

    sys.path.insert(0, str(THIS.parent.parent))
    from eval_2d3ds_pose_v2 import (build_model_imagesonly, load_variant_ckpt,
                                    load_pth)
    from meow_model import VARIANT_OVERRIDES, build_model_with_ar
    from mapanything.utils.image import load_images
    from PIL import Image

    name, path = args.ckpt.split("=", 1)
    variant = ""
    if ":" in path and path.rsplit(":", 1)[1] in VARIANT_OVERRIDES:
        path, variant = path.rsplit(":", 1)
    assert Path(path).is_file(), f"ckpt not found: {path}"
    if variant:
        model = build_model_with_ar(args.device, variant)
        load_variant_ckpt(model, path)
        print(f"[ours:{name}] ckpt loaded strictly (variant {variant})")
    else:
        model = build_model_imagesonly(args.device)
        load_pth(model, path)
        print(f"[ours:{name}] vanilla ckpt loaded")

    spec = load_tuples(args.tuples)
    if args.frames_root:
        spec["frames_root"] = args.frames_root
    if args.ids:
        want = set(args.ids)
        tups = [t for t in spec["tuples"] if t["id"] in want]
        assert len(tups) == len(want), f"ids not all found: {want}"
    else:
        tups = spec["tuples"][: args.limit] if args.limit else spec["tuples"]
    out_dir = Path(args.out) / name
    rng = np.random.default_rng(0)

    secs, vrams = [], []
    n_fail = 0
    for k, tup in enumerate(tups):
      try:                                     # a degenerate tuple (e.g. a
        paths = img_paths(spec, tup)           # singular linalg.solve on a
                                               # fisheye tuple) is logged and
                                               # skipped instead of aborting
        with TupleTimer(args.device) as tt:
            views = load_images(paths, verbose=False, resize_mode=(
                "fixed_squeeze" if args.input_resize == "squeeze" else "fixed_mapping"))
            original_wh = []
            for vi, (v, p) in enumerate(zip(views, paths)):
                with Image.open(p) as im:
                    w0, h0 = im.size
                    original_wh.append((w0, h0))
                    if args.pano == "auto":
                        from erp_detect import detect_full_erp
                        is_pano, _ = detect_full_erp(
                            np.asarray(im.convert("RGB")))
                    else:
                        is_pano = (tup["views"][vi]["kind"] == "erp"
                                   if args.pano == "label" else args.pano == "on")
                if args.pano == "force":   # flag-sensitivity control: every view flagged as a panorama
                    is_pano = True
                # detected (auto) or forced panos: AR token 2.0 (true full-panorama
                # aspect ratio); otherwise the fed image W/H
                ar = 2.0 if (args.pano in ("auto", "force") and is_pano) \
                    else w0 / max(h0, 1)
                if variant:
                    v["aspect_ratio"] = torch.tensor([ar], dtype=torch.float32)
                    v["pano_wrap"] = bool(is_pano)
            for v in views:
                for kk, val in list(v.items()):
                    if torch.is_tensor(val):
                        v[kk] = val.to(args.device, non_blocking=True)
            with torch.no_grad():
                preds = model.infer(views, memory_efficient_inference=False,
                                    use_amp=True, amp_dtype="bf16",
                                    apply_mask=(args.pt_keep != "dense"),
                                    mask_edges=True)
        secs.append(tt.sec)
        vrams.append(tt.vram_gb)
        c2w, pts_l, uv_l, vidx_l = [], [], [], []
        for i, p in enumerate(preds):
            c2w.append(p["camera_poses"][0].detach().cpu().numpy().astype(np.float64))
            pm = p["pts3d"][0].detach().float().cpu().numpy()      # Hp,Wp,3
            hp, wp = pm.shape[:2]
            conf = p.get("conf")
            cf = (conf[0].detach().float().cpu().numpy().reshape(hp, wp)
                  if conf is not None else np.ones((hp, wp), np.float32))
            uv_p = grid_uv(hp, wp, args.pt_stride)
            pv = pm[uv_p[:, 1].astype(int), uv_p[:, 0].astype(int)]
            cv = cf[uv_p[:, 1].astype(int), uv_p[:, 0].astype(int)]
            good = np.isfinite(pv).all(1)
            if args.pt_keep == "median":
                good &= cv >= np.median(cv)
            w0, h0 = tup["views"][i]["w"], tup["views"][i]["h"]
            if args.input_resize == "squeeze":
                w0, h0 = original_wh[i]
                uo = squeeze_uv_map(uv_p[good], (w0, h0), (wp, hp))
            else:
                s, ox, oy = resize_crop_uv_map((w0, h0), (wp, hp))
                uo = np.stack([(uv_p[good, 0] + ox) / s, (uv_p[good, 1] + oy) / s], 1)
            inb = (uo[:, 0] >= 0) & (uo[:, 0] < w0) & (uo[:, 1] >= 0) & (uo[:, 1] < h0)
            pts_l.append(pv[good][inb])
            uv_l.append(uo[inb])
            vidx_l.append(np.full(inb.sum(), i, np.int32))
        pts = np.concatenate(pts_l) if pts_l else np.zeros((0, 3))
        uv = np.concatenate(uv_l) if uv_l else np.zeros((0, 2))
        vidx = np.concatenate(vidx_l) if vidx_l else np.zeros((0,), np.int32)
        if len(pts) > args.pt_cap:
            idx = rng.choice(len(pts), args.pt_cap, replace=False)
            pts, uv, vidx = pts[idx], uv[idx], vidx[idx]
        save_pred_npz(out_dir / f"{tup['id']}.npz", np.stack(c2w), pts, uv, vidx,
                      proc=f"mapanything load_images input_resize={args.input_resize} pano={args.pano}",
                      sec=tt.sec, vram_gb=tt.vram_gb)
        if k % 10 == 0 or k == len(tups) - 1:
            print(f"[ours:{name}] {k + 1}/{len(tups)} tuples "
                  f"(V={len(preds)}, pts={len(pts)}, {tt.sec:.2f}s)")
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
      except Exception as e:
        n_fail += 1
        print(f"[ours:{name}] TUPLE FAILED {tup['id']}: {e!r} (skip)")
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
    if n_fail:
        print(f"[ours:{name}] {n_fail} tuples failed and were skipped")
    timing_summary(f"ours:{name}", secs, vrams)
    print(f"[ours:{name}] DONE -> {out_dir}")


if __name__ == "__main__":
    main()
