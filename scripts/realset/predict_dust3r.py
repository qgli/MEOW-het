#!/usr/bin/env python3
"""DUSt3R predictor (conda env: mast3r, which vendors dust3r).
Official path: load_images(512) -> make_pairs(complete, symmetrized) ->
inference -> global_aligner(PointCloudOptimizer, init=mst, niter=300).
2-view tuples use PairViewer (no optimization), matching the official demo.

uv mapping: dust3r resizes the long side to 512, then centre-crops to
multiples of 16; common.resize_crop_uv_map inverts this composition.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (TupleTimer, grid_uv, img_paths, load_tuples,  # noqa: E402
                    resize_crop_uv_map, save_pred_npz, timing_summary)


def run_tuple(model, paths, dev, scene_graph, niter, bs):
    import torch
    from dust3r.cloud_opt import GlobalAlignerMode, global_aligner
    from dust3r.image_pairs import make_pairs
    from dust3r.inference import inference
    from dust3r.utils.image import load_images
    imgs = load_images(paths, size=512, verbose=False)
    pairs = make_pairs(imgs, scene_graph=scene_graph, prefilter=None,
                       symmetrize=True)
    out = inference(pairs, model, dev, batch_size=bs, verbose=False)
    mode = (GlobalAlignerMode.PointCloudOptimizer if len(paths) > 2
            else GlobalAlignerMode.PairViewer)
    scene = global_aligner(out, device=dev, mode=mode, verbose=False)
    if mode == GlobalAlignerMode.PointCloudOptimizer:
        scene.compute_global_alignment(init="mst", niter=niter,
                                       schedule="cosine", lr=0.01)
    with torch.no_grad():
        poses = scene.get_im_poses().detach().cpu().numpy().astype(np.float64)
        pts3d = [p.detach().cpu().numpy() for p in scene.get_pts3d()]
        confs = [c.detach().cpu().numpy() for c in scene.im_conf]
    return poses, pts3d, confs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tuples")
    ap.add_argument("--frames-root", default=None,
                    help="override the frames_root stored in tuples.json")
    ap.add_argument("--out")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--model-name", default="DUSt3R")
    ap.add_argument("--weights", default="naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt")
    ap.add_argument("--scene-graph", default="complete",
                    help="complete (official) | swin-3 etc. for big tuples")
    ap.add_argument("--niter", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--pt-stride", type=int, default=4)
    ap.add_argument("--pt-cap", type=int, default=150_000)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    import torch  # noqa: F401
    from dust3r.model import AsymmetricCroCo3DStereo
    dev = args.device
    model = AsymmetricCroCo3DStereo.from_pretrained(args.weights).to(dev).eval()

    if args.selftest:
        import tempfile
        from PIL import Image
        rng = np.random.default_rng(0)
        td = Path(tempfile.mkdtemp())
        ps = []
        for i in range(3):
            a = (rng.random((240, 320, 3)) * 255).astype(np.uint8)
            a[:, 100 + 30 * i: 140 + 30 * i] = 255
            p = td / f"t{i}.png"
            Image.fromarray(a).save(p)
            ps.append(str(p))
        poses, pts3d, confs = run_tuple(model, ps, dev, "complete", 50, 2)
        print(f"[selftest] poses={poses.shape} pts0={pts3d[0].shape} "
              f"conf0={confs[0].shape} finite={np.isfinite(poses).all()}")
        assert poses.shape == (3, 4, 4) and pts3d[0].ndim == 3
        print("[selftest] PASS: dust3r pipeline runs end-to-end")
        return
    assert args.tuples and args.out, "--tuples/--out required"

    spec = load_tuples(args.tuples)
    if args.frames_root:
        spec["frames_root"] = args.frames_root
    tups = spec["tuples"][: args.limit] if args.limit else spec["tuples"]
    out_dir = Path(args.out) / args.model_name
    rng = np.random.default_rng(0)

    secs, vrams = [], []
    for k, tup in enumerate(tups):
        paths = img_paths(spec, tup)
        with TupleTimer(dev) as tt:
            poses, pts3d, confs = run_tuple(model, paths, dev, args.scene_graph,
                                            args.niter, args.batch_size)
        secs.append(tt.sec)
        vrams.append(tt.vram_gb)
        pts_l, uv_l, vidx_l = [], [], []
        for i in range(len(paths)):
            pm, cf = pts3d[i], confs[i]
            hp, wp = pm.shape[:2]
            uvp = grid_uv(hp, wp, args.pt_stride)
            pv = pm[uvp[:, 1].astype(int), uvp[:, 0].astype(int)]
            cv = cf[uvp[:, 1].astype(int), uvp[:, 0].astype(int)]
            good = np.isfinite(pv).all(1) & (cv >= np.median(cv))
            w0, h0 = tup["views"][i]["w"], tup["views"][i]["h"]
            s, ox, oy = resize_crop_uv_map((w0, h0), (wp, hp))
            uo = np.stack([(uvp[good, 0] + ox) / s, (uvp[good, 1] + oy) / s], 1)
            inb = (uo[:, 0] >= 0) & (uo[:, 0] < w0) & (uo[:, 1] >= 0) & (uo[:, 1] < h0)
            pts_l.append(pv[good][inb])
            uv_l.append(uo[inb])
            vidx_l.append(np.full(inb.sum(), i, np.int32))
        pts = np.concatenate(pts_l)
        uv = np.concatenate(uv_l)
        vidx = np.concatenate(vidx_l)
        if len(pts) > args.pt_cap:
            idx = rng.choice(len(pts), args.pt_cap, replace=False)
            pts, uv, vidx = pts[idx], uv[idx], vidx[idx]
        save_pred_npz(out_dir / f"{tup['id']}.npz", poses, pts, uv, vidx,
                      proc=f"dust3r 512 {args.scene_graph} niter{args.niter} (incl GA optim)",
                      sec=tt.sec, vram_gb=tt.vram_gb)
        print(f"[dust3r] {k + 1}/{len(tups)} (V={len(paths)}, pts={len(pts)}, {tt.sec:.1f}s)")
        import torch as _t
        _t.cuda.empty_cache()
    timing_summary("DUSt3R", secs, vrams)
    print(f"[dust3r] DONE -> {out_dir}")


if __name__ == "__main__":
    main()
