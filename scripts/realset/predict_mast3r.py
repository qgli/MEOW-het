#!/usr/bin/env python3
"""MASt3R predictor (conda env: mast3r). Primary path = the official demo
pipeline: load_images(512) -> make_pairs -> sparse_global_alignment
(lr1=0.07/niter1=500, lr2=0.014/niter2=200, opt_depth=True) ->
get_im_poses + get_dense_pts3d.

If sparse GA fails on the first tuple (e.g. an API change in the installed
mast3r), the script switches to the dense DUSt3R-style global_aligner over
MASt3R outputs for the rest of the run, prints a warning, and records the
mode in every npz's proc note so the report can label it.

Checkpoint: a local metric MASt3R checkpoint (--weights); no hub download.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (TupleTimer, grid_uv, img_paths, load_tuples,  # noqa: E402
                    resize_crop_uv_map, save_pred_npz, timing_summary)


def run_sparse(model, paths, dev):
    import torch
    from dust3r.image_pairs import make_pairs
    from dust3r.utils.image import load_images
    from mast3r.cloud_opt.sparse_ga import sparse_global_alignment
    imgs = load_images(paths, size=512, verbose=False)
    pairs = make_pairs(imgs, scene_graph="complete", prefilter=None,
                       symmetrize=True)
    cache = tempfile.mkdtemp(prefix="mast3r_ga_")
    try:
        scene = sparse_global_alignment(
            paths, pairs, cache, model,
            lr1=0.07, niter1=500, lr2=0.014, niter2=200,
            device=dev, opt_depth=True, shared_intrinsics=False,
            matching_conf_thr=5.0)
        with torch.no_grad():
            poses = scene.get_im_poses().detach().cpu().numpy().astype(np.float64)
            pts3d, _depth, confs = scene.get_dense_pts3d(clean_depth=True)
            shapes = [im.shape[:2] for im in scene.imgs]
            pts3d = [p.detach().cpu().numpy().reshape(h, w, 3)
                     for p, (h, w) in zip(pts3d, shapes)]
            confs = [c.detach().cpu().numpy().reshape(h, w)
                     for c, (h, w) in zip(confs, shapes)]
    finally:
        shutil.rmtree(cache, ignore_errors=True)
    return poses, pts3d, confs


def run_dense(model, paths, dev):
    """Fallback: DUSt3R dense global aligner over MASt3R pair outputs."""
    import torch
    from dust3r.cloud_opt import GlobalAlignerMode, global_aligner
    from dust3r.image_pairs import make_pairs
    from dust3r.inference import inference
    from dust3r.utils.image import load_images
    imgs = load_images(paths, size=512, verbose=False)
    pairs = make_pairs(imgs, scene_graph="complete", prefilter=None,
                       symmetrize=True)
    out = inference(pairs, model, dev, batch_size=2, verbose=False)
    mode = (GlobalAlignerMode.PointCloudOptimizer if len(paths) > 2
            else GlobalAlignerMode.PairViewer)
    scene = global_aligner(out, device=dev, mode=mode, verbose=False)
    if mode == GlobalAlignerMode.PointCloudOptimizer:
        scene.compute_global_alignment(init="mst", niter=300,
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
    ap.add_argument("--model-name", default="MASt3R")
    ap.add_argument("--weights", required=True,
                    help="local metric MASt3R ckpt path")
    ap.add_argument("--ga", choices=["sparse", "dense"], default="sparse")
    ap.add_argument("--pt-stride", type=int, default=4)
    ap.add_argument("--pt-cap", type=int, default=150_000)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    import torch  # noqa: F401
    from mast3r.model import AsymmetricMASt3R
    dev = args.device
    assert Path(args.weights).is_file(), f"ckpt missing: {args.weights}"
    model = AsymmetricMASt3R.from_pretrained(args.weights).to(dev).eval()

    def run(paths, mode):
        return run_sparse(model, paths, dev) if mode == "sparse" \
            else run_dense(model, paths, dev)

    if args.selftest:
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
        mode = args.ga
        try:
            poses, pts3d, confs = run(ps, mode)
        except Exception as e:
            print(f"[selftest] sparse GA failed ({e!r}) -> trying dense")
            mode = "dense"
            poses, pts3d, confs = run(ps, mode)
        print(f"[selftest] mode={mode} poses={poses.shape} pts0={pts3d[0].shape}")
        assert poses.shape == (3, 4, 4) and pts3d[0].ndim == 3
        print(f"[selftest] PASS (ga={mode}) — use --ga {mode} for the real run")
        return
    assert args.tuples and args.out, "--tuples/--out required"

    spec = load_tuples(args.tuples)
    if args.frames_root:
        spec["frames_root"] = args.frames_root
    tups = spec["tuples"][: args.limit] if args.limit else spec["tuples"]
    out_dir = Path(args.out) / args.model_name
    rng = np.random.default_rng(0)
    mode = args.ga

    secs, vrams = [], []
    for k, tup in enumerate(tups):
        paths = img_paths(spec, tup)
        try:
            with TupleTimer(dev) as tt:
                poses, pts3d, confs = run(paths, mode)
        except Exception as e:
            if mode == "sparse" and k == 0:
                print(f"[mast3r] !! sparse GA failed on first tuple ({e!r}) — "
                      f"switching to the dense GA fallback for all tuples")
                mode = "dense"
                with TupleTimer(dev) as tt:
                    poses, pts3d, confs = run(paths, mode)
            else:
                print(f"[mast3r] tuple {tup['id']} failed: {e!r} — skipped")
                continue
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
                      proc=f"mast3r 512 ga={mode} (incl GA optim)",
                      sec=tt.sec, vram_gb=tt.vram_gb)
        print(f"[mast3r] {k + 1}/{len(tups)} (V={len(paths)}, pts={len(pts)}, ga={mode}, {tt.sec:.1f}s)")
        import torch as _t
        _t.cuda.empty_cache()
    timing_summary("MASt3R", secs, vrams)
    print(f"[mast3r] DONE (ga={mode}) -> {out_dir}")


if __name__ == "__main__":
    main()
