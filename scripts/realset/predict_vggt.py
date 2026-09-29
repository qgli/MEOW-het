#!/usr/bin/env python3
"""VGGT predictor (conda env: pi3vggt). Official inference path:
load_and_preprocess_images(mode='pad') -> VGGT-1B -> pose_enc -> extri/intri,
world_points(+conf). No mapanything imports here.

uv mapping: VGGT pads with white (value 1.0), not zero, so detecting a zero
border does not work. The mapping is analytic instead: vggt_pad_geometry
replicates load_and_preprocess_images(mode='pad') (long side -> 518, short
side rounded to a multiple of 14, which gives a slight per-axis scale
difference; centred padding via //2), and verify_content() checks it at
runtime: the analytic content window must correlate >0.98 with an
independent PIL resize of the original, with no assumption about the pad
value. A change in VGGT preprocessing therefore raises an error instead of
writing misaligned uv.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (TupleTimer, grid_uv, img_paths, load_tuples,  # noqa: E402
                    save_pred_npz, timing_summary)


TARGET = 518


def vggt_pad_geometry(w0: int, h0: int, target: int = TARGET):
    """Replica of vggt load_and_preprocess_images(mode='pad') geometry:
    long side -> target, short side rounded to a multiple of 14, centred
    padding (//2 on top/left). Returns (nw, nh, pad_left, pad_top).
    Map processed uv -> original:  u0 = (u'-pad_left)*(w0/nw),
                                   v0 = (v'-pad_top )*(h0/nh)."""
    if w0 >= h0:
        nw = target
        nh = round(h0 * (target / w0) / 14) * 14
    else:
        nh = target
        nw = round(w0 * (target / h0) / 14) * 14
    return nw, nh, (target - nw) // 2, (target - nh) // 2


def verify_content(img_chw: np.ndarray, path: str, w0: int, h0: int):
    """Check that the analytic content window matches an independent PIL
    resize of the original (corr>0.98) and that the pad border is constant.
    No assumption about the pad value (white/black/other)."""
    from PIL import Image
    nw, nh, left, top = vggt_pad_geometry(w0, h0)
    got = img_chw[:, top:top + nh, left:left + nw].mean(0)
    with Image.open(path) as im:
        ref = np.asarray(im.convert("L").resize((nw, nh), Image.BICUBIC),
                         np.float64) / 255.0
    if got.std() > 1e-6 and ref.std() > 1e-6:
        c = float(np.corrcoef(got.ravel(), ref.ravel())[0, 1])
        assert c > 0.98, (
            f"VGGT content-window mismatch (corr={c:.3f}) for {path} — "
            f"load_and_preprocess_images geometry drifted from "
            f"vggt_pad_geometry; uv coordinates are unreliable until this is fixed")
    else:
        c = float("nan")
    if top > 0:
        assert img_chw[:, :top].std() < 1e-3, "top pad not constant"
    if left > 0:
        assert img_chw[:, :, :left].std() < 1e-3, "left pad not constant"
    return c


def selftest():
    from vggt.utils.load_fn import load_and_preprocess_images
    from PIL import Image
    import tempfile
    td = Path(tempfile.mkdtemp())
    specs = [("tall.png", 100, 400), ("wide.png", 400, 100)]
    paths = []
    for name, w0, h0 in specs:
        yy, xx = np.mgrid[0:h0, 0:w0].astype(np.float64)
        g = (255 * (xx / xx.max() + yy / yy.max()) / 2).astype(np.uint8)
        p = td / name
        Image.fromarray(np.stack([g, g, g], -1)).save(p)
        paths.append(str(p))
    t = load_and_preprocess_images(paths, mode="pad")
    assert t.ndim == 4 and t.shape[-2:] == (TARGET, TARGET), f"tensor {t.shape}"
    for i, (_n, w0, h0) in enumerate(specs):
        nw, nh, left, top = vggt_pad_geometry(w0, h0)
        c = verify_content(t[i].numpy(), paths[i], w0, h0)
        print(f"[selftest] {_n}: proc content {nw}x{nh} pad(l={left},t={top}) "
              f"content-corr={c:.4f}")
    print("[selftest] PASS: analytic pad geometry verified against tensor "
          "(pad-value-agnostic)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tuples")
    ap.add_argument("--frames-root", default=None,
                    help="override the frames_root stored in tuples.json")
    ap.add_argument("--out")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--model-name", default="VGGT")
    ap.add_argument("--pt-stride", type=int, default=4)
    ap.add_argument("--pt-cap", type=int, default=150_000)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return
    assert args.tuples and args.out, "--tuples/--out required"

    import torch
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    dev = args.device
    dtype = (torch.bfloat16 if torch.cuda.get_device_capability(0)[0] >= 8
             else torch.float16)
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(dev).eval()
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
            images = load_and_preprocess_images(paths, mode="pad").to(dev)
            with torch.no_grad(), torch.cuda.amp.autocast(dtype=dtype):
                pred = model(images)
            extri, _intri = pose_encoding_to_extri_intri(pred["pose_enc"],
                                                         images.shape[-2:])
            extri = extri[0].float().cpu().numpy()          # (S,3,4) w2c
            wp = pred["world_points"][0].float().cpu().numpy()      # S,H,W,3
            wc = pred["world_points_conf"][0].float().cpu().numpy() # S,H,W
        secs.append(tt.sec)
        vrams.append(tt.vram_gb)
        c2w, pts_l, uv_l, vidx_l = [], [], [], []
        Hp, Wp = images.shape[-2:]
        for i in range(len(paths)):
            T = np.eye(4)
            T[:3, :4] = extri[i]
            c2w.append(np.linalg.inv(T))
            w0, h0 = tup["views"][i]["w"], tup["views"][i]["h"]
            nw, nh, left, top = vggt_pad_geometry(w0, h0)
            if k == 0:  # pad-geometry check on the first tuple (cheap)
                verify_content(images[i].detach().float().cpu().numpy(),
                               paths[i], w0, h0)
            uvp = grid_uv(Hp, Wp, args.pt_stride)
            pv = wp[i, uvp[:, 1].astype(int), uvp[:, 0].astype(int)]
            cv = wc[i, uvp[:, 1].astype(int), uvp[:, 0].astype(int)]
            good = np.isfinite(pv).all(1) & (cv >= np.median(cv))
            uo = np.stack([(uvp[good, 0] - left) * (w0 / nw),
                           (uvp[good, 1] - top) * (h0 / nh)], 1)
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
        save_pred_npz(out_dir / f"{tup['id']}.npz", np.stack(c2w), pts, uv, vidx,
                      proc=f"vggt pad {Hp}x{Wp}", sec=tt.sec, vram_gb=tt.vram_gb)
        if k % 10 == 0 or k == len(tups) - 1:
            print(f"[vggt] {k + 1}/{len(tups)} (V={len(paths)}, pts={len(pts)}, {tt.sec:.2f}s)")
        torch.cuda.empty_cache()
    timing_summary("VGGT", secs, vrams)
    print(f"[vggt] DONE -> {out_dir}")


if __name__ == "__main__":
    main()
