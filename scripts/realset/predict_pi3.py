#!/usr/bin/env python3
"""pi3 predictor (conda env: pi3vggt). Official model via hub weights
(Pi3.from_pretrained('yyfz233/Pi3')). pi3 is resolution-flexible; all views of
a tuple are resized to one size, derived from the first view's aspect ratio
(multiples of 14 under a ~255k pixel budget, the same budget pi3's own loader
uses), so the uv mapping is a pure per-axis scale.

Output-dict key names are resolved against candidate lists and asserted with
the full available-key set in the error message: if the installed pi3 version
changed its keys, --selftest fails with an actionable message instead of
silently writing wrong outputs.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (TupleTimer, grid_uv, img_paths, load_tuples,  # noqa: E402
                    save_pred_npz, timing_summary)

PIXEL_LIMIT = 255_000


def pick(d: dict, names, what: str):
    for n in names:
        if n in d:
            return d[n]
    raise KeyError(f"pi3 output has no {what} key; tried {names}, "
                   f"available: {sorted(d.keys())}")


def target_hw(w0: int, h0: int):
    scale = min(1.0, (PIXEL_LIMIT / (w0 * h0)) ** 0.5)
    w = max(14, int(round(w0 * scale / 14)) * 14)
    h = max(14, int(round(h0 * scale / 14)) * 14)
    while w * h > PIXEL_LIMIT:
        if w >= h:
            w -= 14
        else:
            h -= 14
    return h, w


def load_batch(paths, tup):
    import torch
    from PIL import Image
    h, w = target_hw(tup["views"][0]["w"], tup["views"][0]["h"])
    ims = []
    for p in paths:
        with Image.open(p) as im:
            im = im.convert("RGB").resize((w, h), Image.BICUBIC)
        ims.append(torch.from_numpy(np.asarray(im)).permute(2, 0, 1).float() / 255.0)
    return torch.stack(ims), (h, w)


def run_model(model, imgs, dev):
    import torch
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            res = model(imgs[None].to(dev))
    return {k: v for k, v in res.items()} if isinstance(res, dict) else res


def selftest():
    import torch  # noqa: F401
    from pi3.models.pi3 import Pi3
    dev = "cuda:0"
    model = Pi3.from_pretrained("yyfz233/Pi3").to(dev).eval()
    rng = np.random.default_rng(0)
    import tempfile
    from PIL import Image
    td = Path(tempfile.mkdtemp())
    ps = []
    for i in range(2):
        a = (rng.random((280, 420, 3)) * 255).astype(np.uint8)
        p = td / f"t{i}.png"
        Image.fromarray(a).save(p)
        ps.append(str(p))
    imgs, hw = load_batch(ps, {"views": [{"w": 420, "h": 280}]})
    res = run_model(model, imgs, dev)
    pts = pick(res, ["points", "pts3d", "world_points"], "points")
    poses = pick(res, ["camera_poses", "cam_poses", "poses"], "poses")
    conf = pick(res, ["conf", "confidence", "points_conf"], "conf")
    print(f"[selftest] proc={hw} points={tuple(pts.shape)} "
          f"poses={tuple(poses.shape)} conf={tuple(conf.shape)}")
    assert pts.shape[1] == 2 and poses.shape[-2:] == (4, 4)
    assert np.isfinite(poses.float().cpu().numpy()).all()
    print("[selftest] PASS: pi3 loads, keys resolved, shapes sane")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tuples")
    ap.add_argument("--frames-root", default=None,
                    help="override the frames_root stored in tuples.json")
    ap.add_argument("--out")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--model-name", default="pi3")
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
    from pi3.models.pi3 import Pi3
    dev = args.device
    model = Pi3.from_pretrained("yyfz233/Pi3").to(dev).eval()
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
            imgs, (hp, wp) = load_batch(paths, tup)
            res = run_model(model, imgs, dev)
            pts_t = pick(res, ["points", "pts3d", "world_points"], "points")[0]
            poses = pick(res, ["camera_poses", "cam_poses", "poses"], "poses")[0]
            conf = pick(res, ["conf", "confidence", "points_conf"], "conf")[0]
        secs.append(tt.sec)
        vrams.append(tt.vram_gb)
        c2w = poses.float().cpu().numpy().astype(np.float64)
        pm = pts_t.float().cpu().numpy()                 # N,H,W,3
        cf = conf.float().cpu().numpy()
        if cf.ndim == 4:
            cf = cf[..., 0]
        pts_l, uv_l, vidx_l = [], [], []
        for i in range(len(paths)):
            w0, h0 = tup["views"][i]["w"], tup["views"][i]["h"]
            uvp = grid_uv(hp, wp, args.pt_stride)
            pv = pm[i, uvp[:, 1].astype(int), uvp[:, 0].astype(int)]
            cv = cf[i, uvp[:, 1].astype(int), uvp[:, 0].astype(int)]
            good = np.isfinite(pv).all(1) & (cv >= np.median(cv))
            uo = np.stack([uvp[good, 0] * (w0 / wp), uvp[good, 1] * (h0 / hp)], 1)
            pts_l.append(pv[good])
            uv_l.append(uo)
            vidx_l.append(np.full(good.sum(), i, np.int32))
        pts = np.concatenate(pts_l)
        uv = np.concatenate(uv_l)
        vidx = np.concatenate(vidx_l)
        if len(pts) > args.pt_cap:
            idx = rng.choice(len(pts), args.pt_cap, replace=False)
            pts, uv, vidx = pts[idx], uv[idx], vidx[idx]
        save_pred_npz(out_dir / f"{tup['id']}.npz", c2w, pts, uv, vidx,
                      proc=f"pi3 resize {hp}x{wp}", sec=tt.sec, vram_gb=tt.vram_gb)
        if k % 10 == 0 or k == len(tups) - 1:
            print(f"[pi3] {k + 1}/{len(tups)} (V={len(paths)}, pts={len(pts)}, {tt.sec:.2f}s)")
        torch.cuda.empty_cache()
    timing_summary("pi3", secs, vrams)
    print(f"[pi3] DONE -> {out_dir}")


if __name__ == "__main__":
    main()
