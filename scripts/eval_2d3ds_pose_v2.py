"""Multi-view relative pose on Stanford 2D-3D-S panoramas through the official inference path.

Sampling modes:
  --tuples-json     a fixed tuple list (no sampling), e.g. the 2D3DS relative-pose list of
                    the Wid3R code
  --wid3r-faithful  sampling as described for the Wid3R 2D3DS protocol: per room, a
                    random subset of k in [10, 30] panoramas (all of them when the room
                    has fewer), no baseline constraint, 10 repeats per room; no free
                    parameters.
  default           a covisibility-sampled sensitivity variant with fixed defaults
                    (min-views 6, max-views 16, repeats 8, baseline 0.1-6.0 m, seed 0,
                    max-cases 20, areas 5a and 5b). A small --min-views (e.g. 2) makes
                    the sampled tuples trivially easy (RRA/RTA reach 100).

Unlike eval_2d3ds_pose.py, which resizes each panorama by hand to H=res, W=2*res
(518x1036, about four times the tokens of the official path) and calls model(views)
directly, this script uses the official pipeline (as in scripts/infer_wild.py):
  - mapanything.utils.image.load_images  -> resolution and aspect bucket (a 2:1
                                             panorama becomes 518x252)
  - model.infer()                        -> full pre- and post-processing
  - pred['camera_poses']                 -> (B,4,4) c2w, read directly
GT loading, sampling and metric helpers are imported from eval_2d3ds_pose.py, so the
protocol is identical; only the model input path differs.

ckpt spec: NAME=path[:wrap|:ar]. Checkpoints trained with the aspect-ratio embedding
(ar_prob=1) and the panorama wrap of the dense head (pano_wrap_dpt=true), such as the
final MEOW model, need :wrap: the model is then built with the aspect-ratio encoder
and the wrap, and the checkpoint loads with no unexpected keys (the encoder is loaded,
not skipped). Their inputs get aspect_ratio and pano_wrap (see predict_c2w_infer), as in
training.

Usage:
  CUDA_VISIBLE_DEVICES=0 python scripts/eval_2d3ds_pose_v2.py \
    --ckpts MEOW=<path>:wrap MapAnything=checkpoints/facebook_map-anything-apache.pth \
    --stanford-root /path/to/2d3ds --areas area_5a area_5b --wid3r-faithful \
    --out experiments/pub_bench/2d3ds_pose_v2
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

THIS = Path(__file__).resolve()
ROOT = THIS.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(THIS.parent))

from mapanything.utils.image import load_images  # noqa: E402

# Reuse the protocol helpers (GT, sampling, metrics) of eval_2d3ds_pose.py unchanged.
from eval_2d3ds_pose import (  # noqa: E402
    STANFORD_ROOT,
    list_frames_by_scene,
    sample_covisible_group,
    load_gt_c2w,
    eval_case,
    auc_min,
)


def build_model_imagesonly(device: str):
    """MapAnything in images_only task (no ray/depth/pose inputs)."""
    from mapanything.utils.hf_utils.hf_helpers import init_hydra_config
    from mapanything.models import init_model

    cfg = init_hydra_config(
        "configs/train.yaml",
        overrides=[
            "machine=default",
            "model=mapanything",
            "model/task=images_only",
            "model.encoder.uses_torch_hub=true",
        ],
    )
    model = init_model(
        model_str=cfg.model.model_str,
        model_config=cfg.model.model_config,
        torch_hub_force_reload=False,
    )
    return model.to(device).eval()


def load_pth(model, path):
    """Loader for the images-only build (checkpoints without the aspect-ratio encoder)."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck.get("model", ck)
    incompat = model.load_state_dict(sd, strict=False)
    miss = [k for k in incompat.missing_keys if not k.startswith("ar_encoder.")]
    unexp = [k for k in incompat.unexpected_keys if not k.startswith("ar_encoder.")]
    if miss or unexp:
        raise RuntimeError(f"ckpt mismatch miss={miss[:6]} unexp={unexp[:6]}")
    if incompat.unexpected_keys:
        raise RuntimeError(
            "ckpt has ar_encoder weights but the model was built without it — "
            "use the NAME=path:wrap spec so build_model_with_ar is used and the "
            "AR conditioning is really loaded (silently dropping it distorts "
            "the results).")


def load_variant_ckpt(model, path):
    """Strict loader for a model built by build_model_with_ar (aspect-ratio encoder required)."""
    from meow_model import load_ckpt_inplace
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck.get("model", ck)
    n_fresh = load_ckpt_inplace(model, sd)  # raises on any unexpected or missing non-encoder key
    if n_fresh:
        raise RuntimeError(
            f"{n_fresh} ar_encoder keys absent from ckpt — this checkpoint has no "
            f"aspect-ratio encoder; drop the :wrap suffix for it")
    del ck, sd


def rgb_path(area, base):
    return str(Path(STANFORD_ROOT) / area / "pano" / "rgb" / f"{base}_rgb.png")


@torch.no_grad()
def predict_c2w_infer(model, paths, device, memory_efficient=False,
                      ar_mode="asfed", true_ar=2.0):
    """Official pipeline: load_images -> model.infer -> camera_poses (4x4 c2w).

    Sets the per-view aspect_ratio (the input of the aspect-ratio embedding, trained
    with ar_prob=1) and pano_wrap according to ar_mode (used for squeezed-panorama
    tests):
      asfed : ar = W/H of the image file as fed, pano_wrap=True (default)
      true  : ar = --true-ar (2.0 = full panorama by definition), pano_wrap=True
      auto  : full-panorama detector (scripts/erp_detect.py, seam and pole cues);
              panoramas get ar=true_ar and pano_wrap=True, other views ar=W/H, no wrap
    """
    from PIL import Image
    views = load_images(paths, verbose=False)
    for v, p in zip(views, paths):
        with Image.open(p) as im:
            w0, h0 = im.size
            if ar_mode == "auto":
                import numpy as _np
                from erp_detect import detect_full_erp
                is_pano, _ = detect_full_erp(_np.asarray(im.convert("RGB")))
        if ar_mode == "true":
            ar, wrap = float(true_ar), True
        elif ar_mode == "auto":
            ar = float(true_ar) if is_pano else w0 / max(h0, 1)
            wrap = bool(is_pano)
        else:  # asfed
            ar, wrap = w0 / max(h0, 1), True
        v["aspect_ratio"] = torch.tensor([ar], dtype=torch.float32)
        v["pano_wrap"] = wrap
    for v in views:
        for k, val in list(v.items()):
            if torch.is_tensor(val):
                v[k] = val.to(device, non_blocking=True)
    preds = model.infer(
        views,
        memory_efficient_inference=memory_efficient,
        minibatch_size=1 if memory_efficient else None,
        use_amp=True, amp_dtype="bf16",
        apply_mask=True, mask_edges=True,
    )
    out = []
    for p in preds:
        cp = p["camera_poses"][0].detach().cpu().numpy()  # (4,4) c2w
        out.append(cp.astype(np.float64))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--areas", nargs="+", default=["area_5a", "area_5b"])
    ap.add_argument("--min-views", type=int, default=6)
    ap.add_argument("--max-views", type=int, default=16)
    ap.add_argument("--repeats", type=int, default=8)
    ap.add_argument("--baseline-min", type=float, default=0.1)
    ap.add_argument("--baseline-max", type=float, default=6.0,
                    help="maximum pairwise camera distance (m) within a tuple; 2D3DS panoramas are sparse, and "
                         "2.5 m leaves only 2-3 views per tuple")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-cases", type=int, default=20)
    ap.add_argument("--memory-efficient", action="store_true")
    ap.add_argument("--wid3r-faithful", action="store_true",
                    help="2D3DS cases sampled after the protocol described in the Wid3R paper: "
                         "per room a random k in [10, 30] panoramas (all of them in smaller rooms; "
                         "no baseline or covisibility constraint), 10 draws per room, shuffled and "
                         "truncated to --max-cases. Overrides min/max-views and the baseline filter.")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--stanford-root", default=None,
                    help="2D3DS root (default: env MEOW_2D3DS_ROOT)")
    ap.add_argument("--pre-squeeze", default=None,
                    help="deliver panos squeezed to this AR (e.g. 16:9) before "
                         "feeding, as a video stream of a 360 camera would; pure "
                         "resize, no crop; default None = native")
    ap.add_argument("--ar-mode", choices=["asfed", "true", "auto"],
                    default="asfed",
                    help="AR-conditioning token source (see predict_c2w_infer)")
    ap.add_argument("--true-ar", type=float, default=2.0)
    ap.add_argument("--tuples-json", type=Path, default=None,
                    help="evaluate a fixed tuple list instead of sampling: either a tuples.json of "
                         "benchmarks/ (seq + views[].img) or a map {area: [[image path, ...], ...]} such "
                         "as evaluation/datasets/seq-id-maps/stfd_relpose_seq-id-map.json of the Wid3R "
                         "code (only the area and the file name of each path are used)")
    ap.add_argument("--save-preds", type=Path, default=None,
                    help="also write the predicted camera-to-world poses of every case to "
                         "<dir>/case_NNN.npz (c2w, frames)")
    ap.add_argument("--keep-duplicates", action="store_true",
                    help="keep repeated draws of the same tuple (the original 20-case set holds one "
                         "tuple twice); by default a repeated tuple is evaluated once")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    squeeze_ar = None
    if args.pre_squeeze:
        a, b = args.pre_squeeze.split(":")
        squeeze_ar = float(a) / float(b)

    if args.stanford_root:
        # helpers read the module attribute at call time
        import eval_2d3ds_pose as _v1
        _v1.STANFORD_ROOT = args.stanford_root
        global STANFORD_ROOT
        STANFORD_ROOT = args.stanford_root

    from meow_model import VARIANT_OVERRIDES, build_model_with_ar

    ckpts = []
    for spec in args.ckpts:
        name, path = spec.split("=", 1)
        variant = ""
        if ":" in path and path.rsplit(":", 1)[1] in VARIANT_OVERRIDES:
            path, variant = path.rsplit(":", 1)
        assert os.path.isfile(path), f"ckpt not found: {path}"
        ckpts.append((name, path, variant))
    ckpts.sort(key=lambda t: t[2])  # group by variant -> fewest rebuilds

    rng = np.random.default_rng(args.seed)
    cases = []
    if args.tuples_json:
        spec = json.load(open(args.tuples_json))

        def base_of(path):
            name = Path(path).name
            return name[:-len("_rgb.png")] if name.endswith("_rgb.png") else Path(name).stem

        if isinstance(spec, dict) and "tuples" in spec:
            for tp in spec["tuples"]:
                cases.append((tp["seq"], [base_of(v["img"]) for v in tp["views"]]))
        else:
            for area, tuples in spec.items():
                for tp in tuples:
                    cases.append((area, [base_of(p) for p in tp]))
        args.areas = sorted({area for area, _ in cases})
    elif args.wid3r_faithful:
        # Wid3R §4.2: per scene (room) randomly pick 10-30 images, repeat 10x.
        # No baseline constraint — same-room ERP panoramas are naturally covisible.
        lo, hi, reps = 10, 30, 10
        for area in args.areas:
            by_scene = list_frames_by_scene(area)
            for room, frames_with_loc in by_scene.items():
                n = len(frames_with_loc)
                if n < 2:
                    continue
                bases = [b for b, _ in frames_with_loc]
                for _ in range(reps):
                    k = min(n, int(rng.integers(lo, hi + 1)))
                    if k < 2:
                        continue
                    idx = rng.choice(n, size=k, replace=False)
                    cases.append((area, [bases[i] for i in idx]))
    else:
        for area in args.areas:
            by_scene = list_frames_by_scene(area)
            for room, frames_with_loc in by_scene.items():
                if len(frames_with_loc) < args.min_views:
                    continue
                for _ in range(args.repeats):
                    k = int(rng.integers(args.min_views, args.max_views + 1))
                    sel = sample_covisible_group(frames_with_loc, k, rng,
                                                 bmin=args.baseline_min,
                                                 bmax=args.baseline_max)
                    if sel is not None and len(sel) >= args.min_views:
                        cases.append((area, sel))
    if not args.tuples_json:
        rng.shuffle(cases)
        if len(cases) > args.max_cases:
            cases = cases[:args.max_cases]
    if not args.keep_duplicates:
        # small rooms can be drawn twice with the same panoramas; keep the first draw
        seen, unique = set(), []
        for area, sel in cases:
            key = (area, frozenset(sel))
            if key not in seen:
                seen.add(key)
                unique.append((area, sel))
        if len(unique) < len(cases):
            print(f"[2d3ds-pose-v2] dropped {len(cases) - len(unique)} repeated case(s)")
        cases = unique
    nviews = [len(s) for _, s in cases]
    print(f"[2d3ds-pose-v2] areas={args.areas} cases={len(cases)} "
          f"views/case min/med/max={min(nviews)}/{int(np.median(nviews))}/{max(nviews)} "
          f"ckpts={[n for n, _p, _v in ckpts]}")

    model, current_variant = None, None

    records = {}
    for name, path, variant in ckpts:
        if model is None or variant != current_variant:
            if model is not None:
                del model
                torch.cuda.empty_cache()
            if variant:
                print(f"[model] build ar+{variant} (ar_prob=1, {VARIANT_OVERRIDES[variant]})")
                model = build_model_with_ar(args.device, variant)
            else:
                print("[model] build vanilla images_only")
                model = build_model_imagesonly(args.device)
            current_variant = variant
        if variant:
            load_variant_ckpt(model, path)
            print(f"  [{name}] ckpt loaded strictly (ar_encoder + wrap active)")
            if args.memory_efficient:
                print("  [warn] --memory-efficient unsupported by wrap-DPT head; forcing off for this ckpt")
        else:
            load_pth(model, path)
        me = args.memory_efficient and not variant
        rra30, rta30, aucs, n_ok = [], [], [], 0
        per_case = []
        for ci, (area, sel) in enumerate(cases):
            td = None
            try:
                paths = [rgb_path(area, b) for b in sel]
                if squeeze_ar is not None:
                    import tempfile
                    from PIL import Image as _Im
                    td = tempfile.mkdtemp(prefix="sqz2d3ds_")
                    sq = []
                    for pi, p in enumerate(paths):
                        with _Im.open(p) as im:
                            w0 = im.size[0]
                            out_p = os.path.join(td, f"{pi:02d}.png")
                            im.resize((w0, int(round(w0 / squeeze_ar))),
                                      _Im.BILINEAR).save(out_p)
                        sq.append(out_p)
                    paths = sq
                pred = predict_c2w_infer(model, paths, args.device, me,
                                         ar_mode=args.ar_mode,
                                         true_ar=args.true_ar)
                if args.save_preds:
                    d = args.save_preds / name
                    d.mkdir(parents=True, exist_ok=True)
                    np.savez(d / f"case_{ci:03d}.npz", c2w=np.stack(pred), frames=np.array(sel), area=area)
                gt = [load_gt_c2w(area, b) for b in sel]
                rerr, terr = eval_case(pred, gt)
                rra30.append((rerr < 30).mean())
                rta30.append((terr < 30).mean())
                auc, _ = auc_min(rerr, terr, 30)
                aucs.append(auc)
                n_ok += 1
                per_case.append({"area": area, "frames": list(sel), "rra30": float(rra30[-1]),
                                 "rta30": float(rta30[-1]), "auc30": float(auc)})
            except Exception as e:
                print(f"  [warn] case {area} n={len(sel)} failed: {e}")
                per_case.append({"area": area, "frames": list(sel), "failed": str(e)[:200]})
            finally:
                if td is not None:
                    import shutil
                    shutil.rmtree(td, ignore_errors=True)
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()
        records[name] = {
            "n_cases": n_ok,
            "RRA@30": float(np.mean(rra30) * 100) if rra30 else 0,
            "RTA@30": float(np.mean(rta30) * 100) if rta30 else 0,
            "AUC@30": float(np.mean(aucs)) if aucs else 0,
            "cases": per_case,
        }
        r = records[name]
        print(f"  {name:8s} cases={r['n_cases']:2d}  RRA@30={r['RRA@30']:.2f}  "
              f"RTA@30={r['RTA@30']:.2f}  AUC@30={r['AUC@30']:.2f}")

    out_json = args.out / "2d3ds_pose_v2_results.json"
    with open(out_json, "w") as f:
        json.dump({"areas": args.areas, "pipeline": "official load_images+model.infer",
                   "protocol": "Wid3R-2D3DS",
                   "tuples_json": str(args.tuples_json) if args.tuples_json else None,
                   "pre_squeeze": args.pre_squeeze, "ar_mode": args.ar_mode,
                   "keep_duplicates": args.keep_duplicates,
                   "true_ar": args.true_ar, "records": records,
                   "reference": {"Wid3R": {"RRA@30": 94.05, "RTA@30": 93.29, "AUC@30": 79.93},
                                 "VGGT": {"RRA@30": 19.30, "AUC@30": 2.60},
                                 "pi3": {"RRA@30": 19.94, "AUC@30": 2.06}}},
                  f, indent=2)
    print(f"\n[out] {out_json}")
    print("[ref] Wid3R 94.05/93.29/79.93 | VGGT 19.30/-/2.60 | pi3 19.94/-/2.06")


if __name__ == "__main__":
    main()
