#!/usr/bin/env python3
"""Unified realset evaluator (mapanything env; CPU by default, optional GPU
pointmap scoring via --scoring-device).

Consumes tuples.json + gt/*.npz + preds/<model>/*.npz and scores every model
(MEOW and the baselines) with the same code:
  pose:     all-pairs relative pose -> RRA@30 / RTA@30 / AUC@30 (+mAA),
            imported from scripts/eval_2d3ds_pose.py (PoseDiffusion-style
            angular errors).
  pointmap: pixel-aligned correspondences (pred uv -> GT world xyz lookup)
            -> Umeyama -> least-squares scale and shift (moge_align) -> ICP
            -> Acc/Comp/N.C., imported from scripts/mp3d_benchmark/pointmap_eval.py
            (the MP3D point-map benchmark code, after the Wid3R §4.3 protocol).

Predictors only write raw predictions; every alignment/metric decision lives
here and is identical for all models.

Output: <out>/results_<model>.json + one combined json. Tuples whose pred npz
is missing are counted and reported (never silently dropped).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

THIS = Path(__file__).resolve()
sys.path.insert(0, str(THIS.parent))
sys.path.insert(0, str(THIS.parent.parent))                 # scripts/
sys.path.insert(0, str(THIS.parent.parent / "mp3d_benchmark"))
from common import load_tuples  # noqa: E402


def corr_from_uv(gt, uv, vidx):
    """Look up GT world xyz at predicted-point uv (fed-image pixels)."""
    xyz_ds, ds = gt["xyz_ds"], int(gt["ds"])
    if xyz_ds.size == 0:
        return None
    V, Hd, Wd = xyz_ds.shape[:3]
    ii = np.clip((uv[:, 1] / ds).round().astype(int), 0, Hd - 1)
    jj = np.clip((uv[:, 0] / ds).round().astype(int), 0, Wd - 1)
    g = xyz_ds[np.clip(vidx, 0, V - 1), ii, jj].astype(np.float64)
    return g


def eval_model(spec, gt_dir: Path, pred_dir: Path, do_pointmap: bool,
               scoring_device: str = "cpu", gt_cache: dict | None = None):
    from eval_2d3ds_pose import auc_min, eval_case, maa
    if do_pointmap:
        import pointmap_eval as pme
        use_gpu = scoring_device.startswith("cuda")
        if use_gpu:
            from pointmap_eval_gpu import estimate_normals_gpu, evaluate_gpu
        gate_checked = False
    rows, missing = [], 0
    for tup in spec["tuples"]:
        pnpz = pred_dir / f"{tup['id']}.npz"
        if not pnpz.is_file():
            missing += 1
            continue
        pred = np.load(pnpz, allow_pickle=False)
        gt = np.load(gt_dir / Path(tup["gt"]).name, allow_pickle=False)
        gt_c2w = gt["c2w"]
        pd_c2w = pred["c2w"]
        assert pd_c2w.shape == gt_c2w.shape, \
            f"{tup['id']}: pred views {pd_c2w.shape} != gt {gt_c2w.shape}"
        rerr, terr = eval_case(list(pd_c2w), list(gt_c2w))
        row = {"id": tup["id"], "V": int(len(gt_c2w)),
               "rra30": float((rerr < 30).mean()),
               "rta30": float((terr < 30).mean()),
               "auc30": float(auc_min(rerr, terr, 30)[0]),
               "maa30": float(maa(rerr, terr, 30))}
        if do_pointmap and gt["gt_pts"].size and pred["pts"].size:
            g = corr_from_uv(gt, pred["uv"], pred["vidx"])
            pts = pred["pts"].astype(np.float64)
            val = np.isfinite(g).all(1)
            if val.sum() >= 500:
                # pointmap_eval is used unchanged; degenerate geometry (e.g. a
                # broken prediction makes the MoGe residual scale overflow ->
                # NaN in ICP) is caught at the call site so one bad tuple
                # cannot abort the whole evaluation.
                try:
                    gt64 = gt["gt_pts"].astype(np.float64)
                    if do_pointmap and use_gpu:
                        # GT normals cached across models (identical inputs ->
                        # identical arrays); the CPU path does not use the cache
                        key = Path(tup["gt"]).name
                        if gt_cache is not None and key in gt_cache:
                            gn = gt_cache[key]
                        else:
                            gn = estimate_normals_gpu(gt64, scoring_device)
                            if gt_cache is not None:
                                gt_cache[key] = gn
                        m = evaluate_gpu(pts.copy(), gt64,
                                         scoring_device, pred_corr=pts[val],
                                         gt_corr=g[val], gt_normals=gn)
                        if not gate_checked:
                            # equivalence check: the model's first scored tuple
                            # is also scored on the CPU
                            m_cpu = pme.evaluate(pts.copy(), gt64,
                                                 pred_corr=pts[val],
                                                 gt_corr=g[val],
                                                 with_normals=True)
                            bad = [k for k in m_cpu
                                   if abs(m[k] - m_cpu[k]) >
                                   1e-6 * max(1.0, abs(m_cpu[k]))]
                            if bad:
                                print(f"[eval] !! GPU/CPU equivalence gate "
                                      f"FAILED on {bad} — falling back to CPU "
                                      f"for the rest of this model")
                                use_gpu = False
                                m = m_cpu
                            else:
                                print(f"[eval] GPU scoring gate PASS "
                                      f"(first tuple, all metrics ≤1e-6 rel)")
                            gate_checked = True
                    else:
                        m = pme.evaluate(pts.copy(), gt64,
                                         pred_corr=pts[val], gt_corr=g[val],
                                         with_normals=True)
                    if not all(np.isfinite(v) for v in m.values()):
                        raise FloatingPointError("non-finite metric")
                    row.update({"acc": m["Acc_mean"], "acc_med": m["Acc_med"],
                                "comp": m["Comp_mean"], "comp_med": m["Comp_med"],
                                "nc": m.get("NC_mean", float("nan")),
                                "n_corr": int(val.sum())})
                except Exception as e:
                    row["pointmap_failed"] = f"degenerate alignment: {e!r}"
            else:
                row["pointmap_skipped"] = f"only {int(val.sum())} valid corr"
        if "sec" in pred.files:
            row["sec"] = float(pred["sec"])
        if "vram_gb" in pred.files:
            row["vram_gb"] = float(pred["vram_gb"])
        rows.append(row)
    agg = {"n_cases": len(rows), "n_missing_preds": missing}
    timed = [r["sec"] for r in rows if "sec" in r]
    if len(timed) >= 2:
        agg["sec_mean"] = float(np.mean(timed[1:]))   # first tuple = warm-up
        agg["sec_p50"] = float(np.median(timed[1:]))
    vr = [r["vram_gb"] for r in rows if "vram_gb" in r and np.isfinite(r["vram_gb"])]
    if vr:
        agg["vram_gb_max"] = float(np.max(vr))
    if rows:
        for k, scale in [("rra30", 100), ("rta30", 100), ("auc30", 1),
                         ("maa30", 1)]:
            agg[k.upper()] = float(np.mean([r[k] for r in rows]) * scale)
        pm = [r for r in rows if "acc" in r]
        agg["n_pointmap"] = len(pm)
        agg["n_pointmap_failed"] = sum(1 for r in rows if "pointmap_failed" in r)
        if pm:
            for k in ("acc", "acc_med", "comp", "comp_med", "nc"):
                agg[k] = float(np.mean([r[k] for r in pm]))
    return agg, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tuples", required=True)
    ap.add_argument("--preds-root", required=True,
                    help="dir containing <model>/ subdirs of npz")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-pointmap", action="store_true")
    ap.add_argument("--scoring-device", default="cpu",
                    help="cpu (default, reference) | cuda:N — GPU pointmap "
                         "scoring (f32 candidates + exact-f64 rescore) with a "
                         "first-tuple CPU-equivalence gate (auto-fallback)")
    args = ap.parse_args()

    spec = load_tuples(args.tuples)
    tup_dir = Path(args.tuples).parent
    gt_dir = tup_dir / "gt"
    if not gt_dir.is_dir():                       # adt layout: gt beside out/
        gt_dir = tup_dir.parent / "gt"
    assert gt_dir.is_dir(), f"gt dir not found near {tup_dir}"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    combined = {"dataset": spec["dataset"], "track": spec.get("track", ""),
                "tuples": args.tuples, "models": {}}
    gt_cache = {}
    for m in args.models:
        pred_dir = Path(args.preds_root) / m
        assert pred_dir.is_dir(), f"missing preds for {m}: {pred_dir}"
        agg, rows = eval_model(spec, gt_dir, pred_dir,
                               do_pointmap=not args.no_pointmap,
                               scoring_device=args.scoring_device,
                               gt_cache=gt_cache)
        combined["models"][m] = agg
        with open(out / f"results_{m}.json", "w") as f:
            json.dump({"aggregate": agg, "per_tuple": rows}, f, indent=1)
        pm = (f"  Acc={agg.get('acc', float('nan')):.3f} "
              f"Comp={agg.get('comp', float('nan')):.3f} "
              f"NC={agg.get('nc', float('nan')):.3f} (n={agg.get('n_pointmap', 0)})"
              if agg.get("n_pointmap") else "  (pose-only)")
        print(f"[eval] {m:10s} n={agg['n_cases']:3d} miss={agg['n_missing_preds']:2d} "
              f"RRA@30={agg.get('RRA30', 0):6.2f} RTA@30={agg.get('RTA30', 0):6.2f} "
              f"AUC@30={agg.get('AUC30', 0):6.2f}{pm}")
    cj = out / f"combined_{spec['dataset']}_{spec.get('track', '')}.json"
    with open(cj, "w") as f:
        json.dump(combined, f, indent=1)
    print(f"[eval] combined -> {cj}")


if __name__ == "__main__":
    main()
