#!/usr/bin/env python3
"""Full-completeness verifier for a production batch: check every scene and
backfill gaps in one final pass.

Per scene (from the blends dir = the intended set), checks:
  1. fixture: mesh.npz + objects.json exist and load;
  2. sample:  <scene>_sample.json exists, poses > 0;
  3. render:  PNG count == pose count (before bridging; the driver deletes PNGs afterwards);
  4. packs:   pack count == pose count; every pack loads with the 8-key
              schema {rgb,rays,depth,mask,sem,inst,flags,normal} and sane
              shapes/dtypes; depth>0 mean above floor; rgb non-constant;
  5. meta:    metadata.json frames == packs; covisibility matrix NxN.

Output: verdict per scene {complete | missing_* | corrupt_*} + a
--scenes list for the one-shot backfill rerun + optional --fix to delete
partial artifacts of broken scenes (so resume redoes them cleanly).

Usage:
  python lenscope/verify_batch_complete.py BLENDS_DIR WORK_ROOT [--spot 3]
      [--fix] [--out report.json]
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

KEYS = ("rgb", "rays", "depth", "mask", "sem", "inst", "flags", "normal")


def check_scene(name, blends, work, spot=3):
    fix = work / "fixtures" / name
    sj = work / "samples" / f"{name}_sample.json"
    rgbd = work / "render" / name
    shard = work / "shards" / "scenes" / name
    if not (fix / "mesh.npz").exists() or not (fix / "objects.json").exists():
        return "missing_fixture", {}
    if not sj.exists():
        return "missing_sample", {}
    try:
        poses = json.loads(sj.read_text())["poses"]
    except Exception as e:
        return "corrupt_sample", {"err": str(e)[:80]}
    n = len(poses)
    if n == 0:
        return "empty_sample", {}
    pngs = sorted(rgbd.glob("*_erp.png")) if rgbd.exists() else []
    packs_present = len(list(shard.glob("*_pack.npz"))) if shard.exists() else 0
    # the PNG count is only meaningful before bridging: production deletes
    # PNGs after a successful bridge (rgb lives in the pack as f16), so
    # complete packs without PNGs are by design, not a failure.
    if len(pngs) < n and packs_present < n:
        return "missing_render", {"png": len(pngs), "poses": n}
    packs = sorted(shard.glob("*_pack.npz")) if shard.exists() else []
    if len(packs) < n:
        return "missing_packs", {"packs": len(packs), "poses": n}
    # verify every pack loads with the full schema; spot-check pixel sanity
    for i, f in enumerate(packs):
        try:
            z = np.load(f)
            missing = [k for k in KEYS if k not in z.files]
            if missing:
                return "corrupt_pack_schema", {"file": f.name,
                                               "missing": missing}
            if i % max(1, len(packs) // max(spot, 1)) == 0:
                rgb, d = z["rgb"], z["depth"]
                if rgb.ndim != 3 or rgb.shape[2] != 3:
                    return "corrupt_pack_rgb", {"file": f.name,
                                                "shape": list(rgb.shape)}
                if float((d > 0).mean()) < 0.5:
                    return "corrupt_pack_depth", {"file": f.name,
                                                  "valid": float(
                                                      (d > 0).mean())}
                if float(np.asarray(rgb, np.float32).std()) < 1e-4:
                    return "corrupt_pack_flat_rgb", {"file": f.name}
        except Exception as e:
            return "corrupt_pack_load", {"file": f.name, "err": str(e)[:80]}
    meta = shard / "metadata.json"
    if not meta.exists():
        return "missing_metadata", {}
    try:
        frames = json.loads(meta.read_text())["frames"]
        if len(frames) != len(packs):
            return "meta_mismatch", {"frames": len(frames),
                                     "packs": len(packs)}
    except Exception as e:
        return "corrupt_metadata", {"err": str(e)[:80]}
    cov = shard / "covisibility" / "v0" / "covisibility.npy"
    if not cov.exists():
        return "missing_covis", {}
    c = np.load(cov)
    if c.shape != (len(packs), len(packs)):
        return "covis_mismatch", {"shape": list(c.shape)}
    return "complete", {"poses": n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("blends_dir")
    ap.add_argument("work_root")
    ap.add_argument("--spot", type=int, default=3)
    ap.add_argument("--fix", action="store_true",
                    help="delete partial artifacts of broken scenes so the "
                         "backfill rerun rebuilds them cleanly")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    blends = Path(a.blends_dir)
    work = Path(a.work_root)
    names = sorted(p.stem for p in blends.glob("genesis_*.blend"))
    rep, bad = {}, []
    for name in names:
        verdict, info = check_scene(name, blends, work, a.spot)
        rep[name] = {"verdict": verdict, **info}
        if verdict != "complete":
            bad.append(name)
            print(f"[{verdict}] {name} {info}")
    n_ok = len(names) - len(bad)
    print(f"== {n_ok}/{len(names)} scenes complete ==")
    if bad:
        print("backfill --scenes:")
        print(",".join(bad))
        if a.fix:
            for name in bad:
                v = rep[name]["verdict"]
                # packs of broken scenes are cheapest re-done whole (the bridge rebuilds them);
                # missing renders need no cleanup, rendering resumes per PNG
                if v.startswith(("corrupt_pack", "meta_mismatch",
                                 "covis_mismatch", "missing_metadata",
                                 "missing_covis")):
                    shutil.rmtree(work / "shards" / "scenes" / name,
                                  ignore_errors=True)
            print("[fix] partial shard dirs of corrupt scenes removed")
    out = a.out or str(work / "batch_completeness_report.json")
    Path(out).write_text(json.dumps(
        {"complete": n_ok, "total": len(names), "bad": bad,
         "detail": rep}, indent=1))
    print(f"report -> {out}")


if __name__ == "__main__":
    main()
