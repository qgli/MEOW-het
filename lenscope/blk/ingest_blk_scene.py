#!/usr/bin/env python3
"""BLK scene ingestion: one command from an uploaded capture to a registered library entry.

Accepts either archive form:
  - result archive (blk2pack and validation already run on the capture side):
    contains metadata.json + blk_pose*_pack.npz [+ validate.json + hole_analysis.json]
  - raw archive: contains Setup*.e57 (+pano+txt+FinalizeReport.pdf) and runs
    the full pipeline (blk2pack --jobs/--device, the five validation checks,
    hole analysis).

Every ingested scene is validated again here regardless of origin
(validation is cheap without --station-dir; raw archives get all five
checks), then appended to the scene registry for cross-scene analysis.

Usage:
  python -m lenscope.blk.ingest_blk_scene --tar <scene.tar> --library <dir> [--scene-id X]
  python -m lenscope.blk.ingest_blk_scene --dir <scene_dir> --library <dir> [--scene-id X]
Options: --jobs N --device auto|cuda|cpu
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tarfile
from datetime import datetime
from pathlib import Path

import numpy as np


def sh(cmd):
    print(f"[ingest] $ {' '.join(str(c) for c in cmd)}", flush=True)
    return subprocess.run([str(c) for c in cmd]).returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tar", default=None)
    ap.add_argument("--dir", default=None)
    ap.add_argument("--scene-id", default=None)
    ap.add_argument("--library", required=True)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    assert bool(args.tar) ^ bool(args.dir), "give exactly one of --tar / --dir"
    lib = Path(args.library)
    lib.mkdir(parents=True, exist_ok=True)

    if args.tar:
        tar = Path(args.tar)
        sid = args.scene_id or tar.stem.replace(".tar", "")
        src = lib / sid / "raw"
        src.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tar) as tf:
            tf.extractall(src)
        # tolerate one level of wrapping dir inside the tar
        entries = [p for p in src.iterdir() if not p.name.startswith(".")]
        if len(entries) == 1 and entries[0].is_dir():
            src = entries[0]
    else:
        src = Path(args.dir)
        sid = args.scene_id or src.name

    e57s = sorted(src.rglob("*.e57"))
    has_packs = bool(list(src.rglob("blk_pose*_pack.npz")))
    pdfs = sorted(src.rglob("FinalizeReport.pdf"))
    py = [sys.executable, "-m"]

    if has_packs and not e57s:                      # result archive
        scene = next(p.parent for p in src.rglob("metadata.json"))
        print(f"[ingest] {sid}: result archive (packs pre-built on capture side)")
        if not (scene / "validate.json").exists():
            print("[ingest] WARNING: no validate.json in result archive")
        cmd = py + ["lenscope.blk.validate_blk_scene", "--scene", scene]
        if pdfs:
            cmd += ["--report", pdfs[0]]
        rc = sh(cmd)
    elif e57s:                                       # raw archive
        scene = lib / sid / "scene"
        print(f"[ingest] {sid}: raw archive, {len(e57s)} stations -> full pipeline")
        rc = sh(py + ["lenscope.blk.blk2pack", "--station-dir", e57s[0].parent,
                      "--out", scene, "--jobs", args.jobs, "--device", args.device])
        assert rc == 0, "blk2pack failed"
        cmd = py + ["lenscope.blk.validate_blk_scene", "--scene", scene,
                    "--station-dir", e57s[0].parent, "--jobs", args.jobs]
        if pdfs:
            cmd += ["--report", pdfs[0]]
        rc = sh(cmd)
        sh(py + ["lenscope.blk.analyze_holes", "--scene", scene,
                 "--device", args.device])
    else:
        sys.exit(f"[ingest] {sid}: neither packs nor e57 found under {src}")

    if rc != 0:
        sys.exit(f"[ingest] {sid}: GATE FAILURE -- not registered (fix or recapture)")

    meta = json.loads((scene / "metadata.json").read_text())["frames"]
    cov = np.load(scene / "covisibility" / "v0" / "covisibility.npy")
    off = cov[~np.eye(len(meta), dtype=bool)]
    holes = {}
    hj = scene / "hole_analysis.json"
    if hj.exists():
        h = json.loads(hj.read_text())
        holes = {"cross_station_fill_mean": h.get("cross_station_fill_mean"),
                 "ghost_mean": h.get("ghost_mean")}
    validate = {}
    vj = scene / "validate.json"
    if vj.exists():
        validate = {g: r.get("pass") for g, r in json.loads(vj.read_text()).items()}

    reg_path = lib / "registry.json"
    reg = json.loads(reg_path.read_text()) if reg_path.exists() else {}
    reg[sid] = {
        "scene_dir": str(scene),
        "n_stations": len(meta),
        "covis_offdiag": {"min": round(float(off.min()), 3),
                          "median": round(float(np.median(off)), 3),
                          "max": round(float(off.max()), 3)},
        "gates": validate, **holes,
        "ingested": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    reg_path.write_text(json.dumps(reg, indent=1, ensure_ascii=False))
    print(f"[ingest] {sid}: REGISTERED -> {reg_path} ({len(reg)} scenes total)")


if __name__ == "__main__":
    main()
