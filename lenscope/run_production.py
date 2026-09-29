#!/usr/bin/env python3
"""Production driver: unattended, multi-day full-chain rendering of second-generation scenes.

Runs the complete per-scene chain and is safe to leave running for days
unattended:

  per scene:  export_fixture -> sample -> render -> bridge -> [move HDR sidecars] -> [rm png]

Hardening (why each exists):
  * resume / skip-if-done  — a crash/reboot at scene 137 does not redo 0..136.
    Each stage checks for its own output; a scene whose bridged packs already
    match its pose count is skipped whole.
  * incremental manifest   — manifest.json is rewritten after every scene, so a
    hard kill still leaves an accurate record of what completed.
  * disk guard             — before each scene, if free < --min-free-gb, stop
    cleanly (a full disk silently corrupts packs).
  * per-scene isolation    — one bad .blend is logged and skipped; the batch
    continues.
  * PNG cleanup            — render PNGs are re-embedded into the bridged packs
    (f16 rgb), so they are deleted after a successful bridge unless --keep-png.
  * canary                 — --canary N renders only the first N scenes, then
    stops for a visual check before committing the machine to a multi-day run.

Blender stages (export, render) shell out to `blender -b`; sample/bridge are
Python subprocesses (sample_pose_graphs / bridge_v1_shard).

Usage (requires Blender and the .blend scenes):
  python lenscope/run_production.py \
      --blend-dir  <dir with *.blend> \
      --work-root  <output dir> \
      --n 200 --erp-w 2048 --samples 32 --min-free-gb 300 \
      [--canary 5] [--keep-png] [--scenes id,id,...]

Output layout under --work-root:
  fixtures/<scene>/{mesh.npz,objects.json}
  samples/<scene>_sample.json (+ _graph.html)
  render/<scene>/pose*_erp.png            (deleted post-bridge unless --keep-png)
  shards/scenes/<scene>/{*_pack.npz,metadata.json,covisibility/v0/}
  shards/splits/train.json                (accumulated, ready for training)
  manifest.json                           (incremental, authoritative progress)
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RENDER = ROOT / "lenscope" / "bpy" / "render_v2.py"
EXPORT = ROOT / "lenscope" / "fixtures" / "export_fixture.py"
SAMPLE = ROOT / "lenscope" / "sample_pose_graphs.py"
BRIDGE = ROOT / "lenscope" / "bridge_v1_shard.py"


def free_gb(path: Path) -> float:
    return shutil.disk_usage(path).free / 1e9


def n_packs(shard_scene: Path) -> int:
    return len(list(shard_scene.glob("*_pack.npz"))) if shard_scene.exists() else 0


def n_poses_of(sample_json: Path) -> int:
    try:
        return len(json.loads(sample_json.read_text())["poses"])
    except Exception:
        return -1


def run(cmd, log, timeout=None):
    """Subprocess with stdout/stderr captured to a per-scene log file."""
    with open(log, "a") as f:
        f.write(f"\n$ {' '.join(str(c) for c in cmd)}\n")
        f.flush()
        r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, timeout=timeout)
    return r.returncode


def main():
    # headless machines abort in the workbench-GL thumbnail (epoxy EGL
    # assert) after mesh.npz is written; skip thumbnails in production (they
    # are a visualization aid, not pipeline data).
    os.environ.setdefault("AGEN_NO_THUMB", "1")
    ap = argparse.ArgumentParser()
    ap.add_argument("--blend-dir", required=True)
    ap.add_argument("--work-root", required=True)
    ap.add_argument("--n", type=int, default=200, help="maximum number of new scenes in this run")
    ap.add_argument("--erp-w", type=int, default=2048)
    ap.add_argument("--samples", type=int, default=32)
    ap.add_argument("--exposure", type=float, default=0.0)
    ap.add_argument("--min-free-gb", type=float, default=300.0)
    ap.add_argument("--canary", type=int, default=0, help="stop after N scenes for human review")
    ap.add_argument("--keep-png", action="store_true")
    ap.add_argument("--scenes", default=None, help="comma list of explicit blend stems")
    ap.add_argument("--blender", default="blender")
    a = ap.parse_args()

    work = Path(a.work_root)
    fix_root = work / "fixtures"
    smp_root = work / "samples"
    rgb_root = work / "render"
    shard_root = work / "shards"
    for d in (fix_root, smp_root, rgb_root, shard_root / "scenes", shard_root / "splits"):
        d.mkdir(parents=True, exist_ok=True)
    man_path = work / "manifest.json"
    manifest = json.loads(man_path.read_text()) if man_path.exists() else {}

    blends = sorted(Path(a.blend_dir).glob("*.blend"))
    if a.scenes:
        want = set(a.scenes.split(","))
        blends = [b for b in blends if b.stem in want or b.stem.split("_")[-1] in want]
    print(f"[prod] {len(blends)} .blend candidates; work-root={work}; "
          f"free={free_gb(work):.0f}GB; target {a.n} new scenes"
          + (f"; canary {a.canary}" if a.canary else ""), flush=True)

    def save_manifest():
        man_path.write_text(json.dumps(manifest, indent=1))
        done = [k for k, v in manifest.items() if v.get("ok")]
        (shard_root / "splits" / "train.json").write_text(json.dumps(sorted(done), indent=1))

    new_done = 0
    for blend in blends:
        name = blend.stem
        shard_scene = shard_root / "scenes" / name
        sample_json = smp_root / f"{name}_sample.json"

        # resume: fully done scene (packs match poses) -> skip
        if manifest.get(name, {}).get("ok") and n_packs(shard_scene) == manifest[name].get("n_poses", -2):
            continue
        if a.n and new_done >= a.n:
            print(f"[prod] reached target {a.n} new scenes; stopping.", flush=True)
            break
        if a.canary and new_done >= a.canary:
            print(f"[prod] canary of {a.canary} scenes reached; stopping for review. "
                  f"Re-run without --canary to continue (resume is automatic).", flush=True)
            break
        # disk guard
        if free_gb(work) < a.min_free_gb:
            print(f"[prod] STOP: free {free_gb(work):.0f}GB < min {a.min_free_gb}GB. "
                  f"{new_done} scenes done this run.", flush=True)
            break

        t0 = time.time()
        log = work / f"log_{name}.txt"
        rec = {"ok": False, "stages": {}}
        try:
            fix_dir = fix_root / name
            # 1) export (Blender) — skip if mesh present
            if not (fix_dir / "mesh.npz").exists():
                rc = run([a.blender, "-b", str(blend), "--python", str(EXPORT),
                          "--", "--out", str(fix_root)], log)
                rec["stages"]["export"] = rc
                # mesh.npz presence is the real contract; a nonzero rc with
                # the dump complete (e.g. GL teardown quirks) is a warning
                if not (fix_dir / "mesh.npz").exists():
                    raise RuntimeError("export failed")
                if rc != 0:
                    print(f"[warn] {name}: export rc={rc} but mesh.npz ok",
                          flush=True)
            # 2) sample (CPU) — skip if sample present
            if not sample_json.exists():
                rc = run([sys.executable, str(SAMPLE), str(fix_root), str(smp_root),
                          "--only", name], log)
                rec["stages"]["sample"] = rc
                if not sample_json.exists():
                    raise RuntimeError("sample failed (no sample.json)")
            npose = n_poses_of(sample_json)
            rec["n_poses"] = npose
            # 3) render (Blender) — skip if all PNGs present
            scene_rgb = rgb_root / name
            have_png = len(list(scene_rgb.glob("*_erp.png"))) if scene_rgb.exists() else 0
            if have_png < npose:
                rc = run([a.blender, "-b", str(blend), "--python", str(RENDER),
                          "--", "--poses", str(sample_json), "--out", str(scene_rgb),
                          "--erp-w", str(a.erp_w), "--samples", str(a.samples),
                          "--pins", "", "--exposure", str(a.exposure)], log)
                rec["stages"]["render"] = rc
                have_png = len(list(scene_rgb.glob("*_erp.png")))
                if have_png < npose * 0.95:      # tolerate a few missing
                    raise RuntimeError(f"render short: {have_png}/{npose} PNG")
            # 4) bridge (CPU)
            rc = run([sys.executable, str(BRIDGE), str(fix_root), str(smp_root),
                      str(rgb_root), str(shard_root), "--W", str(a.erp_w),
                      "--only", name], log)
            rec["stages"]["bridge"] = rc
            got = n_packs(shard_scene)
            if got < npose * 0.95:
                raise RuntimeError(f"bridge short: {got}/{npose} packs")
            rec.update(ok=True, n_packs=got, secs=round(time.time() - t0, 1))
            # 4b) HDR sidecars (GENESIS_HDR_SIDECAR=1 in render_v2): move the
            # linear EXRs into the shard scene dir before the PNG dir is
            # dropped; they are a color-domain archive, not a render temp.
            hdrs = list(scene_rgb.glob("*_hdr.exr")) if scene_rgb.exists() else []
            for h in hdrs:
                shutil.move(str(h), str(shard_scene / h.name))
            if hdrs:
                rec["n_hdr"] = len(hdrs)
            # 5) disk economy: drop render PNGs (rgb is inside packs as f16)
            if not a.keep_png and scene_rgb.exists():
                shutil.rmtree(scene_rgb)
                rec["png_deleted"] = True
            print(f"[ok] {name}: {got}/{npose} packs in {rec['secs']}s, "
                  f"free {free_gb(work):.0f}GB", flush=True)
        except Exception as e:  # noqa: BLE001 — isolate; batch must survive
            rec["error"] = f"{type(e).__name__}: {e}"
            print(f"[ERR] {name}: {rec['error']} (see {log.name})", flush=True)
        manifest[name] = rec
        save_manifest()
        if rec["ok"]:
            new_done += 1

    save_manifest()
    ok = sum(1 for v in manifest.values() if v.get("ok"))
    print(f"[prod] done. {ok} scenes OK total; {new_done} new this run; "
          f"free {free_gb(work):.0f}GB. splits/train.json has {ok} scenes.", flush=True)


if __name__ == "__main__":
    main()
