#!/usr/bin/env python3
"""Quality gates for a converted BLK scene.

Run on every converted scene before it is used. Five gates (keys of validate.json):
  G1 pose sanity       metadata c2w det=-1 (second-generation convention),
                       orthonormal; with --station-dir also re-derives the pose
                       from the de-paged E57 XML and checks that it matches.
  G2 rays contract     pack rays == second-generation equirectangular grid
                       (max angle < 0.1 deg).
  G3 frame law         per station: the azimuth cross-correlation between the
                       cloud-colour panorama and the pack rgb must peak at
                       shift 0 (needs e57; catches panorama/cloud misalignment
                       that covisibility cannot detect).
  G4 covis self-check  reprojection with pack rays + metadata c2w (the exact
                       math a consumer of this convention runs) reproduces the
                       stored covisibility on the top pairs; connectivity report.
  G5 cyclone anchor    optional FinalizeReport.pdf link table vs the
                       computed covisibility (median |delta| threshold; the free
                       ground-truth anchor).

Usage:
  python -m lenscope.blk.validate_blk_scene --scene <dir>
        [--station-dir <dir with Setup*.e57>] [--report <FinalizeReport.pdf>]
Exit code 0 = all gates pass.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

from lenscope.blk.blk2pack import (
    cloud_to_erp, genesis_erp_rays, read_e57_pose, UNICOL_SWAP)


def _world_to_pix(P, c2w, W, H):
    R, t = c2w[:3, :3], c2w[:3, 3]
    d = (P - t) @ np.linalg.inv(R).T
    r = np.linalg.norm(d, axis=1)
    dn = d / np.maximum(r[:, None], 1e-9)
    az_p = np.arctan2(dn[:, 0], dn[:, 2])
    el = np.arcsin(np.clip(dn[:, 1], -1, 1))
    u = ((az_p + np.pi) / (2 * np.pi) * W).astype(np.int64) % W
    v = np.clip(((np.pi / 2 - el) / np.pi * H).astype(np.int64), 0, H - 1)
    return u, v, r


def gate(name, ok, detail):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    return bool(ok)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--station-dir", default=None)
    ap.add_argument("--report", default=None, help="FinalizeReport.pdf")
    ap.add_argument("--covis-pairs", type=int, default=3)
    ap.add_argument("--jobs", type=int, default=0,
                    help="parallel e57 readers for G3 (0 = min(8, cpus//4))")
    args = ap.parse_args()
    scene = Path(args.scene)
    meta = json.loads((scene / "metadata.json").read_text())["frames"]
    cov = np.load(scene / "covisibility" / "v0" / "covisibility.npy")
    n = len(meta)
    results, ok_all = {}, True

    # G1 pose sanity
    dets, ortho = [], []
    for m in meta:
        R = np.array(m["camera_to_world_unicol_4x4"])[:3, :3]
        dets.append(np.linalg.det(R))
        ortho.append(np.abs(R @ R.T - np.eye(3)).max())
    g1 = bool(max(abs(d + 1.0) for d in dets) < 1e-3 and max(ortho) < 1e-3)
    detail = f"det(R) in [{min(dets):+.5f},{max(dets):+.5f}] (want -1), ortho<={max(ortho):.1e}"
    if args.station_dir:
        sdir = Path(args.station_dir)
        worst = 0.0
        for m in meta:
            Rs, t = read_e57_pose(sdir / f"{m['station']}.e57")
            c2w = np.array(m["camera_to_world_unicol_4x4"])
            worst = max(worst,
                        float(np.abs(Rs @ UNICOL_SWAP - c2w[:3, :3]).max()),
                        float(np.abs(t - c2w[:3, 3]).max()))
        g1 = bool(g1 and worst < 1e-5)
        detail += f", xml-rederive dev {worst:.1e}"
    ok_all &= gate("G1 pose", g1, detail)
    results["G1"] = {"pass": g1, "detail": detail}

    # G2 rays contract
    p0 = np.load(scene / meta[0]["pack_file"])
    rp = p0["rays"].astype(np.float32)
    H, W = rp.shape[:2]
    rp /= np.maximum(np.linalg.norm(rp, axis=-1, keepdims=True), 1e-9)
    rg = genesis_erp_rays(W, H)
    ang = np.degrees(np.arccos(np.clip((rp * rg).sum(-1), -1, 1)))
    g2 = bool(float(ang.max()) < 0.1)
    ok_all &= gate("G2 rays", g2, f"max dev {ang.max():.4f} deg vs genesis grid")
    results["G2"] = {"pass": g2, "max_deg": float(ang.max())}

    # G3 frame law (needs e57)
    if args.station_dir:
        sdir = Path(args.station_dir)
        w2, h2 = 512, 256
        shifts = []
        jobs = args.jobs or min(8, max(1, (os.cpu_count() or 4) // 4))
        import multiprocessing as mp
        with mp.get_context("fork").Pool(min(jobs, len(meta))) as pool:
            erps = pool.starmap(cloud_to_erp, [(sdir / f"{m['station']}.e57", w2, h2)
                                               for m in meta])
        for m, (_d, cc, valid) in zip(meta, erps):
            pk = np.load(scene / m["pack_file"])
            rgb = (pk["rgb"].astype(np.float32).mean(-1))
            rgb = rgb.reshape(h2, rgb.shape[0] // h2, w2, -1).mean((1, 3))
            a = cc.astype(np.float32).mean(-1)
            va = valid & (a > 0)
            a = np.where(va, a, 0.0); b = np.where(va, rgb, 0.0)
            a -= va * a.sum() / max(va.sum(), 1); b -= va * b.sum() / max(va.sum(), 1)
            xc = np.fft.irfft(np.fft.rfft(a, axis=1).conj() * np.fft.rfft(b, axis=1),
                              n=w2, axis=1).sum(0)
            s = int(np.argmax(xc)); s = s - w2 if s > w2 // 2 else s
            shifts.append(s)
        g3 = bool(max(abs(s) for s in shifts) <= 1)
        ok_all &= gate("G3 frame-law", g3, f"az shifts px@512 = {shifts} (|s|<=1)")
        results["G3"] = {"pass": g3, "shifts_px": shifts}
    else:
        print("[SKIP] G3 frame-law (no --station-dir)")

    # G4 covisibility self-consistency + connectivity
    packs = [np.load(scene / m["pack_file"]) for m in meta]
    c2ws = [np.array(m["camera_to_world_unicol_4x4"]) for m in meta]
    off = cov.copy(); np.fill_diagonal(off, 0)
    flat = [(off[i, j], i, j) for i in range(n) for j in range(n) if i < j]
    flat.sort(reverse=True)
    devs = []
    for cv, i, j in flat[:args.covis_pairs]:
        di = packs[i]["depth"].astype(np.float32)
        dj = packs[j]["depth"].astype(np.float32)
        Ri, ti = c2ws[i][:3, :3], c2ws[i][:3, 3]
        vi = di > 0
        Pw = ((rp * di[..., None])[vi] @ Ri.T) + ti
        uu, vv, r = _world_to_pix(Pw, c2ws[j], W, H)
        db = dj[vv, uu]
        agree = float(((db > 0) & (np.abs(db - r) < 0.10)).mean())
        devs.append(abs(agree - float(cv)))
    isolated = [meta[i]["station"] for i in range(n) if n > 1 and off[i].max() < 0.25]
    g4 = bool((max(devs) if devs else 0) < 0.05)
    devs = [float(d) for d in devs]
    ok_all &= gate("G4 covis", g4,
                   f"consumer-math reprojection dev {[round(d,3) for d in devs]} (<0.05); "
                   f"isolated@0.25: {isolated or 'none'}")
    results["G4"] = {"pass": g4, "devs": devs, "isolated": isolated}

    # G5 cyclone anchor
    if args.report:
        import pypdf
        t = " ".join(pg.extract_text() for pg in pypdf.PdfReader(args.report).pages)
        t = re.sub(r"\s+", " ", t)
        links = sorted(set(re.findall(r"Setup (\d+) Setup (\d+) (\d+) %", t)))
        sid = {m["station"].split()[-1]: i for i, m in enumerate(meta)}
        deltas = [float(cov[sid[a], sid[b]]) - float(ov) / 100
                  for a, b, ov in links if a in sid and b in sid]
        med = float(np.median(np.abs(deltas))) if deltas else 1.0
        g5 = bool(med < 0.15 and len(deltas) > 0)
        ok_all &= gate("G5 cyclone", g5,
                       f"{len(deltas)} links, median |delta| {med:.3f} (<0.15)")
        results["G5"] = {"pass": g5, "n_links": len(deltas), "median_abs_delta": med}
    else:
        print("[SKIP] G5 cyclone (no --report)")

    (scene / "validate.json").write_text(json.dumps(results, indent=1))
    print(f"\n{'ALL GATES PASS' if ok_all else 'GATE FAILURE'} -> {scene/'validate.json'}")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
