#!/usr/bin/env python3
"""BLK hole provenance, cross-station fill, and ghost-suspect detection.

For every station: coverage; then decompose the holes:
  - fillable by other stations' measurements (cross-station fill: a measurement of one station
    fills a hole of another)
  - unfillable (no station ever measured it: glass/absorber/blind-cone/occlusion)
Ghost-suspect detector: station i holds a valid depth that contradicts free
space proven by neighbor j (j's surface at the same pixel is >0.30 m nearer).
Independent witnesses are counted per pixel (a single-witness contradiction
can be the witness's own virtual point; --min-witnesses defaults to 2), and
FLAG_GHOST_SUSPECT (bit 64) can be written into the pack flags channel
(--write-flags).

Consumes packs in the second-generation convention (rays channel + det=-1
c2w), i.e. exactly what a training/evaluation loader sees. Uses the GPU
(torch) when available.

Usage: python -m lenscope.blk.analyze_holes --scene <dir>
         [--agree 0.10] [--ghost-margin 0.30] [--min-witnesses 2]
         [--write-flags] [--device auto|cuda|cpu]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

FLAG_GHOST_SUSPECT = 64        # mirrors lenscope.core.spec (BLK inferred bit)


def _use_gpu(device: str) -> bool:
    if device == "cpu":
        return False
    try:
        import torch
        ok = torch.cuda.is_available()
    except ImportError:
        ok = False
    if device == "cuda" and not ok:
        raise RuntimeError("--device cuda requested but torch/cuda unavailable")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--agree", type=float, default=0.10)
    ap.add_argument("--ghost-margin", type=float, default=0.30)
    ap.add_argument("--min-witnesses", type=int, default=2)
    ap.add_argument("--write-flags", action="store_true",
                    help="write FLAG_GHOST_SUSPECT(64) into pack flags channel")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    args = ap.parse_args()
    scene = Path(args.scene)
    meta = json.loads((scene / "metadata.json").read_text())["frames"]
    n = len(meta)

    packs = [dict(np.load(scene / m["pack_file"])) for m in meta]
    depths = [p["depth"].astype(np.float32) for p in packs]
    H, W = depths[0].shape
    rays = packs[0]["rays"].astype(np.float32)      # second-generation convention grid
    rays /= np.maximum(np.linalg.norm(rays, axis=-1, keepdims=True), 1e-9)
    c2ws = [np.array(m["camera_to_world_unicol_4x4"]) for m in meta]

    any_valid = np.zeros((H, W), bool)
    for d in depths:
        any_valid |= d > 0
    rows = np.where(any_valid.any(1))[0]
    band = slice(int(rows.min()), int(rows.max()) + 1)

    gpu = _use_gpu(args.device)
    if gpu:
        import torch
        dev = torch.device("cuda")
        rays_t = torch.from_numpy(rays).to(dev)
        D = torch.from_numpy(np.stack(depths)).to(dev)
        Rs = torch.from_numpy(np.stack([c[:3, :3] for c in c2ws]).astype(np.float32)).to(dev)
        Rinv = torch.linalg.inv(Rs)
        ts = torch.from_numpy(np.stack([c[:3, 3] for c in c2ws]).astype(np.float32)).to(dev)

    report = {"n_stations": n, "band_rows": [int(rows.min()), int(rows.max())],
              "min_witnesses": args.min_witnesses, "ghost_margin": args.ghost_margin}
    fill_total, ghost_total = [], []
    for i in range(n):
        di = depths[i][band]
        hole = di == 0
        cov = 1.0 - hole.mean()
        bh = band.stop - band.start
        if gpu:
            import torch
            filled_t = torch.zeros((bh, W), dtype=torch.bool, device=dev)
            wit_t = torch.zeros((bh, W), dtype=torch.int16, device=dev)
            hole_t = torch.from_numpy(hole).to(dev)
            di_t = torch.from_numpy(di).to(dev)
            for j in range(n):
                if j == i:
                    continue
                vj = D[j] > 0
                Pw = (rays_t * D[j].unsqueeze(-1))[vj] @ Rs[j].T + ts[j]
                d = (Pw - ts[i]) @ Rinv[i].T
                r = torch.linalg.norm(d, dim=1)
                dn = d / r.clamp_min(1e-9).unsqueeze(1)
                az_p = torch.atan2(dn[:, 0], dn[:, 2])
                el = torch.asin(dn[:, 1].clamp(-1.0, 1.0))
                uu = ((az_p + np.pi) / (2 * np.pi) * W).long() % W
                vv = (((np.pi / 2 - el) / np.pi * H).long()).clamp(0, H - 1)
                inb = (vv >= band.start) & (vv < band.stop)
                uu, vv, r = uu[inb], vv[inb] - band.start, r[inb]
                h_idx = hole_t[vv, uu]
                filled_t[vv[h_idx], uu[h_idx]] = True
                val = di_t[vv, uu]
                gm = (val > 0) & (r < val - args.ghost_margin)
                pair_ghost = torch.zeros((bh, W), dtype=torch.bool, device=dev)
                pair_ghost[vv[gm], uu[gm]] = True          # dedupe within pair
                wit_t += pair_ghost.to(torch.int16)
            filled = filled_t.cpu().numpy()
            witnesses = wit_t.cpu().numpy()
        else:
            filled = np.zeros_like(hole)
            witnesses = np.zeros(di.shape, np.int16)
            for j in range(n):
                if j == i:
                    continue
                Rj, tj = c2ws[j][:3, :3], c2ws[j][:3, 3]
                vj = depths[j] > 0
                Pw = ((rays * depths[j][..., None])[vj] @ Rj.T) + tj
                d = (Pw - c2ws[i][:3, 3]) @ np.linalg.inv(c2ws[i][:3, :3]).T
                r = np.linalg.norm(d, axis=1)
                dn = d / np.maximum(r[:, None], 1e-9)
                az_p = np.arctan2(dn[:, 0], dn[:, 2])
                el = np.arcsin(np.clip(dn[:, 1], -1, 1))
                uu = ((az_p + np.pi) / (2 * np.pi) * W).astype(np.int64) % W
                vv = np.clip(((np.pi / 2 - el) / np.pi * H).astype(np.int64), 0, H - 1)
                inb = (vv >= band.start) & (vv < band.stop)
                uu, vv, r = uu[inb], vv[inb] - band.start, r[inb]
                h_idx = hole[vv, uu]
                filled[vv[h_idx], uu[h_idx]] = True
                val = di[vv, uu]
                gm = (val > 0) & (r < val - args.ghost_margin)
                pair_ghost = np.zeros(di.shape, bool)
                pair_ghost[vv[gm], uu[gm]] = True
                witnesses += pair_ghost.astype(np.int16)

        ghost = witnesses >= args.min_witnesses
        ghost_any = witnesses >= 1
        fill_rate = filled.sum() / max(hole.sum(), 1)
        ghost_rate = ghost.sum() / max((di > 0).sum(), 1)
        fill_total.append(fill_rate)
        ghost_total.append(ghost_rate)
        report[meta[i]["station"]] = {
            "coverage_band": round(float(cov), 4),
            "hole_fillable_by_others": round(float(fill_rate), 4),
            "residual_unfillable": round(float((hole & ~filled).mean()), 4),
            "ghost_suspect_rate": round(float(ghost_rate), 4),
            "ghost_rate_1witness": round(float(ghost_any.sum() / max((di > 0).sum(), 1)), 4),
        }
        print(f"[{meta[i]['station']}] cov={cov*100:.1f}%  holes fillable={fill_rate*100:.1f}%  "
              f"ghost-suspect={ghost_rate*100:.2f}% (1-witness "
              f"{ghost_any.sum() / max((di > 0).sum(), 1)*100:.2f}%)")

        if args.write_flags:
            fl = packs[i]["flags"].copy()
            gfull = np.zeros((H, W), bool)
            gfull[band] = ghost
            fl[gfull] |= FLAG_GHOST_SUSPECT
            packs[i]["flags"] = fl
            np.savez_compressed(scene / meta[i]["pack_file"], **packs[i])

    report["cross_station_fill_mean"] = round(float(np.mean(fill_total)), 4)
    report["ghost_mean"] = round(float(np.mean(ghost_total)), 4)
    (scene / "hole_analysis.json").write_text(json.dumps(report, indent=1))
    print(f"\ncross-station mean fillability: {np.mean(fill_total)*100:.1f}%  "
          f"| mean ghost-suspect(>={args.min_witnesses}w): {np.mean(ghost_total)*100:.2f}%"
          f"{'  | flags written (bit 64)' if args.write_flags else ''}  -> hole_analysis.json")


if __name__ == "__main__":
    main()
