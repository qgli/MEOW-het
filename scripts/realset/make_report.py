#!/usr/bin/env python3
"""Aggregate combined_*.json files from eval_realset into one markdown report
(rows = models, cols = pose + pointmap metrics, one table per dataset/track,
per-model case counts and missing-pred counts always shown)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ORDER = ["MEOW", "MapAnything", "MA", "VGGT", "pi3", "DUSt3R", "MASt3R"]


def fmt(v, nd=2):
    return "—" if v is None else f"{v:.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--combined", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="Real-set competitor comparison")
    ap.add_argument("--timing-json", default=None,
                    help='efficiency overrides {model:{"sec":..,"vram_gb":..}} '
                         "for models whose preds carry no timing fields "
                         "(e.g. timed separately on a --limit subset)")
    args = ap.parse_args()
    timing = {}
    if args.timing_json:
        with open(args.timing_json) as f:
            timing = json.load(f)

    lines = [f"# {args.title}", "",
             "Pose = all-pairs relative pose (RRA/RTA/AUC@30, PoseDiffusion "
             "angles, math imported from eval_2d3ds_pose). Pointmap = "
             "Umeyama, least-squares scale and shift, then ICP; Acc/Comp/N.C. (math imported from "
             "mp3d_benchmark/pointmap_eval, Wid3R §4.3). Identical eval code "
             "for every row; predictors use each model's official "
             "preprocessing+inference.", ""]
    for cpath in args.combined:
        with open(cpath) as f:
            c = json.load(f)
        models = c["models"]
        names = [m for m in ORDER if m in models] + \
                [m for m in sorted(models) if m not in ORDER]
        has_pm = any(models[m].get("n_pointmap") for m in names)
        def eff(m):
            t = dict(timing.get(m, {}))
            a = models[m]
            t.setdefault("sec", a.get("sec_mean"))
            t.setdefault("vram_gb", a.get("vram_gb_max"))
            return t
        has_eff = any(eff(m).get("sec") is not None for m in names)
        lines.append(f"## {c['dataset']} ({c.get('track', '')})")
        lines.append("")
        hdr = "| model | cases | RRA@30↑ | RTA@30↑ | AUC@30↑ |"
        sep = "|---|---|---|---|---|"
        if has_pm:
            hdr += " Acc↓ | Comp↓ | N.C.↑ | n_pm |"
            sep += "---|---|---|---|"
        if has_eff:
            hdr += " s/tuple↓ | VRAM(GB) |"
            sep += "---|---|"
        lines += [hdr, sep]
        for m in names:
            a = models[m]
            row = (f"| {m} | {a['n_cases']}"
                   f"{'(' + str(a['n_missing_preds']) + ' miss)' if a['n_missing_preds'] else ''} "
                   f"| {fmt(a.get('RRA30'))} | {fmt(a.get('RTA30'))} "
                   f"| {fmt(a.get('AUC30'))} |")
            if has_pm:
                row += (f" {fmt(a.get('acc'), 3)} | {fmt(a.get('comp'), 3)} "
                        f"| {fmt(a.get('nc'), 3)} | {a.get('n_pointmap', 0)} |")
            if has_eff:
                e = eff(m)
                row += f" {fmt(e.get('sec'))} | {fmt(e.get('vram_gb'), 1)} |"
            lines.append(row)
        if has_eff:
            lines.append("")
            lines.append("*Efficiency: same GPU, first tuple excluded as "
                         "warm-up; DUSt3R/MASt3R times include global-alignment "
                         "optimization (their per-paradigm cost).*")
        lines.append("")
    Path(args.out).write_text("\n".join(lines))
    print(f"[report] {len(args.combined)} tables -> {args.out}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
