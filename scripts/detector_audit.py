#!/usr/bin/env python3
"""Count the decisions of the full-panorama detector (scripts/erp_detect.py) on an evaluation set.

The detector runs on the original RGB of every view, exactly as the evaluators call it in automatic
panorama routing. The ground-truth kind of a view comes from the set definition; a view counts as a
panorama when its kind is "erp". Sets whose tuple file or manifest carries no per-view kind (e.g. the
Replica and ADT tuple files, one camera type per file) take it from --default-kind; a kind given in the
file always takes precedence.

Inputs (one of):
  --tuples T.json [--frames-root DIR]   tuple file of the real-image sets and the laser benchmark
                                        (views with "img" and "kind", or "img" only with --default-kind;
                                        frames_root from the file unless overridden)
  --cases-root DIR                      exported heterogeneous 2D3DS cases (manifest.json with
                                        per-case "images" and "kinds")
  --images FILE [FILE ...] --kind KIND  loose images of a single kind (e.g. all panoramas of a set)
  --default-kind KIND                   kind of the --tuples / --cases-root views that have none
                                        (without it, a view without a kind is an error)

Prints the counts (views, panoramas missed, other views flagged, per kind) and optionally writes them
with the per-view decisions to --out.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from erp_detect import detect_full_erp  # noqa: E402


def _kind(kind, default_kind, where):
    if kind is None:
        kind = default_kind
    if kind is None:
        sys.exit(f"{where}: no kind; pass --default-kind KIND")
    return kind


def views_from_tuples(path, frames_root=None, default_kind=None):
    spec = json.loads(Path(path).read_text())
    root = Path(frames_root or spec["frames_root"])
    if not root.is_absolute():             # relative frames_root: relative to the folder of the tuple file
        root = Path(path).resolve().parent / root
    for tup in spec["tuples"]:
        for v in tup["views"]:
            kind = _kind(v.get("kind"), default_kind, f"tuple {tup.get('id')} view {v['img']}")
            yield str(root / v["img"]), kind, tup.get("id")


def views_from_cases(root, default_kind=None):
    root = Path(root)
    man = json.loads((root / "manifest.json").read_text())
    for case in man["cases"]:
        kinds = case.get("kinds") or [None] * len(case["images"])
        for img, kind in zip(case["images"], kinds):
            kind = _kind(kind, default_kind, f"case {case.get('case_id')} image {img}")
            yield str(root / img), kind, case.get("case_id")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--tuples")
    src.add_argument("--cases-root")
    src.add_argument("--images", nargs="+")
    ap.add_argument("--frames-root", default=None)
    ap.add_argument("--kind", default=None, help="kind of all --images")
    ap.add_argument("--default-kind", default=None, metavar="KIND",
                    help="kind of the --tuples / --cases-root views that have none (a kind in the file wins)")
    ap.add_argument("--name", default=None, help="set name for the printed summary")
    ap.add_argument("--out", default=None, help="optional JSON with counts and per-view decisions")
    a = ap.parse_args()

    if a.tuples:
        views = list(views_from_tuples(a.tuples, a.frames_root, a.default_kind))
    elif a.cases_root:
        views = list(views_from_cases(a.cases_root, a.default_kind))
    else:
        if not a.kind:
            ap.error("--images needs --kind")
        views = [(p, a.kind, None) for p in a.images]

    per_view, missed, flagged = [], Counter(), Counter()
    n_by_kind = Counter()
    for path, kind, group in views:
        with Image.open(path) as im:
            is_pano, info = detect_full_erp(np.asarray(im.convert("RGB")))
        true_pano = kind == "erp"
        n_by_kind[kind] += 1
        if true_pano and not is_pano:
            missed[kind] += 1
        if not true_pano and is_pano:
            flagged[kind] += 1
        per_view.append(dict(group=group, image=Path(path).name, kind=kind, detected_panorama=bool(is_pano),
                             **{k: round(float(v), 4) for k, v in info.items()}))

    n_pano = n_by_kind["erp"]
    n_other = sum(n for k, n in n_by_kind.items() if k != "erp")
    summary = dict(set=a.name, views=len(per_view), panoramas=n_pano,
                   panoramas_missed=sum(missed.values()), other_views=n_other,
                   other_views_flagged=sum(flagged.values()),
                   views_by_kind=dict(n_by_kind), flagged_by_kind=dict(flagged))
    print(f"{a.name or 'set'}: {summary['views']} views; missed {summary['panoramas_missed']} / {n_pano} "
          f"panoramas; flagged {summary['other_views_flagged']} / {n_other} other views "
          f"{dict(flagged) if flagged else ''}")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(dict(summary=summary, views=per_view), indent=1))


if __name__ == "__main__":
    main()
