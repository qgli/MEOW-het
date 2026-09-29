#!/usr/bin/env python3
"""Precompute the per-material brightness statistics of the asset library.

The scene builder reads two caches in the library root, .set_means_cache.json (mean linear luminance of
each texture set's colour map) and .set_p95_cache.json (its 95th percentile). A missing entry is computed
on the fly with Pillow; without Pillow (Blender's bundled Python does not ship it) the builder falls back
to constants (mean 0.45, p95 0.8). The statistics decide which texture sets pass the brightness filters
and how each set is tinted, so with the fallback the same seed yields different material choices. Run
this script once, with a Python that has Pillow and NumPy, before building scenes.

Usage:
  python lenscope/genesis/assets/compute_set_stats.py --root /path/to/genesis-assets
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from lenscope.genesis import materials  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", required=True, help="asset library directory")
    a = ap.parse_args()
    try:
        import numpy  # noqa: F401
        from PIL import Image  # noqa: F401
    except ImportError as e:
        sys.exit(f"Pillow and NumPy are required: {e}")

    root = Path(a.root)
    for name in (".set_means_cache.json", ".set_p95_cache.json"):
        (root / name).unlink(missing_ok=True)       # derived data: always recomputed
    lib = materials.scan_library(str(root))
    n = 0
    for sets in lib["pbr"].values():
        for rel in sets:
            materials.set_albedo_mean(str(root), rel)
            materials.set_albedo_p95(str(root), rel)
            n += 1
    means = json.loads((root / ".set_means_cache.json").read_text()) if n else {}
    print(f"[set-stats] {n} texture sets; mean luminance range "
          f"{min(means.values(), default=0):.4f}-{max(means.values(), default=0):.4f}")


if __name__ == "__main__":
    main()
