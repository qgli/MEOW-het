"""Run an existing evaluation with aspect-ratio fusion disabled."""

import argparse
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party" / "uniception"))
from mapanything.models.mapanything.model import MapAnything


def without_aspect_ratio(
    self,
    views,
    num_views,
    batch_size_per_view,
    all_encoder_features_across_views,
):
    return all_encoder_features_across_views


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("target")
    args, remaining = parser.parse_known_args()
    MapAnything._encode_and_fuse_aspect_ratio = without_aspect_ratio
    print("[ar-null] aspect-ratio fusion disabled; panorama routing and wrap unchanged", flush=True)
    sys.argv = [args.target, *remaining]
    runpy.run_path(args.target, run_name="__main__")


if __name__ == "__main__":
    main()
