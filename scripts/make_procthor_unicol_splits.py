"""Generate train/val splits for the first-generation scene packs.

Reads the list of ``proc_scene_*`` folders under ``<root>/scenes`` and writes
``train.json`` / ``val.json`` (sorted lists of scene folder names) to the sibling
``<root>/splits/`` directory using a fixed-seed 80/20 random split.

Idempotent: running again with the same seed and the same set of scenes
produces identical splits. If new scenes have been added, run with ``--force``
to regenerate; otherwise exits without overwriting existing split files.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True,
                   help="Renders root containing scenes/ (proc_scene_* folders); "
                        "train.json and val.json are written to <root>/splits/.")
    p.add_argument("--val_ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=20260513)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    scenes_dir = args.root / "scenes"
    splits_dir = args.root / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    scenes = sorted(
        d.name
        for d in scenes_dir.iterdir()
        if d.is_dir() and d.name.startswith("proc_scene_")
    )
    if not scenes:
        sys.exit(f"No scene_* dirs in {scenes_dir}")

    train_path = splits_dir / "train.json"
    val_path = splits_dir / "val.json"

    if not args.force and (train_path.exists() or val_path.exists()):
        sys.exit(
            f"Refusing to overwrite existing splits in {splits_dir}; pass --force."
        )

    rng = random.Random(args.seed)
    shuffled = scenes.copy()
    rng.shuffle(shuffled)
    n_val = max(1, round(len(shuffled) * args.val_ratio))
    val_scenes = sorted(shuffled[:n_val])
    train_scenes = sorted(shuffled[n_val:])

    train_path.write_text(json.dumps(train_scenes, indent=2))
    val_path.write_text(json.dumps(val_scenes, indent=2))

    print(f"[OK] {len(scenes)} scenes -> {len(train_scenes)} train + {len(val_scenes)} val")
    print(f"     train: {train_path}")
    print(f"     val:   {val_path}")
    print(f"     seed:  {args.seed}")


if __name__ == "__main__":
    main()
