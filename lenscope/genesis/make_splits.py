#!/usr/bin/env python3
"""Deterministic train/val/test split of the second-generation scene folders in --scenes-dir.

The folders are sorted and shuffled with --seed; the first --val-count scenes form the validation split,
the next --test-count the test split and the rest the training split (469/15/15 for the 499 scenes of the
paper).
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--val-count", type=int, default=15)
    parser.add_argument("--test-count", type=int, default=15)
    args = parser.parse_args()

    scenes = sorted(
        path.name
        for path in args.scenes_dir.iterdir()
        if path.is_dir() and path.name.startswith("genesis_")
    )
    if len(scenes) < args.val_count + args.test_count + 1:
        raise ValueError(f"not enough scenes: {len(scenes)}")

    shuffled = scenes.copy()
    random.Random(args.seed).shuffle(shuffled)
    val = sorted(shuffled[: args.val_count])
    test = sorted(shuffled[args.val_count : args.val_count + args.test_count])
    train = sorted(shuffled[args.val_count + args.test_count :])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, values in (("train", train), ("val", val), ("test", test)):
        (args.out_dir / f"{name}.json").write_text(json.dumps(values))  # compact, as the files of the paper
    print(f"train={len(train)} val={len(val)} test={len(test)} seed={args.seed}")


if __name__ == "__main__":
    main()
