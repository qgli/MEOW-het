#!/usr/bin/env python3
"""Write train/validation split files that contain one scene (pipeline check)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    payload = json.dumps([args.scene], indent=2) + "\n"
    (args.out_dir / "train.json").write_text(payload)
    (args.out_dir / "val.json").write_text(payload)
    print(f"train=1 val=1 scene={args.scene}")


if __name__ == "__main__":
    main()
