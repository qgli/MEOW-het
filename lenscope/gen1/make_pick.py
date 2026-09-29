#!/usr/bin/env python3
"""Create the small scene list consumed by the fair-focal renderer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--blend-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--scenes", nargs="*", default=None)
    args = parser.parse_args()

    if args.scenes:
        blends = [args.blend_dir / f"{name}.blend" for name in args.scenes]
    else:
        blends = sorted(args.blend_dir.glob("proc_scene_*.blend"))
    missing = [str(path) for path in blends if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing blend files: " + ", ".join(missing))

    payload = {"top": [{"blend_path": str(path.resolve())} for path in blends]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {len(blends)} scenes to {args.out}")


if __name__ == "__main__":
    main()
