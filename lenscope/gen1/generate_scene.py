#!/usr/bin/env python3
"""Run the vendored first-generation scene generator inside Blender.

Example:
  blender -b --python lenscope/gen1/generate_scene.py -- \
      --seed 0 --output_dir work/gen1/scenes --n_scenes 1
"""

from __future__ import annotations

import sys
from pathlib import Path


DATA_GENERATOR = Path(__file__).resolve().parent / "scene_generator"
sys.path.insert(0, str(DATA_GENERATOR))

from scene.main_orchestrator import main  # noqa: E402


if __name__ == "__main__":
    main()
