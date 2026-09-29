#!/usr/bin/env python3
"""Run the fair-focal renderer ``render_fair_focal_dataset.py`` inside Blender.

The renderer imports its camera, ray and pack helpers as ``render_unicol_dataset``,
which lives in ``scene_generator/``. This wrapper puts that directory on ``sys.path``,
imports the helper module, and then runs ``render_fair_focal_dataset.py`` from this
directory as ``__main__``. Arguments after ``--`` are passed through unchanged:

  blender -b --python lenscope/gen1/render_fair_focal.py -- \\
      --pick work/gen1/scene_pick.json --out work/gen1/packs
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "scene_generator"))
import render_unicol_dataset  # noqa: E402,F401  (camera, ray and pack helpers used by the renderer)

runpy.run_path(str(HERE / "render_fair_focal_dataset.py"), run_name="__main__")
