#!/usr/bin/env python3
"""Run the regular Hydra training entry point without writing checkpoints.

Used for short pipeline checks. Model construction, data loading,
forward, loss, backward, and optimizer updates are unchanged.
"""

from __future__ import annotations

import runpy

from mapanything.train import training


def _skip_checkpoint(*args, **kwargs):
    print("[smoke] checkpoint write skipped")


training.train_tools.save_model = _skip_checkpoint
training.save_final_model = _skip_checkpoint
runpy.run_path("scripts/train.py", run_name="__main__")
