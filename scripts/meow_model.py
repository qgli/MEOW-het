"""Build the MEOW model for inference and load a checkpoint.

The model is MapAnything with the aspect-ratio embedding (``ar_prob=1``); the ``"wrap"``
variant also enables the panorama wrap of the dense head, as used by the final model.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mapanything.models import _init_hydra_config, init_model  # noqa: E402

VARIANT_OVERRIDES = {
    "": [],
    # aspect-ratio embedding only (checkpoints trained without the panorama wrap)
    "ar": [],
    # aspect-ratio embedding and panorama wrap (the final MEOW model)
    "wrap": ["++model.model_config.pano_wrap_dpt=true"],
}


def build_model_with_ar(device: str, variant: str = ""):
    """Build MapAnything with ar_prob=1.0 (so the aspect-ratio encoder exists)."""
    cfg = _init_hydra_config(
        "configs/train.yaml",
        overrides=[
            "model=mapanything",
            "machine=default",
            "model/task=aug_training",
            "++model.task.ar_prob=1.0",
        ]
        + VARIANT_OVERRIDES[variant],
    )
    model = init_model(
        model_str=cfg.model.model_str,
        model_config=cfg.model.model_config,
        torch_hub_force_reload=False,
    )
    return model.to(device).eval()


def load_ckpt_inplace(model, sd):
    """Load a checkpoint state dict; only the aspect-ratio encoder may be missing."""
    incompat = model.load_state_dict(sd, strict=False)
    bad_missing = [k for k in incompat.missing_keys if not k.startswith("ar_encoder.")]
    if bad_missing:
        raise RuntimeError(f"Unexpected missing keys: {bad_missing[:8]}")
    if incompat.unexpected_keys:
        raise RuntimeError(f"Unexpected keys: {incompat.unexpected_keys[:8]}")
    return len(incompat.missing_keys)
