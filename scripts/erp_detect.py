#!/usr/bin/env python3
"""Blind detector of full equirectangular panoramas (ERP) from the pixels alone.

A full equirectangular panorama is horizontally periodic: its leftmost and
rightmost columns are the same scene longitude. That property is invariant
under any aspect-ratio squeeze (a 2:1 ERP delivered as 16:9, e.g. by a
360-camera live stream, is still edge-continuous). A perspective or fisheye
crop has no such continuity.

detect_full_erp(img) -> (is_pano, info), computed on a copy resized to 128 rows:
  seam_ratio   = mean absolute difference between the last and first columns,
                 divided by the 90th percentile of the interior adjacent-column
                 differences (about 1 for a full ERP, much larger otherwise).
  pole_ratio   = horizontal colour std of the top and bottom rows divided by
                 that of the middle rows (small for a full ERP).
  edge_support = fraction of the rows in the central half whose left and right
                 1%-wide edge strips are both mostly non-black (guard against
                 circular fisheyes with black corners).
  is_pano = edge_support >= 0.5 and seam_ratio < seam_thresh (3.0)
            and pole_ratio < pole_thresh (0.55).

Once a full ERP is detected, its content aspect ratio is 2:1 by definition
(360 x 180 degrees) whatever the delivered frame shape, so the aspect-ratio
input can be set to 2.0 and pano_wrap to True (the embedding encodes the
content aspect ratio; the squeeze deformation is within the training
distribution).

Import-light: numpy + PIL only.
"""
from __future__ import annotations

import numpy as np


def detect_full_erp(img: np.ndarray, seam_thresh: float = 3.0,
                    pole_thresh: float = 0.55):
    """img: HxWx3 uint8 (or float). Returns (is_pano, info).

    Cue 1 (seam as an adjacent column pair): in a full ERP the last and first
    columns are adjacent in content, so their difference should sit within the
    distribution of interior adjacent-column differences (ratio ~ 1, robustly
    scored against the p90 of interior steps). A non-panorama's two edges are
    opposite ends of the FOV (ratio >> 1). Squeeze-invariant.

    Cue 2 (pole convergence): ERP top/bottom rows compress all longitudes into
    near-constant colour -> horizontal std of pole rows << mid rows. Also
    squeeze-invariant; fires trivially on panoramas with black poles (2D3DS).

    Black-circle guard: over the central half of the image, both 1%-wide
    edge strips must contain non-black pixels (max RGB > 5) in at least half
    of the rows.  Otherwise the image is directly classified as non-ERP.

    is_pano = edge_support and cue1 and cue2 (conservative: false positives
    are worse than false negatives when overriding the aspect ratio)."""
    from PIL import Image
    a = np.asarray(img)
    if a.dtype != np.uint8:
        a = np.clip(a * (255.0 if a.max() <= 1.5 else 1.0), 0, 255).astype(np.uint8)
    h0, w0 = a.shape[:2]
    H = 128
    W = max(32, int(round(w0 * H / h0)))
    a = np.asarray(Image.fromarray(a).resize((W, H), Image.BILINEAR)).astype(np.float32)
    # Black image-circle guard.  A circular fisheye can make both ERP cues
    # pass trivially because its side and pole regions are all black.
    edge_k = max(2, W // 100)
    row_band = a[H // 4:3 * H // 4]
    nonblack = row_band.max(axis=2) > 5.0
    left_rows = nonblack[:, :edge_k].mean(axis=1) > 0.5
    right_rows = nonblack[:, -edge_k:].mean(axis=1) > 0.5
    edge_support = float(np.mean(left_rows & right_rows))
    # cue 1: adjacent-column step statistics incl. the wrap step
    steps = np.abs(np.diff(a, axis=1)).mean(axis=(0, 2))          # W-1 interior steps
    wrap_step = float(np.abs(a[:, 0] - a[:, -1]).mean())
    p90 = float(np.percentile(steps, 90))
    seam_ratio = wrap_step / max(p90, 1e-6)
    # cue 2: pole horizontal-std vs mid horizontal-std
    k = max(2, H // 16)
    pole_std = float(np.mean([a[:k].std(axis=1).mean(), a[-k:].std(axis=1).mean()]))
    mid_std = float(a[H // 2 - k:H // 2 + k].std(axis=1).mean())
    pole_ratio = pole_std / max(mid_std, 1e-6)
    edge_support_thresh = 0.5
    is_pano = bool(edge_support >= edge_support_thresh and
                   seam_ratio < seam_thresh and pole_ratio < pole_thresh)
    return is_pano, {"seam_ratio": seam_ratio, "pole_ratio": pole_ratio,
                     "edge_support": edge_support,
                     "edge_support_thresh": edge_support_thresh,
                     "seam_thresh": seam_thresh, "pole_thresh": pole_thresh}


if __name__ == "__main__":
    import sys
    from PIL import Image
    for p in sys.argv[1:]:
        im = np.asarray(Image.open(p).convert("RGB"))
        ok, info = detect_full_erp(im)
        print(f"{p}: pano={ok} seam_ratio={info['seam_ratio']:.3f} "
              f"pole_ratio={info['pole_ratio']:.3f} "
              f"edge_support={info['edge_support']:.3f}")
