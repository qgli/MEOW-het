"""Resampling a view from a source camera image into a target camera model (torch, GPU or CPU).

resample_from_camera maps every target pixel to the source in one step: the target model's unprojection
gives an exact unit ray, which is rotated into the source camera frame and projected through the source
model; RGB, depth and mask are sampled there once. The target rays are analytic, and the new camera shares
the source optical centre, so its pose is the source pose rotated by R (world-preserving).
target_coverage returns the fraction of target pixels that land on valid source pixels.

Camera models and parameters follow mapanything.datasets.camera_models. Computation uses the current default
dtype (float64 in the camera sampler).
"""
import torch

def _renorm(v, eps=1e-12):
    return v / v.norm(dim=-1, keepdim=True).clamp_min(eps)


# ----------------------------------------------------------------------------- sampling
def _grid_sample(img, col, row, H, W, wrap_col=False, mode="bilinear"):
    """Sample img (H,W) or (H,W,C) at float (col,row) [pixel coords], bilinear, edge-clamped (no black).
    wrap_col pads one wrapped column on each side so that panorama samples wrap across the longitude seam.
    Returns sampled (out_h,out_w[,C])."""
    if img.dim() == 2:
        inp = img[None, None].to(torch.get_default_dtype()); chan = 1       # 1,1,H,W
    else:
        inp = img.permute(2, 0, 1)[None].to(torch.get_default_dtype()); chan = img.shape[2]  # 1,C,H,W
    if wrap_col:                                                 # pad left/right with wrapped columns
        inp = torch.cat([inp[..., -1:], inp, inp[..., :1]], dim=-1)
        col = col + 1.0; Wp = W + 2
    else:
        Wp = W
    gx = (col / (Wp - 1)) * 2 - 1
    gy = (row / (H - 1)) * 2 - 1
    grid = torch.stack([gx, gy], dim=-1)[None].to(torch.get_default_dtype())  # 1,oh,ow,2
    out = torch.nn.functional.grid_sample(inp, grid, mode=mode, padding_mode="border",
                                          align_corners=True)[0]  # C,oh,ow
    return out[0] if chan == 1 else out.permute(1, 2, 0)


# ----------------------------------------------------------------------------- any source -> any camera
def resample_from_camera(src_model, src_params, src_rgb, src_depth, src_mask,
                         tgt_model, tgt_params, out_H, out_W, R=None, c2w_src=None):
    """Synthesize any central camera from any central-camera source (panorama, fisheye, pinhole, ...). The
    source only needs project(direction -> source pixel), the target only unproject(pixel -> ray).

    For each target pixel: unproject -> exact unit ray (target camera frame) -> rotate by R into the source
    camera frame -> project through the source model -> sample RGB and depth there. The target ray field is
    analytically exact (the model's unprojection). The extrinsics are exact too: the new camera shares the
    source optical centre (only the rays are re-indexed), so c2w_new = c2w_src @ blockdiag(R, 1) (rotation
    only, same translation), which preserves the world frame (det = +1) and multi-view consistency.

    With a limited-field-of-view source, target rays outside the source field of view are marked invalid in
    the mask (black, no garbage); a full-sphere panorama is the only source without holes. Returns rgb, depth,
    ray (target camera frame), mask, c2w_new."""
    from mapanything.datasets import camera_models as CM
    dev = src_rgb.device; H, W = src_rgb.shape[:2]
    ray, valid_t = CM.unproject(tgt_model, out_H, out_W, tgt_params, dev)
    ray = ray.to(torch.get_default_dtype())
    wdir = _renorm(ray @ R.T) if R is not None else _renorm(ray)        # target -> source cam frame
    uv, valid_s = (CM.project(src_model, wdir, src_params, H, W) if src_model == "spherical"
                   else CM.project(src_model, wdir, src_params))
    col, row = uv[..., 0], uv[..., 1]
    full360 = (src_model == "spherical" and float(src_params["hfov"]) >= 2 * torch.pi - 1e-6)
    rgb2 = _grid_sample(src_rgb, col, row, H, W, wrap_col=full360)
    depth2 = _grid_sample(src_depth, col, row, H, W, wrap_col=full360)
    m2 = _grid_sample(src_mask.to(torch.get_default_dtype()), col, row, H, W, wrap_col=full360)
    inb = (col >= 0) & (col <= W - 1) & (row >= 0) & (row <= H - 1)     # inside source frame
    mask2 = (m2 > 0.5) & valid_t & valid_s & (inb | full360)
    # Zero invalid pixels, so that regions outside the source field of view are black rather than
    # grid_sample's border stretch (streaks). No effect where the mask is valid (e.g. a full-sphere panorama);
    # for a limited source (pinhole, fisheye) it keeps RGB consistent with the mask. The full-360 wrap
    # already handles the seam.
    rgb2 = torch.where(mask2.unsqueeze(-1), rgb2, torch.zeros_like(rgb2))
    depth2 = torch.where(mask2, depth2, torch.zeros_like(depth2))
    if c2w_src is None:
        c2w_src = torch.eye(4, device=dev, dtype=torch.get_default_dtype())
    Rp4 = torch.eye(4, device=dev, dtype=torch.get_default_dtype())
    Rp4[:3, :3] = R if R is not None else torch.eye(3, device=dev, dtype=torch.get_default_dtype())
    c2w_new = c2w_src.to(torch.get_default_dtype()) @ Rp4                           # exact: shares centre, rot by R
    return rgb2, depth2, ray, mask2, c2w_new


def target_coverage(src_model, src_params, src_mask, tgt_model, tgt_params, out_H, out_W, R=None):
    """Fraction of target pixels whose ray lands in a valid source pixel (mask only; no RGB or depth needed).
    coverage == 1.0 <=> no new invalid region (the target stays inside the valid source field of view)."""
    from mapanything.datasets import camera_models as CM
    dev = src_mask.device; H, W = src_mask.shape[:2]
    ray, valid_t = CM.unproject(tgt_model, out_H, out_W, tgt_params, dev)
    wdir = _renorm(ray.to(torch.get_default_dtype()) @ R.T) if R is not None else _renorm(ray.to(torch.get_default_dtype()))
    uv, valid_s = (CM.project(src_model, wdir, src_params, H, W) if src_model == "spherical"
                   else CM.project(src_model, wdir, src_params))
    col, row = uv[..., 0], uv[..., 1]
    full360 = (src_model == "spherical" and float(src_params["hfov"]) >= 2 * torch.pi - 1e-6)
    m = _grid_sample(src_mask.to(torch.get_default_dtype()), col, row, H, W, wrap_col=full360)
    inb = (col >= 0) & (col <= W - 1) & (row >= 0) & (row <= H - 1)
    return ((m > 0.5) & valid_t & valid_s & (inb | full360)).float().mean().item()

