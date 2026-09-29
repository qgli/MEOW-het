"""Spherical measure utilities: per-pixel solid angle and the von Mises-Fisher likelihood.

Jacobian from rays: for any central camera the per-pixel solid angle is
dOmega = ||d_du x d_dv|| of the unit ray field; the ray field is its own
Tissot indicatrix, so no camera model is needed. Summed over a view it gives
the covered solid angle (e.g. 62.9% of the sphere for a 210 deg fisheye).

All functions are torch and differentiable-safe (weights are meant to be
treated as constants: callers should .detach() weights derived from ground
truth).
"""
from __future__ import annotations

import torch


def solid_angle_map(rays: torch.Tensor) -> torch.Tensor:
    """Per-pixel solid angle from a unit ray field.

    rays: (..., H, W, 3) unit vectors. Returns (..., H, W), replicate-padded
    central differences (edge pixels use one-sided differences implicitly).
    """
    du = torch.gradient(rays, dim=-2)[0]          # along W
    dv = torch.gradient(rays, dim=-3)[0]          # along H
    return torch.linalg.cross(du, dv, dim=-1).norm(dim=-1)


def solid_angle_weights(rays: torch.Tensor, valid: torch.Tensor | None = None,
                        clamp: tuple = (0.05, 20.0)) -> torch.Tensor:
    """Solid-angle loss weights: dOmega normalized to mean 1 over valid pixels, per view.

    rays: (B, H, W, 3); valid: (B, H, W) bool or None.
    Returns (B, H, W) with mean(w[valid]) == 1 per batch element. On a pinhole view
    the weight is proportional to cos^3 of the off-axis angle (close to 1 everywhere
    only for narrow fields of view).
    """
    om = solid_angle_map(rays)
    if valid is None:
        denom = om.flatten(1).mean(dim=1).clamp_min(1e-12)
    else:
        v = valid.to(om.dtype)
        denom = ((om * v).flatten(1).sum(dim=1)
                 / v.flatten(1).sum(dim=1).clamp_min(1.0)).clamp_min(1e-12)
    w = om / denom.view(-1, *([1] * (om.dim() - 1)))
    return w.clamp(*clamp)


def vmf_nll(cos_angle: torch.Tensor, kappa: torch.Tensor) -> torch.Tensor:
    """von Mises-Fisher negative log-likelihood on S^2 (Fisher 1953).

    NLL = -kappa*cos + log(4*pi) + log(sinh k) - log(k), with the numerically
    stable identity log(sinh k) = k + log1p(-exp(-2k)) - log 2 (valid all k>0).
    kappa broadcastable to cos_angle. The optimum in kappa is finite for a
    nonzero residual, so the confidence cannot collapse to its floor as in the
    unnormalized confidence loss of DUSt3R (Wang et al., CVPR 2024); for a zero
    residual the optimum is unbounded, so callers clamp kappa to about 1e4.
    """
    k = kappa.clamp_min(1e-6) if torch.is_tensor(kappa) else torch.as_tensor(
        max(kappa, 1e-6), dtype=cos_angle.dtype, device=cos_angle.device)
    log_sinh = k + torch.log1p(-torch.exp(-2.0 * k)) - 0.6931471805599453
    log_z = 2.5310242469692907 + log_sinh - torch.log(k)   # log(4*pi) + ...
    return -k * cos_angle + log_z
