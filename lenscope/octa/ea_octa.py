"""Equal-area octahedral (EA-octa) spherical RGBD storage.

Stores omnidirectional RGBD with uniform spherical sampling instead of an
equirectangular panorama (which over-samples the poles), so the same byte
budget buys either higher effective resolution or more scene variety; a
per-pixel ray maps to its texel with an analytic lookup. The bridge can write
depth and ambiguity flags in this layout as an optional pack extension
(GENESIS_PACK_OCTA=1); no released loader reads them.

Why an EA-octa map and not a blue-noise/Fibonacci point set: storage wants
(a) uniform solid-angle coverage, (b) O(1) analytic query, (c) seamless
interpolation.
- The EA-octa map is an analytic perfect hash dir -> (u,v) in [-1,1]^2:
  strictly equal-area (Jacobian: du dv = dOmega / pi, so every texel covers
  exactly 4*pi/R^2 sr), regular-grid bilinear (GPU grid_sample native),
  exact border rule (edge reversal below).
- A blue-noise point set is unbeatable for choosing directions (no lattice
  pattern, no holes), but as storage every query needs a true spatial hash +
  KNN scatter reconstruction: 5-20x the cost, lower interpolation quality,
  and its own low-frequency reconstruction noise. "No lattice pattern" is a
  property of rendering the points themselves; an interpolated storage grid
  shows no lattice in queries. Blue noise therefore stays on the sampling side.
- HEALPix is also equal-area, but hierarchical indexing + neighbour tables
  make interpolation far heavier than one analytic fold + bilinear.

Mapping (upper hemisphere z>=0 -> the |u|+|v|<=1 diamond, lower folded to
the corners):
    r = sqrt(1-|z|);  phi = atan2(|y|,|x|);  sigma = 2*phi/pi
    u = r*(1-sigma)*sign(x);  v = r*sigma*sign(y)
    z<0: (u,v) -> ((1-|v|)*sign(x), (1-|u|)*sign(y))
Equal-area proof sketch: dOmega = |dz| dphi = 2r dr * (pi/2) dsigma
= pi * (r dr dsigma) = pi * du dv within each quadrant.

Frame convention: directions live in the capture frame = core.cast's
camera frame (x right, y down, z forward), the same frame as the source
equirectangular image; one R_cam @ dir converts any consumer camera's pixel rays.
"""
from __future__ import annotations

import numpy as np

_EPS = 1e-12


def _sign_nz(a):
    """sign() that never returns 0 (pole/axis pixels need a branch)."""
    return np.where(a >= 0.0, 1.0, -1.0)


def dir_to_uv(d):
    """(..., 3) unit dirs -> (..., 2) EA-octa uv in [-1, 1]."""
    d = np.asarray(d, np.float64)
    x, y, z = d[..., 0], d[..., 1], d[..., 2]
    az, ay, ax = np.abs(z), np.abs(y), np.abs(x)
    r = np.sqrt(np.maximum(0.0, 1.0 - az))
    phi = np.arctan2(ay, np.maximum(ax, _EPS))
    sigma = phi * (2.0 / np.pi)
    u = r * (1.0 - sigma) * _sign_nz(x)
    v = r * sigma * _sign_nz(y)
    neg = z < 0.0
    uf = (1.0 - np.abs(v)) * _sign_nz(x)
    vf = (1.0 - np.abs(u)) * _sign_nz(y)
    u = np.where(neg, uf, u)
    v = np.where(neg, vf, v)
    return np.stack([u, v], -1)


def uv_to_dir(uv):
    """(..., 2) uv in [-1, 1] -> (..., 3) unit dirs. Inverse of dir_to_uv."""
    uv = np.asarray(uv, np.float64)
    u, v = uv[..., 0], uv[..., 1]
    r = np.abs(u) + np.abs(v)
    lower = r > 1.0
    uf = (1.0 - np.abs(v)) * _sign_nz(u)
    vf = (1.0 - np.abs(u)) * _sign_nz(v)
    u2 = np.where(lower, uf, u)
    v2 = np.where(lower, vf, v)
    r2 = np.abs(u2) + np.abs(v2)
    z = (1.0 - r2 * r2) * np.where(lower, -1.0, 1.0)
    sigma = np.where(r2 > _EPS, np.abs(v2) / np.maximum(r2, _EPS), 0.0)
    phi = sigma * (np.pi / 2.0)
    rho = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    x = _sign_nz(u2) * rho * np.cos(phi)
    y = _sign_nz(v2) * rho * np.sin(phi)
    return np.stack([x, y, z], -1)


def texel_dirs(R):
    """(R, R, 3) unit dirs at texel centers. Row j -> v, col i -> u."""
    c = (np.arange(R) + 0.5) / R * 2.0 - 1.0
    u, v = np.meshgrid(c, c)              # v varies along rows (axis 0)
    return uv_to_dir(np.stack([u, v], -1))


def uv_to_pix(uv, R):
    """uv in [-1,1] -> continuous pixel coords (col, row) in [-0.5, R-0.5]
    (texel centers at integers)."""
    uv = np.asarray(uv, np.float64)
    return (uv + 1.0) * (R / 2.0) - 0.5


def pad_octa(img):
    """(R, R, C) -> (R+2, R+2, C) with the exact octa border rule
    (crossing an edge = 180-deg rotation about that edge's midpoint =>
    pad row/col is the boundary row/col reversed; corners are the
    diagonally-opposite corner texels)."""
    R = img.shape[0]
    C = img.shape[2] if img.ndim == 3 else 1
    src = img.reshape(R, R, C)
    out = np.empty((R + 2, R + 2, C), src.dtype)
    out[1:-1, 1:-1] = src
    out[0, 1:-1] = src[0, ::-1]            # top: u -> -u
    out[-1, 1:-1] = src[-1, ::-1]          # bottom
    out[1:-1, 0] = src[::-1, 0]            # left: v -> -v
    out[1:-1, -1] = src[::-1, -1]          # right
    out[0, 0] = src[-1, -1]                # corners: double fold = diagonal
    out[0, -1] = src[-1, 0]
    out[-1, 0] = src[0, -1]
    out[-1, -1] = src[0, 0]
    return out if img.ndim == 3 else out[..., 0]


def bilinear_octa(img, uv, edge_aware_rel=None):
    """Bilinear lookup on an EA-octa map with the exact border rule.
    img (R,R,C) or (R,R); uv (...,2). edge_aware_rel: if set (e.g. 0.02 for
    depth), when the 4-neighbour relative spread exceeds it, return the
    max-weight neighbour instead of blending (no cross-boundary ghost
    depths)."""
    R = img.shape[0]
    pimg = pad_octa(img if img.ndim == 3 else img[..., None])
    C = pimg.shape[2]
    pq = uv_to_pix(uv, R) + 1.0            # into padded coords
    cx = np.clip(pq[..., 0], 0.0, R + 1 - 1e-6)
    cy = np.clip(pq[..., 1], 0.0, R + 1 - 1e-6)
    x0 = np.floor(cx).astype(np.int64)
    y0 = np.floor(cy).astype(np.int64)
    x0 = np.clip(x0, 0, R)
    y0 = np.clip(y0, 0, R)
    fx = (cx - x0)[..., None]
    fy = (cy - y0)[..., None]
    q00 = pimg[y0, x0].astype(np.float64)
    q01 = pimg[y0, x0 + 1].astype(np.float64)
    q10 = pimg[y0 + 1, x0].astype(np.float64)
    q11 = pimg[y0 + 1, x0 + 1].astype(np.float64)
    out = (q00 * (1 - fx) * (1 - fy) + q01 * fx * (1 - fy)
           + q10 * (1 - fx) * fy + q11 * fx * fy)
    if edge_aware_rel is not None:
        stack = np.stack([q00, q01, q10, q11], 0)
        lo = stack.min(0)
        hi = stack.max(0)
        spread = (hi - lo) / np.maximum(np.abs(hi), _EPS)
        w = np.stack([(1 - fx) * (1 - fy), fx * (1 - fy),
                      (1 - fx) * fy, fx * fy], 0)
        nearest = np.take_along_axis(
            stack, w.argmax(0)[None], 0)[0]
        out = np.where(spread > edge_aware_rel, nearest, out)
    return out if img.ndim == 3 else out[..., 0]


# equirectangular pixels <-> dirs in the same frame as core.cast.erp_dirs

def erp_pix_dirs(W, H):
    """(H, W, 3) camera-frame dirs, identical convention to core.cast."""
    from lenscope.core.cast import erp_dirs
    return erp_dirs(W, H).reshape(H, W, 3)


def dir_to_erp_pix(d, W, H):
    """dirs -> continuous equirectangular (col, row); inverse of erp_pix_dirs."""
    d = np.asarray(d, np.float64)
    x, y, z = d[..., 0], d[..., 1], d[..., 2]
    az = np.arctan2(x, z)                          # centre col = +z fwd
    el = np.arctan2(-y, np.sqrt(x * x + z * z))
    col = (az / (2 * np.pi) + 0.5) * W - 0.5
    row = (0.5 - el / np.pi) * H - 0.5
    return np.stack([col, row], -1)


def erp_bilinear(img, d):
    """Bilinear equirectangular lookup with azimuth wrap (rows clamped)."""
    H, W = img.shape[0], img.shape[1]
    pq = dir_to_erp_pix(d, W, H)
    cx, cy = pq[..., 0], np.clip(pq[..., 1], 0.0, H - 1.0)
    x0 = np.floor(cx).astype(np.int64)
    y0 = np.floor(cy).astype(np.int64)
    y0 = np.clip(y0, 0, H - 2)
    fx = (cx - x0)[..., None]
    fy = (cy - y0)[..., None]
    x0m = np.mod(x0, W)
    x1m = np.mod(x0 + 1, W)
    a = img[y0, x0m].astype(np.float64)
    b = img[y0, x1m].astype(np.float64)
    c = img[y0 + 1, x0m].astype(np.float64)
    e = img[y0 + 1, x1m].astype(np.float64)
    return a * (1 - fx) * (1 - fy) + b * fx * (1 - fy) \
        + c * (1 - fx) * fy + e * fx * fy


def resample_erp_to_octa(erp_img, R):
    """RGB path: area-correct enough via bilinear pull (the panorama source is
    equal-or-over-sampled everywhere when R = 0.564*W_erp; pole rows are
    averaged by the pull's footprint)."""
    d = texel_dirs(R)
    return erp_bilinear(erp_img, d)


def octa_to_erp(octa_img, W, H, edge_aware_rel=None):
    """Reconstruct an equirectangular view of the octa store (visual check:
    npz -> equirectangular RGB + depth images)."""
    d = erp_pix_dirs(W, H)
    return bilinear_octa(octa_img, dir_to_uv(d), edge_aware_rel)


def r_for_erp(W):
    """R giving texels the same solid angle as the equirectangular equator pixel:
    (2*pi/W)^2 = 4*pi/R^2  =>  R = W/sqrt(pi) ~= 0.5642*W. Snapped to /8."""
    R = int(round(W / np.sqrt(np.pi) / 8.0)) * 8
    return max(R, 8)
