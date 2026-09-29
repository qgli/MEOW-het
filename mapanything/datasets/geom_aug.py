"""Random rotations and photometric lens effects for camera-sampled views (numpy).

random_so3 draws the random content rotation of full-panorama views. chromatic (lateral chromatic
aberration) and vignette are photometric: they change the RGB image (HxWx3 uint8) only, so rays, depth, mask
and camera pose stay unchanged.
"""
import numpy as np


def _Ry(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def _Rz(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def random_so3(rng):
    """Random rotation Rz(a) @ Ry(b) @ Rx(c) with angles uniform in [-pi, pi): all of SO(3), not only yaw."""
    def Rx(t):
        c, s = np.cos(t), np.sin(t)
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    a, b, c = rng.uniform(-np.pi, np.pi, 3)
    return (_Rz(a) @ _Ry(b) @ Rx(c)).astype(np.float64)


def _sample(img, ys, xs, wrap_col=False):
    """Bilinear sample img at float (ys,xs). Returns (values, in_frame_bool)."""
    H, W = img.shape[:2]
    inb = (xs >= 0) & (xs <= W - 1) & (ys >= 0) & (ys <= H - 1)
    if wrap_col:
        c0 = np.floor(xs).astype(int) % W; c1 = (c0 + 1) % W
    else:
        c0 = np.clip(np.floor(xs), 0, W - 1).astype(int); c1 = np.clip(c0 + 1, 0, W - 1)
    r0 = np.clip(np.floor(ys), 0, H - 1).astype(int); r1 = np.clip(r0 + 1, 0, H - 1)
    fx = xs - np.floor(xs); fy = ys - np.floor(ys)
    if img.ndim == 3:
        fx = fx[..., None]; fy = fy[..., None]
    v = (img[r0, c0] * (1 - fx) + img[r0, c1] * fx) * (1 - fy) + \
        (img[r1, c0] * (1 - fx) + img[r1, c1] * fx) * fy
    return v, inb


# No per-view horizontal flip. Camera rotations (det = +1) preserve the world point cloud and are safe per
# view; a horizontal flip is a reflection (det = -1) that mirrors the world (P -> M P, M = diag(-1, 1, 1)).
# A reflection is not in SE(3): flipping one view but not its covisible neighbours puts the same physical
# point at P and at M P, which no rigid transform can register, so multi-view geometry breaks. Only a
# whole-scene flip would be consistent (a chiral copy of the scene).


def chromatic(rgb, ca_r=1.003, ca_b=0.997):
    """Lateral chromatic aberration (photometric: RGB only; ground-truth rays, depth and pose unchanged).
    Scales the red and blue sampling grids by ca_r and ca_b about the image centre (green unchanged);
    edge-clamped sampling gives a thin fringe and no black border."""
    H, W = rgb.shape[:2]
    jj, ii = np.meshgrid(np.arange(W), np.arange(H))
    nx = (2 * jj + 1) / W - 1.0; ny = (2 * ii + 1) / H - 1.0
    out = rgb.astype(np.float32).copy()
    for ch, sc in ((0, ca_r), (2, ca_b)):
        col = (nx * sc + 1.0) / 2.0 * W - 0.5; row = (ny * sc + 1.0) / 2.0 * H - 0.5
        out[..., ch], _ = _sample(rgb[..., ch].astype(np.float32), row, col)   # _sample clamps -> no black
    return np.clip(out, 0, 255).astype(rgb.dtype)


def vignette(rgb, strength=0.3, R=0.9):
    """cos^4 optical falloff (photometric: RGB only; ground truth unchanged). Multiplicative darkening
    rather than a remap, so no invalid or out-of-frame region is created and the mask is unchanged."""
    H, W = rgb.shape[:2]
    jj, ii = np.meshgrid(np.arange(W), np.arange(H))
    nx = (2 * jj + 1) / W - 1.0; ny = (2 * ii + 1) / H - 1.0
    r = np.sqrt(nx * nx + ny * ny)
    cos_t = 1.0 / np.sqrt(1.0 + (r / max(R, 1e-6)) ** 2)
    v = (1.0 - strength) + strength * cos_t ** 4
    return np.clip(rgb.astype(np.float32) * v[..., None], 0, 255).astype(rgb.dtype)

