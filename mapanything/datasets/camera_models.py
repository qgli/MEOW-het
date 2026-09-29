"""Stand-alone torch camera models (no external dependencies) for synthesizing any central camera from a
full-sphere panorama render. The models are those of UniK3D (Piccinelli et al., CVPR 2025): Pinhole, OpenCV
(Brown-Conrady radial, tangential and thin-prism terms), EUCM, Spherical (equirectangular crop), Fisheye624
(Kannala-Brandt polynomial in theta) and Mei (unified model with mirror parameter xi).

Each model provides:
  unproject(H, W, p, device) -> unit rays (H,W,3) in the OpenCV cam frame (x right, y down, z fwd)
                                (+ valid mask for models with an invalid region)          [pixel->dir]
  project(xyz, p)            -> pixel (...,2) + valid mask                                [dir->pixel]

unproject gives one ray per output pixel for resampling a panorama into the target camera; project maps
directions into a source camera and allows the round-trip check project(unproject(grid)) == grid to
sub-pixel accuracy. Computation uses the current default dtype (float64 in the camera sampler). Newton or
fixed-point inverses are used where the forward distortion has no closed-form inverse; their correctness is
checked by the round trip rather than against a reference solver.
"""
import torch


def _grid(H, W, device):
    jj, ii = torch.meshgrid(torch.arange(W, device=device, dtype=torch.get_default_dtype()),
                            torch.arange(H, device=device, dtype=torch.get_default_dtype()), indexing="xy")
    return jj, ii                                          # (H,W) col, row


def _norm(v, eps=1e-12):
    return v / v.norm(dim=-1, keepdim=True).clamp_min(eps)


# ============================================================== Pinhole
def pinhole_unproject(H, W, p, device):
    fx, fy, cx, cy = p["fx"], p["fy"], p["cx"], p["cy"]
    u, v = _grid(H, W, device)
    return _norm(torch.stack([(u - cx) / fx, (v - cy) / fy, torch.ones_like(u)], -1))


def pinhole_project(xyz, p):
    x, y, z = xyz.unbind(-1); zc = z.clamp_min(1e-9)
    u = p["fx"] * x / zc + p["cx"]; v = p["fy"] * y / zc + p["cy"]
    return torch.stack([u, v], -1), z > 1e-6


# ============================================================== OpenCV (Brown-Conrady)
def _opencv_distort(x, y, p):
    k1, k2, k3 = p.get("k1", 0.), p.get("k2", 0.), p.get("k3", 0.)
    k4, k5, k6 = p.get("k4", 0.), p.get("k5", 0.), p.get("k6", 0.)
    p1, p2 = p.get("p1", 0.), p.get("p2", 0.)
    s1, s2, s3, s4 = p.get("s1", 0.), p.get("s2", 0.), p.get("s3", 0.), p.get("s4", 0.)
    r2 = x * x + y * y; r4 = r2 * r2; r6 = r4 * r2
    rad = (1 + k1 * r2 + k2 * r4 + k3 * r6) / (1 + k4 * r2 + k5 * r4 + k6 * r6)
    xd = x * rad + 2 * p1 * x * y + p2 * (r2 + 2 * x * x) + s1 * r2 + s2 * r4
    yd = y * rad + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y + s3 * r2 + s4 * r4
    return xd, yd


def opencv_unproject(H, W, p, device, iters=30):
    k1, k2, k3 = p.get("k1", 0.), p.get("k2", 0.), p.get("k3", 0.)
    k4, k5, k6 = p.get("k4", 0.), p.get("k5", 0.), p.get("k6", 0.)
    p1, p2 = p.get("p1", 0.), p.get("p2", 0.)
    s1, s2, s3, s4 = p.get("s1", 0.), p.get("s2", 0.), p.get("s3", 0.), p.get("s4", 0.)
    fx, fy, cx, cy = p["fx"], p["fy"], p["cx"], p["cy"]
    u, v = _grid(H, W, device)
    xd = (u - cx) / fx; yd = (v - cy) / fy
    # Radial Newton first: solve rho*radial(rho) = rd for the undistorted radius rho. The radial map is
    # monotonic (the camera sampler draws k2 accordingly), so Newton converges quadratically and stably,
    # unlike the division-form fixed point, which diverges at large radii for barrel distortion (rad < 1).
    rd = torch.sqrt(xd * xd + yd * yd).clamp_min(1e-12)
    rho = rd.clone()
    for _ in range(12):
        r2 = rho * rho; r4 = r2 * r2; r6 = r4 * r2
        num = 1 + k1 * r2 + k2 * r4 + k3 * r6; den = 1 + k4 * r2 + k5 * r4 + k6 * r6; rad = num / den
        f = rho * rad - rd
        dnum = 2 * rho * (k1 + 2 * k2 * r2 + 3 * k3 * r4); dden = 2 * rho * (k4 + 2 * k5 * r2 + 3 * k6 * r4)
        fp = (rad + rho * (dnum * den - num * dden) / (den * den)).clamp_min(1e-9)
        rho = (rho - f / fp).clamp_min(0.0)
    sc = rho / rd; x = xd * sc; y = yd * sc                # radial solution; refine the small tangential/thin-prism terms
    for _ in range(6):
        r2 = x * x + y * y; r4 = r2 * r2; r6 = r4 * r2
        rad = (1 + k1 * r2 + k2 * r4 + k3 * r6) / (1 + k4 * r2 + k5 * r4 + k6 * r6)
        dxt = 2 * p1 * x * y + p2 * (r2 + 2 * x * x) + s1 * r2 + s2 * r4
        dyt = p1 * (r2 + 2 * y * y) + 2 * p2 * x * y + s3 * r2 + s4 * r4
        x = (xd - dxt) / rad; y = (yd - dyt) / rad
    # Round-trip validity: re-distort the recovered (x,y). Beyond the barrel-fold radius the inverse has no
    # solution (the iteration lands on a wrong branch), so re-distorting does not return to (xd,yd). Those
    # pixels are marked invalid (black) instead of sampling the source along a wrong ray.
    xr, yr = _opencv_distort(x, y, p)
    valid = ((xr - xd) ** 2 + (yr - yd) ** 2) < 1e-3
    return _norm(torch.stack([x, y, torch.ones_like(x)], -1)), valid


def opencv_project(xyz, p):
    x, y, z = xyz.unbind(-1); zc = z.clamp_min(1e-9)
    xd, yd = _opencv_distort(x / zc, y / zc, p)
    return torch.stack([p["fx"] * xd + p["cx"], p["fy"] * yd + p["cy"]], -1), z > 1e-6


# ============================================================== EUCM (Enhanced Unified)
def eucm_unproject(H, W, p, device):
    fx, fy, cx, cy, a, b = p["fx"], p["fy"], p["cx"], p["cy"], p["alpha"], p["beta"]
    u, v = _grid(H, W, device)
    mx = (u - cx) / fx; my = (v - cy) / fy; r2 = mx * mx + my * my
    sq = (1 - (2 * a - 1) * b * r2).clamp_min(1e-9)
    mz = (1 - b * a * a * r2) / (a * torch.sqrt(sq) + (1 - a))
    valid = r2 < (1e9 if a < 0.5 else 1.0 / (b * (2 * a - 1)))
    ray = _norm(torch.stack([mx, my, mz], -1))
    return ray, valid & (ray[..., 2] > -1.0)


def eucm_project(xyz, p):
    fx, fy, cx, cy, a, b = p["fx"], p["fy"], p["cx"], p["cy"], p["alpha"], p["beta"]
    x, y, z = xyz.unbind(-1)
    d = torch.sqrt(b * (x * x + y * y) + z * z)
    den = (a * d + (1 - a) * z).clamp_min(1e-6)
    return torch.stack([fx * x / den + cx, fy * y / den + cy], -1), z > 0


# ============================================================== Spherical (equirectangular crop)
def spherical_unproject(H, W, p, device):
    hfov, vfov = p["hfov"], p["vfov"]
    u, v = _grid(H, W, device)
    lon = (u - (W - 1) / 2) / (W - 1) * hfov
    lat = (v - (H - 1) / 2) / (H - 1) * vfov
    x = torch.cos(lat) * torch.sin(lon); y = torch.sin(lat); z = torch.cos(lat) * torch.cos(lon)
    return _norm(torch.stack([x, y, z], -1))


def spherical_project(xyz, p, H, W):
    hfov, vfov = p["hfov"], p["vfov"]
    x, y, z = xyz.unbind(-1)
    lon = torch.atan2(x, z); lat = torch.asin((y / xyz.norm(dim=-1).clamp_min(1e-9)).clamp(-1, 1))
    u = lon / hfov * (W - 1) + (W - 1) / 2; v = lat / vfov * (H - 1) + (H - 1) / 2
    return torch.stack([u, v], -1), (lon.abs() <= hfov / 2) & (lat.abs() <= vfov / 2)


# ============================================================== Fisheye624 (Kannala-Brandt)
def _kb_rtheta(th, p):                                     # distorted radius r~(theta)
    a = [p.get("a3", 0.), p.get("a5", 0.), p.get("a7", 0.), p.get("a9", 0.), p.get("a11", 0.), p.get("a13", 0.)]
    r = th.clone()
    for i, c in enumerate(a):
        r = r + c * th ** (3 + 2 * i)
    return r


def fisheye624_unproject(H, W, p, device, iters=30):
    fx, fy, cx, cy = p["fx"], p["fy"], p["cx"], p["cy"]
    p1, p2 = p.get("p1", 0.), p.get("p2", 0.)
    s1, s2, s3, s4 = p.get("s1", 0.), p.get("s2", 0.), p.get("s3", 0.), p.get("s4", 0.)
    u, v = _grid(H, W, device)
    xd = (u - cx) / fx; yd = (v - cy) / fy
    xr, yr = xd.clone(), yd.clone()                       # remove tangential + thin-prism (fixed-point)
    for _ in range(iters):
        r2 = xr * xr + yr * yr; r4 = r2 * r2
        ex = 2 * p1 * xr * yr + p2 * (r2 + 2 * xr * xr) + s1 * r2 + s2 * r4
        ey = p1 * (r2 + 2 * yr * yr) + 2 * p2 * xr * yr + s3 * r2 + s4 * r4
        xr = xd - ex; yr = yd - ey
    rd = torch.sqrt(xr * xr + yr * yr)                     # = r~(theta)
    th = rd.clone()                                        # invert r~(theta)=theta+sum a theta^odd (Newton)
    for _ in range(iters):
        f = _kb_rtheta(th, p) - rd
        a = [p.get("a3", 0.), p.get("a5", 0.), p.get("a7", 0.), p.get("a9", 0.), p.get("a11", 0.), p.get("a13", 0.)]
        df = torch.ones_like(th)
        for i, c in enumerate(a):
            df = df + c * (3 + 2 * i) * th ** (2 + 2 * i)
        th = th - f / df.clamp_min(1e-9)
    th = th.clamp_min(0)
    psi = torch.atan2(yr, xr); s = torch.sin(th)
    ray = torch.stack([s * torch.cos(psi), s * torch.sin(psi), torch.cos(th)], -1)
    return _norm(ray), th < (torch.pi * 0.99)


def fisheye624_project(xyz, p):
    fx, fy, cx, cy = p["fx"], p["fy"], p["cx"], p["cy"]
    p1, p2 = p.get("p1", 0.), p.get("p2", 0.)
    s1, s2, s3, s4 = p.get("s1", 0.), p.get("s2", 0.), p.get("s3", 0.), p.get("s4", 0.)
    x, y, z = xyz.unbind(-1)
    rxy = torch.sqrt(x * x + y * y).clamp_min(1e-12)
    th = torch.atan2(rxy, z)
    rd = _kb_rtheta(th, p)
    xr = rd * x / rxy; yr = rd * y / rxy
    r2 = xr * xr + yr * yr; r4 = r2 * r2
    xd = xr + 2 * p1 * xr * yr + p2 * (r2 + 2 * xr * xr) + s1 * r2 + s2 * r4
    yd = yr + p1 * (r2 + 2 * yr * yr) + 2 * p2 * xr * yr + s3 * r2 + s4 * r4
    return torch.stack([fx * xd + cx, fy * yd + cy], -1), z > -rxy   # valid up to ~hemisphere+


# ============================================================== Mei (unified / mirror)
def _mei_undistort(xd, yd, p, iters=20):
    k1, k2, p1, p2 = p.get("k1", 0.), p.get("k2", 0.), p.get("p1", 0.), p.get("p2", 0.)
    x, y = xd.clone(), yd.clone()
    for _ in range(iters):
        r2 = x * x + y * y; r4 = r2 * r2; rad = 1 + k1 * r2 + k2 * r4
        ex = x * rad + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
        ey = y * rad + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
        x = x + (xd - ex); y = y + (yd - ey)
    return x, y


def mei_unproject(H, W, p, device):
    fx, fy, cx, cy, xi = p["fx"], p["fy"], p["cx"], p["cy"], p["xi"]
    u, v = _grid(H, W, device)
    mx, my = _mei_undistort((u - cx) / fx, (v - cy) / fy, p)
    r2 = mx * mx + my * my
    zs = (xi + torch.sqrt((1 + (1 - xi * xi) * r2).clamp_min(1e-9))) / (r2 + 1)
    ray = torch.stack([zs * mx, zs * my, zs - xi], -1)
    return _norm(ray), ray[..., 2] > -1.0


def mei_project(xyz, p):
    fx, fy, cx, cy, xi = p["fx"], p["fy"], p["cx"], p["cy"], p["xi"]
    k1, k2, p1, p2 = p.get("k1", 0.), p.get("k2", 0.), p.get("p1", 0.), p.get("p2", 0.)
    n = xyz.norm(dim=-1).clamp_min(1e-9); x, y, z = (xyz / n.unsqueeze(-1)).unbind(-1)
    den = (z + xi).clamp_min(1e-6); xn = x / den; yn = y / den
    r2 = xn * xn + yn * yn; r4 = r2 * r2; rad = 1 + k1 * r2 + k2 * r4
    xd = xn * rad + 2 * p1 * xn * yn + p2 * (r2 + 2 * xn * xn)
    yd = yn * rad + p1 * (r2 + 2 * yn * yn) + 2 * p2 * xn * yn
    return torch.stack([fx * xd + cx, fy * yd + cy], -1), z > -xi


# ============================================================== registry
MODELS = {
    "pinhole": (pinhole_unproject, pinhole_project),
    "opencv": (opencv_unproject, opencv_project),
    "eucm": (eucm_unproject, eucm_project),
    "spherical": (spherical_unproject, spherical_project),
    "fisheye624": (fisheye624_unproject, fisheye624_project),
    "mei": (mei_unproject, mei_project),
}


def unproject(model, H, W, params, device):
    """Unit rays (H,W,3) + valid mask for the named model. Pinhole/Spherical have no invalid region."""
    fn = MODELS[model][0]
    out = fn(H, W, params, device)
    if isinstance(out, tuple):
        return out
    return out, torch.ones(H, W, dtype=torch.bool, device=device)


def project(model, xyz, params, H=None, W=None):
    fn = MODELS[model][1]
    return fn(xyz, params, H, W) if model == "spherical" else fn(xyz, params)
