"""Online camera sampler: builds covisibility-connected tuples of sampled cameras from a scene.

Given a scene and a tuple size K, the sampler selects K poses with distinct optical centres by a random
walk on the offline covisibility and samples one camera per pose (model, field of view, aspect ratio,
principal point, roll, tilt, distortion). A tuple is planned in one of two modes:

  native mode ("Mode A" in the code): each camera keeps the orientation of its panorama pose.
  aimed mode ("Mode B"): the cameras converge on a common surface point or look along a shared direction
      toward it, with fields of view from 90 deg upwards.

Each view is resampled from a render of its pose: rectilinear views up to 112 deg from the sharpest pinhole
render that covers the whole view, if any, and all other views from the equirectangular panorama (_route).
The sampled views are rendered at low resolution, covisibility is recomputed, and the tuple is kept only if
the covisibility graph (edges > 0.25) is connected; disconnected views are moved toward their native
configuration in up to three steps (connectivity.enforce_connectivity). Each view dict carries rays, depth,
validity and pose (for covisibility), the RGB image and depth map, and the sampled spec (model, field of view,
aspect ratio, roll, tilt, principal point, mode) for the sampling record.

sample_plan runs on the CPU and is safe in dataloader workers; realize renders on the sampler's device. Rays
and poses are in the OpenCV camera convention (the renderer camera frame with its Y axis flipped).
"""
from __future__ import annotations
import os, json, numpy as np, torch, torch.nn.functional as F
from mapanything.datasets import camera_models as CM, gpu_aug as GA, connectivity as CN, geom_aug as GAUG

_FLIP = torch.tensor([[1., 0, 0], [0, -1, 0], [0, 0, 1]], dtype=torch.float64)

# Camera types: (name, camera model, sampling weight, min and max field of view in deg). OpenCV is the bulk
# of the distribution; erp_full is the full 360 x 180 panorama with a random SO(3) content rotation. The
# minimum field of view is 48 deg: telephoto views are rare in the wild, especially for reconstruction. The
# ranges couple field of view and model: narrow
# views are rectilinear, wide views fisheye or spherical (a 48 deg fisheye or a 300 deg pinhole is neither
# realistic nor valid). Rectilinear models form the majority; fisheye models and panoramas are weighted above
# their frequency in the wild, for robustness.
TYPES = [
    ("opencv",     "opencv",     0.40, 48, 120),
    ("pinhole",    "pinhole",    0.15, 48, 95),
    ("fisheye624", "fisheye624", 0.13, 120, 210),
    ("eucm",       "eucm",       0.07, 90, 180),
    ("mei",        "mei",        0.05, 140, 200),
    ("sph_crop",   "spherical",  0.05, 110, 300),   # equirectangular crop; the field of view is horizontal
    ("erp_full",   "spherical",  0.15, 360, 360),   # full panorama: aspect ratio 2:1, random SO(3) content rotation
]
_W = np.array([t[2] for t in TYPES]); _W = _W / _W.sum()


def _model_for_fov(fov):
    """Camera model of a restored native-mode view: rectilinear up to a 120 deg diagonal (the limit in
    _tparams), Fisheye624 up to 210 deg, spherical above."""
    return "pinhole" if fov <= 120 else ("fisheye624" if fov <= 210 else "spherical")


class CameraSampler:
    def __init__(self, data_root, device="cuda", p_modeB=0.55, p_convergent=0.90,
                 pp_max=0.22, tilt_max=12.0, base_res=256, ss=2,
                 walk_thres=0.40, p_portrait=0.24,
                 p_pp_extreme=0.10, pp_sigma=0.012, pp_ground_max=0.05,
                 p_tilt_extreme=0.10, tilt_sigma=4.0, p_roll_extreme=0.10,
                 res_hint=518, walk_thres_b_relax=0.20):
        self.root = data_root; self.dev = device
        self.p_modeB = p_modeB; self.p_conv = p_convergent
        self.p_portrait = float(p_portrait)   # fraction of portrait views (captures in the wild are mostly landscape)
        # Principal point, roll and tilt: with probability 1 - p_*_extreme (0.9) a prior matched to real cameras
        # (small sensor-lens decentering, near-level capture), otherwise a wide uniform tail for robustness.
        self.p_pp_extreme = float(p_pp_extreme); self.pp_sigma = float(pp_sigma); self.pp_ground_max = float(pp_ground_max)
        self.p_tilt_extreme = float(p_tilt_extreme); self.tilt_sigma = float(tilt_sigma)
        self.p_roll_extreme = float(p_roll_extreme)
        # walk_thres is above the online threshold (0.25): chains are selected on the precomputed covisibility
        # with headroom, so that the weakest link tends to stay above 0.25 after camera sampling and rendering.
        self.walk_thres = float(walk_thres)
        # Aimed mode: when no chain of length K exists at walk_thres (e.g. K = 8 in a scene with 8 poses, which
        # needs all of them in one chain), the walk is retried at this relaxed threshold. Only the aimed mode
        # relaxes: its cameras are oriented toward a common point after pose selection, so their rendered
        # covisibility exceeds the precomputed value, and the check after rendering (0.25) still decides. The
        # native mode keeps the single threshold, since its rendered covisibility is close to the precomputed one.
        self.walk_thres_b_relax = float(walk_thres_b_relax)
        self.covis_res = 96      # long side of the low-resolution renders of the connectivity check
        self.pp_max = pp_max; self.tilt_max = tilt_max; self.D = base_res
        self.res_hint = int(res_hint)
        self.ss = int(ss)        # supersampling factor: render at ss x, average RGB over ss x ss blocks, subsample geometry
        self.FLIP = _FLIP.to(device); self._cache = {}
        self.work_dtype = torch.float64   # working precision of the GPU resampling
        self.pack_cache_max = 6      # LRU cap of _load_src (per process or worker; about 40 MB per entry)
        self.scene_cache_max = 32    # cap of the _scene metadata cache (JSON and covisibility, about 0.5 MB per scene)

    # ---------------------------------------------------------------- scene io
    def _scene(self, name):
        if name in self._cache: return self._cache[name]
        p = os.path.join(self.root, name)
        meta = {fr["tag"]: fr for fr in json.load(open(f"{p}/metadata.json"))["frames"]}
        fm = json.load(open(f"{p}/covisibility/v0/frame_meta.json"))["frames"]
        cov = np.load(f"{p}/covisibility/v0/covisibility.npy")
        erp = [i for i in range(len(fm)) if fm[i]["base_type"] == "erp" and os.path.isfile(f"{p}/{fm[i]['tag']}_pack.npz")]
        bic = np.minimum(cov, cov.T)
        # Pose-level covisibility for the aimed-mode walk: pcov[a, b] is the maximum, over all frame pairs of
        # poses a and b, of the symmetric covisibility min(cov, cov.T), i.e. whether any view pair of the two
        # poses is strongly covisible. The aimed-mode orientation is chosen after pose selection, so this is the
        # relevant measure; the panorama-to-panorama covisibility is normalised by the full sphere and is
        # systematically low (two panoramas in adjacent rooms see little of each other's sphere).
        pids = sorted({fm[i]["pose_index"] for i in range(len(fm))})
        pidx = {pid: k for k, pid in enumerate(pids)}
        pof = np.array([pidx[fm[i]["pose_index"]] for i in range(len(fm))])
        P = len(pids); pcov = np.eye(P, dtype=np.float32)
        for a in range(P):
            for b in range(a + 1, P):
                m = float(bic[np.ix_(pof == a, pof == b)].max())
                pcov[a, b] = pcov[b, a] = m
        s = dict(p=p, meta=meta, fm=fm, cov=cov, bicov=bic, erp=erp, pcov=pcov, pidx=pidx,
                 pose={i: fm[i]["pose_index"] for i in range(len(fm))})
        if len(self._cache) >= int(getattr(self, "scene_cache_max", 32)):     # bounded cache: training visits
            self._cache.pop(next(iter(self._cache)))                          # many scenes; drop the oldest entry
        self._cache[name] = s; return s

    def _c2w(self, s, pose):
        C = np.asarray(s["meta"][f"erp_pose{pose:02d}"]["camera_to_world_unicol_4x4"], np.float64)
        wd = torch.get_default_dtype()
        return torch.from_numpy(C[:3, :3]).to(self.dev, wd) @ self.FLIP.to(self.dev, wd), torch.from_numpy(C[:3, 3]).to(self.dev, wd)

    def _load_src(self, s, pose, base_name="erp"):
        """Load a source pack of this pose (panorama, fisheye or pinhole render), decoded once and cached on the
        device. The cache is LRU-bounded (pack_cache_max entries of about 40 MB) and stores float32, which is
        lossless for the float16 packs on disk; tensors are cast to the working dtype on return."""
        key = (s["p"], pose, base_name)
        if not hasattr(self, "_packs"):
            from collections import OrderedDict
            self._packs = OrderedDict()
        if key in self._packs:
            self._packs.move_to_end(key)
        else:
            z = np.load(f"{s['p']}/{base_name}_pose{pose:02d}_pack.npz")
            r = z["rgb"].astype(np.float32); r = r if r.max() <= 1.5 else r / 255
            self._packs[key] = (torch.from_numpy(r).to(self.dev), torch.from_numpy(z["depth"].astype(np.float32)).to(self.dev),
                                torch.from_numpy(z["mask"].astype(bool)).to(self.dev))
            while len(self._packs) > int(getattr(self, "pack_cache_max", 6)):
                self._packs.popitem(last=False)
        r32, d32, m = self._packs[key]
        wd = torch.get_default_dtype()
        return r32.to(wd), d32.to(wd), m

    def _erp(self, s, pose):
        return self._load_src(s, pose, "erp")

    def _src_intr(self, s, base_name):
        """Source camera model + params from metadata (constant across poses; pose00 frame)."""
        f = s["meta"][f"{base_name}_pose00"]; W, H = f["resolution"]; bt = f["base_type"]; intr = f["intrinsics"]
        if bt == "pinhole": fp = intr["f_px"]; return "pinhole", {"fx": fp, "fy": fp, "cx": (W-1)/2, "cy": (H-1)/2}
        if bt == "fisheye": ft = intr["f_theta"]; return "fisheye624", {"fx": ft, "fy": ft, "cx": (W-1)/2, "cy": (H-1)/2}
        return "spherical", {"hfov": torch.tensor(2*np.pi), "vfov": torch.tensor(np.pi)}

    _PIN_BY_SHARPNESS = ("pin_40mm", "pin_24mm", "pin_14mm")   # narrowest field of view (highest px/deg) first
    _ERP = ("erp", "spherical", None)

    def _route(self, s, pose, pref, fov, tgt_model, tgt_params, H, W, Rarg):
        """Choose the source render. The choice affects resolution only, not covisibility.
          'pin': the sharpest pinhole render of the pose that covers the whole view (coverage >= 0.999,
                 18-40 px/deg); otherwise the panorama.
          'erp': the panorama (full sphere, 6 px/deg in the first generation); fisheye views get their ring from
                 the mechanical image circle applied at render time.
        Fisheye renders are not used as sources: fish_220 has 4.9 px/deg, below the panorama, and fish_180 is
        only slightly sharper while covering few views; native-render tuples use them directly."""
        erp = "spherical", {"hfov": torch.tensor(2 * np.pi), "vfov": torch.tensor(np.pi)}
        if pref != "pin":
            return "erp", *erp
        for bn in self._PIN_BY_SHARPNESS:
            if not os.path.isfile(f"{s['p']}/{bn}_pose{pose:02d}_pack.npz"): continue
            sm, sp = self._src_intr(s, bn); _, _, mask = self._load_src(s, pose, bn)
            try: c = GA.target_coverage(sm, sp, mask, tgt_model, tgt_params, H, W, R=Rarg)
            except Exception: c = -1.0
            if c >= 0.999: return bn, sm, sp               # full coverage: sharp source, no black region
        return "erp", *erp

    # ---------------------------------------------------------------- camera priors, distortion, legality check
    def _sample_pp(self, rng):
        """Principal-point offset (ppx, ppy) as a fraction of (W, H). With probability 1 - p_pp_extreme, a
        Gaussian per axis with sigma pp_sigma (1.2%), clipped at pp_ground_max (5%): sensor-lens decentering is
        small and roughly isotropic in calibration statistics of compact and smartphone cameras (Sanz-Ablanedo
        et al., Sensors 2010; Patonis, Sensors 2023). With probability p_pp_extreme (10%), uniform within
        +/-pp_max (22%), covering zoom lenses (Clarke et al., The Photogrammetric Record 1998), off-centre
        fisheye sensors and deliberate crops."""
        if rng.random() < self.p_pp_extreme:
            return float(rng.uniform(-self.pp_max, self.pp_max)), float(rng.uniform(-self.pp_max, self.pp_max))
        g = self.pp_ground_max
        return (float(np.clip(rng.normal(0.0, self.pp_sigma), -g, g)),
                float(np.clip(rng.normal(0.0, self.pp_sigma), -g, g)))

    def _sample_roll(self, rng):
        """Camera roll about the optical axis (deg). With probability 1 - p_roll_extreme: a zero-centred
        two-component Cauchy mixture (scales 0.06 and 5.7 deg, weights one third and two thirds), inspired by
        the roll model of Hold-Geoffroy et al. (TPAMI 2023), sampled by inverse CDF, clipped at +/-30 deg and
        taken mod 360. Otherwise uniform over 0-360 deg. Near-level roll also keeps narrow rectilinear views
        inside the vertical field of view of the pinhole renders, so that they can use the sharp source."""
        if rng.random() < self.p_roll_extreme:
            return float(rng.uniform(0.0, 360.0))
        g = 0.06 if rng.random() < (1.0 / 3.0) else 5.7
        return float(np.clip(g * np.tan(np.pi * (rng.random() - 0.5)), -30.0, 30.0)) % 360.0

    def _sample_tilt(self, rng):
        """Tilt of the optical axis away from the native orientation (deg); its azimuth is drawn in _view_cpu.
        With probability 1 - p_tilt_extreme: Rayleigh with sigma tilt_sigma (4 deg), the magnitude of an
        isotropic 2D Gaussian tilt (mode 4 deg, median about 4.7 deg, about 95% below 10 deg), clipped at
        tilt_max (12 deg). This is a near-level prior for indoor capture: real photographs concentrate near
        level (cf. the horizon statistics of Hold-Geoffroy et al., TPAMI 2023), whereas synthetic pipelines such
        as Perspective Fields (Jin et al., CVPR 2023) sample pitch uniformly by design. Otherwise uniform over
        [0, tilt_max], a wide tail for robustness."""
        if rng.random() < self.p_tilt_extreme:
            return float(rng.uniform(0.0, self.tilt_max))
        return float(min(rng.rayleigh(self.tilt_sigma), self.tilt_max))

    def _sample_dist(self, model, rng, fov=60.0, ar=1.5):
        """Distortion parameters with ranges taken from real calibrations (e.g. GoPro Hero3/4, Reolink).
        OpenCV: mostly barrel (k1 < 0) with k2 > 0, k2/|k1| in [0.35, 0.70], |k2| < |k1| and k3 = 0. Two rules:
          (a) monotonicity: k2 >= 1.2 * k1^2 makes the radial map monotonic everywhere, so the inverse
              converges and the source shows no speckle;
          (b) the barrel magnitude depends on the field of view: the fold occurs at a fixed radius, which a
              narrow view never reaches, so narrow lenses tolerate strong barrel, while wide rectilinear lenses
              are well corrected (strong wide-angle barrel belongs to the fisheye models). The cap on |k1|
              grows as the field of view shrinks: 0.45 at 48 deg, 0.08 at 120 deg."""
        if model == "opencv":
            if rng.random() < 0.72:                                            # barrel (k1<0), dominant in the wild
                k1cap = float(np.clip((125.0 - min(fov, 120.0)) / 77.0, 0.08, 0.45))   # FOV-dependent: small FOV -> larger range
                k1 = -k1cap * float(np.clip(abs(rng.normal(0.0, 0.60)), 0.0, 1.0))
                k2 = max(float(rng.uniform(0.35, 0.70)) * abs(k1), 1.2 * k1 * k1)      # co-sample (calibration ratio) + monotonic floor
            else:                                                              # pincushion (k1>0): all-positive => never folds
                k1 = float(np.clip(abs(rng.normal(0.0, 0.10)), 0.0, 0.20)); k2 = float(rng.uniform(0.0, 0.30)) * k1
            return dict(k1=float(k1), k2=float(k2), k3=0.0, p1=float(rng.normal(0, 7e-4)), p2=float(rng.normal(0, 7e-4)))
        if model == "fisheye624":  # Kannala-Brandt: r_d = theta*(1 + a3 theta^2 + a5 theta^4 + a7 theta^6)
            # Same monotonicity rule as OpenCV, in the incidence angle theta: dr_d/dtheta = 1 + 3 a3 theta^2 +
            # 5 a5 theta^4 > 0 when a5 >= 1.2 * a3^2, so the Newton inverse converges and the source shows no
            # speckle. a7 = 0 (a nonzero a7 could reintroduce a fold).
            a3 = float(rng.uniform(-0.07, 0.03))
            a5 = max(float(rng.uniform(0.0, 0.015)), 1.2 * a3 * a3)
            return dict(a3=a3, a5=float(a5), a7=0.0, p1=float(rng.normal(0, 5e-4)), p2=float(rng.normal(0, 5e-4)))
        if model == "eucm":     return dict(alpha=float(rng.uniform(0.50, 0.72)), beta=float(rng.uniform(0.85, 1.25)))
        if model == "mei":      return dict(xi=float(rng.uniform(0.5, 1.05)))
        return {}               # pinhole / spherical: no distortion

    def _tparams(self, model, fov, W, H, ppx, ppy, dist=None, circle_frac=None):
        cx, cy = (W - 1) / 2 + ppx * W, (H - 1) / 2 + ppy * H
        dist = dist or {}
        if model == "spherical":
            return {"hfov": torch.tensor(np.radians(min(fov, 360.))), "vfov": torch.tensor(np.radians(min(fov * H / W, 179.)))}
        if model in ("pinhole", "opencv"):    # rectilinear: `fov` is the diagonal field of view, which is fixed
            # for a lens and sensor; the horizontal and vertical fields of view follow the aspect ratio (landscape:
            # wide horizontal field, portrait: narrow). The focal length is derived from the image diagonal, and
            # the diagonal is clamped at 120 deg, so the horizontal field stays below it and tan stays finite.
            diag = float(np.hypot(W, H))
            f = (diag / 2) / np.tan(np.radians(min(fov, 120.0)) / 2); p = {"fx": f, "fy": f, "cx": cx, "cy": cy}
        else:                                   # fisheye624/mei/eucm: f = r_circle / (fov / 2) (equidistant), r_circle = circle_frac * short half-side
            short = min(W, H) / 2; rcirc = (circle_frac if circle_frac else 1.0) * short
            f = rcirc / np.radians(min(fov, 220) / 2); p = {"fx": f, "fy": f, "cx": cx, "cy": cy}
        p.update({k: float(v) for k, v in dist.items()})
        return p

    _SAFE_DIST = {"eucm": {"alpha": 0.6, "beta": 1.0}, "mei": {"xi": 0.8}}   # last-resort parameters known to be valid:
    # an empty dict would leave EUCM and Mei without their alpha/beta and xi keys (KeyError at render);
    # OpenCV and Fisheye624 accept {}.

    def _legalize(self, spec, min_valid=0.85, iters=3, device=None):
        """Distortion legality check: detects numerically degenerate distortion (non-finite rays, or a valid
        fraction below min_valid where the unprojection is not invertible), not the mechanical image circle
        (render-time geometry; with the monotonic sampling the fisheye validity domain covers about the full
        frame). Evaluated on a ~48 px grid with consistently scaled size and intrinsics. On failure every
        parameter is moved halfway toward a safe value, up to iters times, and then set to a known-good
        default. Safe values: no distortion, except EUCM alpha -> 0.6 (alpha -> 0 is the perspective model,
        more degenerate at wide fields of view, not less) and beta -> 1, and Mei xi -> 0.8. device='cpu' for
        use in dataloader workers."""
        device = device or self.dev
        sc = 48.0 / max(spec["W"], spec["H"])
        W48, H48 = max(16, int(round(spec["W"] * sc))), max(16, int(round(spec["H"] * sc)))
        ident = {"beta": 1.0, "alpha": 0.6, "xi": 0.8}                          # safe targets (others -> 0.0)
        for _ in range(iters + 1):
            tp = self._tparams(spec["model"], spec["fov"], W48, H48, spec["ppx"], spec["ppy"], spec.get("dist"), spec.get("circle_frac"))
            try:
                ray, valid = CM.unproject(spec["model"], H48, W48, tp, device)
                if torch.isfinite(ray).all() and float(valid.float().mean()) >= min_valid:
                    return spec
            except Exception:
                pass
            spec["dist"] = {k: ident.get(k, 0.0) + (v - ident.get(k, 0.0)) * 0.5 for k, v in spec.get("dist", {}).items()}
        spec["dist"] = dict(self._SAFE_DIST.get(spec["model"], {})); return spec

    def _connected_valid(self, mm):
        """A physical field of view is one simply connected region. Near a distortion fold the round-trip
        validity can be speckled (isolated valid islands, interior holes), which creates internal boundaries
        whose bleed edge filling cannot remove. Keep the largest connected valid component and fill its holes.
        No-op when the mask is all valid or all invalid (rectilinear without fold, panorama, full frame)."""
        if bool(mm.all()) or not bool(mm.any()):
            return mm
        from scipy import ndimage as _ndi
        a = mm.detach().cpu().numpy()
        lab, n = _ndi.label(a)
        if n > 1:
            sizes = _ndi.sum(np.ones(a.shape, np.float32), lab, index=np.arange(1, n + 1))
            a = lab == (1 + int(np.argmax(sizes)))
        a = _ndi.binary_fill_holes(a)
        return torch.from_numpy(a).to(mm.device)

    def _render(self, spec, res=None):
        """res=None: render at the spec's (W, H). res=int: scale (W, H) so that the long side is res, keeping
        the aspect ratio (intrinsics, field of view, mechanical image circle and principal point scale
        consistently). Used for the two-stage pass: the connectivity check at the small covis_res, the final
        render at each view's target resolution (full detail, no upsampling)."""
        s = spec["s"]; Rs, ts = self._c2w(s, spec["pose"])
        Raug = torch.from_numpy(build_R_aug(spec["roll"], spec["tilt"], spec["tilt_az"])).to(self.dev, torch.get_default_dtype())
        Rarg = Rs.T @ spec["Rw"] @ Raug
        W, H = spec["W"], spec["H"]
        if res is not None:
            sc = float(res) / max(W, H); W = max(16, int(round(W * sc))); H = max(16, int(round(H * sc)))
        ss = self.ss; Wf, Hf = W * ss, H * ss
        cfrac = spec.get("circle_frac")
        tp = self._tparams(spec["model"], spec["fov"], Wf, Hf, spec["ppx"], spec["ppy"], spec.get("dist"), cfrac)
        # Source routing (resolution only, covisibility-neutral): a pinhole render that covers the view, else the panorama.
        src_bn, sm, sp = self._route(s, spec["pose"], spec.get("src_pref", "pin"), spec["fov"], spec["model"], tp, Hf, Wf, Rarg)
        rgb, d, m = self._load_src(s, spec["pose"], src_bn)
        c4 = torch.eye(4, device=self.dev, dtype=torch.get_default_dtype()); c4[:3, :3] = Rs; c4[:3, 3] = ts
        r, dd, rr, mm, c2w = GA.resample_from_camera(sm, sp, rgb, d, m, spec["model"], tp, Hf, Wf, R=Rarg, c2w_src=c4)
        if cfrac is not None:                  # mechanical image circle: radius circle_frac * short half-side around the principal point
            cx, cy = (Wf - 1) / 2 + spec["ppx"] * Wf, (Hf - 1) / 2 + spec["ppy"] * Hf
            rpx = cfrac * (min(Wf, Hf) / 2)
            ys, xs = torch.meshgrid(torch.arange(Hf, device=self.dev, dtype=torch.get_default_dtype()), torch.arange(Wf, device=self.dev, dtype=torch.get_default_dtype()), indexing="ij")
            mm = mm & (torch.sqrt((xs - cx) ** 2 + (ys - cy) ** 2) <= rpx)
        mm = self._connected_valid(mm)          # a physical field of view is simply connected: keep the largest
                                                # valid component and fill its holes (a speckled mask near a
                                                # distortion fold would bleed at its boundaries)
        if ss > 1:                                                      # anti-aliasing: average RGB, subsample geometry (depth is not blended)
            # Mask-aware pooling: divide the pooled RGB by the pooled valid fraction, so that invalid pixels do
            # not darken the boundary average (a dark ring). Blocks without valid pixels give 0.
            num = F.avg_pool2d((r * mm.unsqueeze(-1)).permute(2, 0, 1)[None], ss)[0]
            den = F.avg_pool2d(mm.to(r.dtype)[None, None], ss)[0]
            rgb_o = (num / den.clamp_min(1e-6)).permute(1, 2, 0)
            o = ss // 2
            dd = dd[o::ss, o::ss][:H, :W]; mm = mm[o::ss, o::ss][:H, :W]
            rr = rr[o::ss, o::ss][:H, :W]; rr = rr / rr.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        else:
            rgb_o = r
        rgb_o = torch.where(mm.unsqueeze(-1), rgb_o, torch.zeros_like(rgb_o))   # invalid pixels black, exactly as in the mask
        # Unprojection outside the field of view of fisheye, EUCM and Mei models gives NaN rays at invalid pixels;
        # NaN * depth would propagate into pts3d (the base dataset asserts finite values). Set the invalid rays and
        # depth to zero so that pts3d is finite everywhere.
        rr = torch.where(mm.unsqueeze(-1), torch.nan_to_num(rr, nan=0.0, posinf=0.0, neginf=0.0), torch.zeros_like(rr))
        dd = torch.where(mm, torch.nan_to_num(dd, nan=0.0, posinf=0.0, neginf=0.0), torch.zeros_like(dd))
        return dict(rays=rr.reshape(-1, 3), depth=dd.reshape(-1), valid=mm.reshape(-1), R=c2w[:3, :3], t=c2w[:3, 3],
                    H=H, W=W, c2w=c2w, rgb=rgb_o.clamp(0, 1), depth2d=dd, valid2d=mm, spec=spec, src=src_bn)

    def _toward_native(self, spec, frac):
        if spec.get("erp_full"): return dict(spec)
        sp = dict(spec)
        sp["fov"] = spec["fov"] * (1 - frac) + spec["fov_native"] * frac
        sp["ppx"] = spec["ppx"] * (1 - frac); sp["ppy"] = spec["ppy"] * (1 - frac); sp["tilt"] = spec["tilt"] * (1 - frac)
        ar = spec["ar"] * (1 - frac) + 1.0 * frac                          # aspect ratio toward square
        if ar >= 1: W = self.D; H = max(16, int(round(self.D / ar)))
        else: H = self.D; W = max(16, int(round(self.D * ar)))
        sp["ar"] = ar; sp["W"] = W; sp["H"] = H
        new_model = _model_for_fov(sp["fov"]) if spec["mode"] == "A" else spec["model"]
        if new_model in ("opencv", "pinhole"):
            sp["fov"] = min(sp["fov"], 120.0)                  # rectilinear limit: opencv/pinhole cannot image beyond
                                                                # about 120 deg (tan diverges); the aimed-mode restore
                                                                # widens toward fov_native >= 150, so cap at 120
        if new_model != spec["model"]:
            sp["dist"] = {}                                    # model changed on restore: drop its distortion
            sp["circle_frac"] = spec.get("circle_frac") if new_model in ("fisheye624", "mei", "eucm") else None
        sp["model"] = new_model
        return sp

    # ---------------------------------------------------------------- spec sampling
    def _shape(self, rng, fov, model="opencv"):
        """Aspect ratio (sensor aspect >= 1 times orientation) and planned size (long side D). Sensor aspect
        ratios in the wild cluster at 4:3 to 16:9 with a tail to ultrawide; captures are mostly landscape.
        Portrait framing is common for phone cameras, which are rectilinear at any field of view (main camera
        about 80 deg diagonal, ultrawide about 120 deg), so all rectilinear views use p_portrait. Fisheye and
        spherical cameras are mostly landscape-mounted, so their portrait probability is halved rather than set
        near zero: portrait fisheye views and panorama crops exist (vertical action-camera mounts, portrait crops
        of 360 deg images), and a robustness floor is kept, as for the fisheye and panorama type weights."""
        p_port = self.p_portrait if model in ("opencv", "pinhole") else self.p_portrait * 0.5
        # Standard formats as point masses: real aspect ratios concentrate at 16:9 (video), 4:3 and 3:2 (photo),
        # 1:1 and 2:1 (panorama), which a smooth log-normal misses. This also gives every camera type the exact
        # aspect ratios of the native renders (1.0, 16:9, 2.0), so the aspect ratio alone does not reveal the
        # camera type of a native render.
        if rng.random() < p_port:                                                   # portrait (phones): 3:4 .. ~1:2
            if rng.random() < 0.30:
                ar = float(rng.choice([3/4, 2/3, 9/16], p=[0.40, 0.35, 0.25]))
            else:
                ar = 1.0 / float(np.clip(np.exp(rng.normal(np.log(1.5), 0.22)), 1.0, 2.0))
        else:                                                                       # landscape: 4:3 .. ultrawide/pano tail
            if rng.random() < 0.30:
                ar = float(rng.choice([4/3, 3/2, 16/9, 1.0, 2.0], p=[0.26, 0.20, 0.34, 0.12, 0.08]))
            else:
                ar = float(np.clip(np.exp(rng.normal(np.log(1.6), 0.34)), 1.0, 3.2))
        if ar >= 1: W = self.D; H = max(16, int(round(self.D / ar)))
        else: H = self.D; W = max(16, int(round(self.D * ar)))
        return W, H, ar

    def _fov_floor_src(self, W, H, res_target=None):
        """Source-resolution floor: the minimal diagonal field of view d_min (deg) at which a rectilinear view
        rendered from the panorama (6 px/deg) is not upsampled at its final resolution. Every axis must satisfy
        6 * fov_axis >= px_axis, with fov_axis = 2 * atan(tan(d / 2) * axis / diag). About 89 deg at aspect
        ratio 3.2 and 106 deg at aspect ratio 1.0 for a 518 px long side."""
        res = float(res_target or self.res_hint); diag = float(np.hypot(W, H)); ar = W / H
        Px = res if ar >= 1 else res * ar; Py = res / ar if ar >= 1 else res
        t = max(np.tan(np.radians(Px / 12.0)) / (W / diag),          # Px/12 deg = (Px / 6px-per-deg) / 2
                np.tan(np.radians(Py / 12.0)) / (H / diag))
        return float(np.degrees(2.0 * np.arctan(t)))

    def _sample_type(self, rng, fov_floor=48.0):
        """Sample (type, model, field of view). fov_floor raises the lower end of each type's range. The aimed
        mode uses 90 deg: its orientations differ from the native ones, so its views mostly come from the panorama
        (6 px/deg), which reaches the training resolution only from about 86 deg; narrower views come from the
        native mode, whose native orientation can use the sharp pinhole renders."""
        while True:
            i = int(rng.choice(len(TYPES), p=_W)); name, model, _, lo, hi = TYPES[i]
            lo = max(lo, fov_floor)
            if lo <= hi: break                                    # this type's range can satisfy the floor
        # Rectilinear: log-normal around the 26 mm-equivalent smartphone main camera (2*atan(43.3/52) = 79.5 deg
        # diagonal, about 72 deg horizontal in landscape), clipped to the range; the native mode (floor 48) gives
        # the main-camera cluster, the aimed mode (floor 90) ultrawide views, like the main and ultrawide cameras of
        # smartphones. Other models: log-uniform over the range (common focal lengths are roughly geometric).
        if model in ("opencv", "pinhole"):
            fov = float(np.clip(np.exp(rng.normal(np.log(80.0), 0.32)), lo, hi))
        else:
            fov = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
        return name, model, fov

    def _sample_model_for_fov(self, fov, rng):
        """Sample a camera model for a given field of view: among the types whose range covers `fov` (full
        panorama excluded), weighted by the type weights; pinhole below all ranges and spherical above."""
        comp = [(m, w) for (n, m, w, lo, hi) in TYPES if n != "erp_full" and lo <= fov <= hi]
        if not comp:                                             # fov outside all type ranges
            return "pinhole" if fov < 60 else "spherical"
        models = [c[0] for c in comp]; ws = np.array([c[1] for c in comp], float); ws /= ws.sum()
        return str(models[int(rng.choice(len(models), p=ws))])

    def _c2w_np(self, s, pose):
        """Native OpenCV camera-to-world rotation (R_unicol @ FLIP) as a numpy 3x3, for plans built in CPU workers."""
        C = np.asarray(s["meta"][f"erp_pose{pose:02d}"]["camera_to_world_unicol_4x4"], np.float64)
        return (C[:3, :3] @ _FLIP.cpu().numpy())

    def _sample_circle_frac(self, model, W, H, rng):
        """Radius of the mechanical image circle of fisheye models, in units of the short half-side.
        1: inscribed circle (black outside it; only for balanced aspect ratios); long/short: the circle spans
        the long axis and leaves black corners (the common case in the wild); >= diag/short: full frame (no
        black). Mostly sampled between long/short and diag/short, which gives black corners at any aspect ratio
        rather than black bands at the sides."""
        if model not in ("fisheye624", "mei", "eucm"): return None   # rectilinear and spherical views fill the frame
        short = min(W, H); long_frac = max(W, H) / short; diag_frac = float(np.hypot(W, H) / short)
        u = rng.random()
        if u < 0.28: return float(rng.uniform(diag_frac * 1.05, diag_frac * 1.20))   # 28%: full frame, no black
        if u < 0.43 and long_frac < 1.6: return float(rng.uniform(1.0, long_frac))   # 15%: circular image, aspect ratio < 1.6 only
        return float(rng.uniform(max(long_frac * 0.9, 1.0), diag_frac * 0.99))       # otherwise: black corners

    def _src_pref(self, model, fov, rng):
        """Source preference, sampled on the CPU; it affects resolution only (black regions come from the
        mechanical image circle). Rectilinear views up to 112 deg prefer a pinhole render ('pin': 18-40 px/deg,
        sharp full frame); all other views use the panorama ('erp': full coverage, and the mechanical image
        circle reproduces the ring of a real fisheye). Fisheye renders are not used as sources (fish_220 has
        4.9 px/deg, below the panorama's 6; fish_180 is only slightly sharper and covers few views); the real
        fisheye renders keep full quality in native-render tuples."""
        if model in ("pinhole", "opencv") and fov <= 112.0: return "pin"   # 111.7 deg is the diagonal field of
        # view of pin_14mm, the widest pinhole render, i.e. its coverage limit; the per-view coverage check in
        # _route (tilt, roll, principal point and orientation included) decides, so this test only screens
        # candidates. Wide rectilinear views are therefore sharp whenever a pinhole render covers them.
        return "erp"

    def _view_cpu(self, s, pose, mode, model, fov, fov_native, rng, Rw_np=None, gaze=None, type_name=None,
                  res_target=None):
        """Build one per-view CPU spec (picklable; no torch, GPU or scene-tensor references).
        erp_full: full 360 x 180 panorama with aspect ratio 2:1 and a random SO(3) content rotation, without
        principal-point offset, distortion or image circle.
        res_target: final long side of the view, used by the source-resolution floor and the image-circle cap
        (None: res_hint)."""
        if type_name == "erp_full":
            W, H, ar = self.D * 2, self.D, 2.0                                  # equirectangular 2:1
            Rw_np = GAUG.random_so3(rng)
            dist, cfrac, roll, tilt, ppx, ppy = {}, None, 0.0, 0.0, 0.0, 0.0
        else:
            W, H, ar = self._shape(rng, fov, model)
            if model in ("pinhole", "opencv") and (mode == "B" or fov > 112.0):  # rectilinear, aimed or wider than
                # 112 deg (panorama source expected): source-resolution floor. Native-mode views up to 112 deg try a
                # pinhole render (see _src_pref) and are not raised when they fall back to the panorama: raising them
                # would empty part of the joint field-of-view and aspect-ratio distribution, and the modulation-
                # transfer randomisation of the optics layer masks the slight softness of that fallback.
                _c = float(rng.uniform(1.0, 1.10))               # jitter of the target density: an exact floor would
                # put these views at exactly the panorama's density, a cue to the source route; the narrow range
                # [1.0, 1.10] removes it and keeps d_min within a few degrees of the exact floor.
                fov = max(fov, min(self._fov_floor_src(W, H, float(res_target or self.res_hint) * _c), 120.0))   # source-resolution floor (<= 120 deg)
            dist = self._sample_dist(model, rng, fov, ar); cfrac = self._sample_circle_frac(model, W, H, rng)
            if cfrac is not None:                                                # image-circle cap: the circle must not
                shp = (float(res_target or self.res_hint) / max(ar, 1.0 / ar)) / 2.0   # exceed the panorama's 6 px/deg:
                cfrac = float(min(cfrac, max(1.0, 3.0 * fov / shp)))             # cfrac * short_half / (fov / 2) <= 6, i.e.
                                                                                 # cfrac <= 3 * fov / short_half (at least 1.0, the inscribed circle)
            roll, tilt = self._sample_roll(rng), self._sample_tilt(rng)         # 90% near-level priors, 10% wide uniform tails
            ppx, ppy = self._sample_pp(rng)                                     # 90% small decentering, 10% wide tail (crops, zoom lenses)
        sp = dict(mode=mode, pose=int(pose), model=model, fov=float(fov), fov_native=float(fov_native), ar=ar,
                  W=W, H=H, dist=dist, gaze=gaze, src_pref=self._src_pref(model, fov, rng), circle_frac=cfrac,
                  erp_full=(type_name == "erp_full"),
                  Rw_np=(None if Rw_np is None else np.asarray(Rw_np, np.float64).tolist()),
                  ppx=ppx, ppy=ppy, roll=roll, tilt=tilt, tilt_az=float(rng.uniform(0, 360)))
        return self._legalize(sp, device="cpu")

    def _plan_modeA(self, s, K, rng, res_list=None):
        pose_of = np.array([s["pose"][i] for i in range(s["cov"].shape[0])])   # frame -> pose
        walk, ok = _walk(s["cov"], K, rng, thres=self.walk_thres, pose_of=pose_of)   # distinct optical centres, covisibility headroom
        if not ok: return None
        # Native mode: panorama pose and native orientation, camera type and field of view drawn with the same type
        # weights as in the aimed mode. fov_native = 200: a disconnected view is restored by widening toward 200 deg.
        contents = [self._sample_type(rng) for _ in walk]
        out = []
        for slot in range(K):
            pose = s["pose"][int(walk[slot])]
            name, model, fov = contents[slot]
            out.append(self._view_cpu(s, pose, "A", model, fov, 200.0, rng, Rw_np=self._c2w_np(s, pose), type_name=name,
                                      res_target=(None if res_list is None else res_list[slot])))
        return out

    def _plan_modeB(self, s, K, rng, res_list=None):
        erp = s["erp"]                                          # one panorama per pose: distinct panoramas are distinct poses
        if len(erp) < K: return None
        # Walk on the pose-level covisibility (see _scene), not on the panorama-to-panorama covisibility, whose
        # full-sphere normalisation gives systematically low values and leaves few edges above walk_thres at
        # large K. The check after rendering (0.25) decides legality.
        epl = [s["pose"][i] for i in erp]                       # poses with a panorama pack
        ep_idx = [s["pidx"][pid] for pid in epl]
        sub = s["pcov"][np.ix_(ep_idx, ep_idx)]
        walk, ok = _walk(sub, K, rng, thres=self.walk_thres); tier = 1
        if not ok and self.walk_thres_b_relax < self.walk_thres:     # relaxed threshold (see __init__)
            walk, ok = _walk(sub, K, rng, thres=self.walk_thres_b_relax); tier = 2
        if not ok: return None
        poses = [epl[int(i)] for i in walk]
        convergent = rng.random() < self.p_conv
        contents = [self._sample_type(rng, fov_floor=90.0) for _ in poses]   # aimed mode: fields of view from 90 deg
        out = []
        for slot in range(K):                            # orientation set in realize() (needs the surface point), except erp_full (random SO(3) here)
            name, model, fov = contents[slot]
            out.append(self._view_cpu(s, poses[slot], "B", model, fov, max(fov, 150), rng, Rw_np=None,
                                      gaze=("conv" if convergent else "par"), type_name=name,
                                      res_target=(None if res_list is None else res_list[slot])))
        out[0]["b_tier"] = tier          # walk tier (1: walk_thres, 2: relaxed), logged as 'bt' in the sampling record
        return out

    def _surface_point(self, s, poses):
        rgb, d, m = self._erp(s, poses[0]); Rs, t = self._c2w(s, poses[0])
        d = d[::2, ::2]; m = m[::2, ::2]; H, W = d.shape          # half-resolution grid: only the median of the
        # covisible points is needed, and half-resolution sampling of the sphere gives it at a quarter of the cost.
        ray, _ = CM.unproject("spherical", H, W, {"hfov": torch.tensor(2 * np.pi), "vfov": torch.tensor(np.pi)}, self.dev)
        P0 = ((ray.to(torch.get_default_dtype()) * d.unsqueeze(-1)) @ Rs.T + t)[m]
        P0 = P0[torch.randperm(P0.shape[0])[:8000]]; keep = torch.ones(P0.shape[0], dtype=torch.bool, device=self.dev)
        for p in poses[1:]:
            rgb2, d2, m2 = self._erp(s, p); Rs2, t2 = self._c2w(s, p); H2, W2 = d2.shape
            Pc = (P0 - t2) @ Rs2; rg = Pc.norm(dim=-1).clamp_min(1e-8); wd = Pc / rg.unsqueeze(-1)
            uv, _ = CM.project("spherical", wd, {"hfov": torch.tensor(2 * np.pi), "vfov": torch.tensor(np.pi)}, H2, W2)
            col = uv[..., 0].round().long().clamp(0, W2 - 1); row = uv[..., 1].round().long().clamp(0, H2 - 1)
            keep &= m2[row, col] & ((rg - d2[row, col]).abs() <= 0.1 + 0.05 * d2[row, col])
        co = P0[keep]; return co.median(0).values if co.shape[0] > 20 else P0.median(0).values

    # ---------------------------------------------------------------- public: CPU plan / GPU realize (multi-worker)
    def sample_plan(self, scene_name, K, rng, res=None):
        """CPU only (no CUDA), safe in dataloader workers: select a connected pose set and sample the per-view
        specs. Returns a picklable plan {scene_name, mode, K, views} or None. The aimed-mode orientation is set
        in realize() (it needs the surface point, computed on the GPU); the native-mode orientation is the native
        OpenCV rotation of the pose, stored in the plan. res: final long side per view (int or list), used by the
        source-resolution floor (None: res_hint). A plan that fails is retried in the other mode."""
        s = self._scene(scene_name)
        res_list = (list(res) if isinstance(res, (list, tuple)) else [res] * K) if res is not None else None
        mode = "B" if rng.random() < self.p_modeB else "A"
        views = self._plan_modeB(s, K, rng, res_list) if mode == "B" else self._plan_modeA(s, K, rng, res_list)
        if views is None:                                                  # retry in the other mode
            views = self._plan_modeA(s, K, rng, res_list) if mode == "B" else self._plan_modeB(s, K, rng, res_list)
            mode = "A" if mode == "B" else "B"
        if views is None: return None
        return dict(scene_name=scene_name, mode=mode, K=int(K), views=views)

    def realize(self, plan, res=None, max_restore=3):
        """Render a CPU plan into K connected views on the sampler's device.
        Two stages: the connectivity check renders at the small covis_res (covisibility is a geometric overlap
        and robust to resolution, so repeated renders are cheap); the kept views are then rendered once at their
        target resolution (`res`: long side per view, or self.D if None) for full source detail without
        upsampling. The default dtype is set to self.work_dtype (float64) for the whole pass, so that all tensors
        share one dtype, and restored afterwards; the caller casts the outputs to float32."""
        if plan is None: return None
        _prev_dtype = torch.get_default_dtype()
        torch.set_default_dtype(self.work_dtype)
        try:
            s = self._scene(plan["scene_name"]); cpu_views = plan["views"]
            if plan["mode"] == "B":                                            # aimed mode: orientation toward the surface point
                poses = [v["pose"] for v in cpu_views]; C = self._surface_point(s, poses)
                pos = torch.stack([self._c2w(s, p)[1] for p in poses]); dmean = (C - pos.mean(0)); dmean = dmean / dmean.norm()
                Rws = []
                for i, v in enumerate(cpu_views):
                    if v.get("Rw_np") is not None:                                  # erp_full: its SO(3) content rotation, not the aimed orientation
                        Rws.append(torch.tensor(v["Rw_np"], device=self.dev, dtype=torch.get_default_dtype())); continue
                    # aimed orientation; see connectivity.look_rotation for the 180-degree roll of these views
                    d = (C - pos[i]) if v["gaze"] == "conv" else dmean; d = d / d.norm(); Rws.append(CN.look_rotation(d))
            else:
                Rws = [torch.tensor(v["Rw_np"], device=self.dev, dtype=torch.get_default_dtype()) for v in cpu_views]
            specs = [dict(v, s=s, Rw=Rws[i]) for i, v in enumerate(cpu_views)]
            # Disconnected views are moved toward their native configuration; a tuple that stays disconnected is
            # rejected, and the caller falls back to native renders. Disconnected views are not replaced by full
            # panoramas, which would push the share of full panoramas far above its 15% weight.
            r = CN.enforce_connectivity(specs, lambda sp: self._render(sp, res=self.covis_res), self._toward_native,
                                        thres=0.25, max_restore=max_restore)
            if not r["connected"]:                                             # disconnected tuple: the caller rejects it
                return r["views"], dict(mode=plan["mode"], b_tier=plan["views"][0].get("b_tier", 0), connected=False, n_restore=r["n_restore"],   # no final render
                                        dropped=r["dropped"], keep=r["keep"])
            fspecs = r["specs"]                                                # possibly restored specs
            K = len(fspecs)
            bd = r["graph"]["bidir"]                                           # covis_min for the sampling record: weakest best-neighbour edge
            cvmin = float(min(bd[i][np.arange(K) != i].max() for i in range(K))) if K > 1 else 1.0
            reslist = (list(res) if isinstance(res, (list, tuple)) else [res] * K) if res is not None else [self.D] * K
            views = [self._render(fspecs[i], res=int(reslist[i])) for i in range(K)]   # final render at the target resolution, once
            return views, dict(mode=plan["mode"], b_tier=plan["views"][0].get("b_tier", 0), connected=r["connected"], n_restore=r["n_restore"],
                               dropped=r["dropped"], keep=r["keep"], covis_min=cvmin)
        finally:
            torch.set_default_dtype(_prev_dtype)

    def build(self, scene_name, K, rng, res=None, max_restore=3):
        """Convenience: sample_plan (CPU) followed by realize (GPU).
        res: per-view target long side, used by the source-resolution floor and the final render (None: res_hint
        for the floor, self.D for the render)."""
        plan = self.sample_plan(scene_name, K, rng, res=res)
        return self.realize(plan, res=res, max_restore=max_restore) if plan is not None else None


def build_R_aug(roll_deg, tilt_deg, tilt_az):
    """Augmentation rotation in the camera frame: roll about the optical axis (z), then an off-axis
    tilt of ``tilt_deg`` about the in-plane axis at azimuth ``tilt_az`` (radians). The resampled
    orientation is native_c2w_rot @ build_R_aug(...). Returns a numpy 3x3 matrix."""
    rr = np.radians(roll_deg); cr, sr = np.cos(rr), np.sin(rr)
    Rroll = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1.]])
    ax = np.array([np.cos(tilt_az), np.sin(tilt_az), 0.]); a = np.radians(tilt_deg)
    Kx = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    Rtilt = np.eye(3) + np.sin(a) * Kx + (1 - np.cos(a)) * Kx @ Kx
    return Rtilt @ Rroll


def _walk(cov, K, rng, thres=0.25, retries=8, pose_of=None):
    """Connected random walk (depth-first, with backtracking) over a covisibility matrix, as in the base
    dataset's _random_walk_sampling (edge weight: mean of both directions, normalised by the diagonal).
    If pose_of (frame -> pose) is given, candidates whose pose is already in the walk are skipped, so that no
    two views share an optical centre (zero parallax)."""
    N = cov.shape[0]; best = []
    for _ in range(retries):
        start = int(rng.integers(N)); visited = {start}; walk = [start]; stack = [start]
        vposes = {int(pose_of[start])} if pose_of is not None else None
        while len(walk) < K and stack:
            c = stack[-1]; pc = (cov[c, :] + cov[:, c]) / 2.0; pc = pc / (pc[c] + 1e-8); pc[c] = 0
            cand = [j for j in np.flatnonzero(pc > thres)
                    if j not in visited and (pose_of is None or int(pose_of[j]) not in vposes)]
            if cand:
                nx = int(rng.choice(cand)); walk.append(nx); visited.add(nx); stack.append(nx)
                if pose_of is not None: vposes.add(int(pose_of[nx]))
            else: stack.pop()
        if len(walk) > len(best): best = walk
        if len(walk) >= K: return np.array(walk[:K]), True
    return np.array(best), len(best) >= K
