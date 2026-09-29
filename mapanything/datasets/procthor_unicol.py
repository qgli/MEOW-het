# Copyright (c) 2026 The MEOW Authors.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""Multi-camera training dataset over rendered scene packs (both scene generations).

Each scene directory holds ``*_pack.npz`` packs and a ``metadata.json``:

- ``<tag>_pack.npz`` with keys
    - ``rgb``   : float16 ``(H, W, 3)`` in ``[0, 1]``
    - ``rays``  : float16 ``(H, W, 3)`` unit ray directions in the renderer camera frame
                  (X right, Y up, Z forward)
    - ``depth`` : float16 ``(H, W)`` distance along the ray in metres
    - ``mask``  : uint8 ``(H, W)``, 1 = valid pixel
- ``metadata.json`` with per-frame ``camera_to_world_unicol_4x4``, ``base_type``
  (``pinhole`` / ``fisheye`` / ``erp``), ``base_name`` and ``tag``;
- ``covisibility/v0/{covisibility.npy, frame_meta.json}``: the offline covisibility matrix.

The renderer camera frame is converted to the OpenCV convention used by MapAnything (X right,
Y down, Z forward) by flipping the camera Y axis of the rays and of the camera pose; world
geometry is unchanged.

Tuples are either native renders selected on the offline covisibility, or, with
``camera_sampling``, camera-sampled tuples built by the online camera sampler
(:class:`mapanything.datasets.camera_sampler.CameraSampler`), which falls back to native renders
when no connected tuple can be built.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from mapanything.datasets import geom_aug as _gaug

from mapanything.datasets.base.base_dataset import BaseDataset
from mapanything.utils.cropping import (
    crop_image_and_other_optional_info,
    rescale_image_and_other_optional_info,
)


_PACK_RE = re.compile(r"^(?P<base>.+)_pose(?P<pose>\d+)_pack\.npz$")


# --------------------------------------------------------------------------
# Covisibility-aware multi-camera sampling
# --------------------------------------------------------------------------
# Symmetric (minimum of both directions) covisibility thresholds per unordered camera-type
# pair, near the 25th to 30th percentile of each pair type on a sample of scenes. Used by the
# type-diverse sampler (validation of Stage 1).
_PAIR_BICOV_THRES = {
    frozenset(["pinhole", "pinhole"]): 0.05,
    frozenset(["pinhole", "fisheye"]): 0.10,
    frozenset(["pinhole", "erp"]): 0.10,
    frozenset(["fisheye", "fisheye"]): 0.20,
    frozenset(["fisheye", "erp"]): 0.20,
    frozenset(["erp", "erp"]): 0.30,
}


def _bicov(cov: np.ndarray, i: int, j: int) -> float:
    """Symmetric covisibility as min of asymmetric directions."""
    return float(min(cov[i, j], cov[j, i]))


def _pair_visible(cov: np.ndarray, types, i: int, j: int) -> bool:
    thres = _PAIR_BICOV_THRES[frozenset([types[i], types[j]])]
    return _bicov(cov, i, j) >= thres


# --------------------------------------------------------------------------
# Per-camera-pair covisibility thresholds used for native-render tuples
# --------------------------------------------------------------------------
# Minimum symmetric covisibility for each pair of first-generation cameras (keys are
# tuple(sorted([base_name_i, base_name_j]))). The camera pool is the panorama, the two
# fisheyes and the 14, 24 and 40 mm pinholes; the 70 and 135 mm pinholes are excluded.
_PAIR_BICOV_THRES_N5: Dict[tuple, float] = {
    # Panorama x panorama
    ("erp", "erp"):              0.11,
    # Panorama x fisheye
    ("erp", "fish_180"):         0.13,
    ("erp", "fish_220"):         0.13,
    # Panorama x pinhole
    ("erp", "pin_14mm"):         0.09,
    ("erp", "pin_24mm"):         0.07,
    ("erp", "pin_40mm"):         0.06,
    # Fisheye x fisheye
    ("fish_180", "fish_180"):    0.11,
    ("fish_180", "fish_220"):    0.13,
    ("fish_220", "fish_220"):    0.11,
    # Fisheye x pinhole
    ("fish_180", "pin_14mm"):    0.10,
    ("fish_180", "pin_24mm"):    0.09,
    ("fish_180", "pin_40mm"):    0.05,
    ("fish_220", "pin_14mm"):    0.09,
    ("fish_220", "pin_24mm"):    0.06,
    ("fish_220", "pin_40mm"):    0.06,
    # Pinhole x pinhole
    ("pin_14mm", "pin_14mm"):    0.11,
    ("pin_14mm", "pin_24mm"):    0.11,
    ("pin_14mm", "pin_40mm"):    0.08,
    ("pin_24mm", "pin_24mm"):    0.10,
    ("pin_24mm", "pin_40mm"):    0.10,
    ("pin_40mm", "pin_40mm"):    0.09,
}

# Cameras allowed in native-render tuples (pool filter applied before sampling).
_N5_ALLOWED_BASE_NAMES = frozenset(
    bn for pair in _PAIR_BICOV_THRES_N5 for bn in pair
)


def _bicov_thres_n5(base_name_i: str, base_name_j: str) -> float:
    """Look up the covisibility threshold of a camera pair.

    Raises KeyError for the excluded 70 and 135 mm pinholes; callers filter the pool first.
    """
    key = (base_name_i, base_name_j) if base_name_i <= base_name_j else (
        base_name_j,
        base_name_i,
    )
    return _PAIR_BICOV_THRES_N5[key]


def _pair_visible_n5(cov: np.ndarray, bnames, i: int, j: int) -> bool:
    """Symmetric covisibility check with the per-camera-pair thresholds."""
    thres = _bicov_thres_n5(bnames[i], bnames[j])
    return _bicov(cov, i, j) >= thres


def _unicol_to_opencv(rays_unicol: np.ndarray, c2w_unicol: np.ndarray):
    """Flip the Y axis of the camera frame (renderer frame, Y up -> OpenCV frame, Y down).

    Args:
        rays_unicol: ``(H, W, 3)`` ray-direction map in the renderer camera frame.
        c2w_unicol:  ``(4, 4)`` camera-to-world matrix in the renderer convention.

    Returns:
        ``(rays_opencv, c2w_opencv)`` tuple. World geometry is preserved
        because ``R_unicol @ diag(1, -1, 1) @ diag(1, -1, 1) @ r_unicol ==
        R_unicol @ r_unicol``.
    """
    rays_cv = rays_unicol.copy()
    rays_cv[..., 1] *= -1.0
    c2w_cv = c2w_unicol.copy()
    c2w_cv[:3, 1] *= -1.0
    return rays_cv, c2w_cv


class ProcThorUnicol(BaseDataset):
    """Rendered multi-camera scenes of both generations, with the optional camera sampler."""

    # Pinhole cameras kept for training (equivalent focal length up to 50 mm)
    DEFAULT_PINHOLE_FOCAL_NAMES = ("pin_14mm", "pin_24mm", "pin_40mm")

    def __init__(
        self,
        *args,
        ROOT: Optional[str] = None,
        splits_dir: Optional[str] = None,
        camera_types: Optional[List[str]] = None,
        pinhole_focal_names: Optional[List[str]] = None,
        frame_stats_path: Optional[str] = None,
        min_rgb_std: float = 0.05,
        overfit_num_sets: Optional[int] = None,
        sample_specific_scene: bool = False,
        specific_scene_name: Optional[str] = None,
        n5_sampling: bool = False,
        n5_sampling_v2: bool = False,
        n5_v2_feasibility_3type_path: Optional[str] = None,
        n5_v2_feasibility_6name_path: Optional[str] = None,
        n5_v2_tier_weights: Optional[tuple] = None,
        n5_v2_walk_thres: float = 0.25,
        n5_v2_walk_retries: int = 8,
        preserve_info_resize: bool = False,
        preserve_info_types: Optional[List[str]] = None,
        variable_resolution: bool = False,
        camera_sampling: bool = False,
        camera_sampling_p: float = 1.0,
        camera_sampling_device: str = "cuda",
        camera_sampling_photo_extras: bool = True,
        camera_sampling_retries: int = 1,
        photo_mtf_p: float = 0.50,
        photo_mtf_vmin: float = 0.35,
        **kwargs,
    ):
        """
        Args:
            ROOT: Directory containing one sub-directory per scene (required).
            splits_dir: Directory containing ``train.json`` / ``val.json`` (lists of scene
                directory names). Required unless ``sample_specific_scene`` is set.
            camera_types: Subset of ``{"pinhole", "fisheye", "erp"}`` (the ``base_type`` of
                ``metadata.json``) to sample from. ``None`` = all.
            pinhole_focal_names: Pinhole ``base_name`` values to keep
                (default :pyattr:`DEFAULT_PINHOLE_FOCAL_NAMES`).
            frame_stats_path: Optional per-frame statistics file written by
                ``scripts/precompute_mask_frac.py``. When given, frames whose RGB standard
                deviation is below ``min_rgb_std`` (views facing a flat wall) are skipped.
            min_rgb_std: Threshold of the flat-wall filter.
            overfit_num_sets: If set, truncate the scene list (debugging aid).
            sample_specific_scene: If True, only ``specific_scene_name`` is used.
            specific_scene_name: Scene directory name used with ``sample_specific_scene``.
            n5_sampling: Select native-render tuples with the per-camera-pair covisibility
                thresholds (any camera combination, any tuple size).
            n5_sampling_v2: Stratified native-render sampling: connected random walks with
                per-camera-type quotas drawn from the feasibility tables (requires
                ``n5_sampling``).
            n5_v2_feasibility_3type_path / n5_v2_feasibility_6name_path: Feasibility tables
                (default ``resources/gen1_feasibility/`` in the repository).
            n5_v2_tier_weights: Probabilities of (per-pair threshold cascade, 3-type
                stratified walk, 6-camera stratified walk).
            n5_v2_walk_thres: Covisibility threshold of the connected walks.
            n5_v2_walk_retries: Walk attempts per tuple.
            preserve_info_resize: Resize the views of ``preserve_info_types`` to the target
                shape without cropping (full field of view, anisotropic scale).
            preserve_info_types: Camera types resized that way (default panoramas and fisheyes).
            variable_resolution: Each view of a tuple gets its own target shape (Stage 2).
            camera_sampling: Build camera-sampled tuples with the online camera sampler.
            camera_sampling_p: Probability of trying a camera-sampled tuple for a sample.
            camera_sampling_device: Device of the camera sampler ("cuda" resolves to the current rank).
            camera_sampling_photo_extras: Apply the optics layer to camera-sampled tuples and, when
                ``camera_sampling`` is set, also to native-render tuples.
            camera_sampling_retries: Additional attempts, each on a newly drawn scene, before falling
                back to native renders.
            photo_mtf_p / photo_mtf_vmin: Probability and minimum factor of the
                modulation-transfer randomisation of the optics layer.
        """
        super().__init__(*args, **kwargs)
        if ROOT is None:
            raise ValueError("ProcThorUnicol requires ROOT (the directory of scene folders)")
        if splits_dir is None and not sample_specific_scene:
            raise ValueError("ProcThorUnicol requires splits_dir (train.json / val.json)")
        self.ROOT = ROOT
        self.splits_dir = splits_dir
        self.camera_types = (
            set(camera_types)
            if camera_types is not None
            else {"pinhole", "fisheye", "erp"}
        )
        self.pinhole_focal_names = (
            set(pinhole_focal_names)
            if pinhole_focal_names is not None
            else set(self.DEFAULT_PINHOLE_FOCAL_NAMES)
        )
        self.min_rgb_std = float(min_rgb_std)

        # Optional flat-wall filter statistics
        self._frame_stats: Optional[Dict[str, Dict[str, Dict[str, float]]]] = None
        if frame_stats_path:
            if not os.path.isfile(frame_stats_path):
                raise FileNotFoundError(f"frame_stats_path not found: {frame_stats_path}")
            with open(frame_stats_path) as f:
                self._frame_stats = json.load(f)

        self.overfit_num_sets = overfit_num_sets
        self.sample_specific_scene = sample_specific_scene
        self.specific_scene_name = specific_scene_name

        # Full-field-of-view resizing: the configured camera types are resized directly to the
        # target (W, H) without a crop. Rays, depth and mask get the same transform, and the rays
        # are renormalised, so every pixel keeps its correct ray direction.
        self.preserve_info_resize = bool(preserve_info_resize)
        self.preserve_info_types = (
            set(preserve_info_types)
            if preserve_info_types is not None
            else {"erp", "fisheye"}
        )
        # Variable resolution: the base dataset draws one target shape per view; batch
        # consistency is kept by seeding that draw on values shared by the whole batch.
        self.variable_resolution = bool(variable_resolution)

        # Native-render tuple selection with the per-camera-pair thresholds
        self.n5_sampling = bool(n5_sampling)

        # Stratified native-render selection: a tier is drawn per tuple with n5_v2_tier_weights:
        # 0 = per-pair threshold cascade, 1 = connected walk with pinhole / fisheye / panorama
        # quotas, 2 = connected walk with quotas over the six cameras. Quotas are drawn from the
        # feasibility tables built by scripts/precompute_feasibility_map.py.
        self.n5_sampling_v2 = bool(n5_sampling_v2)
        if self.n5_sampling_v2:
            assert self.n5_sampling, "n5_sampling_v2 requires n5_sampling=True"
            self._n5_v2_weights = tuple(
                n5_v2_tier_weights if n5_v2_tier_weights is not None
                else (0.2, 0.6, 0.2)
            )
            assert len(self._n5_v2_weights) == 3 and abs(
                sum(self._n5_v2_weights) - 1.0
            ) < 1e-6, f"n5_v2_tier_weights must sum to 1.0, got {self._n5_v2_weights}"
            self._n5_v2_walk_thres = float(n5_v2_walk_thres)
            self._n5_v2_walk_retries = int(n5_v2_walk_retries)
            default_root = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "..", "resources", "gen1_feasibility",
            )
            p3 = n5_v2_feasibility_3type_path or os.path.normpath(
                os.path.join(default_root, "feasibility_3type_full.parquet")
            )
            p6 = n5_v2_feasibility_6name_path or os.path.normpath(
                os.path.join(default_root, "feasibility_6name_full.parquet")
            )
            self._n5_v2_load_feasibility(p3, p6)
            print(
                f"Stratified native-render sampling: tier_weights={self._n5_v2_weights} "
                f"walk_thres={self._n5_v2_walk_thres}",
                flush=True,
            )

        # Online camera sampler: a fraction camera_sampling_p of the samples are camera-sampled tuples
        # resampled from the panoramas of the scene; failures fall back to native renders.
        self.camera_sampling = bool(camera_sampling)
        self.camera_sampling_p = float(camera_sampling_p)
        # Resolve "cuda" to this rank's device now (in the main process): spawned workers inherit
        # the string, and a bare "cuda" would put every worker on the first GPU.
        if camera_sampling_device == "cuda" and torch.cuda.is_available():
            camera_sampling_device = f"cuda:{torch.cuda.current_device()}"
        self.camera_sampling_device = camera_sampling_device
        self._camera_sampler = None  # created lazily
        self.camera_sampling_photo_extras = bool(camera_sampling_photo_extras)
        self.camera_sampling_retries = int(camera_sampling_retries)
        self.photo_mtf_p = float(photo_mtf_p)
        self.photo_mtf_vmin = float(photo_mtf_vmin)

        self.is_metric_scale = True  # Blender renders are metric (metres)
        self.is_synthetic = True

        self._load_data()

    # ------------------------------------------------------------------ data

    def _load_data(self):
        if self.sample_specific_scene:
            assert self.specific_scene_name is not None
            self.scenes = [self.specific_scene_name]
        else:
            split_file = os.path.join(self.splits_dir, f"{self.split}.json")
            if os.path.isfile(split_file):
                with open(split_file) as f:
                    self.scenes = sorted(json.load(f))
            else:
                raise FileNotFoundError(
                    f"Split file not found: {split_file} (create it with the split scripts)"
                )
        if self.overfit_num_sets is not None:
            self.scenes = self.scenes[: self.overfit_num_sets]
        self.num_of_scenes = len(self.scenes)

    # ----------------------------------------------------------- per-scene

    def _build_frame_index(self, scene_dir: str) -> List[Dict[str, Any]]:
        """Return the list of view descriptors for one scene.

        Each descriptor: ``{"pack_path", "base_name", "base_type",
        "pose_index", "c2w_unicol"}``.
        """
        meta_path = os.path.join(scene_dir, "metadata.json")
        with open(meta_path) as f:
            meta = json.load(f)

        scene_name = os.path.basename(scene_dir.rstrip("/"))
        scene_stats = (
            self._frame_stats.get(scene_name) if self._frame_stats else None
        )

        # Only frames whose base_type is in self.camera_types, whose pack
        # file exists on disk and that pass the focal/wall-hit filters.
        descriptors: List[Dict[str, Any]] = []
        for frame in meta["frames"]:
            bt = frame["base_type"]
            if bt not in self.camera_types:
                continue
            # Pinhole focal-length restriction (≤50 mm equiv by default)
            if bt == "pinhole" and self.pinhole_focal_names:
                if frame["base_name"] not in self.pinhole_focal_names:
                    continue
            # Wall-hit filter (rgb_std too low ⇒ frame is mostly a flat wall)
            if scene_stats is not None and self.min_rgb_std > 0:
                fs = scene_stats.get(frame["tag"])
                if fs is not None and fs.get("rgb_std", 1.0) < self.min_rgb_std:
                    continue
            pack = os.path.join(scene_dir, f"{frame['tag']}_pack.npz")
            if not os.path.isfile(pack):
                continue
            descriptors.append(
                {
                    "pack_path": pack,
                    "base_name": frame["base_name"],
                    "base_type": bt,
                    "pose_index": frame["pose_index"],
                    "c2w_unicol": np.asarray(
                        frame["camera_to_world_unicol_4x4"], dtype=np.float32
                    ),
                    "tag": frame["tag"],
                }
            )
        return descriptors

    # ------------------------------------------ covisibility-aware sampling

    def _load_covis(self, scene_dir: str):
        """Load pre-computed covisibility matrix + metadata for a scene.

        Returns ``(cov, meta_frames)`` where ``meta_frames`` is a list of dicts
        with at least ``{tag, base_name, base_type}`` keys, or ``(None, None)``
        if the precomputed file is missing.
        """
        cov_path = os.path.join(scene_dir, "covisibility", "v0", "covisibility.npy")
        meta_path = os.path.join(scene_dir, "covisibility", "v0", "frame_meta.json")
        if not os.path.isfile(cov_path) or not os.path.isfile(meta_path):
            return None, None
        cov = np.load(cov_path)
        with open(meta_path) as f:
            mo = json.load(f)
        return cov, mo["frames"]

    # -----------------------------------------------------------------
    # Type-diverse covisibility sampler (validation of Stage 1): a cascade
    # from a strict type-diversity constraint to a random draw. It never
    # raises and always returns K indices.
    # -----------------------------------------------------------------
    def _try_covis_strict(
        self, K, cov, types, bnames, erp_idx, fish_idx, pin_idx,
        pin_by_focal, max_retries,
    ):
        """L1: strict type diversity + covisibility. Returns idxs or None."""
        if K == 2:
            # 1 erp + 1 non-erp
            if not erp_idx or (not fish_idx and not pin_idx):
                return None
            for _ in range(max_retries):
                e = int(self._rng.choice(erp_idx))
                cands = [j for j in fish_idx + pin_idx
                         if _pair_visible(cov, types, e, j)]
                if not cands:
                    continue
                o = int(self._rng.choice(cands))
                idxs = np.array([e, o]); self._rng.shuffle(idxs)
                return idxs
            return None
        if K == 3:
            # 1 erp + 1 fish + 1 pin (any focal)
            if not erp_idx or not fish_idx or not pin_idx:
                return None
            for _ in range(max_retries):
                e = int(self._rng.choice(erp_idx))
                f_cands = [j for j in fish_idx if _pair_visible(cov, types, e, j)]
                if not f_cands:
                    continue
                f = int(self._rng.choice(f_cands))
                p_cands = [i for i in pin_idx
                           if _pair_visible(cov, types, e, i)
                           and _pair_visible(cov, types, f, i)]
                if not p_cands:
                    continue
                p = int(self._rng.choice(p_cands))
                idxs = np.array([e, f, p]); self._rng.shuffle(idxs)
                return idxs
            return None
        # K == 4: 1 erp + 1 fish + 2 pin (different focals)
        if not erp_idx or not fish_idx or len(pin_by_focal) < 2:
            return None
        for _ in range(max_retries):
            e = int(self._rng.choice(erp_idx))
            f_cands = [j for j in fish_idx if _pair_visible(cov, types, e, j)]
            if not f_cands:
                continue
            f = int(self._rng.choice(f_cands))
            focals = list(pin_by_focal.keys())
            self._rng.shuffle(focals)
            for fa in focals:
                p1_cands = [i for i in pin_by_focal[fa]
                            if _pair_visible(cov, types, e, i)
                            and _pair_visible(cov, types, f, i)]
                if not p1_cands:
                    continue
                p1 = int(self._rng.choice(p1_cands))
                other_focals = [ff for ff in focals if ff != fa]
                self._rng.shuffle(other_focals)
                for fb in other_focals:
                    p2_cands = [i for i in pin_by_focal[fb]
                                if _pair_visible(cov, types, e, i)
                                and _pair_visible(cov, types, f, i)
                                and _pair_visible(cov, types, p1, i)]
                    if not p2_cands:
                        continue
                    p2 = int(self._rng.choice(p2_cands))
                    idxs = np.array([e, f, p1, p2]); self._rng.shuffle(idxs)
                    return idxs
        return None

    def _try_covis_relaxed_type(
        self, K, cov, types, bnames, erp_idx, fish_idx, pin_idx,
        pin_by_focal, max_retries,
    ):
        """L2: relaxed type diversity + covisibility. Returns idxs or None.

        - K=2: 1 fish + 1 pin (drop erp requirement).
        - K=3: 1 erp + any 2 non-erp (drop strict fish+pin).
        - K=4: 1 erp + 1 fish + 2 pin (any focal, focals may coincide).
        """
        non_erp = fish_idx + pin_idx
        if K == 2:
            if not fish_idx or not pin_idx:
                return None
            for _ in range(max_retries):
                a = int(self._rng.choice(fish_idx))
                cands = [j for j in pin_idx if _pair_visible(cov, types, a, j)]
                if not cands:
                    continue
                b = int(self._rng.choice(cands))
                idxs = np.array([a, b]); self._rng.shuffle(idxs)
                return idxs
            return None
        if K == 3:
            if not erp_idx or len(non_erp) < 2:
                return None
            for _ in range(max_retries):
                e = int(self._rng.choice(erp_idx))
                cands = [j for j in non_erp if _pair_visible(cov, types, e, j)]
                if len(cands) < 2:
                    continue
                # pick 2 from cands that are also mutually covisible
                self._rng.shuffle(cands)
                for i in range(len(cands)):
                    for j in range(i + 1, len(cands)):
                        if _pair_visible(cov, types, cands[i], cands[j]):
                            idxs = np.array([e, cands[i], cands[j]])
                            self._rng.shuffle(idxs)
                            return idxs
            return None
        # K == 4: 1 erp + 1 fish + 2 pin (any focal)
        if not erp_idx or not fish_idx or len(pin_idx) < 2:
            return None
        for _ in range(max_retries):
            e = int(self._rng.choice(erp_idx))
            f_cands = [j for j in fish_idx if _pair_visible(cov, types, e, j)]
            if not f_cands:
                continue
            f = int(self._rng.choice(f_cands))
            p_cands = [i for i in pin_idx
                       if _pair_visible(cov, types, e, i)
                       and _pair_visible(cov, types, f, i)]
            if len(p_cands) < 2:
                continue
            self._rng.shuffle(p_cands)
            for i in range(len(p_cands)):
                for j in range(i + 1, len(p_cands)):
                    if _pair_visible(cov, types, p_cands[i], p_cands[j]):
                        idxs = np.array([e, f, p_cands[i], p_cands[j]])
                        self._rng.shuffle(idxs)
                        return idxs
        return None

    def _try_covis_only(self, K, cov, types, n_frames, max_retries):
        """L3: no type constraint, just symmetric covisibility on every pair.

        Returns idxs or None.
        """
        if n_frames < K:
            return None
        for _ in range(max_retries):
            cand = self._rng.choice(n_frames, size=K, replace=False)
            ok = True
            for i in range(K):
                for j in range(i + 1, K):
                    if not _pair_visible(cov, types, int(cand[i]), int(cand[j])):
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                return cand
        return None

    def _random_fallback(self, K, n_frames):
        """L4: pure random draw; guarantees that sampling never fails."""
        replace = n_frames < K
        return self._rng.choice(n_frames, size=K, replace=replace)

    _LADDER_TIERS = ("L1_strict", "L2_relaxed_type", "L3_covis_only", "L4_random")

    def _covis_type_diverse_sample(
        self,
        cov: np.ndarray,
        meta_frames: list,
        num_views_to_sample: int = 4,
        max_retries: int = 20,
        scene_name: str = "?",
    ) -> np.ndarray:
        """Type-diverse covisibility sampler; never raises.

        For K in {2, 3, 4} the tiers are tried in order:
          L1 strict       : strict type diversity (panorama + fisheye + pinholes of two focals).
          L2 relaxed_type : one diversity constraint dropped.
          L3 covis_only   : any K frames whose pairs all pass the covisibility thresholds.
          L4 random       : random draw without covisibility check.

        The first tier that succeeds is used; fallbacks from L2 on are logged.
        """
        K = int(num_views_to_sample)
        if K not in (2, 3, 4):
            # Out-of-band K: just random fallback.
            return self._random_fallback(K, cov.shape[0])
        if cov.shape[0] != len(meta_frames):
            return self._random_fallback(K, cov.shape[0])

        types = [f["base_type"] for f in meta_frames]
        bnames = [f["base_name"] for f in meta_frames]
        erp_idx = [i for i, t in enumerate(types) if t == "erp"]
        fish_idx = [i for i, t in enumerate(types) if t == "fisheye"]
        pin_idx = [i for i, t in enumerate(types) if t == "pinhole"]
        pin_by_focal: Dict[str, List[int]] = {}
        for i, (t, b) in enumerate(zip(types, bnames)):
            if t == "pinhole":
                pin_by_focal.setdefault(b, []).append(i)

        # L1
        sel = self._try_covis_strict(
            K, cov, types, bnames, erp_idx, fish_idx, pin_idx,
            pin_by_focal, max_retries,
        )
        if sel is not None:
            return sel
        # L2
        sel = self._try_covis_relaxed_type(
            K, cov, types, bnames, erp_idx, fish_idx, pin_idx,
            pin_by_focal, max_retries,
        )
        if sel is not None:
            print(
                f"[covis-sampler] {scene_name} K={K} fell back to L2_relaxed_type",
                flush=True,
            )
            return sel
        # L3
        sel = self._try_covis_only(K, cov, types, cov.shape[0], max_retries)
        if sel is not None:
            print(
                f"[covis-sampler] {scene_name} K={K} fell back to L3_covis_only",
                flush=True,
            )
            return sel
        # L4 -- guaranteed success
        print(
            f"[covis-sampler] {scene_name} K={K} fell back to L4_random",
            flush=True,
        )
        return self._random_fallback(K, cov.shape[0])

    # =========================================================
    # Native-render sampler with per-camera-pair thresholds
    # =========================================================
    # Any combination of the six pool cameras and any K. Cascade:
    #   L1 strict  : all pairs pass the thresholds at K = K_req.
    #   L2 k_decay : K decreases from K_req - 1 to 2 at the same thresholds; the tuple is
    #                padded back to K_req with random pool frames.
    #   L3 random  : random draw from the pool.
    # The thresholds are never relaxed; only the number of mutually covisible views shrinks.

    _N5_LADDER_TIERS = (
        "L1_n5_strict",
        "L2_n5_k_decay",
        "L3_n5_random",
    )

    def _covis_random_k_n5(
        self,
        K: int,
        cov: np.ndarray,
        bnames: list,
        all_idx: list,
        thres_scale: float = 1.0,
        max_retries: int = 300,
    ) -> Optional[np.ndarray]:
        """Draw K indices uniformly from ``all_idx`` such that every pair passes its threshold.

        Returns the K indices in random order, or None when ``max_retries`` draws fail.
        ``all_idx`` must already be restricted to ``_N5_ALLOWED_BASE_NAMES``.
        """
        n_pool = len(all_idx)
        if n_pool < K:
            return None
        all_idx_arr = np.asarray(all_idx, dtype=np.int64)
        for _ in range(max_retries):
            picks = self._rng.choice(all_idx_arr, size=K, replace=False)
            ok = True
            for p in range(K):
                for q in range(p + 1, K):
                    ci, cj = int(picks[p]), int(picks[q])
                    thres = _bicov_thres_n5(bnames[ci], bnames[cj]) * thres_scale
                    if _bicov(cov, ci, cj) < thres:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                self._rng.shuffle(picks)
                return picks
        return None

    def _covis_n5_sample(
        self,
        cov: np.ndarray,
        meta_frames: list,
        num_views_to_sample: int = 4,
        max_retries: int = 300,
        scene_name: str = "?",
    ) -> np.ndarray:
        """Native-render cascade sampler (L1 strict, L2 k_decay, L3 random); never raises.

        Args:
            cov: (N, N) covisibility matrix restricted to the usable frames of the scene.
            meta_frames: list of length N with a ``base_name`` field.
            num_views_to_sample: requested tuple size K_req.
            max_retries: draw budget of L1 (L2 uses at least 100 or half of it).
            scene_name: tag for the fallback log line.

        Returns: int64 array of K_req indices into ``meta_frames``.
        """
        K_req = int(num_views_to_sample)
        n_frames = cov.shape[0]

        # Restrict the pool to the allowed cameras
        bnames = [f["base_name"] for f in meta_frames]
        pool = [i for i, bn in enumerate(bnames) if bn in _N5_ALLOWED_BASE_NAMES]

        if len(pool) < 2 or n_frames != len(meta_frames):
            # Too few pool frames: random draw over the available frames
            print(
                f"[covis-sampler] {scene_name} K={K_req} pool<2 → L3_random",
                flush=True,
            )
            return self._random_fallback(K_req, max(n_frames, K_req))

        pool_arr = np.asarray(pool, dtype=np.int64)
        K_eff = min(K_req, len(pool))

        # L1: all pairs pass the thresholds at K = K_eff
        sel = self._covis_random_k_n5(
            K_eff, cov, bnames, pool, thres_scale=1.0, max_retries=max_retries,
        )
        if sel is not None:
            return self._pad_to_K(sel, K_req, pool_arr)

        # L2: K decreases (K_eff - 1, ..., 2) at the same thresholds
        for K_try in range(K_eff - 1, 1, -1):
            sel = self._covis_random_k_n5(
                K_try, cov, bnames, pool,
                thres_scale=1.0,
                max_retries=max(100, max_retries // 2),
            )
            if sel is not None:
                print(
                    f"[covis-sampler] {scene_name} K_req={K_req} fell back "
                    f"to L2_n5_k_decay (K_actual={K_try}, thres×1.0)",
                    flush=True,
                )
                return self._pad_to_K(sel, K_req, pool_arr)

        # L3: random draw
        print(
            f"[covis-sampler] {scene_name} K={K_req} fell back to L3_n5_random",
            flush=True,
        )
        return self._rng.choice(
            pool_arr, size=K_req, replace=(len(pool_arr) < K_req)
        )

    def _pad_to_K(
        self, sel: np.ndarray, K_req: int, pool_arr: np.ndarray
    ) -> np.ndarray:
        """Pad ``sel`` to length ``K_req`` with random picks from ``pool_arr`` and shuffle.

        Only the frames of ``sel`` are checked for covisibility; the padding keeps the tuple
        size of the batch.
        """
        if len(sel) >= K_req:
            return sel[:K_req]
        n_extra = K_req - len(sel)
        replace = len(pool_arr) < n_extra
        extras = self._rng.choice(pool_arr, size=n_extra, replace=replace)
        out = np.concatenate([np.asarray(sel, dtype=np.int64), extras])
        self._rng.shuffle(out)
        return out

    # ------------------------------------------ stratified native-render sampling

    # Type axes of the stratified walks
    _V2_ORDER_3TYPE = ("pin", "fish", "erp")
    _V2_ORDER_6NAME = ("erp", "fish_180", "fish_220",
                        "pin_14mm", "pin_24mm", "pin_40mm")
    _V2_BTYPE_SHORT = {"pinhole": "pin", "fisheye": "fish", "erp": "erp"}

    def _n5_v2_load_feasibility(self, path_3type: str, path_6name: str) -> None:
        """Load both feasibility tables and index them per scene and tuple size.

        ``self._feas_v2[axis][scene_id][K] = (combos (M, n_types), weights (M,))``, where the
        weights are the measured success rates; only combinations with a success rate above
        zero are kept.
        """
        import pandas as pd

        if not os.path.isfile(path_3type):
            raise FileNotFoundError(
                f"n5_sampling_v2: 3-type feasibility table missing: {path_3type}"
            )
        if not os.path.isfile(path_6name):
            raise FileNotFoundError(
                f"n5_sampling_v2: 6-camera feasibility table missing: {path_6name}"
            )

        self._feas_v2: Dict[str, Dict[str, Dict[int, tuple]]] = {
            "3type": {}, "6name": {},
        }
        for axis, path, order in (
            ("3type", path_3type, self._V2_ORDER_3TYPE),
            ("6name", path_6name, self._V2_ORDER_6NAME),
        ):
            df = pd.read_parquet(path)
            df = df[df["success_rate"] > 0]
            cols = list(order)
            # Group by (scene_id, K) → arrays of combos and weights.
            grp = df.groupby(["scene_id", "K"])
            scene_K: Dict[str, Dict[int, tuple]] = {}
            for (sid, K), sub in grp:
                combos = sub[cols].to_numpy(dtype=np.int8)
                weights = sub["success_rate"].to_numpy(dtype=np.float32)
                scene_K.setdefault(sid, {})[int(K)] = (combos, weights)
            self._feas_v2[axis] = scene_K
        # Cached stats
        n3 = sum(len(v) for v in self._feas_v2["3type"].values())
        n6 = sum(len(v) for v in self._feas_v2["6name"].values())
        print(
            f"Feasibility tables loaded: 3type scenes={len(self._feas_v2['3type'])} "
            f"(scene-K pairs={n3}) | 6name scenes={len(self._feas_v2['6name'])} "
            f"(scene-K pairs={n6})",
            flush=True,
        )

    def _n5_v2_walk_nway(
        self,
        cov: np.ndarray,
        type_labels: list,
        type_order: tuple,
        target_count: np.ndarray,
    ) -> Optional[np.ndarray]:
        """Connected random walk with per-type quotas.

        Returns the walk indices (length K = sum(target_count)) or None when no walk reaches
        the quotas within the retry budget. Same algorithm as scripts/constrained_walk_nway.py,
        drawing from self._rng.
        """
        K = int(target_count.sum())
        n_types = len(type_order)
        type_idx = {t: i for i, t in enumerate(type_order)}
        N = cov.shape[0]
        type_to_nodes: List[List[int]] = [[] for _ in range(n_types)]
        node_type_idx = np.full(N, -1, dtype=np.int64)
        for i, t in enumerate(type_labels):
            ti = type_idx.get(t, -1)
            if ti >= 0:
                type_to_nodes[ti].append(i)
                node_type_idx[i] = ti
        # Structural prune
        for ti in range(n_types):
            if len(type_to_nodes[ti]) < int(target_count[ti]):
                return None

        thres = self._n5_v2_walk_thres
        for _ in range(self._n5_v2_walk_retries):
            cur = [0] * n_types
            visited: set = set()
            walk: List[int] = []
            stack: List[int] = []
            cand_starts: List[int] = []
            for ti in range(n_types):
                if int(target_count[ti]) > 0:
                    cand_starts.extend(type_to_nodes[ti])
            if not cand_starts:
                return None
            start = int(self._rng.choice(cand_starts))
            walk.append(start); visited.add(start); stack.append(start)
            cur[int(node_type_idx[start])] += 1

            while len(walk) < K and stack:
                curr = stack[-1]
                pcov = (cov[curr, :] + cov[:, curr].T) / 2.0
                denom = pcov[curr] + 1e-8
                pcov = pcov / denom
                pcov[curr] = 0
                adj = np.flatnonzero(pcov > thres)
                cands = []
                for j in adj:
                    if int(j) in visited:
                        continue
                    ti = int(node_type_idx[int(j)])
                    if ti < 0:
                        continue
                    if cur[ti] < int(target_count[ti]):
                        cands.append(int(j))
                if cands:
                    nxt = int(self._rng.choice(cands))
                    walk.append(nxt); visited.add(nxt); stack.append(nxt)
                    cur[int(node_type_idx[nxt])] += 1
                else:
                    stack.pop()
                    cur[int(node_type_idx[curr])] -= 1

            if len(walk) == K and tuple(cur) == tuple(int(x) for x in target_count):
                arr = np.asarray(walk, dtype=np.int64)
                self._rng.shuffle(arr)
                return arr
        return None

    def _n5_v2_walk_relaxed(self, cov: np.ndarray, K: int) -> Optional[np.ndarray]:
        """Connected walk without type quotas (fallback of the stratified walks).

        The only constraint is connectivity: each added frame has covisibility above
        walk_thres with the current frame. Returns K connected indices or None.
        """
        N = cov.shape[0]
        if N < int(K):
            return None
        labels = ["any"] * N
        target = np.array([int(K)], dtype=np.int64)
        return self._n5_v2_walk_nway(cov, labels, ("any",), target)

    def _covis_n5_v2_sample(
        self,
        cov: np.ndarray,
        meta_frames: list,
        num_views_to_sample: int,
        scene_name: str,
    ) -> np.ndarray:
        """Stratified native-render sampler.

        A tier is drawn with self._n5_v2_weights: 0 = per-pair threshold cascade, 1 = walk with
        3-type quotas, 2 = walk with 6-camera quotas. A failed quota walk falls back to a walk
        without quotas and then to the cascade, which never raises.
        """
        K_req = int(num_views_to_sample)
        tier = int(self._rng.choice(3, p=np.asarray(self._n5_v2_weights)))

        # L1 / L2 stratified path
        if tier in (1, 2):
            axis = "3type" if tier == 1 else "6name"
            order = self._V2_ORDER_3TYPE if tier == 1 else self._V2_ORDER_6NAME
            entry = self._feas_v2[axis].get(scene_name, {}).get(K_req)
            if entry is not None:
                combos, weights = entry
                # Draw a quota combination weighted by its success rate
                probs = weights / weights.sum()
                ci = int(self._rng.choice(len(combos), p=probs))
                target = combos[ci]
                if tier == 1:
                    labels = [
                        self._V2_BTYPE_SHORT.get(f["base_type"], "?")
                        for f in meta_frames
                    ]
                else:
                    labels = [f["base_name"] for f in meta_frames]
                walk = self._n5_v2_walk_nway(cov, labels, order, target)
                if walk is not None and len(walk) == K_req:
                    return walk
            # Quota walk failed or no table entry: connected walk without quotas
            relaxed = self._n5_v2_walk_relaxed(cov, K_req)
            if relaxed is not None and len(relaxed) == K_req:
                return relaxed

        # Per-pair threshold cascade (tier 0, or last resort); never raises
        return self._covis_n5_sample(
            cov, meta_frames,
            num_views_to_sample=K_req,
            scene_name=scene_name,
        )

    def _resize_and_centre_crop(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        rays: np.ndarray,
        mask: np.ndarray,
        target_wh: tuple,
    ):
        """Lanczos resize so the smaller side fits, then centre-crop to target.

        ``rgb`` is uint8 (HxWx3). ``rays`` is float32 unit vectors
        (auto-renormalised after bilinear resize). ``mask`` is uint8.
        """
        rescaled = rescale_image_and_other_optional_info(
            image=rgb,
            output_resolution=target_wh,  # (W, H)
            depthmap=depth.astype(np.float32),
            camera_intrinsics=None,
            additional_quantities_to_be_resized_with_nearest=[mask.astype(np.uint8)],
            additional_quantities_to_be_resized_with_bilinear=[rays.astype(np.float32)],
        )
        img_pil, depth_r, _, near_list, bil_list = rescaled
        mask_r = near_list[0]
        rays_r = bil_list[0]

        # Centre-crop to exact target_wh
        W, H = img_pil.size
        tgt_W, tgt_H = target_wh
        left = (W - tgt_W) // 2
        top = (H - tgt_H) // 2
        bbox = (left, top, left + tgt_W, top + tgt_H)
        cropped = crop_image_and_other_optional_info(
            image=img_pil,
            crop_bbox=bbox,
            depthmap=depth_r,
            camera_intrinsics=None,
            additional_quantities=[mask_r],
            additional_quantities_bilinear=[rays_r],
        )
        img_pil2, depth_c, _, near_list2, bil_list2 = cropped
        return img_pil2, depth_c, bil_list2[0], near_list2[0]

    # ------------------------------------------ full-field-of-view resizing
    def _resize_preserve_info(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        rays: np.ndarray,
        mask: np.ndarray,
        target_wh: tuple,
    ):
        """Resize directly to target_wh without a crop (full field of view, per-axis scale).

        :meth:`_resize_and_centre_crop` fits the short side and crops, which discards about
        half of the longitudes of a panorama; this resizes the whole image instead. RGB, depth,
        mask and ray map use the same transform, and the rays are renormalised afterwards.

        Returns ``(img_pil, depth_r, rays_r, mask_r)`` like :meth:`_resize_and_centre_crop`.
        """
        from PIL import Image as _Image

        tgt_W, tgt_H = int(target_wh[0]), int(target_wh[1])

        # Fill invalid pixels with their nearest valid value before the bilinear resizes, so
        # interpolation does not blend content with the zero-filled region (a dark ring at the
        # image-circle boundary). Invalid pixels are masked again afterwards; depth and mask use
        # nearest-neighbour resizing. No-op for fully valid frames.
        _mb = np.asarray(mask) > 0
        if _mb.any() and not _mb.all():
            from scipy import ndimage as _ndi
            _idx = tuple(_ndi.distance_transform_edt(~_mb, return_distances=False, return_indices=True))
            rgb = rgb[_idx]; rays = rays[_idx]

        # RGB: PIL bilinear (uint8 HxWx3 -> PIL at target size)
        img_pil = _Image.fromarray(rgb.astype(np.uint8)).resize(
            (tgt_W, tgt_H), resample=_Image.BILINEAR
        )

        # depth: nearest (avoid interpolating across depth discontinuities)
        depth_t = torch.from_numpy(depth.astype(np.float32))[None, None]
        depth_r = (
            F.interpolate(depth_t, size=(tgt_H, tgt_W), mode="nearest")
            .squeeze()
            .numpy()
            .astype(np.float32)
        )

        # mask: nearest then binarise
        mask_t = torch.from_numpy(mask.astype(np.float32))[None, None]
        mask_r = (
            F.interpolate(mask_t, size=(tgt_H, tgt_W), mode="nearest")
            .squeeze()
            .numpy()
        )
        mask_r = (mask_r > 0.5).astype(np.uint8)

        # rays: bilinear (H,W,3) -> (3,H,W) -> resize -> renormalise to unit
        rays_t = torch.from_numpy(rays.astype(np.float32)).permute(2, 0, 1)[None]
        rays_t = F.interpolate(
            rays_t, size=(tgt_H, tgt_W), mode="bilinear", align_corners=False
        )
        rays_t = rays_t / rays_t.norm(dim=1, keepdim=True).clamp_min(1e-8)
        rays_r = rays_t.squeeze(0).permute(1, 2, 0).numpy().astype(np.float32)

        return img_pil, depth_r, rays_r, mask_r

    # ----------------------------------------------------------- get_views

    def _photo_extras(self, rgb_u8, mask, rec=None):
        """Optics layer, applied to RGB only (geometry and mask unchanged).

        With probability photo_mtf_p the view is resampled through a virtual source density
        v ~ logU(photo_mtf_vmin, 1) (bilinear down- and upsampling). Sharpness of the source
        renders depends on the camera route (pinholes are rendered at 18 to 40 pixels per degree,
        panoramas at about 6), so without this randomisation sharpness would reveal the field of
        view. Then lateral chromatic aberration, vignetting and, for half of the views, sensor
        noise are applied. Invalid pixels are filled before resampling and set to black at the end.
        """
        rng = self._rng; r = rgb_u8
        if rec is not None: rec["mtf_v"] = 0.0
        if rng.random() < self.photo_mtf_p:
            H, W = r.shape[:2]
            mb = np.asarray(mask) > 0
            if mb.any() and not mb.all():
                from scipy import ndimage as _ndi
                _idx = tuple(_ndi.distance_transform_edt(~mb, return_distances=False, return_indices=True))
                r = r[_idx]
            v = float(np.exp(rng.uniform(np.log(self.photo_mtf_vmin), 0.0)))
            if rec is not None: rec["mtf_v"] = round(v, 3)
            from PIL import Image as _Im
            dw, dh = max(8, int(round(W * v))), max(8, int(round(H * v)))
            r = np.asarray(_Im.fromarray(r).resize((dw, dh), _Im.BILINEAR).resize((W, H), _Im.BILINEAR))
        r = _gaug.chromatic(r, ca_r=1.0 + rng.uniform(0.0, 0.004), ca_b=1.0 - rng.uniform(0.0, 0.004))
        r = _gaug.vignette(r, strength=rng.uniform(0.0, 0.35), R=rng.uniform(0.8, 1.0))
        if rng.random() < 0.5:                                   # sensor noise
            r = np.clip(r.astype(np.float32) + rng.normal(0, rng.uniform(1.0, 6.0), r.shape), 0, 255).astype(np.uint8)
        r = np.where((mask > 0)[..., None], r, 0).astype(np.uint8)   # invalid pixels stay black
        return r

    def _camera_sampling_views(self, scene_name, num_views_to_sample, resolution):
        """Build a connected camera-sampled tuple with the online camera sampler.

        The sampler returns geometry in the OpenCV convention already, so no frame conversion is
        applied. Returns None when no connected tuple could be built; the caller then falls back
        to native renders.
        """
        if self._camera_sampler is None:
            from mapanything.datasets.camera_sampler import CameraSampler
            self._camera_sampler = CameraSampler(self.ROOT, device=self.camera_sampling_device)
        # The sampler renders the final views at the largest side of each view's target shape; the
        # covisibility check runs internally at a low resolution. The content aspect ratio is
        # independent of the tensor bucket, so the resize deformation stays random and the
        # aspect-ratio embedding carries the content aspect.
        if isinstance(resolution, list):
            _res = [int(max(r)) for r in resolution]
        else:
            _res = int(max(resolution))
        # Retries draw a new scene: many failures at large K come from the pose graph of the scene,
        # which cannot be chained to K poses. The sampling record names the scene actually used.
        out = None; _attempts_used = 0
        for _attempt in range(1 + max(0, int(self.camera_sampling_retries))):
            _attempts_used = _attempt + 1
            if _attempt > 0:                                   # retry on a newly drawn scene
                scene_name = self.scenes[int(self._rng.integers(self.num_of_scenes))]
            o = self._camera_sampler.build(scene_name, int(num_views_to_sample), self._rng, res=_res)
            if o is not None and len(o[0]) == int(num_views_to_sample) and o[1].get("connected", False):
                out = o; break
        if out is None:
            return None   # all attempts failed: fall back to native renders
        gpu_views, info = out
        # Sampling record: one compact JSON per tuple on view 0 (route, mode, camera models, fields
        # of view, aspect ratios, bucket, principal point, roll, tilt, distortion, source, optics,
        # retries, covisibility), appended by the trainer to sampling_logs/rank<r>.jsonl.
        _rec = dict(r="camera_sampling", sc=scene_name, K=int(num_views_to_sample), mode=info.get("mode"),
                    gz=(gpu_views[0]["spec"].get("gaze") or ""), rt=_attempts_used - 1, bt=int(info.get("b_tier", 0)),
                    nres=int(info.get("n_restore", 0)), cv=round(float(info.get("covis_min", -1.0)), 3), v=[])
        views = []
        for pos, v in enumerate(gpu_views):
            res_v = resolution[pos] if isinstance(resolution, list) else resolution
            H, W = v["H"], v["W"]; sp = v["spec"]
            rgb = (v["rgb"].clamp(0, 1) * 255.0).round().to(torch.uint8).cpu().numpy()
            mask = v["valid2d"].cpu().numpy().astype(np.uint8)
            depth = np.where(mask > 0, v["depth2d"].to(torch.float32).cpu().numpy(), 0.0).astype(np.float32)
            rays_cv = v["rays"].reshape(H, W, 3).to(torch.float32).cpu().numpy()   # already in the OpenCV convention
            c2w_cv = v["c2w"].to(torch.float32).cpu().numpy()
            img_pil, depth_r, rays_r, mask_r = self._resize_preserve_info(rgb, depth, rays_cv, mask, tuple(res_v))
            depth_r = np.where(mask_r > 0, depth_r, 0.0).astype(np.float32)
            if (mask_r == 0).any():                               # set invalid pixels of the resized RGB to black
                from PIL import Image as _Im
                _a = np.asarray(img_pil).copy(); _a[mask_r == 0] = 0; img_pil = _Im.fromarray(_a)
            _vrec = dict(m=("erp_full" if sp.get("erp_full") else sp["model"]), fov=round(float(sp["fov"]), 1),
                         ar=round(float(sp["ar"]), 3), bar=round(float(res_v[0]) / float(res_v[1]), 3),
                         ppx=round(float(sp["ppx"]), 3), ppy=round(float(sp["ppy"]), 3),
                         roll=round(float(sp["roll"]), 1), tilt=round(float(sp["tilt"]), 1),
                         cf=(None if sp.get("circle_frac") is None else round(float(sp["circle_frac"]), 2)),
                         src=v.get("src", ""), d={k: round(float(x), 4) for k, x in (sp.get("dist") or {}).items()})
            if self.camera_sampling_photo_extras:
                # Optics layer after the resize, at the same stage as for native renders
                from PIL import Image as _Im
                img_pil = _Im.fromarray(self._photo_extras(np.asarray(img_pil), mask_r, rec=_vrec))
            _rec["v"].append(_vrec)
            views.append(dict(
                img=img_pil, depthmap=depth_r, camera_pose=c2w_cv.astype(np.float32),
                ray_directions_cam_provided=rays_r.astype(np.float32),
                non_ambiguous_mask=(mask_r > 0).astype(np.uint8), aspect_ratio=np.float32(sp["ar"]),
                resize_mode="squeeze", dataset="ProcThorUnicol", label=scene_name,
                base_type=sp["model"], pose_index=int(sp["pose"]),
                instance=f"sampled-{info['mode']}-{sp['model']}-{pos}",
            ))
        views[0]["sampling_log"] = json.dumps(_rec, separators=(",", ":"))
        return views

    def _get_views(self, sampled_idx, num_views_to_sample, resolution):
        scene_name = self.scenes[sampled_idx]
        scene_dir = os.path.join(self.ROOT, scene_name)
        if self.camera_sampling and self._rng.random() < self.camera_sampling_p:
            av = self._camera_sampling_views(scene_name, num_views_to_sample, resolution)
            if av is not None:
                return av
        descriptors = self._build_frame_index(scene_dir)
        if len(descriptors) == 0:
            raise RuntimeError(
                f"No usable frames in {scene_dir} for camera_types={self.camera_types}"
            )

        # --- Native-render tuple selected on the offline covisibility ---
        cov, full_meta = self._load_covis(scene_dir)
        if cov is not None and full_meta is not None:
            # Restrict the covisibility matrix (computed over all frames) to the frames usable for
            # training after the camera-type, focal-length and flat-wall filters.
            tag_to_full_idx = {f["tag"]: f["index"] for f in full_meta}
            desc_full_idx = []
            keep_desc = []
            for di, d in enumerate(descriptors):
                if d["tag"] in tag_to_full_idx:
                    desc_full_idx.append(tag_to_full_idx[d["tag"]])
                    keep_desc.append(di)
            if len(desc_full_idx) == len(descriptors):
                cov_sub = cov[np.ix_(desc_full_idx, desc_full_idx)]
                meta_sub = [full_meta[i] for i in desc_full_idx]
                if self.n5_sampling_v2:
                    sel = self._covis_n5_v2_sample(
                        cov_sub, meta_sub,
                        num_views_to_sample=num_views_to_sample,
                        scene_name=scene_name,
                    )
                elif self.n5_sampling:
                    sel = self._covis_n5_sample(
                        cov_sub, meta_sub,
                        num_views_to_sample=num_views_to_sample,
                        scene_name=scene_name,
                    )
                else:
                    sel = self._covis_type_diverse_sample(
                        cov_sub, meta_sub,
                        num_views_to_sample=num_views_to_sample,
                        scene_name=scene_name,
                    )
            else:
                # The covisibility frames do not match the usable frames: random selection
                print(
                    f"[covis-sampler] {scene_name}: covisibility frames do not match the pack "
                    "files; drawing the tuple without covisibility",
                    flush=True,
                )
                replace = len(descriptors) < num_views_to_sample
                sel = self._rng.choice(
                    len(descriptors), size=num_views_to_sample, replace=replace
                )
        else:
            # No offline covisibility for this scene: random selection
            print(
                f"[covis-sampler] {scene_name}: no covisibility/v0 files; drawing the tuple "
                "without covisibility",
                flush=True,
            )
            replace = len(descriptors) < num_views_to_sample
            sel = self._rng.choice(
                len(descriptors), size=num_views_to_sample, replace=replace
            )

        views = []
        _clean_recs = []  # per-view sampling records of the native-render tuple
        for view_pos, view_idx in enumerate(sel):
            # With variable resolution, ``resolution`` is a per-view list of target shapes
            res_v = resolution[view_pos] if isinstance(resolution, list) else resolution
            d = descriptors[int(view_idx)]
            with np.load(d["pack_path"]) as pack:
                rgb_f16 = np.asarray(pack["rgb"])  # (H, W, 3) fp16
                rays_f16 = np.asarray(pack["rays"])  # (H, W, 3) fp16
                depth_f16 = np.asarray(pack["depth"])  # (H, W)
                mask_u8 = np.asarray(pack["mask"])  # (H, W)

            rgb = (np.clip(rgb_f16.astype(np.float32), 0.0, 1.0) * 255.0).astype(np.uint8)
            depth = depth_f16.astype(np.float32)
            mask = mask_u8.astype(np.uint8)
            rays = rays_f16.astype(np.float32)
            # Re-normalise rays (fp16 round-off)
            rays = rays / np.clip(np.linalg.norm(rays, axis=-1, keepdims=True), 1e-8, None)

            # Aspect ratio W/H of the original (pre-resize) image
            _H_orig, _W_orig = rgb.shape[:2]
            aspect_ratio = float(_W_orig) / float(max(_H_orig, 1))

            # Renderer camera frame -> OpenCV camera frame
            rays_cv, c2w_cv = _unicol_to_opencv(rays, d["c2w_unicol"])

            # Resize to the target resolution (W, H)
            canon_bt = (
                "fisheye" if "fish" in d["base_type"]
                else "erp" if "erp" in d["base_type"]
                else "pinhole"
            )
            if self.preserve_info_resize and canon_bt in self.preserve_info_types:
                # Full-field-of-view resize (no crop)
                img_pil, depth_r, rays_r, mask_r = self._resize_preserve_info(
                    rgb=rgb,
                    depth=depth,
                    rays=rays_cv,
                    mask=mask,
                    target_wh=tuple(res_v),
                )
                resize_mode = "squeeze"
                ar_fed = float(aspect_ratio)
            else:
                img_pil, depth_r, rays_r, mask_r = self._resize_and_centre_crop(
                    rgb=rgb,
                    depth=depth,
                    rays=rays_cv,
                    mask=mask,
                    target_wh=tuple(res_v),
                )
                resize_mode = "centre_crop"
                ar_fed = float(aspect_ratio)

            # Zero invalid depth so that MapAnything's valid-mask logic catches it
            depth_r = np.where(mask_r > 0, depth_r, 0.0).astype(np.float32)

            # When the camera sampler is on, native renders get the same optics layer, so image
            # quality does not reveal the route.
            _cvrec = dict(bn=d.get("base_name", d.get("tag", "?")), ar=round(float(ar_fed), 3),
                          bar=round(float(res_v[0]) / float(res_v[1]), 3))
            if self.camera_sampling and self.camera_sampling_photo_extras:
                from PIL import Image as _Im
                img_pil = _Im.fromarray(self._photo_extras(np.asarray(img_pil), mask_r, rec=_cvrec))
            _clean_recs.append(_cvrec)

            views.append(
                dict(
                    img=img_pil,
                    depthmap=depth_r,
                    camera_pose=c2w_cv.astype(np.float32),
                    ray_directions_cam_provided=rays_r.astype(np.float32),
                    non_ambiguous_mask=(mask_r > 0).astype(np.uint8),
                    aspect_ratio=np.float32(ar_fed),
                    resize_mode=resize_mode,
                    dataset="ProcThorUnicol",
                    label=scene_name,
                    base_type=canon_bt,
                    pose_index=int(d["pose_index"]),
                    instance=d["tag"],
                )
            )

        if views and _clean_recs:  # sampling record of the native-render tuple
            views[0]["sampling_log"] = json.dumps(dict(r="clean", sc=scene_name, K=len(views), v=_clean_recs),
                                                  separators=(",", ":"))
        return views
