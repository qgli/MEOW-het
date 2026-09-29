#!/usr/bin/env python3
"""Compare regenerated first-generation packs with the original training packs (rays, depth, mask, RGB PSNR)."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def compare_pack(reference: Path, regenerated: Path) -> tuple[dict, np.ndarray]:
    with np.load(reference) as ref, np.load(regenerated) as new:
        missing = sorted(set(ref.files) - set(new.files))
        extra = sorted(set(new.files) - set(ref.files))
        if missing or extra:
            raise ValueError(f"key mismatch: missing={missing}, extra={extra}")

        rays_ref = np.asarray(ref["rays"], dtype=np.float32)
        rays_new = np.asarray(new["rays"], dtype=np.float32)
        if rays_ref.shape != rays_new.shape:
            raise ValueError(f"ray shape mismatch: {rays_ref.shape} != {rays_new.shape}")
        if np.array_equal(rays_ref, rays_new):
            max_ray_deg = 0.0
        else:
            rays_ref /= np.maximum(np.linalg.norm(rays_ref, axis=-1, keepdims=True), 1e-12)
            rays_new /= np.maximum(np.linalg.norm(rays_new, axis=-1, keepdims=True), 1e-12)
            dots = np.clip(np.sum(rays_ref * rays_new, axis=-1), -1.0, 1.0)
            max_ray_deg = float(np.degrees(np.arccos(dots)).max())

        mask_ref = np.asarray(ref["mask"], dtype=bool)
        mask_new = np.asarray(new["mask"], dtype=bool)
        intersection = int(np.count_nonzero(mask_ref & mask_new))
        union = int(np.count_nonzero(mask_ref | mask_new))
        mask_iou = float(intersection / union) if union else 1.0

        depth_ref = np.asarray(ref["depth"], dtype=np.float32)
        depth_new = np.asarray(new["depth"], dtype=np.float32)
        valid = (
            mask_ref
            & mask_new
            & np.isfinite(depth_ref)
            & np.isfinite(depth_new)
            & (depth_ref > 0)
        )
        rel = np.abs(depth_new[valid] - depth_ref[valid]) / np.maximum(
            np.abs(depth_ref[valid]), 1e-6
        )

        rgb_ref = np.asarray(ref["rgb"], dtype=np.float32)
        rgb_new = np.asarray(new["rgb"], dtype=np.float32)
        sqerr = np.square(rgb_new - rgb_ref, dtype=np.float32)
        rgb_sse = float(sqerr.sum(dtype=np.float64))
        rgb_n = int(sqerr.size)
        mse = rgb_sse / rgb_n
        psnr = float("inf") if mse == 0 else float(10.0 * math.log10(1.0 / mse))

    return {
        "pack": reference.name,
        "ray_max_angle_deg": max_ray_deg,
        "depth_rel_median": float(np.median(rel)) if rel.size else None,
        "depth_rel_p99": float(np.quantile(rel, 0.99)) if rel.size else None,
        "depth_valid_pixels": int(rel.size),
        "rgb_psnr_db": psnr,
        "rgb_sse": rgb_sse,
        "rgb_values": rgb_n,
        "mask_iou": mask_iou,
        "mask_intersection": intersection,
        "mask_union": union,
    }, rel


def compare_scene(reference: Path, regenerated: Path) -> dict:
    ref_files = {p.name: p for p in reference.glob("*_pack.npz")}
    new_files = {p.name: p for p in regenerated.glob("*_pack.npz")}
    if set(ref_files) != set(new_files):
        raise ValueError(
            f"pack set mismatch: only_reference={sorted(set(ref_files)-set(new_files))}, "
            f"only_regenerated={sorted(set(new_files)-set(ref_files))}"
        )

    packs = []
    rel_parts = []
    for name in sorted(ref_files):
        metrics, rel = compare_pack(ref_files[name], new_files[name])
        packs.append(metrics)
        rel_parts.append(rel)

    rel_all = np.concatenate(rel_parts) if rel_parts else np.empty(0, np.float32)
    rgb_sse = sum(p["rgb_sse"] for p in packs)
    rgb_n = sum(p["rgb_values"] for p in packs)
    rgb_mse = rgb_sse / rgb_n if rgb_n else 0.0
    intersection = sum(p["mask_intersection"] for p in packs)
    union = sum(p["mask_union"] for p in packs)
    return {
        "scene": reference.name,
        "pack_count": len(packs),
        "summary": {
            "ray_max_angle_deg": max(p["ray_max_angle_deg"] for p in packs),
            "depth_rel_median": float(np.median(rel_all)) if rel_all.size else None,
            "depth_rel_p99": float(np.quantile(rel_all, 0.99)) if rel_all.size else None,
            "depth_valid_pixels": int(rel_all.size),
            "rgb_psnr_db": (
                float("inf") if rgb_mse == 0 else float(10.0 * math.log10(1.0 / rgb_mse))
            ),
            "mask_iou": float(intersection / union) if union else 1.0,
        },
        "packs": packs,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--regenerated", type=Path, required=True)
    parser.add_argument("--scenes", nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    results = []
    for scene in args.scenes:
        result = compare_scene(args.reference / scene, args.regenerated / scene)
        results.append(result)
        print(scene, json.dumps(result["summary"], sort_keys=True))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"scenes": results}, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
