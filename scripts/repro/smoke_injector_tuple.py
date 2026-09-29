#!/usr/bin/env python3
"""Load one shard scene through the production tuple sampler and injector.

Prints the first training tuple: per view the instance label, image and depth shapes, the aspect-ratio
input, the camera model, whether the view was drawn by the online camera sampler (``camera_sampling_applied``;
a scene can fall back to its native renders) and whether it is a full panorama (middle-row azimuth
span of its rays above 350 degrees, the rule the model uses for the panorama wrap during training).
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--splits", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    sys.path.insert(0, args.repo)
    from mapanything.datasets.procthor_unicol import ProcThorUnicol

    ds = ProcThorUnicol(
        split="train",
        resolution=[(518, 252)],
        transform="imgnorm",
        data_norm_type="dinov2",
        ROOT=args.root,
        splits_dir=args.splits,
        seed=20260916,
        max_num_retries=2,
        variable_num_views=False,
        n5_sampling=True,
        n5_sampling_v2=True,
        preserve_info_resize=True,
        variable_resolution=False,
        camera_sampling=True,
        camera_sampling_p=1.0,
        camera_sampling_device="cuda:0",
        camera_sampling_photo_extras=True,
        camera_sampling_retries=1,
        photo_mtf_p=0.5,
        photo_mtf_vmin=0.35,
        num_views=4,
    )
    views = ds[0]
    result = {
        "dataset_len": len(ds),
        "num_views": len(views),
        "views": [],
    }
    for view in views:
        rays = view.get("ray_directions_cam")
        full_panorama = None
        if rays is not None:
            mid = np.asarray(rays)[len(rays) // 2]
            azimuth = np.arctan2(mid[:, 0], mid[:, 2])
            full_panorama = bool(azimuth.max() - azimuth.min() > np.deg2rad(350.0))
        instance = view.get("instance")
        applied = view.get("camera_sampling_applied")
        if applied is None:
            applied = isinstance(instance, str) and instance.startswith("sampled-")
        result["views"].append(
            {
                "instance": instance,
                "img_shape": list(view["img"].shape),
                "depth_shape": list(view["depthmap"].shape),
                "aspect_ratio": float(view["aspect_ratio"]),
                "camera_model": view.get("base_type"),
                "full_panorama": full_panorama,
                "camera_sampling_applied": bool(applied),
            }
        )
    with open(args.out, "w") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
