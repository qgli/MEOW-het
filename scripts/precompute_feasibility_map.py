"""Precompute per-(scene, K, type combination) walk feasibility tables.

For every scene of ``--splits``, every K in ``--k-list`` and every combination of
per-type counts that sums to K on the ``--type-axis`` (frames restricted to the six
cameras of ``ALLOWED_BNAMES``; ``3type``: pinhole / fisheye / panorama counts;
``6name``: one count per camera), run ``--trials`` walks of
``constrained_random_walk_nway`` and record the success rate (fraction of walks that
reach exactly that composition).

Output: a parquet file with columns
    scene_id, split, K, <one count column per type>, success_rate, n_trials
(type columns ``pin, fish, erp`` or ``erp, fish_180, fish_220, pin_14mm, pin_24mm,
pin_40mm``), plus ``<out without extension>.meta.json`` describing the run.

Loaded by :class:`mapanything.datasets.procthor_unicol.ProcThorUnicol` (stratified
native-render sampling, ``n5_sampling_v2``) as a pandas DataFrame and indexed per
(scene, K): the combinations with a positive success rate, weighted by that rate, are
the per-type quotas of its connected walks.

Scenes are processed in parallel by a pool of ``--workers`` processes (default 16). The random
stream of a scene is seeded with ``--seed`` plus Python's string hash of the scene id, so a rerun
reproduces the tables exactly only under the same ``PYTHONHASHSEED`` (otherwise statistically).

Usage (parameters of the tables shipped in ``resources/gen1_feasibility/``; the
six-camera table uses ``--type-axis 6name``):
  python scripts/precompute_feasibility_map.py \
      --data-root /path/to/renders/scenes --splits-dir resources/gen1_splits \
      --splits train,val --k-list 2,3,4,5,6,7,8 --trials 30 --ma-thres 0.25 \
      --seed 42 --type-axis 3type --out /path/to/feasibility_3type_full.parquet
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from typing import List, Tuple

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
for p in (REPO_ROOT, SCRIPTS_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from constrained_walk_nway import constrained_random_walk_nway  # noqa: E402
from feasibility_scan import (  # noqa: E402
    load_scene,
    all_type_combos,
    ALLOWED_BNAMES,
)
import itertools


TYPE_SHORT = {"pinhole": "pin", "fisheye": "fish", "erp": "erp"}

TYPE_AXES = {
    "3type": {
        "order": ["pin", "fish", "erp"],
        "label_fn": lambda fr: TYPE_SHORT[fr["base_type"]],
    },
    "6name": {
        "order": ["erp", "fish_180", "fish_220",
                  "pin_14mm", "pin_24mm", "pin_40mm"],
        "label_fn": lambda fr: fr["base_name"],
    },
}


def all_axis_combos(K: int, n_types: int) -> List[Tuple[int, ...]]:
    return [c for c in itertools.product(*[range(K + 1)] * n_types)
            if sum(c) == K]


def _process_scene(args_tuple) -> List[Tuple]:
    """Worker: process one scene, return list of rows
    (scene_id, split, K, *combo, success_rate, n_trials).
    """
    (scene_id, split, scene_dir, k_list, trials, ma_thres, seed,
     type_axis) = args_tuple
    axis = TYPE_AXES[type_axis]
    type_order = axis["order"]
    label_fn = axis["label_fn"]
    n_types = len(type_order)

    loaded = load_scene(scene_dir)
    if loaded is None:
        return []
    cov_full, frames = loaded
    bnames_full = [fr["base_name"] for fr in frames]
    keep = [i for i, bn in enumerate(bnames_full) if bn in ALLOWED_BNAMES]
    if not keep:
        return []
    cov = cov_full[np.ix_(keep, keep)]
    type_labels = [label_fn(frames[i]) for i in keep]

    rng = np.random.default_rng(seed + abs(hash(scene_id)) % (2**31))

    rows = []
    for K in k_list:
        for combo in all_axis_combos(K, n_types):
            succ = 0
            for _ in range(trials):
                _, ok = constrained_random_walk_nway(
                    cov, type_labels, type_order, combo, rng, ma_thres
                )
                if ok:
                    succ += 1
            rate = succ / trials
            rows.append((scene_id, split, K, *combo, rate, trials))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True,
                    help="Directory containing the scene folders "
                         "(each with covisibility/v0/ from compute_covisibility.py)")
    ap.add_argument("--splits-dir", required=True,
                    help="Directory containing the <split>.json scene lists")
    ap.add_argument("--splits", default="train,val")
    ap.add_argument("--k-list", default="2,3,4,5,6,7,8")
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--ma-thres", type=float, default=0.25)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True,
                    help="parquet path (e.g. resources/gen1_feasibility/feasibility_3type_full.parquet)")
    ap.add_argument("--limit-per-split", type=int, default=0,
                    help="0 = all scenes; >0 = process first N (for testing)")
    ap.add_argument("--type-axis", default="3type", choices=list(TYPE_AXES.keys()),
                    help="3type: pin/fish/erp ; 6name: per base_name")
    args = ap.parse_args()

    k_list = [int(x) for x in args.k_list.split(",")]
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    type_order = TYPE_AXES[args.type_axis]["order"]
    n_types = len(type_order)
    type_col_names = list(type_order)

    tasks = []
    split_counts = {}
    for split in splits:
        sp_file = os.path.join(args.splits_dir, f"{split}.json")
        with open(sp_file) as f:
            scenes = json.load(f)
        if args.limit_per_split > 0:
            scenes = scenes[: args.limit_per_split]
        split_counts[split] = len(scenes)
        for sname in scenes:
            sd = os.path.join(args.data_root, sname)
            tasks.append(
                (sname, split, sd, k_list, args.trials, args.ma_thres,
                 args.seed, args.type_axis)
            )

    print(f"[precompute] splits={split_counts} | K={k_list} | "
          f"trials={args.trials} | workers={args.workers} | "
          f"axis={args.type_axis} ({n_types}-way: {type_order}) | "
          f"total_scenes={len(tasks)}", flush=True)

    n_combos_total = sum(len(all_axis_combos(K, n_types)) for K in k_list)
    print(f"[precompute] combos per scene: {n_combos_total} | "
          f"total constrained_walks: {len(tasks) * n_combos_total * args.trials:,}",
          flush=True)

    all_rows: List[Tuple] = []
    t0 = time.time()
    done = 0
    with mp.Pool(args.workers) as pool:
        for rows in pool.imap_unordered(_process_scene, tasks, chunksize=4):
            all_rows.extend(rows)
            done += 1
            if done % 50 == 0 or done == len(tasks):
                el = time.time() - t0
                eta = el / done * (len(tasks) - done)
                print(f"  [{done}/{len(tasks)}] elapsed={el:.1f}s "
                      f"eta={eta:.1f}s rows={len(all_rows):,}", flush=True)

    print(f"[precompute] all scenes done in {time.time()-t0:.1f}s, "
          f"writing parquet ({len(all_rows):,} rows)…", flush=True)

    import pandas as pd
    df = pd.DataFrame(
        all_rows,
        columns=["scene_id", "split", "K", *type_col_names,
                 "success_rate", "n_trials"],
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    df.to_parquet(args.out, index=False)

    # Write companion metadata json
    meta = {
        "data_root": args.data_root,
        "splits_dir": args.splits_dir,
        "splits": splits,
        "split_counts": split_counts,
        "k_list": k_list,
        "trials": args.trials,
        "ma_thres": args.ma_thres,
        "seed": args.seed,
        "type_axis": args.type_axis,
        "type_order": type_order,
        "allowed_bnames": sorted(ALLOWED_BNAMES),
        "n_rows": len(df),
        "wall_seconds": time.time() - t0,
    }
    meta_path = os.path.splitext(args.out)[0] + ".meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[precompute] wrote {args.out} ({len(df):,} rows)")
    print(f"[precompute] wrote {meta_path}")

    # Summary per K: number of scenes with a positive success rate per combination
    # (min, max, max/min ratio) and the combinations feasible in no scene
    print("\n=== Quick summary ===")
    for K in k_list:
        sub = df[df["K"] == K]
        n_combo = sub.groupby(type_col_names).size().shape[0]
        feas_by_combo = sub.groupby(type_col_names)["success_rate"].apply(
            lambda s: (s > 0).sum()
        )
        n_scenes = sub["scene_id"].nunique()
        print(
            f"K={K}: combos={n_combo} | feasible_scenes "
            f"min={feas_by_combo.min()} max={feas_by_combo.max()} "
            f"(ratio {feas_by_combo.max()/max(1,feas_by_combo.min()):.2f}x) | "
            f"zero-scene combos: {(feas_by_combo == 0).sum()}/{n_combo} | "
            f"total_scenes={n_scenes}"
        )


if __name__ == "__main__":
    main()
