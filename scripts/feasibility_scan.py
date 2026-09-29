"""Feasibility scan of pinhole / fisheye / panorama count combinations per tuple size K.

For each K in ``--k-list`` and each combination of counts (e.g. (4,0,0) = 4 pinholes,
(3,1,0) = 3 pinholes + 1 fisheye, ..., (0,0,4) = 4 panoramas):
  - count how many of the ``--num-scenes`` scenes drawn from ``--split`` yield at
    least one successful constrained random walk for this combination
  - report per-combination statistics, the max/min ratio of feasible-scene
    counts and the combinations that no scene can realise

For combination (p, f, e), each trial is a random walk on the covisibility matrix
restricted to the cameras of ``ALLOWED_BNAMES``: it starts at a node of a type the
target needs and only steps to unvisited neighbours whose type still has remaining
quota; it succeeds if the walk reaches exactly p pinholes, f fisheyes and e panoramas.

The success rate is a sampled estimate, sufficient for ranking combinations by
feasibility.

Usage:
  python scripts/feasibility_scan.py \
      --data-root /path/to/renders/scenes --splits-dir resources/gen1_splits \
      --num-scenes 200 --trials-per-combo 30 \
      --k-list 2,3,4,6,8 \
      --out out/feasibility_scan.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import random
import sys
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
for p in (REPO_ROOT, SCRIPTS_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

ALLOWED_BNAMES = {"erp", "fish_180", "fish_220",
                  "pin_14mm", "pin_24mm", "pin_40mm"}


def constrained_random_walk(
    cov: np.ndarray,
    btypes: List[str],
    target_count: Tuple[int, int, int],  # (n_pin, n_fish, n_erp)
    rng: np.random.Generator,
    covisibility_thres: float = 0.25,
    max_retries: int = 8,
    use_bidirectional: bool = True,
) -> Tuple[np.ndarray, bool]:
    """Random walk that only adds a neighbor if doing so keeps the running
    type counts consistent with `target_count` (no overshooting any type).
    Start node is sampled from a type that the target still needs.
    """
    K = sum(target_count)
    type_idx_map = {"pin": 0, "fish": 1, "erp": 2}
    N = cov.shape[0]

    type_to_nodes = {0: [], 1: [], 2: []}
    for i, t in enumerate(btypes):
        type_to_nodes[type_idx_map[t]].append(i)
    # Quick feasibility prune: need enough nodes of each type in scene
    for ti in range(3):
        if len(type_to_nodes[ti]) < target_count[ti]:
            return np.array([], dtype=int), False

    best: List[int] = []
    for _ in range(max_retries):
        cur = [0, 0, 0]
        visited: set = set()
        walk: List[int] = []
        stack: List[int] = []

        # Pick start node from a type that target still needs
        cand_types = [ti for ti in range(3) if target_count[ti] > 0]
        cand_starts: List[int] = []
        for ti in cand_types:
            cand_starts.extend(type_to_nodes[ti])
        if not cand_starts:
            break
        start = int(rng.choice(cand_starts))
        walk.append(start); visited.add(start); stack.append(start)
        cur[type_idx_map[btypes[start]]] += 1

        while len(walk) < K and stack:
            curr = stack[-1]
            if use_bidirectional:
                pcov = (cov[curr, :] + cov[:, curr].T) / 2.0
            else:
                pcov = cov[curr, :].copy()
            pcov = pcov / (pcov[curr] + 1e-8)
            pcov[curr] = 0
            adj = np.flatnonzero(pcov > covisibility_thres)
            # Only accept neighbors whose type still has remaining quota
            cands = []
            for j in adj:
                if j in visited:
                    continue
                ti = type_idx_map[btypes[j]]
                if cur[ti] < target_count[ti]:
                    cands.append(int(j))
            if cands:
                nxt = int(rng.choice(cands))
                walk.append(nxt); visited.add(nxt); stack.append(nxt)
                cur[type_idx_map[btypes[nxt]]] += 1
            else:
                stack.pop()
                # Decrement cur on backtrack to allow other branches
                cur[type_idx_map[btypes[curr]]] -= 1

        if len(walk) > len(best):
            best = walk
        if len(walk) == K and tuple(cur) == tuple(target_count):
            return np.array(walk), True

    return np.array(best), False


def load_scene(scene_dir: str):
    cov_p = os.path.join(scene_dir, "covisibility", "v0", "covisibility.npy")
    meta_p = os.path.join(scene_dir, "covisibility", "v0", "frame_meta.json")
    if not (os.path.isfile(cov_p) and os.path.isfile(meta_p)):
        return None
    cov = np.load(cov_p)
    with open(meta_p) as f:
        frames = json.load(f)["frames"]
    return cov, frames


def all_type_combos(K: int) -> List[Tuple[int, int, int]]:
    """All (p, f, e) with p+f+e=K, p,f,e >= 0."""
    out = []
    for p in range(K + 1):
        for f in range(K + 1 - p):
            e = K - p - f
            out.append((p, f, e))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True,
                    help="Directory containing the scene folders "
                         "(each with covisibility/v0/ from compute_covisibility.py)")
    ap.add_argument("--splits-dir", required=True,
                    help="Directory containing the <split>.json scene lists")
    ap.add_argument("--split", default="train")
    ap.add_argument("--num-scenes", type=int, default=200)
    ap.add_argument("--trials-per-combo", type=int, default=30,
                    help="random_walk trials per (scene, K, combo) to estimate "
                         "success rate")
    ap.add_argument("--k-list", type=str, default="2,3,4,6,8")
    ap.add_argument("--ma-thres", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True, help="Output JSON path")
    args = ap.parse_args()

    K_list = [int(x) for x in args.k_list.split(",")]
    type_short = {"pinhole": "pin", "fisheye": "fish", "erp": "erp"}

    with open(os.path.join(args.splits_dir, f"{args.split}.json")) as f:
        all_scenes = json.load(f)
    rng_sel = random.Random(args.seed)
    scenes = rng_sel.sample(all_scenes, min(args.num_scenes, len(all_scenes)))
    rng = np.random.default_rng(args.seed)

    # feas[K][combo] -> {n_succ_scenes (scenes with >= 1 successful trial),
    #   n_attempted (scenes tried), scene_rates (per-scene success rates)}
    feas = {K: {c: {"n_succ_scenes": 0, "n_attempted": 0,
                    "scene_rates": []}
                for c in all_type_combos(K)}
            for K in K_list}

    for si, sname in enumerate(scenes):
        sd = os.path.join(args.data_root, sname)
        loaded = load_scene(sd)
        if loaded is None:
            continue
        cov_full, frames = loaded
        bnames_full = [fr["base_name"] for fr in frames]
        btypes_full = [type_short[fr["base_type"]] for fr in frames]
        keep = [i for i, bn in enumerate(bnames_full) if bn in ALLOWED_BNAMES]
        if not keep:
            continue
        cov = cov_full[np.ix_(keep, keep)]
        btypes = [btypes_full[i] for i in keep]

        for K in K_list:
            for combo in all_type_combos(K):
                feas[K][combo]["n_attempted"] += 1
                succ = 0
                for _ in range(args.trials_per_combo):
                    _, ok = constrained_random_walk(
                        cov, btypes, combo, rng, args.ma_thres
                    )
                    if ok:
                        succ += 1
                rate = succ / args.trials_per_combo
                feas[K][combo]["scene_rates"].append(rate)
                if succ > 0:
                    feas[K][combo]["n_succ_scenes"] += 1
        if (si + 1) % 20 == 0:
            print(f"  [feasibility] processed {si+1}/{len(scenes)} scenes", flush=True)

    # ---------- print summary ----------
    print("\n=== Per-K combo feasibility summary ===")
    print("(n_succ_scenes = # scenes where >=1 of trials succeeded)")
    print("(mean_rate = mean of per-scene success rates, including 0 rates)\n")
    summary = {}
    for K in K_list:
        print(f"--- K = {K} ---")
        rows = []
        for combo, d in feas[K].items():
            n_att = d["n_attempted"]
            if n_att == 0:
                continue
            mean_rate = float(np.mean(d["scene_rates"]))
            n_succ = d["n_succ_scenes"]
            rows.append((combo, n_succ, mean_rate, n_att))
        rows.sort(key=lambda x: -x[1])
        for combo, n_succ, mean_rate, n_att in rows:
            print(f"  combo (P{combo[0]} F{combo[1]} E{combo[2]}): "
                  f"feasible_scenes={n_succ:>4d}/{n_att} "
                  f"({100*n_succ/n_att:>5.1f}%) | "
                  f"mean_success_rate={mean_rate:.3f}")
        # ratio max_feasible / min_feasible (excluding zero)
        feas_vals = [n_succ for _, n_succ, _, _ in rows if n_succ > 0]
        zero_combos = [combo for combo, n_succ, _, _ in rows if n_succ == 0]
        if feas_vals:
            ratio = max(feas_vals) / max(min(feas_vals), 1)
            print(f"  >>> K={K}: max/min feasible_scenes ratio = {ratio:.1f}x "
                  f"(zero-scene combos: {len(zero_combos)}/{len(rows)})")
            if zero_combos:
                print(f"      combos feasible in no scene: {zero_combos}")
        print()
        summary[K] = {
            "combos": [
                {"combo": list(combo),
                 "n_succ_scenes": n_succ,
                 "mean_rate": mean_rate,
                 "n_attempted": n_att}
                for combo, n_succ, mean_rate, n_att in rows
            ],
            "n_zero_scene_combos": len(zero_combos),
            "ratio_max_min_feasible": (
                max(feas_vals) / max(min(feas_vals), 1) if feas_vals else None
            ),
        }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"args": vars(args), "summary": summary}, f, indent=2)
    print(f"[feasibility] wrote {args.out}")


if __name__ == "__main__":
    main()
