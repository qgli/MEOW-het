"""N-way constrained random walk on a covisibility matrix.

`constrained_random_walk_nway(cov, type_labels, type_order, target_count, rng, ...)`:
  - type_labels[i] is the type label of node i (any hashable, e.g.
    "pin", "erp", or "fish_180", "pin_14mm")
  - type_order is the ordered list of allowed type labels
  - target_count[t] = required count of type type_order[t]
  - Walk only accepts neighbor j if its type still has remaining quota.
  - Backtracks on dead-end and decrements running count for that branch.

Generalises the three-type ``constrained_random_walk`` (pinhole / fisheye /
panorama) to any type axis (e.g. the 3 base types or the 6 camera names).
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np


def constrained_random_walk_nway(
    cov: np.ndarray,
    type_labels: Sequence[str],
    type_order: Sequence[str],
    target_count: Sequence[int],
    rng: np.random.Generator,
    covisibility_thres: float = 0.25,
    max_retries: int = 8,
    use_bidirectional: bool = True,
) -> Tuple[np.ndarray, bool]:
    """Random walk restricted to a target type composition (N-way).

    Returns (walk_indices, success). On failure returns best partial walk.
    """
    K = int(sum(target_count))
    n_types = len(type_order)
    assert len(target_count) == n_types
    type_idx = {t: i for i, t in enumerate(type_order)}
    N = cov.shape[0]

    type_to_nodes: List[List[int]] = [[] for _ in range(n_types)]
    node_type_idx = np.empty(N, dtype=np.int64)
    for i, t in enumerate(type_labels):
        if t in type_idx:
            ti = type_idx[t]
            type_to_nodes[ti].append(i)
            node_type_idx[i] = ti
        else:
            node_type_idx[i] = -1

    # Structural prune
    for ti in range(n_types):
        if len(type_to_nodes[ti]) < target_count[ti]:
            return np.array([], dtype=int), False

    best: List[int] = []
    for _ in range(max_retries):
        cur = [0] * n_types
        visited: set = set()
        walk: List[int] = []
        stack: List[int] = []

        cand_starts: List[int] = []
        for ti in range(n_types):
            if target_count[ti] > 0:
                cand_starts.extend(type_to_nodes[ti])
        if not cand_starts:
            break
        start = int(rng.choice(cand_starts))
        walk.append(start); visited.add(start); stack.append(start)
        cur[int(node_type_idx[start])] += 1

        while len(walk) < K and stack:
            curr = stack[-1]
            if use_bidirectional:
                pcov = (cov[curr, :] + cov[:, curr].T) / 2.0
            else:
                pcov = cov[curr, :].copy()
            pcov = pcov / (pcov[curr] + 1e-8)
            pcov[curr] = 0
            adj = np.flatnonzero(pcov > covisibility_thres)
            cands = []
            for j in adj:
                if j in visited:
                    continue
                ti = int(node_type_idx[j])
                if ti < 0:
                    continue
                if cur[ti] < target_count[ti]:
                    cands.append(int(j))
            if cands:
                nxt = int(rng.choice(cands))
                walk.append(nxt); visited.add(nxt); stack.append(nxt)
                cur[int(node_type_idx[nxt])] += 1
            else:
                stack.pop()
                cur[int(node_type_idx[curr])] -= 1

        if len(walk) > len(best):
            best = walk
        if len(walk) == K and tuple(cur) == tuple(target_count):
            return np.array(walk), True

    return np.array(best), False
