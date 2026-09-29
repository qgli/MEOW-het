#!/usr/bin/env python3
"""Wid3R §3.4 covisibility sampling for MP3D point-map evaluation.

Wid3R: "compute pairwise distances between all images based on their camera
positions, forming a 2D distance matrix. We then apply a softmax operation to the
negative distances to obtain a probability matrix, which is used to sample input
images with higher probability for closer viewpoints."

We sample a set of K panoramas per scene that overlap: pick a random anchor, then
draw the remaining K-1 panos without replacement with probabilities
softmax(-distance to the anchor / tau) (closer = higher prob). This picks
spatially clustered (overlapping) panos rather than a random spread; the __main__
check compares the sampled pairwise distances with those of the whole scene.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from parse_conf import parse_conf_from_zip, scan_dir


def pano_centers(panos):
    uuids = list(panos.keys())
    C = np.array([panos[u].center for u in uuids])
    return uuids, C


def softmax_neg(d, tau):
    """softmax(-d/tau) over a 1D distance array (ignores +inf entries)."""
    x = -d / tau
    x = x - np.nanmax(x[np.isfinite(x)])
    e = np.exp(x)
    e[~np.isfinite(d)] = 0.0
    s = e.sum()
    return e / s if s > 0 else np.full_like(e, 1.0 / len(e))


def sample_covis(centers, k, tau=None, rng=None):
    """Return indices of K panos sampled to overlap (Wid3R §3.4 softmax(-dist)).

    Build the pairwise distance matrix, pick a random anchor, then sample the
    remaining K-1 panos from softmax(-D[anchor]/tau) without replacement so they
    cluster around the anchor (closer = higher prob). tau defaults to the median
    nearest-neighbour distance (adaptive), making the distribution peaked on the
    near neighbours rather than flat.
    """
    rng = rng or np.random.default_rng()
    n = len(centers)
    k = min(k, n)
    D = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=-1)
    if tau is None:
        nn = np.partition(D + np.eye(n) * 1e9, 1, axis=1)[:, 1]  # nearest-neighbour dist
        tau = max(np.median(nn), 1e-3)
    anchor = int(rng.integers(n))
    selected = [anchor]
    cand = [i for i in range(n) if i != anchor]
    d = D[anchor, cand]
    p = softmax_neg(d, tau)
    extra = rng.choice(cand, size=k - 1, replace=False, p=p)
    selected += [int(x) for x in np.atleast_1d(extra)]
    return selected


def sample_scene(scan_path, k=8, tau=2.0, seed=0):
    panos = parse_conf_from_zip(scan_dir(scan_path))
    uuids, C = pano_centers(panos)
    idx = sample_covis(C, k, tau, np.random.default_rng(seed))
    return [uuids[i] for i in idx], C[idx]


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: covis_sample.py <scan_dir>  (a Matterport3D scan directory)")
    scan = sys.argv[1]
    panos = parse_conf_from_zip(scan_dir(scan))
    uuids, C = pano_centers(panos)
    print(f"scan has {len(uuids)} panos; full-scene pairwise dist median="
          f"{np.median([np.linalg.norm(C[i]-C[j]) for i in range(len(C)) for j in range(i+1,len(C))]):.2f}m")
    rng = np.random.default_rng(0)
    sel = sample_covis(C, 8, rng=rng)
    Csel = C[sel]
    pair = [np.linalg.norm(Csel[i]-Csel[j]) for i in range(len(sel)) for j in range(i+1,len(sel))]
    print(f"sampled K={len(sel)} panos: pairwise dist median={np.median(pair):.2f}m max={np.max(pair):.2f}m")
    print("=> the sampled median should be much smaller than the full-scene median (overlap), "
          "confirming covis sampling clusters nearby panos")
