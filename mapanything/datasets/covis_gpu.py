"""GPU covisibility in torch (float64), used to recompute covisibility after camera sampling.

Same definition as scripts/compute_covisibility.py, so recomputed values are comparable with the precomputed
`covisibility/v0/covisibility.npy`:
  cov[i,j] = (# of i's valid points that, in j's cam frame, are forward (z>0), match j's ray-map by
              max-cosine > cos_thres, hit a VALID j-pixel, and are depth-consistent
              (|range - tgt_depth| <= depth_abs + depth_rel*tgt_depth)) / (# of i's valid points)
bicov(i,j) = min(cov[i,j], cov[j,i]).

Works for any view (rendered, cropped or camera-sampled): each view is
{rays (G,3) unit, depth (G,), valid (G,) bool, R (3,3) cam->world, t (3,)} in a common world frame, with
rays and R in the same camera convention (the renderer frame of the packs or the OpenCV frame of the camera
sampler; the result does not depend on it). Device-agnostic; GPU and CPU give identical results in float64.
A 2D crop does not change the pose (same optical centre); a camera-sampled view has R = c2w_src @ R_rel,
a rotation about the source optical centre, which preserves the world frame.
"""
import torch

COS_THRES = 0.998      # ~3.6 deg (matches compute_covisibility; tolerant of 128-downsample quantisation)
DEPTH_ABS = 0.10       # 10 cm
DEPTH_REL = 0.05       # 5 %


def covis_matrix(frames, cos_thres=COS_THRES, depth_abs=DEPTH_ABS, depth_rel=DEPTH_REL,
                 chunk=4096):
    """frames: list of dict with torch tensors (fp64) on a common device:
        rays  (G,3) unit ray dirs in the cam frame (same camera convention as R)
        depth (G,)  ray-range (metres)
        valid (G,)  bool train-visible & depth>0
        R (3,3) cam->world, t (3,) cam centre in world
    Returns NxN cov (fp64, same device). Diagonal = 1."""
    N = len(frames); dev = frames[0]["rays"].device
    # Precompute per-frame world source points (valid only) + the full ray/depth/valid grids for lookup.
    Vt = torch.zeros(N, dtype=torch.get_default_dtype(), device=dev)
    for f in frames:
        pts_cam = f["rays"] * f["depth"].unsqueeze(-1)             # (G,3)
        f["_pw"] = (pts_cam @ f["R"].T) + f["t"]                   # world points (G,3)
        f["_src"] = f["_pw"][f["valid"]]                          # (Vi,3) valid source pts
    for k in range(N):
        Vt[k] = frames[k]["_src"].shape[0]
    cov = torch.eye(N, dtype=torch.get_default_dtype(), device=dev)            # diag = 1; no host sync in the loop
    for i in range(N):
        P_world = frames[i]["_src"]
        Vi = P_world.shape[0]
        if Vi == 0:
            continue
        for j in range(N):
            if i == j:
                continue
            fj = frames[j]
            P_cam_j = (P_world - fj["t"]) @ fj["R"]                # (Vi,3) = R_j^T @ (P - t_j)
            dist = P_cam_j.norm(dim=-1).clamp_min(1e-8)
            d = P_cam_j / dist.unsqueeze(-1)
            fwd = P_cam_j[:, 2] > 0
            ray_j = fj["rays"]; depth_j = fj["depth"]; valid_j = fj["valid"]
            hits = torch.zeros((), dtype=torch.get_default_dtype(), device=dev)
            for s in range(0, Vi, chunk):                          # chunk over all source points (no .item())
                e = min(s + chunk, Vi)
                cos = d[s:e] @ ray_j.T                             # nearest-ray = max cosine
                max_cos, idx = cos.max(dim=-1)
                err = (dist[s:e] - depth_j[idx]).abs()
                tol = depth_abs + depth_rel * depth_j[idx]
                ok = (max_cos > cos_thres) & valid_j[idx] & (err <= tol) & fwd[s:e]
                hits = hits + ok.sum().to(torch.get_default_dtype())
            cov[i, j] = hits / Vt[i].clamp_min(1)                 # tensor op; no host sync
    return cov                                                    # caller .cpu() once

