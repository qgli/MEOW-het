"""Covisibility connectivity check for camera-sampled tuples.

The training loss supervises all views in the frame of view 0, while the loader samples connected
sub-graphs of the covisibility (random walk, `covisibility_thres=0.25`); a view that is linked to the others
only through a chain is still fully supervised, located through the connected graph by attention. The online
check is therefore graph connectivity, not covisibility with a reference view:

    a tuple is valid iff its symmetric covisibility graph (edges with 0.5 * (cov + cov.T) > thres) has a
    single connected component.

Camera sampling (model, field of view, principal point, roll, tilt) changes the covisibility, so it is
recomputed on the sampled views. If the graph is disconnected, the views outside the largest component are
moved toward their native configuration in a few bounded steps. The threshold is never relaxed and no view is
dropped silently: a tuple that stays disconnected is reported with its largest component, and the camera
sampler discards it.

The module is independent of the source and of the spec format: the caller provides `render(spec)->view` and
`toward_native(spec, frac)->spec`.
"""
from __future__ import annotations
import numpy as np
import torch
from mapanything.datasets import covis_gpu as _CG

DEFAULT_THRES = 0.25     # matches base_dataset _random_walk_sampling covisibility_thres


# ----------------------------------------------------------------------------- graph primitives
def _components(adj: np.ndarray):
    """Connected components of a boolean KxK adjacency (undirected). Returns list of index-sets."""
    K = adj.shape[0]; seen = [False] * K; comps = []
    for s in range(K):
        if seen[s]:
            continue
        stack = [s]; seen[s] = True; comp = []
        while stack:
            c = stack.pop(); comp.append(c)
            for j in range(K):
                if adj[c, j] and not seen[j]:
                    seen[j] = True; stack.append(j)
        comps.append(set(comp))
    return comps


def _is_connected_without(adj: np.ndarray, drop: int) -> bool:
    """Is the graph (minus node `drop`) still connected over the remaining nodes?"""
    K = adj.shape[0]; rest = [k for k in range(K) if k != drop]
    if len(rest) <= 1:
        return True
    seen = {rest[0]}; stack = [rest[0]]
    while stack:
        c = stack.pop()
        for j in rest:
            if adj[c, j] and j not in seen:
                seen.add(j); stack.append(j)
    return len(seen) == len(rest)


def covis_graph(views, thres: float = DEFAULT_THRES):
    """Build the connectivity graph of a tuple of views.

    Returns dict:
      cov     (K,K)  asymmetric cov[i,j] = frac of i seen by j (covis_gpu)
      bidir   (K,K)  0.5*(cov+cov.T)  -- the walk's edge metric
      adj     (K,K)  bool  bidir > thres (off-diagonal)
      components     list of index-sets
      connected      bool (single component)
      largest        set  (largest component)
      degree  (K,)   int  number of neighbours
      articulation   set  nodes whose removal disconnects the (currently connected) graph (fragile links)
    """
    cov = _CG.covis_matrix(views).detach().cpu().numpy()
    K = cov.shape[0]
    bidir = 0.5 * (cov + cov.T)
    adj = (bidir > thres) & ~np.eye(K, dtype=bool)
    comps = _components(adj)
    comps_sorted = sorted(comps, key=len, reverse=True)
    largest = comps_sorted[0] if comps_sorted else set()
    connected = len(comps) == 1
    degree = adj.sum(1).astype(int)
    artic = set()
    if connected and K > 2:
        for i in range(K):
            if not _is_connected_without(adj, i):
                artic.add(i)
    return dict(cov=cov, bidir=bidir, adj=adj, components=comps, connected=connected,
                largest=largest, degree=degree, articulation=artic)


# ----------------------------------------------------------------------------- aimed-mode orientation
def look_rotation(d, up=(0.0, 0.0, 1.0)):
    """World rotation (cam->world) whose +z axis (camera forward) points along d; the up vector only fixes
    the roll about d. Used by the aimed mode: parallel views share one d; converging views use
    d = surface_point - camera_centre.

    The camera y axis (the second column) is aligned with the world up vector. In the OpenCV camera
    convention used by the sampler, image y points down, so aimed-mode views come out rotated by
    180 degrees about the optical axis relative to native views (world up at the bottom of the image),
    before the sampled roll is applied. The final model was trained with this behaviour, which is
    kept here unchanged for reproducibility."""
    d = d / (d.norm() + 1e-9)
    up = torch.tensor(up, dtype=d.dtype, device=d.device)
    if abs(float(d @ up)) > 0.95:
        up = torch.tensor([0.0, 1.0, 0.0], dtype=d.dtype, device=d.device)
    r = torch.linalg.cross(up, d); r = r / (r.norm() + 1e-9)
    u = torch.linalg.cross(d, r)
    return torch.stack([r, u, d], dim=1)


# ----------------------------------------------------------------------------- online check and bounded restore
def enforce_connectivity(specs, render, toward_native, thres: float = DEFAULT_THRES,
                         max_restore: int = 3, restore_fracs=(0.5, 0.8, 1.0)):
    """Render a tuple, check graph connectivity at thres, and move the views outside the largest component
    toward their native configuration (bounded number of steps).

    Args:
      specs:          list of opaque per-view spec objects.
      render(spec):   -> view dict {rays,depth,valid,R,t} (the caller routes the source and resamples).
      toward_native(spec, frac): -> spec moved a fraction `frac` toward its native configuration (e.g. field
                      of view, principal point, tilt; frac=1.0 is the full native value). Roll is left unchanged.
      thres:          edge threshold (default 0.25, the loader's covisibility_thres).
      max_restore:    maximum number of restore iterations.
      restore_fracs:  pull fraction per iteration (0.5, 0.8, then 1.0).

    Returns dict: views, specs (possibly restored), graph, connected(bool), keep(list: all views, or the
    largest component if still disconnected), dropped(list: the other views), n_restore(int),
    restored(list of view indices that were moved).
    """
    specs = list(specs); K = len(specs)
    views = [render(s) for s in specs]
    g = covis_graph(views, thres)
    nr = 0; restored = set()
    while not g["connected"] and nr < max_restore:
        frac = restore_fracs[min(nr, len(restore_fracs) - 1)]
        comp = g["largest"]
        for i in range(K):
            if i not in comp:
                specs[i] = toward_native(specs[i], frac)
                views[i] = render(specs[i])
                restored.add(i)
        g = covis_graph(views, thres); nr += 1
    if g["connected"]:
        keep = list(range(K)); dropped = []
    else:
        keep = sorted(g["largest"]); dropped = [i for i in range(K) if i not in g["largest"]]
    return dict(views=views, specs=specs, graph=g, connected=g["connected"],
                keep=keep, dropped=dropped, n_restore=nr, restored=sorted(restored))
