"""Pose-graph sampler: portal-first, covisible by construction.

Consumes a SceneSpec (+ its TriSoup for exact visibility), produces poses.json:
graph-mode poses (tuple training), portal bridge poses (cross-room connectivity),
trajectory-mode sequences (optional), supervision-only marks (novel-view targets).

Guarantees enforced before returning (else raises SamplerFailure for the caller's
regenerate loop): single connected component on edges >= edge_thres, min degree >= 2,
K-walk feasibility for requested K values.

Height stratification: 60% eye 1.5±0.15 / 20% cam 2.2-2.6 / 15% pet 0.3-0.6 /
5% special. Pose count: max(24, area/15*8 + 4*n_portals), no upper cap.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import events
from .mesh import TriSoup, raycast, raycast_solid
from .spec import FLAG_GLASS, FLAG_MIRROR, FLAG_EMISSIVE

# Pixels on glass/mirror/emissive faces and pixels seen through glass carry
# unreliable RGB (transmission, reflection, emission). Covisibility evidence
# excludes them, so that window-heavy pairs do not look covisible through
# pixels that show a different surface than the geometry.
_AMB = FLAG_GLASS | FLAG_MIRROR | FLAG_EMISSIVE
from .spec import SceneSpec


class SamplerFailure(RuntimeError):
    pass


def pose_to_R(yaw, pitch):
    """World z-up; camera OpenCV convention (x right, y down, z forward).
    yaw about world z (0 = +x direction), pitch positive up. Returns R cam->world.
    right = fwd x world_up (the opposite sign, right=(-sy,cy,0), is a 180 deg
    roll about the optical axis: an upside-down panorama, det still +1, so no
    mirror)."""
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    fwd = np.array([cy * cp, sy * cp, sp])
    right = np.array([sy, -cy, 0.0])
    down = np.cross(fwd, right)          # = (sp*cy, sp*sy, -cp); world down at pitch=0
    return np.stack([right, down, fwd], axis=1)  # columns = cam axes in world


def _heights(rng, n):
    cls = rng.choice(4, n, p=[0.60, 0.20, 0.15, 0.05])
    z = np.where(cls == 0, rng.normal(1.5, 0.15, n),
        np.where(cls == 1, rng.uniform(2.2, 2.6, n),
        np.where(cls == 2, rng.uniform(0.3, 0.6, n), rng.uniform(0.8, 2.0, n))))
    return np.clip(z, 0.25, 2.62), cls


def _room_candidates(room, furniture, rng, step=0.5, margin=0.4):
    xs = np.arange(room.x0 + margin, room.x1 - margin + 1e-9, step)
    ys = np.arange(room.y0 + margin, room.y1 - margin + 1e-9, step)
    if len(xs) == 0 or len(ys) == 0:
        return np.zeros((0, 2))
    g = np.stack(np.meshgrid(xs, ys), -1).reshape(-1, 2)
    g = g + rng.uniform(-0.15, 0.15, g.shape)
    keep = np.ones(len(g), bool)
    for fu in furniture:
        if fu.room is not None:
            keep &= ~((np.abs(g[:, 0] - fu.cx) < fu.sx / 2 + 0.25) &
                      (np.abs(g[:, 1] - fu.cy) < fu.sy / 2 + 0.25))
    return g[keep]


def covis_pair(soup: TriSoup, pa, pb, n_rays=128, rng=None, use_embree=False):
    """Fraction of A's supervisable surface points that B also supervisably
    sees (and vice versa; min).

    Glass-aware: treating glass as an opaque first hit makes two opposite
    errors: (1) points on window panes count as covisibility evidence although
    their RGB is transmission and reflection; (2) a sightline through a pane
    counts as occluded although the pane is transparent. Evidence = solid,
    unambiguous points whose paths from both poses cross no glass."""
    rng = rng or np.random.default_rng(0)
    out = []
    for o_from, o_to in [(pa, pb), (pb, pa)]:
        d = rng.normal(size=(n_rays, 3))
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        r = raycast_solid(soup, o_from, d, use_embree=use_embree)
        ok = np.isfinite(r["t"]) & (r["face"] >= 0)
        ok &= (r["flags_accum"] & _AMB) == 0          # path A->pt clean
        ok &= (soup.flags[r["face"].clip(0)] & _AMB) == 0   # pt unambiguous
        if ok.sum() < 8:
            out.append(0.0)
            continue
        pts = o_from + d[ok] * r["t"][ok][:, None]
        seg = pts - o_to
        L = np.linalg.norm(seg, axis=1)
        r2 = raycast_solid(soup, np.broadcast_to(o_to, seg.shape),
                           seg / L[:, None], use_embree=use_embree)
        vis = (r2["t"] >= L - 0.05) & ((r2["flags_accum"] & _AMB) == 0)
        out.append(float(vis.mean()))
    return min(out)


def tri_pair(soup: TriSoup, pa, pb, n_rays=96, rng=None, use_embree=False):
    """Triangulation-quality proxy for a pose pair: median sin(intersection angle)
    over co-visible surface points. Classical grounding: two-ray intersection
    error grows ~1/sin(gamma) along depth (DOP / Forstner-Wrobel 2016), so high
    covisibility with near-parallel rays is still a poor training pair. 0.0 = degenerate.
    """
    rng = rng or np.random.default_rng(0)
    pa, pb = np.asarray(pa, float), np.asarray(pb, float)
    d = rng.normal(size=(n_rays, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    r = raycast_solid(soup, pa, d, use_embree=use_embree)   # same
    hit = np.isfinite(r["t"]) & (r["face"] >= 0)            # glass-aware
    hit &= (r["flags_accum"] & _AMB) == 0                   # population as
    hit &= (soup.flags[r["face"].clip(0)] & _AMB) == 0      # covis_pair
    if hit.sum() < 8:
        return 0.0
    pts = pa + d[hit] * r["t"][hit][:, None]
    seg = pts - pb
    L = np.linalg.norm(seg, axis=1)
    r2 = raycast_solid(soup, np.broadcast_to(pb, seg.shape),
                       seg / L[:, None], use_embree=use_embree)
    vis = (r2["t"] >= L - 0.05) & ((r2["flags_accum"] & _AMB) == 0)
    if vis.sum() < 8:
        return 0.0
    ua = (pa - pts[vis]) / np.linalg.norm(pa - pts[vis], axis=1, keepdims=True)
    ub = (pb - pts[vis]) / np.linalg.norm(pb - pts[vis], axis=1, keepdims=True)
    cosg = np.clip((ua * ub).sum(1), -1.0, 1.0)
    return float(np.median(np.sqrt(1.0 - cosg ** 2)))


def _lambda2(adj):
    deg = adj.sum(1)
    L = np.diag(deg) - adj
    ev = np.linalg.eigvalsh(L)
    return float(ev[1]) if len(ev) > 1 else 0.0


def _k_walk_ok(adj_bool, K, tries=20, rng=None):
    """Does a path of K distinct nodes exist on the thresholded graph? Greedy DFS probes."""
    rng = rng or np.random.default_rng(0)
    n = len(adj_bool)
    if K > n:
        return False
    nodes = list(range(n))
    for _ in range(tries):
        start = int(rng.integers(n))
        path = [start]
        used = {start}
        while len(path) < K:
            nbrs = [j for j in nodes if adj_bool[path[-1], j] and j not in used]
            if not nbrs:
                break
            nxt = int(rng.choice(nbrs))
            path.append(nxt)
            used.add(nxt)
        if len(path) >= K:
            return True
    return False


_CLEAR_DIRS = None


def _clear_dirs():
    global _CLEAR_DIRS
    if _CLEAR_DIRS is None:
        ds = []
        for v in ([1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]):
            ds.append(v)
        for a in (-1, 1):
            for b in (-1, 1):
                ds += [[a, b, 0], [a, 0, b], [0, a, b]]
        for a in (-1, 1):
            for b in (-1, 1):
                for c in (-1, 1):
                    ds.append([a, b, c])
        d = np.asarray(ds, np.float64)
        _CLEAR_DIRS = d / np.linalg.norm(d, axis=1, keepdims=True)
    return _CLEAR_DIRS


def pose_clearance(soup, p3, use_embree=False):
    """26-direction clearance probe: returns (d_min, inside).
    inside = majority-backface among near hits -> camera center in a solid.
    Without a camera-vs-geometry check, poses can be wedged into furniture or
    walls (d_min down to 4mm): pet-height cameras are not blocked by low
    furniture, and bridge poses and the first two poses skip the
    covisibility test."""
    dirs = _clear_dirs()
    o = np.broadcast_to(np.asarray(p3, np.float64), dirs.shape)
    hit = raycast(soup, o, dirs, use_embree=use_embree)
    t, face = hit["t"], hit["face"]
    ok = np.isfinite(t) & (t > 0) & (face >= 0)
    if not ok.any():
        return float("inf"), False
    d_min = float(t[ok].min())
    tri = soup.tri[face[ok]]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    fn /= np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-12)
    back = (dirs[ok] * fn).sum(1) > 0.0
    near = t[ok] < 0.6
    inside = int(near.sum()) >= 6 and float((back & near).sum()) / max(int(near.sum()), 1) >= 0.6
    return d_min, bool(inside)


def sample_scene(spec: SceneSpec, soup: TriSoup, edge_thres=0.30, walk_ks=(4, 8, 12),
                 covis_rays=96, sup_frac=0.15, traj=False, seed=None, use_embree=False):
    rng = np.random.default_rng(spec.seed if seed is None else seed)
    area = sum(r.area for r in spec.rooms)
    n_doors = sum(1 for p in spec.portals if p.kind != "window")
    n_target = max(24, int(np.ceil(area / 15.0 * 8)) + 4 * n_doors)   # no upper cap
    events.log("sample_start", scene=spec.name, area=round(area, 1),
               n_target=n_target, n_doors=n_doors)

    # candidates: per-room grid + portal bridge positions (portal-first)
    cand, cand_room, cand_kind = [], [], []
    _bridge_dir = {}                       # cand idx -> (portal center, unit normal side)
    for ri, room in enumerate(spec.rooms):
        g = _room_candidates(room, [f for f in spec.furniture if f.room == ri], rng)
        cand += list(g)
        cand_room += [ri] * len(g)
        cand_kind += ["graph"] * len(g)
    for p in spec.portals:
        if p.kind == "window":
            continue
        cx, cy = (p.x0 + p.x1) / 2, (p.y0 + p.y1) / 2
        wall = np.array([p.x1 - p.x0, p.y1 - p.y0])
        lat = wall / max(np.linalg.norm(wall), 1e-9)
        nrm = np.array([-wall[1], wall[0]])
        nrm = nrm / max(np.linalg.norm(nrm), 1e-9)
        # Doorway relay pose: panorama covisibility straight through a real
        # doorway between two rooms is intrinsically marginal (~0.2-0.35: each
        # sphere is dominated by its own room). A camera in the opening sees
        # half a sphere of each room, so relay<->side edges are strong;
        # connectivity rides strong edges instead of a lowered threshold. The
        # lateral scan dodges the open leaf.
        ctr3 = np.array([cx, cy, 1.5])
        relay = None
        for off_n in (0.0, 0.2, -0.2):
            for off_l in (0.0, 0.2, -0.2, 0.35, -0.35):
                q = np.array([cx, cy]) + nrm * off_n + lat * off_l
                d_c, ins = pose_clearance(
                    soup, np.array([q[0], q[1], 1.5]), use_embree=use_embree)
                if not ins and d_c >= 0.25 and (relay is None or d_c > relay[0]):
                    relay = (d_c, q)
        if relay is not None:
            cand.append(relay[1])
            cand_room.append(p.room_a)
            cand_kind.append("bridge")
            _bridge_dir[len(cand) - 1] = (np.array([cx, cy]), nrm)
        for sign, ri in [(+1.0, p.room_a), (-1.0, p.room_b)]:
            # line-of-sight fan: door leaves swing into doorways, so a fixed
            # point at +-0.7 along the normal often wedges against the open
            # leaf. Scan a small fan on this side and keep the point with the
            # best clearance among those that still see the portal centre
            # (unblocked ray at doorway height); bridges must look through
            # the opening, not around it.
            best = None
            for off_n in (0.55, 0.75, 1.0, 1.3):
                for off_l in (0.0, 0.3, -0.3):
                    q = np.array([cx, cy]) + nrm * (sign * off_n) + lat * off_l
                    p_try = np.array([q[0], q[1], 1.5])
                    d_c, ins = pose_clearance(soup, p_try, use_embree=use_embree)
                    if ins or d_c < 0.25:
                        continue
                    seg = ctr3 - p_try
                    dist = np.linalg.norm(seg)
                    hit = raycast(soup, p_try[None], (seg / dist)[None],
                                  use_embree=use_embree)
                    t0 = float(hit["t"][0]) if np.isfinite(hit["t"][0]) else np.inf
                    if t0 < dist - 0.05:          # leaf/frame blocks the doorway view
                        continue
                    if best is None or d_c > best[0]:
                        best = (d_c, q)
            q = best[1] if best is not None else np.array([cx, cy]) + nrm * (sign * 0.7)
            cand.append(q)
            cand_room.append(ri)
            cand_kind.append("bridge")
            _bridge_dir[len(cand) - 1] = (np.array([cx, cy]), nrm * sign)
    cand = np.asarray(cand)
    order = rng.permutation(len(cand))
    # bridges first (portal-first: cross-room skeleton before room filler)
    order = np.concatenate([[i for i in order if cand_kind[i] == "bridge"],
                            [i for i in order if cand_kind[i] == "graph"]]).astype(int)

    # greedy acceptance with covisibility edges
    zs, zcls = _heights(rng, len(cand))
    # bridges are connectivity infrastructure: keep them in the doorway sight
    # band (below the ~2.03 m header, above furniture); a pet-height bridge
    # behind a half-open leaf is the worst case
    for i, k in enumerate(cand_kind):
        if k == "bridge":
            zs[i] = float(np.clip(rng.normal(1.5, 0.1), 1.3, 1.7))
    accepted = []            # indices into cand
    P3 = []                  # 3d positions
    edges = {}               # (a,b)->w over accepted indices
    def deg(i):
        return sum(1 for (a, b), w in edges.items() if w >= edge_thres and i in (a, b))
    clear_min = 0.25
    for ci in order:
        if len(accepted) >= n_target:
            break
        p3 = np.array([cand[ci][0], cand[ci][1], zs[ci]])
        if any(np.linalg.norm(p3[:2] - q[:2]) < 0.55 for q in P3):
            continue
        # camera-clearance check for every pose (no exemptions, including
        # bridges and the first two poses)
        d_min, inside = pose_clearance(soup, p3, use_embree=use_embree)
        if inside or d_min < clear_min:
            if cand_kind[ci] != "bridge":
                continue
            # bridge rescue ladder: nudge further from the portal, then raise
            # to eye height — bridges carry cross-room connectivity
            rescued = False
            base = np.array([cand[ci][0], cand[ci][1]])
            hint = _bridge_dir.get(ci)
            for fac in (1.35, 1.7):
                for z_try in (zs[ci], 1.5):
                    q2 = base if hint is None else (hint[0] + hint[1] * (0.7 * fac))
                    p_try = np.array([q2[0], q2[1], z_try])
                    d2, in2 = pose_clearance(soup, p_try, use_embree=use_embree)
                    if not in2 and d2 >= clear_min:
                        p3, rescued = p_try, True
                        break
                if rescued:
                    break
            if not rescued:
                continue
        ws = {}
        for j, q in enumerate(P3):
            same_room = cand_room[ci] == cand_room[accepted[j]]
            near = np.linalg.norm(p3 - q) < 8.0
            if same_room or near or cand_kind[ci] == "bridge" or cand_kind[accepted[j]] == "bridge":
                ws[j] = covis_pair(soup, p3, q, n_rays=covis_rays, rng=rng, use_embree=use_embree)
        strong = sum(1 for w in ws.values() if w >= edge_thres)
        if len(P3) >= 2 and strong < 2 and cand_kind[ci] != "bridge":
            continue
        k = len(accepted)
        accepted.append(ci)
        P3.append(p3)
        for j, w in ws.items():
            edges[(j, k)] = w
    n = len(P3)
    if n < min(24, n_target):
        raise SamplerFailure(f"only {n}/{n_target} poses accepted")

    # connectivity repair: connect components via best cross pair (bridge retry)
    adj = np.zeros((n, n))
    for (a, b), w in edges.items():
        adj[a, b] = adj[b, a] = w
    def components(th):
        seen, comps = set(), []
        for s in range(n):
            if s in seen:
                continue
            stack, comp = [s], []
            while stack:
                u = stack.pop()
                if u in seen:
                    continue
                seen.add(u)
                comp.append(u)
                stack += [v for v in range(n) if adj[u, v] >= th and v not in seen]
            comps.append(comp)
        return comps
    comps = components(edge_thres)
    for _ in range(10):
        if len(comps) == 1:
            break
        A, B = comps[0], comps[1]
        best = (-1, None)
        for a in A:
            for b in B:
                w = adj[a, b] or covis_pair(soup, P3[a], P3[b], n_rays=covis_rays, rng=rng,
                                            use_embree=use_embree)
                adj[a, b] = adj[b, a] = w
                if w > best[0]:
                    best = (w, (a, b))
        if best[0] < edge_thres:
            # last resort: a tree-critical door can legitimately carry
            # near-zero covisibility (leaf mostly closed / long corridor); drop
            # the smaller component instead of failing the scene if the
            # remainder keeps enough poses. Weak cross-door tuples must not
            # exist in the data at all.
            comps_all = sorted(components(edge_thres), key=len, reverse=True)
            keep_idx = sorted(comps_all[0])
            if len(keep_idx) >= max(24, int(0.6 * n_target)):
                events.log("drop_components", scene=spec.name,
                           dropped=[len(c) for c in comps_all[1:]],
                           kept=len(keep_idx), best_covis=round(best[0], 3))
                sel = {old: new for new, old in enumerate(keep_idx)}
                P3 = [P3[i] for i in keep_idx]
                accepted = [accepted[i] for i in keep_idx]
                n = len(P3)
                adj = adj[np.ix_(keep_idx, keep_idx)]
                edges = {(sel[a], sel[b]): w for (a, b), w in edges.items()
                         if a in sel and b in sel}
                comps = components(edge_thres)
                continue
            raise SamplerFailure(f"components not connectable: best covis {best[0]:.2f}")
        comps = components(edge_thres)

    # guarantees
    adj_bool = adj >= edge_thres
    degs = adj_bool.sum(1)
    lam2 = _lambda2(adj_bool.astype(float))
    walk_ok = {K: _k_walk_ok(adj_bool, K, rng=rng) for K in walk_ks}
    if degs.min() < 2 or len(components(edge_thres)) != 1 or not all(walk_ok.values()):
        raise SamplerFailure(f"guarantees failed: min_deg={degs.min()} "
                             f"comps={len(components(edge_thres))} walk={walk_ok}")

    # orientation: POI-weighted 70% / uniform 30%; pitch prior 60% amplitude
    pois = [np.array([(p.x0 + p.x1) / 2, (p.y0 + p.y1) / 2]) for p in spec.portals]
    pois += [np.array([(r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2]) for r in spec.rooms]
    yaw, pitch = [], []
    for p3 in P3:
        if rng.random() < 0.7 and pois:
            tgt = pois[int(rng.integers(len(pois)))]
            yaw.append(float(np.arctan2(tgt[1] - p3[1], tgt[0] - p3[0])) + rng.normal(0, 0.3))
        else:
            yaw.append(float(rng.uniform(-np.pi, np.pi)))
        pitch.append(float(np.clip(rng.normal(0, 0.10), -0.5, 0.5)) * 0.6)

    # supervision-only marks
    sup = rng.random(n) < sup_frac

    poses = [{"id": k, "pos": [round(float(x), 4) for x in P3[k]],
              "yaw": round(yaw[k], 4), "pitch": round(pitch[k], 4),
              "height_class": int(zcls[accepted[k]]), "room": int(cand_room[accepted[k]]),
              "kind": cand_kind[accepted[k]], "supervision_only": bool(sup[k])}
             for k in range(n)]
    # triangulation-angle DOP proxy per retained edge: [a, b, covis, tri_sin]
    edge_list = []
    tri_vals = []
    for a in range(n):
        for b in range(a + 1, n):
            if adj[a, b] >= 0.05:
                ts = tri_pair(soup, P3[a], P3[b], rng=rng, use_embree=use_embree) \
                    if adj[a, b] >= edge_thres else 0.0
                if adj[a, b] >= edge_thres:
                    tri_vals.append(ts)
                edge_list.append([a, b, round(float(adj[a, b]), 3), round(ts, 3)])
    tri_q = (np.percentile(tri_vals, [10, 50, 90]).round(3).tolist()
             if tri_vals else [0.0, 0.0, 0.0])
    report = {"n_poses": n, "n_target": n_target, "min_degree": int(degs.min()),
              "lambda2": round(lam2, 3), "components": 1,
              "k_walk_ok": {str(k): bool(v) for k, v in walk_ok.items()},
              "supervision_share": round(float(sup.mean()), 3),
              "height_hist": np.bincount(zcls[accepted], minlength=4).tolist(),
              "bridge_poses": int(sum(1 for p in poses if p["kind"] == "bridge")),
              "tri_sin_p10_50_90": tri_q}
    events.log("sample_done", scene=spec.name, **{k: v for k, v in report.items()
                                                  if k != "height_hist"})
    result = {"scene": spec.name, "poses": poses, "edges": edge_list, "report": report}
    if traj:
        result["trajectory"] = _trajectory(spec, soup, P3, rng)
    return result


def _trajectory(spec, soup, P3, rng, n_frames=80):
    """Trajectory mode: waypoints through pose anchors (nearest-neighbor chain),
    linearly interpolated, constant eye height."""
    pts = np.array([p[:2] for p in P3])
    start = int(rng.integers(len(pts)))
    chain = [start]
    left = set(range(len(pts))) - {start}
    while left and len(chain) < 8:
        cur = pts[chain[-1]]
        nxt = min(left, key=lambda j: np.linalg.norm(pts[j] - cur))
        chain.append(nxt)
        left.discard(nxt)
    wp = pts[chain]
    t = np.linspace(0, len(wp) - 1, n_frames)
    i0 = np.clip(t.astype(int), 0, len(wp) - 2)
    fr = t - i0
    pos = wp[i0] * (1 - fr[:, None]) + wp[i0 + 1] * fr[:, None]
    d = np.gradient(pos, axis=0)
    yaw = np.arctan2(d[:, 1], d[:, 0])
    return [{"pos": [round(float(x), 4) for x in [p[0], p[1], 1.5]],
             "yaw": round(float(y), 4), "pitch": 0.0} for p, y in zip(pos, yaw)]


def save_poses(result: dict, path: Path):
    Path(path).write_text(json.dumps(result, indent=1))
