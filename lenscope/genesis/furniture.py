"""Furniture mesh kernel for second-generation scenes (bpy-free).

Design contract (avoids black-hole render artifacts):
- every primitive is watertight by construction: welded vertices, each edge
  shared by exactly two triangles in opposite winding, outward orientation,
  signed volume > 0, no degenerate triangles;
- `mesh_checks()` re-verifies all of that numerically; the tests run it over
  every assembly type x many seeds, so geometry correctness is enforced
  before Blender ever sees a mesh (lightable, textureable);
- parts of an assembly never share coplanar faces: they interpenetrate by
  >= 3 mm (the burial rule, applied to furniture);
- pure numpy/math. bpy sees only (verts, faces) lists.

Primitives: beveled_box (rounded-box vertex projection on a welded subdivided
cube), lathe (revolved profile, poles or caps), cushion (beveled box inflated
along residual normals + noise -> fabric volume), wavy_band (curtain ribbon
with thickness). Assemblies are parametric part lists built from these.
"""
from __future__ import annotations

import math

import numpy as np

# mesh core


def _weld(verts, faces, tol=1e-7):
    key = np.round(np.asarray(verts, np.float64) / tol).astype(np.int64)
    _, idx, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    V = np.asarray(verts, np.float64)[idx]
    F = inv[np.asarray(faces, np.int64)]
    keep = ~((F[:, 0] == F[:, 1]) | (F[:, 1] == F[:, 2]) | (F[:, 0] == F[:, 2]))
    return V, F[keep]


def mesh_checks(V, F):
    """Numeric geometry check: closed 2-manifold, coherent outward winding,
    no degenerate faces, positive enclosed volume."""
    V = np.asarray(V, np.float64)
    F = np.asarray(F, np.int64)
    e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    und = np.sort(e, axis=1)
    _, counts = np.unique(und, axis=0, return_counts=True)
    closed = bool((counts == 2).all())
    # winding coherence: every directed edge appears exactly once
    _, dcounts = np.unique(e, axis=0, return_counts=True)
    oriented = bool((dcounts == 1).all())
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    cross = np.cross(b - a, c - a)
    area2 = np.linalg.norm(cross, axis=1)
    vol = float((a * cross).sum() / 6.0)
    return {"closed": closed, "oriented": oriented,
            "min_area": float(area2.min() / 2.0) if len(area2) else 0.0,
            "volume": vol, "n_verts": int(len(V)), "n_tris": int(len(F)),
            "ok": closed and oriented and vol > 1e-9
                  and (len(area2) == 0 or area2.min() / 2.0 > 1e-10)}


def transform(V, yaw_deg=0.0, t=(0.0, 0.0, 0.0)):
    V = np.asarray(V, np.float64)
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return V @ R.T + np.asarray(t, np.float64)


# primitives


def _subdiv_cube(seg):
    """Welded subdivided unit cube [-0.5,0.5]^3, outward winding."""
    verts, faces = [], []
    n = seg + 1
    grid = np.linspace(-0.5, 0.5, n)
    axes = [((0, 1), 2, +0.5), ((0, 1), 2, -0.5),
            ((0, 2), 1, +0.5), ((0, 2), 1, -0.5),
            ((1, 2), 0, +0.5), ((1, 2), 0, -0.5)]
    for (ua, va), wa, wv in axes:
        base = len(verts)
        for i in range(n):
            for j in range(n):
                p = [0.0, 0.0, 0.0]
                p[ua], p[va], p[wa] = grid[i], grid[j], wv
                verts.append(p)
        for i in range(seg):
            for j in range(seg):
                q = [base + i * n + j, base + i * n + j + 1,
                     base + (i + 1) * n + j + 1, base + (i + 1) * n + j]
                # outward orientation depends on the face's handedness
                p0, p1, p2 = (np.asarray(verts[q[0]]), np.asarray(verts[q[1]]),
                              np.asarray(verts[q[2]]))
                nrm = np.cross(p1 - p0, p2 - p0)
                if nrm[wa] * wv < 0:
                    q = q[::-1]
                faces += [[q[0], q[1], q[2]], [q[0], q[2], q[3]]]
    return _weld(verts, faces)


def beveled_box(sx, sy, sz, bevel=0.01, seg=2):
    """Rounded box: subdivided cube projected onto the rounded-box surface
    (clamp-core + radial offset). Watertight; bevel in meters."""
    h = np.array([sx, sy, sz], np.float64) / 2.0
    b = min(bevel, float(h.min()) * 0.9)
    V, F = _subdiv_cube(max(seg, 1) + 1)
    P = V * (h * 2.0)                       # scale unit cube to full box
    core = np.clip(P, -(h - b), (h - b))
    d = P - core
    L = np.linalg.norm(d, axis=1, keepdims=True)
    dn = np.where(L > 1e-12, d / np.maximum(L, 1e-12), 0.0)
    Vp = core + dn * b
    # crisp faces: points on flat regions (d has 1 nonzero comp) stay exact
    return Vp, F


def lathe(profile, n=24, weld_tol=1e-7):
    """Revolve profile [(r,z),...] (bottom->top) around +Z. r=0 endpoints
    become poles; r>0 endpoints get cap fans. Watertight, outward winding
    for CCW-increasing z profiles."""
    prof = [(max(float(r), 0.0), float(z)) for r, z in profile]
    verts, rows = [], []
    for (r, z) in prof:
        if r < 1e-9:
            rows.append(("pole", len(verts)))
            verts.append([0.0, 0.0, z])
        else:
            row = []
            for k in range(n):
                a = 2.0 * math.pi * k / n
                row.append(len(verts))
                verts.append([r * math.cos(a), r * math.sin(a), z])
            rows.append(("ring", row))
    faces = []
    for i in range(len(rows) - 1):
        ta, ra = rows[i]
        tb, rb = rows[i + 1]
        if ta == "ring" and tb == "ring":
            for k in range(n):
                k2 = (k + 1) % n
                faces += [[ra[k], ra[k2], rb[k2]], [ra[k], rb[k2], rb[k]]]
        elif ta == "pole" and tb == "ring":
            for k in range(n):
                faces.append([ra, rb[(k + 1) % n], rb[k]])
        elif ta == "ring" and tb == "pole":
            for k in range(n):
                faces.append([rb, ra[k], ra[(k + 1) % n]])
    # caps for open r>0 ends (ring traversal opposite to the side quads so
    # every directed edge stays unique; bottom normal -z, top normal +z)
    if rows[0][0] == "ring":
        ctr = len(verts)
        verts.append([0.0, 0.0, prof[0][1]])
        for k in range(n):
            faces.append([ctr, rows[0][1][(k + 1) % n], rows[0][1][k]])
    if rows[-1][0] == "ring":
        ctr = len(verts)
        verts.append([0.0, 0.0, prof[-1][1]])
        for k in range(n):
            faces.append([ctr, rows[-1][1][k], rows[-1][1][(k + 1) % n]])
    V, F = _weld(verts, faces, weld_tol)
    ck = mesh_checks(V, F)
    if ck["volume"] < 0:                      # profile wound the other way
        F = F[:, ::-1]
    return V, F


def cylinder(r, h, n=16):
    return lathe([(r, 0.0), (r, h)], n=n)


def cone(r0, r1, h, n=16):
    return lathe([(r0, 0.0), (r1, h)], n=n)


def cushion(sx, sy, sz, puff=0.12, noise=0.015, rng=None, seg=3, bevel=None):
    """Fabric volume: beveled box inflated along the rounding direction with
    a face-center-peaked envelope + low-amp vertex noise. Watertight (only
    vertex displacement)."""
    b = bevel if bevel is not None else min(sx, sy, sz) * 0.25
    h = np.array([sx, sy, sz], np.float64) / 2.0
    bb = min(b, float(h.min()) * 0.9)
    V, F = _subdiv_cube(seg + 2)
    P = V * (h * 2.0)
    core = np.clip(P, -(h - bb), (h - bb))
    d = P - core
    L = np.linalg.norm(d, axis=1, keepdims=True)
    dn = np.where(L > 1e-12, d / np.maximum(L, 1e-12), 0.0)
    Vp = core + dn * bb
    # envelope: 1 at face centers, ->0 at edges/corners (use |normalized pos|)
    u = np.abs(V) * 2.0                       # 0 center .. 1 boundary per axis
    env = np.clip(1.0 - np.sort(u, axis=1)[:, 1], 0.0, 1.0) ** 1.5
    Vp = Vp + dn * (puff * min(sx, sy, sz)) * env[:, None]
    if rng is not None and noise > 0:
        g = np.asarray([[rng.uniform(-1, 1) for _ in range(3)]
                        for _ in range(len(Vp))])
        Vp = Vp + dn * (noise * min(sx, sy, sz)) * env[:, None] \
            * g[:, :1]                        # along inflation dir only
    return Vp, F


def wavy_band(width, height, amp=0.06, waves=5, thickness=0.006, seg_x=48):
    """Curtain ribbon: sine-curved sheet with thickness, closed all around.
    Local frame: x across width, y = depth (wave), z up."""
    xs = np.linspace(0.0, width, seg_x + 1)
    ys = amp * np.sin(np.linspace(0.0, waves * 2.0 * math.pi, seg_x + 1))
    verts, faces = [], []
    nv = len(xs)
    for zi, z in enumerate((0.0, height)):
        for side in (0.0, thickness):
            for k in range(nv):
                verts.append([xs[k], ys[k] + side, z])
    def vid(zi, si, k):
        return zi * 2 * nv + si * nv + k
    for k in range(nv - 1):
        # front sheet (side 0, outward -y-ish), back sheet (side 1, +y)
        faces += [[vid(0, 0, k), vid(1, 0, k), vid(1, 0, k + 1)],
                  [vid(0, 0, k), vid(1, 0, k + 1), vid(0, 0, k + 1)]]
        faces += [[vid(0, 1, k), vid(1, 1, k + 1), vid(1, 1, k)],
                  [vid(0, 1, k), vid(0, 1, k + 1), vid(1, 1, k + 1)]]
        # top + bottom rims
        faces += [[vid(1, 0, k), vid(1, 1, k), vid(1, 1, k + 1)],
                  [vid(1, 0, k), vid(1, 1, k + 1), vid(1, 0, k + 1)]]
        faces += [[vid(0, 0, k), vid(0, 1, k + 1), vid(0, 1, k)],
                  [vid(0, 0, k), vid(0, 0, k + 1), vid(0, 1, k + 1)]]
    # end rims
    for k in (0, nv - 1):
        q = [vid(0, 0, k), vid(0, 1, k), vid(1, 1, k), vid(1, 0, k)]
        if k == 0:
            faces += [[q[0], q[1], q[2]], [q[0], q[2], q[3]]]
        else:
            faces += [[q[0], q[2], q[1]], [q[0], q[3], q[2]]]
    V, F = _weld(verts, faces)
    if mesh_checks(V, F)["volume"] < 0:
        F = F[:, ::-1]
    return V, F


# 3D rotation helper (angled legs, tilted seat backs, leaning books)

def transform3(V, rot_deg=(0.0, 0.0, 0.0), t=(0.0, 0.0, 0.0), pivot=(0, 0, 0)):
    V = np.asarray(V, np.float64) - np.asarray(pivot, np.float64)
    rx, ry, rz = [math.radians(a) for a in rot_deg]
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    R = Rz @ Ry @ Rx
    return V @ R.T + np.asarray(pivot, np.float64) + np.asarray(t, np.float64)


def _part(name, kind, VF, t=(0, 0, 0), rot=(0, 0, 0), pivot=(0, 0, 0),
          obj=None, flags=None):
    V, F = VF
    if rot != (0, 0, 0) or t != (0, 0, 0):
        V = transform3(V, rot, t, pivot)
    return {"name": name, "kind": kind, "verts": V, "faces": F,
            "obj": obj, "flags": flags or {}}


_B = 0.003          # burial: parts interpenetrate >= 3 mm (no coplanar faces)


def _legs(rng_params, w, d, h, inset=0.06):
    """4 legs by drawn leg type; returns parts + leg style dict for manifest."""
    style = rng_params["leg_style"]
    r = rng_params["leg_r"]
    parts = []
    for i, (sx, sy) in enumerate([(-1, -1), (1, -1), (1, 1), (-1, 1)]):
        x = sx * (w / 2 - inset)
        y = sy * (d / 2 - inset)
        if style == "cylinder":
            VF = cylinder(r, h + _B, n=12)
            parts.append(_part(f"leg{i}", "wood", VF, t=(x, y, 0)))
        elif style == "square":
            VF = beveled_box(r * 2, r * 2, h + _B, bevel=r * 0.3, seg=1)
            parts.append(_part(f"leg{i}", "wood", VF, t=(x, y, (h + _B) / 2)))
        elif style == "taper":
            VF = cone(r * 1.25, r * 0.6, h + _B, n=10)
            parts.append(_part(f"leg{i}", "wood", VF, t=(x, y, 0)))
        else:                                   # splay: angled outward
            # rotating about the top lifts the foot by h*(1-cos(7deg))
            # (~0.75% of h), enough for tall coffee-table legs to clear the
            # whole 3mm floor sink. Analytic compensation drops the part by
            # exactly that lift.
            VF = cylinder(r, (h + _B) * 1.04, n=10)
            dz = -h * (1.0 - math.cos(math.radians(7.0)))
            parts.append(_part(f"leg{i}", "wood", VF, t=(x, y, dz),
                               rot=(sy * -7.0, sx * 7.0, 0), pivot=(0, 0, h)))
    return parts


# parametric assemblies
# Every sample_*(rng) draws all randomness -> params dict (goes into the
# manifest); build_item(type, params) is deterministic.

def sample_sofa(rng):
    return {"w": rng.uniform(1.6, 2.2), "d": rng.uniform(0.82, 0.95),
            "seat_h": rng.uniform(0.40, 0.45), "arm_w": rng.uniform(0.14, 0.22),
            "back_h": rng.uniform(0.75, 0.9), "leg_h": rng.uniform(0.05, 0.10),
            "n_seat": rng.choice([2, 3]),
            "leg_style": rng.choice(["cylinder", "square", "taper", "splay"]),
            "leg_r": rng.uniform(0.02, 0.032), "noise_seed": rng.randint(0, 9999)}


def build_sofa(p):
    import random as _rd
    nr = _rd.Random(p["noise_seed"])
    w, d, sh, aw, bh, lh = (p["w"], p["d"], p["seat_h"], p["arm_w"],
                            p["back_h"], p["leg_h"])
    parts = _legs(p, w - 2 * aw, d, lh)
    base_h = sh - lh - 0.10
    parts.append(_part("base", "fabric",
                       beveled_box(w, d, base_h + _B, bevel=0.02),
                       t=(0, 0, lh + (base_h + _B) / 2 - _B)))
    for s, x in ((-1, -(w - aw) / 2), (1, (w - aw) / 2)):
        parts.append(_part(f"arm{s}", "fabric",
                           beveled_box(aw, d, bh * 0.72, bevel=0.045, seg=2),
                           t=(x, 0, bh * 0.72 / 2 + lh - _B)))
    parts.append(_part("back", "fabric",
                       beveled_box(w - 2 * aw + 2 * _B, 0.18, bh - lh, bevel=0.035),
                       t=(0, -(d / 2 - 0.09), (bh - lh) / 2 + lh)))
    span = w - 2 * aw
    cw = span / p["n_seat"]
    for i in range(p["n_seat"]):
        x = -span / 2 + cw * (i + 0.5)
        parts.append(_part(f"seat{i}", "fabric",
                           cushion(cw - 0.015, d - 0.24, 0.14, rng=nr),
                           t=(x, 0.035, sh - 0.02)))
        parts.append(_part(f"bcush{i}", "fabric",
                           cushion(cw - 0.02, 0.13, 0.42, rng=nr),
                           t=(x, -(d / 2 - 0.20), sh + 0.18), rot=(-8, 0, 0)))
    # the seat is a soft support: trays/flat books/laptops land there like
    # in a lived-in room (clutter_supports caps count and
    # restricts kinds on soft surfaces).
    return {"parts": parts,
            "supports": [{"z": sh + 0.10, "soft": True,
                          "poly": [(-(w / 2 - aw - 0.08), -(d / 2 - 0.30)),
                                   ((w / 2 - aw - 0.08), -(d / 2 - 0.30)),
                                   ((w / 2 - aw - 0.08), (d / 2 - 0.16)),
                                   (-(w / 2 - aw - 0.08), (d / 2 - 0.16))]}],
            "half_xy": (w / 2, d / 2), "height": bh}


def sample_bed(rng):
    return {"w": rng.uniform(1.4, 1.8), "l": rng.uniform(1.9, 2.1),
            "frame_h": rng.uniform(0.22, 0.30), "matt_h": rng.uniform(0.18, 0.24),
            "head_h": rng.uniform(0.85, 1.2), "noise_seed": rng.randint(0, 9999)}


def build_bed(p):
    import random as _rd
    nr = _rd.Random(p["noise_seed"])
    # bed aesthetics on a noise_seed sub-stream: no draws from the main
    # furniture stream (layouts do not depend on it), all recorded
    # implicitly via noise_seed (manifest reconstructs deterministically).
    sr = _rd.Random(p["noise_seed"] ^ 0xBED5)
    hb_style = sr.choices(["plain", "slats", "panel"], weights=[40, 30, 30])[0]
    n_pillows = sr.choice([2, 2, 3, 4])
    has_throw = sr.random() < 0.45
    w, l, fh, mh, hh = p["w"], p["l"], p["frame_h"], p["matt_h"], p["head_h"]
    parts = [
        _part("frame", "wood", beveled_box(w, l, fh, bevel=0.015),
              t=(0, 0, fh / 2)),
        _part("mattress", "fabric",
              cushion(w - 0.06, l - 0.08, mh, puff=0.10, rng=nr),
              t=(0, 0.015, fh + mh / 2 - _B)),
    ]
    if hb_style == "slats":
        n_sl = sr.randint(5, 7)
        span = w + 0.04
        sw = span / (2 * n_sl - 1)
        for i in range(n_sl):
            x = -span / 2 + sw / 2 + i * 2 * sw
            parts.append(_part(f"headboard_s{i}", "wood",
                               beveled_box(sw, 0.05, hh - 0.06,
                                           bevel=0.008, seg=1),
                               t=(x, -(l / 2 - 0.02), (hh - 0.06) / 2)))
        parts.append(_part("headboard_rail", "wood",
                           beveled_box(span, 0.06, 0.07, bevel=0.01, seg=1),
                           t=(0, -(l / 2 - 0.02), hh - 0.035)))
    else:
        parts.append(_part("headboard", "wood",
                           beveled_box(w + 0.04, 0.06, hh, bevel=0.025, seg=2),
                           t=(0, -(l / 2 - 0.02), hh / 2)))
        if hb_style == "panel":
            for j in (0.35, 0.65):
                parts.append(_part(f"headboard_p{int(j*100)}", "wood",
                                   beveled_box(w - 0.10, 0.012, 0.05,
                                               bevel=0.004, seg=1),
                                   t=(0, -(l / 2 - 0.02) + 0.03, hh * j)))
    if has_throw:
        parts.append(_part("throw", "fabric",
                           cushion(w - 0.10, 0.55, 0.05, puff=0.30, rng=nr),
                           t=(0, l / 2 - 0.42, fh + mh + 0.012 - _B)))
    # 2-4 pillows (two rows when 3+; small jitter per pillow)
    pw = w * (0.36 if n_pillows <= 2 else 0.30)
    slots = [(-0.22, 0.35), (0.22, 0.35)]
    if n_pillows >= 3:
        slots.append((0.0 if n_pillows == 3 else -0.18, 0.52))
    if n_pillows >= 4:
        slots.append((0.18, 0.52))
    for i, (fx, fy) in enumerate(slots[:n_pillows]):
        parts.append(_part(f"pillow{i}", "fabric",
                           cushion(pw, 0.42, 0.13, puff=0.22, rng=nr),
                           t=(fx * w + sr.uniform(-0.02, 0.02),
                              -(l / 2 - fy) + sr.uniform(-0.02, 0.02),
                              fh + mh + 0.045 + (0.05 if fy > 0.4 else 0)),
                           rot=(-6 - sr.uniform(0, 6), 0,
                                sr.uniform(-4, 4))))
    # foot half of the mattress is a soft support (books/trays/laptops,
    # everyday bed residue; pillows own the head half).
    return {"parts": parts,
            "supports": [{"z": fh + mh + 0.03, "soft": True,
                          "poly": [(-w * 0.30, -l * 0.02), (w * 0.30,
                                                            -l * 0.02),
                                   (w * 0.30, l * 0.36), (-w * 0.30,
                                                          l * 0.36)]}],
            "half_xy": ((w + 0.04) / 2, l / 2), "height": max(hh, fh + mh + 0.15)}


def sample_table(rng, kind="dining"):
    dims = {"dining": (rng.uniform(1.3, 1.7), rng.uniform(0.8, 1.0), 0.74),
            "coffee": (rng.uniform(0.9, 1.2), rng.uniform(0.5, 0.65),
                       rng.uniform(0.42, 0.48)),
            "desk": (rng.uniform(1.1, 1.5), rng.uniform(0.55, 0.7), 0.74)}[kind]
    return {"w": dims[0], "d": dims[1], "h": dims[2],
            "top_t": rng.uniform(0.028, 0.05),
            "leg_style": rng.choice(["cylinder", "square", "taper", "splay"]),
            "leg_r": rng.uniform(0.02, 0.035)}


def build_table(p):
    w, d, h, tt = p["w"], p["d"], p["h"], p["top_t"]
    parts = _legs(p, w, d, h - tt)
    parts.append(_part("top", "wood", beveled_box(w, d, tt, bevel=0.008),
                       t=(0, 0, h - tt / 2)))
    poly = [(-w / 2 + 0.02, -d / 2 + 0.02), (w / 2 - 0.02, -d / 2 + 0.02),
            (w / 2 - 0.02, d / 2 - 0.02), (-w / 2 + 0.02, d / 2 - 0.02)]
    return {"parts": parts, "supports": [{"z": h, "poly": poly}],
            "half_xy": (w / 2, d / 2), "height": h}


def sample_chair(rng):
    return {"s": rng.uniform(0.40, 0.46), "seat_h": rng.uniform(0.43, 0.47),
            "back_h": rng.uniform(0.38, 0.48), "back_tilt": rng.uniform(5, 12),
            "leg_style": rng.choice(["cylinder", "square", "taper"]),
            "leg_r": rng.uniform(0.016, 0.024), "noise_seed": rng.randint(0, 9999)}


def build_chair(p):
    import random as _rd
    nr = _rd.Random(p["noise_seed"])
    s, sh, bh = p["s"], p["seat_h"], p["back_h"]
    parts = _legs(p, s, s, sh - 0.04, inset=0.035)
    parts.append(_part("seat", "fabric",
                       cushion(s, s, 0.055, puff=0.10, noise=0.008, rng=nr),
                       t=(0, 0, sh - 0.02)))
    parts.append(_part("back", "wood",
                       beveled_box(s - 0.04, 0.03, bh, bevel=0.01, seg=2),
                       t=(0, -(s / 2 - 0.025), sh + bh / 2 - 0.02),
                       rot=(-p["back_tilt"], 0, 0),
                       pivot=(0, -(s / 2 - 0.025), sh)))
    return {"parts": parts, "supports": [],
            "half_xy": (s / 2, s / 2 + 0.05), "height": sh + bh}


def sample_wardrobe(rng):
    return {"w": rng.uniform(1.0, 2.0), "d": rng.uniform(0.55, 0.65),
            "h": rng.uniform(1.9, 2.3), "doors": rng.choice([2, 2, 3]),
            "handle_style": rng.choice(["bar", "knob"])}


def build_wardrobe(p):
    w, d, h = p["w"], p["d"], p["h"]
    parts = [_part("body", "wood", beveled_box(w, d, h, bevel=0.008, seg=1),
                   t=(0, 0, h / 2))]
    # door seam grooves: dark thin strips 1 mm proud of the front face
    front = d / 2
    for i in range(1, p["doors"]):
        x = -w / 2 + w * i / p["doors"]
        parts.append(_part(f"seam{i}", "seam",
                           beveled_box(0.006, 0.004, h - 0.10, bevel=0.001, seg=1),
                           t=(x, front - 0.001, h / 2)))
    parts.append(_part("seam_top", "seam",
                       beveled_box(w - 0.06, 0.004, 0.006, bevel=0.001, seg=1),
                       t=(0, front - 0.001, h - 0.09)))
    for i in range(p["doors"]):
        x = -w / 2 + w * (i + 0.5) / p["doors"] + (0.07 if i % 2 == 0 else -0.07)
        if p["handle_style"] == "bar":
            parts.append(_part(f"handle{i}", "metal", cylinder(0.008, 0.26, n=10),
                               t=(x, front + 0.012, h * 0.52 - 0.13)))
        else:
            parts.append(_part(f"handle{i}", "metal",
                               lathe([(0.0, 0.0), (0.006, 0.0), (0.014, 0.02),
                                      (0.0, 0.028)], n=12),
                               t=(x, front - _B, h * 0.52), rot=(-90, 0, 0)))
    poly = [(-w / 2 + 0.02, -d / 2 + 0.02), (w / 2 - 0.02, -d / 2 + 0.02),
            (w / 2 - 0.02, d / 2 - 0.02), (-w / 2 + 0.02, d / 2 - 0.02)]
    return {"parts": parts, "supports": [{"z": h, "poly": poly}],
            "half_xy": (w / 2, d / 2), "height": h}


def sample_nightstand(rng):
    return {"w": rng.uniform(0.42, 0.55), "d": rng.uniform(0.35, 0.45),
            "h": rng.uniform(0.48, 0.60)}


def build_nightstand(p):
    w, d, h = p["w"], p["d"], p["h"]
    parts = [
        _part("body", "wood", beveled_box(w, d, h, bevel=0.006, seg=1),
              t=(0, 0, h / 2)),
        _part("seam", "seam", beveled_box(w - 0.05, 0.004, 0.005,
                                          bevel=0.001, seg=1),
              t=(0, d / 2 - 0.001, h * 0.62)),
        _part("knob", "metal", lathe([(0.0, 0.0), (0.008, 0.0), (0.012, 0.018),
                                      (0.0, 0.024)], n=12),
              t=(0, d / 2 - _B, h * 0.78), rot=(-90, 0, 0)),
    ]
    poly = [(-w / 2 + 0.015, -d / 2 + 0.015), (w / 2 - 0.015, -d / 2 + 0.015),
            (w / 2 - 0.015, d / 2 - 0.015), (-w / 2 + 0.015, d / 2 - 0.015)]
    return {"parts": parts, "supports": [{"z": h, "poly": poly}],
            "half_xy": (w / 2, d / 2), "height": h}


def sample_tv_stand(rng):
    return {"w": rng.uniform(1.2, 1.8), "d": rng.uniform(0.35, 0.45),
            "h": rng.uniform(0.42, 0.55), "tv_w": rng.uniform(0.95, 1.4),
            "tv_ar": rng.choice([16 / 9.0, 16 / 10.0]),
            # screens use the image channel; the set is sometimes on
            "screen_on": rng.random() < 0.45,
            "screen_nits": round(rng.uniform(2.0, 6.5), 2),
            "screen_use_img": rng.random() < 0.55}


def build_tv_stand(p):
    w, d, h = p["w"], p["d"], p["h"]
    parts = [_part("body", "wood", beveled_box(w, d, h, bevel=0.006, seg=1),
                   t=(0, 0, h / 2)),
             _part("seam", "seam", beveled_box(w - 0.06, 0.004, 0.005,
                                               bevel=0.001, seg=1),
                   t=(0, d / 2 - 0.001, h * 0.55))]
    tw = min(p["tv_w"], w * 0.92)
    th = tw / p["tv_ar"]
    parts += [
        _part("tv_foot", "metal", beveled_box(tw * 0.45, d * 0.6, 0.02,
                                              bevel=0.004, seg=1),
              t=(0, 0, h + 0.01 - _B), obj="Television"),
        _part("tv_neck", "metal", cylinder(0.018, 0.12, n=10),
              t=(0, 0, h - _B), obj="Television"),
        _part("tv_screen", "tv", beveled_box(tw, 0.035, th, bevel=0.006, seg=1),
              t=(0, 0, h + 0.10 + th / 2), obj="TvScreen",
              flags={"glossy": True, "screen_on": p.get("screen_on", False),
                     "screen_nits": p.get("screen_nits", 3.0)}),
    ]
    poly = [(-w / 2 + 0.02, -d / 2 + 0.02), (w / 2 - 0.02, -d / 2 + 0.02),
            (w / 2 - 0.02, d / 2 - 0.02), (-w / 2 + 0.02, d / 2 - 0.02)]
    return {"parts": parts, "supports": [{"z": h, "poly": poly,
                                          "occupied_x": (-tw / 2, tw / 2)}],
            "half_xy": (w / 2, d / 2), "height": h + 0.10 + th}


def sample_bookshelf(rng):
    return {"w": rng.uniform(0.8, 1.2), "d": rng.uniform(0.26, 0.34),
            "h": rng.uniform(1.6, 2.1), "n_shelves": rng.randint(3, 5),
            "t": rng.uniform(0.018, 0.028)}


def build_bookshelf(p):
    w, d, h, t = p["w"], p["d"], p["h"], p["t"]
    parts = [
        _part("sideL", "wood", beveled_box(t, d, h, bevel=0.004, seg=1),
              t=(-(w - t) / 2, 0, h / 2)),
        _part("sideR", "wood", beveled_box(t, d, h, bevel=0.004, seg=1),
              t=((w - t) / 2, 0, h / 2)),
        _part("top", "wood", beveled_box(w, d, t, bevel=0.004, seg=1),
              t=(0, 0, h - t / 2)),
        _part("bottom", "wood", beveled_box(w, d, t, bevel=0.004, seg=1),
              t=(0, 0, t / 2)),
        _part("backpanel", "wood", beveled_box(w, 0.012, h, bevel=0.003, seg=1),
              t=(0, -(d / 2 - 0.006), h / 2)),
    ]
    supports = []
    inner_w = w - 2 * t
    poly = [(-inner_w / 2 + 0.01, -d / 2 + 0.02), (inner_w / 2 - 0.01, -d / 2 + 0.02),
            (inner_w / 2 - 0.01, d / 2 - 0.02), (-inner_w / 2 + 0.01, d / 2 - 0.02)]
    for i in range(1, p["n_shelves"] + 1):
        z = t + (h - 2 * t) * i / (p["n_shelves"] + 1)
        parts.append(_part(f"shelf{i}", "wood",
                           beveled_box(inner_w + 2 * _B, d - 0.02, t,
                                       bevel=0.004, seg=1),
                           t=(0, 0.005, z)))
        supports.append({"z": z + t / 2, "poly": poly, "shelf": True,
                         "clear_h": (h - 2 * t) / (p["n_shelves"] + 1) - t})
    top_poly = [(-w / 2 + 0.02, -d / 2 + 0.02), (w / 2 - 0.02, -d / 2 + 0.02),
                (w / 2 - 0.02, d / 2 - 0.02), (-w / 2 + 0.02, d / 2 - 0.02)]
    supports.append({"z": h, "poly": top_poly})
    return {"parts": parts, "supports": supports,
            "half_xy": (w / 2, d / 2), "height": h}


def sample_mirror(rng):
    return {"w": rng.uniform(0.45, 0.7), "h": rng.uniform(0.9, 1.4),
            "frame_w": rng.uniform(0.03, 0.06)}


def build_mirror(p):
    w, h, fw = p["w"], p["h"], p["frame_w"]
    parts = [
        _part("frame", "wood", beveled_box(w, 0.035, h, bevel=0.008, seg=1),
              t=(0, 0, 0)),
        _part("glass", "mirror",
              beveled_box(w - 2 * fw, 0.012, h - 2 * fw, bevel=0.002, seg=1),
              t=(0, 0.014, 0), flags={"mirror": True}),
    ]
    return {"parts": parts, "supports": [], "half_xy": (w / 2, 0.03),
            "height": h}


def sample_curtain(rng, win_w, win_h, sill):
    """Curtain style variety with non-uniform pleats.
    Styles: double (two full panels) / single (one wide panel) / sheer
    (light double, geometrically identical; transparency is deliberately not
    used: alpha/transmission would desync RGB from the depth ground truth) /
    valance (short top band over double) / roman (horizontal-fold partial
    drop). Pleats = 2 incommensurate sinusoids + random phases + smoothed
    per-column jitter (avoids radiator-fin uniformity)."""
    style = rng.choice(["double", "double", "single", "sheer", "valance",
                        "roman"])
    return {"style": style, "w": win_w * rng.uniform(1.15, 1.45),
            "drop": sill + win_h + 0.12,
            "win_h": win_h, "sill": sill,
            "amp": rng.uniform(0.035, 0.085),
            "f1": rng.uniform(3.5, 7.0), "f2": rng.uniform(9.0, 17.0),
            "a2": rng.uniform(0.25, 0.6),
            "ph1": rng.uniform(0.0, 6.283), "ph2": rng.uniform(0.0, 6.283),
            "gap": rng.uniform(0.10, 0.5),
            "folds": rng.randint(3, 6),
            "noise_seed": rng.randint(0, 9999)}


def pleat_band(width, height, amp, f1, f2, a2, ph1, ph2, noise_seed,
               thickness=0.007, seg_x=56):
    """Curtain sheet with an irregular pleat profile (two sinusoids +
    smoothed jitter), closed all around (watertight)."""
    import random as _rd
    rr = _rd.Random(noise_seed)
    xs = np.linspace(0.0, width, seg_x + 1)
    tt = xs / max(width, 1e-9)
    ys = amp * (np.sin(2 * math.pi * f1 * tt + ph1)
                + a2 * np.sin(2 * math.pi * f2 * tt + ph2))
    jit = np.array([rr.uniform(-1, 1) for _ in range(seg_x + 1)])
    for _ in range(3):                       # smooth the jitter
        jit[1:-1] = 0.25 * jit[:-2] + 0.5 * jit[1:-1] + 0.25 * jit[2:]
    ys = ys + amp * 0.35 * jit
    verts, faces = [], []
    nv = len(xs)
    for z in (0.0, height):
        for side in (0.0, thickness):
            for k in range(nv):
                verts.append([xs[k], ys[k] + side, z])
    def vid(zi, si, k):
        return zi * 2 * nv + si * nv + k
    for k in range(nv - 1):
        faces += [[vid(0, 0, k), vid(1, 0, k), vid(1, 0, k + 1)],
                  [vid(0, 0, k), vid(1, 0, k + 1), vid(0, 0, k + 1)]]
        faces += [[vid(0, 1, k), vid(1, 1, k + 1), vid(1, 1, k)],
                  [vid(0, 1, k), vid(0, 1, k + 1), vid(1, 1, k + 1)]]
        faces += [[vid(1, 0, k), vid(1, 1, k), vid(1, 1, k + 1)],
                  [vid(1, 0, k), vid(1, 1, k + 1), vid(1, 0, k + 1)]]
        faces += [[vid(0, 0, k), vid(0, 1, k + 1), vid(0, 1, k)],
                  [vid(0, 0, k), vid(0, 0, k + 1), vid(0, 1, k + 1)]]
    for k in (0, nv - 1):
        q = [vid(0, 0, k), vid(0, 1, k), vid(1, 1, k), vid(1, 0, k)]
        if k == 0:
            faces += [[q[0], q[1], q[2]], [q[0], q[2], q[3]]]
        else:
            faces += [[q[0], q[2], q[1]], [q[0], q[3], q[2]]]
    V, F = _weld(verts, faces)
    if mesh_checks(V, F)["volume"] < 0:
        F = F[:, ::-1]
    return V, F


def build_curtain(p):
    """Style-dispatched curtain assembly (local x across, y depth, z up)."""
    total_w, drop = p["w"], p["drop"]
    style = p.get("style", "double")
    common = dict(amp=p.get("amp", 0.06), f1=p.get("f1", 5.0),
                  f2=p.get("f2", 12.0), a2=p.get("a2", 0.4),
                  ph1=p.get("ph1", 0.0), ph2=p.get("ph2", 1.7),
                  noise_seed=p.get("noise_seed", 0))
    parts = []
    if style == "roman":
        # horizontal soft folds = a pleat_band built along its profile axis,
        # then rotated so the profile runs vertically ((x,y,z)->(-z,y,x) is a
        # proper rotation: winding/orientation preserved; reusing the tested
        # kernel avoids hand-wound quads with wrong orientation)
        part_drop = (p.get("win_h", 1.3) + 0.2) * 0.55
        w2 = total_w * 0.92
        V, F = pleat_band(part_drop, w2,
                          amp=0.05, f1=p.get("folds", 4), f2=9.0, a2=0.15,
                          ph1=-1.5708, ph2=p.get("ph2", 1.7),
                          noise_seed=p.get("noise_seed", 0), seg_x=44)
        V = np.stack([V[:, 2], V[:, 1], -V[:, 0]], axis=1)   # R_y(90): det=+1
        # rotated band: x = old z in [0, w2] -> center; z = -old x in
        # [-part_drop, 0] -> hangs down from the header
        V = V + np.array([-w2 / 2, 0.02, 0.0])
        parts.append(_part("roman", "fabric", (V, F), t=(0, 0, drop)))
        return {"parts": parts, "supports": [],
                "half_xy": (total_w / 2, 0.06), "height": drop}
    if style == "single":
        panel_w = total_w * 0.9
        V, F = pleat_band(panel_w, drop - 0.08, **common)
        parts.append(_part("panel", "fabric", (V, F),
                           t=(-panel_w / 2, 0, 0.06)))
    else:                                    # double / sheer / valance base
        panel_w = max((total_w - p.get("gap", 0.3)) / 2, 0.2)
        for s, x0 in ((-1, -total_w / 2), (1, total_w / 2 - panel_w)):
            V, F = pleat_band(panel_w, drop - 0.08, **common)
            parts.append(_part(f"panel{s}", "fabric", (V, F),
                               t=(x0, 0, 0.06)))
    if style == "valance":
        vd = 0.28
        V, F = pleat_band(total_w, vd, **{**common, "amp": common["amp"] * 0.8})
        parts.append(_part("valance", "fabric", (V, F),
                           t=(-total_w / 2, 0.012, drop - vd)))
    return {"parts": parts, "supports": [], "half_xy": (total_w / 2, 0.10),
            "height": drop}


def sample_rug(rng):
    return {"w": rng.uniform(1.4, 2.4), "d": rng.uniform(0.9, 1.7)}


def build_rug(p):
    return {"parts": [_part("rug", "fabric",
                            beveled_box(p["w"], p["d"], 0.012, bevel=0.004,
                                        seg=1), t=(0, 0, 0.006))],
            "supports": [], "half_xy": (p["w"] / 2, p["d"] / 2), "height": 0.012}


# clutter

def _jitter_radial(VF, seed, amp=0.08):
    """Low-poly organic bodies: per-vertex radial jitter around the lathe
    axis (foliage/fruit read as grown, not ballooned). Topology untouched
    (watertightness preserved); amp is small on purpose. Geometry-level:
    render and ray cast share the same jittered mesh."""
    import random as _rd
    V, F = VF
    nr = _rd.Random(seed)
    V = [(x * (1 + nr.uniform(-amp, amp)), y * (1 + nr.uniform(-amp, amp)), z)
         for (x, y, z) in V] if isinstance(V, list) else V
    if not isinstance(V, list):
        import numpy as _np
        f = 1 + (_np.random.default_rng(seed).uniform(-amp, amp, len(V)))
        V = V.copy()
        V[:, 0] *= f
        V[:, 1] *= f
    return (V, F)


_GLAZE_HUES = [0.02, 0.07, 0.10, 0.33, 0.55, 0.60, 0.83, 0.95]
_FRUIT_HUES = [0.00, 0.05, 0.09, 0.13, 0.26, 0.32]


def _clutter_tint(rng, kind):
    """Per-item macro color + gloss (manifest-recorded, so scenes stay
    reconstructible); a shared per-class material would make every ceramic
    the same beige."""
    import colorsys
    if kind in ("book", "flatbook", "smallbox"):
        h, sat, val = rng.random(), rng.uniform(0.30, 0.85), rng.uniform(0.30, 0.8)
        gloss = rng.uniform(0.55, 0.85)
    elif kind in ("cup", "bowl", "plate", "vase", "figurine", "candle"):
        h = rng.choice(_GLAZE_HUES) + rng.uniform(-0.02, 0.02)
        sat = rng.uniform(0.05, 0.55)
        val = rng.uniform(0.35, 0.9)
        gloss = rng.uniform(0.05, 0.4)
    elif kind == "bottle":
        h = rng.choice([0.08, 0.30, 0.55, 0.60])
        sat, val = rng.uniform(0.25, 0.7), rng.uniform(0.2, 0.6)
        gloss = rng.uniform(0.05, 0.25)
    elif kind == "fruit":
        h = rng.choice(_FRUIT_HUES) + rng.uniform(-0.015, 0.015)
        sat, val = rng.uniform(0.55, 0.9), rng.uniform(0.35, 0.75)
        gloss = rng.uniform(0.25, 0.5)
    elif kind in ("pan", "laptop", "phone"):
        h, sat, val = rng.random(), rng.uniform(0.0, 0.08), rng.uniform(0.2, 0.6)
        gloss = rng.uniform(0.2, 0.5)
    else:                                   # tray & misc: wood tones
        h, sat, val = rng.uniform(0.05, 0.11), rng.uniform(0.3, 0.55), rng.uniform(0.25, 0.55)
        gloss = rng.uniform(0.4, 0.7)
    r, g, b = colorsys.hsv_to_rgb(h % 1.0, sat, val)
    return [round(r, 4), round(g, 4), round(b, 4)], round(gloss, 3)


def _sample_clutter_base(rng, kind=None):
    # wide kind pool: bottles/bowls/candles/fruit/trays/laptops/phones are
    # the everyday residue of real rooms (a narrow pool reads as a staged
    # show flat).
    kind = kind or rng.choice(["book", "cup", "plate", "vase", "smallbox",
                               "figurine", "bottle", "bowl", "candle",
                               "fruit", "tray", "flatbook"])
    if kind == "book":
        return {"kind": kind, "w": rng.uniform(0.13, 0.24),
                "t": rng.uniform(0.018, 0.05), "h": rng.uniform(0.19, 0.30)}
    if kind == "flatbook":
        return {"kind": kind, "w": rng.uniform(0.15, 0.26),
                "d": rng.uniform(0.11, 0.19), "h": rng.uniform(0.015, 0.05)}
    if kind == "cup":
        return {"kind": kind, "r": rng.uniform(0.032, 0.045),
                "h": rng.uniform(0.08, 0.12)}
    if kind == "bottle":
        return {"kind": kind, "r": rng.uniform(0.028, 0.042),
                "h": rng.uniform(0.17, 0.30), "neck": rng.uniform(0.35, 0.55)}
    if kind == "bowl":
        return {"kind": kind, "r": rng.uniform(0.07, 0.12),
                "h": rng.uniform(0.05, 0.09)}
    if kind == "candle":
        return {"kind": kind, "r": rng.uniform(0.025, 0.045),
                "h": rng.uniform(0.06, 0.16)}
    if kind == "fruit":
        return {"kind": kind, "r": rng.uniform(0.032, 0.05)}
    if kind == "tray":
        return {"kind": kind, "w": rng.uniform(0.24, 0.4),
                "d": rng.uniform(0.16, 0.28)}
    if kind == "laptop":
        return {"kind": kind, "w": rng.uniform(0.28, 0.36),
                "d": rng.uniform(0.20, 0.25), "open": rng.uniform(100, 125)}
    if kind == "phone":
        return {"kind": kind, "w": rng.uniform(0.070, 0.082),
                "d": rng.uniform(0.145, 0.165)}
    if kind == "pan":
        return {"kind": kind, "r": rng.uniform(0.09, 0.14),
                "h": rng.uniform(0.045, 0.08)}
    if kind == "plate":
        return {"kind": kind, "r": rng.uniform(0.09, 0.13)}
    if kind == "vase":
        return {"kind": kind, "r": rng.uniform(0.05, 0.09),
                "h": rng.uniform(0.18, 0.34), "waist": rng.uniform(0.4, 0.8),
                "n": rng.randint(22, 28)}
    if kind == "smallbox":
        return {"kind": kind, "w": rng.uniform(0.12, 0.3),
                "d": rng.uniform(0.10, 0.24), "h": rng.uniform(0.06, 0.16)}
    return {"kind": "figurine", "r": rng.uniform(0.03, 0.05),
            "h": rng.uniform(0.10, 0.22), "n": rng.randint(16, 20)}


def sample_clutter_item(rng, kind=None):
    p = _sample_clutter_base(rng, kind)
    p["tint"], p["gloss"] = _clutter_tint(rng, p["kind"])   # per-item color + gloss
    return p


def build_clutter_item(p):
    k = p["kind"]
    if k == "book":
        VF = beveled_box(p["t"], p["w"], p["h"], bevel=0.003, seg=1)
        return {"parts": [_part("book", "paper", VF, t=(0, 0, p["h"] / 2))],
                "footprint_r": max(p["t"], p["w"]) / 2 * 1.05,
                "height": p["h"], "top_r": 0.0}
    if k == "cup":
        VF = lathe([(p["r"] * 0.8, 0.0), (p["r"], p["h"] * 0.25),
                    (p["r"], p["h"])], n=18)
        return {"parts": [_part("cup", "ceramic", VF)],
                "footprint_r": p["r"], "height": p["h"], "top_r": 0.0}
    if k == "plate":
        VF = lathe([(p["r"] * 0.5, 0.0), (p["r"] * 0.8, 0.008),
                    (p["r"], 0.03)], n=18)
        return {"parts": [_part("plate", "ceramic", VF)],
                "footprint_r": p["r"], "height": 0.03, "top_r": 0.0}
    if k == "vase":
        w = p["waist"]
        VF = lathe([(0.0, 0.0), (p["r"] * 0.75, 0.0), (p["r"], p["h"] * 0.3),
                    (p["r"] * w, p["h"] * 0.7), (p["r"] * 0.85 * w, p["h"]),
                    (0.0, p["h"])], n=p["n"])
        return {"parts": [_part("vase", "ceramic", VF)],
                "footprint_r": p["r"], "height": p["h"], "top_r": 0.0}
    if k == "smallbox":
        VF = beveled_box(p["w"], p["d"], p["h"], bevel=0.004, seg=1)
        return {"parts": [_part("smallbox", "paper", VF, t=(0, 0, p["h"] / 2))],
                "footprint_r": math.hypot(p["w"], p["d"]) / 2,
                "height": p["h"],
                "top_r": min(p["w"], p["d"]) / 2 * 0.8}   # stackable
    if k == "flatbook":
        VF = beveled_box(p["w"], p["d"], p["h"], bevel=0.003, seg=1)
        return {"parts": [_part("flatbook", "paper", VF,
                                t=(0, 0, p["h"] / 2))],
                "footprint_r": math.hypot(p["w"], p["d"]) / 2,
                "height": p["h"], "top_r": min(p["w"], p["d"]) / 2 * 0.7}
    if k == "bottle":
        nk = p["neck"]
        VF = lathe([(p["r"] * 0.7, 0.0), (p["r"], p["h"] * 0.06),
                    (p["r"], p["h"] * 0.55), (p["r"] * nk, p["h"] * 0.75),
                    (p["r"] * nk, p["h"] * 0.97), (p["r"] * nk * 1.25,
                                                   p["h"])], n=18)
        return {"parts": [_part("bottle", "ceramic", VF)],
                "footprint_r": p["r"], "height": p["h"], "top_r": 0.0}
    if k == "bowl":
        VF = lathe([(p["r"] * 0.45, 0.0), (p["r"] * 0.9, p["h"] * 0.45),
                    (p["r"], p["h"]), (p["r"] * 0.93, p["h"]),
                    (p["r"] * 0.5, p["h"] * 0.22)], n=20)
        return {"parts": [_part("bowl", "ceramic", VF)],
                "footprint_r": p["r"], "height": p["h"], "top_r": 0.0}
    if k == "candle":
        VF = lathe([(p["r"], 0.0), (p["r"], p["h"]), (p["r"] * 0.3,
                                                      p["h"])], n=12)
        return {"parts": [_part("candle", "ceramic", VF)],
                "footprint_r": p["r"], "height": p["h"], "top_r": 0.0}
    if k == "fruit":
        VF = _jitter_radial(lathe(
            [(0.0, 0.0), (p["r"], p["r"] * 0.85),
             (p["r"] * 0.35, p["r"] * 1.7), (0.0, p["r"] * 1.75)],
            n=16), seed=int(p["r"] * 1e5), amp=0.05)
        return {"parts": [_part("fruit", "foliage", VF)],
                "footprint_r": p["r"], "height": p["r"] * 1.75, "top_r": 0.0}
    if k == "tray":
        VF = beveled_box(p["w"], p["d"], 0.022, bevel=0.005, seg=1)
        return {"parts": [_part("tray", "wood", VF, t=(0, 0, 0.011))],
                "footprint_r": math.hypot(p["w"], p["d"]) / 2,
                "height": 0.022, "top_r": min(p["w"], p["d"]) / 2 * 0.85}
    if k == "laptop":
        base = beveled_box(p["w"], p["d"], 0.014, bevel=0.004, seg=1)
        lid = beveled_box(p["w"], p["d"], 0.008, bevel=0.003, seg=1)
        return {"parts": [
            _part("base", "metal", base, t=(0, 0, 0.007)),
            _part("lid", "metal", lid,
                  t=(0, -p["d"] / 2, 0.012), rot=(-(180 - p["open"]), 0, 0),
                  pivot=(0, p["d"] / 2, 0))],
            "footprint_r": math.hypot(p["w"], p["d"]) / 2,
            "height": 0.014 + p["d"] * 0.9, "top_r": 0.0}
    if k == "phone":
        VF = beveled_box(p["w"], p["d"], 0.008, bevel=0.003, seg=1)
        return {"parts": [_part("phone", "tv", VF, t=(0, 0, 0.004))],
                "footprint_r": math.hypot(p["w"], p["d"]) / 2,
                "height": 0.008, "top_r": 0.0}
    if k == "pan":
        VF = lathe([(p["r"] * 0.75, 0.0), (p["r"], p["h"]),
                    (p["r"] * 0.94, p["h"]), (p["r"] * 0.72, p["h"] * 0.18)],
                   n=16)
        return {"parts": [_part("pan", "metal", VF),
                          _part("handle", "metal",
                                beveled_box(0.14, 0.03, 0.018, bevel=0.005,
                                            seg=1),
                                t=(p["r"] + 0.065, 0, p["h"] - 0.012))],
                "footprint_r": p["r"] + 0.13, "height": p["h"],
                "top_r": 0.0}
    VF = lathe([(0.0, 0.0), (p["r"], 0.0), (p["r"] * 0.5, p["h"] * 0.45),
                (p["r"] * 0.8, p["h"] * 0.72), (0.0, p["h"])], n=p["n"])
    return {"parts": [_part("figurine", "ceramic", VF)],
            "footprint_r": p["r"], "height": p["h"], "top_r": 0.0}


_BUILDERS = {"sofa": build_sofa, "bed": build_bed, "table": build_table,
             "chair": build_chair, "wardrobe": build_wardrobe,
             "nightstand": build_nightstand, "tv_stand": build_tv_stand,
             "bookshelf": build_bookshelf, "mirror": build_mirror,
             "curtain": build_curtain, "rug": build_rug}


def build_item(ftype, params):
    """Deterministic: params (from the manifest) -> parts/supports."""
    return _BUILDERS[ftype](params)


def sample_painting(rng):
    return {"w": rng.uniform(0.40, 0.95), "h": rng.uniform(0.45, 1.05),
            "frame_w": rng.uniform(0.025, 0.05),
            "frame_t": rng.uniform(0.03, 0.05)}


def build_painting(p):
    """Wall art: frame + canvas as a separate object (PaintingCanvas_*) so
    the decal engine can give the canvas its own normalized UV + image."""
    w, h, fw, ft = p["w"], p["h"], p["frame_w"], p["frame_t"]
    parts = [
        _part("frame", "wood", beveled_box(w, ft, h, bevel=0.006, seg=1)),
        _part("canvas", "canvas",
              beveled_box(w - 2 * fw, 0.012, h - 2 * fw, bevel=0.001, seg=1),
              t=(0, ft / 2 - 0.004, 0), obj="PaintingCanvas"),
    ]
    return {"parts": parts, "supports": [], "half_xy": (w / 2, ft / 2),
            "height": h}


def sample_poster(rng):
    return {"w": rng.uniform(0.35, 0.6), "h": rng.uniform(0.5, 0.85)}


def build_poster(p):
    parts = [_part("sheet", "canvas",
                   beveled_box(p["w"], 0.006, p["h"], bevel=0.001, seg=1),
                   obj=None)]
    return {"parts": parts, "supports": [], "half_xy": (p["w"] / 2, 0.004),
            "height": p["h"]}


_BUILDERS["painting"] = build_painting
_BUILDERS["poster"] = build_poster


# metric box-projected UVs (bpy-free twin of build_scene._apply_box_uv)

# per dominant-axis-and-sign (u_idx, u_sign, v_idx, v_sign) chosen right-
# handed (u x v = n): projected UV winding matches 3D winding -> no mirrored
# UV faces (mirrored tangents would invert normal maps)
_UV_AXES = {
    (0, +1): (1, +1.0, 2, +1.0), (0, -1): (1, -1.0, 2, +1.0),
    (1, +1): (0, -1.0, 2, +1.0), (1, -1): (0, +1.0, 2, +1.0),
    (2, +1): (0, +1.0, 1, +1.0), (2, -1): (0, +1.0, 1, -1.0),
}


def box_project_uv(V, F, texel_m=1.0):
    """Per-loop UVs (n_tris, 3, 2): world-coordinate box projection so 1 UV
    tile == texel_m meters everywhere (metric texel; the texel-density check
    holds by construction on axis-aligned faces)."""
    V = np.asarray(V, np.float64)
    F = np.asarray(F, np.int64)
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    n = np.cross(b - a, c - a)
    k = np.abs(n).argmax(axis=1)
    P = np.stack([a, b, c], axis=1)               # (M, 3verts, 3xyz)
    uv = np.empty((len(F), 3, 2), np.float64)
    for axis in range(3):
        for sign in (+1, -1):
            m = (k == axis) & ((n[:, axis] >= 0) == (sign > 0))
            if not m.any():
                continue
            ui, us, vi, vs = _UV_AXES[(axis, sign)]
            uv[m, :, 0] = us * P[m, :, ui] / texel_m
            uv[m, :, 1] = vs * P[m, :, vi] / texel_m
    return uv


def uv_signed_areas(uv):
    """Shoelace per tri; positive == orientation preserved (no flip)."""
    a, b, c = uv[:, 0], uv[:, 1], uv[:, 2]
    return 0.5 * ((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
                  - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1]))


# luminaire shade geometry: lathe-based styles

_SHADE_STYLES = {
    "pendant": ["cone", "dome", "drum", "globe", "multi3"],
    "flush": ["disk", "dome_flat"],
    "bulb": ["globe_s"],
    "floor": ["cone_f", "drum_f"],
    "table": ["cone_t", "dome_t"],
}


def sample_shade_style(rng, fixture):
    styles = _SHADE_STYLES.get(fixture)
    return rng.choice(styles) if styles else ""


def build_lightshade(style, r=0.16, n=36, seed=0):
    # n = 36: with smooth shading the silhouette is the remaining polygon
    # tell on lamps, and 36 segments put it under ~1px at typical viewing
    # distances. Geometry-level, shared by render and ray cast (same mesh ->
    # alignment preserved).
    """One shade (or shade cluster) as watertight lathe parts, local origin
    at the shade's top center (hangs downward for pendants). Returns parts
    with kind='lightshade' (+ 'trim' cords/bars for multi-head)."""
    import random as _rd
    rr = _rd.Random(seed)
    if style == "cone":
        h = r * rr.uniform(1.0, 1.5)
        VF = lathe([(0.028, 0.0), (r, -h * 0.85), (r * 0.96, -h), (0.0, -h)],
                   n=n)
        return [_part("shade", "lightshade", VF)]
    if style == "dome":
        h = r * rr.uniform(0.8, 1.1)
        prof = [(0.028, 0.0)]
        for k in range(1, 6):
            a = k / 5.0
            prof.append((r * math.sin(a * math.pi / 2),
                         -h * (1 - math.cos(a * math.pi / 2)) - 0.001 * k))
        prof.append((r * 0.98, -h))
        prof.append((0.0, -h))
        VF = lathe(prof, n=n)
        return [_part("shade", "lightshade", VF)]
    if style == "drum":
        h = r * rr.uniform(0.9, 1.3)
        VF = lathe([(r * 0.98, 0.0), (r, -0.02), (r, -h), (0.0, -h)], n=n)
        return [_part("shade", "lightshade", VF)]
    if style == "globe":
        VF = lathe([(0.0, 0.0), (r * 0.7, -r * 0.3), (r, -r),
                    (r * 0.7, -r * 1.7), (0.0, -r * 2.0)], n=n)
        return [_part("shade", "lightshade", VF)]
    if style == "multi3":
        parts = [_part("bar", "metal",
                       beveled_box(0.62, 0.03, 0.025, bevel=0.005, seg=1),
                       t=(0, 0, -0.0125))]
        for i, dx in enumerate((-0.25, 0.0, 0.25)):
            parts.append(_part(f"cord{i}", "trim", cylinder(0.004, 0.14, n=8),
                               t=(dx, 0, -0.165)))
            sub = lathe([(0.02, 0.0), (r * 0.55, -r * 0.75),
                         (r * 0.52, -r * 0.85), (0.0, -r * 0.85)], n=14)
            parts.append(_part(f"shade{i}", "lightshade", sub,
                               t=(dx, 0, -0.165)))
        return parts
    if style == "disk":
        VF = lathe([(r * 0.6, 0.0), (r, -0.015), (r * 0.96, -0.07),
                    (0.0, -0.075)], n=n)
        return [_part("shade", "lightshade", VF)]
    if style == "dome_flat":
        VF = lathe([(r, 0.0), (r * 0.85, -0.05), (r * 0.45, -0.085),
                    (0.0, -0.09)], n=n)
        return [_part("shade", "lightshade", VF)]
    if style == "globe_s":
        VF = lathe([(0.0, 0.0), (0.045, -0.02), (0.05, -0.05),
                    (0.035, -0.085), (0.0, -0.1)], n=14)
        return [_part("shade", "lightshade", VF)]
    if style in ("cone_f", "cone_t"):
        rr2 = 0.17 if style == "cone_f" else 0.11
        h = rr2 * 1.35
        VF = lathe([(rr2 * 0.45, 0.0), (rr2, -h), (rr2 * 0.96, -h * 1.05),
                    (0.0, -h * 1.05)], n=n)
        return [_part("shade", "lightshade", VF)]
    if style in ("drum_f", "dome_t"):
        rr2 = 0.17 if style == "drum_f" else 0.10
        h = rr2 * 1.5
        VF = lathe([(rr2 * 0.94, 0.0), (rr2, -0.02), (rr2, -h), (0.0, -h)],
                   n=n)
        return [_part("shade", "lightshade", VF)]
    # fallback: drum
    VF = lathe([(r, 0.0), (r, -r), (0.0, -r)], n=n)
    return [_part("shade", "lightshade", VF)]



def sample_clock(rng):
    return {"r": rng.uniform(0.13, 0.20), "depth": rng.uniform(0.035, 0.05),
            "hour_deg": rng.uniform(0, 360), "minute_deg": rng.uniform(0, 360),
            "ticks": rng.random() < 0.7}


def build_clock(p):
    """Wall clock: lathe body + inset face + hands at random angles + ticks.
    Local origin at face center, +y = out of the wall."""
    r, d = p["r"], p["depth"]
    body = lathe([(r * 0.55, 0.0), (r, -0.008), (r, -d), (0.0, -d)], n=24)
    Vb = transform3(body[0], rot_deg=(-90, 0, 0))
    face = lathe([(r * 0.86, 0.0), (r * 0.86, -0.006), (0.0, -0.006)], n=24)
    Vf = transform3(face[0], rot_deg=(-90, 0, 0), t=(0, 0.004, 0))
    parts = [_part("body", "wood", (Vb, body[1])),
             _part("face", "ceramic", (Vf, face[1]))]
    for name, ang, ln, wd in (("hand_h", p["hour_deg"], r * 0.45, 0.014),
                              ("hand_m", p["minute_deg"], r * 0.68, 0.009)):
        VF = beveled_box(wd, 0.006, ln, bevel=0.002, seg=1)
        V = transform3(VF[0], rot_deg=(0, ang, 0), t=(0, 0.0105, 0),
                       pivot=(0, 0, -ln / 2 + 0.01))
        parts.append(_part(name, "seam", (V, VF[1])))
    if p.get("ticks"):
        for k in range(12):
            a = math.radians(k * 30)
            VF = beveled_box(0.012, 0.005, 0.026, bevel=0.001, seg=1)
            V = transform3(VF[0], t=(r * 0.74 * math.sin(a), 0.009,
                                     r * 0.74 * math.cos(a)))
            parts.append(_part(f"tick{k}", "seam", (V, VF[1])))
    return {"parts": parts, "supports": [], "half_xy": (r, d / 2),
            "height": 2 * r}


_BUILDERS["clock"] = build_clock


# bathroom / kitchen fixtures. Same kernel discipline as the furniture
# above (watertight parts, burial >=3mm, front = local +y, back = -y).

def sample_toilet(rng):
    return {"w": rng.uniform(0.36, 0.42), "d": rng.uniform(0.62, 0.70),
            "h": rng.uniform(0.40, 0.44), "tank_h": rng.uniform(0.32, 0.40),
            "noise_seed": rng.randint(0, 9999)}


def build_toilet(p):
    w, d, h, th = p["w"], p["d"], p["h"], p["tank_h"]
    parts = [
        _part("bowl", "ceramic",
              lathe([(0.0, 0.0), (0.12, 0.0), (0.15, 0.10), (0.13, 0.24),
                     (0.17, 0.34), (0.17, h - 0.03), (0.0, h - 0.03)], n=20),
              t=(0, d * 0.12, 0)),
        _part("seat", "ceramic",
              beveled_box(w, d * 0.62, 0.05, bevel=0.02, seg=2),
              t=(0, d * 0.10, h - 0.025)),
        _part("tank", "ceramic",
              beveled_box(w, 0.17, th, bevel=0.012, seg=1),
              t=(0, -(d / 2 - 0.10), h + th / 2 - _B)),
        _part("flush_btn", "metal", cylinder(0.02, 0.008, n=12),
              t=(0, -(d / 2 - 0.10), h + th - 0.004)),
    ]
    return {"parts": parts, "supports": [],
            "half_xy": (w / 2, d / 2), "height": h + th}


def sample_vanity(rng):
    return {"w": rng.uniform(0.55, 0.95), "d": rng.uniform(0.45, 0.55),
            "h": rng.uniform(0.82, 0.88), "noise_seed": rng.randint(0, 9999)}


def build_vanity(p):
    w, d, h = p["w"], p["d"], p["h"]
    parts = [
        _part("cabinet", "wood", beveled_box(w, d, h - 0.10, bevel=0.008),
              t=(0, 0, (h - 0.10) / 2)),
        _part("top", "stone", beveled_box(w + 0.02, d + 0.02, 0.03,
                                          bevel=0.006, seg=1),
              t=(0, 0, h - 0.085 + 0.015)),
        _part("basin", "ceramic",
              lathe([(0.0, 0.0), (0.14, 0.0), (0.17, 0.06), (0.18, 0.11),
                     (0.0, 0.11)], n=20),
              t=(0, 0.04, h - 0.07)),
        _part("faucet", "metal", cylinder(0.012, 0.16, n=10),
              t=(0, -(d / 2 - 0.10), h + 0.04)),
        _part("seam1", "seam", beveled_box(0.005, 0.004, h - 0.16,
                                           bevel=0.001, seg=1),
              t=(0, d / 2 - 0.001, (h - 0.10) / 2)),
    ]
    return {"parts": parts, "supports": [],
            "half_xy": ((w + 0.02) / 2, (d + 0.02) / 2), "height": h + 0.20}


def sample_bathtub(rng):
    return {"w": rng.uniform(1.5, 1.75), "d": rng.uniform(0.70, 0.80),
            "h": rng.uniform(0.52, 0.58), "noise_seed": rng.randint(0, 9999)}


def build_bathtub(p):
    """Solid tub body + proud rim strips (an open cavity would break the
    watertight-kernel discipline; the rim silhouette reads as a tub)."""
    w, d, h = p["w"], p["d"], p["h"]
    parts = [_part("body", "ceramic", beveled_box(w, d, h, bevel=0.03, seg=2),
                   t=(0, 0, h / 2))]
    for (dx, dy, sx, sy) in ((0, d / 2 - 0.05, w, 0.06),
                             (0, -(d / 2 - 0.05), w, 0.06),
                             (w / 2 - 0.05, 0, 0.06, d - 0.16),
                             (-(w / 2 - 0.05), 0, 0.06, d - 0.16)):
        parts.append(_part(f"rim{dx:.2f}_{dy:.2f}", "ceramic",
                           beveled_box(sx, sy, 0.04, bevel=0.012, seg=1),
                           t=(dx, dy, h + 0.012)))
    parts.append(_part("faucet", "metal", cylinder(0.014, 0.18, n=10),
                       t=(-(w / 2 - 0.14), 0, h + 0.10)))
    return {"parts": parts, "supports": [],
            "half_xy": (w / 2, d / 2), "height": h + 0.20}


def sample_shower(rng):
    return {"w": rng.uniform(0.85, 1.0), "d": rng.uniform(0.85, 1.0),
            "h": rng.uniform(1.95, 2.10), "noise_seed": rng.randint(0, 9999)}


def build_shower(p):
    """Open shower (no glass pane, which would complicate the transmission
    ground truth): tray + corner posts + rail + head. Reads as a shower stall."""
    w, d, h = p["w"], p["d"], p["h"]
    parts = [_part("tray", "ceramic", beveled_box(w, d, 0.08, bevel=0.02),
                   t=(0, 0, 0.04))]
    for i, (sx, sy) in enumerate(((-1, -1), (1, -1))):
        parts.append(_part(f"post{i}", "metal", cylinder(0.018, h, n=10),
                           t=(sx * (w / 2 - 0.03), sy * (d / 2 - 0.03), 0)))
    parts.append(_part("rail", "metal", cylinder(0.012, w - 0.05, n=8),
                       t=(0, -(d / 2 - 0.03), h - 0.02), rot=(0, 90, 0)))
    parts.append(_part("head", "metal",
                       lathe([(0.0, 0.0), (0.05, 0.0), (0.06, 0.02),
                              (0.012, 0.05), (0.0, 0.05)], n=14),
                       t=(0, -(d / 2 - 0.20), h - 0.25)))
    return {"parts": parts, "supports": [],
            "half_xy": (w / 2, d / 2), "height": h}


def sample_towel_bar(rng):
    return {"w": rng.uniform(0.55, 0.75), "noise_seed": rng.randint(0, 9999)}


def build_towel_bar(p):
    w = p["w"]
    parts = [
        _part("bar", "metal", cylinder(0.011, w, n=10),
              t=(0, 0.05, 0), rot=(0, 90, 0)),
        _part("towel", "fabric",
              beveled_box(w * 0.55, 0.05, 0.62, bevel=0.012, seg=2),
              t=(0, 0.055, -0.33)),
    ]
    for sgn in (-1, 1):
        parts.append(_part(f"bracket{sgn}", "metal",
                           beveled_box(0.02, 0.06, 0.02, bevel=0.004, seg=1),
                           t=(sgn * (w / 2 - 0.02), 0.02, 0)))
    return {"parts": parts, "supports": [],
            "half_xy": (w / 2, 0.09), "height": 0.7}


def sample_counter(rng, w):
    return {"w": w, "d": 0.60, "h": 0.90, "doors": max(1, int(w / 0.5)),
            "with_sink": False, "noise_seed": rng.randint(0, 9999)}


def build_counter(p):
    w, d, h = p["w"], p["d"], p["h"]
    parts = [
        _part("base", "wood", beveled_box(w, d - 0.03, h - 0.06, bevel=0.006),
              t=(0, -0.015, (h - 0.06) / 2)),
        _part("top", "stone", beveled_box(w + 0.02, d, 0.035, bevel=0.008,
                                          seg=1),
              t=(0, 0, h - 0.06 + 0.0175 - _B)),
    ]
    for i in range(1, p["doors"]):
        x = -w / 2 + w * i / p["doors"]
        parts.append(_part(f"seam{i}", "seam",
                           beveled_box(0.005, 0.004, h - 0.14, bevel=0.001,
                                       seg=1),
                           t=(x, d / 2 - 0.017, (h - 0.06) / 2)))
    for i in range(p["doors"]):
        x = -w / 2 + w * (i + 0.5) / p["doors"]
        parts.append(_part(f"handle{i}", "metal", cylinder(0.007, 0.12, n=8),
                           t=(x, d / 2 - 0.008, h * 0.62), rot=(90, 0, 0)))
    if p.get("with_sink"):
        parts.append(_part("sink_basin", "metal",
                           beveled_box(0.42, 0.36, 0.02, bevel=0.006, seg=1),
                           t=(0, 0, h - 0.06 + 0.035)))
        parts.append(_part("faucet", "metal", cylinder(0.012, 0.22, n=10),
                           t=(0, -(d / 2 - 0.12), h + 0.06)))
    poly = [(-w / 2 + 0.02, -d / 2 + 0.02), (w / 2 - 0.02, -d / 2 + 0.02),
            (w / 2 - 0.02, d / 2 - 0.06), (-w / 2 + 0.02, d / 2 - 0.06)]
    occ = (-0.26, 0.26) if p.get("with_sink") else None
    sup = {"z": h - 0.06 + 0.035, "poly": poly}
    if occ:
        sup["occupied_x"] = occ
    return {"parts": parts, "supports": [sup],
            "half_xy": ((w + 0.02) / 2, d / 2), "height": h}


def sample_stove(rng):
    return {"w": rng.uniform(0.58, 0.62), "d": 0.60, "h": 0.90,
            "noise_seed": rng.randint(0, 9999)}


def build_stove(p):
    w, d, h = p["w"], p["d"], p["h"]
    parts = [
        _part("body", "metal", beveled_box(w, d - 0.03, h - 0.05,
                                           bevel=0.006),
              t=(0, -0.015, (h - 0.05) / 2)),
        _part("cooktop", "tv", beveled_box(w - 0.02, d - 0.06, 0.02,
                                           bevel=0.004, seg=1),
              t=(0, -0.015, h - 0.05 + 0.01 - _B)),
        _part("oven_handle", "metal", cylinder(0.009, w - 0.12, n=8),
              t=(0, d / 2 - 0.02, h * 0.55), rot=(0, 90, 0)),
    ]
    for i, (sx, sy) in enumerate(((-1, -1), (1, -1), (-1, 1), (1, 1))):
        parts.append(_part(f"burner{i}", "metal",
                           lathe([(0.0, 0.0), (0.075, 0.0), (0.075, 0.015),
                                  (0.0, 0.015)], n=16),
                           t=(sx * w * 0.22, -0.015 + sy * (d - 0.06) * 0.22,
                              h - 0.04)))
    return {"parts": parts, "supports": [],
            "half_xy": (w / 2, d / 2), "height": h}


def sample_fridge(rng):
    return {"w": rng.uniform(0.60, 0.70), "d": rng.uniform(0.62, 0.70),
            "h": rng.uniform(1.65, 1.85), "noise_seed": rng.randint(0, 9999)}


def build_fridge(p):
    w, d, h = p["w"], p["d"], p["h"]
    zsplit = h * 0.68
    parts = [
        _part("body", "metal", beveled_box(w, d, h, bevel=0.010, seg=1),
              t=(0, 0, h / 2)),
        _part("seam_h", "seam", beveled_box(w - 0.04, 0.004, 0.006,
                                            bevel=0.001, seg=1),
              t=(0, d / 2 - 0.001, zsplit)),
    ]
    for z0, z1 in ((zsplit + 0.06, h - 0.10), (0.15, zsplit - 0.06)):
        parts.append(_part(f"handle{z0:.2f}", "metal",
                           cylinder(0.010, z1 - z0, n=8),
                           t=(w / 2 - 0.06, d / 2 + 0.012, (z0 + z1) / 2)))
    return {"parts": parts, "supports": [],
            "half_xy": (w / 2, d / 2), "height": h}


def sample_hood(rng):
    return {"w": rng.uniform(0.58, 0.62), "noise_seed": rng.randint(0, 9999)}


def build_hood(p):
    w = p["w"]
    parts = [
        _part("canopy", "metal", beveled_box(w, 0.48, 0.10, bevel=0.008,
                                             seg=1),
              t=(0, 0, 0.05)),
        _part("chimney", "metal", beveled_box(0.30, 0.30, 0.85, bevel=0.006,
                                              seg=1),
              t=(0, -0.05, 0.10 + 0.425 - _B)),
    ]
    return {"parts": parts, "supports": [],
            "half_xy": (w / 2, 0.24), "height": 0.98}


def sample_wallcabinet(rng, w):
    return {"w": w, "d": 0.34, "h": rng.uniform(0.65, 0.75),
            "doors": max(1, int(w / 0.45)), "noise_seed": rng.randint(0, 9999)}


def build_wallcabinet(p):
    w, d, h = p["w"], p["d"], p["h"]
    parts = [_part("body", "wood", beveled_box(w, d, h, bevel=0.006, seg=1),
                   t=(0, 0, h / 2))]
    for i in range(1, p["doors"]):
        x = -w / 2 + w * i / p["doors"]
        parts.append(_part(f"seam{i}", "seam",
                           beveled_box(0.005, 0.004, h - 0.06, bevel=0.001,
                                       seg=1),
                           t=(x, d / 2 - 0.001, h / 2)))
    return {"parts": parts, "supports": [],
            "half_xy": (w / 2, d / 2), "height": h}


_BUILDERS.update({
    "toilet": build_toilet, "vanity": build_vanity, "bathtub": build_bathtub,
    "shower": build_shower, "towel_bar": build_towel_bar,
    "counter": build_counter, "stove": build_stove, "fridge": build_fridge,
    "hood": build_hood, "wallcabinet": build_wallcabinet,
})


# lived-in density: floor decor + hanging pieces. Hanging bodies are natural
# outside-in multi-view targets: small footprint in the under-ceiling air
# band, near-zero covisibility cost.

def sample_floor_decor(rng, kind):
    if kind == "floorplant":
        return {"kind": kind, "pot_r": rng.uniform(0.10, 0.17),
                "h": rng.uniform(0.7, 1.5), "blobs": rng.randint(2, 4),
                "noise_seed": rng.randint(0, 9999)}
    if kind == "basket":
        return {"kind": kind, "r": rng.uniform(0.14, 0.22),
                "h": rng.uniform(0.22, 0.4), "taper": rng.uniform(0.75, 0.9)}
    if kind == "bookstack":
        return {"kind": kind, "n": rng.randint(3, 7),
                "w": rng.uniform(0.16, 0.26), "d": rng.uniform(0.12, 0.20),
                "noise_seed": rng.randint(0, 9999)}
    return {"kind": "suitcase", "w": rng.uniform(0.38, 0.55),
            "d": rng.uniform(0.16, 0.24), "h": rng.uniform(0.5, 0.72)}


def build_floorplant(p):
    import random as _rd
    nr = _rd.Random(p["noise_seed"])
    pr, h = p["pot_r"], p["h"]
    ph = pr * 1.15
    parts = [_part("pot", "ceramic",
                   lathe([(pr * 0.66, 0.0), (pr, ph * 0.85), (pr * 0.92, ph)],
                         n=22), t=(0, 0, 0)),
             _part("trunk", "wood",
                   lathe([(0.014, 0.0), (0.010, h * 0.72)], n=8),
                   t=(0, 0, ph * 0.7))]
    for i in range(p["blobs"]):
        br = pr * nr.uniform(1.15, 1.8)
        bz = ph + (h - ph) * nr.uniform(0.45, 0.95)
        parts.append(_part(
            f"leaf{i}", "foliage",
            _jitter_radial(lathe(
                [(0.0, -br * 0.75), (br * 0.85, -br * 0.25), (br, 0.05),
                 (br * 0.8, br * 0.35), (0.0, br * 0.75)],
                n=nr.randint(14, 18)), seed=nr.randint(0, 9999), amp=0.12),
            t=(nr.uniform(-0.6, 0.6) * pr, nr.uniform(-0.6, 0.6) * pr, bz),
            rot=(nr.uniform(-14, 14), nr.uniform(-14, 14),
                 nr.uniform(0, 360))))
    return {"parts": parts, "supports": [],
            "half_xy": (pr * 1.9, pr * 1.9), "height": h}


def build_basket(p):
    r, h = p["r"], p["h"]
    VF = lathe([(r * p["taper"], 0.0), (r, h * 0.9), (r * 0.97, h),
                (r * 0.88, h), (r * 0.80 * p["taper"], h * 0.12)], n=22)
    return {"parts": [_part("basket", "fabric", VF)],
            "half_xy": (r, r), "height": h, "supports": []}


def build_bookstack(p):
    import random as _rd
    nr = _rd.Random(p["noise_seed"])
    parts, z = [], 0.0
    for i in range(p["n"]):
        t = nr.uniform(0.02, 0.045)
        w = p["w"] * nr.uniform(0.82, 1.0)
        d = p["d"] * nr.uniform(0.82, 1.0)
        parts.append(_part(f"bk{i}", "paper",
                           beveled_box(w, d, t, bevel=0.003, seg=1),
                           t=(nr.uniform(-0.015, 0.015),
                              nr.uniform(-0.015, 0.015), z + t / 2),
                           rot=(0, 0, nr.uniform(-16, 16))))
        z += t - 0.001
    return {"parts": parts, "supports": [],
            "half_xy": (p["w"] * 0.62, p["d"] * 0.62), "height": z}


def build_suitcase(p):
    w, d, h = p["w"], p["d"], p["h"]
    parts = [_part("body", "leather", beveled_box(w, d, h, bevel=0.02, seg=2),
                   t=(0, 0, h / 2)),
             _part("handle", "leather",
                   beveled_box(0.14, 0.02, 0.035, bevel=0.006, seg=1),
                   t=(0, 0, h + 0.012)),
             _part("seam", "seam",
                   beveled_box(w + 0.004, d + 0.004, 0.006, bevel=0.001,
                               seg=1), t=(0, 0, h * 0.62))]
    return {"parts": parts, "supports": [],
            "half_xy": (w / 2, d / 2), "height": h + 0.03}


def sample_hanging(rng, kind):
    if kind == "hangplant":
        return {"kind": kind, "drop": rng.uniform(0.5, 1.05),
                "pot_r": rng.uniform(0.09, 0.14), "blobs": rng.randint(2, 3),
                "noise_seed": rng.randint(0, 9999)}
    if kind == "lantern":
        return {"kind": kind, "drop": rng.uniform(0.35, 0.8),
                "r": rng.uniform(0.10, 0.22),
                "squash": rng.uniform(0.75, 1.15)}
    if kind == "mobile":
        return {"kind": kind, "drop": rng.uniform(0.5, 0.95),
                "arms": rng.randint(2, 3), "noise_seed": rng.randint(0, 9999)}
    return {"kind": "ceilingfan", "rod": rng.uniform(0.12, 0.3),
            "r": rng.uniform(0.45, 0.62), "blades": rng.choice([3, 4, 5]),
            "noise_seed": rng.randint(0, 9999)}


def build_hangplant(p):
    import random as _rd
    nr = _rd.Random(p["noise_seed"])
    drop, pr = p["drop"], p["pot_r"]
    ph = pr * 0.95
    parts = [_part("cable", "metal",
                   lathe([(0.004, 0.0), (0.004, drop - ph - 0.01)], n=6),
                   t=(0, 0, ph + 0.005)),
             _part("pot", "ceramic",
                   lathe([(pr * 0.6, 0.0), (pr, ph * 0.8), (pr * 0.9, ph)],
                         n=20), t=(0, 0, 0))]
    for i in range(p["blobs"]):
        br = pr * nr.uniform(0.9, 1.5)
        parts.append(_part(
            f"leaf{i}", "foliage",
            _jitter_radial(lathe(
                [(0.0, -br * 0.7), (br * 0.85, -br * 0.2), (br, 0.05),
                 (br * 0.75, br * 0.35), (0.0, br * 0.7)],
                n=nr.randint(12, 16)), seed=nr.randint(0, 9999), amp=0.12),
            t=(nr.uniform(-0.5, 0.5) * pr, nr.uniform(-0.5, 0.5) * pr,
               ph + br * nr.uniform(0.1, 0.45)),
            rot=(nr.uniform(-18, 18), nr.uniform(-18, 18),
                 nr.uniform(0, 360))))
    return {"parts": parts, "supports": [],
            "half_xy": (pr * 1.6, pr * 1.6), "height": drop}


def build_lantern(p):
    drop, r, sq = p["drop"], p["r"], p["squash"]
    body_h = 2 * r * sq
    parts = [_part("cable", "metal",
                   lathe([(0.003, 0.0), (0.003, drop - body_h - 0.01)], n=6),
                   t=(0, 0, body_h + 0.005)),
             _part("shade", "paper",
                   lathe([(0.03, 0.0), (r * 0.85, body_h * 0.18),
                          (r, body_h * 0.5), (r * 0.85, body_h * 0.82),
                          (0.03, body_h)], n=26), t=(0, 0, 0)),
             _part("cap", "metal",
                   lathe([(0.035, 0.0), (0.03, 0.025)], n=10),
                   t=(0, 0, body_h - 0.005))]
    return {"parts": parts, "supports": [],
            "half_xy": (r, r), "height": drop}


def build_mobile(p):
    import random as _rd
    nr = _rd.Random(p["noise_seed"])
    drop = p["drop"]
    parts = [_part("cable", "metal",
                   lathe([(0.003, 0.0), (0.003, drop * 0.35)], n=6),
                   t=(0, 0, drop * 0.65))]
    z = drop * 0.65
    half = 0.0
    for a in range(p["arms"]):
        L = nr.uniform(0.30, 0.5)
        ang = nr.uniform(0, 180)
        half = max(half, L / 2 + 0.06)
        parts.append(_part(f"bar{a}", "wood",
                           beveled_box(L, 0.012, 0.012, bevel=0.003, seg=1),
                           t=(0, 0, z), rot=(0, 0, ang)))
        for s in (-1, 1):
            pend_h = nr.uniform(0.06, 0.16)
            px = s * L / 2 * math.cos(math.radians(ang))
            py = s * L / 2 * math.sin(math.radians(ang))
            parts.append(_part(
                f"p{a}{'+' if s > 0 else '-'}", "ceramic",
                lathe([(0.0, 0.0), (nr.uniform(0.018, 0.035), pend_h * 0.4),
                       (0.0, pend_h)], n=nr.randint(8, 12)),
                t=(px, py, z - pend_h - 0.01)))
        z -= nr.uniform(0.10, 0.16)
    return {"parts": parts, "supports": [],
            "half_xy": (half, half), "height": drop}


def build_ceilingfan(p):
    import random as _rd
    nr = _rd.Random(p["noise_seed"])
    rod, r, nb = p["rod"], p["r"], p["blades"]
    hub_h = 0.09
    total = rod + hub_h + 0.02
    parts = [_part("rod", "metal",
                   lathe([(0.012, 0.0), (0.012, rod)], n=10),
                   t=(0, 0, hub_h + 0.01)),
             _part("hub", "metal",
                   lathe([(0.05, 0.0), (0.075, hub_h * 0.5),
                          (0.05, hub_h)], n=16), t=(0, 0, 0.01))]
    for b in range(nb):
        ang = 360.0 * b / nb + nr.uniform(-6, 6)
        parts.append(_part(
            f"blade{b}", "wood",
            beveled_box(r - 0.09, 0.11, 0.012, bevel=0.004, seg=1),
            t=((r - 0.09) / 2 * math.cos(math.radians(ang)) + 0.045,
               (r - 0.09) / 2 * math.sin(math.radians(ang)),
               0.035),
            rot=(0, nr.uniform(-4, -1), ang), pivot=(-0.045, 0, 0)))
    return {"parts": parts, "supports": [],
            "half_xy": (r, r), "height": total}


_BUILDERS.update({
    "floorplant": build_floorplant, "basket": build_basket,
    "bookstack": build_bookstack, "suitcase": build_suitcase,
    "hangplant": build_hangplant, "lantern": build_lantern,
    "mobile": build_mobile, "ceilingfan": build_ceilingfan,
})
