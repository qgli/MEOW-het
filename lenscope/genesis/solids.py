"""Solid realization of a GenesisSpec (bpy-free).

Uniform solid = oriented box {name, kind, c:[3] center, s:[3] half-sizes,
yaw} (yaw about +z, radians); floor/ceiling are polygon prisms
{name, kind, poly, z0, z1}. Apertures are made without booleans: a wall run
with holes is emitted as the solid segments around each hole (the
first-generation doorway splitting, lifted to 3D and to arbitrary 2D
directions).

bpy-free verification: ray_blocked() handles oriented boxes by rotating the
segment into each box frame (slab test). Polygon prisms are excluded from
ray tests by design (aperture checks only concern walls).
"""
from __future__ import annotations

import math

from .spec import GenesisSpec


def _obox(name, kind, cx, cy, cz, hx, hy, hz, yaw=0.0):
    assert hx > 0 and hy > 0 and hz > 0, f"degenerate {name}"
    return {"name": name, "kind": kind, "c": [float(cx), float(cy), float(cz)],
            "s": [float(hx), float(hy), float(hz)], "yaw": float(yaw)}


def _wall_frame(p1, p2):
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    L = math.hypot(dx, dy)
    ux, uy = dx / L, dy / L                    # along-wall unit
    return L, (ux, uy), (-uy, ux), math.atan2(dy, dx)


def _run_box(name, kind, p1, u, nvec, yaw, t, u0, u1, z0, z1, n_off=0.0):
    """Solid segment of an oriented wall run covering u in [u0,u1], z in
    [z0,z1]; n_off shifts along the wall normal (0 = centered on the line)."""
    um = (u0 + u1) / 2
    cx = p1[0] + u[0] * um + nvec[0] * n_off
    cy = p1[1] + u[1] * um + nvec[1] * n_off
    return _obox(name, kind, cx, cy, (z0 + z1) / 2,
                 (u1 - u0) / 2, t / 2, (z1 - z0) / 2, yaw)


def _wall_run(name_prefix, kind, p1, p2, t, h, apertures, n_off=0.0,
              ext_lo=0.0, ext_hi=0.0):
    """Wall run p1->p2 (thickness t, height h) minus apertures
    [(u0,u1,z0,z1)] -> oriented solid boxes (segments + sills + headers).

    ext_lo / ext_hi extend the run past its endpoints (negative values trim
    it); overlaps with neighbouring runs are solid-in-solid. realize() picks
    them per wall end."""
    L, u, nvec, yaw = _wall_frame(p1, p2)
    aps = sorted(apertures)
    last = 0.0
    for (a0, a1, z0, z1) in aps:
        assert -1e-6 <= a0 < a1 <= L + 1e-6, f"{name_prefix}: aperture outside run"
        assert a0 >= last - 1e-6, f"{name_prefix}: overlapping apertures"
        assert 0.0 <= z0 < z1 <= h + 1e-6, f"{name_prefix}: aperture z"
        last = a1
    boxes, i, cur = [], 0, -ext_lo

    def emit(ua, ub, za, zb):
        nonlocal i
        if ub - ua > 1e-6 and zb - za > 1e-6:
            boxes.append(_run_box(f"{name_prefix}_{i:02d}", kind, p1, u, nvec,
                                  yaw, t, ua, ub, za, zb, n_off))
            i += 1
    for (a0, a1, z0, z1) in aps:
        emit(cur, a0, 0.0, h)
        if z0 > 0.0:
            emit(a0, a1, 0.0, z0)          # sill wall below
        if z1 < h:
            emit(a0, a1, z1, h)            # header above
        cur = a1
    emit(cur, L + ext_hi, 0.0, h)
    return boxes


def realize(spec: GenesisSpec) -> dict:
    from . import furniture as _fu
    H = spec.height
    boxes, leaves, polyslabs = [], [], []

    # a slab over the union-hull bbox of the dwelling is wrong for
    # non-rectangular shapes -> floor/ceiling per room polygon (small overlaps
    # at internal walls are harmless solid-in-solid)
    for r in spec.rooms:
        grow = spec.ext_wall_t                  # extend under walls
        poly = r.poly
        polyslabs.append({"name": f"Floor_{r.name[-2:]}", "kind": "floor",
                          "poly": poly, "z0": -0.10, "z1": 0.0, "grow": grow})
        polyslabs.append({"name": f"Ceiling_{r.name[-2:]}", "kind": "ceiling",
                          "poly": poly, "z0": H, "z1": H + 0.10, "grow": grow})

    walls = {w.id: w for w in spec.walls}
    aps_by_wall = {}
    for d in spec.doors:
        aps_by_wall.setdefault(d.wall, []).append(
            (d.u - d.width / 2, d.u + d.width / 2, 0.0, d.height))
    for w in spec.windows:
        aps_by_wall.setdefault(w.wall, []).append(
            (w.u - w.width / 2, w.u + w.width / 2, w.sill, w.sill + w.height))

    for w in spec.walls:
        prefix = ("InternalWall_" if w.internal else "Wall_") + w.id
        # exterior walls sit outside the room line (the composer hands over
        # boundary lines; for exterior CCW rings they are centered on the
        # line shifted t/2 outward via n_off)
        n_off = 0.0 if w.internal else -w.t / 2
        # per-end extension: a collinear touching neighbour run means this is
        # a T-split of the same line -> butt exactly (0; the end faces meet
        # face-to-face, hidden). Otherwise internal ends bury 8mm into the
        # abutting wall (exterior ends: see below), so coplanar duplicate
        # faces cannot occur.
        def _collinear_touch(pt, other):
            dx, dy = w.p2[0] - w.p1[0], w.p2[1] - w.p1[1]
            ox, oy = other.p2[0] - other.p1[0], other.p2[1] - other.p1[1]
            cross = abs(dx * oy - dy * ox)
            if cross > 1e-6 * max(1.0, abs(dx) + abs(dy)):
                return False
            for q in (other.p1, other.p2):
                if abs(q[0] - pt[0]) < 1e-6 and abs(q[1] - pt[1]) < 1e-6:
                    return True
            return False
        # exterior ends are not extended: (a) at reflex/T junctions an
        # extension pokes a wall stub into a room and lands its face coplanar
        # and same-facing with the neighbour's interior face; (b) reflex
        # corners already self-overlap without extensions (outward bands
        # flank each other); (c) convex corners only leave a t*t void outside
        # (interior faces meet exactly at the corner line, invisible from
        # inside). Interior walls keep the 8mm burial into abutting solids.
        base_ext = 0.008 if w.internal else 0.0
        ext_lo, ext_hi = base_ext, base_ext
        for other in spec.walls:
            if other.id == w.id:
                continue
            if _collinear_touch(w.p1, other):
                ext_lo = 0.0
            if _collinear_touch(w.p2, other):
                ext_hi = 0.0
        # miter by trim: at reflex/T junctions a wall's end plane can
        # coincide, same-facing, with a perpendicular exterior wall's interior
        # face (notch-bottom end face == notch-side wall's room plane) ->
        # coplanar duplicate -> mutual shadow blocking. Resolution: trim this
        # end 2cm into the neighbour's body (the vacated sliver is inside that
        # body; the surface then belongs to exactly one wall).
        if not w.internal:
            Lw, uw, nw, _ = _wall_frame(w.p1, w.p2)
            for other in spec.walls:
                if other.internal or other.id == w.id:
                    continue
                Lo, uo, no_, _ = _wall_frame(other.p1, other.p2)
                if abs(uw[0] * uo[0] + uw[1] * uo[1]) > 0.1:
                    continue                      # need ~perpendicular
                for pt, is_lo, end_out in ((w.p1, True, (-uw[0], -uw[1])),
                                           (w.p2, False, (uw[0], uw[1]))):
                    # pt on other's interior plane? (plane through other.p1
                    # with normal no_); same-facing end?
                    dpl = ((pt[0] - other.p1[0]) * no_[0]
                           + (pt[1] - other.p1[1]) * no_[1])
                    proj = ((pt[0] - other.p1[0]) * uo[0]
                            + (pt[1] - other.p1[1]) * uo[1])
                    if (abs(dpl) < 1e-4 and -1e-4 <= proj <= Lo + 1e-4
                            and end_out[0] * no_[0] + end_out[1] * no_[1] > 0.9):
                        if is_lo:
                            ext_lo = -0.02
                        else:
                            ext_hi = -0.02
        boxes += _wall_run(prefix, "wall", w.p1, w.p2, w.t, H,
                           aps_by_wall.get(w.id, []), n_off=n_off,
                           ext_lo=ext_lo, ext_hi=ext_hi)

    # skirting: thin interior strips along every wall base, cut at doors
    for w in spec.walls:
        L, u, nvec, yaw = _wall_frame(w.p1, w.p2)
        door_iv = [(d.u - d.width / 2 - 0.02, d.u + d.width / 2 + 0.02)
                   for d in spec.doors if d.wall == w.id]
        # floor-to-ceiling windows reach below the skirting band; cut the
        # skirting there too (a strip crossing the glass is an artifact)
        door_iv += [(wd.u - wd.width / 2 - 0.02, wd.u + wd.width / 2 + 0.02)
                    for wd in spec.windows
                    if wd.wall == w.id and wd.sill < spec.skirting_h + 0.02]
        segs, cur = [], 0.0
        for a, b in sorted(door_iv):
            if a > cur + 0.05:
                segs.append((cur, a))
            cur = max(cur, b)
        if cur < L - 0.05:
            segs.append((cur, L))
        sides = [+1, -1] if w.internal else [+1]
        base_off = 0.0 if w.internal else -w.t / 2
        for si, sgn in enumerate(sides):
            for sj, (a, b) in enumerate(segs):
                off = base_off + sgn * (w.t / 2 + 0.006)   # back buried 2mm in wall
                cxy = (w.p1[0] + u[0] * (a + b) / 2 + nvec[0] * off,
                       w.p1[1] + u[1] * (a + b) / 2 + nvec[1] * off)
                boxes.append(_obox(f"WallSkirting_{w.id}_{si}{sj:02d}",
                                   "skirting", cxy[0], cxy[1],
                                   spec.skirting_h / 2, (b - a) / 2, 0.008,
                                   spec.skirting_h / 2, yaw))

    # door dressing: hole linings + hinged leaf.
    # Burial rule: dressing elements must never share a plane with wall
    # solids; every coincident plane is buried >=4mm inside a neighbouring
    # solid, and every visible face is proud by >=14mm. Frames coplanar with
    # the wall reveal planes (u=a0/a1, z=height) z-fight in Cycles and render
    # as fat black bars.
    EPS, LIP, PROUD = 0.004, 0.018, 0.028
    for k, d in enumerate(spec.doors):
        w = walls[d.wall]
        L, u, nvec, yaw = _wall_frame(w.p1, w.p2)
        n_off = 0.0 if w.internal else -w.t / 2
        a0, a1 = d.u - d.width / 2, d.u + d.width / 2
        lin_t = w.t + PROUD                      # proud 14mm each side
        dfin = getattr(d, "finish", "white")     # finish -> material
        for tag, ua, ub, za, zb in [
                ("L", a0 - EPS, a0 + LIP, 0.0, d.height + LIP),
                ("R", a1 - LIP, a1 + EPS, 0.0, d.height + LIP),
                ("T", a0 + LIP - EPS, a1 - LIP + EPS, d.height - LIP,
                 d.height + EPS)]:
            boxes.append(_run_box(f"DoorFrame_{k:02d}_{tag}",
                                  f"doorframe:{dfin}",
                                  w.p1, u, nvec, yaw, lin_t,
                                  ua, ub, za, zb, n_off))
        hinge_u = a0 + LIP + EPS                 # clear of the lining face
        hx = w.p1[0] + u[0] * hinge_u + nvec[0] * n_off
        hy = w.p1[1] + u[1] * hinge_u + nvec[1] * n_off
        leaves.append({"name": f"Door_leaf_{k:02d}", "kind": "leaf",
                       "hinge": [hx, hy, 0.0], "wall_yaw": yaw,
                       "width": d.width - 2 * (LIP + EPS) - 0.01,
                       "height": d.height - LIP - 0.01,
                       "thick": d.leaf_thick, "open_deg": d.leaf_open_deg,
                       "entry": d.entry,
                       "style": getattr(d, "style", "flat"),
                       "handle": getattr(d, "handle", "lever"),
                       "finish": dfin})

    # window dressing (burial rule): the hole is clad with linings on all
    # four edges (real window-frame look), glass is inset into the linings,
    # optional mullions, sill board below.
    for k, wd in enumerate(spec.windows):
        w = walls[wd.wall]
        L, u, nvec, yaw = _wall_frame(w.p1, w.p2)
        n_off = 0.0 if w.internal else -w.t / 2
        a0, a1 = wd.u - wd.width / 2, wd.u + wd.width / 2
        z0, z1 = wd.sill, wd.sill + wd.height
        lin_t = w.t + PROUD
        wfin = getattr(wd, "finish", "white")    # finish -> material
        for tag, ua, ub, za, zb in [
                ("L", a0 - EPS, a0 + LIP, z0 - EPS, z1 + EPS),
                ("R", a1 - LIP, a1 + EPS, z0 - EPS, z1 + EPS),
                ("T", a0 + LIP - EPS, a1 - LIP + EPS, z1 - LIP, z1 + EPS),
                ("B", a0 + LIP - EPS, a1 - LIP + EPS, z0 - EPS, z0 + LIP)]:
            boxes.append(_run_box(f"WindowFrame_{k:02d}_{tag}",
                                  f"windowframe:{wfin}",
                                  w.p1, u, nvec, yaw, lin_t,
                                  ua, ub, za, zb, n_off))
        gi = LIP - 0.006                          # glass edge buried in lining
        boxes.append(_run_box(f"WindowGlass_{k:02d}", "glass", w.p1, u, nvec,
                              yaw, wd.glass_thick,
                              a0 + gi, a1 - gi, z0 + gi, z1 - gi, n_off))
        # mullion grid (mull_nx x mull_ny panes); mull_v/mull_h bools when 0
        nx_b = getattr(wd, "mull_nx", 0) or (2 if getattr(wd, "mull_v", False)
                                             else 1)
        ny_b = getattr(wd, "mull_ny", 0) or (2 if getattr(wd, "mull_h", False)
                                             else 1)
        for i in range(1, nx_b):
            um = a0 + (a1 - a0) * i / nx_b
            boxes.append(_run_box(f"WindowFrame_{k:02d}_MV{i}",
                                  f"windowframe:{wfin}",
                                  w.p1, u, nvec, yaw, wd.glass_thick + 0.03,
                                  um - 0.015, um + 0.015, z0 + gi - EPS,
                                  z1 - gi + EPS, n_off))
        for j in range(1, ny_b):
            zm = z0 + (z1 - z0) * j / ny_b
            boxes.append(_run_box(f"WindowFrame_{k:02d}_MH{j}",
                                  f"windowframe:{wfin}",
                                  w.p1, u, nvec, yaw, wd.glass_thick + 0.03,
                                  a0 + gi - EPS, a1 - gi + EPS,
                                  zm - 0.015, zm + 0.015, n_off))
        # interior sill board: proud into the room, top buried in B lining
        boxes.append(_run_box(f"WindowSill_{k:02d}", "window", w.p1, u, nvec,
                              yaw, w.t + PROUD + 0.05, a0 - 0.03, a1 + 0.03,
                              z0 - 0.03, z0 + EPS, n_off))

    # ceiling border rings (suspended-ceiling trim)
    for r in spec.rooms:
        if r.name not in spec.ceiling_border_rooms or len(r.poly) != 4:
            continue
        (x0, y0), (x1, y1) = r.poly[0], r.poly[2]
        drop, bw = 0.12, 0.35
        # butt joints (overlapping corner volumes z-fight) + tops buried 4mm
        # into the ceiling slab (no coplanar top planes).
        zc = (H - drop / 2) + 0.002
        hz = drop / 2 + 0.002                     # top reaches H + 4mm/2
        for si, (cx, cy, hx, hy) in enumerate([
                ((x0 + x1) / 2, y0 + bw / 2, (x1 - x0) / 2, bw / 2),
                ((x0 + x1) / 2, y1 - bw / 2, (x1 - x0) / 2, bw / 2),
                (x0 + bw / 2, (y0 + y1) / 2 + 0.0, bw / 2,
                 (y1 - y0) / 2 - bw),
                (x1 - bw / 2, (y0 + y1) / 2 + 0.0, bw / 2,
                 (y1 - y0) / 2 - bw)]):
            if hx <= 0.01 or hy <= 0.01:
                continue
            boxes.append(_obox(f"CeilingTrim_{r.name[-2:]}_{si}", "ceiling",
                               cx, cy, zc, hx, hy, hz))

    # luminaire geometry: shades are lathe styles from
    # furniture.build_lightshade (cone/dome/drum/globe/multi-head etc.), drawn
    # per lamp in the spec (LightSpec.style). Cords/poles/bases stay simple
    # solids. Light source objects are added by build_scene; lamps that are
    # off keep their geometry, just no emitter.
    lamp_meshes = []

    def _shade(obj, style, t, seed, r=0.16):
        for pt in _fu.build_lightshade(style, r=r, seed=seed):
            V = _fu.transform(pt["verts"], 0.0, t)
            lamp_meshes.append({"obj": obj, "name": f"{obj}.{pt['name']}",
                                "kind": pt["kind"], "verts": V, "item": obj,
                                "faces": pt["faces"], "flags": pt["flags"]})

    for k, L in enumerate(spec.lights):
        style = getattr(L, "style", "") or None
        if L.fixture == "pendant":
            drop = 0.5
            boxes.append(_obox(f"CeilingLight_{k:02d}_cord", "trim",
                               L.x, L.y, H - drop / 2 + 0.002, 0.008, 0.008,
                               drop / 2 + 0.002))   # top buried in slab
            _shade(f"CeilingLight_{k:02d}", style or "drum",
                   (L.x, L.y, H - drop + 0.002), seed=k)
        elif L.fixture == "bulb":
            _shade(f"CeilingLight_{k:02d}", "globe_s", (L.x, L.y, H - 0.05),
                   seed=k)
        elif L.fixture == "wall":
            yaw = math.radians(L.yaw_deg)
            z = L.z if L.z > 0 else 1.75
            boxes.append(_obox(f"WallLamp_{k:02d}_mount", "trim",
                               L.x, L.y, z, 0.035, 0.02, 0.09, yaw))
            boxes.append(_obox(f"WallLamp_{k:02d}", "lightshade",
                               L.x + 0.055 * math.sin(yaw),
                               L.y + 0.055 * math.cos(yaw),
                               z + 0.02, 0.075, 0.075, 0.11, yaw))
        elif L.fixture == "floor":
            z = L.z if L.z > 0 else 1.45
            boxes.append(_obox(f"FloorLamp_{k:02d}_base", "trim",
                               L.x, L.y, 0.015, 0.14, 0.14, 0.015))
            boxes.append(_obox(f"FloorLamp_{k:02d}_pole", "trim",
                               L.x, L.y, z / 2, 0.014, 0.014, z / 2))
            _shade(f"FloorLamp_{k:02d}", style or "cone_f",
                   (L.x, L.y, z + 0.24), seed=k)
        elif L.fixture == "table":
            z = L.z if L.z > 0 else 0.82        # spec sets = anchor top + 0.3x
            boxes.append(_obox(f"TableLamp_{k:02d}_base", "trim",
                               L.x, L.y, z - 0.30, 0.07, 0.07, 0.012))
            boxes.append(_obox(f"TableLamp_{k:02d}_pole", "trim",
                               L.x, L.y, z - 0.16, 0.011, 0.011, 0.13))
            _shade(f"TableLamp_{k:02d}", style or "cone_t",
                   (L.x, L.y, z + 0.10), seed=k)
        else:                                   # flush
            _shade(f"CeilingLight_{k:02d}", style or "disk",
                   (L.x, L.y, H + 0.004), seed=k)    # top buried 4mm in slab

    # exterior backdrop cards: far massing outside window-bearing facades;
    # sits on GroundPlane (z0=-0.02 top)
    for k, B in enumerate(spec.backdrops):
        boxes.append(_obox(f"Backdrop_{B.kind}_{k:02d}", "backdrop",
                           B.x, B.y, B.sz / 2 - 0.02,
                           B.sx / 2, B.sy / 2, B.sz / 2))

    # exterior ground plane: the Nishita sky is pure black below the horizon,
    # so any sight line to the outside pointing down (entry door clearance
    # slits, windows looked at from above) would render black.
    # Top at z=-0.02: buried inside the floor slab [-0.10, 0] under the
    # dwelling (no coplanar pair), visible & lit outside it. Named
    # GroundPlane: not a Floor -> does not affect the fixture grid bounds;
    # z<0.2 -> outside the sampler walking band.
    xs = [pt[0] for r in spec.rooms for pt in r.poly]
    ys = [pt[1] for r in spec.rooms for pt in r.poly]
    gm = 40.0
    boxes.append(_obox("GroundPlane", "ground",
                       (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2,
                       -0.045, (max(xs) - min(xs)) / 2 + gm,
                       (max(ys) - min(ys)) / 2 + gm, 0.025))

    for p in spec.placeholders:            # retired (list is empty)
        boxes.append(_obox(p.name, "placeholder", p.x, p.y, p.sz / 2,
                           p.sx / 2, p.sy / 2, p.sz / 2,
                           math.radians(p.yaw_deg)))

    # furniture + clutter: real triangle meshes (bpy-free kernel; every part
    # watertight/outward, checked by furniture.mesh_checks in tests)
    fmeshes = list(lamp_meshes)          # lathe lamp shades ride this path
    for fs in spec.furniture:
        built = _fu.build_item(fs.ftype, fs.params)
        # burial rule, extended to furniture: floor-standing items sink 3 mm
        # into the floor slab (a leg bottom exactly at z=0 is coplanar with
        # the floor top and renders as a black hole)
        zoff = -0.003 if (fs.z == 0.0 and fs.ftype != "curtain") else 0.0
        for pt in built["parts"]:
            V = _fu.transform(pt["verts"], fs.yaw_deg,
                              (fs.x, fs.y, fs.z + zoff))
            obj = fs.name if pt["obj"] is None else \
                f"{pt['obj']}_{fs.name.split('_', 1)[1]}"
            fmeshes.append({"obj": obj, "name": f"{obj}.{pt['name']}",
                            "kind": pt["kind"], "verts": V, "item": fs.name,
                            "faces": pt["faces"], "flags": pt["flags"]})
    for cs in spec.clutter:
        built = _fu.build_clutter_item(cs.params)
        lean = cs.params.get("lean_deg", 0.0)
        # burial: supported clutter sinks 2 mm into its support surface
        # (bottom coplanar with a tabletop = black-hole risk); floating
        # distractors (parent=='') hang free, no contact to bury
        zoff = -0.002 if cs.parent else 0.0
        for pt in built["parts"]:
            V = pt["verts"]
            if lean:
                # leaning book: tip the top back onto its run (pivot at the
                # trailing bottom edge; 2mm burial into the neighbour)
                V = _fu.transform3(V, (0.0, -lean, 0.0),
                                   pivot=(-cs.params.get("t", 0.03) / 2, 0, 0))
            V = _fu.transform(V, cs.yaw_deg, (cs.x, cs.y, cs.z + zoff))
            fmeshes.append({"obj": cs.name, "name": f"{cs.name}.{pt['name']}",
                            "kind": pt["kind"], "verts": V, "item": cs.name,
                            "faces": pt["faces"], "flags": pt["flags"]})

    return {"boxes": boxes, "leaves": leaves, "polyslabs": polyslabs,
            "fmeshes": fmeshes,
            "lights": spec.lights, "backdrops": spec.backdrops,
            "world": {"sun_elevation_deg": spec.sun_elevation_deg,
                      "sun_azimuth_deg": spec.sun_azimuth_deg,
                      "sun_intensity": spec.sun_intensity,
                      "sky_strength": spec.sky_strength,
                      "dust_density": spec.dust_density,
                      "sky_archetype": spec.sky_archetype,
                      "efficacy_lm_w": spec.efficacy_lm_w,
                      "film_exposure": spec.film_exposure,
                      "moon": dict(spec.moon),
                      "height": spec.height}}


# bpy-free verification (oriented boxes)

def ray_blocked(boxes, p0, p1) -> bool:
    import numpy as np
    o0 = np.asarray(p0, float); e0 = np.asarray(p1, float)
    for b in boxes:
        c = np.asarray(b["c"]); s = np.asarray(b["s"])
        yaw = b.get("yaw", 0.0)
        ca, sa = math.cos(-yaw), math.sin(-yaw)
        def to_frame(p):
            d = p - c
            return np.array([ca * d[0] - sa * d[1],
                             sa * d[0] + ca * d[1], d[2]])
        o, e = to_frame(o0), to_frame(e0)
        dvec = e - o
        t0, t1, ok = 0.0, 1.0, True
        for ax in range(3):
            if abs(dvec[ax]) < 1e-12:
                if abs(o[ax]) >= s[ax]:
                    ok = False; break
            else:
                ta = (-s[ax] - o[ax]) / dvec[ax]
                tb = (s[ax] - o[ax]) / dvec[ax]
                if ta > tb:
                    ta, tb = tb, ta
                t0, t1 = max(t0, ta), min(t1, tb)
                if t0 >= t1:
                    ok = False; break
        if ok:
            return True
    return False
