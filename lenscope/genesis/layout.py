"""Room furnishing for second-generation scenes (bpy-free; called during spec construction).

All randomness executes here with the rng the spec hands in, and every
decision lands in FurnitureSpec/ClutterSpec entries inside the manifest;
`furniture.build_item(ftype, params)` is deterministic, so a manifest fully
reconstructs the scene.

Placement legality:
- against-wall items sit flush to a wall segment, yaw facing the interior;
- OBB-vs-OBB separation (SAT) between all furniture footprints;
- door-swing clearance: keep-out circle (radius = leaf width + 0.15) at every
  door position on this room's boundary — furniture never blocks a door;
- coverage cap: furniture footprints <= ~45% of room area (walkable rooms);
- clutter: contact z = support z, footprint inside the support poly, no
  2D overlap, stacking depth 2 (smallbox tops only);
- bookshelf shelves get book runs (occupancy 0.3-0.9), last book of a run
  leans 8-14 deg (buried 2 mm into its neighbour, per the burial rule).
"""
from __future__ import annotations

import math

from . import furniture as fu


# geometry

def _obb_corners(x, y, hx, hy, yaw_deg):
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    return [(x + c * dx * hx - s * dy * hy, y + s * dx * hx + c * dy * hy)
            for dx, dy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]


def _obb_overlap(a, b, margin=0.02):
    """SAT for two OBBs given as (x, y, hx, hy, yaw)."""
    ca, cb = _obb_corners(*a), _obb_corners(*b)
    for poly, other in ((ca, cb), (cb, ca)):
        for i in range(4):
            ex = poly[(i + 1) % 4][0] - poly[i][0]
            ey = poly[(i + 1) % 4][1] - poly[i][1]
            ax, ay = -ey, ex
            L = math.hypot(ax, ay)
            if L < 1e-12:
                continue
            ax, ay = ax / L, ay / L
            pa = [ax * px + ay * py for px, py in poly]
            pb = [ax * px + ay * py for px, py in other]
            if max(pa) + margin < min(pb) or max(pb) + margin < min(pa):
                return False
    return True


def _in_poly(x, y, poly):
    inside = False
    j = len(poly) - 1
    for i in range(len(poly)):
        xi, yi = poly[i]; xj, yj = poly[j]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi:
            inside = not inside
        j = i
    return inside


def _obb_inside_room(obb, poly, margin=0.03):
    return all(_in_poly(px, py, poly) or _dist_to_poly(px, py, poly) < margin
               for px, py in _obb_corners(*obb)) and \
        all(_in_poly(px, py, poly) or True for px, py in [_obb_center(obb)])


def _obb_center(obb):
    return (obb[0], obb[1])


def _dist_to_poly(x, y, poly):
    best = float("inf")
    for i in range(len(poly)):
        ax, ay = poly[i]
        bx, by = poly[(i + 1) % len(poly)]
        abx, aby = bx - ax, by - ay
        den = max(abx * abx + aby * aby, 1e-9)
        t = min(max(((x - ax) * abx + (y - ay) * aby) / den, 0.0), 1.0)
        best = min(best, math.hypot(x - (ax + t * abx), y - (ay + t * aby)))
    return best


def _room_edges(poly):
    """Room boundary edges with inward normal (poly is CCW -> inward=left)."""
    out = []
    for i in range(len(poly)):
        ax, ay = poly[i]
        bx, by = poly[(i + 1) % len(poly)]
        L = math.hypot(bx - ax, by - ay)
        if L < 0.8:
            continue
        dx, dy = (bx - ax) / L, (by - ay) / L
        out.append({"a": (ax, ay), "b": (bx, by), "d": (dx, dy), "L": L,
                    "n_in": (-dy, dx)})
    return out


def _edge_of_point(edges, x, y):
    """Index of the room edge nearest to (x, y) + the along-edge coordinate."""
    best = (None, 1e9, 0.0)
    for i, e in enumerate(edges):
        ax, ay = e["a"]
        dx, dy = e["d"]
        u = min(max((x - ax) * dx + (y - ay) * dy, 0.0), e["L"])
        px, py = ax + dx * u, ay + dy * u
        d = math.hypot(x - px, y - py)
        if d < best[1]:
            best = (i, d, u)
    return best[0], best[2]


class WallBoard:
    """One registry for everything hanging on a room's walls (curtains,
    mirrors, paintings, posters, clocks), so that e.g. a mirror and a
    painting never land on the same spot."""

    def __init__(self, edges):
        self.edges = edges
        self.iv = {}                          # edge_idx -> [(u0,u1,z0,z1)]

    def free(self, ei, u0, u1, z0, z1, margin=0.06):
        for (a, b, c, d) in self.iv.get(ei, []):
            if u0 - margin < b and a < u1 + margin \
                    and z0 - 0.02 < d and c < z1 + 0.02:
                return False
        return True

    def claim(self, ei, u0, u1, z0, z1):
        self.iv.setdefault(ei, []).append((u0, u1, z0, z1))


class RoomPlanner:
    """Tracks occupancy + keep-outs for one room while the spec composes it."""

    def __init__(self, room, door_pts, rng, walls=None):
        self.room = room
        self.poly = room.poly
        self.rng = rng
        self.edges = _room_edges(room.poly)
        self.obbs = []                       # placed footprints
        self.keepouts = [(x, y, r) for (x, y, r) in door_pts]
        self.area_used = 0.0
        self.area_cap = 0.45 * room.area
        # room polys are boundary lines; internal walls are centered on them
        # (solids.py n_off=0), so the wall face protrudes t/2 (4-7.5cm) into
        # the room past the poly; gapping against the poly line would embed
        # beds/sofas/wardrobes t/2-0.02 into every internal wall. The inset is
        # queried per placement point, not per edge: rect-union shapes have
        # edges that are part internal, part external (an edge-level match can
        # pick the exterior run with the larger overlap and return 0).
        self.walls = walls or []

    def _wall_inset_at(self, e, u, walls=None):
        """Protrusion of the wall face past the poly line at edge point u."""
        walls = self.walls if walls is None else walls
        px = e["a"][0] + e["d"][0] * u
        py = e["a"][1] + e["d"][1] * u
        best = 0.0
        for w in walls:
            wdx, wdy = w.p2[0] - w.p1[0], w.p2[1] - w.p1[1]
            wl = math.hypot(wdx, wdy)
            if wl < 1e-6:
                continue
            wux, wuy = wdx / wl, wdy / wl
            if abs(wux * e["d"][0] + wuy * e["d"][1]) < 0.98:
                continue                       # not parallel to this edge
            dperp = abs((px - w.p1[0]) * (-wuy) + (py - w.p1[1]) * wux)
            if dperp > w.t / 2 + 0.03:
                continue
            uw = (px - w.p1[0]) * wux + (py - w.p1[1]) * wuy
            if -0.05 <= uw <= wl + 0.05:
                best = max(best, (w.t / 2) if w.internal else 0.0)
        return best

    def _legal(self, obb, wall_margin=-0.005, keepout_scale=1.0):
        x, y, hx, hy, yaw = obb
        if self.area_used + 4 * hx * hy > self.area_cap:
            return False
        for px, py in _obb_corners(*obb) + [(x, y)]:
            # corners may sit up to 8cm outside: wall-flush items with a
            # negative back gap (burial rule) legitimately poke into the wall
            if not _in_poly(px, py, self.poly) \
                    and _dist_to_poly(px, py, self.poly) > 0.08:
                return False
        for (kx, ky, kr) in self.keepouts:
            # conservative: keep-out circle vs footprint circumradius.
            # keepout_scale < 1 is for low counter-height pieces (kitchen
            # runs): a worktop 0.4-0.7m from a door edge is how real small
            # kitchens work, and pose sampling runs after furnishing against
            # the real scene, so camera legality is decided there, not here.
            kr = kr * keepout_scale
            if math.hypot(x - kx, y - ky) < kr + math.hypot(hx, hy):
                if _point_obb_dist(kx, ky, obb) < kr:
                    return False
        for o in self.obbs:
            if _obb_overlap(obb, o):
                return False
        # catch-all: internal runs protrude t/2 past the poly line; sub-0.8m
        # notch stubs are invisible to _room_edges; and companion items
        # (nightstands/dining chairs via pl.at) can poke through the corner
        # allowance into exterior wall bodies too. Reject any footprint
        # penetrating any wall run deeper than 2cm (wall art legitimately
        # buries 1.0-1.2cm by design).
        for w in self.walls:
            wdx, wdy = w.p2[0] - w.p1[0], w.p2[1] - w.p1[1]
            wl = math.hypot(wdx, wdy)
            if wl < 1e-6:
                continue
            # solids.py convention: internal runs are centered on the line;
            # exterior runs offset outward by t/2 (n_off = -t/2 along the
            # inward-left normal). Mirror it or the phantom inner half would
            # reject every legitimately flush exterior-wall placement.
            n_in_x, n_in_y = -wdy / wl, wdx / wl
            off = 0.0 if w.internal else -w.t / 2
            wobb = ((w.p1[0] + w.p2[0]) / 2 + n_in_x * off,
                    (w.p1[1] + w.p2[1]) / 2 + n_in_y * off,
                    wl / 2, w.t / 2,
                    math.degrees(math.atan2(wdy, wdx)))
            if _obb_overlap(obb, wobb, margin=wall_margin):
                return False
        return True

    def _commit(self, obb):
        self.obbs.append(obb)
        self.area_used += 4 * obb[2] * obb[3]

    def against_wall(self, hx, hy, tries=40, edge_filter=None, back_gap=0.02,
                     avoid_win=None, u_hint=None, validate=None,
                     keepout_scale=1.0):
        """Place with the item's -y (local back) against a wall; returns
        (x, y, yaw_deg, edge) or None.

        avoid_win = {edge_idx: [(u0,u1),...]} window intervals; a candidate
        whose wall span intersects one (margin 0.15) is rejected.
        u_hint = (edge_idx, u_center, spread): bias placement near a point
        (e.g. the TV at the sofa's line-of-sight hit)."""
        edges = [(i, e) for i, e in enumerate(self.edges)
                 if e["L"] >= 2 * hx + 0.2 and (edge_filter is None
                                                or edge_filter(e))]
        if u_hint is not None:
            edges = [(i, e) for (i, e) in edges if i == u_hint[0]]
        if not edges:
            return None
        for _ in range(tries):
            ei, e = self.rng.choice(edges)
            if u_hint is not None:
                u = min(max(u_hint[1] + self.rng.uniform(-u_hint[2], u_hint[2]),
                            hx + 0.1), e["L"] - hx - 0.1)
            else:
                u = self.rng.uniform(hx + 0.1, e["L"] - hx - 0.1)
            if avoid_win is not None:
                bad = False
                for (a, b) in avoid_win.get(ei, []):
                    if u - hx - 0.15 < b and a < u + hx + 0.15:
                        bad = True
                        break
                if bad:
                    continue
            nx, ny = e["n_in"]
            off_in = hy + back_gap + self._wall_inset_at(e, u)  # clear
            cx = e["a"][0] + e["d"][0] * u + nx * off_in   # the actual wall
            cy = e["a"][1] + e["d"][1] * u + ny * off_in   # face, not the poly
            # convention: furniture meshes rotate by R_z(+yaw)
            # (furniture.transform), so the local +y front maps to
            # (-sin yaw, cos yaw). Solving front == n_in gives
            # yaw = atan2(-nx, ny); atan2(nx, ny) would map fronts to (-nx, ny),
            # correct on N/S walls but exactly reversed on E/W walls.
            yaw = math.degrees(math.atan2(-nx, ny))   # mesh front (+y) -> inward
            obb = (cx, cy, hx, hy, yaw)
            wm = -0.02 if back_gap < 0 else -0.005   # designed burial vs hard
            if self._legal(obb, wall_margin=wm, keepout_scale=keepout_scale):
                if validate is not None and not validate(cx, cy, yaw, e):
                    continue                      # soft rule veto -> retry
                self._commit(obb)
                return cx, cy, yaw, e
        return None

    def ray_edge(self, x, y, dx, dy):
        """Which room edge does the ray from (x,y) along (dx,dy)
        hit first? Returns (edge_idx, u_at_hit, distance) or None."""
        best = None
        for i, e in enumerate(self.edges):
            ax, ay = e["a"]
            ex, ey = e["d"][0] * e["L"], e["d"][1] * e["L"]
            den = dx * (-ey) - dy * (-ex)
            if abs(den) < 1e-9:
                continue
            t = ((ax - x) * (-ey) - (ay - y) * (-ex)) / den
            # Cramer's rule: s divides by det, not -det
            s = (dx * (ay - y) - dy * (ax - x)) / den
            if t > 0.15 and -1e-6 <= s <= 1.0 + 1e-6:
                if best is None or t < best[2]:
                    best = (i, s * e["L"], t)
        return best

    def free_interior(self, hx, hy, tries=40, center_bias=True,
                      keepout_scale=1.0, validate=None):
        cx0 = sum(p[0] for p in self.poly) / len(self.poly)
        cy0 = sum(p[1] for p in self.poly) / len(self.poly)
        for t in range(tries):
            if center_bias and t < 10:
                x = cx0 + self.rng.uniform(-0.5, 0.5)
                y = cy0 + self.rng.uniform(-0.5, 0.5)
            else:
                xs = [p[0] for p in self.poly]
                ys = [p[1] for p in self.poly]
                x = self.rng.uniform(min(xs) + hx, max(xs) - hx)
                y = self.rng.uniform(min(ys) + hy, max(ys) - hy)
            yaw = self.rng.choice([0.0, 90.0]) if center_bias \
                else self.rng.uniform(0, 360)
            obb = (x, y, hx, hy, yaw)
            if self._legal(obb, keepout_scale=keepout_scale):
                if validate is not None and not validate(x, y, yaw, None):
                    continue                    # zone / soft-rule veto
                self._commit(obb)
                return x, y, yaw
        return None

    def at(self, x, y, hx, hy, yaw, commit=True, keepout_scale=1.0):
        obb = (x, y, hx, hy, yaw)
        if self._legal(obb, keepout_scale=keepout_scale):
            if commit:
                self._commit(obb)
            return True
        return False


def _point_obb_dist(px, py, obb):
    x, y, hx, hy, yaw = obb
    c, s = math.cos(math.radians(-yaw)), math.sin(math.radians(-yaw))
    lx = c * (px - x) - s * (py - y)
    ly = s * (px - x) + c * (py - y)
    dx = max(abs(lx) - hx, 0.0)
    dy = max(abs(ly) - hy, 0.0)
    return math.hypot(dx, dy)


# schedule

def _studio_zones(rng, room, door_pts):
    """One big room -> 2-4 virtual zones (open-plan studio).
    Designer logic: the sleep zone goes farthest from the nearest door,
    living takes the largest remainder, kitchenette/dining/study fill in.
    Zones are soft rectangles used only as placement-center validators --
    collision, wall legality and pose sampling stay global."""
    xs = [p[0] for p in room.poly]
    ys = [p[1] for p in room.poly]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    n = 2 if room.area < 24 else (3 if room.area < 36
                                  else rng.choice([3, 4]))
    t = rng.uniform(0.42, 0.58)
    if (x1 - x0) >= (y1 - y0):
        cells = [(x0, y0, x0 + (x1 - x0) * t, y1),
                 (x0 + (x1 - x0) * t, y0, x1, y1)]
    else:
        cells = [(x0, y0, x1, y0 + (y1 - y0) * t),
                 (x0, y0 + (y1 - y0) * t, x1, y1)]
    while len(cells) < n:
        cells.sort(key=lambda c: -(c[2] - c[0]) * (c[3] - c[1]))
        cx0, cy0, cx1, cy1 = cells.pop(0)
        s = rng.uniform(0.4, 0.6)
        if (cx1 - cx0) >= (cy1 - cy0):
            m = cx0 + (cx1 - cx0) * s
            cells += [(cx0, cy0, m, cy1), (m, cy0, cx1, cy1)]
        else:
            m = cy0 + (cy1 - cy0) * s
            cells += [(cx0, cy0, cx1, m), (cx0, m, cx1, cy1)]

    def _dmin(c):
        cx, cy = (c[0] + c[2]) / 2, (c[1] + c[3]) / 2
        return min((math.hypot(cx - dx, cy - dy)
                    for (dx, dy, _r) in door_pts), default=0.0)

    cells.sort(key=_dmin, reverse=True)          # sleep = most private
    if len(cells) > 2:                           # living = largest remainder
        rest = cells[1:]
        big = max(range(len(rest)),
                  key=lambda i: (rest[i][2] - rest[i][0])
                  * (rest[i][3] - rest[i][1]))
        rest[0], rest[big] = rest[big], rest[0]
        cells = cells[:1] + rest
    funcs = ["bedroom", "living"]
    if room.area >= 20 and rng.random() < 0.75:
        funcs.append("kitchen")
    fill = ["dining", "study"]
    rng.shuffle(fill)
    funcs += fill
    return list(zip(funcs, cells))


def furnish_room(rng, room, door_pts, windows_here, new_item, walls=None,
                 ceil_pts=None, room_h=2.7):
    """Furnish one room by function (aesthetic rules on top of the physical
    legality). Returns (planner, wallboard, stats). new_item(...)
    records a FurnitureSpec. windows_here = [(wall_id, cx, cy, w, h, sill,
    yaw_in, nx, ny, half_wall_thickness)]."""
    pl = RoomPlanner(room, door_pts, rng, walls=walls)
    board = WallBoard(pl.edges)
    f = room.function
    stats = {"solid_wall_fallbacks": 0}
    # window intervals per room edge (window-avoid rules) + curtain claims on the board
    win_iv = {}
    win_edge = {}
    for (wid, cx, cy, ww, wh, sill, wyaw, nx, ny, _th) in windows_here:
        ei, u = _edge_of_point(pl.edges, cx, cy)
        win_iv.setdefault(ei, []).append((u - ww / 2, u + ww / 2))
        win_edge[wid] = (ei, u)

    def try_wall(ftype, params, solid_back=False, tall=False, **kw):
        b = fu.build_item(ftype, params)
        got = None
        if solid_back or tall:
            got = pl.against_wall(b["half_xy"][0], b["half_xy"][1],
                                  avoid_win=win_iv, tries=90, **kw)
            if got is None and solid_back and not tall:
                stats["solid_wall_fallbacks"] += 1     # declared solid-wall fallback
                got = pl.against_wall(b["half_xy"][0], b["half_xy"][1], **kw)
        else:
            got = pl.against_wall(b["half_xy"][0], b["half_xy"][1], **kw)
        if got is None:
            return None
        x, y, yaw, _e = got
        return new_item(ftype, params, x, y, yaw), (x, y, yaw, b)

    def try_free(ftype, params, **kw):
        b = fu.build_item(ftype, params)
        got = pl.free_interior(b["half_xy"][0], b["half_xy"][1], **kw)
        if got is None:
            return None
        x, y, yaw = got
        return new_item(ftype, params, x, y, yaw), (x, y, yaw, b)

    def place_rug(cx, cy, params):
        """Rugs are walkable anchors: no OBB claim, placed under the
        functional group (not squeezed into empty corners)."""
        b = fu.build_item("rug", params)
        x = cx + rng.uniform(-0.15, 0.15)
        y = cy + rng.uniform(-0.15, 0.15)
        if _in_poly(x, y, room.poly):
            new_item("rug", params, x, y, rng.choice([0.0, 90.0]))

    # function blocks below are closures so a "studio" room can run several
    # of them, each soft-constrained to a virtual zone rect.
    # Collision/wall legality stay global (one planner, one board); a zone
    # only vetoes candidate centers, so nothing about downstream legality
    # changes for single-function rooms (zone=None -> validator pass-through).
    state = {"bed": None}
    room_tables = []                   # (x, y) of placed tables (reading chairs)

    def _chair_ok(zone):
        """Reading chairs: free placement, but a chair within 1.5m of a
        table must not turn its back on it (front = local +y)."""
        def _ok(x, y, yaw, _e):
            if zone is not None and not (zone[0] <= x <= zone[2]
                                         and zone[1] <= y <= zone[3]):
                return False
            c2 = math.cos(math.radians(yaw))
            s2 = math.sin(math.radians(yaw))
            for (tx2, ty2) in room_tables:
                dx2, dy2 = tx2 - x, ty2 - y
                L2 = math.hypot(dx2, dy2)
                if 1e-6 < L2 <= 1.5 \
                        and (-s2) * dx2 / L2 + c2 * dy2 / L2 <= 0.5:
                    return False
            return True
        return _ok

    def _zv(zone, extra=None):
        if zone is None and extra is None:
            return None
        def _v(cx, cy, yaw, e):
            if zone is not None and not (zone[0] <= cx <= zone[2]
                                         and zone[1] <= cy <= zone[3]):
                return False
            return extra(cx, cy, yaw, e) if extra is not None else True
        return _v

    def _f_bedroom(zone=None):
        bp = fu.sample_bed(rng)
        bbb = fu.build_item("bed", bp)

        def _bed_ok(cx, cy, yaw, _e):
            """Designer rules (soft; declared fallback below):
            (1) the bed must not directly face a door (privacy / don't wake
            to an open corridor); (2) night path: >=0.45m clear on at least
            one long side (walk-out space)."""
            c2 = math.cos(math.radians(yaw))
            s2 = math.sin(math.radians(yaw))
            fx, fy = -s2, c2
            for (kx, ky, _kr) in self_keepouts:
                vx, vy = kx - cx, ky - cy
                ahead = vx * fx + vy * fy
                perp = abs(-fy * vx + fx * vy)
                if ahead > 0.3 and perp < bbb["half_xy"][0] + 0.25:
                    return False                     # rule 1: faces a door
            ok_sides = 0
            for sgn in (-1, 1):
                px = cx + sgn * c2 * (bbb["half_xy"][0] + 0.28)
                py = cy + sgn * s2 * (bbb["half_xy"][0] + 0.28)
                if _in_poly(px, py, pl.poly) and all(
                        _point_obb_dist(px, py, o) > 0.15 for o in pl.obbs):
                    ok_sides += 1
            return ok_sides >= 1                     # rule 2: a walkable side

        self_keepouts = list(pl.keepouts)
        bed = try_wall("bed", bp, solid_back=True,
                       validate=_zv(zone, _bed_ok))      # solid wall + rules 1-2 (+zone)
        if bed is None:
            stats["bed_soft_rule_fallbacks"] = \
                stats.get("bed_soft_rule_fallbacks", 0) + 1
            bed = try_wall("bed", bp, solid_back=True, validate=_zv(zone))
        if bed is None and zone is not None:
            bed = try_wall("bed", bp, solid_back=True)   # rather a bed than none
        if bed is not None:
            _, (bx, by, byaw, bb) = bed
            c, s = math.cos(math.radians(byaw)), math.sin(math.radians(byaw))
            for sgn in (-1, 1):
                nsp = fu.sample_nightstand(rng)
                nb = fu.build_item("nightstand", nsp)
                lx = sgn * (bb["half_xy"][0] + nb["half_xy"][0] + 0.06)
                ly = -(bb["half_xy"][1] - nb["half_xy"][1])
                nx = bx + c * lx - s * ly
                ny = by + s * lx + c * ly
                if pl.at(nx, ny, nb["half_xy"][0], nb["half_xy"][1], byaw):
                    new_item("nightstand", nsp, nx, ny, byaw)
            if rng.random() < 0.6:
                # rug beside the bed (walk-out side), not in a random void
                side = rng.choice([-1, 1])
                rx = bx + c * side * (bb["half_xy"][0] + 0.5)
                ry = by + s * side * (bb["half_xy"][0] + 0.5)
                place_rug(rx, ry, fu.sample_rug(rng))
        if rng.random() < 0.8:
            try_wall("wardrobe", fu.sample_wardrobe(rng), tall=True,
                     validate=_zv(zone))                  # avoids windows
        # second-function pieces: a desk corner or a reading chair make
        # bedrooms read lived-in, not staged
        if rng.random() < 0.4:
            dk = try_wall("table", fu.sample_table(rng, "desk"),
                          validate=_zv(zone))
            if dk is not None:
                _, (dx2, dy2, dyaw2, db2) = dk
                room_tables.append((dx2, dy2))
                ch2 = fu.sample_chair(rng)
                chb2 = fu.build_item("chair", ch2)
                ly2 = db2["half_xy"][1] + chb2["half_xy"][1] + 0.10
                c3 = math.cos(math.radians(dyaw2))
                s3 = math.sin(math.radians(dyaw2))
                if pl.at(dx2 - s3 * ly2, dy2 + c3 * ly2,
                         chb2["half_xy"][0], chb2["half_xy"][1],
                         dyaw2 + 180.0):
                    new_item("chair", ch2, dx2 - s3 * ly2, dy2 + c3 * ly2,
                             dyaw2 + 180.0)
        elif rng.random() < 0.35:
            try_free("chair", fu.sample_chair(rng), center_bias=False,
                     validate=_chair_ok(zone))
        state["bed"] = bed

    def _f_living(zone=None):
        sofa = try_wall("sofa", fu.sample_sofa(rng), solid_back=True,
                        validate=_zv(zone))               # solid wall
        if sofa is not None:
            _, (sx, sy, syaw, sb) = sofa
            c, s = math.cos(math.radians(syaw)), math.sin(math.radians(syaw))
            face = (-s, c)                     # local +y in world
            ct = fu.sample_table(rng, "coffee")
            cb = fu.build_item("table", ct)
            gap = rng.uniform(0.45, 0.7)
            ly = sb["half_xy"][1] + cb["half_xy"][1] + gap
            tx, ty = sx + face[0] * ly, sy + face[1] * ly
            table_ok = pl.at(tx, ty, cb["half_xy"][0], cb["half_xy"][1], syaw)
            if table_ok:
                new_item("table", ct, tx, ty, syaw)
                room_tables.append((tx, ty))
            # the TV goes on the wall the sofa looks at (2D ray hit); an
            # inward-normal filter is fooled by L-shape notch segments
            # (corner TVs, same-side TVs)
            hit = pl.ray_edge(sx, sy, face[0], face[1])
            if hit is not None and hit[2] > 1.8:
                tvp = fu.sample_tv_stand(rng)
                # no good spot on the sight line -> no TV (a TV backed onto
                # a window is the worst placement; tall=True rides the
                # no-fallback window-avoid path)
                try_wall("tv_stand", tvp, tall=True,
                         u_hint=(hit[0], hit[1], 1.0))
            if rng.random() < 0.7:
                # rug under the coffee-table zone (sofa-group anchor)
                place_rug(sx + face[0] * (ly * 0.9),
                          sy + face[1] * (ly * 0.9), fu.sample_rug(rng))
        if rng.random() < 0.7:
            try_wall("bookshelf", fu.sample_bookshelf(rng), tall=True,
                     validate=_zv(zone))                  # avoids windows
        # sideboard / reading chair as second-function pieces
        if rng.random() < 0.45:
            sb = fu.sample_nightstand(rng)
            sb["w"] = sb["w"] * rng.uniform(1.8, 2.6)     # sideboard
            try_wall("nightstand", sb, validate=_zv(zone))
        if rng.random() < 0.4:
            try_free("chair", fu.sample_chair(rng), center_bias=False,
                     validate=_chair_ok(zone))

    def _f_dining(zone=None):
        tab = try_free("table", fu.sample_table(rng, "dining"),
                       validate=_zv(zone))
        if tab is not None:
            _, (tx, ty, tyaw, tb) = tab
            room_tables.append((tx, ty))
            c, s = math.cos(math.radians(tyaw)), math.sin(math.radians(tyaw))
            n_ch = rng.choice([4, 4, 6])
            slots = [(-1, 0), (1, 0), (0, -1), (0, 1), (-0.5, -1), (0.5, 1)]
            for (ux, uy) in slots[:n_ch]:
                ch = fu.sample_chair(rng)
                chb = fu.build_item("chair", ch)
                lx = ux * (tb["half_xy"][0] + chb["half_xy"][1] + 0.12)
                ly = uy * (tb["half_xy"][1] + chb["half_xy"][1] + 0.12)
                wx = tx + c * lx - s * ly
                wy = ty + s * lx + c * ly
                cyaw = math.degrees(math.atan2(tx - wx, wy - ty)) + 180.0
                if pl.at(wx, wy, chb["half_xy"][0], chb["half_xy"][1], cyaw):
                    new_item("chair", ch, wx, wy, cyaw)
        # dining sideboard
        if rng.random() < 0.5:
            sb = fu.sample_nightstand(rng)
            sb["w"] = sb["w"] * rng.uniform(1.8, 2.6)
            try_wall("nightstand", sb, validate=_zv(zone))

    def _f_study(zone=None):
        desk = try_wall("table", fu.sample_table(rng, "desk"),
                        validate=_zv(zone))
        if desk is not None:
            _, (dx, dy, dyaw, db) = desk
            room_tables.append((dx, dy))
            c, s = math.cos(math.radians(dyaw)), math.sin(math.radians(dyaw))
            ch = fu.sample_chair(rng)
            chb = fu.build_item("chair", ch)
            ly = db["half_xy"][1] + chb["half_xy"][1] + 0.10
            wx, wy = dx - s * ly, dy + c * ly
            if pl.at(wx, wy, chb["half_xy"][0], chb["half_xy"][1],
                     dyaw + 180.0):
                new_item("chair", ch, wx, wy, dyaw + 180.0)
        if rng.random() < 0.8:
            try_wall("bookshelf", fu.sample_bookshelf(rng), tall=True,
                     validate=_zv(zone))                  # avoids windows

    def _f_hallway(zone=None):
        if rng.random() < 0.4:
            p = fu.sample_nightstand(rng)
            p["w"] = p["w"] * 1.6                     # console proportions
            try_wall("nightstand", p, validate=_zv(zone))

    def _f_bathroom(zone=None):
        # toilet + vanity against walls; tub or shower on the longest wall
        # that fits; towel bar on the WallBoard. Clearances ride
        # the standard machinery (against_wall inset + _legal wall collision
        # + door keep-outs).
        try_wall("toilet", fu.sample_toilet(rng))
        van = try_wall("vanity", fu.sample_vanity(rng))
        if van is not None and rng.random() < 0.8:
            _, (vx, vy, vyaw, vb) = van
            mp = fu.sample_mirror(rng)
            mp["w"] = min(mp["w"], vb["half_xy"][0] * 2 * 0.9)
            ei, u = _edge_of_point(pl.edges, vx, vy)
            z = rng.uniform(1.35, 1.5)
            mb = fu.build_item("mirror", mp)
            if board.free(ei, u - mp["w"] / 2, u + mp["w"] / 2,
                          z - mp["h"] / 2, z + mp["h"] / 2):
                board.claim(ei, u - mp["w"] / 2, u + mp["w"] / 2,
                            z - mp["h"] / 2, z + mp["h"] / 2)
                bx2 = vx - math.sin(math.radians(vyaw)) * (-vb["half_xy"][1])
                by2 = vy + math.cos(math.radians(vyaw)) * (-vb["half_xy"][1])
                new_item("mirror", mp, bx2, by2, vyaw, z=z)
        if rng.random() < 0.5:
            if try_wall("bathtub", fu.sample_bathtub(rng)) is None:
                try_wall("shower", fu.sample_shower(rng))  # tub didn't fit
        else:
            try_wall("shower", fu.sample_shower(rng))
        # towel bar: wall-mounted at z, on any free wall stretch
        tp = fu.sample_towel_bar(rng)
        gotb = pl.against_wall(tp["w"] / 2, 0.06, back_gap=-0.010)
        if gotb is not None:
            x, y, yaw, e = gotb
            z = rng.uniform(1.1, 1.3)
            ei, u = _edge_of_point(pl.edges, x, y)
            if board.free(ei, u - tp["w"] / 2, u + tp["w"] / 2,
                          z - 0.35, z + 0.05):
                board.claim(ei, u - tp["w"] / 2, u + tp["w"] / 2,
                            z - 0.35, z + 0.05)
                new_item("towel_bar", tp, x, y, yaw, z=z)

    def _f_kitchen(zone=None):
        # assign_functions deliberately hands the kitchen an ~11 m2 room
        # (longest wall 3.3-3.7m), too short for a fixed 4-segment straight
        # run (~4.0-5.0m of clear wall). Real small kitchens are compact or
        # L-shaped: greedy-pack segments by priority (sink counter mandatory,
        # stove next, fridge/filler optional), spill leftovers around the
        # corner, degrade to a kitchenette. Two phases: dry-run per edge,
        # then commit only a set containing the sink counter
        # (kitchen-defining, no orphan segments).
        segs = {
            "sink": ("counter", dict(fu.sample_counter(
                rng, rng.uniform(1.0, 1.4)), with_sink=True)),
            "stove": ("stove", fu.sample_stove(rng)),
            "fridge": ("fridge", fu.sample_fridge(rng)),
            "fill": ("counter", fu.sample_counter(rng,
                                                  rng.uniform(0.6, 1.1))),
        }
        built_segs = {k: fu.build_item(n2, dict(p2))
                      for k, (n2, p2) in segs.items()}

        def _win_clash(ei2, u0, u1, z0, z1):
            """True if [u0,u1]x[z0,z1] on edge ei2 overlaps a window pane.
            Tall fridges / hoods / wall cabinets must not ride the glass; a
            low counter under the window is fine (and realistic: sinks sit
            under kitchen windows in most real homes)."""
            for (wid3, _wx3, _wy3, ww3, wh3, sill3, _w4, _n3, _n4, _t3) \
                    in windows_here:
                ei3, u3 = win_edge[wid3]
                if ei3 == ei2 and u0 < u3 + ww3 / 2 \
                        and u3 - ww3 / 2 < u1 and z0 < sill3 + wh3 \
                        and sill3 < z1:
                    return True
            return False

        def _plan_run(ei2, order, u_start):
            """Greedy contiguous dry-run along edge ei2 from u_start."""
            e2 = pl.edges[ei2]
            nx2, ny2 = e2["n_in"]
            yaw2 = math.degrees(math.atan2(-nx2, ny2))
            plan, cur = [], u_start
            for key in order:
                name2, p2 = segs[key]
                b2 = built_segs[key]
                hw, hyy = b2["half_xy"][0], b2["half_xy"][1]
                if cur + 2 * hw > e2["L"] - 0.10:
                    continue                      # no room lengthwise
                umid = cur + hw
                if key == "fridge" and _win_clash(
                        ei2, umid - hw, umid + hw, 0.0, b2["height"]):
                    continue                      # tall fridge blocks glass
                inset = pl._wall_inset_at(e2, umid)
                cx = e2["a"][0] + e2["d"][0] * umid \
                    + nx2 * (hyy + 0.02 + inset)
                cy = e2["a"][1] + e2["d"][1] * umid \
                    + ny2 * (hyy + 0.02 + inset)
                if zone is not None and not (zone[0] <= cx <= zone[2]
                                             and zone[1] <= cy <= zone[3]):
                    cur += 2 * hw + 0.02      # keep the kitchenette in its
                    continue                  # studio corner, stay contiguous
                ks = 1.0 if key == "fridge" else 0.65   # low worktops may
                if pl.at(cx, cy, hw, hyy, yaw2, commit=False,  # near a door
                         keepout_scale=ks):
                    plan.append((key, name2, p2, cx, cy, hw, hyy, umid))
                    cur += 2 * hw + 0.02
            return plan, yaw2

        edges_sorted = sorted(range(len(pl.edges)),
                              key=lambda i: -pl.edges[i]["L"])
        order = ["sink", "stove", "fridge", "fill"]
        best = None                               # (score, plan, yaw, ei)
        for ei in edges_sorted:
            e = pl.edges[ei]
            u_start = rng.uniform(0.12, min(0.5, max(0.13, e["L"] - 2.2)))
            plan, yaw = _plan_run(ei, order, u_start)
            keys = {p[0] for p in plan}
            if "sink" not in keys:
                continue
            score = len(plan) + (1 if "stove" in keys else 0)
            if best is None or score > best[0]:
                best = (score, plan, yaw, ei)
            if score >= 5:
                break                             # full straight run found
        placed = {}                               # key -> (ei,cx,cy,umid,hw)
        runs = []                                 # [(ei, u_lo, u_hi, yaw)]
        if best is not None:
            _, plan, yaw, ei = best
            for (key, name2, p2, cx, cy, hw, hyy, umid) in plan:
                if pl.at(cx, cy, hw, hyy, yaw,
                         keepout_scale=1.0 if key == "fridge" else 0.65):
                    new_item(name2, p2, cx, cy, yaw)
                    placed[key] = (ei, cx, cy, umid, hw, yaw)
            got_run = [placed[k] for k in placed if placed[k][0] == ei]
            if got_run:
                runs.append((ei,
                             min(g[3] - g[4] for g in got_run),
                             max(g[3] + g[4] for g in got_run), yaw))
        # L-run: spill missing core segments around the corner (the classic
        # small-kitchen L). Adjacent edges see the committed main run, so
        # the corner overlap is collision-checked, not hand-geometried.
        leftovers = [k for k in ("stove", "fridge") if k not in placed]
        if best is not None and leftovers:
            for adj in ((best[3] + 1) % len(pl.edges),
                        (best[3] - 1) % len(pl.edges)):
                if not leftovers:
                    break
                for u_start in (0.72, 1.0, 1.3):
                    plan2, yaw2 = _plan_run(adj, list(leftovers), u_start)
                    if not plan2:
                        continue
                    lo = hi = None
                    for (key, name2, p2, cx, cy, hw, hyy, umid) in plan2:
                        if pl.at(cx, cy, hw, hyy, yaw2,
                                 keepout_scale=1.0 if key == "fridge"
                                 else 0.65):
                            new_item(name2, p2, cx, cy, yaw2)
                            placed[key] = (adj, cx, cy, umid, hw, yaw2)
                            leftovers.remove(key)
                            lo = umid - hw if lo is None else min(lo,
                                                                  umid - hw)
                            hi = umid + hw if hi is None else max(hi,
                                                                  umid + hw)
                    if lo is not None:
                        runs.append((adj, lo, hi, yaw2))
                        break
        # kitchenette as last resort, never an empty kitchen: any wall spot
        # for the sink counter via the standard machinery.
        if "sink" not in placed:
            stats["kitchen_run_fallbacks"] = \
                stats.get("kitchen_run_fallbacks", 0) + 1
            got_k = try_wall("counter", segs["sink"][1], keepout_scale=0.65)
            if got_k is None:
                # four-door crossroads room: a free-standing island counter
                # (real kitchens do this) rather than an empty room
                ip = dict(fu.sample_counter(rng, rng.uniform(0.9, 1.2)),
                          with_sink=True)
                ib = fu.build_item("counter", ip)
                giv = pl.free_interior(ib["half_xy"][0], ib["half_xy"][1],
                                       tries=60, keepout_scale=0.75,
                                       validate=_zv(zone))
                if giv is not None:
                    new_item("counter", ip, giv[0], giv[1], giv[2])
        if "fridge" not in placed and rng.random() < 0.9:
            try_wall("fridge", segs["fridge"][1], tall=True)  # own wall spot
        stats["kitchen_mode"] = ("full" if len(placed) >= 4 else
                                 "L" if len({p[0] for p in placed.values()})
                                 > 1 else
                                 "compact" if "sink" in placed else
                                 "kitchenette")
        # uppers: hood above the stove, wall cabinets along the runs (both
        # window-aware and WallBoard-claimed)
        if "stove" in placed:
            sei, scx, scy, sumid, shw, syaw = placed["stove"]
            e2 = pl.edges[sei]
            nx2, ny2 = e2["n_in"]
            hp = fu.sample_hood(rng)
            hz = 1.55
            if not _win_clash(sei, sumid - hp["w"] / 2, sumid + hp["w"] / 2,
                              hz, hz + 0.98) \
                    and board.free(sei, sumid - hp["w"] / 2,
                                   sumid + hp["w"] / 2, hz, hz + 0.98):
                board.claim(sei, sumid - hp["w"] / 2, sumid + hp["w"] / 2,
                            hz, hz + 0.98)
                new_item("hood", hp, scx - nx2 * 0.06, scy - ny2 * 0.06,
                         syaw, z=hz)
        # tall-segment intervals (a 1.8m fridge overlaps the 1.45m cabinet
        # band, a real clash; cabinets must skip the fridge's wall span)
        tall_iv = {}
        if "fridge" in placed:
            fei, _fx, _fy, fumid, fhw, _fy2 = placed["fridge"]
            tall_iv.setdefault(fei, []).append((fumid - fhw - 0.03,
                                                fumid + fhw + 0.03))
        for (rei, u_lo, u_hi, ryaw) in runs:
            e2 = pl.edges[rei]
            nx2, ny2 = e2["n_in"]
            u_c = u_lo
            for _ in range(3):                    # denser uppers
                wc = fu.sample_wallcabinet(rng, rng.uniform(0.6, 1.1))
                if u_c + wc["w"] > min(u_hi + 0.3, e2["L"] - 0.1):
                    break
                if any(u_c < b2 and a2 < u_c + wc["w"]
                       for (a2, b2) in tall_iv.get(rei, [])):
                    u_c += wc["w"] * 0.5 + 0.15   # slide past the fridge
                    continue
                if not _win_clash(rei, u_c, u_c + wc["w"], 1.45,
                                  1.45 + wc["h"]) \
                        and board.free(rei, u_c, u_c + wc["w"], 1.45,
                                       1.45 + wc["h"]):
                    board.claim(rei, u_c, u_c + wc["w"], 1.45,
                                1.45 + wc["h"])
                    u_mid = u_c + wc["w"] / 2
                    inset = pl._wall_inset_at(e2, u_mid)
                    wx2 = e2["a"][0] + e2["d"][0] * u_mid \
                        + nx2 * (wc["d"] / 2 + 0.005 + inset)
                    wy2 = e2["a"][1] + e2["d"][1] * u_mid \
                        + ny2 * (wc["d"] / 2 + 0.005 + inset)
                    new_item("wallcabinet", wc, wx2, wy2, ryaw, z=1.45)
                u_c += wc["w"] + rng.uniform(0.06, 0.4)
        if zone is None and rng.random() < 0.55:
            bt = try_free("table", fu.sample_table(rng, "coffee"))
            if bt is not None:                    # breakfast spot + chairs
                _, (tx, ty, tyaw, tb) = bt
                room_tables.append((tx, ty))
                c2 = math.cos(math.radians(tyaw))
                s2 = math.sin(math.radians(tyaw))
                for (ux, uy) in ((-1, 0), (1, 0))[:rng.randint(1, 2)]:
                    ch = fu.sample_chair(rng)
                    chb = fu.build_item("chair", ch)
                    lx = ux * (tb["half_xy"][0] + chb["half_xy"][1] + 0.10)
                    ly = uy * (tb["half_xy"][1] + chb["half_xy"][1] + 0.10)
                    wx = tx + c2 * lx - s2 * ly
                    wy = ty + s2 * lx + c2 * ly
                    cyaw = math.degrees(math.atan2(tx - wx, wy - ty)) + 180.0
                    if pl.at(wx, wy, chb["half_xy"][0], chb["half_xy"][1],
                             cyaw):
                        new_item("chair", ch, wx, wy, cyaw)

    _dispatch = {"bedroom": _f_bedroom, "living": _f_living,
                 "dining": _f_dining, "study": _f_study,
                 "hallway": _f_hallway, "bathroom": _f_bathroom,
                 "kitchen": _f_kitchen}
    if f == "studio":
        # a single big room is an open-plan studio: sleep/living/
        # kitchenette/dining zones coexist, randomized per seed, instead of
        # one mono-function furniture set.
        zs = _studio_zones(rng, room, door_pts)
        stats["studio_zones"] = [(zf, tuple(round(v, 2) for v in rect))
                                 for zf, rect in zs]
        for zf, rect in zs:
            _dispatch[zf](zone=rect)
    elif f in _dispatch:
        _dispatch[f]()

    # wall-mounted extras: curtains (registered on the WallBoard)
    for (wall_id, wx, wy, ww, wh, sill, wyaw, nx, ny, t_half) in windows_here:
        if f in ("living", "bedroom", "studio") and rng.random() < 0.55:
            cp = fu.sample_curtain(rng, ww, wh, sill)
            ei, u = win_edge[wall_id]
            # the curtain span must stay inside its wall segment (a wide
            # curtain drawn near a corner would cut into the perpendicular
            # wall). Clamp on the built half-width (double
            # curtains add a panel gap beyond w/2), scaling w and gap together;
            # windows are edge-contained, so this never under-covers the glass.
            allowed_half = min(u, pl.edges[ei]["L"] - u) - 0.03
            built_half = fu.build_item("curtain", cp)["half_xy"][0]
            if built_half > allowed_half > 0:
                scale = max(allowed_half / built_half,
                            ww / max(cp["w"], 1e-6))
                cp["w"] *= scale
                if "gap" in cp:
                    cp["gap"] *= scale
            # window centres sit on the wall run centerline (t/2 outside the
            # room line on exterior walls), so the offset includes t_half
            # (amp+0.05 alone would hang half the pleats inside the wall).
            off = t_half + 0.02 + cp["amp"]
            if board.free(ei, u - cp["w"] / 2, u + cp["w"] / 2,
                          0.06, cp["drop"]):
                board.claim(ei, u - cp["w"] / 2, u + cp["w"] / 2,
                            0.06, cp["drop"])
                new_item("curtain", cp, wx + nx * off, wy + ny * off, wyaw,
                         z=0.0, wall_id=wall_id)
    if f in ("bedroom", "hallway", "studio") and rng.random() < 0.3:
        mp = fu.sample_mirror(rng)
        mb = fu.build_item("mirror", mp)

        def _mirror_ok(cx, cy, yaw, _e):
            # a bedroom mirror must not face the bed (reflection rule)
            bed = state["bed"]
            if bed is None:
                return True
            _bx, _by = bed[1][0], bed[1][1]
            c2 = math.cos(math.radians(yaw))
            s2 = math.sin(math.radians(yaw))
            vx, vy = _bx - cx, _by - cy
            ahead = vx * (-s2) + vy * c2
            perp = abs(-c2 * vx - s2 * vy)
            bd = math.hypot(bed[1][3]["half_xy"][0], bed[1][3]["half_xy"][1])
            return not (ahead > 0.2 and perp < bd)

        got = pl.against_wall(mb["half_xy"][0], 0.05, back_gap=-0.012,
                              validate=_mirror_ok)
        if got is not None:
            x, y, yaw, e = got
            z = rng.uniform(1.1, 1.4)
            ei, u = _edge_of_point(pl.edges, x, y)
            if board.free(ei, u - mp["w"] / 2, u + mp["w"] / 2,
                          z - mp["h"] / 2, z + mp["h"] / 2):
                board.claim(ei, u - mp["w"] / 2, u + mp["w"] / 2,
                            z - mp["h"] / 2, z + mp["h"] / 2)
                new_item("mirror", mp, x, y, yaw, z=z)

    # lived-in floor decor. Small footprints ride the standard legality
    # machinery (keepouts, wall runs, collisions); covisibility is decided
    # by pose sampling downstream.
    if f not in ("bathroom", "hallway"):
        n_floor = max(1, min(6, int(room.area / 5.0)))
        for _ in range(n_floor):
            kind = rng.choice(["floorplant", "floorplant", "basket",
                               "bookstack", "suitcase"])
            if kind == "suitcase" and f not in ("bedroom", "studio"):
                continue
            p = fu.sample_floor_decor(rng, kind)
            b = fu.build_item(kind, p)
            got = pl.free_interior(b["half_xy"][0], b["half_xy"][1],
                                   tries=25, center_bias=False)
            if got is not None:
                new_item(kind, p, got[0], got[1], got[2])
    # hanging pieces in the under-ceiling air band: each is a natural
    # outside-in multi-view target. No floor
    # OBB is claimed (they occupy air); legality = inside the room poly,
    # clear of walls/curtains (0.45m), ceiling lamps (0.6m), tall furniture
    # (unless the body hangs above 2.35m), and each other.
    if f != "bathroom":
        n_hang = rng.choices([0, 1, 2, 3], weights=[22, 42, 26, 10])[0]
        lamps2 = list(ceil_pts or [])
        hung = []
        xs0 = [q[0] for q in pl.poly]
        ys0 = [q[1] for q in pl.poly]
        for _ in range(n_hang * 6):
            if len(hung) >= n_hang:
                break
            kind = rng.choices(
                ["hangplant", "lantern", "mobile", "ceilingfan"],
                weights=[38, 30, 22, 10])[0]
            if kind == "ceilingfan" and any(k == "ceilingfan"
                                            for k, _x, _y in hung):
                continue                          # one fan per room, max
            p = fu.sample_hanging(rng, kind)
            b = fu.build_item(kind, p)
            hr = b["half_xy"][0]
            x = rng.uniform(min(xs0) + hr + 0.45, max(xs0) - hr - 0.45)
            y = rng.uniform(min(ys0) + hr + 0.45, max(ys0) - hr - 0.45)
            if not _in_poly(x, y, pl.poly) \
                    or _dist_to_poly(x, y, pl.poly) < hr + 0.45:
                continue
            z_bot = room_h - b["height"] - 0.02
            if z_bot < 1.55:
                continue                          # keep heads clear
            if any(math.hypot(x - lx, y - ly) < hr + 0.6
                   for (lx, ly) in lamps2):
                continue
            if any(math.hypot(x - hx2, y - hy2) < hr + 0.55
                   for (_k, hx2, hy2) in hung):
                continue
            if z_bot < 2.35 and any(
                    _point_obb_dist(x, y, o) < hr + 0.30 for o in pl.obbs):
                continue
            new_item(kind, p, x, y, rng.uniform(0, 360), z=round(z_bot, 3))
            hung.append((kind, x, y))
        if hung:
            stats["hanging"] = [k for k, _x, _y in hung]
    return pl, board, stats


# clutter

def clutter_supports(rng, fitems, lamp_pts, new_clutter):
    """Populate every registered support surface. fitems = list of
    (FurnitureSpec, built). lamp_pts = [(x, y)] table-lamp bases to avoid.
    new_clutter(kind_params, parent, x, y, z, yaw, stack_on) records one."""
    for fs, built in fitems:
        c = math.cos(math.radians(fs.yaw_deg))
        s = math.sin(math.radians(fs.yaw_deg))

        def to_world(lx, ly):
            return fs.x + c * lx - s * ly, fs.y + s * lx + c * ly

        for si, sup in enumerate(built["supports"]):
            poly = sup["poly"]
            xs = [p[0] for p in poly]
            ys = [p[1] for p in poly]
            if sup.get("shelf"):
                _fill_shelf(rng, fs, sup, si, to_world, new_clutter)
                continue
            # area-driven count (a dining table takes more than a nightstand
            # top) with a lower bound, since sparse surfaces read as a staged
            # show flat. Soft surfaces (bed foot, sofa seats) cap lower and
            # restrict to bed-plausible kinds.
            occupied = []                    # (lx, ly, r)
            occx = sup.get("occupied_x")
            soft = sup.get("soft", False)
            area = max(0.01, (max(xs) - min(xs)) * (max(ys) - min(ys)))
            if soft:
                n_items = rng.randint(1, 4)
            else:
                n_items = max(3, min(12, int(area / 0.045
                                             * rng.uniform(0.7, 1.4))))
            kitcheny = fs.ftype == "counter"
            placed = 0
            for _ in range(n_items * 6):
                if placed >= n_items:
                    break
                if soft:
                    p = fu.sample_clutter_item(rng, rng.choice(
                        ["flatbook", "tray", "laptop", "smallbox", "phone",
                         "book"]))
                elif kitcheny:
                    p = fu.sample_clutter_item(rng, rng.choice(
                        ["plate", "bowl", "pan", "bottle", "cup", "fruit",
                         "smallbox", "tray"]))
                else:
                    p = fu.sample_clutter_item(rng)
                it = fu.build_clutter_item(p)
                r = it["footprint_r"]
                lx = rng.uniform(min(xs) + r, max(xs) - r) \
                    if max(xs) - min(xs) > 2 * r else None
                ly = rng.uniform(min(ys) + r, max(ys) - r) \
                    if max(ys) - min(ys) > 2 * r else None
                if lx is None or ly is None:
                    continue
                if occx and occx[0] - r < lx < occx[1] + r:
                    continue                  # e.g. TV footprint on tv_stand
                if any(math.hypot(lx - ox, ly - oy) < r + orr + 0.01
                       for ox, oy, orr in occupied):
                    continue
                wx, wy = to_world(lx, ly)
                if any(math.hypot(wx - px, wy - py) < r + 0.13
                       for px, py in lamp_pts):
                    continue
                name = new_clutter(p, fs.name, wx, wy, sup["z"],
                                   rng.uniform(0, 360), "")
                occupied.append((lx, ly, r))
                placed += 1
                # stacking (depth 2): a smallbox top may host one item
                if p["kind"] == "smallbox" and rng.random() < 0.4:
                    p2 = fu.sample_clutter_item(
                        rng, rng.choice(["book", "cup", "figurine"]))
                    it2 = fu.build_clutter_item(p2)
                    if it2["footprint_r"] <= it["top_r"]:
                        new_clutter(p2, fs.name, wx, wy,
                                    sup["z"] + it["height"] - 0.002,
                                    rng.uniform(0, 360), name)


def _fill_shelf(rng, fs, sup, si, to_world, new_clutter):
    """Book runs with occupancy 0.3-0.9 + a leaning last book per run."""
    poly = sup["poly"]
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    span = max(xs) - min(xs)
    clear_h = sup.get("clear_h", 0.35)
    occ = rng.uniform(0.3, 0.9)
    cursor = min(xs) + 0.01
    budget = span * occ
    ly = (min(ys) + max(ys)) / 2
    while budget > 0.03 and cursor < max(xs) - 0.05:
        run_n = rng.randint(3, 10)
        run_books = []
        for _ in range(run_n):
            p = fu.sample_clutter_item(rng, "book")
            p["h"] = min(p["h"], clear_h - 0.015)
            if cursor + p["t"] > max(xs) - 0.01 or budget - p["t"] < 0:
                break
            run_books.append((p, cursor + p["t"] / 2))
            cursor += p["t"]
            budget -= p["t"]
        for bi, (p, lx) in enumerate(run_books):
            lean = 0.0
            if bi == len(run_books) - 1 and len(run_books) > 2 \
                    and rng.random() < 0.7:
                lean = rng.uniform(8.0, 14.0)     # leans onto the run
                p = dict(p, lean_deg=lean)
            wx, wy = to_world(lx, ly)
            new_clutter(p, fs.name, wx, wy, sup["z"],
                        fs.yaw_deg, "")
        cursor += rng.uniform(0.02, 0.12)         # gap between runs
    # occasional non-book item in the remaining gap
    if rng.random() < 0.4 and max(xs) - cursor > 0.12:
        p = fu.sample_clutter_item(rng, rng.choice(["vase", "figurine",
                                                    "smallbox"]))
        it = fu.build_clutter_item(p)
        if it["height"] <= clear_h - 0.02 and it["footprint_r"] * 2 < \
                (max(ys) - min(ys)):
            lx = rng.uniform(cursor + it["footprint_r"],
                             max(xs) - it["footprint_r"])
            wx, wy = to_world(lx, ly)
            new_clutter(p, fs.name, wx, wy, sup["z"], rng.uniform(0, 360), "")


def place_wall_art(rng, room, spec, door_pts, win_pts, new_item, board=None,
                   walls=None):
    """Paintings/posters/clocks: wall-mounted at eye band, avoiding tall
    furniture (>1.2m) near the wall, doors (full-height swing) and windows.
    Every claim goes through the room's shared WallBoard, so art cannot land
    on a mirror or a curtain."""
    pl = RoomPlanner(room, list(door_pts), rng, walls=walls)
    if board is None:
        board = WallBoard(pl.edges)
    for (wx, wy, ww) in win_pts:
        pl.keepouts.append((wx, wy, ww / 2 + 0.3))
    for fs in spec.furniture:
        if fs.room == room.name and fs.height > 1.2 \
                and fs.ftype not in ("curtain",):
            pl.obbs.append((fs.x, fs.y, fs.hx, fs.hy, fs.yaw_deg))
    pl.area_cap = 1e9                     # wall art occupies no floor
    from . import furniture as fu
    plan = {"living": [("painting", 0.75), ("painting", 0.45),
                       ("poster", 0.25), ("clock", 0.35)],
            "bedroom": [("painting", 0.55), ("painting", 0.3),
                        ("poster", 0.25)],
            "dining": [("painting", 0.6), ("clock", 0.3)],
            "study": [("poster", 0.4), ("painting", 0.3), ("clock", 0.3)],
            "hallway": [("painting", 0.4), ("clock", 0.25)],
            # studio: gallery mix of everything
            "studio": [("painting", 0.75), ("painting", 0.4),
                       ("poster", 0.3), ("clock", 0.35)]}.get(
                room.function, [])
    for (kind, p) in plan:
        if rng.random() >= p:
            continue
        params = (fu.sample_painting(rng) if kind == "painting"
                  else fu.sample_poster(rng) if kind == "poster"
                  else fu.sample_clock(rng))
        b = fu.build_item(kind, params)
        got = pl.against_wall(b["half_xy"][0], 0.05, back_gap=-0.010)
        if got is None:
            continue
        x, y, yaw, _e = got
        z = rng.uniform(1.5, 1.85) if kind == "clock" \
            else rng.uniform(1.25, 1.55)
        w2 = b["half_xy"][0]
        h2 = b["height"] / 2
        ei, u = _edge_of_point(pl.edges, x, y)
        if not board.free(ei, u - w2, u + w2, z - h2, z + h2):
            continue                          # never stack wall items
        board.claim(ei, u - w2, u + w2, z - h2, z + h2)
        new_item(kind, params, x, y, yaw, z=z)
