"""GenesisSpec: bpy-free description of a second-generation scene.

One seed -> one complete, serializable, reproducible scene description. All
random decisions happen here; the bpy side (build_scene) only translates.

Floor plans: variable shapes (rect/L/U/T/circle/pentagon + BSP apartments
with an optional hallway corridor), a connectivity guarantee
(max-shared-span spanning tree -> every tree edge gets a door), functional
room typing, skirting/ceiling-border details, a per-room light schedule with
fixture/CCT/lumens randomness, and sky/sun randomness.

Geometry schema: rooms are axis-aligned rect unions (shapes decompose to
rects; BSP is rects) or a single polygon studio (circle/pentagon). Walls are
first-class runs {p1,p2,t,internal}; doors/windows reference wall ids with a
1D coordinate u along p1->p2.
"""
from __future__ import annotations

import json
import math
import random
import zlib
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import floorplan as fp

SPEC_VERSION = "ge3.0"


@dataclass
class RoomSpec:
    name: str
    function: str
    poly: list                      # [[x,y], ...] CCW (rect rooms: 4 pts)

    @property
    def area(self) -> float:
        s = 0.0
        for i in range(len(self.poly)):
            x0, y0 = self.poly[i]
            x1, y1 = self.poly[(i + 1) % len(self.poly)]
            s += x0 * y1 - x1 * y0
        return abs(s) / 2


@dataclass
class WallRun:
    id: str
    p1: list                        # [x,y]
    p2: list
    t: float
    internal: bool


@dataclass
class DoorSpec:
    wall: str
    u: float                        # center along p1->p2 (m)
    width: float = 0.9
    height: float = 2.05
    leaf_open_deg: float = 30.0
    leaf_thick: float = 0.04
    frame_w: float = 0.06
    entry: bool = False             # entry doors render with a ~closed leaf
    # aesthetics (hash-derived, no main-stream draws):
    style: str = "flat"             # flat | panel2 | panel4 | grooved
    handle: str = "lever"           # lever | knob
    # leaf/frame finish: painted colors or wood grain; the handle renders in
    # a dedicated metal material either way
    finish: str = "white"           # white | wood | sage | graphite | navy


@dataclass
class WindowSpec:
    wall: str
    u: float
    width: float = 1.2
    height: float = 1.3
    sill: float = 0.9
    frame_w: float = 0.05
    glass_thick: float = 0.01
    mull_v: bool = False            # vertical mullion bar (window sash look)
    mull_h: bool = False
    # mullion grid (nx x ny panes; 0 = use mull_v/mull_h). Hash-derived.
    mull_nx: int = 0
    mull_ny: int = 0
    # frame finish (white PVC / dark aluminium / wood)
    finish: str = "white"           # white | dark_alu | wood


@dataclass
class LightSpec:
    room: str
    fixture: str                    # flush | pendant | bulb | wall | floor | table
    x: float; y: float
    lumens: float
    cct_k: float
    z: float = -1.0                 # emitter height; -1 = default height per fixture type
    on: bool = True                 # off lamps keep geometry, emit nothing
    yaw_deg: float = 0.0            # wall lamps: shade faces this way (into room)
    style: str = ""                 # lathe shade style (see furniture)


@dataclass
class BackdropSpec:
    """Exterior card: cheap procedural massing outside window-bearing
    facades (parallax + non-sky window content)."""
    kind: str                       # block | tree
    x: float; y: float
    sx: float; sy: float; sz: float
    gray: float = 0.3               # desaturated albedo


@dataclass
class Placeholder:
    name: str
    x: float; y: float
    sx: float; sy: float; sz: float
    yaw_deg: float = 0.0


@dataclass
class FurnitureSpec:
    """One furnished item. params fully determine the geometry via
    furniture.build_item (deterministic), so the manifest reconstructs it."""
    name: str
    ftype: str
    room: str
    x: float; y: float
    yaw_deg: float = 0.0
    z: float = 0.0                  # wall-mounted items (mirror/curtain)
    wall_id: str = ""
    hx: float = 0.0; hy: float = 0.0
    height: float = 0.0
    params: dict = field(default_factory=dict)


@dataclass
class ClutterSpec:
    """One clutter item on a support surface (or a floating distractor
    when parent == '')."""
    name: str
    parent: str
    x: float; y: float; z: float
    yaw_deg: float = 0.0
    stack_on: str = ""
    params: dict = field(default_factory=dict)


@dataclass
class GenesisSpec:
    seed: int
    version: str = SPEC_VERSION
    kind: str = "bsp"               # rect|l_shape|u_shape|t_shape|circle|pentagon|bsp|bsp_hallway|ge0
    height: float = 2.7
    ext_wall_t: float = 0.20
    int_wall_t: float = 0.12
    skirting_h: float = 0.08
    rooms: list = field(default_factory=list)
    walls: list = field(default_factory=list)
    doors: list = field(default_factory=list)
    windows: list = field(default_factory=list)
    lights: list = field(default_factory=list)
    placeholders: list = field(default_factory=list)   # retired (kept for schema compatibility)
    furniture: list = field(default_factory=list)      # [FurnitureSpec]
    clutter: list = field(default_factory=list)        # [ClutterSpec]
    backdrops: list = field(default_factory=list)
    ceiling_border_rooms: list = field(default_factory=list)
    sky_archetype: str = "day"      # day | dusk | night
    sun_elevation_deg: float = 35.0
    sun_azimuth_deg: float = 120.0
    sun_intensity: float = 1.0      # Nishita sun strength multiplier
    sky_strength: float = 1.0       # world background strength (window brightness)
    dust_density: float = 0.0       # Nishita aerosol (dusk warmth/haze)
    efficacy_lm_w: float = 160.0    # lm per radiant watt (unit bridge fixed in the spec)
    film_exposure: float = 1.0      # derived: see exposure_derivation
    exposure_target: float = 0.35   # probe-servo mid-gray target (per scene)
    exposure_derivation: dict = field(default_factory=dict)   # derivation record
    solar: dict = field(default_factory=dict)   # astronomical sun record
    moon: dict = field(default_factory=dict)    # moonlight (night)
    materials: dict = field(default_factory=dict)   # material provenance: key->entry
    material_image_prob: float = 0.0            # image-vs-procedural mix
    assets_lib: dict = field(default_factory=dict)  # asset library snapshot (counts)
    layout_stats: dict = field(default_factory=dict)  # layout fallback counts (solid-wall rule)
    emit_linear_exr: bool = False   # default off; interface reserved

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=1)

    def save(self, path: Path) -> None:
        Path(path).write_text(self.to_json())

    @staticmethod
    def from_json(s: str) -> "GenesisSpec":
        d = json.loads(s)
        sp = GenesisSpec(seed=d["seed"])
        for k, v in d.items():
            if k in ("rooms", "walls", "doors", "windows", "lights",
                     "placeholders", "backdrops", "furniture", "clutter"):
                continue
            setattr(sp, k, v)
        sp.rooms = [RoomSpec(**r) for r in d["rooms"]]
        sp.walls = [WallRun(**w) for w in d["walls"]]
        sp.doors = [DoorSpec(**x) for x in d["doors"]]
        sp.windows = [WindowSpec(**x) for x in d["windows"]]
        sp.lights = [LightSpec(**x) for x in d["lights"]]
        sp.placeholders = [Placeholder(**x) for x in d["placeholders"]]
        sp.furniture = [FurnitureSpec(**x) for x in d.get("furniture", [])]
        sp.clutter = [ClutterSpec(**x) for x in d.get("clutter", [])]
        sp.backdrops = [BackdropSpec(**x) for x in d.get("backdrops", [])]
        return sp


# composer helpers (rect-union dwellings)

def _sub_intervals(lo, hi, covered):
    """[lo,hi] minus covered [(a,b),...] -> remaining intervals."""
    out, cur = [], lo
    for a, b in sorted(covered):
        if a > cur + 1e-9:
            out.append((cur, min(a, hi)))
        cur = max(cur, b)
    if cur < hi - 1e-9:
        out.append((cur, hi))
    return [(a, b) for a, b in out if b - a > 1e-6]


def _exterior_runs(rects, edges):
    """Per rect edge, subtract shared spans -> exterior wall runs.
    Returns [(rect_idx, axis, c, s0, s1, outward_sign)]."""
    shared = {}
    for (i, j, axis, c, s0, s1) in edges:
        for k in (i, j):
            shared.setdefault((k, axis, round(c, 6)), []).append((s0, s1))
    runs = []
    for k, (x0, y0, x1, y1) in enumerate(rects):
        for axis, c, lo, hi, sign in [("x", x0, y0, y1, -1), ("x", x1, y0, y1, +1),
                                      ("y", y0, x0, x1, -1), ("y", y1, x0, x1, +1)]:
            cov = shared.get((k, axis, round(c, 6)), [])
            for (a, b) in _sub_intervals(lo, hi, cov):
                if b - a >= 0.05:          # drop sliver runs (junction noise)
                    runs.append((k, axis, c, a, b, sign))
    return runs


def _room_interior_points(rng, poly, n):
    """n jittered interior points (light positions): centroid fan with a
    minimum spacing (two same-band luminaire shades may not overlap)."""
    cx = sum(p[0] for p in poly) / len(poly)
    cy = sum(p[1] for p in poly) / len(poly)
    pts = []
    for k in range(max(n, 1)):
        q = None
        for spread in (0.9, 1.4, 2.0):        # widen on collision pressure
            for _try in range(8):
                c = (cx + rng.uniform(-spread, spread),
                     cy + rng.uniform(-spread, spread))
                if all((c[0] - p[0]) ** 2 + (c[1] - p[1]) ** 2 > 0.35 ** 2
                       for p in pts):
                    q = c
                    break
            if q is not None:
                break
        pts.append(q if q is not None else
                   (cx + rng.uniform(-2.0, 2.0), cy + rng.uniform(-2.0, 2.0)))
    return pts


_PLACEHOLDER_SETS = {
    "living":  [("Sofa", 1.8, 0.85, 0.75), ("Table", 1.0, 0.6, 0.45)],
    "bedroom": [("Bed", 1.5, 2.0, 0.55)],
    "dining":  [("Table", 1.4, 0.9, 0.74)],
    "study":   [("Table", 1.2, 0.6, 0.74)],
    "hallway": [],
    "bathroom": [],            # fixtures come from layout
    "kitchen": [],
    "studio": [("Bed", 1.5, 2.0, 0.55), ("Sofa", 1.8, 0.85, 0.75)],  # open-plan studio
}


def _compose_rect_union(rng, spec, rects, corridor_idx=None):
    """Fill spec.rooms/walls/doors/windows/lights/placeholders from an
    axis-aligned rect union with the connectivity guarantee."""
    # all shared boundaries (down to slivers) are internal walls and are
    # subtracted from the exterior; only spans >= 1.1 m are door-feasible
    # (sliver boundaries emitted as "exterior" walls would poke into the
    # neighbouring room).
    edges = fp._rect_adjacency(rects, min_span=1e-4)   # catch even slivers
    # (sibling BSP branches split at nearly equal lines leave centimetre
    # overlaps; if missed they surface as "exterior" walls poking into the
    # neighbour. Slivers are subtracted from the exterior but emit no wall
    # (<5cm): the perpendicular internal walls' thickness (>=10cm) covers the
    # junction, so the envelope stays watertight.)
    door_ok = [k for k, e in enumerate(edges) if e[5] - e[4] >= 1.1]
    n = len(rects)
    tree = set()
    if n > 1:
        sub = [edges[k] for k in door_ok]
        tree_sub = fp._max_span_tree(n, sub)
        tree = {door_ok[k] for k in tree_sub}
        assert len(tree) == n - 1, f"spanning tree incomplete ({len(tree)}/{n-1})"

    areas = [(r[2] - r[0]) * (r[3] - r[1]) for r in rects]
    funcs = fp.assign_functions(rng, areas, corridor_idx)
    for k, (x0, y0, x1, y1) in enumerate(rects):
        spec.rooms.append(RoomSpec(f"Room_{k:02d}", funcs[k],
                                   [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]))

    # interior walls = pairwise shared spans; doors on tree edges (+p loop)
    occupied = {}                     # wall_id -> [(u0,u1)] for anti-overlap
    for e_idx, (i, j, axis, c, s0, s1) in enumerate(edges):
        if s1 - s0 < 0.05:
            continue                       # sliver: no wall (see note above)
        wid = f"IW_{e_idx:02d}"
        p1 = [c, s0] if axis == "x" else [s0, c]
        p2 = [c, s1] if axis == "x" else [s1, c]
        spec.walls.append(WallRun(wid, p1, p2, spec.int_wall_t, True))
        span = s1 - s0
        if e_idx in tree or (span >= 1.6 and rng.random() < 0.35):
            dw = max(0.7, min(1.0, span - 0.5))
            u = rng.uniform(0.25 + dw / 2, span - 0.25 - dw / 2) \
                if span - 0.5 - dw > 0 else span / 2
            # 50-105 deg: a 20deg-open leaf blocks ~94% of the doorway, and
            # cross-room covisibility physically collapses on tree-critical
            # doors. Walkthrough-capture realism also favours mostly open
            # interiors.
            _drng = random.Random(
                zlib.crc32(f"{spec.seed}:door:{wid}:{u:.3f}".encode()))
            spec.doors.append(DoorSpec(
                wall=wid, u=u, width=dw,
                leaf_open_deg=rng.uniform(50, 105),
                style=_drng.choice(["panel2", "panel4", "flat", "grooved"]),
                handle=_drng.choice(["lever", "lever", "knob"]),
                finish=_drng.choices(
                    ["white", "wood", "sage", "graphite", "navy"],
                    weights=[30, 28, 12, 15, 15])[0]))
            occupied.setdefault(wid, []).append((u - dw / 2, u + dw / 2))

    # exterior walls from boundary runs
    ext = _exterior_runs(rects, edges)
    # merge collinear exterior runs: per-room-edge emission would make one
    # physical wall two objects meeting at internal-wall T-points; with corner
    # extensions they overlap coplanarly -> mutual shadow-ray blocking ->
    # black bands flanking internal walls (independent of the sample count).
    # One physical wall line = one run; extensions then act only at true
    # perpendicular corners.
    groups = {}
    for (k, axis, c, a, b, sign) in ext:
        groups.setdefault((axis, round(c, 6), sign), []).append((a, b, k))
    merged = []
    for (axis, c, sign), ivs in groups.items():
        ivs.sort()
        cur_a, cur_b, rooms_in = ivs[0][0], ivs[0][1], [(ivs[0][2], ivs[0][0], ivs[0][1])]
        for (a, b, k) in ivs[1:]:
            if a <= cur_b + 1e-6:
                cur_b = max(cur_b, b)
                rooms_in.append((k, a, b))
            else:
                merged.append((axis, c, cur_a, cur_b, sign, rooms_in))
                cur_a, cur_b, rooms_in = a, b, [(k, a, b)]
        merged.append((axis, c, cur_a, cur_b, sign, rooms_in))
    ext_ids = []
    for r_idx, (axis, c, a, b, sign, rooms_in) in enumerate(merged):
        wid = f"EW_{r_idx:02d}"
        # orientation rule: order p1->p2 so the room interior is on the left
        # of the run (wall-frame normal (-uy,ux) points inward); solids.py
        # offsets exterior walls outward by -normal*t/2. sign=+1 means the
        # outward direction is +axis -> reverse the run.
        if axis == "x":
            p1, p2 = ([c, b], [c, a]) if sign < 0 else ([c, a], [c, b])
        else:
            p1, p2 = ([a, c], [b, c]) if sign < 0 else ([b, c], [a, c])
        spec.walls.append(WallRun(wid, p1, p2, spec.ext_wall_t, False))
        # per-room pieces carry their u-range on the merged run so windows
        # and the entry door stay inside the owning room's stretch.
        # The u mapping must follow the p1->p2 order chosen above, which
        # differs per axis:
        #   axis=="x": sign<0 -> p1 at y=b (u = b - y);  sign>=0 -> u = y - a
        #   axis=="y": sign<0 -> p1 at x=a (u = x - a);  sign>=0 -> u = b - x
        # Using the axis=="y" pairing for both axes would mirror every
        # E/W-wall interval: on multi-room runs a room's windows would land in
        # the neighbour's stretch or straddle the interior T-junction (a wall
        # end poking through the glass).
        for (k, pa, pb) in rooms_in:
            if (axis == "y") == (sign < 0):
                u0, u1 = pa - a, pb - a
            else:
                u0, u1 = b - pb, b - pa
            ext_ids.append((wid, k, u0, u1))

    _entry_and_windows(rng, spec, ext_ids, occupied)
    _lights_and_placeholders(rng, spec)


def _compose_polygon_studio(rng, spec, ring):
    """circle/pentagon: one polygon room, oriented perimeter walls."""
    spec.rooms.append(RoomSpec("Room_00", "living", [list(p) for p in ring]))
    ext_ids = []
    for i in range(len(ring)):
        p1, p2 = ring[i], ring[(i + 1) % len(ring)]
        L = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
        wid = f"EW_{i:02d}"
        spec.walls.append(WallRun(wid, list(p1), list(p2), spec.ext_wall_t, False))
        ext_ids.append((wid, 0, 0.0, L))
    _entry_and_windows(rng, spec, ext_ids, {})
    _lights_and_placeholders(rng, spec)


def _entry_and_windows(rng, spec, ext_ids, occupied):
    """One ~closed entry door on the longest exterior run; windows per room
    on its exterior runs with 1D anti-overlap (as in the first-generation
    wall_decorator)."""
    walls = {w.id: w for w in spec.walls}

    def wall_len(wid):
        w = walls[wid]
        return math.hypot(w.p2[0] - w.p1[0], w.p2[1] - w.p1[1])

    ext_sorted = sorted(ext_ids, key=lambda e: -(e[3] - e[2]))
    if ext_sorted and ext_sorted[0][3] - ext_sorted[0][2] >= 1.3:
        wid, _room, pu0, pu1 = ext_sorted[0]
        L = pu1 - pu0
        dw = min(0.95, L - 0.4)                      # short-edge studios: adapt
        u = pu0 + rng.uniform(0.2 + dw / 2, L - 0.2 - dw / 2)
        _drng = random.Random(zlib.crc32(f"{spec.seed}:door:{wid}:{u:.3f}".encode()))
        spec.doors.append(DoorSpec(wall=wid, u=u, width=dw,
                                   leaf_open_deg=rng.uniform(0.0, 6.0),
                                   entry=True,
                                   style=_drng.choice(["panel2", "panel4",
                                                       "flat", "grooved"]),
                                   handle=_drng.choice(["lever", "lever",
                                                        "knob"]),
                                   finish=_drng.choices(
                                       ["white", "wood", "sage", "graphite",
                                        "navy"],
                                       weights=[30, 28, 12, 15, 15])[0]))
        occupied.setdefault(wid, []).append((u - dw / 2 - 0.1, u + dw / 2 + 0.1))
    else:
        raise AssertionError("no exterior run can host an entry door")

    by_room = {}
    for wid, room_idx, pu0, pu1 in ext_ids:
        by_room.setdefault(room_idx, []).append((wid, pu0, pu1))
    for room_idx, runs in by_room.items():
        room = spec.rooms[room_idx]
        n_win = max(0, min(3, int(room.area / 12.0) + 1))
        cands = [rc for rc in runs if rc[2] - rc[1] >= 1.6]
        rng.shuffle(cands)
        placed = 0
        for wid, pu0, pu1 in cands:
            if placed >= n_win:
                break
            L = pu1 - pu0
            ww = rng.uniform(0.8, min(1.6, L - 0.8))
            iv = occupied.setdefault(wid, [])
            for _try in range(8):
                u = pu0 + rng.uniform(0.4 + ww / 2, L - 0.4 - ww / 2) \
                    if L - 0.8 - ww > 0 else -1
                if u < 0:
                    break
                if all(u + ww / 2 <= a or u - ww / 2 >= b for a, b in iv):
                    hh = rng.uniform(1.0, 1.5)
                    ss = rng.uniform(0.8, 1.0)
                    _is_bath = spec.rooms[room_idx].function == "bathroom"
                    if _is_bath:
                        # privacy: small, high-silled bathroom windows
                        ww = min(ww, 0.9)
                        ss = rng.uniform(1.25, 1.45)
                        hh = min(hh, 0.9)
                    mv = rng.random() < 0.55
                    mh = rng.random() < 0.35
                    # window type variety on a derived hash stream (no
                    # main-stream draws, so window positions/counts do not
                    # depend on it; only sizes change).
                    # 60% standard / 25% picture / 15% floor-to-ceiling.
                    # Bay windows (wall bump-out) are not implemented.
                    wrng = random.Random(
                        zlib.crc32(f"{spec.seed}:{wid}:{u:.3f}".encode()))
                    t_roll = wrng.random() if not _is_bath else 0.99
                    if t_roll < 0.25:            # picture window
                        ww2 = min(ww * wrng.uniform(1.3, 1.7),
                                  L - 0.9, 2.6)
                        if all(u + ww2 / 2 <= a or u - ww2 / 2 >= b
                               for a, b in iv) and \
                                pu0 + 0.4 < u - ww2 / 2 and \
                                u + ww2 / 2 < pu1 - 0.4:
                            ww = ww2
                        hh = wrng.uniform(1.5, 1.85)
                        ss = wrng.uniform(0.55, 0.75)
                    elif t_roll < 0.40:          # floor-to-ceiling
                        ss = wrng.uniform(0.05, 0.10)
                        hh = spec.height - ss - wrng.uniform(0.35, 0.55)
                        mv = True
                    hh = min(hh, spec.height - ss - 0.3)
                    iv.append((u - ww / 2, u + ww / 2))
                    # mullion grid variety on the same per-window hash stream
                    # (extra draws are window-local, nothing shifts)
                    grid = wrng.choices(
                        [(1, 1), (2, 1), (2, 2), (3, 2), (1, 2)],
                        weights=[30, 25, 20, 15, 10])[0]
                    spec.windows.append(WindowSpec(
                        wall=wid, u=u, width=ww, height=hh, sill=ss,
                        mull_v=mv, mull_h=mh,
                        mull_nx=grid[0], mull_ny=grid[1],
                        finish=wrng.choices(
                            ["white", "dark_alu", "wood"],
                            weights=[45, 35, 20])[0]))
                    placed += 1
                    break


_FNAME = {"sofa": "Sofa", "bed": "Bed", "table": "Table", "chair": "Chair",
          "wardrobe": "Dresser", "nightstand": "Cabinet",
          "tv_stand": "TvCabinet", "bookshelf": "Bookshelf",
          "mirror": "Mirror", "curtain": "Curtain", "rug": "Rug",
          "painting": "Painting", "poster": "Poster", "clock": "WallClock",
          # bathroom/kitchen: names chosen to resolve additively in _NAME2SEM
          # (toilet33/sink34/bathtub36/shower38/towel27/counter12/stove38/
          # fridge24/hood38/wallcabinet3), no renumbering
          "toilet": "Toilet", "vanity": "SinkVanity", "bathtub": "Bathtub",
          "shower": "Shower", "towel_bar": "TowelBar",
          "counter": "KitchenCounter", "stove": "Stove", "fridge": "Fridge",
          "hood": "RangeHood", "wallcabinet": "WallCabinet",
          # floor decor + hanging pieces; names resolve
          # additively in _NAME2SEM (plant/basket/mobile/fan -> 40 otherprop,
          # bookstack -> 23 books, suitcase -> 37 bag, lantern -> 40)
          "floorplant": "FloorPlant", "basket": "Basket",
          "bookstack": "BookStack", "suitcase": "Suitcase",
          "hangplant": "HangPlant", "lantern": "Lantern",
          "mobile": "MobileDecor", "ceilingfan": "CeilingFan"}
_CNAME = {"book": "Book", "cup": "Decor_cup", "plate": "Decor_plate",
          "vase": "Decor_vase", "smallbox": "Box", "figurine": "Decor_fig"}


def _door_keepouts(spec, room):
    """Keep-out circles at every door on this room's boundary (swing space)."""
    wr = {w.id: w for w in spec.walls}
    outs = []
    for d in spec.doors:
        run = wr.get(d.wall)
        if run is None:
            continue
        dx, dy = run.p2[0] - run.p1[0], run.p2[1] - run.p1[1]
        L = math.hypot(dx, dy)
        if L < 1e-6:
            continue
        cx = run.p1[0] + dx * (d.u / L)
        cy = run.p1[1] + dy * (d.u / L)
        if _poly_dist(cx, cy, room.poly) < 0.35:
            outs.append((cx, cy, d.width + 0.15))
    return outs


def _windows_on_room(spec, room):
    """Curtain anchors: (wall_id, cx, cy, w, h, sill, yaw_in, nx, ny,
    half_wall_thickness) for windows on this room's boundary; normal points
    into the room."""
    wr = {w.id: w for w in spec.walls}
    out = []
    def _room_edge_list(poly):
        out = []
        for i in range(len(poly)):
            ax, ay = poly[i]
            bx, by = poly[(i + 1) % len(poly)]
            eL = math.hypot(bx - ax, by - ay)
            if eL < 0.4:
                continue
            out.append((ax, ay, (bx - ax) / eL, (by - ay) / eL, eL))
        return out

    ctr_x = sum(p[0] for p in room.poly) / len(room.poly)
    ctr_y = sum(p[1] for p in room.poly) / len(room.poly)
    for w in spec.windows:
        run = wr.get(w.wall)
        if run is None:
            continue
        dx, dy = run.p2[0] - run.p1[0], run.p2[1] - run.p1[1]
        L = math.hypot(dx, dy)
        if L < 1e-6:
            continue
        cx = run.p1[0] + dx * (w.u / L)
        cy = run.p1[1] + dy * (w.u / L)
        if _poly_dist(cx, cy, room.poly) > 0.18:
            continue
        # the window must lie on one of this room's edges: a loose 0.35m
        # radius would let a neighbour adopt a window sitting near a shared
        # corner and hang a curtain for it around the corner (floating or
        # wall-cutting curtains). Containment check: the window span must fit
        # the room edge nearest to its centre.
        _edges = _room_edge_list(room.poly)
        _best = None
        for (eax, eay, edx, edy, eL) in _edges:
            uu = min(max((cx - eax) * edx + (cy - eay) * edy, 0.0), eL)
            dd = math.hypot(cx - (eax + edx * uu), cy - (eay + edy * uu))
            if _best is None or dd < _best[0]:
                _best = (dd, uu, eL)
        if _best is None or _best[1] - w.width / 2 < -0.05 \
                or _best[1] + w.width / 2 > _best[2] + 0.05:
            continue
        nx, ny = dy / L, -dx / L
        if (ctr_x - cx) * nx + (ctr_y - cy) * ny < 0:
            nx, ny = -nx, -ny
        # same sign convention as layout.against_wall: meshes rotate
        # R_z(+yaw), so front(+y)->(-sin,cos)==n_in requires atan2(-nx, ny).
        # For the curtain band this also maps its local x axis exactly to the
        # wall direction (curtains stay parallel to non-axis walls of
        # pentagon/circle rooms).
        yaw = math.degrees(math.atan2(-nx, ny))
        out.append((w.wall, cx, cy, w.width, w.height, w.sill, yaw, nx, ny,
                    run.t / 2))
    return out


def _in_poly_pt(x, y, poly):
    inside = False
    j = len(poly) - 1
    for i in range(len(poly)):
        xi, yi = poly[i]; xj, yj = poly[j]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi:
            inside = not inside
        j = i
    return inside


def _poly_dist(x, y, poly):
    best = float("inf")
    for i in range(len(poly)):
        ax, ay = poly[i]
        bx, by = poly[(i + 1) % len(poly)]
        abx, aby = bx - ax, by - ay
        den = max(abx * abx + aby * aby, 1e-9)
        t = min(max(((x - ax) * abx + (y - ay) * aby) / den, 0.0), 1.0)
        best = min(best, math.hypot(x - (ax + t * abx), y - (ay + t * aby)))
    return best


def _lights_and_placeholders(rng, spec):
    from . import furniture as _fu
    from . import layout as _lay
    for room in spec.rooms:
        # the point count must match the round(area/12) lamps placed by
        # light_schedule: otherwise pts[k%len] wraps, two pendants land on the
        # identical coordinate and their coincident shade boxes z-fight to
        # exact-zero black. light_schedule additionally jitters any wrapped
        # point.
        n = max(1, round(room.area / 12.0))
        pts = _room_interior_points(rng, room.poly, n)
        new_lamps = []
        for L in fp.light_schedule(rng, room.name, room.function, room.area, pts):
            spec.lights.append(LightSpec(**L))
            new_lamps.append(spec.lights[-1])
        # lathe shade styles per lamp (derived stream: main stream untouched;
        # randomness added, not narrowed)
        rng7 = random.Random((spec.seed << 28)
                             ^ zlib.crc32(room.name.encode()) ^ 0x5AADE)
        for L in new_lamps:
            L.style = _fu.sample_shade_style(rng7, L.fixture)
        # placeholder boxes are retired, but their main-stream draws are kept
        # (burned): removing them would shift every seed's floorplan
        # (random-stream parity)
        for k, (base, sx, sy, sz) in enumerate(_PLACEHOLDER_SETS[room.function]):
            rng.uniform(-0.6, 0.6); rng.uniform(-0.6, 0.6)
            rng.choice([0.0, 90.0, 180.0, 270.0])
        # furniture layout (derived stream; all decisions recorded)
        rng3 = random.Random((spec.seed << 20)
                             ^ zlib.crc32(room.name.encode()) ^ 0xF00D)
        counter = [0]

        def new_item(ftype, params, x, y, yaw, z=0.0, wall_id="", _r=room):
            built = _fu.build_item(ftype, params)
            name = f"{_FNAME[ftype]}_{_r.name[-2:]}{counter[0]:02d}"
            counter[0] += 1
            spec.furniture.append(FurnitureSpec(
                name=name, ftype=ftype, room=_r.name, x=round(x, 4),
                y=round(y, 4), yaw_deg=round(yaw, 3), z=round(z, 4),
                wall_id=wall_id, hx=round(built["half_xy"][0], 4),
                hy=round(built["half_xy"][1], 4),
                height=round(built["height"], 4), params=params))
            return name

        _pl, _board, _stats = _lay.furnish_room(
            rng3, room, _door_keepouts(spec, room),
            _windows_on_room(spec, room), new_item, walls=spec.walls,
            ceil_pts=[(L.x, L.y) for L in spec.lights
                      if L.room == room.name
                      and L.fixture in ("pendant", "flush", "bulb")],
            room_h=spec.height)
        spec.layout_stats[room.name] = _stats          # solid-wall fallback count
        # wall art (paintings/posters/clocks) on its own derived stream;
        # shares the room's WallBoard with curtains/mirrors
        rng6 = random.Random((spec.seed << 24)
                             ^ zlib.crc32(room.name.encode()) ^ 0xDECA1)
        _lay.place_wall_art(
            rng6, room, spec, _door_keepouts(spec, room),
            [(cx, cy, ww) for (_wid, cx, cy, ww, _h, _s, _y, _nx, _ny, _th)
             in _windows_on_room(spec, room)], new_item, board=_board,
            walls=spec.walls)
        # floating distractors (geometry diversity + occlusion supply;
        # visible to both render and geometry, no hide_viewport anywhere)
        for k in range(rng3.randint(0, 3)):
            p = _fu.sample_clutter_item(rng3, rng3.choice(
                ["smallbox", "vase", "figurine"]))
            px, py = _room_interior_points(rng3, room.poly, 1)[0]
            spec.clutter.append(ClutterSpec(
                name=f"Distractor_{room.name[-2:]}{k}", parent="",
                x=round(px, 4), y=round(py, 4),
                z=round(rng3.uniform(1.6, 2.4), 3),
                yaw_deg=round(rng3.uniform(0, 360), 2), params=p))
        # lighting draws come from a derived stream so the architecture
        # layer's rng consumption (hence every seed's floorplan) is untouched
        rng2 = random.Random((spec.seed << 16)
                             ^ zlib.crc32(room.name.encode()))
        _functional_lights(rng2, spec, room,
                           [fs for fs in spec.furniture if fs.room == room.name])
        if room.function in ("living", "dining") and rng.random() < 0.35:
            spec.ceiling_border_rooms.append(room.name)


# functional luminaires (placement is function-driven; density stays
# area-driven via light_schedule above)
_LAMP_LM = {"table": (300, 800), "floor": (400, 900), "wall": (200, 600)}


def _functional_lights(rng, spec, room, placed):
    """Anchor secondary luminaires to function/furniture: dining pendant over
    the table, bedside table lamps (on nightstands added here), living floor
    lamp by the sofa, study desk lamp, hallway wall lamps."""
    def lamp(fixture, x, y, z, cct=(2500, 3200), yaw=0.0):
        from . import furniture as _fu2
        lo, hi = _LAMP_LM[fixture]
        spec.lights.append(LightSpec(
            room=room.name, fixture=fixture, x=x, y=y, z=z,
            lumens=rng.uniform(lo, hi), cct_k=rng.uniform(*cct), yaw_deg=yaw,
            style=_fu2.sample_shade_style(rng, fixture)))

    # anchors are real furniture (FurnitureSpec list), no placeholders
    by_type = {}
    for fs in placed:
        by_type.setdefault(fs.ftype, []).append(fs)

    # lamp positions are function-driven: anchor the main lamp (largest
    # lumens) to the room's functional focus; secondary lamps keep their
    # random fan points (randomness preserved: anchor jitter +-0.25, all
    # other draws intact).
    ceil = [L for L in spec.lights
            if L.room == room.name and L.fixture in ("pendant", "flush", "bulb")]
    if ceil:
        main = max(ceil, key=lambda L: L.lumens)
        anchor = None
        if room.function == "living":
            grp = [f for f in placed if f.ftype in ("sofa", "table", "tv_stand")]
            if grp:
                anchor = (sum(f.x for f in grp) / len(grp),
                          sum(f.y for f in grp) / len(grp))
        elif room.function == "dining" and by_type.get("table"):
            t = by_type["table"][0]
            anchor = (t.x, t.y)
        if anchor is None:                       # bedroom/study/hallway/fallback
            anchor = (sum(p[0] for p in room.poly) / len(room.poly),
                      sum(p[1] for p in room.poly) / len(room.poly))
        band = {"flush": "hi", "bulb": "hi", "pendant": "pend"}
        for _try in range(6):
            nx = anchor[0] + rng.uniform(-0.25, 0.25)
            ny = anchor[1] + rng.uniform(-0.25, 0.25)
            if all(b is main or band[b.fixture] != band[main.fixture]
                   or (nx - b.x) ** 2 + (ny - b.y) ** 2 > 0.35 ** 2
                   for b in ceil):
                if _in_poly_pt(nx, ny, room.poly):
                    main.x, main.y = round(nx, 4), round(ny, 4)
                break
    if room.function == "dining" and by_type.get("table"):
        # re-aim the room's pendant over the table (physical habit, not a new
        # lamp), but never park it on another pendant's shade (coincident
        # same-band shades render black)
        t = by_type["table"][0]
        pends = [L for L in spec.lights
                 if L.room == room.name and L.fixture == "pendant"]
        for L in pends:
            nx = t.x + rng.uniform(-0.1, 0.1)
            ny = t.y + rng.uniform(-0.1, 0.1)
            if all((o is L) or (nx - o.x) ** 2 + (ny - o.y) ** 2 > 0.40 ** 2
                   for o in pends):
                L.x, L.y = nx, ny
            break
    if room.function == "bedroom":
        for ns in by_type.get("nightstand", []):
            if rng.random() < 0.75:
                lamp("table", ns.x, ns.y, ns.height + 0.30)
    if room.function == "study" and by_type.get("table") \
            and rng.random() < 0.85:
        t = by_type["table"][0]
        lamp("table", t.x + rng.uniform(-0.3, 0.3),
             t.y + rng.uniform(-0.15, 0.15), t.height + 0.30,
             cct=(3500, 5000))
    if room.function == "living" and by_type.get("sofa") \
            and rng.random() < 0.6:
        s = by_type["sofa"][0]
        lamp("floor", s.x + rng.uniform(0.9, 1.3) * rng.choice([-1, 1]),
             s.y + rng.uniform(-0.4, 0.4), 1.45)
    if room.function == "hallway" and rng.random() < 0.5:
        # project interior samples onto the room boundary: mount ON the wall
        # face (proud by the mount depth), shade faces the room interior
        for (x, y) in _room_interior_points(rng, room.poly, 2):
            best = None
            for i in range(len(room.poly)):
                ax, ay = room.poly[i]
                bx, by = room.poly[(i + 1) % len(room.poly)]
                abx, aby = bx - ax, by - ay
                den = max(abx * abx + aby * aby, 1e-9)
                tt = min(max(((x - ax) * abx + (y - ay) * aby) / den, 0.0), 1.0)
                px, py = ax + tt * abx, ay + tt * aby
                d2 = (x - px) ** 2 + (y - py) ** 2
                if best is None or d2 < best[0]:
                    best = (d2, px, py, x - px, y - py)
            _, px, py, inx, iny = best
            ln = math.hypot(inx, iny)
            if ln < 1e-6:
                continue
            inx, iny = inx / ln, iny / ln       # boundary -> interior direction
            yaw = math.degrees(math.atan2(inx, iny))   # matches solids sin/cos use
            lamp("wall", px + inx * 0.022, py + iny * 0.022, 1.75, yaw=yaw)


# astronomical solar model (NOAA SPA-lite, pure math, bpy-free)

def solar_position(lat_deg, lon_deg, day_of_year, hour_utc):
    """NOAA general solar position (Meeus-derived fits): true solar
    azimuth/elevation for latitude/longitude/date/UTC-time. Accuracy well
    under 1 deg -- lighting-grade, not ephemeris-grade."""
    g = 2.0 * math.pi / 365.0 * (day_of_year - 1 + (hour_utc - 12.0) / 24.0)
    eqtime = 229.18 * (0.000075 + 0.001868 * math.cos(g)
                       - 0.032077 * math.sin(g) - 0.014615 * math.cos(2 * g)
                       - 0.040849 * math.sin(2 * g))                # minutes
    decl = (0.006918 - 0.399912 * math.cos(g) + 0.070257 * math.sin(g)
            - 0.006758 * math.cos(2 * g) + 0.000907 * math.sin(2 * g)
            - 0.002697 * math.cos(3 * g) + 0.00148 * math.sin(3 * g))  # rad
    tst = hour_utc * 60.0 + eqtime + 4.0 * lon_deg      # true solar time (min)
    ha = math.radians(tst / 4.0 - 180.0)                # hour angle
    lat = math.radians(lat_deg)
    cos_zen = (math.sin(lat) * math.sin(decl)
               + math.cos(lat) * math.cos(decl) * math.cos(ha))
    cos_zen = min(max(cos_zen, -1.0), 1.0)
    zen = math.acos(cos_zen)
    elev = 90.0 - math.degrees(zen)
    if math.sin(zen) < 1e-9:
        az = 180.0
    else:
        cos_az = ((math.sin(lat) * cos_zen - math.sin(decl))
                  / (math.cos(lat) * math.sin(zen)))
        az = math.degrees(math.acos(min(max(cos_az, -1.0), 1.0)))
        az = 360.0 - az if tst / 4.0 - 180.0 < 0.0 else az
        az = (az + 180.0) % 360.0                       # from-north, cw
    return elev, az


# archetype <-> solar-elevation bands (the 3 archetypes are aliases over the
# continuous model; civil-twilight-like dusk band)
_ARCH_BANDS = {"day": (10.0, 90.0), "dusk": (-4.0, 10.0), "night": (-90.0, -4.0)}


def _sample_sun_astronomical(rng, arch):
    """lat/lon/date drawn from the habitable domain, then the time-of-day is
    picked uniformly among the 24h grid slots whose computed elevation falls
    in the archetype's band (conditioning preserves the 55/20/25 archetype
    mix while every accepted draw is an exact astronomical configuration).
    High-latitude degenerate cases (polar day/night lack a band) resample;
    hard fallback = fixed-range draw, flagged in the returned dict."""
    lo, hi = _ARCH_BANDS[arch]
    for _ in range(25):
        lat = rng.uniform(-56.0, 68.0)
        lon = rng.uniform(-180.0, 180.0)
        doy = rng.randint(1, 365)
        slots = []
        for k in range(96):                              # 15-min grid
            h = k * 0.25
            e, _a = solar_position(lat, lon, doy, h)
            if lo <= e < hi:
                slots.append(h)
        if not slots:
            continue
        hour = min(rng.choice(slots) + rng.uniform(0.0, 0.25), 23.99)
        elev, az = solar_position(lat, lon, doy, hour)
        if lo <= elev < hi:
            return {"model": "noaa", "lat": round(lat, 3), "lon": round(lon, 3),
                    "day_of_year": doy, "hour_utc": round(hour, 3),
                    "elevation_deg": round(elev, 3), "azimuth_deg": round(az, 3)}
    legacy = {"day": (15.0, 62.0), "dusk": (1.5, 8.0), "night": (-25.0, -8.0)}[arch]
    elev = rng.uniform(*legacy)
    return {"model": "legacy-fallback", "elevation_deg": round(elev, 3),
            "azimuth_deg": round(rng.uniform(0.0, 360.0), 3)}


def _moon_phase_factor(phase):
    """Allen's astrophysical magnitude fit: brightness vs full moon.
    phase 1=full, 0=new; alpha = phase angle. Quarter moon -> ~0.11 x full
    (matches observation; lunar brightness is strongly nonlinear)."""
    alpha = 180.0 * (1.0 - min(max(phase, 0.0), 1.0))
    dm = 0.026 * alpha + 4.0e-9 * alpha ** 4
    return 10.0 ** (-0.4 * dm)


def _sky_and_exposure(rng, spec, force_arch=None):
    """Sky archetypes + physical exposure (exposure is derived from the
    photometric budget and is unique per scene: not an arbitrary constant,
    and not a per-view compensator).

    Photometric budget: E_int = sum(on-lumens)*U / A_floor  [lux, utilization
    U~0.72]; E_day = window daylight via a small view-factor k_win on the
    horizontal sky illuminance model 90k*sin(elev)*sun_int (dusk/night -> sky
    term only). film_exposure = E_REF / (E_int + E_day), clamped [0.25, 4];
    E_REF=100 lux is the single calibration constant, anchored on a reference
    brightness point (lm/160 @ exposure 1.0) and validated by the numeric
    brightness check (per-room median in [0.25, 0.65] sRGB)."""
    arch = rng.choices(["day", "dusk", "night"], weights=[0.55, 0.20, 0.25])[0]
    if force_arch is not None:          # tests only; production draws it from the seed
        arch = force_arch
    spec.sky_archetype = arch
    spec.efficacy_lm_w = rng.uniform(150.0, 220.0)
    # continuous astronomical sun (lat/lon/date/time -> exact
    # azimuth/elevation); archetypes survive as elevation-band aliases
    spec.solar = _sample_sun_astronomical(rng, arch)
    spec.sun_elevation_deg = spec.solar["elevation_deg"]
    spec.sun_azimuth_deg = spec.solar["azimuth_deg"]
    if arch == "day":
        spec.sun_intensity = rng.uniform(0.6, 1.4)
        spec.sky_strength = rng.uniform(0.7, 1.4)
        spec.dust_density = rng.uniform(0.0, 1.0)
        p_on_win, p_on_dark = 0.35, 0.9
    elif arch == "dusk":
        spec.sun_intensity = rng.uniform(0.5, 1.2)
        spec.sky_strength = rng.uniform(0.5, 1.0)
        spec.dust_density = rng.uniform(0.8, 3.0)
        p_on_win, p_on_dark = 0.8, 0.95
    else:                                            # night
        spec.sun_intensity = 0.0
        spec.sky_strength = rng.uniform(0.01, 0.06)
        spec.dust_density = rng.uniform(0.0, 1.0)
        p_on_win, p_on_dark = 1.0, 1.0
    # moonlight: the night sky's directional source (cold, faint,
    # phase-modulated). A physical accent (window pools, faint shadows), not
    # a room light: interiors still rely on lamps + the exposure servo.
    spec.moon = {"enabled": False}
    if arch == "night" and rng.random() < 0.85:
        phase = rng.uniform(0.05, 1.0)
        factor = _moon_phase_factor(phase)
        spec.moon = {
            "enabled": True, "elevation_deg": round(rng.uniform(12.0, 60.0), 2),
            "azimuth_deg": round(rng.uniform(0.0, 360.0), 2),
            "phase": round(phase, 3),                # 1=full, 0=new
            "illum_factor": round(factor, 4),
            "lux": round(0.32 * factor, 5),          # full moon ~0.32 lx (physical)
            "cct_k": 4125.0,                          # measured moonlight CCT
            # declared tone surrogate: 0.32 lx vs ~100 lx lamps is below the
            # 8-bit sRGB dynamic range; the boost makes moon pools render
            # dimly visible (like long-exposure night photos). Set 1.0 for
            # strictly physical.
            "render_boost": 30.0,
            # note: the moon position is drawn, not taken from a lunar
            # ephemeris (the sun uses the NOAA model)
        }
        # moonlit night sky is brighter than a moonless one
        spec.sky_strength = min(spec.sky_strength * (1.0 + 2.0 * factor), 0.15)
    # on/off habits (realistic randomness): windowless rooms keep lights on
    # far more often; >=1 luminaire stays on per room (no black rooms)
    wr_by_id = {w.id: w for w in spec.walls}
    win_pts = []
    for w in spec.windows:
        run = wr_by_id.get(w.wall)
        if run is None:
            continue
        dx, dy = run.p2[0] - run.p1[0], run.p2[1] - run.p1[1]
        Lr = math.hypot(dx, dy)
        if Lr > 1e-6:
            uu = min(max(w.u, 0.0), Lr) / Lr
            win_pts.append((run.p1[0] + dx * uu, run.p1[1] + dy * uu))
    def _room_windowed(r):
        for qx, qy in win_pts:
            for i in range(len(r.poly)):
                ax, ay = r.poly[i]
                bx, by = r.poly[(i + 1) % len(r.poly)]
                abx, aby = bx - ax, by - ay
                den = max(abx * abx + aby * aby, 1e-9)
                tt = min(max(((qx - ax) * abx + (qy - ay) * aby) / den, 0.0), 1.0)
                if math.hypot(qx - (ax + tt * abx), qy - (ay + tt * aby)) < 0.35:
                    return True
        return False
    room_lit = {r.name: _room_windowed(r) for r in spec.rooms}
    for L in spec.lights:
        p_on = p_on_win if room_lit.get(L.room, True) else p_on_dark
        L.on = rng.random() < p_on
    by_room = {}
    for L in spec.lights:
        by_room.setdefault(L.room, []).append(L)
    for ls in by_room.values():
        if not any(x.on for x in ls):
            rng.choice(ls).on = True
    # illuminance floor (people light rooms they cannot see in): any room
    # whose on-lumen density is too low turns more lamps on. Day rooms with
    # windows get a lower floor (sun usually carries them) instead of a full
    # exemption: a north-facing/deep windowed room at low sun is still dim
    # and real occupants still switch lights on.
    area_by_room = {r.name: max(r.area, 1.0) for r in spec.rooms}
    for rname, ls in by_room.items():
        # evenings light the whole home, so dusk and night rooms get a higher
        # illuminance floor (outer rooms of large floor plans would otherwise
        # sit at a median brightness of 0.23-0.25, under the threshold).
        # Real materials bring dark floors/furniture (tile albedo ~0.30);
        # dark interiors need more lumens for the same luminance, hence a
        # modest daytime raise (same physical rule as the evening floor).
        if arch in ("dusk", "night"):
            floor_lm = 62.0
        elif room_lit.get(rname, False):
            floor_lm = 30.0
        else:
            floor_lm = 48.0
        # albedo-aware: a dark floor absorbs the light budget, so scale the
        # floor by the measured material albedo (as real lighting design
        # does). This keeps the per-room luminance spread tight under the
        # scene-level exposure (raising all floors would just move the servo).
        fe = spec.materials.get(f"floor/{rname}", {})
        alb = fe.get("albedo_mean") if fe.get("mode") == "image" \
            else (max(fe.get("rgb", [0.45])) if fe else 0.45)
        alb = (alb or 0.45) * fe.get("tint_v", 1.0)
        # compensate darkness only; never reward bright floors with fewer
        # lamps (a windowless corridor with bright tile still needs its base
        # floor, hence the lower clamp at 1.0)
        floor_lm *= min(max(0.42 / max(alb, 0.12), 1.0), 1.9)
        offs = [x for x in ls if not x.on]
        rng.shuffle(offs)
        while offs and (sum(x.lumens for x in ls if x.on)
                        / area_by_room.get(rname, 15.0)) < floor_lm:
            offs.pop().on = True
    # budget -> exposure
    A = max(sum(r.area for r in spec.rooms), 1.0)
    lm_on = sum(L.lumens for L in spec.lights if L.on)
    E_int = lm_on * 0.72 / A
    win_area = sum(w.width * w.height for w in spec.windows)
    sky_lux = (90000.0 * max(math.sin(math.radians(spec.sun_elevation_deg)), 0.0)
               * max(spec.sun_intensity, 0.0)) if arch != "night" else 0.0
    if arch == "dusk":
        sky_lux = max(sky_lux, 3000.0 * spec.sky_strength)
    E_day = 0.06 * win_area * sky_lux / A
    E_REF = 100.0
    spec.film_exposure = min(max(E_REF / max(E_int + E_day, 5.0), 0.25), 4.0)
    # per-scene mid-gray target for the build-stage probe servo (see
    # build_scene): drawn here so scene-to-scene brightness diversity is a
    # declared spec variable, while cross-view exposure stays scene-constant.
    # The analytic value above remains the prior / bpy-free estimate; the
    # Nishita daylight term cannot be converted analytically to the lm/W lamp
    # convention (the sun-window geometry dominates), so the build measures it.
    # Target band 0.36-0.44: realistic material diversity widens the
    # per-room spread, and the band must sit high enough that the dim tail
    # stays above the 0.25 brightness threshold.
    spec.exposure_target = round(rng.uniform(0.36, 0.44), 3)
    spec.exposure_derivation = {
        "E_int_lux": round(E_int, 1), "E_day_lux": round(E_day, 1),
        "E_ref_lux": E_REF, "lm_on": round(lm_on), "area_m2": round(A, 1),
        "win_area_m2": round(win_area, 2), "mode": "analytic-prior"}


def _backdrops(rng, spec):
    """Exterior massing: blocks/trees outside window-bearing facades, far
    enough never to intersect the footprint."""
    xs = [p[0] for r in spec.rooms for p in r.poly]
    ys = [p[1] for r in spec.rooms for p in r.poly]
    bb = (min(xs), min(ys), max(xs), max(ys))
    win_walls = {w.wall for w in spec.windows}
    seen_dirs = set()
    for wr in spec.walls:
        if wr.internal or wr.id not in win_walls:
            continue
        dx, dy = wr.p2[0] - wr.p1[0], wr.p2[1] - wr.p1[1]
        L = math.hypot(dx, dy)
        if L < 1e-6:
            continue
        dx, dy = dx / L, dy / L
        nx, ny = dy, -dx                       # interior-on-left rule -> right = out
        key = (round(nx), round(ny))
        if key in seen_dirs:                    # one cluster per facade direction
            continue
        seen_dirs.add(key)
        for _ in range(rng.randint(1, 3)):
            dist = rng.uniform(12.0, 35.0)
            u = rng.uniform(0.1, 0.9)
            c = (wr.p1[0] + dx * (u * L) + nx * dist,
                 wr.p1[1] + dy * (u * L) + ny * dist)
            kind = "tree" if rng.random() < 0.3 else "block"
            if kind == "block":
                sx, sy, sz = rng.uniform(4, 12), rng.uniform(4, 12), rng.uniform(3.5, 18)
                gray = rng.uniform(0.15, 0.45)
            else:
                sx = sy = rng.uniform(2.0, 4.5)
                sz = rng.uniform(3.0, 8.0)
                gray = rng.uniform(0.08, 0.2)
            if (bb[0] - sx - 2 < c[0] < bb[2] + sx + 2
                    and bb[1] - sy - 2 < c[1] < bb[3] + sy + 2):
                continue                        # would crowd another wing (U/T shapes)
            spec.backdrops.append(BackdropSpec(
                kind=kind, x=float(c[0]), y=float(c[1]), sx=sx, sy=sy, sz=sz,
                gray=round(gray, 3)))


# public builders

def _shape_rect_decomposition(rng, kind):
    """Emit the rect union for a decomposable shape (params re-derived from
    floorplan generators so seams are exact)."""
    if kind == "rect":
        w, d = rng.uniform(6.0, 10.0), rng.uniform(5.0, 8.0)
        if w * d >= 48.0 and rng.random() < 0.5:
            c = w * rng.uniform(0.4, 0.6)
            return [(0, 0, c, d), (c, 0, w, d)]
        return [(0, 0, w, d)]
    if kind == "l_shape":
        w, h = rng.uniform(7.0, 10.0), rng.uniform(7.0, 10.0)
        cx, cy = w * rng.uniform(0.35, 0.6), h * rng.uniform(0.35, 0.6)
        return [(0, 0, w, h - cy), (0, h - cy, w - cx, h)]
    if kind == "u_shape":
        w, h = rng.uniform(8.0, 11.0), rng.uniform(6.0, 9.0)
        nw = w * rng.uniform(0.25, 0.4)
        nh = h * rng.uniform(0.35, 0.55)
        wing = (w - nw) / 2
        return [(0, 0, w, h - nh), (0, h - nh, wing, h),
                (w - wing, h - nh, w, h)]
    if kind == "t_shape":
        top_w = rng.uniform(8.0, 11.0)
        top_h = rng.uniform(2.5, 3.5)
        stem_w = top_w * rng.uniform(0.35, 0.5)
        stem_h = rng.uniform(4.0, 6.0)
        sx = (top_w - stem_w) / 2
        return [(sx, 0, sx + stem_w, stem_h),
                (0, stem_h, top_w, stem_h + top_h)]
    raise ValueError(kind)


def build_ge1_spec(seed: int, force_archetype: str = None,
                   n_rooms: int = None) -> GenesisSpec:
    """Default scene spec of build_scene.py: floor plan, rooms, openings, furniture, materials, lights
    and sky drawn from the seed (build_ge0_spec is the fixed two-room variant)."""
    rng = random.Random(seed)
    spec = GenesisSpec(seed=seed)
    spec.height = rng.uniform(2.5, 3.2)
    # wall thickness ranges (ext 0.15-0.25 / int 0.08-0.15) on a derived
    # stream: room boundary lines (hence floorplans) do not depend on it.
    rngW = random.Random(seed ^ 0x0A11)
    spec.ext_wall_t = round(rngW.uniform(0.15, 0.25), 3)
    spec.int_wall_t = round(rngW.uniform(0.08, 0.15), 3)
    # random-stream parity: these four draws must keep their main-stream
    # positions, or every seed's floorplan would shift. The values are
    # overwritten by _sky_and_exposure (derived stream) below.
    spec.sun_elevation_deg = rng.uniform(12.0, 62.0)
    spec.sun_azimuth_deg = rng.uniform(0.0, 360.0)
    spec.sun_intensity = rng.uniform(0.6, 1.4)
    spec.sky_strength = rng.uniform(0.5, 1.5)

    if n_rooms is not None:
        # explicit room-count knob (supports 10+). All plan draws ride a
        # derived stream keyed by (seed, n), so the default path below is
        # unchanged when the knob is unset. The footprint scales with n;
        # exact count via split-largest-first (floorplan.bsp_n_rooms).
        assert 1 <= n_rooms <= 24, f"n_rooms {n_rooms} out of range [1,24]"
        rngN = random.Random(seed ^ 0x0500B5 ^ (n_rooms * 0x9E37))
        # n_rooms=1 means one big room (open-plan studio), not a
        # default-size cell, so it gets real studio floor area.
        area_per = rngN.uniform(26.0, 52.0) if n_rooms == 1 \
            else rngN.uniform(14.0, 20.0)
        aspect = rngN.uniform(0.75, 1.35)
        Wn = math.sqrt(n_rooms * area_per * aspect)
        Dn = n_rooms * area_per / Wn
        rects = fp.bsp_n_rooms(rngN, Wn, Dn, n_rooms)
        spec.kind = f"bsp_n{n_rooms}"
        _compose_rect_union(rng, spec, rects)
    elif rng.random() < 0.5:
        kind = rng.choice(fp.SHAPES)
        spec.kind = kind
        if kind in ("circle", "pentagon"):
            ring = fp._circle(rng) if kind == "circle" else fp._pentagon(rng)
            _compose_polygon_studio(rng, spec, ring)
        else:
            _compose_rect_union(rng, spec, _shape_rect_decomposition(rng, kind))
    else:
        W = rng.uniform(13.0, 17.0)
        D = rng.uniform(12.0, 16.0)
        if rng.random() < 0.4:                     # hallway corridor variant
            spec.kind = "bsp_hallway"
            cw = rng.uniform(1.5, 2.0)
            cy0 = D * rng.uniform(0.4, 0.55)
            rects = [(0, cy0, W, cy0 + cw)]
            corridor_idx = 0
            rects += fp._bsp(rng, 0, 0, W, cy0, 0, 2, rng.choice([0, 1]))
            rects += fp._bsp(rng, 0, cy0 + cw, W, D, 0, 2, rng.choice([0, 1]))
            _compose_rect_union(rng, spec, rects, corridor_idx=corridor_idx)
        else:
            spec.kind = "bsp"
            rects = fp._bsp(rng, 0, 0, W, D, 0, rng.randint(2, 3),
                            rng.choice([0, 1]))
            _compose_rect_union(rng, spec, rects)
    # sky archetype + on/off habits + physical exposure need the composed
    # rooms/windows/lights, so they run after the composer, on a derived
    # stream (layer decoupling: lighting draws must not shift architecture)
    _materials_pass(spec)          # before sky: floors are albedo-aware
    rng2 = random.Random(seed ^ 0x9E3779B9)
    _sky_and_exposure(rng2, spec, force_arch=force_archetype)
    _backdrops(rng2, spec)
    _clutter_pass(spec)
    return spec


def _materials_pass(spec):
    """Draw every surface's material (real PBR set or physical procedural)
    + decal images; full provenance into spec.materials.
    Empty or missing library -> pure procedural."""
    from . import materials as _mat
    rng5 = random.Random(spec.seed ^ 0x3A7E51A1)
    lib = _mat.scan_library()
    spec.assets_lib = {"root": lib["root"],
                       "pbr_sets": {k: len(v) for k, v in lib["pbr"].items()},
                       "paintings": len(lib["paintings"]),
                       "posters": len(lib["posters"])}
    spec.material_image_prob = round(rng5.uniform(0.75, 0.95), 3) \
        if lib["pbr"] else 0.0
    _mat.assign_materials(rng5, spec, lib)


def _clutter_pass(spec):
    """Populate every registered support surface (tables, nightstand
    and wardrobe tops, bookshelf shelves -- occupancy 0.3-0.9 with leaning
    book runs, 2-layer stacking). Runs after the lamps so clutter avoids
    table-lamp bases. Derived stream; all placements recorded."""
    from . import furniture as _fu
    from . import layout as _lay
    rng4 = random.Random(spec.seed ^ 0xC7A77E5)
    # beds/sofas (soft supports) and kitchen counters are included, so their
    # support surfaces receive clutter too.
    fitems = [(fs, _fu.build_item(fs.ftype, fs.params))
              for fs in spec.furniture
              if fs.ftype in ("table", "wardrobe", "nightstand", "tv_stand",
                              "bookshelf", "bed", "sofa", "counter")]
    lamp_pts = [(L.x, L.y) for L in spec.lights
                if L.fixture in ("table", "floor")]
    counter = [0]

    def new_clutter(params, parent, x, y, z, yaw, stack_on):
        base = _CNAME.get(params["kind"], "Decor")
        name = f"{base}_{counter[0]:04d}"
        counter[0] += 1
        spec.clutter.append(ClutterSpec(
            name=name, parent=parent, x=round(x, 4), y=round(y, 4),
            z=round(z, 4), yaw_deg=round(yaw, 2), stack_on=stack_on,
            params=params))
        return name

    _lay.clutter_supports(rng4, fitems, lamp_pts, new_clutter)


def build_ge0_spec(seed: int) -> GenesisSpec:
    """Fixed two-room skeleton, expressed through the same generic composer
    (kept as a stable regression fixture)."""
    rng = random.Random(seed)
    spec = GenesisSpec(seed=seed, kind="ge0")
    spec.height = rng.uniform(2.6, 2.9)
    w = rng.uniform(7.0, 9.0)
    d = rng.uniform(4.0, 5.5)
    split = rng.uniform(0.42, 0.58) * w
    _compose_rect_union(rng, spec, [(0, 0, split, d), (split, 0, w, d)])
    _materials_pass(spec)
    rng2 = random.Random(seed ^ 0x9E3779B9)
    _sky_and_exposure(rng2, spec)
    _backdrops(rng2, spec)
    _clutter_pass(spec)
    return spec
