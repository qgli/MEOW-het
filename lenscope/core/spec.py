"""SceneSpec: the declarative scene description consumed by the sampler and the ray caster.

Two producers: (a) fixture ingestion from real .blend dumps (fixtures/export_fixture.py
output + room inference below), (b) the synthetic apartment generator below, a
Blender-free test scaffold that produces geometry with analytically known
properties for unit tests and defect injection.

Coordinates: meters, z-up for the spec layer (rooms live in the xy plane, height z).
The camera convention (z-forward, y-down) applies to poses, not the world;
sampler/cast convert explicitly (see sampler.pose_to_R).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

# NYU40 ids used by the synthetic generator (subset)
SEM = {"void": 0, "wall": 1, "floor": 2, "cabinet": 3, "bed": 4, "chair": 5,
       "sofa": 6, "table": 7, "door": 8, "window": 9, "ceiling": 22, "lamp": 35}
FLAG_GLASS, FLAG_MIRROR, FLAG_EMISSIVE, FLAG_WINVIEW, FLAG_THIN, FLAG_PORTAL = 1, 2, 4, 8, 16, 32
# bit 64: BLK real-data inferred ghost-suspect (multi-station free-space
# contradiction, >=2 witnesses; lenscope/blk/analyze_holes.py --write-flags).
# Unlike bits 1-32 this is a detector output, not renderer ground truth.
FLAG_GHOST_SUSPECT = 64


@dataclass
class Portal:
    """Opening between room_a and room_b (or to outside if room_b < 0).
    Segment (x0,y0)-(x1,y1) on the shared wall, z from z0 to z1.
    kind: door (leaf asset, open_deg) | opening | window (glass pane)."""
    room_a: int
    room_b: int
    x0: float; y0: float; x1: float; y1: float
    z0: float; z1: float
    kind: str = "door"
    open_deg: float = 90.0


@dataclass
class Furniture:
    room: int
    sem_id: int
    # axis-aligned box (synthetic scaffold; real assets are meshes)
    cx: float; cy: float; z0: float
    sx: float; sy: float; sz: float
    flags: int = 0


@dataclass
class Room:
    """Rectangular room (synthetic scaffold): [x0,x1]x[y0,y1], floor z=0, height h."""
    x0: float; y0: float; x1: float; y1: float
    h: float = 2.7

    @property
    def area(self):
        return (self.x1 - self.x0) * (self.y1 - self.y0)


@dataclass
class SceneSpec:
    name: str
    seed: int
    rooms: list = field(default_factory=list)      # [Room]
    portals: list = field(default_factory=list)    # [Portal]
    furniture: list = field(default_factory=list)  # [Furniture]

    def save(self, path: Path):
        d = {"name": self.name, "seed": self.seed,
             "rooms": [asdict(r) for r in self.rooms],
             "portals": [asdict(p) for p in self.portals],
             "furniture": [asdict(f) for f in self.furniture]}
        Path(path).write_text(json.dumps(d, indent=1))

    @staticmethod
    def load(path: Path) -> "SceneSpec":
        d = json.loads(Path(path).read_text())
        return SceneSpec(name=d["name"], seed=d["seed"],
                         rooms=[Room(**r) for r in d["rooms"]],
                         portals=[Portal(**p) for p in d["portals"]],
                         furniture=[Furniture(**f) for f in d["furniture"]])


def synthetic_apartment(n_rooms: int = 3, seed: int = 0, furnish: bool = True) -> SceneSpec:
    """Row of connected rectangular rooms with doors (+1 window per room).
    Analytically simple (all boxes) so the ray-cast ground truth can be verified in closed form."""
    rng = np.random.default_rng(seed)
    rooms, portals, furn = [], [], []
    x = 0.0
    for i in range(n_rooms):
        w = float(rng.uniform(3.5, 6.0))
        d = float(rng.uniform(3.0, 5.0))
        rooms.append(Room(x0=x, y0=0.0, x1=x + w, y1=d, h=2.7))
        if i > 0:  # door on shared wall x=const
            yc = float(rng.uniform(1.0, min(d, rooms[i - 1].y1 - rooms[i - 1].y0) - 1.0))
            portals.append(Portal(room_a=i - 1, room_b=i, x0=x, y0=yc - 0.45,
                                  x1=x, y1=yc + 0.45, z0=0.0, z1=2.05, kind="door",
                                  open_deg=float(rng.uniform(70, 110))))
        # one window per room on the y=0 wall
        wx = float(rng.uniform(x + 0.8, x + w - 0.8))
        portals.append(Portal(room_a=i, room_b=-1, x0=wx - 0.6, y0=0.0, x1=wx + 0.6,
                              y1=0.0, z0=0.9, z1=2.1, kind="window"))
        if furnish:
            for _ in range(int(rng.integers(2, 5))):
                sx, sy = rng.uniform(0.5, 1.6, 2)
                sz = float(rng.uniform(0.4, 1.1))
                cx = float(rng.uniform(x + sx / 2 + 0.3, x + w - sx / 2 - 0.3))
                cy = float(rng.uniform(sy / 2 + 0.3, d - sy / 2 - 0.3))
                sem = int(rng.choice([SEM["table"], SEM["sofa"], SEM["cabinet"], SEM["bed"]]))
                furn.append(Furniture(room=i, sem_id=sem, cx=cx, cy=cy, z0=0.0,
                                      sx=float(sx), sy=float(sy), sz=sz))
            # one emissive lamp box per room (ambiguity supply)
            furn.append(Furniture(room=i, sem_id=SEM["lamp"], cx=x + w / 2, cy=d - 0.5,
                                  z0=1.6, sx=0.25, sy=0.25, sz=0.4, flags=FLAG_EMISSIVE))
        x += w
    return SceneSpec(name=f"synth_{n_rooms}r_s{seed}", seed=seed, rooms=rooms,
                     portals=portals, furniture=furn)


def spec_from_fixture(objects_json: Path, name: str, seed: int = 0) -> SceneSpec:
    """Derive a sampler-usable spec from a fixture dump (single-room approximation).

    One Room spans the floor AABB (walkable space is then carved implicitly:
    candidates inside furniture are excluded by the Furniture boxes, and
    unreachable/occluded positions are removed by the covisibility-degree
    guarantee). portals=[] (no bridge poses); the multi-room split is
    spec_from_fixture_v2. Sufficient to exercise the real-mesh
    covisibility/DOP/connectivity machinery end to end.
    """
    objs = json.loads(Path(objects_json).read_text())
    def aabb(o):
        lo, hi = o["aabb_world"]
        return np.asarray(lo, float), np.asarray(hi, float)
    floors = [o for o in objs if o["name"].lower().startswith("floor")]
    ceils = [o for o in objs if o["name"].lower().startswith("ceiling")]
    base = floors if floors else objs
    lo = np.min([aabb(o)[0] for o in base], axis=0)
    hi = np.max([aabb(o)[1] for o in base], axis=0)
    h = float(np.median([aabb(o)[1][2] for o in ceils])) if ceils else 2.7
    room = Room(x0=float(lo[0]), y0=float(lo[1]), x1=float(hi[0]), y1=float(hi[1]), h=h)
    skip = ("floor", "ceiling", "wall")
    room_area = room.area
    furn = []
    for o in objs:
        if o["name"].lower().startswith(skip):
            continue
        a, b = aabb(o)
        sx, sy, sz = (b - a).tolist()
        if sx * sy < 1e-4:            # degenerate/planar decor
            continue
        if sx * sy > 0.4 * room_area:  # scene-root/rug-scale AABB: structural, not an obstacle
            continue                   # (e.g. a 296 m^2 parent AABB would swallow all candidates)
        if a[2] > 1.9 or b[2] < 0.8:   # camera-height slab rule (see spec_from_fixture_v2)
            continue
        furn.append(Furniture(room=0, sem_id=0, cx=float((a[0] + b[0]) / 2),
                              cy=float((a[1] + b[1]) / 2), z0=float(a[2]),
                              sx=float(sx), sy=float(sy), sz=float(sz)))
    return SceneSpec(name=name, seed=seed, rooms=[room], portals=[], furniture=furn)


def spec_from_fixture_v2(objects_json: Path, name: str, seed: int = 0,
                         cell: float = 0.08, door_close: float = 0.45,
                         min_room_area: float = 2.0) -> SceneSpec:
    """Occupancy-grid room split and portal extraction for a fixture dump.

    Walls (name-prefix) are rasterized on a 2D grid; dilating them by
    ~door_close closes doorways, so free-space connected components become
    rooms; portals are then the original-free cells inside the dilated band
    whose neighbourhoods touch two different room labels (watershed on the
    room distance transform). Rooms enter the spec as component bounding boxes
    (sampler guarantees absorb the approximation); furniture is assigned to
    rooms by centroid. Unlike the single-room approximation, this provides
    portal bridge poses, so greedy seeds do not get trapped in wall pockets.
    """
    import scipy.ndimage as ndi

    objs = json.loads(Path(objects_json).read_text())
    def aabb(o):
        lo, hi = o["aabb_world"]
        return np.asarray(lo, float), np.asarray(hi, float)
    floors = [o for o in objs if o["name"].lower().startswith("floor")]
    ceils = [o for o in objs if o["name"].lower().startswith("ceiling")]
    def _is_wall(n):
        n = n.lower()
        return (n.startswith("internalwall")
                or (n.startswith("wall") and not n.startswith("wallshelf")))
    walls = [o for o in objs if _is_wall(o["name"])]
    base = floors if floors else objs
    lo = np.min([aabb(o)[0] for o in base], axis=0)
    hi = np.max([aabb(o)[1] for o in base], axis=0)
    h = float(np.median([aabb(o)[1][2] for o in ceils])) if ceils else 2.7

    nx = max(int(np.ceil((hi[0] - lo[0]) / cell)), 8)
    ny = max(int(np.ceil((hi[1] - lo[1]) / cell)), 8)
    occ = np.zeros((nx, ny), bool)
    for o in walls:
        a, b = aabb(o)
        # walking-band z-slab filter: only wall boxes intersecting z in
        # [0.2, 1.8] block the 2D grid. Real doors have headers (wall above
        # ~2.05m) whose full-AABB projection would seal the doorway (no
        # portals on second-generation scenes). First-generation walls span
        # floor to ceiling, so they all intersect the band and their rooms and
        # portals are unchanged.
        if a[2] > 1.8 or b[2] < 0.2:
            continue
        i0, i1 = int((a[0] - lo[0]) / cell), int(np.ceil((b[0] - lo[0]) / cell))
        j0, j1 = int((a[1] - lo[1]) / cell), int(np.ceil((b[1] - lo[1]) / cell))
        occ[max(i0, 0):min(i1 + 1, nx), max(j0, 0):min(j1 + 1, ny)] = True
    free = ~occ
    # adaptive door sealing: first-generation doorways vary ~0.9-1.4m; escalate the wall
    # dilation until the free space splits (>=2 rooms), else fall back single-room
    lab, keep, it = None, [], 1
    for dc in (door_close, 0.60, 0.75, 0.90):
        it = max(int(round(dc / cell)), 1)
        closed = ~ndi.binary_dilation(occ, iterations=it)
        lab, n_lab = ndi.label(closed)
        keep = [k for k in range(1, n_lab + 1)
                if (lab == k).sum() * cell * cell >= min_room_area]
        if len(keep) >= 2:
            break
    if not keep:                                            # degenerate: fall back
        return spec_from_fixture(objects_json, name, seed=seed)
    remap = {k: i for i, k in enumerate(keep)}
    # nearest-room assignment for every free cell (watershed by EDT per room)
    dist = np.full((len(keep),) + lab.shape, np.inf)
    for i, k in enumerate(keep):
        dist[i] = ndi.distance_transform_edt(lab != k)
    nearest = np.argmin(dist, axis=0)
    # portal cells: free in the original map, not in any sealed room, and with
    # >=2 distinct nearest-rooms within a small window
    band = free & (lab == 0)
    pi, pj = np.where(band)
    portal_pts, portal_rooms = [], []
    for i, j in zip(pi, pj):
        w = nearest[max(i - it, 0):i + it + 1, max(j - it, 0):j + it + 1]
        m = closed[max(i - it, 0):i + it + 1, max(j - it, 0):j + it + 1]
        rooms_near = np.unique(w[m])
        if len(rooms_near) >= 2:
            portal_pts.append((i, j))
            portal_rooms.append(tuple(sorted(rooms_near[:2])))
    portals = []
    if portal_pts:
        pts = np.asarray(portal_pts)
        plab, n_p = ndi.label(np.isin(np.arange(nx * ny).reshape(nx, ny),
                                      pts[:, 0] * ny + pts[:, 1]))
        for k in range(1, n_p + 1):
            ii, jj = np.where(plab == k)
            if len(ii) < 3:
                continue
            xy = np.stack([lo[0] + (ii + 0.5) * cell, lo[1] + (jj + 0.5) * cell], 1)
            c = xy.mean(0)
            d = xy - c
            v = np.linalg.svd(d, full_matrices=False)[2][0]
            ext = d @ v
            p0, p1 = c + v * ext.min(), c + v * ext.max()
            # majority room pair over this cluster's cells (taking the first
            # cell's pair would label every portal with the same pair, and
            # bridges would lose their room identity on multi-door scenes)
            keys = set(zip(ii.tolist(), jj.tolist()))
            pairs = [pr for pt_ij, pr in zip(map(tuple, pts), portal_rooms)
                     if pt_ij in keys]
            from collections import Counter
            rooms_pair = Counter(pairs).most_common(1)[0][0] if pairs else (0, 0)
            ra, rb = (remap.get(r, 0) for r in rooms_pair)
            portals.append(Portal(room_a=int(ra), room_b=int(rb),
                                  x0=float(p0[0]), y0=float(p0[1]),
                                  x1=float(p1[0]), y1=float(p1[1]),
                                  z0=0.0, z1=min(2.0, h - 0.2), kind="opening"))
    rooms = []
    for k in keep:
        ii, jj = np.where(lab == k)
        rooms.append(Room(x0=float(lo[0] + ii.min() * cell), y0=float(lo[1] + jj.min() * cell),
                          x1=float(lo[0] + (ii.max() + 1) * cell), y1=float(lo[1] + (jj.max() + 1) * cell),
                          h=h))
    # furniture (same filters as spec_from_fixture), assigned to rooms by centroid
    room_area = (hi[0] - lo[0]) * (hi[1] - lo[1])
    skip = ("floor", "ceiling", "roomshell")
    furn = []
    for o in objs:
        if o["name"].lower().startswith(skip) or _is_wall(o["name"]):
            continue
        a, b = aabb(o)
        sx, sy, sz = (b - a).tolist()
        if sx * sy < 1e-4 or sx * sy > 0.4 * room_area:
            continue
        # blocking = the box overlaps the camera height slab [0.8, 1.9] m.
        # Cameras are not walkers: low clutter (tables) and floaters above head
        # (FloatDistractor, ceiling lamps) must not eat candidate space (with
        # distractors, a footprint-only rule cut the pose density from 125 to
        # 31 on one fixture).
        if a[2] > 1.9 or b[2] < 0.8:
            continue
        cx, cy = float((a[0] + b[0]) / 2), float((a[1] + b[1]) / 2)
        gi = int(np.clip((cx - lo[0]) / cell, 0, nx - 1))
        gj = int(np.clip((cy - lo[1]) / cell, 0, ny - 1))
        furn.append(Furniture(room=int(nearest[gi, gj]), sem_id=0, cx=cx, cy=cy,
                              z0=float(a[2]), sx=float(sx), sy=float(sy), sz=float(sz)))
    return SceneSpec(name=name, seed=seed, rooms=rooms, portals=portals, furniture=furn)
