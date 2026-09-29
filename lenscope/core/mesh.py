"""Triangle soup + ray casting (ground truth never touches the renderer).

Pure-numpy chunked Möller–Trumbore is the correctness-first reference (tests run
on any machine). Production speed path: embree via trimesh
(`use_embree=True`), same interface, cross-checked by tests.

Face attributes: sem (u8 NYU40), inst (u16), flags (u8 bitmask, spec.FLAG_*).
Glass faces are penetrated by `raycast_solid` (depth = first solid hit; a glass
hit sets flags|=GLASS on the pixel).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .spec import FLAG_GLASS, FLAG_MIRROR, FLAG_EMISSIVE, FLAG_PORTAL, FLAG_WINVIEW, SEM, SceneSpec


@dataclass
class TriSoup:
    verts: np.ndarray                      # (V,3) f32
    faces: np.ndarray                      # (F,3) i32
    sem: np.ndarray                        # (F,) u8
    inst: np.ndarray                       # (F,) u16
    flags: np.ndarray                      # (F,) u8
    _tri: np.ndarray = field(default=None, repr=False)

    @property
    def tri(self):
        if self._tri is None:
            self._tri = self.verts[self.faces].astype(np.float64)  # (F,3,3)
        return self._tri

    def merge(self, other: "TriSoup") -> "TriSoup":
        off = len(self.verts)
        return TriSoup(np.vstack([self.verts, other.verts]),
                       np.vstack([self.faces, other.faces + off]),
                       np.concatenate([self.sem, other.sem]),
                       np.concatenate([self.inst, other.inst]),
                       np.concatenate([self.flags, other.flags]))


def _quad(p0, p1, p2, p3):
    """Two triangles for quad p0-p1-p2-p3 (ccw)."""
    return np.array([p0, p1, p2, p3], np.float32), np.array([[0, 1, 2], [0, 2, 3]], np.int32)


def _box(cx, cy, z0, sx, sy, sz):
    x0, x1 = cx - sx / 2, cx + sx / 2
    y0, y1 = cy - sy / 2, cy + sy / 2
    z1 = z0 + sz
    v = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                  [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]], np.float32)
    f = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4],
                  [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]], np.int32)
    return v, f


def _slab(o, u, v, w):
    """Oriented box from origin o with edge vectors u (along), v (up), w (thickness,
    straddling o by +/-w/2). Same face topology as _box; used for the door leaf."""
    o = np.asarray(o, np.float64); u = np.asarray(u, np.float64)
    v = np.asarray(v, np.float64); w = np.asarray(w, np.float64)
    verts = np.array([o - w / 2, o + u - w / 2, o + u + w / 2, o + w / 2,
                      o - w / 2 + v, o + u - w / 2 + v, o + u + w / 2 + v, o + w / 2 + v], np.float32)
    f = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4],
                  [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]], np.int32)
    return verts, f


class SoupBuilder:
    def __init__(self):
        self.v, self.f, self.sem, self.inst, self.flags = [], [], [], [], []
        self._off = 0
        self._next_inst = 1

    def add(self, verts, faces, sem, flags=0, inst=None):
        if inst is None:
            inst = self._next_inst
            self._next_inst += 1
        self.v.append(np.asarray(verts, np.float32))
        self.f.append(np.asarray(faces, np.int32) + self._off)
        n = len(faces)
        self.sem.append(np.full(n, sem, np.uint8))
        self.inst.append(np.full(n, inst, np.uint16))
        self.flags.append(np.full(n, flags, np.uint8))
        self._off += len(verts)
        return inst

    def build(self) -> TriSoup:
        return TriSoup(np.vstack(self.v), np.vstack(self.f), np.concatenate(self.sem),
                       np.concatenate(self.inst), np.concatenate(self.flags))


def _wall_with_holes(b: SoupBuilder, axis, c, lo, hi, h, holes, sem=SEM["wall"]):
    """Wall plane at {axis}=c spanning [lo,hi] in the other axis, z in [0,h].
    holes: list of (u0,u1,z0,z1) rectangular openings. Emit sub-rectangles around holes."""
    # split the (u,z) rectangle into grid cells by hole boundaries, emit non-hole cells
    us = sorted({lo, hi, *[u for hole in holes for u in hole[:2]]})
    zs = sorted({0.0, h, *[z for hole in holes for z in hole[2:]]})
    for i in range(len(us) - 1):
        for j in range(len(zs) - 1):
            u0, u1, z0, z1 = us[i], us[i + 1], zs[j], zs[j + 1]
            if u1 - u0 < 1e-6 or z1 - z0 < 1e-6:
                continue
            mid_u, mid_z = (u0 + u1) / 2, (z0 + z1) / 2
            if any(h0 <= mid_u <= h1 and hz0 <= mid_z <= hz1 for h0, h1, hz0, hz1 in holes):
                continue
            if axis == "x":
                q = _quad([c, u0, z0], [c, u1, z0], [c, u1, z1], [c, u0, z1])
            else:
                q = _quad([u0, c, z0], [u1, c, z0], [u1, c, z1], [u0, c, z1])
            b.add(*q, sem=sem)


def build_from_spec(spec: SceneSpec) -> TriSoup:
    """Synthetic scaffold geometry: rooms as boxes with portal holes, door leaves
    (partially open), window glass panes, furniture boxes."""
    b = SoupBuilder()
    for ri, r in enumerate(spec.rooms):
        b.add(*_quad([r.x0, r.y0, 0], [r.x1, r.y0, 0], [r.x1, r.y1, 0], [r.x0, r.y1, 0]),
              sem=SEM["floor"])
        b.add(*_quad([r.x0, r.y0, r.h], [r.x0, r.y1, r.h], [r.x1, r.y1, r.h], [r.x1, r.y0, r.h]),
              sem=SEM["ceiling"])
        for axis, c, lo, hi in [("y", r.y0, r.x0, r.x1), ("y", r.y1, r.x0, r.x1),
                                ("x", r.x0, r.y0, r.y1), ("x", r.x1, r.y0, r.y1)]:
            holes = []
            for p in spec.portals:
                if ri not in (p.room_a, p.room_b):
                    continue
                if axis == "x" and abs(p.x0 - c) < 1e-6 and abs(p.x1 - c) < 1e-6:
                    holes.append((min(p.y0, p.y1), max(p.y0, p.y1), p.z0, p.z1))
                if axis == "y" and abs(p.y0 - c) < 1e-6 and abs(p.y1 - c) < 1e-6:
                    holes.append((min(p.x0, p.x1), max(p.x0, p.x1), p.z0, p.z1))
            _wall_with_holes(b, axis, c, lo, hi, r.h, holes)
    for p in spec.portals:
        if p.kind == "window":
            # glass pane in the opening (thin box), penetrated by the ray caster
            cx, cy = (p.x0 + p.x1) / 2, (p.y0 + p.y1) / 2
            sx = max(abs(p.x1 - p.x0), 0.02)
            sy = max(abs(p.y1 - p.y0), 0.02)
            v, f = _box(cx, cy, p.z0, sx, sy, p.z1 - p.z0)
            b.add(v, f, sem=SEM["window"], flags=FLAG_GLASS | FLAG_PORTAL)
        elif p.kind == "door" and p.open_deg < 180:
            # door leaf: solid slab with real thickness (not a zero-thickness,
            # single-sided quad), hinged at (x0,y0) and swung open_deg into room_a.
            w = float(np.hypot(p.x1 - p.x0, p.y1 - p.y0))
            ang = np.radians(p.open_deg)
            wall_dir = np.array([p.x1 - p.x0, p.y1 - p.y0, 0.0]) / max(w, 1e-9)
            wall_nrm = np.array([-wall_dir[1], wall_dir[0], 0.0])
            leaf_dir = np.cos(ang) * wall_dir + np.sin(ang) * wall_nrm     # swing direction
            leaf_nrm = np.array([-leaf_dir[1], leaf_dir[0], 0.0])          # leaf face normal (xy)
            hinge = np.array([p.x0, p.y0, p.z0])
            v, f = _slab(hinge, leaf_dir * w, np.array([0.0, 0.0, p.z1 - p.z0]), leaf_nrm * 0.045)
            b.add(v, f, sem=SEM["door"], flags=FLAG_PORTAL)
    for fu in spec.furniture:
        v, f = _box(fu.cx, fu.cy, fu.z0, fu.sx, fu.sy, fu.sz)
        b.add(v, f, sem=fu.sem_id, flags=fu.flags)
    return b.build()


# name-substring -> NYU40 id. `key in name.lower()`, first hit wins; default
# 39 = otherfurniture. Order matters: specific compound words must precede
# generic structural words, else the generic word shadows them (e.g. "wall"
# would map WallLamp to 1 instead of 35 and WallShelf to 1 instead of 15).
_NAME2SEM = [
    # specific compounds before generic structural words (avoids shadowing)
    ("wallshelf", 15), ("walllamp", 35), ("wallclock", 40),
    ("wallcabinet", 3),
    ("ceilinglight", 35),
    # compounds that would be shadowed by "floor"/"ceiling"
    ("floorplant", 40), ("ceilingfan", 40),
    # without these, FloorLamp would resolve to floor (2) and TableLamp to
    # table (7), painting lamp bases/poles with the floor/table class in every
    # semantic map. Compounds before the stems.
    ("floorlamp", 35), ("tablelamp", 35),
    # structure
    ("internalwall", 1), ("wall", 1), ("floor", 2), ("ceiling", 22),
    ("roomshell", 1),
    # furniture
    ("sofa", 6), ("couch", 6), ("bed", 4), ("chair", 5), ("stool", 5),
    ("table", 7), ("desk", 7), ("bookshelf", 15), ("shelf", 15),
    ("painting", 11), ("picture", 11), ("cabinet", 3), ("dresser", 17),
    ("television", 25), ("tv", 25), ("mirror", 19), ("curtain", 16),
    ("pillow", 18), ("towel", 27), ("box", 29), ("counter", 12),
    ("door", 8), ("window", 9), ("lamp", 35), ("light", 35),
    # rugs, decor, distractors
    ("rug", 20), ("decor", 40), ("distractor", 39),
    # second-generation scene objects (explicit, not default fall-through)
    ("backdrop", 39), ("groundplane", 39),
    # posters read as pictures; books map via "book" -> 23
    ("poster", 11), ("book", 23),
    # bathroom/kitchen: all map to existing NYU40 ids.
    # ("wallcabinet" lives in the top compounds group, since "wall" would
    # shadow it here.) The remaining stems have no earlier-stem collisions:
    ("kitchencounter", 12), ("rangehood", 38),
    ("toilet", 33), ("sink", 34), ("bathtub", 36), ("shower", 38),
    ("stove", 38), ("fridge", 24), ("refrigerator", 24), ("faucet", 34),
    # floor/hanging decor: existing NYU40 ids only
    # (bookstack hits the "book"->23 stem, mobiledecor hits "decor"->40;
    # floorplant/ceilingfan live in the top compounds group)
    ("hangplant", 40), ("plant", 40), ("basket", 40), ("suitcase", 37),
    ("lantern", 40),
]


def semantics_from_names(objects: list) -> np.ndarray:
    """Per-object NYU40 ids from fixture objects.json names."""
    sem = np.full(len(objects), 39, np.uint8)     # otherfurniture default
    for i, o in enumerate(objects):
        n = o["name"].lower()
        for key, sid in _NAME2SEM:
            if key in n:
                sem[i] = sid
                break
    return sem


def load_fixture(mesh_npz, objects_json=None) -> TriSoup:
    """Ingest a fixture dump (fixtures/export_fixture.py). With objects_json,
    per-face semantics are inferred from object names (name->NYU40)."""
    import json as _json
    z = np.load(mesh_npz)
    F = len(z["faces"])
    face_obj = z["face_obj"].astype(np.int64)
    sem = np.zeros(F, np.uint8)
    flags = np.zeros(F, np.uint8)
    if objects_json is not None:
        objs = _json.loads(Path(objects_json).read_text())
        obj_sem = semantics_from_names(objs)
        ok = (face_obj >= 0) & (face_obj < len(objs))
        sem[ok] = obj_sem[face_obj[ok]]
        # per-object material_flags (export_fixture: ["glass"|"mirror"|"emissive"]) -> per-face
        # bitmask, so that the glass-penetrating raycast_solid and the covisibility sampler see
        # glass, mirrors and emitters (with all-zero flags both would treat them as opaque).
        _FLAG_BY_NAME = {"glass": FLAG_GLASS, "mirror": FLAG_MIRROR, "emissive": FLAG_EMISSIVE}
        obj_flags = np.zeros(len(objs), np.uint8)
        for _i, _o in enumerate(objs):
            _b = 0
            for _fn in (_o.get("material_flags") or []):
                _b |= _FLAG_BY_NAME.get(_fn, 0)
            obj_flags[_i] = _b
        flags[ok] = obj_flags[face_obj[ok]]
    return TriSoup(z["verts"], z["faces"], sem,
                   face_obj.astype(np.uint16) + 1, flags)


# ray casting
def raycast(soup: TriSoup, origins, dirs, chunk=512, use_embree=False):
    """First-hit raycast. origins (N,3) or (3,), dirs (N,3) unit.
    Returns dict: t (N) f64 inf=miss, face (N) i64 -1=miss.
    use_embree: trimesh/embree fast path; numpy path is the reference."""
    dirs = np.atleast_2d(np.asarray(dirs, np.float64))
    origins = np.broadcast_to(np.atleast_2d(np.asarray(origins, np.float64)), dirs.shape)
    N = len(dirs)
    if use_embree:
        tm = getattr(soup, "_tm_cache", None)   # built once per soup: repeated
        if tm is None:                          # covis_pair/tri_pair calls are the hot path
            import trimesh
            tm = trimesh.Trimesh(vertices=soup.verts, faces=soup.faces, process=False)
            try:
                soup._tm_cache = tm
            except Exception:
                object.__setattr__(soup, "_tm_cache", tm)
        loc, ray_idx, tri_idx = tm.ray.intersects_location(origins, dirs, multiple_hits=False)
        t = np.full(N, np.inf)
        face = np.full(N, -1, np.int64)
        if len(ray_idx):
            t[ray_idx] = np.linalg.norm(loc - origins[ray_idx], axis=1)
            face[ray_idx] = tri_idx
        return {"t": t, "face": face}
    tri = soup.tri
    e1 = tri[:, 1] - tri[:, 0]
    e2 = tri[:, 2] - tri[:, 0]
    t_out = np.full(N, np.inf)
    f_out = np.full(N, -1, np.int64)
    for s in range(0, N, chunk):
        o = origins[s:s + chunk][:, None, :]         # (c,1,3)
        d = dirs[s:s + chunk][:, None, :]
        p = np.cross(d, e2[None])                    # (c,F,3)
        det = np.einsum("cfk,fk->cf", p, e1)
        inv = np.where(np.abs(det) > 1e-12, 1.0 / det, 0.0)
        tv = o - tri[None, :, 0]
        u = np.einsum("cfk,cfk->cf", tv, p) * inv
        q = np.cross(tv, e1[None])
        v = np.einsum("cfk,cfk->cf", q, d) * inv
        t = np.einsum("cfk,fk->cf", q, e2) * inv
        ok = (np.abs(det) > 1e-12) & (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1 + 1e-9) & (t > 1e-6)
        t = np.where(ok, t, np.inf)
        fmin = t.argmin(1)
        tmin = t[np.arange(len(t)), fmin]
        t_out[s:s + chunk] = tmin
        f_out[s:s + chunk] = np.where(np.isfinite(tmin), fmin, -1)
    return {"t": t_out, "face": f_out}


def raycast_solid(soup: TriSoup, origins, dirs, max_penetrations=4, use_embree=False):
    """First solid hit: glass faces are recorded (flags) then penetrated.
    Returns t, face, flags_accum (u8, glass bit set if any glass crossed)."""
    dirs = np.atleast_2d(np.asarray(dirs, np.float64))
    origins = np.broadcast_to(np.atleast_2d(np.asarray(origins, np.float64)), dirs.shape).copy()
    N = len(dirs)
    t_acc = np.zeros(N)
    fl_acc = np.zeros(N, np.uint8)
    t_out = np.full(N, np.inf)
    f_out = np.full(N, -1, np.int64)
    active = np.arange(N)
    o = origins.copy()
    for _ in range(max_penetrations + 1):
        if len(active) == 0:
            break
        r = raycast(soup, o[active], dirs[active], use_embree=use_embree)
        hit = np.isfinite(r["t"])
        idx = active[hit]
        faces = r["face"][hit]
        glass = (soup.flags[faces] & FLAG_GLASS) > 0
        # solid hits finalize
        sol = idx[~glass]
        t_out[sol] = t_acc[sol] + r["t"][hit][~glass]
        f_out[sol] = faces[~glass]
        # glass hits: mark + continue past
        g = idx[glass]
        fl_acc[g] |= FLAG_GLASS
        t_acc[g] += r["t"][hit][glass] + 1e-5
        o[g] = o[g] + dirs[g] * (r["t"][hit][glass] + 1e-5)[:, None]
        # misses finalize as miss
        active = g
    return {"t": t_out, "face": f_out, "flags_accum": fl_acc}
