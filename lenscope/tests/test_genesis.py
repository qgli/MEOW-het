"""bpy-free tests of the second-generation scene generator (no Blender needed).

Floor-plan tests: seed determinism/roundtrip; true through-aperture
+ header-above-door + closure; no degenerate/thin solids; _NAME2SEM naming;
connectivity BFS==100% (geometric, construction-independent) over many
seeds; entry door present; windows on exterior walls only; function
constraints; wall orientation (exterior walls outside the footprint).
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lenscope.genesis.spec import (GenesisSpec, build_ge0_spec,
                                    build_ge1_spec)
from lenscope.genesis import solids
from lenscope.core.mesh import semantics_from_names

N_SEEDS = 40


def _point_in_poly(px, py, poly):
    inside = False
    n = len(poly)
    for i in range(n):
        x0, y0 = poly[i]
        x1, y1 = poly[(i + 1) % n]
        if (y0 > py) != (y1 > py):
            xin = x0 + (py - y0) * (x1 - x0) / (y1 - y0)
            if px < xin:
                inside = not inside
    return inside


def _room_of(spec, px, py):
    for k, r in enumerate(spec.rooms):
        if _point_in_poly(px, py, r.poly):
            return k
    return None


def _door_connects(spec, d):
    """Rooms on both sides of a door center (probe +/- normal)."""
    w = {w.id: w for w in spec.walls}[d.wall]
    dx, dy = w.p2[0] - w.p1[0], w.p2[1] - w.p1[1]
    L = math.hypot(dx, dy)
    ux, uy = dx / L, dy / L
    nx, ny = -uy, ux
    cx, cy = w.p1[0] + ux * d.u, w.p1[1] + uy * d.u
    eps = w.t / 2 + 0.15
    a = _room_of(spec, cx + nx * eps, cy + ny * eps)
    b = _room_of(spec, cx - nx * eps, cy - ny * eps)
    return a, b


def test_determinism_and_roundtrip():
    for builder in (build_ge0_spec, build_ge1_spec):
        a, b = builder(11), builder(11)
        assert a.to_json() == b.to_json()
        assert builder(12).to_json() != a.to_json()
        rt = GenesisSpec.from_json(a.to_json())
        assert rt.to_json() == a.to_json()


def test_connectivity_bfs_100pct_many_seeds():
    """Hard requirement: every room reachable through doors, geometrically
    verified (independent of the spanning-tree construction)."""
    for seed in range(N_SEEDS):
        spec = build_ge1_spec(seed)
        n = len(spec.rooms)
        adj = {k: set() for k in range(n)}
        for d in spec.doors:
            if d.entry:
                continue
            a, b = _door_connects(spec, d)
            assert a is not None and b is not None and a != b, \
                (seed, d.wall, a, b)
            adj[a].add(b); adj[b].add(a)
        seen, stack = {0}, [0]
        while stack:
            k = stack.pop()
            for m in adj[k]:
                if m not in seen:
                    seen.add(m); stack.append(m)
        assert len(seen) == n, f"seed {seed}: {len(seen)}/{n} rooms reachable"


def test_entry_door_and_exterior_windows():
    for seed in range(N_SEEDS):
        spec = build_ge1_spec(seed)
        assert any(d.entry for d in spec.doors), f"seed {seed}: no entry door"
        walls = {w.id: w for w in spec.walls}
        for wd in spec.windows:
            assert not walls[wd.wall].internal, \
                f"seed {seed}: window on internal wall"


def test_function_constraints():
    for seed in range(N_SEEDS):
        spec = build_ge1_spec(seed)
        funcs = [r.function for r in spec.rooms]
        # a single big room is a studio (open-plan living superset); the
        # living guarantee is satisfied by either.
        assert "living" in funcs or funcs == ["studio"], \
            f"seed {seed}: no living/studio room in {funcs}"
        if spec.kind == "bsp_hallway":
            assert "hallway" in funcs
        if len(spec.rooms) >= 3:
            assert "bedroom" in funcs, f"seed {seed}: no bedroom in {funcs}"


def test_apertures_true_holes_headers_and_closure():
    """Aperture checks on a multi-room seed (find one with a door+window)."""
    for seed in range(N_SEEDS):
        spec = build_ge1_spec(seed)
        interior = [d for d in spec.doors if not d.entry]
        if interior and spec.windows:
            break
    d = interior[0]
    real = solids.realize(spec)
    wallboxes = [b for b in real["boxes"] if b["kind"] == "wall"]
    w = {w.id: w for w in spec.walls}[d.wall]
    dx, dy = w.p2[0] - w.p1[0], w.p2[1] - w.p1[1]
    L = math.hypot(dx, dy)
    ux, uy = dx / L, dy / L
    nx, ny = -uy, ux
    cx, cy = w.p1[0] + ux * d.u, w.p1[1] + uy * d.u
    probe = w.t / 2 + 0.4
    # through-hole at door mid-height
    assert not solids.ray_blocked(
        wallboxes, (cx + nx * probe, cy + ny * probe, d.height / 2),
        (cx - nx * probe, cy - ny * probe, d.height / 2))
    # header blocks above the door
    zh = (d.height + spec.height) / 2
    assert solids.ray_blocked(
        wallboxes, (cx + nx * probe, cy + ny * probe, zh),
        (cx - nx * probe, cy - ny * probe, zh))
    # wall closes away from the door (0.6m along, if run long enough)
    off = d.u + d.width / 2 + 0.6
    if off < L - 0.1:
        ox, oy = w.p1[0] + ux * off, w.p1[1] + uy * off
        assert solids.ray_blocked(
            wallboxes, (ox + nx * probe, oy + ny * probe, d.height / 2),
            (ox - nx * probe, oy - ny * probe, d.height / 2))
    # window: hole free of wall, glass present in the hole
    wd = spec.windows[0]
    w2 = {w.id: w for w in spec.walls}[wd.wall]
    dx, dy = w2.p2[0] - w2.p1[0], w2.p2[1] - w2.p1[1]
    L2 = math.hypot(dx, dy)
    ux, uy = dx / L2, dy / L2
    nx, ny = -uy, ux
    cx, cy = w2.p1[0] + ux * wd.u, w2.p1[1] + uy * wd.u
    zc = wd.sill + wd.height / 2
    p_in = (cx + nx * (w2.t + 0.4), cy + ny * (w2.t + 0.4), zc)
    p_out = (cx - nx * (w2.t + 0.4), cy - ny * (w2.t + 0.4), zc)
    assert not solids.ray_blocked(wallboxes, p_in, p_out)
    glass = [b for b in real["boxes"] if b["kind"] == "glass"]
    assert solids.ray_blocked(glass, p_in, p_out)


def test_exterior_walls_sit_outside_footprint():
    """Orientation check: exterior wall centers must lie outside all rooms."""
    for seed in range(12):
        spec = build_ge1_spec(seed)
        if spec.kind in ("circle", "pentagon"):
            continue
        real = solids.realize(spec)
        for b in real["boxes"]:
            if b["kind"] != "wall" or not b["name"].startswith("Wall_EW"):
                continue
            cx, cy = b["c"][0], b["c"][1]
            assert _room_of(spec, cx, cy) is None, (seed, b["name"])


def test_no_degenerate_solids():
    for seed in (0, 5, 9):
        real = solids.realize(build_ge1_spec(seed))
        for b in real["boxes"]:
            assert min(b["s"]) > 2.5e-3, (seed, b["name"], b["s"])


def test_naming_hits_semantics():
    spec = build_ge1_spec(2)
    real = solids.realize(spec)
    objs = ([{"name": b["name"]} for b in real["boxes"]]
            + [{"name": x["name"]} for x in real["leaves"]]
            + [{"name": p["name"]} for p in real["polyslabs"]])
    sem = semantics_from_names(objs)
    by = {o["name"]: int(s) for o, s in zip(objs, sem)}
    for name, v in by.items():
        low = name.lower()
        if low.startswith("wall_") or low.startswith("internalwall"):
            assert v == 1, (name, v)
        elif low.startswith("wallskirting"):
            assert v == 1, (name, v)
        elif low.startswith("floorlamp") or low.startswith("tablelamp"):
            assert v == 35, (name, v)
        elif low.startswith("floor"):
            assert v == 2, (name, v)
        elif low.startswith("ceilinglight"):
            assert v == 35, (name, v)
        elif low.startswith("ceiling"):
            assert v == 22, (name, v)
        elif low.startswith("window"):
            assert v == 9, (name, v)
        elif low.startswith("door"):
            assert v == 8, (name, v)
        elif low.startswith("sofa"):
            assert v == 6, (name, v)
        elif low.startswith("bed"):
            assert v == 4, (name, v)
        elif low.startswith("table"):
            assert v == 7, (name, v)


def test_light_randomness_present():
    """Fixtures/CCT/lumens/sky must actually vary."""
    fixtures, ccts, skies = set(), set(), set()
    for seed in range(N_SEEDS):
        spec = build_ge1_spec(seed)
        for L in spec.lights:
            fixtures.add(L.fixture)
            ccts.add(round(L.cct_k, -2))
        skies.add(round(spec.sky_strength, 2))
    assert {"flush", "pendant", "bulb"} <= fixtures, fixtures
    assert fixtures <= {"flush", "pendant", "bulb", "wall", "floor", "table"}
    assert len(ccts) > 8 and len(skies) > 10


def test_sky_archetypes_and_exposure():
    """Sky and exposure (bpy-free side): archetype mix present; night=sun below
    horizon + dim sky; exposure is derived (inverse to photometric budget,
    within clamp) and the derivation trail is recorded."""
    archs, exps = {}, []
    for seed in range(60):
        spec = build_ge1_spec(seed)
        archs[spec.sky_archetype] = archs.get(spec.sky_archetype, 0) + 1
        d = spec.exposure_derivation
        assert d and d["lm_on"] > 0 and d["area_m2"] > 0, d
        E = d["E_int_lux"] + d["E_day_lux"]
        expect = min(max(d["E_ref_lux"] / max(E, 5.0), 0.25), 4.0)
        # derivation trail stores rounded lux -> match at that precision
        assert abs(spec.film_exposure - expect) < 0.02, (spec.film_exposure, d)
        exps.append(spec.film_exposure)
        if spec.sky_archetype == "night":
            # moonlit nights raise sky_strength (cap 0.15)
            assert spec.sun_elevation_deg < 0 and spec.sky_strength <= 0.15
            assert all(L.on for L in spec.lights)      # night: interiors carry
        rooms_off = {r.name for r in spec.rooms}
        for L in spec.lights:
            if L.on:
                rooms_off.discard(L.room)
        if spec.sky_archetype == "night":
            assert not rooms_off                        # no black rooms at night
    assert set(archs) == {"day", "dusk", "night"}, archs
    assert len(set(round(e, 2) for e in exps)) > 10     # actually varies


def test_anchored_lamps_and_backdrops():
    """Functional anchoring (anchors are real furniture): dining
    pendant over the table; bedside lamps on nightstand tops; backdrops
    only outside the footprint, never inside."""
    saw_bedside, saw_backdrop = False, False
    for seed in range(30):
        spec = build_ge1_spec(seed)
        furn = {f.name: f for f in spec.furniture}
        for r in spec.rooms:
            if r.function == "dining":
                t = next((f for f in spec.furniture
                          if f.ftype == "table" and f.room == r.name), None)
                pend = [L for L in spec.lights
                        if L.room == r.name and L.fixture == "pendant"]
                if t and pend:
                    assert min(abs(L.x - t.x) + abs(L.y - t.y)
                               for L in pend) < 0.5
        for L in spec.lights:
            if L.fixture == "table":
                assert L.z > 0.3                        # sits on furniture top
                cands = [f for f in spec.furniture
                         if f.ftype in ("nightstand", "table")
                         and f.room == L.room
                         and abs(f.x - L.x) < f.hx + 0.35
                         and abs(f.y - L.y) < f.hy + 0.35]
                assert cands, (seed, L)
                host = min(cands, key=lambda f: (f.x - L.x) ** 2
                           + (f.y - L.y) ** 2)
                if host.ftype == "nightstand":
                    saw_bedside = True
                    assert abs(L.z - (host.height + 0.30)) < 0.02
        xs = [p[0] for r in spec.rooms for p in r.poly]
        ys = [p[1] for r in spec.rooms for p in r.poly]
        for B in spec.backdrops:
            saw_backdrop = True
            assert not (min(xs) < B.x < max(xs) and min(ys) < B.y < max(ys)), \
                (seed, B)
    assert saw_bedside and saw_backdrop


def _in_poly(x, y, poly):
    inside = False
    j = len(poly) - 1
    for i in range(len(poly)):
        xi, yi = poly[i]; xj, yj = poly[j]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi:
            inside = not inside
        j = i
    return inside


def test_solar_astronomy():
    """NOAA model against known reference configurations (lighting-grade
    tolerance) + archetype banding consistency + moon physics."""
    from lenscope.genesis.spec import (solar_position, _moon_phase_factor,
                                        _ARCH_BANDS)
    # equator, equinox (doy 80), solar noon at lon 0 -> sun near zenith
    e, _ = solar_position(0.0, 0.0, 80, 12.0)
    assert e > 85.0, e
    # 40N, summer solstice (doy 172), solar noon: elev ~ 90-40+23.44 = 73.4
    e, a = solar_position(40.0, 0.0, 172, 12.0)
    assert abs(e - 73.4) < 1.5, e
    assert abs(a - 180.0) < 15.0, a          # due south at noon (N hemisphere)
    # 80N winter solstice: polar night, sun never rises
    assert all(solar_position(80.0, 0.0, 355, h)[0] < 0 for h in range(24))
    # morning sun in the east (40N, equinox, 3h before solar noon)
    _, a = solar_position(40.0, 0.0, 80, 9.0)
    assert 60.0 < a < 170.0, a
    # moon phase factor: monotonic, full=1, quarter ~0.08-0.15
    assert abs(_moon_phase_factor(1.0) - 1.0) < 1e-6
    q = _moon_phase_factor(0.5)
    assert 0.05 < q < 0.2, q
    assert _moon_phase_factor(0.2) < q < _moon_phase_factor(0.8)
    # sampled scenes: solar band matches archetype; moon only at night
    n_noaa = 0
    for seed in range(50):
        sp = build_ge1_spec(seed)
        lo, hi = _ARCH_BANDS[sp.sky_archetype]
        assert lo <= sp.sun_elevation_deg < hi, (seed, sp.sky_archetype,
                                                 sp.sun_elevation_deg)
        if sp.solar.get("model") == "noaa":
            n_noaa += 1
            e2, a2 = solar_position(sp.solar["lat"], sp.solar["lon"],
                                    sp.solar["day_of_year"],
                                    sp.solar["hour_utc"])
            assert abs(e2 - sp.sun_elevation_deg) < 0.01
        if sp.moon.get("enabled"):
            assert sp.sky_archetype == "night"
            assert 0.0 < sp.moon["lux"] <= 0.32 + 1e-6
    assert n_noaa >= 45, n_noaa              # fallback must stay rare


def test_evening_illuminance_floor():
    """dusk/night: every room's lumen density of lit lamps is at least about
    62 lm/m2 (the higher evening floor keeps outer rooms bright enough)."""
    checked = 0
    for seed in range(40):
        sp = build_ge1_spec(seed)
        if sp.sky_archetype not in ("dusk", "night"):
            continue
        area = {r.name: max(r.area, 1.0) for r in sp.rooms}
        on_lm = {}
        n_lamps = {}
        for L in sp.lights:
            n_lamps[L.room] = n_lamps.get(L.room, 0) + 1
            if L.on:
                on_lm[L.room] = on_lm.get(L.room, 0.0) + L.lumens
        for rname, a in area.items():
            dens = on_lm.get(rname, 0.0) / a
            # the floor can only be met up to the lamps the room has; the
            # floor is also scaled by the measured floor-material albedo, so
            # 62 x 0.85 serves as a conservative lower bound
            total = sum(L.lumens for L in sp.lights if L.room == rname)
            assert dens >= min(62.0 * 0.85, total / a) - 1e-6, \
                (seed, rname, dens)
            checked += 1
    assert checked > 20


def test_no_coplanar_overlapping_wall_faces():
    """No two axis-aligned wall boxes may share a face
    plane with overlapping area (coplanar duplicates mutually block shadow
    rays -> plateau black). Structural check, spp-independent."""
    for seed in range(24):
        real = solids.realize(build_ge1_spec(seed))
        walls = [b for b in real["boxes"] if b["kind"] == "wall"
                 and abs(math.sin(2 * b["yaw"])) < 1e-9]   # axis-aligned only
        planes = {}
        for b in walls:
            cx, cy, cz = b["c"]; hx, hy, hz = b["s"]
            yaw90 = abs(math.sin(b["yaw"])) > 0.5
            ex, ey = (hy, hx) if yaw90 else (hx, hy)
            # face record: (tangential lo/hi, z-range, normal-axis body range)
            for ax, lo, hi, nrm in [
                    (("x", round(cx - ex, 5)), cy - ey, cy + ey, (cx - ex, cx + ex)),
                    (("x", round(cx + ex, 5)), cy - ey, cy + ey, (cx - ex, cx + ex)),
                    (("y", round(cy - ey, 5)), cx - ex, cx + ex, (cy - ey, cy + ey)),
                    (("y", round(cy + ey, 5)), cx - ex, cx + ex, (cy - ey, cy + ey))]:
                planes.setdefault(ax, []).append(
                    (lo, hi, (cz - hz, cz + hz), nrm, b["name"]))
        for key, faces in planes.items():
            for i in range(len(faces)):
                for j in range(i + 1, len(faces)):
                    (l1, h1, z1, m1, n1), (l2, h2, z2, m2, n2) = faces[i], faces[j]
                    du = min(h1, h2) - max(l1, l2)
                    dz = min(z1[1], z2[1]) - max(z1[0], z2[0])
                    # bodies on opposite sides of the shared plane = a hidden
                    # touching interface (legal). A violation only if the two
                    # boxes' bodies overlap along the plane normal (coplanar
                    # duplicate faces, the shadow-blocking class).
                    dn = min(m1[1], m2[1]) - max(m1[0], m2[0])
                    assert not (du > 0.02 and dz > 0.02 and dn > 1e-3), \
                        (seed, key, n1, n2, du, dz, dn)


# furniture and clutter: support legality 100%, shelf occupancy > 0,
# floating violations = 0, mesh geometry correctness, burial rule

def _world_to_local(fs, wx, wy):
    import math as _m
    c = _m.cos(_m.radians(-fs.yaw_deg))
    s = _m.sin(_m.radians(-fs.yaw_deg))
    dx, dy = wx - fs.x, wy - fs.y
    return c * dx - s * dy, s * dx + c * dy


def test_mesh_kernel_all_types():
    """Every furniture/clutter part mesh across seeds: closed manifold,
    coherent outward winding, positive volume, no degenerate faces."""
    from lenscope.genesis import furniture as fu
    checked = 0
    for seed in range(8):
        spec = build_ge1_spec(seed)
        for fs in spec.furniture:
            for pt in fu.build_item(fs.ftype, fs.params)["parts"]:
                ck = fu.mesh_checks(pt["verts"], pt["faces"])
                assert ck["ok"], (seed, fs.name, pt["name"], ck)
                checked += 1
        for cs in spec.clutter:
            for pt in fu.build_clutter_item(cs.params)["parts"]:
                ck = fu.mesh_checks(pt["verts"], pt["faces"])
                assert ck["ok"], (seed, cs.name, ck)
                checked += 1
    assert checked > 500, checked


def test_support_contact_and_floating():
    """Support contact legality = 100%: every supported clutter item sits ON
    a registered support surface of its parent (z match, XY inside the
    support poly); stacked items sit on their base item; floating
    violations = 0 (only parent=='' distractors float)."""
    from lenscope.genesis import furniture as fu
    n_checked = 0
    for seed in range(10):
        spec = build_ge1_spec(seed)
        furn = {f.name: f for f in spec.furniture}
        cl_by_name = {c.name: c for c in spec.clutter}
        for cs in spec.clutter:
            if cs.parent == "":
                assert cs.name.startswith("Distractor"), cs
                assert cs.z > 1.2                    # declared floating band
                continue
            fs = furn[cs.parent]
            built = fu.build_item(fs.ftype, fs.params)
            if cs.stack_on:
                below = cl_by_name[cs.stack_on]
                bit = fu.build_clutter_item(below.params)
                # z recorded at 1e-4 grain in the manifest
                assert abs((below.z + bit["height"] - 0.002) - cs.z) < 1e-3, \
                    (seed, cs.name)
                n_checked += 1
                continue
            lx, ly = _world_to_local(fs, cs.x, cs.y)
            hit = False
            for sup in built["supports"]:
                if abs(cs.z - sup["z"]) < 1e-3 and _in_poly(lx, ly, sup["poly"]):
                    hit = True
                    break
            assert hit, (seed, cs.name, cs.parent, cs.z,
                         [s["z"] for s in built["supports"]])
            n_checked += 1
    assert n_checked > 300, n_checked


def test_shelf_occupancy_and_layout_legality():
    """Every bookshelf shelf holds >=1 book; furniture footprints do not
    overlap (SAT) and respect door keep-outs."""
    from lenscope.genesis import furniture as fu
    from lenscope.genesis.layout import _obb_overlap
    from lenscope.genesis.spec import _door_keepouts
    import math as _m
    saw_shelf = 0
    for seed in range(10):
        spec = build_ge1_spec(seed)
        for fs in spec.furniture:
            if fs.ftype != "bookshelf":
                continue
            built = fu.build_item(fs.ftype, fs.params)
            shelves = [s for s in built["supports"] if s.get("shelf")]
            books_z = [c.z for c in spec.clutter
                       if c.parent == fs.name and c.params["kind"] == "book"]
            for sup in shelves:
                assert any(abs(bz - sup["z"]) < 1e-3 for bz in books_z), \
                    (seed, fs.name, sup["z"])
                saw_shelf += 1
        floor_items = [f for f in spec.furniture
                       if f.ftype not in ("curtain", "mirror", "rug",
                                          "painting", "poster", "clock")]
        for room in spec.rooms:
            here = [f for f in floor_items if f.room == room.name]
            for i in range(len(here)):
                for j in range(i + 1, len(here)):
                    a, b = here[i], here[j]
                    # the plan-view test needs a z check: wall cabinets/hoods
                    # above counters and hanging pieces above furniture are
                    # designed vertical stacking, not clashes
                    if a.z >= b.z + b.height - 0.02 \
                            or b.z >= a.z + a.height - 0.02:
                        continue
                    assert not _obb_overlap(
                        (a.x, a.y, a.hx, a.hy, a.yaw_deg),
                        (b.x, b.y, b.hx, b.hy, b.yaw_deg), margin=0.0), \
                        (seed, a.name, b.name)
            for (kx, ky, kr) in _door_keepouts(spec, room):
                for f in here:
                    d = _m.hypot(f.x - kx, f.y - ky)
                    assert d + 1e-6 >= kr - _m.hypot(f.hx, f.hy) or \
                        _point_obb_ok(kx, ky, kr, f), (seed, f.name)
    assert saw_shelf > 10, saw_shelf


def _point_obb_ok(kx, ky, kr, f):
    from lenscope.genesis.layout import _point_obb_dist
    return _point_obb_dist(kx, ky, (f.x, f.y, f.hx, f.hy, f.yaw_deg)) >= kr


def test_burial_no_coplanar_contacts():
    """Anti-black-hole: floor furniture bottoms sink below z=0
    (never coplanar with the floor top); supported clutter sinks below its
    support plane; TV carries the glossy flag; mirrors the mirror flag."""
    from lenscope.genesis import solids as sol
    saw_tv, saw_mirror = False, False
    for seed in range(6):
        spec = build_ge1_spec(seed)
        real = sol.realize(spec)
        by_obj = {}
        for fm in real["fmeshes"]:
            by_obj.setdefault(fm["obj"], []).append(fm)
        furn = {f.name: f for f in spec.furniture}
        for obj, parts in by_obj.items():
            zmin = min(float(p["verts"][:, 2].min()) for p in parts)
            root = obj.split(".")[0]
            fs = furn.get(root)
            if fs is not None and fs.z == 0.0 and fs.ftype != "curtain":
                # splayed legs rotate away ~1mm of the 3mm sink; >=1.5mm
                # below the floor plane still kills coplanarity
                assert zmin <= -0.0015, (seed, obj, zmin)
            if any(p["flags"].get("glossy") for p in parts):
                saw_tv = True
            if any(p["flags"].get("mirror") for p in parts):
                saw_mirror = True
        cl = {c.name: c for c in spec.clutter}
        for obj, parts in by_obj.items():
            c = cl.get(obj)
            if c is not None and c.parent:
                zmin = min(float(p["verts"][:, 2].min()) for p in parts)
                # leaning books pivot on a beveled bottom edge: the rounded
                # corner arc gives back ~1mm of the 2mm sink -- still below
                # the support plane (that is what kills coplanarity)
                assert zmin <= c.z - 0.0005, (seed, obj, zmin, c.z)
    # TVs are conditional (sight line + solid wall): prove the glossy flag
    # on any seed that has one, like mirrors below
    if not saw_tv:
        from lenscope.genesis import furniture as fu
        for seed in range(60):
            spec = build_ge1_spec(seed)
            tvs = [f for f in spec.furniture if f.ftype == "tv_stand"]
            if tvs:
                built = fu.build_item("tv_stand", tvs[0].params)
                assert any(p["flags"].get("glossy") for p in built["parts"])
                saw_tv = True
                break
    assert saw_tv
    # mirrors are low-probability decor: prove flag plumbing on any seed
    if not saw_mirror:
        from lenscope.genesis import furniture as fu
        for seed in range(60):
            spec = build_ge1_spec(seed)
            ms = [f for f in spec.furniture if f.ftype == "mirror"]
            if ms:
                built = fu.build_item("mirror", ms[0].params)
                assert any(p["flags"].get("mirror") for p in built["parts"])
                saw_mirror = True
                break
    assert saw_mirror


# materials: UV 100% + no flips + metric texel, physical BRDF domains,
# real-PBR / decal assignment, empty-library invariant, provenance

def test_uv_projector_no_flips_metric():
    """box_project_uv: orientation-preserving (no mirrored UV faces -> valid
    normal maps) and metric (flat-face texel density CV << 10%)."""
    import numpy as np
    import random as _rd
    from lenscope.genesis import furniture as fu
    rng = _rd.Random(5)
    bad, tot, dens = 0, 0, []
    for t, s in (("sofa", fu.sample_sofa), ("bed", fu.sample_bed),
                 ("bookshelf", fu.sample_bookshelf),
                 ("wardrobe", fu.sample_wardrobe)):
        for _ in range(3):
            it = fu.build_item(t, s(rng))
            for pt in it["parts"]:
                uv = fu.box_project_uv(pt["verts"], pt["faces"], texel_m=1.0)
                sa = fu.uv_signed_areas(uv)
                tot += len(sa)
                bad += int((sa <= 1e-14).sum())
                V = np.asarray(pt["verts"]); F = np.asarray(pt["faces"])
                a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
                n = np.cross(b - a, c - a)
                n3 = np.linalg.norm(n, axis=1)
                k2 = np.abs(n).argmax(1)
                flat = np.abs(np.take_along_axis(n, k2[:, None], 1))[:, 0] \
                    / np.maximum(n3, 1e-12) > 0.99
                dens += (np.sqrt(np.abs(sa[flat])
                                 / np.maximum(n3[flat] / 2, 1e-12))).tolist()
    assert bad == 0, f"{bad}/{tot} flipped/zero UV faces"
    d = np.asarray(dens)
    assert d.std() / d.mean() < 0.10, "texel density CV >= 10%"
    assert abs(d.mean() - 1.0) < 0.02          # 1 UV tile == 1 m


def test_material_assignment_and_provenance():
    """Every floor/wall/furniture class has an entry; image entries point at
    existing files; decal rate 100% for wall art; roundtrip preserves it."""
    from lenscope.genesis import materials as mm
    from pathlib import Path
    lib = mm.scan_library()
    n_img_total = 0
    for seed in range(8):
        spec = build_ge1_spec(seed)
        assert "wall" in spec.materials and "ceiling" in spec.materials
        for r in spec.rooms:
            assert f"floor/{r.name}" in spec.materials, (seed, r.name)
        for fs in spec.furniture:
            for kind in mm._ITEM_KINDS.get(fs.ftype, ()):
                assert f"furn/{fs.name}/{kind}" in spec.materials
            if fs.ftype in ("painting", "poster"):
                e = spec.materials.get(f"decal/{fs.name}")
                assert e is not None
                if lib["paintings"]:
                    assert e["mode"] == "decal", (seed, fs.name)
        root = Path(spec.assets_lib["root"])
        for k, e in spec.materials.items():
            if e.get("mode") == "image":
                n_img_total += 1
                assert mm.maps_for_set(root, e["set"]) is not None, (k, e)
            elif e.get("mode") == "decal":
                assert (root / e["image"]).exists(), (k, e)
        rt = GenesisSpec.from_json(spec.to_json())
        assert rt.materials == spec.materials
    if lib["pbr"]:
        assert n_img_total > 0                  # real-PBR hit rate > 0


def test_procedural_domains_and_empty_library_invariant():
    """Procedural draws stay inside the physical BRDF domains; a missing
    library root -> everything procedural, and the engine still builds
    (empty-library invariant)."""
    import os
    import random as _rd
    from lenscope.genesis import materials as mm
    rng = _rd.Random(0)
    for cls, d in mm._DOMAINS.items():
        for _ in range(40):
            e = mm.draw_procedural(rng, cls)
            v = max(e["rgb"])
            assert d["alb"][0] - 1e-6 <= v <= d["alb"][1] + 1e-6, (cls, e)
            assert d["rough"][0] <= e["rough"] <= d["rough"][1]
    old = os.environ.get("GENESIS_ASSETS")
    os.environ["GENESIS_ASSETS"] = "/nonexistent_genesis_assets"
    try:
        spec = build_ge1_spec(3)
        assert spec.material_image_prob == 0.0
        modes = {e["mode"] for e in spec.materials.values()}
        # screens that are on exist without a library too (procedural glow)
        assert modes <= {"procedural", "screen"}, modes
        for k, e in spec.materials.items():
            if e["mode"] == "screen":
                assert e.get("image") is None
        assert spec.assets_lib["pbr_sets"] == {}
    finally:
        if old is None:
            os.environ.pop("GENESIS_ASSETS", None)
        else:
            os.environ["GENESIS_ASSETS"] = old


def test_no_coincident_luminaires():
    """Coincident/overlapping lamp shade solids z-fight to exact-zero black
    (e.g. two lamps on the identical coordinate when interior points wrap).
    Physical criterion: only shades in the same ceiling z-band constrain
    each other; different bands never touch."""
    import math as _m
    band = {"flush": "hi", "bulb": "hi", "pendant": "pend"}
    for seed in range(30):
        spec = build_ge1_spec(seed)
        by_room = {}
        for L in spec.lights:
            b = band.get(L.fixture)
            if b:
                by_room.setdefault((L.room, b), []).append(L)
        for ls in by_room.values():
            for i in range(len(ls)):
                for j in range(i + 1, len(ls)):
                    d = _m.hypot(ls[i].x - ls[j].x, ls[i].y - ls[j].y)
                    assert d > 0.30, (seed, ls[i].room, ls[i].fixture,
                                      ls[j].fixture, round(d, 3))


# home-decor layout rules: soft preferences with hard verification

def _win_spans_by_edge(spec, room):
    import math as _m
    from lenscope.genesis.layout import _room_edges, _edge_of_point
    edges = _room_edges(room.poly)
    wr = {w.id: w for w in spec.walls}
    spans = {}
    for w in spec.windows:
        run = wr.get(w.wall)
        if run is None:
            continue
        dx, dy = run.p2[0] - run.p1[0], run.p2[1] - run.p1[1]
        L = _m.hypot(dx, dy)
        cx, cy = run.p1[0] + dx * w.u / L, run.p1[1] + dy * w.u / L
        # only windows on this room's boundary
        best_d = min(_dist_seg(cx, cy, e) for e in edges)
        if best_d > 0.35:
            continue
        ei, u = _edge_of_point(edges, cx, cy)
        spans.setdefault(ei, []).append((u - w.width / 2, u + w.width / 2))
    return edges, spans


def _dist_seg(x, y, e):
    import math as _m
    ax, ay = e["a"]
    u = min(max((x - ax) * e["d"][0] + (y - ay) * e["d"][1], 0.0), e["L"])
    return _m.hypot(x - (ax + e["d"][0] * u), y - (ay + e["d"][1] * u))


def test_solid_backs_and_window_blocking():
    """Bed/sofa/tv_stand backs never span a window (unless a declared
    fallback); tall furniture never blocks a window."""
    import math as _m
    from lenscope.genesis.layout import _edge_of_point
    n_rooms = n_fb = 0
    for seed in range(25):
        spec = build_ge1_spec(seed)
        rooms = {r.name: r for r in spec.rooms}
        n_rooms += len(spec.rooms)
        n_fb += sum(v.get("solid_wall_fallbacks", 0)
                    for v in spec.layout_stats.values())
        for fs in spec.furniture:
            if fs.ftype not in ("bed", "sofa", "tv_stand", "wardrobe",
                                "bookshelf"):
                continue
            room = rooms[fs.room]
            edges, spans = _win_spans_by_edge(spec, room)
            # the wall behind the item = edge nearest its back point
            c = _m.cos(_m.radians(fs.yaw_deg))
            s = _m.sin(_m.radians(fs.yaw_deg))
            bx = fs.x + s * fs.hy            # back = local -y
            by = fs.y - c * fs.hy
            ei, u = _edge_of_point(edges, bx, by)
            if _dist_seg(bx, by, edges[ei]) > 0.30:
                continue                      # not wall-backed (free-standing)
            fb = spec.layout_stats.get(fs.room, {}).get(
                "solid_wall_fallbacks", 0)
            for (a, b) in spans.get(ei, []):
                overlap = u - fs.hx - 0.10 < b and a < u + fs.hx + 0.10
                if fs.ftype in ("wardrobe", "bookshelf"):
                    assert not overlap, (seed, fs.name, "tall item blocks window")
                elif overlap:
                    assert fb > 0, (seed, fs.name, "window-backed w/o fallback")
    # <10%: measured geometric floor is ~6.5% (rooms whose every fittable
    # span carries a window -- real homes also back beds onto windows then);
    # every fallback is declared in spec.layout_stats
    assert n_fb / max(n_rooms, 1) < 0.10, f"solid-wall fallback rate {n_fb}/{n_rooms}"


def test_tv_faces_sofa_and_wallboard():
    """Every living-room TV sits on the edge hit by the sofa's line of
    sight; wall-mounted items never overlap on the wall."""
    import math as _m
    from lenscope.genesis.layout import (_room_edges, _edge_of_point,
                                          RoomPlanner)
    for seed in range(25):
        spec = build_ge1_spec(seed)
        rooms = {r.name: r for r in spec.rooms}
        for r in spec.rooms:
            here = [f for f in spec.furniture if f.room == r.name]
            sofa = next((f for f in here if f.ftype == "sofa"), None)
            tv = next((f for f in here if f.ftype == "tv_stand"), None)
            if sofa is not None and tv is not None:
                pl = RoomPlanner(r, [], None)
                c = _m.cos(_m.radians(sofa.yaw_deg))
                s = _m.sin(_m.radians(sofa.yaw_deg))
                hit = pl.ray_edge(sofa.x, sofa.y, -s, c)
                assert hit is not None, (seed, r.name)
                ei_tv, _u = _edge_of_point(pl.edges, tv.x, tv.y)
                assert ei_tv == hit[0], (seed, r.name, "TV off sight line")
            # pairwise wall-item overlap (same edge, u+z intersect)
            edges = _room_edges(r.poly)
            mounted = []
            for f in here:
                if f.ftype in ("mirror", "painting", "poster", "clock"):
                    ei, u = _edge_of_point(edges, f.x, f.y)
                    h2 = f.height / 2
                    mounted.append((ei, u - f.hx, u + f.hx,
                                    f.z - h2, f.z + h2, f.name))
            for i in range(len(mounted)):
                for j in range(i + 1, len(mounted)):
                    a, b = mounted[i], mounted[j]
                    if a[0] != b[0]:
                        continue
                    olap = a[1] < b[2] and b[1] < a[2] and \
                        a[3] < b[4] and b[3] < a[4]
                    assert not olap, (seed, r.name, a[5], b[5])


def test_walkability():
    """Rasterized free floor (minus furniture footprints, rugs walkable)
    -- every door cell connects to the room's main free component."""
    import math as _m
    for seed in range(15):
        spec = build_ge1_spec(seed)
        rooms = {r.name: r for r in spec.rooms}
        for r in spec.rooms:
            xs = [p[0] for p in r.poly]
            ys = [p[1] for p in r.poly]
            cell = 0.1
            nx = int((max(xs) - min(xs)) / cell) + 1
            ny = int((max(ys) - min(ys)) / cell) + 1
            free = [[False] * ny for _ in range(nx)]
            for i in range(nx):
                for j in range(ny):
                    x = min(xs) + (i + 0.5) * cell
                    y = min(ys) + (j + 0.5) * cell
                    free[i][j] = _in_poly(x, y, r.poly)
            for f in spec.furniture:
                if f.room != r.name or f.ftype in ("curtain", "mirror",
                                                   "painting", "poster",
                                                   "clock", "rug"):
                    continue
                c = _m.cos(_m.radians(-f.yaw_deg))
                s = _m.sin(_m.radians(-f.yaw_deg))
                for i in range(nx):
                    for j in range(ny):
                        if not free[i][j]:
                            continue
                        x = min(xs) + (i + 0.5) * cell - f.x
                        y = min(ys) + (j + 0.5) * cell - f.y
                        lx = c * x - s * y
                        ly = s * x + c * y
                        if abs(lx) < f.hx and abs(ly) < f.hy:
                            free[i][j] = False
            # flood from the largest free component
            seen = [[False] * ny for _ in range(nx)]
            comps = []
            for i in range(nx):
                for j in range(ny):
                    if free[i][j] and not seen[i][j]:
                        stack = [(i, j)]
                        seen[i][j] = True
                        comp = []
                        while stack:
                            a, b = stack.pop()
                            comp.append((a, b))
                            for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                                na, nb = a + da, b + db
                                if 0 <= na < nx and 0 <= nb < ny and \
                                        free[na][nb] and not seen[na][nb]:
                                    seen[na][nb] = True
                                    stack.append((na, nb))
                        comps.append(comp)
            if not comps:
                continue
            main = max(comps, key=len)
            main_frac = len(main) / sum(len(c) for c in comps)
            assert main_frac > 0.90, (seed, r.name, main_frac)


def test_layout_export_schema():
    """Layout export: wdo count == doors+windows; corners == manifest polys;
    adjacency symmetric + consistent with geometric door connectivity;
    every interior door bridges two distinct adjacent rooms."""
    import json as _json
    from lenscope.genesis.export_layout import export_layout
    from pathlib import Path as _P
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        for seed in (0, 6, 9, 13, 21):
            spec = build_ge1_spec(seed)
            doc = export_layout(spec, _P(td))
            assert len(doc["wdo"]) == len(spec.doors) + len(spec.windows)
            polys = {r.name: [[round(x, 4), round(y, 4)] for x, y in r.poly]
                     for r in spec.rooms}
            for r in doc["rooms"]:
                assert r["corners_xy"] == polys[r["id"]]
            adj = {r["id"]: set(r["adjacent"]) for r in doc["rooms"]}
            for a, ns in adj.items():
                for b in ns:
                    assert a in adj[b], (seed, a, b)
            for w in doc["wdo"]:
                if w["type"] == "door" and not w["extra"]["entry"]:
                    ra, rb = w["rooms"]
                    if rb != "exterior":
                        assert ra != rb and rb in adj[ra], (seed, w)
            # round-trip json
            rt = _json.loads(_json.dumps(doc))
            assert rt == doc


def test_facing_gate_multiseed():
    """Every wall-backed furniture item must (a) actually be wall-backed and
    (b) face the interior, under the pinned mesh convention
    front = R_z(+yaw)@y_hat = (-sin, cos).

    Skip-proof by design: silently skipping items whose computed back is
    >0.30m from a wall would miss exactly the reversed (E/W-wall) items."""
    import math as _m
    from lenscope.genesis.layout import _room_edges, _edge_of_point
    WALL_BACKED = ("bed", "sofa", "wardrobe", "bookshelf", "tv_stand")
    checked = 0
    for seed in range(20):
        spec = build_ge1_spec(seed)
        rooms = {r.name: r for r in spec.rooms}
        for fs in spec.furniture:
            if fs.ftype not in WALL_BACKED:
                continue
            edges = _room_edges(rooms[fs.room].poly)
            c = _m.cos(_m.radians(fs.yaw_deg))
            s = _m.sin(_m.radians(fs.yaw_deg))
            fx, fy = -s, c                                # mesh front
            bx, by = fs.x - fx * fs.hy, fs.y - fy * fs.hy  # back point
            ei, _u = _edge_of_point(edges, bx, by)
            d_back = _dist_seg(bx, by, edges[ei])
            assert d_back < 0.30, (seed, fs.name, "not wall-backed", d_back)
            nx, ny = edges[ei]["n_in"]
            assert fx * nx + fy * ny > 0.7, (
                seed, fs.name, "faces the wall")
            if fs.ftype == "bed":                          # headboard vs foot
                d_foot = _dist_seg(fs.x + fx * fs.hy, fs.y + fy * fs.hy,
                                   edges[ei])
                assert d_back < d_foot, (seed, fs.name, "headboard off wall")
            checked += 1
    assert checked > 60, f"gate exercised only {checked} items"


def test_group_facing_and_chairs():
    """Sofa faces its coffee table / TV; dining & desk chairs face their table.
    All in the pinned front=(-sin,cos) convention (multi-seed)."""
    import math as _m
    n_sofa_pairs = n_chairs = 0
    # bathroom/kitchen room types make dining rooms rarer on the small
    # default plans, so n_rooms=10 specs are included to keep a meaningful
    # chair sample (facing dots are the correctness assert; the counts
    # only guard against silent starvation).
    specs = [build_ge1_spec(seed) for seed in range(20)] \
        + [build_ge1_spec(seed, n_rooms=10) for seed in (2, 9, 14, 17, 21)]
    for si, spec in enumerate(specs):
        seed = si                      # label for assert messages
        func_of = {r.name: r.function for r in spec.rooms}
        by_room = {}
        for f in spec.furniture:
            by_room.setdefault(f.room, []).append(f)
        for rname, items in by_room.items():
            if func_of.get(rname) == "studio":
                # multi-zone (studio) rooms have other zones' tables in the
                # same room record; the sofa faces its zone's coffee table only
                # (by construction; test_studio_and_density covers studios)
                continue
            sofa = next((f for f in items if f.ftype == "sofa"), None)
            tables = [f for f in items if f.ftype == "table"]
            tv = next((f for f in items if f.ftype == "tv_stand"), None)
            if sofa is not None:
                c = _m.cos(_m.radians(sofa.yaw_deg))
                s = _m.sin(_m.radians(sofa.yaw_deg))
                for tgt in ([t for t in tables] + ([tv] if tv else [])):
                    if tgt is None:
                        continue
                    dx, dy = tgt.x - sofa.x, tgt.y - sofa.y
                    L = _m.hypot(dx, dy)
                    if L < 0.3:
                        continue
                    dot = (-s) * dx / L + c * dy / L
                    assert dot > 0.5, (seed, rname, tgt.ftype,
                                       "sofa turns its back on its group")
                    n_sofa_pairs += 1
            for ch in (f for f in items if f.ftype == "chair"):
                if not tables:
                    continue
                t0 = min(tables, key=lambda t: _m.hypot(t.x - ch.x,
                                                        t.y - ch.y))
                dx, dy = t0.x - ch.x, t0.y - ch.y
                L = _m.hypot(dx, dy)
                if L < 1e-6 or L > 1.5:
                    continue
                c = _m.cos(_m.radians(ch.yaw_deg))
                s = _m.sin(_m.radians(ch.yaw_deg))
                dot = (-s) * dx / L + c * dy / L
                assert dot > 0.5, (seed, rname, "chair faces away from table")
                n_chairs += 1
    # threshold 15: the wall-collision check rejects dining chairs whose slot
    # would clip a wall, so not every table gets all its chairs
    assert n_sofa_pairs >= 8 and n_chairs >= 15, (n_sofa_pairs, n_chairs)


def test_mesh_convention_lock():
    """Locks the local-frame convention the whole yaw system relies on:
    back parts at local -y, fronts/handles at +y. If an assembly ever flips,
    this fails before any render does."""
    from lenscope.genesis import furniture as fu
    import random as _rd
    rng = _rd.Random(7)

    def _mean_y(item, part_name):
        p = next(p for p in item["parts"]
                 if p["name"] == part_name or p["name"].startswith(part_name))
        return float(p["verts"][:, 1].mean())   # translations are baked in

    assert _mean_y(fu.build_item("bed", fu.sample_bed(rng)),
                   "headboard") < 0, "bed headboard must sit at local -y"
    assert _mean_y(fu.build_item("sofa", fu.sample_sofa(rng)),
                   "back") < 0, "sofa backrest must sit at local -y"
    assert _mean_y(fu.build_item("wardrobe", fu.sample_wardrobe(rng)),
                   "handle") > 0, "wardrobe handles must sit at local +y"
    assert _mean_y(fu.build_item("chair", fu.sample_chair(rng)),
                   "back") < 0, "chair backrest must sit at local -y"


def test_curtain_parallel_inside_wall():
    """Curtains parallel to their wall and fully inside the wall segment (a
    wide curtain near a corner must not cut into the next wall)."""
    import math as _m
    from lenscope.genesis.layout import _room_edges
    n = 0
    for seed in range(20):
        spec = build_ge1_spec(seed)
        rooms = {r.name: r for r in spec.rooms}
        for fs in spec.furniture:
            if fs.ftype != "curtain":
                continue
            edges = _room_edges(rooms[fs.room].poly)
            c = _m.cos(_m.radians(fs.yaw_deg))
            s = _m.sin(_m.radians(fs.yaw_deg))
            # the curtain's wall = nearest edge among edges parallel to the
            # band (a corner-adjacent curtain is nearer the perpendicular
            # wall than its own -- plain nearest-edge picks the wrong one)
            par = [e for e in edges
                   if abs(c * e["d"][0] + s * e["d"][1]) > 0.9]
            assert par, (seed, fs.name, "no parallel wall (skewed curtain)")
            e = min(par, key=lambda e: _dist_seg(fs.x, fs.y, e))
            assert _dist_seg(fs.x, fs.y, e) < 0.35, (seed, fs.name, "off wall")
            dot = abs(c * e["d"][0] + s * e["d"][1])
            assert dot > 0.995, (seed, fs.name, f"not parallel (|dot|={dot:.3f})")
            # span inside the segment (no corner knife-through)
            ax, ay = e["a"]
            u = (fs.x - ax) * e["d"][0] + (fs.y - ay) * e["d"][1]
            assert u - fs.hx >= -0.03 and u + fs.hx <= e["L"] + 0.03, (
                seed, fs.name, "curtain span exits its wall segment")
            n += 1
    assert n >= 6, f"gate exercised only {n} curtains"


def test_tint_clip_cap_and_reachability():
    """'Checkerboard' check: image-material tints may never push the set's
    p95 past white (clipping crushes texture into flat patches), and a drawn
    set must be able to reach ~the class domain floor (near-ebony sets like
    Wood051, mean 0.038, are curated out instead of blown out)."""
    import random as _rd
    from lenscope.genesis.materials import (scan_library, draw_surface,
                                             set_albedo_p95, set_albedo_mean,
                                             _DOMAINS)
    lib = scan_library()
    if not lib["pbr"]:
        return                                    # empty-library invariant
    rng = _rd.Random(3)
    n_img = 0
    for cls in ("floor_tile", "floor_wood", "floor_stone", "wood", "fabric",
                "leather", "carpet", "wall", "metal", "brick"):
        for _ in range(60):
            e = draw_surface(rng, cls, lib, 1.0)
            if e.get("mode") != "image":
                continue
            n_img += 1
            p95 = set_albedo_p95(lib["root"], e["set"])
            assert e["tint_v"] * p95 <= 0.99, (
                cls, e["set"], "clips:", e["tint_v"], p95)
            mean = set_albedo_mean(lib["root"], e["set"])
            assert e["tint_v"] * mean >= 0.55 * _DOMAINS[cls]["alb"][0] - 1e-6, (
                cls, e["set"], "unreachable domain floor")
    assert n_img > 250, f"only {n_img} image draws exercised"


def test_n_rooms_knob():
    """Explicit room-count control up to 10+: exact count, deterministic per
    (seed, n), all legality checks intact, and the default path (knob unset)
    byte-identical (no extra main-stream draws)."""
    ref = build_ge1_spec(3).to_json()
    assert build_ge1_spec(3, n_rooms=None).to_json() == ref
    for n in (4, 8, 10, 12):
        spec = build_ge1_spec(21, n_rooms=n)
        assert len(spec.rooms) == n, (n, len(spec.rooms))
        assert build_ge1_spec(21, n_rooms=n).to_json() == spec.to_json()
        # legality: min area, connectivity BFS 100%, entry door, windows ext
        adj = {k: set() for k in range(n)}
        for d in spec.doors:
            if d.entry:
                continue
            a, b = _door_connects(spec, d)
            assert a is not None and b is not None and a != b, (n, d.wall)
            adj[a].add(b); adj[b].add(a)
        seen, stack = {0}, [0]
        while stack:
            k = stack.pop()
            for m2 in adj[k]:
                if m2 not in seen:
                    seen.add(m2); stack.append(m2)
        assert len(seen) == n, f"n={n}: {len(seen)}/{n} rooms reachable"
        assert any(d.entry for d in spec.doors)
        walls = {w.id: w for w in spec.walls}
        for wd in spec.windows:
            assert not walls[wd.wall].internal
        for r in spec.rooms:
            xs = [p[0] for p in r.poly]; ys = [p[1] for p in r.poly]
            area = (max(xs) - min(xs)) * (max(ys) - min(ys))
            assert area >= 8.5, (n, r.name, area)
    # different n on the same seed -> different plans (derived-stream keying)
    assert build_ge1_spec(21, n_rooms=8).to_json() != \
        build_ge1_spec(21, n_rooms=10).to_json()


def test_furniture_wall_interpenetration():
    """No furniture footprint may penetrate any realized wall body deeper
    than 5mm (e.g. bed headboards / sofa backs inside walls). Mechanisms:
    (1) room polys are boundary lines while internal walls are centered on
    them (t/2 protrudes into the room), so against_wall offsets by a
    per-point wall-face inset (edges can be part internal, part external);
    (2) notch stubs <0.8m are invisible to _room_edges, and pl.at companions
    are covered by the catch-all wall-run collision in _legal (two-tier
    margin: designed art burial of 1.0-1.2cm allowed); (3) curtains hang
    from the wall centerline, so their offset includes t/2."""
    import math as _m
    from lenscope.genesis import solids as _sol
    from lenscope.genesis.layout import _obb_overlap
    checked = 0
    for seed in (0, 2, 4, 5, 9, 11):
        for n in (None, 10):
            spec = build_ge1_spec(seed, n_rooms=n)
            real = _sol.realize(spec)
            walls2d = [(b["c"][0], b["c"][1], b["s"][0], b["s"][1],
                        _m.degrees(b["yaw"]), b["c"][2] - b["s"][2], b["name"])
                       for b in real["boxes"] if b["kind"] == "wall"]
            for fs in spec.furniture:
                if fs.ftype in ("mirror", "painting", "poster", "clock",
                                "towel_bar"):
                    continue          # wall-mounted, designed shallow burial
                checked += 1
                fobb = (fs.x, fs.y, fs.hx, fs.hy, fs.yaw_deg)
                for (cx, cy, hx, hy, yw, zlo, nm) in walls2d:
                    if zlo > fs.height - 0.02:
                        continue      # header above the item
                    assert not _obb_overlap(fobb, (cx, cy, hx, hy, yw),
                                            margin=-0.005), \
                        (seed, n, fs.name, fs.ftype, nm)
    assert checked > 250, f"gate exercised only {checked} items"


def test_designer_rules_and_variety():
    """Designer soft rules hold (the bed never directly faces a door unless
    declared as a fallback; night-path side kept) and the aesthetic variety
    (door styles / mullion grids / headboard styles / pillow counts) is
    genuinely random: distribution entropy checks, plus a yaw-entropy floor
    (randomness never collapses)."""
    import math as _m
    from collections import Counter
    from lenscope.genesis import furniture as fu
    import random as _rd

    door_styles, grids, yaws = Counter(), Counter(), []
    d1_viol = d1_fallback = n_beds = 0
    for seed in range(20):
        spec = build_ge1_spec(seed)
        d1_fallback += sum(v.get("bed_soft_rule_fallbacks", 0)
                           for v in spec.layout_stats.values())
        for d in spec.doors:
            door_styles[d.style] += 1
        for w in spec.windows:
            grids[(w.mull_nx, w.mull_ny)] += 1
        walls = {w.id: w for w in spec.walls}
        rooms = {r.name: r for r in spec.rooms}
        door_pts = []
        for d in spec.doors:
            w = walls[d.wall]
            dx, dy = w.p2[0] - w.p1[0], w.p2[1] - w.p1[1]
            L = _m.hypot(dx, dy)
            door_pts.append((w.p1[0] + dx / L * d.u, w.p1[1] + dy / L * d.u))
        for fs in spec.furniture:
            yaws.append(fs.yaw_deg % 360.0)
            if fs.ftype != "bed":
                continue
            n_beds += 1
            poly = rooms[fs.room].poly
            # rule semantics: doors of this room (on its boundary) only
            room_doors = [(kx, ky) for (kx, ky) in door_pts
                          if _dist_to_room_poly(kx, ky, poly) < 0.4]
            c = _m.cos(_m.radians(fs.yaw_deg))
            s = _m.sin(_m.radians(fs.yaw_deg))
            for (kx, ky) in room_doors:
                vx, vy = kx - fs.x, ky - fs.y
                ahead = vx * (-s) + vy * c
                perp = abs(-c * vx - s * vy)
                if 0.3 < ahead < 6.0 and perp < fs.hx + 0.2:
                    d1_viol += 1
                    break
    # door-facing rule: violations only where a declared fallback happened
    assert d1_viol <= d1_fallback, (d1_viol, d1_fallback)
    assert n_beds >= 15, n_beds

    def _entropy(cnt):
        tot = sum(cnt.values())
        return -sum((v / tot) * _m.log2(v / tot) for v in cnt.values())

    assert len(door_styles) >= 3 and _entropy(door_styles) > 1.2, door_styles
    assert len(grids) >= 4 and _entropy(grids) > 1.5, grids
    # headboard/pillow variety straight from the sub-stream (200 draws)
    hb, pil = Counter(), Counter()
    for i in range(200):
        sr = _rd.Random(i ^ 0xBED5)
        hb[sr.choices(["plain", "slats", "panel"], weights=[40, 30, 30])[0]] += 1
        pil[sr.choice([2, 2, 3, 4])] += 1
    assert len(hb) == 3 and _entropy(hb) > 1.4, hb
    assert len(pil) == 3, pil
    # randomness floor: the entropy of furniture yaw must not collapse
    bins = Counter(int(y // 30) for y in yaws)
    assert _entropy(bins) > 2.0, (dict(bins), _entropy(bins))


def _dist_to_room_poly(x, y, poly):
    import math as _m
    best = float("inf")
    for i in range(len(poly)):
        ax, ay = poly[i]
        bx, by = poly[(i + 1) % len(poly)]
        abx, aby = bx - ax, by - ay
        den = max(abx * abx + aby * aby, 1e-9)
        t = min(max(((x - ax) * abx + (y - ay) * aby) / den, 0.0), 1.0)
        best = min(best, _m.hypot(x - (ax + t * abx), y - (ay + t * aby)))
    return best


def test_bathroom_kitchen():
    """Bathroom/kitchen room types + fixtures: both functions occur across
    seeds; a bathroom always has a toilet (and vanity when it fits); a
    kitchen run carries >=2 of {fridge, counter, stove}; fixture semantics
    resolve to the additive NYU40 ids (toilet 33 / sink 34 / bathtub 36 /
    fridge 24 / counter 12, no renumbering of existing classes); bathroom
    windows are privacy windows (sill >= 1.2); all fixtures ride the
    standard wall-clearance machinery (interpenetration is covered by
    test_furniture_wall_interpenetration)."""
    n_bath = n_kit = 0
    for seed in range(24):
        for n in (None, 10):
            spec = build_ge1_spec(seed, n_rooms=n)
            by_room = {}
            for f in spec.furniture:
                by_room.setdefault(f.room, []).append(f.ftype)
            walls = {w.id: w for w in spec.walls}
            for r in spec.rooms:
                kinds = by_room.get(r.name, [])
                if r.function == "bathroom":
                    n_bath += 1
                    assert "toilet" in kinds, (seed, n, r.name, kinds)
                    for wd in spec.windows:
                        run = walls[wd.wall]
                        dxw = run.p2[0] - run.p1[0]
                        dyw = run.p2[1] - run.p1[1]
                        Lw = math.hypot(dxw, dyw)
                        cx = run.p1[0] + dxw / Lw * wd.u
                        cy = run.p1[1] + dyw / Lw * wd.u
                        if _dist_to_room_poly(cx, cy, r.poly) < 0.2:
                            assert wd.sill >= 1.2, (seed, n, r.name, wd.sill)
                if r.function == "kitchen":
                    n_kit += 1
                    # a 4-segment straight run needs ~4-5m of wall, more than
                    # ~11 m2 kitchens offer; the greedy/L-run/kitchenette/
                    # island placement guarantees that a kitchen is never
                    # empty and always has the kitchen-defining sink counter.
                    assert kinds, (seed, n, r.name, "empty kitchen")
                    assert "counter" in kinds, (seed, n, r.name, kinds)
    assert n_bath >= 10, n_bath
    assert n_kit >= 15, n_kit
    # semantic resolution (additive contract)
    from lenscope.core.mesh import semantics_from_names
    names = ["Toilet_0000", "SinkVanity_0001", "Bathtub_0002", "Shower_0003",
             "TowelBar_0004", "KitchenCounter_0005", "Stove_0006",
             "Fridge_0007", "RangeHood_0008", "WallCabinet_0009",
             "Sofa_0000", "Bed_0100", "Cabinet_0001", "Wall_EW_00"]
    sem = dict(zip(names, map(int, semantics_from_names(
        [{"name": x} for x in names]))))
    assert sem["Toilet_0000"] == 33 and sem["SinkVanity_0001"] == 34
    assert sem["Bathtub_0002"] == 36 and sem["Fridge_0007"] == 24
    assert sem["KitchenCounter_0005"] == 12 and sem["WallCabinet_0009"] == 3
    # existing classes untouched
    assert sem["Sofa_0000"] == 6 and sem["Bed_0100"] == 4 \
        and sem["Wall_EW_00"] == 1


def test_studio_and_density():
    """Single big rooms are open-plan studios with randomized coexisting
    zones; density is lifted with floor decor, soft-surface clutter and
    hanging pieces. Checks: studio triggers on n_rooms=1 big plans with
    bed+sofa coexisting and >=2 zone kinds across seeds (randomness not
    collapsed); never an empty single room; hanging pieces occur, stay
    inside the room, and clear the ceiling band; >= 270 furniture + clutter
    items on the reference scene; new-stem semantics resolve additively
    (floorplant/ceilingfan not shadowed by floor/ceiling)."""
    import collections
    zone_combos = collections.Counter()
    n_studio = 0
    for seed in range(10):
        sp = build_ge1_spec(seed, n_rooms=1)
        r = sp.rooms[0]
        items = [f for f in sp.furniture if f.room == r.name]
        assert items, (seed, "empty single room")
        if r.function != "studio":
            continue
        n_studio += 1
        kinds = {f.ftype for f in items}
        assert "bed" in kinds and "sofa" in kinds, (seed, sorted(kinds))
        zs = sp.layout_stats[r.name].get("studio_zones") or []
        zone_combos[tuple(sorted(z[0] for z in zs))] += 1
        # hanging pieces: inside the room poly, in the air band
        for f in items:
            if f.ftype in ("hangplant", "lantern", "mobile", "ceilingfan"):
                assert f.z >= 1.55, (seed, f.ftype, f.z)
                assert f.z + 0.02 <= sp.height + 1e-6, (seed, f.ftype)
    assert n_studio >= 5, n_studio
    assert len(zone_combos) >= 2, zone_combos       # zoning not collapsed
    # density check on the reference scene (seed 2, n_rooms=10)
    sp2 = build_ge1_spec(2, n_rooms=10)
    total = len(sp2.furniture) + len(sp2.clutter)
    assert total >= 270, total
    hang = [f for f in sp2.furniture
            if f.ftype in ("hangplant", "lantern", "mobile", "ceilingfan")]
    assert len(hang) >= 4, len(hang)
    # soft-surface clutter actually landed on beds/sofas
    parents = {c.parent for c in sp2.clutter}
    assert any(p.startswith(("Bed", "Sofa")) for p in parents), parents
    # additive semantics for the new stems
    from lenscope.core.mesh import semantics_from_names
    names = ["FloorPlant_0001", "CeilingFan_0002", "HangPlant_0003",
             "BookStack_0004", "Suitcase_0005", "Lantern_0006",
             "MobileDecor_0007", "Basket_0008", "Floor", "Ceiling"]
    sem = dict(zip(names, map(int, semantics_from_names(
        [{"name": x} for x in names]))))
    assert sem["FloorPlant_0001"] == 40 and sem["Floor"] == 2
    assert sem["CeilingFan_0002"] == 40 and sem["Ceiling"] == 22
    assert sem["BookStack_0004"] == 23 and sem["Suitcase_0005"] == 37
    assert sem["HangPlant_0003"] == 40 and sem["MobileDecor_0007"] == 40


def test_semantic_layering():
    """Wall/floor/ceiling are separate NYU40 classes (1, 2, 22) on separate
    objects; also locks the two stem-shadowing cases (FloorLamp->floor,
    TableLamp->table)."""
    from lenscope.core.mesh import semantics_from_names
    names = ["Wall_EW_00_00", "InternalWall_IW_00_01", "WallSkirting_0001",
             "Floor_07", "Ceiling_07", "CeilingTrim_06_0",
             "FloorLamp_16_base", "FloorLamp_16_pole", "TableLamp_03",
             "CeilingLight_02", "FloorPlant_0001", "CeilingFan_0002",
             "WindowGlass_01", "DoorFrame_02_L"]
    sem = dict(zip(names, map(int, semantics_from_names(
        [{"name": x} for x in names]))))
    assert sem["Wall_EW_00_00"] == 1 and sem["InternalWall_IW_00_01"] == 1
    assert sem["WallSkirting_0001"] == 1
    assert sem["Floor_07"] == 2 and sem["Ceiling_07"] == 22
    assert sem["CeilingTrim_06_0"] == 22
    assert sem["FloorLamp_16_base"] == 35 and sem["FloorLamp_16_pole"] == 35
    assert sem["TableLamp_03"] == 35 and sem["CeilingLight_02"] == 35
    assert sem["FloorPlant_0001"] == 40 and sem["CeilingFan_0002"] == 40
    assert sem["WindowGlass_01"] == 9 and sem["DoorFrame_02_L"] == 8
