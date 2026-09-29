"""Thin bpy shell for second-generation scenes: GenesisSpec -> .blend.

No decisions here (spec/solids decide everything). Translates: oriented
boxes -> meshes; polygon slabs -> prisms; hinged leaves; luminaires (lumens ->
watts via the spec's luminous efficacy efficacy_lm_w; CCT via blackbody);
Nishita world with sun/sky randomness from the spec. Only sockets that are
stable across Blender 4.1 and 4.5 are used.

Usage:
  blender -b --python lenscope/genesis/build_scene.py -- \
      --seed 0 --out /path/scenes [--ge0] [--spec spec.json]
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import bpy

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[2]))

from lenscope.genesis.spec import (GenesisSpec, build_ge0_spec,   # noqa: E402
                                    build_ge1_spec)
from lenscope.genesis import solids                                # noqa: E402

_BOX_FACES = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4),
              (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]


def _mesh_obox(b):
    cx, cy, cz = b["c"]; hx, hy, hz = b["s"]; yaw = b.get("yaw", 0.0)
    ca, sa = math.cos(yaw), math.sin(yaw)
    verts = []
    for dz in (-hz, hz):
        for dx, dy in [(-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy)]:
            verts.append((cx + ca * dx - sa * dy, cy + sa * dx + ca * dy,
                          cz + dz))
    me = bpy.data.meshes.new(b["name"])
    me.from_pydata(verts, [], _BOX_FACES)
    me.update()
    return me


def _smooth_mesh(me, angle_deg=30.0):
    """Interpolated shading normals on curved surfaces.
    No geometry change: verts/faces are untouched, so the core.cast depth/
    normal ground truth (which ray-casts the same triangles) stays bitwise
    identical; this is shading-only, like a real photo of the same object.
    30 deg threshold: lathe segments (14-24 sides => 15-26 deg) smooth,
    beveled-box hard edges (90 deg) stay crisp. Never use a subdivision
    modifier here: modifiers change the render mesh only and would desync
    Cycles pixels from the ray-cast ground truth."""
    for poly in me.polygons:
        poly.use_smooth = True
    try:
        me.set_sharp_from_angle(angle=math.radians(angle_deg))  # bpy >= 4.1
    except AttributeError:
        try:
            me.use_auto_smooth = True                           # bpy <= 4.0
            me.auto_smooth_angle = math.radians(angle_deg)
        except AttributeError:
            pass                       # very old bpy: full smooth is still
                                       # correct for lathe-only curved parts


def _mesh_polyslab(ps):
    poly, z0, z1 = ps["poly"], ps["z0"], ps["z1"]
    n = len(poly)
    verts = [(x, y, z0) for x, y in poly] + [(x, y, z1) for x, y in poly]
    faces = [tuple(range(n - 1, -1, -1)), tuple(range(n, 2 * n))]
    # side quads wound outward for a CCW ring
    faces += [((i + 1) % n, i, n + i, n + (i + 1) % n) for i in range(n)]
    me = bpy.data.meshes.new(ps["name"])
    me.from_pydata(verts, [], faces)
    me.update()
    return me


def _mesh_leaf(leaf):
    w, h, t = leaf["width"], leaf["height"], leaf["thick"]
    # swing about +z at the hinge. Closed (open=0) = leaf lies along +u inside
    # the aperture [a0,a1]; opening rotates into the -n side. (A closed leaf
    # along -u would be embedded in the wall solid, and the intersecting
    # solids z-fight as large black wall patches.)
    ang = leaf["wall_yaw"] - math.radians(leaf["open_deg"])
    ca, sa = math.cos(ang), math.sin(ang)
    hx, hy, hz = leaf["hinge"]
    u = (ca, sa)
    nrm = (-sa, ca)
    verts, faces = [], []

    def _prism(du0, du1, dn0, dn1, z0, z1):
        du0, du1 = sorted((du0, du1))
        dn0, dn1 = sorted((dn0, dn1))
        z0, z1 = sorted((z0, z1))
        base = len(verts)
        for du, dn in [(du0, dn0), (du1, dn0), (du1, dn1), (du0, dn1)]:
            verts.append((hx + u[0] * du + nrm[0] * dn,
                          hy + u[1] * du + nrm[1] * dn, hz + z0))
        for du, dn in [(du0, dn0), (du1, dn0), (du1, dn1), (du0, dn1)]:
            verts.append((hx + u[0] * du + nrm[0] * dn,
                          hy + u[1] * du + nrm[1] * dn, hz + z1))
        faces.extend([tuple(base + i for i in f) for f in _BOX_FACES])

    _prism(0, w, -t / 2, t / 2, 0, h)                      # the leaf slab
    # door styles: relief strips proud 2mm on both faces (geometry reads as
    # panels/grooves under raking light). The leaf renders in its per-door
    # finish material and the handle in a dedicated metal slot (returned
    # face range -> material_index 1).
    style = leaf.get("style", "flat")
    P = 0.002
    if style in ("panel2", "panel4"):
        rows = 2 if style == "panel2" else 4
        m = 0.09                                            # stile margin
        for r in range(rows):
            z0 = h * (0.06 + 0.88 * r / rows) + 0.02
            z1 = h * (0.06 + 0.88 * (r + 1) / rows) - 0.02
            for sgn in (-1, 1):
                _prism(m, w - m, sgn * t / 2 - (P if sgn < 0 else 0),
                       sgn * t / 2 + (P if sgn > 0 else 0), z0, z1)
    elif style == "grooved":
        for i in range(3):
            du0 = w * (0.25 + 0.2 * i) - 0.008
            for sgn in (-1, 1):
                _prism(du0, du0 + 0.016,
                       sgn * t / 2 - (P if sgn < 0 else 0),
                       sgn * t / 2 + (P if sgn > 0 else 0),
                       0.08, h - 0.08)
    # handle both sides at the free edge (lever = bar, knob = cube-ish)
    handle_f0 = len(faces)                    # faces >= this index -> metal
    hu = w - 0.075
    hz2 = min(1.02, h * 0.5)
    if leaf.get("handle", "lever") == "lever":
        for sgn in (-1, 1):
            _prism(hu - 0.065, hu + 0.02, sgn * (t / 2), sgn * (t / 2 + 0.022),
                   hz2 - 0.011, hz2 + 0.011)
    else:
        for sgn in (-1, 1):
            _prism(hu - 0.025, hu + 0.025, sgn * (t / 2), sgn * (t / 2 + 0.05),
                   hz2 - 0.025, hz2 + 0.025)
    me = bpy.data.meshes.new(leaf["name"])
    me.from_pydata(verts, [], faces)
    me.update()
    return me, handle_f0




def _cct_to_rgb(k):
    """Planckian locus approximation (Tanner Helland fit), 1000-12000K."""
    k = max(1000.0, min(12000.0, k)) / 100.0
    if k <= 66:
        r = 255.0
        g = 99.4708025861 * math.log(k) - 161.1195681661
        b = 0.0 if k <= 19 else 138.5177312231 * math.log(k - 10) - 305.0447927307
    else:
        r = 329.698727446 * ((k - 60) ** -0.1332047592)
        g = 288.1221695283 * ((k - 60) ** -0.0755148492)
        b = 255.0
    clamp = lambda v: max(0.0, min(255.0, v)) / 255.0
    return (clamp(r), clamp(g), clamp(b))

def _mat_principled(name, rgba, rough=0.6, metallic=0.0):
    m = bpy.data.materials.new(name); m.use_nodes = True
    bsdf = m.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = rgba
    bsdf.inputs["Roughness"].default_value = rough
    bsdf.inputs["Metallic"].default_value = metallic
    return m


def _mat_glass(name):
    m = bpy.data.materials.new(name); m.use_nodes = True
    nt = m.node_tree; nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    g = nt.nodes.new("ShaderNodeBsdfGlass")
    g.inputs["IOR"].default_value = 1.45
    nt.links.new(g.outputs["BSDF"], out.inputs["Surface"])
    return m


def _add_instance_jitter(nt, color_socket, base_v=1.0, hue_off=0.0,
                         v_amp=0.06, hue_amp=0.015):
    """Per-instance ObjectInfo.Random color jitter (the domain-randomization
    mechanism of the first-generation generator; cross-view stable because
    Random is per object). Returns the jittered color socket to plug into
    Base Color."""
    oi = nt.nodes.new("ShaderNodeObjectInfo")
    mrv = nt.nodes.new("ShaderNodeMapRange")
    mrv.inputs["From Min"].default_value = 0.0
    mrv.inputs["From Max"].default_value = 1.0
    mrv.inputs["To Min"].default_value = base_v * (1.0 - v_amp)
    mrv.inputs["To Max"].default_value = base_v * (1.0 + v_amp)
    mrh = nt.nodes.new("ShaderNodeMapRange")
    mrh.inputs["From Min"].default_value = 0.0
    mrh.inputs["From Max"].default_value = 1.0
    mrh.inputs["To Min"].default_value = 0.5 + hue_off - hue_amp
    mrh.inputs["To Max"].default_value = 0.5 + hue_off + hue_amp
    hsv = nt.nodes.new("ShaderNodeHueSaturation")
    nt.links.new(oi.outputs["Random"], mrv.inputs["Value"])
    nt.links.new(oi.outputs["Random"], mrh.inputs["Value"])
    nt.links.new(mrv.outputs["Result"], hsv.inputs["Value"])
    nt.links.new(mrh.outputs["Result"], hsv.inputs["Hue"])
    nt.links.new(color_socket, hsv.inputs["Color"])
    return hsv.outputs["Color"]


def _mat_image_pbr(name, maps, entry):
    """Real PBR set: Color(sRGB)->jitter->Base Color;
    Roughness(Non-Color)->multiply; NormalGL(Non-Color)->NormalMap.
    UVs are metric already (1 tile = texel_m meters), no extra mapping."""
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    uvn = nt.nodes.new("ShaderNodeUVMap")
    uvn.uv_map = "UVMap"
    # UVs are metric (1 UV unit = 1 m); per-material texel via Mapping so
    # mixed-texel objects (wood legs + fabric body) each scale correctly
    mpn = nt.nodes.new("ShaderNodeMapping")
    inv = 1.0 / max(entry.get("texel_m", 1.0), 1e-6)
    mpn.inputs["Scale"].default_value = (inv, inv, 1.0)
    nt.links.new(uvn.outputs["UV"], mpn.inputs["Vector"])
    tc = nt.nodes.new("ShaderNodeTexImage")
    tc.image = bpy.data.images.load(maps["color"], check_existing=True)
    nt.links.new(mpn.outputs["Vector"], tc.inputs["Vector"])
    jout = _add_instance_jitter(nt, tc.outputs["Color"],
                                base_v=entry.get("tint_v", 1.0),
                                hue_off=entry.get("tint_h", 0.0))
    nt.links.new(jout, bsdf.inputs["Base Color"])
    if maps.get("roughness"):
        tr = nt.nodes.new("ShaderNodeTexImage")
        tr.image = bpy.data.images.load(maps["roughness"], check_existing=True)
        tr.image.colorspace_settings.name = "Non-Color"
        nt.links.new(mpn.outputs["Vector"], tr.inputs["Vector"])
        mm = nt.nodes.new("ShaderNodeMath")
        mm.operation = "MULTIPLY"
        mm.inputs[1].default_value = entry.get("rough_mult", 1.0)
        nt.links.new(tr.outputs["Color"], mm.inputs[0])
        nt.links.new(mm.outputs["Value"], bsdf.inputs["Roughness"])
    if maps.get("normal_gl"):
        tn = nt.nodes.new("ShaderNodeTexImage")
        tn.image = bpy.data.images.load(maps["normal_gl"], check_existing=True)
        tn.image.colorspace_settings.name = "Non-Color"
        nt.links.new(mpn.outputs["Vector"], tn.inputs["Vector"])
        nm = nt.nodes.new("ShaderNodeNormalMap")
        nm.uv_map = "UVMap"
        nt.links.new(tn.outputs["Color"], nm.inputs["Color"])
        nt.links.new(nm.outputs["Normal"], bsdf.inputs["Normal"])
    return m


def _mat_procedural_entry(name, entry):
    """Physical-domain procedural color + the per-instance Random chain.

    Flat RGB reads as 'untextured' on chair seats, lampshades, books and
    clutter, hence two ground-truth-safe additions (shading only):
    - per-class noise micro-structure (value mottle x bump);
    - wider per-instance hue for paper (book spines: each book its own
      colour) and ceramic (vases/figurines get glaze variety)."""
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    rgbn = nt.nodes.new("ShaderNodeRGB")
    r, g, b = entry["rgb"]
    rgbn.outputs[0].default_value = (r, g, b, 1.0)
    hue_amp = {"paper": 0.45, "ceramic": 0.35}.get(entry["cls"], 0.015)
    jout = _add_instance_jitter(nt, rgbn.outputs[0], v_amp=0.08,
                                hue_amp=hue_amp)
    # class-tuned micro-structure: (noise scale 1/m, value mottle amp, bump).
    # Scales are sub-centimetre on purpose: at 18-40/m a 2m tabletop renders
    # as crumpled cloth (macro blobs).
    _NOISE = {"fabric": (160.0, 0.08, 0.12), "paper": (120.0, 0.05, 0.08),
              "ceramic": (150.0, 0.03, 0.05), "carpet": (60.0, 0.10, 0.25),
              "wood": (140.0, 0.05, 0.08), "leather": (120.0, 0.06, 0.12)}
    scale, mamp, bstr = _NOISE.get(entry["cls"], (100.0, 0.05, 0.08))
    nz = nt.nodes.new("ShaderNodeTexNoise")
    nz.inputs["Scale"].default_value = scale
    nz.inputs["Detail"].default_value = 4.0
    mrange = nt.nodes.new("ShaderNodeMapRange")
    mrange.inputs["To Min"].default_value = 1.0 - mamp
    mrange.inputs["To Max"].default_value = 1.0 + mamp
    nt.links.new(nz.outputs["Fac"], mrange.inputs["Value"])
    mul = nt.nodes.new("ShaderNodeMixRGB")
    mul.blend_type = "MULTIPLY"
    mul.inputs["Fac"].default_value = 1.0
    nt.links.new(jout, mul.inputs["Color1"])
    nt.links.new(mrange.outputs["Result"], mul.inputs["Color2"])
    nt.links.new(mul.outputs["Color"], bsdf.inputs["Base Color"])
    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = bstr
    nt.links.new(nz.outputs["Fac"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    bsdf.inputs["Roughness"].default_value = entry["rough"]
    bsdf.inputs["Metallic"].default_value = entry.get("metallic", 0.0)
    return m


def _mat_screen(name, entry, assets_root, ar):
    """A television that is on: Emission surface (image content cover-fit
    on the screen's normalized UV, or a procedural glow color). Emission node
    -> _mat_flags 'emissive' (covered by the ground-truth ambiguity flags)."""
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs["Strength"].default_value = entry.get("nits", 3.0)
    if entry.get("image"):
        img = bpy.data.images.load(str(Path(assets_root) / entry["image"]),
                                   check_existing=True)
        iw, ih = (img.size[0] or 1), (img.size[1] or 1)
        ar_img = iw / ih
        su = min(ar / ar_img, 1.0)
        sv = min(ar_img / ar, 1.0)
        uvn = nt.nodes.new("ShaderNodeUVMap")
        uvn.uv_map = "UVMap"
        mp = nt.nodes.new("ShaderNodeMapping")
        mp.inputs["Scale"].default_value = (su, sv, 1.0)
        mp.inputs["Location"].default_value = ((1 - su) / 2, (1 - sv) / 2, 0)
        ti = nt.nodes.new("ShaderNodeTexImage")
        ti.image = img
        nt.links.new(uvn.outputs["UV"], mp.inputs["Vector"])
        nt.links.new(mp.outputs["Vector"], ti.inputs["Vector"])
        nt.links.new(ti.outputs["Color"], em.inputs["Color"])
    else:
        r, g, b = entry.get("rgb", [0.7, 0.8, 0.9])
        em.inputs["Color"].default_value = (r, g, b, 1.0)
    nt.links.new(em.outputs["Emission"], out.inputs["Surface"])
    return m


def _mat_decal(name, img_path, canvas_w, canvas_h):
    """Painting/poster canvas: image cover-fit onto the normalized canvas UV
    (crop, never stretch); sRGB color path."""
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    img = bpy.data.images.load(img_path, check_existing=True)
    iw, ih = (img.size[0] or 1), (img.size[1] or 1)
    ar_img = iw / ih
    ar_c = canvas_w / max(canvas_h, 1e-6)
    su = min(ar_c / ar_img, 1.0)
    sv = min(ar_img / ar_c, 1.0)
    uvn = nt.nodes.new("ShaderNodeUVMap")
    uvn.uv_map = "UVMap"
    mp = nt.nodes.new("ShaderNodeMapping")
    mp.inputs["Scale"].default_value = (su, sv, 1.0)
    mp.inputs["Location"].default_value = ((1 - su) / 2, (1 - sv) / 2, 0.0)
    ti = nt.nodes.new("ShaderNodeTexImage")
    ti.image = img
    nt.links.new(uvn.outputs["UV"], mp.inputs["Vector"])
    nt.links.new(mp.outputs["Vector"], ti.inputs["Vector"])
    nt.links.new(ti.outputs["Color"], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = 0.55
    return m


def _apply_box_uv(me, texel_m=1.0, normalize=False):
    """Metric box-projected UVs on a bpy mesh (twin of furniture.
    box_project_uv, generalized to n-gons). Orientation-preserving axis
    table -> no mirrored UV faces. normalize=True rescales to [0,1] over the
    mesh bbox of the projection (decal canvases)."""
    from lenscope.genesis.furniture import _UV_AXES
    uvl = me.uv_layers.new(name="UVMap")
    verts = me.vertices
    lo_u = lo_v = float("inf")
    hi_u = hi_v = float("-inf")
    vals = []
    for poly in me.polygons:
        n = poly.normal
        k = max(range(3), key=lambda i: abs(n[i]))
        sgn = 1 if n[k] >= 0 else -1
        ui, us, vi, vs = _UV_AXES[(k, sgn)]
        for li in range(poly.loop_start, poly.loop_start + poly.loop_total):
            co = verts[me.loops[li].vertex_index].co
            u = us * co[ui] / texel_m
            v = vs * co[vi] / texel_m
            vals.append((li, u, v))
            lo_u, hi_u = min(lo_u, u), max(hi_u, u)
            lo_v, hi_v = min(lo_v, v), max(hi_v, v)
    if normalize and hi_u > lo_u and hi_v > lo_v:
        for (li, u, v) in vals:
            uvl.data[li].uv = ((u - lo_u) / (hi_u - lo_u),
                               (v - lo_v) / (hi_v - lo_v))
    else:
        for (li, u, v) in vals:
            uvl.data[li].uv = (u, v)


def _mat_emission(name, cct_k, strength=5.0):
    m = bpy.data.materials.new(name); m.use_nodes = True
    nt = m.node_tree; nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    em = nt.nodes.new("ShaderNodeEmission")
    bb = nt.nodes.new("ShaderNodeBlackbody")
    bb.inputs["Temperature"].default_value = cct_k
    nt.links.new(bb.outputs["Color"], em.inputs["Color"])
    em.inputs["Strength"].default_value = strength
    nt.links.new(em.outputs["Emission"], out.inputs["Surface"])
    return m


def _exposure_probe_servo(spec, scene):
    """Exposure servo: tiny equirectangular probe renders (96x48, low spp)
    at furniture-clear room points act as a scene-level light meter. The
    Nishita daylight term cannot be bridged analytically to the lm/W lamp
    convention (sun-window geometry dominates), so the realized median
    luminance is measured at the analytic-prior exposure and servoed to the
    spec's mid-gray target. Unique per scene (cross-view constant);
    scene-to-scene brightness diversity survives via spec.exposure_target.
    This is a declared camera auto-exposure model, not a per-view energy
    cancellation: radiance structure (sun bands, dark corridors, night vs
    day interiors) is untouched."""
    import tempfile
    import numpy as np

    def srgb_to_lin(c):
        c = max(c, 0.0)
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    # probe points: 3 biggest + 2 smallest rooms (real-material albedo
    # diversity widens the per-room median spread; probing only big rooms
    # leaves small dim rooms outside the servo's view, under the threshold)
    by_area = sorted(spec.rooms, key=lambda r: -r.area)
    rooms = by_area[:3] + [r for r in by_area[-2:] if r not in by_area[:3]]
    pts = []
    for room in rooms:
        cx = sum(p[0] for p in room.poly) / len(room.poly)
        cy = sum(p[1] for p in room.poly) / len(room.poly)
        blockers = [f for f in spec.furniture
                    if f.ftype not in ("curtain", "mirror", "rug")]
        deps = bpy.context.evaluated_depsgraph_get()
        def ray_clear(x, y, z=1.4, need=0.30):
            # geometric footprint filters are not enough (a probe inside a
            # wardrobe reads ~0 and the servo overshoots), so verify with
            # actual scene raycasts
            for d in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0),
                      (0, 0, 1), (0, 0, -1)):
                hit, loc, *_ = scene.ray_cast(deps, (x, y, z), d)
                if hit and (abs(loc[0] - x) + abs(loc[1] - y)
                            + abs(loc[2] - z)) < need:
                    return False
            return True
        best = (None, -1.0)
        for (x, y) in [(cx, cy)] + [(cx + dx, cy + dy)
                                    for dx in (-1.0, 0.0, 1.0)
                                    for dy in (-1.0, 0.0, 1.0)]:
            d = min((math.hypot(x - f.x, y - f.y)
                     - math.hypot(f.hx, f.hy) for f in blockers),
                    default=9.0)
            if not ray_clear(x, y):
                continue
            if d > best[1]:
                best = ((x, y), d)
        pts.append(best[0] if best[0] is not None else (cx, cy))

    cam_data = bpy.data.cameras.new("ProbeCam")
    cam_data.type = "PANO"
    cam_data.panorama_type = "EQUIRECTANGULAR"
    cam = bpy.data.objects.new("ProbeCam", cam_data)
    cam.rotation_euler = (math.pi / 2, 0.0, 0.0)
    scene.collection.objects.link(cam)
    old_cam = scene.camera
    scene.camera = cam
    scene.cycles.samples = 16
    scene.cycles.use_denoising = False
    scene.render.resolution_x = 96
    scene.render.resolution_y = 48
    # parity with render_v2, which outputs Standard sRGB; the Blender 4.1
    # default, AgX, would break the EOTF inversion below
    scene.view_settings.view_transform = "Standard"
    scene.render.image_settings.file_format = "PNG"
    tmp = Path(tempfile.gettempdir()) / f"genesis_probe_{spec.seed}.png"
    scene.render.filepath = str(tmp)

    def measure():
        meds = []
        for (x, y) in pts:
            cam.location = (x, y, 1.4)
            bpy.ops.render.render(write_still=True)
            img = bpy.data.images.load(str(tmp))
            w, h = img.size
            a = np.array(img.pixels[:], np.float32).reshape(h, w, 4)[..., :3]
            bpy.data.images.remove(img)
            meds.append(float(np.median(
                a @ np.array([0.2126, 0.7152, 0.0722], np.float32))))
        # a near-zero probe = embedded camera, not a dark room (backup to
        # ray_clear); exclude it from the servo mean
        ok = [m for m in meds if m > 0.01]
        return (float(np.mean(ok)) if ok else float(max(meds))), meds

    prior = spec.film_exposure
    trail = []
    tgt_lin = srgb_to_lin(spec.exposure_target)
    # two servo iterations: exact piecewise-sRGB inversion, second pass trims
    # residual nonlinearity (indirect bounce scales slightly nonlinearly)
    for _ in range(2):
        med, meds = measure()
        trail.append({"exposure": round(scene.cycles.film_exposure, 3),
                      "probe_medians_srgb": [round(m, 4) for m in meds]})
        if med <= 1e-4:
            break
        scale = tgt_lin / max(srgb_to_lin(med), 1e-6)
        newexp = float(min(max(scene.cycles.film_exposure * scale, 0.05), 20.0))
        if abs(newexp - scene.cycles.film_exposure) < 0.02:
            scene.cycles.film_exposure = newexp
            break
        scene.cycles.film_exposure = newexp
    tmp.unlink(missing_ok=True)
    spec.film_exposure = float(scene.cycles.film_exposure)
    spec.exposure_derivation.update({
        "mode": "probe-servo", "target_srgb": spec.exposure_target,
        "analytic_prior": round(prior, 3), "servo_trail": trail,
        "servoed_exposure": round(spec.film_exposure, 3)})
    bpy.data.objects.remove(cam)
    bpy.data.cameras.remove(cam_data)
    scene.camera = old_cam


def build(spec: GenesisSpec, out_dir: Path, pack_textures=True) -> Path:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    col = scene.collection
    real = solids.realize(spec)

    mats = {
        "wall": _mat_principled("GenesisWall", (0.82, 0.80, 0.76, 1.0), 0.85),
        "floor": _mat_principled("GenesisFloor", (0.45, 0.33, 0.22, 1.0), 0.5),
        "ceiling": _mat_principled("GenesisCeiling", (0.90, 0.90, 0.90, 1.0), 0.9),
        "glass": _mat_glass("GenesisGlass"),
        "trim": _mat_principled("GenesisTrim", (0.92, 0.92, 0.93, 1.0), 0.4),
        "skirting": _mat_principled("GenesisSkirting", (0.88, 0.88, 0.86, 1.0), 0.5),
        "placeholder": _mat_principled("GenesisPlaceholder", (0.35, 0.42, 0.50, 1.0), 0.7),
        "ground": _mat_principled("GenesisGround", (0.28, 0.30, 0.24, 1.0), 0.95),
    }
    shade_mats = {}                      # per-on-light CCT emission
    off_shade = None                     # shared dark shade for off lamps
    finish_mats = {}                     # door/window finishes
    bdrop_mats = {}                      # per-backdrop gray

    lights = real["lights"]
    backdrops = real.get("backdrops", [])

    # material engine: spec.materials entries -> node graphs.
    # Empty spec.materials => flat palette fallback.
    from lenscope.genesis import materials as _matmod
    assets_root = spec.assets_lib.get("root") or _matmod.DEFAULT_ASSETS_ROOT
    # material-name prefixes: "G3_" for the materials of spec.materials (image, procedural, decal, screen),
    # "G4t_" for tinted clutter-item parts; the texel checks at the end select image materials by "G3_"
    mat_cache = {}
    room_by_suffix = {r.name[-2:]: r.name for r in spec.rooms}
    accent = spec.materials.get("accent")

    def mat_entry(key, fallback_kind="trim"):
        entry = spec.materials.get(key)
        if entry is None:
            return mats.get(fallback_kind, mats["trim"])
        if key in mat_cache:
            return mat_cache[key]
        m = None
        if entry["mode"] == "image":
            maps = _matmod.maps_for_set(assets_root, entry["set"])
            if maps is not None:
                m = _mat_image_pbr(f"G3_{key.replace('/', '_')}", maps, entry)
        if m is None and entry["mode"] in ("image", "procedural"):
            if entry["mode"] == "image":
                # library moved/deleted after spec build: physical fallback
                entry = _matmod.draw_procedural(
                    __import__("random").Random(hash(key) & 0xffff),
                    entry["cls"])
            m = _mat_procedural_entry(f"G3_{key.replace('/', '_')}", entry)
        mat_cache[key] = m
        return m

    def mat_for(kind, light_idx=None, cct=4000.0, b_idx=None):
        nonlocal off_shade
        if kind == "lightshade":
            if not getattr(lights[light_idx], "on", True):
                if off_shade is None:    # off lamps keep geometry, no glow
                    # a 0.20 albedo makes an off pendant's downward face
                    # render exactly zero over a dark floor (indirect-only).
                    # Real off shades are light fabric or paper (~0.5). The
                    # procedural entry adds fabric micro-structure and
                    # per-instance jitter (else shades read as untextured).
                    off_shade = _mat_procedural_entry(
                        "GenesisShadeOff", {"cls": "fabric",
                                            "rgb": [0.52, 0.49, 0.44],
                                            "rough": 0.7})
                return off_shade
            if light_idx not in shade_mats:
                shade_mats[light_idx] = _mat_emission(
                    f"GenesisShade_{light_idx}", cct)
            return shade_mats[light_idx]
        if kind.startswith(("doorframe:", "windowframe:", "doorleaf:")):
            # per-finish door/window materials. Designer rule:
            # colored leaves get white frames (classic trim pairing); wood
            # stays wood-on-wood; dark_alu windows read as aluminium.
            fam, fin = kind.split(":", 1)
            if fam in ("doorframe", "windowframe") \
                    and fin in ("sage", "graphite", "navy"):
                fin = "white"
            key2 = f"{fam}:{fin}"
            if key2 not in finish_mats:
                pal = {
                    "white": ({"cls": "paper", "rgb": [0.90, 0.90, 0.91],
                               "rough": 0.42}, 0.0),
                    "wood": ({"cls": "wood", "rgb": [0.44, 0.33, 0.23],
                              "rough": 0.5}, 0.0),
                    "sage": ({"cls": "paper", "rgb": [0.58, 0.66, 0.56],
                              "rough": 0.5}, 0.0),
                    "graphite": ({"cls": "paper", "rgb": [0.22, 0.23, 0.25],
                                  "rough": 0.45}, 0.0),
                    "navy": ({"cls": "paper", "rgb": [0.16, 0.21, 0.33],
                              "rough": 0.45}, 0.0),
                    "dark_alu": ({"cls": "metal", "rgb": [0.14, 0.15, 0.16],
                                  "rough": 0.35}, 0.9),
                }[fin]
                ent = dict(pal[0])
                if pal[1] > 0:
                    ent["metallic"] = pal[1]
                finish_mats[key2] = _mat_procedural_entry(
                    f"Genesis_{fam}_{fin}", ent)
            return finish_mats[key2]
        if kind == "handle_metal":
            if "handle_metal" not in finish_mats:
                finish_mats["handle_metal"] = _mat_procedural_entry(
                    "GenesisHandle", {"cls": "metal",
                                      "rgb": [0.72, 0.71, 0.66],
                                      "rough": 0.22, "metallic": 0.95})
            return finish_mats["handle_metal"]
        if kind == "backdrop":
            if b_idx not in bdrop_mats:
                g = backdrops[b_idx].gray if b_idx is not None else 0.3
                gr = (g, g * 1.04, g * 0.94, 1.0)   # slight hue variation
                bdrop_mats[b_idx] = _mat_principled(
                    f"GenesisBackdrop_{b_idx}", gr, 0.9)
            return bdrop_mats[b_idx]
        return mats.get(kind, mats["trim"])

    for b in real["boxes"]:
        me = _mesh_obox(b)
        ob = bpy.data.objects.new(b["name"], me)
        li, cct, bi, key = None, 4000.0, None, None
        if b["kind"] == "lightshade":
            li = int(b["name"].split("_")[1])
            cct = lights[li].cct_k
        elif b["kind"] == "backdrop":
            bi = int(b["name"].split("_")[-1])
        elif b["kind"] == "wall":
            key = "wall"
            if accent and (b["name"].startswith(f"Wall_{accent['wall_run']}_")
                           or b["name"].startswith(
                               f"InternalWall_{accent['wall_run']}_")):
                key = "accent"
        elif b["kind"] == "ground":
            key = "ground"
        if key is not None and key in spec.materials:
            ob.data.materials.append(mat_entry(key, b["kind"]))
        else:
            ob.data.materials.append(mat_for(b["kind"], li, cct, bi))
        _apply_box_uv(me)
        col.objects.link(ob)
    for ps in real["polyslabs"]:
        me = _mesh_polyslab(ps)
        ob = bpy.data.objects.new(ps["name"], me)
        key = None
        if ps["kind"] == "floor":
            room = room_by_suffix.get(ps["name"].split("_")[-1])
            key = f"floor/{room}" if room else None
        elif ps["kind"] == "ceiling":
            key = "ceiling"
        if key is not None and key in spec.materials:
            ob.data.materials.append(mat_entry(key, ps["kind"]))
        else:
            ob.data.materials.append(mat_for(ps["kind"]))
        _apply_box_uv(me)
        col.objects.link(ob)

    # furniture/clutter meshes: per-item material entries (real PBR sets or
    # physical procedural) via mat_entry; fixed engine kinds keep their node
    # forms (tv=dark glossy principled, mirror=metallic principled, as
    # _mat_flags expects; ceramic/paper = procedural with the per-instance
    # Random chain: book spines get their hue variety from it, cross-view
    # stable)
    fmats = {
        # wood/fabric fallbacks (clutter frames, unlisted kinds) get the
        # procedural-entry form too (Random chain + micro-structure)
        "wood": _mat_procedural_entry(
            "GenesisWood", {"cls": "wood", "rgb": [0.42, 0.34, 0.26],
                            "rough": 0.6}),
        "fabric": _mat_procedural_entry(
            "GenesisFabric", {"cls": "fabric", "rgb": [0.52, 0.50, 0.46],
                              "rough": 0.92}),
        "metal": _mat_principled("GenesisMetal", (0.62, 0.63, 0.65, 1.0), 0.35,
                                 metallic=0.9),
        "seam": _mat_principled("GenesisSeam", (0.06, 0.055, 0.05, 1.0), 0.8),
        "ceramic": _mat_procedural_entry(
            "GenesisCeramic", {"cls": "ceramic", "rgb": [0.84, 0.83, 0.80],
                               "rough": 0.15}),
        "paper": _mat_procedural_entry(
            "GenesisPaper", {"cls": "paper", "rgb": [0.60, 0.50, 0.42],
                             "rough": 0.75}),
        # real switched-off screens reflect ~4-5%; 0.012 falls below the
        # 0.02 pure-black check and reads as a defect
        "tv": _mat_principled("GenesisTvGlass", (0.04, 0.04, 0.045, 1.0), 0.08,
                              metallic=0.25),
        "mirror": _mat_principled("GenesisMirror", (0.9, 0.9, 0.92, 1.0), 0.02,
                                  metallic=1.0),
    }
    furn_by_name = {f.name: f for f in spec.furniture}
    clutter_by_name = {c.name: c for c in spec.clutter}   # per-item tints
    by_obj = {}
    for fm in real.get("fmeshes", []):
        by_obj.setdefault(fm["obj"], []).append(fm)
    for obj_name, parts in by_obj.items():
        me = bpy.data.meshes.new(obj_name)
        vs, fcs, mat_ids, mats_here = [], [], [], []
        is_canvas_obj = all(pt["kind"] in ("canvas", "tv") for pt in parts)
        for pt in parts:
            base = len(vs)
            vs += [tuple(v) for v in pt["verts"]]
            item = pt.get("item", obj_name)
            if pt["kind"] == "lightshade":
                li2 = int(item.split("_")[1])
                m = mat_for("lightshade", li2, lights[li2].cct_k)
            elif pt["kind"] == "tv" and pt["flags"].get("screen_on") \
                    and f"screen/{item}" in spec.materials:
                skey = f"screen/{item}"
                if skey not in mat_cache:
                    fs2 = furn_by_name.get(item)
                    ar2 = (fs2.params.get("tv_w", 1.2)
                           / max(fs2.params.get("tv_w", 1.2)
                                 / fs2.params.get("tv_ar", 1.78), 1e-6)) \
                        if fs2 else 1.78
                    mat_cache[skey] = _mat_screen(
                        f"G3_{skey.replace('/', '_')}",
                        spec.materials[skey], assets_root, ar2)
                m = mat_cache[skey]
            elif pt["kind"] == "canvas":
                dkey = f"decal/{item}"
                entry = spec.materials.get(dkey)
                if entry and entry.get("mode") == "decal":
                    fs = furn_by_name.get(item)
                    cw = fs.params.get("w", 0.5) if fs else 0.5
                    ch = fs.params.get("h", 0.7) if fs else 0.7
                    if dkey not in mat_cache:
                        mat_cache[dkey] = _mat_decal(
                            f"G3_{dkey.replace('/', '_')}",
                            str(Path(assets_root) / entry["image"]), cw, ch)
                    m = mat_cache[dkey]
                else:
                    m = mat_entry(dkey, "trim") if entry else fmats["paper"]
            elif f"furn/{item}/{pt['kind']}" in spec.materials:
                m = mat_entry(f"furn/{item}/{pt['kind']}")
            else:
                # per-clutter-item macro tint + gloss from the manifest
                # (small-item variety; one shared class material per kind
                # looks like a staged show flat). Cache-keyed on the
                # quantized tint so the material count stays bounded.
                cs2 = clutter_by_name.get(item)
                tint = (cs2.params or {}).get("tint") if cs2 else None
                if tint is not None and pt["kind"] in (
                        "ceramic", "paper", "wood", "fabric", "metal",
                        "foliage"):
                    gl = float((cs2.params or {}).get("gloss", 0.5))
                    key = ("ct", pt["kind"], round(tint[0], 2),
                           round(tint[1], 2), round(tint[2], 2),
                           round(gl, 1))
                    if key not in mat_cache:
                        ent = {"cls": pt["kind"], "rgb": list(tint),
                               "rough": gl}
                        if pt["kind"] == "metal":
                            ent["metallic"] = 0.9
                        mat_cache[key] = _mat_procedural_entry(
                            f"G4t_{len(mat_cache):04d}", ent)
                    m = mat_cache[key]
                else:
                    m = fmats.get(pt["kind"]) or mats.get(pt["kind"],
                                                          fmats["wood"])
            if m not in mats_here:
                mats_here.append(m)
            mi = mats_here.index(m)
            for f in pt["faces"]:
                fcs.append((base + int(f[0]), base + int(f[1]), base + int(f[2])))
                mat_ids.append(mi)
        me.from_pydata(vs, [], fcs)
        me.update()
        _apply_box_uv(me, normalize=is_canvas_obj)
        _smooth_mesh(me)      # shading normals only, geometry untouched
        ob = bpy.data.objects.new(obj_name, me)
        for m in mats_here:
            ob.data.materials.append(m)
        for pi, poly in enumerate(me.polygons):
            poly.material_index = mat_ids[pi]
        col.objects.link(ob)
    for leaf in real["leaves"]:
        me, handle_f0 = _mesh_leaf(leaf)
        ob = bpy.data.objects.new(leaf["name"], me)
        # slot 0 = the leaf's finish, slot 1 = metal handle
        ob.data.materials.append(
            mat_for(f"doorleaf:{leaf.get('finish', 'white')}"))
        ob.data.materials.append(mat_for("handle_metal"))
        for pi, poly in enumerate(me.polygons):
            poly.material_index = 1 if pi >= handle_f0 else 0
        _apply_box_uv(me)
        col.objects.link(ob)

    H = real["world"]["height"]
    efficacy = real["world"].get("efficacy_lm_w", 160.0)
    for k, L in enumerate(lights):
        if not getattr(L, "on", True):          # on/off habits: no emitter
            continue
        ld = bpy.data.lights.new(f"LightSrc_{k:02d}", type="POINT")
        # node-based lights ignore light.energy (output collapses to the node
        # Emission default ~1W): rooms go near-dark and the camera-visible
        # source disk renders as a black circle on the wall. Plain color +
        # energy keeps watts effective; CCT via a python Planckian
        # approximation; the source disk is hidden from camera rays (the
        # emissive shade is seen, not a floating disk).
        ld.energy = L.lumens / efficacy         # lm/W_rad fixed in the spec
        ld.shadow_soft_size = 0.10 if L.fixture == "pendant" else 0.06
        ld.color = _cct_to_rgb(L.cct_k)
        lz = getattr(L, "z", -1.0)
        if L.fixture == "pendant":
            # an emitter on the solid shade's top face lets the shade block
            # the entire lower hemisphere, so rooms are lit by ceiling bounce
            # only (3x exposures, crushed dark floors, pitch-black
            # under-shade zones). Real pendants throw light down: emitter
            # 2cm below the shade bottom.
            x, y, z = L.x, L.y, H - 0.92
        elif L.fixture == "wall":
            yaw = math.radians(getattr(L, "yaw_deg", 0.0))
            x = L.x + 0.055 * math.sin(yaw)
            y = L.y + 0.055 * math.cos(yaw)
            z = (lz if lz > 0 else 1.75) + 0.02
            ld.shadow_soft_size = 0.05
        elif L.fixture == "floor":
            x, y, z = L.x, L.y, (lz if lz > 0 else 1.45) + 0.10
            ld.shadow_soft_size = 0.05
        elif L.fixture == "table":
            x, y, z = L.x, L.y, (lz if lz > 0 else 0.82)
            ld.shadow_soft_size = 0.05
        else:                                   # flush | bulb
            x, y, z = L.x, L.y, H - 0.16
        ob = bpy.data.objects.new(f"LightSrc_{k:02d}", ld)
        ob.location = (x, y, z)
        ob.visible_camera = False
        col.objects.link(ob)

    world = bpy.data.worlds.new("GenesisWorld"); world.use_nodes = True
    nt = world.node_tree; nt.nodes.clear()
    sky = nt.nodes.new("ShaderNodeTexSky"); sky.sky_type = "NISHITA"
    sky.sun_elevation = math.radians(real["world"]["sun_elevation_deg"])
    sky.sun_rotation = math.radians(real["world"]["sun_azimuth_deg"])
    sky.sun_intensity = real["world"]["sun_intensity"]
    sky.dust_density = real["world"].get("dust_density", 0.0)  # dusk haze
    # moonlight: directional SUN lamp, physically faint (full moon
    # ~0.32 lx -> W/m2 via photopic 683 lm/W), cold 4125K, phase-modulated
    # (Allen fit in spec). Night's directional shadow-caster.
    moon = real["world"].get("moon") or {}
    if moon.get("enabled"):
        md = bpy.data.lights.new("MoonLight", type="SUN")
        md.energy = moon["lux"] / 683.0 * moon.get("render_boost", 1.0)  # W/m2
        md.angle = math.radians(0.53)
        md.color = _cct_to_rgb(moon.get("cct_k", 4125.0))
        mo = bpy.data.objects.new("MoonLight", md)
        el = math.radians(moon["elevation_deg"])
        az = math.radians(moon["azimuth_deg"])
        mo.rotation_euler = (math.pi / 2 - el, 0.0, -az)   # aim from (el, az)
        col.objects.link(mo)
    bg = nt.nodes.new("ShaderNodeBackground")
    bg.inputs["Strength"].default_value = real["world"]["sky_strength"]
    outw = nt.nodes.new("ShaderNodeOutputWorld")
    nt.links.new(sky.outputs["Color"], bg.inputs["Color"])
    nt.links.new(bg.outputs["Background"], outw.inputs["Surface"])
    scene.world = world

    scene.render.engine = "CYCLES"
    scene.cycles.film_exposure = real["world"]["film_exposure"]
    _exposure_probe_servo(spec, scene)

    # numeric acceptance checks (black-artifact classes)
    n_nomat, n_noemit = 0, 0
    for ob in scene.objects:
        if ob.type != "MESH":
            continue
        ok = any(sl.material is not None for sl in ob.material_slots)
        if not ok:
            n_nomat += 1
            print(f"[AUDIT-FAIL] no material: {ob.name}")
    for name, m in shade_mats.items():
        em = next((n for n in m.node_tree.nodes if n.type == "EMISSION"), None)
        if em is None or em.inputs["Strength"].default_value <= 0:
            n_noemit += 1
            print(f"[AUDIT-FAIL] shade emission: {name}")
    assert n_nomat == 0 and n_noemit == 0, "material/emission audit failed"
    print(f"[audit] material coverage 100% ({len([o for o in scene.objects if o.type=='MESH'])} meshes), "
          f"emission>0 for {len(shade_mats)} shades")

    # numeric UV / texel checks (in-build)
    n_meshes = n_nouv = n_flip = 0
    dens = []
    # decal canvases use normalized UVs by design (cover-fit); exclude
    # them from the metric texel population
    img_mats = {m.name for m in bpy.data.materials
                if any(n.type == "TEX_IMAGE" for n in
                       (m.node_tree.nodes if m.use_nodes else []))
                and not m.name.startswith(("G3_decal", "G3_screen"))}
    for ob in scene.objects:
        if ob.type != "MESH":
            continue
        n_meshes += 1
        me = ob.data
        if not me.uv_layers:
            n_nouv += 1
            print(f"[AUDIT-FAIL] no UV: {ob.name}")
            continue
        uvd = me.uv_layers.active.data
        for poly in me.polygons:
            if poly.area < 1e-10:
                continue
            pts = [uvd[li].uv for li in
                   range(poly.loop_start, poly.loop_start + poly.loop_total)]
            s2 = 0.0
            for i in range(len(pts)):
                x0, y0 = pts[i]
                x1, y1 = pts[(i + 1) % len(pts)]
                s2 += x0 * y1 - x1 * y0
            if s2 <= 1e-12:
                n_flip += 1
                if n_flip < 4:
                    print(f"[AUDIT-FAIL] flipped/zero UV: {ob.name} poly")
            # texel CV on axis-aligned faces of image-material objects
            n = poly.normal
            if max(abs(n[0]), abs(n[1]), abs(n[2])) > 0.99 and s2 > 1e-12 \
                    and ob.material_slots and \
                    ob.material_slots[poly.material_index].material and \
                    ob.material_slots[poly.material_index].material.name in img_mats:
                dens.append((abs(s2) / 2.0 / poly.area) ** 0.5)
    cv = 0.0
    if len(dens) > 8:
        mu = sum(dens) / len(dens)
        cv = (sum((d - mu) ** 2 for d in dens) / len(dens)) ** 0.5 / mu
    n_img = sum(1 for e in spec.materials.values() if e.get("mode") == "image")
    n_dec = sum(1 for e in spec.materials.values() if e.get("mode") == "decal")
    used_img = len([m for m in img_mats if m.startswith("G3_")])
    assert n_nouv == 0 and n_flip == 0, "UV audit failed"
    assert cv < 0.10, f"texel CV {cv:.3f} >= 10%"
    if spec.material_image_prob > 0:
        assert used_img > 0, "library present but zero real-PBR materials"
    print(f"[audit] UV coverage 100% ({n_meshes} meshes), 0 flipped; "
          f"texel CV {cv*100:.2f}% over {len(dens)} flat faces; "
          f"entries image/procedural/decal = {n_img}/"
          f"{sum(1 for e in spec.materials.values() if e.get('mode')=='procedural')}/{n_dec}; "
          f"image materials built: {used_img}")

    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"genesis_{spec.seed:06d}"
    spec.save(out_dir / f"{name}_manifest.json")
    blend = out_dir / f"{name}.blend"
    if pack_textures:
        try:
            bpy.ops.file.pack_all()      # self-contained .blend (renders
        except RuntimeError as e:        # without a library path)
            print(f"[warn] pack_all: {e}")
    bpy.ops.wm.save_as_mainfile(filepath=str(blend))
    print(f"[genesis] saved {blend} kind={spec.kind} rooms={len(spec.rooms)} "
          f"doors={len(spec.doors)} windows={len(spec.windows)} "
          f"lights={len(lights)} solids={len(real['boxes'])}")
    return blend


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    def arg(k, default=None):
        return argv[argv.index(k) + 1] if k in argv else default
    seed = int(arg("--seed", 0))
    out = Path(arg("--out", "genesis_scenes"))
    spec_path = arg("--spec")
    if spec_path:
        spec = GenesisSpec.from_json(Path(spec_path).read_text())
    elif "--ge0" in argv:
        spec = build_ge0_spec(seed)
    else:
        # --archetype day|dusk|night: forcing for tests only (production
        # scenes draw the archetype from the seed)
        # --n-rooms N: explicit room count, 10+ supported
        nr = arg("--n-rooms")
        spec = build_ge1_spec(seed, force_archetype=arg("--archetype"),
                              n_rooms=int(nr) if nr else None)
    build(spec, out, pack_textures=("--no-pack" not in argv))


if __name__ == "__main__":
    main()
