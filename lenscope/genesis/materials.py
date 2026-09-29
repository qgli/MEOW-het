"""Material assignment for second-generation scenes (bpy-free).

Every surface's material is drawn during spec construction and recorded in
`spec.materials` with full provenance (which PBR set, decal image or
procedural parameters each surface uses), so the manifest reconstructs the
look exactly. The bpy side (build_scene) only translates entries into node
graphs.

Optional asset library: the CC0 library at $GENESIS_ASSETS provides
  pbr/<category>/<SetName>/<SetName>_1K-JPG_{Color,Roughness,NormalGL,...}.jpg
  decals/{paintings,posters}/*.jpg
An empty or missing library selects pure-procedural mode, so the engine
always runs. The image-vs-procedural mix is a spec parameter.

Physical BRDF domains: albedo and roughness per material class are
constrained to real-material reflectance ranges; image sets inherit their
domain from the photographed material itself, with only a mild declared tint
jitter on top.
"""
from __future__ import annotations

import math
import os
from pathlib import Path

DEFAULT_ASSETS_ROOT = ""

# physical albedo (value) and roughness domains per material class
# (sources: typical measured reflectance ranges; wall paint 0.70-0.85,
# woods 0.30-0.50, fabrics 0.20-0.50, etc.)
_DOMAINS = {
    "wall":     {"alb": (0.70, 0.85), "sat": (0.00, 0.10), "rough": (0.55, 0.90)},
    "ceiling":  {"alb": (0.78, 0.90), "sat": (0.00, 0.04), "rough": (0.70, 0.95)},
    "floor_wood":  {"alb": (0.30, 0.55), "sat": (0.25, 0.50), "rough": (0.25, 0.60)},
    "floor_tile":  {"alb": (0.45, 0.75), "sat": (0.02, 0.15), "rough": (0.10, 0.35)},
    "floor_stone": {"alb": (0.35, 0.65), "sat": (0.02, 0.12), "rough": (0.30, 0.70)},
    "wood":     {"alb": (0.30, 0.50), "sat": (0.30, 0.55), "rough": (0.35, 0.75)},
    "fabric":   {"alb": (0.20, 0.50), "sat": (0.05, 0.45), "rough": (0.85, 1.00)},
    "leather":  {"alb": (0.12, 0.40), "sat": (0.20, 0.50), "rough": (0.30, 0.60)},
    "carpet":   {"alb": (0.20, 0.50), "sat": (0.05, 0.40), "rough": (0.95, 1.00)},
    "metal":    {"alb": (0.55, 0.85), "sat": (0.00, 0.08), "rough": (0.15, 0.50),
                 "metallic": 0.9},
    "ceramic":  {"alb": (0.60, 0.90), "sat": (0.00, 0.20), "rough": (0.05, 0.30)},
    "paper":    {"alb": (0.35, 0.75), "sat": (0.10, 0.60), "rough": (0.60, 0.85)},
    "brick":    {"alb": (0.25, 0.50), "sat": (0.25, 0.50), "rough": (0.70, 0.95)},
    "ground":   {"alb": (0.15, 0.35), "sat": (0.10, 0.30), "rough": (0.85, 1.00)},
    "backdrop": {"alb": (0.10, 0.45), "sat": (0.00, 0.15), "rough": (0.80, 1.00)},
    "stone":    {"alb": (0.30, 0.65), "sat": (0.00, 0.20), "rough": (0.15, 0.55)},
    # plant foliage: green-locked hue (draw_procedural), matte
    "foliage":  {"alb": (0.08, 0.30), "sat": (0.45, 0.85), "rough": (0.55, 0.90)},
}

# surface class -> pbr library category (None = always procedural)
_PBR_CATEGORY = {
    "wall": "wall_plaster_paint", "ceiling": "wall_plaster_paint",
    "floor_wood": "floor_wood", "floor_tile": "floor_tile",
    "floor_stone": "floor_stone",
    "wood": "wood", "fabric": "fabric", "leather": "leather",
    "stone": "floor_stone",
    "carpet": "carpet_rug", "metal": "metal", "brick": "brick",
    "ceramic": None, "paper": None, "ground": None, "backdrop": None,
}

# metric texel: 1 UV tile == this many meters of surface (per class; ambientCG
# sets are photographed at roughly 1m tiles; finer cloth reads better denser)
_TEXEL_M = {
    "wall": 1.5, "ceiling": 1.5, "brick": 1.2,
    "floor_wood": 1.2, "floor_tile": 1.0, "floor_stone": 1.0,
    "wood": 0.8, "fabric": 0.6, "leather": 0.7, "carpet": 1.2, "stone": 0.9,
    "metal": 0.8, "ceramic": 0.5, "paper": 0.4,
    "ground": 4.0, "backdrop": 6.0, "foliage": 0.5,
}

_FLOOR_BY_FUNCTION = {
    # weights over (floor_wood, floor_tile, floor_stone)
    "living":  (0.60, 0.20, 0.20),
    "bedroom": (0.80, 0.05, 0.15),
    "dining":  (0.35, 0.45, 0.20),
    "study":   (0.70, 0.10, 0.20),
    "hallway": (0.30, 0.45, 0.25),
    # wet rooms are tile/stone-dominant
    "bathroom": (0.02, 0.68, 0.30),
    "kitchen":  (0.10, 0.60, 0.30),
    "studio":   (0.55, 0.25, 0.20),   # open-plan single room
}


# Curated-out sets (explicit, reversible): Fabric083 is a strong regular
# gingham (FFT periodicity 25x above the library's next-highest set) that
# renders exactly like Blender's missing-texture checker on cushions and
# aliases downstream, so it reads as a defect. Keep the list short and
# documented.
_CURATED_OUT = {"pbr/fabric/Fabric083"}


def scan_library(root=None):
    """Deterministic library inventory. Missing/empty root -> empty maps
    (pure-procedural invariant)."""
    root = root or os.environ.get("GENESIS_ASSETS", DEFAULT_ASSETS_ROOT)
    out = {"root": str(root), "pbr": {}, "paintings": [], "posters": []}
    if not root:
        return out
    root = Path(root)
    pbr = root / "pbr"
    if pbr.is_dir():
        for cat in sorted(p.name for p in pbr.iterdir() if p.is_dir()):
            sets = []
            for sd in sorted((pbr / cat).iterdir()):
                if not sd.is_dir():
                    continue
                color = sorted(sd.glob("*_Color.jpg")) + \
                    sorted(sd.glob("*_Color.png"))
                if color and f"pbr/{cat}/{sd.name}" not in _CURATED_OUT:
                    sets.append(f"pbr/{cat}/{sd.name}")
            if sets:
                out["pbr"][cat] = sets
    for key, sub in (("paintings", "decals/paintings"),
                     ("posters", "decals/posters")):
        d = root / sub
        if d.is_dir():
            out[key] = [f"{sub}/{p.name}" for p in sorted(d.iterdir())
                        if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
    return out


_MEAN_CACHE = {}


def set_albedo_mean(root, rel_set):
    """Mean luminance of a set's Color map (sRGB-decoded), cached on disk
    beside the library (deterministic; used by the albedo-aware illuminance
    floor -- dark floors need more lumens for equal luminance)."""
    root = str(root)
    if root not in _MEAN_CACHE:
        cache_p = Path(root) / ".set_means_cache.json"
        try:
            import json as _json
            _MEAN_CACHE[root] = _json.loads(cache_p.read_text())
        except Exception:
            _MEAN_CACHE[root] = {}
    cache = _MEAN_CACHE[root]
    if rel_set not in cache:
        maps = maps_for_set(root, rel_set)
        val = 0.45
        if maps is not None:
            try:
                from PIL import Image
                import numpy as _np
                im = _np.asarray(Image.open(maps["color"]).convert("RGB").
                                 resize((64, 64))).astype(_np.float64) / 255.0
                lin = _np.where(im <= 0.04045, im / 12.92,
                                ((im + 0.055) / 1.055) ** 2.4)
                val = float((lin @ [0.2126, 0.7152, 0.0722]).mean())
            except Exception:
                pass
        cache[rel_set] = round(val, 4)
        try:
            import json as _json
            (Path(root) / ".set_means_cache.json").write_text(
                _json.dumps(cache, indent=0))
        except OSError:
            pass
    return cache[rel_set]


_STATS_CACHE = {}


def set_albedo_p95(root, rel_set):
    """p95 of a set's Color map luminance (linear), cached beside the library.

    The domain-normalization tint is capped at 0.97/p95 so the bright tail
    never clips: dark sets (Tiles144 mean 0.16, Wood051 mean 0.04) would need
    a x3-13 gain, and clipping crushes texture detail into flat saturated
    patches that read as a checker grid."""
    root = str(root)
    if root not in _STATS_CACHE:
        cache_p = Path(root) / ".set_p95_cache.json"
        try:
            import json as _json
            _STATS_CACHE[root] = _json.loads(cache_p.read_text())
        except Exception:
            _STATS_CACHE[root] = {}
    cache = _STATS_CACHE[root]
    if rel_set not in cache:
        maps = maps_for_set(root, rel_set)
        val = 0.8
        if maps is not None:
            try:
                from PIL import Image
                import numpy as _np
                im = _np.asarray(Image.open(maps["color"]).convert("RGB").
                                 resize((64, 64))).astype(_np.float64) / 255.0
                lin = _np.where(im <= 0.04045, im / 12.92,
                                ((im + 0.055) / 1.055) ** 2.4)
                # per-channel p95 (a red-heavy set clips in R first)
                val = float(_np.percentile(lin, 95))
            except Exception:
                pass
        cache[rel_set] = round(max(val, 0.05), 4)
        try:
            import json as _json
            (Path(root) / ".set_p95_cache.json").write_text(
                _json.dumps(cache, indent=0))
        except OSError:
            pass
    return cache[rel_set]


def maps_for_set(root, rel_set):
    """Resolve a set's texture files. Returns {color, roughness, normal_gl}
    (absolute paths) or None if incomplete. NormalGL explicitly (Blender is
    GL-convention; DX would invert the green channel -> inverted bumps)."""
    sd = Path(root) / rel_set
    def find(suffix):
        for ext in (".jpg", ".png"):
            hits = sorted(sd.glob(f"*_{suffix}{ext}"))
            if hits:
                return str(hits[0])
        return None
    color = find("Color")
    rough = find("Roughness")
    ngl = find("NormalGL")
    if color is None:
        return None
    return {"color": color, "roughness": rough, "normal_gl": ngl}


def _hsv_to_rgb(h, s, v):
    i = int(h * 6.0) % 6
    f = h * 6.0 - int(h * 6.0)
    p, q, t = v * (1 - s), v * (1 - f * s), v * (1 - (1 - f) * s)
    return [(v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q)][i]


def draw_procedural(rng, cls):
    """Physical-domain procedural color: value from the class albedo range,
    saturation from its sat range, free hue (wood/leather warm-biased)."""
    d = _DOMAINS[cls]
    v = rng.uniform(*d["alb"])
    s = rng.uniform(*d["sat"])
    h = rng.uniform(0.05, 0.11) if cls in ("wood", "leather", "floor_wood",
                                           "brick") \
        else rng.uniform(0.20, 0.38) if cls == "foliage" \
        else rng.uniform(0.0, 1.0)
    rgb = _hsv_to_rgb(h, s, v)
    return {"mode": "procedural", "cls": cls,
            "rgb": [round(c, 4) for c in rgb],
            "rough": round(rng.uniform(*d["rough"]), 3),
            "metallic": d.get("metallic", 0.0),
            "texel_m": _TEXEL_M[cls]}


def draw_surface(rng, cls, lib, image_prob):
    """One surface's material entry: real PBR set (if the class has a
    non-empty category and the coin lands) else physical procedural.
    Floor classes exclude ultra-dark sets (albedo_mean < 0.12, e.g. the
    near-ebony WoodFloor070 at 0.049): a whole ebony floor drags the room
    median below the brightness threshold for physical reasons no lumen
    budget can fix; such floors are rare in real homes, and the sets stay
    available for furniture wood."""
    cat = _PBR_CATEGORY.get(cls)
    sets = lib["pbr"].get(cat, []) if cat else []
    if cls.startswith("floor_") and sets:
        sets = [s for s in sets
                if set_albedo_mean(lib["root"], s) >= 0.12] or sets
    if sets:
        # Generalizes the floor rule to every class: a set is usable only if
        # the clip-safe tint (<=0.97/p95, <=3.0) can lift its mean to >=70% of
        # the lower albedo bound of the class domain. Near-ebony sets (Wood051
        # mean 0.038) would otherwise draw tint 3.0, clip, and render as flat
        # blotches ('checkerboard') on chairs, sofas and floors.
        lo = _DOMAINS[cls]["alb"][0]
        usable = [s for s in sets
                  if min(3.0, 0.97 / set_albedo_p95(lib["root"], s))
                  * set_albedo_mean(lib["root"], s) >= 0.7 * lo]
        sets = usable or []
    if sets and rng.random() < image_prob:
        rel = rng.choice(sets)
        # Domain normalization: ambientCG sets are raw-material photos
        # ("plaster" is grey plaster, not painted walls; several floors are
        # near-ebony). The physical BRDF domain applies to images too: draw a
        # target albedo from the class domain and tint the set so its
        # measured linear mean lands there (texture structure stays, overall
        # reflectance obeys the domain). The clamp keeps extreme sets from
        # over-amplifying.
        mean = set_albedo_mean(lib["root"], rel)
        # domain albedos are linear reflectance (measured-material ranges)
        # and set_albedo_mean is linear too, so the ratio is direct (an extra
        # sRGB->linear conversion here would double-darken textured floors)
        tv = rng.uniform(*_DOMAINS[cls]["alb"])
        # clip-headroom cap: never tint the p95 tail past white (clipping
        # crushes texture into flat patches, the 'checkerboard' artifact).
        cap = min(3.0, 0.97 / set_albedo_p95(lib["root"], rel))
        tint = min(max(tv / max(mean, 0.02), 0.4), cap)
        tint = min(tint * rng.uniform(0.92, 1.08), cap)  # jitter, re-capped
        return {"mode": "image", "cls": cls, "set": rel,
                "albedo_mean": mean,
                "target_albedo_lin": round(tv, 3),
                "tint_v": round(tint, 3),
                "tint_h": round(rng.uniform(-0.02, 0.02), 4),
                "rough_mult": round(rng.uniform(0.9, 1.1), 3),
                "texel_m": _TEXEL_M[cls]}
    return draw_procedural(rng, cls)


def assign_materials(rng, spec, lib):
    """Fill spec.materials: per-room floors, scene walls/ceiling (+ optional
    brick accent run), per-furniture-part classes, decal images for
    paintings/posters. Every entry is full provenance."""
    p_img = spec.material_image_prob
    M = {}
    # architecture
    M["wall"] = draw_surface(rng, "wall", lib, p_img)
    M["ceiling"] = draw_surface(rng, "ceiling", lib,
                                p_img * 0.15)  # homes: smooth painted ceilings
                                               # dominate; textured = rare DR
    M["ground"] = draw_procedural(rng, "ground")
    M["backdrop"] = draw_procedural(rng, "backdrop")
    internal_runs = sorted({w.id for w in spec.walls if w.internal})
    if internal_runs and lib["pbr"].get("brick") and rng.random() < 0.15:
        M["accent"] = {**draw_surface(rng, "brick", lib, 1.0),
                       "wall_run": rng.choice(internal_runs)}
    for room in spec.rooms:
        w = _FLOOR_BY_FUNCTION.get(room.function, (0.5, 0.3, 0.2))
        r = rng.random()
        cls = "floor_wood" if r < w[0] else \
            "floor_tile" if r < w[0] + w[1] else "floor_stone"
        M[f"floor/{room.name}"] = draw_surface(rng, cls, lib, p_img)

    # furniture: one entry per (item, part-kind-class)
    for fs in spec.furniture:
        kinds = _ITEM_KINDS.get(fs.ftype, ())
        for kind in kinds:
            cls = kind
            if fs.ftype == "sofa" and kind == "fabric" \
                    and lib["pbr"].get("leather") and rng.random() < 0.25:
                cls = "leather"
            if fs.ftype == "rug":
                cls = "carpet"
            M[f"furn/{fs.name}/{kind}"] = draw_surface(rng, cls, lib, p_img)
    # screens that are on draw their content (a poster/painting image when
    # screen_use_img is set, else a procedural glow color); provenance as for
    # decals
    for fs in spec.furniture:
        if fs.ftype == "tv_stand" and fs.params.get("screen_on"):
            pool = (lib["posters"] + lib["paintings"]) \
                if fs.params.get("screen_use_img") else []
            if pool:
                M[f"screen/{fs.name}"] = {
                    "mode": "screen", "image": rng.choice(pool),
                    "nits": fs.params.get("screen_nits", 3.0)}
            else:
                M[f"screen/{fs.name}"] = {
                    "mode": "screen", "image": None,
                    "rgb": [round(c, 3) for c in _hsv_to_rgb(
                        rng.uniform(0, 1), rng.uniform(0.05, 0.5),
                        rng.uniform(0.55, 0.95))],
                    "nits": fs.params.get("screen_nits", 3.0)}
    # decals
    for fs in spec.furniture:
        if fs.ftype == "painting":
            pool = lib["paintings"]
            M[f"decal/{fs.name}"] = (
                {"mode": "decal", "image": rng.choice(pool)} if pool
                else draw_procedural(rng, "paper"))
        elif fs.ftype == "poster":
            pool = lib["posters"] or lib["paintings"]
            M[f"decal/{fs.name}"] = (
                {"mode": "decal", "image": rng.choice(pool)} if pool
                else draw_procedural(rng, "paper"))
    spec.materials = M


# which part-kinds each furniture type carries (materials get per-item draws;
# 'seam'/'tv'/'mirror'/'canvas' are fixed engine materials, not drawn here)
_ITEM_KINDS = {
    "sofa": ("wood", "fabric"), "bed": ("wood", "fabric"),
    "table": ("wood",), "chair": ("wood", "fabric"),
    "wardrobe": ("wood", "metal"), "nightstand": ("wood", "metal"),
    "tv_stand": ("wood", "metal"), "bookshelf": ("wood",),
    "mirror": ("wood",), "curtain": ("fabric",), "rug": ("fabric",),
    "painting": ("wood",), "poster": (), "clock": ("wood",),
    # bathroom/kitchen fixtures (ceramic/metal parts keep engine-fixed forms)
    "vanity": ("wood", "stone"), "counter": ("wood", "stone"),
    "wallcabinet": ("wood",), "fridge": ("metal",), "stove": ("metal",),
    "hood": ("metal",), "toilet": (), "bathtub": (), "shower": (),
    "towel_bar": ("fabric",),
    # floor decor and hanging pieces
    "floorplant": ("ceramic", "wood", "foliage"),
    "basket": ("fabric",), "bookstack": ("paper",),
    "suitcase": ("leather",),
    "hangplant": ("ceramic", "metal", "foliage"),
    "lantern": ("paper", "metal"), "mobile": ("wood", "metal", "ceramic"),
    "ceilingfan": ("metal", "wood"),
}
