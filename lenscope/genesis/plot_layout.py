"""Top-down 2D layout plot of a GenesisSpec (bpy-free, PIL).

Layout inspection tool and the floorplan raster of export_layout.
Architectural look: function-tinted room fills, thick walls, door swing
arcs, double-line window glazing, rounded furniture fills, a subtle facing
notch (small triangle on the front edge; --no-facing hides it, the yaw
data always stays in layout.json), lamp icons, legend + scale bar.
Anti-aliased via 2x supersampling.

Usage:
  python lenscope/genesis/plot_layout.py <out_dir> <seed> [seed...] [--no-facing]
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lenscope.genesis.spec import build_ge1_spec              # noqa: E402

SCALE = 60            # px per meter (pre-supersample)
PAD = 52
SS = 2                # supersample factor

_ROOM_FILL = {"living": "#f3e9d7", "bedroom": "#e3edf5", "dining": "#f5e7e3",
              "study": "#e8f0e2", "hallway": "#efeff2"}
_FCOLOR = {"sofa": "#b5651d", "bed": "#4e7f91", "table": "#8b6f47",
           "chair": "#96876f", "wardrobe": "#6b4e71", "nightstand": "#9a8194",
           "tv_stand": "#3e6187", "bookshelf": "#586e46", "rug": "#cdb98f",
           "mirror": "#57b7c2", "curtain": "#9f8cc2", "painting": "#c86e80",
           "poster": "#cfa93a", "clock": "#555555"}
_WALL = "#2b2b2b"
_TEXT = "#3a3a3a"


def _fill_of(ft):
    base = _FCOLOR.get(ft, "#888888")
    r = int(base[1:3], 16); g = int(base[3:5], 16); b = int(base[5:7], 16)
    return (r, g, b, 70)                       # translucent body fill


def plot(spec, path, facing=True):
    xs = [p[0] for r in spec.rooms for p in r.poly]
    ys = [p[1] for r in spec.rooms for p in r.poly]
    w = int((max(xs) - min(xs)) * SCALE * SS) + 2 * PAD * SS
    h = int((max(ys) - min(ys)) * SCALE * SS) + 2 * PAD * SS + 34 * SS
    ox, oy = min(xs), min(ys)

    def T(x, y):
        return (PAD * SS + (x - ox) * SCALE * SS,
                h - 34 * SS - (PAD * SS + (y - oy) * SCALE * SS))

    im = Image.new("RGB", (w, h), "#fbfaf7")
    ov = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    dr = ImageDraw.Draw(im)
    do = ImageDraw.Draw(ov)

    # room fills + names
    for r in spec.rooms:
        dr.polygon([T(*p) for p in r.poly],
                   fill=_ROOM_FILL.get(r.function, "#f0efec"))
    # rugs first (walkable, under everything)
    for f in spec.furniture:
        if f.ftype != "rug":
            continue
        c, s = math.cos(math.radians(f.yaw_deg)), math.sin(math.radians(f.yaw_deg))
        pts = [(f.x + c * dx * f.hx - s * dy * f.hy,
                f.y + s * dx * f.hx + c * dy * f.hy)
               for dx, dy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
        do.polygon([T(*p) for p in pts], fill=(205, 185, 143, 90),
                   outline=(160, 140, 100, 180), width=SS)

    # furniture bodies (+ subtle facing notch on the front edge)
    for f in spec.furniture:
        if f.ftype == "rug":
            continue
        c, s = math.cos(math.radians(f.yaw_deg)), math.sin(math.radians(f.yaw_deg))
        pts = [(f.x + c * dx * f.hx - s * dy * f.hy,
                f.y + s * dx * f.hx + c * dy * f.hy)
               for dx, dy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
        col = _FCOLOR.get(f.ftype, "#888888")
        do.polygon([T(*p) for p in pts], fill=_fill_of(f.ftype),
                   outline=col, width=SS)
        if facing and f.ftype in ("bed", "sofa", "chair", "wardrobe",
                                  "bookshelf", "tv_stand", "nightstand"):
            # front edge midpoint (local +y) with a small inward triangle
            fxv, fyv = -s, c                         # mesh front direction
            mx, my = f.x + fxv * f.hy, f.y + fyv * f.hy
            tx, ty = -fyv, fxv                       # tangent along front edge
            tri = [(mx + tx * 0.07, my + ty * 0.07),
                   (mx - tx * 0.07, my - ty * 0.07),
                   (mx + fxv * 0.10, my + fyv * 0.10)]
            do.polygon([T(*p) for p in tri], fill=col)
        lab = {"nightstand": "nst", "wardrobe": "wrd", "bookshelf": "bks",
               "tv_stand": "tv", "painting": "art", "poster": "art",
               "curtain": "crt", "mirror": "mir"}.get(f.ftype, f.ftype[:4])
        if f.ftype not in ("painting", "poster", "clock", "curtain", "mirror"):
            do.text(T(f.x, f.y), lab, fill=(58, 58, 58, 230), anchor="mm")

    im.paste(Image.alpha_composite(im.convert("RGBA"), ov).convert("RGB"),
             (0, 0))
    dr = ImageDraw.Draw(im)

    # walls on top (crisp)
    for r in spec.rooms:
        dr.polygon([T(*p) for p in r.poly], outline=_WALL, width=3 * SS)

    wr = {x.id: x for x in spec.walls}

    def _run_pt(run, u):
        dx, dy = run.p2[0] - run.p1[0], run.p2[1] - run.p1[1]
        L = math.hypot(dx, dy)
        ux, uy = dx / L, dy / L
        return (run.p1[0] + ux * u, run.p1[1] + uy * u), (ux, uy)

    # doors: white gap + quarter swing arc
    for d in spec.doors:
        run = wr.get(d.wall)
        if run is None:
            continue
        (cx, cy), (ux, uy) = _run_pt(run, d.u)
        a = T(cx - ux * d.width / 2, cy - uy * d.width / 2)
        b = T(cx + ux * d.width / 2, cy + uy * d.width / 2)
        dr.line([a, b], fill="#fbfaf7", width=4 * SS)          # wall gap
        hinge = (cx - ux * d.width / 2, cy - uy * d.width / 2)
        hp = T(*hinge)
        rr = d.width * SCALE * SS
        base_ang = math.degrees(math.atan2(-(uy), ux))          # screen y down
        start, end = sorted((base_ang, base_ang + 90))
        dr.arc([hp[0] - rr, hp[1] - rr, hp[0] + rr, hp[1] + rr],
               start=start, end=end,
               fill="#8a2f2f" if d.entry else "#c0392b", width=SS)
        dr.line([hp, T(cx + ux * d.width / 2, cy + uy * d.width / 2)],
                fill="#8a2f2f" if d.entry else "#c0392b", width=SS)

    # windows: double glazing lines
    for wd in spec.windows:
        run = wr.get(wd.wall)
        if run is None:
            continue
        (cx, cy), (ux, uy) = _run_pt(run, wd.u)
        nx, ny = -uy, ux
        col = "#1f5fae" if wd.sill > 0.2 else "#0f93c9"   # cyan = floor window
        for off in (-0.045, 0.045):
            a = T(cx - ux * wd.width / 2 + nx * off,
                  cy - uy * wd.width / 2 + ny * off)
            b = T(cx + ux * wd.width / 2 + nx * off,
                  cy + uy * wd.width / 2 + ny * off)
            dr.line([a, b], fill=col, width=SS)

    # ceiling lamps
    for L in spec.lights:
        if L.fixture in ("pendant", "flush", "bulb"):
            p = T(L.x, L.y)
            rr = 4 * SS
            dr.ellipse([p[0] - rr, p[1] - rr, p[0] + rr, p[1] + rr],
                       outline="#e2a72e", width=SS)
            for k in range(4):
                a = math.pi / 4 + k * math.pi / 2
                dr.line([(p[0] + rr * math.cos(a), p[1] + rr * math.sin(a)),
                         (p[0] + (rr + 3 * SS) * math.cos(a),
                          p[1] + (rr + 3 * SS) * math.sin(a))],
                        fill="#e2a72e", width=SS)

    # room labels on top
    for r in spec.rooms:
        cx = sum(p[0] for p in r.poly) / len(r.poly)
        cy = sum(p[1] for p in r.poly) / len(r.poly)
        dr.text(T(cx, cy - 0.35), f"{r.function} {r.name[-2:]}",
                fill=_TEXT, anchor="mm")

    # legend + scale bar
    ly = h - 22 * SS
    lx = PAD * SS
    for name, col in (("door", "#c0392b"), ("window", "#1f5fae"),
                      ("lamp", "#e2a72e")):
        dr.line([(lx, ly), (lx + 16 * SS, ly)], fill=col, width=2 * SS)
        dr.text((lx + 20 * SS, ly), name, fill=_TEXT, anchor="lm")
        lx += (30 + 8 * len(name)) * SS
    bx = w - PAD * SS - SCALE * SS
    dr.line([(bx, ly), (bx + SCALE * SS, ly)], fill=_WALL, width=2 * SS)
    for e in (0, SCALE * SS):
        dr.line([(bx + e, ly - 3 * SS), (bx + e, ly + 3 * SS)],
                fill=_WALL, width=SS)
    dr.text((bx + SCALE * SS / 2, ly - 7 * SS), "1 m", fill=_TEXT, anchor="mm")

    im = im.resize((w // SS, h // SS), Image.LANCZOS)
    im.save(path)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--no-facing"]
    facing = "--no-facing" not in sys.argv
    out = Path(args[0])
    out.mkdir(parents=True, exist_ok=True)
    for s in args[1:]:
        spec = build_ge1_spec(int(s))
        plot(spec, out / f"layout_{int(s):04d}.png", facing=facing)
        print(f"seed {s}: {len(spec.furniture)} furn, "
              f"{len(spec.rooms)} rooms -> layout_{int(s):04d}.png")
