"""ZInD-style layout/floorplan/semantic export for second-generation scenes.

From a GenesisSpec (or saved manifest), dumps per scene:
  <scene>_layout.json      rooms (corner rings + function + adjacency),
                           WDO list (doors/windows with full geometry),
                           optional camera poses (--samples)
  <scene>_floorplan.png    raster floorplan (plot_layout)

The engine already carries everything (manifest = GenesisSpec); this is a
cheap projection with a schema aligned to ZInD's room-shell + W/D/O
convention (meters, z-up, floor_z=0, self-described in the json).

Usage:
  python lenscope/genesis/export_layout.py OUT_DIR seed [seed...]
      [--samples DIR]     # merge camera poses from <scene>_sample.json
      [--manifest PATH]   # export one saved manifest instead of seeds
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lenscope.genesis.spec import GenesisSpec, build_ge1_spec   # noqa: E402
from lenscope.genesis.plot_layout import plot                    # noqa: E402


def _room_of_point(spec, x, y):
    for r in spec.rooms:
        inside = False
        j = len(r.poly) - 1
        for i in range(len(r.poly)):
            xi, yi = r.poly[i]
            xj, yj = r.poly[j]
            if (yi > y) != (yj > y) and \
                    x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi:
                inside = not inside
            j = i
        if inside:
            return r.name
    return None


def export_layout(spec: GenesisSpec, out_dir: Path, samples_dir=None):
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"genesis_{spec.seed:06d}"
    wr = {w.id: w for w in spec.walls}

    def wall_frame(wid, u):
        run = wr[wid]
        dx, dy = run.p2[0] - run.p1[0], run.p2[1] - run.p1[1]
        L = math.hypot(dx, dy)
        ux, uy = dx / L, dy / L
        return (run.p1[0] + ux * u, run.p1[1] + uy * u), (-uy, ux)

    wdo = []
    adj = {r.name: set() for r in spec.rooms}
    for d in spec.doors:
        (cx, cy), (nx, ny) = wall_frame(d.wall, d.u)
        eps = wr[d.wall].t / 2 + 0.15
        ra = _room_of_point(spec, cx + nx * eps, cy + ny * eps)
        rb = _room_of_point(spec, cx - nx * eps, cy - ny * eps)
        rooms = [r for r in (ra, rb) if r] or ["exterior"]
        if ra and rb and ra != rb:
            adj[ra].add(rb)
            adj[rb].add(ra)
        wdo.append({"type": "door", "wall": d.wall,
                    "rooms": rooms if len(rooms) == 2 else rooms + ["exterior"],
                    "center_xy": [round(cx, 4), round(cy, 4)],
                    "normal_xy": [round(nx, 4), round(ny, 4)],
                    "width": round(d.width, 4), "z0": 0.0,
                    "z1": round(d.height, 4),
                    "extra": {"leaf_open_deg": round(d.leaf_open_deg, 2),
                              "entry": d.entry}})
    for w in spec.windows:
        (cx, cy), (nx, ny) = wall_frame(w.wall, w.u)
        eps = wr[w.wall].t / 2 + 0.15
        ra = _room_of_point(spec, cx + nx * eps, cy + ny * eps)
        rb = _room_of_point(spec, cx - nx * eps, cy - ny * eps)
        room = ra or rb
        wdo.append({"type": "window", "wall": w.wall,
                    "rooms": [room or "unknown", "exterior"],
                    "center_xy": [round(cx, 4), round(cy, 4)],
                    "normal_xy": [round(nx, 4), round(ny, 4)],
                    "width": round(w.width, 4), "z0": round(w.sill, 4),
                    "z1": round(w.sill + w.height, 4),
                    "extra": {"mull_v": w.mull_v, "mull_h": w.mull_h}})

    cameras = []
    if samples_dir is not None:
        sp = Path(samples_dir) / f"{name}_sample.json"
        if sp.exists():
            for p in json.loads(sp.read_text()).get("poses", []):
                cameras.append({"id": p.get("id"),
                                "pos": [round(v, 4) for v in p["pos"]],
                                "yaw_deg": round(p.get("yaw", 0.0), 3),
                                "pitch_deg": round(p.get("pitch", 0.0), 3)})

    doc = {
        "scene": name, "seed": spec.seed, "spec_version": spec.version,
        "coordinate_system": {"units": "meters", "up": "z", "floor_z": 0.0,
                              "ceiling_z": round(spec.height, 4)},
        "rooms": [{"id": r.name, "function": r.function,
                   "corners_xy": [[round(x, 4), round(y, 4)]
                                  for (x, y) in r.poly],
                   "area_m2": round(r.area, 3),
                   "adjacent": sorted(adj[r.name])} for r in spec.rooms],
        "wdo": wdo,
        "floorplan_png": f"{name}_floorplan.png",
        "cameras": cameras,
        "linked_assets_note": ("per-pixel NYU40 semantics + dense depth GT "
                               "come from the existing fixture/CAST pipeline "
                               "for the same scene name"),
    }
    (out_dir / f"{name}_layout.json").write_text(json.dumps(doc, indent=1))
    plot(spec, out_dir / f"{name}_floorplan.png")
    return doc


def main():
    argv = sys.argv[1:]
    samples = None
    if "--samples" in argv:
        i = argv.index("--samples")
        samples = argv[i + 1]
        argv = argv[:i] + argv[i + 2:]
    if "--manifest" in argv:
        i = argv.index("--manifest")
        spec = GenesisSpec.from_json(Path(argv[i + 1]).read_text())
        export_layout(spec, Path(argv[0]), samples)
        print(f"exported manifest -> {argv[0]}")
        return
    out = Path(argv[0])
    for s in argv[1:]:
        doc = export_layout(build_ge1_spec(int(s)), out, samples)
        print(f"seed {s}: {len(doc['rooms'])} rooms, {len(doc['wdo'])} wdo, "
              f"{len(doc['cameras'])} cams")


if __name__ == "__main__":
    main()
