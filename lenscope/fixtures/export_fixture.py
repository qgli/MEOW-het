#!/usr/bin/env python3
"""Fixture exporter: run inside Blender.

Dumps everything the pose-graph sampler needs from an existing .blend scene,
with no assumptions about object naming (the naming scheme is learned from the
dump itself):

  <out>/<scene>/mesh.npz          triangle soup: verts f32 [N,3] (world), faces i32 [M,3],
                                  face_obj i32 [M] (index into objects.json)
  <out>/<scene>/objects.json      per-object: name, aabb_world, n_faces, material names,
                                  custom props (semantic_id etc. if present)
  <out>/<scene>/thumb.png         top-down orthographic viewport render (visual sanity check)

Usage:
  blender -b <scene.blend> --python export_fixture.py -- --out /path/to/fixtures
Batch:
  for f in /path/to/scenes/proc_scene_*.blend; do
    blender -b "$f" --python export_fixture.py -- --out /path/to/fixtures; done

Logging: prints one JSON line per stage to stdout (grep '^AGEN ').
"""
import json
import os
import sys
from pathlib import Path

import bpy  # noqa: E402
import numpy as np


def log(event, **kw):
    print("AGEN " + json.dumps({"event": event, **kw}), flush=True)


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    out_root = Path(argv[argv.index("--out") + 1]) if "--out" in argv else Path("./fixtures")
    scene_name = Path(bpy.data.filepath).stem or "unsaved"
    out = out_root / scene_name
    out.mkdir(parents=True, exist_ok=True)
    log("start", scene=scene_name, blender=bpy.app.version_string)

    def _mat_flags(obj):
        """Detect glass/mirror/emissive from material node trees (without
        them the ambiguity flags stay empty). Plain Principled defaults =>
        no flag. Blender 4.x input names
        ('Transmission Weight', 'Emission Color') with 3.x fallbacks."""
        flags = set()
        for slot in (obj.data.materials or []):
            if not slot or not getattr(slot, "use_nodes", False):
                continue
            for node in slot.node_tree.nodes:
                t = node.type
                if t in ("BSDF_GLASS", "BSDF_REFRACTION"):
                    flags.add("glass")
                elif t == "EMISSION":
                    flags.add("emissive")
                elif t == "BSDF_PRINCIPLED":
                    def _v(name, d=0.0):
                        i = node.inputs.get(name)
                        if i is None or i.is_linked or not hasattr(i, "default_value"):
                            return d
                        dv = i.default_value
                        return float(dv) if isinstance(dv, (int, float)) else d
                    if _v("Transmission Weight", _v("Transmission")) > 0.5:
                        flags.add("glass")
                    if _v("Metallic") > 0.7 and _v("Roughness", 1.0) < 0.25:
                        flags.add("mirror")
                    if _v("Emission Strength") > 0.0:
                        ec = node.inputs.get("Emission Color") or node.inputs.get("Emission")
                        if ec is not None and not ec.is_linked and max(ec.default_value[:3]) > 0.01:
                            flags.add("emissive")
        return sorted(flags)

    # Iterate the depsgraph's object_instances (== exactly what Cycles renders):
    # this captures (a) real objects and (b) generated instances (collection-
    # instance / geometry-nodes / particle chairs etc.) that never appear in
    # scene.objects, and it uses each instance's own matrix_world. hide_viewport
    # is synced from hide_render first, because the viewport depsgraph would drop
    # render-visible-but-viewport-hidden objects (e.g. FloatDistractor). Rule:
    # the ground-truth geometry set must equal the render set; a
    # `scene.objects + visible_get()` loop silently drops both classes and
    # leaves depth holes on chairs/distractors.
    _saved_hv = {}
    for _o in bpy.data.objects:
        _saved_hv[_o.name] = _o.hide_viewport
        if _o.type == "MESH":
            _o.hide_viewport = _o.hide_render
    bpy.context.view_layer.update()
    deps = bpy.context.evaluated_depsgraph_get()

    verts_all, faces_all, face_obj = [], [], []
    objects = []
    by_name = {}          # original object name -> index in `objects` (instances merged)
    v_off = 0
    for inst in deps.object_instances:
        ob = inst.object
        if ob is None or ob.type != "MESH":
            continue
        try:
            me = ob.to_mesh()
        except RuntimeError:
            continue
        if me is None or len(me.polygons) == 0:
            ob.to_mesh_clear()
            continue
        me.calc_loop_triangles()
        mw = np.array(inst.matrix_world, dtype=np.float64)   # instance transform (not ob.matrix_world)
        v = np.empty(len(me.vertices) * 3, np.float32)
        me.vertices.foreach_get("co", v)
        v = v.reshape(-1, 3)
        vw = (v @ mw[:3, :3].T + mw[:3, 3]).astype(np.float32)
        tris = np.empty(len(me.loop_triangles) * 3, np.int32)
        me.loop_triangles.foreach_get("vertices", tris)
        tris = tris.reshape(-1, 3) + v_off
        orig = ob.original                                   # metadata from the source object
        key = orig.name
        if key not in by_name:
            by_name[key] = len(objects)
            objects.append({
                "name": key,
                "aabb_world": [vw.min(0).tolist(), vw.max(0).tolist()],
                "n_faces": 0,
                "n_instances": 0,
                "materials": [m.name for m in orig.data.materials if m] if orig.data.materials else [],
                "material_flags": _mat_flags(orig),
                "props": {k: (orig[k] if isinstance(orig[k], (int, float, str)) else str(orig[k]))
                          for k in orig.keys() if not k.startswith("_")},
            })
        oi = by_name[key]
        rec = objects[oi]
        rec["n_faces"] += int(len(tris))
        rec["n_instances"] += 1
        rec["aabb_world"] = [np.minimum(rec["aabb_world"][0], vw.min(0)).tolist(),
                             np.maximum(rec["aabb_world"][1], vw.max(0)).tolist()]
        verts_all.append(vw)
        faces_all.append(tris)
        face_obj.append(np.full(len(tris), oi, np.int32))
        v_off += len(vw)
        ob.to_mesh_clear()

    for _o in bpy.data.objects:                              # restore viewport visibility
        if _o.name in _saved_hv:
            _o.hide_viewport = _saved_hv[_o.name]
    bpy.context.view_layer.update()

    V = np.concatenate(verts_all) if verts_all else np.zeros((0, 3), np.float32)
    F = np.concatenate(faces_all) if faces_all else np.zeros((0, 3), np.int32)
    FO = np.concatenate(face_obj) if face_obj else np.zeros(0, np.int32)
    np.savez_compressed(out / "mesh.npz", verts=V, faces=F, face_obj=FO)
    json.dump(objects, open(out / "objects.json", "w"), ensure_ascii=False, indent=1)
    log("mesh_dumped", n_verts=int(len(V)), n_faces=int(len(F)), n_objects=len(objects),
        bytes=(out / "mesh.npz").stat().st_size)

    # top-down thumb (workbench engine, fast, no samples dependency).
    # AGEN_NO_THUMB=1 skips it: on headless machines without a display the
    # workbench GL context aborts the process (epoxy EGL assert) and can race
    # the npz dump. By default the thumb is rendered.
    if os.environ.get("AGEN_NO_THUMB", "0") == "1":
        log("thumb_skipped", reason="AGEN_NO_THUMB=1")
        return
    try:
        scene = bpy.context.scene
        # hide ceiling/roof so the top-down view shows room layout, not just the ceiling
        # (mesh.npz above already captured full geometry incl. ceiling — this only affects thumb)
        for o in scene.objects:
            if o.type == "MESH" and o.name.lower().startswith(("ceiling", "roof")):
                o.hide_render = True
        cam = bpy.data.objects.new("agen_topcam", bpy.data.cameras.new("agen_topcam"))
        scene.collection.objects.link(cam)
        cam.data.type = "ORTHO"
        lo, hi = V.min(0), V.max(0)
        cam.data.ortho_scale = float(max(hi[0] - lo[0], hi[1] - lo[1]) * 1.1 + 1e-3)
        cam.location = ((lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, hi[2] + 5)
        cam.rotation_euler = (0, 0, 0)
        scene.camera = cam
        scene.render.engine = "BLENDER_WORKBENCH"
        scene.render.resolution_x = scene.render.resolution_y = 768
        scene.render.filepath = str(out / "thumb.png")
        bpy.ops.render.render(write_still=True)
        log("thumb_rendered", path=str(out / "thumb.png"))
    except Exception as e:  # thumb is best-effort
        log("thumb_failed", err=str(e)[:200])
    log("done", out=str(out))


if __name__ == "__main__":
    main()
