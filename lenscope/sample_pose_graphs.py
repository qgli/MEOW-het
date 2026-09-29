#!/usr/bin/env python3
"""Run the pose-graph sampler on fixture dumps of Blender scenes.

Per scene: load_fixture (TriSoup from mesh.npz) -> spec_from_fixture_v2 (room split;
single-room spec_from_fixture as fallback) -> sample_scene (real-mesh covisibility
edges + triangulation-angle DOP scores + guarantees) -> pose-graph HTML
visualization + sample.json.

Usage: sample_pose_graphs.py <fixtures_root> <out_dir> [--seed 0] [--only SCENE]
"""
import argparse
import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from lenscope.core.mesh import load_fixture
from lenscope.core.sampler import SamplerFailure, sample_scene
from lenscope.core.spec import spec_from_fixture, spec_from_fixture_v2
from lenscope.viz.pose_graph_html import write_html


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root"); ap.add_argument("out"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", default=None, help="sample only this scene dir name (production per-scene loop)")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    summary = {}
    for scene_dir in sorted(Path(a.root).iterdir()):
        if not (scene_dir / "mesh.npz").exists():
            continue
        name = scene_dir.name
        if a.only is not None and name != a.only:
            continue
        try:
            soup = load_fixture(scene_dir / "mesh.npz", scene_dir / "objects.json")
            # room split + portals first; per-scene fallback to the single-room
            # spec when the split is structurally infeasible (e.g. an
            # over-segmented pocket is visibility-isolated -> components not
            # connectable).
            specs = [("v2", spec_from_fixture_v2(scene_dir / "objects.json", name, seed=a.seed)),
                     ("v1", spec_from_fixture(scene_dir / "objects.json", name, seed=a.seed))]
            res, last, spec_used = None, None, None
            # multi-seed retry: the single-room spec has no portal bridges, so a
            # greedy seed trapped in a wall-separated pocket fails the whole scene
            # (which motivates the portal-first design).
            for tag, spec in specs:
                for sd in range(a.seed, a.seed + 6):
                    try:
                        res = sample_scene(spec, soup, seed=sd, use_embree=True)
                        spec_used = tag
                        break
                    except SamplerFailure as e:
                        last = e
                if res is not None:
                    break
            if res is None:
                raise last
            res["report"]["spec_stage"] = spec_used
            res["report"]["n_rooms"] = len(spec.rooms)
            res["report"]["n_portals"] = len(spec.portals)
            (out / f"{name}_sample.json").write_text(json.dumps(res, indent=1))
            write_html(spec, res, out / f"{name}_graph.html")
            summary[name] = {"ok": True, **res["report"]}
            print(f"[ok] {name}: {res['report']['n_poses']} poses, "
                  f"min_deg {res['report']['min_degree']}, "
                  f"tri_sin p10/50/90 {res['report']['tri_sin_p10_50_90']}")
        except SamplerFailure as e:
            summary[name] = {"ok": False, "reason": str(e)}
            print(f"[FAIL] {name}: {e}")
        except Exception as e:
            summary[name] = {"ok": False, "reason": f"{type(e).__name__}: {e}"}
            print(f"[ERR] {name}: {e}")
            traceback.print_exc()
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print("summary ->", out / "summary.json")


if __name__ == "__main__":
    main()
