#!/usr/bin/env python3
"""
Top-level pipeline of the first-generation procedural scene generator.

Entry point of scene generation: assembles all modules into the full pipeline.

  - generate_floor_plan returns a 6-tuple: (floor_verts, internal_walls, type, bbox, sub_rooms, portals)
  - Light energy compensates for ceiling height only: energy = rng.uniform(30, 80) * room_h**2 (no area factor)
  - Doorways near wall ends: for length >= 3.0, mid = rng.choice([1.0, length - 1.0])
  - Portal data (doorway center + normal) is passed on to camera placement
  - assess_frustum_quality: rejects wall-facing views from the variance and mean of 9 ray distances
  - Two-track target-to-camera (T2C) placement:
    A) portal shot: with 40% probability, a deep view through a doorway into the next room
    B) intra-room shot: aimed at local points of interest, with a 3D spiral nudge

  generate_scene(seed, output_dir, room_type):
    0. AggressiveMemoryManager.clear_scene()
    1. generate_floor_plan -> RoomBuilder.build_all()
    2. ConstraintLayoutSolver.init_from_room_shell()
    3. RoomDecorator.populate_furniture()
    4. WallDecorator.decorate_all()
    5. setup_lighting_and_world(sub_rooms): lights in every sub-room
    6. add_cameras(): 24 mm pinhole cameras aimed at points of interest by category
    7. estimate_auto_exposure
    7.5 add_floating_distractors
    8. bpy.ops.wm.save_as_mainfile -> returns the .blend path

Rules:
  - No bpy.ops (except for saving), no global random state, no destructive scaling
  - Rendering is done by the downstream renderer; this module does not render
"""

from __future__ import annotations

import math
import os
import random
import sys
import time
from typing import Optional, Sequence

import bpy
from mathutils import Vector

from .core_utils import (
    AggressiveMemoryManager,
    guaranteed_interior_point,
    polygon_centroid,
    polygon_signed_area,
    random_interior_point,
)
from .constraint_solver import ConstraintLayoutSolver
from .geometry_factory import BMeshFactory, ProceduralMaterialFactory
from .layout_orchestrator import RoomDecorator
from .room_builder import ROOM_TYPES, RoomBuilder, generate_floor_plan
from .wall_decorator import WallDecorator


# ---- Lighting and world ----

def setup_lighting_and_world(
    rng: random.Random,
    floor_verts: Sequence[tuple[float, float]],
    room_h: float,
    collection: Optional[bpy.types.Collection] = None,
    sub_rooms: Optional[list[list[tuple[float, float]]]] = None,
    internal_walls: Optional[list] = None,
) -> dict:
    """
    Lights for every sub-room plus a Nishita sky-texture world.

    Every sub-room gets at least one light; sub-rooms larger than 15 m^2 get
    int(sub_area / 15.0) lights. Each light is placed inside its own sub-room
    polygon, so no sub-room stays dark while another one is lit.

    Parameters
    ----------
    rng : random.Random
    floor_verts : Sequence[(float, float)]
        Outer room polygon (only used as a fallback when there are no sub-rooms)
    room_h : float
    collection : optional Collection
    sub_rooms : optional list[list[(float, float)]]
        Sub-room polygons from the BSP split
    internal_walls : optional list
        Internal wall segments, passed to random_interior_point to keep lights away from them

    Returns
    -------
    dict
        {
            'world': bpy.types.World,
            'lights': list[bpy.types.Object],
            'total_energy': float,
        }
    """
    from .core_utils import create_data_object

    # World nodes: Nishita sky texture
    world = bpy.data.worlds.new("ProceduralWorld")
    world.use_nodes = True
    nt = world.node_tree
    # Remove the default nodes
    for node in nt.nodes:
        nt.nodes.remove(node)

    # Sky Texture
    sky_tex = nt.nodes.new('ShaderNodeTexSky')
    sky_tex.sky_type = 'NISHITA'
    sky_tex.sun_elevation = math.radians(rng.uniform(15, 65))
    sky_tex.sun_rotation = math.radians(rng.uniform(0, 360))
    sky_tex.sun_intensity = rng.uniform(0.5, 1.5)
    sky_tex.altitude = rng.uniform(0, 100)
    sky_tex.air_density = rng.uniform(0.8, 1.2)
    sky_tex.dust_density = rng.uniform(0.5, 1.5)
    sky_tex.ozone_density = rng.uniform(0.8, 1.2)
    sky_tex.location = (-300, 200)

    # Background
    bg_node = nt.nodes.new('ShaderNodeBackground')
    bg_node.inputs['Strength'].default_value = rng.uniform(0.3, 1.0)
    bg_node.location = (0, 200)

    # World Output
    output = nt.nodes.new('ShaderNodeOutputWorld')
    output.location = (300, 200)

    nt.links.new(sky_tex.outputs['Color'], bg_node.inputs['Color'])
    nt.links.new(bg_node.outputs['Background'], output.inputs['Surface'])

    bpy.context.scene.world = world

    # Lights for every sub-room
    _polys = sub_rooms if sub_rooms else [list(floor_verts)]
    _iw = internal_walls or []

    lights: list[bpy.types.Object] = []
    total_energy = 0.0
    light_idx = 0

    for room_i, sub_poly in enumerate(_polys):
        sub_area = abs(polygon_signed_area(sub_poly))
        # At least one light per sub-room; larger rooms get one per 15 m^2
        n_lights_here = max(1, int(sub_area / 15.0))

        for j in range(n_lights_here):
            light_type = rng.choice(['AREA', 'POINT', 'SPOT'])
            # Compensate for ceiling height only; an extra area factor would make the energy grow quadratically
            energy = rng.uniform(30, 80) * (room_h ** 2)
            total_energy += energy

            name = f"CeilingLight_R{room_i:02d}_{light_idx:02d}"
            light_data = bpy.data.lights.new(name, type=light_type)
            light_data.energy = energy
            light_data.color = (
                rng.uniform(0.90, 1.0),
                rng.uniform(0.85, 1.0),
                rng.uniform(0.75, 1.0),
            )

            # Type-specific light settings
            if light_type == 'AREA':
                light_data.shape = 'RECTANGLE'
                light_data.size = rng.uniform(0.3, 1.0)
                light_data.size_y = rng.uniform(0.3, 1.0)
            elif light_type == 'SPOT':
                light_data.spot_size = math.radians(rng.uniform(40, 90))
                light_data.spot_blend = rng.uniform(0.1, 0.5)
            elif light_type == 'POINT':
                light_data.shadow_soft_size = rng.uniform(0.1, 0.4)

            # Place the light inside the sub-room polygon
            px, py = random_interior_point(
                rng, sub_poly, margin=0.5, internal_walls=_iw)
            pz = room_h - rng.uniform(0.01, 0.15)

            light_obj = create_data_object(name, light_data, collection)
            light_obj.location = (px, py, pz)

            # SPOT / AREA lights point down
            if light_type in ('SPOT', 'AREA'):
                light_obj.rotation_euler = (
                    math.pi, 0, rng.uniform(0, 2 * math.pi))

            lights.append(light_obj)
            light_idx += 1

    return {
        'world': world,
        'lights': lights,
        'total_energy': total_energy,
    }


# ---- Cameras: frustum ray-grid check + two-track T2C placement ----


def assess_frustum_quality(
    scene: bpy.types.Scene,
    depsgraph,
    origin: Vector,
    target: Vector,
    fov_deg: float = 73.7,
) -> bool:
    """
    Check a view with a sparse 3x3 ray grid over the frustum, with a veto for close occluders.
    """
    import numpy as np

    forward = (target - origin).normalized()
    right = forward.cross(Vector((0, 0, 1)))
    if right.length < 1e-5:
        right = Vector((1, 0, 0))
    else:
        right = right.normalized()
    up = right.cross(forward).normalized()

    dists: list[float] = []
    tan_half = math.tan(math.radians(fov_deg / 2))
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            ray_dir = (forward + right * (dx * tan_half)
                       + up * (dy * tan_half)).normalized()
            hit, loc, _, _, _, _ = scene.ray_cast(depsgraph, origin, ray_dir)
            if hit:
                dists.append((loc - origin).length)
            else:
                dists.append(10.0)  # misses count as 10 m, which keeps the variance bounded

    # Veto: if 4 of the 9 rays (almost half) hit within 1.0 m, the view is heavily occluded
    if sum(1 for d in dists if d < 1.0) >= 4:
        return False

    variance = float(np.var(dists))
    mean_dist = float(np.mean(dists))
    
    # Very low variance and a very short mean distance: the camera faces a flat wall
    return not (variance < 0.5 and mean_dist < 1.5)

def add_cameras(
    rng: random.Random,
    floor_verts: Sequence[tuple[float, float]],
    room_h: float,
    n_cameras: int = 5,
    collection: Optional[bpy.types.Collection] = None,
    room_shell: Optional[bpy.types.Object] = None,
    sub_rooms: Optional[list] = None,
    internal_walls: Optional[list] = None,
    poi_objects: Optional[list] = None,
    portals: Optional[list] = None,
) -> list[bpy.types.Object]:
    """
    Two-track target-to-camera (T2C) placement: choose a target, then a camera for it.

    Track A, portal shot (deep view through a doorway):
      Triggered with 40% probability (requires a non-empty portals list).
      The camera backs off against the doorway normal and the target lies ahead
      along it, which gives a deep view across rooms.

    Track B, intra-room shot (details of points of interest in the same room):
      The camera position is sampled in a sub-room and the target is taken from
      that room's local POIs. 3D spiral nudge: raise Z + Archimedean spiral in XY,
      at most 30 attempts.

    Height distribution: mixture (60% eye level / 20% near the ceiling, like a surveillance camera / 15% pet height / 5% uniform)
    Wall check: assess_frustum_quality (variance and mean of 9 ray distances)

    Parameters
    ----------
    rng : random.Random
    floor_verts : outer polygon vertices
    room_h : ceiling height
    n_cameras : number of cameras
    collection : Blender collection
    room_shell : RoomShell object (for ray casting)
    sub_rooms : sub-room polygons
    internal_walls : internal wall segments
    poi_objects : point-of-interest (POI) objects
    portals : doorway data [{'center': (cx,cy), 'normal': (nx,ny)}]
    """
    from .core_utils import (create_data_object, point_in_polygon,
                              distance_to_polygon_boundary)

    cameras: list[bpy.types.Object] = []
    cx_room, cy_room = polygon_centroid(floor_verts)
    _sub = sub_rooms if sub_rooms else [list(floor_verts)]
    _iw = internal_walls or []
    _portals = portals or []

    # Categorize the points of interest
    windows_pois: list[Vector] = []
    paintings_pois: list[Vector] = []
    mounted_pois: list[Vector] = []
    other_pois: list[Vector] = []

    if poi_objects:
        for obj in poi_objects:
            if not hasattr(obj, 'location'):
                continue
            pos = Vector(obj.location)
            name_lower = obj.name.lower()
            if 'window' in name_lower:
                windows_pois.append(pos)
            elif 'paint' in name_lower:
                paintings_pois.append(pos)
            elif any(kw in name_lower for kw in ('wall', 'sconce', 'clock', 'mounted')):
                mounted_pois.append(pos)
            else:
                other_pois.append(pos)

    all_poi_positions = windows_pois + paintings_pois + mounted_pois + other_pois

    if not all_poi_positions:
        for _ in range(20):
            rpoly = rng.choice(_sub)
            rpx, rpy = random_interior_point(rng, rpoly, margin=0.5,
                                             internal_walls=_iw)
            all_poi_positions.append(Vector((rpx, rpy, room_h * rng.uniform(0.2, 0.6))))

    dg = bpy.context.evaluated_depsgraph_get()
    scene = bpy.context.scene

    def _pick_target(cam_index: int) -> Vector:
        """Pick the target of camera cam_index by POI category."""
        if cam_index == 0 and windows_pois:
            return rng.choice(windows_pois)
        if cam_index == 1 and paintings_pois:
            return rng.choice(paintings_pois)
        if cam_index == 2 and mounted_pois:
            return rng.choice(mounted_pois)
        pool = other_pois if other_pois else all_poi_positions
        return rng.choice(pool)

    def _sample_height() -> float:
        """Sample a camera height from the mixture distribution."""
        roll = rng.random()
        if roll < 0.60:
            return max(1.2, min(1.8, rng.gauss(1.5, 0.15)))
        elif roll < 0.80:
            return max(room_h - 0.4, min(room_h - 0.2, rng.gauss(room_h - 0.3, 0.05)))
        elif roll < 0.95:
            return max(0.15, min(0.6, rng.gauss(0.375, 0.1)))
        else:
            return rng.uniform(0.05, room_h - 0.05)

    # Precompute the local POIs of each sub-room (0.2 m tolerance)
    room_local_pois: list[list[Vector]] = [[] for _ in _sub]
    for poi in all_poi_positions:
        for ri, sp in enumerate(_sub):
            if (point_in_polygon(poi.x, poi.y, sp)
                    or distance_to_polygon_boundary(poi.x, poi.y, sp) < 0.2):
                room_local_pois[ri].append(poi)
                break

    # Cross-room targets: add the area centroids of the other sub-rooms to each sub-room
    if len(_sub) > 1:
        _sub_centroids = [polygon_centroid(sp) for sp in _sub]
        for ri in range(len(_sub)):
            for rj, (scx, scy) in enumerate(_sub_centroids):
                if rj != ri:
                    room_local_pois[ri].append(Vector((scx, scy, 1.2)))

    for i in range(n_cameras):
        cam_data = bpy.data.cameras.new(f"Camera_{i:02d}")
        cam_data.type = 'PERSP'
        cam_data.lens = 24
        cam_data.clip_start = 0.05
        cam_data.clip_end = 100.0
        cam_data.sensor_width = 36

        cam_obj = create_data_object(
            f"Camera_{i:02d}", cam_data, collection)

        placed = False

        # Track A: portal shot, a deep view through a doorway
        if _portals and rng.random() < 0.4:
            portal = rng.choice(_portals)
            pcx, pcy = portal['center']
            pnx, pny = portal['normal']
            # Height limited to doorway range: no near-ceiling views that would hit the door frame
            pz = rng.uniform(1.2, 1.7)

            # Target: 2-5 m ahead along the normal
            t_dist = rng.uniform(2.0, 5.0)
            tx = pcx + pnx * t_dist
            ty = pcy + pny * t_dist
            T = Vector((tx, ty, pz))

            # Camera: 2-5 m back against the normal
            c_dist = rng.uniform(2.0, 5.0)
            cam_x = pcx - pnx * c_dist
            cam_y = pcy - pny * c_dist
            C = Vector((cam_x, cam_y, pz))

            # Check: C and T both inside floor_verts, and the frustum check passes
            if (point_in_polygon(C.x, C.y, floor_verts) and
                    point_in_polygon(T.x, T.y, floor_verts) and
                    assess_frustum_quality(scene, dg, C, T)):
                cam_obj.location = C
                direction = (T - C).normalized()
                rot_quat = direction.to_track_quat('-Z', 'Y')
                cam_obj.rotation_euler = rot_quat.to_euler()
                placed = True

        # Track B: intra-room shot aimed at the POIs of the same room
        if not placed:
            best_origin = None
            best_target = None

            for attempt in range(30):
                # Sample a sub-room
                room_idx = rng.randint(0, len(_sub) - 1)
                rpoly = _sub[room_idx]
                px, py = random_interior_point(
                    rng, rpoly, margin=rng.uniform(0.5, 1.5),
                    internal_walls=_iw)
                pz = _sample_height()

                # Candidate camera position (nudged with a 3D spiral below if the check fails)
                candidate_origin = Vector((px, py, pz))

                # Target: drawn by category from this room's local POIs only, never from all POIs
                local = room_local_pois[room_idx]
                local_wins = [p for p in local if p in windows_pois]
                local_paints = [p for p in local if p in paintings_pois]
                local_mounts = [p for p in local if p in mounted_pois]
                if i == 0 and local_wins:
                    target = rng.choice(local_wins)
                elif i == 1 and local_paints:
                    target = rng.choice(local_paints)
                elif i == 2 and local_mounts:
                    target = rng.choice(local_mounts)
                elif local:
                    target = rng.choice(local)
                else:
                    # No POI in this room: use the room centroid
                    _scx, _scy = polygon_centroid(rpoly)
                    target = Vector((_scx, _scy, 1.2))

                direction = (target - candidate_origin)
                if direction.length < 0.1:
                    continue

                # Frustum quality check
                if assess_frustum_quality(scene, dg, candidate_origin, target):
                    best_origin = candidate_origin
                    best_target = target
                    placed = True
                    break
                else:
                    # Spiral nudge: raise Z and move along an Archimedean spiral in XY
                    nudge_ok = False
                    for spiral_k in range(1, 4):
                        nudged = Vector((
                            candidate_origin.x + 0.3 * spiral_k * math.cos(spiral_k * 2.3),
                            candidate_origin.y + 0.3 * spiral_k * math.sin(spiral_k * 2.3),
                            min(room_h - 0.2, candidate_origin.z + 0.5 * spiral_k),
                        ))
                        if (point_in_polygon(nudged.x, nudged.y, floor_verts) and
                                assess_frustum_quality(scene, dg, nudged, target)):
                            best_origin = nudged
                            best_target = target
                            nudge_ok = True
                            break
                    if nudge_ok:
                        placed = True
                        break
                    # Keep the last attempt as a fallback
                    best_origin = candidate_origin
                    best_target = target

            if best_origin is not None:
                cam_obj.location = best_origin
                # Avoid a degenerate view direction: use best_target, else the sub-room centroid
                sub_cx, sub_cy = polygon_centroid(rpoly)
                _fb_target = (best_target if best_target is not None
                              else Vector((sub_cx, sub_cy, 1.2)))
                direction = (_fb_target - best_origin)
                if direction.length < 0.5:
                    direction = Vector((1.0, 0.0, 0.0))
                else:
                    direction = direction.normalized()
                rot_quat = direction.to_track_quat('-Z', 'Y')
                cam_obj.rotation_euler = rot_quat.to_euler()
                placed = True

        # Last resort: random point of a random sub-room at eye height, looking at its centroid
        if not placed:
            rpoly_fb = rng.choice(_sub)
            sub_cx, sub_cy = polygon_centroid(rpoly_fb)
            px_fb, py_fb = random_interior_point(
                rng, rpoly_fb, margin=0.5, internal_walls=_iw)
            origin = Vector((px_fb, py_fb, 1.5))
            target_fb = Vector((sub_cx, sub_cy, 1.2))
            cam_obj.location = origin
            direction = (target_fb - origin)
            if direction.length < 0.5:
                direction = Vector((1.0, 0.0, 0.0))
            else:
                direction = direction.normalized()
            rot_quat = direction.to_track_quat('-Z', 'Y')
            cam_obj.rotation_euler = rot_quat.to_euler()

        cameras.append(cam_obj)

    if cameras:
        bpy.context.scene.camera = cameras[0]

    return cameras


# ---- Auto exposure ----

def estimate_auto_exposure(
    area: float,
    room_h: float,
    total_energy: float,
) -> float:
    """
    Closed-form auto-exposure estimate.

    Formula: film_exposure = 1.0 / max(0.1, total_energy / (area * room_h)) * 5.0

    Parameters
    ----------
    area : float
        Floor area (m^2)
    room_h : float
        Ceiling height (m)
    total_energy : float
        Sum of the energies of all lights in the scene

    Returns
    -------
    float
        Value for scene.cycles.film_exposure
    """
    intensity_density = total_energy / max(0.01, area * room_h)
    exposure = 1.0 / max(0.1, intensity_density) * 5.0
    # Clamp to a reasonable range
    return max(0.1, min(exposure, 20.0))


# ---- Floating distractors ----

def add_floating_distractors(
    rng: random.Random,
    floor_verts: Sequence[tuple[float, float]],
    internal_walls: list,
    room_area: float,
    collection: Optional[bpy.types.Collection] = None,
    sub_rooms: Optional[list] = None,
) -> list[bpy.types.Object]:
    """
    Add floating distractor objects.

    Properties:
      - hide_viewport=True: invisible to ray_cast, so they do not affect camera placement
      - Not added to the BVH: they do not affect furniture placement
      - Z: 1.8-2.8 m (floating in the air)
      - Rendered (hide_render=False)
      - n_clusters = max(2, int(room_area / 15.0))

    Parameters
    ----------
    rng : random.Random
    floor_verts : list[(float, float)]
    internal_walls : list
    room_area : float
    collection : optional Collection

    Returns
    -------
    list[bpy.types.Object]
    """
    from .core_utils import create_data_object

    n_clusters = max(2, int(room_area / 15.0))
    distractors: list[bpy.types.Object] = []
    _sub = sub_rooms if sub_rooms else [list(floor_verts)]

    for cl in range(n_clusters):
        rpoly = rng.choice(_sub)
        px, py = random_interior_point(
            rng, rpoly, margin=0.5,
            internal_walls=internal_walls)
        pz = rng.uniform(1.8, 2.8)

        # Random shape: sphere / cone / torus / icosphere
        shape = rng.choice(['sphere', 'cone', 'torus', 'icosphere'])
        if shape == 'sphere':
            r = rng.uniform(0.03, 0.10)
            mesh = BMeshFactory.create_sphere(
                f"FloatDistractor_{cl:02d}",
                radius=r, u_segments=10, v_segments=6)
        elif shape == 'cone':
            r = rng.uniform(0.03, 0.08)
            h = rng.uniform(0.05, 0.15)
            mesh = BMeshFactory.create_cone(
                f"FloatDistractor_{cl:02d}",
                radius_bottom=r, depth=h, segments=8)
        elif shape == 'torus':
            mesh = BMeshFactory.create_torus(
                f"FloatDistractor_{cl:02d}",
                major_radius=rng.uniform(0.04, 0.12),
                minor_radius=rng.uniform(0.01, 0.04),
                major_segments=16, minor_segments=8)
        else:  # icosphere
            mesh = BMeshFactory.create_icosphere(
                f"FloatDistractor_{cl:02d}",
                radius=rng.uniform(0.03, 0.10),
                subdivisions=rng.randint(1, 3))

        mat = ProceduralMaterialFactory.random_material(
            f"DistractorMat_{cl:02d}", rng)

        obj = create_data_object(
            f"FloatDistractor_{cl:02d}", mesh, collection)
        obj.data.materials.append(mat)
        obj.location = (px, py, pz)
        obj.rotation_euler = (
            rng.uniform(0, math.pi),
            rng.uniform(0, math.pi),
            rng.uniform(0, 2 * math.pi),
        )

        # hide_viewport=True makes ray_cast skip the object; hide_render=False keeps it in renders
        obj.hide_viewport = True
        obj.hide_render = False

        distractors.append(obj)

    return distractors


# ---- Full scene generation pipeline ----

def generate_scene(
    seed: int,
    output_dir: str,
    room_type: str = 'random',
    n_cameras: int = 5,
    n_paintings: int = 2,
    n_windows: int = 2,
    n_mounted: int = 3,
) -> str:
    """
    End-to-end generation of one scene (writes a .blend file; no rendering).

    Pipeline:
      0. clear_scene
      1. generate_floor_plan -> RoomBuilder.build_all
      2. ConstraintLayoutSolver.init_from_room_shell
      3. RoomDecorator.populate_furniture (area-based density)
      4. WallDecorator.decorate_all (windows on external walls only)
      5. setup_lighting_and_world
      6. add_cameras: 24 mm pinhole cameras aimed at points of interest by category
      7. auto exposure
      7.5 floating distractors
      8. save the .blend file and return its path

    Parameters
    ----------
    seed : int
    output_dir : str
    room_type : str
    n_cameras : int
        unused; the camera count follows the floor area (see step 6)
    n_paintings : int
    n_windows : int
    n_mounted : int

    Returns
    -------
    str
        Path of the saved .blend file
    """
    t0 = time.time()
    rng = random.Random(seed)

    # Step 0: clear the scene
    mem = AggressiveMemoryManager()
    mem.clear_scene()

    # Step 1: floor plan and room shell (6-tuple including portals)
    floor_verts, internal_walls, actual_type, bbox, sub_rooms, portals = generate_floor_plan(
        rng, room_type)
    # Ceiling height 2.8-5.0 m (plausible residential range)
    room_h = rng.uniform(2.8, 5.0)

    room_builder = RoomBuilder(
        rng, floor_verts, room_h,
        collection_name=f"Room_{seed:06d}",
        internal_walls=internal_walls,
        sub_rooms=sub_rooms,
    )
    room_data = room_builder.build_all()
    room_col = room_data['collection']

    # Step 2: initialize the constraint solver
    solver = ConstraintLayoutSolver()
    solver.init_from_room_shell(room_data['room_shell'])

    # Step 3: furniture layout
    decorator = RoomDecorator(
        rng, room_builder, solver,
        furniture_collection=room_col,
    )
    furniture_result = decorator.populate_furniture(rng)

    # Step 4: wall decoration
    wall_dec = WallDecorator(
        wall_info=room_data['wall_info'],
        floor_verts=floor_verts,
        ceiling_height=room_h,
        collection=room_col,
    )
    wall_result = wall_dec.decorate_all(
        rng,
        n_paintings=n_paintings,
        n_windows=n_windows,
        n_mounted=n_mounted,
    )

    # Energy of the window lights
    wall_light_energy = sum(
        obj.data.energy
        for obj in wall_dec.window_lights
        if hasattr(obj.data, 'energy')
    )
    # Energy of the wall lamps
    for obj in wall_dec.mounted_objects:
        if obj.type == 'LIGHT' and hasattr(obj.data, 'energy'):
            wall_light_energy += obj.data.energy

    # Step 5: lighting and world (lights in every sub-room)
    lighting = setup_lighting_and_world(
        rng, floor_verts, room_h,
        collection=room_col,
        sub_rooms=sub_rooms,
        internal_walls=internal_walls,
    )
    total_energy = lighting['total_energy'] + wall_light_energy

    # Step 6: cameras (count from area, sub-rooms, aimed at POIs by category)
    area = abs(polygon_signed_area(floor_verts))
    actual_n_cameras = max(5, int(area / 15.0) * 5)
    # POIs: furniture, paintings, windows and wall-mounted objects
    poi_objects = furniture_result.get('placed', [])
    poi_objects.extend(wall_result.get('paintings', []))
    poi_objects.extend(wall_result.get('windows', []))
    poi_objects.extend(wall_result.get('mounted_objects', []))
    cameras = add_cameras(
        rng, floor_verts, room_h,
        n_cameras=actual_n_cameras,
        collection=room_col,
        room_shell=room_data['room_shell'],
        sub_rooms=sub_rooms,
        internal_walls=internal_walls,
        poi_objects=poi_objects,
        portals=portals,
    )

    # Step 7: auto exposure
    exposure = estimate_auto_exposure(area, room_h, total_energy)
    bpy.context.scene.cycles.film_exposure = exposure

    # Step 7.5: floating distractors
    distractors = add_floating_distractors(
        rng, floor_verts, internal_walls, area, room_col,
        sub_rooms=sub_rooms)

    # Render settings
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.samples = 128
    scene.cycles.use_denoising = True
    scene.render.resolution_x = 1920
    scene.render.resolution_y = 1080

    # Step 8: save the .blend file (rendering happens downstream)
    os.makedirs(output_dir, exist_ok=True)

    filename = f"proc_scene_{seed:06d}.blend"
    filepath = os.path.join(output_dir, filename)
    bpy.ops.wm.save_as_mainfile(filepath=filepath)

    dt = time.time() - t0
    print(f"[generate_scene] seed={seed} type={actual_type} "
          f"saved: {filepath} ({dt:.1f}s)")
    print(f"  area={area:.1f}m² h={room_h:.1f}m "
          f"energy={total_energy:.0f} exposure={exposure:.3f}")
    print(f"  poi={len(poi_objects)} "
          f"(furniture={len(furniture_result.get('placed', []))} "
          f"paintings={len(wall_result['paintings'])} "
          f"windows={len(wall_result['windows'])} "
          f"mounted={len(wall_result['mounted_objects'])}) "
          f"cameras={len(cameras)} "
          f"lights={len(lighting['lights'])} "
          f"distractors={len(distractors)} "
          f"internal_walls={len(internal_walls)} "
          f"sub_rooms={len(sub_rooms)} "
          f"portals={len(portals)}")

    return filepath


# ---- Command-line entry point ----

def main() -> None:
    """
    Command-line entry point (run inside Blender).

    Usage:
      blender -b --python lenscope/gen1/generate_scene.py -- \\
        --seed 42 --output_dir ./scene_generator_output --room_type random \\
        --n_scenes 10
    """
    import argparse

    # Arguments after "--"
    argv = sys.argv
    if '--' in argv:
        argv = argv[argv.index('--') + 1:]
    else:
        argv = []

    parser = argparse.ArgumentParser(
        description="First-generation procedural indoor scene generator")
    parser.add_argument(
        '--seed', type=int, default=42,
        help="first random seed (default: 42)")
    parser.add_argument(
        '--output_dir', type=str,
        default=os.environ.get('UNICOL_SCENE_OUTPUT', './scene_generator_output'),
        help="output directory")
    parser.add_argument(
        '--room_type', type=str, default='random',
        choices=ROOM_TYPES + ['random'],
        help="room type (default: random)")
    parser.add_argument(
        '--n_scenes', type=int, default=1,
        help="number of scenes to generate (default: 1)")
    parser.add_argument(
        '--n_cameras', type=int, default=5,
        help="unused: the camera count follows the floor area (at least 5, five per 15 m2)")
    parser.add_argument(
        '--n_paintings', type=int, default=2,
        help="paintings per scene (default: 2)")
    parser.add_argument(
        '--n_windows', type=int, default=2,
        help="windows per scene (default: 2)")
    parser.add_argument(
        '--n_mounted', type=int, default=3,
        help="wall-mounted objects per scene (default: 3)")

    args = parser.parse_args(argv)

    print("=" * 60)
    print("First-generation procedural indoor scene generator")
    print(f"  Seeds: {args.seed} → {args.seed + args.n_scenes - 1}")
    print(f"  Output: {args.output_dir}")
    print(f"  Room type: {args.room_type}")
    print(f"  Mode: generate .blend only (no rendering)")
    print("=" * 60)

    t_total = time.time()
    for i in range(args.n_scenes):
        current_seed = args.seed + i
        try:
            filepath = generate_scene(
                seed=current_seed,
                output_dir=args.output_dir,
                room_type=args.room_type,
                n_cameras=args.n_cameras,
                n_paintings=args.n_paintings,
                n_windows=args.n_windows,
                n_mounted=args.n_mounted,
            )
        except Exception as exc:
            print(f"[ERROR] seed={current_seed}: {exc}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\n{'=' * 60}")
    print(f"Done: {args.n_scenes} scenes in {time.time() - t_total:.1f}s")
    print(f"Output: {args.output_dir}")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
