#!/usr/bin/env python3
"""
Room shells: floor plans, floor, ceiling, external walls and internal walls with sub-room tracking.

Builds the room shell (floor + ceiling + external walls + internal walls) and provides:
  1. generate_floor_plan(): 50% single room of varied shape (40-60 m^2) / 50% BSP-partitioned apartment (200-300 m^2)
  2. RoomBuilder: builds the complete room geometry with BMeshFactory + core_utils
     - Ceiling height is given by the caller
     - External walls + double-sided internal walls (both faces are registered in wall_info)
     - Internal wall ends are capped, so each internal wall is a closed box
     - The RoomShell includes the internal wall geometry (camera rays cannot pass through)
     - All wall faces use make_wall_paint

  The floor plan also returns sub_rooms: list[list[(x,y)]], the physical sub-room polygons after partitioning

Design rules:
  - All polygons are strictly CCW
  - Wall normals point inwards (prevents black walls)
  - Internal walls are double-sided (two back-to-back quads + wall_info for both faces)
  - Doorways: internal wall segments of at least 2.5 m get a 1.2 m opening
  - BSP splitting stops below a sub-room area of 20 m^2
  - No bpy.ops, no global random state, no destructive scaling
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import bpy
from mathutils import Vector

from .core_utils import (
    calculate_inward_normal,
    create_data_object,
    edge_midpoint_and_normal,
    link_object_to_scene,
    polygon_centroid,
    polygon_signed_area,
    point_in_polygon,
)
from .geometry_factory import BMeshFactory, ProceduralMaterialFactory


# ---- Floor plan generator (pure 2D geometry) ----

def _ensure_ccw(verts: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Return the 2D vertices in counter-clockwise (CCW) order."""
    if polygon_signed_area(verts) < 0:
        verts = list(reversed(verts))
    return verts


def _generate_rectangle(
    rng,
    width_range: tuple[float, float] = (6.0, 10.0),
    depth_range: tuple[float, float] = (5.0, 8.0),
) -> list[tuple[float, float]]:
    """Rectangular room (area about 40-60 m^2)."""
    w = rng.uniform(*width_range)
    d = rng.uniform(*depth_range)
    hw, hd = w / 2, d / 2
    verts = [(-hw, -hd), (hw, -hd), (hw, hd), (-hw, hd)]
    return _ensure_ccw(verts)


def _generate_l_shape(
    rng,
    arm_range: tuple[float, float] = (6.0, 10.0),
    notch_ratio: tuple[float, float] = (0.35, 0.65),
) -> list[tuple[float, float]]:
    """
    L-shaped room.

    Shape:
        +------+
        |      |
        |      +--+
        |      |  |
        +------+--+
    """
    w = rng.uniform(*arm_range)
    h = rng.uniform(*arm_range)
    # Notch ratios
    nx = rng.uniform(*notch_ratio)
    ny = rng.uniform(*notch_ratio)
    cut_x = w * nx
    cut_y = h * ny

    verts = [
        (0, 0),
        (w, 0),
        (w, h - cut_y),
        (w - cut_x, h - cut_y),
        (w - cut_x, h),
        (0, h),
    ]
    # Center on the centroid
    cx, cy = polygon_centroid(verts)
    verts = [(x - cx, y - cy) for x, y in verts]
    return _ensure_ccw(verts)


def _generate_u_shape(
    rng,
    width_range: tuple[float, float] = (7.0, 11.0),
    depth_range: tuple[float, float] = (6.0, 10.0),
    notch_ratio: tuple[float, float] = (0.25, 0.45),
) -> list[tuple[float, float]]:
    """
    U-shaped room (notch in the middle).

    Shape:
        +--+    +--+
        |  |    |  |
        |  +----+  |
        |          |
        +----------+
    """
    w = rng.uniform(*width_range)
    h = rng.uniform(*depth_range)
    nw = w * rng.uniform(*notch_ratio)  # notch width
    nh = h * rng.uniform(0.3, 0.6)      # notch depth
    # Width of the left and right wings
    wing_w = (w - nw) / 2

    verts = [
        (0, 0),
        (w, 0),
        (w, h),
        (w - wing_w, h),
        (w - wing_w, h - nh),
        (wing_w, h - nh),
        (wing_w, h),
        (0, h),
    ]
    cx, cy = polygon_centroid(verts)
    verts = [(x - cx, y - cy) for x, y in verts]
    return _ensure_ccw(verts)


def _generate_t_shape(
    rng,
    width_range: tuple[float, float] = (7.0, 11.0),
    depth_range: tuple[float, float] = (5.0, 9.0),
) -> list[tuple[float, float]]:
    """
    T-shaped room.

    Shape:
        +----------+
        |          |
        +--+    +--+
           |    |
           +----+
    """
    top_w = rng.uniform(*width_range)
    top_h = rng.uniform(1.5, 3.5)
    stem_w = top_w * rng.uniform(0.3, 0.55)
    stem_h = rng.uniform(*depth_range)
    # Stem X offset (centered)
    sx = (top_w - stem_w) / 2

    # Vertices, starting from the bottom-left corner of the stem
    verts = [
        (sx, 0),                       # stem bottom-left
        (sx + stem_w, 0),              # stem bottom-right
        (sx + stem_w, stem_h),         # stem top-right = right joint with the top bar
        (top_w, stem_h),               # top bar bottom-right
        (top_w, stem_h + top_h),       # top bar top-right
        (0, stem_h + top_h),           # top bar top-left
        (0, stem_h),                   # top bar bottom-left
        (sx, stem_h),                  # stem top-left = left joint with the top bar
    ]
    cx, cy = polygon_centroid(verts)
    verts = [(x - cx, y - cy) for x, y in verts]
    return _ensure_ccw(verts)


def _generate_circle(
    rng,
    radius_range: tuple[float, float] = (3.8, 4.8),
    segments: int = 24,
) -> list[tuple[float, float]]:
    """Approximately circular room (regular polygon, about 40-60 m^2)."""
    r = rng.uniform(*radius_range)
    n = max(12, segments)
    # CCW: angles increase from 0
    verts = [
        (r * math.cos(2 * math.pi * i / n),
         r * math.sin(2 * math.pi * i / n))
        for i in range(n)
    ]
    return _ensure_ccw(verts)


def _generate_pentagon(
    rng,
    size_range: tuple[float, float] = (4.0, 5.5),
) -> list[tuple[float, float]]:
    """Irregular pentagonal room (about 40-60 m^2)."""
    base_r = rng.uniform(*size_range)
    verts = []
    for i in range(5):
        angle = 2 * math.pi * i / 5 + rng.uniform(-0.15, 0.15)
        r = base_r * rng.uniform(0.75, 1.25)
        verts.append((r * math.cos(angle), r * math.sin(angle)))
    return _ensure_ccw(verts)


# Registry of the single-room floor plan generators
_FLOOR_PLAN_GENERATORS = {
    'rectangle': _generate_rectangle,
    'l_shape': _generate_l_shape,
    'u_shape': _generate_u_shape,
    't_shape': _generate_t_shape,
    'circle': _generate_circle,
    'pentagon': _generate_pentagon,
}

ROOM_TYPES = list(_FLOOR_PLAN_GENERATORS.keys()) + ['bsp_apartment']


# ---- BSP helpers ----

def _bsp_split_rect(
    rng,
    xmin: float, ymin: float,
    xmax: float, ymax: float,
    depth: int,
    max_depth: int,
    axis: int,  # 0=X, 1=Y
    min_room_area: float = 20.0,
) -> tuple[list[tuple[tuple[float, float], tuple[float, float]]],
           list[list[tuple[float, float]]]]:
    """
    Recursive BSP split of a rectangular region.

    Returns (wall_segments, leaf_rooms).
    Splitting stops when a leaf sub-room would fall below min_room_area.
    leaf_rooms holds the CCW vertex list of each leaf rectangle.
    """
    w = xmax - xmin
    h = ymax - ymin
    area = w * h

    if depth >= max_depth or area < min_room_area * 1.5:
        # Leaf
        room = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]
        return [], [room]

    walls = []
    rooms = []

    def _try_split(ax):
        if ax == 0 and w >= 4.0:
            t = rng.uniform(0.35, 0.65)
            cx = xmin + w * t
            left_area = (cx - xmin) * h
            right_area = (xmax - cx) * h
            if left_area >= min_room_area and right_area >= min_room_area:
                walls.append(((cx, ymin), (cx, ymax)))
                w1, r1 = _bsp_split_rect(
                    rng, xmin, ymin, cx, ymax,
                    depth + 1, max_depth, 1, min_room_area)
                w2, r2 = _bsp_split_rect(
                    rng, cx, ymin, xmax, ymax,
                    depth + 1, max_depth, 1, min_room_area)
                walls.extend(w1)
                walls.extend(w2)
                rooms.extend(r1)
                rooms.extend(r2)
                return True
        elif ax == 1 and h >= 4.0:
            t = rng.uniform(0.35, 0.65)
            cy = ymin + h * t
            bottom_area = w * (cy - ymin)
            top_area = w * (ymax - cy)
            if bottom_area >= min_room_area and top_area >= min_room_area:
                walls.append(((xmin, cy), (xmax, cy)))
                w1, r1 = _bsp_split_rect(
                    rng, xmin, ymin, xmax, cy,
                    depth + 1, max_depth, 0, min_room_area)
                w2, r2 = _bsp_split_rect(
                    rng, xmin, cy, xmax, ymax,
                    depth + 1, max_depth, 0, min_room_area)
                walls.extend(w1)
                walls.extend(w2)
                rooms.extend(r1)
                rooms.extend(r2)
                return True
        return False

    if not _try_split(axis):
        if not _try_split(1 - axis):
            # Cannot split: leaf
            room = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]
            return [], [room]

    return walls, rooms


def _apply_doorways(
    rng,  # random.Random
    segments: list[tuple[tuple[float, float], tuple[float, float]]],
    min_door_wall: float = 2.5,
    door_width: float = 1.2,
) -> tuple[list[tuple[tuple[float, float], tuple[float, float]]], list[dict]]:
    """
    Cut doorways into internal wall segments (near a wall end when possible) and return portal data.

    Doorway position:
      - length >= 3.0: near a wall end (1.0 m from a randomly chosen end point)
      - min_door_wall <= length < 3.0: in the middle
      - length < min_door_wall: no doorway (solid wall)

    Returns
    -------
    (result_segments, portals)
      result_segments: wall segments after cutting
      portals: list[dict], one {'center': (cx, cy), 'normal': (nx, ny)} per doorway
    """
    result = []
    portals: list[dict] = []

    for (x1, y1), (x2, y2) in segments:
        dx, dy = x2 - x1, y2 - y1
        length = math.sqrt(dx * dx + dy * dy)
        if length < min_door_wall:
            result.append(((x1, y1), (x2, y2)))
        else:
            ux, uy = dx / length, dy / length

            # Doorway near a wall end, away from T-junctions in the middle of the wall
            if length >= 3.0:
                mid = rng.choice([1.0, length - 1.0])
            else:
                mid = length / 2.0

            half_door = door_width / 2.0
            door_start = mid - half_door
            door_end = mid + half_door

            if door_start > 0.1:
                result.append((
                    (x1, y1),
                    (x1 + ux * door_start, y1 + uy * door_start),
                ))
            if length - door_end > 0.1:
                result.append((
                    (x1 + ux * door_end, y1 + uy * door_end),
                    (x2, y2),
                ))

            # Portal: center of the doorway gap + normal
            portal_cx = x1 + ux * mid
            portal_cy = y1 + uy * mid
            # Normal: left-hand side (-dy, dx) / length
            pnx = -dy / length
            pny = dx / length
            portals.append({
                'center': (portal_cx, portal_cy),
                'normal': (pnx, pny),
            })

    return result, portals


def generate_floor_plan(
    rng,  # random.Random
    room_type: str = 'random',
) -> tuple[list[tuple[float, float]], list, str, tuple, list, list]:
    """
    Floor plan: 50% single room of varied shape vs 50% BSP-partitioned apartment.

    Returns
    -------
    (floor_verts, internal_walls, room_type, bbox, sub_rooms, portals) : tuple
        floor_verts : list[(x,y)], CCW outer vertices
        internal_walls : list[((x1,y1),(x2,y2))], internal wall segments (doorways cut out)
        room_type : str, actual type name
        bbox : (xmin, ymin, xmax, ymax)
        sub_rooms : list[list[(x,y)]], sub-room polygons
        portals : list[dict], doorway data {'center': (cx,cy), 'normal': (nx,ny)}
    """
    # Choose the plan type
    if room_type == 'random':
        use_bsp = rng.random() < 0.5
    elif room_type == 'bsp_apartment':
        use_bsp = True
    elif room_type in _FLOOR_PLAN_GENERATORS:
        use_bsp = False
    else:
        use_bsp = rng.random() < 0.5

    if use_bsp:
        # BSP apartment: 14-18 m per side (200-300 m^2)
        w = rng.uniform(14.0, 18.0)
        h = rng.uniform(14.0, 18.0)
        hw, hh = w / 2, h / 2
        floor_verts = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
        floor_verts = _ensure_ccw(floor_verts)

        max_depth = rng.randint(2, 3)
        start_axis = rng.choice([0, 1])
        raw_walls, sub_rooms = _bsp_split_rect(
            rng, -hw, -hh, hw, hh,
            depth=0, max_depth=max_depth, axis=start_axis,
            min_room_area=20.0)

        internal_walls, portals = _apply_doorways(rng, raw_walls)
        bbox = (-hw, -hh, hw, hh)
        actual_type = 'bsp_apartment'
        return floor_verts, internal_walls, actual_type, bbox, sub_rooms, portals
    else:
        # Single room of varied shape (40-60 m^2)
        if room_type == 'random' or room_type not in _FLOOR_PLAN_GENERATORS:
            chosen = rng.choice(list(_FLOOR_PLAN_GENERATORS.keys()))
        else:
            chosen = room_type
        gen_fn = _FLOOR_PLAN_GENERATORS[chosen]
        floor_verts = gen_fn(rng)
        floor_verts = _ensure_ccw(floor_verts)

        xs = [v[0] for v in floor_verts]
        ys = [v[1] for v in floor_verts]
        bbox = (min(xs), min(ys), max(xs), max(ys))
        # Single room: sub_rooms only contains the outline itself
        sub_rooms = [list(floor_verts)]
        return floor_verts, [], chosen, bbox, sub_rooms, []


# ---- RoomBuilder: room shell ----

class RoomBuilder:
    """
    Builds the complete room geometry with BMeshFactory + core_utils (including internal walls and end caps).

    Steps:
      1. Take the CCW 2D outer vertices + internal_walls + sub_rooms
      2. Build the floor (normal +Z)
      3. Build the ceiling (normal -Z)
      4. Build the external walls edge by edge (normals point inwards)
      5. Build double-sided internal walls with end-cap quads (one quad + one wall_info entry per face)
      6. Assign make_wall_paint to all walls
      7. The RoomShell includes the internal wall geometry
    """

    def __init__(
        self,
        rng,  # random.Random
        floor_verts: list[tuple[float, float]],
        ceiling_height: float = 2.8,
        collection_name: str = "RoomGeometry",
        internal_walls: list = None,
        sub_rooms: list = None,
    ):
        """
        Parameters
        ----------
        rng : random.Random
        floor_verts : list[(float, float)]
            CCW outer vertices
        ceiling_height : float
        collection_name : str
        internal_walls : list[((x1,y1),(x2,y2))], optional
            Internal wall segments (doorways cut out); no internal walls by default
        sub_rooms : list[list[(x,y)]], optional
            Sub-room polygons
        """
        self.rng = rng
        self.floor_verts = floor_verts
        self.ceiling_height = ceiling_height
        self.internal_walls = internal_walls if internal_walls is not None else []
        self.sub_rooms = sub_rooms if sub_rooms is not None else [list(floor_verts)]

        # Dedicated collection
        self.collection = bpy.data.collections.new(collection_name)
        bpy.context.scene.collection.children.link(self.collection)

        # Build results
        self.floor_obj: Optional[bpy.types.Object] = None
        self.ceiling_obj: Optional[bpy.types.Object] = None
        self.wall_objects: list[bpy.types.Object] = []
        self.internal_wall_objects: list[bpy.types.Object] = []
        self.room_shell_obj: Optional[bpy.types.Object] = None

        # Wall info cache (used by the constraint solver)
        self._wall_info: list[dict] = []

        # One wall paint per sub-room, so all walls of a sub-room share one color
        self.room_wall_mats: list[bpy.types.Material] = [
            ProceduralMaterialFactory.make_wall_paint(
                f"RoomWallPaint_{k:02d}", self.rng)
            for k in range(len(self.sub_rooms))
        ]

    # ---- Floor ----

    def build_floor(self) -> bpy.types.Object:
        """
        Build the floor mesh (Z = 0, normal +Z) with a random floor material.

        Returns
        -------
        bpy.types.Object
        """
        mesh = BMeshFactory.create_polygon_face(
            name="Floor",
            verts_2d=self.floor_verts,
            z_height=0.0,
            flip_normal=False,
        )
        self.floor_obj = create_data_object(
            "Floor", mesh, collection=self.collection)

        # Material: random choice from 5 floor material entries
        mat_type = self.rng.choice(['tile', 'wood', 'concrete', 'marble', 'wood'])
        if mat_type == 'tile':
            mat = ProceduralMaterialFactory.make_tile("FloorMat", self.rng)
        elif mat_type == 'wood':
            mat = ProceduralMaterialFactory.make_wood("FloorMat", self.rng)
        elif mat_type == 'concrete':
            mat = ProceduralMaterialFactory.make_concrete("FloorMat", self.rng)
        elif mat_type == 'marble':
            mat = ProceduralMaterialFactory.make_marble("FloorMat", self.rng)
        else:
            mat = ProceduralMaterialFactory.make_wood("FloorMat", self.rng)
        self.floor_obj.data.materials.append(mat)

        return self.floor_obj

    # ---- Ceiling ----

    def build_ceiling(self) -> bpy.types.Object:
        """
        Build the ceiling mesh (Z = ceiling_height, normal -Z).

        Returns
        -------
        bpy.types.Object
        """
        mesh = BMeshFactory.create_polygon_face(
            name="Ceiling",
            verts_2d=self.floor_verts,
            z_height=self.ceiling_height,
            flip_normal=True,  # normal points down
        )
        self.ceiling_obj = create_data_object(
            "Ceiling", mesh, collection=self.collection)

        # Ceiling material: wall paint rather than a plain principled color
        mat = ProceduralMaterialFactory.make_wall_paint(
            "CeilingMat", self.rng)
        self.ceiling_obj.data.materials.append(mat)

        return self.ceiling_obj

    # ---- External walls (one quad per edge) ----

    def build_walls(self) -> list[bpy.types.Object]:
        """
        Build the external walls edge by edge (normals point inwards) with make_wall_paint.
        Also caches wall_info.
        """
        n = len(self.floor_verts)
        self.wall_objects.clear()
        self._wall_info.clear()

        for i in range(n):
            v1 = self.floor_verts[i]
            v2 = self.floor_verts[(i + 1) % n]

            nx, ny = calculate_inward_normal(v1, v2)
            if abs(nx) < 1e-8 and abs(ny) < 1e-8:
                continue

            wall_mesh = BMeshFactory.create_wall_quad(
                name=f"Wall_{i:03d}",
                v1=v1, v2=v2,
                height=self.ceiling_height,
                normal_inward=(nx, ny),
            )
            wall_obj = create_data_object(
                f"Wall_{i:03d}", wall_mesh, collection=self.collection)

            # Find the sub-room of this wall by testing a point slightly offset along the normal
            mx = (v1[0] + v2[0]) / 2.0
            my = (v1[1] + v2[1]) / 2.0
            test_px = mx + nx * 0.05
            test_py = my + ny * 0.05
            k = 0  # fallback if no sub-room contains the test point
            for ki, sub_poly in enumerate(self.sub_rooms):
                if point_in_polygon(test_px, test_py, sub_poly):
                    k = ki
                    break
            wall_obj.data.materials.append(self.room_wall_mats[k])
            self.wall_objects.append(wall_obj)

            mx = (v1[0] + v2[0]) / 2.0
            my = (v1[1] + v2[1]) / 2.0
            length = math.sqrt(
                (v2[0] - v1[0]) ** 2 + (v2[1] - v1[1]) ** 2)
            self._wall_info.append({
                'v1': v1,
                'v2': v2,
                'midpoint': (mx, my),
                'normal_inward': (nx, ny),
                'length': length,
                'index': i,
                'is_internal': False,
            })

        return self.wall_objects

    # ---- Internal walls (double-sided) ----

    def build_internal_walls(self) -> list[bpy.types.Object]:
        """
        Build double-sided internal walls with end-cap quads from internal_walls.

        Each internal wall segment produces:
          - Front quad: normal = left-hand direction (-dy, dx)
          - Back quad: normal = right-hand direction (dy, -dx)
          - Two end caps: perpendicular to the wall, width = wall_thickness, normals pointing outwards

        Both faces are registered in wall_info.
        The end caps close each internal wall into a solid box.
        """
        self.internal_wall_objects.clear()

        if not self.internal_walls:
            return []

        wall_thickness = 0.06  # internal wall thickness: 6 cm

        for idx, ((x1, y1), (x2, y2)) in enumerate(self.internal_walls):
            dx, dy = x2 - x1, y2 - y1
            seg_len = math.sqrt(dx * dx + dy * dy)
            if seg_len < 0.05:
                continue

            # Normal: left-hand side = (-dy, dx) / len
            nx_a = -dy / seg_len
            ny_a = dx / seg_len
            # Normal of the back face
            nx_b = dy / seg_len
            ny_b = -dx / seg_len

            half_t = wall_thickness / 2.0

            # Front quad
            v1a = (x1 + nx_a * half_t, y1 + ny_a * half_t)
            v2a = (x2 + nx_a * half_t, y2 + ny_a * half_t)
            mesh_a = BMeshFactory.create_wall_quad(
                f"InternalWall_{idx:03d}_A",
                v1=v1a, v2=v2a,
                height=self.ceiling_height,
                normal_inward=(nx_a, ny_a),
            )
            obj_a = create_data_object(
                f"InternalWall_{idx:03d}_A", mesh_a,
                collection=self.collection)
            # Sub-room of face A (test point offset along the normal)
            _mx_a = (v1a[0] + v2a[0]) / 2.0
            _my_a = (v1a[1] + v2a[1]) / 2.0
            _tpx_a = _mx_a + nx_a * 0.05
            _tpy_a = _my_a + ny_a * 0.05
            _ka = 0
            for _ki, _sp in enumerate(self.sub_rooms):
                if point_in_polygon(_tpx_a, _tpy_a, _sp):
                    _ka = _ki
                    break
            obj_a.data.materials.append(self.room_wall_mats[_ka])
            self.internal_wall_objects.append(obj_a)

            # Back quad
            v1b = (x1 + nx_b * half_t, y1 + ny_b * half_t)
            v2b = (x2 + nx_b * half_t, y2 + ny_b * half_t)
            mesh_b = BMeshFactory.create_wall_quad(
                f"InternalWall_{idx:03d}_B",
                v1=v1b, v2=v2b,
                height=self.ceiling_height,
                normal_inward=(nx_b, ny_b),
            )
            obj_b = create_data_object(
                f"InternalWall_{idx:03d}_B", mesh_b,
                collection=self.collection)
            # Sub-room of face B (test point offset along the normal)
            _mx_b = (v1b[0] + v2b[0]) / 2.0
            _my_b = (v1b[1] + v2b[1]) / 2.0
            _tpx_b = _mx_b + nx_b * 0.05
            _tpy_b = _my_b + ny_b * 0.05
            _kb = 0
            for _ki, _sp in enumerate(self.sub_rooms):
                if point_in_polygon(_tpx_b, _tpy_b, _sp):
                    _kb = _ki
                    break
            obj_b.data.materials.append(self.room_wall_mats[_kb])
            self.internal_wall_objects.append(obj_b)

            # End caps that close the wall
            # Unit direction of the segment
            ux, uy = dx / seg_len, dy / seg_len

            # Cap at end 1: corners v1a and v1b, normal (-ux, -uy)
            cap1_v1 = (x1 + nx_a * half_t, y1 + ny_a * half_t)
            cap1_v2 = (x1 + nx_b * half_t, y1 + ny_b * half_t)
            cap1_mesh = BMeshFactory.create_wall_quad(
                f"InternalWall_{idx:03d}_Cap1",
                v1=cap1_v1, v2=cap1_v2,
                height=self.ceiling_height,
                normal_inward=(-ux, -uy),
            )
            cap1_obj = create_data_object(
                f"InternalWall_{idx:03d}_Cap1", cap1_mesh,
                collection=self.collection)
            # Cap 1 uses the material of the sub-room of face A
            cap1_obj.data.materials.append(self.room_wall_mats[_ka])
            self.internal_wall_objects.append(cap1_obj)

            # Cap at end 2: corners v2a and v2b, normal (+ux, +uy)
            cap2_v1 = (x2 + nx_b * half_t, y2 + ny_b * half_t)
            cap2_v2 = (x2 + nx_a * half_t, y2 + ny_a * half_t)
            cap2_mesh = BMeshFactory.create_wall_quad(
                f"InternalWall_{idx:03d}_Cap2",
                v1=cap2_v1, v2=cap2_v2,
                height=self.ceiling_height,
                normal_inward=(ux, uy),
            )
            cap2_obj = create_data_object(
                f"InternalWall_{idx:03d}_Cap2", cap2_mesh,
                collection=self.collection)
            # Cap 2 uses the material of the sub-room of face B
            cap2_obj.data.materials.append(self.room_wall_mats[_kb])
            self.internal_wall_objects.append(cap2_obj)

            # wall_info for both faces
            mx = (x1 + x2) / 2.0
            my = (y1 + y2) / 2.0
            base_idx = len(self._wall_info)
            self._wall_info.append({
                'v1': (x1, y1), 'v2': (x2, y2),
                'midpoint': (mx, my),
                'normal_inward': (nx_a, ny_a),
                'length': seg_len,
                'index': base_idx,
                'is_internal': True,
            })
            self._wall_info.append({
                'v1': (x1, y1), 'v2': (x2, y2),
                'midpoint': (mx, my),
                'normal_inward': (nx_b, ny_b),
                'length': seg_len,
                'index': base_idx + 1,
                'is_internal': True,
            })

        return self.internal_wall_objects

    # ---- RoomShell (closed shell for ray casting) ----

    def build_room_shell(self) -> bpy.types.Object:
        """
        Build the RoomShell: the shell and the internal walls merged into one closed mesh.
        Used for ray-cast occlusion checks of cameras; the internal walls are included so rays cannot pass through them.
        """
        mesh_list = []

        # Floor face (normal +Z)
        floor_mesh = BMeshFactory.create_polygon_face(
            "Shell_Floor", self.floor_verts, z_height=0.0,
            flip_normal=False)
        mesh_list.append(floor_mesh)

        # Ceiling face (normal -Z)
        ceiling_mesh = BMeshFactory.create_polygon_face(
            "Shell_Ceiling", self.floor_verts,
            z_height=self.ceiling_height, flip_normal=True)
        mesh_list.append(ceiling_mesh)

        # External wall faces
        n = len(self.floor_verts)
        for i in range(n):
            v1 = self.floor_verts[i]
            v2 = self.floor_verts[(i + 1) % n]
            nx, ny = calculate_inward_normal(v1, v2)
            if abs(nx) < 1e-8 and abs(ny) < 1e-8:
                continue
            wall_mesh = BMeshFactory.create_wall_quad(
                f"Shell_Wall_{i:03d}",
                v1=v1, v2=v2,
                height=self.ceiling_height,
                normal_inward=(nx, ny))
            mesh_list.append(wall_mesh)

        # Internal wall faces are part of the shell too
        for idx, ((x1, y1), (x2, y2)) in enumerate(self.internal_walls):
            dx, dy = x2 - x1, y2 - y1
            seg_len = math.sqrt(dx * dx + dy * dy)
            if seg_len < 0.05:
                continue
            nx = -dy / seg_len
            ny = dx / seg_len
            mesh_list.append(BMeshFactory.create_wall_quad(
                f"Shell_IWall_{idx:03d}",
                v1=(x1, y1), v2=(x2, y2),
                height=self.ceiling_height,
                normal_inward=(nx, ny)))

        # Merge
        shell_mesh = BMeshFactory.merge_meshes("RoomShell", mesh_list)

        # Remove the temporary mesh data blocks
        for m in mesh_list:
            try:
                bpy.data.meshes.remove(m, do_unlink=True)
            except Exception:
                pass

        self.room_shell_obj = create_data_object(
            "RoomShell", shell_mesh, collection=self.collection)
        self.room_shell_obj.hide_render = True
        self.room_shell_obj.display_type = 'WIRE'

        return self.room_shell_obj

    # ---- Build everything ----

    def build_all(self) -> dict:
        """
        Build floor + ceiling + external walls + internal walls (with end caps) + RoomShell.
        """
        floor = self.build_floor()
        ceiling = self.build_ceiling()
        walls = self.build_walls()
        internal = self.build_internal_walls()
        shell = self.build_room_shell()

        return {
            'floor': floor,
            'ceiling': ceiling,
            'walls': walls,
            'internal_walls_obj': internal,
            'room_shell': shell,
            'floor_verts': self.floor_verts,
            'ceiling_height': self.ceiling_height,
            'wall_info': self._wall_info,
            'internal_walls': self.internal_walls,
            'sub_rooms': self.sub_rooms,
            'collection': self.collection,
        }

    # ---- Queries (used by the constraint solver) ----

    @property
    def wall_info(self) -> list[dict]:
        """
        Wall geometry entries (external walls + both faces of the internal walls).

        Each entry has an 'is_internal': bool field.
        """
        return self._wall_info

    def get_nearest_wall(
        self,
        px: float,
        py: float,
    ) -> dict:
        """
        Find the wall closest to the point (px, py).

        Parameters
        ----------
        px, py : float
            Query point

        Returns
        -------
        dict
            wall_info entry of the closest wall, with an added 'distance' field
        """
        if not self._wall_info:
            raise RuntimeError("Walls not built yet. Call build_walls() first.")

        best_dist = float('inf')
        best_wall = self._wall_info[0]

        for wall in self._wall_info:
            v1 = wall['v1']
            v2 = wall['v2']
            # Point-to-segment distance
            dx, dy = v2[0] - v1[0], v2[1] - v1[1]
            length_sq = dx * dx + dy * dy
            if length_sq < 1e-12:
                dist = math.sqrt((px - v1[0]) ** 2 + (py - v1[1]) ** 2)
            else:
                t = max(0.0, min(1.0,
                                 ((px - v1[0]) * dx + (py - v1[1]) * dy)
                                 / length_sq))
                proj_x = v1[0] + t * dx
                proj_y = v1[1] + t * dy
                dist = math.sqrt((px - proj_x) ** 2 + (py - proj_y) ** 2)

            if dist < best_dist:
                best_dist = dist
                best_wall = wall

        result = dict(best_wall)
        result['distance'] = best_dist
        return result
