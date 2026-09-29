#!/usr/bin/env python3
"""
Core memory management and math utilities for the scene generator.

Provides:
  1. AggressiveMemoryManager: aggressive cleanup against C++-side memory leaks
  2. Polygon helpers: area centroid, point-in-polygon test, point-to-boundary distance
  3. Inward wall normals: strict CCW inward normal (-dy, dx), normalized (prevents black walls)

All functions follow three rules:
  - No bpy.ops for geometry construction
  - Never break the physical 3D scale
  - No implicit global randomness (all randomness goes through an explicit rng argument)
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import bpy
from mathutils import Vector


# ---- AggressiveMemoryManager: prevent C++-side memory leaks ----

class AggressiveMemoryManager:
    """
    Defensive memory cleanup.

    Blender's C/C++ allocations are not managed by the Python GC. In large batch
    scene generation, orphan data blocks that are not removed (meshes, materials,
    textures, images) stay resident and exhaust system RAM within hours.

    Cleanup follows the dependency tree in reverse:
      Collections -> Objects -> Meshes -> Curves -> Materials ->
      Node Groups -> Textures -> Images

    so references are released first and the referenced data blocks can then be
    freed safely.
    """

    # Reverse dependency order, from the highest level to the lowest
    _DATA_BLOCK_HIERARCHY: list[str] = [
        "collections",
        "objects",
        "meshes",
        "curves",
        "materials",
        "node_groups",
        "textures",
        "images",
    ]

    @staticmethod
    def clear_scene() -> None:
        """
        Remove all objects and data blocks of the current scene.

        Steps:
          1. Unlink all objects from the scene through the data API
          2. Remove all data blocks in reverse dependency order (do_unlink=True)
          3. Remove lights, cameras and worlds
          4. No bpy.ops operators are used
        """
        scene = bpy.context.scene

        # Step 1: unlink all objects from the scene
        # First unlink objects from all collections
        for col in list(bpy.data.collections):
            for obj in list(col.objects):
                col.objects.unlink(obj)

        # Unlink objects from the scene's master collection
        for obj in list(scene.collection.objects):
            scene.collection.objects.unlink(obj)

        # Unlink child collections from the scene's master collection
        for col in list(scene.collection.children):
            scene.collection.children.unlink(col)

        # Step 2: remove all data blocks in reverse dependency order
        for attr_name in AggressiveMemoryManager._DATA_BLOCK_HIERARCHY:
            block_list = getattr(bpy.data, attr_name, None)
            if block_list is None:
                continue
            # Iterate backwards so removal inside the loop is safe
            for item in list(reversed(list(block_list))):
                try:
                    block_list.remove(item, do_unlink=True)
                except Exception:
                    pass  # some built-in data blocks cannot be removed; ignore

        # Step 3: remove light data blocks
        for light in list(bpy.data.lights):
            try:
                bpy.data.lights.remove(light, do_unlink=True)
            except Exception:
                pass

        # Step 4: remove camera data blocks
        for cam in list(bpy.data.cameras):
            try:
                bpy.data.cameras.remove(cam, do_unlink=True)
            except Exception:
                pass

        # Step 5: remove worlds
        worlds = list(bpy.data.worlds)
        for w in worlds:
            try:
                bpy.data.worlds.remove(w, do_unlink=True)
            except Exception:
                pass

    @staticmethod
    def purge_orphans() -> None:
        """
        Second orphan sweep. Indirectly referenced data blocks can remain after
        clear_scene; this scans again and removes data with users == 0.
        """
        for attr_name in AggressiveMemoryManager._DATA_BLOCK_HIERARCHY:
            block_list = getattr(bpy.data, attr_name, None)
            if block_list is None:
                continue
            orphans = [item for item in block_list if item.users == 0]
            for item in orphans:
                try:
                    block_list.remove(item, do_unlink=True)
                except Exception:
                    pass

        # Also remove orphan lights and cameras
        for light in [l for l in bpy.data.lights if l.users == 0]:
            try:
                bpy.data.lights.remove(light, do_unlink=True)
            except Exception:
                pass
        for cam in [c for c in bpy.data.cameras if c.users == 0]:
            try:
                bpy.data.cameras.remove(cam, do_unlink=True)
            except Exception:
                pass

    @staticmethod
    def full_cleanup() -> None:
        """Full cleanup: clear_scene followed by purge_orphans."""
        AggressiveMemoryManager.clear_scene()
        AggressiveMemoryManager.purge_orphans()


# ---- 2D polygon helpers ----

def polygon_signed_area(verts_2d: Sequence[tuple[float, float]]) -> float:
    """
    Signed area of a 2D polygon (shoelace formula).

    Positive for CCW order, negative for CW order.

    Parameters
    ----------
    verts_2d : list[(float, float)]
        Polygon vertices

    Returns
    -------
    float : signed area
    """
    n = len(verts_2d)
    if n < 3:
        return 0.0
    area = 0.0
    for i in range(n):
        xi, yi = verts_2d[i]
        xj, yj = verts_2d[(i + 1) % n]
        area += xi * yj - xj * yi
    return area / 2.0


def polygon_centroid(verts_2d: Sequence[tuple[float, float]]) -> tuple[float, float]:
    """
    Area centroid (signed-area centroid) of a 2D polygon.

    More accurate than the vertex mean for concave (L- or U-shaped) polygons.
    Falls back to the vertex mean if the area is degenerate (all points collinear).

    Parameters
    ----------
    verts_2d : list[(float, float)]
        Polygon vertices (CCW)

    Returns
    -------
    (cx, cy) : tuple[float, float]
    """
    n = len(verts_2d)
    if n == 0:
        return (0.0, 0.0)

    # Accumulate shoelace cross products
    A2 = 0.0
    cx = 0.0
    cy = 0.0
    for i in range(n):
        xi, yi = verts_2d[i]
        xj, yj = verts_2d[(i + 1) % n]
        cross = xi * yj - xj * yi
        A2 += cross
        cx += (xi + xj) * cross
        cy += (yi + yj) * cross

    if abs(A2) < 1e-12:
        # Degenerate polygon: fall back to the vertex mean
        xs = [v[0] for v in verts_2d]
        ys = [v[1] for v in verts_2d]
        return (sum(xs) / n, sum(ys) / n)

    cx /= (3.0 * A2)
    cy /= (3.0 * A2)
    return (cx, cy)


def point_in_polygon(px: float, py: float,
                     polygon: Sequence[tuple[float, float]]) -> bool:
    """
    Ray-casting test of whether the point (px, py) lies inside a 2D polygon.

    Casts a half-ray from the point towards +X and counts the edge crossings:
    odd = inside, even = outside.

    Parameters
    ----------
    px, py : float
        Test point
    polygon : list[(float, float)]
        Polygon vertices

    Returns
    -------
    bool : True if inside
    """
    n = len(polygon)
    if n < 3:
        return False
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > py) != (yj > py) and
                px < (xj - xi) * (py - yi) / (yj - yi + 1e-12) + xi):
            inside = not inside
        j = i
    return inside


def distance_to_polygon_boundary(
    px: float, py: float,
    polygon: Sequence[tuple[float, float]]
) -> float:
    """
    Shortest distance from the point (px, py) to any edge of the polygon.

    Used to keep object centers at least a margin away from the walls.
    Also correct for concave (L- or U-shaped) polygons.

    Parameters
    ----------
    px, py : float
        Query point
    polygon : list[(float, float)]
        Polygon vertices

    Returns
    -------
    float : shortest distance
    """
    min_dist = float('inf')
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        # Point-to-segment distance
        dx, dy = x2 - x1, y2 - y1
        length_sq = dx * dx + dy * dy
        if length_sq < 1e-12:
            dist = math.sqrt((px - x1) ** 2 + (py - y1) ** 2)
        else:
            t = max(0.0, min(1.0,
                             ((px - x1) * dx + (py - y1) * dy) / length_sq))
            proj_x = x1 + t * dx
            proj_y = y1 + t * dy
            dist = math.sqrt((px - proj_x) ** 2 + (py - proj_y) ** 2)
        min_dist = min(min_dist, dist)
    return min_dist


def _point_to_segment_dist(
    px: float, py: float,
    x1: float, y1: float,
    x2: float, y2: float,
) -> float:
    """Shortest 2D Euclidean distance from a point to a single segment (internal helper)."""
    dx, dy = x2 - x1, y2 - y1
    length_sq = dx * dx + dy * dy
    if length_sq < 1e-12:
        return math.sqrt((px - x1) ** 2 + (py - y1) ** 2)
    t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / length_sq))
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    return math.sqrt((px - proj_x) ** 2 + (py - proj_y) ** 2)


def distance_to_line_segments(
    px: float, py: float,
    segments: Sequence[tuple[tuple[float, float], tuple[float, float]]],
) -> float:
    """
    Shortest 2D Euclidean distance from the point (px, py) to a set of independent segments.

    Parameters
    ----------
    px, py : float
        Query point
    segments : list[((x1,y1), (x2,y2))]
        Independent line segments

    Returns
    -------
    float
        Shortest distance, or inf if segments is empty.
    """
    if not segments:
        return float('inf')
    min_dist = float('inf')
    for (x1, y1), (x2, y2) in segments:
        d = _point_to_segment_dist(px, py, x1, y1, x2, y2)
        if d < min_dist:
            min_dist = d
    return min_dist


def guaranteed_interior_point(
    polygon: Sequence[tuple[float, float]]
) -> tuple[float, float]:
    """
    Return a point inside the polygon, using several fallbacks.

    Order:
      1. Area centroid (works for most convex and concave polygons)
      2. Edge midpoints moved 20%/40%/60% towards the centroid
      3. Midpoints of diagonals
      4. The centroid from step 1 (last resort)

    Parameters
    ----------
    polygon : list[(float, float)]
        Polygon vertices

    Returns
    -------
    (x, y) : tuple[float, float]
    """
    cx, cy = polygon_centroid(polygon)
    if point_in_polygon(cx, cy, polygon):
        return (cx, cy)

    n = len(polygon)

    # Strategy 2: edge midpoints moved towards the centroid
    for i in range(n):
        j = (i + 1) % n
        mx = (polygon[i][0] + polygon[j][0]) / 2.0
        my = (polygon[i][1] + polygon[j][1]) / 2.0
        for t in [0.2, 0.4, 0.6]:
            px = mx + (cx - mx) * t
            py = my + (cy - my) * t
            if point_in_polygon(px, py, polygon):
                return (px, py)

    # Strategy 3: midpoints of diagonals
    for i in range(n):
        for j_off in range(2, min(n // 2 + 1, n)):
            j = (i + j_off) % n
            mx = (polygon[i][0] + polygon[j][0]) / 2.0
            my = (polygon[i][1] + polygon[j][1]) / 2.0
            if point_in_polygon(mx, my, polygon):
                return (mx, my)

    # Last resort
    return (cx, cy)


def random_interior_point(
    rng,  # random.Random instance
    floor_verts: Sequence[tuple[float, float]],
    margin: float = 0.5,
    internal_walls: list = None,
) -> tuple[float, float]:
    """
    Sample a random point inside the polygon, at least margin away from all walls.

    Fallback stages:
      1. Full margin (500 samples)
      2. Margin reduced to 50% (300 samples)
      3. Only require the point to be inside the polygon (200 samples)
      4. guaranteed_interior_point as the last resort

    Parameters
    ----------
    rng : random.Random
        Explicit random number generator (no shared global random state)
    floor_verts : list[(float, float)]
        Room floor polygon (CCW)
    margin : float
        Minimum distance to walls
    internal_walls : list[((x1,y1),(x2,y2))], optional
        Internal wall segments. If given, samples must also be at least margin away from them.

    Returns
    -------
    (x, y) : tuple[float, float]
    """
    xs = [v[0] for v in floor_verts]
    ys = [v[1] for v in floor_verts]
    xmin, xmax = min(xs) + 0.1, max(xs) - 0.1
    ymin, ymax = min(ys) + 0.1, max(ys) - 0.1

    def _check(x, y, m):
        if not point_in_polygon(x, y, floor_verts):
            return False
        if distance_to_polygon_boundary(x, y, floor_verts) < m:
            return False
        if internal_walls and m > 0:
            if distance_to_line_segments(x, y, internal_walls) < m:
                return False
        return True

    # Stage 1: full margin
    for _ in range(500):
        x = rng.uniform(xmin, xmax)
        y = rng.uniform(ymin, ymax)
        if _check(x, y, margin):
            return (x, y)

    # Stage 2: margin reduced to 50%
    reduced = margin * 0.5
    for _ in range(300):
        x = rng.uniform(xmin, xmax)
        y = rng.uniform(ymin, ymax)
        if _check(x, y, reduced):
            return (x, y)

    # Stage 3: inside the polygon only (no distance check)
    for _ in range(200):
        x = rng.uniform(xmin, xmax)
        y = rng.uniform(ymin, ymax)
        if point_in_polygon(x, y, floor_verts):
            return (x, y)

    # Last resort
    return guaranteed_interior_point(floor_verts)


# ---- Inward wall normals from strict CCW order ----

def calculate_inward_normal(
    v1: tuple[float, float],
    v2: tuple[float, float],
) -> tuple[float, float]:
    """
    Inward unit normal of the edge v1 -> v2 of a CCW polygon.

    For a polygon in strict counter-clockwise (CCW) order, the left-hand direction
    (-dy, dx) of the edge vector (dx, dy) always points into the polygon.

    Estimating the normal from the centroid instead fails for concave (L- or U-shaped)
    polygons whose centroid lies outside the polygon: the normal flips outwards and
    the back face of the wall points at the viewer, which renders black.

    Parameters
    ----------
    v1 : (float, float)
        Edge start (CCW order)
    v2 : (float, float)
        Edge end (CCW order)

    Returns
    -------
    (nx, ny) : tuple[float, float]
        Normalized inward unit normal.
        (0, 0) if the edge is degenerate (length close to 0).
    """
    dx = v2[0] - v1[0]
    dy = v2[1] - v1[1]

    # For a CCW polygon the inward normal is always the left-hand direction
    nx = -dy
    ny = dx

    # Normalize
    length = math.sqrt(nx * nx + ny * ny)
    if length < 1e-9:
        return (0.0, 0.0)

    return (nx / length, ny / length)


def edge_midpoint_and_normal(
    v1: tuple[float, float],
    v2: tuple[float, float],
) -> tuple[float, float, float, float]:
    """
    Midpoint and inward unit normal of a wall edge.

    Parameters
    ----------
    v1, v2 : (float, float)
        Edge end points (CCW order)

    Returns
    -------
    (mx, my, nx, ny) : tuple[float, float, float, float]
        mx, my = midpoint
        nx, ny = inward unit normal (normalized)
    """
    mx = (v1[0] + v2[0]) / 2.0
    my = (v1[1] + v2[1]) / 2.0
    nx, ny = calculate_inward_normal(v1, v2)
    return (mx, my, nx, ny)


# ---- Object creation at the data-block level (no bpy.ops) ----

def link_object_to_scene(obj: bpy.types.Object,
                         collection: Optional[bpy.types.Collection] = None
                         ) -> None:
    """
    Link an object to a collection (or the scene's master collection) without bpy.ops.

    Parameters
    ----------
    obj : bpy.types.Object
        Object to link
    collection : bpy.types.Collection, optional
        Target collection. Defaults to the scene's master collection.
    """
    if collection is None:
        collection = bpy.context.scene.collection
    collection.objects.link(obj)


def create_data_object(name: str, data: bpy.types.ID,
                       collection: Optional[bpy.types.Collection] = None
                       ) -> bpy.types.Object:
    """
    Create an object from a data block (Mesh/Light/Camera) and link it to the scene, without bpy.ops.

    Parameters
    ----------
    name : str
        Object name
    data : bpy.types.ID
        Data block (Mesh, Light, Camera, ...)
    collection : bpy.types.Collection, optional
        Target collection

    Returns
    -------
    bpy.types.Object
    """
    obj = bpy.data.objects.new(name, data)
    link_object_to_scene(obj, collection)
    return obj


def create_empty(name: str,
                 collection: Optional[bpy.types.Collection] = None
                 ) -> bpy.types.Object:
    """
    Create an Empty object (used for collection instancing) without bpy.ops.

    Parameters
    ----------
    name : str
        Object name
    collection : bpy.types.Collection, optional
        Target collection

    Returns
    -------
    bpy.types.Object (Empty)
    """
    obj = bpy.data.objects.new(name, None)  # None → Empty
    link_object_to_scene(obj, collection)
    return obj


def safe_remove_object(obj: bpy.types.Object) -> None:
    """
    Safely delete an object: unlink it from all collections, then remove the object
    data block. No bpy.ops.

    Parameters
    ----------
    obj : bpy.types.Object
        Object to delete
    """
    # Unlink from all collections
    for col in list(obj.users_collection):
        try:
            col.objects.unlink(obj)
        except Exception:
            pass

    # Remove the associated data block (if any and it has no other users)
    data = obj.data
    bpy.data.objects.remove(obj, do_unlink=True)

    if data is not None and hasattr(data, 'users') and data.users == 0:
        block_list = None
        if isinstance(data, bpy.types.Mesh):
            block_list = bpy.data.meshes
        elif isinstance(data, bpy.types.Light):
            block_list = bpy.data.lights
        elif isinstance(data, bpy.types.Camera):
            block_list = bpy.data.cameras
        if block_list is not None:
            try:
                block_list.remove(data, do_unlink=True)
            except Exception:
                pass
