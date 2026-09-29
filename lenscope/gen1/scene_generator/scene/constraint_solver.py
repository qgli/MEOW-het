#!/usr/bin/env python3
"""
BVH-based collision detection and layout solver for furniture placement.

Built on mathutils.bvhtree.BVHTree; provides:
  1. ConstraintLayoutSolver: global BVH state + two-stage collision check with nudging
  2. Mixed geometry extraction for unique meshes and collection instances
  3. Archimedean spiral nudge: search over up to 100 candidate positions
  4. Wall alignment: quaternion rotation + depth offset to avoid interpenetration

Design rules:
  - No bpy.ops, no global random state, no destructive scaling
  - The global BVH is updated incrementally (commit -> rebuild)
  - Z is locked: the spiral nudge only moves in the XY plane
  - Collisions are never resolved by scaling (nudge only; delete if placement fails)
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import bpy
from mathutils import Euler, Matrix, Quaternion, Vector
from mathutils.bvhtree import BVHTree

from .core_utils import (
    calculate_inward_normal,
    distance_to_line_segments,
    distance_to_polygon_boundary,
    point_in_polygon,
)
from .geometry_factory import BMeshFactory


# ---- ConstraintLayoutSolver: global BVH collision solver ----

class ConstraintLayoutSolver:
    """
    Scene-level collision detection and layout solver based on BVHTree.

    State:
      - global_verts: list[Vector], vertex pool of the global collision space
      - global_polys: list[tuple[int,...]], polygon index pool of the global collision space
      - global_bvh: BVHTree built from these pools

    Workflow:
      1. Initialization: init_from_room_shell() resets the global collision space
      2. For each object to place: solve_placement() runs the two-stage check with nudging
      3. On success: commit_to_global_space() updates the global BVH
      4. On failure: the caller deletes the candidate object

    Implementation notes:
      - On commit, polygon indices are offset by the current length of global_verts
      - Collection instances are handled by iterating instance_collection.objects
      - The spiral nudge is an Archimedean spiral with an angular step of 1.618 rad
    """

    # Spiral nudge parameters
    SPIRAL_STEP: float = 0.05          # radius increment per attempt (m)
    SPIRAL_GOLDEN_ANGLE: float = 1.618  # angle increment per attempt (rad)
    MAX_NUDGE_ATTEMPTS: int = 100       # maximum number of attempts
    MIN_WALL_DISTANCE: float = 0.15     # minimum distance to walls (m)

    def __init__(self) -> None:
        """Start with an empty global collision space."""
        self.global_verts: list[Vector] = []
        self.global_polys: list[tuple[int, ...]] = []
        self.global_bvh: Optional[BVHTree] = None
        self._initialized: bool = False

    # ---- Global BVH management ----

    def _rebuild_bvh(self) -> None:
        """Rebuild the BVH from global_verts and global_polys."""
        if not self.global_polys:
            self.global_bvh = None
            return
        self.global_bvh = BVHTree.FromPolygons(
            self.global_verts, self.global_polys)

    def commit_to_global_space(
        self,
        new_verts: list[Vector],
        new_polys: list[tuple[int, ...]],
    ) -> None:
        """
        Merge new geometry into the global collision space and rebuild the BVH.

        The indices of the new polygons must be offset by the current length of
        global_verts; otherwise the BVH build indexes out of range.

        Parameters
        ----------
        new_verts : list[Vector]
            New vertices in world space
        new_polys : list[tuple[int, ...]]
            Polygon indices local to new_verts
        """
        offset = len(self.global_verts)

        # Append to the global vertex pool
        self.global_verts.extend(new_verts)

        # Append to the global polygon pool (with index offset)
        for poly in new_polys:
            self.global_polys.append(tuple(idx + offset for idx in poly))

        # Rebuild the BVH
        self._rebuild_bvh()

    def commit_object(self, obj: bpy.types.Object) -> None:
        """
        Extract the world-space geometry of an object and commit it.

        Handles both unique meshes and collection instances.

        Parameters
        ----------
        obj : bpy.types.Object
            Object to add to the global collision space
        """
        verts, polys = self._extract_world_geometry(obj)
        if verts and polys:
            self.commit_to_global_space(verts, polys)

    def init_from_room_shell(self, shell_obj: bpy.types.Object) -> None:
        """
        Initialize the solver for a room (resets the global collision space).

        The RoomShell itself is not added to the global BVH: furniture rests on the
        floor and against walls, so its polygons would report false overlaps with
        the floor and wall faces of the shell. Room boundaries are enforced
        separately by Step 1 of solve_placement
        (point_in_polygon + distance_to_boundary).

        The global BVH only tracks placed furniture, for furniture-furniture collisions.

        Parameters
        ----------
        shell_obj : bpy.types.Object
            RoomShell object built by RoomBuilder (only marks initialization as done)
        """
        # Clear old data; start tracking furniture collisions from scratch
        self.global_verts.clear()
        self.global_polys.clear()
        self.global_bvh = None
        self._initialized = True

    # ---- Mixed geometry extraction ----

    @staticmethod
    def _extract_world_geometry(
        obj: bpy.types.Object,
    ) -> tuple[list[Vector], list[tuple[int, ...]]]:
        """
        Extract the world-space geometry (vertices + polygon indices) of an object.

        Mixed extraction:
          - Unique mesh: obj.data vertices transformed by matrix_world
          - Collection instance (Empty): iterate over instance_collection.objects;
            final coordinate = Empty.matrix_world @ MeshObj.matrix_local @ vertex.co

        Parameters
        ----------
        obj : bpy.types.Object
            Target object (mesh or collection-instance Empty)

        Returns
        -------
        (verts, polys) : tuple[list[Vector], list[tuple[int, ...]]]
            World-space vertices and polygons indexing into them
        """
        all_verts: list[Vector] = []
        all_polys: list[tuple[int, ...]] = []

        if obj.type == 'MESH' and obj.data is not None:
            # Unique mesh: extract directly
            mesh = obj.data
            world_mat = obj.matrix_world
            base_idx = len(all_verts)
            for v in mesh.vertices:
                all_verts.append(world_mat @ v.co)
            for p in mesh.polygons:
                all_polys.append(tuple(vi + base_idx for vi in p.vertices))

        elif (obj.type == 'EMPTY'
              and obj.instance_type == 'COLLECTION'
              and obj.instance_collection is not None):
            # Collection instance: iterate over the mesh objects inside
            empty_world = obj.matrix_world
            for child_obj in obj.instance_collection.objects:
                if child_obj.type != 'MESH' or child_obj.data is None:
                    continue
                child_mesh = child_obj.data
                # Final transform = Empty.matrix_world @ MeshObj.matrix_local
                combined_mat = empty_world @ child_obj.matrix_local
                base_idx = len(all_verts)
                for v in child_mesh.vertices:
                    all_verts.append(combined_mat @ v.co)
                for p in child_mesh.polygons:
                    all_polys.append(
                        tuple(vi + base_idx for vi in p.vertices))

        return all_verts, all_polys

    # ---- AABB helpers ----

    @staticmethod
    def _compute_world_aabb(
        obj: bpy.types.Object,
    ) -> tuple[Vector, Vector]:
        """
        Compute the world-space axis-aligned bounding box (AABB) of an object.

        Parameters
        ----------
        obj : bpy.types.Object

        Returns
        -------
        (min_corner, max_corner) : tuple[Vector, Vector]
        """
        world_mat = obj.matrix_world
        # bound_box holds the 8 corners in local space
        corners = [world_mat @ Vector(c) for c in obj.bound_box]
        xs = [c.x for c in corners]
        ys = [c.y for c in corners]
        zs = [c.z for c in corners]
        return (
            Vector((min(xs), min(ys), min(zs))),
            Vector((max(xs), max(ys), max(zs))),
        )

    @staticmethod
    def _aabb_bottom_corners(
        aabb_min: Vector,
        aabb_max: Vector,
    ) -> list[tuple[float, float]]:
        """
        The 4 corners (2D) of the AABB bottom face (Z = min).

        Used with point_in_polygon to test whether the object lies fully inside the room.
        """
        return [
            (aabb_min.x, aabb_min.y),
            (aabb_max.x, aabb_min.y),
            (aabb_max.x, aabb_max.y),
            (aabb_min.x, aabb_max.y),
        ]

    # ---- Local BVH ----

    def _build_local_bvh(
        self,
        obj: bpy.types.Object,
    ) -> Optional[BVHTree]:
        """
        Build a local BVH (world space) for a candidate object.

        Parameters
        ----------
        obj : bpy.types.Object
            Candidate object

        Returns
        -------
        BVHTree or None
        """
        verts, polys = self._extract_world_geometry(obj)
        if not polys:
            return None
        return BVHTree.FromPolygons(verts, polys)

    # ---- Step 1: boundary check ----

    def _check_boundary(
        self,
        obj: bpy.types.Object,
        floor_verts: Sequence[tuple[float, float]],
        wall_margin: float,
        internal_walls: list = None,
    ) -> bool:
        """
        Step 1 boundary check: all 4 AABB corners lie inside the polygon and at least
        wall_margin away from its boundary and from the internal wall segments.

        Parameters
        ----------
        obj : bpy.types.Object
        floor_verts : Sequence[(float, float)]
        wall_margin : float
        internal_walls : list[((x1,y1),(x2,y2))], optional
            Internal wall segments

        Returns
        -------
        bool : True if the check passes
        """
        aabb_min, aabb_max = self._compute_world_aabb(obj)
        corners = self._aabb_bottom_corners(aabb_min, aabb_max)

        for cx, cy in corners:
            if not point_in_polygon(cx, cy, floor_verts):
                return False
            if distance_to_polygon_boundary(cx, cy, floor_verts) < wall_margin:
                return False
            # Distance to internal walls
            if internal_walls:
                if distance_to_line_segments(cx, cy, internal_walls) < wall_margin:
                    return False
        return True

    # ---- Step 2: BVH collision check ----

    def _check_bvh_collision(
        self,
        obj: bpy.types.Object,
    ) -> bool:
        """
        Step 2 collision check: overlap of the local BVH with the global BVH.

        Parameters
        ----------
        obj : bpy.types.Object
            Candidate object

        Returns
        -------
        bool : True if there is a collision (check fails)
        """
        if self.global_bvh is None:
            return False  # no global geometry: no collision

        local_bvh = self._build_local_bvh(obj)
        if local_bvh is None:
            return False  # no local geometry: no collision

        overlap_pairs = self.global_bvh.overlap(local_bvh)
        return len(overlap_pairs) > 0

    # ---- Step 3: Archimedean spiral nudge ----

    def solve_placement(
        self,
        obj: bpy.types.Object,
        base_loc: Vector,
        rotation_euler: Euler,
        floor_verts: Sequence[tuple[float, float]],
        wall_margin: Optional[float] = None,
        max_attempts: Optional[int] = None,
        internal_walls: list = None,
    ) -> bool:
        """
        Two-stage collision check with Archimedean spiral nudging.

        The boundary check also enforces the distance to internal walls.

        Parameters
        ----------
        obj : bpy.types.Object
        base_loc : Vector
        rotation_euler : Euler
        floor_verts : Sequence[(float, float)]
        wall_margin : float, optional
        max_attempts : int, optional
        internal_walls : list[((x1,y1),(x2,y2))], optional
            Internal wall segments

        Returns
        -------
        bool
        """
        if wall_margin is None:
            wall_margin = self.MIN_WALL_DISTANCE
        if max_attempts is None:
            max_attempts = self.MAX_NUDGE_ATTEMPTS

        # Lock Z
        z_fixed = base_loc.z

        # Set the initial rotation
        obj.rotation_euler = rotation_euler

        for attempt in range(max_attempts):
            if attempt == 0:
                # First attempt: use base_loc as is
                candidate_x = base_loc.x
                candidate_y = base_loc.y
            else:
                # Archimedean spiral nudge
                # r = step * iteration, theta = iteration * golden_angle
                r = self.SPIRAL_STEP * attempt
                theta = attempt * self.SPIRAL_GOLDEN_ANGLE
                candidate_x = base_loc.x + r * math.cos(theta)
                candidate_y = base_loc.y + r * math.sin(theta)

            # Set the candidate location (Z locked)
            obj.location = Vector((candidate_x, candidate_y, z_fixed))

            # Update the view layer so matrix_world is refreshed
            bpy.context.view_layer.update()

            # Step 1: boundary check (including internal walls)
            if not self._check_boundary(
                    obj, floor_verts, wall_margin, internal_walls):
                continue

            # Step 2: BVH collision check
            if self._check_bvh_collision(obj):
                continue

            # Passed: commit to the global space
            self.commit_object(obj)
            return True

        # All candidates exhausted: placement failed
        return False

    # ---- Align to wall ----

    def align_to_wall(
        self,
        obj: bpy.types.Object,
        wall_info: dict,
        depth_along_y: Optional[float] = None,
    ) -> tuple[Vector, Euler]:
        """
        Align an object to a wall: rotate so that its local -Y faces the wall, then
        offset it by half its depth to avoid interpenetration.

        Assumptions:
          - The back of the furniture faces local -Y
          - wall_info comes from RoomBuilder (e.g. RoomBuilder.get_nearest_wall())

        Algorithm:
          1. Take the inward wall normal N_in
          2. The back direction of the furniture is -Y = (0, -1, 0) in local space
          3. Rotate -Y onto -N_in (back against the wall),
             so that the front faces N_in (into the room)
          4. Compute the rotation with the quaternion rotation_difference
          5. Depth offset: translate by depth/2 along N_in (avoids interpenetration)

        Parameters
        ----------
        obj : bpy.types.Object
            Object to align
        wall_info : dict
            Wall entry (from RoomBuilder.wall_info);
            must contain 'midpoint' and 'normal_inward'
        depth_along_y : float, optional
            Depth of the object along Y. Computed from bound_box if not given.

        Returns
        -------
        (location, rotation) : tuple[Vector, Euler]
            Aligned world location and Euler rotation
        """
        nx, ny = wall_info['normal_inward']
        v1 = wall_info['v1']
        v2 = wall_info['v2']

        # Depth of the object along Y (from bound_box)
        if depth_along_y is None:
            bb = obj.bound_box
            # bound_box is in local space; take the Y range
            y_coords = [bb[i][1] for i in range(8)]
            depth_along_y = max(y_coords) - min(y_coords)

        # Rotation
        # Back direction of the furniture = local -Y = (0, -1, 0)
        furniture_back = Vector((0, -1, 0))
        # Outward wall normal = -N_in (the back points out through the wall)
        wall_outward = Vector((-nx, -ny, 0)).normalized()

        # Quaternion rotation from furniture_back to wall_outward
        quat = furniture_back.rotation_difference(wall_outward)
        rotation_euler = quat.to_euler()

        # Location: wall midpoint + N_in * depth/2
        wall_mx, wall_my = wall_info['midpoint']

        # Offset depth/2 into the room along the normal
        offset_dist = depth_along_y / 2.0
        loc_x = wall_mx + nx * offset_dist
        loc_y = wall_my + ny * offset_dist
        loc_z = 0.0  # floor

        location = Vector((loc_x, loc_y, loc_z))

        return location, rotation_euler

    def align_and_solve(
        self,
        obj: bpy.types.Object,
        wall_info: dict,
        floor_verts: Sequence[tuple[float, float]],
        depth_along_y: Optional[float] = None,
        wall_margin: Optional[float] = None,
        internal_walls: list = None,
    ) -> bool:
        """
        Wall alignment followed by collision solving.

        1. Compute the wall-aligned pose
        2. Run solve_placement with spiral nudging (respecting internal_walls)

        Parameters
        ----------
        obj : bpy.types.Object
        wall_info : dict
        floor_verts : Sequence[(float, float)]
        depth_along_y : float, optional
        wall_margin : float, optional
        internal_walls : list, optional

        Returns
        -------
        bool
        """
        location, rotation = self.align_to_wall(
            obj, wall_info, depth_along_y)

        return self.solve_placement(
            obj=obj,
            base_loc=location,
            rotation_euler=rotation,
            floor_verts=floor_verts,
            wall_margin=wall_margin,
            internal_walls=internal_walls,
        )

    # ---- Queries ----

    @property
    def num_global_verts(self) -> int:
        """Total number of vertices in the global collision space."""
        return len(self.global_verts)

    @property
    def num_global_polys(self) -> int:
        """Total number of polygons in the global collision space."""
        return len(self.global_polys)

    def has_bvh(self) -> bool:
        """True once initialized (there may be no furniture colliders yet)."""
        return self._initialized or self.global_bvh is not None

    def reset(self) -> None:
        """Clear the global collision space."""
        self.global_verts.clear()
        self.global_polys.clear()
        self.global_bvh = None
        self._initialized = False
