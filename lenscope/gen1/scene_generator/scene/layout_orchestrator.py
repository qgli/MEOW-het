#!/usr/bin/env python3
"""
Furniture layout of a room.

RoomDecorator combines RoomBuilder, ConstraintLayoutSolver,
ProceduralFurnitureBuilder and InstanceManager to furnish a room.

Features:
  1. Template pre-registration (chair/decor/rug -> InstanceManager)
  2. populate_furniture: places each furniture category in turn
  3. Wall-aligned placement (bookshelf/bed -> align_and_solve)
  4. Table-and-chair sets: chairs on a circle around each table (polar coordinates)
  5. Rugs are exempt from collisions (not committed to the global BVH)
  6. Failed placements are removed (safe_remove_object)

Rules:
  - No bpy.ops, no global random state, no destructive scaling
  - A unique mesh goes through create_data_object + view_layer.update before it is passed to the solver
  - A placed rug is never committed with commit_to_global_space
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import bpy
from mathutils import Euler, Matrix, Quaternion, Vector

from .asset_manager import InstanceManager
from .constraint_solver import ConstraintLayoutSolver
from .core_utils import (
    create_data_object,
    link_object_to_scene,
    polygon_signed_area,
    random_interior_point,
    safe_remove_object,
)
from .furniture_builder import ProceduralFurnitureBuilder
from .geometry_factory import ProceduralMaterialFactory
from .room_builder import RoomBuilder


# ---- RoomDecorator: furniture layout ----

class RoomDecorator:
    """
    Furniture layout scheduler for a room.

    Workflow:
      1. __init__: pre-register template items (chair/decor/rug)
      2. populate_furniture: try to place each furniture category
      3. Large unique furniture: build -> create_data_object -> assign material
         -> view_layer.update -> solve_placement / align_and_solve
      4. Template items: spawn_instance -> view_layer.update
         -> solve_placement
      5. Table-and-chair sets: chairs on a circle around the table
      6. Rugs: exempt from collisions (not committed)
    """

    # Sofa, table, decor and rug counts scale with room_area; only the bed and bookshelf ranges are used
    DEFAULT_FURNITURE_CONFIG: dict = {
        'table': {'min': 1, 'max': 2, 'wall_align': False},
        'sofa': {'min': 0, 'max': 2, 'wall_align': False},
        'bookshelf': {'min': 0, 'max': 2, 'wall_align': True},
        'bed': {'min': 0, 'max': 1, 'wall_align': True},
        'decor': {'min': 0, 'max': 4, 'wall_align': False},
        'rug': {'min': 0, 'max': 2, 'wall_align': False},
    }

    def __init__(
        self,
        rng,  # random.Random
        room_builder: RoomBuilder,
        solver: ConstraintLayoutSolver,
        furniture_collection: Optional[bpy.types.Collection] = None,
    ) -> None:
        """
        Parameters
        ----------
        rng : random.Random
        room_builder : RoomBuilder
        solver : ConstraintLayoutSolver
        furniture_collection : bpy.types.Collection, optional
        """
        self.rng = rng
        self.room_builder = room_builder
        self.solver = solver
        self.floor_verts = room_builder.floor_verts
        self.internal_walls = getattr(room_builder, 'internal_walls', []) or []
        self.sub_rooms = getattr(room_builder, 'sub_rooms', None) or [list(room_builder.floor_verts)]

        # Total room area (drives the furniture density)
        self.room_area = abs(polygon_signed_area(self.floor_verts))

        # Dedicated furniture collection
        if furniture_collection is None:
            self._furniture_col = bpy.data.collections.new("Furniture")
            bpy.context.scene.collection.children.link(self._furniture_col)
        else:
            self._furniture_col = furniture_collection

        # Instance manager
        self.instance_mgr = InstanceManager()

        # Furniture builder
        self.builder = ProceduralFurnitureBuilder()

        # Placed objects
        self.placed_objects: list[bpy.types.Object] = []
        # Rug objects (tracked separately; exempt from collisions)
        self.rug_objects: list[bpy.types.Object] = []

        # Pre-register the templates
        self._register_templates()

    def _register_templates(self) -> None:
        """Pre-register template items (chair/decor/rug) as large pools of variants."""

        # Chair template pool (15 variants)
        _chair_mat_types = ['wood', 'metal', 'principled', 'fabric']
        for i in range(15):
            chair_mesh = self.builder.build_simple_chair(self.rng)
            cmt = self.rng.choice(_chair_mat_types)
            if cmt == 'wood':
                chair_mat = ProceduralMaterialFactory.make_wood(
                    f"ChairMat_{i:02d}", self.rng)
            elif cmt == 'metal':
                chair_mat = ProceduralMaterialFactory.make_metal(
                    f"ChairMat_{i:02d}", self.rng)
            elif cmt == 'fabric':
                chair_mat = ProceduralMaterialFactory.make_fabric(
                    f"ChairMat_{i:02d}", self.rng)
            else:
                chair_mat = ProceduralMaterialFactory.make_principled(
                    f"ChairMat_{i:02d}",
                    ProceduralMaterialFactory.random_color(self.rng),
                    roughness=self.rng.uniform(0.2, 0.8),
                    rng=self.rng)
            self.instance_mgr.register_template(
                f"chair_{i:02d}", chair_mesh, chair_mat)

        # Decor template pool (30 variants)
        for i in range(30):
            decor_mesh = self.builder.build_vase_or_decor(self.rng)
            decor_mat = ProceduralMaterialFactory.random_material(
                f"DecorMat_{i:02d}", self.rng)
            self.instance_mgr.register_template(
                f"decor_{i:02d}", decor_mesh, decor_mat)

        # Rug template pool (10 variants)
        for i in range(10):
            rug_mesh = self.builder.build_rug(self.rng)
            rug_mat = ProceduralMaterialFactory.make_fabric(
                f"RugFabric_{i:02d}", self.rng)
            self.instance_mgr.register_template(
                f"rug_{i:02d}", rug_mesh, rug_mat)

    # ---- Unique furniture placement ----

    def _pick_sub_room_point(self, margin: float = 0.5) -> tuple[float, float]:
        """Sample an interior point of a random sub-room."""
        room_poly = self.rng.choice(self.sub_rooms)
        return random_interior_point(
            self.rng, room_poly, margin=margin,
            internal_walls=self.internal_walls)

    def _place_unique_furniture(
        self,
        mesh: bpy.types.Mesh,
        name: str,
        material: bpy.types.Material,
        wall_align: bool,
    ) -> Optional[bpy.types.Object]:
        """
        Place a unique (non-template) furniture piece.

        Steps:
          1. Wrap with create_data_object and link to the scene
          2. Assign the material
          3. view_layer.update to refresh the depsgraph
          4. wall_align -> align_and_solve, otherwise -> solve_placement
          5. On failure -> safe_remove_object

        Parameters
        ----------
        mesh : bpy.types.Mesh
            Mesh built by BMeshFactory
        name : str
            Object name
        material : bpy.types.Material
            Material to assign
        wall_align : bool
            Whether to align the piece to a wall

        Returns
        -------
        bpy.types.Object or None
            The object on success, None on failure
        """
        # Wrap in an Object and link it
        obj = create_data_object(name, mesh, self._furniture_col)
        obj.data.materials.append(material)

        # The depsgraph must be refreshed before the solver reads the object
        bpy.context.view_layer.update()

        success = False

        if wall_align:
            walls = self.room_builder.wall_info
            if walls:
                wall = self.rng.choice(walls)
                success = self.solver.align_and_solve(
                    obj, wall, self.floor_verts,
                    internal_walls=self.internal_walls)
        else:
            # Sample the position inside a sub-room
            px, py = self._pick_sub_room_point(margin=0.5)
            rot = Euler((0, 0, self.rng.uniform(0, 2 * math.pi)))
            success = self.solver.solve_placement(
                obj, Vector((px, py, 0)), rot, self.floor_verts,
                internal_walls=self.internal_walls)

        if success:
            self.placed_objects.append(obj)
            return obj
        else:
            safe_remove_object(obj)
            return None

    # ---- Table-and-chair sets ----

    def _place_chairs_around_table(
        self,
        table_obj: bpy.types.Object,
        n_chairs: int,
    ) -> list[bpy.types.Object]:
        """
        Place chairs on a circle around a table (polar coordinates).

        Algorithm:
          1. Table AABB -> half extents in X/Y
          2. Chair template bound_box -> depth along Y (front to back)
          3. Circle radius = table radius + chair depth / 2 + 0.1 m gap
          4. Equal angular spacing; a quaternion turns each chair towards the table center
          5. Each chair goes through solve_placement

        Parameters
        ----------
        table_obj : bpy.types.Object
            Placed table object
        n_chairs : int
            Number of chairs (2-4)

        Returns
        -------
        list[bpy.types.Object]
            Chair Empties that were placed
        """
        placed_chairs: list[bpy.types.Object] = []

        # Table AABB in world space
        table_aabb_min, table_aabb_max = \
            ConstraintLayoutSolver._compute_world_aabb(table_obj)
        table_cx = (table_aabb_min.x + table_aabb_max.x) / 2
        table_cy = (table_aabb_min.y + table_aabb_max.y) / 2
        table_rx = (table_aabb_max.x - table_aabb_min.x) / 2
        table_ry = (table_aabb_max.y - table_aabb_min.y) / 2
        table_radius = max(table_rx, table_ry)

        # All chairs of one table use the same template (a matching set)
        chair_template_key = f"chair_{self.rng.randint(0, 14):02d}"

        # Chair depth along Y from the template bound_box
        chair_tpl = self.instance_mgr.get_template_object(chair_template_key)
        bb = chair_tpl.bound_box
        chair_y_depth = max(Vector(c).y for c in bb) - \
            min(Vector(c).y for c in bb)
        chair_gap = 0.1

        orbit_radius = table_radius + chair_y_depth / 2 + chair_gap

        # Equal angular spacing with a random phase
        phase = self.rng.uniform(0, 2 * math.pi)

        for i in range(n_chairs):
            angle = phase + (2 * math.pi * i / n_chairs)
            cx = table_cx + orbit_radius * math.cos(angle)
            cy = table_cy + orbit_radius * math.sin(angle)

            # Quaternion: the chair front (+Y) faces the table center,
            # i.e. local +Y is aligned with (table_center - chair_pos)
            dir_to_table = Vector((table_cx - cx, table_cy - cy, 0))
            if dir_to_table.length > 1e-6:
                dir_to_table.normalize()
                # Rotate (0, 1, 0) onto dir_to_table
                forward = Vector((0, 1, 0))
                quat = forward.rotation_difference(dir_to_table)
                rot_euler = quat.to_euler()
            else:
                rot_euler = Euler((0, 0, angle))

            # Transform matrix
            transform = Matrix.Translation(Vector((cx, cy, 0))) @ \
                quat.to_matrix().to_4x4()

            # Spawn the instance
            chair_empty = self.instance_mgr.spawn_instance(
                chair_template_key, transform, self._furniture_col,
                instance_name=f"Chair_{i}")

            # Refresh the depsgraph
            bpy.context.view_layer.update()

            # Collision check
            success = self.solver.solve_placement(
                chair_empty,
                Vector((cx, cy, 0)),
                rot_euler,
                self.floor_verts,
                internal_walls=self.internal_walls,
            )

            if success:
                placed_chairs.append(chair_empty)
                self.placed_objects.append(chair_empty)
            else:
                safe_remove_object(chair_empty)

        return placed_chairs

    # ---- Rug placement (exempt from collisions) ----

    def _place_rug(self) -> Optional[bpy.types.Object]:
        """Place a rug (collection instance); a placed rug is not committed to the BVH."""
        px, py = self._pick_sub_room_point(margin=0.3)

        rot_angle = self.rng.uniform(0, 2 * math.pi)
        rot_euler = Euler((0, 0, rot_angle))
        transform = Matrix.Translation(Vector((px, py, 0))) @ \
            Matrix.Rotation(rot_angle, 4, 'Z')

        rug_key = f"rug_{self.rng.randint(0, 9):02d}"
        rug_empty = self.instance_mgr.spawn_instance(
            rug_key, transform, self._furniture_col,
            instance_name="Rug")

        bpy.context.view_layer.update()

        # Boundary check only (no BVH collision check)
        from .core_utils import point_in_polygon, distance_to_polygon_boundary
        if (point_in_polygon(px, py, self.floor_verts) and
                distance_to_polygon_boundary(px, py, self.floor_verts) >= 0.2):
            # Placed, but not committed
            self.placed_objects.append(rug_empty)
            self.rug_objects.append(rug_empty)
            return rug_empty
        else:
            safe_remove_object(rug_empty)
            return None

    # ---- Decor placement (template instance) ----

    def _place_decor(self) -> Optional[bpy.types.Object]:
        """Place a decor item (collection instance)."""
        px, py = self._pick_sub_room_point(margin=0.3)
        rot_angle = self.rng.uniform(0, 2 * math.pi)
        rot_euler = Euler((0, 0, rot_angle))
        transform = Matrix.Translation(Vector((px, py, 0))) @ \
            Matrix.Rotation(rot_angle, 4, 'Z')

        decor_key = f"decor_{self.rng.randint(0, 29):02d}"
        decor_empty = self.instance_mgr.spawn_instance(
            decor_key, transform, self._furniture_col,
            instance_name="Decor")

        bpy.context.view_layer.update()

        success = self.solver.solve_placement(
            decor_empty,
            Vector((px, py, 0)),
            rot_euler,
            self.floor_verts,
            internal_walls=self.internal_walls,
        )

        if success:
            self.placed_objects.append(decor_empty)
            return decor_empty
        else:
            safe_remove_object(decor_empty)
            return None

    # ---- Main entry: populate_furniture ----

    def populate_furniture(
        self,
        rng=None,
        n_objects: Optional[int] = None,
        config: Optional[dict] = None,
    ) -> dict:
        """
        Furnish the room densely; furniture counts are derived from room_area.
        """
        if rng is None:
            rng = self.rng
        if config is None:
            config = self.DEFAULT_FURNITURE_CONFIG

        stats: dict[str, int] = {}
        all_chairs: list[bpy.types.Object] = []
        area = self.room_area

        # Step 1: large unique furniture

        # Bed (against a wall)
        n_bed = rng.randint(
            config.get('bed', {}).get('min', 0),
            config.get('bed', {}).get('max', 1))
        bed_count = 0
        for _ in range(n_bed):
            mesh = self.builder.build_bed(rng)
            mat = ProceduralMaterialFactory.random_material("BedMat", rng)
            obj = self._place_unique_furniture(
                mesh, "Bed", mat, wall_align=True)
            if obj is not None:
                bed_count += 1
        stats['bed'] = bed_count

        # Bookshelf (against a wall)
        n_shelf = rng.randint(
            config.get('bookshelf', {}).get('min', 0),
            config.get('bookshelf', {}).get('max', 2))
        shelf_count = 0
        for _ in range(n_shelf):
            mesh = self.builder.build_bookshelf(rng)
            mat = ProceduralMaterialFactory.random_material("ShelfMat", rng)
            obj = self._place_unique_furniture(
                mesh, "Bookshelf", mat, wall_align=True)
            if obj is not None:
                shelf_count += 1
        stats['bookshelf'] = shelf_count

        # Sofa (count from area)
        n_sofa = int(area / 20.0)
        sofa_count = 0
        for _ in range(n_sofa):
            mesh = self.builder.build_sofa(rng)
            mat = ProceduralMaterialFactory.make_fabric("SofaFabric", rng)
            obj = self._place_unique_furniture(
                mesh, "Sofa", mat, wall_align=False)
            if obj is not None:
                sofa_count += 1
        stats['sofa'] = sofa_count

        # Step 2: tables with chair sets (count from area)
        n_table = max(1, int(area / 15.0))
        table_count = 0
        for _ in range(n_table):
            mesh = self.builder.build_table(rng)
            mat = ProceduralMaterialFactory.random_material("TableMat", rng)
            table_obj = self._place_unique_furniture(
                mesh, "Table", mat, wall_align=False)
            if table_obj is not None:
                table_count += 1
                n_ch = rng.randint(2, 4)
                chairs = self._place_chairs_around_table(table_obj, n_ch)
                all_chairs.extend(chairs)
        stats['table'] = table_count
        stats['chair'] = len(all_chairs)

        # Step 3: decor (count from area)
        n_decor = int(area / 8.0)
        decor_count = 0
        for _ in range(n_decor):
            obj = self._place_decor()
            if obj is not None:
                decor_count += 1
        stats['decor'] = decor_count

        # Step 4: rugs (count from area)
        n_rug = int(area / 15.0)
        rug_count = 0
        for _ in range(n_rug):
            obj = self._place_rug()
            if obj is not None:
                rug_count += 1
        stats['rug'] = rug_count

        return {
            'placed': list(self.placed_objects),
            'rugs': list(self.rug_objects),
            'chairs': all_chairs,
            'stats': stats,
        }

    @property
    def furniture_collection(self) -> bpy.types.Collection:
        """Furniture collection."""
        return self._furniture_col
