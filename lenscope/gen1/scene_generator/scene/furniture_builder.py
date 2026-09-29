#!/usr/bin/env python3
"""
Procedural furniture for the scene generator.

Assembles 7 kinds of procedural furniture from BMeshFactory static methods + merge_meshes:
  Unique furniture (returns a Mesh data block):
    1. build_table     - table top + 4 cylindrical legs
    2. build_sofa      - base + backrest + two armrests (fabric material)
    3. build_bookshelf - side panels + shelves + back panel (back panel on -Y)
    4. build_bed       - frame + mattress + headboard
  Template items (return a Mesh for InstanceManager registration):
    5. build_simple_chair - seat + 4 legs + backrest
    6. build_vase_or_decor - random small decor item
    7. build_rug - thin rug that is not degenerate (0.005 m thick)

Rules:
  - No bpy.ops, no global random state, no destructive scaling
  - Local part matrices are composed as mat = Matrix.Translation(vec) @ Matrix.Diagonal(scale)
  - After merge_meshes, all intermediate meshes are removed with bpy.data.meshes.remove
  - Z_min = 0 for every piece (guaranteed by BMeshFactory)
  - The back of the furniture faces local -Y (used by align_to_wall)
"""

from __future__ import annotations

import math
from typing import Optional

import bpy
from mathutils import Matrix, Vector

from .geometry_factory import BMeshFactory, ProceduralMaterialFactory


# ---- ProceduralFurnitureBuilder ----

class ProceduralFurnitureBuilder:
    """
    Procedural furniture builder based on BMeshFactory primitives + merge_meshes.

    All build_* methods return a bpy.types.Mesh data block (not wrapped in an Object).
    The caller wraps it with create_data_object() and links it to the scene.

    Conventions:
      1. Each part bakes its local position/size into the vertices via the transform argument
      2. After merge_meshes all intermediate meshes are removed (prevents memory leaks)
      3. The back of the furniture faces local -Y (for wall alignment)
      4. Z_min = 0 (bottom-aligned)
    """

    @staticmethod
    def _merge_and_cleanup(
        name: str,
        parts: list[bpy.types.Mesh],
    ) -> bpy.types.Mesh:
        """
        Merge the parts and immediately remove the intermediate meshes (prevents orphan-part memory leaks).

        Parameters
        ----------
        name : str
            Name of the output mesh
        parts : list[bpy.types.Mesh]
            Part meshes to merge

        Returns
        -------
        bpy.types.Mesh
            Merged mesh
        """
        merged = BMeshFactory.merge_meshes(name, parts)

        # Remove all intermediate part meshes right away; otherwise they leak memory
        for m in parts:
            try:
                bpy.data.meshes.remove(m, do_unlink=True)
            except Exception:
                pass

        return merged

    # ---- 1. Table ----

    @staticmethod
    def build_table(rng) -> bpy.types.Mesh:
        """
        Build a table: top + 4 cylindrical legs.

        Size ranges:
          top: 0.6-1.5 m x 0.4-1.0 m, 0.03-0.06 m thick
          legs: radius 0.015-0.03 m, height = table height - top thickness

        Parameters
        ----------
        rng : random.Random
            Explicit random number generator

        Returns
        -------
        bpy.types.Mesh
        """
        # Random dimensions
        table_w = rng.uniform(0.6, 1.5)   # X
        table_d = rng.uniform(0.4, 1.0)   # Y
        table_h = rng.uniform(0.65, 0.85)  # total height
        top_thick = rng.uniform(0.03, 0.06)
        leg_r = rng.uniform(0.015, 0.03)
        leg_h = table_h - top_thick

        parts: list[bpy.types.Mesh] = []

        # Top at Z = leg_h
        top_mat = Matrix.Translation(Vector((0, 0, leg_h))) @ \
            Matrix.Diagonal(Vector((1, 1, 1, 1)))
        parts.append(BMeshFactory.create_cube(
            "part_table_top",
            size_x=table_w, size_y=table_d, size_z=top_thick,
            transform=top_mat,
        ))

        # 4 legs
        inset_x = table_w / 2 - leg_r - 0.02
        inset_y = table_d / 2 - leg_r - 0.02
        for sx, sy in [(-1, -1), (1, -1), (1, 1), (-1, 1)]:
            leg_mat = Matrix.Translation(Vector((
                sx * inset_x, sy * inset_y, 0)))
            parts.append(BMeshFactory.create_cylinder(
                "part_table_leg",
                radius=leg_r, depth=leg_h,
                segments=12,
                transform=leg_mat,
                smooth_shading=True,
            ))

        return ProceduralFurnitureBuilder._merge_and_cleanup("Table", parts)

    # ---- 2. Sofa ----

    @staticmethod
    def build_sofa(rng) -> bpy.types.Mesh:
        """
        Build a sofa: base + backrest + two armrests.

        The backrest is on the local -Y side (back against the wall).

        Parameters
        ----------
        rng : random.Random

        Returns
        -------
        bpy.types.Mesh
        """
        sofa_w = rng.uniform(1.2, 2.5)    # X
        sofa_d = rng.uniform(0.6, 1.0)    # Y
        seat_h = rng.uniform(0.35, 0.50)  # seat height
        back_h = rng.uniform(0.30, 0.50)  # backrest height (above the seat)
        arm_w = rng.uniform(0.08, 0.15)   # armrest width
        back_thick = rng.uniform(0.08, 0.15)

        parts: list[bpy.types.Mesh] = []

        # Base (seat)
        parts.append(BMeshFactory.create_cube(
            "part_sofa_seat",
            size_x=sofa_w, size_y=sofa_d, size_z=seat_h,
        ))

        # Backrest (local -Y side, above the seat)
        back_mat = Matrix.Translation(Vector((
            0, -(sofa_d / 2 - back_thick / 2), seat_h)))
        parts.append(BMeshFactory.create_cube(
            "part_sofa_back",
            size_x=sofa_w, size_y=back_thick, size_z=back_h,
            transform=back_mat,
        ))

        # Left armrest
        arm_h = seat_h + back_h * 0.6
        left_mat = Matrix.Translation(Vector((
            -(sofa_w / 2 - arm_w / 2), 0, 0)))
        parts.append(BMeshFactory.create_cube(
            "part_sofa_arm_L",
            size_x=arm_w, size_y=sofa_d, size_z=arm_h,
            transform=left_mat,
        ))

        # Right armrest
        right_mat = Matrix.Translation(Vector((
            sofa_w / 2 - arm_w / 2, 0, 0)))
        parts.append(BMeshFactory.create_cube(
            "part_sofa_arm_R",
            size_x=arm_w, size_y=sofa_d, size_z=arm_h,
            transform=right_mat,
        ))

        return ProceduralFurnitureBuilder._merge_and_cleanup("Sofa", parts)

    # ---- 3. Bookshelf ----

    @staticmethod
    def build_bookshelf(rng) -> bpy.types.Mesh:
        """
        Build a bookshelf: two side panels + several shelves + back panel (on the -Y face).

        Parameters
        ----------
        rng : random.Random

        Returns
        -------
        bpy.types.Mesh
        """
        shelf_w = rng.uniform(0.6, 1.2)    # X
        shelf_d = rng.uniform(0.25, 0.40)  # Y
        shelf_h = rng.uniform(1.2, 2.0)    # total height (Z)
        panel_thick = rng.uniform(0.015, 0.025)
        n_shelves = rng.randint(3, 6)

        parts: list[bpy.types.Mesh] = []

        # Left side panel
        left_mat = Matrix.Translation(Vector((
            -(shelf_w / 2 - panel_thick / 2), 0, 0)))
        parts.append(BMeshFactory.create_cube(
            "part_shelf_side_L",
            size_x=panel_thick, size_y=shelf_d, size_z=shelf_h,
            transform=left_mat,
        ))

        # Right side panel
        right_mat = Matrix.Translation(Vector((
            shelf_w / 2 - panel_thick / 2, 0, 0)))
        parts.append(BMeshFactory.create_cube(
            "part_shelf_side_R",
            size_x=panel_thick, size_y=shelf_d, size_z=shelf_h,
            transform=right_mat,
        ))

        # Shelves (evenly spaced)
        inner_w = shelf_w - 2 * panel_thick
        for i in range(n_shelves + 1):
            z = (shelf_h / (n_shelves)) * i
            # The bottom and top boards are shelves too
            s_mat = Matrix.Translation(Vector((0, 0, z)))
            parts.append(BMeshFactory.create_cube(
                f"part_shelf_board_{i}",
                size_x=inner_w, size_y=shelf_d, size_z=panel_thick,
                transform=s_mat,
            ))

        # Back panel (local -Y face)
        back_mat = Matrix.Translation(Vector((
            0, -(shelf_d / 2 - panel_thick / 2), 0)))
        parts.append(BMeshFactory.create_cube(
            "part_shelf_back",
            size_x=inner_w, size_y=panel_thick, size_z=shelf_h,
            transform=back_mat,
        ))

        return ProceduralFurnitureBuilder._merge_and_cleanup(
            "Bookshelf", parts)

    # ---- 4. Bed ----

    @staticmethod
    def build_bed(rng) -> bpy.types.Mesh:
        """
        Build a bed: frame + slightly smaller mattress + headboard.

        The headboard is on the local -Y side (against the wall).

        Parameters
        ----------
        rng : random.Random

        Returns
        -------
        bpy.types.Mesh
        """
        bed_w = rng.uniform(1.2, 2.0)     # X
        bed_d = rng.uniform(1.8, 2.2)     # Y
        frame_h = rng.uniform(0.20, 0.35)  # frame height
        mattress_h = rng.uniform(0.15, 0.25)
        headboard_h = rng.uniform(0.5, 0.9)
        headboard_thick = rng.uniform(0.04, 0.08)

        parts: list[bpy.types.Mesh] = []

        # Frame
        parts.append(BMeshFactory.create_cube(
            "part_bed_frame",
            size_x=bed_w, size_y=bed_d, size_z=frame_h,
        ))

        # Mattress (slightly smaller than the frame)
        m_inset = 0.03
        m_mat = Matrix.Translation(Vector((0, 0, frame_h)))
        parts.append(BMeshFactory.create_cube(
            "part_bed_mattress",
            size_x=bed_w - 2 * m_inset,
            size_y=bed_d - 2 * m_inset,
            size_z=mattress_h,
            transform=m_mat,
        ))

        # Headboard (-Y side, from the floor up to frame + mattress + headboard height)
        hb_total_h = frame_h + mattress_h + headboard_h
        hb_mat = Matrix.Translation(Vector((
            0, -(bed_d / 2 - headboard_thick / 2), 0)))
        parts.append(BMeshFactory.create_cube(
            "part_bed_headboard",
            size_x=bed_w, size_y=headboard_thick, size_z=hb_total_h,
            transform=hb_mat,
        ))

        return ProceduralFurnitureBuilder._merge_and_cleanup("Bed", parts)

    # ---- 5. Simple chair (template item) ----

    @staticmethod
    def build_simple_chair(rng) -> bpy.types.Mesh:
        """
        Build a simple chair: seat + 4 legs + backrest.

        Suitable for template registration (reused many times).
        The backrest is on the local -Y side.

        Parameters
        ----------
        rng : random.Random

        Returns
        -------
        bpy.types.Mesh
        """
        seat_w = rng.uniform(0.35, 0.50)   # X
        seat_d = rng.uniform(0.35, 0.45)   # Y
        seat_h = rng.uniform(0.40, 0.50)   # seat height above the floor (leg length)
        seat_thick = rng.uniform(0.03, 0.05)
        leg_r = rng.uniform(0.012, 0.020)
        back_h = rng.uniform(0.30, 0.50)
        back_thick = rng.uniform(0.02, 0.04)

        parts: list[bpy.types.Mesh] = []

        # Seat
        seat_mat = Matrix.Translation(Vector((0, 0, seat_h)))
        parts.append(BMeshFactory.create_cube(
            "part_chair_seat",
            size_x=seat_w, size_y=seat_d, size_z=seat_thick,
            transform=seat_mat,
        ))

        # 4 legs
        inset_x = seat_w / 2 - leg_r - 0.01
        inset_y = seat_d / 2 - leg_r - 0.01
        for sx, sy in [(-1, -1), (1, -1), (1, 1), (-1, 1)]:
            leg_mat = Matrix.Translation(Vector((
                sx * inset_x, sy * inset_y, 0)))
            parts.append(BMeshFactory.create_cylinder(
                "part_chair_leg",
                radius=leg_r, depth=seat_h,
                segments=8,
                transform=leg_mat,
            ))

        # Backrest (-Y side, rising from the top of the seat)
        back_z = seat_h + seat_thick
        back_mat = Matrix.Translation(Vector((
            0, -(seat_d / 2 - back_thick / 2), back_z)))
        parts.append(BMeshFactory.create_cube(
            "part_chair_back",
            size_x=seat_w, size_y=back_thick, size_z=back_h,
            transform=back_mat,
        ))

        return ProceduralFurnitureBuilder._merge_and_cleanup(
            "SimpleChair", parts)

    # ---- 6. Vase / decor (template item) ----

    @staticmethod
    def build_vase_or_decor(rng) -> bpy.types.Mesh:
        """
        Build a random small decor item (vase, sphere or cone).

        Parameters
        ----------
        rng : random.Random

        Returns
        -------
        bpy.types.Mesh
        """
        choice = rng.choice(['vase', 'sphere_decor', 'cone_decor'])

        if choice == 'vase':
            # Vase: truncated cone
            r_bottom = rng.uniform(0.04, 0.08)
            r_top = rng.uniform(0.06, 0.12)
            h = rng.uniform(0.15, 0.35)
            return BMeshFactory.create_cone(
                "VaseDecor",
                radius_bottom=r_bottom,
                radius_top=r_top,
                depth=h,
                segments=16,
                smooth_shading=True,
            )
        elif choice == 'sphere_decor':
            # Sphere
            r = rng.uniform(0.05, 0.15)
            return BMeshFactory.create_sphere(
                "SphereDecor",
                radius=r,
                u_segments=16,
                v_segments=8,
                smooth_shading=True,
            )
        else:
            # Cone
            r = rng.uniform(0.05, 0.10)
            h = rng.uniform(0.10, 0.25)
            return BMeshFactory.create_cone(
                "ConeDecor",
                radius_bottom=r,
                radius_top=0.0,
                depth=h,
                segments=12,
                smooth_shading=True,
            )

    # ---- 7. Rug (thin but not degenerate) ----

    @staticmethod
    def build_rug(rng) -> bpy.types.Mesh:
        """
        Build a rug: a thin box only 0.005 m thick.

        Non-degenerate rug:
          - Fixed thickness of 0.005 m, so BVH checks do not treat it as degenerate
          - Size range: 1.0-3.0 m x 0.8-2.5 m
          - Z_min = 0 (on the floor)

        Parameters
        ----------
        rng : random.Random

        Returns
        -------
        bpy.types.Mesh
        """
        rug_w = rng.uniform(1.0, 3.0)
        rug_d = rng.uniform(0.8, 2.5)
        rug_h = 0.005  # fixed, very thin

        return BMeshFactory.create_cube(
            "Rug",
            size_x=rug_w, size_y=rug_d, size_z=rug_h,
        )
