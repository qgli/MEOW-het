#!/usr/bin/env python3
"""
Wall decoration: paintings, windows and wall-mounted objects.

WallDecorator places paintings, windows and wall-mounted items on the walls of a built room.

Features:
  1. add_paintings     - canvas (create_plane + make_painting) + thin frame
  2. add_windows       - glass pane + frame + an AREA light that fakes daylight
  3. add_wall_mounted_objects - wall shelves, wall lamps (POINT light), clocks

Implementation notes:
  - Orientation: to_track_quat('Z', 'Y')
  - Window lights: the AREA light's -Z is aligned with the wall's normal_inward
  - 1D overlap avoidance: occupied parameter intervals per wall, parameterized by t_span

Rules:
  - No bpy.ops, no global random state, no destructive scaling
  - Every wall decoration is offset along the inward normal by at least 0.015 m (prevents z-fighting)
  - The window AREA light sits on the inner side of the glass
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import bpy
from mathutils import Euler, Matrix, Quaternion, Vector

from .core_utils import (
    create_data_object,
    link_object_to_scene,
    polygon_centroid,
)
from .geometry_factory import BMeshFactory, ProceduralMaterialFactory


# ---- WallDecorator ----

class WallDecorator:
    """
    Wall decoration manager.

    Takes wall_info and floor_verts from RoomBuilder and places paintings,
    windows and wall-mounted items on the walls. All decorations are offset
    along the inward normal to prevent z-fighting.
    """

    # Offset from the wall against z-fighting (m)
    WALL_OFFSET: float = 0.015
    # Inward offset of the window lights (m)
    WINDOW_LIGHT_OFFSET: float = 0.05

    def __init__(
        self,
        wall_info: list[dict],
        floor_verts: Sequence[tuple[float, float]],
        ceiling_height: float = 2.8,
        collection: Optional[bpy.types.Collection] = None,
    ) -> None:
        """
        Parameters
        ----------
        wall_info : list[dict]
            Wall entries from RoomBuilder.wall_info
        floor_verts : Sequence[(float, float)]
            Room floor polygon (CCW)
        ceiling_height : float
            Ceiling height (m)
        collection : bpy.types.Collection, optional
            Collection for the decoration objects
        """
        self.wall_info = wall_info
        self.floor_verts = floor_verts
        self.ceiling_height = ceiling_height

        if collection is None:
            self._col = bpy.data.collections.new("WallDecor")
            bpy.context.scene.collection.children.link(self._col)
        else:
            self._col = collection

        # Created objects
        self.paintings: list[bpy.types.Object] = []
        self.windows: list[bpy.types.Object] = []
        self.window_lights: list[bpy.types.Object] = []
        self.mounted_objects: list[bpy.types.Object] = []

        # Room center (reference only; not used to orient the lights)
        cx, cy = polygon_centroid(floor_verts)
        self._room_center = Vector((cx, cy, ceiling_height * 0.5))

        # 1D overlap avoidance: occupied intervals of each wall
        # key = wall['index'], value = list[(t_start, t_end)]
        self._occupied: dict[int, list[tuple[float, float]]] = {}

    # ---- Wall coordinates ----

    def _select_long_walls(
        self,
        rng,
        n: int,
        min_length: float = 1.5,
    ) -> list[dict]:
        """Pick up to n distinct walls that are long enough."""
        candidates = [w for w in self.wall_info if w['length'] >= min_length]
        if not candidates:
            candidates = list(self.wall_info)
        rng.shuffle(candidates)
        return candidates[:min(n, len(candidates))]

    def _wall_point_at_param(
        self,
        wall: dict,
        t: float,
        z_frac: float,
    ) -> tuple[Vector, Vector]:
        """
        World position on the wall at t in [0, 1] (along the edge) and z_frac in [0, 1]
        (fraction of the height), offset along the inward normal.

        Returns
        -------
        (world_pos, normal_3d) : tuple[Vector, Vector]
        """
        v1 = wall['v1']
        v2 = wall['v2']
        nx, ny = wall['normal_inward']

        # Linear interpolation along the edge
        px = v1[0] + (v2[0] - v1[0]) * t
        py = v1[1] + (v2[1] - v1[1]) * t
        pz = z_frac * self.ceiling_height

        # Offset along the inward normal (prevents z-fighting)
        world_pos = Vector((
            px + nx * self.WALL_OFFSET,
            py + ny * self.WALL_OFFSET,
            pz,
        ))
        normal_3d = Vector((nx, ny, 0))

        return world_pos, normal_3d

    def _orient_to_wall_normal(
        self,
        normal_3d: Vector,
    ) -> Euler:
        """
        Rotation computed with to_track_quat.

        Aligns the object's +Z axis (the plane normal) with the inward wall normal,
        with +Y pointing up (Z-up world).
        """
        quat = normal_3d.to_track_quat('Z', 'Y')
        return quat.to_euler()

    # ---- 1D overlap avoidance ----

    def _intervals_overlap(
        self,
        intervals: list[tuple[float, float]],
        new_start: float,
        new_end: float,
    ) -> bool:
        """Return True if the new interval overlaps an existing one."""
        for s, e in intervals:
            if new_start < e and new_end > s:
                return True
        return False

    def _try_sample_wall_t(
        self,
        rng,
        wall: dict,
        item_width: float,
        max_retries: int = 20,
    ) -> Optional[float]:
        """
        1D overlap-free sampling: find a wall parameter t whose span does not overlap occupied intervals.

        Parameters
        ----------
        rng : random.Random
        wall : dict
            Wall entry
        item_width : float
            Item width (m)
        max_retries : int
            Maximum number of retries

        Returns
        -------
        float or None
            Parameter t in [0, 1], or None on failure
        """
        wall_len = wall['length']
        if wall_len < 0.1:
            return None

        t_span = (item_width / 2.0) / wall_len

        # Wall too short for the item
        if 2 * t_span + 0.1 > 1.0:
            return None

        t_lo = t_span + 0.05
        t_hi = 1.0 - t_span - 0.05
        if t_lo >= t_hi:
            return None

        wall_idx = wall['index']
        intervals = self._occupied.setdefault(wall_idx, [])

        for _ in range(max_retries):
            t = rng.uniform(t_lo, t_hi)
            new_start = t - t_span
            new_end = t + t_span
            if not self._intervals_overlap(intervals, new_start, new_end):
                # Mark as occupied
                intervals.append((new_start, new_end))
                return t

        return None

    # ---- Paintings ----

    def add_paintings(
        self,
        rng,
        n_paintings: int = 2,
    ) -> list[bpy.types.Object]:
        """
        Hang paintings on random long walls.

        Each painting = canvas (plane + make_painting) + thin frame (cube).

        Parameters
        ----------
        rng : random.Random
        n_paintings : int

        Returns
        -------
        list[bpy.types.Object]
        """
        walls = self._select_long_walls(rng, n_paintings, min_length=1.5)
        results: list[bpy.types.Object] = []

        for i, wall in enumerate(walls):
            # 1D overlap-free sampling
            pw = rng.uniform(0.3, 0.8)
            ph = rng.uniform(0.3, 0.6)
            t = self._try_sample_wall_t(rng, wall, pw)
            if t is None:
                continue
            z_frac = rng.uniform(0.45, 0.70)
            pos, normal = self._wall_point_at_param(wall, t, z_frac)
            rot = self._orient_to_wall_normal(normal)

            # Canvas (plane)
            canvas_mesh = BMeshFactory.create_plane(
                f"PaintingCanvas_{i:02d}",
                size_x=pw, size_y=ph,
            )
            canvas_obj = create_data_object(
                f"PaintingCanvas_{i:02d}", canvas_mesh, self._col)
            canvas_obj.location = pos
            canvas_obj.rotation_euler = rot

            # Painting material
            paint_mat = ProceduralMaterialFactory.make_painting(
                f"PaintMat_{i:02d}", rng)
            canvas_obj.data.materials.append(paint_mat)

            # Frame (thin cube around the canvas)
            frame_depth = rng.uniform(0.015, 0.03)
            frame_border = rng.uniform(0.02, 0.05)
            frame_mesh = BMeshFactory.create_cube(
                f"PaintingFrame_{i:02d}",
                size_x=pw + 2 * frame_border,
                size_y=ph + 2 * frame_border,
                size_z=frame_depth,
            )
            # Frame slightly behind the canvas
            frame_pos = pos - normal * 0.005
            frame_obj = create_data_object(
                f"PaintingFrame_{i:02d}", frame_mesh, self._col)
            frame_obj.location = frame_pos
            frame_obj.rotation_euler = rot

            # Frame material: random (wood / metal / colored)
            frame_mat_choice = rng.choice(['wood', 'metal', 'principled'])
            if frame_mat_choice == 'wood':
                frame_mat = ProceduralMaterialFactory.make_wood(
                    f"FrameWood_{i:02d}", rng)
            elif frame_mat_choice == 'metal':
                frame_mat = ProceduralMaterialFactory.make_metal(
                    f"FrameMetal_{i:02d}", rng)
            else:
                frame_mat = ProceduralMaterialFactory.make_principled(
                    f"FrameColor_{i:02d}",
                    ProceduralMaterialFactory.random_color(rng, (0.0, 0.3), (0.1, 0.5)),
                    roughness=rng.uniform(0.1, 0.6), rng=rng)
            frame_obj.data.materials.append(frame_mat)

            results.extend([canvas_obj, frame_obj])

        self.paintings.extend(results)
        return results

    # ---- Windows (with an AREA light that fakes daylight) ----

    def add_windows(
        self,
        rng,
        n_windows: int = 2,
    ) -> list[bpy.types.Object]:
        """
        Add windows (glass + frame) to walls, each with an AREA light that fakes daylight.

        The AREA light sits on the inner side of the glass (0.05 m into the room along the
        normal) and shines into the room, imitating sunlight coming through the window.
        It must not be placed outside the wall, where the wall would block it.

        Parameters
        ----------
        rng : random.Random
        n_windows : int

        Returns
        -------
        list[bpy.types.Object]
            All window and light objects
        """
        walls = self._select_long_walls(rng, n_windows, min_length=2.0)
        # Windows only on external walls
        walls = [w for w in walls if not w.get('is_internal', False)]
        results: list[bpy.types.Object] = []

        for i, wall in enumerate(walls):
            nx, ny = wall['normal_inward']

            # 1D overlap-free sampling
            ww = rng.uniform(0.6, 1.2)
            wh = rng.uniform(0.8, 1.4)
            t = self._try_sample_wall_t(rng, wall, ww)
            if t is None:
                continue
            z_frac = rng.uniform(0.35, 0.60)
            pos, normal = self._wall_point_at_param(wall, t, z_frac)
            rot = self._orient_to_wall_normal(normal)

            # Glass pane (plane)
            glass_mesh = BMeshFactory.create_plane(
                f"WindowGlass_{i:02d}",
                size_x=ww, size_y=wh,
            )
            glass_obj = create_data_object(
                f"WindowGlass_{i:02d}", glass_mesh, self._col)
            glass_obj.location = pos
            glass_obj.rotation_euler = rot

            glass_mat = ProceduralMaterialFactory.make_glass(
                f"WindowGlass_{i:02d}", rng)
            glass_obj.data.materials.append(glass_mat)

            # Frame (thin cube)
            frame_border = rng.uniform(0.03, 0.06)
            frame_depth = rng.uniform(0.02, 0.04)
            frame_mesh = BMeshFactory.create_cube(
                f"WindowFrame_{i:02d}",
                size_x=ww + 2 * frame_border,
                size_y=wh + 2 * frame_border,
                size_z=frame_depth,
            )
            frame_pos = pos - normal * 0.005
            frame_obj = create_data_object(
                f"WindowFrame_{i:02d}", frame_mesh, self._col)
            frame_obj.location = frame_pos
            frame_obj.rotation_euler = rot

            frame_mat = ProceduralMaterialFactory.make_principled(
                f"WinFrameMat_{i:02d}",
                (0.9, 0.9, 0.92, 1.0),
                roughness=0.3,
            )
            frame_obj.data.materials.append(frame_mat)

            # Fake daylight: AREA light on the inner side of the glass
            # -Z aligned with normal_inward
            light_data = bpy.data.lights.new(
                f"WindowLight_{i:02d}", type='AREA')
            light_data.energy = rng.uniform(80, 250)
            light_data.color = (
                rng.uniform(0.95, 1.0),
                rng.uniform(0.90, 1.0),
                rng.uniform(0.80, 0.95),
            )
            light_data.shape = 'RECTANGLE'
            light_data.size = ww * 0.8
            light_data.size_y = wh * 0.8

            # Light position: glass + 0.05 m inwards along the normal
            light_pos = pos + normal * self.WINDOW_LIGHT_OFFSET
            light_obj = create_data_object(
                f"WindowLight_{i:02d}", light_data, self._col)
            light_obj.location = light_pos

            # AREA light: -Z aligned with normal_inward
            # Lights emit along -Z by default; align -Z with the inward wall normal
            light_quat = normal.to_track_quat('-Z', 'Y')
            light_obj.rotation_euler = light_quat.to_euler()

            results.extend([glass_obj, frame_obj, light_obj])
            self.window_lights.append(light_obj)

        self.windows.extend(results)
        return results

    # ---- Wall-mounted objects ----

    def add_wall_mounted_objects(
        self,
        rng,
        n_objects: int = 3,
    ) -> list[bpy.types.Object]:
        """
        Mount minimal wall shelves, wall lamps or clocks on the walls.

        Types:
          - shelf: thin box shelf
          - wall_lamp: small sphere + POINT light
          - clock: flat cylinder

        Parameters
        ----------
        rng : random.Random
        n_objects : int

        Returns
        -------
        list[bpy.types.Object]
        """
        walls = self._select_long_walls(rng, n_objects, min_length=1.0)
        results: list[bpy.types.Object] = []

        for i, wall in enumerate(walls):
            obj_type = rng.choice(['shelf', 'wall_lamp', 'clock'])

            # 1D overlap-free sampling
            item_width = 0.5  # conservative width estimate
            t = self._try_sample_wall_t(rng, wall, item_width)
            if t is None:
                continue
            z_frac = rng.uniform(0.40, 0.70)
            pos, normal = self._wall_point_at_param(wall, t, z_frac)

            if obj_type == 'shelf':
                results.extend(self._make_shelf(rng, i, pos, normal))
            elif obj_type == 'wall_lamp':
                results.extend(self._make_wall_lamp(rng, i, pos, normal))
            else:
                results.extend(self._make_clock(rng, i, pos, normal))

        self.mounted_objects.extend(results)
        return results

    def _make_shelf(
        self,
        rng,
        idx: int,
        pos: Vector,
        normal: Vector,
    ) -> list[bpy.types.Object]:
        """Minimal wall shelf: a thin box."""
        sw = rng.uniform(0.3, 0.8)
        sd = rng.uniform(0.12, 0.20)
        sh = rng.uniform(0.015, 0.025)

        mesh = BMeshFactory.create_cube(
            f"WallShelf_{idx:02d}",
            size_x=sw, size_y=sd, size_z=sh,
        )
        obj = create_data_object(
            f"WallShelf_{idx:02d}", mesh, self._col)
        # Position: wall point + offset along the normal so the shelf sticks out
        obj.location = pos + normal * (sd / 2)

        # Rotation: shelf Y axis along the normal (front faces the room)
        quat = normal.to_track_quat('Y', 'Z')
        obj.rotation_euler = quat.to_euler()

        mat = ProceduralMaterialFactory.random_material(
            f"ShelfMat_{idx:02d}", rng)
        obj.data.materials.append(mat)

        return [obj]

    def _make_wall_lamp(
        self,
        rng,
        idx: int,
        pos: Vector,
        normal: Vector,
    ) -> list[bpy.types.Object]:
        """Wall lamp: small sphere + POINT light."""
        r = rng.uniform(0.04, 0.08)

        lamp_mesh = BMeshFactory.create_sphere(
            f"WallLamp_{idx:02d}",
            radius=r,
            u_segments=12,
            v_segments=6,
        )
        # Offset along the normal so the lamp sticks out
        lamp_transform = Matrix.Translation(pos + normal * (r + 0.01))
        lamp_obj = create_data_object(
            f"WallLamp_{idx:02d}", lamp_mesh, self._col)
        lamp_obj.location = pos + normal * (r + 0.01)

        emit_mat = ProceduralMaterialFactory.make_emission(
            f"LampEmit_{idx:02d}", rng, strength_range=(3.0, 8.0))
        lamp_obj.data.materials.append(emit_mat)

        # POINT Light
        light_data = bpy.data.lights.new(
            f"WallLampLight_{idx:02d}", type='POINT')
        light_data.energy = rng.uniform(20, 80)
        light_data.color = (
            rng.uniform(0.95, 1.0),
            rng.uniform(0.85, 1.0),
            rng.uniform(0.70, 0.95),
        )
        light_data.shadow_soft_size = rng.uniform(0.05, 0.15)

        light_obj = create_data_object(
            f"WallLampLight_{idx:02d}", light_data, self._col)
        light_obj.location = pos + normal * (2 * r + 0.02)

        return [lamp_obj, light_obj]

    def _make_clock(
        self,
        rng,
        idx: int,
        pos: Vector,
        normal: Vector,
    ) -> list[bpy.types.Object]:
        """Clock: flat cylinder."""
        cr = rng.uniform(0.10, 0.18)
        cd = 0.02  # thickness

        clock_mesh = BMeshFactory.create_cylinder(
            f"WallClock_{idx:02d}",
            radius=cr,
            depth=cd,
            segments=24,
        )
        obj = create_data_object(
            f"WallClock_{idx:02d}", clock_mesh, self._col)
        # Position, and rotation so that the round face points along the normal
        obj.location = pos + normal * (cd / 2)
        # Rotation: the cylinder axis is Z by default; rotate Z onto the normal
        target = Vector((0, 0, 1))
        quat = target.rotation_difference(normal)
        obj.rotation_euler = quat.to_euler()

        mat = ProceduralMaterialFactory.make_principled(
            f"ClockMat_{idx:02d}",
            ProceduralMaterialFactory.random_color(rng, (0.0, 0.2), (0.1, 0.4)),
            roughness=rng.uniform(0.1, 0.4),
            rng=rng,
        )
        obj.data.materials.append(mat)

        return [obj]

    # ---- Decorate all ----

    def decorate_all(
        self,
        rng,
        n_paintings: int = 2,
        n_windows: int = 2,
        n_mounted: int = 3,
    ) -> dict:
        """
        Run all wall decoration steps.

        Returns
        -------
        dict
            {
                'paintings': list[Object],
                'windows': list[Object],
                'window_lights': list[Object],
                'mounted_objects': list[Object],
            }
        """
        paintings = self.add_paintings(rng, n_paintings)
        windows = self.add_windows(rng, n_windows)
        mounted = self.add_wall_mounted_objects(rng, n_mounted)

        return {
            'paintings': paintings,
            'windows': windows,
            'window_lights': self.window_lights,
            'mounted_objects': mounted,
        }
