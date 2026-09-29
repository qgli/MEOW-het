#!/usr/bin/env python3
"""
geometry_factory.py — C-level geometry and PBR material factories for the first-generation scene generator.

Provides:
  1. BMeshFactory — builds primitive geometry in memory with bmesh (no bpy.ops)
  2. ProceduralMaterialFactory — PBR node materials (with ObjectInfo Random injection)

All functions follow three rules:
  - No bpy.ops for geometry (everything goes through bmesh + bpy.data.meshes.new)
  - Physical 3D scale is never broken (transforms are baked into the vertices)
  - No implicit global randomness (all randomness comes from an explicit rng argument)

Design:
  Matrix baking: geometry methods take a transform matrix and apply it directly to the
    vertices, so the output mesh data block already has the correct size and the object
    scale is always (1,1,1).
  Bottom alignment: the bottom of the furniture (Z_min) is aligned exactly to local Z=0.
  Smooth shading: methods take a smooth_shading argument.
  Degeneracy guard: degenerate transforms such as scale=0 are rejected.
"""

from __future__ import annotations

import colorsys
import math
from contextlib import contextmanager
from typing import Generator, Optional

import bmesh
import bpy
from mathutils import Matrix, Vector


# ---- BMeshFactory: C-level geometry factory ----

class BMeshFactory:
    """
    Factory that builds primitive geometry in memory with bmesh.

    All methods are @staticmethod and return a bpy.types.Mesh data block.
    They create no Object, link nothing to the scene and trigger no depsgraph update.

    Principles:
      - All transforms are baked directly into the vertex coordinates via the transform matrix
      - The Z_min of the output mesh is aligned to local Z=0 (via an internal translation)
      - Smooth shading is controlled by the smooth_shading argument
      - Degenerate transforms (scale=0, NaN) are rejected with ValueError
      - All BMesh work runs inside the managed_bmesh() context manager for memory safety
    """

    # ---- Memory safety: BMesh context manager ----

    @staticmethod
    @contextmanager
    def managed_bmesh() -> Generator[bmesh.types.BMesh, None, None]:
        """
        Context manager that guarantees BMesh memory is freed.

        The underlying C++ memory of the BMesh is released even when a C-level error occurs
        (degenerate matrix, invalid vertices, ...).

        Usage:
            with BMeshFactory.managed_bmesh() as bm:
                bmesh.ops.create_cube(bm, size=2.0)
                mesh = BMeshFactory._finalize(bm, "Cube", ...)
            # bm.free() is called automatically in the finally block

        Yields
        ------
        bmesh.types.BMesh
        """
        bm = bmesh.new()
        try:
            yield bm
        finally:
            bm.free()

    @staticmethod
    def _validate_matrix(mat: Matrix) -> None:
        """
        Validate a transform matrix and reject degenerate or invalid transforms.

        Raises
        ------
        ValueError
            If the matrix contains NaN, has a zero scale component or is not 4x4.
        """
        if not isinstance(mat, Matrix):
            raise ValueError(f"Expected mathutils.Matrix, got {type(mat)}")
        if len(mat) != 4 or len(mat[0]) != 4:
            raise ValueError(f"Expected 4x4 matrix, got {len(mat)}x{len(mat[0])}")

        # Scale components (column vector lengths)
        for col_idx in range(3):
            col_vec = Vector((mat[0][col_idx], mat[1][col_idx], mat[2][col_idx]))
            col_len = col_vec.length
            if col_len < 1e-7:
                raise ValueError(
                    f"Degenerate transform: column {col_idx} scale ≈ 0 "
                    f"(length={col_len:.2e}). This would produce collapsed geometry."
                )
            # NaN check
            if math.isnan(col_len) or math.isinf(col_len):
                raise ValueError(
                    f"Invalid transform: column {col_idx} contains "
                    f"NaN or Inf (length={col_len})."
                )

    @staticmethod
    def _finalize(bm: bmesh.types.BMesh, name: str,
                  transform: Optional[Matrix],
                  smooth: bool,
                  z_align_bottom: bool = True) -> bpy.types.Mesh:
        """
        BMesh → bpy.types.Mesh finalization pipeline.

        Steps:
          1. Bottom alignment (optional): translate so that Z_min = 0
          2. Apply the external transform (matrix baking)
          3. Set smooth shading
          4. to_mesh → free

        Parameters
        ----------
        bm : bmesh.types.BMesh
            In-memory BMesh
        name : str
            Name of the output mesh data block
        transform : Matrix or None
            4x4 affine transform, baked directly into the vertices
        smooth : bool
            Whether to enable smooth shading
        z_align_bottom : bool
            Whether to align Z_min to Z=0

        Returns
        -------
        bpy.types.Mesh
        """
        bm.verts.ensure_lookup_table()

        # Bottom alignment along Z
        if z_align_bottom and len(bm.verts) > 0:
            z_min = min(v.co.z for v in bm.verts)
            if abs(z_min) > 1e-6:
                offset = Matrix.Translation(Vector((0, 0, -z_min)))
                bmesh.ops.transform(bm, matrix=offset, verts=bm.verts)

        # Apply the external transform (matrix baking)
        if transform is not None:
            BMeshFactory._validate_matrix(transform)
            bmesh.ops.transform(bm, matrix=transform, verts=bm.verts)

        # Smooth shading
        if smooth:
            for face in bm.faces:
                face.smooth = True

        # Write the mesh (bm.free() is handled by the managed_bmesh() context manager)
        mesh = bpy.data.meshes.new(name)
        bm.to_mesh(mesh)
        mesh.update()
        return mesh

    # ---- Primitive: cube ----

    @staticmethod
    def create_cube(
        name: str = "Cube",
        size_x: float = 1.0,
        size_y: float = 1.0,
        size_z: float = 1.0,
        transform: Optional[Matrix] = None,
        smooth_shading: bool = False,
    ) -> bpy.types.Mesh:
        """
        Create a box mesh with the origin at the center of the bottom face (Z_min = 0).

        Parameters
        ----------
        name : str
            Mesh name
        size_x, size_y, size_z : float
            Size along X/Y/Z (meters)
        transform : Matrix, optional
            4x4 transform baked into the vertices
        smooth_shading : bool
            Whether to use smooth shading

        Returns
        -------
        bpy.types.Mesh
        """
        with BMeshFactory.managed_bmesh() as bm:
            # bmesh.ops.create_cube creates a 2x2x2 cube centered at the origin;
            # a scale matrix brings it to the target size
            scale_mat = Matrix.Diagonal(Vector((
                size_x / 2.0,
                size_y / 2.0,
                size_z / 2.0,
                1.0,
            )))
            bmesh.ops.create_cube(bm, size=2.0, matrix=scale_mat)

            return BMeshFactory._finalize(bm, name, transform, smooth_shading)

    # ---- Primitive: cylinder ----

    @staticmethod
    def create_cylinder(
        name: str = "Cylinder",
        radius: float = 0.5,
        depth: float = 1.0,
        segments: int = 32,
        transform: Optional[Matrix] = None,
        smooth_shading: bool = True,
    ) -> bpy.types.Mesh:
        """
        Create a cylinder mesh with the origin at the center of the bottom face (Z_min = 0).

        Uses bmesh.ops.create_cone with radius1 = radius2.

        Parameters
        ----------
        name : str
            Mesh name
        radius : float
            Radius (meters)
        depth : float
            Height (meters)
        segments : int
            Number of segments around the circumference
        transform : Matrix, optional
            4x4 transform baked into the vertices
        smooth_shading : bool
            Whether to use smooth shading (default True; cylinders need it)

        Returns
        -------
        bpy.types.Mesh
        """
        with BMeshFactory.managed_bmesh() as bm:
            # create_cone: centered at the origin, height along the Z axis;
            # equal bottom and top radii give a cylinder
            bmesh.ops.create_cone(
                bm,
                cap_ends=True,
                cap_tris=False,
                segments=max(3, segments),
                radius1=radius,
                radius2=radius,
                depth=depth,
            )

            # _finalize handles the bottom alignment along Z
            return BMeshFactory._finalize(bm, name, transform, smooth_shading)

    # ---- Primitive: cone / truncated cone ----

    @staticmethod
    def create_cone(
        name: str = "Cone",
        radius_bottom: float = 0.5,
        radius_top: float = 0.0,
        depth: float = 1.0,
        segments: int = 32,
        transform: Optional[Matrix] = None,
        smooth_shading: bool = True,
    ) -> bpy.types.Mesh:
        """
        Create a cone or truncated-cone mesh with the origin at the bottom-face center (Z_min = 0).

        Parameters
        ----------
        name : str
            Mesh name
        radius_bottom : float
            Bottom radius
        radius_top : float
            Top radius (0 = pointed cone)
        depth : float
            Height (meters)
        segments : int
            Number of segments around the circumference
        transform : Matrix, optional
            4x4 transform baked into the vertices
        smooth_shading : bool
            Whether to use smooth shading

        Returns
        -------
        bpy.types.Mesh
        """
        with BMeshFactory.managed_bmesh() as bm:
            bmesh.ops.create_cone(
                bm,
                cap_ends=True,
                cap_tris=(radius_top < 1e-6),  # a pointed cone is capped with triangles
                segments=max(3, segments),
                radius1=radius_bottom,
                radius2=max(0.0, radius_top),
                depth=depth,
            )

            return BMeshFactory._finalize(bm, name, transform, smooth_shading)

    # ---- Primitive: UV sphere ----

    @staticmethod
    def create_sphere(
        name: str = "Sphere",
        radius: float = 0.5,
        u_segments: int = 32,
        v_segments: int = 16,
        transform: Optional[Matrix] = None,
        smooth_shading: bool = True,
    ) -> bpy.types.Mesh:
        """
        Create a UV sphere mesh with the origin at the bottom center (Z_min = 0).

        Uses bmesh.ops.create_uvsphere.

        Parameters
        ----------
        name : str
            Mesh name
        radius : float
            Radius (meters)
        u_segments : int
            Number of meridian segments (horizontal)
        v_segments : int
            Number of latitude segments (vertical)
        transform : Matrix, optional
            4x4 transform baked into the vertices
        smooth_shading : bool
            Whether to use smooth shading (default True; spheres must be smooth)

        Returns
        -------
        bpy.types.Mesh
        """
        with BMeshFactory.managed_bmesh() as bm:
            bmesh.ops.create_uvsphere(
                bm,
                u_segments=max(3, u_segments),
                v_segments=max(2, v_segments),
                radius=radius,
            )

            return BMeshFactory._finalize(bm, name, transform, smooth_shading)

    # ---- Primitive: torus ----

    @staticmethod
    def create_torus(
        name: str = "Torus",
        major_radius: float = 0.3,
        minor_radius: float = 0.1,
        major_segments: int = 24,
        minor_segments: int = 12,
        transform: Optional[Matrix] = None,
        smooth_shading: bool = True,
    ) -> bpy.types.Mesh:
        """
        Create a torus mesh with the bmesh API.

        Parameters
        ----------
        name : str
        major_radius, minor_radius : float
        major_segments, minor_segments : int
        transform, smooth_shading
        """
        with BMeshFactory.managed_bmesh() as bm:
            # bmesh has no torus operator; build the mesh by hand
            import math as _m
            verts_grid = []
            for i in range(major_segments):
                theta = 2 * _m.pi * i / major_segments
                cx = major_radius * _m.cos(theta)
                cy = major_radius * _m.sin(theta)
                row = []
                for j in range(minor_segments):
                    phi = 2 * _m.pi * j / minor_segments
                    x = cx + minor_radius * _m.cos(phi) * _m.cos(theta)
                    y = cy + minor_radius * _m.cos(phi) * _m.sin(theta)
                    z = minor_radius * _m.sin(phi)
                    row.append(bm.verts.new((x, y, z)))
                verts_grid.append(row)

            bm.verts.ensure_lookup_table()
            for i in range(major_segments):
                ni = (i + 1) % major_segments
                for j in range(minor_segments):
                    nj = (j + 1) % minor_segments
                    bm.faces.new([
                        verts_grid[i][j],
                        verts_grid[ni][j],
                        verts_grid[ni][nj],
                        verts_grid[i][nj],
                    ])

            return BMeshFactory._finalize(bm, name, transform, smooth_shading)

    # ---- Primitive: icosphere ----

    @staticmethod
    def create_icosphere(
        name: str = "Icosphere",
        radius: float = 0.5,
        subdivisions: int = 2,
        transform: Optional[Matrix] = None,
        smooth_shading: bool = True,
    ) -> bpy.types.Mesh:
        """
        Create an icosphere (bmesh.ops.create_icosphere).

        Parameters
        ----------
        name : str
        radius : float
        subdivisions : int
        transform, smooth_shading
        """
        with BMeshFactory.managed_bmesh() as bm:
            bmesh.ops.create_icosphere(
                bm,
                subdivisions=max(1, min(4, subdivisions)),
                radius=radius,
            )
            return BMeshFactory._finalize(bm, name, transform, smooth_shading)

    # ---- Primitive: plane ----

    @staticmethod
    def create_plane(
        name: str = "Plane",
        size_x: float = 1.0,
        size_y: float = 1.0,
        transform: Optional[Matrix] = None,
        smooth_shading: bool = False,
    ) -> bpy.types.Mesh:
        """
        Create a horizontal plane mesh (normal along +Z) lying at Z=0.

        Parameters
        ----------
        name : str
            Mesh name
        size_x, size_y : float
            Size along X/Y (meters)
        transform : Matrix, optional
            4x4 transform baked into the vertices
        smooth_shading : bool
            Whether to use smooth shading

        Returns
        -------
        bpy.types.Mesh
        """
        with BMeshFactory.managed_bmesh() as bm:
            hx = size_x / 2.0
            hy = size_y / 2.0
            v0 = bm.verts.new((-hx, -hy, 0.0))
            v1 = bm.verts.new((hx, -hy, 0.0))
            v2 = bm.verts.new((hx, hy, 0.0))
            v3 = bm.verts.new((-hx, hy, 0.0))
            bm.faces.new([v0, v1, v2, v3])  # CCW → normal +Z

            # The plane lies at Z=0; no bottom alignment needed
            return BMeshFactory._finalize(bm, name, transform, smooth_shading,
                                          z_align_bottom=False)

    # ---- Primitive: wall quad ----

    @staticmethod
    def create_wall_quad(
        name: str = "Wall",
        v1: tuple[float, float] = (0, 0),
        v2: tuple[float, float] = (1, 0),
        height: float = 2.8,
        normal_inward: tuple[float, float] = (0, 1),
        transform: Optional[Matrix] = None,
    ) -> bpy.types.Mesh:
        """
        Create a single-sided wall quad whose normal points into the polygon.

        The vertex winding is derived from the inward normal of the CCW polygon so that the
        normal faces the room interior; this keeps walls from rendering black.

        Parameters
        ----------
        name : str
            Mesh name
        v1, v2 : (float, float)
            The two endpoints of the bottom edge of the wall (2D, an edge of the CCW polygon)
        height : float
            Wall height (meters)
        normal_inward : (float, float)
            Precomputed inward normal (from calculate_inward_normal)
        transform : Matrix, optional
            4x4 transform baked into the vertices

        Returns
        -------
        bpy.types.Mesh
        """
        with BMeshFactory.managed_bmesh() as bm:
            # The four wall corners
            p0 = bm.verts.new((v1[0], v1[1], 0.0))       # bottom left
            p1 = bm.verts.new((v2[0], v2[1], 0.0))       # bottom right
            p2 = bm.verts.new((v2[0], v2[1], height))     # top right
            p3 = bm.verts.new((v1[0], v1[1], height))     # top left

            # Choose the vertex winding so that the normal points inward
            # Edge direction: v1→v2
            edge_dir = Vector((v2[0] - v1[0], v2[1] - v1[1], 0))
            up_dir = Vector((0, 0, height))
            face_normal_candidate = edge_dir.cross(up_dir).normalized()
            inward = Vector((normal_inward[0], normal_inward[1], 0))

            if face_normal_candidate.dot(inward) > 0:
                bm.faces.new([p0, p1, p2, p3])
            else:
                bm.faces.new([p3, p2, p1, p0])

            # Walls need no bottom alignment (they already start at Z=0)
            return BMeshFactory._finalize(bm, name, transform,
                                          smooth=False, z_align_bottom=False)

    # ---- Primitive: polygon face ----

    @staticmethod
    def create_polygon_face(
        name: str = "Polygon",
        verts_2d: list[tuple[float, float]] = None,
        z_height: float = 0.0,
        flip_normal: bool = False,
        transform: Optional[Matrix] = None,
    ) -> bpy.types.Mesh:
        """
        Create an arbitrary polygon face (floor/ceiling).

        Parameters
        ----------
        name : str
            Mesh name
        verts_2d : list[(float, float)]
            2D polygon vertices (CCW order)
        z_height : float
            Z height of the polygon
        flip_normal : bool
            True = normal points down (ceiling), False = normal points up (floor)
        transform : Matrix, optional
            4x4 transform baked into the vertices

        Returns
        -------
        bpy.types.Mesh
        """
        if verts_2d is None or len(verts_2d) < 3:
            raise ValueError("Polygon needs at least 3 vertices")

        with BMeshFactory.managed_bmesh() as bm:
            bm_verts = [bm.verts.new((x, y, z_height)) for x, y in verts_2d]

            if flip_normal:
                bm.faces.new(list(reversed(bm_verts)))
            else:
                bm.faces.new(bm_verts)

            return BMeshFactory._finalize(bm, name, transform,
                                          smooth=False, z_align_bottom=False)

    # ---- Composite: multi-part mesh ----

    @staticmethod
    def merge_meshes(
        name: str,
        mesh_list: list[bpy.types.Mesh],
        transform: Optional[Matrix] = None,
        smooth_shading: bool = False,
    ) -> bpy.types.Mesh:
        """
        Merge several mesh data blocks into a single mesh (for composite furniture).

        The input vertex coordinates are assumed to be final (already baked).
        The optional external transform is applied after merging.

        Note: the meshes in mesh_list are not deleted after merging;
        the caller is responsible for cleaning them up.

        Parameters
        ----------
        name : str
            Name of the output mesh
        mesh_list : list[bpy.types.Mesh]
            Mesh data blocks to merge
        transform : Matrix, optional
            Extra transform applied after merging
        smooth_shading : bool
            Whether to use smooth shading

        Returns
        -------
        bpy.types.Mesh
        """
        with BMeshFactory.managed_bmesh() as bm:
            for mesh in mesh_list:
                with BMeshFactory.managed_bmesh() as bm_temp:
                    bm_temp.from_mesh(mesh)
                    # Merge into the main BMesh
                    # by copying vertices and faces directly
                    verts = bm_temp.verts[:]
                    vert_map = {}
                    for v in verts:
                        new_v = bm.verts.new(v.co.copy())
                        vert_map[v.index] = new_v

                    bm_temp.faces.ensure_lookup_table()
                    for face in bm_temp.faces:
                        try:
                            new_face_verts = [vert_map[v.index] for v in face.verts]
                            new_face = bm.faces.new(new_face_verts)
                            new_face.smooth = face.smooth
                        except Exception:
                            pass  # skip degenerate faces

            return BMeshFactory._finalize(bm, name, transform, smooth_shading,
                                          z_align_bottom=True)

    @staticmethod
    def extract_mesh_geometry(
        mesh: bpy.types.Mesh,
        world_matrix: Optional[Matrix] = None,
    ) -> tuple[list[Vector], list[tuple[int, ...]]]:
        """
        Extract vertex coordinates and polygon vertex indices from a mesh data block.
        Used to build a BVHTree for collision detection.

        Parameters
        ----------
        mesh : bpy.types.Mesh
            Blender mesh data block
        world_matrix : Matrix, optional
            World transform. If given, the vertex coordinates are transformed to world space.

        Returns
        -------
        (verts, polys) : tuple
            verts: list[Vector] — vertex coordinates
            polys: list[tuple[int, ...]] — vertex index tuples of the polygons
        """
        if world_matrix is not None:
            verts = [world_matrix @ v.co for v in mesh.vertices]
        else:
            verts = [v.co.copy() for v in mesh.vertices]

        polys = [tuple(p.vertices) for p in mesh.polygons]
        return verts, polys


# ---- ProceduralMaterialFactory: procedural PBR materials ----

class ProceduralMaterialFactory:
    """
    Factory for procedural PBR materials built from shader nodes.

    Design:
      1. All methods take an explicit rng (random.Random) so that materials do not all look alike
      2. Materials inject the ShaderNodeObjectInfo → Random output into ColorRamp.Fac,
         so that instances of the same object type get different tints
      3. Noise drives roughness/bump → fingerprint, wear and dust detail
      4. Socket names follow Blender 4.x
    """

    # ---- ObjectInfo Random injection (shared helper for the materials) ----

    @staticmethod
    def _inject_object_info_random(
        nt: bpy.types.NodeTree,
        bsdf: bpy.types.ShaderNode,
        rng,  # random.Random
        base_color: tuple[float, float, float, float],
        hue_shift_range: tuple[float, float] = (-0.06, 0.06),
        val_scale_range: tuple[float, float] = (0.6, 1.4),
    ) -> bpy.types.ShaderNode:
        """
        Add an ObjectInfo.Random → ColorRamp → Base Color chain to a material node tree.

        This is the main mechanism against homogeneous appearance: e.g. 100 collection instances
        that share one material each get an independent random number in [0, 1], which the
        ColorRamp maps to a different tint.

        It also adds noise textures that drive small roughness and bump variations (offset per
        instance by ObjectInfo.Random) for wear, dust and fingerprint detail.

        Parameters
        ----------
        nt : bpy.types.NodeTree
            Material node tree
        bsdf : bpy.types.ShaderNode
            Principled BSDF node
        rng : random.Random
            Explicit random number generator
        base_color : (R, G, B, A)
            Base color
        hue_shift_range : (float, float)
            Hue shift range (default ±0.06, conservative; colorful materials/fabric can use ±0.35)
        val_scale_range : (float, float)
            Value (brightness) scale range (default 0.6-1.4; wood/metal use 0.4-1.5)

        Returns
        -------
        bpy.types.ShaderNode
            The ObjectInfo node (for further use by the caller)
        """
        # ObjectInfo node
        obj_info = nt.nodes.new("ShaderNodeObjectInfo")

        # ColorRamp: Random → tint variation
        ramp = nt.nodes.new("ShaderNodeValToRGB")
        nt.links.new(obj_info.outputs["Random"], ramp.inputs["Fac"])

        # ColorRamp stops: base color ± hue shift
        r, g, b, a = base_color
        # Dark variant (lower value)
        h, s, v = colorsys.rgb_to_hsv(r, g, b)
        h_shift = rng.uniform(*hue_shift_range)
        val_lo = val_scale_range[0]   # e.g. 0.6
        val_hi = val_scale_range[1]   # e.g. 1.4
        dark_h = (h + h_shift) % 1.0
        dark_v = max(0.05, v * rng.uniform(val_lo, (val_lo + val_hi) / 2))
        dark_s = min(1.0, s * rng.uniform(0.8, 1.2))
        dr, dg, db = colorsys.hsv_to_rgb(dark_h, dark_s, dark_v)
        ramp.color_ramp.elements[0].color = (dr, dg, db, 1.0)

        # Bright variant (higher value)
        bright_h = (h - h_shift) % 1.0
        bright_v = min(1.0, v * rng.uniform((val_lo + val_hi) / 2, val_hi))
        bright_s = max(0.0, s * rng.uniform(0.7, 1.1))
        br, bg, bb = colorsys.hsv_to_rgb(bright_h, bright_s, bright_v)
        ramp.color_ramp.elements[1].color = (br, bg, bb, 1.0)

        # Middle stop (near the base color)
        mid = ramp.color_ramp.elements.new(rng.uniform(0.35, 0.65))
        mid.color = (r, g, b, 1.0)

        # ColorRamp → Base Color, mixed with a texture through MixRGB;
        # first build the coord/mapping/noise chain for the texture perturbation
        coord = nt.nodes.new("ShaderNodeTexCoord")
        mapping = nt.nodes.new("ShaderNodeMapping")
        scale_val = rng.uniform(2.0, 15.0)
        mapping.inputs["Scale"].default_value = (scale_val, scale_val, scale_val)
        nt.links.new(coord.outputs["Object"], mapping.inputs["Vector"])

        noise = nt.nodes.new("ShaderNodeTexNoise")
        noise.inputs["Scale"].default_value = rng.uniform(4.0, 25.0)
        noise.inputs["Detail"].default_value = rng.uniform(4.0, 10.0)
        noise.inputs["Roughness"].default_value = rng.uniform(0.4, 0.8)
        nt.links.new(mapping.outputs["Vector"], noise.inputs["Vector"])

        # MixRGB: ColorRamp base color + noise color perturbation
        mix = nt.nodes.new("ShaderNodeMixRGB")
        mix.blend_type = 'OVERLAY'
        mix.inputs["Fac"].default_value = rng.uniform(0.05, 0.25)
        nt.links.new(ramp.outputs["Color"], mix.inputs["Color1"])
        nt.links.new(noise.outputs["Color"], mix.inputs["Color2"])
        nt.links.new(mix.outputs["Color"], bsdf.inputs["Base Color"])

        # Roughness variation (noise-driven: fingerprints/wear)
        noise2 = nt.nodes.new("ShaderNodeTexNoise")
        noise2.noise_dimensions = '4D'  # must be set before accessing the "W" input
        noise2.inputs["Scale"].default_value = rng.uniform(8.0, 40.0)
        noise2.inputs["Detail"].default_value = 4.0
        nt.links.new(mapping.outputs["Vector"], noise2.inputs["Vector"])

        # Offset the noise W coordinate by ObjectInfo.Random so that instances sharing
        # a material get different roughness patterns
        math_add = nt.nodes.new("ShaderNodeMath")
        math_add.operation = 'ADD'
        nt.links.new(obj_info.outputs["Random"], math_add.inputs[0])
        math_add.inputs[1].default_value = rng.uniform(0, 100)
        nt.links.new(math_add.outputs["Value"], noise2.inputs["W"])

        rough_mul = nt.nodes.new("ShaderNodeMath")
        rough_mul.operation = 'MULTIPLY_ADD'
        rough_mul.inputs[0].default_value = rng.uniform(0.05, 0.15)
        nt.links.new(noise2.outputs["Fac"], rough_mul.inputs[1])
        rough_mul.inputs[2].default_value = bsdf.inputs["Roughness"].default_value
        nt.links.new(rough_mul.outputs["Value"], bsdf.inputs["Roughness"])

        # Bump micro-relief (dust/blemishes)
        bump = nt.nodes.new("ShaderNodeBump")
        bump.inputs["Strength"].default_value = rng.uniform(0.02, 0.12)
        bump.inputs["Distance"].default_value = 0.005
        nt.links.new(noise.outputs["Fac"], bump.inputs["Height"])
        nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])

        return obj_info

    # ---- HSV helper ----

    @staticmethod
    def random_color(
        rng,
        sat_range: tuple[float, float] = (0.2, 0.8),
        val_range: tuple[float, float] = (0.3, 0.9),
    ) -> tuple[float, float, float, float]:
        """Random RGBA color sampled in HSV."""
        h = rng.random()
        s = rng.uniform(*sat_range)
        v = rng.uniform(*val_range)
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        return (r, g, b, 1.0)

    # ---- Generic Principled BSDF ----

    @staticmethod
    def make_principled(
        name: str,
        color: tuple[float, float, float, float],
        roughness: float = 0.5,
        metallic: float = 0.0,
        rng=None,
    ) -> bpy.types.Material:
        """
        Generic Principled BSDF material with ObjectInfo Random injection.

        If rng is given, the full ObjectInfo Random → ColorRamp chain and the noise-driven
        roughness/bump variation are added.

        Parameters
        ----------
        name : str
            Material name
        color : (R, G, B, A)
            Base color
        roughness : float
            Base roughness
        metallic : float
            Metallic value
        rng : random.Random, optional
            Explicit random number generator

        Returns
        -------
        bpy.types.Material
        """
        mat = bpy.data.materials.new(name)
        mat.use_nodes = True
        nt = mat.node_tree
        nt.nodes.clear()

        out = nt.nodes.new("ShaderNodeOutputMaterial")
        bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
        bsdf.inputs["Roughness"].default_value = roughness
        bsdf.inputs["Metallic"].default_value = metallic
        nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

        # Solid-mode preview color
        mat.diffuse_color = color

        if rng is None:
            # Plain color fallback (for calls without rng)
            bsdf.inputs["Base Color"].default_value = color
            return mat

        # Full ObjectInfo Random injection (principled: large hue shift)
        ProceduralMaterialFactory._inject_object_info_random(
            nt, bsdf, rng, color,
            hue_shift_range=(-0.35, 0.35),
            val_scale_range=(0.5, 1.5))

        return mat

    # ---- Wood ----

    @staticmethod
    def make_wood(name: str, rng) -> bpy.types.Material:
        """
        Wood material: wave (growth rings) + noise (pores/blemishes) + bump.
        ObjectInfo Random injection gives each instance a different tint.

        Parameters
        ----------
        name : str
            Material name
        rng : random.Random
            Explicit random number generator

        Returns
        -------
        bpy.types.Material
        """
        mat = bpy.data.materials.new(name)
        mat.use_nodes = True
        nt = mat.node_tree
        nt.nodes.clear()

        out = nt.nodes.new("ShaderNodeOutputMaterial")
        bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
        bsdf.inputs["Roughness"].default_value = rng.uniform(0.2, 0.5)
        bsdf.inputs["Specular IOR Level"].default_value = rng.uniform(0.3, 0.6)

        # ObjectInfo (per-instance tint)
        obj_info = nt.nodes.new("ShaderNodeObjectInfo")

        # Coordinates and mapping
        coord = nt.nodes.new("ShaderNodeTexCoord")
        mapping = nt.nodes.new("ShaderNodeMapping")
        mapping.inputs["Scale"].default_value = (
            rng.uniform(0.5, 2.0), rng.uniform(2.0, 8.0), 1.0)
        mapping.inputs["Rotation"].default_value = (0, 0, rng.uniform(0, 0.3))
        nt.links.new(coord.outputs["Object"], mapping.inputs["Vector"])

        # Wave texture (main growth-ring pattern)
        wave = nt.nodes.new("ShaderNodeTexWave")
        wave.wave_type = rng.choice(['BANDS', 'RINGS'])
        wave.inputs["Scale"].default_value = rng.uniform(2.0, 8.0)
        wave.inputs["Distortion"].default_value = rng.uniform(2.0, 8.0)
        wave.inputs["Detail"].default_value = rng.uniform(3.0, 8.0)
        wave.inputs["Detail Scale"].default_value = rng.uniform(0.5, 2.0)
        nt.links.new(mapping.outputs["Vector"], wave.inputs["Vector"])

        # Noise (fine blemishes/pores)
        noise = nt.nodes.new("ShaderNodeTexNoise")
        noise.inputs["Scale"].default_value = rng.uniform(8.0, 30.0)
        noise.inputs["Detail"].default_value = rng.uniform(6.0, 12.0)
        noise.inputs["Distortion"].default_value = rng.uniform(0.5, 3.0)
        nt.links.new(mapping.outputs["Vector"], noise.inputs["Vector"])

        # Mix wave + noise
        mix_fac = nt.nodes.new("ShaderNodeMixRGB")
        mix_fac.blend_type = 'OVERLAY'
        mix_fac.inputs["Fac"].default_value = rng.uniform(0.15, 0.4)
        nt.links.new(wave.outputs["Color"], mix_fac.inputs["Color1"])
        nt.links.new(noise.outputs["Color"], mix_fac.inputs["Color2"])

        # Color mapping (ColorRamp: dark → mid → light)
        ramp = nt.nodes.new("ShaderNodeValToRGB")
        # Dark (dark ring lines)
        ramp.color_ramp.elements[0].color = (
            rng.uniform(0.08, 0.20), rng.uniform(0.04, 0.12),
            rng.uniform(0.01, 0.05), 1.0)
        # Light (bright ring areas)
        ramp.color_ramp.elements[1].color = (
            rng.uniform(0.40, 0.70), rng.uniform(0.25, 0.45),
            rng.uniform(0.10, 0.22), 1.0)
        # Middle color stop
        mid = ramp.color_ramp.elements.new(rng.uniform(0.3, 0.6))
        mid.color = (
            rng.uniform(0.25, 0.50), rng.uniform(0.15, 0.30),
            rng.uniform(0.06, 0.14), 1.0)
        nt.links.new(mix_fac.outputs["Color"], ramp.inputs["Fac"])

        # ObjectInfo Random → tint shift (wood: conservative hue, wide value range)
        # A second ColorRamp driven by Random overlays the tint variation
        ramp2 = nt.nodes.new("ShaderNodeValToRGB")
        nt.links.new(obj_info.outputs["Random"], ramp2.inputs["Fac"])
        # Warm tint range with a larger brightness spread
        ramp2.color_ramp.elements[0].color = (
            rng.uniform(0.70, 1.0), rng.uniform(0.55, 0.90),
            rng.uniform(0.40, 0.75), 1.0)
        ramp2.color_ramp.elements[1].color = (
            rng.uniform(0.90, 1.0), rng.uniform(0.80, 1.0),
            rng.uniform(0.65, 0.95), 1.0)

        # Multiply: ring color × instance tint
        mix_inst = nt.nodes.new("ShaderNodeMixRGB")
        mix_inst.blend_type = 'MULTIPLY'
        mix_inst.inputs["Fac"].default_value = rng.uniform(0.3, 0.7)
        nt.links.new(ramp.outputs["Color"], mix_inst.inputs["Color1"])
        nt.links.new(ramp2.outputs["Color"], mix_inst.inputs["Color2"])
        nt.links.new(mix_inst.outputs["Color"], bsdf.inputs["Base Color"])

        # Roughness variation (driven by ObjectInfo.Random)
        rough_noise = nt.nodes.new("ShaderNodeTexNoise")
        rough_noise.inputs["Scale"].default_value = rng.uniform(10.0, 30.0)
        rough_noise.inputs["Detail"].default_value = 4.0
        rough_noise.noise_dimensions = '4D'
        nt.links.new(mapping.outputs["Vector"], rough_noise.inputs["Vector"])
        # Offset the W coordinate by Random
        math_add = nt.nodes.new("ShaderNodeMath")
        math_add.operation = 'ADD'
        nt.links.new(obj_info.outputs["Random"], math_add.inputs[0])
        math_add.inputs[1].default_value = rng.uniform(0, 50)
        nt.links.new(math_add.outputs["Value"], rough_noise.inputs["W"])

        rough_mix = nt.nodes.new("ShaderNodeMath")
        rough_mix.operation = 'MULTIPLY_ADD'
        rough_mix.inputs[0].default_value = rng.uniform(0.05, 0.15)
        nt.links.new(rough_noise.outputs["Fac"], rough_mix.inputs[1])
        rough_mix.inputs[2].default_value = bsdf.inputs["Roughness"].default_value
        nt.links.new(rough_mix.outputs["Value"], bsdf.inputs["Roughness"])

        # Bump (wood grain relief)
        bump = nt.nodes.new("ShaderNodeBump")
        bump.inputs["Strength"].default_value = rng.uniform(0.05, 0.2)
        bump.inputs["Distance"].default_value = 0.01
        nt.links.new(wave.outputs["Fac"], bump.inputs["Height"])
        nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])

        nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

        # Solid-mode preview color: approximate wood color
        mat.diffuse_color = (
            rng.uniform(0.35, 0.55), rng.uniform(0.20, 0.35),
            rng.uniform(0.10, 0.20), 1.0)

        return mat

    # ---- Metal ----

    @staticmethod
    def make_metal(name: str, rng) -> bpy.types.Material:
        """
        Metal material: scratch/fingerprint/oxidation-stain texture + ObjectInfo Random.

        Parameters
        ----------
        name : str
            Material name
        rng : random.Random
            Explicit random number generator

        Returns
        -------
        bpy.types.Material
        """
        palettes = [
            (0.95, 0.64, 0.37, 1.0),  # copper
            (0.95, 0.77, 0.35, 1.0),  # gold
            (0.9, 0.9, 0.92, 1.0),    # silver
            (0.8, 0.8, 0.82, 1.0),    # aluminum
            (0.55, 0.30, 0.22, 1.0),  # rusted iron
        ]
        base_color = palettes[rng.randint(0, len(palettes) - 1)]
        base_rough = rng.uniform(0.02, 0.35)

        mat = bpy.data.materials.new(name)
        mat.use_nodes = True
        nt = mat.node_tree
        nt.nodes.clear()

        out = nt.nodes.new("ShaderNodeOutputMaterial")
        bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
        bsdf.inputs["Metallic"].default_value = 1.0
        bsdf.inputs["Roughness"].default_value = base_rough
        nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

        # ObjectInfo Random injection (metal: conservative hue, wide value range)
        ProceduralMaterialFactory._inject_object_info_random(
            nt, bsdf, rng, base_color,
            hue_shift_range=(-0.05, 0.05),
            val_scale_range=(0.4, 1.5))

        # Solid-mode preview color
        mat.diffuse_color = base_color

        return mat

    # ---- Fabric ----

    @staticmethod
    def make_fabric(name: str, rng) -> bpy.types.Material:
        """
        Fabric material: woven texture + fuzz bump + sheen + ObjectInfo Random.

        Parameters
        ----------
        name : str
            Material name
        rng : random.Random
            Explicit random number generator

        Returns
        -------
        bpy.types.Material
        """
        base = ProceduralMaterialFactory.random_color(
            rng, (0.15, 0.6), (0.2, 0.7))

        mat = bpy.data.materials.new(name)
        mat.use_nodes = True
        nt = mat.node_tree
        nt.nodes.clear()

        out = nt.nodes.new("ShaderNodeOutputMaterial")
        bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
        bsdf.inputs["Roughness"].default_value = rng.uniform(0.85, 1.0)
        bsdf.inputs["Sheen Weight"].default_value = rng.uniform(0.1, 0.5)
        nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

        # ObjectInfo
        obj_info = nt.nodes.new("ShaderNodeObjectInfo")

        # Coordinates
        coord = nt.nodes.new("ShaderNodeTexCoord")
        mapping = nt.nodes.new("ShaderNodeMapping")
        fabric_scale = rng.uniform(20.0, 60.0)
        mapping.inputs["Scale"].default_value = (
            fabric_scale, fabric_scale, fabric_scale)
        nt.links.new(coord.outputs["Object"], mapping.inputs["Vector"])

        # Checker (warp/weft weave)
        checker = nt.nodes.new("ShaderNodeTexChecker")
        checker.inputs["Scale"].default_value = rng.uniform(30.0, 80.0)
        checker.inputs["Color1"].default_value = base
        c2 = list(base)
        for k in range(3):
            c2[k] = max(0, min(1, c2[k] + rng.uniform(-0.08, 0.08)))
        checker.inputs["Color2"].default_value = tuple(c2)
        nt.links.new(mapping.outputs["Vector"], checker.inputs["Vector"])

        # Noise (fine grain)
        noise = nt.nodes.new("ShaderNodeTexNoise")
        noise.inputs["Scale"].default_value = rng.uniform(40.0, 120.0)
        noise.inputs["Detail"].default_value = rng.uniform(6.0, 12.0)
        nt.links.new(mapping.outputs["Vector"], noise.inputs["Vector"])

        # Mix
        mix = nt.nodes.new("ShaderNodeMixRGB")
        mix.blend_type = 'MULTIPLY'
        mix.inputs["Fac"].default_value = rng.uniform(0.10, 0.35)
        nt.links.new(checker.outputs["Color"], mix.inputs["Color1"])
        nt.links.new(noise.outputs["Color"], mix.inputs["Color2"])

        # ObjectInfo Random → ColorRamp tint shift (fabric: large hue shift)
        ramp = nt.nodes.new("ShaderNodeValToRGB")
        nt.links.new(obj_info.outputs["Random"], ramp.inputs["Fac"])
        r, g, b, _ = base
        h, s, v = colorsys.rgb_to_hsv(r, g, b)
        dark_r, dark_g, dark_b = colorsys.hsv_to_rgb(
            (h + rng.uniform(-0.35, 0.35)) % 1.0,
            min(1.0, s * rng.uniform(0.8, 1.2)),
            max(0.05, v * rng.uniform(0.4, 0.75)))
        ramp.color_ramp.elements[0].color = (dark_r, dark_g, dark_b, 1.0)
        bright_r, bright_g, bright_b = colorsys.hsv_to_rgb(
            (h + rng.uniform(-0.35, 0.35)) % 1.0,
            max(0.0, s * rng.uniform(0.7, 1.0)),
            min(1.0, v * rng.uniform(1.15, 1.6)))
        ramp.color_ramp.elements[1].color = (bright_r, bright_g, bright_b, 1.0)

        mix2 = nt.nodes.new("ShaderNodeMixRGB")
        mix2.blend_type = 'OVERLAY'
        mix2.inputs["Fac"].default_value = rng.uniform(0.3, 0.6)
        nt.links.new(mix.outputs["Color"], mix2.inputs["Color1"])
        nt.links.new(ramp.outputs["Color"], mix2.inputs["Color2"])
        nt.links.new(mix2.outputs["Color"], bsdf.inputs["Base Color"])

        # Bump (fuzz relief)
        bump = nt.nodes.new("ShaderNodeBump")
        bump.inputs["Strength"].default_value = rng.uniform(0.08, 0.25)
        bump.inputs["Distance"].default_value = 0.005
        nt.links.new(noise.outputs["Fac"], bump.inputs["Height"])
        nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])

        # Solid-mode preview color
        mat.diffuse_color = base

        return mat

    # ---- Tile ----

    @staticmethod
    def make_tile(name: str, rng) -> bpy.types.Material:
        """
        Tile material: brick texture + per-tile color variation + grout bump + ObjectInfo Random.

        Parameters
        ----------
        name : str
            Material name
        rng : random.Random
            Explicit random number generator

        Returns
        -------
        bpy.types.Material
        """
        mat = bpy.data.materials.new(name)
        mat.use_nodes = True
        nt = mat.node_tree
        nt.nodes.clear()

        out = nt.nodes.new("ShaderNodeOutputMaterial")
        bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
        base_rough = rng.uniform(0.05, 0.35)
        bsdf.inputs["Roughness"].default_value = base_rough
        bsdf.inputs["Specular IOR Level"].default_value = rng.uniform(0.4, 0.7)
        nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

        # ObjectInfo
        obj_info = nt.nodes.new("ShaderNodeObjectInfo")

        # Coordinates
        coord = nt.nodes.new("ShaderNodeTexCoord")
        mapping = nt.nodes.new("ShaderNodeMapping")
        tile_scale = rng.uniform(3.0, 8.0)
        mapping.inputs["Scale"].default_value = (
            tile_scale, tile_scale, tile_scale)
        nt.links.new(coord.outputs["Object"], mapping.inputs["Vector"])

        # Brick texture (tile pattern)
        brick = nt.nodes.new("ShaderNodeTexBrick")
        brick.inputs["Scale"].default_value = tile_scale
        color1 = ProceduralMaterialFactory.random_color(
            rng, (0.0, 0.15), (0.5, 0.9))
        color2 = ProceduralMaterialFactory.random_color(
            rng, (0.0, 0.15), (0.4, 0.8))
        brick.inputs["Color1"].default_value = color1
        brick.inputs["Color2"].default_value = color2
        mortar_color = (rng.uniform(0.75, 0.95),) * 3 + (1.0,)
        brick.inputs["Mortar"].default_value = mortar_color
        brick.inputs["Mortar Size"].default_value = rng.uniform(0.005, 0.025)
        brick.inputs["Mortar Smooth"].default_value = rng.uniform(0.0, 0.15)
        nt.links.new(mapping.outputs["Vector"], brick.inputs["Vector"])

        # Noise (slight per-tile color variation)
        noise = nt.nodes.new("ShaderNodeTexNoise")
        noise.inputs["Scale"].default_value = rng.uniform(8.0, 25.0)
        noise.inputs["Detail"].default_value = rng.uniform(3.0, 8.0)
        nt.links.new(mapping.outputs["Vector"], noise.inputs["Vector"])

        mix_color = nt.nodes.new("ShaderNodeMixRGB")
        mix_color.blend_type = rng.choice(['OVERLAY', 'SOFT_LIGHT', 'MULTIPLY'])
        mix_color.inputs["Fac"].default_value = rng.uniform(0.08, 0.25)
        nt.links.new(brick.outputs["Color"], mix_color.inputs["Color1"])
        nt.links.new(noise.outputs["Color"], mix_color.inputs["Color2"])

        # ObjectInfo Random → tint shift
        ramp = nt.nodes.new("ShaderNodeValToRGB")
        nt.links.new(obj_info.outputs["Random"], ramp.inputs["Fac"])
        r1, g1, b1, _ = color1
        ramp.color_ramp.elements[0].color = (
            r1 * rng.uniform(0.85, 1.0),
            g1 * rng.uniform(0.85, 1.0),
            b1 * rng.uniform(0.85, 1.0), 1.0)
        ramp.color_ramp.elements[1].color = (
            min(1.0, r1 * rng.uniform(1.0, 1.15)),
            min(1.0, g1 * rng.uniform(1.0, 1.15)),
            min(1.0, b1 * rng.uniform(1.0, 1.15)), 1.0)

        mix_inst = nt.nodes.new("ShaderNodeMixRGB")
        mix_inst.blend_type = 'MULTIPLY'
        mix_inst.inputs["Fac"].default_value = rng.uniform(0.2, 0.5)
        nt.links.new(mix_color.outputs["Color"], mix_inst.inputs["Color1"])
        nt.links.new(ramp.outputs["Color"], mix_inst.inputs["Color2"])
        nt.links.new(mix_inst.outputs["Color"], bsdf.inputs["Base Color"])

        # Roughness: tile surface vs grout
        rough_mix = nt.nodes.new("ShaderNodeMath")
        rough_mix.operation = 'MULTIPLY_ADD'
        rough_mix.inputs[0].default_value = rng.uniform(0.2, 0.5)
        nt.links.new(brick.outputs["Fac"], rough_mix.inputs[1])
        rough_mix.inputs[2].default_value = base_rough
        nt.links.new(rough_mix.outputs["Value"], bsdf.inputs["Roughness"])

        # Bump (recessed grout lines)
        bump = nt.nodes.new("ShaderNodeBump")
        bump.inputs["Strength"].default_value = rng.uniform(0.1, 0.4)
        bump.inputs["Distance"].default_value = 0.01
        invert = nt.nodes.new("ShaderNodeInvert")
        nt.links.new(brick.outputs["Fac"], invert.inputs["Color"])
        nt.links.new(invert.outputs["Color"], bump.inputs["Height"])
        nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])

        # Solid-mode preview color: approximate tile color
        r1, g1, b1, _ = color1
        mat.diffuse_color = (r1, g1, b1, 1.0)

        return mat

    # ---- Glass ----

    @staticmethod
    def make_glass(name: str, rng) -> bpy.types.Material:
        """
        Glass material: Glass BSDF with randomized IOR.

        Parameters
        ----------
        name : str
            Material name
        rng : random.Random
            Explicit random number generator

        Returns
        -------
        bpy.types.Material
        """
        mat = bpy.data.materials.new(name)
        mat.use_nodes = True
        nt = mat.node_tree
        nt.nodes.clear()

        out = nt.nodes.new("ShaderNodeOutputMaterial")
        glass = nt.nodes.new("ShaderNodeBsdfGlass")
        glass.inputs["Color"].default_value = (
            rng.uniform(0.8, 1.0), rng.uniform(0.9, 1.0),
            rng.uniform(0.9, 1.0), 1.0)
        glass.inputs["IOR"].default_value = rng.uniform(1.3, 1.6)
        glass.inputs["Roughness"].default_value = rng.uniform(0.0, 0.08)
        nt.links.new(glass.outputs["BSDF"], out.inputs["Surface"])

        # Solid-mode preview color: semi-transparent hint
        mat.diffuse_color = (0.85, 0.92, 0.95, 0.3)

        return mat

    # ---- Emission ----

    @staticmethod
    def make_emission(
        name: str,
        rng,
        strength_range: tuple[float, float] = (2.0, 15.0),
    ) -> bpy.types.Material:
        """
        Emissive material.

        Parameters
        ----------
        name : str
            Material name
        rng : random.Random
            Explicit random number generator
        strength_range : (float, float)
            Emission strength range

        Returns
        -------
        bpy.types.Material
        """
        mat = bpy.data.materials.new(name)
        mat.use_nodes = True
        nt = mat.node_tree
        nt.nodes.clear()

        out = nt.nodes.new("ShaderNodeOutputMaterial")
        em = nt.nodes.new("ShaderNodeEmission")
        em.inputs["Color"].default_value = ProceduralMaterialFactory.random_color(
            rng, (0.3, 1.0), (0.7, 1.0))
        em.inputs["Strength"].default_value = rng.uniform(*strength_range)
        nt.links.new(em.outputs["Emission"], out.inputs["Surface"])

        return mat

    # ---- Painting ----

    @staticmethod
    def make_painting(name: str, rng) -> bpy.types.Material:
        """
        Picture-frame material: abstract painting generated from Voronoi/noise textures.

        Parameters
        ----------
        name : str
            Material name
        rng : random.Random
            Explicit random number generator

        Returns
        -------
        bpy.types.Material
        """
        mat = bpy.data.materials.new(name)
        mat.use_nodes = True
        nt = mat.node_tree
        nt.nodes.clear()

        out = nt.nodes.new("ShaderNodeOutputMaterial")
        bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
        bsdf.inputs["Roughness"].default_value = 0.8
        nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

        tex_type = rng.choice(['voronoi', 'noise', 'wave', 'checker'])
        if tex_type == 'voronoi':
            tex = nt.nodes.new("ShaderNodeTexVoronoi")
            tex.inputs["Scale"].default_value = rng.uniform(2.0, 10.0)
            color_out = tex.outputs["Color"]
        elif tex_type == 'noise':
            tex = nt.nodes.new("ShaderNodeTexNoise")
            tex.inputs["Scale"].default_value = rng.uniform(2.0, 8.0)
            tex.inputs["Detail"].default_value = rng.uniform(2.0, 10.0)
            tex.inputs["Distortion"].default_value = rng.uniform(0.5, 5.0)
            color_out = tex.outputs["Color"]
        elif tex_type == 'wave':
            tex = nt.nodes.new("ShaderNodeTexWave")
            tex.inputs["Scale"].default_value = rng.uniform(1.0, 5.0)
            tex.inputs["Distortion"].default_value = rng.uniform(2.0, 10.0)
            color_out = tex.outputs["Color"]
        else:
            tex = nt.nodes.new("ShaderNodeTexChecker")
            tex.inputs["Scale"].default_value = rng.uniform(4.0, 16.0)
            tex.inputs["Color1"].default_value = \
                ProceduralMaterialFactory.random_color(rng)
            tex.inputs["Color2"].default_value = \
                ProceduralMaterialFactory.random_color(rng)
            color_out = tex.outputs["Color"]

        ramp = nt.nodes.new("ShaderNodeValToRGB")
        ramp.color_ramp.elements[0].color = \
            ProceduralMaterialFactory.random_color(rng, (0.3, 1.0), (0.2, 0.6))
        ramp.color_ramp.elements[1].color = \
            ProceduralMaterialFactory.random_color(rng, (0.3, 1.0), (0.5, 1.0))
        nt.links.new(color_out, ramp.inputs["Fac"])
        nt.links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])

        return mat

    # ---- Wall paint ----

    @staticmethod
    def make_wall_paint(name: str, rng) -> bpy.types.Material:
        """
        Wall paint: probabilistic color distribution + Voronoi perturbation + noise bump.

        Color distribution:
          - 20% white / off-white walls
          - 15% dark / deep vintage colors
          - 65% ordinary colored walls

        Parameters
        ----------
        name : str
        rng : random.Random

        Returns
        -------
        bpy.types.Material
        """
        h = rng.uniform(0.0, 1.0)
        _style_roll = rng.random()
        if _style_roll < 0.20:
            # 20% white / off-white walls
            s = rng.uniform(0.0, 0.05)
            v = rng.uniform(0.85, 0.98)
        elif _style_roll < 0.35:
            # 15% dark / deep vintage colors
            s = rng.uniform(0.3, 0.8)
            v = rng.uniform(0.15, 0.4)
        else:
            # 65% ordinary colored walls
            s = rng.uniform(0.05, 0.4)
            v = rng.uniform(0.4, 0.9)
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        base_color = (r, g, b, 1.0)

        mat = bpy.data.materials.new(name)
        mat.use_nodes = True
        nt = mat.node_tree
        nt.nodes.clear()

        out = nt.nodes.new("ShaderNodeOutputMaterial")
        bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
        bsdf.inputs["Roughness"].default_value = rng.uniform(0.55, 0.95)
        bsdf.inputs["Metallic"].default_value = 0.0
        nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

        # ObjectInfo Random injection: small color differences
        obj_info = nt.nodes.new("ShaderNodeObjectInfo")
        ramp = nt.nodes.new("ShaderNodeValToRGB")
        nt.links.new(obj_info.outputs["Random"], ramp.inputs["Fac"])
        # Dark variant (high contrast)
        dr, dg, db = colorsys.hsv_to_rgb(
            (h + rng.uniform(-0.05, 0.05)) % 1.0,
            min(1.0, s * rng.uniform(1.0, 1.8)),
            max(0.2, v * rng.uniform(0.70, 0.90)))
        ramp.color_ramp.elements[0].color = (dr, dg, db, 1.0)
        # Bright variant (high contrast)
        br_, bg_, bb_ = colorsys.hsv_to_rgb(
            (h + rng.uniform(-0.03, 0.03)) % 1.0,
            max(0.0, s * rng.uniform(0.5, 1.0)),
            min(1.0, v * rng.uniform(1.02, 1.15)))
        ramp.color_ramp.elements[1].color = (br_, bg_, bb_, 1.0)
        mid = ramp.color_ramp.elements.new(rng.uniform(0.3, 0.7))
        mid.color = base_color

        # Large-scale Voronoi perturbation: wall texture
        coord = nt.nodes.new("ShaderNodeTexCoord")
        mapping = nt.nodes.new("ShaderNodeMapping")
        sc = rng.uniform(3.0, 12.0)
        mapping.inputs["Scale"].default_value = (sc, sc, sc)
        nt.links.new(coord.outputs["Object"], mapping.inputs["Vector"])

        voronoi = nt.nodes.new("ShaderNodeTexVoronoi")
        voronoi.inputs["Scale"].default_value = rng.uniform(4.0, 15.0)
        voronoi.voronoi_dimensions = '3D'
        voronoi.feature = 'F1'
        nt.links.new(mapping.outputs["Vector"], voronoi.inputs["Vector"])

        # MixRGB: ColorRamp color × Voronoi perturbation
        mix_voronoi = nt.nodes.new("ShaderNodeMixRGB")
        mix_voronoi.blend_type = 'OVERLAY'
        mix_voronoi.inputs["Fac"].default_value = rng.uniform(0.08, 0.25)
        nt.links.new(ramp.outputs["Color"], mix_voronoi.inputs["Color1"])
        nt.links.new(voronoi.outputs["Color"], mix_voronoi.inputs["Color2"])
        nt.links.new(mix_voronoi.outputs["Color"], bsdf.inputs["Base Color"])

        # Noise bump (dust/brush strokes)
        noise = nt.nodes.new("ShaderNodeTexNoise")
        noise.inputs["Scale"].default_value = rng.uniform(10.0, 40.0)
        noise.inputs["Detail"].default_value = rng.uniform(3.0, 8.0)
        nt.links.new(mapping.outputs["Vector"], noise.inputs["Vector"])

        bump = nt.nodes.new("ShaderNodeBump")
        bump.inputs["Strength"].default_value = rng.uniform(0.02, 0.10)
        bump.inputs["Distance"].default_value = 0.003
        nt.links.new(noise.outputs["Fac"], bump.inputs["Height"])
        nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])

        # Solid-mode preview color: approximate paint color
        mat.diffuse_color = (r, g, b, 1.0)

        return mat

    # ---- Concrete ----

    @staticmethod
    def make_concrete(name: str, rng) -> bpy.types.Material:
        """
        Concrete/cement floor material: Musgrave/Voronoi grain + noise cracks + bump.

        Parameters
        ----------
        name : str
        rng : random.Random

        Returns
        -------
        bpy.types.Material
        """
        mat = bpy.data.materials.new(name)
        mat.use_nodes = True
        nt = mat.node_tree
        nt.nodes.clear()

        out = nt.nodes.new("ShaderNodeOutputMaterial")
        bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
        bsdf.inputs["Roughness"].default_value = rng.uniform(0.65, 0.95)
        bsdf.inputs["Metallic"].default_value = 0.0
        bsdf.inputs["Specular IOR Level"].default_value = rng.uniform(0.2, 0.4)
        nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

        # Coordinates
        coord = nt.nodes.new("ShaderNodeTexCoord")
        mapping = nt.nodes.new("ShaderNodeMapping")
        sc = rng.uniform(1.0, 4.0)
        mapping.inputs["Scale"].default_value = (sc, sc, sc)
        nt.links.new(coord.outputs["Object"], mapping.inputs["Vector"])

        # Base color: gray tones
        v = rng.uniform(0.35, 0.65)
        base_r = v + rng.uniform(-0.03, 0.03)
        base_g = v + rng.uniform(-0.02, 0.02)
        base_b = v + rng.uniform(-0.03, 0.01)

        # Voronoi grain
        voronoi = nt.nodes.new("ShaderNodeTexVoronoi")
        voronoi.inputs["Scale"].default_value = rng.uniform(8.0, 25.0)
        voronoi.voronoi_dimensions = '3D'
        voronoi.feature = rng.choice(['F1', 'F2'])
        nt.links.new(mapping.outputs["Vector"], voronoi.inputs["Vector"])

        # Noise: slight color variation
        noise = nt.nodes.new("ShaderNodeTexNoise")
        noise.inputs["Scale"].default_value = rng.uniform(5.0, 15.0)
        noise.inputs["Detail"].default_value = rng.uniform(4.0, 10.0)
        noise.inputs["Distortion"].default_value = rng.uniform(0.5, 2.0)
        nt.links.new(mapping.outputs["Vector"], noise.inputs["Vector"])

        # ColorRamp: gray tones
        ramp = nt.nodes.new("ShaderNodeValToRGB")
        nt.links.new(voronoi.outputs["Distance"], ramp.inputs["Fac"])
        ramp.color_ramp.elements[0].color = (
            base_r * 0.80, base_g * 0.80, base_b * 0.80, 1.0)
        ramp.color_ramp.elements[1].color = (
            min(1.0, base_r * 1.15), min(1.0, base_g * 1.15),
            min(1.0, base_b * 1.10), 1.0)

        # Mix: Voronoi color × noise
        mix = nt.nodes.new("ShaderNodeMixRGB")
        mix.blend_type = 'OVERLAY'
        mix.inputs["Fac"].default_value = rng.uniform(0.15, 0.35)
        nt.links.new(ramp.outputs["Color"], mix.inputs["Color1"])
        nt.links.new(noise.outputs["Color"], mix.inputs["Color2"])
        nt.links.new(mix.outputs["Color"], bsdf.inputs["Base Color"])

        # Bump (surface relief)
        bump = nt.nodes.new("ShaderNodeBump")
        bump.inputs["Strength"].default_value = rng.uniform(0.05, 0.2)
        bump.inputs["Distance"].default_value = rng.uniform(0.005, 0.02)
        nt.links.new(voronoi.outputs["Distance"], bump.inputs["Height"])
        nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])

        # Solid-mode preview color
        mat.diffuse_color = (base_r, base_g, base_b, 1.0)

        return mat

    # ---- Marble ----

    @staticmethod
    def make_marble(name: str, rng) -> bpy.types.Material:
        """
        Marble material: wave veins + Voronoi cracks + glossy surface.

        Parameters
        ----------
        name : str
        rng : random.Random

        Returns
        -------
        bpy.types.Material
        """
        mat = bpy.data.materials.new(name)
        mat.use_nodes = True
        nt = mat.node_tree
        nt.nodes.clear()

        out = nt.nodes.new("ShaderNodeOutputMaterial")
        bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
        bsdf.inputs["Roughness"].default_value = rng.uniform(0.02, 0.15)
        bsdf.inputs["Specular IOR Level"].default_value = rng.uniform(0.4, 0.7)
        nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

        # Coordinates
        coord = nt.nodes.new("ShaderNodeTexCoord")
        mapping = nt.nodes.new("ShaderNodeMapping")
        sc = rng.uniform(1.5, 5.0)
        mapping.inputs["Scale"].default_value = (sc, sc, sc)
        mapping.inputs["Rotation"].default_value = (0, 0, rng.uniform(0, 1.0))
        nt.links.new(coord.outputs["Object"], mapping.inputs["Vector"])

        # Marble color scheme
        scheme = rng.choice(['white', 'black', 'beige', 'green'])
        if scheme == 'white':
            c1 = (rng.uniform(0.88, 0.98), rng.uniform(0.88, 0.98),
                  rng.uniform(0.88, 0.98), 1.0)
            c2 = (rng.uniform(0.35, 0.55), rng.uniform(0.35, 0.55),
                  rng.uniform(0.35, 0.55), 1.0)
        elif scheme == 'black':
            c1 = (rng.uniform(0.10, 0.25), rng.uniform(0.10, 0.25),
                  rng.uniform(0.10, 0.25), 1.0)
            c2 = (rng.uniform(0.50, 0.70), rng.uniform(0.50, 0.70),
                  rng.uniform(0.50, 0.70), 1.0)
        elif scheme == 'beige':
            c1 = (rng.uniform(0.80, 0.95), rng.uniform(0.70, 0.85),
                  rng.uniform(0.55, 0.70), 1.0)
            c2 = (rng.uniform(0.45, 0.60), rng.uniform(0.35, 0.50),
                  rng.uniform(0.25, 0.40), 1.0)
        else:  # green
            c1 = (rng.uniform(0.25, 0.45), rng.uniform(0.40, 0.60),
                  rng.uniform(0.25, 0.40), 1.0)
            c2 = (rng.uniform(0.10, 0.25), rng.uniform(0.20, 0.35),
                  rng.uniform(0.10, 0.25), 1.0)

        # Wave veins
        wave = nt.nodes.new("ShaderNodeTexWave")
        wave.wave_type = 'BANDS'
        wave.inputs["Scale"].default_value = rng.uniform(2.0, 6.0)
        wave.inputs["Distortion"].default_value = rng.uniform(4.0, 12.0)
        wave.inputs["Detail"].default_value = rng.uniform(4.0, 10.0)
        wave.inputs["Detail Scale"].default_value = rng.uniform(0.5, 2.0)
        nt.links.new(mapping.outputs["Vector"], wave.inputs["Vector"])

        # Voronoi cracks
        voronoi = nt.nodes.new("ShaderNodeTexVoronoi")
        voronoi.inputs["Scale"].default_value = rng.uniform(3.0, 8.0)
        voronoi.voronoi_dimensions = '3D'
        voronoi.feature = 'DISTANCE_TO_EDGE'
        nt.links.new(mapping.outputs["Vector"], voronoi.inputs["Vector"])

        # ColorRamp
        ramp = nt.nodes.new("ShaderNodeValToRGB")
        nt.links.new(wave.outputs["Fac"], ramp.inputs["Fac"])
        ramp.color_ramp.elements[0].color = c1
        ramp.color_ramp.elements[1].color = c2

        # Mix: add the cracks
        mix = nt.nodes.new("ShaderNodeMixRGB")
        mix.blend_type = 'MULTIPLY'
        mix.inputs["Fac"].default_value = rng.uniform(0.05, 0.20)
        nt.links.new(ramp.outputs["Color"], mix.inputs["Color1"])
        nt.links.new(voronoi.outputs["Distance"], mix.inputs["Color2"])
        nt.links.new(mix.outputs["Color"], bsdf.inputs["Base Color"])

        # Bump
        bump = nt.nodes.new("ShaderNodeBump")
        bump.inputs["Strength"].default_value = rng.uniform(0.02, 0.08)
        bump.inputs["Distance"].default_value = 0.005
        nt.links.new(voronoi.outputs["Distance"], bump.inputs["Height"])
        nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])

        # Solid-mode preview color
        mat.diffuse_color = c1

        return mat

    # ---- Random material ----

    @staticmethod
    def random_material(name: str, rng) -> bpy.types.Material:
        """Pick a random opaque material (glass excluded)."""
        choice = rng.choice(['principled', 'wood', 'metal',
                             'fabric', 'tile', 'concrete'])
        if choice == 'principled':
            return ProceduralMaterialFactory.make_principled(
                name,
                ProceduralMaterialFactory.random_color(rng),
                roughness=rng.uniform(0.1, 0.9),
                rng=rng,
            )
        elif choice == 'wood':
            return ProceduralMaterialFactory.make_wood(name, rng)
        elif choice == 'metal':
            return ProceduralMaterialFactory.make_metal(name, rng)
        elif choice == 'fabric':
            return ProceduralMaterialFactory.make_fabric(name, rng)
        elif choice == 'concrete':
            return ProceduralMaterialFactory.make_concrete(name, rng)
        else:
            return ProceduralMaterialFactory.make_tile(name, rng)
