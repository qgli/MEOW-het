#!/usr/bin/env python3
"""
Collection instancing and external asset loading for the scene generator.

Provides:
  1. InstanceManager: hidden template collections + collection instancing
  2. DynamicAssetPipeline: loading objects from external .blend files + physical-size normalization

Design rules:
  - No bpy.ops, no global random state, no destructive scaling
  - The template collection (InstancedTemplates) is never linked into the scene's children
  - Each instance is an Empty(instance_type='COLLECTION'), so it adds no mesh data
  - Matrices are assigned directly to Empty.matrix_world, so no scale is baked in
"""

from __future__ import annotations

import os
from typing import Optional

import bpy
from mathutils import Matrix, Vector

from .core_utils import link_object_to_scene


# ---- InstanceManager: hidden templates + collection instancing ----

class InstanceManager:
    """
    Object reuse through Blender collection instancing.

    How it works:
      1. Create an "InstancedTemplates" collection in bpy.data.collections
      2. Never link it to scene.collection.children, so the templates stay hidden
      3. Register a template: wrap a BMeshFactory mesh in an Object inside this collection
      4. Instantiate: create an Empty(instance_type='COLLECTION') bound to the template collection

    Benefits:
      - 100 chair instances share one copy of the mesh data (99% memory saving)
      - Each Empty only holds a 4x4 matrix
      - ObjectInfo.Random differs per Empty, so materials vary automatically

    Notes:
      - Each template lives in its own sub-collection (template_name -> sub-collection)
      - The root InstancedTemplates collection is the container; sub-collections are bound to Empties
    """

    # Name of the hidden root collection
    ROOT_COLLECTION_NAME: str = "InstancedTemplates"

    def __init__(self) -> None:
        """Create or reuse the hidden root collection."""
        # Find or create the hidden root collection (not linked to the scene)
        existing = bpy.data.collections.get(self.ROOT_COLLECTION_NAME)
        if existing is not None:
            self._root_collection = existing
        else:
            self._root_collection = bpy.data.collections.new(
                self.ROOT_COLLECTION_NAME)
            # scene.collection.children.link() is deliberately not called, so the
            # template collection is invisible in the viewport and in renders

        # Template registry: name -> (sub_collection, template_object)
        self._templates: dict[str, tuple[bpy.types.Collection,
                                         bpy.types.Object]] = {}

    @property
    def root_collection(self) -> bpy.types.Collection:
        """Hidden root collection (exists only in bpy.data.collections)."""
        return self._root_collection

    @property
    def template_names(self) -> list[str]:
        """Names of the registered templates."""
        return list(self._templates.keys())

    def register_template(
        self,
        name: str,
        mesh_data: bpy.types.Mesh,
        material: Optional[bpy.types.Material] = None,
    ) -> bpy.types.Object:
        """
        Register a template: Mesh -> Object -> hidden sub-collection.

        Parameters
        ----------
        name : str
            Template name (unique key)
        mesh_data : bpy.types.Mesh
            Normalized mesh data block built by BMeshFactory
        material : bpy.types.Material, optional
            Material assigned to the template object

        Returns
        -------
        bpy.types.Object
            Template object (inside the hidden collection)

        Raises
        ------
        ValueError
            If the template name is already registered
        """
        if name in self._templates:
            raise ValueError(
                f"Template '{name}' already registered. "
                f"Existing: {self.template_names}"
            )

        # One sub-collection per template
        sub_col = bpy.data.collections.new(f"TPL_{name}")
        self._root_collection.children.link(sub_col)

        # Create the template object inside the sub-collection
        template_obj = bpy.data.objects.new(f"Template_{name}", mesh_data)
        sub_col.objects.link(template_obj)

        # Assign the material
        if material is not None:
            template_obj.data.materials.append(material)

        # Register
        self._templates[name] = (sub_col, template_obj)
        return template_obj

    def spawn_instance(
        self,
        template_name: str,
        transform_matrix: Matrix,
        target_collection: Optional[bpy.types.Collection] = None,
        instance_name: Optional[str] = None,
    ) -> bpy.types.Object:
        """
        Spawn a collection instance (an Empty object).

        Parameters
        ----------
        template_name : str
            Name of a registered template
        transform_matrix : Matrix
            4x4 world-space transform
        target_collection : bpy.types.Collection, optional
            Collection that receives the instance. Defaults to the scene's master collection.
        instance_name : str, optional
            Name of the instance Empty. Generated automatically by default.

        Returns
        -------
        bpy.types.Object
            Collection-instance Empty

        Raises
        ------
        KeyError
            If the template name is not registered
        """
        if template_name not in self._templates:
            raise KeyError(
                f"Template '{template_name}' not found. "
                f"Available: {self.template_names}"
            )

        sub_col, _ = self._templates[template_name]

        # Create the Empty and enable collection instancing
        if instance_name is None:
            instance_name = f"Inst_{template_name}"

        empty = bpy.data.objects.new(instance_name, None)
        empty.instance_type = 'COLLECTION'
        empty.instance_collection = sub_col

        # Link to the target collection
        link_object_to_scene(empty, target_collection)

        # Apply the world matrix
        empty.matrix_world = transform_matrix

        return empty

    def has_template(self, name: str) -> bool:
        """Return True if the template is registered."""
        return name in self._templates

    def get_template_object(self, name: str) -> bpy.types.Object:
        """Return the template object (e.g. to read its bound_box)."""
        if name not in self._templates:
            raise KeyError(f"Template '{name}' not found.")
        _, obj = self._templates[name]
        return obj


# ---- DynamicAssetPipeline: external asset pipeline (stub) ----

class DynamicAssetPipeline:
    """
    Loading pipeline for external .blend assets.

    Provides:
      1. load_and_instantiate_asset: import an Object from an external .blend file
      2. normalize_physical_scale: physical-size normalization (uniform scaling)

    Notes:
      - Uses bpy.data.libraries.load() (not bpy.ops)
      - Raises FileNotFoundError if the file does not exist
    """

    @staticmethod
    def load_and_instantiate_asset(
        filepath: str,
        object_name: str,
        target_collection: Optional[bpy.types.Collection] = None,
    ) -> bpy.types.Object:
        """
        Load the named Object from an external .blend file and link it to the scene.

        Parameters
        ----------
        filepath : str
            Path of the external .blend file
        object_name : str
            Name of the Object to import
        target_collection : bpy.types.Collection, optional
            Target collection. Defaults to the scene's master collection.

        Returns
        -------
        bpy.types.Object
            The imported object

        Raises
        ------
        FileNotFoundError
            The file does not exist
        RuntimeError
            The object is not found in the file
        """
        if not os.path.isfile(filepath):
            raise FileNotFoundError(
                f"Asset file not found: '{filepath}'"
            )

        loaded_objects: list[bpy.types.Object] = []

        with bpy.data.libraries.load(filepath, link=False) as (
            data_from, data_to
        ):
            if object_name not in data_from.objects:
                raise RuntimeError(
                    f"Object '{object_name}' not found in '{filepath}'. "
                    f"Available: {data_from.objects[:20]}"
                )
            data_to.objects = [object_name]

        # data_to.objects now holds the loaded objects (None where loading failed)
        for obj in data_to.objects:
            if obj is not None:
                loaded_objects.append(obj)
                link_object_to_scene(obj, target_collection)

        if not loaded_objects:
            raise RuntimeError(
                f"Failed to load object '{object_name}' from '{filepath}'"
            )

        return loaded_objects[0]

    @staticmethod
    def normalize_physical_scale(
        obj: bpy.types.Object,
        target_height_m: float,
    ) -> float:
        """
        Physical-size normalization: scale uniformly so the object's Z extent equals the target height.

        Takes the Z range of the world-space bound_box, computes a uniform scale factor
        and multiplies all three components of obj.scale by it.

        Parameters
        ----------
        obj : bpy.types.Object
            Target object
        target_height_m : float
            Target height (m)

        Returns
        -------
        float
            Applied scale factor

        Raises
        ------
        ValueError
            The object has zero height (degenerate geometry)
        """
        # The 8 bound_box corners in world space
        world_mat = obj.matrix_world
        corners = [world_mat @ Vector(c) for c in obj.bound_box]
        z_coords = [c.z for c in corners]
        current_height = max(z_coords) - min(z_coords)

        if current_height < 1e-7:
            raise ValueError(
                f"Object '{obj.name}' has near-zero height "
                f"({current_height:.2e}m). Cannot normalize."
            )

        scale_factor = target_height_m / current_height

        # Uniform scaling (keeps proportions)
        obj.scale = (
            obj.scale.x * scale_factor,
            obj.scale.y * scale_factor,
            obj.scale.z * scale_factor,
        )

        return scale_factor
