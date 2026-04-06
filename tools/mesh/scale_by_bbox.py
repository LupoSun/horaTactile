#!/usr/bin/env python3
"""
Uniformly scale a mesh so an axis-aligned bounding-box dimension matches a target size.

Usage:
    # Scale so the longest bbox edge becomes 0.08 units
    python tools/mesh/scale_by_bbox.py shape.obj --target-size 0.08

    # Scale so the z extent becomes 0.05 units
    python tools/mesh/scale_by_bbox.py shape.obj --target-size 0.05 --axis z

    # Write to a custom output path
    python tools/mesh/scale_by_bbox.py shape.obj --target-size 0.08 --output scaled/shape.obj
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

try:
    import trimesh
except ImportError:
    trimesh = None


logger = logging.getLogger(__name__)

_AXIS_TO_INDEX = {"x": 0, "y": 1, "z": 2}


def get_bbox_reference_extent(extents, axis: str = "max") -> float:
    extents = np.asarray(extents, dtype=np.float64)
    if extents.shape != (3,):
        raise ValueError(f"Expected 3 bbox extents, got shape {extents.shape}")
    if axis == "max":
        return float(np.max(extents))
    if axis in _AXIS_TO_INDEX:
        return float(extents[_AXIS_TO_INDEX[axis]])
    raise ValueError(f"Unsupported axis '{axis}'. Expected one of: max, x, y, z")


def compute_uniform_bbox_scale(extents, target_size: float, axis: str = "max") -> float:
    if target_size <= 0:
        raise ValueError(f"Target size must be positive, got {target_size}")

    reference_extent = get_bbox_reference_extent(extents, axis=axis)
    if reference_extent <= 1e-10:
        raise ValueError(
            f"Bounding-box extent along axis '{axis}' is too small to scale safely: {reference_extent}"
        )

    return float(target_size / reference_extent)


def scale_extents(extents, scale_factor: float) -> np.ndarray:
    return np.asarray(extents, dtype=np.float64) * float(scale_factor)


def _require_trimesh():
    if trimesh is None:
        sys.exit("trimesh is required: pip install trimesh")


def load_and_clean_mesh(mesh_path: Path):
    _require_trimesh()

    mesh = trimesh.load(mesh_path, force="mesh")

    if not isinstance(mesh, trimesh.Trimesh):
        if isinstance(mesh, trimesh.Scene):
            meshes = [geom for geom in mesh.geometry.values() if isinstance(geom, trimesh.Trimesh)]
            if not meshes:
                raise ValueError(f"No valid geometry found in {mesh_path}")
            mesh = trimesh.util.concatenate(meshes)
        else:
            raise TypeError(f"Unexpected type from trimesh.load: {type(mesh)}")

    mesh = mesh.copy()
    mesh.remove_duplicate_faces()
    mesh.remove_degenerate_faces()
    trimesh.repair.fix_normals(mesh)
    trimesh.repair.fix_winding(mesh)
    trimesh.repair.fill_holes(mesh)

    if len(mesh.faces) == 0:
        raise ValueError(f"Mesh {mesh_path} has no faces after cleaning")

    return mesh


def scale_mesh_by_bbox(mesh, target_size: float, axis: str = "max"):
    scaled_mesh = mesh.copy()
    original_extents = np.asarray(scaled_mesh.bounding_box.extents, dtype=np.float64)
    scale_factor = compute_uniform_bbox_scale(original_extents, target_size, axis=axis)
    scaled_mesh.vertices *= scale_factor
    scaled_extents = np.asarray(scaled_mesh.bounding_box.extents, dtype=np.float64)
    return scaled_mesh, scale_factor, original_extents, scaled_extents


def _default_output_path(mesh_path: Path) -> Path:
    return mesh_path.with_name(f"{mesh_path.stem}_bbox_scaled{mesh_path.suffix}")


def main():
    parser = argparse.ArgumentParser(
        description="Uniformly scale a mesh to a target axis-aligned bounding-box size.",
    )
    parser.add_argument("mesh", type=Path, help="Input mesh file")
    parser.add_argument(
        "--target-size",
        type=float,
        required=True,
        help="Desired bbox size for the selected axis (same units as the mesh)",
    )
    parser.add_argument(
        "--axis",
        choices=("max", "x", "y", "z"),
        default="max",
        help="Which bbox dimension to match: longest edge (`max`) or a specific axis",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output mesh path (default: <input>_bbox_scaled<suffix>)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the output file if it already exists",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose output",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )

    if not args.mesh.exists():
        sys.exit(f"Input mesh not found: {args.mesh}")

    output_path = args.output or _default_output_path(args.mesh)
    if output_path.exists() and not args.force:
        sys.exit(f"Output already exists: {output_path}. Use --force to overwrite.")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    mesh = load_and_clean_mesh(args.mesh)
    scaled_mesh, scale_factor, original_extents, scaled_extents = scale_mesh_by_bbox(
        mesh,
        target_size=args.target_size,
        axis=args.axis,
    )

    scaled_mesh.export(str(output_path))

    logger.info(f"Input:          {args.mesh}")
    logger.info(f"Output:         {output_path}")
    logger.info(f"Axis mode:      {args.axis}")
    logger.info(f"Target size:    {args.target_size:.6f}")
    logger.info(f"Scale factor:   {scale_factor:.6f}")
    logger.info(f"Original bbox:  {np.round(original_extents, 6).tolist()}")
    logger.info(f"Scaled bbox:    {np.round(scaled_extents, 6).tolist()}")


if __name__ == "__main__":
    main()
