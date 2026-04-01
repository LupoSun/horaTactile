#!/usr/bin/env python3
"""
Mesh preprocessing pipeline for horaTactile.

Takes Rhino-exported meshes (.obj, .stl, .ply, .3dm) and produces
IsaacGym-ready assets with point clouds for geometry-aware training.

Per-object output:
    assets/custom/{name}/
        visual.obj             - cleaned mesh for rendering
        collision.obj          - convex decomposition for physics
        {name}.urdf            - IsaacGym-loadable URDF
        pointcloud_100.npy     - 100 surface points (unit sphere normalised)
        pointcloud_1024.npy    - 1024 surface points (unit sphere normalised)
        metadata.json          - bounding box, volume, surface area, scale factor

Usage:
    # Single mesh
    python tools/mesh/preprocess.py path/to/shape.obj

    # Batch
    python tools/mesh/preprocess.py path/to/meshes/*.obj

    # Custom output directory
    python tools/mesh/preprocess.py shape.obj --output-dir assets/my_objects

    # Skip convex decomposition (use original mesh for collision)
    python tools/mesh/preprocess.py shape.obj --skip-decomposition

    # Custom point counts
    python tools/mesh/preprocess.py shape.obj --n-points 100 512 2048
"""

import argparse
import json
import logging
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from xml.dom import minidom

import numpy as np

try:
    import trimesh
except ImportError:
    sys.exit("trimesh is required: pip install trimesh")

logger = logging.getLogger(__name__)

# Repo root (two levels up from tools/mesh/)
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "assets" / "custom"

# Default inertial values matching existing HORA URDFs
DEFAULT_MASS = 0.05
DEFAULT_INERTIA = 0.0001


# ---------------------------------------------------------------------------
# Mesh loading & cleaning
# ---------------------------------------------------------------------------


def load_and_clean(mesh_path: Path) -> trimesh.Trimesh:
    """Load a mesh file and clean it for simulation use."""
    mesh = trimesh.load(mesh_path, force="mesh")

    if not isinstance(mesh, trimesh.Trimesh):
        # Handle scenes / multi-body exports — merge into single mesh
        if isinstance(mesh, trimesh.Scene):
            meshes = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
            if not meshes:
                raise ValueError(f"No valid geometry found in {mesh_path}")
            mesh = trimesh.util.concatenate(meshes)
        else:
            raise TypeError(f"Unexpected type from trimesh.load: {type(mesh)}")

    # Fix common mesh issues
    mesh.remove_duplicate_faces()
    mesh.remove_degenerate_faces()
    trimesh.repair.fix_normals(mesh)
    trimesh.repair.fix_winding(mesh)
    trimesh.repair.fill_holes(mesh)

    if len(mesh.faces) == 0:
        raise ValueError(f"Mesh {mesh_path} has no faces after cleaning")

    return mesh


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------


def normalise_to_unit_sphere(mesh: trimesh.Trimesh) -> tuple[trimesh.Trimesh, float, np.ndarray]:
    """
    Centre at origin and scale so the bounding sphere has radius 1.

    Returns:
        normalised mesh, scale_factor (original_radius), centroid offset
    """
    centroid = mesh.centroid.copy()
    mesh.vertices -= centroid

    # Bounding sphere radius
    radius = float(np.max(np.linalg.norm(mesh.vertices, axis=1)))
    if radius < 1e-10:
        raise ValueError("Mesh is degenerate (zero bounding radius)")

    mesh.vertices /= radius

    return mesh, radius, centroid


# ---------------------------------------------------------------------------
# Convex decomposition
# ---------------------------------------------------------------------------


def convex_decompose(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """
    Run convex decomposition for physics collision.
    Tries coacd first (better quality), falls back to V-HACD via trimesh.
    """
    try:
        import coacd

        coacd_mesh = coacd.Mesh(mesh.vertices, mesh.faces)
        parts = coacd.run_coacd(coacd_mesh, threshold=0.05)
        decomposed = []
        for vs, fs in parts:
            decomposed.append(trimesh.Trimesh(vertices=vs, faces=fs))
        result = trimesh.util.concatenate(decomposed)
        logger.info(f"  CoACD decomposition: {len(parts)} convex parts")
        return result

    except ImportError:
        logger.info("  coacd not available, trying trimesh convex_decomposition (V-HACD)")

    try:
        parts = mesh.convex_decomposition()
        if isinstance(parts, trimesh.Trimesh):
            parts = [parts]
        result = trimesh.util.concatenate(parts)
        logger.info(f"  V-HACD decomposition: {len(parts)} convex parts")
        return result

    except Exception as e:
        logger.warning(f"  Convex decomposition failed ({e}), using convex hull as fallback")
        return mesh.convex_hull


# ---------------------------------------------------------------------------
# Point cloud sampling
# ---------------------------------------------------------------------------


def sample_point_clouds(mesh: trimesh.Trimesh, counts: list[int]) -> dict[int, np.ndarray]:
    """
    Sample surface point clouds at multiple resolutions.
    Points are in the mesh's coordinate frame (unit sphere after normalisation).
    """
    max_count = max(counts)
    points, _ = trimesh.sample.sample_surface(mesh, max_count)
    points = np.asarray(points, dtype=np.float32)

    result = {}
    for n in sorted(counts):
        if n >= len(points):
            result[n] = points.copy()
        else:
            # Farthest point sampling for better coverage
            result[n] = _farthest_point_sample(points, n)

    return result


def _farthest_point_sample(points: np.ndarray, n: int) -> np.ndarray:
    """Greedy farthest point sampling for spatially uniform subset."""
    selected = [0]
    distances = np.full(len(points), np.inf)

    for _ in range(n - 1):
        last = points[selected[-1]]
        dist_to_last = np.linalg.norm(points - last, axis=1)
        distances = np.minimum(distances, dist_to_last)
        selected.append(int(np.argmax(distances)))

    return points[np.array(selected)].copy()


# ---------------------------------------------------------------------------
# Inertia computation
# ---------------------------------------------------------------------------


def compute_inertia(mesh: trimesh.Trimesh, mass: float) -> dict:
    """Compute inertia tensor for a mesh at given mass."""
    if mesh.is_watertight:
        mesh.density = mass / mesh.volume
        inertia = mesh.moment_inertia
    else:
        # For non-watertight meshes, approximate with bounding box
        extents = mesh.bounding_box.extents
        # Box inertia: I = m/12 * (b^2 + c^2), etc.
        ixx = mass / 12.0 * (extents[1] ** 2 + extents[2] ** 2)
        iyy = mass / 12.0 * (extents[0] ** 2 + extents[2] ** 2)
        izz = mass / 12.0 * (extents[0] ** 2 + extents[1] ** 2)
        inertia = np.diag([ixx, iyy, izz])

    return {
        "ixx": float(inertia[0, 0]),
        "iyy": float(inertia[1, 1]),
        "izz": float(inertia[2, 2]),
        "ixy": float(inertia[0, 1]),
        "ixz": float(inertia[0, 2]),
        "iyz": float(inertia[1, 2]),
    }


# ---------------------------------------------------------------------------
# URDF generation
# ---------------------------------------------------------------------------


def generate_urdf(
    name: str,
    visual_path: str,
    collision_path: str,
    mass: float = DEFAULT_MASS,
    inertia: dict | None = None,
) -> str:
    """
    Generate a URDF string matching the HORA asset convention.

    Mesh paths should be relative to the repo root (IsaacGym asset_root).
    """
    if inertia is None:
        inertia = {k: DEFAULT_INERTIA if k in ("ixx", "iyy", "izz") else 0.0
                   for k in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")}

    robot = ET.Element("robot", name="object")
    link = ET.SubElement(robot, "link", name="object")

    # Visual
    visual = ET.SubElement(link, "visual")
    ET.SubElement(visual, "origin", xyz="0 0 0")
    vis_geom = ET.SubElement(visual, "geometry")
    ET.SubElement(vis_geom, "mesh", filename=visual_path, scale="1 1 1")

    # Collision
    collision = ET.SubElement(link, "collision")
    ET.SubElement(collision, "origin", xyz="0 0 0")
    col_geom = ET.SubElement(collision, "geometry")
    ET.SubElement(col_geom, "mesh", filename=collision_path, scale="1 1 1")

    # Inertial
    inertial = ET.SubElement(link, "inertial")
    ET.SubElement(inertial, "mass", value=str(mass))
    ET.SubElement(inertial, "inertia", **{k: str(v) for k, v in inertia.items()})

    # Pretty-print
    rough = ET.tostring(robot, encoding="unicode")
    dom = minidom.parseString(rough)
    return dom.toprettyxml(indent="  ").split("\n", 1)[1]  # strip xml declaration


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def process_mesh(
    mesh_path: Path,
    output_dir: Path,
    point_counts: list[int],
    skip_decomposition: bool = False,
    mass: float = DEFAULT_MASS,
) -> Path:
    """
    Full processing pipeline for a single mesh.

    Returns path to the created object directory.
    """
    name = mesh_path.stem
    obj_dir = output_dir / name
    obj_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Processing: {mesh_path.name} -> {obj_dir}")

    # 1. Load & clean
    logger.info("  Loading and cleaning mesh...")
    mesh = load_and_clean(mesh_path)
    original_bounds = mesh.bounding_box.extents.tolist()
    original_volume = float(mesh.volume) if mesh.is_watertight else None
    original_area = float(mesh.area)

    # 2. Normalise to unit sphere
    logger.info("  Normalising to unit bounding sphere...")
    mesh, scale_factor, centroid = normalise_to_unit_sphere(mesh)

    # 3. Export visual mesh
    visual_path = obj_dir / "visual.obj"
    mesh.export(str(visual_path))
    logger.info(f"  Visual mesh: {visual_path.name} ({len(mesh.faces)} faces)")

    # 4. Convex decomposition for collision
    if skip_decomposition:
        collision_path = visual_path
        logger.info("  Skipping decomposition, using visual mesh for collision")
    else:
        logger.info("  Running convex decomposition...")
        collision_mesh = convex_decompose(mesh)
        collision_path = obj_dir / "collision.obj"
        collision_mesh.export(str(collision_path))

    # 5. Compute inertia
    inertia = compute_inertia(mesh, mass)

    # 6. Generate URDF
    # Paths in URDF must be relative to repo root (IsaacGym asset_root)
    visual_rel = str(visual_path.relative_to(REPO_ROOT))
    collision_rel = str(collision_path.relative_to(REPO_ROOT))

    urdf_content = generate_urdf(
        name=name,
        visual_path=visual_rel,
        collision_path=collision_rel,
        mass=mass,
        inertia=inertia,
    )
    urdf_path = obj_dir / f"{name}.urdf"
    urdf_path.write_text(urdf_content)
    logger.info(f"  URDF: {urdf_path.name}")

    # 7. Sample point clouds
    logger.info(f"  Sampling point clouds: {point_counts}...")
    clouds = sample_point_clouds(mesh, point_counts)
    for n, pts in clouds.items():
        npy_path = obj_dir / f"pointcloud_{n}.npy"
        np.save(str(npy_path), pts)
        logger.info(f"  Point cloud ({n} pts): {npy_path.name}")

    # 8. Save metadata
    metadata = {
        "name": name,
        "source_file": mesh_path.name,
        "canonical_scale_factor": float(scale_factor),
        "centroid_offset": centroid.tolist(),
        "original_bbox": original_bounds,
        "normalised_bbox": mesh.bounding_box.extents.tolist(),
        "volume_m3": original_volume,
        "surface_area_m2": original_area,
        "n_faces": len(mesh.faces),
        "n_vertices": len(mesh.vertices),
        "is_watertight": mesh.is_watertight,
        "mass_kg": mass,
        "inertia": inertia,
        "point_cloud_counts": sorted(clouds.keys()),
        "urdf_file": f"{name}.urdf",
    }
    meta_path = obj_dir / "metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2))
    logger.info(f"  Metadata: {meta_path.name}")

    return obj_dir


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess meshes for horaTactile (Rhino → IsaacGym + point clouds)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "meshes",
        nargs="+",
        type=Path,
        help="Mesh files to process (.obj, .stl, .ply, .3dm)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR.relative_to(REPO_ROOT)})",
    )
    parser.add_argument(
        "--n-points",
        nargs="+",
        type=int,
        default=[100, 1024],
        help="Point cloud sample counts (default: 100 1024)",
    )
    parser.add_argument(
        "--skip-decomposition",
        action="store_true",
        help="Skip convex decomposition, use original mesh for collision",
    )
    parser.add_argument(
        "--mass",
        type=float,
        default=DEFAULT_MASS,
        help=f"Object mass in kg (default: {DEFAULT_MASS})",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )

    results = []
    errors = []

    for mesh_path in args.meshes:
        if not mesh_path.exists():
            logger.error(f"File not found: {mesh_path}")
            errors.append(mesh_path)
            continue
        try:
            obj_dir = process_mesh(
                mesh_path=mesh_path,
                output_dir=args.output_dir,
                point_counts=args.n_points,
                skip_decomposition=args.skip_decomposition,
                mass=args.mass,
            )
            results.append(obj_dir)
        except Exception as e:
            logger.error(f"Failed to process {mesh_path}: {e}")
            errors.append(mesh_path)

    # Summary
    print(f"\n{'=' * 50}")
    print(f"Processed: {len(results)} / {len(args.meshes)} meshes")
    for d in results:
        print(f"  {d.relative_to(REPO_ROOT)}/")
    if errors:
        print(f"Errors ({len(errors)}):")
        for e in errors:
            print(f"  {e}")

    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
