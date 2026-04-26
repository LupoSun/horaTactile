#!/usr/bin/env python3
"""Generate normalized point cloud sidecars for assets that contain visual.obj."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

try:
    import trimesh
except ImportError:  # pragma: no cover - exercised when local env lacks trimesh
    trimesh = None


REPO_ROOT = Path(__file__).resolve().parents[1]
POINT_COUNT_CHOICES = (100, 200, 300, 500, 1024)


def farthest_point_sample(points: np.ndarray, n: int) -> np.ndarray:
    selected = [0]
    distances = np.full(len(points), np.inf, dtype=np.float64)

    for _ in range(n - 1):
        last = points[selected[-1]]
        dist_to_last = np.linalg.norm(points - last, axis=1)
        distances = np.minimum(distances, dist_to_last)
        selected.append(int(np.argmax(distances)))

    return points[np.asarray(selected)].copy()


def normalize_points(points: np.ndarray, visual_path: Path) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    points = points - points.mean(axis=0, keepdims=True)
    radius = float(np.max(np.linalg.norm(points, axis=1)))
    if radius <= 1e-10:
        raise ValueError(f"Cannot normalize degenerate mesh {visual_path}")
    return (points / radius).astype(np.float32)


def sample_obj_surface(visual_path: Path, n_points: int) -> np.ndarray:
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    for line in visual_path.read_text(errors="ignore").splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        if parts[0] == "v" and len(parts) >= 4:
            vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
        elif parts[0] == "f" and len(parts) >= 4:
            indices = [int(part.split("/")[0]) - 1 for part in parts[1:]]
            for i in range(1, len(indices) - 1):
                faces.append([indices[0], indices[i], indices[i + 1]])

    verts = np.asarray(vertices, dtype=np.float32)
    if verts.size == 0:
        raise ValueError(f"No vertices found in {visual_path}")
    verts = normalize_points(verts, visual_path)
    if not faces:
        return farthest_point_sample(verts, min(n_points, len(verts))).astype(np.float32)

    tris = verts[np.asarray(faces, dtype=np.int64)]
    cross = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    areas = np.linalg.norm(cross, axis=1) / 2.0
    if float(areas.sum()) <= 1e-12:
        return farthest_point_sample(verts, min(n_points, len(verts))).astype(np.float32)

    dense_count = max(n_points * 4, n_points)
    rng = np.random.default_rng(0)
    face_indices = rng.choice(len(tris), size=dense_count, p=areas / areas.sum())
    chosen = tris[face_indices]
    r1 = np.sqrt(rng.random(dense_count, dtype=np.float32))
    r2 = rng.random(dense_count, dtype=np.float32)
    sampled = (
        (1.0 - r1)[:, None] * chosen[:, 0]
        + (r1 * (1.0 - r2))[:, None] * chosen[:, 1]
        + (r1 * r2)[:, None] * chosen[:, 2]
    )
    return farthest_point_sample(sampled.astype(np.float32), n_points).astype(np.float32)


def normalized_mesh_points(visual_path: Path, n_points: int) -> np.ndarray:
    if trimesh is None:
        return sample_obj_surface(visual_path, n_points)

    mesh = trimesh.load(visual_path, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh) or mesh.vertices.size == 0:
        raise ValueError(f"Could not load a non-empty mesh from {visual_path}")

    mesh = mesh.copy()
    mesh.vertices = normalize_points(mesh.vertices, visual_path)
    dense_count = max(n_points * 4, n_points)
    sampled, _ = trimesh.sample.sample_surface(mesh, dense_count)
    sampled = np.asarray(sampled, dtype=np.float32)
    return farthest_point_sample(sampled, n_points).astype(np.float32)


def iter_visual_paths(paths: list[Path]) -> list[Path]:
    visual_paths: list[Path] = []
    for path in paths:
        resolved = path if path.is_absolute() else REPO_ROOT / path
        if resolved.is_file() and resolved.name == "visual.obj":
            visual_paths.append(resolved)
        elif resolved.is_dir():
            visual_paths.extend(sorted(resolved.rglob("visual.obj")))
        else:
            raise FileNotFoundError(f"Expected a directory or visual.obj file: {path}")
    return sorted(set(visual_paths))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        default=[
            REPO_ROOT / "assets" / "custom" / "cylinder_2dcross",
            REPO_ROOT / "assets" / "custom" / "cylinder_3dcross",
        ],
        help="Asset directories or visual.obj files. Defaults to custom cylinder-cross assets.",
    )
    parser.add_argument("--n-points", type=int, nargs="+", choices=POINT_COUNT_CHOICES, default=[200, 300, 500])
    args = parser.parse_args()

    visual_paths = iter_visual_paths(args.paths)
    if not visual_paths:
        parser.error("No visual.obj files found.")

    for visual_path in visual_paths:
        for n_points in args.n_points:
            points = normalized_mesh_points(visual_path, n_points)
            output_path = visual_path.parent / f"pointcloud_{n_points}.npy"
            np.save(output_path, points)
            print(output_path.relative_to(REPO_ROOT))


if __name__ == "__main__":
    main()
