from __future__ import annotations

import math
import os
from glob import glob
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]


def _subset_from_primitive(primitive: str, prefix: str) -> str:
    if primitive == prefix:
        return "default"
    if primitive.startswith(f"{prefix}_"):
        return primitive[len(prefix) + 1:]
    raise ValueError(f"Unsupported primitive '{primitive}' for prefix '{prefix}'")


def _glob_relative_paths(repo_root: Path, patterns: list[str]) -> list[str]:
    matches: list[str] = []
    for pattern in patterns:
        for path_str in sorted(glob(str(repo_root / pattern))):
            path = Path(path_str)
            if path.is_file():
                matches.append(path.relative_to(repo_root).as_posix())
    return matches


def build_object_asset_catalog(
    object_type: str,
    sample_prob: list[float],
    repo_root: Path | None = None,
) -> tuple[list[str], list[float], dict[str, str]]:
    repo_root = REPO_ROOT if repo_root is None else Path(repo_root)
    primitive_list = object_type.split("+")
    if len(sample_prob) != len(primitive_list):
        raise ValueError(
            f"sampleProb length {len(sample_prob)} does not match object types {len(primitive_list)}"
        )
    if abs(sum(sample_prob) - 1.0) > 1e-6:
        raise ValueError("sampleProb must sum to 1.0")

    object_type_prob: list[float] = []
    object_type_list: list[str] = []
    asset_files_dict = {
        "simple_tennis_ball": "assets/ball.urdf",
    }

    for primitive, primitive_prob in zip(primitive_list, sample_prob):
        if primitive.startswith("cuboid"):
            subset_name = _subset_from_primitive(primitive, "cuboid")
            asset_paths = _glob_relative_paths(repo_root, [f"assets/cuboid/{subset_name}/*.urdf"])
        elif primitive.startswith("cylinder"):
            subset_name = _subset_from_primitive(primitive, "cylinder")
            asset_paths = _glob_relative_paths(repo_root, [f"assets/cylinder/{subset_name}/*.urdf"])
        elif primitive.startswith("custom"):
            subset_name = _subset_from_primitive(primitive, "custom")
            asset_paths = _glob_relative_paths(
                repo_root,
                [
                    f"assets/custom/{subset_name}/*.urdf",
                    f"assets/custom/{subset_name}/*/*.urdf",
                ],
            )
        else:
            object_type_list.append(primitive)
            object_type_prob.append(primitive_prob)
            continue

        if not asset_paths:
            raise FileNotFoundError(
                f"No URDF assets found for primitive '{primitive}' under repo root {repo_root}"
            )

        primitive_entries = [f"{primitive}_{i}" for i in range(len(asset_paths))]
        object_type_list.extend(primitive_entries)
        object_type_prob.extend([primitive_prob / len(primitive_entries) for _ in primitive_entries])
        for entry_name, asset_path in zip(primitive_entries, asset_paths):
            asset_files_dict[entry_name] = asset_path.replace(os.sep, "/")

    return object_type_list, object_type_prob, asset_files_dict


def _farthest_point_sample(points: np.ndarray, n: int) -> np.ndarray:
    selected = [0]
    distances = np.full(len(points), np.inf, dtype=np.float64)

    for _ in range(n - 1):
        last = points[selected[-1]]
        dist_to_last = np.linalg.norm(points - last, axis=1)
        distances = np.minimum(distances, dist_to_last)
        selected.append(int(np.argmax(distances)))

    return points[np.asarray(selected)].copy()


def _normalise_points(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    points = points - points.mean(axis=0, keepdims=True)
    radius = float(np.max(np.linalg.norm(points, axis=1)))
    if radius > 1e-10:
        points = points / radius
    return points.astype(np.float32)


def _analytic_cylinder_point_cloud(asset_path: Path, n_points: int) -> np.ndarray:
    tree = ET.parse(asset_path)
    cylinder = tree.find(".//cylinder")
    if cylinder is None:
        raise FileNotFoundError(f"No point cloud sidecar and no cylinder geometry in {asset_path}")

    radius = float(cylinder.attrib["radius"])
    length = float(cylinder.attrib["length"])
    dense_count = max(n_points * 8, 512)
    side_count = dense_count // 2
    cap_count = dense_count - side_count

    side_points = []
    for i in range(side_count):
        z = -length / 2.0 + length * ((i + 0.5) / side_count)
        theta = 2.0 * math.pi * ((i * 0.6180339887498949) % 1.0)
        side_points.append([radius * math.cos(theta), radius * math.sin(theta), z])

    cap_points = []
    for i in range(cap_count):
        theta = 2.0 * math.pi * ((i * 0.6180339887498949) % 1.0)
        r = radius * math.sqrt((i + 0.5) / cap_count)
        z = length / 2.0 if i % 2 == 0 else -length / 2.0
        cap_points.append([r * math.cos(theta), r * math.sin(theta), z])

    points = _normalise_points(np.asarray(side_points + cap_points, dtype=np.float32))
    return _farthest_point_sample(points, n_points).astype(np.float32)


def load_object_point_cloud(asset_file: str, n_points: int, repo_root: Path | None = None) -> np.ndarray:
    repo_root = REPO_ROOT if repo_root is None else Path(repo_root)
    asset_path = repo_root / asset_file
    for sidecar in (
        asset_path.parent / f"{asset_path.stem}_pointcloud_{n_points}.npy",
        asset_path.parent / f"pointcloud_{n_points}.npy",
    ):
        if sidecar.is_file():
            points = np.load(sidecar).astype(np.float32)
            if points.shape != (n_points, 3):
                raise ValueError(f"Expected {sidecar} to have shape {(n_points, 3)}, got {points.shape}")
            return points

    return _analytic_cylinder_point_cloud(asset_path, n_points)
