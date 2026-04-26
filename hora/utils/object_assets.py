from __future__ import annotations

import os
from glob import glob
from pathlib import Path

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


def load_object_point_cloud(asset_file: str, n_points: int, repo_root: Path | None = None) -> np.ndarray:
    repo_root = REPO_ROOT if repo_root is None else Path(repo_root)
    asset_path = repo_root / asset_file
    sidecars = (
        asset_path.parent / f"{asset_path.stem}_pointcloud_{n_points}.npy",
        asset_path.parent / f"pointcloud_{n_points}.npy",
    )
    for sidecar in sidecars:
        if sidecar.is_file():
            points = np.load(sidecar).astype(np.float32)
            if points.shape != (n_points, 3):
                raise ValueError(f"Expected {sidecar} to have shape {(n_points, 3)}, got {points.shape}")
            return points

    expected = " or ".join(str(path.relative_to(repo_root)) for path in sidecars)
    raise FileNotFoundError(
        f"Missing point cloud sidecar for {asset_file}. Expected {expected}. "
        "Generate or add the sidecar before enabling shape privileged info."
    )
