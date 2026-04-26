"""Generate normalized point clouds for primitive cylinder URDF assets."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]


def farthest_point_sample(points: np.ndarray, n: int) -> np.ndarray:
    selected = [0]
    distances = np.full(len(points), np.inf, dtype=np.float64)

    for _ in range(n - 1):
        last = points[selected[-1]]
        dist_to_last = np.linalg.norm(points - last, axis=1)
        distances = np.minimum(distances, dist_to_last)
        selected.append(int(np.argmax(distances)))

    return points[np.asarray(selected)].copy()


def normalize_points(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    points = points - points.mean(axis=0, keepdims=True)
    radius = float(np.max(np.linalg.norm(points, axis=1)))
    if radius <= 1e-10:
        raise ValueError("Cannot normalize a degenerate point cloud")
    return (points / radius).astype(np.float32)


def cylinder_point_cloud(asset_path: Path, n_points: int) -> np.ndarray:
    tree = ET.parse(asset_path)
    cylinder = tree.find(".//cylinder")
    if cylinder is None:
        raise ValueError(f"No cylinder geometry found in {asset_path}")

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

    points = normalize_points(np.asarray(side_points + cap_points, dtype=np.float32))
    return farthest_point_sample(points, n_points).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "asset_dir",
        nargs="?",
        type=Path,
        default=REPO_ROOT / "assets" / "cylinder" / "default",
        help="Directory containing primitive cylinder URDF files.",
    )
    parser.add_argument("--n-points", type=int, choices=(100, 200, 300, 500, 1024), default=1024)
    args = parser.parse_args()

    asset_dir = args.asset_dir
    if not asset_dir.is_absolute():
        asset_dir = REPO_ROOT / asset_dir

    for urdf_path in sorted(asset_dir.glob("*.urdf")):
        points = cylinder_point_cloud(urdf_path, args.n_points)
        output_path = urdf_path.with_name(f"{urdf_path.stem}_pointcloud_{args.n_points}.npy")
        np.save(output_path, points)
        print(output_path.relative_to(REPO_ROOT))


if __name__ == "__main__":
    main()
