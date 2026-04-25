"""Quick visual check of object point clouds.

Usage:
    # Random sample from all custom assets
    python scripts/viz_pointcloud.py
    python scripts/viz_pointcloud.py --n-points 1024

    # Specific objects
    python scripts/viz_pointcloud.py assets/custom/cylinder_2dcross/Stage1_2Dcross_NEW_Rescaled_1
    python scripts/viz_pointcloud.py assets/custom/cylinder_2dcross/Stage1_2Dcross_NEW_Rescaled_*/
    python scripts/viz_pointcloud.py assets/cylinder/default
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

REPO_ROOT = Path(__file__).resolve().parents[1]


def find_pointclouds(paths: list[Path], n_points: int) -> list[Path]:
    results = []
    for p in paths:
        if p.is_dir():
            results.extend(sorted(p.rglob(f"pointcloud_{n_points}.npy")))
            results.extend(sorted(p.rglob(f"*_pointcloud_{n_points}.npy")))
        elif p.suffix == ".npy":
            results.append(p)
    return sorted(set(results))


def plot_pointclouds(npy_files: list[Path], n_points: int, max_plots: int = 16):
    n = min(len(npy_files), max_plots)
    cols = 4
    rows = (n + cols - 1) // cols
    fig = plt.figure(figsize=(4 * cols, 3.5 * rows))
    all_points = [np.load(path) for path in npy_files[:n]]
    stacked = np.concatenate(all_points, axis=0)
    coord_min = float(stacked.min())
    coord_max = float(stacked.max())
    center = (coord_min + coord_max) / 2.0
    half_range = max((coord_max - coord_min) / 2.0, 1e-6)
    axis_limits = (center - half_range, center + half_range)

    for i, (path, pts) in enumerate(zip(npy_files[:n], all_points)):
        ax = fig.add_subplot(rows, cols, i + 1, projection="3d")
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=2, c=pts[:, 2], cmap="viridis")
        ax.set_title(path.parent.name, fontsize=7)
        ax.set_xlim(*axis_limits)
        ax.set_ylim(*axis_limits)
        ax.set_zlim(*axis_limits)
        ax.set_box_aspect((1, 1, 1))
        ax.xaxis.set_major_locator(MaxNLocator(3))
        ax.yaxis.set_major_locator(MaxNLocator(3))
        ax.zaxis.set_major_locator(MaxNLocator(3))

    plt.suptitle(f"{n} point clouds ({n_points} points)", fontsize=10)
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path)
    parser.add_argument("--n-points", type=int, choices=(100, 1024), default=100)
    args = parser.parse_args()
    paths = args.paths or [REPO_ROOT / "assets" / "custom"]

    npy_files = find_pointclouds(paths, args.n_points)
    if not npy_files:
        parser.error(f"No {args.n_points}-point cloud .npy files found.")

    print(f"Found {len(npy_files)} point clouds, showing up to 16.")
    plot_pointclouds(npy_files, args.n_points)


if __name__ == "__main__":
    main()
