"""Quick visual check of 1024-point object point clouds.

Usage:
    # Random sample from all custom assets
    python scripts/viz_pointcloud.py

    # Specific objects
    python scripts/viz_pointcloud.py assets/custom/cylinder_2dcross/Stage1_2Dcross_NEW_Rescaled_1
    python scripts/viz_pointcloud.py assets/custom/cylinder_2dcross/Stage1_2Dcross_NEW_Rescaled_*/
    python scripts/viz_pointcloud.py assets/cylinder/default
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

REPO_ROOT = Path(__file__).resolve().parents[1]


def find_pointclouds(paths: list[Path]) -> list[Path]:
    results = []
    for p in paths:
        if p.is_dir():
            results.extend(sorted(p.rglob("pointcloud_1024.npy")))
            results.extend(sorted(p.rglob("*_pointcloud_1024.npy")))
        elif p.suffix == ".npy":
            results.append(p)
    return sorted(set(results))


def plot_pointclouds(npy_files: list[Path], max_plots: int = 16):
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

    plt.suptitle(f"{n} point clouds (pointcloud_1024.npy)", fontsize=10)
    plt.tight_layout()
    plt.show()


def main():
    if len(sys.argv) > 1:
        paths = [Path(p) for p in sys.argv[1:]]
    else:
        paths = [REPO_ROOT / "assets" / "custom"]

    npy_files = find_pointclouds(paths)
    if not npy_files:
        print("No 1024-point cloud .npy files found.")
        sys.exit(1)

    print(f"Found {len(npy_files)} point clouds, showing up to 16.")
    plot_pointclouds(npy_files)


if __name__ == "__main__":
    main()
