"""Quick visual check of pointcloud_100.npy files.

Usage:
    # Random sample from all custom assets
    python scripts/viz_pointcloud.py

    # Specific objects
    python scripts/viz_pointcloud.py assets/custom/cylinder_2dcross/Stage1_2Dcross_NEW_Rescaled_1
    python scripts/viz_pointcloud.py assets/custom/cylinder_2dcross/Stage1_2Dcross_NEW_Rescaled_*/
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

REPO_ROOT = Path(__file__).resolve().parents[1]


def find_pointclouds(paths: list[Path]) -> list[Path]:
    results = []
    for p in paths:
        if p.is_dir():
            results.extend(sorted(p.rglob("pointcloud_1024.npy")))
        elif p.suffix == ".npy":
            results.append(p)
    return results


def plot_pointclouds(npy_files: list[Path], max_plots: int = 16):
    n = min(len(npy_files), max_plots)
    cols = 4
    rows = (n + cols - 1) // cols
    fig = plt.figure(figsize=(4 * cols, 3.5 * rows))

    for i, path in enumerate(npy_files[:n]):
        pts = np.load(path)
        ax = fig.add_subplot(rows, cols, i + 1, projection="3d")
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=2, c=pts[:, 2], cmap="viridis")
        ax.set_title(path.parent.name, fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

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
        print("No pointcloud_100.npy files found.")
        sys.exit(1)

    print(f"Found {len(npy_files)} point clouds, showing up to 16.")
    plot_pointclouds(npy_files)


if __name__ == "__main__":
    main()
