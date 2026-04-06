import json
from pathlib import Path

import numpy as np
import pytest
import trimesh

from tools.mesh.preprocess import process_mesh


def test_process_mesh_scales_exported_assets_but_keeps_pointclouds_normalized(tmp_path):
    input_mesh = trimesh.creation.box(extents=(1.0, 2.0, 4.0))
    input_path = tmp_path / "box.obj"
    input_mesh.export(input_path)

    output_dir = tmp_path / "assets"
    obj_dir = process_mesh(
        mesh_path=input_path,
        output_dir=output_dir,
        point_counts=[100, 1024],
        skip_decomposition=True,
        mass=0.05,
        target_bbox_size=0.2,
        target_bbox_axis="max",
    )

    visual_path = obj_dir / "visual.obj"
    urdf_path = obj_dir / "box.urdf"
    pointcloud_100_path = obj_dir / "pointcloud_100.npy"
    pointcloud_1024_path = obj_dir / "pointcloud_1024.npy"
    metadata_path = obj_dir / "metadata.json"

    assert visual_path.exists()
    assert urdf_path.exists()
    assert pointcloud_100_path.exists()
    assert pointcloud_1024_path.exists()
    assert metadata_path.exists()

    visual_mesh = trimesh.load(visual_path, force="mesh")
    visual_extents = np.asarray(visual_mesh.bounding_box.extents, dtype=np.float64)
    assert np.max(visual_extents) == pytest.approx(0.2, abs=1e-6)

    pointcloud_100 = np.load(pointcloud_100_path)
    pointcloud_1024 = np.load(pointcloud_1024_path)
    assert pointcloud_100.shape == (100, 3)
    assert pointcloud_1024.shape == (1024, 3)
    assert np.max(np.linalg.norm(pointcloud_100, axis=1)) <= 1.000001
    assert np.max(np.linalg.norm(pointcloud_1024, axis=1)) <= 1.000001

    metadata = json.loads(metadata_path.read_text())
    assert metadata["export_bbox_target_size"] == 0.2
    assert metadata["export_bbox_target_axis"] == "max"
    assert metadata["point_cloud_space"] == "unit_sphere"
    assert max(metadata["export_bbox"]) == pytest.approx(0.2, abs=1e-6)
    expected_radius = np.linalg.norm(np.array([0.5, 1.0, 2.0], dtype=np.float64))
    expected_normalised_bbox = (np.array([1.0, 2.0, 4.0], dtype=np.float64) / expected_radius).tolist()
    assert metadata["normalised_bbox"] == pytest.approx(expected_normalised_bbox, abs=1e-6)
