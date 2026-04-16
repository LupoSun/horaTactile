from pathlib import Path

import pytest

from hora.utils.object_assets import build_object_asset_catalog


def test_build_object_asset_catalog_supports_custom_nested_assets(tmp_path):
    repo_root = tmp_path / "repo"
    asset_dir = repo_root / "assets" / "custom" / "btg13_mean" / "BTG_13"
    asset_dir.mkdir(parents=True)
    (asset_dir / "BTG_13.urdf").write_text("<robot />")

    object_type_list, object_type_prob, asset_files_dict = build_object_asset_catalog(
        "custom_btg13_mean",
        [1.0],
        repo_root=repo_root,
    )

    assert object_type_list == ["custom_btg13_mean_0"]
    assert object_type_prob == [1.0]
    assert asset_files_dict["custom_btg13_mean_0"] == "assets/custom/btg13_mean/BTG_13/BTG_13.urdf"


def test_build_object_asset_catalog_supports_mixed_primitives(tmp_path):
    repo_root = tmp_path / "repo"
    cylinder_dir = repo_root / "assets" / "cylinder" / "default"
    custom_dir = repo_root / "assets" / "custom" / "demo" / "ShapeA"
    cylinder_dir.mkdir(parents=True)
    custom_dir.mkdir(parents=True)
    (cylinder_dir / "0000.urdf").write_text("<robot />")
    (custom_dir / "ShapeA.urdf").write_text("<robot />")

    object_type_list, object_type_prob, asset_files_dict = build_object_asset_catalog(
        "cylinder_default+custom_demo",
        [0.25, 0.75],
        repo_root=repo_root,
    )

    assert object_type_list == ["cylinder_default_0", "custom_demo_0"]
    assert object_type_prob == [0.25, 0.75]
    assert asset_files_dict["cylinder_default_0"] == "assets/cylinder/default/0000.urdf"
    assert asset_files_dict["custom_demo_0"] == "assets/custom/demo/ShapeA/ShapeA.urdf"


def test_build_object_asset_catalog_rejects_missing_assets(tmp_path):
    repo_root = tmp_path / "repo"
    with pytest.raises(FileNotFoundError):
        build_object_asset_catalog("custom_missing", [1.0], repo_root=repo_root)
