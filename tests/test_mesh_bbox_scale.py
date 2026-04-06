import numpy as np
import pytest

from tools.mesh.scale_by_bbox import (
    compute_uniform_bbox_scale,
    get_bbox_reference_extent,
    scale_extents,
)


def test_get_bbox_reference_extent_uses_longest_edge_for_max():
    extents = np.array([0.02, 0.05, 0.03], dtype=np.float64)
    assert get_bbox_reference_extent(extents, axis="max") == pytest.approx(0.05)


def test_compute_uniform_bbox_scale_scales_longest_edge_to_target():
    extents = np.array([0.02, 0.05, 0.03], dtype=np.float64)
    scale_factor = compute_uniform_bbox_scale(extents, target_size=0.10, axis="max")

    assert scale_factor == pytest.approx(2.0)
    assert np.allclose(scale_extents(extents, scale_factor), [0.04, 0.10, 0.06])


def test_compute_uniform_bbox_scale_scales_specific_axis_to_target():
    extents = np.array([0.02, 0.05, 0.03], dtype=np.float64)
    scale_factor = compute_uniform_bbox_scale(extents, target_size=0.06, axis="z")

    assert scale_factor == pytest.approx(2.0)
    assert np.allclose(scale_extents(extents, scale_factor), [0.04, 0.10, 0.06])


def test_compute_uniform_bbox_scale_rejects_non_positive_target():
    with pytest.raises(ValueError):
        compute_uniform_bbox_scale([1.0, 2.0, 3.0], target_size=0.0)


def test_compute_uniform_bbox_scale_rejects_zero_reference_extent():
    with pytest.raises(ValueError):
        compute_uniform_bbox_scale([0.0, 0.0, 0.0], target_size=1.0, axis="max")
