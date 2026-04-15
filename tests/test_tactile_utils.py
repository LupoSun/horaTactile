import pytest

from hora.utils.tactile_utils import resolve_fingertip_body_indices


def test_resolve_fingertip_body_indices_prefers_explicit_tip_links():
    body_dict = {
        "link_3.0_tip": 17,
        "link_7.0_tip": 18,
        "link_11.0_tip": 19,
        "link_15.0_tip": 20,
        "base_link": 0,
    }

    assert resolve_fingertip_body_indices(body_dict) == (17, 18, 19, 20)


def test_resolve_fingertip_body_indices_falls_back_to_parent_links():
    body_dict = {
        "link_3.0": 4,
        "link_7.0": 8,
        "link_11.0": 12,
        "link_15.0": 16,
        "base_link": 0,
    }

    assert resolve_fingertip_body_indices(body_dict) == (4, 8, 12, 16)


def test_resolve_fingertip_body_indices_falls_back_to_known_body_indices():
    body_dict = {
        "finger_a": 4,
        "finger_b": 8,
        "finger_c": 12,
        "finger_d": 16,
        "base_link": 0,
    }

    assert resolve_fingertip_body_indices(body_dict) == (4, 8, 12, 16)


def test_resolve_fingertip_body_indices_raises_when_no_fallback_matches():
    with pytest.raises(KeyError):
        resolve_fingertip_body_indices({"base_link": 0, "link_1.0": 1})
