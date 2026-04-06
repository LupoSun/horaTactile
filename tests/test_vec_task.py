from hora.utils.graphics import resolve_graphics_device_id


def test_resolve_graphics_device_id_disables_rendering_for_headless_no_camera():
    assert resolve_graphics_device_id(3, headless=True, enable_camera_sensors=False) == -1


def test_resolve_graphics_device_id_preserves_requested_id_when_camera_sensors_are_enabled():
    assert resolve_graphics_device_id(3, headless=True, enable_camera_sensors=True) == 3


def test_resolve_graphics_device_id_preserves_requested_id_when_not_headless():
    assert resolve_graphics_device_id(1, headless=False, enable_camera_sensors=False) == 1
