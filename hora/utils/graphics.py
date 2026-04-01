def resolve_graphics_device_id(graphics_device_id: int, headless: bool, enable_camera_sensors: bool) -> int:
    if not enable_camera_sensors and headless:
        return -1
    return graphics_device_id
