DEFAULT_FINGERTIP_LINK_NAMES = (
    "link_3.0_tip",
    "link_7.0_tip",
    "link_11.0_tip",
    "link_15.0_tip",
)
FALLBACK_FINGERTIP_LINK_NAMES = (
    "link_3.0",
    "link_7.0",
    "link_11.0",
    "link_15.0",
)
FALLBACK_FINGERTIP_BODY_INDICES = (4, 8, 12, 16)


def resolve_fingertip_body_indices(body_dict):
    if all(name in body_dict for name in DEFAULT_FINGERTIP_LINK_NAMES):
        return tuple(body_dict[name] for name in DEFAULT_FINGERTIP_LINK_NAMES)

    if all(name in body_dict for name in FALLBACK_FINGERTIP_LINK_NAMES):
        return tuple(body_dict[name] for name in FALLBACK_FINGERTIP_LINK_NAMES)

    available_indices = set(body_dict.values())
    if set(FALLBACK_FINGERTIP_BODY_INDICES).issubset(available_indices):
        return FALLBACK_FINGERTIP_BODY_INDICES

    available_names = ", ".join(sorted(body_dict))
    raise KeyError(
        "Could not resolve fingertip body indices from the loaded Allegro asset. "
        f"Available rigid bodies: {available_names}"
    )
