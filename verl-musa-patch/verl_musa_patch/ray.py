"""Ray compatibility patches for MUSA on older VERL checkouts."""

import os


_INSTALLED = False


def install() -> None:
    """Teach VERL's legacy Ray helper about the MUSA no-set variable."""
    global _INSTALLED
    if _INSTALLED:
        return

    from verl.utils import ray_utils

    original = ray_utils.ray_noset_visible_devices
    if getattr(original, "_verl_musa_patch", False):
        _INSTALLED = True
        return

    def musa_ray_noset_visible_devices(env_vars=os.environ):
        return bool(
            env_vars.get("RAY_EXPERIMENTAL_NOSET_MUSA_VISIBLE_DEVICES")
        ) or original(env_vars)

    musa_ray_noset_visible_devices._verl_musa_patch = True
    ray_utils.ray_noset_visible_devices = musa_ray_noset_visible_devices
    _INSTALLED = True
