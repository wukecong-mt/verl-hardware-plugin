"""MUSA peak-FLOPS override used by VERL's MFU calculation."""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


def _install_device_flops_hook() -> None:
    """Use the configured MUSA BF16 peak when VERL calculates MFU."""
    from verl.utils import flops_counter

    original_get_device_flops = flops_counter.get_device_flops
    if getattr(original_get_device_flops, "_verl_musa_hook", False):
        return

    logged_peak_tflops = None
    unit_scale = {
        "B": 1e9,
        "K": 1e3,
        "M": 1e6,
        "G": 1e9,
        "T": 1e12,
        "P": 1e15,
    }

    def get_musa_device_flops(unit="T", device_name=None):
        nonlocal logged_peak_tflops

        value = os.getenv("MUSA_BF16_DENSE_TFLOPS")
        if value is None:
            return original_get_device_flops(unit=unit, device_name=device_name)

        try:
            peak_tflops = float(value)
        except ValueError:
            logger.warning("Invalid MUSA_BF16_DENSE_TFLOPS=%r; using verl's device FLOPS lookup", value)
            return original_get_device_flops(unit=unit, device_name=device_name)

        if peak_tflops <= 0:
            logger.warning("MUSA_BF16_DENSE_TFLOPS must be positive, got %s; using verl's lookup", peak_tflops)
            return original_get_device_flops(unit=unit, device_name=device_name)
        if unit not in unit_scale:
            raise ValueError(f"Unsupported FLOPS unit: {unit}")

        if logged_peak_tflops != peak_tflops:
            logger.info("MUSA MFU peak throughput: %s BF16 dense TFLOPS", peak_tflops)
            logged_peak_tflops = peak_tflops
        return peak_tflops * 1e12 / unit_scale[unit]

    get_musa_device_flops._verl_musa_hook = True
    flops_counter.get_device_flops = get_musa_device_flops


def install() -> None:
    """Install MUSA patches for VERL's MFU calculation."""
    _install_device_flops_hook()
