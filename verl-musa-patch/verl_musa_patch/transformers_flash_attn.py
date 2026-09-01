"""Teach Transformers' FlashAttention availability check about MUSA."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import os

from packaging import version


def _install_flash_attn_2_availability() -> None:
    """Enable the installed FlashAttention 2 package on an available MUSA device."""
    import torch
    import transformers.modeling_flash_attention_utils as flash_attention_utils
    import transformers.modeling_utils as modeling_utils
    import transformers.utils as transformers_utils
    import transformers.utils.import_utils as import_utils

    original_is_available = import_utils.is_flash_attn_2_available
    if getattr(original_is_available, "_verl_musa_hook", False):
        print(
            f"[VERL_MUSA_FA2 pid={os.getpid()}] availability patch already installed",
            flush=True,
        )
        return

    def is_flash_attn_2_available() -> bool:
        if original_is_available():
            return True

        musa = getattr(torch, "musa", None)
        is_musa_available = getattr(musa, "is_available", None)
        if not callable(is_musa_available) or not is_musa_available():
            return False
        if importlib.util.find_spec("flash_attn") is None:
            return False

        try:
            flash_attn_version = importlib.metadata.version("flash_attn")
            return version.parse(flash_attn_version) >= version.parse("2.1.0")
        except (importlib.metadata.PackageNotFoundError, version.InvalidVersion):
            return False

    is_flash_attn_2_available._verl_musa_hook = True

    # Transformers imports this helper into module globals rather than looking
    # it up through import_utils at each call, so update every runtime alias
    # used by model validation and lazy FlashAttention loading.
    import_utils.is_flash_attn_2_available = is_flash_attn_2_available
    transformers_utils.is_flash_attn_2_available = is_flash_attn_2_available
    modeling_utils.is_flash_attn_2_available = is_flash_attn_2_available
    flash_attention_utils.is_flash_attn_2_available = is_flash_attn_2_available
    try:
        fa2_spec = importlib.util.find_spec("flash_attn")
        fa2_version = importlib.metadata.version("flash_attn")
    except Exception as exc:
        fa2_spec = f"ERROR:{type(exc).__name__}:{exc}"
        fa2_version = "unknown"
    print(
        f"[VERL_MUSA_FA2 pid={os.getpid()}] availability patch installed "
        f"before={original_is_available()} after={is_flash_attn_2_available()} "
        f"version={fa2_version} origin={getattr(fa2_spec, 'origin', fa2_spec)}",
        flush=True,
    )


def install() -> None:
    """Install MUSA compatibility patches for Transformers FlashAttention."""
    _install_flash_attn_2_availability()
