"""Runtime compatibility patches used by VERL on MUSA."""

_APPLIED = False


def apply() -> None:
    """Install the small set of MUSA patches used by VERL workloads."""
    global _APPLIED
    if _APPLIED:
        return

    from .sglang import install as install_sglang

    install_sglang()

    # Explicitly relative: this is verl_musa_patch.flash_attn, not the
    # third-party top-level flash_attn package.
    from .flash_attn import install as install_flash_attn

    install_flash_attn()

    from .nested_tensor import install_jagged_to_padded_dense

    install_jagged_to_padded_dense()

    import transformer_engine.pytorch.utils as te_utils

    te_utils.get_device_compute_capability = lambda: (8, 0)

    from .transformer_engine_rmsnorm import patch_rmsnorm_zero_centered_gamma

    patch_rmsnorm_zero_centered_gamma()

    from .logits_memory_snapshot import install as install_logits_memory_snapshot

    install_logits_memory_snapshot()

    import transformer_engine.pytorch.triton.permutation as te_permutation

    if not getattr(te_permutation.sort_chunks_by_idx, "__module__", "").startswith(
        "musa_patch."
    ):
        from .transformer_engine_moe_permutation import (
            patch_transformer_engine_moe_permutation,
        )

        patch_transformer_engine_moe_permutation()

    _APPLIED = True
