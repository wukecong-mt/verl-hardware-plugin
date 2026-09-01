"""Runtime compatibility patches used by VERL on MUSA."""

_APPLIED = False


def _install_te_compute_capability() -> None:
    import transformer_engine.pytorch.utils as te_utils

    te_utils.get_device_compute_capability = lambda: (8, 0)


def _install_te_moe_permutation() -> None:
    import transformer_engine.pytorch.triton.permutation as te_permutation

    if getattr(te_permutation.sort_chunks_by_idx, "__module__", "").startswith(
        "musa_patch."
    ):
        return

    from .transformer_engine_moe_permutation import install

    install()


def apply() -> None:
    """Install the small set of MUSA patches used by VERL workloads."""
    global _APPLIED
    if _APPLIED:
        return

    from .sglang import install as install_sglang

    # Extend SGLang with MUSA-safe HTTP, weight-sync, and MTP IPC lifetimes.
    install_sglang()

    from .ray import install as install_ray

    # VERL v0.9's legacy Ray helper has a hard-coded no-set variable list.
    install_ray()

    # Explicitly relative: this is verl_musa_patch.flash_attn, not the
    # third-party top-level flash_attn package.
    from .flash_attn import install as install_flash_attn

    # Route supported Transformer Engine varlen attention calls through MATE.
    install_flash_attn()

    from .nested_tensor import install as install_nested_tensor

    # Add the missing MUSA kernel fallback for jagged-to-padded NestedTensor conversion.
    install_nested_tensor()

    from .device_flops import install as install_device_flops

    # Supply MUSA's configured BF16 peak FLOPS for VERL MFU reporting.
    install_device_flops()

    from .transformers_flash_attn import install as install_transformers_flash_attn

    # Let Transformers recognize the installed FlashAttention 2 package on MUSA.
    install_transformers_flash_attn()

    # Report a TE-compatible compute capability for MUSA kernel selection.
    _install_te_compute_capability()

    from .transformer_engine_rmsnorm import install as install_te_rmsnorm

    # Correct zero-centered-gamma handling in MUSA Transformer Engine RMSNorm.
    install_te_rmsnorm()

    # from .logits_memory_snapshot import install as install_logits_memory_snapshot
    # Enable the opt-in MUSA allocator snapshot around Megatron logits computation.
    # install_logits_memory_snapshot()

    # Install MUSA fast paths for Transformer Engine MoE permutation metadata.
    _install_te_moe_permutation()

    from .autocast import install as install_autocast

    # Restore the real MUSA autocast state query after all TE imports/patches.
    install_autocast()

    from .megatron_engine import install as install_megatron_engine

    # Keep MUSA-specific Megatron lifecycle hooks out of the engine registry.
    install_megatron_engine()

    import torch

    torch.backends.mudnn.allow_tf32 = False

    _APPLIED = True
