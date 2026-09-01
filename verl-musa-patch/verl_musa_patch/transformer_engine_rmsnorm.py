"""MUSA TransformerEngine RMSNorm compatibility patches."""

from __future__ import annotations

_PATCHED = False


def patch_rmsnorm_zero_centered_gamma() -> bool:
    """Apply zero-centered gamma in Python when the MUSA TE kernel ignores it.

    TransformerEngine stores zero-centered RMSNorm weights around zero and is
    expected to use ``1 + weight`` as the effective scale. The affected MUSA
    kernels accept the flag but do not apply that offset, so materialize the
    effective scale before calling them and clear the flag.

    The patch covers the low-level extension, fused TE modules that cache
    ``apply_normalization``, and the standalone RMSNorm op. It is idempotent.
    """
    global _PATCHED
    if _PATCHED:
        return True

    try:
        import transformer_engine_torch as tex
    except ImportError:
        return False

    if not hasattr(tex, "rmsnorm_fwd") or not hasattr(tex, "rmsnorm_bwd"):
        return False

    if getattr(tex.rmsnorm_fwd, "_musa_zero_centered_gamma_fix", False) or getattr(
        tex.rmsnorm_fwd, "_slime_zero_centered_gamma_fix", False
    ):
        _PATCHED = True
        return True

    original_fwd = tex.rmsnorm_fwd
    original_bwd = tex.rmsnorm_bwd

    def rmsnorm_fwd(input_, weight, eps, ln_out, quantizer, otype, sm_margin, zero_centered_gamma):
        if zero_centered_gamma:
            weight = weight + 1
        return original_fwd(input_, weight, eps, ln_out, quantizer, otype, sm_margin, False)

    def rmsnorm_bwd(dz, x, rsigma, gamma, sm_margin, zero_centered_gamma):
        if zero_centered_gamma:
            gamma = gamma + 1
        return original_bwd(dz, x, rsigma, gamma, sm_margin, False)

    rmsnorm_fwd._musa_zero_centered_gamma_fix = True
    rmsnorm_bwd._musa_zero_centered_gamma_fix = True
    tex.rmsnorm_fwd = rmsnorm_fwd
    tex.rmsnorm_bwd = rmsnorm_bwd

    # TE fused modules cache this Python symbol at import time.
    def patch_fused_apply_normalization(module):
        original_apply = module.apply_normalization

        def apply_normalization(
            inputmat,
            ln_out,
            ln_weight,
            ln_bias,
            eps,
            output_quantizer,
            output_dtype,
            normalization,
            fwd_ln_sm_margin,
            zero_centered_gamma,
        ):
            if normalization == "RMSNorm" and zero_centered_gamma:
                ln_weight = ln_weight + 1
                zero_centered_gamma = False
            return original_apply(
                inputmat,
                ln_out,
                ln_weight,
                ln_bias,
                eps,
                output_quantizer,
                output_dtype,
                normalization,
                fwd_ln_sm_margin,
                zero_centered_gamma,
            )

        module.apply_normalization = apply_normalization

    from transformer_engine.pytorch.module import layernorm_linear

    patch_fused_apply_normalization(layernorm_linear)
    try:
        from transformer_engine.pytorch.module import layernorm_mlp

        patch_fused_apply_normalization(layernorm_mlp)
    except ImportError:
        pass

    # The standalone TE RMSNorm op imports these symbols by value.
    try:
        from transformer_engine.pytorch.ops.basic import rmsnorm as rmsnorm_op

        rmsnorm_op.rmsnorm_fwd = rmsnorm_fwd
        rmsnorm_op.rmsnorm_bwd = rmsnorm_bwd
    except ImportError:
        pass

    _PATCHED = True
    print("[musa_patch] patched TE RMSNorm zero_centered_gamma forward/backward", flush=True)
    return True


def install() -> None:
    """Install MUSA compatibility patches for Transformer Engine RMSNorm."""
    patch_rmsnorm_zero_centered_gamma()
