"""MUSA compatibility for Megatron/TE FlashAttention paths."""

from __future__ import annotations

import importlib
import inspect
import os

import torch


def _install_mla_asymmetric_v(debug: bool) -> None:
    """Keep DeepSeek MLA's native QK=192/V=128 layout on MUSA THD."""
    # FSDP/SGLang workers do not use Megatron MLA.  In particular, a global
    # Megatron installation may be visible in the environment even when no
    # Megatron engine is selected; do not import or modify it in that case.
    if not os.getenv("MUSA_PATCH_PATH", "").strip():
        print(
            "[VERL_MUSA_FLASH_ATTN] skip MLA patch: MUSA_PATCH_PATH is not set",
            flush=True,
        )
        return

    try:
        mcore_mla = importlib.import_module(
            "megatron.core.transformer.multi_latent_attention"
        )
    except (ImportError, ModuleNotFoundError) as exc:
        print(
            "[VERL_MUSA_FLASH_ATTN] skip MLA patch: MCore MLA import failed: "
            f"{type(exc).__name__}: {exc}",
            flush=True,
        )
        return
    if getattr(mcore_mla, "_verl_musa_asymmetric_v_installed", False):
        return

    original_prepare_mla_value = getattr(
        mcore_mla, "_prepare_mla_core_attention_value", None
    )
    if original_prepare_mla_value is None:
        print(
            "[VERL_MUSA_FLASH_ATTN] skip MLA patch: "
            "MCore has no _prepare_mla_core_attention_value",
            flush=True,
        )
        return

    def _musa_prepare_mla_value(parallel_attention, query, value, packed_seq_params):
        if (
            value is not None
            and value.device.type == "musa"
            and packed_seq_params is not None
            and packed_seq_params.qkv_format == "thd"
            and query.shape[-1] == 192
            and value.shape[-1] == 128
        ):
            if debug and not getattr(_musa_prepare_mla_value, "_trace_printed", False):
                print(
                    "[VERL_MUSA_MLA_TRACE] preserving asymmetric THD heads",
                    "qk_dim=", query.shape[-1],
                    "v_dim=", value.shape[-1],
                    flush=True,
                )
                _musa_prepare_mla_value._trace_printed = True
            return value, False, value.shape[-1], value.shape[-1]
        return original_prepare_mla_value(
            parallel_attention, query, value, packed_seq_params
        )

    mcore_mla._prepare_mla_core_attention_value = _musa_prepare_mla_value
    mcore_mla._verl_musa_asymmetric_v_installed = True


def _install_attention_debug(debug: bool) -> None:
    """Print the effective TE attention decision once per attention module."""
    te_attention = importlib.import_module("transformer_engine.pytorch.attention")
    original = te_attention.DotProductAttention.forward
    if getattr(original, "_verl_musa_attention_debug", False):
        return

    def wrapped(self, query, key, value, *args, **kwargs):
        if debug and not getattr(self, "_verl_musa_attention_debug_printed", False):
            packed = kwargs.get("packed_seq_params")
            flash_impl = getattr(self, "flash_attention", None)
            flash_func = getattr(flash_impl, "__func__", flash_impl)
            flash_code = getattr(flash_func, "__code__", None)
            print(
                "[VERL_MUSA_ATTENTION_DEBUG] TE DotProductAttention.forward:",
                "class=", type(self).__name__,
                "attention_backend=",
                getattr(getattr(self, "config", None), "attention_backend", None),
                "qkv_format=", getattr(self, "qkv_format", None),
                "query=", tuple(query.shape),
                "key=", tuple(key.shape),
                "value=", tuple(value.shape),
                "packed_seq_params=",
                type(packed).__name__ if packed is not None else None,
                "flash_attention=", type(flash_impl).__name__,
                "flash_attention_module=", getattr(flash_func, "__module__", None),
                "flash_attention_file=", getattr(flash_code, "co_filename", None),
                flush=True,
            )
            self._verl_musa_attention_debug_printed = True
        return original(self, query, key, value, *args, **kwargs)

    wrapped._verl_musa_attention_debug = True
    te_attention.DotProductAttention.forward = wrapped


def _install_mate_varlen_attention(debug: bool) -> None:
    # MATE is an optional replacement route.  When it is disabled, do not
    # import its package: the native flash-attn ABI remains the active path.
    if os.getenv("SLIME_PATCH_MUSA_FLASH_ATTN_MATE", "0") != "1":
        if debug:
            print(
                "[VERL_MUSA_FA_TRACE] MATE disabled; native flash-attn path remains active",
                "SLIME_PATCH_MUSA_FLASH_ATTN_MATE=",
                os.getenv("SLIME_PATCH_MUSA_FLASH_ATTN_MATE", "0"),
                flush=True,
            )
        return
    flash_attn_api = importlib.import_module("flash_attn.flash_attn_interface")
    from mate.flash_attention.tilelang.flash_attention_varlen_bwd import (
        flashattn_varlen_bwd_interface,
    )
    from mate.mha_interface import flash_attn_varlen_func as mate_varlen_func

    original_fwd = flash_attn_api._flash_attn_varlen_forward
    original_bwd = flash_attn_api._flash_attn_varlen_backward
    if debug:
        print(
            "[VERL_MUSA_FA_TRACE] MATE installed",
            "flash_attn_interface=", inspect.getfile(flash_attn_api),
            "native_varlen_forward=", getattr(original_fwd, "__module__", None),
            "native_varlen_file=", inspect.getsourcefile(original_fwd),
            "mate_varlen=", getattr(mate_varlen_func, "__module__", None),
            "mate_varlen_file=", inspect.getsourcefile(mate_varlen_func),
            flush=True,
        )

    def use_mate(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> bool:
        return (
            os.getenv("SLIME_PATCH_MUSA_FLASH_ATTN_MATE", "0") == "1"
            and q.device.type == "musa"
            and k.device.type == "musa"
            and v.device.type == "musa"
            and q.ndim == k.ndim == v.ndim == 3
            # [MUSA plugin change] MATE TileLang supports both head_dim=128
            # (Qwen3-8B/Qwen3 family) and head_dim=256 (Qwen3.5 variants).
            and q.shape[-1] == k.shape[-1] == v.shape[-1] in (128, 256)
        )

    def varlen_forward(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        softmax_scale,
        causal,
        window_size=(-1, -1),
        softcap=0.0,
        alibi_slopes=None,
        return_softmax=False,
        block_table=None,
        leftpad_k=None,
        seqused_k=None,
    ):
        if debug and not getattr(varlen_forward, "_trace_printed", False):
            print(
                "[VERL_MUSA_FA_TRACE] varlen_forward",
                "q=", tuple(q.shape), "k=", tuple(k.shape), "v=", tuple(v.shape),
                "q_device=", q.device, "q_dtype=", q.dtype,
                "cu_q=", tuple(cu_seqlens_q.shape), "cu_kv=", tuple(cu_seqlens_k.shape),
                "mate_selected=", use_mate(q, k, v),
                flush=True,
            )
            varlen_forward._trace_printed = True
        if not use_mate(q, k, v):
            return original_fwd(
                q,
                k,
                v,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                dropout_p,
                softmax_scale,
                causal,
                window_size=window_size,
                softcap=softcap,
                alibi_slopes=alibi_slopes,
                return_softmax=return_softmax,
                block_table=block_table,
                leftpad_k=leftpad_k,
                seqused_k=seqused_k,
            )
        if dropout_p != 0.0 or softcap != 0.0 or alibi_slopes is not None:
            raise NotImplementedError(
                "Mate head_dim=256 adapter requires dropout=0, softcap=0, and no ALiBi"
            )
        if block_table is not None or leftpad_k is not None or seqused_k is not None:
            raise NotImplementedError(
                "Mate head_dim=256 adapter does not support paged or left-padded KV inputs"
            )

        out, softmax_lse = mate_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=int(max_seqlen_q),
            max_seqlen_k=int(max_seqlen_k),
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            softcap=0.0,
            deterministic=False,
            return_softmax_lse=True,
            backend="mutlass",
        )
        # flash-attn 2.6 private ABI expected by Transformer Engine:
        # out, q, k, v, out_padded, softmax_lse, S_dmask, rng_state.
        rng_state = torch.empty(0, dtype=torch.uint8, device=q.device)
        return out, q, k, v, out, softmax_lse, None, rng_state

    def varlen_backward(
        dout,
        q,
        k,
        v,
        out,
        softmax_lse,
        dq,
        dk,
        dv,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        softcap,
        alibi_slopes,
        deterministic,
        rng_state=None,
    ):
        if not use_mate(q, k, v):
            return original_bwd(
                dout,
                q,
                k,
                v,
                out,
                softmax_lse,
                dq,
                dk,
                dv,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                dropout_p,
                softmax_scale,
                causal,
                window_size,
                softcap,
                alibi_slopes,
                deterministic,
                rng_state=rng_state,
            )
        if dropout_p != 0.0 or softcap != 0.0 or alibi_slopes is not None:
            raise NotImplementedError(
                "Mate head_dim=256 adapter requires dropout=0, softcap=0, and no ALiBi"
            )

        mate_dq, mate_dk, mate_dv = flashattn_varlen_bwd_interface(
            q,
            k,
            v,
            out,
            dout,
            softmax_lse.contiguous(),
            int(max_seqlen_q),
            int(max_seqlen_k),
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            is_causal=causal,
            smscale=softmax_scale,
            deterministic=deterministic,
        )
        # MUSA TileLang's GQA reduction can run on an internal stream.  The
        # returned tensors must be complete before TE copies them into the
        # autograd-provided buffers; otherwise later layer work changes the
        # observed gradient depending on incidental synchronization.
        if os.getenv("MUSA_MATE_BWD_SYNC", "0") == "1":
            torch.musa.synchronize()
        dq.copy_(mate_dq)
        dk.copy_(mate_dk)
        dv.copy_(mate_dv)
        return None

    # Keep flash_attn's module globals untouched. Its public
    # flash_attn_varlen_func expects the standard four-value private ABI,
    # whereas Transformer Engine 2.6 consumes the older eight-value ABI below.
    # Replacing the module globals would therefore break Transformers/FSDP.
    # Transformer Engine snapshots these private functions at module import,
    # so patch only its aliases.
    import transformer_engine.pytorch.attention as te_attention

    te_attention._flash_attn_varlen_fwd = varlen_forward
    te_attention._flash_attn_varlen_bwd = varlen_backward

    # The MUSA THD correction extension currently requires its destination LSE
    # to be float64, while this TE Python version initializes that tensor as
    # float32.  Use a short-lived float64 buffer only for this correction and
    # copy the result back for the remaining forward/backward path.
    original_second_half_lse_correction = te_attention.tex.thd_second_half_lse_correction

    def second_half_lse_correction(lse, lse_per_step, cu_seqlens, lse_packed):
        if lse.device.type == "musa" and lse.dtype == torch.float32:
            lse_f64 = lse.to(torch.float64)
            original_second_half_lse_correction(
                lse_f64, lse_per_step, cu_seqlens, lse_packed
            )
            lse.copy_(lse_f64)
            return None
        return original_second_half_lse_correction(
            lse, lse_per_step, cu_seqlens, lse_packed
        )

    te_attention.tex.thd_second_half_lse_correction = second_half_lse_correction
    print("[VERL_MUSA_FLASH_ATTN] MATE varlen head_dim=256 adapter installed")


def install() -> None:
    """Install MUSA compatibility patches for FlashAttention."""
    debug = os.getenv("VERL_MUSA_ATTENTION_DEBUG", "0").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    _install_attention_debug(debug)
    _install_mla_asymmetric_v(debug)
    _install_mate_varlen_attention(debug)
