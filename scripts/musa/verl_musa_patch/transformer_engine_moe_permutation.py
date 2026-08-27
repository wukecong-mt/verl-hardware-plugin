"""MUSA fast paths for TransformerEngine fused MoE permutation metadata."""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

_PATCHED = False
_SORT_HIT_LOGGED = False
_SORT_FALLBACK_LOGGED = False
_ROW_HIT_LOGGED = False
_MASK_BWD_HIT_LOGGED = False
_ORIGINAL_SORT = None
_ORIGINAL_MAKE_ROW_ID_MAP = None
_ORIGINAL_MOE_PERMUTE_MASK = None
_ROW_BLOCK_EXPERT = 16
_ROW_BLOCK_TOKEN = 64


@triton.jit
def _build_sort_metadata_kernel(
    split_sizes_ptr,
    sorted_indices_ptr,
    input_ends_ptr,
    output_bases_ptr,
    num_splits,
    NUM_SPLITS_PAD: tl.constexpr,
):
    """Build all chunk metadata with one small device launch.

    ``output_bases`` is indexed by the input chunk id.  Building that layout
    here removes argsort/index_select and both framework cumsum launches from
    every sort invocation.
    """
    offset = tl.arange(0, NUM_SPLITS_PAD)
    valid = offset < num_splits
    split_size = tl.load(split_sizes_ptr + offset, mask=valid, other=0).to(tl.int64)
    input_end = tl.cumsum(split_size)
    tl.store(input_ends_ptr + offset, input_end, mask=valid)

    output_chunk = tl.load(sorted_indices_ptr + offset, mask=valid, other=0).to(tl.int64)
    output_size = tl.load(split_sizes_ptr + output_chunk, mask=valid, other=0).to(tl.int64)
    output_end = tl.cumsum(output_size)
    output_base = output_end - output_size
    # sorted_indices is a permutation, so these stores never alias.
    tl.store(output_bases_ptr + output_chunk, output_base, mask=valid)


@triton.jit
def _sort_chunks_binary_kernel(
    input_ptr, input_ends_ptr, output_bases_ptr, output_ptr, dst_rows_ptr,
    probs_ptr, permuted_probs_ptr, num_splits, hidden_size,
    stride_input_token, stride_input_hidden, stride_output_token,
    stride_output_hidden, stride_probs_token, stride_permuted_probs_token,
    PERMUTE_PROBS: tl.constexpr, BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    lo = 0
    hi = num_splits
    # Specialized for at most 256 splits: 257 boundary positions need 9 steps.
    for _ in range(9):
        mid = (lo + hi) // 2
        boundary = tl.load(input_ends_ptr + mid).to(tl.int64)
        if boundary <= pid:
            lo = mid + 1
        else:
            hi = mid
    input_chunk = lo
    # tl.where evaluates both operands, so clamp the load instead of reading
    # input_ends[-1] for the first chunk.
    previous_chunk = tl.maximum(input_chunk - 1, 0)
    previous_end = tl.load(input_ends_ptr + previous_chunk).to(tl.int64)
    input_base = tl.where(input_chunk == 0, 0, previous_end)
    output_base = tl.load(output_bases_ptr + input_chunk).to(tl.int64)
    dst = output_base + pid - input_base
    tl.store(dst_rows_ptr + pid, dst)

    cur = 0
    while cur < hidden_size:
        off = cur + tl.arange(0, BLOCK_SIZE)
        mask = off < hidden_size
        value = tl.load(
            input_ptr + pid * stride_input_token + off * stride_input_hidden,
            mask=mask,
        )
        tl.store(
            output_ptr + dst * stride_output_token + off * stride_output_hidden,
            value,
            mask=mask,
        )
        cur += BLOCK_SIZE

    if PERMUTE_PROBS:
        prob = tl.load(probs_ptr + pid * stride_probs_token)
        tl.store(permuted_probs_ptr + dst * stride_permuted_probs_token, prob)


@triton.jit
def _row_id_map_pass_2_tiled_kernel(
    row_id_map_ptr, row_id_map_non_trans_ptr, workspace_prefix_ptr,
    num_experts, num_tokens, blocks_per_expert,
    PASS1_BLOCK_SIZE: tl.constexpr,
    BLOCK_EXPERT: tl.constexpr,
    BLOCK_TOKEN: tl.constexpr,
):
    """Apply block prefixes and produce the transposed map in 2-D tiles.

    The old layout launched one program per (expert, 256-token block).  Its
    transpose store advanced by ``num_experts`` int64 elements between lanes.
    With 256 experts that is a 2048-byte stride.  This kernel handles an
    expert-by-token tile and explicitly transposes the register tile so both
    output layouts are contiguous along one tile dimension.
    """
    expert = tl.program_id(0) * BLOCK_EXPERT + tl.arange(0, BLOCK_EXPERT)
    token_start = tl.program_id(1) * BLOCK_TOKEN
    token = token_start + tl.arange(0, BLOCK_TOKEN)
    expert_2d = expert[:, None]
    token_2d = token[None, :]
    valid = (expert_2d < num_experts) & (token_2d < num_tokens)

    # BLOCK_TOKEN divides PASS1_BLOCK_SIZE, so every token in this tile uses
    # the same pass-1 workspace block for a given expert.
    pass1_token_block = token_start // PASS1_BLOCK_SIZE
    chunk = expert * blocks_per_expert + pass1_token_block
    # prefix[1:] stores the inclusive cumsum of workspace.  prefix[0] is
    # intentionally left uninitialized so the host does not have to copy a
    # scalar zero to MUSA on every invocation.  chunk 0 has an exclusive
    # prefix of zero; all other chunks read the preceding inclusive sum from
    # prefix[chunk].
    base = tl.load(
        workspace_prefix_ptr + chunk,
        mask=(expert < num_experts) & (chunk > 0),
        other=0,
    ).to(tl.int64)
    local_row = tl.load(
        row_id_map_ptr + expert_2d * num_tokens + token_2d,
        mask=valid,
        other=0,
    )
    row = tl.where(local_row == 0, -1, local_row + base[:, None] - 1)
    tl.store(
        row_id_map_ptr + expert_2d * num_tokens + token_2d,
        row,
        mask=valid,
    )

    row_t = tl.trans(row)
    valid_t = tl.trans(valid)
    tl.store(
        row_id_map_non_trans_ptr + token[:, None] * num_experts + expert[None, :],
        row_t,
        mask=valid_t,
    )


def _fast_sort_chunks_by_idx(
    inp, split_sizes, sorted_indices, probs, num_tokens, hidden_size, num_splits,
    preallocated_out=None, preallocated_probs=None,
):
    global _SORT_HIT_LOGGED, _SORT_FALLBACK_LOGGED
    if os.environ.get("MUSA_TE_MOE_SORT_VALIDATE", "0") == "1":
        split_sizes_cpu = split_sizes.detach().cpu()
        sorted_indices_cpu = sorted_indices.detach().cpu()
        split_sum = int(split_sizes_cpu.sum())
        split_min = int(split_sizes_cpu.min()) if split_sizes_cpu.numel() else 0
        sorted_min = int(sorted_indices_cpu.min()) if sorted_indices_cpu.numel() else 0
        sorted_max = int(sorted_indices_cpu.max()) if sorted_indices_cpu.numel() else -1
        sorted_unique = int(torch.unique(sorted_indices_cpu).numel())
        probs_rows = None if probs is None else probs.shape[0]
        valid = (
            split_sum == num_tokens
            and split_min >= 0
            and sorted_min >= 0
            and sorted_max < num_splits
            and sorted_unique == num_splits
            and (probs_rows is None or probs_rows == num_tokens)
        )
        if not valid:
            raise RuntimeError(
                "invalid MoE chunk-sort metadata: "
                f"num_tokens={num_tokens} split_sum={split_sum} split_min={split_min} "
                f"num_splits={num_splits} sorted_min={sorted_min} "
                f"sorted_max={sorted_max} sorted_unique={sorted_unique} "
                f"probs_rows={probs_rows}"
            )
    if (
        os.environ.get("MUSA_TE_MOE_SORT_FASTPATH", "1") == "0"
        or num_splits != 256
        or hidden_size not in (2048, 3072)
    ):
        if not _SORT_FALLBACK_LOGGED:
            print(
                "[musa_te_moe_permutation] sort fallback: "
                f"tokens={num_tokens} hidden={hidden_size} splits={num_splits}",
                flush=True,
            )
            _SORT_FALLBACK_LOGGED = True
        return _ORIGINAL_SORT(
            inp, split_sizes, sorted_indices, probs, num_tokens, hidden_size,
            num_splits, preallocated_out, preallocated_probs,
        )

    row_id_map = torch.empty(num_tokens, dtype=torch.int64, device=inp.device)
    if preallocated_out is None:
        output = torch.empty((num_tokens, hidden_size), dtype=inp.dtype, device=inp.device)
    else:
        output = preallocated_out.view(inp.dtype)[: num_tokens * hidden_size].view(num_tokens, hidden_size)

    if probs is None:
        permuted_probs = None
    elif preallocated_probs is None:
        permuted_probs = torch.empty(num_tokens, dtype=probs.dtype, device=probs.device)
    else:
        permuted_probs = preallocated_probs.view(probs.dtype)[:num_tokens].view(num_tokens)

    metadata = torch.empty(2 * num_splits, dtype=torch.int64, device=split_sizes.device)
    input_ends = metadata[:num_splits]
    output_bases = metadata[num_splits:]
    _build_sort_metadata_kernel[(1,)](
        split_sizes,
        sorted_indices,
        input_ends,
        output_bases,
        num_splits,
        NUM_SPLITS_PAD=triton.next_power_of_2(num_splits),
    )

    _sort_chunks_binary_kernel[(num_tokens,)](
        inp, input_ends, output_bases, output, row_id_map, probs, permuted_probs,
        num_splits, hidden_size, inp.stride(0), inp.stride(1), output.stride(0),
        output.stride(1), probs.stride(0) if probs is not None else None,
        permuted_probs.stride(0) if permuted_probs is not None else None,
        PERMUTE_PROBS=probs is not None, BLOCK_SIZE=1024,
    )
    if not _SORT_HIT_LOGGED:
        print(
            f"[musa_te_moe_permutation] sort fast path hit: tokens={num_tokens} "
            f"hidden={hidden_size} splits={num_splits}",
            flush=True,
        )
        _SORT_HIT_LOGGED = True
    return output, row_id_map, permuted_probs


def _fast_make_row_id_map(routing_map, num_tokens, num_experts):
    global _ROW_HIT_LOGGED
    # TE is already faster for tiny maps; the tiled path starts winning at 8K
    # tokens and is intended primarily for the 32K Qwen3.6 workload.
    if num_experts != 256 or num_tokens < 8192:
        return _ORIGINAL_MAKE_ROW_ID_MAP(routing_map, num_tokens, num_experts)

    import transformer_engine.pytorch.triton.permutation as permutation

    pass1_block_size = 256
    blocks_per_expert = triton.cdiv(num_tokens, pass1_block_size)
    pass1_grid = (num_experts, blocks_per_expert)
    row_id_map = torch.empty((num_experts, num_tokens), dtype=torch.int64, device=routing_map.device)
    row_id_map_non_trans = torch.empty((num_tokens, num_experts), dtype=torch.int64, device=routing_map.device)
    workspace = torch.empty(pass1_grid, dtype=torch.int64, device=routing_map.device)
    permutation._row_id_map_pass_1_kernel[pass1_grid](
        routing_map, row_id_map, workspace, num_tokens,
        routing_map.stride(0), routing_map.stride(1), BLOCK_SIZE=pass1_block_size,
    )
    prefix = torch.empty(workspace.numel() + 1, dtype=torch.int64, device=routing_map.device)
    torch.cumsum(workspace.flatten(), dim=0, out=prefix[1:])
    block_expert = _ROW_BLOCK_EXPERT
    block_token = _ROW_BLOCK_TOKEN
    pass2_grid = (
        triton.cdiv(num_experts, block_expert),
        triton.cdiv(num_tokens, block_token),
    )
    _row_id_map_pass_2_tiled_kernel[pass2_grid](
        row_id_map,
        row_id_map_non_trans,
        prefix,
        num_experts,
        num_tokens,
        blocks_per_expert,
        PASS1_BLOCK_SIZE=pass1_block_size,
        BLOCK_EXPERT=block_expert,
        BLOCK_TOKEN=block_token,
    )
    if not _ROW_HIT_LOGGED:
        print(
            f"[musa_te_moe_permutation] row-id-map fast path hit: "
            f"tokens={num_tokens} experts={num_experts}",
            flush=True,
        )
        _ROW_HIT_LOGGED = True
    return row_id_map, row_id_map_non_trans


def _musa_safe_moe_permute_mask(
    dtype,
    inp,
    row_id_map,
    probs,
    num_tokens,
    num_experts,
    num_out_tokens,
    hidden_size,
    preallocated_out,
    preallocated_probs,
):
    """Avoid the MUSA extension only for unpermute backward's transposed map.

    TE's unpermute-mask backward calls ``tex.moe_permute_mask`` with an
    [experts, tokens] row map and no probabilities.  That extension currently
    fails with ``invalid device context`` on MUSA.  TE already provides an
    equivalent Triton implementation for this exact layout.
    """
    global _MASK_BWD_HIT_LOGGED
    is_unpermute_backward = (
        row_id_map.ndim == 2
        and row_id_map.shape[0] == num_experts
        and row_id_map.shape[1] == num_tokens
        and (probs is None or probs.numel() == 0)
    )
    if not is_unpermute_backward:
        return _ORIGINAL_MOE_PERMUTE_MASK(
            dtype,
            inp,
            row_id_map,
            probs,
            num_tokens,
            num_experts,
            num_out_tokens,
            hidden_size,
            preallocated_out,
            preallocated_probs,
        )

    import transformer_engine.pytorch.triton.permutation as permutation

    output, _ = permutation.permute_with_mask_map(
        inp,
        row_id_map,
        None,
        num_tokens,
        num_experts,
        num_out_tokens,
        hidden_size,
        preallocated_out if preallocated_out.numel() else None,
        None,
    )
    if not _MASK_BWD_HIT_LOGGED:
        print(
            f"[musa_te_moe_permutation] mask backward Triton path hit: "
            f"tokens={num_tokens} experts={num_experts} hidden={hidden_size}",
            flush=True,
        )
        _MASK_BWD_HIT_LOGGED = True
    return output, None


def patch_transformer_engine_moe_permutation() -> bool:
    """Install shape-gated fast paths. Safe to call repeatedly."""
    global _PATCHED, _ORIGINAL_SORT, _ORIGINAL_MAKE_ROW_ID_MAP, _ORIGINAL_MOE_PERMUTE_MASK
    if _PATCHED:
        return True
    try:
        import transformer_engine.pytorch.triton.permutation as permutation
    except (ImportError, RuntimeError):
        return False
    _ORIGINAL_SORT = permutation.sort_chunks_by_idx
    _ORIGINAL_MAKE_ROW_ID_MAP = permutation.make_row_id_map
    import transformer_engine.pytorch.permutation as te_permutation

    _ORIGINAL_MOE_PERMUTE_MASK = te_permutation.tex.moe_permute_mask
    permutation.sort_chunks_by_idx = _fast_sort_chunks_by_idx
    permutation.make_row_id_map = _fast_make_row_id_map
    te_permutation.tex.moe_permute_mask = _musa_safe_moe_permute_mask
    _PATCHED = True
    print("[musa_te_moe_permutation] TransformerEngine MoE fast paths installed", flush=True)
    return True
