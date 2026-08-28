# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
# Adapted from Block-Sparse-Attention/bsa_attn_interface.py for FlashInfer integration.
#
# This is the CuTe-DSL replacement for the previous C++/CUDA CUTLASS blk64
# kernel (torch.utils.cpp_extension.load-based). It follows the same
# vendor-kernel + thin-adapter pattern as bsa_attn_sm100_blk128.py: the kernel
# class lives under sm100_blk64/ (CuTe-DSL Python, JIT-compiled via
# cute.compile), and this module handles validation, tensor conversion,
# compile-key/cache management, and dispatch.
#
# JIT-only: unlike upstream's blk64 forward, this integration does not wire up
# the ahead-of-time (AOT) precompiled-artifact path -- FlashInfer's blk128
# CuTe-DSL integration is JIT-only too, and this keeps both backends
# consistent (see dispatch_helpers.py module docstring).

from typing import Optional, Tuple

import torch

import cuda.bindings.driver as cuda

import cutlass.cute as cute

from flashinfer.api_logging import flashinfer_api

from .bsa_utils.cache_utils import get_jit_cache
from .bsa_utils.testing import is_fake_mode
from .bsa_utils import fa_logging
from .sm100_blk64.bsa_fwd_sm100 import BlockSparseAttnForwardSm100Blk64
from .sm100_blk64.cute_dsl_utils import constexpr_tvm_ffi_converter_patched
from .sm100_blk64.dispatch_helpers import (
    _get_device_arch,
    maybe_contiguous,
    torch2cute_dtype_map,
    validate_sm100_blk64_int32_bounds,
    sm100_blk64_requires_int64_kv_strides,
    dynamic_tensors_compile_key,
    sm100_blk64_auto_kv_splits,
    build_sm100_blk64_kv_split_offsets,
    resolve_sm100_blk64_split_workspace,
    choose_sm100_blk64_use_clc,
    make_sm100_blk64_cute_args,
    combine_blk64_kv_bucketed_partials,
    workaround_cutlass_hash_import_bug,
    validate_sm100_blk64_fp8_sage,
)

_sm100_blk64_compile_cache = get_jit_cache("bsa_fwd_blk64")


@flashinfer_api
def bsa_attn_sm100_blk64_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q2k_block_index: torch.Tensor,
    block_sparse_num: int,
    block_sizes: Optional[torch.Tensor] = None,
    q2k_block_nums: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    return_lse: bool = False,
    out: Optional[torch.Tensor] = None,
    lse: Optional[torch.Tensor] = None,
    kv_splits: "int | str" = 1,
    use_clc: Optional[bool] = None,
    q_scale: Optional[torch.Tensor] = None,
    k_scale: Optional[torch.Tensor] = None,
    v_scale: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Forward pass for BSA block-sparse attention using the blk64 CuTe-DSL kernel.

    Supports SM100 and SM103. MHA only (num_kv_heads must equal num_heads);
    head_dim must be 128.

    Args:
        q: Query tensor (batch, seqlen_q, num_heads, head_dim).
        k: Key tensor (batch, seqlen_k, num_heads, head_dim).
        v: Value tensor (batch, seqlen_k, num_heads, head_dim).
        q2k_block_index: Block index tensor (batch, num_heads, num_q_blocks, max_kv_blocks), int32.
        block_sparse_num: Number of KV blocks each Q block attends to (>= 1).
            Ignored when q2k_block_nums is provided.
        block_sizes: Actual token count per KV block (num_kv_blocks,), int32.  Pass None to
            skip per-block padding masking (assumes all blocks are full).
        q2k_block_nums: Per-(batch, head, q_block) number of KV blocks to attend to,
            (batch, num_heads, num_q_blocks) int32.  When None, uses fixed block_sparse_num.
        softmax_scale: Softmax scale (default: 1/sqrt(head_dim)).
        return_lse: Whether to return log-sum-exp.
        out: Pre-allocated output tensor (batch, seqlen_q, num_heads, head_dim).
        lse: Pre-allocated LSE tensor (batch, num_heads, seqlen_q).
        kv_splits: Number of KV splits ("auto", or an explicit int in [1, 256]).
            kv_splits=1 (or an "auto" resolution of 1) reproduces the original
            single-kernel behavior exactly.
        use_clc: Optional scheduler override (True forces the CLC persistent
            scheduler, False forces the static scheduler, None uses the
            shape-based heuristic).
        q_scale, k_scale, v_scale: Sage FP8 quantization scales. All three
            must be provided together to enable the FP8 path (q/k/v must then
            be float8_e4m3fn); otherwise all three must be None and q/k/v must
            be bfloat16. The FP8 path additionally requires batch_size == 1,
            num_head in (4, 8), q2k_block_nums is None, and block_sizes is
            None (upstream kernel limits, not specific to this integration).

    Returns:
        (out, lse) where lse is None if return_lse is False.
    """
    batch_size, seqlen_q, num_head, head_dim = q.shape
    seqlen_k = k.shape[1]
    num_head_kv = k.shape[2]
    head_dim_v = v.shape[-1]

    is_sage_fp8 = q_scale is not None
    if is_sage_fp8:
        assert k_scale is not None and v_scale is not None, "FP8 requires Q/K/V scales"
        assert q.dtype == torch.float8_e4m3fn, "FP8 inputs must use float8_e4m3fn"
        assert k.dtype == q.dtype and v.dtype == q.dtype
        assert (
            q_scale.device == q.device
            and k_scale.device == q.device
            and v_scale.device == q.device
        ), "q_scale/k_scale/v_scale must be on the same CUDA device as Q/K/V"
        validate_sm100_blk64_fp8_sage(
            batch_size,
            num_head,
            q2k_block_nums,
            block_sizes,
            seqlen_q,
            seqlen_k,
            head_dim_v,
            q_scale,
            k_scale,
            v_scale,
        )
    else:
        assert k_scale is None and v_scale is None, (
            "Q/K/V scales must be provided together"
        )
        assert q.dtype == k.dtype == v.dtype == torch.bfloat16, (
            "blk64 CuTeDSL requires Q/K/V to all use bf16"
        )

    assert q.is_cuda and k.is_cuda and v.is_cuda
    assert q.device == k.device == v.device, "Q/K/V must be on the same CUDA device"
    assert head_dim == 128 and head_dim_v == 128, "blk64 CuTeDSL requires D=DV=128"
    assert num_head == num_head_kv, "blk64 CuTeDSL currently supports MHA only"
    assert k.shape == (batch_size, seqlen_k, num_head_kv, head_dim)
    assert v.shape == (batch_size, seqlen_k, num_head_kv, head_dim_v)

    arch = _get_device_arch(q.device.index)
    if arch not in (100, 103):
        raise RuntimeError(
            f"bsa_attn_sm100_blk64_fwd only supports SM100/SM103, got SM{arch}"
        )

    auto_kv_splits = isinstance(kv_splits, str)
    if auto_kv_splits:
        assert kv_splits == "auto", "kv_splits string value must be 'auto'"
        kv_splits_i = 1
    else:
        kv_splits_i = int(kv_splits)
        assert kv_splits_i >= 1, "kv_splits must be >= 1"
        assert kv_splits_i <= 256, "kv_splits must be <= 256"

    # Convert to native BHSD layout for the kernel (public API stays BSHD to
    # match the existing blk64/blk128 FlashInfer call convention). A
    # transposed view keeps the head_dim (last) stride at 1 -- the only
    # thing the TMA loads for Q/K/V actually require -- so maybe_contiguous
    # skips the copy for the common case of physically-contiguous BSHD
    # inputs, matching the pattern already used by the blk128 sibling
    # backend (see maybe_contiguous import above). Verified against
    # kv_splits=1/4/"auto" (split-KV + combine path), asymmetric seqlen, and
    # Sage-FP8 -- all bit-correct with zero-copy views.
    q_bhsd = maybe_contiguous(q.transpose(1, 2))
    k_bhsd = maybe_contiguous(k.transpose(1, 2))
    v_bhsd = maybe_contiguous(v.transpose(1, 2))

    requested_out = out
    requested_lse = lse
    output_dtype = torch.bfloat16 if is_sage_fp8 else q_bhsd.dtype
    if requested_out is not None:
        assert requested_out.shape == (batch_size, seqlen_q, num_head, head_dim_v)
        assert requested_out.dtype == output_dtype
        assert requested_out.device == q_bhsd.device
    if requested_lse is not None:
        assert requested_lse.shape == (batch_size, num_head, seqlen_q)
        assert requested_lse.dtype == torch.float32
        assert requested_lse.device == q_bhsd.device
        assert requested_lse.is_contiguous(), (
            "pre-allocated lse must be contiguous; got strides "
            f"{requested_lse.stride()}"
        )

    num_q_blocks = (seqlen_q + 63) // 64
    num_kv_blocks = (seqlen_k + 63) // 64

    assert q2k_block_index.dtype == torch.int32
    assert q2k_block_index.device == q_bhsd.device
    assert q2k_block_index.ndim == 4
    assert q2k_block_index.shape[:3] == (batch_size, num_head, num_q_blocks)
    q2k_block_index = maybe_contiguous(q2k_block_index)

    has_block_sizes = block_sizes is not None and block_sizes.numel() > 0
    if has_block_sizes:
        block_sizes = maybe_contiguous(block_sizes)
        assert block_sizes.dtype == torch.int32
        assert block_sizes.device == q_bhsd.device
        assert block_sizes.shape == (num_kv_blocks,)
    else:
        block_sizes = None

    has_variable_block_nums = q2k_block_nums is not None and q2k_block_nums.numel() > 0
    if has_variable_block_nums:
        q2k_block_nums = maybe_contiguous(q2k_block_nums)
        assert q2k_block_nums.dtype == torch.int32
        assert q2k_block_nums.device == q_bhsd.device
        assert q2k_block_nums.shape == (batch_size, num_head, num_q_blocks), (
            "q2k_block_nums must be shaped (B, H, ceil(S_q/64)); got "
            f"{tuple(q2k_block_nums.shape)}"
        )
        uniform_block_sparse_num = 0
    else:
        if block_sparse_num <= 0:
            block_sparse_num = int(q2k_block_index.shape[-1])
        assert q2k_block_index.shape[-1] >= block_sparse_num, (
            f"q2k_block_index last dim ({q2k_block_index.shape[-1]}) must be "
            f">= block_sparse_num ({block_sparse_num})"
        )
        uniform_block_sparse_num = int(block_sparse_num)
        q2k_block_nums = None

    # Phantom block masking: when using variable block nums, the kernel pads
    # each row's KV count to a multiple of 8 and fills phantom slots with the
    # last real block.  Phantom blocks are only masked correctly when
    # HasBlockSizes=True.  Passing a full-block sizes tensor (all 64)
    # activates that path and ensures phantom blocks are zeroed in softmax.
    if block_sizes is None and has_variable_block_nums:
        block_sizes = torch.full(
            (num_kv_blocks,), 64, dtype=torch.int32, device=q_bhsd.device
        )
        has_block_sizes = True

    validate_sm100_blk64_int32_bounds(
        q_bhsd,
        k_bhsd,
        v_bhsd,
        q2k_block_index,
        uniform_block_sparse_num,
        block_sizes,
        q2k_block_nums,
    )
    use_int64_kv_strides = sm100_blk64_requires_int64_kv_strides(k_bhsd, v_bhsd)

    if softmax_scale is None:
        softmax_scale = head_dim**-0.5

    dtype = torch2cute_dtype_map[q_bhsd.dtype]
    sparse_block_size = 64
    qhead_per_kvhead = 1
    tile_m = 64
    tile_n = 256

    if auto_kv_splits:
        kv_splits_i = sm100_blk64_auto_kv_splits(
            q_bhsd, q2k_block_index, uniform_block_sparse_num
        )
    kv_splits_i = resolve_sm100_blk64_split_workspace(
        q_bhsd,
        head_dim_v,
        kv_splits_i,
        allow_fallback=auto_kv_splits,
        output_dtype=output_dtype,
    )
    allow_empty_block_nums = kv_splits_i > 1
    if use_clc is None:
        if kv_splits_i > 1:
            use_clc_scheduler = False
        else:
            use_clc_scheduler = choose_sm100_blk64_use_clc(
                q_bhsd,
                uniform_block_sparse_num,
                q2k_block_nums if has_variable_block_nums else None,
                layout="bhsd",
            )
    else:
        use_clc_scheduler = bool(use_clc)
    is_persistent = use_clc_scheduler
    pack_gqa = False

    split_offsets = None
    if kv_splits_i > 1:
        if has_variable_block_nums or use_clc_scheduler:
            split_offsets = build_sm100_blk64_kv_split_offsets(
                q2k_block_nums,
                uniform_block_sparse_num,
                batch_size,
                num_head,
                num_q_blocks,
                kv_splits_i,
                q_bhsd.device,
            )
        out_bhsd = torch.empty(
            (batch_size, kv_splits_i * num_head, seqlen_q, head_dim_v),
            dtype=torch.float32,
            device=q_bhsd.device,
        )
        lse = torch.empty(
            (batch_size, kv_splits_i * num_head, seqlen_q),
            dtype=torch.float32,
            device=q_bhsd.device,
        )
    else:
        # Always allocate a fresh contiguous BHSD output for the kernel (even
        # when the caller pre-allocated a BSHD `out`); a transposed view of
        # `out` would hand the kernel a non-contiguous destination, which the
        # upstream kernel does not expect. Copy back into `out` at the end.
        out_bhsd = torch.empty(
            (batch_size, num_head, seqlen_q, head_dim_v),
            dtype=output_dtype,
            device=q_bhsd.device,
        )
        lse = (
            requested_lse
            if requested_lse is not None
            else torch.empty(
                (batch_size, num_head, seqlen_q),
                dtype=torch.float32,
                device=q_bhsd.device,
            )
        )

    current_stream = (
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        if is_fake_mode()
        else cuda.CUstream(torch.cuda.current_stream(q_bhsd.device).cuda_stream)
    )

    compile_key = dynamic_tensors_compile_key(
        "sm100_blk64_fwd",
        (
            dtype,
            head_dim,
            head_dim_v,
            qhead_per_kvhead,
            pack_gqa,
            tile_m,
            tile_n,
            sparse_block_size,
            arch,
            fa_logging.get_fa_log_level(),
            has_variable_block_nums,
            allow_empty_block_nums,
            has_block_sizes,
            kv_splits_i,
            out_bhsd.dtype,
            is_persistent,
            use_clc_scheduler,
            "bhsd_native",
            use_int64_kv_strides,
            is_sage_fp8,
            "tvm_ffi_env_stream_v1",
        ),
        (
            q_bhsd,
            k_bhsd,
            v_bhsd,
            out_bhsd,
            lse,
            q2k_block_index,
            block_sizes,
            q2k_block_nums,
            split_offsets,
            q_scale,
            k_scale,
            v_scale,
        ),
    )

    if compile_key not in _sm100_blk64_compile_cache:
        workaround_cutlass_hash_import_bug()
        jit_args = make_sm100_blk64_cute_args(
            q_bhsd,
            k_bhsd,
            v_bhsd,
            out_bhsd,
            lse,
            q_scale,
            k_scale,
            v_scale,
            softmax_scale,
            q2k_block_index,
            block_sizes,
            uniform_block_sparse_num,
            q2k_block_nums,
            split_offsets,
            current_stream,
            enable_tvm_ffi=True,
        )

        bsa_fwd = BlockSparseAttnForwardSm100Blk64(
            head_dim,
            head_dim_v,
            qhead_per_kvhead=qhead_per_kvhead,
            pack_gqa=pack_gqa,
            m_block_size=tile_m,
            n_block_size=tile_n,
            sparse_block_size=sparse_block_size,
            is_persistent=is_persistent,
            use_clc_scheduler=use_clc_scheduler,
            allow_empty_block_nums=allow_empty_block_nums,
            has_block_sizes=has_block_sizes,
            num_splits=kv_splits_i,
            use_int64_kv_strides=use_int64_kv_strides,
        )

        with constexpr_tvm_ffi_converter_patched():
            _sm100_blk64_compile_cache[compile_key] = cute.compile(
                bsa_fwd, *jit_args, options="--enable-tvm-ffi"
            )

    if not is_fake_mode():
        with torch.cuda.nvtx.range("bsa_attn_sm100_blk64_fwd_kernel"):
            _sm100_blk64_compile_cache[compile_key](
                q_bhsd.detach(),
                k_bhsd.detach(),
                v_bhsd.detach(),
                out_bhsd.detach(),
                lse,
                q_scale.detach() if is_sage_fp8 else None,
                k_scale.detach() if is_sage_fp8 else None,
                v_scale.detach() if is_sage_fp8 else None,
                softmax_scale,
                q2k_block_index.detach(),
                block_sizes.detach() if has_block_sizes else None,
                uniform_block_sparse_num,
                q2k_block_nums.detach() if has_variable_block_nums else None,
                split_offsets.detach() if split_offsets is not None else None,
                current_stream,
            )

    if kv_splits_i > 1:
        out_bhsd, lse = combine_blk64_kv_bucketed_partials(
            q_bhsd,
            out_bhsd,
            lse,
            kv_splits_i,
            arch,
            output_dtype=torch.bfloat16 if is_sage_fp8 else None,
            use_fast_16_split=(
                kv_splits_i == 16
                and not has_variable_block_nums
                and uniform_block_sparse_num >= kv_splits_i
            ),
        )

    result_out = out_bhsd.transpose(1, 2).contiguous()
    if requested_out is not None and result_out is not requested_out:
        requested_out.copy_(result_out)
        result_out = requested_out
    if requested_lse is not None and lse is not requested_lse:
        requested_lse.copy_(lse)
        lse = requested_lse

    if return_lse:
        return result_out, lse
    return result_out, None


def bsa_attn_blk64_fwd(*args, **kwargs):
    """Deprecated alias for bsa_attn_sm100_blk64_fwd.

    .. deprecated:: 0.6.18
        Use :func:`bsa_attn_sm100_blk64_fwd` instead.
    """
    import warnings

    warnings.warn(
        "bsa_attn_blk64_fwd is deprecated and will be removed in a future release. "
        "Use bsa_attn_sm100_blk64_fwd instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return bsa_attn_sm100_blk64_fwd(*args, **kwargs)
