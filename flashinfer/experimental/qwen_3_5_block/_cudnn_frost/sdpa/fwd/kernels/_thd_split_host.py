# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Shared prepared packed split host for half-precision attention kernels."""

from typing import Optional, Tuple

import cutlass
import cutlass.cute as cute
from cuda.bindings import driver as _cuda_driver

from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.compiled_cache import compile_cached as _compile_cached
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd.kernels.sm100.split_combine import _host_ptr_packed


@cute.jit
def host_independent_kv(
    q_ptr: cute.Pointer,
    k_ptr: cute.Pointer,
    v_ptr: cute.Pointer,
    o_ptr: cute.Pointer,
    lse_ptr: Optional[cute.Pointer],
    sinks_ptr: cute.Pointer,
    meta_ptr: cute.Pointer,
    o_desc_ptr: cute.Pointer,
    problem_size: Tuple[int, int, int, int, int, int],
    q_strides: Tuple[cutlass.Int64, cutlass.Int64, cutlass.Int64],
    k_strides: Tuple[cutlass.Int64, cutlass.Int64, cutlass.Int64],
    v_strides: Tuple[cutlass.Int64, cutlass.Int64, cutlass.Int64],
    o_strides: Tuple[cutlass.Int64, cutlass.Int64, cutlass.Int64],
    lse_ext: cutlass.Int64,
    scale_softmax_log2: cutlass.Float32,
    n_thd_units: cutlass.Int32,
    thd_q_lens_ptr: cute.Pointer,
    thd_kv_lens_ptr: cute.Pointer,
    thd_lens_form: cutlass.Int32,
    o_partial_ptr: cute.Pointer,
    lse_partial_ptr: cute.Pointer,
    partial_o_strides: Tuple[cutlass.Int64, cutlass.Int64, cutlass.Int64],
    lse_kind: cutlass.Constexpr[str],
    block_table_ptr: Optional[cute.Pointer],
    block_table_v_ptr: Optional[cute.Pointer],
    table_strides: Tuple[cutlass.Int64, cutlass.Int64],
    n_pages: cutlass.Int32,
    paged_hnd: cutlass.Constexpr[bool],
    kernel_host: cutlass.Constexpr,
    config: cutlass.Constexpr,
    ragged_q_slots: cutlass.Constexpr[bool],
    gate_slots: cutlass.Constexpr[bool] = False,
    thd_origin_ptrs=None,
    thd_origin_specs: cutlass.Constexpr = None,
    paged_v_hnd: cutlass.Constexpr = None,
    table_v_strides: Optional[Tuple[cutlass.Int64, cutlass.Int64]] = None,
    stream: _cuda_driver.CUstream = None,
) -> None:
    """One prepared host owns setup, split attention and final packed combine.

    The caller reserves bounded partial storage once. The native binder supplies
    compact partial-O strides: deriving TMA strides in this nested host fails
    the DSL 4.7.0 lowering. The live prefix total stays on the device and bounds
    the combine. Partial Stats always use natural logs.
    """
    d_qk, d_v, splits, stats_log2, has_sink = config
    b, qh, _kh, tq, _tkv, _ = problem_size
    tokens, heads = cutlass.Int64(tq), cutlass.Int64(qh)
    args = (
        q_ptr,
        k_ptr,
        v_ptr,
        o_partial_ptr,
        lse_partial_ptr,
        sinks_ptr,
        meta_ptr,
        o_desc_ptr,
        problem_size,
        q_strides,
        k_strides,
        v_strides,
        partial_o_strides,
        (heads * tokens, tokens, cutlass.Int64(1)),
        cutlass.Int32(tq),
        scale_softmax_log2,
        n_thd_units,
        cutlass.Int64(0),
        thd_q_lens_ptr,
        thd_kv_lens_ptr,
        thd_lens_form,
        o_partial_ptr,
        block_table_ptr,
        block_table_v_ptr,
        table_strides,
        n_pages,
    )
    if cutlass.const_expr(ragged_q_slots):
        args += (cutlass.Int64(0), cutlass.Int64(1))
    args += (d_qk, d_v, "dense", paged_hnd)
    if cutlass.const_expr(ragged_q_slots):
        args += (False,)
    if cutlass.const_expr(gate_slots):
        # The d256 half body's epilogue-gate slots (gate_ptr, gate_strides): required runtime slots of its host ABI, never bound on
        # the split path (the fp32 partials bypass the gated epilogue) -- the no-gate pair, as the prepared binders fill it.
        args += (None, (cutlass.Int64(0),) * 3)
    if cutlass.const_expr(paged_v_hnd is not None):
        # Native D576 accepts independently laid-out K and V pools/tables.
        args += (paged_v_hnd, table_v_strides)
    # `stream` BY KEYWORD: the kernel host's `stream` is its last, defaulted slot; the keyword keeps it there whatever a host appends
    # ahead of it.
    if cutlass.const_expr(thd_origin_ptrs is not None):
        kernel_host(*args, thd_origin_ptrs=thd_origin_ptrs, thd_origin_specs=thd_origin_specs, stream=stream)
    else:
        kernel_host(*args, stream=stream)
    final_stats_strides = (
        (cutlass.Int64(0), cutlass.Int64(lse_ext), cutlass.Int64(1)) if cutlass.const_expr(lse_kind == "head") else (cutlass.Int64(0), cutlass.Int64(1), heads)
    )
    _host_ptr_packed(
        o_partial_ptr,
        lse_partial_ptr,
        o_ptr,
        lse_ptr,
        (1, qh, tq, d_v),
        cutlass.Int32(splits),
        (cutlass.Int64(0), o_strides[1], o_strides[2], cutlass.Int64(1)),
        final_stats_strides,
        meta_ptr + cutlass.Int64(2) * cutlass.Int64(b),
        stats_log2,
        stream,
        sinks_ptr=sinks_ptr if cutlass.const_expr(has_sink) else None,
        prefetch_group=(2 if splits <= 2 else 4 if splits <= 4 else 8 if splits <= 8 else 0),
        cu_q_ptr=meta_ptr + cutlass.Int64(b) if cutlass.const_expr(thd_origin_ptrs is not None) else None,
        origin_ptrs=thd_origin_ptrs,
        origin_specs=thd_origin_specs,
        n_batch=cutlass.Int32(b),
    )


@cute.jit
def host(
    q_ptr: cute.Pointer,
    k_ptr: cute.Pointer,
    v_ptr: cute.Pointer,
    o_ptr: cute.Pointer,
    lse_ptr: Optional[cute.Pointer],
    sinks_ptr: cute.Pointer,
    meta_ptr: cute.Pointer,
    o_desc_ptr: cute.Pointer,
    problem_size: Tuple[int, int, int, int, int, int],
    q_strides: Tuple[cutlass.Int64, cutlass.Int64, cutlass.Int64],
    k_strides: Tuple[cutlass.Int64, cutlass.Int64, cutlass.Int64],
    v_strides: Tuple[cutlass.Int64, cutlass.Int64, cutlass.Int64],
    o_strides: Tuple[cutlass.Int64, cutlass.Int64, cutlass.Int64],
    lse_ext: cutlass.Int64,
    scale_softmax_log2: cutlass.Float32,
    n_thd_units: cutlass.Int32,
    thd_q_lens_ptr: cute.Pointer,
    thd_kv_lens_ptr: cute.Pointer,
    thd_lens_form: cutlass.Int32,
    o_partial_ptr: cute.Pointer,
    lse_partial_ptr: cute.Pointer,
    partial_o_strides: Tuple[cutlass.Int64, cutlass.Int64, cutlass.Int64],
    lse_kind: cutlass.Constexpr[str],
    block_table_ptr: Optional[cute.Pointer],
    block_table_v_ptr: Optional[cute.Pointer],
    table_strides: Tuple[cutlass.Int64, cutlass.Int64],
    n_pages: cutlass.Int32,
    paged_hnd: cutlass.Constexpr[bool],
    kernel_host: cutlass.Constexpr,
    config: cutlass.Constexpr,
    ragged_q_slots: cutlass.Constexpr[bool],
    gate_slots: cutlass.Constexpr[bool] = False,
    thd_origin_ptrs=None,
    thd_origin_specs: cutlass.Constexpr = None,
    stream: _cuda_driver.CUstream = None,
) -> None:
    """Retain the existing packed-split host ABI for shared-table kernels."""
    host_independent_kv(
        q_ptr,
        k_ptr,
        v_ptr,
        o_ptr,
        lse_ptr,
        sinks_ptr,
        meta_ptr,
        o_desc_ptr,
        problem_size,
        q_strides,
        k_strides,
        v_strides,
        o_strides,
        lse_ext,
        scale_softmax_log2,
        n_thd_units,
        thd_q_lens_ptr,
        thd_kv_lens_ptr,
        thd_lens_form,
        o_partial_ptr,
        lse_partial_ptr,
        partial_o_strides,
        lse_kind,
        block_table_ptr,
        block_table_v_ptr,
        table_strides,
        n_pages,
        paged_hnd,
        kernel_host,
        config,
        ragged_q_slots,
        gate_slots,
        thd_origin_ptrs,
        thd_origin_specs,
        stream=stream,
    )


def compile_host(
    kernel_host,
    cfg,
    storage_dtype,
    cache_key,
    *,
    has_lse,
    lse_kind,
    paged_hnd,
    ragged_q_slots,
    has_sink=False,
    gate_slots=False,
    thd_origin_specs=None,
    paged_v_hnd=None,
):
    """Specialize one common pointer host; extents and strides stay dynamic."""
    if lse_kind not in ("head", "token"):
        raise ValueError("prepared packed split Stats must be head- or token-major")

    def P(dtype, align=16):
        return cute.runtime.make_ptr(dtype, 16, cute.AddressSpace.gmem, assumed_align=align)

    i32, strides = cutlass.Int32(0), (cutlass.Int64(0),) * 3
    return _compile_cached(
        host if paged_v_hnd is None else host_independent_kv,
        P(storage_dtype),
        P(storage_dtype),
        P(storage_dtype),
        P(storage_dtype),
        P(cutlass.Float32, 4) if has_lse else None,
        P(cutlass.Float32),
        P(cutlass.Int32),
        P(cutlass.Int64),
        (0, 0, 0, 0, 0, 0),
        strides,
        strides,
        strides,
        strides,
        cutlass.Int64(0),
        cutlass.Float32(0),
        i32,
        P(cutlass.Int32, 4),
        P(cutlass.Int32, 4),
        i32,
        P(cutlass.Float32),
        P(cutlass.Float32, 4),
        strides,
        lse_kind,
        P(cutlass.Int32, 4) if cfg.PAGED_KV else None,
        P(cutlass.Int32, 4) if cfg.PAGED_KV else None,
        (cutlass.Int64(0), cutlass.Int64(0)),
        i32,
        paged_hnd,
        kernel_host,
        (cfg.TILE_K, cfg.TILE_O, cfg.SPLIT_KV, bool(cfg.STATS_LOG2), bool(has_sink)),
        ragged_q_slots,
        gate_slots,
        thd_origin_ptrs=(
            tuple(P(cutlass.Int64 if spec[3] == 64 else cutlass.Int32, 8 if spec[3] == 64 else 4) if spec is not None else None for spec in thd_origin_specs)
            if thd_origin_specs is not None
            else None
        ),
        thd_origin_specs=thd_origin_specs,
        **({"paged_v_hnd": paged_v_hnd, "table_v_strides": (cutlass.Int64(0), cutlass.Int64(0)) if cfg.PAGED_KV else None} if paged_v_hnd is not None else {}),
        stream=cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=False),
        options="--enable-tvm-ffi",
        cache_key=cache_key,
        symbol="frost_sdpa_fwd_thd_split",
    )
