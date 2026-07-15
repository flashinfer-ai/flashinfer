# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PyTorch integration for the Blackwell DeepSeek V4 HCA decode kernel.

HCA scans two independent paged pools: a 128-slot sliding-window pool and a
dense heavily-compressed pool. Unlike TRTLLM-GEN's dynamic-token-sparse ABI,
its page tables contain physical page IDs, not arbitrary token-row indices.
"""

from __future__ import annotations

import functools
import math
import os
from typing import Callable, Optional, Tuple

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32

from flashinfer.cute_dsl.utils import (
    _as_cute_dsl_workspace_i8,
    get_max_active_clusters,
)

from ..dsa.hca_fp8 import BlackwellHeavilyCompressedAttentionForwardFP8

_HCA_HEAD_DIM = 512
_HCA_WINDOW_CAPACITY = 128
_HCA_MAX_HEADS = 128
_MMA_QK_TILER_MN = (128, 128)
_MMA_PV_TILER_MN = (128, 256)
_CLUSTER_SHAPE_MNK = (2, 1, 1)
_MAX_NONPERSISTENT_QUERY_ROWS = 65535


def _check_nonpersistent_grid(batch_size: int, q_len: int) -> None:
    """Reject query grids that exceed CUDA's non-persistent grid-y limit."""
    query_rows = batch_size * q_len
    if query_rows > _MAX_NONPERSISTENT_QUERY_ROWS:
        raise ValueError(
            "CuTe DSL HCA non-persistent launch requires batch_size * q_len "
            f"<= {_MAX_NONPERSISTENT_QUERY_ROWS}, got {query_rows}"
        )


@functools.cache
def _get_split_kv_and_workspace_size(
    batch_size: int,
    q_len: int,
    num_heads: int,
    max_hca_len: int,
    max_active_blocks: int,
) -> Tuple[int, int]:
    """Return the uniform split count and required byte workspace."""
    if num_heads < _MMA_QK_TILER_MN[0]:
        # The current HCA epilogue only supports partial head tiles without a
        # split-K reduction.
        split_kv = 1
    else:
        split_kv = BlackwellHeavilyCompressedAttentionForwardFP8.get_split_kv(
            batch_size,
            q_len,
            max_hca_len,
            _MMA_QK_TILER_MN,
            max_active_blocks,
        )
    workspace_size = BlackwellHeavilyCompressedAttentionForwardFP8.get_workspace_size(
        num_heads,
        q_len,
        _HCA_HEAD_DIM,
        batch_size,
        split_kv,
        cutlass.Float32,
    )
    return split_kv, workspace_size


@functools.cache
def _check_can_implement(
    q_len: int,
    num_heads: int,
    max_hca_len: int,
    split_kv: int,
    page_size_compressed: int,
    page_size_window: int,
    is_persistent: bool,
) -> None:
    """Raise a descriptive error for unsupported static configurations."""
    if not BlackwellHeavilyCompressedAttentionForwardFP8.can_implement(
        1,
        q_len,
        max_hca_len,
        num_heads,
        _HCA_HEAD_DIM,
        cutlass.Float8E4M3FN,
        cutlass.BFloat16,
        cutlass.Float32,
        cutlass.Float32,
        _MMA_QK_TILER_MN,
        _MMA_PV_TILER_MN,
        split_kv,
        is_persistent,
        True,
        False,
        page_size_compressed,
        page_size_window,
    ):
        raise ValueError(
            "CuTe DSL HCA does not support this configuration: "
            f"q_len={q_len}, num_heads={num_heads}, "
            f"max_hca_len={max_hca_len}, split_kv={split_kv}, "
            f"page_size_compressed={page_size_compressed}, "
            f"page_size_window={page_size_window}"
        )


def _make_hca_fake_tensors(
    page_size_compressed: int,
    page_size_window: int,
    workspace_is_empty: bool,
):
    """Build symbolic TVM-FFI tensors for one HCA specialization."""
    sym_batch = cute.sym_int()
    sym_q_len = cute.sym_int()
    sym_heads = cute.sym_int()
    sym_head_dim = cute.sym_int(divisibility=16)
    sym_window_pages = cute.sym_int()
    sym_compressed_pages = cute.sym_int()
    sym_table_rows = cute.sym_int()
    sym_window_table_pages = cute.sym_int()
    sym_compressed_table_pages = cute.sym_int()
    sym_workspace_size = cute.sym_int()

    query = cute.runtime.make_fake_compact_tensor(
        cutlass.Float8E4M3FN,
        (sym_batch, sym_q_len, sym_heads, sym_head_dim),
        stride_order=(3, 2, 1, 0),
        assumed_align=16,
    )
    window_cache = cute.runtime.make_fake_compact_tensor(
        cutlass.Float8E4M3FN,
        (sym_window_pages, page_size_window, sym_head_dim),
        stride_order=(2, 1, 0),
        assumed_align=16,
    )
    compressed_cache = cute.runtime.make_fake_compact_tensor(
        cutlass.Float8E4M3FN,
        (sym_compressed_pages, page_size_compressed, sym_head_dim),
        stride_order=(2, 1, 0),
        assumed_align=16,
    )
    window_block_tables = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (sym_table_rows, sym_window_table_pages),
        stride_order=(1, 0),
        assumed_align=16,
    )
    compressed_block_tables = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (sym_table_rows, sym_compressed_table_pages),
        stride_order=(1, 0),
        assumed_align=16,
    )
    out = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16,
        (sym_batch, sym_q_len, sym_heads, sym_head_dim),
        stride_order=(3, 2, 1, 0),
        assumed_align=16,
    )
    lse = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32,
        (sym_batch, sym_q_len, sym_heads),
        stride_order=(2, 1, 0),
        assumed_align=16,
    )
    workspace = None
    if not workspace_is_empty:
        workspace = cute.runtime.make_fake_compact_tensor(
            cutlass.Int8,
            (sym_workspace_size,),
            assumed_align=32,
        )
    hca_seq_lens = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (sym_batch,), assumed_align=16
    )
    sparse_topk_lens = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (sym_table_rows,), assumed_align=16
    )
    window_valid_lens = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (sym_table_rows,), assumed_align=16
    )
    sinks = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32, (_HCA_MAX_HEADS,), assumed_align=16
    )
    return (
        query,
        window_cache,
        compressed_cache,
        window_block_tables,
        compressed_block_tables,
        out,
        lse,
        workspace,
        hca_seq_lens,
        sparse_topk_lens,
        window_valid_lens,
        sinks,
    )


@functools.cache
def _compile_hca_kernel(
    page_size_compressed: int,
    page_size_window: int,
    q_len: int,
    is_causal: bool,
    is_persistent: bool,
    workspace_is_empty: bool,
    compute_capability: tuple[int, int],
) -> Callable:
    """Compile and cache one FP8-input/BF16-output HCA kernel."""
    if compute_capability not in ((10, 0), (10, 3)):
        raise ValueError(
            "CuTe DSL HCA compilation requires SM100/SM103, got "
            f"SM{compute_capability[0]}{compute_capability[1]}"
        )
    kernel = BlackwellHeavilyCompressedAttentionForwardFP8(
        acc_dtype=cutlass.Float32,
        lse_dtype=cutlass.Float32,
        mma_qk_tiler_mn=_MMA_QK_TILER_MN,
        mma_pv_tiler_mn=_MMA_PV_TILER_MN,
        max_active_clusters=get_max_active_clusters(
            _CLUSTER_SHAPE_MNK[0] * _CLUSTER_SHAPE_MNK[1]
        ),
        page_size_cmp=page_size_compressed,
        page_size_win=page_size_window,
        skip_correction_threshold=0.0,
        is_persistent=is_persistent,
        is_var_seq=True,
        is_var_split_kv=False,
        is_causal=is_causal,
        seq_len_q=q_len,
        hca_compress_ratio=128,
    )
    (
        query,
        window_cache,
        compressed_cache,
        window_block_tables,
        compressed_block_tables,
        out,
        lse,
        workspace,
        hca_seq_lens,
        sparse_topk_lens,
        window_valid_lens,
        sinks,
    ) = _make_hca_fake_tensors(
        page_size_compressed,
        page_size_window,
        workspace_is_empty,
    )
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile(
        kernel,
        query,
        window_cache,
        compressed_cache,
        window_block_tables,
        compressed_block_tables,
        out,
        lse,
        workspace,
        Int32(1),
        hca_seq_lens,
        None,
        sparse_topk_lens,
        window_valid_lens,
        Float32(1.0),
        Float32(1.0),
        sinks,
        stream,
        options="--enable-tvm-ffi --opt-level 2",
    )


def _check_tensor(
    tensor: torch.Tensor,
    *,
    name: str,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
    alignment: int = 16,
) -> None:
    if tensor.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(tensor.shape)}")
    if tensor.dtype != dtype:
        raise ValueError(f"{name} must have dtype {dtype}, got {tensor.dtype}")
    if tensor.device != device:
        raise ValueError(f"{name} must be on {device}, got {tensor.device}")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    if tensor.data_ptr() % alignment != 0:
        raise ValueError(f"{name} must be {alignment}-byte aligned")


def _validate_hca_values(
    *,
    window_kv_cache: torch.Tensor,
    compressed_kv_cache: torch.Tensor,
    window_block_tables: torch.Tensor,
    compressed_block_tables: torch.Tensor,
    hca_seq_lens: torch.Tensor,
    sparse_topk_lens: torch.Tensor,
    window_valid_lens: torch.Tensor,
    q_len: int,
    is_causal: bool,
) -> None:
    """Run opt-in device-synchronizing validation for HCA metadata values."""
    if os.environ.get("FLASHINFER_VALIDATE_INPUTS", "0") in ("0", ""):
        return

    max_hca_len = _HCA_WINDOW_CAPACITY + (
        compressed_block_tables.shape[1] * compressed_kv_cache.shape[1]
    )
    invalid_seq_lens = torch.logical_or(
        hca_seq_lens < _HCA_WINDOW_CAPACITY,
        hca_seq_lens > max_hca_len,
    )
    if invalid_seq_lens.any().item():
        raise ValueError(
            "hca_seq_lens values must be between 128 and the block-table "
            f"capacity {max_hca_len}"
        )

    seq_lens_per_row = (
        hca_seq_lens.repeat_interleave(q_len) if is_causal else hca_seq_lens
    )
    invalid_topk_lens = torch.logical_or(
        sparse_topk_lens < _HCA_WINDOW_CAPACITY,
        sparse_topk_lens > seq_lens_per_row,
    )
    if invalid_topk_lens.any().item():
        raise ValueError(
            "sparse_topk_lens values must be between 128 and the matching "
            "hca_seq_lens value"
        )
    invalid_window_lens = torch.logical_or(
        window_valid_lens < 0,
        window_valid_lens > _HCA_WINDOW_CAPACITY,
    )
    if invalid_window_lens.any().item():
        raise ValueError("window_valid_lens values must be between 0 and 128")

    # The kernel always loads the fixed 128-slot window tile, even when some
    # slots are masked by window_valid_lens. Extra table columns are padding.
    window_page_count = (
        _HCA_WINDOW_CAPACITY + window_kv_cache.shape[1] - 1
    ) // window_kv_cache.shape[1]
    active_window_tables = window_block_tables[:, :window_page_count]
    invalid_window_pages = torch.logical_or(
        active_window_tables < 0,
        active_window_tables >= window_kv_cache.shape[0],
    )
    if invalid_window_pages.any().item():
        raise ValueError(
            "window_block_tables active values must be physical page IDs in "
            f"[0, {window_kv_cache.shape[0]})"
        )

    # TMA loads are scheduled from hca_seq_lens in 128-slot tiles, before
    # sparse_topk_lens masks a query row. Validate every compressed page that
    # those tile-rounded loads can touch; only later table columns are padding.
    compressed_pages_per_tile = (
        _HCA_WINDOW_CAPACITY + compressed_kv_cache.shape[1] - 1
    ) // compressed_kv_cache.shape[1]
    compressed_tiles_per_row = torch.div(
        seq_lens_per_row - _HCA_WINDOW_CAPACITY + _HCA_WINDOW_CAPACITY - 1,
        _HCA_WINDOW_CAPACITY,
        rounding_mode="floor",
    )
    compressed_active_pages = compressed_tiles_per_row * compressed_pages_per_tile
    if (compressed_active_pages > compressed_block_tables.shape[1]).any().item():
        raise ValueError(
            "compressed_block_tables must cover the 128-slot tile-rounded "
            "hca_seq_lens footprint"
        )
    compressed_page_indices = torch.arange(
        compressed_block_tables.shape[1],
        device=compressed_block_tables.device,
    ).unsqueeze(0)
    compressed_page_mask = compressed_page_indices < compressed_active_pages.unsqueeze(
        1
    )
    invalid_compressed_pages = torch.logical_and(
        compressed_page_mask,
        torch.logical_or(
            compressed_block_tables < 0,
            compressed_block_tables >= compressed_kv_cache.shape[0],
        ),
    )
    if invalid_compressed_pages.any().item():
        raise ValueError(
            "compressed_block_tables active values must be physical page IDs in "
            f"[0, {compressed_kv_cache.shape[0]})"
        )


def cute_dsl_hca_decode(
    query: torch.Tensor,
    window_kv_cache: torch.Tensor,
    compressed_kv_cache: torch.Tensor,
    workspace_buffer: torch.Tensor,
    window_block_tables: torch.Tensor,
    compressed_block_tables: torch.Tensor,
    hca_seq_lens: torch.Tensor,
    sparse_topk_lens: torch.Tensor,
    window_valid_lens: torch.Tensor,
    softmax_scale: float,
    output_scale: float = 1.0,
    sinks: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    is_causal: bool = True,
    is_persistent: bool = False,
) -> torch.Tensor:
    """Run the FP8 DeepSeek V4 HCA decode kernel on SM100/SM103.

    All tensors use PyTorch-native layouts.  ``query`` is ``[B, Q, H, 512]``;
    each KV pool is ``[num_pages, page_size, 512]``; and each block table is
    ``[B * Q, max_pages]``. The current DSV4 integration is causal-only.
    ``hca_seq_lens`` counts HCA slots (128 window slots plus compressed slots),
    not original uncompressed tokens.
    """
    if query.ndim != 4:
        raise ValueError(f"query must be 4D [B, Q, H, 512], got {query.ndim}D")
    if query.dtype != torch.float8_e4m3fn:
        raise ValueError(f"CuTe DSL HCA requires an FP8 E4M3 query, got {query.dtype}")
    if not query.is_cuda:
        raise ValueError("query must be on a CUDA device")
    compute_capability = torch.cuda.get_device_capability(query.device)
    if compute_capability not in ((10, 0), (10, 3)):
        raise ValueError(
            "CuTe DSL HCA requires SM100/SM103, got "
            f"SM{compute_capability[0]}{compute_capability[1]}"
        )
    if not is_causal:
        raise ValueError("CuTe DSL DSV4 HCA currently supports causal decode only")
    if not query.is_contiguous():
        raise ValueError("query must be contiguous")
    if query.data_ptr() % 16 != 0:
        raise ValueError("query must be 16-byte aligned")
    batch_size, q_len, num_heads, head_dim = query.shape
    if batch_size <= 0 or q_len <= 0:
        raise ValueError(
            f"query batch_size and q_len must be positive, got {batch_size} and {q_len}"
        )
    if not is_persistent:
        _check_nonpersistent_grid(batch_size, q_len)
    if head_dim != _HCA_HEAD_DIM:
        raise ValueError(f"query head dim must be 512, got {head_dim}")
    if num_heads <= 0 or num_heads > _HCA_MAX_HEADS:
        raise ValueError(f"query num_heads must be in [1, 128], got {num_heads}")

    for cache, name in (
        (window_kv_cache, "window_kv_cache"),
        (compressed_kv_cache, "compressed_kv_cache"),
    ):
        if cache.ndim != 3:
            raise ValueError(f"{name} must be 3D, got {cache.ndim}D")
        if cache.shape[0] <= 0 or cache.shape[1] <= 0:
            raise ValueError(
                f"{name} num_pages and page_size must be positive, got "
                f"{cache.shape[0]} and {cache.shape[1]}"
            )
        if cache.shape[-1] != _HCA_HEAD_DIM:
            raise ValueError(f"{name} head dim must be 512, got {cache.shape[-1]}")
        if cache.dtype != query.dtype:
            raise ValueError(f"{name} dtype must match query dtype, got {cache.dtype}")
        if cache.device != query.device:
            raise ValueError(f"{name} must be on {query.device}, got {cache.device}")
        if not cache.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
        if cache.data_ptr() % 16 != 0:
            raise ValueError(f"{name} must be 16-byte aligned")

    page_size_window = window_kv_cache.shape[1]
    page_size_compressed = compressed_kv_cache.shape[1]
    table_rows = batch_size * q_len if is_causal else batch_size
    if window_block_tables.ndim != 2:
        raise ValueError("window_block_tables must be 2D")
    if compressed_block_tables.ndim != 2:
        raise ValueError("compressed_block_tables must be 2D")
    _check_tensor(
        window_block_tables,
        name="window_block_tables",
        shape=(table_rows, window_block_tables.shape[1]),
        dtype=torch.int32,
        device=query.device,
    )
    _check_tensor(
        compressed_block_tables,
        name="compressed_block_tables",
        shape=(table_rows, compressed_block_tables.shape[1]),
        dtype=torch.int32,
        device=query.device,
    )
    if window_block_tables.shape[1] * page_size_window < _HCA_WINDOW_CAPACITY:
        raise ValueError(
            "window_block_tables must cover all 128 window slots; got capacity "
            f"{window_block_tables.shape[1] * page_size_window}"
        )
    if compressed_block_tables.shape[1] == 0:
        raise ValueError("compressed_block_tables must contain at least one page")

    _check_tensor(
        hca_seq_lens,
        name="hca_seq_lens",
        shape=(batch_size,),
        dtype=torch.int32,
        device=query.device,
    )
    for lengths, name in (
        (sparse_topk_lens, "sparse_topk_lens"),
        (window_valid_lens, "window_valid_lens"),
    ):
        _check_tensor(
            lengths,
            name=name,
            shape=(table_rows,),
            dtype=torch.int32,
            device=query.device,
        )

    _validate_hca_values(
        window_kv_cache=window_kv_cache,
        compressed_kv_cache=compressed_kv_cache,
        window_block_tables=window_block_tables,
        compressed_block_tables=compressed_block_tables,
        hca_seq_lens=hca_seq_lens,
        sparse_topk_lens=sparse_topk_lens,
        window_valid_lens=window_valid_lens,
        q_len=q_len,
        is_causal=is_causal,
    )

    workspace_buffer = _as_cute_dsl_workspace_i8(workspace_buffer)
    if workspace_buffer.device != query.device:
        raise ValueError(
            "workspace_buffer must be on the query device, got "
            f"{workspace_buffer.device} and {query.device}"
        )

    max_hca_len = _HCA_WINDOW_CAPACITY + (
        compressed_block_tables.shape[1] * page_size_compressed
    )
    with torch.cuda.device(query.device):
        max_active_blocks = (
            get_max_active_clusters(_CLUSTER_SHAPE_MNK[0] * _CLUSTER_SHAPE_MNK[1])
            * _CLUSTER_SHAPE_MNK[0]
        )
    split_kv, workspace_size = _get_split_kv_and_workspace_size(
        batch_size,
        q_len,
        num_heads,
        max_hca_len,
        max_active_blocks,
    )
    _check_can_implement(
        q_len,
        num_heads,
        max_hca_len,
        split_kv,
        page_size_compressed,
        page_size_window,
        is_persistent,
    )
    if workspace_buffer.numel() < workspace_size:
        raise ValueError(
            f"workspace_buffer too small: got {workspace_buffer.numel()} bytes, "
            f"need {workspace_size} bytes"
        )
    if workspace_size > 0 and workspace_buffer.data_ptr() % 32 != 0:
        raise ValueError("workspace_buffer must be 32-byte aligned")
    workspace = None if workspace_size == 0 else workspace_buffer[:workspace_size]

    output_shape = (batch_size, q_len, num_heads, _HCA_HEAD_DIM)
    if out is None:
        out = torch.empty(output_shape, dtype=torch.bfloat16, device=query.device)
    else:
        _check_tensor(
            out,
            name="out",
            shape=output_shape,
            dtype=torch.bfloat16,
            device=query.device,
        )
    # HCA stores split-reduction LSE in log2 space. It is an internal
    # scratch/result today because the existing DSV4 API returns only output.
    lse = torch.empty(
        (batch_size, q_len, num_heads),
        dtype=torch.float32,
        device=query.device,
    )

    if not math.isfinite(softmax_scale) or softmax_scale <= 0.0:
        raise ValueError(f"softmax_scale must be positive, got {softmax_scale}")
    if not math.isfinite(output_scale):
        raise ValueError(f"output_scale must be finite, got {output_scale}")
    sink_unscaled = torch.full(
        (_HCA_MAX_HEADS,),
        -float("inf"),
        dtype=torch.float32,
        device=query.device,
    )
    if sinks is not None:
        _check_tensor(
            sinks,
            name="sinks",
            shape=(num_heads,),
            dtype=torch.float32,
            device=query.device,
        )
        sink_unscaled[:num_heads].copy_(sinks).div_(softmax_scale)

    with torch.cuda.device(query.device):
        compiled_kernel = _compile_hca_kernel(
            page_size_compressed,
            page_size_window,
            q_len,
            is_causal,
            is_persistent,
            workspace_size == 0,
            compute_capability,
        )
        compiled_kernel(
            query,
            window_kv_cache,
            compressed_kv_cache,
            window_block_tables,
            compressed_block_tables,
            out,
            lse,
            workspace,
            Int32(split_kv),
            hca_seq_lens,
            None,
            sparse_topk_lens,
            window_valid_lens,
            Float32(softmax_scale),
            Float32(output_scale),
            sink_unscaled,
        )
    return out


__all__ = ["cute_dsl_hca_decode"]
