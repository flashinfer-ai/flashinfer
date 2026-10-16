# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Materialize the active pages of a paged NVFP4 KV cache as FP8 E4M3.

This is the fallback used when an FP8 attention kernel exists for a shape but
no kernel consumes NVFP4 K/V directly (for example trtllm-gen context attention
at head_dim 512). One kernel gathers the pages referenced by a block table,
writes them to an FP8 K/V cache, and writes the block table that indexes it.

Only the per-block E4M3 scale is applied (``E2M1 * E4M3 -> E4M3``, the same
rounding the fused PrimTS decode path performs). The tensor-level K/V scales
stay with the caller, who passes them to the FP8 attention kernel exactly as
for a native NVFP4 kernel.

Request ``b``'s page ``j`` lands in output page ``page_indptr[b] + j + 1``.
Callers that know each request's page count on the host pass that prefix sum
so the FP8 copy holds only active pages; otherwise ``page_indptr[b]`` is
``b * max_pages``. There is no host synchronization, and the launch shape
depends only on ``batch_size`` and ``max_kv_len``.
"""

import functools
import os
from typing import Callable, Optional, Tuple

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int16, Int32, Uint32

from ...api_logging import flashinfer_api
from ...cute_dsl.fp4_common import (
    get_ptr_as_int64,
    ld_global_nc_u32,
    st_global_v4_u32,
)
from ...cute_dsl.utils import get_num_sm
from ...jit.cute_dsl_core import build_and_load_cute_dsl_kernel
from ..quantization_cute_dsl_utils import NVFP4_SF_VEC_SIZE, WARP_SIZE


# Memory-bound: each CTA converts one page at a time, so throughput tracks
# resident threads per SM and larger CTAs loop fewer times per page.
_WARPS_PER_BLOCK = 32
_THREADS_PER_BLOCK = _WARPS_PER_BLOCK * WARP_SIZE
_MAX_THREADS_PER_SM = 2048
_BLOCKS_PER_SM = _MAX_THREADS_PER_SM // _THREADS_PER_BLOCK
# 128-bit global accesses: base and page-stride alignment of every tensor.
_ALIGNMENT_BYTES = 16


def _kernel_source_files() -> tuple[str, ...]:
    """Source files whose content invalidates the on-disk kernel cache."""
    from ...cute_dsl import fp4_common

    return (__file__, fp4_common.__file__)


@cute.jit
def _broadcast_low_byte(value: Int32) -> Int32:
    """Broadcast the low byte of ``value`` into four E4M3 scale lanes."""
    return cute.arch.inline_ptx(
        "prmt.b32 {$w0}, {$r0}, {$r0}, 0x0000;",
        write_only_types=[Int32],
        read_only_args=[value],
    )


@cute.jit
def _mul_packed_e2m1x4_e4m3x4(packed_fp4: Int16, packed_sf: Int32) -> Int32:
    """Apply four E4M3 scales to four contiguous packed E2M1 nibbles."""
    if cutlass.const_expr(cutlass.target_version(min_version="13.4")):
        return cute.arch.inline_ptx(
            "mul.e4m3x4.e2m1x4.e4m3x4.satfinite {$w0}, {$r0}, {$r1};",
            write_only_types=[Int32],
            read_only_args=[packed_fp4, packed_sf],
        )
    else:
        # Fallback for toolchains predating PTX 9.4. The E2M1 * E4M3 product
        # is exact in FP16, so both paths round once to E4M3 and match.
        return cute.arch.inline_ptx(
            """
            {
                .reg .b8 fp4_01, fp4_23;
                .reg .b16 sf_01, sf_23, e4m3_01, e4m3_23;
                .reg .b32 h_01, h_23, sf_h_01, sf_h_23;
                mov.b16 {fp4_01, fp4_23}, {$r0};
                mov.b32 {sf_01, sf_23}, {$r1};
                cvt.rn.f16x2.e2m1x2 h_01, fp4_01;
                cvt.rn.f16x2.e2m1x2 h_23, fp4_23;
                cvt.rn.f16x2.e4m3x2 sf_h_01, sf_01;
                cvt.rn.f16x2.e4m3x2 sf_h_23, sf_23;
                mul.rn.f16x2 h_01, h_01, sf_h_01;
                mul.rn.f16x2 h_23, h_23, sf_h_23;
                cvt.rn.satfinite.e4m3x2.f16x2 e4m3_01, h_01;
                cvt.rn.satfinite.e4m3x2.f16x2 e4m3_23, h_23;
                mov.b32 {$w0}, {e4m3_01, e4m3_23};
            }
            """,
            write_only_types=[Int32],
            read_only_args=[packed_fp4, packed_sf],
        )


def _dequantize_block(packed_0: Uint32, packed_1: Uint32, sf: Int32):
    """Dequantize one 16-element NVFP4 block (8 bytes) to 16 E4M3 bytes.

    Plain Python helper: it is inlined while tracing the kernel.
    """
    return (
        _mul_packed_e2m1x4_e4m3x4(Int16(packed_0 & Uint32(0xFFFF)), sf),
        _mul_packed_e2m1x4_e4m3x4(Int16(packed_0 >> Int32(16)), sf),
        _mul_packed_e2m1x4_e4m3x4(Int16(packed_1 & Uint32(0xFFFF)), sf),
        _mul_packed_e2m1x4_e4m3x4(Int16(packed_1 >> Int32(16)), sf),
    )


class NVFP4KVMaterializeKernel:
    """Gather NVFP4 K/V pages and write them as FP8 in the trtllm-gen layout.
    Each CTA processes one output page at a time (grid-stride over pages); its
    threads walk the page's 16-element scale blocks of K and V. Page 0 is
    zeroed for padded block-table entries. Output page ``p >= 1`` belongs to
    the request ``b`` with ``page_indptr[b] <= p - 1 < page_indptr[b + 1]``
    (one binary search per page), column ``j = p - 1 - page_indptr[b]``.
    Pages past the request's length are skipped, so the work tracks the
    reserved pages. Thread 0 writes the block-table entry of each active page;
    the host zero-fills the table beforehand.
    """

    def __init__(self, head_dim: int):
        if head_dim <= 0 or head_dim % (NVFP4_SF_VEC_SIZE * 4) != 0:
            raise ValueError(
                f"head_dim must be positive and divisible by "
                f"{NVFP4_SF_VEC_SIZE * 4}, got {head_dim}"
            )
        self.blocks_per_row = head_dim // NVFP4_SF_VEC_SIZE
        self.scale_quarter = self.blocks_per_row // 4

    @cute.jit
    def __call__(
        self,
        mKData: cute.Tensor,
        mVData: cute.Tensor,
        mKScales: cute.Tensor,
        mVScales: cute.Tensor,
        mKTables: cute.Tensor,
        mVTables: cute.Tensor,
        mSeqLens: cute.Tensor,
        mPageIndptr: cute.Tensor,
        mOutput: cute.Tensor,
        mBlockTables: cute.Tensor,
        num_out_pages: Int32,
        num_blocks: Int32,
        stream,
    ):
        self.kernel(
            mKData,
            mVData,
            mKScales,
            mVScales,
            mKTables,
            mVTables,
            mSeqLens,
            mPageIndptr,
            mOutput,
            mBlockTables,
            num_out_pages,
        ).launch(
            grid=[num_blocks, 1, 1],
            block=[_THREADS_PER_BLOCK, 1, 1],
            max_number_threads=[_THREADS_PER_BLOCK, 1, 1],
            min_blocks_per_mp=_BLOCKS_PER_SM,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mKData: cute.Tensor,
        mVData: cute.Tensor,
        mKScales: cute.Tensor,
        mVScales: cute.Tensor,
        mKTables: cute.Tensor,
        mVTables: cute.Tensor,
        mSeqLens: cute.Tensor,
        mPageIndptr: cute.Tensor,
        mOutput: cute.Tensor,
        mBlockTables: cute.Tensor,
        num_out_pages: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        grid_dim_x, _, _ = cute.arch.grid_dim()

        num_heads = Int32(mKData.shape[1])
        page_size = Int32(mKData.shape[2])
        batch_size = Int32(mSeqLens.shape[0])
        sf_per_page = num_heads * page_size * Int32(self.blocks_per_row)
        zero = Uint32(0)
        # Each CTA owns one output page at a time; its threads walk the page's
        # scale blocks, so page lookup happens once per page, not per block.
        out_page = bidx
        while out_page < num_out_pages:
            if out_page == Int32(0):
                e = tidx
                while e < sf_per_page:
                    row = e // self.blocks_per_row
                    output_byte = (e % self.blocks_per_row) * Int32(NVFP4_SF_VEC_SIZE)
                    head_idx = row // page_size
                    token_idx = row % page_size
                    st_global_v4_u32(
                        get_ptr_as_int64(
                            mOutput[Int32(0), Int32(0), head_idx, token_idx, None],
                            output_byte,
                        ),
                        zero,
                        zero,
                        zero,
                        zero,
                    )
                    st_global_v4_u32(
                        get_ptr_as_int64(
                            mOutput[Int32(0), Int32(1), head_idx, token_idx, None],
                            output_byte,
                        ),
                        zero,
                        zero,
                        zero,
                        zero,
                    )
                    e = e + _THREADS_PER_BLOCK
            else:
                slot = out_page - Int32(1)
                # Largest req with page_indptr[req] <= slot (uniform per CTA).
                lo = Int32(0)
                hi = batch_size
                while hi - lo > Int32(1):
                    mid = (lo + hi) // Int32(2)
                    if Int32(mPageIndptr[mid]) <= slot:
                        lo = mid
                    else:
                        hi = mid
                req = lo
                col = slot - Int32(mPageIndptr[req])
                # col < ceil(seq_len / page_size) without a division. Since
                # seq_len <= max_kv_len, active columns fit the block table.
                if col * page_size < Int32(mSeqLens[req]):
                    if tidx == Int32(0):
                        mBlockTables[req, col] = out_page
                    k_page = Int32(mKTables[req, col])
                    v_page = Int32(mVTables[req, col])
                    e = tidx
                    while e < sf_per_page:
                        row = e // self.blocks_per_row
                        block_idx = e % self.blocks_per_row
                        head_idx = row // page_size
                        token_idx = row % page_size

                        input_byte = block_idx * Int32(8)
                        k_input_ptr = get_ptr_as_int64(
                            mKData[k_page, head_idx, token_idx, None], input_byte
                        )
                        v_input_ptr = get_ptr_as_int64(
                            mVData[v_page, head_idx, token_idx, None], input_byte
                        )
                        k_packed_0 = ld_global_nc_u32(k_input_ptr)
                        k_packed_1 = ld_global_nc_u32(k_input_ptr + Int32(4))
                        v_packed_0 = ld_global_nc_u32(v_input_ptr)
                        v_packed_1 = ld_global_nc_u32(v_input_ptr + Int32(4))

                        k_sf = _broadcast_low_byte(
                            Int32(mKScales[k_page, head_idx, token_idx, block_idx])
                        )
                        # V block scales use trtllm-gen's 4-token-interleaved
                        # layout. Map the logical (token, block) coordinate.
                        v_token_idx = (token_idx // Int32(4)) * Int32(
                            4
                        ) + block_idx // self.scale_quarter
                        v_scale_idx = (block_idx % self.scale_quarter) * Int32(
                            4
                        ) + token_idx % Int32(4)
                        v_sf = _broadcast_low_byte(
                            Int32(mVScales[v_page, head_idx, v_token_idx, v_scale_idx])
                        )

                        k_out = _dequantize_block(k_packed_0, k_packed_1, k_sf)
                        v_out = _dequantize_block(v_packed_0, v_packed_1, v_sf)
                        output_byte = block_idx * Int32(NVFP4_SF_VEC_SIZE)
                        st_global_v4_u32(
                            get_ptr_as_int64(
                                mOutput[out_page, Int32(0), head_idx, token_idx, None],
                                output_byte,
                            ),
                            Uint32(k_out[0]),
                            Uint32(k_out[1]),
                            Uint32(k_out[2]),
                            Uint32(k_out[3]),
                        )
                        st_global_v4_u32(
                            get_ptr_as_int64(
                                mOutput[out_page, Int32(1), head_idx, token_idx, None],
                                output_byte,
                            ),
                            Uint32(v_out[0]),
                            Uint32(v_out[1]),
                            Uint32(v_out[2]),
                            Uint32(v_out[3]),
                        )
                        e = e + _THREADS_PER_BLOCK
            out_page = out_page + grid_dim_x


@functools.cache
def _get_compiled_kernel(head_dim: int) -> Callable:
    physical_pages = cute.sym_int()
    batch_size = cute.sym_int()
    table_cols = cute.sym_int()
    max_pages = cute.sym_int()
    output_pages = cute.sym_int()
    num_heads = cute.sym_int()
    page_size = cute.sym_int(divisibility=4)
    data_dim = head_dim // 2
    scale_dim = head_dim // NVFP4_SF_VEC_SIZE

    def fake_paged(last_dim):
        # Physical pages may be strided (e.g. scales embedded in a larger
        # page); the layout within a page is compact.
        return cute.runtime.make_fake_tensor(
            cutlass.Uint8,
            (physical_pages, num_heads, page_size, last_dim),
            stride=(
                cute.sym_int64(divisibility=_ALIGNMENT_BYTES),
                page_size * last_dim,
                last_dim,
                1,
            ),
            assumed_align=_ALIGNMENT_BYTES,
        )

    def fake_page_table():
        # Row-strided views such as block_tables[:, 0] of a [B, 2, C] table.
        return cute.runtime.make_fake_tensor(
            Int32,
            (batch_size, table_cols),
            stride=(cute.sym_int(), 1),
            assumed_align=4,
        )

    output_fake = cute.runtime.make_fake_tensor(
        cutlass.Float8E4M3FN,
        (output_pages, 2, num_heads, page_size, head_dim),
        # 64-bit page stride: large workspaces exceed 2**31 bytes.
        stride=(
            cute.sym_int64(divisibility=_ALIGNMENT_BYTES),
            cute.sym_int(divisibility=_ALIGNMENT_BYTES),
            page_size * head_dim,
            head_dim,
            1,
        ),
        assumed_align=_ALIGNMENT_BYTES,
    )
    seq_lens_fake = cute.runtime.make_fake_compact_tensor(
        Int32, (batch_size,), stride_order=(0,), assumed_align=4
    )
    page_indptr_fake = cute.runtime.make_fake_compact_tensor(
        Int32, (cute.sym_int(),), stride_order=(0,), assumed_align=4
    )
    block_tables_fake = cute.runtime.make_fake_compact_tensor(
        Int32, (batch_size, max_pages), stride_order=(1, 0), assumed_align=4
    )
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    kernel_obj = NVFP4KVMaterializeKernel(head_dim)
    compile_options = "--enable-tvm-ffi --opt-level 3"
    if os.getenv("CUTE_DSL_KEEP"):
        compile_options += " --keep-ptx --keep-cubin"

    def compile_kernel() -> Callable:
        return cute.compile(
            kernel_obj,
            fake_paged(data_dim),
            fake_paged(data_dim),
            fake_paged(scale_dim),
            fake_paged(scale_dim),
            fake_page_table(),
            fake_page_table(),
            seq_lens_fake,
            page_indptr_fake,
            output_fake,
            block_tables_fake,
            Int32(1),
            Int32(1),
            stream_fake,
            options=compile_options,
        )

    return build_and_load_cute_dsl_kernel(
        "nvfp4_kv_materialize",
        f"h{head_dim}",
        compile_kernel,
        extra_key_files=_kernel_source_files(),
    )


def _check_aligned(name: str, tensor: torch.Tensor) -> None:
    if tensor.data_ptr() % _ALIGNMENT_BYTES != 0:
        raise ValueError(f"{name} must be {_ALIGNMENT_BYTES}-byte aligned")
    if (tensor.stride(0) * tensor.element_size()) % _ALIGNMENT_BYTES != 0:
        raise ValueError(
            f"{name} page stride must be a multiple of {_ALIGNMENT_BYTES} bytes, "
            f"got {tensor.stride(0) * tensor.element_size()}"
        )


@flashinfer_api
def nvfp4_kv_materialize_fp8(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k_block_scales: torch.Tensor,
    v_block_scales: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    max_kv_len: int,
    uses_shared_paged_kv_idx: bool = True,
    out: Optional[torch.Tensor] = None,
    kv_page_indptr: Optional[torch.Tensor] = None,
    num_pages: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Materialize the pages referenced by a paged NVFP4 KV cache as FP8.

    Requires SM100+.

    Parameters
    ----------
    k_cache, v_cache : torch.Tensor
        Packed NVFP4 K/V in HND layout, ``[num_pages, num_kv_heads, page_size,
        head_dim // 2]``, dtype ``torch.uint8``. The page stride may be
        larger than a compact page.
    k_block_scales, v_block_scales : torch.Tensor
        E4M3 block scales stored as ``torch.uint8``,
        ``[num_pages, num_kv_heads, page_size, head_dim // 16]``. K scales are
        linear; V scales use trtllm-gen's 4-token-interleaved layout, as
        produced by :func:`flashinfer.fp4_quantization.nvfp4_quantize_paged_kv_cache`.
    block_tables : torch.Tensor
        ``int32`` page table, ``[batch_size, max_num_pages]`` when
        ``uses_shared_paged_kv_idx`` else ``[batch_size, 2, max_num_pages]``.
        Entries for the first ``ceil(seq_lens[b] / page_size)`` columns must be
        valid page indices; later entries are ignored.
    seq_lens : torch.Tensor
        ``int32`` KV lengths, ``[batch_size]``.
    max_kv_len : int
        Upper bound on ``seq_lens``. Determines the workspace capacity.
    uses_shared_paged_kv_idx : bool
        Whether K and V share page indices.
    out : Optional[torch.Tensor]
        Optional contiguous ``float8_e4m3fn`` workspace with the returned
        shape, reused across calls.
    kv_page_indptr : Optional[torch.Tensor]
        Optional ``int32 [batch_size + 1]`` CUDA prefix sum of the pages each
        request reserves in the FP8 copy, starting at 0. Request ``b`` must get
        at least ``ceil(seq_lens[b] / page_size)`` pages. With mixed lengths
        this sizes the copy by the active pages instead of
        ``batch_size * max_pages_per_seq``. Requires ``num_pages``.
    num_pages : Optional[int]
        Host-side ``kv_page_indptr[-1]``: the number of pages in the FP8 copy.

    Returns
    -------
    kv_cache_fp8 : torch.Tensor
        ``[num_pages + 1, 2, num_kv_heads, page_size, head_dim]``. Without
        ``kv_page_indptr``, ``num_pages = batch_size * max_pages_per_seq`` with
        ``max_pages_per_seq = ceil(max_kv_len / page_size)``. Request ``b``'s
        page ``j`` is at ``kv_page_indptr[b] + j + 1`` (``b * max_pages_per_seq + j + 1``
        by default); page 0 is zero. Pages past each request's length are not written.
    fp8_block_tables : torch.Tensor
        ``int32 [batch_size, max_pages_per_seq]`` page table into ``kv_cache_fp8``
        (padded entries are 0), for use with ``uses_shared_paged_kv_idx=True``.

    Note
    ----
    Only E2M1 values and E4M3 block scales are multiplied. Apply the
    tensor-level K/V scales in the downstream FP8 attention kernel.
    """
    caches = (
        ("k_cache", k_cache),
        ("v_cache", v_cache),
        ("k_block_scales", k_block_scales),
        ("v_block_scales", v_block_scales),
    )
    device = k_cache.device
    for name, tensor in caches:
        if tensor.ndim != 4 or tensor.dtype != torch.uint8:
            raise ValueError(f"{name} must be a 4D torch.uint8 tensor")
        if tensor.device != device or not tensor.is_cuda:
            raise ValueError("all inputs must be on the same CUDA device")
        _check_aligned(name, tensor)
    if k_cache.shape != v_cache.shape:
        raise ValueError(
            f"K/V cache shapes must match, got {k_cache.shape} and {v_cache.shape}"
        )
    num_physical_pages, num_kv_heads, page_size, packed_dim = k_cache.shape
    head_dim = packed_dim * 2
    if head_dim % (NVFP4_SF_VEC_SIZE * 4) != 0:
        raise ValueError(
            f"head_dim must be divisible by {NVFP4_SF_VEC_SIZE * 4}, got {head_dim}"
        )
    if page_size % 4 != 0:
        raise ValueError(f"page_size must be divisible by 4, got {page_size}")
    scale_dim = head_dim // NVFP4_SF_VEC_SIZE
    expected_scale_shape = (num_physical_pages, num_kv_heads, page_size, scale_dim)
    for name, tensor in caches[2:]:
        if tuple(tensor.shape) != expected_scale_shape:
            raise ValueError(
                f"{name} must have shape {expected_scale_shape}, "
                f"got {tuple(tensor.shape)}"
            )
    for name, tensor, last_dim in (
        ("k_cache", k_cache, packed_dim),
        ("v_cache", v_cache, packed_dim),
        ("k_block_scales", k_block_scales, scale_dim),
        ("v_block_scales", v_block_scales, scale_dim),
    ):
        expected = (page_size * last_dim, last_dim, 1)
        if tuple(tensor.stride()[1:]) != expected:
            raise ValueError(
                f"{name} must be compact within a page (strides {expected}), "
                f"got {tensor.stride()[1:]}"
            )

    if seq_lens.ndim != 1 or seq_lens.dtype != torch.int32:
        raise ValueError("seq_lens must be a 1D torch.int32 tensor")
    if block_tables.dtype != torch.int32 or block_tables.stride(-1) != 1:
        raise ValueError("block_tables must be torch.int32 with unit last-dim stride")
    if seq_lens.device != device or block_tables.device != device:
        raise ValueError("all inputs must be on the same CUDA device")
    batch_size = seq_lens.shape[0]
    expected_ndim = 2 if uses_shared_paged_kv_idx else 3
    if block_tables.ndim != expected_ndim or block_tables.shape[0] != batch_size:
        raise ValueError(
            f"block_tables must be {expected_ndim}D with batch_size rows, "
            f"got shape {tuple(block_tables.shape)}"
        )
    if not uses_shared_paged_kv_idx and block_tables.shape[1] != 2:
        raise ValueError("separate K/V block_tables must have shape [B, 2, C]")
    if max_kv_len <= 0:
        raise ValueError(f"max_kv_len must be positive, got {max_kv_len}")
    max_pages_per_seq = (max_kv_len + page_size - 1) // page_size
    if block_tables.shape[-1] < max_pages_per_seq:
        raise ValueError(
            f"block_tables has {block_tables.shape[-1]} columns, fewer than "
            f"ceil(max_kv_len / page_size) = {max_pages_per_seq}"
        )

    if kv_page_indptr is None:
        if num_pages is not None:
            raise ValueError("num_pages is only used together with kv_page_indptr")
        num_pages = batch_size * max_pages_per_seq
        kv_page_indptr = torch.arange(
            0,
            num_pages + 1,
            max_pages_per_seq,
            dtype=torch.int32,
            device=device,
        )
    else:
        if num_pages is None or num_pages <= 0:
            raise ValueError("kv_page_indptr requires a positive num_pages")
        if (
            kv_page_indptr.dtype != torch.int32
            or kv_page_indptr.shape != (batch_size + 1,)
            or kv_page_indptr.device != device
            or not kv_page_indptr.is_contiguous()
        ):
            raise ValueError(
                "kv_page_indptr must be a contiguous int32 [batch_size + 1] "
                "tensor on the input device"
            )
    out_shape = (num_pages + 1, 2, num_kv_heads, page_size, head_dim)
    if out is None:
        out = torch.empty(out_shape, dtype=torch.float8_e4m3fn, device=device)
    else:
        if tuple(out.shape) != out_shape or out.dtype != torch.float8_e4m3fn:
            raise ValueError(
                f"out must be float8_e4m3fn with shape {out_shape}, "
                f"got {out.dtype} {tuple(out.shape)}"
            )
        if out.device != device or not out.is_contiguous():
            raise ValueError("out must be a contiguous tensor on the input device")
        _check_aligned("out", out)

    # One extra output page: page 0 is zeroed by the same launch.
    num_out_pages = num_pages + 1
    if num_out_pages >= 2**31:
        raise ValueError(f"too many pages for 32-bit indexing: {num_pages}")
    num_blocks = min(num_out_pages, get_num_sm(device) * _BLOCKS_PER_SM)

    if uses_shared_paged_kv_idx:
        k_tables = v_tables = block_tables
    else:
        k_tables, v_tables = block_tables[:, 0], block_tables[:, 1]
    fp8_block_tables = torch.zeros(
        (batch_size, max_pages_per_seq), dtype=torch.int32, device=device
    )
    _get_compiled_kernel(head_dim)(
        k_cache,
        v_cache,
        k_block_scales,
        v_block_scales,
        k_tables,
        v_tables,
        seq_lens,
        kv_page_indptr,
        out,
        fp8_block_tables,
        num_out_pages,
        num_blocks,
    )
    return out, fp8_block_tables
