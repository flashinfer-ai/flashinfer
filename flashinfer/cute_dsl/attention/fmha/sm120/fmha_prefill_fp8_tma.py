# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
Fused multi-head attention (FMHA) FP8 kernel for the NVIDIA Blackwell GeForce
SM120 architecture using TMA K/V loads.

This module implements fused multi-head attention using
warp-level ``mma.sync.aligned.m16n8k32`` tensor cores and TMA loads for K/V
tiles. The implementation fuses the Q*K^T matrix multiplication, online softmax
normalization, and softmax(Q*K^T)*V into a single kernel, avoiding intermediate
data movement through global memory.

The kernel implements key optimizations including:
- A dedicated load warp that streams K/V tiles into shared memory using TMA and
  arrival/consumed mbarriers
- Q fragments loaded directly from global memory into registers to reduce shared
  memory footprint
- Online softmax fused into the main loop, with per-lane registers and intra-warp
  threadquad shuffles for max/sum reductions across the 4 lanes that share a Q-row
- FP8 probability packing that feeds the P*V tensor-core path directly from
  registers
- Shared-memory epilogue staging that aliases the K/V buffer after compute warps
  finish reading the last tile
- Optional causal masking for autoregressive models
- Grouped-query attention (GQA), packed variable lengths, and paged K/V pools

The kernel is compiled and launched through the ``cute-dsl-prims`` backend.
Use ``benchmarks/flashinfer_benchmark.py`` for correctness and performance runs.

Constraints:
* Supported input dtype: Float8E4M3FN
* Supported output dtypes: Float16 and BFloat16
* Head dimension must be exactly 32, 64, 128, or 256
* Query and K/V sequence tiles must be 64 or 128 rows
* Q/O and ragged K/V use packed contiguous tensors plus cumulative sequence
  offsets; paged K/V uses HND ``[num_pages, Hkv, num_tokens_per_page, D]`` pools
  and a shared K/V block table in ``block_tables[B, max_pages]``
* Q head count must be divisible by K/V head count
* ``max(kv_pipeline_stages * kv_tile * head_dim * input_dtype_size,
  q_tile * head_dim * output_dtype_size) + 16 * kv_pipeline_stages`` must fit
  within the SM120 SMEM capacity. ``kv_pipeline_stages`` is 3 when
  ``head_dim == 256`` and ``q_tile == kv_tile == 128``, and 2 otherwise
"""

from functools import partial
from types import SimpleNamespace
from typing import Type

import cuda.bindings.driver as cuda_driver
import cutlass
import cutlass.cute as cute
from cutlass.cute.arch import add_packed_f32x2 as cute_add_packed_f32x2
from cutlass.cute.arch import fma_packed_f32x2 as cute_fma_packed_f32x2
from cutlass.cute.arch import mul_packed_f32x2 as cute_mul_packed_f32x2
import cutlass.experimental.cuda as cuda
import cutlass.utils as cutlass_utils

from cutlass.experimental import primitives as prims

# The helper section below retains its original NVIDIA proprietary notice:
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

SUPPORTED_PAGE_SIZES = (16, 32, 64, 128)

# ---------------------------------------------------------------------------
# PTX helpers.
# ---------------------------------------------------------------------------


@cute.jit
def nvvm_threadquad_reduction_max(val: cutlass.Float32) -> cutlass.Float32:
    """Reduce the four lanes that jointly own one softmax row."""
    # XOR distances 2 then 1 cover one four-lane MMA row group without mixing
    # it with the adjacent row group in the same warp.
    val = cute.arch.fmax(
        val,
        prims.shfl_sync(
            thread_mask=0xFFFFFFFF,
            val=val,
            offset=2,
            mask_and_clamp=0x1C03,
            kind=prims.Shfl.BFLY,
        ),
    )
    val = cute.arch.fmax(
        val,
        prims.shfl_sync(
            thread_mask=0xFFFFFFFF,
            val=val,
            offset=1,
            mask_and_clamp=0x1C03,
            kind=prims.Shfl.BFLY,
        ),
    )
    return val


@cute.jit
def nvvm_threadquad_reduction_sum(val: cutlass.Float32) -> cutlass.Float32:
    """Reduce the four lanes that jointly own one softmax row."""
    # Keep the same four-lane butterfly topology as the max reduction above.
    val = val + prims.shfl_sync(
        thread_mask=0xFFFFFFFF,
        val=val,
        offset=2,
        mask_and_clamp=0x1C03,
        kind=prims.Shfl.BFLY,
    )
    val = val + prims.shfl_sync(
        thread_mask=0xFFFFFFFF,
        val=val,
        offset=1,
        mask_and_clamp=0x1C03,
        kind=prims.Shfl.BFLY,
    )
    return val


@cute.jit
def nvvm_threadquad_reduction_max_full(val: cutlass.Float32) -> cutlass.Float32:
    """Original full-warp encoding of a four-lane max reduction."""
    val = cute.arch.fmax(
        val,
        prims.shfl_sync(
            thread_mask=0xFFFFFFFF,
            val=val,
            offset=2,
            mask_and_clamp=0x1F,
            kind=prims.Shfl.BFLY,
        ),
    )
    val = cute.arch.fmax(
        val,
        prims.shfl_sync(
            thread_mask=0xFFFFFFFF,
            val=val,
            offset=1,
            mask_and_clamp=0x1F,
            kind=prims.Shfl.BFLY,
        ),
    )
    return val


@cute.jit
def nvvm_threadquad_reduction_sum_full(val: cutlass.Float32) -> cutlass.Float32:
    """Original full-warp encoding of a four-lane sum reduction."""
    val = val + prims.shfl_sync(
        thread_mask=0xFFFFFFFF,
        val=val,
        offset=2,
        mask_and_clamp=0x1F,
        kind=prims.Shfl.BFLY,
    )
    val = val + prims.shfl_sync(
        thread_mask=0xFFFFFFFF,
        val=val,
        offset=1,
        mask_and_clamp=0x1F,
        kind=prims.Shfl.BFLY,
    )
    return val


@cute.jit
def ptx_mma_m16n8k32_f32(
    a0: cutlass.Int32,
    a1: cutlass.Int32,
    a2: cutlass.Int32,
    a3: cutlass.Int32,
    b0: cutlass.Int32,
    b1: cutlass.Int32,
    c0: cutlass.Float32,
    c1: cutlass.Float32,
    c2: cutlass.Float32,
    c3: cutlass.Float32,
    ab_dtype: cutlass.Constexpr[Type[cutlass.Numeric]],
) -> tuple[cutlass.Float32, cutlass.Float32, cutlass.Float32, cutlass.Float32]:
    """``mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32``."""
    if cutlass.const_expr(ab_dtype != cutlass.Float8E4M3FN):
        raise TypeError(f"Invalid A/B dtype: {ab_dtype}")
    return cute.arch.inline_ptx(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32"
        " {$0,$1,$2,$3}, {$4,$5,$6,$7}, {$8,$9}, {$10,$11,$12,$13};",
        write_only_types=[
            cutlass.Float32,
            cutlass.Float32,
            cutlass.Float32,
            cutlass.Float32,
        ],
        read_only_args=[a0, a1, a2, a3, b0, b1, c0, c1, c2, c3],
    )


@cute.jit
def get_swizzled_col(
    row: cutlass.Int32,
    col: cutlass.Int32,
    row_stride: cutlass.Constexpr[int],
    elem_bytes: cutlass.Constexpr[int],
) -> cutlass.Int32:
    """Return the physical SMEM column for an XOR-swizzled row-major tile.

    XOR swizzling places adjacent rows into distinct SMEM banks. The XOR is
    applied at the 16-byte boundary; ``elem_bytes`` selects the corresponding
    element-domain shift and chunk size.
    """
    row_stride_bytes = row_stride * elem_bytes
    chunk_bytes = 32
    sw_bits = 1
    row_shift = 0
    if row_stride_bytes % 128 == 0:
        chunk_bytes = 128
        sw_bits = 3
    elif row_stride_bytes % 64 == 0:
        chunk_bytes = 64
        sw_bits = 2
        row_shift = 1
    chunk_size = chunk_bytes // elem_bytes
    elems_per_16b = 16 // elem_bytes
    sw_base = elems_per_16b.bit_length() - 1
    chunk = col // chunk_size
    col_in_chunk = col % chunk_size
    bit_msk = (1 << sw_bits) - 1
    return chunk * chunk_size + (
        col_in_chunk ^ (((row >> row_shift) & bit_msk) << sw_base)
    )


@cute.jit
def pack_to_i32(
    src: tuple,
    dtype: cutlass.Constexpr[Type[cutlass.Numeric]],
) -> cutlass.Int32:
    """Pack four 8-bit or two 16-bit values into one 32-bit register."""
    vals = cutlass.Vector.from_elements(src, dtype)
    return vals.bitcast(cutlass.Int32)[0]


@cute.jit
def cvt_f32x4_to_f8x4(
    p0: cutlass.Float32,
    p1: cutlass.Float32,
    p2: cutlass.Float32,
    p3: cutlass.Float32,
    dtype: cutlass.Constexpr[Type[cutlass.Numeric]],
) -> cutlass.Int32:
    """Convert four FP32 values into one packed FP8 MMA operand."""
    if cutlass.const_expr(dtype != cutlass.Float8E4M3FN):
        raise TypeError(f"Invalid FP8 dtype: {dtype}")
    return cute.arch.inline_ptx(
        (
            "{\n"
            "  .reg .b16 lo;\n"
            "  .reg .b16 hi;\n"
            "  cvt.rn.satfinite.e4m3x2.f32 lo, $2, $1;\n"
            "  cvt.rn.satfinite.e4m3x2.f32 hi, $4, $3;\n"
            "  mov.b32 $0, {lo, hi};\n"
            "}"
        ),
        write_only_types=[cutlass.Int32],
        read_only_args=[p0, p1, p2, p3],
    )


def ceil_div(a: int, b: int) -> int:
    """Return the ceiling division of a by b."""
    return (a + b - 1) // b


# Softmax/output FP32 state is sensitive to subnormal flushing and rounding
# drift; bind the packed operations once so both kernels use RN without FTZ.
add_packed_f32x2 = partial(
    cute_add_packed_f32x2,
    ftz=False,
    rnd="rn",
)
mul_packed_f32x2 = partial(
    cute_mul_packed_f32x2,
    ftz=False,
    rnd="rn",
)
fma_packed_f32x2 = partial(
    cute_fma_packed_f32x2,
    ftz=False,
    rnd="rn",
)


@cute.jit
def cvt_f32x2_to_f8x2(
    a: cutlass.Float32,
    b: cutlass.Float32,
    dtype: cutlass.Constexpr[Type[cutlass.Numeric]],
) -> cutlass.Uint16:
    """Convert and pack two FP32 values into one FP8x2 register."""
    if cutlass.const_expr(dtype != cutlass.Float8E4M3FN):
        raise TypeError(f"Invalid FP8 dtype: {dtype}")
    return cute.arch.inline_ptx(
        "cvt.rn.satfinite.e4m3x2.f32 $0, $1, $2;",
        write_only_types=[cutlass.Uint16],
        read_only_args=[a, b],
    )


@cute.jit
def pack_f8x2_pairs(pair0: cutlass.Uint16, pair1: cutlass.Uint16) -> cutlass.Int32:
    """Pack two FP8x2 values into one 32-bit FP8x4 MMA operand."""
    return cute.arch.inline_ptx(
        "mov.b32 $0, {$1, $2};",
        write_only_types=[cutlass.Int32],
        read_only_args=[pair0, pair1],
    )


# ---------------------------------------------------------------------------
# Main kernel class
# ---------------------------------------------------------------------------


class SM120FusedMultiHeadAttentionFP8ForwardTMA:
    """Configure and launch the warp-specialized SM120 FP8 FMHA kernel."""

    SEQ_Q_TILES = [128, 64]
    SEQ_KV_TILES = [128, 64]
    SUPPORTED_HEAD_TILES = [32, 64, 128, 256]
    SUPPORTED_PAGE_SIZES = SUPPORTED_PAGE_SIZES
    MMA_TILER = (16, 8, 32)  # mma.sync.aligned.m16n8k32
    # Barrier 0 is the CTA-wide initialization barrier; barrier 1 synchronizes
    # compute warps before the K/V storage is aliased for the output epilogue.
    COMPUTE_BARRIER_ID = 1
    # K and V operands share a three-slot circular TMA pipeline. D256 uses
    # 96 KiB for these slots, which fits the SM120 shared-memory capacity.
    KV_PIPELINE_STAGES = 3

    def __init__(
        self,
        in_dtype: Type[cutlass.Numeric] = cutlass.Float8E4M3FN,
        out_dtype: Type[cutlass.Numeric] = cutlass.Float16,
        is_causal: bool = False,
        head_tile: int = 128,
        kv_tile: int = SEQ_KV_TILES[0],
        q_tile: int = SEQ_Q_TILES[0],
        use_paged_kv: bool = False,
        num_tokens_per_page: int | None = None,
        balanced_scheduler: bool = False,
    ) -> None:
        """Initialize the FMHA prefill kernel configuration.

        :param in_dtype: Q/K/V element type.
        :param out_dtype: O element type. Must be Float16 or BFloat16.
        :param is_causal: Apply causal masking to QK.
        :param head_tile: Head dimension tile, one of 32, 64, 128, or 256.
        :param q_tile: Query sequence tile size.
        :param kv_tile: Key/Value sequence tile size.
        :param use_paged_kv: Whether K/V are read from paged KV pools.
        :param num_tokens_per_page: Tokens per physical K/V page.
        :param balanced_scheduler: Use the causal load-balanced grid mapping
            and visit Q tiles in reverse order. Compilation must guarantee
            that causal masking is enabled when this option is true.
        """
        # Data types
        if in_dtype != cutlass.Float8E4M3FN:
            raise ValueError("in_dtype must be Float8E4M3FN")
        if out_dtype not in (cutlass.Float16, cutlass.BFloat16):
            raise ValueError("out_dtype must be Float16 or BFloat16")
        self.in_dtype = in_dtype
        self.out_dtype = out_dtype
        self.is_causal = is_causal
        self.use_paged_kv = use_paged_kv
        self.balanced_scheduler = balanced_scheduler
        # All causal paths use bottom-right alignment, so Q is positioned at
        # the end of its logical K/V sequence when their lengths differ.
        self.has_causal_q_offset = self.is_causal

        # Tiling
        if head_tile not in self.SUPPORTED_HEAD_TILES:
            raise ValueError(f"head_tile must be one of {self.SUPPORTED_HEAD_TILES}")
        if q_tile not in self.SEQ_Q_TILES:
            raise ValueError(f"q_tile must be one of {self.SEQ_Q_TILES}")
        if kv_tile not in self.SEQ_KV_TILES:
            raise ValueError(f"kv_tile must be one of {self.SEQ_KV_TILES}")
        self.head_tile = head_tile
        self.q_tile = q_tile
        self.kv_tile = kv_tile
        self.kv_pipeline_stages = (
            3 if head_tile == 256 and q_tile == 128 and kv_tile == 128 else 2
        )
        if self.use_paged_kv:
            if num_tokens_per_page not in self.SUPPORTED_PAGE_SIZES:
                raise ValueError(
                    f"num_tokens_per_page must be one of {self.SUPPORTED_PAGE_SIZES}"
                )
            if self.kv_tile % num_tokens_per_page != 0:
                raise ValueError(
                    "paged KV requires num_tokens_per_page to divide kv_tile"
                )
            self.num_tokens_per_page = num_tokens_per_page
        else:
            if num_tokens_per_page is not None:
                raise ValueError("page size is only valid when use_paged_kv=True")
            self.num_tokens_per_page = self.kv_tile
        # The block-table width is a runtime tensor dimension. Do not specialize
        # on it merely to cache one row in a warp; direct loads keep one compiled
        # artifact valid for every page-table capacity.
        self.cache_page_ids = False

        # Warp layout is compute 0-7/load 8/padding 9-11 for Q128 and compute
        # 0-3/load 4/padding 5-7 for Q64. Padding completes the load warpgroup
        # and donates its register budget via setmaxregister; it is not removable.
        self.compute_warp_ids: tuple[int, ...]
        if self.q_tile == 128:
            self.compute_warp_ids = (0, 1, 2, 3, 4, 5, 6, 7)
            self.load_warp_id = 8
            self.empty_warp_ids = (9, 10, 11)
            self.num_warps = 12
        else:
            self.compute_warp_ids = (0, 1, 2, 3)
            self.load_warp_id = 4
            self.empty_warp_ids = (5, 6, 7)
            self.num_warps = 8
        self.num_compute_warps = len(self.compute_warp_ids)
        self.threads_per_cta = cute.arch.WARP_SIZE * self.num_warps
        self.threads_compute = cute.arch.WARP_SIZE * self.num_compute_warps

        self._setup_attributes()

    def _setup_attributes(self) -> None:
        """Compute derived tile, MMA, and TMA constants from the configuration."""

        # Tiling
        self.qo_tile_elems = self.q_tile * self.head_tile
        self.kv_tile_elems = self.kv_tile * self.head_tile
        self.smem_elems = ceil_div(
            max(
                self.kv_tile_elems * self.kv_pipeline_stages * self.in_dtype.bytes,
                self.qo_tile_elems * self.out_dtype.bytes,
            ),
            self.in_dtype.bytes,
        )

        # MMA
        self.qk_k_frags = self.kv_tile // self.MMA_TILER[1]
        self.qk_d_frags = self.head_tile // self.MMA_TILER[2]
        self.pv_v_frags = self.kv_tile // self.MMA_TILER[2]
        self.pv_d_frags = self.head_tile // self.MMA_TILER[1]

        # TMA
        tma_copy_head_bytes = self.head_tile * self.in_dtype.bytes
        if tma_copy_head_bytes % 128 == 0:
            self.tma_swizzle = cuda.TensorMapSwizzle.s128b
            self.tma_copy_iters = tma_copy_head_bytes // 128
        elif tma_copy_head_bytes % 64 == 0:
            self.tma_swizzle = cuda.TensorMapSwizzle.s64b
            self.tma_copy_iters = tma_copy_head_bytes // 64
        elif tma_copy_head_bytes % 32 == 0:
            self.tma_swizzle = cuda.TensorMapSwizzle.none
            self.tma_copy_iters = tma_copy_head_bytes // 32
        else:
            raise ValueError(
                f"Unsupported TMA inner dimension: {tma_copy_head_bytes} B"
            )

        self.tma_copy_head_per_iter = self.head_tile // self.tma_copy_iters
        self.tma_copy_elems_per_iter = self.kv_tile_elems // self.tma_copy_iters
        self.tma_copy_elems_per_page_iter = (
            self.num_tokens_per_page * self.tma_copy_head_per_iter
        )
        self.kv_page_chunks = self.kv_tile // self.num_tokens_per_page

    @cute.jit
    def _get_swizzled_col(
        self, row: cutlass.Int32, col: cutlass.Int32
    ) -> cutlass.Int32:
        """Return the SMEM column for K/V tiles loaded by this kernel's TMA layout."""
        smem_col = col
        if self.tma_swizzle != cuda.TensorMapSwizzle.none:
            smem_col = get_swizzled_col(
                row,
                col,
                self.tma_copy_head_per_iter,
                self.in_dtype.bytes,
            )
        return smem_col

    @cute.jit
    def get_kv_stage_ptr(
        self, sKV: cutlass.Array, stage: cutlass.Int32
    ) -> cutlass.Pointer:
        """Select one of the three K/V ring slots without a dynamic GEP."""
        ptr = sKV.data_ptr()
        if stage == 1:
            ptr = sKV.data_ptr(self.kv_tile_elems)
        elif stage == 2:
            ptr = sKV.data_ptr(2 * self.kv_tile_elems)
        return ptr

    @cute.jit
    def get_mbar_stage_ptr(
        self, mbars: cutlass.Array, stage: cutlass.Int32
    ) -> cutlass.Pointer:
        """Select one of three ring mbarriers without a dynamic GEP."""
        ptr = mbars.data_ptr()
        if stage == 1:
            ptr = mbars.data_ptr(1)
        elif stage == 2:
            ptr = mbars.data_ptr(2)
        return ptr

    @classmethod
    def _can_implement(
        cls,
        in_dtype: type[cutlass.Numeric],
        out_dtype: type[cutlass.Numeric],
        q_shape: tuple[int, int, int, int],
        k_shape: tuple[int, int, int, int],
        kv_tile: int = SEQ_KV_TILES[0],
        q_tile: int = SEQ_Q_TILES[0],
    ) -> bool:
        """Return whether the kernel supports the structural configuration.

        Validates dtype, exact head-dim support, tile choices, and SM120 SMEM capacity.

        :param in_dtype: Input dtype.
        :param out_dtype: Output dtype.
        :param q_shape: Query shape ``(B, Sq, Hq, D)``.
        :param k_shape: Key/value shape ``(B, Skv, Hkv, D)``.
        :param q_tile: Query sequence tile size.
        :param kv_tile: K/V sequence tile size.
        :return: True if the configuration is supported.
        """
        if len(q_shape) != 4 or len(k_shape) != 4:
            return False
        if in_dtype != cutlass.Float8E4M3FN:
            return False
        if out_dtype not in (cutlass.Float16, cutlass.BFloat16):
            return False

        batch_size, _, num_heads_q, head_dim = q_shape
        batch_size_k, _, num_heads_kv, head_dim_k = k_shape
        if batch_size != batch_size_k:
            return False
        if head_dim != head_dim_k:
            return False
        if num_heads_q <= 0 or num_heads_kv <= 0:
            return False
        if num_heads_q % num_heads_kv != 0:
            return False
        if head_dim not in cls.SUPPORTED_HEAD_TILES:
            return False
        if q_tile not in cls.SEQ_Q_TILES or kv_tile not in cls.SEQ_KV_TILES:
            return False

        kv_pipeline_stages = (
            3 if head_dim == 256 and q_tile == 128 and kv_tile == 128 else 2
        )
        kv_smem_bytes = kv_pipeline_stages * kv_tile * head_dim * in_dtype.bytes
        output_bytes = q_tile * head_dim * out_dtype.bytes
        # K/V and the epilogue output alias one buffer; each stage has one full
        # and one empty mbarrier outside that maximum.
        smem_bytes = max(kv_smem_bytes, output_bytes) + 16 * kv_pipeline_stages
        return smem_bytes <= cutlass_utils.get_smem_capacity_in_bytes("sm_120")

    @classmethod
    def can_implement_paged(
        cls,
        in_dtype: type[cutlass.Numeric],
        out_dtype: type[cutlass.Numeric],
        q_shape: tuple[int, int, int, int],
        k_shape: tuple[int, int, int, int],
        num_tokens_per_page: int,
        kv_tile: int = SEQ_KV_TILES[0],
        q_tile: int = SEQ_Q_TILES[0],
    ) -> bool:
        """Return whether the paged-varlen TMA path supports this configuration."""
        if num_tokens_per_page not in cls.SUPPORTED_PAGE_SIZES:
            return False
        # Validate before modulo so unsupported/non-integral tile values retain
        # the base API's fail-fast False result rather than raising TypeError.
        if kv_tile not in cls.SEQ_KV_TILES:
            return False
        if kv_tile % num_tokens_per_page != 0:
            return False
        return cls._can_implement(
            in_dtype,
            out_dtype,
            q_shape,
            k_shape,
            kv_tile=kv_tile,
            q_tile=q_tile,
        )

    @cute.jit
    def load_one_kv_tile(
        self,
        s_dst: cutlass.Pointer,
        tma_desc_ptr: cutlass.Pointer,
        mbar_arrived: cutlass.Array,
        kv_head_idx: cutlass.Int32,
        seq_coord: cutlass.Int32,
        l2_cache_hint: cutlass.Int64 | None,
    ) -> None:
        """Launch one TMA load for a K/V tile into swizzled SMEM.

        The tensor map is built from packed ``(total_tokens, H, D)`` tensors
        with split D-dimension boxes. Coordinates use the tensor-map convention
        ``(d, head, token)``.

        :param s_dst: Swizzled SMEM destination tile.
        :param tma_desc_ptr: K or V tensor map descriptor pointer.
        :param mbar_arrived: Mbarrier signaled when this TMA tile has arrived.
        :param kv_head_idx: K/V head index.
        :param seq_coord: Starting physical sequence row for the K/V tile.
        :param l2_cache_hint: L2 eviction policy retained across GQA consumers.
        """
        if prims.elect_sync():
            prims.mbarrier_arrive_expect_tx(
                mbar_arrived, self.kv_tile_elems * self.in_dtype.bytes
            )

        for i in cutlass.range_constexpr(self.tma_copy_iters):
            head_offset = i * self.tma_copy_head_per_iter
            coords = (head_offset, kv_head_idx, seq_coord)
            if prims.elect_sync():
                prims.cp_async_bulk_tensor_shared_cta_global(
                    s_dst + i * self.tma_copy_elems_per_iter,
                    tma_desc_ptr,
                    coords,
                    mbar_arrived,
                    l2_cache_hint=l2_cache_hint,
                )

    @cute.jit
    def resolve_paged_kv_tile_pages(
        self,
        batch_idx: cutlass.Int32,
        seq_coord: cutlass.Int32,
        seqlen_kv: cutlass.Int32,
        block_tables: cute.Tensor,
        cached_page_id: cutlass.Int32,
    ) -> tuple:
        """Resolve shared K/V physical pages for one logical K/V tile."""
        page_ids = cutlass.Array(cutlass.Int32, self.kv_page_chunks, alignment=16)
        block_tables_ptr = block_tables.iterator.raw_ptr()
        block_table_offset = batch_idx * cutlass.Int32(block_tables.shape[1])
        base_logical_page = seq_coord // self.num_tokens_per_page
        safe_page_count = cute.math.max(
            (seqlen_kv + self.num_tokens_per_page - 1) // self.num_tokens_per_page,
            cutlass.Int32(1),
        )

        for seq_chunk in cutlass.range_constexpr(self.kv_page_chunks):
            logical_page = base_logical_page + cutlass.Int32(seq_chunk)
            if logical_page >= safe_page_count:
                # TMA still needs a legal full-page address for a partial
                # tile. K tail values are masked before softmax, so clamping
                # cannot make them logically valid. Paged pools, including
                # unused slots, are required to contain finite values.
                logical_page = safe_page_count - cutlass.Int32(1)
            if cutlass.const_expr(self.cache_page_ids):
                page_ids[seq_chunk] = cute.arch.make_warp_uniform(
                    prims.shfl_sync(
                        thread_mask=0xFFFFFFFF,
                        val=cached_page_id,
                        offset=logical_page,
                        mask_and_clamp=0x1F,
                        kind=prims.Shfl.IDX,
                    )
                )
            else:
                page_ids[seq_chunk] = cutlass.Int32(
                    block_tables_ptr[block_table_offset + logical_page]
                )

        return page_ids

    @cute.jit
    def load_one_paged_kv_tile(
        self,
        s_dst: cutlass.Pointer,
        tma_desc_ptr: cutlass.Pointer,
        mbar_arrived: cutlass.Array,
        page_ids: cutlass.Array,
        kv_head_idx: cutlass.Int32,
        l2_cache_hint: cutlass.Int64 | None,
    ) -> None:
        """Launch TMA loads for one logical K/V tile from pre-resolved pages.

        The paged pool uses HND storage
        ``(num_pages, Hkv, num_tokens_per_page, D)``. The
        destination SMEM layout matches the contiguous path:
        ``[D chunk][KV row][D within chunk]``.
        """
        if prims.elect_sync():
            prims.mbarrier_arrive_expect_tx(
                mbar_arrived, self.kv_tile_elems * self.in_dtype.bytes
            )

        token_in_page = cutlass.Int32(0)
        for issue_chunk in cutlass.range_constexpr(self.kv_page_chunks):
            # Preserve the tuned reverse TMA issue order, while deriving page_dst
            # from the logical chunk so MMA still sees left-to-right sequence rows.
            seq_chunk = self.kv_page_chunks - 1 - issue_chunk
            page_id = page_ids[seq_chunk]
            page_dst = seq_chunk * self.tma_copy_elems_per_page_iter

            for i in cutlass.range_constexpr(self.tma_copy_iters):
                head_offset = i * self.tma_copy_head_per_iter
                if prims.elect_sync():
                    prims.cp_async_bulk_tensor_shared_cta_global(
                        s_dst + i * self.tma_copy_elems_per_iter + page_dst,
                        tma_desc_ptr,
                        (
                            head_offset,
                            token_in_page,
                            kv_head_idx,
                            page_id,
                        ),
                        mbar_arrived,
                        l2_cache_hint=l2_cache_hint,
                    )

    @cute.jit
    def load_q_tile(
        self,
        basic_params: SimpleNamespace,
    ) -> cutlass.Array:
        """Load the warp-owned Q tile from GMEM into MMA A registers.

        :param basic_params: Per-CTA tensor metadata, lane mapping, and Q base
            offsets.
        :return: Packed Q fragments arranged for ``mma.sync`` A operands.
        """

        # Each lane loads a contiguous 4-element vector. The shuffles below
        # transpose the four row quads into the m16n8k32 A-fragment layout.
        def load_q_fragment(col0_in_cta: cutlass.Int32):
            q_regs_per_frag = [
                cutlass.Int32(0),
                cutlass.Int32(0),
                cutlass.Int32(0),
                cutlass.Int32(0),
            ]
            for row_quad in cutlass.range_constexpr(self.MMA_TILER[0] // 4):
                row_in_cta = (
                    basic_params.q_warp_row0 + row_quad * 4 + basic_params.lane_div8
                )
                col_in_cta = col0_in_cta + basic_params.lane_mod8 * 4
                cur_q_seq_idx = basic_params.q_seq_idx + row_in_cta
                if (
                    cur_q_seq_idx < basic_params.seqlen_q
                    and col_in_cta < basic_params.head_dim
                ):
                    if cutlass.const_expr(
                        self.head_tile == 256
                        and self.q_tile == 128
                        and self.kv_tile == 128
                    ):
                        q_src = (
                            basic_params.q_ptr
                            + basic_params.q_head_base
                            + cur_q_seq_idx * basic_params.q_seq_stride
                            + col_in_cta
                        )
                        q_packed = prims.load_ext(
                            q_src,
                            dtype=cutlass.Int32,
                            l2_cache_hint=basic_params.q_l2_cache_hint,
                        )
                        q_regs_per_frag[row_quad] = q_packed
                    else:
                        q_vec = (
                            basic_params.q_ptr
                            + basic_params.q_head_base
                            + cur_q_seq_idx * basic_params.q_seq_stride
                            + col_in_cta
                        ).load(count=4, alignment=4)
                        q_regs_per_frag[row_quad] = q_vec.bitcast(cutlass.Int32)[0]
            return q_regs_per_frag

        q_regs = cutlass.Array(cutlass.Int32, self.qk_d_frags * 4, alignment=16)
        src_lane_lo = (basic_params.lane_div4 % 4) * 8 + basic_params.lane_mod4
        src_lane_hi = (basic_params.lane_div4 % 4) * 8 + (basic_params.lane_mod4 + 4)

        q_regs_offset = 0
        col0_in_cta = 0
        for _ in cutlass.range_constexpr(self.qk_d_frags):
            q_rows = load_q_fragment(col0_in_cta)
            q_rows_lo = [
                prims.shfl_sync(
                    thread_mask=0xFFFFFFFF,
                    val=q_rows[0],
                    offset=src_lane_lo,
                    mask_and_clamp=0x1F,
                    kind=prims.Shfl.IDX,
                ),
                prims.shfl_sync(
                    thread_mask=0xFFFFFFFF,
                    val=q_rows[1],
                    offset=src_lane_lo,
                    mask_and_clamp=0x1F,
                    kind=prims.Shfl.IDX,
                ),
                prims.shfl_sync(
                    thread_mask=0xFFFFFFFF,
                    val=q_rows[2],
                    offset=src_lane_lo,
                    mask_and_clamp=0x1F,
                    kind=prims.Shfl.IDX,
                ),
                prims.shfl_sync(
                    thread_mask=0xFFFFFFFF,
                    val=q_rows[3],
                    offset=src_lane_lo,
                    mask_and_clamp=0x1F,
                    kind=prims.Shfl.IDX,
                ),
            ]
            q_rows_hi = [
                prims.shfl_sync(
                    thread_mask=0xFFFFFFFF,
                    val=q_rows[0],
                    offset=src_lane_hi,
                    mask_and_clamp=0x1F,
                    kind=prims.Shfl.IDX,
                ),
                prims.shfl_sync(
                    thread_mask=0xFFFFFFFF,
                    val=q_rows[1],
                    offset=src_lane_hi,
                    mask_and_clamp=0x1F,
                    kind=prims.Shfl.IDX,
                ),
                prims.shfl_sync(
                    thread_mask=0xFFFFFFFF,
                    val=q_rows[2],
                    offset=src_lane_hi,
                    mask_and_clamp=0x1F,
                    kind=prims.Shfl.IDX,
                ),
                prims.shfl_sync(
                    thread_mask=0xFFFFFFFF,
                    val=q_rows[3],
                    offset=src_lane_hi,
                    mask_and_clamp=0x1F,
                    kind=prims.Shfl.IDX,
                ),
            ]

            if basic_params.lane_div4 < 4:
                q_regs[q_regs_offset + 0] = q_rows_lo[0]
                q_regs[q_regs_offset + 1] = q_rows_lo[2]
                q_regs[q_regs_offset + 2] = q_rows_hi[0]
                q_regs[q_regs_offset + 3] = q_rows_hi[2]
            else:
                q_regs[q_regs_offset + 0] = q_rows_lo[1]
                q_regs[q_regs_offset + 1] = q_rows_lo[3]
                q_regs[q_regs_offset + 2] = q_rows_hi[1]
                q_regs[q_regs_offset + 3] = q_rows_hi[3]

            col0_in_cta += self.MMA_TILER[2]
            q_regs_offset += 4

        return q_regs

    @cute.jit
    def mma_qk(
        self,
        basic_params: SimpleNamespace,
        mma_params: SimpleNamespace,
        q_regs: cutlass.Array,
        sK: cutlass.Pointer,
    ) -> cutlass.Array:
        """Compute ``S = Q @ K.T``.

        Q fragments are supplied in registers by ``load_q_tile``. K fragments
        are read from the TMA-populated ``sK`` tile with ``ldmatrix``.

        :param basic_params: Per-CTA tensor metadata and lane mapping.
        :param mma_params: Shared K/V ring and local O accumulator state.
        :param q_regs: Register-resident packed Q fragments.
        :param sK: Shared-memory pointer to this tile's K operand.
        :return: Register-resident QK score fragments.
        """
        s_regs = cutlass.Array(cutlass.Float32, self.qk_k_frags * 4, alignment=16)
        for i in cutlass.range_constexpr(self.qk_k_frags * 4):
            s_regs[i] = cutlass.Float32(0.0)

        k_row_in_frag_pair = (basic_params.lane_div8 // 2) * 8 + basic_params.lane_mod8
        k_col_in_frag_pair = (basic_params.lane_div8 % 2) * 16

        def load_k_fragments(dst, d_offset: cutlass.Constexpr[int]):
            for k_pair in cutlass.range_constexpr(self.qk_k_frags // 2):
                k_row = k_pair * 16 + k_row_in_frag_pair
                k_col = d_offset + k_col_in_frag_pair
                k_chunk = k_col // self.tma_copy_head_per_iter
                k_col_in_chunk = k_col % self.tma_copy_head_per_iter
                sK_ptr = (
                    sK
                    + k_chunk * self.tma_copy_elems_per_iter
                    + k_row * self.tma_copy_head_per_iter
                    + self._get_swizzled_col(k_row, k_col_in_chunk)
                )
                k_vec = prims.ldmatrix(sK_ptr, 4, prims.MMALayout.ROW)
                dst[k_pair * 2] = k_vec[0:2]
                dst[k_pair * 2 + 1] = k_vec[2:4]

        k_cur = [None] * self.qk_k_frags
        load_k_fragments(k_cur, 0)

        for d_frag in cutlass.range_constexpr(self.qk_d_frags):
            # The last wraparound load is discarded; identical unrolled
            # iterations keep the register-load/MMA pipeline branch-free.
            d_next = ((d_frag + 1) % self.qk_d_frags) * self.MMA_TILER[2]
            k_next = [None] * self.qk_k_frags
            load_k_fragments(k_next, d_next)
            q_off = d_frag * 4
            for k_frag in cutlass.range_constexpr(self.qk_k_frags):
                s_off = k_frag * 4
                s_regs[s_off:4] = ptx_mma_m16n8k32_f32(
                    q_regs[q_off + 0],
                    q_regs[q_off + 1],
                    q_regs[q_off + 2],
                    q_regs[q_off + 3],
                    k_cur[k_frag][0],
                    k_cur[k_frag][1],
                    s_regs[s_off + 0],
                    s_regs[s_off + 1],
                    s_regs[s_off + 2],
                    s_regs[s_off + 3],
                    self.in_dtype,
                )
            k_cur = k_next

        return s_regs

    @cute.jit
    def mma_qk_single_buffer(
        self,
        basic_params: SimpleNamespace,
        mma_params: SimpleNamespace,
        q_regs: cutlass.Array,
    ) -> cutlass.Array:
        """Original fixed-K QK path for the two-buffer specialization."""
        s_regs = cutlass.Array(cutlass.Float32, self.qk_k_frags * 4, alignment=16)
        for i in cutlass.range_constexpr(self.qk_k_frags * 4):
            s_regs[i] = cutlass.Float32(0.0)

        k_row_in_frag_pair = (basic_params.lane_div8 // 2) * 8 + basic_params.lane_mod8
        k_col_in_frag_pair = (basic_params.lane_div8 % 2) * 16

        def load_k_fragment_pair(
            d_offset: cutlass.Constexpr[int],
            k_pair: cutlass.Constexpr[int],
        ):
            k_row = k_pair * 16 + k_row_in_frag_pair
            k_col = d_offset + k_col_in_frag_pair
            k_chunk = k_col // self.tma_copy_head_per_iter
            k_col_in_chunk = k_col % self.tma_copy_head_per_iter
            sK_ptr = (
                mma_params.sK.data_ptr()
                + k_chunk * self.tma_copy_elems_per_iter
                + k_row * self.tma_copy_head_per_iter
                + self._get_swizzled_col(k_row, k_col_in_chunk)
            )
            k_vec = prims.ldmatrix(sK_ptr, 4, prims.MMALayout.ROW)
            return k_vec[0:2], k_vec[2:4]

        k_cur = [None] * self.qk_k_frags
        for k_pair in cutlass.range_constexpr(2):
            k_cur[k_pair * 2], k_cur[k_pair * 2 + 1] = load_k_fragment_pair(
                0,
                k_pair,
            )

        for d_frag in cutlass.range_constexpr(self.qk_d_frags):
            d_next = ((d_frag + 1) % self.qk_d_frags) * self.MMA_TILER[2]
            k_next = [None] * self.qk_k_frags
            q_off = d_frag * 4
            for k_block in cutlass.range_constexpr(self.qk_k_frags // 4):
                for k_in_block in cutlass.range_constexpr(4):
                    k_frag = k_block * 4 + k_in_block
                    s_off = k_frag * 4
                    s_regs[s_off:4] = ptx_mma_m16n8k32_f32(
                        q_regs[q_off + 0],
                        q_regs[q_off + 1],
                        q_regs[q_off + 2],
                        q_regs[q_off + 3],
                        k_cur[k_frag][0],
                        k_cur[k_frag][1],
                        s_regs[s_off + 0],
                        s_regs[s_off + 1],
                        s_regs[s_off + 2],
                        s_regs[s_off + 3],
                        self.in_dtype,
                    )
                if cutlass.const_expr(
                    d_frag == 0 and k_block + 1 < self.qk_k_frags // 4
                ):
                    for pair_in_block in cutlass.range_constexpr(2):
                        k_pair = (k_block + 1) * 2 + pair_in_block
                        k_cur[k_pair * 2], k_cur[k_pair * 2 + 1] = load_k_fragment_pair(
                            0,
                            k_pair,
                        )
                for pair_in_block in cutlass.range_constexpr(2):
                    k_pair = k_block * 2 + pair_in_block
                    k_next[k_pair * 2], k_next[k_pair * 2 + 1] = load_k_fragment_pair(
                        d_next,
                        k_pair,
                    )
            k_cur = k_next

        return s_regs

    @cute.jit
    def online_softmax(
        self,
        basic_params: SimpleNamespace,
        mma_params: SimpleNamespace,
        softmax_params: SimpleNamespace,
        s_regs: cutlass.Array,
        row_state: cutlass.Float32,
        kv_seq_idx: cutlass.Int32,
        is_masked_frontier_tile: cutlass.Constexpr[bool],
        uses_softmax_init_path: cutlass.Constexpr[bool],
    ) -> tuple:
        """Online softmax and stage packed P in registers for the PV MMA.

        :param basic_params: Per-CTA tensor metadata and lane mapping.
        :param mma_params: Local output accumulator state to rescale.
        :param softmax_params: Row max/sum state and log2 softmax scale.
        :param s_regs: Register-resident QK score fragments.
        :param row_state: One persistent scalar per lane. Lanes 0/1 in each
            threadquad hold row maxima; lanes 2/3 hold row sums.
        :param kv_seq_idx: Logical row offset within the current K/V sequence.
        :param is_masked_frontier_tile: Marks the causal/K-tail frontier. It
            enables causal limits and explicit K/V-length bounds.
        :param uses_softmax_init_path: Select the cheaper update that bypasses
            merging prior state. The generic path is also correct for a first
            full tile because row state and O start at their neutral values.
        :return: Register-resident P fragments and updated distributed state.
        """
        o_regs = mma_params.o_regs
        softmax_scale_log2 = softmax_params.softmax_scale_log2
        uses_distributed_row_state = cutlass.const_expr(
            self.head_tile == 256 and self.q_tile == 128 and self.kv_tile == 128
        )
        if cutlass.const_expr(not uses_distributed_row_state):
            row_max = softmax_params.row_max
            row_sum = softmax_params.row_sum

        # Noncausal lane ownership permits direct FP8x4 packing. Masked causal
        # rows remain FP8x2 until mma_pv reshuffles the row-local pairs.
        if cutlass.const_expr(self.is_causal):
            p_regs = cutlass.Array(cutlass.Uint16, self.qk_k_frags * 2, alignment=16)
        else:
            p_regs = cutlass.Array(cutlass.Int32, self.qk_k_frags, alignment=16)

        # Each lane owns four S registers split across two Q rows after Q@K^T.
        for row_half in cutlass.range_constexpr(2):
            s_reg_idx_lo = row_half * 2
            s_reg_idx_hi = row_half * 2 + 1

            q_row_in_cta = (
                basic_params.q_warp_row0 + basic_params.lane_div4 + row_half * 8
            )

            # Compute the valid K-column limit for the current Q row.
            valid_cols = basic_params.seqlen_k
            has_valid_cols = True
            if cutlass.const_expr(is_masked_frontier_tile):
                if cutlass.const_expr(self.is_causal):
                    valid_cols = cute.math.min(
                        basic_params.q_seq_idx
                        + q_row_in_cta
                        + basic_params.causal_q_offset
                        + 1,
                        basic_params.seqlen_k,
                    )
                    has_valid_cols = kv_seq_idx < valid_cols

            # Reduce max across this lane's S values for the current Q row.
            cur_max0 = -cutlass.Float32.inf
            cur_max1 = -cutlass.Float32.inf
            for k_frag in cutlass.range_constexpr(self.qk_k_frags):
                s_off = k_frag * 4
                s0 = s_regs[s_off + s_reg_idx_lo]
                s1 = s_regs[s_off + s_reg_idx_hi]
                if cutlass.const_expr(is_masked_frontier_tile):
                    k_col0 = kv_seq_idx + k_frag * 8 + 2 * basic_params.lane_mod4
                    k_col1 = k_col0 + 1
                    if cutlass.const_expr(self.is_causal):
                        valid0 = k_col0 < valid_cols if has_valid_cols else False
                        valid1 = k_col1 < valid_cols if has_valid_cols else False
                    else:
                        valid0 = k_col0 < valid_cols
                        valid1 = k_col1 < valid_cols
                    if not valid0:
                        s0 = -cutlass.Float32.inf
                    if not valid1:
                        s1 = -cutlass.Float32.inf
                s_regs[s_off + s_reg_idx_lo] = s0
                s_regs[s_off + s_reg_idx_hi] = s1
                cur_max0 = cute.arch.fmax(cur_max0, s0)
                cur_max1 = cute.arch.fmax(cur_max1, s1)

            cur_max = cute.arch.fmax(cur_max0, cur_max1)
            row_max_prev = -cutlass.Float32.inf
            if cutlass.const_expr(not uses_softmax_init_path):
                if cutlass.const_expr(uses_distributed_row_state):
                    row_max_prev = prims.shfl_sync(
                        thread_mask=0xFFFFFFFF,
                        val=row_state,
                        offset=row_half,
                        mask_and_clamp=0x1C03,
                        kind=prims.Shfl.IDX,
                    )
            if cutlass.const_expr(uses_distributed_row_state):
                cur_max = nvvm_threadquad_reduction_max(cur_max)
            else:
                cur_max = nvvm_threadquad_reduction_max_full(cur_max)

            # Compute the correction in the max-holder lane and broadcast it
            # once to the other three lanes that own this logical row.
            old_scale = cutlass.Float32(1.0)
            if cutlass.const_expr(uses_softmax_init_path):
                new_max = cur_max
            else:
                if cutlass.const_expr(uses_distributed_row_state):
                    new_max = cute.arch.fmax(row_max_prev, cur_max)
                    old_scale_candidate = cute.math.exp2(
                        (row_max_prev - new_max) * softmax_scale_log2,
                        fastmath=True,
                    )
                    if cutlass.const_expr(is_masked_frontier_tile):
                        if new_max == -cutlass.Float32.inf:
                            old_scale_candidate = cutlass.Float32(1.0)
                    old_scale = old_scale_candidate
                else:
                    row_max_prev = row_max[row_half]
                    new_max = cute.arch.fmax(row_max_prev, cur_max)
                    old_scale = cute.arch.inline_ptx(
                        (
                            "{\n"
                            "  .reg .pred p;\n"
                            "  .reg .f32 delta;\n"
                            "  sub.rn.f32 delta, $1, $2;\n"
                            "  mul.rn.f32 delta, delta, $3;\n"
                            "  setp.gt.f32 p, $2, $1;\n"
                            "  mov.f32 $0, 0f3f800000;\n"
                            "  @p ex2.approx.ftz.f32 $0, delta;\n"
                            "}"
                        ),
                        write_only_types=[cutlass.Float32],
                        read_only_args=[row_max_prev, new_max, softmax_scale_log2],
                    )
            if cutlass.const_expr(not uses_distributed_row_state):
                row_max[row_half] = new_max

            if cutlass.const_expr(not uses_softmax_init_path):
                for d_frag in cutlass.range_constexpr(self.pv_d_frags):
                    o_off = d_frag * 4 + row_half * 2
                    if cutlass.const_expr(uses_distributed_row_state):
                        if old_scale < 1.0:
                            o_regs[o_off + 0], o_regs[o_off + 1] = mul_packed_f32x2(
                                (o_regs[o_off + 0], o_regs[o_off + 1]),
                                (old_scale, old_scale),
                            )
                    else:
                        if new_max > row_max_prev:
                            o_regs[o_off + 0], o_regs[o_off + 1] = mul_packed_f32x2(
                                (o_regs[o_off + 0], o_regs[o_off + 1]),
                                (old_scale, old_scale),
                            )

            # Compute P, accumulate the per-lane partial sum, and stage P.
            exp_max = new_max
            if cutlass.const_expr(is_masked_frontier_tile):
                # A fully masked row has -inf max; use a finite subtraction
                # anchor to avoid -inf - -inf = NaN while preserving zero P.
                if exp_max == -cutlass.Float32.inf:
                    exp_max = cutlass.Float32(0.0)
            neg_exp_max_scaled = -(exp_max * softmax_scale_log2)
            tile_sum0 = cutlass.Float32(0.0)
            tile_sum1 = cutlass.Float32(0.0)
            p_even0 = cutlass.Float32(0.0)
            p_even1 = cutlass.Float32(0.0)
            for k_frag in cutlass.range_constexpr(self.qk_k_frags):
                s_off = k_frag * 4
                s0 = s_regs[s_off + s_reg_idx_lo]
                s1 = s_regs[s_off + s_reg_idx_hi]
                in0, in1 = fma_packed_f32x2(
                    (s0, s1),
                    (softmax_scale_log2, softmax_scale_log2),
                    (neg_exp_max_scaled, neg_exp_max_scaled),
                )
                p0 = cute.math.exp2(in0, fastmath=True)
                p1 = cute.math.exp2(in1, fastmath=True)
                if cutlass.const_expr(uses_distributed_row_state):
                    tile_sum0, tile_sum1 = add_packed_f32x2(
                        (tile_sum0, tile_sum1),
                        (p0, p1),
                    )
                else:
                    tile_sum0 = tile_sum0 + p0
                    tile_sum1 = tile_sum1 + p1
                if cutlass.const_expr(self.is_causal):
                    p_regs[k_frag * 2 + row_half] = cvt_f32x2_to_f8x2(
                        p1, p0, self.in_dtype
                    )
                elif cutlass.const_expr(k_frag % 2 == 0):
                    p_even0 = p0
                    p_even1 = p1
                else:
                    p_regs[(k_frag // 2) * 2 + row_half] = cvt_f32x4_to_f8x4(
                        p_even0, p_even1, p0, p1, self.in_dtype
                    )

            # Reduce tile_sum across the four lanes that own one Q row.
            tile_sum = tile_sum0 + tile_sum1
            if cutlass.const_expr(uses_distributed_row_state):
                if basic_params.lane_mod4 == row_half + 2:
                    tile_sum = tile_sum + row_state * old_scale
            if cutlass.const_expr(uses_distributed_row_state):
                tile_sum = nvvm_threadquad_reduction_sum(tile_sum)
            else:
                tile_sum = nvvm_threadquad_reduction_sum_full(tile_sum)

            # Correct row_sum and rescale O when row_max changes.
            if cutlass.const_expr(uses_distributed_row_state):
                if basic_params.lane_mod4 == row_half:
                    row_state = new_max
                if basic_params.lane_mod4 == row_half + 2:
                    row_state = tile_sum
            elif cutlass.const_expr(uses_softmax_init_path):
                row_sum[row_half] = tile_sum
            else:
                row_sum[row_half] = row_sum[row_half] * old_scale + tile_sum

        return p_regs, row_state

    @cute.jit
    def online_softmax_single_buffer(
        self,
        basic_params: SimpleNamespace,
        mma_params: SimpleNamespace,
        softmax_params: SimpleNamespace,
        s_regs: cutlass.Array,
        kv_seq_idx: cutlass.Int32,
        is_masked_frontier_tile: cutlass.Constexpr[bool],
        uses_softmax_init_path: cutlass.Constexpr[bool],
    ) -> tuple:
        """Original direct row-state softmax for the two-buffer path."""
        o_regs = mma_params.o_regs
        row_max = softmax_params.row_max
        row_sum = softmax_params.row_sum
        softmax_scale_log2 = softmax_params.softmax_scale_log2
        uses_direct_p_pack = cutlass.const_expr(
            self.head_tile == 128 and self.q_tile == 128 and self.kv_tile == 128
        )
        if cutlass.const_expr(self.is_causal and not uses_direct_p_pack):
            p_regs = cutlass.Array(cutlass.Uint16, self.qk_k_frags * 2, alignment=16)
        else:
            p_regs = cutlass.Array(cutlass.Int32, self.qk_k_frags, alignment=16)
        v_initial = cutlass.Array(cutlass.Int32, self.pv_d_frags * 2, alignment=16)

        for row_half in cutlass.range_constexpr(2):
            s_reg_idx_lo = row_half * 2
            s_reg_idx_hi = row_half * 2 + 1
            q_row_in_cta = (
                basic_params.q_warp_row0 + basic_params.lane_div4 + row_half * 8
            )

            valid_cols = basic_params.seqlen_k
            has_valid_cols = True
            if cutlass.const_expr(is_masked_frontier_tile):
                if cutlass.const_expr(self.is_causal):
                    valid_cols = cute.math.min(
                        basic_params.q_seq_idx
                        + q_row_in_cta
                        + basic_params.causal_q_offset
                        + 1,
                        basic_params.seqlen_k,
                    )
                    has_valid_cols = kv_seq_idx < valid_cols

            cur_max0 = -cutlass.Float32.inf
            cur_max1 = -cutlass.Float32.inf
            for k_frag in cutlass.range_constexpr(self.qk_k_frags):
                s_off = k_frag * 4
                s0 = s_regs[s_off + s_reg_idx_lo]
                s1 = s_regs[s_off + s_reg_idx_hi]
                if cutlass.const_expr(is_masked_frontier_tile):
                    k_col0 = kv_seq_idx + k_frag * 8 + 2 * basic_params.lane_mod4
                    k_col1 = k_col0 + 1
                    if cutlass.const_expr(self.is_causal):
                        valid0 = k_col0 < valid_cols if has_valid_cols else False
                        valid1 = k_col1 < valid_cols if has_valid_cols else False
                    else:
                        valid0 = k_col0 < valid_cols
                        valid1 = k_col1 < valid_cols
                    if not valid0:
                        s0 = -cutlass.Float32.inf
                    if not valid1:
                        s1 = -cutlass.Float32.inf
                s_regs[s_off + s_reg_idx_lo] = s0
                s_regs[s_off + s_reg_idx_hi] = s1
                cur_max0 = cute.arch.fmax(cur_max0, s0)
                cur_max1 = cute.arch.fmax(cur_max1, s1)

            cur_max = cute.arch.fmax(cur_max0, cur_max1)
            cur_max = nvvm_threadquad_reduction_max(cur_max)

            old_scale = cutlass.Float32(1.0)
            if cutlass.const_expr(uses_softmax_init_path):
                new_max = cur_max
            else:
                row_max_prev = row_max[row_half]
                new_max = cute.arch.fmax(row_max_prev, cur_max)
                old_scale = cute.arch.inline_ptx(
                    (
                        "{\n"
                        "  .reg .pred p;\n"
                        "  .reg .f32 delta;\n"
                        "  sub.rn.f32 delta, $1, $2;\n"
                        "  mul.rn.f32 delta, delta, $3;\n"
                        "  setp.gt.f32 p, $2, $1;\n"
                        "  mov.f32 $0, 0f3f800000;\n"
                        "  @p ex2.approx.ftz.f32 $0, delta;\n"
                        "}"
                    ),
                    write_only_types=[cutlass.Float32],
                    read_only_args=[row_max_prev, new_max, softmax_scale_log2],
                )
            row_max[row_half] = new_max

            if cutlass.const_expr(not uses_softmax_init_path):
                for d_frag in cutlass.range_constexpr(self.pv_d_frags):
                    o_off = d_frag * 4 + row_half * 2
                    if new_max > row_max_prev:
                        o_regs[o_off + 0], o_regs[o_off + 1] = mul_packed_f32x2(
                            (o_regs[o_off + 0], o_regs[o_off + 1]),
                            (old_scale, old_scale),
                        )

            exp_max = new_max
            if cutlass.const_expr(is_masked_frontier_tile):
                if exp_max == -cutlass.Float32.inf:
                    exp_max = cutlass.Float32(0.0)
            neg_exp_max_scaled = -(exp_max * softmax_scale_log2)
            tile_sum0 = cutlass.Float32(0.0)
            tile_sum1 = cutlass.Float32(0.0)
            p_even0 = cutlass.Float32(0.0)
            p_even1 = cutlass.Float32(0.0)
            for k_frag in cutlass.range_constexpr(self.qk_k_frags):
                s_off = k_frag * 4
                s0 = s_regs[s_off + s_reg_idx_lo]
                s1 = s_regs[s_off + s_reg_idx_hi]
                in0, in1 = fma_packed_f32x2(
                    (s0, s1),
                    (softmax_scale_log2, softmax_scale_log2),
                    (neg_exp_max_scaled, neg_exp_max_scaled),
                )
                p0 = cute.math.exp2(in0, fastmath=True)
                p1 = cute.math.exp2(in1, fastmath=True)
                tile_sum0 = tile_sum0 + p0
                tile_sum1 = tile_sum1 + p1
                if cutlass.const_expr(self.is_causal and not uses_direct_p_pack):
                    p_regs[k_frag * 2 + row_half] = cvt_f32x2_to_f8x2(
                        p1, p0, self.in_dtype
                    )
                elif cutlass.const_expr(k_frag % 2 == 0):
                    p_even0 = p0
                    p_even1 = p1
                else:
                    p_regs[(k_frag // 2) * 2 + row_half] = cvt_f32x4_to_f8x4(
                        p_even0, p_even1, p0, p1, self.in_dtype
                    )

            tile_sum = tile_sum0 + tile_sum1
            tile_sum = nvvm_threadquad_reduction_sum(tile_sum)
            if cutlass.const_expr(uses_softmax_init_path):
                row_sum[row_half] = tile_sum
            else:
                row_sum[row_half] = row_sum[row_half] * old_scale + tile_sum

            if cutlass.const_expr(row_half == 0):
                # Scores for this row half are dead. Load the first V fragment
                # set into that register window while row-half 1 still has
                # scalar softmax work available to hide the shared-memory load.
                v_initial = self.load_v_initial_fragments_single_buffer(
                    basic_params,
                    mma_params,
                )

        return p_regs, v_initial

    @cute.jit
    def mma_pv(
        self,
        basic_params: SimpleNamespace,
        mma_params: SimpleNamespace,
        p_regs: cutlass.Array,
        sV: cutlass.Pointer,
    ) -> None:
        """Compute ``O += P @ V``.

        P fragments are already packed in registers. V fragments are streamed
        from the TMA-populated ``sV`` tile with ``ldmatrix``.

        :param basic_params: Per-CTA tensor metadata and lane mapping.
        :param mma_params: Shared K/V ring and local output accumulator state.
        :param p_regs: Register-resident FP8 P fragments from softmax.
        :param sV: Shared-memory pointer to this tile's V operand.
        """
        o_regs = mma_params.o_regs

        def load_v_fragment_pair(
            v_seq_offset: cutlass.Constexpr[int],
            d_pair: cutlass.Constexpr[int],
        ):
            v_lane = basic_params.lane
            if cutlass.const_expr(not self.is_causal):
                v_k16_lane = v_lane % 16
                v_lane = (
                    (v_lane // 16) * 16
                    + 2 * (v_k16_lane // 4)
                    + (v_k16_lane % 2)
                    + 8 * ((v_k16_lane % 4) // 2)
                )
            v_row = v_seq_offset + v_lane
            v_col = d_pair * 16
            v_chunk = v_col // self.tma_copy_head_per_iter
            v_col_in_chunk = v_col % self.tma_copy_head_per_iter
            sV_ptr = (
                sV
                + v_chunk * self.tma_copy_elems_per_iter
                + v_row * self.tma_copy_head_per_iter
                + self._get_swizzled_col(v_row, v_col_in_chunk)
            )
            v_vec = prims.ldmatrix(
                sV_ptr,
                4,
                prims.MMALayout.COL,
                shape="m16n16",
                src_format="b8",
            )
            return (v_vec[0], v_vec[2]), (v_vec[1], v_vec[3])

        def pack_p_cols(
            k_frag0: cutlass.Constexpr[int],
            row_half: cutlass.Constexpr[int],
        ) -> cutlass.Int32:
            if cutlass.const_expr(not self.is_causal):
                return p_regs[(k_frag0 // 2) * 2 + row_half]

            packed_pairs = pack_f8x2_pairs(
                p_regs[k_frag0 * 2 + row_half],
                p_regs[(k_frag0 + 1) * 2 + row_half],
            )
            pair_lane = (basic_params.lane_mod4 % 2) * 2
            if cutlass.const_expr(
                self.head_tile == 256 and self.q_tile == 128 and self.kv_tile == 128
            ):
                src0 = pair_lane
                src1 = src0 + 1
                lo = prims.shfl_sync(
                    thread_mask=0xFFFFFFFF,
                    val=packed_pairs,
                    offset=src0,
                    mask_and_clamp=0x1C03,
                    kind=prims.Shfl.IDX,
                )
                hi = prims.shfl_sync(
                    thread_mask=0xFFFFFFFF,
                    val=packed_pairs,
                    offset=src1,
                    mask_and_clamp=0x1C03,
                    kind=prims.Shfl.IDX,
                )
                selector = cutlass.Int32(0x5410) + (
                    basic_params.lane_mod4 // 2
                ) * cutlass.Int32(0x2222)
            else:
                src0 = basic_params.lane_div4 * 4 + pair_lane
                src1 = src0 + 1
                lo = prims.shfl_sync(
                    thread_mask=0xFFFFFFFF,
                    val=packed_pairs,
                    offset=src0,
                    mask_and_clamp=0x1F,
                    kind=prims.Shfl.IDX,
                )
                hi = prims.shfl_sync(
                    thread_mask=0xFFFFFFFF,
                    val=packed_pairs,
                    offset=src1,
                    mask_and_clamp=0x1F,
                    kind=prims.Shfl.IDX,
                )
                selector = (
                    cutlass.Int32(0x5410)
                    if basic_params.lane_mod4 < 2
                    else cutlass.Int32(0x7632)
                )
            return cute.arch.inline_ptx(
                "prmt.b32 $0, $1, $2, $3;",
                write_only_types=[cutlass.Int32],
                read_only_args=[lo, hi, selector],
            )

        v_cur = [None] * self.pv_d_frags
        for d_pair in cutlass.range_constexpr(self.pv_d_frags // 2):
            v_cur[d_pair * 2], v_cur[d_pair * 2 + 1] = load_v_fragment_pair(0, d_pair)

        for v_frag in cutlass.range_constexpr(self.pv_v_frags):
            prefetch_next = cutlass.const_expr(
                not (
                    self.head_tile == 256 and self.q_tile == 128 and self.kv_tile == 128
                )
                or v_frag + 1 < self.pv_v_frags
            )
            if cutlass.const_expr(prefetch_next):
                v_seq_off_next = ((v_frag + 1) % self.pv_v_frags) * self.MMA_TILER[2]
                v_next = [None] * self.pv_d_frags

            p_reg0 = pack_p_cols(v_frag * 4 + 0, 0)
            p_reg1 = pack_p_cols(v_frag * 4 + 0, 1)
            p_reg2 = pack_p_cols(v_frag * 4 + 2, 0)
            p_reg3 = pack_p_cols(v_frag * 4 + 2, 1)
            # Interleave each next-fragment x2 prefetch with the two MMAs that
            # consume the matching current D pair. This exposes independent
            # load and MMA work while preserving the x2 carrier pairing.
            for d_pair in cutlass.range_constexpr(self.pv_d_frags // 2):
                if cutlass.const_expr(prefetch_next):
                    v_next[d_pair * 2], v_next[d_pair * 2 + 1] = load_v_fragment_pair(
                        v_seq_off_next, d_pair
                    )
                for d_in_pair in cutlass.range_constexpr(2):
                    d_frag = d_pair * 2 + d_in_pair
                    o_off = d_frag * 4
                    o_regs[o_off:4] = ptx_mma_m16n8k32_f32(
                        p_reg0,
                        p_reg1,
                        p_reg2,
                        p_reg3,
                        v_cur[d_frag][0],
                        v_cur[d_frag][1],
                        o_regs[o_off + 0],
                        o_regs[o_off + 1],
                        o_regs[o_off + 2],
                        o_regs[o_off + 3],
                        self.in_dtype,
                    )
            if cutlass.const_expr(prefetch_next):
                v_cur = v_next

    @cute.jit
    def load_v_initial_fragments_single_buffer(
        self,
        basic_params: SimpleNamespace,
        mma_params: SimpleNamespace,
    ) -> cutlass.Array:
        """Load the first BMM2 V fragment set before online softmax."""
        v_regs = cutlass.Array(cutlass.Int32, self.pv_d_frags * 2, alignment=16)
        v_lane = basic_params.lane
        uses_direct_p_pack = cutlass.const_expr(
            self.head_tile == 128 and self.q_tile == 128 and self.kv_tile == 128
        )
        if cutlass.const_expr(self.is_causal and uses_direct_p_pack):
            # Swap K-row bits 1 and 2.  This is the lane permutation paired
            # with the one-shuffle P transpose below; unlike the fully direct
            # bit rotation, its low three bits remain a bijection for each
            # LDSM address group, avoiding a second shared-memory wavefront.
            v_k16_lane = v_lane % 16
            v_lane = (
                (v_lane // 16) * 16
                + (v_k16_lane & 0x9)
                + ((v_k16_lane & 0x2) << 1)
                + ((v_k16_lane & 0x4) >> 1)
            )
        elif cutlass.const_expr(not self.is_causal):
            v_k16_lane = v_lane % 16
            v_lane = (
                (v_lane // 16) * 16
                + 2 * (v_k16_lane // 4)
                + (v_k16_lane % 2)
                + 8 * ((v_k16_lane % 4) // 2)
            )
        v_row = v_lane
        for d_pair in cutlass.range_constexpr(self.pv_d_frags // 2):
            v_col = d_pair * 16
            v_chunk = v_col // self.tma_copy_head_per_iter
            v_col_in_chunk = v_col % self.tma_copy_head_per_iter
            sV_ptr = (
                mma_params.sV.data_ptr()
                + v_chunk * self.tma_copy_elems_per_iter
                + v_row * self.tma_copy_head_per_iter
                + self._get_swizzled_col(v_row, v_col_in_chunk)
            )
            v_vec = prims.ldmatrix(
                sV_ptr,
                4,
                prims.MMALayout.COL,
                shape="m16n16",
                src_format="b8",
            )
            d_frag = d_pair * 2
            v_regs[(d_frag + 0) * 2 + 0] = v_vec[0]
            v_regs[(d_frag + 0) * 2 + 1] = v_vec[2]
            v_regs[(d_frag + 1) * 2 + 0] = v_vec[1]
            v_regs[(d_frag + 1) * 2 + 1] = v_vec[3]
        return v_regs

    @cute.jit
    def mma_pv_single_buffer_preloaded(
        self,
        basic_params: SimpleNamespace,
        mma_params: SimpleNamespace,
        p_regs: cutlass.Array,
        v_initial: cutlass.Array,
    ) -> None:
        """BMM2 with its first V fragment set already resident in registers."""
        o_regs = mma_params.o_regs

        def load_v_fragment_pair(
            v_seq_offset: cutlass.Constexpr[int],
            d_pair: cutlass.Constexpr[int],
        ):
            v_lane = basic_params.lane
            uses_direct_p_pack = cutlass.const_expr(
                self.head_tile == 128 and self.q_tile == 128 and self.kv_tile == 128
            )
            if cutlass.const_expr(self.is_causal and uses_direct_p_pack):
                v_k16_lane = v_lane % 16
                v_lane = (
                    (v_lane // 16) * 16
                    + (v_k16_lane & 0x9)
                    + ((v_k16_lane & 0x2) << 1)
                    + ((v_k16_lane & 0x4) >> 1)
                )
            elif cutlass.const_expr(not self.is_causal):
                v_k16_lane = v_lane % 16
                v_lane = (
                    (v_lane // 16) * 16
                    + 2 * (v_k16_lane // 4)
                    + (v_k16_lane % 2)
                    + 8 * ((v_k16_lane % 4) // 2)
                )
            v_row = v_seq_offset + v_lane
            v_col = d_pair * 16
            v_chunk = v_col // self.tma_copy_head_per_iter
            v_col_in_chunk = v_col % self.tma_copy_head_per_iter
            sV_ptr = (
                mma_params.sV.data_ptr()
                + v_chunk * self.tma_copy_elems_per_iter
                + v_row * self.tma_copy_head_per_iter
                + self._get_swizzled_col(v_row, v_col_in_chunk)
            )
            v_vec = prims.ldmatrix(
                sV_ptr,
                4,
                prims.MMALayout.COL,
                shape="m16n16",
                src_format="b8",
            )
            return (v_vec[0], v_vec[2]), (v_vec[1], v_vec[3])

        def pack_p_cols(
            k_frag0: cutlass.Constexpr[int],
            row_half: cutlass.Constexpr[int],
        ) -> cutlass.Int32:
            uses_direct_p_pack = cutlass.const_expr(
                self.head_tile == 128 and self.q_tile == 128 and self.kv_tile == 128
            )
            if cutlass.const_expr(not self.is_causal):
                return p_regs[(k_frag0 // 2) * 2 + row_half]
            if cutlass.const_expr(uses_direct_p_pack):
                packed = p_regs[(k_frag0 // 2) * 2 + row_half]
                partner = prims.shfl_sync(
                    thread_mask=0xFFFFFFFF,
                    val=packed,
                    offset=2,
                    mask_and_clamp=0x1C03,
                    kind=prims.Shfl.BFLY,
                )
                selector = (
                    cutlass.Int32(0x5410)
                    if basic_params.lane_mod4 < 2
                    else cutlass.Int32(0x3276)
                )
                return cute.arch.inline_ptx(
                    "prmt.b32 $0, $1, $2, $3;",
                    write_only_types=[cutlass.Int32],
                    read_only_args=[packed, partner, selector],
                )

            packed_pairs = pack_f8x2_pairs(
                p_regs[k_frag0 * 2 + row_half],
                p_regs[(k_frag0 + 1) * 2 + row_half],
            )
            pair_lane = (basic_params.lane_mod4 % 2) * 2
            src0 = basic_params.lane_div4 * 4 + pair_lane
            src1 = src0 + 1
            lo = prims.shfl_sync(
                thread_mask=0xFFFFFFFF,
                val=packed_pairs,
                offset=src0,
                mask_and_clamp=0x1F,
                kind=prims.Shfl.IDX,
            )
            hi = prims.shfl_sync(
                thread_mask=0xFFFFFFFF,
                val=packed_pairs,
                offset=src1,
                mask_and_clamp=0x1F,
                kind=prims.Shfl.IDX,
            )
            selector = (
                cutlass.Int32(0x5410)
                if basic_params.lane_mod4 < 2
                else cutlass.Int32(0x7632)
            )
            return cute.arch.inline_ptx(
                "prmt.b32 $0, $1, $2, $3;",
                write_only_types=[cutlass.Int32],
                read_only_args=[lo, hi, selector],
            )

        v_cur = [None] * self.pv_d_frags
        for d_frag in cutlass.range_constexpr(self.pv_d_frags):
            v_cur[d_frag] = (
                v_initial[d_frag * 2 + 0],
                v_initial[d_frag * 2 + 1],
            )
        for v_frag in cutlass.range_constexpr(self.pv_v_frags):
            v_seq_off_next = ((v_frag + 1) % self.pv_v_frags) * self.MMA_TILER[2]
            v_next = [None] * self.pv_d_frags

            p_reg0 = pack_p_cols(v_frag * 4 + 0, 0)
            p_reg1 = pack_p_cols(v_frag * 4 + 0, 1)
            p_reg2 = pack_p_cols(v_frag * 4 + 2, 0)
            p_reg3 = pack_p_cols(v_frag * 4 + 2, 1)
            for d_block in cutlass.range_constexpr(self.pv_d_frags // 2):
                for d_in_block in cutlass.range_constexpr(2):
                    d_frag = d_block * 2 + d_in_block
                    o_off = d_frag * 4
                    o_regs[o_off:4] = ptx_mma_m16n8k32_f32(
                        p_reg0,
                        p_reg1,
                        p_reg2,
                        p_reg3,
                        v_cur[d_frag][0],
                        v_cur[d_frag][1],
                        o_regs[o_off + 0],
                        o_regs[o_off + 1],
                        o_regs[o_off + 2],
                        o_regs[o_off + 3],
                        self.in_dtype,
                    )
                for pair_in_block in cutlass.range_constexpr(1):
                    d_pair = d_block + pair_in_block
                    v_next[d_pair * 2], v_next[d_pair * 2 + 1] = load_v_fragment_pair(
                        v_seq_off_next, d_pair
                    )
            v_cur = v_next

    @cute.jit
    def wait_pipeline_barrier(
        self,
        mbar: cutlass.Pointer,
        phase: cutlass.Int32,
    ) -> None:
        """Block on one dynamically selected circular-pipeline barrier."""
        while not prims.mbarrier_try_wait_parity(mbar, phase, time_limit=10_000_000):
            pass

    @cute.jit
    def compute_one_kv_tile(
        self,
        basic_params: SimpleNamespace,
        mma_params: SimpleNamespace,
        softmax_params: SimpleNamespace,
        q_regs: cutlass.Array,
        row_state: cutlass.Float32,
        kv_tile_idx: cutlass.Int32,
        k_stage: cutlass.Int32,
        is_masked_frontier_tile: cutlass.Constexpr[bool],
        uses_softmax_init_path: cutlass.Constexpr[bool],
    ) -> cutlass.Float32:
        """One compute-side iteration of the right-to-left FMHA prefill loop.

        K and V are consecutive operands in one three-slot circular pipeline:
        ``K0, V0, K1, V1, ...``. Online softmax sits between consuming K and V
        because it has no V dependency, so the producer can fill later slots
        while scalar work runs.

        :param basic_params: Per-CTA tensor metadata, lane mapping, and K/V
            pipeline mbarriers.
        :param mma_params: Shared-memory tiles and local MMA state.
        :param softmax_params: Online softmax row state.
        :param q_regs: Register-resident packed Q fragments.
        :param row_state: One distributed max/sum scalar per lane.
        :param kv_tile_idx: K/V tile index processed by this iteration.
        :param k_stage: K slot in the repeating ``0, 2, 1`` ring schedule.
        :param is_masked_frontier_tile: Marks the causal/K-tail frontier for
            causal limits and explicit K/V-length bounds.
        :param uses_softmax_init_path: Whether to bypass the neutral initial
            softmax/O state instead of using the generic merge path.
        """

        # Six operands span one three-tile period. Carry the K stage directly;
        # its low bit is the K phase. V is the next slot, and its phase is a
        # divide-by-two shift. This avoids selecting four values from a state
        # table on every inner-loop iteration.
        k_phase = k_stage & cutlass.Int32(1)
        v_stage = k_stage + cutlass.Int32(1)
        if v_stage == self.kv_pipeline_stages:
            v_stage = cutlass.Int32(0)
        v_phase = (k_stage + cutlass.Int32(1)) // cutlass.Int32(2)

        k_arrived = self.get_mbar_stage_ptr(basic_params.mbar_arrived, k_stage)
        self.wait_pipeline_barrier(k_arrived, k_phase)

        sK = self.get_kv_stage_ptr(mma_params.sKV, k_stage)
        s_regs = self.mma_qk(basic_params, mma_params, q_regs, sK)
        if prims.elect_sync():
            k_consumed = self.get_mbar_stage_ptr(basic_params.mbar_consumed, k_stage)
            prims.mbarrier_arrive(
                k_consumed,
                count=cute.arch.WARP_SIZE,
            )

        kv_seq_idx = kv_tile_idx * self.kv_tile
        p_regs, row_state = self.online_softmax(
            basic_params,
            mma_params,
            softmax_params,
            s_regs,
            row_state,
            kv_seq_idx,
            is_masked_frontier_tile,
            uses_softmax_init_path,
        )
        # Softmax has no V dependency. Delay the arrival wait until the first
        # V consumer so its scalar work can hide the outstanding TMA transfer.
        v_arrived = self.get_mbar_stage_ptr(basic_params.mbar_arrived, v_stage)
        self.wait_pipeline_barrier(v_arrived, v_phase)

        sV = self.get_kv_stage_ptr(mma_params.sKV, v_stage)
        self.mma_pv(basic_params, mma_params, p_regs, sV)
        if prims.elect_sync():
            v_consumed = self.get_mbar_stage_ptr(basic_params.mbar_consumed, v_stage)
            prims.mbarrier_arrive(
                v_consumed,
                count=cute.arch.WARP_SIZE,
            )
        return row_state

    @cute.jit
    def compute_one_kv_tile_single_buffer(
        self,
        basic_params: SimpleNamespace,
        mma_params: SimpleNamespace,
        softmax_params: SimpleNamespace,
        q_regs: cutlass.Array,
        num_kv_tiles: cutlass.Int32,
        kv_tile_idx: cutlass.Int32,
        is_masked_frontier_tile: cutlass.Constexpr[bool],
        uses_softmax_init_path: cutlass.Constexpr[bool],
    ) -> None:
        """Original independent fixed-buffer K/V compute iteration."""
        tma_phase = (num_kv_tiles - 1 - kv_tile_idx) & cutlass.Int32(1)
        while not prims.mbarrier_try_wait_parity(
            basic_params.mbar_k_arrived, tma_phase, time_limit=10_000_000
        ):
            pass

        s_regs = self.mma_qk_single_buffer(basic_params, mma_params, q_regs)
        if prims.elect_sync():
            prims.mbarrier_arrive(
                basic_params.mbar_k_consumed,
                count=cute.arch.WARP_SIZE,
            )

        while not prims.mbarrier_try_wait_parity(
            basic_params.mbar_v_arrived, tma_phase, time_limit=10_000_000
        ):
            pass

        kv_seq_idx = kv_tile_idx * self.kv_tile
        p_regs, v_initial = self.online_softmax_single_buffer(
            basic_params,
            mma_params,
            softmax_params,
            s_regs,
            kv_seq_idx,
            is_masked_frontier_tile,
            uses_softmax_init_path,
        )

        self.mma_pv_single_buffer_preloaded(
            basic_params,
            mma_params,
            p_regs,
            v_initial,
        )
        if prims.elect_sync():
            prims.mbarrier_arrive(
                basic_params.mbar_v_consumed,
                count=cute.arch.WARP_SIZE,
            )

    @cute.kernel
    def kernel(
        self,
        q: cute.Tensor,
        k: cute.Tensor,
        v: cute.Tensor,
        o: cute.Tensor,
        lse: cute.Tensor | None,
        tma_k_desc: cutlass.GridConstant[cuda.TensorMap],
        tma_v_desc: cutlass.GridConstant[cuda.TensorMap],
        softmax_scale_log2: cutlass.Float32,
        output_scale: cutlass.Float32,
        seqlens_kv: cute.Tensor | None = None,
        cu_seqlens_q: cute.Tensor | None = None,
        block_tables: cute.Tensor | None = None,
        cu_seqlens_k: cute.Tensor | None = None,
    ) -> None:
        """SM120 FMHA prefill FP8 kernel.

        :param q: Query tensor.
        :param k: Key tensor.
        :param v: Value tensor.
        :param o: Output tensor.
        :param lse: Optional packed log2 LSE tensor ``[total_q,Hq]``.
        :param tma_k_desc: Tensor map descriptor for K.
        :param tma_v_desc: Tensor map descriptor for V.
        :param softmax_scale_log2: ``softmax_scale * log2(e)``, pre-folded host-side.
        :param output_scale: Scalar V dequantization scale folded into the
            final normalization.
        :param seqlens_kv: Optional runtime K/V length for each request.
        :param cu_seqlens_q: Optional cumulative offsets for packed Q/O.
        :param block_tables: Shared K/V page indices with shape ``[B,max_pages]``.
        :param cu_seqlens_k: Optional cumulative offsets for packed K/V.
        """
        tidx, _, _ = cute.arch.thread_idx()
        block_x, block_y, block_z = cute.arch.block_idx()
        if cutlass.const_expr(self.balanced_scheduler):
            q_head_idx = block_x
            q_tile_idx = block_y
            # Causal Q tiles have monotonically increasing K/V work. Schedule
            # the longest rows first so the final waves do not consist only of
            # the most expensive CTAs. This preserves the logical tile mapping
            # while improving load balance for every causal sequence length.
            _, grid_y, _ = cute.arch.grid_dim()
            q_tile_idx = grid_y - cutlass.Int32(1) - block_y
        else:
            q_tile_idx = block_x
            q_head_idx = block_y
        batch_idx = block_z
        q_seq_idx = q_tile_idx * self.q_tile

        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane = tidx % cute.arch.WARP_SIZE
        lane_div4 = lane // 4
        lane_mod4 = lane % 4
        lane_div8 = lane // 8
        lane_mod8 = lane % 8
        lane_div16 = lane // 16

        q_token_base = cutlass.Int32(0)
        k_token_base = cutlass.Int32(0)
        causal_q_offset = cutlass.Int32(0)

        cute.arch.griddepcontrol_wait()
        q_token_base = cutlass.Int32(cu_seqlens_q[batch_idx])
        seqlen_q = cutlass.Int32(cu_seqlens_q[batch_idx + 1]) - q_token_base
        num_heads_q = q.shape[1]
        head_dim = q.shape[2]

        if cutlass.const_expr(self.use_paged_kv):
            seqlen_k = cutlass.Int32(seqlens_kv[batch_idx])
            num_heads_kv = k.shape[1]
        else:
            k_token_base = cutlass.Int32(cu_seqlens_k[batch_idx])
            seqlen_k = cutlass.Int32(cu_seqlens_k[batch_idx + 1]) - k_token_base
            num_heads_kv = k.shape[1]

        if cutlass.const_expr(self.has_causal_q_offset):
            causal_q_offset = seqlen_k - seqlen_q
        q_ptr = q.iterator.raw_ptr()
        o_ptr = o.iterator.raw_ptr()

        kv_head_idx = q_head_idx
        if cutlass.const_expr(num_heads_q != num_heads_kv):
            # Contiguous equal-sized Q-head groups share one K/V head, matching
            # repeat_interleave semantics without materializing repeated K/V.
            q_heads_per_kv_head = num_heads_q // num_heads_kv
            kv_head_idx = q_head_idx // q_heads_per_kv_head

        q_seq_stride = num_heads_q * head_dim
        q_head_base = q_token_base * q_seq_stride + q_head_idx * head_dim
        o_head_base = q_head_base

        num_kv_tiles = ceil_div(seqlen_k, self.kv_tile)
        if cutlass.const_expr(self.is_causal):
            num_kv_tiles_causal = ceil_div(
                q_seq_idx + causal_q_offset + self.q_tile, self.kv_tile
            )
            num_kv_tiles = cute.math.min(num_kv_tiles, num_kv_tiles_causal)

        # The epilogue later aliases this storage as sO after compute warps
        # finish consuming the final K/V tile.
        sKV = cutlass.Array(
            k.dtype,
            self.smem_elems,
            space=cutlass.AddressSpace.smem,
            alignment=128,
        )
        s_mbar = cutlass.Array(
            cutlass.Int64,
            self.kv_pipeline_stages * 2,
            space=cutlass.AddressSpace.smem,
            alignment=8,
        )
        mbar_arrived = s_mbar
        mbar_consumed = s_mbar.subview(self.kv_pipeline_stages)
        tma_k_desc_ptr = tma_k_desc.get_ptr()
        tma_v_desc_ptr = tma_v_desc.get_ptr()

        # Initialize the K/V pipeline only for CTAs that have both a Q row and
        # at least one K/V token. The condition is CTA-uniform, so zero-length
        # requests skip every TMA/mbarrier/MMA operation without an early
        # return (early kernel returns are unsupported by CuTe DSL).
        if q_seq_idx < seqlen_q and seqlen_k > 0:
            if warp == self.load_warp_id:
                if prims.elect_sync():
                    prims.prefetch_tensormap(tma_k_desc_ptr)
                    prims.prefetch_tensormap(tma_v_desc_ptr)
                if prims.elect_sync():
                    prims.mbarrier_init(mbar_arrived.subview(0), 1)
                    prims.mbarrier_init(mbar_arrived.subview(1), 1)
                    if cutlass.const_expr(self.kv_pipeline_stages == 3):
                        prims.mbarrier_init(mbar_arrived.subview(2), 1)
                    prims.mbarrier_init(mbar_consumed.subview(0), self.threads_compute)
                    prims.mbarrier_init(mbar_consumed.subview(1), self.threads_compute)
                    if cutlass.const_expr(self.kv_pipeline_stages == 3):
                        prims.mbarrier_init(
                            mbar_consumed.subview(2), self.threads_compute
                        )
            prims.fence_mbarrier_init()
            prims.barrier_cta_sync(0)

        # /////////////////////////////////////////////////////////////////////////////
        #  LOAD K/V
        # /////////////////////////////////////////////////////////////////////////////
        if warp == self.load_warp_id and q_seq_idx < seqlen_q and seqlen_k > 0:
            prims.setmaxregister(24, prims.SetMaxRegisterAction.DECREASE)

            uses_three_stage_pipeline = cutlass.const_expr(
                self.head_tile == 256 and self.q_tile == 128 and self.kv_tile == 128
            )
            uses_kv_l2_policy = cutlass.const_expr(
                self.head_tile in (128, 256)
                and self.q_tile == 128
                and self.kv_tile == 128
            )
            kv_l2_cache_hint = cutlass.Int64(0)
            if cutlass.const_expr(uses_kv_l2_policy):
                kv_l2_cache_hint = cute.arch.inline_ptx(
                    "createpolicy.fractional.L2::evict_last.b64 $0, 1.0;",
                    write_only_types=[cutlass.Int64],
                    read_only_args=[],
                )
            cached_page = cutlass.Int32(0)
            if cutlass.const_expr(uses_three_stage_pipeline):
                # Treat K and V as one ordered operand stream. Three ring slots
                # initially hold K0, V0, and K1. Thereafter the producer reuses
                # a slot as soon as compute signals that operand consumed.
                operand_idx = cutlass.Int32(0)
                stage = cutlass.Int32(0)
                empty_phase = cutlass.Int32(0)
                num_operands = num_kv_tiles * cutlass.Int32(2)
                while operand_idx < num_operands:
                    if operand_idx >= self.kv_pipeline_stages:
                        empty = self.get_mbar_stage_ptr(mbar_consumed, stage)
                        self.wait_pipeline_barrier(empty, empty_phase)

                    tile_iteration = operand_idx // cutlass.Int32(2)
                    kv_seq_idx = (
                        num_kv_tiles - cutlass.Int32(1) - tile_iteration
                    ) * self.kv_tile
                    if cutlass.const_expr(self.use_paged_kv):
                        page_ids = self.resolve_paged_kv_tile_pages(
                            batch_idx,
                            kv_seq_idx,
                            seqlen_k,
                            block_tables,
                            cached_page,
                        )
                    else:
                        kv_tma_seq_idx = k_token_base + kv_seq_idx

                    s_operand = self.get_kv_stage_ptr(sKV, stage)
                    arrived = self.get_mbar_stage_ptr(mbar_arrived, stage)
                    is_v_operand = operand_idx & cutlass.Int32(1)
                    if cutlass.const_expr(self.use_paged_kv):
                        if is_v_operand == 0:
                            self.load_one_paged_kv_tile(
                                s_operand,
                                tma_k_desc_ptr,
                                arrived,
                                page_ids,
                                kv_head_idx,
                                kv_l2_cache_hint,
                            )
                        else:
                            self.load_one_paged_kv_tile(
                                s_operand,
                                tma_v_desc_ptr,
                                arrived,
                                page_ids,
                                kv_head_idx,
                                kv_l2_cache_hint,
                            )
                    else:
                        if is_v_operand == 0:
                            self.load_one_kv_tile(
                                s_operand,
                                tma_k_desc_ptr,
                                arrived,
                                kv_head_idx,
                                kv_tma_seq_idx,
                                kv_l2_cache_hint,
                            )
                        else:
                            self.load_one_kv_tile(
                                s_operand,
                                tma_v_desc_ptr,
                                arrived,
                                kv_head_idx,
                                kv_tma_seq_idx,
                                kv_l2_cache_hint,
                            )
                    operand_idx += 1
                    stage += 1
                    if stage == self.kv_pipeline_stages:
                        stage = cutlass.Int32(0)
                        # The first wrap only fills the initially empty slots.
                        # Every later wrap advances each stage by one generation.
                        if operand_idx > self.kv_pipeline_stages:
                            empty_phase ^= cutlass.Int32(1)
            else:
                sK = self.get_kv_stage_ptr(sKV, cutlass.Int32(0))
                sV = self.get_kv_stage_ptr(sKV, cutlass.Int32(1))
                k_arrived = self.get_mbar_stage_ptr(mbar_arrived, cutlass.Int32(0))
                v_arrived = self.get_mbar_stage_ptr(mbar_arrived, cutlass.Int32(1))
                k_consumed = self.get_mbar_stage_ptr(mbar_consumed, cutlass.Int32(0))
                v_consumed = self.get_mbar_stage_ptr(mbar_consumed, cutlass.Int32(1))
                fixed_kv_l2_cache_hint = None
                if cutlass.const_expr(uses_kv_l2_policy):
                    fixed_kv_l2_cache_hint = kv_l2_cache_hint

                kv_seq_idx = (num_kv_tiles - 1) * self.kv_tile
                if cutlass.const_expr(self.use_paged_kv):
                    page_ids = self.resolve_paged_kv_tile_pages(
                        batch_idx,
                        kv_seq_idx,
                        seqlen_k,
                        block_tables,
                        cached_page,
                    )
                    self.load_one_paged_kv_tile(
                        sK,
                        tma_k_desc_ptr,
                        k_arrived,
                        page_ids,
                        kv_head_idx,
                        fixed_kv_l2_cache_hint,
                    )
                    self.load_one_paged_kv_tile(
                        sV,
                        tma_v_desc_ptr,
                        v_arrived,
                        page_ids,
                        kv_head_idx,
                        fixed_kv_l2_cache_hint,
                    )
                else:
                    kv_tma_seq_idx = k_token_base + kv_seq_idx
                    self.load_one_kv_tile(
                        sK,
                        tma_k_desc_ptr,
                        k_arrived,
                        kv_head_idx,
                        kv_tma_seq_idx,
                        fixed_kv_l2_cache_hint,
                    )
                    self.load_one_kv_tile(
                        sV,
                        tma_v_desc_ptr,
                        v_arrived,
                        kv_head_idx,
                        kv_tma_seq_idx,
                        fixed_kv_l2_cache_hint,
                    )

                kv_seq_idx -= self.kv_tile
                consumed_phase = cutlass.Int32(0)
                while kv_seq_idx >= 0:
                    if cutlass.const_expr(self.use_paged_kv):
                        page_ids = self.resolve_paged_kv_tile_pages(
                            batch_idx,
                            kv_seq_idx,
                            seqlen_k,
                            block_tables,
                            cached_page,
                        )
                    self.wait_pipeline_barrier(k_consumed, consumed_phase)
                    if cutlass.const_expr(self.use_paged_kv):
                        self.load_one_paged_kv_tile(
                            sK,
                            tma_k_desc_ptr,
                            k_arrived,
                            page_ids,
                            kv_head_idx,
                            fixed_kv_l2_cache_hint,
                        )
                    else:
                        kv_tma_seq_idx = k_token_base + kv_seq_idx
                        self.load_one_kv_tile(
                            sK,
                            tma_k_desc_ptr,
                            k_arrived,
                            kv_head_idx,
                            kv_tma_seq_idx,
                            fixed_kv_l2_cache_hint,
                        )
                    self.wait_pipeline_barrier(v_consumed, consumed_phase)
                    if cutlass.const_expr(self.use_paged_kv):
                        self.load_one_paged_kv_tile(
                            sV,
                            tma_v_desc_ptr,
                            v_arrived,
                            page_ids,
                            kv_head_idx,
                            fixed_kv_l2_cache_hint,
                        )
                    else:
                        self.load_one_kv_tile(
                            sV,
                            tma_v_desc_ptr,
                            v_arrived,
                            kv_head_idx,
                            kv_tma_seq_idx,
                            fixed_kv_l2_cache_hint,
                        )
                    kv_seq_idx -= self.kv_tile
                    consumed_phase ^= cutlass.Int32(1)
        # /////////////////////////////////////////////////////////////////////////////
        #  COMPUTE
        # /////////////////////////////////////////////////////////////////////////////
        elif warp < self.load_warp_id and q_seq_idx < seqlen_q and seqlen_k > 0:
            prims.setmaxregister(240, prims.SetMaxRegisterAction.INCREASE)

            q_l2_cache_hint = cutlass.Int64(0)
            if cutlass.const_expr(
                self.head_tile in (128, 256)
                and self.q_tile == 128
                and self.kv_tile == 128
            ):
                q_l2_cache_hint = cute.arch.inline_ptx(
                    "createpolicy.fractional.L2::evict_first.b64 $0, 1.0;",
                    write_only_types=[cutlass.Int64],
                    read_only_args=[],
                )
            compute_warp_idx = warp
            q_warp_row0 = compute_warp_idx * self.MMA_TILER[0]

            uses_distributed_row_state = cutlass.const_expr(
                self.head_tile == 256 and self.q_tile == 128 and self.kv_tile == 128
            )
            uses_unified_kv_ring = uses_distributed_row_state
            if cutlass.const_expr(uses_distributed_row_state):
                # D256/Q128/KV128 crosses the register cliff with four state
                # scalars. Distribute them across each threadquad: lanes 0/1
                # hold maxima and lanes 2/3 hold sums.
                row_state = cutlass.Float32(0.0)
                if lane_mod4 < 2:
                    row_state = -cutlass.Float32.inf
            else:
                # Smaller live sets fit the direct per-lane representation,
                # which avoids distributed-state shuffles entirely.
                row_state = cutlass.Float32(0.0)
                row_max = cutlass.Array(cutlass.Float32, 2, alignment=16)
                row_sum = cutlass.Array(cutlass.Float32, 2, alignment=16)
                for i in cutlass.range_constexpr(2):
                    row_max[i] = -cutlass.Float32.inf
                    row_sum[i] = 0.0

            # Per-lane fp32 accumulator for O = P @ V.
            o_regs = cutlass.Array(
                cutlass.Float32,
                self.pv_d_frags * 4,
                alignment=16,
            )
            for i in cutlass.range_constexpr(self.pv_d_frags * 4):
                o_regs[i] = 0.0

            basic_params = SimpleNamespace(
                # Problem shape and logical Q coordinates.
                seqlen_q=seqlen_q,
                seqlen_k=seqlen_k,
                head_dim=head_dim,
                q_ptr=q_ptr,
                q_seq_idx=q_seq_idx,
                q_head_base=q_head_base,
                causal_q_offset=causal_q_offset,
                q_seq_stride=q_seq_stride,
                q_l2_cache_hint=q_l2_cache_hint,
                # Compute-warp row and lane ownership.
                q_warp_row0=q_warp_row0,
                lane=lane,
                lane_div4=lane_div4,
                lane_mod4=lane_mod4,
                lane_div8=lane_div8,
                lane_mod8=lane_mod8,
                # Unified K/V ring arrival and reuse handshakes.
                mbar_arrived=mbar_arrived,
                mbar_consumed=mbar_consumed,
                # Fixed two-buffer aliases used by smaller head dimensions.
                mbar_k_arrived=mbar_arrived,
                mbar_v_arrived=mbar_arrived.subview(1),
                mbar_k_consumed=mbar_consumed,
                mbar_v_consumed=mbar_consumed.subview(1),
            )
            if cutlass.const_expr(uses_distributed_row_state):
                mma_params = SimpleNamespace(
                    sKV=sKV,
                    o_regs=o_regs,
                )
                softmax_params = SimpleNamespace(
                    softmax_scale_log2=softmax_scale_log2,
                )
            else:
                mma_params = SimpleNamespace(
                    sKV=sKV,
                    sK=sKV,
                    sV=sKV.subview(self.kv_tile_elems),
                    o_regs=o_regs,
                )
                softmax_params = SimpleNamespace(
                    row_max=row_max,
                    row_sum=row_sum,
                    softmax_scale_log2=softmax_scale_log2,
                )

            # Load Q into registers.
            q_regs = self.load_q_tile(basic_params)

            # K/V runs right-to-left, so tail/causal boundary tiles form a short
            # masked prefix. Once that prefix is consumed, every remaining tile
            # is fully valid in both ragged and paged storage, so the hot loop
            # can skip per-score bounds checks.
            kv_tile_idx = num_kv_tiles - 1
            masked_kv_tile_count = 1
            if cutlass.const_expr(self.is_causal):
                # Packed requests can be rectangular, so reserve one full
                # extra overlap for the bottom-right causal offset.
                masked_kv_tile_count = ceil_div(
                    self.q_tile + self.kv_tile - 1, self.kv_tile
                )

            # Phase 1: potentially masked iterations.
            pipeline_k_stage = cutlass.Int32(0)
            for step in cutlass.range_constexpr(masked_kv_tile_count):
                if kv_tile_idx >= 0:
                    if cutlass.const_expr(uses_distributed_row_state):
                        row_state = self.compute_one_kv_tile(
                            basic_params,
                            mma_params,
                            softmax_params,
                            q_regs,
                            row_state,
                            kv_tile_idx,
                            pipeline_k_stage,
                            is_masked_frontier_tile=True,
                            uses_softmax_init_path=(step == 0),
                        )
                    else:
                        self.compute_one_kv_tile_single_buffer(
                            basic_params,
                            mma_params,
                            softmax_params,
                            q_regs,
                            num_kv_tiles,
                            kv_tile_idx,
                            is_masked_frontier_tile=True,
                            uses_softmax_init_path=(step == 0),
                        )
                kv_tile_idx -= 1
                if cutlass.const_expr(uses_distributed_row_state):
                    if pipeline_k_stage == 0:
                        pipeline_k_stage = cutlass.Int32(2)
                    else:
                        pipeline_k_stage -= 1

            # Phase 2: remaining tiles outside the explicit mask frontier.
            while kv_tile_idx >= 0:
                if cutlass.const_expr(uses_unified_kv_ring):
                    row_state = self.compute_one_kv_tile(
                        basic_params,
                        mma_params,
                        softmax_params,
                        q_regs,
                        row_state,
                        kv_tile_idx,
                        pipeline_k_stage,
                        is_masked_frontier_tile=False,
                        uses_softmax_init_path=False,
                    )
                else:
                    self.compute_one_kv_tile_single_buffer(
                        basic_params,
                        mma_params,
                        softmax_params,
                        q_regs,
                        num_kv_tiles,
                        kv_tile_idx,
                        is_masked_frontier_tile=False,
                        uses_softmax_init_path=False,
                    )
                kv_tile_idx -= 1
                if cutlass.const_expr(uses_unified_kv_ring):
                    if pipeline_k_stage == 0:
                        pipeline_k_stage = cutlass.Int32(2)
                    else:
                        pipeline_k_stage -= 1

            prims.barrier_cta_sync(
                self.COMPUTE_BARRIER_ID, thread_count=self.threads_compute
            )

            if cutlass.const_expr(uses_distributed_row_state):
                quad_lane_base = lane_div4 * 4
                row_max_0 = prims.shfl_sync(
                    thread_mask=0xFFFFFFFF,
                    val=row_state,
                    offset=quad_lane_base,
                    mask_and_clamp=0x1F,
                    kind=prims.Shfl.IDX,
                )
                row_max_1 = prims.shfl_sync(
                    thread_mask=0xFFFFFFFF,
                    val=row_state,
                    offset=quad_lane_base + 1,
                    mask_and_clamp=0x1F,
                    kind=prims.Shfl.IDX,
                )
                row_sum_0 = prims.shfl_sync(
                    thread_mask=0xFFFFFFFF,
                    val=row_state,
                    offset=quad_lane_base + 2,
                    mask_and_clamp=0x1F,
                    kind=prims.Shfl.IDX,
                )
                row_sum_1 = prims.shfl_sync(
                    thread_mask=0xFFFFFFFF,
                    val=row_state,
                    offset=quad_lane_base + 3,
                    mask_and_clamp=0x1F,
                    kind=prims.Shfl.IDX,
                )
            else:
                row_max_0 = row_max[0]
                row_max_1 = row_max[1]
                row_sum_0 = row_sum[0]
                row_sum_1 = row_sum[1]

            # The four lanes in each lane quad hold identical final online
            # softmax state for one Q row. Have one leader write each of the
            # two rows owned by the quad. FlashInfer LSE is log2-base:
            #
            #   log2(sum(exp(score * scale)))
            #     = row_max * scale * log2(e) + log2(row_sum).
            if cutlass.const_expr(lse is not None):
                if lane_mod4 == 0:
                    for row_half in cutlass.range_constexpr(2):
                        lse_q_seq_idx = (
                            q_seq_idx + q_warp_row0 + lane_div4 + row_half * 8
                        )
                        if lse_q_seq_idx < seqlen_q:
                            if cutlass.const_expr(row_half == 0):
                                lse_row_max = row_max_0
                                lse_row_sum = row_sum_0
                            else:
                                lse_row_max = row_max_1
                                lse_row_sum = row_sum_1
                            lse[
                                q_token_base + lse_q_seq_idx,
                                q_head_idx,
                            ] = lse_row_max * softmax_scale_log2 + cute.math.log2(
                                lse_row_sum, fastmath=True
                            )

            # No compute warp may alias sKV as sO while a peer still reads the
            # final sV tile. After that lifetime barrier, normalize O, stage an
            # stmatrix-friendly layout, and store 128b per lane to GMEM.
            sO = cutlass.Pointer(sKV.data_ptr(), dtype=o.dtype)
            row_sum_inv_0 = (
                cute.math.rcp(row_sum_0, approx=True, ftz=True) * output_scale
            )
            row_sum_inv_1 = (
                cute.math.rcp(row_sum_1, approx=True, ftz=True) * output_scale
            )
            for d_frag_pair in cutlass.range_constexpr(self.pv_d_frags // 2):
                o_off = (d_frag_pair * 2) * 4
                o_scaled0 = mul_packed_f32x2(
                    (o_regs[o_off + 0], o_regs[o_off + 1]),
                    (row_sum_inv_0, row_sum_inv_0),
                )
                o_scaled1 = mul_packed_f32x2(
                    (o_regs[o_off + 2], o_regs[o_off + 3]),
                    (row_sum_inv_1, row_sum_inv_1),
                )
                o_scaled2 = mul_packed_f32x2(
                    (o_regs[o_off + 4], o_regs[o_off + 5]),
                    (row_sum_inv_0, row_sum_inv_0),
                )
                o_scaled3 = mul_packed_f32x2(
                    (o_regs[o_off + 6], o_regs[o_off + 7]),
                    (row_sum_inv_1, row_sum_inv_1),
                )
                o_packed0 = pack_to_i32(
                    (o.dtype(o_scaled0[0]), o.dtype(o_scaled0[1])),
                    o.dtype,
                )
                o_packed1 = pack_to_i32(
                    (o.dtype(o_scaled1[0]), o.dtype(o_scaled1[1])),
                    o.dtype,
                )
                o_packed2 = pack_to_i32(
                    (o.dtype(o_scaled2[0]), o.dtype(o_scaled2[1])),
                    o.dtype,
                )
                o_packed3 = pack_to_i32(
                    (o.dtype(o_scaled3[0]), o.dtype(o_scaled3[1])),
                    o.dtype,
                )
                sO_ptr = (
                    sO
                    + compute_warp_idx * (self.pv_d_frags // 2) * (16 * 16)
                    + d_frag_pair * (16 * 16)
                    + lane * 8
                )
                prims.stmatrix(
                    sO_ptr,
                    [o_packed0, o_packed1, o_packed2, o_packed3],
                    prims.MMALayout.ROW,
                )

            store_row = lane_mod8 + ((lane_div8) % 2) * 8
            store_col = lane_div16 * 8
            store_q_seq_idx = q_seq_idx + q_warp_row0 + store_row
            for d_frag_pair in cutlass.range_constexpr(self.pv_d_frags // 2):
                store_col_in_cta = d_frag_pair * 16 + store_col
                sO_ptr = (
                    sO
                    + compute_warp_idx * (self.pv_d_frags // 2) * (16 * 16)
                    + d_frag_pair * (16 * 16)
                    + lane * 8
                )
                o_vec = sO_ptr.load(count=8, alignment=16)
                gO_ptr = (
                    o_ptr
                    + o_head_base
                    + store_q_seq_idx * q_seq_stride
                    + store_col_in_cta
                )
                if store_q_seq_idx < seqlen_q and store_col_in_cta < head_dim:
                    gO_ptr.store(o_vec, alignment=16)

        # A zero-length K/V sequence has a mathematically empty attention
        # reduction. This is ensured by writing O=0 for the
        # valid Q rows in this CTA. Compute warps own disjoint groups of 16
        # rows, and the two lane half-warps cover one aligned 16-element output
        # vector per row, so this path needs neither SMEM nor a CTA barrier.
        elif warp < self.load_warp_id and q_seq_idx < seqlen_q:
            prims.setmaxregister(24, prims.SetMaxRegisterAction.DECREASE)
            zero_vector = cutlass.Vector.from_elements(
                tuple(self.out_dtype(0.0) for _ in range(8)),
                dtype=self.out_dtype,
            )
            zero_q_seq_idx = (
                q_seq_idx + warp * self.MMA_TILER[0] + lane % self.MMA_TILER[0]
            )
            if cutlass.const_expr(lse is not None):
                if lane < self.MMA_TILER[0] and zero_q_seq_idx < seqlen_q:
                    lse[
                        q_token_base + zero_q_seq_idx,
                        q_head_idx,
                    ] = -cutlass.Float32.inf
            for head_chunk in cutlass.range_constexpr(self.head_tile // 16):
                zero_head_col = head_chunk * 16 + lane_div16 * 8
                if zero_q_seq_idx < seqlen_q:
                    (
                        o_ptr
                        + o_head_base
                        + zero_q_seq_idx * q_seq_stride
                        + zero_head_col
                    ).store(zero_vector, alignment=16)

        # /////////////////////////////////////////////////////////////////////////////
        #  EMPTY / INACTIVE CTA
        # /////////////////////////////////////////////////////////////////////////////
        else:
            prims.setmaxregister(24, prims.SetMaxRegisterAction.DECREASE)

        cute.arch.griddepcontrol_launch_dependents()

    @cute.jit
    def __call__(
        self,
        q: cute.Tensor,
        k: cute.Tensor,
        v: cute.Tensor,
        o: cute.Tensor,
        lse: cute.Tensor | None,
        softmax_scale_log2: cutlass.Float32,
        output_scale: cutlass.Float32,
        stream: cuda_driver.CUstream,
        seqlens_kv: cute.Tensor | None = None,
        cu_seqlens_q: cute.Tensor | None = None,
        block_tables: cute.Tensor | None = None,
        cu_seqlens_k: cute.Tensor | None = None,
        max_seqlen_q: cutlass.Int32 | None = None,
        use_pdl: bool = False,
    ) -> None:
        """Launch the SM120 PRIM FMHA FP8 kernel.

        :param q: Packed query ``(total_q, Hq, D)``.
        :param k: Packed key ``(total_k, Hkv, D)`` or HND paged K pool
            ``(num_pages, Hkv, num_tokens_per_page, D)``. Paged pools must
            contain finite values in every slot, including unused padding.
        :param v: Packed or paged value with the same shape and finite-value
            contract as K.
        :param o: Packed output ``(total_q, Hq, D)``.
        :param lse: Optional packed float32 log2 LSE ``(total_q,Hq)``.
        :param softmax_scale_log2: ``softmax_scale * log2(e)``.
        :param output_scale: Scalar multiplier applied to normalized output.
        :param stream: CUDA stream used for the launch.
        :param seqlens_kv: Runtime K/V lengths, one Int32 per request.
        :param cu_seqlens_q: Packed-Q cumulative Int32 offsets.
        :param block_tables: Shared K/V page indices with shape
            ``(B, max_num_pages_per_seq_kv)``.
        :param cu_seqlens_k: Packed-K/V cumulative Int32 offsets.
        :param max_seqlen_q: Contiguous-varlen launch-grid bound.
        :param use_pdl: Whether to use Programmatic Dependent Launch.

        Direct callers must satisfy the same shape contract enforced by
        :meth:`can_implement_paged`.
        """
        head_dim = q.shape[2]
        output_head_dim = o.shape[2]
        num_heads_q = q.shape[1]
        batch_size = cute.size(cu_seqlens_q) - 1
        if cutlass.const_expr(self.use_paged_kv):
            if cutlass.const_expr(
                head_dim != k.shape[3]
                or head_dim != v.shape[3]
                or head_dim != output_head_dim
                or head_dim != self.head_tile
                or k.shape[2] != self.num_tokens_per_page
                or v.shape[2] != self.num_tokens_per_page
            ):
                raise ValueError(
                    "runtime paged tensors must match the compiled structural config"
                )
            tma_k_desc = cuda.create_tensor_map_tiled_from_view(
                k,
                box_dims=(
                    1,
                    1,
                    self.num_tokens_per_page,
                    self.tma_copy_head_per_iter,
                ),
                stride_order=(3, 2, 1, 0),
                swizzle=self.tma_swizzle,
            )
            tma_v_desc = cuda.create_tensor_map_tiled_from_view(
                v,
                box_dims=(
                    1,
                    1,
                    self.num_tokens_per_page,
                    self.tma_copy_head_per_iter,
                ),
                stride_order=(3, 2, 1, 0),
                swizzle=self.tma_swizzle,
            )
        else:
            if cutlass.const_expr(
                head_dim != k.shape[2]
                or head_dim != v.shape[2]
                or head_dim != o.shape[2]
                or head_dim != self.head_tile
            ):
                raise ValueError(
                    "runtime packed tensors must match the kernel head_tile"
                )
            tma_k_desc = cuda.create_tensor_map_tiled_from_view(
                k,
                box_dims=(self.kv_tile, 1, self.tma_copy_head_per_iter),
                stride_order=(2, 1, 0),
                swizzle=self.tma_swizzle,
            )
            tma_v_desc = cuda.create_tensor_map_tiled_from_view(
                v,
                box_dims=(self.kv_tile, 1, self.tma_copy_head_per_iter),
                stride_order=(2, 1, 0),
                swizzle=self.tma_swizzle,
            )
        if cutlass.const_expr(self.balanced_scheduler):
            # Keep Q heads adjacent in launch order. GQA/MQA heads that share
            # K/V are scheduled close together, while Q tiles run from the
            # longest causal rows to the shortest.
            grid = (
                num_heads_q,
                cute.ceil_div(max_seqlen_q, self.q_tile),
                batch_size,
            )
        else:
            grid = (
                cute.ceil_div(max_seqlen_q, self.q_tile),
                num_heads_q,
                batch_size,
            )

        self.kernel(
            q,
            k,
            v,
            o,
            lse,
            tma_k_desc,
            tma_v_desc,
            softmax_scale_log2,
            output_scale,
            seqlens_kv,
            cu_seqlens_q,
            block_tables,
            cu_seqlens_k,
        ).launch(
            grid=grid,
            block=(self.threads_per_cta, 1, 1),
            stream=stream,
            min_blocks_per_mp=1,
            use_pdl=use_pdl,
        )
