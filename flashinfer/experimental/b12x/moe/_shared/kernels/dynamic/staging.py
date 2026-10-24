"""Dynamic MoE operand staging helpers."""

from __future__ import annotations

from cutlass.cutlass_dsl import Int32
from cutlass.cutlass_dsl import Int64
from typing import Tuple
from cutlass.cutlass_dsl import Uint32
from cutlass.cutlass_dsl import Uint8
from b12x._lib.intrinsics.packed import bfloat2_to_float2_scaled
from b12x._lib.intrinsics.memory import cp_async4_shared_global
from b12x._lib.intrinsics.memory import cp_async_u32_shared_global
from b12x._lib.intrinsics.memory import cp_async_u64_shared_global
import cutlass.cute as cute
import cutlass
from cutlass.cutlass_dsl import dsl_user_op
from b12x._lib.intrinsics.math import fabs_f32
from b12x._lib.intrinsics.math import fmax_f32
from b12x._lib.intrinsics.memory import get_ptr_as_int64
from b12x._lib.intrinsics.memory import ld_global_v4_u32
from cutlass._mlir.dialects import llvm
from b12x._lib.intrinsics.atomics import st_shared_u32
from b12x._lib.intrinsics.memory import st_shared_u8


_SF_VEC_SIZE = 16

_TASK_SLICE_CHUNK = 1

_WORK_ITEM_FIELDS = 5

_WORK_EXPERT = 0

_WORK_M_TILE = 1

_WORK_SLICE_BEGIN = 2

_WORK_SLICE_COUNT = 3

_WORK_VALID_ROWS = 4

_CTRL_HAS_WORK = 0

_CTRL_DONE = 1

_CTRL_WORK_BEGIN = 2

_CTRL_CLAIMED_SLOT = 7

_WORK_SOURCE_PERSISTENT_GRID = "persistent_grid"

_WORK_SOURCE_MATERIALIZED_QUEUE = "materialized_queue"

_WORK_SOURCE_READY_QUEUE = "ready_queue"

_WORK_SOURCES = {
    _WORK_SOURCE_PERSISTENT_GRID,
    _WORK_SOURCE_MATERIALIZED_QUEUE,
    _WORK_SOURCE_READY_QUEUE,
}

# w4a8 smem staging geometry: one 128-row x 64-byte packed-FP4 B tile per
# k-tile. Rows pad to 80 bytes: 16-aligned (cp.async.cg requires dst
# alignment = copy size) and 20*g mod 32 spreads the eight g-rows a lane
# quad touches across distinct bank groups.
_W4A8_B_ROW_PAD = 80


@cute.jit
def _load_bf16x32_to_f32(
    a_input: cute.Tensor,
    linear_offset: Int32,
) -> Tuple[cute.Tensor, cutlass.Float32]:
    """Load one aligned BF16x32 quant block with four 128-bit operations.

    Keeping the converted values in one rmem tensor gives the quantizer the
    same convenient scalar view without emitting 32 independent ``ld.u16``
    instructions.  Immediate conversion of each four-word vector also avoids
    extending a second 16-register packed-input tensor across the reduction.
    """

    values = cute.make_rmem_tensor((32,), cutlass.Float32)
    block_max = cutlass.Float32(0.0)
    base = get_ptr_as_int64(a_input, linear_offset)
    for vec in cutlass.range_constexpr(4):
        w0, w1, w2, w3 = ld_global_v4_u32(base + Int64(vec * 16))
        v0, v1 = bfloat2_to_float2_scaled(w0, cutlass.Float32(1.0))
        v2, v3 = bfloat2_to_float2_scaled(w1, cutlass.Float32(1.0))
        v4, v5 = bfloat2_to_float2_scaled(w2, cutlass.Float32(1.0))
        v6, v7 = bfloat2_to_float2_scaled(w3, cutlass.Float32(1.0))
        values[vec * 8 + 0] = v0
        values[vec * 8 + 1] = v1
        values[vec * 8 + 2] = v2
        values[vec * 8 + 3] = v3
        values[vec * 8 + 4] = v4
        values[vec * 8 + 5] = v5
        values[vec * 8 + 6] = v6
        values[vec * 8 + 7] = v7
        block_max = fmax_f32(block_max, fabs_f32(v0))
        block_max = fmax_f32(block_max, fabs_f32(v1))
        block_max = fmax_f32(block_max, fabs_f32(v2))
        block_max = fmax_f32(block_max, fabs_f32(v3))
        block_max = fmax_f32(block_max, fabs_f32(v4))
        block_max = fmax_f32(block_max, fabs_f32(v5))
        block_max = fmax_f32(block_max, fabs_f32(v6))
        block_max = fmax_f32(block_max, fabs_f32(v7))
    return values, block_max


_W4A8_TMA_TILE_BYTES = 128 * 64  # one (128 n, 128 fp4-k) TMA box

_W4A8_B_BUF_BYTES = 128 * _W4A8_B_ROW_PAD


@cute.jit
def _w4a8_stage_b_tile(
    b_u32: cute.Tensor,
    smem_base: Int32,
    base_idx: Int64,
    row_u32_stride: Int32,
    tidx: Int32,
    tcnt: cutlass.Constexpr,
    row_pad: cutlass.Constexpr = _W4A8_B_ROW_PAD,
):
    """cp.async one 128x64B packed-FP4 tile into smem rows.

    Unpadded (64B) rows XOR-swizzle the 16B chunk by the row's low bits —
    without it every consumer B load is a 4-8 way bank conflict."""
    for i in cutlass.range_constexpr((128 * 4 + tcnt - 1) // tcnt):
        idx = tidx + Int32(i * tcnt)
        if idx < Int32(128 * 4):
            r = idx >> Int32(2)
            ch = idx & Int32(3)
            gaddr = get_ptr_as_int64(
                b_u32,
                base_idx + Int64(r) * Int64(row_u32_stride) + Int64(ch * 4),
            )
            cp_async4_shared_global(
                smem_base + r * Int32(row_pad) + ch * Int32(16), gaddr
            )


@cute.jit
def _w4a8_stage_repacked_b_half(
    b_rp_u32: cute.Tensor,
    smem_base: Int32,
    tile_base: Int64,
    half: Int32,
    tidx: Int32,
    tcnt: cutlass.Constexpr,
):
    """Stage one N128 half of an N256 repacked W4A8 B tile.

    The source tile is ``[kb=4, chunk=8, lane=32, n8_in_chunk=4]``.
    Selecting four chunks per ``kb`` leaves four discontiguous 2KB source
    spans; compact them into an 8KB shared tile with the same inner layout.
    Each copy is exactly the four n8 words consumed by one ``ld.shared.v4``.
    """

    transfers = 4 * 4 * 32
    for i in cutlass.range_constexpr((transfers + tcnt - 1) // tcnt):
        idx = tidx + Int32(i * tcnt)
        if idx < Int32(transfers):
            lane = idx & Int32(31)
            kc = idx >> Int32(5)
            chunk = kc & Int32(3)
            kb = kc >> Int32(2)
            src_word = (
                tile_base
                + Int64(kb * 8 * 32 * 4)
                + Int64((half * Int32(4) + chunk) * Int32(32 * 4))
                + Int64(lane * Int32(4))
            )
            cp_async4_shared_global(
                smem_base + (idx << Int32(4)),
                get_ptr_as_int64(b_rp_u32, src_word),
            )


@cute.jit
def _w4a8_stage_repacked_b_full(
    b_rp_u32: cute.Tensor,
    smem_base: Int32,
    tile_base: Int64,
    tidx: Int32,
    tcnt: cutlass.Constexpr,
):
    """Stage one contiguous N256 repacked W4A8 B tile (16KB)."""

    transfers = 4 * 8 * 32
    for i in cutlass.range_constexpr((transfers + tcnt - 1) // tcnt):
        idx = tidx + Int32(i * tcnt)
        if idx < Int32(transfers):
            cp_async4_shared_global(
                smem_base + (idx << Int32(4)),
                get_ptr_as_int64(b_rp_u32, tile_base + Int64(idx * 4)),
            )


@dsl_user_op
def _w4a8_fc2_barrier(threads: Int32, *, loc=None, ip=None):
    # Compute and producer warps reach matching barriers at distinct PCs.
    llvm.inline_asm(
        None,
        [threads.ir_value(loc=loc, ip=ip)],
        "barrier.cta.sync 0, $0;",
        "r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@cute.jit
def _w4a8_stage_repacked_sfb_half(
    sfb_rp_u32: cute.Tensor,
    smem_base: Int32,
    tile_base: Int64,
    half: Int32,
    tidx: Int32,
    tcnt: cutlass.Constexpr,
    csf_reader: cutlass.Constexpr = None,
):
    """Stage the 512B SFB half paired with an N128 B half."""

    if cutlass.const_expr(csf_reader is not None):
        target = cute.make_ptr(
            Uint8, smem_base, cute.AddressSpace.smem, assumed_align=16
        )
        csf_reader.stage_native(
            sfb_rp_u32.iterator,
            (tile_base + half.to(Int64) * Int64(128)) * Int64(4),
            target,
            tidx,
        )
    transfers = (16 * 8) // 4
    for i in cutlass.range_constexpr(
        0 if csf_reader is not None else (transfers + tcnt - 1) // tcnt
    ):
        idx = tidx + Int32(i * tcnt)
        if idx < Int32(transfers):
            src_word = tile_base + Int64(half * Int32(16 * 8) + idx * 4)
            cp_async4_shared_global(
                smem_base + (idx << Int32(4)),
                get_ptr_as_int64(sfb_rp_u32, src_word),
            )


@cute.jit
def _w4a8_stage_repacked_sfb_full(
    sfb_rp_u32: cute.Tensor,
    smem_base: Int32,
    tile_base: Int64,
    tidx: Int32,
    tcnt: cutlass.Constexpr,
    csf_reader: cutlass.Constexpr = None,
):
    """Stage one contiguous N256 repacked SFB tile (1KB)."""

    if cutlass.const_expr(csf_reader is not None):
        for half in cutlass.range_constexpr(2):
            target = cute.make_ptr(
                Uint8,
                smem_base + Int32(half * 512),
                cute.AddressSpace.smem,
                assumed_align=16,
            )
            csf_reader.stage_native(
                sfb_rp_u32.iterator,
                (tile_base + Int64(half * 128)) * Int64(4),
                target,
                tidx,
            )
    transfers = (32 * 8) // 4
    for i in cutlass.range_constexpr(
        0 if csf_reader is not None else (transfers + tcnt - 1) // tcnt
    ):
        idx = tidx + Int32(i * tcnt)
        if idx < Int32(transfers):
            cp_async4_shared_global(
                smem_base + (idx << Int32(4)),
                get_ptr_as_int64(sfb_rp_u32, tile_base + Int64(idx * 4)),
            )


@cute.jit
def _w4a8_stage_n64_b_tile(
    b_rp_u32: cute.Tensor,
    smem_base: Int32,
    expert_idx: Int32,
    n64_first: Int32,
    k128_tile: Int32,
    total_n64: cutlass.Constexpr,
    total_k32: cutlass.Constexpr,
    tidx: Int32,
    tcnt: cutlass.Constexpr,
):
    """Stage an exact packed N128/K128 tile from one or two N64 tiles."""

    transfers = 2 * 4 * 2 * 32
    expert_words = total_n64 * total_k32 * 256
    for i in cutlass.range_constexpr((transfers + tcnt - 1) // tcnt):
        idx = tidx + Int32(i * tcnt)
        if idx < Int32(transfers):
            lane = idx & Int32(31)
            tmp = idx >> Int32(5)
            n32 = tmp & Int32(1)
            tmp = tmp >> Int32(1)
            kb = tmp & Int32(3)
            n64_half = tmp >> Int32(2)
            source_n64 = n64_first + n64_half
            source_k32 = k128_tile * Int32(4) + kb
            dst_transfer = (kb * Int32(4) + n64_half * Int32(2) + n32) * Int32(
                32
            ) + lane
            dst_addr = smem_base + (dst_transfer << Int32(4))
            if source_n64 < Int32(total_n64) and source_k32 < Int32(total_k32):
                source_word = (
                    Int64(expert_idx) * Int64(expert_words)
                    + Int64(source_n64) * Int64(total_k32 * 256)
                    + Int64(source_k32) * Int64(256)
                    + Int64(n32) * Int64(128)
                    + Int64(lane) * Int64(4)
                )
                cp_async4_shared_global(
                    dst_addr,
                    get_ptr_as_int64(b_rp_u32, source_word),
                )
            else:
                st_shared_u32(dst_addr, Uint32(0))
                st_shared_u32(dst_addr + Int32(4), Uint32(0))
                st_shared_u32(dst_addr + Int32(8), Uint32(0))
                st_shared_u32(dst_addr + Int32(12), Uint32(0))


@cute.jit
def _w4a8_stage_n64_sfb_tile(
    sfb_rp_u32: cute.Tensor,
    smem_base: Int32,
    expert_idx: Int32,
    n64_first: Int32,
    k128_tile: Int32,
    total_n64: cutlass.Constexpr,
    total_k32: cutlass.Constexpr,
    tidx: Int32,
    tcnt: cutlass.Constexpr,
):
    """Stage exact N64-packed E8M0 bytes into the N128/K128 SFB tile."""

    sfb_u8 = cute.recast_tensor(sfb_rp_u32, Uint8)
    copies = 2 * 4 * 64
    expert_bytes = total_n64 * total_k32 * 64
    full_k_tiles = total_k32 // 4
    for i in cutlass.range_constexpr((copies + tcnt - 1) // tcnt):
        idx = tidx + Int32(i * tcnt)
        if idx < Int32(copies):
            row = idx & Int32(63)
            tmp = idx >> Int32(6)
            kb = tmp & Int32(3)
            n64_half = tmp >> Int32(2)
            source_n64 = n64_first + n64_half
            source_k32 = k128_tile * Int32(4) + kb
            value = Uint8(0)
            if source_n64 < Int32(total_n64) and source_k32 < Int32(total_k32):
                source_kt = source_k32 >> Int32(2)
                source_kb = source_k32 & Int32(3)
                tile_cols = Int32(4)
                tile_base = source_kt * Int32(256)
                if source_kt == Int32(full_k_tiles):
                    tile_cols = Int32(total_k32 - full_k_tiles * 4)
                    tile_base = Int32(full_k_tiles * 256)
                source_byte = (
                    Int64(expert_idx) * Int64(expert_bytes)
                    + Int64(source_n64) * Int64(total_k32 * 64)
                    + Int64(tile_base)
                    + Int64(row * tile_cols + source_kb)
                )
                value = sfb_u8[source_byte]
            st_shared_u8(
                smem_base + (n64_half * Int32(64) + row) * Int32(4) + kb,
                value,
            )


@cute.jit
def _w4a8_stage_a_tile(
    pa_u32: cute.Tensor,
    smem_base: Int32,
    base_idx: Int64,
    row_u32_stride: Int32,
    tidx: Int32,
    tcnt: cutlass.Constexpr,
    rows: cutlass.Constexpr,
):
    """cp.async one rows x 128B E4M3 A tile into smem rows.

    The compact repacked path keeps rows plain here.  Its raw fragment loads
    use two aligned 64-bit accesses; changing this layout requires changing
    those addresses in lockstep."""
    for i in cutlass.range_constexpr((rows * 8 + tcnt - 1) // tcnt):
        idx = tidx + Int32(i * tcnt)
        if idx < Int32(rows * 8):
            r = idx >> Int32(3)
            ch = idx & Int32(7)
            gaddr = get_ptr_as_int64(
                pa_u32,
                base_idx + Int64(r) * Int64(row_u32_stride) + Int64(ch * 4),
            )
            cp_async4_shared_global(
                smem_base + (r << Int32(7)) + (ch << Int32(4)), gaddr
            )


@cute.jit
def _w4a8_stage_a_tile_gather(
    pa_u32: cute.Tensor,
    token_map: cute.Tensor,
    smem_base: Int32,
    physical_row_base: Int32,
    valid_rows: Int32,
    k_word_base: Int32,
    row_u32_stride: Int32,
    route_divisor: Int32,
    tidx: Int32,
    tcnt: cutlass.Constexpr,
    rows: cutlass.Constexpr,
):
    """Gather one rows x 128B E4M3 tile through physical-row token ids.

    Materialized W4A8 stores each quantized input row once, indexed by token,
    instead of fanning it out to every routed expert row.  FC1 pays the same
    gather it would have paid in a separate grouped GEMM while retaining the
    resident task scheduler and fused activation handoff.

    Deterministic scatter stores routed-pair indices in ``token_map``.
    Divide by top-k to recover the shared activation's token row; atomic
    scatter already stores token indices and supplies a divisor of one.
    """
    for i in cutlass.range_constexpr((rows * 8 + tcnt - 1) // tcnt):
        idx = tidx + Int32(i * tcnt)
        if idx < Int32(rows * 8):
            r = idx >> Int32(3)
            ch = idx & Int32(7)
            src_row = Int32(0)
            if r < valid_rows:
                src_row = token_map[physical_row_base + r].to(Int32) // route_divisor
            gaddr = get_ptr_as_int64(
                pa_u32,
                Int64(src_row) * Int64(row_u32_stride)
                + Int64(k_word_base)
                + Int64(ch << Int32(2)),
            )
            cp_async4_shared_global(
                smem_base + (r << Int32(7)) + (ch << Int32(4)), gaddr
            )


@cute.jit
def _w4a8_stage_scale_tile_gather(
    scale_u8: cute.Tensor,
    token_map: cute.Tensor,
    smem_base: Int32,
    physical_row_base: Int32,
    valid_rows: Int32,
    k_byte_base: Int32,
    row_stride: Int32,
    route_divisor: Int32,
    tidx: Int32,
    tcnt: cutlass.Constexpr,
    rows: cutlass.Constexpr,
):
    """Gather four packed UE8M0 bytes per row through ``token_map``."""
    for i in cutlass.range_constexpr((rows + tcnt - 1) // tcnt):
        r = tidx + Int32(i * tcnt)
        if r < Int32(rows):
            src_row = Int32(0)
            if r < valid_rows:
                src_row = token_map[physical_row_base + r].to(Int32) // route_divisor
            cp_async_u32_shared_global(
                smem_base + (r << Int32(2)),
                get_ptr_as_int64(
                    scale_u8, Int64(src_row) * Int64(row_stride) + Int64(k_byte_base)
                ),
            )


@cute.jit
def _w4a8_stage_bytes4(
    t_u8: cute.Tensor,
    smem_base: Int32,
    base_idx: Int64,
    row_stride: Int32,
    tidx: Int32,
    tcnt: cutlass.Constexpr,
    rows: cutlass.Constexpr = 128,
):
    """cp.async 4 contiguous bytes per row (scale grids)."""
    for i in cutlass.range_constexpr((rows + tcnt - 1) // tcnt):
        r = tidx + Int32(i * tcnt)
        if r < Int32(rows):
            gaddr = get_ptr_as_int64(t_u8, base_idx + Int64(r) * Int64(row_stride))
            cp_async_u32_shared_global(smem_base + (r << Int32(2)), gaddr)


@cute.jit
def _w4a8_stage_bytes8(
    t_u8: cute.Tensor,
    smem_base: Int32,
    base_idx: Int64,
    row_stride: Int32,
    tidx: Int32,
    tcnt: cutlass.Constexpr,
):
    """cp.async 8 contiguous bytes per row for 128 rows (residual grids)."""
    for i in cutlass.range_constexpr((128 + tcnt - 1) // tcnt):
        r = tidx + Int32(i * tcnt)
        if r < Int32(128):
            gaddr = get_ptr_as_int64(t_u8, base_idx + Int64(r) * Int64(row_stride))
            cp_async_u64_shared_global(smem_base + (r << Int32(3)), gaddr)


_PRODUCER_PAIRS_PER_WARP = 2

_FC2_TILE_RECIP_GS_NUM = 6.0 * 448.0
