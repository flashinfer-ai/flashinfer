"""CuTe DSL kernels for chunked delta-rule prefill: prologue, prepare, recurrence.

The prologue (one CTA) clears the ready flags and builds the tile tables.
The prepare kernel (one CTA per chunk tile and
head) turns raw projections into the per-tile operands of the chunked delta
rule. The recurrence kernel (one CTA per sequence, head, and value split)
walks a sequence's tiles with the state resident in registers.

Workspace tile layout (private to these kernels): the ``[16 x 128]`` bf16 tiles
are stored with their 16-byte chunks XOR-swizzled by ``row & 7`` so the
recurrence kernel's ``ldmatrix`` reads are bank-conflict free;
:func:`workspace_tiles` returns de-swizzled views for tests.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Float32, Int32, Int64, Uint32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from b12x._lib.compiler import KernelCompileSpec
from b12x._lib.compiler import compile as b12x_compile
from b12x._lib.compile_plan import attach_programs
from b12x._lib.intrinsics import (
    atomic_add_shared_i32,
    bf16_mma_m16n8k16_f32,
    cp_async_bulk_g2s_mbar,
    ld_shared_f32,
    ld_shared_v4_f32,
    ld_shared_v4_u32,
    ldmatrix_m8n8x4_b16,
    ldmatrix_m8n8x4_trans_b16,
    pack_f32x2_to_bfloat2,
    shared_ptr_to_u32,
    st_global_v4_u32,
    st_shared_v4_f32,
    st_shared_u32,
    warp_reduce,
)
from b12x._lib.runtime_control import raise_if_kernel_resolution_frozen
from b12x._lib.utils import current_cuda_stream, make_ptr
from b12x._lib.program_cache import register_program_cache

from .contract import Binding
from .workspace import WorkspaceRecord as REC

_HEAD_DIM = 128
_CHUNK = 16
_PROLOGUE_THREADS = 256
_PREPARE_THREADS = 128
_LOG2E = 1.4426950408889634
# Tiles of lookahead for the value-row L2 prefetch issued by the producer.
_V_PREFETCH_TILES = 6
_CONVERGENCE_TOKENS = 128

_PROLOGUE_CACHE: dict[tuple, Callable[..., None]] = {}
_PREPARE_CACHE: dict[tuple, Callable[..., None]] = {}
_RECURRENCE_CACHE: dict[tuple, Callable[..., None]] = {}
_WARMED: set[tuple] = set()
register_program_cache(_PROLOGUE_CACHE)
register_program_cache(_PREPARE_CACHE)
register_program_cache(_RECURRENCE_CACHE)


def _add(left: Float32, right: Float32) -> Float32:
    return left + right


@dsl_user_op
def _f32_bits(value: Float32, *, loc=None, ip=None) -> Uint32:
    return Uint32(llvm.bitcast(T.i32(), Float32(value).ir_value(loc=loc, ip=ip), loc=loc, ip=ip))


@dsl_user_op
def _exp2_approx_ftz_f32(a: Float32, *, loc=None, ip=None) -> Float32:
    """Approximate base-two exponential with subnormal results flushed to zero."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(a).ir_value(loc=loc, ip=ip)],
            "ex2.approx.ftz.f32 $0, $1;",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _cp_async_16_zfill(smem_addr: Int32, gmem_addr: Int64, src_bytes: Int32, *, loc=None, ip=None):
    """16-byte ``cp.async.cg`` that zero-fills the bytes past ``src_bytes``."""
    llvm.inline_asm(
        None,
        [
            Int32(smem_addr).ir_value(loc=loc, ip=ip),
            Int64(gmem_addr).ir_value(loc=loc, ip=ip),
            Int32(src_bytes).ir_value(loc=loc, ip=ip),
        ],
        "cp.async.cg.shared.global [$0], [$1], 16, $2;",
        "r,l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _ld_shared_v2_f32(smem_addr: Int32, *, loc=None, ip=None) -> tuple[Float32, Float32]:
    """Load two consecutive fp32 values from a shared-memory byte address."""
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.f32()]),
        [Int32(smem_addr).ir_value(loc=loc, ip=ip)],
        "ld.shared.v2.f32 {$0, $1}, [$2];",
        "=f,=f,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return (
        Float32(llvm.extractvalue(T.f32(), result, [0], loc=loc, ip=ip)),
        Float32(llvm.extractvalue(T.f32(), result, [1], loc=loc, ip=ip)),
    )


@dsl_user_op
def _bf16x2_to_f32x2(packed: Uint32, *, loc=None, ip=None) -> tuple[Float32, Float32]:
    """Unpack a bf16x2 register into (low element, high element) fp32 values."""
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.f32()]),
        [Uint32(packed).ir_value(loc=loc, ip=ip)],
        "{ .reg .b32 t; shl.b32 t, $2, 16; mov.b32 $0, t; and.b32 t, $2, 0xffff0000; mov.b32 $1, t; }",
        "=f,=f,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return (
        Float32(llvm.extractvalue(T.f32(), result, [0], loc=loc, ip=ip)),
        Float32(llvm.extractvalue(T.f32(), result, [1], loc=loc, ip=ip)),
    )


@dsl_user_op
def _stmatrix_x4_trans(smem_addr: Int32, r0: Uint32, r1: Uint32, r2: Uint32, r3: Uint32, *, loc=None, ip=None):
    """``stmatrix.sync.aligned.m8n8.x4.trans.shared.b16`` from four fragment registers."""
    llvm.inline_asm(
        None,
        [
            Int32(smem_addr).ir_value(loc=loc, ip=ip),
            Uint32(r0).ir_value(loc=loc, ip=ip),
            Uint32(r1).ir_value(loc=loc, ip=ip),
            Uint32(r2).ir_value(loc=loc, ip=ip),
            Uint32(r3).ir_value(loc=loc, ip=ip),
        ],
        "stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [$0], {$1, $2, $3, $4};",
        "r,r,r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _ld_acquire_gpu_i32(gmem_addr: Int64, *, loc=None, ip=None) -> Int32:
    """GPU-scope acquire load of one int32."""
    result = llvm.inline_asm(
        T.i32(),
        [Int64(gmem_addr).ir_value(loc=loc, ip=ip)],
        "ld.acquire.gpu.global.b32 $0, [$1];",
        "=r,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return Int32(result)


@dsl_user_op
def _st_release_gpu_i32(gmem_addr: Int64, value: Int32, *, loc=None, ip=None):
    """GPU-scope release store of one int32."""
    llvm.inline_asm(
        None,
        [Int64(gmem_addr).ir_value(loc=loc, ip=ip), Int32(value).ir_value(loc=loc, ip=ip)],
        "st.release.gpu.global.b32 [$0], $1;",
        "l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def _cp_async_mbarrier_arrive_noinc(mbar_addr: Int32, *, loc=None, ip=None):
    """Arrive on ``mbar`` once every prior cp.async of this thread has landed.

    The arrival is one of the barrier's expected arrivals (``.noinc``), so the
    barrier must be initialized with a count that includes it.
    """
    llvm.inline_asm(
        None,
        [Int32(mbar_addr).ir_value(loc=loc, ip=ip)],
        "cp.async.mbarrier.arrive.noinc.shared::cta.b64 [$0];",
        "r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def _prefetch_l2(gmem_addr: Int64, *, loc=None, ip=None):
    """Prefetch the L2 line holding ``gmem_addr``."""
    llvm.inline_asm(
        None,
        [Int64(gmem_addr).ir_value(loc=loc, ip=ip)],
        "prefetch.global.L2 [$0];",
        "l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def _nanosleep(nanoseconds: Int32, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [Int32(nanoseconds).ir_value(loc=loc, ip=ip)],
        "nanosleep.u32 $0;",
        "r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def _pointer_address(ptr: cute.Pointer, offset, *, loc=None, ip=None) -> Int64:
    """Return the global address of ``ptr[offset]`` as an Int64."""
    element = ptr + offset
    return Int64(llvm.ptrtoint(T.i64(), element.llvm_ptr, loc=loc, ip=ip))


def _numeric_type(dtype: torch.dtype) -> type[cutlass.Numeric]:
    if dtype == torch.bfloat16:
        return BFloat16
    if dtype == torch.float32:
        return Float32
    if dtype == torch.int32:
        return Int32
    if dtype == torch.int64:
        return Int64
    if dtype == torch.int8:
        return cutlass.Int8
    raise TypeError(f"unsupported Delta-rule prefill dtype {dtype}")


def _fake_pointer(dtype: type[cutlass.Numeric]) -> cute.Pointer:
    return make_ptr(dtype, 16, cute.AddressSpace.gmem, assumed_align=max(1, dtype.width // 8))


def _pointer(tensor: torch.Tensor, dtype: type[cutlass.Numeric]) -> cute.Pointer:
    return make_ptr(
        dtype, tensor.data_ptr(), cute.AddressSpace.gmem, assumed_align=max(1, dtype.width // 8)
    )


class _PrologueKernel:
    """Clear the ready flags and build the banded tile tables in one CTA.

    Tiles are laid out in bands: band ``l`` holds local tile ``l`` of every
    sequence that has one, ordered by sequence rank (longest sequence first,
    ties by sequence index). Consecutive tile positions therefore interleave
    the live sequences, so a window of positions advances every sequence at
    once. The kernel produces ``band_base[l]`` (first position of band ``l``,
    with ``band_base[l + 1] - band_base[l]`` sequences in the band),
    ``sorted_seq[rank]``, ``rank_of[seq]``, and per position ``pos_seq`` and
    ``pos_local`` (``pos_seq`` is -1 past the live tiles).
    """

    def __init__(
        self,
        *,
        max_seqs: int,
        tiles_capacity: int,
        window_tiles: int,
        max_windows: int,
        flag_count: int,
    ) -> None:
        self.max_seqs = int(max_seqs)
        self.tiles_capacity = int(tiles_capacity)
        self.window_tiles = int(window_tiles)
        self.max_windows = int(max_windows)
        self.flag_count = int(flag_count)
        self.seq_block = (self.max_seqs + _PROLOGUE_THREADS - 1) // _PROLOGUE_THREADS
        # Band tables have tiles_capacity + 2 entries (bands 0..tiles_capacity
        # plus the total).
        self.band_entries = self.tiles_capacity + 2
        self.band_block = (self.band_entries + _PROLOGUE_THREADS - 1) // _PROLOGUE_THREADS

    @cute.jit
    def __call__(
        self,
        cu_seqlens: cute.Pointer,
        num_seqs: cute.Pointer,
        band_base: cute.Pointer,
        sorted_seq: cute.Pointer,
        rank_of: cute.Pointer,
        pos_seq: cute.Pointer,
        pos_local: cute.Pointer,
        window_table: cute.Pointer,
        ready: cute.Pointer,
        seq_capacity: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(
            cu_seqlens, num_seqs, band_base, sorted_seq, rank_of, pos_seq, pos_local,
            window_table, ready, seq_capacity,
        ).launch(grid=(1, 1, 1), block=(_PROLOGUE_THREADS, 1, 1), stream=stream)

    @cute.jit
    def _exclusive_scan(
        self, values: cute.Tensor, block_sums: cute.Tensor, thread: Int32, length: Int32
    ) -> Int32:
        """In-place exclusive prefix scan of ``values[0:length]``; returns the total.

        Each thread scans one contiguous block of ``band_block`` entries, the
        block totals are scanned across the CTA, and the block offsets are
        applied. The caller must synchronize before reading the result.
        """
        running = Int32(0)
        for item in cutlass.range_constexpr(self.band_block):
            index = thread * Int32(self.band_block) + Int32(item)
            if index < length:
                current = values[index]
                values[index] = running
                running += current
        block_sums[thread] = running
        cute.arch.sync_threads()
        for step in cutlass.range_constexpr(8):
            distance = Int32(1 << step)
            addend = Int32(0)
            if thread >= distance:
                addend = block_sums[thread - distance]
            cute.arch.sync_threads()
            block_sums[thread] = block_sums[thread] + addend
            cute.arch.sync_threads()
        offset = block_sums[thread] - running
        for item in cutlass.range_constexpr(self.band_block):
            index = thread * Int32(self.band_block) + Int32(item)
            if index < length:
                values[index] = values[index] + offset
        total = block_sums[Int32(_PROLOGUE_THREADS - 1)]
        cute.arch.sync_threads()
        return total

    @cute.kernel
    def kernel(
        self,
        cu_seqlens: cute.Pointer,
        num_seqs: cute.Pointer,
        band_base: cute.Pointer,
        sorted_seq: cute.Pointer,
        rank_of: cute.Pointer,
        pos_seq: cute.Pointer,
        pos_local: cute.Pointer,
        window_table: cute.Pointer,
        ready: cute.Pointer,
        seq_capacity: Int32,
    ):
        thread, _, _ = cute.arch.thread_idx()
        thread = Int32(thread)
        allocator = cutlass.utils.SmemAllocator()
        counts = allocator.allocate_tensor(
            element_type=Int32,
            layout=cute.make_layout((self.max_seqs,), stride=(1,)),
            byte_alignment=16,
        )
        # hist[c]: sequences with c tiles; becomes its exclusive prefix.
        hist = allocator.allocate_tensor(
            element_type=Int32,
            layout=cute.make_layout((self.band_entries,), stride=(1,)),
            byte_alignment=16,
        )
        # more[l]: sequences with more than l tiles; bands becomes band_base.
        more = allocator.allocate_tensor(
            element_type=Int32,
            layout=cute.make_layout((self.band_entries,), stride=(1,)),
            byte_alignment=16,
        )
        bands = allocator.allocate_tensor(
            element_type=Int32,
            layout=cute.make_layout((self.band_entries,), stride=(1,)),
            byte_alignment=16,
        )
        cursor = allocator.allocate_tensor(
            element_type=Int32,
            layout=cute.make_layout((self.band_entries,), stride=(1,)),
            byte_alignment=16,
        )
        block_sums = allocator.allocate_tensor(
            element_type=Int32,
            layout=cute.make_layout((_PROLOGUE_THREADS + 1,), stride=(1,)),
            byte_alignment=16,
        )
        # Ready flags of both workspace ring slots start clear every run so a
        # recurrence launch can only match flags published by this run.
        position = thread
        while position < Int32(self.flag_count):
            ready[position] = Int32(0)
            position += Int32(_PROLOGUE_THREADS)
        entry = thread
        while entry < Int32(self.band_entries):
            hist[entry] = Int32(0)
            cursor[entry] = Int32(0)
            entry += Int32(_PROLOGUE_THREADS)
        cute.arch.sync_threads()

        live_seqs = num_seqs[Int32(0)].to(Int32)
        bounded_seqs = cutlass.max(Int32(0), cutlass.min(live_seqs, seq_capacity))

        # Per-sequence pass: tile counts and the tile-count histogram.
        seq = thread
        while seq < Int32(self.max_seqs):
            count = Int32(0)
            if seq < bounded_seqs:
                start = cu_seqlens[seq].to(Int32)
                end = cu_seqlens[seq + Int32(1)].to(Int32)
                length = cutlass.max(Int32(0), end - start)
                count = cutlass.min((length + Int32(_CHUNK - 1)) // Int32(_CHUNK), Int32(self.tiles_capacity))
                cute.arch.atomic_add(hist.iterator + count, Int32(1))
            counts[seq] = count
            seq += Int32(_PROLOGUE_THREADS)
        cute.arch.sync_threads()

        # more[l] = sequences with more than l tiles = live - prefix(hist)[l + 1].
        self._exclusive_scan(hist, block_sums, thread, Int32(self.tiles_capacity + 1))
        entry = thread
        while entry < Int32(self.band_entries):
            value = Int32(0)
            if entry < Int32(self.tiles_capacity):
                value = bounded_seqs - hist[entry + Int32(1)]
            more[entry] = value
            bands[entry] = value
            entry += Int32(_PROLOGUE_THREADS)
        cute.arch.sync_threads()
        # bands[l] = first position of band l; bands[tiles_capacity + 1] = total.
        total_tiles = self._exclusive_scan(bands, block_sums, thread, Int32(self.band_entries))
        entry = thread
        while entry < Int32(self.band_entries):
            band_base[entry] = bands[entry]
            entry += Int32(_PROLOGUE_THREADS)

        # Ranks: sequences with more tiles first, ties in arrival order.
        seq = thread
        while seq < bounded_seqs:
            count = counts[seq]
            rank = more[count] + cute.arch.atomic_add(cursor.iterator + count, Int32(1))
            rank_of[seq] = rank
            sorted_seq[rank] = seq
            seq += Int32(_PROLOGUE_THREADS)
        cute.arch.sync_threads()

        # Per window: the band holding its first position and that position's
        # rank within the band.
        window = thread
        while window < Int32(self.max_windows):
            window_begin = window * Int32(self.window_tiles)
            low = Int32(0)
            high = Int32(self.tiles_capacity)
            while high - low > Int32(1):
                mid = (low + high) >> Int32(1)
                if bands[mid] <= window_begin:
                    low = mid
                else:
                    high = mid
            window_table[window * Int32(2)] = low
            window_table[window * Int32(2) + Int32(1)] = window_begin - bands[low]
            window += Int32(_PROLOGUE_THREADS)
        # Position tables (binary search of the band) and the unused tail.
        bounded_tiles = cutlass.min(total_tiles, Int32(self.tiles_capacity))
        tile = thread
        while tile < bounded_tiles:
            low = Int32(0)
            high = Int32(self.tiles_capacity)
            while high - low > Int32(1):
                mid = (low + high) >> Int32(1)
                if bands[mid] <= tile:
                    low = mid
                else:
                    high = mid
            pos_seq[tile] = sorted_seq[tile - bands[low]]
            pos_local[tile] = low
            tile += Int32(_PROLOGUE_THREADS)
        tile = bounded_tiles + thread
        while tile < Int32(self.tiles_capacity):
            pos_seq[tile] = Int32(-1)
            tile += Int32(_PROLOGUE_THREADS)


class _PrepareKernel:
    """Per (tile, head): gates, norms, decayed operands, WY inverse, Mqk."""

    def __init__(
        self,
        *,
        heads: int,
        key_heads: int,
        is_gdn: bool,
        tiles_capacity: int,
        window_tiles: int,
        qk_l2norm: bool,
        a_log_type: type[cutlass.Numeric],
        dt_bias_type: type[cutlass.Numeric],
    ) -> None:
        self.heads = int(heads)
        self.head_ratio = int(heads) // int(key_heads)
        self.is_gdn = bool(is_gdn)
        self.tiles_capacity = int(tiles_capacity)
        self.window_tiles = int(window_tiles)
        self.qk_l2norm = bool(qk_l2norm)
        self.a_log_type = a_log_type
        self.dt_bias_type = dt_bias_type

    @cute.jit
    def __call__(
        self,
        q: cute.Pointer,
        k: cute.Pointer,
        raw_g: cute.Pointer,
        raw_beta: cute.Pointer,
        A_log: cute.Pointer,
        dt_bias: cute.Pointer,
        cu_seqlens: cute.Pointer,
        pos_seq: cute.Pointer,
        pos_local: cute.Pointer,
        ready: cute.Pointer,
        ws_bf16: cute.Pointer,
        ws_f32: cute.Pointer,
        q_stride: Int64,
        k_stride: Int64,
        g_stride: Int64,
        beta_token_stride: Int64,
        beta_head_stride: Int64,
        scale: Float32,
        gate_scale: Float32,
        eps: Float32,
        window: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(
            q, k, raw_g, raw_beta, A_log, dt_bias, cu_seqlens, pos_seq, pos_local,
            ready, ws_bf16, ws_f32, q_stride, k_stride, g_stride,
            beta_token_stride, beta_head_stride, scale, gate_scale, eps, window,
        ).launch(
            grid=(self.window_tiles, self.heads, 1),
            block=(_PREPARE_THREADS, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        q: cute.Pointer,
        k: cute.Pointer,
        raw_g: cute.Pointer,
        raw_beta: cute.Pointer,
        A_log: cute.Pointer,
        dt_bias: cute.Pointer,
        cu_seqlens: cute.Pointer,
        pos_seq: cute.Pointer,
        pos_local: cute.Pointer,
        ready: cute.Pointer,
        ws_bf16: cute.Pointer,
        ws_f32: cute.Pointer,
        q_stride: Int64,
        k_stride: Int64,
        g_stride: Int64,
        beta_token_stride: Int64,
        beta_head_stride: Int64,
        scale: Float32,
        gate_scale: Float32,
        eps: Float32,
        window: Int32,
    ):
        local_tile, head, _ = cute.arch.block_idx()
        thread, _, _ = cute.arch.thread_idx()
        local_tile = Int32(local_tile)
        tile = window * Int32(self.window_tiles) + local_tile
        head = Int32(head)
        column = Int32(thread)
        warp = column // Int32(32)
        lane = Int32(cute.arch.lane_idx())
        seq = Int32(-1)
        local = Int32(0)
        if tile < Int32(self.tiles_capacity):
            seq = pos_seq[tile].to(Int32)
            local = pos_local[tile].to(Int32)
        if seq >= Int32(0):
            allocator = cutlass.utils.SmemAllocator()
            tile_elements = _CHUNK * _HEAD_DIM
            s_part = allocator.allocate_tensor(
                element_type=Float32,
                layout=cute.make_layout((2 * _CHUNK,), stride=(1,)),
                byte_alignment=16,
            )
            s_beta = allocator.allocate_tensor(
                element_type=Float32,
                layout=cute.make_layout((_CHUNK,), stride=(1,)),
                byte_alignment=16,
            )
            if cutlass.const_expr(self.is_gdn):
                s_gate = allocator.allocate_tensor(
                    element_type=Float32,
                    layout=cute.make_layout((2 * _CHUNK,), stride=(1,)),
                    byte_alignment=16,
                )
                s_factors = allocator.allocate_tensor(
                    element_type=Float32,
                    layout=cute.make_layout((4 * _CHUNK,), stride=(1,)),
                    byte_alignment=16,
                )
            # Raw q, k, g rows. Once the gate loop has consumed them, their
            # regions hold the swizzled q~ and k~ tiles and the transposed
            # k_inv tile for the tensor-core products.
            s_raw = allocator.allocate_tensor(
                element_type=BFloat16,
                layout=cute.make_layout(((2 if self.is_gdn else 3) * tile_elements,), stride=(1,)),
                byte_alignment=128,
            )
            s_squares = allocator.allocate_tensor(
                element_type=Float32,
                layout=cute.make_layout((4 * _CHUNK * _CHUNK,), stride=(1,)),
                byte_alignment=16,
            )
            square_layout = cute.make_layout((_CHUNK * _CHUNK,), stride=(1,))
            squares = s_squares.iterator
            s_p = cute.make_tensor(squares, square_layout)
            s_p2 = cute.make_tensor(squares + _CHUNK * _CHUNK, square_layout)
            s_inv = cute.make_tensor(squares + 2 * _CHUNK * _CHUNK, square_layout)
            s_inv2 = cute.make_tensor(squares + 3 * _CHUNK * _CHUNK, square_layout)

            start = cu_seqlens[seq].to(Int32) + local * Int32(_CHUNK)
            end = cu_seqlens[seq + Int32(1)].to(Int32)
            rows = cutlass.min(Int32(_CHUNK), end - start)
            head_elements = (head // Int32(self.head_ratio)).to(Int64) * Int64(_HEAD_DIM)

            # Stage the raw rows: 16 rows x 16 chunks of 16 bytes per tensor,
            # zero-filled past the sequence tail.
            raw_addr = shared_ptr_to_u32(s_raw.iterator)
            for item in cutlass.range_constexpr(4 if self.is_gdn else 6):
                chunk = column + Int32(item * _PREPARE_THREADS)
                tensor_index = chunk // Int32(256)
                local_chunk = chunk % Int32(256)
                row = local_chunk // Int32(16)
                col_chunk = local_chunk % Int32(16)
                live_row = cutlass.min(row, cutlass.max(rows - Int32(1), Int32(0)))
                token = (start + live_row).to(Int64)
                src_bytes = Int32(16)
                if row >= rows:
                    src_bytes = Int32(0)
                col_elements = (col_chunk * Int32(8)).to(Int64)
                target_chunk = chunk
                if cutlass.const_expr(self.is_gdn):
                    target_chunk = tensor_index * Int32(256) + row * Int32(16) + (col_chunk ^ (row & Int32(7)))
                if tensor_index == Int32(0):
                    _cp_async_16_zfill(
                        raw_addr + target_chunk * Int32(16),
                        _pointer_address(q, token * q_stride + head_elements + col_elements),
                        src_bytes,
                    )
                elif tensor_index == Int32(1):
                    _cp_async_16_zfill(
                        raw_addr + target_chunk * Int32(16),
                        _pointer_address(k, token * k_stride + head_elements + col_elements),
                        src_bytes,
                    )
                elif cutlass.const_expr(not self.is_gdn):
                    _cp_async_16_zfill(
                        raw_addr + chunk * Int32(16),
                        _pointer_address(raw_g, token * g_stride + head_elements + col_elements),
                        src_bytes,
                    )
            cute.arch.cp_async_commit_group()
            # Parameter and beta loads overlap the staging copies.
            rate = cute.math.exp(Float32(A_log[head]), fastmath=False)
            if cutlass.const_expr(self.is_gdn):
                bias = Float32(dt_bias[head])
                if column < Int32(_CHUNK):
                    log_decay = Float32(0.0)
                    if column < rows:
                        offset = (start + column).to(Int64) * g_stride + head.to(Int64)
                        a = Float32(raw_g[offset]) + bias
                        softplus = a
                        if a <= Float32(20.0):
                            softplus = cute.math.log1p(
                                cute.math.exp(a, fastmath=False),
                                fastmath=False,
                            )
                        log_decay = -rate * softplus * Float32(_LOG2E)
                    s_gate[column] = log_decay
            else:
                bias = Float32(dt_bias[head * Int32(_HEAD_DIM) + column])
            beta_raw = Float32(0.0)
            if column < rows:
                beta_offset = (
                    (start + column).to(Int64) * beta_token_stride
                    + head.to(Int64) * beta_head_stride
                )
                beta_raw = Float32(raw_beta[beta_offset])
            cute.arch.cp_async_wait_group(0)
            cute.arch.sync_threads()
            # Row sums of squares: eight lanes per row, sixteen strided elements each.
            sum_row = column >> Int32(3)
            sum_part = column & Int32(7)
            q_sq = Float32(0.0)
            k_sq = Float32(0.0)
            for item in cutlass.range_constexpr(_HEAD_DIM // 8):
                element = sum_row * Int32(_HEAD_DIM) + Int32(item * 8) + sum_part
                if cutlass.const_expr(self.is_gdn):
                    element = sum_row * Int32(_HEAD_DIM) + (Int32(item) ^ (sum_row & Int32(7))) * Int32(8) + sum_part
                q_value = Float32(s_raw[element])
                k_value = Float32(s_raw[Int32(tile_elements) + element])
                q_sq += q_value * q_value
                k_sq += k_value * k_value
            q_sq = warp_reduce(q_sq, _add, 8)
            k_sq = warp_reduce(k_sq, _add, 8)
            if sum_part == Int32(0):
                s_part[sum_row] = q_sq
                s_part[Int32(_CHUNK) + sum_row] = k_sq
                if cutlass.const_expr(self.is_gdn):
                    rinv_q, rinv_k = Float32(1.0), Float32(1.0)
                    if cutlass.const_expr(self.qk_l2norm):
                        rinv_q = cute.math.rsqrt(q_sq + eps, fastmath=False)
                        rinv_k = cute.math.rsqrt(k_sq + eps, fastmath=False)
                    prefix, suffix = Float32(0.0), Float32(0.0)
                    for t in cutlass.range_constexpr(_CHUNK):
                        if Int32(t) <= sum_row:
                            prefix += s_gate[Int32(t)]
                    for t in cutlass.range_constexpr(_CHUNK - 1, -1, -1):
                        if Int32(t) > sum_row:
                            suffix += s_gate[Int32(t)]
                    s_factors[sum_row] = rinv_q
                    s_factors[Int32(_CHUNK) + sum_row] = rinv_k
                    s_factors[Int32(2 * _CHUNK) + sum_row] = _exp2_approx_ftz_f32(prefix)
                    s_factors[Int32(3 * _CHUNK) + sum_row] = _exp2_approx_ftz_f32(suffix)

            q_values = cute.make_rmem_tensor((_CHUNK,), Float32)
            k_values = cute.make_rmem_tensor((_CHUNK,), Float32)
            if cutlass.const_expr(not self.is_gdn):
                g_cum = cute.make_rmem_tensor((_CHUNK,), Float32)
                running = Float32(0.0)
            for t in cutlass.range_constexpr(_CHUNK):
                q_value = Float32(0.0)
                k_value = Float32(0.0)
                g2 = Float32(0.0)
                if Int32(t) < rows:
                    raw_column = column
                    if cutlass.const_expr(self.is_gdn):
                        raw_column = ((column >> Int32(3)) ^ Int32(t & 7)) * Int32(8) + (column & Int32(7))
                    q_value = Float32(s_raw[Int32(t * _HEAD_DIM) + raw_column])
                    k_value = Float32(s_raw[Int32(tile_elements + t * _HEAD_DIM) + raw_column])
                    if cutlass.const_expr(not self.is_gdn):
                        g_value = Float32(s_raw[Int32(2 * tile_elements + t * _HEAD_DIM) + column])
                        z = rate * (g_value + bias)
                        sigmoid = cute.arch.rcp_approx(
                            Float32(1.0) + _exp2_approx_ftz_f32(-z * Float32(_LOG2E))
                        )
                        g2 = gate_scale * sigmoid
                q_values[t] = q_value
                k_values[t] = k_value
                if cutlass.const_expr(not self.is_gdn):
                    running += g2
                    g_cum[t] = running
            if column < Int32(_CHUNK):
                beta = Float32(0.0)
                if column < rows:
                    beta = cute.arch.rcp_approx(
                        Float32(1.0) + _exp2_approx_ftz_f32(-beta_raw * Float32(_LOG2E))
                    )
                    if cutlass.const_expr(self.is_gdn):
                        beta = Float32(BFloat16(beta))
                s_beta[column] = beta
            cute.arch.sync_threads()

            ring_index = (window & Int32(1)) * Int32(self.window_tiles) + local_tile
            record = ring_index.to(Int64) * Int64(self.heads) + head.to(Int64)
            rec_bf16 = record * Int64(REC.BYTES // 2)
            rec_f32 = record * Int64(REC.BYTES // 4)
            q_base = rec_bf16 + Int64(REC.Q_TILDE // 2)
            k_base = rec_bf16 + Int64(REC.K_TILDE // 2)
            kr_base = rec_bf16 + Int64(REC.K_R // 2)
            if cutlass.const_expr(self.is_gdn):
                lambda_c = s_factors[Int32(3 * _CHUNK - 1)]
            else:
                last = g_cum[_CHUNK - 1]
                lambda_c = _exp2_approx_ftz_f32(last)
            ws_f32[rec_f32 + Int64(REC.LAMBDA_C // 4) + column.to(Int64)] = lambda_c
            if cutlass.const_expr(self.is_gdn):
                beta_value = s_beta[column % Int32(_CHUNK)]
                summary_finite = ((lambda_c > Float32(float("-inf")))
                                  & (lambda_c < Float32(float("inf")))
                                  & (beta_value > Float32(float("-inf")))
                                  & (beta_value < Float32(float("inf"))))
            if column < Int32(_CHUNK):
                ws_f32[rec_f32 + Int64(REC.BETA // 4) + column.to(Int64)] = s_beta[column]
            for t in cutlass.range_constexpr(_CHUNK):
                if cutlass.const_expr(self.is_gdn):
                    rinv_q = s_factors[Int32(t)]
                    rinv_k = s_factors[Int32(_CHUNK + t)]
                    lam = s_factors[Int32(2 * _CHUNK + t)]
                    lam_r = s_factors[Int32(3 * _CHUNK + t)]
                else:
                    rinv_q, rinv_k = Float32(1.0), Float32(1.0)
                    if cutlass.const_expr(self.qk_l2norm):
                        rinv_q = cute.math.rsqrt(s_part[Int32(t)] + eps, fastmath=False)
                        rinv_k = cute.math.rsqrt(s_part[Int32(_CHUNK + t)] + eps, fastmath=False)
                    lam = _exp2_approx_ftz_f32(g_cum[t])
                    lam_inv = _exp2_approx_ftz_f32(-g_cum[t])
                    lam_r = _exp2_approx_ftz_f32(last - g_cum[t])
                    k_inv = BFloat16(k_values[t] * rinv_k * lam_inv)
                q_tilde = BFloat16(q_values[t] * rinv_q * lam * scale)
                k_tilde = BFloat16(k_values[t] * rinv_k * lam)
                k_r = BFloat16(k_values[t] * rinv_k * lam_r)
                if cutlass.const_expr(self.is_gdn):
                    summary_finite = (summary_finite
                                      & (Float32(k_tilde) > Float32(float("-inf")))
                                      & (Float32(k_tilde) < Float32(float("inf")))
                                      & (Float32(k_r) > Float32(float("-inf")))
                                      & (Float32(k_r) < Float32(float("inf"))))
                physical = (
                    Int32(t * _HEAD_DIM)
                    + ((((column >> Int32(3)) ^ Int32(t & 7)) << Int32(3)) | (column & Int32(7)))
                )
                if cutlass.const_expr(self.is_gdn):
                    s_raw[physical] = BFloat16(q_values[t] * rinv_q)
                    s_raw[Int32(tile_elements) + physical] = BFloat16(k_values[t] * rinv_k)
                else:
                    s_raw[physical] = q_tilde
                    s_raw[Int32(tile_elements) + physical] = k_tilde
                if cutlass.const_expr(not self.is_gdn):
                    s_raw[Int32(2 * tile_elements) + column * Int32(_CHUNK) + Int32(t)] = k_inv
                ws_bf16[q_base + physical.to(Int64)] = q_tilde
                ws_bf16[k_base + physical.to(Int64)] = k_tilde
                ws_bf16[kr_base + physical.to(Int64)] = k_r
            cute.arch.sync_threads()

            # L = beta_i <k~_i, k_inv_j> (j < i) on warp 0 and Mqk = <q~_i, k_inv_j>
            # (j <= i) on warp 1, each one m16n16k128 tensor-core product.
            if warp < Int32(2):
                a_base = raw_addr + Int32(tile_elements * 2)
                if warp == Int32(1):
                    a_base = raw_addr
                b_base = raw_addr + Int32(2 * tile_elements * 2)
                gid = lane >> Int32(2)
                tid = lane & Int32(3)
                matrix = lane >> Int32(3)
                matrix_row = lane & Int32(7)
                a_row = (matrix & Int32(1)) * Int32(8) + matrix_row
                prod = cute.make_rmem_tensor((2, 4), Float32)
                for half in cutlass.range_constexpr(2):
                    for item in cutlass.range_constexpr(4):
                        prod[half, item] = Float32(0.0)
                for kb in cutlass.range_constexpr(8):
                    a_chunk = (Int32(kb * 2) + (matrix >> Int32(1))) ^ (a_row & Int32(7))
                    a0, a1, a2, a3 = ldmatrix_m8n8x4_b16(a_base + a_row * Int32(256) + a_chunk * Int32(16))
                    if cutlass.const_expr(self.is_gdn):
                        b_row = (matrix >> Int32(1)) * Int32(8) + matrix_row
                        b_chunk = (Int32(kb * 2) + (matrix & Int32(1))) ^ (b_row & Int32(7))
                        b0, b1, b2, b3 = ldmatrix_m8n8x4_b16(
                            raw_addr + Int32(tile_elements * 2) + b_row * Int32(256) + b_chunk * Int32(16)
                        )
                    else:
                        b_row = Int32(kb * 16) + (matrix & Int32(1)) * Int32(8) + matrix_row
                        b0, b1, b2, b3 = ldmatrix_m8n8x4_trans_b16(
                            b_base + b_row * Int32(32) + (matrix >> Int32(1)) * Int32(16)
                        )
                    prod[0, 0], prod[0, 1], prod[0, 2], prod[0, 3] = bf16_mma_m16n8k16_f32(
                        prod[0, 0], prod[0, 1], prod[0, 2], prod[0, 3], a0, a1, a2, a3, b0, b1
                    )
                    prod[1, 0], prod[1, 1], prod[1, 2], prod[1, 3] = bf16_mma_m16n8k16_f32(
                        prod[1, 0], prod[1, 1], prod[1, 2], prod[1, 3], a0, a1, a2, a3, b2, b3
                    )
                for half in cutlass.range_constexpr(2):
                    for item in cutlass.range_constexpr(4):
                        row_i = gid + Int32((item >> 1) * 8)
                        col_j = Int32(half * 8) + tid * Int32(2) + Int32(item & 1)
                        index = row_i * Int32(_CHUNK) + col_j
                        value = prod[half, item]
                        if cutlass.const_expr(self.is_gdn):
                            # Sum intervals directly: strong earlier decay must not
                            # erase a later gate through prefix subtraction.
                            log_interval = Float32(0.0)
                            for t in cutlass.range_constexpr(_CHUNK):
                                if (Int32(t) > col_j) & (Int32(t) <= row_i):
                                    log_interval += s_gate[Int32(t)]
                            value *= _exp2_approx_ftz_f32(log_interval)
                            if warp == Int32(1):
                                value *= scale
                        if warp == Int32(0):
                            lower = Float32(0.0)
                            if col_j < row_i:
                                lower = s_beta[row_i] * value
                            identity = Float32(0.0)
                            if col_j == row_i:
                                identity = Float32(1.0)
                            s_p[index] = lower
                            s_inv[index] = identity - lower
                        else:
                            mqk = Float32(0.0)
                            if col_j <= row_i:
                                mqk = value
                            ws_bf16[rec_bf16 + Int64(REC.MQK // 2) + index.to(Int64)] = BFloat16(mqk)
            cute.arch.sync_threads()

            # Neumann series: INV = (I - L)(I + L^2)(I + L^4)(I + L^8).
            for _step in cutlass.range_constexpr(3):
                for entry in cutlass.range_constexpr(2):
                    index = column + Int32(entry * _PREPARE_THREADS)
                    row = index // Int32(_CHUNK)
                    col = index % Int32(_CHUNK)
                    acc = Float32(0.0)
                    for j in cutlass.range_constexpr(_CHUNK):
                        acc += s_p[row * Int32(_CHUNK) + Int32(j)] * s_p[Int32(j * _CHUNK) + col]
                    s_p2[index] = acc
                cute.arch.sync_threads()
                for entry in cutlass.range_constexpr(2):
                    index = column + Int32(entry * _PREPARE_THREADS)
                    row = index // Int32(_CHUNK)
                    col = index % Int32(_CHUNK)
                    acc = s_inv[index]
                    for j in cutlass.range_constexpr(_CHUNK):
                        acc += s_inv[row * Int32(_CHUNK) + Int32(j)] * s_p2[Int32(j * _CHUNK) + col]
                    s_inv2[index] = acc
                cute.arch.sync_threads()
                for entry in cutlass.range_constexpr(2):
                    index = column + Int32(entry * _PREPARE_THREADS)
                    s_p[index] = s_p2[index]
                    s_inv[index] = s_inv2[index]
                cute.arch.sync_threads()
            for entry in cutlass.range_constexpr(2):
                index = column + Int32(entry * _PREPARE_THREADS)
                inverse = BFloat16(s_inv[index])
                ws_bf16[rec_bf16 + Int64(REC.INV // 2) + index.to(Int64)] = inverse
                if cutlass.const_expr(self.is_gdn):
                    summary_finite = (summary_finite
                                      & (Float32(inverse) > Float32(float("-inf")))
                                      & (Float32(inverse) < Float32(float("inf"))))
            if cutlass.const_expr(self.is_gdn):
                warp_finite = cute.arch.vote_all_sync(summary_finite)
                if lane == Int32(0):
                    s_gate[Int32(_CHUNK) + warp] = warp_finite.to(Float32)
            # Publish the record: every thread's stores are ordered before the
            # barrier, and the fence makes them visible at GPU scope before
            # the release store of the flag.
            cute.arch.sync_threads()
            if column == Int32(0):
                if cutlass.const_expr(self.is_gdn):
                    all_finite = Float32(1.0)
                    for w in cutlass.range_constexpr(_PREPARE_THREADS // 32):
                        all_finite *= s_gate[Int32(_CHUNK + w)]
                    ws_f32[rec_f32 + Int64(REC.SUMMARY_FINITE // 4)] = all_finite
                cute.arch.fence_acq_rel_gpu()
                _st_release_gpu_i32(_pointer_address(ready, record), window + Int32(1))


class _RecurrenceKernel:
    """Per (window, sequence, head, value split): walk the tiles with the state in registers.

    A CTA holds ``v_split`` rows of the transposed state for one head. Its MMA
    warps form ``v_split // 16`` row groups of ``k_split`` warps; each warp owns
    ``128 // k_split`` key columns of its group's sixteen rows as fp32 m16n8
    accumulator fragments plus a bf16 shadow packed as k16 A fragments. Every
    per-tile product reuses the previous accumulator as the next A operand; the
    two products that contract over the key axis are reduced across the
    group's warps through shared memory when ``k_split > 1``. One producer warp
    streams the prepared tiles through a ``stages``-deep shared-memory ring
    with bulk async copies and mbarriers, so the MMA warps never wait on loads
    or on a CTA-wide barrier inside the tile loop.

    One launch covers one window of ``window_tiles`` consecutive banded tile
    positions (see the prologue), so every live sequence advances in every
    window. CTA row ``by`` maps to one sequence rank with positions in the
    window; a sequence that began in an earlier window resumes from its final
    state slot, and one that continues past the window leaves its running
    state there. Empty sequences copy initial to final in window 0. The
    producer polls the per-tile ready flags of the prepare kernel, so the two
    kernels of a window overlap.

    Parallel modes are 1: local state, 2: transfer matrix, 3: both summaries,
    4: local state and output with a convergence checkpoint, and 5: output
    correction from the propagated incoming state. Mode 5 reuses a suffix
    only after every FP32 state bit in that value group matches mode 4.
    """

    def __init__(
        self,
        *,
        heads: int,
        tiles_capacity: int,
        window_tiles: int,
        rows: int,
        v_split: int,
        k_split: int,
        stages: int,
        checkpoint_export: bool,
        null_state_index: int | None,
        index_type: type[cutlass.Numeric],
        summary_mode: int = 0,
        max_sequence_tiles: int = 0,
        is_gdn: bool = False,
    ) -> None:
        self.heads = int(heads)
        self.tiles_capacity = int(tiles_capacity)
        self.window_tiles = int(window_tiles)
        self.rows = int(rows)
        self.band_tiles = min(int(window_tiles), int(max_sequence_tiles) or int(window_tiles))
        self.v_split = int(v_split)
        self.k_split = int(k_split)
        self.stages = int(stages)
        if self.stages < 2:
            raise ValueError("the recurrence pipeline needs at least two stages")
        if self.k_split not in (1, 2, 4):
            raise ValueError("k_split must be 1, 2, or 4")
        self.splits = (_HEAD_DIM // self.v_split) * (2 if summary_mode == 3 else 1)
        self.row_groups = self.v_split // 16
        self.mma_warps = self.row_groups * self.k_split
        self.mma_threads = 32 * self.mma_warps
        self.threads = self.mma_threads + 32
        self.cols = _HEAD_DIM // self.k_split
        self.kb_steps = self.cols // 16
        self.nb_blocks = self.cols // 8
        self.checkpoint_export = bool(checkpoint_export)
        self.has_null = null_state_index is not None
        self.null_state_index = 0 if null_state_index is None else int(null_state_index)
        self.index_type = index_type
        self.v_chunks_per_row = self.v_split // 8
        self.summary_mode = int(summary_mode)
        self.is_gdn = bool(is_gdn)
        self.record_skip = REC.K_TILDE if summary_mode in (1, 2, 3) else 0
        self.min_blocks = 3 if is_gdn and summary_mode in (0, 4, 5) and max_sequence_tiles and (v_split, k_split, stages) == (64, 1, 2) else 0

    @cute.jit
    def __call__(
        self,
        v: cute.Pointer,
        cu_seqlens: cute.Pointer,
        band_base: cute.Pointer,
        sorted_seq: cute.Pointer,
        window_table: cute.Pointer,
        initial_indices: cute.Pointer,
        final_indices: cute.Pointer,
        checkpoint_indices: cute.Pointer,
        checkpoint_offsets: cute.Pointer,
        num_seqs: cute.Pointer,
        ready: cute.Pointer,
        ws: cute.Pointer,
        recurrent_state: cute.Pointer,
        output: cute.Pointer,
        v_stride: Int64,
        out_stride: Int64,
        slot_stride: Int64,
        final_stride: Int64,
        window: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(
            v, cu_seqlens, band_base, sorted_seq, window_table, initial_indices, final_indices,
            checkpoint_indices, checkpoint_offsets, num_seqs, ready, ws,
            recurrent_state, output, v_stride, out_stride, slot_stride, final_stride, window,
        ).launch(
            grid=(self.heads * self.splits, self.rows, 1),
            block=(self.threads, 1, 1),
            min_blocks_per_mp=self.min_blocks,
            stream=stream,
        )

    @cute.jit
    def _is_null(self, slot: Int64) -> cutlass.Boolean:
        result = slot != slot
        if cutlass.const_expr(self.has_null):
            result = slot == Int64(self.null_state_index)
        return result

    @cute.jit
    def _store_state(
        self,
        target: cute.Pointer,
        acc: cute.Tensor,
        base: Int64,
        row0: Int32,
        row1: Int32,
        col_base: Int32,
        tid: Int32,
    ):
        for nb in cutlass.range_constexpr(self.nb_blocks):
            kcol = col_base + Int32(nb * 8) + tid * Int32(2)
            offset0 = base + row0.to(Int64) * Int64(_HEAD_DIM) + kcol.to(Int64)
            offset1 = base + row1.to(Int64) * Int64(_HEAD_DIM) + kcol.to(Int64)
            target[offset0] = acc[nb, 0]
            target[offset0 + Int64(1)] = acc[nb, 1]
            target[offset1] = acc[nb, 2]
            target[offset1 + Int64(1)] = acc[nb, 3]

    @cute.jit
    def _load_state(
        self,
        source: cute.Pointer,
        acc: cute.Tensor,
        base: Int64,
        row0: Int32,
        row1: Int32,
        col_base: Int32,
        tid: Int32,
    ):
        for nb in cutlass.range_constexpr(self.nb_blocks):
            kcol = col_base + Int32(nb * 8) + tid * Int32(2)
            offset0 = base + row0.to(Int64) * Int64(_HEAD_DIM) + kcol.to(Int64)
            offset1 = base + row1.to(Int64) * Int64(_HEAD_DIM) + kcol.to(Int64)
            acc[nb, 0] = Float32(source[offset0])
            acc[nb, 1] = Float32(source[offset0 + Int64(1)])
            acc[nb, 2] = Float32(source[offset1])
            acc[nb, 3] = Float32(source[offset1 + Int64(1)])

    @cute.jit
    def _refresh_shadow(self, acc: cute.Tensor, shadow: cute.Tensor):
        for kb in cutlass.range_constexpr(self.kb_steps):
            shadow[kb, 0] = pack_f32x2_to_bfloat2(acc[2 * kb, 0], acc[2 * kb, 1])
            shadow[kb, 1] = pack_f32x2_to_bfloat2(acc[2 * kb, 2], acc[2 * kb, 3])
            shadow[kb, 2] = pack_f32x2_to_bfloat2(acc[2 * kb + 1, 0], acc[2 * kb + 1, 1])
            shadow[kb, 3] = pack_f32x2_to_bfloat2(acc[2 * kb + 1, 2], acc[2 * kb + 1, 3])

    @cute.jit
    def _issue_tile(
        self,
        ws: cute.Pointer,
        v: cute.Pointer,
        stage_addr: Int32,
        ring_index: Int32,
        head: Int32,
        token_base: Int32,
        rows_live: Int32,
        head_elements: Int64,
        split_elements: Int64,
        v_stride: Int64,
        full_bar_u32: Int32,
        full_bar_ptr: cute.Pointer,
        lane: Int32,
        zero_values: cutlass.Boolean,
    ):
        """Issue the copies of ring record ``ring_index`` and its value rows into a stage.

        Every lane copies its share of the ``[16 x v_split]`` value rows with
        16-byte cp.async (zero-filled past the live rows) into the stage's
        sixteen-column-group layout and arrives on the stage barrier when
        they land; lane 0 then posts the record head's byte count and copies
        it with one bulk copy. The barrier expects the 32 lane arrivals plus
        lane 0's, so the phase cannot complete before every value row landed.
        """
        vd = self.v_split
        record = ring_index.to(Int64) * Int64(self.heads) + head.to(Int64)
        record_base = record * Int64(REC.BYTES)
        chunks_per_row = vd // 8
        v_addr = stage_addr + Int32(REC.V - self.record_skip)
        for item in cutlass.range_constexpr((_CHUNK * chunks_per_row) // 32):
            chunk = lane + Int32(item * 32)
            row = chunk // Int32(chunks_per_row)
            col8 = chunk % Int32(chunks_per_row)
            src_bytes = Int32(16)
            if row >= rows_live:
                src_bytes = Int32(0)
            if cutlass.const_expr(self.summary_mode == 2):
                src_bytes = Int32(0)
            if cutlass.const_expr(self.summary_mode == 3):
                if zero_values:
                    src_bytes = Int32(0)
            live_row = cutlass.min(row, cutlass.max(rows_live - Int32(1), Int32(0)))
            element = (token_base + live_row).to(Int64) * v_stride + head_elements + split_elements + (col8 * Int32(8)).to(Int64)
            dst = v_addr + (col8 >> Int32(1)) * Int32(512) + row * Int32(32) + (col8 & Int32(1)) * Int32(16)
            _cp_async_16_zfill(dst, _pointer_address(v, element), src_bytes)
        _cp_async_mbarrier_arrive_noinc(full_bar_u32)
        if lane == Int32(0):
            cute.arch.fence_acq_rel_cta()
            cute.arch.mbarrier_arrive_and_expect_tx(full_bar_ptr, Int32(REC.HEAD_BYTES - self.record_skip))
            # Orders the async-proxy copy after the acquire of the ready flag.
            cute.arch.fence_proxy("async.global")
            cp_async_bulk_g2s_mbar(
                stage_addr, _pointer_address(ws, record_base + Int64(self.record_skip)),
                Int32(REC.HEAD_BYTES - self.record_skip), full_bar_u32
            )

    @cute.jit
    def _group_reduce(
        self,
        partial: cute.Tensor,
        red_addr: Int32,
        group: Int32,
        kq: Int32,
        lane: Int32,
    ):
        """Sum a [16 x 16] fp32 fragment across the group's warps."""
        slot = (group * Int32(self.k_split) + kq) * Int32(256) + lane * Int32(8)
        mine = red_addr + slot * Int32(4)
        st_shared_v4_f32(mine, partial[0, 0], partial[0, 1], partial[0, 2], partial[0, 3])
        st_shared_v4_f32(mine + Int32(16), partial[1, 0], partial[1, 1], partial[1, 2], partial[1, 3])
        cute.arch.barrier(barrier_id=1, number_of_threads=self.mma_threads)
        for other in cutlass.range_constexpr(self.k_split):
            if Int32(other) != kq:
                theirs = red_addr + ((group * Int32(self.k_split) + Int32(other)) * Int32(256) + lane * Int32(8)) * Int32(4)
                a0, a1, a2, a3 = ld_shared_v4_f32(theirs)
                b0, b1, b2, b3 = ld_shared_v4_f32(theirs + Int32(16))
                partial[0, 0] += a0
                partial[0, 1] += a1
                partial[0, 2] += a2
                partial[0, 3] += a3
                partial[1, 0] += b0
                partial[1, 1] += b1
                partial[1, 2] += b2
                partial[1, 3] += b3

    @cute.kernel
    def kernel(
        self,
        v: cute.Pointer,
        cu_seqlens: cute.Pointer,
        band_base: cute.Pointer,
        sorted_seq: cute.Pointer,
        window_table: cute.Pointer,
        initial_indices: cute.Pointer,
        final_indices: cute.Pointer,
        checkpoint_indices: cute.Pointer,
        checkpoint_offsets: cute.Pointer,
        num_seqs: cute.Pointer,
        ready: cute.Pointer,
        ws: cute.Pointer,
        recurrent_state: cute.Pointer,
        output: cute.Pointer,
        v_stride: Int64,
        out_stride: Int64,
        slot_stride: Int64,
        final_stride: Int64,
        window: Int32,
    ):
        bx, by, _ = cute.arch.block_idx()
        thread, _, _ = cute.arch.thread_idx()
        thread = Int32(thread)
        head = Int32(bx) // Int32(self.splits)
        split = Int32(bx) % Int32(self.splits)
        summary_local = cutlass.Boolean(False)
        if cutlass.const_expr(self.summary_mode == 3):
            summary_local = split >= Int32(self.splits // 2)
            split = split % Int32(self.splits // 2)
        warp = thread // Int32(32)
        lane = Int32(cute.arch.lane_idx())
        gid = lane >> Int32(2)
        tid = lane & Int32(3)
        matrix = lane >> Int32(3)
        matrix_row = lane & Int32(7)
        is_producer = warp == Int32(self.mma_warps)
        group = cutlass.min(warp, Int32(self.mma_warps - 1)) // Int32(self.k_split)
        kq = cutlass.min(warp, Int32(self.mma_warps - 1)) % Int32(self.k_split)
        col_base = kq * Int32(self.cols)
        chunk_base = kq * Int32(self.nb_blocks)
        group_lead = kq == Int32(0)
        if cutlass.const_expr(self.k_split == 1):
            group_lead = cutlass.Boolean(True)
        live_seqs = num_seqs[Int32(0)].to(Int32)
        vd = self.v_split
        window_begin = window * Int32(self.window_tiles)
        window_end = window_begin + Int32(self.window_tiles)
        total_tiles = band_base[Int32(self.tiles_capacity + 1)].to(Int32)
        tiled_seqs = band_base[Int32(1)].to(Int32)
        # The band holding the window's first position and the rank of
        # that position within it (from the prologue's window table).
        # Rows first cover the ranks from there to the end of that band,
        # then the ranks before it, which reach the window in the next
        # band.
        band_a = window_table[window * Int32(2)].to(Int32)
        first_rank = window_table[window * Int32(2) + Int32(1)].to(Int32)
        seg1 = band_base[band_a + Int32(1)].to(Int32) - window_begin
        if window_begin >= total_tiles:
            first_rank = Int32(0)
            seg1 = Int32(0)
        rank = Int32(-1)
        l_begin = Int32(0)
        if Int32(by) < seg1:
            rank = first_rank + Int32(by)
            l_begin = band_a
        elif Int32(by) - seg1 < first_rank:
            rank = Int32(by) - seg1
            l_begin = band_a + Int32(1)
        seq = Int32(0)
        start = Int32(0)
        end = Int32(0)
        tiles_s = Int32(0)
        l_end = Int32(0)
        if rank >= Int32(0):
            seq = sorted_seq[rank].to(Int32)
            start = cu_seqlens[seq].to(Int32)
            end = cu_seqlens[seq + Int32(1)].to(Int32)
            tiles_s = cutlass.min(
                (cutlass.max(Int32(0), end - start) + Int32(_CHUNK - 1)) // Int32(_CHUNK),
                Int32(self.tiles_capacity),
            )
            l_begin = cutlass.min(l_begin, tiles_s)
        stage_bytes = REC.V - self.record_skip + vd * 32
        allocator = cutlass.utils.SmemAllocator()
        # Band positions of this CTA's local tiles l_begin .. l_begin + WT.
        s_band = allocator.allocate_tensor(
            element_type=Int32,
            layout=cute.make_layout((self.band_tiles + 1,), stride=(1,)),
            byte_alignment=16,
        )
        s_stage = allocator.allocate_tensor(
            element_type=cutlass.Int8,
            layout=cute.make_layout((self.stages * stage_bytes,), stride=(1,)),
            byte_alignment=128,
        )
        out_addr = Int32(0)
        if cutlass.const_expr(not self.is_gdn):
            s_out = allocator.allocate_tensor(
                element_type=BFloat16,
                layout=cute.make_layout((_CHUNK * vd,), stride=(1,)),
                byte_alignment=128,
            )
            out_addr = shared_ptr_to_u32(s_out.iterator)
        # Cross-warp reduction buffers, only with a key split.
        s_red = allocator.allocate_tensor(
            element_type=Float32,
            layout=cute.make_layout((2 * self.mma_warps * 256 if self.k_split > 1 else 4,), stride=(1,)),
            byte_alignment=128,
        )
        mbar = allocator.allocate_tensor(
            element_type=Int64,
            layout=cute.make_layout((2 * self.stages,), stride=(1,)),
            byte_alignment=8,
        )
        stage_base = shared_ptr_to_u32(s_stage.iterator)
        red_base = shared_ptr_to_u32(s_red.iterator)
        red_a = red_base
        red_c = red_base + Int32(self.mma_warps * 256 * 4)
        full_bar = mbar.iterator
        empty_bar = mbar.iterator + self.stages
        finished_addr = Int32(0)
        if cutlass.const_expr(self.summary_mode in (2, 3, 5) and self.k_split == 1):
            finished_count = allocator.allocate_tensor(
                element_type=Int32, layout=cute.make_layout((1,), stride=(1,)), byte_alignment=4)
            finished_addr = shared_ptr_to_u32(finished_count.iterator)
            if thread == Int32(0):
                finished_count[Int32(0)] = Int32(0)
        if thread == Int32(0):
            for stage_index in cutlass.range_constexpr(self.stages):
                cute.arch.mbarrier_init(full_bar + stage_index, Int32(33))
                cute.arch.mbarrier_init(empty_bar + stage_index, Int32(self.mma_warps))
        if rank >= Int32(0):
            entry = thread
            while entry < Int32(self.band_tiles + 1):
                band = cutlass.min(l_begin + entry, Int32(self.tiles_capacity + 1))
                s_band[entry] = band_base[band].to(Int32)
                entry += Int32(self.threads)
        cute.arch.sync_threads()
        # Last local tile of this window: the first band whose position
        # reaches the window end (at most WT bands lie in a window).
        l_end = l_begin
        if rank >= Int32(0):
            # Smallest staged entry whose position reaches the window end;
            # entries past the sequence's tiles are excluded by the clamp.
            span = cutlass.min(tiles_s - l_begin, Int32(self.band_tiles))
            low = Int32(0)
            high = span
            while high > low:
                mid = (low + high) >> Int32(1)
                if s_band[mid] + rank >= window_end:
                    high = mid
                else:
                    low = mid + Int32(1)
            l_end = l_begin + low
        has_tiles = l_end > l_begin
        if cutlass.const_expr(self.summary_mode != 0):
            has_tiles = has_tiles & (final_indices[seq] != Int32(0))

        head_elements = head.to(Int64) * Int64(_HEAD_DIM)
        split_elements = (split * Int32(vd)).to(Int64)
        v_chunks = self.v_chunks_per_row
        ring_base = (window & Int32(1)) * Int32(self.window_tiles) - window_begin + rank
        expected_flag = window + Int32(1)
        head_base = head.to(Int64) * Int64(_HEAD_DIM * _HEAD_DIM)

        if is_producer:
            # Producer: refill ring stages as they free up, once the
            # prepare kernel has published the tile. Lane 0 waits and
            # polls; every lane then issues its share of the copies.
            prod_phase = Int32(1)
            count = Int32(0)
            if has_tiles:
                can_terminate = cutlass.Boolean(False)
                if cutlass.const_expr(self.summary_mode == 5):
                    can_terminate = cutlass.Boolean(True)
                if cutlass.const_expr(self.summary_mode in (2, 3) and self.k_split == 1):
                    if not summary_local:
                        can_terminate = cutlass.Boolean(True)
                        item = lane
                        while item < l_end - l_begin:
                            record_index = (ring_base + s_band[item]).to(Int64) * Int64(self.heads) + head.to(Int64)
                            finite_address = _pointer_address(ws, record_index * Int64(REC.BYTES) + Int64(REC.SUMMARY_FINITE))
                            can_terminate = can_terminate & (_ld_acquire_gpu_i32(finite_address) == Int32(0x3F800000))
                            item += Int32(32)
                        can_terminate = cute.arch.vote_all_sync(can_terminate)
                # The flag of the next tile is read right after the copies
                # of the current one are issued, so its latency overlaps the
                # wait for a free stage; a stale value falls back to a spin.
                seen = Int32(0)
                if lane == Int32(0):
                    first_flag = _pointer_address(
                        ready, (ring_base + s_band[Int32(0)]).to(Int64) * Int64(self.heads) + head.to(Int64)
                    )
                    seen = _ld_acquire_gpu_i32(first_flag)
                step = Int32(0)
                skip_copy = cutlass.Boolean(False)
                while (step < l_end - l_begin) & (not skip_copy):
                    local = l_begin + step
                    stage = count % Int32(self.stages)
                    ring_index = ring_base + s_band[step]
                    if lane == Int32(0):
                        cute.arch.mbarrier_wait(empty_bar + stage, phase=prod_phase)
                    if lane == Int32(0):
                        flag = _pointer_address(
                            ready, ring_index.to(Int64) * Int64(self.heads) + head.to(Int64)
                        )
                        while seen != expected_flag:
                            _nanosleep(Int32(128))
                            seen = _ld_acquire_gpu_i32(flag)
                    cute.arch.sync_warp()
                    token_base = start + local * Int32(_CHUNK)
                    rows_live = cutlass.min(Int32(_CHUNK), end - token_base)
                    skip_copy = cutlass.Boolean(False)
                    if cutlass.const_expr(self.summary_mode in (2, 3, 5) and self.k_split == 1):
                        if can_terminate:
                            finished_warps = Int32(0)
                            if lane == Int32(0):
                                finished_warps = atomic_add_shared_i32(finished_addr, Int32(0))
                            finished_warps = cute.arch.shuffle_sync(finished_warps, Int32(0))
                            skip_copy = finished_warps == Int32(self.mma_warps)
                    if skip_copy:
                        # Consumers drain issued stages before this termination marker.
                        if lane == Int32(0):
                            st_shared_u32(stage_base + stage * Int32(stage_bytes) + Int32(REC.SUMMARY_FINITE - self.record_skip),
                                          Uint32(0x40000000))
                        cute.arch.sync_warp()
                        cute.arch.mbarrier_arrive(full_bar + stage)
                        if lane == Int32(0):
                            cute.arch.mbarrier_arrive(full_bar + stage)
                    else:
                        self._issue_tile(
                            ws, v, stage_base + stage * Int32(stage_bytes), ring_index, head,
                            token_base, rows_live, head_elements, split_elements, v_stride,
                            shared_ptr_to_u32(full_bar + stage), full_bar + stage, lane,
                            ~summary_local,
                        )
                    if lane == Int32(0):
                        if step + Int32(1) < l_end - l_begin:
                            next_flag = _pointer_address(
                                ready,
                                (ring_base + s_band[step + Int32(1)]).to(Int64) * Int64(self.heads)
                                + head.to(Int64),
                            )
                            seen = _ld_acquire_gpu_i32(next_flag)
                    # Warm L2 with the value rows of a tile a few steps
                    # ahead: one lane per row, one line per v_split * 2 bytes.
                    ahead = local + Int32(_V_PREFETCH_TILES)
                    prefetch_values = cutlass.Boolean(True)
                    if cutlass.const_expr(self.summary_mode == 2):
                        prefetch_values = cutlass.Boolean(False)
                    if cutlass.const_expr(self.summary_mode == 3):
                        prefetch_values = summary_local
                    if (ahead < l_end) & prefetch_values:
                        if lane < Int32(_CHUNK):
                            row_token = start + ahead * Int32(_CHUNK) + lane
                            if row_token < end:
                                line = row_token.to(Int64) * v_stride + head_elements + split_elements
                                for part in cutlass.range_constexpr(max(1, (self.v_split * 2) // 128)):
                                    _prefetch_l2(_pointer_address(v, line + Int64(part * 64)))
                    if stage == Int32(self.stages - 1):
                        prod_phase = prod_phase ^ Int32(1)
                    count += Int32(1)
                    step += Int32(1)
        else:
            # State fragments: rows row0/row1, columns col_base.. of this head.
            row_local0 = group * Int32(16) + gid
            row0 = split * Int32(vd) + row_local0
            row1 = row0 + Int32(8)
            acc = cute.make_rmem_tensor((self.nb_blocks, 4), Float32)
            shadow = cute.make_rmem_tensor((self.kb_steps, 4), Uint32)
            parts = cute.make_rmem_tensor((4, 4), Float32)
            bfrag = cute.make_rmem_tensor((self.kb_steps, 4), Uint32)
            lam = cute.make_rmem_tensor((self.nb_blocks, 2), Float32)
            vp = cute.make_rmem_tensor((2, 4), Float32)
            u = cute.make_rmem_tensor((2, 4), Float32)
            out = cute.make_rmem_tensor((2, 4), Float32)
            cons_phase = Int32(0)
            count = Int32(0)
            if has_tiles:
                initial = Int64(initial_indices[seq])
                zero_state = cutlass.Boolean(False)
                zero_announced = cutlass.Boolean(False)
                converged = cutlass.Boolean(False)
                final = Int64(final_indices[seq.to(Int64) * final_stride])
                if cutlass.const_expr(self.summary_mode == 3):
                    if summary_local:
                        initial = Int64(0)
                        final += Int64(self.rows)
                checkpoint = Int64(checkpoint_indices[seq])
                offset = checkpoint_offsets[seq].to(Int32)
                for nb in cutlass.range_constexpr(self.nb_blocks):
                    acc[nb, 0] = Float32(0.0)
                    acc[nb, 1] = Float32(0.0)
                    acc[nb, 2] = Float32(0.0)
                    acc[nb, 3] = Float32(0.0)
                if l_begin == Int32(0):
                    if not self._is_null(initial):
                        self._load_state(
                            recurrent_state, acc, initial * slot_stride + head_base,
                            row0, row1, col_base, tid,
                        )
                else:
                    # Resume the running state left in the final slot.
                    self._load_state(
                        recurrent_state, acc, final * slot_stride + head_base,
                        row0, row1, col_base, tid,
                    )
                self._refresh_shadow(acc, shadow)
                # The guard is load-bearing: the loop body and its zero
                # trip count are the same without it, but the generated
                # schedule is far worse (the sixteen-head 4096-token case
                # measures 428 us instead of 220 us on RTX PRO 6000).
                if has_tiles:
                    step = Int32(0)
                    stop_summary = cutlass.Boolean(False)
                    while (step < l_end - l_begin) & (not stop_summary):
                        local = l_begin + step
                        stage = count % Int32(self.stages)
                        token_base = start + local * Int32(_CHUNK)
                        rows_live = cutlass.min(Int32(_CHUNK), end - token_base)
                        cute.arch.mbarrier_wait(full_bar + stage, phase=cons_phase)
                        stage_addr = stage_base + stage * Int32(stage_bytes)
                        qt_addr = stage_addr + Int32(REC.Q_TILDE - self.record_skip)
                        kt_addr = stage_addr + Int32(REC.K_TILDE - self.record_skip)
                        kr_addr = stage_addr + Int32(REC.K_R - self.record_skip)
                        inv_addr = stage_addr + Int32(REC.INV - self.record_skip)
                        mqk_addr = stage_addr + Int32(REC.MQK - self.record_skip)
                        lam_addr = stage_addr + Int32(REC.LAMBDA_C - self.record_skip)
                        beta_addr = stage_addr + Int32(REC.BETA - self.record_skip)
                        v_addr = stage_addr + Int32(REC.V - self.record_skip)

                        do_math = cutlass.Boolean(True)
                        stop_summary = cutlass.Boolean(False)
                        if cutlass.const_expr(self.summary_mode == 5):
                            marker = ld_shared_f32(stage_addr + Int32(REC.SUMMARY_FINITE))
                            stop_summary = marker == Float32(2.0)
                            do_math = ~converged
                        if cutlass.const_expr(self.summary_mode in (2, 3) and self.k_split == 1):
                            if not summary_local:
                                finite = ld_shared_f32(stage_addr + Int32(REC.SUMMARY_FINITE - self.record_skip))
                                stop_summary = finite == Float32(2.0)
                                do_math = ~(zero_state & ((finite == Float32(1.0)) | stop_summary))
                        if do_math:
                            zero_state = cutlass.Boolean(False)
                            # Phase A: partial v'^T over this warp's key columns.
                            # Four accumulator chains retain their reduction order.
                            tok_a = (matrix >> Int32(1)) * Int32(8) + matrix_row
                            for chain in cutlass.range_constexpr(4):
                                for item in cutlass.range_constexpr(4):
                                    parts[chain, item] = Float32(0.0)
                            for operand_group in cutlass.range_constexpr(self.kb_steps // 2):
                                for kb in cutlass.range_constexpr(2 * operand_group, 2 * operand_group + 2):
                                    logical_chunk = chunk_base + Int32(kb * 2) + (matrix & Int32(1))
                                    physical = logical_chunk ^ (tok_a & Int32(7))
                                    bfrag[kb % 2, 0], bfrag[kb % 2, 1], bfrag[kb % 2, 2], bfrag[kb % 2, 3] = ldmatrix_m8n8x4_b16(
                                        kt_addr + tok_a * Int32(256) + physical * Int32(16)
                                    )
                                for kb in cutlass.range_constexpr(2 * operand_group, 2 * operand_group + 2):
                                    for half in cutlass.range_constexpr(2):
                                        chain = 2 * (kb % 2) + half
                                        parts[chain, 0], parts[chain, 1], parts[chain, 2], parts[chain, 3] = (
                                            bf16_mma_m16n8k16_f32(
                                                parts[chain, 0], parts[chain, 1], parts[chain, 2], parts[chain, 3],
                                                shadow[kb, 0], shadow[kb, 1], shadow[kb, 2], shadow[kb, 3],
                                                bfrag[kb % 2, 2 * half], bfrag[kb % 2, 2 * half + 1],
                                            )
                                        )
                            for half in cutlass.range_constexpr(2):
                                for item in cutlass.range_constexpr(4):
                                    vp[half, item] = parts[half, item] + parts[2 + half, item]
                            if cutlass.const_expr(self.k_split > 1):
                                self._group_reduce(vp, red_a, group, kq, lane)
                            # v^T in accumulator layout: one transposed ldmatrix of the
                            # group's sixteen value columns (rows tok, cols v).
                            v_tok = (matrix & Int32(1)) * Int32(8) + matrix_row
                            v_col = group * Int32(16) + (matrix >> Int32(1)) * Int32(8)
                            r0, r1, r2, r3 = ldmatrix_m8n8x4_trans_b16(
                                v_addr + group * Int32(512) + v_tok * Int32(32) + (matrix >> Int32(1)) * Int32(16)
                            )
                            beta_lo = beta_addr + tid * Int32(8)
                            beta00, beta01 = _ld_shared_v2_f32(beta_lo)
                            beta10, beta11 = _ld_shared_v2_f32(beta_lo + Int32(32))
                            square_row = (matrix >> Int32(1)) * Int32(8) + matrix_row
                            square_addr = square_row * Int32(32) + (matrix & Int32(1)) * Int32(16)
                            inv0, inv1, inv2, inv3 = ldmatrix_m8n8x4_b16(inv_addr + square_addr)
                            v00, v01 = _bf16x2_to_f32x2(r0)
                            v10, v11 = _bf16x2_to_f32x2(r1)
                            v02, v03 = _bf16x2_to_f32x2(r2)
                            v12, v13 = _bf16x2_to_f32x2(r3)
                            vp[0, 0] = (v00 - vp[0, 0]) * beta00
                            vp[0, 1] = (v01 - vp[0, 1]) * beta01
                            vp[0, 2] = (v02 - vp[0, 2]) * beta00
                            vp[0, 3] = (v03 - vp[0, 3]) * beta01
                            vp[1, 0] = (v10 - vp[1, 0]) * beta10
                            vp[1, 1] = (v11 - vp[1, 1]) * beta11
                            vp[1, 2] = (v12 - vp[1, 2]) * beta10
                            vp[1, 3] = (v13 - vp[1, 3]) * beta11
                            a_vp0 = pack_f32x2_to_bfloat2(vp[0, 0], vp[0, 1])
                            a_vp1 = pack_f32x2_to_bfloat2(vp[0, 2], vp[0, 3])
                            a_vp2 = pack_f32x2_to_bfloat2(vp[1, 0], vp[1, 1])
                            a_vp3 = pack_f32x2_to_bfloat2(vp[1, 2], vp[1, 3])

                            # Phase B: U^T = v'^T INV^T (every warp of the group).
                            u[0, 0], u[0, 1], u[0, 2], u[0, 3] = bf16_mma_m16n8k16_f32(
                                Float32(0.0), Float32(0.0), Float32(0.0), Float32(0.0),
                                a_vp0, a_vp1, a_vp2, a_vp3, inv0, inv1,
                            )
                            u[1, 0], u[1, 1], u[1, 2], u[1, 3] = bf16_mma_m16n8k16_f32(
                                Float32(0.0), Float32(0.0), Float32(0.0), Float32(0.0),
                                a_vp0, a_vp1, a_vp2, a_vp3, inv2, inv3,
                            )
                            a_u0 = pack_f32x2_to_bfloat2(u[0, 0], u[0, 1])
                            a_u1 = pack_f32x2_to_bfloat2(u[0, 2], u[0, 3])
                            a_u2 = pack_f32x2_to_bfloat2(u[1, 0], u[1, 1])
                            a_u3 = pack_f32x2_to_bfloat2(u[1, 2], u[1, 3])

                            # Phase C: out^T = U^T Mqk^T + S^T q~^T over this warp's
                            # columns. The U^T Mqk^T product seeds two of the four
                            # chains (kq == 0 only); the rest is as in phase A.
                            if cutlass.const_expr(self.summary_mode in (0, 4, 5)):
                                for chain in cutlass.range_constexpr(4):
                                    for item in cutlass.range_constexpr(4):
                                        parts[chain, item] = Float32(0.0)
                                if group_lead:
                                    b0, b1, b2, b3 = ldmatrix_m8n8x4_b16(mqk_addr + square_addr)
                                    parts[0, 0], parts[0, 1], parts[0, 2], parts[0, 3] = bf16_mma_m16n8k16_f32(
                                        Float32(0.0), Float32(0.0), Float32(0.0), Float32(0.0),
                                        a_u0, a_u1, a_u2, a_u3, b0, b1,
                                    )
                                    parts[1, 0], parts[1, 1], parts[1, 2], parts[1, 3] = bf16_mma_m16n8k16_f32(
                                        Float32(0.0), Float32(0.0), Float32(0.0), Float32(0.0),
                                        a_u0, a_u1, a_u2, a_u3, b2, b3,
                                    )
                                for operand_group in cutlass.range_constexpr(self.kb_steps // 2):
                                    for kb in cutlass.range_constexpr(2 * operand_group, 2 * operand_group + 2):
                                        logical_chunk = chunk_base + Int32(kb * 2) + (matrix & Int32(1))
                                        physical = logical_chunk ^ (tok_a & Int32(7))
                                        bfrag[kb % 2, 0], bfrag[kb % 2, 1], bfrag[kb % 2, 2], bfrag[kb % 2, 3] = ldmatrix_m8n8x4_b16(
                                            qt_addr + tok_a * Int32(256) + physical * Int32(16)
                                        )
                                    for kb in cutlass.range_constexpr(2 * operand_group, 2 * operand_group + 2):
                                        for half in cutlass.range_constexpr(2):
                                            chain = 2 * (kb % 2) + half
                                            parts[chain, 0], parts[chain, 1], parts[chain, 2], parts[chain, 3] = (
                                                bf16_mma_m16n8k16_f32(
                                                    parts[chain, 0], parts[chain, 1], parts[chain, 2], parts[chain, 3],
                                                    shadow[kb, 0], shadow[kb, 1], shadow[kb, 2], shadow[kb, 3],
                                                    bfrag[kb % 2, 2 * half], bfrag[kb % 2, 2 * half + 1],
                                                )
                                            )
                                for half in cutlass.range_constexpr(2):
                                    for item in cutlass.range_constexpr(4):
                                        out[half, item] = parts[half, item] + parts[2 + half, item]
                                if cutlass.const_expr(self.k_split > 1):
                                    self._group_reduce(out, red_c, group, kq, lane)
                                if group_lead:
                                    # Phase A has consumed this group's V tile; reuse it for output.
                                    output_tile_addr = v_addr + group * Int32(512) + v_tok * Int32(32) + (matrix >> Int32(1)) * Int32(16)
                                    if cutlass.const_expr(not self.is_gdn):
                                        output_tile_addr = out_addr + (v_tok * Int32(vd) + v_col) * Int32(2)
                                    _stmatrix_x4_trans(
                                        output_tile_addr,
                                        pack_f32x2_to_bfloat2(out[0, 0], out[0, 1]),
                                        pack_f32x2_to_bfloat2(out[1, 0], out[1, 1]),
                                        pack_f32x2_to_bfloat2(out[0, 2], out[0, 3]),
                                        pack_f32x2_to_bfloat2(out[1, 2], out[1, 3]),
                                    )
                                    cute.arch.sync_warp()
                                    store_row = lane >> Int32(1)
                                    store_chunk = group * Int32(2) + (lane & Int32(1))
                                    if store_row < rows_live:
                                        output_copy_addr = v_addr + group * Int32(512) + store_row * Int32(32) + (lane & Int32(1)) * Int32(16)
                                        if cutlass.const_expr(not self.is_gdn):
                                            output_copy_addr = out_addr + (store_row * Int32(v_chunks) + store_chunk) * Int32(16)
                                        c0, c1, c2, c3 = ld_shared_v4_u32(
                                            output_copy_addr
                                        )
                                        element = (
                                            (token_base + store_row).to(Int64) * out_stride
                                            + head_elements
                                            + split_elements
                                            + (store_chunk * Int32(8)).to(Int64)
                                        )
                                        st_global_v4_u32(_pointer_address(output, element), c0, c1, c2, c3)

                            # Phase D: S^T <- S^T * lambda_c[k] + U^T k_r over this warp's columns.
                            tok_d = (matrix & Int32(1)) * Int32(8) + matrix_row
                            for pair in cutlass.range_constexpr(self.nb_blocks // 2):
                                logical_chunk = chunk_base + Int32(pair * 2) + (matrix >> Int32(1))
                                physical = logical_chunk ^ (tok_d & Int32(7))
                                bfrag[pair, 0], bfrag[pair, 1], bfrag[pair, 2], bfrag[pair, 3] = (
                                    ldmatrix_m8n8x4_trans_b16(
                                        kr_addr + tok_d * Int32(256) + physical * Int32(16)
                                    )
                                )
                            scalar_lambda = Float32(0.0)
                            if cutlass.const_expr(self.is_gdn):
                                scalar_lambda = ld_shared_f32(lam_addr)
                            else:
                                for nb in cutlass.range_constexpr(self.nb_blocks):
                                    kcol = col_base + Int32(nb * 8) + tid * Int32(2)
                                    lam[nb, 0], lam[nb, 1] = _ld_shared_v2_f32(lam_addr + kcol * Int32(4))
                            # Pair p: scale its two blocks, issue their MMAs, then
                            # refresh the shadow of pair p - 2 (whose MMAs are done).
                            pairs = self.nb_blocks // 2
                            for pair in cutlass.range_constexpr(pairs):
                                for nb in cutlass.range_constexpr(2 * pair, 2 * pair + 2):
                                    lambda0, lambda1 = scalar_lambda, scalar_lambda
                                    if cutlass.const_expr(not self.is_gdn):
                                        lambda0, lambda1 = lam[nb, 0], lam[nb, 1]
                                    acc[nb, 0] = acc[nb, 0] * lambda0
                                    acc[nb, 1] = acc[nb, 1] * lambda1
                                    acc[nb, 2] = acc[nb, 2] * lambda0
                                    acc[nb, 3] = acc[nb, 3] * lambda1
                                acc[2 * pair, 0], acc[2 * pair, 1], acc[2 * pair, 2], acc[2 * pair, 3] = (
                                    bf16_mma_m16n8k16_f32(
                                        acc[2 * pair, 0], acc[2 * pair, 1], acc[2 * pair, 2], acc[2 * pair, 3],
                                        a_u0, a_u1, a_u2, a_u3, bfrag[pair, 0], bfrag[pair, 1],
                                    )
                                )
                                (
                                    acc[2 * pair + 1, 0],
                                    acc[2 * pair + 1, 1],
                                    acc[2 * pair + 1, 2],
                                    acc[2 * pair + 1, 3],
                                ) = bf16_mma_m16n8k16_f32(
                                    acc[2 * pair + 1, 0], acc[2 * pair + 1, 1],
                                    acc[2 * pair + 1, 2], acc[2 * pair + 1, 3],
                                    a_u0, a_u1, a_u2, a_u3, bfrag[pair, 2], bfrag[pair, 3],
                                )
                                if cutlass.const_expr(pair >= 2):
                                    done = pair - 2
                                    shadow[done, 0] = pack_f32x2_to_bfloat2(acc[2 * done, 0], acc[2 * done, 1])
                                    shadow[done, 1] = pack_f32x2_to_bfloat2(acc[2 * done, 2], acc[2 * done, 3])
                                    shadow[done, 2] = pack_f32x2_to_bfloat2(acc[2 * done + 1, 0], acc[2 * done + 1, 1])
                                    shadow[done, 3] = pack_f32x2_to_bfloat2(acc[2 * done + 1, 2], acc[2 * done + 1, 3])
                            for done in cutlass.range_constexpr(max(0, pairs - 2), pairs):
                                shadow[done, 0] = pack_f32x2_to_bfloat2(acc[2 * done, 0], acc[2 * done, 1])
                                shadow[done, 1] = pack_f32x2_to_bfloat2(acc[2 * done, 2], acc[2 * done, 3])
                                shadow[done, 2] = pack_f32x2_to_bfloat2(acc[2 * done + 1, 0], acc[2 * done + 1, 1])
                                shadow[done, 3] = pack_f32x2_to_bfloat2(acc[2 * done + 1, 2], acc[2 * done + 1, 3])
                            if cutlass.const_expr(self.summary_mode in (2, 3) and self.k_split == 1):
                                if (not summary_local) & ((step & Int32(1)) == Int32(1)):
                                    all_zero = cutlass.Boolean(True)
                                    for nb in cutlass.range_constexpr(self.nb_blocks):
                                        for item in cutlass.range_constexpr(4):
                                            all_zero = all_zero & (acc[nb, item] == Float32(0.0))
                                    zero_state = cute.arch.vote_all_sync(all_zero)
                            if cutlass.const_expr(self.summary_mode in (4, 5)):
                                if local == Int32(_CONVERGENCE_TOKENS // _CHUNK - 1):
                                    probe_slot = Int64(2 + 3 * self.rows) + seq.to(Int64)
                                    if cutlass.const_expr(self.summary_mode == 4):
                                        self._store_state(
                                            recurrent_state, acc, probe_slot * slot_stride + head_base,
                                            row0, row1, col_base, tid,
                                        )
                                    else:
                                        equal = cutlass.Boolean(True)
                                        for nb in cutlass.range_constexpr(self.nb_blocks):
                                            col = col_base + Int32(nb * 8) + tid * Int32(2)
                                            for item in cutlass.range_constexpr(4):
                                                row = row0 if item < 2 else row1
                                                address = (probe_slot * slot_stride + head_base
                                                           + row.to(Int64) * Int64(_HEAD_DIM)
                                                           + col.to(Int64) + Int64(item % 2))
                                                value = Float32(recurrent_state[address])
                                                equal = equal & (_f32_bits(acc[nb, item]) == _f32_bits(value))
                                        converged = cute.arch.vote_all_sync(equal)
                                        if converged:
                                            if lane == Int32(0):
                                                atomic_add_shared_i32(finished_addr, Int32(1))
                        if cutlass.const_expr(self.summary_mode in (2, 3) and self.k_split == 1):
                            if (not summary_local) & zero_state & (not zero_announced):
                                if lane == Int32(0):
                                    atomic_add_shared_i32(finished_addr, Int32(1))
                                zero_announced = cutlass.Boolean(True)
                        # Every read of this stage is done: release it to the producer.
                        cute.arch.sync_warp()
                        if lane == Int32(0):
                            cute.arch.mbarrier_arrive(empty_bar + stage)
                        if stage == Int32(self.stages - 1):
                            cons_phase = cons_phase ^ Int32(1)
                        count += Int32(1)
                        step += Int32(1)

                        if cutlass.const_expr(self.checkpoint_export):
                            if (offset > Int32(0)) & ((local + Int32(1)) * Int32(_CHUNK) == offset) & (not converged):
                                if not self._is_null(checkpoint):
                                    self._store_state(
                                        recurrent_state, acc, checkpoint * slot_stride + head_base,
                                        row0, row1, col_base, tid,
                                    )
                    # Final state, or the running state for the next window.
                    if cutlass.const_expr(self.summary_mode == 5):
                        if converged:
                            local_slot = Int64(2 + self.rows) + seq.to(Int64)
                            self._load_state(
                                recurrent_state, acc, local_slot * slot_stride + head_base,
                                row0, row1, col_base, tid,
                            )
                    if not self._is_null(final):
                        self._store_state(
                            recurrent_state, acc, final * slot_stride + head_base,
                            row0, row1, col_base, tid,
                        )
            # Empty sequences (ranks past the tiled ones) copy initial to
            # final in window 0.
            if window == Int32(0):
                empty_rank = tiled_seqs + Int32(by)
                while empty_rank < live_seqs:
                    empty_seq = sorted_seq[empty_rank].to(Int32)
                    empty_initial = Int64(initial_indices[empty_seq])
                    empty_final = Int64(
                        final_indices[empty_seq.to(Int64) * final_stride]
                    )
                    if not self._is_null(empty_final):
                        for nb in cutlass.range_constexpr(self.nb_blocks):
                            acc[nb, 0] = Float32(0.0)
                            acc[nb, 1] = Float32(0.0)
                            acc[nb, 2] = Float32(0.0)
                            acc[nb, 3] = Float32(0.0)
                        if not self._is_null(empty_initial):
                            self._load_state(
                                recurrent_state, acc, empty_initial * slot_stride + head_base,
                                row0, row1, col_base, tid,
                            )
                        self._store_state(
                            recurrent_state, acc, empty_final * slot_stride + head_base,
                            row0, row1, col_base, tid,
                        )
                    empty_rank += Int32(self.rows)


def _recurrence_key(binding: Binding) -> tuple[object, ...]:
    caps = binding._state.caps
    plan = binding._state
    key = (
        "recurrence",
        binding.output.device.index,
        caps.heads,
        caps.tiles_capacity,
        plan.window_tiles,
        plan.max_windows,
        plan.recurrence_rows,
        plan.v_split,
        plan.k_split,
        plan.stages,
        caps.checkpoint_export,
        caps.null_state_index,
        binding.initial_state_indices.dtype,
    )
    if caps.is_gdn:
        key = (*key, "scalar_gate")
    return (*key, "max_sequence_tiles", plan.max_sequence_tiles) if plan.max_sequence_tiles else key


def _compile_recurrence(binding: Binding, summary_mode: int = 0) -> tuple[tuple[object, ...], Callable[..., None]]:
    plan, caps = binding._state, binding._state.caps
    if summary_mode not in (0, 1, 2, 3, 4, 5):
        raise ValueError("unsupported GDN parallel recurrence mode")
    if summary_mode and (not caps.is_gdn or plan.max_windows != 1
                         or caps.null_state_index != 0 or plan.max_sequence_tiles <= 0
                         or plan.recurrence_rows != caps.max_seqs):
        raise ValueError("GDN parallel recurrence requires a single-window segment plan with null slot zero")
    if summary_mode in (4, 5) and (plan.k_split != 1 or caps.max_state_slots < 2 + 4 * caps.max_seqs):
        raise ValueError("GDN output reuse requires unsplit keys and preplanned convergence states")
    base_key = _recurrence_key(binding)
    key = base_key if summary_mode == 0 else (*base_key, "summary", int(summary_mode))
    cached = _RECURRENCE_CACHE.get(key)
    if cached is not None:
        return key, cached
    caps = binding._state.caps
    index_type = _numeric_type(binding.initial_state_indices.dtype)
    kernel = _RecurrenceKernel(
        heads=caps.heads,
        tiles_capacity=caps.tiles_capacity,
        window_tiles=binding._state.window_tiles,
        rows=binding._state.recurrence_rows,
        v_split=binding._state.v_split,
        k_split=binding._state.k_split,
        stages=binding._state.stages,
        checkpoint_export=caps.checkpoint_export,
        null_state_index=caps.null_state_index,
        index_type=index_type,
        summary_mode=summary_mode,
        max_sequence_tiles=binding._state.max_sequence_tiles,
        is_gdn=caps.is_gdn,
    )
    raise_if_kernel_resolution_frozen("cute.compile", target=kernel, cache_key=key)
    raw = b12x_compile(
        kernel,
        _fake_pointer(BFloat16),
        _fake_pointer(Int32),
        _fake_pointer(Int32),
        _fake_pointer(Int32),
        _fake_pointer(Int32),
        _fake_pointer(index_type),
        _fake_pointer(index_type),
        _fake_pointer(index_type),
        _fake_pointer(Int32),
        _fake_pointer(Int32),
        _fake_pointer(Int32),
        _fake_pointer(cutlass.Int8),
        _fake_pointer(Float32),
        _fake_pointer(BFloat16),
        Int64(1),
        Int64(1),
        Int64(1),
        Int64(1),
        Int32(0),
        current_cuda_stream(),
        compile_spec=KernelCompileSpec.from_key("sequence.delta_prefill.recurrence", 2, key),
    )

    def launch(active: Binding, window: int) -> None:
        if _recurrence_key(active) != base_key:
            raise ValueError("compiled delta-rule recurrence kernel does not match the binding")
        raw(
            _pointer(active.v, BFloat16),
            _pointer(active.cu_seqlens, Int32),
            _pointer(active.band_base, Int32),
            _pointer(active.sorted_seq, Int32),
            _pointer(active.window_table, Int32),
            _pointer(active.initial_state_indices, index_type),
            _pointer(active.final_state_indices, index_type),
            _pointer(active.checkpoint_state_indices, index_type),
            _pointer(active.checkpoint_offsets, Int32),
            _pointer(active.num_seqs, Int32),
            _pointer(active.ready_flags, Int32),
            _pointer(active.ws.view(torch.int8), cutlass.Int8),
            _pointer(active.recurrent_state, Float32),
            _pointer(active.output, BFloat16),
            int(active.v.stride(0)),
            int(active.output.stride(0)),
            int(active.recurrent_state.stride(0)),
            int(active.final_state_indices.stride(0)),
            int(window),
            current_cuda_stream(),
        )

    attach_programs(launch, raw)
    _RECURRENCE_CACHE[key] = launch
    return key, launch


def run_recurrence(binding: Binding, *, window: int = 0) -> None:
    """Walk the live tiles of one window (stage 2); requires stages 0 and 1.

    The launch polls the window's ready flags, so it may be issued on a stream
    running concurrently with the same window's prepare launch.
    """
    with torch.cuda.device(binding.output.device):
        _launch_stage(
            lambda b: (_recurrence_key(b), _RECURRENCE_CACHE.get(_recurrence_key(b))),
            _compile_recurrence,
            binding,
            int(window),
        )


def _prologue_key(binding: Binding) -> tuple[object, ...]:
    caps = binding._state.caps
    key = (
        "prologue",
        binding.output.device.index,
        caps.max_seqs,
        caps.tiles_capacity,
        binding._state.window_tiles,
        binding._state.max_windows,
        2 * binding._state.window_tiles * caps.heads,
    )
    return key if binding._state.workspace_windows == 2 else (*key, "workspace_windows", binding._state.workspace_windows)


def _compile_prologue(binding: Binding) -> tuple[tuple[object, ...], Callable[..., None]]:
    key = _prologue_key(binding)
    cached = _PROLOGUE_CACHE.get(key)
    if cached is not None:
        return key, cached
    caps = binding._state.caps
    kernel = _PrologueKernel(
        max_seqs=caps.max_seqs,
        tiles_capacity=caps.tiles_capacity,
        window_tiles=binding._state.window_tiles,
        max_windows=binding._state.max_windows,
        flag_count=binding._state.workspace_windows * binding._state.window_tiles * caps.heads,
    )
    raise_if_kernel_resolution_frozen("cute.compile", target=kernel, cache_key=key)
    raw = b12x_compile(
        kernel,
        _fake_pointer(Int32),
        _fake_pointer(Int32),
        _fake_pointer(Int32),
        _fake_pointer(Int32),
        _fake_pointer(Int32),
        _fake_pointer(Int32),
        _fake_pointer(Int32),
        _fake_pointer(Int32),
        _fake_pointer(Int32),
        Int32(1),
        current_cuda_stream(),
        compile_spec=KernelCompileSpec.from_key("sequence.delta_prefill.prologue", 4, key),
    )

    def launch(active: Binding) -> None:
        if _prologue_key(active) != key:
            raise ValueError("compiled delta-rule prologue does not match the binding")
        raw(
            _pointer(active.cu_seqlens, Int32),
            _pointer(active.num_seqs, Int32),
            _pointer(active.band_base, Int32),
            _pointer(active.sorted_seq, Int32),
            _pointer(active.rank_of, Int32),
            _pointer(active.pos_seq, Int32),
            _pointer(active.pos_local, Int32),
            _pointer(active.window_table, Int32),
            _pointer(active.ready_flags, Int32),
            int(active.seq_capacity),
            current_cuda_stream(),
        )

    attach_programs(launch, raw)
    _PROLOGUE_CACHE[key] = launch
    return key, launch


def _prepare_key(binding: Binding) -> tuple[object, ...]:
    caps = binding._state.caps
    return (
        "prepare",
        binding.output.device.index,
        caps.is_gdn,
        caps.key_heads,
        caps.heads,
        caps.tiles_capacity,
        binding._state.window_tiles,
        caps.qk_l2norm,
        binding.A_log.dtype,
        binding.dt_bias.dtype,
    )


def _compile_prepare(binding: Binding) -> tuple[tuple[object, ...], Callable[..., None]]:
    key = _prepare_key(binding)
    cached = _PREPARE_CACHE.get(key)
    if cached is not None:
        return key, cached
    caps = binding._state.caps
    a_log_type = _numeric_type(binding.A_log.dtype)
    dt_bias_type = _numeric_type(binding.dt_bias.dtype)
    kernel = _PrepareKernel(
        heads=caps.heads,
        key_heads=caps.key_heads,
        is_gdn=caps.is_gdn,
        tiles_capacity=caps.tiles_capacity,
        window_tiles=binding._state.window_tiles,
        qk_l2norm=caps.qk_l2norm,
        a_log_type=a_log_type,
        dt_bias_type=dt_bias_type,
    )
    raise_if_kernel_resolution_frozen("cute.compile", target=kernel, cache_key=key)
    raw = b12x_compile(
        kernel,
        _fake_pointer(BFloat16),
        _fake_pointer(BFloat16),
        _fake_pointer(BFloat16),
        _fake_pointer(BFloat16),
        _fake_pointer(a_log_type),
        _fake_pointer(dt_bias_type),
        _fake_pointer(Int32),
        _fake_pointer(Int32),
        _fake_pointer(Int32),
        _fake_pointer(Int32),
        _fake_pointer(BFloat16),
        _fake_pointer(Float32),
        Int64(1),
        Int64(1),
        Int64(1),
        Int64(1),
        Int64(1),
        Float32(1.0),
        Float32(1.0),
        Float32(1.0),
        Int32(0),
        current_cuda_stream(),
        compile_spec=KernelCompileSpec.from_key("sequence.delta_prefill.prepare", 2, key),
    )

    def launch(active: Binding, scale: float, gate_scale: float, eps: float, window: int) -> None:
        if _prepare_key(active) != key:
            raise ValueError("compiled delta-rule prepare kernel does not match the binding")
        raw(
            _pointer(active.q, BFloat16),
            _pointer(active.k, BFloat16),
            _pointer(active.raw_g, BFloat16),
            _pointer(active.raw_beta, BFloat16),
            _pointer(active.A_log, a_log_type),
            _pointer(active.dt_bias, dt_bias_type),
            _pointer(active.cu_seqlens, Int32),
            _pointer(active.pos_seq, Int32),
            _pointer(active.pos_local, Int32),
            _pointer(active.ready_flags, Int32),
            _pointer(active.ws.view(torch.bfloat16), BFloat16),
            _pointer(active.ws.view(torch.float32), Float32),
            int(active.q.stride(0)),
            int(active.k.stride(0)),
            int(active.raw_g.stride(0)),
            int(active.raw_beta.stride(0)),
            int(active.raw_beta.stride(1)),
            float(scale),
            float(gate_scale),
            float(eps),
            int(window),
            current_cuda_stream(),
        )

    attach_programs(launch, raw)
    _PREPARE_CACHE[key] = launch
    return key, launch


def _launch_stage(cache_lookup, compile_fn, binding: Binding, *args) -> None:
    capturing = torch.cuda.is_current_stream_capturing()
    key, launch = cache_lookup(binding)
    if capturing and (launch is None or key not in _WARMED):
        raise RuntimeError(
            "Delta-rule prefill kernels must be compiled and warm-run before CUDA graph capture"
        )
    if launch is None:
        key, launch = compile_fn(binding)
    launch(binding, *args)
    if not capturing:
        _WARMED.add(key)


def run_prologue(binding: Binding) -> None:
    """Clear the ready flags and build the tile tables (stage 0)."""
    with torch.cuda.device(binding.output.device):
        _launch_stage(
            lambda b: (_prologue_key(b), _PROLOGUE_CACHE.get(_prologue_key(b))),
            _compile_prologue,
            binding,
        )


def run_prepare(
    binding: Binding, *, lower_bound: float, scale: float, eps: float, window: int = 0
) -> None:
    """Fill the workspace ring slot of one window (stage 1); requires the prologue first."""
    with torch.cuda.device(binding.output.device):
        _launch_stage(
            lambda b: (_prepare_key(b), _PREPARE_CACHE.get(_prepare_key(b))),
            _compile_prepare,
            binding,
            float(scale),
            float(lower_bound) * _LOG2E,
            float(eps),
            int(window),
        )


@dataclass
class _SideResources:
    """Per-device side stream and event pool for the window pipeline."""

    stream: torch.cuda.Stream
    events: list[torch.cuda.Event]


_SIDE: dict[int, _SideResources] = {}


def _side_resources(device: torch.device, windows: int) -> _SideResources:
    """Return the side stream and at least ``2 * windows + 1`` initialized events."""
    needed = 2 * int(windows) + 1
    resources = _SIDE.get(device.index)
    capturing = torch.cuda.is_current_stream_capturing()
    if resources is None or len(resources.events) < needed:
        if capturing:
            raise RuntimeError(
                "Delta-rule prefill pipeline resources must be created by a warm run before CUDA graph capture"
            )
        if resources is None:
            resources = _SideResources(stream=torch.cuda.Stream(device=device), events=[])
            _SIDE[device.index] = resources
        current = torch.cuda.current_stream(device)
        while len(resources.events) < needed:
            event = torch.cuda.Event()
            event.record(current)
            resources.events.append(event)
    return resources


def run_prefill(
    binding: Binding, *, programs: tuple, resources: _SideResources,
    lower_bound: float, scale: float, eps: float, windows: int | None = None,
) -> None:
    """Launch the window pipeline: prologue, then prepare and recurrence per window.

    Prepare launches run on a per-device side stream, recurrence launches on
    the current stream. Recurrence waits for its window's prepare, while the
    next prepare can overlap the previous recurrence. Prepare of window ``w``
    waits for the recurrence of window ``w - 2`` before reusing that workspace
    ring slot. Under stream capture the fork and join are recorded as graph
    dependencies.
    """
    device = binding.output.device
    plan = binding._state
    launched = plan.max_windows if windows is None else int(windows)
    if launched < 1 or launched > plan.max_windows:
        raise ValueError(f"windows must be in 1..{plan.max_windows}, got {launched}")
    if len(resources.events) < 2 * launched + 1:
        raise ValueError("prepared prefill event capacity is insufficient")
    prologue, prepare, recurrence = programs
    gate_scale = float(lower_bound) * _LOG2E
    with torch.cuda.device(device):
        main = torch.cuda.current_stream(device)
        side = resources.stream
        fork = resources.events[2 * launched]
        events = resources.events
        prologue(binding)
        # Enqueue window by window so every event is recorded before a stream
        # waits on it (a wait binds to the event's most recent record).
        fork.record(main)
        side.wait_event(fork)
        for window in range(launched):
            with torch.cuda.stream(side):
                if window >= 2:
                    side.wait_event(events[launched + window - 2])
                prepare(binding, float(scale), gate_scale, float(eps), window)
                events[window].record(side)
            main.wait_event(events[window])
            recurrence(binding, window)
            events[launched + window].record(main)
        main.wait_event(events[launched - 1])


def compile_binding(binding: Binding) -> tuple:
    """Return all stage programs without creating pipeline resources or launching."""
    with torch.cuda.device(binding.output.device):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Delta-rule prefill compilation is forbidden during CUDA capture")
        return (
            _compile_prologue(binding)[1],
            _compile_prepare(binding)[1],
            _compile_recurrence(binding)[1],
        )


def prewarm_binding(binding: Binding) -> None:
    """Compile the three stages for ``binding`` and create the pipeline resources."""
    with torch.cuda.device(binding.output.device):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Delta-rule prefill compilation is forbidden during CUDA capture")
        compile_binding(binding)
        _side_resources(binding.output.device, binding._state.max_windows)


def workspace_tiles(binding: Binding, window: int = 0) -> dict[str, torch.Tensor]:
    """Return logical views of the ring slot holding ``window``.

    Operand tiles are de-swizzled copies; ``inv``, ``mqk``, ``lambda_c``,
    ``beta``, and ``ready`` are views into the buffer.
    """
    tiles = binding._state.window_tiles
    heads = binding._state.caps.heads
    slot = slice((window & 1) * tiles, (window & 1) * tiles + tiles)
    ws = binding.ws[slot]
    rows = torch.arange(_CHUNK).view(_CHUNK, 1)
    cols = torch.arange(_HEAD_DIM).view(1, _HEAD_DIM)
    physical = ((((cols >> 3) ^ (rows & 7)) << 3) | (cols & 7)).to(ws.device)
    index = physical.view(1, 1, _CHUNK, _HEAD_DIM).expand(tiles, heads, _CHUNK, _HEAD_DIM)

    def bf16(offset: int, elements: int, *shape: int) -> torch.Tensor:
        return ws[..., offset : offset + 2 * elements].view(torch.bfloat16).view(tiles, heads, *shape)

    def f32(offset: int, elements: int) -> torch.Tensor:
        return ws[..., offset : offset + 4 * elements].view(torch.float32).view(tiles, heads, elements)

    def logical(tile: torch.Tensor) -> torch.Tensor:
        return torch.gather(tile, 3, index)

    return {
        "q_tilde": logical(bf16(REC.Q_TILDE, _CHUNK * _HEAD_DIM, _CHUNK, _HEAD_DIM)),
        "k_tilde": logical(bf16(REC.K_TILDE, _CHUNK * _HEAD_DIM, _CHUNK, _HEAD_DIM)),
        "k_r": logical(bf16(REC.K_R, _CHUNK * _HEAD_DIM, _CHUNK, _HEAD_DIM)),
        "lambda_c": f32(REC.LAMBDA_C, _HEAD_DIM),
        "beta": f32(REC.BETA, _CHUNK),
        "inv": bf16(REC.INV, _CHUNK * _CHUNK, _CHUNK, _CHUNK),
        "mqk": bf16(REC.MQK, _CHUNK * _CHUNK, _CHUNK, _CHUNK),
        "ready": binding.ready_flags[slot],
    }


def clear_caches() -> None:
    _PROLOGUE_CACHE.clear()
    _PREPARE_CACHE.clear()
    _RECURRENCE_CACHE.clear()
    _WARMED.clear()
    _SIDE.clear()


__all__ = [
    "clear_caches",
    "prewarm_binding",
    "run_prefill",
    "run_prepare",
    "run_prologue",
    "run_recurrence",
    "workspace_tiles",
]
