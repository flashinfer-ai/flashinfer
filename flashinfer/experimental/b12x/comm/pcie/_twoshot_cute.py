"""CuTeDSL kernels for the PCIe two-shot FP8 collectives.

The public runtime and CUDA-IPC ownership live in :mod:`pcie_twoshot`.  This
module deliberately keeps the launch ABI small: peer addresses are stored in
stable, device-resident ``uint64`` tables and converted back to typed global
pointers in the kernel.  All offsets derived from rows, packs, blocks, or
ranks are widened before multiplication.
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Sequence

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64, Uint32
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import llvm

from b12x._lib.compiler import KernelCompileSpec
from b12x._lib.compiler import compile as b12x_compile
from b12x._lib.compile_plan import attach_programs
from b12x._lib.intrinsics import (
    cvt_e4m3x4_to_f32x4,
    ld_global_nc_f32,
    ld_global_nc_u32,
    ld_global_nc_v4_u32,
    st_global_v4_u32,
)
from b12x._lib.runtime_control import raise_if_kernel_resolution_frozen
from b12x._lib.utils import current_cuda_stream, make_ptr

from ._cute_intrinsics import (
    graph_epoch_arrive_serialized,
    ld_generic_f32,
    ld_relaxed_gpu_u32,
    pack_f32x2_to_bf16x2,
)


_MAX_BLOCKS = 64
_MAX_RANKS = 16
_FLAG_STRIDE = 32
_SELF_COUNTER_BYTES = _MAX_BLOCKS * _MAX_RANKS * 4
# Word zero of each 128-byte peer flag record is live. Two padding words in
# the first record hold channel-local graph state and cannot alias peer flags.
_GRAPH_EPOCH_OFFSET = _SELF_COUNTER_BYTES + 4
_GRAPH_BLOCKS_ARRIVED_OFFSET = _SELF_COUNTER_BYTES + 8


@dsl_user_op
def _ld_global_u32(address: Int64, *, loc=None, ip=None) -> Uint32:
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Int64(address).ir_value(loc=loc, ip=ip)],
            "ld.global.u32 $0, [$1];",
            "=r,l",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _st_global_u32(address: Int64, value: Uint32, *, loc=None, ip=None) -> None:
    llvm.inline_asm(
        None,
        [
            Int64(address).ir_value(loc=loc, ip=ip),
            Uint32(value).ir_value(loc=loc, ip=ip),
        ],
        "st.global.u32 [$0], $1;",
        "l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _ld_relaxed_sys_u32(address: Int64, *, loc=None, ip=None) -> Uint32:
    """Exact flag load used by the native block-pair barrier."""

    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Int64(address).ir_value(loc=loc, ip=ip)],
            "ld.relaxed.sys.global.u32 $0, [$1];",
            "=r,l",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _st_relaxed_sys_u32(
    address: Int64, value: Uint32, *, loc=None, ip=None
) -> None:
    """Exact flag store used by the native block-pair barrier."""

    llvm.inline_asm(
        None,
        [
            Int64(address).ir_value(loc=loc, ip=ip),
            Uint32(value).ir_value(loc=loc, ip=ip),
        ],
        "st.relaxed.sys.global.u32 [$0], $1;",
        "l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _fence_sc_sys(*, loc=None, ip=None) -> None:
    """PTX emitted for ``__threadfence_system`` by the native kernels."""

    llvm.inline_asm(
        None,
        [],
        "fence.sc.sys;",
        "",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _ld_generic_v4_u32(
    address: Int64,
    *,
    loc=None,
    ip=None,
) -> tuple[Uint32, Uint32, Uint32, Uint32]:
    """Load one IPC-staged payload through its generic/UVA address."""

    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [Int64(address).ir_value(loc=loc, ip=ip)],
        "ld.v4.b32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return tuple(
        Uint32(llvm.extractvalue(T.i32(), result, [index], loc=loc, ip=ip))
        for index in range(4)
    )


@dsl_user_op
def _st_generic_u32(
    address: Int64, value: Uint32, *, loc=None, ip=None
) -> None:
    """Store one scale word through an IPC generic/UVA address."""

    llvm.inline_asm(
        None,
        [
            Int64(address).ir_value(loc=loc, ip=ip),
            Uint32(value).ir_value(loc=loc, ip=ip),
        ],
        "st.b32 [$0], $1;",
        "l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _st_generic_v4_u32(
    address: Int64,
    value0: Uint32,
    value1: Uint32,
    value2: Uint32,
    value3: Uint32,
    *,
    loc=None,
    ip=None,
) -> None:
    """Store one payload pack through an IPC generic/UVA address."""

    llvm.inline_asm(
        None,
        [
            Int64(address).ir_value(loc=loc, ip=ip),
            Uint32(value0).ir_value(loc=loc, ip=ip),
            Uint32(value1).ir_value(loc=loc, ip=ip),
            Uint32(value2).ir_value(loc=loc, ip=ip),
            Uint32(value3).ir_value(loc=loc, ip=ip),
        ],
        "st.v4.b32 [$0], {$1, $2, $3, $4};",
        "l,r,r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


class _TwoShotLaunch:
    def __init__(
        self,
        operation: str,
        world_size: int,
        rank: int,
        device_slot_selection: bool,
        slot_bias: int,
        threads: int,
        row_elems: int,
    ) -> None:
        if operation not in ("reduce_scatter", "all_gather"):
            raise ValueError(f"invalid two-shot operation {operation!r}")
        self._operation = operation
        self._world_size = int(world_size)
        self._rank = int(rank)
        self._device_slot_selection = bool(device_slot_selection)
        self._slot_bias = int(slot_bias) & 1
        self._threads = int(threads)
        self._row_elems = int(row_elems)

    @cute.jit
    def __call__(
        self,
        payload: cute.Pointer,
        scale: cute.Pointer,
        staging0: cute.Pointer,
        staging1: cute.Pointer,
        staging2: cute.Pointer,
        staging3: cute.Pointer,
        staging4: cute.Pointer,
        staging5: cute.Pointer,
        staging6: cute.Pointer,
        staging7: cute.Pointer,
        signal0: cute.Pointer,
        signal1: cute.Pointer,
        signal2: cute.Pointer,
        signal3: cute.Pointer,
        signal4: cute.Pointer,
        signal5: cute.Pointer,
        signal6: cute.Pointer,
        signal7: cute.Pointer,
        output: cute.Pointer,
        rank: Int32,
        pack_stride: Int64,
        scale_offset: Int64,
        scale_stride: Int64,
        slot_bytes: Int64,
        rows_per_rank: Int32,
        row_elems: Int32,
        grid_x: Int32,
        stream: cuda.CUstream,
    ) -> None:
        self.kernel(
            payload,
            scale,
            staging0,
            staging1,
            staging2,
            staging3,
            staging4,
            staging5,
            staging6,
            staging7,
            signal0,
            signal1,
            signal2,
            signal3,
            signal4,
            signal5,
            signal6,
            signal7,
            output,
            rank,
            pack_stride,
            scale_offset,
            scale_stride,
            slot_bytes,
            rows_per_rank,
            row_elems,
        ).launch(
            grid=(grid_x, 1, 1),
            block=[self._threads, 1, 1],
            max_number_threads=(512, 1, 1),
            min_blocks_per_mp=1,
            cluster=(1, 1, 1),
            stream=stream,
        )

    @cute.jit
    def _select_address(
        self,
        pointers: Sequence[cute.Pointer],
        index: Int32,
    ) -> Int64:
        """Select one scalar launch pointer without unrolling peer work."""

        address = Int64(pointers[0].toint())
        if cutlass.const_expr(self._world_size == 2):
            if index == Int32(1):
                address = Int64(pointers[1].toint())
            return address
        if index < Int32(4):
            if index < Int32(2):
                address = Int64(pointers[0].toint())
                if index == Int32(1):
                    address = Int64(pointers[1].toint())
            else:
                address = Int64(pointers[2].toint())
                if index == Int32(3):
                    address = Int64(pointers[3].toint())
        else:
            if index < Int32(6):
                address = Int64(pointers[4].toint())
                if index == Int32(5):
                    address = Int64(pointers[5].toint())
            else:
                address = Int64(pointers[6].toint())
                if cutlass.const_expr(self._world_size == 8):
                    if index == Int32(7):
                        address = Int64(pointers[7].toint())
        return address

    @cute.jit
    def _barrier(
        self,
        signals: Sequence[cute.Pointer],
        rank: Int32,
    ) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        cute.arch.barrier()
        if tidx < Int32(self._world_size):
            _fence_sc_sys()
            if cutlass.const_expr(self._operation == "all_gather"):
                self_base = Int64(signals[self._rank].toint())
            else:
                self_base = self._select_address(signals, rank)
            self_counter_address = self_base + (
                Int64(bidx) * Int64(_MAX_RANKS) + Int64(tidx)
            ) * Int64(4)
            value = _ld_global_u32(self_counter_address) + Uint32(1)
            _st_global_u32(self_counter_address, value)

            flag_slot = Int64(value % Uint32(2))
            peer_base = self._select_address(signals, tidx)
            peer_counter_address = peer_base + Int64(_SELF_COUNTER_BYTES) + (
                (
                    flag_slot * Int64(_MAX_BLOCKS)
                    + Int64(bidx)
                )
                * Int64(_MAX_RANKS * _FLAG_STRIDE)
                + Int64(rank) * Int64(_FLAG_STRIDE)
            ) * Int64(4)
            self_counter_address = self_base + Int64(_SELF_COUNTER_BYTES) + (
                (
                    flag_slot * Int64(_MAX_BLOCKS)
                    + Int64(bidx)
                )
                * Int64(_MAX_RANKS * _FLAG_STRIDE)
                + Int64(tidx) * Int64(_FLAG_STRIDE)
            ) * Int64(4)
            _st_relaxed_sys_u32(peer_counter_address, value)
            observed = _ld_relaxed_sys_u32(self_counter_address)
            while observed != value:
                observed = _ld_relaxed_sys_u32(self_counter_address)
        cute.arch.barrier()

    @cute.jit
    def _load_accumulate_pack_global_nc(
        self,
        accumulator: cute.Tensor,
        address: Int64,
        scale: Float32,
    ) -> None:
        words = ld_global_nc_v4_u32(address)
        for word_index in cutlass.range_constexpr(4):
            values = cvt_e4m3x4_to_f32x4(words[word_index])
            for lane in cutlass.range_constexpr(4):
                element = word_index * 4 + lane
                accumulator[element] = (
                    accumulator[element] + values[lane] * scale
                )

    @cute.jit
    def _load_accumulate_pack_generic(
        self,
        accumulator: cute.Tensor,
        address: Int64,
        scale: Float32,
    ) -> None:
        words = _ld_generic_v4_u32(address)
        for word_index in cutlass.range_constexpr(4):
            values = cvt_e4m3x4_to_f32x4(words[word_index])
            for lane in cutlass.range_constexpr(4):
                element = word_index * 4 + lane
                accumulator[element] = (
                    accumulator[element] + values[lane] * scale
                )

    @cute.jit
    def _store_pack(
        self, output_address: Int64, accumulator: cute.Tensor
    ) -> None:
        st_global_v4_u32(
            output_address,
            pack_f32x2_to_bf16x2(accumulator[0], accumulator[1]),
            pack_f32x2_to_bf16x2(accumulator[2], accumulator[3]),
            pack_f32x2_to_bf16x2(accumulator[4], accumulator[5]),
            pack_f32x2_to_bf16x2(accumulator[6], accumulator[7]),
        )
        st_global_v4_u32(
            output_address + Int64(16),
            pack_f32x2_to_bf16x2(accumulator[8], accumulator[9]),
            pack_f32x2_to_bf16x2(accumulator[10], accumulator[11]),
            pack_f32x2_to_bf16x2(accumulator[12], accumulator[13]),
            pack_f32x2_to_bf16x2(accumulator[14], accumulator[15]),
        )

    @cute.kernel
    def kernel(
        self,
        payload: cute.Pointer,
        scale: cute.Pointer,
        staging0: cute.Pointer,
        staging1: cute.Pointer,
        staging2: cute.Pointer,
        staging3: cute.Pointer,
        staging4: cute.Pointer,
        staging5: cute.Pointer,
        staging6: cute.Pointer,
        staging7: cute.Pointer,
        signal0: cute.Pointer,
        signal1: cute.Pointer,
        signal2: cute.Pointer,
        signal3: cute.Pointer,
        signal4: cute.Pointer,
        signal5: cute.Pointer,
        signal6: cute.Pointer,
        signal7: cute.Pointer,
        output: cute.Pointer,
        rank: Int32,
        pack_stride: Int64,
        scale_offset: Int64,
        scale_stride: Int64,
        slot_bytes: Int64,
        rows_per_rank: Int32,
        row_elems: Int32,
    ) -> None:
        staging = (
            staging0,
            staging1,
            staging2,
            staging3,
            staging4,
            staging5,
            staging6,
            staging7,
        )
        signals = (
            signal0,
            signal1,
            signal2,
            signal3,
            signal4,
            signal5,
            signal6,
            signal7,
        )
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        gdim, _, _ = cute.arch.grid_dim()
        block_threads, _, _ = cute.arch.block_dim()
        if cutlass.const_expr(self._operation == "all_gather"):
            local_rank = Int32(self._rank)
        else:
            local_rank = rank
        packs_per_row = Int32(self._row_elems // 16)
        shard_packs = Int64(rows_per_rank) * Int64(packs_per_row)
        chunk = (shard_packs + Int64(gdim) - Int64(1)) // Int64(gdim)
        begin = Int64(bidx) * chunk
        end = begin + chunk
        if end > shard_packs:
            end = shard_packs
        row_begin = begin // Int64(packs_per_row)
        row_end = (end + Int64(packs_per_row) - Int64(1)) // Int64(
            packs_per_row
        )

        payload_address = Int64(payload.toint())
        scale_address = Int64(scale.toint())
        staging_slot_offset = Int64(0)
        if cutlass.const_expr(self._device_slot_selection):
            if cutlass.const_expr(self._operation == "all_gather"):
                self_signal = Int64(signals[self._rank].toint())
            else:
                self_signal = self._select_address(signals, local_rank)
            generation = ld_relaxed_gpu_u32(
                self_signal + Int64(_GRAPH_EPOCH_OFFSET)
            )
            staging_slot_offset = (
                Int64(
                    (generation + Uint32(self._slot_bias)) % Uint32(2)
                )
                * slot_bytes
            )
            # Once every thread in the CTA has selected this launch's slot,
            # retire its epoch contribution on the last warp.  Other warps
            # can start the remote push while the contended atomic completes,
            # hiding graph accounting under the full phase-one transfer.
            cute.arch.barrier()
            if Int32(tidx) == Int32(self._threads - 1):
                graph_epoch_arrive_serialized(
                    self_signal + Int64(_GRAPH_EPOCH_OFFSET),
                    self_signal + Int64(_GRAPH_BLOCKS_ARRIVED_OFFSET),
                    Uint32(gdim),
                )

        # Push remote shards.  Keep this as a runtime loop, matching the
        # native ``#pragma unroll 1`` rank-staggered traversal.
        peer_index = Int32(1)
        while peer_index < Int32(self._world_size):
            destination = (local_rank + peer_index) % Int32(self._world_size)
            destination_base = (
                self._select_address(staging, destination)
                + staging_slot_offset
            )
            destination_payload = destination_base + (
                Int64(local_rank) * pack_stride * Int64(16)
            )
            destination_scale = (
                destination_base
                + scale_offset
                + Int64(local_rank) * scale_stride * Int64(4)
            )

            if cutlass.const_expr(self._operation == "reduce_scatter"):
                source_pack = Int64(destination) * shard_packs
                source_row = Int64(destination) * Int64(rows_per_rank)
            else:
                source_pack = Int64(0)
                source_row = Int64(0)

            index = begin + Int64(tidx)
            while index < end:
                words = ld_global_nc_v4_u32(
                    payload_address + (source_pack + index) * Int64(16)
                )
                _st_generic_v4_u32(
                    destination_payload + index * Int64(16),
                    words[0],
                    words[1],
                    words[2],
                    words[3],
                )
                index += Int64(block_threads)

            row = row_begin + Int64(tidx)
            while row < row_end:
                source_scale_word = ld_global_nc_u32(
                    scale_address + (source_row + row) * Int64(4),
                )
                _st_generic_u32(
                    destination_scale + row * Int64(4),
                    source_scale_word,
                )
                row += Int64(block_threads)
            peer_index += Int32(1)

        self._barrier(signals, local_rank)

        if cutlass.const_expr(self._operation == "all_gather"):
            self_base = Int64(staging[self._rank].toint())
        else:
            self_base = self._select_address(staging, local_rank)
        self_base += staging_slot_offset
        output_address = Int64(output.toint())
        if cutlass.const_expr(self._operation == "reduce_scatter"):
            index = begin + Int64(tidx)
            while index < end:
                row = index // Int64(packs_per_row)
                accumulator = cute.make_rmem_tensor((16,), cutlass.Float32)
                for lane in cutlass.range_constexpr(16):
                    accumulator[lane] = Float32(0.0)

                local_pack = Int64(local_rank) * shard_packs + index
                local_scale = ld_global_nc_f32(
                    scale_address
                    + (Int64(local_rank) * Int64(rows_per_rank) + row) * Int64(4),
                )
                self._load_accumulate_pack_global_nc(
                    accumulator,
                    payload_address + local_pack * Int64(16),
                    local_scale,
                )

                for peer_index in cutlass.range(
                    Int32(1),
                    Int32(self._world_size),
                    Int32(1),
                    unroll=1,
                ):
                    source_rank = (
                        local_rank + peer_index
                    ) % Int32(self._world_size)
                    staged_pack = (
                        self_base
                        + Int64(source_rank) * pack_stride * Int64(16)
                        + index * Int64(16)
                    )
                    staged_scale = ld_generic_f32(
                        self_base
                        + scale_offset
                        + (
                            Int64(source_rank) * scale_stride + row
                        )
                        * Int64(4),
                    )
                    self._load_accumulate_pack_generic(
                        accumulator,
                        staged_pack,
                        staged_scale,
                    )

                self._store_pack(
                    output_address + index * Int64(32), accumulator
                )
                index += Int64(self._threads)
        else:
            first_index = begin + Int64(tidx)
            iteration_count = Int32(
                (
                    end
                    - first_index
                    + Int64(self._threads - 1)
                )
                // Int64(self._threads)
            )
            peer_index = Int32(0)
            index = Int64(0)
            while peer_index < Int32(self._world_size):
                source_rank = (
                    local_rank + peer_index
                ) % Int32(self._world_size)
                source_payload_base = Int64(0)
                source_scale_base = Int64(0)
                if source_rank == local_rank:
                    source_payload_base = payload_address
                    source_scale_base = scale_address
                else:
                    source_payload_base = (
                        self_base
                        + Int64(source_rank) * pack_stride * Int64(16)
                    )
                    source_scale_base = (
                        self_base
                        + scale_offset
                        + Int64(source_rank) * scale_stride * Int64(4)
                    )
                destination_base = (
                    output_address
                    + Int64(source_rank) * shard_packs * Int64(32)
                )
                for iteration in cutlass.range(
                    Int32(0),
                    iteration_count,
                    Int32(1),
                    unroll=1,
                ):
                    index = first_index + Int64(iteration) * Int64(
                        self._threads
                    )
                    row = index // Int64(packs_per_row)
                    accumulator = cute.make_rmem_tensor((16,), cutlass.Float32)
                    for lane in cutlass.range_constexpr(16):
                        accumulator[lane] = Float32(0.0)

                    self._load_accumulate_pack_generic(
                        accumulator,
                        source_payload_base + index * Int64(16),
                        ld_generic_f32(
                            source_scale_base + row * Int64(4)
                        ),
                    )
                    self._store_pack(
                        destination_base + index * Int64(32), accumulator
                    )
                peer_index += Int32(1)

def _twoshot_process_key(
    operation: str,
    world_size: int,
    rank: int,
    device_slot_selection: bool,
    slot_bias: int,
    threads: int,
    row_elems: int,
    device_index: int,
) -> tuple[object, ...]:
    return (
        str(operation),
        int(world_size),
        int(rank),
        bool(device_slot_selection),
        int(slot_bias) & 1 if device_slot_selection else 0,
        int(threads),
        int(row_elems),
        int(device_index),
    )




@functools.cache
def get_twoshot_launcher(
    operation: str,
    world_size: int,
    rank: int,
    device_slot_selection: bool,
    slot_bias: int,
    threads: int,
    row_elems: int,
    device_index: int,
) -> Callable[..., None]:
    """Compile and retain one static two-shot launcher specialization."""
    process_key = _twoshot_process_key(
        operation,
        world_size,
        rank,
        device_slot_selection,
        slot_bias,
        threads,
        row_elems,
        device_index,
    )
    del device_index  # part of the process-local cache key
    if world_size not in (2, 4, 8):
        raise ValueError(f"unsupported world size {world_size}")
    if rank < 0 or rank >= world_size:
        raise ValueError(f"rank {rank} is outside world size {world_size}")
    if threads <= 0 or threads > 512 or threads % 32 != 0:
        raise ValueError("threads must be a warp-aligned value in [32, 512]")
    if row_elems <= 0 or row_elems % 16 != 0:
        raise ValueError("row_elems must be a positive multiple of 16")

    slot_bias = int(slot_bias) & 1
    launch = _TwoShotLaunch(
        operation,
        world_size,
        rank,
        device_slot_selection,
        slot_bias,
        threads,
        row_elems,
    )
    cache_key = (
        operation,
        int(world_size),
        int(rank),
        bool(device_slot_selection),
        slot_bias,
        int(threads),
        int(row_elems),
    )
    raise_if_kernel_resolution_frozen(
        "cute.compile", target=launch, cache_key=cache_key
    )
    raw = b12x_compile(
        launch,
        make_ptr(cutlass.Uint32, 16, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(cutlass.Float32, 16, cute.AddressSpace.gmem, assumed_align=4),
        *(
            make_ptr(
                cutlass.Uint32,
                16,
                cute.AddressSpace.gmem,
                assumed_align=16,
            )
            for _ in range(8)
        ),
        *(
            make_ptr(
                cutlass.Uint32,
                16,
                cute.AddressSpace.gmem,
                assumed_align=4,
            )
            for _ in range(8)
        ),
        make_ptr(cutlass.Uint32, 16, cute.AddressSpace.gmem, assumed_align=16),
        0,
        1,
        1,
        1,
        256,
        1,
        16,
        1,
        current_cuda_stream(),
        compile_spec=KernelCompileSpec.from_key(
            f"comm.pcie.twoshot.{operation}",
            1,
            cache_key,
        ),
    )

    def run(
        payload_address: int,
        scale_address: int,
        staging_addresses: Sequence[int],
        signal_addresses: Sequence[int],
        output_address: int,
        rank: int,
        pack_stride: int,
        scale_offset: int,
        scale_stride: int,
        slot_bytes: int,
        rows_per_rank: int,
        row_elems: int,
        grid_x: int,
    ) -> None:
        """Launch the compiled two-shot collective with runtime arguments."""
        if len(staging_addresses) != 8 or len(signal_addresses) != 8:
            raise ValueError("two-shot scalar pointer ABI requires eight peers")
        raw_args = (
            make_ptr(
                cutlass.Uint32,
                payload_address,
                cute.AddressSpace.gmem,
                assumed_align=16,
            ),
            make_ptr(
                cutlass.Float32,
                scale_address,
                cute.AddressSpace.gmem,
                assumed_align=4,
            ),
            *(
                make_ptr(
                    cutlass.Uint32,
                    address,
                    cute.AddressSpace.gmem,
                    assumed_align=16,
                )
                for address in staging_addresses
            ),
            *(
                make_ptr(
                    cutlass.Uint32,
                    address,
                    cute.AddressSpace.gmem,
                    assumed_align=4,
                )
                for address in signal_addresses
            ),
            make_ptr(
                cutlass.Uint32,
                output_address,
                cute.AddressSpace.gmem,
                assumed_align=16,
            ),
            rank,
            pack_stride,
            scale_offset,
            scale_stride,
            slot_bytes,
            rows_per_rank,
            row_elems,
            grid_x,
            current_cuda_stream(),
        )
        raw(*raw_args)

    return attach_programs(run, raw)

__all__ = ["get_twoshot_launcher"]
