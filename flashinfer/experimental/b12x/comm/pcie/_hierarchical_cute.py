"""CuTeDSL all-reduce over four-rank islands with a possibly partial tail."""

from __future__ import annotations

from b12x._lib.program_cache import program_cache
from collections.abc import Callable, Sequence

from b12x._lib.compile_plan import attach_programs
import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64, Uint32
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import llvm

from b12x._lib.compiler import KernelCompileSpec
from b12x._lib.compiler import compile as b12x_compile
from b12x._lib.compile_plan import attach_programs
from b12x._lib.runtime_control import raise_if_kernel_resolution_frozen
from b12x._lib.utils import current_cuda_stream, make_ptr

from ._cute_intrinsics import pack_f32x2_to_bf16x2, unpack_bf16x2


_ISLAND_SIZE = 4
_MAX_ISLANDS = 4
_MAX_WORLD_SIZE = _ISLAND_SIZE * _MAX_ISLANDS

# Native Header byte offsets.  Each protocol flag occupies one 128-byte line.
_LOCAL_ARRIVED = 256
_COLLECTIVE_GENERATION = 128
_LEADER_READY = 16_640
_LEADER_CONSUMED = 33_024
_FINAL_READY = 49_408
_LOCAL_CONSUMED = 53_504


@dsl_user_op
def _atomic_add_global_u32(
    address: Int64, value: Uint32, *, loc=None, ip=None
) -> Uint32:
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [
                Int64(address).ir_value(loc=loc, ip=ip),
                Uint32(value).ir_value(loc=loc, ip=ip),
            ],
            "atom.global.add.u32 $0, [$1], $2;",
            "=r,l,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _load_acquire_sys_u32(address: Int64, *, loc=None, ip=None) -> Uint32:
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Int64(address).ir_value(loc=loc, ip=ip)],
            "ld.acquire.sys.global.u32 $0, [$1];",
            "=r,l",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _store_release_sys_u32(
    address: Int64, value: Uint32, *, loc=None, ip=None
) -> None:
    llvm.inline_asm(
        None,
        [
            Int64(address).ir_value(loc=loc, ip=ip),
            Uint32(value).ir_value(loc=loc, ip=ip),
        ],
        "st.release.sys.global.u32 [$0], $1;",
        "l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _fence_sc_sys(*, loc=None, ip=None) -> None:
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
def _nanosleep(cycles: Uint32, *, loc=None, ip=None) -> None:
    llvm.inline_asm(
        None,
        [Uint32(cycles).ir_value(loc=loc, ip=ip)],
        "nanosleep.u32 $0;",
        "r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@cute.jit
def _wait_for(address: Int64, generation: Uint32, cycles: int) -> None:
    observed = _load_acquire_sys_u32(address)
    while observed < generation:
        if cutlass.const_expr(cycles > 0):
            _nanosleep(Uint32(cycles))
        observed = _load_acquire_sys_u32(address)


@cute.jit
def _flag_address(base: Int64, region: int, block: Int32, index: Int32) -> Int64:
    return (
        base
        + Int64(region)
        + (
            Int64(block) * Int64(_ISLAND_SIZE) + Int64(index)
        )
        * Int64(128)
    )


class _HierarchicalLaunch:
    def __init__(
        self,
        world_size: int,
        rank: int,
        *,
        threads: int = 224,
        wait_nanosleep_cycles: int = 24,
        double_buffered: bool = False,
        deferred_consumption: bool = False,
        vectorized_bf16x2: bool = False,
    ) -> None:
        self._world_size = int(world_size)
        self._num_islands = (self._world_size + _ISLAND_SIZE - 1) // _ISLAND_SIZE
        self._rank = int(rank)
        self._threads = int(threads)
        self._wait_nanosleep_cycles = int(wait_nanosleep_cycles)
        self._double_buffered = bool(double_buffered)
        self._deferred_consumption = bool(deferred_consumption)
        self._vectorized_bf16x2 = bool(vectorized_bf16x2)
        if self._world_size not in (9, 10, 12, 16):
            raise ValueError(f"unsupported world size {self._world_size}")
        if not 0 <= self._rank < self._world_size:
            raise ValueError(
                f"invalid rank {self._rank} for world size {self._world_size}"
            )
        if self._vectorized_bf16x2:
            if self._threads != 112:
                raise ValueError("BF16x2 hierarchy requires 112 threads")
        elif not 32 <= self._threads <= 1024 or self._threads % 32:
            raise ValueError("threads must be a multiple of 32 in [32, 1024]")
        if not 0 <= self._wait_nanosleep_cycles <= 1024:
            raise ValueError("wait_nanosleep_cycles must be in [0, 1024]")
        if self._double_buffered and self._deferred_consumption:
            raise ValueError(
                "double buffering and deferred consumption are mutually exclusive"
            )
        self._island = self._rank // _ISLAND_SIZE
        self._local_rank = self._rank % _ISLAND_SIZE
        self._leader_rank = self._island * _ISLAND_SIZE
        # Header slots retain a four-rank stride. Absent ranks in the final
        # island must neither be dereferenced nor waited on by its leader.
        self._local_size = min(_ISLAND_SIZE, self._world_size - self._leader_rank)

    @cute.jit
    def __call__(
        self,
        slab0: cute.Pointer,
        slab1: cute.Pointer,
        slab2: cute.Pointer,
        slab3: cute.Pointer,
        slab4: cute.Pointer,
        slab5: cute.Pointer,
        slab6: cute.Pointer,
        slab7: cute.Pointer,
        slab8: cute.Pointer,
        slab9: cute.Pointer,
        slab10: cute.Pointer,
        slab11: cute.Pointer,
        slab12: cute.Pointer,
        slab13: cute.Pointer,
        slab14: cute.Pointer,
        slab15: cute.Pointer,
        input_ptr: cute.Pointer,
        output_ptr: cute.Pointer,
        stage_offset: Int64,
        partial_offset: Int64,
        final_offset: Int64,
        stage1_offset: Int64,
        partial1_offset: Int64,
        final1_offset: Int64,
        elements: Int64,
        blocks: Int32,
        stream: cuda.CUstream,
    ) -> None:
        if cutlass.const_expr(self._deferred_consumption):
            self.deferred_preamble(
                slab0,
                slab1,
                slab2,
                slab3,
                slab4,
                slab5,
                slab6,
                slab7,
                slab8,
                slab9,
                slab10,
                slab11,
                slab12,
                slab13,
                slab14,
                slab15,
            ).launch(
                grid=(1, 1, 1),
                block=(32, 1, 1),
                cluster=(1, 1, 1),
                stream=stream,
            )
        self.kernel(
            slab0,
            slab1,
            slab2,
            slab3,
            slab4,
            slab5,
            slab6,
            slab7,
            slab8,
            slab9,
            slab10,
            slab11,
            slab12,
            slab13,
            slab14,
            slab15,
            input_ptr,
            output_ptr,
            stage_offset,
            partial_offset,
            final_offset,
            stage1_offset,
            partial1_offset,
            final1_offset,
            elements,
        ).launch(
            grid=(blocks, 1, 1),
            block=[self._threads, 1, 1],
            cluster=(1, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def deferred_preamble(
        self,
        slab0: cute.Pointer,
        slab1: cute.Pointer,
        slab2: cute.Pointer,
        slab3: cute.Pointer,
        slab4: cute.Pointer,
        slab5: cute.Pointer,
        slab6: cute.Pointer,
        slab7: cute.Pointer,
        slab8: cute.Pointer,
        slab9: cute.Pointer,
        slab10: cute.Pointer,
        slab11: cute.Pointer,
        slab12: cute.Pointer,
        slab13: cute.Pointer,
        slab14: cute.Pointer,
        slab15: cute.Pointer,
    ) -> None:
        slabs = (
            slab0,
            slab1,
            slab2,
            slab3,
            slab4,
            slab5,
            slab6,
            slab7,
            slab8,
            slab9,
            slab10,
            slab11,
            slab12,
            slab13,
            slab14,
            slab15,
        )
        tidx, _, _ = cute.arch.thread_idx()
        if Int32(tidx) == Int32(0):
            self_base = Int64(slabs[self._rank].toint())
            generation = _atomic_add_global_u32(
                self_base + Int64(_COLLECTIVE_GENERATION), Uint32(1)
            ) + Uint32(1)
            if cutlass.const_expr(self._local_rank != 0):
                leader_base = Int64(slabs[self._leader_rank].toint())
                _store_release_sys_u32(
                    _flag_address(
                        leader_base,
                        _LOCAL_CONSUMED,
                        Int32(0),
                        Int32(self._local_rank),
                    ),
                    generation,
                )
            else:
                for peer_island in cutlass.range_constexpr(_MAX_ISLANDS):
                    if cutlass.const_expr(peer_island < self._num_islands):
                        if cutlass.const_expr(peer_island != self._island):
                            peer_leader = peer_island * _ISLAND_SIZE
                            _store_release_sys_u32(
                                _flag_address(
                                    Int64(slabs[peer_leader].toint()),
                                    _LEADER_CONSUMED,
                                    Int32(0),
                                    Int32(self._island),
                                ),
                                generation,
                            )
                for peer_island in cutlass.range_constexpr(_MAX_ISLANDS):
                    if cutlass.const_expr(peer_island < self._num_islands):
                        if cutlass.const_expr(peer_island != self._island):
                            _wait_for(
                                _flag_address(
                                    self_base,
                                    _LEADER_CONSUMED,
                                    Int32(0),
                                    Int32(peer_island),
                                ),
                                generation,
                                self._wait_nanosleep_cycles,
                            )
                for peer in cutlass.range_constexpr(1, self._local_size):
                    _wait_for(
                        _flag_address(
                            self_base,
                            _LOCAL_CONSUMED,
                            Int32(0),
                            Int32(peer),
                        ),
                        generation,
                        self._wait_nanosleep_cycles,
                    )

    @cute.kernel
    def kernel(
        self,
        slab0: cute.Pointer,
        slab1: cute.Pointer,
        slab2: cute.Pointer,
        slab3: cute.Pointer,
        slab4: cute.Pointer,
        slab5: cute.Pointer,
        slab6: cute.Pointer,
        slab7: cute.Pointer,
        slab8: cute.Pointer,
        slab9: cute.Pointer,
        slab10: cute.Pointer,
        slab11: cute.Pointer,
        slab12: cute.Pointer,
        slab13: cute.Pointer,
        slab14: cute.Pointer,
        slab15: cute.Pointer,
        input_ptr: cute.Pointer,
        output_ptr: cute.Pointer,
        stage_offset: Int64,
        partial_offset: Int64,
        final_offset: Int64,
        stage1_offset: Int64,
        partial1_offset: Int64,
        final1_offset: Int64,
        elements: Int64,
    ) -> None:
        # Keep every slab address in the kernel parameter bank, matching the
        # native by-value Runtime ABI.  Rank is immutable for a channel and is
        # therefore a compile specialization: all tuple indexing below is
        # constexpr and never falls back to a device-side pointer table.
        slabs = (
            slab0,
            slab1,
            slab2,
            slab3,
            slab4,
            slab5,
            slab6,
            slab7,
            slab8,
            slab9,
            slab10,
            slab11,
            slab12,
            slab13,
            slab14,
            slab15,
        )
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        gdim, _, _ = cute.arch.grid_dim()
        self_base = Int64(slabs[self._rank].toint())

        allocator = cutlass.utils.SmemAllocator()
        shared_generation = allocator.allocate_tensor(
            element_type=cutlass.Uint32,
            layout=cute.make_layout((1,)),
            byte_alignment=4,
        )
        if tidx == Int32(0):
            shared_generation[0] = _atomic_add_global_u32(
                self_base + Int64(bidx) * Int64(4), Uint32(1)
            ) + Uint32(1)
        cute.arch.barrier()
        generation = Uint32(shared_generation[0])
        if cutlass.const_expr(self._double_buffered):
            if generation % Uint32(2) != Uint32(0):
                stage_offset = stage1_offset
                partial_offset = partial1_offset
                final_offset = final1_offset

        self_stage = cute.make_ptr(
            cutlass.BFloat16,
            self_base + stage_offset,
            cute.AddressSpace.gmem,
            assumed_align=16,
        )
        stride = Int64(gdim) * Int64(self._threads)
        if cutlass.const_expr(self._vectorized_bf16x2):
            input_words = cute.recast_ptr(input_ptr.align(4), dtype=Uint32)
            stage_words = cute.recast_ptr(self_stage.align(4), dtype=Uint32)
            pairs = elements // Int64(2)
            index = Int64(bidx) * Int64(self._threads) + Int64(tidx)
            while index < pairs:
                stage_words[index] = input_words[index]
                index += stride
            if (
                elements % Int64(2) != Int64(0)
                and Int32(bidx) == Int32(0)
                and Int32(tidx) == Int32(0)
            ):
                self_stage[elements - Int64(1)] = input_ptr[
                    elements - Int64(1)
                ]
        else:
            index = Int64(bidx) * Int64(self._threads) + Int64(tidx)
            while index < elements:
                self_stage[index] = input_ptr[index]
                index += stride
        cute.arch.barrier()

        if tidx == Int32(0):
            _fence_sc_sys()
            if cutlass.const_expr(self._local_rank != 0):
                leader_base = Int64(slabs[self._leader_rank].toint())
                _store_release_sys_u32(
                    _flag_address(
                        leader_base,
                        _LOCAL_ARRIVED,
                        bidx,
                        Int32(self._local_rank),
                    ),
                    generation,
                )

        if cutlass.const_expr(self._local_rank != 0):
            if tidx == Int32(0):
                _wait_for(
                    self_base
                    + Int64(_FINAL_READY)
                    + Int64(bidx) * Int64(128),
                    generation,
                    self._wait_nanosleep_cycles,
                )
            cute.arch.barrier()

            leader_base = Int64(slabs[self._leader_rank].toint())
            leader_final = cute.make_ptr(
                cutlass.BFloat16,
                leader_base + final_offset,
                cute.AddressSpace.gmem,
                assumed_align=16,
            )
            if cutlass.const_expr(self._vectorized_bf16x2):
                leader_words = cute.recast_ptr(
                    leader_final.align(4), dtype=Uint32
                )
                output_words = cute.recast_ptr(
                    output_ptr.align(4), dtype=Uint32
                )
                pairs = elements // Int64(2)
                index = Int64(bidx) * Int64(self._threads) + Int64(tidx)
                while index < pairs:
                    output_words[index] = leader_words[index]
                    index += stride
                if (
                    elements % Int64(2) != Int64(0)
                    and Int32(bidx) == Int32(0)
                    and Int32(tidx) == Int32(0)
                ):
                    output_ptr[elements - Int64(1)] = leader_final[
                        elements - Int64(1)
                    ]
            else:
                index = Int64(bidx) * Int64(self._threads) + Int64(tidx)
                while index < elements:
                    output_ptr[index] = leader_final[index]
                    index += stride
            if cutlass.const_expr(
                not self._double_buffered and not self._deferred_consumption
            ):
                cute.arch.barrier()
                if tidx == Int32(0):
                    _fence_sc_sys()
                    _store_release_sys_u32(
                        _flag_address(
                            leader_base,
                            _LOCAL_CONSUMED,
                            bidx,
                            Int32(self._local_rank),
                        ),
                        generation,
                    )
        else:
            if tidx == Int32(0):
                for peer in cutlass.range_constexpr(1, self._local_size):
                    _wait_for(
                        _flag_address(
                            self_base,
                            _LOCAL_ARRIVED,
                            bidx,
                            Int32(peer),
                        ),
                        generation,
                        self._wait_nanosleep_cycles,
                    )
            cute.arch.barrier()

            self_partial = cute.make_ptr(
                cutlass.Float32,
                self_base + partial_offset,
                cute.AddressSpace.gmem,
                assumed_align=16,
            )
            if cutlass.const_expr(self._vectorized_bf16x2):
                pairs = elements // Int64(2)
                index = Int64(bidx) * Int64(self._threads) + Int64(tidx)
                while index < pairs:
                    total_lo = Float32(0.0)
                    total_hi = Float32(0.0)
                    for peer in cutlass.range_constexpr(self._local_size):
                        peer_rank = self._leader_rank + peer
                        peer_stage = cute.make_ptr(
                            Uint32,
                            Int64(slabs[peer_rank].toint()) + stage_offset,
                            cute.AddressSpace.gmem,
                            assumed_align=16,
                        )
                        lo, hi = unpack_bf16x2(peer_stage[index])
                        total_lo += lo
                        total_hi += hi
                    self_partial[index * Int64(2)] = total_lo
                    self_partial[index * Int64(2) + Int64(1)] = total_hi
                    index += stride
                if (
                    elements % Int64(2) != Int64(0)
                    and Int32(bidx) == Int32(0)
                    and Int32(tidx) == Int32(0)
                ):
                    tail = elements - Int64(1)
                    total = Float32(0.0)
                    for peer in cutlass.range_constexpr(self._local_size):
                        peer_rank = self._leader_rank + peer
                        peer_stage = cute.make_ptr(
                            cutlass.BFloat16,
                            Int64(slabs[peer_rank].toint()) + stage_offset,
                            cute.AddressSpace.gmem,
                            assumed_align=16,
                        )
                        total += Float32(peer_stage[tail])
                    self_partial[tail] = total
            else:
                index = Int64(bidx) * Int64(self._threads) + Int64(tidx)
                while index < elements:
                    total = Float32(0.0)
                    for peer in cutlass.range_constexpr(self._local_size):
                        peer_rank = self._leader_rank + peer
                        peer_stage = cute.make_ptr(
                            cutlass.BFloat16,
                            Int64(slabs[peer_rank].toint()) + stage_offset,
                            cute.AddressSpace.gmem,
                            assumed_align=16,
                        )
                        total = total + Float32(peer_stage[index])
                    self_partial[index] = total
                    index += stride
            cute.arch.barrier()

            if tidx == Int32(0):
                _fence_sc_sys()
                for peer_island in cutlass.range_constexpr(_MAX_ISLANDS):
                    if cutlass.const_expr(peer_island < self._num_islands):
                        if cutlass.const_expr(peer_island != self._island):
                            peer_leader = peer_island * _ISLAND_SIZE
                            _store_release_sys_u32(
                                _flag_address(
                                    Int64(slabs[peer_leader].toint()),
                                    _LEADER_READY,
                                    bidx,
                                    Int32(self._island),
                                ),
                                generation,
                            )
                for peer_island in cutlass.range_constexpr(_MAX_ISLANDS):
                    if cutlass.const_expr(peer_island < self._num_islands):
                        if cutlass.const_expr(peer_island != self._island):
                            _wait_for(
                                _flag_address(
                                    self_base,
                                    _LEADER_READY,
                                    bidx,
                                    Int32(peer_island),
                                ),
                                generation,
                                self._wait_nanosleep_cycles,
                            )
            cute.arch.barrier()

            self_final = cute.make_ptr(
                cutlass.BFloat16,
                self_base + final_offset,
                cute.AddressSpace.gmem,
                assumed_align=16,
            )
            if cutlass.const_expr(self._vectorized_bf16x2):
                final_words = cute.recast_ptr(
                    self_final.align(4), dtype=Uint32
                )
                output_words = cute.recast_ptr(
                    output_ptr.align(4), dtype=Uint32
                )
                pairs = elements // Int64(2)
                index = Int64(bidx) * Int64(self._threads) + Int64(tidx)
                while index < pairs:
                    total_lo = Float32(0.0)
                    total_hi = Float32(0.0)
                    for peer_island in cutlass.range_constexpr(_MAX_ISLANDS):
                        if cutlass.const_expr(peer_island < self._num_islands):
                            peer_leader = peer_island * _ISLAND_SIZE
                            peer_partial = cute.make_ptr(
                                cutlass.Float32,
                                Int64(slabs[peer_leader].toint())
                                + partial_offset,
                                cute.AddressSpace.gmem,
                                assumed_align=16,
                            )
                            total_lo += Float32(
                                peer_partial[index * Int64(2)]
                            )
                            total_hi += Float32(
                                peer_partial[index * Int64(2) + Int64(1)]
                            )
                    value = pack_f32x2_to_bf16x2(total_lo, total_hi)
                    final_words[index] = value
                    output_words[index] = value
                    index += stride
                if (
                    elements % Int64(2) != Int64(0)
                    and Int32(bidx) == Int32(0)
                    and Int32(tidx) == Int32(0)
                ):
                    tail = elements - Int64(1)
                    total = Float32(0.0)
                    for peer_island in cutlass.range_constexpr(_MAX_ISLANDS):
                        if cutlass.const_expr(peer_island < self._num_islands):
                            peer_leader = peer_island * _ISLAND_SIZE
                            peer_partial = cute.make_ptr(
                                cutlass.Float32,
                                Int64(slabs[peer_leader].toint())
                                + partial_offset,
                                cute.AddressSpace.gmem,
                                assumed_align=16,
                            )
                            total += Float32(peer_partial[tail])
                    value = cutlass.BFloat16(total)
                    self_final[tail] = value
                    output_ptr[tail] = value
            else:
                index = Int64(bidx) * Int64(self._threads) + Int64(tidx)
                while index < elements:
                    total = Float32(0.0)
                    for peer_island in cutlass.range_constexpr(_MAX_ISLANDS):
                        if cutlass.const_expr(peer_island < self._num_islands):
                            peer_leader = peer_island * _ISLAND_SIZE
                            peer_partial = cute.make_ptr(
                                cutlass.Float32,
                                Int64(slabs[peer_leader].toint())
                                + partial_offset,
                                cute.AddressSpace.gmem,
                                assumed_align=16,
                            )
                            total = total + Float32(peer_partial[index])
                    value = cutlass.BFloat16(total)
                    self_final[index] = value
                    output_ptr[index] = value
                    index += stride
            cute.arch.barrier()

            if tidx == Int32(0):
                _fence_sc_sys()
                if cutlass.const_expr(
                    not self._double_buffered
                    and not self._deferred_consumption
                ):
                    for peer_island in cutlass.range_constexpr(_MAX_ISLANDS):
                        if cutlass.const_expr(peer_island < self._num_islands):
                            if cutlass.const_expr(peer_island != self._island):
                                peer_leader = peer_island * _ISLAND_SIZE
                                _store_release_sys_u32(
                                    _flag_address(
                                        Int64(slabs[peer_leader].toint()),
                                        _LEADER_CONSUMED,
                                        bidx,
                                        Int32(self._island),
                                    ),
                                    generation,
                                )
                for peer in cutlass.range_constexpr(1, self._local_size):
                    peer_rank = self._leader_rank + peer
                    _store_release_sys_u32(
                        Int64(slabs[peer_rank].toint())
                        + Int64(_FINAL_READY)
                        + Int64(bidx) * Int64(128),
                        generation,
                    )

                if cutlass.const_expr(
                    not self._double_buffered
                    and not self._deferred_consumption
                ):
                    for peer_island in cutlass.range_constexpr(_MAX_ISLANDS):
                        if cutlass.const_expr(peer_island < self._num_islands):
                            if cutlass.const_expr(peer_island != self._island):
                                _wait_for(
                                    _flag_address(
                                        self_base,
                                        _LEADER_CONSUMED,
                                        bidx,
                                        Int32(peer_island),
                                    ),
                                    generation,
                                    self._wait_nanosleep_cycles,
                                )
                    for peer in cutlass.range_constexpr(1, self._local_size):
                        _wait_for(
                            _flag_address(
                                self_base,
                                _LOCAL_CONSUMED,
                                bidx,
                                Int32(peer),
                            ),
                            generation,
                            self._wait_nanosleep_cycles,
                        )


@program_cache
def get_hierarchical_launcher(
    world_size: int,
    rank: int,
    device_index: int,
    *,
    threads: int = 224,
    wait_nanosleep_cycles: int = 24,
    double_buffered: bool = False,
    deferred_consumption: bool = False,
    vectorized_bf16x2: bool = False,
) -> Callable[..., None]:
    """Return the rank-specialized bounded-peer launcher for this channel."""

    del device_index  # retained in the process-local cache key
    if world_size not in (9, 10, 12, 16):
        raise ValueError(f"unsupported world size {world_size}")
    if not 0 <= rank < world_size:
        raise ValueError(f"invalid rank {rank} for world size {world_size}")
    launch = _HierarchicalLaunch(
        world_size,
        rank,
        threads=threads,
        wait_nanosleep_cycles=wait_nanosleep_cycles,
        double_buffered=double_buffered,
        deferred_consumption=deferred_consumption,
        vectorized_bf16x2=vectorized_bf16x2,
    )
    cache_key = (
        int(world_size),
        int(rank),
        int(threads),
        int(wait_nanosleep_cycles),
        bool(double_buffered),
        bool(deferred_consumption),
        bool(vectorized_bf16x2),
    )
    raise_if_kernel_resolution_frozen(
        "cute.compile", target=launch, cache_key=cache_key
    )
    slab_placeholder = make_ptr(
        cutlass.Uint8,
        256,
        cute.AddressSpace.gmem,
        assumed_align=256,
    )
    raw = b12x_compile(
        launch,
        *(slab_placeholder for _ in range(_MAX_WORLD_SIZE)),
        make_ptr(cutlass.BFloat16, 16, cute.AddressSpace.gmem, assumed_align=2),
        make_ptr(cutlass.BFloat16, 16, cute.AddressSpace.gmem, assumed_align=2),
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        current_cuda_stream(),
        compile_spec=KernelCompileSpec.from_key(
            "comm.pcie.hierarchical.bf16",
            4,
            cache_key,
            labels=(
                "world_size",
                "rank",
                "threads",
                "wait_nanosleep_cycles",
                "double_buffered",
                "deferred_consumption",
                "vectorized_bf16x2",
            ),
        ),
    )

    def run(
        slab_addresses: Sequence[int],
        input_address: int,
        output_address: int,
        stage_offset: int,
        partial_offset: int,
        final_offset: int,
        stage1_offset: int,
        partial1_offset: int,
        final1_offset: int,
        elements: int,
        blocks: int,
    ) -> None:
        """Launch the compiled hierarchical collective with runtime arguments."""
        if len(slab_addresses) != world_size:
            raise ValueError(
                f"expected {world_size} slab addresses, got {len(slab_addresses)}"
            )
        padded_slabs = tuple(int(address) for address in slab_addresses) + (
            0,
        ) * (_MAX_WORLD_SIZE - world_size)
        raw(
            *(
                make_ptr(
                    cutlass.Uint8,
                    address,
                    cute.AddressSpace.gmem,
                    assumed_align=256,
                )
                for address in padded_slabs
            ),
            make_ptr(
                cutlass.BFloat16,
                input_address,
                cute.AddressSpace.gmem,
                assumed_align=2,
            ),
            make_ptr(
                cutlass.BFloat16,
                output_address,
                cute.AddressSpace.gmem,
                assumed_align=2,
            ),
            stage_offset,
            partial_offset,
            final_offset,
            stage1_offset,
            partial1_offset,
            final1_offset,
            elements,
            blocks,
            current_cuda_stream(),
        )

    return attach_programs(run, raw)


__all__ = ["get_hierarchical_launcher"]
