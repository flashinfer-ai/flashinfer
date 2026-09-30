"""CuTeDSL kernel for bounded-degree vocabulary-parallel greedy sampling."""

from __future__ import annotations

from b12x._lib.program_cache import program_cache
from collections.abc import Callable, Sequence

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64, Uint32, Uint64
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from b12x._lib.compiler import KernelCompileSpec
from b12x._lib.compiler import compile as b12x_compile
from b12x._lib.compile_plan import attach_programs
from b12x._lib.runtime_control import raise_if_kernel_resolution_frozen
from b12x._lib.utils import current_cuda_stream, make_ptr


ISLAND_SIZE = 4
MAX_ISLANDS = 4
MAX_WORLD_SIZE = ISLAND_SIZE * MAX_ISLANDS
MAX_BATCH_SIZE = 8
THREADS = 512
SLOTS = 2

# Every protocol flag occupies an independent 128-byte system-scope cache line.
_GENERATION_OFFSET = 0
_LOCAL_READY_OFFSET = 256
_LOCAL_READY_BYTES = SLOTS * MAX_BATCH_SIZE * ISLAND_SIZE * 128
_ISLAND_READY_OFFSET = _LOCAL_READY_OFFSET + _LOCAL_READY_BYTES
_ISLAND_READY_BYTES = SLOTS * MAX_BATCH_SIZE * MAX_ISLANDS * 128
_LOCAL_KEY_OFFSET = _ISLAND_READY_OFFSET + _ISLAND_READY_BYTES
_ISLAND_KEY_OFFSET = _LOCAL_KEY_OFFSET + SLOTS * MAX_BATCH_SIZE * 8
SLAB_BYTES = _ISLAND_KEY_OFFSET + SLOTS * MAX_BATCH_SIZE * 8


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


@dsl_user_op
def _score_order_key(value: Float32, *, loc=None, ip=None) -> Uint32:
    """Map an FP32 score to the exact greedy-sampling comparison order.

    NaNs dominate finite values, positive and negative zero compare equal, and
    unsigned integer order matches numeric order for every remaining value.
    """

    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Float32(value).ir_value(loc=loc, ip=ip)],
            """
            {
                .reg .pred is_nan, is_zero;
                .reg .u32 bits, mask;
                mov.b32 bits, $1;
                shr.u32 mask, bits, 31;
                mul.lo.u32 mask, mask, 0x7fffffff;
                or.b32 mask, mask, 0x80000000;
                xor.b32 $0, bits, mask;
                setp.eq.f32 is_zero, $1, 0f00000000;
                @is_zero mov.u32 $0, 0x80000000;
                setp.nan.f32 is_nan, $1, $1;
                @is_nan mov.u32 $0, 0xffffffff;
            }
            """,
            "=r,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _warp_max_u64(value: Uint64, *, loc=None, ip=None) -> Uint64:
    """Return the warp-wide unsigned maximum of a packed 64-bit key."""

    return Uint64(
        llvm.inline_asm(
            T.i64(),
            [Uint64(value).ir_value(loc=loc, ip=ip)],
            """
            {
                .reg .pred high_matches;
                .reg .u32 low, high, max_high, candidate_low, max_low;
                mov.b64 {low, high}, $1;
                redux.sync.max.u32 max_high, high, 0xffffffff;
                setp.eq.u32 high_matches, high, max_high;
                selp.u32 candidate_low, low, 0, high_matches;
                redux.sync.max.u32 max_low, candidate_low, 0xffffffff;
                mov.b64 $0, {max_low, max_high};
            }
            """,
            "=l,l",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@cute.jit
def _wait_for(address: Int64, generation: Uint32, cycles: int) -> None:
    observed = _load_acquire_sys_u32(address)
    while observed != generation:
        if cutlass.const_expr(cycles > 0):
            _nanosleep(Uint32(cycles))
        observed = _load_acquire_sys_u32(address)


@cute.jit
def _flag_address(
    base: Int64,
    region: int,
    slot: Int32,
    row: Int32,
    index: Int32,
) -> Int64:
    return (
        base
        + Int64(region)
        + (
            (
                Int64(slot) * Int64(MAX_BATCH_SIZE)
                + Int64(row)
            )
            * Int64(ISLAND_SIZE)
            + Int64(index)
        )
        * Int64(128)
    )


@cute.jit
def _key_address(base: Int64, region: int, slot: Int32, row: Int32) -> Int64:
    return (
        base
        + Int64(region)
        + (
            Int64(slot) * Int64(MAX_BATCH_SIZE) + Int64(row)
        )
        * Int64(8)
    )


@cute.jit
def _pack_key(score: Float32, global_index: Int32) -> Uint64:
    return (
        Uint64(_score_order_key(score)) << Uint64(32)
    ) | (Uint64(0xFFFFFFFF) - Uint64(global_index))


@cute.jit
def _key_index(key: Uint64) -> Int64:
    return Int64(Uint64(0xFFFFFFFF) - (key & Uint64(0xFFFFFFFF)))


class _VocabArgmaxLaunch:
    def __init__(
        self,
        world_size: int,
        rank: int,
        *,
        wait_nanosleep_cycles: int,
    ) -> None:
        self._world_size = int(world_size)
        self._num_islands = self._world_size // ISLAND_SIZE
        self._rank = int(rank)
        self._island = self._rank // ISLAND_SIZE
        self._local_rank = self._rank % ISLAND_SIZE
        self._wait_nanosleep_cycles = int(wait_nanosleep_cycles)
        if self._world_size not in (8, 12, 16):
            raise ValueError(f"unsupported world size {self._world_size}")
        if not 0 <= self._rank < self._world_size:
            raise ValueError(
                f"invalid rank {self._rank} for world size {self._world_size}"
            )
        if not 0 <= self._wait_nanosleep_cycles <= 1024:
            raise ValueError("wait_nanosleep_cycles must be in [0, 1024]")

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
        base: cute.Pointer,
        bias: cute.Pointer,
        output: cute.Pointer,
        local_elements: Int64,
        base_row_stride: Int64,
        bias_row_stride: Int64,
        batch: Int32,
        stream: cuda.CUstream,
    ) -> None:
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
            base,
            bias,
            output,
            local_elements,
            base_row_stride,
            bias_row_stride,
        ).launch(
            grid=(batch, 1, 1),
            block=(THREADS, 1, 1),
            cluster=(1, 1, 1),
            stream=stream,
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
        base: cute.Pointer,
        bias: cute.Pointer,
        output: cute.Pointer,
        local_elements: Int64,
        base_row_stride: Int64,
        bias_row_stride: Int64,
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
        row, _, _ = cute.arch.block_idx()
        lane = Int32(tidx) % Int32(32)
        warp = Int32(tidx) // Int32(32)

        initial = Uint64(_score_order_key(Float32(float("-inf")))) << Uint64(32)
        best = initial
        index = Int64(tidx)
        base_offset = Int64(row) * base_row_stride
        bias_offset = Int64(row) * bias_row_stride
        while index < local_elements:
            value = Float32(
                cutlass.BFloat16(
                    Float32(base[base_offset + index])
                    + Float32(bias[bias_offset + index])
                )
            )
            global_index = (
                Int64(self._rank) * local_elements + index
            )
            candidate = _pack_key(value, Int32(global_index))
            best = cutlass.max(best, candidate)
            index += Int64(THREADS)
        best = _warp_max_u64(best)

        allocator = cutlass.utils.SmemAllocator()
        warp_keys = allocator.allocate_tensor(
            element_type=Uint64,
            layout=cute.make_layout((THREADS // 32,)),
            byte_alignment=16,
        )
        shared_generation = allocator.allocate_tensor(
            element_type=Uint32,
            layout=cute.make_layout((1,)),
            byte_alignment=4,
        )
        if lane == Int32(0):
            warp_keys[warp] = best
        cute.arch.barrier()

        block_key = initial
        if warp == Int32(0):
            if lane < Int32(THREADS // 32):
                block_key = warp_keys[lane]
            block_key = _warp_max_u64(block_key)
        self_base = Int64(slabs[self._rank].toint())
        if tidx == Int32(0):
            generation = _atomic_add_global_u32(
                self_base + Int64(row) * Int64(4), Uint32(1)
            ) + Uint32(1)
            shared_generation[0] = generation
        cute.arch.barrier()

        if tidx == Int32(0):
            generation = Uint32(shared_generation[0])
            slot = Int32(generation & Uint32(1))
            local_key_ptr = cute.make_ptr(
                Uint64,
                _key_address(self_base, _LOCAL_KEY_OFFSET, slot, Int32(row)),
                cute.AddressSpace.gmem,
                assumed_align=8,
            )
            local_key_ptr[0] = block_key
            _fence_sc_sys()

            island_base_rank = self._island * ISLAND_SIZE
            for peer_local in cutlass.range_constexpr(ISLAND_SIZE):
                if cutlass.const_expr(peer_local != self._local_rank):
                    peer_rank = island_base_rank + peer_local
                    _store_release_sys_u32(
                        _flag_address(
                            Int64(slabs[peer_rank].toint()),
                            _LOCAL_READY_OFFSET,
                            slot,
                            Int32(row),
                            Int32(self._local_rank),
                        ),
                        generation,
                    )
            for peer_local in cutlass.range_constexpr(ISLAND_SIZE):
                if cutlass.const_expr(peer_local != self._local_rank):
                    _wait_for(
                        _flag_address(
                            self_base,
                            _LOCAL_READY_OFFSET,
                            slot,
                            Int32(row),
                            Int32(peer_local),
                        ),
                        generation,
                        self._wait_nanosleep_cycles,
                    )

            island_key = initial
            for peer_local in cutlass.range_constexpr(ISLAND_SIZE):
                peer_rank = island_base_rank + peer_local
                peer_key_ptr = cute.make_ptr(
                    Uint64,
                    _key_address(
                        Int64(slabs[peer_rank].toint()),
                        _LOCAL_KEY_OFFSET,
                        slot,
                        Int32(row),
                    ),
                    cute.AddressSpace.gmem,
                    assumed_align=8,
                )
                island_key = cutlass.max(island_key, peer_key_ptr[0])
            island_key_ptr = cute.make_ptr(
                Uint64,
                _key_address(self_base, _ISLAND_KEY_OFFSET, slot, Int32(row)),
                cute.AddressSpace.gmem,
                assumed_align=8,
            )
            island_key_ptr[0] = island_key
            _fence_sc_sys()

            for peer_island in cutlass.range_constexpr(MAX_ISLANDS):
                if cutlass.const_expr(peer_island < self._num_islands):
                    if cutlass.const_expr(peer_island != self._island):
                        peer_rank = peer_island * ISLAND_SIZE + self._local_rank
                        _store_release_sys_u32(
                            _flag_address(
                                Int64(slabs[peer_rank].toint()),
                                _ISLAND_READY_OFFSET,
                                slot,
                                Int32(row),
                                Int32(self._island),
                            ),
                            generation,
                        )
            for peer_island in cutlass.range_constexpr(MAX_ISLANDS):
                if cutlass.const_expr(peer_island < self._num_islands):
                    if cutlass.const_expr(peer_island != self._island):
                        _wait_for(
                            _flag_address(
                                self_base,
                                _ISLAND_READY_OFFSET,
                                slot,
                                Int32(row),
                                Int32(peer_island),
                            ),
                            generation,
                            self._wait_nanosleep_cycles,
                        )

            global_key = initial
            for peer_island in cutlass.range_constexpr(MAX_ISLANDS):
                if cutlass.const_expr(peer_island < self._num_islands):
                    peer_rank = peer_island * ISLAND_SIZE + self._local_rank
                    peer_key_ptr = cute.make_ptr(
                        Uint64,
                        _key_address(
                            Int64(slabs[peer_rank].toint()),
                            _ISLAND_KEY_OFFSET,
                            slot,
                            Int32(row),
                        ),
                        cute.AddressSpace.gmem,
                        assumed_align=8,
                    )
                    global_key = cutlass.max(global_key, peer_key_ptr[0])
            output[Int64(row)] = _key_index(global_key)


@program_cache
def get_vocab_argmax_launcher(
    world_size: int,
    rank: int,
    device_index: int,
    *,
    wait_nanosleep_cycles: int,
) -> Callable[..., None]:
    """Return a rank-specialized launcher for TP8, TP12, or TP16."""

    del device_index  # retained in the process-local cache key
    launch = _VocabArgmaxLaunch(
        world_size,
        rank,
        wait_nanosleep_cycles=wait_nanosleep_cycles,
    )
    cache_key = (
        int(world_size),
        int(rank),
        int(wait_nanosleep_cycles),
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
        *(slab_placeholder for _ in range(MAX_WORLD_SIZE)),
        make_ptr(cutlass.BFloat16, 16, cute.AddressSpace.gmem, assumed_align=2),
        make_ptr(cutlass.BFloat16, 16, cute.AddressSpace.gmem, assumed_align=2),
        make_ptr(cutlass.Int64, 16, cute.AddressSpace.gmem, assumed_align=8),
        1,
        1,
        1,
        1,
        current_cuda_stream(),
        compile_spec=KernelCompileSpec.from_key(
            "comm.pcie.vocab_argmax.bf16",
            1,
            cache_key,
            labels=("world_size", "rank", "wait_nanosleep_cycles"),
        ),
    )

    def run(
        slab_addresses: Sequence[int],
        base_address: int,
        bias_address: int,
        output_address: int,
        local_elements: int,
        base_row_stride: int,
        bias_row_stride: int,
        batch: int,
    ) -> None:
        if len(slab_addresses) != world_size:
            raise ValueError(
                f"expected {world_size} slab addresses, got {len(slab_addresses)}"
            )
        padded_slabs = tuple(int(address) for address in slab_addresses) + (
            0,
        ) * (MAX_WORLD_SIZE - world_size)
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
                base_address,
                cute.AddressSpace.gmem,
                assumed_align=2,
            ),
            make_ptr(
                cutlass.BFloat16,
                bias_address,
                cute.AddressSpace.gmem,
                assumed_align=2,
            ),
            make_ptr(
                cutlass.Int64,
                output_address,
                cute.AddressSpace.gmem,
                assumed_align=8,
            ),
            local_elements,
            base_row_stride,
            bias_row_stride,
            batch,
            current_cuda_stream(),
        )

    return attach_programs(run, raw)


__all__ = ["SLAB_BYTES", "get_vocab_argmax_launcher"]
