"""CuTe DSL implementations of the PCIe oneshot collectives.

The launch ABI uses stable device-resident ``uint64`` pointer tables. Payload
transfers use aligned 16-byte records. TP2 graph peer-push records carry their
generation beside the payload so readiness does not depend on a separate peer
atomic operation.
"""

from __future__ import annotations

import functools
from collections.abc import Callable

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64, Uint32

from b12x._lib.compiler import KernelCompileSpec
from b12x._lib.compiler import compile as b12x_compile
from b12x._lib.compile_plan import attach_programs
from b12x._lib.intrinsics import (
    ld_global_v4_u32,
    st_global_v4_f32,
    st_global_v4_u32,
    u32_as_f32,
)
from b12x._lib.runtime_control import raise_if_kernel_resolution_frozen
from b12x._lib.utils import current_cuda_stream, make_ptr

from ._cute_intrinsics import (
    add_bf16x2,
    add_f16x2,
    atomic_add_global_u32,
    f32_as_u32,
    graph_epoch_arrive,
    plain_graph_epoch_arrive,
    ld_global_f32,
    ld_relaxed_gpu_u32,
    ld_relaxed_sys_global_v4_u32,
    pack_f32x2_to_bf16x2,
    pack_f32x2_to_f16x2,
    pcie_arrive_and_wait,
    rsqrt_approx_f32,
    spin_until_changed_acquire_gpu,
    st_global_f32,
    st_global_u32,
    st_relaxed_sys_global_v4_u32,
    threadfence_gpu,
    unpack_bf16x2,
    unpack_f16x2,
)


_MAX_BLOCKS = 36
_MAX_RANKS = 16
_FLAG_STRIDE = 32
_SELF_COUNTER_BYTES = _MAX_BLOCKS * _MAX_RANKS * 4
_PEER_SLOT_BYTES = _MAX_BLOCKS * _MAX_RANKS * _FLAG_STRIDE * 4
_PEER_COUNTER_BYTES = 2 * _PEER_SLOT_BYTES
# Words one through four of the first padded peer-flag record are unused by
# the barrier layout. Graph slot counters live there without changing signal
# allocation; plain peer-push has a separate lifetime from pull and fused ops.
_GRAPH_EPOCH_OFFSET = _SELF_COUNTER_BYTES + 4
_GRAPH_ARRIVED_OFFSET = _SELF_COUNTER_BYTES + 8
_PLAIN_GRAPH_EPOCH_OFFSET = _SELF_COUNTER_BYTES + 12
_PLAIN_GRAPH_ARRIVED_OFFSET = _SELF_COUNTER_BYTES + 16
_PLAIN_GRAPH_LAST_EPOCH = 0xFFFFFFFD
_RMS_ARRIVE_OFFSET = _SELF_COUNTER_BYTES + _PEER_COUNTER_BYTES
_RMS_GEN_OFFSET = 150_016
_RMS_PARTIAL_OFFSET = 150_272
_REG_PACKS = 3

_DTYPE_PACK_ELEMS = {"float32": 4, "float16": 8, "bfloat16": 8}

# CUDA graph capture must never be the first call that compiles or loads a
# specialization.  These process-local keys record successful launcher
# construction; host runtimes query them before entering the capture path.
_PREPARED_ONESHOT_LAUNCHERS: set[tuple[object, ...]] = set()
_PREPARED_FUSED_ONESHOT_LAUNCHERS: set[tuple[object, ...]] = set()


class _PackedMath:
    def __init__(self, dtype_name: str) -> None:
        if dtype_name not in _DTYPE_PACK_ELEMS:
            raise ValueError(f"unsupported oneshot dtype {dtype_name!r}")
        self._dtype_name = dtype_name
        self._pack_elems = _DTYPE_PACK_ELEMS[dtype_name]

    @cute.jit
    def _accumulate_words(
        self,
        accumulator: cute.Tensor,
        word0: Uint32,
        word1: Uint32,
        word2: Uint32,
        word3: Uint32,
        initialize: cutlass.Constexpr[bool],
    ) -> None:
        words = (word0, word1, word2, word3)
        if cutlass.const_expr(self._dtype_name == "float32"):
            for word in cutlass.range_constexpr(4):
                value = u32_as_f32(words[word])
                if cutlass.const_expr(initialize):
                    accumulator[word] = value
                else:
                    accumulator[word] = accumulator[word] + value
        else:
            for word in cutlass.range_constexpr(4):
                if cutlass.const_expr(self._dtype_name == "float16"):
                    lo, hi = unpack_f16x2(words[word])
                else:
                    lo, hi = unpack_bf16x2(words[word])
                lane = word * 2
                if cutlass.const_expr(initialize):
                    accumulator[lane] = lo
                    accumulator[lane + 1] = hi
                else:
                    accumulator[lane] = accumulator[lane] + lo
                    accumulator[lane + 1] = accumulator[lane + 1] + hi

    @cute.jit
    def _load_accumulate(
        self,
        accumulator: cute.Tensor,
        address: Int64,
        initialize: cutlass.Constexpr[bool],
    ) -> None:
        words = ld_global_v4_u32(address)
        self._accumulate_words(
            accumulator,
            words[0],
            words[1],
            words[2],
            words[3],
            initialize,
        )

    @cute.jit
    def _store_accumulator(self, address: Int64, accumulator: cute.Tensor) -> None:
        if cutlass.const_expr(self._dtype_name == "float32"):
            st_global_v4_f32(
                address,
                accumulator[0],
                accumulator[1],
                accumulator[2],
                accumulator[3],
            )
        else:
            packed = cute.make_rmem_tensor((4,), cutlass.Uint32)
            for word in cutlass.range_constexpr(4):
                lane = word * 2
                if cutlass.const_expr(self._dtype_name == "float16"):
                    packed[word] = pack_f32x2_to_f16x2(
                        accumulator[lane], accumulator[lane + 1]
                    )
                else:
                    packed[word] = pack_f32x2_to_bf16x2(
                        accumulator[lane], accumulator[lane + 1]
                    )
            st_global_v4_u32(address, packed[0], packed[1], packed[2], packed[3])

    @cute.jit
    def _copy_pack(self, source: Int64, destination: Int64) -> None:
        words = ld_global_v4_u32(source)
        st_global_v4_u32(destination, words[0], words[1], words[2], words[3])

    @cute.jit
    def _warp_reduce_down(self, value: Float32) -> Float32:
        for offset in (16, 8, 4, 2, 1):
            value = value + cute.arch.shuffle_sync_down(value, offset=offset)
        return value


class _OneshotLaunch(_PackedMath):
    def __init__(
        self,
        dtype_name: str,
        world_size: int,
        rank: int,
        stage_input: bool,
        device_slot_selection: bool,
        slot_bias: int,
        transport: str,
        threads: int,
    ) -> None:
        super().__init__(dtype_name)
        if transport not in (
            "pull",
            "tp2_remote_push",
            "tp2_remote_push_stream",
        ):
            raise ValueError(f"invalid plain oneshot transport {transport!r}")
        if transport.startswith("tp2_remote_push") and (
            world_size != 2
            or not stage_input
            or not device_slot_selection
            or dtype_name not in ("float16", "bfloat16")
        ):
            raise ValueError(
                "TP2 remote push requires a graph-owned staged FP16 or BF16 "
                "input on two ranks"
            )
        self._world_size = int(world_size)
        self._rank = int(rank)
        self._stage_input = bool(stage_input)
        self._device_slot_selection = bool(device_slot_selection)
        self._slot_bias = int(slot_bias) & 1
        self._transport = transport
        self._threads = int(threads)

    @cute.jit
    def __call__(
        self,
        peer_ptrs: cute.Pointer,
        signal_ptrs: cute.Pointer,
        input_ptr: cute.Pointer,
        output_ptr: cute.Pointer,
        size_packs: Int32,
        plain_region_packs: Int64,
        plain_capacity_packs: Int64,
        plain_local_slot0: Int64,
        plain_peer_slot0: Int64,
        plain_local_slot1: Int64,
        plain_peer_slot1: Int64,
        grid_x: Int32,
        stream: cuda.CUstream,
    ) -> None:
        self.kernel(
            peer_ptrs,
            signal_ptrs,
            input_ptr,
            output_ptr,
            size_packs,
            plain_region_packs,
            plain_capacity_packs,
            plain_local_slot0,
            plain_peer_slot0,
            plain_local_slot1,
            plain_peer_slot1,
        ).launch(
            grid=(grid_x, 1, 1),
            block=[self._threads, 1, 1],
            cluster=(1, 1, 1),
            stream=stream,
        )

    @cute.jit
    def _multi_gpu_barrier(self, signal_ptrs: cute.Pointer) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        if cutlass.const_expr(self._stage_input):
            cute.arch.sync_threads()
        if tidx < Int32(self._world_size):
            self_signal = Int64(signal_ptrs[self._rank])
            peer_signal = Int64(signal_ptrs[tidx])
            self_counter = self_signal + (
                Int64(bidx) * Int64(_MAX_RANKS) + Int64(tidx)
            ) * Int64(4)
            peer_slot0 = (
                peer_signal
                + Int64(_SELF_COUNTER_BYTES)
                + (
                    Int64(bidx) * Int64(_MAX_RANKS * _FLAG_STRIDE)
                    + Int64(self._rank * _FLAG_STRIDE)
                )
                * Int64(4)
            )
            wait_slot0 = (
                self_signal
                + Int64(_SELF_COUNTER_BYTES)
                + (
                    Int64(bidx) * Int64(_MAX_RANKS * _FLAG_STRIDE)
                    + Int64(tidx) * Int64(_FLAG_STRIDE)
                )
                * Int64(4)
            )
            pcie_arrive_and_wait(
                self_counter,
                peer_slot0,
                wait_slot0,
                Int64(_PEER_SLOT_BYTES),
            )
        cute.arch.sync_threads()

    @cute.kernel
    def kernel(
        self,
        peer_ptrs: cute.Pointer,
        signal_ptrs: cute.Pointer,
        input_ptr: cute.Pointer,
        output_ptr: cute.Pointer,
        size_packs: Int32,
        plain_region_packs: Int64,
        plain_capacity_packs: Int64,
        plain_local_slot0: Int64,
        plain_peer_slot0: Int64,
        plain_local_slot1: Int64,
        plain_peer_slot1: Int64,
    ) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        gdim, _, _ = cute.arch.grid_dim()
        input_base = Int64(input_ptr.toint())
        output_base = Int64(output_ptr.toint())
        generation = Uint32(0)
        slot = Uint32(0)
        if cutlass.const_expr(self._device_slot_selection):
            if cutlass.const_expr(self._transport == "pull"):
                generation = ld_relaxed_gpu_u32(
                    Int64(signal_ptrs[self._rank]) + Int64(_GRAPH_EPOCH_OFFSET)
                )
            else:
                generation = ld_relaxed_gpu_u32(
                    Int64(signal_ptrs[self._rank]) + Int64(_PLAIN_GRAPH_EPOCH_OFFSET)
                )
            slot = (generation + Uint32(self._slot_bias)) % Uint32(2)
            if cutlass.const_expr(self._transport == "pull"):
                peer_ptrs = peer_ptrs + Int64(slot) * Int64(_MAX_RANKS)
            else:
                # All threads retain the epoch before the final CTA advances it.
                cute.arch.sync_threads()
                if Int32(tidx) == Int32(0):
                    self_signal = Int64(signal_ptrs[self._rank])
                    plain_graph_epoch_arrive(
                        self_signal + Int64(_PLAIN_GRAPH_EPOCH_OFFSET),
                        self_signal + Int64(_PLAIN_GRAPH_ARRIVED_OFFSET),
                        Uint32(gdim),
                    )

        index = Int32(bidx) * Int32(self._threads) + Int32(tidx)
        stride = Int32(gdim) * Int32(self._threads)
        if cutlass.const_expr(self._transport == "pull"):
            if cutlass.const_expr(self._stage_input):
                staging_base = Int64(peer_ptrs[self._rank])
                stage_index = index
                while stage_index < size_packs:
                    self._copy_pack(
                        input_base + Int64(stage_index) * Int64(16),
                        staging_base + Int64(stage_index) * Int64(16),
                    )
                    stage_index += stride

            self._multi_gpu_barrier(signal_ptrs)

            while index < size_packs:
                accumulator = cute.make_rmem_tensor(
                    (self._pack_elems,), cutlass.Float32
                )
                for peer_index in cutlass.range_constexpr(self._world_size):
                    peer_rank = (self._rank + peer_index) % self._world_size
                    peer_base = Int64(peer_ptrs[peer_rank])
                    self._load_accumulate(
                        accumulator,
                        peer_base + Int64(index) * Int64(16),
                        peer_index == 0,
                    )
                self._store_accumulator(
                    output_base + Int64(index) * Int64(16), accumulator
                )
                index += stride

            if cutlass.const_expr(self._device_slot_selection):
                if Int32(tidx) == Int32(0):
                    self_signal = Int64(signal_ptrs[self._rank])
                    graph_epoch_arrive(
                        self_signal + Int64(_GRAPH_EPOCH_OFFSET),
                        self_signal + Int64(_GRAPH_ARRIVED_OFFSET),
                        Uint32(gdim),
                    )
        elif cutlass.const_expr(self._transport.startswith("tp2_remote_push")):
            peer_rank = self._rank ^ 1
            local_scratch_base = plain_local_slot0
            peer_scratch_base = plain_peer_slot0
            if slot != Uint32(0):
                local_scratch_base = plain_local_slot1
                peer_scratch_base = plain_peer_slot1
            peer_write_base = peer_scratch_base + plain_region_packs * Int64(16)
            local_poll_base = local_scratch_base + plain_region_packs * Int64(16)
            # Wire tags zero and one are reserved. Before the epoch restarts,
            # both ranks drain and clear the full incoming capacity, including
            # records not touched by this shape.
            expected_generation = generation + Uint32(2)

            if cutlass.const_expr(self._transport == "tp2_remote_push"):
                publish_index = index
                while publish_index < size_packs:
                    words = ld_global_v4_u32(
                        input_base + Int64(publish_index) * Int64(16)
                    )
                    # Match NCCL LL publication: each naturally aligned
                    # 16-byte line carries eight payload bytes and two copies
                    # of the generation. The protocol does not assume that the
                    # line is atomically visible; repeated generation lanes let
                    # the receiver reject a torn observation.
                    line_base = peer_write_base + Int64(publish_index) * Int64(32)
                    st_relaxed_sys_global_v4_u32(
                        line_base,
                        words[0],
                        expected_generation,
                        words[1],
                        expected_generation,
                    )
                    st_relaxed_sys_global_v4_u32(
                        line_base + Int64(16),
                        words[2],
                        expected_generation,
                        words[3],
                        expected_generation,
                    )
                    publish_index += stride

            while index < size_packs:
                local_words = ld_global_v4_u32(input_base + Int64(index) * Int64(16))
                if cutlass.const_expr(self._transport == "tp2_remote_push_stream"):
                    peer_line_address = peer_write_base + Int64(index) * Int64(32)
                    st_relaxed_sys_global_v4_u32(
                        peer_line_address,
                        local_words[0],
                        expected_generation,
                        local_words[1],
                        expected_generation,
                    )
                    st_relaxed_sys_global_v4_u32(
                        peer_line_address + Int64(16),
                        local_words[2],
                        expected_generation,
                        local_words[3],
                        expected_generation,
                    )
                line_address = local_poll_base + Int64(index) * Int64(32)
                line0 = ld_relaxed_sys_global_v4_u32(line_address)
                line1 = ld_relaxed_sys_global_v4_u32(line_address + Int64(16))
                waiting = (
                    (line0[1] ^ expected_generation)
                    | (line0[3] ^ expected_generation)
                    | (line1[1] ^ expected_generation)
                    | (line1[3] ^ expected_generation)
                )
                while waiting != Uint32(0):
                    line0 = ld_relaxed_sys_global_v4_u32(line_address)
                    line1 = ld_relaxed_sys_global_v4_u32(line_address + Int64(16))
                    waiting = (
                        (line0[1] ^ expected_generation)
                        | (line0[3] ^ expected_generation)
                        | (line1[1] ^ expected_generation)
                        | (line1[3] ^ expected_generation)
                    )
                peer_words = (line0[0], line0[2], line1[0], line1[2])
                packed_sum = cute.make_rmem_tensor((4,), cutlass.Uint32)
                for word in cutlass.range_constexpr(4):
                    if cutlass.const_expr(self._dtype_name == "float16"):
                        packed_sum[word] = add_f16x2(
                            local_words[word], peer_words[word]
                        )
                    else:
                        packed_sum[word] = add_bf16x2(
                            local_words[word], peer_words[word]
                        )
                st_global_v4_u32(
                    output_base + Int64(index) * Int64(16),
                    packed_sum[0],
                    packed_sum[1],
                    packed_sum[2],
                    packed_sum[3],
                )
                index += stride

            if generation == Uint32(_PLAIN_GRAPH_LAST_EPOCH):
                # Matching CTAs finish reading before either rank clears records.
                self._multi_gpu_barrier(signal_ptrs)
                clear_index = Int64(bidx) * Int64(self._threads) + Int64(tidx)
                while clear_index < plain_capacity_packs:
                    for clear_slot in cutlass.range_constexpr(2):
                        clear_base = plain_local_slot0
                        if cutlass.const_expr(clear_slot == 1):
                            clear_base = plain_local_slot1
                        line = clear_base + plain_region_packs * Int64(16) + clear_index * Int64(32)
                        st_global_v4_u32(line, Uint32(0), Uint32(0), Uint32(0), Uint32(0))
                        st_global_v4_u32(line + Int64(16), Uint32(0), Uint32(0), Uint32(0), Uint32(0))
                    clear_index += Int64(stride)
                # The peer cannot publish another epoch until its receiver's
                # matching CTA has completed the clear with system visibility.
                self._multi_gpu_barrier(signal_ptrs)


class _FusedOneshotLaunch(_PackedMath):
    def __init__(
        self,
        dtype_name: str,
        world_size: int,
        rank: int,
        mode: str,
        single_cta: bool,
        register_normalize: bool,
        device_slot_selection: bool,
        slot_bias: int,
        threads: int,
    ) -> None:
        super().__init__(dtype_name)
        if mode not in (
            "registered",
            "stage_pull",
            "stage_push",
            "stage_remote_push",
            "stage_tp8_owner",
        ):
            raise ValueError(f"invalid fused oneshot mode {mode!r}")
        if mode == "stage_remote_push" and world_size not in (2, 4):
            raise ValueError("fused remote push requires TP2 or TP4")
        if mode == "stage_tp8_owner" and world_size != 8:
            raise ValueError("fused owner transport requires TP8")
        if mode == "stage_tp8_owner" and dtype_name not in (
            "float16",
            "bfloat16",
        ):
            raise ValueError("fused TP8 owner transport requires float16 or bfloat16")
        self._world_size = int(world_size)
        self._rank = int(rank)
        self._mode = mode
        self._single_cta = bool(single_cta)
        self._register_normalize = bool(register_normalize)
        self._device_slot_selection = bool(device_slot_selection)
        self._slot_bias = int(slot_bias) & 1
        self._threads = int(threads)
        # C1 uses the ordinary staged protocol while C2-C8 use three scoped
        # phases. Keep their reusable barrier generations in separate rank
        # slots when one graph-owned channel alternates between those shapes.
        self._barrier_rank_offset = 8 if mode == "stage_tp8_owner" else 0

    @cute.jit
    def __call__(
        self,
        peer_ptrs: cute.Pointer,
        signal_ptrs: cute.Pointer,
        input_ptr: cute.Pointer,
        residual_ptr: cute.Pointer,
        weight_ptr: cute.Pointer,
        output_ptr: cute.Pointer,
        residual_output_ptr: cute.Pointer,
        hidden_packs: Int32,
        rows: Int32,
        ctas_per_row: Int32,
        shard_packs: Int64,
        epsilon: Float32,
        grid_x: Int32,
        stream: cuda.CUstream,
    ) -> None:
        self.kernel(
            peer_ptrs,
            signal_ptrs,
            input_ptr,
            residual_ptr,
            weight_ptr,
            output_ptr,
            residual_output_ptr,
            hidden_packs,
            rows,
            ctas_per_row,
            shard_packs,
            epsilon,
        ).launch(
            grid=(grid_x, 1, 1),
            block=[self._threads, 1, 1],
            cluster=(1, 1, 1),
            stream=stream,
        )

    @cute.jit
    def _pair_arrive_and_wait(
        self,
        signal_ptrs: cute.Pointer,
        peer_rank: Int32,
    ) -> None:
        bidx, _, _ = cute.arch.block_idx()
        barrier_peer = peer_rank + Int32(self._barrier_rank_offset)
        self_signal = Int64(signal_ptrs[self._rank])
        peer_signal = Int64(signal_ptrs[peer_rank])
        self_counter = self_signal + (
            Int64(bidx) * Int64(_MAX_RANKS) + Int64(barrier_peer)
        ) * Int64(4)
        peer_slot0 = (
            peer_signal
            + Int64(_SELF_COUNTER_BYTES)
            + (
                Int64(bidx) * Int64(_MAX_RANKS * _FLAG_STRIDE)
                + Int64((self._rank + self._barrier_rank_offset) * _FLAG_STRIDE)
            )
            * Int64(4)
        )
        wait_slot0 = (
            self_signal
            + Int64(_SELF_COUNTER_BYTES)
            + (
                Int64(bidx) * Int64(_MAX_RANKS * _FLAG_STRIDE)
                + Int64(barrier_peer) * Int64(_FLAG_STRIDE)
            )
            * Int64(4)
        )
        pcie_arrive_and_wait(
            self_counter,
            peer_slot0,
            wait_slot0,
            Int64(_PEER_SLOT_BYTES),
        )

    @cute.jit
    def _multi_gpu_barrier(self, signal_ptrs: cute.Pointer) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        if cutlass.const_expr(self._mode != "registered"):
            cute.arch.sync_threads()
        if tidx < Int32(self._world_size):
            self._pair_arrive_and_wait(signal_ptrs, Int32(tidx))
        cute.arch.sync_threads()

    @cute.jit
    def _tp8_island_barrier(self, signal_ptrs: cute.Pointer) -> None:
        """Publish and acquire payloads within one four-rank island."""

        tidx, _, _ = cute.arch.thread_idx()
        island_base = Int32(0 if self._rank < 4 else 4)
        cute.arch.sync_threads()
        if tidx < Int32(4):
            peer_rank = island_base + Int32(tidx)
            if peer_rank != Int32(self._rank):
                self._pair_arrive_and_wait(signal_ptrs, peer_rank)
        cute.arch.sync_threads()

    @cute.jit
    def _tp8_owner_pair_barrier(self, signal_ptrs: cute.Pointer) -> None:
        """Publish and acquire one FP32 partial between paired owners.

        This barrier must remain mutual so paired owners can skew by at most
        one invocation. Eager slots alternate between invocations, making
        that bounded skew use different storage. Both properties are safety
        requirements for local-partial publication and final consumption.
        """

        tidx, _, _ = cute.arch.thread_idx()
        cute.arch.sync_threads()
        if Int32(tidx) == Int32(0):
            self._pair_arrive_and_wait(signal_ptrs, Int32(self._rank ^ 4))
        cute.arch.sync_threads()

    @cute.jit
    def _stage_pack(
        self,
        peer_ptrs: cute.Pointer,
        input_base: Int64,
        index: Int64,
        total_packs: Int64,
        shard_packs: Int64,
    ) -> None:
        if cutlass.const_expr(self._mode == "stage_pull"):
            self._copy_pack(
                input_base + index * Int64(16),
                Int64(peer_ptrs[self._rank]) + index * Int64(16),
            )
        elif cutlass.const_expr(
            self._mode == "stage_push" or self._mode == "stage_remote_push"
        ):
            source = input_base + index * Int64(16)
            words = ld_global_v4_u32(source)
            for peer_index in cutlass.range_constexpr(1, self._world_size):
                destination_rank = (self._rank + peer_index) % self._world_size
                destination = Int64(peer_ptrs[destination_rank]) + (
                    Int64(self._rank) * shard_packs + index
                ) * Int64(16)
                st_global_v4_u32(destination, words[0], words[1], words[2], words[3])
        elif cutlass.const_expr(self._mode == "stage_tp8_owner"):
            part = total_packs // Int64(4)
            chunk = index // part
            if chunk > Int64(3):
                chunk = Int64(3)
            island_base = Int64(0 if self._rank < 4 else 4)
            owner_rank = island_base + chunk
            destination = Int64(peer_ptrs[owner_rank]) + (
                Int64(self._rank) * shard_packs + index
            ) * Int64(16)
            words = ld_global_v4_u32(input_base + index * Int64(16))
            st_global_v4_u32(destination, words[0], words[1], words[2], words[3])

    @cute.jit
    def _tp8_publish_local_partial(
        self,
        peer_ptrs: cute.Pointer,
        index: Int64,
        total_packs: Int64,
        shard_packs: Int64,
    ) -> None:
        part = total_packs // Int64(4)
        chunk = index // part
        if chunk > Int64(3):
            chunk = Int64(3)
        island_base = 0 if self._rank < 4 else 4
        owner_rank = Int32(island_base) + Int32(chunk)
        if Int32(self._rank) == owner_rank:
            local_stage = Int64(peer_ptrs[self._rank])
            local_sum = cute.make_rmem_tensor((self._pack_elems,), cutlass.Float32)
            for peer_local in cutlass.range_constexpr(4):
                source_rank = island_base + peer_local
                self._load_accumulate(
                    local_sum,
                    local_stage
                    + (Int64(source_rank) * shard_packs + index) * Int64(16),
                    peer_local == 0,
                )

            low = (
                f32_as_u32(local_sum[0]),
                f32_as_u32(local_sum[1]),
                f32_as_u32(local_sum[2]),
                f32_as_u32(local_sum[3]),
            )
            high = (
                f32_as_u32(local_sum[4]),
                f32_as_u32(local_sum[5]),
                f32_as_u32(local_sum[6]),
                f32_as_u32(local_sum[7]),
            )
            cross_owner = self._rank ^ 4
            for destination_rank in (self._rank, cross_owner):
                destination_stage = Int64(peer_ptrs[destination_rank])
                st_global_v4_u32(
                    destination_stage
                    + (Int64(self._rank) * shard_packs + index) * Int64(16),
                    low[0],
                    low[1],
                    low[2],
                    low[3],
                )
                st_global_v4_u32(
                    destination_stage
                    + (Int64(self._rank ^ 1) * shard_packs + index) * Int64(16),
                    high[0],
                    high[1],
                    high[2],
                    high[3],
                )

    @cute.jit
    def _tp8_publish_final(
        self,
        peer_ptrs: cute.Pointer,
        index: Int64,
        total_packs: Int64,
        shard_packs: Int64,
    ) -> None:
        part = total_packs // Int64(4)
        chunk = index // part
        if chunk > Int64(3):
            chunk = Int64(3)
        island_base = 0 if self._rank < 4 else 4
        owner_rank = Int32(island_base) + Int32(chunk)
        if Int32(self._rank) == owner_rank:
            local_stage = Int64(peer_ptrs[self._rank])
            cross_owner = self._rank ^ 4
            local_low = ld_global_v4_u32(
                local_stage + (Int64(self._rank) * shard_packs + index) * Int64(16)
            )
            local_high = ld_global_v4_u32(
                local_stage + (Int64(self._rank ^ 1) * shard_packs + index) * Int64(16)
            )
            cross_low = ld_global_v4_u32(
                local_stage + (Int64(cross_owner) * shard_packs + index) * Int64(16)
            )
            cross_high = ld_global_v4_u32(
                local_stage + (Int64(cross_owner ^ 1) * shard_packs + index) * Int64(16)
            )
            final_low = cute.make_rmem_tensor((4,), cutlass.Uint32)
            final_high = cute.make_rmem_tensor((4,), cutlass.Uint32)
            for lane in cutlass.range_constexpr(4):
                final_low[lane] = f32_as_u32(
                    u32_as_f32(local_low[lane]) + u32_as_f32(cross_low[lane])
                )
                final_high[lane] = f32_as_u32(
                    u32_as_f32(local_high[lane]) + u32_as_f32(cross_high[lane])
                )

            for peer_local in cutlass.range_constexpr(4):
                destination_rank = island_base + peer_local
                destination_stage = Int64(peer_ptrs[destination_rank])
                st_global_v4_u32(
                    destination_stage
                    + (Int64(self._rank) * shard_packs + index) * Int64(16),
                    final_low[0],
                    final_low[1],
                    final_low[2],
                    final_low[3],
                )
                st_global_v4_u32(
                    destination_stage
                    + (Int64(self._rank ^ 1) * shard_packs + index) * Int64(16),
                    final_high[0],
                    final_high[1],
                    final_high[2],
                    final_high[3],
                )

    @cute.jit
    def _reduce_pack(
        self,
        peer_ptrs: cute.Pointer,
        input_base: Int64,
        residual_base: Int64,
        index: Int64,
        total_packs: Int64,
        shard_packs: Int64,
        accumulator: cute.Tensor,
    ) -> None:
        if cutlass.const_expr(self._mode == "stage_tp8_owner"):
            part = total_packs // Int64(4)
            chunk = index // part
            if chunk > Int64(3):
                chunk = Int64(3)
            island_base = Int64(0 if self._rank < 4 else 4)
            owner_rank = island_base + chunk
            local_stage = Int64(peer_ptrs[self._rank])
            low = ld_global_v4_u32(
                local_stage + (owner_rank * shard_packs + index) * Int64(16)
            )
            high = ld_global_v4_u32(
                local_stage
                + ((owner_rank ^ Int64(1)) * shard_packs + index) * Int64(16)
            )
            for lane in cutlass.range_constexpr(4):
                accumulator[lane] = u32_as_f32(low[lane])
                accumulator[lane + 4] = u32_as_f32(high[lane])
        elif cutlass.const_expr(self._mode == "stage_remote_push"):
            self._load_accumulate(
                accumulator,
                input_base + index * Int64(16),
                True,
            )
            local_stage = Int64(peer_ptrs[self._rank])
            for peer_index in cutlass.range_constexpr(1, self._world_size):
                source_rank = (self._rank + peer_index) % self._world_size
                self._load_accumulate(
                    accumulator,
                    local_stage
                    + (Int64(source_rank) * shard_packs + index) * Int64(16),
                    False,
                )
        elif cutlass.const_expr(self._mode == "stage_push"):
            self._load_accumulate(
                accumulator,
                input_base + index * Int64(16),
                True,
            )
            local_stage = Int64(peer_ptrs[self._rank])
            initialized = True
            for source_rank in cutlass.range_constexpr(self._world_size):
                if cutlass.const_expr(source_rank != self._rank):
                    self._load_accumulate(
                        accumulator,
                        local_stage
                        + (Int64(source_rank) * shard_packs + index) * Int64(16),
                        not initialized,
                    )
                    initialized = True
        else:
            for peer_index in cutlass.range_constexpr(self._world_size):
                peer_rank = (self._rank + peer_index) % self._world_size
                self._load_accumulate(
                    accumulator,
                    Int64(peer_ptrs[peer_rank]) + index * Int64(16),
                    peer_index == 0,
                )
        self._load_accumulate(
            accumulator,
            residual_base + index * Int64(16),
            False,
        )

    @cute.jit
    def _block_reduce(
        self,
        value: Float32,
        warp_sums: cute.Tensor,
    ) -> Float32:
        tidx, _, _ = cute.arch.thread_idx()
        lane = Int32(tidx) % Int32(32)
        warp = Int32(tidx) // Int32(32)
        value = self._warp_reduce_down(value)
        if lane == Int32(0):
            warp_sums[warp] = value
        cute.arch.sync_threads()
        block_value = Float32(0.0)
        if Int32(tidx) < Int32(self._threads // 32):
            block_value = warp_sums[lane]
        if warp == Int32(0):
            block_value = self._warp_reduce_down(block_value)
        return block_value

    @cute.kernel
    def kernel(
        self,
        peer_ptrs: cute.Pointer,
        signal_ptrs: cute.Pointer,
        input_ptr: cute.Pointer,
        residual_ptr: cute.Pointer,
        weight_ptr: cute.Pointer,
        output_ptr: cute.Pointer,
        residual_output_ptr: cute.Pointer,
        hidden_packs: Int32,
        rows: Int32,
        ctas_per_row: Int32,
        shard_packs: Int64,
        epsilon: Float32,
    ) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        gdim, _, _ = cute.arch.grid_dim()
        if cutlass.const_expr(self._single_cta):
            row = Int32(bidx)
            row_cta = Int32(0)
            col_stride = Int32(self._threads)
        else:
            row = Int32(bidx) // ctas_per_row
            row_cta = Int32(bidx) - row * ctas_per_row
            col_stride = ctas_per_row * Int32(self._threads)
        col0 = row_cta * Int32(self._threads) + Int32(tidx)
        row_offset = Int64(row) * Int64(hidden_packs)
        total_packs = Int64(rows) * Int64(hidden_packs)

        input_base = Int64(input_ptr.toint())
        residual_base = Int64(residual_ptr.toint())
        weight_base = Int64(weight_ptr.toint())
        output_base = Int64(output_ptr.toint())
        residual_output_base = Int64(residual_output_ptr.toint())
        self_signal = Int64(signal_ptrs[self._rank])
        if cutlass.const_expr(self._device_slot_selection):
            generation = ld_relaxed_gpu_u32(self_signal + Int64(_GRAPH_EPOCH_OFFSET))
            slot = (generation + Uint32(self._slot_bias)) % Uint32(2)
            peer_ptrs = peer_ptrs + Int64(slot) * Int64(_MAX_RANKS)

        smem = cutlass.utils.SmemAllocator()
        warp_sums = smem.allocate_tensor(
            element_type=cutlass.Float32,
            layout=cute.make_layout((32,)),
            byte_alignment=16,
        )
        inv_rms_smem = smem.allocate_tensor(
            element_type=cutlass.Float32,
            layout=cute.make_layout((1,)),
            byte_alignment=4,
        )

        row_generation = Uint32(0)
        if cutlass.const_expr(not self._single_cta):
            if Int32(tidx) == Int32(0):
                row_generation = ld_relaxed_gpu_u32(
                    self_signal + Int64(_RMS_GEN_OFFSET) + Int64(row) * Int64(4)
                )

        if cutlass.const_expr(self._register_normalize):
            for pack_number in cutlass.range_constexpr(_REG_PACKS):
                col = col0 + Int32(pack_number) * col_stride
                if col < hidden_packs:
                    self._stage_pack(
                        peer_ptrs,
                        input_base,
                        row_offset + Int64(col),
                        total_packs,
                        shard_packs,
                    )
        elif cutlass.const_expr(self._mode != "registered"):
            col = col0
            while col < hidden_packs:
                self._stage_pack(
                    peer_ptrs,
                    input_base,
                    row_offset + Int64(col),
                    total_packs,
                    shard_packs,
                )
                col += col_stride

        if cutlass.const_expr(self._mode == "stage_tp8_owner"):
            self._tp8_island_barrier(signal_ptrs)
            col = col0
            while col < hidden_packs:
                self._tp8_publish_local_partial(
                    peer_ptrs,
                    row_offset + Int64(col),
                    total_packs,
                    shard_packs,
                )
                col += col_stride
            self._tp8_owner_pair_barrier(signal_ptrs)

            col = col0
            while col < hidden_packs:
                self._tp8_publish_final(
                    peer_ptrs,
                    row_offset + Int64(col),
                    total_packs,
                    shard_packs,
                )
                col += col_stride
            self._tp8_island_barrier(signal_ptrs)
        else:
            self._multi_gpu_barrier(signal_ptrs)

        square_sum = Float32(0.0)
        values = cute.make_rmem_tensor((_REG_PACKS, self._pack_elems), cutlass.Float32)
        if cutlass.const_expr(self._register_normalize):
            for pack_number in cutlass.range_constexpr(_REG_PACKS):
                col = col0 + Int32(pack_number) * col_stride
                if col < hidden_packs:
                    index = row_offset + Int64(col)
                    accumulator = cute.make_rmem_tensor(
                        (self._pack_elems,), cutlass.Float32
                    )
                    self._reduce_pack(
                        peer_ptrs,
                        input_base,
                        residual_base,
                        index,
                        total_packs,
                        shard_packs,
                        accumulator,
                    )
                    self._store_accumulator(
                        residual_output_base + index * Int64(16), accumulator
                    )
                    for lane in cutlass.range_constexpr(self._pack_elems):
                        value = accumulator[lane]
                        square_sum = square_sum + value * value
                        values[pack_number, lane] = value
        else:
            col = col0
            while col < hidden_packs:
                index = row_offset + Int64(col)
                accumulator = cute.make_rmem_tensor(
                    (self._pack_elems,), cutlass.Float32
                )
                self._reduce_pack(
                    peer_ptrs,
                    input_base,
                    residual_base,
                    index,
                    total_packs,
                    shard_packs,
                    accumulator,
                )
                self._store_accumulator(
                    residual_output_base + index * Int64(16), accumulator
                )
                for lane in cutlass.range_constexpr(self._pack_elems):
                    value = accumulator[lane]
                    square_sum = square_sum + value * value
                col += col_stride

        square_sum = self._block_reduce(square_sum, warp_sums)
        hidden_size_f = Float32(hidden_packs * Int32(self._pack_elems))
        if cutlass.const_expr(self._single_cta):
            if Int32(tidx) == Int32(0):
                inv_rms_smem[0] = rsqrt_approx_f32(square_sum / hidden_size_f + epsilon)
        else:
            if Int32(tidx) == Int32(0):
                st_global_f32(
                    self_signal + Int64(_RMS_PARTIAL_OFFSET) + Int64(bidx) * Int64(4),
                    square_sum,
                )
                threadfence_gpu()
                arrive_address = (
                    self_signal + Int64(_RMS_ARRIVE_OFFSET) + Int64(row) * Int64(4)
                )
                prior = atomic_add_global_u32(arrive_address, Uint32(1))
                if prior == Uint32(ctas_per_row - Int32(1)):
                    st_global_u32(arrive_address, Uint32(0))
                    threadfence_gpu()
                    atomic_add_global_u32(
                        self_signal + Int64(_RMS_GEN_OFFSET) + Int64(row) * Int64(4),
                        Uint32(1),
                    )
                else:
                    spin_until_changed_acquire_gpu(
                        self_signal + Int64(_RMS_GEN_OFFSET) + Int64(row) * Int64(4),
                        row_generation,
                    )
            cute.arch.sync_threads()
            if Int32(tidx) < Int32(32):
                partial = Float32(0.0)
                partial_index = Int32(tidx)
                while partial_index < ctas_per_row:
                    partial = partial + ld_global_f32(
                        self_signal
                        + Int64(_RMS_PARTIAL_OFFSET)
                        + (Int64(row) * Int64(ctas_per_row) + Int64(partial_index))
                        * Int64(4)
                    )
                    partial_index += Int32(32)
                partial = self._warp_reduce_down(partial)
                if Int32(tidx) == Int32(0):
                    inv_rms_smem[0] = rsqrt_approx_f32(
                        partial / hidden_size_f + epsilon
                    )
        cute.arch.sync_threads()
        inv_rms = inv_rms_smem[0]

        if cutlass.const_expr(self._register_normalize):
            for pack_number in cutlass.range_constexpr(_REG_PACKS):
                col = col0 + Int32(pack_number) * col_stride
                if col < hidden_packs:
                    scale = cute.make_rmem_tensor((self._pack_elems,), cutlass.Float32)
                    self._load_accumulate(
                        scale, weight_base + Int64(col) * Int64(16), True
                    )
                    normalized = cute.make_rmem_tensor(
                        (self._pack_elems,), cutlass.Float32
                    )
                    for lane in cutlass.range_constexpr(self._pack_elems):
                        normalized[lane] = values[pack_number, lane] * (
                            inv_rms * scale[lane]
                        )
                    self._store_accumulator(
                        output_base + (row_offset + Int64(col)) * Int64(16),
                        normalized,
                    )
        else:
            col = col0
            while col < hidden_packs:
                index = row_offset + Int64(col)
                value = cute.make_rmem_tensor((self._pack_elems,), cutlass.Float32)
                scale = cute.make_rmem_tensor((self._pack_elems,), cutlass.Float32)
                self._load_accumulate(
                    value,
                    residual_output_base + index * Int64(16),
                    True,
                )
                self._load_accumulate(scale, weight_base + Int64(col) * Int64(16), True)
                for lane in cutlass.range_constexpr(self._pack_elems):
                    value[lane] = value[lane] * (inv_rms * scale[lane])
                self._store_accumulator(output_base + index * Int64(16), value)
                col += col_stride

        if cutlass.const_expr(self._device_slot_selection):
            if Int32(tidx) == Int32(0):
                graph_epoch_arrive(
                    self_signal + Int64(_GRAPH_EPOCH_OFFSET),
                    self_signal + Int64(_GRAPH_ARRIVED_OFFSET),
                    Uint32(gdim),
                )


def _dummy(dtype, alignment: int):
    return make_ptr(dtype, 16, cute.AddressSpace.gmem, assumed_align=alignment)


def _oneshot_process_key(
    dtype_name: str,
    world_size: int,
    rank: int,
    stage_input: bool,
    device_slot_selection: bool,
    slot_bias: int,
    transport: str,
    threads: int,
    device_index: int,
) -> tuple[object, ...]:
    return (
        str(dtype_name),
        int(world_size),
        int(rank),
        bool(stage_input),
        bool(device_slot_selection),
        int(slot_bias) & 1 if device_slot_selection else 0,
        str(transport),
        int(threads),
        int(device_index),
    )


def is_oneshot_launcher_prepared(
    dtype_name: str,
    world_size: int,
    rank: int,
    stage_input: bool,
    device_slot_selection: bool,
    slot_bias: int,
    transport: str,
    threads: int,
    device_index: int,
) -> bool:
    """Return whether this exact process-local launcher is already loaded."""

    return (
        _oneshot_process_key(
            dtype_name,
            world_size,
            rank,
            stage_input,
            device_slot_selection,
            slot_bias,
            transport,
            threads,
            device_index,
        )
        in _PREPARED_ONESHOT_LAUNCHERS
    )


@functools.cache
def get_oneshot_launcher(
    dtype_name: str,
    world_size: int,
    rank: int,
    stage_input: bool,
    device_slot_selection: bool,
    slot_bias: int,
    transport: str,
    threads: int,
    device_index: int,
) -> Callable[..., None]:
    process_key = _oneshot_process_key(
        dtype_name,
        world_size,
        rank,
        stage_input,
        device_slot_selection,
        slot_bias,
        transport,
        threads,
        device_index,
    )
    del device_index  # retained in the functools and preparation keys
    slot_bias = int(slot_bias) & 1 if device_slot_selection else 0
    launch = _OneshotLaunch(
        dtype_name,
        world_size,
        rank,
        stage_input,
        device_slot_selection,
        slot_bias,
        transport,
        threads,
    )
    cache_key = (
        dtype_name,
        int(world_size),
        int(rank),
        bool(stage_input),
        bool(device_slot_selection),
        slot_bias,
        str(transport),
        int(threads),
    )
    raise_if_kernel_resolution_frozen(
        "cute.compile", target=launch, cache_key=cache_key
    )
    raw = b12x_compile(
        launch,
        _dummy(cutlass.Uint64, 8),
        _dummy(cutlass.Uint64, 8),
        _dummy(cutlass.Uint32, 16),
        _dummy(cutlass.Uint32, 16),
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        current_cuda_stream(),
        compile_spec=KernelCompileSpec.from_key("comm.pcie.oneshot", 4, cache_key),
    )

    def run(
        peer_table_address: int,
        signal_table_address: int,
        input_address: int,
        output_address: int,
        size_packs: int,
        plain_region_packs: int,
        plain_capacity_packs: int,
        plain_local_slot0: int,
        plain_peer_slot0: int,
        plain_local_slot1: int,
        plain_peer_slot1: int,
        grid_x: int,
    ) -> None:
        raw(
            make_ptr(
                cutlass.Uint64,
                peer_table_address,
                cute.AddressSpace.gmem,
                assumed_align=8,
            ),
            make_ptr(
                cutlass.Uint64,
                signal_table_address,
                cute.AddressSpace.gmem,
                assumed_align=8,
            ),
            make_ptr(
                cutlass.Uint32,
                input_address,
                cute.AddressSpace.gmem,
                assumed_align=16,
            ),
            make_ptr(
                cutlass.Uint32,
                output_address,
                cute.AddressSpace.gmem,
                assumed_align=16,
            ),
            int(size_packs),
            int(plain_region_packs),
            int(plain_capacity_packs),
            int(plain_local_slot0),
            int(plain_peer_slot0),
            int(plain_local_slot1),
            int(plain_peer_slot1),
            int(grid_x),
            current_cuda_stream(),
        )

    _PREPARED_ONESHOT_LAUNCHERS.add(process_key)
    return attach_programs(run, raw)


def _fused_oneshot_process_key(
    dtype_name: str,
    world_size: int,
    rank: int,
    mode: str,
    single_cta: bool,
    register_normalize: bool,
    device_slot_selection: bool,
    slot_bias: int,
    threads: int,
    device_index: int,
) -> tuple[object, ...]:
    return (
        str(dtype_name),
        int(world_size),
        int(rank),
        str(mode),
        bool(single_cta),
        bool(register_normalize),
        bool(device_slot_selection),
        int(slot_bias) & 1 if device_slot_selection else 0,
        int(threads),
        int(device_index),
    )


def is_fused_oneshot_launcher_prepared(
    dtype_name: str,
    world_size: int,
    rank: int,
    mode: str,
    single_cta: bool,
    register_normalize: bool,
    device_slot_selection: bool,
    slot_bias: int,
    threads: int,
    device_index: int,
) -> bool:
    """Return whether this exact fused graph launcher is already loaded."""

    return (
        _fused_oneshot_process_key(
            dtype_name,
            world_size,
            rank,
            mode,
            single_cta,
            register_normalize,
            device_slot_selection,
            slot_bias,
            threads,
            device_index,
        )
        in _PREPARED_FUSED_ONESHOT_LAUNCHERS
    )


@functools.cache
def get_fused_oneshot_launcher(
    dtype_name: str,
    world_size: int,
    rank: int,
    mode: str,
    single_cta: bool,
    register_normalize: bool,
    device_slot_selection: bool,
    slot_bias: int,
    threads: int,
    device_index: int,
) -> Callable[..., None]:
    process_key = _fused_oneshot_process_key(
        dtype_name,
        world_size,
        rank,
        mode,
        single_cta,
        register_normalize,
        device_slot_selection,
        slot_bias,
        threads,
        device_index,
    )
    del device_index
    slot_bias = int(slot_bias) & 1 if device_slot_selection else 0
    launch = _FusedOneshotLaunch(
        dtype_name,
        world_size,
        rank,
        mode,
        single_cta,
        register_normalize,
        device_slot_selection,
        slot_bias,
        threads,
    )
    cache_key = (
        dtype_name,
        int(world_size),
        int(rank),
        mode,
        bool(single_cta),
        bool(register_normalize),
        bool(device_slot_selection),
        slot_bias,
        int(threads),
    )
    raise_if_kernel_resolution_frozen(
        "cute.compile", target=launch, cache_key=cache_key
    )
    raw = b12x_compile(
        launch,
        _dummy(cutlass.Uint64, 8),
        _dummy(cutlass.Uint64, 8),
        _dummy(cutlass.Uint32, 16),
        _dummy(cutlass.Uint32, 16),
        _dummy(cutlass.Uint32, 16),
        _dummy(cutlass.Uint32, 16),
        _dummy(cutlass.Uint32, 16),
        1,
        1,
        1,
        1,
        0.0,
        1,
        current_cuda_stream(),
        compile_spec=KernelCompileSpec.from_key(
            "comm.pcie.oneshot_fused_rmsnorm", 1, cache_key
        ),
    )

    def run(
        peer_table_address: int,
        signal_table_address: int,
        input_address: int,
        residual_address: int,
        weight_address: int,
        output_address: int,
        residual_output_address: int,
        hidden_packs: int,
        rows: int,
        ctas_per_row: int,
        shard_packs: int,
        epsilon: float,
        grid_x: int,
    ) -> None:
        raw(
            make_ptr(
                cutlass.Uint64,
                peer_table_address,
                cute.AddressSpace.gmem,
                assumed_align=8,
            ),
            make_ptr(
                cutlass.Uint64,
                signal_table_address,
                cute.AddressSpace.gmem,
                assumed_align=8,
            ),
            make_ptr(
                cutlass.Uint32, input_address, cute.AddressSpace.gmem, assumed_align=16
            ),
            make_ptr(
                cutlass.Uint32,
                residual_address,
                cute.AddressSpace.gmem,
                assumed_align=16,
            ),
            make_ptr(
                cutlass.Uint32, weight_address, cute.AddressSpace.gmem, assumed_align=16
            ),
            make_ptr(
                cutlass.Uint32, output_address, cute.AddressSpace.gmem, assumed_align=16
            ),
            make_ptr(
                cutlass.Uint32,
                residual_output_address,
                cute.AddressSpace.gmem,
                assumed_align=16,
            ),
            int(hidden_packs),
            int(rows),
            int(ctas_per_row),
            int(shard_packs),
            float(epsilon),
            int(grid_x),
            current_cuda_stream(),
        )

    _PREPARED_FUSED_ONESHOT_LAUNCHERS.add(process_key)
    return attach_programs(run, raw)


__all__ = [
    "get_fused_oneshot_launcher",
    "get_oneshot_launcher",
    "is_fused_oneshot_launcher_prepared",
    "is_oneshot_launcher_prepared",
]
