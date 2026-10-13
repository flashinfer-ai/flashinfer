"""CuTe DSL kernel for the RoCE pipelined two-shot all-reduce.

One launch performs a complete all-reduce of a large message as a
reduce-scatter followed by an all-gather, so each rank puts
``2 (world - 1) / world`` of the message on the wire instead of the one-shot's
``world - 1`` copies.  Rank ``j`` owns shard ``j`` (packs
``[j P / world, (j + 1) P / world)``), and every shard is split into the same
number of chunks so that staging, the wire and the reduction overlap:

1. doorbell: block 0 records the op (two-shot kind, bytes, chunk geometry) in
   ``seq & 1``'s slot of the control record and rings ``seq``;
2. stage: blocks copy chunk ``c`` of every peer's shard into ``pipe_send``; the
   last block to finish a chunk publishes ``ready_rs[c] = seq`` and the proxy
   writes that chunk to its owners;
3. reduce: per chunk of our shard, wait for every peer's stripe flags, sum the
   local input and the peers' chunks in fixed rank order, store the result in
   the output and in ``pipe_send``; the last block to finish publishes
   ``ready_ag[c] = seq`` and the proxy sends the reduced chunk to every peer;
4. gather: per chunk, wait for every peer's reduced chunk and copy it into the
   output;
5. epoch: the last block to finish advances the shared device epoch.

Every shard is reduced by its owner only, so all ranks hold identical bits, and
the arithmetic (FP32 accumulation in rank order) matches the one-shot kernel.
The chunk counters are free-running, which keeps the grid size a power of two
fixed per runtime.  No block waits for another block, so the grid needs no
co-residency.
"""

from __future__ import annotations

from collections.abc import Callable

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64, Uint32
import torch

from b12x._lib.compiler import KernelCompileSpec
from b12x._lib.compiler import compile as b12x_compile
from b12x._lib.compile_plan import attach_programs
from b12x._lib.program_cache import program_cache
from b12x._lib.runtime_control import raise_if_kernel_resolution_frozen
from b12x._lib.utils import current_cuda_stream, make_ptr

from ._cute_intrinsics import (
    atomic_add_relaxed_gpu_u32,
    fence_sc_gpu,
    fence_sc_sys,
    ld_global_v4_u32,
    ld_relaxed_gpu_u32,
    ld_relaxed_sys_u32,
    ld_relaxed_sys_v4_u32,
    spin_until_eq_acquire_sys,
    st_global_v4_u32,
    st_release_gpu_u32,
    st_relaxed_sys_u32,
)
from ._oneshot_cute import _DTYPE_NAMES, _RoceOneshotLaunch, PACK_BYTES

# Control-record words (see _roce_proxy.c).
_CTRL_SLOT_KIND = 8
_CTRL_SLOT_CHUNK_PACKS = 10
_CTRL_SLOT_CHUNKS = 12
_CTRL_READY_RS = 64
_CTRL_READY_AG = 128
_PREPARED_LAUNCHERS: set[tuple[object, ...]] = set()


class _RoceTwoshotLaunch(_RoceOneshotLaunch):
    def __init__(
        self,
        dtype_name: str,
        world_size: int,
        rank: int,
        threads: int,
        flag_stride: int,
        hca_count: int,
        max_chunks: int,
    ) -> None:
        """Bind one specialization: dtype, world size, rank, and layout constants."""
        super().__init__(
            dtype_name, world_size, rank, threads, 1, flag_stride, hca_count
        )
        self._max_chunks = int(max_chunks)

    @cute.jit
    def _shard_start(self, shard: Int32, size_packs: Int32) -> Int32:
        """First pack of ``shard`` (the proxy splits identically)."""
        return Int32((Int64(shard) * Int64(size_packs)) // Int64(self._world_size))

    @cute.jit
    def _wait_chunk(
        self,
        flag_base: Int64,
        phase: cutlass.Constexpr[int],
        chunk: Int32,
        seq: Uint32,
        ctrl_base: Int64,
        poison_ptr: Int64,
        spin_limit: Uint32,
    ) -> None:
        """One thread per (peer, rail) waits for chunk ``chunk``'s stripe flag of ``phase``."""
        tidx, _, _ = cute.arch.thread_idx()
        if Int32(tidx) < Int32(self._world_size * self._hca_count):
            peer = Int32(tidx) // Int32(self._hca_count)
            hca = Int32(tidx) - peer * Int32(self._hca_count)
            if peer != Int32(self._rank):
                flag_addr = flag_base + (
                    (
                        (Int64(phase * self._world_size) + Int64(peer))
                        * Int64(self._max_chunks)
                        + Int64(chunk)
                    )
                    * Int64(self._hca_count)
                    + Int64(hca)
                ) * Int64(self._flag_stride)
                timed_out = spin_until_eq_acquire_sys(flag_addr, seq, spin_limit)
                if timed_out != Uint32(0):
                    st_relaxed_sys_u32(ctrl_base + Int64(12), Uint32(peer))
                    st_relaxed_sys_u32(ctrl_base + Int64(24), Uint32(hca))
                    st_relaxed_sys_u32(ctrl_base + Int64(8), seq)
                    st_release_gpu_u32(poison_ptr, seq)

    @cute.jit
    def _arrive(
        self, counter: Int64, ready: Int64, seq: Uint32, gdim: Int32, ctrl_base: Int64
    ) -> None:
        """Publish this block's writes; the last of ``gdim`` arrivals sets ``ready = seq``.

        Nothing is published once a wait timed out: a block that timed out wrote
        the error word before arriving, so the last arrival sees it, and the
        proxy never sends a chunk whose reduction was skipped.
        """
        tidx, _, _ = cute.arch.thread_idx()
        if Int32(tidx) == Int32(0):
            fence_sc_sys()
            prior = atomic_add_relaxed_gpu_u32(counter, Uint32(1))
            if (prior + Uint32(1)) % Uint32(gdim) == Uint32(0):
                fence_sc_sys()
                if ld_relaxed_sys_u32(ctrl_base + Int64(8)) == Uint32(0):
                    st_relaxed_sys_u32(ready, seq)

    @cute.jit
    def __call__(
        self,
        input_ptr: cute.Pointer,
        output_ptr: cute.Pointer,
        size_packs: Int32,
        nbytes: Int32,
        chunk_packs: Int32,
        chunks: Int32,
        send_base: Int64,
        recv_base: Int64,
        flag_base: Int64,
        shard_cap: Int64,
        ctrl_base: Int64,
        epoch_ptr: Int64,
        counter_base: Int64,
        poison_ptr: Int64,
        spin_limit: Uint32,
        grid_x: Int32,
        stream: cuda.CUstream,
    ) -> None:
        """Host entry: launch the two-shot kernel with runtime scalars."""
        self.kernel(
            input_ptr,
            output_ptr,
            size_packs,
            nbytes,
            chunk_packs,
            chunks,
            send_base,
            recv_base,
            flag_base,
            shard_cap,
            ctrl_base,
            epoch_ptr,
            counter_base,
            poison_ptr,
            spin_limit,
        ).launch(
            grid=(grid_x, 1, 1),
            block=[self._threads, 1, 1],
            cluster=(1, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        input_ptr: cute.Pointer,
        output_ptr: cute.Pointer,
        size_packs: Int32,
        nbytes: Int32,
        chunk_packs: Int32,
        chunks: Int32,
        send_base: Int64,
        recv_base: Int64,
        flag_base: Int64,
        shard_cap: Int64,
        ctrl_base: Int64,
        epoch_ptr: Int64,
        counter_base: Int64,
        poison_ptr: Int64,
        spin_limit: Uint32,
    ) -> None:
        """Device kernel: doorbell, stage, reduce our shard, gather, advance the epoch."""
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        gdim, _, _ = cute.arch.grid_dim()
        input_base = Int64(input_ptr.toint())
        output_base = Int64(output_ptr.toint())
        epoch = ld_relaxed_gpu_u32(epoch_ptr)
        seq = epoch + Uint32(1)
        slot = Int64(seq & Uint32(1))
        index = Int32(bidx) * Int32(self._threads) + Int32(tidx)
        stride = Int32(gdim) * Int32(self._threads)
        stage_counters = counter_base
        reduce_counters = counter_base + Int64(4 * self._max_chunks)
        tail_counter = counter_base + Int64(8 * self._max_chunks)
        own_first = self._shard_start(Int32(self._rank), size_packs)
        own_end = self._shard_start(Int32(self._rank + 1), size_packs)

        poisoned = ld_relaxed_gpu_u32(poison_ptr)
        if poisoned == Uint32(0):
            # 1. the op record, then the doorbell; the proxy then follows the ready words
            if Int32(bidx) == Int32(0):
                if Int32(tidx) == Int32(0):
                    st_relaxed_sys_u32(ctrl_base + Int64(4), Uint32(nbytes))
                    st_relaxed_sys_u32(
                        ctrl_base + Int64(16) + slot * Int64(4), Uint32(nbytes)
                    )
                    st_relaxed_sys_u32(
                        ctrl_base + Int64(4 * _CTRL_SLOT_KIND) + slot * Int64(4),
                        Uint32(1),
                    )
                    st_relaxed_sys_u32(
                        ctrl_base + Int64(4 * _CTRL_SLOT_CHUNK_PACKS) + slot * Int64(4),
                        Uint32(chunk_packs),
                    )
                    st_relaxed_sys_u32(
                        ctrl_base + Int64(4 * _CTRL_SLOT_CHUNKS) + slot * Int64(4),
                        Uint32(chunks),
                    )
                    fence_sc_sys()
                    st_relaxed_sys_u32(ctrl_base, seq)

            # 2. stage every peer's shard, chunk by chunk
            for chunk in cutlass.range(chunks, unroll=1):
                for shard in cutlass.range_constexpr(self._world_size):
                    if cutlass.const_expr(shard != self._rank):
                        first = self._shard_start(Int32(shard), size_packs)
                        end = self._shard_start(Int32(shard + 1), size_packs)
                        lo = first + chunk * chunk_packs
                        hi = cutlass.min(end, lo + chunk_packs)
                        stage_index = lo + index
                        while stage_index < hi:
                            offset = Int64(stage_index) * Int64(PACK_BYTES)
                            words = ld_global_v4_u32(input_base + offset)
                            st_global_v4_u32(
                                send_base + offset,
                                words[0],
                                words[1],
                                words[2],
                                words[3],
                            )
                            stage_index += stride
                cute.arch.sync_threads()
                self._arrive(
                    stage_counters + Int64(4) * Int64(chunk),
                    ctrl_base + Int64(4 * _CTRL_READY_RS) + Int64(4) * Int64(chunk),
                    seq,
                    Int32(gdim),
                    ctrl_base,
                )

            # 3. reduce our shard in fixed rank order as peers' chunks arrive
            for chunk in cutlass.range(chunks, unroll=1):
                if ld_relaxed_gpu_u32(poison_ptr) == Uint32(0):
                    self._wait_chunk(
                        flag_base, 0, chunk, seq, ctrl_base, poison_ptr, spin_limit
                    )
                cute.arch.sync_threads()
                if ld_relaxed_gpu_u32(poison_ptr) == Uint32(0):
                    lo = own_first + chunk * chunk_packs
                    hi = cutlass.min(own_end, lo + chunk_packs)
                    reduce_index = lo + index
                    while reduce_index < hi:
                        accumulator = cute.make_rmem_tensor(
                            (self._pack_elems,), cutlass.Float32
                        )
                        offset = Int64(reduce_index) * Int64(PACK_BYTES)
                        shard_offset = Int64(reduce_index - own_first) * Int64(
                            PACK_BYTES
                        )
                        for source in cutlass.range_constexpr(self._world_size):
                            if cutlass.const_expr(source == self._rank):
                                words = ld_global_v4_u32(input_base + offset)
                            else:
                                words = ld_relaxed_sys_v4_u32(
                                    recv_base
                                    + Int64(source * 2) * shard_cap
                                    + shard_offset
                                )
                            self._accumulate_words(accumulator, words, source == 0)
                        self._store_accumulator(output_base + offset, accumulator)
                        self._store_accumulator(send_base + offset, accumulator)
                        reduce_index += stride
                # Threads may disagree on the poison word, so the barrier and the
                # arrival stay unconditional; a poisoned runtime is never reused.
                cute.arch.sync_threads()
                self._arrive(
                    reduce_counters + Int64(4) * Int64(chunk),
                    ctrl_base + Int64(4 * _CTRL_READY_AG) + Int64(4) * Int64(chunk),
                    seq,
                    Int32(gdim),
                    ctrl_base,
                )

            # 4. gather every peer's reduced shard
            for chunk in cutlass.range(chunks, unroll=1):
                if ld_relaxed_gpu_u32(poison_ptr) == Uint32(0):
                    self._wait_chunk(
                        flag_base, 1, chunk, seq, ctrl_base, poison_ptr, spin_limit
                    )
                cute.arch.sync_threads()
                if ld_relaxed_gpu_u32(poison_ptr) == Uint32(0):
                    for shard in cutlass.range_constexpr(self._world_size):
                        if cutlass.const_expr(shard != self._rank):
                            first = self._shard_start(Int32(shard), size_packs)
                            end = self._shard_start(Int32(shard + 1), size_packs)
                            lo = first + chunk * chunk_packs
                            hi = cutlass.min(end, lo + chunk_packs)
                            gather_index = lo + index
                            while gather_index < hi:
                                words = ld_relaxed_sys_v4_u32(
                                    recv_base
                                    + Int64(shard * 2 + 1) * shard_cap
                                    + Int64(gather_index - first) * Int64(PACK_BYTES)
                                )
                                st_global_v4_u32(
                                    output_base
                                    + Int64(gather_index) * Int64(PACK_BYTES),
                                    words[0],
                                    words[1],
                                    words[2],
                                    words[3],
                                )
                                gather_index += stride

        # 5. the last block to finish publishes the next epoch
        fence_sc_gpu()
        cute.arch.sync_threads()
        if Int32(tidx) == Int32(0):
            prior = atomic_add_relaxed_gpu_u32(tail_counter, Uint32(1))
            if (prior + Uint32(1)) % Uint32(gdim) == Uint32(0):
                fence_sc_gpu()
                if ld_relaxed_sys_u32(ctrl_base + Int64(8)) == Uint32(0):
                    st_release_gpu_u32(epoch_ptr, seq)


def _dummy(dtype, alignment: int):
    """A CUDA pointer of ``dtype`` used to trace launcher argument types."""
    return make_ptr(dtype, 16, cute.AddressSpace.gmem, assumed_align=alignment)


def _process_key(*key) -> tuple[object, ...]:
    return tuple(key)


def is_launcher_prepared(*key) -> bool:
    """Return whether this exact process-local launcher is already loaded."""
    return _process_key(*key) in _PREPARED_LAUNCHERS


@program_cache
def get_launcher(
    dtype_name: str | torch.dtype,
    world_size: int,
    rank: int,
    threads: int,
    flag_stride: int,
    hca_count: int,
    max_chunks: int,
    device_index: int,
) -> Callable[..., None]:
    """Compile the two-shot launcher for ``key`` once and return it."""
    if isinstance(dtype_name, torch.dtype):
        try:
            dtype_name = _DTYPE_NAMES[dtype_name]
        except KeyError:
            raise ValueError(
                f"unsupported RoCE two-shot dtype {dtype_name!r}"
            ) from None
    process_key = _process_key(
        str(dtype_name),
        int(world_size),
        int(rank),
        int(threads),
        int(flag_stride),
        int(hca_count),
        int(max_chunks),
        int(device_index),
    )
    launch = _RoceTwoshotLaunch(
        dtype_name, world_size, rank, threads, flag_stride, hca_count, max_chunks
    )
    cache_key = process_key[:-1]
    raise_if_kernel_resolution_frozen(
        "cute.compile", target=launch, cache_key=cache_key
    )
    raw = b12x_compile(
        launch,
        _dummy(cutlass.Uint32, 16),
        _dummy(cutlass.Uint32, 16),
        1,
        16,
        1,
        1,
        16,
        16,
        16,
        4096,
        16,
        16,
        16,
        16,
        1,
        1,
        current_cuda_stream(),
        compile_spec=KernelCompileSpec.from_key("comm.roce.twoshot", 2, cache_key),
    )

    def run(
        input_address: int,
        output_address: int,
        size_packs: int,
        nbytes: int,
        chunk_packs: int,
        chunks: int,
        send_base: int,
        recv_base: int,
        flag_base: int,
        shard_cap: int,
        ctrl_base: int,
        epoch_address: int,
        counter_base: int,
        poison_address: int,
        spin_limit: int,
        grid_x: int,
    ) -> None:
        """Launch the compiled kernel with runtime scalar arguments."""
        raw(
            make_ptr(
                cutlass.Uint32, input_address, cute.AddressSpace.gmem, assumed_align=16
            ),
            make_ptr(
                cutlass.Uint32, output_address, cute.AddressSpace.gmem, assumed_align=16
            ),
            int(size_packs),
            int(nbytes),
            int(chunk_packs),
            int(chunks),
            int(send_base),
            int(recv_base),
            int(flag_base),
            int(shard_cap),
            int(ctrl_base),
            int(epoch_address),
            int(counter_base),
            int(poison_address),
            int(spin_limit),
            int(grid_x),
            current_cuda_stream(),
        )

    _PREPARED_LAUNCHERS.add(process_key)
    return attach_programs(run, raw)


__all__ = ["get_launcher", "is_launcher_prepared"]
