"""CuTe DSL kernel for the RoCE variable-length all-gather.

Every rank contributes a byte count it and its peers read from the same device
``counts`` tensor, so the sizes are runtime values inside a CUDA graph.  The
contribution streams in chunks through ``varlen_recv[seq & 1][rank]`` in the
pinned region (see ``_roce_proxy.c``), and every rank receives each peer's
chunks in that peer's row:

1. check: every block reads all counts; one that is not a multiple of 16 bytes
   or exceeds the call's limit poisons the runtime on every rank alike, before
   any doorbell, so nobody waits for a contribution that will never come;
2. doorbell: block 0 records the op (variable-length kind, our byte count and
   chunk geometry) in ``seq & 1``'s slot of the control record and rings ``seq``;
3. stage: blocks copy chunk ``c`` of the source into our row and, unless the
   source is the output's row ``rank`` itself, into that row; the last block to finish a chunk publishes
   ``ready_vl[c] = seq`` and the proxy writes that chunk to our row on every peer;
4. receive: per chunk, wait for every peer's stripe flags of the chunks that
   peer sends, then copy them from its row into the output's row for that peer
   with system-scope loads;
5. epoch: the last block to finish advances the shared device epoch.

A contribution of zero bytes still sends one chunk of flags, so receivers wait
the same way for every peer.  The chunk counters are free-running modulo the
runtime's fixed power-of-two grid.  No block waits for another block.
"""

from __future__ import annotations

from collections.abc import Callable

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64, Uint32

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

PACK_BYTES = 16
# Control-record words (see _roce_proxy.c).
_CTRL_SLOT_KIND = 8
_CTRL_SLOT_CHUNK_PACKS = 10
_CTRL_SLOT_CHUNKS = 12
_CTRL_READY_VL = 256
_KIND_VARLEN = 2
# ctrl word 7: why a kernel poisoned the runtime without a timeout.
ERROR_KIND_INVALID_COUNT = 1
_PREPARED_LAUNCHERS: set[tuple[object, ...]] = set()


class _RoceVarlenLaunch:
    def __init__(
        self,
        world_size: int,
        rank: int,
        threads: int,
        flag_stride: int,
        hca_count: int,
        max_chunks: int,
    ) -> None:
        """Bind one specialization: world size, rank, and layout constants."""
        if int(threads) < int(world_size) * int(hca_count):
            raise ValueError(
                "RoCE kernels need threads >= world_size * hca_count "
                f"(one thread per stripe flag), got threads={threads} "
                f"world_size={world_size} hca_count={hca_count}"
            )
        self._world_size = int(world_size)
        self._rank = int(rank)
        self._threads = int(threads)
        self._flag_stride = int(flag_stride)
        self._hca_count = int(hca_count)
        self._max_chunks = int(max_chunks)

    @cute.jit
    def _packs(self, counts_base: Int64, peer: Int32) -> Int32:
        """Packs ``peer`` contributes (the low word of its byte count; validity is checked once)."""
        return Int32(
            ld_relaxed_gpu_u32(counts_base + Int64(8) * Int64(peer)) >> Uint32(4)
        )

    @cute.jit
    def _chunks(self, packs: Int32, chunk_packs: Int32) -> Int32:
        """Chunks that carry ``packs``; an empty contribution sends one chunk of flags."""
        return cutlass.max(Int32(1), (packs + chunk_packs - Int32(1)) // chunk_packs)

    @cute.jit
    def _invalid_peer(self, counts_base: Int64, limit_packs: Int32) -> Int32:
        """The lowest rank whose byte count is unusable, or -1 when all are valid."""
        bad = Int32(-1)
        for peer in cutlass.range_constexpr(self._world_size):
            low = ld_relaxed_gpu_u32(counts_base + Int64(8 * peer))
            high = ld_relaxed_gpu_u32(counts_base + Int64(8 * peer + 4))
            if bad < Int32(0):
                if high != Uint32(0):
                    bad = Int32(peer)
                if (low & Uint32(15)) != Uint32(0):
                    bad = Int32(peer)
                if Int32(low >> Uint32(4)) > limit_packs:
                    bad = Int32(peer)
        return bad

    @cute.jit
    def _arrive(
        self, counter: Int64, ready: Int64, seq: Uint32, gdim: Int32, ctrl_base: Int64
    ) -> None:
        """Publish this block's writes; the last of ``gdim`` arrivals sets ``ready = seq``.

        Nothing is published once the runtime is poisoned, so the proxy never
        sends a chunk of an op that failed.
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
        source_ptr: cute.Pointer,
        output_ptr: cute.Pointer,
        counts_base: Int64,
        limit_packs: Int32,
        output_row_bytes: Int64,
        copy_own: Int32,
        chunk_packs: Int32,
        recv_base: Int64,
        flag_base: Int64,
        row_bytes: Int64,
        ctrl_base: Int64,
        epoch_ptr: Int64,
        counter_base: Int64,
        poison_ptr: Int64,
        spin_limit: Uint32,
        grid_x: Int32,
        stream: cuda.CUstream,
    ) -> None:
        """Host entry: launch the variable-length all-gather kernel with runtime scalars."""
        self.kernel(
            source_ptr,
            output_ptr,
            counts_base,
            limit_packs,
            output_row_bytes,
            copy_own,
            chunk_packs,
            recv_base,
            flag_base,
            row_bytes,
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
        source_ptr: cute.Pointer,
        output_ptr: cute.Pointer,
        counts_base: Int64,
        limit_packs: Int32,
        output_row_bytes: Int64,
        copy_own: Int32,
        chunk_packs: Int32,
        recv_base: Int64,
        flag_base: Int64,
        row_bytes: Int64,
        ctrl_base: Int64,
        epoch_ptr: Int64,
        counter_base: Int64,
        poison_ptr: Int64,
        spin_limit: Uint32,
    ) -> None:
        """Device kernel: check counts, doorbell, stage our chunks, receive every peer's, advance the epoch."""
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        gdim, _, _ = cute.arch.grid_dim()
        source_base = Int64(source_ptr.toint())
        output_base = Int64(output_ptr.toint())
        epoch = ld_relaxed_gpu_u32(epoch_ptr)
        seq = epoch + Uint32(1)
        slot = Int64(seq & Uint32(1))
        index = Int32(bidx) * Int32(self._threads) + Int32(tidx)
        stride = Int32(gdim) * Int32(self._threads)
        stage_counters = counter_base
        tail_counter = counter_base + Int64(4 * self._max_chunks)
        own_row = (
            recv_base + (slot * Int64(self._world_size) + Int64(self._rank)) * row_bytes
        )
        own_output = output_base + Int64(self._rank) * output_row_bytes

        poisoned = ld_relaxed_gpu_u32(poison_ptr)
        bad = self._invalid_peer(counts_base, limit_packs)
        if poisoned == Uint32(0):
            if bad >= Int32(0):
                # 1. every rank reads the same counts, so every rank stops here alike
                if Int32(tidx) == Int32(0):
                    st_relaxed_sys_u32(ctrl_base + Int64(12), Uint32(bad))
                    st_relaxed_sys_u32(
                        ctrl_base + Int64(28), Uint32(ERROR_KIND_INVALID_COUNT)
                    )
                    st_relaxed_sys_u32(ctrl_base + Int64(8), seq)
                    st_release_gpu_u32(poison_ptr, seq)
            else:
                own_packs = self._packs(counts_base, Int32(self._rank))
                own_chunks = self._chunks(own_packs, chunk_packs)
                # 2. the op record, then the doorbell; the proxy then follows the ready words
                if Int32(bidx) == Int32(0):
                    if Int32(tidx) == Int32(0):
                        nbytes = Uint32(own_packs) * Uint32(PACK_BYTES)
                        st_relaxed_sys_u32(ctrl_base + Int64(4), nbytes)
                        st_relaxed_sys_u32(
                            ctrl_base + Int64(16) + slot * Int64(4), nbytes
                        )
                        st_relaxed_sys_u32(
                            ctrl_base + Int64(4 * _CTRL_SLOT_KIND) + slot * Int64(4),
                            Uint32(_KIND_VARLEN),
                        )
                        st_relaxed_sys_u32(
                            ctrl_base
                            + Int64(4 * _CTRL_SLOT_CHUNK_PACKS)
                            + slot * Int64(4),
                            Uint32(chunk_packs),
                        )
                        st_relaxed_sys_u32(
                            ctrl_base + Int64(4 * _CTRL_SLOT_CHUNKS) + slot * Int64(4),
                            Uint32(own_chunks),
                        )
                        fence_sc_sys()
                        st_relaxed_sys_u32(ctrl_base, seq)

                # 3. stage our contribution chunk by chunk into our row and our output row
                for chunk in cutlass.range(own_chunks, unroll=1):
                    lo = chunk * chunk_packs
                    hi = cutlass.min(own_packs, lo + chunk_packs)
                    stage_index = lo + index
                    while stage_index < hi:
                        offset = Int64(stage_index) * Int64(PACK_BYTES)
                        words = ld_global_v4_u32(source_base + offset)
                        st_global_v4_u32(
                            own_row + offset, words[0], words[1], words[2], words[3]
                        )
                        if copy_own != Int32(0):
                            st_global_v4_u32(
                                own_output + offset,
                                words[0],
                                words[1],
                                words[2],
                                words[3],
                            )
                        stage_index += stride
                    cute.arch.sync_threads()
                    self._arrive(
                        stage_counters + Int64(4) * Int64(chunk),
                        ctrl_base + Int64(4 * _CTRL_READY_VL) + Int64(4) * Int64(chunk),
                        seq,
                        Int32(gdim),
                        ctrl_base,
                    )

                # 4. receive every peer's chunks as their flags arrive
                longest = Int32(1)
                for peer in cutlass.range_constexpr(self._world_size):
                    if cutlass.const_expr(peer != self._rank):
                        longest = cutlass.max(
                            longest,
                            self._chunks(
                                self._packs(counts_base, Int32(peer)), chunk_packs
                            ),
                        )
                for chunk in cutlass.range(longest, unroll=1):
                    if ld_relaxed_gpu_u32(poison_ptr) == Uint32(0):
                        if Int32(tidx) < Int32(self._world_size * self._hca_count):
                            peer = Int32(tidx) // Int32(self._hca_count)
                            hca = Int32(tidx) - peer * Int32(self._hca_count)
                            if peer != Int32(self._rank):
                                if chunk < self._chunks(
                                    self._packs(counts_base, peer), chunk_packs
                                ):
                                    flag_addr = flag_base + (
                                        (
                                            (
                                                slot * Int64(self._world_size)
                                                + Int64(peer)
                                            )
                                            * Int64(self._max_chunks)
                                            + Int64(chunk)
                                        )
                                        * Int64(self._hca_count)
                                        + Int64(hca)
                                    ) * Int64(self._flag_stride)
                                    timed_out = spin_until_eq_acquire_sys(
                                        flag_addr, seq, spin_limit
                                    )
                                    if timed_out != Uint32(0):
                                        st_relaxed_sys_u32(
                                            ctrl_base + Int64(12), Uint32(peer)
                                        )
                                        st_relaxed_sys_u32(
                                            ctrl_base + Int64(24), Uint32(hca)
                                        )
                                        st_relaxed_sys_u32(ctrl_base + Int64(8), seq)
                                        st_release_gpu_u32(poison_ptr, seq)
                    cute.arch.sync_threads()
                    if ld_relaxed_gpu_u32(poison_ptr) == Uint32(0):
                        for peer in cutlass.range_constexpr(self._world_size):
                            if cutlass.const_expr(peer != self._rank):
                                packs = self._packs(counts_base, Int32(peer))
                                lo = chunk * chunk_packs
                                hi = cutlass.min(packs, lo + chunk_packs)
                                row = (
                                    recv_base
                                    + (slot * Int64(self._world_size) + Int64(peer))
                                    * row_bytes
                                )
                                output_row = (
                                    output_base + Int64(peer) * output_row_bytes
                                )
                                copy_index = lo + index
                                while copy_index < hi:
                                    offset = Int64(copy_index) * Int64(PACK_BYTES)
                                    words = ld_relaxed_sys_v4_u32(row + offset)
                                    st_global_v4_u32(
                                        output_row + offset,
                                        words[0],
                                        words[1],
                                        words[2],
                                        words[3],
                                    )
                                    copy_index += stride

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
    world_size: int,
    rank: int,
    threads: int,
    flag_stride: int,
    hca_count: int,
    max_chunks: int,
    device_index: int,
) -> Callable[..., None]:
    """Compile the variable-length all-gather launcher for ``key`` once and return it."""
    process_key = _process_key(
        int(world_size),
        int(rank),
        int(threads),
        int(flag_stride),
        int(hca_count),
        int(max_chunks),
        int(device_index),
    )
    launch = _RoceVarlenLaunch(
        world_size, rank, threads, flag_stride, hca_count, max_chunks
    )
    cache_key = process_key[:-1]
    raise_if_kernel_resolution_frozen(
        "cute.compile", target=launch, cache_key=cache_key
    )
    raw = b12x_compile(
        launch,
        _dummy(cutlass.Uint32, 16),
        _dummy(cutlass.Uint32, 16),
        16,
        1,
        4096,
        1,
        1,
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
        compile_spec=KernelCompileSpec.from_key("comm.roce.varlen", 1, cache_key),
    )

    def run(
        source_address: int,
        output_address: int,
        counts_address: int,
        limit_packs: int,
        output_row_bytes: int,
        copy_own: int,
        chunk_packs: int,
        recv_base: int,
        flag_base: int,
        row_bytes: int,
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
                cutlass.Uint32, source_address, cute.AddressSpace.gmem, assumed_align=16
            ),
            make_ptr(
                cutlass.Uint32, output_address, cute.AddressSpace.gmem, assumed_align=16
            ),
            int(counts_address),
            int(limit_packs),
            int(output_row_bytes),
            int(copy_own),
            int(chunk_packs),
            int(recv_base),
            int(flag_base),
            int(row_bytes),
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


__all__ = ["ERROR_KIND_INVALID_COUNT", "get_launcher", "is_launcher_prepared"]
