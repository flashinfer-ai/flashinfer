"""CuTe DSL kernel for the RoCE one-shot all-reduce.

One launch performs a complete all-reduce for one message:

1. stage: copy the device input into ``send[seq & 1]`` in pinned host memory;
2. doorbell: the last block to finish staging publishes ``nbytes`` (in the
   per-slot word ``nbytes[seq & 1]``) and then ``seq`` in the control record
   that the RDMA proxy thread polls.  The doorbell is a level, not a queue:
   a proxy that was descheduled across two doorbells finds ``seq`` two ahead
   and posts both slots, which is why the byte count lives per slot;
3. wait: spin on ``flag[peer][seq & 1][hca] == seq`` for every peer and HCA
   (the peer's proxy writes each flag after that HCA's payload stripe on the
   same reliable QP); a wait that exceeds ``spin_limit`` polls records ``seq``
   in the control record's error word and the host raises instead of hanging;
4. reduce: sum the local input and every peer slot in fixed rank order, so all
   ranks produce bit-identical output, and store the result;
5. epoch: the last block to finish reduction advances the device-resident
   epoch, which makes the sequence number a runtime value rather than a launch
   argument and keeps CUDA-graph replay correct.

Staging arrivals and tail arrivals use two separate counters.  A block that
stages nothing can pass the peer wait (peers do not depend on our doorbell)
and reach the tail before a slower block has staged, so one shared counter
would ring the doorbell early and publish stale bytes.

Each power-of-two grid size has separate staging and tail counters, so message
size and launch grid remain runtime scalars and may vary across graph launches.
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
    f32_as_u32,
    fence_sc_gpu,
    fence_sc_sys,
    ld_global_v4_u32,
    ld_relaxed_gpu_u32,
    ld_relaxed_sys_u32,
    ld_relaxed_sys_v4_u32,
    pack_f32x2_to_bf16x2,
    pack_f32x2_to_f16x2,
    spin_until_eq_acquire_sys,
    st_global_v4_u32,
    st_release_gpu_u32,
    st_relaxed_sys_u32,
    u32_as_f32,
    unpack_bf16x2,
    unpack_f16x2,
)

PACK_BYTES = 16
_DTYPE_PACK_ELEMS = {"float32": 4, "float16": 8, "bfloat16": 8}
_DTYPE_NAMES = {
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
    torch.float32: "float32",
}
_PREPARED_LAUNCHERS: set[tuple[object, ...]] = set()


class _RoceOneshotLaunch:
    def __init__(
        self,
        dtype_name: str,
        world_size: int,
        rank: int,
        threads: int,
        slots: int,
        flag_stride: int,
        hca_count: int,
    ) -> None:
        """Bind one kernel specialization: dtype, world size, rank, and layout constants."""
        if dtype_name not in _DTYPE_PACK_ELEMS:
            raise ValueError(f"unsupported RoCE one-shot dtype {dtype_name!r}")
        if int(threads) < int(world_size) * int(hca_count):
            raise ValueError(
                "RoCE kernels need threads >= world_size * hca_count "
                f"(one thread per stripe flag), got threads={threads} "
                f"world_size={world_size} hca_count={hca_count}"
            )
        self._dtype_name = dtype_name
        self._pack_elems = _DTYPE_PACK_ELEMS[dtype_name]
        self._world_size = int(world_size)
        self._rank = int(rank)
        self._threads = int(threads)
        self._slots = int(slots)
        self._flag_stride = int(flag_stride)
        self._hca_count = int(hca_count)

    @cute.jit
    def _accumulate_words(
        self,
        accumulator: cute.Tensor,
        words,
        initialize: cutlass.Constexpr[bool],
    ) -> None:
        """Add one 16-byte pack of ``dtype`` values to the float32 accumulator."""
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
    def _store_accumulator(self, address: Int64, accumulator: cute.Tensor) -> None:
        """Convert the accumulator back to ``dtype`` and store one 16-byte pack."""
        packed = cute.make_rmem_tensor((4,), cutlass.Uint32)
        if cutlass.const_expr(self._dtype_name == "float32"):
            for word in cutlass.range_constexpr(4):
                packed[word] = f32_as_u32(accumulator[word])
        else:
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
    def __call__(
        self,
        input_ptr: cute.Pointer,
        output_ptr: cute.Pointer,
        size_packs: Int32,
        nbytes: Int32,
        recv_base: Int64,
        flag_base: Int64,
        send_base: Int64,
        ctrl_base: Int64,
        slot_bytes: Int64,
        epoch_ptr: Int64,
        stage_counter_ptr: Int64,
        tail_counter_ptr: Int64,
        poison_ptr: Int64,
        spin_limit: Uint32,
        grid_x: Int32,
        stream: cuda.CUstream,
    ) -> None:
        """Host entry: launch the all-reduce kernel with runtime scalars."""
        self.kernel(
            input_ptr,
            output_ptr,
            size_packs,
            nbytes,
            recv_base,
            flag_base,
            send_base,
            ctrl_base,
            slot_bytes,
            epoch_ptr,
            stage_counter_ptr,
            tail_counter_ptr,
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
        recv_base: Int64,
        flag_base: Int64,
        send_base: Int64,
        ctrl_base: Int64,
        slot_bytes: Int64,
        epoch_ptr: Int64,
        stage_counter_ptr: Int64,
        tail_counter_ptr: Int64,
        poison_ptr: Int64,
        spin_limit: Uint32,
    ) -> None:
        """Device kernel: stage, doorbell, wait for peer flags, reduce, advance the epoch."""
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        gdim, _, _ = cute.arch.grid_dim()
        input_base = Int64(input_ptr.toint())
        output_base = Int64(output_ptr.toint())
        # Every block reads the epoch before any block can advance it: the
        # advance happens only after all blocks arrived at the tail counter.
        epoch = ld_relaxed_gpu_u32(epoch_ptr)
        seq = epoch + Uint32(1)
        slot = Int64(seq & Uint32(1))
        send_slot = send_base + slot * slot_bytes

        index = Int32(bidx) * Int32(self._threads) + Int32(tidx)
        stride = Int32(gdim) * Int32(self._threads)

        # A recorded timeout poisons the runtime: later launches do nothing so
        # the host sees the failure without waiting another spin limit per op.
        # The device poison word (fourth counter) is written by the same waiting
        # threads that write the host error word and only ever goes from 0 to
        # the failed sequence, so a cheap GPU-scope load is enough here.
        poisoned = ld_relaxed_gpu_u32(poison_ptr)
        if poisoned == Uint32(0):
            # 1. stage the input into the pinned send slot
            stage_index = index
            while stage_index < size_packs:
                words = ld_global_v4_u32(
                    input_base + Int64(stage_index) * Int64(PACK_BYTES)
                )
                st_global_v4_u32(
                    send_slot + Int64(stage_index) * Int64(PACK_BYTES),
                    words[0],
                    words[1],
                    words[2],
                    words[3],
                )
                stage_index += stride
            cute.arch.sync_threads()

            # 2. the last block to finish staging rings the proxy doorbell
            if Int32(tidx) == Int32(0):
                fence_sc_sys()
                prior = atomic_add_relaxed_gpu_u32(stage_counter_ptr, Uint32(1))
                if (prior + Uint32(1)) % Uint32(gdim) == Uint32(0):
                    st_relaxed_sys_u32(ctrl_base + Int64(4), Uint32(nbytes))
                    st_relaxed_sys_u32(
                        ctrl_base + Int64(16) + slot * Int64(4), Uint32(nbytes)
                    )
                    fence_sc_sys()
                    st_relaxed_sys_u32(ctrl_base, seq)

            # 3. wait for every peer's payload-stripe flags
            if Int32(tidx) < Int32(self._world_size * self._hca_count):
                peer = Int32(tidx) // Int32(self._hca_count)
                hca = Int32(tidx) - peer * Int32(self._hca_count)
                if peer != Int32(self._rank):
                    flag_addr = flag_base + (
                        (Int64(peer) * Int64(self._slots) + slot)
                        * Int64(self._hca_count)
                        + Int64(hca)
                    ) * Int64(self._flag_stride)
                    timed_out = spin_until_eq_acquire_sys(flag_addr, seq, spin_limit)
                    if timed_out != Uint32(0):
                        st_relaxed_sys_u32(ctrl_base + Int64(12), Uint32(peer))
                        st_relaxed_sys_u32(ctrl_base + Int64(24), Uint32(hca))
                        st_relaxed_sys_u32(ctrl_base + Int64(8), seq)
                        st_release_gpu_u32(poison_ptr, seq)
            cute.arch.sync_threads()
            # A wait that timed out in this block leaves the peer slot unreliable:
            # skip the data phase so nothing derived from it is stored.
            failed = ld_relaxed_gpu_u32(poison_ptr)
            if failed == Uint32(0):
                # 4. reduce in fixed rank order so every rank stores identical bits
                reduce_index = index
                while reduce_index < size_packs:
                    accumulator = cute.make_rmem_tensor(
                        (self._pack_elems,), cutlass.Float32
                    )
                    offset = Int64(reduce_index) * Int64(PACK_BYTES)
                    for source in cutlass.range_constexpr(self._world_size):
                        if cutlass.const_expr(source == self._rank):
                            words = ld_global_v4_u32(input_base + offset)
                        else:
                            peer_slot = (
                                recv_base
                                + (Int64(source) * Int64(self._slots) + slot)
                                * slot_bytes
                            )
                            words = ld_relaxed_sys_v4_u32(peer_slot + offset)
                        self._accumulate_words(accumulator, words, source == 0)
                    self._store_accumulator(output_base + offset, accumulator)
                    reduce_index += stride

            # 5. the last block to finish reduction publishes the next epoch
            fence_sc_gpu()
            cute.arch.sync_threads()
            if Int32(tidx) == Int32(0):
                prior = atomic_add_relaxed_gpu_u32(tail_counter_ptr, Uint32(1))
                if (prior + Uint32(1)) % Uint32(gdim) == Uint32(0):
                    fence_sc_gpu()
                    # Every block's timeout store precedes its tail arrival, so the
                    # error word is final here.  A failed sequence keeps the epoch,
                    # which makes every later launch a no-op until the host raises.
                    if ld_relaxed_sys_u32(ctrl_base + Int64(8)) == Uint32(0):
                        st_release_gpu_u32(epoch_ptr, seq)


def _dummy(dtype, alignment: int):
    """A CUDA tensor of ``dtype`` used to trace launcher argument types."""
    return make_ptr(dtype, 16, cute.AddressSpace.gmem, assumed_align=alignment)


def _process_key(
    dtype_name: str,
    world_size: int,
    rank: int,
    threads: int,
    slots: int,
    flag_stride: int,
    hca_count: int,
    device_index: int,
) -> tuple[object, ...]:
    """Cache key of one compiled launcher specialization."""
    return (
        str(dtype_name),
        int(world_size),
        int(rank),
        int(threads),
        int(slots),
        int(flag_stride),
        int(hca_count),
        int(device_index),
    )


def is_launcher_prepared(*key) -> bool:
    """Return whether this exact process-local launcher is already loaded."""

    return _process_key(*key) in _PREPARED_LAUNCHERS


@program_cache
def get_launcher(
    dtype_name: str | torch.dtype,
    world_size: int,
    rank: int,
    threads: int,
    slots: int,
    flag_stride: int,
    hca_count: int,
    device_index: int,
) -> Callable[..., None]:
    """Compile the launcher for ``key`` once and return it."""
    if isinstance(dtype_name, torch.dtype):
        try:
            dtype_name = _DTYPE_NAMES[dtype_name]
        except KeyError:
            raise ValueError(f"unsupported RoCE one-shot dtype {dtype_name!r}") from None
    process_key = _process_key(
        dtype_name,
        world_size,
        rank,
        threads,
        slots,
        flag_stride,
        hca_count,
        device_index,
    )
    del device_index  # retained in the functools and preparation keys only
    launch = _RoceOneshotLaunch(
        dtype_name, world_size, rank, threads, slots, flag_stride, hca_count
    )
    cache_key = (
        str(dtype_name),
        int(world_size),
        int(rank),
        int(threads),
        int(slots),
        int(flag_stride),
        int(hca_count),
    )
    raise_if_kernel_resolution_frozen(
        "cute.compile", target=launch, cache_key=cache_key
    )
    raw = b12x_compile(
        launch,
        _dummy(cutlass.Uint32, 16),
        _dummy(cutlass.Uint32, 16),
        1,
        16,
        16,
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
        compile_spec=KernelCompileSpec.from_key("comm.roce.oneshot", 5, cache_key),
    )

    def run(
        input_address: int,
        output_address: int,
        size_packs: int,
        nbytes: int,
        recv_base: int,
        flag_base: int,
        send_base: int,
        ctrl_base: int,
        slot_bytes: int,
        epoch_address: int,
        stage_counter_address: int,
        tail_counter_address: int,
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
            int(recv_base),
            int(flag_base),
            int(send_base),
            int(ctrl_base),
            int(slot_bytes),
            int(epoch_address),
            int(stage_counter_address),
            int(tail_counter_address),
            int(poison_address),
            int(spin_limit),
            int(grid_x),
            current_cuda_stream(),
        )

    _PREPARED_LAUNCHERS.add(process_key)
    return attach_programs(run, raw)


__all__ = ["PACK_BYTES", "get_launcher", "is_launcher_prepared"]
