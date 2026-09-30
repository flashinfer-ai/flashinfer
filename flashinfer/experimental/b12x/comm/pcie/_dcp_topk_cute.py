"""CuTe DSL kernel for exact owner-sharded DCP top-k transport."""

from __future__ import annotations

from b12x._lib.program_cache import program_cache
from collections.abc import Callable, Sequence

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64, Uint32

from b12x._lib.compiler import KernelCompileSpec, compile as b12x_compile
from b12x._lib.intrinsics import ld_global_v4_u32, st_global_v4_u32
from b12x._lib.runtime_control import raise_if_kernel_resolution_frozen
from b12x._lib.utils import current_cuda_stream, make_ptr

from ._dcp_cute_common import block_pair_barrier


_MAX_BLOCKS = 128
_MAX_PEERS = 8


@cute.jit
def _copy_16b(source: cute.Pointer, destination: cute.Pointer) -> None:
    values = ld_global_v4_u32(source.toint())
    st_global_v4_u32(destination.toint(), *values)


class _TopKOwnerStageLaunch:
    def __init__(self, world_size: int, rank: int, topk: int, threads: int) -> None:
        self._world_size = int(world_size)
        self._rank = int(rank)
        self._topk = int(topk)
        self._threads = int(threads)

    @cute.jit
    def __call__(
        self,
        local_indices: cute.Pointer,
        local_scores: cute.Pointer,
        candidate0: cute.Pointer,
        candidate1: cute.Pointer,
        candidate2: cute.Pointer,
        candidate3: cute.Pointer,
        candidate4: cute.Pointer,
        candidate5: cute.Pointer,
        candidate6: cute.Pointer,
        candidate7: cute.Pointer,
        signal0: cute.Pointer,
        signal1: cute.Pointer,
        signal2: cute.Pointer,
        signal3: cute.Pointer,
        signal4: cute.Pointer,
        signal5: cute.Pointer,
        signal6: cute.Pointer,
        signal7: cute.Pointer,
        rows: Int32,
        candidate_plane_elems: Int64,
        grid_x: Int32,
        wait_for_prior_consumer: Int32,
        stream: cuda.CUstream,
    ) -> None:
        self.kernel(
            local_indices,
            local_scores,
            candidate0,
            candidate1,
            candidate2,
            candidate3,
            candidate4,
            candidate5,
            candidate6,
            candidate7,
            signal0,
            signal1,
            signal2,
            signal3,
            signal4,
            signal5,
            signal6,
            signal7,
            rows,
            candidate_plane_elems,
            wait_for_prior_consumer,
        ).launch(
            grid=(grid_x, 1, 1),
            block=(self._threads, 1, 1),
            cluster=(1, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        local_indices: cute.Pointer,
        local_scores: cute.Pointer,
        candidate0: cute.Pointer,
        candidate1: cute.Pointer,
        candidate2: cute.Pointer,
        candidate3: cute.Pointer,
        candidate4: cute.Pointer,
        candidate5: cute.Pointer,
        candidate6: cute.Pointer,
        candidate7: cute.Pointer,
        signal0: cute.Pointer,
        signal1: cute.Pointer,
        signal2: cute.Pointer,
        signal3: cute.Pointer,
        signal4: cute.Pointer,
        signal5: cute.Pointer,
        signal6: cute.Pointer,
        signal7: cute.Pointer,
        rows: Int32,
        candidate_plane_elems: Int64,
        wait_for_prior_consumer: Int32,
    ) -> None:
        candidates = (
            candidate0,
            candidate1,
            candidate2,
            candidate3,
            candidate4,
            candidate5,
            candidate6,
            candidate7,
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

        owner_rows = Int64(rows) // Int64(self._world_size)
        packs_per_row = Int64(self._topk // 4)
        owner_packs = owner_rows * packs_per_row
        output_row_packs = Int64(self._world_size) * packs_per_row
        chunk = (owner_packs + Int64(gdim) - Int64(1)) // Int64(gdim)
        begin = Int64(bidx) * chunk
        end = begin + chunk
        if end > owner_packs:
            end = owner_packs

        if wait_for_prior_consumer != Int32(0):
            block_pair_barrier(
                signals,
                self_signal=signals[self._rank],
                rank=self._rank,
                world_size=self._world_size,
                max_blocks=_MAX_BLOCKS,
            )

        candidate_plane_words = candidate_plane_elems
        if cutlass.const_expr(self._world_size in (2, 4)):
            # Scalar runtime pointer selection costs TP2/TP4 an occupancy tier.
            # Retain their smaller proven specializations while keeping the
            # remaining path below faithful to CUDA's ``#pragma unroll 1``.
            for unrolled_step in cutlass.range_constexpr(self._world_size):
                unrolled_destination = (
                    self._rank + unrolled_step
                ) % self._world_size
                unrolled_destination_indices = candidates[unrolled_destination]
                unrolled_destination_scores = (
                    unrolled_destination_indices + candidate_plane_words
                )
                unrolled_input_offset = Int64(unrolled_destination) * owner_packs
                unrolled_pack = begin + Int64(tidx)
                while unrolled_pack < end:
                    unrolled_owner_row = unrolled_pack // packs_per_row
                    unrolled_column_pack = (
                        unrolled_pack - unrolled_owner_row * packs_per_row
                    )
                    unrolled_output_offset = (
                        unrolled_owner_row * output_row_packs
                        + Int64(self._rank) * packs_per_row
                        + unrolled_column_pack
                    )
                    _copy_16b(
                        local_indices
                        + (unrolled_input_offset + unrolled_pack) * Int64(4),
                        unrolled_destination_indices
                        + unrolled_output_offset * Int64(4),
                    )
                    _copy_16b(
                        local_scores
                        + (unrolled_input_offset + unrolled_pack) * Int64(4),
                        unrolled_destination_scores
                        + unrolled_output_offset * Int64(4),
                    )
                    unrolled_pack += Int64(self._threads)
        else:
            # The candidate pointers remain scalar kernel parameters; select
            # their address at runtime so TP3/TP6/TP8 do not clone the payload
            # loop.
            runtime_step = Int32(0)
            while runtime_step < Int32(self._world_size):
                runtime_destination = (
                    Int32(self._rank) + runtime_step
                ) % Int32(self._world_size)
                runtime_destination_address = Int64(candidate0.toint())
                for runtime_peer in cutlass.range_constexpr(1, self._world_size):
                    if runtime_destination == Int32(runtime_peer):
                        runtime_destination_address = Int64(
                            candidates[runtime_peer].toint()
                        )
                runtime_destination_indices = cute.make_ptr(
                    Uint32,
                    runtime_destination_address,
                    cute.AddressSpace.gmem,
                    assumed_align=16,
                )
                runtime_destination_scores = (
                    runtime_destination_indices + candidate_plane_words
                )
                runtime_input_offset = Int64(runtime_destination) * owner_packs
                runtime_pack = begin + Int64(tidx)
                while runtime_pack < end:
                    runtime_owner_row = runtime_pack // packs_per_row
                    runtime_column_pack = (
                        runtime_pack - runtime_owner_row * packs_per_row
                    )
                    runtime_output_offset = (
                        runtime_owner_row * output_row_packs
                        + Int64(self._rank) * packs_per_row
                        + runtime_column_pack
                    )
                    _copy_16b(
                        local_indices
                        + (runtime_input_offset + runtime_pack) * Int64(4),
                        runtime_destination_indices
                        + runtime_output_offset * Int64(4),
                    )
                    _copy_16b(
                        local_scores
                        + (runtime_input_offset + runtime_pack) * Int64(4),
                        runtime_destination_scores
                        + runtime_output_offset * Int64(4),
                    )
                    runtime_pack += Int64(self._threads)
                runtime_step += Int32(1)

        block_pair_barrier(
            signals,
            self_signal=signals[self._rank],
            rank=self._rank,
            world_size=self._world_size,
            max_blocks=_MAX_BLOCKS,
        )


def _pad_ptrs(values: Sequence[int], world_size: int) -> tuple[int, ...]:
    if len(values) != world_size:
        raise ValueError("pointer bundle does not match DCP world size")
    padded = [int(value) for value in values]
    padded.extend([padded[-1]] * (_MAX_PEERS - len(padded)))
    return tuple(padded)


@program_cache
def _get_compiled_topk_stage(
    world_size: int,
    rank: int,
    topk: int,
    threads: int,
) -> Callable:
    launch = _TopKOwnerStageLaunch(world_size, rank, topk, threads)
    key = (int(world_size), int(rank), int(topk), int(threads))
    raise_if_kernel_resolution_frozen(
        "cute.compile",
        target=launch,
        cache_key=key,
    )
    placeholder = make_ptr(
        Uint32,
        16,
        cute.AddressSpace.gmem,
        assumed_align=4,
    )
    raw = b12x_compile(
        launch,
        placeholder,
        placeholder,
        *(placeholder for _ in range(2 * _MAX_PEERS)),
        1,
        1,
        1,
        0,
        current_cuda_stream(),
        compile_spec=KernelCompileSpec.from_key(
            "comm.pcie.dcp_topk.owner_stage",
            4,
            key,
            labels=("world_size", "rank", "topk", "threads"),
        ),
    )

    def run(
        local_indices_ptr: int,
        local_scores_ptr: int,
        candidate_ptrs: Sequence[int],
        signal_ptrs: Sequence[int],
        rows: int,
        candidate_plane_elems: int,
        blocks: int,
        wait_for_prior_consumer: bool,
    ) -> None:
        candidates = _pad_ptrs(candidate_ptrs, world_size)
        signals = _pad_ptrs(signal_ptrs, world_size)
        raw(
            make_ptr(
                Uint32,
                int(local_indices_ptr),
                cute.AddressSpace.gmem,
                assumed_align=16,
            ),
            make_ptr(
                Uint32,
                int(local_scores_ptr),
                cute.AddressSpace.gmem,
                assumed_align=16,
            ),
            *(
                make_ptr(
                    Uint32,
                    ptr,
                    cute.AddressSpace.gmem,
                    assumed_align=16,
                )
                for ptr in candidates
            ),
            *(
                make_ptr(
                    Uint32,
                    ptr,
                    cute.AddressSpace.gmem,
                    assumed_align=4,
                )
                for ptr in signals
            ),
            int(rows),
            int(candidate_plane_elems),
            int(blocks),
            int(bool(wait_for_prior_consumer)),
            current_cuda_stream(),
        )

    return run
