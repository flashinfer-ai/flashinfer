"""Standalone dispatch-kernel entry backed by src.token_comm.

The token communication implementation lives in ``src.token_comm``.  This file
keeps the standalone ``dispatch_kernel`` entry used by tests and experiments.
"""

import cutlass
import cutlass.cute as cute
from cutlass.cutlass_dsl import Int32

try:
    from cutlass.cute import iket as _iket  # type: ignore
except ImportError:  # pragma: no cover -- fallback for wheels without cute.iket
    from .iket_compat import iket as _iket

from .token_comm import (
    TokenInPullTokenBackPush,
)


@cute.kernel
def dispatch_kernel(
    input_token_buffer: cute.Tensor,
    input_sf_buffer: cute.Tensor,
    input_topk_idx_buffer: cute.Tensor,
    input_topk_weights_buffer: cute.Tensor,
    expert_send_count: cute.Tensor,
    expert_recv_count: cute.Tensor,
    expert_recv_count_sum: cute.Tensor,
    src_token_topk_idx: cute.Tensor,
    token_src_metadata: cute.Tensor,
    l1_arrival_count: cute.Tensor,
    l1_token_buffer: cute.Tensor,
    l1_sf_buffer: cute.Tensor,
    l1_topk_weights_buffer: cute.Tensor,
    nvlink_barrier_signal: cute.Tensor,
    nvlink_barrier_counter: cute.Tensor,
    grid_sync_counter: cute.Tensor,
    peer_rank_ptr_mapper,
    local_rank: cutlass.Constexpr[int],
    world_size: cutlass.Constexpr[int],
    num_tokens: cutlass.Constexpr[int],
    num_topk: cutlass.Constexpr[int],
    num_sms: cutlass.Constexpr[int],
    num_experts_per_rank: cutlass.Constexpr[int],
    num_total_experts: cutlass.Constexpr[int],
    block_m: cutlass.Constexpr[int],
    cluster_tile_m: cutlass.Constexpr[int],
    sf_block_m: cutlass.Constexpr[int],
    hidden: cutlass.Constexpr[int],
    fc1_token_dtype: cutlass.Constexpr[type],
    sf_uint32_per_token: cutlass.Constexpr[int],
    num_padded_sf_pool_tokens: cutlass.Constexpr[int],
    flag_batch: cutlass.Constexpr[int] = 1,
):
    """Standalone token-in dispatch kernel used by tests and experiments.

    Launched with ``block = num_dispatch_warps * 32`` (default 128); there
    are no cohabit warps, so ``num_other_warps=0`` and every CTA-wide
    rendezvous in :class:`TokenInPullTokenBackPush` collapses to a
    dispatch-only barrier.
    """
    token_comm = TokenInPullTokenBackPush(
        world_size=world_size,
        num_topk=num_topk,
        num_experts_per_rank=num_experts_per_rank,
        num_total_experts=num_total_experts,
        hidden=hidden,
        fc1_token_dtype=fc1_token_dtype,
        sf_uint32_per_token=sf_uint32_per_token,
        token_padding_block=block_m,
        sf_padding_block=sf_block_m,
        cluster_tile_tokens=cluster_tile_m,
        cluster_shape_mn=(1, 1),
        dispatch_warp_start=0,
        num_other_warps=0,
        flag_batch=flag_batch,
        is_swap_ab=True,
    )
    smem = cutlass.utils.SmemAllocator()
    storage = smem.allocate(token_comm.extra_smem_storage_class())

    sm_idx = cute.arch.block_idx()[0]
    tid = cute.arch.thread_idx()[0]
    warp_idx = tid // 32
    lane_idx = tid % 32

    _iket_active = (sm_idx == Int32(0)) and (warp_idx == Int32(0))
    if _iket_active:
        _iket.range_push("Dispatch_Prep")

    token_comm.dispatch_prep(
        storage,
        input_topk_idx_buffer,
        expert_send_count,
        src_token_topk_idx,
        peer_rank_ptr_mapper,
        sm_idx,
        warp_idx,
        lane_idx,
        local_rank=local_rank,
        num_tokens=num_tokens,
        num_sms=num_sms,
    )

    if _iket_active:
        _iket.range_pop()
        _iket.range_push("Dispatch_Barrier")

    token_comm.dispatch_barrier(
        expert_send_count,
        expert_recv_count,
        expert_recv_count_sum,
        nvlink_barrier_signal,
        grid_sync_counter,
        peer_rank_ptr_mapper,
        sm_idx,
        warp_idx,
        lane_idx,
        local_rank=local_rank,
        num_sms=num_sms,
        nvlink_barrier_counter=nvlink_barrier_counter,
    )

    if _iket_active:
        _iket.range_pop()
        _iket.range_push("Dispatch_Pull")

    _, _ = token_comm.dispatch_pull(
        storage,
        input_token_buffer,
        input_sf_buffer,
        input_topk_weights_buffer,
        src_token_topk_idx,
        expert_recv_count,
        expert_recv_count_sum,
        l1_token_buffer,
        l1_sf_buffer,
        l1_topk_weights_buffer,
        l1_arrival_count,
        token_src_metadata,
        peer_rank_ptr_mapper,
        sm_idx,
        warp_idx,
        lane_idx,
        num_sms=num_sms,
    )

    if _iket_active:
        _iket.range_pop()
        _iket.range_push("Kernel_Tail")

    token_comm.nvlink_barrier(
        nvlink_barrier_signal,
        nvlink_barrier_counter,
        grid_sync_counter,
        peer_rank_ptr_mapper,
        sm_idx,
        warp_idx,
        lane_idx,
        num_sms=num_sms,
        prologue_grid_sync=True,
        epilogue_grid_sync=True,
    )

    if _iket_active:
        _iket.range_pop()
