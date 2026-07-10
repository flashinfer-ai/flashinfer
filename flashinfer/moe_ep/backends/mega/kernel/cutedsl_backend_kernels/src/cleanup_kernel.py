"""Host-callable cuTeDSL cleanup kernel for the dispatch workspace.

Replicates the cleanup chunk of mega_moe's dispatch slice
(``DeepGEMM/deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh``
lines 723-766) as a standalone kernel so the dispatch kernel can be
re-launched without observing residual workspace state from the previous
round.

Cleared regions
---------------
* ``expert_send_count`` (``num_total_experts`` u64) -- the per-rank
  publisher histogram populated by ``_dispatch_prep`` round 2 (mega_moe
  line 462-466). Mega_moe clears this on SM 0 (line 727-728).
* ``expert_recv_count`` (``num_ranks`` x ``num_experts_per_rank`` u64) --
  the per-(src_rank, local_expert) recv-count table populated by
  ``_dispatch_barrier`` (mega_moe line 547). Mega_moe clears this on the
  non-zero SMs as part of the per-expert sweep at lines 748-749; this
  replica flattens the ``num_ranks * num_experts_per_rank`` entries and
  clears them from SM 0 since the DSV4 problem size is small (4 * 64 = 256
  u64 entries).
* ``expert_recv_count_sum`` (``num_experts_per_rank`` u64) -- per-expert
  publisher count + token sum. Mega_moe clears this on the non-zero SMs at
  line 744-745; this replica clears it on SM 0.
* ``l1_arrival_count`` (``num_max_task_tiles`` u32) -- the per-task-tile
  release-add counter the GEMM consumer spins on. Mega_moe clears only the
  *populated* prefix per expert (lines 752-754) using the freshly read
  ``num_recv_m_blocks``; this replica conservatively clears the full pool
  via SMs 1..num_sms-1 because the kernel does not have access to the
  per-expert counts post-barrier (they are themselves cleared above).
* ``nvlink_barrier_signal`` (``num_ranks`` x 2 i32) -- both barrier slots
  (slot 0 = pre-pull, slot 1 = kernel-tail) of the symmetric NVLink signal
  buffer. Mega_moe carries the signal across rounds via a sign-toggle
  scheme (``barrier.cuh`` lines 28-72); the DSV4 cuTeDSL replica fires each
  slot exactly once per launch and resets both to zero here.
* ``grid_sync_counter`` (1 or 2 u32) -- per-rank scratch slot for the
  software grid sync. The phase-flip pattern in ``grid_sync.py`` leaves
  this slot at one of ``{0x80000000, 0}`` after every round, so a future
  round would observe the *previous* terminal value as the round-0 phase
  bit. Resetting to zero here makes every dispatch launch start from the
  same canonical phase regardless of how many barriers the previous
  launch executed.

Not cleared (per DEC-12 + cleanup plan)
---------------------------------------
* ``src_token_topk_idx`` -- overwritten in full by the next dispatch
  round 3 advertise STG (mega_moe line 469-475).
* ``token_src_metadata`` -- only valid pool-token rows are written by
  ``_dispatch_pull`` (mega_moe lines 692-693); the padding rows are
  sentinel-init'd by the next dispatch's pool walker.
* ``l1_token_buffer``, ``l1_sf_buffer``, ``l1_topk_weights_buffer`` --
  data-plane buffers; their valid regions are TMA-stored over by the
  next dispatch.

Work distribution
-----------------
SM 0 clears the small counter region (256 + 256 + 64 u64s + 8 i32s + 2 u32s
~ 4.7 KB total). SMs 1..num_sms-1 cooperatively clear the
``num_max_task_tiles`` u32 ``l1_arrival_count`` slice; for DSV4 the slice is
192 entries which fits in one SM's clearing pass, but the slot_per_sm
striping below scales to larger pools without code changes.
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
from cutlass.cutlass_dsl import Int32, Int64, Uint32


_CLEANUP_THREADS_PER_CTA = 384


@cute.kernel
def cleanup_kernel(
    expert_send_count: cute.Tensor,
    expert_recv_count: cute.Tensor,
    expert_recv_count_sum: cute.Tensor,
    l1_arrival_count: cute.Tensor,
    nvlink_barrier_signal: cute.Tensor,
    grid_sync_counter: cute.Tensor,
    num_total_experts: cutlass.Constexpr[int],
    num_experts_per_rank: cutlass.Constexpr[int],
    num_ranks: cutlass.Constexpr[int],
    num_max_task_tiles: cutlass.Constexpr[int],
    num_sms: cutlass.Constexpr[int],
    grid_sync_counter_len: cutlass.Constexpr[int],
):
    """Zero the dispatch workspace counter / signal regions.

    Mirrors ``sm100_fp8_fp4_mega_moe.cuh`` lines 723-766. SM 0 clears the
    small counter region (``expert_send_count``, ``expert_recv_count``,
    ``expert_recv_count_sum``, ``nvlink_barrier_signal`` slot 0 + slot 1,
    ``grid_sync_counter``); SMs 1..num_sms-1 cooperatively clear
    ``l1_arrival_count``.
    """
    sm_idx = cute.arch.block_idx()[0]
    tid = cute.arch.thread_idx()[0]

    if sm_idx == Int32(0):
        for offset in cutlass.range(0, num_total_experts, _CLEANUP_THREADS_PER_CTA):
            i = Int32(offset) + tid
            if i < Int32(num_total_experts):
                expert_send_count[i] = Int64(0)

        recv_count_total: cutlass.Constexpr[int] = num_ranks * num_experts_per_rank
        for offset in cutlass.range(0, recv_count_total, _CLEANUP_THREADS_PER_CTA):
            i = Int32(offset) + tid
            if i < Int32(recv_count_total):
                rank_idx = i // Int32(num_experts_per_rank)
                expert_idx = i % Int32(num_experts_per_rank)
                expert_recv_count[rank_idx, expert_idx] = Int64(0)

        for offset in cutlass.range(0, num_experts_per_rank, _CLEANUP_THREADS_PER_CTA):
            i = Int32(offset) + tid
            if i < Int32(num_experts_per_rank):
                expert_recv_count_sum[i] = Int64(0)

        # Both NVLink barrier slots: slot 0 (pre-pull) and slot 1 (kernel-tail).
        nvlink_signal_total: cutlass.Constexpr[int] = num_ranks * 2
        for offset in cutlass.range(0, nvlink_signal_total, _CLEANUP_THREADS_PER_CTA):
            i = Int32(offset) + tid
            if i < Int32(nvlink_signal_total):
                nvlink_barrier_signal[i] = Int32(0)

        for offset in cutlass.range(0, grid_sync_counter_len, _CLEANUP_THREADS_PER_CTA):
            i = Int32(offset) + tid
            if i < Int32(grid_sync_counter_len):
                grid_sync_counter[i] = Uint32(0)
    else:
        # SMs 1..num_sms-1 split l1_arrival_count clearing. The
        slot_per_sm: cutlass.Constexpr[int] = (num_max_task_tiles + num_sms - 2) // (
            num_sms - 1
        )
        my_start = (sm_idx - Int32(1)) * Int32(slot_per_sm)
        my_end_unclamped = my_start + Int32(slot_per_sm)
        end_limit = Int32(num_max_task_tiles)
        my_end = my_end_unclamped if my_end_unclamped < end_limit else end_limit

        i = my_start + tid
        while i < my_end:
            l1_arrival_count[i] = Uint32(0)
            i = i + Int32(_CLEANUP_THREADS_PER_CTA)
