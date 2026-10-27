"""Small-batch routed MoE synchronization helpers."""

from __future__ import annotations

from cutlass.cutlass_dsl import Int32
from b12x._lib.intrinsics.atomics import atomic_add_global_i32
import cutlass.cute as cute
from b12x._lib.intrinsics.memory import get_ptr_as_int64
from b12x._lib.intrinsics.sync import spin_wait_global_eq_i32
from b12x._lib.intrinsics.atomics import st_global_i32
from b12x._lib.intrinsics.sync import st_global_release_i32


@cute.jit
def _token_publish_fc1_ready(
    barrier_count: cute.Tensor,
    barrier_epoch: cute.Tensor,
    token_idx: Int32,
    expected_epoch: Int32,
    chunks_per_token: Int32,
    is_cta_leader: Int32,
):
    if is_cta_leader > Int32(0):
        count_addr = get_ptr_as_int64(barrier_count, token_idx)
        epoch_addr = get_ptr_as_int64(barrier_epoch, token_idx)
        arrived = atomic_add_global_i32(count_addr, Int32(1))
        if arrived == chunks_per_token - Int32(1):
            st_global_i32(count_addr, Int32(0))
            st_global_release_i32(epoch_addr, expected_epoch + Int32(1))


@cute.jit
def _token_wait_fc1_ready(
    barrier_epoch: cute.Tensor,
    token_idx: Int32,
    expected_epoch: Int32,
    is_cta_leader: Int32,
):
    cute.arch.sync_threads()
    if is_cta_leader > Int32(0):
        epoch_addr = get_ptr_as_int64(barrier_epoch, token_idx)
        spin_wait_global_eq_i32(epoch_addr, expected_epoch)
    cute.arch.sync_threads()
