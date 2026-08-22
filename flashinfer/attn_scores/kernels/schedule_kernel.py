# Copyright (c) 2025 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Adapted from TensorRT-LLM (Apache 2.0):
# tests/scripts/cute_dsl_kernels/paged_mqa_logits/run_fp4.py (PagedMQALogitsMetadataKernel)
# Original copyright: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""GPU-side CTA schedule kernel for paged MQA logits.

Replaces the CPU numpy schedule computation (~60µs) with a single-warp CUDA
kernel (~8µs) that runs entirely on-device.  This enables CUDA-graph capture
of the full paged MQA logits dispatch pipeline with no CPU-side work per step.

The kernel is a single 32-lane warp that:
  1. Loads context_lens into per-lane registers, computes num_segments
  2. Warp-level inclusive scan with carry → SMEM prefix_sum array
  3. Each lane handles a strip of CTA indices to emit (q_idx, kv_split) rows
"""

import functools
from typing import Tuple

import cutlass
import cutlass.cute as cute
import cuda.bindings.driver as cuda
from cutlass.utils.smem_allocator import SmemAllocator


class PagedMQALogitsScheduleKernel:
    """Single-warp GPU kernel that computes paged MQA logits CTA schedule.

    Compile-time params: aligned_batch_size (multiple of 32), split_kv, num_sms.
    Runtime: context_lens [B] int32 (CUDA), schedule_meta [num_sms+1, 2] int32 (CUDA),
    batch_size int32.

    Produces the same [num_sms+1, 2] int32 output as _compute_schedule_metadata()
    but entirely on-device with no D2H copy of context_lens.
    """

    def __init__(self, aligned_batch_size: int, split_kv: int, num_sms: int):
        assert aligned_batch_size > 0 and aligned_batch_size % 32 == 0
        self.aligned_batch_size = aligned_batch_size
        self.split_kv = split_kv
        self.num_sms = num_sms

    @cute.jit
    def __call__(
        self,
        context_lens: cute.Tensor,
        schedule_meta: cute.Tensor,
        batch_size: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(context_lens, schedule_meta, batch_size).launch(
            grid=(1, 1, 1), block=(32, 1, 1), stream=stream
        )

    @cute.kernel
    def kernel(
        self,
        context_lens: cute.Tensor,
        schedule_meta: cute.Tensor,
        batch_size: cutlass.Int32,
    ):
        kAligned = cutlass.const_expr(self.aligned_batch_size)
        SPLIT_KV = cutlass.const_expr(self.split_kv)
        kNumSMs = cutlass.const_expr(self.num_sms)
        kNumChunks = cutlass.const_expr(kAligned // 32)
        # Halvings needed for the phase-3 partition search to converge.
        kLog2Aligned = cutlass.const_expr(int(kAligned).bit_length() + 1)
        # Cover sm_idx ∈ [0, kNumSMs] (inclusive) in 32-lane strips.
        kMaxSmChunks = cutlass.const_expr((kNumSMs + 32) // 32)

        lane_idx = cute.arch.lane_idx()

        smem = SmemAllocator()
        prefix_sum = smem.allocate_tensor(
            element_type=cutlass.Int32,
            layout=cute.make_ordered_layout((kAligned,), order=(0,)),
            byte_alignment=128,
        )

        # Phase 1: per-lane register array of ceil_div(ctx, SPLIT_KV).
        # Out-of-range lanes contribute 0 (matches CUDA q_idx<batch_size guard).
        num_segs = [cutlass.Int32(0)] * kNumChunks
        for k in cutlass.range_constexpr(kNumChunks):
            q_idx = cutlass.Int32(k * 32) + lane_idx
            ctx_len = cutlass.Int32(0)
            if q_idx < batch_size:
                ctx_len = context_lens[q_idx]
            num_segs[k] = (ctx_len + (SPLIT_KV - 1)) // SPLIT_KV

        # Phase 2: warp-level inclusive scan with carry across chunks → SMEM.
        sum_carry = cutlass.Int32(0)
        for k in cutlass.range_constexpr(kNumChunks):
            x = num_segs[k]
            for i in cutlass.range_constexpr(5):  # log2(32) = 5
                offset = 1 << i
                y = cute.arch.shuffle_sync_up(
                    x, offset, mask=0xFFFFFFFF, mask_and_clamp=0
                )
                if lane_idx >= offset:
                    x = x + y
            x = x + sum_carry
            prefix_sum[k * 32 + lane_idx] = x
            # Broadcast lane-31's inclusive sum to all lanes for next chunk carry.
            sum_carry = cute.arch.shuffle_sync(x, 31)

        # Phase 3: distribute total segments evenly across kNumSMs CTAs.
        # Each lane processes a strip of CTA indices.
        total = sum_carry
        q_div = total // kNumSMs
        r_mod = total % kNumSMs

        for s in cutlass.range_constexpr(kMaxSmChunks):
            sm_idx_local = cutlass.Int32(s * 32) + lane_idx
            if sm_idx_local <= kNumSMs:
                # seg_starts = sm * q_div + min(sm, r_mod)
                seg_starts = sm_idx_local * q_div
                if sm_idx_local <= r_mod:
                    seg_starts = seg_starts + sm_idx_local
                else:
                    seg_starts = seg_starts + r_mod

                # q_idx_out = number of fully-assigned sequences =
                # count{ j < batch_size : prefix_sum[j] <= seg_starts }.
                # prefix_sum is non-decreasing, so the predicate is monotone and
                # this is an upper_bound -- binary search it.
                #
                # A linear scan here (as in the kernel this was adapted from) is
                # O(kAligned) twice over: every lane walks the whole array at
                # runtime, and because range_constexpr pastes the body once per
                # iteration inside the kMaxSmChunks strip loop, the emitted IR
                # grows by ~kMaxSmChunks * kAligned instructions. Compile time
                # then grows faster than linearly, since the passes over the
                # resulting single large basic block are superlinear. Measured on
                # sm_100a at B=2048: 138.6s -> 0.8s to compile, 0.144ms -> 0.014ms
                # per launch.
                #
                # The trip count stays a compile-time constant, so constexpr
                # unrolling remains the right choice: 13 iterations at B=2048.
                # Ties matter -- a zero-length row contributes no segments, so
                # prefix_sum repeats and every duplicate must be counted; that is
                # why the predicate is <= and not <.
                lo = cutlass.Int32(0)
                cnt = cutlass.Int32(batch_size)
                for _ in cutlass.range_constexpr(kLog2Aligned):
                    half = cnt // 2
                    mid = lo + half
                    take = cutlass.Boolean(False)
                    if cnt > 0:
                        if prefix_sum[mid] <= seg_starts:
                            take = cutlass.Boolean(True)
                    if take:
                        lo = mid + 1
                        cnt = cnt - half - 1
                    else:
                        cnt = half
                q_idx_out = lo

                kv_split_idx = seg_starts
                if q_idx_out > 0:
                    kv_split_idx = seg_starts - prefix_sum[q_idx_out - 1]

                schedule_meta[sm_idx_local, 0] = q_idx_out
                schedule_meta[sm_idx_local, 1] = kv_split_idx


@functools.cache
def _cached_schedule_source_files() -> Tuple[str, ...]:
    """Files whose contents change this kernel's generated code.

    Only this module: the schedule kernel's codegen is fully determined by
    ``PagedMQALogitsScheduleKernel`` plus the four values in the cache key.
    Deliberately excludes attn_scores.py -- unlike the FP8/FP4 wrappers, no
    shape preparation for this kernel happens there, so including it would
    invalidate every cached bucket on any unrelated edit to that module.
    """
    return (__file__,)


@functools.cache
def _compile_schedule_kernel(aligned_b: int, split_kv: int, num_sms: int, arch: str):
    """Compile GPU schedule kernel; cached by (aligned_b, split_kv, num_sms, arch).

    ``arch`` pins codegen to the target device rather than to CUTLASS's
    default probe, which queries ordinal 0 unconditionally.

    Backed by the same persistent on-disk cache as the FP8/FP4 kernels. The
    in-process ``functools.cache`` only spares the reload within one process;
    without the disk layer every worker recompiled every batch bucket, because
    ``_gpu_schedule`` specializes on ``ceil(batch_size / 32) * 32``.
    """
    from ...jit.cute_dsl_core import build_and_load_cute_dsl_kernel

    sym_B = cute.sym_int()
    cl_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (sym_B,), stride_order=(0,)
    )
    sm_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (num_sms + 1, 2), stride_order=(1, 0)
    )
    fake_stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

    kern = PagedMQALogitsScheduleKernel(aligned_b, split_kv, num_sms)

    def _compile_fn():
        return cute.compile(
            kern,
            cl_fake,
            sm_fake,
            cutlass.Int32(1),
            fake_stream,
            options=f"--gpu-arch {arch} --enable-tvm-ffi",
        )

    return build_and_load_cute_dsl_kernel(
        "attn_scores_schedule",
        f"sched_b{aligned_b}_split{split_kv}_sms{num_sms}_{arch}",
        _compile_fn,
        extra_key_files=_cached_schedule_source_files(),
    )
