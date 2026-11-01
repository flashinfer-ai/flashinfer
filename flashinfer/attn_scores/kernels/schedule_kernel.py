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
  1. Walks context_lens 32 rows per iteration of a runtime loop; each lane
     computes its row's num_segments
  2. Warp-level inclusive scan with carry → dynamic SMEM prefix_sum array
  3. Each lane binary-searches prefix_sum for a strip of CTA indices and emits
     their (q_idx, kv_split) rows

batch_size is a runtime argument, so one compiled kernel serves every batch
size. prefix_sum takes 4 * batch_size bytes of dynamic SMEM, which caps the
batch at the device's per-block SMEM opt-in / 4 rows (58112 on sm_90a and
sm_100a); larger batches take the CPU path in compute_paged_mqa_logits_schedule.
"""

import functools
from typing import Tuple

import cutlass
import cutlass.cute as cute
import cuda.bindings.driver as cuda


class PagedMQALogitsScheduleKernel:
    """Single-warp GPU kernel that computes paged MQA logits CTA schedule.

    Compile-time params: split_kv, num_sms.
    Runtime: context_lens [B] int32 (CUDA), schedule_meta [num_sms+1, 2] int32 (CUDA),
    batch_size int32 (>= 1; sizes the launch's dynamic SMEM to 4 * batch_size bytes).

    Produces the same [num_sms+1, 2] int32 output as _compute_schedule_metadata()
    but entirely on-device with no D2H copy of context_lens.
    """

    def __init__(self, split_kv: int, num_sms: int):
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
            grid=(1, 1, 1), block=(32, 1, 1), stream=stream, smem=(4 * batch_size)
        )

    @cute.kernel
    def kernel(
        self,
        context_lens: cute.Tensor,
        schedule_meta: cute.Tensor,
        batch_size: cutlass.Int32,
    ):
        SPLIT_KV = cutlass.const_expr(self.split_kv)
        kNumSMs = cutlass.const_expr(self.num_sms)
        num_chunks = (batch_size + 31) // 32
        # bit_length(batch_size): halvings for the phase-3 search to converge.
        num_search_steps = 32 - cutlass.Int32(cute.arch.clz(batch_size))
        # Cover sm_idx ∈ [0, kNumSMs] (inclusive) in 32-lane strips.
        kMaxSmChunks = cutlass.const_expr((kNumSMs + 32) // 32)

        lane_idx = cute.arch.lane_idx()

        smem_ptr = cute.arch.get_dyn_smem(
            element_type=cutlass.Int32,
            alignment=128,
        )
        layout = cute.make_ordered_layout((batch_size,), order=(0,))
        prefix_sum = cute.make_tensor(smem_ptr, layout)

        # Phases 1+2, one 32-row chunk per iteration: each lane computes
        # ceil_div(ctx, SPLIT_KV) for its row, then a warp inclusive scan plus
        # the running carry gives prefix_sum. Out-of-range lanes contribute 0
        # and still join the shuffles, but skip the store: SMEM holds exactly
        # batch_size entries.
        sum_carry = cutlass.Int32(0)
        for k in cutlass.range(num_chunks):
            q_idx = cutlass.Int32(k * 32) + lane_idx
            ctx_len = cutlass.Int32(0)
            if q_idx < batch_size:
                ctx_len = context_lens[q_idx]
            x = (ctx_len + (SPLIT_KV - 1)) // SPLIT_KV
            for i in cutlass.range_constexpr(5):  # log2(32) = 5
                offset = 1 << i
                y = cute.arch.shuffle_sync_up(
                    x, offset, mask=0xFFFFFFFF, mask_and_clamp=0
                )
                if lane_idx >= offset:
                    x = x + y
            x = x + sum_carry
            if q_idx < batch_size:
                prefix_sum[k * 32 + lane_idx] = x
            # Broadcast lane-31's inclusive sum to all lanes for next chunk carry.
            sum_carry = cute.arch.shuffle_sync(x, 31)

        # Phase 3 reads prefix_sum entries other lanes wrote. shuffle_sync
        # converges the warp but does not order SMEM writes against other
        # lanes' reads; without this, racecheck reports intra-warp hazards.
        cute.arch.sync_warp()

        # Phase 3: distribute total segments evenly across kNumSMs CTAs.
        # Each lane processes a strip of CTA indices.
        total = sum_carry
        q_div = total // kNumSMs
        r_mod = total % kNumSMs

        # Lanes past kNumSMs search too (their loads stay in bounds); only the
        # schedule_meta store at the end is guarded.
        seg_starts = [cutlass.Int32(0)] * kMaxSmChunks
        lo = [cutlass.Int32(0)] * kMaxSmChunks
        cnt = [batch_size] * kMaxSmChunks

        for s in cutlass.range_constexpr(kMaxSmChunks):
            sm_idx_local = cutlass.Int32(s * 32) + lane_idx
            # seg = sm * q_div + min(sm, r_mod)
            seg = sm_idx_local * q_div
            if sm_idx_local <= r_mod:
                seg = seg + sm_idx_local
            else:
                seg = seg + r_mod
            seg_starts[s] = seg

        # q_idx_out = count{ j < batch_size : prefix_sum[j] <= seg_start }.
        # prefix_sum is non-decreasing, so this is an upper_bound: binary
        # search it, bit_length(batch_size) halvings. Ties matter -- a
        # zero-length row contributes no segments, so prefix_sum repeats and
        # every duplicate must be counted; hence <= and not <.
        #
        # Steps are the outer loop so all strips advance together, and each
        # step loads unconditionally (mid clamped into [0, batch_size)) with
        # cnt > 0 only selecting the result. A single warp has no other warps
        # to hide SMEM latency; with the load inside the cnt guard, every
        # strip's load sat behind its own branch and the strips ran serially.
        for _ in cutlass.range(num_search_steps):
            for s in cutlass.range_constexpr(kMaxSmChunks):
                half = cnt[s] // 2
                mid = lo[s] + half
                mid_clamped = mid

                if mid > batch_size - 1:
                    mid_clamped = batch_size - 1
                mid_val = prefix_sum[mid_clamped]

                go_right = cutlass.Boolean(False)
                if cnt[s] > 0:
                    if mid_val <= seg_starts[s]:
                        go_right = cutlass.Boolean(True)
                if go_right:
                    lo[s] = mid + 1
                    cnt[s] = cnt[s] - half - 1
                else:
                    cnt[s] = half

        for s in cutlass.range_constexpr(kMaxSmChunks):
            sm_idx_local = cutlass.Int32(s * 32) + lane_idx
            q_idx_out = lo[s]
            kv_split_idx = seg_starts[s]
            if q_idx_out > 0:
                kv_split_idx = seg_starts[s] - prefix_sum[q_idx_out - 1]
            if sm_idx_local <= kNumSMs:
                schedule_meta[sm_idx_local, 0] = q_idx_out
                schedule_meta[sm_idx_local, 1] = kv_split_idx


@functools.cache
def _cached_schedule_source_files() -> Tuple[str, ...]:
    """Files whose contents change this kernel's generated code.

    Only this module: the schedule kernel's codegen is fully determined by
    ``PagedMQALogitsScheduleKernel`` plus the three values in the cache key.
    Deliberately excludes attn_scores.py -- unlike the FP8/FP4 wrappers, no
    shape preparation for this kernel happens there, so including it would
    invalidate the cached kernel on any unrelated edit to that module.
    """
    return (__file__,)


@functools.cache
def _compile_schedule_kernel(split_kv: int, num_sms: int, arch: str):
    """Compile GPU schedule kernel; cached by (split_kv, num_sms, arch).

    ``arch`` pins codegen to the target device rather than to CUTLASS's
    default probe, which queries ordinal 0 unconditionally.

    batch_size is a runtime argument, so this compiles once per device
    configuration and serves every batch size. Backed by the same persistent
    on-disk cache as the FP8/FP4 kernels, so later processes skip the compile.
    """
    from ...jit.cute_dsl_core import build_and_load_cute_dsl_kernel

    sym_B = cute.sym_int()
    # context_lens is read via its layout (context_lens[q_idx]), so any stride
    # is fine: declare it symbolic so a strided view (seq_lens[::next_n]) is
    # accepted zero-copy, matching the fp8/fp4 kernels' declarations.
    cl_fake = cute.runtime.make_fake_tensor(
        cutlass.Int32, (sym_B,), stride=(cute.sym_int(),), assumed_align=4
    )
    sm_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (num_sms + 1, 2), stride_order=(1, 0)
    )
    fake_stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

    kern = PagedMQALogitsScheduleKernel(split_kv, num_sms)

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
        f"sched_split{split_kv}_sms{num_sms}_{arch}",
        _compile_fn,
        extra_key_files=_cached_schedule_source_files(),
    )
