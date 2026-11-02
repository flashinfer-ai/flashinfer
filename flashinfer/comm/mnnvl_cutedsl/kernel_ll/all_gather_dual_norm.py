# Copyright (c) 2026 by FlashInfer team.
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

"""LL column all-gather consumed by a gated residual and the next RMSNorm.

Two kernels share the LL contribution mailbox and stage state:

1. ``_SharedOnlyPublishDeviceKernel`` (unchanged, ``hidden=H/TP``) multicasts
   this rank's ``[M, H/TP]`` column shard to slot ``[stage, rank]`` of every
   rank's ``[3, TP, capacity_m, H/TP]`` mailbox.
2. ``_AllGatherDualNormDeviceKernel`` reads every global column from the one
   rank that owns it (no cross-rank sum), re-arms the slot it consumed, and
   computes over the full row::

       b = branch * (post_weight + post_gain_center)  # or branch
       r1 = rsqrt(mean(b**2) + post_eps)
       residual_out = alpha * residual + (beta * b) * r1  # FP32
       r2 = rsqrt(mean(residual_out**2) + next_eps)
       norm_out = bf16(residual_out * r2 * (next_weight + next_gain_center))

   Both row statistics are explicit FP32 reductions over all ``H`` columns;
   the only rounding is the final BF16 store.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
import torch.distributed as dist
from cutlass import BFloat16, Float32, Int32, Int64, Uint32
from cutlass.cute.runtime import make_fake_compact_tensor

from ..cute_dsl_primitives import (
    VEC_BF16,
    WARP_SIZE,
    bf16x8_to_packed_u32x4,
    load_bf16x8_as_f32,
    load_f32x8,
    packed_u32x4_to_bf16x8,
    store_global_u32x4,
    store_f32x8,
)
from ..runtime import (
    current_cu_stream,
    make_fake_dynamic_compact_tensor,
    to_cute,
    to_cute_dynamic,
)
from .column_gather import LLColumnGather, cluster_row_sum, poll_owned_fragment


@dataclass(frozen=True, slots=True)
class LLAllGatherDualNormTuning:
    publish_threads: int = 128
    publish_release_before_store: bool = False
    cluster_size: int = 8
    threads: int = 128
    enable_pdl: bool = True


@dataclass(frozen=True, slots=True)
class _NormConstants:
    post_eps: float
    next_eps: float
    post_gain_center: float
    next_gain_center: float
    has_post_weight: bool


class _AllGatherDualNormDeviceKernel:
    def __init__(
        self,
        *,
        hidden: int,
        tp: int,
        capacity_m: int,
        cluster_size: int,
        threads: int,
        constants: _NormConstants,
        enable_pdl: bool,
    ) -> None:
        if hidden % (tp * VEC_BF16):
            raise ValueError("hidden must split into 8-element fragments per rank")
        if cluster_size * threads * VEC_BF16 != hidden:
            raise ValueError("cluster_size * threads * 8 must equal hidden")
        if threads % WARP_SIZE or threads < cluster_size:
            raise ValueError("threads must be a warp multiple of at least cluster_size")
        self.hidden = hidden
        self.tp = tp
        self.hidden_local = hidden // tp
        self.capacity_m = capacity_m
        self.cluster_size = cluster_size
        self.threads = threads
        self.warps = threads // WARP_SIZE
        self.constants = constants
        self.enable_pdl = enable_pdl

    def smem_size_in_bytes(self) -> int:
        # One slot per (CTA, warp) for each of the two row statistics.
        return 2 * self.cluster_size * self.warps * 4

    @cute.jit
    def __call__(
        self,
        contribution_mailbox: cute.Tensor,
        residual: cute.Tensor,
        alpha: cute.Tensor,
        beta: cute.Tensor,
        post_weight: cute.Tensor,
        next_weight: cute.Tensor,
        residual_output: cute.Tensor,
        norm_output: cute.Tensor,
        stage_state: cute.Tensor,
        m: Int32,
        stream: cuda.CUstream,
    ) -> None:
        self.kernel(
            contribution_mailbox,
            residual,
            alpha,
            beta,
            post_weight,
            next_weight,
            residual_output,
            norm_output,
            stage_state,
        ).launch(
            grid=(m, self.cluster_size, 1),
            block=(self.threads, 1, 1),
            cluster=(1, self.cluster_size, 1),
            smem=self.smem_size_in_bytes(),
            stream=stream,
            use_pdl=self.enable_pdl,
        )

    @cute.kernel
    def kernel(
        self,
        contribution_mailbox: cute.Tensor,
        residual: cute.Tensor,
        alpha: cute.Tensor,
        beta: cute.Tensor,
        post_weight: cute.Tensor,
        next_weight: cute.Tensor,
        residual_output: cute.Tensor,
        norm_output: cute.Tensor,
        stage_state: cute.Tensor,
    ) -> None:
        constants = self.constants
        tidx, _, _ = cute.arch.thread_idx()
        token, _, _ = cute.arch.block_idx()
        cluster_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane = cute.arch.lane_idx()
        fragment = cluster_rank * self.threads + tidx
        column = Int64(fragment) * VEC_BF16
        row_element = Int64(token) * self.hidden + column
        # Every peer must have started before cluster_row_sum stores into its
        # shared memory; the matching wait sits just before that first store.
        cute.arch.cluster_arrive_relaxed()

        # Parameters and the residual stream do not depend on the publisher.
        alpha_values = load_f32x8(alpha, column)
        beta_values = load_f32x8(beta, column)
        next_gamma = load_bf16x8_as_f32(next_weight, column) + Float32(
            constants.next_gain_center
        )
        if cutlass.const_expr(constants.has_post_weight):
            post_gamma = load_bf16x8_as_f32(post_weight, column) + Float32(
                constants.post_gain_center
            )
        stream_values = load_f32x8(residual, row_element)

        if cutlass.const_expr(self.enable_pdl):
            cute.arch.griddepcontrol_wait()

        branch_packed = cute.make_rmem_tensor(cute.make_layout((4,)), Uint32)
        poll_owned_fragment(
            contribution_mailbox,
            stage_state,
            branch_packed,
            token,
            fragment,
            self.tp,
            self.capacity_m,
            self.hidden_local,
            self.enable_pdl,
        )

        smem = cutlass.utils.SmemAllocator()
        post_slots = smem.allocate_array(Float32, self.cluster_size * self.warps)
        next_slots = smem.allocate_array(Float32, self.cluster_size * self.warps)

        branch = packed_u32x4_to_bf16x8(branch_packed.load()).to(Float32)
        if cutlass.const_expr(constants.has_post_weight):
            branch = branch * post_gamma
        cute.arch.cluster_wait()
        post_sum = cluster_row_sum(
            (branch * branch).reduce(
                cute.ReductionOp.ADD, init_val=Float32(0.0), reduction_profile=0
            ),
            post_slots,
            self.warps,
            self.cluster_size,
            cluster_rank,
            warp,
            lane,
        )
        post_inv_rms = cute.math.rsqrt(
            post_sum * Float32(1.0 / self.hidden) + Float32(constants.post_eps),
            fastmath=True,
        )
        # Same association as gated_residual_next_rmsnorm: no rounded norm(b).
        updated = alpha_values * stream_values + (beta_values * branch) * post_inv_rms
        store_f32x8(residual_output, row_element, updated)

        next_sum = cluster_row_sum(
            (updated * updated).reduce(
                cute.ReductionOp.ADD, init_val=Float32(0.0), reduction_profile=0
            ),
            next_slots,
            self.warps,
            self.cluster_size,
            cluster_rank,
            warp,
            lane,
        )
        next_inv_rms = cute.math.rsqrt(
            next_sum * Float32(1.0 / self.hidden) + Float32(constants.next_eps),
            fastmath=True,
        )
        result = (updated * next_inv_rms * next_gamma).to(BFloat16)
        store_global_u32x4(
            Int64((norm_output.iterator + row_element).toint()),
            bf16x8_to_packed_u32x4(result),
        )


class LLAllGatherDualNorm(LLColumnGather):
    """Column all-gather of ``[M, H/TP]`` shards fused into the dual norm."""

    def __init__(
        self,
        *,
        hidden_size: int,
        tp_size: int,
        rank: int,
        capacity_m: int,
        group: dist.ProcessGroup,
        tuning: LLAllGatherDualNormTuning,
    ) -> None:
        if tuning.cluster_size * tuning.threads * VEC_BF16 != hidden_size:
            raise ValueError("cluster_size * threads * 8 must equal hidden_size")
        self.tuning = tuning
        super().__init__(
            hidden_size=hidden_size,
            tp_size=tp_size,
            rank=rank,
            capacity_m=capacity_m,
            group=group,
            publish_threads=tuning.publish_threads,
            publish_release_before_store=tuning.publish_release_before_store,
            enable_pdl=tuning.enable_pdl,
        )

    def _compile(self, constants: _NormConstants) -> Any:
        device_kernel = _AllGatherDualNormDeviceKernel(
            hidden=self.hidden_size,
            tp=self.tp_size,
            capacity_m=self.capacity_m,
            cluster_size=self.tuning.cluster_size,
            threads=self.tuning.threads,
            constants=constants,
            enable_pdl=self.tuning.enable_pdl,
        )
        weight = make_fake_compact_tensor(
            BFloat16, (self.hidden_size,), assumed_align=16
        )
        gate = make_fake_compact_tensor(Float32, (self.hidden_size,), assumed_align=32)
        rows_f32 = make_fake_dynamic_compact_tensor(
            Float32, alignment=32, divisibility=self.hidden_size
        )
        return cute.compile(
            device_kernel,
            make_fake_compact_tensor(
                BFloat16,
                (self.state.contribution_mailbox.tensor.numel(),),
                assumed_align=16,
            ),
            rows_f32,
            gate,
            gate,
            weight,
            weight,
            rows_f32,
            make_fake_dynamic_compact_tensor(
                BFloat16, alignment=16, divisibility=self.hidden_size
            ),
            make_fake_compact_tensor(Int32, (2,), assumed_align=4),
            Int32(self.capacity_m),
            current_cu_stream(),
        )

    def __call__(
        self,
        branch_shard: torch.Tensor,
        residual: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        next_weight: torch.Tensor,
        residual_output: torch.Tensor,
        norm_output: torch.Tensor,
        *,
        constants: _NormConstants,
        post_weight: torch.Tensor | None,
    ) -> None:
        collective = self.compiled(constants)
        m = Int32(branch_shard.shape[0])
        stream = current_cu_stream()
        self._publish(branch_shard, m, stream)
        collective(
            self._mailbox,
            to_cute_dynamic(residual.flatten(), 32, divisibility=self.hidden_size),
            to_cute(alpha, 32),
            to_cute(beta, 32),
            to_cute(post_weight if post_weight is not None else next_weight, 16),
            to_cute(next_weight, 16),
            to_cute_dynamic(
                residual_output.flatten(), 32, divisibility=self.hidden_size
            ),
            to_cute_dynamic(norm_output.flatten(), 16, divisibility=self.hidden_size),
            self._stage_state,
            m,
            stream,
        )
