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

"""LL MoE finalize with a per-expert RMSNorm, all-reduced across ranks.

Two kernels share an LL contribution mailbox and stage state:

1. ``_NormFinalizePublishDeviceKernel`` (one CTA per token): for each of the
   token's top-k routes held by this rank, RMS-normalize the expert's GEMM2 row
   over the full hidden width, scale by the route weight and accumulate in
   FP32; multicast the BF16 partial sum to slot ``[stage, rank]``.
2. The existing ``_LamportResidualRMSNormDeviceKernel`` with the post-reduce
   RMSNorm disabled sums the ``tp`` partials and stores the BF16 result.

Per route ``r`` with row ``x_r`` and weight ``w_r``::

    s_r = x_r * (gamma + gain_center)                  # weight_before_norm
    partial = sum_r w_r * s_r * rsqrt(mean(s_r**2) + eps)
    out = bf16(sum_over_ranks(bf16(partial)))

With ``weight_before_norm=False`` the weight multiplies after the norm.
"""

from __future__ import annotations

from typing import Any

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
import torch.distributed as dist
from cutlass import BFloat16, Float32, Int32, Int64
from cutlass.cute.runtime import make_fake_compact_tensor

from ..cute_dsl_primitives import (
    VEC_BF16,
    WARP_SIZE,
    bf16x8_to_packed_u32x4,
    load_bf16x8_as_f32,
    load_volatile_u32,
    sanitize_negative_zero_u32x4,
    stmc_bf16x8,
    store_global_u32,
)
from ..runtime import (
    current_cu_stream,
    make_fake_dynamic_compact_tensor,
    to_cute,
    to_cute_dynamic,
)
from .device_kernels import (
    ACTIVE_STAGE,
    NEXT_STAGE,
    _LamportResidualRMSNormDeviceKernel,
)
from .protocol import create_ll_state


class _NormFinalizePublishDeviceKernel:
    def __init__(
        self,
        *,
        hidden: int,
        row_stride: int,
        topk: int,
        tp: int,
        rank: int,
        capacity_m: int,
        threads: int,
        weight_before_norm: bool,
        eps: float,
        gain_center: float,
        enable_pdl: bool,
    ) -> None:
        if threads % WARP_SIZE or hidden % (threads * VEC_BF16):
            raise ValueError("hidden must split evenly into 8-element fragments")
        self.hidden = hidden
        self.row_stride = row_stride
        self.topk = topk
        self.tp = tp
        self.rank = rank
        self.capacity_m = capacity_m
        self.threads = threads
        self.warps = threads // WARP_SIZE
        self.trips = hidden // (threads * VEC_BF16)
        self.weight_before_norm = weight_before_norm
        self.eps = eps
        self.gain_center = gain_center
        self.enable_pdl = enable_pdl

    @cute.jit
    def __call__(
        self,
        routed_rows: cute.Tensor,
        route_weights: cute.Tensor,
        permuted_idx: cute.Tensor,
        norm_weight: cute.Tensor,
        stage_state: cute.Tensor,
        contribution_mailbox_multicast_address: Int64,
        m: Int32,
        num_rows: Int32,
        stream: cuda.CUstream,
    ) -> None:
        self.kernel(
            routed_rows,
            route_weights,
            permuted_idx,
            norm_weight,
            stage_state,
            contribution_mailbox_multicast_address,
            num_rows,
        ).launch(
            grid=(m, 1, 1),
            block=(self.threads, 1, 1),
            smem=self.topk * self.warps * 4,
            stream=stream,
            use_pdl=self.enable_pdl,
        )

    @cute.kernel
    def kernel(
        self,
        routed_rows: cute.Tensor,
        route_weights: cute.Tensor,
        permuted_idx: cute.Tensor,
        norm_weight: cute.Tensor,
        stage_state: cute.Tensor,
        contribution_mailbox_multicast_address: Int64,
        num_rows: Int32,
    ) -> None:
        token, _, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane = cute.arch.lane_idx()

        frag_layout = cute.make_layout((self.trips, VEC_BF16), stride=(VEC_BF16, 1))
        gamma = cute.make_rmem_tensor(frag_layout, Float32)
        for trip in cutlass.range_constexpr(self.trips):
            column = Int64(tidx + trip * self.threads) * VEC_BF16
            gamma[trip, None].store(
                load_bf16x8_as_f32(norm_weight, column) + Float32(self.gain_center)
            )

        if cutlass.const_expr(self.enable_pdl):
            cute.arch.griddepcontrol_wait()

        smem = cutlass.utils.SmemAllocator()
        # One slot set per route avoids a second barrier per route.
        warp_sums = smem.allocate_array(Float32, self.topk * self.warps)
        scaled = cute.make_rmem_tensor(frag_layout, Float32)
        acc = cute.make_rmem_tensor(frag_layout, Float32)
        acc.fill(Float32(0.0))
        for k in cutlass.range_constexpr(self.topk):
            route = Int64(token) * self.topk + k
            row = Int32(permuted_idx[route])
            is_local = row >= Int32(0) and row < num_rows
            # All threads in this token's CTA have the same route index.
            if is_local:
                for trip in cutlass.range_constexpr(self.trips):
                    column = Int64(tidx + trip * self.threads) * VEC_BF16
                    values = load_bf16x8_as_f32(
                        routed_rows, Int64(row) * self.row_stride + column
                    )
                    if cutlass.const_expr(self.weight_before_norm):
                        values = values * gamma[trip, None].load()
                    scaled[trip, None].store(values)
                sum_sq = (scaled.load() * scaled.load()).reduce(
                    cute.ReductionOp.ADD, init_val=Float32(0.0), reduction_profile=0
                )
                for offset in (16, 8, 4, 2, 1):
                    sum_sq = sum_sq + cute.arch.shuffle_sync_bfly(sum_sq, offset=offset)
                if lane == 0:
                    cute.arch.store(
                        (warp_sums + (k * self.warps + warp)).llvm_ptr, sum_sq
                    )
                cute.arch.barrier()
                total = Float32(0.0)
                for w in cutlass.range_constexpr(self.warps):
                    total = total + cute.arch.load(
                        (warp_sums + (k * self.warps + w)).llvm_ptr, Float32
                    )
                inv_rms = cute.math.rsqrt(
                    total * Float32(1.0 / self.hidden) + Float32(self.eps),
                    fastmath=True,
                )
                weight = Float32(route_weights[route])
                for trip in cutlass.range_constexpr(self.trips):
                    normalized = scaled[trip, None].load() * inv_rms
                    if cutlass.const_expr(not self.weight_before_norm):
                        normalized = normalized * gamma[trip, None].load()
                    acc[trip, None].store(acc[trip, None].load() + normalized * weight)

        stage = load_volatile_u32(stage_state.iterator + NEXT_STAGE)
        for trip in cutlass.range_constexpr(self.trips):
            column = Int64(tidx + trip * self.threads) * VEC_BF16
            mailbox_element = (
                (Int64(stage) * self.tp + self.rank) * self.capacity_m + Int64(token)
            ) * self.hidden + column
            stmc_bf16x8(
                contribution_mailbox_multicast_address + mailbox_element * 2,
                sanitize_negative_zero_u32x4(
                    bf16x8_to_packed_u32x4(acc[trip, None].load().to(BFloat16))
                ),
            )
        if token == 0 and tidx == 0:
            store_global_u32(stage_state.iterator + ACTIVE_STAGE, stage)
        cute.arch.barrier()
        if cutlass.const_expr(self.enable_pdl):
            cute.arch.griddepcontrol_launch_dependents()


class LLMoEFinalizeNormAllReduce:
    """Per-expert RMSNorm finalize + all-reduce for one ``[M, hidden]`` boundary.

    Owns its own mailbox and stage state. Construction is collective over
    ``group``; calls on all ranks must use the same ``m`` and must not overlap.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        topk: int,
        tp_size: int,
        rank: int,
        capacity_m: int,
        group: dist.ProcessGroup,
        threads: int = 128,
        reduce_cluster_size: int = 4,
        enable_pdl: bool = True,
    ) -> None:
        self.hidden_size = hidden_size
        self.topk = topk
        self.tp_size = tp_size
        self.rank = rank
        self.capacity_m = capacity_m
        self.threads = threads
        self.enable_pdl = enable_pdl
        self.state = create_ll_state(group, tp_size, rank, capacity_m, hidden_size)
        self._mailbox = to_cute(self.state.contribution_mailbox.tensor.flatten(), 16)
        self._mailbox_multicast = Int64(
            self.state.contribution_mailbox.multicast_address
        )
        self._stage_state = to_cute(self.state.stage_state, 4)
        reduce = _LamportResidualRMSNormDeviceKernel(
            hidden=hidden_size,
            tp=tp_size,
            capacity_m=capacity_m,
            cluster_size=reduce_cluster_size,
            rank_lanes=1,
            threads=threads,
            rms_epsilon=0.0,
            weight_bias=0.0,
            add_residual=False,
            write_residual_output=True,
            apply_rms_norm=False,
            enable_pdl=enable_pdl,
        )
        rows = make_fake_dynamic_compact_tensor(
            BFloat16, alignment=16, divisibility=hidden_size
        )
        self._reduce = cute.compile(
            reduce,
            make_fake_compact_tensor(
                BFloat16,
                (self.state.contribution_mailbox.tensor.numel(),),
                assumed_align=16,
            ),
            rows,
            make_fake_compact_tensor(BFloat16, (hidden_size,), assumed_align=16),
            rows,
            rows,
            None,
            None,
            make_fake_compact_tensor(Int32, (2,), assumed_align=4),
            Int32(capacity_m),
            current_cu_stream(),
        )
        self._publishers: dict[tuple, Any] = {}
        torch.cuda.synchronize()
        dist.barrier(group)

    def publisher(
        self,
        *,
        row_stride: int,
        weight_dtype: torch.dtype,
        weight_before_norm: bool,
        eps: float,
        gain_center: float,
    ) -> Any:
        """The producer for one configuration; compile it before graph capture."""
        key = (row_stride, weight_dtype, weight_before_norm, eps, gain_center)
        publisher = self._publishers.get(key)
        if publisher is not None:
            return publisher
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "Compile the MoE finalize all-reduce before CUDA graph capture"
            )
        device_kernel = _NormFinalizePublishDeviceKernel(
            hidden=self.hidden_size,
            row_stride=row_stride,
            topk=self.topk,
            tp=self.tp_size,
            rank=self.rank,
            capacity_m=self.capacity_m,
            threads=self.threads,
            weight_before_norm=weight_before_norm,
            eps=eps,
            gain_center=gain_center,
            enable_pdl=self.enable_pdl,
        )
        weight_type = Float32 if weight_dtype == torch.float32 else BFloat16
        publisher = cute.compile(
            device_kernel,
            make_fake_dynamic_compact_tensor(
                BFloat16, alignment=16, divisibility=row_stride
            ),
            make_fake_dynamic_compact_tensor(
                weight_type, alignment=weight_type.width // 8, divisibility=self.topk
            ),
            make_fake_dynamic_compact_tensor(
                Int32, alignment=4, divisibility=self.topk
            ),
            make_fake_compact_tensor(BFloat16, (self.hidden_size,), assumed_align=16),
            make_fake_compact_tensor(Int32, (2,), assumed_align=4),
            Int64(0),
            Int32(self.capacity_m),
            Int32(1),
            current_cu_stream(),
        )
        self._publishers[key] = publisher
        return publisher

    def __call__(
        self,
        routed_rows: torch.Tensor,
        route_weights: torch.Tensor,
        permuted_idx: torch.Tensor,
        norm_weight: torch.Tensor,
        output: torch.Tensor,
        *,
        weight_before_norm: bool,
        eps: float,
        gain_center: float,
    ) -> None:
        publisher = self.publisher(
            row_stride=routed_rows.shape[1],
            weight_dtype=route_weights.dtype,
            weight_before_norm=weight_before_norm,
            eps=eps,
            gain_center=gain_center,
        )
        m = Int32(output.shape[0])
        stream = current_cu_stream()
        weight_align = route_weights.element_size()
        gamma = to_cute(norm_weight, 16)
        publisher(
            to_cute_dynamic(
                routed_rows.flatten(), 16, divisibility=routed_rows.shape[1]
            ),
            to_cute_dynamic(
                route_weights.flatten(), weight_align, divisibility=self.topk
            ),
            to_cute_dynamic(permuted_idx.flatten(), 4, divisibility=self.topk),
            gamma,
            self._stage_state,
            self._mailbox_multicast,
            m,
            Int32(routed_rows.shape[0]),
            stream,
        )
        out = to_cute_dynamic(output.flatten(), 16, divisibility=self.hidden_size)
        self._reduce(
            # Unused residual source, gamma and norm output with the norm off.
            self._mailbox,
            out,
            gamma,
            out,
            out,
            None,
            None,
            self._stage_state,
            m,
            stream,
        )
