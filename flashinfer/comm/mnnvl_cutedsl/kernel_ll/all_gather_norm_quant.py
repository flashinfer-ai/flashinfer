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

"""LL column all-gather consumed by an RMSNorm and NVFP4 quantization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
import torch.distributed as dist
from cutlass import BFloat16, Float32, Int32, Int64, Uint8, Uint32
from cutlass.cute.runtime import make_fake_compact_tensor

from ....cute_dsl.fp4_common import (
    cvt_e2m1x8_f32,
    cvt_f32_to_e4m3,
    fabs_f32,
    fmax_f32,
    nvfp4_compute_output_scale,
    rcp_approx_ftz,
)
from ..cute_dsl_primitives import (
    VEC_BF16,
    WARP_SIZE,
    load_bf16x8_as_f32,
    packed_u32x4_to_bf16x8,
    store_global_u32,
)
from ..runtime import (
    current_cu_stream,
    make_fake_dynamic_compact_tensor,
    to_cute,
    to_cute_dynamic,
)
from .column_gather import LLColumnGather, cluster_row_sum, poll_owned_fragment

_THREADS = 128
_NVFP4_BLOCK = 16


@dataclass(frozen=True, slots=True)
class _NormQuantConstants:
    eps: float
    gain_center: float


class _AllGatherNormQuantDeviceKernel:
    def __init__(
        self,
        *,
        hidden: int,
        tp: int,
        capacity_m: int,
        constants: _NormQuantConstants,
        enable_pdl: bool,
    ) -> None:
        self.hidden = hidden
        self.tp = tp
        self.hidden_local = hidden // tp
        self.capacity_m = capacity_m
        self.cluster_size = hidden // (_THREADS * VEC_BF16)
        self.warps = _THREADS // WARP_SIZE
        self.constants = constants
        self.enable_pdl = enable_pdl

    @cute.jit
    def __call__(
        self,
        contribution_mailbox: cute.Tensor,
        weight: cute.Tensor,
        global_scale: cute.Tensor,
        packed_output: cute.Tensor,
        scale_output: cute.Tensor,
        stage_state: cute.Tensor,
        m: Int32,
        stream: cuda.CUstream,
    ) -> None:
        self.kernel(
            contribution_mailbox,
            weight,
            global_scale,
            packed_output,
            scale_output,
            stage_state,
        ).launch(
            grid=(m, self.cluster_size, 1),
            block=(_THREADS, 1, 1),
            cluster=(1, self.cluster_size, 1),
            smem=self.cluster_size * self.warps * 4,
            stream=stream,
            use_pdl=self.enable_pdl,
        )

    @cute.kernel
    def kernel(
        self,
        contribution_mailbox: cute.Tensor,
        weight: cute.Tensor,
        global_scale: cute.Tensor,
        packed_output: cute.Tensor,
        scale_output: cute.Tensor,
        stage_state: cute.Tensor,
    ) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        token, _, _ = cute.arch.block_idx()
        cluster_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane = cute.arch.lane_idx()
        fragment = cluster_rank * _THREADS + tidx
        # Every peer must have started before cluster_row_sum stores into its
        # shared memory; the matching wait sits just before that first store.
        cute.arch.cluster_arrive_relaxed()

        gamma = load_bf16x8_as_f32(weight, Int64(fragment) * VEC_BF16) + Float32(
            self.constants.gain_center
        )
        scale = global_scale[0]

        if cutlass.const_expr(self.enable_pdl):
            cute.arch.griddepcontrol_wait()

        packed = cute.make_rmem_tensor(cute.make_layout((4,)), Uint32)
        poll_owned_fragment(
            contribution_mailbox,
            stage_state,
            packed,
            token,
            fragment,
            self.tp,
            self.capacity_m,
            self.hidden_local,
            self.enable_pdl,
        )

        smem = cutlass.utils.SmemAllocator()
        slots = smem.allocate_array(Float32, self.cluster_size * self.warps)
        x = packed_u32x4_to_bf16x8(packed.load()).to(Float32)
        cute.arch.cluster_wait()
        sum_sq = cluster_row_sum(
            (x * x).reduce(
                cute.ReductionOp.ADD, init_val=Float32(0.0), reduction_profile=0
            ),
            slots,
            self.warps,
            self.cluster_size,
            cluster_rank,
            warp,
            lane,
        )
        inv_rms = cute.math.rsqrt(
            sum_sq * Float32(1.0 / self.hidden) + Float32(self.constants.eps),
            fastmath=True,
        )
        # The unfused norm rounds to BF16 before the quantization reads it.
        values = cute.make_rmem_tensor(cute.make_layout((VEC_BF16,)), Float32)
        values.store(((x * inv_rms) * gamma).to(BFloat16).to(Float32))

        # A 16-column scale block spans this lane and its neighbour.
        amax = Float32(0.0)
        for i in cutlass.range_constexpr(VEC_BF16):
            amax = fmax_f32(amax, fabs_f32(values[i]))
        amax = fmax_f32(amax, cute.arch.shuffle_sync_bfly(amax, offset=1))
        sf = cvt_f32_to_e4m3(scale * (amax * rcp_approx_ftz(Float32(6.0))))
        output_scale = nvfp4_compute_output_scale(sf, scale)
        store_global_u32(
            packed_output.iterator
            + (Int64(token) * (self.hidden // 2) + Int64(fragment) * (VEC_BF16 // 2)),
            cvt_e2m1x8_f32(
                values[0] * output_scale,
                values[1] * output_scale,
                values[2] * output_scale,
                values[3] * output_scale,
                values[4] * output_scale,
                values[5] * output_scale,
                values[6] * output_scale,
                values[7] * output_scale,
            ),
        )
        if lane % 2 == 0:
            scale_output[token * (self.hidden // _NVFP4_BLOCK) + fragment // 2] = Uint8(
                sf & Uint32(0xFF)
            )


class LLAllGatherNormQuant(LLColumnGather):
    """Column all-gather of ``[M, H/TP]`` shards fused into RMSNorm + NVFP4."""

    def __init__(
        self,
        *,
        hidden_size: int,
        tp_size: int,
        rank: int,
        capacity_m: int,
        group: dist.ProcessGroup,
        enable_pdl: bool,
    ) -> None:
        cluster_size, remainder = divmod(hidden_size, _THREADS * VEC_BF16)
        if remainder or not 1 <= cluster_size <= 8:
            raise ValueError("hidden_size must be 1024 * n for n in [1, 8]")
        self.enable_pdl = enable_pdl
        super().__init__(
            hidden_size=hidden_size,
            tp_size=tp_size,
            rank=rank,
            capacity_m=capacity_m,
            group=group,
            publish_threads=_THREADS,
            publish_release_before_store=False,
            enable_pdl=enable_pdl,
        )

    def _compile(self, constants: _NormQuantConstants) -> Any:
        device_kernel = _AllGatherNormQuantDeviceKernel(
            hidden=self.hidden_size,
            tp=self.tp_size,
            capacity_m=self.capacity_m,
            constants=constants,
            enable_pdl=self.enable_pdl,
        )
        return cute.compile(
            device_kernel,
            make_fake_compact_tensor(
                BFloat16,
                (self.state.contribution_mailbox.tensor.numel(),),
                assumed_align=16,
            ),
            make_fake_compact_tensor(BFloat16, (self.hidden_size,), assumed_align=16),
            make_fake_compact_tensor(Float32, (1,), assumed_align=4),
            make_fake_dynamic_compact_tensor(
                Uint8, alignment=16, divisibility=self.hidden_size // 2
            ),
            make_fake_dynamic_compact_tensor(
                Uint8, alignment=4, divisibility=self.hidden_size // _NVFP4_BLOCK
            ),
            make_fake_compact_tensor(Int32, (2,), assumed_align=4),
            Int32(self.capacity_m),
            current_cu_stream(),
        )

    def __call__(
        self,
        shard: torch.Tensor,
        weight: torch.Tensor,
        global_scale: torch.Tensor,
        packed_output: torch.Tensor,
        scale_output: torch.Tensor,
        *,
        constants: _NormQuantConstants,
    ) -> None:
        collective = self.compiled(constants)
        m = Int32(shard.shape[0])
        stream = current_cu_stream()
        self._publish(shard, m, stream)
        collective(
            self._mailbox,
            to_cute(weight, 16),
            to_cute(global_scale, 4),
            to_cute_dynamic(
                packed_output.flatten(), 16, divisibility=self.hidden_size // 2
            ),
            to_cute_dynamic(
                scale_output.flatten(),
                4,
                divisibility=self.hidden_size // _NVFP4_BLOCK,
            ),
            self._stage_state,
            m,
            stream,
        )
