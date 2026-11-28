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

"""SMEM P staging for the 2CTA MLA PV MMA."""

import cutlass.cute as cute
import cutlass
from cutlass import Int32, Int64
from cutlass.experimental import primitives as prims
from cutlass.experimental.task_scheduling.resources import (
    StageInfo,
    TaskLocalVariable,
    consumer_work,
    producer_work,
)
from dataclasses import dataclass, field
from ...device.math import (
    p_desc_layout,
    p_desc_leading_byte_offset,
    p_desc_stride_byte_offset,
    qkv_dtype,
)
from ..config import MlaDecodeConfig
from .common import HighThroughputMlaResource
from typing import Any, ClassVar


# =====================================================================
# SmemPResource — P in SMEM, UmmaConsumerAsync pipeline
# =====================================================================


@dataclass(kw_only=True)
class SmemPResource(HighThroughputMlaResource):
    """SMEM P buffer.  Producer: SoftmaxTask.  Consumer: MmaTask (PV MMA).

    Pipeline: UmmaConsumerAsync, 2 stages.
    """

    # Static configuration
    cfg: cutlass.Constexpr = field(default_factory=MlaDecodeConfig)
    desc_p_base: cutlass.Constexpr[TaskLocalVariable] = (
        TaskLocalVariable.uninitialized()
    )
    task_local_specs: ClassVar[tuple[tuple, ...]] = (
        ("desc_p_base", Int64, Int64(0), "SMEM descriptor for staged P."),
    )

    # Per-task state
    cta_rank: Any = field(init=False, default=None)
    smem_p: Any = None

    @consumer_work(returns=desc_p_base)
    @cute.jit
    def p_desc(self, stage_info: StageInfo):
        """MMA warp builds P SMEM descriptor for PV MMA."""
        cfg = self.cfg
        sp_stage_stride_elems = cutlass.const_expr(
            cfg.mma_pv_tiler[0]
            // cfg.num_mma_ctas
            * cfg.mma_pv_tiler[2]
            * cfg.iterations_pv_k
        )
        sp_ptr = self.smem_p.data_ptr(stage_info.stage_idx * sp_stage_stride_elems)
        desc_p = Int64(
            prims.Tcgen05SmemDesc.build(
                start_address=sp_ptr.toint(Int32),
                leading_byte_offset=p_desc_leading_byte_offset(cfg),
                stride_byte_offset=p_desc_stride_byte_offset(cfg),
                layout=p_desc_layout(cfg),
            )
        )
        return desc_p

    @producer_work
    @cute.jit
    def store_p(self, stage_info: StageInfo, *, qk_acc_regs) -> None:
        """Store P from the even softmax group."""
        self.store_p_impl(stage_info, qk_acc_regs=qk_acc_regs)

    @cute.jit
    def store_p_impl(self, stage_info: StageInfo, *, qk_acc_regs) -> None:
        """Store softmax P tile to SMEM for PV MMA.

        Converts P values to FP16 and stores to SMEM with correct swizzle layout.
        """
        cfg = self.cfg
        stage_idx = stage_info.stage_idx
        num_mma_ctas = cfg.num_mma_ctas

        tidx = cute.arch.thread_idx()[0]
        num_compute_threads = cfg.num_compute_warps * cfg.threads_per_warp
        local_tidx = tidx % num_compute_threads
        lane_idx = local_tidx & 31
        warp_idx = local_tidx >> 5

        # Convert F32 -> QKV dtype into a vectorized local buffer.
        qkv_element_dtype = qkv_dtype(cfg)
        s_regs = cutlass.Array(qkv_element_dtype, 64, space=cutlass.AddressSpace.rmem)
        s_regs.store(qk_acc_regs.load(0, 64).to(qkv_element_dtype), 0)

        # Compute SMEM P stage offset
        sp_stage_stride_elems = cutlass.const_expr(
            cfg.mma_pv_tiler[0]
            // num_mma_ctas
            * cfg.mma_pv_tiler[2]
            * cfg.iterations_pv_k
        )
        smem_p_base_bytes = (
            self.smem_p.data_ptr().toint(cutlass.Int32)
            + stage_idx * sp_stage_stride_elems * cfg.qkv_dtype_bytes
        )

        if cutlass.const_expr(cfg.is_fp8_qkv()):
            # FP8 P layout:
            # S<2,4,3> o ((64,32),1,2,(2,2)):((64,1),0,32,(4096,8192)).
            # Each universal SMEM copy is 128 bits, i.e. 16 E4M3 elements.
            # Lane pairs write opposite halves of the 128B swizzled row.  The
            # signed strides below walk the four 16B blocks inside that row.
            m = ((lane_idx >> 1) & 3) * 128
            base = cutlass.Int32(0)
            base = base + (lane_idx & 1) * 64
            base = base + (lane_idx >> 3) * 512
            base = base + (warp_idx & 1) * 2048
            base = base + (warp_idx >> 1) * 4096
            swizzle_xor = m ^ ((m & 384) >> 3)
            off_base = base + swizzle_xor

            stride_a = cutlass.Int32(16)
            if (m & 128) != 0:
                stride_a = cutlass.Int32(-16)
            stride_b = cutlass.Int32(32)
            if (m & 256) != 0:
                stride_b = cutlass.Int32(-32)

            dst_blk_offs = (
                cutlass.Int32(0),
                stride_a,
                stride_b,
                stride_a + stride_b,
            )
            for blk in cutlass.range_constexpr(4):
                src_blk_base = blk * 16
                dst_blk_addr = smem_p_base_bytes + off_base + dst_blk_offs[blk]
                vec = s_regs.load(src_blk_base, 16)
                smem_ptr = cutlass.inttoptr(dst_blk_addr, 3, qkv_element_dtype)
                smem_ptr.store(vec, alignment=16)
        else:
            # BF16 P layout:
            # S<2,4,3> o ((64,16),1,2,(4,2)):((32,1),0,16,(2048,8192)).
            # Each universal SMEM copy is 128 bits, i.e. 8 BF16 elements.
            # The two K slices live 2048 elements apart in the staged P tile.
            # Per-block strides mirror the FP8 swizzle at half the byte width.
            m = ((lane_idx >> 1) & 3) * 64
            base = cutlass.Int32(0)
            base = base + (lane_idx & 1) * 32
            base = base + (lane_idx >> 3) * 256
            base = base + (warp_idx & 1) * 1024
            base = base + (warp_idx >> 1) * 4096
            swizzle_xor = m ^ ((m & 192) >> 3)
            off_base = base + swizzle_xor

            stride_a = cutlass.Int32(8)
            if (m & 64) != 0:
                stride_a = cutlass.Int32(-8)
            stride_b = cutlass.Int32(16)
            if (m & 128) != 0:
                stride_b = cutlass.Int32(-16)

            dst_blk_offs = (
                cutlass.Int32(0),
                stride_a,
                stride_b,
                stride_a + stride_b,
            )
            for k in cutlass.range_constexpr(2):
                k_base = off_base + k * 2048
                src_k_base = k * 32
                for blk in cutlass.range_constexpr(4):
                    src_blk_base = src_k_base + blk * 8
                    dst_blk_addr = (
                        smem_p_base_bytes
                        + (k_base + dst_blk_offs[blk]) * cfg.qkv_dtype_bytes
                    )
                    vec = s_regs.load(src_blk_base, 8)
                    smem_ptr = cutlass.inttoptr(dst_blk_addr, 3, qkv_element_dtype)
                    smem_ptr.store(vec, alignment=16)

        # Fence between SMEM store and MMA read
        prims.fence_proxy(
            kind=prims.Proxy.ASYNC_SHARED,
            space=prims.SharedSpace.shared_cta,
        )

    @producer_work
    @cute.jit
    def store_p_odd(self, stage_info: StageInfo, *, qk_acc_regs_odd) -> None:
        """Store P from the odd softmax group."""
        self.store_p_impl(stage_info, qk_acc_regs=qk_acc_regs_odd)
