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
"""Generated CuTe DSL source for the Hopper VSA route.  Do not edit."""
from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass._mlir import ir as cutlass_ir
from cutlass._mlir.dialects import arith as cutlass_arith
from cutlass._mlir.dialects import llvm as cutlass_llvm
from cutlass._mlir.dialects import nvvm as cutlass_nvvm
from cutlass._mlir.dialects import cuda as cutlass_cuda
from cutlass.cute.runtime import make_fake_stream, make_fake_tensor
from cutlass.experimental import primitives as prims
from cutlass.experimental.primitives import nvvm_wrapper as prims_nvvm
from cutlass.utils import blackwell_helpers as cutlass_blackwell
from cutlass.utils import hopper_helpers as cutlass_hopper
from cutlass.utils.layout import LayoutEnum as CutlassLayout
from cutlass.utils import blockscaled_layout as cutlass_blockscaled
from cutlass.experimental.cuda.tensor_map import (
    TensorMap,
    TensorMapDataFormat,
    TensorMapFloatOOBFill,
    TensorMapL2Promotion,
    TensorMapSwizzle,
    create_tensor_map_tiled,
)
from cutlass._mlir.dialects.nvvm import CTAGroupKind as DialectCTAGroup
from cutlass._mlir.dialects.nvvm import TMALoadMode as DialectTMALoadMode

NUM_MAIN_STAGES = 1
SMEM_Q_SMEM_OFF = 1024
SMEM_Q_SMEM_STAGE_BYTES = 16384
SMEM_Q_SMEM_STRIDE = 16384
SMEM_K_SMEM_OFF = 17408
SMEM_K_SMEM_STAGE_BYTES = 16384
SMEM_K_SMEM_STRIDE = 16384
SMEM_VT_SMEM_OFF = 33792
SMEM_VT_SMEM_STAGE_BYTES = 16384
SMEM_VT_SMEM_STRIDE = 16384
SMEM_TOTAL = 50176
THREADS = 128
CAKE_TARGET_ARCH = 'sm_90a'
CAKE_SMEM_BYTES = 50176

@cute.kernel
def kernel_vsa_sm90_bf16_small_k1(Q: cutlass.GridConstant[TensorMap], K: cutlass.GridConstant[TensorMap], Vt: cutlass.GridConstant[TensorMap], O: cute.Pointer, plan: cute.Pointer, seqlen_q: cutlass.Int32, seqlen_k: cutlass.Int32, scale_log2: cutlass.Float32):
    tid = cutlass.Int32(cute.arch.thread_idx()[0])
    warp = cutlass.Int32(cute.arch.warp_idx())
    lane = cutlass.Int32(cute.arch.lane_idx())
    bid = cutlass.Int32(cute.arch.block_idx()[0])
    num_bids = cutlass.Int32(cute.arch.grid_dim()[0])
    blockIdx = cute.arch.block_idx()
    gridDim = cute.arch.grid_dim()
    smem_raw = cute.arch.get_dyn_smem(cutlass.Uint8, alignment=1024)
    smem = smem_raw.toint()
    _flat_layout = cute.make_layout((2147483647,), stride=(1,))
    _O = cute.make_tensor(O, _flat_layout)
    q_smem = cute.recast_ptr(smem_raw + 1024, swizzle_=cute.make_swizzle(4, 3, 3), dtype=cutlass.BFloat16)
    _q_smem = cute.make_tensor(q_smem, _flat_layout)
    q_smem_addr = smem + 1024
    k_smem = cute.recast_ptr(smem_raw + 17408, swizzle_=cute.make_swizzle(4, 3, 3), dtype=cutlass.BFloat16)
    _k_smem = cute.make_tensor(k_smem, _flat_layout)
    k_smem_addr = smem + 17408
    vt_smem = cute.recast_ptr(smem_raw + 33792, swizzle_=cute.make_swizzle(4, 3, 3), dtype=cutlass.BFloat16)
    _vt_smem = cute.make_tensor(vt_smem, _flat_layout)
    vt_smem_addr = smem + 33792
    q_full_addr = cute.recast_ptr(smem_raw, dtype=cutlass.Uint64)
    k_full0_addr = cute.recast_ptr(smem_raw + 8, dtype=cutlass.Uint64)
    v_full0_addr = cute.recast_ptr(smem_raw + 16, dtype=cutlass.Uint64)
    if warp == 0:
        with cute.arch.elect_one():
            cute.arch.mbarrier_init(q_full_addr + 0, 1)
            cute.arch.mbarrier_init(k_full0_addr + 0, 1)
            cute.arch.mbarrier_init(v_full0_addr + 0, 1)
    cute.arch.mbarrier_init_fence()
    cute.arch.sync_threads()
    row_max0 = cute.make_rmem_tensor((1,), cutlass.Float32)
    row_max1 = cute.make_rmem_tensor((1,), cutlass.Float32)
    row_sum0 = cute.make_rmem_tensor((1,), cutlass.Float32)
    row_sum1 = cute.make_rmem_tensor((1,), cutlass.Float32)
    _phase_q_full_0 = cute.make_rmem_tensor((1,), cutlass.Uint32)
    _phase_k_full0_0 = cute.make_rmem_tensor((1,), cutlass.Uint32)
    _phase_v_full0_0 = cute.make_rmem_tensor((1,), cutlass.Uint32)
    new_max0 = cute.make_rmem_tensor((1,), cutlass.Float32)
    new_max1 = cute.make_rmem_tensor((1,), cutlass.Float32)
    cute.arch.prefetch(Q.get_ptr(), tensormap=True, predicate=cutlass.Boolean((warp == 0)))
    cute.arch.prefetch(K.get_ptr(), tensormap=True, predicate=cutlass.Boolean((warp == 0)))
    cute.arch.prefetch(Vt.get_ptr(), tensormap=True, predicate=cutlass.Boolean((warp == 0)))
    item = cutlass.Int32(bid)
    mb = cutlass.Int32(cutlass.Int32(cutlass_arith.divsi(cutlass.Int32(seqlen_q).ir_value(), cutlass.Int32(64).ir_value())))
    plan_base = cutlass.Int32((item * 2))
    meta = cutlass.Int32(cute.make_tensor(plan, _flat_layout)[plan_base])
    cnt = cutlass.Int32(meta)
    tile = cutlass.Int32(item)
    blk_base = cutlass.Int32((plan_base + 1))
    head = cutlass.Int32(cutlass.Int32(cutlass_arith.divsi(cutlass.Int32(tile).ir_value(), cutlass.Int32(mb).ir_value())))
    qb = cutlass.Int32((tile - (head * mb)))
    q_row = cutlass.Int32(((head * seqlen_q) + (qb * 64)))
    kv_base = cutlass.Int32((head * seqlen_k))
    if (warp == 0):
        if prims.elect_sync():
            blk_pre = cute.make_rmem_tensor((1,), cutlass.Int32)
            blk_pre[0] = cutlass.Int32(cute.make_tensor(plan, _flat_layout)[blk_base])
            cute.arch.mbarrier_arrive_and_expect_tx(q_full_addr, 16384)
            prims.cp_async_bulk_tensor_shared_cta_global(
                cute.make_ptr(cutlass.Uint8, cutlass.Uint32(q_smem_addr), mem_space=cute.AddressSpace.smem, assumed_align=16),
                Q.get_ptr(),
                [cutlass.Int32(0), cutlass.Int32(q_row), cutlass.Int32(0)],
                q_full_addr,
                mode=prims.TMALoadMode.TILE,
            )
            if (cnt > 0):
                blk_row = cutlass.Int32((kv_base + (blk_pre[0] * 64)))
                cute.arch.mbarrier_arrive_and_expect_tx(k_full0_addr, 16384)
                prims.cp_async_bulk_tensor_shared_cta_global(
                    cute.make_ptr(cutlass.Uint8, cutlass.Uint32(k_smem_addr), mem_space=cute.AddressSpace.smem, assumed_align=16),
                    K.get_ptr(),
                    [cutlass.Int32(0), cutlass.Int32(blk_row), cutlass.Int32(0)],
                    k_full0_addr,
                    mode=prims.TMALoadMode.TILE,
                )
                cute.arch.mbarrier_arrive_and_expect_tx(v_full0_addr, 16384)
                prims.cp_async_bulk_tensor_shared_cta_global(
                    cute.make_ptr(cutlass.Uint8, cutlass.Uint32(vt_smem_addr), mem_space=cute.AddressSpace.smem, assumed_align=16),
                    Vt.get_ptr(),
                    [cutlass.Int32(0), cutlass.Int32(0), cutlass.Int32(cutlass.Int32(cutlass_arith.divsi(cutlass.Int32(blk_row).ir_value(), cutlass.Int32(8).ir_value()))), cutlass.Int32(0)],
                    v_full0_addr,
                    mode=prims.TMALoadMode.TILE,
                )
    m0_local = cutlass.Int32(((warp * 16) + cutlass.Int32(cutlass_arith.divsi(cutlass.Int32(lane).ir_value(), cutlass.Int32(4).ir_value()))))
    m1_local = cutlass.Int32((m0_local + 8))
    d_o = cute.make_rmem_tensor((64,), cutlass.Float32)
    d_qk = cute.make_rmem_tensor((32,), cutlass.Float32)
    p_bf16 = cute.make_rmem_tensor((16,), cutlass.Uint32)
    row_max0[0] = cutlass.Float32((0 - float("inf")))
    row_max1[0] = cutlass.Float32((0 - float("inf")))
    row_sum0[0] = cutlass.Float32(0.0)
    row_sum1[0] = cutlass.Float32(0.0)
    _phase_q_full_0[0] = cutlass.Uint32(0)
    while not prims.mbarrier_wait_parity(q_full_addr, _phase_q_full_0[0], prims.MBarrierWait.TRY, scope=prims.MBarrierScope.CTA, order=prims.MemOrder.ACQUIRE):
        pass
    _phase_q_full_0[0] ^= cutlass.Uint32(1)
    _phase_k_full0_0[0] = cutlass.Uint32(0)
    _wgmma_a_0_0_raw = ((cutlass.Uint64(cutlass.Uint32(q_smem_addr) >> 4) & cutlass.Uint64(0x3FFF)) | (cutlass.Uint64(0) << 16) | (cutlass.Uint64(64) << 32) | (cutlass.Uint64(1) << 62))
    _wgmma_a_0_0 = (cutlass.Uint64(cute.arch.make_warp_uniform(cutlass.Uint32(_wgmma_a_0_0_raw >> 32))) << 32) | cutlass.Uint64(cute.arch.make_warp_uniform(cutlass.Uint32(_wgmma_a_0_0_raw)))
    _wgmma_b_0_1_raw = ((cutlass.Uint64(cutlass.Uint32(k_smem_addr) >> 4) & cutlass.Uint64(0x3FFF)) | (cutlass.Uint64(0) << 16) | (cutlass.Uint64(64) << 32) | (cutlass.Uint64(1) << 62))
    _wgmma_b_0_1 = (cutlass.Uint64(cute.arch.make_warp_uniform(cutlass.Uint32(_wgmma_b_0_1_raw >> 32))) << 32) | cutlass.Uint64(cute.arch.make_warp_uniform(cutlass.Uint32(_wgmma_b_0_1_raw)))
    _wgmma_a_0_2_raw = ((cutlass.Uint64(cutlass.Uint32((q_smem_addr + 8192)) >> 4) & cutlass.Uint64(0x3FFF)) | (cutlass.Uint64(0) << 16) | (cutlass.Uint64(64) << 32) | (cutlass.Uint64(1) << 62))
    _wgmma_a_0_2 = (cutlass.Uint64(cute.arch.make_warp_uniform(cutlass.Uint32(_wgmma_a_0_2_raw >> 32))) << 32) | cutlass.Uint64(cute.arch.make_warp_uniform(cutlass.Uint32(_wgmma_a_0_2_raw)))
    _wgmma_b_0_3_raw = ((cutlass.Uint64(cutlass.Uint32((k_smem_addr + 8192)) >> 4) & cutlass.Uint64(0x3FFF)) | (cutlass.Uint64(0) << 16) | (cutlass.Uint64(64) << 32) | (cutlass.Uint64(1) << 62))
    _wgmma_b_0_3 = (cutlass.Uint64(cute.arch.make_warp_uniform(cutlass.Uint32(_wgmma_b_0_3_raw >> 32))) << 32) | cutlass.Uint64(cute.arch.make_warp_uniform(cutlass.Uint32(_wgmma_b_0_3_raw)))
    _phase_v_full0_0[0] = cutlass.Uint32(0)
    _wgmma_b_0_4_raw = ((cutlass.Uint64(cutlass.Uint32(vt_smem_addr) >> 4) & cutlass.Uint64(0x3FFF)) | (cutlass.Uint64(512) << 16) | (cutlass.Uint64(64) << 32) | (cutlass.Uint64(1) << 62))
    _wgmma_b_0_4 = (cutlass.Uint64(cute.arch.make_warp_uniform(cutlass.Uint32(_wgmma_b_0_4_raw >> 32))) << 32) | cutlass.Uint64(cute.arch.make_warp_uniform(cutlass.Uint32(_wgmma_b_0_4_raw)))
    if (cnt > 0):
        while not prims.mbarrier_wait_parity(k_full0_addr, _phase_k_full0_0[0], prims.MBarrierWait.TRY, scope=prims.MBarrierScope.CTA, order=prims.MemOrder.ACQUIRE):
            pass
        _phase_k_full0_0[0] ^= cutlass.Uint32(1)
        cute.nvgpu.warpgroup.fence()
        _wgmma_0 = cutlass_hopper.make_trivial_tiled_mma(
            cutlass.BFloat16, cutlass.BFloat16,
            cute.nvgpu.OperandMajorMode.K, cute.nvgpu.OperandMajorMode.K,
            cutlass.Float32, (1, 1, 1), (64, 64),
            cute.nvgpu.warpgroup.OperandSource.SMEM,
        )
        _wgmma_0_thr = _wgmma_0.get_slice(tid)
        _wgmma_0_b_ptr = cute.make_ptr(cutlass.BFloat16, cutlass.Uint32((cutlass.Uint64(_wgmma_b_0_1) & cutlass.Uint64(0x3FFF)) << 4), mem_space=cute.AddressSpace.smem, assumed_align=16)
        _wgmma_0_b_layout = cutlass_hopper.make_smem_layout_b(CutlassLayout.ROW_MAJOR, (64, 64, 64), cutlass.BFloat16, 1)
        _wgmma_0_b_ptr = cute.recast_ptr(_wgmma_0_b_ptr, swizzle_=_wgmma_0_b_layout.inner, dtype=cutlass.BFloat16)
        _wgmma_0_b = cute.make_tensor(_wgmma_0_b_ptr, _wgmma_0_b_layout.outer)
        _wgmma_0_b_part = _wgmma_0_thr.partition_B(_wgmma_0_b)
        _wgmma_0_b_frag = _wgmma_0.make_fragment_B(_wgmma_0_b_part)
        _wgmma_0_a_ptr = cute.make_ptr(cutlass.BFloat16, cutlass.Uint32((cutlass.Uint64(_wgmma_a_0_0) & cutlass.Uint64(0x3FFF)) << 4), mem_space=cute.AddressSpace.smem, assumed_align=16)
        _wgmma_0_a_layout = cutlass_hopper.make_smem_layout_a(CutlassLayout.ROW_MAJOR, (64, 64, 64), cutlass.BFloat16, 1)
        _wgmma_0_a_ptr = cute.recast_ptr(_wgmma_0_a_ptr, swizzle_=_wgmma_0_a_layout.inner, dtype=cutlass.BFloat16)
        _wgmma_0_a = cute.make_tensor(_wgmma_0_a_ptr, _wgmma_0_a_layout.outer)
        _wgmma_0_a_part = _wgmma_0_thr.partition_A(_wgmma_0_a)
        _wgmma_0_a_frag = _wgmma_0.make_fragment_A(_wgmma_0_a_part)
        _wgmma_0_d = cute.make_tensor(d_qk.iterator + (0), _wgmma_0.partition_shape_C((64, 64)))
        _wgmma_0.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, False)
        cute.gemm(_wgmma_0, _wgmma_0_d, _wgmma_0_a_frag[None, None, 0, 0], _wgmma_0_b_frag[None, None, 0, 0], _wgmma_0_d)
        _wgmma_1 = cutlass_hopper.make_trivial_tiled_mma(
            cutlass.BFloat16, cutlass.BFloat16,
            cute.nvgpu.OperandMajorMode.K, cute.nvgpu.OperandMajorMode.K,
            cutlass.Float32, (1, 1, 1), (64, 64),
            cute.nvgpu.warpgroup.OperandSource.SMEM,
        )
        _wgmma_1_thr = _wgmma_1.get_slice(tid)
        _wgmma_1_b_ptr = cute.make_ptr(cutlass.BFloat16, cutlass.Uint32((cutlass.Uint64((_wgmma_b_0_1 + 2)) & cutlass.Uint64(0x3FFF)) << 4), mem_space=cute.AddressSpace.smem, assumed_align=16)
        _wgmma_1_b_layout = cutlass_hopper.make_smem_layout_b(CutlassLayout.ROW_MAJOR, (64, 64, 64), cutlass.BFloat16, 1)
        _wgmma_1_b_ptr = cute.recast_ptr(_wgmma_1_b_ptr, swizzle_=_wgmma_1_b_layout.inner, dtype=cutlass.BFloat16)
        _wgmma_1_b = cute.make_tensor(_wgmma_1_b_ptr, _wgmma_1_b_layout.outer)
        _wgmma_1_b_part = _wgmma_1_thr.partition_B(_wgmma_1_b)
        _wgmma_1_b_frag = _wgmma_1.make_fragment_B(_wgmma_1_b_part)
        _wgmma_1_a_ptr = cute.make_ptr(cutlass.BFloat16, cutlass.Uint32((cutlass.Uint64((_wgmma_a_0_0 + 2)) & cutlass.Uint64(0x3FFF)) << 4), mem_space=cute.AddressSpace.smem, assumed_align=16)
        _wgmma_1_a_layout = cutlass_hopper.make_smem_layout_a(CutlassLayout.ROW_MAJOR, (64, 64, 64), cutlass.BFloat16, 1)
        _wgmma_1_a_ptr = cute.recast_ptr(_wgmma_1_a_ptr, swizzle_=_wgmma_1_a_layout.inner, dtype=cutlass.BFloat16)
        _wgmma_1_a = cute.make_tensor(_wgmma_1_a_ptr, _wgmma_1_a_layout.outer)
        _wgmma_1_a_part = _wgmma_1_thr.partition_A(_wgmma_1_a)
        _wgmma_1_a_frag = _wgmma_1.make_fragment_A(_wgmma_1_a_part)
        _wgmma_1_d = cute.make_tensor(d_qk.iterator + (0), _wgmma_1.partition_shape_C((64, 64)))
        _wgmma_1.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, True)
        cute.gemm(_wgmma_1, _wgmma_1_d, _wgmma_1_a_frag[None, None, 0, 0], _wgmma_1_b_frag[None, None, 0, 0], _wgmma_1_d)
        _wgmma_2 = cutlass_hopper.make_trivial_tiled_mma(
            cutlass.BFloat16, cutlass.BFloat16,
            cute.nvgpu.OperandMajorMode.K, cute.nvgpu.OperandMajorMode.K,
            cutlass.Float32, (1, 1, 1), (64, 64),
            cute.nvgpu.warpgroup.OperandSource.SMEM,
        )
        _wgmma_2_thr = _wgmma_2.get_slice(tid)
        _wgmma_2_b_ptr = cute.make_ptr(cutlass.BFloat16, cutlass.Uint32((cutlass.Uint64((_wgmma_b_0_1 + 4)) & cutlass.Uint64(0x3FFF)) << 4), mem_space=cute.AddressSpace.smem, assumed_align=16)
        _wgmma_2_b_layout = cutlass_hopper.make_smem_layout_b(CutlassLayout.ROW_MAJOR, (64, 64, 64), cutlass.BFloat16, 1)
        _wgmma_2_b_ptr = cute.recast_ptr(_wgmma_2_b_ptr, swizzle_=_wgmma_2_b_layout.inner, dtype=cutlass.BFloat16)
        _wgmma_2_b = cute.make_tensor(_wgmma_2_b_ptr, _wgmma_2_b_layout.outer)
        _wgmma_2_b_part = _wgmma_2_thr.partition_B(_wgmma_2_b)
        _wgmma_2_b_frag = _wgmma_2.make_fragment_B(_wgmma_2_b_part)
        _wgmma_2_a_ptr = cute.make_ptr(cutlass.BFloat16, cutlass.Uint32((cutlass.Uint64((_wgmma_a_0_0 + 4)) & cutlass.Uint64(0x3FFF)) << 4), mem_space=cute.AddressSpace.smem, assumed_align=16)
        _wgmma_2_a_layout = cutlass_hopper.make_smem_layout_a(CutlassLayout.ROW_MAJOR, (64, 64, 64), cutlass.BFloat16, 1)
        _wgmma_2_a_ptr = cute.recast_ptr(_wgmma_2_a_ptr, swizzle_=_wgmma_2_a_layout.inner, dtype=cutlass.BFloat16)
        _wgmma_2_a = cute.make_tensor(_wgmma_2_a_ptr, _wgmma_2_a_layout.outer)
        _wgmma_2_a_part = _wgmma_2_thr.partition_A(_wgmma_2_a)
        _wgmma_2_a_frag = _wgmma_2.make_fragment_A(_wgmma_2_a_part)
        _wgmma_2_d = cute.make_tensor(d_qk.iterator + (0), _wgmma_2.partition_shape_C((64, 64)))
        _wgmma_2.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, True)
        cute.gemm(_wgmma_2, _wgmma_2_d, _wgmma_2_a_frag[None, None, 0, 0], _wgmma_2_b_frag[None, None, 0, 0], _wgmma_2_d)
        _wgmma_3 = cutlass_hopper.make_trivial_tiled_mma(
            cutlass.BFloat16, cutlass.BFloat16,
            cute.nvgpu.OperandMajorMode.K, cute.nvgpu.OperandMajorMode.K,
            cutlass.Float32, (1, 1, 1), (64, 64),
            cute.nvgpu.warpgroup.OperandSource.SMEM,
        )
        _wgmma_3_thr = _wgmma_3.get_slice(tid)
        _wgmma_3_b_ptr = cute.make_ptr(cutlass.BFloat16, cutlass.Uint32((cutlass.Uint64((_wgmma_b_0_1 + 6)) & cutlass.Uint64(0x3FFF)) << 4), mem_space=cute.AddressSpace.smem, assumed_align=16)
        _wgmma_3_b_layout = cutlass_hopper.make_smem_layout_b(CutlassLayout.ROW_MAJOR, (64, 64, 64), cutlass.BFloat16, 1)
        _wgmma_3_b_ptr = cute.recast_ptr(_wgmma_3_b_ptr, swizzle_=_wgmma_3_b_layout.inner, dtype=cutlass.BFloat16)
        _wgmma_3_b = cute.make_tensor(_wgmma_3_b_ptr, _wgmma_3_b_layout.outer)
        _wgmma_3_b_part = _wgmma_3_thr.partition_B(_wgmma_3_b)
        _wgmma_3_b_frag = _wgmma_3.make_fragment_B(_wgmma_3_b_part)
        _wgmma_3_a_ptr = cute.make_ptr(cutlass.BFloat16, cutlass.Uint32((cutlass.Uint64((_wgmma_a_0_0 + 6)) & cutlass.Uint64(0x3FFF)) << 4), mem_space=cute.AddressSpace.smem, assumed_align=16)
        _wgmma_3_a_layout = cutlass_hopper.make_smem_layout_a(CutlassLayout.ROW_MAJOR, (64, 64, 64), cutlass.BFloat16, 1)
        _wgmma_3_a_ptr = cute.recast_ptr(_wgmma_3_a_ptr, swizzle_=_wgmma_3_a_layout.inner, dtype=cutlass.BFloat16)
        _wgmma_3_a = cute.make_tensor(_wgmma_3_a_ptr, _wgmma_3_a_layout.outer)
        _wgmma_3_a_part = _wgmma_3_thr.partition_A(_wgmma_3_a)
        _wgmma_3_a_frag = _wgmma_3.make_fragment_A(_wgmma_3_a_part)
        _wgmma_3_d = cute.make_tensor(d_qk.iterator + (0), _wgmma_3.partition_shape_C((64, 64)))
        _wgmma_3.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, True)
        cute.gemm(_wgmma_3, _wgmma_3_d, _wgmma_3_a_frag[None, None, 0, 0], _wgmma_3_b_frag[None, None, 0, 0], _wgmma_3_d)
        _wgmma_4 = cutlass_hopper.make_trivial_tiled_mma(
            cutlass.BFloat16, cutlass.BFloat16,
            cute.nvgpu.OperandMajorMode.K, cute.nvgpu.OperandMajorMode.K,
            cutlass.Float32, (1, 1, 1), (64, 64),
            cute.nvgpu.warpgroup.OperandSource.SMEM,
        )
        _wgmma_4_thr = _wgmma_4.get_slice(tid)
        _wgmma_4_b_ptr = cute.make_ptr(cutlass.BFloat16, cutlass.Uint32((cutlass.Uint64(_wgmma_b_0_3) & cutlass.Uint64(0x3FFF)) << 4), mem_space=cute.AddressSpace.smem, assumed_align=16)
        _wgmma_4_b_layout = cutlass_hopper.make_smem_layout_b(CutlassLayout.ROW_MAJOR, (64, 64, 64), cutlass.BFloat16, 1)
        _wgmma_4_b_ptr = cute.recast_ptr(_wgmma_4_b_ptr, swizzle_=_wgmma_4_b_layout.inner, dtype=cutlass.BFloat16)
        _wgmma_4_b = cute.make_tensor(_wgmma_4_b_ptr, _wgmma_4_b_layout.outer)
        _wgmma_4_b_part = _wgmma_4_thr.partition_B(_wgmma_4_b)
        _wgmma_4_b_frag = _wgmma_4.make_fragment_B(_wgmma_4_b_part)
        _wgmma_4_a_ptr = cute.make_ptr(cutlass.BFloat16, cutlass.Uint32((cutlass.Uint64(_wgmma_a_0_2) & cutlass.Uint64(0x3FFF)) << 4), mem_space=cute.AddressSpace.smem, assumed_align=16)
        _wgmma_4_a_layout = cutlass_hopper.make_smem_layout_a(CutlassLayout.ROW_MAJOR, (64, 64, 64), cutlass.BFloat16, 1)
        _wgmma_4_a_ptr = cute.recast_ptr(_wgmma_4_a_ptr, swizzle_=_wgmma_4_a_layout.inner, dtype=cutlass.BFloat16)
        _wgmma_4_a = cute.make_tensor(_wgmma_4_a_ptr, _wgmma_4_a_layout.outer)
        _wgmma_4_a_part = _wgmma_4_thr.partition_A(_wgmma_4_a)
        _wgmma_4_a_frag = _wgmma_4.make_fragment_A(_wgmma_4_a_part)
        _wgmma_4_d = cute.make_tensor(d_qk.iterator + (0), _wgmma_4.partition_shape_C((64, 64)))
        _wgmma_4.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, True)
        cute.gemm(_wgmma_4, _wgmma_4_d, _wgmma_4_a_frag[None, None, 0, 0], _wgmma_4_b_frag[None, None, 0, 0], _wgmma_4_d)
        _wgmma_5 = cutlass_hopper.make_trivial_tiled_mma(
            cutlass.BFloat16, cutlass.BFloat16,
            cute.nvgpu.OperandMajorMode.K, cute.nvgpu.OperandMajorMode.K,
            cutlass.Float32, (1, 1, 1), (64, 64),
            cute.nvgpu.warpgroup.OperandSource.SMEM,
        )
        _wgmma_5_thr = _wgmma_5.get_slice(tid)
        _wgmma_5_b_ptr = cute.make_ptr(cutlass.BFloat16, cutlass.Uint32((cutlass.Uint64((_wgmma_b_0_3 + 2)) & cutlass.Uint64(0x3FFF)) << 4), mem_space=cute.AddressSpace.smem, assumed_align=16)
        _wgmma_5_b_layout = cutlass_hopper.make_smem_layout_b(CutlassLayout.ROW_MAJOR, (64, 64, 64), cutlass.BFloat16, 1)
        _wgmma_5_b_ptr = cute.recast_ptr(_wgmma_5_b_ptr, swizzle_=_wgmma_5_b_layout.inner, dtype=cutlass.BFloat16)
        _wgmma_5_b = cute.make_tensor(_wgmma_5_b_ptr, _wgmma_5_b_layout.outer)
        _wgmma_5_b_part = _wgmma_5_thr.partition_B(_wgmma_5_b)
        _wgmma_5_b_frag = _wgmma_5.make_fragment_B(_wgmma_5_b_part)
        _wgmma_5_a_ptr = cute.make_ptr(cutlass.BFloat16, cutlass.Uint32((cutlass.Uint64((_wgmma_a_0_2 + 2)) & cutlass.Uint64(0x3FFF)) << 4), mem_space=cute.AddressSpace.smem, assumed_align=16)
        _wgmma_5_a_layout = cutlass_hopper.make_smem_layout_a(CutlassLayout.ROW_MAJOR, (64, 64, 64), cutlass.BFloat16, 1)
        _wgmma_5_a_ptr = cute.recast_ptr(_wgmma_5_a_ptr, swizzle_=_wgmma_5_a_layout.inner, dtype=cutlass.BFloat16)
        _wgmma_5_a = cute.make_tensor(_wgmma_5_a_ptr, _wgmma_5_a_layout.outer)
        _wgmma_5_a_part = _wgmma_5_thr.partition_A(_wgmma_5_a)
        _wgmma_5_a_frag = _wgmma_5.make_fragment_A(_wgmma_5_a_part)
        _wgmma_5_d = cute.make_tensor(d_qk.iterator + (0), _wgmma_5.partition_shape_C((64, 64)))
        _wgmma_5.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, True)
        cute.gemm(_wgmma_5, _wgmma_5_d, _wgmma_5_a_frag[None, None, 0, 0], _wgmma_5_b_frag[None, None, 0, 0], _wgmma_5_d)
        _wgmma_6 = cutlass_hopper.make_trivial_tiled_mma(
            cutlass.BFloat16, cutlass.BFloat16,
            cute.nvgpu.OperandMajorMode.K, cute.nvgpu.OperandMajorMode.K,
            cutlass.Float32, (1, 1, 1), (64, 64),
            cute.nvgpu.warpgroup.OperandSource.SMEM,
        )
        _wgmma_6_thr = _wgmma_6.get_slice(tid)
        _wgmma_6_b_ptr = cute.make_ptr(cutlass.BFloat16, cutlass.Uint32((cutlass.Uint64((_wgmma_b_0_3 + 4)) & cutlass.Uint64(0x3FFF)) << 4), mem_space=cute.AddressSpace.smem, assumed_align=16)
        _wgmma_6_b_layout = cutlass_hopper.make_smem_layout_b(CutlassLayout.ROW_MAJOR, (64, 64, 64), cutlass.BFloat16, 1)
        _wgmma_6_b_ptr = cute.recast_ptr(_wgmma_6_b_ptr, swizzle_=_wgmma_6_b_layout.inner, dtype=cutlass.BFloat16)
        _wgmma_6_b = cute.make_tensor(_wgmma_6_b_ptr, _wgmma_6_b_layout.outer)
        _wgmma_6_b_part = _wgmma_6_thr.partition_B(_wgmma_6_b)
        _wgmma_6_b_frag = _wgmma_6.make_fragment_B(_wgmma_6_b_part)
        _wgmma_6_a_ptr = cute.make_ptr(cutlass.BFloat16, cutlass.Uint32((cutlass.Uint64((_wgmma_a_0_2 + 4)) & cutlass.Uint64(0x3FFF)) << 4), mem_space=cute.AddressSpace.smem, assumed_align=16)
        _wgmma_6_a_layout = cutlass_hopper.make_smem_layout_a(CutlassLayout.ROW_MAJOR, (64, 64, 64), cutlass.BFloat16, 1)
        _wgmma_6_a_ptr = cute.recast_ptr(_wgmma_6_a_ptr, swizzle_=_wgmma_6_a_layout.inner, dtype=cutlass.BFloat16)
        _wgmma_6_a = cute.make_tensor(_wgmma_6_a_ptr, _wgmma_6_a_layout.outer)
        _wgmma_6_a_part = _wgmma_6_thr.partition_A(_wgmma_6_a)
        _wgmma_6_a_frag = _wgmma_6.make_fragment_A(_wgmma_6_a_part)
        _wgmma_6_d = cute.make_tensor(d_qk.iterator + (0), _wgmma_6.partition_shape_C((64, 64)))
        _wgmma_6.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, True)
        cute.gemm(_wgmma_6, _wgmma_6_d, _wgmma_6_a_frag[None, None, 0, 0], _wgmma_6_b_frag[None, None, 0, 0], _wgmma_6_d)
        _wgmma_7 = cutlass_hopper.make_trivial_tiled_mma(
            cutlass.BFloat16, cutlass.BFloat16,
            cute.nvgpu.OperandMajorMode.K, cute.nvgpu.OperandMajorMode.K,
            cutlass.Float32, (1, 1, 1), (64, 64),
            cute.nvgpu.warpgroup.OperandSource.SMEM,
        )
        _wgmma_7_thr = _wgmma_7.get_slice(tid)
        _wgmma_7_b_ptr = cute.make_ptr(cutlass.BFloat16, cutlass.Uint32((cutlass.Uint64((_wgmma_b_0_3 + 6)) & cutlass.Uint64(0x3FFF)) << 4), mem_space=cute.AddressSpace.smem, assumed_align=16)
        _wgmma_7_b_layout = cutlass_hopper.make_smem_layout_b(CutlassLayout.ROW_MAJOR, (64, 64, 64), cutlass.BFloat16, 1)
        _wgmma_7_b_ptr = cute.recast_ptr(_wgmma_7_b_ptr, swizzle_=_wgmma_7_b_layout.inner, dtype=cutlass.BFloat16)
        _wgmma_7_b = cute.make_tensor(_wgmma_7_b_ptr, _wgmma_7_b_layout.outer)
        _wgmma_7_b_part = _wgmma_7_thr.partition_B(_wgmma_7_b)
        _wgmma_7_b_frag = _wgmma_7.make_fragment_B(_wgmma_7_b_part)
        _wgmma_7_a_ptr = cute.make_ptr(cutlass.BFloat16, cutlass.Uint32((cutlass.Uint64((_wgmma_a_0_2 + 6)) & cutlass.Uint64(0x3FFF)) << 4), mem_space=cute.AddressSpace.smem, assumed_align=16)
        _wgmma_7_a_layout = cutlass_hopper.make_smem_layout_a(CutlassLayout.ROW_MAJOR, (64, 64, 64), cutlass.BFloat16, 1)
        _wgmma_7_a_ptr = cute.recast_ptr(_wgmma_7_a_ptr, swizzle_=_wgmma_7_a_layout.inner, dtype=cutlass.BFloat16)
        _wgmma_7_a = cute.make_tensor(_wgmma_7_a_ptr, _wgmma_7_a_layout.outer)
        _wgmma_7_a_part = _wgmma_7_thr.partition_A(_wgmma_7_a)
        _wgmma_7_a_frag = _wgmma_7.make_fragment_A(_wgmma_7_a_part)
        _wgmma_7_d = cute.make_tensor(d_qk.iterator + (0), _wgmma_7.partition_shape_C((64, 64)))
        _wgmma_7.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, True)
        cute.gemm(_wgmma_7, _wgmma_7_d, _wgmma_7_a_frag[None, None, 0, 0], _wgmma_7_b_frag[None, None, 0, 0], _wgmma_7_d)
        cute.nvgpu.warpgroup.commit_group()
        cute.nvgpu.warpgroup.wait_group(0)
        d_qk[0] = cutlass.Float32((d_qk[0] * scale_log2))
        d_qk[1] = cutlass.Float32((d_qk[1] * scale_log2))
        d_qk[2] = cutlass.Float32((d_qk[2] * scale_log2))
        d_qk[3] = cutlass.Float32((d_qk[3] * scale_log2))
        d_qk[4] = cutlass.Float32((d_qk[4] * scale_log2))
        d_qk[5] = cutlass.Float32((d_qk[5] * scale_log2))
        d_qk[6] = cutlass.Float32((d_qk[6] * scale_log2))
        d_qk[7] = cutlass.Float32((d_qk[7] * scale_log2))
        d_qk[8] = cutlass.Float32((d_qk[8] * scale_log2))
        d_qk[9] = cutlass.Float32((d_qk[9] * scale_log2))
        d_qk[10] = cutlass.Float32((d_qk[10] * scale_log2))
        d_qk[11] = cutlass.Float32((d_qk[11] * scale_log2))
        d_qk[12] = cutlass.Float32((d_qk[12] * scale_log2))
        d_qk[13] = cutlass.Float32((d_qk[13] * scale_log2))
        d_qk[14] = cutlass.Float32((d_qk[14] * scale_log2))
        d_qk[15] = cutlass.Float32((d_qk[15] * scale_log2))
        d_qk[16] = cutlass.Float32((d_qk[16] * scale_log2))
        d_qk[17] = cutlass.Float32((d_qk[17] * scale_log2))
        d_qk[18] = cutlass.Float32((d_qk[18] * scale_log2))
        d_qk[19] = cutlass.Float32((d_qk[19] * scale_log2))
        d_qk[20] = cutlass.Float32((d_qk[20] * scale_log2))
        d_qk[21] = cutlass.Float32((d_qk[21] * scale_log2))
        d_qk[22] = cutlass.Float32((d_qk[22] * scale_log2))
        d_qk[23] = cutlass.Float32((d_qk[23] * scale_log2))
        d_qk[24] = cutlass.Float32((d_qk[24] * scale_log2))
        d_qk[25] = cutlass.Float32((d_qk[25] * scale_log2))
        d_qk[26] = cutlass.Float32((d_qk[26] * scale_log2))
        d_qk[27] = cutlass.Float32((d_qk[27] * scale_log2))
        d_qk[28] = cutlass.Float32((d_qk[28] * scale_log2))
        d_qk[29] = cutlass.Float32((d_qk[29] * scale_log2))
        d_qk[30] = cutlass.Float32((d_qk[30] * scale_log2))
        d_qk[31] = cutlass.Float32((d_qk[31] * scale_log2))
        new_max0[0] = cutlass.Float32((0 - float("inf")))
        new_max1[0] = cutlass.Float32((0 - float("inf")))
        _max_0 = cute.arch.fmax(new_max0[0], d_qk[0], ftz=False)
        new_max0[0] = cutlass.Float32(_max_0)
        _max_1 = cute.arch.fmax(new_max0[0], d_qk[1], ftz=False)
        new_max0[0] = cutlass.Float32(_max_1)
        _max_2 = cute.arch.fmax(new_max0[0], d_qk[4], ftz=False)
        new_max0[0] = cutlass.Float32(_max_2)
        _max_3 = cute.arch.fmax(new_max0[0], d_qk[5], ftz=False)
        new_max0[0] = cutlass.Float32(_max_3)
        _max_4 = cute.arch.fmax(new_max0[0], d_qk[8], ftz=False)
        new_max0[0] = cutlass.Float32(_max_4)
        _max_5 = cute.arch.fmax(new_max0[0], d_qk[9], ftz=False)
        new_max0[0] = cutlass.Float32(_max_5)
        _max_6 = cute.arch.fmax(new_max0[0], d_qk[12], ftz=False)
        new_max0[0] = cutlass.Float32(_max_6)
        _max_7 = cute.arch.fmax(new_max0[0], d_qk[13], ftz=False)
        new_max0[0] = cutlass.Float32(_max_7)
        _max_8 = cute.arch.fmax(new_max0[0], d_qk[16], ftz=False)
        new_max0[0] = cutlass.Float32(_max_8)
        _max_9 = cute.arch.fmax(new_max0[0], d_qk[17], ftz=False)
        new_max0[0] = cutlass.Float32(_max_9)
        _max_10 = cute.arch.fmax(new_max0[0], d_qk[20], ftz=False)
        new_max0[0] = cutlass.Float32(_max_10)
        _max_11 = cute.arch.fmax(new_max0[0], d_qk[21], ftz=False)
        new_max0[0] = cutlass.Float32(_max_11)
        _max_12 = cute.arch.fmax(new_max0[0], d_qk[24], ftz=False)
        new_max0[0] = cutlass.Float32(_max_12)
        _max_13 = cute.arch.fmax(new_max0[0], d_qk[25], ftz=False)
        new_max0[0] = cutlass.Float32(_max_13)
        _max_14 = cute.arch.fmax(new_max0[0], d_qk[28], ftz=False)
        new_max0[0] = cutlass.Float32(_max_14)
        _max_15 = cute.arch.fmax(new_max0[0], d_qk[29], ftz=False)
        new_max0[0] = cutlass.Float32(_max_15)
        _max_16 = cute.arch.fmax(new_max1[0], d_qk[2], ftz=False)
        new_max1[0] = cutlass.Float32(_max_16)
        _max_17 = cute.arch.fmax(new_max1[0], d_qk[3], ftz=False)
        new_max1[0] = cutlass.Float32(_max_17)
        _max_18 = cute.arch.fmax(new_max1[0], d_qk[6], ftz=False)
        new_max1[0] = cutlass.Float32(_max_18)
        _max_19 = cute.arch.fmax(new_max1[0], d_qk[7], ftz=False)
        new_max1[0] = cutlass.Float32(_max_19)
        _max_20 = cute.arch.fmax(new_max1[0], d_qk[10], ftz=False)
        new_max1[0] = cutlass.Float32(_max_20)
        _max_21 = cute.arch.fmax(new_max1[0], d_qk[11], ftz=False)
        new_max1[0] = cutlass.Float32(_max_21)
        _max_22 = cute.arch.fmax(new_max1[0], d_qk[14], ftz=False)
        new_max1[0] = cutlass.Float32(_max_22)
        _max_23 = cute.arch.fmax(new_max1[0], d_qk[15], ftz=False)
        new_max1[0] = cutlass.Float32(_max_23)
        _max_24 = cute.arch.fmax(new_max1[0], d_qk[18], ftz=False)
        new_max1[0] = cutlass.Float32(_max_24)
        _max_25 = cute.arch.fmax(new_max1[0], d_qk[19], ftz=False)
        new_max1[0] = cutlass.Float32(_max_25)
        _max_26 = cute.arch.fmax(new_max1[0], d_qk[22], ftz=False)
        new_max1[0] = cutlass.Float32(_max_26)
        _max_27 = cute.arch.fmax(new_max1[0], d_qk[23], ftz=False)
        new_max1[0] = cutlass.Float32(_max_27)
        _max_28 = cute.arch.fmax(new_max1[0], d_qk[26], ftz=False)
        new_max1[0] = cutlass.Float32(_max_28)
        _max_29 = cute.arch.fmax(new_max1[0], d_qk[27], ftz=False)
        new_max1[0] = cutlass.Float32(_max_29)
        _max_30 = cute.arch.fmax(new_max1[0], d_qk[30], ftz=False)
        new_max1[0] = cutlass.Float32(_max_30)
        _max_31 = cute.arch.fmax(new_max1[0], d_qk[31], ftz=False)
        new_max1[0] = cutlass.Float32(_max_31)
        _shfl_xor_0 = cute.arch.shuffle_sync_bfly(new_max0[0], 2, mask=0xFFFFFFFF, mask_and_clamp=31)
        _max_32 = cute.arch.fmax(new_max0[0], _shfl_xor_0, ftz=False)
        new_max0[0] = cutlass.Float32(_max_32)
        _shfl_xor_1 = cute.arch.shuffle_sync_bfly(new_max0[0], 1, mask=0xFFFFFFFF, mask_and_clamp=31)
        _max_33 = cute.arch.fmax(new_max0[0], _shfl_xor_1, ftz=False)
        new_max0[0] = cutlass.Float32(_max_33)
        _shfl_xor_2 = cute.arch.shuffle_sync_bfly(new_max1[0], 2, mask=0xFFFFFFFF, mask_and_clamp=31)
        _max_34 = cute.arch.fmax(new_max1[0], _shfl_xor_2, ftz=False)
        new_max1[0] = cutlass.Float32(_max_34)
        _shfl_xor_3 = cute.arch.shuffle_sync_bfly(new_max1[0], 1, mask=0xFFFFFFFF, mask_and_clamp=31)
        _max_35 = cute.arch.fmax(new_max1[0], _shfl_xor_3, ftz=False)
        new_max1[0] = cutlass.Float32(_max_35)
        row_max0[0] = cutlass.Float32(new_max0[0])
        row_max1[0] = cutlass.Float32(new_max1[0])
        _exp2_2 = cute.math.exp2((d_qk[0] - row_max0[0]), approx=True, ftz=True)
        _exp2_3 = cute.math.exp2((d_qk[1] - row_max0[0]), approx=True, ftz=True)
        d_qk[0] = cutlass.Float32(_exp2_2)
        d_qk[1] = cutlass.Float32(_exp2_3)
        row_sum0[0] += cutlass.Float32((_exp2_2 + _exp2_3))
        _exp2_4 = cute.math.exp2((d_qk[2] - row_max1[0]), approx=True, ftz=True)
        _exp2_5 = cute.math.exp2((d_qk[3] - row_max1[0]), approx=True, ftz=True)
        d_qk[2] = cutlass.Float32(_exp2_4)
        d_qk[3] = cutlass.Float32(_exp2_5)
        row_sum1[0] += cutlass.Float32((_exp2_4 + _exp2_5))
        _exp2_6 = cute.math.exp2((d_qk[4] - row_max0[0]), approx=True, ftz=True)
        _exp2_7 = cute.math.exp2((d_qk[5] - row_max0[0]), approx=True, ftz=True)
        d_qk[4] = cutlass.Float32(_exp2_6)
        d_qk[5] = cutlass.Float32(_exp2_7)
        row_sum0[0] += cutlass.Float32((_exp2_6 + _exp2_7))
        _exp2_8 = cute.math.exp2((d_qk[6] - row_max1[0]), approx=True, ftz=True)
        _exp2_9 = cute.math.exp2((d_qk[7] - row_max1[0]), approx=True, ftz=True)
        d_qk[6] = cutlass.Float32(_exp2_8)
        d_qk[7] = cutlass.Float32(_exp2_9)
        row_sum1[0] += cutlass.Float32((_exp2_8 + _exp2_9))
        _exp2_10 = cute.math.exp2((d_qk[8] - row_max0[0]), approx=True, ftz=True)
        _exp2_11 = cute.math.exp2((d_qk[9] - row_max0[0]), approx=True, ftz=True)
        d_qk[8] = cutlass.Float32(_exp2_10)
        d_qk[9] = cutlass.Float32(_exp2_11)
        row_sum0[0] += cutlass.Float32((_exp2_10 + _exp2_11))
        _exp2_12 = cute.math.exp2((d_qk[10] - row_max1[0]), approx=True, ftz=True)
        _exp2_13 = cute.math.exp2((d_qk[11] - row_max1[0]), approx=True, ftz=True)
        d_qk[10] = cutlass.Float32(_exp2_12)
        d_qk[11] = cutlass.Float32(_exp2_13)
        row_sum1[0] += cutlass.Float32((_exp2_12 + _exp2_13))
        _exp2_14 = cute.math.exp2((d_qk[12] - row_max0[0]), approx=True, ftz=True)
        _exp2_15 = cute.math.exp2((d_qk[13] - row_max0[0]), approx=True, ftz=True)
        d_qk[12] = cutlass.Float32(_exp2_14)
        d_qk[13] = cutlass.Float32(_exp2_15)
        row_sum0[0] += cutlass.Float32((_exp2_14 + _exp2_15))
        _exp2_16 = cute.math.exp2((d_qk[14] - row_max1[0]), approx=True, ftz=True)
        _exp2_17 = cute.math.exp2((d_qk[15] - row_max1[0]), approx=True, ftz=True)
        d_qk[14] = cutlass.Float32(_exp2_16)
        d_qk[15] = cutlass.Float32(_exp2_17)
        row_sum1[0] += cutlass.Float32((_exp2_16 + _exp2_17))
        _exp2_18 = cute.math.exp2((d_qk[16] - row_max0[0]), approx=True, ftz=True)
        _exp2_19 = cute.math.exp2((d_qk[17] - row_max0[0]), approx=True, ftz=True)
        d_qk[16] = cutlass.Float32(_exp2_18)
        d_qk[17] = cutlass.Float32(_exp2_19)
        row_sum0[0] += cutlass.Float32((_exp2_18 + _exp2_19))
        _exp2_20 = cute.math.exp2((d_qk[18] - row_max1[0]), approx=True, ftz=True)
        _exp2_21 = cute.math.exp2((d_qk[19] - row_max1[0]), approx=True, ftz=True)
        d_qk[18] = cutlass.Float32(_exp2_20)
        d_qk[19] = cutlass.Float32(_exp2_21)
        row_sum1[0] += cutlass.Float32((_exp2_20 + _exp2_21))
        _exp2_22 = cute.math.exp2((d_qk[20] - row_max0[0]), approx=True, ftz=True)
        _exp2_23 = cute.math.exp2((d_qk[21] - row_max0[0]), approx=True, ftz=True)
        d_qk[20] = cutlass.Float32(_exp2_22)
        d_qk[21] = cutlass.Float32(_exp2_23)
        row_sum0[0] += cutlass.Float32((_exp2_22 + _exp2_23))
        _exp2_24 = cute.math.exp2((d_qk[22] - row_max1[0]), approx=True, ftz=True)
        _exp2_25 = cute.math.exp2((d_qk[23] - row_max1[0]), approx=True, ftz=True)
        d_qk[22] = cutlass.Float32(_exp2_24)
        d_qk[23] = cutlass.Float32(_exp2_25)
        row_sum1[0] += cutlass.Float32((_exp2_24 + _exp2_25))
        _exp2_26 = cute.math.exp2((d_qk[24] - row_max0[0]), approx=True, ftz=True)
        _exp2_27 = cute.math.exp2((d_qk[25] - row_max0[0]), approx=True, ftz=True)
        d_qk[24] = cutlass.Float32(_exp2_26)
        d_qk[25] = cutlass.Float32(_exp2_27)
        row_sum0[0] += cutlass.Float32((_exp2_26 + _exp2_27))
        _exp2_28 = cute.math.exp2((d_qk[26] - row_max1[0]), approx=True, ftz=True)
        _exp2_29 = cute.math.exp2((d_qk[27] - row_max1[0]), approx=True, ftz=True)
        d_qk[26] = cutlass.Float32(_exp2_28)
        d_qk[27] = cutlass.Float32(_exp2_29)
        row_sum1[0] += cutlass.Float32((_exp2_28 + _exp2_29))
        _exp2_30 = cute.math.exp2((d_qk[28] - row_max0[0]), approx=True, ftz=True)
        _exp2_31 = cute.math.exp2((d_qk[29] - row_max0[0]), approx=True, ftz=True)
        d_qk[28] = cutlass.Float32(_exp2_30)
        d_qk[29] = cutlass.Float32(_exp2_31)
        row_sum0[0] += cutlass.Float32((_exp2_30 + _exp2_31))
        _exp2_32 = cute.math.exp2((d_qk[30] - row_max1[0]), approx=True, ftz=True)
        _exp2_33 = cute.math.exp2((d_qk[31] - row_max1[0]), approx=True, ftz=True)
        d_qk[30] = cutlass.Float32(_exp2_32)
        d_qk[31] = cutlass.Float32(_exp2_33)
        row_sum1[0] += cutlass.Float32((_exp2_32 + _exp2_33))
        _bf16x2_0 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_qk[0]), cutlass.Float32(d_qk[1])))[1]), cutlass.Float32(((cutlass.Float32(d_qk[0]), cutlass.Float32(d_qk[1])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
        p_bf16[0] = cutlass.Uint32(_bf16x2_0)
        _bf16x2_1 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_qk[2]), cutlass.Float32(d_qk[3])))[1]), cutlass.Float32(((cutlass.Float32(d_qk[2]), cutlass.Float32(d_qk[3])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
        p_bf16[1] = cutlass.Uint32(_bf16x2_1)
        _bf16x2_2 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_qk[4]), cutlass.Float32(d_qk[5])))[1]), cutlass.Float32(((cutlass.Float32(d_qk[4]), cutlass.Float32(d_qk[5])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
        p_bf16[2] = cutlass.Uint32(_bf16x2_2)
        _bf16x2_3 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_qk[6]), cutlass.Float32(d_qk[7])))[1]), cutlass.Float32(((cutlass.Float32(d_qk[6]), cutlass.Float32(d_qk[7])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
        p_bf16[3] = cutlass.Uint32(_bf16x2_3)
        _bf16x2_4 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_qk[8]), cutlass.Float32(d_qk[9])))[1]), cutlass.Float32(((cutlass.Float32(d_qk[8]), cutlass.Float32(d_qk[9])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
        p_bf16[4] = cutlass.Uint32(_bf16x2_4)
        _bf16x2_5 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_qk[10]), cutlass.Float32(d_qk[11])))[1]), cutlass.Float32(((cutlass.Float32(d_qk[10]), cutlass.Float32(d_qk[11])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
        p_bf16[5] = cutlass.Uint32(_bf16x2_5)
        _bf16x2_6 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_qk[12]), cutlass.Float32(d_qk[13])))[1]), cutlass.Float32(((cutlass.Float32(d_qk[12]), cutlass.Float32(d_qk[13])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
        p_bf16[6] = cutlass.Uint32(_bf16x2_6)
        _bf16x2_7 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_qk[14]), cutlass.Float32(d_qk[15])))[1]), cutlass.Float32(((cutlass.Float32(d_qk[14]), cutlass.Float32(d_qk[15])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
        p_bf16[7] = cutlass.Uint32(_bf16x2_7)
        _bf16x2_8 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_qk[16]), cutlass.Float32(d_qk[17])))[1]), cutlass.Float32(((cutlass.Float32(d_qk[16]), cutlass.Float32(d_qk[17])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
        p_bf16[8] = cutlass.Uint32(_bf16x2_8)
        _bf16x2_9 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_qk[18]), cutlass.Float32(d_qk[19])))[1]), cutlass.Float32(((cutlass.Float32(d_qk[18]), cutlass.Float32(d_qk[19])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
        p_bf16[9] = cutlass.Uint32(_bf16x2_9)
        _bf16x2_10 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_qk[20]), cutlass.Float32(d_qk[21])))[1]), cutlass.Float32(((cutlass.Float32(d_qk[20]), cutlass.Float32(d_qk[21])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
        p_bf16[10] = cutlass.Uint32(_bf16x2_10)
        _bf16x2_11 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_qk[22]), cutlass.Float32(d_qk[23])))[1]), cutlass.Float32(((cutlass.Float32(d_qk[22]), cutlass.Float32(d_qk[23])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
        p_bf16[11] = cutlass.Uint32(_bf16x2_11)
        _bf16x2_12 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_qk[24]), cutlass.Float32(d_qk[25])))[1]), cutlass.Float32(((cutlass.Float32(d_qk[24]), cutlass.Float32(d_qk[25])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
        p_bf16[12] = cutlass.Uint32(_bf16x2_12)
        _bf16x2_13 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_qk[26]), cutlass.Float32(d_qk[27])))[1]), cutlass.Float32(((cutlass.Float32(d_qk[26]), cutlass.Float32(d_qk[27])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
        p_bf16[13] = cutlass.Uint32(_bf16x2_13)
        _bf16x2_14 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_qk[28]), cutlass.Float32(d_qk[29])))[1]), cutlass.Float32(((cutlass.Float32(d_qk[28]), cutlass.Float32(d_qk[29])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
        p_bf16[14] = cutlass.Uint32(_bf16x2_14)
        _bf16x2_15 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_qk[30]), cutlass.Float32(d_qk[31])))[1]), cutlass.Float32(((cutlass.Float32(d_qk[30]), cutlass.Float32(d_qk[31])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
        p_bf16[15] = cutlass.Uint32(_bf16x2_15)
        while not prims.mbarrier_wait_parity(v_full0_addr, _phase_v_full0_0[0], prims.MBarrierWait.TRY, scope=prims.MBarrierScope.CTA, order=prims.MemOrder.ACQUIRE):
            pass
        _phase_v_full0_0[0] ^= cutlass.Uint32(1)
        cute.nvgpu.warpgroup.fence()
        _wgmma_8 = cutlass_hopper.make_trivial_tiled_mma(
            cutlass.BFloat16, cutlass.BFloat16,
            cute.nvgpu.OperandMajorMode.K, cute.nvgpu.OperandMajorMode.MN,
            cutlass.Float32, (1, 1, 1), (64, 128),
            cute.nvgpu.warpgroup.OperandSource.RMEM,
        )
        _wgmma_8_thr = _wgmma_8.get_slice(tid)
        _wgmma_8_b_ptr = cute.make_ptr(cutlass.BFloat16, cutlass.Uint32((cutlass.Uint64(_wgmma_b_0_4) & cutlass.Uint64(0x3FFF)) << 4), mem_space=cute.AddressSpace.smem, assumed_align=16)
        _wgmma_8_b_layout = cutlass_hopper.make_smem_layout_b(CutlassLayout.COL_MAJOR, (64, 128, 64), cutlass.BFloat16, 1)
        _wgmma_8_b_ptr = cute.recast_ptr(_wgmma_8_b_ptr, swizzle_=_wgmma_8_b_layout.inner, dtype=cutlass.BFloat16)
        _wgmma_8_b = cute.make_tensor(_wgmma_8_b_ptr, _wgmma_8_b_layout.outer)
        _wgmma_8_b_part = _wgmma_8_thr.partition_B(_wgmma_8_b)
        _wgmma_8_b_frag = _wgmma_8.make_fragment_B(_wgmma_8_b_part)
        _wgmma_8_a_ptr = cute.recast_ptr(p_bf16.iterator + (0), dtype=cutlass.BFloat16)
        _wgmma_8_a_frag = cute.make_tensor(_wgmma_8_a_ptr, _wgmma_8.partition_shape_A((64, 16)))
        _wgmma_8_d = cute.make_tensor(d_o.iterator + (0), _wgmma_8.partition_shape_C((64, 128)))
        _wgmma_8.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, cutlass.Boolean((False) != 0))
        cute.gemm(_wgmma_8, _wgmma_8_d, _wgmma_8_a_frag[None, None, 0], _wgmma_8_b_frag[None, None, 0, 0], _wgmma_8_d)
        _wgmma_9 = cutlass_hopper.make_trivial_tiled_mma(
            cutlass.BFloat16, cutlass.BFloat16,
            cute.nvgpu.OperandMajorMode.K, cute.nvgpu.OperandMajorMode.MN,
            cutlass.Float32, (1, 1, 1), (64, 128),
            cute.nvgpu.warpgroup.OperandSource.RMEM,
        )
        _wgmma_9_thr = _wgmma_9.get_slice(tid)
        _wgmma_9_b_ptr = cute.make_ptr(cutlass.BFloat16, cutlass.Uint32((cutlass.Uint64((_wgmma_b_0_4 + 128)) & cutlass.Uint64(0x3FFF)) << 4), mem_space=cute.AddressSpace.smem, assumed_align=16)
        _wgmma_9_b_layout = cutlass_hopper.make_smem_layout_b(CutlassLayout.COL_MAJOR, (64, 128, 64), cutlass.BFloat16, 1)
        _wgmma_9_b_ptr = cute.recast_ptr(_wgmma_9_b_ptr, swizzle_=_wgmma_9_b_layout.inner, dtype=cutlass.BFloat16)
        _wgmma_9_b = cute.make_tensor(_wgmma_9_b_ptr, _wgmma_9_b_layout.outer)
        _wgmma_9_b_part = _wgmma_9_thr.partition_B(_wgmma_9_b)
        _wgmma_9_b_frag = _wgmma_9.make_fragment_B(_wgmma_9_b_part)
        _wgmma_9_a_ptr = cute.recast_ptr(p_bf16.iterator + (4), dtype=cutlass.BFloat16)
        _wgmma_9_a_frag = cute.make_tensor(_wgmma_9_a_ptr, _wgmma_9.partition_shape_A((64, 16)))
        _wgmma_9_d = cute.make_tensor(d_o.iterator + (0), _wgmma_9.partition_shape_C((64, 128)))
        _wgmma_9.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, cutlass.Boolean((True) != 0))
        cute.gemm(_wgmma_9, _wgmma_9_d, _wgmma_9_a_frag[None, None, 0], _wgmma_9_b_frag[None, None, 0, 0], _wgmma_9_d)
        _wgmma_10 = cutlass_hopper.make_trivial_tiled_mma(
            cutlass.BFloat16, cutlass.BFloat16,
            cute.nvgpu.OperandMajorMode.K, cute.nvgpu.OperandMajorMode.MN,
            cutlass.Float32, (1, 1, 1), (64, 128),
            cute.nvgpu.warpgroup.OperandSource.RMEM,
        )
        _wgmma_10_thr = _wgmma_10.get_slice(tid)
        _wgmma_10_b_ptr = cute.make_ptr(cutlass.BFloat16, cutlass.Uint32((cutlass.Uint64((_wgmma_b_0_4 + 256)) & cutlass.Uint64(0x3FFF)) << 4), mem_space=cute.AddressSpace.smem, assumed_align=16)
        _wgmma_10_b_layout = cutlass_hopper.make_smem_layout_b(CutlassLayout.COL_MAJOR, (64, 128, 64), cutlass.BFloat16, 1)
        _wgmma_10_b_ptr = cute.recast_ptr(_wgmma_10_b_ptr, swizzle_=_wgmma_10_b_layout.inner, dtype=cutlass.BFloat16)
        _wgmma_10_b = cute.make_tensor(_wgmma_10_b_ptr, _wgmma_10_b_layout.outer)
        _wgmma_10_b_part = _wgmma_10_thr.partition_B(_wgmma_10_b)
        _wgmma_10_b_frag = _wgmma_10.make_fragment_B(_wgmma_10_b_part)
        _wgmma_10_a_ptr = cute.recast_ptr(p_bf16.iterator + (8), dtype=cutlass.BFloat16)
        _wgmma_10_a_frag = cute.make_tensor(_wgmma_10_a_ptr, _wgmma_10.partition_shape_A((64, 16)))
        _wgmma_10_d = cute.make_tensor(d_o.iterator + (0), _wgmma_10.partition_shape_C((64, 128)))
        _wgmma_10.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, cutlass.Boolean((True) != 0))
        cute.gemm(_wgmma_10, _wgmma_10_d, _wgmma_10_a_frag[None, None, 0], _wgmma_10_b_frag[None, None, 0, 0], _wgmma_10_d)
        _wgmma_11 = cutlass_hopper.make_trivial_tiled_mma(
            cutlass.BFloat16, cutlass.BFloat16,
            cute.nvgpu.OperandMajorMode.K, cute.nvgpu.OperandMajorMode.MN,
            cutlass.Float32, (1, 1, 1), (64, 128),
            cute.nvgpu.warpgroup.OperandSource.RMEM,
        )
        _wgmma_11_thr = _wgmma_11.get_slice(tid)
        _wgmma_11_b_ptr = cute.make_ptr(cutlass.BFloat16, cutlass.Uint32((cutlass.Uint64((_wgmma_b_0_4 + 384)) & cutlass.Uint64(0x3FFF)) << 4), mem_space=cute.AddressSpace.smem, assumed_align=16)
        _wgmma_11_b_layout = cutlass_hopper.make_smem_layout_b(CutlassLayout.COL_MAJOR, (64, 128, 64), cutlass.BFloat16, 1)
        _wgmma_11_b_ptr = cute.recast_ptr(_wgmma_11_b_ptr, swizzle_=_wgmma_11_b_layout.inner, dtype=cutlass.BFloat16)
        _wgmma_11_b = cute.make_tensor(_wgmma_11_b_ptr, _wgmma_11_b_layout.outer)
        _wgmma_11_b_part = _wgmma_11_thr.partition_B(_wgmma_11_b)
        _wgmma_11_b_frag = _wgmma_11.make_fragment_B(_wgmma_11_b_part)
        _wgmma_11_a_ptr = cute.recast_ptr(p_bf16.iterator + (12), dtype=cutlass.BFloat16)
        _wgmma_11_a_frag = cute.make_tensor(_wgmma_11_a_ptr, _wgmma_11.partition_shape_A((64, 16)))
        _wgmma_11_d = cute.make_tensor(d_o.iterator + (0), _wgmma_11.partition_shape_C((64, 128)))
        _wgmma_11.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, cutlass.Boolean((True) != 0))
        cute.gemm(_wgmma_11, _wgmma_11_d, _wgmma_11_a_frag[None, None, 0], _wgmma_11_b_frag[None, None, 0, 0], _wgmma_11_d)
        cute.nvgpu.warpgroup.commit_group()
        cute.nvgpu.warpgroup.wait_group(0)
    _shfl_xor_4 = cute.arch.shuffle_sync_bfly(row_sum0[0], 2, mask=0xFFFFFFFF, mask_and_clamp=31)
    row_sum0[0] += cutlass.Float32(_shfl_xor_4)
    _shfl_xor_5 = cute.arch.shuffle_sync_bfly(row_sum0[0], 1, mask=0xFFFFFFFF, mask_and_clamp=31)
    row_sum0[0] += cutlass.Float32(_shfl_xor_5)
    _shfl_xor_6 = cute.arch.shuffle_sync_bfly(row_sum1[0], 2, mask=0xFFFFFFFF, mask_and_clamp=31)
    row_sum1[0] += cutlass.Float32(_shfl_xor_6)
    _shfl_xor_7 = cute.arch.shuffle_sync_bfly(row_sum1[0], 1, mask=0xFFFFFFFF, mask_and_clamp=31)
    row_sum1[0] += cutlass.Float32(_shfl_xor_7)
    qj = cutlass.Int32((lane & 3))
    _rcp_0 = cute.math.rcp(row_sum0[0], approx=True, ftz=True)
    _rcp_1 = cute.math.rcp(row_sum1[0], approx=True, ftz=True)
    d_o[0] = cutlass.Float32((d_o[0] * _rcp_0))
    d_o[1] = cutlass.Float32((d_o[1] * _rcp_0))
    d_o[2] = cutlass.Float32((d_o[2] * _rcp_1))
    d_o[3] = cutlass.Float32((d_o[3] * _rcp_1))
    d_o[4] = cutlass.Float32((d_o[4] * _rcp_0))
    d_o[5] = cutlass.Float32((d_o[5] * _rcp_0))
    d_o[6] = cutlass.Float32((d_o[6] * _rcp_1))
    d_o[7] = cutlass.Float32((d_o[7] * _rcp_1))
    d_o[8] = cutlass.Float32((d_o[8] * _rcp_0))
    d_o[9] = cutlass.Float32((d_o[9] * _rcp_0))
    d_o[10] = cutlass.Float32((d_o[10] * _rcp_1))
    d_o[11] = cutlass.Float32((d_o[11] * _rcp_1))
    d_o[12] = cutlass.Float32((d_o[12] * _rcp_0))
    d_o[13] = cutlass.Float32((d_o[13] * _rcp_0))
    d_o[14] = cutlass.Float32((d_o[14] * _rcp_1))
    d_o[15] = cutlass.Float32((d_o[15] * _rcp_1))
    d_o[16] = cutlass.Float32((d_o[16] * _rcp_0))
    d_o[17] = cutlass.Float32((d_o[17] * _rcp_0))
    d_o[18] = cutlass.Float32((d_o[18] * _rcp_1))
    d_o[19] = cutlass.Float32((d_o[19] * _rcp_1))
    d_o[20] = cutlass.Float32((d_o[20] * _rcp_0))
    d_o[21] = cutlass.Float32((d_o[21] * _rcp_0))
    d_o[22] = cutlass.Float32((d_o[22] * _rcp_1))
    d_o[23] = cutlass.Float32((d_o[23] * _rcp_1))
    d_o[24] = cutlass.Float32((d_o[24] * _rcp_0))
    d_o[25] = cutlass.Float32((d_o[25] * _rcp_0))
    d_o[26] = cutlass.Float32((d_o[26] * _rcp_1))
    d_o[27] = cutlass.Float32((d_o[27] * _rcp_1))
    d_o[28] = cutlass.Float32((d_o[28] * _rcp_0))
    d_o[29] = cutlass.Float32((d_o[29] * _rcp_0))
    d_o[30] = cutlass.Float32((d_o[30] * _rcp_1))
    d_o[31] = cutlass.Float32((d_o[31] * _rcp_1))
    d_o[32] = cutlass.Float32((d_o[32] * _rcp_0))
    d_o[33] = cutlass.Float32((d_o[33] * _rcp_0))
    d_o[34] = cutlass.Float32((d_o[34] * _rcp_1))
    d_o[35] = cutlass.Float32((d_o[35] * _rcp_1))
    d_o[36] = cutlass.Float32((d_o[36] * _rcp_0))
    d_o[37] = cutlass.Float32((d_o[37] * _rcp_0))
    d_o[38] = cutlass.Float32((d_o[38] * _rcp_1))
    d_o[39] = cutlass.Float32((d_o[39] * _rcp_1))
    d_o[40] = cutlass.Float32((d_o[40] * _rcp_0))
    d_o[41] = cutlass.Float32((d_o[41] * _rcp_0))
    d_o[42] = cutlass.Float32((d_o[42] * _rcp_1))
    d_o[43] = cutlass.Float32((d_o[43] * _rcp_1))
    d_o[44] = cutlass.Float32((d_o[44] * _rcp_0))
    d_o[45] = cutlass.Float32((d_o[45] * _rcp_0))
    d_o[46] = cutlass.Float32((d_o[46] * _rcp_1))
    d_o[47] = cutlass.Float32((d_o[47] * _rcp_1))
    d_o[48] = cutlass.Float32((d_o[48] * _rcp_0))
    d_o[49] = cutlass.Float32((d_o[49] * _rcp_0))
    d_o[50] = cutlass.Float32((d_o[50] * _rcp_1))
    d_o[51] = cutlass.Float32((d_o[51] * _rcp_1))
    d_o[52] = cutlass.Float32((d_o[52] * _rcp_0))
    d_o[53] = cutlass.Float32((d_o[53] * _rcp_0))
    d_o[54] = cutlass.Float32((d_o[54] * _rcp_1))
    d_o[55] = cutlass.Float32((d_o[55] * _rcp_1))
    d_o[56] = cutlass.Float32((d_o[56] * _rcp_0))
    d_o[57] = cutlass.Float32((d_o[57] * _rcp_0))
    d_o[58] = cutlass.Float32((d_o[58] * _rcp_1))
    d_o[59] = cutlass.Float32((d_o[59] * _rcp_1))
    d_o[60] = cutlass.Float32((d_o[60] * _rcp_0))
    d_o[61] = cutlass.Float32((d_o[61] * _rcp_0))
    d_o[62] = cutlass.Float32((d_o[62] * _rcp_1))
    d_o[63] = cutlass.Float32((d_o[63] * _rcp_1))
    qj1 = cutlass.Int32((qj & 1))
    qj2 = cutlass.Int32((qj & 2))
    o_vec = cute.make_rmem_tensor((4,), cutlass.Uint32)
    o_tmp = cute.make_rmem_tensor((4,), cutlass.Uint32)
    o_row_base = cutlass.Int32((q_row * 128))
    m_local_r = cutlass.Int32((m0_local if True else m1_local))
    _bf16x2_16 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_o[0]), cutlass.Float32(d_o[1])))[1]), cutlass.Float32(((cutlass.Float32(d_o[0]), cutlass.Float32(d_o[1])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
    o_vec[0] = cutlass.Uint32(_bf16x2_16)
    _bf16x2_17 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_o[4]), cutlass.Float32(d_o[5])))[1]), cutlass.Float32(((cutlass.Float32(d_o[4]), cutlass.Float32(d_o[5])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
    o_vec[1] = cutlass.Uint32(_bf16x2_17)
    _bf16x2_18 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_o[8]), cutlass.Float32(d_o[9])))[1]), cutlass.Float32(((cutlass.Float32(d_o[8]), cutlass.Float32(d_o[9])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
    o_vec[2] = cutlass.Uint32(_bf16x2_18)
    _bf16x2_19 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_o[12]), cutlass.Float32(d_o[13])))[1]), cutlass.Float32(((cutlass.Float32(d_o[12]), cutlass.Float32(d_o[13])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
    o_vec[3] = cutlass.Uint32(_bf16x2_19)
    _shfl_xor_8 = cute.arch.shuffle_sync_bfly(o_vec[1], 1, mask=0xFFFFFFFF, mask_and_clamp=31)
    o_tmp[0] = cutlass.Uint32(_shfl_xor_8)
    _shfl_xor_9 = cute.arch.shuffle_sync_bfly(o_vec[0], 1, mask=0xFFFFFFFF, mask_and_clamp=31)
    o_tmp[1] = cutlass.Uint32(_shfl_xor_9)
    _shfl_xor_10 = cute.arch.shuffle_sync_bfly(o_vec[3], 1, mask=0xFFFFFFFF, mask_and_clamp=31)
    o_tmp[2] = cutlass.Uint32(_shfl_xor_10)
    _shfl_xor_11 = cute.arch.shuffle_sync_bfly(o_vec[2], 1, mask=0xFFFFFFFF, mask_and_clamp=31)
    o_tmp[3] = cutlass.Uint32(_shfl_xor_11)
    o_vec[0] = cutlass.Uint32((o_tmp[0] if (qj1 != 0) else o_vec[0]))
    o_vec[1] = cutlass.Uint32((o_tmp[1] if (qj1 == 0) else o_vec[1]))
    o_vec[2] = cutlass.Uint32((o_tmp[2] if (qj1 != 0) else o_vec[2]))
    o_vec[3] = cutlass.Uint32((o_tmp[3] if (qj1 == 0) else o_vec[3]))
    _shfl_xor_12 = cute.arch.shuffle_sync_bfly(o_vec[2], 2, mask=0xFFFFFFFF, mask_and_clamp=31)
    o_tmp[0] = cutlass.Uint32(_shfl_xor_12)
    _shfl_xor_13 = cute.arch.shuffle_sync_bfly(o_vec[3], 2, mask=0xFFFFFFFF, mask_and_clamp=31)
    o_tmp[1] = cutlass.Uint32(_shfl_xor_13)
    _shfl_xor_14 = cute.arch.shuffle_sync_bfly(o_vec[0], 2, mask=0xFFFFFFFF, mask_and_clamp=31)
    o_tmp[2] = cutlass.Uint32(_shfl_xor_14)
    _shfl_xor_15 = cute.arch.shuffle_sync_bfly(o_vec[1], 2, mask=0xFFFFFFFF, mask_and_clamp=31)
    o_tmp[3] = cutlass.Uint32(_shfl_xor_15)
    o_vec[0] = cutlass.Uint32((o_tmp[0] if (qj2 != 0) else o_vec[0]))
    o_vec[1] = cutlass.Uint32((o_tmp[1] if (qj2 != 0) else o_vec[1]))
    o_vec[2] = cutlass.Uint32((o_tmp[2] if (qj2 == 0) else o_vec[2]))
    o_vec[3] = cutlass.Uint32((o_tmp[3] if (qj2 == 0) else o_vec[3]))
    o_off = cutlass.Int32(((o_row_base + (m_local_r * 128)) + (qj * 8)))
    _gmem_store_raw_12 = cutlass.Vector.from_elements([cutlass.Uint32(o_vec[0]), cutlass.Uint32(o_vec[1]), cutlass.Uint32(o_vec[2]), cutlass.Uint32(o_vec[3])], cutlass.Uint32)
    prims.store_ext(_gmem_store_raw_12.ir_value(), O + o_off)
    _bf16x2_20 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_o[16]), cutlass.Float32(d_o[17])))[1]), cutlass.Float32(((cutlass.Float32(d_o[16]), cutlass.Float32(d_o[17])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
    o_vec[0] = cutlass.Uint32(_bf16x2_20)
    _bf16x2_21 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_o[20]), cutlass.Float32(d_o[21])))[1]), cutlass.Float32(((cutlass.Float32(d_o[20]), cutlass.Float32(d_o[21])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
    o_vec[1] = cutlass.Uint32(_bf16x2_21)
    _bf16x2_22 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_o[24]), cutlass.Float32(d_o[25])))[1]), cutlass.Float32(((cutlass.Float32(d_o[24]), cutlass.Float32(d_o[25])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
    o_vec[2] = cutlass.Uint32(_bf16x2_22)
    _bf16x2_23 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_o[28]), cutlass.Float32(d_o[29])))[1]), cutlass.Float32(((cutlass.Float32(d_o[28]), cutlass.Float32(d_o[29])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
    o_vec[3] = cutlass.Uint32(_bf16x2_23)
    _shfl_xor_16 = cute.arch.shuffle_sync_bfly(o_vec[1], 1, mask=0xFFFFFFFF, mask_and_clamp=31)
    o_tmp[0] = cutlass.Uint32(_shfl_xor_16)
    _shfl_xor_17 = cute.arch.shuffle_sync_bfly(o_vec[0], 1, mask=0xFFFFFFFF, mask_and_clamp=31)
    o_tmp[1] = cutlass.Uint32(_shfl_xor_17)
    _shfl_xor_18 = cute.arch.shuffle_sync_bfly(o_vec[3], 1, mask=0xFFFFFFFF, mask_and_clamp=31)
    o_tmp[2] = cutlass.Uint32(_shfl_xor_18)
    _shfl_xor_19 = cute.arch.shuffle_sync_bfly(o_vec[2], 1, mask=0xFFFFFFFF, mask_and_clamp=31)
    o_tmp[3] = cutlass.Uint32(_shfl_xor_19)
    o_vec[0] = cutlass.Uint32((o_tmp[0] if (qj1 != 0) else o_vec[0]))
    o_vec[1] = cutlass.Uint32((o_tmp[1] if (qj1 == 0) else o_vec[1]))
    o_vec[2] = cutlass.Uint32((o_tmp[2] if (qj1 != 0) else o_vec[2]))
    o_vec[3] = cutlass.Uint32((o_tmp[3] if (qj1 == 0) else o_vec[3]))
    _shfl_xor_20 = cute.arch.shuffle_sync_bfly(o_vec[2], 2, mask=0xFFFFFFFF, mask_and_clamp=31)
    o_tmp[0] = cutlass.Uint32(_shfl_xor_20)
    _shfl_xor_21 = cute.arch.shuffle_sync_bfly(o_vec[3], 2, mask=0xFFFFFFFF, mask_and_clamp=31)
    o_tmp[1] = cutlass.Uint32(_shfl_xor_21)
    _shfl_xor_22 = cute.arch.shuffle_sync_bfly(o_vec[0], 2, mask=0xFFFFFFFF, mask_and_clamp=31)
    o_tmp[2] = cutlass.Uint32(_shfl_xor_22)
    _shfl_xor_23 = cute.arch.shuffle_sync_bfly(o_vec[1], 2, mask=0xFFFFFFFF, mask_and_clamp=31)
    o_tmp[3] = cutlass.Uint32(_shfl_xor_23)
    o_vec[0] = cutlass.Uint32((o_tmp[0] if (qj2 != 0) else o_vec[0]))
    o_vec[1] = cutlass.Uint32((o_tmp[1] if (qj2 != 0) else o_vec[1]))
    o_vec[2] = cutlass.Uint32((o_tmp[2] if (qj2 == 0) else o_vec[2]))
    o_vec[3] = cutlass.Uint32((o_tmp[3] if (qj2 == 0) else o_vec[3]))
    o_off_0 = cutlass.Int32(((o_row_base + (m_local_r * 128)) + ((4 + qj) * 8)))
    _gmem_store_raw_13 = cutlass.Vector.from_elements([cutlass.Uint32(o_vec[0]), cutlass.Uint32(o_vec[1]), cutlass.Uint32(o_vec[2]), cutlass.Uint32(o_vec[3])], cutlass.Uint32)
    prims.store_ext(_gmem_store_raw_13.ir_value(), O + o_off_0)
    _bf16x2_24 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_o[32]), cutlass.Float32(d_o[33])))[1]), cutlass.Float32(((cutlass.Float32(d_o[32]), cutlass.Float32(d_o[33])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
    o_vec[0] = cutlass.Uint32(_bf16x2_24)
    _bf16x2_25 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_o[36]), cutlass.Float32(d_o[37])))[1]), cutlass.Float32(((cutlass.Float32(d_o[36]), cutlass.Float32(d_o[37])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
    o_vec[1] = cutlass.Uint32(_bf16x2_25)
    _bf16x2_26 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_o[40]), cutlass.Float32(d_o[41])))[1]), cutlass.Float32(((cutlass.Float32(d_o[40]), cutlass.Float32(d_o[41])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
    o_vec[2] = cutlass.Uint32(_bf16x2_26)
    _bf16x2_27 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_o[44]), cutlass.Float32(d_o[45])))[1]), cutlass.Float32(((cutlass.Float32(d_o[44]), cutlass.Float32(d_o[45])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
    o_vec[3] = cutlass.Uint32(_bf16x2_27)
    _shfl_xor_24 = cute.arch.shuffle_sync_bfly(o_vec[1], 1, mask=0xFFFFFFFF, mask_and_clamp=31)
    o_tmp[0] = cutlass.Uint32(_shfl_xor_24)
    _shfl_xor_25 = cute.arch.shuffle_sync_bfly(o_vec[0], 1, mask=0xFFFFFFFF, mask_and_clamp=31)
    o_tmp[1] = cutlass.Uint32(_shfl_xor_25)
    _shfl_xor_26 = cute.arch.shuffle_sync_bfly(o_vec[3], 1, mask=0xFFFFFFFF, mask_and_clamp=31)
    o_tmp[2] = cutlass.Uint32(_shfl_xor_26)
    _shfl_xor_27 = cute.arch.shuffle_sync_bfly(o_vec[2], 1, mask=0xFFFFFFFF, mask_and_clamp=31)
    o_tmp[3] = cutlass.Uint32(_shfl_xor_27)
    o_vec[0] = cutlass.Uint32((o_tmp[0] if (qj1 != 0) else o_vec[0]))
    o_vec[1] = cutlass.Uint32((o_tmp[1] if (qj1 == 0) else o_vec[1]))
    o_vec[2] = cutlass.Uint32((o_tmp[2] if (qj1 != 0) else o_vec[2]))
    o_vec[3] = cutlass.Uint32((o_tmp[3] if (qj1 == 0) else o_vec[3]))
    _shfl_xor_28 = cute.arch.shuffle_sync_bfly(o_vec[2], 2, mask=0xFFFFFFFF, mask_and_clamp=31)
    o_tmp[0] = cutlass.Uint32(_shfl_xor_28)
    _shfl_xor_29 = cute.arch.shuffle_sync_bfly(o_vec[3], 2, mask=0xFFFFFFFF, mask_and_clamp=31)
    o_tmp[1] = cutlass.Uint32(_shfl_xor_29)
    _shfl_xor_30 = cute.arch.shuffle_sync_bfly(o_vec[0], 2, mask=0xFFFFFFFF, mask_and_clamp=31)
    o_tmp[2] = cutlass.Uint32(_shfl_xor_30)
    _shfl_xor_31 = cute.arch.shuffle_sync_bfly(o_vec[1], 2, mask=0xFFFFFFFF, mask_and_clamp=31)
    o_tmp[3] = cutlass.Uint32(_shfl_xor_31)
    o_vec[0] = cutlass.Uint32((o_tmp[0] if (qj2 != 0) else o_vec[0]))
    o_vec[1] = cutlass.Uint32((o_tmp[1] if (qj2 != 0) else o_vec[1]))
    o_vec[2] = cutlass.Uint32((o_tmp[2] if (qj2 == 0) else o_vec[2]))
    o_vec[3] = cutlass.Uint32((o_tmp[3] if (qj2 == 0) else o_vec[3]))
    o_off_1 = cutlass.Int32(((o_row_base + (m_local_r * 128)) + ((8 + qj) * 8)))
    _gmem_store_raw_14 = cutlass.Vector.from_elements([cutlass.Uint32(o_vec[0]), cutlass.Uint32(o_vec[1]), cutlass.Uint32(o_vec[2]), cutlass.Uint32(o_vec[3])], cutlass.Uint32)
    prims.store_ext(_gmem_store_raw_14.ir_value(), O + o_off_1)
    _bf16x2_28 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_o[48]), cutlass.Float32(d_o[49])))[1]), cutlass.Float32(((cutlass.Float32(d_o[48]), cutlass.Float32(d_o[49])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
    o_vec[0] = cutlass.Uint32(_bf16x2_28)
    _bf16x2_29 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_o[52]), cutlass.Float32(d_o[53])))[1]), cutlass.Float32(((cutlass.Float32(d_o[52]), cutlass.Float32(d_o[53])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
    o_vec[1] = cutlass.Uint32(_bf16x2_29)
    _bf16x2_30 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_o[56]), cutlass.Float32(d_o[57])))[1]), cutlass.Float32(((cutlass.Float32(d_o[56]), cutlass.Float32(d_o[57])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
    o_vec[2] = cutlass.Uint32(_bf16x2_30)
    _bf16x2_31 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_o[60]), cutlass.Float32(d_o[61])))[1]), cutlass.Float32(((cutlass.Float32(d_o[60]), cutlass.Float32(d_o[61])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
    o_vec[3] = cutlass.Uint32(_bf16x2_31)
    _shfl_xor_32 = cute.arch.shuffle_sync_bfly(o_vec[1], 1, mask=0xFFFFFFFF, mask_and_clamp=31)
    o_tmp[0] = cutlass.Uint32(_shfl_xor_32)
    _shfl_xor_33 = cute.arch.shuffle_sync_bfly(o_vec[0], 1, mask=0xFFFFFFFF, mask_and_clamp=31)
    o_tmp[1] = cutlass.Uint32(_shfl_xor_33)
    _shfl_xor_34 = cute.arch.shuffle_sync_bfly(o_vec[3], 1, mask=0xFFFFFFFF, mask_and_clamp=31)
    o_tmp[2] = cutlass.Uint32(_shfl_xor_34)
    _shfl_xor_35 = cute.arch.shuffle_sync_bfly(o_vec[2], 1, mask=0xFFFFFFFF, mask_and_clamp=31)
    o_tmp[3] = cutlass.Uint32(_shfl_xor_35)
    o_vec[0] = cutlass.Uint32((o_tmp[0] if (qj1 != 0) else o_vec[0]))
    o_vec[1] = cutlass.Uint32((o_tmp[1] if (qj1 == 0) else o_vec[1]))
    o_vec[2] = cutlass.Uint32((o_tmp[2] if (qj1 != 0) else o_vec[2]))
    o_vec[3] = cutlass.Uint32((o_tmp[3] if (qj1 == 0) else o_vec[3]))
    _shfl_xor_36 = cute.arch.shuffle_sync_bfly(o_vec[2], 2, mask=0xFFFFFFFF, mask_and_clamp=31)
    o_tmp[0] = cutlass.Uint32(_shfl_xor_36)
    _shfl_xor_37 = cute.arch.shuffle_sync_bfly(o_vec[3], 2, mask=0xFFFFFFFF, mask_and_clamp=31)
    o_tmp[1] = cutlass.Uint32(_shfl_xor_37)
    _shfl_xor_38 = cute.arch.shuffle_sync_bfly(o_vec[0], 2, mask=0xFFFFFFFF, mask_and_clamp=31)
    o_tmp[2] = cutlass.Uint32(_shfl_xor_38)
    _shfl_xor_39 = cute.arch.shuffle_sync_bfly(o_vec[1], 2, mask=0xFFFFFFFF, mask_and_clamp=31)
    o_tmp[3] = cutlass.Uint32(_shfl_xor_39)
    o_vec[0] = cutlass.Uint32((o_tmp[0] if (qj2 != 0) else o_vec[0]))
    o_vec[1] = cutlass.Uint32((o_tmp[1] if (qj2 != 0) else o_vec[1]))
    o_vec[2] = cutlass.Uint32((o_tmp[2] if (qj2 == 0) else o_vec[2]))
    o_vec[3] = cutlass.Uint32((o_tmp[3] if (qj2 == 0) else o_vec[3]))
    o_off_2 = cutlass.Int32(((o_row_base + (m_local_r * 128)) + ((12 + qj) * 8)))
    _gmem_store_raw_15 = cutlass.Vector.from_elements([cutlass.Uint32(o_vec[0]), cutlass.Uint32(o_vec[1]), cutlass.Uint32(o_vec[2]), cutlass.Uint32(o_vec[3])], cutlass.Uint32)
    prims.store_ext(_gmem_store_raw_15.ir_value(), O + o_off_2)
    m_local_r_3 = cutlass.Int32((m0_local if False else m1_local))
    _bf16x2_32 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_o[2]), cutlass.Float32(d_o[3])))[1]), cutlass.Float32(((cutlass.Float32(d_o[2]), cutlass.Float32(d_o[3])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
    o_vec[0] = cutlass.Uint32(_bf16x2_32)
    _bf16x2_33 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_o[6]), cutlass.Float32(d_o[7])))[1]), cutlass.Float32(((cutlass.Float32(d_o[6]), cutlass.Float32(d_o[7])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
    o_vec[1] = cutlass.Uint32(_bf16x2_33)
    _bf16x2_34 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_o[10]), cutlass.Float32(d_o[11])))[1]), cutlass.Float32(((cutlass.Float32(d_o[10]), cutlass.Float32(d_o[11])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
    o_vec[2] = cutlass.Uint32(_bf16x2_34)
    _bf16x2_35 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_o[14]), cutlass.Float32(d_o[15])))[1]), cutlass.Float32(((cutlass.Float32(d_o[14]), cutlass.Float32(d_o[15])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
    o_vec[3] = cutlass.Uint32(_bf16x2_35)
    _shfl_xor_40 = cute.arch.shuffle_sync_bfly(o_vec[1], 1, mask=0xFFFFFFFF, mask_and_clamp=31)
    o_tmp[0] = cutlass.Uint32(_shfl_xor_40)
    _shfl_xor_41 = cute.arch.shuffle_sync_bfly(o_vec[0], 1, mask=0xFFFFFFFF, mask_and_clamp=31)
    o_tmp[1] = cutlass.Uint32(_shfl_xor_41)
    _shfl_xor_42 = cute.arch.shuffle_sync_bfly(o_vec[3], 1, mask=0xFFFFFFFF, mask_and_clamp=31)
    o_tmp[2] = cutlass.Uint32(_shfl_xor_42)
    _shfl_xor_43 = cute.arch.shuffle_sync_bfly(o_vec[2], 1, mask=0xFFFFFFFF, mask_and_clamp=31)
    o_tmp[3] = cutlass.Uint32(_shfl_xor_43)
    o_vec[0] = cutlass.Uint32((o_tmp[0] if (qj1 != 0) else o_vec[0]))
    o_vec[1] = cutlass.Uint32((o_tmp[1] if (qj1 == 0) else o_vec[1]))
    o_vec[2] = cutlass.Uint32((o_tmp[2] if (qj1 != 0) else o_vec[2]))
    o_vec[3] = cutlass.Uint32((o_tmp[3] if (qj1 == 0) else o_vec[3]))
    _shfl_xor_44 = cute.arch.shuffle_sync_bfly(o_vec[2], 2, mask=0xFFFFFFFF, mask_and_clamp=31)
    o_tmp[0] = cutlass.Uint32(_shfl_xor_44)
    _shfl_xor_45 = cute.arch.shuffle_sync_bfly(o_vec[3], 2, mask=0xFFFFFFFF, mask_and_clamp=31)
    o_tmp[1] = cutlass.Uint32(_shfl_xor_45)
    _shfl_xor_46 = cute.arch.shuffle_sync_bfly(o_vec[0], 2, mask=0xFFFFFFFF, mask_and_clamp=31)
    o_tmp[2] = cutlass.Uint32(_shfl_xor_46)
    _shfl_xor_47 = cute.arch.shuffle_sync_bfly(o_vec[1], 2, mask=0xFFFFFFFF, mask_and_clamp=31)
    o_tmp[3] = cutlass.Uint32(_shfl_xor_47)
    o_vec[0] = cutlass.Uint32((o_tmp[0] if (qj2 != 0) else o_vec[0]))
    o_vec[1] = cutlass.Uint32((o_tmp[1] if (qj2 != 0) else o_vec[1]))
    o_vec[2] = cutlass.Uint32((o_tmp[2] if (qj2 == 0) else o_vec[2]))
    o_vec[3] = cutlass.Uint32((o_tmp[3] if (qj2 == 0) else o_vec[3]))
    o_off_4 = cutlass.Int32(((o_row_base + (m_local_r_3 * 128)) + (qj * 8)))
    _gmem_store_raw_16 = cutlass.Vector.from_elements([cutlass.Uint32(o_vec[0]), cutlass.Uint32(o_vec[1]), cutlass.Uint32(o_vec[2]), cutlass.Uint32(o_vec[3])], cutlass.Uint32)
    prims.store_ext(_gmem_store_raw_16.ir_value(), O + o_off_4)
    _bf16x2_36 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_o[18]), cutlass.Float32(d_o[19])))[1]), cutlass.Float32(((cutlass.Float32(d_o[18]), cutlass.Float32(d_o[19])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
    o_vec[0] = cutlass.Uint32(_bf16x2_36)
    _bf16x2_37 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_o[22]), cutlass.Float32(d_o[23])))[1]), cutlass.Float32(((cutlass.Float32(d_o[22]), cutlass.Float32(d_o[23])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
    o_vec[1] = cutlass.Uint32(_bf16x2_37)
    _bf16x2_38 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_o[26]), cutlass.Float32(d_o[27])))[1]), cutlass.Float32(((cutlass.Float32(d_o[26]), cutlass.Float32(d_o[27])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
    o_vec[2] = cutlass.Uint32(_bf16x2_38)
    _bf16x2_39 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_o[30]), cutlass.Float32(d_o[31])))[1]), cutlass.Float32(((cutlass.Float32(d_o[30]), cutlass.Float32(d_o[31])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
    o_vec[3] = cutlass.Uint32(_bf16x2_39)
    _shfl_xor_48 = cute.arch.shuffle_sync_bfly(o_vec[1], 1, mask=0xFFFFFFFF, mask_and_clamp=31)
    o_tmp[0] = cutlass.Uint32(_shfl_xor_48)
    _shfl_xor_49 = cute.arch.shuffle_sync_bfly(o_vec[0], 1, mask=0xFFFFFFFF, mask_and_clamp=31)
    o_tmp[1] = cutlass.Uint32(_shfl_xor_49)
    _shfl_xor_50 = cute.arch.shuffle_sync_bfly(o_vec[3], 1, mask=0xFFFFFFFF, mask_and_clamp=31)
    o_tmp[2] = cutlass.Uint32(_shfl_xor_50)
    _shfl_xor_51 = cute.arch.shuffle_sync_bfly(o_vec[2], 1, mask=0xFFFFFFFF, mask_and_clamp=31)
    o_tmp[3] = cutlass.Uint32(_shfl_xor_51)
    o_vec[0] = cutlass.Uint32((o_tmp[0] if (qj1 != 0) else o_vec[0]))
    o_vec[1] = cutlass.Uint32((o_tmp[1] if (qj1 == 0) else o_vec[1]))
    o_vec[2] = cutlass.Uint32((o_tmp[2] if (qj1 != 0) else o_vec[2]))
    o_vec[3] = cutlass.Uint32((o_tmp[3] if (qj1 == 0) else o_vec[3]))
    _shfl_xor_52 = cute.arch.shuffle_sync_bfly(o_vec[2], 2, mask=0xFFFFFFFF, mask_and_clamp=31)
    o_tmp[0] = cutlass.Uint32(_shfl_xor_52)
    _shfl_xor_53 = cute.arch.shuffle_sync_bfly(o_vec[3], 2, mask=0xFFFFFFFF, mask_and_clamp=31)
    o_tmp[1] = cutlass.Uint32(_shfl_xor_53)
    _shfl_xor_54 = cute.arch.shuffle_sync_bfly(o_vec[0], 2, mask=0xFFFFFFFF, mask_and_clamp=31)
    o_tmp[2] = cutlass.Uint32(_shfl_xor_54)
    _shfl_xor_55 = cute.arch.shuffle_sync_bfly(o_vec[1], 2, mask=0xFFFFFFFF, mask_and_clamp=31)
    o_tmp[3] = cutlass.Uint32(_shfl_xor_55)
    o_vec[0] = cutlass.Uint32((o_tmp[0] if (qj2 != 0) else o_vec[0]))
    o_vec[1] = cutlass.Uint32((o_tmp[1] if (qj2 != 0) else o_vec[1]))
    o_vec[2] = cutlass.Uint32((o_tmp[2] if (qj2 == 0) else o_vec[2]))
    o_vec[3] = cutlass.Uint32((o_tmp[3] if (qj2 == 0) else o_vec[3]))
    o_off_5 = cutlass.Int32(((o_row_base + (m_local_r_3 * 128)) + ((4 + qj) * 8)))
    _gmem_store_raw_17 = cutlass.Vector.from_elements([cutlass.Uint32(o_vec[0]), cutlass.Uint32(o_vec[1]), cutlass.Uint32(o_vec[2]), cutlass.Uint32(o_vec[3])], cutlass.Uint32)
    prims.store_ext(_gmem_store_raw_17.ir_value(), O + o_off_5)
    _bf16x2_40 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_o[34]), cutlass.Float32(d_o[35])))[1]), cutlass.Float32(((cutlass.Float32(d_o[34]), cutlass.Float32(d_o[35])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
    o_vec[0] = cutlass.Uint32(_bf16x2_40)
    _bf16x2_41 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_o[38]), cutlass.Float32(d_o[39])))[1]), cutlass.Float32(((cutlass.Float32(d_o[38]), cutlass.Float32(d_o[39])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
    o_vec[1] = cutlass.Uint32(_bf16x2_41)
    _bf16x2_42 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_o[42]), cutlass.Float32(d_o[43])))[1]), cutlass.Float32(((cutlass.Float32(d_o[42]), cutlass.Float32(d_o[43])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
    o_vec[2] = cutlass.Uint32(_bf16x2_42)
    _bf16x2_43 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_o[46]), cutlass.Float32(d_o[47])))[1]), cutlass.Float32(((cutlass.Float32(d_o[46]), cutlass.Float32(d_o[47])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
    o_vec[3] = cutlass.Uint32(_bf16x2_43)
    _shfl_xor_56 = cute.arch.shuffle_sync_bfly(o_vec[1], 1, mask=0xFFFFFFFF, mask_and_clamp=31)
    o_tmp[0] = cutlass.Uint32(_shfl_xor_56)
    _shfl_xor_57 = cute.arch.shuffle_sync_bfly(o_vec[0], 1, mask=0xFFFFFFFF, mask_and_clamp=31)
    o_tmp[1] = cutlass.Uint32(_shfl_xor_57)
    _shfl_xor_58 = cute.arch.shuffle_sync_bfly(o_vec[3], 1, mask=0xFFFFFFFF, mask_and_clamp=31)
    o_tmp[2] = cutlass.Uint32(_shfl_xor_58)
    _shfl_xor_59 = cute.arch.shuffle_sync_bfly(o_vec[2], 1, mask=0xFFFFFFFF, mask_and_clamp=31)
    o_tmp[3] = cutlass.Uint32(_shfl_xor_59)
    o_vec[0] = cutlass.Uint32((o_tmp[0] if (qj1 != 0) else o_vec[0]))
    o_vec[1] = cutlass.Uint32((o_tmp[1] if (qj1 == 0) else o_vec[1]))
    o_vec[2] = cutlass.Uint32((o_tmp[2] if (qj1 != 0) else o_vec[2]))
    o_vec[3] = cutlass.Uint32((o_tmp[3] if (qj1 == 0) else o_vec[3]))
    _shfl_xor_60 = cute.arch.shuffle_sync_bfly(o_vec[2], 2, mask=0xFFFFFFFF, mask_and_clamp=31)
    o_tmp[0] = cutlass.Uint32(_shfl_xor_60)
    _shfl_xor_61 = cute.arch.shuffle_sync_bfly(o_vec[3], 2, mask=0xFFFFFFFF, mask_and_clamp=31)
    o_tmp[1] = cutlass.Uint32(_shfl_xor_61)
    _shfl_xor_62 = cute.arch.shuffle_sync_bfly(o_vec[0], 2, mask=0xFFFFFFFF, mask_and_clamp=31)
    o_tmp[2] = cutlass.Uint32(_shfl_xor_62)
    _shfl_xor_63 = cute.arch.shuffle_sync_bfly(o_vec[1], 2, mask=0xFFFFFFFF, mask_and_clamp=31)
    o_tmp[3] = cutlass.Uint32(_shfl_xor_63)
    o_vec[0] = cutlass.Uint32((o_tmp[0] if (qj2 != 0) else o_vec[0]))
    o_vec[1] = cutlass.Uint32((o_tmp[1] if (qj2 != 0) else o_vec[1]))
    o_vec[2] = cutlass.Uint32((o_tmp[2] if (qj2 == 0) else o_vec[2]))
    o_vec[3] = cutlass.Uint32((o_tmp[3] if (qj2 == 0) else o_vec[3]))
    o_off_6 = cutlass.Int32(((o_row_base + (m_local_r_3 * 128)) + ((8 + qj) * 8)))
    _gmem_store_raw_18 = cutlass.Vector.from_elements([cutlass.Uint32(o_vec[0]), cutlass.Uint32(o_vec[1]), cutlass.Uint32(o_vec[2]), cutlass.Uint32(o_vec[3])], cutlass.Uint32)
    prims.store_ext(_gmem_store_raw_18.ir_value(), O + o_off_6)
    _bf16x2_44 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_o[50]), cutlass.Float32(d_o[51])))[1]), cutlass.Float32(((cutlass.Float32(d_o[50]), cutlass.Float32(d_o[51])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
    o_vec[0] = cutlass.Uint32(_bf16x2_44)
    _bf16x2_45 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_o[54]), cutlass.Float32(d_o[55])))[1]), cutlass.Float32(((cutlass.Float32(d_o[54]), cutlass.Float32(d_o[55])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
    o_vec[1] = cutlass.Uint32(_bf16x2_45)
    _bf16x2_46 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_o[58]), cutlass.Float32(d_o[59])))[1]), cutlass.Float32(((cutlass.Float32(d_o[58]), cutlass.Float32(d_o[59])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
    o_vec[2] = cutlass.Uint32(_bf16x2_46)
    _bf16x2_47 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_o[62]), cutlass.Float32(d_o[63])))[1]), cutlass.Float32(((cutlass.Float32(d_o[62]), cutlass.Float32(d_o[63])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
    o_vec[3] = cutlass.Uint32(_bf16x2_47)
    _shfl_xor_64 = cute.arch.shuffle_sync_bfly(o_vec[1], 1, mask=0xFFFFFFFF, mask_and_clamp=31)
    o_tmp[0] = cutlass.Uint32(_shfl_xor_64)
    _shfl_xor_65 = cute.arch.shuffle_sync_bfly(o_vec[0], 1, mask=0xFFFFFFFF, mask_and_clamp=31)
    o_tmp[1] = cutlass.Uint32(_shfl_xor_65)
    _shfl_xor_66 = cute.arch.shuffle_sync_bfly(o_vec[3], 1, mask=0xFFFFFFFF, mask_and_clamp=31)
    o_tmp[2] = cutlass.Uint32(_shfl_xor_66)
    _shfl_xor_67 = cute.arch.shuffle_sync_bfly(o_vec[2], 1, mask=0xFFFFFFFF, mask_and_clamp=31)
    o_tmp[3] = cutlass.Uint32(_shfl_xor_67)
    o_vec[0] = cutlass.Uint32((o_tmp[0] if (qj1 != 0) else o_vec[0]))
    o_vec[1] = cutlass.Uint32((o_tmp[1] if (qj1 == 0) else o_vec[1]))
    o_vec[2] = cutlass.Uint32((o_tmp[2] if (qj1 != 0) else o_vec[2]))
    o_vec[3] = cutlass.Uint32((o_tmp[3] if (qj1 == 0) else o_vec[3]))
    _shfl_xor_68 = cute.arch.shuffle_sync_bfly(o_vec[2], 2, mask=0xFFFFFFFF, mask_and_clamp=31)
    o_tmp[0] = cutlass.Uint32(_shfl_xor_68)
    _shfl_xor_69 = cute.arch.shuffle_sync_bfly(o_vec[3], 2, mask=0xFFFFFFFF, mask_and_clamp=31)
    o_tmp[1] = cutlass.Uint32(_shfl_xor_69)
    _shfl_xor_70 = cute.arch.shuffle_sync_bfly(o_vec[0], 2, mask=0xFFFFFFFF, mask_and_clamp=31)
    o_tmp[2] = cutlass.Uint32(_shfl_xor_70)
    _shfl_xor_71 = cute.arch.shuffle_sync_bfly(o_vec[1], 2, mask=0xFFFFFFFF, mask_and_clamp=31)
    o_tmp[3] = cutlass.Uint32(_shfl_xor_71)
    o_vec[0] = cutlass.Uint32((o_tmp[0] if (qj2 != 0) else o_vec[0]))
    o_vec[1] = cutlass.Uint32((o_tmp[1] if (qj2 != 0) else o_vec[1]))
    o_vec[2] = cutlass.Uint32((o_tmp[2] if (qj2 == 0) else o_vec[2]))
    o_vec[3] = cutlass.Uint32((o_tmp[3] if (qj2 == 0) else o_vec[3]))
    o_off_7 = cutlass.Int32(((o_row_base + (m_local_r_3 * 128)) + ((12 + qj) * 8)))
    _gmem_store_raw_19 = cutlass.Vector.from_elements([cutlass.Uint32(o_vec[0]), cutlass.Uint32(o_vec[1]), cutlass.Uint32(o_vec[2]), cutlass.Uint32(o_vec[3])], cutlass.Uint32)
    prims.store_ext(_gmem_store_raw_19.ir_value(), O + o_off_7)
    cute.arch.sync_threads()

@cute.jit
def launch_vsa_sm90_bf16_small_k1(Q: cute.Tensor, _cake_tma_Q_dim_0: cutlass.Int64, _cake_tma_Q_dim_1: cutlass.Int64, _cake_tma_Q_dim_2: cutlass.Int64, _cake_tma_Q_stride16_0: cutlass.Int64, _cake_tma_Q_stride16_1: cutlass.Int64, K: cute.Tensor, _cake_tma_K_dim_0: cutlass.Int64, _cake_tma_K_dim_1: cutlass.Int64, _cake_tma_K_dim_2: cutlass.Int64, _cake_tma_K_stride16_0: cutlass.Int64, _cake_tma_K_stride16_1: cutlass.Int64, Vt: cute.Tensor, _cake_tma_Vt_dim_0: cutlass.Int64, _cake_tma_Vt_dim_1: cutlass.Int64, _cake_tma_Vt_dim_2: cutlass.Int64, _cake_tma_Vt_dim_3: cutlass.Int64, _cake_tma_Vt_stride16_0: cutlass.Int64, _cake_tma_Vt_stride16_1: cutlass.Int64, _cake_tma_Vt_stride16_2: cutlass.Int64, O: cute.Tensor, plan: cute.Tensor, seqlen_q: cutlass.Int32, seqlen_k: cutlass.Int32, scale_log2: cutlass.Float32, grid_x: cutlass.Int32, grid_y: cutlass.Int32, grid_z: cutlass.Int32, stream: cuda.CUstream):
    _tma_Q = create_tensor_map_tiled(
        Q.iterator.toint(),
        cutlass.BFloat16,
        [_cake_tma_Q_dim_0, _cake_tma_Q_dim_1, _cake_tma_Q_dim_2],
        [_cake_tma_Q_stride16_0, _cake_tma_Q_stride16_1],
        [64, 64, 2],
        swizzle=TensorMapSwizzle.s128b,
        l2_promotion=TensorMapL2Promotion.none,
        oob_fill=TensorMapFloatOOBFill.none,
    )
    _tma_K = create_tensor_map_tiled(
        K.iterator.toint(),
        cutlass.BFloat16,
        [_cake_tma_K_dim_0, _cake_tma_K_dim_1, _cake_tma_K_dim_2],
        [_cake_tma_K_stride16_0, _cake_tma_K_stride16_1],
        [64, 64, 2],
        swizzle=TensorMapSwizzle.s128b,
        l2_promotion=TensorMapL2Promotion.none,
        oob_fill=TensorMapFloatOOBFill.none,
    )
    _tma_Vt = create_tensor_map_tiled(
        Vt.iterator.toint(),
        cutlass.BFloat16,
        [_cake_tma_Vt_dim_0, _cake_tma_Vt_dim_1, _cake_tma_Vt_dim_2, _cake_tma_Vt_dim_3],
        [_cake_tma_Vt_stride16_0, _cake_tma_Vt_stride16_1, _cake_tma_Vt_stride16_2],
        [64, 8, 8, 2],
        swizzle=TensorMapSwizzle.s128b,
        l2_promotion=TensorMapL2Promotion.none,
        oob_fill=TensorMapFloatOOBFill.none,
    )
    kernel_vsa_sm90_bf16_small_k1(_tma_Q, _tma_K, _tma_Vt, O.iterator, plan.iterator, seqlen_q, seqlen_k, scale_log2).launch(
        grid=(grid_x, grid_y, grid_z),
        block=(128, 1, 1),
        smem=50176,
        min_blocks_per_mp=4,
        stream=stream,
    )

def compile_program():
    return cute.compile(launch_vsa_sm90_bf16_small_k1,
        make_fake_tensor(cutlass.BFloat16, (cute.sym_int64(symbol='Q'),), (1,), assumed_align=16),
        cutlass.Int64(0),
        cutlass.Int64(0),
        cutlass.Int64(0),
        cutlass.Int64(0),
        cutlass.Int64(0),
        make_fake_tensor(cutlass.BFloat16, (cute.sym_int64(symbol='K'),), (1,), assumed_align=16),
        cutlass.Int64(0),
        cutlass.Int64(0),
        cutlass.Int64(0),
        cutlass.Int64(0),
        cutlass.Int64(0),
        make_fake_tensor(cutlass.BFloat16, (cute.sym_int64(symbol='Vt'),), (1,), assumed_align=16),
        cutlass.Int64(0),
        cutlass.Int64(0),
        cutlass.Int64(0),
        cutlass.Int64(0),
        cutlass.Int64(0),
        cutlass.Int64(0),
        cutlass.Int64(0),
        make_fake_tensor(cutlass.BFloat16, (cute.sym_int64(symbol='O'),), (1,), assumed_align=16),
        make_fake_tensor(cutlass.Int16, (cute.sym_int64(symbol='plan'),), (1,), assumed_align=2),
        cutlass.Int32(0),
        cutlass.Int32(0),
        cutlass.Float32(0),
        cutlass.Int32(1),
        cutlass.Int32(1),
        cutlass.Int32(1),
        make_fake_stream(use_tvm_ffi_env_stream=True),
        options='--enable-tvm-ffi --ptxas-options=--opt-level=2 --gpu-arch=sm_90a',
    )
