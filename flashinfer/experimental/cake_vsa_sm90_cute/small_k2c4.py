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
SMEM_MSTATS_OFF = 82944
SMEM_MSTATS_STAGE_BYTES = 1024
SMEM_MSTATS_STRIDE = 1024
SMEM_PART_LO_OFF = 33792
SMEM_PART_LO_STAGE_BYTES = 16384
SMEM_PART_LO_STRIDE = 16384
SMEM_PART_HI_OFF = 66560
SMEM_PART_HI_STAGE_BYTES = 16384
SMEM_PART_HI_STRIDE = 16384
SMEM_PART_HI0_OFF = 17408
SMEM_PART_HI0_STAGE_BYTES = 16384
SMEM_PART_HI0_STRIDE = 16384
SMEM_RECV_SMEM_OFF = 83968
SMEM_RECV_SMEM_STAGE_BYTES = 27648
SMEM_RECV_SMEM_STRIDE = 27648
SMEM_Q_SMEM_OFF = 1024
SMEM_Q_SMEM_STAGE_BYTES = 16384
SMEM_Q_SMEM_STRIDE = 16384
SMEM_K_SMEM_OFF = 17408
SMEM_K_SMEM_STAGE_BYTES = 16384
SMEM_K_SMEM_STRIDE = 16384
SMEM_VT_SMEM_OFF = 50176
SMEM_VT_SMEM_STAGE_BYTES = 16384
SMEM_VT_SMEM_STRIDE = 16384
SMEM_TOTAL = 111616
THREADS = 256
CAKE_TARGET_ARCH = 'sm_90a'
CAKE_SMEM_BYTES = 111616

def _cake_launch_cluster_spread(kernel_launcher, **kwargs):
    block = cutlass_ir.InsertionPoint.current.block
    first_new_op = len(block.operations)
    kernel_launcher.launch(**kwargs)
    launches = [op for op in list(block.operations)[first_new_op:] if op.operation.name == 'cuda.launch_ex']
    if len(launches) != 1:
        raise RuntimeError('expected exactly one typed CUDA launch for cluster scheduling')
    launch = launches[0]
    with cutlass_ir.InsertionPoint(launch):
        cutlass_cuda.launch_cfg_cluster_scheduling_policy(
            launch.operands[0],
            cutlass_cuda.CudaClusterSchedulingPolicy.cudaClusterSchedulingPolicySpread,
        )

@cute.kernel
def kernel_vsa_sm90_bf16_small_k2c4(Q: cutlass.GridConstant[TensorMap], K: cutlass.GridConstant[TensorMap], Vt: cutlass.GridConstant[TensorMap], O: cute.Pointer, plan: cute.Pointer, seqlen_q: cutlass.Int32, seqlen_k: cutlass.Int32, scale_log2: cutlass.Float32):
    tid = cutlass.Int32(cute.arch.thread_idx()[0])
    warp = cutlass.Int32(cute.arch.warp_idx())
    lane = cutlass.Int32(cute.arch.lane_idx())
    bid = cutlass.Int32(cute.arch.block_idx()[0])
    num_bids = cutlass.Int32(cute.arch.grid_dim()[0])
    blockIdx = cute.arch.block_idx()
    gridDim = cute.arch.grid_dim()
    clusterIdx = cute.arch.cluster_idx()
    clusterDim = cute.arch.cluster_dim()
    cluster_id = ((clusterIdx[2] * clusterDim[1] + clusterIdx[1]) * clusterDim[0]) + clusterIdx[0]
    num_clusters = clusterDim[0] * clusterDim[1] * clusterDim[2]
    cta_rank = cute.arch.block_idx_in_cluster()
    smem_raw = cute.arch.get_dyn_smem(cutlass.Uint8, alignment=1024)
    smem = smem_raw.toint()
    _flat_layout = cute.make_layout((2147483647,), stride=(1,))
    _O = cute.make_tensor(O, _flat_layout)
    mstats = cute.recast_ptr(smem_raw + 82944, swizzle_=cute.make_swizzle(4, 3, 3), dtype=cutlass.Float32)
    _mstats = cute.make_tensor(mstats, _flat_layout)
    mstats_addr = smem + 82944
    part_lo = cute.recast_ptr(smem_raw + 33792, swizzle_=cute.make_swizzle(4, 3, 3), dtype=cutlass.Float32)
    _part_lo = cute.make_tensor(part_lo, _flat_layout)
    part_lo_addr = smem + 33792
    part_hi = cute.recast_ptr(smem_raw + 66560, swizzle_=cute.make_swizzle(4, 3, 3), dtype=cutlass.Float32)
    _part_hi = cute.make_tensor(part_hi, _flat_layout)
    part_hi_addr = smem + 66560
    part_hi0 = cute.recast_ptr(smem_raw + 17408, swizzle_=cute.make_swizzle(4, 3, 3), dtype=cutlass.Float32)
    _part_hi0 = cute.make_tensor(part_hi0, _flat_layout)
    part_hi0_addr = smem + 17408
    recv_smem = cute.recast_ptr(smem_raw + 83968, swizzle_=cute.make_swizzle(4, 3, 3), dtype=cutlass.Float32)
    _recv_smem = cute.make_tensor(recv_smem, _flat_layout)
    recv_smem_addr = smem + 83968
    q_smem = cute.recast_ptr(smem_raw + 1024, swizzle_=cute.make_swizzle(4, 3, 3), dtype=cutlass.BFloat16)
    _q_smem = cute.make_tensor(q_smem, _flat_layout)
    q_smem_addr = smem + 1024
    k_smem = cute.recast_ptr(smem_raw + 17408, swizzle_=cute.make_swizzle(4, 3, 3), dtype=cutlass.BFloat16)
    _k_smem = cute.make_tensor(k_smem, _flat_layout)
    k_smem_addr = smem + 17408
    vt_smem = cute.recast_ptr(smem_raw + 50176, swizzle_=cute.make_swizzle(4, 3, 3), dtype=cutlass.BFloat16)
    _vt_smem = cute.make_tensor(vt_smem, _flat_layout)
    vt_smem_addr = smem + 50176
    q_full_addr = cute.recast_ptr(smem_raw, dtype=cutlass.Uint64)
    k_full0_addr = cute.recast_ptr(smem_raw + 8, dtype=cutlass.Uint64)
    k_full1_addr = cute.recast_ptr(smem_raw + 16, dtype=cutlass.Uint64)
    v_full0_addr = cute.recast_ptr(smem_raw + 24, dtype=cutlass.Uint64)
    v_full1_addr = cute.recast_ptr(smem_raw + 32, dtype=cutlass.Uint64)
    recv_full_addr = cute.recast_ptr(smem_raw + 40, dtype=cutlass.Uint64)
    if warp == 0:
        with cute.arch.elect_one():
            cute.arch.mbarrier_init(q_full_addr + 0, 1)
            cute.arch.mbarrier_init(k_full0_addr + 0, 1)
            cute.arch.mbarrier_init(k_full1_addr + 0, 1)
            cute.arch.mbarrier_init(v_full0_addr + 0, 1)
            cute.arch.mbarrier_init(v_full1_addr + 0, 1)
            cute.arch.mbarrier_init(recv_full_addr + 0, 1)
    cute.arch.mbarrier_init_fence()
    cute.arch.sync_threads()
    cute.arch.cluster_arrive(aligned=True)
    # barrier.cluster.wait deferred (WarpConfig.cluster_init_wait_warps=()): the schedule's ClusterSyncWait
    lim_even = cute.make_rmem_tensor((1,), cutlass.Int32)
    lim_odd = cute.make_rmem_tensor((1,), cutlass.Int32)
    row_max0 = cute.make_rmem_tensor((1,), cutlass.Float32)
    row_max1 = cute.make_rmem_tensor((1,), cutlass.Float32)
    row_sum0 = cute.make_rmem_tensor((1,), cutlass.Float32)
    row_sum1 = cute.make_rmem_tensor((1,), cutlass.Float32)
    _phase_q_full_0 = cute.make_rmem_tensor((1,), cutlass.Uint32)
    _phase_k_full0_0 = cute.make_rmem_tensor((1,), cutlass.Uint32)
    _phase_v_full0_0 = cute.make_rmem_tensor((1,), cutlass.Uint32)
    new_max0 = cute.make_rmem_tensor((1,), cutlass.Float32)
    new_max1 = cute.make_rmem_tensor((1,), cutlass.Float32)
    _phase_k_full1_0 = cute.make_rmem_tensor((1,), cutlass.Uint32)
    _phase_v_full1_0 = cute.make_rmem_tensor((1,), cutlass.Uint32)
    new_max0_1 = cute.make_rmem_tensor((1,), cutlass.Float32)
    new_max1_1 = cute.make_rmem_tensor((1,), cutlass.Float32)
    _phase_recv_full_0 = cute.make_rmem_tensor((1,), cutlass.Uint32)
    pslot = cute.make_rmem_tensor((1,), cutlass.Int32)
    pslot_1 = cute.make_rmem_tensor((1,), cutlass.Int32)
    pslot_2 = cute.make_rmem_tensor((1,), cutlass.Int32)
    pslot_3 = cute.make_rmem_tensor((1,), cutlass.Int32)
    merged_max0 = cute.make_rmem_tensor((1,), cutlass.Float32)
    merged_max1 = cute.make_rmem_tensor((1,), cutlass.Float32)
    msum0 = cute.make_rmem_tensor((1,), cutlass.Float32)
    msum1 = cute.make_rmem_tensor((1,), cutlass.Float32)
    fold = cute.make_rmem_tensor((1,), cutlass.Int32)
    pslot_4 = cute.make_rmem_tensor((1,), cutlass.Int32)
    pslot_5 = cute.make_rmem_tensor((1,), cutlass.Int32)
    pslot_6 = cute.make_rmem_tensor((1,), cutlass.Int32)
    pslot_7 = cute.make_rmem_tensor((1,), cutlass.Int32)
    merged_max0_1 = cute.make_rmem_tensor((1,), cutlass.Float32)
    merged_max1_1 = cute.make_rmem_tensor((1,), cutlass.Float32)
    msum0_1 = cute.make_rmem_tensor((1,), cutlass.Float32)
    msum1_1 = cute.make_rmem_tensor((1,), cutlass.Float32)
    fold_1 = cute.make_rmem_tensor((1,), cutlass.Int32)
    cute.arch.prefetch(Q.get_ptr(), tensormap=True, predicate=cutlass.Boolean((warp == 0)))
    cute.arch.prefetch(K.get_ptr(), tensormap=True, predicate=cutlass.Boolean((warp == 0)))
    cute.arch.prefetch(Vt.get_ptr(), tensormap=True, predicate=cutlass.Boolean((warp == 0)))
    item = cutlass.Int32(bid)
    mb = cutlass.Int32(cutlass.Int32(cutlass_arith.divsi(cutlass.Int32(seqlen_q).ir_value(), cutlass.Int32(64).ir_value())))
    plan_base = cutlass.Int32((item * 4))
    meta = cutlass.Int32(cute.make_tensor(plan, _flat_layout)[plan_base])
    cnt = cutlass.Int32((meta & 15))
    tile = cutlass.Int32(cute.make_tensor(plan, _flat_layout)[(plan_base + 1)])
    blk_base = cutlass.Int32((plan_base + 2))
    head = cutlass.Int32(cutlass.Int32(cutlass_arith.divsi(cutlass.Int32(tile).ir_value(), cutlass.Int32(mb).ir_value())))
    qb = cutlass.Int32((tile - (head * mb)))
    q_row = cutlass.Int32(((head * seqlen_q) + (qb * 64)))
    kv_base = cutlass.Int32((head * seqlen_k))
    if (warp == 0):
        if prims.elect_sync():
            if (cta_rank < 4):
                nsets = 1
                recv_tx = (3 * ((nsets * 8192) + 512))
                cute.arch.mbarrier_arrive_and_expect_tx(recv_full_addr, recv_tx)
            blk_pre = cute.make_rmem_tensor((2,), cutlass.Int32)
            blk_pre[0] = cutlass.Int32(cute.make_tensor(plan, _flat_layout)[blk_base])
            blk_pre[1] = cutlass.Int32(cute.make_tensor(plan, _flat_layout)[(blk_base + 1)])
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
            if (cnt > 1):
                blk_row_1 = cutlass.Int32((kv_base + (blk_pre[1] * 64)))
                cute.arch.mbarrier_arrive_and_expect_tx(k_full1_addr, 16384)
                prims.cp_async_bulk_tensor_shared_cta_global(
                    cute.make_ptr(cutlass.Uint8, cutlass.Uint32((k_smem_addr + 16384)), mem_space=cute.AddressSpace.smem, assumed_align=16),
                    K.get_ptr(),
                    [cutlass.Int32(0), cutlass.Int32(blk_row_1), cutlass.Int32(0)],
                    k_full1_addr,
                    mode=prims.TMALoadMode.TILE,
                )
                cute.arch.mbarrier_arrive_and_expect_tx(v_full1_addr, 16384)
                prims.cp_async_bulk_tensor_shared_cta_global(
                    cute.make_ptr(cutlass.Uint8, cutlass.Uint32((vt_smem_addr + 16384)), mem_space=cute.AddressSpace.smem, assumed_align=16),
                    Vt.get_ptr(),
                    [cutlass.Int32(0), cutlass.Int32(0), cutlass.Int32(cutlass.Int32(cutlass_arith.divsi(cutlass.Int32(blk_row_1).ir_value(), cutlass.Int32(8).ir_value()))), cutlass.Int32(0)],
                    v_full1_addr,
                    mode=prims.TMALoadMode.TILE,
                )
    warp_raw = cutlass.Int32(warp)
    _shfl_0 = cute.arch.shuffle_sync(warp_raw, 0, mask=4294967295, mask_and_clamp=31)
    warp_u = cutlass.Int32(_shfl_0)
    wg = cutlass.Int32(cutlass.Int32(cutlass_arith.divsi(cutlass.Int32(warp_u).ir_value(), cutlass.Int32(4).ir_value())))
    warp_in_wg = cutlass.Int32((warp_u - (wg * 4)))
    tid_wg = cutlass.Int32(((warp_in_wg * 32) + lane))
    quad = cutlass.Int32(((warp_in_wg * 8) + cutlass.Int32(cutlass_arith.divsi(cutlass.Int32(lane).ir_value(), cutlass.Int32(4).ir_value()))))
    lim_even[0] = cutlass.Int32(0)
    lim_odd[0] = cutlass.Int32(0)
    if (wg == 0):
        lim_even[0] = cutlass.Int32(cnt)
    else:
        lim_odd[0] = cutlass.Int32(cnt)
    m0_local = cutlass.Int32(((warp_in_wg * 16) + cutlass.Int32(cutlass_arith.divsi(cutlass.Int32(lane).ir_value(), cutlass.Int32(4).ir_value()))))
    m1_local = cutlass.Int32((m0_local + 8))
    d_o = cute.make_rmem_tensor((64,), cutlass.Float32)
    d_qk = cute.make_rmem_tensor((32,), cutlass.Float32)
    p_bf16 = cute.make_rmem_tensor((16,), cutlass.Uint32)
    row_max0[0] = cutlass.Float32((0 - float("inf")))
    row_max1[0] = cutlass.Float32((0 - float("inf")))
    row_sum0[0] = cutlass.Float32(0.0)
    row_sum1[0] = cutlass.Float32(0.0)
    if (cnt <= wg):
        d_o[0] = cutlass.Float32(0.0)
        d_o[1] = cutlass.Float32(0.0)
        d_o[2] = cutlass.Float32(0.0)
        d_o[3] = cutlass.Float32(0.0)
        d_o[4] = cutlass.Float32(0.0)
        d_o[5] = cutlass.Float32(0.0)
        d_o[6] = cutlass.Float32(0.0)
        d_o[7] = cutlass.Float32(0.0)
        d_o[8] = cutlass.Float32(0.0)
        d_o[9] = cutlass.Float32(0.0)
        d_o[10] = cutlass.Float32(0.0)
        d_o[11] = cutlass.Float32(0.0)
        d_o[12] = cutlass.Float32(0.0)
        d_o[13] = cutlass.Float32(0.0)
        d_o[14] = cutlass.Float32(0.0)
        d_o[15] = cutlass.Float32(0.0)
        d_o[16] = cutlass.Float32(0.0)
        d_o[17] = cutlass.Float32(0.0)
        d_o[18] = cutlass.Float32(0.0)
        d_o[19] = cutlass.Float32(0.0)
        d_o[20] = cutlass.Float32(0.0)
        d_o[21] = cutlass.Float32(0.0)
        d_o[22] = cutlass.Float32(0.0)
        d_o[23] = cutlass.Float32(0.0)
        d_o[24] = cutlass.Float32(0.0)
        d_o[25] = cutlass.Float32(0.0)
        d_o[26] = cutlass.Float32(0.0)
        d_o[27] = cutlass.Float32(0.0)
        d_o[28] = cutlass.Float32(0.0)
        d_o[29] = cutlass.Float32(0.0)
        d_o[30] = cutlass.Float32(0.0)
        d_o[31] = cutlass.Float32(0.0)
        d_o[32] = cutlass.Float32(0.0)
        d_o[33] = cutlass.Float32(0.0)
        d_o[34] = cutlass.Float32(0.0)
        d_o[35] = cutlass.Float32(0.0)
        d_o[36] = cutlass.Float32(0.0)
        d_o[37] = cutlass.Float32(0.0)
        d_o[38] = cutlass.Float32(0.0)
        d_o[39] = cutlass.Float32(0.0)
        d_o[40] = cutlass.Float32(0.0)
        d_o[41] = cutlass.Float32(0.0)
        d_o[42] = cutlass.Float32(0.0)
        d_o[43] = cutlass.Float32(0.0)
        d_o[44] = cutlass.Float32(0.0)
        d_o[45] = cutlass.Float32(0.0)
        d_o[46] = cutlass.Float32(0.0)
        d_o[47] = cutlass.Float32(0.0)
        d_o[48] = cutlass.Float32(0.0)
        d_o[49] = cutlass.Float32(0.0)
        d_o[50] = cutlass.Float32(0.0)
        d_o[51] = cutlass.Float32(0.0)
        d_o[52] = cutlass.Float32(0.0)
        d_o[53] = cutlass.Float32(0.0)
        d_o[54] = cutlass.Float32(0.0)
        d_o[55] = cutlass.Float32(0.0)
        d_o[56] = cutlass.Float32(0.0)
        d_o[57] = cutlass.Float32(0.0)
        d_o[58] = cutlass.Float32(0.0)
        d_o[59] = cutlass.Float32(0.0)
        d_o[60] = cutlass.Float32(0.0)
        d_o[61] = cutlass.Float32(0.0)
        d_o[62] = cutlass.Float32(0.0)
        d_o[63] = cutlass.Float32(0.0)
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
    if (lim_even[0] > 0):
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
    _phase_k_full1_0[0] = cutlass.Uint32(0)
    _wgmma_b_0_5_raw = ((cutlass.Uint64(cutlass.Uint32((k_smem_addr + 16384)) >> 4) & cutlass.Uint64(0x3FFF)) | (cutlass.Uint64(0) << 16) | (cutlass.Uint64(64) << 32) | (cutlass.Uint64(1) << 62))
    _wgmma_b_0_5 = (cutlass.Uint64(cute.arch.make_warp_uniform(cutlass.Uint32(_wgmma_b_0_5_raw >> 32))) << 32) | cutlass.Uint64(cute.arch.make_warp_uniform(cutlass.Uint32(_wgmma_b_0_5_raw)))
    _wgmma_b_0_6_raw = ((cutlass.Uint64(cutlass.Uint32(((k_smem_addr + 16384) + 8192)) >> 4) & cutlass.Uint64(0x3FFF)) | (cutlass.Uint64(0) << 16) | (cutlass.Uint64(64) << 32) | (cutlass.Uint64(1) << 62))
    _wgmma_b_0_6 = (cutlass.Uint64(cute.arch.make_warp_uniform(cutlass.Uint32(_wgmma_b_0_6_raw >> 32))) << 32) | cutlass.Uint64(cute.arch.make_warp_uniform(cutlass.Uint32(_wgmma_b_0_6_raw)))
    _phase_v_full1_0[0] = cutlass.Uint32(0)
    _wgmma_b_0_7_raw = ((cutlass.Uint64(cutlass.Uint32((vt_smem_addr + 16384)) >> 4) & cutlass.Uint64(0x3FFF)) | (cutlass.Uint64(512) << 16) | (cutlass.Uint64(64) << 32) | (cutlass.Uint64(1) << 62))
    _wgmma_b_0_7 = (cutlass.Uint64(cute.arch.make_warp_uniform(cutlass.Uint32(_wgmma_b_0_7_raw >> 32))) << 32) | cutlass.Uint64(cute.arch.make_warp_uniform(cutlass.Uint32(_wgmma_b_0_7_raw)))
    if (lim_odd[0] > 1):
        while not prims.mbarrier_wait_parity(k_full1_addr, _phase_k_full1_0[0], prims.MBarrierWait.TRY, scope=prims.MBarrierScope.CTA, order=prims.MemOrder.ACQUIRE):
            pass
        _phase_k_full1_0[0] ^= cutlass.Uint32(1)
        cute.nvgpu.warpgroup.fence()
        _wgmma_12 = cutlass_hopper.make_trivial_tiled_mma(
            cutlass.BFloat16, cutlass.BFloat16,
            cute.nvgpu.OperandMajorMode.K, cute.nvgpu.OperandMajorMode.K,
            cutlass.Float32, (1, 1, 1), (64, 64),
            cute.nvgpu.warpgroup.OperandSource.SMEM,
        )
        _wgmma_12_thr = _wgmma_12.get_slice(tid)
        _wgmma_12_b_ptr = cute.make_ptr(cutlass.BFloat16, cutlass.Uint32((cutlass.Uint64(_wgmma_b_0_5) & cutlass.Uint64(0x3FFF)) << 4), mem_space=cute.AddressSpace.smem, assumed_align=16)
        _wgmma_12_b_layout = cutlass_hopper.make_smem_layout_b(CutlassLayout.ROW_MAJOR, (64, 64, 64), cutlass.BFloat16, 1)
        _wgmma_12_b_ptr = cute.recast_ptr(_wgmma_12_b_ptr, swizzle_=_wgmma_12_b_layout.inner, dtype=cutlass.BFloat16)
        _wgmma_12_b = cute.make_tensor(_wgmma_12_b_ptr, _wgmma_12_b_layout.outer)
        _wgmma_12_b_part = _wgmma_12_thr.partition_B(_wgmma_12_b)
        _wgmma_12_b_frag = _wgmma_12.make_fragment_B(_wgmma_12_b_part)
        _wgmma_12_a_ptr = cute.make_ptr(cutlass.BFloat16, cutlass.Uint32((cutlass.Uint64(_wgmma_a_0_0) & cutlass.Uint64(0x3FFF)) << 4), mem_space=cute.AddressSpace.smem, assumed_align=16)
        _wgmma_12_a_layout = cutlass_hopper.make_smem_layout_a(CutlassLayout.ROW_MAJOR, (64, 64, 64), cutlass.BFloat16, 1)
        _wgmma_12_a_ptr = cute.recast_ptr(_wgmma_12_a_ptr, swizzle_=_wgmma_12_a_layout.inner, dtype=cutlass.BFloat16)
        _wgmma_12_a = cute.make_tensor(_wgmma_12_a_ptr, _wgmma_12_a_layout.outer)
        _wgmma_12_a_part = _wgmma_12_thr.partition_A(_wgmma_12_a)
        _wgmma_12_a_frag = _wgmma_12.make_fragment_A(_wgmma_12_a_part)
        _wgmma_12_d = cute.make_tensor(d_qk.iterator + (0), _wgmma_12.partition_shape_C((64, 64)))
        _wgmma_12.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, False)
        cute.gemm(_wgmma_12, _wgmma_12_d, _wgmma_12_a_frag[None, None, 0, 0], _wgmma_12_b_frag[None, None, 0, 0], _wgmma_12_d)
        _wgmma_13 = cutlass_hopper.make_trivial_tiled_mma(
            cutlass.BFloat16, cutlass.BFloat16,
            cute.nvgpu.OperandMajorMode.K, cute.nvgpu.OperandMajorMode.K,
            cutlass.Float32, (1, 1, 1), (64, 64),
            cute.nvgpu.warpgroup.OperandSource.SMEM,
        )
        _wgmma_13_thr = _wgmma_13.get_slice(tid)
        _wgmma_13_b_ptr = cute.make_ptr(cutlass.BFloat16, cutlass.Uint32((cutlass.Uint64((_wgmma_b_0_5 + 2)) & cutlass.Uint64(0x3FFF)) << 4), mem_space=cute.AddressSpace.smem, assumed_align=16)
        _wgmma_13_b_layout = cutlass_hopper.make_smem_layout_b(CutlassLayout.ROW_MAJOR, (64, 64, 64), cutlass.BFloat16, 1)
        _wgmma_13_b_ptr = cute.recast_ptr(_wgmma_13_b_ptr, swizzle_=_wgmma_13_b_layout.inner, dtype=cutlass.BFloat16)
        _wgmma_13_b = cute.make_tensor(_wgmma_13_b_ptr, _wgmma_13_b_layout.outer)
        _wgmma_13_b_part = _wgmma_13_thr.partition_B(_wgmma_13_b)
        _wgmma_13_b_frag = _wgmma_13.make_fragment_B(_wgmma_13_b_part)
        _wgmma_13_a_ptr = cute.make_ptr(cutlass.BFloat16, cutlass.Uint32((cutlass.Uint64((_wgmma_a_0_0 + 2)) & cutlass.Uint64(0x3FFF)) << 4), mem_space=cute.AddressSpace.smem, assumed_align=16)
        _wgmma_13_a_layout = cutlass_hopper.make_smem_layout_a(CutlassLayout.ROW_MAJOR, (64, 64, 64), cutlass.BFloat16, 1)
        _wgmma_13_a_ptr = cute.recast_ptr(_wgmma_13_a_ptr, swizzle_=_wgmma_13_a_layout.inner, dtype=cutlass.BFloat16)
        _wgmma_13_a = cute.make_tensor(_wgmma_13_a_ptr, _wgmma_13_a_layout.outer)
        _wgmma_13_a_part = _wgmma_13_thr.partition_A(_wgmma_13_a)
        _wgmma_13_a_frag = _wgmma_13.make_fragment_A(_wgmma_13_a_part)
        _wgmma_13_d = cute.make_tensor(d_qk.iterator + (0), _wgmma_13.partition_shape_C((64, 64)))
        _wgmma_13.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, True)
        cute.gemm(_wgmma_13, _wgmma_13_d, _wgmma_13_a_frag[None, None, 0, 0], _wgmma_13_b_frag[None, None, 0, 0], _wgmma_13_d)
        _wgmma_14 = cutlass_hopper.make_trivial_tiled_mma(
            cutlass.BFloat16, cutlass.BFloat16,
            cute.nvgpu.OperandMajorMode.K, cute.nvgpu.OperandMajorMode.K,
            cutlass.Float32, (1, 1, 1), (64, 64),
            cute.nvgpu.warpgroup.OperandSource.SMEM,
        )
        _wgmma_14_thr = _wgmma_14.get_slice(tid)
        _wgmma_14_b_ptr = cute.make_ptr(cutlass.BFloat16, cutlass.Uint32((cutlass.Uint64((_wgmma_b_0_5 + 4)) & cutlass.Uint64(0x3FFF)) << 4), mem_space=cute.AddressSpace.smem, assumed_align=16)
        _wgmma_14_b_layout = cutlass_hopper.make_smem_layout_b(CutlassLayout.ROW_MAJOR, (64, 64, 64), cutlass.BFloat16, 1)
        _wgmma_14_b_ptr = cute.recast_ptr(_wgmma_14_b_ptr, swizzle_=_wgmma_14_b_layout.inner, dtype=cutlass.BFloat16)
        _wgmma_14_b = cute.make_tensor(_wgmma_14_b_ptr, _wgmma_14_b_layout.outer)
        _wgmma_14_b_part = _wgmma_14_thr.partition_B(_wgmma_14_b)
        _wgmma_14_b_frag = _wgmma_14.make_fragment_B(_wgmma_14_b_part)
        _wgmma_14_a_ptr = cute.make_ptr(cutlass.BFloat16, cutlass.Uint32((cutlass.Uint64((_wgmma_a_0_0 + 4)) & cutlass.Uint64(0x3FFF)) << 4), mem_space=cute.AddressSpace.smem, assumed_align=16)
        _wgmma_14_a_layout = cutlass_hopper.make_smem_layout_a(CutlassLayout.ROW_MAJOR, (64, 64, 64), cutlass.BFloat16, 1)
        _wgmma_14_a_ptr = cute.recast_ptr(_wgmma_14_a_ptr, swizzle_=_wgmma_14_a_layout.inner, dtype=cutlass.BFloat16)
        _wgmma_14_a = cute.make_tensor(_wgmma_14_a_ptr, _wgmma_14_a_layout.outer)
        _wgmma_14_a_part = _wgmma_14_thr.partition_A(_wgmma_14_a)
        _wgmma_14_a_frag = _wgmma_14.make_fragment_A(_wgmma_14_a_part)
        _wgmma_14_d = cute.make_tensor(d_qk.iterator + (0), _wgmma_14.partition_shape_C((64, 64)))
        _wgmma_14.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, True)
        cute.gemm(_wgmma_14, _wgmma_14_d, _wgmma_14_a_frag[None, None, 0, 0], _wgmma_14_b_frag[None, None, 0, 0], _wgmma_14_d)
        _wgmma_15 = cutlass_hopper.make_trivial_tiled_mma(
            cutlass.BFloat16, cutlass.BFloat16,
            cute.nvgpu.OperandMajorMode.K, cute.nvgpu.OperandMajorMode.K,
            cutlass.Float32, (1, 1, 1), (64, 64),
            cute.nvgpu.warpgroup.OperandSource.SMEM,
        )
        _wgmma_15_thr = _wgmma_15.get_slice(tid)
        _wgmma_15_b_ptr = cute.make_ptr(cutlass.BFloat16, cutlass.Uint32((cutlass.Uint64((_wgmma_b_0_5 + 6)) & cutlass.Uint64(0x3FFF)) << 4), mem_space=cute.AddressSpace.smem, assumed_align=16)
        _wgmma_15_b_layout = cutlass_hopper.make_smem_layout_b(CutlassLayout.ROW_MAJOR, (64, 64, 64), cutlass.BFloat16, 1)
        _wgmma_15_b_ptr = cute.recast_ptr(_wgmma_15_b_ptr, swizzle_=_wgmma_15_b_layout.inner, dtype=cutlass.BFloat16)
        _wgmma_15_b = cute.make_tensor(_wgmma_15_b_ptr, _wgmma_15_b_layout.outer)
        _wgmma_15_b_part = _wgmma_15_thr.partition_B(_wgmma_15_b)
        _wgmma_15_b_frag = _wgmma_15.make_fragment_B(_wgmma_15_b_part)
        _wgmma_15_a_ptr = cute.make_ptr(cutlass.BFloat16, cutlass.Uint32((cutlass.Uint64((_wgmma_a_0_0 + 6)) & cutlass.Uint64(0x3FFF)) << 4), mem_space=cute.AddressSpace.smem, assumed_align=16)
        _wgmma_15_a_layout = cutlass_hopper.make_smem_layout_a(CutlassLayout.ROW_MAJOR, (64, 64, 64), cutlass.BFloat16, 1)
        _wgmma_15_a_ptr = cute.recast_ptr(_wgmma_15_a_ptr, swizzle_=_wgmma_15_a_layout.inner, dtype=cutlass.BFloat16)
        _wgmma_15_a = cute.make_tensor(_wgmma_15_a_ptr, _wgmma_15_a_layout.outer)
        _wgmma_15_a_part = _wgmma_15_thr.partition_A(_wgmma_15_a)
        _wgmma_15_a_frag = _wgmma_15.make_fragment_A(_wgmma_15_a_part)
        _wgmma_15_d = cute.make_tensor(d_qk.iterator + (0), _wgmma_15.partition_shape_C((64, 64)))
        _wgmma_15.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, True)
        cute.gemm(_wgmma_15, _wgmma_15_d, _wgmma_15_a_frag[None, None, 0, 0], _wgmma_15_b_frag[None, None, 0, 0], _wgmma_15_d)
        _wgmma_16 = cutlass_hopper.make_trivial_tiled_mma(
            cutlass.BFloat16, cutlass.BFloat16,
            cute.nvgpu.OperandMajorMode.K, cute.nvgpu.OperandMajorMode.K,
            cutlass.Float32, (1, 1, 1), (64, 64),
            cute.nvgpu.warpgroup.OperandSource.SMEM,
        )
        _wgmma_16_thr = _wgmma_16.get_slice(tid)
        _wgmma_16_b_ptr = cute.make_ptr(cutlass.BFloat16, cutlass.Uint32((cutlass.Uint64(_wgmma_b_0_6) & cutlass.Uint64(0x3FFF)) << 4), mem_space=cute.AddressSpace.smem, assumed_align=16)
        _wgmma_16_b_layout = cutlass_hopper.make_smem_layout_b(CutlassLayout.ROW_MAJOR, (64, 64, 64), cutlass.BFloat16, 1)
        _wgmma_16_b_ptr = cute.recast_ptr(_wgmma_16_b_ptr, swizzle_=_wgmma_16_b_layout.inner, dtype=cutlass.BFloat16)
        _wgmma_16_b = cute.make_tensor(_wgmma_16_b_ptr, _wgmma_16_b_layout.outer)
        _wgmma_16_b_part = _wgmma_16_thr.partition_B(_wgmma_16_b)
        _wgmma_16_b_frag = _wgmma_16.make_fragment_B(_wgmma_16_b_part)
        _wgmma_16_a_ptr = cute.make_ptr(cutlass.BFloat16, cutlass.Uint32((cutlass.Uint64(_wgmma_a_0_2) & cutlass.Uint64(0x3FFF)) << 4), mem_space=cute.AddressSpace.smem, assumed_align=16)
        _wgmma_16_a_layout = cutlass_hopper.make_smem_layout_a(CutlassLayout.ROW_MAJOR, (64, 64, 64), cutlass.BFloat16, 1)
        _wgmma_16_a_ptr = cute.recast_ptr(_wgmma_16_a_ptr, swizzle_=_wgmma_16_a_layout.inner, dtype=cutlass.BFloat16)
        _wgmma_16_a = cute.make_tensor(_wgmma_16_a_ptr, _wgmma_16_a_layout.outer)
        _wgmma_16_a_part = _wgmma_16_thr.partition_A(_wgmma_16_a)
        _wgmma_16_a_frag = _wgmma_16.make_fragment_A(_wgmma_16_a_part)
        _wgmma_16_d = cute.make_tensor(d_qk.iterator + (0), _wgmma_16.partition_shape_C((64, 64)))
        _wgmma_16.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, True)
        cute.gemm(_wgmma_16, _wgmma_16_d, _wgmma_16_a_frag[None, None, 0, 0], _wgmma_16_b_frag[None, None, 0, 0], _wgmma_16_d)
        _wgmma_17 = cutlass_hopper.make_trivial_tiled_mma(
            cutlass.BFloat16, cutlass.BFloat16,
            cute.nvgpu.OperandMajorMode.K, cute.nvgpu.OperandMajorMode.K,
            cutlass.Float32, (1, 1, 1), (64, 64),
            cute.nvgpu.warpgroup.OperandSource.SMEM,
        )
        _wgmma_17_thr = _wgmma_17.get_slice(tid)
        _wgmma_17_b_ptr = cute.make_ptr(cutlass.BFloat16, cutlass.Uint32((cutlass.Uint64((_wgmma_b_0_6 + 2)) & cutlass.Uint64(0x3FFF)) << 4), mem_space=cute.AddressSpace.smem, assumed_align=16)
        _wgmma_17_b_layout = cutlass_hopper.make_smem_layout_b(CutlassLayout.ROW_MAJOR, (64, 64, 64), cutlass.BFloat16, 1)
        _wgmma_17_b_ptr = cute.recast_ptr(_wgmma_17_b_ptr, swizzle_=_wgmma_17_b_layout.inner, dtype=cutlass.BFloat16)
        _wgmma_17_b = cute.make_tensor(_wgmma_17_b_ptr, _wgmma_17_b_layout.outer)
        _wgmma_17_b_part = _wgmma_17_thr.partition_B(_wgmma_17_b)
        _wgmma_17_b_frag = _wgmma_17.make_fragment_B(_wgmma_17_b_part)
        _wgmma_17_a_ptr = cute.make_ptr(cutlass.BFloat16, cutlass.Uint32((cutlass.Uint64((_wgmma_a_0_2 + 2)) & cutlass.Uint64(0x3FFF)) << 4), mem_space=cute.AddressSpace.smem, assumed_align=16)
        _wgmma_17_a_layout = cutlass_hopper.make_smem_layout_a(CutlassLayout.ROW_MAJOR, (64, 64, 64), cutlass.BFloat16, 1)
        _wgmma_17_a_ptr = cute.recast_ptr(_wgmma_17_a_ptr, swizzle_=_wgmma_17_a_layout.inner, dtype=cutlass.BFloat16)
        _wgmma_17_a = cute.make_tensor(_wgmma_17_a_ptr, _wgmma_17_a_layout.outer)
        _wgmma_17_a_part = _wgmma_17_thr.partition_A(_wgmma_17_a)
        _wgmma_17_a_frag = _wgmma_17.make_fragment_A(_wgmma_17_a_part)
        _wgmma_17_d = cute.make_tensor(d_qk.iterator + (0), _wgmma_17.partition_shape_C((64, 64)))
        _wgmma_17.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, True)
        cute.gemm(_wgmma_17, _wgmma_17_d, _wgmma_17_a_frag[None, None, 0, 0], _wgmma_17_b_frag[None, None, 0, 0], _wgmma_17_d)
        _wgmma_18 = cutlass_hopper.make_trivial_tiled_mma(
            cutlass.BFloat16, cutlass.BFloat16,
            cute.nvgpu.OperandMajorMode.K, cute.nvgpu.OperandMajorMode.K,
            cutlass.Float32, (1, 1, 1), (64, 64),
            cute.nvgpu.warpgroup.OperandSource.SMEM,
        )
        _wgmma_18_thr = _wgmma_18.get_slice(tid)
        _wgmma_18_b_ptr = cute.make_ptr(cutlass.BFloat16, cutlass.Uint32((cutlass.Uint64((_wgmma_b_0_6 + 4)) & cutlass.Uint64(0x3FFF)) << 4), mem_space=cute.AddressSpace.smem, assumed_align=16)
        _wgmma_18_b_layout = cutlass_hopper.make_smem_layout_b(CutlassLayout.ROW_MAJOR, (64, 64, 64), cutlass.BFloat16, 1)
        _wgmma_18_b_ptr = cute.recast_ptr(_wgmma_18_b_ptr, swizzle_=_wgmma_18_b_layout.inner, dtype=cutlass.BFloat16)
        _wgmma_18_b = cute.make_tensor(_wgmma_18_b_ptr, _wgmma_18_b_layout.outer)
        _wgmma_18_b_part = _wgmma_18_thr.partition_B(_wgmma_18_b)
        _wgmma_18_b_frag = _wgmma_18.make_fragment_B(_wgmma_18_b_part)
        _wgmma_18_a_ptr = cute.make_ptr(cutlass.BFloat16, cutlass.Uint32((cutlass.Uint64((_wgmma_a_0_2 + 4)) & cutlass.Uint64(0x3FFF)) << 4), mem_space=cute.AddressSpace.smem, assumed_align=16)
        _wgmma_18_a_layout = cutlass_hopper.make_smem_layout_a(CutlassLayout.ROW_MAJOR, (64, 64, 64), cutlass.BFloat16, 1)
        _wgmma_18_a_ptr = cute.recast_ptr(_wgmma_18_a_ptr, swizzle_=_wgmma_18_a_layout.inner, dtype=cutlass.BFloat16)
        _wgmma_18_a = cute.make_tensor(_wgmma_18_a_ptr, _wgmma_18_a_layout.outer)
        _wgmma_18_a_part = _wgmma_18_thr.partition_A(_wgmma_18_a)
        _wgmma_18_a_frag = _wgmma_18.make_fragment_A(_wgmma_18_a_part)
        _wgmma_18_d = cute.make_tensor(d_qk.iterator + (0), _wgmma_18.partition_shape_C((64, 64)))
        _wgmma_18.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, True)
        cute.gemm(_wgmma_18, _wgmma_18_d, _wgmma_18_a_frag[None, None, 0, 0], _wgmma_18_b_frag[None, None, 0, 0], _wgmma_18_d)
        _wgmma_19 = cutlass_hopper.make_trivial_tiled_mma(
            cutlass.BFloat16, cutlass.BFloat16,
            cute.nvgpu.OperandMajorMode.K, cute.nvgpu.OperandMajorMode.K,
            cutlass.Float32, (1, 1, 1), (64, 64),
            cute.nvgpu.warpgroup.OperandSource.SMEM,
        )
        _wgmma_19_thr = _wgmma_19.get_slice(tid)
        _wgmma_19_b_ptr = cute.make_ptr(cutlass.BFloat16, cutlass.Uint32((cutlass.Uint64((_wgmma_b_0_6 + 6)) & cutlass.Uint64(0x3FFF)) << 4), mem_space=cute.AddressSpace.smem, assumed_align=16)
        _wgmma_19_b_layout = cutlass_hopper.make_smem_layout_b(CutlassLayout.ROW_MAJOR, (64, 64, 64), cutlass.BFloat16, 1)
        _wgmma_19_b_ptr = cute.recast_ptr(_wgmma_19_b_ptr, swizzle_=_wgmma_19_b_layout.inner, dtype=cutlass.BFloat16)
        _wgmma_19_b = cute.make_tensor(_wgmma_19_b_ptr, _wgmma_19_b_layout.outer)
        _wgmma_19_b_part = _wgmma_19_thr.partition_B(_wgmma_19_b)
        _wgmma_19_b_frag = _wgmma_19.make_fragment_B(_wgmma_19_b_part)
        _wgmma_19_a_ptr = cute.make_ptr(cutlass.BFloat16, cutlass.Uint32((cutlass.Uint64((_wgmma_a_0_2 + 6)) & cutlass.Uint64(0x3FFF)) << 4), mem_space=cute.AddressSpace.smem, assumed_align=16)
        _wgmma_19_a_layout = cutlass_hopper.make_smem_layout_a(CutlassLayout.ROW_MAJOR, (64, 64, 64), cutlass.BFloat16, 1)
        _wgmma_19_a_ptr = cute.recast_ptr(_wgmma_19_a_ptr, swizzle_=_wgmma_19_a_layout.inner, dtype=cutlass.BFloat16)
        _wgmma_19_a = cute.make_tensor(_wgmma_19_a_ptr, _wgmma_19_a_layout.outer)
        _wgmma_19_a_part = _wgmma_19_thr.partition_A(_wgmma_19_a)
        _wgmma_19_a_frag = _wgmma_19.make_fragment_A(_wgmma_19_a_part)
        _wgmma_19_d = cute.make_tensor(d_qk.iterator + (0), _wgmma_19.partition_shape_C((64, 64)))
        _wgmma_19.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, True)
        cute.gemm(_wgmma_19, _wgmma_19_d, _wgmma_19_a_frag[None, None, 0, 0], _wgmma_19_b_frag[None, None, 0, 0], _wgmma_19_d)
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
        new_max0_1[0] = cutlass.Float32((0 - float("inf")))
        new_max1_1[0] = cutlass.Float32((0 - float("inf")))
        _max_38 = cute.arch.fmax(new_max0_1[0], d_qk[0], ftz=False)
        new_max0_1[0] = cutlass.Float32(_max_38)
        _max_39 = cute.arch.fmax(new_max0_1[0], d_qk[1], ftz=False)
        new_max0_1[0] = cutlass.Float32(_max_39)
        _max_40 = cute.arch.fmax(new_max0_1[0], d_qk[4], ftz=False)
        new_max0_1[0] = cutlass.Float32(_max_40)
        _max_41 = cute.arch.fmax(new_max0_1[0], d_qk[5], ftz=False)
        new_max0_1[0] = cutlass.Float32(_max_41)
        _max_42 = cute.arch.fmax(new_max0_1[0], d_qk[8], ftz=False)
        new_max0_1[0] = cutlass.Float32(_max_42)
        _max_43 = cute.arch.fmax(new_max0_1[0], d_qk[9], ftz=False)
        new_max0_1[0] = cutlass.Float32(_max_43)
        _max_44 = cute.arch.fmax(new_max0_1[0], d_qk[12], ftz=False)
        new_max0_1[0] = cutlass.Float32(_max_44)
        _max_45 = cute.arch.fmax(new_max0_1[0], d_qk[13], ftz=False)
        new_max0_1[0] = cutlass.Float32(_max_45)
        _max_46 = cute.arch.fmax(new_max0_1[0], d_qk[16], ftz=False)
        new_max0_1[0] = cutlass.Float32(_max_46)
        _max_47 = cute.arch.fmax(new_max0_1[0], d_qk[17], ftz=False)
        new_max0_1[0] = cutlass.Float32(_max_47)
        _max_48 = cute.arch.fmax(new_max0_1[0], d_qk[20], ftz=False)
        new_max0_1[0] = cutlass.Float32(_max_48)
        _max_49 = cute.arch.fmax(new_max0_1[0], d_qk[21], ftz=False)
        new_max0_1[0] = cutlass.Float32(_max_49)
        _max_50 = cute.arch.fmax(new_max0_1[0], d_qk[24], ftz=False)
        new_max0_1[0] = cutlass.Float32(_max_50)
        _max_51 = cute.arch.fmax(new_max0_1[0], d_qk[25], ftz=False)
        new_max0_1[0] = cutlass.Float32(_max_51)
        _max_52 = cute.arch.fmax(new_max0_1[0], d_qk[28], ftz=False)
        new_max0_1[0] = cutlass.Float32(_max_52)
        _max_53 = cute.arch.fmax(new_max0_1[0], d_qk[29], ftz=False)
        new_max0_1[0] = cutlass.Float32(_max_53)
        _max_54 = cute.arch.fmax(new_max1_1[0], d_qk[2], ftz=False)
        new_max1_1[0] = cutlass.Float32(_max_54)
        _max_55 = cute.arch.fmax(new_max1_1[0], d_qk[3], ftz=False)
        new_max1_1[0] = cutlass.Float32(_max_55)
        _max_56 = cute.arch.fmax(new_max1_1[0], d_qk[6], ftz=False)
        new_max1_1[0] = cutlass.Float32(_max_56)
        _max_57 = cute.arch.fmax(new_max1_1[0], d_qk[7], ftz=False)
        new_max1_1[0] = cutlass.Float32(_max_57)
        _max_58 = cute.arch.fmax(new_max1_1[0], d_qk[10], ftz=False)
        new_max1_1[0] = cutlass.Float32(_max_58)
        _max_59 = cute.arch.fmax(new_max1_1[0], d_qk[11], ftz=False)
        new_max1_1[0] = cutlass.Float32(_max_59)
        _max_60 = cute.arch.fmax(new_max1_1[0], d_qk[14], ftz=False)
        new_max1_1[0] = cutlass.Float32(_max_60)
        _max_61 = cute.arch.fmax(new_max1_1[0], d_qk[15], ftz=False)
        new_max1_1[0] = cutlass.Float32(_max_61)
        _max_62 = cute.arch.fmax(new_max1_1[0], d_qk[18], ftz=False)
        new_max1_1[0] = cutlass.Float32(_max_62)
        _max_63 = cute.arch.fmax(new_max1_1[0], d_qk[19], ftz=False)
        new_max1_1[0] = cutlass.Float32(_max_63)
        _max_64 = cute.arch.fmax(new_max1_1[0], d_qk[22], ftz=False)
        new_max1_1[0] = cutlass.Float32(_max_64)
        _max_65 = cute.arch.fmax(new_max1_1[0], d_qk[23], ftz=False)
        new_max1_1[0] = cutlass.Float32(_max_65)
        _max_66 = cute.arch.fmax(new_max1_1[0], d_qk[26], ftz=False)
        new_max1_1[0] = cutlass.Float32(_max_66)
        _max_67 = cute.arch.fmax(new_max1_1[0], d_qk[27], ftz=False)
        new_max1_1[0] = cutlass.Float32(_max_67)
        _max_68 = cute.arch.fmax(new_max1_1[0], d_qk[30], ftz=False)
        new_max1_1[0] = cutlass.Float32(_max_68)
        _max_69 = cute.arch.fmax(new_max1_1[0], d_qk[31], ftz=False)
        new_max1_1[0] = cutlass.Float32(_max_69)
        _shfl_xor_4 = cute.arch.shuffle_sync_bfly(new_max0_1[0], 2, mask=0xFFFFFFFF, mask_and_clamp=31)
        _max_70 = cute.arch.fmax(new_max0_1[0], _shfl_xor_4, ftz=False)
        new_max0_1[0] = cutlass.Float32(_max_70)
        _shfl_xor_5 = cute.arch.shuffle_sync_bfly(new_max0_1[0], 1, mask=0xFFFFFFFF, mask_and_clamp=31)
        _max_71 = cute.arch.fmax(new_max0_1[0], _shfl_xor_5, ftz=False)
        new_max0_1[0] = cutlass.Float32(_max_71)
        _shfl_xor_6 = cute.arch.shuffle_sync_bfly(new_max1_1[0], 2, mask=0xFFFFFFFF, mask_and_clamp=31)
        _max_72 = cute.arch.fmax(new_max1_1[0], _shfl_xor_6, ftz=False)
        new_max1_1[0] = cutlass.Float32(_max_72)
        _shfl_xor_7 = cute.arch.shuffle_sync_bfly(new_max1_1[0], 1, mask=0xFFFFFFFF, mask_and_clamp=31)
        _max_73 = cute.arch.fmax(new_max1_1[0], _shfl_xor_7, ftz=False)
        new_max1_1[0] = cutlass.Float32(_max_73)
        _max_74 = cute.arch.fmax(row_max0[0], new_max0_1[0], ftz=False)
        merged_max0_2 = cutlass.Float32(_max_74)
        _max_75 = cute.arch.fmax(row_max1[0], new_max1_1[0], ftz=False)
        merged_max1_2 = cutlass.Float32(_max_75)
        _exp2_34 = cute.math.exp2((row_max0[0] - merged_max0_2), approx=True, ftz=True)
        _exp2_35 = cute.math.exp2((row_max1[0] - merged_max1_2), approx=True, ftz=True)
        d_o[0] = cutlass.Float32((d_o[0] * _exp2_34))
        d_o[1] = cutlass.Float32((d_o[1] * _exp2_34))
        d_o[4] = cutlass.Float32((d_o[4] * _exp2_34))
        d_o[5] = cutlass.Float32((d_o[5] * _exp2_34))
        d_o[8] = cutlass.Float32((d_o[8] * _exp2_34))
        d_o[9] = cutlass.Float32((d_o[9] * _exp2_34))
        d_o[12] = cutlass.Float32((d_o[12] * _exp2_34))
        d_o[13] = cutlass.Float32((d_o[13] * _exp2_34))
        d_o[16] = cutlass.Float32((d_o[16] * _exp2_34))
        d_o[17] = cutlass.Float32((d_o[17] * _exp2_34))
        d_o[20] = cutlass.Float32((d_o[20] * _exp2_34))
        d_o[21] = cutlass.Float32((d_o[21] * _exp2_34))
        d_o[24] = cutlass.Float32((d_o[24] * _exp2_34))
        d_o[25] = cutlass.Float32((d_o[25] * _exp2_34))
        d_o[28] = cutlass.Float32((d_o[28] * _exp2_34))
        d_o[29] = cutlass.Float32((d_o[29] * _exp2_34))
        d_o[32] = cutlass.Float32((d_o[32] * _exp2_34))
        d_o[33] = cutlass.Float32((d_o[33] * _exp2_34))
        d_o[36] = cutlass.Float32((d_o[36] * _exp2_34))
        d_o[37] = cutlass.Float32((d_o[37] * _exp2_34))
        d_o[40] = cutlass.Float32((d_o[40] * _exp2_34))
        d_o[41] = cutlass.Float32((d_o[41] * _exp2_34))
        d_o[44] = cutlass.Float32((d_o[44] * _exp2_34))
        d_o[45] = cutlass.Float32((d_o[45] * _exp2_34))
        d_o[48] = cutlass.Float32((d_o[48] * _exp2_34))
        d_o[49] = cutlass.Float32((d_o[49] * _exp2_34))
        d_o[52] = cutlass.Float32((d_o[52] * _exp2_34))
        d_o[53] = cutlass.Float32((d_o[53] * _exp2_34))
        d_o[56] = cutlass.Float32((d_o[56] * _exp2_34))
        d_o[57] = cutlass.Float32((d_o[57] * _exp2_34))
        d_o[60] = cutlass.Float32((d_o[60] * _exp2_34))
        d_o[61] = cutlass.Float32((d_o[61] * _exp2_34))
        d_o[2] = cutlass.Float32((d_o[2] * _exp2_35))
        d_o[3] = cutlass.Float32((d_o[3] * _exp2_35))
        d_o[6] = cutlass.Float32((d_o[6] * _exp2_35))
        d_o[7] = cutlass.Float32((d_o[7] * _exp2_35))
        d_o[10] = cutlass.Float32((d_o[10] * _exp2_35))
        d_o[11] = cutlass.Float32((d_o[11] * _exp2_35))
        d_o[14] = cutlass.Float32((d_o[14] * _exp2_35))
        d_o[15] = cutlass.Float32((d_o[15] * _exp2_35))
        d_o[18] = cutlass.Float32((d_o[18] * _exp2_35))
        d_o[19] = cutlass.Float32((d_o[19] * _exp2_35))
        d_o[22] = cutlass.Float32((d_o[22] * _exp2_35))
        d_o[23] = cutlass.Float32((d_o[23] * _exp2_35))
        d_o[26] = cutlass.Float32((d_o[26] * _exp2_35))
        d_o[27] = cutlass.Float32((d_o[27] * _exp2_35))
        d_o[30] = cutlass.Float32((d_o[30] * _exp2_35))
        d_o[31] = cutlass.Float32((d_o[31] * _exp2_35))
        d_o[34] = cutlass.Float32((d_o[34] * _exp2_35))
        d_o[35] = cutlass.Float32((d_o[35] * _exp2_35))
        d_o[38] = cutlass.Float32((d_o[38] * _exp2_35))
        d_o[39] = cutlass.Float32((d_o[39] * _exp2_35))
        d_o[42] = cutlass.Float32((d_o[42] * _exp2_35))
        d_o[43] = cutlass.Float32((d_o[43] * _exp2_35))
        d_o[46] = cutlass.Float32((d_o[46] * _exp2_35))
        d_o[47] = cutlass.Float32((d_o[47] * _exp2_35))
        d_o[50] = cutlass.Float32((d_o[50] * _exp2_35))
        d_o[51] = cutlass.Float32((d_o[51] * _exp2_35))
        d_o[54] = cutlass.Float32((d_o[54] * _exp2_35))
        d_o[55] = cutlass.Float32((d_o[55] * _exp2_35))
        d_o[58] = cutlass.Float32((d_o[58] * _exp2_35))
        d_o[59] = cutlass.Float32((d_o[59] * _exp2_35))
        d_o[62] = cutlass.Float32((d_o[62] * _exp2_35))
        d_o[63] = cutlass.Float32((d_o[63] * _exp2_35))
        row_sum0[0] = cutlass.Float32((row_sum0[0] * _exp2_34))
        row_sum1[0] = cutlass.Float32((row_sum1[0] * _exp2_35))
        row_max0[0] = cutlass.Float32(merged_max0_2)
        row_max1[0] = cutlass.Float32(merged_max1_2)
        _exp2_36 = cute.math.exp2((d_qk[0] - row_max0[0]), approx=True, ftz=True)
        _exp2_37 = cute.math.exp2((d_qk[1] - row_max0[0]), approx=True, ftz=True)
        d_qk[0] = cutlass.Float32(_exp2_36)
        d_qk[1] = cutlass.Float32(_exp2_37)
        row_sum0[0] += cutlass.Float32((_exp2_36 + _exp2_37))
        _exp2_38 = cute.math.exp2((d_qk[2] - row_max1[0]), approx=True, ftz=True)
        _exp2_39 = cute.math.exp2((d_qk[3] - row_max1[0]), approx=True, ftz=True)
        d_qk[2] = cutlass.Float32(_exp2_38)
        d_qk[3] = cutlass.Float32(_exp2_39)
        row_sum1[0] += cutlass.Float32((_exp2_38 + _exp2_39))
        _exp2_40 = cute.math.exp2((d_qk[4] - row_max0[0]), approx=True, ftz=True)
        _exp2_41 = cute.math.exp2((d_qk[5] - row_max0[0]), approx=True, ftz=True)
        d_qk[4] = cutlass.Float32(_exp2_40)
        d_qk[5] = cutlass.Float32(_exp2_41)
        row_sum0[0] += cutlass.Float32((_exp2_40 + _exp2_41))
        _exp2_42 = cute.math.exp2((d_qk[6] - row_max1[0]), approx=True, ftz=True)
        _exp2_43 = cute.math.exp2((d_qk[7] - row_max1[0]), approx=True, ftz=True)
        d_qk[6] = cutlass.Float32(_exp2_42)
        d_qk[7] = cutlass.Float32(_exp2_43)
        row_sum1[0] += cutlass.Float32((_exp2_42 + _exp2_43))
        _exp2_44 = cute.math.exp2((d_qk[8] - row_max0[0]), approx=True, ftz=True)
        _exp2_45 = cute.math.exp2((d_qk[9] - row_max0[0]), approx=True, ftz=True)
        d_qk[8] = cutlass.Float32(_exp2_44)
        d_qk[9] = cutlass.Float32(_exp2_45)
        row_sum0[0] += cutlass.Float32((_exp2_44 + _exp2_45))
        _exp2_46 = cute.math.exp2((d_qk[10] - row_max1[0]), approx=True, ftz=True)
        _exp2_47 = cute.math.exp2((d_qk[11] - row_max1[0]), approx=True, ftz=True)
        d_qk[10] = cutlass.Float32(_exp2_46)
        d_qk[11] = cutlass.Float32(_exp2_47)
        row_sum1[0] += cutlass.Float32((_exp2_46 + _exp2_47))
        _exp2_48 = cute.math.exp2((d_qk[12] - row_max0[0]), approx=True, ftz=True)
        _exp2_49 = cute.math.exp2((d_qk[13] - row_max0[0]), approx=True, ftz=True)
        d_qk[12] = cutlass.Float32(_exp2_48)
        d_qk[13] = cutlass.Float32(_exp2_49)
        row_sum0[0] += cutlass.Float32((_exp2_48 + _exp2_49))
        _exp2_50 = cute.math.exp2((d_qk[14] - row_max1[0]), approx=True, ftz=True)
        _exp2_51 = cute.math.exp2((d_qk[15] - row_max1[0]), approx=True, ftz=True)
        d_qk[14] = cutlass.Float32(_exp2_50)
        d_qk[15] = cutlass.Float32(_exp2_51)
        row_sum1[0] += cutlass.Float32((_exp2_50 + _exp2_51))
        _exp2_52 = cute.math.exp2((d_qk[16] - row_max0[0]), approx=True, ftz=True)
        _exp2_53 = cute.math.exp2((d_qk[17] - row_max0[0]), approx=True, ftz=True)
        d_qk[16] = cutlass.Float32(_exp2_52)
        d_qk[17] = cutlass.Float32(_exp2_53)
        row_sum0[0] += cutlass.Float32((_exp2_52 + _exp2_53))
        _exp2_54 = cute.math.exp2((d_qk[18] - row_max1[0]), approx=True, ftz=True)
        _exp2_55 = cute.math.exp2((d_qk[19] - row_max1[0]), approx=True, ftz=True)
        d_qk[18] = cutlass.Float32(_exp2_54)
        d_qk[19] = cutlass.Float32(_exp2_55)
        row_sum1[0] += cutlass.Float32((_exp2_54 + _exp2_55))
        _exp2_56 = cute.math.exp2((d_qk[20] - row_max0[0]), approx=True, ftz=True)
        _exp2_57 = cute.math.exp2((d_qk[21] - row_max0[0]), approx=True, ftz=True)
        d_qk[20] = cutlass.Float32(_exp2_56)
        d_qk[21] = cutlass.Float32(_exp2_57)
        row_sum0[0] += cutlass.Float32((_exp2_56 + _exp2_57))
        _exp2_58 = cute.math.exp2((d_qk[22] - row_max1[0]), approx=True, ftz=True)
        _exp2_59 = cute.math.exp2((d_qk[23] - row_max1[0]), approx=True, ftz=True)
        d_qk[22] = cutlass.Float32(_exp2_58)
        d_qk[23] = cutlass.Float32(_exp2_59)
        row_sum1[0] += cutlass.Float32((_exp2_58 + _exp2_59))
        _exp2_60 = cute.math.exp2((d_qk[24] - row_max0[0]), approx=True, ftz=True)
        _exp2_61 = cute.math.exp2((d_qk[25] - row_max0[0]), approx=True, ftz=True)
        d_qk[24] = cutlass.Float32(_exp2_60)
        d_qk[25] = cutlass.Float32(_exp2_61)
        row_sum0[0] += cutlass.Float32((_exp2_60 + _exp2_61))
        _exp2_62 = cute.math.exp2((d_qk[26] - row_max1[0]), approx=True, ftz=True)
        _exp2_63 = cute.math.exp2((d_qk[27] - row_max1[0]), approx=True, ftz=True)
        d_qk[26] = cutlass.Float32(_exp2_62)
        d_qk[27] = cutlass.Float32(_exp2_63)
        row_sum1[0] += cutlass.Float32((_exp2_62 + _exp2_63))
        _exp2_64 = cute.math.exp2((d_qk[28] - row_max0[0]), approx=True, ftz=True)
        _exp2_65 = cute.math.exp2((d_qk[29] - row_max0[0]), approx=True, ftz=True)
        d_qk[28] = cutlass.Float32(_exp2_64)
        d_qk[29] = cutlass.Float32(_exp2_65)
        row_sum0[0] += cutlass.Float32((_exp2_64 + _exp2_65))
        _exp2_66 = cute.math.exp2((d_qk[30] - row_max1[0]), approx=True, ftz=True)
        _exp2_67 = cute.math.exp2((d_qk[31] - row_max1[0]), approx=True, ftz=True)
        d_qk[30] = cutlass.Float32(_exp2_66)
        d_qk[31] = cutlass.Float32(_exp2_67)
        row_sum1[0] += cutlass.Float32((_exp2_66 + _exp2_67))
        _bf16x2_16 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_qk[0]), cutlass.Float32(d_qk[1])))[1]), cutlass.Float32(((cutlass.Float32(d_qk[0]), cutlass.Float32(d_qk[1])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
        p_bf16[0] = cutlass.Uint32(_bf16x2_16)
        _bf16x2_17 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_qk[2]), cutlass.Float32(d_qk[3])))[1]), cutlass.Float32(((cutlass.Float32(d_qk[2]), cutlass.Float32(d_qk[3])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
        p_bf16[1] = cutlass.Uint32(_bf16x2_17)
        _bf16x2_18 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_qk[4]), cutlass.Float32(d_qk[5])))[1]), cutlass.Float32(((cutlass.Float32(d_qk[4]), cutlass.Float32(d_qk[5])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
        p_bf16[2] = cutlass.Uint32(_bf16x2_18)
        _bf16x2_19 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_qk[6]), cutlass.Float32(d_qk[7])))[1]), cutlass.Float32(((cutlass.Float32(d_qk[6]), cutlass.Float32(d_qk[7])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
        p_bf16[3] = cutlass.Uint32(_bf16x2_19)
        _bf16x2_20 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_qk[8]), cutlass.Float32(d_qk[9])))[1]), cutlass.Float32(((cutlass.Float32(d_qk[8]), cutlass.Float32(d_qk[9])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
        p_bf16[4] = cutlass.Uint32(_bf16x2_20)
        _bf16x2_21 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_qk[10]), cutlass.Float32(d_qk[11])))[1]), cutlass.Float32(((cutlass.Float32(d_qk[10]), cutlass.Float32(d_qk[11])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
        p_bf16[5] = cutlass.Uint32(_bf16x2_21)
        _bf16x2_22 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_qk[12]), cutlass.Float32(d_qk[13])))[1]), cutlass.Float32(((cutlass.Float32(d_qk[12]), cutlass.Float32(d_qk[13])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
        p_bf16[6] = cutlass.Uint32(_bf16x2_22)
        _bf16x2_23 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_qk[14]), cutlass.Float32(d_qk[15])))[1]), cutlass.Float32(((cutlass.Float32(d_qk[14]), cutlass.Float32(d_qk[15])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
        p_bf16[7] = cutlass.Uint32(_bf16x2_23)
        _bf16x2_24 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_qk[16]), cutlass.Float32(d_qk[17])))[1]), cutlass.Float32(((cutlass.Float32(d_qk[16]), cutlass.Float32(d_qk[17])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
        p_bf16[8] = cutlass.Uint32(_bf16x2_24)
        _bf16x2_25 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_qk[18]), cutlass.Float32(d_qk[19])))[1]), cutlass.Float32(((cutlass.Float32(d_qk[18]), cutlass.Float32(d_qk[19])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
        p_bf16[9] = cutlass.Uint32(_bf16x2_25)
        _bf16x2_26 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_qk[20]), cutlass.Float32(d_qk[21])))[1]), cutlass.Float32(((cutlass.Float32(d_qk[20]), cutlass.Float32(d_qk[21])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
        p_bf16[10] = cutlass.Uint32(_bf16x2_26)
        _bf16x2_27 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_qk[22]), cutlass.Float32(d_qk[23])))[1]), cutlass.Float32(((cutlass.Float32(d_qk[22]), cutlass.Float32(d_qk[23])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
        p_bf16[11] = cutlass.Uint32(_bf16x2_27)
        _bf16x2_28 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_qk[24]), cutlass.Float32(d_qk[25])))[1]), cutlass.Float32(((cutlass.Float32(d_qk[24]), cutlass.Float32(d_qk[25])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
        p_bf16[12] = cutlass.Uint32(_bf16x2_28)
        _bf16x2_29 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_qk[26]), cutlass.Float32(d_qk[27])))[1]), cutlass.Float32(((cutlass.Float32(d_qk[26]), cutlass.Float32(d_qk[27])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
        p_bf16[13] = cutlass.Uint32(_bf16x2_29)
        _bf16x2_30 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_qk[28]), cutlass.Float32(d_qk[29])))[1]), cutlass.Float32(((cutlass.Float32(d_qk[28]), cutlass.Float32(d_qk[29])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
        p_bf16[14] = cutlass.Uint32(_bf16x2_30)
        _bf16x2_31 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(d_qk[30]), cutlass.Float32(d_qk[31])))[1]), cutlass.Float32(((cutlass.Float32(d_qk[30]), cutlass.Float32(d_qk[31])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
        p_bf16[15] = cutlass.Uint32(_bf16x2_31)
        while not prims.mbarrier_wait_parity(v_full1_addr, _phase_v_full1_0[0], prims.MBarrierWait.TRY, scope=prims.MBarrierScope.CTA, order=prims.MemOrder.ACQUIRE):
            pass
        _phase_v_full1_0[0] ^= cutlass.Uint32(1)
        cute.nvgpu.warpgroup.fence()
        _wgmma_20 = cutlass_hopper.make_trivial_tiled_mma(
            cutlass.BFloat16, cutlass.BFloat16,
            cute.nvgpu.OperandMajorMode.K, cute.nvgpu.OperandMajorMode.MN,
            cutlass.Float32, (1, 1, 1), (64, 128),
            cute.nvgpu.warpgroup.OperandSource.RMEM,
        )
        _wgmma_20_thr = _wgmma_20.get_slice(tid)
        _wgmma_20_b_ptr = cute.make_ptr(cutlass.BFloat16, cutlass.Uint32((cutlass.Uint64(_wgmma_b_0_7) & cutlass.Uint64(0x3FFF)) << 4), mem_space=cute.AddressSpace.smem, assumed_align=16)
        _wgmma_20_b_layout = cutlass_hopper.make_smem_layout_b(CutlassLayout.COL_MAJOR, (64, 128, 64), cutlass.BFloat16, 1)
        _wgmma_20_b_ptr = cute.recast_ptr(_wgmma_20_b_ptr, swizzle_=_wgmma_20_b_layout.inner, dtype=cutlass.BFloat16)
        _wgmma_20_b = cute.make_tensor(_wgmma_20_b_ptr, _wgmma_20_b_layout.outer)
        _wgmma_20_b_part = _wgmma_20_thr.partition_B(_wgmma_20_b)
        _wgmma_20_b_frag = _wgmma_20.make_fragment_B(_wgmma_20_b_part)
        _wgmma_20_a_ptr = cute.recast_ptr(p_bf16.iterator + (0), dtype=cutlass.BFloat16)
        _wgmma_20_a_frag = cute.make_tensor(_wgmma_20_a_ptr, _wgmma_20.partition_shape_A((64, 16)))
        _wgmma_20_d = cute.make_tensor(d_o.iterator + (0), _wgmma_20.partition_shape_C((64, 128)))
        _wgmma_20.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, cutlass.Boolean((True) != 0))
        cute.gemm(_wgmma_20, _wgmma_20_d, _wgmma_20_a_frag[None, None, 0], _wgmma_20_b_frag[None, None, 0, 0], _wgmma_20_d)
        _wgmma_21 = cutlass_hopper.make_trivial_tiled_mma(
            cutlass.BFloat16, cutlass.BFloat16,
            cute.nvgpu.OperandMajorMode.K, cute.nvgpu.OperandMajorMode.MN,
            cutlass.Float32, (1, 1, 1), (64, 128),
            cute.nvgpu.warpgroup.OperandSource.RMEM,
        )
        _wgmma_21_thr = _wgmma_21.get_slice(tid)
        _wgmma_21_b_ptr = cute.make_ptr(cutlass.BFloat16, cutlass.Uint32((cutlass.Uint64((_wgmma_b_0_7 + 128)) & cutlass.Uint64(0x3FFF)) << 4), mem_space=cute.AddressSpace.smem, assumed_align=16)
        _wgmma_21_b_layout = cutlass_hopper.make_smem_layout_b(CutlassLayout.COL_MAJOR, (64, 128, 64), cutlass.BFloat16, 1)
        _wgmma_21_b_ptr = cute.recast_ptr(_wgmma_21_b_ptr, swizzle_=_wgmma_21_b_layout.inner, dtype=cutlass.BFloat16)
        _wgmma_21_b = cute.make_tensor(_wgmma_21_b_ptr, _wgmma_21_b_layout.outer)
        _wgmma_21_b_part = _wgmma_21_thr.partition_B(_wgmma_21_b)
        _wgmma_21_b_frag = _wgmma_21.make_fragment_B(_wgmma_21_b_part)
        _wgmma_21_a_ptr = cute.recast_ptr(p_bf16.iterator + (4), dtype=cutlass.BFloat16)
        _wgmma_21_a_frag = cute.make_tensor(_wgmma_21_a_ptr, _wgmma_21.partition_shape_A((64, 16)))
        _wgmma_21_d = cute.make_tensor(d_o.iterator + (0), _wgmma_21.partition_shape_C((64, 128)))
        _wgmma_21.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, cutlass.Boolean((True) != 0))
        cute.gemm(_wgmma_21, _wgmma_21_d, _wgmma_21_a_frag[None, None, 0], _wgmma_21_b_frag[None, None, 0, 0], _wgmma_21_d)
        _wgmma_22 = cutlass_hopper.make_trivial_tiled_mma(
            cutlass.BFloat16, cutlass.BFloat16,
            cute.nvgpu.OperandMajorMode.K, cute.nvgpu.OperandMajorMode.MN,
            cutlass.Float32, (1, 1, 1), (64, 128),
            cute.nvgpu.warpgroup.OperandSource.RMEM,
        )
        _wgmma_22_thr = _wgmma_22.get_slice(tid)
        _wgmma_22_b_ptr = cute.make_ptr(cutlass.BFloat16, cutlass.Uint32((cutlass.Uint64((_wgmma_b_0_7 + 256)) & cutlass.Uint64(0x3FFF)) << 4), mem_space=cute.AddressSpace.smem, assumed_align=16)
        _wgmma_22_b_layout = cutlass_hopper.make_smem_layout_b(CutlassLayout.COL_MAJOR, (64, 128, 64), cutlass.BFloat16, 1)
        _wgmma_22_b_ptr = cute.recast_ptr(_wgmma_22_b_ptr, swizzle_=_wgmma_22_b_layout.inner, dtype=cutlass.BFloat16)
        _wgmma_22_b = cute.make_tensor(_wgmma_22_b_ptr, _wgmma_22_b_layout.outer)
        _wgmma_22_b_part = _wgmma_22_thr.partition_B(_wgmma_22_b)
        _wgmma_22_b_frag = _wgmma_22.make_fragment_B(_wgmma_22_b_part)
        _wgmma_22_a_ptr = cute.recast_ptr(p_bf16.iterator + (8), dtype=cutlass.BFloat16)
        _wgmma_22_a_frag = cute.make_tensor(_wgmma_22_a_ptr, _wgmma_22.partition_shape_A((64, 16)))
        _wgmma_22_d = cute.make_tensor(d_o.iterator + (0), _wgmma_22.partition_shape_C((64, 128)))
        _wgmma_22.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, cutlass.Boolean((True) != 0))
        cute.gemm(_wgmma_22, _wgmma_22_d, _wgmma_22_a_frag[None, None, 0], _wgmma_22_b_frag[None, None, 0, 0], _wgmma_22_d)
        _wgmma_23 = cutlass_hopper.make_trivial_tiled_mma(
            cutlass.BFloat16, cutlass.BFloat16,
            cute.nvgpu.OperandMajorMode.K, cute.nvgpu.OperandMajorMode.MN,
            cutlass.Float32, (1, 1, 1), (64, 128),
            cute.nvgpu.warpgroup.OperandSource.RMEM,
        )
        _wgmma_23_thr = _wgmma_23.get_slice(tid)
        _wgmma_23_b_ptr = cute.make_ptr(cutlass.BFloat16, cutlass.Uint32((cutlass.Uint64((_wgmma_b_0_7 + 384)) & cutlass.Uint64(0x3FFF)) << 4), mem_space=cute.AddressSpace.smem, assumed_align=16)
        _wgmma_23_b_layout = cutlass_hopper.make_smem_layout_b(CutlassLayout.COL_MAJOR, (64, 128, 64), cutlass.BFloat16, 1)
        _wgmma_23_b_ptr = cute.recast_ptr(_wgmma_23_b_ptr, swizzle_=_wgmma_23_b_layout.inner, dtype=cutlass.BFloat16)
        _wgmma_23_b = cute.make_tensor(_wgmma_23_b_ptr, _wgmma_23_b_layout.outer)
        _wgmma_23_b_part = _wgmma_23_thr.partition_B(_wgmma_23_b)
        _wgmma_23_b_frag = _wgmma_23.make_fragment_B(_wgmma_23_b_part)
        _wgmma_23_a_ptr = cute.recast_ptr(p_bf16.iterator + (12), dtype=cutlass.BFloat16)
        _wgmma_23_a_frag = cute.make_tensor(_wgmma_23_a_ptr, _wgmma_23.partition_shape_A((64, 16)))
        _wgmma_23_d = cute.make_tensor(d_o.iterator + (0), _wgmma_23.partition_shape_C((64, 128)))
        _wgmma_23.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, cutlass.Boolean((True) != 0))
        cute.gemm(_wgmma_23, _wgmma_23_d, _wgmma_23_a_frag[None, None, 0], _wgmma_23_b_frag[None, None, 0, 0], _wgmma_23_d)
        cute.nvgpu.warpgroup.commit_group()
        cute.nvgpu.warpgroup.wait_group(0)
    _shfl_xor_8 = cute.arch.shuffle_sync_bfly(row_sum0[0], 2, mask=0xFFFFFFFF, mask_and_clamp=31)
    row_sum0[0] += cutlass.Float32(_shfl_xor_8)
    _shfl_xor_9 = cute.arch.shuffle_sync_bfly(row_sum0[0], 1, mask=0xFFFFFFFF, mask_and_clamp=31)
    row_sum0[0] += cutlass.Float32(_shfl_xor_9)
    _shfl_xor_10 = cute.arch.shuffle_sync_bfly(row_sum1[0], 2, mask=0xFFFFFFFF, mask_and_clamp=31)
    row_sum1[0] += cutlass.Float32(_shfl_xor_10)
    _shfl_xor_11 = cute.arch.shuffle_sync_bfly(row_sum1[0], 1, mask=0xFFFFFFFF, mask_and_clamp=31)
    row_sum1[0] += cutlass.Float32(_shfl_xor_11)
    qj = cutlass.Int32((lane & 3))
    if (wg == 1):
        prims.store_ext(cutlass.Float32(d_o[0]).ir_value(), (part_lo + (tid_wg)))
        prims.store_ext(cutlass.Float32(d_o[1]).ir_value(), (part_lo + ((128 + tid_wg))))
        prims.store_ext(cutlass.Float32(d_o[2]).ir_value(), (part_lo + ((256 + tid_wg))))
        prims.store_ext(cutlass.Float32(d_o[3]).ir_value(), (part_lo + ((384 + tid_wg))))
        prims.store_ext(cutlass.Float32(d_o[4]).ir_value(), (part_lo + ((512 + tid_wg))))
        prims.store_ext(cutlass.Float32(d_o[5]).ir_value(), (part_lo + ((640 + tid_wg))))
        prims.store_ext(cutlass.Float32(d_o[6]).ir_value(), (part_lo + ((768 + tid_wg))))
        prims.store_ext(cutlass.Float32(d_o[7]).ir_value(), (part_lo + ((896 + tid_wg))))
        prims.store_ext(cutlass.Float32(d_o[8]).ir_value(), (part_lo + ((1024 + tid_wg))))
        prims.store_ext(cutlass.Float32(d_o[9]).ir_value(), (part_lo + ((1152 + tid_wg))))
        prims.store_ext(cutlass.Float32(d_o[10]).ir_value(), (part_lo + ((1280 + tid_wg))))
        prims.store_ext(cutlass.Float32(d_o[11]).ir_value(), (part_lo + ((1408 + tid_wg))))
        prims.store_ext(cutlass.Float32(d_o[12]).ir_value(), (part_lo + ((1536 + tid_wg))))
        prims.store_ext(cutlass.Float32(d_o[13]).ir_value(), (part_lo + ((1664 + tid_wg))))
        prims.store_ext(cutlass.Float32(d_o[14]).ir_value(), (part_lo + ((1792 + tid_wg))))
        prims.store_ext(cutlass.Float32(d_o[15]).ir_value(), (part_lo + ((1920 + tid_wg))))
        prims.store_ext(cutlass.Float32(d_o[16]).ir_value(), (part_lo + ((2048 + tid_wg))))
        prims.store_ext(cutlass.Float32(d_o[17]).ir_value(), (part_lo + ((2176 + tid_wg))))
        prims.store_ext(cutlass.Float32(d_o[18]).ir_value(), (part_lo + ((2304 + tid_wg))))
        prims.store_ext(cutlass.Float32(d_o[19]).ir_value(), (part_lo + ((2432 + tid_wg))))
        prims.store_ext(cutlass.Float32(d_o[20]).ir_value(), (part_lo + ((2560 + tid_wg))))
        prims.store_ext(cutlass.Float32(d_o[21]).ir_value(), (part_lo + ((2688 + tid_wg))))
        prims.store_ext(cutlass.Float32(d_o[22]).ir_value(), (part_lo + ((2816 + tid_wg))))
        prims.store_ext(cutlass.Float32(d_o[23]).ir_value(), (part_lo + ((2944 + tid_wg))))
        prims.store_ext(cutlass.Float32(d_o[24]).ir_value(), (part_lo + ((3072 + tid_wg))))
        prims.store_ext(cutlass.Float32(d_o[25]).ir_value(), (part_lo + ((3200 + tid_wg))))
        prims.store_ext(cutlass.Float32(d_o[26]).ir_value(), (part_lo + ((3328 + tid_wg))))
        prims.store_ext(cutlass.Float32(d_o[27]).ir_value(), (part_lo + ((3456 + tid_wg))))
        prims.store_ext(cutlass.Float32(d_o[28]).ir_value(), (part_lo + ((3584 + tid_wg))))
        prims.store_ext(cutlass.Float32(d_o[29]).ir_value(), (part_lo + ((3712 + tid_wg))))
        prims.store_ext(cutlass.Float32(d_o[30]).ir_value(), (part_lo + ((3840 + tid_wg))))
        prims.store_ext(cutlass.Float32(d_o[31]).ir_value(), (part_lo + ((3968 + tid_wg))))
        if (qj == 0):
            prims.store_ext(cutlass.Float32(row_max0[0]).ir_value(), (mstats + ((quad * 4))))
            prims.store_ext(cutlass.Float32(row_max1[0]).ir_value(), (mstats + (((quad * 4) + 1))))
            prims.store_ext(cutlass.Float32(row_sum0[0]).ir_value(), (mstats + (((quad * 4) + 2))))
            prims.store_ext(cutlass.Float32(row_sum1[0]).ir_value(), (mstats + (((quad * 4) + 3))))
    if (wg == 0):
        prims.store_ext(cutlass.Float32(d_o[32]).ir_value(), (part_hi0 + (tid_wg)))
        prims.store_ext(cutlass.Float32(d_o[33]).ir_value(), (part_hi0 + ((128 + tid_wg))))
        prims.store_ext(cutlass.Float32(d_o[34]).ir_value(), (part_hi0 + ((256 + tid_wg))))
        prims.store_ext(cutlass.Float32(d_o[35]).ir_value(), (part_hi0 + ((384 + tid_wg))))
        prims.store_ext(cutlass.Float32(d_o[36]).ir_value(), (part_hi0 + ((512 + tid_wg))))
        prims.store_ext(cutlass.Float32(d_o[37]).ir_value(), (part_hi0 + ((640 + tid_wg))))
        prims.store_ext(cutlass.Float32(d_o[38]).ir_value(), (part_hi0 + ((768 + tid_wg))))
        prims.store_ext(cutlass.Float32(d_o[39]).ir_value(), (part_hi0 + ((896 + tid_wg))))
        prims.store_ext(cutlass.Float32(d_o[40]).ir_value(), (part_hi0 + ((1024 + tid_wg))))
        prims.store_ext(cutlass.Float32(d_o[41]).ir_value(), (part_hi0 + ((1152 + tid_wg))))
        prims.store_ext(cutlass.Float32(d_o[42]).ir_value(), (part_hi0 + ((1280 + tid_wg))))
        prims.store_ext(cutlass.Float32(d_o[43]).ir_value(), (part_hi0 + ((1408 + tid_wg))))
        prims.store_ext(cutlass.Float32(d_o[44]).ir_value(), (part_hi0 + ((1536 + tid_wg))))
        prims.store_ext(cutlass.Float32(d_o[45]).ir_value(), (part_hi0 + ((1664 + tid_wg))))
        prims.store_ext(cutlass.Float32(d_o[46]).ir_value(), (part_hi0 + ((1792 + tid_wg))))
        prims.store_ext(cutlass.Float32(d_o[47]).ir_value(), (part_hi0 + ((1920 + tid_wg))))
        prims.store_ext(cutlass.Float32(d_o[48]).ir_value(), (part_hi0 + ((2048 + tid_wg))))
        prims.store_ext(cutlass.Float32(d_o[49]).ir_value(), (part_hi0 + ((2176 + tid_wg))))
        prims.store_ext(cutlass.Float32(d_o[50]).ir_value(), (part_hi0 + ((2304 + tid_wg))))
        prims.store_ext(cutlass.Float32(d_o[51]).ir_value(), (part_hi0 + ((2432 + tid_wg))))
        prims.store_ext(cutlass.Float32(d_o[52]).ir_value(), (part_hi0 + ((2560 + tid_wg))))
        prims.store_ext(cutlass.Float32(d_o[53]).ir_value(), (part_hi0 + ((2688 + tid_wg))))
        prims.store_ext(cutlass.Float32(d_o[54]).ir_value(), (part_hi0 + ((2816 + tid_wg))))
        prims.store_ext(cutlass.Float32(d_o[55]).ir_value(), (part_hi0 + ((2944 + tid_wg))))
        prims.store_ext(cutlass.Float32(d_o[56]).ir_value(), (part_hi0 + ((3072 + tid_wg))))
        prims.store_ext(cutlass.Float32(d_o[57]).ir_value(), (part_hi0 + ((3200 + tid_wg))))
        prims.store_ext(cutlass.Float32(d_o[58]).ir_value(), (part_hi0 + ((3328 + tid_wg))))
        prims.store_ext(cutlass.Float32(d_o[59]).ir_value(), (part_hi0 + ((3456 + tid_wg))))
        prims.store_ext(cutlass.Float32(d_o[60]).ir_value(), (part_hi0 + ((3584 + tid_wg))))
        prims.store_ext(cutlass.Float32(d_o[61]).ir_value(), (part_hi0 + ((3712 + tid_wg))))
        prims.store_ext(cutlass.Float32(d_o[62]).ir_value(), (part_hi0 + ((3840 + tid_wg))))
        prims.store_ext(cutlass.Float32(d_o[63]).ir_value(), (part_hi0 + ((3968 + tid_wg))))
        if (qj == 0):
            prims.store_ext(cutlass.Float32(row_max0[0]).ir_value(), (mstats + ((128 + (quad * 4)))))
            prims.store_ext(cutlass.Float32(row_max1[0]).ir_value(), (mstats + (((128 + (quad * 4)) + 1))))
            prims.store_ext(cutlass.Float32(row_sum0[0]).ir_value(), (mstats + (((128 + (quad * 4)) + 2))))
            prims.store_ext(cutlass.Float32(row_sum1[0]).ir_value(), (mstats + (((128 + (quad * 4)) + 3))))
    prims.barrier_cta_sync(8, thread_count=256)
    pbase = cutlass.Int32((0 if (wg == 0) else 128))
    pm0 = cutlass.Float32(_mstats[(pbase + (quad * 4))])
    pm1 = cutlass.Float32(_mstats[((pbase + (quad * 4)) + 1)])
    pl0 = cutlass.Float32(_mstats[((pbase + (quad * 4)) + 2)])
    pl1 = cutlass.Float32(_mstats[((pbase + (quad * 4)) + 3)])
    x0m0 = cutlass.Float32((row_max0[0] if (wg == 0) else pm0))
    x0m1 = cutlass.Float32((row_max1[0] if (wg == 0) else pm1))
    x1m0 = cutlass.Float32((pm0 if (wg == 0) else row_max0[0]))
    x1m1 = cutlass.Float32((pm1 if (wg == 0) else row_max1[0]))
    x0l0 = cutlass.Float32((row_sum0[0] if (wg == 0) else pl0))
    x0l1 = cutlass.Float32((row_sum1[0] if (wg == 0) else pl1))
    x1l0 = cutlass.Float32((pl0 if (wg == 0) else row_sum0[0]))
    x1l1 = cutlass.Float32((pl1 if (wg == 0) else row_sum1[0]))
    _max_76 = cute.arch.fmax(x0m0, x1m0, ftz=False)
    mm0 = cutlass.Float32(_max_76)
    _max_77 = cute.arch.fmax(x0m1, x1m1, ftz=False)
    mm1 = cutlass.Float32(_max_77)
    _exp2_68 = cute.math.exp2((x0m0 - mm0), approx=True, ftz=True)
    f00 = cutlass.Float32((0.0 if (x0m0 == (0 - float("inf"))) else _exp2_68))
    _exp2_69 = cute.math.exp2((x0m1 - mm1), approx=True, ftz=True)
    f01 = cutlass.Float32((0.0 if (x0m1 == (0 - float("inf"))) else _exp2_69))
    _exp2_70 = cute.math.exp2((x1m0 - mm0), approx=True, ftz=True)
    f10 = cutlass.Float32((0.0 if (x1m0 == (0 - float("inf"))) else _exp2_70))
    _exp2_71 = cute.math.exp2((x1m1 - mm1), approx=True, ftz=True)
    f11 = cutlass.Float32((0.0 if (x1m1 == (0 - float("inf"))) else _exp2_71))
    a0 = cutlass.Float32((f00 if (wg == 0) else f10))
    a1 = cutlass.Float32((f01 if (wg == 0) else f11))
    b0 = cutlass.Float32((f10 if (wg == 0) else f00))
    b1 = cutlass.Float32((f11 if (wg == 0) else f01))
    if (wg == 0):
        _fma_0 = cute.math.fma(_part_lo[tid_wg], (b1 if False else b0), (d_o[0] * (a1 if False else a0)), rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
        d_o[0] = cutlass.Float32(_fma_0)
        _fma_1 = cute.math.fma(_part_lo[(128 + tid_wg)], (b1 if False else b0), (d_o[1] * (a1 if False else a0)), rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
        d_o[1] = cutlass.Float32(_fma_1)
        _fma_2 = cute.math.fma(_part_lo[(256 + tid_wg)], (b1 if True else b0), (d_o[2] * (a1 if True else a0)), rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
        d_o[2] = cutlass.Float32(_fma_2)
        _fma_3 = cute.math.fma(_part_lo[(384 + tid_wg)], (b1 if True else b0), (d_o[3] * (a1 if True else a0)), rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
        d_o[3] = cutlass.Float32(_fma_3)
        _fma_4 = cute.math.fma(_part_lo[(512 + tid_wg)], (b1 if False else b0), (d_o[4] * (a1 if False else a0)), rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
        d_o[4] = cutlass.Float32(_fma_4)
        _fma_5 = cute.math.fma(_part_lo[(640 + tid_wg)], (b1 if False else b0), (d_o[5] * (a1 if False else a0)), rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
        d_o[5] = cutlass.Float32(_fma_5)
        _fma_6 = cute.math.fma(_part_lo[(768 + tid_wg)], (b1 if True else b0), (d_o[6] * (a1 if True else a0)), rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
        d_o[6] = cutlass.Float32(_fma_6)
        _fma_7 = cute.math.fma(_part_lo[(896 + tid_wg)], (b1 if True else b0), (d_o[7] * (a1 if True else a0)), rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
        d_o[7] = cutlass.Float32(_fma_7)
        _fma_8 = cute.math.fma(_part_lo[(1024 + tid_wg)], (b1 if False else b0), (d_o[8] * (a1 if False else a0)), rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
        d_o[8] = cutlass.Float32(_fma_8)
        _fma_9 = cute.math.fma(_part_lo[(1152 + tid_wg)], (b1 if False else b0), (d_o[9] * (a1 if False else a0)), rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
        d_o[9] = cutlass.Float32(_fma_9)
        _fma_10 = cute.math.fma(_part_lo[(1280 + tid_wg)], (b1 if True else b0), (d_o[10] * (a1 if True else a0)), rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
        d_o[10] = cutlass.Float32(_fma_10)
        _fma_11 = cute.math.fma(_part_lo[(1408 + tid_wg)], (b1 if True else b0), (d_o[11] * (a1 if True else a0)), rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
        d_o[11] = cutlass.Float32(_fma_11)
        _fma_12 = cute.math.fma(_part_lo[(1536 + tid_wg)], (b1 if False else b0), (d_o[12] * (a1 if False else a0)), rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
        d_o[12] = cutlass.Float32(_fma_12)
        _fma_13 = cute.math.fma(_part_lo[(1664 + tid_wg)], (b1 if False else b0), (d_o[13] * (a1 if False else a0)), rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
        d_o[13] = cutlass.Float32(_fma_13)
        _fma_14 = cute.math.fma(_part_lo[(1792 + tid_wg)], (b1 if True else b0), (d_o[14] * (a1 if True else a0)), rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
        d_o[14] = cutlass.Float32(_fma_14)
        _fma_15 = cute.math.fma(_part_lo[(1920 + tid_wg)], (b1 if True else b0), (d_o[15] * (a1 if True else a0)), rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
        d_o[15] = cutlass.Float32(_fma_15)
        _fma_16 = cute.math.fma(_part_lo[(2048 + tid_wg)], (b1 if False else b0), (d_o[16] * (a1 if False else a0)), rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
        d_o[16] = cutlass.Float32(_fma_16)
        _fma_17 = cute.math.fma(_part_lo[(2176 + tid_wg)], (b1 if False else b0), (d_o[17] * (a1 if False else a0)), rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
        d_o[17] = cutlass.Float32(_fma_17)
        _fma_18 = cute.math.fma(_part_lo[(2304 + tid_wg)], (b1 if True else b0), (d_o[18] * (a1 if True else a0)), rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
        d_o[18] = cutlass.Float32(_fma_18)
        _fma_19 = cute.math.fma(_part_lo[(2432 + tid_wg)], (b1 if True else b0), (d_o[19] * (a1 if True else a0)), rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
        d_o[19] = cutlass.Float32(_fma_19)
        _fma_20 = cute.math.fma(_part_lo[(2560 + tid_wg)], (b1 if False else b0), (d_o[20] * (a1 if False else a0)), rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
        d_o[20] = cutlass.Float32(_fma_20)
        _fma_21 = cute.math.fma(_part_lo[(2688 + tid_wg)], (b1 if False else b0), (d_o[21] * (a1 if False else a0)), rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
        d_o[21] = cutlass.Float32(_fma_21)
        _fma_22 = cute.math.fma(_part_lo[(2816 + tid_wg)], (b1 if True else b0), (d_o[22] * (a1 if True else a0)), rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
        d_o[22] = cutlass.Float32(_fma_22)
        _fma_23 = cute.math.fma(_part_lo[(2944 + tid_wg)], (b1 if True else b0), (d_o[23] * (a1 if True else a0)), rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
        d_o[23] = cutlass.Float32(_fma_23)
        _fma_24 = cute.math.fma(_part_lo[(3072 + tid_wg)], (b1 if False else b0), (d_o[24] * (a1 if False else a0)), rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
        d_o[24] = cutlass.Float32(_fma_24)
        _fma_25 = cute.math.fma(_part_lo[(3200 + tid_wg)], (b1 if False else b0), (d_o[25] * (a1 if False else a0)), rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
        d_o[25] = cutlass.Float32(_fma_25)
        _fma_26 = cute.math.fma(_part_lo[(3328 + tid_wg)], (b1 if True else b0), (d_o[26] * (a1 if True else a0)), rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
        d_o[26] = cutlass.Float32(_fma_26)
        _fma_27 = cute.math.fma(_part_lo[(3456 + tid_wg)], (b1 if True else b0), (d_o[27] * (a1 if True else a0)), rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
        d_o[27] = cutlass.Float32(_fma_27)
        _fma_28 = cute.math.fma(_part_lo[(3584 + tid_wg)], (b1 if False else b0), (d_o[28] * (a1 if False else a0)), rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
        d_o[28] = cutlass.Float32(_fma_28)
        _fma_29 = cute.math.fma(_part_lo[(3712 + tid_wg)], (b1 if False else b0), (d_o[29] * (a1 if False else a0)), rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
        d_o[29] = cutlass.Float32(_fma_29)
        _fma_30 = cute.math.fma(_part_lo[(3840 + tid_wg)], (b1 if True else b0), (d_o[30] * (a1 if True else a0)), rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
        d_o[30] = cutlass.Float32(_fma_30)
        _fma_31 = cute.math.fma(_part_lo[(3968 + tid_wg)], (b1 if True else b0), (d_o[31] * (a1 if True else a0)), rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
        d_o[31] = cutlass.Float32(_fma_31)
    if (wg == 1):
        _fma_32 = cute.math.fma(_part_hi0[tid_wg], (b1 if False else b0), (d_o[32] * (a1 if False else a0)), rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
        d_o[32] = cutlass.Float32(_fma_32)
        _fma_33 = cute.math.fma(_part_hi0[(128 + tid_wg)], (b1 if False else b0), (d_o[33] * (a1 if False else a0)), rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
        d_o[33] = cutlass.Float32(_fma_33)
        _fma_34 = cute.math.fma(_part_hi0[(256 + tid_wg)], (b1 if True else b0), (d_o[34] * (a1 if True else a0)), rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
        d_o[34] = cutlass.Float32(_fma_34)
        _fma_35 = cute.math.fma(_part_hi0[(384 + tid_wg)], (b1 if True else b0), (d_o[35] * (a1 if True else a0)), rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
        d_o[35] = cutlass.Float32(_fma_35)
        _fma_36 = cute.math.fma(_part_hi0[(512 + tid_wg)], (b1 if False else b0), (d_o[36] * (a1 if False else a0)), rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
        d_o[36] = cutlass.Float32(_fma_36)
        _fma_37 = cute.math.fma(_part_hi0[(640 + tid_wg)], (b1 if False else b0), (d_o[37] * (a1 if False else a0)), rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
        d_o[37] = cutlass.Float32(_fma_37)
        _fma_38 = cute.math.fma(_part_hi0[(768 + tid_wg)], (b1 if True else b0), (d_o[38] * (a1 if True else a0)), rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
        d_o[38] = cutlass.Float32(_fma_38)
        _fma_39 = cute.math.fma(_part_hi0[(896 + tid_wg)], (b1 if True else b0), (d_o[39] * (a1 if True else a0)), rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
        d_o[39] = cutlass.Float32(_fma_39)
        _fma_40 = cute.math.fma(_part_hi0[(1024 + tid_wg)], (b1 if False else b0), (d_o[40] * (a1 if False else a0)), rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
        d_o[40] = cutlass.Float32(_fma_40)
        _fma_41 = cute.math.fma(_part_hi0[(1152 + tid_wg)], (b1 if False else b0), (d_o[41] * (a1 if False else a0)), rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
        d_o[41] = cutlass.Float32(_fma_41)
        _fma_42 = cute.math.fma(_part_hi0[(1280 + tid_wg)], (b1 if True else b0), (d_o[42] * (a1 if True else a0)), rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
        d_o[42] = cutlass.Float32(_fma_42)
        _fma_43 = cute.math.fma(_part_hi0[(1408 + tid_wg)], (b1 if True else b0), (d_o[43] * (a1 if True else a0)), rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
        d_o[43] = cutlass.Float32(_fma_43)
        _fma_44 = cute.math.fma(_part_hi0[(1536 + tid_wg)], (b1 if False else b0), (d_o[44] * (a1 if False else a0)), rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
        d_o[44] = cutlass.Float32(_fma_44)
        _fma_45 = cute.math.fma(_part_hi0[(1664 + tid_wg)], (b1 if False else b0), (d_o[45] * (a1 if False else a0)), rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
        d_o[45] = cutlass.Float32(_fma_45)
        _fma_46 = cute.math.fma(_part_hi0[(1792 + tid_wg)], (b1 if True else b0), (d_o[46] * (a1 if True else a0)), rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
        d_o[46] = cutlass.Float32(_fma_46)
        _fma_47 = cute.math.fma(_part_hi0[(1920 + tid_wg)], (b1 if True else b0), (d_o[47] * (a1 if True else a0)), rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
        d_o[47] = cutlass.Float32(_fma_47)
        _fma_48 = cute.math.fma(_part_hi0[(2048 + tid_wg)], (b1 if False else b0), (d_o[48] * (a1 if False else a0)), rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
        d_o[48] = cutlass.Float32(_fma_48)
        _fma_49 = cute.math.fma(_part_hi0[(2176 + tid_wg)], (b1 if False else b0), (d_o[49] * (a1 if False else a0)), rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
        d_o[49] = cutlass.Float32(_fma_49)
        _fma_50 = cute.math.fma(_part_hi0[(2304 + tid_wg)], (b1 if True else b0), (d_o[50] * (a1 if True else a0)), rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
        d_o[50] = cutlass.Float32(_fma_50)
        _fma_51 = cute.math.fma(_part_hi0[(2432 + tid_wg)], (b1 if True else b0), (d_o[51] * (a1 if True else a0)), rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
        d_o[51] = cutlass.Float32(_fma_51)
        _fma_52 = cute.math.fma(_part_hi0[(2560 + tid_wg)], (b1 if False else b0), (d_o[52] * (a1 if False else a0)), rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
        d_o[52] = cutlass.Float32(_fma_52)
        _fma_53 = cute.math.fma(_part_hi0[(2688 + tid_wg)], (b1 if False else b0), (d_o[53] * (a1 if False else a0)), rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
        d_o[53] = cutlass.Float32(_fma_53)
        _fma_54 = cute.math.fma(_part_hi0[(2816 + tid_wg)], (b1 if True else b0), (d_o[54] * (a1 if True else a0)), rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
        d_o[54] = cutlass.Float32(_fma_54)
        _fma_55 = cute.math.fma(_part_hi0[(2944 + tid_wg)], (b1 if True else b0), (d_o[55] * (a1 if True else a0)), rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
        d_o[55] = cutlass.Float32(_fma_55)
        _fma_56 = cute.math.fma(_part_hi0[(3072 + tid_wg)], (b1 if False else b0), (d_o[56] * (a1 if False else a0)), rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
        d_o[56] = cutlass.Float32(_fma_56)
        _fma_57 = cute.math.fma(_part_hi0[(3200 + tid_wg)], (b1 if False else b0), (d_o[57] * (a1 if False else a0)), rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
        d_o[57] = cutlass.Float32(_fma_57)
        _fma_58 = cute.math.fma(_part_hi0[(3328 + tid_wg)], (b1 if True else b0), (d_o[58] * (a1 if True else a0)), rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
        d_o[58] = cutlass.Float32(_fma_58)
        _fma_59 = cute.math.fma(_part_hi0[(3456 + tid_wg)], (b1 if True else b0), (d_o[59] * (a1 if True else a0)), rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
        d_o[59] = cutlass.Float32(_fma_59)
        _fma_60 = cute.math.fma(_part_hi0[(3584 + tid_wg)], (b1 if False else b0), (d_o[60] * (a1 if False else a0)), rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
        d_o[60] = cutlass.Float32(_fma_60)
        _fma_61 = cute.math.fma(_part_hi0[(3712 + tid_wg)], (b1 if False else b0), (d_o[61] * (a1 if False else a0)), rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
        d_o[61] = cutlass.Float32(_fma_61)
        _fma_62 = cute.math.fma(_part_hi0[(3840 + tid_wg)], (b1 if True else b0), (d_o[62] * (a1 if True else a0)), rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
        d_o[62] = cutlass.Float32(_fma_62)
        _fma_63 = cute.math.fma(_part_hi0[(3968 + tid_wg)], (b1 if True else b0), (d_o[63] * (a1 if True else a0)), rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
        d_o[63] = cutlass.Float32(_fma_63)
    _fma_64 = cute.math.fma(x1l0, f10, (x0l0 * f00), rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
    row_sum0[0] = cutlass.Float32(_fma_64)
    _fma_65 = cute.math.fma(x1l1, f11, (x0l1 * f01), rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
    row_sum1[0] = cutlass.Float32(_fma_65)
    row_max0[0] = cutlass.Float32(mm0)
    row_max1[0] = cutlass.Float32(mm1)
    cute.arch.cluster_wait()
    _phase_recv_full_0[0] = cutlass.Uint32(0)
    if (wg == 0):
        my_rank = cutlass.Int32(cta_rank)
        my_off = cutlass.Int32((tid_wg * 16))
        if (my_rank != 0):
            pslot[0] = cutlass.Int32(my_rank)
            _if_condition_24 = cutlass.Boolean((my_rank > 0))
            pslot[0] = cutlass.Int32(cutlass.select_(_if_condition_24, cutlass.Int32((my_rank - 1)), pslot[0]))
            _cluster_mapa_25 = cute.arch.map_dsmem_ptr(recv_full_addr, cutlass.Int32(0))
            _mapa_0 = cutlass.Uint32(_cluster_mapa_25.toint())
            _cluster_mapa_26 = cute.arch.map_dsmem_ptr(cute.make_ptr(cutlass.Uint8, cutlass.Uint32(((recv_smem_addr + cutlass.Uint32((pslot[0] * 9216))) + cutlass.Uint32(my_off))), mem_space=cute.AddressSpace.smem, assumed_align=4), cutlass.Int32(0))
            _mapa_1 = cutlass.Uint32(_cluster_mapa_26.toint())
            cutlass_llvm.inline_asm(
                res=None,
                operands_=[(cutlass.Uint32(_mapa_1)).ir_value(), (cutlass.Float32(d_o[0]).bitcast(cutlass.Int32)).ir_value(), (cutlass.Float32(d_o[1]).bitcast(cutlass.Int32)).ir_value(), (cutlass.Float32(d_o[2]).bitcast(cutlass.Int32)).ir_value(), (cutlass.Float32(d_o[3]).bitcast(cutlass.Int32)).ir_value(), (cutlass.Uint32(_mapa_0)).ir_value()],
                asm_string='st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [$0], {$1, $2, $3, $4}, [$5];',
                constraints='r,r,r,r,r,r,~{memory}',
                has_side_effects=True,
                is_align_stack=False,
                asm_dialect=cutlass_llvm.AsmDialect.AD_ATT,
            )
            cutlass_llvm.inline_asm(
                res=None,
                operands_=[(cutlass.Uint32((_mapa_1 + 2048))).ir_value(), (cutlass.Float32(d_o[4]).bitcast(cutlass.Int32)).ir_value(), (cutlass.Float32(d_o[5]).bitcast(cutlass.Int32)).ir_value(), (cutlass.Float32(d_o[6]).bitcast(cutlass.Int32)).ir_value(), (cutlass.Float32(d_o[7]).bitcast(cutlass.Int32)).ir_value(), (cutlass.Uint32(_mapa_0)).ir_value()],
                asm_string='st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [$0], {$1, $2, $3, $4}, [$5];',
                constraints='r,r,r,r,r,r,~{memory}',
                has_side_effects=True,
                is_align_stack=False,
                asm_dialect=cutlass_llvm.AsmDialect.AD_ATT,
            )
            cutlass_llvm.inline_asm(
                res=None,
                operands_=[(cutlass.Uint32((_mapa_1 + 4096))).ir_value(), (cutlass.Float32(d_o[8]).bitcast(cutlass.Int32)).ir_value(), (cutlass.Float32(d_o[9]).bitcast(cutlass.Int32)).ir_value(), (cutlass.Float32(d_o[10]).bitcast(cutlass.Int32)).ir_value(), (cutlass.Float32(d_o[11]).bitcast(cutlass.Int32)).ir_value(), (cutlass.Uint32(_mapa_0)).ir_value()],
                asm_string='st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [$0], {$1, $2, $3, $4}, [$5];',
                constraints='r,r,r,r,r,r,~{memory}',
                has_side_effects=True,
                is_align_stack=False,
                asm_dialect=cutlass_llvm.AsmDialect.AD_ATT,
            )
            cutlass_llvm.inline_asm(
                res=None,
                operands_=[(cutlass.Uint32((_mapa_1 + 6144))).ir_value(), (cutlass.Float32(d_o[12]).bitcast(cutlass.Int32)).ir_value(), (cutlass.Float32(d_o[13]).bitcast(cutlass.Int32)).ir_value(), (cutlass.Float32(d_o[14]).bitcast(cutlass.Int32)).ir_value(), (cutlass.Float32(d_o[15]).bitcast(cutlass.Int32)).ir_value(), (cutlass.Uint32(_mapa_0)).ir_value()],
                asm_string='st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [$0], {$1, $2, $3, $4}, [$5];',
                constraints='r,r,r,r,r,r,~{memory}',
                has_side_effects=True,
                is_align_stack=False,
                asm_dialect=cutlass_llvm.AsmDialect.AD_ATT,
            )
            if (qj == 0):
                _cluster_mapa_27 = cute.arch.map_dsmem_ptr(cute.make_ptr(cutlass.Uint8, cutlass.Uint32((((recv_smem_addr + cutlass.Uint32((pslot[0] * 9216))) + 8192) + cutlass.Uint32((m0_local * 16)))), mem_space=cute.AddressSpace.smem, assumed_align=4), cutlass.Int32(0))
                _mapa_2 = cutlass.Uint32(_cluster_mapa_27.toint())
                cutlass_llvm.inline_asm(
                    res=None,
                    operands_=[(cutlass.Uint32(_mapa_2)).ir_value(), (cutlass.Float32(row_max0[0]).bitcast(cutlass.Int32)).ir_value(), (cutlass.Float32(row_sum0[0]).bitcast(cutlass.Int32)).ir_value(), (cutlass.Float32(row_max1[0]).bitcast(cutlass.Int32)).ir_value(), (cutlass.Float32(row_sum1[0]).bitcast(cutlass.Int32)).ir_value(), (cutlass.Uint32(_mapa_0)).ir_value()],
                    asm_string='st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [$0], {$1, $2, $3, $4}, [$5];',
                    constraints='r,r,r,r,r,r,~{memory}',
                    has_side_effects=True,
                    is_align_stack=False,
                    asm_dialect=cutlass_llvm.AsmDialect.AD_ATT,
                )
        if (my_rank != 1):
            pslot_1[0] = cutlass.Int32(my_rank)
            _if_condition_28 = cutlass.Boolean((my_rank > 1))
            pslot_1[0] = cutlass.Int32(cutlass.select_(_if_condition_28, cutlass.Int32((my_rank - 1)), pslot_1[0]))
            _cluster_mapa_29 = cute.arch.map_dsmem_ptr(recv_full_addr, cutlass.Int32(1))
            _mapa_3 = cutlass.Uint32(_cluster_mapa_29.toint())
            _cluster_mapa_30 = cute.arch.map_dsmem_ptr(cute.make_ptr(cutlass.Uint8, cutlass.Uint32(((recv_smem_addr + cutlass.Uint32((pslot_1[0] * 9216))) + cutlass.Uint32(my_off))), mem_space=cute.AddressSpace.smem, assumed_align=4), cutlass.Int32(1))
            _mapa_4 = cutlass.Uint32(_cluster_mapa_30.toint())
            cutlass_llvm.inline_asm(
                res=None,
                operands_=[(cutlass.Uint32(_mapa_4)).ir_value(), (cutlass.Float32(d_o[16]).bitcast(cutlass.Int32)).ir_value(), (cutlass.Float32(d_o[17]).bitcast(cutlass.Int32)).ir_value(), (cutlass.Float32(d_o[18]).bitcast(cutlass.Int32)).ir_value(), (cutlass.Float32(d_o[19]).bitcast(cutlass.Int32)).ir_value(), (cutlass.Uint32(_mapa_3)).ir_value()],
                asm_string='st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [$0], {$1, $2, $3, $4}, [$5];',
                constraints='r,r,r,r,r,r,~{memory}',
                has_side_effects=True,
                is_align_stack=False,
                asm_dialect=cutlass_llvm.AsmDialect.AD_ATT,
            )
            cutlass_llvm.inline_asm(
                res=None,
                operands_=[(cutlass.Uint32((_mapa_4 + 2048))).ir_value(), (cutlass.Float32(d_o[20]).bitcast(cutlass.Int32)).ir_value(), (cutlass.Float32(d_o[21]).bitcast(cutlass.Int32)).ir_value(), (cutlass.Float32(d_o[22]).bitcast(cutlass.Int32)).ir_value(), (cutlass.Float32(d_o[23]).bitcast(cutlass.Int32)).ir_value(), (cutlass.Uint32(_mapa_3)).ir_value()],
                asm_string='st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [$0], {$1, $2, $3, $4}, [$5];',
                constraints='r,r,r,r,r,r,~{memory}',
                has_side_effects=True,
                is_align_stack=False,
                asm_dialect=cutlass_llvm.AsmDialect.AD_ATT,
            )
            cutlass_llvm.inline_asm(
                res=None,
                operands_=[(cutlass.Uint32((_mapa_4 + 4096))).ir_value(), (cutlass.Float32(d_o[24]).bitcast(cutlass.Int32)).ir_value(), (cutlass.Float32(d_o[25]).bitcast(cutlass.Int32)).ir_value(), (cutlass.Float32(d_o[26]).bitcast(cutlass.Int32)).ir_value(), (cutlass.Float32(d_o[27]).bitcast(cutlass.Int32)).ir_value(), (cutlass.Uint32(_mapa_3)).ir_value()],
                asm_string='st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [$0], {$1, $2, $3, $4}, [$5];',
                constraints='r,r,r,r,r,r,~{memory}',
                has_side_effects=True,
                is_align_stack=False,
                asm_dialect=cutlass_llvm.AsmDialect.AD_ATT,
            )
            cutlass_llvm.inline_asm(
                res=None,
                operands_=[(cutlass.Uint32((_mapa_4 + 6144))).ir_value(), (cutlass.Float32(d_o[28]).bitcast(cutlass.Int32)).ir_value(), (cutlass.Float32(d_o[29]).bitcast(cutlass.Int32)).ir_value(), (cutlass.Float32(d_o[30]).bitcast(cutlass.Int32)).ir_value(), (cutlass.Float32(d_o[31]).bitcast(cutlass.Int32)).ir_value(), (cutlass.Uint32(_mapa_3)).ir_value()],
                asm_string='st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [$0], {$1, $2, $3, $4}, [$5];',
                constraints='r,r,r,r,r,r,~{memory}',
                has_side_effects=True,
                is_align_stack=False,
                asm_dialect=cutlass_llvm.AsmDialect.AD_ATT,
            )
            if (qj == 0):
                _cluster_mapa_31 = cute.arch.map_dsmem_ptr(cute.make_ptr(cutlass.Uint8, cutlass.Uint32((((recv_smem_addr + cutlass.Uint32((pslot_1[0] * 9216))) + 8192) + cutlass.Uint32((m0_local * 16)))), mem_space=cute.AddressSpace.smem, assumed_align=4), cutlass.Int32(1))
                _mapa_5 = cutlass.Uint32(_cluster_mapa_31.toint())
                cutlass_llvm.inline_asm(
                    res=None,
                    operands_=[(cutlass.Uint32(_mapa_5)).ir_value(), (cutlass.Float32(row_max0[0]).bitcast(cutlass.Int32)).ir_value(), (cutlass.Float32(row_sum0[0]).bitcast(cutlass.Int32)).ir_value(), (cutlass.Float32(row_max1[0]).bitcast(cutlass.Int32)).ir_value(), (cutlass.Float32(row_sum1[0]).bitcast(cutlass.Int32)).ir_value(), (cutlass.Uint32(_mapa_3)).ir_value()],
                    asm_string='st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [$0], {$1, $2, $3, $4}, [$5];',
                    constraints='r,r,r,r,r,r,~{memory}',
                    has_side_effects=True,
                    is_align_stack=False,
                    asm_dialect=cutlass_llvm.AsmDialect.AD_ATT,
                )
        if (my_rank != 2):
            pslot_2[0] = cutlass.Int32(my_rank)
            _if_condition_32 = cutlass.Boolean((my_rank > 2))
            pslot_2[0] = cutlass.Int32(cutlass.select_(_if_condition_32, cutlass.Int32((my_rank - 1)), pslot_2[0]))
            _cluster_mapa_33 = cute.arch.map_dsmem_ptr(recv_full_addr, cutlass.Int32(2))
            _mapa_6 = cutlass.Uint32(_cluster_mapa_33.toint())
            if (qj == 0):
                _cluster_mapa_34 = cute.arch.map_dsmem_ptr(cute.make_ptr(cutlass.Uint8, cutlass.Uint32((((recv_smem_addr + cutlass.Uint32((pslot_2[0] * 9216))) + 8192) + cutlass.Uint32((m0_local * 16)))), mem_space=cute.AddressSpace.smem, assumed_align=4), cutlass.Int32(2))
                _mapa_7 = cutlass.Uint32(_cluster_mapa_34.toint())
                cutlass_llvm.inline_asm(
                    res=None,
                    operands_=[(cutlass.Uint32(_mapa_7)).ir_value(), (cutlass.Float32(row_max0[0]).bitcast(cutlass.Int32)).ir_value(), (cutlass.Float32(row_sum0[0]).bitcast(cutlass.Int32)).ir_value(), (cutlass.Float32(row_max1[0]).bitcast(cutlass.Int32)).ir_value(), (cutlass.Float32(row_sum1[0]).bitcast(cutlass.Int32)).ir_value(), (cutlass.Uint32(_mapa_6)).ir_value()],
                    asm_string='st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [$0], {$1, $2, $3, $4}, [$5];',
                    constraints='r,r,r,r,r,r,~{memory}',
                    has_side_effects=True,
                    is_align_stack=False,
                    asm_dialect=cutlass_llvm.AsmDialect.AD_ATT,
                )
        if (my_rank != 3):
            pslot_3[0] = cutlass.Int32(my_rank)
            _if_condition_35 = cutlass.Boolean((my_rank > 3))
            pslot_3[0] = cutlass.Int32(cutlass.select_(_if_condition_35, cutlass.Int32((my_rank - 1)), pslot_3[0]))
            _cluster_mapa_36 = cute.arch.map_dsmem_ptr(recv_full_addr, cutlass.Int32(3))
            _mapa_8 = cutlass.Uint32(_cluster_mapa_36.toint())
            if (qj == 0):
                _cluster_mapa_37 = cute.arch.map_dsmem_ptr(cute.make_ptr(cutlass.Uint8, cutlass.Uint32((((recv_smem_addr + cutlass.Uint32((pslot_3[0] * 9216))) + 8192) + cutlass.Uint32((m0_local * 16)))), mem_space=cute.AddressSpace.smem, assumed_align=4), cutlass.Int32(3))
                _mapa_9 = cutlass.Uint32(_cluster_mapa_37.toint())
                cutlass_llvm.inline_asm(
                    res=None,
                    operands_=[(cutlass.Uint32(_mapa_9)).ir_value(), (cutlass.Float32(row_max0[0]).bitcast(cutlass.Int32)).ir_value(), (cutlass.Float32(row_sum0[0]).bitcast(cutlass.Int32)).ir_value(), (cutlass.Float32(row_max1[0]).bitcast(cutlass.Int32)).ir_value(), (cutlass.Float32(row_sum1[0]).bitcast(cutlass.Int32)).ir_value(), (cutlass.Uint32(_mapa_8)).ir_value()],
                    asm_string='st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [$0], {$1, $2, $3, $4}, [$5];',
                    constraints='r,r,r,r,r,r,~{memory}',
                    has_side_effects=True,
                    is_align_stack=False,
                    asm_dialect=cutlass_llvm.AsmDialect.AD_ATT,
                )
        if (my_rank < 4):
            while not prims.mbarrier_try_wait_parity(recv_full_addr, _phase_recv_full_0[0], time_limit=10000000, scope=prims.MBarrierScope.CLUSTER, order=prims.MemOrder.ACQUIRE):
                pass
            _phase_recv_full_0[0] ^= cutlass.Uint32(1)
            merged_max0[0] = cutlass.Float32(row_max0[0])
            merged_max1[0] = cutlass.Float32(row_max1[0])
            _max_78 = cute.arch.fmax(merged_max0[0], _recv_smem[(2048 + (m0_local * 4))], ftz=False)
            merged_max0[0] = cutlass.Float32(_max_78)
            _max_79 = cute.arch.fmax(merged_max1[0], _recv_smem[((2048 + (m0_local * 4)) + 2)], ftz=False)
            merged_max1[0] = cutlass.Float32(_max_79)
            _max_80 = cute.arch.fmax(merged_max0[0], _recv_smem[(4352 + (m0_local * 4))], ftz=False)
            merged_max0[0] = cutlass.Float32(_max_80)
            _max_81 = cute.arch.fmax(merged_max1[0], _recv_smem[((4352 + (m0_local * 4)) + 2)], ftz=False)
            merged_max1[0] = cutlass.Float32(_max_81)
            _max_82 = cute.arch.fmax(merged_max0[0], _recv_smem[(6656 + (m0_local * 4))], ftz=False)
            merged_max0[0] = cutlass.Float32(_max_82)
            _max_83 = cute.arch.fmax(merged_max1[0], _recv_smem[((6656 + (m0_local * 4)) + 2)], ftz=False)
            merged_max1[0] = cutlass.Float32(_max_83)
            _exp2_72 = cute.math.exp2((row_max0[0] - merged_max0[0]), approx=True, ftz=True)
            _exp2_73 = cute.math.exp2((row_max1[0] - merged_max1[0]), approx=True, ftz=True)
            msum0[0] = cutlass.Float32((row_sum0[0] * _exp2_72))
            msum1[0] = cutlass.Float32((row_sum1[0] * _exp2_73))
            w_peer0 = cute.make_rmem_tensor((3,), cutlass.Float32)
            w_peer1 = cute.make_rmem_tensor((3,), cutlass.Float32)
            sbase = cutlass.Int32((2048 + (m0_local * 4)))
            _exp2_74 = cute.math.exp2((_recv_smem[sbase] - merged_max0[0]), approx=True, ftz=True)
            w_peer0[0] = cutlass.Float32(_exp2_74)
            _exp2_75 = cute.math.exp2((_recv_smem[(sbase + 2)] - merged_max1[0]), approx=True, ftz=True)
            w_peer1[0] = cutlass.Float32(_exp2_75)
            _fma_66 = cute.math.fma(_recv_smem[(sbase + 1)], w_peer0[0], msum0[0], rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
            msum0[0] = cutlass.Float32(_fma_66)
            _fma_67 = cute.math.fma(_recv_smem[(sbase + 3)], w_peer1[0], msum1[0], rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
            msum1[0] = cutlass.Float32(_fma_67)
            sbase_0 = cutlass.Int32((4352 + (m0_local * 4)))
            _exp2_76 = cute.math.exp2((_recv_smem[sbase_0] - merged_max0[0]), approx=True, ftz=True)
            w_peer0[1] = cutlass.Float32(_exp2_76)
            _exp2_77 = cute.math.exp2((_recv_smem[(sbase_0 + 2)] - merged_max1[0]), approx=True, ftz=True)
            w_peer1[1] = cutlass.Float32(_exp2_77)
            _fma_68 = cute.math.fma(_recv_smem[(sbase_0 + 1)], w_peer0[1], msum0[0], rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
            msum0[0] = cutlass.Float32(_fma_68)
            _fma_69 = cute.math.fma(_recv_smem[(sbase_0 + 3)], w_peer1[1], msum1[0], rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
            msum1[0] = cutlass.Float32(_fma_69)
            sbase_1 = cutlass.Int32((6656 + (m0_local * 4)))
            _exp2_78 = cute.math.exp2((_recv_smem[sbase_1] - merged_max0[0]), approx=True, ftz=True)
            w_peer0[2] = cutlass.Float32(_exp2_78)
            _exp2_79 = cute.math.exp2((_recv_smem[(sbase_1 + 2)] - merged_max1[0]), approx=True, ftz=True)
            w_peer1[2] = cutlass.Float32(_exp2_79)
            _fma_70 = cute.math.fma(_recv_smem[(sbase_1 + 1)], w_peer0[2], msum0[0], rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
            msum0[0] = cutlass.Float32(_fma_70)
            _fma_71 = cute.math.fma(_recv_smem[(sbase_1 + 3)], w_peer1[2], msum1[0], rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
            msum1[0] = cutlass.Float32(_fma_71)
            my_tid4 = cutlass.Int32((tid_wg * 4))
            acc = cute.make_rmem_tensor((16,), cutlass.Float32)
            sset = cutlass.Int32(my_rank)
            fold[0] = cutlass.Int32((1 if (sset < 4) else 0))
            _if_condition_38 = cutlass.Boolean((sset < 0))
            fold[0] = cutlass.Int32(cutlass.select_(_if_condition_38, cutlass.Int32(0), fold[0]))
            _if_condition_39 = cutlass.Boolean((sset >= 2))
            fold[0] = cutlass.Int32(cutlass.select_(_if_condition_39, cutlass.Int32(0), fold[0]))
            if (fold[0] != 0):
                if (sset == 0):
                    acc[0] = cutlass.Float32((d_o[0] * (_exp2_73 if False else _exp2_72)))
                    acc[1] = cutlass.Float32((d_o[1] * (_exp2_73 if False else _exp2_72)))
                    acc[2] = cutlass.Float32((d_o[2] * (_exp2_73 if True else _exp2_72)))
                    acc[3] = cutlass.Float32((d_o[3] * (_exp2_73 if True else _exp2_72)))
                    acc[4] = cutlass.Float32((d_o[4] * (_exp2_73 if False else _exp2_72)))
                    acc[5] = cutlass.Float32((d_o[5] * (_exp2_73 if False else _exp2_72)))
                    acc[6] = cutlass.Float32((d_o[6] * (_exp2_73 if True else _exp2_72)))
                    acc[7] = cutlass.Float32((d_o[7] * (_exp2_73 if True else _exp2_72)))
                    acc[8] = cutlass.Float32((d_o[8] * (_exp2_73 if False else _exp2_72)))
                    acc[9] = cutlass.Float32((d_o[9] * (_exp2_73 if False else _exp2_72)))
                    acc[10] = cutlass.Float32((d_o[10] * (_exp2_73 if True else _exp2_72)))
                    acc[11] = cutlass.Float32((d_o[11] * (_exp2_73 if True else _exp2_72)))
                    acc[12] = cutlass.Float32((d_o[12] * (_exp2_73 if False else _exp2_72)))
                    acc[13] = cutlass.Float32((d_o[13] * (_exp2_73 if False else _exp2_72)))
                    acc[14] = cutlass.Float32((d_o[14] * (_exp2_73 if True else _exp2_72)))
                    acc[15] = cutlass.Float32((d_o[15] * (_exp2_73 if True else _exp2_72)))
                if (sset == 1):
                    acc[0] = cutlass.Float32((d_o[16] * (_exp2_73 if False else _exp2_72)))
                    acc[1] = cutlass.Float32((d_o[17] * (_exp2_73 if False else _exp2_72)))
                    acc[2] = cutlass.Float32((d_o[18] * (_exp2_73 if True else _exp2_72)))
                    acc[3] = cutlass.Float32((d_o[19] * (_exp2_73 if True else _exp2_72)))
                    acc[4] = cutlass.Float32((d_o[20] * (_exp2_73 if False else _exp2_72)))
                    acc[5] = cutlass.Float32((d_o[21] * (_exp2_73 if False else _exp2_72)))
                    acc[6] = cutlass.Float32((d_o[22] * (_exp2_73 if True else _exp2_72)))
                    acc[7] = cutlass.Float32((d_o[23] * (_exp2_73 if True else _exp2_72)))
                    acc[8] = cutlass.Float32((d_o[24] * (_exp2_73 if False else _exp2_72)))
                    acc[9] = cutlass.Float32((d_o[25] * (_exp2_73 if False else _exp2_72)))
                    acc[10] = cutlass.Float32((d_o[26] * (_exp2_73 if True else _exp2_72)))
                    acc[11] = cutlass.Float32((d_o[27] * (_exp2_73 if True else _exp2_72)))
                    acc[12] = cutlass.Float32((d_o[28] * (_exp2_73 if False else _exp2_72)))
                    acc[13] = cutlass.Float32((d_o[29] * (_exp2_73 if False else _exp2_72)))
                    acc[14] = cutlass.Float32((d_o[30] * (_exp2_73 if True else _exp2_72)))
                    acc[15] = cutlass.Float32((d_o[31] * (_exp2_73 if True else _exp2_72)))
                base = cutlass.Int32(my_tid4)
                _fma_72 = cute.math.fma(_recv_smem[base], (w_peer1[0] if False else w_peer0[0]), acc[0], rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
                acc[0] = cutlass.Float32(_fma_72)
                _fma_73 = cute.math.fma(_recv_smem[(base + 1)], (w_peer1[0] if False else w_peer0[0]), acc[1], rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
                acc[1] = cutlass.Float32(_fma_73)
                _fma_74 = cute.math.fma(_recv_smem[(base + 2)], (w_peer1[0] if True else w_peer0[0]), acc[2], rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
                acc[2] = cutlass.Float32(_fma_74)
                _fma_75 = cute.math.fma(_recv_smem[(base + 3)], (w_peer1[0] if True else w_peer0[0]), acc[3], rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
                acc[3] = cutlass.Float32(_fma_75)
                _fma_76 = cute.math.fma(_recv_smem[(base + 512)], (w_peer1[0] if False else w_peer0[0]), acc[4], rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
                acc[4] = cutlass.Float32(_fma_76)
                _fma_77 = cute.math.fma(_recv_smem[((base + 512) + 1)], (w_peer1[0] if False else w_peer0[0]), acc[5], rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
                acc[5] = cutlass.Float32(_fma_77)
                _fma_78 = cute.math.fma(_recv_smem[((base + 512) + 2)], (w_peer1[0] if True else w_peer0[0]), acc[6], rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
                acc[6] = cutlass.Float32(_fma_78)
                _fma_79 = cute.math.fma(_recv_smem[((base + 512) + 3)], (w_peer1[0] if True else w_peer0[0]), acc[7], rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
                acc[7] = cutlass.Float32(_fma_79)
                _fma_80 = cute.math.fma(_recv_smem[(base + 1024)], (w_peer1[0] if False else w_peer0[0]), acc[8], rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
                acc[8] = cutlass.Float32(_fma_80)
                _fma_81 = cute.math.fma(_recv_smem[((base + 1024) + 1)], (w_peer1[0] if False else w_peer0[0]), acc[9], rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
                acc[9] = cutlass.Float32(_fma_81)
                _fma_82 = cute.math.fma(_recv_smem[((base + 1024) + 2)], (w_peer1[0] if True else w_peer0[0]), acc[10], rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
                acc[10] = cutlass.Float32(_fma_82)
                _fma_83 = cute.math.fma(_recv_smem[((base + 1024) + 3)], (w_peer1[0] if True else w_peer0[0]), acc[11], rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
                acc[11] = cutlass.Float32(_fma_83)
                _fma_84 = cute.math.fma(_recv_smem[(base + 1536)], (w_peer1[0] if False else w_peer0[0]), acc[12], rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
                acc[12] = cutlass.Float32(_fma_84)
                _fma_85 = cute.math.fma(_recv_smem[((base + 1536) + 1)], (w_peer1[0] if False else w_peer0[0]), acc[13], rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
                acc[13] = cutlass.Float32(_fma_85)
                _fma_86 = cute.math.fma(_recv_smem[((base + 1536) + 2)], (w_peer1[0] if True else w_peer0[0]), acc[14], rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
                acc[14] = cutlass.Float32(_fma_86)
                _fma_87 = cute.math.fma(_recv_smem[((base + 1536) + 3)], (w_peer1[0] if True else w_peer0[0]), acc[15], rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
                acc[15] = cutlass.Float32(_fma_87)
                base_0 = cutlass.Int32((2304 + my_tid4))
                _fma_88 = cute.math.fma(_recv_smem[base_0], (w_peer1[1] if False else w_peer0[1]), acc[0], rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
                acc[0] = cutlass.Float32(_fma_88)
                _fma_89 = cute.math.fma(_recv_smem[(base_0 + 1)], (w_peer1[1] if False else w_peer0[1]), acc[1], rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
                acc[1] = cutlass.Float32(_fma_89)
                _fma_90 = cute.math.fma(_recv_smem[(base_0 + 2)], (w_peer1[1] if True else w_peer0[1]), acc[2], rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
                acc[2] = cutlass.Float32(_fma_90)
                _fma_91 = cute.math.fma(_recv_smem[(base_0 + 3)], (w_peer1[1] if True else w_peer0[1]), acc[3], rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
                acc[3] = cutlass.Float32(_fma_91)
                _fma_92 = cute.math.fma(_recv_smem[(base_0 + 512)], (w_peer1[1] if False else w_peer0[1]), acc[4], rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
                acc[4] = cutlass.Float32(_fma_92)
                _fma_93 = cute.math.fma(_recv_smem[((base_0 + 512) + 1)], (w_peer1[1] if False else w_peer0[1]), acc[5], rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
                acc[5] = cutlass.Float32(_fma_93)
                _fma_94 = cute.math.fma(_recv_smem[((base_0 + 512) + 2)], (w_peer1[1] if True else w_peer0[1]), acc[6], rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
                acc[6] = cutlass.Float32(_fma_94)
                _fma_95 = cute.math.fma(_recv_smem[((base_0 + 512) + 3)], (w_peer1[1] if True else w_peer0[1]), acc[7], rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
                acc[7] = cutlass.Float32(_fma_95)
                _fma_96 = cute.math.fma(_recv_smem[(base_0 + 1024)], (w_peer1[1] if False else w_peer0[1]), acc[8], rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
                acc[8] = cutlass.Float32(_fma_96)
                _fma_97 = cute.math.fma(_recv_smem[((base_0 + 1024) + 1)], (w_peer1[1] if False else w_peer0[1]), acc[9], rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
                acc[9] = cutlass.Float32(_fma_97)
                _fma_98 = cute.math.fma(_recv_smem[((base_0 + 1024) + 2)], (w_peer1[1] if True else w_peer0[1]), acc[10], rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
                acc[10] = cutlass.Float32(_fma_98)
                _fma_99 = cute.math.fma(_recv_smem[((base_0 + 1024) + 3)], (w_peer1[1] if True else w_peer0[1]), acc[11], rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
                acc[11] = cutlass.Float32(_fma_99)
                _fma_100 = cute.math.fma(_recv_smem[(base_0 + 1536)], (w_peer1[1] if False else w_peer0[1]), acc[12], rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
                acc[12] = cutlass.Float32(_fma_100)
                _fma_101 = cute.math.fma(_recv_smem[((base_0 + 1536) + 1)], (w_peer1[1] if False else w_peer0[1]), acc[13], rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
                acc[13] = cutlass.Float32(_fma_101)
                _fma_102 = cute.math.fma(_recv_smem[((base_0 + 1536) + 2)], (w_peer1[1] if True else w_peer0[1]), acc[14], rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
                acc[14] = cutlass.Float32(_fma_102)
                _fma_103 = cute.math.fma(_recv_smem[((base_0 + 1536) + 3)], (w_peer1[1] if True else w_peer0[1]), acc[15], rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
                acc[15] = cutlass.Float32(_fma_103)
                base_1 = cutlass.Int32((4608 + my_tid4))
                _fma_104 = cute.math.fma(_recv_smem[base_1], (w_peer1[2] if False else w_peer0[2]), acc[0], rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
                acc[0] = cutlass.Float32(_fma_104)
                _fma_105 = cute.math.fma(_recv_smem[(base_1 + 1)], (w_peer1[2] if False else w_peer0[2]), acc[1], rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
                acc[1] = cutlass.Float32(_fma_105)
                _fma_106 = cute.math.fma(_recv_smem[(base_1 + 2)], (w_peer1[2] if True else w_peer0[2]), acc[2], rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
                acc[2] = cutlass.Float32(_fma_106)
                _fma_107 = cute.math.fma(_recv_smem[(base_1 + 3)], (w_peer1[2] if True else w_peer0[2]), acc[3], rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
                acc[3] = cutlass.Float32(_fma_107)
                _fma_108 = cute.math.fma(_recv_smem[(base_1 + 512)], (w_peer1[2] if False else w_peer0[2]), acc[4], rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
                acc[4] = cutlass.Float32(_fma_108)
                _fma_109 = cute.math.fma(_recv_smem[((base_1 + 512) + 1)], (w_peer1[2] if False else w_peer0[2]), acc[5], rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
                acc[5] = cutlass.Float32(_fma_109)
                _fma_110 = cute.math.fma(_recv_smem[((base_1 + 512) + 2)], (w_peer1[2] if True else w_peer0[2]), acc[6], rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
                acc[6] = cutlass.Float32(_fma_110)
                _fma_111 = cute.math.fma(_recv_smem[((base_1 + 512) + 3)], (w_peer1[2] if True else w_peer0[2]), acc[7], rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
                acc[7] = cutlass.Float32(_fma_111)
                _fma_112 = cute.math.fma(_recv_smem[(base_1 + 1024)], (w_peer1[2] if False else w_peer0[2]), acc[8], rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
                acc[8] = cutlass.Float32(_fma_112)
                _fma_113 = cute.math.fma(_recv_smem[((base_1 + 1024) + 1)], (w_peer1[2] if False else w_peer0[2]), acc[9], rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
                acc[9] = cutlass.Float32(_fma_113)
                _fma_114 = cute.math.fma(_recv_smem[((base_1 + 1024) + 2)], (w_peer1[2] if True else w_peer0[2]), acc[10], rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
                acc[10] = cutlass.Float32(_fma_114)
                _fma_115 = cute.math.fma(_recv_smem[((base_1 + 1024) + 3)], (w_peer1[2] if True else w_peer0[2]), acc[11], rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
                acc[11] = cutlass.Float32(_fma_115)
                _fma_116 = cute.math.fma(_recv_smem[(base_1 + 1536)], (w_peer1[2] if False else w_peer0[2]), acc[12], rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
                acc[12] = cutlass.Float32(_fma_116)
                _fma_117 = cute.math.fma(_recv_smem[((base_1 + 1536) + 1)], (w_peer1[2] if False else w_peer0[2]), acc[13], rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
                acc[13] = cutlass.Float32(_fma_117)
                _fma_118 = cute.math.fma(_recv_smem[((base_1 + 1536) + 2)], (w_peer1[2] if True else w_peer0[2]), acc[14], rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
                acc[14] = cutlass.Float32(_fma_118)
                _fma_119 = cute.math.fma(_recv_smem[((base_1 + 1536) + 3)], (w_peer1[2] if True else w_peer0[2]), acc[15], rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
                acc[15] = cutlass.Float32(_fma_119)
                _rcp_0 = cute.math.rcp(msum0[0], approx=True, ftz=True)
                _rcp_1 = cute.math.rcp(msum1[0], approx=True, ftz=True)
                acc[0] = cutlass.Float32((acc[0] * (_rcp_1 if False else _rcp_0)))
                acc[1] = cutlass.Float32((acc[1] * (_rcp_1 if False else _rcp_0)))
                acc[2] = cutlass.Float32((acc[2] * (_rcp_1 if True else _rcp_0)))
                acc[3] = cutlass.Float32((acc[3] * (_rcp_1 if True else _rcp_0)))
                acc[4] = cutlass.Float32((acc[4] * (_rcp_1 if False else _rcp_0)))
                acc[5] = cutlass.Float32((acc[5] * (_rcp_1 if False else _rcp_0)))
                acc[6] = cutlass.Float32((acc[6] * (_rcp_1 if True else _rcp_0)))
                acc[7] = cutlass.Float32((acc[7] * (_rcp_1 if True else _rcp_0)))
                acc[8] = cutlass.Float32((acc[8] * (_rcp_1 if False else _rcp_0)))
                acc[9] = cutlass.Float32((acc[9] * (_rcp_1 if False else _rcp_0)))
                acc[10] = cutlass.Float32((acc[10] * (_rcp_1 if True else _rcp_0)))
                acc[11] = cutlass.Float32((acc[11] * (_rcp_1 if True else _rcp_0)))
                acc[12] = cutlass.Float32((acc[12] * (_rcp_1 if False else _rcp_0)))
                acc[13] = cutlass.Float32((acc[13] * (_rcp_1 if False else _rcp_0)))
                acc[14] = cutlass.Float32((acc[14] * (_rcp_1 if True else _rcp_0)))
                acc[15] = cutlass.Float32((acc[15] * (_rcp_1 if True else _rcp_0)))
                qj1 = cutlass.Int32((qj & 1))
                qj2 = cutlass.Int32((qj & 2))
                o_vec = cute.make_rmem_tensor((4,), cutlass.Uint32)
                o_tmp = cute.make_rmem_tensor((4,), cutlass.Uint32)
                o_row_base = cutlass.Int32((q_row * 128))
                m_local_r = cutlass.Int32((m0_local if True else m1_local))
                _bf16x2_32 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(acc[0]), cutlass.Float32(acc[1])))[1]), cutlass.Float32(((cutlass.Float32(acc[0]), cutlass.Float32(acc[1])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
                o_vec[0] = cutlass.Uint32(_bf16x2_32)
                _bf16x2_33 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(acc[4]), cutlass.Float32(acc[5])))[1]), cutlass.Float32(((cutlass.Float32(acc[4]), cutlass.Float32(acc[5])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
                o_vec[1] = cutlass.Uint32(_bf16x2_33)
                _bf16x2_34 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(acc[8]), cutlass.Float32(acc[9])))[1]), cutlass.Float32(((cutlass.Float32(acc[8]), cutlass.Float32(acc[9])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
                o_vec[2] = cutlass.Uint32(_bf16x2_34)
                _bf16x2_35 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(acc[12]), cutlass.Float32(acc[13])))[1]), cutlass.Float32(((cutlass.Float32(acc[12]), cutlass.Float32(acc[13])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
                o_vec[3] = cutlass.Uint32(_bf16x2_35)
                _shfl_xor_12 = cute.arch.shuffle_sync_bfly(o_vec[1], 1, mask=0xFFFFFFFF, mask_and_clamp=31)
                o_tmp[0] = cutlass.Uint32(_shfl_xor_12)
                _shfl_xor_13 = cute.arch.shuffle_sync_bfly(o_vec[0], 1, mask=0xFFFFFFFF, mask_and_clamp=31)
                o_tmp[1] = cutlass.Uint32(_shfl_xor_13)
                _shfl_xor_14 = cute.arch.shuffle_sync_bfly(o_vec[3], 1, mask=0xFFFFFFFF, mask_and_clamp=31)
                o_tmp[2] = cutlass.Uint32(_shfl_xor_14)
                _shfl_xor_15 = cute.arch.shuffle_sync_bfly(o_vec[2], 1, mask=0xFFFFFFFF, mask_and_clamp=31)
                o_tmp[3] = cutlass.Uint32(_shfl_xor_15)
                o_vec[0] = cutlass.Uint32((o_tmp[0] if (qj1 != 0) else o_vec[0]))
                o_vec[1] = cutlass.Uint32((o_tmp[1] if (qj1 == 0) else o_vec[1]))
                o_vec[2] = cutlass.Uint32((o_tmp[2] if (qj1 != 0) else o_vec[2]))
                o_vec[3] = cutlass.Uint32((o_tmp[3] if (qj1 == 0) else o_vec[3]))
                _shfl_xor_16 = cute.arch.shuffle_sync_bfly(o_vec[2], 2, mask=0xFFFFFFFF, mask_and_clamp=31)
                o_tmp[0] = cutlass.Uint32(_shfl_xor_16)
                _shfl_xor_17 = cute.arch.shuffle_sync_bfly(o_vec[3], 2, mask=0xFFFFFFFF, mask_and_clamp=31)
                o_tmp[1] = cutlass.Uint32(_shfl_xor_17)
                _shfl_xor_18 = cute.arch.shuffle_sync_bfly(o_vec[0], 2, mask=0xFFFFFFFF, mask_and_clamp=31)
                o_tmp[2] = cutlass.Uint32(_shfl_xor_18)
                _shfl_xor_19 = cute.arch.shuffle_sync_bfly(o_vec[1], 2, mask=0xFFFFFFFF, mask_and_clamp=31)
                o_tmp[3] = cutlass.Uint32(_shfl_xor_19)
                o_vec[0] = cutlass.Uint32((o_tmp[0] if (qj2 != 0) else o_vec[0]))
                o_vec[1] = cutlass.Uint32((o_tmp[1] if (qj2 != 0) else o_vec[1]))
                o_vec[2] = cutlass.Uint32((o_tmp[2] if (qj2 == 0) else o_vec[2]))
                o_vec[3] = cutlass.Uint32((o_tmp[3] if (qj2 == 0) else o_vec[3]))
                o_off = cutlass.Int32(((o_row_base + (m_local_r * 128)) + (((4 * sset) + qj) * 8)))
                _gmem_store_raw_40 = cutlass.Vector.from_elements([cutlass.Uint32(o_vec[0]), cutlass.Uint32(o_vec[1]), cutlass.Uint32(o_vec[2]), cutlass.Uint32(o_vec[3])], cutlass.Uint32)
                prims.store_ext(_gmem_store_raw_40.ir_value(), O + o_off)
                m_local_r_2 = cutlass.Int32((m0_local if False else m1_local))
                _bf16x2_36 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(acc[2]), cutlass.Float32(acc[3])))[1]), cutlass.Float32(((cutlass.Float32(acc[2]), cutlass.Float32(acc[3])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
                o_vec[0] = cutlass.Uint32(_bf16x2_36)
                _bf16x2_37 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(acc[6]), cutlass.Float32(acc[7])))[1]), cutlass.Float32(((cutlass.Float32(acc[6]), cutlass.Float32(acc[7])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
                o_vec[1] = cutlass.Uint32(_bf16x2_37)
                _bf16x2_38 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(acc[10]), cutlass.Float32(acc[11])))[1]), cutlass.Float32(((cutlass.Float32(acc[10]), cutlass.Float32(acc[11])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
                o_vec[2] = cutlass.Uint32(_bf16x2_38)
                _bf16x2_39 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(acc[14]), cutlass.Float32(acc[15])))[1]), cutlass.Float32(((cutlass.Float32(acc[14]), cutlass.Float32(acc[15])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
                o_vec[3] = cutlass.Uint32(_bf16x2_39)
                _shfl_xor_20 = cute.arch.shuffle_sync_bfly(o_vec[1], 1, mask=0xFFFFFFFF, mask_and_clamp=31)
                o_tmp[0] = cutlass.Uint32(_shfl_xor_20)
                _shfl_xor_21 = cute.arch.shuffle_sync_bfly(o_vec[0], 1, mask=0xFFFFFFFF, mask_and_clamp=31)
                o_tmp[1] = cutlass.Uint32(_shfl_xor_21)
                _shfl_xor_22 = cute.arch.shuffle_sync_bfly(o_vec[3], 1, mask=0xFFFFFFFF, mask_and_clamp=31)
                o_tmp[2] = cutlass.Uint32(_shfl_xor_22)
                _shfl_xor_23 = cute.arch.shuffle_sync_bfly(o_vec[2], 1, mask=0xFFFFFFFF, mask_and_clamp=31)
                o_tmp[3] = cutlass.Uint32(_shfl_xor_23)
                o_vec[0] = cutlass.Uint32((o_tmp[0] if (qj1 != 0) else o_vec[0]))
                o_vec[1] = cutlass.Uint32((o_tmp[1] if (qj1 == 0) else o_vec[1]))
                o_vec[2] = cutlass.Uint32((o_tmp[2] if (qj1 != 0) else o_vec[2]))
                o_vec[3] = cutlass.Uint32((o_tmp[3] if (qj1 == 0) else o_vec[3]))
                _shfl_xor_24 = cute.arch.shuffle_sync_bfly(o_vec[2], 2, mask=0xFFFFFFFF, mask_and_clamp=31)
                o_tmp[0] = cutlass.Uint32(_shfl_xor_24)
                _shfl_xor_25 = cute.arch.shuffle_sync_bfly(o_vec[3], 2, mask=0xFFFFFFFF, mask_and_clamp=31)
                o_tmp[1] = cutlass.Uint32(_shfl_xor_25)
                _shfl_xor_26 = cute.arch.shuffle_sync_bfly(o_vec[0], 2, mask=0xFFFFFFFF, mask_and_clamp=31)
                o_tmp[2] = cutlass.Uint32(_shfl_xor_26)
                _shfl_xor_27 = cute.arch.shuffle_sync_bfly(o_vec[1], 2, mask=0xFFFFFFFF, mask_and_clamp=31)
                o_tmp[3] = cutlass.Uint32(_shfl_xor_27)
                o_vec[0] = cutlass.Uint32((o_tmp[0] if (qj2 != 0) else o_vec[0]))
                o_vec[1] = cutlass.Uint32((o_tmp[1] if (qj2 != 0) else o_vec[1]))
                o_vec[2] = cutlass.Uint32((o_tmp[2] if (qj2 == 0) else o_vec[2]))
                o_vec[3] = cutlass.Uint32((o_tmp[3] if (qj2 == 0) else o_vec[3]))
                o_off_3 = cutlass.Int32(((o_row_base + (m_local_r_2 * 128)) + (((4 * sset) + qj) * 8)))
                _gmem_store_raw_41 = cutlass.Vector.from_elements([cutlass.Uint32(o_vec[0]), cutlass.Uint32(o_vec[1]), cutlass.Uint32(o_vec[2]), cutlass.Uint32(o_vec[3])], cutlass.Uint32)
                prims.store_ext(_gmem_store_raw_41.ir_value(), O + o_off_3)
    if (wg == 1):
        my_rank_1 = cutlass.Int32(cta_rank)
        my_off_1 = cutlass.Int32((tid_wg * 16))
        if (my_rank_1 != 0):
            pslot_4[0] = cutlass.Int32(my_rank_1)
            _if_condition_42 = cutlass.Boolean((my_rank_1 > 0))
            pslot_4[0] = cutlass.Int32(cutlass.select_(_if_condition_42, cutlass.Int32((my_rank_1 - 1)), pslot_4[0]))
            _cluster_mapa_43 = cute.arch.map_dsmem_ptr(recv_full_addr, cutlass.Int32(0))
            _mapa_10 = cutlass.Uint32(_cluster_mapa_43.toint())
        if (my_rank_1 != 1):
            pslot_5[0] = cutlass.Int32(my_rank_1)
            _if_condition_44 = cutlass.Boolean((my_rank_1 > 1))
            pslot_5[0] = cutlass.Int32(cutlass.select_(_if_condition_44, cutlass.Int32((my_rank_1 - 1)), pslot_5[0]))
            _cluster_mapa_45 = cute.arch.map_dsmem_ptr(recv_full_addr, cutlass.Int32(1))
            _mapa_11 = cutlass.Uint32(_cluster_mapa_45.toint())
        if (my_rank_1 != 2):
            pslot_6[0] = cutlass.Int32(my_rank_1)
            _if_condition_46 = cutlass.Boolean((my_rank_1 > 2))
            pslot_6[0] = cutlass.Int32(cutlass.select_(_if_condition_46, cutlass.Int32((my_rank_1 - 1)), pslot_6[0]))
            _cluster_mapa_47 = cute.arch.map_dsmem_ptr(recv_full_addr, cutlass.Int32(2))
            _mapa_12 = cutlass.Uint32(_cluster_mapa_47.toint())
            _cluster_mapa_48 = cute.arch.map_dsmem_ptr(cute.make_ptr(cutlass.Uint8, cutlass.Uint32(((recv_smem_addr + cutlass.Uint32((pslot_6[0] * 9216))) + cutlass.Uint32(my_off_1))), mem_space=cute.AddressSpace.smem, assumed_align=4), cutlass.Int32(2))
            _mapa_13 = cutlass.Uint32(_cluster_mapa_48.toint())
            cutlass_llvm.inline_asm(
                res=None,
                operands_=[(cutlass.Uint32(_mapa_13)).ir_value(), (cutlass.Float32(d_o[32]).bitcast(cutlass.Int32)).ir_value(), (cutlass.Float32(d_o[33]).bitcast(cutlass.Int32)).ir_value(), (cutlass.Float32(d_o[34]).bitcast(cutlass.Int32)).ir_value(), (cutlass.Float32(d_o[35]).bitcast(cutlass.Int32)).ir_value(), (cutlass.Uint32(_mapa_12)).ir_value()],
                asm_string='st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [$0], {$1, $2, $3, $4}, [$5];',
                constraints='r,r,r,r,r,r,~{memory}',
                has_side_effects=True,
                is_align_stack=False,
                asm_dialect=cutlass_llvm.AsmDialect.AD_ATT,
            )
            cutlass_llvm.inline_asm(
                res=None,
                operands_=[(cutlass.Uint32((_mapa_13 + 2048))).ir_value(), (cutlass.Float32(d_o[36]).bitcast(cutlass.Int32)).ir_value(), (cutlass.Float32(d_o[37]).bitcast(cutlass.Int32)).ir_value(), (cutlass.Float32(d_o[38]).bitcast(cutlass.Int32)).ir_value(), (cutlass.Float32(d_o[39]).bitcast(cutlass.Int32)).ir_value(), (cutlass.Uint32(_mapa_12)).ir_value()],
                asm_string='st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [$0], {$1, $2, $3, $4}, [$5];',
                constraints='r,r,r,r,r,r,~{memory}',
                has_side_effects=True,
                is_align_stack=False,
                asm_dialect=cutlass_llvm.AsmDialect.AD_ATT,
            )
            cutlass_llvm.inline_asm(
                res=None,
                operands_=[(cutlass.Uint32((_mapa_13 + 4096))).ir_value(), (cutlass.Float32(d_o[40]).bitcast(cutlass.Int32)).ir_value(), (cutlass.Float32(d_o[41]).bitcast(cutlass.Int32)).ir_value(), (cutlass.Float32(d_o[42]).bitcast(cutlass.Int32)).ir_value(), (cutlass.Float32(d_o[43]).bitcast(cutlass.Int32)).ir_value(), (cutlass.Uint32(_mapa_12)).ir_value()],
                asm_string='st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [$0], {$1, $2, $3, $4}, [$5];',
                constraints='r,r,r,r,r,r,~{memory}',
                has_side_effects=True,
                is_align_stack=False,
                asm_dialect=cutlass_llvm.AsmDialect.AD_ATT,
            )
            cutlass_llvm.inline_asm(
                res=None,
                operands_=[(cutlass.Uint32((_mapa_13 + 6144))).ir_value(), (cutlass.Float32(d_o[44]).bitcast(cutlass.Int32)).ir_value(), (cutlass.Float32(d_o[45]).bitcast(cutlass.Int32)).ir_value(), (cutlass.Float32(d_o[46]).bitcast(cutlass.Int32)).ir_value(), (cutlass.Float32(d_o[47]).bitcast(cutlass.Int32)).ir_value(), (cutlass.Uint32(_mapa_12)).ir_value()],
                asm_string='st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [$0], {$1, $2, $3, $4}, [$5];',
                constraints='r,r,r,r,r,r,~{memory}',
                has_side_effects=True,
                is_align_stack=False,
                asm_dialect=cutlass_llvm.AsmDialect.AD_ATT,
            )
        if (my_rank_1 != 3):
            pslot_7[0] = cutlass.Int32(my_rank_1)
            _if_condition_49 = cutlass.Boolean((my_rank_1 > 3))
            pslot_7[0] = cutlass.Int32(cutlass.select_(_if_condition_49, cutlass.Int32((my_rank_1 - 1)), pslot_7[0]))
            _cluster_mapa_50 = cute.arch.map_dsmem_ptr(recv_full_addr, cutlass.Int32(3))
            _mapa_14 = cutlass.Uint32(_cluster_mapa_50.toint())
            _cluster_mapa_51 = cute.arch.map_dsmem_ptr(cute.make_ptr(cutlass.Uint8, cutlass.Uint32(((recv_smem_addr + cutlass.Uint32((pslot_7[0] * 9216))) + cutlass.Uint32(my_off_1))), mem_space=cute.AddressSpace.smem, assumed_align=4), cutlass.Int32(3))
            _mapa_15 = cutlass.Uint32(_cluster_mapa_51.toint())
            cutlass_llvm.inline_asm(
                res=None,
                operands_=[(cutlass.Uint32(_mapa_15)).ir_value(), (cutlass.Float32(d_o[48]).bitcast(cutlass.Int32)).ir_value(), (cutlass.Float32(d_o[49]).bitcast(cutlass.Int32)).ir_value(), (cutlass.Float32(d_o[50]).bitcast(cutlass.Int32)).ir_value(), (cutlass.Float32(d_o[51]).bitcast(cutlass.Int32)).ir_value(), (cutlass.Uint32(_mapa_14)).ir_value()],
                asm_string='st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [$0], {$1, $2, $3, $4}, [$5];',
                constraints='r,r,r,r,r,r,~{memory}',
                has_side_effects=True,
                is_align_stack=False,
                asm_dialect=cutlass_llvm.AsmDialect.AD_ATT,
            )
            cutlass_llvm.inline_asm(
                res=None,
                operands_=[(cutlass.Uint32((_mapa_15 + 2048))).ir_value(), (cutlass.Float32(d_o[52]).bitcast(cutlass.Int32)).ir_value(), (cutlass.Float32(d_o[53]).bitcast(cutlass.Int32)).ir_value(), (cutlass.Float32(d_o[54]).bitcast(cutlass.Int32)).ir_value(), (cutlass.Float32(d_o[55]).bitcast(cutlass.Int32)).ir_value(), (cutlass.Uint32(_mapa_14)).ir_value()],
                asm_string='st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [$0], {$1, $2, $3, $4}, [$5];',
                constraints='r,r,r,r,r,r,~{memory}',
                has_side_effects=True,
                is_align_stack=False,
                asm_dialect=cutlass_llvm.AsmDialect.AD_ATT,
            )
            cutlass_llvm.inline_asm(
                res=None,
                operands_=[(cutlass.Uint32((_mapa_15 + 4096))).ir_value(), (cutlass.Float32(d_o[56]).bitcast(cutlass.Int32)).ir_value(), (cutlass.Float32(d_o[57]).bitcast(cutlass.Int32)).ir_value(), (cutlass.Float32(d_o[58]).bitcast(cutlass.Int32)).ir_value(), (cutlass.Float32(d_o[59]).bitcast(cutlass.Int32)).ir_value(), (cutlass.Uint32(_mapa_14)).ir_value()],
                asm_string='st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [$0], {$1, $2, $3, $4}, [$5];',
                constraints='r,r,r,r,r,r,~{memory}',
                has_side_effects=True,
                is_align_stack=False,
                asm_dialect=cutlass_llvm.AsmDialect.AD_ATT,
            )
            cutlass_llvm.inline_asm(
                res=None,
                operands_=[(cutlass.Uint32((_mapa_15 + 6144))).ir_value(), (cutlass.Float32(d_o[60]).bitcast(cutlass.Int32)).ir_value(), (cutlass.Float32(d_o[61]).bitcast(cutlass.Int32)).ir_value(), (cutlass.Float32(d_o[62]).bitcast(cutlass.Int32)).ir_value(), (cutlass.Float32(d_o[63]).bitcast(cutlass.Int32)).ir_value(), (cutlass.Uint32(_mapa_14)).ir_value()],
                asm_string='st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [$0], {$1, $2, $3, $4}, [$5];',
                constraints='r,r,r,r,r,r,~{memory}',
                has_side_effects=True,
                is_align_stack=False,
                asm_dialect=cutlass_llvm.AsmDialect.AD_ATT,
            )
        if (my_rank_1 < 4):
            while not prims.mbarrier_try_wait_parity(recv_full_addr, _phase_recv_full_0[0], time_limit=10000000, scope=prims.MBarrierScope.CLUSTER, order=prims.MemOrder.ACQUIRE):
                pass
            _phase_recv_full_0[0] ^= cutlass.Uint32(1)
            merged_max0_1[0] = cutlass.Float32(row_max0[0])
            merged_max1_1[0] = cutlass.Float32(row_max1[0])
            _max_84 = cute.arch.fmax(merged_max0_1[0], _recv_smem[(2048 + (m0_local * 4))], ftz=False)
            merged_max0_1[0] = cutlass.Float32(_max_84)
            _max_85 = cute.arch.fmax(merged_max1_1[0], _recv_smem[((2048 + (m0_local * 4)) + 2)], ftz=False)
            merged_max1_1[0] = cutlass.Float32(_max_85)
            _max_86 = cute.arch.fmax(merged_max0_1[0], _recv_smem[(4352 + (m0_local * 4))], ftz=False)
            merged_max0_1[0] = cutlass.Float32(_max_86)
            _max_87 = cute.arch.fmax(merged_max1_1[0], _recv_smem[((4352 + (m0_local * 4)) + 2)], ftz=False)
            merged_max1_1[0] = cutlass.Float32(_max_87)
            _max_88 = cute.arch.fmax(merged_max0_1[0], _recv_smem[(6656 + (m0_local * 4))], ftz=False)
            merged_max0_1[0] = cutlass.Float32(_max_88)
            _max_89 = cute.arch.fmax(merged_max1_1[0], _recv_smem[((6656 + (m0_local * 4)) + 2)], ftz=False)
            merged_max1_1[0] = cutlass.Float32(_max_89)
            _exp2_80 = cute.math.exp2((row_max0[0] - merged_max0_1[0]), approx=True, ftz=True)
            _exp2_81 = cute.math.exp2((row_max1[0] - merged_max1_1[0]), approx=True, ftz=True)
            msum0_1[0] = cutlass.Float32((row_sum0[0] * _exp2_80))
            msum1_1[0] = cutlass.Float32((row_sum1[0] * _exp2_81))
            w_peer0_1 = cute.make_rmem_tensor((3,), cutlass.Float32)
            w_peer1_1 = cute.make_rmem_tensor((3,), cutlass.Float32)
            sbase_2 = cutlass.Int32((2048 + (m0_local * 4)))
            _exp2_82 = cute.math.exp2((_recv_smem[sbase_2] - merged_max0_1[0]), approx=True, ftz=True)
            w_peer0_1[0] = cutlass.Float32(_exp2_82)
            _exp2_83 = cute.math.exp2((_recv_smem[(sbase_2 + 2)] - merged_max1_1[0]), approx=True, ftz=True)
            w_peer1_1[0] = cutlass.Float32(_exp2_83)
            _fma_120 = cute.math.fma(_recv_smem[(sbase_2 + 1)], w_peer0_1[0], msum0_1[0], rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
            msum0_1[0] = cutlass.Float32(_fma_120)
            _fma_121 = cute.math.fma(_recv_smem[(sbase_2 + 3)], w_peer1_1[0], msum1_1[0], rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
            msum1_1[0] = cutlass.Float32(_fma_121)
            sbase_0_1 = cutlass.Int32((4352 + (m0_local * 4)))
            _exp2_84 = cute.math.exp2((_recv_smem[sbase_0_1] - merged_max0_1[0]), approx=True, ftz=True)
            w_peer0_1[1] = cutlass.Float32(_exp2_84)
            _exp2_85 = cute.math.exp2((_recv_smem[(sbase_0_1 + 2)] - merged_max1_1[0]), approx=True, ftz=True)
            w_peer1_1[1] = cutlass.Float32(_exp2_85)
            _fma_122 = cute.math.fma(_recv_smem[(sbase_0_1 + 1)], w_peer0_1[1], msum0_1[0], rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
            msum0_1[0] = cutlass.Float32(_fma_122)
            _fma_123 = cute.math.fma(_recv_smem[(sbase_0_1 + 3)], w_peer1_1[1], msum1_1[0], rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
            msum1_1[0] = cutlass.Float32(_fma_123)
            sbase_1_1 = cutlass.Int32((6656 + (m0_local * 4)))
            _exp2_86 = cute.math.exp2((_recv_smem[sbase_1_1] - merged_max0_1[0]), approx=True, ftz=True)
            w_peer0_1[2] = cutlass.Float32(_exp2_86)
            _exp2_87 = cute.math.exp2((_recv_smem[(sbase_1_1 + 2)] - merged_max1_1[0]), approx=True, ftz=True)
            w_peer1_1[2] = cutlass.Float32(_exp2_87)
            _fma_124 = cute.math.fma(_recv_smem[(sbase_1_1 + 1)], w_peer0_1[2], msum0_1[0], rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
            msum0_1[0] = cutlass.Float32(_fma_124)
            _fma_125 = cute.math.fma(_recv_smem[(sbase_1_1 + 3)], w_peer1_1[2], msum1_1[0], rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
            msum1_1[0] = cutlass.Float32(_fma_125)
            my_tid4_1 = cutlass.Int32((tid_wg * 4))
            acc_1 = cute.make_rmem_tensor((16,), cutlass.Float32)
            sset_1 = cutlass.Int32(my_rank_1)
            fold_1[0] = cutlass.Int32((1 if (sset_1 < 4) else 0))
            _if_condition_52 = cutlass.Boolean((sset_1 < 2))
            fold_1[0] = cutlass.Int32(cutlass.select_(_if_condition_52, cutlass.Int32(0), fold_1[0]))
            _if_condition_53 = cutlass.Boolean((sset_1 >= 4))
            fold_1[0] = cutlass.Int32(cutlass.select_(_if_condition_53, cutlass.Int32(0), fold_1[0]))
            if (fold_1[0] != 0):
                if (sset_1 == 2):
                    acc_1[0] = cutlass.Float32((d_o[32] * (_exp2_81 if False else _exp2_80)))
                    acc_1[1] = cutlass.Float32((d_o[33] * (_exp2_81 if False else _exp2_80)))
                    acc_1[2] = cutlass.Float32((d_o[34] * (_exp2_81 if True else _exp2_80)))
                    acc_1[3] = cutlass.Float32((d_o[35] * (_exp2_81 if True else _exp2_80)))
                    acc_1[4] = cutlass.Float32((d_o[36] * (_exp2_81 if False else _exp2_80)))
                    acc_1[5] = cutlass.Float32((d_o[37] * (_exp2_81 if False else _exp2_80)))
                    acc_1[6] = cutlass.Float32((d_o[38] * (_exp2_81 if True else _exp2_80)))
                    acc_1[7] = cutlass.Float32((d_o[39] * (_exp2_81 if True else _exp2_80)))
                    acc_1[8] = cutlass.Float32((d_o[40] * (_exp2_81 if False else _exp2_80)))
                    acc_1[9] = cutlass.Float32((d_o[41] * (_exp2_81 if False else _exp2_80)))
                    acc_1[10] = cutlass.Float32((d_o[42] * (_exp2_81 if True else _exp2_80)))
                    acc_1[11] = cutlass.Float32((d_o[43] * (_exp2_81 if True else _exp2_80)))
                    acc_1[12] = cutlass.Float32((d_o[44] * (_exp2_81 if False else _exp2_80)))
                    acc_1[13] = cutlass.Float32((d_o[45] * (_exp2_81 if False else _exp2_80)))
                    acc_1[14] = cutlass.Float32((d_o[46] * (_exp2_81 if True else _exp2_80)))
                    acc_1[15] = cutlass.Float32((d_o[47] * (_exp2_81 if True else _exp2_80)))
                if (sset_1 == 3):
                    acc_1[0] = cutlass.Float32((d_o[48] * (_exp2_81 if False else _exp2_80)))
                    acc_1[1] = cutlass.Float32((d_o[49] * (_exp2_81 if False else _exp2_80)))
                    acc_1[2] = cutlass.Float32((d_o[50] * (_exp2_81 if True else _exp2_80)))
                    acc_1[3] = cutlass.Float32((d_o[51] * (_exp2_81 if True else _exp2_80)))
                    acc_1[4] = cutlass.Float32((d_o[52] * (_exp2_81 if False else _exp2_80)))
                    acc_1[5] = cutlass.Float32((d_o[53] * (_exp2_81 if False else _exp2_80)))
                    acc_1[6] = cutlass.Float32((d_o[54] * (_exp2_81 if True else _exp2_80)))
                    acc_1[7] = cutlass.Float32((d_o[55] * (_exp2_81 if True else _exp2_80)))
                    acc_1[8] = cutlass.Float32((d_o[56] * (_exp2_81 if False else _exp2_80)))
                    acc_1[9] = cutlass.Float32((d_o[57] * (_exp2_81 if False else _exp2_80)))
                    acc_1[10] = cutlass.Float32((d_o[58] * (_exp2_81 if True else _exp2_80)))
                    acc_1[11] = cutlass.Float32((d_o[59] * (_exp2_81 if True else _exp2_80)))
                    acc_1[12] = cutlass.Float32((d_o[60] * (_exp2_81 if False else _exp2_80)))
                    acc_1[13] = cutlass.Float32((d_o[61] * (_exp2_81 if False else _exp2_80)))
                    acc_1[14] = cutlass.Float32((d_o[62] * (_exp2_81 if True else _exp2_80)))
                    acc_1[15] = cutlass.Float32((d_o[63] * (_exp2_81 if True else _exp2_80)))
                base_2 = cutlass.Int32(my_tid4_1)
                _fma_126 = cute.math.fma(_recv_smem[base_2], (w_peer1_1[0] if False else w_peer0_1[0]), acc_1[0], rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
                acc_1[0] = cutlass.Float32(_fma_126)
                _fma_127 = cute.math.fma(_recv_smem[(base_2 + 1)], (w_peer1_1[0] if False else w_peer0_1[0]), acc_1[1], rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
                acc_1[1] = cutlass.Float32(_fma_127)
                _fma_128 = cute.math.fma(_recv_smem[(base_2 + 2)], (w_peer1_1[0] if True else w_peer0_1[0]), acc_1[2], rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
                acc_1[2] = cutlass.Float32(_fma_128)
                _fma_129 = cute.math.fma(_recv_smem[(base_2 + 3)], (w_peer1_1[0] if True else w_peer0_1[0]), acc_1[3], rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
                acc_1[3] = cutlass.Float32(_fma_129)
                _fma_130 = cute.math.fma(_recv_smem[(base_2 + 512)], (w_peer1_1[0] if False else w_peer0_1[0]), acc_1[4], rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
                acc_1[4] = cutlass.Float32(_fma_130)
                _fma_131 = cute.math.fma(_recv_smem[((base_2 + 512) + 1)], (w_peer1_1[0] if False else w_peer0_1[0]), acc_1[5], rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
                acc_1[5] = cutlass.Float32(_fma_131)
                _fma_132 = cute.math.fma(_recv_smem[((base_2 + 512) + 2)], (w_peer1_1[0] if True else w_peer0_1[0]), acc_1[6], rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
                acc_1[6] = cutlass.Float32(_fma_132)
                _fma_133 = cute.math.fma(_recv_smem[((base_2 + 512) + 3)], (w_peer1_1[0] if True else w_peer0_1[0]), acc_1[7], rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
                acc_1[7] = cutlass.Float32(_fma_133)
                _fma_134 = cute.math.fma(_recv_smem[(base_2 + 1024)], (w_peer1_1[0] if False else w_peer0_1[0]), acc_1[8], rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
                acc_1[8] = cutlass.Float32(_fma_134)
                _fma_135 = cute.math.fma(_recv_smem[((base_2 + 1024) + 1)], (w_peer1_1[0] if False else w_peer0_1[0]), acc_1[9], rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
                acc_1[9] = cutlass.Float32(_fma_135)
                _fma_136 = cute.math.fma(_recv_smem[((base_2 + 1024) + 2)], (w_peer1_1[0] if True else w_peer0_1[0]), acc_1[10], rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
                acc_1[10] = cutlass.Float32(_fma_136)
                _fma_137 = cute.math.fma(_recv_smem[((base_2 + 1024) + 3)], (w_peer1_1[0] if True else w_peer0_1[0]), acc_1[11], rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
                acc_1[11] = cutlass.Float32(_fma_137)
                _fma_138 = cute.math.fma(_recv_smem[(base_2 + 1536)], (w_peer1_1[0] if False else w_peer0_1[0]), acc_1[12], rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
                acc_1[12] = cutlass.Float32(_fma_138)
                _fma_139 = cute.math.fma(_recv_smem[((base_2 + 1536) + 1)], (w_peer1_1[0] if False else w_peer0_1[0]), acc_1[13], rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
                acc_1[13] = cutlass.Float32(_fma_139)
                _fma_140 = cute.math.fma(_recv_smem[((base_2 + 1536) + 2)], (w_peer1_1[0] if True else w_peer0_1[0]), acc_1[14], rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
                acc_1[14] = cutlass.Float32(_fma_140)
                _fma_141 = cute.math.fma(_recv_smem[((base_2 + 1536) + 3)], (w_peer1_1[0] if True else w_peer0_1[0]), acc_1[15], rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
                acc_1[15] = cutlass.Float32(_fma_141)
                base_0_1 = cutlass.Int32((2304 + my_tid4_1))
                _fma_142 = cute.math.fma(_recv_smem[base_0_1], (w_peer1_1[1] if False else w_peer0_1[1]), acc_1[0], rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
                acc_1[0] = cutlass.Float32(_fma_142)
                _fma_143 = cute.math.fma(_recv_smem[(base_0_1 + 1)], (w_peer1_1[1] if False else w_peer0_1[1]), acc_1[1], rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
                acc_1[1] = cutlass.Float32(_fma_143)
                _fma_144 = cute.math.fma(_recv_smem[(base_0_1 + 2)], (w_peer1_1[1] if True else w_peer0_1[1]), acc_1[2], rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
                acc_1[2] = cutlass.Float32(_fma_144)
                _fma_145 = cute.math.fma(_recv_smem[(base_0_1 + 3)], (w_peer1_1[1] if True else w_peer0_1[1]), acc_1[3], rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
                acc_1[3] = cutlass.Float32(_fma_145)
                _fma_146 = cute.math.fma(_recv_smem[(base_0_1 + 512)], (w_peer1_1[1] if False else w_peer0_1[1]), acc_1[4], rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
                acc_1[4] = cutlass.Float32(_fma_146)
                _fma_147 = cute.math.fma(_recv_smem[((base_0_1 + 512) + 1)], (w_peer1_1[1] if False else w_peer0_1[1]), acc_1[5], rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
                acc_1[5] = cutlass.Float32(_fma_147)
                _fma_148 = cute.math.fma(_recv_smem[((base_0_1 + 512) + 2)], (w_peer1_1[1] if True else w_peer0_1[1]), acc_1[6], rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
                acc_1[6] = cutlass.Float32(_fma_148)
                _fma_149 = cute.math.fma(_recv_smem[((base_0_1 + 512) + 3)], (w_peer1_1[1] if True else w_peer0_1[1]), acc_1[7], rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
                acc_1[7] = cutlass.Float32(_fma_149)
                _fma_150 = cute.math.fma(_recv_smem[(base_0_1 + 1024)], (w_peer1_1[1] if False else w_peer0_1[1]), acc_1[8], rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
                acc_1[8] = cutlass.Float32(_fma_150)
                _fma_151 = cute.math.fma(_recv_smem[((base_0_1 + 1024) + 1)], (w_peer1_1[1] if False else w_peer0_1[1]), acc_1[9], rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
                acc_1[9] = cutlass.Float32(_fma_151)
                _fma_152 = cute.math.fma(_recv_smem[((base_0_1 + 1024) + 2)], (w_peer1_1[1] if True else w_peer0_1[1]), acc_1[10], rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
                acc_1[10] = cutlass.Float32(_fma_152)
                _fma_153 = cute.math.fma(_recv_smem[((base_0_1 + 1024) + 3)], (w_peer1_1[1] if True else w_peer0_1[1]), acc_1[11], rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
                acc_1[11] = cutlass.Float32(_fma_153)
                _fma_154 = cute.math.fma(_recv_smem[(base_0_1 + 1536)], (w_peer1_1[1] if False else w_peer0_1[1]), acc_1[12], rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
                acc_1[12] = cutlass.Float32(_fma_154)
                _fma_155 = cute.math.fma(_recv_smem[((base_0_1 + 1536) + 1)], (w_peer1_1[1] if False else w_peer0_1[1]), acc_1[13], rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
                acc_1[13] = cutlass.Float32(_fma_155)
                _fma_156 = cute.math.fma(_recv_smem[((base_0_1 + 1536) + 2)], (w_peer1_1[1] if True else w_peer0_1[1]), acc_1[14], rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
                acc_1[14] = cutlass.Float32(_fma_156)
                _fma_157 = cute.math.fma(_recv_smem[((base_0_1 + 1536) + 3)], (w_peer1_1[1] if True else w_peer0_1[1]), acc_1[15], rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
                acc_1[15] = cutlass.Float32(_fma_157)
                base_1_1 = cutlass.Int32((4608 + my_tid4_1))
                _fma_158 = cute.math.fma(_recv_smem[base_1_1], (w_peer1_1[2] if False else w_peer0_1[2]), acc_1[0], rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
                acc_1[0] = cutlass.Float32(_fma_158)
                _fma_159 = cute.math.fma(_recv_smem[(base_1_1 + 1)], (w_peer1_1[2] if False else w_peer0_1[2]), acc_1[1], rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
                acc_1[1] = cutlass.Float32(_fma_159)
                _fma_160 = cute.math.fma(_recv_smem[(base_1_1 + 2)], (w_peer1_1[2] if True else w_peer0_1[2]), acc_1[2], rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
                acc_1[2] = cutlass.Float32(_fma_160)
                _fma_161 = cute.math.fma(_recv_smem[(base_1_1 + 3)], (w_peer1_1[2] if True else w_peer0_1[2]), acc_1[3], rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
                acc_1[3] = cutlass.Float32(_fma_161)
                _fma_162 = cute.math.fma(_recv_smem[(base_1_1 + 512)], (w_peer1_1[2] if False else w_peer0_1[2]), acc_1[4], rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
                acc_1[4] = cutlass.Float32(_fma_162)
                _fma_163 = cute.math.fma(_recv_smem[((base_1_1 + 512) + 1)], (w_peer1_1[2] if False else w_peer0_1[2]), acc_1[5], rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
                acc_1[5] = cutlass.Float32(_fma_163)
                _fma_164 = cute.math.fma(_recv_smem[((base_1_1 + 512) + 2)], (w_peer1_1[2] if True else w_peer0_1[2]), acc_1[6], rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
                acc_1[6] = cutlass.Float32(_fma_164)
                _fma_165 = cute.math.fma(_recv_smem[((base_1_1 + 512) + 3)], (w_peer1_1[2] if True else w_peer0_1[2]), acc_1[7], rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
                acc_1[7] = cutlass.Float32(_fma_165)
                _fma_166 = cute.math.fma(_recv_smem[(base_1_1 + 1024)], (w_peer1_1[2] if False else w_peer0_1[2]), acc_1[8], rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
                acc_1[8] = cutlass.Float32(_fma_166)
                _fma_167 = cute.math.fma(_recv_smem[((base_1_1 + 1024) + 1)], (w_peer1_1[2] if False else w_peer0_1[2]), acc_1[9], rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
                acc_1[9] = cutlass.Float32(_fma_167)
                _fma_168 = cute.math.fma(_recv_smem[((base_1_1 + 1024) + 2)], (w_peer1_1[2] if True else w_peer0_1[2]), acc_1[10], rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
                acc_1[10] = cutlass.Float32(_fma_168)
                _fma_169 = cute.math.fma(_recv_smem[((base_1_1 + 1024) + 3)], (w_peer1_1[2] if True else w_peer0_1[2]), acc_1[11], rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
                acc_1[11] = cutlass.Float32(_fma_169)
                _fma_170 = cute.math.fma(_recv_smem[(base_1_1 + 1536)], (w_peer1_1[2] if False else w_peer0_1[2]), acc_1[12], rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
                acc_1[12] = cutlass.Float32(_fma_170)
                _fma_171 = cute.math.fma(_recv_smem[((base_1_1 + 1536) + 1)], (w_peer1_1[2] if False else w_peer0_1[2]), acc_1[13], rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
                acc_1[13] = cutlass.Float32(_fma_171)
                _fma_172 = cute.math.fma(_recv_smem[((base_1_1 + 1536) + 2)], (w_peer1_1[2] if True else w_peer0_1[2]), acc_1[14], rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
                acc_1[14] = cutlass.Float32(_fma_172)
                _fma_173 = cute.math.fma(_recv_smem[((base_1_1 + 1536) + 3)], (w_peer1_1[2] if True else w_peer0_1[2]), acc_1[15], rounding=cute.math.RoundingMode.NEAREST_EVEN, ftz=False)
                acc_1[15] = cutlass.Float32(_fma_173)
                _rcp_2 = cute.math.rcp(msum0_1[0], approx=True, ftz=True)
                _rcp_3 = cute.math.rcp(msum1_1[0], approx=True, ftz=True)
                acc_1[0] = cutlass.Float32((acc_1[0] * (_rcp_3 if False else _rcp_2)))
                acc_1[1] = cutlass.Float32((acc_1[1] * (_rcp_3 if False else _rcp_2)))
                acc_1[2] = cutlass.Float32((acc_1[2] * (_rcp_3 if True else _rcp_2)))
                acc_1[3] = cutlass.Float32((acc_1[3] * (_rcp_3 if True else _rcp_2)))
                acc_1[4] = cutlass.Float32((acc_1[4] * (_rcp_3 if False else _rcp_2)))
                acc_1[5] = cutlass.Float32((acc_1[5] * (_rcp_3 if False else _rcp_2)))
                acc_1[6] = cutlass.Float32((acc_1[6] * (_rcp_3 if True else _rcp_2)))
                acc_1[7] = cutlass.Float32((acc_1[7] * (_rcp_3 if True else _rcp_2)))
                acc_1[8] = cutlass.Float32((acc_1[8] * (_rcp_3 if False else _rcp_2)))
                acc_1[9] = cutlass.Float32((acc_1[9] * (_rcp_3 if False else _rcp_2)))
                acc_1[10] = cutlass.Float32((acc_1[10] * (_rcp_3 if True else _rcp_2)))
                acc_1[11] = cutlass.Float32((acc_1[11] * (_rcp_3 if True else _rcp_2)))
                acc_1[12] = cutlass.Float32((acc_1[12] * (_rcp_3 if False else _rcp_2)))
                acc_1[13] = cutlass.Float32((acc_1[13] * (_rcp_3 if False else _rcp_2)))
                acc_1[14] = cutlass.Float32((acc_1[14] * (_rcp_3 if True else _rcp_2)))
                acc_1[15] = cutlass.Float32((acc_1[15] * (_rcp_3 if True else _rcp_2)))
                qj1_1 = cutlass.Int32((qj & 1))
                qj2_1 = cutlass.Int32((qj & 2))
                o_vec_1 = cute.make_rmem_tensor((4,), cutlass.Uint32)
                o_tmp_1 = cute.make_rmem_tensor((4,), cutlass.Uint32)
                o_row_base_1 = cutlass.Int32((q_row * 128))
                m_local_r_1 = cutlass.Int32((m0_local if True else m1_local))
                _bf16x2_40 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(acc_1[0]), cutlass.Float32(acc_1[1])))[1]), cutlass.Float32(((cutlass.Float32(acc_1[0]), cutlass.Float32(acc_1[1])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
                o_vec_1[0] = cutlass.Uint32(_bf16x2_40)
                _bf16x2_41 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(acc_1[4]), cutlass.Float32(acc_1[5])))[1]), cutlass.Float32(((cutlass.Float32(acc_1[4]), cutlass.Float32(acc_1[5])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
                o_vec_1[1] = cutlass.Uint32(_bf16x2_41)
                _bf16x2_42 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(acc_1[8]), cutlass.Float32(acc_1[9])))[1]), cutlass.Float32(((cutlass.Float32(acc_1[8]), cutlass.Float32(acc_1[9])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
                o_vec_1[2] = cutlass.Uint32(_bf16x2_42)
                _bf16x2_43 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(acc_1[12]), cutlass.Float32(acc_1[13])))[1]), cutlass.Float32(((cutlass.Float32(acc_1[12]), cutlass.Float32(acc_1[13])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
                o_vec_1[3] = cutlass.Uint32(_bf16x2_43)
                _shfl_xor_28 = cute.arch.shuffle_sync_bfly(o_vec_1[1], 1, mask=0xFFFFFFFF, mask_and_clamp=31)
                o_tmp_1[0] = cutlass.Uint32(_shfl_xor_28)
                _shfl_xor_29 = cute.arch.shuffle_sync_bfly(o_vec_1[0], 1, mask=0xFFFFFFFF, mask_and_clamp=31)
                o_tmp_1[1] = cutlass.Uint32(_shfl_xor_29)
                _shfl_xor_30 = cute.arch.shuffle_sync_bfly(o_vec_1[3], 1, mask=0xFFFFFFFF, mask_and_clamp=31)
                o_tmp_1[2] = cutlass.Uint32(_shfl_xor_30)
                _shfl_xor_31 = cute.arch.shuffle_sync_bfly(o_vec_1[2], 1, mask=0xFFFFFFFF, mask_and_clamp=31)
                o_tmp_1[3] = cutlass.Uint32(_shfl_xor_31)
                o_vec_1[0] = cutlass.Uint32((o_tmp_1[0] if (qj1_1 != 0) else o_vec_1[0]))
                o_vec_1[1] = cutlass.Uint32((o_tmp_1[1] if (qj1_1 == 0) else o_vec_1[1]))
                o_vec_1[2] = cutlass.Uint32((o_tmp_1[2] if (qj1_1 != 0) else o_vec_1[2]))
                o_vec_1[3] = cutlass.Uint32((o_tmp_1[3] if (qj1_1 == 0) else o_vec_1[3]))
                _shfl_xor_32 = cute.arch.shuffle_sync_bfly(o_vec_1[2], 2, mask=0xFFFFFFFF, mask_and_clamp=31)
                o_tmp_1[0] = cutlass.Uint32(_shfl_xor_32)
                _shfl_xor_33 = cute.arch.shuffle_sync_bfly(o_vec_1[3], 2, mask=0xFFFFFFFF, mask_and_clamp=31)
                o_tmp_1[1] = cutlass.Uint32(_shfl_xor_33)
                _shfl_xor_34 = cute.arch.shuffle_sync_bfly(o_vec_1[0], 2, mask=0xFFFFFFFF, mask_and_clamp=31)
                o_tmp_1[2] = cutlass.Uint32(_shfl_xor_34)
                _shfl_xor_35 = cute.arch.shuffle_sync_bfly(o_vec_1[1], 2, mask=0xFFFFFFFF, mask_and_clamp=31)
                o_tmp_1[3] = cutlass.Uint32(_shfl_xor_35)
                o_vec_1[0] = cutlass.Uint32((o_tmp_1[0] if (qj2_1 != 0) else o_vec_1[0]))
                o_vec_1[1] = cutlass.Uint32((o_tmp_1[1] if (qj2_1 != 0) else o_vec_1[1]))
                o_vec_1[2] = cutlass.Uint32((o_tmp_1[2] if (qj2_1 == 0) else o_vec_1[2]))
                o_vec_1[3] = cutlass.Uint32((o_tmp_1[3] if (qj2_1 == 0) else o_vec_1[3]))
                o_off_1 = cutlass.Int32(((o_row_base_1 + (m_local_r_1 * 128)) + (((4 * sset_1) + qj) * 8)))
                _gmem_store_raw_54 = cutlass.Vector.from_elements([cutlass.Uint32(o_vec_1[0]), cutlass.Uint32(o_vec_1[1]), cutlass.Uint32(o_vec_1[2]), cutlass.Uint32(o_vec_1[3])], cutlass.Uint32)
                prims.store_ext(_gmem_store_raw_54.ir_value(), O + o_off_1)
                m_local_r_2_1 = cutlass.Int32((m0_local if False else m1_local))
                _bf16x2_44 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(acc_1[2]), cutlass.Float32(acc_1[3])))[1]), cutlass.Float32(((cutlass.Float32(acc_1[2]), cutlass.Float32(acc_1[3])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
                o_vec_1[0] = cutlass.Uint32(_bf16x2_44)
                _bf16x2_45 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(acc_1[6]), cutlass.Float32(acc_1[7])))[1]), cutlass.Float32(((cutlass.Float32(acc_1[6]), cutlass.Float32(acc_1[7])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
                o_vec_1[1] = cutlass.Uint32(_bf16x2_45)
                _bf16x2_46 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(acc_1[10]), cutlass.Float32(acc_1[11])))[1]), cutlass.Float32(((cutlass.Float32(acc_1[10]), cutlass.Float32(acc_1[11])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
                o_vec_1[2] = cutlass.Uint32(_bf16x2_46)
                _bf16x2_47 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(acc_1[14]), cutlass.Float32(acc_1[15])))[1]), cutlass.Float32(((cutlass.Float32(acc_1[14]), cutlass.Float32(acc_1[15])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
                o_vec_1[3] = cutlass.Uint32(_bf16x2_47)
                _shfl_xor_36 = cute.arch.shuffle_sync_bfly(o_vec_1[1], 1, mask=0xFFFFFFFF, mask_and_clamp=31)
                o_tmp_1[0] = cutlass.Uint32(_shfl_xor_36)
                _shfl_xor_37 = cute.arch.shuffle_sync_bfly(o_vec_1[0], 1, mask=0xFFFFFFFF, mask_and_clamp=31)
                o_tmp_1[1] = cutlass.Uint32(_shfl_xor_37)
                _shfl_xor_38 = cute.arch.shuffle_sync_bfly(o_vec_1[3], 1, mask=0xFFFFFFFF, mask_and_clamp=31)
                o_tmp_1[2] = cutlass.Uint32(_shfl_xor_38)
                _shfl_xor_39 = cute.arch.shuffle_sync_bfly(o_vec_1[2], 1, mask=0xFFFFFFFF, mask_and_clamp=31)
                o_tmp_1[3] = cutlass.Uint32(_shfl_xor_39)
                o_vec_1[0] = cutlass.Uint32((o_tmp_1[0] if (qj1_1 != 0) else o_vec_1[0]))
                o_vec_1[1] = cutlass.Uint32((o_tmp_1[1] if (qj1_1 == 0) else o_vec_1[1]))
                o_vec_1[2] = cutlass.Uint32((o_tmp_1[2] if (qj1_1 != 0) else o_vec_1[2]))
                o_vec_1[3] = cutlass.Uint32((o_tmp_1[3] if (qj1_1 == 0) else o_vec_1[3]))
                _shfl_xor_40 = cute.arch.shuffle_sync_bfly(o_vec_1[2], 2, mask=0xFFFFFFFF, mask_and_clamp=31)
                o_tmp_1[0] = cutlass.Uint32(_shfl_xor_40)
                _shfl_xor_41 = cute.arch.shuffle_sync_bfly(o_vec_1[3], 2, mask=0xFFFFFFFF, mask_and_clamp=31)
                o_tmp_1[1] = cutlass.Uint32(_shfl_xor_41)
                _shfl_xor_42 = cute.arch.shuffle_sync_bfly(o_vec_1[0], 2, mask=0xFFFFFFFF, mask_and_clamp=31)
                o_tmp_1[2] = cutlass.Uint32(_shfl_xor_42)
                _shfl_xor_43 = cute.arch.shuffle_sync_bfly(o_vec_1[1], 2, mask=0xFFFFFFFF, mask_and_clamp=31)
                o_tmp_1[3] = cutlass.Uint32(_shfl_xor_43)
                o_vec_1[0] = cutlass.Uint32((o_tmp_1[0] if (qj2_1 != 0) else o_vec_1[0]))
                o_vec_1[1] = cutlass.Uint32((o_tmp_1[1] if (qj2_1 != 0) else o_vec_1[1]))
                o_vec_1[2] = cutlass.Uint32((o_tmp_1[2] if (qj2_1 == 0) else o_vec_1[2]))
                o_vec_1[3] = cutlass.Uint32((o_tmp_1[3] if (qj2_1 == 0) else o_vec_1[3]))
                o_off_3_1 = cutlass.Int32(((o_row_base_1 + (m_local_r_2_1 * 128)) + (((4 * sset_1) + qj) * 8)))
                _gmem_store_raw_55 = cutlass.Vector.from_elements([cutlass.Uint32(o_vec_1[0]), cutlass.Uint32(o_vec_1[1]), cutlass.Uint32(o_vec_1[2]), cutlass.Uint32(o_vec_1[3])], cutlass.Uint32)
                prims.store_ext(_gmem_store_raw_55.ir_value(), O + o_off_3_1)

@cute.jit
def launch_vsa_sm90_bf16_small_k2c4(Q: cute.Tensor, _cake_tma_Q_dim_0: cutlass.Int64, _cake_tma_Q_dim_1: cutlass.Int64, _cake_tma_Q_dim_2: cutlass.Int64, _cake_tma_Q_stride16_0: cutlass.Int64, _cake_tma_Q_stride16_1: cutlass.Int64, K: cute.Tensor, _cake_tma_K_dim_0: cutlass.Int64, _cake_tma_K_dim_1: cutlass.Int64, _cake_tma_K_dim_2: cutlass.Int64, _cake_tma_K_stride16_0: cutlass.Int64, _cake_tma_K_stride16_1: cutlass.Int64, Vt: cute.Tensor, _cake_tma_Vt_dim_0: cutlass.Int64, _cake_tma_Vt_dim_1: cutlass.Int64, _cake_tma_Vt_dim_2: cutlass.Int64, _cake_tma_Vt_dim_3: cutlass.Int64, _cake_tma_Vt_stride16_0: cutlass.Int64, _cake_tma_Vt_stride16_1: cutlass.Int64, _cake_tma_Vt_stride16_2: cutlass.Int64, O: cute.Tensor, plan: cute.Tensor, seqlen_q: cutlass.Int32, seqlen_k: cutlass.Int32, scale_log2: cutlass.Float32, grid_x: cutlass.Int32, grid_y: cutlass.Int32, grid_z: cutlass.Int32, stream: cuda.CUstream):
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
    _cake_launch_cluster_spread(kernel_vsa_sm90_bf16_small_k2c4(_tma_Q, _tma_K, _tma_Vt, O.iterator, plan.iterator, seqlen_q, seqlen_k, scale_log2),
        grid=(grid_x, grid_y, grid_z),
        block=(256, 1, 1),
        cluster=(4, 1, 1),
        smem=111616,
        min_blocks_per_mp=1,
        stream=stream,
    )

def compile_program():
    return cute.compile(launch_vsa_sm90_bf16_small_k2c4,
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
