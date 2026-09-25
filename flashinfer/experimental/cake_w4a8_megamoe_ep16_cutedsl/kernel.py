# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Generated CuTe DSL implementation of the fused W4A8 forward. Do not edit."""
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

TMEM_NCOLS = 136
TMEM_ACC_OFFSET = 0
TMEM_SFW_OFFSET = 132
TMEM_SFX_OFFSET = 128
NUM_TASK_PIPE_STAGES = 2
NUM_PIPE16_STAGES = 5
NUM_PIPE32_STAGES = 10
NUM_PIPE64_STAGES = 9
NUM_ACC_PIPE_STAGES = 2
SMEM_W16_PAIR_OFF = 31744
SMEM_W16_PAIR_STAGE_BYTES = 32768
SMEM_W16_PAIR_STRIDE = 32768
SMEM_INPUT_ROW_OFF = 3072
SMEM_INPUT_ROW_STAGE_BYTES = 6144
SMEM_INPUT_ROW_STRIDE = 6144
SMEM_COUNTS_OFF = 1024
SMEM_COUNTS_STAGE_BYTES = 2048
SMEM_COUNTS_STRIDE = 2048
SMEM_PULL_STORAGE_OFF = 3072
SMEM_PULL_STORAGE_STAGE_BYTES = 12288
SMEM_PULL_STORAGE_STRIDE = 12288
SMEM_TASKS_OFF = 226304
SMEM_TASKS_STAGE_BYTES = 64
SMEM_TASKS_STRIDE = 64
SMEM_W16_OFF = 31744
SMEM_W16_STAGE_BYTES = 16384
SMEM_W16_STRIDE = 32768
SMEM_X16_OFF = 195584
SMEM_X16_STAGE_BYTES = 1024
SMEM_X16_STRIDE = 2048
SMEM_SW16_OFF = 205824
SMEM_SW16_STAGE_BYTES = 512
SMEM_SW16_STRIDE = 1024
SMEM_SX16_OFF = 210944
SMEM_SX16_STAGE_BYTES = 512
SMEM_SX16_STRIDE = 1024
SMEM_W16_HI_OFF = 48128
SMEM_W16_HI_STAGE_BYTES = 16384
SMEM_W16_HI_STRIDE = 32768
SMEM_X16_HI_OFF = 196608
SMEM_X16_HI_STAGE_BYTES = 1024
SMEM_X16_HI_STRIDE = 2048
SMEM_SW16_HI_OFF = 206336
SMEM_SW16_HI_STAGE_BYTES = 512
SMEM_SW16_HI_STRIDE = 1024
SMEM_SX16_HI_OFF = 211456
SMEM_SX16_HI_STAGE_BYTES = 512
SMEM_SX16_HI_STRIDE = 1024
SMEM_W32_OFF = 31744
SMEM_W32_STAGE_BYTES = 16384
SMEM_W32_STRIDE = 16384
SMEM_X32_OFF = 195584
SMEM_X32_STAGE_BYTES = 2048
SMEM_X32_STRIDE = 2048
SMEM_SW32_OFF = 216064
SMEM_SW32_STAGE_BYTES = 512
SMEM_SW32_STRIDE = 512
SMEM_SX32_OFF = 221184
SMEM_SX32_STAGE_BYTES = 512
SMEM_SX32_STRIDE = 512
SMEM_W64_OFF = 31744
SMEM_W64_STAGE_BYTES = 16384
SMEM_W64_STRIDE = 16384
SMEM_X64_OFF = 179200
SMEM_X64_STAGE_BYTES = 4096
SMEM_X64_STRIDE = 4096
SMEM_SW64_OFF = 216064
SMEM_SW64_STAGE_BYTES = 512
SMEM_SW64_STRIDE = 512
SMEM_SX64_OFF = 220672
SMEM_SX64_STAGE_BYTES = 512
SMEM_SX64_STRIDE = 512
SMEM_SCRATCH_OFF = 1024
SMEM_SCRATCH_STAGE_BYTES = 1024
SMEM_SCRATCH_STRIDE = 1024
SMEM_STAGING_OFF = 15360
SMEM_STAGING_STAGE_BYTES = 16384
SMEM_STAGING_STRIDE = 16384
SMEM_COMBINE_STORAGE_OFF = 1024
SMEM_COMBINE_STORAGE_STAGE_BYTES = 147456
SMEM_COMBINE_STORAGE_STRIDE = 147456
SMEM_TOTAL = 226432
THREADS = 512
ring_blocks = 642
TARGET_ARCH = 'sm_103a'
DYNAMIC_SMEM_BYTES = 226432

def _w4a8_launch_cluster_spread(kernel_launcher, **kwargs):
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
def kernel_w4a8_full_m32_fused_input(caller_x: cute.Pointer, caller_rw: cute.Pointer, staged_x: cute.Pointer, staged_sf: cute.Pointer, staged_ids: cute.Pointer, staged_rw: cute.Pointer, W1: cutlass.GridConstant[TensorMap], W1_pair: cutlass.GridConstant[TensorMap], W2: cutlass.GridConstant[TensorMap], W2_pair: cutlass.GridConstant[TensorMap], X1: cutlass.GridConstant[TensorMap], X2: cutlass.GridConstant[TensorMap], X1_32: cutlass.GridConstant[TensorMap], X2_32: cutlass.GridConstant[TensorMap], X1_8: cutlass.GridConstant[TensorMap], X2_8: cutlass.GridConstant[TensorMap], SW1: cutlass.GridConstant[TensorMap], SW1_1: cutlass.GridConstant[TensorMap], SW2: cutlass.GridConstant[TensorMap], SW2_1: cutlass.GridConstant[TensorMap], SX1: cutlass.GridConstant[TensorMap], SX1_1: cutlass.GridConstant[TensorMap], SX2: cutlass.GridConstant[TensorMap], SX2_1: cutlass.GridConstant[TensorMap], RW: cute.Pointer, Q: cutlass.GridConstant[TensorMap], Q8: cutlass.GridConstant[TensorMap], Q32: cutlass.GridConstant[TensorMap], QData: cute.Pointer, SF: cute.Pointer, output_peers: cute.Pointer, slots: cute.Pointer, output: cute.Pointer, epilogue_grid: cute.Pointer, l1_full: cute.Pointer, l1_empty: cute.Pointer, l2_full: cute.Pointer, l2_empty: cute.Pointer, ids: cute.Pointer, send: cute.Pointer, rank_counts: cute.Pointer, indices: cute.Pointer, src_peers: cute.Pointer, recv_peers: cute.Pointer, sum_peers: cute.Pointer, grid_counter: cute.Pointer, signals: cute.Pointer, status: cute.Pointer, signal_peers: cute.Pointer, rank: cutlass.Uint32, live_tokens: cutlass.Uint32, token_peers: cute.Pointer, sf_peers: cute.Pointer, weight_peers: cute.Pointer, XData: cute.Pointer, XSFData: cute.Pointer, metadata: cute.Pointer, recv: cute.Pointer, claims: cute.Pointer, pool_blocks: cutlass.Uint32, active_n: cutlass.Int32, valid_m: cutlass.Int32, epoch: cutlass.Int32, fc1_tiles: cutlass.Int32, fc2_tiles: cutlass.Int32, fc1_k: cutlass.Int32, fc2_k: cutlass.Int32):
    tid = cutlass.Int32(cute.arch.thread_idx()[0])
    warp = cutlass.Int32(cute.arch.warp_idx())
    lane = cutlass.Int32(cute.arch.lane_idx())
    bid = cutlass.Int32(cute.arch.block_idx()[0])
    num_bids = cutlass.Int32(cute.arch.grid_dim()[0])
    blockIdx = cute.arch.block_idx()
    gridDim = cute.arch.grid_dim()
    clusterIdx = (cutlass.Int32(cutlass.Uint32(blockIdx[0]) >> 1), blockIdx[1], blockIdx[2])
    clusterDim = (cutlass.Int32(cutlass.Uint32(gridDim[0]) >> 1), gridDim[1], gridDim[2])
    cluster_id = ((clusterIdx[2] * clusterDim[1] + clusterIdx[1]) * clusterDim[0]) + clusterIdx[0]
    num_clusters = clusterDim[0] * clusterDim[1] * clusterDim[2]
    cta_rank = cute.arch.block_idx_in_cluster()
    smem_raw = cute.arch.get_dyn_smem(cutlass.Uint8, alignment=1024)
    smem = smem_raw.toint()
    _flat_layout = cute.make_layout((2147483647,), stride=(1,))
    _caller_x = cute.make_tensor(caller_x, _flat_layout)
    _caller_rw = cute.make_tensor(caller_rw, _flat_layout)
    _staged_x = cute.make_tensor(staged_x, _flat_layout)
    _staged_sf = cute.make_tensor(staged_sf, _flat_layout)
    _staged_ids = cute.make_tensor(staged_ids, _flat_layout)
    _staged_rw = cute.make_tensor(staged_rw, _flat_layout)
    _RW = cute.make_tensor(RW, _flat_layout)
    _QData = cute.make_tensor(QData, _flat_layout)
    _SF = cute.make_tensor(SF, _flat_layout)
    _output_peers = cute.make_tensor(output_peers, _flat_layout)
    _slots = cute.make_tensor(slots, _flat_layout)
    _output = cute.make_tensor(output, _flat_layout)
    _epilogue_grid = cute.make_tensor(epilogue_grid, _flat_layout)
    _l1_full = cute.make_tensor(l1_full, _flat_layout)
    _l1_empty = cute.make_tensor(l1_empty, _flat_layout)
    _l2_full = cute.make_tensor(l2_full, _flat_layout)
    _l2_empty = cute.make_tensor(l2_empty, _flat_layout)
    _ids = cute.make_tensor(ids, _flat_layout)
    _send = cute.make_tensor(send, _flat_layout)
    _rank_counts = cute.make_tensor(rank_counts, _flat_layout)
    _indices = cute.make_tensor(indices, _flat_layout)
    _src_peers = cute.make_tensor(src_peers, _flat_layout)
    _recv_peers = cute.make_tensor(recv_peers, _flat_layout)
    _sum_peers = cute.make_tensor(sum_peers, _flat_layout)
    _grid_counter = cute.make_tensor(grid_counter, _flat_layout)
    _signals = cute.make_tensor(signals, _flat_layout)
    _status = cute.make_tensor(status, _flat_layout)
    _signal_peers = cute.make_tensor(signal_peers, _flat_layout)
    _token_peers = cute.make_tensor(token_peers, _flat_layout)
    _sf_peers = cute.make_tensor(sf_peers, _flat_layout)
    _weight_peers = cute.make_tensor(weight_peers, _flat_layout)
    _XData = cute.make_tensor(XData, _flat_layout)
    _XSFData = cute.make_tensor(XSFData, _flat_layout)
    _metadata = cute.make_tensor(metadata, _flat_layout)
    _recv = cute.make_tensor(recv, _flat_layout)
    _claims = cute.make_tensor(claims, _flat_layout)
    w16_pair = cute.recast_ptr(smem_raw + 31744, swizzle_=cute.make_swizzle(4, 3, 3), dtype=cutlass.Uint8)
    _w16_pair = cute.make_tensor(w16_pair, _flat_layout)
    w16_pair_addr = smem + 31744
    input_row = cute.recast_ptr(smem_raw + 3072, swizzle_=cute.make_swizzle(4, 3, 3), dtype=cutlass.BFloat16)
    _input_row = cute.make_tensor(input_row, _flat_layout)
    input_row_addr = smem + 3072
    counts = cute.recast_ptr(smem_raw + 1024, swizzle_=cute.make_swizzle(4, 3, 3), dtype=cutlass.Uint32)
    _counts = cute.make_tensor(counts, _flat_layout)
    counts_addr = smem + 1024
    pull_storage = cute.recast_ptr(smem_raw + 3072, swizzle_=cute.make_swizzle(4, 3, 3), dtype=cutlass.Uint8)
    _pull_storage = cute.make_tensor(pull_storage, _flat_layout)
    pull_storage_addr = smem + 3072
    tasks = cute.recast_ptr(smem_raw + 226304, swizzle_=cute.make_swizzle(4, 3, 3), dtype=cutlass.Uint32)
    _tasks = cute.make_tensor(tasks, _flat_layout)
    tasks_addr = smem + 226304
    w16 = cute.recast_ptr(smem_raw + 31744, swizzle_=cute.make_swizzle(4, 3, 3), dtype=cutlass.Uint8)
    _w16 = cute.make_tensor(w16, _flat_layout)
    w16_addr = smem + 31744
    x16 = cute.recast_ptr(smem_raw + 195584, swizzle_=cute.make_swizzle(4, 3, 3), dtype=cutlass.Uint8)
    _x16 = cute.make_tensor(x16, _flat_layout)
    x16_addr = smem + 195584
    sw16 = cute.recast_ptr(smem_raw + 205824, swizzle_=cute.make_swizzle(4, 3, 3), dtype=cutlass.Uint32)
    _sw16 = cute.make_tensor(sw16, _flat_layout)
    sw16_addr = smem + 205824
    sx16 = cute.recast_ptr(smem_raw + 210944, swizzle_=cute.make_swizzle(4, 3, 3), dtype=cutlass.Uint32)
    _sx16 = cute.make_tensor(sx16, _flat_layout)
    sx16_addr = smem + 210944
    w16_hi = cute.recast_ptr(smem_raw + 48128, swizzle_=cute.make_swizzle(4, 3, 3), dtype=cutlass.Uint8)
    _w16_hi = cute.make_tensor(w16_hi, _flat_layout)
    w16_hi_addr = smem + 48128
    x16_hi = cute.recast_ptr(smem_raw + 196608, swizzle_=cute.make_swizzle(4, 3, 3), dtype=cutlass.Uint8)
    _x16_hi = cute.make_tensor(x16_hi, _flat_layout)
    x16_hi_addr = smem + 196608
    sw16_hi = cute.recast_ptr(smem_raw + 206336, swizzle_=cute.make_swizzle(4, 3, 3), dtype=cutlass.Uint32)
    _sw16_hi = cute.make_tensor(sw16_hi, _flat_layout)
    sw16_hi_addr = smem + 206336
    sx16_hi = cute.recast_ptr(smem_raw + 211456, swizzle_=cute.make_swizzle(4, 3, 3), dtype=cutlass.Uint32)
    _sx16_hi = cute.make_tensor(sx16_hi, _flat_layout)
    sx16_hi_addr = smem + 211456
    w32 = cute.recast_ptr(smem_raw + 31744, swizzle_=cute.make_swizzle(4, 3, 3), dtype=cutlass.Uint8)
    _w32 = cute.make_tensor(w32, _flat_layout)
    w32_addr = smem + 31744
    x32 = cute.recast_ptr(smem_raw + 195584, swizzle_=cute.make_swizzle(4, 3, 3), dtype=cutlass.Uint8)
    _x32 = cute.make_tensor(x32, _flat_layout)
    x32_addr = smem + 195584
    sw32 = cute.recast_ptr(smem_raw + 216064, swizzle_=cute.make_swizzle(4, 3, 3), dtype=cutlass.Uint32)
    _sw32 = cute.make_tensor(sw32, _flat_layout)
    sw32_addr = smem + 216064
    sx32 = cute.recast_ptr(smem_raw + 221184, swizzle_=cute.make_swizzle(4, 3, 3), dtype=cutlass.Uint32)
    _sx32 = cute.make_tensor(sx32, _flat_layout)
    sx32_addr = smem + 221184
    w64 = cute.recast_ptr(smem_raw + 31744, swizzle_=cute.make_swizzle(4, 3, 3), dtype=cutlass.Uint8)
    _w64 = cute.make_tensor(w64, _flat_layout)
    w64_addr = smem + 31744
    x64 = cute.recast_ptr(smem_raw + 179200, swizzle_=cute.make_swizzle(4, 3, 3), dtype=cutlass.Uint8)
    _x64 = cute.make_tensor(x64, _flat_layout)
    x64_addr = smem + 179200
    sw64 = cute.recast_ptr(smem_raw + 216064, swizzle_=cute.make_swizzle(4, 3, 3), dtype=cutlass.Uint32)
    _sw64 = cute.make_tensor(sw64, _flat_layout)
    sw64_addr = smem + 216064
    sx64 = cute.recast_ptr(smem_raw + 220672, swizzle_=cute.make_swizzle(4, 3, 3), dtype=cutlass.Uint32)
    _sx64 = cute.make_tensor(sx64, _flat_layout)
    sx64_addr = smem + 220672
    scratch = cute.recast_ptr(smem_raw + 1024, swizzle_=cute.make_swizzle(4, 3, 3), dtype=cutlass.Float32)
    _scratch = cute.make_tensor(scratch, _flat_layout)
    scratch_addr = smem + 1024
    staging = cute.recast_ptr(smem_raw + 15360, swizzle_=cute.make_swizzle(4, 3, 3), dtype=cutlass.Uint32)
    _staging = cute.make_tensor(staging, _flat_layout)
    staging_addr = smem + 15360
    combine_storage = cute.recast_ptr(smem_raw + 1024, swizzle_=cute.make_swizzle(4, 3, 3), dtype=cutlass.Uint32)
    _combine_storage = cute.make_tensor(combine_storage, _flat_layout)
    combine_storage_addr = smem + 1024
    if (warp == 0):
        if prims.elect_sync():
            cute.arch.prefetch(W1.get_ptr(), tensormap=True)
            cute.arch.prefetch(W1_pair.get_ptr(), tensormap=True)
            cute.arch.prefetch(W2.get_ptr(), tensormap=True)
            cute.arch.prefetch(W2_pair.get_ptr(), tensormap=True)
            cute.arch.prefetch(X1.get_ptr(), tensormap=True)
            cute.arch.prefetch(X2.get_ptr(), tensormap=True)
            cute.arch.prefetch(SW1.get_ptr(), tensormap=True)
            cute.arch.prefetch(SW1_1.get_ptr(), tensormap=True)
            cute.arch.prefetch(SW2.get_ptr(), tensormap=True)
            cute.arch.prefetch(SW2_1.get_ptr(), tensormap=True)
            cute.arch.prefetch(SX1.get_ptr(), tensormap=True)
            cute.arch.prefetch(SX1_1.get_ptr(), tensormap=True)
            cute.arch.prefetch(SX2.get_ptr(), tensormap=True)
            cute.arch.prefetch(SX2_1.get_ptr(), tensormap=True)
            cute.arch.prefetch(Q.get_ptr(), tensormap=True)
            cute.arch.prefetch(Q8.get_ptr(), tensormap=True)
            cute.arch.prefetch(Q32.get_ptr(), tensormap=True)
            cute.arch.prefetch(X1_32.get_ptr(), tensormap=True)
            cute.arch.prefetch(X2_32.get_ptr(), tensormap=True)
            cute.arch.prefetch(X1_8.get_ptr(), tensormap=True)
            cute.arch.prefetch(X2_8.get_ptr(), tensormap=True)
    pull_addr = cute.recast_ptr(smem_raw, dtype=cutlass.Uint64)
    task_full_addr = cute.recast_ptr(smem_raw + 32, dtype=cutlass.Uint64)
    task_empty_addr = cute.recast_ptr(smem_raw + 48, dtype=cutlass.Uint64)
    full16_addr = cute.recast_ptr(smem_raw + 64, dtype=cutlass.Uint64)
    empty16_addr = cute.recast_ptr(smem_raw + 104, dtype=cutlass.Uint64)
    full32_addr = cute.recast_ptr(smem_raw + 144, dtype=cutlass.Uint64)
    empty32_addr = cute.recast_ptr(smem_raw + 224, dtype=cutlass.Uint64)
    full64_addr = cute.recast_ptr(smem_raw + 304, dtype=cutlass.Uint64)
    empty64_addr = cute.recast_ptr(smem_raw + 376, dtype=cutlass.Uint64)
    done_addr = cute.recast_ptr(smem_raw + 448, dtype=cutlass.Uint64)
    released_addr = cute.recast_ptr(smem_raw + 464, dtype=cutlass.Uint64)
    combine_barriers_addr = cute.recast_ptr(smem_raw + 480, dtype=cutlass.Uint64)
    if warp == 1:
        with cute.arch.elect_one():
            cute.arch.mbarrier_init(pull_addr + 0, 1)
            cute.arch.mbarrier_init(pull_addr + 1, 1)
            cute.arch.mbarrier_init(pull_addr + 2, 1)
            cute.arch.mbarrier_init(pull_addr + 3, 1)
    if warp == 0:
        with cute.arch.elect_one():
            cute.arch.mbarrier_init(task_full_addr + 0, 1)
            cute.arch.mbarrier_init(task_full_addr + 1, 1)
            cute.arch.mbarrier_init(task_empty_addr + 0, 512)
            cute.arch.mbarrier_init(task_empty_addr + 1, 512)
            cute.arch.mbarrier_init(full16_addr + 0, 4)
            cute.arch.mbarrier_init(full16_addr + 1, 4)
            cute.arch.mbarrier_init(full16_addr + 2, 4)
            cute.arch.mbarrier_init(full16_addr + 3, 4)
            cute.arch.mbarrier_init(full16_addr + 4, 4)
            cute.arch.mbarrier_init(empty16_addr + 0, 1)
            cute.arch.mbarrier_init(empty16_addr + 1, 1)
            cute.arch.mbarrier_init(empty16_addr + 2, 1)
            cute.arch.mbarrier_init(empty16_addr + 3, 1)
            cute.arch.mbarrier_init(empty16_addr + 4, 1)
            cute.arch.mbarrier_init(full32_addr + 0, 4)
            cute.arch.mbarrier_init(full32_addr + 1, 4)
            cute.arch.mbarrier_init(full32_addr + 2, 4)
            cute.arch.mbarrier_init(full32_addr + 3, 4)
            cute.arch.mbarrier_init(full32_addr + 4, 4)
            cute.arch.mbarrier_init(full32_addr + 5, 4)
            cute.arch.mbarrier_init(full32_addr + 6, 4)
            cute.arch.mbarrier_init(full32_addr + 7, 4)
            cute.arch.mbarrier_init(full32_addr + 8, 4)
            cute.arch.mbarrier_init(full32_addr + 9, 4)
            cute.arch.mbarrier_init(empty32_addr + 0, 1)
            cute.arch.mbarrier_init(empty32_addr + 1, 1)
            cute.arch.mbarrier_init(empty32_addr + 2, 1)
            cute.arch.mbarrier_init(empty32_addr + 3, 1)
            cute.arch.mbarrier_init(empty32_addr + 4, 1)
            cute.arch.mbarrier_init(empty32_addr + 5, 1)
            cute.arch.mbarrier_init(empty32_addr + 6, 1)
            cute.arch.mbarrier_init(empty32_addr + 7, 1)
            cute.arch.mbarrier_init(empty32_addr + 8, 1)
            cute.arch.mbarrier_init(empty32_addr + 9, 1)
            cute.arch.mbarrier_init(full64_addr + 0, 4)
            cute.arch.mbarrier_init(full64_addr + 1, 4)
            cute.arch.mbarrier_init(full64_addr + 2, 4)
            cute.arch.mbarrier_init(full64_addr + 3, 4)
            cute.arch.mbarrier_init(full64_addr + 4, 4)
            cute.arch.mbarrier_init(full64_addr + 5, 4)
            cute.arch.mbarrier_init(full64_addr + 6, 4)
            cute.arch.mbarrier_init(full64_addr + 7, 4)
            cute.arch.mbarrier_init(full64_addr + 8, 4)
            cute.arch.mbarrier_init(empty64_addr + 0, 1)
            cute.arch.mbarrier_init(empty64_addr + 1, 1)
            cute.arch.mbarrier_init(empty64_addr + 2, 1)
            cute.arch.mbarrier_init(empty64_addr + 3, 1)
            cute.arch.mbarrier_init(empty64_addr + 4, 1)
            cute.arch.mbarrier_init(empty64_addr + 5, 1)
            cute.arch.mbarrier_init(empty64_addr + 6, 1)
            cute.arch.mbarrier_init(empty64_addr + 7, 1)
            cute.arch.mbarrier_init(empty64_addr + 8, 1)
            cute.arch.mbarrier_init(done_addr + 0, 1)
            cute.arch.mbarrier_init(done_addr + 1, 1)
            cute.arch.mbarrier_init(released_addr + 0, 512)
            cute.arch.mbarrier_init(released_addr + 1, 512)
    if warp == 2:
        with cute.arch.elect_one():
            cute.arch.mbarrier_init(combine_barriers_addr + 0, 1)
            cute.arch.mbarrier_init(combine_barriers_addr + 1, 1)
            cute.arch.mbarrier_init(combine_barriers_addr + 2, 1)
            cute.arch.mbarrier_init(combine_barriers_addr + 3, 1)
            cute.arch.mbarrier_init(combine_barriers_addr + 4, 1)
            cute.arch.mbarrier_init(combine_barriers_addr + 5, 1)
            cute.arch.mbarrier_init(combine_barriers_addr + 6, 1)
            cute.arch.mbarrier_init(combine_barriers_addr + 7, 1)
            cute.arch.mbarrier_init(combine_barriers_addr + 8, 1)
            cute.arch.mbarrier_init(combine_barriers_addr + 9, 1)
            cute.arch.mbarrier_init(combine_barriers_addr + 10, 1)
            cute.arch.mbarrier_init(combine_barriers_addr + 11, 1)
            cute.arch.mbarrier_init(combine_barriers_addr + 12, 1)
            cute.arch.mbarrier_init(combine_barriers_addr + 13, 1)
            cute.arch.mbarrier_init(combine_barriers_addr + 14, 1)
            cute.arch.mbarrier_init(combine_barriers_addr + 15, 1)
    cute.arch.mbarrier_init_fence()
    cute.arch.sync_threads()
    _tmem_hold = cute.recast_ptr(smem_raw + 608, dtype=cutlass.Uint32)
    cute.arch.cluster_arrive(aligned=True)
    cute.arch.cluster_wait()
    if warp == 3:
        cute.arch.alloc_tmem(256, _tmem_hold, is_two_cta=True, arch='sm_100')
    cute.arch.cluster_arrive(aligned=True)
    cute.arch.cluster_wait()
    prims.tcgen05_fence('after_thread_sync')
    _tmem_base_ptr = cute.arch.retrieve_tmem_ptr(cutlass.Float32, 4, _tmem_hold)
    taddr = _tmem_base_ptr.toint()
    tmem_acc = cutlass.Int32(taddr)
    tmem_sfw = cutlass.Int32(taddr + 132)
    tmem_sfx = cutlass.Int32(taddr + 128)
    # Ordered hardware-warpgroup register redistribution.
    if warp >= 4 and warp <= 7:
        cute.arch.setmaxregister_decrease(40)
    cute.arch.sync_threads()
    if warp <= 3:
        cute.arch.setmaxregister_decrease(48)
        count = cute.make_rmem_tensor((1,), cutlass.Uint32)
        matched = cute.make_rmem_tensor((1,), cutlass.Int32)
        signal_state = cute.make_rmem_tensor((1,), cutlass.Uint32)
        count_1 = cute.make_rmem_tensor((1,), cutlass.Uint64)
        token_end = cute.make_rmem_tensor((1,), cutlass.Uint32)
        block_end = cute.make_rmem_tensor((1,), cutlass.Uint32)
        expert = cute.make_rmem_tensor((1,), cutlass.Int32)
        start = cute.make_rmem_tensor((1,), cutlass.Uint32)
        end = cute.make_rmem_tensor((1,), cutlass.Uint32)
        pool_blocks_0 = cute.make_rmem_tensor((1,), cutlass.Uint32)
        stored = cute.make_rmem_tensor((1,), cutlass.Uint32)
        phase = cute.make_rmem_tensor((1,), cutlass.Uint32)
        remaining = cute.make_rmem_tensor((1,), cutlass.Uint32)
        slot = cute.make_rmem_tensor((1,), cutlass.Uint32)
        offset = cute.make_rmem_tensor((1,), cutlass.Uint32)
        selected = cute.make_rmem_tensor((1,), cutlass.Uint32)
        source_rank = cute.make_rmem_tensor((1,), cutlass.Uint32)
        source_slot = cute.make_rmem_tensor((1,), cutlass.Uint32)
        minimum = cute.make_rmem_tensor((1,), cutlass.Uint32)
        increment = cute.make_rmem_tensor((1,), cutlass.Uint32)
        for expert_1 in cutlass.range(cutlass.Int32(((bid * 4) + warp)), cutlass.Int32(512), cutlass.Int32((num_bids * 4))):
            count[0] = cutlass.Uint32(0)
            for route_base in cutlass.range(cutlass.Int32(0), cutlass.Int32((live_tokens * 8)), cutlass.Int32(32)):
                route = cutlass.Uint32((route_base + lane))
                matched[0] = cutlass.Int32(0)
                if (route < (live_tokens * 8)):
                    matched[0] = cutlass.Int32((cutlass.Int32(1) if (_ids[route] == cutlass.Int64(expert_1)) else cutlass.Int32(0)))
                _vote_0 = cute.arch.vote_ballot_sync(cutlass.Boolean((matched[0] != 0)), cutlass.Uint32(0xFFFFFFFF)).bitcast(cutlass.Uint32)
                mask = cutlass.Uint32(_vote_0)
                if (matched[0] != 0):
                    earlier = cutlass.Uint32(((cutlass.Uint32(1) << cutlass.Uint32(lane)) - 1))
                    _popc_0 = cutlass.Int32(cute.arch.popc(cutlass.Uint32((mask & earlier))))
                    slot_1 = cutlass.Uint32((count[0] + cutlass.Uint32(_popc_0)))
                    offset_1 = cutlass.Uint32((((cutlass.Uint32(((expert_1 % 32) * 16)) + rank) * 6144) + slot_1))
                    prims.store_ext(cutlass.Uint32(route).ir_value(), (cute.make_ptr(cutlass.Uint32, cutlass.Uint64(_src_peers[cutlass.Int32(cutlass_arith.divsi(cutlass.Int32(expert_1).ir_value(), cutlass.Int32(32).ir_value()))]), mem_space=cute.AddressSpace.gmem, assumed_align=4) + (offset_1)))
                _popc_1 = cutlass.Int32(cute.arch.popc(cutlass.Uint32(mask)))
                count[0] = cutlass.Uint32((count[0] + cutlass.Uint32(_popc_1)))
                cute.arch.sync_warp()
            if prims.elect_sync():
                value = cutlass.Uint64(((cutlass.Uint64(num_bids) << 32) | cutlass.Uint64(count[0])))
                prims.store_ext(cutlass.Uint64(value).ir_value(), (send + (expert_1)))
                prims.store_ext(cutlass.Uint64(cutlass.Uint64(count[0])).ir_value(), (cute.make_ptr(cutlass.Uint64, cutlass.Uint64(_recv_peers[cutlass.Int32(cutlass_arith.divsi(cutlass.Int32(expert_1).ir_value(), cutlass.Int32(32).ir_value()))]), mem_space=cute.AddressSpace.gmem, assumed_align=8) + (((rank * 32) + cutlass.Uint32((expert_1 % 32))))))
                _atomic_old_0 = cute.arch.atomic_add(cute.make_ptr(cutlass.Uint64, cutlass.Uint64(_sum_peers[cutlass.Int32(cutlass_arith.divsi(cutlass.Int32(expert_1).ir_value(), cutlass.Int32(32).ir_value()))]), mem_space=cute.AddressSpace.gmem, assumed_align=8) + (expert_1 % 32), cutlass.Uint64(value), sem='relaxed', scope='sys')
            cute.arch.sync_warp()
        prims.barrier_cta_sync(1, thread_count=384)
        signal_state[0] = cutlass.Uint32(0)
        if (tid == 0):
            signal_state[0] = cutlass.Uint32((prims.load_ext((cute.recast_ptr(status, dtype=cutlass.Uint32) + (0)), dtype=cutlass.Uint32, order=prims.MemOrder.VOLATILE) & 3))
        prims.barrier_cta_sync(0, thread_count=128)
        if (tid == 0):
            delta = cutlass.Uint32((cutlass.Int32((2147483649 - num_bids)) if (bid == 0) else cutlass.Int32(1)))
            _atomic_old_1 = cute.arch.atomic_add(cute.recast_ptr(grid_counter, dtype=cutlass.Uint32), cutlass.Uint32(delta), sem='release', scope='gpu')
            while ((prims.load_ext(grid_counter + 0, dtype=cutlass.Uint32, order=prims.MemOrder.ACQUIRE, scope=prims.MemScope.GPU) ^ cutlass.Uint32(_atomic_old_1)) & cutlass.Uint32(0x80000000)) == 0:
                pass
        prims.barrier_cta_sync(0, thread_count=128)
        if (tid == 0):
            phase_1 = cutlass.Uint32((signal_state[0] & 1))
            sign = cutlass.Uint32((signal_state[0] >> 1))
            if (bid == 0):
                _atomic_old_2 = cute.arch.atomic_add(status, cutlass.Uint32(1), sem='relaxed', scope='gpu')
                delta_1 = cutlass.Int32((cutlass.Int32(-1) if (sign != 0) else cutlass.Int32(1)))
                prims.inline_ptx(
                    'multimem.red.release.sys.global.add.s32 [{$r0}], {$r1};',
                    read_only_args=[cutlass.Uint64(cute.make_ptr(cutlass.Int32, cutlass.Uint64((_signal_peers[16] + cutlass.Uint64((phase_1 * 4)))), mem_space=cute.AddressSpace.gmem, assumed_align=4).toint()), cutlass.Int32(delta_1)],
                )
            target = cutlass.Uint32((cutlass.Int32(0) if (sign != 0) else cutlass.Int32(16)))
            while prims.load_ext(signals + phase_1, dtype=cutlass.Uint32, order=prims.MemOrder.ACQUIRE, scope=prims.MemScope.SYS) != cutlass.Uint32(target):
                pass
        prims.barrier_cta_sync(0, thread_count=128)
        prims.barrier_cta_sync(1, thread_count=384)
        count_1[0] = cutlass.Uint64(0)
        while cutlass.Boolean(((count_1[0] >> 32) != 2432)):
            count_1[0] = cutlass.Uint64(prims.load_ext((cute.recast_ptr(recv, dtype=cutlass.Uint64) + (lane)), dtype=cutlass.Uint64, order=prims.MemOrder.VOLATILE))
        _warp_redux_u32_0 = cutlass.Uint32(cutlass_llvm.inline_asm(
            cutlass.Uint32.mlir_type,
            [cutlass.Uint32(cutlass.Uint32(count_1[0])).ir_value()],
            'redux.sync.add.u32 $0, $1, 0xffffffff;', '=r,r',
            has_side_effects=True, is_align_stack=False,
            asm_dialect=cutlass_llvm.AsmDialect.AD_ATT,
        ))
        cute.arch.sync_warp()
        lane_blocks = cutlass.Uint32(cutlass.Uint32(cutlass_arith.divui(cutlass.Uint32(((cutlass.Uint32(count_1[0]) + 64) - 1)).ir_value(), cutlass.Uint32(64).ir_value())))
        token_end[0] = cutlass.Uint32(cutlass.Uint32(count_1[0]))
        block_end[0] = cutlass.Uint32(lane_blocks)
        _shfl_up_0 = cute.arch.shuffle_sync_up(token_end[0], 1, mask=0xFFFFFFFF, mask_and_clamp=0)
        _shfl_up_1 = cute.arch.shuffle_sync_up(block_end[0], 1, mask=0xFFFFFFFF, mask_and_clamp=0)
        _if_condition_0 = cutlass.Boolean((lane >= 1))
        token_end[0] = cutlass.Uint32(cutlass.select_(_if_condition_0, cutlass.Uint32((token_end[0] + _shfl_up_0)), token_end[0]))
        block_end[0] = cutlass.Uint32(cutlass.select_(_if_condition_0, cutlass.Uint32((block_end[0] + _shfl_up_1)), block_end[0]))
        _shfl_up_2 = cute.arch.shuffle_sync_up(token_end[0], 2, mask=0xFFFFFFFF, mask_and_clamp=0)
        _shfl_up_3 = cute.arch.shuffle_sync_up(block_end[0], 2, mask=0xFFFFFFFF, mask_and_clamp=0)
        _if_condition_1 = cutlass.Boolean((lane >= 2))
        token_end[0] = cutlass.Uint32(cutlass.select_(_if_condition_1, cutlass.Uint32((token_end[0] + _shfl_up_2)), token_end[0]))
        block_end[0] = cutlass.Uint32(cutlass.select_(_if_condition_1, cutlass.Uint32((block_end[0] + _shfl_up_3)), block_end[0]))
        _shfl_up_4 = cute.arch.shuffle_sync_up(token_end[0], 4, mask=0xFFFFFFFF, mask_and_clamp=0)
        _shfl_up_5 = cute.arch.shuffle_sync_up(block_end[0], 4, mask=0xFFFFFFFF, mask_and_clamp=0)
        _if_condition_2 = cutlass.Boolean((lane >= 4))
        token_end[0] = cutlass.Uint32(cutlass.select_(_if_condition_2, cutlass.Uint32((token_end[0] + _shfl_up_4)), token_end[0]))
        block_end[0] = cutlass.Uint32(cutlass.select_(_if_condition_2, cutlass.Uint32((block_end[0] + _shfl_up_5)), block_end[0]))
        _shfl_up_6 = cute.arch.shuffle_sync_up(token_end[0], 8, mask=0xFFFFFFFF, mask_and_clamp=0)
        _shfl_up_7 = cute.arch.shuffle_sync_up(block_end[0], 8, mask=0xFFFFFFFF, mask_and_clamp=0)
        _if_condition_3 = cutlass.Boolean((lane >= 8))
        token_end[0] = cutlass.Uint32(cutlass.select_(_if_condition_3, cutlass.Uint32((token_end[0] + _shfl_up_6)), token_end[0]))
        block_end[0] = cutlass.Uint32(cutlass.select_(_if_condition_3, cutlass.Uint32((block_end[0] + _shfl_up_7)), block_end[0]))
        _shfl_up_8 = cute.arch.shuffle_sync_up(token_end[0], 16, mask=0xFFFFFFFF, mask_and_clamp=0)
        _shfl_up_9 = cute.arch.shuffle_sync_up(block_end[0], 16, mask=0xFFFFFFFF, mask_and_clamp=0)
        _if_condition_4 = cutlass.Boolean((lane >= 16))
        token_end[0] = cutlass.Uint32(cutlass.select_(_if_condition_4, cutlass.Uint32((token_end[0] + _shfl_up_8)), token_end[0]))
        block_end[0] = cutlass.Uint32(cutlass.select_(_if_condition_4, cutlass.Uint32((block_end[0] + _shfl_up_9)), block_end[0]))
        token_start = cutlass.Uint32((token_end[0] - cutlass.Uint32(count_1[0])))
        block_start = cutlass.Uint32((block_end[0] - lane_blocks))
        expert[0] = cutlass.Int32(-1)
        start[0] = cutlass.Uint32(0)
        end[0] = cutlass.Uint32(0)
        pool_blocks_0[0] = cutlass.Uint32(0)
        stored[0] = cutlass.Uint32(0)
        phase[0] = cutlass.Uint32(0)
        warp_1 = cutlass.Uint32(cutlass.Int32(cutlass_arith.divsi(cutlass.Int32(tid).ir_value(), cutlass.Int32(32).ir_value())))
        ring_blocks_1 = 642
        for token in cutlass.range(cutlass.Int32((cutlass.Uint32((bid * 4)) + warp_1)), cutlass.Int32(_warp_redux_u32_0), cutlass.Int32((num_bids * 4))):
            previous = cutlass.Int32(expert[0])
            _vote_1 = cute.arch.vote_ballot_sync(cutlass.Boolean(((token_start <= cutlass.Uint32(token)) & (token_end[0] > cutlass.Uint32(token)))), cutlass.Uint32(0xFFFFFFFF)).bitcast(cutlass.Uint32)
            _ffs_reversed_bits_5 = cute.arch.brev(cutlass.Uint32(_vote_1))
            _ffs_leading_one_6 = cutlass.Int32(cute.arch.bfind(_ffs_reversed_bits_5))
            _ffs_0 = cutlass.Int32(cutlass.select_(cutlass.Uint32(_vote_1) == cutlass.Uint32(0), cutlass.Int32(0), cutlass.Int32(32) - _ffs_leading_one_6))
            expert_0 = cutlass.Int32(cutlass.Int32((_ffs_0 - 1)))
            _shfl_0 = cute.arch.shuffle_sync(token_start, expert_0, mask=4294967295, mask_and_clamp=31)
            start_1 = cutlass.Uint32(_shfl_0)
            _shfl_1 = cute.arch.shuffle_sync(token_end[0], expert_0, mask=4294967295, mask_and_clamp=31)
            end_2 = cutlass.Uint32(_shfl_1)
            _shfl_2 = cute.arch.shuffle_sync(block_start, expert_0, mask=4294967295, mask_and_clamp=31)
            pool_blocks_3 = cutlass.Uint32(_shfl_2)
            expert[0] = cutlass.Int32(expert_0)
            start[0] = cutlass.Uint32(start_1)
            end[0] = cutlass.Uint32(end_2)
            pool_blocks_0[0] = cutlass.Uint32(pool_blocks_3)
            if (previous != expert[0]):
                stored[0] = cutlass.Uint32(0)
                if (lane < 16):
                    stored[0] = cutlass.Uint32(cutlass.Uint32(_rank_counts[((lane * 32) + expert[0])]))
            remaining[0] = cutlass.Uint32(stored[0])
            slot[0] = cutlass.Uint32((cutlass.Uint32(token) - start[0]))
            offset[0] = cutlass.Uint32(0)
            selected[0] = cutlass.Uint32(0)
            source_rank[0] = cutlass.Uint32(0)
            source_slot[0] = cutlass.Uint32(0)
            while cutlass.Boolean((selected[0] == 0)):
                active = cutlass.Uint32((cutlass.Int32(1) if (remaining[0] > 0) else cutlass.Int32(0)))
                minimum[0] = cutlass.Uint32(4294967295)
                _if_condition_7 = cutlass.Boolean((remaining[0] > 0))
                minimum[0] = cutlass.Uint32(cutlass.select_(_if_condition_7, cutlass.Uint32(remaining[0]), minimum[0]))
                _warp_redux_u32_1 = cutlass.Uint32(cutlass_llvm.inline_asm(
                    cutlass.Uint32.mlir_type,
                    [cutlass.Uint32(active).ir_value()],
                    'redux.sync.add.u32 $0, $1, 0xffffffff;', '=r,r',
                    has_side_effects=True, is_align_stack=False,
                    asm_dialect=cutlass_llvm.AsmDialect.AD_ATT,
                ))
                _warp_redux_u32_2 = cutlass.Uint32(cutlass_llvm.inline_asm(
                    cutlass.Uint32.mlir_type,
                    [cutlass.Uint32(minimum[0]).ir_value()],
                    'redux.sync.min.u32 $0, $1, 0xffffffff;', '=r,r',
                    has_side_effects=True, is_align_stack=False,
                    asm_dialect=cutlass_llvm.AsmDialect.AD_ATT,
                ))
                round_tokens = cutlass.Uint32((_warp_redux_u32_2 * _warp_redux_u32_1))
                if (slot[0] < round_tokens):
                    _vote_2 = cute.arch.vote_ballot_sync(cutlass.Boolean((remaining[0] > 0)), cutlass.Uint32(0xFFFFFFFF)).bitcast(cutlass.Uint32)
                    _find_nth_set_0 = cutlass.Uint32(cutlass_llvm.inline_asm(
                        cutlass.Uint32.mlir_type,
                        [cutlass.Uint32(_vote_2).ir_value(), cutlass.Uint32(0).ir_value(), cutlass.Uint32(cutlass.Int32(((slot[0] % _warp_redux_u32_1) + 1))).ir_value()],
                        'fns.b32 $0, $1, $2, $3;', '=r,r,r,r',
                        has_side_effects=False, is_align_stack=False,
                        asm_dialect=cutlass_llvm.AsmDialect.AD_ATT,
                    ))
                    source_rank[0] = cutlass.Uint32(_find_nth_set_0)
                    source_slot[0] = cutlass.Uint32((offset[0] + cutlass.Uint32(cutlass_arith.divui(cutlass.Uint32(slot[0]).ir_value(), cutlass.Uint32(_warp_redux_u32_1).ir_value()))))
                    selected[0] = cutlass.Uint32(1)
                else:
                    slot[0] = cutlass.Uint32((slot[0] - round_tokens))
                    offset[0] = cutlass.Uint32((offset[0] + _warp_redux_u32_2))
                    _min_0 = cutlass.min(remaining[0], _warp_redux_u32_2)
                    remaining[0] = cutlass.Uint32((remaining[0] - _min_0))
            src_slot = cutlass.Uint32(_indices[(((cutlass.Uint32((expert[0] * 16)) + source_rank[0]) * 6144) + source_slot[0])])
            src_token = cutlass.Uint32(cutlass.Uint32(cutlass_arith.divui(cutlass.Uint32(src_slot).ir_value(), cutlass.Uint32(8).ir_value())))
            pool_token = cutlass.Uint32((((pool_blocks_0[0] * 64) + cutlass.Uint32(token)) - start[0]))
            pool_block = cutlass.Uint32(cutlass.Uint32(cutlass_arith.divui(cutlass.Uint32(pool_token).ir_value(), cutlass.Uint32(64).ir_value())))
            target_1 = cutlass.Uint32((cutlass.Uint32(cutlass_arith.divui(cutlass.Uint32(pool_block).ir_value(), cutlass.Uint32(ring_blocks_1).ir_value())) * 80))
            if (target_1 > 0):
                while prims.load_ext(l1_empty + (pool_block % ring_blocks_1), dtype=cutlass.Uint32, order=prims.MemOrder.ACQUIRE, scope=prims.MemScope.GPU) < cutlass.Uint32(target_1):
                    pass
            if prims.elect_sync():
                prims.cp_async_bulk_shared_cluster_global(
                    cute.make_ptr(cutlass.Uint8, cutlass.Uint32((pull_storage_addr + (warp_1 * 3072))), mem_space=cute.AddressSpace.smem, assumed_align=16), cute.recast_ptr(cute.make_ptr(cutlass.Uint8, cutlass.Uint64((_token_peers[source_rank[0]] + (cutlass.Uint64(src_token) * 3072))), mem_space=cute.AddressSpace.gmem, assumed_align=1), dtype=cutlass.Uint8), pull_addr + warp_1,
                    cutlass.Int32(3072),
                    l2_cache_hint=0x12F0000000000000,
                )
                cute.arch.mbarrier_arrive_and_expect_tx(pull_addr + warp_1, 3072)
            cute.arch.sync_warp()
            remote_sf = cute.make_ptr(cutlass.Uint32, cutlass.Uint64((_sf_peers[source_rank[0]] + (cutlass.Uint64(src_token) * 96))), mem_space=cute.AddressSpace.gmem, assumed_align=4)
            sf_token = cutlass.Uint32(((cutlass.Uint32(token) - start[0]) % 64))
            sf_row = cutlass.Uint32(((((pool_block % ring_blocks_1) * 128) + ((sf_token & 31) * 4)) + (sf_token >> 5)))
            if (lane < 24):
                prims.store_ext(cutlass.Uint32(cute.make_tensor(remote_sf, _flat_layout)[lane]).ir_value(), (XSFData + ((cutlass.Uint32((lane * 657408)) + sf_row))))
            cute.arch.sync_warp()
            if prims.elect_sync():
                remote_weight = cute.make_ptr(cutlass.Float32, cutlass.Uint64(_weight_peers[source_rank[0]]), mem_space=cute.AddressSpace.gmem, assumed_align=4)
                prims.store_ext(cutlass.Float32(cute.make_tensor(remote_weight, _flat_layout)[src_slot]).ir_value(), (RW + ((pool_token % 41088))))
                prims.store_ext(cutlass.Uint32(source_rank[0]).ir_value(), (metadata + ((pool_token * 3))))
                prims.store_ext(cutlass.Uint32(src_token).ir_value(), (metadata + (((pool_token * 3) + 1))))
                prims.store_ext(cutlass.Uint32((src_slot % 8)).ir_value(), (metadata + (((pool_token * 3) + 2))))
                cutlass_llvm.inline_asm(
                    res=None,
                    operands_=[(cutlass.Uint32((pull_addr + warp_1).toint())).ir_value(), (cutlass.Uint32(phase[0])).ir_value()],
                    asm_string='{ .reg .pred p; WAIT: mbarrier.try_wait.parity.shared::cta.b64 p, [$0], $1; @!p bra WAIT; }',
                    constraints='r,r',
                    has_side_effects=True,
                    is_align_stack=False,
                    asm_dialect=cutlass_llvm.AsmDialect.AD_ATT,
                )
                phase[0] = cutlass.Uint32((phase[0] ^ 1))
                prims.cp_async_bulk_global_shared_cta(cute.recast_ptr(XData + ((pool_token % 41088) * 3072), dtype=cutlass.Uint8), cute.make_ptr(cutlass.Uint8, cutlass.Uint32((pull_storage_addr + (warp_1 * 3072))), mem_space=cute.AddressSpace.smem, assumed_align=16), cutlass.Int32(3072), l2_cache_hint=0x1000000000000000)
                cute.arch.cp_async_bulk_commit_group()
                cute.arch.cp_async_bulk_wait_group(0, read=False)
                increment[0] = cutlass.Uint32(1)
                _if_condition_8 = cutlass.Boolean((cutlass.Uint32(token) == (end[0] - 1)))
                increment[0] = cutlass.Uint32(cutlass.select_(_if_condition_8, cutlass.Uint32((64 - ((cutlass.Uint32(token) - start[0]) % 64))), increment[0]))
                cute.arch.atomic_add(l1_full + (pool_block % ring_blocks_1), cutlass.Uint32(increment[0]), sem='release', scope='gpu')
            cute.arch.sync_warp()
        prims.barrier_cta_sync(1, thread_count=384)
        if (bid == 0):
            blocks = cutlass.Uint32(cutlass.Uint32(cutlass_arith.divui(cutlass.Uint32(((cutlass.Uint32(count_1[0]) + 64) - 1)).ir_value(), cutlass.Uint32(64).ir_value())))
            _warp_redux_u32_3 = cutlass.Uint32(cutlass_llvm.inline_asm(
                cutlass.Uint32.mlir_type,
                [cutlass.Uint32(blocks).ir_value()],
                'redux.sync.add.u32 $0, $1, 0xffffffff;', '=r,r',
                has_side_effects=True, is_align_stack=False,
                asm_dialect=cutlass_llvm.AsmDialect.AD_ATT,
            ))
            total_blocks = cutlass.Uint32(_warp_redux_u32_3)
            _min_1 = cutlass.min(total_blocks, 642)
            touched = cutlass.Uint32(_min_1)
            l1f = cute.recast_ptr(l1_full, dtype=cutlass.Uint32)
            l1e = cute.recast_ptr(l1_empty, dtype=cutlass.Uint32)
            l2f = cute.recast_ptr(l2_full, dtype=cutlass.Uint32)
            l2e = cute.recast_ptr(l2_empty, dtype=cutlass.Uint32)
            for index in cutlass.range(cutlass.Int32(tid), cutlass.Int32(512), cutlass.Int32(128)):
                prims.store_ext(cutlass.Uint64(0).ir_value(), (send + (index)))
                prims.store_ext(cutlass.Uint64(0).ir_value(), (rank_counts + (index)))
            if (tid < 32):
                prims.store_ext(cutlass.Uint64(0).ir_value(), (recv + (tid)))
            if (tid < 2):
                prims.store_ext(cutlass.Uint32(0).ir_value(), (claims + (tid)))
            for index_1 in cutlass.range(cutlass.Int32(tid), cutlass.Int32(touched), cutlass.Int32(128)):
                prims.store_ext(cutlass.Uint32(0).ir_value(), (l1f + (index_1)))
                prims.store_ext(cutlass.Uint32(0).ir_value(), (l1e + (index_1)))
                prims.store_ext(cutlass.Uint32(0).ir_value(), (l2f + (index_1)))
                prims.store_ext(cutlass.Uint32(0).ir_value(), (l2e + (index_1)))
        prims.barrier_cta_sync(1, thread_count=384)
    elif warp >= 8 and warp <= 15:
        cute.arch.setmaxregister_increase(208)
        amax = cute.make_rmem_tensor((1,), cutlass.Float32)
        exponent = cute.make_rmem_tensor((1,), cutlass.Uint32)
        e_stage = cute.make_rmem_tensor((1,), cutlass.Uint32)
        task_stage = cute.make_rmem_tensor((1,), cutlass.Uint32)
        running = cute.make_rmem_tensor((1,), cutlass.Int32)
        _phase_task_full = cute.make_rmem_tensor((1,), cutlass.Uint32)
        _phase_done = cute.make_rmem_tensor((1,), cutlass.Uint32)
        cached = cute.make_rmem_tensor((1,), cutlass.Float32)
        weight0 = cute.make_rmem_tensor((1,), cutlass.Float32)
        weight1 = cute.make_rmem_tensor((1,), cutlass.Float32)
        a0 = cute.make_rmem_tensor((1,), cutlass.Float32)
        a1 = cute.make_rmem_tensor((1,), cutlass.Float32)
        cached_1 = cute.make_rmem_tensor((1,), cutlass.Float32)
        weight0_1 = cute.make_rmem_tensor((1,), cutlass.Float32)
        weight1_1 = cute.make_rmem_tensor((1,), cutlass.Float32)
        a0_1 = cute.make_rmem_tensor((1,), cutlass.Float32)
        a1_1 = cute.make_rmem_tensor((1,), cutlass.Float32)
        cached_2 = cute.make_rmem_tensor((1,), cutlass.Float32)
        weight0_2 = cute.make_rmem_tensor((1,), cutlass.Float32)
        weight1_2 = cute.make_rmem_tensor((1,), cutlass.Float32)
        a0_2 = cute.make_rmem_tensor((1,), cutlass.Float32)
        a1_2 = cute.make_rmem_tensor((1,), cutlass.Float32)
        completion_state = cute.make_rmem_tensor((1,), cutlass.Uint32)
        phase_2 = cute.make_rmem_tensor((1,), cutlass.Uint32)
        valid = cute.make_rmem_tensor((1,), cutlass.Int32)
        phase_3 = cute.make_rmem_tensor((1,), cutlass.Uint32)
        valid_1 = cute.make_rmem_tensor((1,), cutlass.Int32)
        if (tid < 384):
            for token_1 in cutlass.range(cutlass.Int32(bid), cutlass.Int32(384), cutlass.Int32(num_bids)):
                thread_idx = cutlass.Uint32((tid - 256))
                if (cutlass.Uint32(token_1) < live_tokens):
                    for i in cutlass.range_constexpr(0, 3, 1):
                        chunk = cutlass.Uint32((thread_idx + cutlass.Uint32((i * 128))))
                        byte = cutlass.Uint32((chunk * 16))
                        swizzled = cutlass.Uint32((byte ^ ((byte >> 3) & 32)))
                        prims.cp_async_shared_global(cute.make_ptr(cutlass.Uint8, cutlass.Uint32((input_row_addr + swizzled)), mem_space=cute.AddressSpace.smem, assumed_align=16), cute.recast_ptr(caller_x + (cutlass.Uint32((token_1 * 3072)) + (chunk * 8)), dtype=cutlass.Uint8), 16, prims.LoadCacheModifier.CA)
                        if ((i == 1) | (i == 2)):
                            cute.arch.cp_async_commit_group()
                    for j in cutlass.range_constexpr(0, 2, 1):
                        cute.arch.cp_async_commit_group()
                        cute.arch.cp_async_wait_group(2)
                        prims.barrier_cta_sync(2, thread_count=128)
                        chunk_1 = cutlass.Uint32((thread_idx + cutlass.Uint32((j * 128))))
                        values = cute.make_rmem_tensor((16,), cutlass.Float32)
                        amax[0] = cutlass.Float32(0.0)
                        if (chunk_1 < 192):
                            byte_1 = cutlass.Uint32((chunk_1 * 32))
                            swizzled_1 = cutlass.Uint32((byte_1 ^ ((byte_1 >> 3) & 32)))
                            _smem_physical_9 = cute.make_tensor(cute.make_ptr(cutlass.BFloat16, cutlass.Uint32(input_row_addr), mem_space=cute.AddressSpace.smem, assumed_align=2), _flat_layout)
                            _input_row_reg_0 = cute.make_rmem_tensor((16,), cutlass.BFloat16)
                            _input_row_reg_0[0] = _smem_physical_9[(cutlass.Uint32(cutlass_arith.divui(cutlass.Uint32(swizzled_1).ir_value(), cutlass.Uint32(2).ir_value()))) + 0]
                            _input_row_reg_0[1] = _smem_physical_9[(cutlass.Uint32(cutlass_arith.divui(cutlass.Uint32(swizzled_1).ir_value(), cutlass.Uint32(2).ir_value()))) + 1]
                            _input_row_reg_0[2] = _smem_physical_9[(cutlass.Uint32(cutlass_arith.divui(cutlass.Uint32(swizzled_1).ir_value(), cutlass.Uint32(2).ir_value()))) + 2]
                            _input_row_reg_0[3] = _smem_physical_9[(cutlass.Uint32(cutlass_arith.divui(cutlass.Uint32(swizzled_1).ir_value(), cutlass.Uint32(2).ir_value()))) + 3]
                            _input_row_reg_0[4] = _smem_physical_9[(cutlass.Uint32(cutlass_arith.divui(cutlass.Uint32(swizzled_1).ir_value(), cutlass.Uint32(2).ir_value()))) + 4]
                            _input_row_reg_0[5] = _smem_physical_9[(cutlass.Uint32(cutlass_arith.divui(cutlass.Uint32(swizzled_1).ir_value(), cutlass.Uint32(2).ir_value()))) + 5]
                            _input_row_reg_0[6] = _smem_physical_9[(cutlass.Uint32(cutlass_arith.divui(cutlass.Uint32(swizzled_1).ir_value(), cutlass.Uint32(2).ir_value()))) + 6]
                            _input_row_reg_0[7] = _smem_physical_9[(cutlass.Uint32(cutlass_arith.divui(cutlass.Uint32(swizzled_1).ir_value(), cutlass.Uint32(2).ir_value()))) + 7]
                            _input_row_reg_0[8] = _smem_physical_9[(cutlass.Uint32(cutlass_arith.divui(cutlass.Uint32(swizzled_1).ir_value(), cutlass.Uint32(2).ir_value()))) + 8]
                            _input_row_reg_0[9] = _smem_physical_9[(cutlass.Uint32(cutlass_arith.divui(cutlass.Uint32(swizzled_1).ir_value(), cutlass.Uint32(2).ir_value()))) + 9]
                            _input_row_reg_0[10] = _smem_physical_9[(cutlass.Uint32(cutlass_arith.divui(cutlass.Uint32(swizzled_1).ir_value(), cutlass.Uint32(2).ir_value()))) + 10]
                            _input_row_reg_0[11] = _smem_physical_9[(cutlass.Uint32(cutlass_arith.divui(cutlass.Uint32(swizzled_1).ir_value(), cutlass.Uint32(2).ir_value()))) + 11]
                            _input_row_reg_0[12] = _smem_physical_9[(cutlass.Uint32(cutlass_arith.divui(cutlass.Uint32(swizzled_1).ir_value(), cutlass.Uint32(2).ir_value()))) + 12]
                            _input_row_reg_0[13] = _smem_physical_9[(cutlass.Uint32(cutlass_arith.divui(cutlass.Uint32(swizzled_1).ir_value(), cutlass.Uint32(2).ir_value()))) + 13]
                            _input_row_reg_0[14] = _smem_physical_9[(cutlass.Uint32(cutlass_arith.divui(cutlass.Uint32(swizzled_1).ir_value(), cutlass.Uint32(2).ir_value()))) + 14]
                            _input_row_reg_0[15] = _smem_physical_9[(cutlass.Uint32(cutlass_arith.divui(cutlass.Uint32(swizzled_1).ir_value(), cutlass.Uint32(2).ir_value()))) + 15]
                            for i_1 in cutlass.range_constexpr(0, 16, 1):
                                _cvt_f32_0 = cutlass.Float32(_input_row_reg_0[i_1])
                                values[i_1] = cutlass.Float32(_cvt_f32_0)
                                _fabs_0 = cute.math.abs(values[i_1])
                                _max_1 = cute.arch.fmax(amax[0], _fabs_0, ftz=False)
                                amax[0] = cutlass.Float32(_max_1)
                        _shfl_xor_0 = cute.arch.shuffle_sync_bfly(amax[0], 1, mask=0xFFFFFFFF, mask_and_clamp=31)
                        _max_2 = cute.arch.fmax(amax[0], _shfl_xor_0, ftz=False)
                        amax[0] = cutlass.Float32(_max_2)
                        if (chunk_1 < 192):
                            scaled = cutlass.Float32((amax[0] * 0.002232142857142857))
                            bits = cutlass.Uint32(cutlass.Float32(scaled).bitcast(cutlass.Uint32))
                            exponent[0] = cutlass.Uint32(((bits >> 23) + cutlass.Uint32((cutlass.Int32(1) if ((bits & 8388607) != 0) else cutlass.Int32(0)))))
                            _if_condition_10 = cutlass.Boolean((bits <= 4194304))
                            exponent[0] = cutlass.Uint32(cutlass.select_(_if_condition_10, cutlass.Uint32(0), exponent[0]))
                            inverse_bits = cutlass.Uint32((cutlass.Uint32(cutlass.Uint32(2139095039)) if (exponent[0] == 0) else cutlass.Uint32(((254 - exponent[0]) << 23))))
                            inverse = cutlass.Float32(cutlass.Uint32(inverse_bits).bitcast(cutlass.Float32))
                            packed = cute.make_rmem_tensor((4,), cutlass.Uint32)
                            for i_2 in cutlass.range_constexpr(0, 4, 1):
                                _e4m3x2_f32_0 = cutlass.Uint16(cutlass_llvm.inline_asm(
                                    cutlass.Uint16.mlir_type,
                                    [cutlass.Float32((values[((i_2 * 4) + 1)] * inverse)).ir_value(), cutlass.Float32((values[(i_2 * 4)] * inverse)).ir_value()],
                                    'cvt.rn.satfinite.e4m3x2.f32 $0, $1, $2;', '=h,f,f',
                                    has_side_effects=False, is_align_stack=False,
                                    asm_dialect=cutlass_llvm.AsmDialect.AD_ATT,
                                ))
                                _e4m3x2_f32_1 = cutlass.Uint16(cutlass_llvm.inline_asm(
                                    cutlass.Uint16.mlir_type,
                                    [cutlass.Float32((values[((i_2 * 4) + 3)] * inverse)).ir_value(), cutlass.Float32((values[((i_2 * 4) + 2)] * inverse)).ir_value()],
                                    'cvt.rn.satfinite.e4m3x2.f32 $0, $1, $2;', '=h,f,f',
                                    has_side_effects=False, is_align_stack=False,
                                    asm_dialect=cutlass_llvm.AsmDialect.AD_ATT,
                                ))
                                packed[i_2] = cutlass.Uint32((cutlass.Uint32(_e4m3x2_f32_0) | (cutlass.Uint32(_e4m3x2_f32_1) << 16)))
                            _gmem_store_raw_11 = cutlass.Vector.from_elements([cutlass.Uint32(packed[0]), cutlass.Uint32(packed[1]), cutlass.Uint32(packed[2]), cutlass.Uint32(packed[3])], cutlass.Uint32)
                            prims.store_ext(_gmem_store_raw_11.ir_value(), staged_x + (cutlass.Uint32((token_1 * 768)) + (chunk_1 * 4)))
                            if ((chunk_1 % 2) == 0):
                                byte_scale = cutlass.Uint8(exponent[0])
                                prims.store_ext(cutlass.Uint8(byte_scale).ir_value(), (staged_sf + ((cutlass.Uint32((token_1 * 96)) + cutlass.Uint32(cutlass_arith.divui(cutlass.Uint32(chunk_1).ir_value(), cutlass.Uint32(2).ir_value()))))))
                    if (thread_idx < 8):
                        prims.store_ext(cutlass.Int64(_ids[(cutlass.Uint32((token_1 * 8)) + thread_idx)]).ir_value(), (staged_ids + ((cutlass.Uint32((token_1 * 8)) + thread_idx))))
                        prims.store_ext(cutlass.Float32(_caller_rw[(cutlass.Uint32((token_1 * 8)) + thread_idx)]).ir_value(), (staged_rw + ((cutlass.Uint32((token_1 * 8)) + thread_idx))))
                else:
                    if (thread_idx < 8):
                        prims.store_ext(cutlass.Int64(-1).ir_value(), (staged_ids + ((cutlass.Uint32((token_1 * 8)) + thread_idx))))
                prims.barrier_cta_sync(2, thread_count=128)
        prims.barrier_cta_sync(1, thread_count=384)
        prims.barrier_cta_sync(1, thread_count=384)
        e_stage[0] = cutlass.Uint32(0)
        task_stage[0] = cutlass.Uint32(0)
        running[0] = cutlass.Int32(1)
        _phase_task_full[0] = cutlass.Uint32(0)
        _phase_done[0] = cutlass.Uint32(0)
        while cutlass.Boolean((running[0] != 0)):
            while not prims.mbarrier_wait_parity(task_full_addr + task_stage[0], _phase_task_full[0], prims.MBarrierWait.TRY, scope=prims.MBarrierScope.CTA, order=prims.MemOrder.ACQUIRE):
                pass
            _smem_physical_12 = cute.make_tensor(cute.make_ptr(cutlass.Uint32, cutlass.Uint32(tasks_addr), mem_space=cute.AddressSpace.smem, assumed_align=4), _flat_layout)
            _tasks_reg_3 = cute.make_rmem_tensor((8,), cutlass.Uint32)
            _tasks_reg_3[0] = _smem_physical_12[((task_stage[0] * 8)) + 0]
            _tasks_reg_3[1] = _smem_physical_12[((task_stage[0] * 8)) + 1]
            _tasks_reg_3[2] = _smem_physical_12[((task_stage[0] * 8)) + 2]
            _tasks_reg_3[3] = _smem_physical_12[((task_stage[0] * 8)) + 3]
            _tasks_reg_3[4] = _smem_physical_12[((task_stage[0] * 8)) + 4]
            _tasks_reg_3[5] = _smem_physical_12[((task_stage[0] * 8)) + 5]
            _tasks_reg_3[6] = _smem_physical_12[((task_stage[0] * 8)) + 6]
            _tasks_reg_3[7] = _smem_physical_12[((task_stage[0] * 8)) + 7]
            if (_tasks_reg_3[0] == 0):
                running[0] = cutlass.Int32(0)
            else:
                phase_4 = cutlass.Uint32((_tasks_reg_3[0] - 1))
                tile = cutlass.Uint32(_tasks_reg_3[3])
                expert_2 = cutlass.Uint32(_tasks_reg_3[1])
                logical_block = cutlass.Uint32(_tasks_reg_3[4])
                pool_block_1 = cutlass.Uint32((logical_block % cutlass.Uint32(ring_blocks)))
                ring_epoch = cutlass.Uint32(cutlass.Uint32(cutlass_arith.divui(cutlass.Uint32(logical_block).ir_value(), cutlass.Uint32(cutlass.Uint32(ring_blocks)).ir_value())))
                task_valid = cutlass.Uint32(_tasks_reg_3[5])
                task_n = cutlass.Uint32((cutlass.Uint32(cutlass.Uint32(64)) if (task_valid > 32) else cutlass.Uint32((cutlass.Uint32(cutlass_arith.divui(cutlass.Uint32((task_valid + 15)).ir_value(), cutlass.Uint32(16).ir_value())) * 16))))
                k_count = cutlass.Uint32(cutlass.Uint32(cutlass_arith.divui(cutlass.Uint32(_tasks_reg_3[7]).ir_value(), cutlass.Uint32(128).ir_value())))
                while not prims.mbarrier_wait_parity(done_addr + e_stage[0], _phase_done[0], prims.MBarrierWait.TRY, scope=prims.MBarrierScope.CTA, order=prims.MemOrder.ACQUIRE):
                    pass
                prims.tcgen05_fence('after_thread_sync')
                _mbarrier_cluster_arrive_13 = prims.mapa(task_empty_addr + task_stage[0], cutlass.Int32(cta_rank & 0xFFFFFFFE))
                prims.mbarrier_arrive(_mbarrier_cluster_arrive_13, count=cutlass.Int32(1), scope=prims.MemScope.CTA)
                if (phase_4 == 0):
                    while prims.load_ext(l2_empty + pool_block_1, dtype=cutlass.Uint32, order=prims.MemOrder.ACQUIRE, scope=prims.MemScope.GPU) != cutlass.Uint32((ring_epoch * 24)):
                        pass
                    if (task_n == 16):
                        warp_0 = cutlass.Int32((warp % 4))
                        wg = cutlass.Int32(cutlass.Int32(cutlass_arith.divsi(cutlass.Int32((warp - 8)).ir_value(), cutlass.Int32(4).ir_value())))
                        if (task_valid <= cutlass.Uint32((wg * 8))):
                            prims.tcgen05_fence('before_thread_sync')
                            _mbarrier_cluster_arrive_14 = prims.mapa(released_addr + e_stage[0], cutlass.Int32(cta_rank & 0xFFFFFFFE))
                            prims.mbarrier_arrive(_mbarrier_cluster_arrive_14, count=cutlass.Int32(1), scope=prims.MemScope.CTA)
                        else:
                            cached[0] = cutlass.Float32(0.0)
                            if (lane < 8):
                                cached[0] = cutlass.Float32(_RW[(((pool_block_1 * 64) + cutlass.Uint32((wg * 8))) + cutlass.Uint32(lane))])
                            values_1 = cute.make_rmem_tensor((4,), cutlass.Float32)
                            maxima = cute.make_rmem_tensor((2,), cutlass.Float32)
                            for atom in cutlass.range_constexpr(0, 1, 1):
                                token_2 = cutlass.Int32((((wg * 8) + (atom * 8)) + ((lane % 4) * 2)))
                                weight0[0] = cutlass.Float32(0.0)
                                weight1[0] = cutlass.Float32(0.0)
                                _shfl_8 = cute.arch.shuffle_sync(cached[0], ((atom * 8) + ((lane % 4) * 2)), mask=4294967295, mask_and_clamp=31)
                                weight0[0] = cutlass.Float32(_shfl_8)
                                _shfl_9 = cute.arch.shuffle_sync(cached[0], (((atom * 8) + ((lane % 4) * 2)) + 1), mask=4294967295, mask_and_clamp=31)
                                weight1[0] = cutlass.Float32(_shfl_9)
                                address = cutlass.Int32((((taddr + (e_stage[0] * 64)) + cutlass.Uint32((wg * 8))) + cutlass.Uint32((atom * 8))))
                                _tmem_load_0 = cute.make_rmem_tensor((4,), cutlass.Float32)
                                _tmem_load_15_atom = cute.make_copy_atom(cute.nvgpu.tcgen05.Ld16x256bOp(cute.nvgpu.tcgen05.Repetition.x1), cutlass.Float32)
                                _tmem_load_15_src = cute.make_tensor(cute.make_ptr(cutlass.Float32, cutlass.Uint32(address), mem_space=cutlass.AddressSpace.tmem, assumed_align=16), cute.make_layout((4,), stride=(1,)))
                                _tmem_load_15_dst = cute.make_tensor(_tmem_load_0.iterator + 0, cute.make_layout((4,), stride=(1,)))
                                cute.copy(_tmem_load_15_atom, _tmem_load_15_src, _tmem_load_15_dst)
                                cute.arch.fence_view_async_tmem_load()
                                _tmem_load_1 = cute.make_rmem_tensor((4,), cutlass.Float32)
                                _tmem_load_16_atom = cute.make_copy_atom(cute.nvgpu.tcgen05.Ld16x256bOp(cute.nvgpu.tcgen05.Repetition.x1), cutlass.Float32)
                                _tmem_load_16_src = cute.make_tensor(cute.make_ptr(cutlass.Float32, cutlass.Uint32((address + 1048576)), mem_space=cutlass.AddressSpace.tmem, assumed_align=16), cute.make_layout((4,), stride=(1,)))
                                _tmem_load_16_dst = cute.make_tensor(_tmem_load_1.iterator + 0, cute.make_layout((4,), stride=(1,)))
                                cute.copy(_tmem_load_16_atom, _tmem_load_16_src, _tmem_load_16_dst)
                                cute.arch.fence_view_async_tmem_load()
                                if (atom == 0):
                                    prims.tcgen05_fence('before_thread_sync')
                                    _mbarrier_cluster_arrive_17 = prims.mapa(released_addr + e_stage[0], cutlass.Int32(cta_rank & 0xFFFFFFFE))
                                    prims.mbarrier_arrive(_mbarrier_cluster_arrive_17, count=cutlass.Int32(1), scope=prims.MemScope.CTA)
                                if (not True):
                                    weight0[0] = cutlass.Float32(_RW[((pool_block_1 * 64) + cutlass.Uint32(token_2))])
                                    weight1[0] = cutlass.Float32(_RW[(((pool_block_1 * 64) + cutlass.Uint32(token_2)) + 1)])
                                _cvt_bf16_0 = cutlass.BFloat16(_tmem_load_0[0])
                                _cvt_f32_1 = cutlass.Float32(_cvt_bf16_0)
                                _cvt_bf16_1 = cutlass.BFloat16(_tmem_load_0[1])
                                _cvt_f32_2 = cutlass.Float32(_cvt_bf16_1)
                                _f2_0 = (cutlass.Float32(_cvt_f32_1), cutlass.Float32(_cvt_f32_2))
                                _cvt_bf16_2 = cutlass.BFloat16(_tmem_load_0[2])
                                _cvt_f32_3 = cutlass.Float32(_cvt_bf16_2)
                                _cvt_bf16_3 = cutlass.BFloat16(_tmem_load_0[3])
                                _cvt_f32_4 = cutlass.Float32(_cvt_bf16_3)
                                _f2_1 = (cutlass.Float32(_cvt_f32_3), cutlass.Float32(_cvt_f32_4))
                                _expf_0 = cute.math.exp2(cutlass.Float32((0 - _f2_0[0])) * cutlass.Float32(1.4426950408889634), approx=True, ftz=True)
                                _expf_1 = cute.math.exp2(cutlass.Float32((0 - _f2_0[1])) * cutlass.Float32(1.4426950408889634), approx=True, ftz=True)
                                _f2_2 = (cutlass.Float32(_expf_0), cutlass.Float32(_expf_1))
                                _f2_3 = (cutlass.Float32(1.0), cutlass.Float32(1.0))
                                _packed_add_f32x2_0 = cute.arch.add_packed_f32x2((cutlass.Float32(_f2_3[0]), cutlass.Float32(_f2_3[1])), (cutlass.Float32(_f2_2[0]), cutlass.Float32(_f2_2[1])), ftz=False)
                                _rcp_0 = cute.math.rcp(_packed_add_f32x2_0[0], approx=True, ftz=True)
                                _rcp_1 = cute.math.rcp(_packed_add_f32x2_0[1], approx=True, ftz=True)
                                _f2_4 = (cutlass.Float32(_rcp_0), cutlass.Float32(_rcp_1))
                                _mul_f32x2_0 = cute.arch.mul_packed_f32x2((cutlass.Float32(_f2_0[0]), cutlass.Float32(_f2_0[1])), (cutlass.Float32(_f2_4[0]), cutlass.Float32(_f2_4[1])), rnd='rn', ftz=False)
                                _mul_f32x2_1 = cute.arch.mul_packed_f32x2((cutlass.Float32(_mul_f32x2_0[0]), cutlass.Float32(_mul_f32x2_0[1])), (cutlass.Float32(_f2_1[0]), cutlass.Float32(_f2_1[1])), rnd='rn', ftz=False)
                                _f2_5 = (cutlass.Float32(weight0[0]), cutlass.Float32(weight1[0]))
                                _mul_f32x2_2 = cute.arch.mul_packed_f32x2((cutlass.Float32(_mul_f32x2_1[0]), cutlass.Float32(_mul_f32x2_1[1])), (cutlass.Float32(_f2_5[0]), cutlass.Float32(_f2_5[1])), rnd='rn', ftz=False)
                                _cvt_bf16_4 = cutlass.BFloat16(_tmem_load_1[0])
                                _cvt_f32_5 = cutlass.Float32(_cvt_bf16_4)
                                _cvt_bf16_5 = cutlass.BFloat16(_tmem_load_1[1])
                                _cvt_f32_6 = cutlass.Float32(_cvt_bf16_5)
                                _f2_6 = (cutlass.Float32(_cvt_f32_5), cutlass.Float32(_cvt_f32_6))
                                _cvt_bf16_6 = cutlass.BFloat16(_tmem_load_1[2])
                                _cvt_f32_7 = cutlass.Float32(_cvt_bf16_6)
                                _cvt_bf16_7 = cutlass.BFloat16(_tmem_load_1[3])
                                _cvt_f32_8 = cutlass.Float32(_cvt_bf16_7)
                                _f2_7 = (cutlass.Float32(_cvt_f32_7), cutlass.Float32(_cvt_f32_8))
                                _expf_2 = cute.math.exp2(cutlass.Float32((0 - _f2_6[0])) * cutlass.Float32(1.4426950408889634), approx=True, ftz=True)
                                _expf_3 = cute.math.exp2(cutlass.Float32((0 - _f2_6[1])) * cutlass.Float32(1.4426950408889634), approx=True, ftz=True)
                                _f2_8 = (cutlass.Float32(_expf_2), cutlass.Float32(_expf_3))
                                _f2_9 = (cutlass.Float32(1.0), cutlass.Float32(1.0))
                                _packed_add_f32x2_1 = cute.arch.add_packed_f32x2((cutlass.Float32(_f2_9[0]), cutlass.Float32(_f2_9[1])), (cutlass.Float32(_f2_8[0]), cutlass.Float32(_f2_8[1])), ftz=False)
                                _rcp_2 = cute.math.rcp(_packed_add_f32x2_1[0], approx=True, ftz=True)
                                _rcp_3 = cute.math.rcp(_packed_add_f32x2_1[1], approx=True, ftz=True)
                                _f2_10 = (cutlass.Float32(_rcp_2), cutlass.Float32(_rcp_3))
                                _mul_f32x2_3 = cute.arch.mul_packed_f32x2((cutlass.Float32(_f2_6[0]), cutlass.Float32(_f2_6[1])), (cutlass.Float32(_f2_10[0]), cutlass.Float32(_f2_10[1])), rnd='rn', ftz=False)
                                _mul_f32x2_4 = cute.arch.mul_packed_f32x2((cutlass.Float32(_mul_f32x2_3[0]), cutlass.Float32(_mul_f32x2_3[1])), (cutlass.Float32(_f2_7[0]), cutlass.Float32(_f2_7[1])), rnd='rn', ftz=False)
                                _f2_11 = (cutlass.Float32(weight0[0]), cutlass.Float32(weight1[0]))
                                _mul_f32x2_5 = cute.arch.mul_packed_f32x2((cutlass.Float32(_mul_f32x2_4[0]), cutlass.Float32(_mul_f32x2_4[1])), (cutlass.Float32(_f2_11[0]), cutlass.Float32(_f2_11[1])), rnd='rn', ftz=False)
                                values_1[(atom * 4)] = cutlass.Float32(_mul_f32x2_2[0])
                                values_1[((atom * 4) + 1)] = cutlass.Float32(_mul_f32x2_2[1])
                                values_1[((atom * 4) + 2)] = cutlass.Float32(_mul_f32x2_5[0])
                                values_1[((atom * 4) + 3)] = cutlass.Float32(_mul_f32x2_5[1])
                                _fabs_1 = cute.math.abs(values_1[(atom * 4)])
                                _fabs_2 = cute.math.abs(values_1[((atom * 4) + 2)])
                                _max_3 = cute.arch.fmax(_fabs_1, _fabs_2, ftz=False)
                                a0[0] = cutlass.Float32(_max_3)
                                _fabs_3 = cute.math.abs(values_1[((atom * 4) + 1)])
                                _fabs_4 = cute.math.abs(values_1[((atom * 4) + 3)])
                                _max_4 = cute.arch.fmax(_fabs_3, _fabs_4, ftz=False)
                                a1[0] = cutlass.Float32(_max_4)
                                _shfl_xor_1 = cute.arch.shuffle_sync_bfly(a0[0], 4, mask=0xFFFFFFFF, mask_and_clamp=31)
                                _max_5 = cute.arch.fmax(a0[0], _shfl_xor_1, ftz=False)
                                a0[0] = cutlass.Float32(_max_5)
                                _shfl_xor_2 = cute.arch.shuffle_sync_bfly(a1[0], 4, mask=0xFFFFFFFF, mask_and_clamp=31)
                                _max_6 = cute.arch.fmax(a1[0], _shfl_xor_2, ftz=False)
                                a1[0] = cutlass.Float32(_max_6)
                                _shfl_xor_3 = cute.arch.shuffle_sync_bfly(a0[0], 8, mask=0xFFFFFFFF, mask_and_clamp=31)
                                _max_7 = cute.arch.fmax(a0[0], _shfl_xor_3, ftz=False)
                                a0[0] = cutlass.Float32(_max_7)
                                _shfl_xor_4 = cute.arch.shuffle_sync_bfly(a1[0], 8, mask=0xFFFFFFFF, mask_and_clamp=31)
                                _max_8 = cute.arch.fmax(a1[0], _shfl_xor_4, ftz=False)
                                a1[0] = cutlass.Float32(_max_8)
                                _shfl_xor_5 = cute.arch.shuffle_sync_bfly(a0[0], 16, mask=0xFFFFFFFF, mask_and_clamp=31)
                                _max_9 = cute.arch.fmax(a0[0], _shfl_xor_5, ftz=False)
                                a0[0] = cutlass.Float32(_max_9)
                                _shfl_xor_6 = cute.arch.shuffle_sync_bfly(a1[0], 16, mask=0xFFFFFFFF, mask_and_clamp=31)
                                _max_10 = cute.arch.fmax(a1[0], _shfl_xor_6, ftz=False)
                                a1[0] = cutlass.Float32(_max_10)
                                maxima[(atom * 2)] = cutlass.Float32(a0[0])
                                maxima[((atom * 2) + 1)] = cutlass.Float32(a1[0])
                                if (lane < 4):
                                    prims.store_ext(cutlass.Uint32(cutlass.Float32(a0[0]).bitcast(cutlass.Uint32)).ir_value(), cute.make_ptr(cutlass.Uint32, cutlass.Uint32((((scratch_addr + cutlass.Uint32((((wg * 4) + warp_0) * 32))) + cutlass.Uint32((atom * 32))) + cutlass.Uint32((lane * 8)))), mem_space=cute.AddressSpace.smem, assumed_align=4))
                                    prims.store_ext(cutlass.Uint32(cutlass.Float32(a1[0]).bitcast(cutlass.Uint32)).ir_value(), cute.make_ptr(cutlass.Uint32, cutlass.Uint32(((((scratch_addr + cutlass.Uint32((((wg * 4) + warp_0) * 32))) + cutlass.Uint32((atom * 32))) + cutlass.Uint32((lane * 8))) + 4)), mem_space=cute.AddressSpace.smem, assumed_align=4))
                                cute.arch.sync_warp()
                            if (warp_0 == 0):
                                if prims.elect_sync():
                                    cute.arch.cp_async_bulk_wait_group(1, read=False)
                            prims.barrier_cta_sync((10 + wg), thread_count=128)
                            for atom_1 in cutlass.range_constexpr(0, 1, 1):
                                token_3 = cutlass.Int32((((wg * 8) + (atom_1 * 8)) + ((lane % 4) * 2)))
                                _smem_physical_18 = cute.make_tensor(cute.make_ptr(cutlass.Float32, cutlass.Uint32(scratch_addr), mem_space=cute.AddressSpace.smem, assumed_align=4), _flat_layout)
                                _scratch_reg_0 = cute.make_rmem_tensor((2,), cutlass.Float32)
                                _scratch_reg_0[0] = _smem_physical_18[((((((wg * 4) + (warp_0 ^ 1)) * 8) + (atom_1 * 8)) + ((lane % 4) * 2))) + 0]
                                _scratch_reg_0[1] = _smem_physical_18[((((((wg * 4) + (warp_0 ^ 1)) * 8) + (atom_1 * 8)) + ((lane % 4) * 2))) + 1]
                                quantized = cute.make_rmem_tensor((4,), cutlass.Float32)
                                for t in cutlass.range_constexpr(0, 2, 1):
                                    _max_11 = cute.arch.fmax(maxima[((atom_1 * 2) + t)], _scratch_reg_0[t], ftz=False)
                                    amax_1 = cutlass.Float32(_max_11)
                                    scaled_1 = cutlass.Float32((amax_1 * 0.002232142857142857))
                                    bits_1 = cutlass.Uint32(cutlass.Float32(scaled_1).bitcast(cutlass.Uint32))
                                    exponent_1 = cutlass.Uint32(((bits_1 >> 23) + cutlass.Uint32((cutlass.Int32(1) if ((bits_1 & 8388607) != 0) else cutlass.Int32(0)))))
                                    inv_bits = cutlass.Uint32(((254 - exponent_1) << 23))
                                    inverse_1 = cutlass.Float32(cutlass.Uint32(inv_bits).bitcast(cutlass.Float32))
                                    if (((warp_0 % 2) == 0) & (lane < 4)):
                                        sf_byte = cutlass.Uint8(exponent_1)
                                        prims.store_ext(cutlass.Uint8(sf_byte).ir_value(), (SF + ((((((((tile * pool_blocks) * 512) + (pool_block_1 * 512)) + cutlass.Uint32((((token_3 + t) & 31) * 16))) + cutlass.Uint32((((token_3 + t) >> 5) * 4))) + cutlass.Uint32((cta_rank * 2))) + cutlass.Uint32(cutlass.Int32(cutlass_arith.divsi(cutlass.Int32(warp_0).ir_value(), cutlass.Int32(2).ir_value())))))))
                                    for k in cutlass.range_constexpr(0, 2, 1):
                                        channel = cutlass.Int32(((((cta_rank * 64) + (warp_0 * 16)) + cutlass.Int32(cutlass_arith.divsi(cutlass.Int32(lane).ir_value(), cutlass.Int32(4).ir_value()))) + (k * 8)))
                                        value_1 = cutlass.Float32(values_1[(((atom_1 * 4) + (k * 2)) + t)])
                                        quantized[((k * 2) + t)] = cutlass.Float32((value_1 * inverse_1))
                                _e4m3x2_f32_2 = cutlass.Uint16(cutlass_llvm.inline_asm(
                                    cutlass.Uint16.mlir_type,
                                    [cutlass.Float32(quantized[1]).ir_value(), cutlass.Float32(quantized[0]).ir_value()],
                                    'cvt.rn.satfinite.e4m3x2.f32 $0, $1, $2;', '=h,f,f',
                                    has_side_effects=False, is_align_stack=False,
                                    asm_dialect=cutlass_llvm.AsmDialect.AD_ATT,
                                ))
                                _e4m3x2_f32_3 = cutlass.Uint16(cutlass_llvm.inline_asm(
                                    cutlass.Uint16.mlir_type,
                                    [cutlass.Float32(quantized[3]).ir_value(), cutlass.Float32(quantized[2]).ir_value()],
                                    'cvt.rn.satfinite.e4m3x2.f32 $0, $1, $2;', '=h,f,f',
                                    has_side_effects=False, is_align_stack=False,
                                    asm_dialect=cutlass_llvm.AsmDialect.AD_ATT,
                                ))
                                packed_1 = cutlass.Uint32((cutlass.Uint32(_e4m3x2_f32_2) | (cutlass.Uint32(_e4m3x2_f32_3) << 16)))
                                addr = cutlass.Uint32(((((staging_addr + cutlass.Uint32((wg * 1024))) + cutlass.Uint32((atom_1 * 512))) + cutlass.Uint32((lane * 64))) + cutlass.Uint32(((warp_0 ^ cutlass.Int32(cutlass_arith.divsi(cutlass.Int32(lane).ir_value(), cutlass.Int32(2).ir_value()))) * 16))))
                                cutlass_llvm.inline_asm(
                                    res=None,
                                    operands_=[(cutlass.Uint32(addr)).ir_value(), (cutlass.Uint32(packed_1)).ir_value()],
                                    asm_string='stmatrix.sync.aligned.m16n8.x1.trans.shared.b8 [$0], {$1};',
                                    constraints='r,r,~{memory}',
                                    has_side_effects=True,
                                    is_align_stack=False,
                                    asm_dialect=cutlass_llvm.AsmDialect.AD_ATT,
                                )
                                cute.arch.sync_warp()
                            cute.arch.fence_proxy("async.shared", space="cta")
                            prims.barrier_cta_sync((10 + wg), thread_count=128)
                            if (warp_0 == 0):
                                if prims.elect_sync():
                                    prims.cp_async_bulk_tensor_global_shared_cta(
                                        Q8.get_ptr(),
                                        cute.make_ptr(cutlass.Uint8, cutlass.Uint32((staging_addr + cutlass.Uint32((wg * 1024)))), mem_space=cute.AddressSpace.smem, assumed_align=16),
                                        [cutlass.Int32(((tile * 128) + cutlass.Uint32((cta_rank * 64)))), cutlass.Int32(((pool_block_1 * 64) + cutlass.Uint32((wg * 8))))],
                                        mode=prims.TMAStoreMode.TILE,
                                    )
                                    cute.arch.cp_async_bulk_commit_group()
                        if (warp_0 == 0):
                            if prims.elect_sync():
                                cute.arch.cp_async_bulk_wait_group(0, read=False)
                        prims.barrier_cta_sync(15, thread_count=256)
                    else:
                        if (task_n == 32):
                            warp_0_1 = cutlass.Int32((warp % 4))
                            wg_1 = cutlass.Int32(cutlass.Int32(cutlass_arith.divsi(cutlass.Int32((warp - 8)).ir_value(), cutlass.Int32(4).ir_value())))
                            if (task_valid <= cutlass.Uint32((wg_1 * 16))):
                                prims.tcgen05_fence('before_thread_sync')
                                _mbarrier_cluster_arrive_19 = prims.mapa(released_addr + e_stage[0], cutlass.Int32(cta_rank & 0xFFFFFFFE))
                                prims.mbarrier_arrive(_mbarrier_cluster_arrive_19, count=cutlass.Int32(1), scope=prims.MemScope.CTA)
                            else:
                                cached_1[0] = cutlass.Float32(0.0)
                                if (lane < 16):
                                    cached_1[0] = cutlass.Float32(_RW[(((pool_block_1 * 64) + cutlass.Uint32((wg_1 * 16))) + cutlass.Uint32(lane))])
                                values_2 = cute.make_rmem_tensor((8,), cutlass.Float32)
                                maxima_1 = cute.make_rmem_tensor((4,), cutlass.Float32)
                                for atom_2 in cutlass.range_constexpr(0, 2, 1):
                                    token_4 = cutlass.Int32((((wg_1 * 16) + (atom_2 * 8)) + ((lane % 4) * 2)))
                                    weight0_1[0] = cutlass.Float32(0.0)
                                    weight1_1[0] = cutlass.Float32(0.0)
                                    _shfl_10 = cute.arch.shuffle_sync(cached_1[0], ((atom_2 * 8) + ((lane % 4) * 2)), mask=4294967295, mask_and_clamp=31)
                                    weight0_1[0] = cutlass.Float32(_shfl_10)
                                    _shfl_11 = cute.arch.shuffle_sync(cached_1[0], (((atom_2 * 8) + ((lane % 4) * 2)) + 1), mask=4294967295, mask_and_clamp=31)
                                    weight1_1[0] = cutlass.Float32(_shfl_11)
                                    address_1 = cutlass.Int32((((taddr + (e_stage[0] * 64)) + cutlass.Uint32((wg_1 * 16))) + cutlass.Uint32((atom_2 * 8))))
                                    _tmem_load_2 = cute.make_rmem_tensor((4,), cutlass.Float32)
                                    _tmem_load_20_atom = cute.make_copy_atom(cute.nvgpu.tcgen05.Ld16x256bOp(cute.nvgpu.tcgen05.Repetition.x1), cutlass.Float32)
                                    _tmem_load_20_src = cute.make_tensor(cute.make_ptr(cutlass.Float32, cutlass.Uint32(address_1), mem_space=cutlass.AddressSpace.tmem, assumed_align=16), cute.make_layout((4,), stride=(1,)))
                                    _tmem_load_20_dst = cute.make_tensor(_tmem_load_2.iterator + 0, cute.make_layout((4,), stride=(1,)))
                                    cute.copy(_tmem_load_20_atom, _tmem_load_20_src, _tmem_load_20_dst)
                                    cute.arch.fence_view_async_tmem_load()
                                    _tmem_load_3 = cute.make_rmem_tensor((4,), cutlass.Float32)
                                    _tmem_load_21_atom = cute.make_copy_atom(cute.nvgpu.tcgen05.Ld16x256bOp(cute.nvgpu.tcgen05.Repetition.x1), cutlass.Float32)
                                    _tmem_load_21_src = cute.make_tensor(cute.make_ptr(cutlass.Float32, cutlass.Uint32((address_1 + 1048576)), mem_space=cutlass.AddressSpace.tmem, assumed_align=16), cute.make_layout((4,), stride=(1,)))
                                    _tmem_load_21_dst = cute.make_tensor(_tmem_load_3.iterator + 0, cute.make_layout((4,), stride=(1,)))
                                    cute.copy(_tmem_load_21_atom, _tmem_load_21_src, _tmem_load_21_dst)
                                    cute.arch.fence_view_async_tmem_load()
                                    if (atom_2 == 1):
                                        prims.tcgen05_fence('before_thread_sync')
                                        _mbarrier_cluster_arrive_22 = prims.mapa(released_addr + e_stage[0], cutlass.Int32(cta_rank & 0xFFFFFFFE))
                                        prims.mbarrier_arrive(_mbarrier_cluster_arrive_22, count=cutlass.Int32(1), scope=prims.MemScope.CTA)
                                    if (not True):
                                        weight0_1[0] = cutlass.Float32(_RW[((pool_block_1 * 64) + cutlass.Uint32(token_4))])
                                        weight1_1[0] = cutlass.Float32(_RW[(((pool_block_1 * 64) + cutlass.Uint32(token_4)) + 1)])
                                    _cvt_bf16_8 = cutlass.BFloat16(_tmem_load_2[0])
                                    _cvt_f32_9 = cutlass.Float32(_cvt_bf16_8)
                                    _cvt_bf16_9 = cutlass.BFloat16(_tmem_load_2[1])
                                    _cvt_f32_10 = cutlass.Float32(_cvt_bf16_9)
                                    _f2_12 = (cutlass.Float32(_cvt_f32_9), cutlass.Float32(_cvt_f32_10))
                                    _cvt_bf16_10 = cutlass.BFloat16(_tmem_load_2[2])
                                    _cvt_f32_11 = cutlass.Float32(_cvt_bf16_10)
                                    _cvt_bf16_11 = cutlass.BFloat16(_tmem_load_2[3])
                                    _cvt_f32_12 = cutlass.Float32(_cvt_bf16_11)
                                    _f2_13 = (cutlass.Float32(_cvt_f32_11), cutlass.Float32(_cvt_f32_12))
                                    _expf_4 = cute.math.exp2(cutlass.Float32((0 - _f2_12[0])) * cutlass.Float32(1.4426950408889634), approx=True, ftz=True)
                                    _expf_5 = cute.math.exp2(cutlass.Float32((0 - _f2_12[1])) * cutlass.Float32(1.4426950408889634), approx=True, ftz=True)
                                    _f2_14 = (cutlass.Float32(_expf_4), cutlass.Float32(_expf_5))
                                    _f2_15 = (cutlass.Float32(1.0), cutlass.Float32(1.0))
                                    _packed_add_f32x2_2 = cute.arch.add_packed_f32x2((cutlass.Float32(_f2_15[0]), cutlass.Float32(_f2_15[1])), (cutlass.Float32(_f2_14[0]), cutlass.Float32(_f2_14[1])), ftz=False)
                                    _rcp_4 = cute.math.rcp(_packed_add_f32x2_2[0], approx=True, ftz=True)
                                    _rcp_5 = cute.math.rcp(_packed_add_f32x2_2[1], approx=True, ftz=True)
                                    _f2_16 = (cutlass.Float32(_rcp_4), cutlass.Float32(_rcp_5))
                                    _mul_f32x2_6 = cute.arch.mul_packed_f32x2((cutlass.Float32(_f2_12[0]), cutlass.Float32(_f2_12[1])), (cutlass.Float32(_f2_16[0]), cutlass.Float32(_f2_16[1])), rnd='rn', ftz=False)
                                    _mul_f32x2_7 = cute.arch.mul_packed_f32x2((cutlass.Float32(_mul_f32x2_6[0]), cutlass.Float32(_mul_f32x2_6[1])), (cutlass.Float32(_f2_13[0]), cutlass.Float32(_f2_13[1])), rnd='rn', ftz=False)
                                    _f2_17 = (cutlass.Float32(weight0_1[0]), cutlass.Float32(weight1_1[0]))
                                    _mul_f32x2_8 = cute.arch.mul_packed_f32x2((cutlass.Float32(_mul_f32x2_7[0]), cutlass.Float32(_mul_f32x2_7[1])), (cutlass.Float32(_f2_17[0]), cutlass.Float32(_f2_17[1])), rnd='rn', ftz=False)
                                    _cvt_bf16_12 = cutlass.BFloat16(_tmem_load_3[0])
                                    _cvt_f32_13 = cutlass.Float32(_cvt_bf16_12)
                                    _cvt_bf16_13 = cutlass.BFloat16(_tmem_load_3[1])
                                    _cvt_f32_14 = cutlass.Float32(_cvt_bf16_13)
                                    _f2_18 = (cutlass.Float32(_cvt_f32_13), cutlass.Float32(_cvt_f32_14))
                                    _cvt_bf16_14 = cutlass.BFloat16(_tmem_load_3[2])
                                    _cvt_f32_15 = cutlass.Float32(_cvt_bf16_14)
                                    _cvt_bf16_15 = cutlass.BFloat16(_tmem_load_3[3])
                                    _cvt_f32_16 = cutlass.Float32(_cvt_bf16_15)
                                    _f2_19 = (cutlass.Float32(_cvt_f32_15), cutlass.Float32(_cvt_f32_16))
                                    _expf_6 = cute.math.exp2(cutlass.Float32((0 - _f2_18[0])) * cutlass.Float32(1.4426950408889634), approx=True, ftz=True)
                                    _expf_7 = cute.math.exp2(cutlass.Float32((0 - _f2_18[1])) * cutlass.Float32(1.4426950408889634), approx=True, ftz=True)
                                    _f2_20 = (cutlass.Float32(_expf_6), cutlass.Float32(_expf_7))
                                    _f2_21 = (cutlass.Float32(1.0), cutlass.Float32(1.0))
                                    _packed_add_f32x2_3 = cute.arch.add_packed_f32x2((cutlass.Float32(_f2_21[0]), cutlass.Float32(_f2_21[1])), (cutlass.Float32(_f2_20[0]), cutlass.Float32(_f2_20[1])), ftz=False)
                                    _rcp_6 = cute.math.rcp(_packed_add_f32x2_3[0], approx=True, ftz=True)
                                    _rcp_7 = cute.math.rcp(_packed_add_f32x2_3[1], approx=True, ftz=True)
                                    _f2_22 = (cutlass.Float32(_rcp_6), cutlass.Float32(_rcp_7))
                                    _mul_f32x2_9 = cute.arch.mul_packed_f32x2((cutlass.Float32(_f2_18[0]), cutlass.Float32(_f2_18[1])), (cutlass.Float32(_f2_22[0]), cutlass.Float32(_f2_22[1])), rnd='rn', ftz=False)
                                    _mul_f32x2_10 = cute.arch.mul_packed_f32x2((cutlass.Float32(_mul_f32x2_9[0]), cutlass.Float32(_mul_f32x2_9[1])), (cutlass.Float32(_f2_19[0]), cutlass.Float32(_f2_19[1])), rnd='rn', ftz=False)
                                    _f2_23 = (cutlass.Float32(weight0_1[0]), cutlass.Float32(weight1_1[0]))
                                    _mul_f32x2_11 = cute.arch.mul_packed_f32x2((cutlass.Float32(_mul_f32x2_10[0]), cutlass.Float32(_mul_f32x2_10[1])), (cutlass.Float32(_f2_23[0]), cutlass.Float32(_f2_23[1])), rnd='rn', ftz=False)
                                    values_2[(atom_2 * 4)] = cutlass.Float32(_mul_f32x2_8[0])
                                    values_2[((atom_2 * 4) + 1)] = cutlass.Float32(_mul_f32x2_8[1])
                                    values_2[((atom_2 * 4) + 2)] = cutlass.Float32(_mul_f32x2_11[0])
                                    values_2[((atom_2 * 4) + 3)] = cutlass.Float32(_mul_f32x2_11[1])
                                    _fabs_5 = cute.math.abs(values_2[(atom_2 * 4)])
                                    _fabs_6 = cute.math.abs(values_2[((atom_2 * 4) + 2)])
                                    _max_12 = cute.arch.fmax(_fabs_5, _fabs_6, ftz=False)
                                    a0_1[0] = cutlass.Float32(_max_12)
                                    _fabs_7 = cute.math.abs(values_2[((atom_2 * 4) + 1)])
                                    _fabs_8 = cute.math.abs(values_2[((atom_2 * 4) + 3)])
                                    _max_13 = cute.arch.fmax(_fabs_7, _fabs_8, ftz=False)
                                    a1_1[0] = cutlass.Float32(_max_13)
                                    _shfl_xor_7 = cute.arch.shuffle_sync_bfly(a0_1[0], 4, mask=0xFFFFFFFF, mask_and_clamp=31)
                                    _max_14 = cute.arch.fmax(a0_1[0], _shfl_xor_7, ftz=False)
                                    a0_1[0] = cutlass.Float32(_max_14)
                                    _shfl_xor_8 = cute.arch.shuffle_sync_bfly(a1_1[0], 4, mask=0xFFFFFFFF, mask_and_clamp=31)
                                    _max_15 = cute.arch.fmax(a1_1[0], _shfl_xor_8, ftz=False)
                                    a1_1[0] = cutlass.Float32(_max_15)
                                    _shfl_xor_9 = cute.arch.shuffle_sync_bfly(a0_1[0], 8, mask=0xFFFFFFFF, mask_and_clamp=31)
                                    _max_16 = cute.arch.fmax(a0_1[0], _shfl_xor_9, ftz=False)
                                    a0_1[0] = cutlass.Float32(_max_16)
                                    _shfl_xor_10 = cute.arch.shuffle_sync_bfly(a1_1[0], 8, mask=0xFFFFFFFF, mask_and_clamp=31)
                                    _max_17 = cute.arch.fmax(a1_1[0], _shfl_xor_10, ftz=False)
                                    a1_1[0] = cutlass.Float32(_max_17)
                                    _shfl_xor_11 = cute.arch.shuffle_sync_bfly(a0_1[0], 16, mask=0xFFFFFFFF, mask_and_clamp=31)
                                    _max_18 = cute.arch.fmax(a0_1[0], _shfl_xor_11, ftz=False)
                                    a0_1[0] = cutlass.Float32(_max_18)
                                    _shfl_xor_12 = cute.arch.shuffle_sync_bfly(a1_1[0], 16, mask=0xFFFFFFFF, mask_and_clamp=31)
                                    _max_19 = cute.arch.fmax(a1_1[0], _shfl_xor_12, ftz=False)
                                    a1_1[0] = cutlass.Float32(_max_19)
                                    maxima_1[(atom_2 * 2)] = cutlass.Float32(a0_1[0])
                                    maxima_1[((atom_2 * 2) + 1)] = cutlass.Float32(a1_1[0])
                                    if (lane < 4):
                                        prims.store_ext(cutlass.Uint32(cutlass.Float32(a0_1[0]).bitcast(cutlass.Uint32)).ir_value(), cute.make_ptr(cutlass.Uint32, cutlass.Uint32((((scratch_addr + cutlass.Uint32((((wg_1 * 4) + warp_0_1) * 64))) + cutlass.Uint32((atom_2 * 32))) + cutlass.Uint32((lane * 8)))), mem_space=cute.AddressSpace.smem, assumed_align=4))
                                        prims.store_ext(cutlass.Uint32(cutlass.Float32(a1_1[0]).bitcast(cutlass.Uint32)).ir_value(), cute.make_ptr(cutlass.Uint32, cutlass.Uint32(((((scratch_addr + cutlass.Uint32((((wg_1 * 4) + warp_0_1) * 64))) + cutlass.Uint32((atom_2 * 32))) + cutlass.Uint32((lane * 8))) + 4)), mem_space=cute.AddressSpace.smem, assumed_align=4))
                                    cute.arch.sync_warp()
                                if (warp_0_1 == 0):
                                    if prims.elect_sync():
                                        cute.arch.cp_async_bulk_wait_group(1, read=False)
                                prims.barrier_cta_sync((10 + wg_1), thread_count=128)
                                for atom_3 in cutlass.range_constexpr(0, 2, 1):
                                    token_5 = cutlass.Int32((((wg_1 * 16) + (atom_3 * 8)) + ((lane % 4) * 2)))
                                    _smem_physical_23 = cute.make_tensor(cute.make_ptr(cutlass.Float32, cutlass.Uint32(scratch_addr), mem_space=cute.AddressSpace.smem, assumed_align=4), _flat_layout)
                                    _scratch_reg_1 = cute.make_rmem_tensor((2,), cutlass.Float32)
                                    _scratch_reg_1[0] = _smem_physical_23[((((((wg_1 * 4) + (warp_0_1 ^ 1)) * 16) + (atom_3 * 8)) + ((lane % 4) * 2))) + 0]
                                    _scratch_reg_1[1] = _smem_physical_23[((((((wg_1 * 4) + (warp_0_1 ^ 1)) * 16) + (atom_3 * 8)) + ((lane % 4) * 2))) + 1]
                                    quantized_1 = cute.make_rmem_tensor((4,), cutlass.Float32)
                                    for t_1 in cutlass.range_constexpr(0, 2, 1):
                                        _max_20 = cute.arch.fmax(maxima_1[((atom_3 * 2) + t_1)], _scratch_reg_1[t_1], ftz=False)
                                        amax_2 = cutlass.Float32(_max_20)
                                        scaled_2 = cutlass.Float32((amax_2 * 0.002232142857142857))
                                        bits_2 = cutlass.Uint32(cutlass.Float32(scaled_2).bitcast(cutlass.Uint32))
                                        exponent_2 = cutlass.Uint32(((bits_2 >> 23) + cutlass.Uint32((cutlass.Int32(1) if ((bits_2 & 8388607) != 0) else cutlass.Int32(0)))))
                                        inv_bits_1 = cutlass.Uint32(((254 - exponent_2) << 23))
                                        inverse_2 = cutlass.Float32(cutlass.Uint32(inv_bits_1).bitcast(cutlass.Float32))
                                        if (((warp_0_1 % 2) == 0) & (lane < 4)):
                                            sf_byte_1 = cutlass.Uint8(exponent_2)
                                            prims.store_ext(cutlass.Uint8(sf_byte_1).ir_value(), (SF + ((((((((tile * pool_blocks) * 512) + (pool_block_1 * 512)) + cutlass.Uint32((((token_5 + t_1) & 31) * 16))) + cutlass.Uint32((((token_5 + t_1) >> 5) * 4))) + cutlass.Uint32((cta_rank * 2))) + cutlass.Uint32(cutlass.Int32(cutlass_arith.divsi(cutlass.Int32(warp_0_1).ir_value(), cutlass.Int32(2).ir_value())))))))
                                        for k_1 in cutlass.range_constexpr(0, 2, 1):
                                            channel_1 = cutlass.Int32(((((cta_rank * 64) + (warp_0_1 * 16)) + cutlass.Int32(cutlass_arith.divsi(cutlass.Int32(lane).ir_value(), cutlass.Int32(4).ir_value()))) + (k_1 * 8)))
                                            value_2 = cutlass.Float32(values_2[(((atom_3 * 4) + (k_1 * 2)) + t_1)])
                                            quantized_1[((k_1 * 2) + t_1)] = cutlass.Float32((value_2 * inverse_2))
                                    _e4m3x2_f32_4 = cutlass.Uint16(cutlass_llvm.inline_asm(
                                        cutlass.Uint16.mlir_type,
                                        [cutlass.Float32(quantized_1[1]).ir_value(), cutlass.Float32(quantized_1[0]).ir_value()],
                                        'cvt.rn.satfinite.e4m3x2.f32 $0, $1, $2;', '=h,f,f',
                                        has_side_effects=False, is_align_stack=False,
                                        asm_dialect=cutlass_llvm.AsmDialect.AD_ATT,
                                    ))
                                    _e4m3x2_f32_5 = cutlass.Uint16(cutlass_llvm.inline_asm(
                                        cutlass.Uint16.mlir_type,
                                        [cutlass.Float32(quantized_1[3]).ir_value(), cutlass.Float32(quantized_1[2]).ir_value()],
                                        'cvt.rn.satfinite.e4m3x2.f32 $0, $1, $2;', '=h,f,f',
                                        has_side_effects=False, is_align_stack=False,
                                        asm_dialect=cutlass_llvm.AsmDialect.AD_ATT,
                                    ))
                                    packed_2 = cutlass.Uint32((cutlass.Uint32(_e4m3x2_f32_4) | (cutlass.Uint32(_e4m3x2_f32_5) << 16)))
                                    addr_1 = cutlass.Uint32(((((staging_addr + cutlass.Uint32((wg_1 * 2048))) + cutlass.Uint32((atom_3 * 512))) + cutlass.Uint32((lane * 64))) + cutlass.Uint32(((warp_0_1 ^ cutlass.Int32(cutlass_arith.divsi(cutlass.Int32(lane).ir_value(), cutlass.Int32(2).ir_value()))) * 16))))
                                    cutlass_llvm.inline_asm(
                                        res=None,
                                        operands_=[(cutlass.Uint32(addr_1)).ir_value(), (cutlass.Uint32(packed_2)).ir_value()],
                                        asm_string='stmatrix.sync.aligned.m16n8.x1.trans.shared.b8 [$0], {$1};',
                                        constraints='r,r,~{memory}',
                                        has_side_effects=True,
                                        is_align_stack=False,
                                        asm_dialect=cutlass_llvm.AsmDialect.AD_ATT,
                                    )
                                    cute.arch.sync_warp()
                                cute.arch.fence_proxy("async.shared", space="cta")
                                prims.barrier_cta_sync((10 + wg_1), thread_count=128)
                                if (warp_0_1 == 0):
                                    if prims.elect_sync():
                                        prims.cp_async_bulk_tensor_global_shared_cta(
                                            Q.get_ptr(),
                                            cute.make_ptr(cutlass.Uint8, cutlass.Uint32((staging_addr + cutlass.Uint32((wg_1 * 2048)))), mem_space=cute.AddressSpace.smem, assumed_align=16),
                                            [cutlass.Int32(((tile * 128) + cutlass.Uint32((cta_rank * 64)))), cutlass.Int32(((pool_block_1 * 64) + cutlass.Uint32((wg_1 * 16))))],
                                            mode=prims.TMAStoreMode.TILE,
                                        )
                                        cute.arch.cp_async_bulk_commit_group()
                            if (warp_0_1 == 0):
                                if prims.elect_sync():
                                    cute.arch.cp_async_bulk_wait_group(0, read=False)
                            prims.barrier_cta_sync(15, thread_count=256)
                        else:
                            warp_0_2 = cutlass.Int32((warp % 4))
                            wg_2 = cutlass.Int32(cutlass.Int32(cutlass_arith.divsi(cutlass.Int32((warp - 8)).ir_value(), cutlass.Int32(4).ir_value())))
                            if (task_valid <= cutlass.Uint32((wg_2 * 32))):
                                prims.tcgen05_fence('before_thread_sync')
                                _mbarrier_cluster_arrive_24 = prims.mapa(released_addr + e_stage[0], cutlass.Int32(cta_rank & 0xFFFFFFFE))
                                prims.mbarrier_arrive(_mbarrier_cluster_arrive_24, count=cutlass.Int32(1), scope=prims.MemScope.CTA)
                            else:
                                cached_2[0] = cutlass.Float32(0.0)
                                if (lane < 32):
                                    cached_2[0] = cutlass.Float32(_RW[(((pool_block_1 * 64) + cutlass.Uint32((wg_2 * 32))) + cutlass.Uint32(lane))])
                                values_3 = cute.make_rmem_tensor((16,), cutlass.Float32)
                                maxima_2 = cute.make_rmem_tensor((8,), cutlass.Float32)
                                for atom_4 in cutlass.range_constexpr(0, 4, 1):
                                    token_6 = cutlass.Int32((((wg_2 * 32) + (atom_4 * 8)) + ((lane % 4) * 2)))
                                    weight0_2[0] = cutlass.Float32(0.0)
                                    weight1_2[0] = cutlass.Float32(0.0)
                                    _shfl_12 = cute.arch.shuffle_sync(cached_2[0], ((atom_4 * 8) + ((lane % 4) * 2)), mask=4294967295, mask_and_clamp=31)
                                    weight0_2[0] = cutlass.Float32(_shfl_12)
                                    _shfl_13 = cute.arch.shuffle_sync(cached_2[0], (((atom_4 * 8) + ((lane % 4) * 2)) + 1), mask=4294967295, mask_and_clamp=31)
                                    weight1_2[0] = cutlass.Float32(_shfl_13)
                                    address_2 = cutlass.Int32((((taddr + (e_stage[0] * 64)) + cutlass.Uint32((wg_2 * 32))) + cutlass.Uint32((atom_4 * 8))))
                                    _tmem_load_4 = cute.make_rmem_tensor((4,), cutlass.Float32)
                                    _tmem_load_25_atom = cute.make_copy_atom(cute.nvgpu.tcgen05.Ld16x256bOp(cute.nvgpu.tcgen05.Repetition.x1), cutlass.Float32)
                                    _tmem_load_25_src = cute.make_tensor(cute.make_ptr(cutlass.Float32, cutlass.Uint32(address_2), mem_space=cutlass.AddressSpace.tmem, assumed_align=16), cute.make_layout((4,), stride=(1,)))
                                    _tmem_load_25_dst = cute.make_tensor(_tmem_load_4.iterator + 0, cute.make_layout((4,), stride=(1,)))
                                    cute.copy(_tmem_load_25_atom, _tmem_load_25_src, _tmem_load_25_dst)
                                    cute.arch.fence_view_async_tmem_load()
                                    _tmem_load_5 = cute.make_rmem_tensor((4,), cutlass.Float32)
                                    _tmem_load_26_atom = cute.make_copy_atom(cute.nvgpu.tcgen05.Ld16x256bOp(cute.nvgpu.tcgen05.Repetition.x1), cutlass.Float32)
                                    _tmem_load_26_src = cute.make_tensor(cute.make_ptr(cutlass.Float32, cutlass.Uint32((address_2 + 1048576)), mem_space=cutlass.AddressSpace.tmem, assumed_align=16), cute.make_layout((4,), stride=(1,)))
                                    _tmem_load_26_dst = cute.make_tensor(_tmem_load_5.iterator + 0, cute.make_layout((4,), stride=(1,)))
                                    cute.copy(_tmem_load_26_atom, _tmem_load_26_src, _tmem_load_26_dst)
                                    cute.arch.fence_view_async_tmem_load()
                                    if (atom_4 == 3):
                                        prims.tcgen05_fence('before_thread_sync')
                                        _mbarrier_cluster_arrive_27 = prims.mapa(released_addr + e_stage[0], cutlass.Int32(cta_rank & 0xFFFFFFFE))
                                        prims.mbarrier_arrive(_mbarrier_cluster_arrive_27, count=cutlass.Int32(1), scope=prims.MemScope.CTA)
                                    if (not True):
                                        weight0_2[0] = cutlass.Float32(_RW[((pool_block_1 * 64) + cutlass.Uint32(token_6))])
                                        weight1_2[0] = cutlass.Float32(_RW[(((pool_block_1 * 64) + cutlass.Uint32(token_6)) + 1)])
                                    _cvt_bf16_16 = cutlass.BFloat16(_tmem_load_4[0])
                                    _cvt_f32_17 = cutlass.Float32(_cvt_bf16_16)
                                    _cvt_bf16_17 = cutlass.BFloat16(_tmem_load_4[1])
                                    _cvt_f32_18 = cutlass.Float32(_cvt_bf16_17)
                                    _f2_24 = (cutlass.Float32(_cvt_f32_17), cutlass.Float32(_cvt_f32_18))
                                    _cvt_bf16_18 = cutlass.BFloat16(_tmem_load_4[2])
                                    _cvt_f32_19 = cutlass.Float32(_cvt_bf16_18)
                                    _cvt_bf16_19 = cutlass.BFloat16(_tmem_load_4[3])
                                    _cvt_f32_20 = cutlass.Float32(_cvt_bf16_19)
                                    _f2_25 = (cutlass.Float32(_cvt_f32_19), cutlass.Float32(_cvt_f32_20))
                                    _expf_8 = cute.math.exp2(cutlass.Float32((0 - _f2_24[0])) * cutlass.Float32(1.4426950408889634), approx=True, ftz=True)
                                    _expf_9 = cute.math.exp2(cutlass.Float32((0 - _f2_24[1])) * cutlass.Float32(1.4426950408889634), approx=True, ftz=True)
                                    _f2_26 = (cutlass.Float32(_expf_8), cutlass.Float32(_expf_9))
                                    _f2_27 = (cutlass.Float32(1.0), cutlass.Float32(1.0))
                                    _packed_add_f32x2_4 = cute.arch.add_packed_f32x2((cutlass.Float32(_f2_27[0]), cutlass.Float32(_f2_27[1])), (cutlass.Float32(_f2_26[0]), cutlass.Float32(_f2_26[1])), ftz=False)
                                    _rcp_8 = cute.math.rcp(_packed_add_f32x2_4[0], approx=True, ftz=True)
                                    _rcp_9 = cute.math.rcp(_packed_add_f32x2_4[1], approx=True, ftz=True)
                                    _f2_28 = (cutlass.Float32(_rcp_8), cutlass.Float32(_rcp_9))
                                    _mul_f32x2_12 = cute.arch.mul_packed_f32x2((cutlass.Float32(_f2_24[0]), cutlass.Float32(_f2_24[1])), (cutlass.Float32(_f2_28[0]), cutlass.Float32(_f2_28[1])), rnd='rn', ftz=False)
                                    _mul_f32x2_13 = cute.arch.mul_packed_f32x2((cutlass.Float32(_mul_f32x2_12[0]), cutlass.Float32(_mul_f32x2_12[1])), (cutlass.Float32(_f2_25[0]), cutlass.Float32(_f2_25[1])), rnd='rn', ftz=False)
                                    _f2_29 = (cutlass.Float32(weight0_2[0]), cutlass.Float32(weight1_2[0]))
                                    _mul_f32x2_14 = cute.arch.mul_packed_f32x2((cutlass.Float32(_mul_f32x2_13[0]), cutlass.Float32(_mul_f32x2_13[1])), (cutlass.Float32(_f2_29[0]), cutlass.Float32(_f2_29[1])), rnd='rn', ftz=False)
                                    _cvt_bf16_20 = cutlass.BFloat16(_tmem_load_5[0])
                                    _cvt_f32_21 = cutlass.Float32(_cvt_bf16_20)
                                    _cvt_bf16_21 = cutlass.BFloat16(_tmem_load_5[1])
                                    _cvt_f32_22 = cutlass.Float32(_cvt_bf16_21)
                                    _f2_30 = (cutlass.Float32(_cvt_f32_21), cutlass.Float32(_cvt_f32_22))
                                    _cvt_bf16_22 = cutlass.BFloat16(_tmem_load_5[2])
                                    _cvt_f32_23 = cutlass.Float32(_cvt_bf16_22)
                                    _cvt_bf16_23 = cutlass.BFloat16(_tmem_load_5[3])
                                    _cvt_f32_24 = cutlass.Float32(_cvt_bf16_23)
                                    _f2_31 = (cutlass.Float32(_cvt_f32_23), cutlass.Float32(_cvt_f32_24))
                                    _expf_10 = cute.math.exp2(cutlass.Float32((0 - _f2_30[0])) * cutlass.Float32(1.4426950408889634), approx=True, ftz=True)
                                    _expf_11 = cute.math.exp2(cutlass.Float32((0 - _f2_30[1])) * cutlass.Float32(1.4426950408889634), approx=True, ftz=True)
                                    _f2_32 = (cutlass.Float32(_expf_10), cutlass.Float32(_expf_11))
                                    _f2_33 = (cutlass.Float32(1.0), cutlass.Float32(1.0))
                                    _packed_add_f32x2_5 = cute.arch.add_packed_f32x2((cutlass.Float32(_f2_33[0]), cutlass.Float32(_f2_33[1])), (cutlass.Float32(_f2_32[0]), cutlass.Float32(_f2_32[1])), ftz=False)
                                    _rcp_10 = cute.math.rcp(_packed_add_f32x2_5[0], approx=True, ftz=True)
                                    _rcp_11 = cute.math.rcp(_packed_add_f32x2_5[1], approx=True, ftz=True)
                                    _f2_34 = (cutlass.Float32(_rcp_10), cutlass.Float32(_rcp_11))
                                    _mul_f32x2_15 = cute.arch.mul_packed_f32x2((cutlass.Float32(_f2_30[0]), cutlass.Float32(_f2_30[1])), (cutlass.Float32(_f2_34[0]), cutlass.Float32(_f2_34[1])), rnd='rn', ftz=False)
                                    _mul_f32x2_16 = cute.arch.mul_packed_f32x2((cutlass.Float32(_mul_f32x2_15[0]), cutlass.Float32(_mul_f32x2_15[1])), (cutlass.Float32(_f2_31[0]), cutlass.Float32(_f2_31[1])), rnd='rn', ftz=False)
                                    _f2_35 = (cutlass.Float32(weight0_2[0]), cutlass.Float32(weight1_2[0]))
                                    _mul_f32x2_17 = cute.arch.mul_packed_f32x2((cutlass.Float32(_mul_f32x2_16[0]), cutlass.Float32(_mul_f32x2_16[1])), (cutlass.Float32(_f2_35[0]), cutlass.Float32(_f2_35[1])), rnd='rn', ftz=False)
                                    values_3[(atom_4 * 4)] = cutlass.Float32(_mul_f32x2_14[0])
                                    values_3[((atom_4 * 4) + 1)] = cutlass.Float32(_mul_f32x2_14[1])
                                    values_3[((atom_4 * 4) + 2)] = cutlass.Float32(_mul_f32x2_17[0])
                                    values_3[((atom_4 * 4) + 3)] = cutlass.Float32(_mul_f32x2_17[1])
                                    _fabs_9 = cute.math.abs(values_3[(atom_4 * 4)])
                                    _fabs_10 = cute.math.abs(values_3[((atom_4 * 4) + 2)])
                                    _max_21 = cute.arch.fmax(_fabs_9, _fabs_10, ftz=False)
                                    a0_2[0] = cutlass.Float32(_max_21)
                                    _fabs_11 = cute.math.abs(values_3[((atom_4 * 4) + 1)])
                                    _fabs_12 = cute.math.abs(values_3[((atom_4 * 4) + 3)])
                                    _max_22 = cute.arch.fmax(_fabs_11, _fabs_12, ftz=False)
                                    a1_2[0] = cutlass.Float32(_max_22)
                                    _shfl_xor_13 = cute.arch.shuffle_sync_bfly(a0_2[0], 4, mask=0xFFFFFFFF, mask_and_clamp=31)
                                    _max_23 = cute.arch.fmax(a0_2[0], _shfl_xor_13, ftz=False)
                                    a0_2[0] = cutlass.Float32(_max_23)
                                    _shfl_xor_14 = cute.arch.shuffle_sync_bfly(a1_2[0], 4, mask=0xFFFFFFFF, mask_and_clamp=31)
                                    _max_24 = cute.arch.fmax(a1_2[0], _shfl_xor_14, ftz=False)
                                    a1_2[0] = cutlass.Float32(_max_24)
                                    _shfl_xor_15 = cute.arch.shuffle_sync_bfly(a0_2[0], 8, mask=0xFFFFFFFF, mask_and_clamp=31)
                                    _max_25 = cute.arch.fmax(a0_2[0], _shfl_xor_15, ftz=False)
                                    a0_2[0] = cutlass.Float32(_max_25)
                                    _shfl_xor_16 = cute.arch.shuffle_sync_bfly(a1_2[0], 8, mask=0xFFFFFFFF, mask_and_clamp=31)
                                    _max_26 = cute.arch.fmax(a1_2[0], _shfl_xor_16, ftz=False)
                                    a1_2[0] = cutlass.Float32(_max_26)
                                    _shfl_xor_17 = cute.arch.shuffle_sync_bfly(a0_2[0], 16, mask=0xFFFFFFFF, mask_and_clamp=31)
                                    _max_27 = cute.arch.fmax(a0_2[0], _shfl_xor_17, ftz=False)
                                    a0_2[0] = cutlass.Float32(_max_27)
                                    _shfl_xor_18 = cute.arch.shuffle_sync_bfly(a1_2[0], 16, mask=0xFFFFFFFF, mask_and_clamp=31)
                                    _max_28 = cute.arch.fmax(a1_2[0], _shfl_xor_18, ftz=False)
                                    a1_2[0] = cutlass.Float32(_max_28)
                                    maxima_2[(atom_4 * 2)] = cutlass.Float32(a0_2[0])
                                    maxima_2[((atom_4 * 2) + 1)] = cutlass.Float32(a1_2[0])
                                    if (lane < 4):
                                        prims.store_ext(cutlass.Uint32(cutlass.Float32(a0_2[0]).bitcast(cutlass.Uint32)).ir_value(), cute.make_ptr(cutlass.Uint32, cutlass.Uint32((((scratch_addr + cutlass.Uint32((((wg_2 * 4) + warp_0_2) * 128))) + cutlass.Uint32((atom_4 * 32))) + cutlass.Uint32((lane * 8)))), mem_space=cute.AddressSpace.smem, assumed_align=4))
                                        prims.store_ext(cutlass.Uint32(cutlass.Float32(a1_2[0]).bitcast(cutlass.Uint32)).ir_value(), cute.make_ptr(cutlass.Uint32, cutlass.Uint32(((((scratch_addr + cutlass.Uint32((((wg_2 * 4) + warp_0_2) * 128))) + cutlass.Uint32((atom_4 * 32))) + cutlass.Uint32((lane * 8))) + 4)), mem_space=cute.AddressSpace.smem, assumed_align=4))
                                    cute.arch.sync_warp()
                                if (warp_0_2 == 0):
                                    if prims.elect_sync():
                                        cute.arch.cp_async_bulk_wait_group(1, read=False)
                                prims.barrier_cta_sync((10 + wg_2), thread_count=128)
                                for atom_5 in cutlass.range_constexpr(0, 4, 1):
                                    token_7 = cutlass.Int32((((wg_2 * 32) + (atom_5 * 8)) + ((lane % 4) * 2)))
                                    _smem_physical_28 = cute.make_tensor(cute.make_ptr(cutlass.Float32, cutlass.Uint32(scratch_addr), mem_space=cute.AddressSpace.smem, assumed_align=4), _flat_layout)
                                    _scratch_reg_2 = cute.make_rmem_tensor((2,), cutlass.Float32)
                                    _scratch_reg_2[0] = _smem_physical_28[((((((wg_2 * 4) + (warp_0_2 ^ 1)) * 32) + (atom_5 * 8)) + ((lane % 4) * 2))) + 0]
                                    _scratch_reg_2[1] = _smem_physical_28[((((((wg_2 * 4) + (warp_0_2 ^ 1)) * 32) + (atom_5 * 8)) + ((lane % 4) * 2))) + 1]
                                    quantized_2 = cute.make_rmem_tensor((4,), cutlass.Float32)
                                    for t_2 in cutlass.range_constexpr(0, 2, 1):
                                        _max_29 = cute.arch.fmax(maxima_2[((atom_5 * 2) + t_2)], _scratch_reg_2[t_2], ftz=False)
                                        amax_3 = cutlass.Float32(_max_29)
                                        scaled_3 = cutlass.Float32((amax_3 * 0.002232142857142857))
                                        bits_3 = cutlass.Uint32(cutlass.Float32(scaled_3).bitcast(cutlass.Uint32))
                                        exponent_3 = cutlass.Uint32(((bits_3 >> 23) + cutlass.Uint32((cutlass.Int32(1) if ((bits_3 & 8388607) != 0) else cutlass.Int32(0)))))
                                        inv_bits_2 = cutlass.Uint32(((254 - exponent_3) << 23))
                                        inverse_3 = cutlass.Float32(cutlass.Uint32(inv_bits_2).bitcast(cutlass.Float32))
                                        if (((warp_0_2 % 2) == 0) & (lane < 4)):
                                            sf_byte_2 = cutlass.Uint8(exponent_3)
                                            prims.store_ext(cutlass.Uint8(sf_byte_2).ir_value(), (SF + ((((((((tile * pool_blocks) * 512) + (pool_block_1 * 512)) + cutlass.Uint32((((token_7 + t_2) & 31) * 16))) + cutlass.Uint32((((token_7 + t_2) >> 5) * 4))) + cutlass.Uint32((cta_rank * 2))) + cutlass.Uint32(cutlass.Int32(cutlass_arith.divsi(cutlass.Int32(warp_0_2).ir_value(), cutlass.Int32(2).ir_value())))))))
                                        for k_2 in cutlass.range_constexpr(0, 2, 1):
                                            channel_2 = cutlass.Int32(((((cta_rank * 64) + (warp_0_2 * 16)) + cutlass.Int32(cutlass_arith.divsi(cutlass.Int32(lane).ir_value(), cutlass.Int32(4).ir_value()))) + (k_2 * 8)))
                                            value_3 = cutlass.Float32(values_3[(((atom_5 * 4) + (k_2 * 2)) + t_2)])
                                            quantized_2[((k_2 * 2) + t_2)] = cutlass.Float32((value_3 * inverse_3))
                                    _e4m3x2_f32_6 = cutlass.Uint16(cutlass_llvm.inline_asm(
                                        cutlass.Uint16.mlir_type,
                                        [cutlass.Float32(quantized_2[1]).ir_value(), cutlass.Float32(quantized_2[0]).ir_value()],
                                        'cvt.rn.satfinite.e4m3x2.f32 $0, $1, $2;', '=h,f,f',
                                        has_side_effects=False, is_align_stack=False,
                                        asm_dialect=cutlass_llvm.AsmDialect.AD_ATT,
                                    ))
                                    _e4m3x2_f32_7 = cutlass.Uint16(cutlass_llvm.inline_asm(
                                        cutlass.Uint16.mlir_type,
                                        [cutlass.Float32(quantized_2[3]).ir_value(), cutlass.Float32(quantized_2[2]).ir_value()],
                                        'cvt.rn.satfinite.e4m3x2.f32 $0, $1, $2;', '=h,f,f',
                                        has_side_effects=False, is_align_stack=False,
                                        asm_dialect=cutlass_llvm.AsmDialect.AD_ATT,
                                    ))
                                    packed_3 = cutlass.Uint32((cutlass.Uint32(_e4m3x2_f32_6) | (cutlass.Uint32(_e4m3x2_f32_7) << 16)))
                                    addr_2 = cutlass.Uint32(((((staging_addr + cutlass.Uint32((wg_2 * 4096))) + cutlass.Uint32((atom_5 * 512))) + cutlass.Uint32((lane * 64))) + cutlass.Uint32(((warp_0_2 ^ cutlass.Int32(cutlass_arith.divsi(cutlass.Int32(lane).ir_value(), cutlass.Int32(2).ir_value()))) * 16))))
                                    cutlass_llvm.inline_asm(
                                        res=None,
                                        operands_=[(cutlass.Uint32(addr_2)).ir_value(), (cutlass.Uint32(packed_3)).ir_value()],
                                        asm_string='stmatrix.sync.aligned.m16n8.x1.trans.shared.b8 [$0], {$1};',
                                        constraints='r,r,~{memory}',
                                        has_side_effects=True,
                                        is_align_stack=False,
                                        asm_dialect=cutlass_llvm.AsmDialect.AD_ATT,
                                    )
                                    cute.arch.sync_warp()
                                cute.arch.fence_proxy("async.shared", space="cta")
                                prims.barrier_cta_sync((10 + wg_2), thread_count=128)
                                if (warp_0_2 == 0):
                                    if prims.elect_sync():
                                        prims.cp_async_bulk_tensor_global_shared_cta(
                                            Q32.get_ptr(),
                                            cute.make_ptr(cutlass.Uint8, cutlass.Uint32((staging_addr + cutlass.Uint32((wg_2 * 4096)))), mem_space=cute.AddressSpace.smem, assumed_align=16),
                                            [cutlass.Int32(((tile * 128) + cutlass.Uint32((cta_rank * 64)))), cutlass.Int32(((pool_block_1 * 64) + cutlass.Uint32((wg_2 * 32))))],
                                            mode=prims.TMAStoreMode.TILE,
                                        )
                                        cute.arch.cp_async_bulk_commit_group()
                            if (warp_0_2 == 0):
                                if prims.elect_sync():
                                    cute.arch.cp_async_bulk_wait_group(0, read=False)
                            prims.barrier_cta_sync(15, thread_count=256)
                    if (warp == 8):
                        if prims.elect_sync():
                            cute.arch.atomic_add(l2_full + pool_block_1, cutlass.Uint32(1), sem='release', scope='gpu')
                            _atomic_old_5 = cute.arch.atomic_add(cute.recast_ptr(l1_empty, dtype=cutlass.Uint32) + pool_block_1, cutlass.Uint32(1), sem='relaxed', scope='gpu')
                else:
                    if (warp == 8):
                        if prims.elect_sync():
                            _atomic_old_6 = cute.arch.atomic_add(cute.recast_ptr(l2_empty, dtype=cutlass.Uint32) + pool_block_1, cutlass.Uint32(1), sem='relaxed', scope='gpu')
                    if (task_n == 16):
                        warp_0_3 = cutlass.Int32((warp % 4))
                        wg_3 = cutlass.Int32(cutlass.Int32(cutlass_arith.divsi(cutlass.Int32((warp - 8)).ir_value(), cutlass.Int32(4).ir_value())))
                        if (task_valid <= cutlass.Uint32((wg_3 * 8))):
                            prims.tcgen05_fence('before_thread_sync')
                            _mbarrier_cluster_arrive_29 = prims.mapa(released_addr + e_stage[0], cutlass.Int32(cta_rank & 0xFFFFFFFE))
                            prims.mbarrier_arrive(_mbarrier_cluster_arrive_29, count=cutlass.Int32(1), scope=prims.MemScope.CTA)
                        else:
                            for atom_6 in cutlass.range_constexpr(0, 1, 1):
                                address_3 = cutlass.Int32((((taddr + (e_stage[0] * 64)) + cutlass.Uint32((wg_3 * 8))) + cutlass.Uint32((atom_6 * 8))))
                                _tmem_load_6 = cute.make_rmem_tensor((4,), cutlass.Float32)
                                _tmem_load_30_atom = cute.make_copy_atom(cute.nvgpu.tcgen05.Ld16x256bOp(cute.nvgpu.tcgen05.Repetition.x1), cutlass.Float32)
                                _tmem_load_30_src = cute.make_tensor(cute.make_ptr(cutlass.Float32, cutlass.Uint32(address_3), mem_space=cutlass.AddressSpace.tmem, assumed_align=16), cute.make_layout((4,), stride=(1,)))
                                _tmem_load_30_dst = cute.make_tensor(_tmem_load_6.iterator + 0, cute.make_layout((4,), stride=(1,)))
                                cute.copy(_tmem_load_30_atom, _tmem_load_30_src, _tmem_load_30_dst)
                                cute.arch.fence_view_async_tmem_load()
                                _tmem_load_7 = cute.make_rmem_tensor((4,), cutlass.Float32)
                                _tmem_load_31_atom = cute.make_copy_atom(cute.nvgpu.tcgen05.Ld16x256bOp(cute.nvgpu.tcgen05.Repetition.x1), cutlass.Float32)
                                _tmem_load_31_src = cute.make_tensor(cute.make_ptr(cutlass.Float32, cutlass.Uint32((address_3 + 1048576)), mem_space=cutlass.AddressSpace.tmem, assumed_align=16), cute.make_layout((4,), stride=(1,)))
                                _tmem_load_31_dst = cute.make_tensor(_tmem_load_7.iterator + 0, cute.make_layout((4,), stride=(1,)))
                                cute.copy(_tmem_load_31_atom, _tmem_load_31_src, _tmem_load_31_dst)
                                cute.arch.fence_view_async_tmem_load()
                                if (atom_6 == 0):
                                    prims.tcgen05_fence('before_thread_sync')
                                    _mbarrier_cluster_arrive_32 = prims.mapa(released_addr + e_stage[0], cutlass.Int32(cta_rank & 0xFFFFFFFE))
                                    prims.mbarrier_arrive(_mbarrier_cluster_arrive_32, count=cutlass.Int32(1), scope=prims.MemScope.CTA)
                                token_8 = cutlass.Int32((((wg_3 * 8) + (atom_6 * 8)) + ((lane % 4) * 2)))
                                channel_3 = cutlass.Int32((((cta_rank * 128) + (warp_0_3 * 32)) + cutlass.Int32(cutlass_arith.divsi(cutlass.Int32(lane).ir_value(), cutlass.Int32(4).ir_value()))))
                                _bf16x2_0 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(_tmem_load_6[0]), cutlass.Float32(_tmem_load_6[1])))[1]), cutlass.Float32(((cutlass.Float32(_tmem_load_6[0]), cutlass.Float32(_tmem_load_6[1])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
                                _bf16x2_1 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(_tmem_load_6[2]), cutlass.Float32(_tmem_load_6[3])))[1]), cutlass.Float32(((cutlass.Float32(_tmem_load_6[2]), cutlass.Float32(_tmem_load_6[3])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
                                _bf16x2_2 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(_tmem_load_7[0]), cutlass.Float32(_tmem_load_7[1])))[1]), cutlass.Float32(((cutlass.Float32(_tmem_load_7[0]), cutlass.Float32(_tmem_load_7[1])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
                                _bf16x2_3 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(_tmem_load_7[2]), cutlass.Float32(_tmem_load_7[3])))[1]), cutlass.Float32(((cutlass.Float32(_tmem_load_7[2]), cutlass.Float32(_tmem_load_7[3])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
                                r0 = cutlass.Uint32(cutlass.Uint32(_bf16x2_0))
                                r1 = cutlass.Uint32(cutlass.Uint32(_bf16x2_1))
                                r2 = cutlass.Uint32(cutlass.Uint32(_bf16x2_2))
                                r3 = cutlass.Uint32(cutlass.Uint32(_bf16x2_3))
                                row = cutlass.Int32((lane % 8))
                                col = cutlass.Int32((((warp_0_3 % 2) * 4) + cutlass.Int32(cutlass_arith.divsi(cutlass.Int32(lane).ir_value(), cutlass.Int32(8).ir_value()))))
                                addr_3 = cutlass.Uint32((((((staging_addr + cutlass.Uint32((wg_3 * 2048))) + cutlass.Uint32((cutlass.Int32(cutlass_arith.divsi(cutlass.Int32(warp_0_3).ir_value(), cutlass.Int32(2).ir_value())) * 1024))) + cutlass.Uint32((atom_6 * 1024))) + cutlass.Uint32((row * 128))) + cutlass.Uint32(((col ^ row) * 16))))
                                prims.stmatrix(cute.make_ptr(cutlass.Uint8, cutlass.Uint32(addr_3), mem_space=cute.AddressSpace.smem, assumed_align=16), [cutlass.Uint32(r0), cutlass.Uint32(r1), cutlass.Uint32(r2), cutlass.Uint32(r3)], prims.MMALayout.COL, shape=prims.StoreShape.M8N8)
                            prims.barrier_cta_sync((10 + wg_3), thread_count=128)
                            for atom_7 in cutlass.range_constexpr(0, 1, 1):
                                out_row = cutlass.Int32((((atom_7 * 8) + (warp_0_3 * 2)) + cutlass.Int32(cutlass_arith.divsi(cutlass.Int32(lane).ir_value(), cutlass.Int32(16).ir_value()))))
                                row_in_atom = cutlass.Int32((out_row % 8))
                                offset_2 = cutlass.Int32(cutlass.Int32(cutlass_arith.divsi(cutlass.Int32(((((wg_3 * 2048) + (cutlass.Int32(cutlass_arith.divsi(cutlass.Int32((lane % 16)).ir_value(), cutlass.Int32(8).ir_value())) * 1024)) + (out_row * 128)) + (((lane % 8) ^ row_in_atom) * 16))).ir_value(), cutlass.Int32(4).ir_value())))
                                if (task_valid > cutlass.Uint32(((wg_3 * 8) + out_row))):
                                    _smem_physical_33 = cute.make_tensor(cute.make_ptr(cutlass.Uint32, cutlass.Uint32(staging_addr), mem_space=cute.AddressSpace.smem, assumed_align=4), _flat_layout)
                                    _staging_reg_0 = cute.make_rmem_tensor((4,), cutlass.Uint32)
                                    _staging_reg_0[0] = _smem_physical_33[(offset_2) + 0]
                                    _staging_reg_0[1] = _smem_physical_33[(offset_2) + 1]
                                    _staging_reg_0[2] = _smem_physical_33[(offset_2) + 2]
                                    _staging_reg_0[3] = _smem_physical_33[(offset_2) + 3]
                                    owner = cutlass.Uint32(_metadata[((((logical_block * 64) + cutlass.Uint32((wg_3 * 8))) + cutlass.Uint32(out_row)) * 3)])
                                    token_9 = cutlass.Uint32(_metadata[(((((logical_block * 64) + cutlass.Uint32((wg_3 * 8))) + cutlass.Uint32(out_row)) * 3) + 1)])
                                    slot_2 = cutlass.Uint32(_metadata[(((((logical_block * 64) + cutlass.Uint32((wg_3 * 8))) + cutlass.Uint32(out_row)) * 3) + 2)])
                                    element = cutlass.Uint64(((((cutlass.Uint64(slot_2) * 384) + cutlass.Uint64(token_9)) * cutlass.Uint64((fc2_tiles * 256))) + cutlass.Uint64((((tile * 256) + cutlass.Uint32((cta_rank * 128))) + cutlass.Uint32(((lane % 16) * 8))))))
                                    _gmem_store_raw_34 = cutlass.Vector.from_elements([cutlass.Uint32(_staging_reg_0[0]), cutlass.Uint32(_staging_reg_0[1]), cutlass.Uint32(_staging_reg_0[2]), cutlass.Uint32(_staging_reg_0[3])], cutlass.Uint32)
                                    prims.store_ext(_gmem_store_raw_34.ir_value(), cute.make_ptr(cutlass.Uint32, cutlass.Uint64((_output_peers[owner] + (element * 2))), mem_space=cute.AddressSpace.gmem, assumed_align=4))
                        prims.barrier_cta_sync(15, thread_count=256)
                    else:
                        if (task_n == 32):
                            warp_0_4 = cutlass.Int32((warp % 4))
                            wg_4 = cutlass.Int32(cutlass.Int32(cutlass_arith.divsi(cutlass.Int32((warp - 8)).ir_value(), cutlass.Int32(4).ir_value())))
                            if (task_valid <= cutlass.Uint32((wg_4 * 16))):
                                prims.tcgen05_fence('before_thread_sync')
                                _mbarrier_cluster_arrive_35 = prims.mapa(released_addr + e_stage[0], cutlass.Int32(cta_rank & 0xFFFFFFFE))
                                prims.mbarrier_arrive(_mbarrier_cluster_arrive_35, count=cutlass.Int32(1), scope=prims.MemScope.CTA)
                            else:
                                for atom_8 in cutlass.range_constexpr(0, 2, 1):
                                    address_4 = cutlass.Int32((((taddr + (e_stage[0] * 64)) + cutlass.Uint32((wg_4 * 16))) + cutlass.Uint32((atom_8 * 8))))
                                    _tmem_load_8 = cute.make_rmem_tensor((4,), cutlass.Float32)
                                    _tmem_load_36_atom = cute.make_copy_atom(cute.nvgpu.tcgen05.Ld16x256bOp(cute.nvgpu.tcgen05.Repetition.x1), cutlass.Float32)
                                    _tmem_load_36_src = cute.make_tensor(cute.make_ptr(cutlass.Float32, cutlass.Uint32(address_4), mem_space=cutlass.AddressSpace.tmem, assumed_align=16), cute.make_layout((4,), stride=(1,)))
                                    _tmem_load_36_dst = cute.make_tensor(_tmem_load_8.iterator + 0, cute.make_layout((4,), stride=(1,)))
                                    cute.copy(_tmem_load_36_atom, _tmem_load_36_src, _tmem_load_36_dst)
                                    cute.arch.fence_view_async_tmem_load()
                                    _tmem_load_9 = cute.make_rmem_tensor((4,), cutlass.Float32)
                                    _tmem_load_37_atom = cute.make_copy_atom(cute.nvgpu.tcgen05.Ld16x256bOp(cute.nvgpu.tcgen05.Repetition.x1), cutlass.Float32)
                                    _tmem_load_37_src = cute.make_tensor(cute.make_ptr(cutlass.Float32, cutlass.Uint32((address_4 + 1048576)), mem_space=cutlass.AddressSpace.tmem, assumed_align=16), cute.make_layout((4,), stride=(1,)))
                                    _tmem_load_37_dst = cute.make_tensor(_tmem_load_9.iterator + 0, cute.make_layout((4,), stride=(1,)))
                                    cute.copy(_tmem_load_37_atom, _tmem_load_37_src, _tmem_load_37_dst)
                                    cute.arch.fence_view_async_tmem_load()
                                    if (atom_8 == 1):
                                        prims.tcgen05_fence('before_thread_sync')
                                        _mbarrier_cluster_arrive_38 = prims.mapa(released_addr + e_stage[0], cutlass.Int32(cta_rank & 0xFFFFFFFE))
                                        prims.mbarrier_arrive(_mbarrier_cluster_arrive_38, count=cutlass.Int32(1), scope=prims.MemScope.CTA)
                                    token_10 = cutlass.Int32((((wg_4 * 16) + (atom_8 * 8)) + ((lane % 4) * 2)))
                                    channel_4 = cutlass.Int32((((cta_rank * 128) + (warp_0_4 * 32)) + cutlass.Int32(cutlass_arith.divsi(cutlass.Int32(lane).ir_value(), cutlass.Int32(4).ir_value()))))
                                    _bf16x2_4 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(_tmem_load_8[0]), cutlass.Float32(_tmem_load_8[1])))[1]), cutlass.Float32(((cutlass.Float32(_tmem_load_8[0]), cutlass.Float32(_tmem_load_8[1])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
                                    _bf16x2_5 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(_tmem_load_8[2]), cutlass.Float32(_tmem_load_8[3])))[1]), cutlass.Float32(((cutlass.Float32(_tmem_load_8[2]), cutlass.Float32(_tmem_load_8[3])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
                                    _bf16x2_6 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(_tmem_load_9[0]), cutlass.Float32(_tmem_load_9[1])))[1]), cutlass.Float32(((cutlass.Float32(_tmem_load_9[0]), cutlass.Float32(_tmem_load_9[1])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
                                    _bf16x2_7 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(_tmem_load_9[2]), cutlass.Float32(_tmem_load_9[3])))[1]), cutlass.Float32(((cutlass.Float32(_tmem_load_9[2]), cutlass.Float32(_tmem_load_9[3])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
                                    r0_1 = cutlass.Uint32(cutlass.Uint32(_bf16x2_4))
                                    r1_1 = cutlass.Uint32(cutlass.Uint32(_bf16x2_5))
                                    r2_1 = cutlass.Uint32(cutlass.Uint32(_bf16x2_6))
                                    r3_1 = cutlass.Uint32(cutlass.Uint32(_bf16x2_7))
                                    row_1 = cutlass.Int32((lane % 8))
                                    col_1 = cutlass.Int32((((warp_0_4 % 2) * 4) + cutlass.Int32(cutlass_arith.divsi(cutlass.Int32(lane).ir_value(), cutlass.Int32(8).ir_value()))))
                                    addr_4 = cutlass.Uint32((((((staging_addr + cutlass.Uint32((wg_4 * 4096))) + cutlass.Uint32((cutlass.Int32(cutlass_arith.divsi(cutlass.Int32(warp_0_4).ir_value(), cutlass.Int32(2).ir_value())) * 2048))) + cutlass.Uint32((atom_8 * 1024))) + cutlass.Uint32((row_1 * 128))) + cutlass.Uint32(((col_1 ^ row_1) * 16))))
                                    prims.stmatrix(cute.make_ptr(cutlass.Uint8, cutlass.Uint32(addr_4), mem_space=cute.AddressSpace.smem, assumed_align=16), [cutlass.Uint32(r0_1), cutlass.Uint32(r1_1), cutlass.Uint32(r2_1), cutlass.Uint32(r3_1)], prims.MMALayout.COL, shape=prims.StoreShape.M8N8)
                                prims.barrier_cta_sync((10 + wg_4), thread_count=128)
                                for atom_9 in cutlass.range_constexpr(0, 2, 1):
                                    out_row_1 = cutlass.Int32((((atom_9 * 8) + (warp_0_4 * 2)) + cutlass.Int32(cutlass_arith.divsi(cutlass.Int32(lane).ir_value(), cutlass.Int32(16).ir_value()))))
                                    row_in_atom_1 = cutlass.Int32((out_row_1 % 8))
                                    offset_3 = cutlass.Int32(cutlass.Int32(cutlass_arith.divsi(cutlass.Int32(((((wg_4 * 4096) + (cutlass.Int32(cutlass_arith.divsi(cutlass.Int32((lane % 16)).ir_value(), cutlass.Int32(8).ir_value())) * 2048)) + (out_row_1 * 128)) + (((lane % 8) ^ row_in_atom_1) * 16))).ir_value(), cutlass.Int32(4).ir_value())))
                                    if (task_valid > cutlass.Uint32(((wg_4 * 16) + out_row_1))):
                                        _smem_physical_39 = cute.make_tensor(cute.make_ptr(cutlass.Uint32, cutlass.Uint32(staging_addr), mem_space=cute.AddressSpace.smem, assumed_align=4), _flat_layout)
                                        _staging_reg_1 = cute.make_rmem_tensor((4,), cutlass.Uint32)
                                        _staging_reg_1[0] = _smem_physical_39[(offset_3) + 0]
                                        _staging_reg_1[1] = _smem_physical_39[(offset_3) + 1]
                                        _staging_reg_1[2] = _smem_physical_39[(offset_3) + 2]
                                        _staging_reg_1[3] = _smem_physical_39[(offset_3) + 3]
                                        owner_1 = cutlass.Uint32(_metadata[((((logical_block * 64) + cutlass.Uint32((wg_4 * 16))) + cutlass.Uint32(out_row_1)) * 3)])
                                        token_11 = cutlass.Uint32(_metadata[(((((logical_block * 64) + cutlass.Uint32((wg_4 * 16))) + cutlass.Uint32(out_row_1)) * 3) + 1)])
                                        slot_3 = cutlass.Uint32(_metadata[(((((logical_block * 64) + cutlass.Uint32((wg_4 * 16))) + cutlass.Uint32(out_row_1)) * 3) + 2)])
                                        element_1 = cutlass.Uint64(((((cutlass.Uint64(slot_3) * 384) + cutlass.Uint64(token_11)) * cutlass.Uint64((fc2_tiles * 256))) + cutlass.Uint64((((tile * 256) + cutlass.Uint32((cta_rank * 128))) + cutlass.Uint32(((lane % 16) * 8))))))
                                        _gmem_store_raw_40 = cutlass.Vector.from_elements([cutlass.Uint32(_staging_reg_1[0]), cutlass.Uint32(_staging_reg_1[1]), cutlass.Uint32(_staging_reg_1[2]), cutlass.Uint32(_staging_reg_1[3])], cutlass.Uint32)
                                        prims.store_ext(_gmem_store_raw_40.ir_value(), cute.make_ptr(cutlass.Uint32, cutlass.Uint64((_output_peers[owner_1] + (element_1 * 2))), mem_space=cute.AddressSpace.gmem, assumed_align=4))
                            prims.barrier_cta_sync(15, thread_count=256)
                        else:
                            warp_0_5 = cutlass.Int32((warp % 4))
                            wg_5 = cutlass.Int32(cutlass.Int32(cutlass_arith.divsi(cutlass.Int32((warp - 8)).ir_value(), cutlass.Int32(4).ir_value())))
                            if (task_valid <= cutlass.Uint32((wg_5 * 32))):
                                prims.tcgen05_fence('before_thread_sync')
                                _mbarrier_cluster_arrive_41 = prims.mapa(released_addr + e_stage[0], cutlass.Int32(cta_rank & 0xFFFFFFFE))
                                prims.mbarrier_arrive(_mbarrier_cluster_arrive_41, count=cutlass.Int32(1), scope=prims.MemScope.CTA)
                            else:
                                for atom_10 in cutlass.range_constexpr(0, 4, 1):
                                    address_5 = cutlass.Int32((((taddr + (e_stage[0] * 64)) + cutlass.Uint32((wg_5 * 32))) + cutlass.Uint32((atom_10 * 8))))
                                    _tmem_load_10 = cute.make_rmem_tensor((4,), cutlass.Float32)
                                    _tmem_load_42_atom = cute.make_copy_atom(cute.nvgpu.tcgen05.Ld16x256bOp(cute.nvgpu.tcgen05.Repetition.x1), cutlass.Float32)
                                    _tmem_load_42_src = cute.make_tensor(cute.make_ptr(cutlass.Float32, cutlass.Uint32(address_5), mem_space=cutlass.AddressSpace.tmem, assumed_align=16), cute.make_layout((4,), stride=(1,)))
                                    _tmem_load_42_dst = cute.make_tensor(_tmem_load_10.iterator + 0, cute.make_layout((4,), stride=(1,)))
                                    cute.copy(_tmem_load_42_atom, _tmem_load_42_src, _tmem_load_42_dst)
                                    cute.arch.fence_view_async_tmem_load()
                                    _tmem_load_11 = cute.make_rmem_tensor((4,), cutlass.Float32)
                                    _tmem_load_43_atom = cute.make_copy_atom(cute.nvgpu.tcgen05.Ld16x256bOp(cute.nvgpu.tcgen05.Repetition.x1), cutlass.Float32)
                                    _tmem_load_43_src = cute.make_tensor(cute.make_ptr(cutlass.Float32, cutlass.Uint32((address_5 + 1048576)), mem_space=cutlass.AddressSpace.tmem, assumed_align=16), cute.make_layout((4,), stride=(1,)))
                                    _tmem_load_43_dst = cute.make_tensor(_tmem_load_11.iterator + 0, cute.make_layout((4,), stride=(1,)))
                                    cute.copy(_tmem_load_43_atom, _tmem_load_43_src, _tmem_load_43_dst)
                                    cute.arch.fence_view_async_tmem_load()
                                    if (atom_10 == 3):
                                        prims.tcgen05_fence('before_thread_sync')
                                        _mbarrier_cluster_arrive_44 = prims.mapa(released_addr + e_stage[0], cutlass.Int32(cta_rank & 0xFFFFFFFE))
                                        prims.mbarrier_arrive(_mbarrier_cluster_arrive_44, count=cutlass.Int32(1), scope=prims.MemScope.CTA)
                                    token_12 = cutlass.Int32((((wg_5 * 32) + (atom_10 * 8)) + ((lane % 4) * 2)))
                                    channel_5 = cutlass.Int32((((cta_rank * 128) + (warp_0_5 * 32)) + cutlass.Int32(cutlass_arith.divsi(cutlass.Int32(lane).ir_value(), cutlass.Int32(4).ir_value()))))
                                    _bf16x2_8 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(_tmem_load_10[0]), cutlass.Float32(_tmem_load_10[1])))[1]), cutlass.Float32(((cutlass.Float32(_tmem_load_10[0]), cutlass.Float32(_tmem_load_10[1])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
                                    _bf16x2_9 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(_tmem_load_10[2]), cutlass.Float32(_tmem_load_10[3])))[1]), cutlass.Float32(((cutlass.Float32(_tmem_load_10[2]), cutlass.Float32(_tmem_load_10[3])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
                                    _bf16x2_10 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(_tmem_load_11[0]), cutlass.Float32(_tmem_load_11[1])))[1]), cutlass.Float32(((cutlass.Float32(_tmem_load_11[0]), cutlass.Float32(_tmem_load_11[1])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
                                    _bf16x2_11 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(_tmem_load_11[2]), cutlass.Float32(_tmem_load_11[3])))[1]), cutlass.Float32(((cutlass.Float32(_tmem_load_11[2]), cutlass.Float32(_tmem_load_11[3])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
                                    r0_2 = cutlass.Uint32(cutlass.Uint32(_bf16x2_8))
                                    r1_2 = cutlass.Uint32(cutlass.Uint32(_bf16x2_9))
                                    r2_2 = cutlass.Uint32(cutlass.Uint32(_bf16x2_10))
                                    r3_2 = cutlass.Uint32(cutlass.Uint32(_bf16x2_11))
                                    row_2 = cutlass.Int32((lane % 8))
                                    col_2 = cutlass.Int32((((warp_0_5 % 2) * 4) + cutlass.Int32(cutlass_arith.divsi(cutlass.Int32(lane).ir_value(), cutlass.Int32(8).ir_value()))))
                                    addr_5 = cutlass.Uint32((((((staging_addr + cutlass.Uint32((wg_5 * 8192))) + cutlass.Uint32((cutlass.Int32(cutlass_arith.divsi(cutlass.Int32(warp_0_5).ir_value(), cutlass.Int32(2).ir_value())) * 4096))) + cutlass.Uint32((atom_10 * 1024))) + cutlass.Uint32((row_2 * 128))) + cutlass.Uint32(((col_2 ^ row_2) * 16))))
                                    prims.stmatrix(cute.make_ptr(cutlass.Uint8, cutlass.Uint32(addr_5), mem_space=cute.AddressSpace.smem, assumed_align=16), [cutlass.Uint32(r0_2), cutlass.Uint32(r1_2), cutlass.Uint32(r2_2), cutlass.Uint32(r3_2)], prims.MMALayout.COL, shape=prims.StoreShape.M8N8)
                                prims.barrier_cta_sync((10 + wg_5), thread_count=128)
                                for atom_11 in cutlass.range_constexpr(0, 4, 1):
                                    out_row_2 = cutlass.Int32((((atom_11 * 8) + (warp_0_5 * 2)) + cutlass.Int32(cutlass_arith.divsi(cutlass.Int32(lane).ir_value(), cutlass.Int32(16).ir_value()))))
                                    row_in_atom_2 = cutlass.Int32((out_row_2 % 8))
                                    offset_4 = cutlass.Int32(cutlass.Int32(cutlass_arith.divsi(cutlass.Int32(((((wg_5 * 8192) + (cutlass.Int32(cutlass_arith.divsi(cutlass.Int32((lane % 16)).ir_value(), cutlass.Int32(8).ir_value())) * 4096)) + (out_row_2 * 128)) + (((lane % 8) ^ row_in_atom_2) * 16))).ir_value(), cutlass.Int32(4).ir_value())))
                                    if (task_valid > cutlass.Uint32(((wg_5 * 32) + out_row_2))):
                                        _smem_physical_45 = cute.make_tensor(cute.make_ptr(cutlass.Uint32, cutlass.Uint32(staging_addr), mem_space=cute.AddressSpace.smem, assumed_align=4), _flat_layout)
                                        _staging_reg_2 = cute.make_rmem_tensor((4,), cutlass.Uint32)
                                        _staging_reg_2[0] = _smem_physical_45[(offset_4) + 0]
                                        _staging_reg_2[1] = _smem_physical_45[(offset_4) + 1]
                                        _staging_reg_2[2] = _smem_physical_45[(offset_4) + 2]
                                        _staging_reg_2[3] = _smem_physical_45[(offset_4) + 3]
                                        owner_2 = cutlass.Uint32(_metadata[((((logical_block * 64) + cutlass.Uint32((wg_5 * 32))) + cutlass.Uint32(out_row_2)) * 3)])
                                        token_13 = cutlass.Uint32(_metadata[(((((logical_block * 64) + cutlass.Uint32((wg_5 * 32))) + cutlass.Uint32(out_row_2)) * 3) + 1)])
                                        slot_4 = cutlass.Uint32(_metadata[(((((logical_block * 64) + cutlass.Uint32((wg_5 * 32))) + cutlass.Uint32(out_row_2)) * 3) + 2)])
                                        element_2 = cutlass.Uint64(((((cutlass.Uint64(slot_4) * 384) + cutlass.Uint64(token_13)) * cutlass.Uint64((fc2_tiles * 256))) + cutlass.Uint64((((tile * 256) + cutlass.Uint32((cta_rank * 128))) + cutlass.Uint32(((lane % 16) * 8))))))
                                        _gmem_store_raw_46 = cutlass.Vector.from_elements([cutlass.Uint32(_staging_reg_2[0]), cutlass.Uint32(_staging_reg_2[1]), cutlass.Uint32(_staging_reg_2[2]), cutlass.Uint32(_staging_reg_2[3])], cutlass.Uint32)
                                        prims.store_ext(_gmem_store_raw_46.ir_value(), cute.make_ptr(cutlass.Uint32, cutlass.Uint64((_output_peers[owner_2] + (element_2 * 2))), mem_space=cute.AddressSpace.gmem, assumed_align=4))
                            prims.barrier_cta_sync(15, thread_count=256)
                _advanced_stage_47 = cutlass.Uint32(e_stage[0] + 1)
                _stage_wrapped_48 = cutlass.Boolean(_advanced_stage_47 == 2)
                e_stage[0] = cutlass.Uint32(cutlass.select_(_stage_wrapped_48, 0, _advanced_stage_47))
                _phase_done[0] = cutlass.Uint32(cutlass.select_(_stage_wrapped_48, _phase_done[0] ^ cutlass.Uint32(1), _phase_done[0]))
            _advanced_stage_49 = cutlass.Uint32(task_stage[0] + 1)
            _stage_wrapped_50 = cutlass.Boolean(_advanced_stage_49 == 2)
            task_stage[0] = cutlass.Uint32(cutlass.select_(_stage_wrapped_50, 0, _advanced_stage_49))
            _phase_task_full[0] = cutlass.Uint32(cutlass.select_(_stage_wrapped_50, _phase_task_full[0] ^ cutlass.Uint32(1), _phase_task_full[0]))
        if (warp == 8):
            cute.arch.dealloc_tmem(cute.make_ptr(cutlass.Float32, cutlass.Uint32(0), mem_space=cutlass.AddressSpace.tmem, assumed_align=1), 256, is_two_cta=True, arch='sm_100')
        completion_state[0] = cutlass.Uint32(0)
        if (tid == 256):
            completion_state[0] = cutlass.Uint32((prims.load_ext((cute.recast_ptr(status, dtype=cutlass.Uint32) + (0)), dtype=cutlass.Uint32, order=prims.MemOrder.VOLATILE) & 3))
        prims.barrier_cta_sync(2, thread_count=256)
        if (tid == 256):
            delta_2 = cutlass.Uint32((cutlass.Int32((2147483649 - num_bids)) if (bid == 0) else cutlass.Int32(1)))
            _atomic_old_7 = cute.arch.atomic_add(cute.recast_ptr(epilogue_grid, dtype=cutlass.Uint32), cutlass.Uint32(delta_2), sem='release', scope='gpu')
            while ((prims.load_ext(epilogue_grid + 0, dtype=cutlass.Uint32, order=prims.MemOrder.ACQUIRE, scope=prims.MemScope.GPU) ^ cutlass.Uint32(_atomic_old_7)) & cutlass.Uint32(0x80000000)) == 0:
                pass
        prims.barrier_cta_sync(2, thread_count=256)
        prims.barrier_cta_sync(1, thread_count=384)
        prims.barrier_cta_sync(1, thread_count=384)
        if (tid == 256):
            phase_5 = cutlass.Uint32((completion_state[0] & 1))
            sign_1 = cutlass.Uint32((completion_state[0] >> 1))
            if (bid == 0):
                _atomic_old_8 = cute.arch.atomic_add(status, cutlass.Uint32(1), sem='relaxed', scope='gpu')
                delta_3 = cutlass.Int32((cutlass.Int32(-1) if (sign_1 != 0) else cutlass.Int32(1)))
                prims.inline_ptx(
                    'multimem.red.release.sys.global.add.s32 [{$r0}], {$r1};',
                    read_only_args=[cutlass.Uint64(cute.make_ptr(cutlass.Int32, cutlass.Uint64((_signal_peers[16] + cutlass.Uint64((phase_5 * 4)))), mem_space=cute.AddressSpace.gmem, assumed_align=4).toint()), cutlass.Int32(delta_3)],
                )
            target_2 = cutlass.Uint32((cutlass.Int32(0) if (sign_1 != 0) else cutlass.Int32(16)))
            while prims.load_ext(signals + phase_5, dtype=cutlass.Uint32, order=prims.MemOrder.ACQUIRE, scope=prims.MemScope.SYS) != cutlass.Uint32(target_2):
                pass
        prims.barrier_cta_sync(2, thread_count=256)
        if (not True):
            prims.barrier_cta_sync(1, thread_count=384)
        if ((live_tokens * 12) <= cutlass.Uint32((num_bids * 8))):
            warp_0_6 = cutlass.Uint32((warp - 8))
            phase_2[0] = cutlass.Uint32(0)
            for item in cutlass.range(cutlass.Int32((cutlass.Uint32((bid * 8)) + warp_0_6)), cutlass.Int32((live_tokens * 12)), cutlass.Int32((num_bids * 8))):
                token_14 = cutlass.Uint32(cutlass.Int32(cutlass_arith.divsi(cutlass.Int32(item).ir_value(), cutlass.Int32(12).ir_value())))
                chunk_2 = cutlass.Uint32((item % 12))
                valid[0] = cutlass.Int32(0)
                if (lane < 8):
                    valid[0] = cutlass.Int32((cutlass.Int32(1) if (_ids[((token_14 * 8) + cutlass.Uint32(lane))] >= 0) else cutlass.Int32(0)))
                _vote_4 = cute.arch.vote_ballot_sync(cutlass.Boolean((valid[0] != 0)), cutlass.Uint32(0xFFFFFFFF)).bitcast(cutlass.Uint32)
                mask_1 = cutlass.Uint32(_vote_4)
                reduced = cute.make_rmem_tensor((8,), cutlass.Float32)
                for i_3 in cutlass.range_constexpr(0, 8, 1):
                    reduced[i_3] = cutlass.Float32(0.0)
                if (mask_1 != 0):
                    if prims.elect_sync():
                        _popc_2 = cutlass.Int32(cute.arch.popc(cutlass.Uint32(mask_1)))
                        cute.arch.mbarrier_arrive_and_expect_tx(combine_barriers_addr + (warp_0_6 * 2), (_popc_2 * 512))
                        for slot_5 in cutlass.range_constexpr(0, 8, 1):
                            if ((mask_1 & cutlass.Uint32((1 << slot_5))) != 0):
                                prims.cp_async_bulk_shared_cluster_global(
                                    cute.make_ptr(cutlass.Uint8, cutlass.Uint32((combine_storage_addr + (((warp_0_6 * 8) + cutlass.Uint32(slot_5)) * 512))), mem_space=cute.AddressSpace.smem, assumed_align=16), cute.recast_ptr(slots + ((((cutlass.Uint64(slot_5) * 384) + cutlass.Uint64(token_14)) * 6144) + cutlass.Uint64((chunk_2 * 512))), dtype=cutlass.Uint8), combine_barriers_addr + (warp_0_6 * 2),
                                    cutlass.Int32(512),
                                )
                    cute.arch.sync_warp()
                    cutlass_llvm.inline_asm(
                        res=None,
                        operands_=[(cutlass.Uint32((combine_barriers_addr + (warp_0_6 * 2)).toint())).ir_value(), (cutlass.Uint32(phase_2[0])).ir_value()],
                        asm_string='{ .reg .pred p; WAIT: mbarrier.try_wait.parity.shared::cta.b64 p, [$0], $1; @!p bra WAIT; }',
                        constraints='r,r',
                        has_side_effects=True,
                        is_align_stack=False,
                        asm_dialect=cutlass_llvm.AsmDialect.AD_ATT,
                    )
                    for slot_6 in cutlass.range_constexpr(0, 8, 1):
                        if ((mask_1 & cutlass.Uint32((1 << slot_6))) != 0):
                            for j_1 in cutlass.range_constexpr(0, 1, 1):
                                _smem_physical_51 = cute.make_tensor(cute.make_ptr(cutlass.Uint32, cutlass.Uint32(combine_storage_addr), mem_space=cute.AddressSpace.smem, assumed_align=4), _flat_layout)
                                _combine_storage_reg_0 = cute.make_rmem_tensor((4,), cutlass.Uint32)
                                _combine_storage_reg_0[0] = _smem_physical_51[(((((warp_0_6 * 8) + cutlass.Uint32(slot_6)) * 128) + cutlass.Uint32((((j_1 * 32) + lane) * 4)))) + 0]
                                _combine_storage_reg_0[1] = _smem_physical_51[(((((warp_0_6 * 8) + cutlass.Uint32(slot_6)) * 128) + cutlass.Uint32((((j_1 * 32) + lane) * 4)))) + 1]
                                _combine_storage_reg_0[2] = _smem_physical_51[(((((warp_0_6 * 8) + cutlass.Uint32(slot_6)) * 128) + cutlass.Uint32((((j_1 * 32) + lane) * 4)))) + 2]
                                _combine_storage_reg_0[3] = _smem_physical_51[(((((warp_0_6 * 8) + cutlass.Uint32(slot_6)) * 128) + cutlass.Uint32((((j_1 * 32) + lane) * 4)))) + 3]
                                for c in cutlass.range_constexpr(0, 4, 1):
                                    _bf16x2_add_f32_0 = cute.make_rmem_tensor((2,), cutlass.Float32)
                                    _bf16x2_add_52 = cutlass.Vector.from_elements([cutlass.Uint32(_combine_storage_reg_0[c])], cutlass.Uint32).bitcast(cutlass.BFloat16).to(cutlass.Float32)
                                    _bf16x2_add_f32_0[0] = cutlass.Float32(_bf16x2_add_52[0]) + cutlass.Float32(reduced[((j_1 * 8) + (c * 2))])
                                    _bf16x2_add_f32_0[1] = cutlass.Float32(_bf16x2_add_52[1]) + cutlass.Float32(reduced[(((j_1 * 8) + (c * 2)) + 1)])
                                    reduced[((j_1 * 8) + (c * 2))] = cutlass.Float32(_bf16x2_add_f32_0[0])
                                    reduced[(((j_1 * 8) + (c * 2)) + 1)] = cutlass.Float32(_bf16x2_add_f32_0[1])
                    cute.arch.sync_warp()
                    phase_2[0] = cutlass.Uint32((phase_2[0] ^ 1))
                cute.arch.cp_async_bulk_wait_group(0, read=False)
                cute.arch.sync_warp()
                for j_2 in cutlass.range_constexpr(0, 1, 1):
                    packed_output = cute.make_rmem_tensor((4,), cutlass.Uint32)
                    for c_1 in cutlass.range_constexpr(0, 4, 1):
                        _bf16x2_12 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(reduced[((j_2 * 8) + (c_1 * 2))]), cutlass.Float32(reduced[(((j_2 * 8) + (c_1 * 2)) + 1)])))[1]), cutlass.Float32(((cutlass.Float32(reduced[((j_2 * 8) + (c_1 * 2))]), cutlass.Float32(reduced[(((j_2 * 8) + (c_1 * 2)) + 1)])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
                        packed_output[c_1] = cutlass.Uint32(cutlass.Uint32(_bf16x2_12))
                    _smem_store_vector_53 = cutlass.Vector.from_elements([cutlass.Uint32(packed_output[(0) + 0]), cutlass.Uint32(packed_output[(0) + 1]), cutlass.Uint32(packed_output[(0) + 2]), cutlass.Uint32(packed_output[(0) + 3])], cutlass.Uint32)
                    prims.store_ext(_smem_store_vector_53.ir_value(), cute.make_ptr(cutlass.Uint32, cutlass.Uint32(((combine_storage_addr + ((64 + warp_0_6) * 512)) + cutlass.Uint32((((j_2 * 32) + lane) * 16)))), mem_space=cute.AddressSpace.smem, assumed_align=16))
                cute.arch.sync_warp()
                if prims.elect_sync():
                    cute.arch.fence_proxy("async.shared", space="cta")
                    prims.cp_async_bulk_global_shared_cta(cute.recast_ptr(output + ((cutlass.Uint64(token_14) * 6144) + cutlass.Uint64((chunk_2 * 512))), dtype=cutlass.Uint8), cute.make_ptr(cutlass.Uint8, cutlass.Uint32((combine_storage_addr + ((64 + warp_0_6) * 512))), mem_space=cute.AddressSpace.smem, assumed_align=16), cutlass.Int32(512))
                    cute.arch.cp_async_bulk_commit_group()
                cute.arch.sync_warp()
        else:
            warp_0_7 = cutlass.Uint32((warp - 8))
            phase_3[0] = cutlass.Uint32(0)
            for item_1 in cutlass.range(cutlass.Int32((cutlass.Uint32((bid * 8)) + warp_0_7)), cutlass.Int32((live_tokens * 3)), cutlass.Int32((num_bids * 8))):
                token_15 = cutlass.Uint32(cutlass.Int32(cutlass_arith.divsi(cutlass.Int32(item_1).ir_value(), cutlass.Int32(3).ir_value())))
                chunk_3 = cutlass.Uint32((item_1 % 3))
                valid_1[0] = cutlass.Int32(0)
                if (lane < 8):
                    valid_1[0] = cutlass.Int32((cutlass.Int32(1) if (_ids[((token_15 * 8) + cutlass.Uint32(lane))] >= 0) else cutlass.Int32(0)))
                _vote_5 = cute.arch.vote_ballot_sync(cutlass.Boolean((valid_1[0] != 0)), cutlass.Uint32(0xFFFFFFFF)).bitcast(cutlass.Uint32)
                mask_2 = cutlass.Uint32(_vote_5)
                reduced_1 = cute.make_rmem_tensor((32,), cutlass.Float32)
                for i_4 in cutlass.range_constexpr(0, 32, 1):
                    reduced_1[i_4] = cutlass.Float32(0.0)
                if (mask_2 != 0):
                    if prims.elect_sync():
                        _popc_3 = cutlass.Int32(cute.arch.popc(cutlass.Uint32(mask_2)))
                        cute.arch.mbarrier_arrive_and_expect_tx(combine_barriers_addr + (warp_0_7 * 2), (_popc_3 * 2048))
                        for slot_7 in cutlass.range_constexpr(0, 8, 1):
                            if ((mask_2 & cutlass.Uint32((1 << slot_7))) != 0):
                                prims.cp_async_bulk_shared_cluster_global(
                                    cute.make_ptr(cutlass.Uint8, cutlass.Uint32((combine_storage_addr + (((warp_0_7 * 8) + cutlass.Uint32(slot_7)) * 2048))), mem_space=cute.AddressSpace.smem, assumed_align=16), cute.recast_ptr(slots + ((((cutlass.Uint64(slot_7) * 384) + cutlass.Uint64(token_15)) * 6144) + cutlass.Uint64((chunk_3 * 2048))), dtype=cutlass.Uint8), combine_barriers_addr + (warp_0_7 * 2),
                                    cutlass.Int32(2048),
                                )
                    cute.arch.sync_warp()
                    cutlass_llvm.inline_asm(
                        res=None,
                        operands_=[(cutlass.Uint32((combine_barriers_addr + (warp_0_7 * 2)).toint())).ir_value(), (cutlass.Uint32(phase_3[0])).ir_value()],
                        asm_string='{ .reg .pred p; WAIT: mbarrier.try_wait.parity.shared::cta.b64 p, [$0], $1; @!p bra WAIT; }',
                        constraints='r,r',
                        has_side_effects=True,
                        is_align_stack=False,
                        asm_dialect=cutlass_llvm.AsmDialect.AD_ATT,
                    )
                    for slot_8 in cutlass.range_constexpr(0, 8, 1):
                        if ((mask_2 & cutlass.Uint32((1 << slot_8))) != 0):
                            for j_3 in cutlass.range_constexpr(0, 4, 1):
                                _smem_physical_54 = cute.make_tensor(cute.make_ptr(cutlass.Uint32, cutlass.Uint32(combine_storage_addr), mem_space=cute.AddressSpace.smem, assumed_align=4), _flat_layout)
                                _combine_storage_reg_1 = cute.make_rmem_tensor((4,), cutlass.Uint32)
                                _combine_storage_reg_1[0] = _smem_physical_54[(((((warp_0_7 * 8) + cutlass.Uint32(slot_8)) * 512) + cutlass.Uint32((((j_3 * 32) + lane) * 4)))) + 0]
                                _combine_storage_reg_1[1] = _smem_physical_54[(((((warp_0_7 * 8) + cutlass.Uint32(slot_8)) * 512) + cutlass.Uint32((((j_3 * 32) + lane) * 4)))) + 1]
                                _combine_storage_reg_1[2] = _smem_physical_54[(((((warp_0_7 * 8) + cutlass.Uint32(slot_8)) * 512) + cutlass.Uint32((((j_3 * 32) + lane) * 4)))) + 2]
                                _combine_storage_reg_1[3] = _smem_physical_54[(((((warp_0_7 * 8) + cutlass.Uint32(slot_8)) * 512) + cutlass.Uint32((((j_3 * 32) + lane) * 4)))) + 3]
                                for c_2 in cutlass.range_constexpr(0, 4, 1):
                                    _bf16x2_add_f32_1 = cute.make_rmem_tensor((2,), cutlass.Float32)
                                    _bf16x2_add_55 = cutlass.Vector.from_elements([cutlass.Uint32(_combine_storage_reg_1[c_2])], cutlass.Uint32).bitcast(cutlass.BFloat16).to(cutlass.Float32)
                                    _bf16x2_add_f32_1[0] = cutlass.Float32(_bf16x2_add_55[0]) + cutlass.Float32(reduced_1[((j_3 * 8) + (c_2 * 2))])
                                    _bf16x2_add_f32_1[1] = cutlass.Float32(_bf16x2_add_55[1]) + cutlass.Float32(reduced_1[(((j_3 * 8) + (c_2 * 2)) + 1)])
                                    reduced_1[((j_3 * 8) + (c_2 * 2))] = cutlass.Float32(_bf16x2_add_f32_1[0])
                                    reduced_1[(((j_3 * 8) + (c_2 * 2)) + 1)] = cutlass.Float32(_bf16x2_add_f32_1[1])
                    cute.arch.sync_warp()
                    phase_3[0] = cutlass.Uint32((phase_3[0] ^ 1))
                cute.arch.cp_async_bulk_wait_group(0, read=False)
                cute.arch.sync_warp()
                for j_4 in cutlass.range_constexpr(0, 4, 1):
                    packed_output_1 = cute.make_rmem_tensor((4,), cutlass.Uint32)
                    for c_3 in cutlass.range_constexpr(0, 4, 1):
                        _bf16x2_13 = prims.cvt_packfloat_f32(cutlass.Float32(((cutlass.Float32(reduced_1[((j_4 * 8) + (c_3 * 2))]), cutlass.Float32(reduced_1[(((j_4 * 8) + (c_3 * 2)) + 1)])))[1]), cutlass.Float32(((cutlass.Float32(reduced_1[((j_4 * 8) + (c_3 * 2))]), cutlass.Float32(reduced_1[(((j_4 * 8) + (c_3 * 2)) + 1)])))[0]), 0, prims.CVTPackFloat.BF16X2, rnd=prims.FPRoundingMode.RN)
                        packed_output_1[c_3] = cutlass.Uint32(cutlass.Uint32(_bf16x2_13))
                    _smem_store_vector_56 = cutlass.Vector.from_elements([cutlass.Uint32(packed_output_1[(0) + 0]), cutlass.Uint32(packed_output_1[(0) + 1]), cutlass.Uint32(packed_output_1[(0) + 2]), cutlass.Uint32(packed_output_1[(0) + 3])], cutlass.Uint32)
                    prims.store_ext(_smem_store_vector_56.ir_value(), cute.make_ptr(cutlass.Uint32, cutlass.Uint32(((combine_storage_addr + ((64 + warp_0_7) * 2048)) + cutlass.Uint32((((j_4 * 32) + lane) * 16)))), mem_space=cute.AddressSpace.smem, assumed_align=16))
                cute.arch.sync_warp()
                if prims.elect_sync():
                    cute.arch.fence_proxy("async.shared", space="cta")
                    prims.cp_async_bulk_global_shared_cta(cute.recast_ptr(output + ((cutlass.Uint64(token_15) * 6144) + cutlass.Uint64((chunk_3 * 2048))), dtype=cutlass.Uint8), cute.make_ptr(cutlass.Uint8, cutlass.Uint32((combine_storage_addr + ((64 + warp_0_7) * 2048))), mem_space=cute.AddressSpace.smem, assumed_align=16), cutlass.Int32(2048))
                    cute.arch.cp_async_bulk_commit_group()
                cute.arch.sync_warp()
    elif warp == 4:
        a16_stage = cute.make_rmem_tensor((1,), cutlass.Uint32)
        a32_stage = cute.make_rmem_tensor((1,), cutlass.Uint32)
        a64_stage = cute.make_rmem_tensor((1,), cutlass.Uint32)
        task_stage_1 = cute.make_rmem_tensor((1,), cutlass.Uint32)
        running_1 = cute.make_rmem_tensor((1,), cutlass.Int32)
        _phase_task_full_1 = cute.make_rmem_tensor((1,), cutlass.Uint32)
        _phase_empty16 = cute.make_rmem_tensor((1,), cutlass.Uint32)
        _phase_empty32 = cute.make_rmem_tensor((1,), cutlass.Uint32)
        _phase_empty64 = cute.make_rmem_tensor((1,), cutlass.Uint32)
        a16_stage[0] = cutlass.Uint32(0)
        a32_stage[0] = cutlass.Uint32(0)
        a64_stage[0] = cutlass.Uint32(0)
        task_stage_1[0] = cutlass.Uint32(0)
        running_1[0] = cutlass.Int32(1)
        _phase_task_full_1[0] = cutlass.Uint32(0)
        _phase_empty16[0] = cutlass.Uint32(1)
        _phase_empty32[0] = cutlass.Uint32(1)
        _phase_empty64[0] = cutlass.Uint32(1)
        while cutlass.Boolean((running_1[0] != 0)):
            while not prims.mbarrier_wait_parity(task_full_addr + task_stage_1[0], _phase_task_full_1[0], prims.MBarrierWait.TRY, scope=prims.MBarrierScope.CTA, order=prims.MemOrder.ACQUIRE):
                pass
            _smem_physical_57 = cute.make_tensor(cute.make_ptr(cutlass.Uint32, cutlass.Uint32(tasks_addr), mem_space=cute.AddressSpace.smem, assumed_align=4), _flat_layout)
            _tasks_reg_0 = cute.make_rmem_tensor((8,), cutlass.Uint32)
            _tasks_reg_0[0] = _smem_physical_57[((task_stage_1[0] * 8)) + 0]
            _tasks_reg_0[1] = _smem_physical_57[((task_stage_1[0] * 8)) + 1]
            _tasks_reg_0[2] = _smem_physical_57[((task_stage_1[0] * 8)) + 2]
            _tasks_reg_0[3] = _smem_physical_57[((task_stage_1[0] * 8)) + 3]
            _tasks_reg_0[4] = _smem_physical_57[((task_stage_1[0] * 8)) + 4]
            _tasks_reg_0[5] = _smem_physical_57[((task_stage_1[0] * 8)) + 5]
            _tasks_reg_0[6] = _smem_physical_57[((task_stage_1[0] * 8)) + 6]
            _tasks_reg_0[7] = _smem_physical_57[((task_stage_1[0] * 8)) + 7]
            if (_tasks_reg_0[0] == 0):
                running_1[0] = cutlass.Int32(0)
            else:
                phase_6 = cutlass.Uint32((_tasks_reg_0[0] - 1))
                tile_1 = cutlass.Uint32(_tasks_reg_0[3])
                expert_3 = cutlass.Uint32(_tasks_reg_0[1])
                logical_block_1 = cutlass.Uint32(_tasks_reg_0[4])
                pool_block_2 = cutlass.Uint32((logical_block_1 % cutlass.Uint32(ring_blocks)))
                ring_epoch_1 = cutlass.Uint32(cutlass.Uint32(cutlass_arith.divui(cutlass.Uint32(logical_block_1).ir_value(), cutlass.Uint32(cutlass.Uint32(ring_blocks)).ir_value())))
                task_valid_1 = cutlass.Uint32(_tasks_reg_0[5])
                task_n_1 = cutlass.Uint32((cutlass.Uint32(cutlass.Uint32(64)) if (task_valid_1 > 32) else cutlass.Uint32((cutlass.Uint32(cutlass_arith.divui(cutlass.Uint32((task_valid_1 + 15)).ir_value(), cutlass.Uint32(16).ir_value())) * 16))))
                layout_n = cutlass.Uint32((_tasks_reg_0[6] >> 16))
                k_steps = cutlass.Uint32(cutlass.Uint32(cutlass_arith.divui(cutlass.Uint32(_tasks_reg_0[7]).ir_value(), cutlass.Uint32(128).ir_value())))
                if (phase_6 == 0):
                    while prims.load_ext(l1_full + pool_block_2, dtype=cutlass.Uint32, order=prims.MemOrder.ACQUIRE, scope=prims.MemScope.GPU) != cutlass.Uint32(((ring_epoch_1 + 1) * 64)):
                        pass
                else:
                    while prims.load_ext(l2_full + pool_block_2, dtype=cutlass.Uint32, order=prims.MemOrder.ACQUIRE, scope=prims.MemScope.GPU) != cutlass.Uint32(((ring_epoch_1 + 1) * 80)):
                        pass
                if (phase_6 == 0):
                    if (layout_n == 16):
                        for k_3 in cutlass.range(cutlass.Int32(0), cutlass.Int32(cutlass.Uint32(cutlass_arith.divui(cutlass.Uint32(k_steps).ir_value(), cutlass.Uint32(2).ir_value()))), cutlass.Int32(1)):
                            while not prims.mbarrier_wait_parity(empty16_addr + a16_stage[0], _phase_empty16[0], prims.MBarrierWait.TRY, scope=prims.MBarrierScope.CTA, order=prims.MemOrder.ACQUIRE):
                                pass
                            if prims.elect_sync():
                                _tma_mbar_58 = cutlass.Array((full16_addr + a16_stage[0]).to_llvm_ptr(), shape=1, dtype=cutlass.Uint64)
                                prims.cp_async_bulk_tensor_shared_cluster_global(
                                    cute.make_ptr(cutlass.Uint8, cutlass.Uint32((x16_addr + (a16_stage[0] * 2048))), mem_space=cute.AddressSpace.smem, assumed_align=16),
                                    X1_8.get_ptr(),
                                    [cutlass.Int32((k_3 * 256)), cutlass.Int32(((pool_block_2 * 64) + cutlass.Uint32((cta_rank * 8))))],
                                    _tma_mbar_58,
                                    [],
                                    l2_cache_hint=0x14F0000000000000,
                                    mode=prims.TMALoadMode.TILE,
                                    group=prims.CTAGroup.CTA_2,
                                )
                                _tma_mbar_59 = cutlass.Array((full16_addr + a16_stage[0]).to_llvm_ptr(), shape=1, dtype=cutlass.Uint64)
                                prims.cp_async_bulk_tensor_shared_cluster_global(
                                    cute.make_ptr(cutlass.Uint8, cutlass.Uint32((x16_hi_addr + (a16_stage[0] * 2048))), mem_space=cute.AddressSpace.smem, assumed_align=16),
                                    X1_8.get_ptr(),
                                    [cutlass.Int32(((k_3 * 256) + 128)), cutlass.Int32(((pool_block_2 * 64) + cutlass.Uint32((cta_rank * 8))))],
                                    _tma_mbar_59,
                                    [],
                                    l2_cache_hint=0x14F0000000000000,
                                    mode=prims.TMALoadMode.TILE,
                                    group=prims.CTAGroup.CTA_2,
                                )
                                _tma_mbar_60 = cutlass.Array((full16_addr + a16_stage[0]).to_llvm_ptr(), shape=1, dtype=cutlass.Uint64)
                                prims.cp_async_bulk_tensor_shared_cluster_global(
                                    cute.make_ptr(cutlass.Uint8, cutlass.Uint32((sx16_addr + (a16_stage[0] * 1024))), mem_space=cute.AddressSpace.smem, assumed_align=16),
                                    SX1.get_ptr(),
                                    [cutlass.Int32((pool_block_2 * 128)), cutlass.Int32((k_3 * 2))],
                                    _tma_mbar_60,
                                    [],
                                    l2_cache_hint=0x14F0000000000000,
                                    mode=prims.TMALoadMode.TILE,
                                    group=prims.CTAGroup.CTA_2,
                                )
                                if (cta_rank == 0):
                                    _mbarrier_cluster_expect_tx_61 = prims.mapa(full16_addr + a16_stage[0], cutlass.Int32(cta_rank & 0xFFFFFFFE))
                                    prims.mbarrier_arrive_expect_tx(_mbarrier_cluster_expect_tx_61, 6144, scope=prims.MemScope.CTA)
                                else:
                                    _mbarrier_cluster_arrive_62 = prims.mapa(full16_addr + a16_stage[0], cutlass.Int32(cta_rank & 0xFFFFFFFE))
                                    prims.mbarrier_arrive(_mbarrier_cluster_arrive_62, count=cutlass.Int32(1), scope=prims.MemScope.CTA)
                            _advanced_stage_63 = cutlass.Uint32(a16_stage[0] + 1)
                            _stage_wrapped_64 = cutlass.Boolean(_advanced_stage_63 == 5)
                            a16_stage[0] = cutlass.Uint32(cutlass.select_(_stage_wrapped_64, 0, _advanced_stage_63))
                            _phase_empty16[0] = cutlass.Uint32(cutlass.select_(_stage_wrapped_64, _phase_empty16[0] ^ cutlass.Uint32(1), _phase_empty16[0]))
                    else:
                        if (layout_n == 32):
                            if (task_n_1 == 16):
                                for k_4 in cutlass.range(cutlass.Int32(0), cutlass.Int32(k_steps), cutlass.Int32(1)):
                                    while not prims.mbarrier_wait_parity(empty32_addr + a32_stage[0], _phase_empty32[0], prims.MBarrierWait.TRY, scope=prims.MBarrierScope.CTA, order=prims.MemOrder.ACQUIRE):
                                        pass
                                    if prims.elect_sync():
                                        _tma_mbar_65 = cutlass.Array((full32_addr + a32_stage[0]).to_llvm_ptr(), shape=1, dtype=cutlass.Uint64)
                                        prims.cp_async_bulk_tensor_shared_cluster_global(
                                            cute.make_ptr(cutlass.Uint8, cutlass.Uint32((x32_addr + (a32_stage[0] * 2048))), mem_space=cute.AddressSpace.smem, assumed_align=16),
                                            X1_8.get_ptr(),
                                            [cutlass.Int32((k_4 * 128)), cutlass.Int32(((pool_block_2 * 64) + cutlass.Uint32((cta_rank * 8))))],
                                            _tma_mbar_65,
                                            [],
                                            l2_cache_hint=0x14F0000000000000,
                                            mode=prims.TMALoadMode.TILE,
                                            group=prims.CTAGroup.CTA_2,
                                        )
                                        _tma_mbar_66 = cutlass.Array((full32_addr + a32_stage[0]).to_llvm_ptr(), shape=1, dtype=cutlass.Uint64)
                                        prims.cp_async_bulk_tensor_shared_cluster_global(
                                            cute.make_ptr(cutlass.Uint8, cutlass.Uint32((sx32_addr + (a32_stage[0] * 512))), mem_space=cute.AddressSpace.smem, assumed_align=16),
                                            SX1_1.get_ptr(),
                                            [cutlass.Int32((pool_block_2 * 128)), cutlass.Int32(k_4)],
                                            _tma_mbar_66,
                                            [],
                                            l2_cache_hint=0x14F0000000000000,
                                            mode=prims.TMALoadMode.TILE,
                                            group=prims.CTAGroup.CTA_2,
                                        )
                                        if (cta_rank == 0):
                                            _mbarrier_cluster_expect_tx_67 = prims.mapa(full32_addr + a32_stage[0], cutlass.Int32(cta_rank & 0xFFFFFFFE))
                                            prims.mbarrier_arrive_expect_tx(_mbarrier_cluster_expect_tx_67, 3072, scope=prims.MemScope.CTA)
                                        else:
                                            _mbarrier_cluster_arrive_68 = prims.mapa(full32_addr + a32_stage[0], cutlass.Int32(cta_rank & 0xFFFFFFFE))
                                            prims.mbarrier_arrive(_mbarrier_cluster_arrive_68, count=cutlass.Int32(1), scope=prims.MemScope.CTA)
                                    _advanced_stage_69 = cutlass.Uint32(a32_stage[0] + 1)
                                    _stage_wrapped_70 = cutlass.Boolean(_advanced_stage_69 == 10)
                                    a32_stage[0] = cutlass.Uint32(cutlass.select_(_stage_wrapped_70, 0, _advanced_stage_69))
                                    _phase_empty32[0] = cutlass.Uint32(cutlass.select_(_stage_wrapped_70, _phase_empty32[0] ^ cutlass.Uint32(1), _phase_empty32[0]))
                            else:
                                for k_5 in cutlass.range(cutlass.Int32(0), cutlass.Int32(k_steps), cutlass.Int32(1)):
                                    while not prims.mbarrier_wait_parity(empty32_addr + a32_stage[0], _phase_empty32[0], prims.MBarrierWait.TRY, scope=prims.MBarrierScope.CTA, order=prims.MemOrder.ACQUIRE):
                                        pass
                                    if prims.elect_sync():
                                        _tma_mbar_71 = cutlass.Array((full32_addr + a32_stage[0]).to_llvm_ptr(), shape=1, dtype=cutlass.Uint64)
                                        prims.cp_async_bulk_tensor_shared_cluster_global(
                                            cute.make_ptr(cutlass.Uint8, cutlass.Uint32((x32_addr + (a32_stage[0] * 2048))), mem_space=cute.AddressSpace.smem, assumed_align=16),
                                            X1.get_ptr(),
                                            [cutlass.Int32((k_5 * 128)), cutlass.Int32(((pool_block_2 * 64) + cutlass.Uint32((cta_rank * 16))))],
                                            _tma_mbar_71,
                                            [],
                                            l2_cache_hint=0x14F0000000000000,
                                            mode=prims.TMALoadMode.TILE,
                                            group=prims.CTAGroup.CTA_2,
                                        )
                                        _tma_mbar_72 = cutlass.Array((full32_addr + a32_stage[0]).to_llvm_ptr(), shape=1, dtype=cutlass.Uint64)
                                        prims.cp_async_bulk_tensor_shared_cluster_global(
                                            cute.make_ptr(cutlass.Uint8, cutlass.Uint32((sx32_addr + (a32_stage[0] * 512))), mem_space=cute.AddressSpace.smem, assumed_align=16),
                                            SX1_1.get_ptr(),
                                            [cutlass.Int32((pool_block_2 * 128)), cutlass.Int32(k_5)],
                                            _tma_mbar_72,
                                            [],
                                            l2_cache_hint=0x14F0000000000000,
                                            mode=prims.TMALoadMode.TILE,
                                            group=prims.CTAGroup.CTA_2,
                                        )
                                        if (cta_rank == 0):
                                            _mbarrier_cluster_expect_tx_73 = prims.mapa(full32_addr + a32_stage[0], cutlass.Int32(cta_rank & 0xFFFFFFFE))
                                            prims.mbarrier_arrive_expect_tx(_mbarrier_cluster_expect_tx_73, 5120, scope=prims.MemScope.CTA)
                                        else:
                                            _mbarrier_cluster_arrive_74 = prims.mapa(full32_addr + a32_stage[0], cutlass.Int32(cta_rank & 0xFFFFFFFE))
                                            prims.mbarrier_arrive(_mbarrier_cluster_arrive_74, count=cutlass.Int32(1), scope=prims.MemScope.CTA)
                                    _advanced_stage_75 = cutlass.Uint32(a32_stage[0] + 1)
                                    _stage_wrapped_76 = cutlass.Boolean(_advanced_stage_75 == 10)
                                    a32_stage[0] = cutlass.Uint32(cutlass.select_(_stage_wrapped_76, 0, _advanced_stage_75))
                                    _phase_empty32[0] = cutlass.Uint32(cutlass.select_(_stage_wrapped_76, _phase_empty32[0] ^ cutlass.Uint32(1), _phase_empty32[0]))
                        else:
                            if (task_n_1 == 16):
                                for k_6 in cutlass.range(cutlass.Int32(0), cutlass.Int32(k_steps), cutlass.Int32(1)):
                                    while not prims.mbarrier_wait_parity(empty64_addr + a64_stage[0], _phase_empty64[0], prims.MBarrierWait.TRY, scope=prims.MBarrierScope.CTA, order=prims.MemOrder.ACQUIRE):
                                        pass
                                    if prims.elect_sync():
                                        _tma_mbar_77 = cutlass.Array((full64_addr + a64_stage[0]).to_llvm_ptr(), shape=1, dtype=cutlass.Uint64)
                                        prims.cp_async_bulk_tensor_shared_cluster_global(
                                            cute.make_ptr(cutlass.Uint8, cutlass.Uint32((x64_addr + (a64_stage[0] * 4096))), mem_space=cute.AddressSpace.smem, assumed_align=16),
                                            X1_8.get_ptr(),
                                            [cutlass.Int32((k_6 * 128)), cutlass.Int32(((pool_block_2 * 64) + cutlass.Uint32((cta_rank * 8))))],
                                            _tma_mbar_77,
                                            [],
                                            l2_cache_hint=0x14F0000000000000,
                                            mode=prims.TMALoadMode.TILE,
                                            group=prims.CTAGroup.CTA_2,
                                        )
                                        _tma_mbar_78 = cutlass.Array((full64_addr + a64_stage[0]).to_llvm_ptr(), shape=1, dtype=cutlass.Uint64)
                                        prims.cp_async_bulk_tensor_shared_cluster_global(
                                            cute.make_ptr(cutlass.Uint8, cutlass.Uint32((sx64_addr + (a64_stage[0] * 512))), mem_space=cute.AddressSpace.smem, assumed_align=16),
                                            SX1_1.get_ptr(),
                                            [cutlass.Int32((pool_block_2 * 128)), cutlass.Int32(k_6)],
                                            _tma_mbar_78,
                                            [],
                                            l2_cache_hint=0x14F0000000000000,
                                            mode=prims.TMALoadMode.TILE,
                                            group=prims.CTAGroup.CTA_2,
                                        )
                                        if (cta_rank == 0):
                                            _mbarrier_cluster_expect_tx_79 = prims.mapa(full64_addr + a64_stage[0], cutlass.Int32(cta_rank & 0xFFFFFFFE))
                                            prims.mbarrier_arrive_expect_tx(_mbarrier_cluster_expect_tx_79, 3072, scope=prims.MemScope.CTA)
                                        else:
                                            _mbarrier_cluster_arrive_80 = prims.mapa(full64_addr + a64_stage[0], cutlass.Int32(cta_rank & 0xFFFFFFFE))
                                            prims.mbarrier_arrive(_mbarrier_cluster_arrive_80, count=cutlass.Int32(1), scope=prims.MemScope.CTA)
                                    _advanced_stage_81 = cutlass.Uint32(a64_stage[0] + 1)
                                    _stage_wrapped_82 = cutlass.Boolean(_advanced_stage_81 == 9)
                                    a64_stage[0] = cutlass.Uint32(cutlass.select_(_stage_wrapped_82, 0, _advanced_stage_81))
                                    _phase_empty64[0] = cutlass.Uint32(cutlass.select_(_stage_wrapped_82, _phase_empty64[0] ^ cutlass.Uint32(1), _phase_empty64[0]))
                            else:
                                if (task_n_1 == 32):
                                    for k_7 in cutlass.range(cutlass.Int32(0), cutlass.Int32(k_steps), cutlass.Int32(1)):
                                        while not prims.mbarrier_wait_parity(empty64_addr + a64_stage[0], _phase_empty64[0], prims.MBarrierWait.TRY, scope=prims.MBarrierScope.CTA, order=prims.MemOrder.ACQUIRE):
                                            pass
                                        if prims.elect_sync():
                                            _tma_mbar_83 = cutlass.Array((full64_addr + a64_stage[0]).to_llvm_ptr(), shape=1, dtype=cutlass.Uint64)
                                            prims.cp_async_bulk_tensor_shared_cluster_global(
                                                cute.make_ptr(cutlass.Uint8, cutlass.Uint32((x64_addr + (a64_stage[0] * 4096))), mem_space=cute.AddressSpace.smem, assumed_align=16),
                                                X1.get_ptr(),
                                                [cutlass.Int32((k_7 * 128)), cutlass.Int32(((pool_block_2 * 64) + cutlass.Uint32((cta_rank * 16))))],
                                                _tma_mbar_83,
                                                [],
                                                l2_cache_hint=0x14F0000000000000,
                                                mode=prims.TMALoadMode.TILE,
                                                group=prims.CTAGroup.CTA_2,
                                            )
                                            _tma_mbar_84 = cutlass.Array((full64_addr + a64_stage[0]).to_llvm_ptr(), shape=1, dtype=cutlass.Uint64)
                                            prims.cp_async_bulk_tensor_shared_cluster_global(
                                                cute.make_ptr(cutlass.Uint8, cutlass.Uint32((sx64_addr + (a64_stage[0] * 512))), mem_space=cute.AddressSpace.smem, assumed_align=16),
                                                SX1_1.get_ptr(),
                                                [cutlass.Int32((pool_block_2 * 128)), cutlass.Int32(k_7)],
                                                _tma_mbar_84,
                                                [],
                                                l2_cache_hint=0x14F0000000000000,
                                                mode=prims.TMALoadMode.TILE,
                                                group=prims.CTAGroup.CTA_2,
                                            )
                                            if (cta_rank == 0):
                                                _mbarrier_cluster_expect_tx_85 = prims.mapa(full64_addr + a64_stage[0], cutlass.Int32(cta_rank & 0xFFFFFFFE))
                                                prims.mbarrier_arrive_expect_tx(_mbarrier_cluster_expect_tx_85, 5120, scope=prims.MemScope.CTA)
                                            else:
                                                _mbarrier_cluster_arrive_86 = prims.mapa(full64_addr + a64_stage[0], cutlass.Int32(cta_rank & 0xFFFFFFFE))
                                                prims.mbarrier_arrive(_mbarrier_cluster_arrive_86, count=cutlass.Int32(1), scope=prims.MemScope.CTA)
                                        _advanced_stage_87 = cutlass.Uint32(a64_stage[0] + 1)
                                        _stage_wrapped_88 = cutlass.Boolean(_advanced_stage_87 == 9)
                                        a64_stage[0] = cutlass.Uint32(cutlass.select_(_stage_wrapped_88, 0, _advanced_stage_87))
                                        _phase_empty64[0] = cutlass.Uint32(cutlass.select_(_stage_wrapped_88, _phase_empty64[0] ^ cutlass.Uint32(1), _phase_empty64[0]))
                                else:
                                    for k_8 in cutlass.range(cutlass.Int32(0), cutlass.Int32(k_steps), cutlass.Int32(1)):
                                        while not prims.mbarrier_wait_parity(empty64_addr + a64_stage[0], _phase_empty64[0], prims.MBarrierWait.TRY, scope=prims.MBarrierScope.CTA, order=prims.MemOrder.ACQUIRE):
                                            pass
                                        if prims.elect_sync():
                                            _tma_mbar_89 = cutlass.Array((full64_addr + a64_stage[0]).to_llvm_ptr(), shape=1, dtype=cutlass.Uint64)
                                            prims.cp_async_bulk_tensor_shared_cluster_global(
                                                cute.make_ptr(cutlass.Uint8, cutlass.Uint32((x64_addr + (a64_stage[0] * 4096))), mem_space=cute.AddressSpace.smem, assumed_align=16),
                                                X1_32.get_ptr(),
                                                [cutlass.Int32((k_8 * 128)), cutlass.Int32(((pool_block_2 * 64) + cutlass.Uint32((cta_rank * 32))))],
                                                _tma_mbar_89,
                                                [],
                                                l2_cache_hint=0x14F0000000000000,
                                                mode=prims.TMALoadMode.TILE,
                                                group=prims.CTAGroup.CTA_2,
                                            )
                                            _tma_mbar_90 = cutlass.Array((full64_addr + a64_stage[0]).to_llvm_ptr(), shape=1, dtype=cutlass.Uint64)
                                            prims.cp_async_bulk_tensor_shared_cluster_global(
                                                cute.make_ptr(cutlass.Uint8, cutlass.Uint32((sx64_addr + (a64_stage[0] * 512))), mem_space=cute.AddressSpace.smem, assumed_align=16),
                                                SX1_1.get_ptr(),
                                                [cutlass.Int32((pool_block_2 * 128)), cutlass.Int32(k_8)],
                                                _tma_mbar_90,
                                                [],
                                                l2_cache_hint=0x14F0000000000000,
                                                mode=prims.TMALoadMode.TILE,
                                                group=prims.CTAGroup.CTA_2,
                                            )
                                            if (cta_rank == 0):
                                                _mbarrier_cluster_expect_tx_91 = prims.mapa(full64_addr + a64_stage[0], cutlass.Int32(cta_rank & 0xFFFFFFFE))
                                                prims.mbarrier_arrive_expect_tx(_mbarrier_cluster_expect_tx_91, 9216, scope=prims.MemScope.CTA)
                                            else:
                                                _mbarrier_cluster_arrive_92 = prims.mapa(full64_addr + a64_stage[0], cutlass.Int32(cta_rank & 0xFFFFFFFE))
                                                prims.mbarrier_arrive(_mbarrier_cluster_arrive_92, count=cutlass.Int32(1), scope=prims.MemScope.CTA)
                                        _advanced_stage_93 = cutlass.Uint32(a64_stage[0] + 1)
                                        _stage_wrapped_94 = cutlass.Boolean(_advanced_stage_93 == 9)
                                        a64_stage[0] = cutlass.Uint32(cutlass.select_(_stage_wrapped_94, 0, _advanced_stage_93))
                                        _phase_empty64[0] = cutlass.Uint32(cutlass.select_(_stage_wrapped_94, _phase_empty64[0] ^ cutlass.Uint32(1), _phase_empty64[0]))
                else:
                    if (layout_n == 16):
                        for k_9 in cutlass.range(cutlass.Int32(0), cutlass.Int32(cutlass.Uint32(cutlass_arith.divui(cutlass.Uint32(k_steps).ir_value(), cutlass.Uint32(2).ir_value()))), cutlass.Int32(1)):
                            while not prims.mbarrier_wait_parity(empty16_addr + a16_stage[0], _phase_empty16[0], prims.MBarrierWait.TRY, scope=prims.MBarrierScope.CTA, order=prims.MemOrder.ACQUIRE):
                                pass
                            if prims.elect_sync():
                                _tma_mbar_95 = cutlass.Array((full16_addr + a16_stage[0]).to_llvm_ptr(), shape=1, dtype=cutlass.Uint64)
                                prims.cp_async_bulk_tensor_shared_cluster_global(
                                    cute.make_ptr(cutlass.Uint8, cutlass.Uint32((x16_addr + (a16_stage[0] * 2048))), mem_space=cute.AddressSpace.smem, assumed_align=16),
                                    X2_8.get_ptr(),
                                    [cutlass.Int32((k_9 * 256)), cutlass.Int32(((pool_block_2 * 64) + cutlass.Uint32((cta_rank * 8))))],
                                    _tma_mbar_95,
                                    [],
                                    l2_cache_hint=0x14F0000000000000,
                                    mode=prims.TMALoadMode.TILE,
                                    group=prims.CTAGroup.CTA_2,
                                )
                                _tma_mbar_96 = cutlass.Array((full16_addr + a16_stage[0]).to_llvm_ptr(), shape=1, dtype=cutlass.Uint64)
                                prims.cp_async_bulk_tensor_shared_cluster_global(
                                    cute.make_ptr(cutlass.Uint8, cutlass.Uint32((x16_hi_addr + (a16_stage[0] * 2048))), mem_space=cute.AddressSpace.smem, assumed_align=16),
                                    X2_8.get_ptr(),
                                    [cutlass.Int32(((k_9 * 256) + 128)), cutlass.Int32(((pool_block_2 * 64) + cutlass.Uint32((cta_rank * 8))))],
                                    _tma_mbar_96,
                                    [],
                                    l2_cache_hint=0x14F0000000000000,
                                    mode=prims.TMALoadMode.TILE,
                                    group=prims.CTAGroup.CTA_2,
                                )
                                _tma_mbar_97 = cutlass.Array((full16_addr + a16_stage[0]).to_llvm_ptr(), shape=1, dtype=cutlass.Uint64)
                                prims.cp_async_bulk_tensor_shared_cluster_global(
                                    cute.make_ptr(cutlass.Uint8, cutlass.Uint32((sx16_addr + (a16_stage[0] * 1024))), mem_space=cute.AddressSpace.smem, assumed_align=16),
                                    SX2.get_ptr(),
                                    [cutlass.Int32((pool_block_2 * 128)), cutlass.Int32((k_9 * 2))],
                                    _tma_mbar_97,
                                    [],
                                    l2_cache_hint=0x14F0000000000000,
                                    mode=prims.TMALoadMode.TILE,
                                    group=prims.CTAGroup.CTA_2,
                                )
                                if (cta_rank == 0):
                                    _mbarrier_cluster_expect_tx_98 = prims.mapa(full16_addr + a16_stage[0], cutlass.Int32(cta_rank & 0xFFFFFFFE))
                                    prims.mbarrier_arrive_expect_tx(_mbarrier_cluster_expect_tx_98, 6144, scope=prims.MemScope.CTA)
                                else:
                                    _mbarrier_cluster_arrive_99 = prims.mapa(full16_addr + a16_stage[0], cutlass.Int32(cta_rank & 0xFFFFFFFE))
                                    prims.mbarrier_arrive(_mbarrier_cluster_arrive_99, count=cutlass.Int32(1), scope=prims.MemScope.CTA)
                            _advanced_stage_100 = cutlass.Uint32(a16_stage[0] + 1)
                            _stage_wrapped_101 = cutlass.Boolean(_advanced_stage_100 == 5)
                            a16_stage[0] = cutlass.Uint32(cutlass.select_(_stage_wrapped_101, 0, _advanced_stage_100))
                            _phase_empty16[0] = cutlass.Uint32(cutlass.select_(_stage_wrapped_101, _phase_empty16[0] ^ cutlass.Uint32(1), _phase_empty16[0]))
                    else:
                        if (layout_n == 32):
                            if (task_n_1 == 16):
                                for k_10 in cutlass.range(cutlass.Int32(0), cutlass.Int32(k_steps), cutlass.Int32(1)):
                                    while not prims.mbarrier_wait_parity(empty32_addr + a32_stage[0], _phase_empty32[0], prims.MBarrierWait.TRY, scope=prims.MBarrierScope.CTA, order=prims.MemOrder.ACQUIRE):
                                        pass
                                    if prims.elect_sync():
                                        _tma_mbar_102 = cutlass.Array((full32_addr + a32_stage[0]).to_llvm_ptr(), shape=1, dtype=cutlass.Uint64)
                                        prims.cp_async_bulk_tensor_shared_cluster_global(
                                            cute.make_ptr(cutlass.Uint8, cutlass.Uint32((x32_addr + (a32_stage[0] * 2048))), mem_space=cute.AddressSpace.smem, assumed_align=16),
                                            X2_8.get_ptr(),
                                            [cutlass.Int32((k_10 * 128)), cutlass.Int32(((pool_block_2 * 64) + cutlass.Uint32((cta_rank * 8))))],
                                            _tma_mbar_102,
                                            [],
                                            l2_cache_hint=0x14F0000000000000,
                                            mode=prims.TMALoadMode.TILE,
                                            group=prims.CTAGroup.CTA_2,
                                        )
                                        _tma_mbar_103 = cutlass.Array((full32_addr + a32_stage[0]).to_llvm_ptr(), shape=1, dtype=cutlass.Uint64)
                                        prims.cp_async_bulk_tensor_shared_cluster_global(
                                            cute.make_ptr(cutlass.Uint8, cutlass.Uint32((sx32_addr + (a32_stage[0] * 512))), mem_space=cute.AddressSpace.smem, assumed_align=16),
                                            SX2_1.get_ptr(),
                                            [cutlass.Int32((pool_block_2 * 128)), cutlass.Int32(k_10)],
                                            _tma_mbar_103,
                                            [],
                                            l2_cache_hint=0x14F0000000000000,
                                            mode=prims.TMALoadMode.TILE,
                                            group=prims.CTAGroup.CTA_2,
                                        )
                                        if (cta_rank == 0):
                                            _mbarrier_cluster_expect_tx_104 = prims.mapa(full32_addr + a32_stage[0], cutlass.Int32(cta_rank & 0xFFFFFFFE))
                                            prims.mbarrier_arrive_expect_tx(_mbarrier_cluster_expect_tx_104, 3072, scope=prims.MemScope.CTA)
                                        else:
                                            _mbarrier_cluster_arrive_105 = prims.mapa(full32_addr + a32_stage[0], cutlass.Int32(cta_rank & 0xFFFFFFFE))
                                            prims.mbarrier_arrive(_mbarrier_cluster_arrive_105, count=cutlass.Int32(1), scope=prims.MemScope.CTA)
                                    _advanced_stage_106 = cutlass.Uint32(a32_stage[0] + 1)
                                    _stage_wrapped_107 = cutlass.Boolean(_advanced_stage_106 == 10)
                                    a32_stage[0] = cutlass.Uint32(cutlass.select_(_stage_wrapped_107, 0, _advanced_stage_106))
                                    _phase_empty32[0] = cutlass.Uint32(cutlass.select_(_stage_wrapped_107, _phase_empty32[0] ^ cutlass.Uint32(1), _phase_empty32[0]))
                            else:
                                for k_11 in cutlass.range(cutlass.Int32(0), cutlass.Int32(k_steps), cutlass.Int32(1)):
                                    while not prims.mbarrier_wait_parity(empty32_addr + a32_stage[0], _phase_empty32[0], prims.MBarrierWait.TRY, scope=prims.MBarrierScope.CTA, order=prims.MemOrder.ACQUIRE):
                                        pass
                                    if prims.elect_sync():
                                        _tma_mbar_108 = cutlass.Array((full32_addr + a32_stage[0]).to_llvm_ptr(), shape=1, dtype=cutlass.Uint64)
                                        prims.cp_async_bulk_tensor_shared_cluster_global(
                                            cute.make_ptr(cutlass.Uint8, cutlass.Uint32((x32_addr + (a32_stage[0] * 2048))), mem_space=cute.AddressSpace.smem, assumed_align=16),
                                            X2.get_ptr(),
                                            [cutlass.Int32((k_11 * 128)), cutlass.Int32(((pool_block_2 * 64) + cutlass.Uint32((cta_rank * 16))))],
                                            _tma_mbar_108,
                                            [],
                                            l2_cache_hint=0x14F0000000000000,
                                            mode=prims.TMALoadMode.TILE,
                                            group=prims.CTAGroup.CTA_2,
                                        )
                                        _tma_mbar_109 = cutlass.Array((full32_addr + a32_stage[0]).to_llvm_ptr(), shape=1, dtype=cutlass.Uint64)
                                        prims.cp_async_bulk_tensor_shared_cluster_global(
                                            cute.make_ptr(cutlass.Uint8, cutlass.Uint32((sx32_addr + (a32_stage[0] * 512))), mem_space=cute.AddressSpace.smem, assumed_align=16),
                                            SX2_1.get_ptr(),
                                            [cutlass.Int32((pool_block_2 * 128)), cutlass.Int32(k_11)],
                                            _tma_mbar_109,
                                            [],
                                            l2_cache_hint=0x14F0000000000000,
                                            mode=prims.TMALoadMode.TILE,
                                            group=prims.CTAGroup.CTA_2,
                                        )
                                        if (cta_rank == 0):
                                            _mbarrier_cluster_expect_tx_110 = prims.mapa(full32_addr + a32_stage[0], cutlass.Int32(cta_rank & 0xFFFFFFFE))
                                            prims.mbarrier_arrive_expect_tx(_mbarrier_cluster_expect_tx_110, 5120, scope=prims.MemScope.CTA)
                                        else:
                                            _mbarrier_cluster_arrive_111 = prims.mapa(full32_addr + a32_stage[0], cutlass.Int32(cta_rank & 0xFFFFFFFE))
                                            prims.mbarrier_arrive(_mbarrier_cluster_arrive_111, count=cutlass.Int32(1), scope=prims.MemScope.CTA)
                                    _advanced_stage_112 = cutlass.Uint32(a32_stage[0] + 1)
                                    _stage_wrapped_113 = cutlass.Boolean(_advanced_stage_112 == 10)
                                    a32_stage[0] = cutlass.Uint32(cutlass.select_(_stage_wrapped_113, 0, _advanced_stage_112))
                                    _phase_empty32[0] = cutlass.Uint32(cutlass.select_(_stage_wrapped_113, _phase_empty32[0] ^ cutlass.Uint32(1), _phase_empty32[0]))
                        else:
                            if (task_n_1 == 16):
                                for k_12 in cutlass.range(cutlass.Int32(0), cutlass.Int32(k_steps), cutlass.Int32(1)):
                                    while not prims.mbarrier_wait_parity(empty64_addr + a64_stage[0], _phase_empty64[0], prims.MBarrierWait.TRY, scope=prims.MBarrierScope.CTA, order=prims.MemOrder.ACQUIRE):
                                        pass
                                    if prims.elect_sync():
                                        _tma_mbar_114 = cutlass.Array((full64_addr + a64_stage[0]).to_llvm_ptr(), shape=1, dtype=cutlass.Uint64)
                                        prims.cp_async_bulk_tensor_shared_cluster_global(
                                            cute.make_ptr(cutlass.Uint8, cutlass.Uint32((x64_addr + (a64_stage[0] * 4096))), mem_space=cute.AddressSpace.smem, assumed_align=16),
                                            X2_8.get_ptr(),
                                            [cutlass.Int32((k_12 * 128)), cutlass.Int32(((pool_block_2 * 64) + cutlass.Uint32((cta_rank * 8))))],
                                            _tma_mbar_114,
                                            [],
                                            l2_cache_hint=0x14F0000000000000,
                                            mode=prims.TMALoadMode.TILE,
                                            group=prims.CTAGroup.CTA_2,
                                        )
                                        _tma_mbar_115 = cutlass.Array((full64_addr + a64_stage[0]).to_llvm_ptr(), shape=1, dtype=cutlass.Uint64)
                                        prims.cp_async_bulk_tensor_shared_cluster_global(
                                            cute.make_ptr(cutlass.Uint8, cutlass.Uint32((sx64_addr + (a64_stage[0] * 512))), mem_space=cute.AddressSpace.smem, assumed_align=16),
                                            SX2_1.get_ptr(),
                                            [cutlass.Int32((pool_block_2 * 128)), cutlass.Int32(k_12)],
                                            _tma_mbar_115,
                                            [],
                                            l2_cache_hint=0x14F0000000000000,
                                            mode=prims.TMALoadMode.TILE,
                                            group=prims.CTAGroup.CTA_2,
                                        )
                                        if (cta_rank == 0):
                                            _mbarrier_cluster_expect_tx_116 = prims.mapa(full64_addr + a64_stage[0], cutlass.Int32(cta_rank & 0xFFFFFFFE))
                                            prims.mbarrier_arrive_expect_tx(_mbarrier_cluster_expect_tx_116, 3072, scope=prims.MemScope.CTA)
                                        else:
                                            _mbarrier_cluster_arrive_117 = prims.mapa(full64_addr + a64_stage[0], cutlass.Int32(cta_rank & 0xFFFFFFFE))
                                            prims.mbarrier_arrive(_mbarrier_cluster_arrive_117, count=cutlass.Int32(1), scope=prims.MemScope.CTA)
                                    _advanced_stage_118 = cutlass.Uint32(a64_stage[0] + 1)
                                    _stage_wrapped_119 = cutlass.Boolean(_advanced_stage_118 == 9)
                                    a64_stage[0] = cutlass.Uint32(cutlass.select_(_stage_wrapped_119, 0, _advanced_stage_118))
                                    _phase_empty64[0] = cutlass.Uint32(cutlass.select_(_stage_wrapped_119, _phase_empty64[0] ^ cutlass.Uint32(1), _phase_empty64[0]))
                            else:
                                if (task_n_1 == 32):
                                    for k_13 in cutlass.range(cutlass.Int32(0), cutlass.Int32(k_steps), cutlass.Int32(1)):
                                        while not prims.mbarrier_wait_parity(empty64_addr + a64_stage[0], _phase_empty64[0], prims.MBarrierWait.TRY, scope=prims.MBarrierScope.CTA, order=prims.MemOrder.ACQUIRE):
                                            pass
                                        if prims.elect_sync():
                                            _tma_mbar_120 = cutlass.Array((full64_addr + a64_stage[0]).to_llvm_ptr(), shape=1, dtype=cutlass.Uint64)
                                            prims.cp_async_bulk_tensor_shared_cluster_global(
                                                cute.make_ptr(cutlass.Uint8, cutlass.Uint32((x64_addr + (a64_stage[0] * 4096))), mem_space=cute.AddressSpace.smem, assumed_align=16),
                                                X2.get_ptr(),
                                                [cutlass.Int32((k_13 * 128)), cutlass.Int32(((pool_block_2 * 64) + cutlass.Uint32((cta_rank * 16))))],
                                                _tma_mbar_120,
                                                [],
                                                l2_cache_hint=0x14F0000000000000,
                                                mode=prims.TMALoadMode.TILE,
                                                group=prims.CTAGroup.CTA_2,
                                            )
                                            _tma_mbar_121 = cutlass.Array((full64_addr + a64_stage[0]).to_llvm_ptr(), shape=1, dtype=cutlass.Uint64)
                                            prims.cp_async_bulk_tensor_shared_cluster_global(
                                                cute.make_ptr(cutlass.Uint8, cutlass.Uint32((sx64_addr + (a64_stage[0] * 512))), mem_space=cute.AddressSpace.smem, assumed_align=16),
                                                SX2_1.get_ptr(),
                                                [cutlass.Int32((pool_block_2 * 128)), cutlass.Int32(k_13)],
                                                _tma_mbar_121,
                                                [],
                                                l2_cache_hint=0x14F0000000000000,
                                                mode=prims.TMALoadMode.TILE,
                                                group=prims.CTAGroup.CTA_2,
                                            )
                                            if (cta_rank == 0):
                                                _mbarrier_cluster_expect_tx_122 = prims.mapa(full64_addr + a64_stage[0], cutlass.Int32(cta_rank & 0xFFFFFFFE))
                                                prims.mbarrier_arrive_expect_tx(_mbarrier_cluster_expect_tx_122, 5120, scope=prims.MemScope.CTA)
                                            else:
                                                _mbarrier_cluster_arrive_123 = prims.mapa(full64_addr + a64_stage[0], cutlass.Int32(cta_rank & 0xFFFFFFFE))
                                                prims.mbarrier_arrive(_mbarrier_cluster_arrive_123, count=cutlass.Int32(1), scope=prims.MemScope.CTA)
                                        _advanced_stage_124 = cutlass.Uint32(a64_stage[0] + 1)
                                        _stage_wrapped_125 = cutlass.Boolean(_advanced_stage_124 == 9)
                                        a64_stage[0] = cutlass.Uint32(cutlass.select_(_stage_wrapped_125, 0, _advanced_stage_124))
                                        _phase_empty64[0] = cutlass.Uint32(cutlass.select_(_stage_wrapped_125, _phase_empty64[0] ^ cutlass.Uint32(1), _phase_empty64[0]))
                                else:
                                    for k_14 in cutlass.range(cutlass.Int32(0), cutlass.Int32(k_steps), cutlass.Int32(1)):
                                        while not prims.mbarrier_wait_parity(empty64_addr + a64_stage[0], _phase_empty64[0], prims.MBarrierWait.TRY, scope=prims.MBarrierScope.CTA, order=prims.MemOrder.ACQUIRE):
                                            pass
                                        if prims.elect_sync():
                                            _tma_mbar_126 = cutlass.Array((full64_addr + a64_stage[0]).to_llvm_ptr(), shape=1, dtype=cutlass.Uint64)
                                            prims.cp_async_bulk_tensor_shared_cluster_global(
                                                cute.make_ptr(cutlass.Uint8, cutlass.Uint32((x64_addr + (a64_stage[0] * 4096))), mem_space=cute.AddressSpace.smem, assumed_align=16),
                                                X2_32.get_ptr(),
                                                [cutlass.Int32((k_14 * 128)), cutlass.Int32(((pool_block_2 * 64) + cutlass.Uint32((cta_rank * 32))))],
                                                _tma_mbar_126,
                                                [],
                                                l2_cache_hint=0x14F0000000000000,
                                                mode=prims.TMALoadMode.TILE,
                                                group=prims.CTAGroup.CTA_2,
                                            )
                                            _tma_mbar_127 = cutlass.Array((full64_addr + a64_stage[0]).to_llvm_ptr(), shape=1, dtype=cutlass.Uint64)
                                            prims.cp_async_bulk_tensor_shared_cluster_global(
                                                cute.make_ptr(cutlass.Uint8, cutlass.Uint32((sx64_addr + (a64_stage[0] * 512))), mem_space=cute.AddressSpace.smem, assumed_align=16),
                                                SX2_1.get_ptr(),
                                                [cutlass.Int32((pool_block_2 * 128)), cutlass.Int32(k_14)],
                                                _tma_mbar_127,
                                                [],
                                                l2_cache_hint=0x14F0000000000000,
                                                mode=prims.TMALoadMode.TILE,
                                                group=prims.CTAGroup.CTA_2,
                                            )
                                            if (cta_rank == 0):
                                                _mbarrier_cluster_expect_tx_128 = prims.mapa(full64_addr + a64_stage[0], cutlass.Int32(cta_rank & 0xFFFFFFFE))
                                                prims.mbarrier_arrive_expect_tx(_mbarrier_cluster_expect_tx_128, 9216, scope=prims.MemScope.CTA)
                                            else:
                                                _mbarrier_cluster_arrive_129 = prims.mapa(full64_addr + a64_stage[0], cutlass.Int32(cta_rank & 0xFFFFFFFE))
                                                prims.mbarrier_arrive(_mbarrier_cluster_arrive_129, count=cutlass.Int32(1), scope=prims.MemScope.CTA)
                                        _advanced_stage_130 = cutlass.Uint32(a64_stage[0] + 1)
                                        _stage_wrapped_131 = cutlass.Boolean(_advanced_stage_130 == 9)
                                        a64_stage[0] = cutlass.Uint32(cutlass.select_(_stage_wrapped_131, 0, _advanced_stage_130))
                                        _phase_empty64[0] = cutlass.Uint32(cutlass.select_(_stage_wrapped_131, _phase_empty64[0] ^ cutlass.Uint32(1), _phase_empty64[0]))
            _advanced_stage_132 = cutlass.Uint32(task_stage_1[0] + 1)
            _stage_wrapped_133 = cutlass.Boolean(_advanced_stage_132 == 2)
            task_stage_1[0] = cutlass.Uint32(cutlass.select_(_stage_wrapped_133, 0, _advanced_stage_132))
            _phase_task_full_1[0] = cutlass.Uint32(cutlass.select_(_stage_wrapped_133, _phase_task_full_1[0] ^ cutlass.Uint32(1), _phase_task_full_1[0]))
    elif warp == 5:
        w16_stage = cute.make_rmem_tensor((1,), cutlass.Uint32)
        w32_stage = cute.make_rmem_tensor((1,), cutlass.Uint32)
        w64_stage = cute.make_rmem_tensor((1,), cutlass.Uint32)
        task_stage_2 = cute.make_rmem_tensor((1,), cutlass.Uint32)
        running_2 = cute.make_rmem_tensor((1,), cutlass.Int32)
        _phase_task_full_2 = cute.make_rmem_tensor((1,), cutlass.Uint32)
        _phase_empty16_1 = cute.make_rmem_tensor((1,), cutlass.Uint32)
        _phase_empty32_1 = cute.make_rmem_tensor((1,), cutlass.Uint32)
        _phase_empty64_1 = cute.make_rmem_tensor((1,), cutlass.Uint32)
        w16_stage[0] = cutlass.Uint32(0)
        w32_stage[0] = cutlass.Uint32(0)
        w64_stage[0] = cutlass.Uint32(0)
        task_stage_2[0] = cutlass.Uint32(0)
        running_2[0] = cutlass.Int32(1)
        _phase_task_full_2[0] = cutlass.Uint32(0)
        _phase_empty16_1[0] = cutlass.Uint32(1)
        _phase_empty32_1[0] = cutlass.Uint32(1)
        _phase_empty64_1[0] = cutlass.Uint32(1)
        while cutlass.Boolean((running_2[0] != 0)):
            while not prims.mbarrier_wait_parity(task_full_addr + task_stage_2[0], _phase_task_full_2[0], prims.MBarrierWait.TRY, scope=prims.MBarrierScope.CTA, order=prims.MemOrder.ACQUIRE):
                pass
            _smem_physical_134 = cute.make_tensor(cute.make_ptr(cutlass.Uint32, cutlass.Uint32(tasks_addr), mem_space=cute.AddressSpace.smem, assumed_align=4), _flat_layout)
            _tasks_reg_1 = cute.make_rmem_tensor((8,), cutlass.Uint32)
            _tasks_reg_1[0] = _smem_physical_134[((task_stage_2[0] * 8)) + 0]
            _tasks_reg_1[1] = _smem_physical_134[((task_stage_2[0] * 8)) + 1]
            _tasks_reg_1[2] = _smem_physical_134[((task_stage_2[0] * 8)) + 2]
            _tasks_reg_1[3] = _smem_physical_134[((task_stage_2[0] * 8)) + 3]
            _tasks_reg_1[4] = _smem_physical_134[((task_stage_2[0] * 8)) + 4]
            _tasks_reg_1[5] = _smem_physical_134[((task_stage_2[0] * 8)) + 5]
            _tasks_reg_1[6] = _smem_physical_134[((task_stage_2[0] * 8)) + 6]
            _tasks_reg_1[7] = _smem_physical_134[((task_stage_2[0] * 8)) + 7]
            if (_tasks_reg_1[0] == 0):
                running_2[0] = cutlass.Int32(0)
            else:
                phase_7 = cutlass.Uint32((_tasks_reg_1[0] - 1))
                tile_2 = cutlass.Uint32(_tasks_reg_1[3])
                expert_4 = cutlass.Uint32(_tasks_reg_1[1])
                logical_block_2 = cutlass.Uint32(_tasks_reg_1[4])
                pool_block_3 = cutlass.Uint32((logical_block_2 % cutlass.Uint32(ring_blocks)))
                ring_epoch_2 = cutlass.Uint32(cutlass.Uint32(cutlass_arith.divui(cutlass.Uint32(logical_block_2).ir_value(), cutlass.Uint32(cutlass.Uint32(ring_blocks)).ir_value())))
                task_valid_2 = cutlass.Uint32(_tasks_reg_1[5])
                task_n_2 = cutlass.Uint32((cutlass.Uint32(cutlass.Uint32(64)) if (task_valid_2 > 32) else cutlass.Uint32((cutlass.Uint32(cutlass_arith.divui(cutlass.Uint32((task_valid_2 + 15)).ir_value(), cutlass.Uint32(16).ir_value())) * 16))))
                layout_n_1 = cutlass.Uint32((_tasks_reg_1[6] >> 16))
                k_steps_1 = cutlass.Uint32(cutlass.Uint32(cutlass_arith.divui(cutlass.Uint32(_tasks_reg_1[7]).ir_value(), cutlass.Uint32(128).ir_value())))
                if (layout_n_1 == 16):
                    for k_15 in cutlass.range(cutlass.Int32(0), cutlass.Int32(cutlass.Uint32(cutlass_arith.divui(cutlass.Uint32(k_steps_1).ir_value(), cutlass.Uint32(2).ir_value()))), cutlass.Int32(1)):
                        while not prims.mbarrier_wait_parity(empty16_addr + w16_stage[0], _phase_empty16_1[0], prims.MBarrierWait.TRY, scope=prims.MBarrierScope.CTA, order=prims.MemOrder.ACQUIRE):
                            pass
                        if prims.elect_sync():
                            if (phase_7 == 0):
                                _tma_mbar_135 = cutlass.Array((full16_addr + w16_stage[0]).to_llvm_ptr(), shape=1, dtype=cutlass.Uint64)
                                prims.cp_async_bulk_tensor_shared_cluster_global(
                                    cute.make_ptr(cutlass.Uint8, cutlass.Uint32((w16_pair_addr + (w16_stage[0] * 32768))), mem_space=cute.AddressSpace.smem, assumed_align=16),
                                    W1_pair.get_ptr(),
                                    [cutlass.Int32(0), cutlass.Int32(((((((expert_4 * 40) + tile_2) * 12) + cutlass.Uint32(k_15)) * 512) + cutlass.Uint32((cta_rank * 256))))],
                                    _tma_mbar_135,
                                    [],
                                    l2_cache_hint=0x12F0000000000000,
                                    mode=prims.TMALoadMode.TILE,
                                    group=prims.CTAGroup.CTA_2,
                                )
                                _tma_mbar_136 = cutlass.Array((full16_addr + w16_stage[0]).to_llvm_ptr(), shape=1, dtype=cutlass.Uint64)
                                prims.cp_async_bulk_tensor_shared_cluster_global(
                                    cute.make_ptr(cutlass.Uint8, cutlass.Uint32((sw16_addr + (w16_stage[0] * 1024))), mem_space=cute.AddressSpace.smem, assumed_align=16),
                                    SW1.get_ptr(),
                                    [cutlass.Int32(((tile_2 * 256) + cutlass.Uint32((cta_rank * 128)))), cutlass.Int32(((expert_4 * 24) + cutlass.Uint32((k_15 * 2))))],
                                    _tma_mbar_136,
                                    [],
                                    l2_cache_hint=0x12F0000000000000,
                                    mode=prims.TMALoadMode.TILE,
                                    group=prims.CTAGroup.CTA_2,
                                )
                            else:
                                _tma_mbar_137 = cutlass.Array((full16_addr + w16_stage[0]).to_llvm_ptr(), shape=1, dtype=cutlass.Uint64)
                                prims.cp_async_bulk_tensor_shared_cluster_global(
                                    cute.make_ptr(cutlass.Uint8, cutlass.Uint32((w16_pair_addr + (w16_stage[0] * 32768))), mem_space=cute.AddressSpace.smem, assumed_align=16),
                                    W2_pair.get_ptr(),
                                    [cutlass.Int32(0), cutlass.Int32(((((((expert_4 * 12) + tile_2) * 20) + cutlass.Uint32(k_15)) * 512) + cutlass.Uint32((cta_rank * 256))))],
                                    _tma_mbar_137,
                                    [],
                                    l2_cache_hint=0x12F0000000000000,
                                    mode=prims.TMALoadMode.TILE,
                                    group=prims.CTAGroup.CTA_2,
                                )
                                _tma_mbar_138 = cutlass.Array((full16_addr + w16_stage[0]).to_llvm_ptr(), shape=1, dtype=cutlass.Uint64)
                                prims.cp_async_bulk_tensor_shared_cluster_global(
                                    cute.make_ptr(cutlass.Uint8, cutlass.Uint32((sw16_addr + (w16_stage[0] * 1024))), mem_space=cute.AddressSpace.smem, assumed_align=16),
                                    SW2.get_ptr(),
                                    [cutlass.Int32(((tile_2 * 256) + cutlass.Uint32((cta_rank * 128)))), cutlass.Int32(((expert_4 * 40) + cutlass.Uint32((k_15 * 2))))],
                                    _tma_mbar_138,
                                    [],
                                    l2_cache_hint=0x12F0000000000000,
                                    mode=prims.TMALoadMode.TILE,
                                    group=prims.CTAGroup.CTA_2,
                                )
                            if (cta_rank == 0):
                                _mbarrier_cluster_expect_tx_139 = prims.mapa(full16_addr + w16_stage[0], cutlass.Int32(cta_rank & 0xFFFFFFFE))
                                prims.mbarrier_arrive_expect_tx(_mbarrier_cluster_expect_tx_139, 34816, scope=prims.MemScope.CTA)
                            else:
                                _mbarrier_cluster_arrive_140 = prims.mapa(full16_addr + w16_stage[0], cutlass.Int32(cta_rank & 0xFFFFFFFE))
                                prims.mbarrier_arrive(_mbarrier_cluster_arrive_140, count=cutlass.Int32(1), scope=prims.MemScope.CTA)
                        _advanced_stage_141 = cutlass.Uint32(w16_stage[0] + 1)
                        _stage_wrapped_142 = cutlass.Boolean(_advanced_stage_141 == 5)
                        w16_stage[0] = cutlass.Uint32(cutlass.select_(_stage_wrapped_142, 0, _advanced_stage_141))
                        _phase_empty16_1[0] = cutlass.Uint32(cutlass.select_(_stage_wrapped_142, _phase_empty16_1[0] ^ cutlass.Uint32(1), _phase_empty16_1[0]))
                else:
                    if (layout_n_1 == 32):
                        for k_16 in cutlass.range(cutlass.Int32(0), cutlass.Int32(k_steps_1), cutlass.Int32(1)):
                            while not prims.mbarrier_wait_parity(empty32_addr + w32_stage[0], _phase_empty32_1[0], prims.MBarrierWait.TRY, scope=prims.MBarrierScope.CTA, order=prims.MemOrder.ACQUIRE):
                                pass
                            if prims.elect_sync():
                                if (phase_7 == 0):
                                    _tma_mbar_143 = cutlass.Array((full32_addr + w32_stage[0]).to_llvm_ptr(), shape=1, dtype=cutlass.Uint64)
                                    prims.cp_async_bulk_tensor_shared_cluster_global(
                                        cute.make_ptr(cutlass.Uint8, cutlass.Uint32((w32_addr + (w32_stage[0] * 16384))), mem_space=cute.AddressSpace.smem, assumed_align=16),
                                        W1.get_ptr(),
                                        [cutlass.Int32(0), cutlass.Int32((((((((expert_4 * 40) + tile_2) * 12) + cutlass.Uint32(cutlass.Int32(cutlass_arith.divsi(cutlass.Int32(k_16).ir_value(), cutlass.Int32(2).ir_value())))) * 512) + cutlass.Uint32((cta_rank * 256))) + cutlass.Uint32(((k_16 % 2) * 128))))],
                                        _tma_mbar_143,
                                        [],
                                        l2_cache_hint=0x12F0000000000000,
                                        mode=prims.TMALoadMode.TILE,
                                        group=prims.CTAGroup.CTA_2,
                                    )
                                    _tma_mbar_144 = cutlass.Array((full32_addr + w32_stage[0]).to_llvm_ptr(), shape=1, dtype=cutlass.Uint64)
                                    prims.cp_async_bulk_tensor_shared_cluster_global(
                                        cute.make_ptr(cutlass.Uint8, cutlass.Uint32((sw32_addr + (w32_stage[0] * 512))), mem_space=cute.AddressSpace.smem, assumed_align=16),
                                        SW1_1.get_ptr(),
                                        [cutlass.Int32(((tile_2 * 256) + cutlass.Uint32((cta_rank * 128)))), cutlass.Int32(((expert_4 * 24) + cutlass.Uint32(k_16)))],
                                        _tma_mbar_144,
                                        [],
                                        l2_cache_hint=0x12F0000000000000,
                                        mode=prims.TMALoadMode.TILE,
                                        group=prims.CTAGroup.CTA_2,
                                    )
                                else:
                                    _tma_mbar_145 = cutlass.Array((full32_addr + w32_stage[0]).to_llvm_ptr(), shape=1, dtype=cutlass.Uint64)
                                    prims.cp_async_bulk_tensor_shared_cluster_global(
                                        cute.make_ptr(cutlass.Uint8, cutlass.Uint32((w32_addr + (w32_stage[0] * 16384))), mem_space=cute.AddressSpace.smem, assumed_align=16),
                                        W2.get_ptr(),
                                        [cutlass.Int32(0), cutlass.Int32((((((((expert_4 * 12) + tile_2) * 20) + cutlass.Uint32(cutlass.Int32(cutlass_arith.divsi(cutlass.Int32(k_16).ir_value(), cutlass.Int32(2).ir_value())))) * 512) + cutlass.Uint32((cta_rank * 256))) + cutlass.Uint32(((k_16 % 2) * 128))))],
                                        _tma_mbar_145,
                                        [],
                                        l2_cache_hint=0x12F0000000000000,
                                        mode=prims.TMALoadMode.TILE,
                                        group=prims.CTAGroup.CTA_2,
                                    )
                                    _tma_mbar_146 = cutlass.Array((full32_addr + w32_stage[0]).to_llvm_ptr(), shape=1, dtype=cutlass.Uint64)
                                    prims.cp_async_bulk_tensor_shared_cluster_global(
                                        cute.make_ptr(cutlass.Uint8, cutlass.Uint32((sw32_addr + (w32_stage[0] * 512))), mem_space=cute.AddressSpace.smem, assumed_align=16),
                                        SW2_1.get_ptr(),
                                        [cutlass.Int32(((tile_2 * 256) + cutlass.Uint32((cta_rank * 128)))), cutlass.Int32(((expert_4 * 40) + cutlass.Uint32(k_16)))],
                                        _tma_mbar_146,
                                        [],
                                        l2_cache_hint=0x12F0000000000000,
                                        mode=prims.TMALoadMode.TILE,
                                        group=prims.CTAGroup.CTA_2,
                                    )
                                if (cta_rank == 0):
                                    _mbarrier_cluster_expect_tx_147 = prims.mapa(full32_addr + w32_stage[0], cutlass.Int32(cta_rank & 0xFFFFFFFE))
                                    prims.mbarrier_arrive_expect_tx(_mbarrier_cluster_expect_tx_147, 17408, scope=prims.MemScope.CTA)
                                else:
                                    _mbarrier_cluster_arrive_148 = prims.mapa(full32_addr + w32_stage[0], cutlass.Int32(cta_rank & 0xFFFFFFFE))
                                    prims.mbarrier_arrive(_mbarrier_cluster_arrive_148, count=cutlass.Int32(1), scope=prims.MemScope.CTA)
                            _advanced_stage_149 = cutlass.Uint32(w32_stage[0] + 1)
                            _stage_wrapped_150 = cutlass.Boolean(_advanced_stage_149 == 10)
                            w32_stage[0] = cutlass.Uint32(cutlass.select_(_stage_wrapped_150, 0, _advanced_stage_149))
                            _phase_empty32_1[0] = cutlass.Uint32(cutlass.select_(_stage_wrapped_150, _phase_empty32_1[0] ^ cutlass.Uint32(1), _phase_empty32_1[0]))
                    else:
                        for k_17 in cutlass.range(cutlass.Int32(0), cutlass.Int32(k_steps_1), cutlass.Int32(1)):
                            while not prims.mbarrier_wait_parity(empty64_addr + w64_stage[0], _phase_empty64_1[0], prims.MBarrierWait.TRY, scope=prims.MBarrierScope.CTA, order=prims.MemOrder.ACQUIRE):
                                pass
                            if prims.elect_sync():
                                if (phase_7 == 0):
                                    _tma_mbar_151 = cutlass.Array((full64_addr + w64_stage[0]).to_llvm_ptr(), shape=1, dtype=cutlass.Uint64)
                                    prims.cp_async_bulk_tensor_shared_cluster_global(
                                        cute.make_ptr(cutlass.Uint8, cutlass.Uint32((w64_addr + (w64_stage[0] * 16384))), mem_space=cute.AddressSpace.smem, assumed_align=16),
                                        W1.get_ptr(),
                                        [cutlass.Int32(0), cutlass.Int32((((((((expert_4 * 40) + tile_2) * 12) + cutlass.Uint32(cutlass.Int32(cutlass_arith.divsi(cutlass.Int32(k_17).ir_value(), cutlass.Int32(2).ir_value())))) * 512) + cutlass.Uint32((cta_rank * 256))) + cutlass.Uint32(((k_17 % 2) * 128))))],
                                        _tma_mbar_151,
                                        [],
                                        l2_cache_hint=0x12F0000000000000,
                                        mode=prims.TMALoadMode.TILE,
                                        group=prims.CTAGroup.CTA_2,
                                    )
                                    _tma_mbar_152 = cutlass.Array((full64_addr + w64_stage[0]).to_llvm_ptr(), shape=1, dtype=cutlass.Uint64)
                                    prims.cp_async_bulk_tensor_shared_cluster_global(
                                        cute.make_ptr(cutlass.Uint8, cutlass.Uint32((sw64_addr + (w64_stage[0] * 512))), mem_space=cute.AddressSpace.smem, assumed_align=16),
                                        SW1_1.get_ptr(),
                                        [cutlass.Int32(((tile_2 * 256) + cutlass.Uint32((cta_rank * 128)))), cutlass.Int32(((expert_4 * 24) + cutlass.Uint32(k_17)))],
                                        _tma_mbar_152,
                                        [],
                                        l2_cache_hint=0x12F0000000000000,
                                        mode=prims.TMALoadMode.TILE,
                                        group=prims.CTAGroup.CTA_2,
                                    )
                                else:
                                    _tma_mbar_153 = cutlass.Array((full64_addr + w64_stage[0]).to_llvm_ptr(), shape=1, dtype=cutlass.Uint64)
                                    prims.cp_async_bulk_tensor_shared_cluster_global(
                                        cute.make_ptr(cutlass.Uint8, cutlass.Uint32((w64_addr + (w64_stage[0] * 16384))), mem_space=cute.AddressSpace.smem, assumed_align=16),
                                        W2.get_ptr(),
                                        [cutlass.Int32(0), cutlass.Int32((((((((expert_4 * 12) + tile_2) * 20) + cutlass.Uint32(cutlass.Int32(cutlass_arith.divsi(cutlass.Int32(k_17).ir_value(), cutlass.Int32(2).ir_value())))) * 512) + cutlass.Uint32((cta_rank * 256))) + cutlass.Uint32(((k_17 % 2) * 128))))],
                                        _tma_mbar_153,
                                        [],
                                        l2_cache_hint=0x12F0000000000000,
                                        mode=prims.TMALoadMode.TILE,
                                        group=prims.CTAGroup.CTA_2,
                                    )
                                    _tma_mbar_154 = cutlass.Array((full64_addr + w64_stage[0]).to_llvm_ptr(), shape=1, dtype=cutlass.Uint64)
                                    prims.cp_async_bulk_tensor_shared_cluster_global(
                                        cute.make_ptr(cutlass.Uint8, cutlass.Uint32((sw64_addr + (w64_stage[0] * 512))), mem_space=cute.AddressSpace.smem, assumed_align=16),
                                        SW2_1.get_ptr(),
                                        [cutlass.Int32(((tile_2 * 256) + cutlass.Uint32((cta_rank * 128)))), cutlass.Int32(((expert_4 * 40) + cutlass.Uint32(k_17)))],
                                        _tma_mbar_154,
                                        [],
                                        l2_cache_hint=0x12F0000000000000,
                                        mode=prims.TMALoadMode.TILE,
                                        group=prims.CTAGroup.CTA_2,
                                    )
                                if (cta_rank == 0):
                                    _mbarrier_cluster_expect_tx_155 = prims.mapa(full64_addr + w64_stage[0], cutlass.Int32(cta_rank & 0xFFFFFFFE))
                                    prims.mbarrier_arrive_expect_tx(_mbarrier_cluster_expect_tx_155, 17408, scope=prims.MemScope.CTA)
                                else:
                                    _mbarrier_cluster_arrive_156 = prims.mapa(full64_addr + w64_stage[0], cutlass.Int32(cta_rank & 0xFFFFFFFE))
                                    prims.mbarrier_arrive(_mbarrier_cluster_arrive_156, count=cutlass.Int32(1), scope=prims.MemScope.CTA)
                            _advanced_stage_157 = cutlass.Uint32(w64_stage[0] + 1)
                            _stage_wrapped_158 = cutlass.Boolean(_advanced_stage_157 == 9)
                            w64_stage[0] = cutlass.Uint32(cutlass.select_(_stage_wrapped_158, 0, _advanced_stage_157))
                            _phase_empty64_1[0] = cutlass.Uint32(cutlass.select_(_stage_wrapped_158, _phase_empty64_1[0] ^ cutlass.Uint32(1), _phase_empty64_1[0]))
            _advanced_stage_159 = cutlass.Uint32(task_stage_2[0] + 1)
            _stage_wrapped_160 = cutlass.Boolean(_advanced_stage_159 == 2)
            task_stage_2[0] = cutlass.Uint32(cutlass.select_(_stage_wrapped_160, 0, _advanced_stage_159))
            _phase_task_full_2[0] = cutlass.Uint32(cutlass.select_(_stage_wrapped_160, _phase_task_full_2[0] ^ cutlass.Uint32(1), _phase_task_full_2[0]))
    elif warp == 6:
        m16_stage = cute.make_rmem_tensor((1,), cutlass.Uint32)
        m32_stage = cute.make_rmem_tensor((1,), cutlass.Uint32)
        m64_stage = cute.make_rmem_tensor((1,), cutlass.Uint32)
        acc_stage = cute.make_rmem_tensor((1,), cutlass.Uint32)
        _phase_task_full_3 = cute.make_rmem_tensor((1,), cutlass.Uint32)
        _phase_released = cute.make_rmem_tensor((1,), cutlass.Uint32)
        _phase_full16 = cute.make_rmem_tensor((1,), cutlass.Uint32)
        _phase_full32 = cute.make_rmem_tensor((1,), cutlass.Uint32)
        _phase_full64 = cute.make_rmem_tensor((1,), cutlass.Uint32)
        task_stage_3 = cute.make_rmem_tensor((1,), cutlass.Uint32)
        running_3 = cute.make_rmem_tensor((1,), cutlass.Int32)
        m16_stage[0] = cutlass.Uint32(0)
        m32_stage[0] = cutlass.Uint32(0)
        m64_stage[0] = cutlass.Uint32(0)
        acc_stage[0] = cutlass.Uint32(0)
        _phase_task_full_3[0] = cutlass.Uint32(0)
        _phase_released[0] = cutlass.Uint32(1)
        _phase_full16[0] = cutlass.Uint32(0)
        _phase_full32[0] = cutlass.Uint32(0)
        _phase_full64[0] = cutlass.Uint32(0)
        if cta_rank == 0:
            task_stage_3[0] = cutlass.Uint32(0)
            running_3[0] = cutlass.Int32(1)
            while cutlass.Boolean((running_3[0] != 0)):
                while not prims.mbarrier_wait_parity(task_full_addr + task_stage_3[0], _phase_task_full_3[0], prims.MBarrierWait.TRY, scope=prims.MBarrierScope.CTA, order=prims.MemOrder.ACQUIRE):
                    pass
                _smem_physical_161 = cute.make_tensor(cute.make_ptr(cutlass.Uint32, cutlass.Uint32(tasks_addr), mem_space=cute.AddressSpace.smem, assumed_align=4), _flat_layout)
                _tasks_reg_2 = cute.make_rmem_tensor((8,), cutlass.Uint32)
                _tasks_reg_2[0] = _smem_physical_161[((task_stage_3[0] * 8)) + 0]
                _tasks_reg_2[1] = _smem_physical_161[((task_stage_3[0] * 8)) + 1]
                _tasks_reg_2[2] = _smem_physical_161[((task_stage_3[0] * 8)) + 2]
                _tasks_reg_2[3] = _smem_physical_161[((task_stage_3[0] * 8)) + 3]
                _tasks_reg_2[4] = _smem_physical_161[((task_stage_3[0] * 8)) + 4]
                _tasks_reg_2[5] = _smem_physical_161[((task_stage_3[0] * 8)) + 5]
                _tasks_reg_2[6] = _smem_physical_161[((task_stage_3[0] * 8)) + 6]
                _tasks_reg_2[7] = _smem_physical_161[((task_stage_3[0] * 8)) + 7]
                if (_tasks_reg_2[0] == 0):
                    running_3[0] = cutlass.Int32(0)
                else:
                    phase_8 = cutlass.Uint32((_tasks_reg_2[0] - 1))
                    tile_3 = cutlass.Uint32(_tasks_reg_2[3])
                    expert_5 = cutlass.Uint32(_tasks_reg_2[1])
                    logical_block_3 = cutlass.Uint32(_tasks_reg_2[4])
                    pool_block_4 = cutlass.Uint32((logical_block_3 % cutlass.Uint32(ring_blocks)))
                    ring_epoch_3 = cutlass.Uint32(cutlass.Uint32(cutlass_arith.divui(cutlass.Uint32(logical_block_3).ir_value(), cutlass.Uint32(cutlass.Uint32(ring_blocks)).ir_value())))
                    task_valid_3 = cutlass.Uint32(_tasks_reg_2[5])
                    task_n_3 = cutlass.Uint32((cutlass.Uint32(cutlass.Uint32(64)) if (task_valid_3 > 32) else cutlass.Uint32((cutlass.Uint32(cutlass_arith.divui(cutlass.Uint32((task_valid_3 + 15)).ir_value(), cutlass.Uint32(16).ir_value())) * 16))))
                    layout_n_2 = cutlass.Uint32((_tasks_reg_2[6] >> 16))
                    k_steps_2 = cutlass.Uint32(cutlass.Uint32(cutlass_arith.divui(cutlass.Uint32(_tasks_reg_2[7]).ir_value(), cutlass.Uint32(128).ir_value())))
                    while not prims.mbarrier_wait_parity(released_addr + acc_stage[0], _phase_released[0], prims.MBarrierWait.TRY, scope=prims.MBarrierScope.CTA, order=prims.MemOrder.ACQUIRE):
                        pass
                    if (layout_n_2 == 16):
                        for k_18 in cutlass.range(cutlass.Int32(0), cutlass.Int32(cutlass.Uint32(cutlass_arith.divui(cutlass.Uint32(k_steps_2).ir_value(), cutlass.Uint32(2).ir_value()))), cutlass.Int32(1)):
                            while not prims.mbarrier_wait_parity(full16_addr + m16_stage[0], _phase_full16[0], prims.MBarrierWait.TRY, scope=prims.MBarrierScope.CTA, order=prims.MemOrder.ACQUIRE):
                                pass
                            prims.tcgen05_fence('after_thread_sync')
                            if prims.elect_sync():
                                _scale_copy_162_src_0 = cutlass.Uint64(cutlass.Uint32((sw16_addr + (m16_stage[0] * 1024))))
                                _scale_copy_162_desc_0 = ((_scale_copy_162_src_0 >> 4) & cutlass.Uint64(0x3FFF)) | cutlass.Uint64(0x400800000000)
                                _scale_copy_162_dst_0 = prims.make_tmem_ptr(cutlass.Int32(tmem_sfw), cutlass.Int32)
                                prims.tcgen05_cp(prims.Tcgen05CpShape.SHAPE_32X128B, _scale_copy_162_dst_0, _scale_copy_162_desc_0, group=prims.CTAGroup.CTA_2, multicast=prims.Tcgen05CpMulticast.WARPX4)
                                _scale_copy_163_src_0 = cutlass.Uint64(cutlass.Uint32((sx16_addr + (m16_stage[0] * 1024))))
                                _scale_copy_163_desc_0 = ((_scale_copy_163_src_0 >> 4) & cutlass.Uint64(0x3FFF)) | cutlass.Uint64(0x400800000000)
                                _scale_copy_163_dst_0 = prims.make_tmem_ptr(cutlass.Int32(tmem_sfx), cutlass.Int32)
                                prims.tcgen05_cp(prims.Tcgen05CpShape.SHAPE_32X128B, _scale_copy_163_dst_0, _scale_copy_163_desc_0, group=prims.CTAGroup.CTA_2, multicast=prims.Tcgen05CpMulticast.WARPX4)
                                _mma_a_lo_0 = cutlass.Int32((((w16_addr >> 4) & 0x3FFF) + m16_stage[0] * 2048))
                                _mma_b_lo_0 = cutlass.Int32((((x16_addr >> 4) & 0x3FFF) + m16_stage[0] * 128))
                                _mma_mxf8f6f4_164_d = prims.make_tmem_ptr(cutlass.Int32((tmem_acc + (acc_stage[0] * 64))), cutlass.Int32)
                                _mma_mxf8f6f4_164_sfa = prims.make_tmem_ptr(cutlass.Int32(tmem_sfw), cutlass.Int32)
                                _mma_mxf8f6f4_164_sfb = prims.make_tmem_ptr(cutlass.Int32(tmem_sfx), cutlass.Int32)
                                _mma_mxf8f6f4_164_a = cutlass.Uint64(cutlass.Uint32(_mma_a_lo_0)) | (cutlass.Uint64(0x40004040) << 32)
                                _mma_mxf8f6f4_164_b = cutlass.Uint64(cutlass.Uint32(_mma_b_lo_0)) | (cutlass.Uint64(0x40004040) << 32)
                                prims.tcgen05_mma_block_scale(
                                    prims.Tcgen05MMAKind.MXF8F6F4, prims.CTAGroup.CTA_2,
                                    _mma_mxf8f6f4_164_d, _mma_mxf8f6f4_164_a + 0, _mma_mxf8f6f4_164_b + 0,
                                    (cutlass.Uint32(0x10800280) | ((cutlass.Uint32(task_n_3) >> 3) << 17)), cutlass.Boolean(((k_18 == 0)) == 0), _mma_mxf8f6f4_164_sfa, _mma_mxf8f6f4_164_sfb,
                                    scale_vec_size=prims.Tcgen05MMAScaleVecSize.X1,
                                )
                                prims.tcgen05_mma_block_scale(
                                    prims.Tcgen05MMAKind.MXF8F6F4, prims.CTAGroup.CTA_2,
                                    _mma_mxf8f6f4_164_d, _mma_mxf8f6f4_164_a + 2, _mma_mxf8f6f4_164_b + 2,
                                    (cutlass.Uint32(0x30800290) | ((cutlass.Uint32(task_n_3) >> 3) << 17)), True, _mma_mxf8f6f4_164_sfa, _mma_mxf8f6f4_164_sfb,
                                    scale_vec_size=prims.Tcgen05MMAScaleVecSize.X1,
                                )
                                prims.tcgen05_mma_block_scale(
                                    prims.Tcgen05MMAKind.MXF8F6F4, prims.CTAGroup.CTA_2,
                                    _mma_mxf8f6f4_164_d, _mma_mxf8f6f4_164_a + 4, _mma_mxf8f6f4_164_b + 4,
                                    (cutlass.Uint32(0x508002A0) | ((cutlass.Uint32(task_n_3) >> 3) << 17)), True, _mma_mxf8f6f4_164_sfa, _mma_mxf8f6f4_164_sfb,
                                    scale_vec_size=prims.Tcgen05MMAScaleVecSize.X1,
                                )
                                prims.tcgen05_mma_block_scale(
                                    prims.Tcgen05MMAKind.MXF8F6F4, prims.CTAGroup.CTA_2,
                                    _mma_mxf8f6f4_164_d, _mma_mxf8f6f4_164_a + 6, _mma_mxf8f6f4_164_b + 6,
                                    (cutlass.Uint32(0x708002B0) | ((cutlass.Uint32(task_n_3) >> 3) << 17)), True, _mma_mxf8f6f4_164_sfa, _mma_mxf8f6f4_164_sfb,
                                    scale_vec_size=prims.Tcgen05MMAScaleVecSize.X1,
                                )
                                _scale_copy_165_src_0 = cutlass.Uint64(cutlass.Uint32((sw16_hi_addr + (m16_stage[0] * 1024))))
                                _scale_copy_165_desc_0 = ((_scale_copy_165_src_0 >> 4) & cutlass.Uint64(0x3FFF)) | cutlass.Uint64(0x400800000000)
                                _scale_copy_165_dst_0 = prims.make_tmem_ptr(cutlass.Int32(tmem_sfw), cutlass.Int32)
                                prims.tcgen05_cp(prims.Tcgen05CpShape.SHAPE_32X128B, _scale_copy_165_dst_0, _scale_copy_165_desc_0, group=prims.CTAGroup.CTA_2, multicast=prims.Tcgen05CpMulticast.WARPX4)
                                _scale_copy_166_src_0 = cutlass.Uint64(cutlass.Uint32((sx16_hi_addr + (m16_stage[0] * 1024))))
                                _scale_copy_166_desc_0 = ((_scale_copy_166_src_0 >> 4) & cutlass.Uint64(0x3FFF)) | cutlass.Uint64(0x400800000000)
                                _scale_copy_166_dst_0 = prims.make_tmem_ptr(cutlass.Int32(tmem_sfx), cutlass.Int32)
                                prims.tcgen05_cp(prims.Tcgen05CpShape.SHAPE_32X128B, _scale_copy_166_dst_0, _scale_copy_166_desc_0, group=prims.CTAGroup.CTA_2, multicast=prims.Tcgen05CpMulticast.WARPX4)
                                _mma_a_lo_1 = cutlass.Int32((((w16_hi_addr >> 4) & 0x3FFF) + m16_stage[0] * 2048))
                                _mma_b_lo_1 = cutlass.Int32((((x16_hi_addr >> 4) & 0x3FFF) + m16_stage[0] * 128))
                                _mma_mxf8f6f4_167_d = prims.make_tmem_ptr(cutlass.Int32((tmem_acc + (acc_stage[0] * 64))), cutlass.Int32)
                                _mma_mxf8f6f4_167_sfa = prims.make_tmem_ptr(cutlass.Int32(tmem_sfw), cutlass.Int32)
                                _mma_mxf8f6f4_167_sfb = prims.make_tmem_ptr(cutlass.Int32(tmem_sfx), cutlass.Int32)
                                _mma_mxf8f6f4_167_a = cutlass.Uint64(cutlass.Uint32(_mma_a_lo_1)) | (cutlass.Uint64(0x40004040) << 32)
                                _mma_mxf8f6f4_167_b = cutlass.Uint64(cutlass.Uint32(_mma_b_lo_1)) | (cutlass.Uint64(0x40004040) << 32)
                                prims.tcgen05_mma_block_scale(
                                    prims.Tcgen05MMAKind.MXF8F6F4, prims.CTAGroup.CTA_2,
                                    _mma_mxf8f6f4_167_d, _mma_mxf8f6f4_167_a + 0, _mma_mxf8f6f4_167_b + 0,
                                    (cutlass.Uint32(0x10800280) | ((cutlass.Uint32(task_n_3) >> 3) << 17)), cutlass.Boolean((0) == 0), _mma_mxf8f6f4_167_sfa, _mma_mxf8f6f4_167_sfb,
                                    scale_vec_size=prims.Tcgen05MMAScaleVecSize.X1,
                                )
                                prims.tcgen05_mma_block_scale(
                                    prims.Tcgen05MMAKind.MXF8F6F4, prims.CTAGroup.CTA_2,
                                    _mma_mxf8f6f4_167_d, _mma_mxf8f6f4_167_a + 2, _mma_mxf8f6f4_167_b + 2,
                                    (cutlass.Uint32(0x30800290) | ((cutlass.Uint32(task_n_3) >> 3) << 17)), True, _mma_mxf8f6f4_167_sfa, _mma_mxf8f6f4_167_sfb,
                                    scale_vec_size=prims.Tcgen05MMAScaleVecSize.X1,
                                )
                                prims.tcgen05_mma_block_scale(
                                    prims.Tcgen05MMAKind.MXF8F6F4, prims.CTAGroup.CTA_2,
                                    _mma_mxf8f6f4_167_d, _mma_mxf8f6f4_167_a + 4, _mma_mxf8f6f4_167_b + 4,
                                    (cutlass.Uint32(0x508002A0) | ((cutlass.Uint32(task_n_3) >> 3) << 17)), True, _mma_mxf8f6f4_167_sfa, _mma_mxf8f6f4_167_sfb,
                                    scale_vec_size=prims.Tcgen05MMAScaleVecSize.X1,
                                )
                                prims.tcgen05_mma_block_scale(
                                    prims.Tcgen05MMAKind.MXF8F6F4, prims.CTAGroup.CTA_2,
                                    _mma_mxf8f6f4_167_d, _mma_mxf8f6f4_167_a + 6, _mma_mxf8f6f4_167_b + 6,
                                    (cutlass.Uint32(0x708002B0) | ((cutlass.Uint32(task_n_3) >> 3) << 17)), True, _mma_mxf8f6f4_167_sfa, _mma_mxf8f6f4_167_sfb,
                                    scale_vec_size=prims.Tcgen05MMAScaleVecSize.X1,
                                )
                            with cute.arch.elect_one():
                                cute.nvgpu.tcgen05.commit(empty16_addr + m16_stage[0], 3, cute.nvgpu.tcgen05.CtaGroup.TWO)
                            _advanced_stage_168 = cutlass.Uint32(m16_stage[0] + 1)
                            _stage_wrapped_169 = cutlass.Boolean(_advanced_stage_168 == 5)
                            m16_stage[0] = cutlass.Uint32(cutlass.select_(_stage_wrapped_169, 0, _advanced_stage_168))
                            _phase_full16[0] = cutlass.Uint32(cutlass.select_(_stage_wrapped_169, _phase_full16[0] ^ cutlass.Uint32(1), _phase_full16[0]))
                    else:
                        if (layout_n_2 == 32):
                            for k_19 in cutlass.range(cutlass.Int32(0), cutlass.Int32(k_steps_2), cutlass.Int32(1)):
                                while not prims.mbarrier_wait_parity(full32_addr + m32_stage[0], _phase_full32[0], prims.MBarrierWait.TRY, scope=prims.MBarrierScope.CTA, order=prims.MemOrder.ACQUIRE):
                                    pass
                                prims.tcgen05_fence('after_thread_sync')
                                if prims.elect_sync():
                                    _scale_copy_170_src_0 = cutlass.Uint64(cutlass.Uint32((sw32_addr + (m32_stage[0] * 512))))
                                    _scale_copy_170_desc_0 = ((_scale_copy_170_src_0 >> 4) & cutlass.Uint64(0x3FFF)) | cutlass.Uint64(0x400800000000)
                                    _scale_copy_170_dst_0 = prims.make_tmem_ptr(cutlass.Int32(tmem_sfw), cutlass.Int32)
                                    prims.tcgen05_cp(prims.Tcgen05CpShape.SHAPE_32X128B, _scale_copy_170_dst_0, _scale_copy_170_desc_0, group=prims.CTAGroup.CTA_2, multicast=prims.Tcgen05CpMulticast.WARPX4)
                                    _scale_copy_171_src_0 = cutlass.Uint64(cutlass.Uint32((sx32_addr + (m32_stage[0] * 512))))
                                    _scale_copy_171_desc_0 = ((_scale_copy_171_src_0 >> 4) & cutlass.Uint64(0x3FFF)) | cutlass.Uint64(0x400800000000)
                                    _scale_copy_171_dst_0 = prims.make_tmem_ptr(cutlass.Int32(tmem_sfx), cutlass.Int32)
                                    prims.tcgen05_cp(prims.Tcgen05CpShape.SHAPE_32X128B, _scale_copy_171_dst_0, _scale_copy_171_desc_0, group=prims.CTAGroup.CTA_2, multicast=prims.Tcgen05CpMulticast.WARPX4)
                                    _mma_a_lo_2 = cutlass.Int32((((w32_addr >> 4) & 0x3FFF) + m32_stage[0] * 1024))
                                    _mma_b_lo_2 = cutlass.Int32((((x32_addr >> 4) & 0x3FFF) + m32_stage[0] * 128))
                                    _mma_mxf8f6f4_172_d = prims.make_tmem_ptr(cutlass.Int32((tmem_acc + (acc_stage[0] * 64))), cutlass.Int32)
                                    _mma_mxf8f6f4_172_sfa = prims.make_tmem_ptr(cutlass.Int32(tmem_sfw), cutlass.Int32)
                                    _mma_mxf8f6f4_172_sfb = prims.make_tmem_ptr(cutlass.Int32(tmem_sfx), cutlass.Int32)
                                    _mma_mxf8f6f4_172_a = cutlass.Uint64(cutlass.Uint32(_mma_a_lo_2)) | (cutlass.Uint64(0x40004040) << 32)
                                    _mma_mxf8f6f4_172_b = cutlass.Uint64(cutlass.Uint32(_mma_b_lo_2)) | (cutlass.Uint64(0x40004040) << 32)
                                    prims.tcgen05_mma_block_scale(
                                        prims.Tcgen05MMAKind.MXF8F6F4, prims.CTAGroup.CTA_2,
                                        _mma_mxf8f6f4_172_d, _mma_mxf8f6f4_172_a + 0, _mma_mxf8f6f4_172_b + 0,
                                        (cutlass.Uint32(0x10800280) | ((cutlass.Uint32(task_n_3) >> 3) << 17)), cutlass.Boolean(((k_19 == 0)) == 0), _mma_mxf8f6f4_172_sfa, _mma_mxf8f6f4_172_sfb,
                                        scale_vec_size=prims.Tcgen05MMAScaleVecSize.X1,
                                    )
                                    prims.tcgen05_mma_block_scale(
                                        prims.Tcgen05MMAKind.MXF8F6F4, prims.CTAGroup.CTA_2,
                                        _mma_mxf8f6f4_172_d, _mma_mxf8f6f4_172_a + 2, _mma_mxf8f6f4_172_b + 2,
                                        (cutlass.Uint32(0x30800290) | ((cutlass.Uint32(task_n_3) >> 3) << 17)), True, _mma_mxf8f6f4_172_sfa, _mma_mxf8f6f4_172_sfb,
                                        scale_vec_size=prims.Tcgen05MMAScaleVecSize.X1,
                                    )
                                    prims.tcgen05_mma_block_scale(
                                        prims.Tcgen05MMAKind.MXF8F6F4, prims.CTAGroup.CTA_2,
                                        _mma_mxf8f6f4_172_d, _mma_mxf8f6f4_172_a + 4, _mma_mxf8f6f4_172_b + 4,
                                        (cutlass.Uint32(0x508002A0) | ((cutlass.Uint32(task_n_3) >> 3) << 17)), True, _mma_mxf8f6f4_172_sfa, _mma_mxf8f6f4_172_sfb,
                                        scale_vec_size=prims.Tcgen05MMAScaleVecSize.X1,
                                    )
                                    prims.tcgen05_mma_block_scale(
                                        prims.Tcgen05MMAKind.MXF8F6F4, prims.CTAGroup.CTA_2,
                                        _mma_mxf8f6f4_172_d, _mma_mxf8f6f4_172_a + 6, _mma_mxf8f6f4_172_b + 6,
                                        (cutlass.Uint32(0x708002B0) | ((cutlass.Uint32(task_n_3) >> 3) << 17)), True, _mma_mxf8f6f4_172_sfa, _mma_mxf8f6f4_172_sfb,
                                        scale_vec_size=prims.Tcgen05MMAScaleVecSize.X1,
                                    )
                                with cute.arch.elect_one():
                                    cute.nvgpu.tcgen05.commit(empty32_addr + m32_stage[0], 3, cute.nvgpu.tcgen05.CtaGroup.TWO)
                                _advanced_stage_173 = cutlass.Uint32(m32_stage[0] + 1)
                                _stage_wrapped_174 = cutlass.Boolean(_advanced_stage_173 == 10)
                                m32_stage[0] = cutlass.Uint32(cutlass.select_(_stage_wrapped_174, 0, _advanced_stage_173))
                                _phase_full32[0] = cutlass.Uint32(cutlass.select_(_stage_wrapped_174, _phase_full32[0] ^ cutlass.Uint32(1), _phase_full32[0]))
                        else:
                            for k_20 in cutlass.range(cutlass.Int32(0), cutlass.Int32(k_steps_2), cutlass.Int32(1)):
                                while not prims.mbarrier_wait_parity(full64_addr + m64_stage[0], _phase_full64[0], prims.MBarrierWait.TRY, scope=prims.MBarrierScope.CTA, order=prims.MemOrder.ACQUIRE):
                                    pass
                                prims.tcgen05_fence('after_thread_sync')
                                if prims.elect_sync():
                                    _scale_copy_175_src_0 = cutlass.Uint64(cutlass.Uint32((sw64_addr + (m64_stage[0] * 512))))
                                    _scale_copy_175_desc_0 = ((_scale_copy_175_src_0 >> 4) & cutlass.Uint64(0x3FFF)) | cutlass.Uint64(0x400800000000)
                                    _scale_copy_175_dst_0 = prims.make_tmem_ptr(cutlass.Int32(tmem_sfw), cutlass.Int32)
                                    prims.tcgen05_cp(prims.Tcgen05CpShape.SHAPE_32X128B, _scale_copy_175_dst_0, _scale_copy_175_desc_0, group=prims.CTAGroup.CTA_2, multicast=prims.Tcgen05CpMulticast.WARPX4)
                                    _scale_copy_176_src_0 = cutlass.Uint64(cutlass.Uint32((sx64_addr + (m64_stage[0] * 512))))
                                    _scale_copy_176_desc_0 = ((_scale_copy_176_src_0 >> 4) & cutlass.Uint64(0x3FFF)) | cutlass.Uint64(0x400800000000)
                                    _scale_copy_176_dst_0 = prims.make_tmem_ptr(cutlass.Int32(tmem_sfx), cutlass.Int32)
                                    prims.tcgen05_cp(prims.Tcgen05CpShape.SHAPE_32X128B, _scale_copy_176_dst_0, _scale_copy_176_desc_0, group=prims.CTAGroup.CTA_2, multicast=prims.Tcgen05CpMulticast.WARPX4)
                                    _mma_a_lo_3 = cutlass.Int32((((w64_addr >> 4) & 0x3FFF) + m64_stage[0] * 1024))
                                    _mma_b_lo_3 = cutlass.Int32((((x64_addr >> 4) & 0x3FFF) + m64_stage[0] * 256))
                                    _mma_mxf8f6f4_177_d = prims.make_tmem_ptr(cutlass.Int32((tmem_acc + (acc_stage[0] * 64))), cutlass.Int32)
                                    _mma_mxf8f6f4_177_sfa = prims.make_tmem_ptr(cutlass.Int32(tmem_sfw), cutlass.Int32)
                                    _mma_mxf8f6f4_177_sfb = prims.make_tmem_ptr(cutlass.Int32(tmem_sfx), cutlass.Int32)
                                    _mma_mxf8f6f4_177_a = cutlass.Uint64(cutlass.Uint32(_mma_a_lo_3)) | (cutlass.Uint64(0x40004040) << 32)
                                    _mma_mxf8f6f4_177_b = cutlass.Uint64(cutlass.Uint32(_mma_b_lo_3)) | (cutlass.Uint64(0x40004040) << 32)
                                    prims.tcgen05_mma_block_scale(
                                        prims.Tcgen05MMAKind.MXF8F6F4, prims.CTAGroup.CTA_2,
                                        _mma_mxf8f6f4_177_d, _mma_mxf8f6f4_177_a + 0, _mma_mxf8f6f4_177_b + 0,
                                        (cutlass.Uint32(0x10800280) | ((cutlass.Uint32(task_n_3) >> 3) << 17)), cutlass.Boolean(((k_20 == 0)) == 0), _mma_mxf8f6f4_177_sfa, _mma_mxf8f6f4_177_sfb,
                                        scale_vec_size=prims.Tcgen05MMAScaleVecSize.X1,
                                    )
                                    prims.tcgen05_mma_block_scale(
                                        prims.Tcgen05MMAKind.MXF8F6F4, prims.CTAGroup.CTA_2,
                                        _mma_mxf8f6f4_177_d, _mma_mxf8f6f4_177_a + 2, _mma_mxf8f6f4_177_b + 2,
                                        (cutlass.Uint32(0x30800290) | ((cutlass.Uint32(task_n_3) >> 3) << 17)), True, _mma_mxf8f6f4_177_sfa, _mma_mxf8f6f4_177_sfb,
                                        scale_vec_size=prims.Tcgen05MMAScaleVecSize.X1,
                                    )
                                    prims.tcgen05_mma_block_scale(
                                        prims.Tcgen05MMAKind.MXF8F6F4, prims.CTAGroup.CTA_2,
                                        _mma_mxf8f6f4_177_d, _mma_mxf8f6f4_177_a + 4, _mma_mxf8f6f4_177_b + 4,
                                        (cutlass.Uint32(0x508002A0) | ((cutlass.Uint32(task_n_3) >> 3) << 17)), True, _mma_mxf8f6f4_177_sfa, _mma_mxf8f6f4_177_sfb,
                                        scale_vec_size=prims.Tcgen05MMAScaleVecSize.X1,
                                    )
                                    prims.tcgen05_mma_block_scale(
                                        prims.Tcgen05MMAKind.MXF8F6F4, prims.CTAGroup.CTA_2,
                                        _mma_mxf8f6f4_177_d, _mma_mxf8f6f4_177_a + 6, _mma_mxf8f6f4_177_b + 6,
                                        (cutlass.Uint32(0x708002B0) | ((cutlass.Uint32(task_n_3) >> 3) << 17)), True, _mma_mxf8f6f4_177_sfa, _mma_mxf8f6f4_177_sfb,
                                        scale_vec_size=prims.Tcgen05MMAScaleVecSize.X1,
                                    )
                                with cute.arch.elect_one():
                                    cute.nvgpu.tcgen05.commit(empty64_addr + m64_stage[0], 3, cute.nvgpu.tcgen05.CtaGroup.TWO)
                                _advanced_stage_178 = cutlass.Uint32(m64_stage[0] + 1)
                                _stage_wrapped_179 = cutlass.Boolean(_advanced_stage_178 == 9)
                                m64_stage[0] = cutlass.Uint32(cutlass.select_(_stage_wrapped_179, 0, _advanced_stage_178))
                                _phase_full64[0] = cutlass.Uint32(cutlass.select_(_stage_wrapped_179, _phase_full64[0] ^ cutlass.Uint32(1), _phase_full64[0]))
                    with cute.arch.elect_one():
                        cute.nvgpu.tcgen05.commit(done_addr + acc_stage[0], 3, cute.nvgpu.tcgen05.CtaGroup.TWO)
                    _advanced_stage_180 = cutlass.Uint32(acc_stage[0] + 1)
                    _stage_wrapped_181 = cutlass.Boolean(_advanced_stage_180 == 2)
                    acc_stage[0] = cutlass.Uint32(cutlass.select_(_stage_wrapped_181, 0, _advanced_stage_180))
                    _phase_released[0] = cutlass.Uint32(cutlass.select_(_stage_wrapped_181, _phase_released[0] ^ cutlass.Uint32(1), _phase_released[0]))
                _advanced_stage_182 = cutlass.Uint32(task_stage_3[0] + 1)
                _stage_wrapped_183 = cutlass.Boolean(_advanced_stage_182 == 2)
                task_stage_3[0] = cutlass.Uint32(cutlass.select_(_stage_wrapped_183, 0, _advanced_stage_182))
                _phase_task_full_3[0] = cutlass.Uint32(cutlass.select_(_stage_wrapped_183, _phase_task_full_3[0] ^ cutlass.Uint32(1), _phase_task_full_3[0]))
    elif warp == 7:
        _phase_task_empty = cute.make_rmem_tensor((1,), cutlass.Uint32)
        count_2 = cute.make_rmem_tensor((1,), cutlass.Uint64)
        warmup = cute.make_rmem_tensor((1,), cutlass.Uint32)
        cached_inclusive = cute.make_rmem_tensor((1,), cutlass.Uint32)
        stage = cute.make_rmem_tensor((1,), cutlass.Uint32)
        live = cute.make_rmem_tensor((1,), cutlass.Int32)
        selected_1 = cute.make_rmem_tensor((1,), cutlass.Int32)
        phase_9 = cute.make_rmem_tensor((1,), cutlass.Uint32)
        task_idx = cute.make_rmem_tensor((1,), cutlass.Uint32)
        expert_6 = cute.make_rmem_tensor((1,), cutlass.Uint32)
        local_m = cute.make_rmem_tensor((1,), cutlass.Uint32)
        n_cluster = cute.make_rmem_tensor((1,), cutlass.Uint32)
        pool_block_5 = cute.make_rmem_tensor((1,), cutlass.Uint32)
        valid_m_0 = cute.make_rmem_tensor((1,), cutlass.Uint32)
        shape_n = cute.make_rmem_tensor((1,), cutlass.Uint32)
        shape_k = cute.make_rmem_tensor((1,), cutlass.Uint32)
        inclusive = cute.make_rmem_tensor((1,), cutlass.Uint32)
        issued = cute.make_rmem_tensor((1,), cutlass.Uint32)
        _phase_task_empty[0] = cutlass.Uint32(1)
        if (cta_rank == 0):
            count_2[0] = cutlass.Uint64(0)
            while cutlass.Boolean(((count_2[0] >> 32) != 2432)):
                count_2[0] = cutlass.Uint64(prims.load_ext((cute.recast_ptr(recv, dtype=cutlass.Uint64) + (lane)), dtype=cutlass.Uint64, order=prims.MemOrder.VOLATILE))
            tokens = cutlass.Uint32(cutlass.Uint32(count_2[0]))
            cute.arch.sync_warp()
            _warp_redux_u32_4 = cutlass.Uint32(cutlass_llvm.inline_asm(
                cutlass.Uint32.mlir_type,
                [cutlass.Uint32(tokens).ir_value()],
                'redux.sync.max.u32 $0, $1, 0xffffffff;', '=r,r',
                has_side_effects=True, is_align_stack=False,
                asm_dialect=cutlass_llvm.AsmDialect.AD_ATT,
            ))
            max_tokens = cutlass.Uint32(_warp_redux_u32_4)
            rank_width = cutlass.Uint32((cutlass.Uint32(cutlass.Uint32(16)) if (max_tokens <= 16) else cutlass.Uint32((cutlass.Uint32(cutlass.Uint32(32)) if (max_tokens <= 32) else cutlass.Uint32(cutlass.Uint32(64))))))
            blocks_1 = cutlass.Uint32(cutlass.Uint32(cutlass_arith.divui(cutlass.Uint32(((tokens + 64) - 1)).ir_value(), cutlass.Uint32(64).ir_value())))
            _warp_redux_u32_5 = cutlass.Uint32(cutlass_llvm.inline_asm(
                cutlass.Uint32.mlir_type,
                [cutlass.Uint32(blocks_1).ir_value()],
                'redux.sync.add.u32 $0, $1, 0xffffffff;', '=r,r',
                has_side_effects=True, is_align_stack=False,
                asm_dialect=cutlass_llvm.AsmDialect.AD_ATT,
            ))
            total = cutlass.Uint32(_warp_redux_u32_5)
            clusters = cutlass.Uint32(cutlass.Int32(cutlass_arith.divsi(cutlass.Int32(num_bids).ir_value(), cutlass.Int32(2).ir_value())))
            l1_waves = cutlass.Uint32(cutlass.Uint32(cutlass_arith.divui(cutlass.Uint32((((total * 40) + clusters) - 1)).ir_value(), cutlass.Uint32(clusters).ir_value())))
            first_wave = cutlass.Uint32(cutlass.Uint32(cutlass_arith.divui(cutlass.Uint32((((cutlass.Uint32(cutlass_arith.divui(cutlass.Uint32((clusters + 11)).ir_value(), cutlass.Uint32(12).ir_value())) * 40) + clusters) - 1)).ir_value(), cutlass.Uint32(clusters).ir_value())))
            interleave = cutlass.Int32((cutlass.Int32(cutlass_arith.divsi(cutlass.Int32((((40 + ((cutlass.Int32(total) - 1) * 28)) + cutlass.Int32(clusters)) - 1)).ir_value(), cutlass.Int32(cutlass.Int32(clusters)).ir_value())) + 1))
            _max_0 = cutlass.max(first_wave, interleave)
            _min_2 = cutlass.min(l1_waves, _max_0)
            warmup[0] = cutlass.Uint32(_min_2)
            cached_inclusive[0] = cutlass.Uint32(blocks_1)
            _shfl_up_10 = cute.arch.shuffle_sync_up(cached_inclusive[0], 1, mask=0xFFFFFFFF, mask_and_clamp=0)
            _if_condition_184 = cutlass.Boolean((lane >= 1))
            cached_inclusive[0] = cutlass.Uint32(cutlass.select_(_if_condition_184, cutlass.Uint32((cached_inclusive[0] + _shfl_up_10)), cached_inclusive[0]))
            _shfl_up_11 = cute.arch.shuffle_sync_up(cached_inclusive[0], 2, mask=0xFFFFFFFF, mask_and_clamp=0)
            _if_condition_185 = cutlass.Boolean((lane >= 2))
            cached_inclusive[0] = cutlass.Uint32(cutlass.select_(_if_condition_185, cutlass.Uint32((cached_inclusive[0] + _shfl_up_11)), cached_inclusive[0]))
            _shfl_up_12 = cute.arch.shuffle_sync_up(cached_inclusive[0], 4, mask=0xFFFFFFFF, mask_and_clamp=0)
            _if_condition_186 = cutlass.Boolean((lane >= 4))
            cached_inclusive[0] = cutlass.Uint32(cutlass.select_(_if_condition_186, cutlass.Uint32((cached_inclusive[0] + _shfl_up_12)), cached_inclusive[0]))
            _shfl_up_13 = cute.arch.shuffle_sync_up(cached_inclusive[0], 8, mask=0xFFFFFFFF, mask_and_clamp=0)
            _if_condition_187 = cutlass.Boolean((lane >= 8))
            cached_inclusive[0] = cutlass.Uint32(cutlass.select_(_if_condition_187, cutlass.Uint32((cached_inclusive[0] + _shfl_up_13)), cached_inclusive[0]))
            _shfl_up_14 = cute.arch.shuffle_sync_up(cached_inclusive[0], 16, mask=0xFFFFFFFF, mask_and_clamp=0)
            _if_condition_188 = cutlass.Boolean((lane >= 16))
            cached_inclusive[0] = cutlass.Uint32(cutlass.select_(_if_condition_188, cutlass.Uint32((cached_inclusive[0] + _shfl_up_14)), cached_inclusive[0]))
            stage[0] = cutlass.Uint32(0)
            live[0] = cutlass.Int32(1)
            while cutlass.Boolean((live[0] != 0)):
                while not prims.mbarrier_wait_parity(task_empty_addr + stage[0], _phase_task_empty[0], prims.MBarrierWait.TRY, scope=prims.MBarrierScope.CTA, order=prims.MemOrder.ACQUIRE):
                    pass
                selected_1[0] = cutlass.Int32(0)
                phase_9[0] = cutlass.Uint32(0)
                task_idx[0] = cutlass.Uint32(0)
                while cutlass.Boolean((selected_1[0] == 0)):
                    if ((warmup[0] != 4294967295) & (warmup[0] != 0)):
                        warmup[0] = cutlass.Uint32((warmup[0] - 1))
                        if prims.elect_sync():
                            _atomic_old_3 = cute.arch.atomic_add(claims + 0, cutlass.Uint32(1), sem='relaxed', scope='gpu')
                            task_idx[0] = cutlass.Uint32(_atomic_old_3)
                        _shfl_3 = cute.arch.shuffle_sync(task_idx[0], 0, mask=4294967295, mask_and_clamp=31)
                        task_idx[0] = cutlass.Uint32(_shfl_3)
                        if (task_idx[0] >= (total * 40)):
                            warmup[0] = cutlass.Uint32(4294967295)
                        else:
                            phase_9[0] = cutlass.Uint32(1)
                            selected_1[0] = cutlass.Int32(1)
                    else:
                        if prims.elect_sync():
                            _atomic_old_4 = cute.arch.atomic_add(claims + 1, cutlass.Uint32(1), sem='relaxed', scope='gpu')
                            task_idx[0] = cutlass.Uint32(_atomic_old_4)
                        _shfl_4 = cute.arch.shuffle_sync(task_idx[0], 0, mask=4294967295, mask_and_clamp=31)
                        task_idx[0] = cutlass.Uint32(_shfl_4)
                        if (task_idx[0] >= (total * 12)):
                            live[0] = cutlass.Int32(0)
                            selected_1[0] = cutlass.Int32(1)
                        else:
                            _if_condition_189 = cutlass.Boolean((warmup[0] != 4294967295))
                            warmup[0] = cutlass.Uint32(cutlass.select_(_if_condition_189, cutlass.Uint32(1), warmup[0]))
                            phase_9[0] = cutlass.Uint32(2)
                            selected_1[0] = cutlass.Int32(1)
                expert_6[0] = cutlass.Uint32(0)
                local_m[0] = cutlass.Uint32(0)
                n_cluster[0] = cutlass.Uint32(0)
                pool_block_5[0] = cutlass.Uint32(0)
                valid_m_0[0] = cutlass.Uint32(0)
                shape_n[0] = cutlass.Uint32(0)
                shape_k[0] = cutlass.Uint32(0)
                if (live[0] != 0):
                    n_clusters = cutlass.Uint32((cutlass.Int32(40) if (phase_9[0] == 1) else cutlass.Int32(12)))
                    pool_block_5[0] = cutlass.Uint32(cutlass.Uint32(cutlass_arith.divui(cutlass.Uint32(task_idx[0]).ir_value(), cutlass.Uint32(n_clusters).ir_value())))
                    n_cluster[0] = cutlass.Uint32((task_idx[0] % n_clusters))
                    inclusive[0] = cutlass.Uint32(blocks_1)
                    inclusive[0] = cutlass.Uint32(cached_inclusive[0])
                    offset_5 = cutlass.Uint32((inclusive[0] - blocks_1))
                    _vote_3 = cute.arch.vote_ballot_sync(cutlass.Boolean(((pool_block_5[0] >= offset_5) & (pool_block_5[0] < inclusive[0]))), cutlass.Uint32(0xFFFFFFFF)).bitcast(cutlass.Uint32)
                    _ffs_reversed_bits_190 = cute.arch.brev(cutlass.Uint32(_vote_3))
                    _ffs_leading_one_191 = cutlass.Int32(cute.arch.bfind(_ffs_reversed_bits_190))
                    _ffs_1 = cutlass.Int32(cutlass.select_(cutlass.Uint32(_vote_3) == cutlass.Uint32(0), cutlass.Int32(0), cutlass.Int32(32) - _ffs_leading_one_191))
                    owner_lane = cutlass.Uint32((_ffs_1 - 1))
                    lane_m = cutlass.Uint32((pool_block_5[0] - offset_5))
                    _min_3 = cutlass.min((tokens - (lane_m * 64)), 64)
                    lane_valid = cutlass.Uint32(_min_3)
                    _shfl_5 = cute.arch.shuffle_sync(lane, owner_lane, mask=4294967295, mask_and_clamp=31)
                    expert_6[0] = cutlass.Uint32(_shfl_5)
                    _shfl_6 = cute.arch.shuffle_sync(lane_m, owner_lane, mask=4294967295, mask_and_clamp=31)
                    local_m[0] = cutlass.Uint32(_shfl_6)
                    _shfl_7 = cute.arch.shuffle_sync(lane_valid, owner_lane, mask=4294967295, mask_and_clamp=31)
                    valid_m_0[0] = cutlass.Uint32(_shfl_7)
                    shape_n[0] = cutlass.Uint32((cutlass.Int32(10240) if (phase_9[0] == 1) else cutlass.Int32(3072)))
                    shape_k[0] = cutlass.Uint32((cutlass.Int32(3072) if (phase_9[0] == 1) else cutlass.Int32(5120)))
                    if (phase_9[0] == 2):
                        issued[0] = cutlass.Uint32(0)
                        while cutlass.Boolean((issued[0] < ((pool_block_5[0] + 1) * 40))):
                            issued[0] = cutlass.Uint32(prims.load_ext((cute.recast_ptr(claims, dtype=cutlass.Uint32) + (0)), dtype=cutlass.Uint32, order=prims.MemOrder.VOLATILE))
                shape_n[0] = cutlass.Uint32((shape_n[0] | (rank_width << 16)))
                if (lane < 2):
                    _cluster_mapa_192 = cute.arch.map_dsmem_ptr(task_full_addr + stage[0], cutlass.Int32(lane))
                    _mapa_0 = cutlass.Uint32(_cluster_mapa_192.toint())
                    _cluster_mapa_193 = cute.arch.map_dsmem_ptr(cute.make_ptr(cutlass.Uint8, cutlass.Uint32((tasks_addr + (stage[0] * 32))), mem_space=cute.AddressSpace.smem, assumed_align=4), cutlass.Int32(lane))
                    _mapa_1 = cutlass.Uint32(_cluster_mapa_193.toint())
                    _cluster_barrier_arrive_194_ptr = cute.make_ptr(
                        cutlass.Uint64, cutlass.Uint32(_mapa_0),
                        mem_space=cutlass.AddressSpace.dsmem, assumed_align=8,
                    )
                    prims.mbarrier_arrive_expect_tx(_cluster_barrier_arrive_194_ptr, 32, scope=prims.MemScope.CLUSTER)
                    cutlass_llvm.inline_asm(
                        res=None,
                        operands_=[(cutlass.Uint32(_mapa_1)).ir_value(), (cutlass.Uint32(phase_9[0])).ir_value(), (cutlass.Uint32(expert_6[0])).ir_value(), (cutlass.Uint32(local_m[0])).ir_value(), (cutlass.Uint32(n_cluster[0])).ir_value(), (cutlass.Uint32(_mapa_0)).ir_value()],
                        asm_string='st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [$0], {$1, $2, $3, $4}, [$5];',
                        constraints='r,r,r,r,r,r,~{memory}',
                        has_side_effects=True,
                        is_align_stack=False,
                        asm_dialect=cutlass_llvm.AsmDialect.AD_ATT,
                    )
                    cutlass_llvm.inline_asm(
                        res=None,
                        operands_=[(cutlass.Uint32((_mapa_1 + 16))).ir_value(), (cutlass.Uint32(pool_block_5[0])).ir_value(), (cutlass.Uint32(valid_m_0[0])).ir_value(), (cutlass.Uint32(shape_n[0])).ir_value(), (cutlass.Uint32(shape_k[0])).ir_value(), (cutlass.Uint32(_mapa_0)).ir_value()],
                        asm_string='st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [$0], {$1, $2, $3, $4}, [$5];',
                        constraints='r,r,r,r,r,r,~{memory}',
                        has_side_effects=True,
                        is_align_stack=False,
                        asm_dialect=cutlass_llvm.AsmDialect.AD_ATT,
                    )
                cute.arch.sync_warp()
                _advanced_stage_195 = cutlass.Uint32(stage[0] + 1)
                _stage_wrapped_196 = cutlass.Boolean(_advanced_stage_195 == 2)
                stage[0] = cutlass.Uint32(cutlass.select_(_stage_wrapped_196, 0, _advanced_stage_195))
                _phase_task_empty[0] = cutlass.Uint32(cutlass.select_(_stage_wrapped_196, _phase_task_empty[0] ^ cutlass.Uint32(1), _phase_task_empty[0]))

@cute.jit
def launch_w4a8_full_m32_fused_input(caller_x: cute.Tensor, caller_rw: cute.Tensor, staged_x: cute.Tensor, staged_sf: cute.Tensor, staged_ids: cute.Tensor, staged_rw: cute.Tensor, W1: cute.Tensor, _w4a8_tma_W1_dim_0: cutlass.Int64, _w4a8_tma_W1_dim_1: cutlass.Int64, _w4a8_tma_W1_stride16_0: cutlass.Int64, W1_pair: cute.Tensor, _w4a8_tma_W1_pair_dim_0: cutlass.Int64, _w4a8_tma_W1_pair_dim_1: cutlass.Int64, _w4a8_tma_W1_pair_stride16_0: cutlass.Int64, W2: cute.Tensor, _w4a8_tma_W2_dim_0: cutlass.Int64, _w4a8_tma_W2_dim_1: cutlass.Int64, _w4a8_tma_W2_stride16_0: cutlass.Int64, W2_pair: cute.Tensor, _w4a8_tma_W2_pair_dim_0: cutlass.Int64, _w4a8_tma_W2_pair_dim_1: cutlass.Int64, _w4a8_tma_W2_pair_stride16_0: cutlass.Int64, X1: cute.Tensor, _w4a8_tma_X1_dim_0: cutlass.Int64, _w4a8_tma_X1_dim_1: cutlass.Int64, _w4a8_tma_X1_stride16_0: cutlass.Int64, X2: cute.Tensor, _w4a8_tma_X2_dim_0: cutlass.Int64, _w4a8_tma_X2_dim_1: cutlass.Int64, _w4a8_tma_X2_stride16_0: cutlass.Int64, X1_32: cute.Tensor, _w4a8_tma_X1_32_dim_0: cutlass.Int64, _w4a8_tma_X1_32_dim_1: cutlass.Int64, _w4a8_tma_X1_32_stride16_0: cutlass.Int64, X2_32: cute.Tensor, _w4a8_tma_X2_32_dim_0: cutlass.Int64, _w4a8_tma_X2_32_dim_1: cutlass.Int64, _w4a8_tma_X2_32_stride16_0: cutlass.Int64, X1_8: cute.Tensor, _w4a8_tma_X1_8_dim_0: cutlass.Int64, _w4a8_tma_X1_8_dim_1: cutlass.Int64, _w4a8_tma_X1_8_stride16_0: cutlass.Int64, X2_8: cute.Tensor, _w4a8_tma_X2_8_dim_0: cutlass.Int64, _w4a8_tma_X2_8_dim_1: cutlass.Int64, _w4a8_tma_X2_8_stride16_0: cutlass.Int64, SW1: cute.Tensor, _w4a8_tma_SW1_dim_0: cutlass.Int64, _w4a8_tma_SW1_dim_1: cutlass.Int64, _w4a8_tma_SW1_stride16_0: cutlass.Int64, SW1_1: cute.Tensor, _w4a8_tma_SW1_1_dim_0: cutlass.Int64, _w4a8_tma_SW1_1_dim_1: cutlass.Int64, _w4a8_tma_SW1_1_stride16_0: cutlass.Int64, SW2: cute.Tensor, _w4a8_tma_SW2_dim_0: cutlass.Int64, _w4a8_tma_SW2_dim_1: cutlass.Int64, _w4a8_tma_SW2_stride16_0: cutlass.Int64, SW2_1: cute.Tensor, _w4a8_tma_SW2_1_dim_0: cutlass.Int64, _w4a8_tma_SW2_1_dim_1: cutlass.Int64, _w4a8_tma_SW2_1_stride16_0: cutlass.Int64, SX1: cute.Tensor, _w4a8_tma_SX1_dim_0: cutlass.Int64, _w4a8_tma_SX1_dim_1: cutlass.Int64, _w4a8_tma_SX1_stride16_0: cutlass.Int64, SX1_1: cute.Tensor, _w4a8_tma_SX1_1_dim_0: cutlass.Int64, _w4a8_tma_SX1_1_dim_1: cutlass.Int64, _w4a8_tma_SX1_1_stride16_0: cutlass.Int64, SX2: cute.Tensor, _w4a8_tma_SX2_dim_0: cutlass.Int64, _w4a8_tma_SX2_dim_1: cutlass.Int64, _w4a8_tma_SX2_stride16_0: cutlass.Int64, SX2_1: cute.Tensor, _w4a8_tma_SX2_1_dim_0: cutlass.Int64, _w4a8_tma_SX2_1_dim_1: cutlass.Int64, _w4a8_tma_SX2_1_stride16_0: cutlass.Int64, RW: cute.Tensor, Q: cute.Tensor, _w4a8_tma_Q_dim_0: cutlass.Int64, _w4a8_tma_Q_dim_1: cutlass.Int64, _w4a8_tma_Q_stride16_0: cutlass.Int64, Q8: cute.Tensor, _w4a8_tma_Q8_dim_0: cutlass.Int64, _w4a8_tma_Q8_dim_1: cutlass.Int64, _w4a8_tma_Q8_stride16_0: cutlass.Int64, Q32: cute.Tensor, _w4a8_tma_Q32_dim_0: cutlass.Int64, _w4a8_tma_Q32_dim_1: cutlass.Int64, _w4a8_tma_Q32_stride16_0: cutlass.Int64, QData: cute.Tensor, SF: cute.Tensor, output_peers: cute.Tensor, slots: cute.Tensor, output: cute.Tensor, epilogue_grid: cute.Tensor, l1_full: cute.Tensor, l1_empty: cute.Tensor, l2_full: cute.Tensor, l2_empty: cute.Tensor, ids: cute.Tensor, send: cute.Tensor, rank_counts: cute.Tensor, indices: cute.Tensor, src_peers: cute.Tensor, recv_peers: cute.Tensor, sum_peers: cute.Tensor, grid_counter: cute.Tensor, signals: cute.Tensor, status: cute.Tensor, signal_peers: cute.Tensor, rank: cutlass.Uint32, live_tokens: cutlass.Uint32, token_peers: cute.Tensor, sf_peers: cute.Tensor, weight_peers: cute.Tensor, XData: cute.Tensor, XSFData: cute.Tensor, metadata: cute.Tensor, recv: cute.Tensor, claims: cute.Tensor, pool_blocks: cutlass.Uint32, active_n: cutlass.Int32, valid_m: cutlass.Int32, epoch: cutlass.Int32, fc1_tiles: cutlass.Int32, fc2_tiles: cutlass.Int32, fc1_k: cutlass.Int32, fc2_k: cutlass.Int32, grid_x: cutlass.Int32, grid_y: cutlass.Int32, grid_z: cutlass.Int32, stream: cuda.CUstream):
    _tma_W1 = create_tensor_map_tiled(
        W1.iterator.toint(),
        cutlass.Float4E2M1FN,
        [_w4a8_tma_W1_dim_0, _w4a8_tma_W1_dim_1],
        [_w4a8_tma_W1_stride16_0],
        [128, 128],
        swizzle=TensorMapSwizzle.s128b,
        l2_promotion=TensorMapL2Promotion.l2_256b,
        oob_fill=TensorMapFloatOOBFill.none,
    )
    _tma_W1_pair = create_tensor_map_tiled(
        W1_pair.iterator.toint(),
        cutlass.Float4E2M1FN,
        [_w4a8_tma_W1_pair_dim_0, _w4a8_tma_W1_pair_dim_1],
        [_w4a8_tma_W1_pair_stride16_0],
        [128, 256],
        swizzle=TensorMapSwizzle.s128b,
        l2_promotion=TensorMapL2Promotion.l2_256b,
        oob_fill=TensorMapFloatOOBFill.none,
    )
    _tma_W2 = create_tensor_map_tiled(
        W2.iterator.toint(),
        cutlass.Float4E2M1FN,
        [_w4a8_tma_W2_dim_0, _w4a8_tma_W2_dim_1],
        [_w4a8_tma_W2_stride16_0],
        [128, 128],
        swizzle=TensorMapSwizzle.s128b,
        l2_promotion=TensorMapL2Promotion.l2_256b,
        oob_fill=TensorMapFloatOOBFill.none,
    )
    _tma_W2_pair = create_tensor_map_tiled(
        W2_pair.iterator.toint(),
        cutlass.Float4E2M1FN,
        [_w4a8_tma_W2_pair_dim_0, _w4a8_tma_W2_pair_dim_1],
        [_w4a8_tma_W2_pair_stride16_0],
        [128, 256],
        swizzle=TensorMapSwizzle.s128b,
        l2_promotion=TensorMapL2Promotion.l2_256b,
        oob_fill=TensorMapFloatOOBFill.none,
    )
    _tma_X1 = create_tensor_map_tiled(
        X1.iterator.toint(),
        cutlass.Uint8,
        [_w4a8_tma_X1_dim_0, _w4a8_tma_X1_dim_1],
        [_w4a8_tma_X1_stride16_0],
        [128, 16],
        swizzle=TensorMapSwizzle.s128b,
        l2_promotion=TensorMapL2Promotion.l2_256b,
        oob_fill=TensorMapFloatOOBFill.none,
    )
    _tma_X2 = create_tensor_map_tiled(
        X2.iterator.toint(),
        cutlass.Uint8,
        [_w4a8_tma_X2_dim_0, _w4a8_tma_X2_dim_1],
        [_w4a8_tma_X2_stride16_0],
        [128, 16],
        swizzle=TensorMapSwizzle.s128b,
        l2_promotion=TensorMapL2Promotion.l2_256b,
        oob_fill=TensorMapFloatOOBFill.none,
    )
    _tma_X1_32 = create_tensor_map_tiled(
        X1_32.iterator.toint(),
        cutlass.Uint8,
        [_w4a8_tma_X1_32_dim_0, _w4a8_tma_X1_32_dim_1],
        [_w4a8_tma_X1_32_stride16_0],
        [128, 32],
        swizzle=TensorMapSwizzle.s128b,
        l2_promotion=TensorMapL2Promotion.l2_256b,
        oob_fill=TensorMapFloatOOBFill.none,
    )
    _tma_X2_32 = create_tensor_map_tiled(
        X2_32.iterator.toint(),
        cutlass.Uint8,
        [_w4a8_tma_X2_32_dim_0, _w4a8_tma_X2_32_dim_1],
        [_w4a8_tma_X2_32_stride16_0],
        [128, 32],
        swizzle=TensorMapSwizzle.s128b,
        l2_promotion=TensorMapL2Promotion.l2_256b,
        oob_fill=TensorMapFloatOOBFill.none,
    )
    _tma_X1_8 = create_tensor_map_tiled(
        X1_8.iterator.toint(),
        cutlass.Uint8,
        [_w4a8_tma_X1_8_dim_0, _w4a8_tma_X1_8_dim_1],
        [_w4a8_tma_X1_8_stride16_0],
        [128, 8],
        swizzle=TensorMapSwizzle.s128b,
        l2_promotion=TensorMapL2Promotion.l2_256b,
        oob_fill=TensorMapFloatOOBFill.none,
    )
    _tma_X2_8 = create_tensor_map_tiled(
        X2_8.iterator.toint(),
        cutlass.Uint8,
        [_w4a8_tma_X2_8_dim_0, _w4a8_tma_X2_8_dim_1],
        [_w4a8_tma_X2_8_stride16_0],
        [128, 8],
        swizzle=TensorMapSwizzle.s128b,
        l2_promotion=TensorMapL2Promotion.l2_256b,
        oob_fill=TensorMapFloatOOBFill.none,
    )
    _tma_SW1 = create_tensor_map_tiled(
        SW1.iterator.toint(),
        cutlass.Uint32,
        [_w4a8_tma_SW1_dim_0, _w4a8_tma_SW1_dim_1],
        [_w4a8_tma_SW1_stride16_0],
        [128, 2],
        swizzle=TensorMapSwizzle.none,
        l2_promotion=TensorMapL2Promotion.l2_256b,
        oob_fill=TensorMapFloatOOBFill.none,
    )
    _tma_SW1_1 = create_tensor_map_tiled(
        SW1_1.iterator.toint(),
        cutlass.Uint32,
        [_w4a8_tma_SW1_1_dim_0, _w4a8_tma_SW1_1_dim_1],
        [_w4a8_tma_SW1_1_stride16_0],
        [128, 1],
        swizzle=TensorMapSwizzle.none,
        l2_promotion=TensorMapL2Promotion.l2_256b,
        oob_fill=TensorMapFloatOOBFill.none,
    )
    _tma_SW2 = create_tensor_map_tiled(
        SW2.iterator.toint(),
        cutlass.Uint32,
        [_w4a8_tma_SW2_dim_0, _w4a8_tma_SW2_dim_1],
        [_w4a8_tma_SW2_stride16_0],
        [128, 2],
        swizzle=TensorMapSwizzle.none,
        l2_promotion=TensorMapL2Promotion.l2_256b,
        oob_fill=TensorMapFloatOOBFill.none,
    )
    _tma_SW2_1 = create_tensor_map_tiled(
        SW2_1.iterator.toint(),
        cutlass.Uint32,
        [_w4a8_tma_SW2_1_dim_0, _w4a8_tma_SW2_1_dim_1],
        [_w4a8_tma_SW2_1_stride16_0],
        [128, 1],
        swizzle=TensorMapSwizzle.none,
        l2_promotion=TensorMapL2Promotion.l2_256b,
        oob_fill=TensorMapFloatOOBFill.none,
    )
    _tma_SX1 = create_tensor_map_tiled(
        SX1.iterator.toint(),
        cutlass.Uint32,
        [_w4a8_tma_SX1_dim_0, _w4a8_tma_SX1_dim_1],
        [_w4a8_tma_SX1_stride16_0],
        [128, 2],
        swizzle=TensorMapSwizzle.none,
        l2_promotion=TensorMapL2Promotion.l2_256b,
        oob_fill=TensorMapFloatOOBFill.none,
    )
    _tma_SX1_1 = create_tensor_map_tiled(
        SX1_1.iterator.toint(),
        cutlass.Uint32,
        [_w4a8_tma_SX1_1_dim_0, _w4a8_tma_SX1_1_dim_1],
        [_w4a8_tma_SX1_1_stride16_0],
        [128, 1],
        swizzle=TensorMapSwizzle.none,
        l2_promotion=TensorMapL2Promotion.l2_256b,
        oob_fill=TensorMapFloatOOBFill.none,
    )
    _tma_SX2 = create_tensor_map_tiled(
        SX2.iterator.toint(),
        cutlass.Uint32,
        [_w4a8_tma_SX2_dim_0, _w4a8_tma_SX2_dim_1],
        [_w4a8_tma_SX2_stride16_0],
        [128, 2],
        swizzle=TensorMapSwizzle.none,
        l2_promotion=TensorMapL2Promotion.l2_256b,
        oob_fill=TensorMapFloatOOBFill.none,
    )
    _tma_SX2_1 = create_tensor_map_tiled(
        SX2_1.iterator.toint(),
        cutlass.Uint32,
        [_w4a8_tma_SX2_1_dim_0, _w4a8_tma_SX2_1_dim_1],
        [_w4a8_tma_SX2_1_stride16_0],
        [128, 1],
        swizzle=TensorMapSwizzle.none,
        l2_promotion=TensorMapL2Promotion.l2_256b,
        oob_fill=TensorMapFloatOOBFill.none,
    )
    _tma_Q = create_tensor_map_tiled(
        Q.iterator.toint(),
        cutlass.Uint8,
        [_w4a8_tma_Q_dim_0, _w4a8_tma_Q_dim_1],
        [_w4a8_tma_Q_stride16_0],
        [64, 16],
        swizzle=TensorMapSwizzle.s64b,
        l2_promotion=TensorMapL2Promotion.none,
        oob_fill=TensorMapFloatOOBFill.none,
    )
    _tma_Q8 = create_tensor_map_tiled(
        Q8.iterator.toint(),
        cutlass.Uint8,
        [_w4a8_tma_Q8_dim_0, _w4a8_tma_Q8_dim_1],
        [_w4a8_tma_Q8_stride16_0],
        [64, 8],
        swizzle=TensorMapSwizzle.s64b,
        l2_promotion=TensorMapL2Promotion.none,
        oob_fill=TensorMapFloatOOBFill.none,
    )
    _tma_Q32 = create_tensor_map_tiled(
        Q32.iterator.toint(),
        cutlass.Uint8,
        [_w4a8_tma_Q32_dim_0, _w4a8_tma_Q32_dim_1],
        [_w4a8_tma_Q32_stride16_0],
        [64, 32],
        swizzle=TensorMapSwizzle.s64b,
        l2_promotion=TensorMapL2Promotion.none,
        oob_fill=TensorMapFloatOOBFill.none,
    )
    _w4a8_launch_cluster_spread(kernel_w4a8_full_m32_fused_input(caller_x.iterator, caller_rw.iterator, staged_x.iterator, staged_sf.iterator, staged_ids.iterator, staged_rw.iterator, _tma_W1, _tma_W1_pair, _tma_W2, _tma_W2_pair, _tma_X1, _tma_X2, _tma_X1_32, _tma_X2_32, _tma_X1_8, _tma_X2_8, _tma_SW1, _tma_SW1_1, _tma_SW2, _tma_SW2_1, _tma_SX1, _tma_SX1_1, _tma_SX2, _tma_SX2_1, RW.iterator, _tma_Q, _tma_Q8, _tma_Q32, QData.iterator, SF.iterator, output_peers.iterator, slots.iterator, output.iterator, epilogue_grid.iterator, l1_full.iterator, l1_empty.iterator, l2_full.iterator, l2_empty.iterator, ids.iterator, send.iterator, rank_counts.iterator, indices.iterator, src_peers.iterator, recv_peers.iterator, sum_peers.iterator, grid_counter.iterator, signals.iterator, status.iterator, signal_peers.iterator, rank, live_tokens, token_peers.iterator, sf_peers.iterator, weight_peers.iterator, XData.iterator, XSFData.iterator, metadata.iterator, recv.iterator, claims.iterator, pool_blocks, active_n, valid_m, epoch, fc1_tiles, fc2_tiles, fc1_k, fc2_k),
        grid=(grid_x, grid_y, grid_z),
        block=(512, 1, 1),
        cluster=(2, 1, 1),
        smem=226432,
        stream=stream,
    )

def compile_program():
    return cute.compile(launch_w4a8_full_m32_fused_input,
        make_fake_tensor(cutlass.BFloat16, (cute.sym_int64(symbol='caller_x'),), (1,), assumed_align=2),
        make_fake_tensor(cutlass.Float32, (cute.sym_int64(symbol='caller_rw'),), (1,), assumed_align=4),
        make_fake_tensor(cutlass.Uint32, (cute.sym_int64(symbol='staged_x'),), (1,), assumed_align=4),
        make_fake_tensor(cutlass.Uint8, (cute.sym_int64(symbol='staged_sf'),), (1,), assumed_align=1),
        make_fake_tensor(cutlass.Int64, (cute.sym_int64(symbol='staged_ids'),), (1,), assumed_align=8),
        make_fake_tensor(cutlass.Float32, (cute.sym_int64(symbol='staged_rw'),), (1,), assumed_align=4),
        make_fake_tensor(cutlass.Uint8, (cute.sym_int64(symbol='W1'),), (1,), assumed_align=16),
        cutlass.Int64(0),
        cutlass.Int64(0),
        cutlass.Int64(0),
        make_fake_tensor(cutlass.Uint8, (cute.sym_int64(symbol='W1_pair'),), (1,), assumed_align=16),
        cutlass.Int64(0),
        cutlass.Int64(0),
        cutlass.Int64(0),
        make_fake_tensor(cutlass.Uint8, (cute.sym_int64(symbol='W2'),), (1,), assumed_align=16),
        cutlass.Int64(0),
        cutlass.Int64(0),
        cutlass.Int64(0),
        make_fake_tensor(cutlass.Uint8, (cute.sym_int64(symbol='W2_pair'),), (1,), assumed_align=16),
        cutlass.Int64(0),
        cutlass.Int64(0),
        cutlass.Int64(0),
        make_fake_tensor(cutlass.Uint8, (cute.sym_int64(symbol='X1'),), (1,), assumed_align=16),
        cutlass.Int64(0),
        cutlass.Int64(0),
        cutlass.Int64(0),
        make_fake_tensor(cutlass.Uint8, (cute.sym_int64(symbol='X2'),), (1,), assumed_align=16),
        cutlass.Int64(0),
        cutlass.Int64(0),
        cutlass.Int64(0),
        make_fake_tensor(cutlass.Uint8, (cute.sym_int64(symbol='X1_32'),), (1,), assumed_align=16),
        cutlass.Int64(0),
        cutlass.Int64(0),
        cutlass.Int64(0),
        make_fake_tensor(cutlass.Uint8, (cute.sym_int64(symbol='X2_32'),), (1,), assumed_align=16),
        cutlass.Int64(0),
        cutlass.Int64(0),
        cutlass.Int64(0),
        make_fake_tensor(cutlass.Uint8, (cute.sym_int64(symbol='X1_8'),), (1,), assumed_align=16),
        cutlass.Int64(0),
        cutlass.Int64(0),
        cutlass.Int64(0),
        make_fake_tensor(cutlass.Uint8, (cute.sym_int64(symbol='X2_8'),), (1,), assumed_align=16),
        cutlass.Int64(0),
        cutlass.Int64(0),
        cutlass.Int64(0),
        make_fake_tensor(cutlass.Uint32, (cute.sym_int64(symbol='SW1'),), (1,), assumed_align=16),
        cutlass.Int64(0),
        cutlass.Int64(0),
        cutlass.Int64(0),
        make_fake_tensor(cutlass.Uint32, (cute.sym_int64(symbol='SW1_1'),), (1,), assumed_align=16),
        cutlass.Int64(0),
        cutlass.Int64(0),
        cutlass.Int64(0),
        make_fake_tensor(cutlass.Uint32, (cute.sym_int64(symbol='SW2'),), (1,), assumed_align=16),
        cutlass.Int64(0),
        cutlass.Int64(0),
        cutlass.Int64(0),
        make_fake_tensor(cutlass.Uint32, (cute.sym_int64(symbol='SW2_1'),), (1,), assumed_align=16),
        cutlass.Int64(0),
        cutlass.Int64(0),
        cutlass.Int64(0),
        make_fake_tensor(cutlass.Uint32, (cute.sym_int64(symbol='SX1'),), (1,), assumed_align=16),
        cutlass.Int64(0),
        cutlass.Int64(0),
        cutlass.Int64(0),
        make_fake_tensor(cutlass.Uint32, (cute.sym_int64(symbol='SX1_1'),), (1,), assumed_align=16),
        cutlass.Int64(0),
        cutlass.Int64(0),
        cutlass.Int64(0),
        make_fake_tensor(cutlass.Uint32, (cute.sym_int64(symbol='SX2'),), (1,), assumed_align=16),
        cutlass.Int64(0),
        cutlass.Int64(0),
        cutlass.Int64(0),
        make_fake_tensor(cutlass.Uint32, (cute.sym_int64(symbol='SX2_1'),), (1,), assumed_align=16),
        cutlass.Int64(0),
        cutlass.Int64(0),
        cutlass.Int64(0),
        make_fake_tensor(cutlass.Float32, (cute.sym_int64(symbol='RW'),), (1,), assumed_align=4),
        make_fake_tensor(cutlass.Uint8, (cute.sym_int64(symbol='Q'),), (1,), assumed_align=16),
        cutlass.Int64(0),
        cutlass.Int64(0),
        cutlass.Int64(0),
        make_fake_tensor(cutlass.Uint8, (cute.sym_int64(symbol='Q8'),), (1,), assumed_align=16),
        cutlass.Int64(0),
        cutlass.Int64(0),
        cutlass.Int64(0),
        make_fake_tensor(cutlass.Uint8, (cute.sym_int64(symbol='Q32'),), (1,), assumed_align=16),
        cutlass.Int64(0),
        cutlass.Int64(0),
        cutlass.Int64(0),
        make_fake_tensor(cutlass.Uint8, (cute.sym_int64(symbol='QData'),), (1,), assumed_align=1),
        make_fake_tensor(cutlass.Uint8, (cute.sym_int64(symbol='SF'),), (1,), assumed_align=1),
        make_fake_tensor(cutlass.Uint64, (cute.sym_int64(symbol='output_peers'),), (1,), assumed_align=8),
        make_fake_tensor(cutlass.Uint8, (cute.sym_int64(symbol='slots'),), (1,), assumed_align=1),
        make_fake_tensor(cutlass.Uint8, (cute.sym_int64(symbol='output'),), (1,), assumed_align=1),
        make_fake_tensor(cutlass.Uint32, (cute.sym_int64(symbol='epilogue_grid'),), (1,), assumed_align=4),
        make_fake_tensor(cutlass.Uint32, (cute.sym_int64(symbol='l1_full'),), (1,), assumed_align=4),
        make_fake_tensor(cutlass.Uint32, (cute.sym_int64(symbol='l1_empty'),), (1,), assumed_align=4),
        make_fake_tensor(cutlass.Uint32, (cute.sym_int64(symbol='l2_full'),), (1,), assumed_align=4),
        make_fake_tensor(cutlass.Uint32, (cute.sym_int64(symbol='l2_empty'),), (1,), assumed_align=4),
        make_fake_tensor(cutlass.Int64, (cute.sym_int64(symbol='ids'),), (1,), assumed_align=8),
        make_fake_tensor(cutlass.Uint64, (cute.sym_int64(symbol='send'),), (1,), assumed_align=8),
        make_fake_tensor(cutlass.Uint64, (cute.sym_int64(symbol='rank_counts'),), (1,), assumed_align=8),
        make_fake_tensor(cutlass.Uint32, (cute.sym_int64(symbol='indices'),), (1,), assumed_align=4),
        make_fake_tensor(cutlass.Uint64, (cute.sym_int64(symbol='src_peers'),), (1,), assumed_align=8),
        make_fake_tensor(cutlass.Uint64, (cute.sym_int64(symbol='recv_peers'),), (1,), assumed_align=8),
        make_fake_tensor(cutlass.Uint64, (cute.sym_int64(symbol='sum_peers'),), (1,), assumed_align=8),
        make_fake_tensor(cutlass.Uint32, (cute.sym_int64(symbol='grid_counter'),), (1,), assumed_align=4),
        make_fake_tensor(cutlass.Uint32, (cute.sym_int64(symbol='signals'),), (1,), assumed_align=4),
        make_fake_tensor(cutlass.Uint32, (cute.sym_int64(symbol='status'),), (1,), assumed_align=4),
        make_fake_tensor(cutlass.Uint64, (cute.sym_int64(symbol='signal_peers'),), (1,), assumed_align=8),
        cutlass.Uint32(0),
        cutlass.Uint32(0),
        make_fake_tensor(cutlass.Uint64, (cute.sym_int64(symbol='token_peers'),), (1,), assumed_align=8),
        make_fake_tensor(cutlass.Uint64, (cute.sym_int64(symbol='sf_peers'),), (1,), assumed_align=8),
        make_fake_tensor(cutlass.Uint64, (cute.sym_int64(symbol='weight_peers'),), (1,), assumed_align=8),
        make_fake_tensor(cutlass.Uint8, (cute.sym_int64(symbol='XData'),), (1,), assumed_align=1),
        make_fake_tensor(cutlass.Uint32, (cute.sym_int64(symbol='XSFData'),), (1,), assumed_align=4),
        make_fake_tensor(cutlass.Uint32, (cute.sym_int64(symbol='metadata'),), (1,), assumed_align=4),
        make_fake_tensor(cutlass.Uint64, (cute.sym_int64(symbol='recv'),), (1,), assumed_align=8),
        make_fake_tensor(cutlass.Uint32, (cute.sym_int64(symbol='claims'),), (1,), assumed_align=4),
        cutlass.Uint32(0),
        cutlass.Int32(0),
        cutlass.Int32(0),
        cutlass.Int32(0),
        cutlass.Int32(0),
        cutlass.Int32(0),
        cutlass.Int32(0),
        cutlass.Int32(0),
        cutlass.Int32(1),
        cutlass.Int32(1),
        cutlass.Int32(1),
        make_fake_stream(use_tvm_ffi_env_stream=True),
        options='--enable-tvm-ffi --ptxas-options=--opt-level=2 --gpu-arch=sm_103a',
    )
