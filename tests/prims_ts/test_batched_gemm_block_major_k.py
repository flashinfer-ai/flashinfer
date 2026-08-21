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

"""BlockMajorK weight-layout coverage for Prims-TS BatchedGemm."""

import pytest
import torch

from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
    ActKind,
    BatchMode,
    DType,
    RouteImpl,
    SfLayout,
    TileScheduler,
    uniform_pipeline_stage_overrides,
)
from flashinfer.tllm_enums import WeightLayout
from flashinfer.utils import is_sm100a_supported

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required"),
    pytest.mark.skipif(
        (torch.cuda.is_available() and not is_sm100a_supported(torch.device("cuda"))),
        reason="kernels require Blackwell sm_100+ (skip on sm_120a)",
    ),
]


def _reference_check(**kwargs):
    from flashinfer.prims_ts.batched_gemm.batched_gemm_run import reference_check

    assert reference_check(weight_layout=int(WeightLayout.BlockMajorK), **kwargs)


def test_bf16_non_swap_uses_block_major_k_weight_b():
    _reference_check(
        num_experts=2,
        num_tokens=128,
        top_k=1,
        tile_n=16,
        tile_k=128,
        mma_n=16,
        epi_tile_n=16,
        **uniform_pipeline_stage_overrides(4),
        batch_mode=int(BatchMode.BATCH_M),
        transpose_mma_output=0,
        route_act=int(RouteImpl.NONE),
        tile_scheduler=int(TileScheduler.STATIC),
        act_kind=int(ActKind.NONE),
        sf_layout_b=int(SfLayout.R8c4),
        cluster_m=1,
        tile_m=128,
        mma_m=128,
        mma_k=16,
        dtype_a=int(DType.BF16),
        dtype_b=int(DType.BF16),
        dtype_c=int(DType.BF16),
        epilogue_regs=160,
        mma_regs=24,
        load_regs=24,
        padding_regs=24,
        workid_regs=24,
        use_unroll_loop_2x_for_mma=0,
        use_max_tmem_overlap=0,
    )


def test_bf16_swap_uses_block_major_k_weight_a():
    _reference_check(
        num_experts=2,
        num_tokens=128,
        top_k=1,
        tile_n=16,
        tile_k=128,
        mma_n=16,
        epi_tile_n=16,
        **uniform_pipeline_stage_overrides(4),
        batch_mode=int(BatchMode.BATCH_N),
        transpose_mma_output=1,
        route_act=int(RouteImpl.TMA),
        tile_scheduler=int(TileScheduler.STATIC),
        act_kind=int(ActKind.NONE),
        sf_layout_b=int(SfLayout.R8c4),
        cluster_m=1,
        tile_m=128,
        mma_m=128,
        mma_k=16,
        dtype_a=int(DType.BF16),
        dtype_b=int(DType.BF16),
        dtype_c=int(DType.BF16),
        epilogue_regs=160,
        mma_regs=48,
        load_regs=48,
        gather_regs=48,
        padding_regs=48,
        use_unroll_loop_2x_for_mma=0,
        use_max_tmem_overlap=0,
    )


def test_nvfp4_swap_uses_block_major_k_weight_a():
    _reference_check(
        num_experts=2,
        num_tokens=128,
        top_k=1,
        tile_n=8,
        tile_k=256,
        mma_n=8,
        epi_tile_n=8,
        **uniform_pipeline_stage_overrides(5),
        batch_mode=int(BatchMode.BATCH_N),
        transpose_mma_output=1,
        route_act=int(RouteImpl.TMA),
        act_kind=int(ActKind.SWIGLU),
        route_sfs_act=int(RouteImpl.NONE),
        sf_layout_b=int(SfLayout.R8c4),
        cluster_m=1,
        tile_m=128,
        mma_m=128,
        mma_k=64,
        dtype_a=int(DType.E2M1),
        dtype_b=int(DType.E2M1),
        dtype_c=int(DType.BF16),
        sf_bits=8,
        epilogue_regs=128,
        mma_regs=48,
        load_regs=48,
        load_sf_regs=48,
        copy_sf_regs=48,
        workid_regs=48,
        padding_regs=48,
        gather_regs=48,
        use_unroll_loop_2x_for_mma=0,
        use_max_tmem_overlap=0,
    )


def test_fp8_swap_uses_block_major_k_weight_a():
    _reference_check(
        num_experts=2,
        num_tokens=16,
        top_k=1,
        tile_n=8,
        tile_k=256,
        mma_n=8,
        epi_tile_n=8,
        **uniform_pipeline_stage_overrides(6),
        batch_mode=int(BatchMode.BATCH_N),
        transpose_mma_output=1,
        route_act=int(RouteImpl.NONE),
        act_kind=int(ActKind.NONE),
        tile_scheduler=int(TileScheduler.PERSISTENT),
        dtype_a=int(DType.E4M3),
        dtype_b=int(DType.E4M3),
        dtype_c=int(DType.BF16),
        tile_m=128,
        epi_tile_m=128,
        mma_m=128,
        mma_k=32,
        cluster_m=1,
        num_stages_tmem_acc=2,
        use_tma_store=1,
        use_tma_oob_opt=1,
        use_global_scales=1,
        use_unroll_loop_2x_for_mma=0,
    )
