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

"""Large-MNK MoE smoke tests that fail fast on compile or runtime hangs."""

import pytest

import torch

from flashinfer.utils import is_sm100a_supported

from flashinfer.prims_ts.batched_gemm.batched_gemm_run import benchmark
from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
    ActKind,
    BatchMode,
    BiasType,
    DType,
    RouteImpl,
    SfLayout,
    TileScheduler,
    uniform_pipeline_stage_overrides,
)

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required"),
    pytest.mark.skipif(
        (torch.cuda.is_available() and not is_sm100a_supported(torch.device("cuda"))),
        reason="kernels require Blackwell sm_100+ (skip on sm_120a)",
    ),
]


LARGE_MNK_CASES = [
    (
        "bf16",
        "bf16_fc1_t32_k128_persistent",
        {
            "problem_n": 4096,
            "problem_k": 7168,
            "kwargs": {
                "batch_mode": int(BatchMode.BATCH_N),
                "transpose_mma_output": 1,
                "route_act": 2,
                "route_sfs_act": 0,
                "act_kind": 1,
                "tile_scheduler": 1,
                "tile_m": 128,
                "tile_n": 32,
                "tile_k": 128,
                "epi_tile_m": 128,
                "epi_tile_n": 32,
                "mma_m": 128,
                "mma_n": 32,
                "mma_k": 16,
                "cluster_m": 1,
                **uniform_pipeline_stage_overrides(5),
                "num_stages_tmem_acc": 2,
                "dtype_a": int(DType.BF16),
                "dtype_b": int(DType.BF16),
                "dtype_c": int(DType.BF16),
                "epilogue_regs": 128,
                "mma_regs": 96,
                "load_regs": 96,
                "padding_regs": 96,
                "workid_regs": 96,
                "gather_regs": 96,
                "use_tma_store": 1,
                "use_tma_oob_opt": 1,
                "use_early_exit": 1,
                "use_clc_fast_drain": 0,
                "use_max_tmem_overlap": 0,
                "use_unroll_loop_2x_for_mma": 0,
                "bias_type": 0,
                "has_gemm1_clamp_limit": 1,
            },
        },
    ),
    (
        "fp4",
        "fp4_fc1_t32_k256_persistent",
        {
            "problem_n": 4096,
            "problem_k": 7168,
            "kwargs": {
                "batch_mode": int(BatchMode.BATCH_N),
                "transpose_mma_output": 1,
                "route_act": 1,
                "route_sfs_act": 2,
                "act_kind": 1,
                "tile_scheduler": 1,
                "tile_m": 128,
                "tile_n": 32,
                "tile_k": 256,
                "epi_tile_m": 128,
                "epi_tile_n": 32,
                "mma_m": 128,
                "mma_n": 32,
                "mma_k": 64,
                "cluster_m": 1,
                **uniform_pipeline_stage_overrides(9),
                "num_stages_tmem_acc": 2,
                "dtype_a": int(DType.E2M1),
                "dtype_b": int(DType.E2M1),
                "dtype_c": int(DType.E2M1),
                "sf_bits": 8,
                "sf_layout_a": 3,
                "sf_layout_b": 2,
                "sf_layout_c": 1,
                "epilogue_regs": 160,
                "mma_regs": 48,
                "load_regs": 48,
                "load_sf_regs": 48,
                "copy_sf_regs": 48,
                "padding_regs": 48,
                "workid_regs": 48,
                "use_tma_store": 1,
                "use_tma_oob_opt": 1,
                "use_early_exit": 1,
                "use_clc_fast_drain": 0,
                "use_global_scales": 1,
                "use_max_tmem_overlap": 0,
                "use_unroll_loop_2x_for_mma": 0,
                "bias_type": 1,
                "has_gemm1_clamp_limit": 1,
            },
        },
    ),
    (
        "fp8",
        "fp8_fc1_t32_k256_persistent",
        {
            "problem_n": 4096,
            "problem_k": 7168,
            "kwargs": {
                "batch_mode": int(BatchMode.BATCH_N),
                "transpose_mma_output": 1,
                "route_act": 1,
                "route_sfs_act": 0,
                "act_kind": 1,
                "tile_scheduler": 1,
                "tile_m": 128,
                "tile_n": 32,
                "tile_k": 256,
                "epi_tile_m": 128,
                "epi_tile_n": 32,
                "mma_m": 128,
                "mma_n": 32,
                "mma_k": 32,
                "cluster_m": 1,
                **uniform_pipeline_stage_overrides(5),
                "num_stages_tmem_acc": 2,
                "dtype_a": int(DType.E4M3),
                "dtype_b": int(DType.E4M3),
                "dtype_c": int(DType.E4M3),
                "per_token_sf_dtype": int(DType.BF16),
                "epilogue_regs": 160,
                "mma_regs": 48,
                "load_regs": 48,
                "padding_regs": 48,
                "workid_regs": 48,
                "gather_regs": 48,
                "use_tma_store": 1,
                "use_tma_oob_opt": 1,
                "use_early_exit": 1,
                "use_clc_fast_drain": 0,
                "use_global_scales": 1,
                "use_per_token_sf_a": 0,
                "use_per_token_sf_b": 0,
                "use_max_tmem_overlap": 0,
                "use_unroll_loop_2x_for_mma": 0,
                "has_gemm1_clamp_limit": 1,
            },
        },
    ),
]

def _run_large_mnk_case(selection, variant_name, variant):
    kwargs = dict(variant["kwargs"])
    for stage_key in (
        "num_stages_a",
        "num_stages_b",
        "num_stages_smem_sfa",
        "num_stages_smem_sfb",
        "num_stages_tmem_sfa",
        "num_stages_tmem_sfb",
    ):
        if stage_key in kwargs:
            kwargs[stage_key] = min(kwargs[stage_key], 3)

    print(f"large_mnk {selection}: {variant_name}", flush=True)
    benchmark(
        num_experts=256,
        num_tokens=1024,
        top_k=8,
        problem_n=variant["problem_n"],
        problem_k=variant["problem_k"],
        warmup_iters=1,
        bench_iters=1,
        num_rotated_buffers=0,
        **kwargs,
    )

def test_large_mnk_variants_do_not_hang():
    for selection, variant_name, variant in LARGE_MNK_CASES:
        _run_large_mnk_case(selection, variant_name, variant)

def test_swap_ab_hidden_m_workqueue_without_throttle_does_not_hang():
    """Focused persistent CLC repro for many hidden-M work IDs in swapAB mode."""
    benchmark(
        num_experts=1,
        num_tokens=1,
        top_k=1,
        problem_n=8192,
        problem_k=512,
        warmup_iters=0,
        bench_iters=1,
        num_rotated_buffers=0,
        use_cuda_graphs=False,
        gemm1_clamp_limit_value=2.0,
        batch_mode=int(BatchMode.BATCH_N),
        route_act=int(RouteImpl.TMA),
        route_sfs_act=int(RouteImpl.LDGSTS),
        tile_scheduler=int(TileScheduler.PERSISTENT),
        act_kind=int(ActKind.SWIGLU),
        sf_layout_a=int(SfLayout.R128c4),
        sf_layout_b=int(SfLayout.LINEAR),
        sf_layout_c=int(SfLayout.R128c4),
        cluster_m=2,
        tile_m=128,
        tile_n=128,
        tile_k=256,
        epi_tile_n=32,
        mma_m=256,
        mma_n=128,
        mma_k=32,
        dtype_a=int(DType.MXE2M1),
        dtype_b=int(DType.MXE4M3),
        dtype_c=int(DType.MXE4M3),
        sf_block_size_a=32,
        sf_block_size_b=32,
        sf_block_size_c=32,
        num_stages_a=4,
        num_stages_b=4,
        num_stages_smem_sfa=4,
        num_stages_smem_sfb=4,
        num_stages_tmem_sfa=4,
        num_stages_tmem_sfb=4,
        num_stages_tmem_acc=2,
        use_unroll_loop_2x_for_mma=0,
        use_early_exit=0,
        use_clc_fast_drain=0,
        use_work_throttle=0,
        bias_type=int(BiasType.M),
        has_gemm1_clamp_limit=1,
    )

def test_mxe4m3_ldgsts_routed_sfs_does_not_hang():
    """Focused GPT-OSS MXE4M3 route-SF LDGSTS repro."""
    benchmark(
        num_experts=4,
        num_tokens=256,
        top_k=4,
        problem_n=2048,
        problem_k=7168,
        warmup_iters=0,
        bench_iters=1,
        num_rotated_buffers=0,
        use_cuda_graphs=False,
        gemm1_clamp_limit_value=2.0,
        batch_mode=int(BatchMode.BATCH_N),
        route_act=int(RouteImpl.TMA),
        route_sfs_act=int(RouteImpl.LDGSTS),
        tile_scheduler=int(TileScheduler.PERSISTENT),
        act_kind=int(ActKind.SWIGLU),
        bias_type=int(BiasType.M),
        has_gemm1_clamp_limit=1,
        cluster_m=2,
        copy_sf_regs=48,
        dtype_a=int(DType.MXE4M3),
        dtype_b=int(DType.MXE4M3),
        dtype_c=int(DType.MXE4M3),
        epi_tile_n=32,
        epilogue_regs=152,
        load_b_regs=48,
        load_regs=48,
        load_sf_regs=48,
        load_sfb_regs=48,
        mma_k=32,
        mma_m=256,
        mma_n=128,
        mma_regs=48,
        num_load_b_warps=8,
        num_load_sfb_warps=4,
        num_stages_a=7,
        num_stages_b=7,
        num_stages_smem_sfa=7,
        num_stages_smem_sfb=7,
        num_stages_tmem_acc=2,
        num_stages_tmem_sfa=7,
        num_stages_tmem_sfb=7,
        padding_regs=48,
        sf_block_size_a=32,
        sf_block_size_b=32,
        sf_block_size_c=32,
        sf_layout_a=int(SfLayout.R128c4),
        sf_layout_b=int(SfLayout.LINEAR),
        sf_layout_c=int(SfLayout.R128c4),
        tile_k=128,
        tile_m=128,
        tile_n=128,
        use_clc_fast_drain=0,
        use_early_exit=1,
        use_max_tmem_overlap=0,
        use_tma_oob_opt=1,
        use_tma_store=1,
        use_unroll_loop_2x_for_mma=0,
        workid_regs=48,
    )
