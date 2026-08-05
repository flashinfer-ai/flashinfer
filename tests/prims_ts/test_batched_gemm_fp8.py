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

"""Tests for plain FP8 per-tensor BatchedGemm TS kernels.

The variants mirror ``batched_gemm_config_fi_fp8.json``.
"""

import pytest
import torch

from flashinfer.utils import is_sm100a_supported

from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
    uniform_pipeline_stage_overrides,
)

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required"),
    pytest.mark.skipif(
        (torch.cuda.is_available() and not is_sm100a_supported(torch.device("cuda"))),
        reason="kernels require Blackwell sm_100+ (skip on sm_120a)",
    ),
]

def _common_fp8_base(**overrides):
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
        BatchMode,
        DType,
        TileScheduler,
    )

    cfg = dict(
        batch_mode=int(BatchMode.BATCH_N),
        tile_scheduler=int(TileScheduler.PERSISTENT),
        dtype_a=int(DType.E4M3),
        dtype_b=int(DType.E4M3),
        tile_m=128,
        epi_tile_m=128,
        mma_k=32,
        num_stages_tmem_acc=2,
        use_tma_store=1,
        use_tma_oob_opt=1,
        use_global_scales=1,
        use_unroll_loop_2x_for_mma=0,
    )
    cfg.update(overrides)
    return cfg

def _fc2_variant(
    tile_n,
    tile_k,
    pipeline_stages,
    epi_tile_n,
    mma_m,
    cluster_m,
    *,
    use_per_token_sf_a=0,
    per_token_sf_dtype=None,
):
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
        ActKind,
        DType,
        RouteImpl,
    )

    per_token_sf_dtype = (
        int(DType.BF16) if per_token_sf_dtype is None else int(per_token_sf_dtype)
    )
    return _common_fp8_base(
        route_act=int(RouteImpl.NONE),
        act_kind=int(ActKind.NONE),
        dtype_c=int(DType.BF16),
        tile_n=tile_n,
        mma_n=tile_n,
        epi_tile_n=epi_tile_n,
        tile_k=tile_k,
        **uniform_pipeline_stage_overrides(pipeline_stages),
        mma_m=mma_m,
        cluster_m=cluster_m,
        use_per_token_sf_a=use_per_token_sf_a,
        per_token_sf_dtype=per_token_sf_dtype,
    )

def _fc1_variant(
    tile_n,
    tile_k,
    pipeline_stages,
    epi_tile_n,
    mma_m,
    cluster_m,
    *,
    use_per_token_sf_a=0,
    use_per_token_sf_b=0,
    per_token_sf_dtype=None,
):
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
        ActKind,
        DType,
        RouteImpl,
    )

    per_token_sf_dtype = (
        int(DType.BF16) if per_token_sf_dtype is None else int(per_token_sf_dtype)
    )
    return _common_fp8_base(
        route_act=int(RouteImpl.TMA),
        act_kind=int(ActKind.SWIGLU),
        dtype_c=int(DType.E4M3),
        tile_n=tile_n,
        mma_n=tile_n,
        epi_tile_n=epi_tile_n,
        tile_k=tile_k,
        **uniform_pipeline_stage_overrides(pipeline_stages),
        mma_m=mma_m,
        cluster_m=cluster_m,
        use_per_token_sf_a=use_per_token_sf_a,
        use_per_token_sf_b=use_per_token_sf_b,
        per_token_sf_dtype=per_token_sf_dtype,
    )

def _fp8_relu2_variant(
    tile_n,
    tile_k,
    pipeline_stages,
    epi_tile_n,
    mma_m,
    cluster_m,
    *,
    use_per_token_sf_a=0,
    use_per_token_sf_b=0,
    per_token_sf_dtype=None,
):
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
        ActKind,
        DType,
        RouteImpl,
    )

    per_token_sf_dtype = (
        int(DType.BF16) if per_token_sf_dtype is None else int(per_token_sf_dtype)
    )
    return _common_fp8_base(
        route_act=int(RouteImpl.TMA),
        act_kind=int(ActKind.RELU2),
        dtype_c=int(DType.E4M3),
        tile_n=tile_n,
        mma_n=tile_n,
        epi_tile_n=epi_tile_n,
        tile_k=tile_k,
        **uniform_pipeline_stage_overrides(pipeline_stages),
        mma_m=mma_m,
        cluster_m=cluster_m,
        use_per_token_sf_a=use_per_token_sf_a,
        use_per_token_sf_b=use_per_token_sf_b,
        per_token_sf_dtype=per_token_sf_dtype,
    )

def _non_swap_fp8_fc1_variant(*, use_tma_store):
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import BatchMode

    return _fc1_variant(
        16,
        128,
        5,
        16,
        128,
        1,
    ) | {
        "batch_mode": int(BatchMode.BATCH_M),
        "transpose_mma_output": 0,
        "use_tma_store": use_tma_store,
    }

def _non_swap_fp8_relu2_variant(*, use_tma_store):
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import BatchMode

    return _fp8_relu2_variant(
        16,
        128,
        5,
        16,
        128,
        1,
    ) | {
        "batch_mode": int(BatchMode.BATCH_M),
        "transpose_mma_output": 0,
        "use_tma_store": use_tma_store,
    }

def _deepseek_fp8_fc1_variant():
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
        ActKind,
        DType,
        RouteImpl,
        TileScheduler,
    )

    return _common_fp8_base(
        route_act=int(RouteImpl.TMA),
        tile_scheduler=int(TileScheduler.STATIC),
        act_kind=int(ActKind.NONE),
        dtype_c=int(DType.E4M3),
        tile_n=8,
        mma_n=8,
        epi_tile_n=8,
        tile_k=128,
        **uniform_pipeline_stage_overrides(4),
        mma_m=64,
        cluster_m=1,
        use_deepseek_fp8=1,
        num_load_sfab_warps=1,
        num_load_b_warps=2,
        epilogue_regs=160,
        mma_regs=48,
        load_regs=48,
        padding_regs=48,
        workid_regs=48,
        load_sfab_regs=48,
    )

def _deepseek_fp8_fc1_bf16_output_variant():
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
        DType,
    )

    cfg = _deepseek_fp8_fc1_variant()
    cfg.update(
        dtype_c=int(DType.BF16),
        num_stages_c_smem=1,
    )
    return cfg

def _deepseek_fp8_fc2_variant():
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
        ActKind,
        DType,
        RouteImpl,
        TileScheduler,
    )

    return _common_fp8_base(
        route_act=int(RouteImpl.NONE),
        tile_scheduler=int(TileScheduler.STATIC),
        act_kind=int(ActKind.NONE),
        dtype_c=int(DType.BF16),
        tile_n=8,
        mma_n=8,
        epi_tile_n=8,
        tile_k=128,
        **uniform_pipeline_stage_overrides(4),
        mma_m=64,
        cluster_m=1,
        use_deepseek_fp8=1,
        num_load_sfab_warps=1,
        num_stages_c_smem=1,
        epilogue_regs=160,
        mma_regs=48,
        load_regs=48,
        padding_regs=48,
        workid_regs=48,
        load_sfab_regs=48,
    )


FP8_CONFIG_VARIANTS = [
    pytest.param(
        "fc1_ll_n8",
        _fc1_variant(8, 256, 6, 8, 128, 1),
        id="fc1_ll_n8",
    ),
    pytest.param(
        "fc1_ll_n16",
        _fc1_variant(16, 256, 6, 16, 128, 1),
        id="fc1_ll_n16",
    ),
    pytest.param(
        "fc1_ll_n32",
        _fc1_variant(32, 256, 5, 32, 128, 1),
        id="fc1_ll_n32",
    ),
    pytest.param(
        "fc1_ht_n64",
        _fc1_variant(64, 256, 5, 64, 256, 2),
        id="fc1_ht_n64",
    ),
    pytest.param(
        "fc1_ht_n128",
        _fc1_variant(128, 128, 8, 64, 256, 2),
        id="fc1_ht_n128",
    ),
    pytest.param(
        "fc1_ht_n256",
        _fc1_variant(256, 128, 6, 64, 256, 2),
        id="fc1_ht_n256",
    ),
    pytest.param(
        "fc2_ll_n8",
        _fc2_variant(8, 256, 6, 8, 128, 1),
        id="fc2_ll_n8",
    ),
    pytest.param(
        "fc2_ll_n16",
        _fc2_variant(16, 256, 6, 16, 128, 1),
        id="fc2_ll_n16",
    ),
    pytest.param(
        "fc2_ll_n32",
        _fc2_variant(32, 256, 5, 32, 128, 1),
        id="fc2_ll_n32",
    ),
    pytest.param(
        "fc2_ht_n64",
        _fc2_variant(64, 128, 8, 64, 256, 2),
        id="fc2_ht_n64",
    ),
    pytest.param(
        "fc2_ht_n128",
        _fc2_variant(128, 128, 7, 32, 256, 2),
        id="fc2_ht_n128",
    ),
    pytest.param(
        "fc2_ht_n256",
        _fc2_variant(256, 128, 6, 32, 256, 2),
        id="fc2_ht_n256",
    ),
]


@pytest.mark.parametrize("name,cfg", FP8_CONFIG_VARIANTS)
def test_fp8_config_schedule_validates(name, cfg):
    from flashinfer.prims_ts.batched_gemm.batched_gemm_kernel import (
        build_batched_gemm_task_manager,
    )

    del name
    build_batched_gemm_task_manager(
        num_experts=2,
        num_tokens=16,
        top_k=1,
        **cfg,
    )

def test_fp8_fc2_low_latency_correctness_tile8():
    from flashinfer.prims_ts.batched_gemm.batched_gemm_run import (
        reference_check,
    )

    assert reference_check(
        num_experts=2,
        num_tokens=16,
        top_k=1,
        **_fc2_variant(8, 256, 6, 8, 128, 1),
    )

def test_fp8_fc1_low_latency_tma_route_correctness_tile8():
    from flashinfer.prims_ts.batched_gemm.batched_gemm_run import (
        reference_check,
    )

    assert reference_check(
        num_experts=2,
        num_tokens=16,
        top_k=1,
        **_fc1_variant(8, 256, 6, 8, 128, 1),
    )


@pytest.mark.parametrize("use_tma_store", (0, 1))
def test_fp8_fc1_non_swap_plain_output_correctness(use_tma_store):
    from flashinfer.prims_ts.batched_gemm.batched_gemm_run import reference_check

    assert reference_check(
        num_experts=2,
        num_tokens=16,
        top_k=1,
        problem_n=128,
        problem_k=128,
        **_non_swap_fp8_fc1_variant(use_tma_store=use_tma_store),
    )


@pytest.mark.parametrize("use_tma_store", (0, 1))
def test_fp8_relu2_non_swap_plain_output_correctness(use_tma_store):
    from flashinfer.prims_ts.batched_gemm.batched_gemm_run import reference_check

    assert reference_check(
        num_experts=2,
        num_tokens=16,
        top_k=1,
        problem_n=128,
        problem_k=128,
        **_non_swap_fp8_relu2_variant(use_tma_store=use_tma_store),
    )


def test_fp8_fc1_non_swap_per_token_sfb_correctness():
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import DType
    from flashinfer.prims_ts.batched_gemm.batched_gemm_run import reference_check

    cfg = _non_swap_fp8_fc1_variant(use_tma_store=0)
    cfg.update(use_per_token_sf_b=1, per_token_sf_dtype=int(DType.FP32))
    ok, out = reference_check(
        num_experts=2,
        num_tokens=16,
        top_k=1,
        problem_n=128,
        problem_k=128,
        scale_c_value=64.0,
        return_output=True,
        **cfg,
    )
    assert ok
    assert out.float().abs().max().item() > 0.0


def test_fp8_relu2_non_swap_tma_store_per_token_sfb_correctness():
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import DType
    from flashinfer.prims_ts.batched_gemm.batched_gemm_run import reference_check

    cfg = _non_swap_fp8_relu2_variant(use_tma_store=1)
    cfg.update(use_per_token_sf_b=1, per_token_sf_dtype=int(DType.FP32))
    assert reference_check(
        num_experts=2,
        num_tokens=16,
        top_k=1,
        problem_n=128,
        problem_k=128,
        **cfg,
    )


@pytest.mark.timeout(240)
def test_fp8_deepseek_fc1_low_latency_tma_route_correctness_tile8_small():
    from flashinfer.prims_ts.batched_gemm.batched_gemm_run import (
        reference_check,
    )

    assert reference_check(
        num_experts=1,
        num_tokens=1,
        top_k=1,
        problem_n=128,
        problem_k=128,
        **_deepseek_fp8_fc1_variant(),
    )


@pytest.mark.timeout(240)
def test_fp8_deepseek_fc1_low_latency_tma_route_bf16_output_correctness_tile8_small():
    from flashinfer.prims_ts.batched_gemm.batched_gemm_run import (
        reference_check,
    )

    assert reference_check(
        num_experts=1,
        num_tokens=1,
        top_k=1,
        problem_n=128,
        problem_k=128,
        **_deepseek_fp8_fc1_bf16_output_variant(),
    )


@pytest.mark.timeout(240)
def test_fp8_deepseek_fc2_low_latency_correctness_tile8_small():
    from flashinfer.prims_ts.batched_gemm.batched_gemm_run import (
        reference_check,
    )

    assert reference_check(
        num_experts=1,
        num_tokens=1,
        top_k=1,
        problem_n=128,
        problem_k=128,
        **_deepseek_fp8_fc2_variant(),
    )

def test_fp8_fc2_high_throughput_cluster2_correctness_tile64():
    from flashinfer.prims_ts.batched_gemm.batched_gemm_run import (
        reference_check,
    )

    assert reference_check(
        num_experts=2,
        num_tokens=64,
        top_k=1,
        **_fc2_variant(64, 128, 8, 64, 256, 2),
    )

def test_fp8_fc1_high_throughput_cluster2_tma_route_correctness_tile64():
    from flashinfer.prims_ts.batched_gemm.batched_gemm_run import (
        reference_check,
    )

    assert reference_check(
        num_experts=2,
        num_tokens=64,
        top_k=1,
        **_fc1_variant(64, 256, 5, 64, 256, 2),
    )

def test_fp8_fc1_low_latency_tma_route_per_token_sfb_schedule_validates():
    from flashinfer.prims_ts.batched_gemm.batched_gemm_kernel import (
        build_batched_gemm_task_manager,
    )

    build_batched_gemm_task_manager(
        num_experts=2,
        num_tokens=16,
        top_k=1,
        **_fc1_variant(8, 256, 6, 8, 128, 1, use_per_token_sf_b=1),
    )

def test_fp8_fc1_low_latency_tma_route_per_token_sfb_correctness_tile8():
    from flashinfer.prims_ts.batched_gemm.batched_gemm_run import (
        reference_check,
    )

    ok, out = reference_check(
        num_experts=2,
        num_tokens=16,
        top_k=1,
        scale_c_value=64.0,
        return_output=True,
        **_fc1_variant(8, 256, 6, 8, 128, 1, use_per_token_sf_b=1),
    )
    assert ok
    assert out.float().abs().max().item() > 0.0

def test_fp8_fc1_low_latency_tma_route_per_token_sfa_sfb_correctness_tile8():
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import DType
    from flashinfer.prims_ts.batched_gemm.batched_gemm_run import (
        reference_check,
    )

    ok, out = reference_check(
        num_experts=2,
        num_tokens=16,
        top_k=1,
        scale_c_value=64.0,
        return_output=True,
        **_fc1_variant(
            8,
            256,
            6,
            8,
            128,
            1,
            use_per_token_sf_a=1,
            use_per_token_sf_b=1,
            per_token_sf_dtype=int(DType.FP32),
        ),
    )
    assert ok
    assert out.float().abs().max().item() > 0.0

def test_fp8_fc2_low_latency_per_token_sfa_correctness_tile8():
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import DType
    from flashinfer.prims_ts.batched_gemm.batched_gemm_run import (
        reference_check,
    )

    assert reference_check(
        num_experts=2,
        num_tokens=16,
        top_k=1,
        **_fc2_variant(
            8,
            256,
            6,
            8,
            128,
            1,
            use_per_token_sf_a=1,
            per_token_sf_dtype=int(DType.FP32),
        ),
    )

def test_fp8_relu2_plain_fp8_output_scale_c_nonzero_correctness_tile16():
    from flashinfer.prims_ts.batched_gemm.batched_gemm_run import (
        reference_check,
    )

    ok, out = reference_check(
        num_experts=2,
        num_tokens=16,
        top_k=1,
        problem_n=128,
        problem_k=256,
        scale_c_value=64.0,
        return_output=True,
        **_fp8_relu2_variant(16, 256, 6, 16, 128, 1, use_per_token_sf_b=1),
    )
    assert ok
    assert out.float().abs().max().item() > 0.25

def test_fp8_relu2_per_token_sfb_correctness_multi_epilogue_subtile():
    from flashinfer.prims_ts.batched_gemm.batched_gemm_run import (
        reference_check,
    )

    assert reference_check(
        num_experts=2,
        num_tokens=32,
        top_k=1,
        problem_n=128,
        problem_k=128,
        scale_c_value=64.0,
        **_fp8_relu2_variant(32, 128, 5, 16, 128, 1, use_per_token_sf_b=1),
    )
