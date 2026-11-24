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

import pytest
import torch

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
from flashinfer.prims_ts.batched_gemm.config_types import (
    LocalitySideSource,
    LocalitySplit,
)
from flashinfer.utils import is_sm100a_supported


pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required"),
    pytest.mark.skipif(
        torch.cuda.is_available() and not is_sm100a_supported(torch.device("cuda")),
        reason="locality scheduler tests require SM100+",
    ),
]


def _bf16_persistent_reference_check(routing, **overrides):
    from flashinfer.prims_ts.batched_gemm.batched_gemm_run import reference_check

    kwargs = dict(
        num_experts=4,
        num_tokens=257,
        top_k=2,
        routing=routing,
        repeat_launches=3,
        tile_n=16,
        tile_k=128,
        mma_n=16,
        epi_tile_n=16,
        batch_mode=int(BatchMode.BATCH_N),
        transpose_mma_output=1,
        route_act=1,
        tile_scheduler=int(TileScheduler.PERSISTENT),
        use_early_exit=1,
        use_clc_fast_drain=0,
        act_kind=0,
        sf_layout_b=1,
        umma_cta_count=1,
        mma_m=128,
        mma_k=16,
        dtype_a=int(DType.BF16),
        dtype_b=int(DType.BF16),
        dtype_c=int(DType.BF16),
        epilogue_regs=160,
        mma_regs=48,
        load_regs=48,
        load_b_regs=48,
        padding_regs=48,
        use_unroll_loop_2x_for_mma=0,
        use_max_tmem_overlap=0,
        **uniform_pipeline_stage_overrides(4),
    )
    kwargs.update(overrides)
    return reference_check(**kwargs)


@pytest.mark.parametrize("routing", ("balanced_round_robin", "uniform"))
def test_bf16_persistent_baseline_matches_reference(routing):
    """The locality shape with the stock CLC scheduler, as a control."""
    assert _bf16_persistent_reference_check(routing)


@pytest.mark.parametrize(
    "side_source",
    (LocalitySideSource.ZERO, LocalitySideSource.SMID_PARITY),
    ids=("all_steal", "smid_parity"),
)
@pytest.mark.parametrize(
    "num_persistent_clusters",
    (1, None),
    ids=("forced_multiwave", "resident_grid"),
)
@pytest.mark.parametrize("routing", ("balanced_round_robin", "uniform"))
def test_bf16_locality_scheduler_matches_reference(
    side_source, num_persistent_clusters, routing
):
    assert _bf16_persistent_reference_check(
        routing,
        locality_split=int(LocalitySplit.WEIGHTS_OUTPUT),
        locality_side_source=int(side_source),
        num_persistent_clusters=num_persistent_clusters,
    )


@pytest.mark.timeout(600)
@pytest.mark.parametrize("umma_cta_count", (1, 2), ids=("1cta", "2cta"))
def test_bf16_locality_split_activations_matches_reference(umma_cta_count):
    """Side 1 loads its own activations; a 2-CTA cluster splits each B box."""
    assert _bf16_persistent_reference_check(
        "uniform",
        route_act=0,
        umma_cta_count=umma_cta_count,
        mma_m=128 * umma_cta_count,
        locality_split=int(LocalitySplit.WEIGHTS_ACTIVATIONS_OUTPUT),
        locality_side_source=int(LocalitySideSource.SMID_PARITY),
    )


@pytest.mark.timeout(600)
@pytest.mark.parametrize(
    "dtype_a,dtype_b,dtype_c,mma_k",
    (
        (DType.E2M1, DType.E2M1, DType.E2M1, 64),
        (DType.MXE2M1, DType.MXE4M3, DType.MXE4M3, 32),
    ),
    ids=("fp4_output", "mxfp8_output"),
)
def test_mxfp4_locality_side1_scale_and_quantized_output(
    dtype_a, dtype_b, dtype_c, mma_k
):
    """Side 1 uses doubled SFA and separate C/SF-C, so RELU2 yields 4x side 0.

    Non-gated quantized outputs require RELU2 or SILU, which rules out the
    negated-weight check used for linear outputs.
    """
    from flashinfer.prims_ts.batched_gemm.batched_gemm_run import reference_check

    assert reference_check(
        num_experts=2,
        num_tokens=129,
        top_k=1,
        routing="uniform",
        repeat_launches=2,
        problem_n=128,
        problem_k=256,
        batch_mode=int(BatchMode.BATCH_N),
        transpose_mma_output=1,
        route_act=int(RouteImpl.NONE),
        route_sfs_act=int(RouteImpl.NONE),
        act_kind=int(ActKind.RELU2),
        bias_type=int(BiasType.NONE),
        tile_n=8,
        tile_k=256,
        epi_tile_n=8,
        mma_m=128,
        mma_n=8,
        mma_k=mma_k,
        dtype_a=int(dtype_a),
        dtype_b=int(dtype_b),
        dtype_c=int(dtype_c),
        sf_layout_a=int(SfLayout.R128c4),
        sf_layout_b=int(SfLayout.R8c4),
        sf_layout_c=int(SfLayout.R8c4),
        tile_scheduler=int(TileScheduler.PERSISTENT),
        use_early_exit=1,
        use_clc_fast_drain=0,
        use_tma_store=1,
        use_tma_oob_opt=1,
        epilogue_regs=128,
        mma_regs=48,
        load_regs=48,
        load_sf_regs=48,
        cast_a_regs=160,
        copy_sf_regs=48,
        workid_regs=48,
        padding_regs=48,
        locality_split=int(LocalitySplit.WEIGHTS_OUTPUT),
        locality_side_source=int(LocalitySideSource.SMID_PARITY),
        **uniform_pipeline_stage_overrides(6),
    )
