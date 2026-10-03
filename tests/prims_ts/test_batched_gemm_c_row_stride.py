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

"""Strided C output: one partition writing into a wider shared buffer.

``c_row_stride`` decouples the C row pitch from a launch's own output width so
several partitions can fill one buffer. With the default pitch the two are
equal, which is why every other test would pass with a wrong pitch.
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

# swapAB, out_hidden defaults to tile_m; a gated epilogue halves the stored width.
FC2_WIDTH = 128
FC1_WIDTH = 64


def _common(**overrides):
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
        BatchMode,
        DType,
        TileScheduler,
    )

    cfg = dict(
        # c_row_stride is swapAB-only.
        batch_mode=int(BatchMode.BATCH_N),
        tile_scheduler=int(TileScheduler.PERSISTENT),
        dtype_a=int(DType.E4M3),
        dtype_b=int(DType.E4M3),
        tile_m=FC2_WIDTH,
        epi_tile_m=FC2_WIDTH,
        mma_k=32,
        num_stages_tmem_acc=2,
        use_global_scales=1,
        use_unroll_loop_2x_for_mma=0,
        tile_n=8,
        mma_n=8,
        epi_tile_n=8,
        tile_k=256,
        mma_m=128,
        cluster_m=1,
        per_token_sf_dtype=int(DType.BF16),
        **uniform_pipeline_stage_overrides(6),
    )
    cfg.update(overrides)
    return cfg


def _fc2_bf16_out(*, use_tma_store):
    """Non-gated with BF16 C; stored width == tile_m."""
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
        ActKind,
        DType,
        RouteImpl,
    )

    return _common(
        route_act=int(RouteImpl.NONE),
        act_kind=int(ActKind.NONE),
        dtype_c=int(DType.BF16),
        use_per_token_sf_a=0,
        use_tma_store=use_tma_store,
        use_tma_oob_opt=use_tma_store,
    )


def _fc1_fp8_out():
    """Gated SWIGLU with FP8 C; stored width == tile_m // 2.

    TMA-store only: swapAB gated FP8 aborts at launch with use_tma_store=0,
    with or without a declared pitch.  The BF16 cases cover the non-TMA store.
    """
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
        ActKind,
        DType,
        RouteImpl,
    )

    return _common(
        route_act=int(RouteImpl.TMA),
        act_kind=int(ActKind.SWIGLU),
        dtype_c=int(DType.E4M3),
        use_per_token_sf_a=0,
        use_per_token_sf_b=0,
        use_tma_store=1,
        use_tma_oob_opt=1,
    )


def _run(cfg, c_row_stride, offset=0):
    from flashinfer.prims_ts.batched_gemm.batched_gemm_run import reference_check

    return reference_check(
        num_experts=2,
        num_tokens=16,
        top_k=1,
        c_row_stride=c_row_stride,
        c_partition_offset=offset,
        **cfg,
    )


@pytest.mark.parametrize("use_tma_store", (0, 1))
def test_stride_equal_to_width_matches_default(use_tma_store):
    """Declaring the pitch explicitly must behave like the implicit default."""
    assert _run(_fc2_bf16_out(use_tma_store=use_tma_store), FC2_WIDTH)


@pytest.mark.parametrize("use_tma_store", (0, 1))
@pytest.mark.parametrize("stride", (160, 256))
def test_stride_wider_buffer_bf16_output(use_tma_store, stride):
    """Partition lands correctly and the slack keeps its sentinel."""
    assert _run(_fc2_bf16_out(use_tma_store=use_tma_store), stride)


def test_fp8_gated_baseline_no_stride():
    """Baseline: the config works without a declared pitch."""
    assert _run(_fc1_fp8_out(), None)


def test_stride_equal_to_width_fp8_gated():
    """Declaring the pitch explicitly must behave like the implicit default."""
    assert _run(_fc1_fp8_out(), FC1_WIDTH)


@pytest.mark.parametrize("stride", (96, 256))
def test_stride_wider_buffer_fp8_output_gated(stride):
    """Same, for FP8 C and a gated epilogue writing the half-width output."""
    assert _run(_fc1_fp8_out(), stride)


@pytest.mark.parametrize("use_tma_store", (0, 1))
def test_partition_offset_bf16(use_tma_store):
    """Partition at a non-zero offset, so sentinel sits on both sides of it."""
    assert _run(_fc2_bf16_out(use_tma_store=use_tma_store), 256, offset=128)


def test_partition_offset_fp8_gated():
    """Same, for the half-width gated FP8 output."""
    assert _run(_fc1_fp8_out(), 128, offset=64)


def test_partition_offset_past_pitch_rejected():
    with pytest.raises(ValueError, match="does not fit inside pitch"):
        _run(_fc2_bf16_out(use_tma_store=0), 160, offset=64)


def test_stride_narrower_than_width_rejected():
    with pytest.raises(ValueError, match="narrower than the output width"):
        _run(_fc2_bf16_out(use_tma_store=0), FC1_WIDTH)
