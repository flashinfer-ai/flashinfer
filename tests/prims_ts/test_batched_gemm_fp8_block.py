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

"""Focused FP8 block-scale BatchedGemm tests for FlashInfer Prims-TS."""

import pytest
import torch

from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
    uniform_pipeline_stage_overrides,
)
from flashinfer.utils import get_compute_capability


pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required"),
]


def _require_sm100():
    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU required")
    major, _minor = get_compute_capability(torch.device("cuda"))
    if major != 10:
        pytest.skip("Prims-TS FP8 block-scale tests require an SM100-class GPU")


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
        epi_tile_m=128,
        mma_k=32,
        num_stages_tmem_acc=2,
        use_tma_store=1,
        use_tma_oob_opt=1,
        use_global_scales=1,
        use_unroll_loop_2x_for_mma=0,
        transpose_mma_output=1,
    )
    cfg.update(overrides)
    return cfg


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
        umma_cta_count=1,
        use_deepseek_fp8=1,
        num_load_b_warps=2,
        epilogue_regs=160,
        mma_regs=48,
        load_regs=48,
        padding_regs=48,
        workid_regs=48,
        load_sfab_regs=48,
    )


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
        umma_cta_count=1,
        use_deepseek_fp8=1,
        num_stages_c_smem=1,
        epilogue_regs=160,
        mma_regs=48,
        load_regs=48,
        padding_regs=48,
        workid_regs=48,
        load_sfab_regs=48,
    )


def _with_deepseek_tile(cfg, tile_n):
    cfg = dict(cfg)
    cfg.update(tile_n=tile_n, mma_n=tile_n, epi_tile_n=tile_n)
    return cfg


def _with_deepseek_block32(cfg):
    """[32,32] scales: four scale sub-blocks per 128-wide K tile."""
    cfg = dict(cfg)
    cfg.update(deepseek_fp8_block_size=32)
    return cfg


def _mx_fc1_base():
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
        ActKind,
        BiasType,
        DType,
        RouteImpl,
        SfLayout,
        TileScheduler,
    )

    return dict(
        batch_mode=0,
        route_act=int(RouteImpl.TMA),
        route_sfs_act=int(RouteImpl.LDGSTS),
        act_kind=int(ActKind.SWIGLU),
        mma_m=128,
        mma_k=32,
        dtype_a=int(DType.MXE4M3),
        dtype_b=int(DType.MXE4M3),
        dtype_c=int(DType.MXE4M3),
        sf_layout_a=int(SfLayout.R128c4),
        sf_layout_b=int(SfLayout.LINEAR),
        sf_layout_c=int(SfLayout.R8c4),
        use_tma_store=1,
        use_tma_oob_opt=1,
        tile_scheduler=int(TileScheduler.PERSISTENT),
        num_stages_tmem_acc=2,
        transpose_mma_output=1,
        bias_type=int(BiasType.M),
        has_gemm1_clamp_limit=1,
        epilogue_regs=160,
        mma_regs=48,
        load_regs=48,
        load_sf_regs=48,
        copy_sf_regs=48,
        workid_regs=48,
        padding_regs=48,
        load_b_regs=48,
    )


def _mx_fc2_base():
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
        BiasType,
        DType,
        SfLayout,
    )

    return dict(
        batch_mode=0,
        route_act=0,
        route_sfs_act=-1,
        act_kind=0,
        mma_m=128,
        mma_k=32,
        dtype_a=int(DType.MXE4M3),
        dtype_b=int(DType.MXE4M3),
        dtype_c=int(DType.BF16),
        sf_layout_a=int(SfLayout.R128c4),
        sf_layout_b=int(SfLayout.R8c4),
        epilogue_regs=128,
        mma_regs=48,
        load_regs=48,
        load_sf_regs=48,
        copy_sf_regs=48,
        workid_regs=48,
        padding_regs=48,
        transpose_mma_output=1,
        bias_type=int(BiasType.M),
        use_tma_oob_opt=1,
    )


@pytest.mark.parametrize("tile_n", [8, 16, 32, 64, 128])
def test_deepseek_fp8_fc1_correctness_small(tile_n):
    _require_sm100()
    from flashinfer.prims_ts.batched_gemm.batched_gemm_run import reference_check

    assert reference_check(
        num_experts=1,
        num_tokens=max(tile_n, 8),
        top_k=1,
        problem_n=128,
        problem_k=128,
        **_with_deepseek_tile(_deepseek_fp8_fc1_variant(), tile_n),
    )


@pytest.mark.parametrize("tile_n", [8, 16, 32, 64, 128])
def test_deepseek_fp8_fc2_correctness_small(tile_n):
    _require_sm100()
    from flashinfer.prims_ts.batched_gemm.batched_gemm_run import reference_check

    assert reference_check(
        num_experts=1,
        num_tokens=max(tile_n, 8),
        top_k=1,
        problem_n=128,
        problem_k=128,
        **_with_deepseek_tile(_deepseek_fp8_fc2_variant(), tile_n),
    )


def test_deepseek_fp8_fc1_partial_weight_tile():
    """M=320 (2 x 160) at block 128: the last 128-row weight tile is partial.

    swapAB zero-fills the missing rows through the A TMA map and predicates
    them in the epilogue, so whole-tile scales still cover the remainder.
    """
    _require_sm100()
    from flashinfer.prims_ts.batched_gemm.batched_gemm_run import reference_check

    assert reference_check(
        num_experts=2,
        num_tokens=8,
        top_k=1,
        problem_n=320,
        problem_k=256,
        **_with_deepseek_tile(_deepseek_fp8_fc1_variant(), 8),
    )


@pytest.mark.parametrize("tile_n", [8, 64, 128])
def test_deepseek_fp8_block32_fc2_correctness(tile_n):
    """FC2 at [32,32] scales with K=160: the TP4 contract's five K blocks."""
    _require_sm100()
    from flashinfer.prims_ts.batched_gemm.batched_gemm_run import reference_check

    assert reference_check(
        num_experts=2,
        num_tokens=max(tile_n, 8),
        top_k=1,
        problem_n=256,
        problem_k=160,
        **_with_deepseek_block32(
            _with_deepseek_tile(_deepseek_fp8_fc2_variant(), tile_n)
        ),
    )


@pytest.mark.parametrize("tile_n", [8, 64, 128])
def test_deepseek_fp8_block32_fc1_correctness(tile_n):
    """FC1 at [32,32] scales with M=320: ten row groups over three weight tiles.

    ``tile_n=64`` takes the atomic absmax scratch, the others the SMEM
    exchange path.
    """
    _require_sm100()
    from flashinfer.prims_ts.batched_gemm.batched_gemm_run import reference_check

    assert reference_check(
        num_experts=2,
        num_tokens=max(tile_n, 8),
        top_k=1,
        problem_n=320,
        problem_k=256,
        **_with_deepseek_block32(
            _with_deepseek_tile(_deepseek_fp8_fc1_variant(), tile_n)
        ),
    )


@pytest.mark.parametrize("tile_n", [8, 128])
@pytest.mark.parametrize(
    ("stage", "problem_n", "problem_k"),
    [
        pytest.param("fc1", 320, 2560, id="fc1"),
        pytest.param("fc2", 2560, 160, id="fc2"),
    ],
)
def test_deepseek_fp8_block32_moe_default_config_contract_shape(
    stage, problem_n, problem_k, tile_n
):
    """Each GEMM of the TP4 contract (H=2560, I=160) on the mapped MoE tactic.

    The small block-32 tests use a four-stage pipeline; the MoE rows stage
    five to eight K tiles and, at ``tile_n=128``, 2 KB of activation scales
    per stage.
    """
    _require_sm100()
    from flashinfer.prims_ts.batched_gemm.batched_gemm_run import reference_check
    from flashinfer.prims_ts.moe.config_mapper import (
        map_trtllm_deepseek_fp8_moe_tactic,
    )

    pair = map_trtllm_deepseek_fp8_moe_tactic(
        [tile_n, 0],
        num_tokens=256,
        top_k=10,
        num_local_experts=512,
        enable_pdl=True,
        deepseek_fp8_block_size=32,
    )
    config_spec = pair.fc1.cfg if stage == "fc1" else pair.fc2.cfg

    assert reference_check(
        num_experts=512,
        num_tokens=256,
        top_k=10,
        problem_n=problem_n,
        problem_k=problem_k,
        **config_spec.kwargs,
    )


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        pytest.param({"tile_k": 256}, "tile_k=128", id="tile_k-stays-128"),
        pytest.param({"mma_m": 128}, "split-M", id="block32-needs-split-m-epilogue"),
        pytest.param(
            {"deepseek_fp8_block_size": 64}, "must be 32 or 128", id="block-64"
        ),
    ],
)
def test_deepseek_fp8_block32_config_rejections(overrides, match):
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
        BatchedGemmConfig,
        validate_config,
    )

    cfg = _with_deepseek_block32(_with_deepseek_tile(_deepseek_fp8_fc2_variant(), 8))
    cfg.update(overrides)
    with pytest.raises(ValueError, match=match):
        validate_config(BatchedGemmConfig(**cfg))


@pytest.mark.parametrize(
    "num_tokens",
    [128, 100],
    ids=["full-tiles", "partial-tiles"],
)
@pytest.mark.parametrize(
    "token_capacity",
    [None, 1472],
    ids=["routed-capacity", "worst-case-capacity"],
)
@pytest.mark.parametrize(
    ("stage", "problem_n"),
    [
        pytest.param("fc1", 1024, id="fc1"),
        pytest.param("fc2", 512, id="fc2"),
    ],
)
def test_deepseek_fp8_moe_default_config_realistic_shape(
    stage, problem_n, token_capacity, num_tokens
):
    """Check each GEMM at the shape used by the DeepSeek MoE smoke test.

    The small tests above cover one expert, top-1 routing, and one 128-wide K
    tile. The MoE uses 64 experts, top-8 routing, K=512, and the mapped
    default tactic, which exercises substantially more routing metadata and
    pipeline iterations. Keeping FC1 and FC2 as separate parameter cases tells
    us whether either standalone GEMM fails before activation and finalization.

    128 tokens spread over 64 experts at top-8 gives every expert 16 tokens, a
    multiple of the token tile, so that routing has neither padding nor a
    partial tile and its padded count is exact. Real MoE routing has both, so
    the two extra axes reintroduce them: ``partial-tiles`` drops to 100 tokens,
    leaving each expert 12 or 13 of its 16 padded rows and every second tile
    partially valid, and ``worst-case-capacity`` declares a token extent wider
    than the routed count, which the MoE runner must do because that count is
    only known on the device.
    """
    _require_sm100()
    from flashinfer.prims_ts.batched_gemm.batched_gemm_run import reference_check
    from flashinfer.prims_ts.moe.config_mapper import (
        map_trtllm_deepseek_fp8_moe_tactic,
    )

    # Map at the MoE's token count regardless of the routing axis, so every
    # case runs the exact kernel configuration the MoE uses.
    pair = map_trtllm_deepseek_fp8_moe_tactic(
        [-1, -1],
        num_tokens=128,
        top_k=8,
        num_local_experts=64,
        enable_pdl=True,
    )
    config_spec = pair.fc1.cfg if stage == "fc1" else pair.fc2.cfg

    assert reference_check(
        num_experts=64,
        num_tokens=num_tokens,
        top_k=8,
        problem_n=problem_n,
        problem_k=512,
        token_capacity=token_capacity,
        **config_spec.kwargs,
    )


def test_mxfp8_mxfp8_fc2_tile8_correctness():
    _require_sm100()
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import TileScheduler
    from flashinfer.prims_ts.batched_gemm.batched_gemm_run import reference_check

    cfg = {
        **_mx_fc2_base(),
        "tile_n": 8,
        "mma_n": 8,
        "epi_tile_n": 8,
        "tile_k": 256,
        **uniform_pipeline_stage_overrides(6),
        "tile_scheduler": int(TileScheduler.STATIC),
        "num_stages_tmem_acc": 1,
    }
    assert reference_check(
        num_experts=2,
        num_tokens=128,
        top_k=1,
        problem_k=256,
        **cfg,
    )


def test_mxfp8_mxfp8_fc1_tile32_correctness():
    _require_sm100()
    from flashinfer.prims_ts.batched_gemm.batched_gemm_run import reference_check

    cfg = {
        **_mx_fc1_base(),
        "tile_n": 32,
        "mma_n": 32,
        "epi_tile_n": 32,
        "tile_k": 256,
        **uniform_pipeline_stage_overrides(5),
        "mma_m": 256,
        "umma_cta_count": 2,
    }
    assert reference_check(
        num_experts=2,
        num_tokens=256,
        top_k=1,
        problem_k=256,
        **cfg,
    )
