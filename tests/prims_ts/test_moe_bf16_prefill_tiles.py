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

"""BF16 token-tile geometry and shuffled BatchM layout regression checks."""

import pytest
import torch

from flashinfer.prims_ts.moe.config_mapper import (
    map_trtllm_bf16_moe_tactic,
    valid_prims_ts_bf16_moe_tactics,
)
from flashinfer.prims_ts.batched_gemm.batched_gemm_config import validate_config
from flashinfer.utils import is_sm100a_supported


_PREFILL_SHAPE = dict(num_tokens=8192, top_k=12, num_local_experts=256, sm_arch=100)


def test_bf16_prefill_tactics_have_consistent_token_geometry():
    tile_ns = set()
    for tactic in valid_prims_ts_bf16_moe_tactics(**_PREFILL_SHAPE):
        pair = map_trtllm_bf16_moe_tactic(tactic, **_PREFILL_SHAPE)
        configs = (pair.fc1.cfg.build(), pair.fc2.cfg.build())
        assert configs[0].batch_mode == configs[1].batch_mode
        for cfg in configs:
            validate_config(cfg)
            assert cfg.tile_token == tactic[0]
            tile_ns.add(cfg.tile_n)
    assert 192 in tile_ns


@pytest.mark.parametrize("bias_fc", ["fc1_has_bias", "fc2_has_bias"])
def test_biased_bf16_tactics_exclude_batch_m(bias_fc):
    for tactic in valid_prims_ts_bf16_moe_tactics(**_PREFILL_SHAPE, **{bias_fc: True}):
        pair = map_trtllm_bf16_moe_tactic(tactic, **_PREFILL_SHAPE, **{bias_fc: True})
        assert pair.fc1.cfg.build().is_swap_ab


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required")
@pytest.mark.parametrize("output_dtype", ["bf16", "fp16"])
@pytest.mark.parametrize("epi_tile_n", [32, 64])
def test_batch_n_deferred_store_wait_tails(output_dtype, epi_tile_n):
    if not is_sm100a_supported(torch.device("cuda")):
        pytest.skip("Requires SM100-family GPU")
    from flashinfer.prims_ts.batched_gemm.batched_gemm_run import reference_check
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import DType

    shape = dict(num_tokens=8192, top_k=12, num_local_experts=256)
    for tactic in valid_prims_ts_bf16_moe_tactics(**shape):
        spec = map_trtllm_bf16_moe_tactic(tactic, **shape).fc2.cfg
        cfg = spec.build()
        if cfg.is_swap_ab and cfg.tile_n == 192:
            kwargs = dict(spec.kwargs)
            kwargs.update(
                num_stages_c_smem=2,
                epi_tile_n=epi_tile_n,
                dtype_c=int(DType.BF16 if output_dtype == "bf16" else DType.FP16),
            )
            # Partial expert tiles and repeated persistent work exercise both
            # scratch reuse and the last outstanding TMA store at task exit.
            assert reference_check(
                num_experts=7,
                num_tokens=1025,
                top_k=3,
                problem_n=2048,
                problem_k=512,
                repeat_launches=10,
                output_guard_elements=256,
                **kwargs,
            )
            return
    pytest.fail("No N192 BatchN tactic found for the deferred-store test")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required")
@pytest.mark.parametrize("tile_n", [128, 192])
@pytest.mark.parametrize("fc", ["fc1", "fc2"])
def test_bf16_prefill_kernel_reference_with_tail(tile_n, fc):
    if not is_sm100a_supported(torch.device("cuda")):
        pytest.skip("Requires SM100-family GPU")
    from flashinfer.prims_ts.batched_gemm.batched_gemm_run import reference_check

    shape = dict(num_tokens=8192, top_k=12, num_local_experts=256)
    for tactic in valid_prims_ts_bf16_moe_tactics(**shape):
        spec = getattr(map_trtllm_bf16_moe_tactic(tactic, **shape), fc).cfg
        if spec.build().tile_n == tile_n:
            # A non-divisible token count exercises partial token tiles.
            assert reference_check(
                num_experts=7,
                num_tokens=1025,
                top_k=3,
                problem_n=1024 if fc == "fc1" else 2048,
                problem_k=2048 if fc == "fc1" else 512,
                repeat_launches=3,
                output_guard_elements=256,
                **spec.kwargs,
            )
            return
    pytest.fail(f"Missing BatchN tile_n={tile_n} {fc} config")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required")
@pytest.mark.parametrize("tile_n", [128, 256])
@pytest.mark.parametrize("fc", ["fc1", "fc2"])
def test_bf16_prefill_batch_m_kernel_reference_with_tail(tile_n, fc):
    if not is_sm100a_supported(torch.device("cuda")):
        pytest.skip("Requires SM100-family GPU")
    from flashinfer.prims_ts.batched_gemm.batched_gemm_run import reference_check

    shape = dict(num_tokens=8192, top_k=12, num_local_experts=256)
    for tactic in valid_prims_ts_bf16_moe_tactics(**shape):
        spec = getattr(map_trtllm_bf16_moe_tactic(tactic, **shape), fc).cfg
        cfg = spec.build()
        if not cfg.is_swap_ab and cfg.tile_n == tile_n and cfg.epi_tile_n == 128:
            assert cfg.num_stages_tmem_acc == 2
            assert cfg.use_shuffled_matrix == 1
            assert reference_check(
                num_experts=7,
                num_tokens=1025,
                top_k=3,
                problem_n=1024 if fc == "fc1" else 2048,
                problem_k=2048 if fc == "fc1" else 512,
                repeat_launches=3,
                output_guard_elements=256,
                **spec.kwargs,
            )
            return
    pytest.fail(f"Missing BatchM tile_n={tile_n} {fc} config")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required")
@pytest.mark.parametrize("fc", ["fc1", "fc2"])
def test_one_stage_work_queue_does_not_alias_live_epilogue(fc):
    if not is_sm100a_supported(torch.device("cuda")):
        pytest.skip("Requires SM100-family GPU")
    from flashinfer.prims_ts.batched_gemm.batched_gemm_run import reference_check

    pair = map_trtllm_bf16_moe_tactic(
        [192, 0], num_tokens=8192, top_k=12, num_local_experts=256
    )
    kwargs = dict(getattr(pair, fc).cfg.kwargs, num_stages_workid=1)
    # Several work tiles per resident CTA expose next-tile operand prefetch.
    assert reference_check(
        num_experts=7,
        num_tokens=4097,
        top_k=3,
        problem_n=1024 if fc == "fc1" else 2048,
        problem_k=2048 if fc == "fc1" else 512,
        repeat_launches=5,
        output_guard_elements=256,
        **kwargs,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required")
@pytest.mark.parametrize("dtype", ["fp8", "nvfp4", "mxfp4", "mxfp8"])
@pytest.mark.parametrize("gated", [False, True])
def test_shuffled_batch_m_is_not_bf16_specific(dtype, gated):
    _check_shuffled_batch_m(dtype, gated=gated)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required")
@pytest.mark.parametrize("tile_k", [256, 512])
@pytest.mark.parametrize("umma_cta_count", [1, 2])
def test_nvfp4_batch_m_sfb_copy_tile_geometry(tile_k, umma_cta_count):
    # N128 needs a contiguous SFB copy for both four and eight K groups.
    _check_shuffled_batch_m("nvfp4", 128, tile_k, umma_cta_count)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required")
@pytest.mark.parametrize("gated", [False, True])
def test_nvfp4_batch_m_wide_sfb_tma(gated):
    _check_shuffled_batch_m("nvfp4", 256, 256, 1, gated=gated)


def _check_shuffled_batch_m(
    dtype, tile_n=128, tile_k=None, umma_cta_count=2, gated=False
):
    if not is_sm100a_supported(torch.device("cuda")):
        pytest.skip("Requires SM100-family GPU")
    from flashinfer.prims_ts.batched_gemm.batched_gemm_run import reference_check
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
        BatchMode,
        DType,
        SfLayout,
        uniform_pipeline_stage_overrides,
    )

    input_dtype = {
        "fp8": DType.E4M3,
        "nvfp4": DType.E2M1,
        "mxfp4": DType.MXE2M1,
        "mxfp8": DType.MXE4M3,
    }[dtype]
    input_is_8bit = dtype in ("fp8", "mxfp8")
    assert reference_check(
        num_experts=3,
        num_tokens=769,
        top_k=2,
        problem_n=512,
        problem_k=512,
        repeat_launches=3,
        output_guard_elements=256,
        batch_mode=int(BatchMode.BATCH_M),
        transpose_mma_output=0,
        use_shuffled_matrix=1,
        umma_cta_count=umma_cta_count,
        mma_m=128 * umma_cta_count,
        tile_n=tile_n,
        mma_n=tile_n,
        epi_tile_m=128,
        epi_tile_n=64,
        tile_k=tile_k or (128 if input_is_8bit else 256),
        mma_k=64 if dtype == "nvfp4" else 32,
        dtype_a=int(input_dtype),
        dtype_b=int(input_dtype),
        dtype_c=int(DType.BF16),
        sf_layout_b=int(SfLayout.R128c4),
        route_act=0,
        act_kind=1 if gated else 0,
        use_tma_store=1,
        tile_scheduler=1,
        num_stages_tmem_acc=2 if tile_n == 128 else 1,
        use_unroll_loop_2x_for_mma=0,
        **uniform_pipeline_stage_overrides(
            2 if tile_n > 128 or tile_k == 512 or dtype in ("mxfp4", "mxfp8") else 4
        ),
    )
