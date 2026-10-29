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

from dataclasses import replace

import pytest
import torch

from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
    DType,
    RouteImpl,
    TileScheduler,
)
from flashinfer.prims_ts.moe.config_mapper import (
    _DType,
    _json_config_by_global_index,
    _json_config_kwargs,
    _make_json_moe_config_pair,
    _pdl_overrides,
    map_trtllm_nvfp4_moe_tactic,
    valid_prims_ts_nvfp4_moe_tactics,
)
from flashinfer.tllm_enums import ActivationType
from flashinfer.utils import is_sm100a_supported


_requires_sm100_or_sm103_tactic = pytest.mark.skipif(
    torch.cuda.is_available() and torch.cuda.get_device_capability() == (10, 7),
    reason="This test asserts SM100/SM103 tactic geometry; SM107 uses Rubin tactics",
)


def _find_bs1_ldgsts_persistent_pair(sm_arch: int = 100, *, enable_pdl: bool = False):
    tactics = valid_prims_ts_nvfp4_moe_tactics(
        sm_arch=sm_arch,
        num_tokens=1,
        top_k=8,
        num_local_experts=256,
    )

    for tactic in tactics:
        pair = _make_json_moe_config_pair(
            sm_arch=sm_arch,
            tile_n=tactic[0],
            moe_config_index=tactic[1],
            activation_type=int(ActivationType.Swiglu),
            dtype_a=int(_DType.E2M1),
            dtype_b=int(_DType.E2M1),
            fc1_dtype_c=int(_DType.E2M1),
            fc2_dtype_c=int(_DType.BF16),
            dtype_label="NVFP4xNVFP4",
            enable_pdl=enable_pdl,
        )
        if pair is None:
            continue
        fc1 = pair.fc1.cfg.kwargs
        fc2 = pair.fc2.cfg.kwargs
        if (
            fc1["tile_n"] == 8
            and fc1["tile_k"] == 512
            and fc1["umma_cta_count"] == 1
            and fc1["route_act"] == int(RouteImpl.LDGSTS)
            and fc1["route_sfs_act"] == int(RouteImpl.LDGSTS)
            and fc1["tile_scheduler"] == int(TileScheduler.PERSISTENT)
            and fc1["use_clc_fast_drain"] == 1
            and fc1["use_work_throttle"] == 1
            and fc2["tile_n"] == 8
            and fc2["tile_k"] == 512
            and fc2["umma_cta_count"] == 1
            and fc2["num_stages_a"] == 4
            and fc2["tile_scheduler"] == int(TileScheduler.PERSISTENT)
            and fc2["use_clc_fast_drain"] == 0
            and fc2["use_work_throttle"] == 0
        ):
            return pair

    return None


def test_nvfp4_search_contains_bs1_ldgsts_persistent_pair():
    assert _find_bs1_ldgsts_persistent_pair() is not None


@pytest.mark.parametrize("enable_pdl", (False, True))
def test_nvfp4_moe_pair_derives_skip_early_exit_pdl(enable_pdl):
    pair = _find_bs1_ldgsts_persistent_pair(enable_pdl=enable_pdl)
    assert pair is not None

    fc1 = pair.fc1.cfg.kwargs
    fc2 = pair.fc2.cfg.kwargs
    assert fc1["use_pdl"] == int(enable_pdl)
    assert fc1["skip_early_exit_pdl"] == 0
    assert fc2["use_pdl"] == int(enable_pdl)
    assert fc2["skip_early_exit_pdl"] == int(enable_pdl)


def test_fc2_without_early_exit_does_not_skip_pdl_wait():
    overrides = _pdl_overrides(
        {"use_early_exit": 0},
        enable_pdl=True,
        skip_early_exit_pdl=True,
    )
    assert overrides == {"use_pdl": 1, "skip_early_exit_pdl": 0}


@pytest.mark.parametrize(
    ("json_value", "raises"),
    (
        pytest.param(False, False, id="false-opt-out"),
        pytest.param(True, True, id="true-rejected"),
    ),
)
def test_nvfp4_fc2_json_can_only_opt_out_of_skip_early_exit_pdl(json_value, raises):
    pair = _find_bs1_ldgsts_persistent_pair()
    assert pair is not None
    json_cfg = _json_config_by_global_index(
        pair.fc2.prims_ts_gemm_config_index,
        sm_arch=100,
    )
    json_cfg = replace(
        json_cfg,
        options={**json_cfg.options, "skip_early_exit_pdl": json_value},
    )

    def mapped_kwargs():
        return _json_config_kwargs(
            json_cfg,
            sm_arch=100,
            fc="fc2",
            activation_type=int(ActivationType.Swiglu),
            has_bias=False,
            default_dtype_a=int(_DType.E2M1),
            default_dtype_b=int(_DType.E2M1),
            default_dtype_c=int(_DType.BF16),
        )

    if raises:
        with pytest.raises(ValueError, match="may only set it to false"):
            mapped_kwargs()
        return

    kwargs = mapped_kwargs()
    assert kwargs["skip_early_exit_pdl"] == 0
    assert _pdl_overrides(
        kwargs,
        enable_pdl=True,
        skip_early_exit_pdl=True,
    ) == {"use_pdl": 1, "skip_early_exit_pdl": 0}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required")
@pytest.mark.skipif(
    torch.cuda.is_available() and not is_sm100a_supported(torch.device("cuda")),
    reason="NVFP4 PrimsTS kernels require Blackwell SM100A+",
)
@_requires_sm100_or_sm103_tactic
@pytest.mark.parametrize(
    (
        "stage",
        "problem_n",
        "problem_k",
        "num_experts",
        "top_k",
        "early_exit_max_token_ctas",
        "dtype_c_override",
        "seed",
    ),
    (
        pytest.param(
            "fc1", 4096, 7168, 16, 8, 16, int(DType.BF16), 42, id="fc1-mainloop"
        ),
        pytest.param("fc1", 256, 512, 2, 1, 4, None, 123, id="fc1-quantized-output"),
        pytest.param("fc2", 7168, 2048, 16, 8, 16, None, 42, id="fc2"),
    ),
)
def test_bs1_deepseek_persistent_pair_gpu_correctness(
    stage,
    problem_n,
    problem_k,
    num_experts,
    top_k,
    early_exit_max_token_ctas,
    dtype_c_override,
    seed,
):
    """Exercise the BS=1 tactic, including skipped persistent work tiles."""
    from flashinfer.prims_ts.batched_gemm.batched_gemm_run import reference_check

    pair = _find_bs1_ldgsts_persistent_pair()
    assert pair is not None
    cfg = dict(pair.fc1.cfg.kwargs if stage == "fc1" else pair.fc2.cfg.kwargs)
    if dtype_c_override is not None:
        # The standalone random-input reference overstates E2M1 quantization
        # error for DeepSeek's large K. Use BF16 here to isolate the exact FC1
        # LDGSTS/persistent mainloop; the second FC1 row covers its real E2M1
        # output epilogue with the same tile and pipeline schedule.
        cfg["dtype_c"] = dtype_c_override

    assert reference_check(
        num_experts=num_experts,
        num_tokens=1,
        top_k=top_k,
        problem_n=problem_n,
        problem_k=problem_k,
        seed=seed,
        early_exit_max_token_ctas=early_exit_max_token_ctas,
        **cfg,
    )


@pytest.mark.parametrize("sm_arch", (100, 107))
def test_nvfp4_tile128_fused_tmem_allocation_has_no_standalone_sf_columns(sm_arch):
    from flashinfer.prims_ts.batched_gemm.batched_gemm_kernel import (
        _build_schedule_validate,
    )

    kwargs = dict(
        sm_arch=sm_arch,
        num_tokens=8192,
        top_k=8,
        num_local_experts=256,
        activation_type=int(ActivationType.Swiglu),
    )
    for tactic in valid_prims_ts_nvfp4_moe_tactics(**kwargs):
        if tactic[0] != 128:
            continue
        cfg = map_trtllm_nvfp4_moe_tactic(tactic, **kwargs).fc1.cfg.build()
        if (
            cfg.fuse_sf_copy_to_mma
            and not cfg.use_combined_sfab_copy
            and not cfg.use_tile256_tmem_overlap
        ):
            break
    else:
        pytest.fail(f"no tile-128 NVFP4 FC1 tactic with fused SF copy for sm_{sm_arch}")

    _, _, _, tmem_allocator = _build_schedule_validate(cfg)

    # A single SF slot follows the accumulator stages instead of per-stage SF rings.
    assert tmem_allocator.total_tmem_columns == cfg.tmem_required_cols
    assert cfg.tmem_required_cols == (
        cfg.tmem_c_cols_per_stage * cfg.num_stages_tmem_acc
        + cfg.tmem_sfa_cols
        + cfg.tmem_sfb_cols
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required")
@pytest.mark.skipif(
    torch.cuda.is_available() and not is_sm100a_supported(torch.device("cuda")),
    reason="NVFP4 PrimsTS kernels require Blackwell SM100A+",
)
@_requires_sm100_or_sm103_tactic
@pytest.mark.parametrize(
    ("problem_k", "dtype_c_override"),
    (
        pytest.param(512, None, id="quantized-output"),
        pytest.param(1024, int(DType.BF16), id="two-k-tiles"),
    ),
)
def test_nvfp4_tile128_fused_fc1_gpu_correctness(problem_k, dtype_c_override):
    from flashinfer.prims_ts.batched_gemm.batched_gemm_run import reference_check

    pair = map_trtllm_nvfp4_moe_tactic(
        [128, 1],
        num_tokens=8192,
        top_k=8,
        num_local_experts=256,
        activation_type=int(ActivationType.Swiglu),
    )
    cfg = dict(pair.fc1.cfg.kwargs)
    assert cfg["use_clc_fast_drain"]
    if dtype_c_override is not None:
        cfg["dtype_c"] = dtype_c_override

    assert reference_check(
        num_experts=2,
        num_tokens=257,
        top_k=1,
        problem_n=256,
        problem_k=problem_k,
        seed=123,
        early_exit_max_token_ctas=16,
        **cfg,
    )


def test_nvfp4_bs256_has_matching_gen_tile32_pair():
    tactics = valid_prims_ts_nvfp4_moe_tactics(
        sm_arch=100,
        num_tokens=256,
        top_k=8,
        num_local_experts=256,
        activation_type=int(ActivationType.Swiglu),
    )

    for tactic in tactics:
        pair = map_trtllm_nvfp4_moe_tactic(
            tactic,
            sm_arch=100,
            num_tokens=256,
            top_k=8,
            num_local_experts=256,
            activation_type=int(ActivationType.Swiglu),
        )
        fc1 = pair.fc1.cfg.build()
        fc2 = pair.fc2.cfg.build()
        if (
            fc1.tile_n == 32
            and fc1.tile_k == 512
            and fc1.route_act == int(RouteImpl.LDGSTS)
            and fc1.route_sfs_act == int(RouteImpl.LDGSTS)
            and fc1.num_stages_a == 4
            and fc2.tile_n == 32
            and fc2.tile_k == 512
            and fc2.umma_cta_count == 2
            and fc2.num_stages_a == 4
            and fc2.use_unroll_loop_2x_for_mma
            and fc1.tile_scheduler == int(TileScheduler.PERSISTENT)
            and fc1.use_clc_fast_drain
            and fc2.tile_scheduler == int(TileScheduler.PERSISTENT)
            and fc2.use_clc_fast_drain
        ):
            break
    else:
        pytest.fail("missing the Gen-matched NVFP4 BS=256 tile-32 config pair")
