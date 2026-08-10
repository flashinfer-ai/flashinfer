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

from types import SimpleNamespace

import pytest
import torch

from flashinfer.prims_ts.moe.config_mapper import (
    SUPPORTED_MXFP4_BF16_TILE_N,
    _DType,
    _expanded_prims_ts_json_configs,
    _expanded_trtllm_gen_json_configs,
    _json_config_by_global_index,
    _make_json_moe_config_pair,
    map_trtllm_bf16_moe_tactic,
    map_trtllm_deepseek_fp8_moe_tactic,
    map_trtllm_mxfp4_mxfp8_moe_tactic,
    map_trtllm_mxfp4_bf16_moe_tactic,
    map_trtllm_nvfp4_moe_tactic,
    valid_prims_ts_mxfp4_mxfp8_moe_tactics,
    valid_prims_ts_mxfp8_mxfp8_moe_tactics,
)
from flashinfer.prims_ts.moe import runner as runner_module
from flashinfer.prims_ts.moe import support
from flashinfer.prims_ts.moe.runner import (
    PrimsTsMxfp4Mxfp8MoERunner,
    _filter_valid_moe_tactics,
)
from flashinfer.tllm_enums import (
    ActivationType,
    DtypeTrtllmGen,
    RoutingMethodType,
    WeightLayout,
)
from flashinfer.utils import is_sm100a_supported

def _runner(**overrides):
    values = dict(
        dtype_act=DtypeTrtllmGen.Bfloat16,
        dtype_weights=DtypeTrtllmGen.Bfloat16,
        activation_type=ActivationType.Swiglu,
        weight_layout=WeightLayout.MajorK,
        use_shuffled_weight=True,
        intermediate_size=256,
        use_per_token_scaling=False,
        top_k=8,
        num_local_experts=1,
        hidden_size=128,
    )
    values.update(overrides)
    return SimpleNamespace(**values)

def _inputs(**overrides):
    values = dict(
        hidden_states=torch.empty((4, 128), dtype=torch.bfloat16),
        gemm1_lora_delta=None,
        hidden_states_scale=None,
        per_token_scale=None,
    )
    values.update(overrides)
    return SimpleNamespace(**values)

def _first_buildable_pair(mapper, tile_n, **kwargs):
    for config_index in range(512):
        try:
            pair = mapper([tile_n, config_index], **kwargs)
            pair.fc1.cfg.build()
            pair.fc2.cfg.build()
        except Exception:
            continue
        return pair
    pytest.fail(f"no buildable tactic found for tile_N={tile_n}")

_MXFP4_MXFP8_FAST_DRAIN_FC1_COMMENT = "MxFp4xMxFp8_FC1_LowLatencyFastDrain"


def _mxfp4_mxfp8_json_pair(
    tactic,
    *,
    enable_pdl=False,
    activation_type=int(ActivationType.Swiglu),
    dtype_a=int(_DType.MXE2M1),
):
    return _make_json_moe_config_pair(
        tile_n=tactic[0],
        moe_config_index=tactic[1],
        activation_type=activation_type,
        dtype_a=dtype_a,
        dtype_b=int(_DType.MXE4M3),
        fc1_dtype_c=int(_DType.MXE4M3),
        fc2_dtype_c=int(_DType.BF16),
        dtype_label=("MXFP4xMXFP8" if dtype_a == int(_DType.MXE2M1) else "MXFP8xMXFP8"),
        enable_pdl=enable_pdl,
    )


def _fc1_comments(tactics, **pair_kwargs):
    return {
        _json_config_by_global_index(
            _mxfp4_mxfp8_json_pair(tactic, **pair_kwargs).fc1.prims_ts_gemm_config_index
        ).comment
        for tactic in tactics
    }


def _flat_moe_inputs(num_tokens, hidden_size=3072):
    inputs = [None] * 8
    inputs[4] = torch.empty((num_tokens, hidden_size), device="meta")
    return inputs


def test_config_mapper_loads_local_prims_ts_json():
    assert _expanded_trtllm_gen_json_configs()

def test_config_mapper_json_uses_prims_ts_option_names():
    stale_keys = {
        "cluster_dim_z",
        "num_slices_for_split_k",
        "slice_k",
        "use_shuffled_matrix",
    }
    for cfg in _expanded_prims_ts_json_configs():
        assert not stale_keys.intersection(cfg.options)
        assert all(key == key.lower() for key in cfg.options)

def test_config_mapper_uses_local_json_for_bf16_tactic():
    num_configs = len(_expanded_prims_ts_json_configs())
    pair = _first_buildable_pair(
        map_trtllm_bf16_moe_tactic,
        8,
        activation_type=int(ActivationType.Swiglu),
    )

    assert pair.tile_n == 8
    assert 0 <= pair.fc1.trtllm_gemm_config_index < num_configs
    assert 0 <= pair.fc2.trtllm_gemm_config_index < num_configs
    assert pair.fc1.cfg.kwargs["transpose_mma_output"] == 1
    assert pair.fc2.cfg.kwargs["transpose_mma_output"] == 1
    assert pair.fc1.cfg.kwargs["use_clc_fast_drain"] == 0
    assert pair.fc1.cfg.kwargs["use_tma_store"] == 1


def test_config_mapper_materializes_block_major_k_layout():
    pair = map_trtllm_bf16_moe_tactic(
        [-1, -1],
        activation_type=int(ActivationType.Swiglu),
        num_tokens=8,
        top_k=8,
        num_local_experts=128,
        weight_layout=int(WeightLayout.BlockMajorK),
    )

    assert pair.fc1.cfg.build().weight_layout_kind == int(WeightLayout.BlockMajorK)
    assert pair.fc2.cfg.build().weight_layout_kind == int(WeightLayout.BlockMajorK)


def test_config_mapper_maps_activation_types():
    pair = map_trtllm_bf16_moe_tactic(
        [-1, -1],
        activation_type=int(ActivationType.Swiglu),
        num_tokens=8,
        top_k=8,
        num_local_experts=128,
    )
    assert pair.fc1.cfg.kwargs["act_kind"] == 1

    pair = map_trtllm_bf16_moe_tactic(
        [-1, -1],
        activation_type=int(ActivationType.Relu2),
        num_tokens=8,
        top_k=8,
        num_local_experts=128,
    )
    assert pair.fc1.cfg.kwargs["act_kind"] == 3

def test_config_mapper_keeps_mxfp4_bf16_tile256_disabled():
    assert 256 not in SUPPORTED_MXFP4_BF16_TILE_N
    with pytest.raises(ValueError, match="Unsupported Prims-TS MXFP4xBF16 tile_N"):
        map_trtllm_mxfp4_bf16_moe_tactic([256, 0])

def test_config_mapper_supports_nvfp4_per_token_sfb_e2m1_fc1():
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import DType, RouteImpl

    pair = map_trtllm_nvfp4_moe_tactic(
        [-1, -1],
        num_tokens=8,
        top_k=8,
        num_local_experts=128,
        use_per_token_sf_b=True,
        per_token_sf_dtype=int(DType.FP32),
    )

    fc1 = pair.fc1.cfg.build()
    fc2 = pair.fc2.cfg.build()
    assert fc1.dtype_c_kind == int(DType.E2M1)
    assert fc1.has_epilogue_quant
    assert fc1.route_act == int(RouteImpl.LDGSTS)
    assert fc1.route_sfs_act == int(RouteImpl.LDGSTS)
    assert fc1.use_per_token_sf_b == 1
    assert fc1.per_token_sf_dtype == int(DType.FP32)
    assert fc2.use_per_token_sf_b == 0

def test_nvfp4_tile32_packed_gather_uses_two_warps():
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
        compute_warp_layout,
    )

    pair = map_trtllm_nvfp4_moe_tactic(
        [32, 9],
        activation_type=int(ActivationType.Swiglu),
        num_tokens=512,
        top_k=4,
        num_local_experts=128,
        fc1_has_bias=True,
        fc2_has_bias=True,
    )
    fc1 = pair.fc1.cfg.build()
    compute_warp_layout(fc1)

    assert fc1.tile_n == 32
    assert fc1.num_gather_warps == 2
    assert fc1.threads_per_cta == 512


def test_config_mapper_mxfp4_mxfp8_uses_local_json_config_pair():
    num_configs = len(_expanded_prims_ts_json_configs())
    pair = _first_buildable_pair(
        map_trtllm_mxfp4_mxfp8_moe_tactic,
        64,
        activation_type=int(ActivationType.Swiglu),
        num_tokens=256,
        top_k=8,
        num_local_experts=32,
    )

    assert 0 <= pair.fc1.trtllm_gemm_config_index < num_configs
    assert 0 <= pair.fc2.trtllm_gemm_config_index < num_configs
    assert "num_stages_a" in pair.fc1.cfg.kwargs
    assert "num_stages_a" in pair.fc2.cfg.kwargs
    assert pair.fc1.cfg.build().tile_n == 64
    assert pair.fc2.cfg.build().tile_n == 64

def test_config_mapper_mxfp4_mxfp8_supports_geglu():
    pair = _first_buildable_pair(
        map_trtllm_mxfp4_mxfp8_moe_tactic,
        8,
        activation_type=int(ActivationType.Geglu),
        num_tokens=1024,
        top_k=8,
        num_local_experts=128,
    )

    assert pair.tile_n == 8
    assert pair.fc1.cfg.kwargs["act_kind"] == 2
    assert pair.fc2.cfg.kwargs["act_kind"] == 0


def test_config_mapper_exposes_gpt_oss_low_latency_fc1():
    kwargs = dict(
        activation_type=int(ActivationType.Swiglu),
        num_tokens=1,
        top_k=4,
        num_local_experts=128,
        fc1_has_bias=True,
        fc2_has_bias=True,
        enable_pdl=True,
    )

    for tactic in valid_prims_ts_mxfp4_mxfp8_moe_tactics(**kwargs):
        if tactic[0] != 8:
            continue
        pair = map_trtllm_mxfp4_mxfp8_moe_tactic(tactic, **kwargs)
        fc1 = pair.fc1.cfg.kwargs
        if not fc1["use_work_throttle"]:
            continue

        assert fc1["route_act"] == 2
        assert fc1["route_sfs_act"] == 2
        assert fc1["use_clc_fast_drain"] == 1
        return

    pytest.fail("GPT-OSS low-latency MXFP4xMXFP8 FC1 config is missing")


@pytest.mark.parametrize(
    ("num_tokens", "tile_n", "num_stages"),
    [
        (256, 16, 6),
        (1024, 32, 5),
    ],
)
def test_config_mapper_exposes_gpt_oss_mid_batch_fc2(num_tokens, tile_n, num_stages):
    kwargs = dict(
        activation_type=int(ActivationType.Swiglu),
        num_tokens=num_tokens,
        top_k=4,
        num_local_experts=128,
        fc1_has_bias=True,
        fc2_has_bias=True,
        enable_pdl=True,
    )

    for tactic in valid_prims_ts_mxfp4_mxfp8_moe_tactics(**kwargs):
        if tactic[0] != tile_n:
            continue
        pair = map_trtllm_mxfp4_mxfp8_moe_tactic(tactic, **kwargs)
        fc2 = pair.fc2.cfg.kwargs
        if not (
            fc2["tile_k"] == 256
            and fc2["num_stages_a"] == num_stages
            and fc2["use_unroll_loop_2x_for_mma"]
        ):
            continue

        assert fc2["tile_scheduler"] == 1
        assert fc2["num_stages_tmem_acc"] == 2
        assert fc2["use_clc_fast_drain"] == 1
        assert fc2["use_work_throttle"] == 1
        return

    pytest.fail(f"GPT-OSS mid-batch MXFP4xMXFP8 FC2 tile-N{tile_n} config is missing")


@pytest.mark.parametrize("num_tokens", [256, 1024])
def test_mxfp4_mxfp8_fast_drain_fc1_is_a_generic_autotune_candidate(num_tokens):
    kwargs = dict(
        activation_type=int(ActivationType.Swiglu),
        num_tokens=num_tokens,
        top_k=4,
        num_local_experts=128,
        enable_pdl=True,
    )
    tactics = valid_prims_ts_mxfp4_mxfp8_moe_tactics(**kwargs)
    pairs = [
        _mxfp4_mxfp8_json_pair(tactic, enable_pdl=True)
        for tactic in tactics
        if tactic[0] == 32
    ]
    fast_drain_pairs = [
        pair
        for pair in pairs
        if _json_config_by_global_index(
            pair.fc1.prims_ts_gemm_config_index
        ).comment
        == _MXFP4_MXFP8_FAST_DRAIN_FC1_COMMENT
    ]

    assert fast_drain_pairs
    assert len({pair.fc2.prims_ts_gemm_config_index for pair in fast_drain_pairs}) > 1
    fc1 = fast_drain_pairs[0].fc1.cfg.kwargs
    assert fc1["tile_n"] == 32
    assert fc1["tile_k"] == 256
    assert fc1["cluster_m"] == 2
    assert fc1["mma_m"] == 256
    assert fc1["tile_scheduler"] == 1
    assert fc1["route_act"] == 1
    assert fc1["route_sfs_act"] == 2
    assert fc1["use_clc_fast_drain"] == 1
    assert fc1["use_work_throttle"] == 0
    assert fc1["use_pdl"] == 1
    assert all(
        fc1[field] == 5
        for field in (
            "num_stages_a",
            "num_stages_b",
            "num_stages_smem_sfa",
            "num_stages_smem_sfb",
            "num_stages_tmem_sfa",
            "num_stages_tmem_sfb",
        )
    )


@pytest.mark.parametrize("num_tokens", [1, 256])
def test_gpt_oss_legacy_tactic_maps_to_expected_configs(num_tokens):
    tactics = valid_prims_ts_mxfp4_mxfp8_moe_tactics(
        activation_type=int(ActivationType.Swiglu),
        num_tokens=num_tokens,
        top_k=4,
        num_local_experts=128,
        enable_pdl=True,
    )

    assert [32, 155] in tactics
    pair = _mxfp4_mxfp8_json_pair([32, 155], enable_pdl=True)
    fc1_json = _json_config_by_global_index(pair.fc1.prims_ts_gemm_config_index)
    fc2_json = _json_config_by_global_index(pair.fc2.prims_ts_gemm_config_index)
    assert fc1_json.comment == "MxFp4xMxFp8_FC1_LowLatency"
    assert fc2_json.comment == "MxFp4xMxFp8_FC2_GptOssMidBatch"
    assert pair.fc1.cfg.kwargs["use_clc_fast_drain"] == 0


def test_mxfp4_mxfp8_fast_drain_fc1_is_not_visible_to_mxfp8_moe():
    tactics = valid_prims_ts_mxfp8_mxfp8_moe_tactics(
        activation_type=int(ActivationType.Swiglu),
        num_tokens=1024,
        top_k=4,
        num_local_experts=128,
        enable_pdl=True,
    )

    assert _MXFP4_MXFP8_FAST_DRAIN_FC1_COMMENT not in _fc1_comments(
        tactics,
        activation_type=int(ActivationType.Swiglu),
        dtype_a=int(_DType.MXE4M3),
        enable_pdl=True,
    )


def test_mxfp4_mxfp8_runner_passes_generic_flags_to_tactic_mapper(monkeypatch):
    captured = {}

    def _capture_valid_tactics(**kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(
        runner_module,
        "valid_prims_ts_mxfp4_mxfp8_moe_tactics",
        _capture_valid_tactics,
    )
    monkeypatch.setattr(PrimsTsMxfp4Mxfp8MoERunner, "valid_tactics_dict", {})
    runner = PrimsTsMxfp4Mxfp8MoERunner(
        None,
        top_k=4,
        num_local_experts=128,
        hidden_size=3072,
        intermediate_size=3072,
    )
    runner.set_cache_key_static_extras(enable_pdl=True)
    inputs = _flat_moe_inputs(1024)

    assert runner.get_valid_tactics(inputs, None) == [-1]
    assert captured["enable_pdl"] is True
    assert "hidden_size" not in captured
    assert "intermediate_size" not in captured


def test_config_mapper_exposes_gpt_oss_high_throughput_pair():
    kwargs = dict(
        activation_type=int(ActivationType.Swiglu),
        num_tokens=8192,
        top_k=4,
        num_local_experts=128,
        fc1_has_bias=True,
        fc2_has_bias=True,
        enable_pdl=True,
    )

    for tactic in valid_prims_ts_mxfp4_mxfp8_moe_tactics(**kwargs):
        if tactic[0] != 128:
            continue
        pair = map_trtllm_mxfp4_mxfp8_moe_tactic(tactic, **kwargs)
        fc1 = pair.fc1.cfg.kwargs
        fc2 = pair.fc2.cfg.kwargs
        if not (
            fc1["route_act"] == 2
            and fc1["tile_k"] == 256
            and fc1["use_clc_fast_drain"]
            and not fc1["use_work_throttle"]
            and fc2["tile_k"] == 256
            and fc2.get("num_stages_c_smem") == 1
        ):
            continue

        assert fc1["route_sfs_act"] == 2
        assert fc1["num_stages_tmem_sfa"] == 1
        assert fc1["fuse_operand_sf_loads"] == 1
        assert fc2["use_unroll_loop_2x_for_mma"] == 1
        from flashinfer.prims_ts.batched_gemm.batched_gemm_kernel import (
            build_batched_gemm_task_manager,
        )

        manager = build_batched_gemm_task_manager(
            num_experts=128,
            num_tokens=8192,
            top_k=4,
            verbose=False,
            **fc1,
        )
        assert {task.name for task in manager.tasks} >= {
            "MmaTask0",
            "FusedGatherSfBTask",
            "WorkScheduleTask",
        }
        return

    pytest.fail("GPT-OSS high-throughput MXFP4xMXFP8 config pair is missing")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required")
@pytest.mark.skipif(
    torch.cuda.is_available()
    and not is_sm100a_supported(torch.device("cuda")),
    reason="MXFP4 PrimsTS kernels require Blackwell SM100A+",
)
def test_gpt_oss_high_throughput_fused_fc1_gpu_correctness():
    from flashinfer.prims_ts.batched_gemm.batched_gemm_run import reference_check

    pair = map_trtllm_mxfp4_mxfp8_moe_tactic(
        [128, 6],
        activation_type=int(ActivationType.Swiglu),
        num_tokens=8192,
        top_k=4,
        num_local_experts=128,
        fc1_has_bias=True,
        fc2_has_bias=True,
    )

    assert reference_check(
        num_experts=2,
        num_tokens=257,
        top_k=1,
        problem_n=256,
        problem_k=512,
        seed=123,
        **pair.fc1.cfg.kwargs,
    )


def test_runner_filter_drops_unbuildable_json_tactics():
    valid_pair = _first_buildable_pair(
        map_trtllm_bf16_moe_tactic,
        8,
        activation_type=int(ActivationType.Swiglu),
    )
    filtered = _filter_valid_moe_tactics(
        [[8, 9999], [8, valid_pair.moe_config_index]],
        lambda tactic: map_trtllm_bf16_moe_tactic(
            tactic,
            activation_type=int(ActivationType.Swiglu),
        ),
    )

    assert filtered == [[8, valid_pair.moe_config_index]]

def test_config_mapper_matches_default_tile_selection():
    pair = map_trtllm_bf16_moe_tactic(
        [-1, -1],
        num_tokens=8,
        top_k=8,
        num_local_experts=128,
    )
    assert pair.tile_n == 8

def test_deepseek_scheduler_variants_preserve_persisted_pair_indices():
    legacy_pairs = [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
    ]
    for tactic_index, expected_schedulers in enumerate(legacy_pairs):
        pair = map_trtllm_deepseek_fp8_moe_tactic([32, tactic_index])
        fc1 = pair.fc1.cfg.build()
        fc2 = pair.fc2.cfg.build()
        assert (fc1.tile_scheduler, fc2.tile_scheduler) == expected_schedulers
        assert (fc1.use_work_throttle, fc2.use_work_throttle) == (0, 0)
        assert (fc1.use_clc_fast_drain, fc2.use_clc_fast_drain) == (0, 0)

    throttled_pair = map_trtllm_deepseek_fp8_moe_tactic([32, 8])
    assert throttled_pair.fc1.cfg.build().use_work_throttle == 1
    assert throttled_pair.fc2.cfg.build().use_work_throttle == 1
    assert throttled_pair.fc1.cfg.build().use_clc_fast_drain == 0
    assert throttled_pair.fc2.cfg.build().use_clc_fast_drain == 0

    fast_drain_pair = map_trtllm_deepseek_fp8_moe_tactic([32, 24])
    assert fast_drain_pair.fc1.cfg.build().use_work_throttle == 1
    assert fast_drain_pair.fc2.cfg.build().use_work_throttle == 1
    assert fast_drain_pair.fc1.cfg.build().use_clc_fast_drain == 1
    assert fast_drain_pair.fc2.cfg.build().use_clc_fast_drain == 1


def test_config_mapper_rejects_unknown_tile():
    with pytest.raises(ValueError, match="Unsupported Prims-TS BF16 tile_N"):
        map_trtllm_bf16_moe_tactic([512, 0])

def test_support_reports_missing_dependencies(monkeypatch):
    monkeypatch.setattr(support, "is_prims_ts_available", lambda: False)
    ok, reason = support.is_prims_ts_bf16_supported(
        _runner(), _inputs(), [128, 0], weight_layout=WeightLayout.MajorK
    )
    assert not ok
    assert "dependencies" in reason

def test_support_rejects_lora_after_dependency_and_device_checks(monkeypatch):
    monkeypatch.setattr(support, "is_prims_ts_available", lambda: True)
    monkeypatch.setattr(support, "_device_supports_prims_ts", lambda device: True)
    ok, reason = support.is_prims_ts_bf16_supported(
        _runner(),
        _inputs(gemm1_lora_delta=torch.empty((1,), dtype=torch.bfloat16)),
        [128, 0],
        weight_layout=WeightLayout.MajorK,
    )
    assert not ok
    assert "gemm1_lora_delta" in reason

def test_support_accepts_shuffled_major_k(monkeypatch):
    monkeypatch.setattr(support, "is_prims_ts_available", lambda: True)
    monkeypatch.setattr(support, "_device_supports_prims_ts", lambda device: True)

    ok, reason = support.is_prims_ts_bf16_supported(
        _runner(),
        _inputs(),
        [-1, -1],
        weight_layout=WeightLayout.MajorK,
        use_shuffled_weight=True,
    )

    assert ok
    assert reason == ""

def test_support_accepts_block_major_k(monkeypatch):
    monkeypatch.setattr(support, "is_prims_ts_available", lambda: True)
    monkeypatch.setattr(support, "_device_supports_prims_ts", lambda device: True)

    ok, reason = support.is_prims_ts_bf16_supported(
        _runner(weight_layout=WeightLayout.BlockMajorK),
        _inputs(),
        [-1, -1],
        weight_layout=WeightLayout.BlockMajorK,
        use_shuffled_weight=True,
    )

    assert ok
    assert reason == ""

def test_support_accepts_swiglu_oa_params(monkeypatch):
    monkeypatch.setattr(support, "is_prims_ts_available", lambda: True)
    monkeypatch.setattr(support, "_device_supports_prims_ts", lambda device: True)

    ok, reason = support.is_prims_ts_bf16_supported(
        _runner(),
        _inputs(),
        [-1, -1],
        weight_layout=WeightLayout.MajorK,
        use_shuffled_weight=True,
        gemm1_alpha=torch.ones((1,), dtype=torch.float32),
        gemm1_beta=torch.ones((1,), dtype=torch.float32),
        gemm1_clamp_limit=torch.ones((1,), dtype=torch.float32),
    )

    assert ok
    assert reason == ""

def test_support_rejects_oa_params_for_non_swiglu(monkeypatch):
    monkeypatch.setattr(support, "is_prims_ts_available", lambda: True)
    monkeypatch.setattr(support, "_device_supports_prims_ts", lambda device: True)

    ok, reason = support.is_prims_ts_bf16_supported(
        _runner(activation_type=ActivationType.Geglu),
        _inputs(),
        [128, 0],
        weight_layout=WeightLayout.MajorK,
        use_shuffled_weight=True,
        activation_type=ActivationType.Geglu,
        gemm1_alpha=torch.ones((1,), dtype=torch.float32),
    )

    assert not ok
    assert "Swiglu" in reason


@pytest.mark.parametrize(
    "activation_type",
    [
        ActivationType.Swiglu,
        ActivationType.Relu2,
        ActivationType.Identity,
    ],
)
def test_support_accepts_locally_configured_activations(
    monkeypatch, activation_type: ActivationType
):
    monkeypatch.setattr(support, "is_prims_ts_available", lambda: True)
    monkeypatch.setattr(support, "_device_supports_prims_ts", lambda device: True)

    ok, reason = support.is_prims_ts_bf16_supported(
        _runner(activation_type=activation_type),
        _inputs(),
        [-1, -1],
        weight_layout=WeightLayout.MajorK,
        use_shuffled_weight=True,
        activation_type=activation_type,
    )

    assert ok
    assert reason == ""


@pytest.mark.parametrize(
    "activation_type",
    [
        ActivationType.Geglu,
        ActivationType.Silu,
    ],
)
def test_support_rejects_activations_without_local_config(
    monkeypatch, activation_type: ActivationType
):
    monkeypatch.setattr(support, "is_prims_ts_available", lambda: True)
    monkeypatch.setattr(support, "_device_supports_prims_ts", lambda device: True)

    ok, reason = support.is_prims_ts_bf16_supported(
        _runner(activation_type=activation_type),
        _inputs(),
        [-1, -1],
        weight_layout=WeightLayout.MajorK,
        use_shuffled_weight=True,
        activation_type=activation_type,
    )

    assert not ok
    assert "No buildable local Prims-TS MoE config" in reason

def test_support_accepts_nvfp4_per_token_scale_local_config(monkeypatch):
    monkeypatch.setattr(support, "is_prims_ts_available", lambda: True)
    monkeypatch.setattr(support, "_device_supports_prims_ts", lambda device: True)

    ok, reason = support.is_prims_ts_nvfp4_supported(
        _runner(
            dtype_act=DtypeTrtllmGen.E2m1,
            dtype_weights=DtypeTrtllmGen.E2m1,
            weight_layout=WeightLayout.BlockMajorK,
            hidden_size=256,
            use_per_token_scaling=True,
        ),
        _inputs(
            hidden_states=torch.empty((4, 128), dtype=torch.uint8),
            hidden_states_scale=torch.empty((4, 4), dtype=torch.float8_e4m3fn),
            per_token_scale=torch.ones((4,), dtype=torch.float32),
        ),
        [-1, -1],
        weight_layout=WeightLayout.BlockMajorK,
        use_shuffled_weight=True,
        gemm1_weights_scale=torch.empty((1,), dtype=torch.float8_e4m3fn),
        gemm2_weights_scale=torch.empty((1,), dtype=torch.float8_e4m3fn),
        output1_scale_scalar=torch.ones((1,), dtype=torch.float32),
        output1_scale_gate_scalar=torch.ones((1,), dtype=torch.float32),
        output2_scale_scalar=torch.ones((1,), dtype=torch.float32),
    )

    assert ok, reason

def test_support_accepts_fp8_per_tensor_llama4_routing_scale_tactic_with_sfa(
    monkeypatch,
):
    monkeypatch.setattr(support, "is_prims_ts_available", lambda: True)
    monkeypatch.setattr(support, "_device_supports_prims_ts", lambda device: True)

    ok, reason = support.is_prims_ts_fp8_per_tensor_supported(
        _runner(
            dtype_act=DtypeTrtllmGen.E4m3,
            dtype_weights=DtypeTrtllmGen.E4m3,
            weight_layout=WeightLayout.BlockMajorK,
            hidden_size=128,
            top_k=1,
        ),
        _inputs(
            hidden_states=torch.empty((4, 128), dtype=torch.float8_e4m3fn),
            routing_logits=torch.empty((4, 1), dtype=torch.float32),
        ),
        [-1, -1],
        weight_layout=WeightLayout.BlockMajorK,
        use_shuffled_weight=True,
        routing_method_type=RoutingMethodType.Llama4,
        use_routing_scales_on_input=True,
        fc1_per_channel_weight_scale=torch.ones((256,), dtype=torch.float32),
        output1_scale_scalar=torch.ones((1,), dtype=torch.float32),
        output1_scale_gate_scalar=torch.ones((1,), dtype=torch.float32),
        output2_scale_scalar=torch.ones((1,), dtype=torch.float32),
    )

    assert ok, reason

def test_support_accepts_fp8_per_tensor_routing_scale_without_sfa(monkeypatch):
    monkeypatch.setattr(support, "is_prims_ts_available", lambda: True)
    monkeypatch.setattr(support, "_device_supports_prims_ts", lambda device: True)

    ok, reason = support.is_prims_ts_fp8_per_tensor_supported(
        _runner(
            dtype_act=DtypeTrtllmGen.E4m3,
            dtype_weights=DtypeTrtllmGen.E4m3,
            hidden_size=128,
            top_k=1,
        ),
        _inputs(
            hidden_states=torch.empty((4, 128), dtype=torch.float8_e4m3fn),
            routing_logits=torch.empty((4, 1), dtype=torch.float32),
        ),
        [-1, -1],
        weight_layout=WeightLayout.MajorK,
        use_shuffled_weight=True,
        routing_method_type=RoutingMethodType.Llama4,
        use_routing_scales_on_input=True,
        output1_scale_scalar=torch.ones((1,), dtype=torch.float32),
        output1_scale_gate_scalar=torch.ones((1,), dtype=torch.float32),
        output2_scale_scalar=torch.ones((1,), dtype=torch.float32),
    )

    assert ok, reason


def test_support_accepts_fp8_per_tensor_bf16_routing_scale_dtype(
    monkeypatch,
):
    monkeypatch.setattr(support, "is_prims_ts_available", lambda: True)
    monkeypatch.setattr(support, "_device_supports_prims_ts", lambda device: True)

    ok, reason = support.is_prims_ts_fp8_per_tensor_supported(
        _runner(
            dtype_act=DtypeTrtllmGen.E4m3,
            dtype_weights=DtypeTrtllmGen.E4m3,
            hidden_size=128,
            top_k=1,
        ),
        _inputs(
            hidden_states=torch.empty((4, 128), dtype=torch.float8_e4m3fn),
            routing_logits=torch.empty((4, 1), dtype=torch.bfloat16),
        ),
        [-1, -1],
        weight_layout=WeightLayout.MajorK,
        use_shuffled_weight=True,
        routing_method_type=RoutingMethodType.Llama4,
        use_routing_scales_on_input=True,
        output1_scale_scalar=torch.ones((1,), dtype=torch.float32),
        output1_scale_gate_scalar=torch.ones((1,), dtype=torch.float32),
        output2_scale_scalar=torch.ones((1,), dtype=torch.float32),
    )

    assert ok, reason


def test_support_rejects_fp8_per_tensor_mismatched_sfa_sfb_dtype(monkeypatch):
    monkeypatch.setattr(support, "is_prims_ts_available", lambda: True)
    monkeypatch.setattr(support, "_device_supports_prims_ts", lambda device: True)

    ok, reason = support.is_prims_ts_fp8_per_tensor_supported(
        _runner(
            dtype_act=DtypeTrtllmGen.E4m3,
            dtype_weights=DtypeTrtllmGen.E4m3,
            hidden_size=128,
            top_k=1,
        ),
        _inputs(
            hidden_states=torch.empty((4, 128), dtype=torch.float8_e4m3fn),
            routing_logits=torch.empty((4, 1), dtype=torch.float32),
        ),
        [-1, -1],
        weight_layout=WeightLayout.MajorK,
        use_shuffled_weight=True,
        routing_method_type=RoutingMethodType.Llama4,
        use_routing_scales_on_input=True,
        fc1_per_channel_weight_scale=torch.ones((256,), dtype=torch.bfloat16),
        output1_scale_scalar=torch.ones((1,), dtype=torch.float32),
        output1_scale_gate_scalar=torch.ones((1,), dtype=torch.float32),
        output2_scale_scalar=torch.ones((1,), dtype=torch.float32),
    )

    assert not ok
    assert "fc1_per_channel_weight_scale and routing_logits must use the same dtype" in reason

def test_support_rejects_fp8_per_tensor_sigmoid_routing(monkeypatch):
    monkeypatch.setattr(support, "is_prims_ts_available", lambda: True)
    monkeypatch.setattr(support, "_device_supports_prims_ts", lambda device: True)

    ok, reason = support.is_prims_ts_fp8_per_tensor_supported(
        _runner(
            dtype_act=DtypeTrtllmGen.E4m3,
            dtype_weights=DtypeTrtllmGen.E4m3,
            hidden_size=128,
        ),
        _inputs(hidden_states=torch.empty((4, 128), dtype=torch.float8_e4m3fn)),
        [-1, -1],
        weight_layout=WeightLayout.MajorK,
        use_shuffled_weight=True,
        routing_method_type=RoutingMethodType.Sigmoid,
    )

    assert not ok
    assert "Sigmoid routing" in reason

def test_support_rejects_fp8_per_tensor_deepseekv3_non_gated(monkeypatch):
    monkeypatch.setattr(support, "is_prims_ts_available", lambda: True)
    monkeypatch.setattr(support, "_device_supports_prims_ts", lambda device: True)

    ok, reason = support.is_prims_ts_fp8_per_tensor_supported(
        _runner(
            dtype_act=DtypeTrtllmGen.E4m3,
            dtype_weights=DtypeTrtllmGen.E4m3,
            activation_type=ActivationType.Relu2,
            hidden_size=128,
        ),
        _inputs(hidden_states=torch.empty((4, 128), dtype=torch.float8_e4m3fn)),
        [-1, -1],
        weight_layout=WeightLayout.MajorK,
        use_shuffled_weight=True,
        routing_method_type=RoutingMethodType.DeepSeekV3,
    )

    assert not ok
    assert "DeepSeekV3 routing requires a gated activation" in reason

def test_support_rejects_unshuffled_major_k(monkeypatch):
    monkeypatch.setattr(support, "is_prims_ts_available", lambda: True)
    monkeypatch.setattr(support, "_device_supports_prims_ts", lambda device: True)

    ok, reason = support.is_prims_ts_bf16_supported(
        _runner(use_shuffled_weight=False),
        _inputs(),
        [128, 0],
        weight_layout=WeightLayout.MajorK,
        use_shuffled_weight=False,
    )

    assert not ok
    assert "shuffled weights" in reason
