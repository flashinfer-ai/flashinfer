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
    _expanded_prims_ts_json_configs,
    _expanded_trtllm_gen_json_configs,
    map_trtllm_bf16_moe_tactic,
    map_trtllm_deepseek_fp8_moe_tactic,
    map_trtllm_mxfp4_mxfp8_moe_tactic,
    map_trtllm_mxfp4_bf16_moe_tactic,
    map_trtllm_nvfp4_moe_tactic,
    valid_prims_ts_mxfp4_mxfp8_moe_tactics,
)
from flashinfer.prims_ts.moe.runner import _filter_valid_moe_tactics
from flashinfer.prims_ts.moe import support
from flashinfer.tllm_enums import (
    ActivationType,
    DtypeTrtllmGen,
    RoutingMethodType,
    WeightLayout,
)

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
            use_per_token_scaling=True,
        ),
        _inputs(
            hidden_states=torch.empty((4, 64), dtype=torch.uint8),
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
