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

from flashinfer.fused_moe.backends.prims_ts.fp8_op import _resolve_routing_inputs
from flashinfer.fused_moe.shared.inputs import RoutingInputMode
from flashinfer.prims_ts.moe.config_mapper import (
    map_trtllm_deepseek_fp8_moe_tactic,
    map_trtllm_fp8_per_tensor_moe_tactic,
    map_trtllm_mxfp8_mxfp8_moe_tactic,
)
from flashinfer.prims_ts.moe import support
from flashinfer.tllm_enums import (
    ActivationType,
    DtypeTrtllmGen,
    Fp8QuantizationType,
    WeightLayout,
)

def _runner(**overrides):
    values = dict(
        dtype_act=DtypeTrtllmGen.MxE4m3,
        dtype_weights=DtypeTrtllmGen.MxE4m3,
        fp8_quantization_type=Fp8QuantizationType.MxFp8,
        activation_type=ActivationType.Swiglu,
        weight_layout=WeightLayout.MajorK,
        use_shuffled_weight=True,
        hidden_size=128,
        intermediate_size=256,
        use_per_token_scaling=False,
        top_k=1,
        num_local_experts=1,
    )
    values.update(overrides)
    return SimpleNamespace(**values)

def _inputs(**overrides):
    values = dict(
        hidden_states=torch.empty((4, 128), dtype=torch.float8_e4m3fn),
        hidden_states_scale=torch.empty((4, 4), dtype=torch.uint8),
        gemm1_lora_delta=None,
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

def test_mxfp8_mxfp8_mapper_supports_tile256():
    pair = map_trtllm_mxfp8_mxfp8_moe_tactic(
        [256, 0],
        activation_type=int(ActivationType.Swiglu),
        num_tokens=128,
        top_k=8,
        num_local_experts=8,
    )

    fc1 = pair.fc1.cfg.build()
    fc2 = pair.fc2.cfg.build()
    assert pair.tile_n == 256
    assert fc1.tile_n == 256
    assert fc2.tile_n == 256
    assert fc1.uses_mxfp8_output_quant
    assert not fc2.has_epilogue_quant


@pytest.mark.parametrize("tile_n", [8, 16, 32, 64, 128])
def test_deepseek_fp8_mapper_supports_trtllm_tile_ladder(tile_n):
    pair = _first_buildable_pair(
        map_trtllm_deepseek_fp8_moe_tactic,
        tile_n,
        num_tokens=256,
        top_k=8,
        num_local_experts=256,
    )

    fc1 = pair.fc1.cfg.build()
    fc2 = pair.fc2.cfg.build()
    assert pair.tile_n == tile_n
    assert fc1.tile_n == tile_n
    assert fc2.tile_n == tile_n
    assert fc1.has_deepseek_fp8
    assert fc2.has_deepseek_fp8
    assert fc1.uses_fp8_output
    assert fc2.dtype_c_kind != fc1.dtype_c_kind
    assert fc2.num_stages_c_smem == 1

def test_fp8_per_tensor_mapper_supports_no_per_token_sfb_tile8():
    pair = _first_buildable_pair(
        map_trtllm_fp8_per_tensor_moe_tactic,
        8,
        activation_type=int(ActivationType.Swiglu),
        num_tokens=512,
        top_k=8,
        num_local_experts=384,
        use_per_token_sf_b=False,
    )

    fc1 = pair.fc1.cfg.build()
    fc2 = pair.fc2.cfg.build()
    assert pair.tile_n == 8
    assert fc1.tile_n == 8
    assert fc2.tile_n == 8
    assert fc1.use_per_token_sf_a == 0
    assert fc1.use_per_token_sf_b == 0
    assert fc2.use_per_token_sf_a == 0
    assert fc2.use_per_token_sf_b == 0

def test_fp8_per_tensor_mapper_uses_sfa_and_fc1_sfb_tile8():
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import DType

    pair = _first_buildable_pair(
        map_trtllm_fp8_per_tensor_moe_tactic,
        8,
        activation_type=int(ActivationType.Swiglu),
        num_tokens=512,
        top_k=8,
        num_local_experts=384,
        fc1_use_per_token_sf_a=True,
        fc2_use_per_token_sf_a=True,
        use_per_token_sf_b=True,
    )

    fc1 = pair.fc1.cfg.build()
    fc2 = pair.fc2.cfg.build()
    assert pair.tile_n == 8
    assert fc1.tile_n == 8
    assert fc2.tile_n == 8
    assert fc1.use_per_token_sf_a == 1
    assert fc1.use_per_token_sf_b == 1
    assert fc1.per_token_sf_dtype == int(DType.FP32)
    assert fc2.use_per_token_sf_a == 1
    assert fc2.use_per_token_sf_b == 0
    assert fc2.per_token_sf_dtype == int(DType.FP32)


def test_fp8_per_tensor_mapper_supports_fc1_per_channel_only_tile8():
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import DType

    pair = _first_buildable_pair(
        map_trtllm_fp8_per_tensor_moe_tactic,
        8,
        activation_type=int(ActivationType.Swiglu),
        num_tokens=512,
        top_k=8,
        num_local_experts=384,
        fc1_use_per_token_sf_a=True,
        fc2_use_per_token_sf_a=False,
        use_per_token_sf_b=False,
    )

    fc1 = pair.fc1.cfg.build()
    fc2 = pair.fc2.cfg.build()
    assert fc1.use_per_token_sf_a == 1
    assert fc1.use_per_token_sf_b == 0
    assert fc1.per_token_sf_dtype == int(DType.FP32)
    assert fc2.use_per_token_sf_a == 0
    assert fc2.use_per_token_sf_b == 0


def test_fp8_per_tensor_mapper_supports_fc2_per_channel_only_tile8():
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import DType

    pair = _first_buildable_pair(
        map_trtllm_fp8_per_tensor_moe_tactic,
        8,
        activation_type=int(ActivationType.Swiglu),
        num_tokens=512,
        top_k=8,
        num_local_experts=384,
        fc1_use_per_token_sf_a=False,
        fc2_use_per_token_sf_a=True,
        use_per_token_sf_b=False,
    )

    fc1 = pair.fc1.cfg.build()
    fc2 = pair.fc2.cfg.build()
    assert fc1.use_per_token_sf_a == 0
    assert fc1.use_per_token_sf_b == 0
    assert fc2.use_per_token_sf_a == 1
    assert fc2.use_per_token_sf_b == 0
    assert fc2.per_token_sf_dtype == int(DType.FP32)


def test_fp8_per_tensor_mapper_supports_fc1_sfb_without_sfa_tile8():
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import DType

    pair = _first_buildable_pair(
        map_trtllm_fp8_per_tensor_moe_tactic,
        8,
        activation_type=int(ActivationType.Swiglu),
        num_tokens=512,
        top_k=8,
        num_local_experts=384,
        fc1_use_per_token_sf_a=False,
        fc2_use_per_token_sf_a=False,
        use_per_token_sf_b=True,
    )

    fc1 = pair.fc1.cfg.build()
    fc2 = pair.fc2.cfg.build()
    assert fc1.use_per_token_sf_a == 0
    assert fc1.use_per_token_sf_b == 1
    assert fc1.per_token_sf_dtype == int(DType.FP32)
    assert fc2.use_per_token_sf_a == 0
    assert fc2.use_per_token_sf_b == 0


def test_fp8_per_tensor_mapper_accepts_bf16_per_token_sf_dtype():
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import DType

    pair = map_trtllm_fp8_per_tensor_moe_tactic(
        [8, 0],
        activation_type=int(ActivationType.Swiglu),
        num_tokens=512,
        top_k=8,
        num_local_experts=384,
        fc1_use_per_token_sf_a=False,
        fc2_use_per_token_sf_a=False,
        use_per_token_sf_b=True,
        per_token_sf_dtype=int(DType.BF16),
    )

    assert pair.fc1.cfg.build().per_token_sf_dtype == int(DType.BF16)


@pytest.mark.parametrize(
    ("num_tokens", "expected_tile_n"),
    [
        (64, 8),
        (128, 16),
    ],
)
def test_deepseek_fp8_mapper_default_matches_trtllm_fallback(
    num_tokens,
    expected_tile_n,
):
    pair = map_trtllm_deepseek_fp8_moe_tactic(
        [-1, -1],
        num_tokens=num_tokens,
        top_k=2,
        num_local_experts=8,
    )

    assert pair.tile_n == expected_tile_n

def test_deepseek_fp8_mapper_rejects_trtllm_unsupported_tile256():
    with pytest.raises(ValueError, match="DeepSeek FP8 tile_N=256"):
        map_trtllm_deepseek_fp8_moe_tactic([256, 0])

def test_mxfp8_block_support_accepts_swiglu_oa_params(monkeypatch):
    monkeypatch.setattr(support, "is_prims_ts_available", lambda: True)
    monkeypatch.setattr(support, "_device_supports_prims_ts", lambda device: True)

    ok, reason = support.is_prims_ts_fp8_block_scale_supported(
        _runner(weight_layout=WeightLayout.BlockMajorK),
        _inputs(),
        [-1, -1],
        weight_layout=WeightLayout.BlockMajorK,
        use_shuffled_weight=True,
        gemm1_weights_scale=torch.empty((1,), dtype=torch.uint8),
        gemm2_weights_scale=torch.empty((1,), dtype=torch.uint8),
        gemm1_alpha=torch.ones((1,), dtype=torch.float32),
        gemm1_beta=torch.zeros((1,), dtype=torch.float32),
        gemm1_clamp_limit=torch.ones((1,), dtype=torch.float32),
    )

    assert ok
    assert reason == ""

def test_mxfp8_block_support_accepts_bias(monkeypatch):
    monkeypatch.setattr(support, "is_prims_ts_available", lambda: True)
    monkeypatch.setattr(support, "_device_supports_prims_ts", lambda device: True)

    ok, reason = support.is_prims_ts_fp8_block_scale_supported(
        _runner(),
        _inputs(),
        [-1, -1],
        weight_layout=WeightLayout.MajorK,
        use_shuffled_weight=True,
        gemm1_weights_scale=torch.empty((1,), dtype=torch.uint8),
        gemm2_weights_scale=torch.empty((1,), dtype=torch.uint8),
        gemm1_bias=torch.empty((1, 512), dtype=torch.float32),
        gemm2_bias=torch.empty((1, 128), dtype=torch.float32),
    )

    assert ok
    assert reason == ""

def test_deepseek_fp8_support_rejects_oa_params(monkeypatch):
    monkeypatch.setattr(support, "is_prims_ts_available", lambda: True)
    monkeypatch.setattr(support, "_device_supports_prims_ts", lambda device: True)

    ok, reason = support.is_prims_ts_fp8_block_scale_supported(
        _runner(
            dtype_act=DtypeTrtllmGen.E4m3,
            dtype_weights=DtypeTrtllmGen.E4m3,
            fp8_quantization_type=Fp8QuantizationType.DeepSeekFp8,
        ),
        _inputs(hidden_states_scale=torch.empty((4, 1), dtype=torch.float32)),
        [8, 0],
        weight_layout=WeightLayout.MajorK,
        use_shuffled_weight=True,
        gemm1_alpha=torch.ones((1,), dtype=torch.float32),
    )

    assert not ok
    assert "DeepSeek FP8 Prims-TS OA params" in reason

def test_deepseek_fp8_support_rejects_bias(monkeypatch):
    monkeypatch.setattr(support, "is_prims_ts_available", lambda: True)
    monkeypatch.setattr(support, "_device_supports_prims_ts", lambda device: True)

    ok, reason = support.is_prims_ts_fp8_block_scale_supported(
        _runner(
            dtype_act=DtypeTrtllmGen.E4m3,
            dtype_weights=DtypeTrtllmGen.E4m3,
            fp8_quantization_type=Fp8QuantizationType.DeepSeekFp8,
        ),
        _inputs(hidden_states_scale=torch.empty((4, 1), dtype=torch.float32)),
        [8, 0],
        weight_layout=WeightLayout.MajorK,
        use_shuffled_weight=True,
        gemm1_weights_scale=torch.empty((1,), dtype=torch.float32),
        gemm2_weights_scale=torch.empty((1,), dtype=torch.float32),
        gemm1_bias=torch.empty((1, 512), dtype=torch.float32),
    )

    assert not ok
    assert "DeepSeek FP8 Prims-TS bias" in reason

def test_fp8_block_routing_resolver_accepts_unpacked_precomputed():
    hidden_states = torch.empty((4, 128), dtype=torch.bfloat16)
    topk_ids = torch.empty((4, 2), dtype=torch.int32)
    topk_weights = torch.empty((4, 2), dtype=torch.bfloat16)

    routing_logits, resolved_ids, resolved_weights = _resolve_routing_inputs(
        routing_input_mode=RoutingInputMode.UnpackedPrecomputed,
        routing_logits=None,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        hidden_states=hidden_states,
    )

    assert routing_logits is None
    assert resolved_ids is topk_ids
    assert resolved_weights is topk_weights

def test_fp8_block_routing_resolver_rejects_unpacked_fp32_weights():
    hidden_states = torch.empty((4, 128), dtype=torch.bfloat16)
    topk_ids = torch.empty((4, 2), dtype=torch.int32)
    topk_weights = torch.empty((4, 2), dtype=torch.float32)

    with pytest.raises(ValueError, match="topk_weights must be bfloat16"):
        _resolve_routing_inputs(
            routing_input_mode=RoutingInputMode.UnpackedPrecomputed,
            routing_logits=None,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            hidden_states=hidden_states,
        )
