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

import inspect
from types import SimpleNamespace

import pytest
import torch

from flashinfer.fused_moe.backends.prims_ts.bf16_op import (
    prims_ts_bf16_moe,
    prims_ts_bf16_moe_partitioned,
    prims_ts_bf16_routed_moe,
)
from flashinfer.fused_moe.backends.prims_ts.fp4_op import (
    prims_ts_fp4_block_scale_moe,
    prims_ts_fp4_block_scale_moe_partitioned,
    prims_ts_fp4_block_scale_routed_moe,
)
from flashinfer.fused_moe.backends.prims_ts.fp8_op import (
    _fake_prims_ts_fp8_per_tensor_scale_moe,
    prims_ts_fp8_block_scale_moe,
    prims_ts_fp8_block_scale_moe_partitioned,
    prims_ts_fp8_block_scale_routed_moe,
    prims_ts_fp8_per_tensor_scale_moe,
    prims_ts_fp8_per_tensor_scale_moe_partitioned,
)
from flashinfer.fused_moe.core import trtllm_fp8_per_tensor_scale_moe


@pytest.mark.cpu_only
def test_prims_ts_fp8_positional_contract_matches_trtllm():
    trtllm_params = list(
        inspect.signature(trtllm_fp8_per_tensor_scale_moe).parameters.values()
    )
    prims_ts_params = list(
        inspect.signature(prims_ts_fp8_per_tensor_scale_moe).parameters.values()
    )

    assert [parameter.name for parameter in prims_ts_params[: len(trtllm_params)]] == [
        parameter.name for parameter in trtllm_params
    ]
    assert all(
        parameter.kind == expected.kind
        for parameter, expected in zip(prims_ts_params, trtllm_params, strict=False)
    )

    backend_params = prims_ts_params[len(trtllm_params) :]
    assert [parameter.name for parameter in backend_params] == [
        "weight_layout",
        "fc1_per_channel_weight_scale",
        "fc2_per_channel_weight_scale",
    ]
    assert all(
        parameter.kind == inspect.Parameter.KEYWORD_ONLY for parameter in backend_params
    )
    assert inspect.signature(
        _fake_prims_ts_fp8_per_tensor_scale_moe
    ) == inspect.signature(prims_ts_fp8_per_tensor_scale_moe)


@pytest.mark.cpu_only
def test_partition_resources_only_appear_on_partitioned_ops():
    ordinary_ops = (
        prims_ts_bf16_moe,
        prims_ts_bf16_routed_moe,
        prims_ts_fp4_block_scale_moe,
        prims_ts_fp4_block_scale_routed_moe,
        prims_ts_fp8_per_tensor_scale_moe,
        prims_ts_fp8_block_scale_moe,
        prims_ts_fp8_block_scale_routed_moe,
    )

    for op in ordinary_ops:
        assert "partition" not in inspect.signature(op).parameters
        assert "partition_resources" not in inspect.signature(op).parameters

    partitioned_ops = (
        prims_ts_bf16_moe_partitioned,
        prims_ts_fp4_block_scale_moe_partitioned,
        prims_ts_fp8_per_tensor_scale_moe_partitioned,
        prims_ts_fp8_block_scale_moe_partitioned,
    )
    for op in partitioned_ops:
        parameters = inspect.signature(op).parameters
        assert "partition_resources" in parameters
        assert "partition_strategy" in parameters
        assert parameters["enable_pdl"].default is None
        assert (
            not {
                "gemm1_bias",
                "gemm2_bias",
                "gemm1_lora_delta",
                "num_fused_shared_experts",
            }
            & parameters.keys()
        )


def _routing_args():
    return dict(
        routing_logits=torch.empty((4, 8)),
        routing_bias=None,
        topk_ids=None,
        expert_weights=None,
    )


def _moe_shape_args():
    return dict(
        num_experts=8,
        top_k=2,
        n_group=None,
        topk_group=None,
        intermediate_size=128,
        local_expert_offset=0,
        local_num_experts=8,
    )


@pytest.mark.cpu_only
def test_partitioned_ops_require_resources_with_strategy():
    bf16_weight = torch.empty((8, 256, 128), dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="partition_resources is required"):
        prims_ts_bf16_moe_partitioned(
            **_routing_args(),
            hidden_states=torch.empty((4, 128), dtype=torch.bfloat16),
            gemm1_weights=[bf16_weight, bf16_weight],
            gemm2_weights=[bf16_weight, bf16_weight],
            **_moe_shape_args(),
            partition_resources=None,
            partition_strategy="tp_n_tp_n",
        )

    fp8_weight = torch.empty((8, 256, 128), dtype=torch.float8_e4m3fn)
    scale = torch.ones(8)
    with pytest.raises(ValueError, match="partition_resources is required"):
        prims_ts_fp8_per_tensor_scale_moe_partitioned(
            **_routing_args(),
            hidden_states=torch.empty((4, 128), dtype=torch.float8_e4m3fn),
            gemm1_weights=[fp8_weight, fp8_weight],
            output1_scales_scalar=scale,
            output1_scales_gate_scalar=scale,
            gemm2_weights=[fp8_weight, fp8_weight],
            output2_scales_scalar=scale,
            **_moe_shape_args(),
            partition_resources=None,
            partition_strategy="tp_n_tp_n",
        )


@pytest.mark.cpu_only
def test_partitioned_op_rejects_wrong_shard_count():
    weight = torch.empty((8, 256, 128), dtype=torch.float8_e4m3fn)
    scale = torch.empty(0, dtype=torch.uint8)
    with pytest.raises(ValueError, match="gemm1_weights must contain exactly 2"):
        prims_ts_fp8_block_scale_moe_partitioned(
            **_routing_args(),
            hidden_states=torch.empty((4, 128), dtype=torch.float8_e4m3fn),
            hidden_states_scale=torch.empty(0),
            gemm1_weights=[weight],
            gemm1_weights_scale=[scale, scale],
            gemm2_weights=[weight, weight],
            gemm2_weights_scale=[scale, scale],
            **_moe_shape_args(),
            partition_resources=SimpleNamespace(),
            partition_strategy="tp_n_tp_n",
        )
