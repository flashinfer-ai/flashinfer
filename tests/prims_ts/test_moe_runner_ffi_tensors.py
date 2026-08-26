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

import ast
import inspect

import torch
import tvm_ffi

from flashinfer.fused_moe.shared.inputs import RoutingInputMode
from flashinfer.fused_moe.shared.tuning import moe_topk_ids_init
from flashinfer.fused_moe.backends.prims_ts import bf16_op, fp4_op, fp8_op
from flashinfer.prims_ts.moe import runner as runner_module
from flashinfer.prims_ts.moe.runner import (
    _moe_topk_ids_init_for_routing,
    _routed_token_capacity,
    _torch_views_of_ffi_tensors,
)


def test_torch_tensor_is_preserved():
    tensor = torch.arange(4)

    (view,) = _torch_views_of_ffi_tensors([tensor])

    assert view is tensor


def test_optional_ffi_tensor_is_preserved():
    assert _torch_views_of_ffi_tensors([None]) == [None]


def test_ffi_tensor_can_be_converted_repeatedly():
    tensor = torch.arange(4)
    ffi_tensor = tvm_ffi.from_dlpack(tensor)

    (first_view,) = _torch_views_of_ffi_tensors([ffi_tensor])
    (second_view,) = _torch_views_of_ffi_tensors([ffi_tensor])

    assert first_view.data_ptr() == tensor.data_ptr()
    assert second_view.data_ptr() == tensor.data_ptr()
    torch.testing.assert_close(first_view, tensor)
    torch.testing.assert_close(second_view, tensor)


def test_topk_initializer_matches_routing_representation():
    num_experts = 8

    assert _moe_topk_ids_init_for_routing(
        num_experts, RoutingInputMode.PackedPrecomputed
    ) is moe_topk_ids_init(num_experts, packed=True)
    assert _moe_topk_ids_init_for_routing(
        num_experts, RoutingInputMode.FromLogits
    ) is moe_topk_ids_init(num_experts, packed=True)
    assert _moe_topk_ids_init_for_routing(
        num_experts, RoutingInputMode.UnpackedPrecomputed
    ) is moe_topk_ids_init(num_experts, packed=False)


def test_gemm_launches_use_local_expert_count():
    tree = ast.parse(inspect.getsource(runner_module))
    launch_builders = {
        "build_bf16_launch_io",
        "build_fp8_block_scale_launch_io",
        "build_fp8_per_tensor_launch_io",
        "build_mxfp4_bf16_launch_io",
        "build_mxfp4_mxfp8_launch_io",
        "build_nvfp4_launch_io",
    }
    launch_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in launch_builders
    ]

    assert len(launch_calls) == 14
    direct_values = [
        keyword.value
        for call in launch_calls
        for keyword in call.keywords
        if keyword.arg == "num_experts"
    ]
    shared_values = [
        keyword.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "dict"
        for keyword in node.keywords
        if keyword.arg == "num_experts"
    ]

    assert len(direct_values) == 2
    assert len(shared_values) == 6
    for value in [*direct_values, *shared_values]:
        assert ast.unparse(value) == "self.num_local_experts"


def test_prims_ts_ops_forward_routing_mode_to_tuning():
    calls = []
    for module in (bf16_op, fp4_op, fp8_op):
        tree = ast.parse(inspect.getsource(module))
        calls.extend(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "_make_tuning_config"
        )

    assert len(calls) == 4
    assert all(
        any(keyword.arg == "routing_input_mode" for keyword in call.keywords)
        for call in calls
    )


def test_routed_token_capacity_uses_local_expert_count(monkeypatch):
    from flashinfer.prims_ts.batched_gemm import batched_gemm_config

    captured = []

    def fake_compute(**kwargs):
        captured.append(kwargs)
        return 1

    monkeypatch.setattr(
        batched_gemm_config,
        "compute_max_num_ctas_in_token_dim_for_moe",
        fake_compute,
    )
    runner = type("Runner", (), {"num_local_experts": 4, "top_k": 2})()
    inputs = type("Inputs", (), {"hidden_states": torch.empty((3, 8))})()

    assert (
        _routed_token_capacity(
            runner,
            inputs,
            [8, 0],
            torch.tensor([0], dtype=torch.int32),
            {"num_experts": 16},
        )
        == 8
    )
    assert captured[0]["num_experts"] == 4
