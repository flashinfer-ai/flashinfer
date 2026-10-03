# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Exercise the Python dispatch policy without compiling or launching CUDA."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from flashinfer.fused_moe import core


# Position in the native trtllm_fp4_block_scale_moe ABI (zero based).
_NATIVE_PDL_ARGUMENT = 32


@pytest.fixture
def dispatch(monkeypatch):
    native = MagicMock()
    native.trtllm_fp4_block_scale_moe.return_value = []
    spec = SimpleNamespace(build_and_load=lambda: native, get_library_paths=lambda: [])
    monkeypatch.setattr(
        core, "gen_trtllm_gen_fused_moe_sm100_module", lambda **kw: spec
    )
    monkeypatch.setattr(core, "register_custom_op", lambda *a, **kw: lambda fn: fn)
    monkeypatch.setattr(core, "register_fake_op", lambda *a, **kw: lambda fn: fn)
    monkeypatch.setattr(core, "get_compute_capability", lambda device: (10, 7))
    monkeypatch.setattr(core, "device_support_pdl", lambda device: True)
    core._device_support_moe_pdl.cache_clear()
    tuner = MagicMock()
    tuner.choose_one.return_value = (None, -1)
    monkeypatch.setattr(core.AutoTuner, "get", lambda: tuner)
    monkeypatch.setattr(core, "TrtllmMoERunner", MagicMock())
    monkeypatch.setattr(core, "_enabled_trtllm_da_config", lambda: None)
    monkeypatch.setattr(
        core, "_alloc_trtllm_moe_output", lambda n, h, *a: torch.empty(n, h)
    )
    monkeypatch.setattr(core, "_unpack_trtllm_moe_output", lambda *a: a[1])
    module = core._get_trtllm_moe_sm100_module_impl.__wrapped__(True)
    yield module, native, tuner
    core._device_support_moe_pdl.cache_clear()


def arguments():
    # Compact CPU tensors with real NVFP4 scale ratios; no dtype deduction mocks.
    e, h, i, n, k = 4, 128, 128, 2, 2
    return dict(
        routing_input_mode=core.RoutingInputMode.PackedPrecomputed,
        routing_logits=None,
        topk_ids=torch.empty(n, k, dtype=torch.int32),
        topk_weights=torch.empty(n, k, dtype=torch.bfloat16),
        routing_bias=None,
        hidden_states=torch.empty(n, h // 2, dtype=torch.uint8),
        hidden_states_scale=torch.empty(n, h // 16, dtype=torch.float8_e4m3fn),
        gemm1_weights=torch.empty(e, 2 * i, h // 2, dtype=torch.uint8),
        gemm1_weights_scale=torch.empty(e, 2 * i, h // 16, dtype=torch.float8_e4m3fn),
        gemm1_bias=None,
        gemm1_lora_delta=None,
        gemm1_alpha=None,
        gemm1_beta=None,
        gemm1_clamp_limit=None,
        gemm2_weights=torch.empty(e, h, i // 2, dtype=torch.uint8),
        gemm2_weights_scale=torch.empty(e, h, i // 16, dtype=torch.float8_e4m3fn),
        gemm2_bias=None,
        output1_scale_scalar=None,
        output1_scale_gate_scalar=None,
        output2_scale_scalar=None,
        per_token_scale=None,
        num_experts=e,
        top_k=k,
        num_fused_shared_experts=0,
        n_group=None,
        topk_group=None,
        intermediate_size=i,
        local_expert_offset=0,
        num_local_experts=e,
        routed_scaling_factor=None,
        routing_method_type=core.RoutingMethodType.Renormalize.value,
        do_finalize=True,
        enable_pdl=True,
        activation_type=core.ActivationType.Situ.value,
    )


@pytest.mark.parametrize(
    "requested,expected", [(True, True), (False, False), (None, False)]
)
@pytest.mark.parametrize("unpacked", [False, True])
def test_rubin_nvfp4_pdl_forwarding(dispatch, requested, expected, unpacked):
    module, native, tuner = dispatch
    kw = arguments()
    kw["enable_pdl"] = requested
    if unpacked:
        kw["routing_input_mode"] = core.RoutingInputMode.UnpackedPrecomputed
    module.trtllm_fp4_block_scale_moe(**kw)
    assert tuner.choose_one.call_args.kwargs["enable_pdl"] is expected
    assert (
        native.trtllm_fp4_block_scale_moe.call_args.args[_NATIVE_PDL_ARGUMENT]
        is expected
    )


@pytest.mark.parametrize(
    "variant",
    [
        "mxfp4_act",
        "mxfp4_weight",
        "bf16_act",
        "logits",
        "swiglu",
        "per_token",
        "lora",
        "bias",
        "clamp",
        "unfinalized",
        "valid_hidden",
        "valid_intermediate",
        "replay",
        "expert_parallel",
        "da",
        "other_routing",
    ],
)
def test_rubin_unvalidated_fp4_modes_stay_disabled(dispatch, monkeypatch, variant):
    module, native, tuner = dispatch
    kw = arguments()
    if variant == "mxfp4_act":
        kw["hidden_states_scale"] = torch.empty(2, 4, dtype=torch.float8_e4m3fn)
    elif variant == "mxfp4_weight":
        kw["gemm1_weights_scale"] = torch.empty(4, 256, 4, dtype=torch.float8_e4m3fn)
    elif variant == "bf16_act":
        kw["hidden_states"] = torch.empty(2, 128, dtype=torch.bfloat16)
        kw["hidden_states_scale"] = None
    elif variant == "logits":
        kw["routing_input_mode"] = core.RoutingInputMode.FromLogits
        kw["routing_logits"] = torch.empty(2, 4)
    elif variant == "swiglu":
        kw["activation_type"] = core.ActivationType.Swiglu.value
    elif variant in ("per_token", "lora", "bias", "alpha", "beta", "clamp", "replay"):
        field = dict(
            per_token="per_token_scale",
            lora="gemm1_lora_delta",
            bias="gemm1_bias",
            alpha="gemm1_alpha",
            beta="gemm1_beta",
            clamp="gemm1_clamp_limit",
            replay="routing_replay_out",
        )[variant]
        kw[field] = torch.empty(4)
    elif variant == "unfinalized":
        kw["do_finalize"] = False
    elif variant == "valid_hidden":
        kw["valid_hidden_size"] = 128
    elif variant == "valid_intermediate":
        kw["valid_intermediate_size"] = 128
    elif variant == "expert_parallel":
        kw["num_local_experts"] = 2
    elif variant == "other_routing":
        kw["routing_method_type"] = core.RoutingMethodType.TopK.value
    elif variant == "da":
        monkeypatch.setattr(core, "_enabled_trtllm_da_config", lambda: object())

        # Stop after tuner observes the flag, before entering DA orchestration.
        class StopAfterPolicy(Exception):
            pass

        tuner.choose_one.side_effect = StopAfterPolicy
        with pytest.raises(StopAfterPolicy):
            module.trtllm_fp4_block_scale_moe(**kw)
        assert tuner.choose_one.call_args.kwargs["enable_pdl"] is False
        return
    module.trtllm_fp4_block_scale_moe(**kw)
    assert tuner.choose_one.call_args.kwargs["enable_pdl"] is False
    assert (
        native.trtllm_fp4_block_scale_moe.call_args.args[_NATIVE_PDL_ARGUMENT] is False
    )


@pytest.mark.parametrize("cc", [(10, 0), (10, 3)])
@pytest.mark.parametrize(
    "requested,expected", [(True, True), (False, False), (None, True)]
)
def test_blackwell_policy_unchanged(dispatch, monkeypatch, cc, requested, expected):
    module, native, tuner = dispatch
    monkeypatch.setattr(core, "get_compute_capability", lambda device: cc)
    kw = arguments()
    kw["enable_pdl"] = requested
    module.trtllm_fp4_block_scale_moe(**kw)
    assert (
        native.trtllm_fp4_block_scale_moe.call_args.args[_NATIVE_PDL_ARGUMENT]
        is expected
    )


def test_general_rubin_guard_unchanged(dispatch):
    # All other typed entry points continue using this guard.
    assert core._device_support_moe_pdl(torch.device("cpu")) is False


@pytest.mark.parametrize(
    "requested,expected", [(True, True), (False, False), (None, False)]
)
@pytest.mark.parametrize("controls", ["alpha", "beta", "both", "per_expert"])
def test_situ_scale_tensors_keep_pdl_opt_in(dispatch, requested, expected, controls):
    module, native, tuner = dispatch
    kw = arguments()
    kw["enable_pdl"] = requested
    if controls in ("alpha", "both", "per_expert"):
        kw["gemm1_alpha"] = torch.full((4,), 4.0)
    if controls in ("beta", "both", "per_expert"):
        kw["gemm1_beta"] = torch.full((4,), 25.0)
    if controls == "per_expert":
        kw["gemm1_alpha"] = torch.tensor([2.0, 3.0, 4.0, 5.0])
        kw["gemm1_beta"] = torch.tensor([10.0, 15.0, 20.0, 25.0])
    module.trtllm_fp4_block_scale_moe(**kw)
    assert tuner.choose_one.call_args.kwargs["enable_pdl"] is expected
    args = native.trtllm_fp4_block_scale_moe.call_args.args
    assert args[_NATIVE_PDL_ARGUMENT] is expected
    assert args[11] is kw["gemm1_alpha"]
    assert args[12] is kw["gemm1_beta"]


@pytest.mark.parametrize("field", ["gemm1_alpha", "gemm1_beta"])
@pytest.mark.parametrize("invalid", ["dtype", "shape", "stride", "device"])
def test_situ_scale_metadata_must_match(dispatch, field, invalid):
    module, native, tuner = dispatch
    kw = arguments()
    kw[field] = {
        "dtype": lambda: torch.ones(4, dtype=torch.float64),
        "shape": lambda: torch.ones(1, 4),
        "stride": lambda: torch.ones(8)[::2],
        "device": lambda: torch.empty(4, device="meta"),
    }[invalid]()
    module.trtllm_fp4_block_scale_moe(**kw)
    assert tuner.choose_one.call_args.kwargs["enable_pdl"] is False
    assert (
        native.trtllm_fp4_block_scale_moe.call_args.args[_NATIVE_PDL_ARGUMENT] is False
    )


def test_swiglu_with_scale_tensors_stays_disabled(dispatch):
    module, native, tuner = dispatch
    kw = arguments()
    kw.update(
        activation_type=core.ActivationType.Swiglu.value,
        gemm1_alpha=torch.full((4,), 4.0),
        gemm1_beta=torch.full((4,), 25.0),
    )
    module.trtllm_fp4_block_scale_moe(**kw)
    assert tuner.choose_one.call_args.kwargs["enable_pdl"] is False
    assert (
        native.trtllm_fp4_block_scale_moe.call_args.args[_NATIVE_PDL_ARGUMENT] is False
    )
