"""The trtllm-gen MoE cache key must not depend on ``do_finalize``.

A ``do_finalize=False`` call without ``output`` gets a ``(num_tokens, 0)``
placeholder. It runs the same GEMMs as the finalized call, so it must look up
the entry tuned with a ``(num_tokens, hidden_size)`` output (#6077).

These tests are CPU-only: the JIT module is mocked.
"""

from unittest.mock import MagicMock

import pytest
import torch

from flashinfer.autotuner import AutoTuner
from flashinfer.fused_moe.backends.trtllm import MoERunner
from flashinfer.fused_moe.shared.inputs import MoeRunnerInputs
from flashinfer.tllm_enums import (
    DtypeTrtllmGen,
    Fp8QuantizationType,
    RoutingInputMode,
    SfLayout,
)

HIDDEN_SIZE = 5120
NUM_EXPERTS = 384
TOP_K = 6


def _make_runner():
    return MoERunner(
        MagicMock(),
        top_k=TOP_K,
        num_local_experts=NUM_EXPERTS,
        dtype_act=DtypeTrtllmGen.E2m1,
        dtype_weights=DtypeTrtllmGen.E2m1,
        fp8_quantization_type=Fp8QuantizationType.NoneFp8,
        hidden_size=HIDDEN_SIZE,
        intermediate_size=2304,
        num_experts=NUM_EXPERTS,
    )


def _make_inputs(num_tokens: int, output_width: int) -> MoeRunnerInputs:
    return MoeRunnerInputs(
        output=torch.empty((num_tokens, output_width), dtype=torch.bfloat16),
        routing_logits=None,
        topk_ids=torch.zeros((num_tokens, TOP_K), dtype=torch.int32),
        expert_weights=torch.ones((num_tokens, TOP_K), dtype=torch.bfloat16),
        hidden_states=torch.empty((num_tokens, HIDDEN_SIZE // 2), dtype=torch.uint8),
        hidden_states_scale=torch.ones(
            (num_tokens, HIDDEN_SIZE // 16), dtype=torch.uint8
        ),
        gemm1_lora_delta=None,
        per_token_scale=None,
    )


def _cache_key(runner, moe_inputs, inputs):
    config = runner._make_tuning_config(
        moe_inputs,
        routing_input_mode=RoutingInputMode.PackedPrecomputed,
        hidden_states_scale_layout=SfLayout.layout_linear,
    )
    tuner = AutoTuner.get()
    return AutoTuner._get_cache_key(
        "flashinfer::trtllm_fp4_block_scale_moe",
        runner,
        tuner._get_input_sizes(inputs),
        config,
        runner.get_cache_key_extras(inputs),
    )


@pytest.mark.parametrize("num_tokens", [1, 8, 64])
def test_deferred_output_uses_finalized_cache_key(num_tokens):
    runner = _make_runner()
    finalized = _make_inputs(num_tokens, HIDDEN_SIZE)
    deferred = _make_inputs(num_tokens, 0)

    finalized_key = _cache_key(runner, finalized, finalized.to_tuning_list(HIDDEN_SIZE))
    deferred_key = _cache_key(runner, deferred, deferred.to_tuning_list(HIDDEN_SIZE))
    assert deferred_key == finalized_key
    # The key used to include the placeholder's zero width.
    assert _cache_key(runner, deferred, deferred.to_list()) != finalized_key


def test_tuning_list_keeps_real_output():
    moe_inputs = _make_inputs(8, HIDDEN_SIZE)
    assert moe_inputs.to_tuning_list(HIDDEN_SIZE)[0] is moe_inputs.output


def test_deferred_profile_allocates_finalized_output():
    runner = _make_runner()
    deferred = _make_inputs(8, 0)
    config = runner._make_tuning_config(
        deferred,
        tune_max_num_tokens=64,
        routing_input_mode=RoutingInputMode.PackedPrecomputed,
        hidden_states_scale_layout=SfLayout.layout_linear,
    )
    tuner = AutoTuner.get()
    inputs = deferred.to_tuning_list(HIDDEN_SIZE)
    for profile in tuner._generate_optimization_profiles(config, inputs):
        output = tuner._prepare_input_tensors(profile, inputs)[0]
        assert output.shape[1] == HIDDEN_SIZE
        assert output.is_contiguous()
