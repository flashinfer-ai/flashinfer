"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""Regression tests for bfloat16 ``hidden_states`` in ``trtllm_fp4_block_scale_moe``.

Covers the two failures reported in issue #2657:

1. ``hidden_states_scale`` was annotated ``Optional[torch.Tensor]`` but carried no
   default, so omitting it — the correct thing to do for bfloat16 activations, which
   have no block scales — raised ``TypeError`` instead of running.
2. Pairing bfloat16 activations with non-MXFP4 weights tripped a raw ``TVM_FFI_ICHECK``
   inside ``FP4BlockScaleLauncher::check_moe`` ("Only MxE2m1 weights are supported ...")
   at kernel-launch time. The constraint is real, but the diagnostic should be raised
   in Python, before the kernel launches.
"""

import inspect

import pytest
import torch

from flashinfer.fused_moe import trtllm_fp4_block_scale_moe
from flashinfer.utils import is_sm100a_supported


def test_hidden_states_scale_is_optional():
    """``hidden_states_scale`` must be omissible (issue #2657, error 1).

    A signature-level check, so it guards the regression on any machine — no
    Blackwell GPU and no JIT build required.
    """
    param = inspect.signature(trtllm_fp4_block_scale_moe).parameters.get(
        "hidden_states_scale"
    )
    assert param is not None, "hidden_states_scale parameter is missing"
    assert param.default is None, (
        "hidden_states_scale must default to None so bf16 callers can omit it; "
        f"got default {param.default!r}"
    )
    assert param.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")
def test_bf16_hidden_states_with_non_mxfp4_weights_raises():
    """bf16 activations + NVFP4 (``E2m1``) weights must fail in Python (issue #2657, error 2).

    ``E2m1`` weight scales are one per 16 elements; ``MxE2m1`` (MXFP4) uses one per 32.
    Building the scale tensor with the //16 shape therefore selects the unsupported
    combination, which must surface as a ValueError rather than a C++ check failure.
    """
    device = torch.device("cuda")
    if not is_sm100a_supported(device):
        pytest.skip("trtllm_fp4_block_scale_moe requires SM100a and CUDA 12.8+.")

    num_tokens, hidden_size, intermediate_size = 8, 128, 128
    num_experts, top_k = 2, 1

    hidden_states = torch.randn(
        num_tokens, hidden_size, dtype=torch.bfloat16, device=device
    )
    routing_logits = torch.zeros(
        num_tokens, num_experts, dtype=torch.float32, device=device
    )

    def _fp4_weights(out_features, in_features):
        weights = torch.randint(
            0,
            256,
            (num_experts, out_features, in_features // 2),
            dtype=torch.uint8,
            device=device,
        )
        # //16 -> deduced as E2m1 (NVFP4), which bf16 activations do not support.
        scale = torch.ones(
            num_experts,
            out_features,
            in_features // 16,
            dtype=torch.float8_e4m3fn,
            device=device,
        )
        return weights, scale

    gemm1_weights, gemm1_weights_scale = _fp4_weights(
        2 * intermediate_size, hidden_size
    )
    gemm2_weights, gemm2_weights_scale = _fp4_weights(hidden_size, intermediate_size)

    with pytest.raises(ValueError, match="MxE2m1"):
        trtllm_fp4_block_scale_moe(
            routing_logits=routing_logits,
            routing_bias=None,
            hidden_states=hidden_states,
            gemm1_weights=gemm1_weights,
            gemm1_weights_scale=gemm1_weights_scale,
            gemm1_bias=None,
            gemm1_alpha=None,
            gemm1_beta=None,
            gemm1_clamp_limit=None,
            gemm2_weights=gemm2_weights,
            gemm2_weights_scale=gemm2_weights_scale,
            gemm2_bias=None,
            output1_scale_scalar=None,
            output1_scale_gate_scalar=None,
            output2_scale_scalar=None,
            num_experts=num_experts,
            top_k=top_k,
            n_group=None,
            topk_group=None,
            intermediate_size=intermediate_size,
            local_expert_offset=0,
            local_num_experts=num_experts,
            routed_scaling_factor=None,
            # hidden_states_scale deliberately omitted: bf16 activations carry no
            # block scales, and omitting it must not raise TypeError.
        )
