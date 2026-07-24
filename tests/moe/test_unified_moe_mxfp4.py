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

import pytest
import torch

from flashinfer.fused_moe import (
    ActivationConfig,
    BackendOptions,
    ExpertConfig,
    MoEActivationPack,
    MoEConfig,
    MoELayer,
    MoEWeightPack,
    QuantConfig,
    QuantVariant,
    RoutingConfig,
    TrtllmFp4Config,
)
from flashinfer.utils import get_compute_capability
from tests.moe.test_cute_dsl_fused_moe import (
    check_accuracy,
    compute_reference_moe_fp4,
)


def _sm100_family() -> bool:
    return torch.cuda.is_available() and get_compute_capability(
        torch.device("cuda")
    ) in ((10, 0), (10, 3))


pytestmark = pytest.mark.skipif(
    not _sm100_family(),
    reason="TRTLLM MXFP4 unified tests require SM100 or SM103",
)


@pytest.mark.parametrize("variant", [QuantVariant.MXFP4, QuantVariant.W4A16])
def test_trtllm_mxfp4_unified_matches_reference(variant: QuantVariant):
    if variant is QuantVariant.W4A16 and get_compute_capability(
        torch.device("cuda")
    ) == (10, 3):
        pytest.xfail("TRTLLM MXFP4×BF16 is currently disabled on SM103")

    torch.manual_seed(42)
    device = torch.device("cuda")
    num_tokens = 8
    hidden_size = 1024
    intermediate_size = 512
    num_experts = 8
    top_k = 2

    hidden_states_bf16 = (
        torch.randn(num_tokens, hidden_size, device=device, dtype=torch.bfloat16) * 0.1
    )
    w1_bf16 = (
        torch.randn(
            num_experts,
            2 * intermediate_size,
            hidden_size,
            device=device,
            dtype=torch.bfloat16,
        )
        * 0.1
    )
    w2_bf16 = (
        torch.randn(
            num_experts,
            hidden_size,
            intermediate_size,
            device=device,
            dtype=torch.bfloat16,
        )
        * 0.1
    )
    topk_ids = (
        torch.arange(num_tokens, device=device, dtype=torch.int32)[:, None]
        + torch.arange(top_k, device=device, dtype=torch.int32)[None, :]
    ) % num_experts
    topk_weights = torch.full(
        (num_tokens, top_k),
        1.0 / top_k,
        device=device,
        dtype=torch.float32,
    )

    hidden_states_q, hidden_states_scale = TrtllmFp4Config.prepare_activations(
        hidden_states_bf16,
        variant=variant,
    )
    act_pack = MoEActivationPack(
        hidden_states_q=hidden_states_q,
        hidden_states_scale=hidden_states_scale,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
    )
    weight_pack = MoEWeightPack()
    weight_pack.prepare_for(
        "trtllm_fp4_routed",
        TrtllmFp4Config.prepare_weights(
            w1_bf16,
            w2_bf16,
            variant=variant,
            num_local_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            device=device,
        ),
    )
    config = MoEConfig(
        routing=RoutingConfig(num_experts=num_experts, top_k=top_k),
        quant=QuantConfig(variant=variant),
        experts=ExpertConfig(intermediate_size=intermediate_size),
        activation=ActivationConfig.swiglu,
        backend=BackendOptions(candidates=(TrtllmFp4Config(),)),
    )

    output = MoELayer(config, device=device)(act_pack, weight_pack)
    ones = torch.ones(num_experts, device=device, dtype=torch.float32)
    reference = compute_reference_moe_fp4(
        hidden_states=hidden_states_bf16.float(),
        gemm1_weights=w1_bf16.float(),
        gemm2_weights=w2_bf16.float(),
        gemm1_alpha=ones,
        gemm2_alpha=ones,
        token_selected_experts=topk_ids,
        token_final_scales=topk_weights,
        num_tokens=num_tokens,
        num_experts=num_experts,
        top_k=top_k,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        fc2_input_scale=torch.ones(1, device=device, dtype=torch.float32),
    )
    passed, pct, atol = check_accuracy(output, reference)
    assert passed, (
        f"{variant.name}: only {pct * 100:.2f}% values within tolerance "
        f"(atol={atol:.4f})"
    )
