"""
Copyright (c) 2025 by FlashInfer team.

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
from typing import Dict
import torch
import torch.nn.functional as F

from flashinfer import (
    RoutingMethodType,
    ActivationType,
)
from flashinfer.fused_moe import (
    trtllm_fp4_block_scale_routed_moe,
)
from flashinfer.fp4_quantization import (
    block_scale_interleave,
    e2m1_and_ufp8sf_scale_to_float,
)
from flashinfer.fused_moe.core import (
    get_w2_permute_indices_with_cache,
    _maybe_get_cached_w3_w1_permute_indices,
)
from flashinfer.utils import device_support_pdl, get_compute_capability
from tests.test_helpers.utils_fp4 import nvfp4_global_decode_scale_te, ref_fp4_quant_te
from .test_trtllm_gen_fused_moe import (
    routing_reference_topk,
)

torch.manual_seed(42)
cache_permute_indices: Dict[tuple, torch.Tensor] = {}


def _ref_nvfp4_quantize_te(
    x: torch.Tensor,
    global_amax: torch.Tensor,
    *,
    per_token_rowwise: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    original_shape = x.shape
    x_flat = x.reshape(-1, original_shape[-1])
    if per_token_rowwise:
        global_amax = global_amax.reshape(-1)
    q, scale = ref_fp4_quant_te(
        x_flat,
        global_amax,
        per_token_rowwise=per_token_rowwise,
    )
    return (
        q.reshape(*original_shape[:-1], original_shape[-1] // 2),
        scale.reshape(*original_shape[:-1], original_shape[-1] // 16),
    )


@pytest.mark.parametrize("num_tokens", [1, 8, 1024])
@pytest.mark.parametrize("hidden_size", [1024, 2048, 4096])
@pytest.mark.parametrize("intermediate_size", [1024, 2048, 4096])
@pytest.mark.parametrize("num_experts", [32])
@pytest.mark.parametrize("top_k", [4])
def test_routed_fused_moe(
    num_tokens: int,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    top_k: int,
):
    device = torch.device("cuda:0")
    compute_capability = get_compute_capability(torch.device(device="cuda"))
    if compute_capability[0] not in [10]:
        pytest.skip("These tests are only guaranteed to work on SM100 and SM103 GPUs.")
    enable_pdl = device_support_pdl(device)

    # ======== Input Tensors ========
    hidden_states_bf16 = torch.randn(
        num_tokens,
        hidden_size,
        device=device,
        dtype=torch.bfloat16,
    )
    w13_bf16 = (
        torch.randn(
            num_experts,
            intermediate_size * 2,
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

    # ======== Routing ========
    routing_logits = torch.rand(num_tokens, num_experts, device=device).to(
        torch.bfloat16
    )
    permute_info, expert_weights = routing_reference_topk(
        routing_logits, top_k, num_experts, 8
    )
    topk_ids = permute_info["topKIndices"].to(torch.int32)
    expert_weights = expert_weights.view(num_tokens, num_experts)[
        torch.arange(num_tokens).unsqueeze(1), topk_ids
    ].to(torch.bfloat16)

    # ======== Quantize =======
    hidden_states_amax = hidden_states_bf16.abs().max(dim=1).values.to(torch.float32)
    per_token_scale_inv = nvfp4_global_decode_scale_te(hidden_states_amax)
    hidden_states, hidden_states_scale = _ref_nvfp4_quantize_te(
        hidden_states_bf16,
        hidden_states_amax,
        per_token_rowwise=True,
    )

    w13_global_amax = w13_bf16.abs().amax().to(torch.float32)
    w2_global_amax = w2_bf16.abs().amax().to(torch.float32)
    w13_global_scale_inv = nvfp4_global_decode_scale_te(w13_global_amax)
    w2_global_scale_inv = nvfp4_global_decode_scale_te(w2_global_amax)
    w13, w13_scale = _ref_nvfp4_quantize_te(w13_bf16, w13_global_amax)
    w2, w2_scale = _ref_nvfp4_quantize_te(w2_bf16, w2_global_amax)

    # ======== Dequantize ========
    # Use dquantized input for more accurate comparison

    hidden_states_dequant = e2m1_and_ufp8sf_scale_to_float(
        hidden_states.cpu(),
        hidden_states_scale.cpu().view(torch.uint8).reshape(-1),
        torch.tensor([1.0], device="cpu"),
        16,
        1,
        False,
    ).to(device)
    hidden_states_dequant *= per_token_scale_inv.unsqueeze(1)
    hidden_states_dequant = hidden_states_dequant.to(torch.bfloat16)

    w13_dequant = torch.empty_like(w13_bf16)
    w2_dequant = torch.empty_like(w2_bf16)
    for i in range(num_experts):
        w13_dequant_ = e2m1_and_ufp8sf_scale_to_float(
            w13[i].cpu(),
            w13_scale[i].cpu().view(torch.uint8).reshape(-1),
            w13_global_scale_inv.cpu(),
            16,
            1,
            False,
        ).to(device)
        w13_dequant[i] = w13_dequant_.to(torch.bfloat16)

        w2_dequant_ = e2m1_and_ufp8sf_scale_to_float(
            w2[i].cpu(),
            w2_scale[i].cpu().view(torch.uint8).reshape(-1),
            w2_global_scale_inv.cpu(),
            16,
            1,
            False,
        ).to(device)
        w2_dequant[i] = w2_dequant_.to(torch.bfloat16)

    # ======== Reference Result ========
    # Flatten token/top_k dims: [num_tokens*top_k]
    flat_ids = topk_ids.reshape(-1)  # [num_tokens*top_k]
    flat_weights = expert_weights.reshape(-1)  # [num_tokens*top_k]
    # Gather hidden states for each (token, expert) pair: [num_tokens*top_k, hidden_size]
    flat_hidden = (
        hidden_states_dequant.unsqueeze(1)
        .expand(-1, top_k, -1)
        .reshape(-1, hidden_size)
    )
    # Iterate over experts to avoid materializing huge gathered weight tensors
    expert_out = torch.zeros(
        num_tokens * top_k, hidden_size, device=device, dtype=torch.bfloat16
    )
    for e in range(num_experts):
        mask = flat_ids == e  # [num_tokens*top_k]
        e_hidden = flat_hidden[mask]  # [n_e, hidden_size]
        up_gate_e = e_hidden @ w13_dequant[e].T  # [n_e, 2*intermediate_size]
        up_e, gate_e = up_gate_e.chunk(2, dim=-1)
        inter_e = F.silu(gate_e) * up_e  # [n_e, intermediate_size]
        expert_out[mask] = inter_e @ w2_dequant[e].T  # [n_e, hidden_size]
    # Weighted sum back to [num_tokens, hidden_size]
    reference = (
        (flat_weights.unsqueeze(-1) * expert_out)
        .reshape(num_tokens, top_k, hidden_size)
        .sum(dim=1)
    )

    # ======== Prepare MoE ========

    epilogue_tile_m = 128
    w13_shuffled = []
    w13_scale_shuffled = []
    w2_shuffled = []
    w2_scale_shuffled = []
    for i in range(num_experts):
        permute_indices = _maybe_get_cached_w3_w1_permute_indices(
            cache_permute_indices,
            w13[i].view(torch.uint8),
            epilogue_tile_m,
        )
        w13_shuffled.append(
            w13[i].view(torch.uint8)[permute_indices.to(device)].contiguous()
        )
        permute_sf_indices = _maybe_get_cached_w3_w1_permute_indices(
            cache_permute_indices,
            w13_scale[i].view(torch.uint8),
            epilogue_tile_m,
            num_elts_per_sf=16,
        )
        w13_scale_shuffled.append(
            block_scale_interleave(
                w13_scale.reshape(num_experts, intermediate_size * 2, -1)[i]
                .view(torch.uint8)[permute_sf_indices.to(device)]
                .contiguous()
            )
        )
        permute_indices = get_w2_permute_indices_with_cache(
            cache_permute_indices,
            w2[i].view(torch.uint8),
            epilogue_tile_m,
        )
        w2_shuffled.append(
            w2[i].view(torch.uint8)[permute_indices.to(device)].contiguous()
        )
        permute_sf_indices = get_w2_permute_indices_with_cache(
            cache_permute_indices,
            w2_scale[i].view(torch.uint8),
            epilogue_tile_m,
            num_elts_per_sf=16,
        )
        w2_scale_shuffled.append(
            block_scale_interleave(
                w2_scale.reshape(num_experts, hidden_size, -1)[i]
                .view(torch.uint8)[permute_sf_indices.to(device)]
                .contiguous()
            )
        )
    w13_shuffled = torch.stack(w13_shuffled)
    w13_scale_shuffled = torch.stack(w13_scale_shuffled).view(torch.float8_e4m3fn)
    w2_shuffled = torch.stack(w2_shuffled)
    w2_scale_shuffled = torch.stack(w2_scale_shuffled).view(torch.float8_e4m3fn)

    output1_scale_scalar = torch.stack([w13_global_scale_inv] * num_experts)
    output1_scale_gate_scalar = torch.stack([w13_global_scale_inv] * num_experts)
    output2_scale_scalar = torch.stack([w2_global_scale_inv] * num_experts)

    packed_tensor = (topk_ids.to(torch.int32) << 16) | expert_weights.to(
        torch.bfloat16
    ).view(torch.int16)

    # ======== Launch MoE ========
    from functools import partial

    fn = partial(
        trtllm_fp4_block_scale_routed_moe,
        packed_tensor,
        None,  # routing_bias
        hidden_states,
        hidden_states_scale,
        w13_shuffled,
        w13_scale_shuffled,
        None,  # w13_bias
        None,  # gemm1_alpha
        None,  # gemm1_beta
        None,  # gemm1_clamp_limit
        w2_shuffled,
        w2_scale_shuffled,
        None,  # w2_bias
        output1_scale_scalar,
        output1_scale_gate_scalar,
        output2_scale_scalar,
        num_experts,
        top_k,
        None,  # n_group
        None,  # topk_group
        intermediate_size,
        0,  # local_expert_offset
        num_experts,
        None,  # routed_scaling_factor
        RoutingMethodType.TopK.value,
        True,  # do_finalize
        enable_pdl,
        ActivationType.Swiglu.value,  # act_type
        per_token_scale_inv,
        None,
    )

    from flashinfer.autotuner import autotune

    with autotune(False):
        result = fn()[0]

    torch.cuda.synchronize()

    # mismatch percentage
    rtol = 0.2
    mask = torch.abs((reference - result) * torch.reciprocal(reference)) < rtol
    mismatch_rate = (~mask).float().mean().item()
    print(f"Mismatch: {mismatch_rate * 100}%")
    assert mismatch_rate < 0.3
