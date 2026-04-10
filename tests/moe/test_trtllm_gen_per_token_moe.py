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

from flashinfer import (
    RoutingMethodType,
    ActivationType,
)
from flashinfer.fused_moe import (
    trtllm_fp4_block_scale_routed_moe,
)
from flashinfer.utils import device_support_pdl
from flashinfer.fp4_quantization import (
    block_scale_interleave,
    e2m1_and_ufp8sf_scale_to_float,
    fp4_quantize,
)
from flashinfer.fused_moe.core import (
    get_w2_permute_indices_with_cache,
    _maybe_get_cached_w3_w1_permute_indices,
)
from .test_trtllm_gen_fused_moe import (
    routing_reference_topk,
)

torch.manual_seed(42)
cache_permute_indices: Dict[tuple, torch.Tensor] = {}


@pytest.mark.parametrize("num_tokens", [8, 256, 512, 1024])
@pytest.mark.parametrize("hidden_size", [1024, 2048, 3072, 4096])
@pytest.mark.parametrize("intermediate_size", [1024, 2048, 3072, 4096])
@pytest.mark.parametrize("num_experts", [128, 256])
@pytest.mark.parametrize("top_k", [4, 8])
def test_trtllm_gen_routed_fused_moe(
    num_tokens: int,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    top_k: int,
):
    device = torch.device("cuda:0")
    # enable_pdl = device_support_pdl(device)
    enable_pdl = False

    # ======== Input Tensors ========
    hidden_states_bf16 = torch.randn(
        num_tokens,
        hidden_size,
        device=device,
        dtype=torch.bfloat16,
    )
    w13_bf16 = torch.randn(
        num_experts,
        intermediate_size * 2,
        hidden_size,
        device=device,
        dtype=torch.bfloat16,
    ) * 0.1
    w2_bf16 = torch.randn(
        num_experts,
        hidden_size,
        intermediate_size,
        device=device,
        dtype=torch.bfloat16,
    ) * 0.1

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
    per_token_scale_inv = hidden_states_bf16.abs().max(dim=1).values / 448.0 / 6.0
    per_token_scale_inv = per_token_scale_inv.to(torch.float32)
    hidden_states, hidden_states_scale = fp4_quantize(
        hidden_states_bf16,
        per_token_scale_inv,
        sf_vec_size=16,
        sf_use_ue8m0=False,
        is_sf_swizzled_layout=False,
        is_global_scale_inversed=True,
    )
    hidden_states_scale = hidden_states_scale.view(torch.float8_e4m3fn).reshape(
        num_tokens, -1
    )

    w13_global_scale_inv = w13_bf16.abs().amax().to(torch.float32) / 448.0 / 6.0
    w2_global_scale_inv = w2_bf16.abs().amax().to(torch.float32) / 448.0 / 6.0
    w13, w13_scale = fp4_quantize(
        w13_bf16,
        w13_global_scale_inv,
        sf_vec_size=16,
        sf_use_ue8m0=False,
        is_global_scale_inversed=True,
    )
    w13_scale = w13_scale.view(torch.float8_e4m3fn).reshape(
        num_experts, intermediate_size * 2, -1
    )
    w2, w2_scale = fp4_quantize(
        w2_bf16,
        w2_global_scale_inv,
        sf_vec_size=16,
        sf_use_ue8m0=False,
        is_global_scale_inversed=True,
    )
    w2_scale = w2_scale.view(torch.float8_e4m3fn).reshape(
        num_experts, hidden_size, -1
    )

    # ======== Dequantize ========
    # Use dquantized input for more accurate comparison

    hidden_states_dequant = e2m1_and_ufp8sf_scale_to_float(
        hidden_states.cpu(),
        hidden_states_scale.cpu().view(torch.uint8).reshape(-1),
        torch.tensor([1.0], device='cpu'),
        16,
        1,
        False,
    ).to(device)
    hidden_states_dequant *= per_token_scale_inv.unsqueeze(1)
    hidden_states_dequant = hidden_states_dequant.to(torch.bfloat16)

    w13_dequant = torch.empty_like(w13_bf16)
    w2_dequant = torch.empty_like(w2_bf16)
    for i in range(num_experts):
        w13_dequant_= e2m1_and_ufp8sf_scale_to_float(
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
    reference = torch.zeros((num_tokens, hidden_size), device="cuda", dtype=torch.bfloat16)
    for token_idx in range(num_tokens):
        for expert_rank in range(top_k):
            expert_id = topk_ids[token_idx, expert_rank].item()
            weight = expert_weights[token_idx, expert_rank].item()
            # w1: [2*n, k] @ [k] -> [2*n]
            up_gate = hidden_states_dequant[token_idx] @ w13_dequant[expert_id].T  # [2*n]
            # gate, up = up_gate.chunk(2, dim=0)
            up, gate = up_gate.chunk(2, dim=0)
            intermediate = torch.nn.functional.silu(gate) * up  # [n]
            # w2: [k, n] @ [n] -> [k]
            expert_out = intermediate @ w2_dequant[expert_id].T  # [k]
            reference[token_idx] += weight * expert_out
    
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
            w13[i]
            .view(torch.uint8)[permute_indices.to(device)]
            .contiguous()
        )
        permute_sf_indices = _maybe_get_cached_w3_w1_permute_indices(
            cache_permute_indices,
            w13_scale[i].view(torch.uint8),
            epilogue_tile_m,
            num_elts_per_sf=16,
        )
        w13_scale_shuffled.append(
            block_scale_interleave(
                w13_scale.reshape(num_experts, intermediate_size*2, -1)[i]
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
            w2[i]
            .view(torch.uint8)[permute_indices.to(device)]
            .contiguous()
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
        per_token_scale_inv,
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
        None,
    )

    from flashinfer.autotuner import autotune
    with autotune(False):
        result = fn()[0]

    torch.cuda.synchronize()

    print(f'{reference=}')
    print(f'{result=}')

    # mismatch percentage
    rtol = 0.2
    mask = torch.abs((reference - result) * torch.reciprocal(reference)) < rtol
    mismatch_rate = (~mask).float().mean().item()
    print(f'Mismatch: {mismatch_rate * 100}%')
    assert mismatch_rate < 0.3