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

Three-way DeepSeek-V4.1 routing parity on the trtllm-gen FP4 block-scale MoE
(flashinfer-ai/flashinfer#5190):

    torch DSV4 oracle == internal fused routing (RoutingMethodType.SqrtSoftplus)
                      == externally-routed trtllm_fp4_block_scale_routed_moe

for the same router logits and correction bias at the DeepSeek-V4.1-Flash
geometry (384 experts, top-6, routed_scaling_factor=1.5, norm_topk_prob=True).
Expert ids must match exactly; routing weights are compared at bf16 precision
(the kernel emits bf16 weights); the finalized MoE outputs of the internal and
external paths must agree.

The file also pins the failure mode the issue reports: substituting
RoutingMethodType.DeepSeekV3 either trips the grouped-routing contract
(topk_group <= 4) or, with n_group <= 1, silently routes with sigmoid scores
and selects different experts.
"""

import pytest
import torch

from flashinfer import RoutingMethodType, fp4_quantize
from flashinfer.fused_moe import (
    trtllm_fp4_block_scale_moe,
    trtllm_fp4_block_scale_routed_moe,
    trtllm_gen_routing,
)
from flashinfer.tllm_enums import ActivationType, SfLayout
from flashinfer.utils import device_support_pdl, get_compute_capability

pytestmark = pytest.mark.solo

# DeepSeek-V4.1-Flash text_config (deepseek-ai/DeepSeek-V4.1-Flash).
NUM_EXPERTS = 384
TOP_K = 6
ROUTED_SCALING_FACTOR = 1.5

# bf16 routing weights: 8 mantissa bits -> one ulp is 2^-8 relative.
WEIGHT_ATOL = 1e-2
WEIGHT_RTOL = 2e-2


@pytest.fixture(autouse=True)
def require_sm100():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if get_compute_capability(torch.device("cuda"))[0] != 10:
        pytest.skip("trtllm-gen FP4 block-scale MoE requires SM100/SM103")


def dsv4_oracle(logits, bias, top_k, routed_scaling_factor, norm_topk_prob=True):
    """DeepSeek-V4.1 gate (inference/model.py Gate.forward), ids sorted by expert."""
    scores = torch.sqrt(torch.nn.functional.softplus(logits.float()))
    selection = scores + bias.float().unsqueeze(0)
    ids = torch.argsort(selection, dim=-1, descending=True, stable=True)[:, :top_k]
    w = scores.gather(1, ids)
    if norm_topk_prob:
        w = w / (w.sum(dim=-1, keepdim=True) + 1e-20)
    w = w * routed_scaling_factor
    ids, order = torch.sort(ids, dim=1)
    return ids, w.gather(1, order)


def make_router_inputs(num_tokens, seed):
    gen = torch.Generator().manual_seed(seed)
    # fp32 router logits (the DSV4 gate runs in fp32); bf16 correction bias
    # (DSV4 checkpoints store e_score_correction_bias in bf16).
    logits = (torch.randn(num_tokens, NUM_EXPERTS, generator=gen) * 2.0).cuda()
    bias = (torch.randn(NUM_EXPERTS, generator=gen) * 0.5).to(torch.bfloat16).cuda()
    return logits, bias


def make_nvfp4_experts(num_tokens, hidden_size, intermediate_size, seed):
    torch.manual_seed(seed)
    device = torch.device("cuda")
    hidden_states = (
        torch.randn(num_tokens, hidden_size, device=device).to(torch.bfloat16) * 0.1
    )
    hidden_states, hidden_states_scale = fp4_quantize(
        hidden_states,
        torch.tensor([448.0 * 6.0], device=device),
        sf_vec_size=16,
        sf_use_ue8m0=False,
        is_sf_swizzled_layout=False,
    )
    hidden_states_scale = hidden_states_scale.view(torch.float8_e4m3fn).reshape(
        num_tokens, -1
    )
    w13 = (
        torch.randn(NUM_EXPERTS, intermediate_size * 2, hidden_size, device=device).to(
            torch.bfloat16
        )
        * 0.1
    )
    w2 = (
        torch.randn(NUM_EXPERTS, hidden_size, intermediate_size, device=device).to(
            torch.bfloat16
        )
        * 0.1
    )
    w13, w13_scale = fp4_quantize(
        w13,
        torch.tensor([448.0 * 6.0], device=device),
        sf_vec_size=16,
        sf_use_ue8m0=False,
    )
    w13_scale = w13_scale.view(torch.float8_e4m3fn).reshape(
        NUM_EXPERTS, intermediate_size * 2, -1
    )
    w2, w2_scale = fp4_quantize(
        w2,
        torch.tensor([448.0 * 6.0], device=device),
        sf_vec_size=16,
        sf_use_ue8m0=False,
    )
    w2_scale = w2_scale.view(torch.float8_e4m3fn).reshape(NUM_EXPERTS, hidden_size, -1)
    global_scale = 1.0 / 448.0 / 6.0
    scale_vec = torch.tensor([global_scale * global_scale] * NUM_EXPERTS, device=device)
    return dict(
        hidden_states=hidden_states,
        hidden_states_scale=hidden_states_scale,
        w13=w13,
        w13_scale=w13_scale,
        w2=w2,
        w2_scale=w2_scale,
        output1_scale_scalar=scale_vec,
        output1_scale_gate_scalar=scale_vec.clone(),
        output2_scale_scalar=scale_vec.clone(),
    )


def _common_moe_args(experts, intermediate_size):
    return (
        experts["hidden_states"],
        experts["hidden_states_scale"],
        experts["w13"],
        experts["w13_scale"],
        None,  # gemm1_bias
        None,  # gemm1_alpha
        None,  # gemm1_beta
        None,  # gemm1_clamp_limit
        experts["w2"],
        experts["w2_scale"],
        None,  # gemm2_bias
        experts["output1_scale_scalar"],
        experts["output1_scale_gate_scalar"],
        experts["output2_scale_scalar"],
        NUM_EXPERTS,
        TOP_K,
        None,  # n_group  (DSV4.1: ungrouped)
        None,  # topk_group
        intermediate_size,
        0,  # local_expert_offset
        NUM_EXPERTS,  # local_num_experts
    )


@pytest.mark.parametrize("num_tokens", [8, 150, 1024])
def test_dsv41_internal_routing_matches_oracle_and_routed_path(num_tokens):
    hidden_size, intermediate_size = 512, 512
    enable_pdl = device_support_pdl(torch.device("cuda"))
    logits, bias = make_router_inputs(num_tokens, seed=0x5190 + num_tokens)
    ref_ids, ref_w = dsv4_oracle(logits, bias, TOP_K, ROUTED_SCALING_FACTOR)
    experts = make_nvfp4_experts(num_tokens, hidden_size, intermediate_size, seed=42)
    common = _common_moe_args(experts, intermediate_size)

    # --- internal fused routing: expert ids via routing replay, weights via
    # do_finalize=False (the routing kernel's own bf16 output), then the
    # finalized output.
    replay = torch.full((num_tokens, TOP_K), -1, dtype=torch.int16, device="cuda")
    gemm2_out, kernel_w, _ = trtllm_fp4_block_scale_moe(
        logits,
        bias,
        *common,
        ROUTED_SCALING_FACTOR,
        RoutingMethodType.SqrtSoftplus.value,
        False,  # do_finalize
        enable_pdl,
        ActivationType.Swiglu.value,
        None,
        routing_replay_out=replay,
        hidden_states_scale_layout=SfLayout.layout_linear,
    )
    kernel_ids, order = torch.sort(replay.long(), dim=1)
    kernel_w = kernel_w.float().gather(1, order)
    assert torch.equal(kernel_ids, ref_ids), (
        f"internal routing selected different experts on "
        f"{int((kernel_ids != ref_ids).any(dim=1).sum())}/{num_tokens} tokens"
    )
    torch.testing.assert_close(kernel_w, ref_w, atol=WEIGHT_ATOL, rtol=WEIGHT_RTOL)
    torch.testing.assert_close(
        kernel_w.sum(dim=1),
        torch.full((num_tokens,), ROUTED_SCALING_FACTOR, device="cuda"),
        atol=2e-2,
        rtol=0.0,
    )

    # The standalone routing op runs the same Routing::Runner dispatch.
    standalone = trtllm_gen_routing(
        logits,
        bias,
        RoutingMethodType.SqrtSoftplus,
        TOP_K,
        routed_scaling_factor=ROUTED_SCALING_FACTOR,
    )
    s_ids, s_order = torch.sort(standalone.topk_ids.long(), dim=1)
    assert torch.equal(s_ids, ref_ids)
    torch.testing.assert_close(
        standalone.topk_weights.float().gather(1, s_order),
        ref_w,
        atol=WEIGHT_ATOL,
        rtol=WEIGHT_RTOL,
    )

    out_internal = trtllm_fp4_block_scale_moe(
        logits,
        bias,
        *common,
        ROUTED_SCALING_FACTOR,
        RoutingMethodType.SqrtSoftplus.value,
        True,  # do_finalize
        enable_pdl,
        ActivationType.Swiglu.value,
        None,
        hidden_states_scale_layout=SfLayout.layout_linear,
    )[0].float()

    # --- externally-routed path fed with the oracle's ids/weights.
    out_external = trtllm_fp4_block_scale_routed_moe(
        (ref_ids.to(torch.int32), ref_w.to(torch.bfloat16)),
        None,  # routing_bias
        *common,
        None,  # routed_scaling_factor: already folded into the weights
        RoutingMethodType.SqrtSoftplus.value,
        True,  # do_finalize
        enable_pdl,
        ActivationType.Swiglu.value,
        None,
        hidden_states_scale_layout=SfLayout.layout_linear,
    )[0].float()

    assert torch.isfinite(out_internal).all() and torch.isfinite(out_external).all()
    close = torch.isclose(out_internal, out_external, rtol=1e-2, atol=1e-2)
    mismatch_pct = (~close).float().mean().item() * 100
    assert mismatch_pct < 0.5, (
        f"internal vs external MoE output mismatch {mismatch_pct:.3f}% "
        f"(max abs diff {(out_internal - out_external).abs().max().item():.4g})"
    )


def test_dsv41_deepseekv3_substitution_is_not_a_workaround():
    """The issue's misdiagnosis: DeepSeekV3 with SGLang's default groups trips the
    grouped-routing check, and with n_group <= 1 it silently routes with sigmoid
    scores, selecting different experts than the DSV4.1 gate."""
    num_tokens = 64
    logits, bias = make_router_inputs(num_tokens, seed=0x5190)
    ref_ids, _ = dsv4_oracle(logits, bias, TOP_K, ROUTED_SCALING_FACTOR)

    with pytest.raises(Exception, match="topk_group <= 4"):
        trtllm_gen_routing(
            logits,
            bias,
            RoutingMethodType.DeepSeekV3,
            TOP_K,
            n_group=8,
            topk_group=8,
            routed_scaling_factor=ROUTED_SCALING_FACTOR,
        )

    sigmoid_path = trtllm_gen_routing(
        logits,
        bias,
        RoutingMethodType.DeepSeekV3,
        TOP_K,
        n_group=1,
        topk_group=1,
        routed_scaling_factor=ROUTED_SCALING_FACTOR,
    )
    sig_ids, _ = torch.sort(sigmoid_path.topk_ids.long(), dim=1)
    misrouted = (sig_ids != ref_ids).any(dim=1)
    assert misrouted.any(), "sigmoid (DeepSeekV3) routing unexpectedly matched DSV4.1"

    correct = trtllm_gen_routing(
        logits,
        bias,
        RoutingMethodType.SqrtSoftplus,
        TOP_K,
        routed_scaling_factor=ROUTED_SCALING_FACTOR,
    )
    assert torch.equal(torch.sort(correct.topk_ids.long(), dim=1)[0], ref_ids)
