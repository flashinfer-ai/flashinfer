#!/usr/bin/env python
"""Verify trtllm_fp4_block_scale_routed_moe: packed tensor vs (ids, weights) tuple.

Background: vLLM's TrtLlmNvFp4ExpertsModular used to pack routing into a single
int32 tensor (`trtllm_moe_pack_topk_ids_weights`, `(expert_id<<16)|bf16(weight)`)
and pass `topk_ids=packed`. A change replaced that with `topk_ids=(ids, weights)`
(the kernel's "unpacked" format). On the deployment that change made decode
responses never emit EOS (run to max_tokens), so we want to confirm whether the
unpacked tuple path produces the SAME numerical result as the packed path on
this flashinfer build.

This builds NvFP4xNvFP4 MoE inputs (the deployment quant mode), computes a single
routing decision, then runs the routed-moe kernel three ways on identical inputs:
  (P1) packed via vLLM's trtllm_moe_pack_topk_ids_weights  (production packer)
  (P2) packed via the flashinfer test formula               (ids<<16 | weight.int16)
  (U)  unpacked tuple (ids, weights)
and compares P1/P2/U to each other and to the full-routing reference
(trtllm_fp4_block_scale_moe). A large U-vs-P mismatch is the regression.

Run on SM100 (B200). Needs flashinfer + vLLM importable (see submit wrapper).
Place under tests/moe/ so `trtllm_gen_fused_moe_utils` imports as a sibling.
"""

import torch

from flashinfer import RoutingMethodType, ActivationType, fp4_quantize
from flashinfer.fused_moe import (
    trtllm_fp4_block_scale_moe,
    trtllm_fp4_block_scale_routed_moe,
)
from flashinfer.utils import device_support_pdl, get_compute_capability
from .trtllm_gen_fused_moe_utils import (
    routing_reference_renormalize,
    routing_reference_renormalize_naive,
    routing_reference_topk,
)
from vllm.model_executor.layers.fused_moe.utils import trtllm_moe_pack_topk_ids_weights

ROUTING_REF = {
    RoutingMethodType.Renormalize: routing_reference_renormalize,
    RoutingMethodType.RenormalizeNaive: routing_reference_renormalize_naive,
    RoutingMethodType.TopK: routing_reference_topk,
}


def _build_nvfp4_inputs(num_tokens, hidden_size, intermediate_size, num_experts, device):
    routing_logits = torch.rand(num_tokens, num_experts, device=device).to(torch.bfloat16)
    hidden_states = torch.randn(num_tokens, hidden_size, device=device).to(torch.bfloat16) * 0.1
    hidden_states, hidden_states_scale = fp4_quantize(
        hidden_states, torch.tensor([448.0 * 6.0], device=device),
        sf_vec_size=16, sf_use_ue8m0=False, is_sf_swizzled_layout=False,
    )
    hidden_states_scale = hidden_states_scale.view(torch.float8_e4m3fn).reshape(num_tokens, -1)
    hs_gs = 1.0 / 448.0 / 6.0

    w13 = torch.randn(num_experts, intermediate_size * 2, hidden_size, device=device).to(torch.bfloat16) * 0.1
    w2 = torch.randn(num_experts, hidden_size, intermediate_size, device=device).to(torch.bfloat16) * 0.1
    w13, w13_scale = fp4_quantize(w13, torch.tensor([448.0 * 6.0], device=device), sf_vec_size=16, sf_use_ue8m0=False)
    w13_scale = w13_scale.view(torch.float8_e4m3fn).reshape(num_experts, intermediate_size * 2, -1)
    w2, w2_scale = fp4_quantize(w2, torch.tensor([448.0 * 6.0], device=device), sf_vec_size=16, sf_use_ue8m0=False)
    w2_scale = w2_scale.view(torch.float8_e4m3fn).reshape(num_experts, hidden_size, -1)
    w_gs = 1.0 / 448.0 / 6.0

    s1 = torch.tensor([hs_gs * w_gs] * num_experts, device=device)
    s2 = torch.tensor([hs_gs * w_gs] * num_experts, device=device)
    return dict(
        routing_logits=routing_logits, hidden_states=hidden_states,
        hidden_states_scale=hidden_states_scale, w13=w13, w13_scale=w13_scale,
        w2=w2, w2_scale=w2_scale, s1=s1, s2=s2,
    )


def _call_routed(routing_input, b, num_experts, top_k, intermediate_size, enable_pdl):
    return trtllm_fp4_block_scale_routed_moe(
        routing_input, None, b["hidden_states"], b["hidden_states_scale"],
        b["w13"], b["w13_scale"], None, None, None, None,
        b["w2"], b["w2_scale"], None,
        b["s1"], b["s1"], b["s2"],
        num_experts, top_k, None, None, intermediate_size, 0, num_experts, None,
        1,  # routing_method_type (precomputed -> "not used", matches vLLM)
        True, enable_pdl, ActivationType.Swiglu.value, None,
    )[0].to(torch.float)


def _mismatch_pct(a, b, rtol=1e-3, atol=1e-3):
    return (~torch.isclose(a, b, rtol=rtol, atol=atol)).float().mean().item() * 100.0


def run_case(num_tokens, hidden_size, intermediate_size, top_k, num_experts, method, device, enable_pdl):
    torch.manual_seed(42)
    b = _build_nvfp4_inputs(num_tokens, hidden_size, intermediate_size, num_experts, device)

    reference = trtllm_fp4_block_scale_moe(
        b["routing_logits"], None, b["hidden_states"], b["hidden_states_scale"],
        b["w13"], b["w13_scale"], None, None, None, None,
        b["w2"], b["w2_scale"], None, b["s1"], b["s1"], b["s2"],
        num_experts, top_k, None, None, intermediate_size, 0, num_experts, None,
        method.value, True, enable_pdl, ActivationType.Swiglu.value, None,
    )[0].to(torch.float)

    permute_info, expert_weights = ROUTING_REF[method](b["routing_logits"], top_k, num_experts, 8)
    topk_ids = permute_info["topKIndices"].to(torch.int32)
    topk_weights = expert_weights.view(num_tokens, num_experts)[
        torch.arange(num_tokens).unsqueeze(1), topk_ids
    ].to(torch.bfloat16)

    packed_vllm = trtllm_moe_pack_topk_ids_weights(
        topk_ids.contiguous(), topk_weights.contiguous()
    )
    packed_test = (topk_ids.to(torch.int32) << 16) | topk_weights.view(torch.int16)

    out_p1 = _call_routed(packed_vllm, b, num_experts, top_k, intermediate_size, enable_pdl)
    out_p2 = _call_routed(packed_test, b, num_experts, top_k, intermediate_size, enable_pdl)
    out_u = _call_routed((topk_ids, topk_weights), b, num_experts, top_k, intermediate_size, enable_pdl)

    print(
        f"[{method.name:16s} T={num_tokens:<4d} H={hidden_size} I={intermediate_size} "
        f"E={num_experts} k={top_k}]  "
        f"pack_vllm_vs_unpacked={_mismatch_pct(out_p1, out_u):6.2f}%  "
        f"pack_vllm_vs_ref={_mismatch_pct(out_p1, reference):6.2f}%  "
        f"unpacked_vs_ref={_mismatch_pct(out_u, reference):6.2f}%  "
        f"pack_vllm_vs_pack_test={_mismatch_pct(out_p1, out_p2):6.2f}%"
    )


def main():
    cc = get_compute_capability(torch.device("cuda"))
    if cc[0] != 10:
        print(f"WARNING: compute capability {cc}; kernel guaranteed only on SM100/103.")
    device = torch.device("cuda:0")
    enable_pdl = device_support_pdl(device)
    print("Columns: mismatch%% (rtol=atol=1e-3). 'pack_vllm_vs_unpacked' is THE question.")
    # Mistral-Large-3-ish (128 experts, top_k=4) + a couple of shapes.
    for method in (RoutingMethodType.Renormalize, RoutingMethodType.RenormalizeNaive, RoutingMethodType.TopK):
        for (T, H, I) in ((8, 2048, 2048), (1024, 4096, 2048), (8, 7168, 2048)):
            try:
                run_case(T, H, I, top_k=4, num_experts=128, method=method, device=device, enable_pdl=enable_pdl)
            except Exception as e:
                print(f"[{method.name} T={T} H={H} I={I}] ERROR: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
