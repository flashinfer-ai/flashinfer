"""Single-GPU checks: split-path fused_moe compute kernel vs torch oracles.

The split EP path computes with the fused_moe ``MoELayer`` kernel
(``trtllm_bf16_routed`` / ``trtllm_fp4_routed`` via
``materialize_fused_moe_weights``).  The multirank tests assert
``EP == non-EP kernel`` (dispatch/combine correctness); this file supplies the
missing anchor, ``non-EP kernel == pure-torch oracle``, on a single GPU for
both dtype paths.  Together they close the chain

    EP layer == fused_moe kernel == torch oracle

for LOW_LATENCY and HIGH_THROUGHPUT alike (both algorithms share this compute
kernel; they differ only in dispatch/combine, which the multirank tests pin).

trtllm-gen gated-act convention (same as ``tests/moe`` references):
``gemm1 → [x1 | x2] → silu(x2) * x1 → gemm2 → topk-weighted sum``.

Run on one Blackwell GPU (no torchrun required)::

    CUDA_VISIBLE_DEVICES=0 pytest \\
        tests/moe_ep/test_split_fused_moe_kernel_vs_reference.py -v \\
        -m arch_blackwell
"""

from __future__ import annotations

import pytest

NUM_EXPERTS = 16
TOP_K = 4
NUM_TOKENS = 64
HIDDEN = 2048
INTERMEDIATE = 1024


def _require_blackwell():
    import torch

    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    if torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("trtllm-gen fused_moe requires SM100+")


def _make_problem():
    import torch

    gw = torch.Generator(device="cuda").manual_seed(2024)
    w13 = (
        torch.randn(NUM_EXPERTS, 2 * INTERMEDIATE, HIDDEN, device="cuda", generator=gw)
        * (HIDDEN**-0.5)
    ).to(torch.bfloat16)
    w2 = (
        torch.randn(NUM_EXPERTS, HIDDEN, INTERMEDIATE, device="cuda", generator=gw)
        * (INTERMEDIATE**-0.5)
    ).to(torch.bfloat16)

    g = torch.Generator(device="cuda").manual_seed(1000)
    x = torch.randn(NUM_TOKENS, HIDDEN, device="cuda", generator=g).to(torch.bfloat16)
    scores = torch.randn(NUM_TOKENS, NUM_EXPERTS, device="cuda", generator=g)
    topk_ids = scores.topk(TOP_K, dim=-1).indices.to(torch.int64)
    topk_weights = torch.softmax(
        torch.randn(NUM_TOKENS, TOP_K, device="cuda", generator=g), dim=-1
    )
    return x, w13, w2, topk_ids, topk_weights


def _build_moe_config(variant_str):
    from flashinfer.fused_moe.api import (
        BackendOptions,
        ExecutionConfig,
        ExpertConfig,
        MoEConfig,
        QuantConfig,
        QuantVariant,
        RoutingConfig,
        TrtllmBf16Config,
        TrtllmFp4Config,
    )

    variant, backend = {
        "bf16": (QuantVariant.BF16, TrtllmBf16Config()),
        "nvfp4": (QuantVariant.NVFP4, TrtllmFp4Config()),
    }[variant_str]
    return MoEConfig(
        routing=RoutingConfig(num_experts=NUM_EXPERTS, top_k=TOP_K),
        quant=QuantConfig(variant=variant),
        experts=ExpertConfig(
            intermediate_size=INTERMEDIATE,
            local_expert_offset=0,
            local_num_experts=NUM_EXPERTS,
        ),
        backend=BackendOptions(candidates=(backend,)),
        execution=ExecutionConfig(tune_max_num_tokens=NUM_TOKENS),
    )


def _dense_moe_reference(
    x_f32, w13_f32, w2_f32, topk_ids, topk_weights, *, act_roundtrip
):
    """Vectorized fp32 dense MoE with the trtllm-gen ``silu(x2) * x1`` split."""
    import torch

    out = torch.zeros(
        x_f32.shape[0], w2_f32.shape[1], dtype=torch.float32, device=x_f32.device
    )
    for e in range(NUM_EXPERTS):
        routing_mask = topk_ids == e
        if not routing_mask.any():
            continue
        routed = routing_mask.nonzero(as_tuple=False)
        tokens, slots = routed[:, 0], routed[:, 1]

        g1 = x_f32[tokens] @ w13_f32[e].transpose(0, 1)  # (R, 2I)
        x1 = g1[:, :INTERMEDIATE]
        x2 = g1[:, INTERMEDIATE:]
        act = torch.nn.functional.silu(x2) * x1
        act = act_roundtrip(act)
        g2 = act @ w2_f32[e].transpose(0, 1)  # (R, H)
        out.index_put_(
            (tokens,),
            g2 * topk_weights[tokens, slots].float().unsqueeze(-1),
            accumulate=True,
        )
    return out


def _fp4_quant_dequant(t_2d_bf16):
    """NVFP4 quantize→dequantize with global scale 1 (the EP weight-prep recipe)."""
    import torch

    from flashinfer.quantization.fp4_quantization import (
        e2m1_and_ufp8sf_scale_to_float,
        fp4_quantize,
    )

    gs = torch.ones(1, dtype=torch.float32, device=t_2d_bf16.device)
    q, sf = fp4_quantize(
        t_2d_bf16.contiguous(),
        global_scale=gs,
        sf_vec_size=16,
        is_sf_swizzled_layout=False,
    )
    deq = e2m1_and_ufp8sf_scale_to_float(
        q.cpu(),
        sf.cpu().view(torch.uint8).reshape(-1),
        (1 / gs).cpu(),
        16,
        1,  # ufp8_type: e4m3
        False,  # is_sf_swizzled_layout
    )
    return deq.to(t_2d_bf16.device)


@pytest.mark.arch_blackwell
def test_split_bf16_kernel_matches_torch_reference():
    """``trtllm_bf16_routed`` (all experts local) matches the fp32 torch oracle."""
    _require_blackwell()

    import torch

    from flashinfer.fused_moe.api import MoEActivationPack
    from flashinfer.fused_moe.layer import MoELayer
    from flashinfer.moe_ep import MoEWeightPack
    from flashinfer.moe_ep.backends.split.kernel.fused_moe.weights import (
        materialize_fused_moe_weights,
    )

    x, w13, w2, topk_ids, topk_weights = _make_problem()
    cfg = _build_moe_config("bf16")
    wp = materialize_fused_moe_weights(MoEWeightPack(w13=w13, w2=w2), cfg)

    act = MoEActivationPack(
        hidden_states_q=x,
        hidden_states_scale=torch.empty(0, device=x.device),
        selected_experts=topk_ids.to(torch.int32),
        final_scales=topk_weights.to(torch.float32),
    )
    y_kernel = MoELayer(cfg)(act, wp)
    torch.cuda.synchronize()

    # The bf16 kernel keeps the gemm1→gemm2 hand-off in bf16.
    y_ref = _dense_moe_reference(
        x.float(),
        w13.float(),
        w2.float(),
        topk_ids,
        topk_weights,
        act_roundtrip=lambda a: a.to(torch.bfloat16).float(),
    )

    yk, yr = y_kernel.float(), y_ref
    rel_l2 = (yk - yr).norm() / yr.norm().clamp_min(1e-6)
    print(
        f"[split bf16 oracle] rel_l2={rel_l2.item():.4g} "
        f"max|Δ|={(yk - yr).abs().max().item():.4g} "
        f"amax(ref)={yr.abs().max().item():.4g}"
    )
    torch.testing.assert_close(yk, yr, rtol=3e-2, atol=3e-2)


@pytest.mark.arch_blackwell
def test_split_nvfp4_kernel_matches_torch_reference():
    """``trtllm_fp4_routed`` (all experts local) matches the dequant torch oracle."""
    _require_blackwell()

    import torch

    from flashinfer.fused_moe.api import MoEActivationPack
    from flashinfer.fused_moe.layer import MoELayer
    from flashinfer.moe_ep import MoEWeightPack
    from flashinfer.moe_ep.backends.split.kernel.fused_moe.weights import (
        materialize_fused_moe_weights,
    )
    from flashinfer.quantization.fp4_quantization import fp4_quantize

    x, w13, w2, topk_ids, topk_weights = _make_problem()
    cfg = _build_moe_config("nvfp4")
    wp = materialize_fused_moe_weights(MoEWeightPack(w13=w13, w2=w2), cfg)

    gs = torch.ones(1, dtype=torch.float32, device=x.device)
    x_q, x_sf = fp4_quantize(
        x, global_scale=gs, sf_vec_size=16, is_sf_swizzled_layout=False
    )
    if x_sf.dim() > 2:
        x_sf = x_sf.squeeze(-1)
    act = MoEActivationPack(
        hidden_states_q=x_q,
        hidden_states_scale=x_sf,
        selected_experts=topk_ids.to(torch.int32),
        final_scales=topk_weights.to(torch.float32),
    )
    y_kernel = MoELayer(cfg)(act, wp)
    torch.cuda.synchronize()

    # Oracle operands: the SAME fp4 quantization the kernel consumes
    # (fp4_quantize, global scale 1, per-16 e4m3 SF), dequantized to fp32,
    # with the gemm1→gemm2 hand-off round-tripped through NVFP4 as the
    # kernel's epilogue does (output1 scale scalars are 1 on this path).
    x_deq = _fp4_quant_dequant(x)
    w13_deq = _fp4_quant_dequant(
        w13.reshape(NUM_EXPERTS * 2 * INTERMEDIATE, HIDDEN)
    ).reshape(NUM_EXPERTS, 2 * INTERMEDIATE, HIDDEN)
    w2_deq = _fp4_quant_dequant(w2.reshape(NUM_EXPERTS * HIDDEN, INTERMEDIATE)).reshape(
        NUM_EXPERTS, HIDDEN, INTERMEDIATE
    )

    y_ref = _dense_moe_reference(
        x_deq,
        w13_deq,
        w2_deq,
        topk_ids,
        topk_weights,
        act_roundtrip=lambda a: _fp4_quant_dequant(a.to(torch.bfloat16)),
    )

    yk, yr = y_kernel.float(), y_ref
    rel_l2 = (yk - yr).norm() / yr.norm().clamp_min(1e-6)
    print(
        f"[split nvfp4 oracle] rel_l2={rel_l2.item():.4g} "
        f"max|Δ|={(yk - yr).abs().max().item():.4g} "
        f"amax(ref)={yr.abs().max().item():.4g}"
    )
    # Measured on GB200: rel_l2≈0.032, max|Δ|≈0.10 on |y|~O(1) (fp4 RTNE flips
    # at the gemm1→gemm2 round-trip dominate; 51/131072 cells past 5e-2).
    torch.testing.assert_close(yk, yr, rtol=5e-2, atol=0.15)
    assert rel_l2.item() < 0.05
