"""Parity gate: pre-quantized activations vs the fused quant prologue.

Runs the dynamic nvfp4 TP MoE twice on identical weights/routing:
baseline (kernel quantizes bf16 activations with a scalar a1_gscale via
the shared-input branch) vs prequant (activations quantized token-major
with vLLM's scaled_fp4_quant linear-SF layout and consumed directly).
If the two quantizers implement the same NVFP4 recipe the outputs must
match bitwise; report max deltas either way.

Run: CUDA_VISIBLE_DEVICES=0 python tests/test_tp_moe_prequant_parity.py
"""

import torch

from b12x.moe.fused_moe._impl import (
    TPMoEScratchCaps,
    b12x_moe_fp4,
    plan_b12x_fp4_moe_weights,
    plan_tp_moe_scratch,
    prepare_b12x_fp4_moe_weights,
)
E, M, K, N_TP, TOPK = 32, 64, 6144, 512, 8


def main() -> None:
    torch.cuda.set_device(0)
    torch.manual_seed(7)
    device = torch.device("cuda", 0)

    w1_n = 2 * N_TP  # gated [up, gate]
    w1_fp4 = torch.randint(
        0, 256, (E, w1_n, K // 2), dtype=torch.uint8, device=device
    )
    w2_fp4 = torch.randint(
        0, 256, (E, K, N_TP // 2), dtype=torch.uint8, device=device
    )
    w1_bs = torch.randint(
        110, 126, (E, w1_n, K // 16), dtype=torch.uint8, device=device
    ).view(torch.float8_e4m3fn)
    w2_bs = torch.randint(
        110, 126, (E, K, N_TP // 16), dtype=torch.uint8, device=device
    ).view(torch.float8_e4m3fn)
    w_gs = torch.ones(E, dtype=torch.float32, device=device)
    a1_gs = torch.full((1,), 1.7, dtype=torch.float32, device=device)
    a2_gs = torch.ones(1, dtype=torch.float32, device=device)

    weight_plan = plan_b12x_fp4_moe_weights(
        quant_modes="nvfp4",
        source_format="modelopt_nvfp4",
        activation="silu",
        params_dtype=torch.bfloat16,
        num_experts=E,
        hidden_size=K,
        intermediate_size=N_TP,
        w13_layout="w31",
    )
    prepared = prepare_b12x_fp4_moe_weights(
        plan=weight_plan,
        w1_fp4=w1_fp4,
        w1_blockscale=w1_bs,
        w1_global_scale=w_gs,
        a1_gscale=a1_gs,
        w2_fp4=w2_fp4,
        w2_blockscale=w2_bs,
        w2_global_scale=w_gs,
        a2_gscale=a2_gs,
        params_dtype=torch.bfloat16,
    )

    caps = TPMoEScratchCaps(
        max_tokens=M,
        num_topk=TOPK,
        device=device,
        weight_plan=weight_plan,
        quant_mode="nvfp4",
        apply_router_weight_on_input=False,
    )
    plan = plan_tp_moe_scratch(caps)
    scratch = {
        spec.name: torch.zeros(spec.shape, dtype=spec.dtype, device=spec.device)
        for spec in plan.scratch_specs()
    }

    a = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    topk_ids = torch.stack(
        [torch.randperm(E, device=device)[:TOPK] for _ in range(M)]
    ).to(torch.int32)
    topk_weights = torch.rand(M, TOPK, dtype=torch.float32, device=device)
    topk_weights = topk_weights / topk_weights.sum(-1, keepdim=True)

    def run(a_prequant=None, a_prequant_scale=None):
        out = torch.zeros(M, K, dtype=torch.bfloat16, device=device)
        binding = plan.bind(
            scratch=scratch,
            a=a,
            experts=prepared,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            output=out,
            input_scales_static=True,
            a_prequant=a_prequant,
            a_prequant_scale=a_prequant_scale,
        )
        b12x_moe_fp4(binding=binding)
        torch.cuda.synchronize()
        return out

    baseline = run()
    baseline2 = run()
    print("baseline finite:", bool(baseline.isfinite().all().item()),
          "amax:", baseline.float().abs().amax().item())
    print("baseline deterministic:", torch.equal(baseline, baseline2))
    self_delta = (baseline.float() - baseline2.float()).abs()
    self_denom = baseline.float().abs().amax().clamp(min=1e-6)
    print(
        f"baseline self-noise: rel max {(self_delta.max() / self_denom).item():.6e}"
        f", mismatched {(baseline != baseline2).float().mean().item() * 100:.4f}%"
    )

    from vllm import _custom_ops as ops

    packed, sf = ops.scaled_fp4_quant(a, a1_gs, is_sf_swizzled_layout=False)
    sf_u8 = sf.view(torch.uint8).reshape(M, K // 16).contiguous()
    prequant = run(a_prequant=packed.contiguous(), a_prequant_scale=sf_u8)
    print("prequant finite:", bool(prequant.isfinite().all().item()),
          "amax:", prequant.float().abs().amax().item())
    print("packed shape:", tuple(packed.shape), packed.dtype,
          "sf shape:", tuple(sf_u8.shape))

    delta = (baseline.float() - prequant.float()).abs()
    denom = baseline.float().abs().amax().clamp(min=1e-6)
    print(f"max |delta| = {delta.max().item():.6e}")
    print(f"rel max     = {(delta.max() / denom).item():.6e}")
    print(f"bitwise equal: {torch.equal(baseline, prequant)}")
    mismatch = (baseline != prequant).float().mean().item()
    print(f"mismatched elements: {mismatch * 100:.4f}%")
    assert (delta.max() / denom).item() < 5e-2, "prequant parity failed"
    print("PREQUANT-PARITY-OK")


if __name__ == "__main__":
    main()
