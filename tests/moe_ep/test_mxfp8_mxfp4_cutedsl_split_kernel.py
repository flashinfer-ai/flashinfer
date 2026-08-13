"""Unit + single-GPU parity tests for the CuTeDSL W4A8 split kernel backend.

``sm100_mxfp8_mxfp4_bf16_cutedsl`` MXFP8-quantizes dispatched BF16 tokens and
runs ``cute_dsl_fused_moe_mxfp8_mxfp4`` over this rank's MXFP4 expert shard.
The direct kernel is oracle-tested in
``tests/moe/test_cute_dsl_mxfp8_mxfp4_fused_moe.py``; this file pins the
backend's weight prep, routing synthesis (EXPERT_MAJOR top_k=1 and RANK_MAJOR
masking), and activation quantization to that kernel with identical inputs.

Run on one Blackwell GPU (no torchrun required)::

    CUDA_VISIBLE_DEVICES=0 pytest \\
        tests/moe_ep/test_mxfp8_mxfp4_cutedsl_split_kernel.py -v
"""

from __future__ import annotations

import pytest

NUM_EXPERTS = 4
CAP = 16  # dispatched rows per expert (EXPERT_MAJOR dim1)
HIDDEN = 256
INTERMEDIATE = 128
TOP_K = 2


def _require_gpu_backend():
    import torch

    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
        pytest.skip("sm100_mxfp8_mxfp4_bf16_cutedsl needs an SM100-family GPU")
    from flashinfer.cute_dsl import is_cute_dsl_available

    if not is_cute_dsl_available():
        pytest.skip("CuTeDSL is not available")


class TestConfigAndRegistry:
    def test_registry_resolves_and_requires_weights(self) -> None:
        from flashinfer.moe_ep import (
            Sm100_Mxfp8_Mxfp4_Bf16_Cutedsl_SplitConfig,
            kernel_requires_weights,
        )
        from flashinfer.moe_ep.backends.split.kernel.sm100.mxfp8_mxfp4_bf16_cutedsl import (
            Mxfp8Mxfp4CutedslSplitKernelBackend,
        )
        from flashinfer.moe_ep.core.kernel.registry import create_split_kernel

        cfg = Sm100_Mxfp8_Mxfp4_Bf16_Cutedsl_SplitConfig()
        assert cfg.kernel_name == "sm100_mxfp8_mxfp4_bf16_cutedsl"
        assert kernel_requires_weights(cfg) is True
        kernel = create_split_kernel(cfg)
        assert isinstance(kernel, Mxfp8Mxfp4CutedslSplitKernelBackend)

    def test_rejects_foreign_config(self) -> None:
        from flashinfer.moe_ep import IdentityConfig
        from flashinfer.moe_ep.backends.split.kernel.sm100.mxfp8_mxfp4_bf16_cutedsl import (
            Mxfp8Mxfp4CutedslSplitKernelBackend,
        )

        with pytest.raises(TypeError):
            Mxfp8Mxfp4CutedslSplitKernelBackend(IdentityConfig())

    def test_compute_requires_preprocess_and_init(self) -> None:
        import torch

        from flashinfer.moe_ep import FleetParams, SplitKernelContext
        from flashinfer.moe_ep.backends.split.kernel.sm100.mxfp8_mxfp4_bf16_cutedsl import (
            Mxfp8Mxfp4CutedslSplitKernelBackend,
            Sm100_Mxfp8_Mxfp4_Bf16_Cutedsl_SplitConfig,
        )

        kernel = Mxfp8Mxfp4CutedslSplitKernelBackend(
            Sm100_Mxfp8_Mxfp4_Bf16_Cutedsl_SplitConfig()
        )
        ctx = SplitKernelContext(
            expert_tensors=torch.zeros(2, 2, HIDDEN, dtype=torch.bfloat16),
            num_tokens=4,
            fleet_params=FleetParams(
                num_experts=2, max_tokens_per_rank=2, token_hidden_size=HIDDEN
            ),
        )
        with pytest.raises(RuntimeError, match="preprocess_weights"):
            kernel.compute(ctx)

    def test_weight_prep_rejects_unaligned_geometry(self) -> None:
        import torch

        from flashinfer.moe_ep.backends.split.kernel.sm100.mxfp8_mxfp4_bf16_cutedsl.weights import (
            preprocess_split_weights,
        )
        from flashinfer.moe_ep.weights import MoEWeightPack

        w13 = torch.zeros(2, 2 * 96, 256, dtype=torch.bfloat16)
        w2 = torch.zeros(2, 256, 96, dtype=torch.bfloat16)
        with pytest.raises(ValueError, match="multiples of 128"):
            preprocess_split_weights(MoEWeightPack(w13=w13, w2=w2))


def _make_backend_and_weights(*, layout, world_size=1, rank=0):
    import torch

    from flashinfer.moe_ep import BootstrapConfig, FleetParams
    from flashinfer.moe_ep.backends.split.kernel.sm100.mxfp8_mxfp4_bf16_cutedsl import (
        Mxfp8Mxfp4CutedslSplitKernelBackend,
        Sm100_Mxfp8_Mxfp4_Bf16_Cutedsl_SplitConfig,
    )
    from flashinfer.moe_ep.weights import MoEWeightPack

    gw = torch.Generator(device="cuda").manual_seed(7)
    w13 = (
        torch.randn(NUM_EXPERTS, 2 * INTERMEDIATE, HIDDEN, device="cuda", generator=gw)
        * (HIDDEN**-0.5)
    ).to(torch.bfloat16)
    w2 = (
        torch.randn(NUM_EXPERTS, HIDDEN, INTERMEDIATE, device="cuda", generator=gw)
        * (INTERMEDIATE**-0.5)
    ).to(torch.bfloat16)

    fleet_params = FleetParams(
        num_experts=NUM_EXPERTS,
        max_tokens_per_rank=CAP,
        token_hidden_size=HIDDEN,
        layout=layout,
    )
    bootstrap = BootstrapConfig(world_size=world_size, rank=rank, auto_bootstrap=False)
    kernel = Mxfp8Mxfp4CutedslSplitKernelBackend(
        Sm100_Mxfp8_Mxfp4_Bf16_Cutedsl_SplitConfig()
    )
    kernel.validate_init(bootstrap, fleet_params)
    tw = kernel.preprocess_weights(MoEWeightPack(w13=w13, w2=w2), fleet_params)
    return kernel, tw, fleet_params, w13, w2


def _direct_kernel_reference(flat_bf16, selected_experts, final_scales, tw, top_k):
    """Run the oracle-anchored direct kernel on identically quantized inputs."""
    import torch

    from flashinfer.fused_moe.cute_dsl.fused_moe_mxfp8_mxfp4 import (
        cute_dsl_fused_moe_mxfp8_mxfp4,
    )
    from flashinfer.quantization.fp8_quantization import mxfp8_quantize

    x_q, x_sf = mxfp8_quantize(flat_bf16.contiguous(), is_sf_swizzled_layout=False)
    x_sf = x_sf.view(torch.uint8).reshape(flat_bf16.shape[0], HIDDEN // 32)
    return cute_dsl_fused_moe_mxfp8_mxfp4(
        x_q,
        x_sf,
        selected_experts,
        final_scales,
        tw.w1_weight,
        tw.w1_weight_sf,
        tw.w1_alpha,
        tw.w2_weight,
        tw.w2_weight_sf,
        tw.w2_alpha,
        num_experts=NUM_EXPERTS,
        top_k=top_k,
    )


def test_expert_major_matches_direct_kernel():
    """EXPERT_MAJOR: synthesized top_k=1 routing == direct kernel call."""
    _require_gpu_backend()

    import torch

    from flashinfer.moe_ep import EpLayout, SplitKernelContext

    kernel, tw, fleet_params, _, _ = _make_backend_and_weights(
        layout=EpLayout.EXPERT_MAJOR
    )
    g = torch.Generator(device="cuda").manual_seed(11)
    expert_tensors = (
        torch.randn(NUM_EXPERTS, CAP, HIDDEN, device="cuda", generator=g) * 0.5
    ).to(torch.bfloat16)

    out = kernel.compute(
        SplitKernelContext(
            expert_tensors=expert_tensors,
            num_tokens=NUM_EXPERTS * CAP,
            fleet_params=fleet_params,
        )
    )
    torch.cuda.synchronize()
    assert out.shape == (NUM_EXPERTS, CAP, HIDDEN)
    assert out.dtype == torch.bfloat16

    m = NUM_EXPERTS * CAP
    ids = (
        torch.arange(NUM_EXPERTS, device="cuda", dtype=torch.int32)
        .repeat_interleave(CAP)
        .reshape(m, 1)
        .contiguous()
    )
    ones = torch.ones(m, 1, dtype=torch.float32, device="cuda")
    ref = _direct_kernel_reference(
        expert_tensors.reshape(m, HIDDEN), ids, ones, tw, top_k=1
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(
        out.reshape(m, HIDDEN).float(), ref.float(), rtol=1e-2, atol=1e-2
    )


def _e8m0_to_float(sf_uint8):
    """Decode UE8M0 scale bytes; code 0 decodes to 0 (matches kernel tests)."""
    import torch

    decoded = torch.ldexp(
        torch.ones_like(sf_uint8, dtype=torch.float32),
        sf_uint8.to(torch.int32) - 127,
    )
    return torch.where(sf_uint8 == 0, torch.zeros_like(decoded), decoded)


def _mxfp8_roundtrip(t_bf16):
    """MXFP8 quantize→dequantize with the same linear block-32 rule the backend uses."""
    import torch

    from flashinfer.quantization.fp8_quantization import mxfp8_quantize

    rows, cols = t_bf16.shape
    q, sf = mxfp8_quantize(t_bf16.contiguous(), is_sf_swizzled_layout=False)
    scales = _e8m0_to_float(sf.view(torch.uint8).reshape(rows, cols // 32))
    return (q.float().reshape(rows, cols // 32, 32) * scales.unsqueeze(-1)).reshape(
        rows, cols
    )


def _mxfp4_quant_dequant_3d(w_3d_bf16):
    """MXFP4 quantize→dequantize with the backend's weight recipe (block-32 UE8M0)."""
    import torch

    from flashinfer import e2m1_and_ufp8sf_scale_to_float, fp4_quantize

    experts, rows, k = w_3d_bf16.shape
    flat = w_3d_bf16.reshape(experts * rows, k).contiguous()
    packed, swizzled_sf = fp4_quantize(
        flat,
        global_scale=torch.ones(1, dtype=torch.float32, device=flat.device),
        sf_vec_size=32,
        sf_use_ue8m0=True,
        is_sf_swizzled_layout=True,
    )
    deq = e2m1_and_ufp8sf_scale_to_float(
        packed.detach().cpu(),
        swizzled_sf.detach().cpu().view(torch.uint8).reshape(-1),
        torch.ones(1, dtype=torch.float32),
        sf_vec_size=32,
        ufp8_type=0,
        is_sf_swizzled_layout=True,
    ).to(flat.device)
    return deq.view(experts, rows, k)


def test_expert_major_matches_torch_oracle():
    """Backend output == pure-torch dense MoE over quant-dequant operands.

    Oracle operands: MXFP8 quantize→dequantize activations, MXFP4
    quantize→dequantize weights (block quantization is row-independent, so it
    commutes with the gemm1 gate/linear interleave), and the gemm1→gemm2
    hand-off round-tripped through MXFP8 as the kernel's epilogue does.
    SwiGLU convention (defaults alpha=1, beta=0, no limit):
    ``silu(gate) * linear`` with linear = first half of the gemm1 output.
    """
    _require_gpu_backend()

    import torch

    from flashinfer.moe_ep import EpLayout, SplitKernelContext

    kernel, tw, fleet_params, w13, w2 = _make_backend_and_weights(
        layout=EpLayout.EXPERT_MAJOR
    )
    g = torch.Generator(device="cuda").manual_seed(17)
    expert_tensors = (
        torch.randn(NUM_EXPERTS, CAP, HIDDEN, device="cuda", generator=g) * 0.5
    ).to(torch.bfloat16)
    m = NUM_EXPERTS * CAP

    out = kernel.compute(
        SplitKernelContext(
            expert_tensors=expert_tensors,
            num_tokens=m,
            fleet_params=fleet_params,
        )
    )
    torch.cuda.synchronize()

    x_deq = _mxfp8_roundtrip(expert_tensors.reshape(m, HIDDEN))
    w13_deq = _mxfp4_quant_dequant_3d(w13)
    w2_deq = _mxfp4_quant_dequant_3d(w2)

    ref = torch.zeros(m, HIDDEN, dtype=torch.float32, device="cuda")
    for e in range(NUM_EXPERTS):
        rows = slice(e * CAP, (e + 1) * CAP)
        g1 = x_deq[rows] @ w13_deq[e].transpose(0, 1)
        linear = g1[:, :INTERMEDIATE]
        gate = g1[:, INTERMEDIATE:]
        act = torch.nn.functional.silu(gate) * linear
        act = _mxfp8_roundtrip(act.to(torch.bfloat16))
        ref[rows] = act @ w2_deq[e].transpose(0, 1)

    yk = out.reshape(m, HIDDEN).float()
    rel_l2 = (yk - ref).norm() / ref.norm().clamp_min(1e-6)
    print(
        f"[split w4a8 oracle] rel_l2={rel_l2.item():.4g} "
        f"max|Δ|={(yk - ref).abs().max().item():.4g} "
        f"amax(ref)={ref.abs().max().item():.4g}"
    )
    # Measured on B200 (job 2390487): rel_l2≈0.0155, max|Δ|≈0.016 on
    # amax(ref)≈0.78. The kernel's intermediate MXFP8 scale codes may differ
    # from mxfp8_quantize by ±1 near block boundaries (see
    # tests/moe/test_cute_dsl_mxfp8_mxfp4_grouped_gemm.py) — bounds carry
    # ~3x headroom over the measured band.
    torch.testing.assert_close(yk, ref, rtol=5e-2, atol=0.05)
    assert rel_l2.item() < 0.05


def test_rank_major_masks_non_local_picks():
    """RANK_MAJOR: received routing with -1 picks == pre-masked direct call."""
    _require_gpu_backend()

    import torch

    from flashinfer.moe_ep import EpLayout, SplitKernelContext

    kernel, tw, fleet_params, _, _ = _make_backend_and_weights(
        layout=EpLayout.RANK_MAJOR
    )
    world, per_rank = 1, CAP
    m = world * per_rank
    g = torch.Generator(device="cuda").manual_seed(13)
    recv = (torch.randn(world, per_rank, HIDDEN, device="cuda", generator=g) * 0.5).to(
        torch.bfloat16
    )
    # Local ids in [0, NUM_EXPERTS); make some picks non-local (-1).
    idx = torch.randint(
        0, NUM_EXPERTS, (m, TOP_K), device="cuda", dtype=torch.int64, generator=g
    )
    idx[::3, 0] = -1
    weights = torch.rand(m, TOP_K, device="cuda", generator=g).float()

    out = kernel.compute(
        SplitKernelContext(
            expert_tensors=recv,
            num_tokens=m,
            fleet_params=fleet_params,
            recv_topk_idx=idx,
            recv_topk_weights=weights,
        )
    )
    torch.cuda.synchronize()
    assert out.shape == (world, per_rank, HIDDEN)

    is_local = (idx >= 0) & (idx < NUM_EXPERTS)
    masked_ids = (
        torch.where(is_local, idx, torch.zeros_like(idx)).to(torch.int32).contiguous()
    )
    masked_w = torch.where(is_local, weights, torch.zeros_like(weights)).contiguous()
    ref = _direct_kernel_reference(
        recv.reshape(m, HIDDEN), masked_ids, masked_w, tw, top_k=TOP_K
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(
        out.reshape(m, HIDDEN).float(), ref.float(), rtol=1e-2, atol=1e-2
    )
