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

from flashinfer.fused_moe.api import (
    TrtllmBf16Config,
    TrtllmFp4Config,
    TrtllmFp8BlockConfig,
    TrtllmFp8PerTensorConfig,
    TrtllmMxInt4Config,
)

NUM_EXPERTS = 16
TOP_K = 4
NUM_TOKENS = 64
HIDDEN = 2048
INTERMEDIATE = 1024


def _require_backend(config_cls):
    import torch

    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")

    major, minor = torch.cuda.get_device_capability()
    arch = major * 10 + minor
    if not config_cls.supported(arch):
        pytest.skip(f"{config_cls.__name__} is not supported on sm{arch}")


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
        CuteDslConfig,
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
        "w4a8": (QuantVariant.MXFP4, CuteDslConfig()),
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
    _require_backend(TrtllmBf16Config)

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
        topk_ids=topk_ids.to(torch.int32),
        topk_weights=topk_weights.to(torch.float32),
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
    _require_backend(TrtllmFp4Config)

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
        topk_ids=topk_ids.to(torch.int32),
        topk_weights=topk_weights.to(torch.float32),
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


@pytest.mark.arch_blackwell
def test_split_w4a8_kernel_matches_direct_runner():
    import torch

    import flashinfer.fused_moe as fm
    import flashinfer.moe_ep as ep

    _require_backend(fm.CuteDslConfig)
    if torch.cuda.get_device_capability() == (10, 7):
        pytest.skip("CuTe-DSL W4A8 is not supported on SM107")
    from flashinfer.cute_dsl import is_cute_dsl_available

    if not is_cute_dsl_available():
        pytest.skip("CuTeDSL is not available")

    from dataclasses import replace

    from flashinfer.moe_ep.backends.split.kernel.fused_moe.backend import (
        FusedMoeSplitKernelBackend,
    )
    from flashinfer.quantization.fp8_quantization import mxfp8_quantize

    x, w13, w2, _, _ = _make_problem()
    cfg = _build_moe_config("w4a8")
    fleet = ep.FleetParams(
        num_experts=NUM_EXPERTS,
        max_tokens_per_rank=NUM_TOKENS // NUM_EXPERTS,
        token_hidden_size=HIDDEN,
    )
    backend = FusedMoeSplitKernelBackend(ep.FusedMoeKernelConfig(moe_config=cfg))
    assert backend._transformed_weights is None
    assert backend.pack_dispatch_payload(x) is x
    backend.validate_init(ep.BootstrapConfig(world_size=1, rank=0), fleet)
    weights = backend.preprocess_weights(ep.MoEWeightPack(w13=w13, w2=w2), fleet)
    assert backend._transformed_weights is weights

    expert = x.view(NUM_EXPERTS, -1, HIDDEN)
    actual = backend.compute(
        ep.SplitKernelContext(
            expert_tensors=expert,
            num_tokens=NUM_TOKENS,
            fleet_params=fleet,
        )
    ).reshape(NUM_TOKENS, HIDDEN)

    x_q, x_sf = mxfp8_quantize(x.contiguous(), is_sf_swizzled_layout=False)
    ids = torch.arange(NUM_EXPERTS, device="cuda", dtype=torch.int32)
    ids = ids.repeat_interleave(expert.shape[1]).reshape(-1, 1)
    act = fm.MoEActivationPack(
        hidden_states_q=x_q,
        hidden_states_scale=x_sf.view(torch.uint8).reshape(NUM_TOKENS, HIDDEN // 32),
        topk_ids=ids,
        topk_weights=torch.ones(NUM_TOKENS, 1, device="cuda"),
    )
    ref_cfg = replace(cfg, routing=replace(cfg.routing, top_k=1))
    expected = fm.MoELayer(ref_cfg)(act, weights)
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize(
    ("arch", "expected"),
    [
        (90, False),
        (100, True),
        (103, True),
        (107, True),
        (110, False),
        (120, False),
        (121, False),
    ],
)
@pytest.mark.parametrize(
    "config_cls",
    [TrtllmBf16Config, TrtllmFp4Config, TrtllmMxInt4Config],
)
def test_trtllm_routed_moe_supported_architectures(config_cls, arch, expected):
    """CPU-only: lock down the backend support contract."""
    assert config_cls.supported(arch) is expected


@pytest.mark.parametrize(
    ("arch", "expected"),
    [
        (90, False),
        (100, True),
        (103, True),
        (107, False),
        (110, False),
        (120, False),
        (121, False),
    ],
)
@pytest.mark.parametrize(
    "config_cls",
    [TrtllmFp8BlockConfig, TrtllmFp8PerTensorConfig],
)
def test_trtllm_routed_moe_fp8_supported_architectures(config_cls, arch, expected):
    """CPU-only: the FP8 backends are SM100-family only, unlike FP4/BF16/MxInt4."""
    assert config_cls.supported(arch) is expected


def test_no_trtllm_routed_backend_claims_sm120():
    """Regression guard for #4107: no trtllm-gen routed backend may claim SM120/121.

    The batched-GEMM runner has no cubins there and aborts during construction, so a
    backend that reports itself supported turns a fallback into a hard failure.
    """
    from flashinfer.fused_moe.api import _DEFAULT_BACKEND

    for arch in (110, 120, 121):
        claimed = [
            type(c).__name__
            for c in _DEFAULT_BACKEND
            if type(c).__name__.startswith("Trtllm") and type(c).supported(arch)
        ]
        assert claimed == [], f"trtllm backends wrongly claim sm{arch}: {claimed}"
