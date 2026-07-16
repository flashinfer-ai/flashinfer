"""Single-GPU checks: NVFP4 ``nvfp4_mega_moe`` vs a pure-torch oracle.

NVFP4 counterpart of ``test_mxfp8_cutedsl_preprocess_vs_reference.py``: validates
that ``nvfp4_cutedsl.preprocess_mega_weights`` produces fp4 weights consistent
with an independent plain quant, and that a single-rank ``nvfp4_mega_moe``
launch matches a pure-torch dequant reference (fp32 GEMMs + SwiGLU + fc1-out
NVFP4 round-trip) after the in-kernel top-k reduction.

The torch oracle here is intentionally independent of the CuTeDSL-backed
``compute_megamoe_reference`` (whose GEMMs run on a reference device kernel):
everything below is plain torch ops on dequantized values, so it validates the
kernel's math end to end, not just its plumbing.

Run on one Blackwell GPU from the FlashInfer repo root (no torchrun required)::

    cd /path/to/flashinfer
    export PYTHONPATH="${PWD}:${PYTHONPATH}"
    MEGA_NO_DIST=1 CUDA_VISIBLE_DEVICES=0 pytest \\
        tests/moe_ep/test_nvfp4_cutedsl_kernel_vs_reference.py -v \\
        -m arch_blackwell --confcutdir=tests/moe_ep
"""

from __future__ import annotations

import pytest

# Verify only through the cutedsl_megamoe shim public API (plus the FI backend
# helpers); never import the src/ kernel packages directly, so a new src/ drop
# can't silently break this test.
pytest.importorskip("flashinfer.moe_ep.kernel_src.cutedsl_megamoe")

NVFP4_BLOCK = 16


def _require_cuda():
    import torch

    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")


def _single_rank_problem():
    import torch

    hidden = 2048
    intermediate = 1024
    num_tokens = 32
    max_tokens = 64
    num_experts = 4
    topk = 4
    num_local_experts = num_experts
    gate_up_clamp = 10.0

    g = torch.Generator(device="cuda").manual_seed(7)
    hidden_states = torch.randn(
        num_tokens, hidden, dtype=torch.bfloat16, device="cuda", generator=g
    )
    scores = torch.randn(
        num_tokens, num_experts, dtype=torch.float32, device="cuda", generator=g
    )
    topk_weights, topk_ids = torch.topk(
        scores, topk, dim=-1, largest=True, sorted=False
    )

    g = torch.Generator(device="cuda").manual_seed(13)
    w13 = torch.randn(
        num_local_experts,
        2 * intermediate,
        hidden,
        dtype=torch.bfloat16,
        device="cuda",
        generator=g,
    )
    w2 = torch.randn(
        num_local_experts,
        hidden,
        intermediate,
        dtype=torch.bfloat16,
        device="cuda",
        generator=g,
    )

    return dict(
        hidden=hidden,
        intermediate=intermediate,
        num_tokens=num_tokens,
        max_tokens=max_tokens,
        num_experts=num_experts,
        topk=topk,
        gate_up_clamp=gate_up_clamp,
        hidden_states=hidden_states,
        topk_weights=topk_weights.to(torch.float32),
        topk_ids=topk_ids.to(torch.int64),
        w13=w13,
        w2=w2,
    )


def _e2m1_decode_table(device):
    import torch

    # Standard E2M1 code points, low nibble = even element (matches the
    # kernel-side pack/unpack convention).
    return torch.tensor(
        [
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
            3.0,
            4.0,
            6.0,
            -0.0,
            -0.5,
            -1.0,
            -1.5,
            -2.0,
            -3.0,
            -4.0,
            -6.0,
        ],
        dtype=torch.float32,
        device=device,
    )


def _dequant_nvfp4(packed, sf_fp8, *, logical_cols):
    """Packed fp4 codes + plain per-16 e4m3 scales → fp32 (rows, logical_cols)."""
    import torch

    raw = packed.view(torch.uint8).reshape(packed.shape[0], -1)
    lut = _e2m1_decode_table(raw.device)
    lo = lut[(raw & 0x0F).to(torch.int64)]
    hi = lut[(raw >> 4).to(torch.int64)]
    vals = torch.empty(
        raw.shape[0], raw.shape[1] * 2, dtype=torch.float32, device=raw.device
    )
    vals[:, ::2] = lo
    vals[:, 1::2] = hi
    vals = vals[:, :logical_cols]
    n_blocks = logical_cols // NVFP4_BLOCK
    scales = (
        sf_fp8[:, :n_blocks].to(torch.float32).repeat_interleave(NVFP4_BLOCK, dim=-1)
    )
    return vals * scales


def _plain_nvfp4_from_bf16(problem: dict):
    """bf16 weights → kernel fp4 + plain e4m3 SF (pre-swizzle layout)."""
    import torch

    from flashinfer.moe_ep.backends.mega.kernel.nvfp4_cutedsl.weights import (
        _interleave_gate_up_16,
    )
    from flashinfer.moe_ep.kernel_src.cutedsl_megamoe import (
        nvfp4_quantize_per_block_16,
    )

    intermediate = problem["intermediate"]
    num_experts = problem["w13"].shape[0]
    norm_const = 1.0

    w13_interleaved = _interleave_gate_up_16(
        problem["w13"], intermediate_size=intermediate
    )

    fc1_weights, fc1_plain_sf = [], []
    fc2_weights, fc2_plain_sf = [], []
    for expert in range(num_experts):
        fc1_q, fc1_sf = nvfp4_quantize_per_block_16(
            w13_interleaved[expert].to(torch.float32), norm_const
        )
        fc1_weights.append(fc1_q)
        fc1_plain_sf.append(fc1_sf)
        fc2_q, fc2_sf = nvfp4_quantize_per_block_16(
            problem["w2"][expert].to(torch.float32), norm_const
        )
        fc2_weights.append(fc2_q)
        fc2_plain_sf.append(fc2_sf)

    return (
        torch.stack([w.view(torch.uint8) for w in fc1_weights], dim=0),
        torch.stack(fc1_plain_sf, dim=0),
        torch.stack([w.view(torch.uint8) for w in fc2_weights], dim=0),
        torch.stack(fc2_plain_sf, dim=0),
    )


def _torch_nvfp4_mega_reference(
    *,
    act_packed,  # (T, hidden//2) packed fp4 (uint8 view ok)
    act_sf,  # (T, >=hidden//16) e4m3 plain
    topk_idx,  # (T, topk) int64
    topk_weights,  # (T, topk) fp32
    fc1_weight,  # (E, 2I, hidden//2) packed fp4 codes (uint8)
    fc1_sf,  # (E, 2I, hidden//16) e4m3 plain
    fc2_weight,  # (E, hidden, I//2) packed fp4 codes (uint8)
    fc2_sf,  # (E, hidden, I//16) e4m3 plain
    hidden,
    intermediate,
    gate_up_clamp,
):
    """Pure-torch NVFP4 MegaMoE oracle (apply_topk_in_fc1=True graph).

    Mirrors the kernel's data path — dequant → fp32 fc1 GEMM → 16-interleaved
    SwiGLU fold (+clamp) → per-token topk weight folded in BEFORE the fc1-out
    NVFP4 round-trip → fp32 fc2 GEMM — so kernel-vs-oracle disagreement is
    bounded by NVFP4 RTNE flips at fc1-out plus GEMM accumulation-order noise.
    """
    import torch

    from flashinfer.moe_ep.kernel_src.cutedsl_megamoe import (
        nvfp4_quantize_per_block_16,
    )

    num_tokens, topk = topk_idx.shape
    num_experts = fc1_weight.shape[0]

    act_fp32 = _dequant_nvfp4(act_packed, act_sf, logical_cols=hidden)

    out = torch.zeros(
        num_tokens, topk, hidden, dtype=torch.float32, device=act_fp32.device
    )
    for expert in range(num_experts):
        routing_mask = topk_idx == expert
        if not routing_mask.any():
            continue
        routed = routing_mask.nonzero(as_tuple=False)
        tokens, slots = routed[:, 0], routed[:, 1]

        fc1_w = _dequant_nvfp4(
            fc1_weight[expert], fc1_sf[expert], logical_cols=hidden
        )  # (2I, hidden)
        fc1_out = act_fp32[tokens] @ fc1_w.transpose(0, 1)  # (R, 2I)

        # SwiGLU over the 16-column gate/up interleave used by the NVFP4 kernel.
        m = fc1_out.shape[0]
        n_pairs = fc1_out.shape[1] // (2 * NVFP4_BLOCK)
        reshaped = fc1_out.view(m, n_pairs, 2, NVFP4_BLOCK)
        gate = reshaped[:, :, 0, :]
        up = reshaped[:, :, 1, :]
        if gate_up_clamp is not None:
            limit = abs(float(gate_up_clamp))
            gate = gate.clamp(max=limit)
            up = up.clamp(min=-limit, max=limit)
        swiglu = (gate * torch.sigmoid(gate) * up).reshape(m, intermediate)

        # apply_topk_in_fc1=True: weight folded in before the fp4 round-trip
        # (post-hoc weighting would NOT match — quant changes the magnitude).
        swiglu = swiglu * topk_weights[tokens, slots].unsqueeze(-1)

        fc1_q, fc1_q_sf = nvfp4_quantize_per_block_16(swiglu, 1.0)
        swiglu_rt = _dequant_nvfp4(fc1_q, fc1_q_sf, logical_cols=intermediate)

        fc2_w = _dequant_nvfp4(
            fc2_weight[expert], fc2_sf[expert], logical_cols=intermediate
        )  # (hidden, I)
        out[tokens, slots] = swiglu_rt @ fc2_w.transpose(0, 1)

    return out.sum(dim=1).to(torch.bfloat16)


@pytest.mark.arch_blackwell
def test_nvfp4_preprocess_fp4_weights_match_plain_quant():
    """``preprocess_mega_weights`` fp4 tensors match an independent plain quant."""
    _require_cuda()

    import torch

    from flashinfer.moe_ep import MoEWeightPack
    from flashinfer.moe_ep.backends.mega.kernel.nvfp4_cutedsl.weights import (
        preprocess_mega_weights,
    )
    from flashinfer.moe_ep.kernel_src.cutedsl_megamoe import to_blocked

    problem = _single_rank_problem()
    pack = MoEWeightPack(w13=problem["w13"], w2=problem["w2"])

    transformed_l1, transformed_l2 = preprocess_mega_weights(
        pack,
        intermediate_size=problem["intermediate"],
        hidden_size=problem["hidden"],
        gate_up_clamp=problem["gate_up_clamp"],
    )
    fc1_kernel, fc1_kernel_sf = transformed_l1
    fc2_kernel, fc2_kernel_sf = transformed_l2

    fc1_plain, fc1_sf, fc2_plain, fc2_sf = _plain_nvfp4_from_bf16(problem)

    # Kernel weights are logically (E, K//2, N) but MUST keep the packed K
    # axis stride-1 (transpose view over K-major memory) — the kernel's TMA
    # descriptors read K-major, so a materialized N-stride-1 tensor scrambles
    # every weight. Pin the stride contract explicitly.
    assert fc1_kernel.stride(1) == 1, (
        f"fc1 packed-K axis must be stride-1, got strides {fc1_kernel.stride()}"
    )
    assert fc2_kernel.stride(1) == 1, (
        f"fc2 packed-K axis must be stride-1, got strides {fc2_kernel.stride()}"
    )
    # Values: compare the K-major memory against the independent plain quant.
    torch.testing.assert_close(
        fc1_kernel.transpose(1, 2).contiguous().view(torch.uint8),
        fc1_plain,
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(
        fc2_kernel.transpose(1, 2).contiguous().view(torch.uint8),
        fc2_plain,
        atol=0,
        rtol=0,
    )
    # Scales: preprocess swizzles the plain SF per expert.
    num_experts = fc1_plain.shape[0]
    for e in range(num_experts):
        torch.testing.assert_close(
            fc1_kernel_sf[e].view(torch.uint8).reshape(-1),
            to_blocked(fc1_sf[e]).view(torch.uint8).reshape(-1),
            atol=0,
            rtol=0,
        )
        torch.testing.assert_close(
            fc2_kernel_sf[e].view(torch.uint8).reshape(-1),
            to_blocked(fc2_sf[e]).view(torch.uint8).reshape(-1),
            atol=0,
            rtol=0,
        )


@pytest.mark.arch_blackwell
def test_nvfp4_kernel_matches_torch_reference(monkeypatch):
    """Single-rank ``nvfp4_mega_moe`` output matches the pure-torch oracle."""
    _require_cuda()

    import torch

    cap = torch.cuda.get_device_capability()
    if cap[0] != 10:
        pytest.skip(
            f"nvfp4_mega_moe requires sm_100a or sm_103a; got sm_{cap[0]}{cap[1]}"
        )
    pytest.importorskip("triton")

    from flashinfer.moe_ep import MoEWeightPack
    from flashinfer.moe_ep.backends.mega.kernel.nvfp4_cutedsl.staging import (
        stage_mega_moe_inputs,
    )
    from flashinfer.moe_ep.backends.mega.kernel.nvfp4_cutedsl.weights import (
        preprocess_mega_weights,
    )
    from flashinfer.moe_ep.kernel_src.cutedsl_megamoe import (
        get_symm_buffer_for_mega_moe,
        nvfp4_mega_moe,
    )

    # monkeypatch (not os.environ): restored after the test, so it cannot
    # silently downgrade later nvshmem-path tests in the same process.
    monkeypatch.setenv("MEGA_NO_DIST", "1")
    problem = _single_rank_problem()
    rank = 0
    world_size = 1
    num_tokens = problem["num_tokens"]

    pack = MoEWeightPack(w13=problem["w13"], w2=problem["w2"])
    transformed_l1, transformed_l2 = preprocess_mega_weights(
        pack,
        intermediate_size=problem["intermediate"],
        hidden_size=problem["hidden"],
        gate_up_clamp=problem["gate_up_clamp"],
    )

    fc1_plain, fc1_sf, fc2_plain, fc2_sf = _plain_nvfp4_from_bf16(problem)

    # NOTE: the nvfp4 shim's ``intermediate`` is the fc1 output width (2*I),
    # matching the backend's ``2 * intermediate_size`` convention.
    symm_buffer = get_symm_buffer_for_mega_moe(
        problem["num_experts"],
        problem["max_tokens"],
        problem["topk"],
        problem["hidden"],
        2 * problem["intermediate"],
        rank,
        world_size,
        gate_up_clamp=problem["gate_up_clamp"],
    )
    try:
        stage_mega_moe_inputs(
            problem["hidden_states"],
            problem["topk_weights"],
            problem["topk_ids"],
            symm_buffer.x,
            symm_buffer.x_sf,
            symm_buffer.topk_idx,
            symm_buffer.topk_weights,
        )

        y_ref = _torch_nvfp4_mega_reference(
            act_packed=symm_buffer.x[:num_tokens],
            act_sf=symm_buffer.x_sf[:num_tokens],
            topk_idx=symm_buffer.topk_idx[:num_tokens],
            topk_weights=symm_buffer.topk_weights[:num_tokens],
            fc1_weight=fc1_plain,
            fc1_sf=fc1_sf,
            fc2_weight=fc2_plain,
            fc2_sf=fc2_sf,
            hidden=problem["hidden"],
            intermediate=problem["intermediate"],
            gate_up_clamp=problem["gate_up_clamp"],
        )

        y_kernel = torch.empty(
            num_tokens, problem["hidden"], dtype=torch.bfloat16, device="cuda"
        )
        nvfp4_mega_moe(
            y_kernel,
            transformed_l1,
            transformed_l2,
            symm_buffer,
            num_tokens=num_tokens,
            gate_up_clamp=problem["gate_up_clamp"],
        )
        torch.cuda.synchronize()

        assert torch.isfinite(y_kernel).all()
        yk = y_kernel.to(torch.float32)
        yr = y_ref.to(torch.float32)
        rel_l2 = (yk - yr).norm() / yr.norm().clamp_min(1e-6)
        print(
            f"[nvfp4 oracle] rel_l2={rel_l2.item():.4g} "
            f"max|Δ|={(yk - yr).abs().max().item():.4g} "
            f"amax(ref)={yr.abs().max().item():.4g}"
        )
        # The oracle shares the kernel's quantized operands, so the residual is
        # NVFP4 RTNE flips at fc1-out + accumulation-order noise (measured
        # rel_l2≈0.0027 on GB200; kernel is bit-exact vs the CuTeDSL reference
        # launcher on the same operands). atol scales with the output range
        # (random unscaled weights put |y|~1e4 here).
        atol = 2e-3 * yr.abs().max().item()
        torch.testing.assert_close(yk, yr, atol=atol, rtol=0.05)
        assert rel_l2.item() < 0.02
    finally:
        symm_buffer.destroy()
