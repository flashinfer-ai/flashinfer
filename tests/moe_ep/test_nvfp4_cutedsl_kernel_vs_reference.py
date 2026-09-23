"""Single-GPU checks: NVFP4 ``nvfp4_mega_moe`` vs a pure-torch oracle.

NVFP4 counterpart of ``test_mxfp8_cutedsl_preprocess_vs_reference.py``: validates
that ``sm100_nvfp4_nvfp4_bf16_cutedsl.preprocess_mega_weights`` produces fp4 weights consistent
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
pytest.importorskip("flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe")

NVFP4_BLOCK = 16


def _require_cuda():
    import torch

    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")


def _single_rank_problem(
    hidden=2048,
    intermediate=1024,
    *,
    num_experts=4,
    topk=4,
    num_tokens=32,
    max_tokens=64,
):
    import torch

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


def _bf16_ulp(x: float) -> float:
    """bf16 spacing at magnitude ``x`` (8 significand bits incl. the hidden one)."""
    import math

    if x <= 0.0:
        return 0.0
    return 2.0 ** (math.floor(math.log2(x)) - 7)


def _oracle_atol(amax_ref: float) -> float:
    """Absolute tolerance for kernel-vs-oracle compares.

    ``2e-3 * amax`` covers NVFP4 RTNE flips at fc1-out plus GEMM accumulation-
    order noise (measured rel_l2 <= 3e-3).  The extra ``2 * bf16_ulp(amax)``
    covers one bf16 rounding flip of a per-(token, topk) fc2 term: both the
    kernel and the oracle store each term in bf16 before the top-k sum, so an
    fp32 accumulation-order difference at a rounding boundary shows up as one
    term ulp even where the reduced output is small (terms cancel), which the
    relative part of the compare cannot absorb.
    """
    return 2e-3 * amax_ref + 2.0 * _bf16_ulp(amax_ref)


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

    from flashinfer.moe_ep.backends.mega.kernel.sm100.nvfp4_nvfp4_bf16_cutedsl.weights import (
        _interleave_gate_up_16,
    )
    from flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe import (
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
    term_transform=None,
    swiglu_alpha=None,
    swiglu_beta=None,
    fc1_activation_per_token_scale=None,  # (T,) fp32 or None
    fc1_alpha=None,  # (E,) fp32 per-expert or None (identity)
    fc2_alpha=None,  # (E,) fp32 per-expert or None (identity)
    fc1_norm_const=None,  # (E,) fp32 per-expert or None (1.0)
):
    """Pure-torch NVFP4 MegaMoE oracle (apply_topk_in_fc1=True graph).

    Mirrors the kernel's data path — dequant → fp32 fc1 GEMM → 16-interleaved
    SwiGLU fold (+clamp) → per-token topk weight folded in BEFORE the fc1-out
    NVFP4 round-trip → fp32 fc2 GEMM → bf16 terms — so disagreement is
    bounded by NVFP4 RTNE flips at fc1-out plus GEMM accumulation-order noise.

    ``term_transform``, when set, is applied to each per-(token, topk) fc2
    output term before the topk sum; the multirank oracle uses it to model the
    quantized cross-rank combine wire (``combine_roundtrip_to_fp32``).

    ``fc1_activation_per_token_scale``, when set, multiplies token ``t``'s
    dequantized fc1 output by ``scale[t]`` BEFORE the clamp / activation --
    the kernel's ``enable_fc1_activation_per_token_scale`` fold
    (``real = fc1_alpha * scale_t * acc``).

    ``fc1_alpha`` / ``fc2_alpha`` / ``fc1_norm_const`` model the per-expert
    epilogue scalars with the kernel's semantics: ``fc1_alpha[e]`` is folded
    into the same single fp32 multiply as the per-token scale, the fc1-out
    NVFP4 round-trip uses ``fc1_norm_const[e]`` (host quantizer mirrors the
    kernel: SF = amax/6*c, data = y*c/SF, so the dequant is ``y*c`` and is NOT
    divided back), and ``fc2_alpha[e]`` scales the fp32 fc2 accumulator
    before the bf16 store.  ``None`` keeps the identity path.
    """
    import torch

    from flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe import (
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
        # Dequant fold: one fp32 multiply by ``fc1_alpha[e] * scale_t`` (the
        # kernel forms the product first, then multiplies the accumulator once),
        # applied BEFORE the clamp / activation.
        dequant = None
        if fc1_alpha is not None:
            dequant = fc1_alpha[expert].to(torch.float32)
        if fc1_activation_per_token_scale is not None:
            per_token = fc1_activation_per_token_scale[tokens].to(torch.float32)
            dequant = per_token if dequant is None else dequant * per_token
        if dequant is not None:
            if dequant.dim() == 0:
                fc1_out = fc1_out * dequant
            else:
                fc1_out = fc1_out * dequant.unsqueeze(-1)

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
        alpha = 1.0 if swiglu_alpha is None else swiglu_alpha
        beta = 0.0 if swiglu_beta is None else swiglu_beta
        swiglu = (gate * torch.sigmoid(alpha * gate) * (up + beta)).reshape(
            m, intermediate
        )

        # apply_topk_in_fc1=True: weight folded in before the fp4 round-trip
        # (post-hoc weighting would NOT match — quant changes the magnitude).
        swiglu = swiglu * topk_weights[tokens, slots].unsqueeze(-1)

        norm_const = 1.0 if fc1_norm_const is None else float(fc1_norm_const[expert])
        fc1_q, fc1_q_sf = nvfp4_quantize_per_block_16(swiglu, norm_const)
        # With a norm const the dequant is ``swiglu * norm_const`` (in range);
        # the kernel does not divide it back -- the caller folds 1/c into
        # fc2_alpha on the host.
        swiglu_rt = _dequant_nvfp4(fc1_q, fc1_q_sf, logical_cols=intermediate)

        fc2_w = _dequant_nvfp4(
            fc2_weight[expert], fc2_sf[expert], logical_cols=intermediate
        )  # (hidden, I)
        fc2_out = swiglu_rt @ fc2_w.transpose(0, 1)
        if fc2_alpha is not None:
            fc2_out = fc2_out * fc2_alpha[expert].to(torch.float32)
        # FC2 stores each expert term in BF16 before the top-k reduction.
        fc2_out = fc2_out.to(torch.bfloat16).float()
        if term_transform is not None:
            fc2_out = term_transform(fc2_out)
        out[tokens, slots] = fc2_out

    return out.sum(dim=1).to(torch.bfloat16)


@pytest.mark.arch_blackwell
def test_nvfp4_preprocess_fp4_weights_match_plain_quant():
    """``preprocess_mega_weights`` fp4 tensors match an independent plain quant."""
    _require_cuda()

    import torch

    from flashinfer.moe_ep import MoEWeightPack
    from flashinfer.moe_ep.backends.mega.kernel.sm100.nvfp4_nvfp4_bf16_cutedsl.weights import (
        preprocess_mega_weights,
    )
    from flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe import to_blocked

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
@pytest.mark.parametrize(
    "hidden,intermediate,num_experts,topk,activation_clamp",
    [
        pytest.param(2048, 1024, 4, 4, "standard", id="regular-e4"),
        # 128-misaligned (hidden % 128 == 64): exercises the ceil-div K-tail
        # and predicated epilogue paths the %64 validation relaxation opened
        # up (gpt-oss-120b geometry class).
        pytest.param(2880, 2880, 4, 4, "standard", id="tail-e4"),
        pytest.param(2048, 1024, 1, 1, "standard", id="singleton-e1"),
        pytest.param(2048, 1024, 4, 4, 0.5, id="minimax-clamp"),
        pytest.param(2048, 1024, 4, 4, None, id="minimax-no-clamp"),
    ],
)
def test_nvfp4_kernel_matches_torch_reference(
    monkeypatch, hidden, intermediate, num_experts, topk, activation_clamp
):
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
    from flashinfer.moe_ep.backends.mega.kernel.sm100.nvfp4_nvfp4_bf16_cutedsl.staging import (
        stage_mega_moe_inputs,
    )
    from flashinfer.moe_ep.backends.mega.kernel.sm100.nvfp4_nvfp4_bf16_cutedsl.weights import (
        preprocess_mega_weights,
    )
    from flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe import (
        get_symm_buffer_for_mega_moe,
        nvfp4_mega_moe,
    )

    # monkeypatch (not os.environ): restored after the test, so it cannot
    # silently downgrade later nvshmem-path tests in the same process.
    monkeypatch.setenv("MEGA_NO_DIST", "1")
    problem = _single_rank_problem(
        hidden=hidden,
        intermediate=intermediate,
        num_experts=num_experts,
        topk=topk,
    )
    activation_pairs = [(None, None)]
    if activation_clamp != "standard":
        problem["gate_up_clamp"] = activation_clamp
        # Keep gates near zero so ignoring alpha changes the output visibly.
        # A 0.5 clamp exercises both saturated and unsaturated gate/up values.
        problem["hidden_states"].mul_(0.1)
        problem["w13"].mul_(0.1)
        activation_pairs = [(1.702, 1.0), (1.0, 1.0), (1.0, 0.0), (1.702, 1.0)]
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
        swiglu_alpha=activation_pairs[0][0],
        swiglu_beta=activation_pairs[0][1],
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

        for step, (alpha, beta) in enumerate(activation_pairs):
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
                swiglu_alpha=alpha,
                swiglu_beta=beta,
            )

            y_kernel = torch.empty(
                num_tokens, problem["hidden"], dtype=torch.bfloat16, device="cuda"
            )
            nvfp4_mega_moe(
                y_kernel,
                transformed_l1,
                transformed_l2,
                symm_buffer,
                **({} if step == 0 else dict(swiglu_alpha=alpha, swiglu_beta=beta)),
                num_tokens=num_tokens,
                gate_up_clamp=problem["gate_up_clamp"],
            )
            torch.cuda.synchronize()

            assert torch.isfinite(y_kernel).all()
            yk = y_kernel.to(torch.float32)
            yr = y_ref.to(torch.float32)
            rel_l2 = (yk - yr).norm() / yr.norm().clamp_min(1e-6)
            print(
                f"[nvfp4 oracle alpha={alpha} beta={beta}] rel_l2={rel_l2.item():.4g} "
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


@pytest.mark.arch_blackwell
@pytest.mark.parametrize(
    "hidden,intermediate,num_experts,topk,num_tokens,activation,fc1_alpha",
    [
        # 32 tokens routed to every expert: exactly the first (pre-transpose,
        # shuffle-fed) half of one 64-token epilogue subtile per expert.
        pytest.param(2048, 1024, 4, 4, 32, "standard", None, id="regular-e4"),
        # 128-misaligned hidden: ceil-div K tail + predicated epilogue.
        pytest.param(2880, 2880, 4, 4, 32, "standard", None, id="tail-e4"),
        pytest.param(2048, 1024, 4, 4, 32, "no-clamp", None, id="no-clamp"),
        # MiniMax-style (1.702, 1.0) activation with a tight clamp: mixed
        # saturated / unsaturated gate-up values, so the pre-activation
        # position of the scale is observable.
        pytest.param(2048, 1024, 4, 4, 32, "minimax", None, id="minimax-clamp"),
        # ~96 tokens per expert (192 tokens, topk 2 of 4): a full 64-token
        # subtile (both the shuffle-fed 0..31 half and the one-scale-per-lane
        # 32..63 half carry valid tokens) plus a partial second subtile.
        pytest.param(2048, 1024, 4, 2, 192, "standard", None, id="two-subtiles"),
        # Same routing with a non-identity per-expert fc1_alpha: exercises the
        # ``alpha * scale_t`` fold on both halves (the oracle folds 2*scale).
        pytest.param(2048, 1024, 4, 2, 192, "standard", 2.0, id="alpha-fold"),
        # Singleton expert (static expert extent 1) with 3 subtiles of tokens.
        pytest.param(2048, 1024, 1, 1, 192, "standard", None, id="singleton-e1"),
    ],
)
def test_nvfp4_kernel_per_token_scale_matches_torch_reference(
    monkeypatch,
    hidden,
    intermediate,
    num_experts,
    topk,
    num_tokens,
    activation,
    fc1_alpha,
):
    """``enable_fc1_activation_per_token_scale``: the kernel folds a runtime
    ``(T,)`` fp32 per-token scale into the fc1 dequant before the clamp /
    activation, matching the pure-torch oracle with the same fold.

    The epilogue applies the scale on two register schedules per 64-token
    subtile: tokens 0..31 are activated before the register transpose (scale
    fetched per register via warp shuffle) and tokens 32..63 after it (one
    scale per lane).  The 32-token cases cover the first half only; the
    192-token cases put valid tokens on both halves and across subtile
    boundaries.  Random {0.5, 1, 1.5, 2} scales differ per token, so a
    lane/token mix-up shows up as a large error.  ``fc1_alpha`` (broadcast
    per-expert scalar) checks the ``alpha * scale_t`` fold: the oracle has no
    alpha leg, so it is fed ``alpha * scale`` instead (exact in fp32).
    """
    _require_cuda()

    import torch

    cap = torch.cuda.get_device_capability()
    if cap[0] != 10:
        pytest.skip(
            f"nvfp4_mega_moe requires sm_100a or sm_103a; got sm_{cap[0]}{cap[1]}"
        )
    pytest.importorskip("triton")

    from flashinfer.moe_ep import MoEWeightPack
    from flashinfer.moe_ep.backends.mega.kernel.sm100.nvfp4_nvfp4_bf16_cutedsl.staging import (
        stage_mega_moe_inputs,
    )
    from flashinfer.moe_ep.backends.mega.kernel.sm100.nvfp4_nvfp4_bf16_cutedsl.weights import (
        preprocess_mega_weights,
    )
    from flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe import (
        get_symm_buffer_for_mega_moe,
        nvfp4_mega_moe,
    )

    monkeypatch.setenv("MEGA_NO_DIST", "1")
    problem = _single_rank_problem(
        hidden=hidden,
        intermediate=intermediate,
        num_experts=num_experts,
        topk=topk,
        num_tokens=num_tokens,
        max_tokens=max(64, num_tokens),
    )
    swiglu_alpha, swiglu_beta = None, None
    if activation == "no-clamp":
        problem["gate_up_clamp"] = None
    elif activation == "minimax":
        problem["gate_up_clamp"] = 0.5
        problem["hidden_states"].mul_(0.1)
        problem["w13"].mul_(0.1)
        swiglu_alpha, swiglu_beta = 1.702, 1.0
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

    symm_buffer = get_symm_buffer_for_mega_moe(
        problem["num_experts"],
        problem["max_tokens"],
        problem["topk"],
        problem["hidden"],
        2 * problem["intermediate"],
        rank,
        world_size,
        swiglu_alpha=swiglu_alpha,
        swiglu_beta=swiglu_beta,
        gate_up_clamp=problem["gate_up_clamp"],
        fc1_alpha=fc1_alpha,
        enable_fc1_activation_per_token_scale=True,
    )
    try:
        scale_buf = symm_buffer.fc1_activation_per_token_scale
        assert scale_buf is not None
        assert scale_buf.shape == (problem["max_tokens"],)
        assert scale_buf.dtype == torch.float32
        # Allocation-time default is the neutral scale.
        assert torch.equal(scale_buf, torch.ones_like(scale_buf))

        stage_mega_moe_inputs(
            problem["hidden_states"],
            problem["topk_weights"],
            problem["topk_ids"],
            symm_buffer.x,
            symm_buffer.x_sf,
            symm_buffer.topk_idx,
            symm_buffer.topk_weights,
        )
        # Kernel-team tester value set {0.5, 1.0, 1.5, 2.0}: exactly
        # representable, per-token distinct, and far from the neutral 1.0.
        g = torch.Generator(device="cuda").manual_seed(23)
        scale = (
            torch.randint(1, 5, (num_tokens,), generator=g, device="cuda").to(
                torch.float32
            )
            * 0.5
        )
        scale_buf[:num_tokens].copy_(scale)

        ref_kwargs = dict(
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
            swiglu_alpha=swiglu_alpha,
            swiglu_beta=swiglu_beta,
        )
        # The oracle has no fc1_alpha leg: fold the (power-of-two) broadcast
        # alpha into the per-token scale it is fed -- exact in fp32.
        ref_scale = scale if fc1_alpha is None else scale * fc1_alpha
        y_ref = _torch_nvfp4_mega_reference(
            **ref_kwargs, fc1_activation_per_token_scale=ref_scale
        )
        y_ref_unscaled = _torch_nvfp4_mega_reference(**ref_kwargs)

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
        yu = y_ref_unscaled.to(torch.float32)
        rel_l2 = (yk - yr).norm() / yr.norm().clamp_min(1e-6)
        rel_l2_unscaled = (yk - yu).norm() / yu.norm().clamp_min(1e-6)
        print(
            f"[nvfp4 per-token-scale oracle {activation} tokens={num_tokens} "
            f"experts={num_experts} topk={topk} alpha={fc1_alpha}] "
            f"rel_l2={rel_l2.item():.4g} "
            f"rel_l2(vs unscaled ref)={rel_l2_unscaled.item():.4g} "
            f"max|d|={(yk - yr).abs().max().item():.4g} "
            f"amax(ref)={yr.abs().max().item():.4g}"
        )
        # Same tolerance model as the unscaled oracle test (see _oracle_atol):
        # NVFP4 RTNE flips at fc1-out + accumulation-order noise + one bf16
        # term-rounding flip.
        atol = _oracle_atol(yr.abs().max().item())
        torch.testing.assert_close(yk, yr, atol=atol, rtol=0.05)
        assert rel_l2.item() < 0.02
        # The check has teeth: a kernel that ignored the scale (or applied it
        # to the wrong tokens) would sit far from the scaled reference.
        assert rel_l2_unscaled.item() > 0.05, (
            "per-token scale had no observable effect -- test is not discriminating"
        )
    finally:
        symm_buffer.destroy()


@pytest.mark.arch_blackwell
@pytest.mark.parametrize(
    "epilogue",
    [
        # Random per-expert fc1_alpha / fc2_alpha / fc1_norm_const (the same
        # value sets as make_dummy_epilogue_params) on top of the per-token
        # scale: checks the alpha*scale_t fold, the norm-const SF shift and
        # the fc2 alpha against an independent reference.
        pytest.param("random", id="random-scalars"),
        # Deployment recipe from the kernel-team design: quantize the fc1
        # output with norm const 16 (SF shifted up 4 binades) and fold the
        # 1/16 into fc2_alpha on the host.  Small-magnitude MiniMax-style
        # values keep every SF inside the e4m3 normal range, where this must
        # be indistinguishable from norm const 1 with the unfolded fc2_alpha.
        pytest.param("deploy-norm16", id="deploy-norm16"),
    ],
)
def test_nvfp4_kernel_per_token_scale_with_epilogue_scalars(monkeypatch, epilogue):
    """Per-token scale combined with non-identity per-expert epilogue scalars,
    vs the pure-torch oracle carrying the same scalars."""
    _require_cuda()

    import torch

    cap = torch.cuda.get_device_capability()
    if cap[0] != 10:
        pytest.skip(
            f"nvfp4_mega_moe requires sm_100a or sm_103a; got sm_{cap[0]}{cap[1]}"
        )
    pytest.importorskip("triton")

    from flashinfer.moe_ep import MoEWeightPack
    from flashinfer.moe_ep.backends.mega.kernel.sm100.nvfp4_nvfp4_bf16_cutedsl.staging import (
        stage_mega_moe_inputs,
    )
    from flashinfer.moe_ep.backends.mega.kernel.sm100.nvfp4_nvfp4_bf16_cutedsl.weights import (
        preprocess_mega_weights,
    )
    from flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe import (
        get_symm_buffer_for_mega_moe,
        make_dummy_epilogue_params,
        nvfp4_mega_moe,
    )

    monkeypatch.setenv("MEGA_NO_DIST", "1")
    num_experts, topk, num_tokens = 4, 2, 192
    problem = _single_rank_problem(
        num_experts=num_experts, topk=topk, num_tokens=num_tokens, max_tokens=192
    )
    swiglu_alpha, swiglu_beta = None, None
    g = torch.Generator(device="cuda").manual_seed(29)
    fc1_alpha, fc2_alpha, fc1_norm_const = make_dummy_epilogue_params(
        num_experts, generator=g
    )
    if epilogue == "deploy-norm16":
        problem["gate_up_clamp"] = 0.5
        problem["hidden_states"].mul_(0.1)
        problem["w13"].mul_(0.1)
        swiglu_alpha, swiglu_beta = 1.702, 1.0
        fc2_alpha_unfolded = fc2_alpha
        fc1_norm_const = torch.full_like(fc1_norm_const, 16.0)
        fc2_alpha = fc2_alpha_unfolded / 16.0  # exact (power of two)

    pack = MoEWeightPack(w13=problem["w13"], w2=problem["w2"])
    transformed_l1, transformed_l2 = preprocess_mega_weights(
        pack,
        intermediate_size=problem["intermediate"],
        hidden_size=problem["hidden"],
        gate_up_clamp=problem["gate_up_clamp"],
    )
    fc1_plain, fc1_sf, fc2_plain, fc2_sf = _plain_nvfp4_from_bf16(problem)

    symm_buffer = get_symm_buffer_for_mega_moe(
        problem["num_experts"],
        problem["max_tokens"],
        problem["topk"],
        problem["hidden"],
        2 * problem["intermediate"],
        0,
        1,
        swiglu_alpha=swiglu_alpha,
        swiglu_beta=swiglu_beta,
        gate_up_clamp=problem["gate_up_clamp"],
        fc1_alpha=fc1_alpha,
        fc2_alpha=fc2_alpha,
        fc1_norm_const=fc1_norm_const,
        enable_fc1_activation_per_token_scale=True,
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
        scale = (
            torch.randint(1, 5, (num_tokens,), generator=g, device="cuda").to(
                torch.float32
            )
            * 0.5
        )
        symm_buffer.fc1_activation_per_token_scale[:num_tokens].copy_(scale)

        ref_kwargs = dict(
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
            swiglu_alpha=swiglu_alpha,
            swiglu_beta=swiglu_beta,
            fc1_activation_per_token_scale=scale,
            fc1_alpha=fc1_alpha,
        )
        # Exact kernel math: norm const in the SF, fc2_alpha applied as-is.
        y_ref = _torch_nvfp4_mega_reference(
            **ref_kwargs, fc2_alpha=fc2_alpha, fc1_norm_const=fc1_norm_const
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
        msg = (
            f"[nvfp4 per-token-scale + epilogue scalars {epilogue}] "
            f"rel_l2={rel_l2.item():.4g} max|d|={(yk - yr).abs().max().item():.4g} "
            f"amax(ref)={yr.abs().max().item():.4g}"
        )
        atol = _oracle_atol(yr.abs().max().item())
        torch.testing.assert_close(yk, yr, atol=atol, rtol=0.05)
        assert rel_l2.item() < 0.02

        if epilogue == "deploy-norm16":
            # The recipe is a pure SF-exponent shift (+4 binades): for every
            # block whose SF stays inside the e4m3 normal range under both
            # norm consts it reproduces the norm-const-1 / unfolded-fc2_alpha
            # result bit for bit, so the two oracles differ only through the
            # blocks the shift is meant to help (SF subnormal / flushed at c=1)
            # or hurt (SF saturated at 448 for c=16; excluded here by the small
            # magnitudes).  Bound that residual instead of demanding equality.
            y_plain = _torch_nvfp4_mega_reference(
                **ref_kwargs, fc2_alpha=fc2_alpha_unfolded, fc1_norm_const=None
            )
            yp = y_plain.to(torch.float32)
            rel_l2_recipe = (yr - yp).norm() / yp.norm().clamp_min(1e-6)
            msg += f" recipe-vs-plain rel_l2={rel_l2_recipe.item():.3g}"
            assert rel_l2_recipe.item() < 0.02, (
                "norm16 + fc2_alpha/16 deviates from norm 1 + fc2_alpha by "
                f"rel_l2={rel_l2_recipe.item():.3g}: more than a few blocks left "
                "the e4m3 normal range"
            )
        print(msg)
    finally:
        symm_buffer.destroy()


def test_nvfp4_per_token_scale_forward_validation():
    """``MoEEpTensors.fc1_activation_per_token_scale`` must be present iff the
    kernel config enables it, and be ``(num_tokens,)`` fp32 CUDA."""
    _require_cuda()

    import torch

    from flashinfer.moe_ep import MoEEpConfigError
    from flashinfer.moe_ep.backends.mega.kernel.sm100.nvfp4_nvfp4_bf16_cutedsl.staging import (
        validate_fc1_activation_per_token_scale,
    )

    ok = torch.ones(8, dtype=torch.float32, device="cuda")
    validate_fc1_activation_per_token_scale(ok, num_tokens=8, enabled=True)
    validate_fc1_activation_per_token_scale(None, num_tokens=8, enabled=False)
    with pytest.raises(MoEEpConfigError, match="required"):
        validate_fc1_activation_per_token_scale(None, num_tokens=8, enabled=True)
    with pytest.raises(
        MoEEpConfigError, match="enable_fc1_activation_per_token_scale=False"
    ):
        validate_fc1_activation_per_token_scale(ok, num_tokens=8, enabled=False)
    with pytest.raises(MoEEpConfigError, match="shape"):
        validate_fc1_activation_per_token_scale(ok[:4], num_tokens=8, enabled=True)
    with pytest.raises(MoEEpConfigError, match="float32"):
        validate_fc1_activation_per_token_scale(
            ok.to(torch.bfloat16), num_tokens=8, enabled=True
        )
    with pytest.raises(MoEEpConfigError, match="CUDA"):
        validate_fc1_activation_per_token_scale(ok.cpu(), num_tokens=8, enabled=True)


def test_nvfp4_shim_per_token_scale_input_validation():
    """The shim rejects a per-token scale that disagrees with the session flag
    (present-but-disabled, missing-but-enabled, wrong shape / dtype) before
    any compile or launch."""
    _require_cuda()

    import torch

    from flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe import (
        MegaMoENvfp4Config,
        MegaMoENvfp4Inputs,
    )
    from flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe.shim.nvfp4 import (
        MegaMoENvfp4Frontend,
    )

    tokens, topk, experts, hidden, intermediate = 64, 2, 4, 256, 256
    base = dict(
        rank=0,
        world_size=1,
        num_tokens_per_rank=tokens,
        num_topk=topk,
        num_total_experts=experts,
        hidden=hidden,
        intermediate=intermediate,
    )
    dev = "cuda"
    e = experts

    def make_inputs(scale):
        return MegaMoENvfp4Inputs(
            activation=torch.zeros(
                tokens, hidden // 2, dtype=torch.uint8, device=dev
            ).view(torch.float4_e2m1fn_x2),
            activation_sf=torch.zeros(
                tokens, hidden // 16, dtype=torch.float8_e4m3fn, device=dev
            ),
            topk_idx=torch.zeros(tokens, topk, dtype=torch.int64, device=dev),
            topk_weights=torch.zeros(tokens, topk, dtype=torch.float32, device=dev),
            fc1_weight=torch.zeros(
                e, hidden // 2, intermediate, dtype=torch.uint8, device=dev
            ).view(torch.float4_e2m1fn_x2),
            fc1_weight_sf=torch.zeros(e, 8, dtype=torch.float8_e4m3fn, device=dev),
            fc2_weight=torch.zeros(
                e, intermediate // 4, hidden, dtype=torch.uint8, device=dev
            ).view(torch.float4_e2m1fn_x2),
            fc2_weight_sf=torch.zeros(e, 8, dtype=torch.float8_e4m3fn, device=dev),
            fc1_alpha=torch.ones(e, dtype=torch.float32, device=dev),
            fc2_alpha=torch.ones(e, dtype=torch.float32, device=dev),
            fc1_norm_const=torch.ones(e, dtype=torch.float32, device=dev),
            output_activation=torch.zeros(
                tokens, hidden, dtype=torch.bfloat16, device=dev
            ),
            fc1_activation_per_token_scale=scale,
        )

    scale = torch.ones(tokens, dtype=torch.float32, device=dev)
    on = MegaMoENvfp4Frontend(
        MegaMoENvfp4Config(**base, enable_fc1_activation_per_token_scale=True)
    )
    off = MegaMoENvfp4Frontend(MegaMoENvfp4Config(**base))

    on._validate_inputs(make_inputs(scale), num_tokens=tokens)
    off._validate_inputs(make_inputs(None), num_tokens=tokens)
    with pytest.raises(ValueError, match="required"):
        on._validate_inputs(make_inputs(None), num_tokens=tokens)
    with pytest.raises(ValueError, match="enable_fc1_activation_per_token_scale=False"):
        off._validate_inputs(make_inputs(scale), num_tokens=tokens)
    with pytest.raises(ValueError, match="shape"):
        on._validate_inputs(make_inputs(scale[: tokens // 2]), num_tokens=tokens)
    with pytest.raises(ValueError, match="float32"):
        on._validate_inputs(make_inputs(scale.to(torch.bfloat16)), num_tokens=tokens)
