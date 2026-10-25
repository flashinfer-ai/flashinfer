"""Independent BF16/FP32 W4A16 oracle shared by MegaMoE and split tests."""

from __future__ import annotations

import functools
import ctypes

import pytest
import torch
import torch.distributed as dist


def make_w4a16_weights(hidden, intermediate, experts, rank=0):
    """Directly sample packed E2M1 codes and finite positive E4M3 scales."""
    from flashinfer.moe_ep import PrequantizedMoEWeights

    generator = torch.Generator(device="cuda").manual_seed(20260915 + rank)
    tensors = []
    for rows, columns in ((2 * intermediate, hidden), (hidden, intermediate)):
        packed = torch.randint(
            256,
            (experts, rows, columns // 2),
            dtype=torch.uint8,
            device="cuda",
            generator=generator,
        )
        # 0.25 .. 1.875, with varied mantissas; no BF16 weight quantizer.
        scales = torch.randint(
            0x28,
            0x40,
            (experts, rows, columns // 16),
            dtype=torch.uint8,
            device="cuda",
            generator=generator,
        ).view(torch.float8_e4m3fn)
        tensors.append((packed, scales))
    return PrequantizedMoEWeights(
        w13=tensors[0][0],
        w13_scale=tensors[0][1],
        w2=tensors[1][0],
        w2_scale=tensors[1][1],
    )


def _dequantize(packed, scales):
    lookup = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6],
        dtype=torch.float32,
        device=packed.device,
    )
    packed = packed.view(torch.uint8)
    codes = torch.stack((packed & 15, packed >> 4), dim=-1).long()
    values = lookup[codes].reshape(*scales.shape, 16)
    return (values * scales.float().unsqueeze(-1)).flatten(-2).bfloat16()


@functools.cache
def _swiglu_kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def kernel(
        FC1,
        SCORES,
        NORM,
        OUT,
        I: tl.constexpr,
        N: tl.constexpr,
        CLAMP: tl.constexpr,
        WEIGHTED: tl.constexpr,
        SWIGLU_ALPHA: tl.constexpr,
        SWIGLU_BETA: tl.constexpr,
        SITU_BETA: tl.constexpr,
        SITU_LINEAR_BETA: tl.constexpr,
    ):
        index = tl.program_id(0) * 256 + tl.arange(0, 256)
        offset = (index // I) * (2 * I) + index % I
        gate = tl.load(FC1 + offset, index < N, other=0)
        up = tl.load(FC1 + offset + I, index < N, other=0)
        if CLAMP is not None:
            limit = tl.full((), CLAMP, tl.float32)
            gate = tl.inline_asm_elementwise(
                "min.NaN.f32 $0, $1, $2;",
                constraints="=f,f,f",
                args=[gate, limit],
                dtype=tl.float32,
                is_pure=True,
                pack=1,
            )
            up = tl.inline_asm_elementwise(
                "{ .reg .f32 lo; min.NaN.f32 $0, $1, $2; "
                "neg.f32 lo, $2; max.NaN.f32 $0, $0, lo; }",
                constraints="=f,f,f",
                args=[up, limit],
                dtype=tl.float32,
                is_pure=True,
                pack=1,
            )
        if SITU_BETA is not None:
            # Match the upstream exp2/rcp tanh approximation, not libdevice tanh.
            tanh_asm: tl.constexpr = """{
                .reg .f32 x, exponent, denominator, inverse, result;
                mul.rn.f32 x, $1, $3;
                mul.rn.f32 x, x, 0fC038AA3B;
                ex2.approx.ftz.f32 exponent, x;
                add.rn.f32 denominator, exponent, 0f3F800000;
                rcp.approx.ftz.f32 inverse, denominator;
                mul.rn.f32 result, inverse, 0f40000000;
                sub.rn.f32 result, result, 0f3F800000;
                mul.rn.f32 $0, $2, result;
            }"""
            bounded_gate = tl.inline_asm_elementwise(
                tanh_asm,
                constraints="=f,f,f,f",
                args=[
                    gate,
                    tl.full((), SITU_BETA, tl.float32),
                    tl.full((), 1.0 / SITU_BETA, tl.float32),
                ],
                dtype=tl.float32,
                is_pure=True,
                pack=1,
            )
            if SITU_LINEAR_BETA is not None:
                up = tl.inline_asm_elementwise(
                    tanh_asm,
                    constraints="=f,f,f,f",
                    args=[
                        up,
                        tl.full((), SITU_LINEAR_BETA, tl.float32),
                        tl.full((), 1.0 / SITU_LINEAR_BETA, tl.float32),
                    ],
                    dtype=tl.float32,
                    is_pure=True,
                    pack=1,
                )
            activated = tl.inline_asm_elementwise(
                """{
                    .reg .f32 neg, exponent, denominator, sigmoid, bounded;
                    mul.rn.f32 neg, $1, 0fBFB8AA3B;
                    ex2.approx.ftz.f32 exponent, neg;
                    add.rn.f32 denominator, exponent, 0f3F800000;
                    rcp.approx.ftz.f32 sigmoid, denominator;
                    mul.rn.f32 bounded, $3, sigmoid;
                    mul.rn.f32 $0, $2, bounded;
                }""",
                constraints="=f,f,f,f",
                args=[gate, up, bounded_gate],
                dtype=tl.float32,
                is_pure=True,
                pack=1,
            )
        elif SWIGLU_ALPHA is not None:
            activated = tl.inline_asm_elementwise(
                """{
                    .reg .f32 neg, exponent, denominator, sigmoid, shifted, product;
                    mul.rn.f32 neg, $1, $3;
                    ex2.approx.ftz.f32 exponent, neg;
                    add.rn.f32 denominator, exponent, 0f3F800000;
                    rcp.approx.ftz.f32 sigmoid, denominator;
                    add.rn.f32 shifted, $2, $4;
                    mul.rn.f32 product, shifted, $1;
                    mul.rn.f32 $0, product, sigmoid;
                }""",
                constraints="=f,f,f,f,f",
                args=[
                    gate,
                    up,
                    tl.full((), -SWIGLU_ALPHA * 1.4426950408889634, tl.float32),
                    tl.full((), SWIGLU_BETA, tl.float32),
                ],
                dtype=tl.float32,
                is_pure=True,
                pack=1,
            )
        else:
            # Preserve the original default SwiGLU operation association.
            activated = tl.inline_asm_elementwise(
                """{
                .reg .f32 neg, exponent, denominator, sigmoid, silu;
                mul.rn.f32 neg, $1, 0fBFB8AA3B;
                ex2.approx.ftz.f32 exponent, neg;
                add.rn.f32 denominator, exponent, 0f3F800000;
                rcp.approx.ftz.f32 sigmoid, denominator;
                mul.rn.f32 silu, $1, sigmoid;
                mul.rn.f32 $0, $2, silu;
            }""",
                constraints="=f,f,f",
                args=[gate, up],
                dtype=tl.float32,
                is_pure=True,
                pack=1,
            )
        if WEIGHTED:
            score = tl.load(SCORES + index // I, index < N, other=0)
            activated = tl.inline_asm_elementwise(
                "mul.rn.f32 $0, $1, $2;",
                constraints="=f,f,f",
                args=[activated, score],
                dtype=tl.float32,
                is_pure=True,
                pack=1,
            )
        if NORM is not None:
            activated = tl.inline_asm_elementwise(
                "mul.rn.f32 $0, $1, $2;",
                constraints="=f,f,f",
                args=[activated, tl.load(NORM)],
                dtype=tl.float32,
                is_pure=True,
                pack=1,
            )
        tl.store(
            OUT + index,
            activated.to(tl.bfloat16, fp_downcast_rounding="rtne"),
            index < N,
        )

    return kernel


def _gather_rows(tensor, counts):
    if len(counts) == 1:
        return tensor
    padded = torch.zeros(
        (max(counts), *tensor.shape[1:]), dtype=tensor.dtype, device=tensor.device
    )
    padded[: tensor.shape[0]].copy_(tensor)
    parts = [torch.empty_like(padded) for _ in counts]
    dist.all_gather(parts, padded)
    return torch.cat([part[:n] for part, n in zip(parts, counts, strict=True)])


@torch.no_grad()
def _megamoe_reference(
    problem,
    weights,
    *,
    fc1_alpha=None,
    fc2_alpha=None,
    fc1_norm_const=None,
    apply_routing_weights_before_fc2=False,
    in_kernel_fc2_reduce=False,
):
    """Compute on expert owners, transfer BF16 route bits, combine on sources.

    IKR returns weighted BF16 route terms for a separate atomic-rounding band;
    deterministic mode returns the finite, ordered FP32-combine BF16 result.
    """
    activation_kind = problem.get("activation", "swiglu")
    assert activation_kind in ("swiglu", "situ"), activation_kind
    assert (problem.get("situ_beta") is not None) == (activation_kind == "situ")
    matmul = torch.backends.cuda.matmul
    if not hasattr(matmul, "allow_bf16_reduced_precision_reduction_split_k"):
        pytest.skip("Bit-exact W4A16 oracle requires PyTorch 2.12 BF16 split-K control")
    pytest.importorskip("triton")
    x = problem["hidden_states"]
    rank, world = (
        (dist.get_rank(), dist.get_world_size()) if dist.is_initialized() else (0, 1)
    )
    counts = [None] * world
    if world > 1:
        dist.all_gather_object(counts, x.shape[0])
    else:
        counts[0] = x.shape[0]
    if not sum(counts):
        shape = (
            (0, problem["topk_ids"].shape[1], x.shape[1])
            if in_kernel_fc2_reduce
            else x.shape
        )
        return torch.empty(shape, dtype=torch.bfloat16, device=x.device)
    ids = _gather_rows(problem["topk_ids"], counts)
    scores = _gather_rows(problem["topk_weights"], counts)
    x = _gather_rows(x, counts)
    terms = torch.zeros(
        (sum(counts), ids.shape[1], x.shape[1]), dtype=torch.int32, device=x.device
    )
    experts = weights.w13.shape[0]
    fc1_alpha = torch.ones(experts, device=x.device) if fc1_alpha is None else fc1_alpha
    fc2_alpha = torch.ones(experts, device=x.device) if fc2_alpha is None else fc2_alpha
    previous_blas = torch.backends.cuda.preferred_blas_library()
    previous_reduction = (
        matmul.allow_bf16_reduced_precision_reduction,
        matmul.allow_bf16_reduced_precision_reduction_split_k,
    )
    previous_tf32 = matmul.allow_tf32
    try:
        torch.backends.cuda.preferred_blas_library("cublaslt")
        matmul.allow_bf16_reduced_precision_reduction = (False, False)
        matmul.allow_tf32 = False
        for expert in range(experts):
            rows, slots = torch.where(ids == rank * experts + expert)
            if not rows.numel():
                continue
            w13 = _dequantize(weights.w13[expert], weights.w13_scale[expert])
            w2 = _dequantize(weights.w2[expert], weights.w2_scale[expert])
            for start in range(0, rows.numel(), 256):
                batch, route_slots = (
                    rows[start : start + 256],
                    slots[start : start + 256],
                )
                fc1 = torch.mm(x[batch], w13.T, out_dtype=torch.float32)
                fc1.mul_(fc1_alpha[expert])
                intermediate = w2.shape[1]
                activation = torch.empty(
                    (batch.numel(), intermediate), dtype=torch.bfloat16, device=x.device
                )
                routing = (
                    scores[batch, route_slots]
                    if apply_routing_weights_before_fc2
                    else None
                )
                _swiglu_kernel()[((activation.numel() + 255) // 256,)](
                    fc1,
                    routing,
                    fc1_norm_const[expert] if fc1_norm_const is not None else None,
                    activation,
                    intermediate,
                    activation.numel(),
                    problem["gate_up_clamp"],
                    apply_routing_weights_before_fc2,
                    problem.get("swiglu_alpha"),
                    problem.get("swiglu_beta"),
                    problem.get("situ_beta"),
                    problem.get("situ_linear_beta"),
                )
                fc2 = torch.mm(activation, w2.T, out_dtype=torch.float32)
                fc2.mul_(fc2_alpha[expert])
                term = fc2.bfloat16()
                if in_kernel_fc2_reduce and not apply_routing_weights_before_fc2:
                    term = (term.float() * scores[batch, route_slots, None]).bfloat16()
                terms[batch, route_slots] = term.view(torch.int16).to(torch.int32)
    finally:
        matmul.allow_bf16_reduced_precision_reduction = previous_reduction
        torch.backends.cuda.preferred_blas_library(previous_blas)
        matmul.allow_tf32 = previous_tf32
    # Exactly one expert owner contributes each route: integer SUM transports
    # BF16 bits (including signed zero), without a floating-point reduction.
    if world > 1:
        dist.all_reduce(terms)
    offset = sum(counts[:rank])
    terms = terms[offset : offset + counts[rank]].to(torch.int16).view(torch.bfloat16)
    if in_kernel_fc2_reduce:
        return terms
    scores = problem["topk_weights"]
    output = terms[:, 0].float()
    if not apply_routing_weights_before_fc2:
        output = output * scores[:, 0, None]
    for slot in range(1, ids.shape[1]):
        if apply_routing_weights_before_fc2:
            output = output + terms[:, slot].float()
        else:
            # PyTorch 2.12 addcmul(value=1) uses FP32 std::fma on CUDA.
            output = torch.addcmul(
                output, terms[:, slot].float(), scores[:, slot, None], value=1
            )
    return output.bfloat16()


def assert_w4a16_bits(actual, expected):
    assert actual.dtype == expected.dtype == torch.bfloat16
    assert actual.shape == expected.shape
    assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
    torch.testing.assert_close(
        actual.view(torch.int16), expected.view(torch.int16), rtol=0, atol=0
    )


def canonical_split_w4a16_weights(tensors, gated):
    """Decode split's packed layout without its production dequantizer.

    Preserve the actual E2M1 codes and E4M3 scales; only undo the scale layout
    and the gated FC1 row interleave. The oracle then dequantizes these bytes.
    """
    from flashinfer.moe_ep import PrequantizedMoEWeights

    def unpack(name):
        packed = tensors[f"{name}_weight"]
        experts, rows, packed_k = packed.shape
        assert packed_k % 8 == 0
        # Logical scale coordinates are (row%32, row//32%4, row//128,
        # block%4, block//4, expert). Crop padding before reordering FC1.
        padded_rows = ((rows + 127) // 128) * 128
        padded_blocks = ((2 * packed_k + 63) // 64) * 4
        scales = (
            tensors[f"{name}_weight_sf"]
            .permute(5, 2, 1, 0, 4, 3)
            .contiguous()
            .view(torch.uint8)
            .reshape(experts, padded_rows, padded_blocks)
        )[:, :rows, : packed_k // 8].contiguous()
        if name == "w1" and gated:
            # Split stores [up64,gate64,...]; the oracle consumes
            # canonical [all_gate,all_up], including matching scales.
            assert rows % 128 == 0

            def gate_first(value):
                return (
                    value.reshape(experts, rows // 128, 2, 64, value.shape[-1])
                    .transpose(1, 2)
                    .flip(1)
                    .reshape_as(value)
                    .contiguous()
                )

            packed, scales = gate_first(packed), gate_first(scales)
        return packed, scales.view(torch.float8_e4m3fn)

    w13, w13_scale = unpack("w1")
    w2, w2_scale = unpack("w2")
    return PrequantizedMoEWeights(w13, w2, w13_scale, w2_scale)


def prepare_split_w4a16_exact_fused(tensors):
    """Partition FC2 output channels for exact fused tests with top-k > 2.

    Every expert retains arbitrary packed weights on its owned channels, but
    at most two experts contribute a nonzero term to each output coordinate.
    With distinct routed experts, the zero-seeded BF16 atomic sum is therefore
    independent of arrival order. Dense unfused and top-k <= 2 fused tests do
    not need this fixture and retain their original weights.
    """
    weight = tensors["w2_weight"]
    experts, hidden, _ = weight.shape
    assert 0 < experts <= 2 * hidden
    expert = torch.arange(experts, device=weight.device)[:, None]
    column = torch.arange(hidden, device=weight.device)[None, :]
    owns_channel = column % min(experts, hidden) == expert % hidden
    weight.masked_fill_(~owns_channel[:, :, None], 0)


def split_w4a16_reference_from_tensors(
    tensors,
    activation_type=None,
    use_fused_finalize=True,
    local_expert_offset=0,
    swiglu_alpha=1.0,
    swiglu_beta=0.0,
    swiglu_limit=3.4028234663852886e38,
    situ_beta=None,
    situ_linear_beta=None,
    token_selected_experts=None,
    token_final_scales=None,
    wrong_formula=False,
):
    """Adapt the existing split test tensor pack to the shared exact oracle.

    The activation defaults to SwiGLU and finalization defaults to fused, as
    in the split functional API. Routing overrides support local expert slices
    and graph tests whose live routing differs from the initial tensor pack.
    ``wrong_formula`` replaces ReLU2 with plain ReLU as a negative control.
    """
    from flashinfer.tllm_enums import ActivationType

    activation_type = (
        ActivationType.Swiglu if activation_type is None else activation_type
    )
    if activation_type == ActivationType.Swiglu:
        activation = "swiglu"
    elif activation_type == ActivationType.GegluTanh:
        activation = "geglu_tanh"
    elif activation_type == ActivationType.Relu2:
        activation = "relu2"
    elif activation_type == ActivationType.Situ:
        activation = "situ"
    else:
        raise ValueError(f"Unsupported split W4A16 activation: {activation_type}")
    if situ_beta is not None:
        assert activation in ("swiglu", "situ")
        activation = "situ"
    if wrong_formula:
        assert activation == "relu2"
        activation = "relu"
    problem = {
        "hidden_states": tensors["x_bf16"],
        "topk_ids": (
            tensors["token_selected_experts"]
            if token_selected_experts is None
            else token_selected_experts
        ),
        "topk_weights": (
            tensors["token_final_scales"]
            if token_final_scales is None
            else token_final_scales
        ),
        "activation": activation,
        "gate_up_clamp": swiglu_limit,
        "swiglu_alpha": swiglu_alpha,
        "swiglu_beta": swiglu_beta,
        "situ_beta": situ_beta,
        "situ_linear_beta": situ_linear_beta,
    }
    return w4a16_reference(
        problem,
        canonical_split_w4a16_weights(tensors, activation not in ("relu2", "relu")),
        fc1_alpha=tensors["w1_alpha"],
        fc2_alpha=tensors["w2_alpha"],
        backend="split",
        local_expert_offset=local_expert_offset,
        use_fused_finalize=use_fused_finalize,
    )


@functools.cache
def _split_combine_kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def kernel(
        TERMS,
        SCORES,
        LOCAL,
        OUT,
        H: tl.constexpr,
        K: tl.constexpr,
        N: tl.constexpr,
        FUSED: tl.constexpr,
    ):
        index = tl.program_id(0) * 256 + tl.arange(0, 256)
        token, column = index // H, index % H
        acc = tl.full((256,), 0.0, tl.float32)
        for slot in tl.static_range(K):
            valid = tl.load(LOCAL + token * K + slot, index < N, other=False)
            value = tl.load(
                TERMS + (token * K + slot) * H + column,
                (index < N) & valid,
                other=0,
            ).to(tl.float32)
            if FUSED:
                # Terms have already been weighted and rounded to BF16.
                updated = (
                    tl.inline_asm_elementwise(
                        "add.rn.f32 $0, $1, $2;",
                        constraints="=f,f,f",
                        args=[acc, value],
                        dtype=tl.float32,
                        is_pure=True,
                        pack=1,
                    )
                    .to(tl.bfloat16, fp_downcast_rounding="rtne")
                    .to(tl.float32)
                )
            else:
                score = tl.load(
                    SCORES + token * K + slot, (index < N) & valid, other=0
                ).to(tl.float32)
                if slot == 0:
                    updated = tl.inline_asm_elementwise(
                        "mul.rn.f32 $0, $1, $2;",
                        constraints="=f,f,f",
                        args=[value, score],
                        dtype=tl.float32,
                        is_pure=True,
                        pack=1,
                    )
                else:
                    updated = tl.inline_asm_elementwise(
                        "fma.rn.f32 $0, $1, $2, $3;",
                        constraints="=f,f,f,f",
                        args=[value, score, acc],
                        dtype=tl.float32,
                        is_pure=True,
                        pack=1,
                    )
            # Skipping an invalid route must preserve signed zero and must not
            # read its uninitialized term or poisoned routing score.
            acc = tl.where(valid, updated, acc)
        tl.store(
            OUT + index, acc.to(tl.bfloat16, fp_downcast_rounding="rtne"), index < N
        )

    return kernel


def combine_w4a16_split_terms(
    terms_bf16, scores_fp32, local_mask_bool, use_fused_finalize=False
):
    """Combine canonical [token, slot, hidden] rows with exact split semantics.

    Deterministic mode uses FP32 multiplication for slot zero and FP32 FMA for
    later slots, without FTZ, then one BF16 cast. Invalid slots are skipped.
    Fused mode accepts already weighted BF16 terms and reproduces zero-seeded
    BF16 atomic addition only for at most two nonzero local terms per output
    coordinate. Zero additions leave this finite sum unchanged, including its
    zero-seeded +0 result. Larger unordered sums lack a unique bit-exact oracle.
    """
    assert terms_bf16.dtype == torch.bfloat16 and terms_bf16.ndim == 3
    assert scores_fp32.dtype == torch.float32
    assert local_mask_bool.dtype == torch.bool
    assert terms_bf16.shape[:2] == scores_fp32.shape == local_mask_bool.shape
    if use_fused_finalize:
        nonzero = (terms_bf16 != 0) & local_mask_bool[:, :, None]
        if torch.any(nonzero.sum(dim=1) > 2).item():
            raise ValueError(
                "Exact fused split combine supports at most two nonzero local "
                "terms per output coordinate"
            )
    terms, scores, local = (
        terms_bf16.contiguous(),
        scores_fp32.contiguous(),
        local_mask_bool.contiguous(),
    )
    tokens, topk, hidden = terms.shape
    output = torch.empty((tokens, hidden), dtype=torch.bfloat16, device=terms.device)
    if output.numel():
        _split_combine_kernel()[((output.numel() + 255) // 256,)](
            terms,
            scores,
            local,
            output,
            hidden,
            topk,
            output.numel(),
            use_fused_finalize,
        )
    return output


@functools.cache
def _split_activation_kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def tanh_reference(x):
        return tl.inline_asm_elementwise(
            """{
                .reg .f32 doubled, exponent, denominator, inverse;
                mul.rn.f32 doubled, $1, 0fC0000000;
                mul.rn.f32 exponent, doubled, 0f3FB8AA3B;
                ex2.approx.ftz.f32 exponent, exponent;
                add.rn.f32 denominator, exponent, 0f3F800000;
                rcp.approx.ftz.f32 inverse, denominator;
                fma.rn.f32 $0, inverse, 0f40000000, 0fBF800000;
            }""",
            constraints="=f,f",
            args=[x],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )

    @triton.jit
    def kernel(
        FC1,
        OUT,
        I: tl.constexpr,
        N: tl.constexpr,
        ACTIVATION: tl.constexpr,
        PARAMETERIZED: tl.constexpr,
        NEG_ALPHA_LOG2E: tl.constexpr,
        BETA: tl.constexpr,
        CLAMP: tl.constexpr,
        SITU_BETA: tl.constexpr,
        SITU_INV_BETA: tl.constexpr,
        LINEAR_BETA: tl.constexpr,
        LINEAR_INV_BETA: tl.constexpr,
    ):
        index = tl.program_id(0) * 256 + tl.arange(0, 256)
        if ACTIVATION == "relu2":
            value = tl.load(FC1 + index, index < N, other=0)
            result = tl.inline_asm_elementwise(
                "{ .reg .f32 r; max.f32 r, $1, 0f00000000; mul.rn.f32 $0, r, r; }",
                constraints="=f,f",
                args=[value],
                dtype=tl.float32,
                is_pure=True,
                pack=1,
            )
        elif ACTIVATION == "relu":
            # Test-only counterfactual for the ReLU2 activation contract.
            value = tl.load(FC1 + index, index < N, other=0)
            result = tl.inline_asm_elementwise(
                "max.f32 $0, $1, 0f00000000;",
                constraints="=f,f",
                args=[value],
                dtype=tl.float32,
                is_pure=True,
                pack=1,
            )
        elif (
            ACTIVATION == "geglu_tanh" or ACTIVATION == "swiglu" or ACTIVATION == "situ"
        ):
            offset = (index // I) * (2 * I) + index % I
            gate = tl.load(FC1 + offset, index < N, other=0)
            up = tl.load(FC1 + offset + I, index < N, other=0)
            if ACTIVATION == "geglu_tanh":
                inner = tl.inline_asm_elementwise(
                    """{
                        .reg .f32 cubic, sum;
                        mul.rn.f32 cubic, $1, $2;
                        mul.rn.f32 cubic, cubic, $1;
                        fma.rn.f32 sum, $1, cubic, $1;
                        mul.rn.f32 $0, sum, $3;
                    }""",
                    constraints="=f,f,f,f",
                    args=[
                        gate,
                        tl.full((), 0.044715, tl.float32),
                        tl.full((), 0.7978845608028654, tl.float32),
                    ],
                    dtype=tl.float32,
                    is_pure=True,
                    pack=1,
                )
                bounded = tanh_reference(inner)
                result = tl.inline_asm_elementwise(
                    """{
                        .reg .f32 half, shifted, activated;
                        mul.rn.f32 half, $1, 0f3F000000;
                        add.rn.f32 shifted, $2, 0f3F800000;
                        mul.rn.f32 activated, half, shifted;
                        mul.rn.f32 $0, $3, activated;
                    }""",
                    constraints="=f,f,f,f",
                    args=[gate, bounded, up],
                    dtype=tl.float32,
                    is_pure=True,
                    pack=1,
                )
            elif ACTIVATION == "swiglu" or ACTIVATION == "situ":
                if PARAMETERIZED:
                    gate, up = tl.inline_asm_elementwise(
                        """{
                            .reg .f32 low;
                            min.NaN.f32 $0, $2, $4;
                            min.NaN.f32 $1, $3, $4;
                            neg.f32 low, $4;
                            max.NaN.f32 $1, $1, low;
                        }""",
                        constraints="=f,=f,f,f,f",
                        args=[gate, up, tl.full((), CLAMP, tl.float32)],
                        dtype=(tl.float32, tl.float32),
                        is_pure=True,
                        pack=1,
                    )
                sigmoid = tl.inline_asm_elementwise(
                    """{
                        .reg .f32 exponent, denominator;
                        mul.rn.f32 exponent, $1, $2;
                        ex2.approx.ftz.f32 exponent, exponent;
                        add.rn.f32 denominator, exponent, 0f3F800000;
                        rcp.approx.ftz.f32 $0, denominator;
                    }""",
                    constraints="=f,f,f",
                    args=[gate, tl.full((), NEG_ALPHA_LOG2E, tl.float32)],
                    dtype=tl.float32,
                    is_pure=True,
                    pack=1,
                )
                if ACTIVATION == "situ":
                    scaled = tl.inline_asm_elementwise(
                        "mul.rn.f32 $0, $1, $2;",
                        constraints="=f,f,f",
                        args=[gate, tl.full((), SITU_INV_BETA, tl.float32)],
                        dtype=tl.float32,
                        is_pure=True,
                        pack=1,
                    )
                    bounded = tanh_reference(scaled)
                    gate = tl.inline_asm_elementwise(
                        "mul.rn.f32 $0, $1, $2;",
                        constraints="=f,f,f",
                        args=[bounded, tl.full((), SITU_BETA, tl.float32)],
                        dtype=tl.float32,
                        is_pure=True,
                        pack=1,
                    )
                    if LINEAR_BETA is not None:
                        scaled_up = tl.inline_asm_elementwise(
                            "mul.rn.f32 $0, $1, $2;",
                            constraints="=f,f,f",
                            args=[up, tl.full((), LINEAR_INV_BETA, tl.float32)],
                            dtype=tl.float32,
                            is_pure=True,
                            pack=1,
                        )
                        bounded_up = tanh_reference(scaled_up)
                        up = tl.inline_asm_elementwise(
                            "mul.rn.f32 $0, $1, $2;",
                            constraints="=f,f,f",
                            args=[bounded_up, tl.full((), LINEAR_BETA, tl.float32)],
                            dtype=tl.float32,
                            is_pure=True,
                            pack=1,
                        )
                if PARAMETERIZED:
                    up = tl.inline_asm_elementwise(
                        "add.rn.f32 $0, $1, $2;",
                        constraints="=f,f,f",
                        args=[up, tl.full((), BETA, tl.float32)],
                        dtype=tl.float32,
                        is_pure=True,
                        pack=1,
                    )
                # Split parameterized SwiGLU retains up * (gate * sigmoid),
                # unlike Mega's ((up + beta) * gate) * sigmoid association.
                result = tl.inline_asm_elementwise(
                    "{ .reg .f32 silu; mul.rn.f32 silu, $1, $2; mul.rn.f32 $0, $3, silu; }",
                    constraints="=f,f,f,f",
                    args=[gate, sigmoid, up],
                    dtype=tl.float32,
                    is_pure=True,
                    pack=1,
                )
            else:
                tl.static_assert(False, "Unsupported gated split W4A16 activation")
        else:
            tl.static_assert(False, "Unsupported split W4A16 activation")
        tl.store(
            OUT + index, result.to(tl.bfloat16, fp_downcast_rounding="rtne"), index < N
        )

    return kernel


@torch.no_grad()
def _split_reference(
    problem, weights, fc1_alpha, fc2_alpha, local_expert_offset, use_fused_finalize
):
    """Single-rank split oracle from canonical gate-first/up-second weights."""
    if dist.is_initialized() and dist.get_world_size() != 1:
        raise ValueError("The split reference supports a single rank")
    activation_kind = problem["activation"]
    if activation_kind not in ("swiglu", "geglu_tanh", "relu2", "relu", "situ"):
        raise ValueError(f"Unsupported split W4A16 activation: {activation_kind}")
    matmul = torch.backends.cuda.matmul
    if not hasattr(matmul, "allow_bf16_reduced_precision_reduction_split_k"):
        pytest.skip("Bit-exact W4A16 oracle requires PyTorch 2.12 BF16 split-K control")
    pytest.importorskip("triton")
    x, ids, scores = (problem[k] for k in ("hidden_states", "topk_ids", "topk_weights"))
    assert scores.dtype == torch.float32
    experts = weights.w13.shape[0]
    offset = 0 if local_expert_offset is None else local_expert_offset
    local = (ids >= offset) & (ids < offset + experts)
    terms = torch.zeros((*ids.shape, x.shape[1]), dtype=torch.bfloat16, device=x.device)
    fc1_alpha = torch.ones(experts, device=x.device) if fc1_alpha is None else fc1_alpha
    fc2_alpha = torch.ones(experts, device=x.device) if fc2_alpha is None else fc2_alpha
    assert fc1_alpha.dtype == fc2_alpha.dtype == torch.float32

    f32 = lambda value: ctypes.c_float(float(value)).value
    swiglu_alpha = problem.get("swiglu_alpha")
    swiglu_beta = problem.get("swiglu_beta")
    swiglu_alpha = 1.0 if swiglu_alpha is None else swiglu_alpha
    swiglu_beta = 0.0 if swiglu_beta is None else swiglu_beta
    clamp = problem.get("gate_up_clamp")
    clamp = torch.finfo(torch.float32).max if clamp is None else clamp
    parameterized = activation_kind == "swiglu" and (
        swiglu_alpha != 1.0
        or swiglu_beta != 0.0
        or clamp != torch.finfo(torch.float32).max
    )
    beta = problem.get("situ_beta")
    linear_beta = problem.get("situ_linear_beta")
    assert (beta is not None) == (activation_kind == "situ")
    neg_alpha_log2e = -f32(f32(swiglu_alpha) * f32(1.4426950408889634))
    previous_blas = torch.backends.cuda.preferred_blas_library()
    previous_reduction = (
        matmul.allow_bf16_reduced_precision_reduction,
        matmul.allow_bf16_reduced_precision_reduction_split_k,
    )
    previous_tf32 = matmul.allow_tf32
    try:
        torch.backends.cuda.preferred_blas_library("cublaslt")
        matmul.allow_bf16_reduced_precision_reduction = (False, False)
        matmul.allow_tf32 = False
        for expert in range(experts):
            rows, slots = torch.where(ids == offset + expert)
            if not rows.numel():
                continue
            w13 = _dequantize(weights.w13[expert], weights.w13_scale[expert])
            w2 = _dequantize(weights.w2[expert], weights.w2_scale[expert])
            intermediate = w2.shape[1]
            assert w13.shape[0] == intermediate * (
                1 if activation_kind in ("relu2", "relu") else 2
            )
            for start in range(0, rows.numel(), 256):
                batch, route_slots = (
                    rows[start : start + 256],
                    slots[start : start + 256],
                )
                fc1 = torch.mm(x[batch], w13.T, out_dtype=torch.float32)
                fc1.mul_(fc1_alpha[expert])
                activation = torch.empty(
                    (batch.numel(), intermediate), dtype=torch.bfloat16, device=x.device
                )
                _split_activation_kernel()[((activation.numel() + 255) // 256,)](
                    fc1,
                    activation,
                    intermediate,
                    activation.numel(),
                    activation_kind,
                    parameterized,
                    neg_alpha_log2e,
                    swiglu_beta,
                    clamp,
                    beta,
                    None if beta is None else f32(1.0 / f32(beta)),
                    linear_beta,
                    None if linear_beta is None else f32(1.0 / f32(linear_beta)),
                )
                fc2 = torch.mm(activation, w2.T, out_dtype=torch.float32)
                if use_fused_finalize:
                    # Split combines the two FP32 scales before weighting the
                    # raw accumulator; there is no unweighted BF16 FC2 cast.
                    scale = fc2_alpha[expert] * scores[batch, route_slots]
                    term = (fc2 * scale[:, None]).bfloat16()
                else:
                    fc2.mul_(fc2_alpha[expert])
                    term = fc2.bfloat16()
                terms[batch, route_slots] = term
    finally:
        matmul.allow_bf16_reduced_precision_reduction = previous_reduction
        torch.backends.cuda.preferred_blas_library(previous_blas)
        matmul.allow_tf32 = previous_tf32
    return combine_w4a16_split_terms(
        terms, scores, local, use_fused_finalize=use_fused_finalize
    )


def w4a16_reference(
    problem,
    weights,
    fc1_alpha=None,
    fc2_alpha=None,
    fc1_norm_const=None,
    apply_routing_weights_before_fc2=False,
    in_kernel_fc2_reduce=False,
    backend="megamoe",
    local_expert_offset=None,
    use_fused_finalize=False,
):
    """Shared exact oracle with explicit MegaMoE and split numerical contracts.

    Both modes accept canonical logical weights, not a production packed layout.
    The default MegaMoE path retains its existing distributed and IKR behavior.
    Split mode supports local expert slices on one rank; its fused reference is
    limited to two nonzero terms per coordinate because larger BF16 sums reorder.
    """
    if backend == "megamoe":
        if local_expert_offset is not None or use_fused_finalize:
            raise ValueError("Split-only options require backend='split'")
        return _megamoe_reference(
            problem,
            weights,
            fc1_alpha=fc1_alpha,
            fc2_alpha=fc2_alpha,
            fc1_norm_const=fc1_norm_const,
            apply_routing_weights_before_fc2=apply_routing_weights_before_fc2,
            in_kernel_fc2_reduce=in_kernel_fc2_reduce,
        )
    elif backend == "split":
        if (
            fc1_norm_const is not None
            or apply_routing_weights_before_fc2
            or in_kernel_fc2_reduce
        ):
            raise ValueError(
                "MegaMoE-only options are not supported by backend='split'"
            )
        return _split_reference(
            problem,
            weights,
            fc1_alpha=fc1_alpha,
            fc2_alpha=fc2_alpha,
            local_expert_offset=local_expert_offset,
            use_fused_finalize=use_fused_finalize,
        )
    else:
        raise ValueError(f"Unknown W4A16 reference backend: {backend}")
