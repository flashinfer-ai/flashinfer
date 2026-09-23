"""Independent BF16/FP32 oracle for deterministic W4A16 MegaMoE tests."""

from __future__ import annotations

import functools

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
def w4a16_reference(
    problem,
    weights,
    *,
    fc1_alpha=None,
    fc2_alpha=None,
    fc1_norm_const=None,
    apply_topk_in_fc1=False,
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
                routing = scores[batch, route_slots] if apply_topk_in_fc1 else None
                _swiglu_kernel()[((activation.numel() + 255) // 256,)](
                    fc1,
                    routing,
                    fc1_norm_const[expert] if fc1_norm_const is not None else None,
                    activation,
                    intermediate,
                    activation.numel(),
                    problem["gate_up_clamp"],
                    apply_topk_in_fc1,
                    problem.get("swiglu_alpha"),
                    problem.get("swiglu_beta"),
                    problem.get("situ_beta"),
                    problem.get("situ_linear_beta"),
                )
                fc2 = torch.mm(activation, w2.T, out_dtype=torch.float32)
                fc2.mul_(fc2_alpha[expert])
                term = fc2.bfloat16()
                if in_kernel_fc2_reduce and not apply_topk_in_fc1:
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
    if not apply_topk_in_fc1:
        output = output * scores[:, 0, None]
    for slot in range(1, ids.shape[1]):
        if apply_topk_in_fc1:
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
