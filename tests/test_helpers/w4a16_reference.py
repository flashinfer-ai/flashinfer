"""Bit-exact, zero-tolerance reference for the following W4A16 contracts.

``assert_w4a16_bits`` requires finite BF16 outputs with identical storage bits
(including signed zero): ``atol=0`` and ``rtol=0`` on their integer views.

Supported production backends and APIs:

* Dense: ``flashinfer.mm_bf16_fp4(..., backend="cute-dsl")`` on
  SM100/103/107, using standard full-K tactics and the prepared six-dimensional
  scale layout. ``dense_w4a16_reference`` applies FP32 alpha after the GEMM
  and rounds the result to BF16.
* Split MoE: ``flashinfer.cute_dsl_fused_moe`` and
  ``flashinfer.CuteDslMoEWrapper`` with ``quant_mode="w4a16"`` and
  ``use_fused_finalize=False``. The same contract covers
  ``flashinfer.fused_moe.MoELayer`` with ``CuteDslConfig`` (backend key
  ``"cute_dsl"``), NVFP4 weights/BF16 activations, and unfused finalization.
* MegaMoE W4A16: ``bf16_nvfp4_mega_moe`` in
  ``flashinfer.moe_ep.cute_dsl.megamoe.bf16_nvfp4``, and
  ``flashinfer.moe_ep.MoEEpMegaLayer`` with ``MegaConfig`` and
  ``Sm100_Bf16_Nvfp4_Bf16_Cutedsl_MegaMoeConfig``, using deterministic combine
  (``in_kernel_fc2_reduce=False``).

Dense split-K, stream-K, SM11x/SM12x kernels, GEMV, CAKE, and other backend
implementations are outside this exact contract. Split MoE fused TMA-reduce
finalization is also excluded. With MegaMoE ``in_kernel_fc2_reduce=True``, the
reference returns BF16 per-route terms, not an exact final atomic sum; tests
retain their existing tolerance band for the atomic BF16 combine.

All covered GEMMs share BF16 decoded weights, FP32 accumulation and separate
FP32 alpha scaling. The oracle disables reduced-precision reduction/split-K
in cuBLASLt; it requires PyTorch's BF16 split-K control and Triton for MoE.
"""

from __future__ import annotations

import functools
from contextlib import contextmanager

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


@contextmanager
def _w4a16_matmul_precision():
    matmul = torch.backends.cuda.matmul
    if not hasattr(matmul, "allow_bf16_reduced_precision_reduction_split_k"):
        pytest.skip("Bit-exact W4A16 oracle requires PyTorch 2.12 BF16 split-K control")
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
        yield
    finally:
        matmul.allow_bf16_reduced_precision_reduction = previous_reduction
        torch.backends.cuda.preferred_blas_library(previous_blas)
        matmul.allow_tf32 = previous_tf32


def _scaled_bf16_matmul(activation, weight, alpha):
    """FP32 accumulation and post-GEMM scaling, before any BF16 output rounding."""
    result = torch.mm(activation, weight.T, out_dtype=torch.float32)
    if alpha is not None:
        result.mul_(alpha)
    return result


def _unswizzle_mma_scales(scales, rows, k, groups):
    """Undo the shared dense/split six-dimensional scale view independently."""
    shape = (32, 4, (rows + 127) // 128, 4, (k + 63) // 64, groups)
    if tuple(scales.shape) != shape or k % 16:
        raise ValueError(f"Expected six-dimensional W4A16 scales {shape} and K%16=0")
    if scales.dtype not in (torch.uint8, torch.float8_e4m3fn):
        raise TypeError("W4A16 scales must contain E4M3 bytes")
    # Logical coordinates: (row%32, row//32%4, row//128,
    # block%4, block//4, group). Crop both padded dimensions.
    return (
        scales.permute(5, 2, 1, 0, 4, 3)
        .contiguous()
        .view(torch.uint8)
        .reshape(groups, shape[2] * 128, shape[4] * 4)
    )[:, :rows, : k // 16].contiguous()


@torch.no_grad()
def dense_w4a16_reference(a, packed, scales, alpha=None):
    """Reference for standard full-K SM10x CuTe DSL GEMM prepared weights.

    ``packed`` is (N, K/2) E2M1 and ``scales`` is the six-dimensional E4M3
    view shared with split MoE. Alpha scales the FP32 accumulator; the result
    rounds once to BF16. Split-K, stream-K, and other backend layouts are outside
    this contract.
    """
    if a.ndim != 2 or packed.ndim != 2:
        raise ValueError("Dense W4A16 activation and packed weights must be matrices")
    if a.dtype != torch.bfloat16 or packed.dtype != torch.uint8:
        raise TypeError("Dense W4A16 requires BF16 activation and uint8 packed weights")
    m, k = a.shape
    n, packed_k = packed.shape
    if n <= 0 or k <= 0 or k != 2 * packed_k or k % 16:
        raise ValueError("Dense W4A16 requires matching positive N/K and K%16=0")
    if not a.is_cuda or packed.device != a.device or scales.device != a.device:
        raise ValueError("Dense W4A16 inputs must share a CUDA device")
    if alpha is not None:
        if alpha.dtype != torch.float32 or alpha.numel() != 1:
            raise TypeError("Dense W4A16 alpha must be a single FP32 value")
        if alpha.device != a.device:
            raise ValueError("Dense W4A16 alpha must share the input CUDA device")
    linear_scales = _unswizzle_mma_scales(scales, n, k, 1)[0]
    weight = _dequantize(packed, linear_scales.view(torch.float8_e4m3fn))
    output = torch.empty((m, n), dtype=torch.bfloat16, device=a.device)
    with _w4a16_matmul_precision():
        for start in range(0, m, 256):
            output[start : start + 256] = _scaled_bf16_matmul(
                a[start : start + 256], weight, alpha
            ).bfloat16()
    return output


@functools.cache
def _activation_kernel():
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
        ACTIVATION: tl.constexpr,
        CLAMP: tl.constexpr,
        WEIGHTED: tl.constexpr,
        SWIGLU_ALPHA: tl.constexpr,
        SWIGLU_BETA: tl.constexpr,
        SITU_BETA: tl.constexpr,
        SITU_LINEAR_BETA: tl.constexpr,
    ):
        index = tl.program_id(0) * 256 + tl.arange(0, 256)
        if ACTIVATION == "relu2":
            value = tl.load(FC1 + index, index < N, other=0)
            activated = tl.inline_asm_elementwise(
                "{ .reg .f32 r; max.f32 r, $1, 0f00000000; mul.rn.f32 $0, r, r; }",
                constraints="=f,f",
                args=[value],
                dtype=tl.float32,
                is_pure=True,
                pack=1,
            )
        elif ACTIVATION == "relu":
            # Plain ReLU is a counterfactual used by the ReLU2 numerical test.
            value = tl.load(FC1 + index, index < N, other=0)
            activated = tl.inline_asm_elementwise(
                "max.f32 $0, $1, 0f00000000;",
                constraints="=f,f",
                args=[value],
                dtype=tl.float32,
                is_pure=True,
                pack=1,
            )
        elif (
            ACTIVATION == "swiglu" or ACTIVATION == "situ" or ACTIVATION == "geglu_tanh"
        ):
            offset = (index // I) * (2 * I) + index % I
            gate = tl.load(FC1 + offset, index < N, other=0)
            up = tl.load(FC1 + offset + I, index < N, other=0)
            if CLAMP is not None and (ACTIVATION == "swiglu" or ACTIVATION == "situ"):
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
            if ACTIVATION == "geglu_tanh":
                # Match the split kernel's cubic/FMA and exp2/rcp tanh sequence.
                activated = tl.inline_asm_elementwise(
                    """{
                        .reg .f32 cubic, inner, exponent, inverse, bounded, half;
                        mul.rn.f32 cubic, $1, $3;
                        mul.rn.f32 cubic, cubic, $1;
                        fma.rn.f32 inner, $1, cubic, $1;
                        mul.rn.f32 inner, inner, $4;
                        mul.rn.f32 exponent, inner, 0fC0000000;
                        mul.rn.f32 exponent, exponent, 0f3FB8AA3B;
                        ex2.approx.ftz.f32 exponent, exponent;
                        add.rn.f32 exponent, exponent, 0f3F800000;
                        rcp.approx.ftz.f32 inverse, exponent;
                        fma.rn.f32 bounded, inverse, 0f40000000, 0fBF800000;
                        mul.rn.f32 half, $1, 0f3F000000;
                        add.rn.f32 bounded, bounded, 0f3F800000;
                        mul.rn.f32 half, half, bounded;
                        mul.rn.f32 $0, $2, half;
                    }""",
                    constraints="=f,f,f,f,f",
                    args=[
                        gate,
                        up,
                        tl.full((), 0.044715, tl.float32),
                        tl.full((), 0.7978845608028654, tl.float32),
                    ],
                    dtype=tl.float32,
                    is_pure=True,
                    pack=1,
                )
            elif ACTIVATION == "situ":
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
            elif ACTIVATION == "swiglu" and SWIGLU_ALPHA is not None:
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
            elif ACTIVATION == "swiglu":
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
            else:
                tl.static_assert(False, "Unsupported gated W4A16 activation")
        else:
            tl.static_assert(False, "Unsupported W4A16 activation")
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
    fc1_alpha=None,
    fc2_alpha=None,
    fc1_norm_const=None,
    apply_routing_weights_before_fc2=False,
    in_kernel_fc2_reduce=False,
    local_expert_offset=None,
):
    """Compute on expert owners, transfer BF16 route bits, combine on sources.

    IKR returns weighted BF16 route terms for a separate atomic-rounding band;
    deterministic mode returns the finite, ordered FP32-combine BF16 result.
    """
    if "activation" in problem:
        activation_kind = problem["activation"]
    else:
        activation_kind = "swiglu"
    if activation_kind not in ("swiglu", "situ", "geglu_tanh", "relu2", "relu"):
        raise ValueError(f"Unsupported W4A16 activation: {activation_kind}")
    assert (problem.get("situ_beta") is not None) == (activation_kind == "situ")
    pytest.importorskip("triton")
    x = problem["hidden_states"]
    rank, world = (
        (dist.get_rank(), dist.get_world_size()) if dist.is_initialized() else (0, 1)
    )
    if world > 1 and local_expert_offset is not None:
        raise ValueError(
            "Explicit local expert offsets require a single-rank reference"
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
    expert_offset = (
        rank * experts if local_expert_offset is None else local_expert_offset
    )
    fc1_alpha = torch.ones(experts, device=x.device) if fc1_alpha is None else fc1_alpha
    fc2_alpha = torch.ones(experts, device=x.device) if fc2_alpha is None else fc2_alpha
    with _w4a16_matmul_precision():
        for expert in range(experts):
            rows, slots = torch.where(ids == expert_offset + expert)
            if not rows.numel():
                continue
            w13 = _dequantize(weights.w13[expert], weights.w13_scale[expert])
            w2 = _dequantize(weights.w2[expert], weights.w2_scale[expert])
            expected_rows = w2.shape[1] * (
                1 if activation_kind in ("relu2", "relu") else 2
            )
            assert w13.shape[0] == expected_rows
            for start in range(0, rows.numel(), 256):
                batch, route_slots = (
                    rows[start : start + 256],
                    slots[start : start + 256],
                )
                fc1 = _scaled_bf16_matmul(x[batch], w13, fc1_alpha[expert])
                intermediate = w2.shape[1]
                activation = torch.empty(
                    (batch.numel(), intermediate), dtype=torch.bfloat16, device=x.device
                )
                routing = (
                    scores[batch, route_slots]
                    if apply_routing_weights_before_fc2
                    else None
                )
                _activation_kernel()[((activation.numel() + 255) // 256,)](
                    fc1,
                    routing,
                    fc1_norm_const[expert] if fc1_norm_const is not None else None,
                    activation,
                    intermediate,
                    activation.numel(),
                    activation_kind,
                    problem["gate_up_clamp"],
                    apply_routing_weights_before_fc2,
                    problem.get("swiglu_alpha"),
                    problem.get("swiglu_beta"),
                    problem.get("situ_beta"),
                    problem.get("situ_linear_beta"),
                )
                fc2 = _scaled_bf16_matmul(activation, w2, fc2_alpha[expert])
                term = fc2.bfloat16()
                if in_kernel_fc2_reduce and not apply_routing_weights_before_fc2:
                    term = (term.float() * scores[batch, route_slots, None]).bfloat16()
                terms[batch, route_slots] = term.view(torch.int16).to(torch.int32)
    # Exactly one expert owner contributes each route: integer SUM transports
    # BF16 bits (including signed zero), without a floating-point reduction.
    if world > 1:
        dist.all_reduce(terms)
    offset = sum(counts[:rank])
    terms = terms[offset : offset + counts[rank]].to(torch.int16).view(torch.bfloat16)
    if in_kernel_fc2_reduce:
        return terms
    source_ids = problem["topk_ids"]
    first_expert = 0 if local_expert_offset is None else local_expert_offset
    local = (source_ids >= first_expert) & (source_ids < first_expert + world * experts)
    return combine_w4a16_terms(
        terms, problem["topk_weights"], local, apply_routing_weights_before_fc2
    )


def combine_w4a16_terms(terms, scores, local, routing_weights_applied=False):
    """Ordered FP32 combine shared by both kernels, preserving masked signed zero."""
    output = terms[:, 0].float()
    if not routing_weights_applied:
        output = output * scores[:, 0, None]
    output = torch.where(local[:, 0, None], output, 0.0)
    for slot in range(1, terms.shape[1]):
        if routing_weights_applied:
            updated = output + terms[:, slot].float()
        else:
            # PyTorch 2.12 addcmul(value=1) uses FP32 std::fma on CUDA.
            updated = torch.addcmul(
                output, terms[:, slot].float(), scores[:, slot, None], value=1
            )
        output = torch.where(local[:, slot, None], updated, output)
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
        scales = _unswizzle_mma_scales(
            tensors[f"{name}_weight_sf"], rows, 2 * packed_k, experts
        )
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


def split_w4a16_reference_from_tensors(
    tensors,
    activation_type=None,
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
    """Adapt split test tensors to the shared deterministic W4A16 oracle.

    The activation defaults to SwiGLU. Routing overrides support local expert
    slices and graph tests whose live routing differs from the initial pack.
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
    parameterized = activation == "swiglu" and (
        swiglu_alpha != 1.0
        or swiglu_beta != 0.0
        or swiglu_limit != torch.finfo(torch.float32).max
    )
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
        "gate_up_clamp": swiglu_limit if parameterized else None,
        "swiglu_alpha": swiglu_alpha if parameterized else None,
        "swiglu_beta": swiglu_beta if parameterized else None,
        "situ_beta": situ_beta,
        "situ_linear_beta": situ_linear_beta,
    }
    return w4a16_reference(
        problem,
        canonical_split_w4a16_weights(tensors, activation not in ("relu2", "relu")),
        fc1_alpha=tensors["w1_alpha"],
        fc2_alpha=tensors["w2_alpha"],
        local_expert_offset=local_expert_offset,
    )
