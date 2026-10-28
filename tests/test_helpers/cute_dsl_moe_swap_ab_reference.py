"""Bit-exact, zero-tolerance reference for SM100/SM103 swap-AB CuTe MoE.

Supported production backends and APIs, with ``swap_ab=True`` and
``use_fused_finalize=False``:

* ``flashinfer.cute_dsl_fused_moe`` and ``flashinfer.CuteDslMoEWrapper`` with
  ``quant_mode="w4a4"`` (NVFP4 x NVFP4), ``quant_mode="w4a8"`` (MXFP8 x
  MXFP4), or ``quant_mode="mxfp8"`` (MXFP8 x MXFP8). The W4A4 contract includes
  per-token activation scales and NVFP4 4over6.
  This also covers ``flashinfer.fused_moe.MoELayer`` with ``CuteDslConfig``
  and the corresponding activation/weight quantization pair.
* ``launch_gather_gemm`` followed by ``launch_finalize_gemm`` in
  ``flashinfer.fused_moe.cute_dsl.blackwell.moe_swap_ab`` and the standalone
  BF16 ``moe_unpermute`` finalizer in ``flashinfer.fused_moe.cute_dsl``:
  NVFP4 x NVFP4, MXFP8(E4M3) x MXFP8(E4M3), or MXFP8(E4M3) x MXFP4.

The oracle decodes the actual prepared payload/scale bytes, independently
undoes the up16/gate16 layout, models native accumulator arithmetic with
FP64 products and explicit truncation, applies explicit activation and
intermediate-quantization rounding, and combines stored route outputs with
ordered FP32 multiply/FMA. Existing independent
NVFP4/4over6 and MXFP8 quantization references supply the quantization steps.
The per-tensor NVFP4 epilogue shares standalone standard fast scale arithmetic
and quantizes FP32 activation values directly, without an intermediate BF16
conversion or 4over6 search. The standalone strict-mode switch does not change
this fused quantization contract.
Per-token 4over6 with quantization fast math disabled is TE-identical. The
per-tensor epilogue instead models FlashInfer's supplied global-scale contract:
it does not have TE's original AMAX and is not claimed to be TE-identical.
``assert_cute_dsl_moe_bits`` requires finite outputs with identical storage
bits, including signed zero: ``atol=0`` and ``rtol=0`` on integer views.

The numerical accumulation model supports only finite NV E2M1 x E2M1
with SF16 UE4M3, and MX E4M3 x E4M3 or mixed E2M1/E4M3 in either order
with SF32 UE8M0. It starts from zero and truncates at each K64 NV or K32 MX
update. FP64 holds the required low-precision products and aligned sums exactly;
the final FP32 round-toward-zero conversion gives the carried update rounding.
Unsupported formats, nonfinite operands/scales, negative nonzero NV scales,
and overflowing FP32 updates are rejected. The per-update rules are empirically qualified
against native B200 (SM100) and B300 (SM103) outputs; they are not an ISA
guarantee.

Legacy non-swap-AB kernels, atomic fused finalization, distributed communication,
W4A16, BF16-only MoE, other architectures, and unrelated MoE backends are outside
this contract.
Nonfinite arithmetic is excluded by the finite-output assertion and the
accumulator model's contract.
This test oracle requires Triton; it does not depend on a new PyTorch scaled
matmul API or invoke the production MoE, quantization, or tcgen05 kernels.
"""

import ctypes
import math

import torch
import triton as tr
import triton.language as tl
import triton.language.extra.cuda.libdevice as libdevice

from flashinfer.tllm_enums import (
    ActivationType,
    DEFAULT_SWIGLU_ALPHA,
    DEFAULT_SWIGLU_BETA,
    DEFAULT_SWIGLU_LIMIT,
)
from tests.test_helpers.utils_fp4 import (
    ref_fp4_quant,
    ref_fp4_quant_te,
    pack_fp4_values,
)
from tests.utils_fp8 import mxfp8_quantize_reference


# Numerical accumulation.


@tr.jit
def _decode_operand(pointer, row, k, K: tl.constexpr, FORMAT: tl.constexpr, valid):
    if FORMAT == "e2m1":
        byte = tl.load(pointer + row * (K // 2) + k // 2, valid, 0).to(tl.int32)
        code = (byte >> ((k % 2) * 4)) & 15
        exponent = (code >> 1) & 3
        mantissa = (code & 1) + tl.where(exponent == 0, 0, 2)
        raw_exponent = tl.maximum(exponent, 1) - 1
        value = libdevice.ldexp(mantissa.to(tl.float64), raw_exponent - 1)
        value = tl.where((code & 8) != 0, -value, value)
        invalid = tl.full(value.shape, False, tl.int1)
    elif FORMAT == "e4m3":
        code = tl.load(pointer + row * K + k, valid, 0).to(tl.int32)
        exponent = (code >> 3) & 15
        mantissa = (code & 7) + tl.where(exponent == 0, 0, 8)
        raw_exponent = tl.maximum(exponent, 1) - 7
        value = libdevice.ldexp(mantissa.to(tl.float64), raw_exponent - 3)
        value = tl.where((code & 128) != 0, -value, value)
        invalid = (code & 127) == 127
    else:
        tl.static_assert(False, "Unsupported operand format")
    return value, raw_exponent, invalid


@tr.jit
def _decode_scale(pointer, row, k, K: tl.constexpr, FORMAT: tl.constexpr, valid):
    if FORMAT == "ue4m3":
        code = tl.load(pointer + row * (K // 16) + k // 16, valid, 0).to(tl.int32)
        exponent = (code >> 3) & 15
        mantissa = (code & 7) + tl.where(exponent == 0, 0, 8)
        raw_exponent = tl.maximum(exponent, 1) - 7
        value = libdevice.ldexp(mantissa.to(tl.float64), raw_exponent - 3)
        # Signed zero is numerically nonnegative and is accepted.
        invalid = ((code & 127) == 127) | ((code > 128) & ((code & 128) != 0))
    elif FORMAT == "ue8m0":
        code = tl.load(pointer + row * (K // 32) + k // 32, valid, 127).to(tl.int32)
        raw_exponent = code - 127
        value = libdevice.ldexp(tl.full(code.shape, 1, tl.float64), raw_exponent)
        invalid = code == 255
    else:
        tl.static_assert(False, "Unsupported scale format")
    return value, raw_exponent, invalid


@tr.jit
def _blockscaled_matmul_kernel(
    A,
    B,
    SA,
    SB,
    OUTPUT,
    STATUS,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    FORMAT_A: tl.constexpr,
    FORMAT_B: tl.constexpr,
    SCALE_FORMAT: tl.constexpr,
    OUTPUTS: tl.constexpr,
):
    if SCALE_FORMAT == "ue4m3":
        UPDATE: tl.constexpr = 64
    elif SCALE_FORMAT == "ue8m0":
        UPDATE: tl.constexpr = 32
    else:
        tl.static_assert(False, "Unsupported scale format")

    output = tl.program_id(0) * OUTPUTS + tl.arange(0, OUTPUTS)
    row_a, row_b = output // N, output % N
    valid = output < M * N
    carry = tl.full((OUTPUTS,), 0, tl.float32)
    status = tl.full((OUTPUTS,), 0, tl.int32)
    for start in range(0, K, UPDATE):
        k = start + tl.arange(0, UPDATE)[None, :]
        av, ae, ai = _decode_operand(A, row_a[:, None], k, K, FORMAT_A, valid[:, None])
        bv, be, bi = _decode_operand(B, row_b[:, None], k, K, FORMAT_B, valid[:, None])
        sa, se_a, si_a = _decode_scale(
            SA, row_a[:, None], k, K, SCALE_FORMAT, valid[:, None]
        )
        sb, se_b, si_b = _decode_scale(
            SB, row_b[:, None], k, K, SCALE_FORMAT, valid[:, None]
        )
        status |= (tl.sum((ai | bi | si_a | si_b).to(tl.int32), 1) != 0).to(tl.int32)

        product = ((av * bv) * sa) * sb
        nonzero = product != 0
        scale_exponent = se_a + se_b
        pmax = tl.max(tl.where(nonzero, ae + be + scale_exponent, -1024), 1)
        carry_bits = carry.to(tl.uint32, bitcast=True)
        carry_nonzero = (carry_bits & 0x7FFFFFFF) != 0
        carry_exponent = tl.maximum(((carry_bits >> 23) & 255).to(tl.int32), 1) - 127
        carry_exponent = tl.where(carry_nonzero, carry_exponent, -1024)
        largest_exponent = tl.maximum(pmax, carry_exponent)

        if SCALE_FORMAT == "ue4m3":
            smax = tl.max(tl.where(nonzero, scale_exponent, -1024), 1)
            carry_dominates = carry_nonzero & (carry_exponent >= pmax)
            quantum_exponent = tl.where(
                carry_dominates,
                carry_exponent - 35,
                tl.maximum(largest_exponent - 39, smax - 35),
            )
        elif SCALE_FORMAT == "ue8m0":
            quantum_exponent = largest_exponent - 25
        else:
            tl.static_assert(False, "Unsupported scale format")
        # An entirely zero update needs no alignment; keep ldexp in range.
        quantum_exponent = tl.where(largest_exponent == -1024, 0, quantum_exponent)

        if SCALE_FORMAT == "ue4m3":
            sum4 = tl.sum(tl.reshape(product, (OUTPUTS, 16, 4)), 2)
            sum16 = tl.sum(tl.reshape(product, (OUTPUTS, 4, 16)), 2)
            units4 = tl.sum(
                libdevice.trunc(libdevice.ldexp(sum4, -quantum_exponent[:, None])), 1
            )
            units16 = tl.sum(
                libdevice.trunc(libdevice.ldexp(sum16, -quantum_exponent[:, None])), 1
            )
            product_units = tl.where(carry_dominates, units16, units4)
        elif SCALE_FORMAT == "ue8m0":
            product_units = tl.sum(
                libdevice.trunc(libdevice.ldexp(product, -quantum_exponent[:, None])), 1
            )
        else:
            tl.static_assert(False, "Unsupported scale format")
        carry_units = libdevice.trunc(
            libdevice.ldexp(carry.to(tl.float64), -quantum_exponent)
        )
        updated = libdevice.ldexp(product_units + carry_units, quantum_exponent)
        # Overflow is outside this finite-only model, even when RZ saturates.
        status |= (tl.abs(updated) > 3.4028234663852886e38).to(tl.int32) * 2
        carry = tl.inline_asm_elementwise(
            "cvt.rz.f32.f64 $0, $1;",
            constraints="=f,d",
            args=[updated],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
        carry = tl.where(
            (carry.to(tl.uint32, bitcast=True) & 0x7FFFFFFF) == 0, 0.0, carry
        )
    tl.store(OUTPUT + output, carry, valid)
    tl.atomic_or(STATUS, tl.max(tl.where(valid, status, 0), 0), sem="relaxed")


def blockscaled_matmul(a, b, scale_a, scale_b, format_a, format_b, scale_format):
    """Return FP32 ``a[M, K] @ b[N, K].T`` with zero initial accumulator.

    E2M1 is uint8 packed low nibble first; E4M3 is raw uint8 or
    torch.float8_e4m3fn. Scales have logical shape [rows, K / block_size],
    with block_size 16 for UE4M3 and 32 for UE8M0. They may be uint8 bytes
    or their matching torch float8 dtype. Noncontiguous inputs are copied.

    Only finite values and finite FP32 update results are supported. Invalid
    inputs or overflow raise ValueError after one device-status read. This
    host-orchestrated test helper is not suitable for CUDA graph capture.
    """
    if scale_format == "ue4m3":
        if (format_a, format_b) != ("e2m1", "e2m1"):
            raise ValueError("UE4M3 supports only NV E2M1 x E2M1")
        block_size, update = 16, 64
        scale_dtype = torch.float8_e4m3fn
    elif scale_format == "ue8m0":
        if (format_a, format_b) not in (
            ("e4m3", "e4m3"),
            ("e2m1", "e4m3"),
            ("e4m3", "e2m1"),
        ):
            raise ValueError("UE8M0 supports only E4M3 x E4M3 or mixed E2M1/E4M3")
        block_size, update = 32, 32
        scale_dtype = getattr(torch, "float8_e8m0fnu", None)
    else:
        raise ValueError(f"Unsupported scale format: {scale_format!r}")

    tensors = (a, b, scale_a, scale_b)
    if any(t.ndim != 2 for t in tensors):
        raise ValueError("Operands and logical scales must all be two-dimensional")
    if a.device.type != "cuda" or any(t.device != a.device for t in tensors):
        raise ValueError("Operands and scales must share one CUDA device")
    for tensor, fmt in ((a, format_a), (b, format_b)):
        allowed = (
            (torch.uint8,) if fmt == "e2m1" else (torch.uint8, torch.float8_e4m3fn)
        )
        if tensor.dtype not in allowed:
            raise TypeError(f"Unsupported {fmt} storage dtype: {tensor.dtype}")
    if any(t.dtype not in (torch.uint8, scale_dtype) for t in (scale_a, scale_b)):
        raise TypeError(f"Scales must use uint8 or matching {scale_format} storage")
    m, n = a.shape[0], b.shape[0]
    k = a.shape[1] * (2 if format_a == "e2m1" else 1)
    kb = b.shape[1] * (2 if format_b == "e2m1" else 1)
    if m == 0 or n == 0 or k == 0 or k != kb or k % update:
        raise ValueError(
            f"Nonempty operands need equal logical K divisible by {update}"
        )
    if scale_a.shape != (m, k // block_size) or scale_b.shape != (n, k // block_size):
        raise ValueError(f"Logical scales must have shape [rows, K / {block_size}]")

    raw = tuple(t.contiguous().view(torch.uint8) for t in tensors)
    result = torch.empty((m, n), dtype=torch.float32, device=a.device)
    status = torch.zeros((), dtype=torch.int32, device=a.device)
    _blockscaled_matmul_kernel[(tr.cdiv(m * n, 8),)](
        *raw,
        result,
        status,
        m,
        n,
        k,
        format_a,
        format_b,
        scale_format,
        8,
        num_warps=4,
        enable_fp_fusion=False,
    )
    error = status.item()
    if error:
        reason = (
            "nonfinite operands/scales or negative UE4M3 scales"
            if error & 1
            else "FP32 update overflow"
        )
        raise ValueError(f"Outside finite block-scaled reference contract: {reason}")
    return result


# FC1 scale and activation.


def _f32(value):
    return ctypes.c_float(float(value)).value


def _row_scale(scale, rows, device):
    if scale.dtype != torch.float32 or scale.device != device:
        raise ValueError("row scales must be FP32 tensors on the accumulator device")
    if scale.numel() not in (1, rows):
        raise ValueError("row scales must contain one value or one value per row")
    return scale.contiguous().reshape(-1)


@tr.jit
def _mul_rn_f32(a, b):
    return tl.inline_asm_elementwise(
        "mul.rn.f32 $0, $1, $2;", "=f,f,f", [a, b], tl.float32, True, 1
    )


@tr.jit
def _sigmoid_approx_f32(x):
    return tl.inline_asm_elementwise(
        """{
            .reg .f32 exponent, denominator;
            mul.rn.f32 exponent, $1, 0fBFB8AA3B;
            ex2.approx.ftz.f32 exponent, exponent;
            add.rn.f32 denominator, exponent, 0f3F800000;
            rcp.approx.ftz.f32 $0, denominator;
        }""",
        "=f,f",
        [x],
        tl.float32,
        True,
        1,
    )


@tr.jit
def _tanh_approx_f32(x):
    return tl.inline_asm_elementwise(
        """{
            .reg .f32 exponent, inverse;
            mul.rn.f32 exponent, $1, 0fC0000000;
            mul.rn.f32 exponent, exponent, 0f3FB8AA3B;
            ex2.approx.ftz.f32 exponent, exponent;
            add.rn.f32 exponent, exponent, 0f3F800000;
            rcp.approx.ftz.f32 inverse, exponent;
            fma.rn.f32 $0, inverse, 0f40000000, 0fBF800000;
        }""",
        "=f,f",
        [x],
        tl.float32,
        True,
        1,
    )


@tr.jit
def _activation_kernel(
    ACC,
    ALPHA,
    ROW_SCALE,
    OUT,
    N: tl.constexpr,
    WIDTH: tl.constexpr,
    ALPHA_SCALAR: tl.constexpr,
    ROW_SCALE_SCALAR: tl.constexpr,
    ACTIVATION: tl.constexpr,
    NEG_ALPHA_LOG2E: tl.constexpr,
    BIAS: tl.constexpr,
    LIMIT: tl.constexpr,
    SITU_BETA: tl.constexpr,
    INV_SITU_BETA: tl.constexpr,
    LINEAR_BETA: tl.constexpr,
    INV_LINEAR_BETA: tl.constexpr,
):
    index = tl.program_id(0) * 256 + tl.arange(0, 256)
    row, column = index // WIDTH, index % WIDTH
    if ALPHA_SCALAR:
        alpha = tl.load(ALPHA)
    else:
        alpha = tl.load(ALPHA + row, index < N, other=0)
    if ROW_SCALE is not None:
        if ROW_SCALE_SCALAR:
            row_scale = tl.load(ROW_SCALE)
        else:
            row_scale = tl.load(ROW_SCALE + row, index < N, other=0)
        alpha = _mul_rn_f32(alpha, row_scale)
    if ACTIVATION == "relu2":
        up = _mul_rn_f32(tl.load(ACC + index, index < N, other=0), alpha)
        result = tl.inline_asm_elementwise(
            "{ .reg .f32 r; max.f32 r, $1, 0f00000000; mul.rn.f32 $0, r, r; }",
            "=f,f",
            [up],
            tl.float32,
            True,
            1,
        )
    else:
        offset = row * (2 * WIDTH) + column
        up = _mul_rn_f32(tl.load(ACC + offset, index < N, other=0), alpha)
        gate = _mul_rn_f32(tl.load(ACC + offset + WIDTH, index < N, other=0), alpha)
        if ACTIVATION == "swiglu":
            result = tl.inline_asm_elementwise(
                """{
                    .reg .f32 u, g, exponent, inverse;
                    min.NaN.f32 g, $2, $5;
                    min.NaN.f32 u, $1, $5;
                    neg.f32 u, u;
                    min.NaN.f32 u, u, $5;
                    neg.f32 u, u;
                    mul.rn.f32 exponent, g, $3;
                    ex2.approx.ftz.f32 exponent, exponent;
                    add.rn.f32 exponent, exponent, 0f3F800000;
                    rcp.approx.ftz.f32 inverse, exponent;
                    mul.rn.f32 g, g, inverse;
                    add.rn.f32 u, u, $4;
                    mul.rn.f32 $0, u, g;
                }""",
                "=f,f,f,f,f,f",
                [
                    up,
                    gate,
                    tl.full((), NEG_ALPHA_LOG2E, tl.float32),
                    tl.full((), BIAS, tl.float32),
                    tl.full((), LIMIT, tl.float32),
                ],
                tl.float32,
                True,
                1,
            )
        elif ACTIVATION == "geglu_tanh":
            # The shared CuTe GELU helper contracts its cubic sum and
            # 2*sigmoid-1 expression into FP32 FMAs.
            result = tl.inline_asm_elementwise(
                """{
                    .reg .f32 cubic, inner, exponent, inverse, bounded, half;
                    mul.rn.f32 cubic, $2, $3;
                    mul.rn.f32 cubic, cubic, $2;
                    fma.rn.f32 inner, $2, cubic, $2;
                    mul.rn.f32 inner, inner, $4;
                    mul.rn.f32 exponent, inner, 0fC0000000;
                    mul.rn.f32 exponent, exponent, 0f3FB8AA3B;
                    ex2.approx.ftz.f32 exponent, exponent;
                    add.rn.f32 exponent, exponent, 0f3F800000;
                    rcp.approx.ftz.f32 inverse, exponent;
                    fma.rn.f32 bounded, inverse, 0f40000000, 0fBF800000;
                    mul.rn.f32 half, $2, 0f3F000000;
                    add.rn.f32 bounded, bounded, 0f3F800000;
                    mul.rn.f32 half, half, bounded;
                    mul.rn.f32 $0, $1, half;
                }""",
                "=f,f,f,f,f",
                [
                    up,
                    gate,
                    tl.full((), 0.044715, tl.float32),
                    tl.full((), 0.7978845608028654, tl.float32),
                ],
                tl.float32,
                True,
                1,
            )
        elif ACTIVATION == "situ":
            if LINEAR_BETA is not None:
                up = _mul_rn_f32(
                    tl.full((), LINEAR_BETA, tl.float32),
                    _tanh_approx_f32(
                        _mul_rn_f32(up, tl.full((), INV_LINEAR_BETA, tl.float32))
                    ),
                )
            bounded_gate = _mul_rn_f32(
                tl.full((), SITU_BETA, tl.float32),
                _tanh_approx_f32(
                    _mul_rn_f32(gate, tl.full((), INV_SITU_BETA, tl.float32))
                ),
            )
            result = _mul_rn_f32(
                up, _mul_rn_f32(bounded_gate, _sigmoid_approx_f32(gate))
            )
        else:
            tl.static_assert(False, "Unsupported split MoE activation")
    tl.store(OUT + index, result, index < N)


def activate_fc1(
    accumulators,
    alpha,
    per_token_scale=None,
    activation_type=ActivationType.Swiglu,
    swiglu_alpha=None,
    swiglu_beta=None,
    swiglu_limit=None,
    situ_beta=None,
    situ_linear_beta=None,
):
    """Apply the FC1 FP32 scale and activation before intermediate quantization."""
    if accumulators.ndim != 2 or accumulators.dtype != torch.float32:
        raise ValueError("FC1 accumulators must be an FP32 matrix")
    if not accumulators.is_cuda:
        raise ValueError("split MoE arithmetic references require CUDA")
    if activation_type == ActivationType.Relu2:
        kind, width = "relu2", accumulators.shape[1]
    elif activation_type in (ActivationType.Swiglu, ActivationType.GegluTanh):
        if accumulators.shape[1] % 2:
            raise ValueError("gated FC1 accumulators require equal up/gate halves")
        width = accumulators.shape[1] // 2
        kind = "swiglu" if activation_type == ActivationType.Swiglu else "geglu_tanh"
    else:
        raise ValueError(f"Unsupported split MoE activation: {activation_type}")
    if situ_beta is not None:
        if activation_type != ActivationType.Swiglu:
            raise ValueError("SiTU requires ActivationType.Swiglu")
        if not math.isfinite(situ_beta) or situ_beta <= 0:
            raise ValueError("situ_beta must be finite and positive")
        kind = "situ"
    if situ_linear_beta is not None:
        if situ_beta is None:
            raise ValueError("situ_linear_beta requires situ_beta")
        if not math.isfinite(situ_linear_beta) or situ_linear_beta <= 0:
            raise ValueError("situ_linear_beta must be finite and positive")
    rows = accumulators.shape[0]
    alpha = _row_scale(alpha, rows, accumulators.device)
    if per_token_scale is not None:
        per_token_scale = _row_scale(per_token_scale, rows, accumulators.device)
    output = torch.empty((rows, width), dtype=torch.float32, device=accumulators.device)
    if not output.numel():
        return output
    sigmoid_alpha = DEFAULT_SWIGLU_ALPHA if swiglu_alpha is None else swiglu_alpha
    # Round both operands before the compile-time product, just as Float32 does.
    neg_alpha_log2e = -_f32(_f32(sigmoid_alpha) * _f32(1.4426950408889634))
    _activation_kernel[((output.numel() + 255) // 256,)](
        accumulators.contiguous(),
        alpha,
        per_token_scale,
        output,
        output.numel(),
        width,
        alpha.numel() == 1,
        per_token_scale is not None and per_token_scale.numel() == 1,
        kind,
        neg_alpha_log2e,
        DEFAULT_SWIGLU_BETA if swiglu_beta is None else swiglu_beta,
        DEFAULT_SWIGLU_LIMIT if swiglu_limit is None else swiglu_limit,
        _f32(situ_beta) if situ_beta is not None else None,
        _f32(1.0 / _f32(situ_beta)) if situ_beta is not None else None,
        _f32(situ_linear_beta) if situ_linear_beta is not None else None,
        _f32(1.0 / _f32(situ_linear_beta)) if situ_linear_beta is not None else None,
        enable_fp_fusion=False,
    )
    return output


# FC2 scale and storage.


@tr.jit
def _fc2_scale_kernel(
    ACC,
    ALPHA,
    ROW_SCALE,
    OUT,
    N: tl.constexpr,
    WIDTH: tl.constexpr,
    ALPHA_SCALAR: tl.constexpr,
    ROW_SCALE_SCALAR: tl.constexpr,
    SCALE_DTYPE: tl.constexpr,
):
    index = tl.program_id(0) * 256 + tl.arange(0, 256)
    row = index // WIDTH
    if ALPHA_SCALAR:
        scale = tl.load(ALPHA)
    else:
        scale = tl.load(ALPHA + row, index < N, other=0)
    if ROW_SCALE is not None:
        if ROW_SCALE_SCALAR:
            row_scale = tl.load(ROW_SCALE)
        else:
            row_scale = tl.load(ROW_SCALE + row, index < N, other=0)
        scale = tl.inline_asm_elementwise(
            "mul.rn.f32 $0, $1, $2;",
            "=f,f,f",
            [scale, row_scale],
            tl.float32,
            True,
            1,
        )
    scale = scale.to(SCALE_DTYPE, fp_downcast_rounding="rtne").to(tl.float32)
    value = tl.load(ACC + index, index < N, other=0)
    value = tl.inline_asm_elementwise(
        "mul.rn.f32 $0, $1, $2;",
        "=f,f,f",
        [value, scale],
        tl.float32,
        True,
        1,
    )
    tl.store(
        OUT + index,
        value.to(OUT.dtype.element_ty, fp_downcast_rounding="rtne"),
        index < N,
    )


def scale_fc2(
    accumulators,
    alpha,
    final_scale_dtype=torch.float32,
    per_token_scale=None,
    output_dtype=torch.bfloat16,
):
    """Round the FC2 row scale to the routing dtype, then scale/store each row."""
    if accumulators.ndim != 2 or accumulators.dtype != torch.float32:
        raise ValueError("FC2 accumulators must be an FP32 matrix")
    if not accumulators.is_cuda:
        raise ValueError("split MoE arithmetic references require CUDA")
    if final_scale_dtype == torch.float32:
        scale_dtype = tl.float32
    elif final_scale_dtype == torch.bfloat16:
        scale_dtype = tl.bfloat16
    elif final_scale_dtype == torch.float16:
        scale_dtype = tl.float16
    else:
        raise ValueError("routing scales must be FP32, BF16, or FP16")
    if output_dtype not in (torch.float32, torch.bfloat16, torch.float16):
        raise ValueError("FC2 output must be FP32, BF16, or FP16")
    rows, width = accumulators.shape
    alpha = _row_scale(alpha, rows, accumulators.device)
    if per_token_scale is not None:
        per_token_scale = _row_scale(per_token_scale, rows, accumulators.device)
    output = torch.empty_like(accumulators, dtype=output_dtype)
    if not output.numel():
        return output
    _fc2_scale_kernel[((output.numel() + 255) // 256,)](
        accumulators.contiguous(),
        alpha,
        per_token_scale,
        output,
        output.numel(),
        width,
        alpha.numel() == 1,
        per_token_scale is not None and per_token_scale.numel() == 1,
        scale_dtype,
        enable_fp_fusion=False,
    )
    return output


# Ordered route combine.


@tr.jit
def _combine_kernel(
    ROWS, WEIGHTS, VALID, OUT, N: tl.constexpr, H: tl.constexpr, K: tl.constexpr
):
    index = tl.program_id(0) * 256 + tl.arange(0, 256)
    token, channel = index // H, index % H
    result = tl.full((256,), 0.0, tl.float32)
    for slot in tl.static_range(K):
        route = token * K + slot
        valid = tl.load(VALID + route, index < N, other=False)
        # Invalid routes never read their potentially uninitialized row.
        value = tl.load(ROWS + route * H + channel, (index < N) & valid, other=0).to(
            tl.float32
        )
        weight = tl.load(WEIGHTS + route, (index < N) & valid, other=0).to(tl.float32)
        if slot == 0:
            term = tl.inline_asm_elementwise(
                "mul.rn.f32 $0, $1, $2;",
                "=f,f,f",
                [value, weight],
                tl.float32,
                True,
                1,
            )
        else:
            term = tl.inline_asm_elementwise(
                "fma.rn.f32 $0, $1, $2, $3;",
                "=f,f,f,f",
                [value, weight, result],
                tl.float32,
                True,
                1,
            )
        result = tl.where(valid, term, result)
    tl.store(
        OUT + index,
        result.to(OUT.dtype.element_ty, fp_downcast_rounding="rtne"),
        index < N,
    )


def combine_routes(route_outputs, route_weights, valid_routes):
    """Combine [token, slot, channel] rows, returning the input storage dtype.

    The standalone SM100 production finalizer currently accepts BF16 only.
    FP16 here models the same ordered arithmetic without extending that API.
    """
    if route_outputs.ndim != 3 or route_outputs.dtype not in (
        torch.bfloat16,
        torch.float16,
    ):
        raise ValueError("deterministic combine requires BF16 or FP16 route outputs")
    if route_weights.dtype not in (torch.float32, torch.bfloat16, torch.float16):
        raise ValueError("routing weights must be FP32, BF16, or FP16")
    if valid_routes.dtype != torch.bool:
        raise ValueError("valid_routes must be a boolean route mask")
    if (
        route_weights.shape != route_outputs.shape[:2]
        or valid_routes.shape != route_weights.shape
    ):
        raise ValueError(
            "routing weights and validity must match the token/slot dimensions"
        )
    if not route_outputs.is_cuda or any(
        tensor.device != route_outputs.device
        for tensor in (route_weights, valid_routes)
    ):
        raise ValueError("combine tensors must share one CUDA device")
    tokens, top_k, hidden = route_outputs.shape
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    output = torch.empty(
        (tokens, hidden), dtype=route_outputs.dtype, device=route_outputs.device
    )
    if not output.numel():
        return output
    _combine_kernel[((output.numel() + 255) // 256,)](
        route_outputs.contiguous(),
        route_weights.contiguous(),
        valid_routes.contiguous(),
        output,
        output.numel(),
        hidden,
        top_k,
        enable_fp_fusion=False,
    )
    return output


# End-to-end reference and exact assertion.


def _linear_scales(scales, rows, k, experts, block):
    shape = (32, 4, tr.cdiv(rows, 128), 4, tr.cdiv(k, block * 4), experts)
    if tuple(scales.shape) != shape:
        raise ValueError(f"Expected prepared scale shape {shape}, got {scales.shape}")
    return (
        scales.permute(5, 2, 1, 0, 4, 3)
        .contiguous()
        .view(torch.uint8)
        .reshape(experts, shape[2] * 128, shape[4] * 4)
    )[:, :rows, : k // block].contiguous()


@torch.no_grad()
def cute_dsl_moe_reference(
    x,
    x_sf,
    token_selected_experts,
    token_final_scales,
    w1_weight,
    w1_weight_sf,
    w1_alpha,
    fc2_input_scale,
    w2_weight,
    w2_weight_sf,
    w2_alpha,
    per_token_scale=None,
    local_expert_offset=0,
    output_dtype=torch.bfloat16,
    activation_type=ActivationType.Swiglu,
    swiglu_alpha=None,
    swiglu_beta=None,
    swiglu_limit=None,
    situ_beta=None,
    situ_linear_beta=None,
    nvfp4_4over6_config=None,
    disable_fp4_quant_fast_math=False,
    a_dtype="float4_e2m1fn",
    b_dtype="float4_e2m1fn",
    sf_vec_size=16,
    swap_ab=True,
):
    """Evaluate unfused swap-AB MoE from the exact prepared kernel operands.

    Routing IDs may name experts outside the local contiguous expert window;
    those slots contribute nothing. Gated weights use up16/gate16 interleave.
    Input scales are linear; weight scales use the six-dimensional MMA view.
    """
    if swap_ab is not True:
        raise ValueError("The bit-exact block-scaled reference requires swap_ab=True")
    formats = {
        ("float4_e2m1fn", "float4_e2m1fn", 16),
        ("float8_e4m3fn", "float8_e4m3fn", 32),
        ("float8_e4m3fn", "float4_e2m1fn", 32),
    }
    if (a_dtype, b_dtype, sf_vec_size) not in formats:
        raise ValueError("Unsupported block-scaled MoE operand combination")
    if output_dtype != torch.bfloat16:
        raise ValueError("The standalone SM100 MoE finalizer requires BF16 output")
    if per_token_scale is not None and sf_vec_size != 16:
        raise ValueError("Per-token scaling is an NVFP4 contract")
    if activation_type == ActivationType.Relu2:
        gated = False
    elif activation_type in (ActivationType.Swiglu, ActivationType.GegluTanh):
        gated = True
    else:
        raise ValueError(f"Unsupported activation {activation_type}")
    tokens, topk = token_selected_experts.shape
    experts, rows, _ = w1_weight.shape
    hidden = w2_weight.shape[1]
    intermediate = rows // 2 if gated else rows
    sx = x_sf.view(torch.uint8).reshape(tokens, hidden // sf_vec_size)
    activation_format = "e2m1" if a_dtype == "float4_e2m1fn" else "e4m3"
    weight_format = "e2m1" if b_dtype == "float4_e2m1fn" else "e4m3"
    scale_format = "ue4m3" if sf_vec_size == 16 else "ue8m0"
    s1 = _linear_scales(w1_weight_sf, rows, hidden, experts, sf_vec_size)
    s2 = _linear_scales(w2_weight_sf, hidden, intermediate, experts, sf_vec_size)
    w1 = w1_weight.view(torch.uint8)
    if gated:
        # Rows are [up16, gate16, ...]; math helper consumes [all_up, all_gate].
        def uninterleave(value):
            width = value.shape[-1]
            return (
                value.reshape(experts, rows // 32, 2, 16, width)
                .transpose(1, 2)
                .reshape(experts, rows, width)
                .contiguous()
            )

        w1, s1 = uninterleave(w1), uninterleave(s1)
    routes = torch.zeros((tokens, topk, hidden), device=x.device, dtype=output_dtype)
    local = (token_selected_experts >= local_expert_offset) & (
        token_selected_experts < local_expert_offset + experts
    )
    for expert in range(experts):
        ids = (token_selected_experts == expert + local_expert_offset).nonzero()
        if ids.numel() == 0:
            continue
        token_ids, slots = ids[:, 0], ids[:, 1]
        input_scale = None if per_token_scale is None else per_token_scale[token_ids]
        acc = blockscaled_matmul(
            w1[expert],
            x.view(torch.uint8)[token_ids],
            s1[expert],
            sx[token_ids],
            weight_format,
            activation_format,
            scale_format,
        ).T.contiguous()
        activated = activate_fc1(
            acc,
            w1_alpha[expert],
            input_scale,
            activation_type,
            swiglu_alpha,
            swiglu_beta,
            swiglu_limit,
            situ_beta,
            situ_linear_beta,
        )
        if sf_vec_size == 16:
            if per_token_scale is None:
                q, sf = ref_fp4_quant(activated, fc2_input_scale, 16)
                sf = sf.to(torch.float8_e4m3fn)
                route_scale = None
            else:
                stored = activated.to(output_dtype)
                row_amax = stored.float().abs().amax(dim=-1)
                q, sf, route_scale, _ = ref_fp4_quant_te(
                    stored,
                    row_amax,
                    per_token_rowwise=True,
                    nvfp4_4over6_config=nvfp4_4over6_config,
                    disable_fp4_quant_fast_math=disable_fp4_quant_fast_math,
                    global_scale_inv=fc2_input_scale,
                )
            fc2 = pack_fp4_values(q)
            sf = sf.view(torch.uint8)
        elif sf_vec_size == 32:
            q, sf = mxfp8_quantize_reference(
                activated,
                is_sf_swizzled_layout=False,
            )
            fc2 = q.view(torch.uint8)
            sf = sf.reshape(activated.shape[0], -1)
            route_scale = None
        else:
            raise ValueError(f"Unsupported scale block {sf_vec_size}")
        acc2 = blockscaled_matmul(
            w2_weight[expert],
            fc2,
            s2[expert],
            sf,
            weight_format,
            activation_format,
            scale_format,
        ).T.contiguous()
        routes[token_ids, slots] = scale_fc2(
            acc2,
            w2_alpha[expert],
            token_final_scales.dtype,
            route_scale,
            output_dtype,
        )
    return combine_routes(routes, token_final_scales, local)


def assert_cute_dsl_moe_bits(actual, expected):
    """Require finite BF16/FP16 outputs and exact storage-bit agreement."""
    assert actual.dtype == expected.dtype
    assert actual.dtype in (torch.bfloat16, torch.float16)
    assert actual.shape == expected.shape
    assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
    torch.testing.assert_close(
        actual.view(torch.int16), expected.view(torch.int16), atol=0, rtol=0
    )
