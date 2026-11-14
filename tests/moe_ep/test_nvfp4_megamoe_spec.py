# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Device conformance of NVFP4 MegaMoE's ordered scalar arithmetic.

Expected RN arithmetic and conversions are evaluated independently on CPU.
Only opaque exponential/reciprocal instructions are sampled on the device.
The oracle does not call the production activation or quantization helpers.
"""

from __future__ import annotations

import math

import cutlass
import cutlass.cute as cute
import pytest

from flashinfer.batch_invariant.fp_interface import exact
from flashinfer.moe_ep.batch_invariant.nvfp4_device import PackedF32, ScalarF32
from flashinfer.moe_ep.batch_invariant.nvfp4_spec import (
    fc1_activation,
    fc2_output,
    input_quantize_block_precast,
    quantize_block_precast,
)

from .test_bf16_megamoe_batch_invariant import _require_cuda
from .test_bf16_megamoe_spec import _assert_numeric_bits, _launch, _sample

pytestmark = pytest.mark.gpu


class _Activation:
    def __init__(self, packed, scaled, weighted, clamp):
        self.packed, self.scaled, self.weighted, self.clamp = (
            packed,
            scaled,
            weighted,
            clamp,
        )

    @cute.jit
    def __call__(
        self,
        g: cute.Tensor,
        u: cute.Tensor,
        a: cute.Tensor,
        p: cute.Tensor,
        y: cute.Tensor,
        stream,
    ):
        self.kernel(g, u, a, p, y).launch(
            grid=(cute.ceil_div(g.shape[0], 128), 1, 1),
            block=(128, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        g: cute.Tensor,
        u: cute.Tensor,
        a: cute.Tensor,
        p: cute.Tensor,
        y: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        i = bidx * 128 + tidx
        if cutlass.const_expr(self.packed):
            i = i * 2
            if i + 1 < g.shape[0]:
                alpha = None
                weight = None
                if cutlass.const_expr(self.scaled):
                    alpha = (a[i], a[i + 1])
                if cutlass.const_expr(self.weighted):
                    weight = (p[i], p[i + 1])
                y[i], y[i + 1] = fc1_activation(
                    PackedF32,
                    (g[i], g[i + 1]),
                    (u[i], u[i + 1]),
                    alpha=alpha,
                    weight=weight,
                    clamp=self.clamp,
                )
        elif i < g.shape[0]:
            alpha = None
            weight = None
            if cutlass.const_expr(self.scaled):
                alpha = a[i]
            if cutlass.const_expr(self.weighted):
                weight = p[i]
            y[i] = fc1_activation(
                ScalarF32, g[i], u[i], alpha=alpha, weight=weight, clamp=self.clamp
            )


def _minimum(a, b):
    # PTX min.f32: number wins over a single NaN; opposite signed zeros give -0.
    if math.isnan(a):
        return b
    if math.isnan(b):
        return a
    if a == b == 0:
        return -0.0 if math.copysign(1, a) < 0 or math.copysign(1, b) < 0 else 0.0
    return min(a, b)


def _activation_oracle(gates, ups, alphas, weights, clamp):
    rows = []
    for i, (gate, up) in enumerate(zip(gates, ups, strict=True)):
        if alphas is not None:
            gate, up = exact.mul(gate, alphas[i]), exact.mul(up, alphas[i])
        if clamp is not None:
            limit = exact.to_f32(abs(clamp))
            gate = _minimum(gate, limit)
            up = math.copysign(_minimum(abs(up), limit), up)
        rows.append((gate, up))
    exponents = [exact.mul(g, exact.to_f32(-1.4426950408889634)) for g, _ in rows]
    exponentials = _sample("ex2", exponents)
    reciprocals = _sample("rcp", [exact.add(v, 1) for v in exponentials])
    out = []
    for i, ((gate, up), sigmoid) in enumerate(zip(rows, reciprocals, strict=True)):
        value = exact.mul(exact.mul(up, gate), sigmoid)
        out.append(value if weights is None else exact.mul(value, weights[i]))
    return out


@pytest.mark.parametrize("clamp", (None, 1.25, -0.0))
@pytest.mark.parametrize("scaled,weighted", ((False, False), (True, True)))
def test_nvfp4_fc1_scalar_and_packed_match_exact_chain(clamp, scaled, weighted):
    _require_cuda()
    import torch

    generator = torch.Generator().manual_seed(4717)
    edges = [
        0.0,
        -0.0,
        2.0**-149,
        -(2.0**-149),
        2.0**-126,
        -(2.0**-126),
        1.25,
        -1.25,
        1.2500001192092896,
        1.2499998807907104,
        80.0,
        -80.0,
    ]
    gates = edges + torch.randn(52, generator=generator).mul_(3).tolist()
    g = torch.tensor(gates, dtype=torch.float32, device="cuda")
    u = torch.randn(64, generator=generator).to("cuda")
    a = torch.linspace(0.25, 1.5, 64, device="cuda")
    p = torch.randn(64, generator=generator).to("cuda")
    u[:4] = torch.tensor([0.0, -0.0, 1.0, -1.0], device="cuda")
    p[:4] = torch.tensor([0.0, -0.0, -1.0, 0.75], device="cuda")
    expected = _activation_oracle(
        g.cpu().tolist(),
        u.cpu().tolist(),
        a.cpu().tolist() if scaled else None,
        p.cpu().tolist() if weighted else None,
        clamp,
    )
    outputs = []
    for packed in (False, True):
        y = torch.empty_like(g)
        _launch(_Activation(packed, scaled, weighted, clamp), g, u, a, p, y)
        _assert_numeric_bits(y, expected)
        outputs.append(y)
    assert torch.equal(outputs[0].view(torch.int32), outputs[1].view(torch.int32))


class _Quantize:
    def __init__(self, input_stage, norm):
        self.input_stage, self.norm = input_stage, norm

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        scaled: cute.Tensor,
        q: cute.Tensor,
        sf: cute.Tensor,
        stream,
    ):
        self.kernel(x, scaled, q, sf).launch(
            grid=(cute.ceil_div(x.shape[0], 128), 1, 1),
            block=(128, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self, x: cute.Tensor, scaled: cute.Tensor, q: cute.Tensor, sf: cute.Tensor
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        row = bidx * 128 + tidx
        if row < x.shape[0]:
            values = tuple(x[row, i] for i in range(16))
            norm = ScalarF32.const(self.norm)
            if cutlass.const_expr(self.input_stage):
                precast, scale = input_quantize_block_precast(
                    ScalarF32, values, norm=norm
                )
            else:
                precast, scale = quantize_block_precast(ScalarF32, values, norm=norm)
            registers = cute.make_rmem_tensor((16,), cutlass.Float32)
            for i in cutlass.range_constexpr(16):
                registers[i] = precast[i]
                scaled[row, i] = precast[i]
            sf[row] = scale
            fp4 = cute.make_rmem_tensor((16,), cutlass.Float4E2M1FN)
            fp4.store(registers.load().to(cutlass.Float4E2M1FN))
            packed = cute.recast_tensor(fp4, cutlass.Uint8)
            for i in cutlass.range_constexpr(8):
                q[row, i] = packed[i]


def _quantization_oracle(blocks, norm):
    norm = exact.to_f32(norm)
    scales = []
    for values in blocks:
        # Test inputs are finite: both production amax orders select this same
        # nonnegative maximum. Their distinct primitive traces are CPU/Lean pins.
        maximum = max(abs(v) for v in values)
        candidate = exact.mul(exact.mul(maximum, exact.to_f32(1 / 6)), norm)
        scales.append(exact.to_fp8(candidate, "e4m3"))
    reciprocals = _sample("rcp", scales)
    scaled, codes = [], []
    for values, scale, reciprocal in zip(blocks, scales, reciprocals, strict=True):
        reciprocal = _minimum(exact.mul(norm, reciprocal), exact.F32_MAX)
        mask = _minimum(exact.mul(scale, exact.to_f32(1e30)), 1.0)
        encoding = exact.mul(reciprocal, mask)
        row = [exact.mul(v, encoding) for v in values]
        nibbles = [exact.e2m1_code(exact.to_e2m1(v)) for v in row]
        scaled.append(row)
        codes.append([nibbles[i] | nibbles[i + 1] << 4 for i in range(0, 16, 2)])
    return scaled, codes, scales


@pytest.mark.parametrize("input_stage", (False, True))
@pytest.mark.parametrize("norm", (1.0, 0.25))
def test_nvfp4_block16_quantization_matches_independent_exact_reference(
    input_stage, norm
):
    _require_cuda()
    import torch

    generator = torch.Generator().manual_seed(1717)
    rows = torch.randn((16, 16), generator=generator).mul_(3)
    # All-zero / signed-zero, denormals, E4M3 scale underflow and scale ties.
    rows[0].zero_()
    rows[0, 1::2] = -0.0
    rows[1] = torch.tensor([(-1.0) ** i * 2.0**-149 for i in range(16)])
    rows[2] = torch.linspace(-(2.0**-126), 2.0**-126, 16)
    rows[3] = torch.linspace(-(2.0**-10), 2.0**-10, 16)
    rows[4] = torch.linspace(-6.0, 6.0, 16)
    rows[5] = torch.tensor(
        [
            0.25,
            0.75,
            1.25,
            1.75,
            2.5,
            3.5,
            5.0,
            6.0,
            -0.25,
            -0.75,
            -1.25,
            -1.75,
            -2.5,
            -3.5,
            -5.0,
            -6.0,
        ]
    )
    rows[6] = torch.linspace(-6 * 2.0**-10, 6 * 2.0**-10, 16)
    if input_stage:
        # The input staging boundary widens BF16 values before the recipe.
        rows = rows.to(torch.bfloat16).to(torch.float32)
    expected_scaled, expected_q, expected_sf = _quantization_oracle(rows.tolist(), norm)
    x = rows.to("cuda")
    scaled = torch.empty_like(x)
    q = torch.empty((16, 8), dtype=torch.uint8, device="cuda")
    sf = torch.empty((16,), dtype=torch.float8_e4m3fn, device="cuda")
    _launch(_Quantize(input_stage, norm), x, scaled, q, sf)
    _assert_numeric_bits(scaled.flatten(), [v for row in expected_scaled for v in row])
    assert torch.equal(q.cpu(), torch.tensor(expected_q, dtype=torch.uint8))
    assert torch.equal(
        sf.cpu().view(torch.uint8),
        torch.tensor(expected_sf, dtype=torch.float8_e4m3fn).view(torch.uint8),
    )


class _Fc2:
    @cute.jit
    def __call__(self, x: cute.Tensor, a: cute.Tensor, y: cute.Tensor, stream):
        self.kernel(x, a, y).launch(grid=(1, 1, 1), block=(128, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, x: cute.Tensor, a: cute.Tensor, y: cute.Tensor):
        i, _, _ = cute.arch.thread_idx()
        if i < x.shape[0]:
            y[i] = fc2_output(ScalarF32, x[i], alpha=a[i])


def test_nvfp4_fc2_scale_precedes_bf16_wire_rounding():
    _require_cuda()
    import torch

    x = torch.tensor(
        [
            0.0,
            -0.0,
            2.0**-149,
            -(2.0**-149),
            1.00390625,
            -1.00390625,
            3.4028234663852886e38,
            -80.0,
        ],
        device="cuda",
    )
    a = torch.tensor([1.0, 0.5, 1.25, 1.25, 0.75, 0.75, 0.25, -0.5], device="cuda")
    y = torch.empty_like(x, dtype=torch.bfloat16)
    _launch(_Fc2(), x, a, y)
    expected = [
        exact.mul(v, s) for v, s in zip(x.cpu().tolist(), a.cpu().tolist(), strict=True)
    ]
    _assert_numeric_bits(y, expected, bf16=True)
