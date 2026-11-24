# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Exact RN/conversion oracles for legacy and next MXFP8 arithmetic leaves."""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
import pytest

from flashinfer.batch_invariant.fp_interface import exact
from flashinfer.moe_ep.batch_invariant.mxfp8_device import ScalarF32, PackedF32
from flashinfer.moe_ep.batch_invariant.mxfp8_spec import (
    half_amax,
    input_half_amax,
    combine_half_amax,
    input_quantize_amax_precast,
    legacy_quantize_block_precast,
    quantize_half_precast,
    legacy_fc1_activation,
    next_fc1_activation,
)

from .test_bf16_megamoe_batch_invariant import _require_cuda
from .test_bf16_megamoe_spec import _assert_numeric_bits, _launch, _sample
from .test_nvfp4_megamoe_spec import _activation_oracle as _next_activation_oracle

pytestmark = pytest.mark.gpu


class _Quantize:
    def __init__(self, kind, recipe):
        self.kind, self.recipe = kind, recipe

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        pre: cute.Tensor,
        out: cute.Tensor,
        sf: cute.Tensor,
        stream,
    ):
        self.kernel(x, pre, out, sf).launch(
            grid=(1, 1, 1), block=(128, 1, 1), stream=stream
        )

    @cute.kernel
    def kernel(
        self, x: cute.Tensor, pre: cute.Tensor, out: cute.Tensor, sf: cute.Tensor
    ):
        row, _, _ = cute.arch.thread_idx()
        if row < x.shape[0]:
            values = tuple(x[row, i] for i in range(32))
            if cutlass.const_expr(self.recipe == "legacy"):
                scaled, scale = legacy_quantize_block_precast(
                    ScalarF32, values, quant_kind=self.kind
                )
                stored = ScalarF32.cvt_out(scale, "ue8m0")
            elif cutlass.const_expr(self.recipe == "input"):
                maximum = combine_half_amax(
                    ScalarF32,
                    input_half_amax(ScalarF32, values[:16]),
                    input_half_amax(ScalarF32, values[16:]),
                )
                scaled, stored = input_quantize_amax_precast(
                    ScalarF32, values, maximum, self.kind
                )
            else:
                low, high = (
                    half_amax(ScalarF32, values[:16]),
                    half_amax(ScalarF32, values[16:]),
                )
                lo, stored = quantize_half_precast(
                    ScalarF32, values[:16], low, high, quant_kind=self.kind
                )
                hi, _ = quantize_half_precast(
                    ScalarF32, values[16:], low, high, quant_kind=self.kind
                )
                scaled = lo + hi
            sf[row] = stored
            regs = cute.make_rmem_tensor((32,), cutlass.Float32)
            for i in cutlass.range_constexpr(32):
                regs[i], pre[row, i] = scaled[i], scaled[i]
            if cutlass.const_expr(self.kind == "mxfp8_e4m3"):
                converted = cute.make_rmem_tensor((32,), cutlass.Float8E4M3FN)
                converted.store(regs.load().to(cutlass.Float8E4M3FN))
            else:
                converted = cute.make_rmem_tensor((32,), cutlass.Float8E5M2)
                converted.store(regs.load().to(cutlass.Float8E5M2))
            bytes_ = cute.recast_tensor(converted, cutlass.Uint8)
            for i in cutlass.range_constexpr(32):
                out[row, i] = bytes_[i]


def _quant_oracle(rows, kind, recipe):
    fmt = "e4m3" if kind == "mxfp8_e4m3" else "e5m2"
    limit = 448.0 if fmt == "e4m3" else 57344.0
    scales = []
    for values in rows:
        candidate = exact.mul(max(abs(v) for v in values), exact.to_f32(1 / limit))
        if recipe == "input":
            candidate = max(candidate, 2.0**-126)
        elif recipe == "legacy":
            candidate = exact.mul(candidate, 1.0)
        scales.append(exact.to_ue8m0(candidate, "rp"))
    reciprocals = _sample("rcp", scales)
    scaled, data = [], []
    for values, scale, reciprocal in zip(rows, scales, reciprocals, strict=True):
        if recipe == "legacy":
            reciprocal = exact.mul(1.0, reciprocal)
        encoding = min(reciprocal, exact.F32_MAX)
        if recipe == "next":
            encoding = exact.mul(
                encoding, min(exact.mul(scale, exact.to_f32(1e30)), 1.0)
            )
        row = [exact.mul(v, encoding) for v in values]
        scaled.append(row)
        data.append([exact.to_fp8(v, fmt) for v in row])
    return scaled, data, [exact.ue8m0_byte(exact.to_ue8m0(v, "rp")) for v in scales]


@pytest.mark.parametrize("kind", ("mxfp8_e4m3", "mxfp8_e5m2"))
@pytest.mark.parametrize("recipe", ("input", "legacy", "next"))
def test_mxfp8_quantization_matches_independent_exact_reference(kind, recipe):
    _require_cuda()
    import torch

    gen = torch.Generator().manual_seed(8123)
    rows = torch.randn((16, 32), generator=gen).mul_(4)
    rows[0].zero_()
    rows[0, 1::2] = -0.0
    for index, exponent in enumerate((-149, -126, -120, -112, -110), 1):
        rows[index] = torch.linspace(-(2.0**exponent), 2.0**exponent, 32)
    rows[6] = torch.linspace(-448.0, 448.0, 32)
    rows[7] = torch.linspace(-57344.0, 57344.0, 32)
    rows[8, :16] *= 128
    rows[9, 16:] *= 128
    rows[10] = torch.linspace(-3.0e38, 3.0e38, 32, dtype=torch.float64).float()
    if recipe == "input":
        rows = rows.to(torch.bfloat16).float()
    expected_pre, expected_data, expected_sf = _quant_oracle(
        rows.tolist(), kind, recipe
    )
    x = rows.to("cuda")
    pre, output = torch.empty_like(x), torch.empty_like(x, dtype=torch.uint8)
    sf = torch.empty((16,), device="cuda", dtype=torch.float8_e8m0fnu)
    _launch(_Quantize(kind, recipe), x, pre, output, sf)
    _assert_numeric_bits(pre.flatten(), [v for row in expected_pre for v in row])
    dtype = torch.float8_e4m3fn if kind == "mxfp8_e4m3" else torch.float8_e5m2
    assert torch.equal(
        output.cpu(), torch.tensor(expected_data, dtype=dtype).view(torch.uint8)
    )
    assert torch.equal(
        sf.cpu().view(torch.uint8), torch.tensor(expected_sf, dtype=torch.uint8)
    )


class _Activation:
    def __init__(self, legacy, packed, clamp):
        self.legacy, self.packed, self.clamp = legacy, packed, clamp

    @cute.jit
    def __call__(
        self, g: cute.Tensor, u: cute.Tensor, p: cute.Tensor, out: cute.Tensor, stream
    ):
        self.kernel(g, u, p, out).launch(
            grid=(1, 1, 1), block=(128, 1, 1), stream=stream
        )

    @cute.kernel
    def kernel(self, g: cute.Tensor, u: cute.Tensor, p: cute.Tensor, out: cute.Tensor):
        i, _, _ = cute.arch.thread_idx()
        if cutlass.const_expr(self.packed):
            i *= 2
            if i + 1 < g.shape[0]:
                if cutlass.const_expr(self.legacy):
                    result = legacy_fc1_activation(
                        PackedF32,
                        (g[i], g[i + 1]),
                        (u[i], u[i + 1]),
                        weight=(p[i], p[i + 1]),
                        clamp=self.clamp,
                    )
                else:
                    result = next_fc1_activation(
                        PackedF32,
                        (g[i], g[i + 1]),
                        (u[i], u[i + 1]),
                        weight=(p[i], p[i + 1]),
                        clamp=self.clamp,
                    )
                out[i], out[i + 1] = result
        elif i < g.shape[0]:
            if cutlass.const_expr(self.legacy):
                out[i] = legacy_fc1_activation(
                    ScalarF32, g[i], u[i], weight=p[i], clamp=self.clamp
                )
            else:
                out[i] = next_fc1_activation(
                    ScalarF32, g[i], u[i], weight=p[i], clamp=self.clamp
                )


def _legacy_activation_oracle(gates, ups, weights, clamp):
    if clamp is not None:
        limit = abs(clamp)
        gates = [min(g, limit) for g in gates]
        ups = [max(min(u, limit), -limit) for u in ups]
    exponentials = _sample(
        "ex2", [exact.mul(g, exact.to_f32(-1.4426950408889634)) for g in gates]
    )
    reciprocals = _sample("rcp", [exact.add(e, 1.0) for e in exponentials])
    return [
        exact.mul(exact.mul(exact.mul(s, g), u), w)
        for s, g, u, w in zip(reciprocals, gates, ups, weights, strict=True)
    ]


@pytest.mark.parametrize("legacy", (False, True))
@pytest.mark.parametrize("clamp", (None, -2.5))
def test_mxfp8_activation_preserves_each_provider_association(legacy, clamp):
    _require_cuda()
    import torch

    gen = torch.Generator().manual_seed(8317)
    g, u, w = [torch.randn(64, generator=gen).to("cuda") for _ in range(3)]
    g[:4] = torch.tensor([0.0, -0.0, 2.5, -2.5], device="cuda")
    w[:4] = torch.tensor([0.0, -0.0, 1.0, -1.0], device="cuda")
    expected = (
        _legacy_activation_oracle(
            g.cpu().tolist(), u.cpu().tolist(), w.cpu().tolist(), clamp
        )
        if legacy
        else _next_activation_oracle(
            g.cpu().tolist(), u.cpu().tolist(), None, w.cpu().tolist(), clamp
        )
    )
    results = []
    for packed in (False, True):
        output = torch.empty_like(g)
        _launch(_Activation(legacy, packed, clamp), g, u, w, output)
        _assert_numeric_bits(output, expected)
        results.append(output)
    assert torch.equal(results[0].view(torch.int32), results[1].view(torch.int32))
