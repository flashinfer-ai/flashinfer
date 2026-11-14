# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Independent device checks of MXFP4 MegaMoE's arithmetic leaves."""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
import pytest

from flashinfer.batch_invariant.fp_interface import exact
from flashinfer.moe_ep.batch_invariant.mxfp4_device import PackedF32, ScalarF32
from flashinfer.moe_ep.batch_invariant.mxfp4_spec import (
    half_amax,
    input_quantize_block_precast,
    quantize_half_precast,
)
from flashinfer.moe_ep.batch_invariant.nvfp4_spec import fc1_activation

from .test_bf16_megamoe_batch_invariant import _require_cuda
from .test_bf16_megamoe_spec import _assert_numeric_bits, _launch, _sample
from .test_nvfp4_megamoe_spec import _activation_oracle

pytestmark = pytest.mark.gpu


class _Scale:
    @cute.jit
    def __call__(self, x: cute.Tensor, y: cute.Tensor, sf: cute.Tensor, stream):
        self.kernel(x, y, sf).launch(
            grid=(cute.ceil_div(x.shape[0], 128), 1, 1),
            block=(128, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(self, x: cute.Tensor, y: cute.Tensor, sf: cute.Tensor):
        t, _, _ = cute.arch.thread_idx()
        b, _, _ = cute.arch.block_idx()
        i = b * 128 + t
        if i < x.shape[0]:
            y[i] = ScalarF32.ue8m0_roundtrip(x[i])
            sf[i] = ScalarF32.cvt_out(y[i], "ue8m0")


def test_ue8m0_roundtrip_and_stored_bytes_match_exact_reference():
    _require_cuda()
    import torch

    values = [
        0.0,
        -0.0,
        2.0**-149,
        -(2.0**-149),
        2.0**-127,
        2.0**-126,
        float("inf"),
        float("-inf"),
        float("nan"),
    ]
    for exponent in range(-126, 128, 8):
        value = torch.tensor(2.0**exponent, dtype=torch.float32)
        values += [
            float(value),
            float(torch.nextafter(value, torch.tensor(0.0))),
            float(torch.nextafter(value, torch.tensor(float("inf")))),
        ]
    x = torch.tensor(values, device="cuda", dtype=torch.float32)
    y = torch.empty_like(x)
    sf = torch.empty_like(x, dtype=torch.float8_e8m0fnu)
    _launch(_Scale(), x, y, sf)
    expected = [exact.to_ue8m0(v, "rp") for v in x.cpu().tolist()]
    _assert_numeric_bits(y, expected)
    codes = [exact.ue8m0_byte(exact.to_ue8m0(v, "rp")) for v in expected]
    assert torch.equal(
        sf.cpu().view(torch.uint8), torch.tensor(codes, dtype=torch.uint8)
    )
    assert y[0].item() == 2.0**-127 and sf[0].view(torch.uint8).item() == 0


class _Quantize:
    def __init__(self, kind, staging):
        self.kind, self.staging = kind, staging

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        scaled: cute.Tensor,
        data: cute.Tensor,
        sf: cute.Tensor,
        peer_sf: cute.Tensor,
        stream,
    ):
        self.kernel(x, scaled, data, sf, peer_sf).launch(
            grid=(cute.ceil_div(x.shape[0], 128), 1, 1),
            block=(128, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        x: cute.Tensor,
        scaled: cute.Tensor,
        data: cute.Tensor,
        sf: cute.Tensor,
        peer_sf: cute.Tensor,
    ):
        t, _, _ = cute.arch.thread_idx()
        b, _, _ = cute.arch.block_idx()
        row = b * 128 + t
        if row < x.shape[0]:
            values = tuple(x[row, i] for i in range(32))
            if cutlass.const_expr(self.staging):
                precast, scale = input_quantize_block_precast(
                    ScalarF32, values, self.kind
                )
                other_scale = scale
            else:
                low = half_amax(ScalarF32, values[:16])
                high = half_amax(ScalarF32, values[16:])
                lo, scale = quantize_half_precast(
                    ScalarF32, values[:16], low, high, quant_kind=self.kind
                )
                hi, other_scale = quantize_half_precast(
                    ScalarF32, values[16:], low, high, quant_kind=self.kind
                )
                precast = lo + hi
            registers = cute.make_rmem_tensor((32,), cutlass.Float32)
            for i in cutlass.range_constexpr(32):
                registers[i] = precast[i]
                scaled[row, i] = precast[i]
            sf[row], peer_sf[row] = scale, other_scale
            if cutlass.const_expr(self.kind == "mxfp4"):
                quantized = cute.make_rmem_tensor((32,), cutlass.Float4E2M1FN)
                quantized.store(registers.load().to(cutlass.Float4E2M1FN))
                packed = cute.recast_tensor(quantized, cutlass.Uint8)
                for i in cutlass.range_constexpr(16):
                    data[row, i] = packed[i]
            else:
                quantized = cute.make_rmem_tensor((32,), cutlass.Float8E4M3FN)
                quantized.store(registers.load().to(cutlass.Float8E4M3FN))
                packed = cute.recast_tensor(quantized, cutlass.Uint8)
                for i in cutlass.range_constexpr(32):
                    data[row, i] = packed[i]


def _quantization_oracle(blocks, kind, staging):
    # This oracle deliberately spells out the arithmetic independently of the
    # production helpers. The finite data domain makes both ordered amax
    # algorithms equal to this maximum; CPU/Lean pins check their exact trees.
    maximum = 6.0 if kind == "mxfp4" else 448.0
    scales = []
    for row in blocks:
        candidate = exact.mul(max(abs(v) for v in row), exact.to_f32(1 / maximum))
        if staging:
            candidate = max(candidate, 2.0**-126)
        scales.append(exact.to_ue8m0(candidate, "rp"))
    reciprocals = _sample("rcp", scales)
    scaled, quantized = [], []
    for row, scale, reciprocal in zip(blocks, scales, reciprocals, strict=True):
        encoding = min(reciprocal, exact.F32_MAX)
        if not staging:
            encoding = exact.mul(
                encoding, min(exact.mul(scale, exact.to_f32(1e30)), 1.0)
            )
        values = [exact.mul(v, encoding) for v in row]
        scaled.append(values)
        if kind == "mxfp4":
            nibbles = [exact.e2m1_code(exact.to_e2m1(v)) for v in values]
            quantized.append(
                [nibbles[i] | nibbles[i + 1] << 4 for i in range(0, 32, 2)]
            )
        else:
            quantized.append([exact.to_fp8(v, "e4m3") for v in values])
    codes = [exact.ue8m0_byte(exact.to_ue8m0(v, "rp")) for v in scales]
    return scaled, quantized, codes


@pytest.mark.parametrize("kind", ("mxfp4", "mxfp4_mxfp8"))
@pytest.mark.parametrize("staging", (False, True))
def test_block32_quantization_matches_independent_exact_reference(kind, staging):
    _require_cuda()
    import torch

    generator = torch.Generator().manual_seed(4321)
    rows = torch.randn((16, 32), generator=generator).mul_(4)
    rows[0].zero_()
    rows[0, 1::2] = -0.0
    rows[1] = torch.tensor([(-1.0) ** i * 2.0**-149 for i in range(32)])
    rows[2] = torch.linspace(-(2.0**-126), 2.0**-126, 32)
    rows[3] = torch.linspace(-(2.0**-110), 2.0**-110, 32)
    rows[4] = torch.linspace(-6.0, 6.0, 32)
    rows[5] = torch.linspace(-448.0, 448.0, 32)
    # The maximum belongs to a different logical half in each block.
    rows[6, :16] *= 128
    rows[7, 16:] *= 128
    rows[8] = torch.linspace(-3.0e38, 3.0e38, 32, dtype=torch.float64).float()
    if staging:
        rows = rows.to(torch.bfloat16).to(torch.float32)
    expected_scaled, expected_data, expected_sf = _quantization_oracle(
        rows.tolist(), kind, staging
    )
    x = rows.to("cuda")
    scaled = torch.empty_like(x)
    data = torch.empty(
        (16, 16 if kind == "mxfp4" else 32), dtype=torch.uint8, device="cuda"
    )
    sf = torch.empty((16,), dtype=torch.float8_e8m0fnu, device="cuda")
    peer_sf = torch.empty_like(sf)
    _launch(_Quantize(kind, staging), x, scaled, data, sf, peer_sf)
    _assert_numeric_bits(scaled.flatten(), [v for row in expected_scaled for v in row])
    if kind == "mxfp4":
        expected = torch.tensor(expected_data, dtype=torch.uint8)
    else:
        expected = torch.tensor(expected_data, dtype=torch.float8_e4m3fn).view(
            torch.uint8
        )
    assert torch.equal(data.cpu(), expected)
    assert torch.equal(
        sf.cpu().view(torch.uint8), torch.tensor(expected_sf, dtype=torch.uint8)
    )
    assert torch.equal(sf.view(torch.uint8), peer_sf.view(torch.uint8))


class _Activation:
    def __init__(self, packed, clamp):
        self.packed, self.clamp = packed, clamp

    @cute.jit
    def __call__(
        self,
        gate: cute.Tensor,
        up: cute.Tensor,
        weight: cute.Tensor,
        output: cute.Tensor,
        stream,
    ):
        self.kernel(gate, up, weight, output).launch(
            grid=(1, 1, 1), block=(128, 1, 1), stream=stream
        )

    @cute.kernel
    def kernel(
        self,
        gate: cute.Tensor,
        up: cute.Tensor,
        weight: cute.Tensor,
        output: cute.Tensor,
    ):
        i, _, _ = cute.arch.thread_idx()
        if cutlass.const_expr(self.packed):
            i *= 2
            if i + 1 < gate.shape[0]:
                output[i], output[i + 1] = fc1_activation(
                    PackedF32,
                    (gate[i], gate[i + 1]),
                    (up[i], up[i + 1]),
                    weight=(weight[i], weight[i + 1]),
                    clamp=self.clamp,
                )
        elif i < gate.shape[0]:
            output[i] = fc1_activation(
                ScalarF32, gate[i], up[i], weight=weight[i], clamp=self.clamp
            )


@pytest.mark.parametrize("clamp", (None, -2.5))
def test_mxfp4_scalar_and_packed_activation_reuse_the_ordered_chain(clamp):
    _require_cuda()
    import torch

    generator = torch.Generator().manual_seed(4815)
    gate, up, weight = [
        torch.randn(64, generator=generator).to("cuda") for _ in range(3)
    ]
    gate[:4] = torch.tensor([0.0, -0.0, 2.5, -2.5], device="cuda")
    weight[:4] = torch.tensor([0.0, -0.0, 1.0, -1.0], device="cuda")
    expected = _activation_oracle(
        gate.cpu().tolist(), up.cpu().tolist(), None, weight.cpu().tolist(), clamp
    )
    outputs = []
    for packed in (False, True):
        output = torch.empty_like(gate)
        _launch(_Activation(packed, clamp), gate, up, weight, output)
        _assert_numeric_bits(output, expected)
        outputs.append(output)
    assert torch.equal(outputs[0].view(torch.int32), outputs[1].view(torch.int32))
