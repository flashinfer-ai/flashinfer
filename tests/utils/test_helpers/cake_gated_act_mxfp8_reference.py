"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Plain-torch reference of the fused SwiGLU + MXFP8 quantization (forward and
backward), exact to the byte, for the Cake backend suite.

The math is ``s = 1 / (1 + 2^(-x log2 e))``, ``act = x * s``; the forward
quantizes ``act * up``, the backward ``dgate = (fma(act, 1 - s, s) * grad) *
up`` and ``dup = act * grad`` (columns ``[0, K)`` and ``[K, 2K)``), each step
one round-to-nearest FP32 operation and ``2^t`` torch's ``exp2`` (the
``ex2.approx.f32`` the kernels use) on the device.  Each value is rounded to
BF16 and quantized per 32-element block of a row and of a column: the E8M0
scale is RCEIL (the exponent of the block's BF16 absolute maximum, plus one
when its mantissa is above 1.75, less 8, clamped at 0, and 255 for an Inf or
NaN maximum, whose codes are all NaN, 0x7F), and a code is the BF16 value
times ``2^(127 - scale)`` converted to E4M3, round to nearest even.  The
scales are laid out in swizzled 128 x 4 tiles of 512 bytes.

``values`` picks the inputs: ``"normal"``, unit normals; ``"scaled"``, unit
normals with each 32-column block of a row scaled by a power of two (2^-8 to
2^8), so the scale bytes span a range; ``"special"``, the scaled normals with
NaN, Inf, -0, zero, subnormal and overflowing blocks and gates at and below
-87; ``"bits"``, every BF16 bit pattern as a gate.
"""

from __future__ import annotations

import torch

LOG2E = 1.4426950408889634
BLOCK = 32
TILE = 128
SF_TILE_BYTES = 512
NAN_CODE = 0x7F
NAN_SCALE = 255
VALUES = ("normal", "scaled", "special", "bits")


def make_inputs(
    M: int, K: int, *, values: str = "normal", seed: int = 0, device: str = "cuda"
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(gated_input [M, 2K], grad_output [M, K])``, both BF16 on ``device``."""

    if values not in VALUES:
        raise ValueError(f"values is one of {VALUES}, got {values!r}")
    generator = torch.Generator(device=device).manual_seed(seed)
    draw = _unit_normals if values == "normal" else _scaled_normals
    gate = draw(M, K, generator, device)
    up = draw(M, K, generator, device)
    grad = draw(M, K, generator, device)
    if values == "special":
        _inject_specials(gate, up, grad)
    elif values == "bits":
        bits = (torch.arange(M * K, device=device, dtype=torch.int32) % 65536).to(
            torch.int16
        )
        gate = bits.view(torch.bfloat16).float().reshape(M, K)
    gated_input = torch.cat([gate, up], dim=1).to(torch.bfloat16).contiguous()
    return gated_input, grad.to(torch.bfloat16).contiguous()


def _unit_normals(M, K, generator, device) -> torch.Tensor:
    return torch.randn(M, K, device=device, generator=generator)


def _scaled_normals(M, K, generator, device) -> torch.Tensor:
    blocks = torch.randn(M, K // BLOCK, BLOCK, device=device, generator=generator)
    exponent = torch.randint(
        -8, 9, (M, K // BLOCK, 1), device=device, generator=generator
    )
    return (blocks * torch.exp2(exponent.float())).reshape(M, K)


def _inject_specials(gate: torch.Tensor, up: torch.Tensor, grad: torch.Tensor) -> None:
    """Special values in rows 0-17 (``M >= 32``, ``K >= 128``), each in a block of its own."""

    gate[0, 0:32] = torch.linspace(
        -100.0, -80.0, 32
    )  # around 1 + 2^(-x log2 e) = 2^126
    gate[1, 32:64] = torch.linspace(-89.5, -86.5, 32)
    gate[2, 64:96] = float("nan")
    gate[3, 96:128] = float("inf")
    gate[4, 0:32] = float("-inf")
    up[5, 32:64] = float("inf")
    up[6, 64:96] = float("nan")
    gate[7, 96:128] = 0.0  # a zero block: scale 0
    up[7, 96:128] = 0.0
    gate[8, 0:32] = -0.0
    up[9, 32:64] *= 2.0**-128  # BF16 subnormal products: the scale clamps at 0
    gate[10, 64:96] = 10.0  # the product overflows FP32 to Inf: scale 255
    up[10, 64:96] = 3.0e38
    gate[11, 96:128] = 10.0  # finite products near the BF16 maximum
    up[11, 96:128] = 2.0e37
    gate[12, 0:16] = float("nan")  # NaN and Inf in one block
    gate[12, 16:32] = float("inf")
    grad[13, 0:32] = float("nan")
    grad[14, 32:64] = float("inf")
    grad[15, 64:96] = 0.0
    grad[16, 96:128] = -0.0
    grad[17, 0:32] = 3.0e38  # dact * grad and act * grad overflow for some gates


def _sigmoid(x: torch.Tensor) -> torch.Tensor:
    return 1.0 / (1.0 + torch.exp2(-x * LOG2E))


def _fma(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """``fma.rn.f32(a, b, c)`` of FP32 tensors: ``a * b`` is exact in FP64 and the
    sum rounds there once; a sum that FP64 rounded onto an FP32 midpoint moves to
    the side of its exact value (the TwoSum error) before rounding to FP32."""

    product = a.double() * b.double()
    total = product + c.double()
    virtual = total - product
    error = (product - (total - virtual)) + (c.double() - virtual)
    rounded = total.float()
    away = torch.nextafter(
        rounded,
        torch.where(total > rounded.double(), torch.inf, -torch.inf).float(),
    )
    midpoint = ((rounded.double() + away.double()) / 2 == total) & total.isfinite()
    toward_away = (error > 0) == (away > rounded)
    return torch.where(midpoint & (error != 0) & toward_away, away, rounded)


def _block_scale(amax: torch.Tensor) -> torch.Tensor:
    bits = amax.view(torch.int32)
    scale = (((bits + 0x1F0000) >> 23) - 8).clamp(min=0)
    return torch.where((bits & 0x7F800000) == 0x7F800000, NAN_SCALE, scale)


def _quantize_blocks(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    scale = _block_scale(values.abs().amax(-1))
    inverse = ((254 - scale.clamp(max=254)) << 23).view(torch.float32).unsqueeze(-1)
    codes = (values * inverse).to(torch.float8_e4m3fn).view(torch.uint8)
    codes = torch.where((scale == NAN_SCALE).unsqueeze(-1), NAN_CODE, codes)
    return codes.to(torch.uint8), scale.to(torch.uint8)


def _scale_index(rows: int, cols: int) -> torch.Tensor:
    """The swizzled offset of the scale of row ``r``, block ``b`` of 32 columns."""

    r = torch.arange(rows, device="cuda").unsqueeze(1)
    b = torch.arange(cols // BLOCK, device="cuda").unsqueeze(0)
    tile = (r // TILE) * (cols // TILE) + b // 4
    return tile * SF_TILE_BYTES + (r % 32) * 16 + (r % TILE // 32) * 4 + b % 4


def _quantize(values: torch.Tensor, *, rowwise: bool, colwise: bool) -> dict:
    """The bytes of the outputs of the BF16-exact FP32 ``values`` ``[M, N]``:
    ``row_output`` ``[M, N]``, ``col_output`` as its ``[N, M]`` storage, and the
    flat swizzled scale bytes."""

    M, N = values.shape
    expected = {}
    if rowwise:
        codes, scale = _quantize_blocks(values.reshape(M, N // BLOCK, BLOCK))
        scales = torch.empty(M * N // BLOCK, device="cuda", dtype=torch.uint8)
        scales[_scale_index(M, N).reshape(-1)] = scale.reshape(-1)
        expected["row_output"] = codes.reshape(M, N)
        expected["row_scales"] = scales
    if colwise:
        codes, scale = _quantize_blocks(values.t().reshape(N, M // BLOCK, BLOCK))
        scales = torch.empty(N * M // BLOCK, device="cuda", dtype=torch.uint8)
        scales[_scale_index(N, M).reshape(-1)] = scale.reshape(-1)
        expected["col_output"] = codes.reshape(N, M)
        expected["col_scales"] = scales
    return expected


def forward(gated_input: torch.Tensor, *, rowwise: bool, colwise: bool) -> dict:
    """The expected bytes of ``silu_and_mul_mxfp8_quantize``."""

    K = gated_input.shape[1] // 2
    gate, up = gated_input.float().split(K, dim=1)
    product = ((gate * _sigmoid(gate)) * up).to(torch.bfloat16).float()
    return _quantize(product, rowwise=rowwise, colwise=colwise)


def backward(
    gated_input: torch.Tensor,
    grad_output: torch.Tensor,
    *,
    rowwise: bool,
    colwise: bool,
) -> dict:
    """The expected bytes of ``silu_and_mul_mxfp8_quantize_backward``."""

    K = gated_input.shape[1] // 2
    x, up = gated_input.float().split(K, dim=1)
    grad = grad_output.float()
    s = _sigmoid(x)
    act = x * s
    dact = _fma(act, 1.0 - s, s)
    dgate = ((dact * grad) * up).to(torch.bfloat16).float()
    dup = (act * grad).to(torch.bfloat16).float()
    return _quantize(torch.cat([dgate, dup], dim=1), rowwise=rowwise, colwise=colwise)


def output_bytes(outputs, *, rowwise: bool, colwise: bool) -> dict:
    """The bytes of the four tensors an entry point returns, in the reference's form."""

    row_output, col_output, row_scales, col_scales = outputs
    got = {}
    if rowwise:
        got["row_output"] = row_output.view(torch.uint8)
        got["row_scales"] = row_scales.view(torch.uint8).reshape(-1)
    if colwise:
        got["col_output"] = col_output.t().view(torch.uint8)
        got["col_scales"] = col_scales.view(torch.uint8).reshape(-1)
    return got
