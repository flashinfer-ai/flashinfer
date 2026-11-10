# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Fused Triton NVFP4 input quantizer: bit-identical to :func:`nvfp4_quantize_torch` for a scalar ``norm``.

The torch reference is ~25 elementwise passes over ``[rows, H]``; this kernel reads the BF16/FP32 input
once and writes the packed E2M1 codes and the E4M3 block-16 scales (same rule: ``scale =
e4m3(min((amax / 6) * norm, 448))``, ``codes = e2m1(x * (norm * (1 / scale)))`` with an IEEE reciprocal,
ties to the even code).
"""

from __future__ import annotations

import torch

try:  # Triton ships with PyTorch CUDA builds; the torch reference is the fallback
    import triton
    import triton.language as tl

    _HAVE_TRITON = True
except ImportError:  # pragma: no cover
    _HAVE_TRITON = False

if _HAVE_TRITON:

    @triton.jit
    def _e2m1_code(v):
        mag = tl.abs(v)
        code = (
            (mag > 0.25).to(tl.int32)
            + (mag > 1.25).to(tl.int32)
            + (mag > 2.5).to(tl.int32)
            + (mag > 5.0).to(tl.int32)
        )
        code += (
            (mag >= 0.75).to(tl.int32)
            + (mag >= 1.75).to(tl.int32)
            + (mag >= 3.5).to(tl.int32)
        )
        neg = (v.to(tl.int32, bitcast=True) < 0).to(tl.int32)
        return code | (neg << 3)

    @triton.jit
    def _nvfp4_quant_kernel(
        x_ptr, codes_ptr, sf_ptr, norm, hidden, BLOCKS: tl.constexpr
    ):
        # one program = one row x BLOCKS consecutive 16-value blocks
        row = tl.program_id(0)
        blk0 = tl.program_id(1) * BLOCKS
        b = blk0 + tl.arange(0, BLOCKS)[:, None]  # [BLOCKS, 1]
        j = tl.arange(0, 8)[None, :]  # pair index inside a block
        col_e = b * 16 + 2 * j
        mask = (b * 16 + 16) <= hidden
        base = x_ptr + row.to(tl.int64) * hidden
        xe = tl.load(base + col_e, mask=mask, other=0.0).to(tl.float32)
        xo = tl.load(base + col_e + 1, mask=mask, other=0.0).to(tl.float32)
        amax = tl.maximum(
            tl.max(tl.abs(xe), axis=1), tl.max(tl.abs(xo), axis=1)
        )  # [BLOCKS]
        raw = (amax * 0.16666666666666666) * norm
        sc8 = tl.minimum(raw, 448.0).to(tl.float8e4nv)
        scale = sc8.to(tl.float32)
        inv = tl.where(scale > 0.0, norm * tl.math.div_rn(1.0, scale), 0.0)
        lo = _e2m1_code(xe * inv[:, None])
        hi = _e2m1_code(xo * inv[:, None])
        packed = (lo | (hi << 4)).to(tl.uint8)
        tl.store(
            codes_ptr + row.to(tl.int64) * (hidden // 2) + b * 8 + j, packed, mask=mask
        )
        bo = blk0 + tl.arange(0, BLOCKS)
        tl.store(
            sf_ptr + row.to(tl.int64) * (hidden // 16) + bo,
            sc8.to(tl.uint8, bitcast=True),
            mask=(bo * 16 + 16) <= hidden,
        )


def nvfp4_quantize_fused(
    values: torch.Tensor,
    norm_const: float = 1.0,
    *,
    codes: torch.Tensor | None = None,
    sf: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``[rows, H]`` BF16/FP32 CUDA tensor -> (packed E2M1 ``uint8 [rows, H // 2]``, E4M3 bytes ``uint8 [rows, H // 16]``).

    ``codes`` / ``sf`` optionally receive the result in place (leading ``rows`` rows of larger buffers).
    """

    if not _HAVE_TRITON:
        raise RuntimeError("triton is not available")
    rows, hidden = values.shape
    if hidden % 16:
        raise ValueError(
            f"the last dim must be a multiple of 16, got {tuple(values.shape)}"
        )
    values = values.contiguous()
    if codes is None:
        codes = torch.empty(
            (rows, hidden // 2), dtype=torch.uint8, device=values.device
        )
    if sf is None:
        sf = torch.empty((rows, hidden // 16), dtype=torch.uint8, device=values.device)
    if rows:
        blocks = 16
        grid = (rows, triton.cdiv(hidden // 16, blocks))
        _nvfp4_quant_kernel[grid](
            values, codes, sf, float(norm_const), hidden, BLOCKS=blocks
        )
    return codes[:rows], sf[:rows]


__all__ = ["nvfp4_quantize_fused"]
