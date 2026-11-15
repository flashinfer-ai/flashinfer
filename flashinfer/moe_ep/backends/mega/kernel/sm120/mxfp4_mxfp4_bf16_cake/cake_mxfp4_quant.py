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

"""MXFP4 activation quantizer and fused stager of the Cake SM120 MXFP4 x MXFP4 (W4A4) MegaMoE backend.

Rule (per 32 values, MOE_ACT_CODEC=2 wire format):

* ``ratio = amax * fl32(1 / 6)`` -- a single FP32 multiplication by the rounded reciprocal (what
  torch evaluates for ``tensor / 6.0`` on CUDA, and the factor of the kernel's in-kernel SwiGLU
  requantization, ``REQUANT_INV_RANGE``); a NaN anywhere in the block makes ``amax`` NaN;
* ``e = clamp(exp(ratio) + (mantissa(ratio) != 0), 1, 254)`` (round the ratio up to a power of two;
  an all-zero block gets the scale byte 1, NaN / inf blocks 254);
* the UE8M0 scale byte is ``e`` (scale ``2^(e - 127)``) and the codes are
  ``e2m1(x * 2^(127 - e))``: round to nearest, ties to the even code, saturating at 6, sign = the
  FP32 sign bit (NaN -> code 0 with its sign).  The power-of-two scale makes the product exact.

:func:`mxfp4_quantize_torch` is the reference (device independent: the reciprocal is explicit);
:func:`mxfp4_quantize_fused` is its bit-identical Triton twin: one launch reads the BF16 / FP16 /
FP32 activation once, writes the packed codes and the scale bytes straight into the workspace's
staging rows and (optionally) copies the routing rows, so one launch stages a step.  No host
synchronization; stream-ordered and CUDA-graph capturable.
"""

from __future__ import annotations

import torch

from ..nvfp4_nvfp4_bf16_cake.cake_nvfp4_quant import _HAVE_TRITON
from ..nvfp4_nvfp4_bf16_cake.cake_nvfp4_weights import e2m1_codes

MXFP4_BLOCK = 32
UE8M0_SCALE_DTYPES = (torch.uint8, torch.float8_e8m0fnu)
# fl32(1 / 6) == np.float32(1) / np.float32(6)
RCP_6 = 0.1666666716337204


def _check_norm(norm_const: float) -> None:
    if float(norm_const) != 1.0:
        raise ValueError(
            f"MXFP4 activations carry no global scale (norm_const must be 1.0, got {norm_const!r})"
        )


def mxfp4_quantize_torch(
    values: torch.Tensor, norm_const: float = 1.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """fp32/fp16/bf16 ``[..., K]`` -> (packed E2M1 ``uint8 [..., K // 2]``, UE8M0 bytes ``uint8 [..., K // 32]``)."""

    _check_norm(norm_const)
    if values.shape[-1] % MXFP4_BLOCK:
        raise ValueError(
            f"the last dim must be a multiple of {MXFP4_BLOCK}, got {tuple(values.shape)}"
        )
    shape = values.shape
    blocks = values.float().reshape(*shape[:-1], shape[-1] // MXFP4_BLOCK, MXFP4_BLOCK)
    rcp = torch.tensor(RCP_6, dtype=torch.float32, device=values.device)
    bits = (blocks.abs().amax(dim=-1) * rcp).contiguous().view(torch.int32)
    exponent = (((bits >> 23) & 255) + ((bits & 0x7FFFFF) != 0).to(torch.int32)).clamp(
        min=1, max=254
    )
    inverse = ((254 - exponent) << 23).view(torch.float32)
    nibbles = e2m1_codes(blocks * inverse.unsqueeze(-1)).reshape(shape)
    packed = (nibbles[..., 0::2] | (nibbles[..., 1::2] << 4)).to(torch.uint8)
    return packed, exponent.to(torch.uint8)


def mxfp4_dequantize_torch(packed: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`mxfp4_quantize_torch` (float32 ``[..., K]``)."""

    table = torch.tensor(
        (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0),
        dtype=torch.float32,
        device=packed.device,
    )
    raw = packed.view(torch.uint8)
    codes = (
        torch.stack((raw & 15, raw >> 4), dim=-1)
        .reshape(*raw.shape[:-1], raw.shape[-1] * 2)
        .long()
    )
    values = table[codes & 7] * (1.0 - 2.0 * (codes >> 3).float())
    scale = (scales.view(torch.uint8).to(torch.int32) << 23).view(torch.float32)
    return values * scale.repeat_interleave(MXFP4_BLOCK, dim=-1)


if _HAVE_TRITON:
    import triton
    import triton.language as tl

    from ..nvfp4_nvfp4_bf16_cake.cake_nvfp4_quant import _e2m1_code

    @triton.jit
    def _mxfp4_quant_kernel(
        x_ptr,
        x_stride,
        codes_ptr,
        sf_ptr,
        hidden,
        ids_ptr,
        ids_stride,
        w_ptr,
        w_stride,
        ids_out_ptr,
        w_out_ptr,
        top_k,
        RCP: tl.constexpr,
        BLOCKS: tl.constexpr,
        COPY_ROUTING: tl.constexpr,
        TOPK_P2: tl.constexpr,
    ):
        # one program = one row x BLOCKS consecutive 32-value blocks
        row = tl.program_id(0).to(tl.int64)
        pid1 = tl.program_id(1)
        nblk = hidden // 32
        b = pid1 * BLOCKS + tl.arange(0, BLOCKS)  # [BLOCKS]
        bmask = b < nblk
        col = b[:, None] * 32 + tl.arange(0, 32)[None, :]  # [BLOCKS, 32]
        m2 = bmask[:, None]
        x = tl.load(x_ptr + row * x_stride + col, mask=m2, other=0.0).to(tl.float32)
        amax = tl.max(tl.abs(x), axis=1)
        has_nan = tl.max((x != x).to(tl.int32), axis=1)
        bits = (amax * RCP).to(tl.int32, bitcast=True)
        e = ((bits >> 23) & 255) + (((bits & 0x7FFFFF) + 0x7FFFFF) >> 23)
        e = tl.where(has_nan > 0, 254, tl.minimum(tl.maximum(e, 1), 254))
        inv = ((254 - e) << 23).to(tl.float32, bitcast=True)
        y = x * inv[:, None]
        lo, hi = tl.split(tl.reshape(y, (BLOCKS, 16, 2)))  # even / odd values
        packed = (_e2m1_code(lo) | (_e2m1_code(hi) << 4)).to(tl.uint8)  # [BLOCKS, 16]
        pcol = b[:, None] * 16 + tl.arange(0, 16)[None, :]
        tl.store(codes_ptr + row * (hidden // 2) + pcol, packed, mask=m2)
        tl.store(sf_ptr + row * nblk + b, e.to(tl.uint8), mask=bmask)
        if COPY_ROUTING:
            if pid1 == 0:
                k = tl.arange(0, TOPK_P2)
                km = k < top_k
                ids = tl.load(ids_ptr + row * ids_stride + k, mask=km)
                tl.store(ids_out_ptr + row * top_k + k, ids, mask=km)
                w = tl.load(w_ptr + row * w_stride + k, mask=km)
                tl.store(w_out_ptr + row * top_k + k, w, mask=km)


_DTYPES = (torch.bfloat16, torch.float16, torch.float32)
# 64 blocks (2048 values) per program, 4 warps: measured at 1024 x 4096 BF16 on RTX PRO 5000 (CUDA graph, routing
# copy included) 6.4 us hot-L2 / 10.4 us cold vs 8.3 / 10.9 us for 16 blocks; 8 / 32 blocks and 2 / 8 warps and
# strided even/odd loads instead of the [BLOCKS, 32] tile + split were not faster (mx4/qbench/qvar.py).
_BLOCKS = 64


def mxfp4_fused_supported(x: torch.Tensor) -> bool:
    return (
        _HAVE_TRITON
        and x.is_cuda
        and x.dim() == 2
        and x.dtype in _DTYPES
        and x.shape[1] % MXFP4_BLOCK == 0
        and (x.shape[0] == 0 or x.stride(1) == 1)
    )


def _routing_fusable(src: torch.Tensor, out: torch.Tensor, dtype: torch.dtype) -> bool:
    return (
        src.dtype == dtype
        and out.dtype == dtype
        and src.is_cuda
        and src.dim() == 2
        and out.dim() == 2
        and src.shape[1] == out.shape[1]
        and (src.shape[0] == 0 or src.stride(1) == 1)
        and out.stride(1) == 1
        and out.stride(0) == out.shape[1]
    )


def mxfp4_quantize_fused(
    values: torch.Tensor,
    norm_const: float = 1.0,
    *,
    codes: torch.Tensor | None = None,
    sf: torch.Tensor | None = None,
    topk_ids: torch.Tensor | None = None,
    topk_weights: torch.Tensor | None = None,
    topk_ids_out: torch.Tensor | None = None,
    topk_weights_out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``[rows, H]`` BF16/FP16/FP32 CUDA tensor -> (packed E2M1 ``uint8 [rows, H // 2]``, UE8M0 ``uint8 [rows, H // 32]``).

    ``codes`` / ``sf`` optionally receive the result in place (leading ``rows`` rows of larger,
    row-contiguous ``uint8`` buffers).  With all four routing tensors given, the routing rows are
    copied by the same launch (``topk_ids`` int64 / ``topk_weights`` float32 into the leading
    ``rows`` rows of the contiguous ``*_out`` buffers; ``copy_`` for other layouts).  Inputs with a
    non-unit inner stride are made contiguous first.
    """

    if not _HAVE_TRITON:
        raise RuntimeError("triton is not available")
    _check_norm(norm_const)
    if values.dim() != 2 or values.shape[1] % MXFP4_BLOCK:
        raise ValueError(
            f"fused MXFP4 quantizer needs a [rows, H] tensor with H % {MXFP4_BLOCK} == 0, got {tuple(values.shape)}"
        )
    if not mxfp4_fused_supported(values):
        values = values.contiguous()
        if not mxfp4_fused_supported(values):
            raise ValueError(
                f"fused MXFP4 quantizer needs a CUDA bf16/fp16/fp32 tensor, got {values.dtype} on {values.device}"
            )
    rows, hidden = (int(v) for v in values.shape)
    if codes is None:
        codes = torch.empty(
            (rows, hidden // 2), dtype=torch.uint8, device=values.device
        )
    if sf is None:
        sf = torch.empty(
            (rows, hidden // MXFP4_BLOCK), dtype=torch.uint8, device=values.device
        )
    for name, buf, width in (
        ("codes", codes, hidden // 2),
        ("sf", sf, hidden // MXFP4_BLOCK),
    ):
        if (
            buf.dtype != torch.uint8
            or buf.dim() != 2
            or buf.shape[0] < rows
            or buf.shape[1] != width
            or buf.stride(1) != 1
            or buf.stride(0) != width
        ):
            raise ValueError(
                f"{name} must be a row-contiguous uint8 [>= {rows}, {width}] buffer"
            )
    routing = topk_ids is not None
    fuse_routing = (
        routing
        and _routing_fusable(topk_ids, topk_ids_out, torch.int64)  # type: ignore[arg-type]
        and _routing_fusable(topk_weights, topk_weights_out, torch.float32)  # type: ignore[arg-type]
        and topk_ids.shape == topk_weights.shape  # type: ignore[union-attr]
        and topk_ids_out.shape[0] >= rows  # type: ignore[union-attr]
        and topk_weights_out.shape[0] >= rows  # type: ignore[union-attr]
    )
    if rows:
        top_k = int(topk_ids.shape[1]) if fuse_routing else 1  # type: ignore[union-attr]
        grid = (rows, triton.cdiv(hidden // MXFP4_BLOCK, _BLOCKS))
        _mxfp4_quant_kernel[grid](
            values,
            values.stride(0),
            codes,
            sf,
            hidden,
            topk_ids if fuse_routing else values,
            topk_ids.stride(0) if fuse_routing else 0,  # type: ignore[union-attr]
            topk_weights if fuse_routing else values,
            topk_weights.stride(0) if fuse_routing else 0,  # type: ignore[union-attr]
            topk_ids_out if fuse_routing else codes,
            topk_weights_out if fuse_routing else codes,
            top_k,
            RCP=RCP_6,
            BLOCKS=_BLOCKS,
            COPY_ROUTING=fuse_routing,
            TOPK_P2=triton.next_power_of_2(top_k),
            num_warps=4,
        )
    if routing and not fuse_routing:
        topk_ids_out[:rows].copy_(topk_ids)  # type: ignore[index,union-attr]
        topk_weights_out[:rows].copy_(topk_weights)  # type: ignore[index,union-attr]
    return codes[:rows], sf[:rows]


__all__ = [
    "MXFP4_BLOCK",
    "RCP_6",
    "UE8M0_SCALE_DTYPES",
    "mxfp4_dequantize_torch",
    "mxfp4_fused_supported",
    "mxfp4_quantize_fused",
    "mxfp4_quantize_torch",
]
