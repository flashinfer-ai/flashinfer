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

"""Fused Triton MXFP8 (e4m3 + UE8M0, K block 32) input quantizer and stager.

Bit-identical to :func:`~.cake_staging.quantize_activations_k32` (the eager torch reference,
~10 elementwise / reduction passes) for every BF16 / FP16 / FP32 input, including zeros,
denormals, infinities and NaNs.  It reads the activation once and writes the e4m3 codes and the
UE8M0 exponents straight into the workspace's staging rows; optionally the same launch copies
the routing rows (``topk_ids`` int64 / ``topk_weights`` float32), so one call stages a step.

The rule the reference implements (and this kernel reproduces operation for operation):

* ``ratio = amax * fl32(1 / 448)`` -- torch evaluates ``tensor / python_scalar`` on CUDA as a
  multiplication by the FP32 reciprocal; a NaN anywhere in the block makes ``amax`` NaN;
* ``e = clamp(exp(ratio) + (mantissa(ratio) != 0), 0, 254)`` (round the ratio up to a power of
  two; NaN / inf blocks get ``e = 254``, i.e. the inverse scale ``0``);
* ``codes = e4m3(x * 2^(127 - e))`` with torch's e4m3fn conversion: round to nearest even,
  ``|v| > 464`` and NaN become the NaN code ``0x7f | sign`` (no saturation).  The product is a
  single FP32 multiplication (no contraction possible), the conversion is the hardware
  ``cvt.rn.satfinite`` plus the NaN fix-up above.
"""

from __future__ import annotations

import torch

try:  # Triton ships with PyTorch CUDA builds; the torch reference is the fallback
    import triton
    import triton.language as tl

    _HAVE_TRITON = True
except ImportError:  # pragma: no cover
    _HAVE_TRITON = False

K_BLOCK = 32
# fl32(1 / 448) (== np.float32(1) / np.float32(448)): what torch multiplies by for ``amax / 448.0``
_RCP_448 = 0.002232142863795161

if _HAVE_TRITON:

    @triton.jit
    def _mxfp8_quant_kernel(
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
        e = tl.where(has_nan > 0, 254, tl.minimum(tl.maximum(e, 0), 254))
        inv = ((254 - e) << 23).to(tl.float32, bitcast=True)
        y = x * inv[:, None]
        yb = y.to(tl.int32, bitcast=True)
        sign = (yb >> 24) & 0x80
        mag = yb & 0x7FFFFFFF
        hw = y.to(tl.float8e4nv).to(tl.uint8, bitcast=True).to(tl.int32)
        # 0x43E80000 = 464.0: above it (incl. inf / NaN) torch's e4m3fn conversion yields NaN
        code = tl.where(mag > 0x43E80000, sign | 0x7F, hw)
        tl.store(codes_ptr + row * hidden + col, code.to(tl.uint8), mask=m2)
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
_BLOCKS = 16  # 512 values per program


def mxfp8_fused_supported(x: torch.Tensor) -> bool:
    return (
        _HAVE_TRITON
        and x.is_cuda
        and x.dim() == 2
        and x.dtype in _DTYPES
        and x.shape[1] % K_BLOCK == 0
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


def mxfp8_quantize_fused(
    x: torch.Tensor,
    *,
    codes: torch.Tensor | None = None,
    sf: torch.Tensor | None = None,
    topk_ids: torch.Tensor | None = None,
    topk_weights: torch.Tensor | None = None,
    topk_ids_out: torch.Tensor | None = None,
    topk_weights_out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``[rows, H]`` BF16/FP16/FP32 CUDA tensor -> (e4m3 codes ``uint8 [rows, H]``, UE8M0 ``uint8 [rows, H // 32]``).

    ``codes`` / ``sf`` optionally receive the result in place (leading ``rows`` rows of larger,
    row-contiguous ``uint8`` buffers).  With all four routing tensors given, the routing rows
    are copied by the same launch (``topk_ids`` / ``topk_weights`` into the leading ``rows``
    rows of the contiguous ``*_out`` buffers; falls back to ``copy_`` for other layouts).
    One launch, no host synchronization; stream-ordered and CUDA-graph capturable.
    """

    if not mxfp8_fused_supported(x):
        raise ValueError(
            f"fused MXFP8 quantizer needs a CUDA bf16/fp16/fp32 [rows, H] tensor with H % 32 == 0 "
            f"and a unit inner stride, got {tuple(x.shape)} {x.dtype} on {x.device}"
        )
    rows, hidden = (int(v) for v in x.shape)
    if codes is None:
        codes = torch.empty((rows, hidden), dtype=torch.uint8, device=x.device)
    if sf is None:
        sf = torch.empty((rows, hidden // K_BLOCK), dtype=torch.uint8, device=x.device)
    for name, buf, width in (("codes", codes, hidden), ("sf", sf, hidden // K_BLOCK)):
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
        grid = (rows, triton.cdiv(hidden // K_BLOCK, _BLOCKS))
        _mxfp8_quant_kernel[grid](
            x,
            x.stride(0),
            codes,
            sf,
            hidden,
            topk_ids if fuse_routing else x,
            topk_ids.stride(0) if fuse_routing else 0,  # type: ignore[union-attr]
            topk_weights if fuse_routing else x,
            topk_weights.stride(0) if fuse_routing else 0,  # type: ignore[union-attr]
            topk_ids_out if fuse_routing else codes,
            topk_weights_out if fuse_routing else codes,
            top_k,
            RCP=_RCP_448,
            BLOCKS=_BLOCKS,
            COPY_ROUTING=fuse_routing,
            TOPK_P2=triton.next_power_of_2(top_k),
        )
    if routing and not fuse_routing:
        topk_ids_out[:rows].copy_(topk_ids)  # type: ignore[index,union-attr]
        topk_weights_out[:rows].copy_(topk_weights)  # type: ignore[index,union-attr]
    return codes[:rows], sf[:rows]


__all__ = ["mxfp8_fused_supported", "mxfp8_quantize_fused"]
