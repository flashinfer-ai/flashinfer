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

"""Weight preprocessing for the SM120 MXFP4 x MXFP8 decode MegaMoE backend.

Canonical input (same contract as the sibling ``mxfp4_mxfp8_bf16_cutedsl`` and
``mxfp4_mxfp8_bf16_cake`` backends): a
:class:`~flashinfer.moe_ep.weights.PrequantizedMoEWeights` with

* ``w13``: packed E2M1 ``[local_E, 2 * I, H // 2]`` (gate rows then up rows,
  low nibble first), ``uint8`` / ``int8`` / ``float4_e2m1fn_x2``;
* ``w2``: packed E2M1 ``[local_E, H, I // 2]``;
* ``w13_scale``: UE8M0 exponent bytes ``[local_E, 2 * I, H // 32]``;
* ``w2_scale``: UE8M0 exponent bytes ``[local_E, H, I // 32]``.

Kernel layout, per expert one ``uint8`` row: the packed weights in the
``m16n8k32`` lane-fragment order, then the scales. The kernel loads weights
global -> registers without shared memory: per 16-row block and K64 step, 32
lanes x 16 bytes, lane ``(g, c)`` holding rows ``g`` and ``g + 8`` at the K
offsets its A fragments read. Scales: per 16-row block and K128, 16 rows x 4
bytes. ``w13`` rows are interleaved per 16-row block (gate rows ``8b..8b+7``,
then up rows ``8b..8b+7``) so each SwiGLU pair lands in one thread.
"""

from __future__ import annotations

import torch

from ......weights import MoEWeightPack, PrequantizedMoEWeights

FP4_PACKED_DTYPES = (torch.uint8, torch.int8, torch.float4_e2m1fn_x2)
SCALE_DTYPES = (torch.uint8, torch.float8_e8m0fnu)


def weight_strides(hidden: int, intermediate: int) -> tuple[int, int]:
    """Bytes per expert of the kernel's ``w13`` / ``w2`` rows: packed FP4, then one UE8M0 byte per 32 values."""
    return (
        2 * intermediate * hidden // 2 + 2 * intermediate * hidden // 32,
        hidden * intermediate // 2 + hidden * intermediate // 32,
    )


def _codes(packed: torch.Tensor) -> torch.Tensor:
    """[N, K/2] packed E2M1 (element 2j in the low nibble of byte j) -> [N, K] codes."""
    return torch.stack((packed & 15, packed >> 4), dim=-1).flatten(-2)


def _fragments(codes: torch.Tensor) -> torch.Tensor:
    """[N, K] codes -> the lane-fragment layout (see the module docs), flattened."""
    n, k = codes.shape
    x = codes.reshape(
        n // 16, 2, 8, k // 64, 2, 2, 4, 4
    )  # [blk, half, g, k64, step, pair, c, j]
    b = x[:, 0] | (x[:, 1] << 4)
    return b.permute(0, 2, 1, 5, 3, 4, 6).reshape(-1)  # [blk, k64, g, c, step, pair, j]


def _scales(sf: torch.Tensor) -> torch.Tensor:
    """[N, K/32] UE8M0 -> per 16-row block and K128: 16 rows x 4 bytes (one word per row), flattened."""
    n, ks = sf.shape
    return sf.reshape(n // 16, 16, ks // 4, 4).permute(0, 2, 1, 3).reshape(-1)


def _w13_rows(intermediate: int, device: torch.device) -> torch.Tensor:
    """Kernel row -> checkpoint row of ``w13``: per 16-row block, gate rows 8b..8b+7 then up rows 8b..8b+7."""
    r = torch.arange(2 * intermediate, device=device)
    blk, w = r // 16, r % 16
    return torch.where(w < 8, blk * 8 + w, intermediate + blk * 8 + (w - 8))


def _transform(packed: torch.Tensor, scales: torch.Tensor, gated: bool) -> torch.Tensor:
    e, n, half_k = packed.shape
    out = torch.empty(
        (e, n * half_k + n * half_k // 16), dtype=torch.uint8, device=packed.device
    )
    rows = _w13_rows(n // 2, packed.device) if gated else None
    for expert in range(e):  # one expert at a time bounds the temporaries
        codes, sf = _codes(packed[expert]), scales[expert]
        if rows is not None:
            codes, sf = codes[rows], sf[rows]
        out[expert, : n * half_k] = _fragments(codes)
        out[expert, n * half_k :] = _scales(sf)
    return out


def _as_bytes(
    tensor: torch.Tensor, dtypes: tuple[torch.dtype, ...], name: str
) -> torch.Tensor:
    if tensor.dtype not in dtypes or not tensor.is_cuda:
        raise ValueError(
            f"{name} must be a CUDA tensor of dtype {dtypes}, got {tensor.dtype}"
        )
    return tensor.contiguous().view(torch.uint8)


def preprocess_mega_weights(
    weights: MoEWeightPack,
    *,
    hidden_size: int,
    intermediate_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``PrequantizedMoEWeights`` -> (``w13``, ``w2``) in the kernel layout, ``uint8 [local_E, stride]``."""
    if not isinstance(weights, PrequantizedMoEWeights):
        raise NotImplementedError(
            "SM120 MXFP4 x MXFP8 requires pre-quantized packed E2M1 weights "
            "and both UE8M0 scale planes"
        )
    w13 = _as_bytes(weights.w13, FP4_PACKED_DTYPES, "w13")
    w2 = _as_bytes(weights.w2, FP4_PACKED_DTYPES, "w2")
    s13 = _as_bytes(weights.w13_scale, SCALE_DTYPES, "w13_scale")
    s2 = _as_bytes(weights.w2_scale, SCALE_DTYPES, "w2_scale")
    h, i = hidden_size, intermediate_size
    e = w13.shape[0]
    for name, tensor, shape in (
        ("w13", w13, (e, 2 * i, h // 2)),
        ("w2", w2, (e, h, i // 2)),
        ("w13_scale", s13, (e, 2 * i, h // 32)),
        ("w2_scale", s2, (e, h, i // 32)),
    ):
        if tuple(tensor.shape) != shape:
            raise ValueError(
                f"{name} must have shape {shape}, got {tuple(tensor.shape)}"
            )
    return _transform(w13, s13, True), _transform(w2, s2, False)


def validate_transformed_mega_weights(
    transformed_weights: tuple[torch.Tensor, torch.Tensor],
    *,
    hidden_size: int,
    intermediate_size: int,
    local_experts: int,
) -> None:
    try:
        w13, w2 = transformed_weights
    except (TypeError, ValueError) as exc:
        raise ValueError("transformed weights must be (w13, w2)") from exc
    for name, tensor, stride in zip(
        ("w13", "w2"),
        (w13, w2),
        weight_strides(hidden_size, intermediate_size),
        strict=True,
    ):
        if (
            tuple(tensor.shape) != (local_experts, stride)
            or tensor.dtype != torch.uint8
            or not tensor.is_cuda
            or not tensor.is_contiguous()
        ):
            raise ValueError(
                f"{name} must be a contiguous CUDA uint8 tensor of shape "
                f"{(local_experts, stride)}, got {tensor.dtype} {tuple(tensor.shape)}"
            )


__all__ = [
    "preprocess_mega_weights",
    "validate_transformed_mega_weights",
    "weight_strides",
]
