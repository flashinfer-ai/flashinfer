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
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch

__all__ = [
    "Sm90PushWeights",
    "make_sm90_push_weights",
    "transform_weights_for_sm90_push",
]


def _per_block_cast_128x128(w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """(N, K) -> e4m3 (N, K) + f32 scales (N/128, K/128), 128x128 blockwise."""
    N, K = w.shape
    if N % 128 != 0 or K % 128 != 0:
        raise ValueError(f"weight ({N}, {K}) must be 128-aligned")
    t = w.float().reshape(N // 128, 128, K // 128, 128)
    amax = t.abs().amax(dim=(1, 3))
    sc = torch.where(amax > 0, amax / 448.0, torch.ones_like(amax))
    q = (
        (t / sc[:, None, :, None])
        .clamp(-448.0, 448.0)
        .to(torch.float8_e4m3fn)
        .reshape(N, K)
    )
    return q, sc


def transform_weights_for_sm90_push(
    w13: torch.Tensor,
    w2: torch.Tensor,
    weight_format: str = "bf16",
    interleave_gate_up: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert dense BF16 weights into FP8 block-scale layout."""
    if weight_format != "bf16":
        raise ValueError(
            f"unsupported weight_format {weight_format!r}; "
            "SM90 push accepts BF16 checkpoint weights only"
        )
    if w13.dtype != torch.bfloat16 or w2.dtype != torch.bfloat16:
        raise ValueError("w13/w2 must be BF16 checkpoint tensors")
    if w13.ndim != 3 or w2.ndim != 3 or w13.shape[0] != w2.shape[0]:
        raise ValueError("w13/w2 must be (E, 2I, H) / (E, H, I)")
    E, two_i, H = w13.shape
    _, H2, I = w2.shape
    if H2 != H or two_i != 2 * I:
        raise ValueError(
            f"inconsistent weight shapes: w13 {tuple(w13.shape)}, w2 {tuple(w2.shape)}"
        )
    if interleave_gate_up and I % 128 != 0:
        raise ValueError(f"interleave_gate_up needs I % 128 == 0, got I={I}")
    dev = w13.device
    w13_fp8 = torch.empty(E, two_i, H, device=dev, dtype=torch.float8_e4m3fn)
    w13_sf = torch.empty(E, two_i // 128, H // 128, device=dev, dtype=torch.float32)
    w2_fp8 = torch.empty(E, H, I, device=dev, dtype=torch.float8_e4m3fn)
    w2_sf = torch.empty(E, H // 128, I // 128, device=dev, dtype=torch.float32)
    for e in range(E):
        a, b = w13[e], w2[e]
        q, s = _per_block_cast_128x128(a)
        w13_fp8[e].copy_(q)
        w13_sf[e].copy_(s)
        q, s = _per_block_cast_128x128(b)
        w2_fp8[e].copy_(q)
        w2_sf[e].copy_(s)
    if interleave_gate_up:
        nb = I // 128
        w13_fp8 = (
            w13_fp8.reshape(E, 2, nb, 128, H)
            .transpose(1, 2)
            .reshape(E, two_i, H)
            .contiguous()
        )
        w13_sf = (
            w13_sf.reshape(E, 2, nb, H // 128)
            .transpose(1, 2)
            .reshape(E, two_i // 128, H // 128)
            .contiguous()
        )
    return w13_fp8, w13_sf, w2_fp8, w2_sf


@dataclass(frozen=True)
class Sm90PushWeights:
    """Transformed SM90-push weights WITH their layout tag."""

    w13_fp8: torch.Tensor
    w13_sf: torch.Tensor
    w2_fp8: torch.Tensor
    w2_sf: torch.Tensor
    w13_interleaved: bool = False

    def __post_init__(self):
        if (
            self.w13_fp8.dtype != torch.float8_e4m3fn
            or self.w2_fp8.dtype != torch.float8_e4m3fn
        ):
            raise ValueError("Sm90PushWeights payloads must be float8_e4m3fn")
        if self.w13_sf.dtype != torch.float32 or self.w2_sf.dtype != torch.float32:
            raise ValueError("Sm90PushWeights scales must be float32")
        if self.w13_fp8.ndim != 3:
            raise ValueError("w13_fp8 must be (E, 2I, H)")
        two_i = self.w13_fp8.shape[1]
        if two_i <= 0 or two_i % 256 != 0:
            raise ValueError(
                f"w13 second dim (2I) must be a positive multiple of 256, got {two_i}"
            )


def make_sm90_push_weights(
    w13: torch.Tensor,
    w2: torch.Tensor,
    weight_format: str = "bf16",
    interleave_gate_up: bool = False,
) -> Sm90PushWeights:
    """:func:`transform_weights_for_sm90_push` + layout tag, in one step."""
    w13_fp8, w13_sf, w2_fp8, w2_sf = transform_weights_for_sm90_push(
        w13, w2, weight_format=weight_format, interleave_gate_up=interleave_gate_up
    )
    return Sm90PushWeights(
        w13_fp8=w13_fp8,
        w13_sf=w13_sf,
        w2_fp8=w2_fp8,
        w2_sf=w2_sf,
        w13_interleaved=interleave_gate_up,
    )
