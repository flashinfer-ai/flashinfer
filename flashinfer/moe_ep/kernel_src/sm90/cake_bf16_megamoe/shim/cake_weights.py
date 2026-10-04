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

import torch

__all__ = [
    "GATE_UP_GROUP",
    "Sm90CakeBf16Weights",
    "interleave_gate_up",
    "make_sm90_cake_bf16_weights",
]

# The Cake FC1 kernel consumes gate/up rows in alternating groups of this many
# rows so one 256-column WGMMA tile holds matching gate and up columns and the
# SwiGLU epilogue never crosses tiles.  Must match the generator's
# ``GATE_UP_GROUP`` (Cake ``sm90_bf16_moe_grouped_gemm``).
GATE_UP_GROUP = 32


def interleave_gate_up(
    w13: torch.Tensor, intermediate: int, group: int = GATE_UP_GROUP
) -> torch.Tensor:
    """Reorder ``w13`` rows ``[gate(I) | up(I)]`` into ``group``-row gate/up pairs."""
    num_experts, two_i, hidden = w13.shape
    if two_i != 2 * intermediate:
        raise ValueError(
            f"w13 second dim must be 2 * intermediate = {2 * intermediate}, got {two_i}"
        )
    if intermediate % group != 0:
        raise ValueError(
            f"intermediate_size must be a multiple of {group}, got {intermediate}"
        )
    return (
        w13.view(num_experts, 2, intermediate // group, group, hidden)
        .transpose(1, 2)
        .reshape(num_experts, two_i, hidden)
        .contiguous()
    )


@dataclass(frozen=True)
class Sm90CakeBf16Weights:
    """Kernel-ready BF16 expert weights for ``sm90_bf16_bf16_bf16_push_cake``.

    ``w13``: ``[E, 2I, H]`` bf16 with gate/up rows interleaved in
    :data:`GATE_UP_GROUP`-row groups; ``w2``: ``[E, H, I]`` bf16.  Both are
    contiguous CUDA tensors on the same device.  No scales: the kernels read
    bf16 operands directly and accumulate in fp32.
    """

    w13: torch.Tensor
    w2: torch.Tensor
    num_local_experts: int
    hidden_size: int
    intermediate_size: int
    w13_interleaved: bool = True

    def __post_init__(self) -> None:
        if self.w13.dtype != torch.bfloat16 or self.w2.dtype != torch.bfloat16:
            raise ValueError(
                f"Sm90CakeBf16Weights expects bf16 tensors, got w13={self.w13.dtype}, w2={self.w2.dtype}"
            )
        e, h, i = self.num_local_experts, self.hidden_size, self.intermediate_size
        if tuple(self.w13.shape) != (e, 2 * i, h):
            raise ValueError(
                f"w13 must be ({e}, {2 * i}, {h}), got {tuple(self.w13.shape)}"
            )
        if tuple(self.w2.shape) != (e, h, i):
            raise ValueError(f"w2 must be ({e}, {h}, {i}), got {tuple(self.w2.shape)}")
        if (
            not (self.w13.is_cuda and self.w2.is_cuda)
            or self.w13.device != self.w2.device
        ):
            raise ValueError("w13 and w2 must be CUDA tensors on one device")
        if not (self.w13.is_contiguous() and self.w2.is_contiguous()):
            raise ValueError("w13 and w2 must be contiguous")
        if not self.w13_interleaved:
            raise ValueError("the Cake FC1 kernel requires gate/up-interleaved w13")

    @property
    def device(self) -> torch.device:
        return self.w13.device


def make_sm90_cake_bf16_weights(
    w13: torch.Tensor, w2: torch.Tensor
) -> Sm90CakeBf16Weights:
    """Transform canonical ``[E, 2I, H]`` / ``[E, H, I]`` bf16 weights for the kernels.

    The only transformation is the gate/up row interleave of ``w13``; ``w2`` is
    used as-is (a contiguous copy is made only when needed).
    """
    if w13.ndim != 3 or w2.ndim != 3:
        raise ValueError("w13 and w2 must be rank-3 [E, rows, cols] tensors")
    num_experts, two_i, hidden = w13.shape
    if two_i % 2 != 0:
        raise ValueError(f"w13 second dim must be even, got {two_i}")
    intermediate = two_i // 2
    if tuple(w2.shape) != (num_experts, hidden, intermediate):
        raise ValueError(
            f"w2 must be ({num_experts}, {hidden}, {intermediate}) to match w13, got {tuple(w2.shape)}"
        )
    if w13.dtype != torch.bfloat16 or w2.dtype != torch.bfloat16:
        raise ValueError(
            "w13 and w2 must be bf16 (native BF16 backend; no requantization)"
        )
    return Sm90CakeBf16Weights(
        w13=interleave_gate_up(w13, intermediate),
        w2=w2.contiguous(),
        num_local_experts=num_experts,
        hidden_size=hidden,
        intermediate_size=intermediate,
    )
