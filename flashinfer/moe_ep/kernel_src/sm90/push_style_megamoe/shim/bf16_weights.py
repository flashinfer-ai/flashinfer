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

Typed BF16 weights for the SM90 push MegaMoE runner.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Sm90PushBf16Weights:
    """Resident row-major BF16 weights for FC1 and FC2."""

    w13: torch.Tensor
    w2: torch.Tensor

    def __post_init__(self) -> None:
        if self.w13.ndim != 3 or self.w2.ndim != 3:
            raise ValueError("SM90 push BF16 weights must be rank-3 tensors")
        if self.w13.dtype != torch.bfloat16 or self.w2.dtype != torch.bfloat16:
            raise ValueError("SM90 push BF16 weights must have dtype torch.bfloat16")
        if not self.w13.is_cuda or not self.w2.is_cuda:
            raise ValueError("SM90 push BF16 weights must be CUDA tensors")
        if self.w13.device != self.w2.device:
            raise ValueError("SM90 push BF16 weights must share a CUDA device")
        if not self.w13.is_contiguous() or not self.w2.is_contiguous():
            raise ValueError("SM90 push BF16 weights must be contiguous")
        experts, two_intermediate, hidden = self.w13.shape
        if experts <= 0 or hidden <= 0 or two_intermediate <= 0:
            raise ValueError("SM90 push BF16 weight dimensions must be positive")
        if two_intermediate % 2:
            raise ValueError("SM90 push BF16 w13 output dimension must be even")
        intermediate = two_intermediate // 2
        if tuple(self.w2.shape) != (experts, hidden, intermediate):
            raise ValueError(
                "SM90 push BF16 w2 must have shape "
                f"({experts}, {hidden}, {intermediate})"
            )


def make_sm90_push_bf16_weights(
    w13: torch.Tensor,
    w2: torch.Tensor,
) -> Sm90PushBf16Weights:
    """Bind canonical contiguous BF16 expert weights without requantization."""

    return Sm90PushBf16Weights(w13=w13, w2=w2)


__all__ = ["Sm90PushBf16Weights", "make_sm90_push_bf16_weights"]
