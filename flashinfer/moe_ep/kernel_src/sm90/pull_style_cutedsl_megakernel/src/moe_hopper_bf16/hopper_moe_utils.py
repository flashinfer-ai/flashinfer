# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Host-side utility helpers for the BF16 Hopper MoE path."""

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch

# The Hopper MoE package is BF16-only.  ``--kind`` survives as a single-choice
# flag so the shell harnesses keep a stable command line.
BF16_KIND_CHOICES = ("bf16",)

#: Real-valued magnitude of a non-zero synthetic input element.
#:
#: Every operand the GEMM sees has magnitude ``0.25``: it is exactly
#: representable in BF16, so the synthetic inputs stay exact, and it keeps
#: the accumulation regime that the reference tolerances were derived for.
Bf16NonzeroValue = 0.25


def bf16_kind_to_cutlass_dtype(kind: str):
    """Map the supported Hopper kind string to its Cutlass dtype."""
    import cutlass

    return {"bf16": cutlass.BFloat16}[kind]


def bf16_kind_to_torch_dtype(kind: str) -> torch.dtype:
    """Map the supported Hopper kind string to its torch dtype."""
    return {"bf16": torch.bfloat16}[kind]


def create_bf16_tensor(
    shape: Tuple[int, ...],
    *,
    perf_run: bool,
    nonzero_prob: float = 0.20,
    nonzero_value: float = Bf16NonzeroValue,
    positive_prob: Optional[float] = None,
    negative_prob: Optional[float] = None,
    device: Union[str, torch.device] = "cuda",
    generator: Optional[torch.Generator] = None,
    perf_positive_only: bool = False,
) -> torch.Tensor:
    """Create a synthetic BF16 activation / weight payload.

    Correctness mode builds sparse signed data directly in BF16: by default
    ``nonzero_prob`` is split evenly across ``+nonzero_value`` and
    ``-nonzero_value``.  Perf mode fills the buffer with dense uniform values
    in the same magnitude band, which needs no FP32 staging tensor.
    """
    if not 0.0 <= float(nonzero_prob) <= 1.0:
        raise ValueError(f"nonzero_prob must be in [0, 1], got {nonzero_prob}.")
    if positive_prob is None and negative_prob is None:
        pos_prob = float(nonzero_prob) * 0.5
        neg_prob = float(nonzero_prob) * 0.5
    elif positive_prob is not None and negative_prob is not None:
        pos_prob = float(positive_prob)
        neg_prob = float(negative_prob)
    else:
        raise ValueError("positive_prob and negative_prob must be provided together.")
    if pos_prob < 0.0 or neg_prob < 0.0:
        raise ValueError(
            f"positive_prob and negative_prob must be non-negative, got "
            f"{pos_prob} and {neg_prob}."
        )

    magnitude = float(nonzero_value)
    if perf_run:
        out = torch.empty(shape, dtype=torch.bfloat16, device=device)
        low = 0.0 if perf_positive_only else -magnitude
        out.uniform_(low, magnitude, generator=generator)
        return out

    pos_threshold = pos_prob
    neg_threshold = pos_prob + neg_prob
    out = torch.zeros(shape, dtype=torch.bfloat16, device=device)
    rand = torch.rand(shape, device=device, generator=generator)
    out[rand < pos_threshold] = magnitude
    out[(rand >= pos_threshold) & (rand < neg_threshold)] = -magnitude
    return out


def bf16_reference_mm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Reference BF16 GEMM: promote both operands and accumulate in FP32.

    This mirrors the kernel, whose WGMMA reads BF16 operands and accumulates
    into an FP32 accumulator.  Only the summation order differs.
    """
    return a.to(torch.float32) @ b.to(torch.float32)


__all__ = [
    "BF16_KIND_CHOICES",
    "Bf16NonzeroValue",
    "bf16_kind_to_cutlass_dtype",
    "bf16_kind_to_torch_dtype",
    "bf16_reference_mm",
    "create_bf16_tensor",
]
