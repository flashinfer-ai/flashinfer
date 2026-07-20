# SPDX-FileCopyrightText: 2026 FlashInfer team
# SPDX-License-Identifier: Apache-2.0
"""Public surface for quantization.nvfp4 (docs in the op ``__init__``)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from ..._lib.gating import default_is_supported
from ._impl import (
    BF16ToFP4TMAOutputs as Outputs,
)
from ._impl import (
    allocate_bf16_to_fp4_tma_outputs as _allocate,
)
from ._impl import (
    compile_bf16_to_fp4_tma as _compile,
)
from . import META


@dataclass(frozen=True)
class Plan:
    """A compiled (m, k) shape; produced by :func:`plan`."""

    m: int
    k: int
    launch: Callable[..., None]


def plan(m: int, k: int) -> Plan:
    """Compile the quantizer for (m, k); host-side, cached per shape."""
    return Plan(m=int(m), k=int(k), launch=_compile(int(m), int(k)))


def allocate_outputs(plan: Plan, *, device: torch.device | str = "cuda") -> Outputs:
    """Allocate the packed-FP4 + MMA-layout-scale output pair for a plan."""
    return _allocate(plan.m, plan.k, device=torch.device(device))


def run(
    *,
    plan: Plan,
    x: torch.Tensor,
    global_scale: torch.Tensor,
    outputs: Outputs,
) -> None:
    """Quantize ``x`` into ``outputs`` (allocation-free, capture safe)."""
    plan.launch(x, global_scale, outputs.packed_a_flat, outputs.scale_flat)


def is_supported(device=None) -> bool:
    """True on SM120/SM121 with nvidia-cutlass-dsl >= 4.6.0."""
    return default_is_supported(device, requires=META.requires)


__all__ = ["Outputs", "Plan", "plan", "allocate_outputs", "run", "is_supported"]
