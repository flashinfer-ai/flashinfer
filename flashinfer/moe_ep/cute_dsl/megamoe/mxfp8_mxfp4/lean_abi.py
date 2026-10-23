# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Communication-free FC12 launch contract.

Expert indices are local weight indices, never global routing IDs. The bridge
owns sorting, route restoration, and FP32 weighting after the BF16 route-term
rounding. This ABI deliberately contains neither scores nor peer addresses.

The input scale plane is *linear* E8M0/K32. Data and scales may have different
per-expert padding; their offsets count rows, not bytes. The launcher owns a
lossless byte permutation to the kernel's atom-swizzled scale layout. It must
not dequantize/requantize inputs. Its private FC1 scale plane has independent
128-row padding. Only valid output rows are defined, in input data-row order.

FC1 weights and their scales use gate8/up8 row interleaving before the
K-major transpose. Canonical weight packs are converted by backend preprocessing;
callers supplying this low-level ABI must prepare the same row permutation.

Metadata stays on-device during launch/capture. ``build_lean_row_plan`` is a
host fixture/planning utility, not a way to read GPU counts in a launch path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True)
class LeanFc12Config:
    local_num_experts: int
    hidden: int
    intermediate: int
    data_row_capacity: int
    scale_row_capacity: int
    data_row_alignment: int = 64
    scale_row_alignment: int = 128

    def __post_init__(self) -> None:
        if self.local_num_experts <= 0:
            raise ValueError("local_num_experts must be positive")
        for name in ("hidden", "intermediate"):
            if getattr(self, name) <= 0 or getattr(self, name) % 128:
                raise ValueError(f"{name} must be a positive multiple of 128")
        for name in ("data_row_alignment", "scale_row_alignment"):
            if getattr(self, name) <= 0 or getattr(self, name) % 32:
                raise ValueError(f"{name} must be a positive multiple of 32")
        for prefix in ("data", "scale"):
            capacity = getattr(self, f"{prefix}_row_capacity")
            alignment = getattr(self, f"{prefix}_row_alignment")
            if capacity < 0 or capacity % alignment:
                raise ValueError(
                    f"{prefix}_row_capacity must be aligned and nonnegative"
                )


@dataclass(frozen=True)
class LeanRowPlan:
    expert_token_sizes: tuple[int, ...]
    expert_data_row_offsets: tuple[int, ...]
    expert_scale_row_offsets: tuple[int, ...]


def build_lean_row_plan(
    expert_token_sizes: Sequence[int], config: LeanFc12Config
) -> LeanRowPlan:
    """Build padded E+1 exclusive prefix sums, preserving empty experts."""
    counts = tuple(expert_token_sizes)
    if len(counts) != config.local_num_experts:
        raise ValueError("expert_token_sizes must have local_num_experts entries")
    if any(type(count) is not int or count < 0 for count in counts):
        raise ValueError("expert_token_sizes must contain nonnegative integers")

    def offsets(alignment: int, capacity: int) -> tuple[int, ...]:
        result = [0]
        for count in counts:
            result.append(result[-1] + (count + alignment - 1) // alignment * alignment)
        if result[-1] > capacity:
            raise ValueError("padded expert rows exceed row capacity")
        if result[-1] > 2**31 - 1:
            raise ValueError("row offsets exceed the int32 ABI")
        return tuple(result)

    return LeanRowPlan(
        counts,
        offsets(config.data_row_alignment, config.data_row_capacity),
        offsets(config.scale_row_alignment, config.scale_row_capacity),
    )


@dataclass(frozen=True)
class LeanFc12Inputs:
    activation: torch.Tensor
    activation_scales: torch.Tensor
    fc1_weight: torch.Tensor
    fc1_weight_scales: torch.Tensor
    fc2_weight: torch.Tensor
    fc2_weight_scales: torch.Tensor
    output: torch.Tensor
    expert_token_sizes: torch.Tensor
    expert_data_row_offsets: torch.Tensor
    expert_scale_row_offsets: torch.Tensor


def validate_lean_inputs(
    config: LeanFc12Config, inputs: LeanFc12Inputs, *, require_cuda: bool = True
) -> None:
    """Check shapes/layouts without reading counts or synchronizing the GPU.

    The producer guarantees int32 counts >=0, offsets starting at zero, aligned
    monotonic spans >=counts, and terminal offsets <=capacity. These content
    invariants must be tested outside graph capture; this validator checks only
    metadata. Launchers must retain buffers until execution completes.
    """
    import torch

    from .packed_weight_views import PackedWeightViews
    from .weight_scale_views import SwizzledWeightScaleViews

    device = inputs.activation.device
    for name, tensor in vars(inputs).items():
        if tensor.device != device or (require_cuda and not tensor.is_cuda):
            raise ValueError(f"{name} must be on the same CUDA device as activation")

    def check(name: str, shape: tuple[int, ...], dtypes: tuple) -> None:
        tensor = getattr(inputs, name)
        if tuple(tensor.shape) != shape or tensor.dtype not in dtypes:
            raise ValueError(f"{name} must have shape {shape} and dtype in {dtypes}")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")

    scale_dtypes = (torch.uint8, getattr(torch, "float8_e8m0fnu", torch.uint8))
    check(
        "activation",
        (config.data_row_capacity, config.hidden),
        (torch.float8_e4m3fn,),
    )
    check(
        "activation_scales",
        (config.scale_row_capacity, config.hidden // 32),
        scale_dtypes,
    )
    check("output", (config.data_row_capacity, config.hidden), (torch.bfloat16,))
    check("expert_token_sizes", (config.local_num_experts,), (torch.int32,))
    for name in ("expert_data_row_offsets", "expert_scale_row_offsets"):
        check(name, (config.local_num_experts + 1,), (torch.int32,))
    for cls, first, second in (
        (PackedWeightViews, inputs.fc1_weight, inputs.fc2_weight),
        (SwizzledWeightScaleViews, inputs.fc1_weight_scales, inputs.fc2_weight_scales),
    ):
        views = cls(
            first,
            second,
            hidden_size=config.hidden,
            intermediate_size=config.intermediate,
        )
        if views.num_experts != config.local_num_experts:
            raise ValueError("weights must contain exactly local_num_experts")


__all__ = [
    "LeanFc12Config",
    "LeanFc12Inputs",
    "LeanRowPlan",
    "build_lean_row_plan",
    "validate_lean_inputs",
]
