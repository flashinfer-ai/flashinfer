# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host-side shared-memory stage fitting for MXFP8 x MXFP4 MegaMoE.

The CuTe layouts are the authority for the byte counts.  This module only
performs the integer budget calculation, which keeps it importable on hosts
without a CuTe DSL installation and makes the policy straightforward to test.
"""

from dataclasses import dataclass


MX_BLOCK_SIZE = 32
SM100_TMEM_CAPACITY_COLUMNS = 512


@dataclass(frozen=True)
class OperandStageBytes:
    """Exact bytes occupied by one A/B/SFA/SFB pipeline stage."""

    weight_a: int
    activation_b: int
    weight_sfa: int
    activation_sfb: int

    def __post_init__(self) -> None:
        for name, value in vars(self).items():
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")

    @property
    def total(self) -> int:
        return self.weight_a + self.activation_b + self.weight_sfa + self.activation_sfb


@dataclass(frozen=True)
class StageFit:
    """Result of fitting operand stages in the per-CTA SMEM budget."""

    stages: int
    bytes_per_stage: int
    available_bytes: int
    fixed_bytes: int

    @property
    def used_bytes(self) -> int:
        return self.fixed_bytes + self.stages * self.bytes_per_stage

    @property
    def remaining_bytes(self) -> int:
        return self.available_bytes - self.used_bytes


def compute_stage_fit(
    *,
    smem_capacity_bytes: int,
    occupancy: int,
    operand_bytes: OperandStageBytes,
    fixed_bytes_per_cta: int,
    minimum_stages: int = 2,
    maximum_stages: int | None = None,
) -> StageFit:
    """Fit A/B/SFA/SFB stages into one CTA's share of shared memory.

    ``fixed_bytes_per_cta`` includes barriers, scheduler state, epilogue
    buffers, and any other shared allocation not multiplied by the mainloop
    stage count.  No alignment estimate is made here: callers must use
    ``cute.size_in_bytes`` on the actual one-stage layouts.
    """

    if smem_capacity_bytes <= 0:
        raise ValueError("smem_capacity_bytes must be positive")
    if occupancy <= 0:
        raise ValueError("occupancy must be positive")
    if fixed_bytes_per_cta < 0:
        raise ValueError("fixed_bytes_per_cta must be non-negative")
    if minimum_stages <= 0:
        raise ValueError("minimum_stages must be positive")
    if maximum_stages is not None and maximum_stages < minimum_stages:
        raise ValueError("maximum_stages must be >= minimum_stages")

    available = smem_capacity_bytes // occupancy
    stage_budget = available - fixed_bytes_per_cta
    stages = max(0, stage_budget // operand_bytes.total)
    if maximum_stages is not None:
        stages = min(stages, maximum_stages)
    if stages < minimum_stages:
        required = fixed_bytes_per_cta + minimum_stages * operand_bytes.total
        raise ValueError(
            "MXFP8 x MXFP4 mainloop does not fit: "
            f"need {required} bytes per CTA for {minimum_stages} stages, "
            f"have {available}"
        )

    return StageFit(
        stages=stages,
        bytes_per_stage=operand_bytes.total,
        available_bytes=available,
        fixed_bytes=fixed_bytes_per_cta,
    )
