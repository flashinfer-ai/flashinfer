# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""SM90 signed INT4/K128-BF16 weights with E4M3 activations."""

from dataclasses import dataclass
from typing import ClassVar, Literal

from ..fp8_mxfp4_bf16_pull_cutedsl.config import (
    Sm90_Fp8_Mxfp4_Bf16_PullCutedsl_MegaMoeConfig,
)


@dataclass
class Sm90_Fp8_Int4_Bf16_PullCutedsl_MegaMoeConfig(
    Sm90_Fp8_Mxfp4_Bf16_PullCutedsl_MegaMoeConfig
):
    """Group-128 symmetric INT4, with explicit, cached, or automatic tactics."""

    kernel_name: str = "sm90_fp8_int4_bf16_pull_cutedsl"
    humming_max_range: ClassVar[Literal[11]] = 11  # type: ignore[misc]
    _weight_format: ClassVar[str] = "INT4"
    _kind_contract: ClassVar[str] = "INT4 x FP8"


__all__ = ["Sm90_Fp8_Int4_Bf16_PullCutedsl_MegaMoeConfig"]
