# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native CuTeDSL MXFP8-activation/MXFP4-weight MegaMoE backend package."""

from .backend import Mxfp8Mxfp4CutedslMegaKernelBackend
from .config import (
    KERNEL_NAME,
    Mxfp8Mxfp4Tactic,
    Sm100_Mxfp8_Mxfp4_Bf16_Cutedsl_MegaMoeConfig,
    candidate_tactics,
    default_tactic,
    resolve_tactic,
    validate_tactic,
)
from .weights import TransformedMegaWeights, preprocess_mega_weights

__all__ = [
    "KERNEL_NAME",
    "Mxfp8Mxfp4CutedslMegaKernelBackend",
    "Mxfp8Mxfp4Tactic",
    "Sm100_Mxfp8_Mxfp4_Bf16_Cutedsl_MegaMoeConfig",
    "TransformedMegaWeights",
    "candidate_tactics",
    "default_tactic",
    "preprocess_mega_weights",
    "resolve_tactic",
    "validate_tactic",
]
