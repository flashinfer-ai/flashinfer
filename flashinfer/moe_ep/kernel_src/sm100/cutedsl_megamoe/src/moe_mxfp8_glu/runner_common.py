# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Shared runner descriptors for the MXFP8 GLU fused fc1+fc2 path."""

from dataclasses import dataclass
from typing import Tuple

from moe_nvfp4_swapab.runner_fc12_common import ImplDesc


@dataclass
class TrainingImplDesc(ImplDesc):
    """Kernel configuration for MXFP8 GLU training.

    Extends :class:`ImplDesc` with the validated MXFP8 tile/cluster geometry
    and training-oriented defaults.  ``generate_c`` defaults to ``True`` so the
    kernel retains the pre-SwiGLU fc1 gate+up activations needed for backward.
    """

    mma_tiler_mnk: Tuple[int, int, int] = (256, 256, 128)
    cluster_shape_mnk: Tuple[int, int, int] = (2, 1, 1)
    use_2cta_instrs: bool = True
    generate_c: bool = True
    use_stg_fc1: bool = False

    def __str__(self) -> str:
        base = super().__str__().replace("ImplDesc:", "TrainingImplDesc:", 1)
        return f"{base} generate_c={self.generate_c} use_stg_fc1={self.use_stg_fc1}"
