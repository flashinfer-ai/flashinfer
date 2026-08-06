# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Shared runner descriptors for the MXFP8 GLU fused fc1+fc2 path."""

from dataclasses import dataclass
from typing import Tuple

from moe_nvfp4_swapab.runner_fc12_common import ImplDesc


@dataclass
class TrainingForwardImplDesc(ImplDesc):
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
    act_func: str = "swiglu"

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.act_func not in ("swiglu", "geglu"):
            raise ValueError(
                f"act_func must be 'swiglu' or 'geglu'; got {self.act_func!r}."
            )

    def __str__(self) -> str:
        base = super().__str__().replace("ImplDesc:", "TrainingForwardImplDesc:", 1)
        return (
            f"{base} generate_c={self.generate_c} "
            f"use_stg_fc1={self.use_stg_fc1} "
            f"act_func={self.act_func}"
        )


@dataclass
class TrainingBackwardImplDesc(ImplDesc):
    """Kernel configuration for MXFP8 GLU training.

    Extends :class:`ImplDesc` with the validated MXFP8 tile/cluster geometry
    and training-oriented defaults.
    """

    mma_tiler_mnk: Tuple[int, int, int] = (256, 256, 128)
    cluster_shape_mnk: Tuple[int, int, int] = (2, 1, 1)
    use_2cta_instrs: bool = True
    use_stg_fc1: bool = False
    # Recompute activation for other backward paths
    dfc2_recompute: bool = False
    # MXFP8 col-quantized grad_y1 alongside the existing row-quant fc1_output
    dfc2_col_output: bool = False
    # GLU activation variant.
    act_func: str = "swiglu"

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.act_func not in ("swiglu", "geglu"):
            raise ValueError(
                f"act_func must be 'swiglu' or 'geglu'; got {self.act_func!r}."
            )

    def __str__(self) -> str:
        base = super().__str__().replace("ImplDesc:", "TrainingBackwardImplDesc:", 1)
        return (
            f"use_stg_fc1={self.use_stg_fc1} "
            f"dfc2_recompute={self.dfc2_recompute} "
            f"dfc2_col_output={self.dfc2_col_output} "
            f"act_func={self.act_func}"
        )
