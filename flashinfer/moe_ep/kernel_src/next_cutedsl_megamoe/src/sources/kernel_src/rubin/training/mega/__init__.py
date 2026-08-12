# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Rubin training MegaMoE (mxfp8) kernels.

Organised into two subpackages:

* ``fwd_glu`` -- the forward fused FC1+SwiGLU+FC2 MoE kernel.
* ``bwd_dglu`` -- the backward fused dfc2+dswiglu+dfc1 MoE kernel.

The forward symbols are re-exported here so existing ``rubin.training.mega``
importers keep resolving after the fwd_glu/bwd_dglu reorg.
"""

from .fwd_glu import (
    Fc2OutputDest,
    GluMxFp8Fc12SchedExtension,
    GluMxfp8Epilogue,
    Sm107MegaMoEMxfp8GluKernel,
    Sm107Mxfp8GluFc12Kernel,
    TensorRole,
)
