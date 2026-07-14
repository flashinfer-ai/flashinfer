# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""CuTeDSL MegaMoE thin frontend API (shipped with FlashInfer ``moe_ep``).

Library use (NVFP4)::

    from flashinfer.moe_ep.backends.mega.kernel.cutedsl_backend_kernels.frontend import (
        create_dummy_inputs,
        get_symm_buffer_for_mega_moe,
        init_dist,
        nvfp4_mega_moe,
    )

Library use (MXFP8)::

    from flashinfer.moe_ep.backends.mega.kernel.cutedsl_backend_kernels.frontend import (
        create_dummy_mxfp8_inputs,
        get_symm_buffer_for_mxfp8_mega_moe,
        init_dist,
        mxfp8_mega_moe,
    )

Symm-buffer helpers stage activations, routing, and combine buffers.  Expert
weights are passed separately to ``nvfp4_mega_moe`` / ``mxfp8_mega_moe`` as
kernel-ready ``(weight, scale)`` tuples.
"""

from __future__ import annotations

from .._bootstrap_paths import bootstrap_paths

bootstrap_paths()

from .mxfp8 import (
    MegaMoEMxfp8SymmBuffer,
    create_dummy_inputs as create_dummy_mxfp8_inputs,
    get_symm_buffer_for_mxfp8_mega_moe,
    mxfp8_mega_moe,
)
from .nvfp4 import (
    MegaMoESymmBuffer,
    TransformedWeights,
    create_dummy_inputs,
    get_symm_buffer_for_mega_moe,
    init_dist,
    make_dummy_epilogue_params,
    nvfp4_mega_moe,
)

__all__ = [
    "MegaMoEMxfp8SymmBuffer",
    "MegaMoESymmBuffer",
    "TransformedWeights",
    "create_dummy_inputs",
    "create_dummy_mxfp8_inputs",
    "get_symm_buffer_for_mega_moe",
    "get_symm_buffer_for_mxfp8_mega_moe",
    "init_dist",
    "make_dummy_epilogue_params",
    "mxfp8_mega_moe",
    "nvfp4_mega_moe",
]
