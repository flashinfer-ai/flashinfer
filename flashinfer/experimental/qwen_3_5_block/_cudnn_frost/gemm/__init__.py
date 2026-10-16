# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""cudnn.gemm: the dense / grouped / fused GEMM operation family.

Layout::

    cudnn.gemm.cutedsl.dense.<fusion>            dense GEMM + fused epilogue
    cudnn.gemm.cutedsl.grouped.<fusion>          MoE grouped GEMM + fused epilogue
    cudnn.gemm.cutedsl.discrete_grouped.<fusion> discrete-weight grouped GEMM
    cudnn.gemm.ops                               backend-independent op contracts
    cudnn.gemm.reference                         pure-PyTorch correctness engine

Every public symbol is also re-exported at the top level (``cudnn.<symbol>``),
which is the supported entry point for users.
"""

from typing import Any

_LAZY_EXPORTS = {
    "MhcProjectionBackward": ("cudnn.gemm.mhc_projection_bwd", "MhcProjectionBackward"),
    "mhc_projection_backward": ("cudnn.gemm.mhc_projection_bwd", "mhc_projection_backward"),
    "moe_grouped_matmul": ("cudnn.gemm.ops", "moe_grouped_matmul"),
    "situ_mlp": ("cudnn.gemm.ops", "situ_mlp"),
    "swiglu_mlp": ("cudnn.gemm.ops", "swiglu_mlp"),
    "swiglu_moe": ("cudnn.gemm.ops", "swiglu_moe"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attr_name = _LAZY_EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    import importlib

    value = getattr(importlib.import_module(module_name), attr_name)
    globals()[name] = value
    return value


__all__ = list(_LAZY_EXPORTS)
