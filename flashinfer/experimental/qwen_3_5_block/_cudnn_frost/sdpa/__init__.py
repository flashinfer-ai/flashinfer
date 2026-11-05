# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""cudnn.sdpa: scaled dot-product attention (forward and backward).

This package level is reserved for the ARCH-AGNOSTIC standalone entry points
(a single ``sdpa_fwd_wrapper`` / ``sdpa_bwd_wrapper`` that resolves the
adapter from the device, mirroring how the other FE OSS families expose one
entry point). Until those land it deliberately exports nothing.

The per-architecture adapters and wrappers — the pinning / benchmarking tier —
live one level down and are imported from there directly:

    from cudnn.sdpa.fwd import SdpaFwdDslSm100, SdpaFwdDslSm120, SdpaFwdDslSm80
    from cudnn.sdpa.fwd import sdpa_fwd_wrapper_dsl_sm100, sdpa_fwd_wrapper_dsl_sm120, sdpa_fwd_wrapper_sm80
    from cudnn.sdpa.bwd import SdpaBwdDslSm120, SdpaBwdDslSm80
    from cudnn.sdpa.bwd import sdpa_bwd_wrapper_dsl_sm120, sdpa_bwd_wrapper_sm80

Submodule imports stay lazy there (PEP 562): eager imports used to drag the
CuTe DSL — measured ~1.0 s and 357 modules — into any process that merely
asked whether an SDPA engine might serve a graph. Support checks need the
capability tables, not the lowering, so the lowering is not imported until
something decides to build.
"""

from typing import Any

_LAZY_EXPORTS: dict = {}

__all__ = list(_LAZY_EXPORTS)


def __getattr__(name: str) -> Any:
    raise AttributeError(
        f"module {__name__!r} has no attribute {name!r} — the per-arch SDPA APIs are imported from "
        f"cudnn.sdpa.fwd / cudnn.sdpa.bwd; this level is reserved for the arch-agnostic wrappers"
    )


def __dir__():
    return sorted(set(globals()) | set(__all__))
