# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""cudnn.sdpa.fwd: the SDPA forward opset.

Exports resolve on first attribute access (PEP 562). Eager imports here used to
drag the CuTe DSL — measured ~1.0 s and 357 modules — into any process that
merely asked whether an SDPA engine might serve a graph. Support checks need
the capability tables, not the lowering, so the lowering is not imported until
something decides to build.
"""

import importlib
from typing import Any

_LAZY_EXPORTS = {
    "SdpaFwdDsl": (".api_dsl", "SdpaFwdDsl"),
    "SdpaFwdDslSm100": (".api_dsl", "SdpaFwdDslSm100"),
    "SdpaFwdDslSm120": (".api_dsl", "SdpaFwdDslSm120"),
    "sdpa_fwd_wrapper_dsl_sm100": (".api_dsl", "sdpa_fwd_wrapper_dsl_sm100"),
    "sdpa_fwd_wrapper_dsl_sm120": (".api_dsl", "sdpa_fwd_wrapper_dsl_sm120"),
    "SdpaFwdDslSm90": (".api_dsl", "SdpaFwdDslSm90"),
    "sdpa_fwd_wrapper_dsl_sm90": (".api_dsl", "sdpa_fwd_wrapper_dsl_sm90"),
    "SdpaFwdDslSm80": (".api_dsl", "SdpaFwdDslSm80"),
    "sdpa_fwd_wrapper_sm80": (".api_dsl", "sdpa_fwd_wrapper_sm80"),
}

__all__ = list(_LAZY_EXPORTS)


def __getattr__(name: str) -> Any:
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module, attr = target
    value = getattr(importlib.import_module(module, __name__), attr)
    globals()[name] = value  # resolve once
    return value


def __dir__():
    # Union, not just __all__: returning only the lazy names hid every normal
    # module attribute, __name__ included.
    return sorted(set(globals()) | set(__all__))
