# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""cudnn.sdpa.bwd: the SDPA backward opset.

Exports resolve on first attribute access (PEP 562). Eager imports here used to
drag the CuTe DSL — measured ~1.0 s and 357 modules — into any process that
merely asked whether an SDPA engine might serve a graph. Support checks need
the capability tables, not the lowering, so the lowering is not imported until
something decides to build.
"""

import importlib
from typing import Any

_LAZY_EXPORTS = {
    "CompactGqaBackward": (".compact_gqa", "CompactGqaBackward"),
    "compact_gqa_backward": (".compact_gqa", "compact_gqa_backward"),
    "SdpaBwdDsl": (".api_dsl", "SdpaBwdDsl"),
    "SdpaBwdDslSm120": (".api_dsl", "SdpaBwdDslSm120"),
    "sdpa_bwd_wrapper_dsl_sm120": (".api_dsl", "sdpa_bwd_wrapper_dsl_sm120"),
    "SdpaBwdDslSm80": (".api_dsl", "SdpaBwdDslSm80"),
    "sdpa_bwd_wrapper_sm80": (".api_dsl", "sdpa_bwd_wrapper_sm80"),
    "SdpaBwdDslSm107": (".api_dsl_sm107", "SdpaBwdDslSm107"),
    "SdpaBwdDslSm107Fp8": (".api_dsl_sm107", "SdpaBwdDslSm107Fp8"),
    "SdpaBwdDslSm100D256": (".api_dsl_sm100_d256", "SdpaBwdDslSm100D256"),
    "Nvfp4AttentionQatBackward": (".qat", "Nvfp4AttentionQatBackward"),
    "nvfp4_attention_qat_backward": (".qat", "nvfp4_attention_qat_backward"),
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
