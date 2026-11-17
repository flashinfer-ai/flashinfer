"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# Experimental Cake backend: the MLA value projection fused with the sigmoid
# output gate on SM100 / SM103 -- out = bf16(bf16(latent @ w_uv) * sigmoid(gate))
# per head, the sigmoid evaluated in FP32, padded rows written as zero.  The
# public entry point is ``flashinfer.cake_mla_value_gate.prepare_mla_value_gate``
# (there is no automatic selection); the host checks, the runner, the registry
# and loader glue and the generated sources live here.  The host module is
# imported on first attribute access, so importing the package loads neither
# torch nor the libraries.

from __future__ import annotations

import importlib
from typing import Any

_BACKEND = (
    "LATENT_DIM",
    "MAX_ROWS",
    "SIGMOID_PRECISIONS",
    "MlaValueGateRunner",
    "prepare_mla_value_gate",
)
_JIT = ("ARCHES", "ARCH_BY_CAPABILITY", "MODULES", "VALUE_DIM")

__all__ = [*sorted(_BACKEND), *_JIT]


def __getattr__(name: str) -> Any:
    if name in _BACKEND:
        return getattr(importlib.import_module(f"{__name__}.cake_backend"), name)
    if name in _JIT:
        return getattr(importlib.import_module(f"{__name__}.cake_jit"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
