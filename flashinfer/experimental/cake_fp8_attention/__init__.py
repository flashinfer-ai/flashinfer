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

# Experimental Cake backend: FP8 attention on SM90 (dense, noncausal, D = 128,
# BF16 in / BF16 out over logical ``[B, H, S, 128]`` views with any strides):
# a four-program chain -- the statistics partials, their finalize, the
# quantizer (INT8 Q and K, E4M3 V) and the quantized attention -- launched
# under programmatic dependent launch from a caller-owned workspace.  The
# public entry points are ``flashinfer.prefill.prepare_fp8_attention`` and
# ``flashinfer.prefill.fp8_attention_workspace_bytes`` with ``backend="cake"``
# (there is no automatic selection); the host plan, workspace, launch
# sequence, registry and generated sources live here.  The host module is
# imported on first attribute access, so importing the package loads neither
# torch nor the libraries.

from __future__ import annotations

import importlib
from typing import Any

_BACKEND = (
    "ALIGN",
    "QK_FORMAT",
    "RECIPE_ID",
    "STRIDE_ALIGN",
    "FP8AttentionRunner",
    "WorkspaceLayout",
    "fp8_attention_workspace_bytes",
    "prepare_fp8_attention",
    "workspace_layout",
)
_JIT = ("ARCHES", "ARCH_BY_CAPABILITY", "HEAD_DIM", "MODULES", "STAGES")

__all__ = [*sorted(_BACKEND), *_JIT]


def __getattr__(name: str) -> Any:
    if name in _BACKEND:
        return getattr(importlib.import_module(f"{__name__}.cake_backend"), name)
    if name in _JIT:
        return getattr(importlib.import_module(f"{__name__}.cake_jit"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
