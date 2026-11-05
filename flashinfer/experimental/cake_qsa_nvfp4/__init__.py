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

# Experimental Cake backend: QSA sparse paged GQA decode over an NVFP4 KV
# cache on SM80, SM90, SM100, SM103 and SM120 (the mma.sync split-KV decode of
# SM80 / SM90 / SM120 with its whole-tile build for the wide grids of SM80 and
# its cooperative build that merges the splits in the same launch, the tcgen05
# decode of SM100 / SM103 in the same two forms, and the separate split-KV merge with the sigmoid
# output gate).  The public entry point is
# ``flashinfer.sparse.qsa_sparse_paged_attention``: its default
# ``backend="auto"`` selects this package on every exported capability above 9.0 and
# ``backend="cake"`` selects it explicitly; the host plan, workspace, launch
# sequence, registry and generated sources live here.  The host module is imported on first attribute access, so importing
# the package loads neither torch nor the libraries.

from __future__ import annotations

import importlib
from typing import Any

_BACKEND = (
    "SPLIT_OVERRIDES",
    "fused_capacity",
    "is_supported",
    "num_splits",
    "qsa_sparse_paged_attention",
    "qsa_workspace_bytes",
    "reset_workspace",
    "select_form",
    "select_wide",
)
_JIT = ("ARCHES", "ARCH_BY_CAPABILITY", "MAX_SPLITS")

__all__ = [*sorted(_BACKEND), *_JIT]


def __getattr__(name: str) -> Any:
    if name in _BACKEND:
        return getattr(importlib.import_module(f"{__name__}.cake_backend"), name)
    if name in _JIT:
        return getattr(importlib.import_module(f"{__name__}.cake_jit"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
