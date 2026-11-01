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

# Deprecated alias of ``flashinfer.experimental.cake_moe_grouped_det_gemm``.
#
# The ragged BF16 MoE grouped GEMM package was renamed to
# ``cake_moe_grouped_det_gemm`` (the deterministic-family name).  This package
# keeps the old import path working: an attribute (a name of the package,
# before or after the rename, e.g. ``grouped_gemm_fwd`` or
# ``grouped_det_gemm_fwd``) resolves on first use to the very same object of
# the new package, and each submodule below (``cake_backend``,
# ``cake_autograd``, ``cake_jit``, ``cake_build``) is the new package's module
# itself.  Importing this package imports nothing else.  New code should import
# ``flashinfer.experimental.cake_moe_grouped_det_gemm``.

from __future__ import annotations

import importlib
from typing import Any

_TARGET = "flashinfer.experimental.cake_moe_grouped_det_gemm"
_SUBMODULES = frozenset({"cake_autograd", "cake_backend", "cake_build", "cake_jit"})


def __getattr__(name: str) -> Any:
    if name in _SUBMODULES:
        return importlib.import_module(f"{__name__}.{name}")
    if name.startswith("__"):
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(_TARGET), name)
