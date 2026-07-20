# SPDX-FileCopyrightText: 2026 FlashInfer team
# SPDX-License-Identifier: Apache-2.0
"""flashinfer.experimental — fast-moving, arch-scoped kernels.

Everything under this namespace is exempt from FlashInfer's stability and
support guarantees: APIs may change or disappear in any release, ops target
non-flagship architectures only (currently SM12x consumer Blackwell), nothing
here is ever compiled ahead-of-time or shipped in the jit-cache/cubin wheels,
and core FlashInfer never imports it.  See flashinfer/experimental/README.md
for the governing rules.
"""

from __future__ import annotations

import importlib
import os
import warnings
from typing import Any

if os.environ.get("FLASHINFER_EXP_QUIET", "").strip().lower() not in {
    "1",
    "true",
    "yes",
    "on",
}:
    warnings.warn(
        "flashinfer.experimental APIs are experimental: no API stability and "
        "no support guarantees; interfaces may change or be removed in any "
        "release. Set FLASHINFER_EXP_QUIET=1 to silence this warning.",
        FutureWarning,
        stacklevel=2,
    )

_ARCH_NAMESPACES = ("sm12x",)


def __getattr__(name: str) -> Any:
    if name in _ARCH_NAMESPACES:
        module = importlib.import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(_ARCH_NAMESPACES)


__all__ = list(_ARCH_NAMESPACES)
