# SPDX-FileCopyrightText: 2026 FlashInfer team
# SPDX-License-Identifier: Apache-2.0
"""Attention ops for flashinfer.experimental.sm12x.

- ``paged``: paged-KV self-attention (decode + extend, FP8 KV, MSA
  block-sparse variant) with on-device graph-replay metadata staging.
- ``sparse_mla``: top-k-selected MLA decode/extend (DeepSeek-V3.2 / GLM NSA).
- ``compressed_mla``: MLA decode directly from compressed KV pages (DSV4).
- ``nsa_indexer``: the NSA index stage — quantize -> score -> select.
- ``varlen``: contiguous batched/varlen attention (reduced-assurance tier).
"""

from __future__ import annotations

import importlib
from typing import Any

_OP_MODULES = ("paged", "sparse_mla", "compressed_mla", "nsa_indexer", "varlen")


def __getattr__(name: str) -> Any:
    if name in _OP_MODULES:
        module = importlib.import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(_OP_MODULES)


__all__ = list(_OP_MODULES)
