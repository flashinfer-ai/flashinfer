"""Attention ops for b12x.

- ``paged``: paged-KV self-attention (decode + extend, FP8 KV, MSA
  block-sparse variant) with on-device graph-replay metadata staging.
- ``dense_mla``: dense compressed-cache MLA with strided physical records and
  optional causal sliding-window masking.
- ``sparse_mla``: top-k-selected MLA, including strided physical records.
- ``compressed_sparse_mla``: sparse MLA directly from compressed KV pages
  (DSV4).
- ``dsa_indexer``: the DSA index stage — quantize -> score -> select.
- ``qsa``: grouped-selector sparse GQA over caller-populated, read-only main
  BF16 paged K/V.
- ``varlen``: contiguous batched/varlen attention (reduced-assurance tier).
- ``mla_compress``: nonoverlapping CSA latent pooling before index-key projection.
"""

from __future__ import annotations

import importlib
from typing import Any

_OP_MODULES = (
    "paged",
    "dense_mla",
    "sparse_mla",
    "compressed_sparse_mla",
    "dsa_indexer",
    "qsa",
    "varlen",
    "mla_compress",
)


def __getattr__(name: str) -> Any:
    if name in _OP_MODULES:
        module = importlib.import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(_OP_MODULES)


__all__ = list(_OP_MODULES)
