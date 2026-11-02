"""Stateful token-sequence feature kernels for b12x.

- ``ple_hash``: prime-hashed learned n-gram embedding IDs.
- ``ple_embedding``: fused PLE hashing and BF16, FP8, or NVFP4 table lookup.
- ``ple``: stateful PLE residual contribution and short-convolution state.
- ``gdn_decode``: packed gated-delta recurrent decode and output gating.
- ``gdn_prefill``: chunked scalar-gated GDN prefill over the state pool.
- ``kda_prefill``: chunked lower-bounded KDA prefill over the state pool.
- ``mtp_feedback``: MTP token/multi-stream feedback fusion.
- ``engram``: compressed-token n-gram hashes and row-sharded FP8 table lookup.
- ``embedding``: exact unquantized token-row lookup with caller-owned output.
"""

from __future__ import annotations

import importlib
from typing import Any

_OP_MODULES = (
    "ple_hash",
    "ple_embedding",
    "ple",
    "gdn_decode",
    "kda_prefill",
    "gdn_prefill",
    "mtp_feedback",
    "engram",
    "embedding",
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
