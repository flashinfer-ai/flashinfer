"""
Copyright (c) 2025 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

KDA (Kimi Delta Attention) Kernels - CuTe DSL Implementations
=============================================================

This subpackage provides CuTe-DSL implementations of KDA chunk-prefill
kernels for NVIDIA Blackwell (SM100+).

The user-facing API is :func:`flashinfer.kda.chunk_kda_fwd`. This subpackage
exposes the same function as well as the underlying low-level building
blocks for callers that want to wire the kernels themselves.

Exported:
- ``chunk_kda_fwd``       : top-level prefill API (K123 -> akk_inv -> K4).
- ``prepare_chunk_indices``, ``prepare_chunk_offsets`` : varlen helpers.
"""

try:
    from .kda_chunk_fwd import (
        chunk_kda_fwd,
        prepare_chunk_indices,
        prepare_chunk_offsets,
    )

    _has_cute_dsl = True
except (ImportError, RuntimeError):
    _has_cute_dsl = False
    chunk_kda_fwd = None  # type: ignore
    prepare_chunk_indices = None  # type: ignore
    prepare_chunk_offsets = None  # type: ignore


__all__ = [
    "chunk_kda_fwd",
    "prepare_chunk_indices",
    "prepare_chunk_offsets",
]
