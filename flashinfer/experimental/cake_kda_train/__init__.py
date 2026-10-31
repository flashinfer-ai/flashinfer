# Copyright (c) 2026 by FlashInfer team. Licensed under Apache-2.0.
"""Compatibility import for flashinfer.experimental.cake_kda_det_train.

The names of this package are the det package's objects.  ``__all__`` is the list this package exported before
the move, so ``from flashinfer.experimental.cake_kda_train import *`` binds the same names as before; any other
attribute resolves on the det package at first access.  The submodules ``cake_backend``, ``cake_build`` and
``cake_jit`` are the det package's module objects and ``cake_kcp``, ``cake_kcp_jit`` and ``cake_kcp_plan`` those of
``flashinfer.experimental.cake_kda_det_train_kcp``, so the registries, the workspace caches and the loaded libraries
exist once.  This package holds no sources and no manifest; importing it imports what the det package's
``__init__`` imports, never the KCP host module.
"""

from importlib import import_module

from flashinfer.experimental.cake_kda_det_train import *  # noqa: F403
from flashinfer.experimental.cake_kda_det_train import (  # noqa: F401
    ChunkKDAFunction,
    KdaChunkSaved,
    KdaWorkspace,
    chunk_kda,
    chunk_kda_bwd,
    chunk_kda_fwd,
)

__all__ = [
    "ARCHES",
    "ARCH_BY_CAPABILITY",
    "SAVE_MODES",
    "SAVED_VERSION",
    "SAVED_VERSION_TAPE",
    "SCALE",
    "SEG_SAVE_MIN_TOKENS_PER_SEG",
    "ChunkKDAFunction",
    "KdaChunkSaved",
    "KdaWorkspace",
    "Plan",
    "chunk_kda",
    "chunk_kda_bwd",
    "chunk_kda_fwd",
    "is_supported",
    "memory_report",
    "output_bytes",
    "plan",
    "saved_bytes",
    "seg_states_saved",
    "workspace_bytes",
]


def __getattr__(name):
    return getattr(import_module("flashinfer.experimental.cake_kda_det_train"), name)
