# Copyright (c) 2026 by FlashInfer team. Licensed under Apache-2.0.
"""Compatibility import for flashinfer.cake_causal_conv1d_det_train_backend.

An attribute of this package resolves on the det package at first access, and the submodules ``cake_build``,
``cake_jit``, ``kcp`` and ``kcp_jit`` are the det package's module objects.  Importing this package imports
nothing else: the KCP host module loads only when ``kcp`` is imported.
"""

from importlib import import_module


def __getattr__(name):
    return getattr(
        import_module("flashinfer.cake_causal_conv1d_det_train_backend"), name
    )
