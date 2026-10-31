# Copyright (c) 2026 by FlashInfer team. Licensed under Apache-2.0.
"""Compatibility import for flashinfer.fused_moe.backends.cudnn_frost.mxfp8_mxfp4 (one release)."""

from importlib import import_module


def __getattr__(name):
    return getattr(
        import_module("flashinfer.fused_moe.backends.cudnn_frost.mxfp8_mxfp4"), name
    )
