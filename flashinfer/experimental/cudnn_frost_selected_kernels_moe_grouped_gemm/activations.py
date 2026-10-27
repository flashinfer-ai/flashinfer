# Copyright (c) 2026 by FlashInfer team. Licensed under Apache-2.0.
"""Compatibility import for flashinfer.fused_moe.backends.cudnn_frost.activations (one release)."""

from importlib import import_module
import sys

sys.modules[__name__] = import_module(
    "flashinfer.fused_moe.backends.cudnn_frost.activations"
)
