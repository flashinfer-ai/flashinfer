# Copyright (c) 2026 by FlashInfer team. Licensed under Apache-2.0.
"""Compatibility import for flashinfer.fused_moe.backends.cudnn_frost.nvfp4.fma_fc1_o4 (one release)."""

from importlib import import_module
import sys

sys.modules[__name__] = import_module(
    "flashinfer.fused_moe.backends.cudnn_frost.nvfp4.fma_fc1_o4"
)
