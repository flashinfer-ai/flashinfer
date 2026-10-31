# Copyright (c) 2026 by FlashInfer team. Licensed under Apache-2.0.
"""Compatibility import for flashinfer.experimental.cake_kda_det_train_kcp.cake_kcp (the same module object)."""

from importlib import import_module
import sys
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # Type checkers do not follow the sys.modules replacement below.
    def __getattr__(name: str) -> Any: ...


sys.modules[__name__] = import_module(
    "flashinfer.experimental.cake_kda_det_train_kcp.cake_kcp"
)
