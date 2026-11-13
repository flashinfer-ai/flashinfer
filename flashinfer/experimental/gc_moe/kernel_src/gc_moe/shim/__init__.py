"""Shim layer over the vendored GC MoE sources.

The vendored tree under ``src/`` tracks the upstream commit in ``VENDOR.md``
plus its documented local patches; everything that adapts it to FlashInfer
lives here and is re-exported through the drop's package ``__init__``.
"""

from .env import PTXAS_ENV, prepare as prepare_environment, resolve_ptxas
from .errors import translated
from .launch import block_m, close_runtime, run, sources, transform
from .workspace import create as create_buffer
from .workspace import destroy as destroy_buffer

__all__ = [
    "PTXAS_ENV",
    "block_m",
    "close_runtime",
    "create_buffer",
    "destroy_buffer",
    "prepare_environment",
    "resolve_ptxas",
    "run",
    "sources",
    "transform",
    "translated",
]
