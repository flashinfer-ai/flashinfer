"""Vendored GC MoE drop.

Consumers import this package only; the ``shim`` layer is the sole code that
touches ``src/``, and ``src/`` tracks the upstream commit recorded in
``VENDOR.md`` plus its documented local patches.
"""

from .shim import (
    PTXAS_ENV,
    block_m,
    close_runtime,
    create_buffer,
    destroy_buffer,
    prepare_environment,
    resolve_ptxas,
    run,
    transform,
    translated,
)

UPSTREAM_VERSION = "0.6.2"
UPSTREAM_COMMIT = "7902831ec4a6691b6268b308bcff5ec0cb15ec86"

__all__ = [
    "PTXAS_ENV",
    "UPSTREAM_COMMIT",
    "UPSTREAM_VERSION",
    "block_m",
    "close_runtime",
    "create_buffer",
    "destroy_buffer",
    "prepare_environment",
    "resolve_ptxas",
    "run",
    "transform",
    "translated",
]
