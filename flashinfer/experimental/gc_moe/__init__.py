"""Experimental GC MoE backend.

Green-Context megakernel MoE for SM90, SM100/SM103 and SM120. The public
functions live in core (``flashinfer.gc_moe``,
``flashinfer.GCMoEWorkspace``, ``flashinfer.transform_gc_moe_weights``); this
package holds the routing, the workspace and weight adapters, and the vendored
upstream sources under ``kernel_src/gc_moe``.
"""

from .dispatch import run_gc_moe
from .kernel_src.gc_moe import UPSTREAM_COMMIT, UPSTREAM_VERSION, close_runtime
from .plan import GcMoePlan, block_m
from .support import (
    BACKENDS,
    backend_for_compute_capability,
    is_supported,
    resolve_backend,
)
from .weights import transform_weights
from .workspace import GCMoEWorkspace, create_workspace

__all__ = [
    "BACKENDS",
    "GCMoEWorkspace",
    "GcMoePlan",
    "UPSTREAM_COMMIT",
    "UPSTREAM_VERSION",
    "backend_for_compute_capability",
    "block_m",
    "close_runtime",
    "create_workspace",
    "is_supported",
    "resolve_backend",
    "run_gc_moe",
    "transform_weights",
]
