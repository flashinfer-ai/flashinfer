# SPDX-FileCopyrightText: Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Experimental fused MoE routing glue and the SM120 kernels serving it.

"Experimental" describes where this code *lives*, not how it is called: the
public API is exported at the top level like the other fused-MoE APIs, so
consumers write::

    import flashinfer

    if flashinfer.moe_routing_supported(m, hidden, num_experts, top_k):
        flashinfer.moe_routing_align(...)      # before the expert GEMMs
        ...
        flashinfer.moe_routing_finalize(...)   # after the w2 GEMM
    else:
        ...  # keep your own composition

The three entry points -- :func:`moe_routing_prologue`,
:func:`moe_routing_align` and :func:`moe_routing_finalize` -- are the non-GEMM
glue a serving engine runs around the routed-expert GEMMs of one MoE block.
The interface (this re-export, the guards, the portable torch specification and
the dispatch) lives in :mod:`.moe_routing`; the CUDA source the specialized
path compiles lives in :mod:`.kernel` (see README.md).

Importing this package pulls in :mod:`torch` and nothing else: the allowlist,
the JIT machinery and the kernel are all reached lazily, on the first probe or
call.  ``flashinfer/__init__.py`` therefore re-exports the public names
unconditionally, and a consumer's capability check (``getattr(flashinfer,
"moe_routing_finalize", None)``) costs nothing and never compiles -- it is
present whenever the library is new enough and absent on an older one, with no
third "present but broken" state.
"""

from .moe_routing import (
    BLOCK_SIZE_M as BLOCK_SIZE_M,
    moe_routing_align as moe_routing_align,
    moe_routing_finalize as moe_routing_finalize,
    moe_routing_precompile as moe_routing_precompile,
    moe_routing_prologue as moe_routing_prologue,
    moe_routing_ready_for_graph_capture as moe_routing_ready_for_graph_capture,
    moe_routing_stats as moe_routing_stats,
    moe_routing_supported as moe_routing_supported,
)

__all__ = [
    "BLOCK_SIZE_M",
    "moe_routing_align",
    "moe_routing_finalize",
    "moe_routing_precompile",
    "moe_routing_prologue",
    "moe_routing_ready_for_graph_capture",
    "moe_routing_stats",
    "moe_routing_supported",
]
