"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import contextlib
import functools
from typing import FrozenSet, Optional

import torch

#: A QSA selection -- score a compressed cache, take the top blocks, expand
#: them into a token route.
QSA_CAP_SELECTION = 1 << 0
#: Sparse attention over a route of physical KV slots in a paged cache.
QSA_CAP_ATTENTION_PAGED = 1 << 1
#: A packed NVFP4 cache, read through its e4m3 scale planes.
QSA_CAP_NVFP4 = 1 << 2
#: An e4m3 cache with one host scale per tensor.
QSA_CAP_FP8 = 1 << 3
#: The output gate, folded in by a kernel rather than by elementwise ops.
QSA_CAP_OUTPUT_GATE = 1 << 4

_NAMES = {
    QSA_CAP_SELECTION: "selection",
    QSA_CAP_ATTENTION_PAGED: "attention_paged",
    QSA_CAP_NVFP4: "nvfp4",
    QSA_CAP_FP8: "fp8",
    QSA_CAP_OUTPUT_GATE: "output_gate",
}

#: The gate module's own bits: 1 for float16, 2 for bfloat16.
_GATE_FP16 = 1
_GATE_BF16 = 2


def _module_has(loader, *symbols) -> bool:
    """Build and load a module, and say whether it exports these.

    Loading is what separates "this package has the Python for it" from "this
    build has the kernels": the module is compiled on first use, and a source
    tree whose kernels do not build fails here rather than in a forward pass.
    A cached artifact older than the source answers with what it was built
    with, which is the point.
    """
    try:
        module = loader()
    except Exception:
        return False
    return all(getattr(module, symbol, None) is not None for symbol in symbols)


def _normalized(device: Optional[torch.device]) -> torch.device:
    """One name per device, so the cache below is keyed on the device itself."""
    resolved = torch.device(device) if device is not None else torch.device("cuda")
    if resolved.type == "cuda" and resolved.index is None:
        resolved = torch.device("cuda", torch.cuda.current_device())
    return resolved


def qsa_capabilities(device: Optional[torch.device] = None) -> int:
    """What this build can do for QSA on this device, as a bitmask.

    The answer is per device: the kernels are compiled for an architecture,
    and a machine with two of them can give two answers.
    """
    return _capabilities(_normalized(device))


@functools.cache
def _capabilities(device: torch.device) -> int:
    """What this build of FlashInfer can do for QSA, as a bitmask.

    A caller deciding whether to use this route should not have to guess from a
    function signature or find out by catching an exception in the middle of a
    forward pass.

    What each bit rests on differs, and the difference matters:

    * ``selection`` and ``output_gate`` are **compiled** capabilities. Their
      modules are built and their symbols looked for, so a build missing one
      of them reports it absent.
    * ``attention_paged``, ``nvfp4`` and ``fp8`` are **API** capabilities: this
      package carries the paged route and the two quantized formats, and the
      kernels behind them are compiled per geometry when a plan is made. Their
      presence here says the route exists, not that a particular shape will
      build. Nothing here answers that, on purpose: the answer is building
      :class:`~flashinfer.qsa_attention.QSAAttention` with the buffers the
      caller reserved, which raises with the reason. See the note at the foot
      of this module.

    Returns
    -------
    int
        The OR of the ``QSA_CAP_*`` bits. Zero means this build serves none of
        it, which is a usable answer rather than an error.
    """
    # The loaders below compile for whatever device is current, and not all of
    # them take one as an argument, so the question is asked from inside the
    # device being asked about.
    context = (
        torch.cuda.device(device) if device.type == "cuda" else contextlib.nullcontext()
    )
    with context:
        return _capabilities_here(device)


def _capabilities_here(device: torch.device) -> int:
    bits = 0

    # Selection is three pieces: the scorer, the route expansion, and a top-k
    # whose scratch the caller owns. Any one of them missing is no selection.
    try:
        from .sparse_route import get_sparse_route_module
        from .sparse_scores import get_sparse_scores_module
        from .topk import get_topk_module

        from .qsa_selection import QSASelection  # noqa: F401

        pieces = (
            _module_has(
                lambda: get_sparse_scores_module(device), "sparse_paged_scores"
            ),
            _module_has(
                get_sparse_route_module,
                "expand_block_route",
                "qsa_route_from_logical",
            ),
            _module_has(
                get_topk_module,
                "radix_topk_ragged_transform",
                "cub_topk_ragged_transform_workspace_size_for",
            ),
        )
        if all(pieces):
            bits |= QSA_CAP_SELECTION
    except ImportError:
        pass

    # The gate is one module and it answers for itself.
    gate = 0
    try:
        from .qsa_output_gate import get_qsa_output_gate_module

        module = get_qsa_output_gate_module()
        query = getattr(module, "qsa_output_gate_capabilities", None)
        # A build that predates the query may still carry the kernel, but it
        # cannot say so, and a capability nobody can confirm is one a caller
        # should not rely on.
        gate = int(query()) if query is not None else 0
    except Exception:
        gate = 0
    if gate & _GATE_BF16:
        bits |= QSA_CAP_OUTPUT_GATE

    # Attention needs the paged block-sparse wrapper, its allocation-free
    # sizing query, and -- because the route is gated on the way out -- the
    # gate. The quantized formats ride the same wrapper, so they are claimed
    # only when it is there.
    try:
        from .sparse import BlockSparseAttentionWrapper

        from .qsa_attention import QSAAttention  # noqa: F401

        # An API capability, not a compiled one: the block-sparse kernels are
        # built per geometry, so what is checked here is that the route and
        # its allocation-free sizing query exist at all.
        paged = hasattr(BlockSparseAttentionWrapper, "query_workspace_size") and (
            "kv_cache_page_size"
            in BlockSparseAttentionWrapper.plan.__code__.co_varnames
        )
        if paged and bits & QSA_CAP_OUTPUT_GATE:
            bits |= QSA_CAP_ATTENTION_PAGED
            if hasattr(BlockSparseAttentionWrapper, "run"):
                bits |= QSA_CAP_NVFP4 | QSA_CAP_FP8
    except ImportError:
        pass

    return bits


def qsa_capability_names() -> FrozenSet[str]:
    """The same answer as names, for a log line or an error message."""
    available = qsa_capabilities()
    return frozenset(name for bit, name in _NAMES.items() if available & bit)


# There is deliberately no ``supports_qsa_config()`` here.
#
# Whether a particular geometry works is answered by building it, and building
# it means a float workspace, an integer arena and a JIT compile -- all of
# which the caller is going to provide and pay for anyway. A probe that
# allocated its own would answer for a workspace nobody runs with (how much
# room split-k has changes what the planner lays down), and a caller that
# probed and then built would compile and allocate twice.
#
# So the answer is the construction: build :class:`~flashinfer.qsa_attention.
# QSAAttention` with the buffers the deployment reserved and bind it. A shape
# the library cannot serve raises there, with the reason -- a ValueError for a
# geometry, the compiler's own error for a build failure, an out-of-memory for
# memory -- rather than collapsing all three into False.
