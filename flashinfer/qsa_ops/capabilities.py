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
import inspect
from typing import FrozenSet, Optional

import torch

from ..api_logging import flashinfer_api

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
_GATE_BF16 = 2


def _module_has(loader, *symbols) -> bool:
    """Build and load a module, and say whether it exports these: what separates
    having the Python from having the kernels. A cached artifact answers for itself."""
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


@flashinfer_api
def qsa_capabilities(device: Optional[torch.device] = None) -> int:
    """What this build can do for QSA on this device, as a bitmask.

    The answer is per device: the kernels are compiled for an architecture,
    and a machine with two of them can give two answers. See ``_capabilities``
    below for what each bit rests on.

    Parameters
    ----------
    device : Optional[torch.device]
        The device to answer for; the current CUDA device when omitted.

    Returns
    -------
    int
        The OR of the ``QSA_CAP_*`` bits, zero when none is available.
    """
    return _capabilities(_normalized(device))


@functools.cache
def _capabilities(device: torch.device) -> int:
    """The ``QSA_CAP_*`` bits this build serves on ``device``; zero is an answer.

    ``selection`` and ``output_gate`` are **compiled** capabilities: their
    modules are built and their symbols looked for. ``attention_paged``,
    ``nvfp4`` and ``fp8`` are **API** capabilities: the block-sparse kernels
    behind them are compiled per geometry when a plan is made, so whether a
    shape builds is answered by building :class:`~flashinfer.qsa_ops.QSAAttention`
    with the caller's buffers, which raises with the reason.
    """
    # The loaders compile for the current device, so ask from inside it.
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
        from .route import get_qsa_route_module
        from .scores import get_qsa_scores_module, qsa_paged_scores
        from ..topk import get_topk_module

        from .selection import QSASelection  # noqa: F401

        def scorer():
            # The module may load on a device the kernel does not support, when
            # the build also targets one it does.
            major, minor = torch.cuda.get_device_capability(device)
            if not qsa_paged_scores.is_compute_capability_supported(major * 10 + minor):
                raise RuntimeError("the scorer needs SM80 or newer")
            return get_qsa_scores_module()

        pieces = (
            _module_has(scorer, "qsa_paged_scores"),
            _module_has(
                get_qsa_route_module,
                "qsa_expand_block_route",
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
        from .output_gate import get_qsa_output_gate_module

        gate = int(get_qsa_output_gate_module().qsa_output_gate_capabilities())
    except Exception:
        gate = 0
    if gate & _GATE_BF16:
        bits |= QSA_CAP_OUTPUT_GATE

    # Attention needs the paged wrapper, its sizing query and the gate; the
    # quantized formats ride the same wrapper.
    try:
        from ..sparse import BlockSparseAttentionWrapper

        from .attention import QSAAttention  # noqa: F401

        paged = hasattr(BlockSparseAttentionWrapper, "query_workspace_size") and (
            "kv_cache_page_size"
            in inspect.signature(BlockSparseAttentionWrapper.plan).parameters
        )
        if paged and bits & QSA_CAP_OUTPUT_GATE:
            bits |= QSA_CAP_ATTENTION_PAGED
            if hasattr(BlockSparseAttentionWrapper, "run"):
                bits |= QSA_CAP_NVFP4 | QSA_CAP_FP8
    except ImportError:
        pass

    return bits


@flashinfer_api
def qsa_capability_names() -> FrozenSet[str]:
    """The same answer as names, for a log line or an error message."""
    available = qsa_capabilities()
    return frozenset(name for bit, name in _NAMES.items() if available & bit)


# There is deliberately no ``supports_qsa_config()``: a probe would allocate and
# compile what the caller is about to anyway, for a workspace nobody runs with.
# Building QSAAttention with the deployment's buffers raises with the reason
# (a ValueError, the compiler's error or an out-of-memory) instead of False.
