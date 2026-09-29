# SPDX-FileCopyrightText: Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Fused GDN decode step -- single-launch persistent CUDA impl (SM120).

Host side of the ``cuda_sm120_persistent`` registry impl: one B-dynamic JIT
module per layer geometry (``gdn_fused_decode_sm120.cu`` in this package,
two template instantiations each: B==1 / general) launched on the caller's
current stream.  The kernel runs the whole fused step in a single persistent
launch, synchronized by a device-wide grid barrier.

Implements the impl-module interface documented in ../README.md.
Compilation is lazy: the first eager :func:`execute` of a geometry builds and
loads its module and allocates the per-device barrier and the
per-(geometry, batch, device) launch scratch; once warm, calls are capture-safe
(regular launches only, no allocation).  Consumed by
:mod:`flashinfer.gdn_kernels.experimental.gdn_fused_decode_specialized`;
import errors are tolerated there.
"""

from typing import Optional, Tuple

import torch

from ....jit.gdn_fused_decode import gen_gdn_fused_decode_module
from ._stream_order import order_after_previous_stream

# K-split factor of the kernel's b/a GEMV (fp32 partials in ba_scratch).
_GEMV_NSPLIT = 160

# Zero-initialized two-slot grid-barrier buffer per device. The kernel's
# device-wide barrier resets its own arrival counter and spins on a monotonic
# release generation, so the buffer is allocated (and zeroed) exactly once per
# device and safely reused by every launch on that device -- no per-call
# memset.
#
# That reuse assumes AT MOST ONE kernel using the buffer is in flight (see the
# "Requires:" note on grid_barrier in gdn_fused_decode_sm120.cu): a second
# concurrent kernel would add its blocks to the same arrival counter and both
# grids would sync on a count neither of them has, hanging or releasing early.
# Launches on one stream are serialized by the stream itself;
# order_after_previous_stream() (see _stream_order.py) extends that to a caller
# that switches streams, so the earlier persistent kernel has retired before
# the next one arrives at the same barrier buffer.
_barrier_cache: dict = {}

# Per-(geometry, batch, device) launch scratch: the conv output and the fp32
# GEMV partials. Both are pure scratch -- written before they are read within
# the one launch that uses them -- so they are cached rather than reallocated
# per call, like the CuTe-DSL impl's workspace: it keeps the allocator off the
# per-layer decode path and, more importantly, keeps allocation out of CUDA
# graph capture, which is what ready_for_graph_capture() is allowed to promise.
# Shared calls fall under the same cross-stream ordering as the barrier
# (order_after_previous_stream below covers all three buffers).
_scratch_cache: dict = {}

# Stream that last launched with the shared per-device barrier and cached scratch.
_barrier_stream: dict = {}

_launch_count = 0


# Compiled modules by layer geometry.  The kernel is B-dynamic and takes the
# query scale and the conv-state strides as runtime parameters, but the layer
# geometry is a compile-time parameter of its translation unit, so there is
# one module per geometry (a serving process runs one model, hence one).
#
# An explicit dict rather than @functools.cache: capture-readiness has to
# answer "is the module for THIS geometry resident", and ``cache_info()``
# only exposes a COUNT of memoized entries, not membership of one key.  With
# a single geometry the two were equivalent; with more than one, a count
# would report a warm 27B module as readiness for a 35B call.  Reading
# ``geometry in _modules`` is side-effect free, which is the property the
# readiness check actually needs.
_modules: dict = {}


def _geometry_from_tensors(
    hidden_states: torch.Tensor,
    w_ba: torch.Tensor,
    mixed_qkv: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_state: torch.Tensor,
    A_log: torch.Tensor,
    ssm_state: torch.Tensor,
) -> tuple:
    """The compile-time geometry key of a validated call.

    ``h_q`` is derived the same way the dispatch guard derives it, from the
    q/k/v head split of ``qkv_dim``.
    """
    hidden = int(hidden_states.shape[1])
    n_ba = int(w_ba.shape[1])
    qkv_dim = int(mixed_qkv.shape[1])
    hv = int(A_log.shape[0])
    d = int(ssm_state.shape[-1])
    conv_width = int(conv_weight.shape[1])
    conv_state_len = int(conv_state.shape[2])
    h_q = (qkv_dim - hv * d) // (2 * d)
    return (hidden, n_ba, qkv_dim, h_q, hv, d, conv_width, conv_state_len)


def _get_module(geometry: tuple):
    """Build-and-load this geometry's JIT module once per process (the
    Python-level module cache; the on-disk .so cache is the second level)."""
    module = _modules.get(geometry)
    if module is None:
        module = gen_gdn_fused_decode_module(*geometry).build_and_load()
        _modules[geometry] = module
    return module


def geometry_key(signature: dict) -> tuple:
    """Compile-time geometry key of a dispatch signature."""
    return (
        int(signature["hidden"]),
        int(signature["n_ba"]),
        int(signature["qkv_dim"]),
        int(signature["h_q"]),
        int(signature["hv"]),
        int(signature["d"]),
        int(signature["conv_width"]),
        int(signature["conv_state_len"]),
    )


def _get_barrier(device: torch.device) -> torch.Tensor:
    """The device's grid-barrier buffer, allocated (and zeroed) on first use.

    Cached per device: the kernel's barrier resets its own arrival counter and
    spins on a monotonic generation, so one zeroed buffer serves every launch
    on that device and no per-call memset is needed.  Allocating here rather
    than per call is also what makes capture-readiness checkable in advance.
    """
    key = str(device)
    barrier = _barrier_cache.get(key)
    if barrier is None:
        barrier = torch.zeros((2,), dtype=torch.int32, device=device)
        _barrier_cache[key] = barrier
    return barrier


def _scratch_key(
    geometry: tuple, hidden_states: torch.Tensor, conv_state: torch.Tensor
) -> tuple:
    """Cache key of the launch scratch for one layer geometry and call shape.

    The full geometry is required even where two models share ``qkv_dim``:
    ``n_ba`` sizes the fp32 GEMV partials, and capture readiness must not
    mistake another geometry's allocation for this call's scratch.
    """
    return (
        geometry,
        int(hidden_states.shape[0]),
        int(conv_state.shape[1]),
        str(hidden_states.device),
    )


def _get_scratch(
    geometry: tuple,
    hidden_states: torch.Tensor,
    conv_state: torch.Tensor,
    n_ba: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """The (conv_out, ba_part) scratch pair for this call, allocated once."""
    key = _scratch_key(geometry, hidden_states, conv_state)
    B, qkv_dim = key[1], key[2]
    ba_elems = _GEMV_NSPLIT * B * n_ba
    scratch = _scratch_cache.get(key)
    if scratch is None or scratch[1].numel() != ba_elems:
        device = hidden_states.device
        scratch = (
            torch.empty((B, qkv_dim), dtype=torch.bfloat16, device=device),
            torch.empty((ba_elems,), dtype=torch.float32, device=device),
        )
        _scratch_cache[key] = scratch
    return scratch


def ready_for_graph_capture(
    signature: dict,
    hidden_states: torch.Tensor,
    conv_state: torch.Tensor,
    scale: float,
) -> bool:
    """True when a call is capture-safe: the module for *this geometry* is
    resident and the persistent barrier and the launch scratch for this call
    already exist, so neither compilation nor allocation can happen during
    capture.  The kernel is B-dynamic and takes scale and the conv-state
    strides as runtime parameters, so readiness does not depend on those --
    but it does depend on the batch size and ``qkv_dim``, which size the
    scratch, and on the layer geometry, which is compiled into the module.

    The geometry comes from the matched dispatch signature rather than being
    re-derived here: readiness must be answered about the exact variant the
    dispatcher is about to run.  Without it, a process warm for one model
    would report a call from a differently-shaped model capture-ready and
    bake the wrong-geometry kernel into the graph."""
    geometry = geometry_key(signature)
    return (
        geometry in _modules
        and str(hidden_states.device) in _barrier_cache
        and _scratch_key(geometry, hidden_states, conv_state) in _scratch_cache
    )


def execute(
    hidden_states: torch.Tensor,
    w_ba: torch.Tensor,
    mixed_qkv: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor,
    conv_state: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    scale: float,
    ssm_state: torch.Tensor,
    state_indices: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the fused step on the caller's current stream; raise on failure.

    The dispatch layer has already validated the call against the registry
    and the op contract.  Both state pools are updated in place.
    """
    global _launch_count
    geometry = _geometry_from_tensors(
        hidden_states, w_ba, mixed_qkv, conv_weight, conv_state, A_log, ssm_state
    )
    module = _get_module(geometry)
    B = hidden_states.shape[0]
    n_ba = w_ba.shape[1]
    hv = A_log.shape[0]
    d = ssm_state.shape[-1]
    device = hidden_states.device
    output = (
        out
        if out is not None
        else torch.empty((B, 1, hv, d), dtype=torch.bfloat16, device=device)
    )
    conv_out_scratch, ba_scratch = _get_scratch(
        geometry, hidden_states, conv_state, n_ba
    )
    barrier = _get_barrier(device)
    order_after_previous_stream(_barrier_stream, device)
    module.gdn_fused_decode(
        hidden_states,
        w_ba,
        mixed_qkv,
        conv_weight,
        conv_bias,
        conv_state,
        A_log,
        dt_bias,
        ssm_state,
        state_indices,
        output,
        conv_out_scratch,
        ba_scratch,
        barrier,
        float(scale),
    )
    _launch_count += 1
    return output, conv_state, ssm_state


def launch_count() -> int:
    """Host-side dispatches so far (a CUDA-graph capture counts once)."""
    return _launch_count


def _geometry_tag(geometry: tuple) -> str:
    hidden, n_ba, qkv_dim, h_q, hv, d, conv_width, conv_state_len = geometry
    return (
        f"sm120_persistent_b_dynamic_h{hidden}_nba{n_ba}_qkv{qkv_dim}"
        f"_hq{h_q}_hv{hv}_d{d}_w{conv_width}_s{conv_state_len}"
    )


def compiled_variant_keys() -> list:
    """Compiled-kernel descriptors resident in this process."""
    return sorted(_geometry_tag(geometry) for geometry in _modules)


def variant_plan(rows) -> set:
    """Distinct compiled kernels this impl needs for its registry rows: one
    B-dynamic module per layer geometry (batch size, conv-state layout and
    scale are runtime parameters)."""
    return {_geometry_tag(geometry_key(row)) for row in rows}
