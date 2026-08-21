# SPDX-FileCopyrightText: Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Fused GDN decode step -- single-launch persistent CUDA impl (SM120).

Host side of the ``cuda_sm120_persistent`` registry impl: one B-dynamic JIT
module (``gdn_fused_decode_sm120.cu`` in this package, two template
instantiations: B==1 / general) launched on the caller's current stream.
The kernel runs the whole fused step in a single persistent launch,
synchronized by a device-wide grid barrier.

Implements the impl-module interface documented in ../README.md.
Compilation is lazy: the first eager :func:`execute` builds and loads the
module and allocates the per-device barrier and the per-(batch, device)
launch scratch; once warm, calls are capture-safe (regular launches only,
no allocation).  Consumed by
:mod:`flashinfer.gdn_kernels.experimental.gdn_fused_decode_specialized`;
import errors are tolerated there.
"""

import functools
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

# Per-(batch, device) launch scratch: the conv output and the fp32 GEMV
# partials. Both are pure scratch -- written before they are read within the
# one launch that uses them -- so they are cached rather than reallocated per
# call, like the CuTe-DSL impl's workspace: it keeps the allocator off the
# per-layer decode path and, more importantly, keeps allocation out of CUDA
# graph capture, which is what ready_for_graph_capture() is allowed to promise.
# Shared per device, so they fall under the same cross-stream ordering as the
# barrier (order_after_previous_stream below covers all three).
_scratch_cache: dict = {}

# Stream that last launched with the shared per-device barrier and scratch.
_barrier_stream: dict = {}

_launch_count = 0


@functools.cache
def _get_module():
    """Build-and-load the JIT module once per process (the repo's standard
    Python-level module cache; the on-disk .so cache is the second level).

    ``_module_is_resident()`` reads the memo without populating it, which is
    what makes the capture-readiness check side-effect free.
    """
    return gen_gdn_fused_decode_module().build_and_load()


def _module_is_resident() -> bool:
    """True once :func:`_get_module` has built and loaded the module."""
    return _get_module.cache_info().currsize > 0


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


def _scratch_key(hidden_states: torch.Tensor, conv_state: torch.Tensor) -> tuple:
    """Cache key of the launch scratch, from the two tensors both the
    readiness check and :func:`execute` are given.

    ``conv_state`` is the logical ``[P, qkv_dim, state_len]`` view, so it
    carries ``qkv_dim``; together with the batch size and the device that
    fixes ``conv_out``'s shape.  ``n_ba`` is not derivable here, but it is
    pinned per geometry by the dispatch guard, so it cannot vary
    independently of ``qkv_dim`` for a registered call -- :func:`_get_scratch`
    re-allocates if it ever does.
    """
    return (
        int(hidden_states.shape[0]),
        int(conv_state.shape[1]),
        str(hidden_states.device),
    )


def _get_scratch(
    hidden_states: torch.Tensor, conv_state: torch.Tensor, n_ba: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """The (conv_out, ba_part) scratch pair for this call, allocated once."""
    key = _scratch_key(hidden_states, conv_state)
    B, qkv_dim = key[0], key[1]
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
    hidden_states: torch.Tensor, conv_state: torch.Tensor, scale: float
) -> bool:
    """True when a call is capture-safe: the compiled module is resident and
    the persistent barrier and the launch scratch for this call already exist,
    so neither compilation nor allocation can happen during capture.  The
    kernel is B-dynamic and takes scale and the conv-state strides as runtime
    parameters, so readiness does not depend on those -- but it does depend on
    the batch size and ``qkv_dim``, which size the scratch."""
    return (
        _module_is_resident()
        and str(hidden_states.device) in _barrier_cache
        and _scratch_key(hidden_states, conv_state) in _scratch_cache
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
    module = _get_module()
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
    conv_out_scratch, ba_scratch = _get_scratch(hidden_states, conv_state, n_ba)
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


def compiled_variant_keys() -> list:
    """Compiled-kernel descriptors resident in this process."""
    return ["sm120_persistent_b_dynamic"] if _module_is_resident() else []


def variant_plan(rows) -> set:
    """Distinct compiled kernels this impl needs for its registry rows: one
    B-dynamic module serves every row (the conv-state layout and scale are
    runtime parameters)."""
    return {"sm120_persistent_b_dynamic"} if rows else set()
