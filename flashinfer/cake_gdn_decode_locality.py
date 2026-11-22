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

Locality-domain FP32 GDN decode state on a dual-die SM107a GPU (Cake GDN).

A dual-die SM107a GPU (Rubin) has two locality domains (dies), each with its own
SMs and memory. :func:`localize_gdn_decode_state` copies a K-major FP32 decode
state ``[B, HV, K, V]`` into one allocation whose batch rows ``[0, rows0)`` are
memory of die 0 and the rest memory of die 1
(:func:`flashinfer.cake_locality.split_empty`). Passed as ``state`` to
:func:`flashinfer.gdn_decode.gated_delta_rule_decode`, a T=1 call on that state
runs the locality-domain twin of the Cake nontranspose FP32 decode body in the
cells its route table lists
(``flashinfer.jit.cake_gdn.CAKE_GDN_NONTRANSPOSE_FP32_T1_DOMAIN_ROUTES``): one
launch over both dies whose CTAs take the (batch row, value head) items of their
own die's rows first. Its output and updated state are those of the whole-GPU
body bit for bit. Every other call that the Cake kernels serve runs the whole-GPU
route; one they do not serve runs the FlashInfer kernel under ``backend="auto"``
and raises ``CakeGDNUnsupportedError`` under ``backend="cake_gdn"``. A call with
the dtypes, shapes and contiguity those kernels take raises ``ValueError`` under
both, on every row and GPU, if q or k is not 8-byte aligned or the state is not
16-byte aligned.
"""

from __future__ import annotations

import contextlib
import functools
from typing import Optional

import torch

from . import cake_locality
from .api_logging import flashinfer_api

__all__ = [
    "DOMAIN_COUNTER_WORDS",
    "localize_gdn_decode_state",
    "nontranspose_domain_variant",
]

# One int32 claim word per die: each launch of the twin takes a zeroed pair of
# them (flashinfer.cake_locality.launch_counters) and leaves it zero.
DOMAIN_COUNTER_WORDS = 1
_HEAD_SIZE = 128


@functools.cache
def _device_info(device_index: int) -> tuple[Optional[str], int]:
    """``(Cake GDN architecture or None, SM count)`` of CUDA device ``device_index``."""
    from .jit import cake_gdn

    properties = torch.cuda.get_device_properties(device_index)
    try:
        arch: Optional[str] = cake_gdn.arch_for_compute_capability(
            properties.major, properties.minor
        )
    except cake_gdn.CakeGDNUnsupportedError:
        arch = None
    return arch, int(properties.multi_processor_count)


def nontranspose_domain_variant(
    *,
    arch: str,
    sm_count: int,
    die_class: str,
    batch_size: int,
    num_q_heads: int,
    num_k_heads: int,
    num_v_heads: int,
    scale: float,
    use_qk_l2norm: bool = True,
) -> Optional[str]:
    """The twin record a nontranspose FP32 T=1 call of this row runs on a homed state
    on an ``arch`` device of ``sm_count`` SMs whose dies are of ``die_class``
    (``flashinfer.cake_locality.die_class``), or ``None``: the row's whole-GPU
    route (``select_cake_gdn_decode_variant``) has no twin listed in that cell, or
    the row has no Cake route at all."""
    from .jit import cake_gdn

    try:
        whole = cake_gdn.select_cake_gdn_decode_variant(
            arch=arch,
            batch_size=int(batch_size),
            io_dtype="bfloat16",
            state_dtype="float32",
            head_size=_HEAD_SIZE,
            layout="nontranspose",
            num_k_heads=int(num_k_heads),
            num_q_heads=int(num_q_heads),
            num_v_heads=int(num_v_heads),
            scale=scale,
            seq_len=1,
            use_qk_l2norm=use_qk_l2norm,
        )
    except cake_gdn.CakeGDNUnsupportedError:
        return None
    return cake_gdn.select_cake_gdn_decode_domain_variant(
        arch=arch,
        whole_variant=whole.variant_name,
        sm_count=int(sm_count),
        die_class=die_class,
        batch_size=int(batch_size),
    )


@flashinfer_api
def localize_gdn_decode_state(
    state: torch.Tensor,
    num_q_heads: int,
    scale: Optional[float] = None,
    rows0: Optional[int] = None,
) -> torch.Tensor:
    r"""A copy of the K-major FP32 GDN decode state ``state`` (``[B, HV, K, V]``,
    contiguous) homed by batch rows on the two dies of a dual-die SM107a GPU: rows
    ``[0, rows0)`` in die 0's memory, the rest in die 1's, in one virtual address
    range.

    Pass the returned tensor as ``state`` to
    :func:`flashinfer.gdn_decode.gated_delta_rule_decode` from then on (the call
    updates it in place, as any state). A T=1 call with ``num_q_heads`` query
    heads, this ``scale``, contiguous BF16 q/k/v/a/b and output, FP32 ``A_log`` and
    ``dt_bias`` and ``use_qk_l2norm=True`` under ``backend="auto"`` or
    ``backend="cake_gdn"`` then runs the locality-domain twin of the Cake
    nontranspose FP32 decode body: one launch over both dies whose CTAs take the
    (batch row, value head) items of their own die's rows first, so most state
    bytes stay on the die that reads and writes them. The output and the updated
    state are the whole-GPU body's bit for bit, for any ``rows0``.

    ``rows0`` defaults to the byte cut: half the rows on each die, the odd row on
    die 0 (``split_units(B, (1, 1))[0]``). ``scale`` defaults to ``K ** -0.5``,
    the default of the decode call.

    The returned tensor is a new allocation of the size of ``state``, made by the
    CUDA driver outside PyTorch's caching allocator
    (``torch.cuda.memory_allocated()`` does not count it). ``state`` itself is not
    freed: drop the last reference to it after this call so that the state is not
    held twice.

    Returns ``state`` itself, without a copy, when the call is under CUDA graph
    capture, or when it does not home the state: ``state`` is not FP32 with
    ``K = V = 128`` on an SM107a device, ``state`` is a ``__torch_dispatch__``
    wrapper subclass, the call row (``B``, ``num_q_heads``, ``HV``, ``scale``) has
    no twin listed for the device's SM count in any die class (the twin covers
    ``H=16``, ``HV=32`` from ``B=48`` on, at ``scale`` ``K ** -0.5`` or ``1.0``, on
    212-SM devices), the device has no SM table
    (:func:`flashinfer.cake_locality.sm_table` is ``None``: a CUDA 13.5 toolkit and
    driver are needed), the row is not listed for the die class of the device, or
    the cut is not a multiple of the allocation granularity of locality-domain
    memory. The checks that need no device work come first: a refused row is
    returned before the SM table is probed or anything is allocated. A result other
    than ``state`` is therefore a new homed copy, and ``state`` itself as the result
    means only that no copy was made, not which kernel a decode call runs: a call
    on a ``state`` that was never homed never runs the twin, but a ``state`` that
    was already homed (for example by an earlier call of this helper) stays homed,
    and a call of a listed row on it can still run the twin. Raises ``ValueError``
    for a ``state`` that is not a contiguous 4-D CUDA tensor, a non-positive
    ``num_q_heads`` or a ``rows0`` outside ``[0, B]``.

    Only the whole returned allocation is homed: the tensor itself and every
    alias of it with the same shape (a ``detach()``, a view of the same shape).
    A view of part of it, such as ``state[:B']``, and a copy run the whole-GPU
    body, which on homed memory may be slower than on ordinary memory. A state of
    another batch needs its own homed allocation.

    Homing a state also fills the device's reserve of pre-zeroed claim counter
    pairs for captured calls (:func:`flashinfer.cake_locality.reserve_launch_counters`);
    a failure there does not fail this call. Make one eager decode call before a
    CUDA graph capture (it builds the kernel and probes the SM table). The memory
    is freed with the returned tensor, and freeing it synchronizes the device; a
    CUDA graph that reads the tensor after it is freed faults on replay, so keep
    it alive as long as such a graph. One graph must not replay on two streams at
    once.

    Args:
        state: K-major FP32 decode state ``[B, HV, K, V]``, contiguous, on CUDA.
        num_q_heads: Number of query (and key) heads ``H`` of the decode calls.
        scale: Query scale of the decode calls (default ``K ** -0.5``).
        rows0: Batch rows homed on die 0 (default: the byte cut).
    """
    if (
        not isinstance(state, torch.Tensor)
        or state.device.type != "cuda"
        or state.ndim != 4
    ):
        raise ValueError("state must be a CUDA torch.Tensor of shape [B, HV, K, V]")
    if not state.is_contiguous():
        raise ValueError("state must be contiguous")
    batch_size, num_v_heads, head_k, head_v = (int(extent) for extent in state.shape)
    if int(num_q_heads) <= 0:
        raise ValueError(f"num_q_heads must be positive, got {num_q_heads}")
    if rows0 is not None and not 0 <= int(rows0) <= batch_size:
        raise ValueError(f"rows0 must be in [0, {batch_size}], got {rows0}")
    if state.dtype != torch.float32 or (head_k, head_v) != (_HEAD_SIZE, _HEAD_SIZE):
        return state
    if type(state).__torch_dispatch__ is not torch.Tensor.__torch_dispatch__:
        # A __torch_dispatch__ (wrapper) subclass keeps its data in inner tensors,
        # which a homed copy of the outer tensor would not carry.
        return state
    index = (
        int(state.device.index)
        if state.device.index is not None
        else int(torch.cuda.current_device())
    )
    if cake_locality._stream_capturing(index):
        return state
    arch, sm_count = _device_info(index)
    if arch != cake_locality.LOCALITY_ARCH:
        return state
    scale = head_k**-0.5 if scale is None else float(scale)

    def variant(die_class: str) -> Optional[str]:
        return nontranspose_domain_variant(
            arch=arch,
            sm_count=sm_count,
            die_class=die_class,
            batch_size=batch_size,
            num_q_heads=int(num_q_heads),
            num_k_heads=int(num_q_heads),
            num_v_heads=num_v_heads,
            scale=scale,
        )

    # Before any probe, allocation or synchronization: some die class must list
    # this row.
    if not any(variant(die) is not None for die in ("even", "uneven")):
        return state
    table = cake_locality.sm_table(index)
    if table is None:
        return state
    if variant(cake_locality.die_class(table.counts)) is None:
        return state
    if rows0 is None:
        rows0 = cake_locality.split_units(batch_size, (1, 1))[0]
    row_bytes = num_v_heads * head_k * head_v * state.element_size()
    cut = int(rows0) * row_bytes
    if 0 < cut < batch_size * row_bytes and cut % cake_locality.granularity(index):
        return state
    homed = cake_locality.split_empty(tuple(state.shape), state.dtype, cut, index)
    with torch.no_grad():
        homed.copy_(state)
    # A failure here does not fail the call: captured calls then zero their own
    # counters until an eager call refills the reserve.
    with contextlib.suppress(Exception):
        cake_locality.reserve_launch_counters(DOMAIN_COUNTER_WORDS, index)
    return homed
