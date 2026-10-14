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

# Prepared FP16 decode for exact SM110a (Thor), CUDA 13 or newer.

from __future__ import annotations

import functools
import math
from typing import Any

from .jit import (
    PREPARED_ROUTES,
    ROUTES,
    SHORT_CAPACITY_MAX,
    _check_exact_sm110a,
    load_sm110_gqa_decode_module,
)

_NUM_Q_HEADS = 32
_NUM_KV_HEADS = 8
_HEAD_DIM = 128
_HEADS_PER_GROUP = _NUM_Q_HEADS // _NUM_KV_HEADS
_SOFTMAX_SCALE_LOG2 = 1.0 / math.sqrt(_HEAD_DIM) / math.log(2.0)
_ACCEPTED_SPLITS = (1, 2, 4, 8, 10, 16)


@functools.cache
def _launcher(route: str) -> Any:
    record = ROUTES[route]
    module = load_sm110_gqa_decode_module(module=record["module"])
    return getattr(module, record["ffi_entry"])


def _select_route(batch: int, capacity: int, num_splits: int | None) -> str:
    """Resolve the launch route from host-known shape facts only."""

    if capacity <= SHORT_CAPACITY_MAX:
        return "short"
    route = PREPARED_ROUTES.get(f"{batch}:{capacity}", "long")
    if num_splits is None:
        return route
    if num_splits == 1:
        # The explicit one-split override always selects the original long
        # kernel, including on the B4/256 direct route.
        return "long"
    if ROUTES[route]["num_splits"] != num_splits:
        if ROUTES[route]["num_splits"] > 1:
            raise ValueError("the exported fused route has a fixed split count")
        raise ValueError("shape does not select an exported split tile")
    return route


def prepare_for_launch(
    inputs: dict[str, Any], num_splits: int | None = None
) -> dict[str, Any]:
    """Prepare a fixed shape without copying sequence lengths to the host.

    The caller supplies contiguous CUDA tensors ``Q`` [B,32,128], ``KV``
    [B,2,8,capacity,128], ``O`` [B,32,128] (all FP16), and int32
    ``sequence_lengths`` [B]. ``O`` must not share storage with any input.
    Optional ``q_scale`` is finite and positive, and defaults to 1.

    Every sequence length must be in [1, capacity] at every launch. Their
    GPU values may change between launches; this precondition is the caller's
    responsibility and is not checked by a host synchronization here.

    Default routes specialize B=4/capacity=256 and B=1/capacity=1024 or 4096.
    Other capacities above 64 use the original long route. ``num_splits=1``
    explicitly selects original long above capacity 64, including B=4/256;
    the two specialized long routes also accept their fixed split count 10.
    Capacities up to 64 always use original short. Unsupported split choices
    are rejected rather than generating a new kernel.

    Prepare and warm up outside Graph capture. Keep the returned object alive
    until all launches complete and any captured Graphs are retired. Its
    bindings, route, and workspace are opaque and must not be modified.
    """
    import torch

    q, kv, output, lengths = (inputs[k] for k in ("Q", "KV", "O", "sequence_lengths"))
    if q.ndim != 3 or tuple(q.shape[1:]) != (_NUM_Q_HEADS, _HEAD_DIM):
        raise ValueError("Q must have shape [B,32,128]")
    batch = int(q.shape[0])
    if batch < 1 or tuple(output.shape) != tuple(q.shape):
        raise ValueError("O must match nonempty Q")
    if (
        kv.ndim != 5
        or tuple(kv.shape[:3]) != (batch, 2, _NUM_KV_HEADS)
        or kv.shape[-1] != _HEAD_DIM
    ):
        raise ValueError("KV must have shape [B,2,8,capacity,128]")
    capacity = int(kv.shape[-2])
    if capacity < 1:
        raise ValueError("KV capacity must be positive")
    if any(t.dtype != torch.float16 for t in (q, kv, output)):
        raise TypeError("Q, KV and O must be FP16")
    if lengths.dtype != torch.int32 or tuple(lengths.shape) != (batch,):
        raise TypeError("sequence_lengths must be int32[B]")
    tensors = (q, kv, output, lengths)
    if q.device.type != "cuda" or any(t.device != q.device for t in tensors):
        raise ValueError("all tensors must share one CUDA device")
    if not all(t.is_contiguous() for t in tensors):
        raise ValueError("prepared tensors must be contiguous")
    if output.untyped_storage().data_ptr() in tuple(
        t.untyped_storage().data_ptr() for t in (q, kv, lengths)
    ):
        raise ValueError("O must not alias any input storage")
    _check_exact_sm110a(q.device)
    q_scale = float(inputs.get("q_scale", 1.0))
    if not math.isfinite(q_scale) or q_scale <= 0:
        raise ValueError("q_scale must be finite and positive")
    if num_splits is not None and (
        type(num_splits) is not int or num_splits not in _ACCEPTED_SPLITS
    ):
        raise ValueError("num_splits must be None or an integer in {1,2,4,8,10,16}")
    route = _select_route(batch, capacity, num_splits)
    split_count = int(ROUTES[route]["num_splits"])

    bindings = {
        "Q": q.view(batch, _NUM_KV_HEADS, _HEADS_PER_GROUP, _HEAD_DIM).transpose(1, 2),
        "K": kv[:, 0],
        "V": kv[:, 1],
        "O": output,
        "sequence_lengths": lengths,
        "kv_capacity": capacity,
        "softmax_scale_log2": q_scale * _SOFTMAX_SCALE_LOG2,
    }
    workspace: tuple[torch.Tensor, ...] = ()
    arguments: tuple[Any, ...]
    if split_count == 1:
        arguments = (
            bindings["Q"],
            bindings["K"],
            bindings["V"],
            output,
            lengths,
            bindings["softmax_scale_log2"],
            batch * _NUM_KV_HEADS,
            1,
            1,
        )
    else:
        partial_o = torch.empty(
            (batch, _NUM_Q_HEADS, split_count, _HEAD_DIM),
            dtype=torch.float32,
            device=q.device,
        )
        partial_max = torch.empty(
            (batch, _NUM_Q_HEADS, split_count), dtype=torch.float32, device=q.device
        )
        partial_sum = torch.empty_like(partial_max)
        # The last CTA resets its completion counter after merging. Ordered
        # launches and Graph replays need no additional reset kernel.
        completed = torch.zeros(
            (batch * _NUM_KV_HEADS,), dtype=torch.uint32, device=q.device
        )
        workspace = (partial_o, partial_max, partial_sum, completed)
        bindings.update(
            partial_O=partial_o,
            partial_max=partial_max,
            partial_sum=partial_sum,
            completed=completed,
        )
        arguments = (
            bindings["Q"],
            bindings["K"],
            bindings["V"],
            partial_o,
            partial_max,
            partial_sum,
            completed,
            output,
            lengths,
            bindings["softmax_scale_log2"],
            batch * _NUM_KV_HEADS * split_count,
            1,
            1,
        )
    stages = ((_launcher(route), arguments, ROUTES[route]["kernel_symbol"]),)
    return {
        "route": route,
        "num_splits": split_count,
        "O": output,
        "bindings": bindings,
        "workspace": workspace,
        "stages": stages,
        "launch_names": [stage[2] for stage in stages],
        "workspace_bytes": sum(t.numel() * t.element_size() for t in workspace),
    }


def launch_prepared(prepared: dict[str, Any]) -> Any:
    """Launch on the current PyTorch CUDA stream and return the caller's O.

    Retain ``prepared`` until asynchronous work finishes, or throughout the
    lifetime of any CUDA Graph capturing this launch. Concurrent invocations
    require distinct prepared objects and output/workspace storage. Reuse on
    ordered streams, including event-ordered handoffs, is supported.
    """
    import tvm_ffi

    with tvm_ffi.use_torch_stream():
        for call, arguments, _name in prepared["stages"]:
            call(*arguments)
    return prepared["O"]
