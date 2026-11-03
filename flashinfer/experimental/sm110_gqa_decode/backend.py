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

# Exact-SM110 FP16 GQA decode backend: the host side of the Cake package.
#
# The registry (``cake_jit.MODULES``) carries one record per exported program
# and the host plan as data: ``select_record`` picks the record from the batch,
# the cache capacity and the caller's split request (the size-based rule), and
# ``launch_grid`` shapes the launch.  This module validates the caller's
# tensors, allocates the workspace a split record names, binds the positional
# argument list of the record's ``run`` entry once (``argument_list``) and
# launches it on the current torch stream.

from __future__ import annotations

import functools
import math
from collections.abc import Sequence
from typing import Any, Callable

import torch
import tvm_ffi

from . import cake_jit

NUM_Q_HEADS = cake_jit.NUM_Q_HEADS
NUM_KV_HEADS = cake_jit.NUM_KV_HEADS
HEAD_DIM = cake_jit.HEAD_DIM
#: The public ``num_splits`` values (``cake_jit.ACCEPTED_SPLITS``).
ACCEPTED_SPLITS = cake_jit.ACCEPTED_SPLITS
#: The smallest cache capacity the public API accepts: the 64-token TMA box of
#: the short route.  The previous release rejected a smaller cache in its
#: binding; this release keeps the rejection, and its message, on the host.
MIN_CAPACITY = cake_jit.SHORT_CAPACITY_MAX
_HEADS_PER_GROUP = NUM_Q_HEADS // NUM_KV_HEADS
_SOFTMAX_SCALE_LOG2 = 1.0 / math.sqrt(HEAD_DIM) / math.log(2.0)


def _cuda_version() -> tuple[int, int]:
    text = torch.version.cuda or "0.0"
    parts = text.split(".")
    return (
        int(parts[0]),
        int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0,
    )


@functools.cache
def require_sm110(device: Any = None) -> None:
    """Refuse a device other than the one exported target: compute capability
    11.0 (``cake_jit.ARCH_BY_CAPABILITY``) with CUDA 13.0 or newer."""
    resolved = torch.device("cuda") if device is None else torch.device(device)
    if resolved.type != "cuda":
        raise ValueError("device must be a cuda device")
    capability = tuple(torch.cuda.get_device_capability(resolved))
    if cake_jit.arch_for_capability(capability) is None or _cuda_version() < (13, 0):
        raise RuntimeError(
            "SM110 GQA decode requires compute capability 11.0 and CUDA 13.0 or newer; "
            f"got compute capability {capability[0]}.{capability[1]} with CUDA {torch.version.cuda}"
        )


def _device_index(device: torch.device) -> int:
    return device.index if device.index is not None else torch.cuda.current_device()


def _require_capacity(capacity: int) -> None:
    """Refuse a cache smaller than the short route's 64-token TMA box (the
    previous release's check and message, kept as the public contract)."""
    if capacity < MIN_CAPACITY:
        raise ValueError(
            f"TMA box (64, {MIN_CAPACITY}, 1, 1) exceeds resolved global dims for 'K'"
        )


def _entry(record: str, device_index: int) -> Callable:
    """The ``run`` entry of a registry record for the device (built or loaded once per process)."""
    return cake_jit.entry(record, device_index)


def _require_tensor(
    tensor: torch.Tensor,
    *,
    name: str,
    shape: Sequence[int],
    dtype: torch.dtype,
    device: torch.device | None = None,
) -> None:
    if tuple(tensor.shape) != tuple(shape):
        raise ValueError(
            f"{name} must have shape {tuple(shape)}, got {tuple(tensor.shape)}"
        )
    if tensor.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}, got {tensor.dtype}")
    if not tensor.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    if device is not None and tensor.device != device:
        raise ValueError(f"{name} must be on {device}, got {tensor.device}")


def route(
    batch: int, capacity: int, num_splits: int | None = None
) -> tuple[str, int, tuple[int, int, int]]:
    """The record, the run-time split count and the launch grid of a decode problem.

    ``num_splits`` ``None`` or ``1`` leaves the record and the split count to the
    size rule (``cake_jit.select_route``: the table row of the shape names both;
    today one split on every row -- the one-CTA ``short`` / ``long`` routes for
    one request over at most 128 keys, else the one-split build of the split-KV
    kernel the row names); an accepted count above one launches the
    run-time-split program of the shape's tile with that many splits; a value
    outside ``ACCEPTED_SPLITS`` raises ``ValueError``.
    """
    record, splits = cake_jit.select_route(
        batch=batch, capacity=capacity, num_splits=num_splits
    )
    grid = cake_jit.launch_grid(record, batch=batch, num_splits=splits)
    return record, splits, grid


def workspace_for(
    record: str, batch: int, splits: int, device: torch.device
) -> dict[str, torch.Tensor]:
    """The buffers the record's ``workspace`` leaf names: FP32 partials
    ``partial_O [B, 32, S, 128]``, ``partial_max`` / ``partial_sum [B, 32, S]`` and
    the ``completed`` counters ``uint32 [B * 8]``.  The run-time-split programs get
    the counters zeroed here (the last CTA of each (request, KV head) resets them
    after its merge, so a captured graph replays without a reset node); the
    one-split direct builds keep the counter pointer in their ABI but never read
    or write it, so they take an uninitialised buffer and skip the fill kernel.
    Empty for the one-CTA routes."""
    names = cake_jit.MODULES[record].get("workspace", [])
    buffers: dict[str, torch.Tensor] = {}
    for name in names:
        if name == "partial_O":
            buffers[name] = torch.empty(
                (batch, NUM_Q_HEADS, splits, HEAD_DIM),
                dtype=torch.float32,
                device=device,
            )
        elif name in ("partial_max", "partial_sum"):
            buffers[name] = torch.empty(
                (batch, NUM_Q_HEADS, splits), dtype=torch.float32, device=device
            )
        elif name == "completed":
            rule = cake_jit.MODULES[record]["grid_rule"]
            fill = torch.zeros if rule.get("num_splits") == "runtime" else torch.empty
            buffers[name] = fill(
                (batch * NUM_KV_HEADS,), dtype=torch.uint32, device=device
            )
        else:
            raise RuntimeError(
                f"record {record} names an unknown workspace buffer {name!r}"
            )
    return buffers


def _bindings(
    q: torch.Tensor,
    kv: torch.Tensor,
    out: torch.Tensor,
    lengths: torch.Tensor,
    q_scale: float,
) -> dict[str, Any]:
    """The logical argument values of every record: the query viewed
    ``[B, 4, 8, 128]`` (the four query heads of a KV head side by side, the TMA
    descriptor source), the K and V selectors of the cache, the output, the
    lengths and the log2-domain softmax scale."""
    batch = int(q.shape[0])
    return {
        "Q": q.view(batch, NUM_KV_HEADS, _HEADS_PER_GROUP, HEAD_DIM).transpose(1, 2),
        "K": kv[:, 0],
        "V": kv[:, 1],
        "O": out,
        "sequence_lengths": lengths,
        "softmax_scale_log2": float(q_scale) * _SOFTMAX_SCALE_LOG2,
    }


def sm110_gqa_decode(
    q: torch.Tensor,
    kv: torch.Tensor,
    sequence_lengths: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
    q_scale: float = 1.0,
) -> torch.Tensor:
    """Run the fixed Hq32/Hkv8/D128 FP16 decode specialization once.

    ``sequence_lengths`` is consumed on the device: every value must be in
    ``[1, capacity]`` when the kernel runs. This precondition is the caller's
    responsibility, as for the prepared API; the host never reads the lengths
    back, so the call enqueues without a device synchronization.  The capacity
    is at least ``MIN_CAPACITY`` (64, the short route's TMA box; a smaller
    cache raises ``ValueError`` as before).  The route is the size rule's
    (``num_splits`` unset); a split record's workspace is allocated for the
    call.
    """

    if q.ndim != 3:
        raise ValueError("q must have shape [batch, 32, 128]")
    batch = int(q.shape[0])
    if batch <= 0:
        raise ValueError("batch must be positive")
    _require_tensor(
        q,
        name="q",
        shape=(batch, NUM_Q_HEADS, HEAD_DIM),
        dtype=torch.float16,
    )
    if kv.ndim != 5:
        raise ValueError("kv must have shape [batch, 2, 8, capacity, 128]")
    capacity = int(kv.shape[-2])
    if capacity <= 0:
        raise ValueError("kv capacity must be positive")
    _require_tensor(
        kv,
        name="kv",
        shape=(batch, 2, NUM_KV_HEADS, capacity, HEAD_DIM),
        dtype=torch.float16,
        device=q.device,
    )
    _require_tensor(
        sequence_lengths,
        name="sequence_lengths",
        shape=(batch,),
        dtype=torch.int32,
        device=q.device,
    )

    if out is None:
        out = torch.empty_like(q)
    else:
        _require_tensor(
            out,
            name="out",
            shape=q.shape,
            dtype=torch.float16,
            device=q.device,
        )
        if out.data_ptr() == q.data_ptr():
            raise ValueError("out must not alias q")

    require_sm110(q.device)
    _require_capacity(capacity)
    record, splits, grid = route(batch, capacity, None)
    values = _bindings(q, kv, out, sequence_lengths, q_scale)
    values.update(workspace_for(record, batch, splits, q.device))
    args = cake_jit.argument_list(record, values, grid=grid)
    launch = _entry(record, _device_index(q.device))
    with torch.cuda.device(q.device), tvm_ffi.use_torch_stream():
        launch(*args)
    return out


def prepare_for_launch(
    inputs: dict[str, Any], num_splits: int | None = None
) -> dict[str, Any]:
    """Prepare a fixed shape without copying sequence lengths to the host.

    The caller supplies contiguous CUDA tensors ``Q`` [B,32,128], ``KV``
    [B,2,8,capacity,128] with ``capacity >= 64`` (``MIN_CAPACITY``; a smaller
    cache raises ``ValueError`` as before), ``O`` [B,32,128] (all FP16), and
    int32 ``sequence_lengths`` [B]. ``O`` must not share storage with any
    input. Optional ``q_scale`` is finite and positive, and defaults to 1.

    Every sequence length must be in [1, capacity] at every launch. Their
    GPU values may change between launches; this precondition is the caller's
    responsibility and is not checked by a host synchronization here.

    ``num_splits=None`` (or ``1``) selects the size-based route and its split
    count (the route table's row; one split on every row today): one request
    over at most 64 keys takes the ``short`` program, over 65..128 keys ``long``;
    one request over 129..256 keys or two requests over at most 512 keys the
    one-split N64 build of the split-KV program, longer caches of one or two
    requests its four-slot N64 build, and three or more requests its one-split
    N32 build. An explicit count in ``{2, 4, 8, 10, 16}`` launches the
    run-time-split program of the shape's tile (N64 for one or two requests over
    at most 4095 keys, N32 otherwise) with that many splits; any other value is
    rejected.

    Prepare and warm up outside Graph capture. Keep the returned object alive
    until all launches complete and any captured Graphs are retired. Its
    bindings, route, and workspace are opaque and must not be modified.
    """
    q, kv, output, lengths = (inputs[k] for k in ("Q", "KV", "O", "sequence_lengths"))
    if q.ndim != 3 or tuple(q.shape[1:]) != (NUM_Q_HEADS, HEAD_DIM):
        raise ValueError("Q must have shape [B,32,128]")
    batch = int(q.shape[0])
    if batch < 1 or tuple(output.shape) != tuple(q.shape):
        raise ValueError("O must match nonempty Q")
    if (
        kv.ndim != 5
        or tuple(kv.shape[:3]) != (batch, 2, NUM_KV_HEADS)
        or kv.shape[-1] != HEAD_DIM
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
    require_sm110(q.device)
    q_scale = float(inputs.get("q_scale", 1.0))
    if not math.isfinite(q_scale) or q_scale <= 0:
        raise ValueError("q_scale must be finite and positive")
    if num_splits is not None and (
        type(num_splits) is not int or num_splits not in ACCEPTED_SPLITS
    ):
        raise ValueError("num_splits must be None or an integer in {1,2,4,8,10,16}")
    _require_capacity(capacity)
    record, split_count, grid = route(batch, capacity, num_splits)

    bindings = _bindings(q, kv, output, lengths, q_scale)
    buffers = workspace_for(record, batch, split_count, q.device)
    bindings.update(buffers)
    workspace = tuple(buffers.values())
    arguments = tuple(cake_jit.argument_list(record, bindings, grid=grid))
    stages = ((_entry(record, _device_index(q.device)), arguments),)
    return {
        "route": record,
        "num_splits": split_count,
        "grid": grid,
        "O": output,
        "bindings": bindings,
        "workspace": workspace,
        "stages": stages,
        "workspace_bytes": sum(t.numel() * t.element_size() for t in workspace),
    }


def launch_prepared(prepared: dict[str, Any]) -> Any:
    """Launch on the current PyTorch CUDA stream and return the caller's O.

    Retain ``prepared`` until asynchronous work finishes, or throughout the
    lifetime of any CUDA Graph capturing this launch. Concurrent invocations
    require distinct prepared objects and output/workspace storage. Reuse on
    ordered streams, including event-ordered handoffs, is supported.
    """
    with tvm_ffi.use_torch_stream():
        for call, arguments in prepared["stages"]:
            call(*arguments)
    return prepared["O"]


__all__ = [
    "ACCEPTED_SPLITS",
    "HEAD_DIM",
    "MIN_CAPACITY",
    "NUM_KV_HEADS",
    "NUM_Q_HEADS",
    "launch_prepared",
    "prepare_for_launch",
    "require_sm110",
    "route",
    "sm110_gqa_decode",
    "workspace_for",
]
