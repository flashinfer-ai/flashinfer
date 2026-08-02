# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Atomic wrapper plan state and CUDA-stream lifetime management."""

from collections.abc import Callable
from dataclasses import dataclass
import functools
import _thread
from typing import Concatenate, ParamSpec, Protocol, TypeVar, cast

import torch

from .prepared import _PreparedBlockSparseLayout

_P = ParamSpec("_P")
_R = TypeVar("_R")


class _PlanLockOwner(Protocol):
    """Structural type required by the plan-serialization decorator."""

    _plan_lock: _thread.LockType


_S = TypeVar("_S", bound=_PlanLockOwner)


def _serialize_plan(
    method: Callable[Concatenate[_S, _P], _R],
) -> Callable[Concatenate[_S, _P], _R]:
    """Serialize plan calls without making run wait for a replan."""

    @functools.wraps(method)
    def serialized(self: _S, /, *args: _P.args, **kwargs: _P.kwargs) -> _R:
        with self._plan_lock:
            return method(self, *args, **kwargs)

    return cast(Callable[Concatenate[_S, _P], _R], serialized)


@dataclass(frozen=True)
class _BlockSparsePlanState:
    """One complete launch state published by a block-sparse wrapper.

    Every state executes one ``prepare -> prepared-route attention`` adapter,
    including a pattern whose rows select every KV block. The state retains
    caller BSR tensors because the prepare launch consumes their live values
    on every run.

    Runtime geometry, dtypes, metadata tensors, the compiled launch, and the
    readiness event are published together. ``run()`` therefore sees either
    the complete old state or the complete new one, never a mix.

    BSR tensors and any effective caller mask are borrowed. They must remain
    immutable until their queued work completes unless the public plan opted
    into dynamic metadata. That mode permits in-place updates to indptr,
    block-index, and token-mask values between ordered launches while every
    BSR row remains within its planned capacity, strictly increasing, unique,
    and in range. Plan-owned row offsets describe capacity slices rather than
    live BSR boundaries. Mutable route scratch, the dummy mask, and the event
    are also state-owned.
    Policy and the route layout are immutable after publication; the
    cached compiled adapter is shared read-only.

    One revision owns one mutable route workspace. Ordered runs on one
    stream, or externally synchronized cross-stream runs, are valid. Unordered
    concurrent runs of the same revision are unsupported because their prepare
    launches would race. Different wrappers and published revisions own
    independent workspaces.

    ``record_stream()`` extends allocator lifetime only—it cannot prevent the
    caller from modifying borrowed metadata in place.
    """

    device: torch.device
    batch_size: int
    seq_len_q: int
    seq_len_kv: int
    num_heads: int
    head_dim: int
    q_dtype: torch.dtype
    kv_dtype: torch.dtype
    output_dtype: torch.dtype

    # Caller-owned canonical BSR read directly by every run.
    block_indptr: torch.Tensor
    block_indices: torch.Tensor

    # ``kv_valid_bits`` is either the caller mask or a plan-owned
    # shape-correct dummy required by the compiled ABI.
    kv_valid_bits: torch.Tensor

    # Immutable row capacities and mutable per-run route payload.
    route_layout: _PreparedBlockSparseLayout
    row_route_offsets: torch.Tensor
    route_workspace: torch.Tensor
    # Optional semantic row bound; unlike route capacity, this distinguishes
    # two B64 blocks packed into one prepared KV128 record.
    max_blocks_per_row: int | None

    policy: tuple[tuple[str, object], ...]
    compiled: Callable[..., object]

    # All plan-stream work happens-before run after waiting on this event.
    ready_event: torch.cuda.Event
    ready_stream_handle: int


def _allocate_dummy_kv_valid_bits(
    *,
    batch_size: int,
    seq_len_kv: int,
    device: torch.device,
) -> torch.Tensor:
    """Allocate the shape-correct placeholder required by the prepare ABI."""

    return torch.zeros(
        (batch_size, (seq_len_kv + 31) // 32),
        dtype=torch.uint32,
        device=device,
    )


def _record_block_sparse_plan_ready_event(
    stream: torch.cuda.Stream,
) -> torch.cuda.Event:
    """Record the event that closes every plan-owned GPU operation."""

    # External events remain legal wait dependencies inside CUDA Graph capture.
    event = torch.cuda.Event(external=True)
    event.record(stream)
    return event


def _wait_and_record_block_sparse_plan(
    state: _BlockSparsePlanState,
    stream: torch.cuda.Stream,
) -> None:
    """Acquire one state on ``stream`` and retain all launch storage."""

    if stream.device != state.device:
        raise ValueError("run stream must share the planned CUDA device")
    if stream.cuda_stream != state.ready_stream_handle:
        stream.wait_event(state.ready_event)

    # Raw BSR and route storage are true run-time launch tensors. Extending
    # all of them is required for eager cross-stream use and captured
    # revisions.
    state.block_indptr.record_stream(stream)
    state.block_indices.record_stream(stream)
    state.kv_valid_bits.record_stream(stream)
    state.row_route_offsets.record_stream(stream)
    state.route_workspace.record_stream(stream)


__all__ = [
    "_BlockSparsePlanState",
    "_allocate_dummy_kv_valid_bits",
    "_record_block_sparse_plan_ready_event",
    "_serialize_plan",
    "_wait_and_record_block_sparse_plan",
]
