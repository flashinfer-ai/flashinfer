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

    Every state executes the raw-BSR kernel, including a pattern whose
    rows select every KV block. The state retains caller BSR tensors because
    run consumes them directly; no route-remapping payload exists in the
    public lifecycle.

    Runtime geometry, dtypes, metadata tensors, the compiled launch, and the
    readiness event are published together. ``run()`` therefore sees either
    the complete old state or the complete new one, never a mix.

    BSR tensors and any effective caller mask are borrowed. They must remain
    immutable until their queued work completes unless the public plan opted
    into dynamic metadata. That mode permits in-place updates to block indices
    and token-mask values between ordered launches while row offsets stay
    fixed and every BSR row remains strictly increasing. The dummy mask and
    event are state-owned. Policy is immutable after publication; the cached
    compiled adapter is shared read-only.
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

    # ``runtime_kv_valid_bits`` is either the caller mask or a plan-owned
    # shape-correct dummy required by the compiled ABI.
    runtime_kv_valid_bits: torch.Tensor
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
    """Allocate the shape-correct placeholder required by the raw kernel ABI."""

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

    # Raw BSR is a true run-time input. Extending all three launch tensors is
    # required for eager cross-stream use and captured revisions.
    state.block_indptr.record_stream(stream)
    state.block_indices.record_stream(stream)
    state.runtime_kv_valid_bits.record_stream(stream)


__all__ = [
    "_BlockSparsePlanState",
    "_allocate_dummy_kv_valid_bits",
    "_record_block_sparse_plan_ready_event",
    "_serialize_plan",
    "_wait_and_record_block_sparse_plan",
]
